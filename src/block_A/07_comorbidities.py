#!/usr/bin/env python3
"""Section 5: baseline comorbidity burden and longitudinal progression.

This is deliberately a single, auditable analysis entry point.  Canonical
patient/visit timing, Pop classification, and ESSDAI-domain flags are consumed
from upstream artifacts; only the prespecified baseline comorbidity flags are
derived here.  Associations produced by this script are not causal effects.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402
import config  # noqa: E402
from src.derivations.visit_dates import add_parsed_visit_dates  # noqa: E402

PATIENT_ID_COL = "ids__patient_record_number"
VISIT_DATE_COL = "ids__visit_date"
AGE_COL = "ids__age_at_visit"
SEX_COL = "ids__sex"
# The deployed analytic extract contains only this total-score field.  Keep one
# explicit source of truth rather than requiring an unavailable alias.  The
# ``*_raw_qc`` output is retained for schema compatibility and is
# therefore an identical audit copy of the primary value in this extract.
ESSDAI_PRIMARY_COL = config.ESSDAI_TOTAL_RAW
ESSDAI_RAW_QC_COL = config.ESSDAI_TOTAL_RAW
# Section 5 progression uses the same moderate-to-severe threshold as Pop 1.
SEVERE_THRESHOLD = config.ESSDAI_SEVERE
RANDOM_SEED = 20260728
SCRIPT_VERSION = "1.2.0"

FIGURES_DIR = common.OUTPUTS_DIR / "figures" / "blockA"
TABLES_DIR = common.OUTPUTS_DIR / "tables" / "blockA"
QC_DIR = common.OUTPUTS_DIR / "qc" / "blockA"
LOG_PATH = common.OUTPUTS_DIR / "logs" / "07_comorbidities.log"

BASELINE_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidities_baseline_patient.parquet"
LONGITUDINAL_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidities_analysis_longitudinal.parquet"
SEVERE_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidity_severe5_survival.parquet"
NEW_DOMAIN_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidity_new_domain_survival.parquet"
DOMAIN_AUDIT_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidity_new_domain_patient_domain.parquet"

INTERMEDIATE_PATHS = [BASELINE_PATH, LONGITUDINAL_PATH, SEVERE_PATH, NEW_DOMAIN_PATH, DOMAIN_AUDIT_PATH]

ORGAN_DOMAINS = [
    "eg_constitutional_active", "eg_lymphadenopathy_active", "eg_articular_active",
    "eg_cutaneous_active", "eg_pulmonary_active", "eg_renal_active",
    "eg_muscular_active", "eg_pns_active", "eg_cns_active",
    "eg_hematologic_active",
]
ALL_DOMAINS = ORGAN_DOMAINS + ["eg_biological_active"]
DOMAIN_EVALUABLE = {domain: domain.removesuffix("_active") + "_evaluable" for domain in ALL_DOMAINS}


@dataclass(frozen=True)
class Condition:
    name: str
    label: str
    primary: tuple[str, ...]
    condition_family: str
    clinical_category: str
    detail_columns: tuple[str, ...] = ()
    notes: str = ""


def _pmh(name: str, label: str, source: str, category: str, *details: str) -> Condition:
    return Condition(name, label, (source,), "past_medical_history", category, tuple(details),
                     "Documented history at baseline; blank means no history documented in this source.")


def _rheum(name: str, label: str, general: str, history: str, confirmed: str,
           family: str, category: str = "Rheumatologic") -> Condition:
    # Source order is a public invariant: general, history, confirmed.
    return Condition(name, label, (general, history, confirmed), family, category)


PAST_MEDICAL_HISTORY_CONDITIONS = [
    _pmh("hypertension", "Hypertension", "past_medical_history__hypertension_", "Cardiovascular/metabolic"),
    _pmh("hyperlipidemia", "Hyperlipidemia", "past_medical_history__hyperlipidemia_", "Cardiovascular/metabolic"),
    _pmh("cerebrovascular_disease", "Cerebrovascular disease", "past_medical_history__cva", "Cardiovascular/metabolic"),
    _pmh("myocardial_infarction", "Myocardial infarction", "past_medical_history__cardio_hx_mycrdl", "Cardiovascular/metabolic"),
    _pmh("coronary_artery_disease", "Coronary artery disease", "past_medical_history__cardio_hx_cad", "Cardiovascular/metabolic"),
    _pmh("peripheral_vascular_disease", "Peripheral vascular disease", "past_medical_history__cardio_hx_pvd", "Cardiovascular/metabolic"),
    _pmh("valvular_disease", "Valvular disease", "past_medical_history__cardio_valve_disease", "Cardiovascular/metabolic"),
    _pmh("pulmonary_embolism", "Pulmonary embolism", "past_medical_history__pulmonary_embolism", "Cardiovascular/metabolic"),
    _pmh("thyroid_disease", "Thyroid disease", "past_medical_history__thyroid_disease", "Endocrine", "past_medical_history__thyroid_disease_spfy"),
    _pmh("diabetes_type_1", "Diabetes type 1", "past_medical_history__endcrn_hx_mellitus_i", "Endocrine"),
    _pmh("diabetes_type_2", "Diabetes type 2", "past_medical_history__endcrn_hx_mellitus_ii", "Endocrine"),
    _pmh("asthma", "Asthma", "past_medical_history__asthma", "Respiratory"),
    _pmh("copd", "COPD", "past_medical_history__respiratory_hx_copd", "Respiratory"),
    _pmh("chronic_bronchitis", "Chronic bronchitis", "past_medical_history__respiratory_hx_bronchitis", "Respiratory"),
    _pmh("recurrent_sinusitis", "Recurrent sinusitis", "past_medical_history__sinusitis", "Respiratory"),
    _pmh("gerd", "GERD", "past_medical_history__gi_hx_gerd", "Gastrointestinal"),
    _pmh("irritable_bowel_syndrome", "Irritable bowel syndrome", "past_medical_history__gi_hx_ibs", "Gastrointestinal"),
    _pmh("celiac_disease", "Celiac disease", "past_medical_history__gi_hx_celiar", "Gastrointestinal"),
    _pmh("autoimmune_hepatitis", "Autoimmune hepatitis", "past_medical_history__gi_hx_auto_hepat", "Gastrointestinal"),
    _pmh("primary_sclerosing_cholangitis", "Primary sclerosing cholangitis", "past_medical_history__gi_hx_sclerosing", "Gastrointestinal"),
    _pmh("pancreatitis", "Pancreatitis", "past_medical_history__pancreatitis", "Gastrointestinal"),
    _pmh("depression", "Depression", "past_medical_history__neuro_hx_depression", "Neuropsychiatric"),
    _pmh("multiple_sclerosis", "Multiple sclerosis", "past_medical_history__neuro_hx_mult_sclerosis", "Neuropsychiatric"),
    _pmh("seizure_disorder", "Seizure disorder", "past_medical_history__neuro_seizrs", "Neuropsychiatric"),
    _pmh("kidney_stones", "Kidney stones", "past_medical_history__renal_hx_kidney_stones", "Genitourinary"),
    _pmh("recurrent_urinary_tract_infections", "Recurrent urinary tract infections", "past_medical_history__renal_hx_recurr_uti", "Genitourinary"),
    _pmh("interstitial_cystitis", "Interstitial cystitis", "past_medical_history__interstitial_cyst", "Genitourinary"),
    _pmh("psoriasis", "Psoriasis", "past_medical_history__cutaneous_hx_psoriasis", "Dermatologic"),
    _pmh("vitiligo", "Vitiligo", "past_medical_history__cutaneous_hx_vitiligo", "Dermatologic"),
    *[_pmh(n, l, c, "Malignancy") for n,l,c in [
      ("breast_cancer","Breast cancer","past_medical_history__malignancy_hx_breast_ca"),("lung_cancer","Lung cancer","past_medical_history__malignancy_hx_lung_ca"),("colon_cancer","Colon cancer","past_medical_history__malignancy_hx_colon_ca"),("thyroid_cancer","Thyroid cancer","past_medical_history__malignancy_hx_thyroid_ca"),("head_neck_cancer","Head/neck cancer","past_medical_history__malignancy_hx_head_ca"),("other_malignancy","Other malignancy","past_medical_history__malignancy_hx_other")]],
    *[_pmh(n, l, c, "Chronic infection") for n,l,c in [
      ("hepatitis_b","Hepatitis B","past_medical_history__gi_hx_hepatitis_b"),("hepatitis_c","Hepatitis C","past_medical_history__gi_hx_hepatitis_c"),("hiv_aids","HIV/AIDS","past_medical_history__ec_aids"),("htlv_infection","HTLV infection","past_medical_history__htlv_infection")]],
]

RHEUMATOLOGIC_NON_SAID_CONDITIONS = [
    _rheum("fibromyalgia","Fibromyalgia","rheumatological_comorbidities__fibromyalgia1","rheumatological_comorbidities__fibromyalgia1_hx","rheumatological_comorbidities__fibromyalgia1_confirm","rheumatologic_non_said"),
    _rheum("osteoporosis","Osteoporosis","rheumatological_comorbidities__osteoporosis1","rheumatological_comorbidities__osteoporosis1_hx","rheumatological_comorbidities__osteoporosis1_confirm","rheumatologic_non_said"),
    _rheum("osteopenia","Osteopenia","rheumatological_comorbidities__osteopenia","rheumatological_comorbidities__osteopenia_hx","rheumatological_comorbidities__osteopenia_confirm","rheumatologic_non_said"),
    _rheum("osteoarthritis","Osteoarthritis","rheumatological_comorbidities__osteoarthritis","rheumatological_comorbidities__osteoarthritis_hx","rheumatological_comorbidities__osteoarthritis_confirm","rheumatologic_non_said"),
    _rheum("crystalline_arthropathy","Crystalline arthropathy","rheumatological_comorbidities__crystalline_arthropathy","rheumatological_comorbidities__crystalline_arthropathy_hx","rheumatological_comorbidities__crystalline_arthro_confirm","rheumatologic_non_said"),
]

_said = [
("systemic_lupus_erythematosus","Systemic lupus erythematosus","sle1","sle_hx","sle_confirmed"),("rheumatoid_arthritis","Rheumatoid arthritis","ra","ra_hx","ra_confirm"),("systemic_sclerosis","Systemic sclerosis","systemic_sclerosis","systmc_sclerosis_hx","systmc_sclerosis_confirm"),("polymyositis","Polymyositis","polymyositis","polymyositis_hx","polymyositis_confirm"),("dermatomyositis","Dermatomyositis","dermatomyositis","dermatomyositis_hx","dermatomyositis_confirm"),("mixed_connective_tissue_disease","Mixed connective tissue disease","mixed_connective_tissue_disease","mixed_connect_tissue_hx","mixed_connect_tissue_confirm"),("antiphospholipid_syndrome","Antiphospholipid syndrome","antiphospholipid_syndrome","antiphospholipid_syn_hx","antiphospholipid_syn_confirm"),("primary_biliary_cholangitis","Primary biliary cholangitis","primary_billiary_cirrhosis","prim_billiary_cirrhosis_hx","prim_billiary_cirrhosis_confirm"),("inflammatory_bowel_disease","Inflammatory bowel disease","inflam_bowel","inflam_bowel_hx","inflam_bowel_confirm"),("sarcoidosis","Sarcoidosis","sarcoidosis","sarcoidosis_hx","sarcoidosis_confirm")]
CONCOMITANT_SAID_CONDITIONS = [_rheum(n,l,*("rheumatological_comorbidities__"+x for x in (g,h,c)),"concomitant_said","Systemic autoimmune/inflammatory") for n,l,g,h,c in _said]
def iter_analysis_conditions() -> Iterable[Condition]:
    yield from PAST_MEDICAL_HISTORY_CONDITIONS
    yield from RHEUMATOLOGIC_NON_SAID_CONDITIONS
    yield from CONCOMITANT_SAID_CONDITIONS


CONDITION_NAMES = [c.name for c in iter_analysis_conditions()]
PROGRESSION_CONDITIONS = RHEUMATOLOGIC_NON_SAID_CONDITIONS
PROGRESSION_CONDITION_NAMES = {c.name for c in PROGRESSION_CONDITIONS}
PROGRESSION_CONDITION_NAMES_ORDERED = [c.name for c in PROGRESSION_CONDITIONS]
SUBTYPE_COLS = ("past_medical_history__thyroid_disease_spfy", "rheumatological_comorbidities__inflam_bowel_spfy")
PROHIBITED_SOURCE_PREFIXES = ("sjogren's_syndrome_history__", "sjogren's_syndrome_disease_damage_index__", "systems_review_for_physician__", "ans__", "autonomic_nervous_system_questionnaire__")
ACTIVITY_THRESHOLD_SECTION5 = SEVERE_THRESHOLD
DOMAIN_COLS = ORGAN_DOMAINS
DOMAIN_EVALUABLE_COLS = DOMAIN_EVALUABLE
UNAVAILABLE: dict[str, str] = {}

UPSTREAM = {
    common.VISIT_SPINE_PARQUET: "src/00_build_visit_spine.py",
    common.POP_LONGITUDINAL_PARQUET: "src/block_A/01_pop_distribution.py",
    common.OVERLAP_LONGITUDINAL_PARQUET: "src/block_A/06_overlap_glandular_followup.py",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=common.DEFAULT_ANALYTIC_DATASET)
    p.add_argument("--rebuild-upstream", action="store_true")
    p.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    p.add_argument("--run-sparse-monte-carlo", action="store_true",
                   help="Opt in to a 10,000-replicate sparse-table global test")
    p.add_argument("--monte-carlo-replicates", type=int, default=10_000)
    p.add_argument("--minimum-events", type=int, default=10)
    return p.parse_args(argv)


def ensure_directories() -> None:
    for path in (FIGURES_DIR, TABLES_DIR, QC_DIR, LOG_PATH.parent, common.INTERMEDIATE_DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("07_comorbidities")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def check_upstream_artifacts(input_path: Path, rebuild: bool, logger: logging.Logger) -> dict[str, str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Raw input not found: {input_path}")
    stale_or_missing = [p for p in UPSTREAM if not p.exists() or p.stat().st_mtime < input_path.stat().st_mtime]
    if stale_or_missing and rebuild:
        for artifact in stale_or_missing:
            script = PROJECT_ROOT / UPSTREAM[artifact]
            logger.info("Rebuilding upstream artifact with %s", script)
            if artifact == common.OVERLAP_LONGITUDINAL_PARQUET:
                # The overlap generator predates the repository CLI convention.
                # Execute its main function after replacing its input global so a
                # custom Section 5 input can never be mixed with the default cohort.
                runner = (
                    "import runpy,sys; from pathlib import Path; "
                    "ns=runpy.run_path(sys.argv[1]); "
                    "ns['INPUT_PARQUET']=Path(sys.argv[2]); ns['main']()"
                )
                command = [sys.executable, "-c", runner, str(script), str(input_path)]
            else:
                command = [sys.executable, str(script), "--input", str(input_path)]
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    failures = [p for p in UPSTREAM if not p.exists()]
    if failures:
        detail = "; ".join(f"{p} (generated by {UPSTREAM[p]})" for p in failures)
        raise FileNotFoundError(f"Required upstream artifact(s) missing: {detail}. Use --rebuild-upstream to generate them.")
    stale = [p for p in UPSTREAM if p.stat().st_mtime < input_path.stat().st_mtime]
    if stale:
        detail = "; ".join(f"{p} (generated by {UPSTREAM[p]})" for p in stale)
        raise RuntimeError(f"Required upstream artifact(s) are older than raw input: {detail}. Use --rebuild-upstream.")
    return {str(p): datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat() for p in UPSTREAM}


def _read_required(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    available = available_columns(path)
    missing = set(columns) - available
    if missing:
        raise KeyError(f"{path} lacks required columns: {sorted(missing)}")
    return pd.read_parquet(path, columns=list(columns))


def available_columns(path: Path) -> set[str]:
    return set(pq.read_schema(path).names)


def parse_age_first_value(series: pd.Series) -> pd.Series:
    """Parse age as numeric, taking the first value from pipe-delimited entries."""
    first = series.astype("string").str.split("|", regex=False).str[0].str.strip()
    return pd.to_numeric(first, errors="coerce")


def load_visit_spine() -> pd.DataFrame:
    # Patient and canonical visit identity are the integration keys.
    cols = ["patient_id", "visit_id", "visit_date", "visit_number", "observed_baseline_date", "time_since_observed_baseline_days", "time_since_observed_baseline_years", "age_at_visit", "sex"]
    df = _read_required(common.VISIT_SPINE_PARQUET, cols)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    df["observed_baseline_date"] = pd.to_datetime(df["observed_baseline_date"])
    df["age_at_visit"] = parse_age_first_value(df["age_at_visit"])
    if df["patient_id"].isna().any() or not df["visit_id"].is_unique or df.duplicated(["patient_id", "visit_date"]).any():
        raise ValueError("Visit spine violates patient/visit identity constraints")
    if (df["time_since_observed_baseline_days"] < 0).any():
        raise ValueError("Visit spine contains negative time since baseline")
    return df


def load_pop_classification() -> pd.DataFrame:
    cols = ["patient_id", "visit_id", "visit_date", "visit_number", "essdai_total", "esspri_total", "pop_status", "baseline_pop_status"]
    return _read_required(common.POP_LONGITUDINAL_PARQUET, cols)


def load_domain_flags() -> pd.DataFrame:
    cols = (["patient_id", "visit_id", "visit_date"] + ALL_DOMAINS
            + list(DOMAIN_EVALUABLE.values()) + ["n_extraglandular_domains_active"])
    df = _read_required(common.OVERLAP_LONGITUDINAL_PARQUET, cols)
    for col in ALL_DOMAINS + list(DOMAIN_EVALUABLE.values()):
        df[col] = normalize_binary_flag(df[col])
    return df


def selected_raw_columns(input_path: Path) -> list[str]:
    schema = available_columns(input_path)
    desired = [PATIENT_ID_COL, VISIT_DATE_COL, AGE_COL, SEX_COL, ESSDAI_PRIMARY_COL, ESSDAI_RAW_QC_COL, *SUBTYPE_COLS]
    desired.extend(col for c in iter_analysis_conditions() for col in (*c.primary, *c.detail_columns))
    prohibited = [c for c in desired if c.startswith(PROHIBITED_SOURCE_PREFIXES)]
    if prohibited:
        raise AssertionError(f"Prohibited Section 5 source(s): {prohibited}")
    required = {PATIENT_ID_COL, VISIT_DATE_COL, ESSDAI_PRIMARY_COL, ESSDAI_RAW_QC_COL}
    if required - schema:
        raise KeyError(f"Raw input lacks required columns: {sorted(required - schema)}")
    return list(dict.fromkeys(c for c in desired if c in schema))


def load_selected_raw_columns(input_path: Path) -> pd.DataFrame:
    return pd.read_parquet(input_path, columns=selected_raw_columns(input_path))


_UNRECOGNIZED: list[dict[str, Any]] = []


def normalize_binary_flag(series: pd.Series) -> pd.Series:
    """Map common binary encodings to pandas nullable Boolean values."""
    positive = {"1", "1.0", "true", "yes", "y", "positive", "present", "confirmed", "history"}
    negative = {"0", "0.0", "false", "no", "n", "negative", "absent"}
    missing = {str(x).strip().lower() for x in config.MISSING_STRINGS}
    result = pd.Series(pd.NA, index=series.index, dtype="boolean", name=series.name)
    for idx, value in series.items():
        if value is None or pd.isna(value):
            continue
        token = str(value).strip().lower()
        if token in positive:
            result.at[idx] = True
        elif token in negative:
            result.at[idx] = False
        elif token not in missing:
            _UNRECOGNIZED.append({"source_column": str(series.name), "original_value": str(value), "row_index": str(idx)})
    return result


def nullable_or(frame: pd.DataFrame) -> pd.Series:
    """Three-valued OR: true if any true, false if evaluated and none true."""
    if frame.shape[1] == 0:
        return pd.Series(pd.NA, index=frame.index, dtype="boolean")
    positives = frame.fillna(False).astype(bool).any(axis=1)
    evaluated = frame.notna().any(axis=1)
    out = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    out.loc[evaluated] = positives.loc[evaluated]
    return out


def derive_condition_status(general: pd.Series, history: pd.Series, confirmed: pd.Series,
                            evaluated: pd.Series | None = None) -> pd.Series:
    """Classify rheumatologic evidence without promoting history/general flags."""
    g = general.astype("boolean").fillna(False)
    h = history.astype("boolean").fillna(False)
    c = confirmed.astype("boolean").fillna(False)
    status = pd.Series("no_comorbidity", index=general.index, dtype="string")
    status.loc[g] = "status_uncertain"
    status.loc[h] = "history_only"
    status.loc[c] = "confirmed_present"
    return status


def _source_flag(out: pd.DataFrame, column: str) -> pd.Series:
    name = f"source__{column}"
    if name not in out:
        out[name] = (normalize_binary_flag(out[column]) if column in out else
                     pd.Series(pd.NA, index=out.index, dtype="boolean"))
    return out[name]


def derive_comorbidity_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive family-specific baseline indicators and retain normalized sources."""
    out = raw.copy()
    for condition in PAST_MEDICAL_HISTORY_CONDITIONS:
        # PMH blank is deliberately false only for *documented history*.  It is
        # never evidence of absence of current disease.
        flag = _source_flag(out, condition.primary[0])
        out[condition.name] = flag.fillna(False).astype("boolean")
        out[f"{condition.name}_documented_history"] = out[condition.name]
    for condition in [*RHEUMATOLOGIC_NON_SAID_CONDITIONS, *CONCOMITANT_SAID_CONDITIONS]:
        general, history, confirmed = [_source_flag(out, col) for col in condition.primary]
        status = derive_condition_status(general, history, confirmed)
        out[f"{condition.name}_status"] = status
        out[condition.name] = status.eq("confirmed_present").astype("boolean")
        out[f"{condition.name}_primary_exposure"] = status.map(
            {"confirmed_present": 1.0, "no_comorbidity": 0.0}
        ).astype(float)
    return out


def apply_exposure_definition(frame: pd.DataFrame, condition: Condition,
                              definition: str = "confirmed_present_vs_no_comorbidity") -> pd.DataFrame:
    if definition != "confirmed_present_vs_no_comorbidity":
        raise ValueError(f"Unsupported exposure definition: {definition}")
    out = frame.copy()
    status = out[f"{condition.name}_status"].astype("string")
    out["exposure"] = status.map({"confirmed_present": 1.0, "no_comorbidity": 0.0})
    return out

def collapse_same_patient_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    binary_cols = [c for c in df if str(df[c].dtype) == "boolean"]
    subtype_cols = [c for c in SUBTYPE_COLS if c in df]
    records, audit = [], []
    for (pid, date), group in df.groupby(["patient_id", "visit_date"], dropna=False, sort=False):
        row: dict[str, Any] = {"patient_id": pid, "visit_date": date}
        for col in binary_cols:
            row[col] = nullable_or(group[[col]]).iloc[0] if len(group) == 1 else nullable_or(group[[col]].T.reset_index(drop=True).T).iloc[0]
            # Above one-column OR is row-wise; collapse group explicitly.
            vals = group[col]
            row[col] = True if vals.eq(True).any() else (False if vals.eq(False).any() else pd.NA)
        conflict_cols = []
        for col in subtype_cols:
            vals = group[col].dropna().astype(str).str.strip().unique()
            row[col] = vals[0] if len(vals) == 1 else pd.NA
            row[f"{col}_conflict"] = len(vals) > 1
            if len(vals) > 1:
                conflict_cols.append(col)
        records.append(row)
        if len(group) > 1:
            audit.append({"patient_id": pid, "visit_date": date, "n_source_rows": len(group), "conflicting_subtype_columns": "|".join(conflict_cols)})
    collapsed = pd.DataFrame(records)
    for col in binary_cols:
        collapsed[col] = collapsed[col].astype("boolean")
    return collapsed, pd.DataFrame(audit, columns=["patient_id", "visit_date", "n_source_rows", "conflicting_subtype_columns"])


def build_baseline_comorbidity_dataset(raw: pd.DataFrame, spine: pd.DataFrame, pop: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    parsed = add_parsed_visit_dates(raw, PATIENT_ID_COL, VISIT_DATE_COL)
    n_pipe = int(parsed["had_pipe_delimited_date"].sum())
    derived = derive_comorbidity_indicators(parsed)
    collapsed, duplicate_audit = collapse_same_patient_date(derived)
    base_spine = spine.loc[spine["visit_number"].eq(0)].copy()
    if base_spine["patient_id"].duplicated().any():
        raise ValueError("More than one visit_number==0 row for a patient")
    base = base_spine.merge(collapsed, on=["patient_id", "visit_date"], how="left", validate="one_to_one")
    base_pop = pop.loc[pop["visit_number"].eq(0), ["patient_id", "baseline_pop_status", "essdai_total"]].drop_duplicates("patient_id")
    base = base.merge(base_pop, on="patient_id", how="left", validate="one_to_one")
    base = base.rename(columns={"visit_id": "baseline_visit_id", "visit_date": "baseline_date", "age_at_visit": "age_baseline", "baseline_pop_status": "baseline_pop", "essdai_total": "baseline_essdai_pop_pipeline"})
    # Primary ESSDAI comes from the available total-score column at canonical baseline.
    baseline_raw = parsed.merge(base_spine[["patient_id", "visit_date"]], on=["patient_id", "visit_date"], how="inner")
    ess = baseline_raw.groupby("patient_id", as_index=False).agg(
        baseline_essdai=(ESSDAI_PRIMARY_COL, lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0] if pd.to_numeric(s, errors="coerce").notna().any() else np.nan),
        baseline_essdai_raw_qc=(ESSDAI_RAW_QC_COL, lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0] if pd.to_numeric(s, errors="coerce").notna().any() else np.nan),
    )
    essdai_source_columns = list(dict.fromkeys(c for c in (ESSDAI_PRIMARY_COL, ESSDAI_RAW_QC_COL) if c in base))
    base = base.drop(columns=essdai_source_columns).merge(ess, on="patient_id", how="left", validate="one_to_one")
    # These fields are absence-by-default checkboxes in the clinical extract:
    # an empty value means that the patient does not have the condition, rather
    # than that the condition was not evaluated.  Resolve that convention once
    # in the baseline dataset so prevalence plots and downstream models use the
    # full relevant cohort as their denominator/reference population.
    pmh_names = [c.name for c in PAST_MEDICAL_HISTORY_CONDITIONS]
    base[pmh_names] = base[pmh_names].fillna(False).astype("boolean")
    for condition in [*RHEUMATOLOGIC_NON_SAID_CONDITIONS, *CONCOMITANT_SAID_CONDITIONS]:
        flags = [base.get(f"source__{col}", pd.Series(pd.NA, index=base.index,
                                                     dtype="boolean"))
                 for col in condition.primary]
        status = derive_condition_status(*flags)
        base[f"{condition.name}_status"] = status
        base[condition.name] = status.eq("confirmed_present").astype("boolean")
        base[f"{condition.name}_primary_exposure"] = status.map(
            {"confirmed_present": 1.0, "no_comorbidity": 0.0})
    non_said = [c.name for c in RHEUMATOLOGIC_NON_SAID_CONDITIONS]
    said = [c.name for c in CONCOMITANT_SAID_CONDITIONS]
    base["n_general_medical_history"] = base[pmh_names].astype(int).sum(axis=1)
    base["any_general_medical_history"] = base["n_general_medical_history"].gt(0).astype("boolean")
    base["n_rheumatologic_non_said"] = base[non_said].astype(int).sum(axis=1)
    base["any_rheumatologic_non_said"] = base["n_rheumatologic_non_said"].gt(0).astype("boolean")
    base["n_concomitant_said"] = base[said].astype(int).sum(axis=1)
    base["concomitant_said_any"] = base["n_concomitant_said"].gt(0).astype("boolean")
    malignancies = ["breast_cancer", "lung_cancer", "colon_cancer", "thyroid_cancer",
                    "head_neck_cancer", "other_malignancy"]
    base["any_non_lymphoma_malignancy_history"] = base[malignancies].any(axis=1).astype("boolean")
    if len(base) != base["patient_id"].nunique():
        raise ValueError("Baseline dataset is not one row per patient")
    if not base["baseline_date"].equals(base["observed_baseline_date"]):
        raise ValueError("Baseline date is not the canonical observed baseline date")
    return base, duplicate_audit, n_pipe


def summarize_historical_family(base: pd.DataFrame, conditions: Sequence[Condition],
                                family_label: str = "past_medical_history") -> pd.DataFrame:
    n_total = len(base); rows = []
    for c in conditions:
        positive = base.get(c.name, pd.Series(False, index=base.index)).fillna(False).eq(True)
        n = int(positive.sum())
        rows.append({"condition": c.name, "display_label": c.label,
                     "condition_family": family_label, "clinical_category": c.clinical_category,
                     "n_documented_history": n, "n_total_patients": n_total,
                     "percent_documented_total_cohort": 100*n/n_total if n_total else np.nan,
                     "summary_label": "Minimum documented proportion in the baseline cohort",
                     "source_columns": "|".join(c.primary)})
    return pd.DataFrame(rows)


def summarize_confirmed_family(base: pd.DataFrame, conditions: Sequence[Condition]) -> pd.DataFrame:
    rows=[]
    for c in conditions:
        status=base.get(f"{c.name}_status", pd.Series("no_comorbidity", index=base.index)).fillna("no_comorbidity")
        counts=status.value_counts(); N=len(status)
        rows.append({"condition":c.name,"display_label":c.label,"condition_family":c.condition_family,
          "clinical_category":c.clinical_category,"n_confirmed_present":int(counts.get("confirmed_present",0)),
          "n_history_only":int(counts.get("history_only",0)),"n_status_uncertain":int(counts.get("status_uncertain",0)),
          "n_no_comorbidity":int(counts.get("no_comorbidity",0)),"N_evaluable":N,
          "pct_confirmed":100*int(counts.get("confirmed_present",0))/N if N else np.nan})
    return pd.DataFrame(rows)


def summarize_overall_prevalence(base: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper: confirmed rheumatologic conditions only."""
    out=summarize_confirmed_family(base,[*RHEUMATOLOGIC_NON_SAID_CONDITIONS,*CONCOMITANT_SAID_CONDITIONS])
    return out.rename(columns={"N_evaluable":"n_evaluable_primary","pct_confirmed":"pct_confirmed_among_evaluable"}).assign(
      n_total_cohort=len(base),n_missing=0,pct_confirmed_total_cohort=lambda x:100*x.n_confirmed_present/len(base) if len(base) else np.nan)


def _monte_carlo_p(table: np.ndarray, replicates: int, rng: np.random.Generator) -> float:
    observed=chi2_contingency(table,correction=False)[0]; outcomes=np.repeat([1,0],table.sum(axis=0)); groups=np.repeat(np.arange(3),table.sum(axis=1)); exceed=0
    for _ in range(replicates):
        shuffled=rng.permutation(outcomes); sim=np.array([[np.sum(shuffled[groups==g]==1),np.sum(shuffled[groups==g]==0)] for g in range(3)])
        exceed += chi2_contingency(sim,correction=False)[0] >= observed
    return (exceed+1)/(replicates+1)


def calculate_or_and_fisher(table: np.ndarray) -> dict[str, Any]:
    empty={"odds_ratio_pop2_vs_pop3":np.nan,"or_ci95_low":np.nan,"or_ci95_high":np.nan,"fisher_exact_p_value":np.nan,"zero_cell_correction_used":False}
    if table.shape != (2,2) or table.sum(axis=1).min()==0 or table.sum(axis=0).min()==0: return empty
    _,p=fisher_exact(table); corrected=bool((table==0).any()); a,b,c,d=table.ravel().astype(float)
    if corrected: a,b,c,d=a+.5,b+.5,c+.5,d+.5
    odds=a*d/(b*c); se=math.sqrt(1/a+1/b+1/c+1/d)
    return {"odds_ratio_pop2_vs_pop3":odds,"or_ci95_low":math.exp(math.log(odds)-1.96*se),"or_ci95_high":math.exp(math.log(odds)+1.96*se),"fisher_exact_p_value":p,"zero_cell_correction_used":corrected}


def summarize_family_by_pop(base: pd.DataFrame, conditions: Sequence[Condition], *,
                            run_sparse_monte_carlo: bool=False, replicates: int=10_000,
                            seed: int=RANDOM_SEED) -> pd.DataFrame:
    rng=np.random.default_rng(seed); rows=[]
    for c in conditions:
        row={"condition":c.name,"display_label":c.label,"condition_family":c.condition_family}; table=[]
        for i,pop_name in enumerate(("Pop1","Pop2","Pop3"),1):
            values=base.loc[base.baseline_pop.eq(pop_name),c.name].fillna(False).astype(bool); n=int(values.sum()); N=len(values)
            row.update({f"n_pop{i}":n,f"N_pop{i}":N,f"pct_pop{i}":100*n/N if N else np.nan}); table.append([n,N-n])
        arr=np.asarray(table,dtype=int)
        total_positive=int(arr[:,0].sum())
        denominators = arr.sum(axis=1)
        rates = np.divide(arr[:, 0], denominators, out=np.zeros(3), where=denominators > 0)
        if (denominators==0).any() or total_positive==0 or np.allclose(rates, rates[0]):
            test,p,expected_min,sparse="not estimable",np.nan,np.nan,False
        else:
            _,asym_p,_,expected=chi2_contingency(arr,correction=False); expected_min=float(expected.min()); sparse=expected_min<5
            if not sparse: test,p="Pearson chi-square",asym_p
            elif run_sparse_monte_carlo: test,p=f"Monte Carlo chi-square ({replicates} replicates)",_monte_carlo_p(arr,replicates,rng)
            else: test,p="descriptive only - sparse table",np.nan
        row.update({"global_test":test,"global_p_value":p,"minimum_expected_cell":expected_min,"sparse_table_flag":sparse})
        row.update(calculate_or_and_fisher(arr[1:3])); rows.append(row)
    out=pd.DataFrame(rows); out["fdr_bh_q_value"]=apply_fdr(out["global_p_value"]); return out


def summarize_prevalence_by_pop(base: pd.DataFrame, replicates: int=10_000, seed: int=RANDOM_SEED,
                                run_sparse_monte_carlo: bool=False) -> pd.DataFrame:
    return summarize_family_by_pop(base,[*RHEUMATOLOGIC_NON_SAID_CONDITIONS,*CONCOMITANT_SAID_CONDITIONS],run_sparse_monte_carlo=run_sparse_monte_carlo,replicates=replicates,seed=seed)


def apply_fdr(p_values: pd.Series) -> pd.Series:
    out=pd.Series(np.nan,index=p_values.index,dtype=float); valid=p_values.notna()
    if valid.any(): out.loc[valid]=multipletests(p_values.loc[valid],method="fdr_bh")[1]
    return out

def build_longitudinal_essdai_dataset(raw: pd.DataFrame, spine: pd.DataFrame, pop: pd.DataFrame, domains: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    parsed = add_parsed_visit_dates(raw, PATIENT_ID_COL, VISIT_DATE_COL)
    values = parsed.assign(essdai_total_recoded=pd.to_numeric(parsed[ESSDAI_PRIMARY_COL], errors="coerce"), essdai_total_raw_qc=pd.to_numeric(parsed[ESSDAI_RAW_QC_COL], errors="coerce"))
    values = values.groupby(["patient_id", "visit_date"], as_index=False).agg(essdai_total_recoded=("essdai_total_recoded", "first"), essdai_total_raw_qc=("essdai_total_raw_qc", "first"))
    long = spine.merge(values, on=["patient_id", "visit_date"], how="left", validate="one_to_one")
    popcols = pop[["visit_id", "pop_status"]].drop_duplicates("visit_id")
    long = long.merge(popcols, on="visit_id", how="left", validate="one_to_one").merge(domains.drop(columns=["patient_id", "visit_date"]), on="visit_id", how="left", validate="one_to_one")
    rheum_status_cols = [f"{c.name}_status" for c in RHEUMATOLOGIC_NON_SAID_CONDITIONS]
    bcols = ["patient_id", "baseline_essdai", "baseline_pop", "age_baseline", "sex",
             *CONDITION_NAMES, *rheum_status_cols]
    long = long.drop(columns=["sex"], errors="ignore").merge(base[bcols], on="patient_id", how="left", validate="many_to_one")
    if long["patient_id"].isna().any() or not long["visit_id"].is_unique:
        raise ValueError("Longitudinal analytic dataset violates identity constraints")
    valid = long["essdai_total_recoded"].dropna()
    if not valid.between(0, 123).all(): raise ValueError("Recoded ESSDAI outside 0..123")
    return long


def build_severe5_survival_dataset(long: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in long.groupby("patient_id"):
        b = base.loc[base["patient_id"].eq(pid)].iloc[0]
        follow = g.loc[(g["visit_number"] > 0) & g["essdai_total_recoded"].notna()].sort_values("visit_date")
        if pd.isna(b["baseline_essdai"]) or b["baseline_essdai"] >= SEVERE_THRESHOLD or follow.empty: continue
        event_rows = follow.loc[follow["essdai_total_recoded"] >= SEVERE_THRESHOLD]
        event_date = event_rows["visit_date"].iloc[0] if not event_rows.empty else pd.NaT
        last = follow["visit_date"].max(); end = event_date if pd.notna(event_date) else last
        row = b.to_dict(); row.update({"last_evaluable_date": last, "event_date": event_date, "followup_days": (end-b["baseline_date"]).days, "severe5_event": int(pd.notna(event_date))})
        rows.append(row)
    out = pd.DataFrame(rows); out["followup_years"] = out.get("followup_days", pd.Series(dtype=float))/365.25
    if len(out) and ((out["followup_days"] <= 0).any() or (out["event_date"].dropna() > out.loc[out["event_date"].notna(), "last_evaluable_date"]).any()): raise ValueError("Invalid severe-event timing")
    return out


def build_new_domain_survival_dataset(long: pd.DataFrame, base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, audit = [], []
    for pid, g in long.sort_values("visit_date").groupby("patient_id"):
        baseline_g = g.loc[g["visit_number"].eq(0)]
        follow = g.loc[g["visit_number"] > 0]
        if baseline_g.empty or follow.empty: continue
        bdom = baseline_g.iloc[0]; event_candidates = []; censoring_dates = []
        n_eval = 0; n_inactive = 0
        for domain in ORGAN_DOMAINS:
            evaluable_col = DOMAIN_EVALUABLE[domain]
            baseline_evaluable = bool(bdom[evaluable_col]) if pd.notna(bdom[evaluable_col]) else False
            state = bool(bdom[domain]) if baseline_evaluable else pd.NA
            at_risk = baseline_evaluable and not bool(state)
            n_eval += int(baseline_evaluable)
            n_inactive += int(at_risk)
            evaluable_follow = follow.loc[follow[evaluable_col].eq(True)] if at_risk else follow.iloc[0:0]
            events = evaluable_follow.loc[evaluable_follow[domain].eq(True), "visit_date"]
            date = events.min() if not events.empty else pd.NaT
            last_domain_evaluable = evaluable_follow["visit_date"].max() if not evaluable_follow.empty else pd.NaT
            audit.append({"patient_id": pid, "domain": domain, "baseline_evaluable": baseline_evaluable,
                          "baseline_state": state, "at_risk": at_risk,
                          "last_domain_evaluable_date": last_domain_evaluable,
                          "domain_event_date": date})
            if pd.notna(date): event_candidates.append((date, domain))
            if pd.notna(last_domain_evaluable): censoring_dates.append(last_domain_evaluable)
        # A patient contributes only with at least one baseline-inactive domain
        # and a later evaluation of at least one such domain.
        if n_inactive == 0 or not censoring_dates: continue
        first_date, first_name = min(event_candidates) if event_candidates else (pd.NaT, pd.NA)
        last = max(censoring_dates); end = first_date if pd.notna(first_date) else last
        b = base.loc[base["patient_id"].eq(pid)].iloc[0].to_dict()
        b.update({"first_new_domain_date": first_date, "first_new_domain_name": first_name, "new_domain_event": int(pd.notna(first_date)), "n_domains_inactive_at_baseline": n_inactive, "n_domains_evaluable_at_baseline": n_eval, "last_evaluable_date": last, "followup_days": (end-b["baseline_date"]).days})
        rows.append(b)
    out = pd.DataFrame(rows); out["followup_years"] = out.get("followup_days", pd.Series(dtype=float))/365.25
    if len(out) and (out["followup_days"] <= 0).any(): raise ValueError("New-domain event/censoring time must be positive")
    return out, pd.DataFrame(audit)


def _empty_progression(c: Condition, outcome: str, estimand: str, warning: str, **counts: Any) -> dict[str, Any]:
    return {"comorbidity": c.name, "display_label": c.label, "outcome": outcome, "estimand": estimand, "model_type": "not fitted", "effect_measure": "Not estimable", "estimate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p_value": np.nan, "n_patients": counts.get("n_patients", 0), "n_followup_observations": counts.get("n_followup_observations", 0), "n_events": counts.get("n_events", np.nan), "n_complete_cases": counts.get("n_complete_cases", 0), "baseline_reference_group": "Comorbidity absent", "adjustment_covariates": "baseline ESSDAI; baseline Pop; age; sex", "time_scale": "years since observed baseline", "threshold": SEVERE_THRESHOLD if outcome == "Progression to ESSDAI >=5" else np.nan, "model_converged": False, "proportional_hazards_p": np.nan, "sparse_event_flag": True, "model_status": "not_estimable", "warning": warning, "interpretation": "Not estimable; no causal interpretation is warranted."}


def restrict_to_primary_exposure(data: pd.DataFrame, c: Condition) -> pd.DataFrame:
    """Keep only confirmed-present and no-comorbidity primary contrast rows."""
    status_col = f"{c.name}_status"
    if status_col not in data:
        raise KeyError(f"Primary model input lacks {status_col}")
    out = data.loc[data[status_col].isin(["confirmed_present", "no_comorbidity"])].copy()
    out[c.name] = out[status_col].eq("confirmed_present").astype(int)
    return out


def fit_mixed_model(long: pd.DataFrame, c: Condition) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import statsmodels.formula.api as smf
    status_col = f"{c.name}_status"
    cols = ["patient_id", "essdai_total_recoded", "time_since_observed_baseline_years",
            status_col, "baseline_essdai", "baseline_pop", "age_baseline", "sex", "visit_number"]
    data = restrict_to_primary_exposure(long.loc[long["visit_number"] > 0, cols], c)
    data = data.dropna(subset=["patient_id", "essdai_total_recoded",
                              "time_since_observed_baseline_years", "baseline_essdai",
                              "baseline_pop", "age_baseline", "sex"])
    eligible = data.groupby("patient_id").size(); n = len(eligible)
    exposure_counts = data.drop_duplicates("patient_id")[c.name].value_counts()
    if n < 5 or data[c.name].nunique() < 2 or exposure_counts.get(1, 0) < 5:
        row = _empty_progression(c, "Longitudinal ESSDAI trajectory", "Time x comorbidity", "Too few complete patients or no exposure variation", n_patients=n, n_followup_observations=len(data), n_complete_cases=n)
        return [row], {"comorbidity": c.name, "outcome": "ESSDAI trajectory", "convergence": False, "warning": row["warning"]}
    formula = "essdai_total_recoded ~ time_since_observed_baseline_years * Q('%s') + baseline_essdai + C(baseline_pop) + age_baseline + C(sex)" % c.name
    model_type, fit, warning_text = "Linear mixed model (random intercept)", None, ""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always"); fit = smf.mixedlm(formula, data, groups=data["patient_id"]).fit(reml=False, method="lbfgs")
            warning_text = "; ".join(str(w.message) for w in caught)
        variance = float(fit.cov_re.iloc[0, 0]); singular = variance < 1e-8
        if not fit.converged or singular: raise RuntimeError(f"Mixed model unstable: converged={fit.converged}, random-intercept variance={variance:.3g}")
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        warning_text = f"Mixed model failed ({exc}); used GEE fallback"
        try:
            fit = GEE.from_formula(formula, groups="patient_id", data=data, family=Gaussian(), cov_struct=Exchangeable()).fit(); model_type = "Gaussian GEE (exchangeable)"
        except (ValueError, np.linalg.LinAlgError) as gee_exc:
            row = _empty_progression(c, "Longitudinal ESSDAI trajectory", "Time x comorbidity", f"Mixed model and GEE failed: {gee_exc}", n_patients=n, n_followup_observations=len(data), n_complete_cases=n)
            return [row], {"comorbidity": c.name, "outcome": "ESSDAI trajectory", "convergence": False, "warning": row["warning"]}
    interaction = f"time_since_observed_baseline_years:Q('{c.name}')"
    exposure = f"Q('{c.name}')"
    result_rows = []
    for term, estimand, measure in ((interaction, "Difference in annual ESSDAI slope", "Beta per year"), (exposure, "Adjusted mean difference during follow-up", "Adjusted mean difference")):
        est, se, p = float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term])
        result_rows.append({**_empty_progression(c, "Longitudinal ESSDAI trajectory", estimand, warning_text, n_patients=n, n_followup_observations=len(data), n_complete_cases=n), "model_type": model_type, "effect_measure": measure, "estimate": est, "ci95_low": est-1.96*se, "ci95_high": est+1.96*se, "p_value": p, "model_converged": True, "sparse_event_flag": False, "model_status": "fitted", "interpretation": f"Adjusted association estimate ({measure}); this is not a causal effect."})
    return result_rows, {"comorbidity": c.name, "outcome": "ESSDAI trajectory", "convergence": True, "model_type": model_type, "warning": warning_text}


def fit_cox_model(data: pd.DataFrame, c: Condition, event_col: str, outcome: str, minimum_events: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit confirmed-present versus no-comorbidity; exclude ambiguous statuses."""
    status_col = f"{c.name}_status"
    cols = ["followup_years", event_col, status_col, "baseline_essdai",
            "baseline_pop", "age_baseline", "sex"]
    d = restrict_to_primary_exposure(data[cols], c)
    n_exposure_excluded = int(len(data) - len(d))
    d = d.dropna().copy()
    counts = {"n_patients": len(d), "n_events": int(d[event_col].sum()) if len(d) else 0,
              "n_complete_cases": len(d)}
    exposure_counts = d[c.name].value_counts()
    if (len(d) < 5 or d[c.name].nunique() < 2 or counts["n_events"] < minimum_events
            or exposure_counts.get(1, 0) < 5):
        row = _empty_progression(c, outcome, "Baseline comorbidity", f"Insufficient events (<{minimum_events}) or exposure variation", **counts)
        row.update({"model_status": "insufficient_events",
                    "baseline_reference_group": "Comorbidity absent",
                    "n_ambiguous_status_excluded": n_exposure_excluded})
        return row, {"comorbidity": c.name, "outcome": outcome, "convergence": False,
                     "events": counts["n_events"],
                     "n_ambiguous_status_excluded": n_exposure_excluded,
                     "warning": row["warning"]}
    if importlib.util.find_spec("lifelines") is None:
        row = _empty_progression(c, outcome, "Confirmed present vs no comorbidity", "lifelines is not installed; Cox model not executed", **counts)
        row.update({"baseline_reference_group": "No comorbidity",
                    "n_ambiguous_status_excluded": n_exposure_excluded})
        return row, {"comorbidity": c.name, "outcome": outcome, "convergence": False,
                     "n_ambiguous_status_excluded": n_exposure_excluded,
                     "warning": row["warning"]}
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    d[c.name] = d[c.name].astype(int); d = pd.get_dummies(d, columns=["baseline_pop", "sex"], drop_first=True, dtype=float)
    # Full model only when event support is reasonable; otherwise preserve the
    # exposure and baseline ESSDAI in a prespecified reduced model.
    covars = [x for x in d if x not in ("followup_years", event_col)]
    reduced = counts["n_events"] / max(len(covars), 1) < 5
    if reduced:
        keep = ["followup_years", event_col, c.name, "baseline_essdai", "age_baseline"]
        d = d[keep]; model_type = "Cox PH (prespecified reduced adjustment)"
    else: model_type = "Cox proportional hazards"
    try:
        fitter = CoxPHFitter(penalizer=0.0); fitter.fit(d, duration_col="followup_years", event_col=event_col)
        summary = fitter.summary.loc[c.name]; ph = proportional_hazard_test(fitter, d, time_transform="rank")
        ph_p = float(ph.summary.loc[c.name, "p"])
        warning_text = "Proportional-hazards assumption may be violated." if ph_p < .05 else ""
        row = {**_empty_progression(c, outcome, "Confirmed present vs no comorbidity", warning_text, **counts), "model_type": model_type, "effect_measure": "Hazard ratio", "estimate": float(summary["exp(coef)"]), "ci95_low": float(summary["exp(coef) lower 95%"]), "ci95_high": float(summary["exp(coef) upper 95%"]), "p_value": float(summary["p"]), "model_converged": True, "proportional_hazards_p": ph_p, "sparse_event_flag": reduced, "model_status": "reduced_adjustment" if reduced else "fitted", "baseline_reference_group": "No comorbidity", "n_ambiguous_status_excluded": n_exposure_excluded, "interpretation": "Adjusted confirmed-present versus no-comorbidity association; this is not a causal effect."}
        return row, {"comorbidity": c.name, "outcome": outcome, "convergence": True, "events": counts["n_events"], "events_per_parameter": counts["n_events"]/max(len(d.columns)-2, 1), "n_ambiguous_status_excluded": n_exposure_excluded, "proportional_hazards_p": ph_p, "warning": warning_text}
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        row = _empty_progression(c, outcome, "Baseline comorbidity", f"Cox model failed: {exc}", **counts)
        row.update({"model_status": "failed", "baseline_reference_group": "Comorbidity absent",
                    "n_ambiguous_status_excluded": n_exposure_excluded})
        return row, {"comorbidity": c.name, "outcome": outcome, "convergence": False,
                     "events": counts["n_events"],
                     "n_ambiguous_status_excluded": n_exposure_excluded,
                     "warning": row["warning"]}


def _plot_save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", metadata={"Title": "Section 5 comorbidity analysis"}); plt.close(fig)
    if path.stat().st_size < 1000: raise IOError(f"Generated figure appears empty: {path}")


def write_intermediate_dataset(data: pd.DataFrame, parquet_path: Path) -> tuple[Path, Path]:
    """Write an intermediate dataset in both Parquet and review-friendly CSV."""
    csv_path = parquet_path.with_suffix(".csv")
    data.to_parquet(parquet_path, index=False)
    data.to_csv(csv_path, index=False)
    return parquet_path, csv_path


def _nonnegative_interval_errors(
    estimate: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> np.ndarray:
    """Return matplotlib-compatible CI distances, guarding round-off error.

    Matplotlib rejects even tiny negative error distances. Confidence interval
    distances are clipped at zero after validating the bound ordering.
    """
    estimate_array = np.asarray(estimate, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    finite = np.isfinite(estimate_array) & np.isfinite(lower_array) & np.isfinite(upper_array)
    if np.any(lower_array[finite] > upper_array[finite]):
        raise ValueError("Confidence interval lower bound exceeds upper bound")
    return np.vstack((np.maximum(estimate_array - lower_array, 0.0),
                      np.maximum(upper_array - estimate_array, 0.0)))


def create_dotplot(overall: pd.DataFrame) -> None:
    d = overall.sort_values(["pct_total_cohort", "display_label"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(10, max(6, .45*len(d))))
    y = np.arange(len(d)); pct = d["pct_total_cohort"].to_numpy(dtype=float)
    bars = ax.barh(y, pct, color="#2c7fb8")
    ax.set_yticks(y, d["display_label"]); ax.set_xlabel("Patients at baseline (%)"); ax.grid(axis="x", alpha=.25)
    xmax = max(5.0, float(np.nanmax(pct)) if len(pct) else 5.0)
    ax.set_xlim(0, xmax * 1.25)
    for bar, (_, r) in zip(bars, d.iterrows()):
        ax.annotate(f"{int(r.n_positive)}/{int(r.n_total_cohort)} ({r.pct_total_cohort:.1f}%)",
                    (bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
    fig.text(.01, .01, "Bars use the total baseline cohort denominator; empty condition fields are coded as absent.", fontsize=8)
    fig.subplots_adjust(bottom=.1, left=.3); _plot_save(fig, FIGURES_DIR/"07_comorbidities_dotplot.pdf")


def create_grouped_barplot(by_pop: pd.DataFrame) -> None:
    d = by_pop.sort_values(["pct_total_cohort", "display_label"], ascending=[True, False]).reset_index(drop=True)
    values = d[[f"pct_pop{i}" for i in range(1, 4)]].to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#d9d9d9")
    fig, ax = plt.subplots(figsize=(8, max(7, .45*len(d))))
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=max(1.0, float(np.nanmax(values)) if np.isfinite(values).any() else 1.0))
    ax.set_xticks(range(3), ["Pop 1", "Pop 2", "Pop 3"])
    ax.set_yticks(range(len(d)), d["display_label"])
    for row_idx, r in d.iterrows():
        for col_idx in range(3):
            pct = r[f"pct_pop{col_idx+1}"]; n = r[f"n_pop{col_idx+1}"]; N = r[f"N_pop{col_idx+1}"]
            label = "Not estimable" if pd.isna(pct) else f"{pct:.1f}%\n({int(n)}/{int(N)})"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=7,
                    color="white" if pd.notna(pct) and pct > 0.6*np.nanmax(values) else "black")
    ax.set_title("Baseline comorbidity prevalence by Pop")
    cbar = fig.colorbar(image, ax=ax); cbar.set_label("Prevalence (%)")
    fig.text(.01, .01, "Cells use baseline Pop denominators; empty condition fields are coded as absent. Grey cells are not estimable.", fontsize=8)
    fig.subplots_adjust(left=.35, bottom=.12); _plot_save(fig, FIGURES_DIR/"07_comorbidities_grouped_bar.pdf")


def create_progression_forestplot(progression: pd.DataFrame, path: Path | None = None) -> None:
    panels = [("Longitudinal ESSDAI trajectory", "Difference in annual ESSDAI slope", 0, False), ("Progression to ESSDAI >=5", "Progression to ESSDAI ≥5", 1, True), ("New ESSDAI-domain involvement", "Development of new ESSDAI-domain involvement", 1, True)]
    fig, axes = plt.subplots(1, 3, figsize=(18, max(7, .5*len(PROGRESSION_CONDITIONS))), sharey=True); order = PROGRESSION_CONDITION_NAMES_ORDERED[::-1]; labels={c.name:c.label for c in PROGRESSION_CONDITIONS}
    for ax, (outcome, title, null, logscale) in zip(axes, panels):
        d = progression[(progression["outcome"] == outcome) & ((progression["effect_measure"] == "Beta per year") if "trajectory" in outcome else True)].set_index("comorbidity")
        for y, name in enumerate(order):
            if name in d.index and np.isfinite(d.loc[name, ["estimate", "ci95_low", "ci95_high"]].astype(float)).all():
                r=d.loc[name]; errors = _nonnegative_interval_errors([r.estimate], [r.ci95_low], [r.ci95_high])
                ax.errorbar(r.estimate, y, xerr=errors, fmt="o", color="#2166ac" if r.model_status=="fitted" else "#b2182b", capsize=2)
                ax.annotate(f"n={int(r.n_patients)}" + (f", e={int(r.n_events)}" if pd.notna(r.n_events) else ""), (r.estimate,y), xytext=(5,4), textcoords="offset points", fontsize=6)
            else: ax.plot(null, y, marker="x", color="gray")
        ax.axvline(null, color="black", ls="--", lw=.8); ax.set_title(title); ax.grid(axis="x", alpha=.2)
        if logscale: ax.set_xscale("log"); ax.set_xlabel("Hazard ratio (log scale)")
        else: ax.set_xlabel("Adjusted beta per year")
    axes[0].set_yticks(range(len(order)), [labels[x] for x in order]); fig.text(.01,.01,"Models include only confirmed-present and no-comorbidity patients and adjust for baseline ESSDAI, baseline Pop, age, and sex when support permits. History-only and uncertain statuses are excluded. X marks not estimable; red denotes reduced models. Associations are not causal.",fontsize=8)
    fig.subplots_adjust(left=.18,bottom=.1,wspace=.15); _plot_save(fig, path or FIGURES_DIR/"07_rheumatologic_comorbidities_progression_forestplot.pdf")


def run_qc_checks(base: pd.DataFrame, long: pd.DataFrame, severe: pd.DataFrame, new_domain: pd.DataFrame, domain_audit: pd.DataFrame) -> dict[str, Any]:
    if base["patient_id"].isna().any() or base["patient_id"].duplicated().any(): raise ValueError("Invalid baseline patient identity")
    if not base["baseline_date"].eq(base["observed_baseline_date"]).all(): raise ValueError("Noncanonical baseline detected")
    for col in CONDITION_NAMES + ALL_DOMAINS + list(DOMAIN_EVALUABLE.values()):
        source = base[col] if col in base else long[col]
        if str(source.dtype) != "boolean": raise TypeError(f"{col} is not nullable boolean")
    if len(domain_audit):
        if ((domain_audit["baseline_state"] == True) & domain_audit["domain_event_date"].notna()).any(): raise ValueError("Baseline-active domain counted as new")  # noqa: E712
        if ((~domain_audit["baseline_evaluable"]) & domain_audit["at_risk"]).any():
            raise ValueError("Baseline-unevaluable domain entered the new-domain risk set")
        if (domain_audit["domain_event_date"].notna() & ~domain_audit["at_risk"]).any():
            raise ValueError("New-domain event occurred outside its domain-specific risk set")
    both = int((base["fibromyalgia"].eq(True) & base["depression"].eq(True)).sum())
    return {"n_fibromyalgia_and_depression": both, "sum_condition_prevalences_pct": float(100*base[CONDITION_NAMES].eq(True).sum().sum()/len(base)) if len(base) else np.nan}


def missingness_table(base: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for group, data in [("Overall", base), *[(p, base[base["baseline_pop"].eq(p)]) for p in ("Pop1","Pop2","Pop3","Unclassifiable")]]:
        for col in CONDITION_NAMES + ["baseline_essdai", "age_baseline", "sex"]:
            rows.append({"group":group,"variable":col,"n_total":len(data),"n_missing":int(data[col].isna().sum()),"pct_missing":100*data[col].isna().mean() if len(data) else np.nan})
    return pd.DataFrame(rows)


def source_mapping(raw_columns: Iterable[str]) -> pd.DataFrame:
    raw_columns=set(raw_columns); rows=[]
    for c in iter_analysis_conditions():
        prohibited=[x for x in (*c.primary,*c.detail_columns) if x.startswith(PROHIBITED_SOURCE_PREFIXES)]
        if prohibited: raise AssertionError(f"Prohibited source selected: {prohibited}")
        rows.append({"condition":c.name,"condition_family":c.condition_family,
          "clinical_category":c.clinical_category,"source_columns":"|".join(c.primary),
          "status_definition":("documented_history=True only when PMH source is positive; blank means no documented history"
             if c.condition_family=="past_medical_history" else
             "confirmed_present > history_only > status_uncertain > no_comorbidity; confirmed_present is primary positive"),
          "availability":"available" if any(x in raw_columns for x in c.primary) else "unavailable"})
    return pd.DataFrame(rows)


def create_family_plot(table: pd.DataFrame, path: Path, historical: bool=False) -> None:
    value="percent_documented_total_cohort" if historical else "pct_confirmed"
    d=table.sort_values(value); fig,ax=plt.subplots(figsize=(10,max(5,.32*len(d))))
    ax.barh(np.arange(len(d)),d[value],color="#2c7fb8"); ax.set_yticks(np.arange(len(d)),d.display_label)
    ax.set_xlabel("Patients with documented history (%)" if historical else "Patients with confirmed condition (%)")
    ax.grid(axis="x",alpha=.25); _plot_save(fig,path)


def main(argv: Sequence[str] | None = None) -> int:
    args=parse_args(argv); ensure_directories(); logger=setup_logging(); np.random.seed(args.random_seed)
    logger.info("[1/7] Loading canonical sources")
    timestamps=check_upstream_artifacts(args.input,args.rebuild_upstream,logger)
    spine=load_visit_spine(); pop=load_pop_classification(); domains=load_domain_flags(); raw=load_selected_raw_columns(args.input)
    logger.info("[2/7] Deriving three separate baseline condition families")
    base,duplicates,n_pipe=build_baseline_comorbidity_dataset(raw,spine,pop); write_intermediate_dataset(base,BASELINE_PATH)
    families=[("past_medical_history",PAST_MEDICAL_HISTORY_CONDITIONS,True),
      ("rheumatologic_comorbidities",RHEUMATOLOGIC_NON_SAID_CONDITIONS,False),
      ("concomitant_said",CONCOMITANT_SAID_CONDITIONS,False)]
    logger.info("[3/7] Writing separate descriptive outputs")
    produced=[]
    for stem,conditions,historical in families:
        overall=(summarize_historical_family(base,conditions) if historical else summarize_confirmed_family(base,conditions))
        by_pop=summarize_family_by_pop(base,conditions,run_sparse_monte_carlo=args.run_sparse_monte_carlo,
                                       replicates=args.monte_carlo_replicates,seed=args.random_seed)
        op=TABLES_DIR/f"07_{stem}_overall.csv"; bp=TABLES_DIR/f"07_{stem}_by_pop.csv"
        overall.to_csv(op,index=False); by_pop.to_csv(bp,index=False); produced.extend([op,bp])
        fp=FIGURES_DIR/f"07_{stem}.pdf"; create_family_plot(overall,fp,historical); produced.append(fp)
    logger.info("[4/7] Preserving canonical longitudinal outcome datasets")
    long=build_longitudinal_essdai_dataset(raw,spine,pop,domains,base); write_intermediate_dataset(long,LONGITUDINAL_PATH)
    severe=build_severe5_survival_dataset(long,base); write_intermediate_dataset(severe,SEVERE_PATH)
    new_domain,domain_audit=build_new_domain_survival_dataset(long,base); write_intermediate_dataset(new_domain,NEW_DOMAIN_PATH); write_intermediate_dataset(domain_audit,DOMAIN_AUDIT_PATH)
    logger.info("[5/7] Fitting confirmed non-SAiD rheumatologic progression models")
    progression_rows: list[dict[str, Any]]=[]; diagnostics: list[dict[str, Any]]=[]
    for condition in RHEUMATOLOGIC_NON_SAID_CONDITIONS:
        rows,diagnostic=fit_mixed_model(long,condition); progression_rows.extend(rows); diagnostics.append(diagnostic)
        row,diagnostic=fit_cox_model(severe,condition,"severe5_event","Progression to ESSDAI >=5",args.minimum_events); progression_rows.append(row); diagnostics.append(diagnostic)
        row,diagnostic=fit_cox_model(new_domain,condition,"new_domain_event","New ESSDAI-domain involvement",args.minimum_events); progression_rows.append(row); diagnostics.append(diagnostic)
    progression=pd.DataFrame(progression_rows)
    progression["fdr_bh_q_value"]=progression.groupby(["outcome","estimand"])["p_value"].transform(apply_fdr)
    progression_path=TABLES_DIR/"07_rheumatologic_comorbidities_progression.csv"
    progression.to_csv(progression_path,index=False); produced.append(progression_path)
    progression_plot=FIGURES_DIR/"07_rheumatologic_comorbidities_progression_forestplot.pdf"
    create_progression_forestplot(progression, progression_plot); produced.append(progression_plot)
    pd.DataFrame(diagnostics).to_csv(QC_DIR/"07_comorbidities_model_diagnostics.csv",index=False)
    logger.info("[6/7] Writing auditable source map and QC")
    mapping=source_mapping(raw.columns); mapping.to_csv(QC_DIR/"07_comorbidities_source_mapping.csv",index=False)
    duplicates.to_csv(QC_DIR/"07_comorbidities_patient_duplicates.csv",index=False)
    missingness_table(base).to_csv(QC_DIR/"07_comorbidities_missingness.csv",index=False)
    pd.DataFrame(_UNRECOGNIZED,columns=["source_column","original_value","row_index"]).drop_duplicates().to_csv(QC_DIR/"07_comorbidities_unrecognized_values.csv",index=False)
    prohibited_used=sorted(x for c in iter_analysis_conditions() for x in (*c.primary,*c.detail_columns) if x.startswith(PROHIBITED_SOURCE_PREFIXES))
    qc={"input_path":str(args.input),"script_version":SCRIPT_VERSION,"run_timestamp":datetime.now(timezone.utc).isoformat(),
      "n_baseline_patients":len(base),"n_pipe_delimited_visit_dates":n_pipe,"essdai_threshold":SEVERE_THRESHOLD,
      "condition_family_counts":{f:sum(c.condition_family==f for c in iter_analysis_conditions()) for f in ("past_medical_history","rheumatologic_non_said","concomitant_said")},
      "prohibited_sources_used":prohibited_used,"prohibited_source_check_passed":not prohibited_used,
      "monte_carlo_enabled":args.run_sparse_monte_carlo,"upstream_file_timestamps":timestamps,
      "separate_burden_columns":["n_general_medical_history","n_rheumatologic_non_said","n_concomitant_said"]}
    (QC_DIR/"07_comorbidities_qc.json").write_text(json.dumps(qc,indent=2,default=str)+"\n")
    logger.info("[7/7] Complete: progression models restricted to confirmed non-SAiD rheumatologic comorbidities")
    print("Generated files:"); [print(x.resolve()) for x in produced+[QC_DIR/"07_comorbidities_qc.json",QC_DIR/"07_comorbidities_source_mapping.csv"]]
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

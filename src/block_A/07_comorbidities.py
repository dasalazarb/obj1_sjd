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

PATIENT_ID_COL = "ids__patient_record_number"
AGE_COL = "ids__age_at_visit"
SEX_COL = "ids__sex"
# The deployed analytic extract contains only this total-score field.  Keep one
# explicit source of truth rather than requiring an unavailable alias.  The
# ``*_raw_qc`` output is retained for schema compatibility and is
# therefore an identical audit copy of the primary value in this extract.
ESSDAI_PRIMARY_COL = "essdai_total"  # canonical upstream episode-level score
# Section 5 progression uses the same moderate-to-severe threshold as Pop 1.
SEVERE_THRESHOLD = config.ESSDAI_SEVERE
RANDOM_SEED = 20260728
SCRIPT_VERSION = "2.1.0"
SPARSE_EXPOSURE_THRESHOLD = 10

FIGURES_DIR = common.OUTPUTS_DIR / "figures" / "blockA" / "07_comorbidities"
TABLES_DIR = common.OUTPUTS_DIR / "tables" / "blockA" / "07_comorbidities"
QC_DIR = common.OUTPUTS_DIR / "qc" / "blockA" / "07_comorbidities"
LOG_PATH = common.OUTPUTS_DIR / "logs" / "07_comorbidities" / "07_comorbidities.log"

BASELINE_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidities" / "07_comorbidities_baseline_patient.parquet"
LONGITUDINAL_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidities" / "07_comorbidities_analysis_longitudinal.parquet"
SEVERE_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidities" / "07_comorbidity_severe5_survival.parquet"
NEW_DOMAIN_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidities" / "07_comorbidity_new_domain_survival.parquet"
DOMAIN_AUDIT_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidities" / "07_comorbidity_new_domain_patient_domain.parquet"

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
    return Condition(name, label, (source,), "general_medical", category, tuple(details),
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
    ("systemic_lupus_erythematosus", "Systemic lupus erythematosus", "sle1", "sle_hx", "sle_confirmed"),
    ("rheumatoid_arthritis", "Rheumatoid arthritis", "ra", "ra_hx", "ra_confirm"),
    ("systemic_sclerosis", "Systemic sclerosis", "systemic_sclerosis", "systmc_sclerosis_hx", "systmc_sclerosis_confirm"),
    ("polymyositis", "Polymyositis", "polymyositis", "polymyositis_hx", "polymyositis_confirm"),
    ("dermatomyositis", "Dermatomyositis", "dermatomyositis", "dermatomyositis_hx", "dermatomyositis_confirm"),
    ("mixed_connective_tissue_disease", "Mixed connective tissue disease", "mixed_connective_tissue_disease", "mixed_connect_tissue_hx", "mixed_connect_tissue_confirm"),
    ("antiphospholipid_syndrome", "Antiphospholipid syndrome", "antiphospholipid_syndrome", "antiphospholipid_syn_hx", "antiphospholipid_syn_confirm"),
]
CONCOMITANT_SAID_CONDITIONS = [_rheum(n, l, *("rheumatological_comorbidities__" + x for x in (g, h, c)), "concomitant_said", "Systemic autoimmune/inflammatory") for n, l, g, h, c in _said]

_other_immune = [
    ("primary_biliary_cholangitis", "Primary biliary cholangitis", "primary_billiary_cirrhosis", "prim_billiary_cirrhosis_hx", "prim_billiary_cirrhosis_confirm"),
    ("sarcoidosis", "Sarcoidosis", "sarcoidosis", "sarcoidosis_hx", "sarcoidosis_confirm"),
    ("inflammatory_bowel_disease", "Inflammatory bowel disease", "inflam_bowel", "inflam_bowel_hx", "inflam_bowel_confirm"),
]
OTHER_IMMUNE_MEDIATED_SYSTEMIC_CONDITIONS = [_rheum(n, l, *("rheumatological_comorbidities__" + x for x in (g, h, c)), "other_immune_mediated_systemic", "Other immune-mediated/systemic") for n, l, g, h, c in _other_immune]
RHEUMATOLOGIC_MANIFESTATIONS = [
    _rheum("raynaud", "Raynaud's phenomenon", "rheumatological_comorbidities__integ_raynds", "rheumatological_comorbidities__integ_raynds_hx", "rheumatological_comorbidities__integ_raynds_confirm", "sjd_associated_manifestation", "Rheumatologic manifestations / associated features"),
    _rheum("cryoglobulinemia", "Cryoglobulinemia", "rheumatological_comorbidities__cryoglobulinemia", "rheumatological_comorbidities__cryoglobulinemia_hx", "rheumatological_comorbidities__cryoglobulinemia_confirm", "sjd_associated_manifestation", "Rheumatologic manifestations / associated features"),
]
RHEUMATOLOGIC_ANALYSIS_CONDITIONS = [*RHEUMATOLOGIC_NON_SAID_CONDITIONS, *CONCOMITANT_SAID_CONDITIONS, *OTHER_IMMUNE_MEDIATED_SYSTEMIC_CONDITIONS]
RHEUMATOLOGIC_DESCRIPTIVE_CONDITIONS = [*RHEUMATOLOGIC_ANALYSIS_CONDITIONS, *RHEUMATOLOGIC_MANIFESTATIONS]

def iter_analysis_conditions() -> Iterable[Condition]:
    yield from PAST_MEDICAL_HISTORY_CONDITIONS
    yield from RHEUMATOLOGIC_NON_SAID_CONDITIONS
    yield from CONCOMITANT_SAID_CONDITIONS
    yield from OTHER_IMMUNE_MEDIATED_SYSTEMIC_CONDITIONS
    yield from RHEUMATOLOGIC_MANIFESTATIONS


CONDITION_NAMES = [c.name for c in iter_analysis_conditions()]
# These are presentation families only.  Every condition is passed through the
# already-prespecified Cox and longitudinal model functions; no formula,
# outcome, threshold, or adjustment set varies by family.
PROGRESSION_FAMILIES = (
    ("general_medical_comorbidities", PAST_MEDICAL_HISTORY_CONDITIONS),
    ("rheumatologic_comorbidities", RHEUMATOLOGIC_NON_SAID_CONDITIONS),
    ("concomitant_said", CONCOMITANT_SAID_CONDITIONS),
    ("other_immune_mediated_systemic", OTHER_IMMUNE_MEDIATED_SYSTEMIC_CONDITIONS),
    ("rheumatologic_manifestations", RHEUMATOLOGIC_MANIFESTATIONS),
)
PROGRESSION_CONDITIONS = list(iter_analysis_conditions())
PROGRESSION_CONDITION_NAMES = {c.name for c in PROGRESSION_CONDITIONS}
PROGRESSION_CONDITION_NAMES_ORDERED = [c.name for c in PROGRESSION_CONDITIONS]
SUBTYPE_COLS = ("past_medical_history__thyroid_disease_spfy", "rheumatological_comorbidities__inflam_bowel_spfy")
PROHIBITED_SOURCE_PREFIXES = ("sjogren's_syndrome_history__", "sjogren's_syndrome_disease_damage_index__", "systems_review_for_physician__", "ans__", "autonomic_nervous_system_questionnaire__")
ACTIVITY_THRESHOLD_SECTION5 = SEVERE_THRESHOLD
DOMAIN_COLS = ORGAN_DOMAINS
DOMAIN_EVALUABLE_COLS = DOMAIN_EVALUABLE
UNAVAILABLE: dict[str, str] = {}

UPSTREAM = {
    common.CLINICAL_VISIT_SPINE_PARQUET: "src/00_build_visit_spine.py",
    common.POP_LONGITUDINAL_PARQUET: "src/block_A/01_pop_distribution.py",
    common.OVERLAP_LONGITUDINAL_PARQUET: "src/block_A/06_overlap_glandular.py",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=common.SOURCE_EPISODE_SPINE)
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
    """Load the authoritative clinical-episode spine without rebuilding time."""
    cols = ["patient_id", "clinical_episode_id", "clinical_anchor_date", "clinical_visit",
            "clinical_visit_number", "clinical_baseline_episode_id", "clinical_baseline_date",
            "is_clinical_baseline", "time_since_clinical_baseline_days",
            "time_since_clinical_baseline_years"]
    available = available_columns(common.CLINICAL_VISIT_SPINE_PARQUET)
    optional = [c for c in ("age_at_visit", "sex") if c in available]
    df = _read_required(common.CLINICAL_VISIT_SPINE_PARQUET, cols + optional)
    df = df.loc[df["clinical_visit"].eq(True)].copy()
    for col in ("clinical_anchor_date", "clinical_baseline_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    if "age_at_visit" in df:
        df["age_at_visit"] = parse_age_first_value(df["age_at_visit"])
    keys = ["patient_id", "clinical_episode_id"]
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        raise ValueError("Clinical episode spine violates canonical identity")
    baseline = df.loc[df["is_clinical_baseline"].eq(True)]
    if baseline["patient_id"].duplicated().any():
        raise ValueError("Multiple clinical baselines for a patient")
    if not baseline["clinical_episode_id"].eq(baseline["clinical_baseline_episode_id"]).all():
        raise ValueError("Clinical baseline episode mismatch")
    if not baseline["clinical_anchor_date"].eq(baseline["clinical_baseline_date"]).all():
        raise ValueError("Clinical baseline date mismatch")
    return df


def load_pop_classification() -> pd.DataFrame:
    cols = ["patient_id", "clinical_episode_id", "essdai_total", "pop_status"]
    available = available_columns(common.POP_LONGITUDINAL_PARQUET)
    optional = [c for c in ("baseline_pop_status", "pop_baseline_status") if c in available]
    return _read_required(common.POP_LONGITUDINAL_PARQUET, cols + optional)


def load_domain_flags() -> pd.DataFrame:
    cols = ["patient_id", "clinical_episode_id"] + ALL_DOMAINS + list(DOMAIN_EVALUABLE.values())
    available = available_columns(common.OVERLAP_LONGITUDINAL_PARQUET)
    if "n_extraglandular_domains_active" in available:
        cols.append("n_extraglandular_domains_active")
    df = _read_required(common.OVERLAP_LONGITUDINAL_PARQUET, cols)
    for col in ALL_DOMAINS + list(DOMAIN_EVALUABLE.values()):
        df[col] = normalize_binary_flag(df[col])
    return df


def selected_raw_columns(input_path: Path) -> list[str]:
    schema = available_columns(input_path)
    desired = ["patient_id", "clinical_episode_id", "is_clinical_baseline", AGE_COL, SEX_COL, *SUBTYPE_COLS]
    desired.extend(col for c in iter_analysis_conditions() for col in (*c.primary, *c.detail_columns))
    prohibited = [c for c in desired if c.startswith(PROHIBITED_SOURCE_PREFIXES)]
    if prohibited:
        raise AssertionError(f"Prohibited Section 5 source(s): {prohibited}")
    required = {"patient_id", "clinical_episode_id", "is_clinical_baseline"}
    if required - schema:
        raise KeyError(f"Canonical source lacks required columns: {sorted(required - schema)}")
    return list(dict.fromkeys(c for c in desired if c in schema))


def load_selected_raw_columns(input_path: Path) -> pd.DataFrame:
    return pd.read_parquet(input_path, columns=selected_raw_columns(input_path))


_UNRECOGNIZED: list[dict[str, Any]] = []


def normalize_binary_flag(series: pd.Series) -> pd.Series:
    """Map common binary encodings to pandas nullable Boolean values."""
    positive = {"1", "1.0", "true", "yes", "y", "positive", "present", "confirmed", "history", "on"}
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
    for condition in RHEUMATOLOGIC_DESCRIPTIVE_CONDITIONS:
        general, history, confirmed = [_source_flag(out, col) for col in condition.primary]
        status = derive_condition_status(general, history, confirmed)
        out[f"{condition.name}_status"] = status
        exposed = (general.fillna(False) | history.fillna(False) |
                   confirmed.fillna(False)).astype("boolean")
        out[condition.name] = exposed
        out[f"{condition.name}_primary_exposure"] = exposed.astype(float)
    return out


def apply_exposure_definition(frame: pd.DataFrame, condition: Condition,
                              definition: str = "any_source_positive") -> pd.DataFrame:
    if definition != "any_source_positive":
        raise ValueError(f"Unsupported exposure definition: {definition}")
    out = frame.copy()
    out["exposure"] = out[condition.name].fillna(False).astype(int)
    return out

def build_baseline_comorbidity_dataset(raw: pd.DataFrame, spine: pd.DataFrame, pop: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Derive exposures once, exclusively on authoritative baseline episodes."""
    keys = ["patient_id", "clinical_episode_id"]
    raw_base = raw.loc[raw["is_clinical_baseline"].eq(True)].copy()
    if raw_base.duplicated(keys).any():
        raise ValueError("Raw canonical input has duplicate baseline episodes")
    derived = derive_comorbidity_indicators(raw_base)
    base_spine = spine.loc[spine["is_clinical_baseline"].eq(True)].copy()
    base = base_spine.merge(derived.drop(columns=["is_clinical_baseline"], errors="ignore"), on=keys,
                            how="left", validate="one_to_one")
    pop_cols = pop[[*keys, "pop_status", "essdai_total"]].drop_duplicates(keys)
    base = base.merge(pop_cols, on=keys, how="left", validate="one_to_one")
    base = base.rename(columns={"clinical_episode_id": "clinical_baseline_episode_id_source",
        "clinical_anchor_date": "clinical_baseline_date_source", "pop_status": "baseline_pop",
        "essdai_total": "baseline_essdai", "age_at_visit": "age_baseline"})
    # Preserve required canonical names and assert their equality to the baseline row.
    base["clinical_baseline_episode_id"] = base["clinical_baseline_episode_id_source"]
    base["clinical_baseline_date"] = base["clinical_baseline_date_source"]
    if AGE_COL in base and "age_baseline" not in base: base["age_baseline"] = parse_age_first_value(base[AGE_COL])
    if SEX_COL in base and "sex" not in base: base["sex"] = base[SEX_COL]
    pmh = [c.name for c in PAST_MEDICAL_HISTORY_CONDITIONS]
    for name in [c.name for c in iter_analysis_conditions()]:
        if name not in base: base[name] = pd.Series(False, index=base.index, dtype="boolean")
        base[name] = base[name].fillna(False).astype("boolean")
    families = {
      "general_medical": pmh,
      "rheumatologic_non_said": [c.name for c in RHEUMATOLOGIC_NON_SAID_CONDITIONS],
      "concomitant_said": [c.name for c in CONCOMITANT_SAID_CONDITIONS],
      "other_immune_mediated_systemic": [c.name for c in OTHER_IMMUNE_MEDIATED_SYSTEMIC_CONDITIONS],
      "sjd_associated_manifestations": [c.name for c in RHEUMATOLOGIC_MANIFESTATIONS]}
    for family, names in families.items():
        base[f"n_{family}"] = base[names].astype(int).sum(axis=1)
        base[f"any_{family}"] = base[f"n_{family}"].gt(0).astype("boolean")
    base["concomitant_said_any"] = base["any_concomitant_said"]
    base["any_other_immune_mediated_systemic"] = base["any_other_immune_mediated_systemic"]
    base["any_sjd_associated_manifestation"] = base["any_sjd_associated_manifestations"]
    all_names = [c.name for c in iter_analysis_conditions()]
    base["n_total_conditions"] = base[all_names].astype(int).sum(axis=1)
    base["any_condition"] = base["n_total_conditions"].gt(0).astype("boolean")
    if base["patient_id"].duplicated().any(): raise ValueError("Baseline is not one row per patient")
    return base, pd.DataFrame(columns=[*keys, "n_source_rows"]), 0


def summarize_historical_family(base: pd.DataFrame, conditions: Sequence[Condition],
                                family_label: str = "general_medical") -> pd.DataFrame:
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
        positive=base.get(c.name, pd.Series(False, index=base.index)).fillna(False).eq(True)
        n_positive=int(positive.sum())
        rows.append({"condition":c.name,"display_label":c.label,"condition_family":c.condition_family,
          "clinical_category":c.clinical_category,"n_confirmed_present":int(counts.get("confirmed_present",0)),
          "n_history_only":int(counts.get("history_only",0)),"n_status_uncertain":int(counts.get("status_uncertain",0)),
          "n_no_comorbidity":int(counts.get("no_comorbidity",0)),"N_evaluable":N,
          "n_positive_any_yes":n_positive,
          "pct_positive_any_yes":100*n_positive/N if N else np.nan,
          "pct_confirmed":100*int(counts.get("confirmed_present",0))/N if N else np.nan})
    return pd.DataFrame(rows)


def summarize_overall_prevalence(base: pd.DataFrame) -> pd.DataFrame:
    """Summarize primary any-source-positive rheumatologic prevalence."""
    out=summarize_confirmed_family(base,RHEUMATOLOGIC_ANALYSIS_CONDITIONS)
    return out.rename(columns={"N_evaluable":"n_evaluable_primary","pct_confirmed":"pct_confirmed_among_evaluable"}).assign(
      n_total_cohort=len(base),n_missing=0,
      pct_positive_total_cohort=lambda x:100*x.n_positive_any_yes/len(base) if len(base) else np.nan)


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
    return summarize_family_by_pop(base,RHEUMATOLOGIC_ANALYSIS_CONDITIONS,run_sparse_monte_carlo=run_sparse_monte_carlo,replicates=replicates,seed=seed)


def apply_fdr(p_values: pd.Series) -> pd.Series:
    out=pd.Series(np.nan,index=p_values.index,dtype=float); valid=p_values.notna()
    if valid.any(): out.loc[valid]=multipletests(p_values.loc[valid],method="fdr_bh")[1]
    return out

def build_longitudinal_essdai_dataset(raw: pd.DataFrame, spine: pd.DataFrame, pop: pd.DataFrame, domains: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Merge validated upstream outcomes by canonical episode identity."""
    keys = ["patient_id", "clinical_episode_id"]
    popcols = pop[[*keys, "essdai_total", "pop_status"]].drop_duplicates(keys)
    long = spine.merge(popcols, on=keys, how="left", validate="one_to_one")
    long = long.merge(domains, on=keys, how="left", validate="one_to_one")
    status_cols = [f"{c.name}_status" for c in RHEUMATOLOGIC_DESCRIPTIVE_CONDITIONS if f"{c.name}_status" in base]
    burden_cols = [c for c in base if c.startswith(("n_", "any_")) or c == "concomitant_said_any"]
    bcols = ["patient_id", "baseline_essdai", "baseline_pop", "age_baseline", "sex",
             *[c.name for c in iter_analysis_conditions()], *status_cols, *burden_cols]
    bcols = list(dict.fromkeys(c for c in bcols if c in base))
    long = long.drop(columns=["sex"], errors="ignore").merge(base[bcols], on="patient_id", how="left", validate="many_to_one")
    if long.duplicated(keys).any(): raise ValueError("Duplicate canonical patient episode")
    valid = long["essdai_total"].dropna()
    if not valid.between(0, 123).all(): raise ValueError("Canonical ESSDAI outside 0..123")
    return long


def build_severe5_survival_dataset(long: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for pid,g in long.groupby("patient_id"):
        b=base.loc[base.patient_id.eq(pid)].iloc[0]
        follow=g.loc[(g.clinical_anchor_date > b.clinical_baseline_date) & g.essdai_total.notna()].sort_values("clinical_anchor_date")
        if pd.isna(b.baseline_essdai) or b.baseline_essdai >= SEVERE_THRESHOLD or follow.empty: continue
        events=follow.loc[follow.essdai_total >= SEVERE_THRESHOLD]
        event_date=events.clinical_anchor_date.iloc[0] if len(events) else pd.NaT
        last=follow.clinical_anchor_date.max(); end=event_date if pd.notna(event_date) else last
        row=b.to_dict(); row.update(last_evaluable_date=last,event_date=event_date,
          followup_days=(end-b.clinical_baseline_date).days,severe5_event=int(pd.notna(event_date)))
        rows.append(row)
    out=pd.DataFrame(rows); out["followup_years"]=out.get("followup_days",pd.Series(dtype=float))/365.25
    if len(out) and (out.followup_days < 0).any(): raise ValueError("Negative severe-event follow-up")
    return out


def build_new_domain_survival_dataset(long: pd.DataFrame, base: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    patient_rows=[]; domain_rows=[]
    for pid,g in long.sort_values("clinical_anchor_date").groupby("patient_id"):
        baseline=g.loc[g.is_clinical_baseline.eq(True)]
        if len(baseline)!=1: continue
        b=baseline.iloc[0]; follow=g.loc[g.clinical_anchor_date > b.clinical_baseline_date]
        event_candidates=[]; censor_dates=[]; n_risk=0
        for domain in ALL_DOMAINS:
            evcol=DOMAIN_EVALUABLE[domain]
            baseline_eval=bool(b[evcol]) if pd.notna(b[evcol]) else False
            baseline_active=bool(b[domain]) if baseline_eval else pd.NA
            eval_follow=follow.loc[follow[evcol].eq(True)] if baseline_eval and not bool(baseline_active) else follow.iloc[:0]
            at_risk=bool(baseline_eval and not bool(baseline_active) and len(eval_follow))
            events=eval_follow.loc[eval_follow[domain].eq(True),"clinical_anchor_date"]
            event_date=events.min() if len(events) else pd.NaT
            censor=eval_follow.clinical_anchor_date.max() if len(eval_follow) else pd.NaT
            record={"patient_id":pid,"domain":domain,"baseline_evaluable":baseline_eval,
              "baseline_active":baseline_active,"at_risk":at_risk,"domain_event":int(pd.notna(event_date)),
              "domain_event_date":event_date,"last_domain_evaluable_date":censor}
            if at_risk:
                end=event_date if pd.notna(event_date) else censor
                record["followup_days"]=(end-b.clinical_baseline_date).days
                n_risk+=1; censor_dates.append(censor)
                if pd.notna(event_date): event_candidates.append((event_date,domain))
            domain_rows.append(record)
        if not n_risk: continue
        first_date,first_domain=min(event_candidates) if event_candidates else (pd.NaT,pd.NA)
        last=max(censor_dates); end=first_date if pd.notna(first_date) else last
        row=base.loc[base.patient_id.eq(pid)].iloc[0].to_dict()
        row.update(first_new_domain_date=first_date,first_new_domain_name=first_domain,
          new_domain_event=int(pd.notna(first_date)),n_domains_inactive_at_baseline=n_risk,
          last_evaluable_date=last,followup_days=(end-b.clinical_baseline_date).days)
        patient_rows.append(row)
    out=pd.DataFrame(patient_rows); out["followup_years"]=out.get("followup_days",pd.Series(dtype=float))/365.25
    audit=pd.DataFrame(domain_rows)
    if len(out) and (out.followup_days < 0).any(): raise ValueError("Negative domain follow-up")
    return out,audit


def _empty_progression(c: Condition, outcome: str, estimand: str, warning: str, **counts: Any) -> dict[str, Any]:
    n_patients = counts.get("n_patients", 0)
    n_exposed = counts.get("n_exposed", np.nan)
    n_unexposed = n_patients - n_exposed if pd.notna(n_exposed) else np.nan
    sparse = bool(n_exposed < SPARSE_EXPOSURE_THRESHOLD) if pd.notna(n_exposed) else False
    return {
        "condition": c.name, "comorbidity": c.name, "display_label": c.label,
        "condition_family": c.condition_family, "outcome": outcome, "estimand": estimand,
        "model_type": "not fitted", "model_attempted": "LME" if outcome == "Longitudinal ESSDAI trajectory" else "Cox PH",
        "model_used": "not_fitted", "effect_measure": "Not estimable", "estimate": np.nan,
        "ci95_low": np.nan, "ci95_high": np.nan, "p_value": np.nan,
        "n_patients": n_patients, "n_observations": counts.get("n_followup_observations", 0),
        "n_followup_observations": counts.get("n_followup_observations", 0),
        "n_at_risk": n_patients, "n_events": counts.get("n_events", np.nan),
        "n_exposed": n_exposed, "n_unexposed": n_unexposed,
        "n_exposed_events": counts.get("n_exposed_events", np.nan),
        "n_complete_cases": counts.get("n_complete_cases", 0),
        "baseline_reference_group": "Comorbidity absent",
        "adjustment_covariates": "baseline ESSDAI; baseline Pop; age; sex",
        "time_scale": "years since clinical baseline",
        "threshold": SEVERE_THRESHOLD if outcome == "Progression to ESSDAI >=5" else np.nan,
        "model_converged": False, "converged": False, "singular_fit": False,
        "fallback_used": False, "proportional_hazards_p": np.nan,
        "ph_assumption_p_value": np.nan, "ph_assumption_status": "not_estimable",
        "sparse_event_flag": True, "sparse_exposure": sparse, "model_status": "model_failed",
        "model_warning": warning, "warning": warning, "result_interpretability": "not_estimable",
        "primary_interpretation_flag": "not_estimable",
        "interpretation": "Not estimable; no causal interpretation is warranted."
    }


def restrict_to_primary_exposure(data: pd.DataFrame, c: Condition) -> pd.DataFrame:
    """Use the canonical any-source exposure, treating blank as unexposed."""
    if c.name not in data:
        raise KeyError(f"Primary model input lacks {c.name}")
    out = data.copy()
    out[c.name] = out[c.name].fillna(False).astype(int)
    return out


def progression_exposure_column(c: Condition) -> str:
    """Return the baseline exposure column required by progression models."""
    return c.name


def fit_mixed_model(long: pd.DataFrame, c: Condition) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import statsmodels.formula.api as smf
    cols = ["patient_id", "essdai_total", "time_since_clinical_baseline_years",
            c.name, "baseline_essdai", "baseline_pop", "age_baseline", "sex"]
    data = restrict_to_primary_exposure(
        long.loc[long["time_since_clinical_baseline_days"] > 0, cols], c
    ).dropna(subset=["patient_id", "essdai_total", "time_since_clinical_baseline_years",
                     "baseline_essdai", "baseline_pop", "age_baseline", "sex"])
    patients = data.drop_duplicates("patient_id")
    n = len(patients); n_exposed = int(patients[c.name].sum())
    counts = {"n_patients": n, "n_followup_observations": len(data),
              "n_complete_cases": n, "n_exposed": n_exposed}
    sparse = n_exposed < SPARSE_EXPOSURE_THRESHOLD
    if n < 5 or data[c.name].nunique() < 2 or n_exposed < 5:
        status = "no_variation" if data[c.name].nunique() < 2 else "insufficient_exposure"
        warning = "Too few complete patients or insufficient exposure variation"
        row = _empty_progression(c, "Longitudinal ESSDAI trajectory",
                                 "annual_slope_difference", warning, **counts)
        row["model_status"] = status
        return [row], _model_qc_row(row)

    formula = "essdai_total ~ time_since_clinical_baseline_years * Q('%s') + baseline_essdai + C(baseline_pop) + age_baseline + C(sex)" % c.name
    fit = None; model_used = "LME"; singular = False; fallback = False; warning_text = ""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = smf.mixedlm(formula, data, groups=data["patient_id"]).fit(reml=False, method="lbfgs")
        caught_text = "; ".join(str(w.message) for w in caught)
        variance = float(fit.cov_re.iloc[0, 0])
        singular = variance < 1e-8 or "singular" in caught_text.lower()
        if not fit.converged or singular:
            detail = "Random effects covariance is singular" if singular else "LME did not converge"
            raise RuntimeError(detail)
        warning_text = caught_text
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        fallback = True; model_used = "GEE_fallback"
        warning_text = f"Mixed model failed ({exc}); GEE fallback used"
        try:
            fit = GEE.from_formula(formula, groups="patient_id", data=data,
                                   family=Gaussian(), cov_struct=Exchangeable()).fit()
        except (ValueError, np.linalg.LinAlgError) as gee_exc:
            row = _empty_progression(c, "Longitudinal ESSDAI trajectory",
                                     "annual_slope_difference",
                                     f"Mixed model and GEE failed: {gee_exc}", **counts)
            row.update({"singular_fit": singular, "fallback_used": True})
            return [row], _model_qc_row(row)

    interpretability = ("caution_sparse_and_fallback" if sparse and fallback else
                        "caution_sparse_exposure" if sparse else
                        "caution_gee_fallback" if fallback else "standard")
    flag = "exploratory_only" if sparse or fallback else "eligible"
    status = "lme_failed_gee_used" if fallback else "ok"
    model_type = "Gaussian GEE (exchangeable)" if fallback else "Linear mixed model (random intercept)"
    terms = ((f"Q('{c.name}')", "mean_difference_followup", "Adjusted mean difference"),
             (f"time_since_clinical_baseline_years:Q('{c.name}')", "annual_slope_difference", "Beta per year"))
    rows = []
    for term, estimand, measure in terms:
        est, se, p_value = float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term])
        row = _empty_progression(c, "Longitudinal ESSDAI trajectory", estimand, warning_text, **counts)
        row.update({"model_type": model_type, "model_used": model_used, "effect_measure": measure,
                    "estimate": est, "ci95_low": est - 1.96 * se, "ci95_high": est + 1.96 * se,
                    "p_value": p_value, "model_converged": True, "converged": True,
                    "singular_fit": singular, "fallback_used": fallback, "sparse_event_flag": False,
                    "model_status": status, "result_interpretability": interpretability,
                    "primary_interpretation_flag": flag,
                    "interpretation": "Exploratory / unstable estimate." if flag == "exploratory_only" else "Adjusted association estimate; not a causal effect."})
        rows.append(row)
    return rows, _model_qc_row(rows[0])


def _model_qc_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return one normalized model-attempt audit row."""
    return {key: row.get(key) for key in (
        "condition", "outcome", "model_attempted", "model_used", "converged",
        "singular_fit", "fallback_used", "n_exposed", "n_events",
        "sparse_exposure", "model_status", "model_warning"
    )}


def finalize_model_metadata(results: pd.DataFrame) -> pd.DataFrame:
    """Standardize machine-readable QC and interpretation fields."""
    out = results.copy()
    fitted = out["model_converged"].fillna(False).astype(bool)
    cox = out["outcome"].ne("Longitudinal ESSDAI trajectory")
    out.loc[cox & fitted, "model_used"] = "Cox_PH"
    out.loc[cox, "model_attempted"] = "Cox PH"
    out["converged"] = fitted
    out["fallback_used"] = out["model_used"].eq("GEE_fallback")
    out["sparse_exposure"] = out["n_exposed"].lt(SPARSE_EXPOSURE_THRESHOLD)
    out["n_unexposed"] = out["n_patients"] - out["n_exposed"]
    out["n_at_risk"] = out["n_patients"]
    out["n_observations"] = out["n_followup_observations"]
    out["model_warning"] = out["warning"].fillna("")
    ph_p = out["proportional_hazards_p"]
    out["ph_assumption_p_value"] = ph_p
    out["ph_assumption_status"] = np.where(~cox | ~fitted, "not_estimable",
                                            np.where(ph_p.isna(), "not_tested",
                                                     np.where(ph_p < .05, "warning", "pass")))
    # Sparse exposure and PH warnings never suppress estimates, but prevent
    # their promotion as primary, robust findings.
    caution = out["sparse_exposure"] | out["fallback_used"] | out["ph_assumption_status"].eq("warning")
    out["primary_interpretation_flag"] = np.where(~fitted, "not_estimable",
                                                   np.where(caution, "exploratory_only", "eligible"))
    out.loc[~fitted, "result_interpretability"] = "not_estimable"
    out.loc[fitted & out["sparse_exposure"] & out["fallback_used"], "result_interpretability"] = "caution_sparse_and_fallback"
    out.loc[fitted & out["sparse_exposure"] & ~out["fallback_used"], "result_interpretability"] = "caution_sparse_exposure"
    out.loc[fitted & ~out["sparse_exposure"] & out["fallback_used"], "result_interpretability"] = "caution_gee_fallback"
    out.loc[fitted & ~out["sparse_exposure"] & ~out["fallback_used"], "result_interpretability"] = "standard"
    out["HR"] = np.where(cox & fitted, out["estimate"], np.nan)
    out["CI95_low"] = out["ci95_low"]
    out["CI95_high"] = out["ci95_high"]
    out["model_adjustment"] = out["model_type"]
    return out


def fit_cox_model(data: pd.DataFrame, c: Condition, event_col: str, outcome: str, minimum_events: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the existing Cox specification with a family-appropriate exposure."""
    exposure_col = progression_exposure_column(c)
    cols = ["followup_years", event_col, exposure_col, "baseline_essdai",
            "baseline_pop", "age_baseline", "sex"]
    d = restrict_to_primary_exposure(data[cols], c)
    n_exposure_excluded = 0
    d = d.dropna().copy()
    counts = {"n_patients": len(d), "n_events": int(d[event_col].sum()) if len(d) else 0,
              "n_complete_cases": len(d), "n_exposed": int(d[c.name].sum()) if len(d) else 0,
              "n_exposed_events": int(d.loc[d[c.name].eq(1), event_col].sum()) if len(d) else 0}
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
        row = _empty_progression(c, outcome, "Any source positive vs none positive", "lifelines is not installed; Cox model not executed", **counts)
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
        row = {**_empty_progression(c, outcome, "Any source positive vs none positive", warning_text, **counts), "model_type": model_type, "effect_measure": "Hazard ratio", "estimate": float(summary["exp(coef)"]), "ci95_low": float(summary["exp(coef) lower 95%"]), "ci95_high": float(summary["exp(coef) upper 95%"]), "p_value": float(summary["p"]), "model_converged": True, "proportional_hazards_p": ph_p, "sparse_event_flag": reduced, "model_status": "reduced_adjustment" if reduced else "fitted", "baseline_reference_group": "No positive source", "n_ambiguous_status_excluded": n_exposure_excluded, "interpretation": "Adjusted any-source-positive versus no-positive-source association; this is not a causal effect."}
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


def create_progression_forestplot(
    progression: pd.DataFrame,
    sections: Sequence[tuple[str, Sequence[Condition]]],
    path: Path,
) -> None:
    """Create one forest plot with clinical group headers and separators."""
    panels = [("Longitudinal ESSDAI trajectory", "Difference in annual ESSDAI slope", 0, False), ("Progression to ESSDAI >=5", "Progression to ESSDAI ≥5", 1, True), ("New ESSDAI-domain involvement", "Development of new ESSDAI-domain involvement", 1, True)]
    plot_rows: list[tuple[str, Condition | None]] = []
    for section, conditions in sections:
        plot_rows.append((section, None))
        plot_rows.extend((condition.label, condition) for condition in conditions)
    plot_rows = plot_rows[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(20, max(8, .42*len(plot_rows))), sharey=True)
    for ax, (outcome, title, null, logscale) in zip(axes, panels):
        d = progression[(progression["outcome"] == outcome) & ((progression["effect_measure"] == "Beta per year") if "trajectory" in outcome else True)].set_index("comorbidity")
        for y, (_, condition) in enumerate(plot_rows):
            if condition is None:
                ax.axhline(y - .5, color="#bdbdbd", lw=.8, zorder=0)
                continue
            name = condition.name
            if name in d.index and np.isfinite(d.loc[name, ["estimate", "ci95_low", "ci95_high"]].astype(float)).all():
                r=d.loc[name]; errors = _nonnegative_interval_errors([r.estimate], [r.ci95_low], [r.ci95_high])
                ax.errorbar(r.estimate, y, xerr=errors, fmt="o", color="#2166ac" if r.model_status=="fitted" else "#b2182b", capsize=2)
                counts = (f"exposed={int(r.n_exposed)}" if pd.notna(r.n_exposed)
                          else f"n={int(r.n_patients)}")
                if pd.notna(r.n_exposed_events):
                    counts += f", exposed events={int(r.n_exposed_events)}"
                if pd.notna(r.get("fdr_bh_q_value", np.nan)):
                    counts += f", q={float(r.fdr_bh_q_value):.3g}"
                elif pd.notna(r.p_value):
                    counts += f", p={float(r.p_value):.3g}"
                ax.annotate(counts, (r.estimate,y), xytext=(5,4), textcoords="offset points", fontsize=6)
            else: ax.plot(null, y, marker="x", color="gray")
        ax.axvline(null, color="black", ls="--", lw=.8); ax.set_title(title); ax.grid(axis="x", alpha=.2)
        if logscale: ax.set_xscale("log"); ax.set_xlabel("Hazard ratio (log scale)")
        else: ax.set_xlabel("Adjusted beta per year")
    axes[0].set_yticks(range(len(plot_rows)), [label for label, _ in plot_rows])
    for tick, (_, condition) in zip(axes[0].get_yticklabels(), plot_rows):
        if condition is None:
            tick.set_fontweight("bold"); tick.set_color("#333333")
    fig.text(.01,.01,"Rheumatologic exposure is general OR history OR confirmed, with blank source fields treated as false. Models adjust for baseline ESSDAI, baseline Pop, age, and sex when support permits. X marks not estimable; red denotes reduced models. Associations are not causal.",fontsize=8)
    fig.subplots_adjust(left=.24,bottom=.08,wspace=.12); _plot_save(fig, path)


def run_qc_checks(base: pd.DataFrame, long: pd.DataFrame, severe: pd.DataFrame, new_domain: pd.DataFrame, domain_audit: pd.DataFrame) -> dict[str, Any]:
    if base["patient_id"].isna().any() or base["patient_id"].duplicated().any(): raise ValueError("Invalid baseline patient identity")
    if not base["clinical_baseline_episode_id_source"].eq(base["clinical_baseline_episode_id"]).all(): raise ValueError("Baseline episode mismatch")
    if not base["clinical_baseline_date_source"].eq(base["clinical_baseline_date"]).all(): raise ValueError("Baseline date mismatch")
    if long.duplicated(["patient_id", "clinical_episode_id"]).any(): raise ValueError("Duplicate patient episode")
    for col in CONDITION_NAMES + ALL_DOMAINS + list(DOMAIN_EVALUABLE.values()):
        source = base[col] if col in base else long[col]
        if str(source.dtype) != "boolean": raise TypeError(f"{col} is not nullable boolean")
    if len(domain_audit):
        if ((domain_audit["baseline_active"] == True) & domain_audit["domain_event_date"].notna()).any(): raise ValueError("Baseline-active domain counted as new")  # noqa: E712
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
    for c in list(iter_analysis_conditions()):
        prohibited=[x for x in (*c.primary,*c.detail_columns) if x.startswith(PROHIBITED_SOURCE_PREFIXES)]
        if prohibited: raise AssertionError(f"Prohibited source selected: {prohibited}")
        rows.append({"condition":c.name,"condition_family":c.condition_family,
          "clinical_category":c.clinical_category,"source_columns":"|".join(c.primary),
          "status_definition":("documented_history=True only when PMH source is positive; blank means no documented history"
             if c.condition_family=="general_medical" else
             "primary exposure = general OR history OR confirmed; detailed status retained for audit"),
          "availability":"available" if any(x in raw_columns for x in c.primary) else "unavailable"})
    return pd.DataFrame(rows)


def create_family_plot(table: pd.DataFrame, path: Path, historical: bool=False) -> None:
    value="percent_documented_total_cohort" if historical else "pct_confirmed"
    d=table.sort_values(value); fig,ax=plt.subplots(figsize=(10,max(5,.32*len(d))))
    ax.barh(np.arange(len(d)),d[value],color="#2c7fb8"); ax.set_yticks(np.arange(len(d)),d.display_label)
    ax.set_xlabel("Patients with documented history (%)" if historical else "Patients with confirmed condition (%)")
    ax.grid(axis="x",alpha=.25); _plot_save(fig,path)


def create_sectioned_comorbidity_plot(base: pd.DataFrame,
                                      sections: Sequence[tuple[str, Sequence[Condition]]],
                                      path: Path, xlabel: str) -> None:
    """Plot one condition figure with labeled clinical sections and separators."""
    rows: list[tuple[str, Condition | None, float]] = []
    for section, conditions in sections:
        rows.append((section, None, np.nan))
        for condition in conditions:
            prevalence = 100 * base[condition.name].fillna(False).astype(bool).mean() if len(base) else np.nan
            rows.append((condition.label, condition, prevalence))
    fig, ax = plt.subplots(figsize=(11, max(7, .34 * len(rows))))
    y = np.arange(len(rows))[::-1]
    for ypos, (label, condition, prevalence) in zip(y, rows):
        if condition is None:
            ax.text(0, ypos, label, va="center", ha="left", fontweight="bold",
                    transform=ax.get_yaxis_transform())
            ax.axhline(ypos - .5, color="#bdbdbd", lw=.7)
        else:
            ax.barh(ypos, prevalence, color="#2c7fb8", height=.72)
    ax.set_yticks([pos for pos, (_, condition, _) in zip(y, rows) if condition is not None],
                  [label for label, condition, _ in rows if condition is not None])
    ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.25)
    fig.subplots_adjust(left=.35, bottom=.08)
    _plot_save(fig, path)


def condition_dictionary() -> pd.DataFrame:
    rows=[]
    for c in iter_analysis_conditions():
        role = {"general_medical":"general_medical_comorbidity",
                "other_immune_mediated_systemic":"other_immune_mediated"}.get(c.condition_family,c.condition_family)
        rows.append({"condition":c.name,"display_label":c.label,"condition_family":c.condition_family,
          "clinical_category":c.clinical_category,"source_columns":"|".join(c.primary),
          "derivation_rule":("positive checkbox = documented history; blank = no documented history" if c.condition_family=="general_medical" else "general OR history OR confirmed"),
          "analysis_role":role})
    return pd.DataFrame(rows)


def burden_summary(base: pd.DataFrame) -> pd.DataFrame:
    columns=["n_general_medical","n_rheumatologic_non_said","n_concomitant_said",
      "n_other_immune_mediated_systemic","n_sjd_associated_manifestations","n_total_conditions"]
    rows=[]
    groups=[("Overall",base)]+[(p,base.loc[base.baseline_pop.eq(p)]) for p in ("Pop1","Pop2","Pop3")]
    for group,data in groups:
        for col in columns:
            x=pd.to_numeric(data[col],errors="coerce").dropna()
            rows.append({"group":group,"burden":col,"N":len(x),"mean":x.mean(),"SD":x.std(),
              "median":x.median(),"IQR":x.quantile(.75)-x.quantile(.25),"min":x.min(),"max":x.max()})
    return pd.DataFrame(rows)


def source_qc(raw: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for c in iter_analysis_conditions():
        for col in c.primary:
            exists=col in raw
            normalized=normalize_binary_flag(raw[col]) if exists else pd.Series(dtype="boolean")
            unrec=[x for x in _UNRECOGNIZED if x["source_column"]==col]
            rows.append({"condition":c.name,"condition_family":c.condition_family,"source_column":col,
              "source_exists":exists,"n_nonmissing":int(raw[col].notna().sum()) if exists else 0,
              "n_yes":int(normalized.eq(True).sum()),"n_no":int(normalized.eq(False).sum()),
              "n_unrecognized":len(unrec),"unique_unrecognized_values":"|".join(sorted({x["original_value"] for x in unrec}))})
    return pd.DataFrame(rows)


def condition_prevalence_qc(base: pd.DataFrame) -> pd.DataFrame:
    """Reconcile the one canonical exposure over the full and Pop subsets."""
    classified = base["baseline_pop"].isin(("Pop1", "Pop2", "Pop3"))
    rows = []
    for condition in iter_analysis_conditions():
        positive = base[condition.name].fillna(False).astype(bool)
        pop_counts = [int((positive & base["baseline_pop"].eq(pop)).sum())
                      for pop in ("Pop1", "Pop2", "Pop3")]
        n_classified = int((positive & classified).sum())
        pop_sum = sum(pop_counts)
        rows.append({
            "condition": condition.name,
            "condition_family": condition.condition_family,
            "n_positive_full_cohort": int(positive.sum()),
            "n_positive_pop_classified": n_classified,
            "n_pop1": pop_counts[0], "n_pop2": pop_counts[1], "n_pop3": pop_counts[2],
            "sum_positive_pop": pop_sum,
            "overall_pop_consistency_flag": n_classified == pop_sum,
        })
    return pd.DataFrame(rows)

def create_standard_figures(overall: pd.DataFrame, by_pop: pd.DataFrame, burden: pd.DataFrame) -> None:
    """Create requested figures exclusively from their published tables."""
    d=overall.sort_values("pct")
    fig,ax=plt.subplots(figsize=(10,max(6,.25*len(d))))
    ax.scatter(d.pct,np.arange(len(d))); ax.set_yticks(np.arange(len(d)),d.display_label)
    ax.set_xlabel("Baseline prevalence (%)"); ax.grid(axis="x",alpha=.25)
    _plot_save(fig,FIGURES_DIR/"07_comorbidities_dotplot.pdf")
    x=np.arange(len(by_pop)); fig,ax=plt.subplots(figsize=(max(12,.22*len(x)),7))
    for i,pop in enumerate((1,2,3)): ax.bar(x+(i-1)*.25,by_pop[f"pct_pop{pop}"],.25,label=f"Pop{pop}")
    ax.set_xticks(x,by_pop.display_label,rotation=90); ax.set_ylabel("Baseline prevalence (%)"); ax.legend()
    _plot_save(fig,FIGURES_DIR/"07_comorbidities_by_pop_grouped_bar.pdf")
    b=burden.loc[burden.group.eq("Overall")]
    fig,ax=plt.subplots(figsize=(10,5)); ax.bar(b.burden,b["mean"],yerr=b.SD.fillna(0))
    ax.tick_params(axis="x",rotation=35); ax.set_ylabel("Mean baseline condition count")
    _plot_save(fig,FIGURES_DIR/"07_comorbidity_burden_by_family.pdf")

def main(argv: Sequence[str] | None = None) -> int:
    args=parse_args(argv); ensure_directories(); logger=setup_logging(); np.random.seed(args.random_seed)
    _UNRECOGNIZED.clear()
    logger.info("[1/7] Loading canonical sources")
    timestamps=check_upstream_artifacts(args.input,args.rebuild_upstream,logger)
    spine=load_visit_spine(); pop=load_pop_classification(); domains=load_domain_flags(); raw=load_selected_raw_columns(args.input)
    logger.info("[2/7] Deriving three separate baseline condition families")
    base,duplicates,n_pipe=build_baseline_comorbidity_dataset(raw,spine,pop); write_intermediate_dataset(base,BASELINE_PATH)
    families=[("past_medical_history",PAST_MEDICAL_HISTORY_CONDITIONS,True),
      ("rheumatologic_comorbidities",RHEUMATOLOGIC_NON_SAID_CONDITIONS,False),
      ("concomitant_said",CONCOMITANT_SAID_CONDITIONS,False),
      ("other_immune_mediated_systemic",OTHER_IMMUNE_MEDIATED_SYSTEMIC_CONDITIONS,False),
      ("sjd_associated_manifestations",RHEUMATOLOGIC_MANIFESTATIONS,False)]
    logger.info("[3/7] Writing separate descriptive outputs")
    produced=[]
    for stem,conditions,historical in families:
        overall=(summarize_historical_family(base,conditions) if historical else summarize_confirmed_family(base,conditions))
        by_pop=summarize_family_by_pop(base,conditions,run_sparse_monte_carlo=args.run_sparse_monte_carlo,
                                       replicates=args.monte_carlo_replicates,seed=args.random_seed)
        op=TABLES_DIR/f"07_{stem}_overall.csv"; bp=TABLES_DIR/f"07_{stem}_by_pop.csv"
        overall.to_csv(op,index=False); by_pop.to_csv(bp,index=False); produced.extend([op,bp])
    all_conditions=list(iter_analysis_conditions())
    overall=pd.concat([summarize_historical_family(base,PAST_MEDICAL_HISTORY_CONDITIONS),
      summarize_confirmed_family(base,[c for c in all_conditions if c.condition_family!="general_medical"])],ignore_index=True)
    overall["n_positive"]=overall.get("n_documented_history").fillna(overall.get("n_positive_any_yes"))
    overall["N"]=overall.get("n_total_patients").fillna(overall.get("N_evaluable"))
    overall["pct"]=100*overall["n_positive"]/overall["N"]
    overall.to_csv(TABLES_DIR/"07_comorbidities_overall.csv",index=False)
    by_pop=summarize_family_by_pop(base,all_conditions,run_sparse_monte_carlo=args.run_sparse_monte_carlo,replicates=args.monte_carlo_replicates,seed=args.random_seed)
    by_pop["fdr_family"]="prevalence_by_pop"; by_pop.to_csv(TABLES_DIR/"07_comorbidities_by_pop.csv",index=False)
    burden_summary(base).to_csv(TABLES_DIR/"07_comorbidity_burden_summary.csv",index=False)
    condition_dictionary().to_csv(TABLES_DIR/"07_comorbidity_condition_dictionary.csv",index=False)
    create_standard_figures(overall,by_pop,burden_summary(base))
    produced.extend([TABLES_DIR/"07_comorbidities_overall.csv",TABLES_DIR/"07_comorbidities_by_pop.csv",TABLES_DIR/"07_comorbidity_burden_summary.csv",TABLES_DIR/"07_comorbidity_condition_dictionary.csv"])
    pmh_section_order = [
        ("Cardiovascular / metabolic", "Cardiovascular/metabolic"),
        ("Respiratory", "Respiratory"), ("Endocrine", "Endocrine"),
        ("Gastrointestinal / hepatobiliary", "Gastrointestinal"),
        ("Neurologic / psychiatric", "Neuropsychiatric"),
        ("Renal / genitourinary", "Genitourinary"),
        ("Dermatologic", "Dermatologic"), ("Malignancy", "Malignancy"),
        ("Chronic infection", "Chronic infection"),
    ]
    pmh_sections = [(label, [c for c in PAST_MEDICAL_HISTORY_CONDITIONS
                             if c.clinical_category == category])
                    for label, category in pmh_section_order]
    pmh_plot = FIGURES_DIR / "07_past_medical_history.pdf"
    create_sectioned_comorbidity_plot(base, pmh_sections, pmh_plot,
                                      "Patients with documented history (%)")
    rheum_sections = [
        ("Rheumatologic Non-SAiD", RHEUMATOLOGIC_NON_SAID_CONDITIONS),
        ("Concomitant SAIDs", CONCOMITANT_SAID_CONDITIONS),
        ("Other immune-mediated/systemic conditions", OTHER_IMMUNE_MEDIATED_SYSTEMIC_CONDITIONS),
        ("Rheumatologic manifestations / associated features", RHEUMATOLOGIC_MANIFESTATIONS),
    ]
    rheum_plot = FIGURES_DIR / "07_rheumatological_comorbidities_all.pdf"
    create_sectioned_comorbidity_plot(base, rheum_sections, rheum_plot,
                                      "Patients exposed at baseline (%)")
    produced.extend([pmh_plot, rheum_plot])
    logger.info("[4/7] Preserving canonical longitudinal outcome datasets")
    long=build_longitudinal_essdai_dataset(raw,spine,pop,domains,base); write_intermediate_dataset(long,LONGITUDINAL_PATH)
    severe=build_severe5_survival_dataset(long,base); write_intermediate_dataset(severe,SEVERE_PATH)
    new_domain,domain_audit=build_new_domain_survival_dataset(long,base); write_intermediate_dataset(new_domain,NEW_DOMAIN_PATH); write_intermediate_dataset(domain_audit,DOMAIN_AUDIT_PATH)
    logger.info("[5/7] Fitting existing progression models by comorbidity family")
    progression_rows: list[dict[str, Any]]=[]; diagnostics: list[dict[str, Any]]=[]
    for condition in PROGRESSION_CONDITIONS:
        rows,diagnostic=fit_mixed_model(long,condition); progression_rows.extend(rows); diagnostics.append(diagnostic)
        row,diagnostic=fit_cox_model(severe,condition,"severe5_event","Progression to ESSDAI >=5",args.minimum_events); progression_rows.append(row); diagnostics.append(diagnostic)
        row,diagnostic=fit_cox_model(new_domain,condition,"new_domain_event","New ESSDAI-domain involvement",args.minimum_events); progression_rows.append(row); diagnostics.append(diagnostic)
    progression=finalize_model_metadata(pd.DataFrame(progression_rows))
    progression["fdr_family"]=progression["outcome"]
    progression["q_value"]=progression.groupby(["outcome","estimand"])["p_value"].transform(apply_fdr)
    progression["fdr_bh_q_value"]=progression["q_value"]
    model_outputs={"Longitudinal ESSDAI trajectory":"07_comorbidity_essdai_longitudinal_models.csv",
      "Progression to ESSDAI >=5":"07_comorbidity_severe5_models.csv",
      "New ESSDAI-domain involvement":"07_comorbidity_new_domain_models.csv"}
    for outcome,name in model_outputs.items():
        progression.loc[progression.outcome.eq(outcome)].to_csv(TABLES_DIR/name,index=False); produced.append(TABLES_DIR/name)
    interpretation = progression.assign(
        fallback_used=progression["model_used"].eq("GEE_fallback"),
        ph_warning=progression["ph_assumption_status"].eq("warning")
    )[["condition", "condition_family", "outcome", "estimand", "n_exposed", "n_events",
       "model_used", "model_status", "estimate", "CI95_low", "CI95_high", "p_value", "q_value",
       "sparse_exposure", "fallback_used", "ph_warning", "primary_interpretation_flag"]]
    interpretation_path = TABLES_DIR / "07_comorbidity_model_interpretation_summary.csv"
    interpretation.to_csv(interpretation_path, index=False); produced.append(interpretation_path)

    for stem, conditions in PROGRESSION_FAMILIES:
        names = {c.name for c in conditions}
        family_results = progression.loc[progression["comorbidity"].isin(names)].copy()
        progression_path=TABLES_DIR/f"07_{stem}_progression.csv"
        family_results.to_csv(progression_path,index=False); produced.append(progression_path)
    pmh_progression_plot = FIGURES_DIR / "07_past_medical_history_progression_forestplot.pdf"
    create_progression_forestplot(
        progression.loc[progression["comorbidity"].isin(
            {c.name for c in PAST_MEDICAL_HISTORY_CONDITIONS})],
        pmh_sections, pmh_progression_plot)
    rheum_progression_plot = FIGURES_DIR / "07_rheumatological_comorbidities_progression_forestplot.pdf"
    create_progression_forestplot(
        progression.loc[progression["comorbidity"].isin(
            {c.name for c in RHEUMATOLOGIC_DESCRIPTIVE_CONDITIONS})],
        rheum_sections, rheum_progression_plot)
    produced.extend([pmh_progression_plot, rheum_progression_plot])
    pd.DataFrame(diagnostics).to_csv(QC_DIR/"07_comorbidities_model_diagnostics.csv",index=False)
    hard_qc=run_qc_checks(base,long,severe,new_domain,domain_audit)
    logger.info("[6/7] Writing auditable source map and QC")
    mapping=source_mapping(raw.columns); mapping.to_csv(QC_DIR/"07_comorbidities_source_mapping.csv",index=False)
    duplicates.to_csv(QC_DIR/"07_comorbidities_patient_duplicates.csv",index=False)
    missingness_table(base).to_csv(QC_DIR/"07_comorbidities_missingness.csv",index=False)
    # Re-scan the complete source exactly once.  Baseline derivation may have
    # populated the collector for only a subset of rows.
    _UNRECOGNIZED.clear()
    source_qc(raw).to_csv(QC_DIR/"07_comorbidity_source_qc.csv",index=False)
    pd.DataFrame(_UNRECOGNIZED,columns=["source_column","original_value","row_index"]).drop_duplicates().to_csv(QC_DIR/"07_comorbidities_unrecognized_values.csv",index=False)
    unrecognized=pd.DataFrame(_UNRECOGNIZED,columns=["source_column","original_value","row_index"])
    if len(unrecognized): unrecognized=unrecognized.groupby(["source_column","original_value"],as_index=False).size().rename(columns={"size":"n_occurrences"})
    else: unrecognized=pd.DataFrame(columns=["source_column","original_value","n_occurrences"])
    unrecognized.to_csv(QC_DIR/"07_comorbidity_unrecognized_values.csv",index=False)
    audit_cols=["patient_id","clinical_baseline_episode_id","clinical_baseline_date","baseline_pop","baseline_essdai","age_baseline","sex",*CONDITION_NAMES]
    base[[c for c in audit_cols if c in base]].to_csv(QC_DIR/"07_comorbidity_baseline_patient_audit.csv",index=False)
    prevalence_qc=condition_prevalence_qc(base)
    model_qc = progression.drop_duplicates(["condition", "outcome"])[[
        "condition", "outcome", "model_attempted", "model_used", "converged",
        "singular_fit", "fallback_used", "n_exposed", "n_events", "sparse_exposure", "model_status"
    ]]
    model_qc.to_csv(QC_DIR/"07_comorbidity_model_qc.csv",index=False)
    pd.DataFrame({"metric":["n_longitudinal_rows","n_unmatched_pop","n_unmatched_domains"],"value":[len(long),int(long.pop_status.isna().sum()),int(long[ALL_DOMAINS].isna().all(axis=1).sum())]}).to_csv(QC_DIR/"07_comorbidity_merge_qc.csv",index=False)
    prohibited_used=sorted(x for c in iter_analysis_conditions() for x in (*c.primary,*c.detail_columns) if x.startswith(PROHIBITED_SOURCE_PREFIXES))
    qc={"input_path":str(args.input),"script_version":SCRIPT_VERSION,"run_timestamp":datetime.now(timezone.utc).isoformat(),
      "n_baseline_patients":len(base),"essdai_threshold":SEVERE_THRESHOLD,
      "condition_family_counts":{f:sum(c.condition_family==f for c in iter_analysis_conditions()) for f in ("general_medical","rheumatologic_non_said","concomitant_said","other_immune_mediated_systemic")},
      "prohibited_sources_used":prohibited_used,"prohibited_source_check_passed":not prohibited_used,
      "monte_carlo_enabled":args.run_sparse_monte_carlo,"upstream_file_timestamps":timestamps,
      "separate_burden_columns":["n_general_medical_history","n_rheumatologic_non_said","n_concomitant_said","n_other_immune_mediated_systemic"]}
    dictionary=condition_dictionary().set_index("condition")["condition_family"]
    family_mismatches=int(sum(
        table.set_index("condition")["condition_family"].ne(dictionary).sum()
        for table in (overall,by_pop,prevalence_qc)
    ))
    unrecognized_on=sum(
        str(value).strip().lower()=="on" for value in unrecognized.get("original_value", pd.Series(dtype=str))
    )
    qc_summary={"n_patients_clinical_spine":spine.patient_id.nunique(),"n_clinical_episodes":len(spine),
      "n_baseline_patients":len(base),"n_duplicate_patient_episode":int(spine.duplicated(["patient_id","clinical_episode_id"]).sum()),
      "n_patients_multiple_baseline":int(spine.loc[spine.is_clinical_baseline.eq(True)].patient_id.duplicated().sum()),
      "n_patients_without_baseline":int(spine.patient_id.nunique()-len(base)),"n_pop_baseline_available":int(base.baseline_pop.notna().sum()),
      "n_essdai_baseline_available":int(base.baseline_essdai.notna().sum()),"n_conditions_total":len(CONDITION_NAMES),
      "n_general_medical_conditions":len(PAST_MEDICAL_HISTORY_CONDITIONS),"n_rheumatologic_non_said_conditions":len(RHEUMATOLOGIC_NON_SAID_CONDITIONS),
      "n_concomitant_said_conditions":len(CONCOMITANT_SAID_CONDITIONS),"n_other_immune_conditions":len(OTHER_IMMUNE_MEDIATED_SYSTEMIC_CONDITIONS),
      "n_sjd_associated_manifestations":len(RHEUMATOLOGIC_MANIFESTATIONS),
      "n_conditions_with_family_mismatch":family_mismatches,
      "n_conditions_with_overall_vs_pop_definition_mismatch":int((~prevalence_qc.overall_pop_consistency_flag).sum()),
      "n_unrecognized_tokens":int(unrecognized.n_occurrences.sum()) if len(unrecognized) else 0,
      "n_unrecognized_on_tokens":int(unrecognized_on)}
    longitudinal_qc = model_qc[model_qc.outcome.eq("Longitudinal ESSDAI trajectory")]
    qc_summary.update({
      "n_longitudinal_models_total": len(longitudinal_qc),
      "n_longitudinal_lme_success": int(longitudinal_qc.model_used.eq("LME").sum()),
      "n_longitudinal_gee_fallback": int(longitudinal_qc.model_used.eq("GEE_fallback").sum()),
      "n_longitudinal_not_estimable": int(longitudinal_qc.model_used.eq("not_fitted").sum()),
      "n_sparse_exposure_models": int(model_qc.sparse_exposure.sum()),
      "n_sparse_and_gee_models": int((model_qc.sparse_exposure & model_qc.model_used.eq("GEE_fallback")).sum())})
    longitudinal_rows = progression[progression.outcome.eq("Longitudinal ESSDAI trajectory")]
    if not (longitudinal_rows.n_exposed + longitudinal_rows.n_unexposed).eq(longitudinal_rows.n_patients).all():
        raise AssertionError("Longitudinal exposure counts do not sum to modeled patients")
    fallback_inconsistent = (
        (longitudinal_rows.model_used.eq("GEE_fallback") & ~longitudinal_rows.fallback_used)
        | (longitudinal_rows.model_used.eq("LME") & longitudinal_rows.fallback_used)
    )
    if fallback_inconsistent.any():
        raise AssertionError("Longitudinal fallback metadata is inconsistent")
    for outcome in ("Progression to ESSDAI >=5", "New ESSDAI-domain involvement"):
        check = progression[progression.outcome.eq(outcome)]
        if (check.n_events > check.n_at_risk).any():
            raise AssertionError(f"Events exceed risk set for {outcome}")
    if len(severe) and not severe.baseline_essdai.lt(SEVERE_THRESHOLD).all():
        raise AssertionError("Severe5 risk set contains baseline ESSDAI >= threshold")
    if family_mismatches or not prevalence_qc.overall_pop_consistency_flag.all() or unrecognized_on:
        raise AssertionError("Comorbidity family/exposure/token QC hard check failed")
    pd.DataFrame([qc_summary]).to_csv(QC_DIR/"07_comorbidities_qc_summary.csv",index=False)
    (QC_DIR/"07_comorbidities_qc.json").write_text(json.dumps(qc,indent=2,default=str)+"\n")
    logger.info("[7/7] Complete: progression results written separately by comorbidity family")
    print("Generated files:"); [print(x.resolve()) for x in produced+[QC_DIR/"07_comorbidities_qc.json",QC_DIR/"07_comorbidities_source_mapping.csv"]]
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

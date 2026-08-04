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
SCRIPT_VERSION = "1.3.0"

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
    sensitivity: tuple[str, ...] = ()
    definition_type: str = "primary"
    notes: str = ""
    category: str = "rheumatological"


EXISTING_AND_RHEUMATOLOGICAL_CONDITIONS = [
    Condition("fibromyalgia", "Fibromyalgia", ("rheumatological_comorbidities__fibromyalgia1", "rheumatological_comorbidities__fibromyalgia1_hx", "rheumatological_comorbidities__fibromyalgia1_confirm"), category="rheumatological"),
    Condition("osteoporosis", "Osteoporosis", ("rheumatological_comorbidities__osteoporosis1", "rheumatological_comorbidities__osteoporosis1_hx", "rheumatological_comorbidities__osteoporosis1_confirm"), category="rheumatological"),
    Condition("ild", "Interstitial lung disease", ("past_medical_history__respiratory_hx_lung", "past_medical_history__resp_hx_lung_fibro"), ("sjogren's_syndrome_disease_damage_index__fibrosis_interstitial",), category="pulmonary"),
    Condition("thyroid_disease", "Thyroid disease", ("past_medical_history__thyroid_disease",), ("sjogren's_syndrome_history__thyroiditis",), category="endocrine"),
    Condition("depression", "Depression", ("past_medical_history__neuro_hx_depression",), ("systems_review_for_physician__psych_depr",), category="neurologic"),
    Condition("anxiety", "Physician-recorded anxiety symptoms", ("systems_review_for_physician__adr_q22_anxiety",), ("ans__anxiety", "ans__anxiety_severity", "autonomic_nervous_system_questionnaire__anxiety", "autonomic_nervous_system_questionnaire__anxiety_severity"), category="neurologic", notes="Physician-recorded symptom indicator; not a confirmed anxiety diagnosis."),
    Condition("raynaud", "Raynaud's phenomenon", ("rheumatological_comorbidities__integ_raynds", "rheumatological_comorbidities__integ_raynds_hx", "rheumatological_comorbidities__integ_raynds_confirm"), ("sjogren's_syndrome_history__raynaud_phenom",), category="rheumatological"),
    Condition("peripheral_neuropathy", "Peripheral neuropathy", ("past_medical_history__neuro_hx_neuropathy",), ("sjogren's_syndrome_disease_damage_index__neuro_cran_periph",), category="neurologic"),
    Condition("renal_tubular_acidosis", "Renal tubular acidosis", ("past_medical_history__renal_hx_tubular_acid",), category="renal_urinary"),
    Condition("myositis", "Myositis", ("rheumatological_comorbidities__polymyositis", "rheumatological_comorbidities__polymyositis_hx", "rheumatological_comorbidities__polymyositis_confirm", "rheumatological_comorbidities__dermatomyositis", "rheumatological_comorbidities__dermatomyositis_hx", "rheumatological_comorbidities__dermatomyositis_confirm"), ("sjogren's_syndrome_history__myositis_myalgia",), category="rheumatological"),
    Condition("cryoglobulinemia", "Cryoglobulinemia", ("rheumatological_comorbidities__cryoglobulinemia", "rheumatological_comorbidities__cryoglobulinemia_hx", "rheumatological_comorbidities__cryoglobulinemia_confirm"), category="rheumatological"),
    Condition("chronic_bronchitis", "Chronic bronchitis", ("past_medical_history__respiratory_hx_bronchitis",), category="pulmonary"),
    Condition("sle", "Systemic lupus erythematosus", ("rheumatological_comorbidities__sle1", "rheumatological_comorbidities__sle_hx", "rheumatological_comorbidities__sle_confirmed"), category="rheumatological"),
    Condition("rheumatoid_arthritis", "Rheumatoid arthritis", ("rheumatological_comorbidities__ra", "rheumatological_comorbidities__ra_hx", "rheumatological_comorbidities__ra_confirm"), category="rheumatological"),
    Condition("systemic_sclerosis", "Systemic sclerosis", ("rheumatological_comorbidities__systemic_sclerosis", "rheumatological_comorbidities__systmc_sclerosis_hx", "rheumatological_comorbidities__systmc_sclerosis_confirm"), category="rheumatological"),
    Condition("mixed_connective_tissue_disease", "Mixed connective tissue disease", ("rheumatological_comorbidities__mixed_connective_tissue_disease", "rheumatological_comorbidities__mixed_connect_tissue_hx", "rheumatological_comorbidities__mixed_connect_tissue_confirm"), category="rheumatological"),
    Condition("antiphospholipid_syndrome", "Antiphospholipid syndrome", ("rheumatological_comorbidities__antiphospholipid_syndrome", "rheumatological_comorbidities__antiphospholipid_syn_hx", "rheumatological_comorbidities__antiphospholipid_syn_confirm"), category="rheumatological"),
    Condition("primary_biliary_cholangitis", "Primary biliary cholangitis", ("rheumatological_comorbidities__primary_billiary_cirrhosis", "rheumatological_comorbidities__prim_billiary_cirrhosis_hx", "rheumatological_comorbidities__prim_billiary_cirrhosis_confirm", "past_medical_history__gi_hx_cirrhosis"), category="gastrointestinal"),
    Condition("osteopenia", "Osteopenia", ("rheumatological_comorbidities__osteopenia", "rheumatological_comorbidities__osteopenia_hx", "rheumatological_comorbidities__osteopenia_confirm"), category="rheumatological"),
    Condition("osteoarthritis", "Osteoarthritis", ("rheumatological_comorbidities__osteoarthritis", "rheumatological_comorbidities__osteoarthritis_hx", "rheumatological_comorbidities__osteoarthritis_confirm"), category="rheumatological"),
    Condition("sarcoidosis", "Sarcoidosis", ("rheumatological_comorbidities__sarcoidosis", "rheumatological_comorbidities__sarcoidosis_hx", "rheumatological_comorbidities__sarcoidosis_confirm"), category="rheumatological"),
    Condition("crystalline_arthropathy", "Crystalline arthropathy", ("rheumatological_comorbidities__crystalline_arthropathy", "rheumatological_comorbidities__crystalline_arthropathy_hx", "rheumatological_comorbidities__crystalline_arthro_confirm"), category="rheumatological"),
    Condition("inflammatory_bowel_disease", "Inflammatory bowel disease", ("rheumatological_comorbidities__inflam_bowel", "rheumatological_comorbidities__inflam_bowel_hx", "rheumatological_comorbidities__inflam_bowel_confirm", "past_medical_history__inflam_bowel"), category="gastrointestinal"),
    Condition("other_rheumatological_condition", "Other rheumatological condition", ("rheumatological_comorbidities__rheumatological_other", "rheumatological_comorbidities__rheumatological_other_hx", "rheumatological_comorbidities__rheumatological_other_confirm"), category="rheumatological"),
]

PAST_MEDICAL_HISTORY_CONDITIONS = [
    Condition("hypertension", "Hypertension", ("past_medical_history__hypertension_",), category="cardiovascular"),
    Condition("hyperlipidemia", "Hyperlipidemia", ("past_medical_history__hyperlipidemia_",), category="cardiovascular"),
    Condition("coronary_artery_disease", "Coronary artery disease", ("past_medical_history__cardio_hx_cad",), category="cardiovascular"),
    Condition("myocardial_infarction", "Myocardial infarction", ("past_medical_history__cardio_hx_mycrdl",), category="cardiovascular"),
    Condition("peripheral_vascular_disease", "Peripheral vascular disease", ("past_medical_history__cardio_hx_pvd",), category="cardiovascular"),
    Condition("pericarditis", "Pericarditis", ("past_medical_history__cardio_hx_pericard",), category="cardiovascular"),
    Condition("valvular_disease", "Valvular disease", ("past_medical_history__cardio_valve_disease",), category="cardiovascular"),
    Condition("cerebrovascular_disease", "Cerebrovascular disease", ("past_medical_history__cva",), category="cardiovascular"),
    Condition("pulmonary_embolism", "Pulmonary embolism", ("past_medical_history__pulmonary_embolism",), category="cardiovascular"),
    Condition("diabetes_mellitus", "Diabetes mellitus", ("past_medical_history__endcrn_hx_mellitus_i", "past_medical_history__endcrn_hx_mellitus_ii"), category="endocrine"),
    Condition("asthma", "Asthma", ("past_medical_history__asthma",), category="pulmonary"),
    Condition("copd", "Emphysema or COPD", ("past_medical_history__respiratory_hx_copd",), category="pulmonary"),
    Condition("pulmonary_hypertension", "Pulmonary hypertension", ("past_medical_history__resp_hx_pulm_hyper",), category="pulmonary"),
    Condition("pulmonary_granulomas", "Pulmonary granulomas", ("past_medical_history__resp_hx_pulm_gran",), category="pulmonary"),
    Condition("pleuritis", "Pleuritis", ("past_medical_history__respiratory_hx_pleuritis",), category="pulmonary"),
    Condition("autoimmune_hepatitis", "Autoimmune hepatitis", ("past_medical_history__gi_hx_auto_hepat",), category="gastrointestinal"),
    Condition("celiac_disease", "Celiac disease", ("past_medical_history__gi_hx_celiar",), category="gastrointestinal"),
    Condition("primary_sclerosing_cholangitis", "Primary sclerosing cholangitis", ("past_medical_history__gi_hx_sclerosing",), category="gastrointestinal"),
    Condition("gerd", "Gastroesophageal reflux disease", ("past_medical_history__gi_hx_gerd",), category="gastrointestinal"),
    Condition("irritable_bowel_syndrome", "Irritable bowel syndrome", ("past_medical_history__gi_hx_ibs",), category="gastrointestinal"),
    Condition("pancreatitis", "Pancreatitis", ("past_medical_history__pancreatitis",), category="gastrointestinal"),
    Condition("hepatitis_b", "Hepatitis B", ("past_medical_history__gi_hx_hepatitis_b",), category="gastrointestinal"),
    Condition("hepatitis_c", "Hepatitis C", ("past_medical_history__gi_hx_hepatitis_c",), category="gastrointestinal"),
    Condition("psoriasis", "Psoriasis", ("past_medical_history__cutaneous_hx_psoriasis",), category="cutaneous_autoimmune"),
    Condition("vitiligo", "Vitiligo", ("past_medical_history__cutaneous_hx_vitiligo",), category="cutaneous_autoimmune"),
    Condition("vasculitis", "Vasculitis", ("past_medical_history__vasculitis",), category="cutaneous_autoimmune"),
    Condition("multiple_sclerosis", "Multiple sclerosis", ("past_medical_history__neuro_hx_mult_sclerosis",), category="neurologic"),
    Condition("seizures", "Seizures", ("past_medical_history__neuro_seizrs",), category="neurologic"),
    Condition("chorea", "Chorea", ("past_medical_history__neuro_hx_chorea",), category="neurologic"),
    Condition("cognitive_dysfunction", "Cognitive dysfunction", ("past_medical_history__neuro_hx_cog_dyfsfun",), category="neurologic"),
    Condition("headaches", "Headaches", ("past_medical_history__neuro_hx_headaches",), category="neurologic"),
    Condition("autonomic_dysfunction", "Autonomic dysfunction", ("past_medical_history__dysfunctn_autonomic",), category="neurologic"),
    Condition("chronic_fatigue_syndrome", "Chronic fatigue syndrome", ("past_medical_history__chronic_fatigue_syndrome_dx",), category="neurologic"),
    Condition("glomerulonephritis", "Glomerulonephritis", ("past_medical_history__renal_hx_glomer",), category="renal_urinary"),
    Condition("nephrotic_syndrome", "Nephrotic syndrome", ("past_medical_history__renal_hx_nephrotic",), category="renal_urinary"),
    Condition("kidney_stones", "Kidney stones", ("past_medical_history__renal_hx_kidney_stones",), category="renal_urinary"),
    Condition("pyelonephritis", "Pyelonephritis", ("past_medical_history__renal_hx_pyelonephritis",), category="renal_urinary"),
    Condition("recurrent_uti", "Recurrent urinary tract infections", ("past_medical_history__renal_hx_recurr_uti",), category="renal_urinary"),
    Condition("interstitial_cystitis", "Interstitial cystitis", ("past_medical_history__interstitial_cyst",), category="renal_urinary"),
    Condition("bladder_incontinence", "Bladder incontinence", ("past_medical_history__incontinence_",), category="renal_urinary"),
    Condition("hiv_aids", "HIV/AIDS", ("past_medical_history__ec_aids",), category="infection"),
    Condition("htlv_infection", "HTLV-I/II infection", ("past_medical_history__htlv_infection",), category="infection"),
    Condition("infectious_mononucleosis", "Infectious mononucleosis", ("past_medical_history__infect_mononucleosis",), category="infection"),
    Condition("lyme_disease", "Lyme disease", ("past_medical_history__lyme_disease",), category="infection"),
    Condition("graft_versus_host_disease", "Graft-versus-host disease", ("past_medical_history__ec_graft_host_disease",), category="infection"),
    Condition("recurrent_sinusitis", "Frequent or recurrent sinusitis", ("past_medical_history__sinusitis",), category="infection"),
    Condition("lymphoma", "Lymphoma", ("past_medical_history__malignancy_hx_lymphoma",), category="malignancy"),
    Condition("breast_cancer", "Breast cancer", ("past_medical_history__malignancy_hx_breast_ca",), category="malignancy"),
    Condition("colon_cancer", "Colon cancer", ("past_medical_history__malignancy_hx_colon_ca",), category="malignancy"),
    Condition("head_neck_cancer", "Head or neck cancer", ("past_medical_history__malignancy_hx_head_ca",), category="malignancy"),
    Condition("lung_cancer", "Lung cancer", ("past_medical_history__malignancy_hx_lung_ca",), category="malignancy"),
    Condition("thyroid_cancer", "Thyroid cancer", ("past_medical_history__malignancy_hx_thyroid_ca",), category="malignancy"),
    Condition("other_malignancy", "Other malignancy", ("past_medical_history__malignancy_hx_other",), category="malignancy"),
]

CONDITIONS = EXISTING_AND_RHEUMATOLOGICAL_CONDITIONS + PAST_MEDICAL_HISTORY_CONDITIONS
CONDITION_NAMES = [c.name for c in CONDITIONS]
PROGRESSION_CONDITION_NAMES = {
    "fibromyalgia",
    "osteoporosis",
    "ild",
    "thyroid_disease",
    "depression",
    "anxiety",
    "raynaud",
    "peripheral_neuropathy",
    "renal_tubular_acidosis",
    "myositis",
    "cryoglobulinemia",
    "chronic_bronchitis",
}
PROGRESSION_CONDITIONS = [c for c in CONDITIONS if c.name in PROGRESSION_CONDITION_NAMES]
PROGRESSION_CONDITION_NAMES_ORDERED = [c.name for c in PROGRESSION_CONDITIONS]
SUBTYPE_COLS = ("past_medical_history__thyroid_disease_spfy", "past_medical_history__neuro_hx_neuropathy_spfy")

UNAVAILABLE = {
    "bronchiectasis": "No exact physician-documented bronchiectasis variable; chronic bronchitis is not a substitute.",
    "psoriatic_arthritis": "No exact psoriatic arthritis variable; psoriasis is not a substitute.",
    "hypokalemia": "No exact physician-documented hypokalemia variable; an isolated potassium result is not a diagnosis.",
    "gastritis": "No exact gastritis variable; a nonspecific gastrointestinal 'other' field is not a substitute.",
    "neutropenia": "No exact baseline neutropenia diagnosis; hematologic ESSDAI activity is not a substitute.",
}
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
    p.add_argument("--monte-carlo-replicates", type=int, default=100_000)
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
    available = set(pq.read_schema(path).names)
    missing = set(columns) - available
    if missing:
        raise KeyError(f"{path} lacks required columns: {sorted(missing)}")
    return pd.read_parquet(path, columns=list(columns))


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
    schema = set(pq.read_schema(input_path).names)
    desired = [PATIENT_ID_COL, VISIT_DATE_COL, AGE_COL, SEX_COL, ESSDAI_PRIMARY_COL, ESSDAI_RAW_QC_COL, *SUBTYPE_COLS]
    desired.extend(col for c in CONDITIONS for col in (*c.primary, *c.sensitivity))
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


def derive_comorbidity_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    for condition in CONDITIONS:
        primary_flags = []
        for col in condition.primary:
            if col in out:
                flag_col = f"source__{col}"
                out[flag_col] = normalize_binary_flag(out[col])
                primary_flags.append(flag_col)
        out[condition.name] = nullable_or(out[primary_flags])
        sens_flags = primary_flags.copy()
        for col in condition.sensitivity:
            if col in out:
                flag_col = f"source__{col}"
                out[flag_col] = normalize_binary_flag(out[col])
                sens_flags.append(flag_col)
        if condition.sensitivity:
            out[f"{condition.name}_sensitivity"] = nullable_or(out[sens_flags])
    # Explicit provenance and neuropathy subtypes.
    out["fibromyalgia_history"] = out.get("source__rheumatological_comorbidities__fibromyalgia1_hx", pd.Series(pd.NA, index=out.index, dtype="boolean"))
    out["fibromyalgia_confirmed"] = out.get("source__rheumatological_comorbidities__fibromyalgia1_confirm", pd.Series(pd.NA, index=out.index, dtype="boolean"))
    subtype = out.get(SUBTYPE_COLS[1], pd.Series(pd.NA, index=out.index)).astype("string").str.lower()
    base_eval = out["peripheral_neuropathy"].notna()
    sensory = subtype.str.contains(r"sensor|small fiber", na=False)
    motor = subtype.str.contains(r"motor", na=False)
    cranial = subtype.str.contains(r"cranial", na=False)
    for name, values in (("sensory_neuropathy", sensory), ("motor_neuropathy", motor), ("cranial_nerve_involvement", cranial), ("sensory_motor_neuropathy", sensory & motor)):
        flag = pd.Series(pd.NA, index=out.index, dtype="boolean")
        flag.loc[base_eval] = values.loc[base_eval]
        out[name] = flag
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
    base[CONDITION_NAMES] = base[CONDITION_NAMES].fillna(False).astype("boolean")
    base["n_prespecified_comorbidities"] = base[CONDITION_NAMES].astype(int).sum(axis=1)
    base["n_comorbidities_evaluable"] = base[CONDITION_NAMES].notna().sum(axis=1)
    base["any_comorbidity"] = (base["n_prespecified_comorbidities"] > 0).astype("boolean")
    base["two_or_more_comorbidities"] = (base["n_prespecified_comorbidities"] >= 2).astype("boolean")
    if len(base) != base["patient_id"].nunique():
        raise ValueError("Baseline dataset is not one row per patient")
    if not base["baseline_date"].equals(base["observed_baseline_date"]):
        raise ValueError("Baseline date is not the canonical observed baseline date")
    return base, duplicate_audit, n_pipe


def summarize_overall_prevalence(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_total = len(base)
    for c in CONDITIONS:
        s = base[c.name].fillna(False).astype("boolean")
        n_eval, n_pos = int(s.notna().sum()), int(s.eq(True).sum())
        rows.append({"condition": c.name, "display_label": c.label, "definition_type": c.definition_type,
                     "category": c.category, "source_columns": "|".join(c.primary), "n_total_cohort": n_total, "n_evaluable": n_eval,
                     "n_positive": n_pos, "n_negative": int(s.eq(False).sum()), "n_missing": int(s.isna().sum()),
                     "pct_total_cohort": 100*n_pos/n_total if n_total else np.nan,
                     "pct_among_evaluable": 100*n_pos/n_eval if n_eval else np.nan,
                     "availability_status": "available", "notes": c.notes})
    out = pd.DataFrame(rows).sort_values(["pct_total_cohort", "display_label"], ascending=[False, True]).reset_index(drop=True)
    out["rank_by_prevalence"] = out["pct_total_cohort"].rank(method="min", ascending=False).astype("Int64")
    return out


def _monte_carlo_p(table: np.ndarray, replicates: int, rng: np.random.Generator) -> float:
    observed = chi2_contingency(table, correction=False)[0]
    groups = np.repeat(np.arange(table.shape[0]), table.sum(axis=1))
    outcomes = np.repeat([1, 0], table.sum(axis=0))
    exceed = 0
    for _ in range(replicates):
        shuffled = rng.permutation(outcomes)
        sim = np.array([[np.sum(shuffled[groups == g] == 1), np.sum(shuffled[groups == g] == 0)] for g in range(table.shape[0])])
        exceed += chi2_contingency(sim, correction=False)[0] >= observed
    return (exceed + 1) / (replicates + 1)


def calculate_or_and_fisher(table: np.ndarray) -> dict[str, Any]:
    if table.shape != (2, 2) or table.sum(axis=1).min() == 0:
        return {"odds_ratio_pop2_vs_pop3": np.nan, "or_ci95_low": np.nan, "or_ci95_high": np.nan, "fisher_exact_p_value": np.nan, "zero_cell_correction_used": False}
    _, p = fisher_exact(table)
    corrected = bool((table == 0).any()); a, b, c, d = table.ravel().astype(float)
    if corrected: a, b, c, d = a+.5, b+.5, c+.5, d+.5
    odds = a*d/(b*c); se = math.sqrt(1/a+1/b+1/c+1/d)
    return {"odds_ratio_pop2_vs_pop3": odds, "or_ci95_low": math.exp(math.log(odds)-1.96*se), "or_ci95_high": math.exp(math.log(odds)+1.96*se), "fisher_exact_p_value": p, "zero_cell_correction_used": corrected}


def summarize_prevalence_by_pop(base: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed); rows = []
    n_total = len(base)
    for c in CONDITIONS:
        row: dict[str, Any] = {"condition": c.name, "display_label": c.label, "category": c.category,
                                "n_positive_total_cohort": int(base[c.name].fillna(False).eq(True).sum()),
                                "pct_total_cohort": 100*int(base[c.name].fillna(False).eq(True).sum())/n_total if n_total else np.nan}
        table = []
        for i, pop_name in enumerate(("Pop1", "Pop2", "Pop3"), 1):
            s = base.loc[base["baseline_pop"].eq(pop_name), c.name].fillna(False).astype("boolean")
            n, N = int(s.eq(True).sum()), int(s.notna().sum())
            row.update({f"n_pop{i}": n, f"N_pop{i}": N, f"pct_pop{i}": 100*n/N if N else np.nan})
            table.append([n, int(s.eq(False).sum())])
        arr = np.asarray(table)
        if (arr.sum(axis=1) == 0).any() or (arr.sum(axis=0) == 0).any():
            p, expected_min, test = np.nan, np.nan, "not estimable"
        else:
            _, asym_p, _, expected = chi2_contingency(arr, correction=False); expected_min = float(expected.min())
            if expected_min < 5:
                p, test = _monte_carlo_p(arr, replicates, rng), f"Monte Carlo chi-square ({replicates} replicates)"
            else: p, test = asym_p, "Pearson chi-square"
        row.update({"global_test": test, "global_p_value": p, "minimum_expected_cell": expected_min, "sparse_table_flag": bool(np.isfinite(expected_min) and expected_min < 5)})
        row.update(calculate_or_and_fisher(arr[1:3]))
        pct2, pct3 = row["pct_pop2"], row["pct_pop3"]
        direction = "higher" if pct2 > pct3 else "lower" if pct2 < pct3 else "similar"
        if np.isfinite(row["or_ci95_low"]) and row["fisher_exact_p_value"] < .05:
            row["interpretation_status"] = f"Prevalence was {direction} in Pop 2 than Pop 3; this is an association, not a causal effect."
        else:
            row["interpretation_status"] = f"Prevalence was numerically {direction} in Pop 2, but the estimate was imprecise and did not provide clear evidence of a between-group association."
        rows.append(row)
    out = pd.DataFrame(rows)
    out["fdr_bh_q_value"] = apply_fdr(out["global_p_value"])
    return out



def summarize_overall_by_category(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_total = len(base)
    for category in sorted({c.category for c in CONDITIONS}):
        names = [c.name for c in CONDITIONS if c.category == category]
        counts = base[names].fillna(False).astype(int).sum(axis=1)
        n_positive = int((counts > 0).sum())
        if len(counts):
            q1, q3 = counts.quantile([.25, .75])
            median = float(counts.median())
            iqr = f"{q1:.1f}-{q3:.1f}"
        else:
            median, iqr = np.nan, ""
        rows.append({
            "category": category,
            "n_patients_with_at_least_one_condition": n_positive,
            "denominator": n_total,
            "pct_patients": 100*n_positive/n_total if n_total else np.nan,
            "median_conditions_per_patient": median,
            "iqr_conditions_per_patient": iqr,
        })
    return pd.DataFrame(rows).sort_values(["pct_patients", "category"], ascending=[False, True]).reset_index(drop=True)

def apply_fdr(p_values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p_values.index, dtype=float); valid = p_values.notna()
    if valid.any(): out.loc[valid] = multipletests(p_values.loc[valid], method="fdr_bh")[1]
    return out


def build_longitudinal_essdai_dataset(raw: pd.DataFrame, spine: pd.DataFrame, pop: pd.DataFrame, domains: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    parsed = add_parsed_visit_dates(raw, PATIENT_ID_COL, VISIT_DATE_COL)
    values = parsed.assign(essdai_total_recoded=pd.to_numeric(parsed[ESSDAI_PRIMARY_COL], errors="coerce"), essdai_total_raw_qc=pd.to_numeric(parsed[ESSDAI_RAW_QC_COL], errors="coerce"))
    values = values.groupby(["patient_id", "visit_date"], as_index=False).agg(essdai_total_recoded=("essdai_total_recoded", "first"), essdai_total_raw_qc=("essdai_total_raw_qc", "first"))
    long = spine.merge(values, on=["patient_id", "visit_date"], how="left", validate="one_to_one")
    popcols = pop[["visit_id", "pop_status"]].drop_duplicates("visit_id")
    long = long.merge(popcols, on="visit_id", how="left", validate="one_to_one").merge(domains.drop(columns=["patient_id", "visit_date"]), on="visit_id", how="left", validate="one_to_one")
    bcols = ["patient_id", "baseline_essdai", "baseline_pop", "age_baseline", "sex", *CONDITION_NAMES]
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


def fit_mixed_model(long: pd.DataFrame, c: Condition) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import statsmodels.formula.api as smf
    cols = ["patient_id", "essdai_total_recoded", "time_since_observed_baseline_years", c.name, "baseline_essdai", "baseline_pop", "age_baseline", "sex", "visit_number"]
    data = long.loc[long["visit_number"] > 0, cols].dropna().copy(); data[c.name] = data[c.name].astype(int)
    eligible = data.groupby("patient_id").size(); n = len(eligible)
    if n < 5 or data[c.name].nunique() < 2:
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
    """Fit a Cox model using condition absence as the reference exposure."""
    counts = {"n_patients": len(data), "n_events": int(data[event_col].sum()) if len(data) else 0, "n_complete_cases": 0}
    cols = ["followup_years", event_col, c.name, "baseline_essdai", "baseline_pop", "age_baseline", "sex"]
    d = data[cols].copy()
    n_exposure_missing_as_negative = int(d[c.name].isna().sum())
    d[c.name] = d[c.name].fillna(False).astype("boolean")
    # The baseline builder normally resolves empty absence-by-default fields;
    # fill again here so callers supplying an older intermediate get the same
    # clinical coding. Complete-case exclusion still applies to other fields.
    d = d.dropna().copy()
    counts.update({"n_complete_cases": len(d), "n_patients": len(d),
                   "n_events": int(d[event_col].sum()) if len(d) else 0})
    if importlib.util.find_spec("lifelines") is None:
        row = _empty_progression(c, outcome, "Baseline comorbidity", "lifelines is not installed; Cox model not executed", **counts)
        row.update({"baseline_reference_group": "Comorbidity absent",
                    "n_exposure_missing_assigned_negative": n_exposure_missing_as_negative})
        return row, {"comorbidity": c.name, "outcome": outcome, "convergence": False,
                     "n_exposure_missing_assigned_negative": n_exposure_missing_as_negative,
                     "warning": row["warning"]}
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    if len(d) < 5 or d[c.name].nunique() < 2 or counts["n_events"] < minimum_events:
        row = _empty_progression(c, outcome, "Baseline comorbidity", f"Insufficient events (<{minimum_events}) or exposure variation", **counts)
        row.update({"model_status": "insufficient_events",
                    "baseline_reference_group": "Comorbidity absent",
                    "n_exposure_missing_assigned_negative": n_exposure_missing_as_negative})
        return row, {"comorbidity": c.name, "outcome": outcome, "convergence": False,
                     "events": counts["n_events"],
                     "n_exposure_missing_assigned_negative": n_exposure_missing_as_negative,
                     "warning": row["warning"]}
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
        row = {**_empty_progression(c, outcome, "Baseline comorbidity", warning_text, **counts), "model_type": model_type, "effect_measure": "Hazard ratio", "estimate": float(summary["exp(coef)"]), "ci95_low": float(summary["exp(coef) lower 95%"]), "ci95_high": float(summary["exp(coef) upper 95%"]), "p_value": float(summary["p"]), "model_converged": True, "proportional_hazards_p": ph_p, "sparse_event_flag": reduced, "model_status": "reduced_adjustment" if reduced else "fitted", "baseline_reference_group": "Comorbidity absent", "n_exposure_missing_assigned_negative": n_exposure_missing_as_negative, "interpretation": "Adjusted hazard association using comorbidity absence as the reference. This is not a causal effect."}
        return row, {"comorbidity": c.name, "outcome": outcome, "convergence": True, "events": counts["n_events"], "events_per_parameter": counts["n_events"]/max(len(d.columns)-2, 1), "n_exposure_missing_assigned_negative": n_exposure_missing_as_negative, "proportional_hazards_p": ph_p, "warning": warning_text}
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        row = _empty_progression(c, outcome, "Baseline comorbidity", f"Cox model failed: {exc}", **counts)
        row.update({"model_status": "failed", "baseline_reference_group": "Comorbidity absent",
                    "n_exposure_missing_assigned_negative": n_exposure_missing_as_negative})
        return row, {"comorbidity": c.name, "outcome": outcome, "convergence": False,
                     "events": counts["n_events"],
                     "n_exposure_missing_assigned_negative": n_exposure_missing_as_negative,
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
    d = overall.sort_values(["pct_total_cohort", "display_label"], ascending=[False, True]).head(20)
    d = d.sort_values(["pct_total_cohort", "display_label"], ascending=[True, False])
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
    fig.text(.01, .01, "Top 20 conditions by baseline prevalence; complete results are provided in the accompanying table. Empty condition fields are coded as absent.", fontsize=8)
    fig.subplots_adjust(bottom=.1, left=.3); _plot_save(fig, FIGURES_DIR/"07_comorbidities_dotplot.pdf")


def create_grouped_barplot(by_pop: pd.DataFrame) -> None:
    d = by_pop.loc[by_pop["n_positive_total_cohort"].ge(5)].sort_values(["pct_total_cohort", "display_label"], ascending=[False, True]).reset_index(drop=True)
    if d.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(.5, .5, "No comorbidity had at least five positive patients.", ha="center", va="center")
        ax.axis("off")
        _plot_save(fig, FIGURES_DIR/"07_comorbidities_grouped_bar.pdf")
        return
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
            label = "Not estimable" if pd.isna(pct) else f"{pct:.1f}% ({int(n)}/{int(N)})"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=7,
                    color="white" if pd.notna(pct) and pct > 0.6*np.nanmax(values) else "black")
    ax.set_title("Baseline comorbidity prevalence by Pop")
    cbar = fig.colorbar(image, ax=ax); cbar.set_label("Prevalence (%)")
    fig.text(.01, .01, "Cells use baseline Pop denominators; empty condition fields are coded as absent. Grey cells are not estimable.", fontsize=8)
    fig.subplots_adjust(left=.35, bottom=.12); _plot_save(fig, FIGURES_DIR/"07_comorbidities_grouped_bar.pdf")


def create_progression_forestplot(progression: pd.DataFrame) -> None:
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
    axes[0].set_yticks(range(len(order)), [labels[x] for x in order]); fig.text(.01,.01,"Models integrate patients across visits and adjust for baseline ESSDAI, baseline Pop, age, and sex when event support permits. Comorbidity absence is the reference. X marks not estimable; red denotes reduced models. Associations are not causal.",fontsize=8)
    fig.subplots_adjust(left=.18,bottom=.1,wspace=.15); _plot_save(fig, FIGURES_DIR/"07_comorbidities_progression_forestplot.pdf")


def run_qc_checks(base: pd.DataFrame, long: pd.DataFrame, severe: pd.DataFrame, new_domain: pd.DataFrame, domain_audit: pd.DataFrame) -> dict[str, Any]:
    if base["patient_id"].isna().any() or base["patient_id"].duplicated().any(): raise ValueError("Invalid baseline patient identity")
    if not base["baseline_date"].eq(base["observed_baseline_date"]).all(): raise ValueError("Noncanonical baseline detected")
    for col in CONDITION_NAMES + ALL_DOMAINS + list(DOMAIN_EVALUABLE.values()):
        source = base[col] if col in base else long[col]
        if str(source.dtype) != "boolean": raise TypeError(f"{col} is not nullable boolean")
    # Subtype implication is an enforceable invariant.
    for subtype in ("sensory_neuropathy", "motor_neuropathy", "cranial_nerve_involvement", "sensory_motor_neuropathy"):
        if subtype in base and (base[subtype].eq(True) & ~base["peripheral_neuropathy"].eq(True)).any(): raise ValueError(f"Positive {subtype} without peripheral neuropathy")
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
    raw_columns=set(raw_columns); counts=pd.DataFrame(_UNRECOGNIZED)
    rows=[]
    for c in CONDITIONS:
        present=[x for x in c.primary if x in raw_columns]
        rows.append({"condition":c.name,"primary_source_columns":"|".join(c.primary),"sensitivity_source_columns":"|".join(c.sensitivity),"derivation_rule":"Logical OR across available primary sources; empty baseline condition coded absent","definition_type":c.definition_type,"category":c.category,"availability":"available" if present else "unavailable","n_unrecognized_values":int(counts["source_column"].isin(c.primary+c.sensitivity).sum()) if len(counts) else 0})
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args=parse_args(argv); ensure_directories(); np.random.seed(args.random_seed)
    intermediate_outputs = [path for parquet_path in INTERMEDIATE_PATHS for path in (parquet_path, parquet_path.with_suffix(".csv"))]
    outputs=intermediate_outputs+[TABLES_DIR/"07_comorbidities_overall.csv",TABLES_DIR/"07_comorbidities_by_pop.csv",TABLES_DIR/"07_comorbidities_overall_by_category.csv",TABLES_DIR/"07_comorbidities_progression.csv",FIGURES_DIR/"07_comorbidities_dotplot.pdf",FIGURES_DIR/"07_comorbidities_grouped_bar.pdf",FIGURES_DIR/"07_comorbidities_progression_forestplot.pdf",QC_DIR/"07_comorbidities_qc.json",QC_DIR/"07_comorbidities_missingness.csv",QC_DIR/"07_comorbidities_source_mapping.csv",QC_DIR/"07_comorbidities_model_diagnostics.csv",QC_DIR/"07_comorbidities_patient_duplicates.csv",QC_DIR/"07_comorbidities_unavailable_conditions.csv",QC_DIR/"07_comorbidities_unrecognized_values.csv",LOG_PATH]
    existing = [p for p in outputs if p.exists()]
    logger=setup_logging()
    if existing:
        logger.info("Replacing %d existing Section 5 output(s)", len(existing))
    logger.info("[1/8] Loading canonical sources")
    timestamps=check_upstream_artifacts(args.input,args.rebuild_upstream,logger); spine=load_visit_spine(); pop=load_pop_classification(); domains=load_domain_flags(); raw=load_selected_raw_columns(args.input)
    logger.info("[2/8] Building baseline comorbidity indicators")
    base,duplicates,n_pipe=build_baseline_comorbidity_dataset(raw,spine,pop)
    logger.info("[3/8] Writing baseline intermediate dataset"); write_intermediate_dataset(base,BASELINE_PATH)
    logger.info("[4/8] Estimating overall prevalence"); overall=summarize_overall_prevalence(base); overall.to_csv(TABLES_DIR/"07_comorbidities_overall.csv",index=False); summarize_overall_by_category(base).to_csv(TABLES_DIR/"07_comorbidities_overall_by_category.csv",index=False); create_dotplot(overall)
    logger.info("[5/8] Comparing prevalence across Pop 1-3"); by_pop=summarize_prevalence_by_pop(base,args.monte_carlo_replicates,args.random_seed); by_pop.to_csv(TABLES_DIR/"07_comorbidities_by_pop.csv",index=False); create_grouped_barplot(by_pop)
    logger.info("[6/8] Building longitudinal outcomes"); long=build_longitudinal_essdai_dataset(raw,spine,pop,domains,base); write_intermediate_dataset(long,LONGITUDINAL_PATH); severe=build_severe5_survival_dataset(long,base); write_intermediate_dataset(severe,SEVERE_PATH); new_domain,domain_audit=build_new_domain_survival_dataset(long,base); write_intermediate_dataset(new_domain,NEW_DOMAIN_PATH); write_intermediate_dataset(domain_audit,DOMAIN_AUDIT_PATH)
    logger.info("[7/8] Fitting progression models"); progression_rows=[]; diagnostics=[]
    for c in PROGRESSION_CONDITIONS:
        rows,diag=fit_mixed_model(long,c); progression_rows.extend(rows); diagnostics.append(diag)
        row,diag=fit_cox_model(severe,c,"severe5_event","Progression to ESSDAI >=5",args.minimum_events); progression_rows.append(row); diagnostics.append(diag)
        row,diag=fit_cox_model(new_domain,c,"new_domain_event","New ESSDAI-domain involvement",args.minimum_events); progression_rows.append(row); diagnostics.append(diag)
    progression=pd.DataFrame(progression_rows); progression["fdr_bh_q_value"]=progression.groupby(["outcome","estimand"])["p_value"].transform(apply_fdr); progression.to_csv(TABLES_DIR/"07_comorbidities_progression.csv",index=False)
    logger.info("[8/8] Writing tables, figures, and QC"); create_progression_forestplot(progression); qc_extra=run_qc_checks(base,long,severe,new_domain,domain_audit)
    missingness_table(base).to_csv(QC_DIR/"07_comorbidities_missingness.csv",index=False); source_mapping(raw.columns).to_csv(QC_DIR/"07_comorbidities_source_mapping.csv",index=False); pd.DataFrame(diagnostics).to_csv(QC_DIR/"07_comorbidities_model_diagnostics.csv",index=False); duplicates.to_csv(QC_DIR/"07_comorbidities_patient_duplicates.csv",index=False)
    pd.DataFrame([{"condition":k,"availability_status":"unavailable","reason":v} for k,v in UNAVAILABLE.items()]).to_csv(QC_DIR/"07_comorbidities_unavailable_conditions.csv",index=False)
    pd.DataFrame(_UNRECOGNIZED,columns=["source_column","original_value","row_index"]).drop_duplicates().to_csv(QC_DIR/"07_comorbidities_unrecognized_values.csv",index=False)
    both=long[["essdai_total_recoded","essdai_total_raw_qc"]].dropna(); diff=both["essdai_total_recoded"]-both["essdai_total_raw_qc"]
    pop_counts=base["baseline_pop"].fillna("Unclassifiable").value_counts(); classifiable=int(base["baseline_pop"].isin(["Pop1","Pop2","Pop3"]).sum()); with_followup=int(long.loc[(long.visit_number>0)&long.essdai_total_recoded.notna(),"patient_id"].nunique())
    qc={"input_path":str(args.input),"input_modification_time":datetime.fromtimestamp(args.input.stat().st_mtime,timezone.utc).isoformat(),"script_version":SCRIPT_VERSION,"run_timestamp":datetime.now(timezone.utc).isoformat(),"random_seed":args.random_seed,"n_input_rows":len(raw),"n_input_patients":int(raw[PATIENT_ID_COL].nunique()),"n_canonical_visits":len(spine),"n_baseline_patients":len(base),"n_duplicate_patient_dates":len(duplicates),"n_pipe_delimited_visit_dates":n_pipe,"n_pop_classifiable":classifiable,"n_pop_unclassifiable":len(base)-classifiable,"pop_counts":pop_counts.to_dict(),"n_with_followup_essdai":with_followup,"n_at_risk_severe5":len(severe),"n_severe5_events":int(severe.get("severe5_event",pd.Series(dtype=int)).sum()),"n_at_risk_new_domain":len(new_domain),"n_new_domain_events":int(new_domain.get("new_domain_event",pd.Series(dtype=int)).sum()),"severe_threshold_used":SEVERE_THRESHOLD,"essdai_primary_column":ESSDAI_PRIMARY_COL,"essdai_raw_qc_column":ESSDAI_RAW_QC_COL,"upstream_files_used":[str(p) for p in UPSTREAM],"upstream_file_timestamps":timestamps,"essdai_reconciliation":{"n_both":len(both),"n_concordant":int(diff.eq(0).sum()),"n_discordant":int(diff.ne(0).sum()),"mean_difference":float(diff.mean()) if len(diff) else None,"median_difference":float(diff.median()) if len(diff) else None,"maximum_absolute_difference":float(diff.abs().max()) if len(diff) else None},"warnings":["The deployed extract has only essdai__essdai_total_score; it is used as the primary longitudinal ESSDAI source and duplicated in the raw-QC compatibility field."],**qc_extra}
    (QC_DIR/"07_comorbidities_qc.json").write_text(json.dumps(qc,indent=2,default=str)+"\n")
    claim=overall.head(3); ild=float(overall.loc[overall.condition.eq("ild"),"pct_total_cohort"].iloc[0]); logger.info("Claim: %s was the most prevalent baseline comorbidity (%.1f%%), followed by %s (%.1f%%) and %s (%.1f%%). Interstitial lung disease was present in %.1f%% of patients. Empty condition fields were coded as absent.",claim.iloc[0].display_label,claim.iloc[0].pct_total_cohort,claim.iloc[1].display_label,claim.iloc[1].pct_total_cohort,claim.iloc[2].display_label,claim.iloc[2].pct_total_cohort,ild)
    fitted=int(progression.model_status.isin(["fitted","reduced_adjustment"]).sum()); not_est=len(progression)-fitted
    print(f"Total baseline patients: {len(base)}\nClassifiable Pop patients: {classifiable}\nPatients with follow-up ESSDAI: {with_followup}\nESSDAI >=5 progression events: {qc['n_severe5_events']}\nNew-domain events: {qc['n_new_domain_events']}\nNumber of models fitted: {fitted}\nNumber of models not estimable: {not_est}\nGenerated files:")
    for path in outputs+[QC_DIR/"07_comorbidities_qc.json",QC_DIR/"07_comorbidities_missingness.csv",QC_DIR/"07_comorbidities_source_mapping.csv",QC_DIR/"07_comorbidities_model_diagnostics.csv",QC_DIR/"07_comorbidities_patient_duplicates.csv",QC_DIR/"07_comorbidities_unavailable_conditions.csv",QC_DIR/"07_comorbidities_unrecognized_values.csv",LOG_PATH]: print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

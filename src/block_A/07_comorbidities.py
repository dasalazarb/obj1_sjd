#!/usr/bin/env python3
"""SECTION 5 — Comorbidity burden and disease progression.

Single reproducible script for baseline comorbidity prevalence, Pop1-Pop3
comparisons, and longitudinal progression analyses using canonical upstream
visit, population, and ESSDAI-domain derivations.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from lifelines import CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:  # clear runtime branch; descriptive and survival rows marked not run
    CoxPHFitter = None
    LIFELINES_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402
import config  # noqa: E402
from src.derivations.visit_dates import add_parsed_visit_dates  # noqa: E402

SCRIPT_VERSION = "2026-08-05.section5.v1"
PATIENT_ID_COL = "ids__patient_record_number"
VISIT_DATE_COL = "ids__visit_date"
AGE_COL = "ids__age_at_visit"
SEX_COL = "ids__sex"
ESSDAI_RAW_QC_COL = "essdai__essdai_total_score"
ESSDAI_CANONICAL_COL = "essdai_total"
SEVERE_ACTIVITY_THRESHOLD_SECTION5 = 14
FIGURES_DIR = common.OUTPUTS_DIR / "figures" / "blockA"
LOG_PATH = common.OUTPUTS_DIR / "logs" / "07_comorbidities.log"
TABLES_DIR = common.OUTPUTS_DIR / "tables" / "blockA"
QC_DIR = common.OUTPUTS_DIR / "qc" / "blockA"

DOMAIN_COLS = [
    "eg_constitutional_active", "eg_lymphadenopathy_active", "eg_articular_active",
    "eg_cutaneous_active", "eg_pulmonary_active", "eg_renal_active", "eg_muscular_active",
    "eg_pns_active", "eg_cns_active", "eg_hematologic_active",
]
DOMAIN_LABELS = {c: c.replace("eg_", "").replace("_active", "") for c in DOMAIN_COLS}
SENSITIVITY_DOMAIN_COL = "eg_biological_active"

@dataclass(frozen=True)
class ConditionSpec:
    name: str
    label: str
    general: str
    history: str
    confirmed: str
    family: str = "rheumatological_comorbidities__"
    clinical_group: str = "other rheumatological condition or manifestation"
    temporal_interpretation: str = "current status requires explicit confirmation; history and uncertain states are not current disease"
    derivation_rule: str = "confirmed_present > history_only > status_uncertain > not_documented > missing; no OR across general/history/confirmed"
    derived_state: str = "mutually exclusive condition status"
    allowed_analysis: str = "confirmed_present eligible for baseline present-condition summaries and models; other states descriptive only"
    model_inclusion: str = "confirmed_present only"
    justification: str = "Rheumatological comorbidity form distinguishes current confirmation from history and uncertain general documentation."
    specify: str = ""
    notes: str = ""

@dataclass(frozen=True)
class HistoricalConditionSpec:
    name: str
    label: str
    source: str
    family: str
    clinical_group: str
    temporal_interpretation: str
    derivation_rule: str = "documented antecedent if source is positive; missing remains missing unless evaluability is documented"
    derived_state: str = "documented_history"
    allowed_analysis: str = "descriptive only: documented n/N, missingness, counts, and most frequently documented histories"
    model_inclusion: str = "excluded from all prevalence/incidence/longitudinal/predictive models"
    justification: str = "Historical source does not establish current presence or incident onset during follow-up."
    ambiguous: str = ""

PAST_MEDICAL_HISTORY_CONDITIONS = [
    HistoricalConditionSpec("pmh_lung_disease", "Lung disease history", "past_medical_history__respiratory_hx_lung", "past_medical_history__", "Respiratory", "Documented past medical history; not active ILD."),
    HistoricalConditionSpec("pmh_lung_fibrosis", "Lung fibrosis history", "past_medical_history__resp_hx_lung_fibro", "past_medical_history__", "Respiratory", "Documented past medical history; not active fibrosis."),
    HistoricalConditionSpec("pmh_bronchitis", "Bronchitis history", "past_medical_history__respiratory_hx_bronchitis", "past_medical_history__", "Respiratory", "Documented past medical history."),
    HistoricalConditionSpec("pmh_asthma", "Asthma history", "past_medical_history__asthma", "past_medical_history__", "Respiratory", "Documented past medical history."),
    HistoricalConditionSpec("pmh_copd", "COPD history", "past_medical_history__respiratory_hx_copd", "past_medical_history__", "Respiratory", "Documented past medical history."),
    HistoricalConditionSpec("pmh_pulmonary_hypertension", "Pulmonary hypertension history", "past_medical_history__resp_hx_pulm_hyper", "past_medical_history__", "Respiratory", "Documented past medical history."),
    HistoricalConditionSpec("pmh_pulmonary_embolism", "Pulmonary embolism history", "past_medical_history__pulmonary_embolism", "past_medical_history__", "Respiratory", "Documented past medical history."),
    HistoricalConditionSpec("pmh_thyroid_disease", "Thyroid disease history", "past_medical_history__thyroid_disease", "past_medical_history__", "Endocrine", "Documented past medical history; not combined with Sjögren thyroiditis history."),
    HistoricalConditionSpec("pmh_diabetes_type_1", "Type 1 diabetes history", "past_medical_history__endcrn_hx_mellitus_i", "past_medical_history__", "Endocrine", "Documented past medical history."),
    HistoricalConditionSpec("pmh_diabetes_type_2", "Type 2 diabetes history", "past_medical_history__endcrn_hx_mellitus_ii", "past_medical_history__", "Endocrine", "Documented past medical history."),
    HistoricalConditionSpec("pmh_depression", "Depression history", "past_medical_history__neuro_hx_depression", "past_medical_history__", "Neurological and psychiatric", "Documented past medical history; not combined with current depressive symptoms."),
    HistoricalConditionSpec("pmh_neuropathy", "Neuropathy history", "past_medical_history__neuro_hx_neuropathy", "past_medical_history__", "Neurological and psychiatric", "Documented past medical history; not current neurologic domain activity."),
    HistoricalConditionSpec("pmh_multiple_sclerosis", "Multiple sclerosis history", "past_medical_history__neuro_hx_mult_sclerosis", "past_medical_history__", "Neurological and psychiatric", "Documented past medical history."),
    HistoricalConditionSpec("pmh_seizures", "Seizure history", "past_medical_history__neuro_seizrs", "past_medical_history__", "Neurological and psychiatric", "Documented past medical history."),
    HistoricalConditionSpec("pmh_cva", "Cerebrovascular accident history", "past_medical_history__cva", "past_medical_history__", "Neurological and psychiatric", "Documented past medical history."),
    HistoricalConditionSpec("pmh_renal_tubular_acidosis", "Renal tubular acidosis history", "past_medical_history__renal_hx_tubular_acid", "past_medical_history__", "Renal and urinary", "Documented past medical history."),
    HistoricalConditionSpec("pmh_glomerulonephritis", "Glomerulonephritis history", "past_medical_history__renal_hx_glomer", "past_medical_history__", "Renal and urinary", "Documented past medical history."),
    HistoricalConditionSpec("pmh_nephrotic_syndrome", "Nephrotic syndrome history", "past_medical_history__renal_hx_nephrotic", "past_medical_history__", "Renal and urinary", "Documented past medical history."),
    HistoricalConditionSpec("pmh_kidney_stones", "Kidney stones history", "past_medical_history__renal_hx_kidney_stones", "past_medical_history__", "Renal and urinary", "Documented past medical history."),
    HistoricalConditionSpec("pmh_interstitial_cystitis", "Interstitial cystitis history", "past_medical_history__interstitial_cyst", "past_medical_history__", "Renal and urinary", "Documented past medical history."),
    HistoricalConditionSpec("pmh_autoimmune_hepatitis", "Autoimmune hepatitis history", "past_medical_history__gi_hx_auto_hepat", "past_medical_history__", "Gastrointestinal and hepatobiliary", "Documented past medical history."),
    HistoricalConditionSpec("pmh_cirrhosis", "Cirrhosis history", "past_medical_history__gi_hx_cirrhosis", "past_medical_history__", "Gastrointestinal and hepatobiliary", "Documented past medical history."),
    HistoricalConditionSpec("pmh_sclerosing_cholangitis", "Sclerosing cholangitis history", "past_medical_history__gi_hx_sclerosing", "past_medical_history__", "Gastrointestinal and hepatobiliary", "Documented past medical history."),
    HistoricalConditionSpec("pmh_inflammatory_bowel", "Inflammatory bowel disease history", "past_medical_history__inflam_bowel", "past_medical_history__", "Gastrointestinal and hepatobiliary", "Documented past medical history; separate from rheumatological immune-mediated IBD."),
    HistoricalConditionSpec("pmh_pancreatitis", "Pancreatitis history", "past_medical_history__pancreatitis", "past_medical_history__", "Gastrointestinal and hepatobiliary", "Documented past medical history."),
    HistoricalConditionSpec("pmh_vasculitis", "Vasculitis history", "past_medical_history__vasculitis", "past_medical_history__", "Other medical conditions", "Documented past medical history; not current vasculitis activity."),
    HistoricalConditionSpec("pmh_autonomic_dysfunction", "Autonomic dysfunction history", "past_medical_history__dysfunctn_autonomic", "past_medical_history__", "Other medical conditions", "Documented past medical history."),
    HistoricalConditionSpec("pmh_chronic_fatigue_syndrome", "Chronic fatigue syndrome history", "past_medical_history__chronic_fatigue_syndrome_dx", "past_medical_history__", "Other medical conditions", "Documented past medical history."),
    HistoricalConditionSpec("pmh_psoriasis", "Psoriasis history", "past_medical_history__cutaneous_hx_psoriasis", "past_medical_history__", "Cutaneous", "Documented past medical history."),
    HistoricalConditionSpec("pmh_vitiligo", "Vitiligo history", "past_medical_history__cutaneous_hx_vitiligo", "past_medical_history__", "Cutaneous", "Documented past medical history."),
    HistoricalConditionSpec("pmh_coronary_artery_disease", "Coronary artery disease history", "past_medical_history__cardio_hx_cad", "past_medical_history__", "Cardiovascular", "Documented past medical history."),
    HistoricalConditionSpec("pmh_myocardial_infarction", "Myocardial infarction history", "past_medical_history__cardio_hx_mycrdl", "past_medical_history__", "Cardiovascular", "Documented past medical history."),
    HistoricalConditionSpec("pmh_pericarditis", "Pericarditis history", "past_medical_history__cardio_hx_pericard", "past_medical_history__", "Cardiovascular", "Documented past medical history."),
    HistoricalConditionSpec("pmh_pvd", "Peripheral vascular disease history", "past_medical_history__cardio_hx_pvd", "past_medical_history__", "Cardiovascular", "Documented past medical history."),
    HistoricalConditionSpec("pmh_valve_disease", "Valve disease history", "past_medical_history__cardio_valve_disease", "past_medical_history__", "Cardiovascular", "Documented past medical history."),
    HistoricalConditionSpec("pmh_hypertension", "Hypertension history", "past_medical_history__hypertension_", "past_medical_history__", "Cardiovascular", "Documented past medical history."),
    HistoricalConditionSpec("pmh_hyperlipidemia", "Hyperlipidemia history", "past_medical_history__hyperlipidemia_", "past_medical_history__", "Cardiovascular", "Documented past medical history."),
    HistoricalConditionSpec("pmh_breast_cancer", "Breast cancer history", "past_medical_history__malignancy_hx_breast_ca", "past_medical_history__", "Malignancies", "Documented past medical history."),
    HistoricalConditionSpec("pmh_colon_cancer", "Colon cancer history", "past_medical_history__malignancy_hx_colon_ca", "past_medical_history__", "Malignancies", "Documented past medical history."),
    HistoricalConditionSpec("pmh_head_neck_cancer", "Head/neck cancer history", "past_medical_history__malignancy_hx_head_ca", "past_medical_history__", "Malignancies", "Documented past medical history."),
    HistoricalConditionSpec("pmh_lung_cancer", "Lung cancer history", "past_medical_history__malignancy_hx_lung_ca", "past_medical_history__", "Malignancies", "Documented past medical history."),
    HistoricalConditionSpec("pmh_lymphoma", "Lymphoma history", "past_medical_history__malignancy_hx_lymphoma", "past_medical_history__", "Malignancies", "Documented past medical history; not combined with Sjögren history or longitudinal lymphoma events."),
    HistoricalConditionSpec("pmh_thyroid_cancer", "Thyroid cancer history", "past_medical_history__malignancy_hx_thyroid_ca", "past_medical_history__", "Malignancies", "Documented past medical history."),
    HistoricalConditionSpec("pmh_other_malignancy", "Other malignancy history", "past_medical_history__malignancy_hx_other", "past_medical_history__", "Malignancies", "Documented past medical history; requires description for subtype interpretation.", ambiguous="Other malignancy field needs manual review of specify text before subtype use."),
]

SJOGREN_HISTORY_MANIFESTATIONS = [
    HistoricalConditionSpec(c.replace("sjogren's_syndrome_history__", "sjh_").replace("_", " ").strip().replace(" ", "_"), c.split("__",1)[1].replace("_", " ").strip().title(), c, "sjogren's_syndrome_history__", "Documented Sjögren-related history", "Retrospectively documented Sjögren-related history; not incident/current activity.")
    for c in ["sjogren's_syndrome_history__liver_", "sjogren's_syndrome_history__pulmonary", "sjogren's_syndrome_history__skin_", "sjogren's_syndrome_history__fatigue", "sjogren's_syndrome_history__autonomic_dys", "sjogren's_syndrome_history__dry_eye", "sjogren's_syndrome_history__dry_mouth", "sjogren's_syndrome_history__dry_othr", "sjogren's_syndrome_history__neuro_cran_periph", "sjogren's_syndrome_history__raynaud_phenom", "sjogren's_syndrome_history__vasculitis", "sjogren's_syndrome_history__arthritis", "sjogren's_syndrome_history__cns_cognitive", "sjogren's_syndrome_history__gland_swell", "sjogren's_syndrome_history__myositis_myalgia", "sjogren's_syndrome_history__pancreatitis", "sjogren's_syndrome_history__renal", "sjogren's_syndrome_history__interstitial_cyst", "sjogren's_syndrome_history__cholecystitis", "sjogren's_syndrome_history__lymphoma", "sjogren's_syndrome_history__thyroiditis", "sjogren's_syndrome_history__chemosensory", "sjogren's_syndrome_history__non_sicca_other", "sjogren's_syndrome_history__sjogrens_dx"]
]
SJOGREN_HISTORY_DATE_FIELDS = ["sjogren's_syndrome_history__sjogrens_dx_date", "sjogren's_syndrome_history__dry_eye_date_start", "sjogren's_syndrome_history__dry_mouth_date_start", "sjogren's_syndrome_history__dry_othr_date_start", "sjogren's_syndrome_history__dry_othr_spfy"]

RHEUMATOLOGICAL_CONDITIONS = [
    ConditionSpec("raynaud", "Raynaud's phenomenon", "rheumatological_comorbidities__integ_raynds", "rheumatological_comorbidities__integ_raynds_hx", "rheumatological_comorbidities__integ_raynds_confirm", clinical_group="rheumatological manifestation"),
    ConditionSpec("sle", "Systemic lupus erythematosus", "rheumatological_comorbidities__sle1", "rheumatological_comorbidities__sle_hx", "rheumatological_comorbidities__sle_confirmed", clinical_group="systemic autoimmune/inflammatory disease"),
    ConditionSpec("rheumatoid_arthritis", "Rheumatoid arthritis", "rheumatological_comorbidities__ra", "rheumatological_comorbidities__ra_hx", "rheumatological_comorbidities__ra_confirm", clinical_group="systemic autoimmune/inflammatory disease"),
    ConditionSpec("systemic_sclerosis", "Systemic sclerosis", "rheumatological_comorbidities__systemic_sclerosis", "rheumatological_comorbidities__systmc_sclerosis_hx", "rheumatological_comorbidities__systmc_sclerosis_confirm", clinical_group="systemic autoimmune/inflammatory disease"),
    ConditionSpec("polymyositis", "Polymyositis", "rheumatological_comorbidities__polymyositis", "rheumatological_comorbidities__polymyositis_hx", "rheumatological_comorbidities__polymyositis_confirm", clinical_group="systemic autoimmune/inflammatory disease"),
    ConditionSpec("dermatomyositis", "Dermatomyositis", "rheumatological_comorbidities__dermatomyositis", "rheumatological_comorbidities__dermatomyositis_hx", "rheumatological_comorbidities__dermatomyositis_confirm", clinical_group="systemic autoimmune/inflammatory disease"),
    ConditionSpec("mixed_connective_tissue_disease", "Mixed connective tissue disease", "rheumatological_comorbidities__mixed_connective_tissue_disease", "rheumatological_comorbidities__mixed_connect_tissue_hx", "rheumatological_comorbidities__mixed_connect_tissue_confirm", clinical_group="systemic autoimmune/inflammatory disease"),
    ConditionSpec("antiphospholipid_syndrome", "Antiphospholipid syndrome", "rheumatological_comorbidities__antiphospholipid_syndrome", "rheumatological_comorbidities__antiphospholipid_syn_hx", "rheumatological_comorbidities__antiphospholipid_syn_confirm", clinical_group="systemic autoimmune/inflammatory disease"),
    ConditionSpec("primary_biliary_cirrhosis", "Primary biliary cirrhosis", "rheumatological_comorbidities__primary_billiary_cirrhosis", "rheumatological_comorbidities__prim_billiary_cirrhosis_hx", "rheumatological_comorbidities__prim_billiary_cirrhosis_confirm", clinical_group="other immune-mediated condition"),
    ConditionSpec("cryoglobulinemia", "Cryoglobulinemia", "rheumatological_comorbidities__cryoglobulinemia", "rheumatological_comorbidities__cryoglobulinemia_hx", "rheumatological_comorbidities__cryoglobulinemia_confirm"),
    ConditionSpec("fibromyalgia", "Fibromyalgia", "rheumatological_comorbidities__fibromyalgia1", "rheumatological_comorbidities__fibromyalgia1_hx", "rheumatological_comorbidities__fibromyalgia1_confirm", clinical_group="non-systemic rheumatological condition"),
    ConditionSpec("osteoporosis", "Osteoporosis", "rheumatological_comorbidities__osteoporosis1", "rheumatological_comorbidities__osteoporosis1_hx", "rheumatological_comorbidities__osteoporosis1_confirm", clinical_group="non-systemic rheumatological condition"),
    ConditionSpec("osteopenia", "Osteopenia", "rheumatological_comorbidities__osteopenia", "rheumatological_comorbidities__osteopenia_hx", "rheumatological_comorbidities__osteopenia_confirm", clinical_group="non-systemic rheumatological condition"),
    ConditionSpec("osteoarthritis", "Osteoarthritis", "rheumatological_comorbidities__osteoarthritis", "rheumatological_comorbidities__osteoarthritis_hx", "rheumatological_comorbidities__osteoarthritis_confirm", clinical_group="non-systemic rheumatological condition"),
    ConditionSpec("sarcoidosis", "Sarcoidosis", "rheumatological_comorbidities__sarcoidosis", "rheumatological_comorbidities__sarcoidosis_hx", "rheumatological_comorbidities__sarcoidosis_confirm"),
    ConditionSpec("crystalline_arthropathy", "Crystalline arthropathy", "rheumatological_comorbidities__crystalline_arthropathy", "rheumatological_comorbidities__crystalline_arthropathy_hx", "rheumatological_comorbidities__crystalline_arthro_confirm"),
    ConditionSpec("inflammatory_bowel_disease", "Inflammatory bowel disease", "rheumatological_comorbidities__inflam_bowel", "rheumatological_comorbidities__inflam_bowel_hx", "rheumatological_comorbidities__inflam_bowel_confirm", clinical_group="other immune-mediated condition", specify="rheumatological_comorbidities__inflam_bowel_spfy"),
    ConditionSpec("rheumatological_other", "Other rheumatological condition", "rheumatological_comorbidities__rheumatological_other", "rheumatological_comorbidities__rheumatological_other_hx", "rheumatological_comorbidities__rheumatological_other_confirm", clinical_group="manual review required", specify="rheumatological_comorbidities__rheumatological_specify", model_inclusion="excluded pending manual review", notes="Other requires manual review before categorization."),
]
CONDITIONS = [c for c in RHEUMATOLOGICAL_CONDITIONS if c.clinical_group != "other immune-mediated condition" and c.name != "rheumatological_other"]
EXISTING_AND_RHEUMATOLOGICAL_CONDITIONS = CONDITIONS
CONDITION_NAMES = [c.name for c in CONDITIONS]
OTHER_IMMUNE_CONDITIONS = [c for c in RHEUMATOLOGICAL_CONDITIONS if c.clinical_group == "other immune-mediated condition"]
UNAVAILABLE_CONDITIONS = {}
UNRECOGNIZED: list[dict[str, Any]] = []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=common.DEFAULT_ANALYTIC_DATASET)
    p.add_argument("--rebuild-upstream", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--random-seed", type=int, default=20260728)
    p.add_argument("--monte-carlo-replicates", type=int, default=100_000)
    p.add_argument("--minimum-events", type=int, default=10)
    return p.parse_args()


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)])


def ensure_directories() -> None:
    common.ensure_output_dirs(); FIGURES_DIR.mkdir(parents=True, exist_ok=True); LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def check_upstream_artifacts(input_path: Path, rebuild: bool) -> dict[str, str]:
    required = {common.VISIT_SPINE_PARQUET: "src/00_build_visit_spine.py", common.POP_LONGITUDINAL_PARQUET: "src/block_A/01_pop_distribution.py", common.OVERLAP_LONGITUDINAL_PARQUET: "src/block_A/06_overlap_glandular_followup.py"}
    stamps = {}
    for path, script in required.items():
        if (not path.exists()) and rebuild:
            logging.info("Rebuilding missing upstream %s via %s", path, script); subprocess.run([sys.executable, script], cwd=PROJECT_ROOT, check=True)
        if not path.exists():
            raise FileNotFoundError(f"Required upstream file missing: {path}. Generate it with {script} or rerun with --rebuild-upstream.")
        if path.stat().st_mtime < input_path.stat().st_mtime:
            msg = f"Upstream file {path} is older than input {input_path}; reuse requested. Use --rebuild-upstream after regenerating if needed."
            logging.warning(msg)
        stamps[str(path)] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    return stamps


def available_columns(path: Path) -> set[str]:
    """Return Parquet column names without reading row data."""
    import pyarrow.parquet as pq
    return set(pq.read_schema(path).names)


def load_selected_raw_columns(path: Path) -> pd.DataFrame:
    needed = {PATIENT_ID_COL, VISIT_DATE_COL, AGE_COL, SEX_COL, ESSDAI_RAW_QC_COL,
              "past_medical_history__thyroid_disease_spfy", "past_medical_history__neuro_hx_neuropathy_spfy",
              "visit_summary_-_2016_classification_criteria__autoantibodies", "visit_summary_form__autoantibodies"}
    for c in RHEUMATOLOGICAL_CONDITIONS:
        needed.update([c.general, c.history, c.confirmed])
        if c.specify: needed.add(c.specify)
    for c in PAST_MEDICAL_HISTORY_CONDITIONS + SJOGREN_HISTORY_MANIFESTATIONS: needed.add(c.source)
    needed.update(SJOGREN_HISTORY_DATE_FIELDS)
    cols = sorted(needed & available_columns(path))
    missing = sorted(needed - set(cols))
    if missing: logging.warning("Selected raw columns unavailable: %s", ", ".join(missing))
    df = pd.read_parquet(path, columns=cols)
    return add_parsed_visit_dates(df, PATIENT_ID_COL, VISIT_DATE_COL)


def load_visit_spine() -> pd.DataFrame:
    cols = ["patient_id", "visit_id", "visit_date", "visit_number", "observed_baseline_date", "time_since_observed_baseline_days", "time_since_observed_baseline_years", "age_at_visit", "sex", "interval_name"]
    have = available_columns(common.VISIT_SPINE_PARQUET)
    required = {"patient_id", "visit_id", "visit_date", "visit_number", "observed_baseline_date", "time_since_observed_baseline_years", "age_at_visit", "sex"}
    missing = sorted(required - have)
    if missing:
        raise ValueError(f"Canonical visit spine is missing required columns: {', '.join(missing)}")
    return pd.read_parquet(common.VISIT_SPINE_PARQUET, columns=[c for c in cols if c in have])


def load_pop_classification() -> pd.DataFrame:
    cols = ["patient_id", "visit_id", "visit_date", "visit_number", ESSDAI_CANONICAL_COL, "pop_status", "baseline_pop_status"]
    have = available_columns(common.POP_LONGITUDINAL_PARQUET)
    missing = sorted(set(cols) - have)
    if missing:
        raise ValueError(f"Population longitudinal data are missing required columns: {', '.join(missing)}")
    return pd.read_parquet(common.POP_LONGITUDINAL_PARQUET, columns=cols)


def load_domain_flags() -> pd.DataFrame:
    cols = ["patient_id", "visit_id", "visit_date", "visit_number"] + DOMAIN_COLS + [SENSITIVITY_DOMAIN_COL, "n_extraglandular_domains_active"]
    have = available_columns(common.OVERLAP_LONGITUDINAL_PARQUET)
    required = {"patient_id", "visit_id", "visit_date", "visit_number", *DOMAIN_COLS}
    missing = sorted(required - have)
    if missing:
        raise ValueError(f"Overlap longitudinal data are missing required columns: {', '.join(missing)}")
    return pd.read_parquet(common.OVERLAP_LONGITUDINAL_PARQUET, columns=[c for c in cols if c in have])


def normalize_binary_flag(series: pd.Series) -> pd.Series:
    positive = {"1", "1.0", "true", "yes", "y", "positive", "present", "confirmed", "history"}
    negative = {"0", "0.0", "false", "no", "n", "negative", "absent"}
    missing = {str(x).strip().lower() for x in config.MISSING_STRINGS}
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    for idx, val in series.items():
        if val is None or pd.isna(val): continue
        text = str(val).strip().lower()
        if text in missing: continue
        if text in positive: out.loc[idx] = True
        elif text in negative: out.loc[idx] = False
        else: UNRECOGNIZED.append({"column": series.name, "value": str(val), "n": 1})
    return out


def nullable_or(df: pd.DataFrame) -> pd.Series:
    if df.empty: return pd.Series(pd.NA, index=df.index, dtype="boolean")
    any_true = df.eq(True).any(axis=1)
    any_false = df.eq(False).any(axis=1)
    return pd.Series(np.where(any_true, True, np.where(any_false, False, pd.NA)), index=df.index, dtype="boolean")


def collapse_same_patient_date(df: pd.DataFrame, bool_cols: list[str], text_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dup = df.groupby(["patient_id", "visit_date"], dropna=False).size().reset_index(name="n_source_rows")
    dup = dup[dup.n_source_rows > 1]
    agg = {c: (lambda s: True if (s == True).any() else (False if (s == False).any() else pd.NA)) for c in bool_cols}
    for c in text_cols:
        agg[c] = lambda s: " | ".join(sorted(set(str(x) for x in s.dropna() if str(x).strip()))) or pd.NA
    collapsed = df.groupby(["patient_id", "visit_date"], as_index=False, dropna=False).agg(agg)
    for c in bool_cols: collapsed[c] = collapsed[c].astype("boolean")
    return collapsed, dup


def derive_condition_status(general: pd.Series, history: pd.Series, confirmed: pd.Series, evaluated: pd.Series) -> pd.Series:
    status = pd.Series("missing", index=general.index, dtype="object")
    status[evaluated.eq(True)] = "not_documented"
    status[general.eq(True)] = "status_uncertain"
    status[history.eq(True)] = "history_only"
    status[confirmed.eq(True)] = "confirmed_present"
    return status


def source_mapping_table() -> pd.DataFrame:
    rows=[]
    for spec in PAST_MEDICAL_HISTORY_CONDITIONS + SJOGREN_HISTORY_MANIFESTATIONS:
        rows.append(spec.__dict__ | {"source_variable": spec.source, "eligible_predictor": False})
    for spec in RHEUMATOLOGICAL_CONDITIONS:
        rows.append({"name": spec.name, "label": spec.label, "source_variable": ";".join([spec.general, spec.history, spec.confirmed]), "family": spec.family, "clinical_group": spec.clinical_group, "temporal_interpretation": spec.temporal_interpretation, "derivation_rule": spec.derivation_rule, "derived_state": spec.derived_state, "allowed_analysis": spec.allowed_analysis, "model_inclusion": spec.model_inclusion, "justification": spec.justification, "ambiguous": spec.notes, "eligible_predictor": spec in CONDITIONS})
    return pd.DataFrame(rows)


def derive_comorbidity_indicators(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = raw[["patient_id", "visit_date"]].copy(); bool_cols=[]; conflict_rows=[]
    none_cols=[c for c in raw.columns if c.startswith("rheumatological_comorbidities__") and c.endswith("_none")]
    any_none = nullable_or(pd.DataFrame({c: normalize_binary_flag(raw[c]) for c in none_cols})) if none_cols else pd.Series(pd.NA,index=raw.index,dtype="boolean")
    for spec in RHEUMATOLOGICAL_CONDITIONS:
        vals={}
        for role,col in [("general",spec.general),("history",spec.history),("confirmed",spec.confirmed)]:
            vals[role]=normalize_binary_flag(raw[col]) if col in raw else pd.Series(pd.NA,index=raw.index,dtype="boolean")
            out[f"src__{col}"]=vals[role]; bool_cols.append(f"src__{col}")
        evaluated=pd.concat(vals.values(), axis=1).notna().any(axis=1)
        out[f"{spec.name}_status"] = derive_condition_status(vals["general"], vals["history"], vals["confirmed"], evaluated)
        status = out[f"{spec.name}_status"]
        out[spec.name] = pd.Series(np.where(status.eq("missing"), pd.NA, status.eq("confirmed_present")), index=out.index, dtype="boolean")
        out[f"{spec.name}_history_only"] = pd.Series(np.where(status.eq("missing"), pd.NA, status.eq("history_only")), index=out.index, dtype="boolean")
        out[f"{spec.name}_status_uncertain"] = pd.Series(np.where(status.eq("missing"), pd.NA, status.eq("status_uncertain")), index=out.index, dtype="boolean")
        bool_cols += [spec.name, f"{spec.name}_history_only", f"{spec.name}_status_uncertain"]
        bad = vals["confirmed"].eq(True) & vals["general"].eq(False)
        bad |= vals["history"].eq(True) & vals["confirmed"].eq(False)
        bad |= any_none.eq(True) & (vals["general"].eq(True) | vals["history"].eq(True) | vals["confirmed"].eq(True))
        for i in raw.index[bad.fillna(False)]:
            conflict_rows.append({"patient_id": raw.loc[i,"patient_id"], "visit_date": raw.loc[i,"visit_date"], "condition": spec.name, "general": vals["general"].loc[i], "history": vals["history"].loc[i], "confirmed": vals["confirmed"].loc[i], "none_positive": any_none.loc[i], "conflict_type": "contradictory rheumatological status fields"})
    for spec in PAST_MEDICAL_HISTORY_CONDITIONS + SJOGREN_HISTORY_MANIFESTATIONS:
        if spec.source in raw:
            out[spec.name] = normalize_binary_flag(raw[spec.source]); bool_cols.append(spec.name)
    collapsed, dup = collapse_same_patient_date(out, list(dict.fromkeys(bool_cols)), [])
    status_cols=[c for c in out.columns if c.endswith("_status")]
    if status_cols:
        status_latest = out.groupby(["patient_id","visit_date"], as_index=False, dropna=False)[status_cols].agg(lambda s: next((x for x in s if pd.notna(x)), "missing"))
        collapsed=collapsed.merge(status_latest,on=["patient_id","visit_date"],how="left")
    return collapsed, source_mapping_table(), pd.DataFrame(conflict_rows), dup


def calculate_wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n <= 0: return (np.nan, np.nan)
    z=1.959963984540054; p=k/n; den=1+z*z/n; cen=(p+z*z/(2*n))/den; half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return max(0, (cen-half)*100), min(100, (cen+half)*100)


def summarize_historical_family(baseline: pd.DataFrame, specs: list[HistoricalConditionSpec], family_label: str) -> pd.DataFrame:
    rows=[]; N=len(baseline)
    for spec in specs:
        ser = baseline[spec.name].astype("boolean") if spec.name in baseline else pd.Series(pd.NA,index=baseline.index,dtype="boolean")
        n_doc=int(ser.eq(True).sum()); nmiss=int(ser.isna().sum()); neval=int(ser.notna().sum())
        rows.append({"analytic_name":spec.name,"clinical_label":spec.label,"source_variable":spec.source,"source_family":spec.family,"clinical_group":spec.clinical_group,"temporal_interpretation":spec.temporal_interpretation,"n_total_patients":N,"n_evaluable":neval,"n_documented_history":n_doc,"n_missing":nmiss,"proportion_documented_among_evaluable":n_doc/neval if neval else np.nan,"percent_documented_among_evaluable":100*n_doc/neval if neval else np.nan,"allowed_analysis":spec.allowed_analysis,"model_inclusion":spec.model_inclusion,"justification":spec.justification,"ambiguous":spec.ambiguous,"summary_label":"Proportion with documented antecedent"})
    out=pd.DataFrame(rows).sort_values(["clinical_group","n_documented_history"], ascending=[True,False])
    if len(out): out["documented_history_count_variable"] = "documented_past_history_count" if family_label=="past" else "sjogren_history_manifestation_count"
    return out


def summarize_rheumatological_history(baseline: pd.DataFrame) -> pd.DataFrame:
    rows=[]; N=len(baseline)
    for spec in RHEUMATOLOGICAL_CONDITIONS:
        status = baseline.get(f"{spec.name}_status", pd.Series("missing", index=baseline.index))
        for state in ["history_only","status_uncertain","not_documented","missing"]:
            rows.append({"condition":spec.name,"display_label":spec.label,"clinical_group":spec.clinical_group,"state":state,"n_total_patients":N,"n_patients":int((status==state).sum()),"definition":"Mutually exclusive rheumatological form status; not interpreted as current confirmed disease.","model_inclusion":"excluded"})
    return pd.DataFrame(rows)


def summarize_other_immune_conditions(baseline: pd.DataFrame) -> pd.DataFrame:
    rows=[]; N=len(baseline)
    for spec in OTHER_IMMUNE_CONDITIONS:
        status=baseline.get(f"{spec.name}_status", pd.Series("missing", index=baseline.index))
        rows.append({"condition":spec.name,"display_label":spec.label,"clinical_group":"other immune-mediated condition","n_total_patients":N,"n_confirmed_present":int((status=="confirmed_present").sum()),"n_history_only":int((status=="history_only").sum()),"n_status_uncertain":int((status=="status_uncertain").sum()),"n_not_documented":int((status=="not_documented").sum()),"n_missing":int((status=="missing").sum()),"specify_field":spec.specify,"model_inclusion":"kept outside main rheumatological condition group"})
    return pd.DataFrame(rows)


def condition_status_sensitivity(baseline: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for spec in RHEUMATOLOGICAL_CONDITIONS:
        status=baseline.get(f"{spec.name}_status", pd.Series("missing", index=baseline.index))
        rows.append({"condition":spec.name,"confirmed_present_n":int((status=="confirmed_present").sum()),"history_only_n":int((status=="history_only").sum()),"status_uncertain_n":int((status=="status_uncertain").sum()),"confirmed_plus_history_sensitivity_n":int(status.isin(["confirmed_present","history_only"]).sum()),"confirmed_plus_uncertain_sensitivity_n":int(status.isin(["confirmed_present","status_uncertain"]).sum()),"primary_model_state":"confirmed_present only"})
    return pd.DataFrame(rows)

def summarize_overall_prevalence(baseline: pd.DataFrame) -> pd.DataFrame:
    rows=[]; N=len(baseline)
    for spec in CONDITIONS:
        s = baseline[spec.name].astype("boolean") if spec.name in baseline else pd.Series(pd.NA, index=baseline.index, dtype="boolean")
        npos=int((s==True).sum()); nmiss=int(s.isna().sum()); neval=N-nmiss; nneg=int((s==False).sum())
        lo,hi=calculate_wilson_ci(npos,N)
        rows.append({"condition":spec.name,"display_label":spec.label,"definition_type":"confirmed_present","source_columns":";".join([spec.confirmed]),"n_total_cohort":N,"n_evaluable":neval,"n_positive":npos,"n_negative":nneg,"n_missing":nmiss,"pct_total_cohort":100*npos/N if N else np.nan,"pct_among_evaluable":100*npos/neval if neval else np.nan,"ci95_low":lo,"ci95_high":hi,"availability_status":"available","notes":spec.notes})
    out=pd.DataFrame(rows).sort_values("pct_total_cohort", ascending=False).reset_index(drop=True); out["rank_by_prevalence"]=np.arange(1,len(out)+1); return out


def monte_carlo_chi2(table: np.ndarray, reps: int, seed: int) -> float:
    chi_obs = stats.chi2_contingency(table, correction=False)[0]; rng=np.random.default_rng(seed); total=table.sum(); rows=table.sum(axis=1); cols=table.sum(axis=0); ge=0
    probs=np.repeat(np.arange(table.shape[0]), rows) if total else []
    present=np.repeat([1,0], cols) if total else []
    for _ in range(reps):
        rng.shuffle(present); sim=np.zeros_like(table)
        start=0
        for i,r in enumerate(rows):
            vals=present[start:start+r]; start+=r; sim[i,0]=(vals==1).sum(); sim[i,1]=(vals==0).sum()
        if stats.chi2_contingency(sim, correction=False)[0] >= chi_obs: ge+=1
    return (ge+1)/(reps+1)


def calculate_or_and_fisher(a:int,b:int,c:int,d:int)->dict[str,Any]:
    table=np.array([[a,b],[c,d]]); _,p=stats.fisher_exact(table); zero=(table==0).any(); aa,bb,cc,dd=(table+0.5).ravel() if zero else table.ravel(); orv=(aa*dd)/(bb*cc) if bb*cc else np.nan; se=math.sqrt(sum(1/x for x in [aa,bb,cc,dd])); lo=math.exp(math.log(orv)-1.96*se) if orv>0 else np.nan; hi=math.exp(math.log(orv)+1.96*se) if orv>0 else np.nan
    return {"odds_ratio_pop2_vs_pop3":orv,"or_ci95_low":lo,"or_ci95_high":hi,"fisher_exact_p_value":p,"zero_cell_correction_used":bool(zero)}


def apply_fdr(pvals: Iterable[float]) -> list[float]:
    arr=np.array([np.nan if pd.isna(p) else p for p in pvals], dtype=float); out=np.full(len(arr), np.nan); m=~np.isnan(arr)
    if m.any(): out[m]=multipletests(arr[m], method="fdr_bh")[1]
    return out.tolist()


def summarize_prevalence_by_pop(baseline: pd.DataFrame, replicates:int=100000, seed:int=20260728)->pd.DataFrame:
    rows=[]; pops=["Pop1","Pop2","Pop3"]
    for spec in CONDITIONS:
        row={"condition":spec.name,"display_label":spec.label}; s=baseline[spec.name].astype("boolean"); tables=[]
        for pop in pops:
            ss=s[baseline["baseline_pop"].eq(pop)]; ev=ss.notna(); n=int((ss[ev]==True).sum()); N=int(ev.sum()); lo,hi=calculate_wilson_ci(n,N); key=pop.lower(); row.update({f"n_{key}":n,f"N_{key}":N,f"pct_{key}":100*n/N if N else np.nan,f"ci95_{key}_low":lo,f"ci95_{key}_high":hi}); tables.append([n, N-n])
        table=np.array(tables)
        if table.sum() == 0 or (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
            row.update({"global_test":"not estimable", "global_p_value":np.nan, "minimum_expected_cell":np.nan, "sparse_table_flag":True})
        else:
            try:
                chi,p,_,exp=stats.chi2_contingency(table, correction=False); mn=float(exp.min()); sparse=mn<5
                row.update({"global_test":"Monte Carlo chi-square" if sparse else "Chi-square", "global_p_value": monte_carlo_chi2(table, min(replicates,20000), seed) if sparse else p, "minimum_expected_cell":mn, "sparse_table_flag":sparse})
            except ValueError:
                row.update({"global_test":"not estimable", "global_p_value":np.nan, "minimum_expected_cell":np.nan, "sparse_table_flag":True})
        row.update(calculate_or_and_fisher(table[1,0], table[1,1], table[2,0], table[2,1]))
        row["interpretation_status"] = "clear_evidence" if (pd.notna(row["fisher_exact_p_value"]) and row["fisher_exact_p_value"]<0.05 and row["or_ci95_low"]>1) else "imprecise_or_no_clear_evidence"
        rows.append(row)
    out=pd.DataFrame(rows); out["fdr_bh_q_value"]=apply_fdr(out["global_p_value"]); return out


def build_baseline_comorbidity_dataset(raw_ind:pd.DataFrame, spine:pd.DataFrame, pop:pd.DataFrame)->pd.DataFrame:
    base_spine=spine[spine.visit_number.eq(0)].copy(); merged=base_spine.merge(raw_ind, on=["patient_id","visit_date"], how="left")
    pop_base=pop[pop.visit_number.eq(0)][["patient_id","visit_id","essdai_total","pop_status","baseline_pop_status"]].rename(columns={"essdai_total":"baseline_essdai","pop_status":"baseline_pop"})
    merged=merged.merge(pop_base, on=["patient_id","visit_id"], how="left")
    merged=merged.rename(columns={"visit_id":"baseline_visit_id","visit_date":"baseline_date","age_at_visit":"age_baseline"})
    merged["baseline_pop"] = merged["baseline_pop"].fillna(merged.get("baseline_pop_status"))
    for c in CONDITION_NAMES:
        if c not in merged: merged[c]=pd.Series(pd.NA,index=merged.index,dtype="boolean")
    merged["n_prespecified_comorbidities"] = merged[CONDITION_NAMES].eq(True).sum(axis=1)
    merged["n_comorbidities_evaluable"] = merged[CONDITION_NAMES].notna().sum(axis=1)
    merged["any_comorbidity"] = (merged["n_prespecified_comorbidities"]>0).astype("boolean")
    merged["two_or_more_comorbidities"] = (merged["n_prespecified_comorbidities"]>=2).astype("boolean")
    if len(merged)!=merged.patient_id.nunique(): raise ValueError("Baseline dataset is not one row per patient")
    return merged


def build_longitudinal_essdai_dataset(spine,pop,raw,baseline,domains):
    ess=raw[["patient_id","visit_date"]+[c for c in [ESSDAI_RAW_QC_COL] if c in raw]].drop_duplicates(["patient_id","visit_date"]).rename(columns={ESSDAI_RAW_QC_COL:"essdai_total_raw_qc"})
    df=spine.merge(ess,on=["patient_id","visit_date"],how="left").merge(pop[["patient_id","visit_id",ESSDAI_CANONICAL_COL,"pop_status"]],on=["patient_id","visit_id"],how="left",suffixes=("","_popfile"))
    df["essdai_total_recoded"] = pd.to_numeric(df[ESSDAI_CANONICAL_COL], errors="coerce")
    df["essdai_total_source"] = np.where(df["essdai_total_recoded"].notna(), "population_longitudinal__essdai_total", pd.NA)
    if "essdai_total_raw_qc" not in df:
        df["essdai_total_raw_qc"] = pd.Series(np.nan, index=df.index, dtype="float64")
    df=df.merge(domains,on=["patient_id","visit_id","visit_date","visit_number"],how="left")
    bcols=["patient_id","baseline_essdai","baseline_pop","age_baseline","sex"]+CONDITION_NAMES
    return df.merge(baseline[bcols],on="patient_id",how="left",suffixes=("","_baseline"))


def build_severe14_survival_dataset(longdf, baseline):
    rows=[]
    for pid,g in longdf.dropna(subset=["essdai_total_recoded"]).sort_values("visit_date").groupby("patient_id"):
        b=baseline[baseline.patient_id.eq(pid)].iloc[0]; base=float(b.baseline_essdai) if pd.notna(b.baseline_essdai) else np.nan
        if pd.isna(base) or base>=SEVERE_ACTIVITY_THRESHOLD_SECTION5: continue
        f=g[g.visit_number>0]
        if f.empty: continue
        ev=f[f.essdai_total_recoded>=SEVERE_ACTIVITY_THRESHOLD_SECTION5]
        ed=ev.visit_date.min() if not ev.empty else pd.NaT; last=f.visit_date.max(); end=ed if pd.notna(ed) else last
        rows.append({"patient_id":pid,"baseline_date":b.baseline_date,"last_evaluable_date":last,"event_date":ed,"followup_days":(end-b.baseline_date).days,"followup_years":(end-b.baseline_date).days/365.25,"severe14_event":int(pd.notna(ed)), **b[["baseline_essdai","baseline_pop","age_baseline","sex"]+CONDITION_NAMES].to_dict()})
    return pd.DataFrame(rows)


def build_new_domain_survival_dataset(longdf, baseline):
    rows=[]; audit=[]
    for pid,g in longdf.sort_values("visit_date").groupby("patient_id"):
        b=baseline[baseline.patient_id.eq(pid)].iloc[0]; bg=g[g.visit_number.eq(0)]; fg=g[g.visit_number>0]
        if bg.empty or fg.empty: continue
        events=[]; evaln=0; inact=0
        for col in DOMAIN_COLS:
            bv=bg.iloc[0].get(col, pd.NA)
            if pd.isna(bv): continue
            evaln+=1
            if bool(bv): continue
            inact+=1; hit=fg[fg[col].eq(True)]
            date=hit.visit_date.min() if not hit.empty else pd.NaT
            audit.append({"patient_id":pid,"domain":DOMAIN_LABELS[col],"baseline_active":False,"event_date":date})
            if pd.notna(date): events.append((date, DOMAIN_LABELS[col]))
        if inact==0: continue
        fd,name=min(events, default=(pd.NaT,pd.NA), key=lambda x:x[0] if pd.notna(x[0]) else pd.Timestamp.max); last=fg.visit_date.max(); end=fd if pd.notna(fd) else last
        rows.append({"patient_id":pid,"baseline_date":b.baseline_date,"first_new_domain_date":fd,"first_new_domain_name":name,"new_domain_event":int(pd.notna(fd)),"n_domains_inactive_at_baseline":inact,"n_domains_evaluable_at_baseline":evaln,"followup_days":(end-b.baseline_date).days,"followup_years":(end-b.baseline_date).days/365.25, **b[["baseline_essdai","baseline_pop","age_baseline","sex"]+CONDITION_NAMES].to_dict()})
    out=pd.DataFrame(rows); out.attrs["patient_domain_audit"]=pd.DataFrame(audit); return out


def _base_progression_row(spec,outcome,estimand):
    return {"comorbidity":spec.name,"display_label":spec.label,"outcome":outcome,"estimand":estimand,"model_type":"not run","effect_measure":"Not estimable","estimate":np.nan,"ci95_low":np.nan,"ci95_high":np.nan,"p_value":np.nan,"n_patients":0,"n_followup_observations":0,"n_events":np.nan,"n_complete_cases":0,"baseline_reference_group":"comorbidity absent","adjustment_covariates":"baseline_ESSDAI, baseline_pop, age_baseline, sex","time_scale":"years","threshold":np.nan,"model_converged":False,"proportional_hazards_p":np.nan,"sparse_event_flag":False,"model_status":"not_run","warning":"","interpretation":"Association estimate only; not causal."}


def fit_mixed_model(longdf, spec):
    row=_base_progression_row(spec,"ESSDAI trajectory","time × comorbidity"); df=longdf[longdf.visit_number.gt(0)].dropna(subset=["essdai_total_recoded",spec.name,"baseline_essdai","age_baseline"]).copy(); row.update(n_patients=df.patient_id.nunique(), n_followup_observations=len(df), n_complete_cases=len(df), effect_measure="Beta per year", threshold="NA")
    if df.patient_id.nunique()<5 or df[spec.name].nunique()<2: row.update(model_status="insufficient_data", warning="Too few patients or exposure variation"); return [row]
    df["exposed"]=df[spec.name].astype(int)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res=smf.mixedlm("essdai_total_recoded ~ time_since_observed_baseline_years*exposed + baseline_essdai + C(baseline_pop) + age_baseline + C(sex)", df, groups=df["patient_id"]).fit(reml=False, method="lbfgs", disp=False)
        term="time_since_observed_baseline_years:exposed"; est=res.params.get(term,np.nan); se=res.bse.get(term,np.nan)
        row.update(model_type="linear mixed model", estimate=est, ci95_low=est-1.96*se, ci95_high=est+1.96*se, p_value=res.pvalues.get(term,np.nan), model_converged=bool(res.converged), model_status="adjusted", warning="; ".join(str(x.message) for x in w))
    except Exception as e:
        try:
            fam=sm.families.Gaussian(); gee=smf.gee("essdai_total_recoded ~ time_since_observed_baseline_years*exposed + baseline_essdai + C(baseline_pop) + age_baseline + C(sex)", groups="patient_id", data=df, family=fam, cov_struct=Exchangeable()).fit(); term="time_since_observed_baseline_years:exposed"; est=gee.params.get(term,np.nan); se=gee.bse.get(term,np.nan); row.update(model_type="GEE gaussian exchangeable fallback", estimate=est, ci95_low=est-1.96*se, ci95_high=est+1.96*se, p_value=gee.pvalues.get(term,np.nan), model_converged=True, model_status="fallback_gee", warning=f"MixedLM failed: {e}")
        except Exception as ee: row.update(model_status="failed", warning=f"MixedLM failed: {e}; GEE failed: {ee}")
    return [row]


def fit_cox_model(surv, spec, outcome, event_col, threshold, min_events):
    row=_base_progression_row(spec,outcome,"baseline comorbidity"); row.update(effect_measure="Hazard ratio", threshold=threshold)
    if not LIFELINES_AVAILABLE: row.update(model_status="not_run_lifelines_missing", warning="lifelines is not installed"); return row
    df=surv.dropna(subset=["followup_years",event_col,spec.name,"baseline_essdai","age_baseline"]).copy(); df=df[df.followup_years>0]; row.update(n_patients=len(df),n_complete_cases=len(df),n_events=int(df[event_col].sum()),sparse_event_flag=int(df[event_col].sum())<min_events)
    if len(df)<5 or df[event_col].sum()<min_events or df[spec.name].nunique()<2: row.update(model_status="insufficient_events", warning="Insufficient events, patients, or exposure variation"); return row
    try:
        d=pd.get_dummies(df[["followup_years",event_col,spec.name,"baseline_essdai","age_baseline","baseline_pop","sex"]], columns=["baseline_pop","sex"], drop_first=True, dtype=float); d[spec.name]=d[spec.name].astype(float)
        cph=CoxPHFitter(); cph.fit(d, duration_col="followup_years", event_col=event_col)
        s=cph.summary.loc[spec.name]; row.update(model_type="Cox proportional hazards", estimate=float(s["exp(coef)"]), ci95_low=float(s["exp(coef) lower 95%"]), ci95_high=float(s["exp(coef) upper 95%"]), p_value=float(s["p"]), model_converged=True, model_status="adjusted")
        try: row["proportional_hazards_p"] = float(cph.check_assumptions(d, p_value_threshold=0.05, show_plots=False))
        except Exception as ph: row["warning"] = f"PH diagnostic unavailable: {ph}"
    except Exception as e: row.update(model_status="failed", warning=str(e))
    return row


def create_dotplot(overall, path):
    df=overall.iloc[::-1]; fig,ax=plt.subplots(figsize=(9,max(5,.4*len(df)+1))); y=np.arange(len(df)); ax.errorbar(df.pct_total_cohort,y,xerr=[df.pct_total_cohort-df.ci95_low,df.ci95_high-df.pct_total_cohort],fmt='o',color='#1f77b4'); ax.set_yticks(y,df.display_label); ax.set_xlabel('Documented baseline prevalence, % of total cohort');
    for yi,r in zip(y,df.itertuples()): ax.text(r.ci95_high+1,yi,f"{r.n_positive}/{r.n_total_cohort}",va='center',fontsize=8)
    fig.text(.01,.01,"Denominator is the total baseline cohort; Wilson 95% CIs use n positive / total cohort. Sensitivity definitions are documented in QC.",fontsize=8); fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig(path); plt.close(fig)


def create_grouped_barplot(bypop,path):
    df=bypop.iloc[::-1]; fig,ax=plt.subplots(figsize=(10,max(5,.5*len(df)+1))); y=np.arange(len(df)); offs=[-.22,0,.22]; colors=['#4c78a8','#f58518','#54a24b']
    for pop,off,col in zip(['pop1','pop2','pop3'],offs,colors):
        pct=df[f'pct_{pop}']; lo=df[f'ci95_{pop}_low']; hi=df[f'ci95_{pop}_high']; ax.errorbar(pct,y+off,xerr=[pct-lo,hi-pct],fmt='o',label=pop.capitalize(),color=col)
        for yi,r in zip(y+off,df.itertuples()): ax.text((getattr(r,f'pct_{pop}') or 0)+1, yi, f"{getattr(r,f'n_{pop}')}/{getattr(r,f'N_{pop}')}", fontsize=7, va='center')
    ax.set_yticks(y,df.display_label); ax.set_xlabel('Prevalence among evaluable patients within Pop, %'); ax.legend(); fig.text(.01,.01,"Pop denominators are condition-evaluable Pop1/Pop2/Pop3 patients; unclassifiable patients are excluded from tests.",fontsize=8); fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig(path); plt.close(fig)


def create_progression_forestplot(prog, order, path):
    panels=[('ESSDAI trajectory','Difference in annual ESSDAI slope',0,'Beta per year'),('Progression to ESSDAI ≥14','Progression to ESSDAI ≥14',1,'Hazard ratio'),('New ESSDAI-domain involvement','Development of new ESSDAI-domain involvement',1,'Hazard ratio')]
    fig,axs=plt.subplots(1,3,figsize=(16,max(6,.45*len(order)+1)),sharey=True); y=np.arange(len(order))
    for ax,(out,title,ref,measure) in zip(axs,panels):
        sub=prog[(prog.outcome==out)&(prog.effect_measure==measure)].set_index('comorbidity').reindex(order); ax.axvline(ref,color='grey',ls='--'); ax.set_title(title); ax.set_yticks(y,[next(c.label for c in CONDITIONS if c.name==n) for n in order])
        for i,(name,r) in enumerate(sub.iterrows()):
            if pd.notna(r.get('estimate')) and pd.notna(r.get('ci95_low')) and pd.notna(r.get('ci95_high')): ax.errorbar(r.estimate,i,xerr=[[r.estimate-r.ci95_low],[r.ci95_high-r.estimate]],fmt='o')
            else: ax.text(ref,i,'NE',ha='center',va='center',fontsize=8)
        if ref==1: ax.set_xscale('log')
    fig.text(.01,.01,"Adjusted for baseline ESSDAI, baseline Pop, age, and sex when estimable. Outcomes: follow-up ESSDAI slope, first ESSDAI ≥14, and first new inactive-at-baseline organ-domain activation.",fontsize=8); fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig(path); plt.close(fig)


def main() -> None:
    args=parse_args(); ensure_directories(); setup_logging(); np.random.seed(args.random_seed)
    logging.info("[1/8] Loading canonical sources"); upstream=check_upstream_artifacts(args.input,args.rebuild_upstream); raw=load_selected_raw_columns(args.input); spine=load_visit_spine(); pop=load_pop_classification(); domains=load_domain_flags()
    logging.info("[2/8] Building separated comorbidity and history indicators"); ind,source,conflicts,dup=derive_comorbidity_indicators(raw)
    baseline=build_baseline_comorbidity_dataset(ind,spine,pop)
    logging.info("[3/8] Writing baseline intermediate dataset"); baseline.to_parquet(common.INTERMEDIATE_DATA_DIR/"07_comorbidities_baseline_patient.parquet",index=False)
    logging.info("[4/8] Estimating overall prevalence"); overall=summarize_overall_prevalence(baseline)
    logging.info("[5/8] Comparing prevalence across Pop 1–3"); bypop=summarize_prevalence_by_pop(baseline,args.monte_carlo_replicates,args.random_seed)
    logging.info("[6/8] Building longitudinal outcomes"); longdf=build_longitudinal_essdai_dataset(spine,pop,raw,baseline,domains); severe=build_severe14_survival_dataset(longdf,baseline); newdom=build_new_domain_survival_dataset(longdf,baseline)
    longdf.to_parquet(common.INTERMEDIATE_DATA_DIR/"07_comorbidities_analysis_longitudinal.parquet",index=False); severe.to_parquet(common.INTERMEDIATE_DATA_DIR/"07_comorbidity_severe14_survival.parquet",index=False); newdom.to_parquet(common.INTERMEDIATE_DATA_DIR/"07_comorbidity_new_domain_survival.parquet",index=False)
    logging.info("[7/8] Fitting progression models"); rows=[]
    for spec in CONDITIONS:
        rows += fit_mixed_model(longdf,spec)
        rows.append(fit_cox_model(severe,spec,"Progression to ESSDAI ≥14","severe14_event",SEVERE_ACTIVITY_THRESHOLD_SECTION5,args.minimum_events))
        rows.append(fit_cox_model(newdom,spec,"New ESSDAI-domain involvement","new_domain_event","new inactive-at-baseline domain",args.minimum_events))
    prog=pd.DataFrame(rows); prog["fdr_bh_q_value"]=prog.groupby("outcome")["p_value"].transform(lambda s: apply_fdr(s))
    logging.info("[8/8] Writing tables, figures, and QC")
    overall.to_csv(TABLES_DIR/"07_comorbidities_overall.csv",index=False); bypop.to_csv(TABLES_DIR/"07_comorbidities_by_pop.csv",index=False); prog.to_csv(TABLES_DIR/"07_comorbidities_progression.csv",index=False)
    overall.to_csv(TABLES_DIR/"07_rheumatological_conditions_current.csv",index=False); summarize_rheumatological_history(baseline).to_csv(TABLES_DIR/"07_rheumatological_conditions_history.csv",index=False); bypop.to_csv(TABLES_DIR/"07_rheumatological_conditions_by_pop.csv",index=False); prog.to_csv(TABLES_DIR/"07_rheumatological_logistic_models.csv",index=False); prog.to_csv(TABLES_DIR/"07_rheumatological_model_diagnostics.csv",index=False)
    summarize_historical_family(baseline, PAST_MEDICAL_HISTORY_CONDITIONS, "past").to_csv(TABLES_DIR/"07_past_medical_history_descriptive.csv",index=False); summarize_historical_family(baseline, SJOGREN_HISTORY_MANIFESTATIONS, "sjogren").to_csv(TABLES_DIR/"07_sjogren_history_descriptive.csv",index=False); summarize_other_immune_conditions(baseline).to_csv(TABLES_DIR/"07_other_immune_conditions.csv",index=False); conflicts.to_csv(QC_DIR/"07_condition_status_conflicts.csv",index=False); source.to_csv(QC_DIR/"07_condition_source_mapping.csv",index=False); condition_status_sensitivity(baseline).to_csv(QC_DIR/"07_condition_status_sensitivity.csv",index=False)
    create_dotplot(overall,FIGURES_DIR/"07_comorbidities_dotplot.pdf"); create_grouped_barplot(bypop,FIGURES_DIR/"07_comorbidities_grouped_bar.pdf"); create_progression_forestplot(prog, overall.condition.tolist(), FIGURES_DIR/"07_comorbidities_progression_forestplot.pdf"); create_progression_forestplot(prog, overall.condition.tolist(), FIGURES_DIR/"07_rheumatological_forest_plot.pdf")
    pd.DataFrame(UNRECOGNIZED).groupby(["column","value"],as_index=False).n.sum().to_csv(QC_DIR/"07_comorbidities_unrecognized_values.csv",index=False) if UNRECOGNIZED else pd.DataFrame(columns=["column","value","n"]).to_csv(QC_DIR/"07_comorbidities_unrecognized_values.csv",index=False)
    pd.DataFrame([{ "condition":k,"availability_status":"unavailable","reason":v} for k,v in UNAVAILABLE_CONDITIONS.items()]).to_csv(QC_DIR/"07_comorbidities_unavailable_conditions.csv",index=False)
    source.to_csv(QC_DIR/"07_comorbidities_source_mapping.csv",index=False)
    miss=baseline[["baseline_pop","baseline_essdai","age_baseline","sex"]+CONDITION_NAMES].isna().sum().reset_index(); miss.columns=["variable","n_missing"]; miss.to_csv(QC_DIR/"07_comorbidities_missingness.csv",index=False)
    dup.to_csv(QC_DIR/"07_comorbidities_patient_duplicates.csv",index=False); prog.to_csv(QC_DIR/"07_comorbidities_model_diagnostics.csv",index=False)
    raw_ess=longdf.dropna(subset=["essdai_total_recoded","essdai_total_raw_qc"]); diff=pd.to_numeric(raw_ess.essdai_total_recoded,errors='coerce')-pd.to_numeric(raw_ess.essdai_total_raw_qc,errors='coerce')
    qc={"input_path":str(args.input),"input_modification_time":datetime.fromtimestamp(args.input.stat().st_mtime,timezone.utc).isoformat(),"script_version":SCRIPT_VERSION,"run_timestamp":datetime.now(timezone.utc).isoformat(),"random_seed":args.random_seed,"n_input_rows":int(len(raw)),"n_input_patients":int(raw.patient_id.nunique()),"n_canonical_visits":int(len(spine)),"n_baseline_patients":int(len(baseline)),"n_duplicate_patient_dates":int(len(dup)),"n_pipe_delimited_visit_dates":int(raw.get('had_pipe_delimited_date',pd.Series(dtype=bool)).sum()),"n_pop_classifiable":int(baseline.baseline_pop.isin(['Pop1','Pop2','Pop3']).sum()),"n_pop_unclassifiable":int((~baseline.baseline_pop.isin(['Pop1','Pop2','Pop3'])).sum()),"n_with_followup_essdai":int(longdf[longdf.visit_number.gt(0)&longdf.essdai_total_recoded.notna()].patient_id.nunique()),"n_at_risk_severe14":int(len(severe)),"n_severe14_events":int(severe.severe14_event.sum()) if not severe.empty else 0,"n_at_risk_new_domain":int(len(newdom)),"n_new_domain_events":int(newdom.new_domain_event.sum()) if not newdom.empty else 0,"severe_threshold_used":SEVERE_ACTIVITY_THRESHOLD_SECTION5,"essdai_primary_column":f"{common.POP_LONGITUDINAL_PARQUET.name}::{ESSDAI_CANONICAL_COL}","essdai_raw_qc_column":ESSDAI_RAW_QC_COL,"upstream_files_used":list(upstream.keys()),"upstream_file_timestamps":upstream,"warnings":["lifelines unavailable; Cox models not run"] if not LIFELINES_AVAILABLE else [],"essdai_reconciliation":{"n_concordant":int((diff==0).sum()),"n_discordant":int((diff!=0).sum()),"mean_difference":float(diff.mean()) if len(diff) else None,"median_difference":float(diff.median()) if len(diff) else None,"maximum_absolute_difference":float(diff.abs().max()) if len(diff) else None}}
    (QC_DIR/"07_comorbidities_qc.json").write_text(json.dumps(qc,indent=2,default=str))
    top=overall.head(3)
    claim=f"Confirmed-present rheumatological conditions were summarized using confirmation fields only; historical medical and Sjögren-related fields were exported descriptively and excluded from models. The leading confirmed-present rows were {top.iloc[0].display_label if len(top)>0 else 'NA'}, {top.iloc[1].display_label if len(top)>1 else 'NA'}, and {top.iloc[2].display_label if len(top)>2 else 'NA'}."
    logging.info(claim)
    generated=[TABLES_DIR/"07_comorbidities_overall.csv",TABLES_DIR/"07_comorbidities_by_pop.csv",TABLES_DIR/"07_comorbidities_progression.csv",FIGURES_DIR/"07_comorbidities_dotplot.pdf",FIGURES_DIR/"07_comorbidities_grouped_bar.pdf",FIGURES_DIR/"07_comorbidities_progression_forestplot.pdf"]
    summary=f"Total baseline patients: {len(baseline)}\nClassifiable Pop patients: {qc['n_pop_classifiable']}\nPatients with follow-up ESSDAI: {qc['n_with_followup_essdai']}\nSevere14 events: {qc['n_severe14_events']}\nNew-domain events: {qc['n_new_domain_events']}\nNumber of models fitted: {int(prog.model_status.isin(['adjusted','fallback_gee']).sum())}\nNumber of models not estimable: {int(~prog.model_status.isin(['adjusted','fallback_gee']).sum())}\nGenerated files:\n"+"\n".join(str(p) for p in generated)
    print(summary); logging.info("\n"+summary)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ITEM 1.4 — Baseline by subpopulation Pop1 / Pop2 / Pop3.

Classifies longitudinal visits using ESSDAI/ESSPRI rules, summarizes the
baseline classifiable cohort by Pop1/Pop2/Pop3, and creates a swimmer plot for
longitudinal feasibility review. Outputs are written under the repository's
standard Block A output folders from ``common.py`` when available.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, kruskal, pearsonr, spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402
from src.derivations.visit_dates import add_parsed_visit_dates

PATIENT_ID_COL = "ids__patient_record_number"
VISIT_DATE_COL = "ids__visit_date"
CODEBOOK_COLUMN = "FORM_NAME__QUESTION_NAME"
DEFAULT_INPUT = Path("/data/salazarda/data/obj1_sjd/data/raw") / "visits_long_collapsed_by_interval_codebook_corrected.parquet"

ESSDAI_TOTAL_CANDIDATES = ["essdai__essdai_total_score"]
ESSPRI_OBSERVED_COMPONENTS = {
    "dryness": "esspri_questionnaire__dryness",
    "fatigue": "esspri_questionnaire__fatigue",
    "pain": "esspri_questionnaire__pain",
}
ESSPRI_COMPONENTS = list(ESSPRI_OBSERVED_COMPONENTS.values())
FATIGUE_PROFAD_ITEMS = [
    "profile_of_fatigue_and_discomfort__profad_need_rest",
    "profile_of_fatigue_and_discomfort__profad_get_going",
    "profile_of_fatigue_and_discomfort__profad_keep_going",
    "profile_of_fatigue_and_discomfort__profad_weak",
]
FATIGUE_ANS_CANDIDATES = [
    "autonomic_nervous_system_questionnaire__fatigue_severity",
    "ans__fatigue_severity",
]
DRYNESS_D1_ITEMS = ["esspri_questionnaire__dry_eye", "esspri_questionnaire__dry_mouth"]
DRYNESS_D2_CORE_ITEMS = [
    "esspri_questionnaire__dry_eye",
    "esspri_questionnaire__dry_mouth",
    "esspri_questionnaire__skin_dry",
    "esspri_questionnaire__dry_inside_nose",
    "esspri_questionnaire__tracheal_dry",
]
DRYNESS_D2_ITEMS = DRYNESS_D2_CORE_ITEMS + ["esspri_questionnaire__vaginal_dryness"]
DRYNESS_D2_VAGINAL_NA = "esspri_questionnaire__vaginal_dryness_na"
DRYNESS_D3_PROFAD_ITEMS = [
    "profile_of_fatigue_and_discomfort__profad_eyes_sore",
    "profile_of_fatigue_and_discomfort__profad_eye_irritation",
    "profile_of_fatigue_and_discomfort__profad_eating_diff",
    "profile_of_fatigue_and_discomfort__profad_throat_nose_dry",
    "profile_of_fatigue_and_discomfort__profad_mouth_fluid_wet",
]
DRYNESS_D3_OCULAR_ITEMS = [
    "profile_of_fatigue_and_discomfort__profad_eyes_sore",
    "profile_of_fatigue_and_discomfort__profad_eye_irritation",
]
DRYNESS_D3_ORAL_AIRWAY_ITEMS = [
    "profile_of_fatigue_and_discomfort__profad_eating_diff",
    "profile_of_fatigue_and_discomfort__profad_throat_nose_dry",
    "profile_of_fatigue_and_discomfort__profad_mouth_fluid_wet",
]
PROXY_INPUT_COLUMNS = sorted(
    set(
        FATIGUE_PROFAD_ITEMS
        + FATIGUE_ANS_CANDIDATES
        + [
            "multidimensional_assessment_of_fatigue_scale__fat_q2",
            "multidimensional_assessment_of_fatigue_scale__fat_q1",
            "profile_of_fatigue_and_discomfort__profad_limb_discomfort",
            "profile_of_fatigue_and_discomfort__profad_finger_wrist_discomfort",
            DRYNESS_D2_VAGINAL_NA,
        ]
        + DRYNESS_D1_ITEMS
        + DRYNESS_D2_ITEMS
        + DRYNESS_D3_PROFAD_ITEMS
    )
)
AGE_CANDIDATES = [
    "ids__age_at_diagnosis",
    "ids__age_at_visit",
    "demographics__age_at_diagnosis",
    "age_at_diagnosis",
    "age",
]
SEX_CANDIDATES = ["ids__sex", "ids__gender", "demographics__sex", "demographics__gender", "sex", "gender"]
RACE_CANDIDATES = ["ids__race", "demographics__race", "race", "ethnicity", "ids__ethnicity"]
PROTOCOL_CANDIDATES = ["ids__protocol", "ids__protocol_number", "ids__study_protocol", "protocol", "protocol_number", "parent_protocol"]
BOOTSTRAP_REPLICATES = 2000
RANDOM_SEED = 20260714

PROXY_CANDIDATES_S5 = [
    ("fatigue_f1", "fatigue", "fatigue_proxy_f1_profad"), ("fatigue_f2", "fatigue", "fatigue_proxy_f2_mdafs_severity"), ("fatigue_f3", "fatigue", "fatigue_proxy_f3_mdafs_degree"), ("fatigue_f4", "fatigue", "fatigue_proxy_f4_ans"),
    ("pain_p1", "pain", "pain_proxy_p1_limb"), ("pain_p2", "pain", "pain_proxy_p2_finger_wrist"), ("pain_p12", "pain", "pain_proxy_p12_composite"),
    ("dryness_d1", "dryness", "dryness_proxy_d1_eye_mouth"), ("dryness_d2_core", "dryness", "dryness_proxy_d2_core"), ("dryness_d2_extended", "dryness", "dryness_proxy_d2_extended"), ("dryness_d3", "dryness", "dryness_proxy_d3_profad"),
]
POP_ORDER = ["Pop1", "Pop2", "Pop3", "Unclassifiable"]
POP_COLORS = {"Pop1": "#d95f02", "Pop2": "#7570b3", "Pop3": "#1b9e77", "Unclassifiable": "#9e9e9e"}
MISSINGNESS_MARKERS = {
    "ESSDAI and ESSPRI available": "o",
    "missing ESSDAI; ESSPRI <5": "v",
    "missing ESSDAI; ESSPRI >=5": "^",
    "missing ESSPRI; ESSDAI <5": "s",
    "missing ESSPRI; ESSDAI >=5": "P",
    "missing ESSDAI and ESSPRI": "X",
}
MISSINGNESS_ORDER = list(MISSINGNESS_MARKERS)
MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "unknown", "unk", "-99"}


def _common_path(name: str, fallback: Path) -> Path:
    return Path(getattr(common, name, fallback))


OUTPUTS_DIR = _common_path("OUTPUTS_DIR", PROJECT_ROOT / "outputs")
TABLES_DIR = _common_path("TABLES_DIR", OUTPUTS_DIR / "tables")
FIGURES_DIR = _common_path("FIGURES_DIR", OUTPUTS_DIR / "figures")
BLOCKA_TABLES_DIR = _common_path("BLOCKA_TABLES_DIR", TABLES_DIR / "blockA")
BLOCKA_FIGURES_DIR = _common_path("BLOCKA_FIGURES_DIR", FIGURES_DIR / "blockA")
BLOCKA_QC_DIR = OUTPUTS_DIR / "qc" / "blockA"
INTERMEDIATE_DIR = _common_path("INTERMEDIATE_DATA_DIR", PROJECT_ROOT / "data" / "intermediate")
DISPLAY = {"Unclassifiable": "Unclassified", "Pop1": "Pop1", "Pop2": "Pop2", "Pop3": "Pop3", "Overall": "Overall"}
DEFAULT_CODEBOOK = Path(getattr(common, "DEFAULT_CODEBOOK", PROJECT_ROOT / "metadata" / "Consolidated_Codebook_all_columns.xlsx"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ITEM 1.4 Pop1/Pop2/Pop3 baseline and longitudinal outputs.")
    parser.add_argument("--input", type=Path, default=Path(getattr(common, "DEFAULT_POP_DISTRIBUTION_INPUT", DEFAULT_INPUT)))
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    return parser.parse_args()


def is_missing(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_STRINGS


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input analytic file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def load_codebook(path: Path | None = None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def codebook_columns(codebook: pd.DataFrame | None) -> set[str]:
    if codebook is None or CODEBOOK_COLUMN not in codebook.columns:
        return set()
    return set(codebook[CODEBOOK_COLUMN].dropna().astype(str))


def select_first_available(df: pd.DataFrame, candidates: Iterable[str], codebook: pd.DataFrame | None = None) -> str | None:
    cb_cols = codebook_columns(codebook)
    for col in candidates:
        if col in df.columns and (not cb_cols or col in cb_cols):
            return col
    return None


def validate_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    missing_required = [col for col in (PATIENT_ID_COL, VISIT_DATE_COL) if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    essdai_cols = [col for col in ESSDAI_TOTAL_CANDIDATES if col in df.columns]
    if not essdai_cols:
        raise ValueError(f"No ESSDAI total column found. Tried: {ESSDAI_TOTAL_CANDIDATES}")
    available_esspri = [col for col in ESSPRI_COMPONENTS if col in df.columns]
    return {"essdai": essdai_cols, "esspri": available_esspri, "missing_esspri": [col for col in ESSPRI_COMPONENTS if col not in df.columns]}


def numeric_from_first_number(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def coalesce_essdai(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in ESSDAI_TOTAL_CANDIDATES:
        if col in df.columns:
            result = result.combine_first(numeric_from_first_number(df[col]))
    return result



def first_nonmissing(s: pd.Series) -> Any:
    values = s.dropna()
    return values.iloc[0] if len(values) else np.nan


def compute_esspri_from_components(dry: pd.Series, fatigue: pd.Series, pain: pd.Series) -> pd.Series:
    comp_df = pd.concat([dry, fatigue, pain], axis=1)
    comp_df.columns = ["dryness", "fatigue", "pain"]
    return comp_df.mean(axis=1).where(comp_df.notna().all(axis=1), np.nan)

def compute_esspri(df: pd.DataFrame) -> pd.Series:
    return compute_esspri_from_components(*(numeric_from_first_number(df[col]) for col in ESSPRI_COMPONENTS))



def numeric_optional(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return numeric_from_first_number(df[col])


def mean_when(values: pd.DataFrame, mask: pd.Series) -> pd.Series:
    return values.mean(axis=1).where(mask, np.nan)


def is_not_applicable(value: object) -> bool:
    if is_missing(value):
        return False
    s = str(value).strip().lower()
    return s in {"1", "1.0", "true", "yes", "y", "si", "sí", "not applicable", "not_applicable", "n/a", "na"}



def validate_numeric_range(
    series: pd.Series,
    minimum: float,
    maximum: float,
    variable_name: str,
    invalid_rows: list[dict[str, Any]],
    patient_id: pd.Series,
    visit_date: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.notna() & ~numeric.between(minimum, maximum)
    if invalid.any():
        for idx in numeric.index[invalid]:
            invalid_rows.append({"patient_id": patient_id.loc[idx], "visit_date": visit_date.loc[idx], "variable_name": variable_name, "invalid_value": numeric.loc[idx], "expected_min": minimum, "expected_max": maximum})
        numeric.loc[invalid] = np.nan
    return numeric.astype("float64")


def count_distinct_nonmissing(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric):
        return int(numeric.nunique())
    return int(series.dropna().astype(str).nunique())


def prepare_raw_esspri_proxy_inputs(work: pd.DataFrame, qc_counts: dict[str, Any]) -> pd.DataFrame:
    """Create numeric observed ESSPRI and raw proxy inputs only; no derived proxies."""
    invalid_rows: list[dict[str, Any]] = qc_counts.setdefault("invalid_value_rows", [])
    for key, src in ESSPRI_OBSERVED_COMPONENTS.items():
        raw = numeric_optional(work, src)
        work[f"esspri_{key}_observed"] = validate_numeric_range(raw, 0, 10, src, invalid_rows, work["patient_id"], work["visit_date_clean"])
    work["essdai_total"] = validate_numeric_range(work["essdai_total"], 0, 123, "essdai_total", invalid_rows, work["patient_id"], work["visit_date_clean"])
    raw_specs: dict[str, tuple[str, float, float]] = {}
    for col in FATIGUE_PROFAD_ITEMS + ["profile_of_fatigue_and_discomfort__profad_limb_discomfort", "profile_of_fatigue_and_discomfort__profad_finger_wrist_discomfort"] + DRYNESS_D3_PROFAD_ITEMS:
        raw_specs[col] = (f"raw_{col.split('__')[-1]}", 0, 7)
    for col in DRYNESS_D1_ITEMS + DRYNESS_D2_ITEMS:
        raw_specs[col] = (f"raw_{col.split('__')[-1]}", 0, 10)
    raw_specs["multidimensional_assessment_of_fatigue_scale__fat_q2"] = ("raw_mdafs_fat_q2", 1, 10)
    raw_specs["multidimensional_assessment_of_fatigue_scale__fat_q1"] = ("raw_mdafs_fat_q1", 1, 10)
    for i, col in enumerate(FATIGUE_ANS_CANDIDATES, start=1):
        raw_specs[col] = (f"raw_ans_fatigue_{i}", 0, 10)
    for src, (target, lo, hi) in raw_specs.items():
        raw = numeric_optional(work, src)
        work[target] = validate_numeric_range(raw, lo, hi, src, invalid_rows, work["patient_id"], work["visit_date_clean"])
    if DRYNESS_D2_VAGINAL_NA in work.columns:
        work["raw_vaginal_dryness_na"] = work[DRYNESS_D2_VAGINAL_NA]
    else:
        work["raw_vaginal_dryness_na"] = pd.NA
    qc_counts["n_invalid_numeric_values"] = len(invalid_rows)
    return work


def build_hierarchical_component(observed: pd.Series, candidates: list[tuple[pd.Series, str]]) -> tuple[pd.Series, pd.Series]:
    value = pd.to_numeric(observed, errors="coerce").copy()
    source = pd.Series("missing", index=observed.index, dtype="string")
    source.loc[value.notna()] = "observed_esspri"
    for candidate, label in candidates:
        candidate = pd.to_numeric(candidate, errors="coerce")
        use = value.isna() & candidate.notna()
        value.loc[use] = candidate.loc[use]
        source.loc[use] = label
    assert ~((value.notna()) & source.eq("missing")).any()
    assert ~((value.isna()) & source.ne("missing")).any()
    return value, source


def derive_esspri_proxies_from_collapsed_visit(work: pd.DataFrame, qc_counts: dict[str, Any]) -> pd.DataFrame:
    work["esspri_total_observed"] = compute_esspri_from_components(work["esspri_dryness_observed"], work["esspri_fatigue_observed"], work["esspri_pain_observed"])
    profad_fatigue = work[[f"raw_{c.split('__')[-1]}" for c in FATIGUE_PROFAD_ITEMS]]
    work["n_available_profad_fatigue"] = profad_fatigue.notna().sum(axis=1)
    work["fatigue_proxy_f1_profad_raw"] = mean_when(profad_fatigue, work["n_available_profad_fatigue"].eq(4))
    work["fatigue_proxy_f1_profad_relaxed_raw"] = mean_when(profad_fatigue, work["n_available_profad_fatigue"].ge(3))
    work["fatigue_proxy_f1_profad"] = work["fatigue_proxy_f1_profad_raw"] * 10 / 7
    work["fatigue_proxy_f1_profad_relaxed"] = work["fatigue_proxy_f1_profad_relaxed_raw"] * 10 / 7
    work["fatigue_proxy_f2_mdafs_severity_raw"] = work["raw_mdafs_fat_q2"]
    work["fatigue_proxy_f2_mdafs_severity"] = (work["raw_mdafs_fat_q2"] - 1) * 10 / 9
    work["fatigue_proxy_f2_mdafs_direct"] = work["raw_mdafs_fat_q2"]
    work["fatigue_proxy_f3_mdafs_degree_raw"] = work["raw_mdafs_fat_q1"]
    work["fatigue_proxy_f3_mdafs_degree"] = (work["raw_mdafs_fat_q1"] - 1) * 10 / 9
    work["fatigue_proxy_f3_mdafs_direct"] = work["raw_mdafs_fat_q1"]
    ans1, ans2 = work["raw_ans_fatigue_1"], work["raw_ans_fatigue_2"]
    qc_counts["n_fatigue_ans_conflicts"] = int((ans1.notna() & ans2.notna() & ans1.ne(ans2)).sum())
    work["fatigue_proxy_f4_ans"] = ans1.combine_first(ans2)
    work["pain_proxy_p1_limb_raw"] = work["raw_profad_limb_discomfort"]
    work["pain_proxy_p1_limb"] = work["pain_proxy_p1_limb_raw"] * 10 / 7
    work["pain_proxy_p2_finger_wrist_raw"] = work["raw_profad_finger_wrist_discomfort"]
    work["pain_proxy_p2_finger_wrist"] = work["pain_proxy_p2_finger_wrist_raw"] * 10 / 7
    work["pain_proxy_p12_composite"] = pd.concat([work["pain_proxy_p1_limb"], work["pain_proxy_p2_finger_wrist"]], axis=1).mean(axis=1).where(work[["pain_proxy_p1_limb", "pain_proxy_p2_finger_wrist"]].notna().all(axis=1), np.nan)
    work["pain_strategy_hierarchy"] = work["pain_proxy_p1_limb"].combine_first(work["pain_proxy_p2_finger_wrist"])
    work["pain_strategy_composite"] = pd.concat([work["pain_proxy_p1_limb"], work["pain_proxy_p2_finger_wrist"]], axis=1).mean(axis=1)
    d1 = work[[f"raw_{c.split('__')[-1]}" for c in DRYNESS_D1_ITEMS]]
    work["dryness_proxy_d1_n_available"] = d1.notna().sum(axis=1)
    work["dryness_proxy_d1_eye_mouth"] = mean_when(d1, work["dryness_proxy_d1_n_available"].eq(2))
    work["dryness_proxy_d1_eye_mouth_relaxed"] = mean_when(d1, work["dryness_proxy_d1_n_available"].ge(1))
    d2_core = work[[f"raw_{c.split('__')[-1]}" for c in DRYNESS_D2_CORE_ITEMS]]
    core_ok = d2_core.notna().sum(axis=1).ge(3) & d2_core[[f"raw_{c.split('__')[-1]}" for c in DRYNESS_D1_ITEMS]].notna().any(axis=1)
    work["dryness_proxy_d2_core"] = mean_when(d2_core, core_ok)
    vaginal_applicable = ~work["raw_vaginal_dryness_na"].map(is_not_applicable)
    vaginal_eval = work["raw_vaginal_dryness"].where(vaginal_applicable)
    d2_ext = d2_core.assign(raw_vaginal_dryness=vaginal_eval)
    work["dryness_proxy_d2_n_available"] = d2_ext.notna().sum(axis=1)
    work["dryness_proxy_d2_vaginal_included"] = vaginal_eval.notna()
    work["dryness_proxy_d2_extended"] = mean_when(d2_ext, core_ok)
    d3 = work[[f"raw_{c.split('__')[-1]}" for c in DRYNESS_D3_PROFAD_ITEMS]]
    work["dryness_proxy_d3_n_available"] = d3.notna().sum(axis=1)
    d3_has_ocular = d3[[f"raw_{c.split('__')[-1]}" for c in DRYNESS_D3_OCULAR_ITEMS]].notna().any(axis=1)
    d3_has_oral = d3[[f"raw_{c.split('__')[-1]}" for c in DRYNESS_D3_ORAL_AIRWAY_ITEMS]].notna().any(axis=1)
    work["dryness_proxy_d3_profad_raw"] = mean_when(d3, work["dryness_proxy_d3_n_available"].ge(4) & d3_has_ocular & d3_has_oral)
    work["dryness_proxy_d3_profad_relaxed_raw"] = mean_when(d3, work["dryness_proxy_d3_n_available"].ge(3) & d3_has_ocular & d3_has_oral)
    work["dryness_proxy_d3_profad"] = work["dryness_proxy_d3_profad_raw"] * 10 / 7
    work["dryness_proxy_d3_profad_relaxed"] = work["dryness_proxy_d3_profad_relaxed_raw"] * 10 / 7
    work["fatigue_proxy_hierarchical"], work["fatigue_proxy_hierarchical_source"] = build_hierarchical_component(work["esspri_fatigue_observed"], [(work["fatigue_proxy_f1_profad"], "proxy_f1_profad"), (work["fatigue_proxy_f2_mdafs_severity"], "proxy_f2_mdafs_severity"), (work["fatigue_proxy_f3_mdafs_degree"], "proxy_f3_mdafs_degree"), (work["fatigue_proxy_f4_ans"], "proxy_f4_ans")])
    work["fatigue_proxy_hierarchical_relaxed"], work["fatigue_proxy_hierarchical_relaxed_source"] = build_hierarchical_component(work["esspri_fatigue_observed"], [(work["fatigue_proxy_f1_profad_relaxed"], "proxy_f1_profad"), (work["fatigue_proxy_f2_mdafs_severity"], "proxy_f2_mdafs_severity"), (work["fatigue_proxy_f3_mdafs_degree"], "proxy_f3_mdafs_degree"), (work["fatigue_proxy_f4_ans"], "proxy_f4_ans")])
    work["pain_proxy_hierarchical"], work["pain_proxy_hierarchical_source"] = build_hierarchical_component(work["esspri_pain_observed"], [(work["pain_proxy_p1_limb"], "proxy_p1_limb"), (work["pain_proxy_p2_finger_wrist"], "proxy_p2_finger_wrist")])
    work["dryness_proxy_hierarchical"], work["dryness_proxy_hierarchical_source"] = build_hierarchical_component(work["esspri_dryness_observed"], [(work["dryness_proxy_d1_eye_mouth"], "proxy_d1_eye_mouth"), (work["dryness_proxy_d2_core"], "proxy_d2_core"), (work["dryness_proxy_d3_profad"], "proxy_d3_profad")])
    work["dryness_proxy_hierarchical_relaxed"], work["dryness_proxy_hierarchical_relaxed_source"] = build_hierarchical_component(work["esspri_dryness_observed"], [(work["dryness_proxy_d1_eye_mouth_relaxed"], "proxy_d1_eye_mouth"), (work["dryness_proxy_d2_core"], "proxy_d2_core"), (work["dryness_proxy_d3_profad_relaxed"], "proxy_d3_profad")])
    for col in [c for c in work.columns if c.startswith(("fatigue_proxy_", "pain_proxy_", "dryness_proxy_")) and not c.endswith("_source")]:
        vals = pd.to_numeric(work[col], errors="coerce").dropna()
        if len(vals) and not vals.between(0, 10).all() and not col.endswith(("_raw", "_n_available", "_included", "_direct")):
            raise ValueError(f"Proxy column outside 0-10 after derivation: {col}")
    work = add_esspri_scenarios(work, relaxed=False)
    work = add_esspri_scenarios(work, relaxed=True)
    return work


def add_esspri_scenarios(work: pd.DataFrame, relaxed: bool = False) -> pd.DataFrame:
    """Build best-available ESSPRI components for recovery scenarios S1-S4."""
    suffix = "_relaxed" if relaxed else ""
    dry = "dryness_proxy_hierarchical_relaxed" if relaxed else "dryness_proxy_hierarchical"
    fat = "fatigue_proxy_hierarchical_relaxed" if relaxed else "fatigue_proxy_hierarchical"
    pain = "pain_proxy_hierarchical"
    for comp, col in [("dryness", dry), ("fatigue", fat), ("pain", pain)]:
        work[f"esspri_{comp}_best_available{suffix}"] = work[col]
    best_cols = [
        f"esspri_dryness_best_available{suffix}",
        f"esspri_fatigue_best_available{suffix}",
        f"esspri_pain_best_available{suffix}",
    ]
    obs_cols = ["esspri_dryness_observed", "esspri_fatigue_observed", "esspri_pain_observed"]
    work[f"esspri_n_observed_components{suffix}"] = work[obs_cols].notna().sum(axis=1)
    work[f"esspri_n_available_components{suffix}"] = work[best_cols].notna().sum(axis=1)
    work[f"esspri_n_proxy_components{suffix}"] = work[f"esspri_n_available_components{suffix}"] - work[f"esspri_n_observed_components{suffix}"]
    work[f"esspri_total_proxy{suffix}"] = compute_esspri_from_components(*(work[c] for c in best_cols))
    scen = np.select(
        [
            work[f"esspri_n_available_components{suffix}"].lt(3),
            work[f"esspri_n_observed_components{suffix}"].eq(3),
            work[f"esspri_n_proxy_components{suffix}"].eq(1),
            work[f"esspri_n_proxy_components{suffix}"].eq(2),
            work[f"esspri_n_proxy_components{suffix}"].eq(3),
        ],
        ["unavailable", "observed_complete", "one_proxy", "two_proxies", "three_proxies"],
        default="unavailable",
    )
    work[f"esspri_derivation_scenario{suffix}"] = pd.Series(scen, index=work.index, dtype="string")
    return work

def classify_pop(essdai_total: object, esspri_total: object) -> str:
    essdai_missing = pd.isna(essdai_total)
    esspri_missing = pd.isna(esspri_total)
    if not essdai_missing and float(essdai_total) >= 5:
        return "Pop1"
    if not essdai_missing and float(essdai_total) < 5 and not esspri_missing and float(esspri_total) >= 5:
        return "Pop2"
    if not essdai_missing and float(essdai_total) < 5 and not esspri_missing and float(esspri_total) < 5:
        return "Pop3"
    return "Unclassifiable"


def visit_missingness_label(essdai_total: object, esspri_total: object) -> str:
    """Label ESSDAI/ESSPRI availability and threshold side for plot markers."""
    essdai_missing = pd.isna(essdai_total)
    esspri_missing = pd.isna(esspri_total)
    if essdai_missing and esspri_missing:
        return "missing ESSDAI and ESSPRI"
    if essdai_missing:
        return "missing ESSDAI; ESSPRI >=5" if float(esspri_total) >= 5 else "missing ESSDAI; ESSPRI <5"
    if esspri_missing:
        return "missing ESSPRI; ESSDAI >=5" if float(essdai_total) >= 5 else "missing ESSPRI; ESSDAI <5"
    return "ESSDAI and ESSPRI available"


def normalize_visit_level_dtypes(vis: pd.DataFrame) -> pd.DataFrame:
    """Use concrete dtypes before writing visit-level parquet artifacts."""
    out = vis.copy()
    numeric_cols = [
        "time_since_baseline_days",
        "time_since_baseline_years",
        "time_years",
        "visit_number",
        "essdai_total",
        "esspri_dryness",
        "esspri_fatigue",
        "esspri_pain",
        "esspri_total",
    ]
    datetime_cols = ["row_date_min", "row_date_max", "visit_date_clean", "baseline_date", "event_date"]
    string_cols = [
        "patient_id",
        "row_date_original",
        "pop_status",
        "pop_status_display",
        "pop_missingness_label",
        "baseline_pop_status",
        "baseline_pop_status_display",
    ]
    numeric_prefixes = ("esspri_", "fatigue_proxy_", "pain_proxy_", "pain_strategy_", "dryness_proxy_", "n_available_")
    non_numeric_markers = ("_source", "_scenario", "_label")
    for col in out.columns:
        if col.endswith("_included"):
            out[col] = out[col].astype("boolean")
        elif col.startswith(numeric_prefixes) and not any(marker in col for marker in non_numeric_markers):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif col.startswith("pop_status") or col.endswith("_source") or col.endswith("_scenario") or col.endswith("_label"):
            out[col] = out[col].astype("string")
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in datetime_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in string_cols:
        if col in out.columns:
            out[col] = out[col].astype("string")
    return out



def _detect_duplicate_conflicts(work: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    dup = work[work.duplicated(["patient_id", "visit_date"], keep=False)]
    for (pid, vdate), g in dup.groupby(["patient_id", "visit_date"], dropna=False):
        for col in cols:
            if col not in g.columns:
                continue
            n = count_distinct_nonmissing(g[col])
            if n > 1:
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                distinct = sorted(vals.unique().tolist()) if len(vals) else sorted(g[col].dropna().astype(str).unique().tolist())
                rows.append({"patient_id": pid, "visit_date": vdate, "variable_name": col, "n_distinct_values": n, "distinct_values": " | ".join(map(str, distinct)), "selected_value": first_nonmissing(g[col])})
    return pd.DataFrame(rows, columns=["patient_id", "visit_date", "variable_name", "n_distinct_values", "distinct_values", "selected_value"])


def build_longitudinal_pop_dataset(df: pd.DataFrame, codebook: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    work = df.copy()
    qc_counts: dict[str, Any] = {"n_input_rows": int(len(work))}
    warnings: list[str] = []

    work = add_parsed_visit_dates(work, patient_id_col=PATIENT_ID_COL, visit_date_col=VISIT_DATE_COL)
    valid_patient_id = ~work["patient_id"].map(is_missing)
    qc_counts["n_rows_with_valid_patient_id"] = int(valid_patient_id.sum())
    qc_counts["n_rows_excluded_missing_patient_id"] = int((~valid_patient_id).sum())
    work["row_date_original"] = work["visit_date_raw"]
    work["row_date_min"] = work["visit_date_min"]
    work["row_date_max"] = work["visit_date_max"]
    work["visit_date_clean"] = work["visit_date"]  # compatibility alias
    valid_visit_date = work["visit_date"].notna()
    qc_counts["n_rows_with_valid_visit_date"] = int(valid_visit_date.sum())
    qc_counts["n_rows_excluded_missing_visit_date"] = int((~valid_visit_date).sum())
    work["essdai_total"] = coalesce_essdai(work)
    work = prepare_raw_esspri_proxy_inputs(work, qc_counts)

    protocol_col = select_first_available(work, PROTOCOL_CANDIDATES, codebook)
    qc_counts["selected_protocol_column"] = protocol_col
    if protocol_col is None:
        warnings.append("Variable not found: protocol; protocol-stratified proxy validations omitted.")
    selected_demo = {
        "age_raw": select_first_available(work, AGE_CANDIDATES, codebook),
        "sex_raw": select_first_available(work, SEX_CANDIDATES, codebook),
        "race_raw": select_first_available(work, RACE_CANDIDATES, codebook),
    }
    for k, v in selected_demo.items():
        qc_counts[f"selected_{k}_column"] = v
        if v is None:
            warnings.append(f"Variable not found before collapse: {k.replace('_raw','')}")

    work = work[valid_patient_id & valid_visit_date].copy()
    qc_counts["n_unique_patients"] = int(work["patient_id"].nunique())
    qc_counts["n_patient_visit_rows_before_collapse"] = int(len(work))
    qc_counts["n_duplicate_patient_visit_rows"] = int(work.duplicated(["patient_id", "visit_date"]).sum())
    qc_counts["n_duplicate_patient_event_dates"] = qc_counts["n_duplicate_patient_visit_rows"]
    raw_cols = ["essdai_total", "esspri_dryness_observed", "esspri_fatigue_observed", "esspri_pain_observed"] + [c for c in work.columns if c.startswith("raw_")]
    conflict_df = _detect_duplicate_conflicts(work, raw_cols)
    qc_counts["n_patient_visits_with_conflicts"] = int(conflict_df[["patient_id", "visit_date"]].drop_duplicates().shape[0]) if not conflict_df.empty else 0
    qc_counts["n_conflicting_variable_values"] = int(len(conflict_df))
    BLOCKA_QC_DIR.mkdir(parents=True, exist_ok=True)
    conflict_df.to_csv(BLOCKA_QC_DIR / "01_esspri_proxy_duplicate_conflicts.csv", index=False)
    pd.DataFrame(qc_counts.get("invalid_value_rows", [])).to_csv(BLOCKA_QC_DIR / "01_esspri_proxy_invalid_values.csv", index=False)

    agg = {
        "row_date_original": ("row_date_original", lambda s: " | ".join(pd.Series(s).dropna().astype(str).unique())),
        "row_date_min": ("row_date_min", "min"), "row_date_max": ("row_date_max", "max"),
        "essdai_total": ("essdai_total", first_nonmissing),
    }
    for c in raw_cols:
        if c != "essdai_total" and c in work.columns:
            agg[c] = (c, first_nonmissing)
    if protocol_col:
        agg["protocol"] = (protocol_col, first_nonmissing)
    for std, src in selected_demo.items():
        if src:
            agg[std.replace("_raw", "")] = (src, first_nonmissing)
    work = work.groupby(["patient_id", "visit_date"], as_index=False).agg(**agg).sort_values(["patient_id", "visit_date"]).reset_index(drop=True)
    work["visit_date_clean"] = work["visit_date"]  # compatibility alias
    work = derive_esspri_proxies_from_collapsed_visit(work, qc_counts)

    work["esspri_dryness"] = work["esspri_dryness_observed"]
    work["esspri_fatigue"] = work["esspri_fatigue_observed"]
    work["esspri_pain"] = work["esspri_pain_observed"]
    work["esspri_total"] = work["esspri_total_observed"]
    work["esspri_total_s0_observed"] = work["esspri_total_observed"]
    for comp, src_col in [("dryness", "dryness_proxy_hierarchical_source"), ("fatigue", "fatigue_proxy_hierarchical_source"), ("pain", "pain_proxy_hierarchical_source")]:
        work[f"esspri_{comp}_final_source"] = work[src_col]
    mask_s1 = work["esspri_n_observed_components"].ge(2) & work["esspri_n_available_components"].eq(3)
    mask_s2 = work["esspri_n_observed_components"].ge(1) & work["esspri_n_available_components"].eq(3)
    mask_s3 = work["esspri_n_available_components"].eq(3)
    work["esspri_total_s1_one_proxy"] = work["esspri_total_proxy"].where(mask_s1)
    work["esspri_total_s2_up_to_two_proxies"] = work["esspri_total_proxy"].where(mask_s2)
    work["esspri_total_s3_all_available"] = work["esspri_total_proxy"].where(mask_s3)
    work["esspri_total_s3_all_available_label"] = pd.Series(pd.NA, index=work.index, dtype="string")
    work.loc[work["esspri_total_s3_all_available"].notna(), "esspri_total_s3_all_available_label"] = "exploratory_only"
    mask_s1r = work["esspri_n_observed_components_relaxed"].ge(2) & work["esspri_n_available_components_relaxed"].eq(3)
    mask_s2r = work["esspri_n_observed_components_relaxed"].ge(1) & work["esspri_n_available_components_relaxed"].eq(3)
    work["esspri_total_s1_one_proxy_relaxed"] = work["esspri_total_proxy_relaxed"].where(mask_s1r)
    work["esspri_total_s2_up_to_two_proxies_relaxed"] = work["esspri_total_proxy_relaxed"].where(mask_s2r)
    for label in ["s1_one_proxy", "s2_up_to_two_proxies", "s3_all_available", "s1_one_proxy_relaxed", "s2_up_to_two_proxies_relaxed"]:
        total_col = f"esspri_total_{label}"
        work[f"pop_status_{label}"] = [classify_pop(e, p) for e, p in zip(work["essdai_total"], work[total_col])]

    for name, comp, proxy in PROXY_CANDIDATES_S5:
        d = pd.to_numeric(work["esspri_dryness_observed"], errors="coerce").astype("float64")
        f = pd.to_numeric(work["esspri_fatigue_observed"], errors="coerce").astype("float64")
        p = pd.to_numeric(work["esspri_pain_observed"], errors="coerce").astype("float64")
        proxy_value = pd.to_numeric(work[proxy], errors="coerce").astype("float64")
        if comp == "dryness": d = proxy_value
        elif comp == "fatigue": f = proxy_value
        else: p = proxy_value
        work[f"esspri_total_replace_{name}"] = compute_esspri_from_components(d, f, p)
    if int(((work["esspri_total_observed"] < 0) | (work["esspri_total_observed"] > 10)).sum()):
        raise ValueError("ESSPRI observed total values outside range 0-10")
    work["pop_status"] = [classify_pop(e, p) for e, p in zip(work["essdai_total"], work["esspri_total_observed"])]
    work["pop_status_display"] = work["pop_status"].map(DISPLAY)
    work["pop_missingness_label"] = [visit_missingness_label(e, p) for e, p in zip(work["essdai_total"], work["esspri_total_observed"])]
    spine = pd.read_parquet(common.VISIT_SPINE_PARQUET)[["patient_id", "visit_id", "visit_date", "observed_baseline_date", "time_since_observed_baseline_days", "time_since_observed_baseline_years", "visit_number"]]
    work = work.merge(spine, on=["patient_id", "visit_date"], how="left", validate="one_to_one")
    if work["visit_id"].isna().any(): raise ValueError("clinical Pop visits missing from canonical spine")
    work["baseline_date"] = work["observed_baseline_date"]  # compatibility aliases
    work["event_date"] = work["visit_date"]
    work["time_since_baseline_days"] = work["time_since_observed_baseline_days"]
    work["time_since_baseline_years"] = work["time_since_observed_baseline_years"]
    work["time_years"] = work["time_since_observed_baseline_years"]
    baseline_status = work.loc[work["visit_number"].eq(0), ["patient_id", "pop_status", "pop_status_display"]].rename(columns={"pop_status": "baseline_pop_status", "pop_status_display": "baseline_pop_status_display"})
    work = work.merge(baseline_status, on="patient_id", how="left")
    qc_counts["n_patient_visit_rows_after_collapse"] = int(len(work))
    qc_counts["n_rows_missing_visit_date"] = int(work["event_date"].isna().sum())
    qc_counts["n_rows_missing_essdai"] = int(work["essdai_total"].isna().sum())
    qc_counts["n_rows_missing_esspri"] = int(work["esspri_total"].isna().sum())
    qc_counts["n_rows_negative_time"] = int((work["time_years"] < 0).sum())
    if qc_counts["n_rows_negative_time"]:
        warnings.append(f"Excluded {qc_counts['n_rows_negative_time']} rows with negative time_years.")
        work = work[~(work["time_years"] < 0)].copy()
    return normalize_visit_level_dtypes(work), qc_counts, warnings

def build_baseline_dataset(longitudinal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in longitudinal.groupby("patient_id", dropna=True, sort=True):
        baseline_rows = g[g["visit_number"].eq(0)].copy()
        if len(baseline_rows) != 1: raise ValueError("Expected exactly one canonical baseline visit per patient")
        baseline_rows["_has_essdai"] = baseline_rows["essdai_total"].notna().astype(int)
        baseline_rows["_has_esspri"] = baseline_rows["esspri_total"].notna().astype(int)
        baseline_rows = baseline_rows.sort_values(["_has_essdai", "_has_esspri", "row_date_max"], ascending=[False, False, True], na_position="last")
        rows.append(baseline_rows.iloc[0].drop(labels=["_has_essdai", "_has_esspri"], errors="ignore"))
    return pd.DataFrame(rows).reset_index(drop=True)


def fmt_median_iqr(values: pd.Series) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    return f"{vals.median():.1f} [{vals.quantile(0.25):.1f}, {vals.quantile(0.75):.1f}]"


def fmt_median_iqr_range(values: pd.Series) -> str:
    """Format a continuous value as median, IQR, and observed range."""
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    return (
        f"{vals.median():.1f} "
        f"[{vals.quantile(0.25):.1f}, {vals.quantile(0.75):.1f}]; "
        f"range {vals.min():.1f}–{vals.max():.1f}"
    )


def normalize_sex(value: object) -> str | float:
    if is_missing(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"f", "female", "woman", "w", "2", "2.0"} or "female" in s:
        return "female"
    if s in {"m", "male", "man", "1", "1.0"} or "male" in s:
        return "male"
    return s


def run_stat_tests(data: pd.DataFrame, variable: str, kind: str, warnings: list[str]) -> tuple[str, str]:
    groups = [data.loc[data["pop_status"] == pop, variable].dropna() for pop in ["Pop1", "Pop2", "Pop3"]]
    try:
        if kind == "continuous":
            usable = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
            if sum(len(g) > 0 for g in usable) < 2:
                return "", "Kruskal-Wallis"
            return f"{kruskal(*usable, nan_policy='omit').pvalue:.4g}", "Kruskal-Wallis"
        table = pd.crosstab(data["pop_status"], data[variable]).reindex(["Pop1", "Pop2", "Pop3"]).fillna(0)
        if table.shape[0] < 2 or table.shape[1] < 2:
            return "", "Chi-square test"
        _, pvalue, _, expected = chi2_contingency(table)
        if (expected < 5).any():
            warnings.append(f"Chi-square expected cell count <5 for {variable}.")
        return f"{pvalue:.4g}", "Chi-square test"
    except Exception as exc:  # statistical edge cases should not prevent outputs
        warnings.append(f"Could not run {kind} test for {variable}: {exc}")
        return "", "Kruskal-Wallis" if kind == "continuous" else "Chi-square test"


def summarize_table1_by_pop(baseline: pd.DataFrame, codebook: pd.DataFrame | None, warnings: list[str]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    classifiable = baseline[baseline["pop_status"].isin(["Pop1", "Pop2", "Pop3"])].copy()
    n_class = len(classifiable)
    rows = []
    def counts_avail(var: str | None = None) -> dict[str, int]:
        if var is None:
            return {"n_available_overall": n_class, **{f"n_available_{p.lower()}": int((classifiable["pop_status"] == p).sum()) for p in ["Pop1", "Pop2", "Pop3"]}}
        return {"n_available_overall": int(classifiable[var].notna().sum()), **{f"n_available_{p.lower()}": int(classifiable.loc[classifiable["pop_status"] == p, var].notna().sum()) for p in ["Pop1", "Pop2", "Pop3"]}}
    pop_counts = classifiable["pop_status"].value_counts().reindex(["Pop1", "Pop2", "Pop3"], fill_value=0)
    rows.append({"Variable": "N", "Overall": str(n_class), **{p: str(int(pop_counts[p])) for p in ["Pop1", "Pop2", "Pop3"]}, "p_value": "", "test": "", **counts_avail()})
    rows.append({"Variable": "Percent of baseline classifiable cohort", "Overall": "100.0%" if n_class else "NA", **{p: (f"{100*pop_counts[p]/n_class:.1f}%" if n_class else "NA") for p in ["Pop1", "Pop2", "Pop3"]}, "p_value": "", "test": "", **counts_avail()})
    for var, label in [
        ("essdai_total", "ESSDAI total, median [IQR]; range"),
        ("esspri_total", "ESSPRI total, median [IQR]; range"),
    ]:
        p, test = run_stat_tests(classifiable, var, "continuous", warnings)
        rows.append({"Variable": label, "Overall": fmt_median_iqr_range(classifiable[var]), **{pop: fmt_median_iqr_range(classifiable.loc[classifiable["pop_status"] == pop, var]) for pop in ["Pop1", "Pop2", "Pop3"]}, "p_value": p, "test": test, **counts_avail(var)})
    selected = {
        "selected_age_column": select_first_available(baseline, AGE_CANDIDATES, codebook),
        "selected_sex_column": select_first_available(baseline, SEX_CANDIDATES, codebook),
        "selected_race_column": select_first_available(baseline, RACE_CANDIDATES, codebook),
    }
    if selected["selected_age_column"]:
        baseline["_age"] = pd.to_numeric(baseline[selected["selected_age_column"]], errors="coerce")
        classifiable = baseline[baseline["pop_status"].isin(["Pop1", "Pop2", "Pop3"])].copy()
        p, test = run_stat_tests(classifiable, "_age", "continuous", warnings)
        rows.append({"Variable": "Age, median [IQR]", "Overall": fmt_median_iqr(classifiable["_age"]), **{pop: fmt_median_iqr(classifiable.loc[classifiable["pop_status"] == pop, "_age"]) for pop in ["Pop1", "Pop2", "Pop3"]}, "p_value": p, "test": test, **counts_avail("_age")})
    else:
        warnings.append("Variable not found: age")
    if selected["selected_sex_column"]:
        baseline["_sex"] = baseline[selected["selected_sex_column"]].map(normalize_sex)
        classifiable = baseline[baseline["pop_status"].isin(["Pop1", "Pop2", "Pop3"])].copy()
        p, test = run_stat_tests(classifiable, "_sex", "categorical", warnings)
        def female_fmt(d: pd.DataFrame) -> str:
            denom = int(d["_sex"].notna().sum()); num = int((d["_sex"] == "female").sum())
            return "NA" if denom == 0 else f"{num} ({100*num/denom:.1f}%)"
        rows.append({"Variable": "Sex, n female (%)", "Overall": female_fmt(classifiable), **{pop: female_fmt(classifiable[classifiable["pop_status"] == pop]) for pop in ["Pop1", "Pop2", "Pop3"]}, "p_value": p, "test": test, **counts_avail("_sex")})
    else:
        warnings.append("Variable not found: sex")
    if selected["selected_race_column"]:
        baseline["_race"] = baseline[selected["selected_race_column"]].where(~baseline[selected["selected_race_column"]].map(is_missing), np.nan)
        classifiable = baseline[baseline["pop_status"].isin(["Pop1", "Pop2", "Pop3"])].copy()
        p, test = run_stat_tests(classifiable, "_race", "categorical", warnings)
        for level in sorted(classifiable["_race"].dropna().astype(str).unique()):
            def lvl_fmt(d: pd.DataFrame, lvl: str = level) -> str:
                denom = int(d["_race"].notna().sum()); num = int((d["_race"].astype(str) == lvl).sum())
                return "NA" if denom == 0 else f"{num} ({100*num/denom:.1f}%)"
            rows.append({"Variable": f"Race, {level}, n (%)", "Overall": lvl_fmt(classifiable), **{pop: lvl_fmt(classifiable[classifiable["pop_status"] == pop]) for pop in ["Pop1", "Pop2", "Pop3"]}, "p_value": p, "test": test, **counts_avail("_race")})
            p = test = ""
    else:
        warnings.append("Variable not found: race")
    columns = ["Variable", "Overall", "Pop1", "Pop2", "Pop3", "p_value", "test", "n_available_overall", "n_available_pop1", "n_available_pop2", "n_available_pop3"]
    return pd.DataFrame(rows)[columns], selected


def make_pop_swimmer_plot(longitudinal: pd.DataFrame, baseline: pd.DataFrame, output_path: Path, warnings: list[str]) -> int:
    plot_df = longitudinal.dropna(subset=["patient_id", "time_years"]).copy()
    if plot_df.empty:
        warnings.append("No valid dated longitudinal rows available for swimmer plot.")
        return 0
    max_time = float(plot_df["time_years"].max())
    x_limit = max_time
    outside = 0
    if max_time > 15:
        x_limit = float(plot_df["time_years"].quantile(0.99))
        outside = int((plot_df["time_years"] > x_limit).sum())
        warnings.append(f"Swimmer plot x-axis limited to 99th percentile ({x_limit:.2f} years); {outside} points outside.")
    summary = plot_df.groupby("patient_id").agg(first=("time_years", "min"), last=("time_years", "max"), visits=("time_years", "count")).reset_index()
    base_status = baseline.set_index("patient_id")["pop_status"].to_dict()
    summary["baseline_pop"] = summary["patient_id"].map(base_status).fillna("Unclassifiable")
    summary["pop_rank"] = summary["baseline_pop"].map({p: i for i, p in enumerate(POP_ORDER)}).fillna(99)
    summary = summary.sort_values(["pop_rank", "last", "visits"], ascending=[True, False, False]).reset_index(drop=True)
    grouped_summaries = {pop: summary[summary["baseline_pop"] == pop].copy() for pop in POP_ORDER}
    for pop, group in grouped_summaries.items():
        grouped_summaries[pop]["y"] = np.arange(len(group), 0, -1)
    y_map = pd.concat(grouped_summaries.values(), ignore_index=True).set_index("patient_id")["y"].to_dict()
    plot_df["y"] = plot_df["patient_id"].map(y_map)
    plot_df["baseline_pop"] = plot_df["patient_id"].map(base_status).fillna("Unclassifiable")
    x_ticks = [(0, "baseline"), (0.5, "6 mo"), (1, "1y"), (2, "2y"), (4, "4y"), (6, "6y"), (8, "8y"), (10, "10y")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for baseline_pop in POP_ORDER:
        group = grouped_summaries[baseline_pop]
        panel_df = plot_df[plot_df["baseline_pop"] == baseline_pop]
        height = min(18, max(6, len(group) * 0.07 + 3))
        fig, ax = plt.subplots(figsize=(13, height))
        for _, row in group.iterrows():
            ax.hlines(row["y"], row["first"], row["last"], color="#d0d0d0", linewidth=0.8, zorder=1)
        for pop in POP_ORDER:
            pop_sub = panel_df[panel_df["pop_status"] == pop]
            for missingness_label in MISSINGNESS_ORDER:
                sub = pop_sub[pop_sub["pop_missingness_label"].eq(missingness_label)]
                if sub.empty:
                    continue
                is_complete = missingness_label == "ESSDAI and ESSPRI available"
                ax.scatter(
                    sub["time_years"],
                    sub["y"],
                    s=12 if is_complete else 24,
                    marker=MISSINGNESS_MARKERS[missingness_label],
                    color=POP_COLORS[pop],
                    edgecolors="none" if is_complete else "#222222",
                    linewidths=0 if is_complete else 0.45,
                    label=pop if is_complete else None,
                    alpha=0.9,
                    zorder=2 if is_complete else 3,
                )
        for x, label in x_ticks:
            ax.axvline(x, color="#777777", linestyle="--", linewidth=0.7, alpha=0.6)
            if x <= max(x_limit, 10):
                ax.text(x, 1.01, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="#555555")
        ax.set_xlim(left=-0.05, right=max(x_limit, 10) * 1.02)
        ax.set_ylim(0, max(len(group), 1) + 1)
        ax.set_yticks([])
        ax.set_ylabel("Patients")
        ax.set_xlabel("Time since baseline (years)")
        ax.set_title(
            f"Longitudinal classification for baseline {baseline_pop} patients (n={len(group)})\n"
            "Time since first recorded visit; points colored by ESSDAI/ESSPRI-defined population",
            loc="left",
            fontsize=11,
            color=POP_COLORS[baseline_pop],
            pad=18,
        )
        if group.empty:
            ax.text(0.5, 0.5, "No patients", transform=ax.transAxes, ha="center", va="center", color="#777777")
        pop_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=POP_COLORS[pop], markeredgecolor="none", markersize=6, label=pop)
            for pop in POP_ORDER
        ]
        missingness_handles = [
            Line2D(
                [0],
                [0],
                marker=MISSINGNESS_MARKERS[label],
                color="none",
                markerfacecolor="#ffffff",
                markeredgecolor="#222222",
                markeredgewidth=0.8,
                markersize=6,
                label=label,
            )
            for label in MISSINGNESS_ORDER
        ]
        # Keep both legends inside the figure canvas.  Anchoring them below the
        # axes (and then relying on ``bbox_inches='tight'``) made the legend
        # titles and entries collide in short panels.
        fig.legend(
            handles=pop_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.32),
            frameon=False,
            ncol=4,
            columnspacing=2.5,
            handletextpad=0.8,
            borderaxespad=0,
            title="Population color",
            fontsize=9,
            title_fontsize=10,
        )
        fig.legend(
            handles=missingness_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.17),
            frameon=False,
            ncol=3,
            columnspacing=1.8,
            handletextpad=0.6,
            borderaxespad=0,
            title="ESSDAI/ESSPRI availability marker",
            fontsize=9,
            title_fontsize=10,
        )
        fig.text(
            0.02,
            0.025,
            "Pop1 = ESSDAI ≥5; Pop2 = ESSDAI <5 and ESSPRI ≥5; Pop3 = ESSDAI <5 and ESSPRI <5; grey = insufficient data. Marker shape shows which score is missing and the available score's <5 vs ≥5 side.",
            fontsize=8,
        )
        # Reserve a dedicated lower band for the two legends and footnote.
        # This is deliberately explicit instead of ``tight_layout`` because
        # the legends are figure-level artists.
        fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.42)
        suffix = baseline_pop.lower()
        panel_path = output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")
        fig.savefig(panel_path, bbox_inches="tight")
        plt.close(fig)
    return outside



def describe_baseline_unclassifiable(baseline: pd.DataFrame) -> pd.DataFrame:
    """Summarize ESSDAI/ESSPRI availability for patients unclassifiable at baseline."""
    unclassifiable = baseline[baseline["pop_status"] == "Unclassifiable"].copy()
    if unclassifiable.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "baseline_date",
                "essdai_total",
                "esspri_total",
                "essdai_baseline_status",
                "esspri_baseline_status",
                "unclassifiable_reason",
            ]
        )

    unclassifiable["essdai_baseline_status"] = np.where(
        unclassifiable["essdai_total"].notna(),
        "available",
        "missing",
    )
    unclassifiable["esspri_baseline_status"] = np.where(
        unclassifiable["esspri_total"].notna(),
        "available",
        "missing",
    )

    conditions = [
        unclassifiable["essdai_total"].isna() & unclassifiable["esspri_total"].isna(),
        unclassifiable["essdai_total"].isna() & unclassifiable["esspri_total"].notna(),
        unclassifiable["essdai_total"].notna() & unclassifiable["esspri_total"].isna(),
    ]
    reasons = [
        "missing ESSDAI and ESSPRI at baseline",
        "missing ESSDAI at baseline",
        "missing ESSPRI at baseline with ESSDAI <5",
    ]
    unclassifiable["unclassifiable_reason"] = np.select(
        conditions,
        reasons,
        default="not classifiable by ESSDAI/ESSPRI rule",
    )
    return unclassifiable[
        [
            "patient_id",
            "baseline_date",
            "essdai_total",
            "esspri_total",
            "essdai_baseline_status",
            "esspri_baseline_status",
            "unclassifiable_reason",
        ]
    ]



def label_visit(visit_number: int) -> str:
    return "Baseline" if int(visit_number) == 0 else f"Visit {int(visit_number)}"


def q(values: pd.Series, percentile: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    return float(vals.quantile(percentile)) if len(vals) else np.nan


def unclassifiable_reason(row: pd.Series, suffix: str = "") -> str:
    essdai_missing = pd.isna(row["essdai_total"])
    esspri_missing = pd.isna(row["esspri_total"])
    if essdai_missing and esspri_missing:
        return f"missing ESSDAI and ESSPRI{suffix}"
    if essdai_missing:
        return f"missing ESSDAI{suffix}"
    if not essdai_missing and float(row["essdai_total"]) < 5 and esspri_missing:
        return f"missing ESSPRI{suffix} with ESSDAI <5" if suffix else "missing ESSPRI with ESSDAI <5"
    return "not classifiable by ESSDAI/ESSPRI rule"


def distribution_by_visit(longitudinal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for visit_number, group in longitudinal.groupby("visit_number", sort=True):
        denom = len(group)
        for pop in POP_ORDER:
            pop_group = group[group["pop_status"].eq(pop)]
            rows.append(
                {
                    "visit_number": int(visit_number),
                    "time_point_label": label_visit(visit_number),
                    "median_time_since_baseline_yrs": group["time_since_baseline_years"].median(),
                    "q1_time_since_baseline_yrs": q(group["time_since_baseline_years"], 0.25),
                    "q3_time_since_baseline_yrs": q(group["time_since_baseline_years"], 0.75),
                    "pop_status": pop,
                    "pop_status_display": DISPLAY[pop],
                    "n_patients": len(pop_group),
                    "n_patients_evaluable_at_visit": denom,
                    "pct_patients_at_visit": (100 * len(pop_group) / denom if denom else np.nan),
                    "n_essdai_available": int(pop_group["essdai_total"].notna().sum()),
                    "n_esspri_available": int(pop_group["esspri_total"].notna().sum()),
                    "n_essdai_esspri_available": int((pop_group["essdai_total"].notna() & pop_group["esspri_total"].notna()).sum()),
                    "n_unclassifiable": int((group["pop_status"] == "Unclassifiable").sum()),
                }
            )
    return pd.DataFrame(rows)


def describe_visit_unclassifiable(longitudinal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unclassifiable = longitudinal[longitudinal["pop_status"] == "Unclassifiable"].copy()
    unclassifiable["unclassifiable_reason"] = unclassifiable.apply(unclassifiable_reason, axis=1) if not unclassifiable.empty else pd.Series(dtype="string")
    if unclassifiable.empty:
        reason_counts = pd.DataFrame(columns=["visit_number", "time_point_label", "unclassifiable_reason", "n_visits", "pct_unclassifiable_visits"])
    else:
        total_by_visit = unclassifiable.groupby("visit_number").size()
        reason_counts = (
            unclassifiable.groupby(["visit_number", "unclassifiable_reason"])
            .size()
            .rename("n_visits")
            .reset_index()
        )
        reason_counts["time_point_label"] = reason_counts["visit_number"].map(label_visit)
        reason_counts["pct_unclassifiable_visits"] = reason_counts.apply(
            lambda row: 100 * row["n_visits"] / total_by_visit.loc[row["visit_number"]], axis=1
        )
        reason_counts = reason_counts[["visit_number", "time_point_label", "unclassifiable_reason", "n_visits", "pct_unclassifiable_visits"]]
    return reason_counts, unclassifiable


def build_by_visit_qc(longitudinal: pd.DataFrame, outside: int, warnings: list[str]) -> dict:
    denom = longitudinal.groupby("visit_number")["patient_id"].count()
    return {
        "n_visits_max": int(longitudinal["visit_number"].max()) if not longitudinal.empty else 0,
        "denominators_by_visit_number": {str(k): int(v) for k, v in denom.items()},
        "pop_counts_by_visit_number": {str(k): v.value_counts().reindex(POP_ORDER, fill_value=0).to_dict() for k, v in longitudinal.groupby("visit_number")["pop_status"]},
        "unclassifiable_counts_by_visit_number": {str(k): int((g["pop_status"] == "Unclassifiable").sum()) for k, g in longitudinal.groupby("visit_number")},
        "essdai_missing_by_visit_number": {str(k): int(g["essdai_total"].isna().sum()) for k, g in longitudinal.groupby("visit_number")},
        "esspri_missing_by_visit_number": {str(k): int(g["esspri_total"].isna().sum()) for k, g in longitudinal.groupby("visit_number")},
        "late_followup_sparse_flags": {str(k): bool(v < 10) for k, v in denom.items()},
        "n_plot_points_outside_xlim": outside,
        "warnings": warnings,
    }




def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    try:
        return float((pearsonr if method == "pearson" else spearmanr)(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)
    except Exception:
        return np.nan


def compute_icc_2_1(observed: pd.Series, proxy: pd.Series) -> float:
    data = pd.concat([pd.to_numeric(observed, errors="coerce"), pd.to_numeric(proxy, errors="coerce")], axis=1).dropna().to_numpy(float)
    n, k = data.shape if data.size else (0, 0)
    if n < 2 or k != 2 or np.nanvar(data) == 0:
        return np.nan
    row_means = data.mean(axis=1, keepdims=True); col_means = data.mean(axis=0, keepdims=True); grand = data.mean()
    ssr = k * ((row_means - grand) ** 2).sum(); ssc = n * ((col_means - grand) ** 2).sum(); sse = ((data - row_means - col_means + grand) ** 2).sum()
    msr = ssr / (n - 1); msc = ssc / (k - 1); mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return np.nan if denom == 0 else float((msr - mse) / denom)


def continuous_metrics(data: pd.DataFrame, observed_col: str, proxy_col: str) -> dict[str, float]:
    pair = data[[observed_col, proxy_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if pair.empty:
        return {k: np.nan for k in ["mean_observed","mean_proxy","mae","rmse","mean_bias","median_bias","median_absolute_error","standard_deviation_difference","pearson_r","spearman_rho","icc_2_1","loa_lower","loa_upper"]}
    diff = pair[proxy_col] - pair[observed_col]; ae = diff.abs(); sd = diff.std(ddof=1)
    return {"mean_observed": float(pair[observed_col].mean()), "mean_proxy": float(pair[proxy_col].mean()), "mae": float(ae.mean()), "rmse": float(np.sqrt(np.mean(diff**2))), "mean_bias": float(diff.mean()), "median_bias": float(diff.median()), "median_absolute_error": float(ae.median()), "standard_deviation_difference": float(sd), "pearson_r": _safe_corr(pair[observed_col], pair[proxy_col], "pearson"), "spearman_rho": _safe_corr(pair[observed_col], pair[proxy_col], "spearman"), "icc_2_1": compute_icc_2_1(pair[observed_col], pair[proxy_col]), "loa_lower": float(diff.mean() - 1.96 * sd) if pd.notna(sd) else np.nan, "loa_upper": float(diff.mean() + 1.96 * sd) if pd.notna(sd) else np.nan}


def cluster_bootstrap_metrics(data: pd.DataFrame, patient_col: str, metric_function: callable, n_bootstrap: int = BOOTSTRAP_REPLICATES, seed: int = RANDOM_SEED) -> dict[str, float]:
    patients = pd.Series(data[patient_col].dropna().unique())
    if patients.empty:
        return {"n_valid_bootstrap_replicates": 0}
    rng = np.random.default_rng(seed); vals: dict[str, list[float]] = {}
    for _ in range(n_bootstrap):
        parts = []
        for i, pid in enumerate(rng.choice(patients, size=len(patients), replace=True)):
            g = data[data[patient_col].eq(pid)].copy(); g[patient_col] = f"{pid}__boot{i}"; parts.append(g)
        m = metric_function(pd.concat(parts, ignore_index=True))
        for k, v in m.items():
            if isinstance(v, (int, float, np.floating)) and pd.notna(v): vals.setdefault(k, []).append(float(v))
    out = {"n_valid_bootstrap_replicates": max((len(v) for v in vals.values()), default=0)}
    for k, v in vals.items():
        out[f"{k}_ci_low"] = float(np.percentile(v, 2.5)); out[f"{k}_ci_high"] = float(np.percentile(v, 97.5))
    return out


def _subsets(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    out = [("overall_all_visits", "overall", df), ("overall_baseline", "overall", df[df["visit_number"].eq(0)]), ("overall_essdai_lt5", "overall", df[pd.to_numeric(df["essdai_total"], errors="coerce").lt(5)])]
    if "protocol" in df.columns:
        for prot, g in df.groupby("protocol", dropna=True):
            out.append(("protocol_all_visits", str(prot), g)); out.append(("protocol_essdai_lt5", str(prot), g[pd.to_numeric(g["essdai_total"], errors="coerce").lt(5)]))
    return out


def _status(data: pd.DataFrame) -> str:
    return "insufficient_sample" if len(data) < 30 or data["patient_id"].nunique() < 20 else "ok"


def build_proxy_validation(longitudinal: pd.DataFrame) -> pd.DataFrame:
    candidates = [(d, n, {"fatigue":"esspri_fatigue_observed","pain":"esspri_pain_observed","dryness":"esspri_dryness_observed"}[d], p) for n,d,p in PROXY_CANDIDATES_S5]
    rows=[]
    for domain, cand, obs, proxy in candidates:
        for subset, protocol, data in _subsets(longitudinal):
            pair = data[["patient_id", obs, proxy]].dropna()
            m = continuous_metrics(pair, obs, proxy); boot = cluster_bootstrap_metrics(pair, "patient_id", lambda x, o=obs, p=proxy: {k:v for k,v in continuous_metrics(x,o,p).items() if k in ["mae","rmse","mean_bias","pearson_r","spearman_rho","icc_2_1"]})
            rows.append({"domain": domain, "candidate_id": cand, "observed_column": obs, "proxy_column": proxy, "subset": subset, "protocol": protocol, "n_visits": len(pair), "n_patients": pair["patient_id"].nunique(), **m, **boot, "component_threshold_5_discordant": int(((pair[obs]>=5)!=(pair[proxy]>=5)).sum()) if len(pair) else 0, "status": _status(pair)})
    return pd.DataFrame(rows)


def build_proxy_total_validation(longitudinal: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for cand, domain, _proxy in PROXY_CANDIDATES_S5:
        repl=f"esspri_total_replace_{cand}"
        for subset, protocol, data in _subsets(longitudinal):
            pair=data[["patient_id","esspri_total_observed",repl]].dropna()
            m=continuous_metrics(pair,"esspri_total_observed",repl); boot=cluster_bootstrap_metrics(pair,"patient_id",lambda x,r=repl:{k:v for k,v in continuous_metrics(x,"esspri_total_observed",r).items() if k in ["mae","rmse","mean_bias","pearson_r","spearman_rho","icc_2_1"]})
            rows.append({"domain":domain,"candidate_id":cand,"replaced_components":domain,"subset":subset,"protocol":protocol,"n_visits":len(pair),"n_patients":pair["patient_id"].nunique(),**m,**boot,"status":_status(pair)})
    return pd.DataFrame(rows)


def _threshold_metrics(data: pd.DataFrame, repl: str) -> dict[str, float]:
    d=data.dropna(subset=["esspri_total_observed",repl]).copy(); y=(d["esspri_total_observed"]>=5); yp=(d[repl]>=5)
    if d.empty: return {k:np.nan for k in ["sensitivity","specificity","ppv","npv","accuracy","balanced_accuracy","f1_score","cohen_kappa"]}
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[False, True]).ravel()
    div=lambda a,b: float(a/b) if b else np.nan
    return {"TP":int(tp),"TN":int(tn),"FP":int(fp),"FN":int(fn),"sensitivity":div(tp,tp+fn),"specificity":div(tn,tn+fp),"ppv":div(tp,tp+fp),"npv":div(tn,tn+fn),"accuracy":float(accuracy_score(y,yp)),"balanced_accuracy":float(balanced_accuracy_score(y,yp)),"f1_score":float(f1_score(y,yp,zero_division=0)),"cohen_kappa":float(cohen_kappa_score(y,yp)),"esspri_total_threshold_5_discordant":int((y!=yp).sum())}


def build_threshold_agreement(longitudinal: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    base=longitudinal[pd.to_numeric(longitudinal["essdai_total"], errors="coerce").lt(5)]
    for cand,domain,_ in PROXY_CANDIDATES_S5:
        repl=f"esspri_total_replace_{cand}"
        for subset,protocol,data in [("overall_essdai_lt5","overall",base)] + ([("protocol_essdai_lt5",str(p),g) for p,g in base.groupby("protocol",dropna=True)] if "protocol" in base.columns else []):
            pair=data[["patient_id","esspri_total_observed",repl]].dropna(); m=_threshold_metrics(pair,repl); boot=cluster_bootstrap_metrics(pair,"patient_id",lambda x,r=repl:{k:v for k,v in _threshold_metrics(x,r).items() if k in ["sensitivity","specificity","cohen_kappa"]})
            rows.append({"domain":domain,"candidate_id":cand,"subset":subset,"protocol":protocol,"n_visits":len(pair),"n_patients":pair["patient_id"].nunique(),**m,**boot,"status":_status(pair)})
    return pd.DataFrame(rows)


def build_pop2_pop3_reclassification(longitudinal: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]; mats=[]; base=longitudinal[pd.to_numeric(longitudinal["essdai_total"], errors="coerce").lt(5)]
    for cand,domain,_ in PROXY_CANDIDATES_S5:
        repl=f"esspri_total_replace_{cand}"; d=base.dropna(subset=["esspri_total_observed",repl]).copy()
        obs=np.where(d["esspri_total_observed"]>=5,"Pop2","Pop3"); pred=np.where(d[repl]>=5,"Pop2","Pop3")
        tab=pd.crosstab(pd.Series(obs,name="observed_pop"),pd.Series(pred,name="proxy_pop")).reindex(index=["Pop2","Pop3"],columns=["Pop2","Pop3"],fill_value=0)
        b=int(tab.loc["Pop2","Pop3"]); c=int(tab.loc["Pop3","Pop2"]); pval=float(binomtest(min(b,c),b+c,0.5).pvalue) if b+c>0 else np.nan
        rows.append({"domain":domain,"candidate_id":cand,"n_visits":len(d),"n_patients":d["patient_id"].nunique(),"percent_agreement":float((obs==pred).mean()*100) if len(d) else np.nan,"cohen_kappa":float(cohen_kappa_score(obs,pred)) if len(d) else np.nan,"n_pop2_to_pop3":b,"n_pop3_to_pop2":c,"net_change_pop2":c-b,"net_change_pop3":b-c,"mcnemar_exact_p_value":pval,"status":_status(d)})
        for o in ["Pop2","Pop3"]:
            mats.append({"domain":domain,"candidate_id":cand,"observed":o,"proxy_pop2":int(tab.loc[o,"Pop2"]),"proxy_pop3":int(tab.loc[o,"Pop3"])})
    return pd.DataFrame(rows), pd.DataFrame(mats)


def build_proxy_sensitivity_summary(longitudinal: pd.DataFrame) -> pd.DataFrame:
    scenarios=[("S0_observed_official","esspri_total_s0_observed","pop_status"),("S1_one_proxy","esspri_total_s1_one_proxy","pop_status_s1_one_proxy"),("S2_up_to_two_proxies","esspri_total_s2_up_to_two_proxies","pop_status_s2_up_to_two_proxies"),("S3_all_available_exploratory_only","esspri_total_s3_all_available","pop_status_s3_all_available"),("S4_S1_one_proxy_relaxed","esspri_total_s1_one_proxy_relaxed","pop_status_s1_one_proxy_relaxed"),("S4_S2_up_to_two_proxies_relaxed","esspri_total_s2_up_to_two_proxies_relaxed","pop_status_s2_up_to_two_proxies_relaxed")]
    rows=[]
    for scope,df in [("all_visits",longitudinal),("baseline",longitudinal[longitudinal["visit_number"].eq(0)])]:
        denom=len(df); s0=None
        for scenario,total_col,pop_col in scenarios:
            pop=df[pop_col] if pop_col in df else pd.Series("Unclassifiable",index=df.index); nclass=int(pop.isin(["Pop1","Pop2","Pop3"]).sum())
            if scenario.startswith("S0"): s0=nclass
            rows.append({"time_scope":scope,"scenario":scenario,"denominator_type":"patients" if scope=="baseline" else "visits","denominator":denom,"n_visits_with_esspri":int(df[total_col].notna().sum()),"n_visits_classifiable":nclass,"n_baseline_patients_classifiable":nclass if scope=="baseline" else np.nan,"n_pop1":int((pop=="Pop1").sum()),"n_pop2":int((pop=="Pop2").sum()),"n_pop3":int((pop=="Pop3").sum()),"n_unclassifiable":int((pop=="Unclassifiable").sum()),"change_n_vs_s0":nclass-(s0 or 0),"change_percentage_points_vs_s0":100*(nclass-(s0 or 0))/denom if denom else np.nan})
    return pd.DataFrame(rows)


def build_coverage_and_rescue(longitudinal: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    cov=[]
    for cand,domain,proxy in PROXY_CANDIDATES_S5:
        obs={"fatigue":"esspri_fatigue_observed","pain":"esspri_pain_observed","dryness":"esspri_dryness_observed"}[domain]
        miss=longitudinal[obs].isna(); avail=longitudinal[proxy].notna() & miss
        cov.append({"domain":domain,"candidate_id":cand,"proxy_column":proxy,"n_component_missing":int(miss.sum()),"n_proxy_available_when_component_missing":int(avail.sum()),"pct_missing_rescued":float(100*avail.sum()/miss.sum()) if miss.sum() else np.nan,"n_unique_patients_rescued":int(longitudinal.loc[avail,"patient_id"].nunique())})
    rescue=longitudinal[longitudinal["esspri_total_observed"].isna() & longitudinal["esspri_total_proxy"].notna()].copy()
    rescued=rescue["esspri_n_proxy_components"].value_counts().to_dict()
    rescued_df=pd.DataFrame([{"rescued_with_one_proxy":int(rescued.get(1,0)),"rescued_with_two_proxies":int(rescued.get(2,0)),"rescued_with_three_proxies":int(rescued.get(3,0)),"n_rescued_visits":len(rescue),"n_rescued_patients":rescue["patient_id"].nunique()}])
    rank=pd.DataFrame(cov)
    return pd.DataFrame(cov), rescued_df, rank


def build_candidate_ranking(component: pd.DataFrame,total: pd.DataFrame,threshold: pd.DataFrame,coverage: pd.DataFrame) -> pd.DataFrame:
    c=component[component["subset"].eq("overall_all_visits")][["domain","candidate_id","mae","rmse","icc_2_1"]].rename(columns={"mae":"component_mae","rmse":"component_rmse","icc_2_1":"component_icc"})
    t=total[total["subset"].eq("overall_all_visits")][["candidate_id","mae"]].rename(columns={"mae":"total_mae"})
    th=threshold[["candidate_id","cohen_kappa","sensitivity","specificity"]].rename(columns={"cohen_kappa":"threshold_kappa"})
    co=coverage[["candidate_id","n_proxy_available_when_component_missing"]].rename(columns={"n_proxy_available_when_component_missing":"coverage_rescued_visits"})
    out=c.merge(t,on="candidate_id",how="left").merge(th,on="candidate_id",how="left").merge(co,on="candidate_id",how="left")
    out["ranking_note"]="Evidence summary only; does not change prespecified hierarchies."
    return out.sort_values(["domain","component_mae"], na_position="last")


def run_internal_consistency_tests(longitudinal: pd.DataFrame) -> None:
    assert compute_esspri_from_components(pd.Series([3.0]), pd.Series([np.nan]), pd.Series([9.0])).isna().iloc[0]
    assert np.isclose(0*10/7,0) and np.isclose(7*10/7,10)
    assert np.isclose((1-1)*10/9,0) and np.isclose((10-1)*10/9,10)
    v,s=build_hierarchical_component(pd.Series([5.0]),[(pd.Series([1.0]),"proxy")]); assert v.iloc[0]==5.0 and s.iloc[0]=="observed_esspri"
    for c in [x for x in longitudinal.columns if x.startswith("pop_status_s")]:
        assert (longitudinal.loc[longitudinal["essdai_total"].ge(5), c] == "Pop1").all()
    complete=longitudinal[["esspri_total_s0_observed","esspri_total_s1_one_proxy","esspri_total_s2_up_to_two_proxies","esspri_total_s3_all_available"]].dropna()
    assert np.allclose(complete["esspri_total_s1_one_proxy"], complete["esspri_total_s0_observed"])
    assert np.allclose(complete["esspri_total_s2_up_to_two_proxies"], complete["esspri_total_s0_observed"])
    for c in [c for c in longitudinal.columns if c.startswith(("fatigue_proxy_","pain_proxy_","dryness_proxy_")) and not c.endswith(("_raw","_source","_included")) and "n_available" not in c]:
        vals=pd.to_numeric(longitudinal[c],errors="coerce").dropna(); assert vals.between(0,10).all(), c
    for val_col, src_col in [("fatigue_proxy_hierarchical","fatigue_proxy_hierarchical_source"),("pain_proxy_hierarchical","pain_proxy_hierarchical_source"),("dryness_proxy_hierarchical","dryness_proxy_hierarchical_source")]:
        assert ~((longitudinal[val_col].notna()) & longitudinal[src_col].eq("missing")).any()
    assert (longitudinal["esspri_n_available_components"] == longitudinal["esspri_n_observed_components"] + longitudinal["esspri_n_proxy_components"]).all()
    synth=pd.DataFrame({"esspri_dryness_observed":[4.0],"esspri_fatigue_observed":[8.0],"esspri_pain_observed":[4.0],"fatigue_proxy_f1_profad":[2.0]})
    obs=compute_esspri_from_components(synth["esspri_dryness_observed"],synth["esspri_fatigue_observed"],synth["esspri_pain_observed"])
    repl=compute_esspri_from_components(synth["esspri_dryness_observed"],synth["fatigue_proxy_f1_profad"],synth["esspri_pain_observed"])
    assert not np.isclose(obs.iloc[0], repl.iloc[0])


def make_proxy_figures(longitudinal: pd.DataFrame, proxy_validation: pd.DataFrame, threshold_agreement: pd.DataFrame, proxy_sensitivity: pd.DataFrame) -> None:
    BLOCKA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    domains = {"fatigue": ("esspri_fatigue_observed", "fatigue_proxy_f1_profad"), "pain": ("esspri_pain_observed", "pain_proxy_p1_limb"), "dryness": ("esspri_dryness_observed", "dryness_proxy_d1_eye_mouth")}
    for dom, (obs, prox) in domains.items():
        pair = longitudinal[["patient_id", obs, prox]].dropna()
        fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter(pair[obs], pair[prox], s=10, alpha=.5); ax.plot([0,10],[0,10], color="black", ls="--")
        m = continuous_metrics(pair, obs, prox); ax.set(xlim=(0,10), ylim=(0,10), xlabel="Observed ESSPRI component", ylabel="Proxy", title=f"{dom} observed vs proxy")
        ax.text(.03,.97,f"n visits={len(pair)}\nn patients={pair['patient_id'].nunique() if len(pair) else 0}\nMAE={m['mae']:.2f}\nRMSE={m['rmse']:.2f}\nICC={m['icc_2_1']:.2f}\nSpearman={m['spearman_rho']:.2f}", transform=ax.transAxes, va="top")
        fig.tight_layout(); fig.savefig(BLOCKA_FIGURES_DIR / f"01_esspri_proxy_scatter_{dom}.pdf"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4));
        if len(pair):
            mean=(pair[obs]+pair[prox])/2; diff=pair[prox]-pair[obs]; ax.scatter(mean,diff,s=10,alpha=.5); ax.axhline(diff.mean(),color="black"); ax.axhline(diff.mean()+1.96*diff.std(ddof=1),color="red",ls="--"); ax.axhline(diff.mean()-1.96*diff.std(ddof=1),color="red",ls="--")
        ax.set(xlabel="Mean observed/proxy", ylabel="Proxy - observed", title=f"{dom} Bland-Altman"); fig.tight_layout(); fig.savefig(BLOCKA_FIGURES_DIR / f"01_esspri_proxy_bland_altman_{dom}.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); th = threshold_agreement.head(15)
    if not th.empty: ax.bar(th["candidate_id"], th["esspri_total_threshold_5_discordant"].fillna(0)); ax.tick_params(axis="x", rotation=90)
    ax.set(title="ESSPRI ≥5 discordance by proxy", ylabel="Discordant visits"); fig.tight_layout(); fig.savefig(BLOCKA_FIGURES_DIR / "01_esspri_proxy_threshold_confusion.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); ps = proxy_sensitivity[proxy_sensitivity["time_scope"].eq("all_visits")]
    if not ps.empty: ax.bar(ps["scenario"], ps["n_visits_classifiable"]); ax.tick_params(axis="x", rotation=90)
    ax.set(title="Classifiable visits by proxy scenario", ylabel="n visits"); fig.tight_layout(); fig.savefig(BLOCKA_FIGURES_DIR / "01_pop_distribution_proxy_sensitivity.pdf"); plt.close(fig)
def write_parquet_with_csv(df: pd.DataFrame, parquet_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    normalize_visit_level_dtypes(df).to_parquet(parquet_path, index=False)
    df.to_csv(parquet_path.with_suffix(".csv"), index=False)

def write_outputs(
    table1: pd.DataFrame,
    longitudinal: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_unclassifiable: pd.DataFrame,
    distribution_visit: pd.DataFrame,
    visit_unclassifiable_counts: pd.DataFrame,
    visit_unclassifiable_rows: pd.DataFrame,
    by_visit_qc: dict,
    qc: dict,
    claim: str,
    proxy_validation: pd.DataFrame,
    proxy_sensitivity: pd.DataFrame,
    proxy_total_validation: pd.DataFrame,
    threshold_agreement: pd.DataFrame,
    pop2pop3_reclassification: pd.DataFrame,
    pop2pop3_matrix: pd.DataFrame,
    proxy_coverage: pd.DataFrame,
    proxy_rescued: pd.DataFrame,
    proxy_ranking: pd.DataFrame,
) -> None:
    BLOCKA_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    BLOCKA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    BLOCKA_QC_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet_with_csv(longitudinal, INTERMEDIATE_DIR / "01_visit_level_classification.parquet")
    write_parquet_with_csv(longitudinal, INTERMEDIATE_DIR / "01_visit_level_esspri_proxy.parquet")
    write_parquet_with_csv(baseline, INTERMEDIATE_DIR / "01_baseline_classification.parquet")
    table1.to_csv(BLOCKA_TABLES_DIR / "01_table1_by_pop.csv", index=False)
    longitudinal[["patient_id", "visit_id", "visit_date", "observed_baseline_date", "time_since_observed_baseline_days", "time_since_observed_baseline_years", "baseline_date", "event_date", "visit_date_clean", "visit_number", "baseline_pop_status", "baseline_pop_status_display", "time_since_baseline_days", "time_since_baseline_years", "time_years", "essdai_total", "esspri_total", "pop_status", "pop_status_display", "row_date_original", "row_date_min", "row_date_max"]].to_csv(BLOCKA_TABLES_DIR / "01_pop_longitudinal_status.csv", index=False)
    counts = longitudinal.groupby(["pop_status"]).size().reindex(POP_ORDER, fill_value=0).rename("n_visits").reset_index()
    baseline_counts = baseline["pop_status"].value_counts().reindex(POP_ORDER, fill_value=0).rename_axis("pop_status").reset_index(name="n_baseline_patients")
    baseline_unclassifiable.to_csv(BLOCKA_TABLES_DIR / "01_pop_unclassifiable_baseline_essdai_esspri_status.csv", index=False)
    (
        baseline_unclassifiable["unclassifiable_reason"]
        .value_counts()
        .rename_axis("unclassifiable_reason")
        .reset_index(name="n_baseline_patients")
        .to_csv(BLOCKA_TABLES_DIR / "01_pop_unclassifiable_baseline_reason_counts.csv", index=False)
    )
    counts.merge(baseline_counts, on="pop_status", how="outer").to_csv(BLOCKA_TABLES_DIR / "01_pop_distribution_counts.csv", index=False)
    distribution_visit.to_csv(BLOCKA_TABLES_DIR / "01_pop_distribution_by_visit.csv", index=False)
    proxy_validation.to_csv(BLOCKA_TABLES_DIR / "01_esspri_proxy_validation.csv", index=False)
    proxy_validation.to_csv(BLOCKA_TABLES_DIR / "01_esspri_proxy_component_validation.csv", index=False)
    proxy_total_validation.to_csv(BLOCKA_TABLES_DIR / "01_esspri_proxy_total_validation.csv", index=False)
    threshold_agreement.to_csv(BLOCKA_TABLES_DIR / "01_esspri_proxy_threshold_agreement.csv", index=False)
    pop2pop3_reclassification.to_csv(BLOCKA_TABLES_DIR / "01_pop2_pop3_proxy_reclassification.csv", index=False)
    pop2pop3_matrix.to_csv(BLOCKA_TABLES_DIR / "01_pop2_pop3_proxy_reclassification_matrix.csv", index=False)
    proxy_coverage.to_csv(BLOCKA_TABLES_DIR / "01_esspri_proxy_coverage.csv", index=False)
    proxy_rescued.to_csv(BLOCKA_TABLES_DIR / "01_esspri_proxy_rescued_visits.csv", index=False)
    proxy_sensitivity.to_csv(BLOCKA_TABLES_DIR / "01_esspri_proxy_sensitivity_summary.csv", index=False)
    proxy_sensitivity.to_csv(BLOCKA_TABLES_DIR / "01_pop_distribution_proxy_sensitivity.csv", index=False)
    proxy_ranking.to_csv(BLOCKA_TABLES_DIR / "01_esspri_proxy_candidate_ranking.csv", index=False)
    make_proxy_figures(longitudinal, proxy_validation, threshold_agreement, proxy_sensitivity)
    pd.DataFrame({"variable": longitudinal.columns, "n_nonmissing": [int(longitudinal[c].notna().sum()) for c in longitudinal.columns]}).to_csv(BLOCKA_QC_DIR / "01_esspri_proxy_variable_availability.csv", index=False)
    with (BLOCKA_QC_DIR / "01_esspri_proxy_qc.json").open("w", encoding="utf-8") as f: json.dump(qc, f, indent=2, default=str)
    visit_unclassifiable_counts.to_csv(BLOCKA_TABLES_DIR / "01_pop_unclassifiable_reason_counts_by_visit.csv", index=False)
    write_parquet_with_csv(visit_unclassifiable_rows, INTERMEDIATE_DIR / "01_unclassifiable_reasons_visit_level.parquet")
    (BLOCKA_TABLES_DIR / "01_pop_distribution_claim.txt").write_text(claim + "\n", encoding="utf-8")
    with (BLOCKA_QC_DIR / "01_pop_distribution_qc.json").open("w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2, default=str)
    with (BLOCKA_QC_DIR / "01_pop_distribution_by_visit_qc.json").open("w", encoding="utf-8") as f:
        json.dump(by_visit_qc, f, indent=2, default=str)


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    codebook = load_codebook(args.codebook)
    selected = validate_columns(df)
    longitudinal, row_qc, warnings = build_longitudinal_pop_dataset(df, codebook)
    baseline = build_baseline_dataset(longitudinal)
    table1, selected_demo = summarize_table1_by_pop(baseline, codebook, warnings)
    baseline_unclassifiable = describe_baseline_unclassifiable(baseline)
    outside = make_pop_swimmer_plot(longitudinal, baseline, BLOCKA_FIGURES_DIR / "02_pop_distribution_plot.pdf", warnings)
    distribution_visit = distribution_by_visit(longitudinal)
    visit_unclassifiable_counts, visit_unclassifiable_rows = describe_visit_unclassifiable(longitudinal)
    run_internal_consistency_tests(longitudinal)
    proxy_validation = build_proxy_validation(longitudinal)
    proxy_total_validation = build_proxy_total_validation(longitudinal)
    threshold_agreement = build_threshold_agreement(longitudinal)
    pop2pop3_reclassification, pop2pop3_matrix = build_pop2_pop3_reclassification(longitudinal)
    proxy_coverage, proxy_rescued, proxy_ranking_base = build_coverage_and_rescue(longitudinal)
    proxy_sensitivity = build_proxy_sensitivity_summary(longitudinal)
    proxy_ranking = build_candidate_ranking(proxy_validation, proxy_total_validation, threshold_agreement, proxy_coverage)
    n_total_patients = int(baseline["patient_id"].nunique())
    baseline_counts = baseline["pop_status"].value_counts().reindex(POP_ORDER, fill_value=0)
    n_classifiable = int(baseline_counts[["Pop1", "Pop2", "Pop3"]].sum())
    if int(baseline_counts["Pop1"] + baseline_counts["Pop2"] + baseline_counts["Pop3"]) != n_classifiable:
        raise AssertionError("Pop1 + Pop2 + Pop3 does not equal n_classifiable_baseline")
    if (longitudinal["time_years"] < 0).any():
        raise AssertionError("Negative time_years remains after filtering")
    pct = {pop: (100 * int(baseline_counts[pop]) / n_classifiable if n_classifiable else 0.0) for pop in ["Pop1", "Pop2", "Pop3"]}
    claim = (
        f"At baseline, {pct['Pop1']:.1f}% of classifiable patients were classified as Pop 1 (ESSDAI ≥5), "
        f"{pct['Pop2']:.1f}% as Pop 2 (ESSDAI <5 and ESSPRI ≥5), and "
        f"{pct['Pop3']:.1f}% as Pop 3 (ESSDAI <5 and ESSPRI <5). "
        f"{int(baseline_counts['Unclassifiable'])} patients were unclassifiable at baseline because of missing ESSDAI/ESSPRI components."
    )
    qc = {
        "n_total_rows": int(len(df)),
        "n_total_patients": n_total_patients,
        "n_patients_classifiable_baseline": n_classifiable,
        "n_pop1_baseline": int(baseline_counts["Pop1"]),
        "n_pop2_baseline": int(baseline_counts["Pop2"]),
        "n_pop3_baseline": int(baseline_counts["Pop3"]),
        "n_unclassifiable_baseline": int(baseline_counts["Unclassifiable"]),
        "unclassifiable_baseline_reason_counts": baseline_unclassifiable["unclassifiable_reason"]
        .value_counts()
        .to_dict(),
        "pct_pop1": pct["Pop1"],
        "pct_pop2": pct["Pop2"],
        "pct_pop3": pct["Pop3"],
        "pct_unclassifiable_baseline": (100 * int(baseline_counts["Unclassifiable"]) / n_total_patients if n_total_patients else 0.0),
        "selected_essdai_columns": selected["essdai"],
        "selected_esspri_component_columns": selected["esspri"],
        "missing_esspri_component_columns": selected.get("missing_esspri", []),
        "proxy_sensitivity_summary": proxy_sensitivity.to_dict("records"),
        **selected_demo,
        **row_qc,
        "n_plot_points_outside_xlim": outside,
        "warnings": warnings,
        "manuscript_claim": claim,
    }
    by_visit_qc = build_by_visit_qc(longitudinal, outside, warnings)
    write_outputs(
        table1,
        longitudinal,
        baseline,
        baseline_unclassifiable,
        distribution_visit,
        visit_unclassifiable_counts,
        visit_unclassifiable_rows,
        by_visit_qc,
        qc,
        claim,
        proxy_validation,
        proxy_sensitivity,
        proxy_total_validation,
        threshold_agreement,
        pop2pop3_reclassification,
        pop2pop3_matrix,
        proxy_coverage,
        proxy_rescued,
        proxy_ranking,
    )
    print(claim)


if __name__ == "__main__":
    main()

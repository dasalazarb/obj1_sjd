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
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402

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
POP_ORDER = ["Pop1", "Pop2", "Pop3", "Unclassifiable"]
POP_COLORS = {"Pop1": "#d95f02", "Pop2": "#7570b3", "Pop3": "#1b9e77", "Unclassifiable": "#9e9e9e"}
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


def parse_visit_dates(value: object) -> list[pd.Timestamp]:
    if is_missing(value):
        return []
    dates = []
    for fragment in str(value).split("|"):
        parsed = pd.to_datetime(fragment.strip(), errors="coerce")
        if pd.notna(parsed):
            dates.append(pd.Timestamp(parsed).normalize())
    return dates


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


def source_from_observed_proxy(observed: pd.Series, proxy: pd.Series, proxy_source: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([observed.notna(), proxy.notna()], ["observed_esspri", proxy_source], default="missing"),
        index=observed.index,
        dtype="object",
    )


def populate_esspri_proxies(work: pd.DataFrame, qc_counts: dict[str, int]) -> pd.DataFrame:
    for key, src in ESSPRI_OBSERVED_COMPONENTS.items():
        work[f"esspri_{key}_observed"] = numeric_optional(work, src)

    observed_cols = ["esspri_dryness_observed", "esspri_fatigue_observed", "esspri_pain_observed"]
    invalid = pd.DataFrame({c: (work[c] < 0) | (work[c] > 10) for c in observed_cols})
    qc_counts["n_invalid_esspri_component_values"] = int(invalid.sum().sum())
    if qc_counts["n_invalid_esspri_component_values"]:
        bad = work.loc[invalid.any(axis=1), ["patient_id", "row_date_original", *observed_cols]].head(25).to_dict("records")
        qc_counts["invalid_esspri_component_examples"] = bad
        raise ValueError(f"ESSPRI observed component values outside range 0-10: {qc_counts['n_invalid_esspri_component_values']}")

    work["esspri_total_observed"] = compute_esspri_from_components(*(work[c] for c in observed_cols))

    profad_fatigue = pd.DataFrame({c: numeric_optional(work, c) for c in FATIGUE_PROFAD_ITEMS})
    work["n_available_profad_fatigue"] = profad_fatigue.notna().sum(axis=1)
    work["fatigue_proxy_f1_profad_raw"] = mean_when(profad_fatigue, work["n_available_profad_fatigue"].eq(4))
    work["fatigue_proxy_f1_profad_relaxed_raw"] = mean_when(profad_fatigue, work["n_available_profad_fatigue"].ge(3))
    work["fatigue_proxy_f1_profad"] = work["fatigue_proxy_f1_profad_raw"] * 10 / 7
    work["fatigue_proxy_f1_profad_relaxed"] = work["fatigue_proxy_f1_profad_relaxed_raw"] * 10 / 7
    work["fatigue_proxy_f2_mdafs_severity_raw"] = numeric_optional(work, "multidimensional_assessment_of_fatigue_scale__fat_q2")
    work["fatigue_proxy_f2_mdafs_severity"] = (work["fatigue_proxy_f2_mdafs_severity_raw"] - 1) * 10 / 9
    work["fatigue_proxy_f2_mdafs_direct"] = work["fatigue_proxy_f2_mdafs_severity_raw"]
    work["fatigue_proxy_f3_mdafs_degree_raw"] = numeric_optional(work, "multidimensional_assessment_of_fatigue_scale__fat_q1")
    work["fatigue_proxy_f3_mdafs_degree"] = (work["fatigue_proxy_f3_mdafs_degree_raw"] - 1) * 10 / 9
    work["fatigue_proxy_f3_mdafs_direct"] = work["fatigue_proxy_f3_mdafs_degree_raw"]
    ans1, ans2 = (numeric_optional(work, c) for c in FATIGUE_ANS_CANDIDATES)
    conflict = ans1.notna() & ans2.notna() & ans1.ne(ans2)
    qc_counts["n_fatigue_ans_conflicts"] = int(conflict.sum())
    work["fatigue_proxy_f4_ans"] = ans1.combine_first(ans2)

    work["pain_proxy_p1_limb_raw"] = numeric_optional(work, "profile_of_fatigue_and_discomfort__profad_limb_discomfort")
    work["pain_proxy_p1_limb"] = work["pain_proxy_p1_limb_raw"] * 10 / 7
    work["pain_proxy_p2_finger_wrist_raw"] = numeric_optional(work, "profile_of_fatigue_and_discomfort__profad_finger_wrist_discomfort")
    work["pain_proxy_p2_finger_wrist"] = work["pain_proxy_p2_finger_wrist_raw"] * 10 / 7
    work["pain_proxy_p12_composite"] = pd.concat([work["pain_proxy_p1_limb"], work["pain_proxy_p2_finger_wrist"]], axis=1).mean(axis=1).where(work[["pain_proxy_p1_limb", "pain_proxy_p2_finger_wrist"]].notna().all(axis=1), np.nan)
    work["pain_strategy_hierarchy"] = work["pain_proxy_p1_limb"].combine_first(work["pain_proxy_p2_finger_wrist"])
    work["pain_strategy_composite"] = pd.concat([work["pain_proxy_p1_limb"], work["pain_proxy_p2_finger_wrist"]], axis=1).mean(axis=1)

    d1 = pd.DataFrame({c: numeric_optional(work, c) for c in DRYNESS_D1_ITEMS})
    work["dryness_proxy_d1_n_available"] = d1.notna().sum(axis=1)
    work["dryness_proxy_d1_eye_mouth"] = mean_when(d1, work["dryness_proxy_d1_n_available"].eq(2))
    work["dryness_proxy_d1_eye_mouth_relaxed"] = mean_when(d1, work["dryness_proxy_d1_n_available"].ge(1))
    d2_core = pd.DataFrame({c: numeric_optional(work, c) for c in DRYNESS_D2_CORE_ITEMS})
    core_ok = d2_core.notna().sum(axis=1).ge(3) & d2_core[DRYNESS_D1_ITEMS].notna().any(axis=1)
    work["dryness_proxy_d2_core"] = mean_when(d2_core, core_ok)
    vaginal = numeric_optional(work, "esspri_questionnaire__vaginal_dryness")
    vaginal_applicable = ~work.get(DRYNESS_D2_VAGINAL_NA, pd.Series(np.nan, index=work.index)).map(is_not_applicable)
    vaginal_eval = vaginal.where(vaginal_applicable)
    d2_ext = d2_core.assign(esspri_questionnaire__vaginal_dryness=vaginal_eval)
    work["dryness_proxy_d2_n_available"] = d2_ext.notna().sum(axis=1)
    work["dryness_proxy_d2_vaginal_included"] = vaginal_eval.notna()
    work["dryness_proxy_d2_extended"] = mean_when(d2_ext, core_ok)
    d3 = pd.DataFrame({c: numeric_optional(work, c) for c in DRYNESS_D3_PROFAD_ITEMS})
    work["dryness_proxy_d3_n_available"] = d3.notna().sum(axis=1)
    d3_has_ocular = d3[DRYNESS_D3_OCULAR_ITEMS].notna().any(axis=1)
    d3_has_oral = d3[DRYNESS_D3_ORAL_AIRWAY_ITEMS].notna().any(axis=1)
    work["dryness_proxy_d3_profad_raw"] = mean_when(d3, work["dryness_proxy_d3_n_available"].ge(4) & d3_has_ocular & d3_has_oral)
    work["dryness_proxy_d3_profad_relaxed_raw"] = mean_when(d3, work["dryness_proxy_d3_n_available"].ge(3) & d3_has_ocular & d3_has_oral)
    work["dryness_proxy_d3_profad"] = work["dryness_proxy_d3_profad_raw"] * 10 / 7
    work["dryness_proxy_d3_profad_relaxed"] = work["dryness_proxy_d3_profad_relaxed_raw"] * 10 / 7

    def hierarchy(observed: pd.Series, candidates: list[tuple[str, str]]) -> tuple[pd.Series, pd.Series]:
        value = pd.to_numeric(observed, errors="coerce").astype("float64")
        source = pd.Series("missing", index=work.index, dtype="object")
        source.loc[value.notna()] = "observed_esspri"
        for col, label in candidates:
            candidate = pd.to_numeric(work[col], errors="coerce").astype("float64")
            use = value.isna() & candidate.notna()
            value.loc[use] = candidate.loc[use]
            source.loc[use] = label
        return value, source
    work["fatigue_proxy_hierarchical"], work["fatigue_proxy_hierarchical_source"] = hierarchy(work["esspri_fatigue_observed"], [("fatigue_proxy_f1_profad", "proxy_f1_profad"), ("fatigue_proxy_f2_mdafs_severity", "proxy_f2_mdafs_severity"), ("fatigue_proxy_f3_mdafs_degree", "proxy_f3_mdafs_degree"), ("fatigue_proxy_f4_ans", "proxy_f4_ans")])
    work["fatigue_proxy_hierarchical_relaxed"], work["fatigue_proxy_hierarchical_relaxed_source"] = hierarchy(work["esspri_fatigue_observed"], [("fatigue_proxy_f1_profad_relaxed", "proxy_f1_profad"), ("fatigue_proxy_f2_mdafs_severity", "proxy_f2_mdafs_severity"), ("fatigue_proxy_f3_mdafs_degree", "proxy_f3_mdafs_degree"), ("fatigue_proxy_f4_ans", "proxy_f4_ans")])
    work["pain_proxy_hierarchical"], work["pain_proxy_hierarchical_source"] = hierarchy(work["esspri_pain_observed"], [("pain_proxy_p1_limb", "proxy_p1_limb"), ("pain_proxy_p2_finger_wrist", "proxy_p2_finger_wrist")])
    work["dryness_proxy_hierarchical"], work["dryness_proxy_hierarchical_source"] = hierarchy(work["esspri_dryness_observed"], [("dryness_proxy_d1_eye_mouth", "proxy_d1_eye_mouth"), ("dryness_proxy_d2_core", "proxy_d2_core"), ("dryness_proxy_d3_profad", "proxy_d3_profad")])
    work["dryness_proxy_hierarchical_relaxed"], work["dryness_proxy_hierarchical_relaxed_source"] = hierarchy(work["esspri_dryness_observed"], [("dryness_proxy_d1_eye_mouth_relaxed", "proxy_d1_eye_mouth"), ("dryness_proxy_d2_core", "proxy_d2_core"), ("dryness_proxy_d3_profad_relaxed", "proxy_d3_profad")])
    return work


def add_esspri_scenarios(work: pd.DataFrame, relaxed: bool = False) -> pd.DataFrame:
    suffix = "_relaxed" if relaxed else ""
    dry = "dryness_proxy_hierarchical_relaxed" if relaxed else "dryness_proxy_hierarchical"
    fat = "fatigue_proxy_hierarchical_relaxed" if relaxed else "fatigue_proxy_hierarchical"
    pain = "pain_proxy_hierarchical"
    for comp, col in [("dryness", dry), ("fatigue", fat), ("pain", pain)]:
        work[f"esspri_{comp}_best_available{suffix}"] = work[col]
    best_cols = [f"esspri_dryness_best_available{suffix}", f"esspri_fatigue_best_available{suffix}", f"esspri_pain_best_available{suffix}"]
    obs_cols = ["esspri_dryness_observed", "esspri_fatigue_observed", "esspri_pain_observed"]
    work[f"esspri_n_observed_components{suffix}"] = work[obs_cols].notna().sum(axis=1)
    work[f"esspri_n_available_components{suffix}"] = work[best_cols].notna().sum(axis=1)
    work[f"esspri_n_proxy_components{suffix}"] = work[f"esspri_n_available_components{suffix}"] - work[f"esspri_n_observed_components{suffix}"]
    work[f"esspri_total_proxy{suffix}"] = compute_esspri_from_components(*(work[c] for c in best_cols))
    scen = np.select([work[f"esspri_n_available_components{suffix}"].lt(3), work[f"esspri_n_observed_components{suffix}"].eq(3), work[f"esspri_n_proxy_components{suffix}"].eq(1), work[f"esspri_n_proxy_components{suffix}"].eq(2), work[f"esspri_n_proxy_components{suffix}"].eq(3)], ["unavailable", "observed_complete", "one_proxy", "two_proxies", "three_proxies"], default="unavailable")
    work[f"esspri_derivation_scenario{suffix}"] = scen
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


def build_longitudinal_pop_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    work = df.copy()
    qc_counts: dict[str, int] = {"n_input_rows": int(len(work))}
    warnings: list[str] = []

    work["patient_id"] = work[PATIENT_ID_COL].astype("string").str.strip()
    valid_patient_id = ~work["patient_id"].map(is_missing)
    qc_counts["n_rows_with_valid_patient_id"] = int(valid_patient_id.sum())
    qc_counts["n_rows_excluded_missing_patient_id"] = int((~valid_patient_id).sum())

    work["row_date_original"] = work[VISIT_DATE_COL]
    parsed_lists = work[VISIT_DATE_COL].map(parse_visit_dates)
    work["row_date_min"] = parsed_lists.map(lambda x: min(x) if x else pd.NaT)
    work["row_date_max"] = parsed_lists.map(lambda x: max(x) if x else pd.NaT)
    work["visit_date_clean"] = work["row_date_min"]
    valid_visit_date = work["visit_date_clean"].notna()
    qc_counts["n_rows_with_valid_visit_date"] = int(valid_visit_date.sum())
    qc_counts["n_rows_excluded_missing_visit_date"] = int((~valid_visit_date).sum())

    work["essdai_total"] = coalesce_essdai(work)
    work = populate_esspri_proxies(work, qc_counts)

    out_of_range_essdai = int(((work["essdai_total"] < 0) | (work["essdai_total"] > 123)).sum())
    qc_counts["n_invalid_essdai_values"] = out_of_range_essdai
    if out_of_range_essdai:
        raise ValueError(f"ESSDAI values outside plausible range 0-123: {out_of_range_essdai}")

    work = work[valid_patient_id & valid_visit_date].copy()
    qc_counts["n_unique_patients"] = int(work["patient_id"].nunique())
    qc_counts["n_patient_visit_rows_before_collapse"] = int(len(work))
    qc_counts["n_duplicate_patient_event_dates"] = int(work.duplicated(["patient_id", "visit_date_clean"]).sum())
    qc_counts["n_duplicate_patient_visit_rows_before_collapse"] = qc_counts["n_duplicate_patient_event_dates"]

    work = (
        work.groupby(["patient_id", "visit_date_clean"], as_index=False)
        .agg(
            row_date_original=("row_date_original", lambda s: " | ".join(pd.Series(s).dropna().astype(str).unique())),
            row_date_min=("row_date_min", "min"),
            row_date_max=("row_date_max", "max"),
            essdai_total=("essdai_total", first_nonmissing),
            esspri_dryness_observed=("esspri_dryness_observed", first_nonmissing),
            esspri_fatigue_observed=("esspri_fatigue_observed", first_nonmissing),
            esspri_pain_observed=("esspri_pain_observed", first_nonmissing),
            esspri_total_observed=("esspri_total_observed", first_nonmissing),
            n_available_profad_fatigue=("n_available_profad_fatigue", first_nonmissing),
            fatigue_proxy_f1_profad_raw=("fatigue_proxy_f1_profad_raw", first_nonmissing),
            fatigue_proxy_f1_profad_relaxed_raw=("fatigue_proxy_f1_profad_relaxed_raw", first_nonmissing),
            fatigue_proxy_f1_profad=("fatigue_proxy_f1_profad", first_nonmissing),
            fatigue_proxy_f1_profad_relaxed=("fatigue_proxy_f1_profad_relaxed", first_nonmissing),
            fatigue_proxy_f2_mdafs_severity_raw=("fatigue_proxy_f2_mdafs_severity_raw", first_nonmissing),
            fatigue_proxy_f2_mdafs_severity=("fatigue_proxy_f2_mdafs_severity", first_nonmissing),
            fatigue_proxy_f2_mdafs_direct=("fatigue_proxy_f2_mdafs_direct", first_nonmissing),
            fatigue_proxy_f3_mdafs_degree_raw=("fatigue_proxy_f3_mdafs_degree_raw", first_nonmissing),
            fatigue_proxy_f3_mdafs_degree=("fatigue_proxy_f3_mdafs_degree", first_nonmissing),
            fatigue_proxy_f3_mdafs_direct=("fatigue_proxy_f3_mdafs_direct", first_nonmissing),
            fatigue_proxy_f4_ans=("fatigue_proxy_f4_ans", first_nonmissing),
            pain_proxy_p1_limb_raw=("pain_proxy_p1_limb_raw", first_nonmissing),
            pain_proxy_p1_limb=("pain_proxy_p1_limb", first_nonmissing),
            pain_proxy_p2_finger_wrist_raw=("pain_proxy_p2_finger_wrist_raw", first_nonmissing),
            pain_proxy_p2_finger_wrist=("pain_proxy_p2_finger_wrist", first_nonmissing),
            pain_proxy_p12_composite=("pain_proxy_p12_composite", first_nonmissing),
            pain_strategy_hierarchy=("pain_strategy_hierarchy", first_nonmissing),
            pain_strategy_composite=("pain_strategy_composite", first_nonmissing),
            dryness_proxy_d1_n_available=("dryness_proxy_d1_n_available", first_nonmissing),
            dryness_proxy_d1_eye_mouth=("dryness_proxy_d1_eye_mouth", first_nonmissing),
            dryness_proxy_d1_eye_mouth_relaxed=("dryness_proxy_d1_eye_mouth_relaxed", first_nonmissing),
            dryness_proxy_d2_core=("dryness_proxy_d2_core", first_nonmissing),
            dryness_proxy_d2_n_available=("dryness_proxy_d2_n_available", first_nonmissing),
            dryness_proxy_d2_vaginal_included=("dryness_proxy_d2_vaginal_included", first_nonmissing),
            dryness_proxy_d2_extended=("dryness_proxy_d2_extended", first_nonmissing),
            dryness_proxy_d3_n_available=("dryness_proxy_d3_n_available", first_nonmissing),
            dryness_proxy_d3_profad_raw=("dryness_proxy_d3_profad_raw", first_nonmissing),
            dryness_proxy_d3_profad_relaxed_raw=("dryness_proxy_d3_profad_relaxed_raw", first_nonmissing),
            dryness_proxy_d3_profad=("dryness_proxy_d3_profad", first_nonmissing),
            dryness_proxy_d3_profad_relaxed=("dryness_proxy_d3_profad_relaxed", first_nonmissing),
            fatigue_proxy_hierarchical=("fatigue_proxy_hierarchical", first_nonmissing),
            fatigue_proxy_hierarchical_source=("fatigue_proxy_hierarchical_source", first_nonmissing),
            fatigue_proxy_hierarchical_relaxed=("fatigue_proxy_hierarchical_relaxed", first_nonmissing),
            fatigue_proxy_hierarchical_relaxed_source=("fatigue_proxy_hierarchical_relaxed_source", first_nonmissing),
            pain_proxy_hierarchical=("pain_proxy_hierarchical", first_nonmissing),
            pain_proxy_hierarchical_source=("pain_proxy_hierarchical_source", first_nonmissing),
            dryness_proxy_hierarchical=("dryness_proxy_hierarchical", first_nonmissing),
            dryness_proxy_hierarchical_source=("dryness_proxy_hierarchical_source", first_nonmissing),
            dryness_proxy_hierarchical_relaxed=("dryness_proxy_hierarchical_relaxed", first_nonmissing),
            dryness_proxy_hierarchical_relaxed_source=("dryness_proxy_hierarchical_relaxed_source", first_nonmissing),
        )
        .sort_values(["patient_id", "visit_date_clean"])
        .reset_index(drop=True)
    )
    work["esspri_total_observed"] = compute_esspri_from_components(work["esspri_dryness_observed"], work["esspri_fatigue_observed"], work["esspri_pain_observed"])
    work = add_esspri_scenarios(work, relaxed=False)
    work = add_esspri_scenarios(work, relaxed=True)
    work["esspri_dryness"] = work["esspri_dryness_observed"]
    work["esspri_fatigue"] = work["esspri_fatigue_observed"]
    work["esspri_pain"] = work["esspri_pain_observed"]
    work["esspri_total"] = work["esspri_total_observed"]
    work["esspri_total_s0_observed"] = work["esspri_total_observed"]
    work["esspri_total_proxy"] = work["esspri_total_proxy"]
    for comp, src_col in [("dryness", "dryness_proxy_hierarchical_source"), ("fatigue", "fatigue_proxy_hierarchical_source"), ("pain", "pain_proxy_hierarchical_source")]:
        work[f"esspri_{comp}_final_source"] = work[src_col]
    mask_s1 = work["esspri_n_observed_components"].ge(2) & work["esspri_n_available_components"].eq(3)
    mask_s2 = work["esspri_n_observed_components"].ge(1) & work["esspri_n_available_components"].eq(3)
    mask_s3 = work["esspri_n_available_components"].eq(3)
    work["esspri_total_s1_one_proxy"] = work["esspri_total_proxy"].where(mask_s1)
    work["esspri_total_s2_up_to_two_proxies"] = work["esspri_total_proxy"].where(mask_s2)
    work["esspri_total_s3_all_available"] = work["esspri_total_proxy"].where(mask_s3)
    work["esspri_total_s3_all_available_label"] = np.where(work["esspri_total_s3_all_available"].notna(), "exploratory_only", np.nan)
    mask_s1r = work["esspri_n_observed_components_relaxed"].ge(2) & work["esspri_n_available_components_relaxed"].eq(3)
    mask_s2r = work["esspri_n_observed_components_relaxed"].ge(1) & work["esspri_n_available_components_relaxed"].eq(3)
    work["esspri_total_s1_one_proxy_relaxed"] = work["esspri_total_proxy_relaxed"].where(mask_s1r)
    work["esspri_total_s2_up_to_two_proxies_relaxed"] = work["esspri_total_proxy_relaxed"].where(mask_s2r)
    work["pop_status_s1_one_proxy"] = [classify_pop(e, p) for e, p in zip(work["essdai_total"], work["esspri_total_s1_one_proxy"])]
    work["pop_status_s2_up_to_two_proxies"] = [classify_pop(e, p) for e, p in zip(work["essdai_total"], work["esspri_total_s2_up_to_two_proxies"])]
    work["pop_status_s3_all_available"] = [classify_pop(e, p) for e, p in zip(work["essdai_total"], work["esspri_total_s3_all_available"])]
    work["pop_status_s1_one_proxy_relaxed"] = [classify_pop(e, p) for e, p in zip(work["essdai_total"], work["esspri_total_s1_one_proxy_relaxed"])]
    work["pop_status_s2_up_to_two_proxies_relaxed"] = [classify_pop(e, p) for e, p in zip(work["essdai_total"], work["esspri_total_s2_up_to_two_proxies_relaxed"])]
    for name, comp, proxy in [
        ("fatigue_f1", "fatigue", "fatigue_proxy_f1_profad"), ("fatigue_f2", "fatigue", "fatigue_proxy_f2_mdafs_severity"), ("fatigue_f3", "fatigue", "fatigue_proxy_f3_mdafs_degree"), ("fatigue_f4", "fatigue", "fatigue_proxy_f4_ans"),
        ("pain_p1", "pain", "pain_proxy_p1_limb"), ("pain_p2", "pain", "pain_proxy_p2_finger_wrist"), ("pain_p12", "pain", "pain_proxy_p12_composite"),
        ("dryness_d1", "dryness", "dryness_proxy_d1_eye_mouth"), ("dryness_d2_core", "dryness", "dryness_proxy_d2_core"), ("dryness_d2_extended", "dryness", "dryness_proxy_d2_extended"), ("dryness_d3", "dryness", "dryness_proxy_d3_profad"),
    ]:
        d = pd.to_numeric(work["esspri_dryness_observed"], errors="coerce").astype("float64")
        f = pd.to_numeric(work["esspri_fatigue_observed"], errors="coerce").astype("float64")
        p = pd.to_numeric(work["esspri_pain_observed"], errors="coerce").astype("float64")
        proxy_value = pd.to_numeric(work[proxy], errors="coerce").astype("float64")
        if comp == "dryness":
            d = d.combine_first(proxy_value)
        elif comp == "fatigue":
            f = f.combine_first(proxy_value)
        else:
            p = p.combine_first(proxy_value)
        work[f"esspri_total_replace_{name}"] = compute_esspri_from_components(d, f, p)
    out_of_range_esspri = int(((work["esspri_total_observed"] < 0) | (work["esspri_total_observed"] > 10)).sum())
    if out_of_range_esspri:
        raise ValueError(f"ESSPRI observed total values outside range 0-10: {out_of_range_esspri}")

    work["pop_status"] = [classify_pop(e, p) for e, p in zip(work["essdai_total"], work["esspri_total_observed"])]
    work["pop_status_display"] = work["pop_status"].map(DISPLAY)
    work["baseline_date"] = work.groupby("patient_id")["visit_date_clean"].transform("min")
    work["event_date"] = work["visit_date_clean"]
    work["time_since_baseline_days"] = (work["event_date"] - work["baseline_date"]).dt.days
    work["time_since_baseline_years"] = work["time_since_baseline_days"] / 365.25
    work["time_years"] = work["time_since_baseline_years"]
    work["visit_number"] = work.groupby("patient_id").cumcount()
    baseline_status = work.loc[work["visit_number"].eq(0), ["patient_id", "pop_status", "pop_status_display"]].rename(
        columns={"pop_status": "baseline_pop_status", "pop_status_display": "baseline_pop_status_display"}
    )
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
        baseline_date = g["baseline_date"].iloc[0]
        baseline_rows = g[g["row_date_min"] == baseline_date].copy()
        if baseline_rows.empty:
            baseline_rows = g.sort_values(["row_date_min", "row_date_max"], na_position="last").head(1).copy()
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
    for var, label in [("essdai_total", "ESSDAI total, median [IQR]"), ("esspri_total", "ESSPRI total, median [IQR]")]:
        p, test = run_stat_tests(classifiable, var, "continuous", warnings)
        rows.append({"Variable": label, "Overall": fmt_median_iqr(classifiable[var]), **{pop: fmt_median_iqr(classifiable.loc[classifiable["pop_status"] == pop, var]) for pop in ["Pop1", "Pop2", "Pop3"]}, "p_value": p, "test": test, **counts_avail(var)})
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
            sub = panel_df[panel_df["pop_status"] == pop]
            ax.scatter(sub["time_years"], sub["y"], s=12, color=POP_COLORS[pop], label=pop, alpha=0.9, zorder=2)
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
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=False,
            ncol=4,
            columnspacing=2.5,
            handletextpad=0.8,
            borderaxespad=1.2,
        )
        fig.text(
            0.01,
            0.01,
            "Pop1 = ESSDAI ≥5; Pop2 = ESSDAI <5 and ESSPRI ≥5; Pop3 = ESSDAI <5 and ESSPRI <5; grey = insufficient data.",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
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



def build_proxy_validation(longitudinal: pd.DataFrame) -> pd.DataFrame:
    """Validate candidate proxies where official ESSPRI components are also observed."""
    candidates = [
        ("fatigue", "F1_PROFAD_strict", "esspri_fatigue_observed", "fatigue_proxy_f1_profad"),
        ("fatigue", "F1_PROFAD_relaxed", "esspri_fatigue_observed", "fatigue_proxy_f1_profad_relaxed"),
        ("fatigue", "F2_MDAFS_severity", "esspri_fatigue_observed", "fatigue_proxy_f2_mdafs_severity"),
        ("fatigue", "F2_MDAFS_direct", "esspri_fatigue_observed", "fatigue_proxy_f2_mdafs_direct"),
        ("fatigue", "F3_MDAFS_degree", "esspri_fatigue_observed", "fatigue_proxy_f3_mdafs_degree"),
        ("fatigue", "F3_MDAFS_direct", "esspri_fatigue_observed", "fatigue_proxy_f3_mdafs_direct"),
        ("fatigue", "F4_ANS", "esspri_fatigue_observed", "fatigue_proxy_f4_ans"),
        ("pain", "P1_limb", "esspri_pain_observed", "pain_proxy_p1_limb"),
        ("pain", "P2_finger_wrist", "esspri_pain_observed", "pain_proxy_p2_finger_wrist"),
        ("pain", "P12_composite", "esspri_pain_observed", "pain_proxy_p12_composite"),
        ("dryness", "D1_eye_mouth", "esspri_dryness_observed", "dryness_proxy_d1_eye_mouth"),
        ("dryness", "D1_eye_mouth_relaxed", "esspri_dryness_observed", "dryness_proxy_d1_eye_mouth_relaxed"),
        ("dryness", "D2_core", "esspri_dryness_observed", "dryness_proxy_d2_core"),
        ("dryness", "D2_extended", "esspri_dryness_observed", "dryness_proxy_d2_extended"),
        ("dryness", "D3_PROFAD", "esspri_dryness_observed", "dryness_proxy_d3_profad"),
        ("dryness", "D3_PROFAD_relaxed", "esspri_dryness_observed", "dryness_proxy_d3_profad_relaxed"),
    ]
    rows = []
    for component, proxy_name, observed_col, proxy_col in candidates:
        if observed_col not in longitudinal.columns or proxy_col not in longitudinal.columns:
            continue
        pair = longitudinal[[observed_col, proxy_col]].apply(pd.to_numeric, errors="coerce").dropna().astype("float64")
        diff = pair[proxy_col] - pair[observed_col] if not pair.empty else pd.Series(dtype="float64")
        if len(pair) > 1 and pair[observed_col].nunique(dropna=True) > 1 and pair[proxy_col].nunique(dropna=True) > 1:
            pearson = float(np.corrcoef(pair[observed_col].to_numpy(dtype="float64"), pair[proxy_col].to_numpy(dtype="float64"))[0, 1])
        else:
            pearson = np.nan
        rows.append(
            {
                "component": component,
                "proxy": proxy_name,
                "observed_column": observed_col,
                "proxy_column": proxy_col,
                "n_overlap_visits": int(len(pair)),
                "mean_observed": float(pair[observed_col].mean()) if len(pair) else np.nan,
                "mean_proxy": float(pair[proxy_col].mean()) if len(pair) else np.nan,
                "mean_proxy_minus_observed": float(diff.mean()) if len(diff) else np.nan,
                "median_abs_error": float(diff.abs().median()) if len(diff) else np.nan,
                "pearson_correlation": pearson,
                "n_threshold_5_discordant": int(((pair[observed_col] >= 5) != (pair[proxy_col] >= 5)).sum()) if len(pair) else 0,
            }
        )
    return pd.DataFrame(rows)


def build_proxy_sensitivity_summary(longitudinal: pd.DataFrame) -> pd.DataFrame:
    scenarios = [
        ("S0_observed_official", "esspri_total_s0_observed", "pop_status"),
        ("S1_one_proxy", "esspri_total_s1_one_proxy", "pop_status_s1_one_proxy"),
        ("S2_up_to_two_proxies", "esspri_total_s2_up_to_two_proxies", "pop_status_s2_up_to_two_proxies"),
        ("S3_all_available_exploratory_only", "esspri_total_s3_all_available", "pop_status_s3_all_available"),
        ("S4_S1_one_proxy_relaxed", "esspri_total_s1_one_proxy_relaxed", "pop_status_s1_one_proxy_relaxed"),
        ("S4_S2_up_to_two_proxies_relaxed", "esspri_total_s2_up_to_two_proxies_relaxed", "pop_status_s2_up_to_two_proxies_relaxed"),
    ]
    rows = []
    for scenario, total_col, pop_col in scenarios:
        if total_col not in longitudinal.columns:
            continue
        pop = longitudinal[pop_col] if pop_col in longitudinal.columns else pd.Series("Unclassifiable", index=longitudinal.index)
        rows.append(
            {
                "scenario": scenario,
                "n_visits_with_esspri_total": int(longitudinal[total_col].notna().sum()),
                "n_esspri_ge_5": int((longitudinal[total_col] >= 5).sum()),
                "n_pop2": int((pop == "Pop2").sum()),
                "n_pop3": int((pop == "Pop3").sum()),
                "n_unclassifiable": int((pop == "Unclassifiable").sum()),
            }
        )
    return pd.DataFrame(rows)

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
) -> None:
    BLOCKA_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    BLOCKA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    BLOCKA_QC_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet_with_csv(longitudinal, INTERMEDIATE_DIR / "01_visit_level_classification.parquet")
    write_parquet_with_csv(baseline, INTERMEDIATE_DIR / "01_baseline_classification.parquet")
    table1.to_csv(BLOCKA_TABLES_DIR / "01_table1_by_pop.csv", index=False)
    longitudinal[["patient_id", "baseline_date", "event_date", "visit_date_clean", "visit_number", "baseline_pop_status", "baseline_pop_status_display", "time_since_baseline_days", "time_since_baseline_years", "time_years", "essdai_total", "esspri_total", "pop_status", "pop_status_display", "row_date_original", "row_date_min", "row_date_max"]].to_csv(BLOCKA_TABLES_DIR / "01_pop_longitudinal_status.csv", index=False)
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
    proxy_sensitivity.to_csv(BLOCKA_TABLES_DIR / "01_esspri_proxy_sensitivity_summary.csv", index=False)
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
    longitudinal, row_qc, warnings = build_longitudinal_pop_dataset(df)
    baseline = build_baseline_dataset(longitudinal)
    table1, selected_demo = summarize_table1_by_pop(baseline, codebook, warnings)
    baseline_unclassifiable = describe_baseline_unclassifiable(baseline)
    outside = make_pop_swimmer_plot(longitudinal, baseline, BLOCKA_FIGURES_DIR / "02_pop_distribution_plot.pdf", warnings)
    distribution_visit = distribution_by_visit(longitudinal)
    visit_unclassifiable_counts, visit_unclassifiable_rows = describe_visit_unclassifiable(longitudinal)
    proxy_validation = build_proxy_validation(longitudinal)
    proxy_sensitivity = build_proxy_sensitivity_summary(longitudinal)
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
    )
    print(claim)


if __name__ == "__main__":
    main()

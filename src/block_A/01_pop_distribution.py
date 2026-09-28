#!/usr/bin/env python3
"""ITEM 1.4 — Baseline by subpopulation Pop1 / Pop2 / Pop3.

Classifies longitudinal visits using observed ESSDAI/ESSPRI rules only,
summarizes the baseline classifiable cohort by Pop1/Pop2/Pop3, and creates a swimmer plot for
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
from scipy.stats import chi2_contingency, kruskal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402
import config  # noqa: E402
PATIENT_ID_COL = "patient_id"
CODEBOOK_COLUMN = "FORM_NAME__QUESTION_NAME"
DEFAULT_INPUT = Path(common.CLINICAL_VISIT_SPINE_PARQUET)

ESSDAI_TOTAL_CANDIDATES = ["essdai__essdai_total_score"]
ESSPRI_OBSERVED_COMPONENTS = {
    "dryness": "esspri_questionnaire__dryness",
    "fatigue": "esspri_questionnaire__fatigue",
    "pain": "esspri_questionnaire__pain",
}
ESSPRI_COMPONENTS = list(ESSPRI_OBSERVED_COMPONENTS.values())
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
MISSING_STRINGS = config.MISSING_STRINGS


def _common_path(name: str, fallback: Path) -> Path:
    return Path(getattr(common, name, fallback))


OUTPUTS_DIR = _common_path("OUTPUTS_DIR", PROJECT_ROOT / "outputs")
TABLES_DIR = _common_path("TABLES_DIR", OUTPUTS_DIR / "tables")
FIGURES_DIR = _common_path("FIGURES_DIR", OUTPUTS_DIR / "figures")
BLOCKA_TABLES_DIR = _common_path("BLOCKA_TABLES_DIR", TABLES_DIR / "blockA") / "01_pop_distribution"
BLOCKA_FIGURES_DIR = _common_path("BLOCKA_FIGURES_DIR", FIGURES_DIR / "blockA") / "01_pop_distribution"
BLOCKA_QC_DIR = OUTPUTS_DIR / "qc" / "blockA" / "01_pop_distribution"
INTERMEDIATE_DIR = _common_path("INTERMEDIATE_DATA_DIR", PROJECT_ROOT / "data" / "intermediate") / "01_pop_distribution"
DISPLAY = {"Unclassifiable": "Unclassified", "Pop1": "Pop1", "Pop2": "Pop2", "Pop3": "Pop3", "Overall": "Overall"}
DEFAULT_CODEBOOK = Path(getattr(common, "DEFAULT_CODEBOOK", PROJECT_ROOT / "metadata" / "Consolidated_Codebook_all_columns.xlsx"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ITEM 1.4 Pop1/Pop2/Pop3 baseline and longitudinal outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
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
    required = [
        "patient_id", "clinical_episode_id", "clinical_anchor_date",
        "clinical_visit_number", "clinical_baseline_episode_id",
        "clinical_baseline_date", "is_clinical_baseline",
        "time_since_clinical_baseline_days",
        "time_since_clinical_baseline_years", *ESSDAI_TOTAL_CANDIDATES,
    ]
    missing_required = [col for col in required if col not in df.columns]
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


def first_nonmissing(s: pd.Series) -> Any:
    values = s.dropna()
    return values.iloc[0] if len(values) else np.nan


def compute_esspri_from_components(dry: pd.Series, fatigue: pd.Series, pain: pd.Series) -> pd.Series:
    comp_df = pd.concat([dry, fatigue, pain], axis=1)
    comp_df.columns = ["dryness", "fatigue", "pain"]
    return comp_df.mean(axis=1).where(comp_df.notna().all(axis=1), np.nan)

def classify_pop(essdai_total: object, esspri_total: object) -> str:
    essdai_missing = pd.isna(essdai_total)
    esspri_missing = pd.isna(esspri_total)
    if not essdai_missing and float(essdai_total) >= config.ESSDAI_SEVERE:
        return "Pop1"
    if not essdai_missing and float(essdai_total) < config.ESSDAI_SEVERE and not esspri_missing and float(esspri_total) >= config.ESSPRI_THRESHOLD:
        return "Pop2"
    if not essdai_missing and float(essdai_total) < config.ESSDAI_SEVERE and not esspri_missing and float(esspri_total) < config.ESSPRI_THRESHOLD:
        return "Pop3"
    return "Unclassifiable"


def visit_missingness_label(essdai_total: object, esspri_total: object) -> str:
    """Label ESSDAI/ESSPRI availability and threshold side for plot markers."""
    essdai_missing = pd.isna(essdai_total)
    esspri_missing = pd.isna(esspri_total)
    if essdai_missing and esspri_missing:
        return "missing ESSDAI and ESSPRI"
    if essdai_missing:
        return "missing ESSDAI; ESSPRI >=5" if float(esspri_total) >= config.ESSPRI_THRESHOLD else "missing ESSDAI; ESSPRI <5"
    if esspri_missing:
        return "missing ESSPRI; ESSDAI >=5" if float(essdai_total) >= config.ESSDAI_SEVERE else "missing ESSPRI; ESSDAI <5"
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
    numeric_prefixes = ("esspri_",)
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



def build_longitudinal_pop_dataset(df: pd.DataFrame, codebook: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Classify the authoritative clinical spine without rebuilding its episodes."""
    work = df.copy()
    input_episode_ids = work["clinical_episode_id"].copy()
    qc_counts: dict[str, Any] = {"n_input_rows": int(len(work))}
    warnings: list[str] = []

    if work[["patient_id", "clinical_episode_id"]].duplicated().any():
        raise AssertionError("patient_id + clinical_episode_id must be unique")
    if work["patient_id"].map(is_missing).any() or work["clinical_episode_id"].map(is_missing).any():
        raise AssertionError("Authoritative patient and clinical episode IDs cannot be missing")
    work["clinical_anchor_date"] = pd.to_datetime(work["clinical_anchor_date"], errors="coerce")
    work["clinical_baseline_date"] = pd.to_datetime(work["clinical_baseline_date"], errors="coerce")
    if work["clinical_anchor_date"].isna().any():
        raise AssertionError("clinical_anchor_date cannot be missing or invalid")
    work["is_clinical_baseline"] = work["is_clinical_baseline"].fillna(False).astype(bool)
    if work.groupby("patient_id")["is_clinical_baseline"].sum().gt(1).any():
        raise AssertionError("More than one clinical baseline for a patient")

    # Compatibility aliases are direct copies of the authoritative spine fields.
    work["visit_id"] = work["clinical_episode_id"]
    work["visit_date"] = work["clinical_anchor_date"]
    work["visit_date_clean"] = work["clinical_anchor_date"]
    work["event_date"] = work["clinical_anchor_date"]
    work["visit_number"] = work["clinical_visit_number"]
    work["baseline_date"] = work["clinical_baseline_date"]
    work["observed_baseline_date"] = work["clinical_baseline_date"]
    work["time_since_observed_baseline_days"] = work["time_since_clinical_baseline_days"]
    work["time_since_observed_baseline_years"] = work["time_since_clinical_baseline_years"]
    work["time_since_baseline_days"] = work["time_since_clinical_baseline_days"]
    work["time_since_baseline_years"] = work["time_since_clinical_baseline_years"]
    work["time_years"] = work["time_since_clinical_baseline_years"]
    work["row_date_original"] = work["clinical_anchor_date"]
    work["row_date_min"] = work["clinical_anchor_date"]
    work["row_date_max"] = work["clinical_anchor_date"]
    protocol_col = select_first_available(work, PROTOCOL_CANDIDATES, codebook)
    qc_counts["selected_protocol_column"] = protocol_col
    if protocol_col:
        work["protocol"] = work[protocol_col]
    elif "protocol" not in work:
        warnings.append("Variable not found: protocol; protocol-stratified summaries omitted.")

    invalid_rows: list[dict[str, Any]] = qc_counts.setdefault("invalid_value_rows", [])
    # Canonical ESSDAI is parsed directly; no candidate coalescing or pipe resolution.
    work["essdai_total"] = pd.to_numeric(work["essdai__essdai_total_score"], errors="coerce")
    invalid_essdai = work["essdai_total"].notna() & ~work["essdai_total"].between(0, 123)
    for idx in work.index[invalid_essdai]:
        invalid_rows.append({"patient_id": work.at[idx, "patient_id"], "clinical_episode_id": work.at[idx, "clinical_episode_id"], "variable_name": "essdai__essdai_total_score", "invalid_value": work.at[idx, "essdai_total"], "expected_min": 0, "expected_max": 123})
    work.loc[invalid_essdai, "essdai_total"] = np.nan
    qc_counts["n_invalid_essdai"] = int(invalid_essdai.sum())

    invalid_components = pd.Series(False, index=work.index)
    for component, source in ESSPRI_OBSERVED_COMPONENTS.items():
        raw = pd.to_numeric(work[source], errors="coerce") if source in work else pd.Series(np.nan, index=work.index)
        invalid = raw.notna() & ~raw.between(0, 10)
        invalid_components |= invalid
        for idx in work.index[invalid]:
            invalid_rows.append({"patient_id": work.at[idx, "patient_id"], "clinical_episode_id": work.at[idx, "clinical_episode_id"], "variable_name": source, "invalid_value": raw.at[idx], "expected_min": 0, "expected_max": 10})
        work[f"esspri_{component}_observed"] = raw.mask(invalid).astype(float)
    qc_counts["n_invalid_esspri_components"] = int(invalid_components.sum())

    # Official population classification uses observed ESSPRI only.
    # ESSPRI is available only when dryness, fatigue, and pain are all observed
    # in the same authoritative clinical episode. No proxy substitution,
    # cross-instrument reconstruction, or imputation is performed here.
    work["esspri_dryness"] = work["esspri_dryness_observed"]
    work["esspri_fatigue"] = work["esspri_fatigue_observed"]
    work["esspri_pain"] = work["esspri_pain_observed"]
    work["esspri_total"] = compute_esspri_from_components(
        work["esspri_dryness"],
        work["esspri_fatigue"],
        work["esspri_pain"],
    )
    work["esspri_total_observed"] = work["esspri_total"]

    e, ptotal = work["essdai_total"], work["esspri_total_observed"]
    work["pop_status_detailed"] = np.select(
        [e.ge(5), e.lt(5) & ptotal.ge(5), e.lt(5) & ptotal.lt(5), e.lt(5) & ptotal.isna()],
        ["Pop1", "Pop2", "Pop3", "Unclassified_low_ESSDAI_missing_ESSPRI"],
        default="Unclassified_missing_ESSDAI",
    )
    work["pop_status"] = work["pop_status_detailed"].where(work["pop_status_detailed"].isin(["Pop1", "Pop2", "Pop3"]), "Unclassifiable")
    work["pop_status_display"] = work["pop_status"].map(DISPLAY)
    work["pop_missingness_label"] = [visit_missingness_label(x, y) for x, y in zip(e, ptotal)]

    classifiable = work[work["pop_status"].isin(["Pop1", "Pop2", "Pop3"])].sort_values(["patient_id", "clinical_anchor_date", "clinical_episode_id"])
    pop_base = classifiable.drop_duplicates("patient_id")[["patient_id", "clinical_episode_id", "clinical_anchor_date", "pop_status"]].rename(columns={"clinical_episode_id": "pop_baseline_episode_id", "clinical_anchor_date": "pop_baseline_date", "pop_status": "pop_baseline_status"})
    work = work.merge(pop_base, on="patient_id", how="left", validate="many_to_one")
    work["is_pop_baseline"] = work["clinical_episode_id"].eq(work["pop_baseline_episode_id"]) & work["pop_baseline_episode_id"].notna()
    work["time_since_pop_baseline_days"] = (work["clinical_anchor_date"] - work["pop_baseline_date"]).dt.days.astype("Int64")
    work["time_since_pop_baseline_years"] = work["time_since_pop_baseline_days"] / 365.25

    clinical_base = work.loc[work["is_clinical_baseline"], ["patient_id", "pop_status", "pop_status_detailed"]].rename(columns={"pop_status": "clinical_baseline_pop_status", "pop_status_detailed": "clinical_baseline_pop_status_detailed"})
    work = work.merge(clinical_base, on="patient_id", how="left", validate="many_to_one")
    work["baseline_pop_status"] = work["clinical_baseline_pop_status"]
    work["baseline_pop_status_display"] = work["baseline_pop_status"].map(DISPLAY)

    qc_counts.update({"n_unique_patients": int(work["patient_id"].nunique()), "n_rows_missing_essdai": int(e.isna().sum()), "n_rows_missing_esspri": int(ptotal.isna().sum()), "n_rows_negative_time": int(pd.to_numeric(work["time_since_clinical_baseline_days"], errors="coerce").lt(0).sum())})
    if qc_counts["n_rows_negative_time"]:
        warnings.append(f"QC error: {qc_counts['n_rows_negative_time']} clinical episodes have negative time since clinical baseline; rows retained.")
    if len(work) != len(df): raise AssertionError("Input and longitudinal output row counts differ")
    if not work["clinical_episode_id"].reset_index(drop=True).equals(input_episode_ids.reset_index(drop=True)): raise AssertionError("clinical_episode_id changed during processing")
    if not work["visit_id"].eq(work["clinical_episode_id"]).all(): raise AssertionError("visit_id differs from clinical_episode_id")
    if work.groupby("patient_id")["is_pop_baseline"].sum().gt(1).any(): raise AssertionError("More than one Pop baseline for a patient")
    if (work.loc[work["is_pop_baseline"], "pop_status"] == "Unclassifiable").any(): raise AssertionError("Unclassifiable episode selected as Pop baseline")
    if ((work["pop_status"] == "Pop1") & (~e.ge(5))).any(): raise AssertionError("Invalid Pop1 classification")
    if ((work["pop_status"] == "Pop2") & (~e.lt(5) | ~ptotal.ge(5))).any(): raise AssertionError("Invalid Pop2 classification")
    if ((work["pop_status"] == "Pop3") & (~e.lt(5) | ~ptotal.lt(5))).any(): raise AssertionError("Invalid Pop3 classification")
    return normalize_visit_level_dtypes(work), qc_counts, warnings

def build_baseline_dataset(longitudinal: pd.DataFrame) -> pd.DataFrame:
    """Return the upstream-defined clinical baseline (never the Pop baseline)."""
    baseline = longitudinal.loc[longitudinal["is_clinical_baseline"]].copy()
    if baseline["patient_id"].duplicated().any():
        raise AssertionError("More than one clinical baseline for a patient")
    return baseline.reset_index(drop=True)

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
    if not essdai_missing and float(row["essdai_total"]) < config.ESSDAI_SEVERE and esspri_missing:
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
) -> None:
    BLOCKA_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    BLOCKA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    BLOCKA_QC_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet_with_csv(longitudinal, INTERMEDIATE_DIR / "01_visit_level_classification.parquet")
    write_parquet_with_csv(baseline, INTERMEDIATE_DIR / "01_baseline_classification.parquet")
    pop_baseline = longitudinal.loc[longitudinal["is_pop_baseline"]].copy()
    write_parquet_with_csv(pop_baseline, INTERMEDIATE_DIR / "01_pop_baseline_classification.parquet")
    discrepancy_columns = [
        "patient_id", "clinical_baseline_episode_id", "clinical_baseline_date",
        "clinical_baseline_pop_status", "clinical_baseline_pop_status_detailed",
        "pop_baseline_episode_id", "pop_baseline_date", "pop_baseline_status",
    ]
    discrepancy = longitudinal.sort_values(["patient_id", "clinical_anchor_date", "clinical_episode_id"]).drop_duplicates("patient_id")[discrepancy_columns].copy()
    discrepancy["same_episode"] = discrepancy["clinical_baseline_episode_id"].eq(discrepancy["pop_baseline_episode_id"]) & discrepancy["pop_baseline_episode_id"].notna()
    discrepancy["days_between_clinical_and_pop_baseline"] = (pd.to_datetime(discrepancy["pop_baseline_date"]) - pd.to_datetime(discrepancy["clinical_baseline_date"])).dt.days
    discrepancy.to_csv(BLOCKA_QC_DIR / "01_clinical_vs_pop_baseline.csv", index=False)
    detailed_counts = longitudinal["pop_status_detailed"].value_counts()
    clinical_status = discrepancy["clinical_baseline_pop_status"]
    migration_qc = {
        "n_patients": int(longitudinal["patient_id"].nunique()),
        "n_clinical_visits_input": int(qc["n_input_rows"]),
        "n_clinical_visits_output": int(len(longitudinal)),
        "n_unique_clinical_episode_ids": int(longitudinal["clinical_episode_id"].nunique()),
        "n_duplicate_patient_episode_ids": int(longitudinal.duplicated(["patient_id", "clinical_episode_id"]).sum()),
        "n_essdai_available": int(longitudinal["essdai_total"].notna().sum()),
        "n_esspri_dryness_available": int(longitudinal["esspri_dryness_observed"].notna().sum()),
        "n_esspri_fatigue_available": int(longitudinal["esspri_fatigue_observed"].notna().sum()),
        "n_esspri_pain_available": int(longitudinal["esspri_pain_observed"].notna().sum()),
        "n_esspri_complete": int(longitudinal["esspri_total_observed"].notna().sum()),
        "n_pop1": int(detailed_counts.get("Pop1", 0)), "n_pop2": int(detailed_counts.get("Pop2", 0)), "n_pop3": int(detailed_counts.get("Pop3", 0)),
        "n_unclassified_low_essdai_missing_esspri": int(detailed_counts.get("Unclassified_low_ESSDAI_missing_ESSPRI", 0)),
        "n_unclassified_missing_essdai": int(detailed_counts.get("Unclassified_missing_ESSDAI", 0)),
        "n_patients_with_clinical_baseline": int(discrepancy["clinical_baseline_episode_id"].notna().sum()),
        "n_patients_with_classifiable_clinical_baseline": int(clinical_status.isin(["Pop1", "Pop2", "Pop3"]).sum()),
        "n_patients_with_unclassifiable_clinical_baseline": int(clinical_status.eq("Unclassifiable").sum()),
        "n_patients_with_pop_baseline": int(discrepancy["pop_baseline_episode_id"].notna().sum()),
        "n_patients_without_pop_baseline": int(discrepancy["pop_baseline_episode_id"].isna().sum()),
        "n_patients_clinical_baseline_equals_pop_baseline": int(discrepancy["same_episode"].sum()),
        "n_patients_clinical_baseline_differs_from_pop_baseline": int((discrepancy["pop_baseline_episode_id"].notna() & ~discrepancy["same_episode"]).sum()),
        "n_invalid_essdai": int(qc.get("n_invalid_essdai", 0)),
        "n_invalid_esspri_components": int(qc.get("n_invalid_esspri_components", 0)),
    }
    pd.DataFrame([migration_qc]).to_csv(BLOCKA_QC_DIR / "01_pop_migration_qc.csv", index=False)
    with (BLOCKA_QC_DIR / "01_pop_migration_qc.json").open("w", encoding="utf-8") as f:
        json.dump(migration_qc, f, indent=2)
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
    n_total_patients = int(baseline["patient_id"].nunique())
    baseline_counts = baseline["pop_status"].value_counts().reindex(POP_ORDER, fill_value=0)
    n_classifiable = int(baseline_counts[["Pop1", "Pop2", "Pop3"]].sum())
    if int(baseline_counts["Pop1"] + baseline_counts["Pop2"] + baseline_counts["Pop3"]) != n_classifiable:
        raise AssertionError("Pop1 + Pop2 + Pop3 does not equal n_classifiable_baseline")
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
    )
    print(claim)


if __name__ == "__main__":
    main()

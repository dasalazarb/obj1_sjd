#!/usr/bin/env python3
"""ITEM 7.1 — Baseline patient-reported outcome profile.

This script creates a one-row-per-patient baseline dataset using the earliest
valid visit date per patient, then summarizes baseline patient-reported
outcomes (ESSPRI, SF-36, PROFAD, and MDAFS). It intentionally does not perform
longitudinal modeling, population comparisons, imputation, or follow-up change
analyses.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402
from src.derivations.visit_dates import add_parsed_visit_dates

LOG = logging.getLogger("pros_baseline")

PATIENT_ID_COL = "ids__patient_record_number"
VISIT_DATE_COL = "ids__visit_date"
INTERVAL_COL = "ids__interval_name"

from src.derivations.pro_scoring import (
    ESSPRI_COMPONENTS, SF36_ITEMS, SF36_MEASURES, PROFAD_ITEMS, MDAFS_ITEMS,
    MDAFS_ACTIVITY_FLAGS, score_all_pros,
)

MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "unknown", "not available", "missing"}
NOT_VALIDATED = "scoring_algorithm_not_validated"

def path_from_common(name: str, fallback: Path) -> Path:
    """Return a path from common.py when present, otherwise a fallback."""
    return Path(getattr(common, name, fallback))


DATA_INTERMEDIATE_DIR = path_from_common("INTERMEDIATE_DATA_DIR", PROJECT_ROOT / "data" / "intermediate")
TABLES_DIR = path_from_common("BLOCKA_TABLES_DIR", PROJECT_ROOT / "outputs" / "tables" / "blockA")
QC_DIR = path_from_common("BLOCKA_QC_DIR", PROJECT_ROOT / "outputs" / "qc" / "blockA")
LOGS_DIR = path_from_common("LOGS_DIR", PROJECT_ROOT / "outputs" / "logs")
DEFAULT_INPUT = path_from_common(
    "DEFAULT_ANALYTIC_DATASET",
    PROJECT_ROOT / "data" / "raw" / "visits_long_collapsed_by_interval_codebook_corrected.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline PRO profile for Block A ITEM 7.1.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Overwrite previously generated 09_pros_baseline outputs (default).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail before replacing an existing output file.",
    )
    return parser.parse_args()


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "09_pros_baseline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    LOG.info("Run started at %s", datetime.now().isoformat())
    return log_path


def load_input_data(path: Path) -> pd.DataFrame:
    """Load CSV, Excel, or Parquet input data."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    LOG.info("Loaded input %s with shape %s", path, df.shape)
    return df


def is_missing_value(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_STRINGS


def normalize_patient_id(value: Any) -> str | float:
    """Normalize patient identifiers without stripping meaningful leading zeroes."""
    if is_missing_value(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return np.nan
    text = re.sub(r"\.0$", "", text)
    return text if text else np.nan


def nonmissing_distinct(values: Iterable[Any]) -> list[str]:
    vals = []
    for v in values:
        if not is_missing_value(v):
            s = str(v).strip()
            if s not in vals:
                vals.append(s)
    return vals


def collapse_patient_visit_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Collapse to patient_id x visit_date, combining complements and nulling conflicts."""
    before = len(df)
    dedup = df.drop_duplicates()
    n_exact = before - len(dedup)
    counts = dedup.groupby(["patient_id", "visit_date"], dropna=False).size()
    multi_keys = set(counts[counts > 1].index)
    records, conflicts = [], []
    for (pid, vdate), group in dedup.groupby(["patient_id", "visit_date"], dropna=False):
        row = {"patient_id": pid, "visit_date": vdate}
        for col in dedup.columns:
            if col in {"patient_id", "visit_date"}:
                continue
            vals = nonmissing_distinct(group[col])
            if len(vals) == 0:
                row[col] = np.nan
            elif len(vals) == 1:
                row[col] = vals[0]
            else:
                row[col] = np.nan
                conflicts.append({"patient_id": pid, "visit_date": vdate, "variable": col, "observed_values": " | ".join(vals), "n_distinct_values": len(vals), "resolution_status": "conflict_set_missing", "selected_value": np.nan, "resolution_reason": "Multiple distinct non-missing values within patient-date; no validated resolution rule."})
        records.append(row)
    collapsed = pd.DataFrame(records)
    metrics = {
        "n_exact_duplicate_rows": n_exact,
        "n_patient_dates_with_multiple_rows": len(multi_keys),
        "n_patient_dates_combined": len(multi_keys),
        "n_patient_dates_with_conflicts": pd.DataFrame(conflicts)[["patient_id", "visit_date"]].drop_duplicates().shape[0] if conflicts else 0,
        "n_conflicting_variables": pd.DataFrame(conflicts)["variable"].nunique() if conflicts else 0,
    }
    return collapsed, pd.DataFrame(conflicts), metrics


def derive_parent_protocol(interval: Any) -> str:
    """Map interval labels to parent protocol using established convention."""
    if is_missing_value(interval):
        return "Unknown"
    text = str(interval).strip()
    if text == "Natural History Protocol 478 Interval":
        return "15D"
    if "interval" in text.lower() or "protocol" in text.lower() or text:
        return "11D"
    return "Unknown"


def select_global_baseline(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select observed baseline visits defined by the canonical spine."""
    valid = df[df["patient_id"].notna() & df["visit_date"].notna()].copy()
    spine = pd.read_parquet(common.VISIT_SPINE_PARQUET)[["patient_id", "visit_date", "visit_id", "visit_number", "observed_baseline_date", "time_since_observed_baseline_days", "time_since_observed_baseline_years"]]
    valid = valid.merge(spine, on=["patient_id", "visit_date"], how="left", validate="one_to_one")
    if valid["visit_id"].isna().any():
        raise ValueError("PRO visits missing from canonical spine")
    valid["baseline_date"] = valid["observed_baseline_date"]  # compatibility alias
    valid["is_baseline_visit"] = valid["visit_number"].eq(0)
    audit = valid.groupby("patient_id").agg(n_valid_visit_dates=("visit_date", "nunique"), earliest_visit_date=("visit_date", "min"), selected_baseline_date=("baseline_date", "min"), n_rows_on_baseline_date=("is_baseline_visit", "sum")).reset_index()
    audit["baseline_selection_status"] = np.where(audit["n_rows_on_baseline_date"].eq(1), "selected_unique_earliest_visit", "multiple_rows_on_earliest_date_after_collapse")
    baseline = valid[valid["is_baseline_visit"]].copy().sort_values(["patient_id", "visit_date"]).drop_duplicates("patient_id")
    assert baseline["patient_id"].notna().all()
    assert baseline["baseline_date"].notna().all()
    assert baseline["patient_id"].is_unique
    baseline["parent_protocol"] = baseline[INTERVAL_COL].map(derive_parent_protocol) if INTERVAL_COL in baseline else "Unknown"
    return baseline, audit


def inspect_response_codes(df: pd.DataFrame, instrument: str, items: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Collect observed response codes and missing columns for a QC audit."""
    missing = [c for c in items if c not in df.columns]
    observed = {c: sorted(nonmissing_distinct(df[c]))[:100] for c in items if c in df.columns}
    LOG.info("%s missing columns: %s", instrument, missing)
    return observed, missing

def inspect_sf36_response_codes(df: pd.DataFrame) -> tuple[dict[str, list[str]], list[str]]:
    return inspect_response_codes(df, "SF-36", SF36_ITEMS)

def inspect_profad_response_codes(df: pd.DataFrame) -> tuple[dict[str, list[str]], list[str]]:
    return inspect_response_codes(df, "PROFAD", PROFAD_ITEMS)

def inspect_mdafs_response_codes(df: pd.DataFrame) -> tuple[dict[str, list[str]], list[str]]:
    return inspect_response_codes(df, "MDAFS", MDAFS_ITEMS + MDAFS_ACTIVITY_FLAGS)

def summary_stats(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {"n_available": int(s.size), "mean": s.mean() if s.size else np.nan, "sd": s.std(ddof=1) if s.size > 1 else np.nan, "median": s.median() if s.size else np.nan, "q1": s.quantile(0.25) if s.size else np.nan, "q3": s.quantile(0.75) if s.size else np.nan, "min": s.min() if s.size else np.nan, "max": s.max() if s.size else np.nan}


def fmt_num(x: Any, digits: int = 1) -> str:
    return "" if pd.isna(x) else f"{float(x):.{digits}f}"


def summarize_baseline_measure(df: pd.DataFrame, instrument: str, measure: str, label: str, unit: str, status: str, threshold: float | None = None, normative: str | None = None, notes: str = "") -> dict[str, Any]:
    n_total = len(df)
    stats = summary_stats(df[measure]) if measure in df else summary_stats(pd.Series(dtype=float))
    n_avail = stats["n_available"]
    n_missing = n_total - n_avail
    n_above = pct_above = np.nan
    if threshold is not None and n_avail:
        n_above = int((pd.to_numeric(df[measure], errors="coerce") >= threshold).sum())
        pct_above = 100 * n_above / n_avail
    diff_norm = stats["mean"] - 50 if normative and pd.notna(stats["mean"]) else np.nan
    display = "Not calculable" if n_avail == 0 else f"{fmt_num(stats['median'])} (IQR {fmt_num(stats['q1'])}–{fmt_num(stats['q3'])})"
    if normative and n_avail:
        display = f"{fmt_num(stats['mean'])} ± {fmt_num(stats['sd'])}"
    return {"section": "ITEM 7.1", "item": "Baseline patient-reported outcomes", "instrument": instrument, "measure": measure, "measure_label": label, "unit_or_scale": unit, "n_total_baseline": n_total, "n_available": n_avail, "n_missing": n_missing, "pct_available": 100 * n_avail / n_total if n_total else np.nan, "pct_missing": 100 * n_missing / n_total if n_total else np.nan, **{k: stats[k] for k in ["mean", "sd", "median", "q1", "q3", "min", "max"]}, "n_above_clinical_threshold": n_above, "pct_above_clinical_threshold": pct_above, "clinical_threshold": threshold, "normative_reference": normative, "difference_from_normative_mean": diff_norm, "summary_display": display, "scoring_status": status, "notes": notes}


def build_baseline_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize_baseline_measure(df, "ESSPRI", "esspri_total", "ESSPRI total", "0-10", "scored_validated", 5, notes="Total requires all three components; partial mean shown separately as sensitivity.")]
    for c, lab in [("esspri_dryness", "Dryness"), ("esspri_fatigue", "Fatigue"), ("esspri_pain", "Pain"), ("esspri_partial_mean", "Partial mean (sensitivity)")]:
        rows.append(summarize_baseline_measure(df, "ESSPRI", c, lab, "0-10", "scored_validated" if c != "esspri_partial_mean" else "sensitivity_available_cases"))
    for m, lab in SF36_MEASURES.items():
        rows.append(summarize_baseline_measure(df, "SF-36", m, lab, "0-100 or norm-based T-score", "scored_validated", normative="50 ± 10" if m in {"sf36_pcs", "sf36_mcs"} else None, notes="SF-36 v1 item recoding; PCS/MCS use 1998 US norm means, SDs, and factor coefficients."))
    rows.append(summarize_baseline_measure(df, "PROFAD", "profad_total", "PROFAD total", "0-7 mean item score", "scored_validated", notes="Mean of available 0-7 PROFAD items when at least half of the 19 items are answered."))
    rows.append(summarize_baseline_measure(df, "MDAFS", "mdafs_global", "MDAFS global fatigue index", "1-50", "scored_validated", notes="MAF/MDAFS global fatigue index: q1+q2+q3+mean(q4-q14)+(q15*2.5); q16 is not included."))
    return pd.DataFrame(rows)


def build_baseline_availability_table(df: pd.DataFrame) -> pd.DataFrame:
    specs = [("ESSPRI", list(ESSPRI_COMPONENTS.values()), "esspri_total", "scored_validated", ""), ("SF-36", SF36_ITEMS, "sf36_pcs", "scored_validated", ""), ("PROFAD", PROFAD_ITEMS, "profad_total", "scored_validated", ""), ("MDAFS", MDAFS_ITEMS + MDAFS_ACTIVITY_FLAGS, "mdafs_global", "scored_validated", "")]
    rows = []
    for inst, items, score, status, reason in specs:
        present = [c for c in items if c in df.columns]
        any_item = df[present].apply(lambda r: any(not is_missing_value(v) for v in r), axis=1) if present else pd.Series(False, index=df.index)
        complete = df[score].notna() if score in df else pd.Series(False, index=df.index)
        for proto, g in df.groupby("parent_protocol", dropna=False):
            idx = g.index
            dates = g.loc[idx[any_item.loc[idx]], "baseline_date"] if any_item.loc[idx].any() else pd.Series(dtype="datetime64[ns]")
            rows.append({"instrument": inst, "parent_protocol": proto, "n_total_baseline": len(g), "n_with_any_item": int(any_item.loc[idx].sum()), "n_with_complete_score": int(complete.loc[idx].sum()), "pct_with_any_item": 100 * any_item.loc[idx].mean() if len(g) else np.nan, "pct_with_complete_score": 100 * complete.loc[idx].mean() if len(g) else np.nan, "n_missing_all_items": int((~any_item.loc[idx]).sum()), "earliest_observed_date": dates.min(), "latest_observed_date": dates.max(), "scoring_status": status, "reason_not_scored": reason})
    return pd.DataFrame(rows)


def build_manuscript_numbers(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    def row(metric: str, inst: str, measure: str, field: str, placeholder: str) -> dict[str, Any]:
        rec = summary.loc[summary["measure"].eq(measure)].iloc[0]
        est = rec[field] if field in rec and pd.notna(rec[field]) else np.nan
        formatted = "Not calculable" if pd.isna(est) else fmt_num(est)
        note = rec["notes"] if pd.isna(est) else "Available-case baseline descriptive estimate."
        return {"item": "ITEM 7.1", "metric": metric, "instrument": inst, "estimate": est, "sd": rec["sd"], "q1": rec["q1"], "q3": rec["q3"], "n": rec["n_available"], "denominator": rec["n_total_baseline"], "formatted_value": formatted, "claim_placeholder": placeholder, "interpretation_note": note}
    metrics = [
        row("median_esspri_total_bl", "ESSPRI", "esspri_total", "median", "X"), row("q1_esspri_total_bl", "ESSPRI", "esspri_total", "q1", "Y"), row("q3_esspri_total_bl", "ESSPRI", "esspri_total", "q3", "Z"),
        row("median_esspri_dryness_bl", "ESSPRI", "esspri_dryness", "median", "A"), row("median_esspri_fatigue_bl", "ESSPRI", "esspri_fatigue", "median", "component"), row("median_esspri_pain_bl", "ESSPRI", "esspri_pain", "median", "component"),
        row("sf36_pcs_mean_bl", "SF-36", "sf36_pcs", "mean", "B"), row("sf36_pcs_sd_bl", "SF-36", "sf36_pcs", "sd", "SD"), row("sf36_mcs_mean_bl", "SF-36", "sf36_mcs", "mean", "MCS mean"), row("sf36_mcs_sd_bl", "SF-36", "sf36_mcs", "sd", "MCS SD"),
        row("profad_total_median_bl", "PROFAD", "profad_total", "median", "PROFAD median"), row("profad_total_q1_bl", "PROFAD", "profad_total", "q1", "PROFAD Q1"), row("profad_total_q3_bl", "PROFAD", "profad_total", "q3", "PROFAD Q3"),
        row("mdafs_global_median_bl", "MDAFS", "mdafs_global", "median", "MDAFS median"), row("mdafs_global_q1_bl", "MDAFS", "mdafs_global", "q1", "MDAFS Q1"), row("mdafs_global_q3_bl", "MDAFS", "mdafs_global", "q3", "MDAFS Q3"),
    ]
    med = summary[summary["measure"].isin(["esspri_dryness", "esspri_fatigue", "esspri_pain"])]
    valid = med.dropna(subset=["median"])
    if valid.empty:
        highest, est, note = "Not calculable", np.nan, "No ESSPRI component medians calculable."
    else:
        mx = valid["median"].max(); labs = valid.loc[valid["median"].eq(mx), "measure_label"].tolist(); highest, est, note = ", ".join(labs), mx, "Highest median component(s); ties retained."
    metrics.insert(6, {"item": "ITEM 7.1", "metric": "highest_esspri_component", "instrument": "ESSPRI", "estimate": est, "sd": np.nan, "q1": np.nan, "q3": np.nan, "n": int(valid["n_available"].max()) if not valid.empty else 0, "denominator": len(df), "formatted_value": highest, "claim_placeholder": "component with highest median", "interpretation_note": note})
    return pd.DataFrame(metrics)


def build_missingness(df: pd.DataFrame) -> pd.DataFrame:
    specs = {"ESSPRI": ["esspri_dryness", "esspri_fatigue", "esspri_pain", "esspri_total", "esspri_partial_mean"], "SF-36": list(SF36_MEASURES), "PROFAD": ["profad_total"], "MDAFS": ["mdafs_global"]}
    rows = []
    for inst, cols in specs.items():
        for c in cols:
            n_avail = int(df[c].notna().sum()) if c in df else 0
            rows.append({"instrument": inst, "variable": c, "n_total": len(df), "n_available": n_avail, "n_missing": len(df) - n_avail, "pct_missing": 100 * (len(df) - n_avail) / len(df) if len(df) else np.nan})
    return pd.DataFrame(rows)


def build_scoring_status(df: pd.DataFrame, missing_maps: dict[str, list[str]], observed_maps: dict[str, dict[str, list[str]]]) -> pd.DataFrame:
    """Build scoring status QC for all scored instruments."""
    rows = []
    specs = [("ESSPRI", ["esspri_total"], "ESSPRI mean of dryness/fatigue/pain", "3 of 3 valid components"), ("SF-36", list(SF36_MEASURES), "SF-36 v1 0-100 domains; Ware/Kosinski norm-based PCS/MCS", "At least 50% items per domain; all 8 domains for PCS/MCS"), ("PROFAD", ["profad_total"], "PROFAD 0-7 mean item score", "At least 10 of 19 items"), ("MDAFS", ["mdafs_global"], "MAF/MDAFS global fatigue index", "q1-q3 and q15 required; at least 6 of q4-q14 activities")]
    for inst, measures, version, rule in specs:
        for m in measures:
            n_scored = int(df[m].notna().sum()) if m in df else 0
            rows.append({
                "instrument": inst,
                "measure": m,
                "scoring_status": "scored_validated" if n_scored else "insufficient_items",
                "n_scored": n_scored,
                "n_not_scored": len(df) - n_scored,
                "missing_columns": missing_maps.get(inst, []),
                "unexpected_response_codes": json.dumps(observed_maps.get(inst, {}), default=str)[:30000] if inst != "ESSPRI" else "See range violations",
                "algorithm_version": version,
                "completeness_rule": rule,
                "reason_not_scored": "" if n_scored else "No baseline records met completeness/range requirements for this measure.",
            })
    return pd.DataFrame(rows)


def write_file(path: Path, writer, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists and --overwrite was not specified: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer(path)
    LOG.info("Wrote %s", path)


def write_outputs(outputs: dict[str, Any], overwrite: bool) -> None:
    write_file(DATA_INTERMEDIATE_DIR / "09_pros_baseline_patient_level.parquet", lambda p: outputs["patient"].to_parquet(p, index=False), overwrite)
    write_file(DATA_INTERMEDIATE_DIR / "09_pros_baseline_patient_level.csv", lambda p: outputs["patient"].to_csv(p, index=False), overwrite)
    write_file(TABLES_DIR / "09_pros_baseline.csv", lambda p: outputs["summary"].to_csv(p, index=False), overwrite)
    write_file(TABLES_DIR / "09_pros_baseline_availability.csv", lambda p: outputs["availability"].to_csv(p, index=False), overwrite)
    write_file(TABLES_DIR / "09_pros_baseline_manuscript_numbers.csv", lambda p: outputs["manuscript"].to_csv(p, index=False), overwrite)
    write_file(QC_DIR / "09_pros_baseline_qc_summary.json", lambda p: p.write_text(json.dumps(outputs["qc_summary"], indent=2, default=str)), overwrite)
    for key, name in [("date_parsing", "09_pros_baseline_date_parsing.csv"), ("conflicts", "09_pros_baseline_duplicate_visit_conflicts.csv"), ("range_violations", "09_pros_baseline_range_violations.csv"), ("missingness", "09_pros_baseline_missingness.csv"), ("scoring_status", "09_pros_baseline_scoring_status.csv"), ("selection_audit", "09_pros_baseline_selection_audit.csv")]:
        write_file(QC_DIR / name, lambda p, k=key: outputs[k].to_csv(p, index=False), overwrite)


def main() -> None:
    args = parse_args(); log_path = setup_logging()
    df = load_input_data(args.input)
    required = {PATIENT_ID_COL, VISIT_DATE_COL, INTERVAL_COL, *ESSPRI_COMPONENTS.values()}
    LOG.info("Required columns found: %s", sorted(required & set(df.columns)))
    LOG.info("Required columns absent: %s", sorted(required - set(df.columns)))
    if PATIENT_ID_COL not in df or VISIT_DATE_COL not in df:
        raise ValueError(f"Input must include {PATIENT_ID_COL} and {VISIT_DATE_COL}")
    df = add_parsed_visit_dates(df, patient_id_col=PATIENT_ID_COL, visit_date_col=VISIT_DATE_COL)
    parsed = df
    # Assign parsed analytic date columns explicitly instead of concatenating.
    # Some extracts may already contain helper columns such as ``visit_date``;
    # duplicate column names make ``df["visit_date"]`` return a DataFrame and
    # can break boolean filtering in pandas when indexes contain mixed types.
    date_qc = df[["patient_id", "visit_date_raw", "n_date_fragments", "n_valid_date_fragments", "visit_date", "date_parse_status"]].copy()
    valid_date_mask = df["visit_date"].notna()
    valid = df[df["patient_id"].notna() & valid_date_mask].copy()
    collapsed, conflicts, dup_metrics = collapse_patient_visit_duplicates(valid)
    # Score every collapsed patient-visit before selecting baseline so this uses
    # the identical visit-level algorithm as the longitudinal script.
    scored_visits = score_all_pros(collapsed)
    baseline, selection_audit = select_global_baseline(scored_visits)
    sf_obs, sf_missing = inspect_sf36_response_codes(baseline)
    prof_obs, prof_missing = inspect_profad_response_codes(baseline)
    mdafs_obs, mdafs_missing = inspect_mdafs_response_codes(baseline)
    violations = scored_visits.attrs.get("pro_range_violations", [])
    esspri_viol = sf_viol = prof_viol = mdafs_viol = pd.DataFrame(violations)
    summary = build_baseline_summary_table(baseline)
    availability = build_baseline_availability_table(baseline)
    manuscript = build_manuscript_numbers(baseline, summary)
    missingness = build_missingness(baseline)
    scoring = build_scoring_status(baseline, {"SF-36": sf_missing, "PROFAD": prof_missing, "MDAFS": mdafs_missing}, {"SF-36": sf_obs, "PROFAD": prof_obs, "MDAFS": mdafs_obs})
    if conflicts.empty:
        conflicts = pd.DataFrame(columns=["patient_id", "visit_date", "variable", "observed_values", "n_distinct_values", "resolution_status", "selected_value", "resolution_reason"])
    range_violations = pd.DataFrame(violations).rename(columns={"visit_date": "baseline_date"})
    if range_violations.empty:
        range_violations = pd.DataFrame(columns=["patient_id", "baseline_date", "instrument", "variable", "raw_value", "expected_min", "expected_max", "action_taken"])
    patient_cols = ["patient_id", "visit_id", "baseline_date", INTERVAL_COL, "parent_protocol", "visit_date_raw", "date_parse_status", "esspri_dryness", "esspri_fatigue", "esspri_pain", "esspri_n_components", "esspri_total", "esspri_partial_mean", *SF36_MEASURES.keys(), "profad_total", "mdafs_global", "is_baseline_visit", "profad_scoring_status", "mdafs_scoring_status"]
    for c in patient_cols:
        if c not in baseline:
            baseline[c] = np.nan
    qc_summary = {"n_input_rows": len(df), "n_unique_raw_patient_ids": int(df[PATIENT_ID_COL].nunique(dropna=True)), "n_invalid_patient_ids": int(df["patient_id"].isna().sum()), "n_rows_without_valid_date": int(df["visit_date"].isna().sum()), "n_unique_patients_with_valid_date": int(valid["patient_id"].nunique()), "n_duplicate_patient_dates": int(dup_metrics["n_patient_dates_with_multiple_rows"]), "n_baseline_patients": len(baseline), "n_patients_with_any_esspri": int(baseline[["esspri_dryness", "esspri_fatigue", "esspri_pain"]].notna().any(axis=1).sum()), "n_patients_with_complete_esspri": int(baseline["esspri_total"].notna().sum()), "n_patients_with_any_sf36": int(baseline[[c for c in SF36_ITEMS if c in baseline]].notna().any(axis=1).sum()) if any(c in baseline for c in SF36_ITEMS) else 0, "n_patients_with_valid_pcs": int(baseline["sf36_pcs"].notna().sum()), "n_patients_with_any_profad": int(baseline[[c for c in PROFAD_ITEMS if c in baseline]].notna().any(axis=1).sum()) if any(c in baseline for c in PROFAD_ITEMS) else 0, "n_patients_with_valid_profad": int(baseline["profad_total"].notna().sum()), "n_patients_with_any_mdafs": int(baseline[[c for c in MDAFS_ITEMS if c in baseline]].notna().any(axis=1).sum()) if any(c in baseline for c in MDAFS_ITEMS) else 0, "n_patients_with_valid_mdafs": int(baseline["mdafs_global"].notna().sum()), **dup_metrics, "log_path": str(log_path)}
    LOG.info("QC summary: %s", qc_summary)
    LOG.info("Algorithms used: ESSPRI complete-component mean; SF-36 v1 domains and norm-based PCS/MCS; PROFAD mean 0-7 score; MAF/MDAFS global fatigue index.")
    write_outputs({"patient": baseline[patient_cols], "summary": summary, "availability": availability, "manuscript": manuscript, "qc_summary": qc_summary, "date_parsing": date_qc, "conflicts": conflicts, "range_violations": range_violations, "missingness": missingness, "scoring_status": scoring, "selection_audit": selection_audit}, args.overwrite)
    LOG.info("Run completed successfully")


if __name__ == "__main__":
    main()

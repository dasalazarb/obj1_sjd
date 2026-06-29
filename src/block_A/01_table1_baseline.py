#!/usr/bin/env python3
"""ITEM 1.1 — Overall cohort demographics for Sjögren's disease.

Generates a one-row-per-patient baseline table internally, then exports a tidy
Table 1 plus QC files. No identifiers or dates of birth are exported.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Allow execution as `python src/block_A/01_table1_baseline.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402

LOG = logging.getLogger(__name__)

PATIENT_ID_COL = "ids__patient_record_number"
FALLBACK_PATIENT_ID_COL = "ids__subject_number"
VISIT_DATE_COL = "ids__visit_date"
INTERVAL_COL = "ids__interval_name"
SEX_COL = "ids__sex"
RACE_COL = "ids__race"
DOB_COL = "ids__dob"
AGE_AT_VISIT_COL = "ids__age_at_visit"
DX_DATE_COL = "sjogren's_syndrome_history__sjogrens_dx_date"
DX_YES_COL = "sjogren's_syndrome_history__sjogrens_dx"
SYMPTOM_ONSET_CANDIDATES = [
    "sjogren's_syndrome_history__dry_mouth_date_start",
    "sjogren's_syndrome_history__dry_eye_date_start",
    "sjogren's_syndrome_history__dry_othr_date_start",
]
SJOGREN_CLASS_COL = "visit_summary_form__sjogrens_class"

REQUIRED_DATASET_VARS = {
    DX_DATE_COL,
    DX_YES_COL,
    *SYMPTOM_ONSET_CANDIDATES,
    SJOGREN_CLASS_COL,
    PATIENT_ID_COL,
    FALLBACK_PATIENT_ID_COL,
    VISIT_DATE_COL,
    SEX_COL,
    RACE_COL,
    DOB_COL,
    AGE_AT_VISIT_COL,
}

MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "unknown", "unk", "-99"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Block A Table 1 overall cohort demographics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data/salazarda/data/eda_sjd/data_analytic/visits_long_collapsed_by_interval_codebook_corrected.parquet"),
        help="Analytic visit-level CSV/Parquet/XLSX file.",
    )
    parser.add_argument("--outdir", type=Path, default=common.BLOCKA_TABLES_DIR, help="Output directory for Block A tables.")
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=common.PROJECT_ROOT / "data_intermediate" / "block_A",
        help="Directory for patient-level intermediate files used to manually audit Table 1 metrics.",
    )
    parser.add_argument("--eligibility", type=Path, default=common.BLOCKA_TABLES_DIR / "00_analytic_cohort_ids.csv", help="Optional prior eligibility patient ID file.")
    return parser.parse_args()


def is_missing_value(x: object) -> bool:
    if pd.isna(x):
        return True
    return str(x).strip().lower() in MISSING_STRINGS


def first_nonmissing(values: Iterable[object]) -> object:
    for value in values:
        if not is_missing_value(value):
            return value
    return np.nan


def earliest_nonmissing_date(values: Iterable[object]) -> pd.Timestamp:
    """Return the earliest parseable date after ignoring missing diagnosis values."""
    parsed_dates = [parse_partial_date(value) for value in values if not is_missing_value(value)]
    parsed_dates = [date for date in parsed_dates if pd.notna(date)]
    return min(parsed_dates) if parsed_dates else pd.NaT


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def validate_hardcoded_vars(df: pd.DataFrame) -> list[str]:
    missing = sorted(REQUIRED_DATASET_VARS - set(df.columns))
    if PATIENT_ID_COL in missing and FALLBACK_PATIENT_ID_COL in missing:
        raise ValueError(f"No patient identifier column found: tried {PATIENT_ID_COL}, {FALLBACK_PATIENT_ID_COL}")
    return missing


def select_patient_id_col(df: pd.DataFrame) -> str:
    for col in (PATIENT_ID_COL, FALLBACK_PATIENT_ID_COL):
        if col in df.columns and df[col].map(lambda x: not is_missing_value(x)).any():
            return col
    raise ValueError(f"No usable patient identifier found: tried {PATIENT_ID_COL}, {FALLBACK_PATIENT_ID_COL}")


def parse_partial_date(x: object, prefer_midpoint: bool = True) -> pd.Timestamp:
    """Parse full or partial dates; month/year gets day 15 and year-only gets July 1."""
    if is_missing_value(x):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(x, errors="coerce")
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        # Excel serial date; reject tiny category codes by requiring a plausible serial range.
        if 20000 <= float(x) <= 60000:
            return pd.to_datetime(float(x), unit="D", origin="1899-12-30", errors="coerce")
    s = str(x).strip()
    if s.lower() in MISSING_STRINGS:
        return pd.NaT
    if s.isdigit() and len(s) == 4:
        return pd.Timestamp(year=int(s), month=7 if prefer_midpoint else 1, day=1)
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
            month, year = int(parts[0]), int(parts[1])
            if 1 <= month <= 12 and 1800 <= year <= 2200:
                return pd.Timestamp(year=year, month=month, day=15 if prefer_midpoint else 1)
    return pd.to_datetime(s, errors="coerce")


def normalize_sex(x: object) -> str | float:
    if is_missing_value(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"f", "female", "woman", "w"}:
        return "female"
    if s in {"m", "male", "man"}:
        return "male"
    return np.nan


def normalize_sjogren_class(x: object) -> str:
    if is_missing_value(x):
        return "unknown"
    s = str(x).strip().lower()
    mapping = {
        "1": "primary_sjd", "1.0": "primary_sjd", "primary sjogren's syndrome": "primary_sjd",
        "2": "secondary_sjd", "2.0": "secondary_sjd", "secondary sjogren's syndrome": "secondary_sjd",
        "3": "ea_excluded", "3.0": "ea_excluded", "ea excluded sjogren's syndrome": "ea_excluded",
        "4": "incomplete", "4.0": "incomplete", "incomplete sjogren's syndrome": "incomplete",
        "5": "hv", "5.0": "hv", "hv": "hv",
        "6": "rssa", "6.0": "rssa", "rssa": "rssa",
        "7": "rssu", "7.0": "rssu", "rssu": "rssu",
        "8": "other", "8.0": "other", "other": "other",
    }
    return mapping.get(s, "unknown")


def class_is_target_sjd(x: object) -> bool:
    """Return True for Sjögren classification codes included in this cohort."""
    return normalize_sjogren_class(x) in {"primary_sjd", "secondary_sjd", "incomplete"}


def filter_to_target_sjogren_class_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Keep patients whose modal Sjögren class is primary, secondary, or incomplete."""
    patient_id_source = select_patient_id_col(df)
    work = df.copy()
    work["patient_id"] = work[patient_id_source].astype("string")
    target_patient_ids = set()
    work["_visit_date_parsed"] = work[VISIT_DATE_COL].map(parse_partial_date) if VISIT_DATE_COL in work else pd.NaT
    for patient_id, g in work.groupby("patient_id", sort=True, dropna=True):
        g = g.sort_values("_visit_date_parsed", na_position="last")
        _, class_norm = modal_sjogren_class_value(g[SJOGREN_CLASS_COL])
        if class_norm in {"primary_sjd", "secondary_sjd", "incomplete"}:
            target_patient_ids.add(patient_id)
    return work[work["patient_id"].isin(target_patient_ids)].drop(columns=["patient_id", "_visit_date_parsed"]).copy()


def modal_sjogren_class_value(values: Iterable[object]) -> tuple[object, str]:
    """Return the modal non-missing Sjögren classification raw value and normalized label.

    When there is a tie, keep the first tied class observed in the patient's
    visit order so the result is deterministic without inventing a priority.
    """
    counts: dict[str, int] = {}
    first_raw_by_norm: dict[str, object] = {}
    first_order_by_norm: dict[str, int] = {}
    for order, value in enumerate(values):
        if is_missing_value(value):
            continue
        norm = normalize_sjogren_class(value)
        if norm == "unknown":
            continue
        counts[norm] = counts.get(norm, 0) + 1
        if norm not in first_raw_by_norm:
            first_raw_by_norm[norm] = value
            first_order_by_norm[norm] = order
    if not counts:
        return np.nan, "unknown"
    modal_norm = max(counts, key=lambda norm: (counts[norm], -first_order_by_norm[norm]))
    return first_raw_by_norm[modal_norm], modal_norm


def coalesce_same_date(group: pd.DataFrame) -> pd.Series:
    return group.apply(first_nonmissing, axis=0)


def build_baseline_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    patient_id_source = select_patient_id_col(df)
    work = df.copy()
    work["patient_id"] = work[patient_id_source].astype("string")
    work["_visit_date_parsed"] = work[VISIT_DATE_COL].map(parse_partial_date) if VISIT_DATE_COL in work else pd.NaT
    rows = []
    for patient_id, g in work.groupby("patient_id", sort=True, dropna=True):
        g = g.copy().sort_values("_visit_date_parsed", na_position="last")
        valid_dates = g["_visit_date_parsed"].dropna()
        if not valid_dates.empty:
            baseline_date = valid_dates.min()
            baseline_rows = g[g["_visit_date_parsed"] == baseline_date]
        else:
            baseline_date = pd.NaT
            baseline_rows = g.head(1)
        baseline = coalesce_same_date(baseline_rows)

        dx_date = earliest_nonmissing_date(g[DX_DATE_COL]) if DX_DATE_COL in g else pd.NaT
        symptom_dates = [parse_partial_date(first_nonmissing(g[c])) for c in SYMPTOM_ONSET_CANDIDATES if c in g]
        symptom_dates = [d for d in symptom_dates if pd.notna(d)]
        symptom_onset = min(symptom_dates) if symptom_dates else pd.NaT
        dob = parse_partial_date(first_nonmissing(g[DOB_COL])) if DOB_COL in g else pd.NaT
        age_at_visit = pd.to_numeric(pd.Series([first_nonmissing(g[AGE_AT_VISIT_COL])]), errors="coerce").iloc[0] if AGE_AT_VISIT_COL in g else np.nan
        visit_date_for_age = baseline_date

        age_dx = np.nan
        if pd.notna(dx_date) and pd.notna(dob):
            age_dx = (dx_date - dob).days / 365.25
        elif pd.notna(dx_date) and pd.notna(visit_date_for_age) and pd.notna(age_at_visit):
            age_dx = age_at_visit - ((visit_date_for_age - dx_date).days / 365.25)

        dx_delay = np.nan
        if pd.notna(dx_date) and pd.notna(symptom_onset):
            dx_delay = (dx_date - symptom_onset).days / 365.25

        if SJOGREN_CLASS_COL in g:
            class_raw, class_norm = modal_sjogren_class_value(g[SJOGREN_CLASS_COL])
        else:
            class_raw, class_norm = np.nan, "unknown"

        sex_raw = first_nonmissing([baseline.get(SEX_COL, np.nan)])
        if is_missing_value(sex_raw) and SEX_COL in g:
            sex_raw = first_nonmissing(g[SEX_COL])

        race_raw = first_nonmissing([baseline.get(RACE_COL, np.nan)])
        if is_missing_value(race_raw) and RACE_COL in g:
            race_raw = first_nonmissing(g[RACE_COL])

        rows.append({
            "patient_id": patient_id,
            "baseline_visit_date": baseline_date,
            "sex_raw": sex_raw,
            "sex_norm": normalize_sex(sex_raw),
            "race": np.nan if is_missing_value(race_raw) else str(race_raw).strip(),
            "dob": dob,
            "dx_date": dx_date,
            "symptom_onset_date": symptom_onset,
            "age_dx": age_dx,
            "dx_delay_yrs": dx_delay,
            "sjogren_class_raw": class_raw,
            "sjogren_class_norm": class_norm,
            "is_primary_sjd": class_norm == "primary_sjd",
            "is_secondary_sjd": class_norm == "secondary_sjd",
            "is_incomplete_sjd": class_norm == "incomplete",
        })
    baseline_df = pd.DataFrame(rows)
    if baseline_df.empty:
        raise ValueError("Could not create any baseline patient rows")
    return baseline_df


def n_pct(n: int, denom: int, digits: int = 1) -> str:
    pct = np.nan if denom == 0 else round(n / denom * 100, digits)
    return f"{n} ({pct:.{digits}f}%)" if not pd.isna(pct) else f"{n} (NA%)"


def median_iqr(series: pd.Series, digits: int = 1) -> tuple[str, dict[str, float | None]]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        raw = {"median": None, "q1": None, "q3": None}
        return "NA", raw
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    raw = {"median": round(float(med), digits), "q1": round(float(q1), digits), "q3": round(float(q3), digits)}
    return f"{raw['median']:.{digits}f} ({raw['q1']:.{digits}f}–{raw['q3']:.{digits}f})", raw


def apply_eligibility(baseline: pd.DataFrame, eligibility_path: Path) -> tuple[pd.DataFrame, str]:
    if not eligibility_path.exists():
        return baseline, "No prior eligibility file found; used all unique patients."
    elig = read_table(eligibility_path)
    candidate_cols = ["patient_id", PATIENT_ID_COL, FALLBACK_PATIENT_ID_COL]
    id_col = next((c for c in candidate_cols if c in elig.columns), None)
    if id_col is None:
        return baseline, f"Eligibility file {eligibility_path} lacked an ID column; used all unique patients."
    ids = set(elig[id_col].dropna().astype("string"))
    return baseline[baseline["patient_id"].astype("string").isin(ids)].copy(), f"Filtered to {len(ids)} IDs from {eligibility_path}."


def safe_file_stem(path: Path) -> str:
    """Return a compact filesystem-safe stem that identifies the input source."""
    stem = path.stem or "input"
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem.lower()).strip("_")
    return safe[:80] or "input"


def add_metric_audit_flags(baseline: pd.DataFrame) -> pd.DataFrame:
    """Add explicit inclusion/exclusion flags for Table 1 manual metric audits."""
    audit = baseline.copy()
    audit["age_dx_excluded_from_stats"] = audit["age_dx"].notna() & ((audit["age_dx"] < 0) | (audit["age_dx"] < 18) | (audit["age_dx"] > 100))
    audit["age_dx_included_in_stats"] = audit["age_dx"].notna() & ~audit["age_dx_excluded_from_stats"]
    audit["dx_delay_negative"] = audit["dx_delay_yrs"].notna() & (audit["dx_delay_yrs"] < 0)
    audit["dx_delay_gt60"] = audit["dx_delay_yrs"].notna() & (audit["dx_delay_yrs"] > 60)
    audit["dx_delay_excluded_from_stats"] = audit["dx_delay_negative"] | audit["dx_delay_gt60"]
    audit["dx_delay_included_in_stats"] = audit["dx_delay_yrs"].notna() & ~audit["dx_delay_excluded_from_stats"]
    audit["sex_included_in_denominator"] = audit["sex_norm"].notna()
    audit["race_included_in_denominator"] = audit["race"].notna()
    audit["classification_known"] = audit["sjogren_class_norm"] != "unknown"
    return audit


def write_metric_intermediates(
    baseline_pre_eligibility: pd.DataFrame,
    baseline_eligible: pd.DataFrame,
    input_path: Path,
    intermediate_dir: Path,
) -> list[Path]:
    """Save patient-level files used to calculate Table 1 metrics for manual review."""
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    source_stem = safe_file_stem(input_path)
    columns_for_audit = [
        "patient_id",
        "baseline_visit_date",
        "sex_raw",
        "sex_norm",
        "race",
        "dx_date",
        "symptom_onset_date",
        "age_dx",
        "dx_delay_yrs",
        "sjogren_class_raw",
        "sjogren_class_norm",
        "is_primary_sjd",
        "is_secondary_sjd",
        "is_incomplete_sjd",
        "age_dx_excluded_from_stats",
        "age_dx_included_in_stats",
        "dx_delay_negative",
        "dx_delay_gt60",
        "dx_delay_excluded_from_stats",
        "dx_delay_included_in_stats",
        "sex_included_in_denominator",
        "race_included_in_denominator",
        "classification_known",
    ]
    outputs = []
    for label, data in (
        ("baseline_patient_metrics_before_eligibility", baseline_pre_eligibility),
        ("baseline_patient_metrics_after_eligibility", baseline_eligible),
    ):
        audit = add_metric_audit_flags(data)
        audit = audit[[col for col in columns_for_audit if col in audit.columns]].copy()
        audit.insert(0, "source_file", str(input_path))
        path = intermediate_dir / f"01_table1_from_{source_stem}__{label}.csv"
        audit.to_csv(path, index=False)
        outputs.append(path)
    return outputs


def build_outputs(baseline: pd.DataFrame, dataset_missing: list[str], eligibility_detail: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_overall = len(baseline)
    sex_nonmissing = baseline["sex_norm"].notna().sum()
    n_female = int((baseline["sex_norm"] == "female").sum())
    n_male = int((baseline["sex_norm"] == "male").sum())
    n_missing_sex = int(baseline["sex_norm"].isna().sum())
    race_nonmissing = int(baseline["race"].notna().sum())
    n_missing_race = int(baseline["race"].isna().sum())
    race_counts = baseline["race"].dropna().astype(str).value_counts().sort_index()

    age_out = baseline["age_dx"].notna() & ((baseline["age_dx"] < 0) | (baseline["age_dx"] < 18) | (baseline["age_dx"] > 100))
    age_for_stats = baseline.loc[~age_out, "age_dx"]
    age_text, age_raw = median_iqr(age_for_stats)

    delay_negative = baseline["dx_delay_yrs"].notna() & (baseline["dx_delay_yrs"] < 0)
    delay_gt60 = baseline["dx_delay_yrs"].notna() & (baseline["dx_delay_yrs"] > 60)
    delay_excluded = delay_negative | delay_gt60
    delay_text, delay_raw = median_iqr(baseline.loc[~delay_excluded, "dx_delay_yrs"])

    n_primary = int(baseline["is_primary_sjd"].sum())
    n_secondary = int(baseline["is_secondary_sjd"].sum())
    n_incomplete = int(baseline["is_incomplete_sjd"].sum())

    rows = [
        ["Overall cohort", "N patients", n_overall, 0, str(n_overall), json.dumps({"n": n_overall})],
        ["Demographics", "Female, n (%)", n_female, n_missing_sex, n_pct(n_female, sex_nonmissing), json.dumps({"n": n_female, "denom": int(sex_nonmissing), "pct": None if sex_nonmissing == 0 else round(n_female / sex_nonmissing * 100, 1)})],
        ["Demographics", "Male, n (%)", n_male, n_missing_sex, n_pct(n_male, sex_nonmissing), json.dumps({"n": n_male, "denom": int(sex_nonmissing), "pct": None if sex_nonmissing == 0 else round(n_male / sex_nonmissing * 100, 1)})],
        ["Clinical history", "Age at diagnosis, years, median (IQR)", int(age_for_stats.notna().sum()), int(baseline["age_dx"].isna().sum()), age_text, json.dumps(age_raw)],
        ["Clinical history", "Disease duration from symptom onset to diagnosis, years, median (IQR)", int(baseline.loc[~delay_excluded, "dx_delay_yrs"].notna().sum()), int(baseline["dx_delay_yrs"].isna().sum()), delay_text, json.dumps(delay_raw)],
        ["Classification", "Primary SjD, n (%)", n_primary, int((baseline["sjogren_class_norm"] == "unknown").sum()), n_pct(n_primary, n_overall), json.dumps({"n": n_primary, "denom": n_overall, "pct": round(n_primary / n_overall * 100, 1) if n_overall else None})],
        ["Classification", "Secondary SjD, n (%)", n_secondary, int((baseline["sjogren_class_norm"] == "unknown").sum()), n_pct(n_secondary, n_overall), json.dumps({"n": n_secondary, "denom": n_overall, "pct": round(n_secondary / n_overall * 100, 1) if n_overall else None})],
        ["Classification", "Incomplete SjD, n (%)", n_incomplete, int((baseline["sjogren_class_norm"] == "unknown").sum()), n_pct(n_incomplete, n_overall), json.dumps({"n": n_incomplete, "denom": n_overall, "pct": round(n_incomplete / n_overall * 100, 1) if n_overall else None})],
    ]
    rows.extend(
        [
            "Demographics",
            f"Race, {race_level}, n (%)",
            int(race_n),
            n_missing_race,
            n_pct(int(race_n), race_nonmissing),
            json.dumps({
                "n": int(race_n),
                "denom": race_nonmissing,
                "pct": None if race_nonmissing == 0 else round(int(race_n) / race_nonmissing * 100, 1),
            }),
        ]
        for race_level, race_n in race_counts.items()
    )

    table = pd.DataFrame(rows, columns=["section", "variable", "n", "missing", "overall", "raw_value"])

    female_pct = np.nan if sex_nonmissing == 0 else n_female / sex_nonmissing * 100
    class_counts = baseline["sjogren_class_norm"].value_counts(dropna=False).to_dict()
    qc_rows = [
        ["n_unique_patients", n_overall, "pass", eligibility_detail],
        ["n_duplicate_patient_rows_after_baseline", int(baseline["patient_id"].duplicated().sum()), "pass" if not baseline["patient_id"].duplicated().any() else "fail", "Baseline table should be one row per patient."],
        ["sex_missing_n", n_missing_sex, "warning" if n_missing_sex else "pass", "Missing/unknown sex after normalization."],
        ["race_missing_n", n_missing_race, "warning" if n_missing_race else "pass", "Missing/unknown race."],
        ["female_pct_plausibility", None if pd.isna(female_pct) else round(female_pct, 1), "warning" if pd.isna(female_pct) or female_pct < 70 or female_pct > 98 else "pass", "Warning if female percentage is outside 70–98%."],
        ["age_dx_missing_n", int(baseline["age_dx"].isna().sum()), "warning" if baseline["age_dx"].isna().any() else "pass", "Age at diagnosis missing after DOB or age-at-visit fallback."],
        ["age_dx_out_of_range_n", int(age_out.sum()), "warning" if age_out.any() else "pass", "Excluded from median if <18, <0, or >100 years."],
        ["dx_date_missing_n", int(baseline["dx_date"].isna().sum()), "warning" if baseline["dx_date"].isna().any() else "pass", "Missing/unparseable diagnosis date."],
        ["symptom_onset_missing_n", int(baseline["symptom_onset_date"].isna().sum()), "warning" if baseline["symptom_onset_date"].isna().any() else "pass", "No parseable symptom onset candidate date."],
        ["dx_delay_negative_n", int(delay_negative.sum()), "warning" if delay_negative.any() else "pass", "Diagnosis delay <0; onset after diagnosis."],
        ["dx_delay_gt60_n", int(delay_gt60.sum()), "warning" if delay_gt60.any() else "pass", "Diagnosis delay >60 years."],
        ["classification_missing_n", int((baseline["sjogren_class_norm"] == "unknown").sum()), "warning" if (baseline["sjogren_class_norm"] == "unknown").any() else "pass", json.dumps(class_counts, default=str)],
        ["primary_secondary_other_sum_check", sum(class_counts.values()), "pass" if sum(class_counts.values()) == n_overall else "fail", json.dumps(class_counts, default=str)],
        [
            "required_dataset_columns_present",
            not dataset_missing,
            "pass" if not dataset_missing else "warning",
            "Missing dataset variables are treated as missing where possible: " + ", ".join(dataset_missing),
        ],
    ]
    qc = pd.DataFrame(qc_rows, columns=["qc_check", "value", "status", "details"])
    return table, qc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()
    common.ensure_output_dirs()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.input)
    dataset_missing = validate_hardcoded_vars(df)
    LOG.info("Loaded analytic dataset: %s rows, %s columns", df.shape[0], df.shape[1])

    if dataset_missing:
        LOG.warning("Expected variables absent from dataset will be treated as missing where possible: %s", ", ".join(dataset_missing))
        for col in dataset_missing:
            if col not in {PATIENT_ID_COL, FALLBACK_PATIENT_ID_COL}:  # patient ID handled explicitly
                df[col] = np.nan

    df = filter_to_target_sjogren_class_patients(df)
    if df.empty:
        raise ValueError(f"No patients have {SJOGREN_CLASS_COL} equal to 1, 2, or 4")
    LOG.info(
        "Filtered to patients with %s in {1, 2, 4}: %s rows",
        SJOGREN_CLASS_COL,
        df.shape[0],
    )

    baseline = build_baseline_patient_table(df)
    baseline_pre_eligibility = baseline.copy()
    baseline, eligibility_detail = apply_eligibility(baseline, args.eligibility)
    if baseline.empty:
        raise ValueError("No patients remain after baseline construction/eligibility filtering")
    LOG.info("Built baseline patient table: %s unique patients", baseline["patient_id"].nunique())

    intermediate_paths = write_metric_intermediates(baseline_pre_eligibility, baseline, args.input, args.intermediate_dir)
    for intermediate_path in intermediate_paths:
        LOG.info("Wrote %s", intermediate_path.relative_to(common.PROJECT_ROOT) if intermediate_path.is_relative_to(common.PROJECT_ROOT) else intermediate_path)

    table, qc = build_outputs(baseline, dataset_missing, eligibility_detail)
    if dataset_missing:
        qc = pd.concat([qc, pd.DataFrame([{"qc_check": "dataset_columns_missing_but_allowed", "value": len(dataset_missing), "status": "warning", "details": ", ".join(dataset_missing)}])], ignore_index=True)

    csv_path = args.outdir / "01_table1_overall.csv"
    xlsx_path = args.outdir / "01_table1_overall.xlsx"
    qc_path = args.outdir / "01_table1_overall_qc.csv"
    table.to_csv(csv_path, index=False)
    qc.to_csv(qc_path, index=False)
    with pd.ExcelWriter(xlsx_path) as writer:
        table.to_excel(writer, sheet_name="Table1_Overall", index=False)
        qc.to_excel(writer, sheet_name="QC", index=False)

    LOG.info("Wrote %s", csv_path.relative_to(common.PROJECT_ROOT) if csv_path.is_relative_to(common.PROJECT_ROOT) else csv_path)
    LOG.info("Wrote %s", xlsx_path.relative_to(common.PROJECT_ROOT) if xlsx_path.is_relative_to(common.PROJECT_ROOT) else xlsx_path)
    LOG.info("Wrote %s", qc_path.relative_to(common.PROJECT_ROOT) if qc_path.is_relative_to(common.PROJECT_ROOT) else qc_path)
    LOG.info("QC warnings: %s", int((qc["status"] == "warning").sum()))


if __name__ == "__main__":
    main()

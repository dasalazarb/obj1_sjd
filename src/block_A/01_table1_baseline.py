#!/usr/bin/env python3
"""ITEM 1.1 — Overall cohort demographics for Sjögren's disease.

Consumes the authoritative clinical episode spine, then exports a tidy Table 1
plus patient-level provenance and QC files.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
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
import config  # noqa: E402

LOG = logging.getLogger(__name__)
plt = None


def load_pyplot():
    """Load the optional plotting backend without blocking table generation."""
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        return importlib.import_module("matplotlib.pyplot")
    except (ImportError, OSError) as exc:
        LOG.warning(
            "Matplotlib is unavailable; table, QC, and data outputs will be "
            "generated, but follow-up figures will be skipped: %s",
            exc,
        )
        return None

PATIENT_ID_COL = "ids__patient_record_number"
FALLBACK_PATIENT_ID_COL = "ids__subject_number"
CANONICAL_PATIENT_ID_COL = "patient_id"
CLINICAL_EPISODE_COL = "clinical_episode_id"
CLINICAL_ANCHOR_DATE_COL = "clinical_anchor_date"
CLINICAL_VISIT_COL = "clinical_visit"
CLINICAL_BASELINE_EPISODE_COL = "clinical_baseline_episode_id"
CLINICAL_BASELINE_DATE_COL = "clinical_baseline_date"
IS_CLINICAL_BASELINE_COL = "is_clinical_baseline"
SJD_COHORT_COL = "sjd_ever_1_2_4"
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
    CLINICAL_EPISODE_COL,
    CLINICAL_ANCHOR_DATE_COL,
    CLINICAL_VISIT_COL,
    CLINICAL_BASELINE_EPISODE_COL,
    CLINICAL_BASELINE_DATE_COL,
    IS_CLINICAL_BASELINE_COL,
    SJD_COHORT_COL,
    SEX_COL,
    RACE_COL,
    DOB_COL,
    AGE_AT_VISIT_COL,
}
REQUIRED_SPINE_VARS = {
    CLINICAL_EPISODE_COL,
    CLINICAL_ANCHOR_DATE_COL,
    CLINICAL_VISIT_COL,
    CLINICAL_BASELINE_EPISODE_COL,
    CLINICAL_BASELINE_DATE_COL,
    IS_CLINICAL_BASELINE_COL,
    SJD_COHORT_COL,
}

MISSING_STRINGS = config.MISSING_STRINGS

RETENTION_THRESHOLDS = {
    "6 months": 182, "1 year": 365, "2 years": 730,
    "3 years": 1095, "5 years": 1826, "10 years": 3652,
}
RETENTION_COLUMNS = {
    "6 months": "has_followup_6mo", "1 year": "has_followup_1yr",
    "2 years": "has_followup_2yr", "3 years": "has_followup_3yr",
    "5 years": "has_followup_5yr", "10 years": "has_followup_10yr",
}
PROTOCOL_CANDIDATES = [
    "source_protocol", "ids__protocol", "ids__protocol_number",
    "ids__study_protocol", "protocol", "protocol_number", "parent_protocol",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Block A Table 1 overall cohort demographics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=common.SOURCE_EPISODE_SPINE,
        help="Authoritative SjD clinical episode spine (CSV/Parquet/XLSX).",
    )
    parser.add_argument("--outdir", type=Path, default=common.BLOCKA_TABLES_DIR / "01_table1_baseline", help="Output directory for Block A tables.")
    parser.add_argument("--qc-dir", type=Path, default=common.BLOCKA_QC_DIR / "01_table1_baseline", help="Output directory for Block A quality-control artifacts.")
    parser.add_argument("--figures-dir", type=Path, default=common.OUTPUTS_DIR / "figures" / "blockA" / "01_table1_baseline")
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=common.INTERMEDIATE_DATA_DIR / "block_A" / "01_table1_baseline",
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
    missing_spine = sorted(REQUIRED_SPINE_VARS - set(df.columns))
    if missing_spine:
        raise ValueError("Authoritative clinical episode spine columns missing: " + ", ".join(missing_spine))
    missing = sorted(REQUIRED_DATASET_VARS - set(df.columns))
    if not ({CANONICAL_PATIENT_ID_COL, PATIENT_ID_COL, FALLBACK_PATIENT_ID_COL} & set(df.columns)):
        raise ValueError("No patient identifier column found")
    return missing


def select_patient_id_col(df: pd.DataFrame) -> str:
    for col in (CANONICAL_PATIENT_ID_COL, PATIENT_ID_COL, FALLBACK_PATIENT_ID_COL):
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


def build_baseline_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """Take the upstream-designated clinical baseline episode without fallback."""
    patient_id_source = select_patient_id_col(df)
    work = df.copy()
    work["patient_id"] = work[patient_id_source].astype("string")
    baseline_flag = work[IS_CLINICAL_BASELINE_COL].eq(True).fillna(False)  # noqa: E712
    clinical_flag = work[CLINICAL_VISIT_COL].eq(True).fillna(False)  # noqa: E712
    flagged = work.loc[baseline_flag].copy()

    counts = flagged.groupby("patient_id", dropna=True).size()
    multiple = counts[counts > 1]
    episode_mismatch = ~flagged[CLINICAL_EPISODE_COL].eq(flagged[CLINICAL_BASELINE_EPISODE_COL])
    anchor_dates = pd.to_datetime(flagged[CLINICAL_ANCHOR_DATE_COL], errors="coerce")
    baseline_dates = pd.to_datetime(flagged[CLINICAL_BASELINE_DATE_COL], errors="coerce")
    date_mismatch = ~anchor_dates.eq(baseline_dates)
    nonclinical = ~clinical_flag.loc[flagged.index]
    qc = {
        "n_patients_input": int(work["patient_id"].nunique()),
        "n_patients_with_clinical_baseline": int(flagged["patient_id"].nunique()),
        "n_patients_without_clinical_baseline": int(work["patient_id"].nunique() - flagged["patient_id"].nunique()),
        "n_baseline_rows": int(len(flagged)),
        "n_patients_with_multiple_baselines": int(len(multiple)),
        "n_baseline_episode_mismatches": int(episode_mismatch.sum()),
        "n_baseline_date_mismatches": int(date_mismatch.sum()),
        "n_baseline_nonclinical_episodes": int(nonclinical.sum()),
    }
    structural_failures = {key: value for key, value in qc.items() if key in {
        "n_patients_with_multiple_baselines", "n_baseline_episode_mismatches",
        "n_baseline_date_mismatches", "n_baseline_nonclinical_episodes",
    } and value}
    if structural_failures:
        raise ValueError(f"Invalid authoritative clinical baseline structure: {structural_failures}; QC={qc}")

    # Both predicates are intentionally explicit: clinical state comes only from
    # the episode that upstream marked as the clinical baseline.
    selected = work.loc[clinical_flag & baseline_flag].copy()
    rows = []
    for _, baseline in selected.iterrows():
        patient_id = baseline["patient_id"]
        g = work.loc[work["patient_id"].eq(patient_id)]
        dx_date = earliest_nonmissing_date(g[DX_DATE_COL]) if DX_DATE_COL in g else pd.NaT
        symptom_dates = [parse_partial_date(first_nonmissing(g[c])) for c in SYMPTOM_ONSET_CANDIDATES if c in g]
        symptom_dates = [d for d in symptom_dates if pd.notna(d)]
        symptom_onset = min(symptom_dates) if symptom_dates else pd.NaT
        dob = parse_partial_date(first_nonmissing(g[DOB_COL])) if DOB_COL in g else pd.NaT
        age_at_visit = pd.to_numeric(
            pd.Series([baseline.get(AGE_AT_VISIT_COL, np.nan)]), errors="coerce"
        ).iloc[0]
        visit_date_for_age = parse_partial_date(baseline[CLINICAL_ANCHOR_DATE_COL])

        age_dx = np.nan
        if pd.notna(dx_date) and pd.notna(dob):
            age_dx = (dx_date - dob).days / 365.25
        elif pd.notna(dx_date) and pd.notna(visit_date_for_age) and pd.notna(age_at_visit):
            age_dx = age_at_visit - ((visit_date_for_age - dx_date).days / 365.25)

        dx_delay = np.nan
        if pd.notna(dx_date) and pd.notna(symptom_onset):
            dx_delay = (dx_date - symptom_onset).days / 365.25

        class_raw = baseline.get(SJOGREN_CLASS_COL, np.nan)
        class_norm = normalize_sjogren_class(class_raw)

        sex_raw = first_nonmissing([baseline.get(SEX_COL, np.nan)])
        race_raw = first_nonmissing([baseline.get(RACE_COL, np.nan)])

        rows.append({
            "patient_id": patient_id,
            CLINICAL_EPISODE_COL: baseline[CLINICAL_EPISODE_COL],
            CLINICAL_ANCHOR_DATE_COL: baseline[CLINICAL_ANCHOR_DATE_COL],
            CLINICAL_BASELINE_EPISODE_COL: baseline[CLINICAL_BASELINE_EPISODE_COL],
            CLINICAL_BASELINE_DATE_COL: baseline[CLINICAL_BASELINE_DATE_COL],
            IS_CLINICAL_BASELINE_COL: baseline[IS_CLINICAL_BASELINE_COL],
            CLINICAL_VISIT_COL: baseline[CLINICAL_VISIT_COL],
            "baseline_status": "with_clinical_baseline",
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
        baseline_df = pd.DataFrame(columns=["patient_id", CLINICAL_EPISODE_COL, CLINICAL_ANCHOR_DATE_COL,
                                            CLINICAL_BASELINE_EPISODE_COL, CLINICAL_BASELINE_DATE_COL,
                                            IS_CLINICAL_BASELINE_COL, CLINICAL_VISIT_COL])
    baseline_df.attrs["baseline_qc"] = qc
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


def safe_file_stem(path: Path) -> str:
    """Return a compact filesystem-safe stem that identifies the input source."""
    stem = path.stem or "input"
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem.lower()).strip("_")
    return safe[:80] or "input"


def add_metric_audit_flags(baseline: pd.DataFrame) -> pd.DataFrame:
    """Add explicit inclusion/exclusion flags for Table 1 manual metric audits."""
    audit = baseline.copy()
    audit["age_dx_excluded_from_stats"] = False
    audit["age_dx_included_in_stats"] = audit["age_dx"].notna()
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
        CLINICAL_EPISODE_COL,
        CLINICAL_ANCHOR_DATE_COL,
        CLINICAL_BASELINE_EPISODE_COL,
        CLINICAL_BASELINE_DATE_COL,
        IS_CLINICAL_BASELINE_COL,
        CLINICAL_VISIT_COL,
        "baseline_status",
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


def build_patient_audit(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Include baseline provenance and explicitly list patients lacking one."""
    patient_id_source = select_patient_id_col(df)
    input_ids = pd.Index(df[patient_id_source].dropna().astype("string").unique(), name="patient_id")
    baseline_ids = pd.Index(baseline["patient_id"].astype("string"))
    without_ids = input_ids.difference(baseline_ids, sort=False)
    missing = pd.DataFrame({"patient_id": without_ids, "baseline_status": "without_clinical_baseline"})
    audit = pd.concat([baseline.copy(), missing], ignore_index=True, sort=False)
    required = [
        "patient_id", CLINICAL_EPISODE_COL, CLINICAL_ANCHOR_DATE_COL,
        CLINICAL_BASELINE_EPISODE_COL, CLINICAL_BASELINE_DATE_COL,
        IS_CLINICAL_BASELINE_COL, CLINICAL_VISIT_COL, "baseline_status",
        "sex_raw", "age_dx", "dx_date", "symptom_onset_date",
    ]
    return audit[[column for column in required if column in audit.columns]]


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
    age_for_stats = baseline["age_dx"]
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
        ["age_dx_out_of_range_n", int(age_out.sum()), "warning" if age_out.any() else "pass", "Flagged if <18, <0, or >100 years; included in median/IQR."],
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


def resolve_protocol_column(df: pd.DataFrame) -> str | None:
    """Return the first populated protocol field containing a recognized code."""
    for column in PROTOCOL_CANDIDATES:
        if column not in df.columns:
            continue
        populated = df[column].map(lambda value: not is_missing_value(value))
        if populated.any() and df.loc[populated, column].map(normalize_protocol_membership).ne("").any():
            return column
    return None


def normalize_protocol_membership(value: object) -> str:
    """Normalize 11D/15D labels while retaining dual membership."""
    if is_missing_value(value):
        return ""
    compact = str(value).upper().replace("-", "").replace(" ", "")
    memberships = []
    if "11D" in compact:
        memberships.append("11D")
    if "15D" in compact:
        memberships.append("15D")
    return " | ".join(memberships)


def prepare_longitudinal_clinical_episodes(
    df: pd.DataFrame, baseline_patient_ids: Iterable[object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and audit the authoritative clinical-episode unit."""
    work = df.copy()
    source = select_patient_id_col(work)
    work[CANONICAL_PATIENT_ID_COL] = work[source].astype("string")
    clinical = work.loc[work[CLINICAL_VISIT_COL].eq(True).fillna(False)].copy()  # noqa: E712
    if baseline_patient_ids is not None:
        wanted = pd.Index(pd.Series(list(baseline_patient_ids), dtype="string"))
        clinical = clinical.loc[clinical[CANONICAL_PATIENT_ID_COL].isin(wanted)].copy()
    duplicate = clinical.duplicated([CANONICAL_PATIENT_ID_COL, CLINICAL_EPISODE_COL], keep=False)
    if duplicate.any():
        examples = clinical.loc[duplicate, [CANONICAL_PATIENT_ID_COL, CLINICAL_EPISODE_COL]].head().to_dict("records")
        raise ValueError(f"Duplicate patient_id + clinical_episode_id values: {examples}")
    clinical[CLINICAL_ANCHOR_DATE_COL] = pd.to_datetime(clinical[CLINICAL_ANCHOR_DATE_COL], errors="coerce")
    clinical[CLINICAL_BASELINE_DATE_COL] = pd.to_datetime(clinical[CLINICAL_BASELINE_DATE_COL], errors="coerce")
    protocol_col = resolve_protocol_column(clinical)
    clinical["protocol_membership"] = clinical[protocol_col].map(normalize_protocol_membership) if protocol_col else ""
    clinical["has_valid_anchor_date"] = clinical[CLINICAL_ANCHOR_DATE_COL].notna()
    clinical["is_prebaseline"] = (
        clinical["has_valid_anchor_date"] & clinical[CLINICAL_BASELINE_DATE_COL].notna()
        & (clinical[CLINICAL_ANCHOR_DATE_COL] < clinical[CLINICAL_BASELINE_DATE_COL])
    )
    clinical["included_in_primary_followup"] = (
        clinical["has_valid_anchor_date"] & clinical[CLINICAL_BASELINE_DATE_COL].notna()
        & ~clinical["is_prebaseline"]
    )
    audit_columns = [CANONICAL_PATIENT_ID_COL, CLINICAL_EPISODE_COL, CLINICAL_ANCHOR_DATE_COL,
                     CLINICAL_BASELINE_DATE_COL, CLINICAL_VISIT_COL, "protocol_membership",
                     "is_prebaseline", "has_valid_anchor_date", "included_in_primary_followup"]
    return clinical, clinical[audit_columns].copy()


def build_intervisit_gaps(episodes: pd.DataFrame) -> pd.DataFrame:
    columns = [CANONICAL_PATIENT_ID_COL, "previous_clinical_episode_id", CLINICAL_EPISODE_COL,
               "previous_visit_date", "visit_date", "gap_days", "gap_order",
               "gap_zero_days", "gap_negative", "protocol_membership"]
    dated = episodes.loc[episodes["included_in_primary_followup"]].copy()
    dated["_episode_sort"] = dated[CLINICAL_EPISODE_COL].astype("string")
    dated = dated.sort_values([CANONICAL_PATIENT_ID_COL, CLINICAL_ANCHOR_DATE_COL, "_episode_sort"], kind="stable")
    rows = []
    for patient_id, group in dated.groupby(CANONICAL_PATIENT_ID_COL, sort=False):
        previous = None
        for _, row in group.iterrows():
            if previous is not None:
                gap = int((row[CLINICAL_ANCHOR_DATE_COL] - previous[CLINICAL_ANCHOR_DATE_COL]).days)
                rows.append([patient_id, previous[CLINICAL_EPISODE_COL], row[CLINICAL_EPISODE_COL],
                             previous[CLINICAL_ANCHOR_DATE_COL], row[CLINICAL_ANCHOR_DATE_COL], gap,
                             len(rows) + 1, gap == 0, gap < 0, row["protocol_membership"]])
            previous = row
        # Gap order is patient-specific, not a global row number.
        start = len(rows) - max(len(group) - 1, 0)
        for order, target in enumerate(range(start, len(rows)), 1):
            rows[target][6] = order
    return pd.DataFrame(rows, columns=columns)


def build_patient_followup_metrics(
    episodes: pd.DataFrame, patient_ids: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Build exactly one longitudinal metrics row for every requested patient."""
    if patient_ids is None:
        patient_ids = episodes[CANONICAL_PATIENT_ID_COL].dropna().unique()
    ids = pd.Series(list(patient_ids), dtype="string").drop_duplicates()
    gaps = build_intervisit_gaps(episodes)
    rows = []
    for patient_id in ids:
        all_patient = episodes.loc[episodes[CANONICAL_PATIENT_ID_COL].eq(patient_id)]
        # Known pre-baseline episodes remain audit-only.  Undated authoritative
        # episodes cannot be placed on the timeline, but still count as clinical
        # episodes; only dated, non-pre-baseline episodes drive temporal metrics.
        countable = all_patient.loc[~all_patient["is_prebaseline"]]
        dated = all_patient.loc[all_patient["included_in_primary_followup"]]
        dates = dated[CLINICAL_ANCHOR_DATE_COL].dropna()
        baseline_dates = all_patient[CLINICAL_BASELINE_DATE_COL].dropna()
        baseline_date = baseline_dates.iloc[0] if not baseline_dates.empty else pd.NaT
        first_date, last_date = (dates.min(), dates.max()) if not dates.empty else (pd.NaT, pd.NaT)
        followup_days = float((last_date - baseline_date).days) if pd.notna(last_date) and pd.notna(baseline_date) else np.nan
        patient_gaps = gaps.loc[gaps[CANONICAL_PATIENT_ID_COL].eq(patient_id)]
        valid_gaps = patient_gaps.loc[~patient_gaps["gap_negative"], "gap_days"]
        n_episodes = int(len(countable))
        n_dated_episodes = int(len(dates))
        row = {
            CANONICAL_PATIENT_ID_COL: patient_id, CLINICAL_BASELINE_DATE_COL: baseline_date,
            "first_clinical_date": first_date, "last_clinical_date": last_date,
            "n_clinical_episodes": n_episodes, "n_dated_clinical_episodes": n_dated_episodes,
            "followup_days": followup_days, "followup_years": followup_days / 365.25,
            "median_gap_days": valid_gaps.median() if not valid_gaps.empty else np.nan,
            "max_gap_days": valid_gaps.max() if not valid_gaps.empty else np.nan,
            "has_gap_over_180d": bool((valid_gaps > 180).any()),
            "has_gap_over_365d": bool((valid_gaps > 365).any()),
            "has_gap_over_730d": bool((valid_gaps > 730).any()),
            "in_protocol_11d": bool(all_patient["protocol_membership"].str.contains("11D", na=False).any()),
            "in_protocol_15d": bool(all_patient["protocol_membership"].str.contains("15D", na=False).any()),
        }
        for label, days in RETENTION_THRESHOLDS.items():
            row[RETENTION_COLUMNS[label]] = bool(pd.notna(followup_days) and followup_days >= days)
        rows.append(row)
    metrics = pd.DataFrame(rows)
    followup_years = metrics["followup_years"].to_numpy(dtype=float)
    n_clinical_episodes = metrics["n_clinical_episodes"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics["visits_per_followup_year"] = np.where(
            followup_years > 0,
            (n_clinical_episodes - 1) / followup_years,
            np.nan,
        )
    return metrics


def _summary_values(metrics: pd.DataFrame, gaps: pd.DataFrame) -> dict[str, object]:
    n = len(metrics)
    valid_gaps = pd.to_numeric(gaps.loc[~gaps["gap_negative"], "gap_days"], errors="coerce").dropna()
    def count_text(mask: pd.Series) -> str:
        return n_pct(int(mask.sum()), n)
    follow_text, _ = median_iqr(metrics["followup_years"])
    episodes_text, _ = median_iqr(metrics["n_clinical_episodes"])
    rate_text, _ = median_iqr(metrics["visits_per_followup_year"])
    gap_q1_q3 = (
        f"{valid_gaps.quantile(.25):.1f}–{valid_gaps.quantile(.75):.1f}"
        if not valid_gaps.empty else "NA"
    )
    values = {
        "Clinical episodes": int(metrics["n_clinical_episodes"].sum()), "Unique patients": n,
        "Patients with exactly 1 clinical episode": count_text(metrics["n_clinical_episodes"].eq(1)),
        "Patients with >=2 clinical episodes": count_text(metrics["n_clinical_episodes"].ge(2)),
        "Patients with >=3 clinical episodes": count_text(metrics["n_clinical_episodes"].ge(3)),
        "Patients with >=5 clinical episodes": count_text(metrics["n_clinical_episodes"].ge(5)),
        "Patients with >=10 clinical episodes": count_text(metrics["n_clinical_episodes"].ge(10)),
        "Follow-up, median (IQR), years": follow_text,
        "Clinical episodes per patient, median (IQR)": episodes_text,
        "Clinical episodes per patient, mean": round(metrics["n_clinical_episodes"].mean(), 1) if n else np.nan,
        "Maximum clinical episodes per patient": int(metrics["n_clinical_episodes"].max()) if n else np.nan,
        "Median inter-visit gap, days": round(valid_gaps.median(), 1) if not valid_gaps.empty else np.nan,
        "IQR inter-visit gap, days": gap_q1_q3,
        "P90 inter-visit gap, days": round(valid_gaps.quantile(.9), 1) if not valid_gaps.empty else np.nan,
        "Clinical episodes per follow-up year, median (IQR)": rate_text,
    }
    for percentile in (10, 25, 50, 75, 90):
        values[f"Follow-up P{percentile}, years"] = round(metrics["followup_years"].quantile(percentile / 100), 1) if n else np.nan
    values["Maximum follow-up, years"] = round(metrics["followup_years"].max(), 1) if n else np.nan
    for label, column in RETENTION_COLUMNS.items():
        values[f"Follow-up >={label}"] = count_text(metrics[column])
    for days in (180, 365, 730):
        values[f"Patients with at least one gap >{days} days"] = count_text(metrics[f"has_gap_over_{days}d"])
    return values


def build_followup_summary(
    overall_metrics: pd.DataFrame, overall_gaps: pd.DataFrame,
    protocol_metrics: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohorts = dict(protocol_metrics or {})
    cohorts["Overall"] = (overall_metrics, overall_gaps)
    values = {name: _summary_values(*parts) for name, parts in cohorts.items()}
    indicators = list(values["Overall"])
    wide = pd.DataFrame({"Indicator": indicators, **{name: [value[i] for i in indicators] for name, value in values.items()}})
    long = wide.melt(id_vars="Indicator", var_name="Cohort", value_name="Value")
    return wide, long


def build_retention_table(cohort_metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cohort, metrics in cohort_metrics.items():
        denominator = len(metrics)
        for label, days in RETENTION_THRESHOLDS.items():
            retained = int(metrics[RETENTION_COLUMNS[label]].sum())
            rows.append({"cohort": cohort, "time": label, "days": days, "n_retained": retained,
                         "denominator": denominator, "pct_retained": round(100 * retained / denominator, 1) if denominator else np.nan})
    return pd.DataFrame(rows)


def build_followup_qc(episodes: pd.DataFrame, metrics: pd.DataFrame, gaps: pd.DataFrame, protocol_column: str | None) -> pd.DataFrame:
    missing = episodes[CLINICAL_ANCHOR_DATE_COL].isna()
    pre = episodes["is_prebaseline"]
    negative = gaps["gap_negative"] if not gaps.empty else pd.Series(dtype=bool)
    checks = {
        "followup_n_unique_patients": metrics[CANONICAL_PATIENT_ID_COL].nunique(),
        "followup_duplicate_patient_episode_ids": episodes.duplicated([CANONICAL_PATIENT_ID_COL, CLINICAL_EPISODE_COL]).sum(),
        "followup_n_patients_missing_baseline_date": metrics[CLINICAL_BASELINE_DATE_COL].isna().sum(),
        "followup_n_clinical_episodes_missing_anchor_date": missing.sum(),
        "followup_n_patients_with_missing_anchor_date": episodes.loc[missing, CANONICAL_PATIENT_ID_COL].nunique(),
        "followup_n_prebaseline_clinical_episodes": pre.sum(),
        "followup_n_patients_with_prebaseline_clinical_episodes": episodes.loc[pre, CANONICAL_PATIENT_ID_COL].nunique(),
        "followup_n_zero_day_gaps": gaps["gap_zero_days"].sum() if not gaps.empty else 0,
        "followup_n_negative_gaps": negative.sum(),
        "followup_n_patients_with_negative_gaps": gaps.loc[negative, CANONICAL_PATIENT_ID_COL].nunique() if not gaps.empty else 0,
        "followup_protocol_column_found": bool(protocol_column),
        "followup_protocol_column_name": protocol_column if protocol_column else np.nan,
        "followup_n_protocol_11d_patients": metrics["in_protocol_11d"].sum() if protocol_column else np.nan,
        "followup_n_protocol_15d_patients": metrics["in_protocol_15d"].sum() if protocol_column else np.nan,
        "followup_n_dual_protocol_patients": (metrics["in_protocol_11d"] & metrics["in_protocol_15d"]).sum() if protocol_column else np.nan,
    }
    rows = [{"qc_check": key, "value": value, "status": "pass", "details": "Longitudinal descriptive QC."} for key, value in checks.items()]
    warning_keys = {"followup_n_clinical_episodes_missing_anchor_date", "followup_n_prebaseline_clinical_episodes",
                    "followup_n_zero_day_gaps", "followup_n_negative_gaps"}
    for row in rows:
        if row["qc_check"] in warning_keys and row["value"]:
            row["status"] = "warning"
        if row["qc_check"] == "followup_protocol_column_found" and not row["value"]:
            row.update(status="warning", details="protocol_column_missing")
        if row["qc_check"] == "followup_protocol_column_name" and not protocol_column:
            row.update(status="warning", details="no_candidate_contains_recognized_11D_or_15D_values")
    return pd.DataFrame(rows)


def validate_followup_hard_qc(metrics: pd.DataFrame, baseline: pd.DataFrame, retention: pd.DataFrame) -> None:
    if metrics[CANONICAL_PATIENT_ID_COL].duplicated().any():
        raise ValueError("Patient follow-up metrics must contain one row per patient")
    if set(metrics[CANONICAL_PATIENT_ID_COL]) != set(baseline[CANONICAL_PATIENT_ID_COL]):
        raise ValueError("Longitudinal and Table 1 patient universes differ")
    if (metrics["n_dated_clinical_episodes"] > metrics["n_clinical_episodes"]).any():
        raise ValueError("Dated clinical episode counts cannot exceed all clinical episode counts")
    if (metrics["followup_days"].dropna() < 0).any() or (metrics["last_clinical_date"].dropna() < metrics.loc[metrics["last_clinical_date"].notna(), CLINICAL_BASELINE_DATE_COL]).any():
        raise ValueError("Invalid negative follow-up")
    ordered = [RETENTION_COLUMNS[x] for x in RETENTION_THRESHOLDS]
    for earlier, later in zip(ordered, ordered[1:]):
        if (metrics[later] & ~metrics[earlier]).any():
            raise ValueError("Retention thresholds are not monotonic")
    if not retention["pct_retained"].dropna().between(0, 100).all():
        raise ValueError("Retention percentages outside 0–100")


def append_followup_rows_to_table1(table: pd.DataFrame, metrics: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    n = len(metrics)
    valid_gaps = gaps.loc[~gaps["gap_negative"], "gap_days"]
    specifications = []
    for variable, series in [
        ("Clinical episodes per patient, median (IQR)", metrics["n_clinical_episodes"]),
        ("Follow-up, median (IQR), years", metrics["followup_years"]),
        ("Median inter-visit gap, days", valid_gaps),
    ]:
        text, raw = median_iqr(series)
        specifications.append([variable, int(series.notna().sum()), int(series.isna().sum()), text, raw])
    for variable, mask in [
        ("Patients with >=2 clinical episodes", metrics["n_clinical_episodes"].ge(2)),
        ("Patients with >=3 clinical episodes", metrics["n_clinical_episodes"].ge(3)),
        ("Follow-up >=1 year", metrics["has_followup_1yr"]),
        ("Follow-up >=2 years", metrics["has_followup_2yr"]),
        ("Follow-up >=5 years", metrics["has_followup_5yr"]),
    ]:
        count = int(mask.sum()); pct = round(100 * count / n, 1) if n else None
        specifications.append([variable, count, 0, n_pct(count, n), {"n": count, "denominator": n, "pct": pct}])
    added = pd.DataFrame([["Follow-up / longitudinal observation", v, count, missing, text, json.dumps(raw)]
                          for v, count, missing, text, raw in specifications], columns=table.columns)
    return pd.concat([table, added], ignore_index=True)


def plot_followup_distribution(cohorts: dict[str, pd.DataFrame], path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, metrics in cohorts.items():
        plt.hist(metrics["followup_years"].dropna(), bins=20, alpha=.45, label=name)
    plt.xlabel("Follow-up (years)"); plt.ylabel("Patients"); plt.title("Follow-up distribution"); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_retention(retention: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for cohort, data in retention.groupby("cohort", sort=False):
        plt.step(data["days"] / 365.25, data["pct_retained"], where="post", label=cohort)
    plt.xlabel("Follow-up (years)"); plt.ylabel("Patients retained (%)")
    plt.title("Descriptive follow-up retention curve (not Kaplan-Meier)"); plt.ylim(0, 105); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_visits_per_patient(metrics: pd.DataFrame, path: Path) -> None:
    visits = metrics["n_clinical_episodes"]
    categories = ["1 visit", "2 visits", "3–4 visits", "5–9 visits", ">=10 visits"]
    counts = [(visits == 1).sum(), (visits == 2).sum(), visits.between(3, 4).sum(), visits.between(5, 9).sum(), (visits >= 10).sum()]
    plt.figure(figsize=(8, 5)); plt.bar(categories, counts); plt.ylabel("Patients"); plt.title("Clinical episodes per patient"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_followup_vs_visits(metrics: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 5)); plt.scatter(metrics["followup_years"], metrics["n_clinical_episodes"], alpha=.6)
    plt.xlabel("Follow-up (years)"); plt.ylabel("Clinical episodes"); plt.title("Follow-up vs clinical episodes"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_swimmer_followup(episodes: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    ordered = metrics.sort_values(["n_clinical_episodes", "followup_days"], ascending=False)[CANONICAL_PATIENT_ID_COL].tolist()
    positions = {patient_id: index for index, patient_id in enumerate(ordered)}
    dated = episodes.loc[episodes["included_in_primary_followup"]].copy()
    dated["days_from_clinical_baseline"] = (dated[CLINICAL_ANCHOR_DATE_COL] - dated[CLINICAL_BASELINE_DATE_COL]).dt.days
    plt.figure(figsize=(9, max(5, min(14, len(ordered) * .12))))
    for _, row in metrics.iterrows():
        if pd.notna(row["followup_days"]): plt.hlines(positions[row[CANONICAL_PATIENT_ID_COL]], 0, row["followup_days"], color="grey", lw=.7)
    plt.scatter(dated["days_from_clinical_baseline"], dated[CANONICAL_PATIENT_ID_COL].map(positions), s=8)
    plt.xlabel("Days from clinical baseline"); plt.ylabel("Patients"); plt.title("Clinical episode follow-up by patient"); plt.yticks([]); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main() -> None:
    global plt
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()
    common.ensure_output_dirs()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.input)
    dataset_missing = validate_hardcoded_vars(df)
    LOG.info("Loaded analytic dataset: %s rows, %s columns", df.shape[0], df.shape[1])

    if dataset_missing:
        LOG.warning("Expected variables absent from dataset will be treated as missing where possible: %s", ", ".join(dataset_missing))
        for col in dataset_missing:
            if col not in {PATIENT_ID_COL, FALLBACK_PATIENT_ID_COL}:  # patient ID handled explicitly
                df[col] = np.nan

    baseline = build_baseline_patient_table(df)
    baseline_qc = baseline.attrs["baseline_qc"]
    baseline_pre_eligibility = baseline.copy()
    # The SjD spine is already the authoritative cohort.  The legacy eligibility
    # argument remains accepted for CLI compatibility but is deliberately ignored.
    eligibility_detail = "Authoritative SjD spine used without downstream cohort reselection."
    if baseline.empty:
        raise ValueError(f"No patients have an authoritative clinical baseline; QC={baseline_qc}")
    LOG.info("Built baseline patient table: %s unique patients", baseline["patient_id"].nunique())

    intermediate_paths = write_metric_intermediates(baseline_pre_eligibility, baseline, args.input, args.intermediate_dir)
    for intermediate_path in intermediate_paths:
        LOG.info("Wrote %s", intermediate_path.relative_to(common.PROJECT_ROOT) if intermediate_path.is_relative_to(common.PROJECT_ROOT) else intermediate_path)

    table, qc = build_outputs(baseline, dataset_missing, eligibility_detail)
    structural_qc = pd.DataFrame([
        {
            "qc_check": key,
            "value": value,
            "status": "pass" if key not in {
                "n_patients_with_multiple_baselines", "n_baseline_episode_mismatches",
                "n_baseline_date_mismatches", "n_baseline_nonclinical_episodes",
            } or value == 0 else "fail",
            "details": "Authoritative clinical episode spine baseline selection.",
        }
        for key, value in baseline_qc.items()
    ])
    qc = pd.concat([structural_qc, qc], ignore_index=True)
    if dataset_missing:
        qc = pd.concat([qc, pd.DataFrame([{"qc_check": "dataset_columns_missing_but_allowed", "value": len(dataset_missing), "status": "warning", "details": ", ".join(dataset_missing)}])], ignore_index=True)

    # Longitudinal extension: use precisely the Table 1 patient universe and the
    # same authoritative clinical episode spine.
    episodes, followup_audit = prepare_longitudinal_clinical_episodes(df, baseline[CANONICAL_PATIENT_ID_COL])
    gaps = build_intervisit_gaps(episodes)
    metrics = build_patient_followup_metrics(episodes, baseline[CANONICAL_PATIENT_ID_COL])
    protocol_column = resolve_protocol_column(df)
    protocol_parts: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    cohort_metrics = {"Overall": metrics}
    if protocol_column:
        for code, name in (("11D", "Protocol 11D"), ("15D", "Protocol 15D")):
            subset = episodes.loc[episodes["protocol_membership"].str.contains(code, na=False)].copy()
            ids = subset[CANONICAL_PATIENT_ID_COL].drop_duplicates()
            if not ids.empty:
                subset_gaps = build_intervisit_gaps(subset)
                subset_metrics = build_patient_followup_metrics(subset, ids)
                protocol_parts[name] = (subset_metrics, subset_gaps)
                cohort_metrics[name] = subset_metrics
    summary, summary_long = build_followup_summary(metrics, gaps, protocol_parts)
    retention = build_retention_table(cohort_metrics)
    followup_qc = build_followup_qc(episodes, metrics, gaps, protocol_column)
    validate_followup_hard_qc(metrics, baseline, retention)
    table = append_followup_rows_to_table1(table, metrics, gaps)

    csv_path = args.outdir / "01_table1_overall.csv"
    xlsx_path = args.outdir / "01_table1_overall.xlsx"
    qc_path = args.qc_dir / "01_table1_overall_qc.csv"
    audit_path = args.qc_dir / "01_table1_baseline_patient_audit.csv"
    followup_paths = {
        "summary": args.outdir / "01_followup_summary_by_protocol.csv",
        "long": args.outdir / "01_followup_summary_long.csv",
        "metrics": args.outdir / "01_patient_followup_metrics.csv",
        "retention": args.outdir / "01_retention_by_time.csv",
        "gaps": args.outdir / "01_intervisit_gaps.csv",
        "qc": args.qc_dir / "01_followup_qc.csv",
        "excel": args.outdir / "01_followup_summary.xlsx",
    }
    followup_audit_path = args.intermediate_dir / "01_followup_episode_audit.csv"
    table.to_csv(csv_path, index=False)
    qc.to_csv(qc_path, index=False)
    build_patient_audit(df, baseline).to_csv(audit_path, index=False)
    summary.to_csv(followup_paths["summary"], index=False)
    summary_long.to_csv(followup_paths["long"], index=False)
    metrics.to_csv(followup_paths["metrics"], index=False)
    retention.to_csv(followup_paths["retention"], index=False)
    gaps.to_csv(followup_paths["gaps"], index=False)
    followup_qc.to_csv(followup_paths["qc"], index=False)
    followup_audit.to_csv(followup_audit_path, index=False)
    with pd.ExcelWriter(xlsx_path) as writer:
        table.to_excel(writer, sheet_name="Table1_Overall", index=False)
        qc.to_excel(writer, sheet_name="QC", index=False)
    with pd.ExcelWriter(followup_paths["excel"]) as writer:
        summary.to_excel(writer, sheet_name="followup_summary", index=False)
        summary_long.to_excel(writer, sheet_name="followup_summary_long", index=False)
        metrics.to_excel(writer, sheet_name="patient_metrics", index=False)
        retention.to_excel(writer, sheet_name="retention", index=False)
        gaps.to_excel(writer, sheet_name="intervisit_gaps", index=False)
        followup_qc.to_excel(writer, sheet_name="qc", index=False)

    plt = load_pyplot()
    if plt is not None:
        plot_followup_distribution(cohort_metrics, args.figures_dir / "01_followup_distribution.png")
        plot_retention(retention, args.figures_dir / "01_retention_curve.png")
        plot_visits_per_patient(metrics, args.figures_dir / "01_visits_per_patient.png")
        plot_followup_vs_visits(metrics, args.figures_dir / "01_followup_vs_visits.png")
        plot_swimmer_followup(episodes, metrics, args.figures_dir / "01_swimmer_followup.png")

    LOG.info("Wrote %s", csv_path.relative_to(common.PROJECT_ROOT) if csv_path.is_relative_to(common.PROJECT_ROOT) else csv_path)
    LOG.info("Wrote %s", xlsx_path.relative_to(common.PROJECT_ROOT) if xlsx_path.is_relative_to(common.PROJECT_ROOT) else xlsx_path)
    LOG.info("Wrote %s", qc_path.relative_to(common.PROJECT_ROOT) if qc_path.is_relative_to(common.PROJECT_ROOT) else qc_path)
    LOG.info("Wrote %s", audit_path.relative_to(common.PROJECT_ROOT) if audit_path.is_relative_to(common.PROJECT_ROOT) else audit_path)
    LOG.info("QC warnings: %s", int((qc["status"] == "warning").sum()))


if __name__ == "__main__":
    main()

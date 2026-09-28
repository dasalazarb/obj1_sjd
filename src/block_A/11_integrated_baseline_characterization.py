#!/usr/bin/env python3
"""Characterize the official clinical baseline from the integrated dataset.

Step 11 is deliberately a consumer: it does not select measurements or derive
Pop, ESSDAI, ESSPRI, laboratory, serology, overlap, or PRO phenotypes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402

KEYS = ["patient_id", "clinical_episode_id"]
REQUIRED_COLUMNS = [
    "patient_id", "clinical_episode_id", "clinical_anchor_date",
    "clinical_visit_number", "clinical_baseline_episode_id",
    "clinical_baseline_date", "is_clinical_baseline", "clinical_visit",
]
POP_ORDER = ["Pop1", "Pop2", "Pop3", "Unclassifiable"]
DATE_COLUMNS = ["clinical_anchor_date", "clinical_baseline_date", "episode_start_date", "episode_end_date"]
IDENTIFIER_COLUMNS = {
    "patient_id", "ids__patient_record_number", "ids__subject_number",
    "clinical_episode_id", "clinical_baseline_episode_id",
}
SENSITIVITY_TOKENS = ("sensitivity", "_proxy", "_s0_", "_s1_", "_s2_", "relaxed")

# This is the publication contract.  The master baseline remains deliberately
# wide; only these canonical, clinically interpretable fields may reach Table 1.
TABLE1_VARIABLES = {
    "Cohort / demographics": ["age_at_baseline", "sex", "race"],
    "Disease history": ["age_at_diagnosis", "disease_duration", "diagnostic_delay", "sjogren_class_norm"],
    "Disease activity": ["essdai_total", "esspri_total", "esspri_dryness", "esspri_fatigue", "esspri_pain", "pop_status"],
    "Serology": ["anti_ro_ssa__ever_positive_through_episode", "anti_la_ssb__ever_positive_through_episode",
                 "ana__ever_positive_through_episode", "rf__ever_positive_through_episode"],
    "Laboratories": ["complement_c3__value", "complement_c4__value", "igg__value", "esr__value", "crp__value"],
    "Glandular / extended phenotype": ["biopsy_focus_score", "salivary_flow_unstimulated", "ocular_schirmer_min", "ocular_staining_positive", "sicca_any_symptom"],
    "Imaging / SGUS": ["sgus_available", "sgus_summary_score", "sgus_abnormal"],
    "Organ involvement": ["n_extraglandular_domains_active", "overlap_status"],
    "PROs": ["sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global"],
}
CANONICAL_VARIABLES = {variable for variables in TABLE1_VARIABLES.values() for variable in variables}
RANGES = {
    "essdai_total": (0, 123), "esspri_total": (0, 10),
    "esspri_dryness": (0, 10), "esspri_fatigue": (0, 10),
    "esspri_pain": (0, 10), "sf36_pcs": (0, 100), "sf36_mcs": (0, 100),
    "salivary_flow_unstimulated": (0, np.inf), "ocular_schirmer_min": (0, np.inf),
    "biopsy_focus_score": (0, np.inf), "n_extraglandular_domains_active": (0, 11),
}
DOB_COLUMN = "ids__dob"
DX_DATE_COLUMN = "sjogren's_syndrome_history__sjogrens_dx_date"
CLASS_COLUMN = "visit_summary_form__sjogrens_class"
ONSET_COLUMNS = ["sjogren's_syndrome_history__dry_mouth_date_start",
                 "sjogren's_syndrome_history__dry_eye_date_start",
                 "sjogren's_syndrome_history__dry_othr_date_start"]

# Metadata for contract fields and prominent scientific products. All other
# integrated columns are retained and documented dynamically as Integrated.
VARIABLE_SCHEMA = {
    "patient_id": ("Cohort", "Canonical patient identifier", "clinical episode spine"),
    "clinical_episode_id": ("Cohort", "Official baseline episode identifier", "clinical episode spine"),
    "clinical_anchor_date": ("Cohort", "Clinical episode anchor date", "clinical episode spine"),
    "clinical_visit_number": ("Cohort", "Official clinical visit sequence", "clinical episode spine"),
    "clinical_baseline_episode_id": ("Cohort", "Patient clinical baseline episode", "clinical episode spine"),
    "clinical_baseline_date": ("Cohort", "Patient clinical baseline date", "clinical episode spine"),
    "is_clinical_baseline": ("Cohort", "Official baseline indicator", "clinical episode spine"),
    "pop_status": ("Disease activity", "Upstream Pop classification", "01_pop_distribution.py"),
    "essdai_total": ("Disease activity", "Upstream ESSDAI total", "01_pop_distribution.py"),
    "esspri_total": ("Disease activity", "Upstream ESSPRI total", "01_pop_distribution.py"),
    "overlap_status": ("Overlap", "Upstream glandular/extraglandular overlap", "06_overlap_glandular.py"),
    "sf36_pcs": ("PRO", "Upstream SF-36 physical component", "09_pros_longitudinal.py"),
    "sf36_mcs": ("PRO", "Upstream SF-36 mental component", "09_pros_longitudinal.py"),
    "profad_total": ("PRO", "Upstream PROFAD total", "09_pros_longitudinal.py"),
    "mdafs_global": ("PRO", "Upstream MDAFS global", "09_pros_longitudinal.py"),
    "biopsy_focus_score": ("Extended phenotype", "Biopsy focus score", "01_extended_clinical_phenotype.py"),
    "salivary_flow_unstimulated": ("Extended phenotype", "Unstimulated salivary flow", "01_extended_clinical_phenotype.py"),
    "ocular_schirmer_min": ("Extended phenotype", "Minimum Schirmer measurement", "01_extended_clinical_phenotype.py"),
    "ocular_staining_positive": ("Extended phenotype", "Positive ocular staining", "01_extended_clinical_phenotype.py"),
    "sicca_any_symptom": ("Extended phenotype", "Any sicca symptom", "01_extended_clinical_phenotype.py"),
    "sgus_available": ("Extended phenotype", "SGUS availability", "01_extended_clinical_phenotype.py"),
}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def normalize_integrated_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in set(DATE_COLUMNS) & set(frame):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ["patient_id", "clinical_episode_id", "clinical_baseline_episode_id"]:
        if column in frame:
            frame[column] = frame[column].astype("string")
    for column in ["is_clinical_baseline", "clinical_visit"]:
        if column in frame:
            frame[column] = frame[column].astype("boolean")
    return frame


def validate_integrated(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply hard structural QC and return the official patient baseline."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Integrated dataset is missing required columns: {missing}")
    if frame["patient_id"].isna().any():
        raise ValueError("patient_id must not be missing")
    duplicates = frame.loc[frame.duplicated(KEYS, keep=False), KEYS]
    if not duplicates.empty:
        raise ValueError(f"Duplicate patient_id + clinical_episode_id keys: {duplicates.head(10).to_dict('records')}")
    marked = frame["is_clinical_baseline"].fillna(False)
    counts = marked.groupby(frame["patient_id"], dropna=False).sum()
    if counts.gt(1).any():
        raise ValueError(f"More than one clinical baseline: {counts[counts.gt(1)].index.tolist()[:10]}")
    if counts.eq(0).any():
        raise ValueError(f"No clinical baseline: {counts[counts.eq(0)].index.tolist()[:10]}")
    baseline = frame.loc[marked].copy()
    if (baseline["clinical_episode_id"] != baseline["clinical_baseline_episode_id"]).fillna(True).any():
        raise ValueError("Clinical baseline episode identity is inconsistent")
    left = pd.to_datetime(baseline["clinical_anchor_date"], errors="coerce")
    right = pd.to_datetime(baseline["clinical_baseline_date"], errors="coerce")
    if (left.isna().ne(right.isna()) | (left.notna() & left.ne(right))).any():
        raise ValueError("Clinical baseline date is inconsistent")
    if ~baseline["clinical_visit"].fillna(False).all():
        raise ValueError("Clinical baseline is marked as a non-clinical visit")
    if baseline["patient_id"].duplicated().any() or len(baseline) != frame["patient_id"].nunique():
        raise ValueError("Baseline must contain exactly one row per patient")
    return baseline.sort_values("patient_id", kind="stable").reset_index(drop=True)


def parse_partial_date(value: object) -> pd.Timestamp:
    """Parse legacy year/year-month fields deterministically at mid-period."""
    if pd.isna(value) or str(value).strip().lower() in {"", "na", "n/a", "unknown", "nan"}:
        return pd.NaT
    text = str(value).strip()
    if len(text) == 4 and text.isdigit():
        return pd.Timestamp(f"{text}-07-01")
    if len(text) == 7 and text[4] in "-/":
        parsed = pd.to_datetime(text.replace("/", "-") + "-15", errors="coerce")
        return parsed
    return pd.to_datetime(text, errors="coerce")


def normalize_sjogren_class(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if "primary" in text or text in {"1", "p", "psjd", "pss"}:
        return "primary"
    if "secondary" in text or text in {"2", "s", "ssjd", "sss"}:
        return "secondary"
    if "incomplete" in text or text in {"3", "i"}:
        return "incomplete"
    return "unknown"


def add_demographic_history_derivations(baseline: pd.DataFrame) -> pd.DataFrame:
    """Preserve legacy descriptive fields using only the integrated row."""
    baseline = baseline.copy()
    if "age_at_baseline" not in baseline and DOB_COLUMN in baseline:
        dob = baseline[DOB_COLUMN].map(parse_partial_date)
        baseline["age_at_baseline"] = (baseline.clinical_anchor_date - dob).dt.days / 365.25
    if "dx_date" not in baseline and DX_DATE_COLUMN in baseline:
        baseline["dx_date"] = baseline[DX_DATE_COLUMN].map(parse_partial_date)
    if "age_at_diagnosis" not in baseline and "dx_date" in baseline and DOB_COLUMN in baseline:
        dob = baseline[DOB_COLUMN].map(parse_partial_date)
        baseline["age_at_diagnosis"] = (pd.to_datetime(baseline.dx_date) - dob).dt.days / 365.25
    if "disease_duration" not in baseline and "dx_date" in baseline:
        baseline["disease_duration"] = (
            pd.to_datetime(baseline.clinical_anchor_date) - pd.to_datetime(baseline.dx_date)
        ).dt.days / 365.25
    # Compatibility alias for studies that historically named this metric age_dx.
    if "age_dx" not in baseline and "age_at_diagnosis" in baseline:
        baseline["age_dx"] = baseline["age_at_diagnosis"]
    if "symptom_onset_date" not in baseline:
        candidates = [column for column in ONSET_COLUMNS if column in baseline]
        if candidates:
            baseline["symptom_onset_date"] = baseline[candidates].apply(
                lambda row: min((value for value in row.map(parse_partial_date) if pd.notna(value)), default=pd.NaT), axis=1)
    if "diagnostic_delay" not in baseline and {"dx_date", "symptom_onset_date"}.issubset(baseline):
        baseline["diagnostic_delay"] = (pd.to_datetime(baseline.dx_date) - pd.to_datetime(baseline.symptom_onset_date)).dt.days / 365.25
    if "sjogren_class_norm" not in baseline and CLASS_COLUMN in baseline:
        baseline["sjogren_class_norm"] = baseline[CLASS_COLUMN].map(normalize_sjogren_class)
    return baseline


def validity_mask(frame: pd.DataFrame, variable: str) -> pd.Series:
    """Respect an upstream validity/evaluability/has-data flag when present."""
    candidates = [
        f"{variable}_scoring_valid", f"{variable}_valid", f"{variable}_evaluable",
        f"has_{variable}",
    ]
    stems = {"sf36_pcs": "sf36", "sf36_mcs": "sf36", "profad_total": "profad", "mdafs_global": "mdafs", "esspri_total": "esspri"}
    if variable in stems:
        candidates = [f"{stems[variable]}_scoring_valid", *candidates]
    for candidate in candidates:
        if candidate in frame:
            return frame[variable].notna() & frame[candidate].eq(True).fillna(False)
    return frame[variable].notna()


def clinical_block(variable: str) -> str:
    for block, variables in TABLE1_VARIABLES.items():
        if variable in variables:
            return block
    if variable in VARIABLE_SCHEMA:
        return VARIABLE_SCHEMA[variable][0]
    low = variable.lower()
    if low.startswith("eg_") or "domain" in low:
        return "Organ involvement"
    return "Other / Integrated"


def variable_role(baseline: pd.DataFrame, variable: str) -> str:
    """Assign metadata roles without treating raw source fields as laboratory data."""
    if variable in IDENTIFIER_COLUMNS or variable.endswith("_id"):
        return "identifier"
    if pd.api.types.is_datetime64_any_dtype(baseline[variable]) or variable in DATE_COLUMNS or variable.endswith("_date"):
        return "datetime"
    if any(token in variable.lower() for token in SENSITIVITY_TOKENS):
        return "sensitivity"
    if variable in {"age_at_baseline", "age_at_diagnosis", "disease_duration", "diagnostic_delay", "sjogren_class_norm"}:
        return "derived"
    if variable in CANONICAL_VARIABLES:
        return "clinical_measure"
    if any(token in variable.lower() for token in ("flag", "valid", "evaluable", "provenance", "version")):
        return "technical"
    return "raw"


def resolved_table1_manifest(baseline: pd.DataFrame) -> dict[str, list[str]]:
    manifest = {section: [v for v in variables if v in baseline] for section, variables in TABLE1_VARIABLES.items()}
    selected = [v for variables in manifest.values() for v in variables]
    if len(selected) != len(set(selected)):
        raise ValueError("Table 1 manifest contains duplicate concepts")
    forbidden = [v for v in selected if variable_role(baseline, v) in {"identifier", "datetime", "sensitivity", "technical"}]
    if forbidden:
        raise ValueError(f"Forbidden Table 1 variables: {forbidden}")
    return manifest


def variable_availability(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in baseline.columns:
        available = validity_mask(baseline, variable)
        nonmissing = baseline[variable].notna()
        rows.append({"variable": variable, "clinical_block": clinical_block(variable),
                     "n_baseline_total": len(baseline), "n_nonmissing": int(nonmissing.sum()),
                     "pct_nonmissing": 100 * nonmissing.mean(),
                     "n_valid": int(available.sum()), "pct_valid": 100 * available.mean(),
                     # Compatibility aliases for downstream consumers.
                     "n_available_for_analysis": int(available.sum()),
                     "pct_available_for_analysis": 100 * available.mean()})
    return pd.DataFrame(rows)


def missingness_qc(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in baseline:
        n_missing = int(baseline[variable].isna().sum())
        pct = 100 * n_missing / len(baseline) if len(baseline) else np.nan
        category = "<10% missing" if pct < 10 else "10–25%" if pct <= 25 else "25–50%" if pct <= 50 else ">50%"
        rows.append({"variable": variable, "clinical_block": clinical_block(variable), "n_total": len(baseline),
                     "n_missing": n_missing, "pct_missing": pct, "missingness_category": category})
    return pd.DataFrame(rows)


def _numeric_summary(series: pd.Series) -> dict[str, object]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {"n": len(values), "missing": int(series.isna().sum()),
            "mean": values.mean(), "sd": values.std(), "median": values.median(),
            "q1": values.quantile(.25), "q3": values.quantile(.75),
            "minimum": values.min(), "maximum": values.max()}


def continuous_summary(baseline: pd.DataFrame, variables: list[str] | None = None) -> pd.DataFrame:
    rows = []
    excluded = set(KEYS + DATE_COLUMNS)
    for variable in (variables if variables is not None else baseline.columns):
        if (variable in excluded or pd.api.types.is_bool_dtype(baseline[variable])
                or pd.api.types.is_datetime64_any_dtype(baseline[variable])):
            continue
        converted = pd.to_numeric(baseline[variable], errors="coerce")
        if converted.notna().sum() and (pd.api.types.is_numeric_dtype(baseline[variable]) or converted.notna().mean() > .9):
            mask = validity_mask(baseline, variable)
            for group in ["Overall", *POP_ORDER]:
                selected = mask if group == "Overall" else mask & baseline.get("pop_status", pd.Series(pd.NA, index=baseline.index)).eq(group)
                rows.append({"variable": variable, "clinical_block": clinical_block(variable), "group": group,
                             **_numeric_summary(baseline.loc[selected, variable])})
    return pd.DataFrame(rows, columns=["variable", "clinical_block", "group", "n", "missing", "mean", "sd", "median", "q1", "q3", "minimum", "maximum"])


def categorical_summary(baseline: pd.DataFrame, variables: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for variable in (variables if variables is not None else baseline.columns):
        series = baseline[variable]
        if (variable in KEYS + DATE_COLUMNS or pd.api.types.is_datetime64_any_dtype(series)
                or (pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))):
            continue
        # IDs/free text are not useful Table 1 categories.
        if series.nunique(dropna=True) > min(30, max(10, len(series) // 2)):
            continue
        for group in ["Overall", *POP_ORDER]:
            subset = baseline if group == "Overall" else baseline.loc[baseline.get("pop_status", pd.Series(pd.NA, index=baseline.index)).eq(group)]
            denominator = len(subset)
            for level, count in subset[variable].astype("string").fillna("Missing").value_counts(dropna=False, sort=False).items():
                rows.append({"variable": variable, "clinical_block": clinical_block(variable), "level": level,
                             "group": group, "n": int(count), "denominator": denominator,
                             "pct": 100 * count / denominator if denominator else np.nan})
    return pd.DataFrame(rows, columns=["variable", "clinical_block", "level", "group", "n", "denominator", "pct"])


def build_table1(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = resolved_table1_manifest(baseline)
    variables = [v for section in manifest.values() for v in section]
    continuous = continuous_summary(baseline, variables)
    categorical = categorical_summary(baseline, variables)
    rows = [{"Section": "Cohort / demographics", "Variable": "N patients", "N available": len(baseline), "N missing": 0, "Summary": str(len(baseline))}]
    for row in continuous.loc[continuous["group"].eq("Overall")].itertuples():
        value = "NA" if not row.n else f"{row.median:.1f} ({row.q1:.1f}–{row.q3:.1f}); n={row.n}"
        rows.append({"Section": row.clinical_block, "Variable": row.variable, "N available": row.n,
                     "N missing": len(baseline) - row.n, "Summary": value})
    for row in categorical.loc[categorical["group"].eq("Overall")].itertuples():
        rows.append({"Section": row.clinical_block, "Variable": f"{row.variable}: {row.level}",
                     "N available": row.denominator, "N missing": int(baseline[row.variable].isna().sum()),
                     "Summary": f"{row.n} ({row.pct:.1f}%)"})
    overall = pd.DataFrame(rows)
    pop = baseline.get("pop_status", pd.Series(pd.NA, index=baseline.index))
    long_rows = [{"Section": "Cohort / demographics", "Variable": "N patients", "group": group,
                  "value": str(len(baseline) if group == "Overall" else int(pop.eq(group).sum()))}
                 for group in ["Overall", *POP_ORDER]]
    for row in continuous.itertuples():
        long_rows.append({"Section": row.clinical_block, "Variable": row.variable, "group": row.group,
                          "value": "NA" if not row.n else f"{row.median:.1f} ({row.q1:.1f}–{row.q3:.1f}); n={row.n}"})
    for row in categorical.itertuples():
        long_rows.append({"Section": row.clinical_block, "Variable": f"{row.variable}: {row.level}", "group": row.group,
                          "value": f"{row.n} ({row.pct:.1f}%)"})
    by_pop = pd.DataFrame(long_rows).pivot(index=["Section", "Variable"], columns="group", values="value").reset_index()
    by_pop.columns.name = None
    return overall, by_pop.reindex(columns=["Section", "Variable", "Overall", *POP_ORDER])


def regression_comparison(old: pd.DataFrame | None, baseline: pd.DataFrame) -> pd.DataFrame:
    """Compare a retained legacy export without making it an analytic input."""
    columns = ["metric", "old_value", "new_value", "difference", "status", "explanation"]
    if old is None:
        return pd.DataFrame([["legacy_export", np.nan, len(baseline), np.nan, "not_run",
                              "No --legacy-baseline audit export was supplied."]], columns=columns)
    patient_column = "patient_id" if "patient_id" in old else next((c for c in ["ids__patient_record_number", "ids__subject_number"] if c in old), None)
    if patient_column is None:
        raise ValueError("Legacy baseline export has no patient identifier")
    old_ids, new_ids = set(old[patient_column].astype("string").dropna()), set(baseline.patient_id.astype("string").dropna())
    rows = [["n_unique_patients", len(old_ids), len(new_ids), len(new_ids)-len(old_ids), "pass" if old_ids == new_ids else "different",
             "Patient universes are identical." if old_ids == new_ids else "Review patients present in only one upstream product."]]
    for variable in ["sex", "race", "age_at_diagnosis", "age_dx", "diagnostic_delay", "sjogren_class_norm"]:
        if variable not in old or variable not in baseline:
            continue
        left = old[[patient_column, variable]].rename(columns={patient_column: "patient_id", variable: "old"})
        right = baseline[["patient_id", variable]].rename(columns={variable: "new"})
        merged = left.merge(right, on="patient_id", how="inner")
        same = merged.old.astype("string").fillna("<missing>").eq(merged.new.astype("string").fillna("<missing>"))
        rows.append([f"{variable}_matching_patients", int(same.sum()), len(merged), int(len(merged)-same.sum()),
                     "pass" if same.all() else "different", "Exact patient-level comparison of shared values."])
    return pd.DataFrame(rows, columns=columns)


def range_violations(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable, (low, high) in RANGES.items():
        if variable not in baseline:
            continue
        values = pd.to_numeric(baseline[variable], errors="coerce")
        for index in baseline.index[values.notna() & ~values.between(low, high)]:
            rows.append({**baseline.loc[index, KEYS].to_dict(), "variable": variable,
                         "observed_value": values[index], "expected_min": low, "expected_max": high})
    return pd.DataFrame(rows, columns=[*KEYS, "variable", "observed_value", "expected_min", "expected_max"])


def structural_qc(integrated: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "integrated_duplicate_patient_episode": int(integrated.duplicated(KEYS).sum()),
        "patients_without_baseline": int(integrated.patient_id.nunique() - baseline.patient_id.nunique()),
        "baseline_duplicate_patient": int(baseline.patient_id.duplicated().sum()),
        "baseline_episode_id_mismatch": int((baseline.clinical_episode_id != baseline.clinical_baseline_episode_id).sum()),
        "baseline_date_mismatch": int((baseline.clinical_anchor_date != baseline.clinical_baseline_date).sum()),
        "baseline_nonclinical": int((~baseline.clinical_visit.fillna(False)).sum()),
    }
    return pd.DataFrame([{"qc_check": key, "value": value, "status": "pass" if value == 0 else "fail"} for key, value in checks.items()])


def possible_duplicate_concepts(baseline: pd.DataFrame) -> pd.DataFrame:
    """Flag related names for human review; never remove columns automatically."""
    concepts = {
        "ESSDAI": ("essdai",), "ESSPRI": ("esspri",), "sex": ("sex",), "race": ("race",),
        "SSA": ("ssa", "ro52", "ro60"), "SSB": ("ssb", "la_"), "SGUS": ("sgus",),
        "Schirmer": ("schirmer",), "focus score": ("focus_score", "focus score"),
    }
    rows = []
    for concept, stems in concepts.items():
        matches = [column for column in baseline if any(stem in column.lower() for stem in stems)]
        if len(matches) > 1:
            rows.append({"concept": concept, "n_columns": len(matches), "variables": " | ".join(matches),
                         "review_status": "manual_review_required"})
    return pd.DataFrame(rows, columns=["concept", "n_columns", "variables", "review_status"])


def variable_qc(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in baseline:
        series = baseline[variable]
        checks = [("all missing", int(series.isna().all()), "Variable has no observed baseline values."),
                  ("zero variance", int(series.notna().any() and series.nunique(dropna=True) <= 1),
                   "All observed baseline values are identical.")]
        for qc_type, violations, details in checks:
            rows.append({"variable": variable, "qc_type": qc_type, "n_violations": violations,
                         "status": "warning" if violations else "pass", "details": details})
    for variable, count in range_violations(baseline).groupby("variable").size().items():
        rows.append({"variable": variable, "qc_type": "range violation", "n_violations": int(count),
                     "status": "warning", "details": f"Outside documented range {RANGES[variable]}."})
    return pd.DataFrame(rows, columns=["variable", "qc_type", "n_violations", "status", "details"])


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    parser.add_argument("--output", type=Path, default=common.INTEGRATED_BASELINE_PARQUET)
    parser.add_argument("--tables-dir", type=Path, default=common.BLOCKA_TABLES_DIR / "11_integrated_baseline_characterization")
    parser.add_argument("--qc-dir", type=Path, default=common.BLOCKA_QC_DIR / "11_integrated_baseline_characterization")
    parser.add_argument("--legacy-baseline", type=Path, help="Optional frozen pre-refactor export used only for regression audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    integrated = normalize_integrated_dtypes(read_table(args.input))
    baseline = add_demographic_history_derivations(validate_integrated(integrated))
    for directory in [args.output.parent, args.tables_dir, args.qc_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    baseline.to_parquet(args.output, index=False)
    baseline.to_csv(args.output.with_suffix(".csv"), index=False)
    availability = variable_availability(baseline)
    overall, by_pop = build_table1(baseline)
    overall.to_csv(args.tables_dir / "11_table1_overall.csv", index=False)
    by_pop.to_csv(args.tables_dir / "11_table1_by_pop.csv", index=False)
    availability.to_csv(args.tables_dir / "11_baseline_variable_availability.csv", index=False)
    availability.to_csv(args.tables_dir / "11_baseline_full_variable_inventory.csv", index=False)
    with pd.ExcelWriter(args.tables_dir / "11_table1.xlsx") as writer:
        overall.to_excel(writer, sheet_name="Overall", index=False)
        by_pop.to_excel(writer, sheet_name="By Pop", index=False)
        availability.to_excel(writer, sheet_name="Availability", index=False)
    structural_qc(integrated, baseline).to_csv(args.qc_dir / "11_baseline_structural_qc.csv", index=False)
    variable_qc(baseline).to_csv(args.qc_dir / "11_baseline_variable_qc.csv", index=False)
    missingness_qc(baseline).to_csv(args.qc_dir / "11_baseline_missingness_qc.csv", index=False)
    possible_duplicate_concepts(baseline).to_csv(args.qc_dir / "11_baseline_possible_duplicate_concepts.csv", index=False)
    baseline.to_csv(args.qc_dir / "11_baseline_patient_audit.csv", index=False)
    range_violations(baseline).to_csv(args.qc_dir / "11_baseline_range_violations.csv", index=False)
    old = read_table(args.legacy_baseline) if args.legacy_baseline else None
    regression_comparison(old, baseline).to_csv(args.qc_dir / "baseline_refactor_regression_comparison.csv", index=False)
    selected = {v for values in resolved_table1_manifest(baseline).values() for v in values}
    dictionary = [{"variable": column, "clinical_block": clinical_block(column),
                   "role": variable_role(baseline, column), "is_canonical": column in CANONICAL_VARIABLES,
                   "include_in_table1": column in selected,
                   "description": VARIABLE_SCHEMA.get(column, (None, "Integrated longitudinal variable", None))[1],
                   "source_script": VARIABLE_SCHEMA.get(column, (None, None, "10_build_integrated_longitudinal_dataset.py"))[2],
                   "dtype": str(baseline[column].dtype)} for column in baseline]
    pd.DataFrame(dictionary).to_csv(args.tables_dir / "11_variable_dictionary.csv", index=False)
    provenance = {"input_file": str(args.input.resolve()), "input_integration_version": sorted(map(str, baseline.get("integration_version", pd.Series(dtype=str)).dropna().unique())),
                  "run_date": date.today().isoformat(), "git_commit": git_commit(),
                  "baseline_definition": "is_clinical_baseline == True", "unit_of_analysis": "patient x official clinical baseline",
                  "n_patients": len(baseline), "script": Path(__file__).name}
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Wrote {len(baseline):,} official patient baselines to {args.output}")


if __name__ == "__main__":
    main()

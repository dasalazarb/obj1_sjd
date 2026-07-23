#!/usr/bin/env python3
"""Build the canonical one-row-per-patient-date visit spine."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402
from src.derivations.visit_dates import add_parsed_visit_dates, add_visit_timing, collapse_patient_visit_rows

PATIENT = "ids__patient_record_number"
DATE = "ids__visit_date"
OPTIONAL = {"ids__subject_number": "subject_number", "ids__interval_name": "interval_name",
            "ids__protocol": "protocol", "ids__protocol_number": "protocol_number",
            "ids__study_protocol": "study_protocol", "ids__age_at_visit": "age_at_visit",
            "ids__sex": "sex", "ids__race": "race", "ids__ethnicity": "ethnicity",
            "sjogren's_syndrome_history__sjogrens_dx_date": "dx_date_raw"}


def progress(step: int, total: int, message: str) -> None:
    """Print a compact, readable progress message for command-line runs."""
    width = 24
    completed = round(width * step / total)
    bar = "█" * completed + "░" * (width - completed)
    print(f"\n[{bar}] {step}/{total}  {message}", flush=True)


def report_output(path: Path) -> None:
    """Print each artifact as it is written, relative to the project when possible."""
    try:
        display_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = path
    print(f"  ✓ Generated: {display_path}", flush=True)


def first_nonmissing_if_consistent(values: pd.Series) -> tuple[object, bool]:
    clean = values.dropna().astype("string").str.strip()
    clean = clean[~clean.str.lower().isin({"", "na", "n/a", "nan", "none", "unknown", "unk", "missing", "-99"})]
    unique = clean.str.casefold().unique()
    return (clean.iloc[0] if len(unique) == 1 and len(clean) else pd.NA, len(unique) > 1)

def build_spine(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if missing := [c for c in (PATIENT, DATE) if c not in df]:
        raise ValueError(f"Missing required columns: {missing}")
    parsed = add_parsed_visit_dates(df, PATIENT, DATE)
    valid = parsed[parsed.patient_id.notna() & parsed.visit_date.notna()].copy()
    keys, duplicates = collapse_patient_visit_rows(valid)
    metadata_conflicts, rows = [], []
    for (pid, date), group in valid.groupby(["patient_id", "visit_date"], sort=True):
        row = {"patient_id": pid, "visit_date": date}
        for source, target in OPTIONAL.items():
            if source not in group:
                continue
            value, conflict = first_nonmissing_if_consistent(group[source])
            row[target] = value
            if conflict:
                metadata_conflicts.append({"patient_id": pid, "visit_date": date, "variable": target,
                                           "distinct_values": " | ".join(group[source].dropna().astype(str).unique())})
        for col, rule in {"visit_date_raw": lambda x: " | ".join(pd.Series(x).dropna().astype(str).unique()),
                          "visit_date_min": "min", "visit_date_max": "max", "n_date_fragments": "sum",
                          "n_valid_date_fragments": "sum", "date_parse_status": lambda x: " | ".join(pd.Series(x).unique()),
                          "had_pipe_delimited_date": "max"}.items():
            row[col] = rule(group[col]) if callable(rule) else getattr(group[col], rule)()
        rows.append(row)
    spine = pd.DataFrame(rows).merge(duplicates[["patient_id", "visit_date", "n_source_rows"]], on=["patient_id", "visit_date"], how="left")
    spine = add_visit_timing(spine)
    pipe = valid[valid.had_pipe_delimited_date | valid.date_parse_status.eq("multiple_valid_dates_different_days")].copy()
    return spine, duplicates, pd.DataFrame(metadata_conflicts), pipe

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--input", type=Path, default=common.DEFAULT_ANALYTIC_DATASET)
    p.add_argument("--overwrite", dest="overwrite", action="store_true", default=True); p.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = p.parse_args(); common.ensure_output_dirs()
    print("\n╔══════════════════════════════════════════════════════════════╗", flush=True)
    print(f"║  {'Building visit spine':<60}║", flush=True)
    print("╚══════════════════════════════════════════════════════════════╝", flush=True)
    total_steps = 3
    if not args.overwrite and (common.VISIT_SPINE_PARQUET.exists() or common.VISIT_SPINE_CSV.exists()): raise FileExistsError("spine output exists")
    progress(1, total_steps, "Reading source dataset")
    source = pd.read_parquet(args.input) if args.input.suffix == ".parquet" else pd.read_csv(args.input)
    print(f"  • {'Source':<12} {len(source):>6,} rows   | {source[PATIENT].nunique():>5,} patients", flush=True)
    progress(2, total_steps, "Building canonical visit spine")
    spine, duplicates, conflicts, pipe = build_spine(source)
    progress(3, total_steps, "Writing datasets and quality-control reports")
    spine.to_parquet(common.VISIT_SPINE_PARQUET, index=False); report_output(common.VISIT_SPINE_PARQUET)
    spine.to_csv(common.VISIT_SPINE_CSV, index=False); report_output(common.VISIT_SPINE_CSV)
    duplicates_path = common.BLOCKA_QC_DIR / "00_visit_spine_duplicate_patient_dates.csv"
    duplicates.to_csv(duplicates_path, index=False); report_output(duplicates_path)
    conflicts_path = common.BLOCKA_QC_DIR / "00_visit_spine_metadata_conflicts.csv"
    conflicts.to_csv(conflicts_path, index=False); report_output(conflicts_path)
    pipe_path = common.BLOCKA_QC_DIR / "00_visit_spine_pipe_date_audit.csv"
    pipe.to_csv(pipe_path, index=False); report_output(pipe_path)
    reconciliation = pd.DataFrame([{"script_name": name, "n_rows_before": pd.NA, "n_rows_after": pd.NA, "n_patients_before": pd.NA, "n_patients_after": pd.NA, "n_unique_dates_before": pd.NA, "n_unique_dates_after": pd.NA, "n_baseline_patients_before": pd.NA, "n_baseline_patients_after": pd.NA, "n_unmatched_visit_ids": pd.NA, "n_duplicate_visit_ids": pd.NA, "status": "pending_cross_script_run"} for name in ["01_pop_distribution", "06_overlap_glandular", "06_overlap_glandular_followup", "09_pros_baseline"]])
    reconciliation_path = common.BLOCKA_QC_DIR / "00_visit_spine_cross_script_reconciliation.csv"
    reconciliation.to_csv(reconciliation_path, index=False); report_output(reconciliation_path)
    qc_path = common.BLOCKA_QC_DIR / "00_visit_spine_qc.json"
    qc_path.write_text(json.dumps({"n_input_rows": len(source), "n_spine_rows": len(spine), "n_patients": int(spine.patient_id.nunique()), "n_duplicate_patient_dates": int((duplicates.n_source_rows > 1).sum()), "n_pipe_dates": len(pipe)}, indent=2)); report_output(qc_path)
    print(f"\n✓ Complete: {len(spine):,} visits across {spine.patient_id.nunique():,} patients.\n", flush=True)

if __name__ == "__main__": main()

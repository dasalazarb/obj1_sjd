"""Read-only, aggregate-only BTRIS medication coverage audit for the longitudinal cohort.

Compare filtered EDA medication CSVs with the Parquet inputs consumed by Step 08.
Never export patient IDs, drug strings, row IDs or raw records.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd

SOURCES = {
    "11D": ("data_intermediate/BTRIS/11D/Medication288444-1.csv",
            "data_analytic/BTRIS/11D/medication_records.parquet"),
    "15D": ("data_intermediate/BTRIS/15D/Medication288476-1.csv",
            "data_analytic/BTRIS/15D/medication_records.parquet"),
}
RITUX = r"rituximab|rituxan|mabthera|truxima|ruxience|riabni"


def patient_key(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if re.fullmatch(r"\d+\.0+", value):
        value = value.split(".")[0]
    return re.sub(r"^0+", "", re.sub(r"[-/\\\s]", "", value))


def col(frame, name):
    if name not in frame.columns:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[name].astype("string").fillna("").str.strip()


def summarise(frame, eligible):
    needed = ["MRN", "Medication Name", "Drug Ingredient"]
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise ValueError("BTRIS_input_missing_columns:" + ",".join(missing))
    in_cohort = frame.MRN.map(patient_key).isin(eligible)
    subset = frame.loc[in_cohort].copy()
    name = col(subset, "Medication Name")
    ingredient = col(subset, "Drug Ingredient")
    target = (name.str.contains(RITUX, case=False, regex=True, na=False)
              | ingredient.str.contains(RITUX, case=False, regex=True, na=False))
    start = col(subset, "Order Date/Start Date").ne("")
    end = col(subset, "Order End Date").ne("")
    consent = col(subset, "Initial Consent Date").ne("")
    return {
        "all_rows": len(frame),
        "longitudinal_cohort_rows": len(subset),
        "longitudinal_cohort_distinct_patients": int(
            subset.MRN.map(patient_key).nunique()),
        "longitudinal_named_medication_rows": int(
            (name.ne("") | ingredient.ne("")).sum()),
        "longitudinal_rituximab_name_or_ingredient_rows": int(target.sum()),
        "longitudinal_rituximab_distinct_patients": int(
            subset.loc[target, "MRN"].map(patient_key).nunique()),
        "longitudinal_rituximab_with_nonempty_order_start": int(
            (target & start).sum()),
        "longitudinal_rituximab_without_order_start": int(
            (target & ~start).sum()),
        "longitudinal_rituximab_consent_only_no_order_dates": int(
            (target & ~start & ~end & consent).sum()),
        "longitudinal_records_without_order_start": int((~start).sum()),
    }


def audit_source(csv_frame, parquet_frame, eligible):
    previous = summarise(csv_frame, eligible)
    current = summarise(parquet_frame, eligible)
    return {
        "filtered_intermediate_csv": previous,
        "step08_analytic_parquet": current,
        "rituximab_eligible_rows_not_in_step08_input": (
            previous["longitudinal_rituximab_name_or_ingredient_rows"]
            if current["longitudinal_rituximab_name_or_ingredient_rows"] == 0
            else None),
        "interpretation": (
            "This compares source coverage, not exact order-level provenance. "
            "Nonempty start date is not necessarily a valid or administered date."),
    }


def run(eda_root, cohort_csv, audit_root):
    if not cohort_csv.is_file():
        raise FileNotFoundError("longitudinal_cohort_file_missing")
    cohort = pd.read_csv(cohort_csv, dtype="string", usecols=["patient_id"])
    eligible = set(cohort.patient_id.map(patient_key)) - {""}
    if not eligible:
        raise ValueError("empty_longitudinal_cohort")
    report = {
        "audit_version": "08-btris-source-coverage-1.0.0",
        "n_longitudinal_cohort_patients": len(eligible),
        "unit": "source_rows_not_confirmed_administrations",
        "note": "Outputs contain only aggregate counts; no date imputation or source rewriting.",
        "sources": {},
    }
    for source, (csv_rel, parquet_rel) in SOURCES.items():
        csv_file, parquet_file = eda_root / csv_rel, eda_root / parquet_rel
        if not csv_file.is_file() or not parquet_file.is_file():
            raise FileNotFoundError("missing_BTRIS_" + source + "_input")
        csv_data = pd.read_csv(csv_file, dtype="string", low_memory=False)
        parquet_data = pd.read_parquet(parquet_file)
        report["sources"][source] = audit_source(csv_data, parquet_data, eligible)
    root = audit_root / "restricted"
    if not root.is_dir():
        raise FileNotFoundError("Step08_restricted_root_missing")
    destination = root / "btris_source_coverage_audit"
    destination.mkdir(exist_ok=True, mode=0o700)
    os.chmod(destination, 0o700)
    target = destination / "08_btris_source_coverage_audit.json"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".coverage_", suffix=".json",
            dir=destination, delete=False
        ) as output:
            temp_path = Path(output.name)
            json.dump(report, output, indent=2, sort_keys=True)
            output.write("\n")
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return report, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eda-root", type=Path,
        default=Path("/vf/users/salazarda/data/eda_sjd"))
    parser.add_argument(
        "--cohort", type=Path,
        default=Path("data/raw/list_ids_longitudinal.csv"))
    parser.add_argument(
        "--audit-root", type=Path,
        default=Path("outputs/audit/08_treatments_v2"))
    args = parser.parse_args()
    report, target = run(args.eda_root, args.cohort, args.audit_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Protected summary:", target)


if __name__ == "__main__":
    main()

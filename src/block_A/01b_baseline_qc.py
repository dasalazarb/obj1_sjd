#!/usr/bin/env python3
"""Read-only QC of clinical, Table 1, and Pop baseline relationships.

This script consumes existing pipeline artifacts.  It does not rebuild a
baseline, classify a visit, or write to an analytic/intermediate data folder.
All outputs are audit tables written to the Block A QC directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402

LOG = logging.getLogger(__name__)
VALID_POP = {"Pop1", "Pop2", "Pop3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-spine", type=Path, default=common.CLINICAL_VISIT_SPINE_PARQUET)
    parser.add_argument(
        "--table1-audit",
        type=Path,
        default=common.BLOCKA_QC_DIR / "01_table1_baseline" / "01_table1_baseline_patient_audit.csv",
    )
    parser.add_argument("--pop-longitudinal", type=Path, default=common.POP_LONGITUDINAL_PARQUET)
    parser.add_argument("--outdir", type=Path, default=common.BLOCKA_QC_DIR / "01b_baseline_qc")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required existing pipeline artifact not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def require_columns(df: pd.DataFrame, columns: set[str], artifact: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{artifact} is missing required columns: {', '.join(missing)}")


def as_true(series: pd.Series) -> pd.Series:
    """Interpret only explicit boolean/boolean-like true values as true."""
    return series.eq(True) | series.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})  # noqa: E712


def missing_as_false(series: pd.Series) -> pd.Series:
    """Return a non-nullable bool Series without object downcasting warnings."""
    return series.astype("boolean").fillna(False).astype(bool)


def ids_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    left = left.astype("string").str.strip()
    right = right.astype("string").str.strip()
    return left.notna() & right.notna() & left.eq(right)


def first_patient_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.sort_values(["patient_id"], kind="stable").drop_duplicates("patient_id")[columns].copy()


def build_audit(
    clinical: pd.DataFrame, table1: pd.DataFrame, pop: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    require_columns(
        clinical,
        {"patient_id", "clinical_episode_id", "clinical_anchor_date", "clinical_visit",
         "clinical_baseline_episode_id", "clinical_baseline_date", "is_clinical_baseline"},
        "clinical episode spine",
    )
    require_columns(table1, {"patient_id", "clinical_episode_id", "clinical_anchor_date"}, "Table 1 audit")
    require_columns(
        pop,
        {"patient_id", "clinical_episode_id", "clinical_visit", "is_clinical_baseline",
         "pop_status", "pop_baseline_episode_id", "pop_baseline_date", "pop_baseline_status",
         "essdai_total", "esspri_total_observed"},
        "Pop longitudinal artifact",
    )

    clinical = clinical.copy()
    table1 = table1.copy()
    pop = pop.copy()
    for frame in (clinical, table1, pop):
        frame["patient_id"] = frame["patient_id"].astype("string")

    clinical_flag = as_true(clinical["is_clinical_baseline"])
    clinical_counts = clinical.loc[clinical_flag].groupby("patient_id").size()
    clinical_base = clinical.loc[clinical_flag].copy()
    clinical_base = clinical_base.rename(columns={"clinical_visit": "clinical_baseline_clinical_visit"})
    clinical_base = first_patient_rows(clinical_base, [
        "patient_id", "clinical_baseline_episode_id", "clinical_baseline_date",
        "clinical_baseline_clinical_visit",
    ])

    table1_duplicate_count = int(table1["patient_id"].duplicated(keep=False).sum())
    table1 = table1.rename(columns={"clinical_episode_id": "table1_episode_id", "clinical_anchor_date": "table1_date"})
    table1 = first_patient_rows(table1, ["patient_id", "table1_episode_id", "table1_date"])

    # Pop fields are pipeline outputs and are only inspected, never re-derived.
    pop_patient = first_patient_rows(pop, [
        "patient_id", "pop_baseline_episode_id", "pop_baseline_date", "pop_baseline_status"
    ])
    pop_episode = pop.loc[ids_equal(pop["clinical_episode_id"], pop["pop_baseline_episode_id"])].copy()
    pop_episode = pop_episode.rename(columns={"clinical_visit": "pop_baseline_clinical_visit"})
    if pop_episode["patient_id"].duplicated().any():
        # Retain a deterministic row so that the audit can still be emitted before hard failure.
        pop_episode = pop_episode.drop_duplicates("patient_id")
    pop_episode = pop_episode[["patient_id", "pop_baseline_clinical_visit"]]
    clinical_pop_rows = pop.loc[as_true(pop["is_clinical_baseline"])].copy()
    # The longitudinal Pop artifact already carries a propagated column named
    # ``clinical_baseline_pop_status``.  Construct a fresh frame instead of
    # renaming ``pop_status`` to that name, which would create duplicate labels.
    clinical_pop = pd.DataFrame(index=clinical_pop_rows.index)
    clinical_pop["patient_id"] = clinical_pop_rows["patient_id"]
    clinical_pop["clinical_baseline_pop_status"] = clinical_pop_rows["pop_status"]
    clinical_pop["clinical_baseline_essdai_available"] = pd.to_numeric(
        clinical_pop_rows["essdai_total"], errors="coerce"
    ).notna()
    clinical_pop["clinical_baseline_esspri_complete"] = pd.to_numeric(
        clinical_pop_rows["esspri_total_observed"], errors="coerce"
    ).notna()
    clinical_pop = first_patient_rows(
        clinical_pop,
        [
            "patient_id", "clinical_baseline_pop_status",
            "clinical_baseline_essdai_available", "clinical_baseline_esspri_complete",
        ],
    )

    patients = pd.DataFrame({"patient_id": pd.concat([
        clinical["patient_id"], table1["patient_id"], pop["patient_id"]
    ]).dropna().drop_duplicates()})
    audit = patients.merge(clinical_base, on="patient_id", how="left", validate="one_to_one")
    audit = audit.merge(table1, on="patient_id", how="left", validate="one_to_one")
    audit = audit.merge(pop_patient, on="patient_id", how="left", validate="one_to_one")
    audit = audit.merge(pop_episode, on="patient_id", how="left", validate="one_to_one")
    audit = audit.merge(clinical_pop, on="patient_id", how="left", validate="one_to_one")
    for col in ["clinical_baseline_date", "table1_date", "pop_baseline_date"]:
        audit[col] = pd.to_datetime(audit[col], errors="coerce")

    audit["days_clinical_to_pop_baseline"] = (
        audit["pop_baseline_date"] - audit["clinical_baseline_date"]
    ).dt.days.astype("Int64")
    episode_match = ids_equal(audit["table1_episode_id"], audit["clinical_baseline_episode_id"])
    date_match = audit["table1_date"].notna() & audit["clinical_baseline_date"].notna() & audit["table1_date"].eq(audit["clinical_baseline_date"])
    audit["table1_matches_clinical_baseline"] = episode_match & date_match
    audit["pop_matches_clinical_baseline"] = ids_equal(
        audit["pop_baseline_episode_id"], audit["clinical_baseline_episode_id"]
    )

    no_pop = audit["pop_baseline_episode_id"].isna()
    essdai_available = missing_as_false(audit["clinical_baseline_essdai_available"])
    esspri_complete = missing_as_false(audit["clinical_baseline_esspri_complete"])
    low_missing_esspri = (
        audit["clinical_baseline_pop_status"].eq("Unclassifiable")
        & essdai_available
        & ~esspri_complete
    )
    audit["pop_baseline_shift_reason"] = np.select(
        [audit["pop_matches_clinical_baseline"], no_pop,
         ~essdai_available, low_missing_esspri],
        ["same_episode", "no_pop_classifiable_episode", "clinical_baseline_missing_essdai",
         "clinical_baseline_low_essdai_missing_esspri"],
        default="clinical_baseline_unclassifiable_other",
    )

    clinical_nonclinical = ~as_true(audit["clinical_baseline_clinical_visit"]) & audit["clinical_baseline_episode_id"].notna()
    pop_nonclinical = ~as_true(audit["pop_baseline_clinical_visit"]) & audit["pop_baseline_episode_id"].notna()
    pop_before = audit["days_clinical_to_pop_baseline"].lt(0).fillna(False)
    pop_invalid = audit["pop_baseline_episode_id"].notna() & ~audit["pop_baseline_status"].isin(VALID_POP)
    table1_mismatch = audit["table1_episode_id"].notna() & ~audit["table1_matches_clinical_baseline"]
    unexplained = (
        audit["pop_baseline_episode_id"].notna() & ~audit["pop_matches_clinical_baseline"]
        & audit["clinical_baseline_pop_status"].isin(VALID_POP)
    )
    long_shift = audit["days_clinical_to_pop_baseline"].gt(365).fillna(False)
    hard_patient = clinical_nonclinical | pop_nonclinical | pop_before | pop_invalid | table1_mismatch
    review = hard_patient | long_shift | unexplained
    audit["qc_status"] = np.select([hard_patient, review], ["hard_fail", "review"], default="pass")

    metrics = {
        "n_patients": int(len(audit)),
        "n_patients_with_multiple_clinical_baselines": int(clinical_counts.gt(1).sum()),
        "n_table1_duplicate_patient_rows": table1_duplicate_count,
        "n_table1_episode_mismatches": int((audit["table1_episode_id"].notna() & ~episode_match).sum()),
        "n_table1_date_mismatches": int((audit["table1_episode_id"].notna() & ~date_match).sum()),
        "n_nonclinical_clinical_baselines": int(clinical_nonclinical.sum()),
        "n_nonclinical_pop_baselines": int(pop_nonclinical.sum()),
        "n_pop_baselines_before_clinical_baseline": int(pop_before.sum()),
        "n_pop_baselines_with_invalid_status": int(pop_invalid.sum()),
        "n_unexplained_inconsistencies": int(unexplained.sum()),
        "n_shifts_over_365_days": int(long_shift.sum()),
    }
    return audit.sort_values("patient_id"), metrics


def write_outputs(audit: pd.DataFrame, metrics: dict[str, int], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(outdir / "01b_baseline_patient_audit.csv", index=False, date_format="%Y-%m-%d")

    shifted = audit.loc[audit["days_clinical_to_pop_baseline"].gt(0), "days_clinical_to_pop_baseline"].astype(float)
    relationships: list[dict[str, Any]] = [
        {"relationship": "clinical_equals_pop", "n_patients": int(audit["pop_matches_clinical_baseline"].sum()), "value": np.nan},
        {"relationship": "pop_after_clinical", "n_patients": int(audit["days_clinical_to_pop_baseline"].gt(0).sum()), "value": np.nan},
        {"relationship": "pop_before_clinical", "n_patients": int(audit["days_clinical_to_pop_baseline"].lt(0).sum()), "value": np.nan},
        {"relationship": "no_pop_baseline", "n_patients": int(audit["pop_baseline_episode_id"].isna().sum()), "value": np.nan},
    ]
    for name, value in {
        "later_pop_days_median": shifted.median(), "later_pop_days_q1": shifted.quantile(.25),
        "later_pop_days_q3": shifted.quantile(.75), "later_pop_days_iqr": shifted.quantile(.75) - shifted.quantile(.25),
        "later_pop_days_min": shifted.min(), "later_pop_days_max": shifted.max(),
    }.items():
        relationships.append({"relationship": name, "n_patients": int(len(shifted)), "value": value})
    pd.DataFrame(relationships).to_csv(outdir / "01b_baseline_relationship_summary.csv", index=False)

    reasons = audit["pop_baseline_shift_reason"].value_counts(dropna=False).rename_axis("pop_baseline_shift_reason").reset_index(name="n_patients")
    reasons["percent_patients"] = reasons["n_patients"] / len(audit) * 100 if len(audit) else np.nan
    reasons.to_csv(outdir / "01b_baseline_shift_reasons.csv", index=False)
    audit.loc[audit["qc_status"].ne("pass")].to_csv(
        outdir / "01b_baseline_cases_for_review.csv", index=False, date_format="%Y-%m-%d"
    )

    hard_metrics = {
        "n_patients_with_multiple_clinical_baselines", "n_table1_duplicate_patient_rows",
        "n_table1_episode_mismatches", "n_table1_date_mismatches",
        "n_nonclinical_clinical_baselines", "n_nonclinical_pop_baselines",
        "n_pop_baselines_before_clinical_baseline", "n_pop_baselines_with_invalid_status",
    }
    warning_metrics = {"n_unexplained_inconsistencies", "n_shifts_over_365_days"}
    summary = pd.DataFrame([
        {"qc_check": key, "value": value,
         "status": "fail" if key in hard_metrics and value else (
             "warning" if key in warning_metrics and value else "pass"
         )}
        for key, value in metrics.items()
    ])
    summary.to_csv(outdir / "01b_baseline_qc_summary.csv", index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    clinical = read_table(args.clinical_spine)
    table1 = read_table(args.table1_audit)
    pop = read_table(args.pop_longitudinal)
    audit, metrics = build_audit(clinical, table1, pop)
    write_outputs(audit, metrics, args.outdir)
    LOG.info("Wrote five baseline QC artifacts to %s", args.outdir)
    hard_failures = {key: value for key, value in metrics.items() if key in {
        "n_patients_with_multiple_clinical_baselines", "n_table1_duplicate_patient_rows",
        "n_table1_episode_mismatches", "n_table1_date_mismatches",
        "n_nonclinical_clinical_baselines", "n_nonclinical_pop_baselines",
        "n_pop_baselines_before_clinical_baseline", "n_pop_baselines_with_invalid_status",
    } and value}
    if hard_failures:
        raise AssertionError(f"Baseline hard QC failed: {hard_failures}")


if __name__ == "__main__":
    main()

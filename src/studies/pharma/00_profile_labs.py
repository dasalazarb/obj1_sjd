#!/usr/bin/env python3
"""Discover and profile laboratory fields in the integrated episode dataset.

This is an inventory step only: it never changes the input data, harmonizes
units, derives transitions, or fits statistical models.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

import common
from src.studies._shared import load_parquet, validate_integrated_dataset

VALUE_SUFFIX = "__value"
ASSOCIATED_SUFFIXES = (
    "text", "unit", "reference_status", "measurement_date",
    "days_from_anchor", "n_measurements", "conflict", "selection_status",
    "episode_status",
)
OUTPUT_COLUMNS = [
    "lab", "value_column", "text_column", "unit_column",
    "reference_status_column", "measurement_date_column",
    "days_from_anchor_column", "n_measurements_column", "conflict_column",
    "selection_status_column", "episode_status_column", "n_nonmissing",
    "pct_nonmissing", "n_patients", "n_unique", "numeric_parse_pct",
    "inferred_type", "sample_values", "numeric_min", "numeric_median",
    "numeric_max", "n_units", "units_unique", "n_text_unique",
    "text_sample", "reference_status_unique", "recommended_use",
    "review_reason",
]
RECOMMENDATION_ORDER = {
    "numeric_longitudinal": 0,
    "categorical_longitudinal": 1,
    "descriptive_only": 2,
    "review_or_exclude": 3,
}
CLINICAL_CATEGORIES = {
    "normal", "abnormal", "high", "low", "positive", "negative",
    "pos", "neg", "+", "-", "reactive", "nonreactive", "detected",
    "not detected", "present", "absent",
}


def _unique(series: pd.Series) -> list[object]:
    """Return nonmissing unique values in their original encounter order."""
    return series.dropna().drop_duplicates().tolist()


def _display(values: list[object], limit: int = 20) -> str:
    """Serialize a bounded, human-readable value list for one CSV cell."""
    shown = values if len(values) <= limit else values[:limit]
    return " | ".join(str(value) for value in shown)


def _infer_type(n_nonmissing: int, n_unique: int,
                numeric_parse_pct: float) -> str:
    """Apply deliberately small and inspectable type-inference rules."""
    if n_nonmissing == 0:
        return "unknown"
    if numeric_parse_pct >= 0.90 and n_unique > 1:
        return "numeric"
    if numeric_parse_pct <= 0.10 and n_unique <= 20:
        return "categorical"
    if 0.10 < numeric_parse_pct < 0.90:
        return "mixed"
    return "unknown"


def _recommend(inferred_type: str, values: list[object], n_nonmissing: int,
               n_unique: int, n_units: int) -> tuple[str, str]:
    reasons: list[str] = []
    if n_units > 1:
        reasons.append("multiple_units")
    if n_nonmissing < 2:
        reasons.append("very_low_availability")
    if n_nonmissing and n_unique <= 1:
        reasons.append("constant_variable")
    if inferred_type == "mixed":
        reasons.append("mixed_numeric_text")
    if inferred_type == "unknown" and n_nonmissing >= 2 and n_unique > 1:
        reasons.append("unclear_categorical_encoding")
    if reasons:
        return "review_or_exclude", ";".join(reasons)
    if inferred_type == "numeric":
        return "numeric_longitudinal", ""
    if inferred_type == "categorical":
        normalized = {str(value).strip().lower() for value in values}
        if normalized and normalized.issubset(CLINICAL_CATEGORIES):
            return "categorical_longitudinal", ""
        return "descriptive_only", ""
    return "review_or_exclude", "unclear_categorical_encoding"


def profile_labs(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row-per-laboratory inventory without mutating ``frame``."""
    if "patient_id" not in frame:
        raise ValueError("Integrated dataset missing required column: patient_id")
    value_columns = sorted(column for column in frame if column.endswith(VALUE_SUFFIX))
    rows: list[dict[str, object]] = []
    for value_column in value_columns:
        lab = value_column.removesuffix(VALUE_SUFFIX)
        associated = {
            suffix: f"{lab}__{suffix}" if f"{lab}__{suffix}" in frame else ""
            for suffix in ASSOCIATED_SUFFIXES
        }
        series = frame[value_column]
        values = _unique(series)
        nonmissing = series.notna()
        n_nonmissing = int(nonmissing.sum())
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_parse_pct = (
            float(numeric[nonmissing].notna().mean()) if n_nonmissing else 0.0
        )
        n_unique = len(values)
        inferred_type = _infer_type(n_nonmissing, n_unique, numeric_parse_pct)

        unit_values = _unique(frame[associated["unit"]]) if associated["unit"] else []
        text_values = _unique(frame[associated["text"]]) if associated["text"] else []
        reference_values = (
            _unique(frame[associated["reference_status"]])
            if associated["reference_status"] else []
        )
        recommended, reason = _recommend(
            inferred_type, values, n_nonmissing, n_unique, len(unit_values)
        )
        numeric_values = numeric.dropna()
        rows.append({
            "lab": lab,
            "value_column": value_column,
            **{f"{suffix}_column": column for suffix, column in associated.items()},
            "n_nonmissing": n_nonmissing,
            "pct_nonmissing": n_nonmissing / len(frame) if len(frame) else 0.0,
            "n_patients": int(frame.loc[nonmissing, "patient_id"].nunique()),
            "n_unique": n_unique,
            "numeric_parse_pct": numeric_parse_pct,
            "inferred_type": inferred_type,
            "sample_values": _display(values),
            "numeric_min": numeric_values.min() if not numeric_values.empty else pd.NA,
            "numeric_median": numeric_values.median() if not numeric_values.empty else pd.NA,
            "numeric_max": numeric_values.max() if not numeric_values.empty else pd.NA,
            "n_units": len(unit_values),
            "units_unique": _display(unit_values),
            "n_text_unique": len(text_values),
            "text_sample": _display(text_values),
            "reference_status_unique": _display(reference_values),
            "recommended_use": recommended,
            "review_reason": reason,
        })

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if result.empty:
        return result
    result["_recommendation_order"] = result["recommended_use"].map(RECOMMENDATION_ORDER)
    return (result.sort_values(
        ["_recommendation_order", "pct_nonmissing", "lab"],
        ascending=[True, False, True], kind="stable",
    ).drop(columns="_recommendation_order").reset_index(drop=True))


def _print_summary(profile: pd.DataFrame) -> None:
    inferred = profile["inferred_type"]
    recommended = profile["recommended_use"]
    print(f"Total labs detected: {len(profile)}")
    print(f"Numeric labs: {(inferred == 'numeric').sum()}")
    print(f"Categorical labs: {(inferred == 'categorical').sum()}")
    print(f"Mixed/unknown labs: {inferred.isin(['mixed', 'unknown']).sum()}")
    for category in RECOMMENDATION_ORDER:
        print(f"Recommended {category}: {(recommended == category).sum()}")
    print("\nReview first:\n00_pharma_lab_profile.csv")


def run(args: argparse.Namespace) -> pd.DataFrame:
    master = load_parquet(args.integrated)
    validate_integrated_dataset(master)
    profile = profile_labs(master)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(args.output, index=False)
    _print_summary(profile)
    return profile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integrated", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET,
        help="Integrated clinical-episode Parquet dataset",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("00_pharma_lab_profile.csv"),
        help="Destination CSV (default: 00_pharma_lab_profile.csv)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

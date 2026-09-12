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
    "n_text_nonmissing", "pct_text_nonmissing", "n_text_unique",
    "n_reference_status_nonmissing", "pct_reference_status_nonmissing",
    "n_any_result", "pct_any_result", "inferred_type", "result_source",
    "sample_values", "numeric_min", "numeric_median",
    "numeric_max", "n_units", "units_unique",
    "text_sample", "reference_status_unique", "recommended_use",
    "data_sufficiency", "review_reason",
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
DATA_SUFFICIENCY_ORDER = {"adequate": 0, "limited": 1, "insufficient": 2}


def _clean_text(series: pd.Series) -> pd.Series:
    """Return valid textual values without modifying the source series."""
    cleaned = series.dropna().copy()
    cleaned = cleaned.map(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    is_string = cleaned.map(lambda value: isinstance(value, str))
    return cleaned.loc[~(is_string & cleaned.eq(""))]


def _unique(series: pd.Series, *, textual: bool = False) -> list[object]:
    """Return valid unique values in their original encounter order."""
    cleaned = _clean_text(series) if textual else series.dropna()
    return cleaned.drop_duplicates().tolist()


def _valid_mask(series: pd.Series, *, textual: bool = False) -> pd.Series:
    """Identify nonempty results, treating whitespace-only strings as missing."""
    mask = series.notna()
    if textual:
        strings = series.map(lambda value: isinstance(value, str))
        blank_strings = series.astype("string").str.strip().eq("").fillna(False)
        mask &= ~(strings & blank_strings)
    return mask


def _display(values: list[object], limit: int = 20) -> str:
    """Serialize a bounded, human-readable value list for one CSV cell."""
    shown = values if len(values) <= limit else values[:limit]
    return " | ".join(str(value) for value in shown)


def _infer_type(n_nonmissing: int, n_unique: int, numeric_parse_pct: float,
                reference_values: list[object], n_reference: int,
                text_values: list[object], n_text: int) -> tuple[str, str]:
    """Apply deliberately small and inspectable type-inference rules."""
    if n_nonmissing and numeric_parse_pct >= 0.90:
        return "numeric", "value"
    if n_reference >= 2 and len(reference_values) <= 20:
        return "categorical", "reference_status"
    if n_text >= 2 and len(text_values) <= 20:
        return "categorical", "text"
    if n_text:
        return "categorical", "text"
    if 0.10 < numeric_parse_pct < 0.90:
        return "mixed", "value"
    if n_nonmissing and n_unique <= 20:
        return "categorical", "value"
    return "unknown", "none"


def _recommend(inferred_type: str, result_values: list[object], n_any_result: int,
               n_unique: int, n_units: int) -> tuple[str, str]:
    reasons: list[str] = []
    if n_units > 1:
        reasons.append("multiple_units")
    if n_any_result < 2:
        reasons.append("very_low_availability")
    if n_any_result >= 2 and n_unique <= 1:
        reasons.append("constant_variable")
    if inferred_type == "mixed":
        reasons.append("mixed_numeric_text")
    if inferred_type == "unknown" and n_any_result:
        reasons.append("unclear_categorical_encoding")
    if not n_any_result:
        return "review_or_exclude", "no_valid_result"
    blocking_reasons = {"multiple_units", "constant_variable", "mixed_numeric_text"}
    if blocking_reasons.intersection(reasons):
        return "review_or_exclude", ";".join(reasons)
    if inferred_type == "numeric":
        return "numeric_longitudinal", ";".join(reasons)
    if inferred_type == "categorical":
        normalized = {str(value).strip().lower() for value in result_values}
        if normalized and normalized.issubset(CLINICAL_CATEGORIES):
            return "categorical_longitudinal", ";".join(reasons)
        return "descriptive_only", ";".join(reasons)
    return "review_or_exclude", ";".join(reasons or ["unclear_categorical_encoding"])


def _data_sufficiency(n_any_result: int, n_patients: int) -> str:
    """Use intentionally simple coverage bands, independent of inferred type."""
    if n_any_result < 2 or n_patients < 2:
        return "insufficient"
    if n_any_result < 10 or n_patients < 5:
        return "limited"
    return "adequate"


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
        values = _unique(series, textual=True)
        nonmissing = _valid_mask(series, textual=True)
        n_nonmissing = int(nonmissing.sum())
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_parse_pct = (
            float(numeric[nonmissing].notna().mean()) if n_nonmissing else 0.0
        )
        n_unique = len(values)

        unit_series = frame[associated["unit"]] if associated["unit"] else None
        text_series = frame[associated["text"]] if associated["text"] else None
        reference_series = (
            frame[associated["reference_status"]]
            if associated["reference_status"] else None
        )
        unit_values = (
            _unique(unit_series, textual=True) if unit_series is not None else []
        )
        text_values = (
            _unique(text_series, textual=True) if text_series is not None else []
        )
        reference_values = (
            _unique(reference_series, textual=True)
            if reference_series is not None else []
        )
        text_mask = (
            _valid_mask(text_series, textual=True)
            if text_series is not None else pd.Series(False, index=frame.index)
        )
        reference_mask = (
            _valid_mask(reference_series, textual=True)
            if reference_series is not None else pd.Series(False, index=frame.index)
        )
        n_text = int(text_mask.sum())
        n_reference = int(reference_mask.sum())
        any_result = nonmissing | text_mask | reference_mask
        n_any_result = int(any_result.sum())
        n_patients = int(frame.loc[any_result, "patient_id"].nunique())
        inferred_type, result_source = _infer_type(
            n_nonmissing, n_unique, numeric_parse_pct,
            reference_values, n_reference, text_values, n_text,
        )
        source_values = {
            "value": values, "reference_status": reference_values,
            "text": text_values, "none": [],
        }[result_source]
        recommended, reason = _recommend(
            inferred_type, source_values, n_any_result, len(source_values),
            len(unit_values),
        )
        numeric_values = numeric.dropna()
        rows.append({
            "lab": lab,
            "value_column": value_column,
            **{f"{suffix}_column": column for suffix, column in associated.items()},
            "n_nonmissing": n_nonmissing,
            "pct_nonmissing": n_nonmissing / len(frame) if len(frame) else 0.0,
            "n_patients": n_patients,
            "n_unique": n_unique,
            "numeric_parse_pct": numeric_parse_pct,
            "n_text_nonmissing": n_text,
            "pct_text_nonmissing": n_text / len(frame) if len(frame) else 0.0,
            "n_text_unique": len(text_values),
            "n_reference_status_nonmissing": n_reference,
            "pct_reference_status_nonmissing": (
                n_reference / len(frame) if len(frame) else 0.0
            ),
            "n_any_result": n_any_result,
            "pct_any_result": n_any_result / len(frame) if len(frame) else 0.0,
            "inferred_type": inferred_type,
            "result_source": result_source,
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
            "data_sufficiency": _data_sufficiency(n_any_result, n_patients),
            "review_reason": reason,
        })

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if result.empty:
        return result
    result["_recommendation_order"] = result["recommended_use"].map(
        RECOMMENDATION_ORDER
    )
    result["_sufficiency_order"] = result["data_sufficiency"].map(
        DATA_SUFFICIENCY_ORDER
    )
    return (result.sort_values(
        ["_recommendation_order", "_sufficiency_order", "pct_any_result", "lab"],
        ascending=[True, True, False, True], kind="stable",
    ).drop(
        columns=["_recommendation_order", "_sufficiency_order"]
    ).reset_index(drop=True))


def _print_summary(profile: pd.DataFrame) -> None:
    inferred = profile["inferred_type"]
    recommended = profile["recommended_use"]
    print(f"Total labs detected: {len(profile)}")
    print(f"Numeric labs: {(inferred == 'numeric').sum()}")
    print(f"Categorical labs: {(inferred == 'categorical').sum()}")
    print(f"Mixed labs: {(inferred == 'mixed').sum()}")
    print(f"Unknown labs: {(inferred == 'unknown').sum()}")
    for category in RECOMMENDATION_ORDER:
        print(f"Recommended {category}: {(recommended == category).sum()}")
    sufficiency = profile["data_sufficiency"]
    for category in DATA_SUFFICIENCY_ORDER:
        print(f"{category.capitalize()} data: {(sufficiency == category).sum()}")
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

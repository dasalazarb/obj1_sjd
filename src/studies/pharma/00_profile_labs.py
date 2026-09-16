#!/usr/bin/env python3
"""Inventory, profile, and classify laboratory results without changing them."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

import common
from src.studies._shared import load_parquet, validate_integrated_dataset

SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = common.STUDIES_TABLES_DIR / "pharma" / SCRIPT_NAME
DEFAULT_OUTPUT = OUTPUT_DIR / "00_pharma_lab_profile.csv"
EXPECTED_AUDIT_NAME = "00_pharma_expected_lab_audit.csv"
GRAPH_CANDIDATES_NAME = "00_pharma_graph_lab_candidates.csv"

LAB_RESULT_SUFFIXES = ("__value", "__text", "__reference_status")
LAB_METADATA_SUFFIXES = (
    "__unit", "__measurement_date", "__days_from_anchor", "__n_measurements",
    "__conflict", "__selection_status", "__episode_status",
)
REPRESENTATIONS = {
    "value_column": "__value", "text_column": "__text",
    "reference_status_column": "__reference_status", "unit_column": "__unit",
    "measurement_date_column": "__measurement_date",
    "days_from_anchor_column": "__days_from_anchor",
    "n_measurements_column": "__n_measurements", "conflict_column": "__conflict",
    "selection_status_column": "__selection_status",
    "episode_status_column": "__episode_status",
}

EXPECTED_LAB_ALIASES = {
    "anti_ro_ssa": ["anti_ro_ssa", "ssa", "ro52", "ro60", "anti_ro"],
    "anti_la_ssb": ["anti_la_ssb", "ssb", "anti_la"],
    "ana": ["ana", "antinuclear_antibody", "antinuclear_antibodies"],
    "rheumatoid_factor": ["rheumatoid_factor", "rf"],
    "cryoglobulin": ["cryoglobulin", "cryoglobulins"],
    "complement_c3": ["complement_c3", "c3"], "complement_c4": ["complement_c4", "c4"],
    "igg": ["igg", "immunoglobulin_g"], "iga": ["iga", "immunoglobulin_a"],
    "igm": ["igm", "immunoglobulin_m"], "esr": ["esr", "sedimentation_rate"],
    "crp": ["crp", "c_reactive_protein"], "wbc": ["wbc", "white_blood_cell_count"],
    "hemoglobin": ["hemoglobin", "hgb"], "hematocrit": ["hematocrit", "hct"],
    "platelet_count": ["platelet_count", "platelets"],
    "anc": ["anc", "absolute_neutrophil_count"],
    "lymphocyte_count": ["lymphocyte_count", "absolute_lymphocyte_count"],
    "protein_total": ["protein_total", "total_protein"], "albumin": ["albumin"],
    "gamma_globulin": ["gamma_globulin", "gamma_globulins"],
    "protein_total_electrophoresis": ["protein_total_electrophoresis", "spep_total_protein"],
    "creatinine": ["creatinine"], "bun": ["bun", "blood_urea_nitrogen"],
    "ast": ["ast", "aspartate_aminotransferase"], "alt": ["alt", "alanine_aminotransferase"],
    "alkaline_phosphatase": ["alkaline_phosphatase", "alp"],
    "bilirubin_total": ["bilirubin_total", "total_bilirubin"],
    "glucose": ["glucose"], "hemoglobin_a1c": ["hemoglobin_a1c", "hba1c", "a1c"],
}

CLINICAL_CATEGORIES = {
    "normal", "abnormal", "high", "low", "positive", "negative", "pos", "neg",
    "+", "-", "reactive", "nonreactive", "non-reactive", "detected", "not detected",
    "present", "absent", "within range", "within_range",
}
PLACEHOLDERS = {"", "unknown", "not reported", "not_reported", "uninterpretable", "indeterminate", "n/a", "na"}
RECOMMENDATION_ORDER = {
    "numeric_longitudinal": 0, "numeric_requires_harmonization": 1,
    "categorical_longitudinal": 2, "review_required": 3,
    "descriptive_only": 4, "exclude": 5,
}
DATA_SUFFICIENCY_ORDER = {"adequate": 0, "limited": 1, "insufficient": 2}


def _valid_mask(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index, dtype=bool)
    return series.notna() & ~series.astype("string").str.strip().eq("").fillna(False)


def _values(series: pd.Series | None) -> list[object]:
    if series is None:
        return []
    mask = _valid_mask(series, series.index)
    return series.loc[mask].map(lambda x: x.strip() if isinstance(x, str) else x).drop_duplicates().tolist()


def _display(values: list[object], limit: int = 20) -> str:
    return " | ".join(map(str, values[:limit]))


def _norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _clinical(values: list[object]) -> bool:
    normalized = {_norm(value) for value in values} - PLACEHOLDERS
    return bool(normalized) and normalized.issubset(CLINICAL_CATEGORIES)


def _sufficiency(results: int, patients: int) -> str:
    if results < 2 or patients < 2:
        return "insufficient"
    return "limited" if results < 10 or patients < 5 else "adequate"


def _columns(frame: pd.DataFrame, lab: str) -> dict[str, str]:
    return {name: f"{lab}{suffix}" if f"{lab}{suffix}" in frame else "" for name, suffix in REPRESENTATIONS.items()}


def _classify(*, n_value: int, n_value_unique: int, parse_pct: float,
              value_values: list[object], text_values: list[object],
              reference_values: list[object], n_any: int, n_units: int,
              has_conflict: bool) -> tuple[str, str, str, str, list[str]]:
    reasons: list[str] = []
    numeric_variable = n_value > 0 and parse_pct >= .90 and n_value_unique > 1
    mixed = n_value > 0 and 0 < parse_pct < .90
    if n_units > 1:
        reasons.append("multiple_units")
    if mixed:
        reasons.append("mixed_numeric_text")
    if not n_any:
        reasons.append("no_valid_result")
    elif n_any < 2:
        reasons.append("very_low_availability")
    if n_any and max(n_value_unique, len(text_values), len(reference_values)) <= 1:
        reasons.append("constant_result")
    sources = sum(bool(v) for v in (value_values, text_values, reference_values))
    if sources > 1:
        reasons.append("multiple_candidate_result_sources")

    if numeric_variable:
        primary = "value_numeric"
    elif _clinical(reference_values):
        primary = "reference_status_categorical"
    elif text_values and any(_norm(value) not in PLACEHOLDERS for value in text_values):
        primary = "text_categorical"
    elif _clinical(value_values) and parse_pct < .90:
        primary = "value_categorical"
    elif mixed or (n_any and sources > 1):
        primary = "mixed_review"
    elif n_any:
        primary = "none"
    else:
        primary = "none"

    if not n_any or "constant_result" in reasons:
        recommended = "exclude"
    elif primary == "value_numeric":
        recommended = "numeric_requires_harmonization" if n_units > 1 or has_conflict else "numeric_longitudinal"
    elif primary in {"reference_status_categorical", "value_categorical"}:
        recommended = "categorical_longitudinal"
    elif primary == "text_categorical":
        recommended = "categorical_longitudinal" if _clinical(text_values) else "descriptive_only"
    elif primary == "mixed_review":
        recommended = "review_required"
    else:
        recommended = "descriptive_only" if any(_norm(v) not in PLACEHOLDERS for v in text_values + reference_values + value_values) else "exclude"

    graph_status = {
        "numeric_longitudinal": "include_numeric", "categorical_longitudinal": "include_categorical",
        "numeric_requires_harmonization": "review_before_graph", "review_required": "review_before_graph",
        "descriptive_only": "exclude", "exclude": "exclude",
    }[recommended]
    if recommended == "numeric_longitudinal":
        hint = "continuous"
    elif recommended == "categorical_longitudinal":
        chosen = {"reference_status_categorical": reference_values, "text_categorical": text_values, "value_categorical": value_values}[primary]
        hint = "binary" if len({_norm(v) for v in chosen}) == 2 else "categorical_one_hot"
    elif graph_status == "review_before_graph":
        hint = "review"
    else:
        hint = "none"
    return primary, recommended, graph_status, hint, reasons


def profile_labs(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one auditable row per stem discovered from any result suffix."""
    if "patient_id" not in frame:
        raise ValueError("Integrated dataset missing required column: patient_id")
    stems = sorted({column.removesuffix(suffix) for column in frame for suffix in LAB_RESULT_SUFFIXES if column.endswith(suffix)})
    baseline_mask = frame.get("is_clinical_baseline", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    rows: list[dict[str, object]] = []
    denominator = len(frame)
    baseline_denominator = int(baseline_mask.sum())
    for lab in stems:
        cols = _columns(frame, lab)
        value = frame[cols["value_column"]] if cols["value_column"] else None
        text = frame[cols["text_column"]] if cols["text_column"] else None
        reference = frame[cols["reference_status_column"]] if cols["reference_status_column"] else None
        masks = {"value": _valid_mask(value, frame.index), "text": _valid_mask(text, frame.index), "reference": _valid_mask(reference, frame.index)}
        any_mask = masks["value"] | masks["text"] | masks["reference"]
        value_values, text_values, reference_values = _values(value), _values(text), _values(reference)
        numeric = pd.to_numeric(value, errors="coerce") if value is not None else pd.Series(float("nan"), index=frame.index)
        n_value = int(masks["value"].sum())
        parse_pct = float(numeric[masks["value"]].notna().mean()) if n_value else 0.0
        numeric_mask = masks["value"] & numeric.notna()
        categorical_mask = masks["reference"] | masks["text"] | (masks["value"] & ~numeric_mask)
        units = _values(frame[cols["unit_column"]] if cols["unit_column"] else None)
        conflict = frame[cols["conflict_column"]] if cols["conflict_column"] else None
        has_conflict = bool(conflict.fillna(False).astype(bool).any()) if conflict is not None else False
        n_any = int(any_mask.sum())
        n_patients = int(frame.loc[any_mask, "patient_id"].nunique())
        primary, recommended, graph_status, hint, reasons = _classify(
            n_value=n_value, n_value_unique=len(value_values), parse_pct=parse_pct,
            value_values=value_values, text_values=text_values, reference_values=reference_values,
            n_any=n_any, n_units=len(units), has_conflict=has_conflict,
        )
        patient_counts = frame.loc[any_mask].groupby("patient_id", dropna=True).size()
        dual = bool(n_value and masks["reference"].any())
        comparable = masks["value"] & masks["reference"] if value is not None and reference is not None else pd.Series(False, index=frame.index)
        discordant = bool(comparable.any() and any(
            _norm(a) in CLINICAL_CATEGORIES and _norm(b) in CLINICAL_CATEGORIES and _norm(a) != _norm(b)
            for a, b in zip(value.loc[comparable], reference.loc[comparable])
        ))
        if discordant:
            reasons.append("discordant_value_vs_reference_status")
        numeric_values = numeric.dropna()
        baseline_any = any_mask & baseline_mask
        row = {
            "lab": lab, **cols,
            "n_value_nonmissing": n_value, "pct_value_nonmissing": n_value / denominator if denominator else 0.0,
            "n_value_unique": len(value_values), "numeric_parse_pct": parse_pct,
            "numeric_min": numeric_values.min() if len(numeric_values) else pd.NA,
            "numeric_median": numeric_values.median() if len(numeric_values) else pd.NA,
            "numeric_max": numeric_values.max() if len(numeric_values) else pd.NA,
            "n_text_nonmissing": int(masks["text"].sum()), "pct_text_nonmissing": float(masks["text"].mean()) if denominator else 0.0,
            "n_text_unique": len(text_values), "text_unique_sample": _display(text_values),
            "n_reference_status_nonmissing": int(masks["reference"].sum()),
            "pct_reference_status_nonmissing": float(masks["reference"].mean()) if denominator else 0.0,
            "n_reference_status_unique": len(reference_values), "reference_status_unique": _display(reference_values),
            "n_any_result": n_any, "pct_any_result": n_any / denominator if denominator else 0.0,
            "n_patients_any_result": n_patients, "primary_result_source": primary,
            "recommended_use": recommended, "graph_candidate_status": graph_status, "graph_encoding_hint": hint,
            "data_sufficiency": _sufficiency(n_any, n_patients), "n_units": len(units), "units_unique": _display(units),
            "requires_unit_review": len(units) > 1, "multiple_units": len(units) > 1,
            "mixed_numeric_text": n_value > 0 and 0 < parse_pct < .90,
            "constant_result": "constant_result" in reasons,
            "very_low_availability": n_any < 2, "high_missingness": (n_any / denominator < .10) if denominator else True,
            "multiple_candidate_result_sources": sum(bool(v) for v in (value_values, text_values, reference_values)) > 1,
            "discordant_value_vs_reference_status": discordant, "has_dual_representation": dual,
            "n_baseline_any_result": int(baseline_any.sum()),
            "pct_baseline_any_result": float(baseline_any.sum() / baseline_denominator) if baseline_denominator else 0.0,
            "n_baseline_patients": int(frame.loc[baseline_any, "patient_id"].nunique()),
            "n_baseline_numeric": int((numeric_mask & baseline_mask).sum()),
            "pct_baseline_numeric": float((numeric_mask & baseline_mask).sum() / baseline_denominator) if baseline_denominator else 0.0,
            "n_baseline_categorical": int((categorical_mask & baseline_mask).sum()),
            "pct_baseline_categorical": float((categorical_mask & baseline_mask).sum() / baseline_denominator) if baseline_denominator else 0.0,
            "n_patients_with_ge1_result": int((patient_counts >= 1).sum()),
            "n_patients_with_ge2_results": int((patient_counts >= 2).sum()),
            "n_patients_with_ge3_results": int((patient_counts >= 3).sum()),
            "review_reason": ";".join(dict.fromkeys(reasons)),
            # Compatibility aliases retained for downstream readers during migration.
            "n_nonmissing": n_value, "pct_nonmissing": n_value / denominator if denominator else 0.0,
            "n_patients": n_patients, "n_unique": len(value_values), "sample_values": _display(value_values),
            "text_sample": _display(text_values),
            "inferred_type": {"value_numeric": "numeric", "reference_status_categorical": "categorical", "text_categorical": "categorical", "value_categorical": "categorical", "mixed_review": "mixed", "none": "unknown"}[primary],
            "result_source": {"value_numeric": "value", "reference_status_categorical": "reference_status", "text_categorical": "text", "value_categorical": "value", "mixed_review": "value", "none": "none"}[primary],
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["_order"] = result.recommended_use.map(RECOMMENDATION_ORDER)
    result["_sufficiency"] = result.data_sufficiency.map(DATA_SUFFICIENCY_ORDER)
    return result.sort_values(["_order", "_sufficiency", "pct_any_result", "lab"], ascending=[True, True, False, True], kind="stable").drop(columns=["_order", "_sufficiency"]).reset_index(drop=True)


def expected_lab_audit(profile: pd.DataFrame) -> pd.DataFrame:
    """Match expected concepts to stems without silently resolving ambiguity."""
    rows = []
    stems = profile["lab"].tolist() if "lab" in profile else []
    for expected, aliases in EXPECTED_LAB_ALIASES.items():
        matches = sorted({stem for stem in stems if any(stem == alias or stem.startswith(alias + "_") or stem.endswith("_" + alias) for alias in aliases)})
        status = "not_found" if not matches else "found" if len(matches) == 1 else "found_multiple_candidates"
        selected = profile.loc[profile.lab == matches[0]].iloc[0] if len(matches) == 1 else None
        rows.append({
            "expected_lab": expected, "matched_lab_stem": " | ".join(matches), "match_status": status,
            **{column: (selected[column] if selected is not None else "") for column in (
                "value_column", "text_column", "reference_status_column", "primary_result_source",
                "recommended_use", "n_any_result", "n_patients_any_result", "review_reason")},
        })
    return pd.DataFrame(rows)


GRAPH_COLUMNS = [
    "lab", "primary_result_source", "recommended_use", "graph_candidate_status",
    "graph_encoding_hint", "value_column", "text_column", "reference_status_column",
    "pct_baseline_any_result", "n_baseline_patients", "n_patients_with_ge2_results",
    "n_units", "units_unique", "review_reason",
]


def graph_candidates(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame(columns=GRAPH_COLUMNS)
    return profile.loc[profile.graph_candidate_status.isin({"include_numeric", "include_categorical", "review_before_graph"}), GRAPH_COLUMNS].copy()


def _print_summary(profile: pd.DataFrame, audit: pd.DataFrame) -> None:
    recommended = profile.recommended_use if "recommended_use" in profile else pd.Series(dtype=str)
    graph = profile.graph_candidate_status if "graph_candidate_status" in profile else pd.Series(dtype=str)
    found = audit.match_status.ne("not_found")
    review = audit.recommended_use.isin(["numeric_requires_harmonization", "review_required"])
    print("LAB PROFILING SUMMARY\n")
    print(f"Total lab stems detected: {len(profile)}")
    for label, category in (("Numeric longitudinal", "numeric_longitudinal"), ("Numeric requiring harmonization", "numeric_requires_harmonization"), ("Categorical longitudinal", "categorical_longitudinal"), ("Review required", "review_required"), ("Excluded", "exclude")):
        print(f"{label}: {(recommended == category).sum()}")
    print(f"\nExpected SjD labs found: {found.sum()}")
    print(f"Expected SjD labs missing: {(~found).sum()}")
    print(f"Expected labs requiring review: {review.sum()}")
    print("\nBaseline graph candidates:")
    print(f"  numeric: {(graph == 'include_numeric').sum()}")
    print(f"  categorical: {(graph == 'include_categorical').sum()}")
    print(f"  review: {(graph == 'review_before_graph').sum()}")
    print("\nORDER TO REVIEW LAB OUTPUTS")
    print(f"1. {EXPECTED_AUDIT_NAME}\n2. {GRAPH_CANDIDATES_NAME}\n3. {DEFAULT_OUTPUT.name}")


def run(args: argparse.Namespace) -> pd.DataFrame:
    master = load_parquet(args.integrated)
    validate_integrated_dataset(master)
    profile = profile_labs(master)
    audit = expected_lab_audit(profile)
    candidates = graph_candidates(profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(args.output, index=False)
    audit.to_csv(args.output.parent / EXPECTED_AUDIT_NAME, index=False)
    candidates.to_csv(args.output.parent / GRAPH_CANDIDATES_NAME, index=False)
    _print_summary(profile, audit)
    return profile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

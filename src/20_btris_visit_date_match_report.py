#!/usr/bin/env python3
"""BTRIS visit-date matching and unmapped-laboratory audit helpers.

The audit in this module is deliberately descriptive: semantic suggestions are
written only to the audit output and never copied into the longitudinal labs
frame.  The production pipeline should call :func:`write_unmapped_lab_audit`
immediately after ``attach_clinical_context``.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

UNMAPPED_CANDIDATES_NAME = "20_btris_unmapped_lab_candidates.csv"
UNMAPPED_SUMMARY_NAME = "20_btris_unmapped_lab_summary.csv"
AUDIT_PRIORITY_ORDER = [
    "HIGH_PRIORITY_MAP",
    "POSSIBLE_MAP",
    "AMBIGUOUS_REVIEW",
    "LOW_COVERAGE",
    "ADMINISTRATIVE_OR_TEXT",
]
PAIR_COLUMNS = ["order_name_original", "cluster_name_original"]
CANDIDATE_COLUMNS = [
    *PAIR_COLUMNS,
    "n_rows", "n_patients", "n_patients_ge2", "n_patients_ge3",
    "min_date", "max_date", "n_valid_results", "pct_valid_results",
    "n_numeric_exact", "pct_numeric_exact", "n_text_results",
    "pct_text_results", "n_unique_units", "units", "mapping_status",
    "semantic_mapping_status", "present_in_reference",
    "known_cluster_semantic", "suggested_canonical_analyte",
    "suggested_lab_family", "suggested_analytic_role",
    "suggested_mapping_source", "audit_priority",
]


def _first_column(frame: pd.DataFrame, names: tuple[str, ...], *, required: bool = False) -> str | None:
    column = next((name for name in names if name in frame.columns), None)
    if required and column is None:
        raise KeyError(f"Required column not found; tried {', '.join(names)}")
    return column


def _nonblank(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype("string").str.strip().ne("")


def _semantic_fields(value: Any) -> tuple[Any, Any, Any]:
    """Return analyte, family and role from existing semantic dictionary shapes."""
    if value is None:
        return pd.NA, pd.NA, pd.NA
    if isinstance(value, Mapping):
        return (
            value.get("canonical_analyte", pd.NA),
            value.get("lab_family", pd.NA),
            value.get("analytic_role", pd.NA),
        )
    if hasattr(value, "canonical_analyte"):
        return (
            getattr(value, "canonical_analyte", pd.NA),
            getattr(value, "lab_family", pd.NA),
            getattr(value, "analytic_role", pd.NA),
        )
    if isinstance(value, (tuple, list)):
        padded = [*value, pd.NA, pd.NA, pd.NA]
        return tuple(padded[:3])
    return value, pd.NA, pd.NA


def _lookup(mapping: Mapping[Any, Any], key: Any) -> Any:
    """Look up exact keys while tolerating dictionaries keyed by ``order|cluster``."""
    if key in mapping:
        return mapping[key]
    if isinstance(key, tuple):
        for separator in ("|", " + ", "::"):
            joined = separator.join(str(item) for item in key)
            if joined in mapping:
                return mapping[joined]
    return None


def _semantic_suggestion(
    order: Any,
    cluster: Any,
    cluster_semantics: Mapping[Any, Any],
    pair_semantics: Mapping[Any, Any],
    semantic_overrides: Mapping[Any, Any],
) -> dict[str, Any]:
    pair = (order, cluster)
    for mapping, source in (
        (semantic_overrides, "semantic_override"),
        (pair_semantics, "pair_semantic"),
    ):
        value = _lookup(mapping, pair)
        if value is not None:
            analyte, family, role = _semantic_fields(value)
            return {
                "known_cluster_semantic": cluster in cluster_semantics,
                "suggested_canonical_analyte": analyte,
                "suggested_lab_family": family,
                "suggested_analytic_role": role,
                "suggested_mapping_source": source,
            }
    value = _lookup(cluster_semantics, cluster)
    if value is not None:
        analyte, family, role = _semantic_fields(value)
        return {
            "known_cluster_semantic": True,
            "suggested_canonical_analyte": analyte,
            "suggested_lab_family": family,
            "suggested_analytic_role": role,
            "suggested_mapping_source": "cluster_semantic_fallback",
        }
    return {
        "known_cluster_semantic": False,
        "suggested_canonical_analyte": pd.NA,
        "suggested_lab_family": pd.NA,
        "suggested_analytic_role": pd.NA,
        "suggested_mapping_source": "unresolved",
    }


def _constant_or_mixed(series: pd.Series) -> Any:
    values = series.dropna().drop_duplicates()
    return values.iloc[0] if len(values) == 1 else "mixed"


def build_unmapped_lab_candidates(
    labs: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    *,
    cluster_semantics: Mapping[Any, Any] | None = None,
    pair_semantics: Mapping[Any, Any] | None = None,
    semantic_overrides: Mapping[Any, Any] | None = None,
) -> pd.DataFrame:
    """Aggregate observed, semantically unmapped BTRIS pairs for manual review."""
    missing = [column for column in PAIR_COLUMNS if column not in labs]
    if missing:
        raise KeyError(f"Missing audit grouping columns: {', '.join(missing)}")

    cluster_semantics = cluster_semantics if cluster_semantics is not None else globals().get("CLUSTER_SEMANTICS", {})
    pair_semantics = pair_semantics if pair_semantics is not None else globals().get("PAIR_SEMANTICS", {})
    semantic_overrides = semantic_overrides if semantic_overrides is not None else globals().get("SEMANTIC_OVERRIDES", {})
    canonical = labs.get("canonical_analyte", pd.Series(pd.NA, index=labs.index))
    semantic_status = labs.get("semantic_mapping_status", pd.Series(pd.NA, index=labs.index)).astype("string")
    unmapped = labs[canonical.isna() | semantic_status.eq("unexpected_unmapped").fillna(False)].copy()
    if unmapped.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    patient_col = _first_column(unmapped, ("patient_id", "subject_id", "person_id", "mrn"), required=True)
    date_col = _first_column(unmapped, ("lab_date", "result_date", "specimen_date", "order_date"))
    valid_col = _first_column(unmapped, ("result_original", "result_raw", "result_value_original", "result"))
    numeric_col = _first_column(unmapped, ("result_numeric", "result_numeric_exact"))
    text_col = _first_column(unmapped, ("result_text", "result_text_normalized", "result_original", "result_raw"))
    unit_col = _first_column(unmapped, ("unit_original", "units_original", "unit", "units"))

    reference_pairs: set[tuple[Any, Any]] = set()
    if reference is not None:
        reference_order = _first_column(reference, ("order_name_original", "order_name"))
        reference_cluster = _first_column(reference, ("cluster_name_original", "cluster_name"))
        if reference_order and reference_cluster:
            reference_pairs = set(
                reference[[reference_order, reference_cluster]].itertuples(index=False, name=None)
            )

    rows: list[dict[str, Any]] = []
    for pair, group in unmapped.groupby(PAIR_COLUMNS, dropna=False, sort=False):
        patient_counts = group.groupby(patient_col, dropna=True).size()
        valid = _nonblank(group[valid_col]) if valid_col else pd.Series(False, index=group.index)
        numeric = group[numeric_col].notna() if numeric_col else pd.Series(False, index=group.index)
        text = _nonblank(group[text_col]) if text_col else valid & ~numeric
        if text_col in {"result_original", "result_raw"}:
            text &= ~numeric
        units = group.loc[_nonblank(group[unit_col]), unit_col].astype("string").drop_duplicates().tolist() if unit_col else []
        n_rows = len(group)
        suggestion = _semantic_suggestion(pair[0], pair[1], cluster_semantics, pair_semantics, semantic_overrides)
        row = {
            "order_name_original": pair[0], "cluster_name_original": pair[1],
            "n_rows": n_rows, "n_patients": int(patient_counts.size),
            "n_patients_ge2": int(patient_counts.ge(2).sum()),
            "n_patients_ge3": int(patient_counts.ge(3).sum()),
            "min_date": group[date_col].min() if date_col else pd.NaT,
            "max_date": group[date_col].max() if date_col else pd.NaT,
            "n_valid_results": int(valid.sum()),
            "pct_valid_results": 100 * float(valid.mean()),
            "n_numeric_exact": int(numeric.sum()),
            "pct_numeric_exact": 100 * float(numeric.mean()),
            "n_text_results": int(text.sum()),
            "pct_text_results": 100 * float(text.mean()),
            "n_unique_units": len(units), "units": " | ".join(sorted(units)),
            "mapping_status": _constant_or_mixed(group["mapping_status"]) if "mapping_status" in group else pd.NA,
            "semantic_mapping_status": _constant_or_mixed(group["semantic_mapping_status"]) if "semantic_mapping_status" in group else pd.NA,
            "present_in_reference": (
                pair in reference_pairs
                if reference_pairs
                else bool(group.get(
                    "present_in_reference",
                    group.get("mapping_status", pd.Series("unexpected_unmapped", index=group.index))
                    .astype("string").ne("unexpected_unmapped"),
                ).fillna(False).any())
            ),
            **suggestion,
        }
        has_suggestion = pd.notna(row["suggested_canonical_analyte"])
        pair_label = f"{pair[0]} {pair[1]}".lower()
        looks_qualitative_serology = any(
            token in pair_label for token in ("antibody", "serology", "immunoglobulin", "igg", "igm", "iga")
        )
        if (row["pct_numeric_exact"] == 0 and row["pct_text_results"] >= 90
                and not has_suggestion and not looks_qualitative_serology):
            priority = "ADMINISTRATIVE_OR_TEXT"
        elif row["n_patients"] >= 10 and has_suggestion:
            priority = "HIGH_PRIORITY_MAP"
        elif row["n_patients"] >= 5 and has_suggestion:
            priority = "POSSIBLE_MAP"
        elif row["n_patients"] >= 5:
            priority = "AMBIGUOUS_REVIEW"
        else:
            priority = "LOW_COVERAGE"
        row["audit_priority"] = priority
        rows.append(row)

    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    candidates["audit_priority"] = pd.Categorical(
        candidates["audit_priority"], categories=AUDIT_PRIORITY_ORDER, ordered=True
    )
    candidates = candidates.sort_values(
        ["audit_priority", "n_patients", "n_rows"], ascending=[True, False, False], kind="stable"
    ).reset_index(drop=True)
    candidates["audit_priority"] = candidates["audit_priority"].astype("string")
    return candidates


def build_unmapped_lab_summary(candidates: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate pairs and unique exposed patients by priority."""
    patient_col = _first_column(labs, ("patient_id", "subject_id", "person_id", "mrn"), required=True)
    canonical = labs.get("canonical_analyte", pd.Series(pd.NA, index=labs.index))
    status = labs.get("semantic_mapping_status", pd.Series(pd.NA, index=labs.index)).astype("string")
    unmapped = labs[canonical.isna() | status.eq("unexpected_unmapped").fillna(False)].copy()
    priorities = candidates[PAIR_COLUMNS + ["audit_priority"]]
    labelled = unmapped.merge(priorities, on=PAIR_COLUMNS, how="inner")
    rows = []
    for priority in AUDIT_PRIORITY_ORDER:
        subset = candidates[candidates["audit_priority"].eq(priority)]
        patients = labelled.loc[labelled["audit_priority"].eq(priority), patient_col]
        rows.append({"audit_priority": priority, "n_pairs": len(subset),
                     "n_rows": int(subset["n_rows"].sum()),
                     "n_patients_unique": int(patients.nunique(dropna=True))})
    return pd.DataFrame(rows)


def write_unmapped_lab_audit(
    labs: pd.DataFrame,
    reference: pd.DataFrame,
    output_dir: str | Path,
    **semantic_mappings: Mapping[Any, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build, save, and print the two non-mutating BTRIS audit reports."""
    candidates = build_unmapped_lab_candidates(labs, reference, **semantic_mappings)
    summary = build_unmapped_lab_summary(candidates, labs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / UNMAPPED_CANDIDATES_NAME, index=False)
    summary.to_csv(output_dir / UNMAPPED_SUMMARY_NAME, index=False)

    patient_col = _first_column(labs, ("patient_id", "subject_id", "person_id", "mrn"), required=True)
    canonical = labs.get("canonical_analyte", pd.Series(pd.NA, index=labs.index))
    status = labs.get("semantic_mapping_status", pd.Series(pd.NA, index=labs.index)).astype("string")
    unmapped_mask = canonical.isna() | status.eq("unexpected_unmapped").fillna(False)
    observed_pairs = labs[PAIR_COLUMNS].drop_duplicates().shape[0]
    unmapped_pairs = candidates.shape[0]
    print("BTRIS UNMAPPED LAB AUDIT")
    print("------------------------")
    print(f"Total observed lab pairs: {observed_pairs}")
    print(f"Mapped pairs: {observed_pairs - unmapped_pairs}")
    print(f"Unmapped pairs: {unmapped_pairs}\n")
    print(f"Unmapped rows: {int(unmapped_mask.sum())}")
    print(f"Patients with >=1 unmapped lab: {labs.loc[unmapped_mask, patient_col].nunique(dropna=True)}\n")
    counts = candidates["audit_priority"].value_counts()
    for priority in AUDIT_PRIORITY_ORDER:
        print(f"{priority}: {int(counts.get(priority, 0))}")
    print("\nOutputs:")
    print(f"- {UNMAPPED_CANDIDATES_NAME}")
    print(f"- {UNMAPPED_SUMMARY_NAME}")
    return candidates, summary

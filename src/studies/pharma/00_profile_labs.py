#!/usr/bin/env python3
"""Audit observed Step-10 laboratory representations without transforming the master.

Model eligibility requires representation-specific support and an evidenced assay
approval. Context is read-only metadata; reference status is never a new result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

import common
from src.studies._shared import load_parquet, validate_integrated_dataset

SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = common.STUDIES_TABLES_DIR / "pharma" / SCRIPT_NAME
DEFAULT_OUTPUT = OUTPUT_DIR / "00_pharma_lab_profile.csv"
DEFAULT_REGISTRY = Path(__file__).with_name("expected_lab_concepts.yaml")
EXPECTED_AUDIT_NAME = "00_pharma_expected_lab_audit.csv"
GRAPH_CANDIDATES_NAME = "00_pharma_graph_lab_candidates.csv"
SEROLOGY_NAME = "00_pharma_serology_profile.csv"
TEXT_AUDIT_NAME = "00_pharma_text_code_audit.csv"
IMPACT_NAME = "00_pharma_downstream_impact.csv"
SUMMARY_NAME = "00_pharma_profile_summary.json"
CONTRACT_VERSION = "pharma_lab_profile_v2"
KEYS = ["patient_id", "clinical_episode_id"]
LAB_RESULT_SUFFIXES = ("__value", "__text")
CONTEXT_SUFFIXES = ("__unit", "__reference_status", "__measurement_date",
                    "__days_from_anchor", "__n_measurements")
REPRESENTATIONS = {
    "value_column": "__value", "text_column": "__text",
    "reference_status_column": "__reference_status", "unit_column": "__unit",
    "measurement_date_column": "__measurement_date",
    "days_from_anchor_column": "__days_from_anchor",
    "n_measurements_column": "__n_measurements", "conflict_column": "__conflict",
    "selection_status_column": "__selection_status", "episode_status_column": "__episode_status",
}
BINARY_CODES = {"positive": "positive", "pos": "positive", "negative": "negative",
                "neg": "negative", "reactive": "positive", "nonreactive": "negative",
                "non-reactive": "negative"}
PLACEHOLDERS = {"", "unknown", "not reported", "not_reported", "uninterpretable",
                "indeterminate", "n/a", "na", "not tested", "not_tested", "not obtained"}
ORDINAL_CODES = {"negative", "neg", "trace", "1+", "2+", "3+", "4+"}
CENSOR = re.compile(r"^(<=|>=|<|>|≤|≥)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(%)?$")
GRAPH_COLUMNS = [
    "lab", "primary_result_source", "recommended_use", "graph_candidate_status",
    "graph_encoding_hint", "value_column", "text_column", "reference_status_column",
    "pct_baseline_any_result", "n_baseline_patients", "n_patients_with_ge2_results",
    "n_units", "units_unique", "review_reason",
]


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict:
    registry = yaml.safe_load(Path(path).read_text())
    if not isinstance(registry, dict) or not all(k in registry for k in ("registry_version", "concepts", "assays", "thresholds")):
        raise ValueError("Registry requires version, concepts, assays and thresholds")
    required = {"min_numeric_episodes", "min_numeric_patients", "min_numeric_repeated_dates_patients",
                "min_categorical_episodes", "min_categorical_patients", "min_categorical_repeated_dates_patients",
                "min_numeric_parse_fraction"}
    if required - registry["thresholds"].keys():
        raise ValueError("Registry is missing representation support thresholds")
    for key, value in registry["thresholds"].items():
        if key == "min_numeric_parse_fraction":
            if not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise ValueError("Numeric parse fraction must be in (0, 1]")
        elif not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Invalid positive integer threshold: {key}")
    for lab, rule in registry["assays"].items():
        if not lab.startswith("lab__"):
            raise ValueError("Assay approvals must use public lab__ stems")
        if rule.get("approval_status") == "approved" and not all(str(rule.get(k, "")).strip() for k in ("decision_id", "reviewer", "evidence")):
            raise ValueError(f"Approved assay {lab} requires decision_id, reviewer and evidence")
    return registry


def _norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def parse_result(value: object, rule: dict | None = None) -> dict:
    """Describe a token, never impute a censored value or interpret free text.

    Lexically recognized binary tokens remain unapproved until assay review.
    Only approved assay-specific ordinal lists confer ordinal meaning.
    """
    rule = rule or {}
    if pd.isna(value) or _norm(value) in PLACEHOLDERS:
        return {"kind": "missing", "category": None}
    token = _norm(value)
    match = CENSOR.fullmatch(token)
    if match:
        operator = match[1].replace("≤", "<=").replace("≥", ">=")
        return {"kind": "censored_left" if operator.startswith("<") else "censored_right",
                "operator": operator, "bound": float(match[2]), "bound_unit": "%" if match[3] else "",
                "category": None}
    ordinal = [_norm(x) for x in rule.get("ordinal_categories", [])]
    if rule.get("approval_status") == "approved" and token in ordinal:
        return {"kind": "ordinal_approved", "category": token, "order": ordinal.index(token)}
    if token in BINARY_CODES:
        return {"kind": "binary_code", "category": BINARY_CODES[token]}
    if token in ORDINAL_CODES:
        return {"kind": "ordinal_pending", "category": None}
    return {"kind": "unstructured_review", "category": None}


def lab_stem(column: str) -> str | None:
    if column.startswith("lab__"):
        for suffix in LAB_RESULT_SUFFIXES:
            if column.endswith(suffix) and len(column) > len("lab__") + len(suffix):
                return column[:-len(suffix)]
    return None


def _validate_keys(frame: pd.DataFrame, label: str) -> None:
    if not frame.columns.is_unique:
        raise ValueError(f"{label} contains duplicate columns")
    if set(KEYS) - set(frame):
        raise ValueError(f"{label} missing required keys: {KEYS}")
    if frame[KEYS].isna().any().any():
        raise ValueError(f"{label} contains null keys")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{label} contains duplicate patient_id + clinical_episode_id keys")


def _validate_profile_frame(frame: pd.DataFrame) -> None:
    _validate_keys(frame, "Integrated dataset")
    missing = {"is_clinical_baseline", "clinical_anchor_date"} - set(frame)
    if missing:
        raise ValueError(f"Integrated dataset missing required columns: {sorted(missing)}")
    if pd.to_datetime(frame.clinical_anchor_date, errors="coerce").isna().any():
        raise ValueError("Invalid clinical_anchor_date")
    counts = frame.is_clinical_baseline.fillna(False).astype(bool).groupby(frame.patient_id).sum()
    if counts.gt(1).any():
        raise ValueError("At most one clinical baseline is allowed per patient")


def join_lab_context(master: pd.DataFrame, context: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Left join only allowed metadata, preserving input rows, order and values."""
    _validate_keys(master, "Integrated dataset")
    info = {"context_supplied": context is not None, "n_context_keys_unmatched": None,
            "n_context_extra_keys": None, "metadata_columns_from_context": []}
    if context is None:
        return master.copy(), info
    _validate_keys(context, "Context")
    columns = sorted(c for c in context if c.startswith("lab__") and c.endswith(CONTEXT_SUFFIXES))
    # Master metadata, if present in another supported schema, is authoritative.
    columns = [c for c in columns if c not in master]
    joined = master.merge(context[KEYS + columns], how="left", on=KEYS,
                          validate="one_to_one", sort=False, indicator=True)
    info["n_context_keys_unmatched"] = int(joined._merge.eq("left_only").sum())
    info["n_context_extra_keys"] = int((~pd.MultiIndex.from_frame(context[KEYS]).isin(
        pd.MultiIndex.from_frame(master[KEYS]))).sum())
    joined = joined.drop(columns="_merge")
    joined.index = master.index
    pd.testing.assert_frame_equal(joined[master.columns], master)
    info["metadata_columns_from_context"] = columns
    return joined, info


def _valid_mask(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index, dtype=bool)
    return series.notna() & ~series.astype("string").str.strip().eq("").fillna(False)


def _values(series: pd.Series | None) -> list:
    if series is None:
        return []
    return series.loc[_valid_mask(series, series.index)].map(
        lambda x: x.strip() if isinstance(x, str) else x).drop_duplicates().tolist()


def _display(values: list) -> str:
    return " | ".join(map(str, values[:20]))


def _safe_code(value: object) -> str:
    """Export allowlisted lexical codes; arbitrary comments may contain PHI."""
    parsed = parse_result(value)
    token = _norm(value)
    if token in BINARY_CODES or token in ORDINAL_CODES or token in PLACEHOLDERS:
        return token.upper()
    if parsed["kind"].startswith("censored"):
        return parsed["kind"]  # Bounds stay internal; counts suffice for text review.
    return "[UNSTRUCTURED_TEXT_REDACTED]"


def extract_lab_representations(master: pd.DataFrame, context: pd.DataFrame | None = None) -> tuple[list[dict], pd.DataFrame, dict]:
    frame, info = join_lab_context(master, context)
    representations = []
    for lab in sorted({stem for c in master if (stem := lab_stem(c))}):
        columns = {name: f"{lab}{suffix}" if f"{lab}{suffix}" in frame else ""
                   for name, suffix in REPRESENTATIONS.items()}
        representations.append({"lab": lab, **columns})
    return representations, frame, info


def _coverage(frame: pd.DataFrame, mask: pd.Series, baseline: pd.Series) -> dict:
    subset = frame.loc[mask]
    dates = pd.to_datetime(subset.clinical_anchor_date).dt.normalize()
    date_counts = dates.groupby(subset.patient_id).nunique()
    episode_counts = subset.groupby("patient_id").clinical_episode_id.nunique()
    return {"episodes": int(mask.sum()), "patients": int(subset.patient_id.nunique()),
            "baseline_patients": int(frame.loc[mask & baseline, "patient_id"].nunique()),
            "ge2_dates": int(date_counts.ge(2).sum()), "ge2_episodes": int(episode_counts.ge(2).sum())}


def _support(coverage: dict, thresholds: dict, representation: str) -> bool:
    return (coverage["episodes"] >= thresholds[f"min_{representation}_episodes"]
            and coverage["patients"] >= thresholds[f"min_{representation}_patients"]
            and coverage["ge2_dates"] >= thresholds[f"min_{representation}_repeated_dates_patients"])


def _metadata_status(frame: pd.DataFrame, master: pd.DataFrame, column: str, context_supplied: bool) -> tuple[str, str]:
    if column:
        return ("available" if _valid_mask(frame[column], frame.index).any() else "not_assessed", "available")
    return "not_assessed", "not_present_in_context" if context_supplied else "not_available_in_integrated"


def assess_lab_quality_and_coverage(rep: dict, frame: pd.DataFrame, master: pd.DataFrame, registry: dict, context_supplied: bool) -> dict:
    lab = rep["lab"]
    rule = registry["assays"].get(lab, {})
    baseline = frame.is_clinical_baseline.fillna(False).astype(bool)
    series = lambda key: frame[rep[key]] if rep[key] else pd.Series(pd.NA, index=frame.index)
    value, text, reference = series("value_column"), series("text_column"), series("reference_status_column")
    vm, tm, rm = (_valid_mask(x, frame.index) for x in (value, text, reference))
    numeric = pd.to_numeric(value, errors="coerce").astype(float)
    nm = vm & pd.Series(np.isfinite(numeric), index=frame.index)
    numeric = numeric.where(nm)
    parsed = text.map(lambda x: parse_result(x, rule))
    kinds = parsed.map(lambda x: x["kind"])
    cm = kinds.isin(["binary_code", "ordinal_approved"])
    categories = parsed.map(lambda x: x.get("category"))
    nc, cc = _coverage(frame, nm, baseline), _coverage(frame, cm, baseline)
    any_mask = vm | tm  # Reference status is metadata, never an observed result.
    any_counts = frame.loc[any_mask].groupby("patient_id").size()
    units = sorted(map(str, _values(series("unit_column"))))
    u_status, u_avail = _metadata_status(frame, master, rep["unit_column"], context_supplied)
    r_status, r_avail = _metadata_status(frame, master, rep["reference_status_column"], context_supplied)
    unit_complete = bool(nm.any() and rep["unit_column"] and _valid_mask(series("unit_column"), frame.index)[nm].all())
    if len(units) > 1:
        u_status = "multiple_units_review"
    elif rep["unit_column"] and not unit_complete:
        u_status = "incomplete_metadata_review" if units else "not_assessed"
    conflict = series("conflict_column")
    conflict_tokens = conflict.map(_norm)
    conflict_mask = _valid_mask(conflict, frame.index)
    explicit_true = conflict_mask & conflict_tokens.isin({"true", "1", "1.0", "yes"})
    explicit_false = conflict_mask & conflict_tokens.isin({"false", "0", "0.0", "no", "none"})
    qc_complete = bool(any_mask.any() and explicit_false.loc[any_mask].all())
    has_conflict = True if explicit_true.any() else False if qc_complete else pd.NA
    conflict_status = ("conflict_detected" if explicit_true.any() else "available" if qc_complete
                       else "incomplete_qc_review" if conflict_mask.any()
                       else "not_evaluable_without_upstream_qc")
    sources = sorted({"integrated" if c in master else "context" for k, c in rep.items()
                      if k.endswith("_column") and c and k not in {"value_column", "text_column"}})
    n, bd = len(frame), int(baseline.sum())
    frac = lambda count, denominator: count / denominator if denominator else 0.0
    numeric_unique = int(numeric.nunique())
    parse_pct = frac(nc["episodes"], int(vm.sum()))
    numeric_supported = _support(nc, registry["thresholds"], "numeric") and parse_pct >= registry["thresholds"]["min_numeric_parse_fraction"]
    categorical_supported = _support(cc, registry["thresholds"], "categorical")
    overlap = vm & tm
    categorical_approved = rule.get("approval_status") == "approved" and rule.get("binary_approved") is True
    # Existing consumers read RAW columns. Do not promote normalized categories
    # while their raw text could encode censored/unknown/comment results as levels.
    raw_tokens = set(text.loc[tm].map(_norm)) - PLACEHOLDERS
    raw_compatible = bool(raw_tokens) and (raw_tokens <= {"positive", "negative"} or raw_tokens <= {"pos", "neg"})
    numeric_frame = pd.DataFrame({"patient": frame.patient_id, "value": numeric}).dropna()
    patient_means = numeric_frame.groupby("patient").value.mean()
    vv = _values(value)
    row = {
        **rep, "profile_contract_version": CONTRACT_VERSION,
        "n_value_nonmissing": int(vm.sum()), "pct_value_nonmissing": frac(int(vm.sum()), n),
        "n_value_unique": len(vv), "numeric_parse_pct": parse_pct,
        "numeric_min": numeric.min(), "numeric_median": numeric.median() if nm.any() else np.nan, "numeric_max": numeric.max(),
        "n_text_nonmissing": int(tm.sum()), "pct_text_nonmissing": frac(int(tm.sum()), n),
        "n_text_unique": int(text.loc[tm].nunique()),
        "text_unique_sample": _display(sorted({_safe_code(x) for x in _values(text)})),
        "n_reference_status_nonmissing": int(rm.sum()), "pct_reference_status_nonmissing": frac(int(rm.sum()), n),
        "n_reference_status_unique": int(reference.loc[rm].nunique()),
        "reference_status_unique": _display(sorted({_safe_code(x) for x in _values(reference)})),
        "n_any_result": int(any_mask.sum()), "pct_any_result": frac(int(any_mask.sum()), n),
        "n_patients_any_result": int(frame.loc[any_mask, "patient_id"].nunique()),
        "n_baseline_denominator": bd, "n_patient_denominator": int(frame.patient_id.nunique()),
        "n_baseline_any_result": int((any_mask & baseline).sum()),
        "pct_baseline_any_result": frac(int((any_mask & baseline).sum()), bd),
        "n_baseline_patients": int(frame.loc[any_mask & baseline, "patient_id"].nunique()),
        "n_baseline_numeric": nc["baseline_patients"], "pct_baseline_numeric": frac(nc["baseline_patients"], bd),
        "n_baseline_categorical": cc["baseline_patients"], "pct_baseline_categorical": frac(cc["baseline_patients"], bd),
        "pct_baseline_interpretable_categorical": frac(cc["baseline_patients"], bd),
        "n_patients_with_ge1_result": int(any_counts.ge(1).sum()),
        "n_patients_with_ge2_results": int(any_counts.ge(2).sum()),
        "n_patients_with_ge3_results": int(any_counts.ge(3).sum()),
        "n_numeric_episodes": nc["episodes"], "n_numeric_patients": nc["patients"],
        "n_numeric_baseline_patients": nc["baseline_patients"], "n_numeric_patients_ge2_dates": nc["ge2_dates"],
        "n_episodes_value": int(vm.sum()), "n_patients_value": int(frame.loc[vm, "patient_id"].nunique()),
        "n_baseline_patients_value": int(frame.loc[vm & baseline, "patient_id"].nunique()),
        "n_patients_ge2_numeric_episodes": nc["ge2_episodes"], "n_patients_ge2_numeric_dates": nc["ge2_dates"],
        "n_raw_text_episodes": int(tm.sum()), "n_episodes_text_raw": int(tm.sum()),
        "n_categorical_interpretable_episodes": cc["episodes"], "n_episodes_text_interpretable": cc["episodes"],
        "n_categorical_interpretable_patients": cc["patients"],
        "n_categorical_baseline_patients": cc["baseline_patients"],
        "n_categorical_patients_ge2_dates": cc["ge2_dates"], "n_patients_ge2_categorical_dates": cc["ge2_dates"],
        "n_left_censored": int(kinds.eq("censored_left").sum()), "n_right_censored": int(kinds.eq("censored_right").sum()),
        "n_unstructured_text": int(kinds.eq("unstructured_review").sum()),
        "n_text_censored_left": int(kinds.eq("censored_left").sum()), "n_text_censored_right": int(kinds.eq("censored_right").sum()),
        "n_text_unstructured": int(kinds.eq("unstructured_review").sum()),
        "n_text_ordinal_pending": int(kinds.eq("ordinal_pending").sum()),
        "n_text_unknown": int((tm & kinds.eq("missing")).sum()),
        "n_episodes_interpretable": int((nm | cm).sum()),
        "n_value_text_overlap": int(overlap.sum()), "n_overlap_censored": int((overlap & kinds.str.startswith("censored")).sum()),
        "n_overlap_informative": int((overlap & cm).sum()),
        "n_overlap_comment": int((overlap & kinds.eq("unstructured_review")).sum()),
        "has_dual_representation": bool(overlap.any()), "dual_representation_assessment_status": "available",
        "discordant_value_vs_reference_status": pd.NA, "concordance_assessment_status": "not_assessed",
        "n_units": len(units), "units_unique": _display(units),
        "multiple_units": len(units) > 1 if u_avail == "available" else pd.NA,
        "requires_unit_review": u_status != "available", "unit_assessment_status": u_status,
        "unit_status": u_status, "unit_availability_status": u_avail,
        "reference_assessment_status": r_status, "reference_availability_status": r_avail,
        "conflict_assessment_status": conflict_status, "has_conflict": has_conflict,
        "metadata_source": "+".join(sources) if sources else "none",
        "numeric_support_met": numeric_supported, "categorical_support_met": categorical_supported,
        "n_nonfinite_numeric_values": int((vm & pd.to_numeric(value, errors="coerce").isin([np.inf, -np.inf])).sum()),
        "numeric_n_unique": numeric_unique, "categorical_n_unique": int(categories.loc[cm].nunique()),
        "raw_categorical_consumer_compatible": raw_compatible,
        "observed_value_type": "genetic" if lab.startswith("lab__hla_") else "mixed" if nm.any() and (~nm & vm).any() else "numeric_point" if nm.any() else "unstructured_text" if vm.any() else "absent",
        "observed_text_type": "genetic" if lab.startswith("lab__hla_") else "mixed" if kinds.loc[tm].nunique() > 1 else kinds.loc[tm].iloc[0] if tm.any() else "absent",
        "value_support_status": "adequate" if numeric_supported else "limited_for_longitudinal" if nm.any() else "absent",
        "text_parse_status": "approved" if categorical_approved and cm.any() else "pending_review" if cm.any() else "unstructured_review" if tm.any() else "absent",
        "data_present": bool(any_mask.any()), "parseable_result": bool((nm | cm).any()),
        "clinically_approved": rule.get("approval_status", "pending"),
        "approval_decision_id": rule.get("decision_id", ""),
        "within_patient_variation": numeric_frame.groupby("patient").value.var().mean(),
        "between_patient_variation": patient_means.var(),
        "mixed_numeric_text": bool(vm.any() and (~nm & vm).any()),
        "multiple_candidate_result_sources": bool(vm.any() and tm.any()),
        "very_low_availability": int(any_mask.sum()) < 2, "high_missingness": frac(int(any_mask.sum()), n) < .1,
        "n_nonmissing": int(vm.sum()), "pct_nonmissing": frac(int(vm.sum()), n),
        "n_patients": int(frame.loc[any_mask, "patient_id"].nunique()), "n_unique": len(vv),
        "sample_values": _display(numeric.dropna().drop_duplicates().tolist()),
    }
    row["text_sample"] = row["text_unique_sample"]
    return row


def decide_downstream_eligibility(row: dict, rule: dict) -> dict:
    """Keep legacy consumer labels; approval and support are independent gates."""
    numeric = row["n_numeric_episodes"] > 0
    categorical = row["n_categorical_interpretable_episodes"] > 0
    primary = "value_numeric" if numeric else "text_categorical" if row["n_text_nonmissing"] else "mixed_review" if row["n_value_nonmissing"] else "none"
    source = row["value_column"] if primary in {"value_numeric", "mixed_review"} else row["text_column"] if primary == "text_categorical" else ""
    reasons = []
    approved = rule.get("approval_status") == "approved" and all(rule.get(k) for k in ("decision_id", "reviewer", "evidence"))
    genetic = row["lab"].startswith("lab__hla_")
    constant = (numeric and row["numeric_n_unique"] <= 1 and not row["n_text_nonmissing"])
    status = "requires_clinical_text_review" if row["n_text_nonmissing"] and not numeric else "limited_for_longitudinal"
    recommended = "descriptive_only"
    chosen_baseline = row["n_numeric_baseline_patients"] if numeric else row["n_categorical_baseline_patients"]
    chosen_longitudinal = row["n_numeric_patients_ge2_dates"] if numeric else row["n_categorical_patients_ge2_dates"]
    if not row["data_present"]:
        recommended, status = "exclude", "exclude_no_data"
        reasons.append("no_valid_result")
    elif genetic:
        status = "genetic_descriptive_only"
        reasons.append("genotype_not_dynamic_measurement")
    elif constant:
        recommended, status = "exclude", "exclude_no_variability"
        reasons.append("constant_numeric_result")
    elif numeric:
        if not row["numeric_support_met"]:
            reasons.append("insufficient_numeric_representation_support")
        elif row["n_nonfinite_numeric_values"]:
            recommended, status = "review_required", "requires_numeric_quality_review"
            reasons.append("nonfinite_values_not_safe_for_existing_consumers")
        elif row["unit_assessment_status"] in {"multiple_units_review", "incomplete_metadata_review"} or row["has_conflict"] is True:
            recommended, status = "numeric_requires_harmonization", "requires_metadata_review"
            reasons.append("multiple_units" if row["n_units"] > 1 else "incomplete_units_or_conflict")
        elif not approved or rule.get("numeric_approved") is not True:
            recommended, status = "review_required", "pending_clinical_approval"
            reasons.append("numeric_assay_approval_pending")
        elif row["unit_assessment_status"] == "not_assessed" and rule.get("allow_unassessed_units") is not True:
            recommended, status = "review_required", "requires_metadata_review"
            reasons.append("unit_comparability_not_assessed")
        else:
            recommended, status = "numeric_longitudinal", "model_ready"
            reasons.append("approved_numeric_representation_support")
            if row["unit_assessment_status"] == "not_assessed":
                reasons.append("provisional_units_explicitly_approved_not_harmonized")
    elif categorical:
        if not approved or rule.get("binary_approved") is not True:
            reasons.append("categorical_semantics_pending_approval")
        elif not row["categorical_support_met"]:
            reasons.append("insufficient_categorical_representation_support")
        elif row["categorical_n_unique"] < 2:
            recommended, status = "exclude", "exclude_no_variability"
            reasons.append("constant_categorical_result")
        elif row["has_conflict"] is True:
            recommended, status = "review_required", "requires_metadata_review"
            reasons.append("conflict_detected")
        elif not row["raw_categorical_consumer_compatible"]:
            recommended, status = "review_required", "requires_consumer_encoding_review"
            reasons.append("raw_text_not_safe_for_existing_consumers")
        else:
            recommended, status = "categorical_longitudinal", "model_ready"
            reasons.append("approved_binary_representation_support")
    else:
        reasons.append("no_validated_structured_text_encoding")
    graph_status = {"numeric_longitudinal": "include_numeric", "categorical_longitudinal": "include_categorical",
                    "numeric_requires_harmonization": "review_before_graph", "review_required": "review_before_graph",
                    "descriptive_only": "exclude", "exclude": "exclude"}[recommended]
    ready = status == "model_ready"
    return {"primary_result_source": primary, "source_column": source, "public_column": source,
            "result_source": "value" if primary in {"value_numeric", "mixed_review"} else "text" if source else "none",
            "inferred_type": "numeric" if numeric else "categorical" if categorical else "unknown",
            "recommended_use": recommended, "graph_candidate_status": graph_status,
            "graph_encoding_hint": "continuous" if graph_status == "include_numeric" else "binary" if graph_status == "include_categorical" else "review" if graph_status == "review_before_graph" else "none",
            "data_sufficiency": "adequate" if (row["numeric_support_met"] if numeric else row["categorical_support_met"]) else "limited" if row["parseable_result"] else "insufficient",
            "status": status, "model_ready": ready, "clinical_review_required": row["data_present"] and not ready,
            "clinical_interpretation_review": bool(row["n_text_nonmissing"] and not ready),
            "constant_result": bool(constant), "eligibility_reason": ";".join(reasons),
            "review_reason": ";".join(reasons), "exclusion_reason": ";".join(reasons) if recommended in {"exclude", "descriptive_only"} else "",
            "n_eligible_baseline_patients": chosen_baseline if ready else 0,
            "n_eligible_longitudinal_patients": chosen_longitudinal if ready else 0}


def profile_labs(frame: pd.DataFrame, context: pd.DataFrame | None = None, *, registry: dict | None = None) -> pd.DataFrame:
    _validate_profile_frame(frame)
    registry = registry if registry is not None else load_registry()
    reps, joined, info = extract_lab_representations(frame, context)
    rows = []
    for rep in reps:
        row = assess_lab_quality_and_coverage(rep, joined, frame, registry, context is not None)
        row.update(decide_downstream_eligibility(row, registry["assays"].get(rep["lab"], {})))
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=list(dict.fromkeys(["lab", *REPRESENTATIONS, *GRAPH_COLUMNS,
                                                        "result_source", "source_column", "status"])))
    result.attrs["context_audit"] = info
    return result


def expected_lab_audit(profile: pd.DataFrame, *, registry: dict | None = None) -> pd.DataFrame:
    registry = registry if registry is not None else load_registry()
    stems = set(profile.lab)
    rows = []
    for concept, entry in registry["concepts"].items():
        preferred = entry["preferred_lab_stem"]
        aliases = [s for s in entry.get("approved_exact_aliases", []) if s in stems]
        related = [s for s in entry.get("related_not_equivalent", []) if s in stems]
        selected_stem = preferred if preferred in stems else aliases[0] if len(aliases) == 1 else ""
        selected = profile.loc[profile.lab.eq(selected_stem)].iloc[0] if selected_stem else {}
        distinct = related or (len(aliases) > 1 and preferred not in stems)
        status = "multiple_distinct_assays_review" if distinct else "found" if selected_stem else "not_found"
        rows.append({"expected_lab": concept, "matched_lab_stem": selected_stem,
                     "match_status": status, "preferred_lab_stem": preferred,
                     "preferred_match_status": "exact" if preferred in stems else "approved_exact_alias" if selected_stem else "not_found",
                     "related_lab_stems": " | ".join(related), "related_assays": " | ".join(related),
                     "related_match_status": "multiple_distinct_assays_review" if related else "none_present",
                     "concept_family": entry["concept_family"], "clinical_equivalence_status": entry["clinical_equivalence_status"],
                     "selected_source_column": selected.get("source_column", ""),
                     "requires_clinical_review": bool(related) or selected.get("clinically_approved", "pending") != "approved",
                     "note": entry.get("note", ""),
                     **{column: selected.get(column, "") for column in (
                         "value_column", "text_column", "reference_status_column", "primary_result_source",
                         "recommended_use", "n_any_result", "n_patients_any_result", "review_reason")}})
    return pd.DataFrame(rows)


def graph_candidates(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame(columns=GRAPH_COLUMNS)
    return profile.loc[profile.graph_candidate_status.isin(
        {"include_numeric", "include_categorical", "review_before_graph"}), GRAPH_COLUMNS + [
            "source_column", "profile_contract_version", "clinical_review_required", "clinically_approved",
            "unit_assessment_status", "eligibility_reason"]].copy()


SEROLOGY_COLUMNS = ["column", "kind", "clinical_concept", "temporal_semantics", "grain",
                    "n_nonmissing_episodes", "n_patients", "n_baseline_patients", "source_status",
                    "usable_for_descriptive", "potential_baseline_predictor", "potential_time_varying_predictor",
                    "leakage_risk", "clinical_review_required"]


def serology_profile(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = frame.is_clinical_baseline.fillna(False).astype(bool)
    for column in sorted(c for c in frame if c.startswith("sero__")):
        tail = column[len("sero__"):]
        if tail.startswith("baseline_"):
            kind, temporal, grain = "baseline_summary", "baseline_definition_requires_upstream_review", "patient_summary_repeated_on_episodes"
        elif tail.endswith("__ever_positive_through_episode"):
            kind, temporal, grain = "cumulative_asof", "cumulative_history_not_current_result_asof_requires_audit", "patient_episode_derived_state"
        elif tail.endswith("__patient_consensus_value"):
            kind, temporal, grain = "patient_level_consensus", "may_use_future_observations_not_temporal_predictor", "patient_summary_repeated_on_episodes"
        elif tail.endswith("__asof_value"):
            kind, temporal, grain = "genetic_asof", "knowledge_date_requires_validation_not_dynamic_measurement", "patient_episode_known_genotype"
        else:
            kind, temporal, grain = "unknown_derived", "unverified", "unknown"
        mask = _valid_mask(frame[column], frame.index)
        rows.append({"column": column, "kind": kind, "clinical_concept": tail.removeprefix("baseline_").split("__")[0],
                     "temporal_semantics": temporal, "grain": grain,
                     "n_nonmissing_episodes": int(mask.sum()), "n_patients": int(frame.loc[mask, "patient_id"].nunique()),
                     "n_baseline_patients": int(frame.loc[mask & baseline, "patient_id"].nunique()),
                     "source_status": "derived_from_step10_not_episode_lab_measurement", "usable_for_descriptive": True,
                     "potential_baseline_predictor": False, "potential_time_varying_predictor": False,
                     "leakage_risk": "high_future_information" if kind == "patient_level_consensus" else "temporal_construction_unverified",
                     "clinical_review_required": True})
    return pd.DataFrame(rows, columns=SEROLOGY_COLUMNS)


def text_code_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in sorted(c for c in frame if lab_stem(c) and c.endswith("__text")):
        codes = frame.loc[_valid_mask(frame[column], frame.index), column].map(_safe_code).value_counts()
        for code, count in sorted(codes.items()):
            rows.append({"lab": lab_stem(column), "source_column": column,
                         "original_code_or_redacted_type": code, "n_episodes": int(count)})
    return pd.DataFrame(rows, columns=["lab", "source_column", "original_code_or_redacted_type", "n_episodes"])


def downstream_impact(profile: pd.DataFrame, previous: pd.DataFrame | None) -> pd.DataFrame:
    """Compare genuine per-lab snapshots only; never invent historical rows."""
    if previous is not None and ("lab" not in previous or previous.lab.duplicated().any()):
        raise ValueError("Previous profile requires unique lab stems")
    old = previous.set_index("lab").to_dict("index") if previous is not None else {}
    new = profile.set_index("lab").to_dict("index")
    rows = []
    for lab in sorted(set(old) | set(new)):
        before, after = old.get(lab, {}), new.get(lab, {})
        eligibility = lambda r: r.get("recommended_use") in {"numeric_longitudinal", "categorical_longitudinal"}
        graph_eligible = lambda r: r.get("graph_candidate_status") in {"include_numeric", "include_categorical"}
        def impact(predicate):
            if previous is None:
                return "not_comparable_previous_snapshot_missing"
            return "enters" if predicate(after) and not predicate(before) else "exits" if predicate(before) and not predicate(after) else "retained" if predicate(after) else "remains_ineligible"
        rows.append({"lab": lab, "old_status": before.get("recommended_use", "not_provided" if previous is None else "not_in_snapshot"),
                     "new_status": after.get("recommended_use", "not_in_current_schema"),
                     "old_source_column": before.get("source_column", ""), "new_source_column": after.get("source_column", ""),
                     "old_eligible": eligibility(before) if previous is not None else "unknown",
                     "new_eligible": eligibility(after), "reason": after.get("eligibility_reason", "removed_from_schema"),
                     "impact_pharma01": impact(eligibility), "impact_graph01": impact(graph_eligible),
                     "requires_manual_approval": impact(eligibility) in {"enters", "exits"} or impact(graph_eligible) in {"enters", "exits"} or previous is None,
                     **{k: after.get(k, 0) for k in ("n_numeric_episodes", "n_numeric_patients", "n_numeric_baseline_patients",
                                                    "n_numeric_patients_ge2_dates", "n_categorical_interpretable_episodes", "n_categorical_patients_ge2_dates")}})
    return pd.DataFrame(rows, columns=["lab", "old_status", "new_status", "old_source_column", "new_source_column",
                                      "old_eligible", "new_eligible", "reason", "impact_pharma01", "impact_graph01",
                                      "requires_manual_approval", "n_numeric_episodes", "n_numeric_patients",
                                      "n_numeric_baseline_patients", "n_numeric_patients_ge2_dates",
                                      "n_categorical_interpretable_episodes", "n_categorical_patients_ge2_dates"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> pd.DataFrame:
    integrated, output = Path(args.integrated), Path(args.output)
    context_path = getattr(args, "context", None)
    if getattr(args, "no_context", False):
        context_path, mode = None, "integrated_only_explicit"
    elif context_path is not None:
        context_path, mode = Path(context_path), "explicit_context"
        if not context_path.is_file():
            raise FileNotFoundError(f"Explicit context file not found: {context_path}")
    else:
        companion = integrated.with_name("10_integrated_longitudinal_context.parquet")
        context_path = companion if companion.is_file() else None
        mode = "adjacent_context" if context_path else "integrated_only_no_companion"
    logging.info("Laboratory metadata mode: %s", mode)
    registry_path = Path(getattr(args, "registry", DEFAULT_REGISTRY))
    previous_path = getattr(args, "previous_profile", None)
    inputs = [integrated, registry_path] + ([context_path] if context_path else []) + ([Path(previous_path)] if previous_path else [])
    destinations = [output] + [output.parent / name for name in (EXPECTED_AUDIT_NAME, GRAPH_CANDIDATES_NAME,
                    SEROLOGY_NAME, TEXT_AUDIT_NAME, IMPACT_NAME, SUMMARY_NAME)]
    if output.suffix.lower() != ".csv" or len(set(p.resolve() for p in destinations)) != len(destinations):
        raise ValueError("--output must be a distinct profile CSV path")
    if any(out.resolve() == inp.resolve() for out in destinations for inp in inputs):
        raise ValueError("Profiling outputs must not overwrite any input")
    hashes = {path: _sha256(path) for path in inputs}
    registry = load_registry(registry_path)
    master = load_parquet(integrated)
    _validate_profile_frame(master)
    contract = validate_integrated_dataset(master)
    context = load_parquet(context_path) if context_path else None
    profile = profile_labs(master, context, registry=registry)
    audit = expected_lab_audit(profile, registry=registry)
    candidates = graph_candidates(profile)
    serology = serology_profile(master)
    impact = downstream_impact(profile, pd.read_csv(previous_path).fillna("") if previous_path else None)
    counts = lambda table, column: {str(k): int(v) for k, v in table[column].value_counts().sort_index().items()} if column in table else {}
    schema = {"n_columns": len(master.columns), "lab_stems": len(profile),
              "lab_value_columns": sum(c.startswith("lab__") and c.endswith("__value") for c in master),
              "lab_text_columns": sum(c.startswith("lab__") and c.endswith("__text") for c in master), "serology_columns": len(serology)}
    expected_schema = {"n_columns": 485, "lab_stems": 177, "lab_value_columns": 100, "lab_text_columns": 177, "serology_columns": 36}
    summary = {"profile_contract_version": CONTRACT_VERSION, "registry_version": registry["registry_version"],
               "thresholds": registry["thresholds"], "dataset_contract": contract,
               "input_hashes": [{"role": "integrated" if p == integrated else "registry" if p == registry_path else "previous_profile" if previous_path and p == Path(previous_path) else "context", "sha256": sha} for p, sha in hashes.items()],
               "context_mode": mode, "context_audit": profile.attrs["context_audit"],
               "schema": schema, "reference_schema": expected_schema,
               "schema_comparison_scope": "documented_family_counts_only_exact_header_not_supplied", "schema_matches_reference": schema == expected_schema,
               "schema_drift": {key: {"expected": expected_schema[key], "observed": value} for key, value in schema.items() if value != expected_schema[key]},
               "reference_snapshot": {"source": "investigator_specification_2026-10-16_not_independently_verified", "labs": 177,
                                      "numeric_longitudinal": 99, "categorical_longitudinal": 11, "descriptive_only": 41, "exclude": 26, "graph_candidates": 110},
               "availability_lab_counts": {key: int(profile[key].gt(0).sum()) if key in profile else 0 for key in (
                   "n_numeric_episodes", "n_numeric_baseline_patients", "n_numeric_patients_ge2_dates",
                   "n_categorical_interpretable_episodes", "n_categorical_baseline_patients",
                   "n_categorical_patients_ge2_dates", "n_eligible_baseline_patients", "n_eligible_longitudinal_patients")},
               "new_recommended_use_counts": counts(profile, "recommended_use"), "new_status_counts": counts(profile, "status"),
               "graph_candidate_counts": counts(candidates, "graph_candidate_status"), "expected_concept_counts": counts(audit, "match_status"),
               "serology_kind_counts": counts(serology, "kind"), "previous_snapshot_available": previous_path is not None,
               "downstream_impact_counts": counts(impact, "impact_pharma01"),
               "downstream_graph_impact_counts": counts(impact, "impact_graph01"),
               "release_recommendation": "hold_for_clinical_diff_and_temporal_review_no_downstream_reruns_authorized",
               "blockers": ["review_per_lab_impact_before_downstream_reruns", "pending_assay_approvals_and_metadata",
                            "serology_temporal_construction_unverified"] + ([] if previous_path else ["previous_row_level_snapshot_missing"]),
               "privacy": "aggregate_only_free_text_redacted_no_patient_records", "inputs_unchanged": True}
    for path, digest in hashes.items():
        if _sha256(path) != digest:
            raise RuntimeError("Input changed during profiling; refusing to publish outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    for table, path in [(profile, output), (audit, output.parent / EXPECTED_AUDIT_NAME),
                        (candidates, output.parent / GRAPH_CANDIDATES_NAME), (serology, output.parent / SEROLOGY_NAME),
                        (text_code_audit(master), output.parent / TEXT_AUDIT_NAME), (impact, output.parent / IMPACT_NAME)]:
        table.to_csv(path, index=False)
    (output.parent / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Laboratory metadata mode: {mode}")
    print(f"Lab stems: {len(profile)}; serology fields (separate): {len(serology)}")
    print(f"Recommended use: {summary['new_recommended_use_counts']}")
    print(f"Schema family counts match investigator reference: {summary['schema_matches_reference']}")
    print(f"Downstream release: HOLD; review {IMPACT_NAME} and {EXPECTED_AUDIT_NAME}")
    return profile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    context = parser.add_mutually_exclusive_group()
    context.add_argument("--context", type=Path)
    context.add_argument("--no-context", action="store_true")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--previous-profile", type=Path, help="Archived real profile for row-level downstream drift; never overwritten")
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(parse_args())

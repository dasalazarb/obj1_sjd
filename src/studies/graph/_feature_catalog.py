"""Exact-column, aggregate-only Step-10 feature contract shared by Graph 01/02.

Discovery is exhaustive. Technical safety, coverage, policy and clinical review
are independent decisions; the lab registry is never a discovery whitelist.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd

CONTRACT_VERSION = "graph_features_v1"
FAMILIES = ("LAB", "SERO", "PRO", "EXT", "CLINICAL", "METADATA")
POLICIES = ("phenotype_independent", "clinical_inclusive", "legacy_with_overlap")
MISSING = {"", "nan", "none", "<na>", "unknown", "not reported", "not_reported",
           "n/a", "na", "indeterminate", "uninterpretable", "not tested", "not_tested"}
BINARY = {"positive": 1., "pos": 1., "negative": 0., "neg": 0.,
          "reactive": 1., "nonreactive": 0., "non-reactive": 0.,
          "detected": 1., "not detected": 0., "present": 1., "absent": 0.,
          "true": 1., "false": 0., "yes": 1., "no": 0.}
OUTCOMES = {"pop__status", "essdai__total", "esspri__total"}
# These PROs are discovery inputs. Their follow-up models are descriptive,
# conditional trajectories, never independent baseline validation outcomes.
RESERVED_COMPONENTS = {"esspri__dryness", "esspri__fatigue", "esspri__pain"}
CONTEXT_SUFFIXES = ("__unit", "__reference_status", "__measurement_date", "__days_from_anchor",
                    "__n_measurements", "__conflict", "__selection_status", "__episode_status")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    # IDs/order contribute to the hash but are never emitted in audits.
    payload = json.dumps([(str(c), str(t)) for c, t in frame.dtypes.items()]).encode()
    return hashlib.sha256(payload + pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def lab_stem(column: str) -> str | None:
    if column.startswith("lab__"):
        for suffix in ("__value", "__text"):
            if column.endswith(suffix):
                return column[:-len(suffix)]
    return None


def namespace(column: str) -> str:
    return column.split("__", 1)[0] if "__" in column else "structural"


def family(column: str) -> str:
    ns = namespace(column)
    return ns.upper() if ns in {"lab", "sero", "pro", "ext"} else (
        "CLINICAL" if ns in {"essdai", "esspri", "ovl", "demo"} else "METADATA")


def domain(column: str) -> str:
    ns = namespace(column)
    tail = column.partition("__")[2]
    if ns == "ext":
        for prefix in ("ocular", "sgus", "sicca", "sjddi", "salivary", "biopsy", "lacrimal"):
            if tail.startswith(prefix + "_"):
                return prefix
    return ns


def clean(series: pd.Series) -> pd.Series:
    return series.map(lambda v: None if pd.isna(v) or str(v).strip().lower() in MISSING
                      else re.sub(r"\s+", " ", str(v).strip().lower()))


def encode(series: pd.Series, column: str, spec: dict | None = None
           ) -> tuple[dict[str, pd.Series], str, str]:
    """Never interpret censoring or free text as a point value or a negative."""
    spec = spec or {}
    if column == "ovl__status":
        series = series.where(~clean(series).eq("unclassifiable"))
        spec = {**spec, "encoding": "nominal", "categories": ["overlap", "glandular_only", "extraglandular_only", "neither"]}
    tokens = clean(series)
    observed = set(tokens.dropna())
    if not observed:
        return {}, "none", "no_observed_baseline_data"
    if spec.get("ordinal_values"):
        order = {str(k).strip().lower(): float(v) for k, v in spec["ordinal_values"].items()}
        if not observed <= order.keys():
            return {}, "ordinal", "unrecognized_ordinal_codes"
        return {column + "__ordinal": tokens.map(order).astype(float)}, "ordinal", ""
    if observed <= BINARY.keys():
        return {column + "__binary": tokens.map(BINARY).astype(float)}, "binary", ""
    if column == "demo__sex" or spec.get("encoding") == "nominal":
        allowed = {"female", "male", "f", "m"} if column == "demo__sex" else set(spec.get("categories", []))
        if not observed <= allowed:
            return {}, "one_hot", "unverified_nominal_categories"
        result = {column + "__cat_" + re.sub(r"[^a-z0-9]+", "_", category):
                  tokens.eq(category).astype(float).where(tokens.notna()) for category in sorted(observed)}
        if len(result) != len(observed):
            return {}, "one_hot", "category_name_collision"
        return result, "one_hot_missing_as_nan_median_imputation", ""
    # LAB text is a separate categorical result, never a numeric rescue path.
    if column.startswith("lab__") and column.endswith("__text"):
        return {}, "unencoded_text", "unverified_text_encoding"
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    numeric = numeric.where(np.isfinite(numeric))
    if (tokens.notna() & numeric.isna()).any():
        return {}, "numeric", "non_point_or_unparseable_values"
    return {column: numeric}, "binary_numeric" if set(numeric.dropna()) <= {0., 1.} else "numeric", ""


def policy_reason(column: str, policy: str, spec: dict) -> tuple[str, str]:
    ns = namespace(column)
    tail = column.partition("__")[2]
    if column in OUTCOMES:
        return "annotation_or_outcome", "not_predictor"
    if ns == "sero":
        if column.endswith("__patient_consensus_value"):
            return "future_patient_consensus", "future_information"
        validation = spec.get("temporal_validation", {})
        if not (validation.get("no_future_data") is True and validation.get("no_criteria_circularity") is True
                and validation.get("decision_id") and validation.get("knowledge_date_verified") is True):
            return "exclude_temporal_not_verified", "unverified"
        return "", "verified_asof_baseline"
    if ns in {"structural", "ids", "spine", "pop"} or family(column) == "METADATA":
        return "administrative_metadata", "not_predictor"
    if column.endswith(CONTEXT_SUFFIXES) or (ns != "lab" and tail.endswith(("_date", "_source", "_available", "_evaluable", "_text", "_qc"))):
        return "context_or_provenance", "not_predictor"
    if any(part in column.lower() for part in ("ever_positive", "through_episode", "future", "next_", "delta_", "post_baseline", "age_at_visit")):
        return "temporal_leakage_risk", "unverified"
    if ns == "demo" and column not in {"demo__age_at_baseline", "demo__sex"}:
        return "administrative_demographics", "not_predictor"
    if ns in {"essdai", "ovl"} and policy == "phenotype_independent":
        return "definition_related_excluded_by_policy", "baseline"
    if ns == "essdai" and policy == "legacy_with_overlap":
        return "definition_related_excluded_by_policy", "baseline"
    if column in RESERVED_COMPONENTS and policy != "clinical_inclusive":
        return "reserved_outcome_component", "baseline"
    return "", "baseline"


def build_catalog(baseline: pd.DataFrame, candidates: pd.DataFrame, minimum_coverage: float = .70,
                  *, feature_policy: str = "phenotype_independent", analysis_mode: str = "exploratory",
                  metadata: dict | None = None, source_hash: str = ""
                  ) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, str], pd.DataFrame]:
    if feature_policy not in POLICIES or analysis_mode not in {"exploratory", "clinical_release"}:
        raise ValueError("Unknown feature policy or analysis mode")
    if not 0 <= minimum_coverage <= 1 or not baseline.columns.is_unique:
        raise ValueError("Invalid coverage or duplicate source columns")
    if "patient_id" in baseline and baseline.patient_id.duplicated().any():
        raise ValueError("Baseline requires one row per patient")
    metadata = metadata or {}
    candidate_rows = candidates.fillna("").to_dict("records")
    by_source = {r.get("source_column"): r for r in candidate_rows if r.get("source_column")}
    rows, values, lab_names = [], {}, {}
    for column in baseline:
        ns, fam, stem = namespace(column), family(column), lab_stem(column)
        raw, spec = baseline[column], dict(metadata.get(column, {}))
        if column == "ovl__status" and "ovl__evaluable" in baseline:
            raw = raw.where(baseline["ovl__evaluable"].eq(True).fillna(False))
        candidate = by_source.get(column, {})
        if candidate.get("ordinal_values_json"):
            spec["ordinal_values"] = json.loads(candidate["ordinal_values_json"])
        if column == "ovl__status" and feature_policy == "legacy_with_overlap":
            spec["ordinal_values"] = {"overlap": 1, "glandular_only": 0, "extraglandular_only": 0, "neither": 0}
        reason, temporal = policy_reason(column, feature_policy, spec)
        if fam == "LAB" and stem is None:
            reason = "lab_context_not_result"
        encoded, encoding, parse_reason = encode(raw, column, spec)
        if stem and candidate.get("conflict_assessment_status") == "conflict_detected":
            reason = "upstream_result_conflict"
        if stem and candidate.get("n_units", 0) and float(candidate["n_units"]) > 1:
            reason = "multiple_units_not_harmonized"
        if stem and (stem + "__unit") in baseline and baseline[stem + "__unit"].dropna().nunique() > 1:
            reason = "multiple_units_not_harmonized"
        approval = candidate.get("clinical_approval_status", "pending") or "pending"
        clinical_reason = ""
        if (analysis_mode == "clinical_release" and stem
                and candidate.get("longitudinal_graph_candidate_status") not in {"include_numeric", "include_categorical"}):
            clinical_reason = "clinical_longitudinal_support_not_established"
        if analysis_mode == "clinical_release" and stem and approval != "approved":
            clinical_reason = "clinical_approval_not_established"
        reason = reason or parse_reason
        parsed = next(iter(encoded.values()), pd.Series(np.nan, index=baseline.index))
        n, unique = int(parsed.notna().sum()), int(parsed.nunique())
        if not reason and unique < 2:
            reason = "no_baseline_variability"
        technical = not reason
        coverage = n / len(baseline) if len(baseline) else 0.
        coverage_pass = coverage >= minimum_coverage
        if technical and not coverage_pass:
            reason = "below_minimum_baseline_coverage"
        reason = reason or clinical_reason
        included = technical and coverage_pass and not clinical_reason
        features = encoded if included else {column: parsed}
        for feature, series in features.items():
            if included:
                values[feature] = series
                if stem:
                    lab_names.setdefault(stem.removeprefix("lab__"), feature)
            observed = int(clean(raw).notna().sum())
            rows.append({"feature": feature, "family": fam.lower(), "source_namespace": ns,
                         "clinical_domain": domain(column), "source_column": column,
                         "source": "pharma_00_graph_lab_candidates" if stem else "integrated_master",
                         "lab": stem or "", "available": True, "data_type": str(raw.dtype),
                         "representation_type": encoding, "encoding": encoding,
                         "observed_n_baseline": observed, "baseline_n": len(baseline),
                         "baseline_fraction": coverage, "parseable": bool(n),
                         "n_parseable_baseline": n, "n_unique": unique,
                         "n_nonmissing_baseline": n, "n_baseline_nonmissing": n,
                         "pct_nonmissing_baseline": coverage, "pct_baseline_nonmissing": coverage,
                         "n_unique_baseline": unique, "technical_eligibility": technical,
                         "technical_eligible": technical, "technical_status": "eligible" if technical else "excluded",
                         "technical_exclusion_reason": "" if technical else reason,
                         "clinical_review_status": approval, "clinical_approval_status": approval,
                         "temporal_status": temporal, "feature_policy": feature_policy,
                         "analysis_mode": analysis_mode, "source_hash": source_hash,
                         "feature_contract_version": CONTRACT_VERSION,
                         "unit_review_status": candidate.get("unit_assessment_status", "not_assessed") if stem else "not_applicable",
                         "coverage_pass": coverage_pass, "redundancy_pass": True,
                         "included": included, "included_pca": included, "included_in_mapper": included,
                         "exclusion_reason": reason,
                         "graph_candidate_status": "include_numeric" if included and "numeric" in encoding else "include_categorical" if included else "exclude",
                         "graph_encoding_hint": encoding})
    manifest = pd.DataFrame(rows)
    # The same assay's value/text are alternate representations, not independent
    # measurements. Prefer greater parseable coverage, numeric on ties. Keep
    # every excluded representation in the manifest with an explicit reason.
    for stem, part in manifest.loc[manifest.included & manifest.lab.ne("")].groupby("lab"):
        if part.source_column.nunique() > 1:
            preferred = sorted(part.source_column.unique(), key=lambda c: (
                -float(part.loc[part.source_column.eq(c), "pct_nonmissing_baseline"].max()),
                not c.endswith("__value"), c))[0]
            rejected = part.index[part.source_column.ne(preferred)]
            for feature in manifest.loc[rejected, "feature"]:
                values.pop(feature, None)
            manifest.loc[rejected, ["included", "included_pca", "included_in_mapper", "redundancy_pass"]] = False
            manifest.loc[rejected, "exclusion_reason"] = "alternate_assay_representation_prefer:" + preferred
            lab_names[stem.removeprefix("lab__")] = manifest.loc[part.index[part.source_column.eq(preferred)], "feature"].iloc[0]
    labs = manifest.loc[manifest.source_namespace.eq("lab") & manifest.lab.ne("")].copy()
    labs["included_in_graph"] = labs.included
    labs["source_column_found"] = labs.available
    labs["baseline_nonmissing"] = labs.n_nonmissing_baseline
    labs["baseline_coverage"] = labs.pct_nonmissing_baseline
    return manifest, values, lab_names, labs


def namespace_audit(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"source_namespace": ns, "n_columns": sum(namespace(c) == ns for c in frame)}
                         for ns in sorted({namespace(c) for c in frame})])


def family_flow(manifest: pd.DataFrame, selected=None, *, scenario="canonical") -> pd.DataFrame:
    selected = set(manifest.loc[manifest.included, "feature"]) if selected is None else set(selected)
    rows = []
    for fam in FAMILIES:
        part = manifest.loc[manifest.family.eq(fam.lower())]
        count = int(part.feature.isin(selected).sum())
        rows.append({"scenario": scenario, "family": fam, "requested": part.source_column.nunique(),
                     "present": part.loc[part.available, "source_column"].nunique(),
                     "parseable": part.loc[part.parseable, "source_column"].nunique(),
                     "eligible": part.loc[part.technical_eligible, "source_column"].nunique(),
                     "selected": count, "pca": count,
                     "condition": "selected" if count else "EMPTY_AFTER_FILTERS"})
    return pd.DataFrame(rows)


def redundancy_audit(manifest: pd.DataFrame, values: dict[str, pd.Series], threshold=.95) -> pd.DataFrame:
    """Aggregate relationship audit; canonical PCA retains derived clinical sides.

    S2–S4 apply the existing Spearman pruning; no assay equivalence is inferred
    from a correlation. The alternate representation decision is already traced.
    """
    rows = []
    for row in manifest.loc[~manifest.redundancy_pass].itertuples():
        rows.append({"feature": row.feature, "related_feature": row.exclusion_reason.partition(":")[2],
            "reason": "alternate_representation_of_same_exact_assay", "spearman_rho": np.nan,
            "canonical_action": "prefer_other_representation"})
    matrix = pd.DataFrame(values)
    corr = matrix.corr(method="spearman")
    for i, left in enumerate(matrix):
        for right in list(matrix)[i+1:]:
            rho = corr.at[left,right]
            ocular = any(left.startswith(prefix) and right.startswith(prefix) for prefix in (
                "ext__ocular_schirmer_", "ext__ocular_tbut_", "ext__ocular_oxford_", "ext__ocular_van_bijsterveld_"))
            if ocular or (pd.notna(rho) and abs(rho) >= threshold):
                rows.append({"feature": left, "related_feature": right,
                    "reason": "ocular_derived_and_lateral_comparison" if ocular else "high_absolute_spearman_not_clinical_equivalence",
                    "spearman_rho": rho, "canonical_action": "retain_compare_S0_vs_S2"})
    return pd.DataFrame(rows, columns=["feature", "related_feature", "reason", "spearman_rho", "canonical_action"])


def domain_flow(manifest: pd.DataFrame, selected, scenario="canonical") -> pd.DataFrame:
    rows=[]
    for (fam, ns, clinical_domain), part in manifest.groupby(["family", "source_namespace", "clinical_domain"], sort=True):
        rows.append({"scenario":scenario,"family":fam.upper(),"source_namespace":ns,"clinical_domain":clinical_domain,
            "requested":part.source_column.nunique(),"present":part.loc[part.available,"source_column"].nunique(),
            "parseable":part.loc[part.parseable,"source_column"].nunique(),
            "eligible":part.loc[part.technical_eligible,"source_column"].nunique(),
            "selected":int(part.feature.isin(selected).sum()),"pca":int(part.feature.isin(selected).sum())})
    return pd.DataFrame(rows)

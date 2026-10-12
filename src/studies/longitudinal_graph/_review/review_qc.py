"""Companion audits with explicit missing evidence and patient denominators."""
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from src.studies.longitudinal_graph import validation as qc


def recover_administration(keys, metadata, source):
    """Validated 1:1 visit join; metadata has priority, source fills missing cells.

    Protocol and era are never guessed from dates or clinical values. Conflicts
    are retained as blocking provenance evidence, rather than silently resolved.
    """
    from src.studies._shared import CONCEPTS
    metadata = metadata.copy()
    source = source.copy() if source is not None else None
    selectors = {}
    for label, frame in (("metadata", metadata), ("integrated", source)):
        if frame is not None and "protocol" not in frame:
            column = next((c for c in CONCEPTS["protocol"] if c in frame), None)
            if column is not None:
                frame["protocol"] = frame[column]
                selectors[label] = column
    joined = qc.align_source(keys, metadata, [c for c in metadata if c not in qc.KEYS])
    rows = []
    for variable in ("protocol", "demo__protocol", "era", "visit_type", "spine__interval_name"):
        supplied = variable in joined
        available = source is not None and variable in source
        conflict = 0
        if available:
            recovered = qc.align_source(keys, source, [variable])[variable]
            if supplied:
                a, b = joined[variable], recovered
                conflict = int((a.notna() & b.notna() & a.astype(str).ne(b.astype(str))).sum())
                joined[variable] = a.where(a.notna(), b)
            else:
                joined[variable] = recovered
        n = int(joined[variable].notna().sum()) if variable in joined else 0
        rows.append({"variable": variable, "status": "conflicting_sources" if conflict else "available" if n else "not_tested_missing_source",
                     "n_observed": n, "n_conflicting_visits": conflict,
                     "source": "script01_metadata_then_integrated_1_to_1" if available else "script01_metadata" if supplied else "unavailable",
                     "protocol_alias_selectors": str(selectors) if variable == "protocol" else ""})
    return joined, pd.DataFrame(rows)


def extended_administration(membership, metadata, embedding, config):
    """Keep categorical V; continuous covariates use eta-squared without bins.

    The associated p value exchanges full nominal label vectors among patients
    with equal valid series lengths. This is an administrative association test,
    not a substitute for the unimplemented distance-based T3 adaptation.
    """
    categorical = metadata.drop(columns=[c for c in ("clinical_anchor_date", "calendar_year", "state_coverage_fraction", "n_visits_per_patient") if c in metadata])
    assoc, availability, strata, lens = qc.administrative_audit(membership, categorical, embedding, config)
    # The historical helper creates visits-per-patient internally; do not use its bins.
    continuous = ["calendar_year", "state_coverage_fraction", "n_visits_per_patient"]
    assoc = assoc.loc[~assoc.administrative_variable.isin(continuous)].copy()
    availability = availability.loc[~availability.variable.isin(continuous)].copy()
    strata = strata.loc[~strata.variable.isin(continuous)].copy()
    lens = lens.loc[~lens.variable.isin(continuous)].copy()
    joined = qc.align_source(membership, metadata, [c for c in metadata if c not in membership])
    joined["n_visits_per_patient"] = joined.patient_id.map(joined.groupby("patient_id").size())
    if "clinical_anchor_date" in joined:
        joined["calendar_year"] = pd.to_datetime(joined.clinical_anchor_date).dt.year
    rows, available, lens_rows = [], [], []
    rng = np.random.default_rng(config["random_seed"])
    permutations = int(config.get("mapper_validation", {}).get("admin_permutations", 199))
    def effect(x, labels):
        total = np.square(x - np.mean(x)).sum()
        if not total or len(np.unique(labels)) < 2:
            return np.nan
        return sum(np.sum(labels == label) * (np.mean(x[labels == label]) - np.mean(x)) ** 2 for label in np.unique(labels)) / total
    for variable in continuous:
        if variable not in joined:
            available.append({"variable": variable, "status": "not_tested_missing_variable", "n_observed": 0})
            continue
        valid = joined.loc[joined[variable].notna() & joined.hard_macrostate.notna()].copy()
        available.append({"variable": variable, "status": "completed" if len(valid) else "not_tested_no_valid_assignments", "n_observed": len(valid)})
        if not len(valid):
            continue
        valid = valid.sort_values(qc.KEYS).reset_index(drop=True)
        x = valid[variable].astype(float).to_numpy()
        labels = valid.hard_macrostate.astype(str).to_numpy()
        value = effect(x, labels)
        blocks = {}
        for _, group in valid.groupby("patient_id", sort=True):
            blocks.setdefault(len(group), []).append(group.index.to_numpy())
        null = []
        if np.isfinite(value) and any(len(group) > 1 for group in blocks.values()):
            for _ in range(permutations):
                permuted = labels.copy()
                for group in blocks.values():
                    for target, source in zip(group, rng.permutation(len(group))):
                        permuted[target] = labels[group[source]]
                null.append(effect(x, permuted))
        rows.append({"administrative_variable": variable, "cramers_v": np.nan, "eta_squared": value,
                     "n_visits": len(valid), "n_patients": valid.patient_id.nunique(),
                     "administrative_confounding_flag": False, "flag_threshold": np.nan,
                     "patient_permutation_p": (1 + sum(v >= value for v in null)) / (1 + len(null)) if null else None,
                     "n_permutations": len(null), "inference_status": "patient_vector_permutation_same_valid_series_length" if null else "not_tested_insufficient_exchangeability_or_variance",
                     "visit_p_value_used": False, "decision_status": "requires_review_no_posthoc_effect_cutoff", "binning": "none_continuous_covariate"})
        for i in range(2):
            full = joined.loc[joined[variable].notna()]
            from scipy.stats import spearmanr
            rho = spearmanr(full[variable], embedding[full.index, i]).statistic if full[variable].nunique() > 1 else None
            lens_rows.append({"variable": variable, "component": f"PC{i+1}", "category": "all", "spearman_descriptive": rho,
                              "n_visits": len(full), "inference_status": "descriptive_correlated_visits"})
    return (pd.concat([assoc, pd.DataFrame(rows)], ignore_index=True), pd.concat([availability, pd.DataFrame(available)], ignore_index=True),
            strata, pd.concat([lens, pd.DataFrame(lens_rows)], ignore_index=True))


def clinical_characterization(scenario, raw, metadata, features):
    """Descriptive observed values at the earliest visit per patient × community.

    Communities overlap, so patient denominators must never be added. Missing
    values at that first visit are not replaced by a later convenient measurement.
    """
    if raw is None:
        return pd.DataFrame([{"status": "not_tested_missing_original_values"}])
    values = qc.align_source(scenario["visit_membership"][qc.KEYS], raw, [f for f in features if f in raw])
    values = qc.align_source(values, metadata, ["clinical_anchor_date"])
    rows = []
    for macro in scenario["macrostate_table"].macrostate_id:
        indices = set().union(*(members for node, members in scenario["graph"]["nodes"].items() if scenario["node_to_macrostate"].get(node) == macro))
        group = values.iloc[sorted(indices)].sort_values("clinical_anchor_date").drop_duplicates("patient_id")
        for feature in features:
            if feature not in group or not pd.api.types.is_numeric_dtype(group[feature]):
                continue
            observed = pd.to_numeric(group[feature], errors="coerce").astype(float)
            observed = observed[np.isfinite(observed)]
            rows.append({"macrostate_id": macro, "feature": feature, "n_patients_membership": len(group),
                         "n_patients_observed": len(observed), "median": observed.median(), "q1": observed.quantile(.25), "q3": observed.quantile(.75),
                         "aggregation_rule": "earliest_visit_per_patient_per_community; not_exclusive_between_communities",
                         "role": "defining_descriptive" if feature in scenario["features"] else "candidate_anchor_descriptive_not_validated",
                         "independent_validation": False, "status": "observed_values_descriptive"})
    return pd.DataFrame(rows)


def raw_feature_qc(raw, features, source_status, feature_manifest):
    manifest = feature_manifest.set_index("feature") if not feature_manifest.empty else pd.DataFrame()
    rows = []
    for feature in features:
        available = raw is not None and feature in raw
        values = pd.to_numeric(raw[feature], errors="coerce").astype("float64") if available else pd.Series(dtype=float)
        finite = values[np.isfinite(values)]
        info = manifest.loc[feature].to_dict() if feature in manifest.index else {}
        q1, q3 = finite.quantile(.25), finite.quantile(.75)
        iqr = q3 - q1
        rows.append({"feature": feature, "source": source_status, "status": "completed" if available else "not_tested_missing_source",
                     "n_visits": len(raw) if raw is not None else None, "n_finite": len(finite) if available else None,
                     "n_missing": int(raw[feature].isna().sum()) if available else None,
                     "n_invalid": int((raw[feature].notna() & ~np.isfinite(values)).sum()) if available else None,
                     "n_zero": int(finite.eq(0).sum()) if available else None,
                     "n_negative": int(finite.lt(0).sum()) if available else None,
                     "n_observed_patients": raw.loc[np.isfinite(values), "patient_id"].nunique() if available else None,
                     "median": finite.median(), "q1": q1, "q3": q3, "iqr": iqr,
                     "p01": finite.quantile(.01), "p99": finite.quantile(.99), "min": finite.min(), "max": finite.max(),
                     "n_outliers_3iqr": int(((finite < q1 - 3 * iqr) | (finite > q3 + 3 * iqr)).sum()) if available else None,
                     "unit": info.get("unit", "unknown"), "temporal_scope": info.get("temporal_scope", "unknown"),
                     "observation_provenance": "unknown_requires_upstream_observation_history",
                     "carried_forward_status": "unknown", "structurally_unavailable_status": "unknown",
                     "effective_observation_date": "unknown", "missingness_mechanism": "unknown_not_assumed_zero"})
    return pd.DataFrame(rows)


def coverage_partition(membership):
    out = membership.copy()
    hard = out.hard_macrostate.notna()
    tie = out.macrostate_membership_tie.astype(bool)
    covered = out.mapper_graph_covered.astype(bool)
    if (hard & tie).any() or ((hard | tie) & ~covered).any():
        raise ValueError("Invalid coverage partition")
    out["coverage_category"] = np.select([hard, tie, covered], ["hard_assigned", "tie", "covered_unsupported_macrostate_or_node"], default="uncovered")
    counts = out.coverage_category.value_counts()
    if counts.sum() != len(out):
        raise AssertionError("Visit coverage is not an exclusive partition")
    priority = {"hard_assigned": 0, "tie": 1, "covered_unsupported_macrostate_or_node": 2, "uncovered": 3}
    strongest = out.groupby("patient_id").coverage_category.agg(lambda x: min(x, key=priority.get))
    out["patient_coverage_category"] = out.patient_id.map(strongest)
    return out


def validate_intervals(intervals, metadata):
    required = ["patient_id", "from_clinical_episode_id", "to_clinical_episode_id", "interval_days", "interval_years"]
    if any(c not in intervals for c in required):
        raise ValueError("Script01 interval schema is incomplete")
    if intervals[required[:3]].isna().any().any() or intervals[required[:3]].duplicated().any():
        raise ValueError("Duplicate or null interval keys")
    qc.validate_keys(metadata, "interval metadata")
    if "clinical_anchor_date" not in metadata:
        raise ValueError("Visit dates unavailable: chronology cannot be validated")
    dated = metadata[qc.KEYS + ["clinical_anchor_date"]].copy()
    dated["clinical_anchor_date"] = pd.to_datetime(dated.clinical_anchor_date, errors="raise")
    if dated.clinical_anchor_date.isna().any():
        raise ValueError("Null visit dates")
    out = intervals.copy()
    for side in ("from", "to"):
        mapping = dated.rename(columns={"clinical_episode_id": f"{side}_clinical_episode_id", "clinical_anchor_date": f"{side}_date"})
        out = out.merge(mapping, on=["patient_id", f"{side}_clinical_episode_id"], how="left", validate="many_to_one")
        if out[f"{side}_date"].isna().any():
            raise ValueError("Interval endpoint missing from the same patient's visits")
    delta = (out.to_date - out.from_date).dt.days
    if (delta <= 0).any() or not np.allclose(delta, out.interval_days) or not np.allclose(delta / 365.25, out.interval_years):
        raise ValueError("Interval dates, order or delta_t disagree with Script01")
    # Require adjacency of original clinical visits; no skipped uncovered visits.
    adjacent = set()
    order = dated.sort_values(["patient_id", "clinical_anchor_date", "clinical_episode_id"])
    for patient, group in order.groupby("patient_id", sort=False):
        ids = group.clinical_episode_id.tolist()
        adjacent.update((patient, a, b) for a, b in zip(ids[:-1], ids[1:]))
    if any(tuple(row) not in adjacent for row in intervals[required[:3]].itertuples(index=False, name=None)):
        raise ValueError("Interval skips a clinical visit")
    return out


def interval_contract(intervals, membership, metadata, run_id, scenario, maximum_years=None):
    out = validate_intervals(intervals, metadata)
    qc.validate_keys(membership, "interval membership")
    columns = [c for c in membership if c not in qc.KEYS]
    for side in ("from", "to"):
        source = membership.rename(columns={"clinical_episode_id": f"{side}_clinical_episode_id",
                                            **{c: f"{side}__{c}" for c in columns}})
        out = out.merge(source, on=["patient_id", f"{side}_clinical_episode_id"], how="left", validate="many_to_one")
    for kind, field in (("graph", "mapper_graph_covered"), ("supported", "mapper_covered")):
        out[f"both_endpoints_{kind}_covered"] = out[f"from__{field}"].eq(True) & out[f"to__{field}"].eq(True)
    out["both_endpoints_hard_assigned"] = out["from__hard_macrostate"].notna() & out["to__hard_macrostate"].notna()
    out["temporal_window_eligible"] = True if maximum_years is None else out.interval_years.le(maximum_years)
    out["exclusion_reason"] = np.select([~out.temporal_window_eligible, ~out.both_endpoints_graph_covered, ~out.both_endpoints_supported_covered, ~out.both_endpoints_hard_assigned],
        ["maximum_interval_exceeded", "endpoint_uncovered", "endpoint_outside_supported_macrostate", "endpoint_tied"], default="hard_and_soft_eligible")
    out["source_run_id"] = run_id
    out["scenario"] = scenario
    out["schema_version"] = "longitudinal_mapper_v2"
    out["inference_authorized"] = False
    return out


def coverage_comparison(raw, partition, seed, replicates):
    """Visit-level descriptive SMD with confidence intervals resampling patients.

    Observed values only; no imputed clinical prevalence or independent-visit p.
    """
    if raw is None:
        return pd.DataFrame([{"status": "not_tested_missing_original_values"}])
    joined = qc.align_source(partition[qc.KEYS + ["mapper_graph_covered"]], raw, [c for c in raw if c not in qc.KEYS])
    features = [c for c in raw if c not in qc.KEYS and pd.api.types.is_numeric_dtype(raw[c])]
    rng = np.random.default_rng(seed)
    patient_rows = [g.index.to_numpy() for _, g in joined.groupby("patient_id", sort=True)]
    draws = [np.concatenate([patient_rows[i] for i in rng.integers(0, len(patient_rows), len(patient_rows))]) for _ in range(replicates)]
    def smd(group, feature):
        a = pd.to_numeric(group.loc[group.mapper_graph_covered, feature], errors="coerce").astype(float)
        b = pd.to_numeric(group.loc[~group.mapper_graph_covered, feature], errors="coerce").astype(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return (a.mean() - b.mean()) / pooled if len(a) > 1 and len(b) > 1 and pooled > 0 else np.nan
    rows = []
    for feature in features:
        bootstrap = np.array([smd(joined.iloc[draw], feature) for draw in draws])
        bootstrap = bootstrap[np.isfinite(bootstrap)]
        row = {"feature": feature, "standardized_mean_difference": smd(joined, feature),
               "ci_low": np.quantile(bootstrap, .025) if len(bootstrap) else None,
               "ci_high": np.quantile(bootstrap, .975) if len(bootstrap) else None,
               "n_bootstrap_valid": len(bootstrap), "resampling_unit": "patient_all_visits",
               "estimand": "observed_visit_distribution_graph_covered_vs_uncovered",
               "status": "descriptive_requires_selection_review"}
        for label, mask in (("covered", joined.mapper_graph_covered), ("uncovered", ~joined.mapper_graph_covered)):
            finite = np.isfinite(pd.to_numeric(joined[feature], errors="coerce").astype(float)) & mask
            row[f"n_{label}_visits_observed"] = int(finite.sum())
            row[f"n_{label}_patients_observed"] = int(joined.loc[finite, "patient_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def feature_roles(features, source, config):
    review = config.get("methodology_review", {})
    anchors = review.get("validation_anchors", [])
    rules = {row["feature"]: row for row in anchors}
    rows = []
    for feature in dict.fromkeys(list(features) + list(rules)):
        rule = rules.get(feature, {})
        derived = rule.get("derived_from", [])
        used = feature in features
        circular = used or bool(set(derived) & set(features))
        available = source is not None and feature in source and source[feature].notna().any()
        # A future outcome requires its own timing/leakage plan; string mitigation alone is insufficient.
        eligibility = "descriptive_only_circular" if circular else "eligible_for_prespecified_test" if (
            available and rule.get("approved_by") and rule.get("independence_review") == "passed" and
            rule.get("temporal_scope") in {"episode_resolved", "baseline_static", "patient_static"}) else "not_tested_requires_independence_availability_and_timing_review"
        rows.append({"feature": feature, "construct": rule.get("construct", "unknown"), "role_in_embedding": used,
                     "role_in_lens": used, "role_in_clustering": used,
                     "candidate_validation_anchor": bool(rule), "candidate_outcome": bool(rule.get("outcome", False)),
                     "derived_from": "|".join(derived), "same_visit_information": rule.get("same_visit_information", "unknown"),
                     "longitudinal_availability": "available" if available else "not_tested_missing_source",
                     "circularity_flag": circular, "mitigation": rule.get("mitigation", "none_recorded"),
                     "approved_by": rule.get("approved_by"), "anchor_eligibility": eligibility,
                     "validation_status": "not_run_no_independent_anchor_test_executed"})
    return pd.DataFrame(rows)


def decision_gates(reference, graph, admin_availability, bootstrap, run_id, admin_associations=None):
    """No scientific auto-approval; missing T3, anchors and branches stay blockers."""
    rows = []
    def add(gate, status, reason, evidence):
        rows.append({"gate": gate, "status": status, "reason": reason, "evidence": evidence,
                     "run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                     "criteria_applied": reason, "responsible": "pipeline_QC_not_researcher_approval"})
    add("G0", "passed", "frozen_reference_byte_integrity", reference["reference_manifest_sha256"])
    add("G1", "inconclusive", "units_and_effective_observation_dates_require_upstream_review", "visit_keys_and_interval_chronology_checked_when_available")
    add("G2", graph["status"], "full_and_supported_degree_edge_component_reconciliation", graph["views"])
    add("G3", "inconclusive", "dominance_report_requires_prespecified_clinical_review", "squared_loadings_are_not_total_dataset_variance")
    add("G4", "blocked", "branch_definition_not_approved; H1_not_established", bootstrap)
    missing = admin_availability.loc[admin_availability.status.ne("completed"), "variable"].tolist()
    flagged = admin_associations is not None and admin_associations.administrative_confounding_flag.any()
    add("G5", "failed" if flagged else "inconclusive", "administrative_association_flag_requires_resolution" if flagged else
        "T3_patient_unit_adaptation_not_validated; missing_controls_are_not_negative", {"missing": missing, "association_flag": bool(flagged)})
    add("G6", "not_tested", "independent_anchor_tests_and_proxy_circularity_review_pending", "feature_role_matrix")
    add("G7", "inconclusive", "selection_bias_and_hard_soft_sensitivity_require_review", "interval_contract_and_denominators")
    add("G8", "not_tested", "held_out_projection_not_run", "bootstrap_is_in_sample")
    add("G9", "blocked", "no_recorded_researcher_selection_approval", "no_scenario_selected")
    return {"status": "blocked", "gates": rows, "selected_scenario": None, "approval_status": "not_recorded",
            "temporal_flow_executed": False, "H1_status": "not_established_no_defined_branches"}

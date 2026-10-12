"""Prespecified S4 alternatives and training-only preprocessing.

Clinical whitelists, units and independent anchors require explicit evidence;
these helpers do not invent instruments or approve scientific interpretations.
"""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np
import pandas as pd
from src.studies.longitudinal_graph.validation import KEYS, weighted_quantile

HSCRP = "lab__crp_high_sensitivity__value"


def candidate_registry(features, config):
    review = config.get("methodology_review", {})
    names = [("S4-visit-comparator", "identity"), ("S4-hscrp-log-robust", "log1p"),
             ("S4-hscrp-winsor", "winsor"), ("S4-no-hscrp", "without_hscrp"),
             ("S4-patient-mass", "patient_mass"), ("S4-no-pro", "without_pro"),
             ("S4-pro-summary", "pro_summary"), ("S4-common-features", "common_features"),
             ("S4-preimputation-pruning", "observed_pruning"), ("S4-full80", "full80")]
    rows = []
    for name, change in names:
        rows.append({"name": name, "status": "planned", "parent_reference": "S3-primary",
                     "description": change, "isolated_change": change,
                     "feature_set_id": "historical_postpruning" if change not in {"observed_pruning", "pro_summary", "common_features"} else change,
                     "transformation_id": change if change in {"log1p", "winsor"} else "identity",
                     "weight_rule": "equal_patient_mass" if change == "patient_mass" else "equal_visit_mass",
                     "pca_rule": "target_80_training" if change == "full80" else "historical_cap_10",
                     "mapper_rule": "global_eps_minmax", "cover": copy.deepcopy(config["mapper"]),
                     "validation_plan_id": "patient_bootstrap_components_and_macrostates_not_branches",
                     "seed": config["random_seed"], "approval_status": "not_selected_requires_researcher",
                     "reason": "prespecified_single_change", "feature_list": list(features)})
    for row in rows:
        change = row["isolated_change"]
        reason = None
        if change in {"log1p", "winsor", "without_hscrp"} and HSCRP not in features:
            reason = "hscrp_not_retained"
        if change == "log1p" and (review.get("hscrp_unit_status") != "verified_consistent" or
                                   review.get("hscrp_prior_transformation") != "identity"):
            reason = "hscrp_units_or_prior_transformation_not_verified"
        if change == "without_hscrp":
            row["feature_list"] = [f for f in features if f != HSCRP]
        if change == "without_pro":
            row["feature_list"] = [f for f in features if not f.startswith("pro__")]
            if row["feature_list"] == features:
                reason = "no_PRO_features_retained"
        if change in {"pro_summary", "common_features"}:
            whitelist = review.get(change, {})
            selected = whitelist.get("features", [])
            if not selected or not whitelist.get("approved_by") or not whitelist.get("rationale"):
                reason = "clinical_whitelist_not_defined_and_approved"
            elif not set(selected) <= set(features):
                reason = "approved_features_not_available"
            elif change == "pro_summary" and not all(f.startswith("pro__") for f in selected):
                reason = "PRO_whitelist_contains_non_PRO_features"
            else:
                row["feature_list"] = [f for f in features if f in selected or (change == "pro_summary" and not f.startswith("pro__"))]
        if len(row["feature_list"]) < 2:
            reason = reason or "fewer_than_two_features"
        if reason:
            row.update(status="not_applicable", reason=reason)
        frozen = {k: v for k, v in row.items() if k not in {"status", "reason", "approval_status"}}
        row["definition_sha256"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    return rows


def observed_pruning(raw, features, threshold, minimum_pairs, minimum_patients):
    numeric = raw.reindex(columns=features).astype("float64").replace([np.inf, -np.inf], np.nan)
    priority = lambda f: (-numeric[f].notna().mean(), f)
    order = sorted(features, key=priority)
    retained = set(features)
    audit = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            mask = numeric[a].notna() & numeric[b].notna()
            n, patients = int(mask.sum()), int(raw.loc[mask, "patient_id"].nunique())
            testable = n >= minimum_pairs and patients >= minimum_patients
            rho = numeric.loc[mask, [a, b]].corr(method="spearman").iloc[0, 1] if testable else np.nan
            remove = bool(a in retained and b in retained and np.isfinite(rho) and abs(rho) >= threshold)
            if remove:
                retained.remove(b)
            audit.append({"feature_a": a, "feature_b": b, "n_pairwise": n, "n_pairwise_patients": patients,
                          "pairwise_available_fraction": n / max(len(raw), 1), "spearman": rho,
                          "status": "completed" if np.isfinite(rho) else "not_testable_constant" if testable else "not_testable_low_overlap",
                          "action": "remove_redundant" if remove else "retain", "removed_feature": b if remove else None,
                          "rationale": "prespecified_coverage_then_name_priority; semantic_review_pending"})
    return [f for f in features if f in retained], pd.DataFrame(audit, columns=[
        "feature_a", "feature_b", "n_pairwise", "n_pairwise_patients", "pairwise_available_fraction", "spearman", "status", "action", "removed_feature", "rationale"])


class TrainingPreprocessor:
    """Learn transformations, median imputation and pruning on the fit rows only."""
    def __init__(self, definition, config, prune):
        self.definition = definition
        self.config = config
        self.prune = prune

    def fit(self, raw, features):
        self.features_in_ = list(features)
        values = raw[self.features_in_].astype("float64").replace([np.inf, -np.inf], np.nan)
        change = self.definition["isolated_change"]
        self.bounds_ = None
        if change == "log1p":
            if values[HSCRP].lt(0).any():
                raise ValueError("negative_hscrp_observations; no_clinical_correction_assumed")
            values[HSCRP] = np.log1p(values[HSCRP])
        if change == "winsor":
            if not values[HSCRP].notna().any():
                raise ValueError("hscrp_original_observations_unavailable")
            self.bounds_ = tuple(values[HSCRP].quantile([.01, .99]))
            values[HSCRP] = values[HSCRP].clip(*self.bounds_)
        # Match the fixed Script01 median-imputation contract. Patient-mass is
        # a single fitting-rule change in robust scaling/PCA, not a second
        # imputation experiment. Each bootstrap relearns these medians on its sample.
        self.medians_ = values.median()
        usable = self.medians_.index[self.medians_.notna()].tolist()
        if change == "observed_pruning":
            review = self.config.get("methodology_review", {})
            self.features_, self.redundancy_ = observed_pruning(raw, usable,
                self.config["representation"]["redundancy_spearman_abs_threshold"],
                int(review.get("redundancy_minimum_pairs", 20)), int(review.get("redundancy_minimum_patients", 10)))
        else:
            self.features_, self.redundancy_ = self.prune(values[usable].fillna(self.medians_[usable]),
                self.config["representation"]["redundancy_spearman_abs_threshold"])
        if len(self.features_) < 2:
            raise ValueError("fewer_than_two_training_features")
        return self

    def transform(self, raw):
        values = raw[self.features_].astype("float64").replace([np.inf, -np.inf], np.nan)
        if self.definition["isolated_change"] == "log1p" and HSCRP in values:
            if values[HSCRP].lt(0).any():
                raise ValueError("negative_hscrp_projection_values")
            values[HSCRP] = np.log1p(values[HSCRP])
        if self.bounds_ is not None and HSCRP in values:
            values[HSCRP] = values[HSCRP].clip(*self.bounds_)
        return pd.concat([raw[KEYS].reset_index(drop=True), values.fillna(self.medians_).reset_index(drop=True)], axis=1)


def negative_patient_blocks(raw, features, rng):
    """Independently exchange feature trajectories among equal-length patients.

    Keys and within-trajectory order remain fixed. Unequal-length groups are not
    interpolated. Report exchangeability; a singleton stratum supplies no control.
    """
    out = raw.copy()
    groups = [(g.sort_values("clinical_anchor_date") if "clinical_anchor_date" in g else g).index.to_numpy()
              for _, g in raw.groupby("patient_id", sort=True)]
    strata = {}
    for idx in groups:
        strata.setdefault(len(idx), []).append(idx)
    for feature in features:
        for blocks in strata.values():
            for target, source in zip(blocks, rng.permutation(len(blocks))):
                out.loc[target, feature] = raw.loc[blocks[source], feature].to_numpy()
    return out, {"method": "independent_columns_patient_trajectory_permutation_within_equal_visit_counts",
                 "n_exchangeable_patients": sum(len(b) for b in strata.values() if len(b) > 1),
                 "n_fixed_singleton_patients": sum(len(b) for b in strata.values() if len(b) == 1),
                 "geometry_changed": not out[features].equals(raw[features])}

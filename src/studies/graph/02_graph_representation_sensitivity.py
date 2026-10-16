#!/usr/bin/env python3
"""Test whether baseline Mapper geometry is robust to representation choices.

Pop labels are external annotations only.  Every bootstrap replicate refits its
feature filtering, imputation, scaling/balancing, PCA, Mapper, and communities.
This analysis deliberately does not optimize Mapper or run longitudinal models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler

import common
from src.studies._shared import create_study_dirs, resolve_concept


def _load_canonical():
    path = Path(__file__).with_name("01_run_topological_phenotyping.py")
    spec = importlib.util.spec_from_file_location("graph_topological_phenotyping", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import canonical Graph helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANON = _load_canonical()
DEFAULT_LAB_GROUP_MAP = ROOT / "data" / "raw" / "graph_s4_lab_group_map.csv"
SCENARIOS = {
    "S0": {"coverage": False, "redundancy": False, "balance": False, "lab_group_balance": False},
    "S1": {"coverage": True, "redundancy": False, "balance": False, "lab_group_balance": False},
    "S2": {"coverage": True, "redundancy": True, "balance": False, "lab_group_balance": False},
    "S3": {"coverage": True, "redundancy": True, "balance": True, "lab_group_balance": False},
    "S4": {"coverage": True, "redundancy": True, "balance": True, "lab_group_balance": True},
}
FAMILY_ORDER = ("LAB", "PRO", "CLINICAL")
BALANCE_COLUMNS = ["scenario", "level", "block", "n_features", "n_patients",
                   "sigma1_before_weighting", "weight_applied", "nominal_weight",
                   "effective_weight", "status", "n_active_groups"]
REDUNDANCY_DTYPES = {"scenario": "string", "feature_kept": "string",
                     "feature_removed": "string", "spearman_rho": "float64",
                     "n_pairwise": "Int64", "coverage_kept": "float64",
                     "coverage_removed": "float64", "reason": "string"}


class MappingError(ValueError):
    """A fixed input contract error; never count this as bootstrap instability."""


class BlockNotEstimable(ValueError):
    """A representation block cannot be constructed for this patient sample."""

    def __init__(self, message: str, audit: pd.DataFrame | None = None) -> None:
        super().__init__(message)
        self.balance_audit = audit if audit is not None else pd.DataFrame(columns=BALANCE_COLUMNS)


class Representation:
    def __init__(self, raw: pd.DataFrame, families: dict[str, str],
                 embedding: np.ndarray, pca: PCA,
                 transformed_features: list[str], balance_audit: pd.DataFrame) -> None:
        self.raw = raw
        self.families = families
        self.embedding = embedding
        self.pca = pca
        self.transformed_features = transformed_features
        self.balance_audit = balance_audit


def family_name(value: str) -> str:
    return "LAB" if value == "lab" else "PRO" if value == "pro" else "CLINICAL"


canonical_lab_name = CANON.canonical_lab_name


def feature_stem(feature: str) -> str:
    """Remove encoding suffixes without truncating compound lab concepts."""
    stem = canonical_lab_name(feature).split("__cat_", 1)[0]
    for suffix in ("__value", "__binary"):
        stem = stem.removesuffix(suffix)
    return stem


def load_lab_group_map(path: Path) -> pd.DataFrame:
    """Load and strictly validate the fixed clinical metadata used by S4."""
    if not path.is_file():
        raise FileNotFoundError(f"S4 lab group map does not exist: {path}")
    mapping = pd.read_csv(path)
    required = {"lab", "s4_group", "s4_subgroup", "mapping_status"}
    missing = required - set(mapping)
    if missing:
        raise ValueError(f"S4 lab group map lacks required columns: {sorted(missing)}")
    for column in ("lab", "s4_group"):
        blank = mapping[column].isna() | mapping[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(f"S4 lab group map has blank {column} values at rows "
                             f"{(mapping.index[blank] + 2).tolist()}")
    mapping = mapping.copy()
    mapping["lab"] = mapping["lab"].astype(str).map(canonical_lab_name)
    if mapping["lab"].eq("").any():
        raise ValueError("S4 lab group map has blank lab names after namespace normalization")
    mapping["s4_group"] = mapping["s4_group"].astype(str).str.strip()
    mapping["s4_subgroup"] = mapping["s4_subgroup"].fillna("").astype(str).str.strip()
    mapping["mapping_status"] = mapping["mapping_status"].astype(str).str.strip()
    duplicated = mapping.loc[mapping["lab"].duplicated(keep=False), "lab"].unique().tolist()
    if duplicated:
        raise ValueError(f"S4 lab group map has duplicate lab values: {duplicated}")
    return mapping


def lab_group_lookup(mapping: pd.DataFrame) -> dict[str, str]:
    used = mapping.loc[mapping["mapping_status"].eq("MAPPED")]
    return used.set_index("lab")["s4_group"].to_dict()


def validate_s4_mapping(values: dict[str, pd.Series], families: dict[str, str],
                        mapping: pd.DataFrame,
                        feature_to_lab: dict[str, str],
                        included: set[str] | None = None, *,
                        require_all: bool = False, audit_path: Path | None = None) -> pd.DataFrame:
    """Audit the bootstrap universe, optionally failing before any scenario starts."""
    indexed = mapping.set_index("lab")
    rows = []
    for feature in sorted(values):
        if families[feature] != "LAB":
            continue
        lab = canonical_lab_name(feature_to_lab.get(feature, ""))
        if not lab:
            raise MappingError(
                f"LAB feature is missing canonical lab metadata in manifest: {feature}"
            )
        known = lab in indexed.index
        status = indexed.at[lab, "mapping_status"] if known else "MISSING"
        mapped = known and status == "MAPPED"
        rows.append({"feature": feature, "lab": lab,
                     "s4_group": indexed.at[lab, "s4_group"] if known else "",
                     "s4_subgroup": indexed.at[lab, "s4_subgroup"] if known else "",
                     "mapping_status": status, "eligible_for_bootstrap": True,
                     "mapping_validation_result": "PASS" if mapped else "MAPPING_ERROR",
                     "mapped": mapped,
                     "included_S4": feature in included if included is not None else False})
    audit = pd.DataFrame(rows, columns=["feature", "lab", "s4_group", "s4_subgroup",
        "mapping_status", "mapped", "eligible_for_bootstrap", "included_S4", "mapping_validation_result"])
    if audit_path is not None:
        audit.to_csv(audit_path, index=False)
    bad = audit.loc[(audit["eligible_for_bootstrap"] if require_all else audit["included_S4"])
                    & ~audit["mapped"]]
    if len(bad):
        raise MappingError(
            "MAPPING_ERROR: S4 LAB features lack a MAPPED clinical group. "
            f"Labs: {sorted(bad.lab.unique())}; features: {bad.feature.tolist()}. "
            "Complete and clinically approve graph_s4_lab_group_map.csv before running any bootstrap.")
    return audit


def coverage_filter(values: dict[str, pd.Series], families: dict[str, str],
                    config: dict) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    thresholds = {
        "LAB": float(config["feature_selection"]["lab_minimum_baseline_coverage"]),
        "PRO": float(config["feature_selection"]["pro_minimum_baseline_coverage"]),
        "CLINICAL": float(config["feature_selection"]["clinical_minimum_baseline_coverage"]),
    }
    kept, rows = {}, []
    for feature, series in values.items():
        family = families[feature]
        coverage = float(series.notna().mean())
        included = coverage >= thresholds[family]
        rows.append({"feature": feature, "family": family,
                     "coverage_before": coverage,
                     "included_after_coverage_filter": included,
                     "exclusion_reason": "" if included else
                     f"below_{family.lower()}_minimum_baseline_coverage_{thresholds[family]:.2f}"})
        if included:
            kept[feature] = series
    return kept, pd.DataFrame(rows)


def _priority(feature: str, coverage: dict[str, float]) -> tuple[float, int, str]:
    # Higher coverage wins; a count/absolute/value is more clinically primary
    # than a percent or calculated estimate; lexical order is the final tie-break.
    stem = feature_stem(feature)
    secondary = int("percent" in stem or "estimated" in stem)
    return (-coverage[feature], secondary, feature)


def reduce_redundancy(values: dict[str, pd.Series], config: dict
                      ) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Remove configured conceptual pairs, then pairwise high correlations."""
    remaining = dict(values)
    coverage = {name: float(series.notna().mean()) for name, series in values.items()}
    rows: list[dict] = []

    def remove(kept: str, removed: str, rho: float, n: int, reason: str) -> None:
        rows.append({"feature_kept": kept, "feature_removed": removed,
                     "spearman_rho": rho, "n_pairwise": n,
                     "coverage_kept": coverage[kept],
                     "coverage_removed": coverage[removed], "reason": reason})
        remaining.pop(removed, None)

    for primary, secondary in config["redundancy"].get("conceptual_pairs", []):
        first = sorted(name for name in remaining if feature_stem(name) == primary)
        second = sorted(name for name in remaining if feature_stem(name) == secondary)
        for left, right in zip(first, second):
            kept, removed = sorted((left, right), key=lambda x: _priority(x, coverage))
            pair = pd.concat([remaining[left], remaining[right]], axis=1).dropna()
            rho = float(pair.corr(method="spearman").iloc[0, 1]) if len(pair) >= 2 else np.nan
            remove(kept, removed, rho, len(pair),
                   "configured_conceptual_pair; higher_coverage_then_clinical_primacy_then_lexical")

    threshold = float(config["redundancy"]["spearman_abs_threshold"])
    # Greedy deterministic pruning makes every decision reproducible.
    ordered = sorted(remaining, key=lambda name: _priority(name, coverage))
    for i, left in enumerate(ordered):
        if left not in remaining:
            continue
        for right in ordered[i + 1:]:
            if right not in remaining:
                continue
            pair = pd.concat([remaining[left], remaining[right]], axis=1).dropna()
            if len(pair) < 2:
                continue
            rho = float(pair.corr(method="spearman").iloc[0, 1])
            if np.isfinite(rho) and abs(rho) >= threshold:
                kept, removed = sorted((left, right), key=lambda x: _priority(x, coverage))
                reason = (f"abs_spearman>={threshold:.2f}; higher_coverage_then_"
                          "clinical_primacy_then_lexical")
                remove(kept, removed, rho, len(pair), reason)
                if removed == left:
                    break
    columns = ["feature_kept", "feature_removed", "spearman_rho", "n_pairwise",
               "coverage_kept", "coverage_removed", "reason"]
    return remaining, pd.DataFrame(rows, columns=columns)


def select_features(all_values: dict[str, pd.Series], families: dict[str, str],
                    scenario: str, config: dict, manifest: pd.DataFrame
                    ) -> tuple[dict[str, pd.Series], pd.DataFrame, pd.DataFrame]:
    settings = SCENARIOS[scenario]
    values = dict(all_values)
    if settings["coverage"]:
        values, audit = coverage_filter(values, families, config)
    else:
        audit = pd.DataFrame([
            {"feature": feature, "family": families[feature],
             "coverage_before": float(series.notna().mean()),
             "included_after_coverage_filter": True, "exclusion_reason": ""}
            for feature, series in values.items()])
    redundancy = pd.DataFrame(columns=["feature_kept", "feature_removed", "spearman_rho",
                                       "n_pairwise", "coverage_kept", "coverage_removed", "reason"])
    if settings["redundancy"]:
        values, redundancy = reduce_redundancy(values, config)
    if len(values) < 2:
        raise ValueError(f"{scenario} has fewer than two features after selection")
    return values, audit, redundancy


def feature_selection_audit(master: pd.DataFrame, baseline: pd.DataFrame,
                            candidates: pd.DataFrame, manifest: pd.DataFrame,
                            universe: dict[str, pd.Series], coverage: pd.DataFrame,
                            redundancy: pd.DataFrame, rep: Representation,
                            mapping: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Trace availability separately from approval, selection and actual PCA input.

    Manifest rows retain rejected candidates. LAB source columns absent from the
    Pharma catalog are availability-only records, never additional model inputs.
    """
    rows = manifest.to_dict("records")
    catalog_sources = {
        str(value).strip() for field in ("value_column", "text_column", "reference_status_column")
        if field in candidates for value in candidates[field].dropna() if str(value).strip()
    }
    for column in master:
        if (column.startswith("lab__") and column.endswith(("__value", "__text", "__reference_status"))
                and column not in catalog_sources):
            lab = column.rsplit("__", 1)[0]
            rows.append({"feature": column, "family": "lab", "lab": lab,
                         "source": "integrated_master_uncatalogued",
                         "source_column": column, "representation_type": "not_encoded",
                         "graph_candidate_status": "", "graph_encoding_hint": "",
                         "pct_nonmissing_baseline": (float(baseline[column].notna().mean())
                                                     if len(baseline) else 0.0),
                         "included": False, "exclusion_reason": "not_catalogued_by_pharma00"})
    coverage_index = coverage.set_index("feature")
    removed = set(redundancy.feature_removed)
    mapping_index = mapping.set_index("lab")
    audit = []
    for row in rows:
        feature, source_column = row["feature"], row["source_column"]
        is_lab = row["family"] == "lab"
        lab = canonical_lab_name(row["lab"]) if is_lab else ""
        catalogued = is_lab and row["source"] == "pharma_00_graph_lab_candidates"
        approved = catalogued and row["graph_candidate_status"] in {"include_numeric", "include_categorical"}
        encoded = bool(row["included"])
        eligible = feature in universe
        covered = (bool(coverage_index.at[feature, "included_after_coverage_filter"])
                   if feature in coverage_index.index else False)
        mapping_status = (mapping_index.at[lab, "mapping_status"] if lab in mapping_index.index
                          else "MISSING" if is_lab else "NOT_APPLICABLE")
        reason = row["exclusion_reason"] if not encoded else ""
        if encoded and not eligible:
            reason = "excluded_from_conventional_manifest"
        elif eligible and not covered:
            reason = coverage_index.at[feature, "exclusion_reason"]
        elif feature in removed:
            reason = "redundancy"
        audit.append({"scenario": scenario, "feature": feature, "family": family_name(row["family"]),
                      "lab": lab, "source": row["source"], "source_column": source_column,
                      "present_in_integrated_parquet": source_column in master,
                      "source_column_found": source_column in baseline,
                      "catalogued_by_pharma00": catalogued, "approved_by_pharma00": approved,
                      "graph_candidate_status": row["graph_candidate_status"],
                      "baseline_coverage": row["pct_nonmissing_baseline"],
                      "representation_type": row["representation_type"],
                      "encoded_in_expanded_manifest": encoded,
                      "eligible_in_scenario": eligible, "passed_coverage_filter": covered,
                      "selected_after_redundancy": feature in rep.raw,
                      "s4_mapping_status": mapping_status,
                      "mapped_s4": mapping_status == "MAPPED",
                      "present_in_pca": feature in rep.transformed_features,
                      "exclusion_reason": reason})
    return pd.DataFrame(audit)


def _first_singular_value(block: np.ndarray, name: str) -> float:
    if not np.isfinite(block).all():
        raise BlockNotEstimable(f"{name}: invalid non-finite scaled values")
    singular = np.linalg.svd(block, full_matrices=False, compute_uv=False)
    sigma1 = float(singular[0]) if len(singular) else 0.0
    # Relative precision only: no absolute epsilon that could erase small signal.
    tolerance = np.finfo(float).eps * max(block.shape, default=0) * float(np.max(np.abs(block), initial=0.))
    if sigma1 <= tolerance:
        return 0.0
    if not np.isfinite(sigma1) or not np.isfinite(1.0 / sigma1):
        raise BlockNotEstimable(f"{name}: invalid first singular value or normalization weight")
    return sigma1


def scale_mfa_block(block: pd.DataFrame, block_name: str, *,
                    allow_zero: bool = False) -> tuple[np.ndarray, float]:
    medians = block.median(axis=0)
    if not np.isfinite(medians.to_numpy(dtype=float)).all():
        raise BlockNotEstimable(f"{block_name}: features lack a finite median")
    filled = block.fillna(medians).to_numpy(dtype=float)
    if not np.isfinite(filled).all():
        raise BlockNotEstimable(f"{block_name}: invalid non-finite input after imputation")
    scaled = RobustScaler().fit_transform(filled)
    sigma1 = _first_singular_value(scaled, block_name)
    if sigma1 == 0:
        if allow_zero:
            return np.zeros_like(scaled), 0.0
        raise BlockNotEstimable(f"{block_name}: ZERO_VARIANCE_BLOCK; family has no active signal")
    return scaled / sigma1, sigma1


def _weight_existing_block(block: np.ndarray, block_name: str) -> tuple[np.ndarray, float]:
    sigma1 = _first_singular_value(block, block_name)
    if sigma1 == 0:
        raise BlockNotEstimable(f"{block_name}: ZERO_VARIANCE_BLOCK; all groups are degenerate")
    return block / sigma1, sigma1


def prepare_matrix(raw: pd.DataFrame, families: dict[str, str], balance: bool,
                   lab_group_balance: bool = False,
                   lab_group_by_lab: dict[str, str] | None = None,
                   feature_to_lab: dict[str, str] | None = None,
                   scenario: str = "") -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Refit imputation and MFA; constant LAB groups contribute zero, never epsilon."""
    blocks, names, audit = [], [], []
    if lab_group_balance and (not balance or lab_group_by_lab is None or feature_to_lab is None):
        raise MappingError("S4 LAB-group balancing requires family balancing and canonical lab metadata")

    def record(level, name, columns, sigma, status, active_groups=0):
        weight = 1.0 / sigma if sigma > 0 else 0.0
        audit.append({"scenario": scenario, "level": level, "block": name,
                      "n_features": len(columns), "n_patients": len(raw),
                      "sigma1_before_weighting": sigma, "weight_applied": weight,
                      "nominal_weight": weight, "effective_weight": weight,
                      "status": status, "n_active_groups": active_groups})

    for family in FAMILY_ORDER if balance else ("ALL",):
        columns = list(raw) if family == "ALL" else [x for x in raw if families[x] == family]
        if not columns:
            continue
        active = 0
        try:
            if family == "LAB" and lab_group_balance:
                grouped = {}
                for column in columns:
                    lab = canonical_lab_name(feature_to_lab.get(column, ""))
                    if not lab:
                        raise MappingError(f"S4 LAB feature is absent from canonical manifest mapping: {column}")
                    if lab not in lab_group_by_lab:
                        raise MappingError(f"S4 LAB feature has no mapped clinical group: feature={column}, lab={lab}")
                    grouped.setdefault(lab_group_by_lab[lab], []).append(column)
                nested, columns = [], []
                group_audit_start = len(audit)
                for group in sorted(grouped):
                    group_columns = grouped[group]
                    try:
                        weighted, sigma = scale_mfa_block(raw[group_columns], f"LAB_GROUP:{group}", allow_zero=True)
                    except (BlockNotEstimable, np.linalg.LinAlgError):
                        record("LAB_GROUP", group, group_columns, np.nan, "invalid")
                        raise
                    active += int(sigma > 0)
                    nested.append(weighted)
                    columns.extend(group_columns)
                    record("LAB_GROUP", group, group_columns, sigma,
                           "active" if sigma > 0 else "zero_variance")
                for row in audit[group_audit_start:]:
                    row["n_active_groups"] = active
                scaled, sigma = _weight_existing_block(np.column_stack(nested), "LAB")
                # The effective subgroup weight includes the outer LAB normalization.
                for row in audit[group_audit_start:]:
                    row["effective_weight"] *= 1.0 / sigma
            elif balance:
                scaled, sigma = scale_mfa_block(raw[columns], family)
                active = 1
            else:
                medians = raw[columns].median(axis=0)
                if not np.isfinite(medians.to_numpy(dtype=float)).all():
                    raise BlockNotEstimable("Features lack a finite median")
                scaled = RobustScaler().fit_transform(raw[columns].fillna(medians))
                if _first_singular_value(scaled, "ALL") == 0:
                    raise BlockNotEstimable("ALL: ZERO_VARIANCE_BLOCK; no active signal")
                sigma = np.nan
        except MappingError:
            raise
        except (BlockNotEstimable, np.linalg.LinAlgError) as exc:
            record("FAMILY", family, columns, np.nan, "invalid", active)
            raise BlockNotEstimable(str(exc), pd.DataFrame(audit, columns=BALANCE_COLUMNS)) from exc
        if balance:
            record("FAMILY", family, columns, sigma, "active", active)
        blocks.append(scaled)
        names.extend(columns)
    if not blocks:
        raise BlockNotEstimable("No representation families can be constructed")
    transformed = np.column_stack(blocks)
    order = [names.index(column) for column in raw]
    transformed = transformed[:, order]
    if not np.isfinite(transformed).all():
        raise BlockNotEstimable("Non-finite representation", pd.DataFrame(audit, columns=BALANCE_COLUMNS))
    return transformed, list(raw), pd.DataFrame(audit, columns=BALANCE_COLUMNS)


def fit_embedding(raw: pd.DataFrame, families: dict[str, str], scenario: str,
                  config: dict,
                  lab_group_by_lab: dict[str, str] | None = None,
                  feature_to_lab: dict[str, str] | None = None) -> Representation:
    transformed, names, balance_audit = prepare_matrix(
        raw, families, SCENARIOS[scenario]["balance"],
        SCENARIOS[scenario]["lab_group_balance"], lab_group_by_lab,
        feature_to_lab, scenario)
    try:
        max_possible = min(len(raw) - 1, transformed.shape[1])
        if max_possible < 2:
            raise ValueError("At least three patients and two features are required")
        configured = (config["current_pca"]["max_components"] if scenario == "S0"
                      else config["pca"]["max_components"])
        limit = max_possible if configured is None else min(int(configured), max_possible)
        seed = int(config["random_seed"])
        probe = PCA(n_components=limit, random_state=seed).fit(transformed)
        cumulative = np.cumsum(probe.explained_variance_ratio_)
        target = float(config["pca"]["variance_target"])
        hits = np.flatnonzero(cumulative >= target)
        if scenario != "S0" and not len(hits):
            raise ValueError(f"{scenario} PCA could not reach target variance {target:.0%}")
        dimensions = max(2, int(hits[0] + 1) if len(hits) else limit)
        pca = PCA(n_components=dimensions, random_state=seed).fit(transformed)
        return Representation(raw, families, pca.transform(transformed), pca, names, balance_audit)
    except (ValueError, np.linalg.LinAlgError) as exc:
        exc.balance_audit = balance_audit
        raise



def analyze_mapper(rep: Representation, config: dict) -> dict:
    mapper = config["mapper"]
    eps = CANON.eps_kdist(rep.embedding, int(mapper["eps_k_neighbors"]),
                          float(mapper["eps_percentile"]))
    graph = CANON.run_mapper(rep.embedding[:, :2], rep.embedding, config, eps)
    component_counts, components, nodes, _ = CANON.branches_from_nerve(
        graph, len(rep.raw), int(mapper["minimum_node_support"]), 0)
    nerve, communities, counts, modularity = CANON.mapper_communities(
        graph, nodes, len(rep.raw))
    probabilities, covered, hard = CANON.soft_membership(
        counts, float(mapper["soft_membership_tau"]))
    ties = covered & np.isnan(hard)
    community_sets = [set().union(*(nodes[node] for node in group)) for group in communities]
    minimum = int(config["community_detection"]["minimum_patients"])
    supported = np.array([len(group) >= minimum for group in community_sets])
    return {"graph": graph, "nerve": nerve, "nodes": nodes, "components": components,
            "communities": communities, "community_sets": community_sets,
            "modularity": modularity, "covered": covered, "hard": hard,
            "probabilities": probabilities, "ties": ties, "supported": supported,
            "component_counts": component_counts}


class BootstrapResult:
    """Technical execution, two denominators, and QC-gated primary stability."""

    def __init__(self, matches, requested, success, minimum_success_fraction,
                 replicate_audit, balance_audit):
        self.matched_successes = np.asarray(matches, dtype=int)
        self.conditional = (self.matched_successes / success if success
                            else np.full(len(matches), np.nan))
        self.conservative = (self.matched_successes / requested if requested
                             else np.full(len(matches), np.nan))
        fraction = success / requested if requested else np.nan
        self.qc_status = ("INVALID_BOOTSTRAP" if not success else
                          "BOOTSTRAP_QC_PASS" if fraction >= minimum_success_fraction else
                          "BOOTSTRAP_QC_FAIL")
        self.primary = (self.conditional.copy() if self.qc_status == "BOOTSTRAP_QC_PASS"
                        else np.full(len(matches), np.nan))
        self.replicate_audit = replicate_audit
        self.balance_audit = balance_audit
        failures = replicate_audit.loc[replicate_audit.status.ne("success")]
        self.summary = {"n_requested": requested, "n_attempted": len(replicate_audit),
                        "n_success": success, "n_failed": len(failures),
                        "success_fraction": fraction, "minimum_success_fraction": minimum_success_fraction,
                        "bootstrap_qc_status": self.qc_status,
                        "primary_estimator": "stability_conditional_if_qc_pass",
                        "matching_object": "mapper_node_union_on_sampled_patients"}
        for cause in ("MAPPING_ERROR", "ZERO_VARIANCE_BLOCK", "FEATURE_SELECTION", "PCA", "MAPPER", "OTHER"):
            self.summary[f"n_failed_{cause}"] = int(failures.failure_type.eq(cause).sum())


def bootstrap_stability(all_values: dict[str, pd.Series], families: dict[str, str],
                        manifest: pd.DataFrame, scenario: str,
                        reference_sets: list[set[int]], config: dict,
                        lab_group_by_lab: dict[str, str] | None = None,
                        feature_to_lab: dict[str, str] | None = None) -> BootstrapResult:
    """Refit every requested patient subsample, retaining failures and denominators."""
    settings = config["bootstrap"]
    requested = int(settings["replicates"])
    minimum = float(settings.get("minimum_success_fraction", .95))
    fraction = float(settings["patient_fraction"])
    if requested < 0 or not 0 < minimum <= 1 or not 0 < fraction <= 1:
        raise ValueError("Invalid bootstrap budget, minimum success fraction, or patient fraction")
    if scenario == "S4":
        missing = sorted({canonical_lab_name((feature_to_lab or {}).get(feature, "")) or feature
                          for feature in all_values if families[feature] == "LAB"
                          and canonical_lab_name((feature_to_lab or {}).get(feature, ""))
                          not in (lab_group_by_lab or {})})
        if missing:
            raise MappingError(f"MAPPING_ERROR before bootstrap: complete graph_s4_lab_group_map.csv for {missing}")
    n = len(next(iter(all_values.values())))
    sample_n = min(n, max(3, math.floor(fraction * n)))
    seed = int(config["random_seed"])
    rng = np.random.default_rng(seed)
    matches = np.zeros(len(reference_sets), dtype=int)
    success, rows, balance_rows = 0, [], []
    for replicate_id in range(requested):
        indices = np.sort(rng.choice(n, sample_n, replace=False))
        sample_text = ",".join(map(str, indices))
        row = {"scenario": scenario, "replicate_id": replicate_id, "seed": seed,
               "sample_hash": hashlib.sha256(sample_text.encode()).hexdigest(),
               "sample_indices": sample_text, "n_sampled": sample_n,
               "status": "success", "failure_stage": "", "failure_type": "", "failure_message": "",
               "n_features_selected": 0, "n_communities": 0, "n_supported_communities": 0}
        sampled = {name: series.iloc[indices].reset_index(drop=True)
                   for name, series in all_values.items()}
        stage = "FEATURE_SELECTION"
        local_audit = pd.DataFrame(columns=BALANCE_COLUMNS)
        try:
            selected, _, _ = select_features(sampled, families, scenario, config, manifest)
            raw = pd.DataFrame(selected)
            row["n_features_selected"] = raw.shape[1]
            stage = "PCA"
            fitted = fit_embedding(raw, {name: families[name] for name in raw}, scenario, config,
                                   lab_group_by_lab=lab_group_by_lab, feature_to_lab=feature_to_lab)
            local_audit = fitted.balance_audit
            stage = "MAPPER"
            result = analyze_mapper(fitted, config)
            row["n_communities"] = len(result["community_sets"])
            row["n_supported_communities"] = int(np.asarray(result["supported"]).sum())
            candidates = [set(indices[list(group)]) for group in result["community_sets"]]
            sampled_ids = set(indices)
            for community, reference in enumerate(reference_sets):
                ref = reference & sampled_ids
                best = max((len(ref & other) / len(ref | other) if ref | other else 0.
                            for other in candidates), default=0.)
                matches[community] += best >= float(settings["branch_match_jaccard"])
            success += 1
        except MappingError:
            raise
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            # KeyError/TypeError/AttributeError and other programming bugs propagate.
            non_estimable = isinstance(exc, BlockNotEstimable)
            cause = ("ZERO_VARIANCE_BLOCK" if non_estimable and "ZERO_VARIANCE_BLOCK" in str(exc)
                     else "OTHER" if non_estimable else stage)
            row.update(status="not_estimable" if non_estimable else "failed",
                       failure_stage="MATRIX" if non_estimable else stage,
                       failure_type=cause, failure_message=f"{type(exc).__name__}: {exc}")
            local_audit = getattr(exc, "balance_audit", local_audit)
            logging.warning("%s bootstrap attempt %d %s (%s): %s", scenario, replicate_id,
                            row["status"], cause, exc)
        rows.append(row)
        if not local_audit.empty:
            local_audit = local_audit.copy()
            local_audit.insert(1, "replicate_id", replicate_id)
            balance_rows.append(local_audit)
    columns = ["scenario", "replicate_id", "seed", "sample_hash", "sample_indices", "n_sampled", "status",
               "failure_stage", "failure_type", "failure_message", "n_features_selected",
               "n_communities", "n_supported_communities"]
    audit = pd.DataFrame(rows, columns=columns)
    block_audit = (pd.concat(balance_rows, ignore_index=True) if balance_rows else
                   pd.DataFrame(columns=["scenario", "replicate_id", *BALANCE_COLUMNS[1:]]))
    return BootstrapResult(matches, requested, success, minimum, audit, block_audit)


def patient_membership(scenario: str, baseline: pd.DataFrame, mapped: dict) -> pd.DataFrame:
    """Create the common patient-level assignment contract for one scenario."""
    pop_column = resolve_concept(baseline, "pop")
    hard, probabilities = mapped["hard"], mapped["probabilities"]
    result = pd.DataFrame({
        "scenario": scenario, "patient_id": baseline["patient_id"].to_numpy(),
        "mapper_covered": mapped["covered"], "community_membership_tie": mapped["ties"],
        "hard_community": pd.array(hard, dtype="Int64"),
        "community_supported": [bool(mapped["supported"][int(value)])
                                if np.isfinite(value) else False for value in hard],
        "max_community_membership": (probabilities.max(axis=1)
                                     if probabilities.shape[1] else np.nan),
        "baseline_pop": baseline[pop_column].to_numpy(),
    })
    for community in range(probabilities.shape[1]):
        result[f"pi_community_{community}"] = probabilities[:, community]
    return result


def community_pop_crosswalk(membership: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    columns = ["scenario", "hard_community", "baseline_pop", "n_patients",
               "pct_within_community", "community_supported", "bootstrap_stability",
               "stability_status"]
    valid = (membership.mapper_covered & ~membership.community_membership_tie
             & membership.community_supported & membership.hard_community.notna()
             & membership.baseline_pop.isin(CANON.POPS))
    counts = (membership.loc[valid].groupby(
        ["scenario", "hard_community", "baseline_pop"], observed=True)
        .size().rename("n_patients").reset_index())
    if counts.empty:
        return pd.DataFrame(columns=columns)
    counts["pct_within_community"] = counts.n_patients / counts.groupby(
        ["scenario", "hard_community"]).n_patients.transform("sum")
    counts["community_supported"] = True
    merged = counts.merge(stability, left_on=["scenario", "hard_community"],
                          right_on=["scenario", "community"], how="left")
    return merged.drop(columns="community")[columns]


def eligible_community_sets(membership: pd.DataFrame) -> dict[int, set]:
    """Patient sets for direct overlap: Mapper-covered hard assignments without ties."""
    valid = (membership.mapper_covered & ~membership.community_membership_tie
             & membership.hard_community.notna())
    return {int(community): set(group.patient_id) for community, group in
            membership.loc[valid].groupby("hard_community")}


def cross_scenario_tables(memberships: dict[str, pd.DataFrame], stable_ids: dict[str, int | None]
                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overlap_rows, stable_rows, ari_rows = [], [], []
    scenarios = list(SCENARIOS)
    for i, scenario_a in enumerate(scenarios):
        frame_a = memberships[scenario_a]
        sets_a = eligible_community_sets(frame_a)
        for scenario_b in scenarios[i + 1:]:
            frame_b = memberships[scenario_b]
            sets_b = eligible_community_sets(frame_b)
            for community_a, patients_a in sets_a.items():
                for community_b, patients_b in sets_b.items():
                    intersection, union = patients_a & patients_b, patients_a | patients_b
                    overlap_rows.append({"scenario_a": scenario_a, "community_a": community_a,
                        "scenario_b": scenario_b, "community_b": community_b,
                        "n_a": len(patients_a), "n_b": len(patients_b),
                        "n_intersection": len(intersection), "n_union": len(union),
                        "jaccard": len(intersection) / len(union) if union else np.nan,
                        "overlap_coefficient": len(intersection) / min(len(patients_a), len(patients_b))
                        if patients_a and patients_b else np.nan})
            stable_a, stable_b = stable_ids[scenario_a], stable_ids[scenario_b]
            patients_a = sets_a.get(stable_a, set()) if stable_a is not None else set()
            patients_b = sets_b.get(stable_b, set()) if stable_b is not None else set()
            intersection, union = patients_a & patients_b, patients_a | patients_b
            stable_rows.append({"scenario_a": scenario_a, "stable_community_a": stable_a,
                "scenario_b": scenario_b, "stable_community_b": stable_b,
                "n_a": len(patients_a), "n_b": len(patients_b),
                "n_intersection": len(intersection),
                "jaccard": (len(intersection) / len(union)
                            if stable_a is not None and stable_b is not None and union else np.nan),
                "overlap_coefficient": (len(intersection) / min(len(patients_a), len(patients_b))
                                        if patients_a and patients_b else np.nan)})
            joined = frame_a.merge(frame_b, on="patient_id", suffixes=("_a", "_b"))
            valid = (joined.mapper_covered_a & joined.mapper_covered_b
                     & ~joined.community_membership_tie_a & ~joined.community_membership_tie_b
                     & joined.community_supported_a & joined.community_supported_b
                     & joined.hard_community_a.notna() & joined.hard_community_b.notna())
            used = joined.loc[valid]
            ari_rows.append({"scenario_a": scenario_a, "scenario_b": scenario_b,
                             "n_patients_used": len(used),
                             "ari": adjusted_rand_score(used.hard_community_a, used.hard_community_b)
                             if len(used) else np.nan})
    total = len(set().union(*(set(frame.patient_id) for frame in memberships.values())))
    for row in ari_rows:
        row.update(n_patients_total=total,
                   fraction_total_used=row["n_patients_used"] / total if total else np.nan,
                   assignment_object="supported_unambiguous_hard_assignment",
                   ari_status="estimated_pairwise_descriptive" if row["n_patients_used"] else "not_estimable")
    for row in stable_rows:
        row.update(n_patients_total=total,
                   n_union=row["n_a"] + row["n_b"] - row["n_intersection"],
                   coverage_a=row["n_a"] / total if total else np.nan,
                   coverage_b=row["n_b"] / total if total else np.nan,
                   matching_object="unambiguous_hard_assignment")
    overlap_columns = ["scenario_a", "community_a", "scenario_b", "community_b", "n_a", "n_b",
                       "n_intersection", "n_union", "jaccard", "overlap_coefficient"]
    return pd.DataFrame(overlap_rows, columns=overlap_columns), pd.DataFrame(stable_rows), pd.DataFrame(ari_rows)


def pca_table(rep: Representation) -> pd.DataFrame:
    ratios = rep.pca.explained_variance_ratio_
    return pd.DataFrame({"PC": [f"PC{i}" for i in range(1, len(ratios) + 1)],
                         "n_features": rep.raw.shape[1],
                         "n_components": len(ratios),
                         "explained_variance_PC1": float(ratios[0]),
                         "explained_variance_PC2": float(ratios[1]),
                         "explained_variance_ratio": ratios,
                         "cumulative_variance": np.cumsum(ratios)})


def draw_figures(results: dict[str, dict], stability_rows: pd.DataFrame,
                 figure_dir: Path) -> None:
    n_scenarios = len(results)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(4.5 * n_scenarios, 4.4))
    axes = np.atleast_1d(axes)
    for axis, (scenario, result) in zip(axes, results.items()):
        embedding, nodes, nerve = result["rep"].embedding, result["mapper"]["nodes"], result["mapper"]["nerve"]
        centers = {node: embedding[list(members), :2].mean(axis=0) for node, members in nodes.items()}
        for left, right in nerve.edges:
            axis.plot([centers[left][0], centers[right][0]], [centers[left][1], centers[right][1]],
                      color="#999999", lw=1, zorder=1)
        communities = {node: i for i, group in enumerate(result["mapper"]["communities"]) for node in group}
        for node, center in centers.items():
            axis.scatter(*center, s=18 + 7 * len(nodes[node]),
                         color=CANON.COLORS[communities[node] % len(CANON.COLORS)],
                         edgecolor="white", linewidth=.5, zorder=2)
        axis.set(title=scenario, xlabel="PC1", ylabel="PC2")
    fig.suptitle("Graph representation sensitivity: fixed Mapper", fontweight="bold")
    fig.tight_layout(); fig.savefig(figure_dir / "02_graph_representation_sensitivity_mapper.png", bbox_inches="tight"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    if not stability_rows.empty:
        labels = stability_rows.scenario + " · C" + stability_rows.community.astype(str)
        colors = [CANON.COLORS[i % len(CANON.COLORS)] for i in range(len(labels))]
        axis.bar(labels, stability_rows.bootstrap_stability, color=colors)
        axis.set_xlim(-.5, len(labels) - .5)
        for position, row in enumerate(stability_rows.itertuples(index=False)):
            if pd.isna(row.bootstrap_stability):
                axis.text(position, .02, row.bootstrap_qc_status, rotation=90, fontsize=7)
    axis.axhline(float(next(iter(results.values()))["config"]["bootstrap"]["stable_community_threshold"]),
                 color="#333333", linestyle="--", label="stable threshold")
    axis.set(ylabel="Match proportion among valid replicates (QC gated)", ylim=(0, 1.05),
             title="Community stability by representation")
    axis.tick_params(axis="x", rotation=45); axis.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figure_dir / "02_graph_representation_sensitivity_stability.png", bbox_inches="tight"); plt.close(fig)


def draw_stable_core_heatmap(stable_overlap: pd.DataFrame, figure_dir: Path) -> None:
    scenarios = list(SCENARIOS)
    matrix = np.full((len(scenarios), len(scenarios)), np.nan)
    positions = {scenario: index for index, scenario in enumerate(scenarios)}
    for row in stable_overlap.itertuples(index=False):
        left, right = positions[row.scenario_a], positions[row.scenario_b]
        matrix[left, right] = matrix[right, left] = row.jaccard
        if pd.notna(row.stable_community_a):
            matrix[left, left] = 1.0
        if pd.notna(row.stable_community_b):
            matrix[right, right] = 1.0
    fig, axis = plt.subplots(figsize=(6, 5.2))
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap="Blues", vmin=0, vmax=1)
    for row in range(len(scenarios)):
        for column in range(len(scenarios)):
            label = "NA" if np.isnan(matrix[row, column]) else f"{matrix[row, column]:.2f}"
            axis.text(column, row, label, ha="center", va="center",
                      color="white" if np.isfinite(matrix[row, column]) and matrix[row, column] > .55 else "black")
    axis.set(xticks=range(len(scenarios)), yticks=range(len(scenarios)),
             xticklabels=scenarios, yticklabels=scenarios,
             title="Stable-core Jaccard across representations")
    fig.colorbar(image, ax=axis, label="Jaccard")
    fig.tight_layout()
    fig.savefig(figure_dir / "02_graph_stable_core_overlap.png", bbox_inches="tight")
    plt.close(fig)


def common_intersection_ari(memberships: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    """A descriptive ARI on exactly the same supported patients in every scenario."""
    eligible = {}
    total = len(set().union(*(set(frame.patient_id) for frame in memberships.values())))
    for scenario, frame in memberships.items():
        valid = (frame.mapper_covered & ~frame.community_membership_tie
                 & frame.community_supported & frame.hard_community.notna())
        eligible[scenario] = frame.loc[valid].set_index("patient_id").hard_community
    common = set.intersection(*(set(series.index) for series in eligible.values()))
    ids = sorted(common, key=str)
    fraction = len(ids) / total if total else 0.
    settings = config.get("comparisons", {})
    sufficient = (len(ids) >= int(settings.get("minimum_common_patients", 20))
                  and fraction >= float(settings.get("minimum_common_fraction", .5)))
    rows = []
    for i, left in enumerate(SCENARIOS):
        for right in list(SCENARIOS)[i + 1:]:
            a, b = eligible[left].reindex(ids), eligible[right].reindex(ids)
            # Single-cluster partitions cannot establish multiple reproducible branches.
            estimable = sufficient and a.nunique() >= 2 and b.nunique() >= 2
            rows.append({"scenario_a": left, "scenario_b": right,
                         "n_patients_used": len(ids), "n_patients_total": total,
                         "fraction_total_used": fraction,
                         "ari": float(adjusted_rand_score(a, b)) if estimable else np.nan,
                         "ari_status": "estimated_common_intersection_descriptive" if estimable else "not_estimable",
                         "reason": "" if estimable else "insufficient_common_sample_or_degenerate_partition",
                         "assignment_object": "supported_unambiguous_hard_assignment_all_scenarios"})
    return pd.DataFrame(rows)


def node_union_core_overlap(results: dict, stable_ids: dict, baseline: pd.DataFrame) -> pd.DataFrame:
    """A second, explicitly named overlap object; never replace hard-core Jaccard."""
    rows = []
    for i, left in enumerate(SCENARIOS):
        for right in list(SCENARIOS)[i + 1:]:
            a, b = stable_ids[left], stable_ids[right]
            left_set = results[left]["mapper"]["community_sets"][a] if a is not None else set()
            right_set = results[right]["mapper"]["community_sets"][b] if b is not None else set()
            intersection, union = left_set & right_set, left_set | right_set
            rows.append({"scenario_a": left, "scenario_b": right,
                         "stable_community_a": a, "stable_community_b": b,
                         "n_a": len(left_set), "n_b": len(right_set),
                         "n_intersection": len(intersection), "n_union": len(union),
                         "n_patients_total": len(baseline),
                         "coverage_a": len(left_set) / len(baseline),
                         "coverage_b": len(right_set) / len(baseline),
                         "node_union_jaccard": len(intersection) / len(union)
                         if a is not None and b is not None and union else np.nan,
                         "matching_object": "mapper_node_union"})
    return pd.DataFrame(rows)


def interpretation_check(summary: pd.DataFrame, stable_overlap: pd.DataFrame | None = None,
                         cross_ari: pd.DataFrame | None = None,
                         config: dict | None = None) -> dict:
    """Separate execution QC, a hard-assignment core, and multi-branch structure.

    No between-scenario scientific threshold is inferred from observed results.
    Optional thresholds require an explicit pre-run justification in the config.
    """
    settings = (config or {}).get("interpretation", {})
    core_threshold = settings.get("core_jaccard_threshold")
    ari_threshold = settings.get("global_ari_threshold")
    if any(value is not None for value in (core_threshold, ari_threshold)):
        if not settings.get("threshold_justification"):
            raise ValueError("Between-scenario thresholds require a pre-run threshold_justification")
        if any(not 0 <= float(value) <= 1 for value in (core_threshold, ari_threshold) if value is not None):
            raise ValueError("Between-scenario similarity thresholds must be in [0, 1]")
    qc_pass = (set(summary.scenario) == set(SCENARIOS)
               and "bootstrap_qc_status" in summary
               and summary.bootstrap_qc_status.eq("BOOTSTRAP_QC_PASS").all())
    cleaned = summary.loc[summary.scenario.ne("S0")]
    report = {"COMPUTATIONAL_QC": "COMPUTATIONAL_QC_PASS" if qc_pass else "COMPUTATIONAL_QC_FAIL",
              "CORE_EVIDENCE": "NOT_EVALUABLE", "GLOBAL_PARTITION_ROBUSTNESS": "NOT_EVALUABLE",
              "stable_communities_by_scenario": {
                  row.scenario: int(row.n_stable_communities) if pd.notna(row.n_stable_communities) else None
                  for row in summary.itertuples()},
              "core_matching_object": "unambiguous_hard_assignment",
              "bootstrap_matching_object": "mapper_node_union_on_sampled_patients",
              "reason": "QC failed" if not qc_pass else
                        "Only prospectively configured criteria are evaluated; null cutoffs remain NOT_EVALUABLE",
              "core_pair_evidence": [], "partition_pair_evidence": [],
              "threshold_justification": settings.get("threshold_justification")}

    def evidence(table, column, threshold, output):
        if table is None:
            return "NOT_EVALUABLE"
        pairs = table.loc[table.scenario_a.ne("S0") & table.scenario_b.ne("S0")]
        usable, supporting = 0, 0
        floor = (config or {}).get("comparisons", {})
        for row in json.loads(pairs.to_json(orient="records")):
            value = row[column]
            adequate = (pd.notna(value) and
                        (column != "ari" or (row["n_patients_used"] >= int(floor.get("minimum_common_patients", 20))
                         and row["fraction_total_used"] >= float(floor.get("minimum_common_fraction", .5)))))
            adequate = adequate and (column != "jaccard" or min(row["n_a"], row["n_b"]) >= int(floor.get("minimum_common_patients", 20)))
            status = ("NOT_EVALUABLE" if not qc_pass or threshold is None or not adequate else
                      "SUPPORTED" if value >= threshold else "UNSUPPORTED")
            output.append({**row, "evidence_status": status})
            if adequate:
                usable += 1
                supporting += int(threshold is not None and value >= threshold)
        if not qc_pass or threshold is None or usable < 6:
            return "NOT_EVALUABLE"
        return "SUPPORTED" if supporting == 6 else "PARTIAL" if supporting else "UNSUPPORTED"

    report["CORE_EVIDENCE"] = evidence(stable_overlap, "jaccard", core_threshold, report["core_pair_evidence"])
    global_status = evidence(cross_ari, "ari", ari_threshold, report["partition_pair_evidence"])
    if global_status != "NOT_EVALUABLE" and not cleaned.n_stable_communities.ge(2).all():
        global_status = "PARTIAL" if cleaned.n_stable_communities.ge(2).any() else "UNSUPPORTED"
    report["GLOBAL_PARTITION_ROBUSTNESS"] = global_status
    return report


def concat_redundancy_audits(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Use an explicit schema even when every scenario produces an empty audit."""
    typed = [frame[list(REDUNDANCY_DTYPES)].astype(REDUNDANCY_DTYPES)
             for frame in frames if not frame.empty]
    return (pd.concat(typed, ignore_index=True) if typed else
            pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in REDUNDANCY_DTYPES.items()}))


def run(args: argparse.Namespace) -> None:
    config = CANON.load_config(args.config)
    # Validate optional claim criteria before observing results or starting work.
    interpretation_check(pd.DataFrame(columns=["scenario", "n_stable_communities"]), config=config)
    lab_group_frame = load_lab_group_map(args.lab_group_map)
    lab_group_by_lab = lab_group_lookup(lab_group_frame)
    dirs = create_study_dirs("graph/02_graph_representation_sensitivity")
    CANON._configure_logging(dirs["logs"] / "02_graph_representation_sensitivity.log")
    logging.info("Environment maintenance: if pandas warns about numexpr, check dependency compatibility "
                 "for numexpr >= 2.8.4 in the executing Python environment; no packages are changed here.")
    master, candidates, _ = CANON.load_real_inputs(args.integrated, args.graph_lab_candidates)
    baseline = CANON.select_baseline(master)
    pop = baseline[resolve_concept(baseline, "pop")].to_numpy()
    age = CANON.load_age_at_diagnosis(args.baseline_metrics)
    baseline = baseline.merge(age, on="patient_id", how="left", validate="one_to_one", suffixes=("", "_blockA"))
    if "age_dx_blockA" in baseline:
        baseline["age_dx"] = baseline.pop("age_dx_blockA")

    # A zero threshold obtains the complete eligible universe. S0 is then rebuilt
    # with the canonical threshold to reproduce current coverage exceptions exactly.
    full_manifest, full_values, _, _ = CANON.build_feature_manifest(baseline, candidates, 0.0)
    feature_to_lab = {
        str(row.feature): canonical_lab_name(str(row.lab))
        for row in full_manifest.loc[
            full_manifest["included"] & full_manifest["family"].eq("lab")
        ].itertuples(index=False)
    }
    current_manifest, current_values, _, _ = CANON.build_feature_manifest(
        baseline, candidates, float(CANON.load_config(Path(__file__).with_name("config.yaml"))["feature_selection"]["minimum_baseline_coverage"]))
    full_families = {row.feature: family_name(row.family)
                     for row in full_manifest.loc[full_manifest.included].itertuples()}
    current_families = {name: full_families[name] for name in current_values}
    mapping_audit_path = dirs["tables"] / "02_graph_S4_lab_group_mapping_audit.csv"
    # The full eligible universe can re-enter in a subsample, even if pruned in S4.
    s4_mapping_audit = validate_s4_mapping(
        full_values, full_families, lab_group_frame, feature_to_lab,
        require_all=True, audit_path=mapping_audit_path)
    lab_group_index = lab_group_frame.set_index("lab")
    results, memberships, stable_ids = {}, {}, {}
    summary_rows, coverage_rows, redundancy_rows, feature_rows, stability_rows = [], [], [], [], []
    bootstrap_audits, bootstrap_summaries, bootstrap_balance = [], [], []
    selection_audits = []

    for scenario in SCENARIOS:
        universe = current_values if scenario == "S0" else full_values
        families = current_families if scenario == "S0" else full_families
        selected, coverage, redundancy = select_features(universe, families, scenario, config, full_manifest)
        coverage.insert(0, "scenario", scenario); coverage_rows.append(coverage)
        redundancy.insert(0, "scenario", scenario); redundancy_rows.append(redundancy)
        raw = pd.DataFrame(selected, index=baseline.index)
        final_families = {name: families[name] for name in raw}
        if scenario == "S4":
            s4_mapping_audit["included_S4"] = s4_mapping_audit.feature.isin(raw.columns)
            s4_mapping_audit.to_csv(mapping_audit_path, index=False)
        for feature in sorted(universe):
            lab = feature_to_lab.get(feature, "") if families[feature] == "LAB" else ""
            map_row = lab_group_index.loc[lab] if lab and lab in lab_group_index.index else None
            feature_rows.append({"scenario": scenario, "feature": feature,
                                 "family": families[feature], "included": feature in raw,
                                 "lab": lab,
                                 "s4_group": (map_row["s4_group"] if map_row is not None else ""),
                                 "s4_subgroup": (map_row["s4_subgroup"] if map_row is not None else ""),
                                 "exclusion_stage": "" if feature in raw else
                                 ("redundancy" if feature in set(redundancy.feature_removed) else "coverage_or_encoding")})
        try:
            rep = fit_embedding(
                raw, final_families, scenario, config,
                lab_group_by_lab=lab_group_by_lab, feature_to_lab=feature_to_lab)
        except MappingError:
            raise
        except (ValueError, np.linalg.LinAlgError) as exc:
            getattr(exc, "balance_audit", pd.DataFrame(columns=BALANCE_COLUMNS)).to_csv(
                dirs["tables"] / f"02_graph_{scenario}_full_fit_balance_audit.csv", index=False)
            raise ValueError(f"Full-sample {scenario} representation is not estimable: {exc}") from exc
        rep.balance_audit.to_csv(
            dirs["tables"] / f"02_graph_{scenario}_full_fit_balance_audit.csv", index=False)
        selection_audits.append(feature_selection_audit(
            master, baseline, candidates, full_manifest, universe, coverage,
            redundancy, rep, lab_group_frame, scenario))
        pd.concat(selection_audits, ignore_index=True).to_csv(
            dirs["tables"] / "02_graph_feature_selection_audit.csv", index=False)
        mapped = analyze_mapper(rep, config)
        boot = bootstrap_stability(universe, families, full_manifest, scenario,
                                        mapped["community_sets"], config,
                                        lab_group_by_lab=lab_group_by_lab,
                                        feature_to_lab=feature_to_lab)
        bootstrap_audits.append(boot.replicate_audit)
        bootstrap_summaries.append({"scenario": scenario, **boot.summary})
        if not boot.balance_audit.empty:
            bootstrap_balance.append(boot.balance_audit)
            pd.concat(bootstrap_balance, ignore_index=True).to_csv(
                dirs["tables"] / "02_graph_bootstrap_balance_audit.csv", index=False)
        # Persist after each scenario so completed execution diagnostics survive later failures.
        pd.concat(bootstrap_audits, ignore_index=True).to_csv(
            dirs["tables"] / "02_graph_bootstrap_replicate_audit.csv", index=False)
        pd.DataFrame(bootstrap_summaries).to_csv(
            dirs["tables"] / "02_graph_bootstrap_qc_summary.csv", index=False)
        stability = boot.primary
        if boot.qc_status != "BOOTSTRAP_QC_PASS":
            logging.warning("%s %s: %d/%d valid bootstrap attempts; stability claims are not evaluable",
                            scenario, boot.qc_status, boot.summary["n_success"], boot.summary["n_requested"])
        stable_threshold = float(config["bootstrap"]["stable_community_threshold"])
        for community, score in enumerate(stability):
            stability_rows.append({"scenario": scenario, "community": community,
                                   "n_patients_node_union": len(mapped["community_sets"][community]),
                                   "bootstrap_stability": score,
                                   "stability_status": ("stable" if score >= stable_threshold else "provisional")
                                   if boot.qc_status == "BOOTSTRAP_QC_PASS" else boot.qc_status,
                                   "matched_successes": int(boot.matched_successes[community]),
                                   "n_bootstrap_requested": boot.summary["n_requested"],
                                   "n_bootstrap_success": boot.summary["n_success"],
                                   "bootstrap_success_fraction": boot.summary["success_fraction"],
                                   "stability_conditional": boot.conditional[community],
                                   "stability_conservative": boot.conservative[community],
                                   "bootstrap_qc_status": boot.qc_status,
                                   "primary_denominator": "n_bootstrap_success_if_qc_pass",
                                   "matching_object": "mapper_node_union_on_sampled_patients"})
        # When several communities pass the descriptive threshold, use one
        # deterministic representative for the pairwise stable-core comparison.
        stable_candidates = [community for community, score in enumerate(stability)
                             if score >= stable_threshold]
        stable_id = (sorted(stable_candidates, key=lambda community:
                     (-stability[community], -len(mapped["community_sets"][community]), community))[0]
                     if stable_candidates else None)
        stable_ids[scenario] = stable_id
        memberships[scenario] = patient_membership(scenario, baseline, mapped)
        hard, covered, ties, supported = mapped["hard"], mapped["covered"], mapped["ties"], mapped["supported"]
        valid = covered & ~ties & np.isfinite(hard) & np.isin(pop, CANON.POPS)
        valid &= np.array([supported[int(x)] if np.isfinite(x) else False for x in hard])
        if supported.sum() >= 2:
            ari, ari_p = CANON.ari_permutation(hard[valid].astype(int), pop[valid],
                                               int(config["permutation"]["ari_replicates"]), int(config["random_seed"]))
        else:
            ari = ari_p = np.nan
        ph = CANON.persistent_homology_optional(rep.embedding, config)
        ratios = rep.pca.explained_variance_ratio_
        pca_table(rep).to_csv(dirs["tables"] / f"02_graph_{scenario}_pca_variance.csv", index=False)
        unambiguous = int((covered & ~ties & np.isfinite(hard)).sum())
        missing = int(raw.isna().sum().sum())
        summary_rows.append({"scenario": scenario, "n_baseline_patients": len(baseline),
            "n_original_features": len(universe), "n_final_features": raw.shape[1],
            "n_lab_features": sum(x == "LAB" for x in final_families.values()),
            "n_pro_features": sum(x == "PRO" for x in final_families.values()),
            "n_clinical_features": sum(x == "CLINICAL" for x in final_families.values()),
            "n_missing_before_imputation": missing, "pct_missing_before_imputation": missing / raw.size,
            "pca_n_components": len(ratios), "pca_cumulative_variance": float(ratios.sum()),
            "pc1_variance": float(ratios[0]), "pc2_variance": float(ratios[1]),
            "mapper_coverage": float(covered.mean()), "n_mapper_nodes": len(mapped["nodes"]),
            "n_connected_components": len(mapped["components"]), "n_communities": len(mapped["communities"]),
            "n_supported_communities": int(supported.sum()),
            "n_stable_communities": int((stability >= stable_threshold).sum())
            if boot.qc_status == "BOOTSTRAP_QC_PASS" else np.nan,
            "bootstrap_n_requested": boot.summary["n_requested"],
            "bootstrap_n_attempted": boot.summary["n_attempted"],
            "bootstrap_n_success": boot.summary["n_success"],
            "bootstrap_n_failed": boot.summary["n_failed"],
            "bootstrap_success_fraction": boot.summary["success_fraction"],
            "bootstrap_qc_status": boot.qc_status,
            "n_age_dx_observed": int(pd.to_numeric(baseline.age_dx, errors="coerce").notna().sum()),
            "stable_community_id": stable_id,
            "stable_community_n_patients": (len(mapped["community_sets"][stable_id])
                                             if stable_id is not None else np.nan),
            "stable_community_bootstrap_stability": (stability[stable_id]
                                                      if stable_id is not None else np.nan),
            "modularity": mapped["modularity"], "n_unambiguous": unambiguous,
            "pct_unambiguous": unambiguous / len(baseline), "ari_vs_pop": ari,
            "ari_permutation_p": ari_p, "ari_n_used": int(valid.sum()),
            "h1_total_persistence": ph.get("total_h1_persistence", np.nan),
            "h1_permutation_p": ph.get("permutation_p_value", np.nan)})
        results[scenario] = {"rep": rep, "mapper": mapped, "config": config}

    summary = pd.DataFrame(summary_rows)
    s3_features = set(results["S3"]["rep"].raw)
    s4_features = set(results["S4"]["rep"].raw)
    if s3_features != s4_features:
        raise AssertionError("S3 and S4 feature sets differ")
    s3_missing = int(results["S3"]["rep"].raw.isna().sum().sum())
    s4_missing = int(results["S4"]["rep"].raw.isna().sum().sum())
    if s3_missing != s4_missing:
        raise AssertionError("S3 and S4 missing-value counts differ")
    pd.testing.assert_frame_equal(results["S3"]["rep"].raw, results["S4"]["rep"].raw)
    coverage_all = pd.concat(coverage_rows, ignore_index=True)
    redundancy_all = concat_redundancy_audits(redundancy_rows)
    stability_all = pd.DataFrame(stability_rows, columns=["scenario", "community", "n_patients_node_union",
        "bootstrap_stability", "stability_status", "matched_successes", "n_bootstrap_requested",
        "n_bootstrap_success", "bootstrap_success_fraction", "stability_conditional",
        "stability_conservative", "bootstrap_qc_status", "primary_denominator", "matching_object"])
    membership_all = pd.concat(memberships.values(), ignore_index=True, sort=False)
    crosswalk = community_pop_crosswalk(membership_all, stability_all)
    overlap, stable_overlap, cross_ari = cross_scenario_tables(memberships, stable_ids)
    common_ari = common_intersection_ari(memberships, config)
    union_overlap = node_union_core_overlap(results, stable_ids, baseline)
    common_ari.to_csv(dirs["tables"] / "02_graph_common_intersection_ari.csv", index=False)
    union_overlap.to_csv(dirs["tables"] / "02_graph_stable_core_node_union_overlap.csv", index=False)
    block_audit = (pd.concat(bootstrap_balance, ignore_index=True) if bootstrap_balance else
                   pd.DataFrame(columns=["scenario", "replicate_id", *BALANCE_COLUMNS[1:]]))
    block_audit.to_csv(dirs["tables"] / "02_graph_bootstrap_balance_audit.csv", index=False)
    interpretation = interpretation_check(summary, stable_overlap, cross_ari, config)
    CANON.write_json(dirs["tables"] / "02_graph_interpretation_qc.json", interpretation)
    coverage_all.to_csv(dirs["tables"] / "02_graph_coverage_audit.csv", index=False)
    redundancy_all.to_csv(dirs["tables"] / "02_graph_redundancy_audit.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(dirs["tables"] / "02_graph_scenario_feature_sets.csv", index=False)
    stability_all.to_csv(dirs["tables"] / "02_graph_scenario_community_stability.csv", index=False)
    membership_all.to_csv(dirs["tables"] / "02_graph_scenario_patient_membership.csv", index=False)
    crosswalk.to_csv(dirs["tables"] / "02_graph_scenario_community_pop_crosswalk.csv", index=False)
    overlap.to_csv(dirs["tables"] / "02_graph_cross_scenario_community_overlap.csv", index=False)
    stable_overlap.to_csv(dirs["tables"] / "02_graph_stable_core_overlap.csv", index=False)
    cross_ari.to_csv(dirs["tables"] / "02_graph_cross_scenario_ari.csv", index=False)
    summary.to_csv(dirs["tables"] / "02_graph_representation_sensitivity_summary.csv", index=False)
    if s4_mapping_audit is None:
        raise RuntimeError("S4 mapping audit was not generated")
    s4_mapping_audit.to_csv(
        dirs["tables"] / "02_graph_S4_lab_group_mapping_audit.csv", index=False)
    results["S4"]["rep"].balance_audit.to_csv(
        dirs["tables"] / "02_graph_S4_balance_weights.csv", index=False)
    if not args.dry_run:
        draw_figures(results, stability_all, dirs["figures"])
        draw_stable_core_heatmap(stable_overlap, dirs["figures"])

    print("\nGRAPH REPRESENTATION SENSITIVITY")
    print(f"\nBaseline patients: {len(baseline)}")
    for row in summary.itertuples(index=False):
        print(f"\nScenario {row.scenario}\nFeatures: {row.n_final_features}"
              f"\nMissing/imputed: {row.n_missing_before_imputation} ({row.pct_missing_before_imputation:.1%})"
              f"\nPCA components: {row.pca_n_components}\nPCA cumulative variance: {row.pca_cumulative_variance:.3f}"
              f"\nMapper nodes: {row.n_mapper_nodes}\nConnected components: {row.n_connected_components}"
              f"\nCommunities: {row.n_communities}\nSupported: {row.n_supported_communities}"
              f"\nBootstrap valid/requested: {row.bootstrap_n_success}/{row.bootstrap_n_requested}"
              f"\nBootstrap QC: {row.bootstrap_qc_status}"
              f"\nStable: {int(row.n_stable_communities) if pd.notna(row.n_stable_communities) else row.bootstrap_qc_status}"
              f"\nARI vs Pop: {row.ari_vs_pop:.3g}")
    print("\nINTERPRETATION CHECK")
    print(json.dumps(interpretation, indent=2, default=str))
    print("ARI tables are descriptive: ARI near 0 is chance-level agreement; negatives are possible. "
          "Pairwise ARI uses different patient intersections and does not establish global robustness.")
    print("Bootstrap matches Mapper node unions; core Jaccard uses unambiguous hard assignments. "
          "Both objects are reported separately.")
    print("\nCROSS-SCENARIO STABLE CORE")
    for scenario in SCENARIOS:
        community = stable_ids[scenario]
        print(f"{scenario} stable community: {'C' + str(community) if community is not None else 'NA'}")
    print("\nStable-core Jaccard:")
    for row in stable_overlap.itertuples(index=False):
        value = "NA" if pd.isna(row.jaccard) else f"{row.jaccard:.3f}"
        print(f"{row.scenario_a}-{row.scenario_b} = {value}")
    print("\nCross-scenario ARI:")
    for row in cross_ari.itertuples(index=False):
        value = "NA" if pd.isna(row.ari) else f"{row.ari:.3f}"
        print(f"{row.scenario_a}-{row.scenario_b} = {value} "
              f"(n={row.n_patients_used}/{row.n_patients_total}; {row.fraction_total_used:.1%})")
    print("\nORDER TO REVIEW GRAPH REPRESENTATION SENSITIVITY")
    names = ["02_graph_S4_lab_group_mapping_audit.csv", "02_graph_feature_selection_audit.csv",
             "02_graph_bootstrap_replicate_audit.csv", "02_graph_bootstrap_qc_summary.csv",
             "02_graph_bootstrap_balance_audit.csv", "02_graph_common_intersection_ari.csv",
             "02_graph_stable_core_node_union_overlap.csv", "02_graph_interpretation_qc.json",
             "02_graph_scenario_feature_sets.csv", "02_graph_coverage_audit.csv",
             "02_graph_redundancy_audit.csv", "02_graph_S4_balance_weights.csv",
             "02_graph_representation_sensitivity_summary.csv",
             "02_graph_scenario_community_stability.csv",
             "02_graph_scenario_patient_membership.csv",
             "02_graph_scenario_community_pop_crosswalk.csv",
             "02_graph_cross_scenario_community_overlap.csv",
             "02_graph_stable_core_overlap.csv", "02_graph_cross_scenario_ari.csv",
             "PCA variance outputs",
             "02_graph_representation_sensitivity_mapper.png",
             "02_graph_representation_sensitivity_stability.png",
             "02_graph_stable_core_overlap.png"]
    for index, name in enumerate(names, 1):
        print(f"{index}. {name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    parser.add_argument("--graph-lab-candidates", type=Path,
                        default=common.STUDIES_TABLES_DIR / "pharma" / "00_profile_labs" / "00_pharma_graph_lab_candidates.csv")
    parser.add_argument("--baseline-metrics", type=Path,
                        default=common.INTEGRATED_BASELINE_CSV)
    parser.add_argument("--config", type=Path, default=folder / "config_sensitivity.yaml")
    parser.add_argument("--lab-group-map", type=Path, default=DEFAULT_LAB_GROUP_MAP,
                        help="CSV mapping canonical lab stems to S4 clinical laboratory groups")
    parser.add_argument("--dry-run", action="store_true", help="Generate tables but omit figures")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

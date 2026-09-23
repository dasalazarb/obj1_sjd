#!/usr/bin/env python3
"""Test whether baseline Mapper geometry is robust to representation choices.

Pop labels are external annotations only.  Every bootstrap replicate refits its
feature filtering, imputation, scaling/balancing, PCA, Mapper, and communities.
This analysis deliberately does not optimize Mapper or run longitudinal models.
"""
from __future__ import annotations

import argparse
import importlib.util
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
from src.studies._shared import create_study_dirs


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


def feature_stem(feature: str) -> str:
    """Return the lab concept before encoding suffixes."""
    return feature.split("__", 1)[0].strip().lower()


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
    mapping["lab"] = mapping["lab"].astype(str).str.strip().str.lower()
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
                        included: set[str] | None = None) -> pd.DataFrame:
    """Audit mapping of LAB features and reject any retained, unmapped feature."""
    indexed = mapping.set_index("lab")
    rows = []
    for feature in sorted(values):
        if families[feature] != "LAB":
            continue
        lab = feature_stem(feature)
        mapped = lab in indexed.index and indexed.at[lab, "mapping_status"] == "MAPPED"
        rows.append({"feature": feature, "lab": lab,
                     "s4_group": indexed.at[lab, "s4_group"] if mapped else "",
                     "s4_subgroup": indexed.at[lab, "s4_subgroup"] if mapped else "",
                     "mapped": mapped,
                     "included_S4": feature in included if included is not None else False})
    audit = pd.DataFrame(rows, columns=["feature", "lab", "s4_group", "s4_subgroup",
                                        "mapped", "included_S4"])
    bad = audit.loc[audit["included_S4"] & ~audit["mapped"], "feature"].tolist()
    if bad:
        raise ValueError(f"S4 retained LAB features lack a MAPPED clinical group: {bad}")
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


def scale_mfa_block(block: pd.DataFrame, block_name: str) -> tuple[np.ndarray, float]:
    medians = block.median(axis=0)
    if medians.isna().any():
        raise ValueError(f"{block_name} features lack a finite median: "
                         f"{medians[medians.isna()].index.tolist()}")
    scaled = RobustScaler().fit_transform(block.fillna(medians))
    singular = np.linalg.svd(scaled, full_matrices=False, compute_uv=False)
    sigma1 = float(singular[0]) if len(singular) else 0.0
    if not np.isfinite(sigma1) or sigma1 <= 0:
        raise ValueError(f"{block_name} block has no positive first singular value")
    return scaled / sigma1, sigma1


def _weight_existing_block(block: np.ndarray, block_name: str) -> tuple[np.ndarray, float]:
    singular = np.linalg.svd(block, full_matrices=False, compute_uv=False)
    sigma1 = float(singular[0]) if len(singular) else 0.0
    if not np.isfinite(sigma1) or sigma1 <= 0:
        raise ValueError(f"{block_name} block has no positive first singular value")
    return block / sigma1, sigma1


def prepare_matrix(raw: pd.DataFrame, families: dict[str, str], balance: bool,
                   lab_group_balance: bool = False,
                   lab_group_by_stem: dict[str, str] | None = None,
                   scenario: str = "") -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Fit median imputation/RobustScaler, optionally followed by MFA weights."""
    blocks, names, audit = [], [], []
    if lab_group_balance and not balance:
        raise ValueError("LAB-group balancing requires family balancing")
    if lab_group_balance and lab_group_by_stem is None:
        raise ValueError("S4 LAB-group balancing requires a lab group mapping")
    groups = FAMILY_ORDER if balance else ("ALL",)
    for family in groups:
        columns = list(raw) if family == "ALL" else [x for x in raw if families[x] == family]
        if not columns:
            continue
        if family == "LAB" and lab_group_balance:
            grouped: dict[str, list[str]] = {}
            for column in columns:
                stem = feature_stem(column)
                if stem not in lab_group_by_stem:
                    raise ValueError(f"S4 LAB feature has no mapped clinical group: {column}")
                grouped.setdefault(lab_group_by_stem[stem], []).append(column)
            nested, columns = [], []
            for group in sorted(grouped):
                group_columns = grouped[group]
                weighted, sigma1 = scale_mfa_block(raw[group_columns], f"LAB_GROUP:{group}")
                nested.append(weighted); columns.extend(group_columns)
                audit.append({"scenario": scenario, "level": "LAB_GROUP", "block": group,
                              "n_features": len(group_columns), "sigma1_before_weighting": sigma1,
                              "weight_applied": 1.0 / sigma1})
            scaled, sigma1 = _weight_existing_block(np.column_stack(nested), "LAB")
        elif balance:
            scaled, sigma1 = scale_mfa_block(raw[columns], family)
        else:
            medians = raw[columns].median(axis=0)
            if medians.isna().any():
                raise ValueError(f"Features lack a finite median: {medians[medians.isna()].index.tolist()}")
            scaled = RobustScaler().fit_transform(raw[columns].fillna(medians))
            sigma1 = np.nan
        if balance:
            audit.append({"scenario": scenario, "level": "FAMILY", "block": family,
                          "n_features": len(columns), "sigma1_before_weighting": sigma1,
                          "weight_applied": 1.0 / sigma1})
        blocks.append(scaled); names.extend(columns)
    audit_columns = ["scenario", "level", "block", "n_features",
                     "sigma1_before_weighting", "weight_applied"]
    return np.column_stack(blocks), names, pd.DataFrame(audit, columns=audit_columns)


def fit_embedding(raw: pd.DataFrame, families: dict[str, str], scenario: str,
                  config: dict,
                  lab_group_by_stem: dict[str, str] | None = None) -> Representation:
    transformed, names, balance_audit = prepare_matrix(
        raw, families, SCENARIOS[scenario]["balance"],
        SCENARIOS[scenario]["lab_group_balance"], lab_group_by_stem, scenario)
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


def bootstrap_stability(all_values: dict[str, pd.Series], families: dict[str, str],
                        manifest: pd.DataFrame, scenario: str,
                        reference_sets: list[set[int]], config: dict,
                        lab_group_by_stem: dict[str, str] | None = None) -> np.ndarray:
    settings = config["bootstrap"]
    replicates, n = int(settings["replicates"]), len(next(iter(all_values.values())))
    scores = np.zeros(len(reference_sets))
    if not reference_sets or not replicates:
        return scores
    rng = np.random.default_rng(int(config["random_seed"]))
    sample_n = max(3, min(n, math.floor(float(settings["patient_fraction"]) * n)))
    for _ in range(replicates):
        indices = np.sort(rng.choice(n, sample_n, replace=False))
        sampled = {name: series.iloc[indices].reset_index(drop=True)
                   for name, series in all_values.items()}
        try:
            selected, _, _ = select_features(sampled, families, scenario, config, manifest)
            raw = pd.DataFrame(selected)
            local_families = {name: families[name] for name in raw}
            # All learned S4 weights are deliberately refit within each replicate.
            fitted = fit_embedding(raw, local_families, scenario, config,
                                   lab_group_by_stem=lab_group_by_stem)
            result = analyze_mapper(fitted, config)
        except (ValueError, RuntimeError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
            logging.warning("Skipped %s bootstrap replicate: %s", scenario, exc)
            continue
        candidates = [set(indices[list(group)]) for group in result["community_sets"]]
        sampled_ids = set(indices)
        for community, reference in enumerate(reference_sets):
            ref = reference & sampled_ids
            best = max((len(ref & other) / len(ref | other) if ref | other else 0.0
                        for other in candidates), default=0.0)
            scores[community] += best >= float(settings["branch_match_jaccard"])
    return scores / replicates


def patient_membership(scenario: str, baseline: pd.DataFrame, mapped: dict) -> pd.DataFrame:
    """Create the common patient-level assignment contract for one scenario."""
    hard, probabilities = mapped["hard"], mapped["probabilities"]
    result = pd.DataFrame({
        "scenario": scenario, "patient_id": baseline["patient_id"].to_numpy(),
        "mapper_covered": mapped["covered"], "community_membership_tie": mapped["ties"],
        "hard_community": pd.array(hard, dtype="Int64"),
        "community_supported": [bool(mapped["supported"][int(value)])
                                if np.isfinite(value) else False for value in hard],
        "max_community_membership": (probabilities.max(axis=1)
                                     if probabilities.shape[1] else np.nan),
        "baseline_pop": baseline["pop_status"].to_numpy(),
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
    return pd.DataFrame(overlap_rows), pd.DataFrame(stable_rows), pd.DataFrame(ari_rows)


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
    axis.axhline(float(next(iter(results.values()))["config"]["bootstrap"]["stable_community_threshold"]),
                 color="#333333", linestyle="--", label="stable threshold")
    axis.set(ylabel="Bootstrap match proportion", ylim=(0, 1.05), title="Community stability by representation")
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


def interpretation_check(summary: pd.DataFrame) -> str:
    """Transparent QC only; never a scientific scenario-selection rule."""
    cleaned = summary.loc[summary.scenario.isin(["S1", "S2", "S3", "S4"])]
    connected = int(cleaned.n_connected_components.eq(1).sum())
    has_stable = int(cleaned.n_stable_communities.ge(1).sum())
    separated = int(cleaned.n_stable_communities.gt(1).sum())
    if connected >= 2 and has_stable >= 2 and separated < 2:
        return "YES"
    if connected == 0 and separated >= 2:
        return "NO"
    return "MIXED"


def run(args: argparse.Namespace) -> None:
    config = CANON.load_config(args.config)
    lab_group_frame = load_lab_group_map(args.lab_group_map)
    lab_group_by_stem = lab_group_lookup(lab_group_frame)
    dirs = create_study_dirs("graph/02_graph_representation_sensitivity")
    CANON._configure_logging(dirs["logs"] / "02_graph_representation_sensitivity.log")
    master, candidates, _ = CANON.load_real_inputs(args.integrated, args.graph_lab_candidates)
    baseline = CANON.select_baseline(master)
    age = CANON.load_age_at_diagnosis(args.baseline_metrics)
    baseline = baseline.merge(age, on="patient_id", how="left", validate="one_to_one", suffixes=("", "_blockA"))
    if "age_dx_blockA" in baseline:
        baseline["age_dx"] = baseline.pop("age_dx_blockA")

    # A zero threshold obtains the complete eligible universe. S0 is then rebuilt
    # with the canonical threshold to reproduce current coverage exceptions exactly.
    full_manifest, full_values, _, _ = CANON.build_feature_manifest(baseline, candidates, 0.0)
    current_manifest, current_values, _, _ = CANON.build_feature_manifest(
        baseline, candidates, float(CANON.load_config(Path(__file__).with_name("config.yaml"))["feature_selection"]["minimum_baseline_coverage"]))
    full_families = {row.feature: family_name(row.family)
                     for row in full_manifest.loc[full_manifest.included].itertuples()}
    current_families = {name: full_families[name] for name in current_values}
    lab_group_index = lab_group_frame.set_index("lab")
    results, memberships, stable_ids = {}, {}, {}
    summary_rows, coverage_rows, redundancy_rows, feature_rows, stability_rows = [], [], [], [], []
    s4_mapping_audit: pd.DataFrame | None = None

    for scenario in SCENARIOS:
        universe = current_values if scenario == "S0" else full_values
        families = current_families if scenario == "S0" else full_families
        selected, coverage, redundancy = select_features(universe, families, scenario, config, full_manifest)
        coverage.insert(0, "scenario", scenario); coverage_rows.append(coverage)
        redundancy.insert(0, "scenario", scenario); redundancy_rows.append(redundancy)
        raw = pd.DataFrame(selected, index=baseline.index)
        final_families = {name: families[name] for name in raw}
        if scenario == "S4":
            s4_mapping_audit = validate_s4_mapping(
                universe, families, lab_group_frame, included=set(raw))
        for feature in sorted(universe):
            lab = feature_stem(feature) if families[feature] == "LAB" else ""
            map_row = lab_group_index.loc[lab] if lab and lab in lab_group_index.index else None
            feature_rows.append({"scenario": scenario, "feature": feature,
                                 "family": families[feature], "included": feature in raw,
                                 "lab": lab,
                                 "s4_group": (map_row["s4_group"] if map_row is not None else ""),
                                 "s4_subgroup": (map_row["s4_subgroup"] if map_row is not None else ""),
                                 "exclusion_stage": "" if feature in raw else
                                 ("redundancy" if feature in set(redundancy.feature_removed) else "coverage_or_encoding")})
        rep = fit_embedding(raw, final_families, scenario, config,
                            lab_group_by_stem=lab_group_by_stem)
        mapped = analyze_mapper(rep, config)
        stability = bootstrap_stability(universe, families, full_manifest, scenario,
                                        mapped["community_sets"], config,
                                        lab_group_by_stem=lab_group_by_stem)
        stable_threshold = float(config["bootstrap"]["stable_community_threshold"])
        for community, score in enumerate(stability):
            stability_rows.append({"scenario": scenario, "community": community,
                                   "n_patients_node_union": len(mapped["community_sets"][community]),
                                   "bootstrap_stability": score,
                                   "stability_status": "stable" if score >= stable_threshold else "provisional"})
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
        pop = baseline["pop_status"].to_numpy()
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
            "n_supported_communities": int(supported.sum()), "n_stable_communities": int((stability >= stable_threshold).sum()),
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
    coverage_all = pd.concat(coverage_rows, ignore_index=True)
    redundancy_all = pd.concat(redundancy_rows, ignore_index=True)
    stability_all = pd.DataFrame(stability_rows)
    membership_all = pd.concat(memberships.values(), ignore_index=True, sort=False)
    crosswalk = community_pop_crosswalk(membership_all, stability_all)
    overlap, stable_overlap, cross_ari = cross_scenario_tables(memberships, stable_ids)
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
              f"\nStable: {row.n_stable_communities}\nARI vs Pop: {row.ari_vs_pop:.3g}")
    print("\nINTERPRETATION CHECK")
    print("Does the stable-core + peripheral-continuum pattern persist across cleaned representations?")
    print(interpretation_check(summary))
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
        print(f"{row.scenario_a}-{row.scenario_b} = {value} (n={row.n_patients_used})")
    print("\nORDER TO REVIEW GRAPH REPRESENTATION SENSITIVITY")
    names = ["02_graph_S4_lab_group_mapping_audit.csv",
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
                        default=common.BLOCKA_INTERMEDIATE_DATA_DIR / "01_table1_baseline" /
                        "01_table1_from_clinical_episode_spine_sjd__baseline_patient_metrics_after_eligibility.csv")
    parser.add_argument("--config", type=Path, default=folder / "config_sensitivity.yaml")
    parser.add_argument("--lab-group-map", type=Path, default=DEFAULT_LAB_GROUP_MAP,
                        help="CSV mapping canonical lab stems to S4 clinical laboratory groups")
    parser.add_argument("--dry-run", action="store_true", help="Generate tables but omit figures")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

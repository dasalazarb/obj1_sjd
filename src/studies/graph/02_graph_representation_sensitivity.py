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
SCENARIOS = {
    "S0": {"coverage": False, "redundancy": False, "balance": False},
    "S1": {"coverage": True, "redundancy": False, "balance": False},
    "S2": {"coverage": True, "redundancy": True, "balance": False},
    "S3": {"coverage": True, "redundancy": True, "balance": True},
}
FAMILY_ORDER = ("LAB", "PRO", "CLINICAL")


class Representation:
    def __init__(self, raw: pd.DataFrame, families: dict[str, str],
                 embedding: np.ndarray, pca: PCA,
                 transformed_features: list[str]) -> None:
        self.raw = raw
        self.families = families
        self.embedding = embedding
        self.pca = pca
        self.transformed_features = transformed_features


def family_name(value: str) -> str:
    return "LAB" if value == "lab" else "PRO" if value == "pro" else "CLINICAL"


def feature_stem(feature: str) -> str:
    """Return the lab concept before encoding suffixes."""
    return feature.split("__", 1)[0].strip().lower()


def drop_reference_one_hot(values: dict[str, pd.Series], manifest: pd.DataFrame
                           ) -> tuple[dict[str, pd.Series], set[str]]:
    """Use k-1 columns per categorical lab while preserving NaN rows."""
    result, removed = dict(values), set()
    included = manifest.loc[manifest["included"].astype(bool)]
    categorical = included.loc[included.representation_type.eq("categorical_one_hot")]
    for _, group in categorical.groupby("lab", sort=True):
        columns = sorted(set(group.feature) & set(result))
        if columns:
            removed.add(columns[0])  # deterministic lexicographic reference category
            result.pop(columns[0], None)
    return result, removed


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
        for kept, removed in zip(first, second):
            pair = pd.concat([remaining[kept], remaining[removed]], axis=1).dropna()
            rho = float(pair.corr(method="spearman").iloc[0, 1]) if len(pair) >= 2 else np.nan
            remove(kept, removed, rho, len(pair),
                   "conceptual_or_deterministic_pair; configured_primary_kept")

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
    reference_removed: set[str] = set()
    if settings["coverage"]:
        values, reference_removed = drop_reference_one_hot(values, manifest)
        values, audit = coverage_filter(values, families, config)
        for feature in sorted(reference_removed):
            audit.loc[len(audit)] = [feature, families[feature],
                                     float(all_values[feature].notna().mean()), False,
                                     "one_hot_reference_category_k_minus_1"]
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


def prepare_matrix(raw: pd.DataFrame, families: dict[str, str], balance: bool
                  ) -> tuple[np.ndarray, list[str]]:
    """Fit median imputation/RobustScaler, optionally followed by MFA weights."""
    blocks, names = [], []
    groups = FAMILY_ORDER if balance else ("ALL",)
    for family in groups:
        columns = list(raw) if family == "ALL" else [x for x in raw if families[x] == family]
        if not columns:
            continue
        block = raw[columns]
        medians = block.median(axis=0)
        if medians.isna().any():
            raise ValueError(f"Features lack a finite median: {medians[medians.isna()].index.tolist()}")
        scaled = RobustScaler().fit_transform(block.fillna(medians))
        if balance:
            singular = np.linalg.svd(scaled, full_matrices=False, compute_uv=False)
            first = float(singular[0]) if len(singular) else 0.0
            if not np.isfinite(first) or first <= 0:
                raise ValueError(f"{family} block has no positive first singular value")
            scaled = scaled / first
        blocks.append(scaled); names.extend(columns)
    return np.column_stack(blocks), names


def fit_embedding(raw: pd.DataFrame, families: dict[str, str], scenario: str,
                  config: dict) -> Representation:
    transformed, names = prepare_matrix(raw, families, SCENARIOS[scenario]["balance"])
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
    return Representation(raw, families, pca.transform(transformed), pca, names)


def analyze_mapper(rep: Representation, config: dict) -> dict:
    mapper = config["mapper"]
    eps = CANON.eps_kdist(rep.embedding, int(mapper["eps_k_neighbors"]),
                          float(mapper["eps_percentile"]))
    graph = CANON.run_mapper(rep.embedding[:, :2], rep.embedding, config, eps)
    component_counts, components, nodes, _ = CANON.branches_from_nerve(
        graph, len(rep.raw), int(mapper["minimum_node_support"]), 0)
    nerve, communities, counts, modularity = CANON.mapper_communities(
        graph, nodes, len(rep.raw))
    _, covered, hard = CANON.soft_membership(counts, float(mapper["soft_membership_tau"]))
    ties = covered & np.isnan(hard)
    community_sets = [set().union(*(nodes[node] for node in group)) for group in communities]
    minimum = int(config["community_detection"]["minimum_patients"])
    supported = np.array([len(group) >= minimum for group in community_sets])
    return {"graph": graph, "nerve": nerve, "nodes": nodes, "components": components,
            "communities": communities, "community_sets": community_sets,
            "modularity": modularity, "covered": covered, "hard": hard,
            "ties": ties, "supported": supported, "component_counts": component_counts}


def bootstrap_stability(all_values: dict[str, pd.Series], families: dict[str, str],
                        manifest: pd.DataFrame, scenario: str,
                        reference_sets: list[set[int]], config: dict) -> np.ndarray:
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
            fitted = fit_embedding(raw, local_families, scenario, config)
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
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
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


def interpretation_check(summary: pd.DataFrame) -> str:
    """Transparent QC only; never a scientific scenario-selection rule."""
    cleaned = summary.loc[summary.scenario.isin(["S1", "S2", "S3"])]
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
    results, summary_rows, coverage_rows, redundancy_rows, feature_rows, stability_rows = {}, [], [], [], [], []

    for scenario in SCENARIOS:
        universe = current_values if scenario == "S0" else full_values
        families = current_families if scenario == "S0" else full_families
        selected, coverage, redundancy = select_features(universe, families, scenario, config, full_manifest)
        coverage.insert(0, "scenario", scenario); coverage_rows.append(coverage)
        redundancy.insert(0, "scenario", scenario); redundancy_rows.append(redundancy)
        raw = pd.DataFrame(selected, index=baseline.index)
        final_families = {name: families[name] for name in raw}
        for feature in sorted(universe):
            feature_rows.append({"scenario": scenario, "feature": feature,
                                 "family": families[feature], "included": feature in raw,
                                 "exclusion_stage": "" if feature in raw else
                                 ("redundancy" if feature in set(redundancy.feature_removed) else "coverage_or_encoding")})
        rep = fit_embedding(raw, final_families, scenario, config)
        mapped = analyze_mapper(rep, config)
        stability = bootstrap_stability(universe, families, full_manifest, scenario,
                                        mapped["community_sets"], config)
        stable_threshold = float(config["bootstrap"]["stable_community_threshold"])
        for community, score in enumerate(stability):
            stability_rows.append({"scenario": scenario, "community": community,
                                   "n_patients_node_union": len(mapped["community_sets"][community]),
                                   "bootstrap_stability": score,
                                   "stability_status": "stable" if score >= stable_threshold else "provisional"})
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
            "modularity": mapped["modularity"], "n_unambiguous": unambiguous,
            "pct_unambiguous": unambiguous / len(baseline), "ari_vs_pop": ari,
            "ari_permutation_p": ari_p, "ari_n_used": int(valid.sum()),
            "h1_total_persistence": ph.get("total_h1_persistence", np.nan),
            "h1_permutation_p": ph.get("permutation_p_value", np.nan)})
        results[scenario] = {"rep": rep, "mapper": mapped, "config": config}

    summary = pd.DataFrame(summary_rows)
    coverage_all = pd.concat(coverage_rows, ignore_index=True)
    redundancy_all = pd.concat(redundancy_rows, ignore_index=True)
    stability_all = pd.DataFrame(stability_rows)
    coverage_all.to_csv(dirs["tables"] / "02_graph_coverage_audit.csv", index=False)
    redundancy_all.to_csv(dirs["tables"] / "02_graph_redundancy_audit.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(dirs["tables"] / "02_graph_scenario_feature_sets.csv", index=False)
    stability_all.to_csv(dirs["tables"] / "02_graph_scenario_community_stability.csv", index=False)
    summary.to_csv(dirs["tables"] / "02_graph_representation_sensitivity_summary.csv", index=False)
    if not args.dry_run:
        draw_figures(results, stability_all, dirs["figures"])

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
    print("\nORDER TO REVIEW GRAPH REPRESENTATION SENSITIVITY")
    names = ["02_graph_coverage_audit.csv", "02_graph_redundancy_audit.csv",
             "02_graph_scenario_feature_sets.csv", "02_graph_representation_sensitivity_summary.csv",
             "02_graph_scenario_community_stability.csv", "PCA variance outputs",
             "02_graph_representation_sensitivity_mapper.png",
             "02_graph_representation_sensitivity_stability.png"]
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
    parser.add_argument("--dry-run", action="store_true", help="Generate tables but omit figures")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

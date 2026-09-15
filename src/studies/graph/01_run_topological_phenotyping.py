#!/usr/bin/env python3
"""Discover baseline patient structure with PCA and Mapper on the real master.

Pop, ESSDAI and ESSPRI are deliberately annotations/outcomes rather than inputs
to discovery. Mapper settings are initial descriptive settings, not optimized
parameters for this cohort.
"""
from __future__ import annotations

import argparse
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
import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import statsmodels.formula.api as smf

import common
from src.studies._shared import (benjamini_hochberg, create_study_dirs,
                                 load_parquet, validate_integrated_dataset,
                                 write_json)

POPS = ("Pop1", "Pop2", "Pop3")
PROS = ("sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global")
OUTCOMES = ("essdai_total", "esspri_total_observed", "sf36_pcs",
            "profad_total", "mdafs_global")
ADMIN = ("protocol", "ids__protocol", "ids__protocol_number", "parent_protocol")
AGES = ("ids__age_at_visit", "age_at_visit", "age")
SEXES = ("ids__sex", "ids__gender", "sex", "gender")
IDS = {"patient_id", "clinical_episode_id", "clinical_baseline_episode_id",
       "previous_clinical_episode_id"}
STRUCTURAL = {"clinical_anchor_date", "clinical_baseline_date", "episode_start_date",
              "episode_end_date", "clinical_visit", "clinical_visit_number", "visit_type",
              "is_clinical_baseline", "is_last_clinical_visit", "integration_version",
              "integration_run_date"}
EXACT_POP = {"pop_status", "previous_pop_status", "essdai_total", "esspri_total",
             "esspri_total_observed", "previous_essdai_total", "previous_esspri_total",
             "delta_essdai_from_previous", "delta_esspri_from_previous"}
COLORS = ["#2E5A87", "#C1666B", "#5B8C5A", "#B08B4F", "#6C6C8C",
          "#7A5C8E", "#3C8DAD", "#C77C3B"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the Graph configuration") from exc
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Graph configuration must be a mapping")
    return config


def load_real_inputs(integrated: Path, lab_profile: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    master = load_parquet(integrated)
    contract = validate_integrated_dataset(master)
    profile = pd.read_csv(lab_profile)
    required = {"recommended_use", "value_column"}
    missing = required - set(profile)
    if missing:
        raise ValueError(f"Laboratory profile missing required columns: {sorted(missing)}")
    return master, profile, contract


def select_baseline(master: pd.DataFrame) -> pd.DataFrame:
    baseline = master.loc[master["is_clinical_baseline"].fillna(False).astype(bool)].copy()
    if baseline["patient_id"].duplicated().any():
        raise ValueError("Clinical baseline must contain at most one row per patient")
    return baseline.reset_index(drop=True)


def _exclusion(column: str) -> tuple[str, str]:
    """Return simple manifest family and a leakage-safe exclusion reason."""
    low = column.lower()
    if column in IDS:
        return "structural", "identifier"
    if column in STRUCTURAL:
        return "structural", "visit_or_date_structure"
    if column in ADMIN or any(x in low for x in ("protocol_number", "site_id", "center_id")):
        return "administrative", "administrative"
    esspri_component = low in {"dryness", "fatigue", "pain"} or (
        "esspri" in low and any(x in low for x in ("dry", "fatigue", "pain")))
    domain_names = ("constitutional", "lymphadenopathy", "articular", "cutaneous",
                    "pulmonary", "renal", "muscular", "pns", "cns", "hematolog",
                    "biological")
    essdai_component = (("essdai" in low and any(x in low for x in ("domain", "score", "active")))
                        or (any(x in low for x in domain_names)
                            and (low.endswith("_domain_score") or low.endswith("_active"))))
    pop_derived = low.startswith("pop_") or low.endswith("_pop") or "pop_status" in low
    if column in EXACT_POP or pop_derived or esspri_component or essdai_component:
        return "pop_essdai_esspri", "pop_essdai_esspri_leakage"
    if low.startswith(("time_since_", "time_from_previous_", "previous_", "delta_", "next_", "to_")):
        return "longitudinal_derived", "longitudinal_or_future"
    suffixes = ("_n_measurements", "_days_from_anchor", "_conflict", "_selection_status",
                "_episode_status", "_measurement_date", "_unit", "_text")
    if low.startswith("has_") or low in {"n_integrated_blocks_available", "n_clinical_visits_patient"} or low.endswith(suffixes):
        return "metadata", "availability_qc_or_metadata"
    # Every laboratory result is governed by the Pharma profile.  The selected
    # numeric columns are overridden to ``lab`` in build_feature_manifest.
    if low.endswith("__value"):
        return "lab", "laboratory_not_recommended_numeric_longitudinal"
    if column in PROS:
        return "pro", ""
    if "overlap" in low or low.startswith("ov_"):
        return "overlap", ""
    if "gland" in low:
        return "glandular", ""
    return "other_clinical", ""


def _numeric_candidate(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("Float64")
    numeric = pd.to_numeric(series, errors="coerce")
    # Do not treat incidental numeric-looking identifiers/categories as clinical measures.
    observed = int(series.notna().sum())
    return numeric if observed and int(numeric.notna().sum()) == observed else None


def build_feature_manifest(baseline: pd.DataFrame, profile: pd.DataFrame,
                           minimum_coverage: float) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, str]]:
    lab_rows = profile.loc[profile["recommended_use"].eq("numeric_longitudinal")].copy()
    allowed_labs: dict[str, str] = {}
    lab_names: dict[str, str] = {}
    for row in lab_rows.itertuples(index=False):
        value = getattr(row, "value_column", None)
        if pd.notna(value) and str(value).strip() in baseline.columns:
            column = str(value).strip()
            allowed_labs[column] = str(getattr(row, "lab", column))
            lab_names[str(getattr(row, "lab", column)).lower()] = column

    rows, values = [], {}
    for column in baseline.columns:
        family, reason = _exclusion(column)
        source = "integrated_master"
        numeric = None
        if column in allowed_labs:
            family, source = "lab", "pharma_lab_profile"
            reason = ""
            numeric = pd.to_numeric(baseline[column], errors="coerce")
        elif not reason:
            numeric = _numeric_candidate(baseline[column])
        nonmissing = int(numeric.notna().sum()) if numeric is not None else int(baseline[column].notna().sum())
        pct = nonmissing / len(baseline) if len(baseline) else 0.0
        unique = int(numeric.nunique(dropna=True)) if numeric is not None else int(baseline[column].nunique(dropna=True))
        if not reason and numeric is None:
            reason = "not_numeric_or_boolean"
        elif not reason and pct < minimum_coverage:
            reason = "below_minimum_baseline_coverage"
        elif not reason and unique < 2:
            reason = "no_baseline_variability"
        included = not reason
        if included:
            values[column] = numeric.astype(float)
        rows.append({"feature": column, "family": family, "source": source,
                     "dtype": str(baseline[column].dtype), "n_nonmissing_baseline": nonmissing,
                     "pct_nonmissing_baseline": pct, "n_unique_baseline": unique,
                     "included_in_embedding": included, "exclusion_reason": reason})
    return pd.DataFrame(rows), values, lab_names


def build_baseline_matrix(values: dict[str, pd.Series]) -> tuple[pd.DataFrame, int, float]:
    matrix = pd.DataFrame(values)
    if matrix.shape[1] < 2:
        raise ValueError(f"Only {matrix.shape[1]} features met the configured coverage and leakage rules; at least 2 are required for PCA. Review 01_graph_feature_manifest.csv and change config.yaml manually if scientifically justified.")
    n_missing = int(matrix.isna().sum().sum())
    medians = matrix.median(axis=0)
    if medians.isna().any():
        raise ValueError("At least one included feature has no finite baseline median")
    filled = matrix.fillna(medians)
    pct = n_missing / matrix.size if matrix.size else 0.0
    return filled, n_missing, pct


def fit_patient_embedding(matrix: pd.DataFrame, config: dict) -> tuple[np.ndarray, PCA, RobustScaler]:
    scaled_by = RobustScaler().fit(matrix)
    scaled = scaled_by.transform(matrix)
    limit = min(int(config["pca"]["max_components"]), scaled.shape[0], scaled.shape[1])
    if limit < 2:
        raise ValueError("At least two PCA dimensions are required for the Mapper lens")
    probe = PCA(n_components=limit, random_state=int(config["random_seed"])).fit(scaled)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    hits = np.flatnonzero(cumulative >= float(config["pca"]["variance_target"]))
    dimensions = int(hits[0] + 1) if len(hits) else limit
    dimensions = max(2, dimensions)
    pca = PCA(n_components=dimensions, random_state=int(config["random_seed"])
              ).fit(scaled)
    return pca.transform(scaled), pca, scaled_by


def eps_kdist(data: np.ndarray, k: int = 4, percentile: float = 65) -> float:
    if len(data) < 2:
        raise ValueError("At least two patients are required to estimate DBSCAN epsilon")
    neighbors = min(int(k) + 1, len(data))
    distances, _ = NearestNeighbors(n_neighbors=neighbors).fit(data).kneighbors(data)
    eps = float(np.percentile(distances[:, -1], percentile))
    if not np.isfinite(eps) or eps <= 0:
        positive = distances[:, -1][distances[:, -1] > 0]
        if not len(positive):
            raise ValueError("DBSCAN epsilon is zero; patient embeddings are indistinguishable")
        eps = float(np.median(positive))
    return eps


def run_mapper(lens: np.ndarray, cluster_space: np.ndarray, config: dict, eps: float) -> dict:
    try:
        import kmapper as km
    except ImportError as exc:
        raise RuntimeError("kmapper is required for the core Graph analysis") from exc
    settings = config["mapper"]
    return km.KeplerMapper(verbose=0).map(
        lens, cluster_space,
        cover=km.Cover(n_cubes=int(settings["n_cubes"]),
                       perc_overlap=float(settings["perc_overlap"])),
        clusterer=DBSCAN(eps=eps, min_samples=int(settings["min_samples"]),
                         metric="euclidean"))


def branches_from_nerve(graph: dict, n_patients: int, minimum_node_support: int,
                        minimum_branch_patients: int) -> tuple[np.ndarray, list[list[str]], dict, dict]:
    nodes = {node: set(map(int, members)) for node, members in graph.get("nodes", {}).items()
             if len(members) >= minimum_node_support}
    adjacency = {node: set() for node in nodes}
    for node, linked in graph.get("links", {}).items():
        if node not in nodes:
            continue
        for other in linked:
            if other in nodes:
                adjacency[node].add(other); adjacency[other].add(node)
    seen, all_components = set(), []
    for node in nodes:
        if node in seen:
            continue
        stack, component = [node], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current); component.append(current)
            stack.extend(adjacency[current] - seen)
        all_components.append(component)
    all_components.sort(key=lambda c: -len(set().union(*(nodes[x] for x in c))))
    retained, discarded_patients, discarded_nodes = [], set(), 0
    for component in all_components:
        patients = set().union(*(nodes[x] for x in component))
        if len(patients) >= minimum_branch_patients:
            retained.append(component)
        else:
            discarded_patients.update(patients); discarded_nodes += len(component)
    counts = np.zeros((n_patients, len(retained)), dtype=float)
    for branch, component in enumerate(retained):
        for node in component:
            counts[list(nodes[node]), branch] += 1
    details = {"nodes_before_support_pruning": len(graph.get("nodes", {})),
               "nodes_after_support_pruning": len(nodes),
               "discarded_small_component_nodes": discarded_nodes,
               "discarded_small_component_patients": len(discarded_patients)}
    return counts, retained, nodes, details


def soft_membership(counts: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, branches = counts.shape
    covered = counts.sum(axis=1) > 0
    probabilities = np.full((n, branches), np.nan)
    hard = np.full(n, np.nan)
    if branches:
        probabilities[~covered] = 1.0 / branches
        logits = counts[covered] / float(tau)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities[covered] = exp / exp.sum(axis=1, keepdims=True)
        hard[covered] = probabilities[covered].argmax(axis=1)
    return probabilities, covered, hard


def bootstrap_stability(raw_matrix: pd.DataFrame, reference_sets: list[set[int]],
                        config: dict) -> np.ndarray:
    boot = config["bootstrap"]; mapper = config["mapper"]
    replicates = int(boot["replicates"]); scores = np.zeros(len(reference_sets))
    if not len(reference_sets) or not replicates:
        return scores
    rng = np.random.default_rng(int(config["random_seed"])); n = len(raw_matrix)
    sample_n = max(2, min(n, int(math.floor(float(boot["patient_fraction"]) * n))))
    for _ in range(replicates):
        indices = np.sort(rng.choice(n, size=sample_n, replace=False))
        sample = raw_matrix.iloc[indices]
        sample = sample.fillna(sample.median(axis=0))
        scaled = RobustScaler().fit_transform(sample)
        limit = min(int(config["pca"]["max_components"]), *scaled.shape)
        if limit < 2:
            continue
        probe = PCA(n_components=limit, random_state=int(config["random_seed"])).fit(scaled)
        cumulative = np.cumsum(probe.explained_variance_ratio_)
        hit = np.flatnonzero(cumulative >= float(config["pca"]["variance_target"]))
        dimensions = max(2, int(hit[0] + 1) if len(hit) else limit)
        embedded = PCA(n_components=dimensions, random_state=int(config["random_seed"])).fit_transform(scaled)
        try:
            eps = eps_kdist(embedded, int(mapper["eps_k_neighbors"]), float(mapper["eps_percentile"]))
            graph = run_mapper(embedded[:, :2], embedded, config, eps)
            counts, _, _, _ = branches_from_nerve(
                graph, sample_n, int(mapper["minimum_node_support"]),
                int(mapper["minimum_branch_patients"]))
        except (ValueError, RuntimeError):
            continue
        sampled_sets = [set(indices[np.flatnonzero(counts[:, c] > 0)]) for c in range(counts.shape[1])]
        sampled_reference = [reference & set(indices) for reference in reference_sets]
        for branch, reference in enumerate(sampled_reference):
            best = max((len(reference & candidate) / len(reference | candidate)
                        if reference | candidate else 0.0 for candidate in sampled_sets), default=0.0)
            scores[branch] += best >= float(boot["branch_match_jaccard"])
    return scores / replicates


def ari_permutation(branch: np.ndarray, pop: np.ndarray, replicates: int,
                    seed: int) -> tuple[float, float]:
    if len(branch) < 2 or len(np.unique(branch)) < 1 or len(np.unique(pop)) < 1:
        return np.nan, np.nan
    observed = float(adjusted_rand_score(pop, branch))
    rng = np.random.default_rng(seed)
    null = np.array([adjusted_rand_score(rng.permutation(pop), branch)
                     for _ in range(replicates)])
    return observed, float((1 + np.sum(null >= observed)) / (replicates + 1))


def branch_pop_crosswalk(membership: pd.DataFrame) -> pd.DataFrame:
    valid = membership.mapper_covered & membership.baseline_pop.isin(POPS)
    frame = membership.loc[valid, ["hard_branch", "baseline_pop"]]
    if frame.empty:
        return pd.DataFrame(columns=["hard_branch", "baseline_pop", "n_patients", "pct_within_branch"])
    table = frame.groupby(["hard_branch", "baseline_pop"], observed=True).size().rename("n_patients").reset_index()
    table["pct_within_branch"] = table.n_patients / table.groupby("hard_branch").n_patients.transform("sum")
    return table


def administrative_sensitivity(baseline: pd.DataFrame, membership: pd.DataFrame) -> dict:
    column = next((x for x in ADMIN if x in baseline), None)
    if not column:
        logging.info("Administrative sensitivity omitted: no protocol column is available")
        return {"status": "omitted", "reason": "no protocol column available"}
    frame = membership[["patient_id", "mapper_covered", "hard_branch"]].merge(
        baseline[["patient_id", column]], on="patient_id", validate="one_to_one")
    frame = frame.loc[frame.mapper_covered].dropna(subset=["hard_branch", column])
    contingency = pd.crosstab(frame.hard_branch, frame[column])
    if contingency.empty or min(contingency.shape) < 2:
        value = np.nan
    else:
        chi, _, _, _ = chi2_contingency(contingency)
        value = math.sqrt(chi / (contingency.to_numpy().sum() * (min(contingency.shape) - 1)))
    records = []
    for branch, row in contingency.iterrows():
        total = row.sum()
        records.extend({"hard_branch": int(branch), "protocol": str(protocol),
                        "n_patients": int(count), "pct_within_branch": float(count / total)}
                       for protocol, count in row.items() if count)
    return {"status": "completed", "protocol_column": column, "cramers_v": value,
            "crosswalk": records}


def branch_characterization(baseline: pd.DataFrame, membership: pd.DataFrame,
                            features: list[str], lab_names: dict[str, str]) -> pd.DataFrame:
    extras = [lab_names[key] for key in ("protein_total", "mcv", "bun", "lymphocyte_count")
              if key in lab_names]
    columns = list(dict.fromkeys(features + extras))
    data = membership[["patient_id", "mapper_covered", "hard_branch"]].merge(
        baseline[["patient_id", *columns]], on="patient_id", validate="one_to_one")
    data = data.loc[data.mapper_covered]
    rows = []
    for branch, group in data.groupby("hard_branch"):
        for feature in columns:
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            rows.append({"hard_branch": int(branch), "feature": feature, "n": len(values),
                         "median": values.median(), "q1": values.quantile(.25),
                         "q3": values.quantile(.75), "mean": values.mean(),
                         "sd": values.std()})
    return pd.DataFrame(rows, columns=["hard_branch", "feature", "n", "median", "q1", "q3", "mean", "sd"])


def _baseline_covariate(baseline: pd.DataFrame, choices: tuple[str, ...]) -> str | None:
    return next((column for column in choices if column in baseline), None)


def fit_longitudinal_models(master: pd.DataFrame, baseline: pd.DataFrame,
                            membership: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    result_columns = ["outcome", "n_observations", "n_patients", "n_branches", "model",
                      "interaction_lr", "interaction_df", "interaction_p_value",
                      "interaction_q_value", "model_status"]
    if "time_since_clinical_baseline_years" not in master:
        logging.warning("Longitudinal models omitted: time_since_clinical_baseline_years unavailable")
        return pd.DataFrame(columns=result_columns), pd.DataFrame(), []
    assigned = membership.loc[membership.mapper_covered, ["patient_id", "hard_branch"]]
    long = master.merge(assigned, on="patient_id", how="inner", validate="many_to_one")
    long["time"] = pd.to_numeric(long["time_since_clinical_baseline_years"], errors="coerce")
    age, sex = _baseline_covariate(baseline, AGES), _baseline_covariate(baseline, SEXES)
    covars = ["patient_id"] + [x for x in (age, sex) if x]
    base_covars = baseline[covars].rename(columns={age: "baseline_age", sex: "baseline_sex"})
    long = long.merge(base_covars, on="patient_id", how="left", validate="many_to_one")
    rows, plotted = [], []
    for outcome in OUTCOMES:
        if outcome not in long:
            rows.append({"outcome": outcome, "n_observations": 0, "n_patients": 0,
                         "n_branches": 0, "model": "", "interaction_lr": np.nan,
                         "interaction_df": np.nan, "interaction_p_value": np.nan,
                         "interaction_q_value": np.nan,
                         "model_status": "outcome_not_available"})
            continue
        keep = ["patient_id", "hard_branch", "time", outcome]
        if age: keep.append("baseline_age")
        if sex: keep.append("baseline_sex")
        data = long[keep].copy(); data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
        if age: data["baseline_age"] = pd.to_numeric(data.baseline_age, errors="coerce")
        data = data.dropna(); n_branches = data.hard_branch.nunique()
        adjustment = (["baseline_age"] if age else []) + (["C(baseline_sex)"] if sex else [])
        rhs_full = "C(hard_branch) * time" + (" + " + " + ".join(adjustment) if adjustment else "")
        rhs_reduced = "C(hard_branch) + time" + (" + " + " + ".join(adjustment) if adjustment else "")
        model_name = f"{outcome} ~ {rhs_full}; random intercept patient"
        row = {"outcome": outcome, "n_observations": len(data),
               "n_patients": data.patient_id.nunique(), "n_branches": n_branches,
               "model": model_name, "interaction_lr": np.nan, "interaction_df": np.nan,
               "interaction_p_value": np.nan, "interaction_q_value": np.nan,
               "model_status": "insufficient_information"}
        if len(data) >= 10 and data.patient_id.nunique() >= 3 and n_branches >= 2 and data.time.nunique() >= 2:
            try:
                full = smf.mixedlm(f"Q('{outcome}') ~ {rhs_full}", data,
                                   groups=data.patient_id, re_formula="~1").fit(reml=False, method="lbfgs")
                reduced = smf.mixedlm(f"Q('{outcome}') ~ {rhs_reduced}", data,
                                      groups=data.patient_id, re_formula="~1").fit(reml=False, method="lbfgs")
                lr = max(0.0, 2 * (full.llf - reduced.llf)); dfree = n_branches - 1
                row.update(interaction_lr=lr, interaction_df=dfree,
                           interaction_p_value=float(chi2.sf(lr, dfree)), model_status="estimated")
                if len(plotted) < 3: plotted.append(outcome)
            except Exception as exc:
                row["model_status"] = f"failed: {type(exc).__name__}: {exc}"
        rows.append(row)
    results = pd.DataFrame(rows, columns=result_columns)
    if len(results):
        results["interaction_q_value"] = benjamini_hochberg(results.interaction_p_value)
    return results, long, plotted


def persistent_homology_optional(embedding: np.ndarray, config: dict) -> dict:
    settings = config["persistent_homology"]
    if not settings.get("enabled", False):
        return {"status": "disabled"}
    try:
        from ripser import ripser
    except ImportError as exc:
        logging.error("Persistent homology enabled but ripser is unavailable: %s", exc)
        return {"status": "dependency_error", "error": "ripser is required when persistent_homology.enabled is true"}
    diagram = ripser(embedding, maxdim=1)["dgms"][1]
    finite = diagram[np.isfinite(diagram).all(axis=1)] if len(diagram) else diagram
    observed = float(np.sum(finite[:, 1] - finite[:, 0])) if len(finite) else 0.0
    replicates = int(settings["permutation_replicates"]); rng = np.random.default_rng(int(config["random_seed"]))
    null = np.empty(replicates)
    for index in range(replicates):
        permuted = np.column_stack([rng.permutation(embedding[:, j]) for j in range(embedding.shape[1])])
        other = ripser(permuted, maxdim=1)["dgms"][1]
        other = other[np.isfinite(other).all(axis=1)] if len(other) else other
        null[index] = float(np.sum(other[:, 1] - other[:, 0])) if len(other) else 0.0
    return {"status": "completed", "analysis_label": "topological sensitivity analysis",
            "total_h1_persistence": observed,
            "permutation_p_value": float((1 + np.sum(null >= observed)) / (replicates + 1)),
            "diagram": diagram, "null": null}


def _save(fig, path: Path) -> None:
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def make_figures(embedding: np.ndarray, membership: pd.DataFrame, graph: dict,
                 components: list[list[str]], nodes: dict, stability: np.ndarray,
                 crosswalk: pd.DataFrame, ari: float, ari_p: float,
                 longitudinal: pd.DataFrame, models: pd.DataFrame, plotted: list[str],
                 t4: dict, figure_dir: Path) -> None:
    hard = membership.hard_branch.to_numpy(dtype=float); covered = membership.mapper_covered.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].scatter(embedding[~covered, 0], embedding[~covered, 1], c="#BBBBBB", s=20, label="not covered")
    for branch in range(len(components)):
        mask = covered & (hard == branch)
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1], s=24,
                        color=COLORS[branch % len(COLORS)], label=f"branch {branch}")
    axes[0].set(xlabel="PC1", ylabel="PC2", title="A · Baseline patient embedding"); axes[0].legend(frameon=False, fontsize=8)
    branch_of = {node: branch for branch, component in enumerate(components) for node in component}
    centers = {node: embedding[list(members), :2].mean(axis=0) for node, members in nodes.items()}
    for node, linked in graph.get("links", {}).items():
        if node not in centers: continue
        for other in linked:
            if other in centers:
                axes[1].plot(*zip(centers[node], centers[other]), color="#999999", lw=.8, zorder=1)
    for node, center in centers.items():
        color = COLORS[branch_of[node] % len(COLORS)] if node in branch_of else "#BBBBBB"
        axes[1].scatter(*center, s=30 + 7 * len(nodes[node]), color=color, edgecolor="white", zorder=2)
    axes[1].set(xlabel="PC1", ylabel="PC2", title=f"B · Mapper nerve ({len(nodes)} retained nodes)")
    fig.suptitle("Multimodal baseline patient representation and Mapper branches", fontweight="bold")
    _save(fig, figure_dir / "01_graph_embedding_mapper.png")

    fig, ax = plt.subplots(figsize=(6, 4)); x = np.arange(len(stability))
    ax.bar(x, stability, color=[COLORS[i % len(COLORS)] for i in x]); ax.axhline(.8, color="#333", ls="--")
    ax.set(xticks=x, xticklabels=[f"branch {i}" for i in x], ylim=(0, 1), ylabel="Bootstrap stability",
           title="Mapper branch stability (Jaccard matching)"); _save(fig, figure_dir / "02_graph_branch_stability.png")

    pi_cols = [x for x in membership if x.startswith("pi_branch_")]
    order = membership.assign(_max=membership[pi_cols].max(axis=1)).sort_values(["hard_branch", "_max"], ascending=[True, False]).index
    fig, ax = plt.subplots(figsize=(8, max(2.8, .45 * len(pi_cols))))
    if pi_cols: im = ax.imshow(membership.loc[order, pi_cols].to_numpy().T, aspect="auto", cmap="viridis", vmin=0, vmax=1); fig.colorbar(im, ax=ax, label="π_map")
    ax.set(yticks=np.arange(len(pi_cols)), yticklabels=pi_cols, xlabel="Patients (ordered; no identifiers)", title="Soft Mapper membership")
    _save(fig, figure_dir / "03_graph_soft_membership.png")

    fig, ax = plt.subplots(figsize=(7, 4)); pivot = crosswalk.pivot(index="hard_branch", columns="baseline_pop", values="pct_within_branch").fillna(0) if len(crosswalk) else pd.DataFrame()
    bottom = np.zeros(len(pivot))
    for index, pop in enumerate(POPS):
        values = pivot[pop].to_numpy() if pop in pivot else np.zeros(len(pivot)); ax.bar(pivot.index.astype(str), values, bottom=bottom, label=pop, color=COLORS[index]); bottom += values
    ax.set(xlabel="Mapper branch", ylabel="Proportion within branch", ylim=(0, 1), title=f"Branches vs baseline Pop · ARI={ari:.3f}, permutation p={ari_p:.4g}"); ax.legend(frameon=False)
    _save(fig, figure_dir / "04_graph_branches_vs_pop.png")

    fig, axes = plt.subplots(1, max(1, len(plotted)), figsize=(4.7 * max(1, len(plotted)), 4), squeeze=False)
    if not plotted: axes[0, 0].text(.5, .5, "No longitudinal outcome was estimable", ha="center", va="center"); axes[0, 0].set_axis_off()
    for ax, outcome in zip(axes[0], plotted):
        data = longitudinal[["hard_branch", "time", outcome]].copy(); data[outcome] = pd.to_numeric(data[outcome], errors="coerce"); data = data.dropna()
        data["time_bin"] = data.time.round(1)
        for branch, group in data.groupby("hard_branch"):
            summary = group.groupby("time_bin")[outcome].agg(["mean", "sem"]); ax.plot(summary.index, summary["mean"], "-o", ms=3, color=COLORS[int(branch) % len(COLORS)], label=f"branch {int(branch)}"); ax.fill_between(summary.index.to_numpy(float), (summary["mean"]-summary["sem"].fillna(0)).to_numpy(float), (summary["mean"]+summary["sem"].fillna(0)).to_numpy(float), alpha=.15, color=COLORS[int(branch) % len(COLORS)])
        row = models.loc[models.outcome.eq(outcome)].iloc[0]; ax.set(xlabel="Years since clinical baseline", ylabel=outcome, title=f"{outcome}\nbranch×time p={row.interaction_p_value:.3g}, q={row.interaction_q_value:.3g}"); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Longitudinal outcomes after baseline branch assignment", fontweight="bold"); _save(fig, figure_dir / "05_graph_branch_trajectories.png")

    if t4.get("status") == "completed":
        diagram, null = t4["diagram"], t4["null"]; finite = diagram[np.isfinite(diagram).all(axis=1)] if len(diagram) else diagram
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4));
        if len(finite): axes[0].scatter(finite[:, 0], finite[:, 1], color=COLORS[0]); limit=max(float(finite.max()), 1e-6)
        else: limit=1.0
        axes[0].plot([0, limit], [0, limit], "--", color="#999"); axes[0].set(xlabel="Birth", ylabel="Death", title="H1 persistence diagram")
        axes[1].hist(null, bins=30, color="#BBBBBB"); axes[1].axvline(t4["total_h1_persistence"], color=COLORS[1], lw=2); axes[1].set(xlabel="Total H1 persistence under column permutation", ylabel="Frequency", title=f"Observed={t4['total_h1_persistence']:.3g}; p={t4['permutation_p_value']:.3g}")
        fig.suptitle("Topological sensitivity analysis", fontweight="bold"); _save(fig, figure_dir / "06_graph_persistent_homology.png")


def _configure_logging(path: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(path, mode="w"), logging.StreamHandler()])


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dirs = create_study_dirs("graph/01_run_topological_phenotyping")
    _configure_logging(dirs["logs"] / "01_run_topological_phenotyping.log")
    logging.info("Integrated input: %s", args.integrated)
    master, profile, contract = load_real_inputs(args.integrated, args.lab_profile)
    baseline = select_baseline(master)
    n_master = master.patient_id.nunique(); n_baseline = baseline.patient_id.nunique()
    logging.info("Master episodes=%d patients=%d baseline patients=%d excluded without baseline=%d", len(master), n_master, n_baseline, n_master-n_baseline)
    manifest, numeric_values, lab_names = build_feature_manifest(
        baseline, profile, float(config["feature_selection"]["minimum_baseline_coverage"]))
    manifest_path = dirs["tables"] / "01_graph_feature_manifest.csv"; manifest.to_csv(manifest_path, index=False)
    logging.info("Candidate features=%d included features=%d", len(manifest), len(numeric_values))
    # Keep raw missingness for bootstrap preprocessing; impute only representation copies.
    raw_matrix = pd.DataFrame(numeric_values)
    matrix, n_imputed, pct_imputed = build_baseline_matrix(numeric_values)
    logging.info("Imputed values=%d (%.2f%%)", n_imputed, 100*pct_imputed)
    embedding, pca, _ = fit_patient_embedding(matrix, config)
    cumulative = float(pca.explained_variance_ratio_.sum())
    logging.info("PCA retained=%d cumulative variance=%.4f", embedding.shape[1], cumulative)
    mapper_cfg = config["mapper"]
    eps = eps_kdist(embedding, int(mapper_cfg["eps_k_neighbors"]), float(mapper_cfg["eps_percentile"]))
    graph = run_mapper(embedding[:, :2], embedding, config, eps)
    counts, components, nodes, pruning = branches_from_nerve(
        graph, len(baseline), int(mapper_cfg["minimum_node_support"]),
        int(mapper_cfg["minimum_branch_patients"]))
    if not components:
        logging.warning("Mapper produced no analytic branch after configured pruning")
    probabilities, covered, hard = soft_membership(counts, float(mapper_cfg["soft_membership_tau"]))
    baseline_episode = baseline["clinical_episode_id"]
    patient_embedding = pd.DataFrame({"patient_id": baseline.patient_id,
                                      "baseline_clinical_episode_id": baseline_episode,
                                      "baseline_pop": baseline.pop_status})
    for index in range(embedding.shape[1]): patient_embedding[f"PC{index+1}"] = embedding[:, index]
    patient_embedding["mapper_covered"] = covered; patient_embedding["hard_branch"] = pd.array(hard, dtype="Int64")
    for branch in range(probabilities.shape[1]): patient_embedding[f"pi_branch_{branch}"] = probabilities[:, branch]
    membership_columns = ["patient_id", "baseline_clinical_episode_id", "baseline_pop", "mapper_covered", "hard_branch"]
    membership = patient_embedding[membership_columns].copy()
    membership["max_membership"] = np.nanmax(probabilities, axis=1) if probabilities.shape[1] else np.nan
    for branch in range(probabilities.shape[1]): membership[f"pi_branch_{branch}"] = probabilities[:, branch]
    reference_sets = [set(np.flatnonzero(counts[:, branch] > 0)) for branch in range(counts.shape[1])]
    stability = bootstrap_stability(raw_matrix, reference_sets, config)
    stable = [int(i) for i, value in enumerate(stability) if value >= float(config["bootstrap"]["stable_branch_threshold"])]
    valid = covered & baseline.pop_status.isin(POPS).to_numpy()
    ari, ari_p = ari_permutation(hard[valid].astype(int), baseline.loc[valid, "pop_status"].to_numpy(), int(config["permutation"]["ari_replicates"]), int(config["random_seed"]))
    crosswalk = branch_pop_crosswalk(membership)
    administrative = administrative_sensitivity(baseline, membership)
    included = manifest.loc[manifest.included_in_embedding, "feature"].tolist()
    characterization = branch_characterization(baseline, membership, included, lab_names)
    models, longitudinal, plotted = fit_longitudinal_models(master, baseline, membership)
    t4 = persistent_homology_optional(embedding, config)
    logging.info("Mapper eps=%.4g nodes=%d branches=%d coverage=%.3f sizes=%s", eps, len(nodes), len(components), covered.mean(), [int(np.sum(covered & (hard == x))) for x in range(len(components))])
    logging.info("Bootstrap stability=%s; ARI vs Pop=%.4g p=%.4g", stability.tolist(), ari, ari_p)
    logging.info("Longitudinal statuses=%s; T4=%s", models.model_status.tolist() if len(models) else [], t4.get("status"))
    patient_embedding.to_csv(dirs["tables"] / "01_graph_patient_embedding.csv", index=False)
    membership.to_csv(dirs["tables"] / "01_graph_branch_membership.csv", index=False)
    crosswalk.to_csv(dirs["tables"] / "01_graph_branch_pop_crosswalk.csv", index=False)
    characterization.to_csv(dirs["tables"] / "01_graph_branch_characterization.csv", index=False)
    models.to_csv(dirs["tables"] / "01_graph_longitudinal_models.csv", index=False)
    summary = {"input": str(args.integrated), "contract_validation": contract,
               "n_patients_master": int(n_master), "n_baseline_patients": int(n_baseline),
               "n_patients_excluded_no_baseline": int(n_master-n_baseline),
               "n_embedding_features": len(included), "n_imputed_values": n_imputed,
               "pct_imputed_values": pct_imputed, "pca_retained_components": embedding.shape[1],
               "pca_cumulative_variance": cumulative, "mapper_parameters": mapper_cfg,
               "dbscan_eps": eps, "number_mapper_nodes": len(nodes),
               "number_retained_branches": len(components), "branch_sizes": {str(x): int(np.sum(covered & (hard == x))) for x in range(len(components))},
               "mapper_coverage": float(covered.mean()), "mapper_pruning": pruning,
               "branch_bootstrap_stability": {str(x): float(v) for x, v in enumerate(stability)},
               "stable_branches": stable, "ari_vs_baseline_pop": ari,
               "ari_permutation_p": ari_p, "administrative_sensitivity": administrative,
               "longitudinal_outcomes_modeled": models.loc[models.model_status.eq("estimated"), "outcome"].tolist() if len(models) else [],
               "persistent_homology": {key: value for key, value in t4.items() if key not in {"diagram", "null"}}}
    write_json(dirs["tables"] / "01_graph_results_summary.json", summary)
    if not args.dry_run:
        make_figures(embedding, membership, graph, components, nodes, stability,
                     crosswalk, ari, ari_p, longitudinal, models, plotted, t4, dirs["figures"])
    names = ["01_graph_feature_manifest.csv", "01_graph_patient_embedding.csv",
             "01_graph_branch_membership.csv", "01_graph_branch_pop_crosswalk.csv",
             "01_graph_branch_characterization.csv", "01_graph_longitudinal_models.csv",
             "01_graph_results_summary.json", "01_graph_embedding_mapper.png",
             "02_graph_branch_stability.png", "03_graph_soft_membership.png",
             "04_graph_branches_vs_pop.png", "05_graph_branch_trajectories.png"]
    if config["persistent_homology"].get("enabled"): names.append("06_graph_persistent_homology.png   [if available]")
    print("\nORDER TO REVIEW GRAPH OUTPUTS")
    for index, name in enumerate(names, 1): print(f"{index}. {name}")
    print(f"\nBaseline patients: {n_baseline}\nEmbedding features: {len(included)}\nPCA dimensions: {embedding.shape[1]}\nMapper coverage: {covered.mean():.1%}\nMapper branches: {len(components)}\nStable branches: {stable}\nARI vs Pop: {ari:.4g}\nLongitudinal outcomes modeled: {summary['longitudinal_outcomes_modeled']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    parser.add_argument("--lab-profile", type=Path, default=common.STUDIES_TABLES_DIR / "pharma" / "00_profile_labs" / "00_pharma_lab_profile.csv")
    parser.add_argument("--config", type=Path, default=folder / "config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Generate tables/JSON but omit figures")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

"""Inspectable QC for the time-blind visit Mapper; no clinical interpretation."""
from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, skew, spearmanr
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler

KEYS = ["patient_id", "clinical_episode_id"]


def study_paths(common):
    """Resolve defaults without creating directories (especially in dry runs)."""
    return {k: getattr(common, f"STUDIES_{v}_DIR") / "longitudinal_graph"
            for k, v in {"analytic": "ANALYTIC", "tables": "TABLES", "figures": "FIGURES",
                         "qc": "QC", "logs": "LOGS"}.items()}


def validate_keys(frame, label):
    if frame[KEYS].isna().any().any() or frame[KEYS].duplicated().any():
        raise ValueError(f"{label}: missing or duplicate visit keys")


def align_source(keys, source, columns):
    validate_keys(keys, "visit state")
    validate_keys(source, "original source")
    out = keys.merge(source[KEYS + columns], on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not out._merge.eq("both").all():
        raise ValueError("Original source does not contain every state visit")
    return out.drop(columns="_merge")


def run_manifest(root, config, inputs, seed, scenario, sha256_file):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    versions = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "networkx", "matplotlib", "pyarrow"):
        versions[package] = importlib.metadata.version(package)
    import platform
    return {"git_commit": commit, "config_sha256": sha256_file(config),
            "git_dirty": bool(subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip()),
            "code_sha256": {p.name:sha256_file(p) for p in (Path(root)/"src/studies/longitudinal_graph").glob("*.py")},
            "input_sha256": {str(p): sha256_file(p) for p in inputs if p.exists()},
            "script_sha256": sha256_file(Path(root) / "src/studies/longitudinal_graph/02_run_longitudinal_mapper.py"),
            "seed": seed, "date_utc": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario, "python": platform.python_version(), "dependencies": versions,
            "checks": {}, "real_data_status": "not_run_missing_data"}


def weighted_quantile(values, weights, q):
    idx = np.argsort(values, kind="stable")
    x, w = np.asarray(values)[idx], np.asarray(weights)[idx]
    # Combine tied values before interpolation, so repeated copies cannot move quantiles.
    unique, first = np.unique(x, return_index=True)
    x, w = unique, np.add.reduceat(w, first)
    positions = (np.cumsum(w) - .5 * w) / w.sum()
    return float(np.interp(q, positions, x))


def weighted_embedding(matrix, config, family_weights, patient_ids):
    """Equal total mass per patient in robust scaling and PCA, with visit projection."""
    x = np.asarray(matrix, dtype=float)
    counts = pd.Series(patient_ids).value_counts()
    w = np.array([1. / counts[p] for p in patient_ids]); w /= w.sum()
    med = np.array([weighted_quantile(x[:, j], w, .5) for j in range(x.shape[1])])
    iqr = np.array([weighted_quantile(x[:, j], w, .75) - weighted_quantile(x[:, j], w, .25)
                    for j in range(x.shape[1])])
    scale = np.where(iqr > np.finfo(float).eps, iqr, 1.)
    scaled = (x - med) / scale
    if family_weights is not None:
        scaled *= np.asarray(family_weights)
    center = np.average(scaled, axis=0, weights=w)
    _, singular, vectors = np.linalg.svd((scaled - center) * np.sqrt(w[:, None]), full_matrices=False)
    total = np.square(singular).sum()
    if total <= 0: raise ValueError("No weighted PCA variance")
    ratio = np.square(singular) / total
    cap = config["pca"]["max_components"]
    limit = len(ratio) if cap is None else min(int(cap), len(ratio))
    if limit < 2: raise ValueError("PCA requires at least two visits and two features")
    hits = np.flatnonzero(np.cumsum(ratio[:limit]) >= config["pca"]["variance_target"])
    dims = max(2, int(hits[0] + 1) if len(hits) else limit)
    components = vectors[:dims].copy()
    # Fix sign deterministically, independently of sampling order.
    for component in components:
        if component[np.argmax(np.abs(component))] < 0: component *= -1
    pca = SimpleNamespace(components_=components, explained_variance_ratio_=ratio[:dims], n_components_=dims,
                          center_=center, median_=med, scale_=scale)
    return (scaled - center) @ components.T, pca


def audited_mapper(lens, space, config, eps, eps_kdist, patients=None):
    m = config["mapper"]; cubes = int(m["n_cubes"]); overlap = float(m["perc_overlap"])
    if cubes < 1 or not 0 <= overlap < 1: raise ValueError("Invalid Mapper cover")
    rule = m.get("cover_quantiles"); axes = []
    for dim in range(2):
        lo, hi = np.quantile(lens[:, dim], rule) if rule else (lens[:, dim].min(), lens[:, dim].max())
        if hi == lo:
            axes.append([(float(lo), float(hi))] * cubes)
            continue
        width = (hi - lo) / (cubes - (cubes - 1) * overlap); step = width * (1 - overlap)
        axis = [(lo + i * step, lo + i * step + width) for i in range(cubes)]
        axis[-1] = (axis[-1][0], hi)  # Include the exact maximum despite roundoff.
        axes.append(axis)
    nodes = {}; qc = []; in_cover = np.zeros(len(lens), dtype=int); any_cluster = np.zeros(len(lens), dtype=bool)
    for ix, xr in enumerate(axes[0]):
        for iy, yr in enumerate(axes[1]):
            mask = (lens[:, 0] >= xr[0]) & (lens[:, 0] <= xr[1]) & (lens[:, 1] >= yr[0]) & (lens[:, 1] <= yr[1])
            idx = np.flatnonzero(mask); in_cover[idx] += 1
            local_eps = eps; fallback = ""; labels = np.full(len(idx), -1)
            if m.get("eps_mode", "global") == "local":
                if len(idx) >= max(2, int(m["eps_k_neighbors"])):
                    local_eps = eps_kdist(space[idx], m["eps_k_neighbors"], m["eps_percentile"])
                else: fallback = "global_eps_small_preimage"
            if len(idx):
                labels = DBSCAN(eps=local_eps, min_samples=int(m["min_samples"])).fit_predict(space[idx])
            clusters = sorted(set(labels) - {-1})
            for label in clusters:
                members = set(idx[labels == label]); nodes[f"cube{ix}_{iy}_cluster{label}"] = members
                any_cluster[list(members)] = True
            qc.append({"cube_x": ix, "cube_y": iy, "x_low": xr[0], "x_high": xr[1], "y_low": yr[0], "y_high": yr[1],
                       "n_points": len(idx), "n_patients": len(set(np.asarray(patients)[idx])) if patients is not None else None,
                       "eps": local_eps, "eps_mode": m.get("eps_mode", "global"), "fallback": fallback,
                       "min_samples": m["min_samples"], "n_noise": int((labels == -1).sum()), "n_clusters": len(clusters),
                       "pct_input_visits": 100 * len(idx) / len(lens), "boundary_rule": "inclusive_[low,high]"})
    links = {n: [] for n in nodes}; names = sorted(nodes)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if nodes[left] & nodes[right]: links[left].append(right); links[right].append(left)
    noise = pd.DataFrame({"n_cover_cells": in_cover, "outside_cover": in_cover == 0,
                          "noise_only": (in_cover > 0) & ~any_cluster, "in_any_cluster": any_cluster})
    return {"nodes": nodes, "links": links}, pd.DataFrame(qc), noise


def topology_identity(scenario):
    raw = nx.Graph(); raw.add_nodes_from(scenario["graph"]["nodes"])
    for node, links in scenario["graph"]["links"].items():
        raw.add_edges_from((node, other) for other in links)
    components = sorted(nx.connected_components(raw), key=lambda x: (-len(x), min(x)))
    ids = {node: i for i, component in enumerate(components) for node in component}
    scenario["node_table"]["connected_component_id"] = scenario["node_table"].node_id.map(ids)
    scenario["node_table"]["community_id"] = scenario["node_table"].node_id.map(scenario["node_to_macrostate"])
    scenario["node_table"]["branch_id"] = pd.NA
    scenario["node_table"]["branch_status"] = "not_defined_requires_researcher_definition"
    scenario["summary"]["n_connected_components"] = len(components)
    scenario["summary"]["graph_degree_distribution"] = {str(d): sum(value==d for _,value in raw.degree()) for d in set(dict(raw.degree()).values())}
    scenario["summary"]["n_articulation_nodes"] = len(list(nx.articulation_points(raw)))
    scenario["summary"]["n_supported_nerve_components"] = nx.number_connected_components(scenario["nerve"]) if len(scenario["nerve"]) else 0
    scenario["summary"]["branch_status"] = "not_defined_requires_researcher_definition"
    scenario["raw_nerve"] = raw
    return scenario


def representation_qc(state, scenario, family, raw=None):
    features = scenario["features"]; name = scenario["summary"]["scenario"]
    loading_rows = []; family_rows = []; lens_rows = []; scaling_rows = []
    for i, component in enumerate(scenario["pca"].components_[:10]):
        denominator = float(np.square(component).sum()); ranks = np.argsort(-np.abs(component), kind="stable")
        for rank, j in enumerate(ranks):
            loading_rows.append({"scenario": name, "component": f"PC{i+1}", "feature": features[j], "family": family(features[j]),
                                 "loading": component[j], "squared_loading": component[j] ** 2,
                                 "squared_loading_fraction": component[j] ** 2 / denominator,
                                 "absolute_loading_rank": rank + 1, "top20": rank < 20,
                                 "denominator": "sum_squared_loadings_within_component"})
        for fam in sorted(set(map(family, features))):
            fraction = sum(component[j] ** 2 for j, f in enumerate(features) if family(f) == fam) / denominator
            family_rows.append({"scenario": name, "component": f"PC{i+1}", "family": fam,
                                "squared_loading_fraction": fraction, "denominator": "sum_squared_loadings_within_component"})
    for i in range(2):
        x = scenario["embedding"][:, i]; q = np.quantile(x, [0, .01, .25, .5, .75, .99, 1]); iqr = q[4] - q[2]
        lens_rows.append({"scenario": name, "component": f"PC{i+1}", "min": q[0], "p01": q[1], "q1": q[2],
                          "median": q[3], "q3": q[4], "p99": q[5], "max": q[6], "iqr": iqr,
                          "mad": np.median(abs(x-q[3])), "skewness": float(skew(x)) if np.std(x) else 0.,
                          "n_extreme_3iqr": int(((x < q[2]-3*iqr) | (x > q[4]+3*iqr)).sum())})
    # Use the scenario's actual scaling, including its patient-weighted alternative.
    if scenario["summary"].get("weight_rule") == "equal_patient_mass":
        scaled = (state[features].to_numpy() - scenario["pca"].median_) / scenario["pca"].scale_
    else: scaled = RobustScaler().fit_transform(state[features])
    counts = pd.Series(list(map(family, features))).value_counts()
    for j, feature in enumerate(features):
        # Linear quantiles subtract neighboring values; NumPy cannot subtract bools.
        # Work on a numeric copy of finite observations, never fill the raw QC source.
        original = (pd.to_numeric(raw[feature], errors="coerce").astype("float64")
                    if raw is not None and feature in raw else pd.Series(dtype=float))
        original = original.loc[np.isfinite(original)]
        x = scaled[:, j]; siqr = np.quantile(x, .75) - np.quantile(x, .25)
        oi = original.quantile(.75) - original.quantile(.25) if len(original) else None
        balance = 1/math.sqrt(counts[family(feature)]) if scenario["summary"]["family_balance"] else 1.
        scaling_rows.append({"scenario": name, "feature": feature, "original_iqr": oi, "scaled_iqr": siqr,
                             "max_abs": abs(x).max(), "p99_abs": np.quantile(abs(x), .99),
                             "max_over_iqr": abs(x).max()/siqr if siqr else None, "zero_scaled_iqr": siqr == 0,
                             "postbalance_iqr": siqr*balance, "postbalance_max_abs": abs(x).max()*balance,
                             "original_status": "completed" if len(original) else "not_applicable"})
    n = state.groupby("patient_id").size(); equal = scenario["summary"].get("weight_rule") == "equal_patient_mass"
    patients = pd.DataFrame({"patient_id": n.index, "n_visits_per_patient": n.values,
                             "effective_pca_weight": 1/len(n) if equal else n.values/len(state), "scenario": name})
    return {"pca_loading_dominance": pd.DataFrame(loading_rows), "pca_family_contributions": pd.DataFrame(family_rows),
            "lens_distribution_qc": pd.DataFrame(lens_rows), "scaling_feature_diagnostics": pd.DataFrame(scaling_rows),
            "patient_visit_weight_qc": patients}


def coverage_detail(membership, scenario):
    out = membership.copy(); graph = scenario["graph"]; mapping = scenario["node_to_macrostate"]
    out["unsupported_node_only"] = [bool(ns) and all(n not in mapping for n in ns)
                                    for ns in ([n for n, m in graph["nodes"].items() if i in m] for i in range(len(out)))]
    out["unsupported_macrostate_only"] = out.mapper_graph_covered & ~out.mapper_covered & ~out.unsupported_node_only
    out["mixed_supported_and_unsupported"] = [out.iloc[i].mapper_covered and any(n not in mapping for n, m in graph["nodes"].items() if i in m)
                                              for i in range(len(out))]
    out["mixed_supported_and_unsupported"] |= out.mapper_covered & out.in_unsupported_macrostate
    out["hard_assigned"] = out.hard_macrostate.notna(); out["membership_tie"] = out.macrostate_membership_tie
    out["graph_covered_without_supported_hard_assignment"] = out.mapper_graph_covered & ~out.hard_assigned
    out["uncovered"] = ~out.mapper_graph_covered
    out["coverage_category"] = np.select([out.hard_assigned, out.membership_tie, out.unsupported_node_only,
                                         out.unsupported_macrostate_only],
                                        ["hard_assigned", "membership_tie", "unsupported_node_only", "unsupported_macrostate_only"],
                                        default="uncovered")
    # Patient categories are exclusive, using strongest available representation.
    priority = {"hard_assigned": 0, "membership_tie": 1, "unsupported_macrostate_only": 2,
                "unsupported_node_only": 3, "uncovered": 4}
    categories = out.groupby("patient_id").coverage_category.agg(lambda x: min(x, key=priority.get))
    out["patient_coverage_category"] = out.patient_id.map(categories)
    return out


def coverage_by_stratum(detail, metadata):
    validate_keys(metadata, "metadata")
    out = detail.merge(metadata.drop(columns=[c for c in metadata if c in detail and c not in KEYS]), on=KEYS, validate="one_to_one")
    if len(out) != len(detail): raise ValueError("Metadata does not match state visits")
    out["n_visits_per_patient"] = out.patient_id.map(out.groupby("patient_id").size())
    if "clinical_anchor_date" in out:
        out["calendar_year"] = pd.to_datetime(out.clinical_anchor_date, errors="coerce").dt.year
    variables = [c for c in ("n_visits_per_patient", "is_clinical_baseline", "calendar_year", "protocol", "demo__protocol",
                             "visit_type", "spine__interval_name", "era", "state_coverage_fraction") if c in out]
    rows = []
    for variable in [None] + variables:
        groups = [("all", out)] if variable is None else out.groupby(variable, dropna=False)
        for value, group in groups:
            for category in ("hard_assigned", "membership_tie", "unsupported_node_only", "unsupported_macrostate_only", "uncovered"):
                mask = group.coverage_category.eq(category)
                rows.append({"stratum": variable or "all", "value": str(value), "coverage_category": category,
                             "n_visits": int(mask.sum()), "pct_visits": 100*mask.mean(), "denominator_visits": len(group),
                             "n_patients_any": group.loc[mask, "patient_id"].nunique(),
                             "n_patients_exclusive": group.loc[group.patient_coverage_category.eq(category), "patient_id"].nunique(),
                             "denominator_patients": group.patient_id.nunique()})
    # Macrostate memberships overlap; these rows use their own explicit denominator.
    for macro,group in out.loc[out.hard_macrostate.notna()].groupby("hard_macrostate"):
        rows.append({"stratum":"hard_macrostate","value":str(macro),"coverage_category":"hard_assigned",
                     "n_visits":len(group),"pct_visits":100*len(group)/len(out),"denominator_visits":len(out),
                     "n_patients_any":group.patient_id.nunique(),"n_patients_exclusive":None,
                     "denominator_patients":out.patient_id.nunique()})
    return pd.DataFrame(rows)


def interval_endpoint_qc(intervals, membership):
    validate_keys(membership, "membership")
    out = intervals.copy()
    for side in ("from", "to"):
        frame = membership[KEYS + ["mapper_graph_covered", "mapper_covered", "hard_macrostate"]].rename(
            columns={"clinical_episode_id": f"{side}_clinical_episode_id", "mapper_graph_covered": f"{side}_graph_covered",
                     "mapper_covered": f"{side}_supported_covered", "hard_macrostate": f"{side}_hard_macrostate"})
        out = out.merge(frame, on=["patient_id", f"{side}_clinical_episode_id"], how="left", validate="many_to_one")
    for kind in ("graph", "supported"):
        out[f"both_endpoints_{kind}_covered"] = out[f"from_{kind}_covered"].eq(True) & out[f"to_{kind}_covered"].eq(True)
    out["both_endpoints_hard_assigned"] = out.from_hard_macrostate.notna() & out.to_hard_macrostate.notna()
    if len(out) != len(intervals): raise AssertionError("Endpoint join changed interval count")
    return out


def raw_redundancy_audit(audit, raw):
    rows = []
    for row in audit.to_dict("records"):
        a, b = row["feature_a"], row["feature_b"]
        pair = raw[[a, b]].dropna() if a in raw and b in raw else pd.DataFrame()
        rho = pair.corr(method="spearman").iloc[0, 1] if len(pair) > 1 else None
        rows.append({**row, "original_pair_n_observed": len(pair), "original_pair_spearman": rho,
                     "original_status": "completed" if len(pair) > 1 else "not_tested_missing_original_pairs",
                     "pruning_basis": "historical_imputed_state; unchanged_primary"})
    return pd.DataFrame(rows, columns=list(audit.columns) + ["original_pair_n_observed", "original_pair_spearman", "original_status", "pruning_basis"])


def observability_audit(raw, metadata, features):
    present=[f for f in features if f in raw]
    columns=["stratum","value","feature","n_visits","n_observed","n_missing_unknown","n_patients_observed","pct_observed","status"]
    if not present:return pd.DataFrame([{"status":"not_tested_missing_original_observations"}],columns=columns)
    joined=align_source(raw,metadata,[c for c in metadata if c not in raw])
    if "clinical_anchor_date" in joined:
        joined["calendar_year"]=pd.to_datetime(joined.clinical_anchor_date,errors="coerce").dt.year
    rows=[]
    for variable in [None]+[c for c in ("protocol","demo__protocol","visit_type","spine__interval_name","calendar_year","patient_id") if c in joined]:
        groups=[("all",joined)] if variable is None else joined.groupby(variable,dropna=False)
        for value,group in groups:
            for feature in present:
                obs=group[feature].notna()
                rows.append({"stratum":variable or "all","value":str(value),"feature":feature,"n_visits":len(group),
                             "n_observed":int(obs.sum()),"n_missing_unknown":int((~obs).sum()),
                             "n_patients_observed":group.loc[obs,"patient_id"].nunique(),"pct_observed":100*obs.mean(),"status":"completed"})
    return pd.DataFrame(rows,columns=columns)


def pro_pair_audit(raw,state,features):
    pro=[f for f in features if f.startswith("pro__")];rows=[]
    for i,a in enumerate(pro):
        for b in pro[i+1:]:
            pair=raw[[a,b]].dropna() if a in raw and b in raw else pd.DataFrame()
            rows.append({"feature_a":a,"feature_b":b,"imputed_spearman":state[[a,b]].corr(method="spearman").iloc[0,1],
                         "original_spearman":pair.corr(method="spearman").iloc[0,1] if len(pair)>1 else None,
                         "n_original_pairs":len(pair),"status":"completed" if len(pair)>1 else "not_tested_missing_original_pairs",
                         "semantic_decision":"requires_clinical_total_subscale_review; no_automatic_semantic_pruning"})
    return pd.DataFrame(rows,columns=["feature_a","feature_b","imputed_spearman","original_spearman","n_original_pairs","status","semantic_decision"])


def biomarker_audit(source, manifest, features):
    targets = {"SSA/Ro": ("anti_ro", "anti_ssa", "ssa_ro"), "SSB/La": ("anti_la", "anti_ssb", "ssb_la"),
               "ANA": ("__ana", "antinuclear"), "RF": ("__rf", "rheumatoid_factor"), "IgG": ("igg",),
               "C3": ("__c3", "complement_c3"), "C4": ("__c4", "complement_c4"),
               "cryoglobulins": ("cryoglob",), "ESR": ("__esr", "sedimentation"), "hsCRP": ("crp_high_sensitivity",),
               "glandular/oral/ocular": ("gland", "oral", "ocular", "schirmer", "salivary")}
    rows = []; mf = manifest.set_index("feature") if not manifest.empty else pd.DataFrame()
    for biomarker, tokens in targets.items():
        known=list(dict.fromkeys(list(source)+list(features)+(manifest.feature.tolist() if not manifest.empty else [])))
        matches = [c for c in known if any(t in c.lower() for t in tokens)]
        for feature in matches or [None]:
            info = mf.loc[feature].to_dict() if feature in mf.index else {}
            included = bool(info.get("included", False)); used = feature in features
            scope = info.get("temporal_scope", "unknown")
            available=feature in source if feature is not None else False
            rows.append({"biomarker": biomarker, "feature": feature, "source": "integrated_master_or_original_source" if available else "state_or_manifest_only" if feature else "unavailable",
                         "temporal_scope": scope, "n_observed": int(source[feature].notna().sum()) if available else None,
                         "available": available, "eligible_episode_resolved": included and scope == "episode_resolved",
                         "retained": used, "used_in_primary_mapper": used,
                         "decision": "retained" if used else "excluded_with_reason" if feature else "unavailable",
                         "reason": info.get("exclusion_reason") or ("postpruning_exclusion" if included else "scope_or_eligibility_not_verified"),
                         "role": "defining" if used else "held_out_anchor" if scope in {"baseline_static", "patient_static", "cumulative_history"} else "excluded" if feature else "unavailable",
                         "circularity_risk": "used_in_construction" if used else "proxy_dependence_requires_review",
                         "unit_validation": "unknown", "assay_era_validation": "unknown"})
    return pd.DataFrame(rows)


def circularity_audit(manifest, features, source):
    rows = []
    for feature in dict.fromkeys(list(source)+list(features)):
        if feature in KEYS: continue
        used = feature in features
        rows.append({"feature": feature, "used_in_actual_mapper": used, "used_in_primary_mapper": used,
                     "role": "defining" if used else "held_out_candidate",
                     "independence_status": "not_external" if used else "proxy_dependence_requires_review",
                     "reason": "actual_postpruning_feature_list"})
    return pd.DataFrame(rows, columns=["feature", "used_in_actual_mapper", "used_in_primary_mapper", "role", "independence_status", "reason"])


def administrative_audit(membership, metadata, embedding, config):
    """Descriptive visit V plus patient-vector permutations within series lengths."""
    validate_keys(metadata, "metadata")
    joined = membership.merge(metadata.drop(columns=[c for c in metadata if c in membership and c not in KEYS]), on=KEYS, validate="one_to_one")
    if len(joined) != len(membership): raise ValueError("Metadata does not match membership")
    joined["n_visits_per_patient"] = joined.patient_id.map(joined.groupby("patient_id").size())
    if "clinical_anchor_date" in joined:
        joined["calendar_year"] = pd.to_datetime(joined.clinical_anchor_date, errors="coerce").dt.year
    variables = ["protocol", "demo__protocol", "era", "calendar_year", "visit_type", "spine__interval_name", "state_coverage_fraction", "n_visits_per_patient"]
    availability = []; associations = []; strata = []; lens_rows = []
    threshold = config["administrative_qc"]["cramers_v_flag_threshold"]
    permutations = int(config.get("mapper_validation", {}).get("admin_permutations", 199)); rng = np.random.default_rng(config["random_seed"])
    def cramer(x, y):
        tab = pd.crosstab(x, y)
        if min(tab.shape) < 2: return np.nan
        return math.sqrt(chi2_contingency(tab, correction=False)[0] / (tab.to_numpy().sum() * (min(tab.shape)-1)))
    for variable in variables:
        if variable not in joined:
            availability.append({"variable": variable, "status": "not_tested_missing_variable", "n_observed": 0}); continue
        observed = joined.loc[joined[variable].notna() & joined.hard_macrostate.notna()].copy()
        availability.append({"variable": variable, "status": "completed" if len(observed) else "not_tested_no_valid_assignments", "n_observed": len(observed)})
        if not len(observed): continue
        numeric = pd.api.types.is_numeric_dtype(observed[variable])
        category = observed[variable].astype(str)
        if numeric and observed[variable].nunique() > 8:
            category = pd.qcut(observed[variable], 4, duplicates="drop").astype(str)
        observed["_category"] = category
        observed["_label"] = observed.hard_macrostate.astype(str)
        value = cramer(observed._category, observed._label); null = []
        order = observed.sort_values(KEYS).reset_index(drop=True)
        blocks = [g.index.to_numpy() for _, g in order.groupby("patient_id", sort=True)]
        by_length = {}
        for idx in blocks: by_length.setdefault(len(idx), []).append(idx)
        exchangeable = sum(len(x) for x in by_length.values() if len(x) > 1)
        if np.isfinite(value) and exchangeable > 1:
            for _ in range(permutations):
                labels = order._label.to_numpy().copy()
                for group in by_length.values():
                    for target, j in zip(group, rng.permutation(len(group))): labels[target] = order._label.to_numpy()[group[j]]
                null.append(cramer(order._category, labels))
        p = (1 + sum(x >= value for x in null)) / (1 + len(null)) if null else None
        associations.append({"administrative_variable": variable, "cramers_v": value, "n_visits": len(observed),
                             "n_patients": observed.patient_id.nunique(), "administrative_confounding_flag": bool(np.isfinite(value) and value >= threshold),
                             "flag_threshold": threshold, "patient_permutation_p": p, "n_permutations": len(null),
                             "inference_status": "patient_vector_permutation_same_series_length" if null else "not_tested_insufficient_exchangeable_patients",
                             "visit_p_value_used": False})
        for (category, macro), g in observed.groupby(["_category", "hard_macrostate"]):
            strata.append({"variable": variable, "value": category, "macrostate_id": macro, "n_visits": len(g),
                           "n_patients": g.patient_id.nunique(), "pct_within_macrostate": 100*len(g)/observed.hard_macrostate.eq(macro).sum(),
                           "pct_within_category": 100*len(g)/observed._category.eq(category).sum()})
        for pc in range(2):
            full = joined.loc[joined[variable].notna()]
            if numeric:
                rho = spearmanr(full[variable], embedding[full.index, pc]).statistic if full[variable].nunique() > 1 else None
                lens_rows.append({"variable": variable, "component": f"PC{pc+1}", "category": "all", "spearman_descriptive": rho,
                                  "n_visits": len(full), "inference_status": "descriptive_correlated_visits"})
            else:
                for category, g in full.groupby(variable):
                    scores = embedding[g.index, pc]
                    lens_rows.append({"variable": variable, "component": f"PC{pc+1}", "category": category, "n_visits": len(g),
                                      "score_median": np.median(scores), "score_p01": np.quantile(scores,.01), "score_p99": np.quantile(scores,.99),
                                      "inference_status": "descriptive_correlated_visits"})
    return (pd.DataFrame(associations, columns=["administrative_variable", "cramers_v", "n_visits", "n_patients", "administrative_confounding_flag", "flag_threshold", "patient_permutation_p", "n_permutations", "inference_status", "visit_p_value_used"]),
            pd.DataFrame(availability), pd.DataFrame(strata, columns=["variable", "value", "macrostate_id", "n_visits", "n_patients", "pct_within_macrostate", "pct_within_category"]),
            pd.DataFrame(lens_rows, columns=["variable", "component", "category", "n_visits", "spearman_descriptive", "score_median", "score_p01", "score_p99", "inference_status"]))


def archive_and_publish(staged, destination, manifest, write_json):
    """Publish only a completed bundle, with an external archive of every old 02 file."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = destination["tables"].parent / "longitudinal_graph_history" / stamp
    old = [(kind, p) for kind, root in destination.items() for p in root.glob("02_*") if p.is_file()]
    for kind, p in old:
        target = archive / kind / p.name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
    manifest["historical_archive"] = str(archive) if old else None
    write_json(staged["qc"] / "02_run_manifest.json", manifest)
    # Remove only archived files, including stale sensitivities, then install this bundle.
    for _, p in old: p.unlink()
    for kind, root in staged.items():
        destination[kind].mkdir(parents=True, exist_ok=True)
        for p in root.iterdir():
            if p.is_file(): shutil.copy2(p, destination[kind] / p.name)

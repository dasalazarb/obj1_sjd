#!/usr/bin/env python3
"""Statistically characterize Mapper communities from S2 and S3 representations.

This script is an interpretation layer over 02_graph_representation_sensitivity.py.
It does not redefine communities. It consumes S2/S3 Mapper assignments and tests
which baseline variables distinguish each supported community.

Primary principles
------------------
1. S2 and S3 are co-primary representation scenarios.
2. Community IDs are aligned across scenarios by maximum patient-set Jaccard;
   numeric labels are never assumed to correspond.
3. Hard-assignment inference uses Mapper-covered, non-tied patients assigned to
   supported communities.
4. Raw baseline values are used for interpretation; imputed/scaled PCA values are
   not used for clinical summary statistics.
5. Variables used to build the representation are labeled "defining_feature".
   Variables deliberately held out of discovery are labeled
   "external_characterizer".
6. A variable is called statistically characteristic only if it passes FDR and
   a minimum effect-size threshold. Robust signatures must reproduce in both S2
   and S3 with the same direction after community alignment.
7. Soft-membership analyses are descriptive gradients, not independent validation.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
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
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu, spearmanr

import common
from src.studies._shared import create_study_dirs

SCENARIOS = ("S2", "S3")
POPS = ("Pop1", "Pop2", "Pop3")


def _load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANON = _load_module("01_run_topological_phenotyping.py", "graph_topological_phenotyping")
SENS = _load_module("02_graph_representation_sensitivity.py", "graph_representation_sensitivity")


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    return cfg


def bh_qvalues(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    result = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.notna() & np.isfinite(p)
    if not valid.any():
        return result
    vals = p.loc[valid].to_numpy(float)
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    back = np.empty(m)
    back[order] = q
    result.loc[valid] = back
    return result


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not len(x) or not len(y):
        return np.nan
    # N is small enough here; chunk to avoid unnecessary memory expansion.
    total = 0
    for chunk in np.array_split(x, max(1, math.ceil(len(x) / 256))):
        total += np.sign(chunk[:, None] - y[None, :]).sum()
    return float(total / (len(x) * len(y)))


def odds_ratio_2x2(a: int, b: int, c: int, d: int) -> float:
    # Haldane-Anscombe correction only when needed for a finite descriptive OR.
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    return float((a * d) / (b * c))


def direction_from_effect(effect: float, kind: str) -> str:
    if pd.isna(effect):
        return "NA"
    if kind == "continuous":
        return "higher" if effect > 0 else "lower" if effect < 0 else "same"
    return "enriched" if effect > 1 else "depleted" if effect < 1 else "same"


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}. Run 02_graph_representation_sensitivity.py first.")
    return path


def build_baseline_sources(args: argparse.Namespace):
    master, candidates, _ = CANON.load_real_inputs(args.integrated, args.graph_lab_candidates)
    baseline = CANON.select_baseline(master)
    age = CANON.load_age_at_diagnosis(args.baseline_metrics)
    baseline = baseline.merge(age, on="patient_id", how="left", validate="one_to_one",
                              suffixes=("", "_blockA"))
    if "age_dx_blockA" in baseline:
        baseline["age_dx"] = baseline.pop("age_dx_blockA")

    manifest, values, _, _ = CANON.build_feature_manifest(baseline, candidates, 0.0)
    families = {row.feature: SENS.family_name(row.family)
                for row in manifest.loc[manifest.included].itertuples()}
    raw = baseline[["patient_id"]].copy()
    for feature, series in values.items():
        raw[feature] = pd.to_numeric(series, errors="coerce").to_numpy()
    return baseline, manifest, raw, families


def external_variables(baseline: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    out = baseline[["patient_id"]].copy()
    kinds: dict[str, str] = {}
    for name in cfg.get("external_numeric", []):
        if name in baseline:
            out[name] = pd.to_numeric(baseline[name], errors="coerce")
            kinds[name] = "continuous"
    for name in cfg.get("external_categorical", []):
        if name in baseline:
            out[name] = baseline[name].astype("string")
            kinds[name] = "categorical"
    return out, kinds


def scenario_feature_roles(feature_sets: pd.DataFrame) -> dict[tuple[str, str], str]:
    roles = {}
    for row in feature_sets.itertuples(index=False):
        if row.scenario in SCENARIOS:
            roles[(row.scenario, row.feature)] = "defining_feature" if bool(row.included) else "not_in_embedding"
    return roles


def eligible_members(membership: pd.DataFrame, scenario: str) -> pd.DataFrame:
    frame = membership.loc[membership.scenario.eq(scenario)].copy()
    valid = (frame.mapper_covered.fillna(False)
             & ~frame.community_membership_tie.fillna(False)
             & frame.community_supported.fillna(False)
             & frame.hard_community.notna())
    frame = frame.loc[valid].copy()
    frame["hard_community"] = frame["hard_community"].astype(int)
    return frame


def community_sets(membership: pd.DataFrame, scenario: str) -> dict[int, set]:
    frame = eligible_members(membership, scenario)
    return {int(k): set(g.patient_id) for k, g in frame.groupby("hard_community")}


def align_s2_s3(membership: pd.DataFrame) -> pd.DataFrame:
    s2 = community_sets(membership, "S2")
    s3 = community_sets(membership, "S3")
    rows = []
    used_s3: set[int] = set()
    # Greedy maximum-Jaccard matching; deterministic tie-breaks.
    candidates = []
    for c2, p2 in s2.items():
        for c3, p3 in s3.items():
            inter, union = p2 & p3, p2 | p3
            j = len(inter) / len(union) if union else np.nan
            candidates.append((j, len(inter), c2, c3, len(p2), len(p3)))
    matched_s2: set[int] = set()
    for j, inter, c2, c3, n2, n3 in sorted(
            candidates, key=lambda x: (-np.nan_to_num(x[0], nan=-1), -x[1], x[2], x[3])):
        if c2 in matched_s2 or c3 in used_s3:
            continue
        rows.append({"s2_community": c2, "s3_community": c3, "jaccard": j,
                     "n_intersection": inter, "n_s2": n2, "n_s3": n3})
        matched_s2.add(c2); used_s3.add(c3)
    return pd.DataFrame(rows)


def is_binary(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna().unique()
    if len(vals) != 2:
        return False
    return set(np.round(vals.astype(float), 12)).issubset({0.0, 1.0})


def one_vs_rest_numeric(frame: pd.DataFrame, feature: str, community: int) -> dict:
    inside = pd.to_numeric(frame.loc[frame.hard_community.eq(community), feature], errors="coerce").dropna()
    outside = pd.to_numeric(frame.loc[~frame.hard_community.eq(community), feature], errors="coerce").dropna()
    if len(inside) < 3 or len(outside) < 3:
        return {"status": "insufficient_n", "p_value": np.nan}
    if is_binary(frame[feature]):
        a = int((inside == 1).sum()); b = int((inside == 0).sum())
        c = int((outside == 1).sum()); d = int((outside == 0).sum())
        table = np.array([[a, b], [c, d]])
        _, p = fisher_exact(table, alternative="two-sided")
        effect = odds_ratio_2x2(a, b, c, d)
        return {
            "status": "estimated", "test": "Fisher_exact", "variable_type": "binary",
            "n_community": len(inside), "n_rest": len(outside),
            "community_summary": float(inside.mean()), "rest_summary": float(outside.mean()),
            "summary_metric": "proportion_1", "effect_name": "odds_ratio",
            "effect": effect, "direction": direction_from_effect(effect, "binary"),
            "p_value": float(p),
        }
    _, p = mannwhitneyu(inside, outside, alternative="two-sided")
    delta = cliffs_delta(inside.to_numpy(), outside.to_numpy())
    return {
        "status": "estimated", "test": "Mann_Whitney_U", "variable_type": "continuous",
        "n_community": len(inside), "n_rest": len(outside),
        "community_summary": float(inside.median()), "rest_summary": float(outside.median()),
        "summary_metric": "median", "effect_name": "cliffs_delta",
        "effect": delta, "direction": direction_from_effect(delta, "continuous"),
        "p_value": float(p),
    }


def omnibus_numeric(frame: pd.DataFrame, feature: str) -> dict:
    groups = []
    communities = []
    for community, group in frame.groupby("hard_community"):
        values = pd.to_numeric(group[feature], errors="coerce").dropna()
        if len(values):
            groups.append(values.to_numpy())
            communities.append(int(community))
    if len(groups) < 2 or any(len(g) < 3 for g in groups):
        return {"status": "insufficient_n", "p_value": np.nan}
    if is_binary(frame[feature]):
        tmp = frame[["hard_community", feature]].copy()
        tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
        tmp = tmp.dropna()
        tab = pd.crosstab(tmp.hard_community, tmp[feature])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            return {"status": "not_estimable", "p_value": np.nan}
        chi, p, _, _ = chi2_contingency(tab)
        n = tab.to_numpy().sum()
        phi2 = chi / n if n else np.nan
        r, k = tab.shape
        v = math.sqrt(phi2 / max(1, min(k - 1, r - 1))) if np.isfinite(phi2) else np.nan
        return {"status": "estimated", "test": "chi_square", "variable_type": "binary",
                "effect_name": "cramers_v", "effect": v, "p_value": float(p)}
    stat, p = kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eps2 = max(0.0, float((stat - k + 1) / (n - k))) if n > k else np.nan
    return {"status": "estimated", "test": "Kruskal_Wallis", "variable_type": "continuous",
            "effect_name": "epsilon_squared", "effect": eps2, "p_value": float(p)}


def pairwise_numeric_tests(frame: pd.DataFrame, features: list[str],
                           scenario: str, feature_role: str) -> list[dict]:
    """Direct community-vs-community tests; useful when K > 2 and transparent when K = 2."""
    rows: list[dict] = []
    communities = sorted(frame.hard_community.dropna().astype(int).unique())
    for left, right in itertools.combinations(communities, 2):
        for feature in features:
            if feature not in frame:
                continue
            x = pd.to_numeric(
                frame.loc[frame.hard_community.eq(left), feature], errors="coerce"
            ).dropna()
            y = pd.to_numeric(
                frame.loc[frame.hard_community.eq(right), feature], errors="coerce"
            ).dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            if is_binary(frame[feature]):
                a = int((x == 1).sum()); b = int((x == 0).sum())
                c = int((y == 1).sum()); d = int((y == 0).sum())
                _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
                effect = odds_ratio_2x2(a, b, c, d)
                rows.append({
                    "scenario": scenario, "community_a": left, "community_b": right,
                    "feature": feature, "feature_role": feature_role,
                    "variable_type": "binary", "test": "Fisher_exact",
                    "n_a": len(x), "n_b": len(y),
                    "summary_a": float(x.mean()), "summary_b": float(y.mean()),
                    "summary_metric": "proportion_1", "effect_name": "odds_ratio_a_vs_b",
                    "effect": effect, "p_value": float(p),
                })
            else:
                _, p = mannwhitneyu(x, y, alternative="two-sided")
                delta = cliffs_delta(x.to_numpy(), y.to_numpy())
                rows.append({
                    "scenario": scenario, "community_a": left, "community_b": right,
                    "feature": feature, "feature_role": feature_role,
                    "variable_type": "continuous", "test": "Mann_Whitney_U",
                    "n_a": len(x), "n_b": len(y),
                    "summary_a": float(x.median()), "summary_b": float(y.median()),
                    "summary_metric": "median", "effect_name": "cliffs_delta_a_vs_b",
                    "effect": delta, "p_value": float(p),
                })
    return rows


def pairwise_tables(membership: pd.DataFrame, baseline_raw: pd.DataFrame,
                    feature_sets: pd.DataFrame, external: pd.DataFrame,
                    external_kinds: dict[str, str]) -> pd.DataFrame:
    rows: list[dict] = []
    for scenario in SCENARIOS:
        mem = eligible_members(membership, scenario)
        if mem.hard_community.nunique() < 2:
            continue
        defining = feature_sets.loc[
            feature_sets.scenario.eq(scenario) & feature_sets.included, "feature"
        ].tolist()
        data = mem.merge(baseline_raw, on="patient_id", how="left", validate="one_to_one")
        rows.extend(pairwise_numeric_tests(data, defining, scenario, "defining_feature"))

        numeric_external = [f for f, kind in external_kinds.items()
                            if kind == "continuous" and f in external]
        ext = mem.merge(external, on="patient_id", how="left", validate="one_to_one")
        rows.extend(pairwise_numeric_tests(
            ext, numeric_external, scenario, "external_characterizer"
        ))

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = out.groupby(
            ["scenario", "feature_role"], group_keys=False
        )["p_value"].transform(bh_qvalues)
    return out


def categorical_external_tests(frame: pd.DataFrame, feature: str, community: int) -> list[dict]:
    data = frame[["hard_community", feature]].dropna().copy()
    data["in_community"] = data.hard_community.eq(community)
    rows = []
    for level in sorted(data[feature].astype(str).unique()):
        positive = data[feature].astype(str).eq(level)
        a = int((positive & data.in_community).sum())
        b = int((~positive & data.in_community).sum())
        c = int((positive & ~data.in_community).sum())
        d = int((~positive & ~data.in_community).sum())
        if a + b < 3 or c + d < 3:
            continue
        _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        orr = odds_ratio_2x2(a, b, c, d)
        rows.append({
            "feature": feature, "level": level, "variable_type": "categorical_level",
            "test": "Fisher_exact", "n_community": a + b, "n_rest": c + d,
            "community_summary": a / (a + b), "rest_summary": c / (c + d),
            "summary_metric": "proportion_level", "effect_name": "odds_ratio",
            "effect": orr, "direction": direction_from_effect(orr, "binary"),
            "p_value": float(p), "status": "estimated",
        })
    return rows


def add_characteristic_flag(rows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["q_value"] = rows.groupby(["scenario", "feature_role"], group_keys=False)["p_value"].transform(bh_qvalues)
    alpha = float(cfg["statistics"]["fdr_alpha"])
    delta_min = float(cfg["statistics"]["minimum_abs_cliffs_delta"])
    or_min = float(cfg["statistics"]["minimum_odds_ratio"])
    rows["passes_effect_threshold"] = False
    cont = rows.effect_name.eq("cliffs_delta")
    rows.loc[cont, "passes_effect_threshold"] = rows.loc[cont, "effect"].abs().ge(delta_min)
    odds = rows.effect_name.eq("odds_ratio")
    rows.loc[odds, "passes_effect_threshold"] = (
        rows.loc[odds, "effect"].ge(or_min)
        | rows.loc[odds, "effect"].le(1.0 / or_min)
    )
    rows["statistically_characteristic"] = (
        rows.status.eq("estimated") & rows.q_value.lt(alpha) & rows.passes_effect_threshold
    )
    return rows


def hard_signature_tables(membership: pd.DataFrame, baseline_raw: pd.DataFrame,
                          feature_sets: pd.DataFrame, external: pd.DataFrame,
                          external_kinds: dict[str, str], cfg: dict):
    feature_roles = scenario_feature_roles(feature_sets)
    signature_rows, omnibus_rows, missing_rows = [], [], []

    for scenario in SCENARIOS:
        mem = eligible_members(membership, scenario)
        if mem.hard_community.nunique() < 2:
            logging.warning("%s has fewer than two eligible supported communities", scenario)
            continue
        data = mem.merge(baseline_raw, on="patient_id", how="left", validate="one_to_one")
        ext = mem.merge(external, on="patient_id", how="left", validate="one_to_one")

        included_features = feature_sets.loc[
            feature_sets.scenario.eq(scenario) & feature_sets.included, "feature"
        ].tolist()

        for feature in included_features:
            if feature not in data:
                continue
            global_test = omnibus_numeric(data, feature)
            omnibus_rows.append({
                "scenario": scenario, "feature": feature, "feature_role": "defining_feature",
                **global_test
            })
            for community in sorted(data.hard_community.unique()):
                result = one_vs_rest_numeric(data, feature, int(community))
                signature_rows.append({
                    "scenario": scenario, "community": int(community), "feature": feature,
                    "level": "", "feature_role": feature_roles.get((scenario, feature), "defining_feature"),
                    **result
                })
                indicator = data[feature].isna().astype(int)
                miss = data[["hard_community"]].assign(_missing=indicator)
                if miss._missing.nunique() == 2:
                    mres = one_vs_rest_numeric(miss.rename(columns={"_missing": feature}), feature, int(community))
                    missing_rows.append({
                        "scenario": scenario, "community": int(community), "feature": feature,
                        "missingness_rate_community": float(
                            data.loc[data.hard_community.eq(community), feature].isna().mean()),
                        "missingness_rate_rest": float(
                            data.loc[~data.hard_community.eq(community), feature].isna().mean()),
                        "p_value": mres.get("p_value", np.nan),
                    })

        for feature, kind in external_kinds.items():
            if feature not in ext:
                continue
            if kind == "continuous":
                global_test = omnibus_numeric(ext, feature)
                omnibus_rows.append({
                    "scenario": scenario, "feature": feature,
                    "feature_role": "external_characterizer", **global_test
                })
                for community in sorted(ext.hard_community.unique()):
                    result = one_vs_rest_numeric(ext, feature, int(community))
                    signature_rows.append({
                        "scenario": scenario, "community": int(community), "feature": feature,
                        "level": "", "feature_role": "external_characterizer", **result
                    })
            else:
                tab = pd.crosstab(ext.hard_community, ext[feature])
                if tab.shape[0] >= 2 and tab.shape[1] >= 2:
                    chi, p, _, _ = chi2_contingency(tab)
                    n = tab.to_numpy().sum()
                    v = math.sqrt((chi / n) / max(1, min(tab.shape[0]-1, tab.shape[1]-1))) if n else np.nan
                    omnibus_rows.append({
                        "scenario": scenario, "feature": feature,
                        "feature_role": "external_characterizer", "status": "estimated",
                        "test": "chi_square", "variable_type": "categorical",
                        "effect_name": "cramers_v", "effect": v, "p_value": float(p)
                    })
                for community in sorted(ext.hard_community.unique()):
                    for result in categorical_external_tests(ext, feature, int(community)):
                        signature_rows.append({
                            "scenario": scenario, "community": int(community),
                            "feature_role": "external_characterizer", **result
                        })

    signatures = add_characteristic_flag(pd.DataFrame(signature_rows), cfg)
    omnibus = pd.DataFrame(omnibus_rows)
    if not omnibus.empty:
        omnibus["q_value"] = omnibus.groupby(["scenario", "feature_role"], group_keys=False)["p_value"].transform(bh_qvalues)
    missing = pd.DataFrame(missing_rows)
    if not missing.empty:
        missing["q_value"] = missing.groupby("scenario", group_keys=False)["p_value"].transform(bh_qvalues)
        missing["differential_missingness"] = missing.q_value.lt(float(cfg["statistics"]["fdr_alpha"]))
    return signatures, omnibus, missing


def soft_membership_table(membership: pd.DataFrame, baseline_raw: pd.DataFrame,
                          feature_sets: pd.DataFrame, external: pd.DataFrame,
                          cfg: dict) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        mem = membership.loc[membership.scenario.eq(scenario) & membership.mapper_covered.fillna(False)].copy()
        pi_cols = [c for c in mem if c.startswith("pi_community_")]
        included = feature_sets.loc[
            feature_sets.scenario.eq(scenario) & feature_sets.included, "feature"
        ].tolist()
        data = mem.merge(baseline_raw, on="patient_id", how="left", validate="one_to_one")
        data = data.merge(external, on="patient_id", how="left", validate="one_to_one",
                          suffixes=("", "_external"))
        candidate_features = [f for f in included if f in data]
        candidate_features += [f for f in cfg.get("external_numeric", []) if f in data]
        candidate_features = list(dict.fromkeys(candidate_features))
        for pi in pi_cols:
            community = int(pi.rsplit("_", 1)[-1])
            pivec = pd.to_numeric(data[pi], errors="coerce")
            for feature in candidate_features:
                values = pd.to_numeric(data[feature], errors="coerce")
                valid = pivec.notna() & values.notna()
                if valid.sum() < 5 or values.loc[valid].nunique() < 2:
                    continue
                rho, p = spearmanr(pivec.loc[valid], values.loc[valid])
                rows.append({
                    "scenario": scenario, "community": community, "feature": feature,
                    "feature_role": "external_characterizer" if feature in cfg.get("external_numeric", [])
                    else "defining_feature",
                    "n": int(valid.sum()), "spearman_rho": float(rho), "p_value": float(p),
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value"] = result.groupby("scenario", group_keys=False)["p_value"].transform(bh_qvalues)
    return result


def robust_s2_s3_signatures(signatures: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    if signatures.empty or alignment.empty:
        return pd.DataFrame()
    rows = []
    keycols = ["feature", "level", "feature_role"]
    for match in alignment.itertuples(index=False):
        left = signatures.loc[
            signatures.scenario.eq("S2") & signatures.community.eq(match.s2_community)
        ].copy()
        right = signatures.loc[
            signatures.scenario.eq("S3") & signatures.community.eq(match.s3_community)
        ].copy()
        merged = left.merge(right, on=keycols, suffixes=("_s2", "_s3"))
        for row in merged.itertuples(index=False):
            robust = (
                bool(row.statistically_characteristic_s2)
                and bool(row.statistically_characteristic_s3)
                and row.direction_s2 == row.direction_s3
            )
            rows.append({
                "s2_community": int(match.s2_community),
                "s3_community": int(match.s3_community),
                "community_alignment_jaccard": float(match.jaccard),
                "feature": row.feature, "level": row.level,
                "feature_role": row.feature_role,
                "direction_s2": row.direction_s2, "direction_s3": row.direction_s3,
                "effect_s2": row.effect_s2, "effect_s3": row.effect_s3,
                "q_value_s2": row.q_value_s2, "q_value_s3": row.q_value_s3,
                "robust_signature_s2_s3": robust,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["robust_signature_s2_s3", "feature_role", "q_value_s3", "q_value_s2"],
            ascending=[False, True, True, True]
        )
    return out


def make_figures(signatures: pd.DataFrame, robust: pd.DataFrame, figure_dir: Path) -> None:
    robust_yes = robust.loc[robust.robust_signature_s2_s3].copy() if not robust.empty else robust
    if robust_yes is not None and not robust_yes.empty:
        labels = robust_yes["feature"].astype(str)
        levels = robust_yes["level"].fillna("").astype(str)
        labels = labels.where(levels.eq(""), labels + "=" + levels)
        y = np.arange(len(robust_yes))
        fig, ax = plt.subplots(figsize=(9, max(4, .38 * len(robust_yes) + 1)))
        ax.scatter(robust_yes.effect_s2, y, label="S2", marker="o")
        ax.scatter(robust_yes.effect_s3, y, label="S3", marker="x")
        ax.axvline(0, color="#777777", lw=1)
        ax.set(yticks=y, yticklabels=labels, xlabel="Effect size (Cliff's delta; ORs shown on native scale)",
               title="Robust community differentiators reproduced in S2 and S3")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / "03_graph_robust_community_signatures.png", bbox_inches="tight")
        plt.close(fig)

    if signatures.empty:
        return
    sig = signatures.loc[signatures.statistically_characteristic].copy()
    if sig.empty:
        return
    sig["label"] = sig.feature.astype(str)
    sig.loc[sig.level.fillna("").astype(str).ne(""), "label"] += "=" + sig.level.fillna("").astype(str)
    top = sig.sort_values(["q_value", "effect"], ascending=[True, False]).head(30)
    pivot = top.pivot_table(index="label", columns=["scenario", "community"], values="effect", aggfunc="first")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, .35 * len(pivot) + 1)))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set(yticks=np.arange(len(pivot.index)), yticklabels=pivot.index,
           xticks=np.arange(len(pivot.columns)),
           xticklabels=[f"{a}-C{b}" for a, b in pivot.columns], title="Community signature effect sizes")
    ax.tick_params(axis="x", rotation=45)
    fig.colorbar(image, ax=ax, label="Effect size")
    fig.tight_layout()
    fig.savefig(figure_dir / "03_graph_community_signature_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    dirs = create_study_dirs("graph/03_graph_community_characterization")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    membership = pd.read_csv(require_file(args.membership, "02 membership"))
    feature_sets = pd.read_csv(require_file(args.feature_sets, "02 feature sets"))
    stability = pd.read_csv(require_file(args.stability, "02 community stability"))

    baseline, manifest, raw, families = build_baseline_sources(args)
    external, external_kinds = external_variables(baseline, cfg)

    alignment = align_s2_s3(membership)
    signatures, omnibus, missing = hard_signature_tables(
        membership, raw, feature_sets, external, external_kinds, cfg
    )
    pairwise = pairwise_tables(membership, raw, feature_sets, external, external_kinds)
    soft = soft_membership_table(membership, raw, feature_sets, external, cfg)
    robust = robust_s2_s3_signatures(signatures, alignment)

    # Attach 02 bootstrap stability to each hard-signature result.
    if not signatures.empty:
        signatures = signatures.merge(
            stability[["scenario", "community", "bootstrap_stability", "stability_status"]],
            left_on=["scenario", "community"], right_on=["scenario", "community"], how="left"
        )

    alignment.to_csv(dirs["tables"] / "03_graph_s2_s3_community_alignment.csv", index=False)
    omnibus.to_csv(dirs["tables"] / "03_graph_community_feature_omnibus.csv", index=False)
    signatures.to_csv(dirs["tables"] / "03_graph_community_signatures.csv", index=False)
    pairwise.to_csv(dirs["tables"] / "03_graph_community_pairwise.csv", index=False)
    robust.to_csv(dirs["tables"] / "03_graph_robust_s2_s3_signatures.csv", index=False)
    soft.to_csv(dirs["tables"] / "03_graph_soft_membership_associations.csv", index=False)
    missing.to_csv(dirs["tables"] / "03_graph_community_missingness.csv", index=False)

    if not args.dry_run:
        make_figures(signatures, robust, dirs["figures"])

    n_robust = int(robust.robust_signature_s2_s3.sum()) if not robust.empty else 0
    print("\nGRAPH COMMUNITY CHARACTERIZATION")
    print(f"S2/S3 aligned community pairs: {len(alignment)}")
    print(f"Hard-signature tests: {len(signatures)}")
    print(f"Robust S2/S3 signatures: {n_robust}")
    print("\nInterpretation rule:")
    print("A robust signature must pass FDR + minimum effect size in BOTH S2 and S3,")
    print("with the same direction in Jaccard-aligned communities.")
    print("\nORDER TO REVIEW")
    for i, name in enumerate([
        "03_graph_s2_s3_community_alignment.csv",
        "03_graph_community_feature_omnibus.csv",
        "03_graph_community_signatures.csv",
        "03_graph_community_pairwise.csv",
        "03_graph_robust_s2_s3_signatures.csv",
        "03_graph_soft_membership_associations.csv",
        "03_graph_community_missingness.csv",
        "03_graph_robust_community_signatures.png",
        "03_graph_community_signature_heatmap.png",
    ], 1):
        print(f"{i}. {name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    tables02 = common.STUDIES_TABLES_DIR / "graph" / "02_graph_representation_sensitivity"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    parser.add_argument("--graph-lab-candidates", type=Path,
                        default=common.STUDIES_TABLES_DIR / "pharma" / "00_profile_labs" /
                        "00_pharma_graph_lab_candidates.csv")
    parser.add_argument("--baseline-metrics", type=Path,
                        default=common.BLOCKA_INTERMEDIATE_DATA_DIR / "01_table1_baseline" /
                        "01_table1_from_clinical_episode_spine_sjd__baseline_patient_metrics_after_eligibility.csv")
    parser.add_argument("--membership", type=Path,
                        default=tables02 / "02_graph_scenario_patient_membership.csv")
    parser.add_argument("--feature-sets", type=Path,
                        default=tables02 / "02_graph_scenario_feature_sets.csv")
    parser.add_argument("--stability", type=Path,
                        default=tables02 / "02_graph_scenario_community_stability.csv")
    parser.add_argument("--config", type=Path, default=folder / "config_characterization.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

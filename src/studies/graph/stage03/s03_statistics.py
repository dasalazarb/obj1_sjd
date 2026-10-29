"""Statistical primitives preserved from Graph03; exploratory estimates only."""
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu

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

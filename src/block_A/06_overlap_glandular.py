#!/usr/bin/env python3
"""ITEMS 4.1/4.2 — canonical longitudinal glandular overlap analysis.

Phenotypes are derived exactly once on the authoritative clinical-visit spine.
Every baseline, prevalence, incidence, and association result is subsequently
computed from that episode-level product; this module never reconstructs visits
or chooses a baseline from dates.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402
from src.derivations.overlap_flags import (  # noqa: E402
    EXTRAGLANDULAR_DOMAINS,
    GLANDULAR_COLS,
    derive_extraglandular_flags,
    derive_glandular_flags,
    derive_overlap_flags,
    normalize_text,
)

LOG = logging.getLogger(__name__)
DAY_PER_YEAR = 365.25
SPINE_COLUMNS = [
    "patient_id",
    "clinical_episode_id",
    "clinical_anchor_date",
    "clinical_visit",
    "clinical_visit_number",
    "clinical_baseline_episode_id",
    "clinical_baseline_date",
    "is_clinical_baseline",
    "time_since_clinical_baseline_days",
    "time_since_clinical_baseline_years",
]
STATUS_ORDER = [
    "overlap",
    "glandular_only",
    "extraglandular_only",
    "neither",
    "unclassifiable",
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_parquet(path)


def normalize_spine(source: pd.DataFrame) -> pd.DataFrame:
    """Select clinical episodes without changing the upstream episode spine."""
    work = source.copy()
    if "patient_id" not in work and "ids__patient_record_number" in work:
        work["patient_id"] = work["ids__patient_record_number"]
    missing = [c for c in SPINE_COLUMNS if c not in work]
    if missing:
        raise ValueError(
            "Authoritative clinical visit spine columns missing: " + ", ".join(missing)
        )
    work["clinical_anchor_date"] = pd.to_datetime(
        work["clinical_anchor_date"], errors="coerce"
    )
    work["clinical_baseline_date"] = pd.to_datetime(
        work["clinical_baseline_date"], errors="coerce"
    )
    clinical = work["clinical_visit"].eq(True).fillna(False)  # noqa: E712
    return work.loc[clinical].copy()


def validate_spine(episodes: pd.DataFrame) -> None:
    keys = ["patient_id", "clinical_episode_id"]
    assert not episodes.duplicated(keys).any(), (
        "Duplicate patient/clinical episode rows"
    )
    baseline = episodes[episodes["is_clinical_baseline"].eq(True)]  # noqa: E712
    assert baseline.groupby("patient_id").size().le(1).all(), (
        "Multiple clinical baselines for a patient"
    )
    assert baseline["clinical_visit"].eq(True).all(), (
        "A baseline row is not a clinical visit"
    )  # noqa: E712
    assert (
        baseline["clinical_episode_id"]
        .eq(baseline["clinical_baseline_episode_id"])
        .all()
    ), "Baseline episode mismatch"
    assert (
        baseline["clinical_anchor_date"].eq(baseline["clinical_baseline_date"]).all()
    ), "Baseline date mismatch"
    assert baseline["clinical_visit_number"].eq(1).all(), (
        "Clinical baseline is not clinical visit number 1"
    )
    for _, group in episodes.groupby("patient_id", sort=False):
        ordered = group.sort_values(
            ["clinical_visit_number", "clinical_anchor_date"], kind="stable"
        )
        assert ordered["clinical_visit_number"].is_monotonic_increasing, (
            "Clinical visit number is not monotonic"
        )
        assert not ordered["clinical_visit_number"].duplicated().any(), (
            "Clinical visit number does not increase strictly"
        )
        assert ordered["clinical_anchor_date"].is_monotonic_increasing, (
            "Clinical anchor date is not monotonic"
        )


def derive_episode_level(source: pd.DataFrame) -> pd.DataFrame:
    episodes = normalize_spine(source)
    validate_spine(episodes)
    flags = pd.concat(
        [derive_glandular_flags(episodes), derive_extraglandular_flags(episodes)],
        axis=1,
    )
    out = pd.concat([episodes, flags], axis=1)
    out = derive_overlap_flags(out)
    out["overlap_status"] = out["overlap_status"].replace(
        "insufficient_info", "unclassifiable"
    )
    for key, meta in EXTRAGLANDULAR_DOMAINS.items():
        # Friendly domain columns accompany the derivation module's audit columns.
        out[key.upper() if key in {"pns", "cns"} else key] = out[meta["active_col"]]
    return out


def pct(n: float, d: float) -> float:
    return float(100 * n / d) if d else np.nan


def overlap_summary(group: pd.DataFrame) -> dict[str, float | int]:
    counts = group["overlap_status"].value_counts()
    n_eval = int(group["overlap_evaluable"].sum())
    result: dict[str, float | int] = {
        "n_patients": int(group["patient_id"].nunique()),
        "n_evaluable": n_eval,
    }
    for status in STATUS_ORDER:
        n = int(counts.get(status, 0))
        denominator = len(group) if status == "unclassifiable" else n_eval
        result[f"n_{status}"] = n
        result[f"pct_{status}"] = pct(n, denominator)
    result["clinical_percentage_denominator"] = (
        "overlap_evaluable (unclassifiable: all patients)"
    )
    return result


def visit_summaries(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    visits = pd.DataFrame(
        [
            {"clinical_visit_number": number, **overlap_summary(group)}
            for number, group in episodes.groupby(
                "clinical_visit_number", dropna=False, sort=True
            )
        ]
    )
    domain_rows = []
    for number, group in episodes.groupby(
        "clinical_visit_number", dropna=False, sort=True
    ):
        for key, meta in EXTRAGLANDULAR_DOMAINS.items():
            evaluable = group[f"eg_{key}_evaluable"]
            active = evaluable.eq(True) & group[meta["active_col"]].eq(True)
            domain_rows.append(
                {
                    "clinical_visit_number": number,
                    "domain": meta["label"],
                    "n_evaluable": int(evaluable.sum()),
                    "n_active": int(active.sum()),
                    "pct_active": pct(active.sum(), evaluable.sum()),
                }
            )
    return visits, pd.DataFrame(domain_rows)


def baseline_summary(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "measure": status,
            "n": int((baseline["overlap_status"] == status).sum()),
            "denominator": int(baseline["overlap_evaluable"].sum())
            if status != "unclassifiable"
            else len(baseline),
            "pct": pct(
                (baseline["overlap_status"] == status).sum(),
                baseline["overlap_evaluable"].sum()
                if status != "unclassifiable"
                else len(baseline),
            ),
        }
        for status in STATUS_ORDER
    ]
    for key, meta in EXTRAGLANDULAR_DOMAINS.items():
        ev = baseline[f"eg_{key}_evaluable"]
        n = int((baseline[meta["active_col"]].eq(True) & ev.eq(True)).sum())
        rows.append(
            {
                "measure": f"domain_{key}",
                "n": n,
                "denominator": int(ev.sum()),
                "pct": pct(n, ev.sum()),
            }
        )
    return pd.DataFrame(rows)


def domain_incidence(episodes: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, meta in EXTRAGLANDULAR_DOMAINS.items():
        baseline_negative = baseline[
            baseline[f"eg_{key}_evaluable"].eq(True)
            & baseline[meta["active_col"]].eq(False)
        ]
        durations, event_dates, event_times = [], [], []
        eligible_patients = []
        for patient, base_date in baseline_negative.set_index("patient_id")[
            "clinical_anchor_date"
        ].items():
            follow = episodes[
                (episodes["patient_id"] == patient)
                & (episodes["clinical_anchor_date"] > base_date)
                & episodes[f"eg_{key}_evaluable"].eq(True)
            ].sort_values("clinical_anchor_date")
            if follow.empty:
                continue

            eligible_patients.append(patient)
            event = follow[follow[meta["active_col"]].eq(True)]
            end = (
                event.iloc[0]["clinical_anchor_date"]
                if not event.empty
                else follow.iloc[-1]["clinical_anchor_date"]
            )
            duration = max(0.0, (end - base_date).days / DAY_PER_YEAR)
            durations.append(duration)
            if not event.empty:
                event_dates.append(end)
                event_times.append(duration)

        n_at_risk = len(eligible_patients)
        assert len(event_dates) <= n_at_risk, (
            f"Incident {key} events exceed the population at risk"
        )
        assert n_at_risk <= len(baseline_negative), (
            f"The {key} population at risk exceeds baseline-evaluable inactive patients"
        )
        assert len(set(eligible_patients)) == n_at_risk, (
            f"Duplicate patients in the {key} population at risk"
        )
        py = float(sum(durations))
        rows.append(
            {
                "domain": meta["label"],
                "n_at_risk": n_at_risk,
                "n_incident": len(event_dates),
                "pct_incident": pct(len(event_dates), n_at_risk),
                "person_years_observed": py,
                "incidence_rate_per_100_py": 100 * len(event_dates) / py
                if py
                else np.nan,
                "median_time_to_domain_yrs": float(np.median(event_times))
                if event_times
                else np.nan,
                "first_event_date_min": min(event_dates) if event_dates else pd.NaT,
                "first_event_date_max": max(event_dates) if event_dates else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def global_incidence(episodes: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    candidates = baseline[
        baseline.glandular_active.eq(True)
        & baseline.glandular_evaluable.eq(True)
        & baseline.extraglandular_evaluable.eq(True)
        & baseline.extraglandular_active.eq(False)
    ]
    events, times, domains = 0, [], []
    n_at_risk = 0
    for _, base in candidates.iterrows():
        follow = episodes[
            (episodes.patient_id == base.patient_id)
            & (episodes.clinical_anchor_date > base.clinical_anchor_date)
            & episodes.extraglandular_evaluable.eq(True)
        ].sort_values("clinical_anchor_date")
        if follow.empty:
            continue
        n_at_risk += 1
        event = follow[follow.extraglandular_active.eq(True)]
        if event.empty:
            continue
        first = event.iloc[0]
        events += 1
        times.append(
            (first.clinical_anchor_date - base.clinical_anchor_date).days / DAY_PER_YEAR
        )
        domains.extend(
            key
            for key, meta in EXTRAGLANDULAR_DOMAINS.items()
            if pd.notna(first[meta["active_col"]]) and bool(first[meta["active_col"]])
        )
    common_domain = pd.Series(domains).value_counts().index[0] if domains else pd.NA
    return pd.DataFrame(
        [
            {
                "n_at_risk": n_at_risk,
                "n_incident": events,
                "pct_incident": pct(events, n_at_risk),
                "most_common_incident_domain": common_domain,
                "median_time_to_first_incident_extraglandular_yrs": float(
                    np.median(times)
                )
                if times
                else np.nan,
            }
        ]
    )


def _ratio_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    if (a + b) == 0 or (c + d) == 0:
        return np.nan, np.nan, np.nan
    r1, r0 = a / (a + b), c / (c + d)
    if c == 0:
        return (np.inf if r1 > 0 else np.nan), np.nan, np.nan
    if a == 0:
        return 0.0, np.nan, np.nan
    pr = r1 / r0
    se = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    return (
        pr,
        float(np.exp(np.log(pr) - 1.96 * se)),
        float(np.exp(np.log(pr) + 1.96 * se)),
    )


def _bh_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply Benjamini-Hochberg once to the non-missing p-values."""
    out = frame.copy()
    out["q_value_BH_FDR"] = np.nan
    pvals = out.loc[out.p_value.notna(), "p_value"].sort_values()
    if not pvals.empty:
        adjusted = (
            (pvals * len(pvals) / np.arange(1, len(pvals) + 1))[::-1]
            .cummin()[::-1]
            .clip(upper=1)
        )
        out.loc[adjusted.index, "q_value_BH_FDR"] = adjusted
    return out


def _wilson(events: int, total: int) -> tuple[float, float]:
    if not total:
        return np.nan, np.nan
    proportion, z = events / total, 1.959963984540054
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return float(centre - half_width), float(centre + half_width)


def binary_exposure_domain_associations(
    baseline: pd.DataFrame,
    exposure_col: str,
    exposure_evaluable_col: str,
    exposure_label: str,
    analysis_tier: str,
    apply_fdr: bool = True,
) -> pd.DataFrame:
    """Generic complete-case 2x2 analysis preserving tri-state missingness."""
    rows = []
    exposure_known = baseline[exposure_col].notna()
    if exposure_evaluable_col != exposure_col:
        exposure_known &= baseline[exposure_evaluable_col].eq(True)
    for key, meta in EXTRAGLANDULAR_DOMAINS.items():
        outcome_known = baseline[f"eg_{key}_evaluable"].eq(True) & baseline[meta["active_col"]].notna()
        data = baseline[exposure_known & outcome_known]
        exposure = data[exposure_col].astype(bool)
        outcome = data[meta["active_col"]].astype(bool)
        a = int((exposure & outcome).sum())
        b = int((exposure & ~outcome).sum())
        c = int((~exposure & outcome).sum())
        d = int((~exposure & ~outcome).sum())
        assert len(data) == a + b + c + d
        n_pos, n_neg = a + b, c + d
        estimable = n_pos > 0 and n_neg > 0
        table = np.array([[a, b], [c, d]])
        if estimable:
            if table.sum(axis=0).min() == 0:
                odds, p = fisher_exact(table)
                test = "Fisher exact"
            else:
                _, _, _, expected = chi2_contingency(table, correction=False)
                if (table < 5).any() or (expected < 5).any():
                    odds, p = fisher_exact(table)
                    test = "Fisher exact"
                else:
                    _, p, _, _ = chi2_contingency(table, correction=False)
                    odds = a * d / (b * c) if b * c else np.inf
                    test = "chi-square"
        else:
            odds, p, test = np.nan, np.nan, "not estimable"
        pr, pr_l, pr_u = _ratio_ci(a, b, c, d)
        pos_l, pos_u = _wilson(a, n_pos)
        neg_l, neg_u = _wilson(c, n_neg)
        if not n_pos:
            note = f"not estimable: no {exposure_label}-positive patients"
        elif not n_neg:
            note = f"not estimable: no {exposure_label}-negative patients"
        elif c == 0 and a > 0:
            note = "reference group exists, zero observed prevalence; PR is infinite"
        elif a == 0 and c == 0:
            note = "both observed prevalences are zero"
        else:
            note = "estimable"
        if all(x > 0 for x in (a, b, c, d)):
            se = np.sqrt(sum(1 / x for x in (a, b, c, d)))
            or_l, or_u = np.exp(np.log(odds) + np.array([-1, 1]) * 1.96 * se)
        else:
            or_l = or_u = np.nan
        rows.append({
            "exposure": exposure_label, "domain": meta["label"],
            "n_total_baseline": len(baseline),
            "n_evaluable_exposure": int(exposure_known.sum()),
            "n_evaluable_outcome": int(outcome_known.sum()),
            "n_complete": len(data),
            "n_excluded_exposure_unknown": int((~exposure_known).sum()),
            "n_excluded_outcome_unknown": int((exposure_known & ~outcome_known).sum()),
            "n_missing_or_not_evaluable": len(baseline) - len(data),
            "component_pos_domain_pos": a, "component_pos_domain_neg": b,
            "component_neg_domain_pos": c, "component_neg_domain_neg": d,
            "n_component_positive": n_pos, "n_component_negative": n_neg,
            "pct_domain_active_if_component_pos": pct(a, n_pos),
            "pct_domain_active_if_component_neg": pct(c, n_neg),
            "prevalence_positive_group": a / n_pos if n_pos else np.nan,
            "prevalence_positive_group_95CI_lower": pos_l,
            "prevalence_positive_group_95CI_upper": pos_u,
            "prevalence_negative_group": c / n_neg if n_neg else np.nan,
            "prevalence_negative_group_95CI_lower": neg_l,
            "prevalence_negative_group_95CI_upper": neg_u,
            "prevalence_ratio": pr, "PR_95_CI_lower": pr_l,
            "PR_95_CI_upper": pr_u,
            "risk_difference": a / n_pos - c / n_neg if estimable else np.nan,
            "odds_ratio": odds, "OR_95_CI_lower": or_l, "OR_95_CI_upper": or_u,
            "test_used": test, "p_value": p, "analysis_tier": analysis_tier,
            "estimability_note": note,
        })
    result = pd.DataFrame(rows)
    return _bh_fdr(result) if apply_fdr else result.assign(q_value_BH_FDR=np.nan)


def associations(baseline: pd.DataFrame) -> pd.DataFrame:
    """Compatibility output for the non-identifiable global comparison."""
    out = binary_exposure_domain_associations(
        baseline, "glandular_active", "glandular_evaluable", "glandular", "QC"
    )
    # Preserve historical names used by downstream consumers.
    return out.rename(columns={
        "component_pos_domain_pos": "glandular_pos_domain_pos",
        "component_pos_domain_neg": "glandular_pos_domain_neg",
        "component_neg_domain_pos": "glandular_neg_domain_pos",
        "component_neg_domain_neg": "glandular_neg_domain_neg",
        "pct_domain_active_if_component_pos": "pct_domain_active_if_glandular_pos",
        "pct_domain_active_if_component_neg": "pct_domain_active_if_glandular_neg",
    }).assign(
        analysis_status=lambda x: np.where(x.test_used.eq("not estimable"), "not estimable", "estimable"),
        reason=lambda x: np.where(
            x.n_component_negative.eq(0),
            "no observed glandular-negative patients among evaluable baseline patients",
            x.estimability_note,
        ),
    )


GLANDULAR_COMPONENTS = [
    ("Eye dryness", "glandular_eye_dryness_active", "descriptive_only"),
    ("Mouth dryness", "glandular_mouth_dryness_active", "descriptive_only"),
    ("Objective eye findings", "glandular_objective_eye_active", "secondary"),
    ("Objective mouth findings", "glandular_objective_mouth_active", "secondary"),
    ("Salivary gland swelling", "glandular_salivary_gland_swelling_active", "exploratory"),
]

GLANDULAR_COMPONENT_COLS = [column for _, column, _ in GLANDULAR_COMPONENTS]
NO_BIO_HEME_DOMAINS = [
    key for key in EXTRAGLANDULAR_DOMAINS if key not in {"biological", "hematologic"}
]


def add_sensitivity_phenotypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add analysis-only completeness and burden fields without redefining flags."""
    out = frame.copy()
    out["glandular_phenotype_complete"] = out[
        GLANDULAR_COMPONENT_COLS
    ].notna().all(axis=1)
    active_cols = [EXTRAGLANDULAR_DOMAINS[key]["active_col"] for key in NO_BIO_HEME_DOMAINS]
    evaluable_cols = [f"eg_{key}_evaluable" for key in NO_BIO_HEME_DOMAINS]
    complete = out[evaluable_cols].eq(True).all(axis=1)
    out["n_extraglandular_domains_active_no_bio_heme"] = (
        out[active_cols].sum(axis=1).astype("Int64").where(complete)
    )
    return out


def glandular_component_prevalence(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, column, _ in GLANDULAR_COMPONENTS:
        positive = int(baseline[column].eq(True).sum())
        negative = int(baseline[column].eq(False).sum())
        unknown = int(baseline[column].isna().sum())
        evaluable = positive + negative
        rows.append({
            "component": label, "n_total": len(baseline), "n_positive": positive,
            "n_negative": negative, "n_unknown": unknown, "n_evaluable": evaluable,
            "pct_positive_among_evaluable": pct(positive, evaluable),
            "pct_negative_among_evaluable": pct(negative, evaluable),
            "pct_unknown_total": pct(unknown, len(baseline)),
        })
    return pd.DataFrame(rows)


def glandular_burden_distribution(baseline: pd.DataFrame) -> pd.DataFrame:
    burden = baseline.n_glandular_manifestations_active.dropna().astype(int)
    counts = burden.value_counts()
    return pd.DataFrame({
        "n_glandular_manifestations_active": range(6),
        "n_patients": [int(counts.get(i, 0)) for i in range(6)],
        "pct_complete_glandular_phenotype": [pct(counts.get(i, 0), len(burden)) for i in range(6)],
    })


def component_associations(baseline: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, column, tier in GLANDULAR_COMPONENTS:
        frame = binary_exposure_domain_associations(
            baseline, column, column, label, tier, apply_fdr=False
        ).rename(columns={"exposure": "component"})
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    # Prespecified 33-test family; subjective sicca components remain descriptive.
    family = out.analysis_tier.isin(["secondary", "exploratory"])
    adjusted = _bh_fdr(out.loc[family].drop(columns="q_value_BH_FDR"))
    out.loc[family, "q_value_BH_FDR"] = adjusted.q_value_BH_FDR
    return out


def burden_analyses(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete = baseline[
        baseline.n_glandular_manifestations_active.notna()
        & baseline.n_extraglandular_domains_active.notna()
    ].copy()
    rows = []
    for burden in range(1, 6):
        values = complete.loc[
            complete.n_glandular_manifestations_active.eq(burden),
            "n_extraglandular_domains_active",
        ].astype(float)
        rows.append({
            "n_glandular_manifestations_active": burden, "n": len(values),
            "mean": values.mean(), "sd": values.std(), "median": values.median(),
            "q1": values.quantile(.25), "q3": values.quantile(.75),
            "min": values.min(), "max": values.max(),
        })
    if len(complete) >= 2:
        rho, p_value = spearmanr(
            complete.n_glandular_manifestations_active.astype(float),
            complete.n_extraglandular_domains_active.astype(float),
        )
    else:
        rho = p_value = np.nan
    correlation = pd.DataFrame([{"n": len(complete), "spearman_rho": rho,
                                 "p_value": p_value,
                                 "ci_method": "not implemented"}])
    y = complete.n_extraglandular_domains_active.astype(float)
    mean, variance = y.mean(), y.var(ddof=1)
    zero = float(y.eq(0).mean()) if len(y) else np.nan
    model_row = {"n": len(complete), "outcome_mean": mean,
                 "outcome_variance": variance, "proportion_zeros": zero,
                 "variance_to_mean": variance / mean if mean else np.nan,
                 "model_family": "not estimable", "effect_per_plus_1": np.nan,
                 "effect_95_CI_lower": np.nan, "effect_95_CI_upper": np.nan,
                 "p_value": np.nan}
    if len(complete) >= 3 and y.nunique() > 1:
        import statsmodels.api as sm
        x = sm.add_constant(complete.n_glandular_manifestations_active.astype(float))
        family = sm.families.NegativeBinomial(alpha=1.0) if variance > 1.5 * mean else sm.families.Poisson()
        fit = sm.GLM(y, x, family=family).fit()
        coefficient = fit.params["n_glandular_manifestations_active"]
        ci = fit.conf_int().loc["n_glandular_manifestations_active"]
        model_row.update({
            "model_family": "negative binomial" if variance > 1.5 * mean else "Poisson",
            "effect_per_plus_1": np.exp(coefficient),
            "effect_95_CI_lower": np.exp(ci.iloc[0]),
            "effect_95_CI_upper": np.exp(ci.iloc[1]),
            "p_value": fit.pvalues["n_glandular_manifestations_active"],
        })
    return pd.DataFrame(rows), correlation, pd.DataFrame([model_row])


def _poisson_burden_result(
    baseline: pd.DataFrame, outcome: str, definition: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the prespecified complete-case correlation and Poisson model."""
    exposure = "n_glandular_manifestations_active"
    complete = baseline.dropna(subset=[exposure, outcome]).copy()
    rho = p_spearman = np.nan
    if len(complete) >= 2:
        rho, p_spearman = spearmanr(
            complete[exposure].astype(float), complete[outcome].astype(float)
        )
    correlation = pd.DataFrame([{
        "outcome_definition": definition,
        "n_complete": len(complete),
        "spearman_rho": rho,
        "p_value": p_spearman,
    }])
    y = complete[outcome].astype(float)
    mean = y.mean()
    variance = y.var(ddof=1)
    model = {
        "model_family": "Poisson",
        "outcome_definition": definition,
        "n_complete": len(complete),
        "mean_outcome": mean,
        "variance_outcome": variance,
        "variance_to_mean": variance / mean if pd.notna(mean) and mean != 0 else np.nan,
        "beta": np.nan,
        "exp_beta": np.nan,
        "CI_95_lower": np.nan,
        "CI_95_upper": np.nan,
        "p_value": np.nan,
    }
    if len(complete) >= 3 and y.nunique() > 1:
        import statsmodels.api as sm

        x = sm.add_constant(complete[exposure].astype(float))
        fit = sm.GLM(y, x, family=sm.families.Poisson()).fit()
        beta = fit.params[exposure]
        ci = fit.conf_int().loc[exposure]
        model.update({
            "beta": beta,
            "exp_beta": np.exp(beta),
            "CI_95_lower": np.exp(ci.iloc[0]),
            "CI_95_upper": np.exp(ci.iloc[1]),
            "p_value": fit.pvalues[exposure],
        })
    return correlation, pd.DataFrame([model])


def leave_one_domain_out(baseline: pd.DataFrame) -> pd.DataFrame:
    """Re-estimate burden associations after omitting each systemic domain."""
    rows = []
    exclusions: list[tuple[str, str | None]] = [("NONE", None)] + [
        (meta["label"], key) for key, meta in EXTRAGLANDULAR_DOMAINS.items()
    ]
    for label, excluded in exclusions:
        keys = [key for key in EXTRAGLANDULAR_DOMAINS if key != excluded]
        active = [EXTRAGLANDULAR_DOMAINS[key]["active_col"] for key in keys]
        evaluable = [f"eg_{key}_evaluable" for key in keys]
        outcome = baseline[active].sum(axis=1).astype("Int64").where(
            baseline[evaluable].eq(True).all(axis=1)
        )
        work = baseline.assign(_leave_one_outcome=outcome)
        corr, model = _poisson_burden_result(
            work, "_leave_one_outcome", f"Excludes {label}" if excluded else "All 11 domains"
        )
        rows.append({
            "excluded_domain": label,
            "n_complete": int(corr.loc[0, "n_complete"]),
            "spearman_rho": corr.loc[0, "spearman_rho"],
            "spearman_p": corr.loc[0, "p_value"],
            "model_family": model.loc[0, "model_family"],
            "exp_beta": model.loc[0, "exp_beta"],
            "CI_95_lower": model.loc[0, "CI_95_lower"],
            "CI_95_upper": model.loc[0, "CI_95_upper"],
            "model_p": model.loc[0, "p_value"],
        })
    return pd.DataFrame(rows)


def completeness_domain_comparison(baseline: pd.DataFrame) -> pd.DataFrame:
    """Compare each evaluable domain by glandular phenotype completeness."""
    rows = []
    complete_group = baseline.glandular_phenotype_complete.astype(bool)
    for key, meta in EXTRAGLANDULAR_DOMAINS.items():
        known = baseline[f"eg_{key}_evaluable"].eq(True) & baseline[meta["active_col"]].notna()
        outcome = baseline[meta["active_col"]].astype("boolean")
        a = int((known & complete_group & outcome.eq(True)).sum())
        b = int((known & complete_group & outcome.eq(False)).sum())
        c = int((known & ~complete_group & outcome.eq(True)).sum())
        d = int((known & ~complete_group & outcome.eq(False)).sum())
        table = np.array([[a, b], [c, d]])
        row_totals = table.sum(axis=1)
        col_totals = table.sum(axis=0)
        if row_totals.min() == 0:
            p_value = np.nan
            test = "not estimable"
            estimability_note = (
                "not estimable: no patients in one completeness group"
            )
        elif col_totals.min() == 0:
            p_value = np.nan
            test = "not estimable"
            if col_totals[0] == 0:
                estimability_note = (
                    "not estimable: no domain-positive patients in either group"
                )
            else:
                estimability_note = (
                    "not estimable: all evaluable patients are domain-positive"
                )
        else:
            _, _, _, expected = chi2_contingency(table, correction=False)
            if (table < 5).any() or (expected < 5).any():
                _, p_value = fisher_exact(table)
                test = "Fisher exact"
            else:
                _, p_value, _, _ = chi2_contingency(table, correction=False)
                test = "chi-square"
            estimability_note = "estimable"
        pr, lower, upper = _ratio_ci(a, b, c, d)
        rows.append({
            "domain": meta["label"],
            "n_complete_group_evaluable": a + b,
            "n_incomplete_group_evaluable": c + d,
            "complete_group_domain_positive": a,
            "complete_group_domain_negative": b,
            "incomplete_group_domain_positive": c,
            "incomplete_group_domain_negative": d,
            "pct_domain_active_complete": pct(a, a + b),
            "pct_domain_active_incomplete": pct(c, c + d),
            "prevalence_ratio": pr,
            "PR_95_CI_lower": lower,
            "PR_95_CI_upper": upper,
            "risk_difference": a / (a + b) - c / (c + d) if (a + b) and (c + d) else np.nan,
            "test_used": test,
            "estimability_note": estimability_note,
            "p_value": p_value,
        })
    return _bh_fdr(pd.DataFrame(rows))


def _continuous_summary(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "N=0"
    return (
        f"N={len(values)}; mean={values.mean():.3g}; SD={values.std():.3g}; "
        f"median={values.median():.3g}; IQR={values.quantile(.25):.3g}-{values.quantile(.75):.3g}"
    )


def completeness_baseline_comparison(baseline: pd.DataFrame) -> pd.DataFrame:
    """Describe available demographics and systemic burden by completeness."""
    from scipy.stats import mannwhitneyu

    group = baseline.glandular_phenotype_complete.astype(bool)
    rows = []
    continuous = [("n_extraglandular_domains_active", "Extraglandular burden")]
    for candidate in ("age", "age_at_baseline", "demographics__age"):
        if candidate in baseline:
            continuous.append((candidate, "Age"))
            break
    for column, label in continuous:
        complete = pd.to_numeric(baseline.loc[group, column], errors="coerce").dropna()
        incomplete = pd.to_numeric(baseline.loc[~group, column], errors="coerce").dropna()
        p_value = mannwhitneyu(complete, incomplete, alternative="two-sided").pvalue if len(complete) and len(incomplete) else np.nan
        rows.append({
            "variable": label,
            "group_complete_n": len(complete),
            "group_incomplete_n": len(incomplete),
            "complete_summary": _continuous_summary(complete),
            "incomplete_summary": _continuous_summary(incomplete),
            "effect_measure": "mean difference; median difference",
            "effect_value": f"{complete.mean() - incomplete.mean():.6g}; {complete.median() - incomplete.median():.6g}" if len(complete) and len(incomplete) else pd.NA,
            "CI_95_lower": np.nan,
            "CI_95_upper": np.nan,
            "test_used": "Mann-Whitney U",
            "p_value": p_value,
            "q_value_BH_FDR": np.nan,
        })
    # Categorical demographics are included only when already on the spine.
    for candidates, label in [
        (("sex", "gender", "demographics__sex"), "Sex"),
        (("race_ethnicity", "race/ethnicity", "race", "demographics__race"), "Race/ethnicity"),
    ]:
        column = next((item for item in candidates if item in baseline), None)
        if column is None:
            continue
        known = baseline[column].notna()
        contingency = pd.crosstab(group[known], baseline.loc[known, column])
        p_value, test = np.nan, "not estimable"
        if contingency.shape[0] == 2 and contingency.shape[1] >= 2:
            if contingency.shape == (2, 2) and (contingency.to_numpy() < 5).any():
                _, p_value = fisher_exact(contingency)
                test = "Fisher exact"
            else:
                _, p_value, _, _ = chi2_contingency(contingency, correction=False)
                test = "chi-square"
        for category in sorted(baseline.loc[known, column].astype(str).unique()):
            c_n = int((known & group & baseline[column].astype(str).eq(category)).sum())
            i_n = int((known & ~group & baseline[column].astype(str).eq(category)).sum())
            c_d = int((known & group).sum()); i_d = int((known & ~group).sum())
            rows.append({
                "variable": f"{label}: {category}", "group_complete_n": c_d,
                "group_incomplete_n": i_d,
                "complete_summary": f"{c_n} ({pct(c_n, c_d):.1f}%)",
                "incomplete_summary": f"{i_n} ({pct(i_n, i_d):.1f}%)",
                "effect_measure": "risk difference", "effect_value": c_n / c_d - i_n / i_d,
                "CI_95_lower": np.nan, "CI_95_upper": np.nan,
                "test_used": test, "p_value": p_value, "q_value_BH_FDR": np.nan,
            })
    return pd.DataFrame(rows)


def completeness_followup(episodes: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed clinical follow-up from the unchanged episode spine."""
    patient = episodes[episodes.patient_id.isin(baseline.patient_id)].groupby("patient_id").agg(
        n_visits=("clinical_episode_id", "size"),
        last_date=("clinical_anchor_date", "max"),
    )
    base = baseline.set_index("patient_id")
    patient["followup_years"] = (
        patient.last_date - base.loc[patient.index, "clinical_anchor_date"]
    ).dt.days / DAY_PER_YEAR
    patient["complete"] = base.loc[patient.index, "glandular_phenotype_complete"].astype(bool)
    rows = []
    for complete, label in [(True, "COMPLETE"), (False, "INCOMPLETE")]:
        data = patient[patient.complete.eq(complete)]
        duration = data.followup_years.dropna()
        visits = data.n_visits
        has_followup = visits.gt(1)
        rows.append({
            "group": label, "n_patients": len(data),
            "median_clinical_visits": visits.median(),
            "IQR_clinical_visits": f"{visits.quantile(.25):.3g}-{visits.quantile(.75):.3g}",
            "n_with_followup": int(has_followup.sum()),
            "pct_with_followup": pct(has_followup.sum(), len(data)),
            "median_followup_years": duration.median(),
            "IQR_followup_years": f"{duration.quantile(.25):.3g}-{duration.quantile(.75):.3g}" if len(duration) else pd.NA,
        })
    return pd.DataFrame(rows)


def qc_summary(
    source: pd.DataFrame, episodes: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    patient_col = (
        "patient_id" if "patient_id" in source else "ids__patient_record_number"
    )
    baseline_counts = baseline.overlap_status.value_counts()
    n_sicca_positive = int(baseline.sicca_active.eq(True).sum())
    n_sicca_negative = int(baseline.sicca_active.eq(False).sum())
    n_sicca_evaluable = int(baseline.sicca_evaluable.eq(True).sum())
    assert n_sicca_positive + n_sicca_negative == n_sicca_evaluable
    for active_col, evaluable_col in [
        ("objective_glandular_dysfunction_active", "objective_glandular_dysfunction_evaluable"),
        ("objective_or_swelling_glandular_active", "objective_or_swelling_glandular_evaluable"),
    ]:
        positive = int(baseline[active_col].eq(True).sum())
        negative = int(baseline[active_col].eq(False).sum())
        unknown = int(baseline[active_col].isna().sum())
        evaluable = int(baseline[evaluable_col].sum())
        assert positive + negative == evaluable
        assert evaluable + unknown == len(baseline)
    raw_sicca = baseline.get(
        GLANDULAR_COLS["symptom_dry_eye_or_mouth"],
        pd.Series(pd.NA, index=baseline.index),
    ).map(normalize_text)
    values = {
        "n_patients_input": source[patient_col].nunique(),
        "n_clinical_episodes_input": source.clinical_episode_id.nunique(),
        "n_patients_episode_level": episodes.patient_id.nunique(),
        "n_episode_rows_output": len(episodes),
        "duplicate_patient_episode_count": episodes.duplicated(
            ["patient_id", "clinical_episode_id"]
        ).sum(),
        "n_clinical_baseline_rows": len(baseline),
        "n_patients_with_clinical_baseline": baseline.patient_id.nunique(),
        "n_patients_without_clinical_baseline": episodes.patient_id.nunique()
        - baseline.patient_id.nunique(),
        "n_patients_with_multiple_clinical_baselines": (
            baseline.groupby("patient_id").size() > 1
        ).sum(),
        "n_glandular_positive_baseline": int(baseline.glandular_active.eq(True).sum()),
        "n_glandular_negative_baseline": int(baseline.glandular_active.eq(False).sum()),
        "n_glandular_missing_baseline": int(baseline.glandular_active.isna().sum()),
        "n_glandular_evaluable_baseline": int(baseline.glandular_evaluable.sum()),
        "n_objective_glandular_positive_baseline": int(baseline.objective_glandular_dysfunction_active.eq(True).sum()),
        "n_objective_glandular_negative_baseline": int(baseline.objective_glandular_dysfunction_active.eq(False).sum()),
        "n_objective_glandular_unknown_baseline": int(baseline.objective_glandular_dysfunction_active.isna().sum()),
        "n_objective_glandular_evaluable_baseline": int(baseline.objective_glandular_dysfunction_evaluable.sum()),
        "n_objective_or_swelling_positive_baseline": int(baseline.objective_or_swelling_glandular_active.eq(True).sum()),
        "n_objective_or_swelling_negative_baseline": int(baseline.objective_or_swelling_glandular_active.eq(False).sum()),
        "n_objective_or_swelling_unknown_baseline": int(baseline.objective_or_swelling_glandular_active.isna().sum()),
        "n_objective_or_swelling_evaluable_baseline": int(baseline.objective_or_swelling_glandular_evaluable.sum()),
        "n_complete_glandular_burden_baseline": int(baseline.n_glandular_manifestations_active.notna().sum()),
        "n_complete_glandular_and_extraglandular_burden_baseline": int((baseline.n_glandular_manifestations_active.notna() & baseline.n_extraglandular_domains_active.notna()).sum()),
        "n_complete_for_burden_analysis": int((baseline.glandular_phenotype_complete & baseline.n_extraglandular_domains_active.notna()).sum()),
        "n_sicca_positive_baseline": n_sicca_positive,
        "n_sicca_negative_baseline": n_sicca_negative,
        "n_sicca_missing_baseline": len(baseline) - n_sicca_evaluable,
        "n_sicca_evaluable_baseline": n_sicca_evaluable,
        "n_raw_sicca_nonmissing_serializations": int(raw_sicca.ne("").sum()),
        "n_raw_sicca_unparsed_nonmissing": int(
            (raw_sicca.ne("") & baseline.sicca_active.isna()).sum()
        ),
        "n_extraglandular_evaluable_baseline": baseline.extraglandular_evaluable.sum(),
        "n_overlap_evaluable_baseline": baseline.overlap_evaluable.sum(),
        **{f"n_baseline_{s}": baseline_counts.get(s, 0) for s in STATUS_ORDER},
    }
    components = {
        "eye_dryness": "glandular_eye_dryness_active",
        "mouth_dryness": "glandular_mouth_dryness_active",
        "objective_eye": "glandular_objective_eye_active",
        "objective_mouth": "glandular_objective_mouth_active",
        "gland_swelling": "glandular_salivary_gland_swelling_active",
    }
    for label, column in components.items():
        values[f"n_{label}_positive_baseline"] = int(baseline[column].eq(True).sum())
        values[f"n_{label}_negative_baseline"] = int(baseline[column].eq(False).sum())
        values[f"n_{label}_missing_baseline"] = int(baseline[column].isna().sum())
    return pd.DataFrame({"metric": values.keys(), "value": values.values()})


def make_figures(visits: pd.DataFrame, domains: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(visits.clinical_visit_number, visits.pct_overlap, marker="o")
    ax.set(xlabel="Clinical visit number", ylabel="Overlap among evaluable (%)")
    fig.tight_layout()
    fig.savefig(figure_dir / "06_overlap_by_clinical_visit_number.pdf")
    plt.close(fig)
    pivot = domains.pivot(
        index="domain", columns="clinical_visit_number", values="pct_active"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(pivot, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set(
        yticks=range(len(pivot)),
        yticklabels=pivot.index,
        xticks=range(len(pivot.columns)),
        xticklabels=pivot.columns,
        xlabel="Clinical visit number",
    )
    fig.colorbar(image, ax=ax, label="Active among evaluable (%)")
    fig.tight_layout()
    fig.savefig(figure_dir / "06_extraglandular_domains_by_clinical_visit_number.pdf")
    plt.close(fig)


def make_glandular_figures(
    baseline: pd.DataFrame, prevalence: pd.DataFrame, burden: pd.DataFrame,
    objective: pd.DataFrame, components: pd.DataFrame, figure_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    left = np.zeros(len(prevalence))
    for column, label, color in [
        ("n_positive", "Positive", "#2878B5"), ("n_negative", "Negative", "#9DC3E6"),
        ("n_unknown", "Unknown", "#D9D9D9")]:
        ax.barh(prevalence.component, prevalence[column], left=left, label=label, color=color)
        left += prevalence[column].to_numpy()
    ax.invert_yaxis(); ax.set_xlabel("Patients"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figure_dir / "06_glandular_components_status_baseline.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(burden.n_glandular_manifestations_active, burden.n_patients, color="#2878B5")
    ax.set(xlabel="Active glandular manifestations", ylabel="Patients", xticks=range(6))
    fig.tight_layout(); fig.savefig(figure_dir / "06_glandular_burden_distribution_baseline.pdf"); plt.close(fig)

    forest = objective.replace([np.inf, -np.inf], np.nan)
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(forest))
    finite = forest.prevalence_ratio.notna() & forest.PR_95_CI_lower.notna()
    ax.errorbar(forest.loc[finite, "prevalence_ratio"], y[finite],
                xerr=[forest.loc[finite, "prevalence_ratio"] - forest.loc[finite, "PR_95_CI_lower"],
                      forest.loc[finite, "PR_95_CI_upper"] - forest.loc[finite, "prevalence_ratio"]],
                fmt="o", color="#2878B5")
    ax.axvline(1, color="grey", linestyle="--"); ax.set_yticks(y, objective.domain)
    ax.set_xlabel("Prevalence ratio (95% CI)"); ax.invert_yaxis()
    for i, row in objective.iterrows():
        if np.isinf(row.prevalence_ratio): ax.text(ax.get_xlim()[1], i, "∞", ha="right")
    fig.tight_layout(); fig.savefig(figure_dir / "06_objective_glandular_domain_PR_forest.pdf"); plt.close(fig)

    matrix = components.pivot(index="component", columns="domain", values="prevalence_ratio")
    values = np.log(matrix.replace([np.inf, -np.inf], np.nan))
    fig, ax = plt.subplots(figsize=(11, 4.5)); image = ax.imshow(values, cmap="coolwarm", aspect="auto")
    ax.set(yticks=range(len(values.index)), yticklabels=values.index,
           xticks=range(len(values.columns)), xticklabels=values.columns)
    ax.tick_params(axis="x", rotation=45); fig.colorbar(image, ax=ax, label="log(PR)")
    fig.tight_layout(); fig.savefig(figure_dir / "06_glandular_component_extraglandular_heatmap.pdf"); plt.close(fig)

    complete = baseline.dropna(subset=["n_glandular_manifestations_active", "n_extraglandular_domains_active"])
    fig, ax = plt.subplots(figsize=(7, 4.5)); rng = np.random.default_rng(20261005)
    ax.scatter(complete.n_glandular_manifestations_active.astype(float) + rng.uniform(-.12, .12, len(complete)),
               complete.n_extraglandular_domains_active.astype(float) + rng.uniform(-.12, .12, len(complete)),
               alpha=.5, s=20)
    medians = complete.groupby("n_glandular_manifestations_active")["n_extraglandular_domains_active"].median()
    ax.plot(medians.index, medians.values, "o-", color="#C43C39", label="Median")
    ax.set(xlabel="Active glandular manifestations", ylabel="Active extraglandular domains")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figure_dir / "06_glandular_vs_extraglandular_burden.pdf"); plt.close(fig)


def make_sensitivity_figures(
    baseline: pd.DataFrame, leave_one_out: pd.DataFrame, figure_dir: Path
) -> None:
    """Plot sensitivity results using the primary burden figure's styling."""
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    outcome = "n_extraglandular_domains_active_no_bio_heme"
    complete = baseline.dropna(
        subset=["n_glandular_manifestations_active", outcome]
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(20261005)
    ax.scatter(
        complete.n_glandular_manifestations_active.astype(float)
        + rng.uniform(-0.12, 0.12, len(complete)),
        complete[outcome].astype(float) + rng.uniform(-0.12, 0.12, len(complete)),
        alpha=0.5,
        s=20,
    )
    medians = complete.groupby("n_glandular_manifestations_active")[outcome].median()
    ax.plot(medians.index, medians.values, "o-", color="#C43C39", label="Median")
    ax.set(
        xlabel="Active glandular manifestations",
        ylabel="Active extraglandular domains\n(excluding Biological/Hematologic)",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "06_glandular_vs_extraglandular_burden_no_bio_heme.pdf")
    plt.close(fig)

    forest = leave_one_out.replace([np.inf, -np.inf], np.nan)
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(len(forest))
    finite = forest[["exp_beta", "CI_95_lower", "CI_95_upper"]].notna().all(axis=1)
    ax.errorbar(
        forest.loc[finite, "exp_beta"], y[finite],
        xerr=[
            forest.loc[finite, "exp_beta"] - forest.loc[finite, "CI_95_lower"],
            forest.loc[finite, "CI_95_upper"] - forest.loc[finite, "exp_beta"],
        ],
        fmt="o", color="#2878B5",
    )
    labels = ["Primary: exclude none" if x == "NONE" else f"Exclude {x}" for x in forest.excluded_domain]
    ax.axvline(1, color="grey", linestyle="--")
    ax.set(yticks=y, yticklabels=labels, xlabel="Poisson ratio per +1 glandular manifestation (95% CI)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(figure_dir / "06_glandular_burden_leave_one_eg_domain_out_forest.pdf")
    plt.close(fig)


def run(
    input_path: Path, intermediate_dir: Path, table_dir: Path, figure_dir: Path
) -> None:
    source = read_table(input_path)
    episodes = add_sensitivity_phenotypes(derive_episode_level(source))
    baseline = episodes[episodes.is_clinical_baseline.eq(True)].copy()  # noqa: E712
    strict_known = baseline.glandular_objective_eye_active.notna() & baseline.glandular_objective_mouth_active.notna()
    baseline["objective_glandular_complete_case"] = pd.Series(pd.NA, index=baseline.index, dtype="boolean")
    baseline.loc[strict_known, "objective_glandular_complete_case"] = (
        baseline.loc[strict_known, "glandular_objective_eye_active"].astype(bool)
        | baseline.loc[strict_known, "glandular_objective_mouth_active"].astype(bool)
    )
    baseline["objective_glandular_complete_case_evaluable"] = strict_known.astype("boolean")
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(
        intermediate_dir / "06_overlap_episode_level.parquet", index=False
    )
    episodes.to_csv(intermediate_dir / "06_overlap_episode_level.csv", index=False)
    phenotype_audit_cols = [
        "sicca_active",
        "sicca_evaluable",
        "glandular_eye_dryness_active",
        "glandular_mouth_dryness_active",
        "glandular_objective_eye_active",
        "glandular_objective_mouth_active",
        "glandular_salivary_gland_swelling_active",
        "glandular_active",
        "glandular_evaluable",
        "objective_glandular_dysfunction_active",
        "objective_glandular_dysfunction_evaluable",
        "objective_or_swelling_glandular_active",
        "objective_or_swelling_glandular_evaluable",
        "objective_glandular_complete_case",
        "objective_glandular_complete_case_evaluable",
        "n_glandular_manifestations_active",
        "extraglandular_active",
        "extraglandular_evaluable",
        "overlap_active",
        "overlap_evaluable",
        "overlap_status",
        "active_extraglandular_domains",
        "n_extraglandular_domains_active",
        "n_extraglandular_domains_active_no_bio_heme",
        "glandular_phenotype_complete",
    ]
    source_audit_cols = [col for col in GLANDULAR_COLS.values() if col in baseline]
    audit_cols = (
        SPINE_COLUMNS[:5]
        + SPINE_COLUMNS[5:8]
        + source_audit_cols
        + phenotype_audit_cols
    )
    baseline[audit_cols].to_csv(
        intermediate_dir / "06_overlap_baseline_patient_audit.csv", index=False
    )
    qc_summary(source, episodes, baseline).to_csv(
        intermediate_dir / "06_overlap_qc_summary.csv", index=False
    )
    visits, domains = visit_summaries(episodes)
    baseline_summary(baseline).to_csv(
        table_dir / "06_overlap_baseline.csv", index=False
    )
    visits.to_csv(table_dir / "06_overlap_by_clinical_visit_number.csv", index=False)
    domains.to_csv(
        table_dir / "06_extraglandular_domains_by_clinical_visit_number.csv",
        index=False,
    )
    domain_incidence(episodes, baseline).to_csv(
        table_dir / "06_incident_extraglandular_domains.csv", index=False
    )
    global_incidence(episodes, baseline).to_csv(
        table_dir / "06_incident_extraglandular.csv", index=False
    )
    associations(baseline).to_csv(
        table_dir / "06_pairwise_domain_associations_clinical_baseline.csv", index=False
    )
    prevalence = glandular_component_prevalence(baseline)
    burden_distribution = glandular_burden_distribution(baseline)
    objective = binary_exposure_domain_associations(
        baseline, "objective_glandular_dysfunction_active",
        "objective_glandular_dysfunction_evaluable", "objective glandular dysfunction", "primary"
    )
    components = component_associations(baseline)
    objective_or_swelling = binary_exposure_domain_associations(
        baseline, "objective_or_swelling_glandular_active",
        "objective_or_swelling_glandular_evaluable", "objective dysfunction or swelling", "sensitivity"
    )
    strict = binary_exposure_domain_associations(
        baseline, "objective_glandular_complete_case",
        "objective_glandular_complete_case_evaluable", "objective complete case", "sensitivity"
    )
    burden_summary, burden_correlation, burden_model = burden_analyses(baseline)
    no_bio_heme_correlation, no_bio_heme_model = _poisson_burden_result(
        baseline,
        "n_extraglandular_domains_active_no_bio_heme",
        "9 domains excluding Biological and Hematologic",
    )
    completeness_baseline = completeness_baseline_comparison(baseline)
    completeness_domains = completeness_domain_comparison(baseline)
    completeness_followup_table = completeness_followup(episodes, baseline)
    leave_one_out = leave_one_domain_out(baseline)
    prevalence.to_csv(table_dir / "06_glandular_component_prevalence_baseline.csv", index=False)
    burden_distribution.to_csv(table_dir / "06_glandular_burden_distribution_baseline.csv", index=False)
    objective.to_csv(table_dir / "06_objective_glandular_domain_associations_baseline.csv", index=False)
    components.to_csv(table_dir / "06_glandular_component_domain_associations_baseline.csv", index=False)
    objective_or_swelling.to_csv(table_dir / "06_objective_or_swelling_domain_associations_sensitivity.csv", index=False)
    strict.to_csv(table_dir / "06_objective_complete_case_domain_associations_sensitivity.csv", index=False)
    burden_summary.to_csv(table_dir / "06_glandular_vs_extraglandular_burden_summary.csv", index=False)
    burden_correlation.to_csv(table_dir / "06_glandular_extraglandular_burden_correlation.csv", index=False)
    burden_model.to_csv(table_dir / "06_glandular_extraglandular_burden_model.csv", index=False)
    no_bio_heme_correlation.to_csv(
        table_dir / "06_glandular_extraglandular_burden_no_bio_heme_correlation.csv", index=False
    )
    no_bio_heme_model.to_csv(
        table_dir / "06_glandular_extraglandular_burden_no_bio_heme_model.csv", index=False
    )
    completeness_baseline.to_csv(
        table_dir / "06_glandular_completeness_comparison_baseline.csv", index=False
    )
    completeness_domains.to_csv(
        table_dir / "06_glandular_completeness_extraglandular_domains.csv", index=False
    )
    completeness_followup_table.to_csv(
        table_dir / "06_glandular_completeness_followup.csv", index=False
    )
    leave_one_out.to_csv(
        table_dir / "06_glandular_burden_leave_one_eg_domain_out.csv", index=False
    )
    leave_one_out.rename(columns={"excluded_domain": "analysis"}).assign(
        analysis=lambda x: x.analysis.map(
            lambda value: "Primary all EG domains" if value == "NONE" else f"Exclude {value}"
        )
    )[["analysis", "n_complete", "spearman_rho", "spearman_p", "exp_beta", "CI_95_lower", "CI_95_upper", "model_p"]].to_csv(
        table_dir / "06_glandular_burden_robustness_summary.csv", index=False
    )
    make_figures(visits, domains, figure_dir)
    make_glandular_figures(baseline, prevalence, burden_distribution, objective, components, figure_dir)
    make_sensitivity_figures(baseline, leave_one_out, figure_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=common.CLINICAL_VISIT_SPINE_PARQUET
    )
    parser.add_argument(
        "--intermediate-dir", type=Path, default=common.BLOCKA_INTERMEDIATE_DATA_DIR / "06_overlap_glandular"
    )
    parser.add_argument("--table-dir", type=Path, default=common.BLOCKA_TABLES_DIR / "06_overlap_glandular")
    parser.add_argument(
        "--figure-dir", type=Path, default=common.OUTPUTS_DIR / "figures" / "blockA" / "06_overlap_glandular"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    run(args.input, args.intermediate_dir, args.table_dir, args.figure_dir)


if __name__ == "__main__":
    main()

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
from scipy.stats import chi2_contingency, fisher_exact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402
from src.derivations.overlap_flags import (  # noqa: E402
    EXTRAGLANDULAR_DOMAINS,
    derive_extraglandular_flags,
    derive_glandular_flags,
    derive_overlap_flags,
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
            active = group[meta["active_col"]] & evaluable
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
        n = int((baseline[meta["active_col"]] & ev).sum())
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
            baseline[f"eg_{key}_evaluable"] & ~baseline[meta["active_col"]]
        ]
        durations, event_dates, event_times = [], [], []
        eligible_patients = []
        for patient, base_date in baseline_negative.set_index("patient_id")[
            "clinical_anchor_date"
        ].items():
            follow = episodes[
                (episodes["patient_id"] == patient)
                & (episodes["clinical_anchor_date"] > base_date)
                & episodes[f"eg_{key}_evaluable"]
            ].sort_values("clinical_anchor_date")
            if follow.empty:
                continue

            eligible_patients.append(patient)
            event = follow[follow[meta["active_col"]]]
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
        baseline.glandular_active
        & baseline.extraglandular_evaluable
        & ~baseline.extraglandular_active
    ]
    events, times, domains = 0, [], []
    n_at_risk = 0
    for _, base in candidates.iterrows():
        follow = episodes[
            (episodes.patient_id == base.patient_id)
            & (episodes.clinical_anchor_date > base.clinical_anchor_date)
            & episodes.extraglandular_evaluable
        ].sort_values("clinical_anchor_date")
        if follow.empty:
            continue
        n_at_risk += 1
        event = follow[follow.extraglandular_active]
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
            if first[meta["active_col"]]
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
    if a == 0 or c == 0:
        return r1 / r0 if r0 else np.nan, np.nan, np.nan
    pr = r1 / r0
    se = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    return (
        pr,
        float(np.exp(np.log(pr) - 1.96 * se)),
        float(np.exp(np.log(pr) + 1.96 * se)),
    )


def associations(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, meta in EXTRAGLANDULAR_DOMAINS.items():
        complete = baseline.glandular_evaluable & baseline[f"eg_{key}_evaluable"]
        data = baseline[complete]
        g = data.glandular_active
        e = data[meta["active_col"]]
        a, b, c, d = (
            int((g & e).sum()),
            int((g & ~e).sum()),
            int((~g & e).sum()),
            int((~g & ~e).sum()),
        )
        table = np.array([[a, b], [c, d]])
        estimable = bool(
            table.sum() and table.sum(axis=0).min() and table.sum(axis=1).min()
        )
        if estimable:
            _, _, _, expected = chi2_contingency(table, correction=False)
            sparse = (table < 5).any() or (expected < 5).any()
            if sparse:
                odds, p = fisher_exact(table)
                test = "Fisher exact"
            else:
                _, p, _, _ = chi2_contingency(table, correction=False)
                odds = a * d / (b * c) if b * c else np.inf
                test = "chi-square"
        else:
            odds, p, test = np.nan, np.nan, "not estimable"
        pr, pr_l, pr_u = _ratio_ci(a, b, c, d)
        if all(x > 0 for x in (a, b, c, d)):
            se = np.sqrt(sum(1 / x for x in (a, b, c, d)))
            or_l, or_u = np.exp(np.log(odds) + np.array([-1, 1]) * 1.96 * se)
        else:
            or_l = or_u = np.nan
        rows.append(
            {
                "domain": meta["label"],
                "n_complete": len(data),
                "n_missing_or_not_evaluable": len(baseline) - len(data),
                "glandular_pos_domain_pos": a,
                "glandular_pos_domain_neg": b,
                "glandular_neg_domain_pos": c,
                "glandular_neg_domain_neg": d,
                "pct_domain_active_if_glandular_pos": pct(a, a + b),
                "pct_domain_active_if_glandular_neg": pct(c, c + d),
                "prevalence_ratio": pr,
                "PR_95_CI_lower": pr_l,
                "PR_95_CI_upper": pr_u,
                "risk_difference": (a / (a + b) - c / (c + d))
                if (a + b) * (c + d)
                else np.nan,
                "odds_ratio": odds,
                "OR_95_CI_lower": or_l,
                "OR_95_CI_upper": or_u,
                "test_used": test,
                "p_value": p,
            }
        )
    out = pd.DataFrame(rows)
    valid = out.p_value.notna()
    pvals = out.loc[valid, "p_value"].sort_values()
    adjusted = (
        (pvals * len(pvals) / np.arange(1, len(pvals) + 1))[::-1]
        .cummin()[::-1]
        .clip(upper=1)
    )
    out["q_value_BH_FDR"] = np.nan
    out.loc[adjusted.index, "q_value_BH_FDR"] = adjusted
    return out


def qc_summary(
    source: pd.DataFrame, episodes: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    patient_col = (
        "patient_id" if "patient_id" in source else "ids__patient_record_number"
    )
    baseline_counts = baseline.overlap_status.value_counts()
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
        "n_glandular_evaluable_baseline": baseline.glandular_evaluable.sum(),
        "n_extraglandular_evaluable_baseline": baseline.extraglandular_evaluable.sum(),
        "n_overlap_evaluable_baseline": baseline.overlap_evaluable.sum(),
        **{f"n_baseline_{s}": baseline_counts.get(s, 0) for s in STATUS_ORDER},
    }
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


def run(
    input_path: Path, intermediate_dir: Path, table_dir: Path, figure_dir: Path
) -> None:
    source = read_table(input_path)
    episodes = derive_episode_level(source)
    baseline = episodes[episodes.is_clinical_baseline.eq(True)].copy()  # noqa: E712
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(
        intermediate_dir / "06_overlap_episode_level.parquet", index=False
    )
    episodes.to_csv(intermediate_dir / "06_overlap_episode_level.csv", index=False)
    audit_cols = (
        SPINE_COLUMNS[:5]
        + SPINE_COLUMNS[5:8]
        + [
            "glandular_active",
            "glandular_evaluable",
            "extraglandular_active",
            "extraglandular_evaluable",
            "overlap_active",
            "overlap_evaluable",
            "overlap_status",
            "active_extraglandular_domains",
            "n_extraglandular_domains_active",
        ]
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
    make_figures(visits, domains, figure_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=common.CLINICAL_VISIT_SPINE_PARQUET
    )
    parser.add_argument(
        "--intermediate-dir", type=Path, default=common.BLOCKA_INTERMEDIATE_DATA_DIR
    )
    parser.add_argument("--table-dir", type=Path, default=common.BLOCKA_TABLES_DIR)
    parser.add_argument(
        "--figure-dir", type=Path, default=common.OUTPUTS_DIR / "figures" / "blockA"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    run(args.input, args.intermediate_dir, args.table_dir, args.figure_dir)


if __name__ == "__main__":
    main()

"""Controlled-research companion figures for observed ESSDAI patterns v2.

Never use these post-index, retrospective shapes as prospective predictors.
Each panel labels its observational denominator; no patient IDs are plotted.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

LABELS = {
    "within_threshold_observed": "Within threshold",
    "worsening_observed": "Worsening",
    "improvement_observed": "Improvement",
    "intermittent_observed": "Intermittent",
}
COLORS = {
    "within_threshold_observed": "#8d9aa5",
    "worsening_observed": "#d28641",
    "improvement_observed": "#369a90",
    "intermittent_observed": "#597fbd",
}
STRATA = {
    "index_source": ("essdai_index_source", [
        ("official_baseline", "Official ESSDAI baseline"),
        ("first_observed_after_official_baseline", "Later first ESSDAI")]),
    "index_state": ("index_activity_state", [
        ("Low", "Low"), ("Moderate", "Moderate"), ("High", "High")]),
    "baseline_pop": ("baseline_pop_official", [
        ("Pop1", "Pop 1"), ("Pop2", "Pop 2"),
        ("Pop3", "Pop 3"), ("Unclassifiable", "Unclassifiable")]),
}


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    return path


def plot_pattern_strata(audit, dimension, path):
    """Count mutually exclusive v2 labels within each stratum, at each threshold."""
    column, groups = STRATA[dimension]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.5, len(groups) * 1.15 + 2.5)),
                             sharex=True, sharey=True)
    for ax, k in zip(axes, (3, 5)):
        base = audit.loc[audit.threshold_abs.eq(k)]
        y = np.arange(len(groups))
        left = np.zeros(len(groups))
        group_n = []
        for key, _ in groups:
            g = base.loc[base[column].eq(key)]
            group_n.append(int(g.n_distinct_essdai_dates.ge(2).sum()))
        for pattern in LABELS:
            n = np.array([int((base[column].eq(key) &
                    base.observed_pattern.eq(pattern) &
                    base.n_distinct_essdai_dates.ge(2)).sum()) for key, _ in groups])
            ax.barh(y, n, left=left, color=COLORS[pattern],
                    height=.68, label=LABELS[pattern])
            left += n
        for i, total in enumerate(group_n):
            ax.text(total + .3, i, f"n={total}", va="center", fontsize=9)
        ax.set_yticks(y, [label for _, label in groups])
        ax.invert_yaxis()
        ax.set(title=f"Observed ESSDAI patterns (±{k})", xlabel="Patients with 2+ ESSDAI dates")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(x=.17)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("v2: complete observed-history classifications, not fixed-horizon risks", fontsize=13)
    fig.text(.5, .025,
             "Intermittence requires 3+ dates; patients with only 2 dates have provisional direction.\n"
             "Official baseline and later ESSDAI index are distinct. Controlled research; disclosure review required.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .93))
    return _save(fig, path)


def plot_index_support(audit, path):
    """Clearly distinguish cohort eligibility from longitudinal shape eligibility."""
    frame = audit.loc[audit.threshold_abs.eq(3)]
    counts = [
        int(frame.n_distinct_essdai_dates.lt(2).sum()),
        int(frame.n_distinct_essdai_dates.eq(2).sum()),
        int(frame.n_distinct_essdai_dates.ge(3).sum()),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(3)
    bars = ax.barh(y, counts, color=["#a6adb6", "#89b4cb", "#356583"], height=.6)
    ax.bar_label(bars, padding=4)
    ax.set_yticks(y, ["<2 distinct dates: insufficient", "2 dates: provisional direction",
                      "3+ dates: intermittence evaluable"])
    ax.invert_yaxis()
    ax.set(xlabel="Patients", title=f"Step 13 v2 cohort support (Step 11 N={len(frame)})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=.18)
    fig.text(.5, .02, "Later first ESSDAI is not the official clinical baseline. Controlled research.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    return _save(fig, path)


def plot_pattern_trajectories(audit, sequence, k, path):
    """Patient paths use real dates relative to first observed ESSDAI, never nominal visits."""
    frame = audit.loc[(audit.threshold_abs.eq(k)) &
                      audit.n_distinct_essdai_dates.ge(2)]
    lookup = frame.set_index("patient_id")
    series = sequence.loc[sequence.patient_id.isin(lookup.index)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    flat = axes.flatten()
    # Show delta from the individual ESSDAI index, not raw score. No patient labels.
    for ax, (pattern, title) in zip(flat, LABELS.items()):
        g = frame.loc[frame.observed_pattern.eq(pattern)]
        subset = series.loc[series.patient_id.isin(g.patient_id)]
        for pid, visits in subset.groupby("patient_id", sort=False):
            visits = visits.sort_values("time_since_essdai_index_years")
            xs = pd.to_numeric(visits.time_since_essdai_index_years).to_numpy(dtype=float)
            ys = pd.to_numeric(visits["essdai__total"]).to_numpy(dtype=float)
            if not len(xs):
                continue
            n = int(lookup.loc[pid, "n_distinct_essdai_dates"])
            ax.plot(xs, ys - ys[0], color=COLORS[pattern],
                    alpha=.4 if n >= 3 else .3, linewidth=1.3,
                    linestyle="-" if n >= 3 else "--")
            ax.scatter(xs, ys - ys[0], color=COLORS[pattern], alpha=.5, s=9)
        ax.axhline(0, color="0.6", lw=.8)
        ax.axhline(k, color="#d28641", ls=":", lw=.9)
        ax.axhline(-k, color="#369a90", ls=":", lw=.9)
        n_two = int(g.n_distinct_essdai_dates.eq(2).sum())
        ax.set_title(f"{title}: n={len(g)} (2 dates: {n_two})")
        ax.set(xlabel="Years since first observed ESSDAI index", ylabel="Change in ESSDAI")
    fig.suptitle(f"Step 13 v2 observed trajectories, ±{k} points", fontsize=14)
    fig.text(.5, .01, "Dashed lines: only 2 dates; paths join intermittent assessments, not continuous progression.\n"
             "No IDs shown. Trajectories require institutional disclosure review before external release.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    return _save(fig, path)


def make_companion_figures(audit, sequence, directory):
    directory = Path(directory)
    paths = [plot_index_support(audit, directory/"13_essdai_index_support_v2.pdf")]
    for dimension in STRATA:
        paths.append(plot_pattern_strata(
            audit, dimension, directory/f"13_essdai_pattern_by_{dimension}_v2.pdf"))
    for k in (3, 5):
        paths.append(plot_pattern_trajectories(
            audit, sequence, k, directory/f"13_essdai_individual_trajectories_delta{k}_v2.pdf"))
    return paths

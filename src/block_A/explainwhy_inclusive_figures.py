"""Controlled figures; individual relative times remain disclosure-sensitive."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.block_A import observed_patterns_v2 as v2
from src.block_A.explainwhy_inclusive import PREFIX, DATE, SCORE


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def plot_trajectories(cohort, sequence, clinical_clock=False):
    eligible = cohort.loc[cohort.eligible].set_index("patient_id")
    fig, ax = plt.subplots(figsize=(10, 6))
    n_markers = 0
    for pid, g in sequence.loc[sequence.patient_id.isin(eligible.index)].groupby("patient_id"):
        g = g.sort_values(DATE)
        r = eligible.loc[pid]
        origin = r.official_clinical_baseline_date if clinical_clock else r.essdai_index_date
        xs = (g[DATE]-origin).dt.days / 365.25
        fallback = r.index_source == v2.FALLBACK
        color = "#d28641" if fallback else "#356583"
        ax.plot(xs, g[SCORE], alpha=.35, lw=.9, color=color)
        artist = ax.scatter([xs.iloc[0]], [g[SCORE].iloc[0]], marker="x" if fallback else "o",
                            s=44 if fallback else 8, color=color, zorder=5)
        n_markers += len(artist.get_offsets()) if fallback else 0
    n_fallback = int(eligible.index_source.eq(v2.FALLBACK).sum())
    n_official = int(eligible.index_source.eq(v2.OFFICIAL).sum())
    ax.legend(handles=[Line2D([], [], marker="x", linestyle="none", color="#d28641",
        label=f"× Alternative ESSDAI index (n={n_fallback})"),
        Line2D([], [], marker="o", linestyle="none", color="#356583",
        label=f"Official ESSDAI baseline (n={n_official})")])
    ax.set(xlabel="Years since official clinical baseline" if clinical_clock else "Years since ESSDAI index",
           ylabel="Observed ESSDAI", title="Clinical clock: delayed ESSDAI entry" if clinical_clock else "Inclusive primary: ESSDAI index clock")
    fig.text(.5, .01, "Controlled individual trajectories; institutional disclosure review required.\n"
        "No pre-index observations imputed; joined points do not establish continuous progression.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, 1))
    return fig, dict(check="clinical_clock_markers" if clinical_clock else "index_clock_markers",
        n_expected_x=n_fallback, n_observed_x=n_markers, status="PASS" if n_markers == n_fallback else "FAIL")


def make_figures(cohort, sequence, patterns, results, summaries, directory, input_origin):
    directory = Path(directory)
    checks = []
    for clinical, name in ((False, "essdai_spaghetti_index_markers"), (True, "essdai_clinical_clock_entry")):
        fig, qc = plot_trajectories(cohort, sequence, clinical)
        fig.suptitle(input_origin + " · " + "PENDING CLINICAL / PRIVACY REVIEW", fontsize=10)
        _save(fig, directory / (PREFIX+name+".pdf"))
        checks.append(qc)
    # One point per eligible person; fallback shape does not imply causality.
    p = patterns.loc[patterns.threshold_abs.eq(3)]
    fig, ax = plt.subplots(figsize=(8, 5))
    observed_x = 0
    for source, marker in ((v2.OFFICIAL, "o"), (v2.FALLBACK, "x")):
        g = p.loc[p.essdai_index_source.eq(source)]
        artist = ax.scatter(g.followup_years_from_essdai_index, g.max_delta_from_essdai_index,
                            marker=marker, alpha=.65, label=f"{source} (n={len(g)})")
        observed_x += len(artist.get_offsets()) if source == v2.FALLBACK else 0
    ax.set(xlabel="Observed years since ESSDAI index", ylabel="Maximum post-index ESSDAI change",
           title="Controlled individual summary; disclosure review pending")
    ax.legend(fontsize=8)
    _save(fig, directory/(PREFIX+"patient_change_scatter.pdf"))
    checks.append(dict(check="scatter_unique_patients", n_expected_x=int(p.essdai_index_source.eq(v2.FALLBACK).sum()),
        n_observed_x=observed_x, status="PASS" if observed_x == p.essdai_index_source.eq(v2.FALLBACK).sum()
        and not p.patient_id.duplicated().any() else "FAIL"))
    fig, ax = plt.subplots(figsize=(9, 5))
    g = summaries.loc[summaries.index_source.eq("all")]
    ax.bar(np.arange(len(g)), g.n_events)
    ax.set_xticks(np.arange(len(g)), [f"{r.analysis}\n+{r.threshold_abs}; N={r.n_eligible}" for r in g.itertuples()], rotation=20, ha="right", fontsize=7)
    ax.set(ylabel="First observed detections", title="Descriptive counts; unequal follow-up is not comparable incidence")
    fig.tight_layout()
    _save(fig, directory/(PREFIX+"first_event_comparison.pdf"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, pop in zip(axes, ("inclusive_primary", "official_only")):
        g = results.loc[results.population.eq(pop) & results.analysis_type.eq("baseline") &
                        results.outcome.eq("first_delta3") & results.valid_for_inference.eq(True)]
        if g.empty:
            ax.text(.5, .5, "not_estimable", transform=ax.transAxes, ha="center")
        else:
            y = np.arange(len(g))
            ax.errorbar(g.estimate, y, xerr=[g.estimate-g.ci95_low, g.ci95_high-g.estimate], fmt="o")
            ax.set_yticks(y, g.predictor_label, fontsize=7)
            ax.set_xscale("log")
        ax.axvline(1, color="0.5", ls=":")
        eligible = cohort.loc[cohort.eligible]
        ax.set(title=pop, xlabel="HR: first observed +3 detection")
        ax.text(.5, -.19, f"Cohort official={eligible.index_source.eq(v2.OFFICIAL).sum()}, "
            f"fallback={eligible.index_source.eq(v2.FALLBACK).sum()}; fitted fallback in model table", transform=ax.transAxes,
            ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .06, 1, 1))
    _save(fig, directory/(PREFIX+"delta3_forest.pdf"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, support in zip(axes, (2, 3)):
        data = p.loc[p.n_distinct_essdai_dates.eq(2) if support == 2 else p.n_distinct_essdai_dates.ge(3)]
        bottom = np.zeros(2)
        for pattern in v2.PATTERNS:
            counts = [int((data.essdai_index_source.eq(s) & data.observed_pattern.eq(pattern)).sum()) for s in (v2.OFFICIAL, v2.FALLBACK)]
            ax.bar([0, 1], counts, bottom=bottom, label=pattern, hatch="//" if support == 2 else None)
            bottom += counts
        ax.set_xticks([0, 1], ["Official", "Alternative"])
        ax.set(title="2 dates: direction only" if support == 2 else "3+ dates: intermittence evaluable", ylabel="Patients")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=7, ncol=2)
    fig.tight_layout(rect=(0, .11, 1, 1))
    _save(fig, directory/(PREFIX+"pattern_by_index_source.pdf"))
    return pd.DataFrame(checks)

"""Controlled figures; individual relative times remain disclosure-sensitive."""
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

from src.block_A import observed_patterns_v2 as v2
from src.block_A.explainwhy_inclusive import PREFIX, DATE, SCORE


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def plot_trajectories(cohort, sequence, clinical_clock=False, input_origin="unspecified"):
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
        ax.plot(xs, g[SCORE], alpha=.65 if fallback else .18, lw=1.1 if fallback else .7, color=color)
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
    fig.text(.5, .01, input_origin + " · PENDING CLINICAL / PRIVACY REVIEW\n"
        + f"Patients={len(eligible)}; observations={len(sequence.loc[sequence.patient_id.isin(eligible.index)])}. "
        + "Controlled trajectories; institutional disclosure review required.\n"
        "No pre-index observations imputed; joined points do not establish continuous progression.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .96))
    return fig, dict(check="clinical_clock_markers" if clinical_clock else "index_clock_markers",
        n_expected_x=n_fallback, n_observed_x=n_markers, status="PASS" if n_markers == n_fallback else "FAIL")


def plot_delta3_forest(results, input_origin="unspecified"):
    """One ordered predictor catalogue, including unsupported models, on both axes."""
    selected = results.loc[results.population.isin(["inclusive_primary", "official_only"]) &
        results.analysis_type.eq("baseline") & results.outcome.eq("first_delta3") &
        results.sensitivity.isin(["none", "continuous_laboratory", "official_only:none",
                                "official_only:continuous_laboratory"])].copy()
    if selected.duplicated(["population", "predictor"]).any():
        raise ValueError("forest: ambiguous primary predictor/population rows")
    catalog = selected[["predictor", "predictor_label"]].drop_duplicates("predictor").sort_values("predictor")
    fig, axes = plt.subplots(1, 2, figsize=(20, max(5, .48*len(catalog)+2)), sharey=True)
    checks, bounds = [], [1.]
    for ax, population in zip(axes, ("inclusive_primary", "official_only")):
        records = selected.loc[selected.population.eq(population)].set_index("predictor")
        plotted = 0
        for y, item in enumerate(catalog.itertuples(index=False)):
            row = records.loc[item.predictor] if item.predictor in records.index else None
            valid = row is not None and row.valid_for_inference
            if valid:
                lo, estimate, hi = float(row.ci95_low), float(row.estimate), float(row.ci95_high)
                if not (np.isfinite([lo, estimate, hi]).all() and 0 < lo <= estimate <= hi):
                    raise ValueError("forest: invalid HR confidence interval")
                color = "#d28641" if "continuous_laboratory" in row.sensitivity else "#356583"
                ax.errorbar(estimate, y, xerr=[[estimate-lo], [hi-estimate]], fmt="o", color=color)
                bounds.extend([lo, hi])
                plotted += 1
            else:
                ax.text(.5, y, "not estimable", transform=ax.get_yaxis_transform(), ha="center", fontsize=7, color="0.4")
            if row is not None:
                def number(field):
                    value = row.get(field, np.nan)
                    return str(int(value)) if pd.notna(value) else "NA"
                def probability(field):
                    value = row.get(field, np.nan)
                    return f"{value:.3g}" if pd.notna(value) else "NA"
                stats = (f"N={number('n_fitted')}; E={number('n_events')}; alt={number('n_fitted_fallback')}\n"
                         f"p={probability('p_value')}; qF={probability('q_value_family')}; qO={probability('q_value_outcome_wide')}")
            else:
                stats = "model row unavailable"
            ax.text(1.03, y, stats, transform=ax.get_yaxis_transform(), va="center", fontsize=7)
        ax.set_xscale("log")
        ax.axvline(1, color="0.5", ls=":")
        ax.set(title=population, xlabel="HR: first observed ΔESSDAI ≥3", ylim=(len(catalog)-.4, -.8))
        ax.set_yticks(np.arange(len(catalog)), [textwrap.fill(label, 30) for label in catalog.predictor_label], fontsize=8)
        ax.tick_params(axis="y", labelleft=True)
        checks.append(dict(check="forest_rows:"+population, n_expected_x=len(catalog),
            n_observed_x=len(ax.get_yticks()), n_estimate_artists=plotted,
            status="PASS" if len(ax.containers) == plotted and len(ax.get_yticks()) == len(catalog) else "FAIL"))
    limits = (min(bounds)/1.15, max(bounds)*1.15)
    for ax in axes:
        ax.set_xlim(limits)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=8))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.text(.5, .025, input_origin+" · PENDING CLINICAL / PRIVACY REVIEW\n"
        "N: fitted patients; E: model events; alt: fitted alternative-index patients. Unsupported N=0 is not cohort N.\n"
        "Blue: standard analyses; orange: continuous-laboratory sensitivity. qF: BH within the reported model family; "
        "qO: separate outcome-wide primary BH (NA when ineligible).\n"
        "Nominal p is not FDR significance. Estimates use model-specific samples; no test of cohort superiority.",
        ha="center", fontsize=8)
    fig.subplots_adjust(left=.15, right=.83, wspace=1.15, top=.94, bottom=.15)
    return fig, checks


def make_figures(cohort, sequence, patterns, results, summaries, directory, input_origin):
    directory = Path(directory)
    checks = []
    for clinical, name in ((False, "essdai_spaghetti_index_markers"), (True, "essdai_clinical_clock_entry")):
        fig, qc = plot_trajectories(cohort, sequence, clinical, input_origin)
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
    expected_fallback = int((cohort.eligible & cohort.index_source.eq(v2.FALLBACK)).sum())
    checks.append(dict(check="scatter_unique_patients", n_expected_x=expected_fallback,
        n_observed_x=observed_x, status="PASS" if observed_x == expected_fallback
        and not p.patient_id.duplicated().any()
        and set(p.patient_id) == set(cohort.loc[cohort.eligible, "patient_id"]) else "FAIL"))
    fig, ax = plt.subplots(figsize=(9, 5))
    g = summaries.loc[summaries.index_source.eq("all")]
    ax.bar(np.arange(len(g)), g.n_events)
    ax.set_xticks(np.arange(len(g)), [f"{r.analysis}\n+{r.threshold_abs}; N={r.n_eligible}" for r in g.itertuples()], rotation=20, ha="right", fontsize=7)
    ax.set(ylabel="First observed detections", title="Descriptive counts; unequal follow-up is not comparable incidence")
    fig.tight_layout()
    _save(fig, directory/(PREFIX+"first_event_comparison.pdf"))
    fig, forest_checks = plot_delta3_forest(results, input_origin)
    _save(fig, directory/(PREFIX+"delta3_forest.pdf"))
    checks.extend(forest_checks)
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

"""Four fixed Step 13 panels; observed points only; PNG output only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.block_A.observed_patterns_v2_figures import COLORS, LABELS
from src.block_A.secondary_trajectory_analysis import summarize


def build_figure(frame, groups, support, spec, threshold, kind, time_axis, unit,
                 origin, minimum_support=5):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    profile = kind == "first_available_profile"
    data = frame.sort_values("variable_measurement_date").groupby("patient_id").head(1) if profile else frame
    numeric, category, _ = summarize(data, spec, threshold, time_axis, unit, minimum_support,
                                   profile=profile and spec.type != "numeric")
    for ax, (pattern, title) in zip(axes.flat, LABELS.items()):
        g = groups.loc[groups.observed_pattern.eq(pattern)]
        subset = data.loc[data.observed_pattern.eq(pattern)]
        s = support.loc[support.observed_pattern.eq(pattern)].iloc[0]
        two = int(g.n_distinct_essdai_dates.eq(2).sum())
        ax.set_title(f"{title}: N_group={len(g)}; any={s.n_any}; repeated={s.n_repeated}\n"
                     f"2 ESSDAI dates (provisional)={two}; reference delayed={s.n_ref_delayed}", fontsize=10, color=COLORS[pattern])
        if spec.type == "numeric":
            ax.set_ylabel(("Change from individual reference" if kind == "delta" else "Observed value")+f" ({unit})")
        else:
            ax.set_ylabel("Cumulative known positivity (%)" if spec.observation_semantics == "cumulative_through_episode"
                          else "Observed positive (%)" if spec.type == "binary" else "Evaluable patients by category (%)")
        if subset.empty:
            ax.text(.5, .5, "No observed data", ha="center", va="center", transform=ax.transAxes)
        elif spec.type == "numeric":
            if kind == "delta" and not subset.n_variable_dates.ge(2).any():
                ax.text(.5, .5, "No repeated observations for change", ha="center", va="center", transform=ax.transAxes)
            for _, visits in subset.groupby("patient_id", sort=False):
                if kind == "delta" and visits.n_variable_dates.iloc[0] < 2:
                    continue
                visits = visits.sort_values("time")
                ys = visits.delta if kind == "delta" else visits.value
                if len(visits) >= 2 and not profile:
                    ax.plot(visits.time, ys, color=COLORS[pattern], alpha=.4, linewidth=1.3,
                            linestyle="--" if visits.n_distinct_essdai_dates.iloc[0] == 2 else "-")
                ax.scatter(visits.time, ys, color=COLORS[pattern], alpha=.6, s=12)
            if kind == "delta":
                ax.axhline(0, color="0.6", lw=.8)
            # No population curve or implied extrapolation is drawn. Exact
            # patient-denominated median/IQR remain inspectable in CSV.
            summary = numeric.loc[numeric.group.eq(pattern)]
            for row in summary.itertuples():
                n = row.n_delta_at_window if kind == "delta" else row.n_patients_unique
                x = subset.loc[subset.window.eq(row.window), "time"].median()
                ax.text(x, .01, f"n={n}"+("*" if n < minimum_support else ""),
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
        else:
            summary = category.loc[category.group.eq(pattern)]
            # A stable set of levels/colors across all panels, including zeros.
            all_levels = sorted(category.category.unique(), key=str)
            palette = plt.get_cmap("viridis", max(2, len(all_levels)))
            for window, rows in summary.groupby("window"):
                bottom = 0.
                x = 0 if profile else (rows.window_start.iloc[0]+rows.window_end.iloc[0])/2
                total = int(rows.n_evaluable.iloc[0])
                for row in rows.itertuples():
                    color = COLORS[pattern] if spec.type == "binary" else palette(all_levels.index(row.category))
                    if spec.type == "binary":
                        # Sparse estimates are isolated points, never curves.
                        ax.scatter([x], [row.percentage], color=color, s=28)
                        if total >= minimum_support:
                            ax.vlines(x, row.ci95_low, row.ci95_high, color=color)
                    else:
                        ax.bar(x, row.percentage, bottom=bottom, width=.7 if profile else .7*(row.window_end-row.window_start),
                               color=color, alpha=1 if total >= minimum_support else .5)
                        bottom += row.percentage
                ax.text(x, .01, f"n={total}"+("*" if total < minimum_support else ""),
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
            ax.set_ylim(-10, 105)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            if profile:
                ax.set_xticks([0], ["First available known profile"])
        if not (profile and spec.type != "numeric"):
            ax.set_xlabel("Years since "+("official clinical baseline" if time_axis == "clinical_baseline" else "ESSDAI index (may differ from clinical baseline)"))
        ax.spines[["top", "right"]].set_visible(False)
    source = groups.essdai_index_source.value_counts()
    mode = "First available profile; no longitudinal inference" if profile else "Observed individual change" if kind == "delta" else "Observed measurements by period"
    fig.suptitle(f"{spec.label} — {mode}\nFixed observed ESSDAI patterns v2 (±{threshold} points)", fontsize=14)
    handles = [Line2D([0], [0], color="0.4", lw=1.3, label="3+ ESSDAI dates"),
               Line2D([0], [0], color="0.4", lw=1.3, ls="--", label="2 ESSDAI dates: provisional")]
    if spec.type in ("ordinal", "categorical") and not category.empty:
        handles += [Line2D([0], [0], color=palette(i), lw=6, label=str(level)) for i, level in enumerate(all_levels)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .087), ncol=min(len(handles), 4), fontsize=8)
    label = "SYNTHETIC SOFTWARE VALIDATION" if origin == "synthetic_validation" else "Controlled research observations"
    note = ("Partial mechanical dependence: these measures also contribute to ESSDAI. " if spec.is_essdai_derived else "")
    note += "Known history, not new tests or seroconversion. " if spec.observation_semantics == "cumulative_through_episode" else ""
    fig.text(.5, .012,
        f"{label}; ESSDAI index source: official={source.get('official_baseline', 0)}, later first ESSDAI={source.get('first_observed_after_official_baseline', 0)}.\n"
        "Reference: own official observed baseline or first valid later measurement, at its actual date; no imputation.\n"
        "Paths join intermittent observations; retrospective descriptive selection, unequal follow-up, informative missingness.\n"
        f"{note}* insufficient support (<{minimum_support}); cross-group comparisons require support in all four groups.",
        ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .13, 1, .92))
    return fig


def plot_variable(frame, groups, support, spec, threshold, kind, path, *, time_axis, unit,
                  origin, dpi=250, minimum_support=5):
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Step 15 output format is PNG only")
    fig = build_figure(frame, groups, support, spec, threshold, kind, time_axis, unit, origin, minimum_support)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", format="png", metadata={"Software": "obj1_sjd Step 15", "Source": origin})
    finally:
        plt.close(fig)
    return path

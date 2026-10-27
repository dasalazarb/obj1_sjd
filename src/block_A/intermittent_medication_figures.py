"""Private three-band review figures; markers are documentation, never exposure."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import textwrap

from src.block_A.intermittent_medication_analysis import DOMAINS, NOTICE


def timeline(visits, jumps, history, k, title, zoom=None):
    """Return a figure in memory. Callers must save only in protected storage."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [3, 3, 4]}, sharex=True)
    activity, domain_ax, medication = axes
    x = pd.to_datetime(visits.clinical_anchor_date)
    y = visits["essdai__total"].to_numpy(dtype=float)
    index = float(visits.essdai_index_score.iloc[0])
    activity.plot(x, y, "o--", color="#293c50", lw=.9, label="Observed ESSDAI (guides only)")
    activity.axhline(index, color="gray", label="ESSDAI index")
    for boundary in (index-k, index+k):
        activity.axhline(boundary, color="#aa6600", ls=":", lw=1)
    for jump in jumps.to_dict("records"):
        color = "#bb4444" if jump["delta_essdai_interval"] > 0 else "#267d70"
        marker = "^" if jump["delta_essdai_interval"] > 0 else "v" if jump["delta_essdai_interval"] < 0 else "s"
        activity.scatter(jump["to_date"], jump["essdai_to"], c=color, marker=marker, s=60, zorder=3)
        if jump["explains_observed_pattern"]:
            activity.scatter(jump["to_date"], jump["essdai_to"], facecolors="none", edgecolors="black", s=135, zorder=4)
        midpoint = jump["from_date"] + (jump["to_date"]-jump["from_date"])/2
        activity.annotate(f'{jump["delta_essdai_interval"]:+g}', (midpoint, (jump["essdai_from"]+jump["essdai_to"])/2), fontsize=8)
        if zoom is None:
            for ax in axes:
                ax.axvspan(jump["from_date"], jump["to_date"], color=color, alpha=.035)
    activity.set_ylabel("Observed ESSDAI")
    activity.legend(fontsize=7, loc="upper left")
    activity.set_title(title + f" | k={k} | rings: threshold entry/exit or sign reversal", fontsize=10)
    cmap = ListedColormap(["#f0f4f7", "#b6cde1", "#6c9bc3", "#305f87"])
    cmap.set_bad("#dddddd")
    numeric_dates = mdates.date2num(x)
    domain_ax.set_yticks(range(len(DOMAINS)), DOMAINS, fontsize=8)
    for row, domain in enumerate(DOMAINS):
        col = f"essdai__{domain}_ordinal_score"
        values = pd.to_numeric(visits[col], errors="coerce").to_numpy(dtype=float) if col in visits else np.full(len(x), np.nan)
        # One column of discrete squares at each observed date: no coloring
        # of the unmeasured space between visits.
        domain_ax.scatter(numeric_dates, [row]*len(x), c=np.ma.masked_invalid(values),
                          marker="s", s=100, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4), plotnonfinite=True)
        for day, value in zip(numeric_dates, values):
            domain_ax.annotate("NA" if np.isnan(value) else f"{value:g}", (day, row), ha="center", va="center", fontsize=6)
    domain_ax.invert_yaxis()
    domain_ax.set_title("Original ordinal domains; NA retained; no weighted point attribution", fontsize=9)
    labels = {}
    unusable = 0
    review_labels = []
    for event in history.to_dict("records"):
        name = next((str(event[c]) for c in ("canonical_medication_name", "medication_name")
                     if pd.notna(event.get(c)) and str(event[c]).strip()), "unresolved")
        if not event["temporally_usable"]:
            unusable += 1
            review_labels.append(f'x {name} / {event["source_kind"]} / {event["reference_kind"]} / {event["temporal_status"]}')
            continue
        label = f'{name} | {event["source_kind"]} | indication unverified'
        lane = labels.setdefault(label, len(labels))
        ctdb = event["source_kind"] == "CTDB"
        marker = "o" if ctdb else ">" if event["reference_kind"] == "order_start" else "<"
        medication.scatter(event["selected_date"], lane, marker=marker, color="#267d70" if ctdb else "#8962a5", s=38)
    medication.set_yticks(list(labels.values()), list(labels), fontsize=7)
    medication.set_ylim(-1, max(1, len(labels)))
    medication.invert_yaxis()
    medication.set_title("CTDB report: circle; BTRIS order_start: >; order_end: < (not a stop)", fontsize=9)
    # Undated and consent-only evidence is explicitly outside the temporal axis.
    medication.text(.01, -.19, f"Not temporally usable: {unusable} references (consent, partial/unknown/future/conflicting dates).\n"
                    "See private evidence table; markers do not imply medication duration or administration.",
                    transform=medication.transAxes, fontsize=8)
    if zoom is not None:
        for ax in axes:
            ax.axvspan(zoom[0], zoom[1], alpha=.1, color="#d5a026")
        medication.set_xlim(zoom[0]-pd.Timedelta(days=90), zoom[1]+pd.Timedelta(days=90))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=9)
    medication.xaxis.set_major_locator(locator)
    medication.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for ax in axes:
        ax.grid(axis="x", alpha=.15)
    review = fig.add_axes([.34, .06, .63, .09])
    review.axis("off")
    lines = [textwrap.shorten(label, width=115, placeholder="…") for label in review_labels[:3]]
    if len(review_labels) > 3:
        lines.append(f"+ {len(review_labels)-3} references retained in the private evidence table")
    review.text(0, 1, "NON-TEMPORAL REVIEW (x; no date position)\n" + "\n".join(lines or ["No uncertain references documented"]), va="top", fontsize=7)
    fig.text(.015, .008, textwrap.fill("RESTRICTED — " + NOTICE, width=180), fontsize=7)
    fig.subplots_adjust(left=.34, right=.97, top=.94, bottom=.25, hspace=.35)
    return fig


def cohort_figure(summary, k, origin="canonical"):
    fig, ax = plt.subplots(figsize=(10, 5))
    subset = summary.loc[summary.temporal_relation.eq("documented_in_interval")]
    labels = subset.jump_direction + " / " + subset.source_kind
    values = subset.n_intervals_documented.fillna(0).to_numpy(dtype=float)
    ax.barh(range(len(subset)), values, color="#527a94")
    ax.set_yticks(range(len(subset)), labels, fontsize=8)
    for i, (_, row) in enumerate(subset.iterrows()):
        if row.suppressed:
            ax.text(0, i, "suppressed", va="center", fontsize=8)
    ax.set_xlabel("Intervals with documented medication references (not confirmed exposure)")
    prefix = "SYNTHETIC VALIDATION | " if origin == "synthetic_validation" else ""
    ax.set_title(prefix + f"Controlled-access descriptive review | k={k} | thresholds never pooled", fontsize=10)
    fig.text(.02, .01, "Sparse rows suppressed. Unique patients, documents and unknown-date proportions: companion table.", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    return fig

#!/usr/bin/env python3
"""Integrate previously-derived Pop, overlap, and PRO data by patient-visit.

This module deliberately only joins existing clinical derivations and creates
longitudinal variables; it never recalculates clinical scores or classifications.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402
from src.derivations.visit_dates import normalize_patient_id  # noqa: E402

KEYS = ["patient_id", "visit_id"]
LEGACY_PATIENT_ID_COLUMN = "ids__patient_record_number"
SHARED_SPINE_COLUMNS = ["visit_date", "visit_number", "observed_baseline_date",
                        "time_since_observed_baseline_days", "time_since_observed_baseline_years",
                        "protocol", "interval_name"]
POP_COLUMNS = ["essdai_total", "esspri_dryness", "esspri_fatigue", "esspri_pain", "esspri_total",
               "esspri_total_observed", "pop_status", "pop_status_display", "pop_missingness_label",
               "baseline_pop_status", "baseline_pop_status_display", "esspri_total_s1_one_proxy",
               "pop_status_s1_one_proxy", "esspri_total_s2_up_to_two_proxies",
               "pop_status_s2_up_to_two_proxies", "esspri_total_s3_all_available", "pop_status_s3_all_available"]
OVERLAP_COLUMNS = ["glandular_active", "glandular_evaluable", "extraglandular_active",
                   "extraglandular_evaluable", "overlap_active", "overlap_status",
                   "n_extraglandular_domains_active", "eg_constitutional_active",
                   "eg_lymphadenopathy_active", "eg_articular_active", "eg_cutaneous_active",
                   "eg_pulmonary_active", "eg_renal_active", "eg_muscular_active", "eg_pns_active",
                   "eg_cns_active", "eg_hematologic_active", "eg_biological_active",
                   "time_since_diagnosis_days", "time_since_diagnosis_years", "dx_date", "dx_date_precision"]
PRO_COLUMNS = ["sf36_physical_functioning", "sf36_role_physical", "sf36_bodily_pain",
               "sf36_general_health", "sf36_vitality", "sf36_social_functioning", "sf36_role_emotional",
               "sf36_mental_health", "sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global",
               "sf36_scoring_valid", "profad_scoring_valid",
               "mdafs_scoring_valid", "esspri_scoring_valid"]


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def canonicalize_patient_id(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return a frame with the canonical patient ID column when possible.

    The diagnosis-anchored overlap export historically retained its raw input
    identifier (``ids__patient_record_number``) instead of ``patient_id``.
    Normalize that identifier with the same routine used to build the visit
    spine so its values remain valid merge keys (for example, ``"123.0"``
    becomes ``"123"``).
    """
    if "patient_id" in frame:
        return frame
    if LEGACY_PATIENT_ID_COLUMN not in frame:
        return frame
    canonical = frame.copy()
    canonical["patient_id"] = canonical[LEGACY_PATIENT_ID_COLUMN].map(normalize_patient_id).astype("string")
    return canonical


def assert_unique_keys(frame: pd.DataFrame, source: str) -> None:
    require_columns(frame, KEYS, source)
    duplicates = frame.loc[frame.duplicated(KEYS, keep=False), KEYS]
    if not duplicates.empty:
        examples = duplicates.drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"{source} has {len(duplicates.drop_duplicates())} duplicate patient_id + visit_id keys; examples: {examples}")


def compare_shared_column(base: pd.DataFrame, other: pd.DataFrame, column: str) -> int:
    """Return the count of non-missing, discordant values for a shared column."""
    if column not in base or column not in other:
        return 0
    compared = base[KEYS + [column]].merge(other[KEYS + [column]], on=KEYS, how="inner", suffixes=("_base", "_other"))
    left, right = compared[f"{column}_base"], compared[f"{column}_other"]
    return int((left.notna() & right.notna() & ~left.eq(right)).sum())


def select_source(frame: pd.DataFrame, columns: list[str], source: str) -> pd.DataFrame:
    require_columns(frame, KEYS + ["visit_date"], source)
    return frame[KEYS + ["visit_date"] + [col for col in columns if col in frame]].copy()


def baseline_value(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame.loc[frame.visit_number.eq(0), ["patient_id", column]].set_index("patient_id")[column]
    return frame.patient_id.map(values)


def build_integrated(visit_spine: pd.DataFrame, pop: pd.DataFrame, overlap: pd.DataFrame, pros: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the canonical longitudinal table, returning it and date discrepancy counts."""
    visit_spine, pop, overlap, pros = [
        canonicalize_patient_id(frame, source)
        for source, frame in [("visit_spine", visit_spine), ("pop", pop), ("overlap", overlap), ("pros", pros)]
    ]
    for source, frame in [("visit_spine", visit_spine), ("pop", pop), ("overlap", overlap), ("pros", pros)]:
        assert_unique_keys(frame, source)
    require_columns(visit_spine, KEYS + ["visit_date", "visit_number", "time_since_observed_baseline_days"], "visit_spine")
    spine = visit_spine.copy()
    spine["visit_date"] = pd.to_datetime(spine["visit_date"])
    discrepancy_counts = {f"{source}_{column}": compare_shared_column(spine, frame, column)
                          for source, frame in [("pop", pop), ("overlap", overlap), ("pros", pros)]
                          for column in SHARED_SPINE_COLUMNS if column in frame}
    pop = select_source(pop, POP_COLUMNS, "pop").drop(columns="visit_date")
    overlap = select_source(overlap, OVERLAP_COLUMNS, "overlap").drop(columns="visit_date")
    pros = select_source(pros, PRO_COLUMNS, "pros").drop(columns="visit_date")
    integrated = spine.merge(pop, on=KEYS, how="left", validate="one_to_one")
    integrated = integrated.merge(overlap, on=KEYS, how="left", validate="one_to_one")
    integrated = integrated.merge(pros, on=KEYS, how="left", validate="one_to_one")
    return derive_longitudinal(integrated), discrepancy_counts


def derive_longitudinal(integrated: pd.DataFrame) -> pd.DataFrame:
    """Add availability, lagged, transition, and change variables to an integrated frame."""
    integrated = integrated.sort_values(["patient_id", "visit_date"]).reset_index(drop=True).copy()
    grouped = integrated.groupby("patient_id", sort=False)
    if grouped.visit_number.diff().dropna().lt(0).any():
        raise ValueError("visit_number is inconsistent with patient visit_date ordering")
    integrated["has_pop_data"] = integrated.pop_status.notna()
    integrated["has_overlap_data"] = integrated.overlap_status.notna()
    integrated["has_pro_data"] = integrated[[c for c in ["sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global"] if c in integrated]].notna().any(axis=1)
    for output, source in [("has_essdai", "essdai_total"), ("has_esspri", "esspri_total"), ("has_sf36_pcs", "sf36_pcs"), ("has_sf36_mcs", "sf36_mcs"), ("has_profad", "profad_total"), ("has_mdafs", "mdafs_global")]:
        integrated[output] = integrated[source].notna()
    integrated["n_data_blocks_available"] = integrated[["has_pop_data", "has_overlap_data", "has_pro_data"]].sum(axis=1)
    integrated["n_visits_patient"] = grouped.visit_id.transform("count")
    integrated["is_observed_baseline"] = integrated.visit_number.eq(0)
    integrated["is_last_observed_visit"] = grouped.visit_date.transform("max").eq(integrated.visit_date)
    for output, source, periods in [("previous_visit_id", "visit_id", 1), ("next_visit_id", "visit_id", -1), ("previous_visit_date", "visit_date", 1), ("next_visit_date", "visit_date", -1), ("previous_pop", "pop_status", 1), ("next_pop", "pop_status", -1), ("previous_overlap_status", "overlap_status", 1), ("next_overlap_status", "overlap_status", -1), ("previous_extraglandular_active", "extraglandular_active", 1)]:
        integrated[output] = grouped[source].shift(periods)
    integrated["time_from_previous_visit_days"] = (integrated.visit_date - integrated.previous_visit_date).dt.days
    integrated["time_to_next_visit_days"] = (integrated.next_visit_date - integrated.visit_date).dt.days
    integrated["time_from_previous_visit_years"] = integrated.time_from_previous_visit_days / 365.25
    integrated["time_to_next_visit_years"] = integrated.time_to_next_visit_days / 365.25
    integrated["baseline_pop"] = integrated.get("baseline_pop_status", pd.Series(pd.NA, index=integrated.index)).combine_first(baseline_value(integrated, "pop_status"))
    integrated["baseline_overlap_status"] = baseline_value(integrated, "overlap_status")
    integrated["baseline_extraglandular_active"] = baseline_value(integrated, "extraglandular_active")
    integrated["pop_transition_from_previous"] = transition(integrated.previous_pop, integrated.pop_status)
    integrated["pop_transition_to_next"] = transition(integrated.pop_status, integrated.next_pop)
    valid_pop = {"Pop1", "Pop2", "Pop3"}
    integrated["pop_transition_evaluable"] = integrated.previous_pop.isin(valid_pop) & integrated.pop_status.isin(valid_pop)
    integrated["changed_pop_from_previous"] = pd.Series(pd.NA, index=integrated.index, dtype="boolean")
    mask = integrated.pop_transition_evaluable
    integrated.loc[mask, "changed_pop_from_previous"] = integrated.loc[mask, "previous_pop"].ne(integrated.loc[mask, "pop_status"])
    integrated["overlap_transition_from_previous"] = transition(integrated.previous_overlap_status, integrated.overlap_status)
    sufficient = lambda s: s.notna() & ~s.astype(str).str.contains("insufficient|unknown|unclass", case=False, na=False)
    integrated["overlap_transition_evaluable"] = sufficient(integrated.previous_overlap_status) & sufficient(integrated.overlap_status)
    evaluable = grouped.extraglandular_evaluable.shift(1).eq(True) & integrated.extraglandular_evaluable.eq(True)
    incident = integrated.previous_extraglandular_active.eq(False) & integrated.extraglandular_active.eq(True) & evaluable
    integrated["incident_extraglandular_from_previous"] = incident.astype("boolean").where(evaluable, pd.NA)
    integrated["first_incident_extraglandular"] = (incident & ~integrated.baseline_extraglandular_active.eq(True) & ~incident.groupby(integrated.patient_id).shift(fill_value=False).groupby(integrated.patient_id).cummax()).astype("boolean").where(evaluable, pd.NA)
    delta_map = {"delta_essdai": "essdai_total", "delta_esspri": "esspri_total", "delta_pcs": "sf36_pcs", "delta_mcs": "sf36_mcs", "delta_profad_total": "profad_total", "delta_mdafs_global": "mdafs_global", "delta_n_extraglandular_domains": "n_extraglandular_domains_active"}
    for output, source in delta_map.items(): integrated[output] = grouped[source].diff()
    baseline_map = {"change_from_baseline_essdai": "essdai_total", "change_from_baseline_esspri": "esspri_total", "change_from_baseline_pcs": "sf36_pcs", "change_from_baseline_mcs": "sf36_mcs", "change_from_baseline_profad_total": "profad_total", "change_from_baseline_mdafs_global": "mdafs_global", "change_from_baseline_n_extraglandular_domains": "n_extraglandular_domains_active"}
    for output, source in baseline_map.items(): integrated[output] = integrated[source] - baseline_value(integrated, source)
    for output, source in [("next_essdai", "essdai_total"), ("next_esspri", "esspri_total"), ("next_pcs", "sf36_pcs"), ("next_mcs", "sf36_mcs"), ("next_profad_total", "profad_total"), ("next_mdafs_global", "mdafs_global"), ("next_extraglandular_active", "extraglandular_active"), ("next_n_extraglandular_domains", "n_extraglandular_domains_active")]: integrated[output] = grouped[source].shift(-1)
    integrated["transition_to_pop1_next_visit"] = integrated.pop_status.isin(["Pop2", "Pop3"]) & integrated.next_pop.eq("Pop1")
    integrated["at_risk_transition_to_pop1"] = integrated.pop_status.isin(["Pop2", "Pop3"]) & integrated.next_pop.isin(valid_pop)
    integrated["pcs_decreased_from_previous"] = integrated.delta_pcs < 0
    integrated["mcs_decreased_from_previous"] = integrated.delta_mcs < 0
    integrated["integration_version"] = "v1_observed_pop_overlap_pros"
    integrated["integration_run_date"] = date.today().isoformat()
    return integrated


def transition(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left.astype("string") + "_to_" + right.astype("string")).where(left.notna() & right.notna(), pd.NA)


def coverage(integrated: pd.DataFrame) -> pd.DataFrame:
    measures = {"pop_status": integrated.has_pop_data, "overlap_status": integrated.has_overlap_data, "essdai_total": integrated.has_essdai, "esspri_total": integrated.has_esspri, "sf36_pcs": integrated.has_sf36_pcs, "sf36_mcs": integrated.has_sf36_mcs, "profad_total": integrated.has_profad, "mdafs_global": integrated.has_mdafs, "complete_pop_overlap": integrated.has_pop_data & integrated.has_overlap_data, "complete_pop_pro": integrated.has_pop_data & integrated.has_pro_data, "complete_overlap_pro": integrated.has_overlap_data & integrated.has_pro_data, "complete_all_three_blocks": integrated.n_data_blocks_available.eq(3)}
    n_patients = integrated.patient_id.nunique()
    return pd.DataFrame([{"measure": name, "n_visits_available": int(mask.sum()), "pct_visits_available": 100 * mask.mean(), "n_patients_available": int(integrated.loc[mask, "patient_id"].nunique()), "pct_patients_available": 100 * integrated.loc[mask, "patient_id"].nunique() / n_patients if n_patients else np.nan} for name, mask in measures.items()])


def progress(step: int, total: int, message: str) -> None:
    """Print a compact, readable progress message for command-line runs."""
    width = 24
    completed = round(width * step / total)
    bar = "█" * completed + "░" * (width - completed)
    print(f"\n[{bar}] {step}/{total}  {message}", flush=True)


def report_output(path: Path) -> None:
    """Print each artifact as it is written, relative to the project when possible."""
    try:
        display_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = path
    print(f"  ✓ Generated: {display_path}", flush=True)


# =============================================================================
# FIGURES: comparable longitudinal patient timelines, one panel per baseline Pop
# =============================================================================
FIGURES_DIR = common.OUTPUTS_DIR / "figures" / "blockA"

POP_ORDER = ["Pop1", "Pop2", "Pop3", "Unclassifiable"]
POP_COLORS = {"Pop1": "#E66101", "Pop2": "#5E5AA8", "Pop3": "#1B9E77", "Unclassifiable": "#9E9E9E"}
OVERLAP_ORDER = ["neither", "glandular_only", "extraglandular_only", "overlap", "insufficient_info"]
OVERLAP_COLORS = {"neither": "#9E9E9E", "glandular_only": "#4C78A8", "extraglandular_only": "#59A14F",
                  "overlap": "#E15759", "insufficient_info": "#D9D2CE"}

FIGURE_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.edgecolor": "#8A8A8A",
    "axes.linewidth": 0.8, "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
}
TIME_TICKS = [(0.0, "baseline"), (0.5, "6 mo"), (1.0, "1y"), (2.0, "2y"), (4.0, "4y"),
              (6.0, "6y"), (8.0, "8y"), (10.0, "10y"), (12.0, "12y"), (14.0, "14y")]
TITLE_COLOR = "#C2560B"
GRID_COLOR = "#CFCFCF"
SPAN_COLOR = "#DCDCDC"
MISSING_COLOR = "#BDBDBD"

ROW_HEIGHT_IN = 0.105        # vertical space per patient (compressed y-axis)
PANEL_MIN_IN = 0.85          # floor so tiny Pop groups stay readable
PANEL_GAP_IN = 0.34
FIG_WIDTH_IN = 13.0
TITLE_BLOCK_IN = 1.15        # room for title, subtitle and the top time axis
POINT_SIZE = 26
LEFT_FRAC, RIGHT_FRAC = 0.085, 0.985


def patient_display_order(integrated: pd.DataFrame) -> pd.DataFrame:
    """One patient order shared by every figure: grouped by baseline Pop, longest follow-up first."""
    patient_ids = pd.Index(pd.unique(integrated["patient_id"]), name="patient_id")
    baseline = (integrated.loc[integrated["is_observed_baseline"]]
                .drop_duplicates("patient_id")
                .set_index("patient_id")["baseline_pop"]
                .reindex(patient_ids))
    follow_up = (integrated.groupby("patient_id")["time_since_observed_baseline_years"]
                 .max().reindex(patient_ids).fillna(0.0))
    order = pd.DataFrame({"patient_id": patient_ids, "baseline_pop": baseline.to_numpy(),
                          "followup_years": follow_up.to_numpy()})
    order["baseline_pop"] = order["baseline_pop"].where(order["baseline_pop"].isin(POP_ORDER[:-1]), "Unclassifiable")
    order["pop_rank"] = order["baseline_pop"].map({pop: rank for rank, pop in enumerate(POP_ORDER)})
    order = order.sort_values(["pop_rank", "followup_years", "patient_id"],
                              ascending=[True, False, True], kind="stable").reset_index(drop=True)
    order["row"] = order.groupby("baseline_pop", sort=False).cumcount()
    return order


def _style_panel(ax: plt.Axes, population: str, n_patients: int, xlim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(n_patients - 0.5, -0.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=3, color="#8A8A8A")
    ax.text(-0.008, 0.5, f"{population}\n(n={n_patients})", transform=ax.transAxes, ha="right",
            va="center", color=POP_COLORS[population], fontsize=9, fontweight="bold")
    for position, _ in TIME_TICKS:
        if xlim[0] <= position <= xlim[1]:
            ax.axvline(position, color=GRID_COLOR, linestyle="--", linewidth=0.7, zorder=0)


def _draw_spans(ax: plt.Axes, panel: pd.DataFrame) -> None:
    spans = panel.groupby("row")["time_since_observed_baseline_years"].agg(["min", "max"])
    ax.hlines(spans.index.to_numpy(), spans["min"].to_numpy(), spans["max"].to_numpy(),
              color=SPAN_COLOR, linewidth=0.9, zorder=1)


def plot_measure_by_pop(integrated: pd.DataFrame, order: pd.DataFrame, path: Path, title: str,
                        subtitle: str, value_columns: list[str], value_label: str,
                        footnote: str, categorical: bool = False,
                        marker_labels: list[str] | None = None) -> None:
    """Stack one panel per baseline Pop in a single PDF, sharing scale, order and time axis."""
    panels = order[["patient_id", "baseline_pop", "row"]].rename(columns={"baseline_pop": "panel_pop"})
    data = integrated.drop(columns=["panel_pop"], errors="ignore").merge(panels, on="patient_id", how="inner")
    data = data.loc[data["time_since_observed_baseline_years"].notna()].copy()
    groups = [(pop, int(order.baseline_pop.eq(pop).sum())) for pop in POP_ORDER]
    groups = [(pop, size) for pop, size in groups if size]
    heights = [max(PANEL_MIN_IN, size * ROW_HEIGHT_IN) for _, size in groups]

    two_series = (not categorical) and len(value_columns) > 1
    legend_block_in = 1.60 if (two_series or categorical) else 1.20
    fig_height = TITLE_BLOCK_IN + sum(heights) + PANEL_GAP_IN * (len(groups) - 1) + legend_block_in

    with plt.rc_context(FIGURE_STYLE):
        fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height))
        grid = fig.add_gridspec(len(groups), 1, height_ratios=heights,
                                hspace=PANEL_GAP_IN / (sum(heights) / len(groups)),
                                left=LEFT_FRAC, right=RIGHT_FRAC,
                                top=1 - TITLE_BLOCK_IN / fig_height,
                                bottom=legend_block_in / fig_height)
        axes = [fig.add_subplot(grid[index]) for index in range(len(groups))]

        max_years = float(data["time_since_observed_baseline_years"].max()) if not data.empty else 1.0
        max_years = max(1.0, max_years)
        xlim = (-0.015 * max_years, max_years * 1.04)

        if categorical:
            column = value_columns[0]
            data["plot_category"] = data[column].where(data[column].isin(OVERLAP_ORDER), "insufficient_info")
            normalizer = None
        else:
            observed = data[value_columns].to_numpy(dtype="float64")
            observed = observed[np.isfinite(observed)]
            low, high = (float(observed.min()), float(observed.max())) if observed.size else (0.0, 1.0)
            if low == high:
                low, high = low - 0.5, high + 0.5
            normalizer = Normalize(vmin=low, vmax=high)

        for ax, (population, size) in zip(axes, groups):
            panel = data.loc[data["panel_pop"].eq(population)]
            _style_panel(ax, population, size, xlim)
            if not panel.empty:
                _draw_spans(ax, panel)
            if categorical:
                for category in OVERLAP_ORDER:
                    subset = panel.loc[panel["plot_category"].eq(category)]
                    ax.scatter(subset["time_since_observed_baseline_years"], subset["row"], s=POINT_SIZE,
                               color=OVERLAP_COLORS[category], edgecolor="white", linewidth=0.4, zorder=3)
            else:
                unmeasured = panel.loc[panel[value_columns].isna().all(axis=1)]
                ax.scatter(unmeasured["time_since_observed_baseline_years"], unmeasured["row"], s=13,
                           facecolor="none", edgecolor=MISSING_COLOR, linewidth=0.7, zorder=2)
                for column, marker in zip(value_columns, ["o", "s"]):
                    subset = panel.loc[panel[column].notna()]
                    ax.scatter(subset["time_since_observed_baseline_years"], subset["row"],
                               c=subset[column], cmap="viridis", norm=normalizer, s=POINT_SIZE,
                               marker=marker, edgecolor="white", linewidth=0.4, zorder=3)
            ax.set_xticks(np.arange(0, np.floor(max_years) + 1, 2))
            if ax is not axes[-1]:
                ax.set_xticklabels([])

        top_axis = axes[0].secondary_xaxis("top")
        visible_ticks = [(position, label) for position, label in TIME_TICKS if xlim[0] <= position <= xlim[1]]
        top_axis.set_xticks([position for position, _ in visible_ticks])
        top_axis.set_xticklabels([label for _, label in visible_ticks], fontsize=8, color="#5A5A5A")
        top_axis.tick_params(length=0, pad=2)
        top_axis.spines["top"].set_visible(False)

        axes[-1].set_xlabel("Time since observed baseline (years)")
        fig.supylabel("Patients (shared order across figures; longest follow-up first)", x=0.012, fontsize=9)

        fig.text(LEFT_FRAC, 1 - 0.28 / fig_height, title, color=TITLE_COLOR, fontsize=14.5, ha="left", va="top")
        fig.text(LEFT_FRAC, 1 - 0.56 / fig_height, subtitle, color=TITLE_COLOR, fontsize=10.5, ha="left", va="top")

        if categorical:
            handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=OVERLAP_COLORS[category],
                              markeredgecolor="white", markersize=7, label=category.replace("_", " "))
                       for category in OVERLAP_ORDER]
            fig.legend(handles=handles, title="Overlap status", loc="lower center",
                       bbox_to_anchor=(0.5, 0.55 / fig_height), ncol=5, columnspacing=1.6, handletextpad=0.4)
        else:
            colorbar_axes = fig.add_axes([0.36, 0.62 / fig_height, 0.28, 0.115 / fig_height])
            colorbar = fig.colorbar(ScalarMappable(norm=normalizer, cmap="viridis"),
                                    cax=colorbar_axes, orientation="horizontal")
            colorbar.set_label(value_label, fontsize=9)
            colorbar.outline.set_linewidth(0.6)
            colorbar.ax.tick_params(labelsize=8, length=2.5)
            handles = [Line2D([0], [0], marker=marker, linestyle="none", markerfacecolor="#4C4C4C",
                              markeredgecolor="white", markersize=7, label=label)
                       for marker, label in zip(["o", "s"], marker_labels or [])] if two_series else []
            handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                                  markeredgecolor=MISSING_COLOR, markersize=6, label="visit without this measure"))
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02 / fig_height),
                       ncol=len(handles), columnspacing=1.8, handletextpad=0.4)

        fig.text(0.5, 0.20 / fig_height, footnote, ha="center", va="bottom", fontsize=8.2, color="#4A4A4A")
        fig.savefig(path, bbox_inches=None)
        plt.close(fig)


def generate_longitudinal_figures(integrated: pd.DataFrame) -> list[Path]:
    """Write one PDF per measure, each stacking the four baseline-Pop panels."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    order = patient_display_order(integrated)
    shared_subtitle = ("One panel per baseline Pop; within a panel each row is a patient "
                       "(longest follow-up first) and each point is a visit.")
    figures = [
        dict(filename="10_longitudinal_profad.pdf", title="Longitudinal PROFAD",
             value_columns=["profad_total"], value_label="PROFAD total", categorical=False,
             marker_labels=None,
             footnote=("PROFAD total per visit, colored on a common scale across all panels. "
                       "Open grey circles are visits with no PROFAD score.")),
        dict(filename="10_longitudinal_sf36_pcs.pdf", title="Longitudinal SF-36 PCS",
             value_columns=["sf36_pcs"], value_label="SF-36 PCS", categorical=False, marker_labels=None,
             footnote=("SF-36 physical component summary per visit, colored on a common scale across all "
                       "panels. Open grey circles are visits with no PCS score.")),
        dict(filename="10_longitudinal_sf36_mcs.pdf", title="Longitudinal SF-36 MCS",
             value_columns=["sf36_mcs"], value_label="SF-36 MCS", categorical=False, marker_labels=None,
             footnote=("SF-36 mental component summary per visit, colored on a common scale across all "
                       "panels. Open grey circles are visits with no MCS score.")),
        dict(filename="10_longitudinal_mdafs.pdf", title="Longitudinal MDAFS",
             value_columns=["mdafs_global"], value_label="MDAFS global", categorical=False,
             marker_labels=None,
             footnote=("MDAFS global per visit, colored on a common scale across all panels. "
                       "Open grey circles are visits with no MDAFS score.")),
        dict(filename="10_longitudinal_overlapping.pdf",
             title="Longitudinal glandular / extraglandular overlap",
             value_columns=["overlap_status"], value_label="", categorical=True, marker_labels=None,
             footnote=("Overlap status per visit. 'insufficient info' marks visits where glandular or "
                       "extraglandular activity could not be evaluated.")),
    ]
    paths = []
    for figure in figures:
        path = FIGURES_DIR / figure.pop("filename")
        plot_measure_by_pop(integrated, order, path, subtitle=shared_subtitle, **figure)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spine", type=Path, default=common.VISIT_SPINE_PARQUET); parser.add_argument("--pop", type=Path, default=common.POP_LONGITUDINAL_PARQUET); parser.add_argument("--overlap", type=Path, default=common.OVERLAP_LONGITUDINAL_PARQUET); parser.add_argument("--pros", type=Path, default=common.PROS_LONGITUDINAL_PARQUET); parser.add_argument("--output", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    args = parser.parse_args()
    common.ensure_output_dirs()
    total_steps = 5
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  Building integrated longitudinal dataset                    ║")
    print("╚══════════════════════════════════════════════════════════════╝", flush=True)
    progress(1, total_steps, "Reading visit spine, Pop, overlap, and PRO sources")
    frames = [
        canonicalize_patient_id(pd.read_parquet(path), source)
        for source, path in zip(["visit_spine", "pop", "overlap", "pros"], [args.spine, args.pop, args.overlap, args.pros])
    ]
    for source, frame in zip(["Visit spine", "Pop", "Overlap", "PROs"], frames):
        print(f"  • {source:<12} {len(frame):>6,} visits | {frame.patient_id.nunique():>5,} patients", flush=True)

    progress(2, total_steps, "Merging sources and deriving longitudinal variables")
    integrated, discrepancies = build_integrated(*frames)
    assert integrated.visit_id.is_unique and not integrated.duplicated(KEYS).any()
    assert integrated.time_since_observed_baseline_days.ge(0).all()
    assert integrated.loc[integrated.visit_number.eq(0)].groupby("patient_id").size().eq(1).all()
    assert integrated.time_to_next_visit_days.dropna().ge(0).all()
    progress(3, total_steps, "Writing integrated dataset and coverage table")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_csv = args.output.with_suffix(".csv")
    coverage_path = common.BLOCKA_TABLES_DIR / "10_integrated_longitudinal_coverage.csv"
    integrated.to_parquet(args.output, index=False)
    integrated.to_csv(output_csv, index=False)
    coverage(integrated).to_csv(coverage_path, index=False)
    for path in [args.output, output_csv, coverage_path]:
        report_output(path)

    progress(4, total_steps, "Creating comparable Pop-grouped patient timelines")
    for path in generate_longitudinal_figures(integrated):
        report_output(path)

    progress(5, total_steps, "Writing merge and quality-control reports")
    spine_ids = set(frames[0].visit_id)
    summary = []
    unmatched = []
    for name, frame in zip(["visit_spine", "pop", "overlap", "pros"], frames):
        matched = frame.visit_id.isin(spine_ids); summary.append({"source": name, "n_source_rows": len(frame), "n_unique_patients": frame.patient_id.nunique(), "n_matched_visit_ids": int(matched.sum()), "n_unmatched_visit_ids": int((~matched).sum()), "pct_matched_visit_ids": 100 * matched.mean() if len(frame) else np.nan})
        if (~matched).any(): unmatched.append(frame.loc[~matched, ["patient_id", "visit_id", "visit_date"]].assign(source=name))
    merge_summary_path = common.BLOCKA_QC_DIR / "10_integrated_merge_summary.csv"
    pd.DataFrame(summary).to_csv(merge_summary_path, index=False)
    report_output(merge_summary_path)
    if unmatched:
        unmatched_path = common.BLOCKA_QC_DIR / "10_integrated_unmatched_visits.csv"
        pd.concat(unmatched).loc[:, ["source", "patient_id", "visit_id", "visit_date"]].to_csv(unmatched_path, index=False)
        report_output(unmatched_path)
    qc = {"n_rows_integrated": len(integrated), "n_unique_patients": int(integrated.patient_id.nunique()), "n_unique_visit_ids": int(integrated.visit_id.nunique()), "n_baseline_rows": int(integrated.is_observed_baseline.sum()), "n_rows_with_pop": int(integrated.has_pop_data.sum()), "n_rows_with_overlap": int(integrated.has_overlap_data.sum()), "n_rows_with_pro": int(integrated.has_pro_data.sum()), "n_rows_with_all_three_blocks": int(integrated.n_data_blocks_available.eq(3).sum()), "n_pop_transitions_evaluable": int(integrated.pop_transition_evaluable.sum()), "n_overlap_transitions_evaluable": int(integrated.overlap_transition_evaluable.sum()), "n_incident_extraglandular_events": int(integrated.incident_extraglandular_from_previous.sum()), "n_transition_to_pop1_events": int(integrated.transition_to_pop1_next_visit.sum()), "n_negative_visit_intervals": int(integrated.time_to_next_visit_days.dropna().lt(0).sum()), "n_duplicate_visit_ids": int(integrated.visit_id.duplicated().sum()), "shared_column_discrepancies": discrepancies}
    qc_path = common.BLOCKA_QC_DIR / "10_integrated_longitudinal_qc.json"
    qc_path.write_text(json.dumps(qc, indent=2))
    report_output(qc_path)
    print(f"\n✓ Complete: {len(integrated):,} visits across {integrated.patient_id.nunique():,} patients.\n", flush=True)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Profile observable clinical history for the biomarker transitions in step 03.

This is a feasibility analysis only.  It enriches the frozen clinical intervals
with observable prior-state information and reports the support remaining after
requiring that information; it does not fit a statistical model.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
from src.studies._shared import create_study_dirs, load_parquet  # noqa: E402

STUDY_NAME = "pharma/03b_pharma_history_feasibility"
STEP_03_TABLES = common.STUDIES_TABLES_DIR / "pharma" / "03_pharma_multistate_biomarkers"
DEFAULT_ANALYTIC = (common.STUDIES_ANALYTIC_DIR / "pharma" / "01_run_pharma_main" /
                    "01_pharma_transition_intervals.parquet")
DEFAULT_SUPPORT = STEP_03_TABLES / "03_pharma_multistate_support.csv"
DEFAULT_MODELS = STEP_03_TABLES / "03_pharma_multistate_models.csv"
AGE_COLUMN = "from_ids__age_at_visit"
YEAR_DAYS = 365.25
OBSERVED_POPS = ("Pop1", "Pop2", "Pop3")

SUPPORT_COLUMNS = [
    "biomarker", "transition_id", "from_pop", "to_pop", "n_intervals_03",
    "n_patients_03", "n_events_03", "n_intervals_with_previous_pop",
    "n_patients_with_previous_pop", "n_events_with_previous_pop",
    "pct_intervals_with_previous_pop", "pct_events_with_previous_pop",
    "n_intervals_with_state_entry_observed", "n_patients_with_state_entry_observed",
    "n_events_with_state_entry_observed", "pct_intervals_with_state_entry_observed",
    "pct_events_with_state_entry_observed", "median_observed_time_in_current_state",
    "iqr_observed_time_in_current_state", "min_observed_time_in_current_state",
    "max_observed_time_in_current_state", "previous_pop_feasibility",
    "state_time_feasibility",
]
CELL_COLUMNS = [
    "biomarker", "transition_id", "from_pop", "to_pop", "previous_pop",
    "n_intervals", "n_patients", "n_events", "cell_ge_minimum",
]
SUMMARY_COLUMNS = [
    "biomarker", "n_supported_transitions_from_03",
    "n_transitions_previous_pop_feasible", "n_transitions_previous_pop_borderline",
    "n_transitions_previous_pop_not_feasible", "n_transitions_state_time_feasible",
    "n_transitions_state_time_borderline", "n_transitions_state_time_not_feasible",
    "overall_history_feasibility",
]


def load_config(path: Path) -> tuple[int, int]:
    import yaml
    config = yaml.safe_load(path.read_text())
    try:
        minimum = config["minimum_counts"]
        return (int(minimum["events_for_multivariable_model"]),
                int(minimum["cell_for_modeling"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Configuration must define both history feasibility thresholds") from exc


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    ).fillna(False).astype(bool)


def select_step_03_combinations(models: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    """Return distinct modeled combinations that step 03 marked as supported."""
    keys = ["biomarker", "transition_id", "from_pop", "to_pop"]
    for name, frame, required in [
        ("models", models, set(keys)),
        ("support", support, set(keys + ["supported_for_model"])),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Step-03 {name} table missing required columns: {missing}")
    modeled = models[keys].dropna(subset=keys).drop_duplicates()
    supported = support.loc[_as_bool(support.supported_for_model), keys].drop_duplicates()
    return modeled.merge(supported, on=keys, how="inner").sort_values(keys).reset_index(drop=True)


def enrich_history(intervals: pd.DataFrame) -> pd.DataFrame:
    """Derive prior observed state and time since an observed entry into FROM."""
    required = {"patient_id", "from_pop", "to_pop", "from_clinical_anchor_date",
                "to_clinical_anchor_date"}
    missing = sorted(required - set(intervals.columns))
    if missing:
        raise ValueError(f"Analytic intervals missing required columns: {missing}")
    data = intervals.copy()
    data["from_clinical_anchor_date"] = pd.to_datetime(
        data.from_clinical_anchor_date, errors="coerce")
    data["to_clinical_anchor_date"] = pd.to_datetime(
        data.to_clinical_anchor_date, errors="coerce")
    if data[["from_clinical_anchor_date", "to_clinical_anchor_date"]].isna().any().any():
        raise ValueError("Clinical anchor dates must be available to derive observable history")
    data["_source_order"] = np.arange(len(data))
    data = data.sort_values(
        ["patient_id", "from_clinical_anchor_date", "to_clinical_anchor_date", "_source_order"],
        kind="stable").reset_index(drop=True)
    data["previous_pop"] = data.groupby("patient_id", sort=False).from_pop.shift(1)
    data["previous_pop_observed"] = data.previous_pop.notna()
    data["state_entry_observed"] = False
    data["observed_time_in_current_state"] = np.nan

    for _, positions in data.groupby("patient_id", sort=False).indices.items():
        entry_date = None
        previous = None
        for position in positions:
            current = data.loc[position]
            if previous is not None:
                continuous = previous.to_pop == current.from_pop
                if not continuous:
                    entry_date = None
                elif previous.from_pop != previous.to_pop:
                    entry_date = previous.to_clinical_anchor_date
            if entry_date is not None:
                elapsed = (current.from_clinical_anchor_date - entry_date).days / YEAR_DAYS
                if elapsed >= 0:
                    data.at[position, "state_entry_observed"] = True
                    data.at[position, "observed_time_in_current_state"] = elapsed
            previous = current
    return data.drop(columns="_source_order")


def _pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else np.nan


def profile_combination(data: pd.DataFrame, item: pd.Series, minimum_events: int,
                        minimum_cell: int) -> tuple[dict, list[dict]]:
    """Profile one step-03 risk set using its original covariate availability."""
    biomarker = str(item.biomarker)
    lab_column = f"from_lab__{biomarker}"
    if lab_column not in data:
        raise ValueError(f"Analytic intervals missing step-03 biomarker column: {lab_column}")
    lab = pd.to_numeric(data[lab_column], errors="coerce")
    # Match step 03's analytic sample: both states must be classifiable, the
    # FROM biomarker must be finite, and age is required when that column was
    # available to the age-adjusted step-03 analysis.
    mask = (data.from_pop.eq(item.from_pop) & data.to_pop.isin(OBSERVED_POPS) &
            np.isfinite(lab))
    if AGE_COLUMN in data:
        mask &= np.isfinite(pd.to_numeric(data[AGE_COLUMN], errors="coerce"))
    risk = data.loc[mask].copy()
    risk["event"] = risk.to_pop.eq(item.to_pop)
    previous = risk.loc[risk.previous_pop_observed]
    state_time = risk.loc[risk.state_entry_observed &
                          risk.observed_time_in_current_state.notna()]
    events = int(risk.event.sum())
    previous_events = int(previous.event.sum())
    state_events = int(state_time.event.sum())

    cell_rows = []
    for value, cell in previous.groupby("previous_pop", sort=True, dropna=False):
        cell_events = int(cell.event.sum())
        cell_rows.append({
            **{key: item[key] for key in ["biomarker", "transition_id", "from_pop", "to_pop"]},
            "previous_pop": value, "n_intervals": len(cell),
            "n_patients": int(cell.patient_id.nunique()), "n_events": cell_events,
            "cell_ge_minimum": cell_events >= minimum_cell,
        })
    small_cell = any(not row["cell_ge_minimum"] for row in cell_rows)
    if previous_events < minimum_events:
        previous_feasibility = "not_feasible"
    elif small_cell:
        previous_feasibility = "borderline"
    else:
        previous_feasibility = "feasible"
    state_feasibility = "feasible" if state_events >= minimum_events else "not_feasible"
    times = pd.to_numeric(state_time.observed_time_in_current_state, errors="coerce")
    row = {
        **{key: item[key] for key in ["biomarker", "transition_id", "from_pop", "to_pop"]},
        "n_intervals_03": len(risk), "n_patients_03": int(risk.patient_id.nunique()),
        "n_events_03": events, "n_intervals_with_previous_pop": len(previous),
        "n_patients_with_previous_pop": int(previous.patient_id.nunique()),
        "n_events_with_previous_pop": previous_events,
        "pct_intervals_with_previous_pop": _pct(len(previous), len(risk)),
        "pct_events_with_previous_pop": _pct(previous_events, events),
        "n_intervals_with_state_entry_observed": len(state_time),
        "n_patients_with_state_entry_observed": int(state_time.patient_id.nunique()),
        "n_events_with_state_entry_observed": state_events,
        "pct_intervals_with_state_entry_observed": _pct(len(state_time), len(risk)),
        "pct_events_with_state_entry_observed": _pct(state_events, events),
        "median_observed_time_in_current_state": float(times.median()) if len(times) else np.nan,
        "iqr_observed_time_in_current_state": (float(times.quantile(.75) - times.quantile(.25))
                                                if len(times) else np.nan),
        "min_observed_time_in_current_state": float(times.min()) if len(times) else np.nan,
        "max_observed_time_in_current_state": float(times.max()) if len(times) else np.nan,
        "previous_pop_feasibility": previous_feasibility,
        "state_time_feasibility": state_feasibility,
    }
    return row, cell_rows


def summarize(support: pd.DataFrame, biomarkers: list[str]) -> pd.DataFrame:
    rows = []
    for biomarker in biomarkers:
        group = support.loc[support.biomarker.eq(biomarker)]
        previous = group.previous_pop_feasibility.value_counts()
        state = group.state_time_feasibility.value_counts()
        any_feasible = previous.get("feasible", 0) + state.get("feasible", 0) > 0
        any_borderline = previous.get("borderline", 0) + state.get("borderline", 0) > 0
        overall = "GO" if any_feasible else ("LIMITED" if any_borderline else "STOP")
        rows.append({
            "biomarker": biomarker, "n_supported_transitions_from_03": len(group),
            "n_transitions_previous_pop_feasible": int(previous.get("feasible", 0)),
            "n_transitions_previous_pop_borderline": int(previous.get("borderline", 0)),
            "n_transitions_previous_pop_not_feasible": int(previous.get("not_feasible", 0)),
            "n_transitions_state_time_feasible": int(state.get("feasible", 0)),
            "n_transitions_state_time_borderline": int(state.get("borderline", 0)),
            "n_transitions_state_time_not_feasible": int(state.get("not_feasible", 0)),
            "overall_history_feasibility": overall,
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def run(args: argparse.Namespace) -> None:
    minimum_events, minimum_cell = load_config(args.config)
    dirs = create_study_dirs(STUDY_NAME)
    logging.basicConfig(filename=dirs["logs"] / "03b_pharma_history_feasibility.log",
                        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    models = pd.read_csv(args.models)
    step_03_support = pd.read_csv(args.support)
    combinations = select_step_03_combinations(models, step_03_support)
    biomarkers = combinations.biomarker.astype(str).drop_duplicates().tolist()
    intervals = load_parquet(args.analytic)
    enriched = enrich_history(intervals)
    logging.info("Biomarkers evaluated: %s", ", ".join(biomarkers))
    logging.info("Transitions evaluated: %d", len(combinations))
    logging.info("Clinical intervals loaded: %d", len(enriched))
    logging.info("Patients loaded: %d", enriched.patient_id.nunique())
    logging.info("Intervals with previous_pop: %d", enriched.previous_pop_observed.sum())
    logging.info("Intervals with observed state entry: %d", enriched.state_entry_observed.sum())

    support_rows, cell_rows = [], []
    for _, item in combinations.iterrows():
        support_row, cells = profile_combination(
            enriched, item, minimum_events, minimum_cell)
        support_rows.append(support_row)
        cell_rows.extend(cells)
    history_support = pd.DataFrame(support_rows, columns=SUPPORT_COLUMNS)
    previous_cells = pd.DataFrame(cell_rows, columns=CELL_COLUMNS)
    history_summary = summarize(history_support, biomarkers)

    history_support.to_csv(dirs["tables"] / "03b_pharma_history_support.csv", index=False)
    previous_cells.to_csv(dirs["tables"] / "03b_pharma_previous_pop_cells.csv", index=False)
    history_summary.to_csv(dirs["tables"] / "03b_pharma_history_summary.csv", index=False)
    trace_columns = [
        column for column in ["patient_id", "from_clinical_episode_id",
                              "to_clinical_episode_id", "from_clinical_anchor_date",
                              "to_clinical_anchor_date", "from_pop", "to_pop", "interval_years",
                              AGE_COLUMN, *[f"from_lab__{value}" for value in biomarkers],
                              "previous_pop", "previous_pop_observed", "state_entry_observed",
                              "observed_time_in_current_state"] if column in enriched
    ]
    enriched[trace_columns].to_parquet(
        dirs["analytic"] / "03b_pharma_history_enriched.parquet", index=False)

    previous_events = int(history_support.n_events_with_previous_pop.sum())
    state_events = int(history_support.n_events_with_state_entry_observed.sum())
    logging.info("Events retained after history requirements: previous_pop=%d; state_entry=%d",
                 previous_events, state_events)
    for _, item in history_summary.iterrows():
        logging.info("%s: %s", item.biomarker, item.overall_history_feasibility)

    print("ORDER TO REVIEW PHARMA HISTORY FEASIBILITY OUTPUTS\n\n"
          "1. 03b_pharma_history_support.csv\n"
          "2. 03b_pharma_previous_pop_cells.csv\n"
          "3. 03b_pharma_history_summary.csv\n"
          "4. 03b_pharma_history_enriched.parquet\n")
    print(f"Biomarkers evaluated: {len(biomarkers)}")
    print(f"Transitions evaluated: {len(combinations)}")
    print("Previous-pop feasible transitions: "
          f"{int(history_support.previous_pop_feasibility.eq('feasible').sum())}")
    print("State-time feasible transitions: "
          f"{int(history_support.state_time_feasibility.eq('feasible').sum())}")
    for status in ["GO", "LIMITED", "STOP"]:
        print(f"Biomarkers {status}: "
              f"{int(history_summary.overall_history_feasibility.eq(status).sum())}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic", type=Path, default=DEFAULT_ANALYTIC)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

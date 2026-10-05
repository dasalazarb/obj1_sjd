#!/usr/bin/env python3
"""History-aware sensitivity models for Step-03 biomarker associations.

Only combinations passing the Step-03b feasibility gate are fitted.  These are
observational, adjacent-episode intensity associations, not causal effects.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import common
from src.studies._shared import create_study_dirs, load_parquet, validate_predictors

STUDY_NAME = "pharma/04_pharma_history_aware"
BASE = common.STUDIES_TABLES_DIR / "pharma"
DEFAULT_ANALYTIC = (common.STUDIES_ANALYTIC_DIR / "pharma" /
                    "03b_pharma_history_feasibility" /
                    "03b_pharma_history_enriched.parquet")
DEFAULT_SUPPORT = BASE / "03b_pharma_history_feasibility" / "03b_pharma_history_support.csv"
DEFAULT_CELLS = BASE / "03b_pharma_history_feasibility" / "03b_pharma_previous_pop_cells.csv"
DEFAULT_STEP03 = BASE / "03_pharma_multistate_biomarkers" / "03_pharma_multistate_models.csv"
AGE_COLUMNS = ("from_demo__age_at_visit", "from_ids__age_at_visit", "from_age_at_visit", "from_age")


def eligible_history_models(support: pd.DataFrame) -> pd.DataFrame:
    """Return combination-level GO rows; LIMITED and STOP are never modeled."""
    required = {"biomarker", "from_pop", "to_pop", "previous_pop_feasibility",
                "state_time_feasibility"}
    missing = required - set(support)
    if missing:
        raise ValueError(f"History support missing required columns: {sorted(missing)}")
    return support.loc[(support.previous_pop_feasibility == "feasible") |
                       (support.state_time_feasibility == "feasible")].copy()


def _fit(data: pd.DataFrame, terms: list[str]):
    import statsmodels.api as sm
    validate_predictors(["biomarker_z", "age", *terms])
    design = pd.DataFrame({"biomarker_z": data.biomarker_z, "age": data.age})
    if "observed_time_in_current_state" in terms:
        design["observed_time_in_current_state"] = data.observed_time_in_current_state
    if "previous_pop" in terms:
        dummies = pd.get_dummies(data.previous_pop, prefix="previous_pop", drop_first=True,
                                 dtype=float)
        design = pd.concat([design, dummies], axis=1)
    design = sm.add_constant(design.astype(float), has_constant="add")
    return sm.GLM(data.event.astype(float), design,
                  family=sm.families.Poisson(),
                  offset=np.log(data.interval_years.astype(float))).fit(
                      cov_type="cluster",
                      cov_kwds={"groups": data.patient_id.to_numpy()})


def fit_combination(item: pd.Series, analytic: pd.DataFrame, cells: pd.DataFrame,
                    minimum_events: int, minimum_cell: int) -> list[dict]:
    biomarker = str(item.biomarker)
    exposure = f"from_lab__{biomarker}"
    validate_predictors([exposure])
    age_column = next((c for c in AGE_COLUMNS if c in analytic), None)
    base = {"biomarker": biomarker, "from_pop": item.from_pop, "to_pop": item.to_pop,
            "transition_id": item.get("transition_id", f"{item.from_pop}→{item.to_pop}")}
    variants = [
        ("previous_state", ["previous_pop"], item.previous_pop_feasibility == "feasible"),
        ("state_time", ["observed_time_in_current_state"], item.state_time_feasibility == "feasible"),
        ("full_history", ["previous_pop", "observed_time_in_current_state"],
         item.previous_pop_feasibility == "feasible" and item.state_time_feasibility == "feasible"),
    ]
    rows = []
    for name, terms, allowed in variants:
        row = {**base, "history_model": name, "estimate": np.nan, "ci95_low": np.nan,
               "ci95_high": np.nan, "p_value": np.nan, "se_type": "cluster_patient",
               "cluster_variable": "patient_id", "model_status": "not_estimable"}
        if not allowed or exposure not in analytic or age_column is None:
            row["interpretability_reason"] = "feasibility_gate_not_go" if not allowed else "required_column_missing"
            rows.append(row); continue
        data = analytic.loc[analytic.from_pop.eq(item.from_pop)].copy()
        data["event"] = data.to_pop.eq(item.to_pop).astype(float)
        data["x"] = pd.to_numeric(data[exposure], errors="coerce")
        data["age"] = pd.to_numeric(data[age_column], errors="coerce")
        required = ["patient_id", "event", "interval_years", "x", "age", *terms]
        data = data.dropna(subset=required)
        data = data.loc[pd.to_numeric(data.interval_years, errors="coerce").gt(0)]
        sd = data.x.std(ddof=1)
        data["biomarker_z"] = (data.x - data.x.mean()) / sd if pd.notna(sd) and sd > 0 else np.nan
        events, nonevents = int(data.event.sum()), int((1 - data.event).sum())
        row.update(n_intervals=len(data), n_patients=int(data.patient_id.nunique()),
                   n_clusters=int(data.patient_id.nunique()), events=events, nonevents=nonevents)
        sparse = False
        if "previous_pop" in terms:
            relevant = cells.loc[(cells.biomarker.astype(str) == biomarker) &
                                 (cells.from_pop == item.from_pop) & (cells.to_pop == item.to_pop)]
            sparse = relevant.empty or relevant.n_events.min() < minimum_cell
        if events < minimum_events or nonevents < minimum_events or sparse or data.biomarker_z.isna().all():
            row["interpretability_reason"] = ("insufficient_previous_pop_cells" if sparse
                                               else "insufficient_events_or_variability")
            rows.append(row); continue
        try:
            fit = _fit(data, terms)
            beta, se = fit.params["biomarker_z"], fit.bse["biomarker_z"]
            row.update(estimate=math.exp(beta), ci95_low=math.exp(beta - 1.96 * se),
                       ci95_high=math.exp(beta + 1.96 * se),
                       p_value=fit.pvalues["biomarker_z"], model_status="success",
                       interpretability_reason="")
        except Exception as exc:
            row["interpretability_reason"] = f"model_failed:{type(exc).__name__}:{str(exc)[:120]}"
        rows.append(row)
    return rows


def compare_with_step03(models: pd.DataFrame, step03: pd.DataFrame) -> pd.DataFrame:
    primary = step03.loc[step03.model_version.eq("age_adjusted_clustered")].copy()
    if primary.empty:
        primary = step03.loc[step03.model_version.eq("same_sample_clustered")].copy()
    old = primary[["biomarker", "from_pop", "to_pop", "estimate", "p_value"]].rename(
        columns={"estimate": "IR_step03", "p_value": "p_value_step03"})
    out = models.merge(old, on=["biomarker", "from_pop", "to_pop"], how="left")
    out = out.rename(columns={"estimate": "IR_history", "p_value": "p_value_history"})
    epsilon = np.finfo(float).eps
    out["percent_change_in_log_IR"] = (np.log(out.IR_history).sub(np.log(out.IR_step03)).abs() /
                                       np.log(out.IR_step03).abs().clip(lower=epsilon))
    out["direction_consistent"] = ((out.IR_history - 1) * (out.IR_step03 - 1)).ge(0)
    out["significance_consistent"] = out.p_value_history.le(.05).eq(out.p_value_step03.le(.05))
    out["history_model_status"] = out.model_status
    return out


def run(args: argparse.Namespace) -> None:
    import yaml
    config = yaml.safe_load(args.config.read_text())
    minimum_events = int(config["minimum_counts"]["events_for_multivariable_model"])
    minimum_cell = int(config["minimum_counts"]["cell_for_modeling"])
    analytic, support = load_parquet(args.analytic), pd.read_csv(args.support)
    cells, step03 = pd.read_csv(args.cells), pd.read_csv(args.step03)
    eligible = eligible_history_models(support)
    rows = []
    for _, item in eligible.iterrows():
        rows.extend(fit_combination(item, analytic, cells, minimum_events, minimum_cell))
    models = pd.DataFrame(rows)
    comparison = compare_with_step03(models, step03) if not models.empty else models.copy()
    summary = pd.DataFrame([{"n_combinations_go": len(eligible),
                             "n_models_attempted": len(models),
                             "n_models_successful": int(models.model_status.eq("success").sum()) if len(models) else 0}])
    dirs = create_study_dirs(STUDY_NAME)
    logging.basicConfig(filename=dirs["logs"] / "04_pharma_history_aware.log", level=logging.INFO, force=True)
    models.to_csv(dirs["tables"] / "04_pharma_history_models.csv", index=False)
    comparison.to_csv(dirs["tables"] / "04_pharma_history_comparison.csv", index=False)
    summary.to_csv(dirs["tables"] / "04_pharma_history_summary.csv", index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic", type=Path, default=DEFAULT_ANALYTIC)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--step03", type=Path, default=DEFAULT_STEP03)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

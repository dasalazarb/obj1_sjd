#!/usr/bin/env python3
"""Age-adjusted sensitivity analysis for signals selected in Pharma MAIN.

This program deliberately starts from the frozen outputs of
``01_run_pharma_main.py``.  It does not rebuild episodes or transitions and it
does not use the sensitivity models to discover additional signals.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

import common
from src.studies._shared import create_study_dirs, load_parquet

MAIN_TABLES = common.STUDIES_TABLES_DIR / "pharma" / "01_run_pharma_main"
MAIN_ANALYTIC = (common.STUDIES_ANALYTIC_DIR / "pharma" / "01_run_pharma_main" /
                 "01_pharma_transition_intervals.parquet")
AGE_COLUMN = "from_ids__age_at_visit"
PROTOCOL_NAMES = {"protocol", "ids__protocol", "ids__protocol_number", "parent_protocol"}
CANDIDATE_COLUMNS = [
    "from_pop", "to_pop", "transition_pair", "exposure", "exposure_family",
    "main_estimate", "main_ci95_low", "main_ci95_high", "main_p_value",
    "selection_q_value", "main_model_status", "main_interpretability_status",
    "selection_tier", "selected", "selection_reason",
]
MODEL_COLUMNS = [
    "from_pop", "to_pop", "transition_pair", "exposure", "exposure_family",
    "selection_tier", "selection_q_value", "main_n_intervals", "main_n_patients",
    "main_events", "main_estimate", "main_ci95_low", "main_ci95_high", "main_p_value",
    "same_sample_n_intervals", "same_sample_n_patients", "same_sample_events",
    "same_sample_nonevents", "same_sample_estimate", "same_sample_ci95_low",
    "same_sample_ci95_high", "same_sample_p_value", "adjusted_estimate",
    "adjusted_ci95_low", "adjusted_ci95_high", "adjusted_p_value", "age_estimate",
    "age_ci95_low", "age_ci95_high", "age_p_value", "age_ir_per_10_years",
    "model_status", "interpretability_status", "interpretability_reason",
]
SUMMARY_EXTRA = [
    "direction_main", "direction_same_sample", "direction_adjusted",
    "direction_consistent", "percent_change_main_to_same_sample",
    "percent_change_same_sample_to_adjusted", "precision_main", "precision_adjusted",
    "robustness_status",
]


def load_config(path: Path) -> dict:
    import yaml
    result = yaml.safe_load(path.read_text())
    if not isinstance(result, dict) or "minimum_counts" not in result:
        raise ValueError("Pharma configuration must define minimum_counts")
    return result


def _bh_adjust(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, retaining the input index."""
    answer = pd.Series(np.nan, index=values.index, dtype=float)
    valid = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if valid.empty:
        return answer
    adjusted = valid * len(valid) / np.arange(1, len(valid) + 1)
    answer.loc[valid.index] = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    return answer


def _harmonize(frame: pd.DataFrame, laboratory: bool) -> pd.DataFrame:
    exposure_column = "lab" if laboratory else "exposure"
    required = {"from_pop", "to_pop", exposure_column, "estimate", "ci95_low",
                "ci95_high", "p_value", "model_status", "interpretability_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MAIN association table missing columns: {sorted(missing)}")
    family = pd.Series("laboratory" if laboratory else "clinical", index=frame.index)
    if not laboratory and "family" in frame:
        family = frame["family"].fillna("clinical").astype(str)
    elif not laboratory:
        # Current MAIN files predate an explicit family column.  These labels
        # reproduce MAIN's feature metadata; they have no role in selection.
        names = frame[exposure_column].astype(str).str.lower()
        family.loc[names.isin({"ssa", "ssb", "ana", "rf", "cryoglobulinemia"})] = "serology"
        family.loc[names.isin({"esspri_total", "dryness", "fatigue", "pain"})] = "esspri"
        family.loc[names.str.startswith("essdai_domain_")] = "essdai_domain"
    out = pd.DataFrame({
        "from_pop": frame.from_pop, "to_pop": frame.to_pop,
        "transition_pair": frame.from_pop.astype(str) + " -> " + frame.to_pop.astype(str),
        "exposure": frame[exposure_column].astype(str), "exposure_family": family,
        "main_n_intervals": frame.get("n_intervals", np.nan),
        "main_n_patients": frame.get("n_patients", np.nan),
        "main_events": frame.get("events", np.nan), "main_estimate": frame.estimate,
        "main_ci95_low": frame.ci95_low, "main_ci95_high": frame.ci95_high,
        "main_p_value": frame.p_value, "main_model_status": frame.model_status,
        "main_interpretability_status": frame.interpretability_status,
    })
    return out


def build_candidates(clinical: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    """Combine MAIN results, perform within-transition BH, and select signals."""
    result = pd.concat([_harmonize(clinical, False), _harmonize(labs, True)],
                       ignore_index=True)
    result["selection_q_value"] = np.nan
    result["selection_tier"] = pd.NA
    result["selected"] = False
    valid = (result.main_model_status.eq("success") &
             result.main_interpretability_status.eq("interpretable"))
    age = result.exposure.str.lower().isin({"age", "ids__age_at_visit"})
    protocol = result.exposure.str.lower().isin(PROTOCOL_NAMES)
    directional = result.from_pop.ne(result.to_pop)
    eligible = valid & ~age & ~protocol & directional
    result["selection_reason"] = "q_above_0.20"
    result.loc[~valid, "selection_reason"] = "not_interpretable"
    result.loc[valid & ~directional, "selection_reason"] = "stable_transition_excluded"
    result.loc[valid & age, "selection_reason"] = "age_is_adjustment_covariate"
    result.loc[valid & protocol, "selection_reason"] = "protocol_excluded"
    result.loc[eligible, "selection_q_value"] = result.loc[eligible].groupby(
        ["from_pop", "to_pop"])["main_p_value"].transform(_bh_adjust)

    tier1 = eligible & result.selection_q_value.le(.05)
    result.loc[tier1, ["selection_tier", "selected", "selection_reason"]] = [
        "fdr_significant", True, "selection_q_value<=0.05"]
    borderline = result.loc[eligible & result.selection_q_value.gt(.05) &
                            result.selection_q_value.le(.20)].sort_values(
        ["selection_q_value", "main_p_value", "transition_pair", "exposure"],
        kind="stable")
    chosen = borderline.head(2).index
    result.loc[borderline.index, "selection_reason"] = "borderline_not_top2"
    result.loc[chosen, ["selection_tier", "selected", "selection_reason"]] = [
        "borderline_fdr", True, "lowest_q_between_0.05_and_0.20"]
    return result


def _numeric_exposure(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        return numeric
    text = series.astype("string").str.strip().str.lower()
    binary = text.map({"negative": 0, "neg": 0, "no": 0, "false": 0,
                       "inactive": 0, "positive": 1, "pos": 1, "yes": 1,
                       "true": 1, "active": 1})
    return numeric.fillna(binary)


def _resolve_exposure(frame: pd.DataFrame, exposure: str, family: str) -> str | None:
    candidates = ([f"from_lab__{exposure}"] if family == "laboratory" else []) + [
        f"from_{exposure}"]
    for column in candidates:
        if column in frame:
            return column
    # MAIN feature labels can be shorter than their source columns.  Match only
    # an unambiguous FROM column; never infer a new variable or cutoff.
    token = exposure.lower()
    matches = [c for c in frame if c.startswith("from_") and token in c.lower() and
               not c.startswith("from_clinical_")]
    return matches[0] if len(matches) == 1 else None


def _fit_poisson(data: pd.DataFrame, adjusted: bool) -> tuple[dict, bool]:
    import statsmodels.api as sm
    columns = ["x", "age"] if adjusted else ["x"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sm.GLM(data.event, sm.add_constant(data[columns], has_constant="add"),
                     family=sm.families.Poisson(), offset=np.log(data.time)).fit(cov_type="HC0")
    perfect = any(issubclass(item.category, PerfectSeparationWarning) for item in caught)
    values = {}
    for name in columns:
        beta, se = fit.params[name], fit.bse[name]
        values[name] = (math.exp(beta), math.exp(beta - 1.96 * se),
                        math.exp(beta + 1.96 * se), fit.pvalues[name], beta)
    return values, perfect


def fit_selected(row: pd.Series, analytic: pd.DataFrame, config: dict) -> dict:
    output = {name: np.nan for name in MODEL_COLUMNS}
    for name in ["from_pop", "to_pop", "transition_pair", "exposure", "exposure_family",
                 "selection_tier", "selection_q_value", "main_n_intervals",
                 "main_n_patients", "main_events", "main_estimate", "main_ci95_low",
                 "main_ci95_high", "main_p_value"]:
        output[name] = row.get(name, np.nan)
    output.update(model_status="not_estimable", interpretability_status="not_estimable")
    exposure_column = _resolve_exposure(analytic, row.exposure, row.exposure_family)
    if exposure_column is None:
        output["interpretability_reason"] = "exposure_column_not_found_or_ambiguous"
        return output
    if AGE_COLUMN not in analytic:
        output["interpretability_reason"] = f"missing_required_age_column:{AGE_COLUMN}"
        return output
    origin = analytic.loc[analytic.from_pop.eq(row.from_pop)]
    data = pd.DataFrame({"patient_id": origin.patient_id,
                         "event": origin.to_pop.eq(row.to_pop).astype(float),
                         "time": pd.to_numeric(origin.interval_years, errors="coerce"),
                         "x": _numeric_exposure(origin[exposure_column]),
                         "age": pd.to_numeric(origin[AGE_COLUMN], errors="coerce")})
    data = data.dropna()
    data = data.loc[np.isfinite(data[["time", "x", "age"]]).all(axis=1) & data.time.gt(0)]
    events, nonevents = int(data.event.sum()), int((1 - data.event).sum())
    output.update(same_sample_n_intervals=len(data),
                  same_sample_n_patients=int(data.patient_id.nunique()),
                  same_sample_events=events, same_sample_nonevents=nonevents)
    minimum = int(config["minimum_counts"]["events_for_multivariable_model"])
    reason = None
    if events < minimum: reason = "insufficient_events"
    elif nonevents < minimum: reason = "insufficient_nonevents"
    elif data.x.nunique() <= 1: reason = "insufficient_exposure_variability"
    elif data.age.nunique() <= 1: reason = "insufficient_age_variability"
    if reason:
        output["interpretability_reason"] = reason
        return output
    try:
        unadjusted, perfect_u = _fit_poisson(data, False)
        adjusted, perfect_a = _fit_poisson(data, True)
    except Exception as exc:
        logging.warning("Sensitivity model failed for %s %s: %s", row.transition_pair,
                        row.exposure, exc)
        output["interpretability_reason"] = f"model_failed:{type(exc).__name__}"
        return output
    ux, ax, age = unadjusted["x"], adjusted["x"], adjusted["age"]
    output.update(same_sample_estimate=ux[0], same_sample_ci95_low=ux[1],
                  same_sample_ci95_high=ux[2], same_sample_p_value=ux[3],
                  adjusted_estimate=ax[0], adjusted_ci95_low=ax[1],
                  adjusted_ci95_high=ax[2], adjusted_p_value=ax[3], age_estimate=age[0],
                  age_ci95_low=age[1], age_ci95_high=age[2], age_p_value=age[3],
                  age_ir_per_10_years=math.exp(age[4] * 10), model_status="success")
    if perfect_u or perfect_a:
        status, reason = "unstable_perfect_separation", "perfect_separation_detected"
    elif set(data.x.unique()).issubset({0., 1.}) and min(
            int(((data.x == x) & (data.event == y)).sum()) for x in (0, 1) for y in (0, 1)
    ) < int(config["minimum_counts"]["cell_for_modeling"]):
        status, reason = "unstable_sparse_cells", "one_or_more_2x2_cells_below_threshold"
    elif not all(np.isfinite(v) and v > 0 for v in (ax[0], ax[1], ax[2])):
        status, reason = "unstable_extreme_estimate", "ci_or_estimate_non_finite_or_nonpositive"
    elif ax[0] < .1 or ax[0] > 10 or ax[1] < .1 or ax[2] > 10:
        status, reason = "unstable_extreme_estimate", "estimate_or_ci_outside_0.1_to_10"
    else:
        status, reason = "interpretable", ""
    output.update(interpretability_status=status, interpretability_reason=reason)
    return output


def _direction(value) -> str | float:
    if pd.isna(value) or value <= 0: return np.nan
    if np.isclose(value, 1): return "null"
    return "increased" if value > 1 else "decreased"


def _percent_change(before, after) -> float:
    if pd.isna(before) or pd.isna(after) or before <= 0 or after <= 0:
        return np.nan
    denominator = abs(math.log(before))
    return np.nan if denominator < 1e-6 else abs(math.log(after) - math.log(before)) / denominator * 100


def summarize(models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, model in models.iterrows():
        item = model.to_dict()
        directions = [_direction(model.get(x)) for x in
                      ("main_estimate", "same_sample_estimate", "adjusted_estimate")]
        item.update(zip(SUMMARY_EXTRA[:3], directions))
        item["direction_consistent"] = (len(set(directions)) == 1
                                         if all(pd.notna(value) for value in directions) else False)
        item["percent_change_main_to_same_sample"] = _percent_change(
            model.main_estimate, model.same_sample_estimate)
        item["percent_change_same_sample_to_adjusted"] = _percent_change(
            model.same_sample_estimate, model.adjusted_estimate)
        item["precision_main"] = (math.log(model.main_ci95_high) - math.log(model.main_ci95_low)
                                  if model.main_ci95_low > 0 and model.main_ci95_high > 0 else np.nan)
        item["precision_adjusted"] = (math.log(model.adjusted_ci95_high) - math.log(model.adjusted_ci95_low)
                                      if model.adjusted_ci95_low > 0 and model.adjusted_ci95_high > 0 else np.nan)
        if model.model_status != "success" or model.interpretability_status != "interpretable":
            robustness = "not_estimable"
        elif directions[1] != directions[2]:
            robustness = "direction_changed"
        elif abs(math.log(model.adjusted_estimate)) <= .5 * abs(math.log(model.same_sample_estimate)):
            robustness = "attenuated"
        elif (pd.notna(item["precision_main"]) and item["precision_main"] > 0 and
              item["precision_adjusted"] >= 1.5 * item["precision_main"]):
            robustness = "precision_reduced"
        else:
            robustness = "robust"
        item["robustness_status"] = robustness
        rows.append(item)
    return pd.DataFrame(rows, columns=MODEL_COLUMNS + SUMMARY_EXTRA)


def forest_plot(models: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    shown = models[(models.model_status == "success") &
                   (models.interpretability_status == "interpretable")].copy()
    if shown.empty:
        fig, ax = plt.subplots(figsize=(8, 3)); ax.axis("off")
        ax.text(.5, .5, "No selected, estimable Pharma sensitivity associations",
                ha="center", va="center")
    else:
        shown = shown.reset_index(drop=True); y = np.arange(len(shown)) * 2
        fig, ax = plt.subplots(figsize=(10, max(4, .65 * len(shown))))
        for delta, prefix, label, marker in [(-.25, "same_sample", "Same-sample unadjusted", "o"),
                                             (.25, "adjusted", "Age-adjusted", "s")]:
            estimate = shown[f"{prefix}_estimate"].astype(float)
            ax.errorbar(estimate, y + delta,
                        xerr=[estimate - shown[f"{prefix}_ci95_low"],
                              shown[f"{prefix}_ci95_high"] - estimate],
                        fmt=marker, capsize=2, label=label)
        labels = shown.from_pop + "→" + shown.to_pop + " : " + shown.exposure
        ax.axvline(1, color="grey", ls="--"); ax.set_xscale("log")
        ax.set_yticks(y, labels); ax.set_xlabel("Intensity ratio (95% CI)"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dirs = create_study_dirs("pharma/02_pharma_sensitivity")
    logging.basicConfig(filename=dirs["logs"] / "02_pharma_sensitivity.log",
                        level=logging.INFO, force=True)
    candidates = build_candidates(pd.read_csv(args.clinical), pd.read_csv(args.labs))
    candidates[CANDIDATE_COLUMNS].to_csv(
        dirs["tables"] / "02_pharma_sensitivity_candidates.csv", index=False)
    analytic = load_parquet(args.analytic)
    selected = candidates.loc[candidates.selected].copy()
    models = pd.DataFrame([fit_selected(row, analytic, config)
                           for _, row in selected.iterrows()], columns=MODEL_COLUMNS)
    models.to_csv(dirs["tables"] / "02_pharma_sensitivity_models.csv", index=False)
    summary = summarize(models)
    summary.to_csv(dirs["tables"] / "02_pharma_sensitivity_summary.csv", index=False)
    forest_plot(models, dirs["figures"] / "02_pharma_sensitivity_forest.pdf")
    valid = (candidates.main_model_status.eq("success") &
             candidates.main_interpretability_status.eq("interpretable"))
    print("ORDER TO REVIEW PHARMA SENSITIVITY OUTPUTS\n\n"
          "1. 02_pharma_sensitivity_candidates.csv\n"
          "2. 02_pharma_sensitivity_models.csv\n"
          "3. 02_pharma_sensitivity_summary.csv\n"
          "4. 02_pharma_sensitivity_forest.pdf\n")
    print(f"Total interpretable candidates: {int(valid.sum())}")
    print(f"Tier 1 selected: {int(candidates.selection_tier.eq('fdr_significant').sum())}")
    print(f"Tier 2 selected: {int(candidates.selection_tier.eq('borderline_fdr').sum())}")
    print(f"Total sensitivity models attempted: {len(models)}")
    successful = int(((models.model_status == "success") &
                      (models.interpretability_status == "interpretable")).sum()) if len(models) else 0
    print(f"Successful adjusted models: {successful}")
    print(f"Non-estimable models: {len(models) - successful}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", type=Path,
                        default=MAIN_TABLES / "01_pharma_transition_associations.csv")
    parser.add_argument("--labs", type=Path,
                        default=MAIN_TABLES / "01_pharma_lab_transition_associations.csv")
    parser.add_argument("--analytic", type=Path, default=MAIN_ANALYTIC)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""Observed-state multistate analysis of Pharma biomarkers selected by step 02.

Clinical states are observed only at irregularly timed visits, so the exact
transition time between two visits is unknown (and an unobserved intermediate
state may have occurred).  This script therefore models *observed*
state-to-state evolution.  A Poisson model with ``log(interval_years)`` offset
is used as a piecewise-exponential approximation to observed transition
intensities.  Biomarkers are time-varying covariates, updated from the FROM
episode for every interval, and standard errors are clustered by patient.

The reported effects are associations, not causal effects.  In particular,
this is not a joint longitudinal--multistate model, a hidden-state model, or a
semi-Markov model; no intermediate states or biomarker values are imputed.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common  # noqa: E402
from src.studies._shared import (  # noqa: E402
    benjamini_hochberg, create_study_dirs, load_parquet,
)

STUDY_NAME = "pharma/03_pharma_multistate_biomarkers"
DEFAULT_ANALYTIC = (common.STUDIES_ANALYTIC_DIR / "pharma" / "01_run_pharma_main" /
                    "01_pharma_transition_intervals.parquet")
DEFAULT_SENSITIVITY = (common.STUDIES_TABLES_DIR / "pharma" / "02_pharma_sensitivity" /
                       "02_pharma_sensitivity_summary.csv")
AGE_COLUMN = "from_ids__age_at_visit"
POPS = ("Pop1", "Pop2", "Pop3")
TRANSITIONS = tuple((origin, destination) for origin in POPS for destination in POPS
                    if origin != destination)

CANDIDATE_COLUMNS = [
    "exposure", "exposure_family", "selection_tier", "selection_q_value",
    "robustness_status", "adjusted_estimate", "direction_consistent",
    "eligible_for_03", "eligibility_reason",
]
SUPPORT_COLUMNS = [
    "biomarker", "transition_id", "from_pop", "to_pop", "n_intervals_at_risk",
    "n_patients_at_risk", "n_events", "n_patients_with_event",
    "n_intervals_with_biomarker", "n_patients_with_biomarker",
    "median_intervals_per_patient", "iqr_intervals_per_patient",
    "n_patients_with_1_measurement", "n_patients_with_ge2_measurements",
    "n_patients_with_ge3_measurements", "supported_for_model", "support_reason",
]
MODEL_COLUMNS = [
    "biomarker", "transition_id", "from_pop", "to_pop", "model_version",
    "n_intervals", "n_patients", "n_events", "biomarker_mean", "biomarker_sd",
    "estimate_per_sd", "ci95_low", "ci95_high", "p_value", "q_value",
    "age_estimate", "age_ci95_low", "age_ci95_high", "age_p_value",
    "selection_tier", "selection_q_value", "robustness_status",
    "adjusted_estimate_02", "direction_consistent_with_02", "model_status",
    "interpretability_status", "interpretability_reason",
]
LONG_COLUMNS = [
    "patient_id", "from_clinical_episode_id", "to_clinical_episode_id",
    "from_clinical_anchor_date", "to_clinical_anchor_date", "from_pop",
    "observed_to_pop", "transition_id", "risk_destination", "event",
    "interval_years", "biomarker", "biomarker_value", "biomarker_z", "age",
]
SUMMARY_COLUMNS = [
    "biomarker", "n_patients", "n_intervals", "median_intervals_per_patient",
    "iqr_intervals_per_patient", "n_patients_with_1_measurement",
    "n_patients_with_ge2_measurements", "n_patients_with_ge3_measurements",
    "n_supported_transitions", "n_estimable_effects",
    "n_directionally_comparable_with_02", "n_directionally_consistent_with_02",
    "overall_status",
]


def load_config(path: Path) -> dict:
    import yaml
    config = yaml.safe_load(path.read_text())
    try:
        int(config["minimum_counts"]["events_for_multivariable_model"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Configuration must define minimum_counts.events_for_multivariable_model") from exc
    return config


def _as_bool(series: pd.Series) -> pd.Series:
    """Read booleans safely from either CSV strings or native boolean values."""
    return series.astype("string").str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    ).fillna(False).astype(bool)


def build_candidate_trace(sensitivity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one trace row per laboratory biomarker and eligible step-02 rows."""
    required = {"exposure", "exposure_family", "model_status",
                "interpretability_status", "direction_consistent"}
    missing = sorted(required - set(sensitivity.columns))
    if missing:
        raise ValueError(f"Sensitivity summary missing required columns: {missing}")
    # Step 03 is restricted to laboratory biomarkers; clinical step-02 signals
    # are not candidate biomarkers and therefore do not enter this trace.
    source = sensitivity.loc[sensitivity.exposure_family.eq("laboratory")].copy()
    source["_direction_consistent"] = _as_bool(source["direction_consistent"])
    source["_eligible"] = (source.exposure_family.eq("laboratory") &
                           source.model_status.eq("success") &
                           source.interpretability_status.eq("interpretable") &
                           source._direction_consistent)
    rows = []
    for exposure, group in source.groupby("exposure", sort=False, dropna=False):
        eligible = group.loc[group._eligible]
        representative = (eligible if not eligible.empty else group).iloc[0]
        item = {column: representative.get(column, np.nan) for column in CANDIDATE_COLUMNS}
        item["direction_consistent"] = bool(representative["_direction_consistent"])
        item["eligible_for_03"] = not eligible.empty
        if not eligible.empty:
            reason = "eligible_selected_laboratory_from_02"
        elif not group.exposure_family.eq("laboratory").any():
            reason = "not_laboratory"
        elif not group.model_status.eq("success").any():
            reason = "model_not_successful_in_02"
        elif not group.interpretability_status.eq("interpretable").any():
            reason = "not_interpretable_in_02"
        else:
            reason = "direction_not_consistent_in_02"
        item["eligibility_reason"] = reason
        rows.append(item)
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS), source.loc[source._eligible].copy()


def _validate_analytic(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "from_pop", "to_pop", "interval_years"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Analytic intervals missing required columns: {missing}")
    result = frame.copy()
    result["interval_years"] = pd.to_numeric(result.interval_years, errors="coerce")
    if result.interval_years.isna().any() or (~np.isfinite(result.interval_years)).any() or \
            result.interval_years.le(0).any():
        raise ValueError("interval_years must be finite and > 0")
    return result


def _density(data: pd.DataFrame) -> dict:
    counts = data.groupby("patient_id", dropna=False).size()
    return {
        "n_intervals_with_biomarker": len(data),
        "n_patients_with_biomarker": int(data.patient_id.nunique()),
        "median_intervals_per_patient": float(counts.median()) if len(counts) else np.nan,
        "iqr_intervals_per_patient": (float(counts.quantile(.75) - counts.quantile(.25))
                                      if len(counts) else np.nan),
        "n_patients_with_1_measurement": int(counts.eq(1).sum()),
        "n_patients_with_ge2_measurements": int(counts.ge(2).sum()),
        "n_patients_with_ge3_measurements": int(counts.ge(3).sum()),
    }


def _prepare_intervals(analytic: pd.DataFrame, biomarker: str) -> tuple[pd.DataFrame, float, float, str]:
    column = f"from_lab__{biomarker}"
    if column not in analytic:
        return pd.DataFrame(), np.nan, np.nan, "missing_from_biomarker_column"
    if AGE_COLUMN not in analytic:
        # Preserve a usable unadjusted analysis; age-adjusted rows explicitly
        # report why they cannot be estimated.
        age = pd.Series(np.nan, index=analytic.index, dtype=float)
    else:
        age = pd.to_numeric(analytic[AGE_COLUMN], errors="coerce")
    data = analytic.copy()
    data["biomarker_value"] = pd.to_numeric(data[column], errors="coerce")
    data["age"] = age
    valid = (data.from_pop.isin(POPS) & data.to_pop.isin(POPS) &
             np.isfinite(data.biomarker_value) & np.isfinite(data.interval_years))
    if AGE_COLUMN in analytic:
        valid &= np.isfinite(data.age)
    data = data.loc[valid].copy()
    mean = float(data.biomarker_value.mean()) if len(data) else np.nan
    sd = float(data.biomarker_value.std(ddof=1)) if len(data) else np.nan
    if not np.isfinite(sd) or sd <= 0:
        return data, mean, sd, "insufficient_biomarker_variability"
    data["biomarker_z"] = (data.biomarker_value - mean) / sd
    return data, mean, sd, ""


def stack_intervals(data: pd.DataFrame, biomarker: str) -> pd.DataFrame:
    """Create one risk row for each competing destination of every interval."""
    pieces = []
    for origin, destination in TRANSITIONS:
        part = data.loc[data.from_pop.eq(origin)].copy()
        if part.empty:
            continue
        part["observed_to_pop"] = part.to_pop
        part["risk_destination"] = destination
        part["transition_id"] = f"{origin}→{destination}"
        part["event"] = part.to_pop.eq(destination).astype(int)
        part["biomarker"] = biomarker
        for column in LONG_COLUMNS:
            if column not in part:
                part[column] = pd.NA
        pieces.append(part[LONG_COLUMNS])
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=LONG_COLUMNS)


def transition_support(data: pd.DataFrame, stacked: pd.DataFrame, biomarker: str,
                       minimum_events: int) -> pd.DataFrame:
    density = _density(data)
    rows = []
    for origin, destination in TRANSITIONS:
        transition_id = f"{origin}→{destination}"
        risk = stacked.loc[stacked.transition_id.eq(transition_id)]
        events = risk.loc[risk.event.eq(1)]
        n_events = len(events)
        rows.append({
            "biomarker": biomarker, "transition_id": transition_id,
            "from_pop": origin, "to_pop": destination,
            "n_intervals_at_risk": len(risk),
            "n_patients_at_risk": int(risk.patient_id.nunique()),
            "n_events": n_events,
            "n_patients_with_event": int(events.patient_id.nunique()),
            **density, "supported_for_model": n_events >= minimum_events,
            "support_reason": "supported" if n_events >= minimum_events else "insufficient_events",
        })
    return pd.DataFrame(rows, columns=SUPPORT_COLUMNS)


def _empty_model_row(biomarker: str, transition_id: str, version: str,
                     support: pd.Series, mean: float, sd: float,
                     metadata: pd.Series | None) -> dict:
    row = {column: np.nan for column in MODEL_COLUMNS}
    row.update(biomarker=biomarker, transition_id=transition_id,
               from_pop=support.from_pop, to_pop=support.to_pop,
               model_version=version, n_intervals=support.n_intervals_at_risk,
               n_patients=support.n_patients_at_risk, n_events=support.n_events,
               biomarker_mean=mean, biomarker_sd=sd, model_status="not_estimable",
               interpretability_status="not_estimable")
    if metadata is not None:
        row.update(selection_tier=metadata.get("selection_tier", np.nan),
                   selection_q_value=metadata.get("selection_q_value", np.nan),
                   robustness_status=metadata.get("robustness_status", np.nan),
                   adjusted_estimate_02=metadata.get("adjusted_estimate", np.nan))
    return row


def _fit_stacked(stacked: pd.DataFrame, supported: list[str], adjusted: bool):
    import statsmodels.api as sm
    data = stacked.loc[stacked.transition_id.isin(supported)].copy()
    intercepts = pd.get_dummies(data.transition_id, dtype=float).reindex(columns=supported, fill_value=0)
    intercepts.columns = [f"intercept::{value}" for value in supported]
    effects = intercepts.mul(data.biomarker_z.to_numpy(), axis=0)
    effects.columns = [f"biomarker::{value}" for value in supported]
    exog = pd.concat([intercepts, effects], axis=1)
    if adjusted:
        exog["age"] = pd.to_numeric(data.age, errors="coerce").to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(data.event.astype(float), exog.astype(float),
                     family=sm.families.Poisson(),
                     offset=np.log(data.interval_years.astype(float))).fit(
                         cov_type="cluster", cov_kwds={"groups": data.patient_id.to_numpy()})
    return fit


def _effect(fit, name: str) -> tuple[float, float, float, float]:
    beta, se = float(fit.params[name]), float(fit.bse[name])
    try:
        return math.exp(beta), math.exp(beta - 1.96 * se), math.exp(beta + 1.96 * se), \
            float(fit.pvalues[name])
    except OverflowError:
        return np.inf, 0.0, np.inf, float(fit.pvalues[name])


def _metadata_for_transition(source: pd.DataFrame, biomarker: str,
                             origin: str, destination: str) -> pd.Series | None:
    rows = source.loc[source.exposure.eq(biomarker)]
    if "from_pop" in rows and "to_pop" in rows:
        exact = rows.loc[rows.from_pop.eq(origin) & rows.to_pop.eq(destination)]
        if not exact.empty:
            return exact.iloc[0]
    if "transition_pair" in rows:
        normalized = rows.transition_pair.astype(str).str.replace(" ", "", regex=False).str.replace("->", "→", regex=False)
        exact = rows.loc[normalized.eq(f"{origin}→{destination}")]
        if not exact.empty:
            return exact.iloc[0]
    return None


def fit_biomarker(biomarker: str, stacked: pd.DataFrame, support: pd.DataFrame,
                  mean: float, sd: float, eligible_source: pd.DataFrame,
                  age_available: bool, preparation_reason: str) -> list[dict]:
    supported = support.loc[support.supported_for_model, "transition_id"].tolist()
    rows = []
    fits: dict[str, object] = {}
    failures: dict[str, str] = {}
    for version, adjusted in (("same_sample_clustered", False),
                              ("age_adjusted_clustered", True)):
        if preparation_reason:
            failures[version] = preparation_reason
        elif adjusted and not age_available:
            failures[version] = f"missing_required_age_column:{AGE_COLUMN}"
        elif not supported:
            failures[version] = "insufficient_transition_support"
        else:
            try:
                fits[version] = _fit_stacked(stacked, supported, adjusted)
            except Exception as exc:
                failures[version] = f"model_failed:{type(exc).__name__}:{str(exc)[:160]}"
                logging.warning("Model failure for %s (%s): %s", biomarker, version, exc)

    for support_row in support.itertuples(index=False):
        metadata = _metadata_for_transition(eligible_source, biomarker,
                                            support_row.from_pop, support_row.to_pop)
        for version in ("same_sample_clustered", "age_adjusted_clustered"):
            row = _empty_model_row(biomarker, support_row.transition_id, version,
                                   support_row, mean, sd, metadata)
            if not support_row.supported_for_model:
                row["model_status"] = "insufficient_transition_support"
                row["interpretability_reason"] = "insufficient_transition_events"
            elif version in failures:
                row["interpretability_reason"] = failures[version]
            else:
                fit = fits[version]
                values = _effect(fit, f"biomarker::{support_row.transition_id}")
                row.update(estimate_per_sd=values[0], ci95_low=values[1],
                           ci95_high=values[2], p_value=values[3], model_status="success")
                if version == "age_adjusted_clustered":
                    age_values = _effect(fit, "age")
                    row.update(age_estimate=age_values[0], age_ci95_low=age_values[1],
                               age_ci95_high=age_values[2], age_p_value=age_values[3])
                finite_positive = all(np.isfinite(value) and value > 0 for value in values[:3])
                if finite_positive:
                    row.update(interpretability_status="interpretable", interpretability_reason="")
                else:
                    row.update(interpretability_status="not_interpretable",
                               interpretability_reason="ci_or_estimate_non_finite_or_nonpositive")
                prior = pd.to_numeric(pd.Series([row["adjusted_estimate_02"]]), errors="coerce").iloc[0]
                if np.isfinite(prior) and prior > 0 and finite_positive:
                    row["direction_consistent_with_02"] = bool(
                        np.sign(math.log(values[0])) == np.sign(math.log(prior)))
            rows.append(row)
    return rows


def summarize_biomarkers(candidates: pd.DataFrame, long_data: pd.DataFrame,
                         support: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for biomarker in candidates.loc[candidates.eligible_for_03, "exposure"]:
        intervals = long_data.loc[long_data.biomarker.eq(biomarker)].drop_duplicates(
            ["patient_id", "from_clinical_episode_id", "to_clinical_episode_id"])
        density = _density(intervals)
        biomarker_support = support.loc[support.biomarker.eq(biomarker)]
        effects = models.loc[(models.biomarker.eq(biomarker)) &
                             models.model_status.eq("success") &
                             models.interpretability_status.eq("interpretable")]
        # Each biomarker-transition is one scientific hypothesis.  Prefer the
        # primary age-adjusted result and use the unadjusted result only when
        # that transition has no estimable adjusted model.
        adjusted_effects = effects.loc[effects.model_version.eq("age_adjusted_clustered")]
        fallback_effects = effects.loc[
            effects.model_version.eq("same_sample_clustered") &
            ~effects.transition_id.isin(adjusted_effects.transition_id)
        ]
        selected_effects = pd.concat([adjusted_effects, fallback_effects], ignore_index=True)
        # Comparability with step 02 is assessed only with the primary model;
        # unadjusted and adjusted versions must not duplicate a comparison.
        comparable = adjusted_effects.direction_consistent_with_02.dropna()
        if effects.empty:
            overall = ("insufficient_transition_support" if not biomarker_support.supported_for_model.any()
                       else "not_estimable")
        else:
            overall = "estimable"
        rows.append({
            "biomarker": biomarker, "n_patients": density["n_patients_with_biomarker"],
            "n_intervals": density["n_intervals_with_biomarker"],
            **{key: density[key] for key in (
                "median_intervals_per_patient", "iqr_intervals_per_patient",
                "n_patients_with_1_measurement", "n_patients_with_ge2_measurements",
                "n_patients_with_ge3_measurements")},
            "n_supported_transitions": int(biomarker_support.supported_for_model.sum()),
            "n_estimable_effects": selected_effects.transition_id.nunique(),
            "n_directionally_comparable_with_02": len(comparable),
            "n_directionally_consistent_with_02": int(comparable.astype(bool).sum()),
            "overall_status": overall,
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def apply_fdr_by_model(models: pd.DataFrame) -> None:
    """Assign BH q-values within each model-version family, in place."""
    for model_version in ("same_sample_clustered", "age_adjusted_clustered"):
        estimable = (models.model_version.eq(model_version) &
                     models.model_status.eq("success") &
                     pd.to_numeric(models.p_value, errors="coerce").notna())
        models.loc[estimable, "q_value"] = benjamini_hochberg(
            pd.to_numeric(models.loc[estimable, "p_value"], errors="coerce").to_numpy())


def forest_plot(models: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    valid = models.loc[(models.model_status == "success") &
                       (models.interpretability_status == "interpretable")].copy()
    chosen = []
    for _, group in valid.groupby(["biomarker", "transition_id"], sort=False):
        adjusted = group.loc[group.model_version.eq("age_adjusted_clustered")]
        chosen.append((adjusted if not adjusted.empty else group).iloc[0])
    shown = pd.DataFrame(chosen)
    if shown.empty:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.axis("off")
        ax.text(.5, .5, "No estimable multistate biomarker effects", ha="center", va="center")
    else:
        shown = shown.sort_values(["biomarker", "transition_id"]).reset_index(drop=True)
        y = np.arange(len(shown))
        fig, ax = plt.subplots(figsize=(10, max(4, .42 * len(shown))))
        estimates = shown.estimate_per_sd.astype(float)
        ax.errorbar(estimates, y, xerr=[estimates - shown.ci95_low,
                                       shown.ci95_high - estimates], fmt="o", capsize=2)
        versions = shown.model_version.map({"age_adjusted_clustered": "age-adjusted",
                                            "same_sample_clustered": "unadjusted"})
        ax.set_yticks(y, shown.biomarker + " | " + shown.transition_id + " (" + versions + ")")
        ax.axvline(1, color="grey", linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel("Intensity Ratio per 1 SD increase (95% CI)")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dirs = create_study_dirs(STUDY_NAME)
    logging.basicConfig(filename=dirs["logs"] / "03_pharma_multistate_biomarkers.log",
                        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    sensitivity = pd.read_csv(args.sensitivity_summary)
    candidates, eligible_source = build_candidate_trace(sensitivity)
    analytic = _validate_analytic(load_parquet(args.analytic))
    logging.info("Candidate biomarkers from 02: %d", len(candidates))
    logging.info("Eligible biomarkers for 03: %d", int(candidates.eligible_for_03.sum()))
    logging.info("Loaded intervals: %d; unique patients: %d", len(analytic), analytic.patient_id.nunique())
    candidates.to_csv(dirs["tables"] / "03_pharma_multistate_candidates.csv", index=False)

    minimum = int(config["minimum_counts"]["events_for_multivariable_model"])
    all_long, all_support, all_models = [], [], []
    for biomarker in candidates.loc[candidates.eligible_for_03, "exposure"]:
        intervals, mean, sd, reason = _prepare_intervals(analytic, str(biomarker))
        stacked = stack_intervals(intervals, str(biomarker))
        support = transition_support(intervals, stacked, str(biomarker), minimum)
        models = fit_biomarker(str(biomarker), stacked, support, mean, sd,
                               eligible_source, AGE_COLUMN in analytic, reason)
        all_long.append(stacked)
        all_support.append(support)
        all_models.extend(models)
        logging.info("Support for %s: intervals=%d patients=%d supported_transitions=%d",
                     biomarker, len(intervals), intervals.patient_id.nunique(),
                     support.supported_for_model.sum())

    long_data = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame(columns=LONG_COLUMNS)
    support = pd.concat(all_support, ignore_index=True) if all_support else pd.DataFrame(columns=SUPPORT_COLUMNS)
    models = pd.DataFrame(all_models, columns=MODEL_COLUMNS)
    if len(models):
        apply_fdr_by_model(models)
    summary = summarize_biomarkers(candidates, long_data, support, models)
    support.to_csv(dirs["tables"] / "03_pharma_multistate_support.csv", index=False)
    models.to_csv(dirs["tables"] / "03_pharma_multistate_models.csv", index=False)
    summary.to_csv(dirs["tables"] / "03_pharma_multistate_summary.csv", index=False)
    long_data.to_parquet(dirs["analytic"] / "03_pharma_multistate_long.parquet", index=False)
    if not args.dry_run:
        forest_plot(models, dirs["figures"] / "03_pharma_multistate_forest.pdf")

    good = ((models.model_status == "success") &
            (models.interpretability_status == "interpretable")) if len(models) else pd.Series(dtype=bool)
    estimable_effects = (models.loc[good, ["biomarker", "transition_id"]].drop_duplicates().shape[0]
                         if len(models) else 0)
    possible_effects = (models[["biomarker", "transition_id"]].drop_duplicates().shape[0]
                        if len(models) else 0)
    logging.info("Model failures/non-estimable rows: %d", len(models) - int(good.sum()))
    logging.info("Final estimable biomarker-transition effects: %d", estimable_effects)
    print("ORDER TO REVIEW PHARMA MULTISTATE OUTPUTS\n\n"
          "1. 03_pharma_multistate_candidates.csv\n"
          "2. 03_pharma_multistate_support.csv\n"
          "3. 03_pharma_multistate_models.csv\n"
          "4. 03_pharma_multistate_summary.csv\n"
          "5. 03_pharma_multistate_forest.pdf\n"
          "6. 03_pharma_multistate_long.parquet\n")
    biomarker_intervals = long_data.drop_duplicates(
        ["biomarker", "patient_id", "from_clinical_episode_id", "to_clinical_episode_id"])
    clinical_intervals = long_data.drop_duplicates(
        ["patient_id", "from_clinical_episode_id", "to_clinical_episode_id"])
    print(f"Candidate biomarkers from 02: {len(candidates)}")
    print(f"Eligible biomarkers: {int(candidates.eligible_for_03.sum())}")
    print(f"Patients modeled: {long_data.patient_id.nunique() if len(long_data) else 0}")
    print(f"Unique clinical intervals modeled: {len(clinical_intervals)}")
    print(f"Biomarker-interval observations modeled: {len(biomarker_intervals)}")
    print(f"Supported transitions: {int(support.supported_for_model.sum()) if len(support) else 0}")
    print(f"Estimable biomarker-transition effects: {estimable_effects}")
    print(f"Non-estimable effects: {possible_effects - estimable_effects}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic", type=Path, default=DEFAULT_ANALYTIC)
    parser.add_argument("--sensitivity-summary", type=Path, default=DEFAULT_SENSITIVITY)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Create analytic tables but omit the forest figure")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""Main longitudinal Pharma analysis of transitions between Pop1, Pop2 and Pop3.

The analysis consumes the frozen integrated episode master and the canonical
adjacent-episode intervals.  Effects are associations (not causal effects), and
all transition covariates are taken exclusively from the FROM episode.
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
from src.studies._shared import (create_study_dirs, enrich_transition_intervals,
                                 load_parquet, validate_integrated_dataset,
                                 validate_predictors, validate_transition_intervals)

POPS = ("Pop1", "Pop2", "Pop3")
CORE = ["patient_id", "from_clinical_episode_id", "to_clinical_episode_id",
        "from_clinical_anchor_date", "to_clinical_anchor_date", "interval_years",
        "from_pop", "to_pop", "transition_pair"]
FEATURES = {
    "age": ("clinical", ["ids__age_at_visit", "age_at_visit", "age"]),
    "sex": ("clinical", ["ids__sex", "ids__gender", "sex", "gender"]),
    "disease_duration": ("clinical", ["time_since_diagnosis_years", "disease_duration_years"]),
    "protocol": ("clinical", ["protocol", "ids__protocol", "ids__protocol_number", "parent_protocol"]),
    "ssa": ("serology", ["anti_ro_ssa_status", "ssa_status", "baseline_anti_ro_ssa"]),
    "ssb": ("serology", ["anti_la_ssb_status", "ssb_status", "baseline_anti_la_ssb"]),
    "ana": ("serology", ["ana_status", "baseline_ana"]),
    "rf": ("serology", ["rf_status", "baseline_rf"]),
    "cryoglobulinemia": ("serology", ["cryoglobulinemia_status", "baseline_cryoglobulinemia"]),
    "igg": ("laboratory", ["igg__value", "immunoglobulin_g__value", "igg"]),
    "c3": ("laboratory", ["c3__value", "complement_c3__value", "c3"]),
    "c4": ("laboratory", ["c4__value", "complement_c4__value", "c4"]),
    "essdai_total": ("essdai", ["essdai_total"]),
    "esspri_total": ("esspri", ["esspri_total_observed"]),
    "dryness": ("esspri", ["esspri_dryness", "dryness"]),
    "fatigue": ("esspri", ["esspri_fatigue", "fatigue"]),
    "pain": ("esspri", ["esspri_pain", "pain"]),
}
DOMAIN_NAMES = ("constitutional", "lymphadenopathy", "glandular", "articular",
                "cutaneous", "pulmonary", "renal", "muscular", "pns", "cns",
                "hematological", "hematologic", "biological")


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the Pharma configuration") from exc
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Pharma configuration must be a mapping")
    return config


def _resolve(master: pd.DataFrame) -> tuple[dict[str, tuple[str, str]], list[str]]:
    resolved: dict[str, tuple[str, str]] = {}
    for feature, (family, choices) in FEATURES.items():
        found = [column for column in choices if column in master.columns]
        if found:
            resolved[feature] = (family, found[0])
    domains = []
    for domain in DOMAIN_NAMES:
        score = [f"essdai_{domain}_score", f"{domain}_domain_score"]
        flag = [f"essdai_{domain}_active", f"eg_{domain}_active", f"{domain}_active"]
        found = next((x for x in score if x in master.columns), None)
        found = found or next((x for x in flag if x in master.columns), None)
        if found and found not in domains:
            domains.append(found)
            resolved[f"essdai_domain_{domain}"] = ("essdai_domain", found)
    return resolved, domains


def _binary(series: pd.Series) -> pd.Series:
    """Interpret explicit statuses while preserving missing/unknown values."""
    numeric = pd.to_numeric(series, errors="coerce").where(lambda x: x.isin([0, 1]))
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map({"negative": 0, "neg": 0, "no": 0, "false": 0, "-": 0,
                       "positive": 1, "pos": 1, "yes": 1, "true": 1, "+": 1,
                       "inactive": 0, "active": 1})
    return numeric.fillna(mapped).astype("Float64")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _availability(master: pd.DataFrame, resolved: dict, coverage: float) -> pd.DataFrame:
    rows = []
    requested = list(FEATURES) + [f"essdai_domain_{x}" for x in DOMAIN_NAMES]
    for feature in requested:
        item = resolved.get(feature)
        family, column = item if item else (FEATURES.get(feature, ("essdai_domain", []))[0], "")
        nonmissing = int(master[column].notna().sum()) if column else 0
        pct = nonmissing / len(master) if len(master) else 0.0
        patients = int(master.loc[master[column].notna(), "patient_id"].nunique()) if column else 0
        sufficient = bool(column and pct >= coverage)
        rows.append({"feature": feature, "family": family, "resolved_column": column,
                     "n_nonmissing": nonmissing, "pct_nonmissing": pct,
                     "n_patients": patients, "available_for_transition_model": sufficient,
                     "available_for_longitudinal_model": bool(column and patients >= 2 and nonmissing >= 4),
                     "status": "available" if column else "not_available"})
    return pd.DataFrame(rows)


def build_analytic(intervals: pd.DataFrame, master: pd.DataFrame, resolved: dict) -> pd.DataFrame:
    columns = list(dict.fromkeys(column for _, column in resolved.values()))
    enriched = enrich_transition_intervals(intervals, master, columns, columns)
    if not {"from_clinical_anchor_date", "to_clinical_anchor_date"}.issubset(enriched):
        dates = master[["patient_id", "clinical_episode_id", "clinical_anchor_date"]]
        for side in ("from", "to"):
            right = dates.rename(columns={"clinical_episode_id": f"{side}_clinical_episode_id",
                                          "clinical_anchor_date": f"{side}_clinical_anchor_date"})
            enriched = enriched.merge(right, on=["patient_id", f"{side}_clinical_episode_id"],
                                      how="left", validate="many_to_one")
    valid = enriched.from_pop.isin(POPS) & enriched.to_pop.isin(POPS)
    enriched = enriched.loc[valid].copy()
    if enriched.empty or (_numeric(enriched.interval_years) <= 0).any():
        if not enriched.empty:
            raise ValueError("interval_years must be positive")
    enriched["transition_pair"] = enriched.from_pop.astype(str) + " -> " + enriched.to_pop.astype(str)
    rename = {"from_esspri_total_observed": "from_esspri_total",
              "to_esspri_total_observed": "to_esspri_total"}
    enriched.rename(columns=rename, inplace=True)
    aliases = {feature: column for feature, (_, column) in resolved.items()}
    for feature in ("essdai_total", "esspri_total", "dryness", "fatigue", "pain", "igg", "c3", "c4"):
        source = aliases.get(feature)
        if source:
            fcol = "from_esspri_total" if feature == "esspri_total" else f"from_{source}"
            tcol = "to_esspri_total" if feature == "esspri_total" else f"to_{source}"
            canonical = "essdai" if feature == "essdai_total" else feature
            if fcol in enriched and tcol in enriched:
                both = enriched[fcol].notna() & enriched[tcol].notna()
                enriched[f"delta_{canonical}"] = (_numeric(enriched[tcol]) - _numeric(enriched[fcol])).where(both)
                enriched[f"delta_{canonical}_per_year"] = enriched[f"delta_{canonical}"] / enriched.interval_years
                if source != canonical and feature not in ("essdai_total", "esspri_total"):
                    enriched.rename(columns={fcol: f"from_{canonical}", tcol: f"to_{canonical}"}, inplace=True)
    for feature, (family, source) in resolved.items():
        if family == "essdai_domain":
            if source.endswith("_active"):
                from_value = _binary(enriched[f"from_{source}"])
                to_value = _binary(enriched[f"to_{source}"])
            else:
                from_value = _numeric(enriched[f"from_{source}"])
                to_value = _numeric(enriched[f"to_{source}"])
            both = from_value.notna() & to_value.notna()
            enriched[f"delta_{source}"] = (to_value - from_value).where(both)
    return enriched[CORE + [c for c in enriched.columns if c not in CORE]]


def _change_table(frame: pd.DataFrame, variables: dict[str, str]) -> pd.DataFrame:
    rows = []
    for pair, group in frame.groupby("transition_pair", sort=True):
        for label, stem in variables.items():
            endpoint_stem = "essdai_total" if stem == "essdai" else stem
            fcol, tcol, dcol = f"from_{endpoint_stem}", f"to_{endpoint_stem}", f"delta_{stem}"
            if not {fcol, tcol, dcol}.issubset(group):
                continue
            paired = group[group[fcol].notna() & group[tcol].notna()]
            delta = _numeric(paired[dcol])
            rows.append({"transition_pair": pair, "variable": label, "n_paired": len(paired),
                         "from_median": _numeric(paired[fcol]).median(), "to_median": _numeric(paired[tcol]).median(),
                         "median_change": delta.median(), "q1_change": delta.quantile(.25),
                         "q3_change": delta.quantile(.75),
                         "median_change_per_year": (delta / paired.interval_years).median()})
    return pd.DataFrame(rows, columns=["transition_pair", "variable", "n_paired", "from_median",
        "to_median", "median_change", "q1_change", "q3_change", "median_change_per_year"])


def _serology(frame: pd.DataFrame, resolved: dict) -> pd.DataFrame:
    rows = []
    for feature, (family, source) in resolved.items():
        if family != "serology": continue
        before, after = _binary(frame[f"from_{source}"]), _binary(frame[f"to_{source}"])
        status = pd.Series("unknown", index=frame.index)
        known = before.notna() & after.notna()
        status.loc[known] = before[known].map({0.: "negative", 1.: "positive"}) + " -> " + after[known].map({0.: "negative", 1.: "positive"})
        work = pd.DataFrame({"transition_pair": frame.transition_pair, "marker": feature, "status": status})
        counts = work.groupby(["transition_pair", "marker", "status"]).size().rename("n").reset_index()
        counts["n_intervals"] = counts.groupby(["transition_pair", "marker"])["n"].transform("sum")
        counts["pct"] = counts.n / counts.n_intervals
        rows.append(counts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["transition_pair", "marker", "status", "n", "n_intervals", "pct"])


def transition_associations(frame: pd.DataFrame, availability: pd.DataFrame, resolved: dict, config: dict) -> pd.DataFrame:
    """Fit transition-specific proportional intensity models with FROM covariates.

    A Poisson likelihood with log(interval_years) offset estimates a cause-specific
    continuous-time transition intensity ratio for each pre-transition exposure.
    """
    minimum = int(config["minimum_counts"]["events_for_multivariable_model"])
    eligible = availability.loc[availability.available_for_transition_model &
                                ~availability.feature.isin(["essdai_total", "esspri_total"]), "feature"]
    rows = []
    for origin in POPS:
        origin_frame = frame[frame.from_pop.eq(origin)]
        for destination in POPS:
            if destination == origin: continue
            events = int(origin_frame.to_pop.eq(destination).sum())
            for feature in eligible:
                source = resolved[feature][1]
                canonical_exposure = f"from_{feature}"
                source_exposure = f"from_{source}"
                exposure = (
                    canonical_exposure
                    if canonical_exposure in origin_frame.columns
                    else source_exposure
                )
                validate_predictors([exposure])
                sample = origin_frame[["patient_id", "to_pop", "interval_years", exposure]].dropna().copy()
                sample["event"] = sample.to_pop.eq(destination).astype(int)
                status = "insufficient_events"; estimate = low = high = pvalue = np.nan
                if int(sample.event.sum()) >= minimum and int((1-sample.event).sum()) >= minimum and sample[exposure].nunique() > 1:
                    try:
                        import statsmodels.api as sm
                        x = sample[exposure]
                        if not pd.api.types.is_numeric_dtype(x):
                            interpreted = _binary(x)
                            if interpreted.notna().sum() == 0 and x.dropna().nunique() == 2:
                                interpreted = pd.Series(pd.factorize(x)[0], index=x.index).where(x.notna())
                            x = interpreted
                        fitdata = pd.DataFrame({"event": sample.event, "x": _numeric(x), "time": sample.interval_years})
                        fitdata["event"] = pd.to_numeric(fitdata["event"], errors="coerce").astype(float)
                        fitdata["x"] = pd.to_numeric(fitdata["x"], errors="coerce").astype(float)
                        fitdata["time"] = pd.to_numeric(fitdata["time"], errors="coerce").astype(float)
                        fitdata = fitdata.dropna()
                        model = sm.GLM(fitdata.event, sm.add_constant(fitdata.x), family=sm.families.Poisson(),
                                       offset=np.log(fitdata.time)).fit(cov_type="HC0")
                        beta, se = model.params["x"], model.bse["x"]
                        estimate, low, high, pvalue = math.exp(beta), math.exp(beta-1.96*se), math.exp(beta+1.96*se), model.pvalues["x"]
                        status = "success"
                    except Exception as exc:
                        logging.warning("Intensity model failed for %s %s->%s: %s", feature, origin, destination, exc)
                        status = "model_failed"
                rows.append({"from_pop": origin, "to_pop": destination, "exposure": feature,
                             "n_intervals": len(sample), "n_patients": int(sample.patient_id.nunique()),
                             "events": int(sample.event.sum()), "estimate": estimate, "ci95_low": low,
                             "ci95_high": high, "p_value": pvalue, "model_status": status})
    return pd.DataFrame(rows, columns=["from_pop", "to_pop", "exposure", "n_intervals", "n_patients",
                                      "events", "estimate", "ci95_low", "ci95_high", "p_value", "model_status"])


def ml_feasibility(frame: pd.DataFrame, availability: pd.DataFrame, config: dict) -> pd.DataFrame:
    thresholds, rows = config["minimum_counts"], []
    candidates = availability[~availability.family.isin(["essdai"])]
    covered = int(candidates.available_for_transition_model.sum())
    for origin in POPS:
        g = frame[frame.from_pop.eq(origin)]
        for destination in POPS:
            if destination == origin: continue
            events = int(g.to_pop.eq(destination).sum()); nonevents = len(g) - events
            if g.patient_id.nunique() < thresholds["patients_for_prediction"]: status = "insufficient_patients"
            elif events < thresholds["events_for_prediction"]: status = "insufficient_events"
            elif nonevents < thresholds["nonevents_for_prediction"]: status = "insufficient_nonevents"
            elif covered == 0: status = "insufficient_variable_coverage"
            else: status = "potentially_feasible"
            rows.append({"target": f"{origin} -> {destination}", "origin_pop": origin,
                         "destination_pop": destination, "n_intervals": len(g),
                         "n_patients": int(g.patient_id.nunique()), "events": events,
                         "nonevents": nonevents, "n_candidate_variables": len(candidates),
                         "n_variables_with_sufficient_coverage": covered, "status": status})
    return pd.DataFrame(rows)


def longitudinal_models(master: pd.DataFrame, resolved: dict) -> pd.DataFrame:
    """Fit simple time-aware repeated-measure models when data support them."""
    rows = []
    for feature, (family, source) in resolved.items():
        if feature not in ("essdai_total", "esspri_total", "igg", "c3", "c4") and family != "essdai_domain":
            continue
        work = master[["patient_id", "clinical_anchor_date", source]].copy()
        work["value"] = _binary(work[source]) if family == "essdai_domain" else _numeric(work[source])
        work.dropna(subset=["value", "clinical_anchor_date"], inplace=True)
        work["time_years"] = (pd.to_datetime(work.clinical_anchor_date) -
                              pd.to_datetime(work.groupby("patient_id").clinical_anchor_date.transform("min"))).dt.days / 365.25
        repeated = work.groupby("patient_id").size().ge(2).sum()
        status, estimate, low, high, pvalue = "insufficient_repeated_measures", *([np.nan] * 4)
        if len(work) >= 10 and repeated >= 3 and work.time_years.nunique() > 1:
            try:
                import statsmodels.api as sm
                if family == "essdai_domain":
                    fit = sm.GEE(work.value.astype(float), sm.add_constant(work.time_years),
                                 groups=work.patient_id, family=sm.families.Binomial()).fit()
                    model = "GEE_binomial"; beta, se = fit.params["time_years"], fit.bse["time_years"]
                    estimate, low, high = math.exp(beta), math.exp(beta-1.96*se), math.exp(beta+1.96*se)
                else:
                    fit = sm.MixedLM.from_formula("value ~ time_years", groups="patient_id", data=work).fit(reml=False)
                    model = "linear_mixed_effects"; estimate, se = fit.params["time_years"], fit.bse["time_years"]
                    low, high = estimate-1.96*se, estimate+1.96*se
                pvalue, status = fit.pvalues["time_years"], "success"
            except Exception as exc:
                logging.warning("Longitudinal model failed for %s: %s", feature, exc)
                model, status = ("GEE_binomial" if family == "essdai_domain" else "linear_mixed_effects"), "model_failed"
        else:
            model = "GEE_binomial" if family == "essdai_domain" else "linear_mixed_effects"
        rows.append({"feature": feature, "model": model, "n_observations": len(work),
                     "n_patients": int(work.patient_id.nunique()), "patients_with_repeated_measures": int(repeated),
                     "time_effect": estimate, "ci95_low": low, "ci95_high": high,
                     "p_value": pvalue, "model_status": status})
    return pd.DataFrame(rows)


def _plot_heatmap(table: pd.DataFrame, row: str, value: str, path: Path, title: str) -> bool:
    if table.empty or table[value].notna().sum() == 0: return False
    import matplotlib.pyplot as plt
    pivot = table.pivot(index=row, columns="transition_pair", values=value)
    fig, ax = plt.subplots(figsize=(max(8, .9*len(pivot.columns)), max(3, .55*len(pivot))))
    image = ax.imshow(pivot, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index); ax.set_title(title)
    fig.colorbar(image, ax=ax, label=value.replace("_", " ")); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return True


def _plot_longitudinal_trajectory(master: pd.DataFrame, source: str, path: Path,
                                  title: str, ylabel: str) -> bool:
    """Plot observations relative to each patient's first available assessment."""
    work = master[["patient_id", "clinical_anchor_date", source]].copy()
    work["value"] = _numeric(work[source])
    work.dropna(subset=["value", "clinical_anchor_date"], inplace=True)
    if work.empty:
        return False
    work["clinical_anchor_date"] = pd.to_datetime(work.clinical_anchor_date)
    first_assessment = work.groupby("patient_id").clinical_anchor_date.transform("min")
    work["time_years"] = (work.clinical_anchor_date - first_assessment).dt.days / 365.25

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for _, patient in work.sort_values("time_years").groupby("patient_id"):
        ax.plot(patient.time_years, patient.value, color="tab:blue", alpha=.12,
                linewidth=.7, marker="o", markersize=2)

    if work.time_years.nunique() == 1:
        aggregate = work.groupby("time_years", as_index=False).value.median()
    else:
        bins = min(12, max(2, int(np.sqrt(len(work)))))
        edges = np.linspace(work.time_years.min(), work.time_years.max(), bins + 1)
        work["time_bin"] = pd.cut(work.time_years, bins=np.unique(edges), include_lowest=True)
        aggregate = work.groupby("time_bin", observed=True).agg(
            time_years=("time_years", "median"), value=("value", "median")).dropna()
    ax.plot(aggregate.time_years, aggregate.value, color="black", linewidth=2.2,
            marker="o", markersize=4, label="Median trajectory")
    ax.set_xlabel("Years since first available assessment")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{work.patient_id.nunique()} patients; {len(work)} observations")
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return True


def make_plots(frame: pd.DataFrame, associations: pd.DataFrame, essdai: pd.DataFrame,
               domains: pd.DataFrame, components: pd.DataFrame, labs: pd.DataFrame,
               master: pd.DataFrame, resolved: dict, figures: Path) -> set[str]:
    import matplotlib.pyplot as plt
    made = set(); success = associations[associations.model_status.eq("success")]
    path = figures / "01_pharma_transition_associations_forest.pdf"
    if len(success):
        labels = success.from_pop + "→" + success.to_pop + ": " + success.exposure
        y = np.arange(len(success)); fig, ax = plt.subplots(figsize=(9, max(4, .3*len(success))))
        ax.errorbar(success.estimate, y, xerr=[success.estimate-success.ci95_low, success.ci95_high-success.estimate], fmt="o")
        ax.axvline(1, color="grey", ls="--"); ax.set_xscale("log"); ax.set_yticks(y, labels); ax.set_xlabel("Intensity ratio (95% CI)")
        fig.tight_layout(); fig.savefig(path); plt.close(fig); made.add(path.name)
    path = figures / "01_pharma_essdai_change_by_transition.pdf"
    if "delta_essdai" in frame and frame.delta_essdai.notna().any():
        groups = [(name, g.delta_essdai.dropna()) for name, g in frame.groupby("transition_pair") if g.delta_essdai.notna().any()]
        fig, ax = plt.subplots(figsize=(10, 5)); ax.boxplot([x[1] for x in groups], labels=[f"{x[0]}\nn={len(x[1])}" for x in groups])
        ax.axhline(0, color="grey", ls="--"); ax.set_ylabel("Change in ESSDAI"); ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); fig.savefig(path); plt.close(fig); made.add(path.name)
    specs = [(domains, "domain", "median_change", "01_pharma_essdai_domains_heatmap.pdf", "ESSDAI domain change"),
             (components, "variable", "median_change", "01_pharma_esspri_components_heatmap.pdf", "ESSPRI component change")]
    for table, row, value, name, title in specs:
        if _plot_heatmap(table, row, value, figures/name, title): made.add(name)
    if not labs.empty:
        standardized = labs.copy()
        standardized["standardized_change"] = standardized.groupby("variable").median_change.transform(lambda x: x / x.std() if x.std() else np.nan)
        name = "01_pharma_labs_change_heatmap.pdf"
        if _plot_heatmap(standardized, "variable", "standardized_change", figures/name, "Standardized laboratory change"): made.add(name)
    longitudinal_specs = [
        ("essdai_total", "01_pharma_essdai_longitudinal_trajectory.pdf",
         "Longitudinal ESSDAI trajectory", "ESSDAI total"),
        ("esspri_total", "01_pharma_esspri_longitudinal_trajectory.pdf",
         "Longitudinal ESSPRI trajectory", "ESSPRI total observed"),
    ]
    for feature, name, title, ylabel in longitudinal_specs:
        if feature in resolved and _plot_longitudinal_trajectory(
                master, resolved[feature][1], figures/name, title, ylabel):
            made.add(name)
    return made


def _save(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False)


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config); coverage = float(config["coverage"]["minimum_feature_coverage"])
    dirs = create_study_dirs("pharma/01_run_pharma_main")
    logging.basicConfig(filename=dirs["logs"] / "01_run_pharma_main.log", level=logging.INFO, force=True)
    master, intervals = load_parquet(args.integrated), load_parquet(args.transitions)
    validate_integrated_dataset(master); validate_transition_intervals(intervals, master)
    resolved, domain_sources = _resolve(master)
    availability = _availability(master, resolved, coverage)
    analytic = build_analytic(intervals, master, resolved)
    analytic.to_parquet(dirs["analytic"] / "01_pharma_transition_intervals.parquet", index=False)
    associations = transition_associations(analytic, availability, resolved, config)
    essdai = _change_table(analytic, {"ESSDAI total": "essdai"})
    esspri = _change_table(analytic, {"ESSPRI total": "esspri_total"})
    components = _change_table(analytic, {"ESSPRI total": "esspri_total",
                                          "dryness": "dryness", "fatigue": "fatigue", "pain": "pain"})
    domain_map = {source: source for source in domain_sources}
    domains = _change_table(analytic, domain_map).rename(columns={"variable": "domain"})
    if not domains.empty:
        domains["from_summary"] = domains.from_median; domains["to_summary"] = domains.to_median
        domains["change_summary"] = domains.median_change
        domains = domains[["transition_pair", "domain", "n_paired", "from_summary", "to_summary", "change_summary", "median_change"]]
    labs = _change_table(analytic, {x.upper() if x == "igg" else x.upper(): x for x in ("igg", "c3", "c4")})
    serology = _serology(analytic, resolved)
    feasibility = ml_feasibility(analytic, availability, config)
    longitudinal = longitudinal_models(master, resolved)
    tables = {"01_pharma_feature_availability.csv": availability,
              "01_pharma_transition_associations.csv": associations,
              "01_pharma_essdai_change_by_transition.csv": essdai.drop(columns="variable", errors="ignore"),
              "01_pharma_essdai_domains_change_by_transition.csv": domains,
              "01_pharma_esspri_change_by_transition.csv": esspri.drop(columns="variable", errors="ignore"),
              "01_pharma_esspri_components_change_by_transition.csv": components,
              "01_pharma_labs_change_by_transition.csv": labs,
              "01_pharma_serology_status_by_transition.csv": serology,
              "01_pharma_ml_feasibility.csv": feasibility,
              "01_pharma_longitudinal_models.csv": longitudinal}
    for name, table in tables.items(): _save(table, dirs["tables"] / name)
    made = set() if args.dry_run else make_plots(
        analytic, associations, essdai, domains, components, labs, master, resolved, dirs["figures"])
    order = ["01_pharma_feature_availability.csv", "01_pharma_transition_associations.csv",
             "01_pharma_transition_associations_forest.pdf", "01_pharma_essdai_change_by_transition.csv",
             "01_pharma_essdai_change_by_transition.pdf", "01_pharma_essdai_domains_change_by_transition.csv",
             "01_pharma_essdai_domains_heatmap.pdf", "01_pharma_esspri_change_by_transition.csv",
             "01_pharma_esspri_components_change_by_transition.csv", "01_pharma_esspri_components_heatmap.pdf",
             "01_pharma_labs_change_by_transition.csv", "01_pharma_labs_change_heatmap.pdf",
             "01_pharma_serology_status_by_transition.csv", "01_pharma_ml_feasibility.csv",
             "01_pharma_longitudinal_models.csv", "01_pharma_essdai_longitudinal_trajectory.pdf",
             "01_pharma_esspri_longitudinal_trajectory.pdf"]
    print("ORDER TO REVIEW PHARMA MAIN OUTPUTS\n")
    for number, name in enumerate(order, 1):
        available = name in tables or name in made
        print(f"{number}. {name}" + ("" if available else " — NOT AVAILABLE"))
    print(f"\nAnalytic dataset:\n{len(order) + 1}. 01_pharma_transition_intervals.parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    parser.add_argument("--transitions", type=Path, default=common.POP_TRANSITION_INTERVALS_PARQUET)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

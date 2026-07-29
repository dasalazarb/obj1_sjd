#!/usr/bin/env python3
"""Adjusted longitudinal associations across six clinical domains.

The script consumes (and never re-derives) the canonical observation-process
visits and baseline phenotype covariates.  Baseline Pop is partly defined by
baseline ESSDAI and ESSPRI, so their trajectories may reflect group definition
and regression to the mean.  Estimates are observational, not causal effects.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402

POPS = ("Pop1", "Pop2", "Pop3")
TIME = "time_since_observed_baseline_years"
REQUIRED = ["patient_id", "visit_id", "visit_date", "visit_number", "is_observed_baseline",
            "time_since_observed_baseline_days", TIME]
OUTCOME_SPECS: Dict[str, Dict[str, Any]] = {
    "essdai": {"variable":"essdai_total", "label":"ESSDAI", "availability":"avail_essdai", "unit":"ESSDAI points", "higher_burden":True},
    "esspri": {"variable":"esspri_total", "label":"ESSPRI", "availability":"avail_esspri_complete", "unit":"ESSPRI points", "higher_burden":True},
    "pcs": {"variable":"sf36_pcs", "label":"SF-36 PCS", "availability":"avail_sf36_complete", "unit":"PCS points", "higher_burden":False},
    "mcs": {"variable":"sf36_mcs", "label":"SF-36 MCS", "availability":"avail_sf36_complete", "unit":"MCS points", "higher_burden":False},
    "active_domains": {"variable":"n_extraglandular_domains_active", "label":"Active extraglandular domains", "availability":"avail_overlap_evaluable", "unit":"active domains", "higher_burden":True},
}
FATIGUE_SPECS = {
    "mdafs_global": {"variable":"mdafs_global", "label":"Fatigue (MDAFS global)", "availability":"avail_mdafs_complete", "unit":"MDAFS points", "higher_burden":True},
    "profad_total": {"variable":"profad_total", "label":"Fatigue (PROFAD total)", "availability":"avail_profad_complete", "unit":"PROFAD points", "higher_burden":True},
}
FULL_TERMS = f'{TIME} * C(baseline_pop, Treatment(reference="Pop3")) + age_baseline + C(sex) + C(protocol_group) + baseline_time_since_diagnosis_years'
REDUCED_TERMS = f'{TIME} * C(baseline_pop, Treatment(reference="Pop3")) + age_baseline + C(protocol_group)'

RESULT_COLUMNS = ["outcome_key","outcome_label","source_variable","fatigue_measure_used","outcome_unit","model_type","model_specification","model_status","formula","estimand_type","estimand","baseline_pop","reference_pop","contrast","estimate","standard_error","ci95_low","ci95_high","p_value","fdr_bh_q_value","n_valid_visits","n_eligible_patients","n_pop1_patients","n_pop2_patients","n_pop3_patients","median_valid_visits_per_patient","median_followup_years","interpretation"]
STANDARD_COLUMNS = ["outcome_key","outcome_label","source_variable","baseline_pop","estimand","raw_estimate","outcome_standard_deviation","standardization_source","direction_multiplier","standardized_adverse_estimate","standardized_standard_error","standardized_ci95_low","standardized_ci95_high","p_value","n_eligible_patients","model_type","model_status"]
DIAGNOSTIC_COLUMNS = ["outcome_key","source_variable","availability_flag","n_input_visits","n_valid_visits","n_patients_with_any_valid_visit","n_patients_with_two_or_more_valid_visits","n_patients_meeting_followup_gap","n_eligible_patients","n_pop1_patients","n_pop2_patients","n_pop3_patients","median_valid_visits_per_patient","median_followup_years","maximum_followup_years","outcome_baseline_sd","standardization_source","model_type","model_specification","model_status","converged","random_intercept_variance","fallback_used","fatigue_measure_requested","fatigue_measure_selected","selection_reason","warning"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=common.INTERMEDIATE_DATA_DIR / "20_observation_process_patient_visit.parquet")
    p.add_argument("--baseline-input", type=Path, default=common.INTERMEDIATE_DATA_DIR / "21_multidimensional_baseline_patient.parquet")
    p.add_argument("--min-followup-days", type=int, default=180)
    p.add_argument("--min-patients", type=int, default=20)
    p.add_argument("--fatigue-outcome", choices=["auto", *FATIGUE_SPECS], default="auto")
    g = p.add_mutually_exclusive_group(); g.add_argument("--overwrite", action="store_true", default=True); g.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    p.add_argument("--skip-models", action="store_true")
    return p.parse_args(argv)


def load_inputs(input_path: Path, baseline_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not input_path.exists():
        raise FileNotFoundError("Run src/block_B/20_missingness_and_observation_process.py first.")
    if not baseline_path.exists():
        raise FileNotFoundError("Run src/block_B/21_multidimensional_baseline_phenotypes.py first.")
    read = lambda p: pd.read_parquet(p) if p.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(p)
    return read(input_path), read(baseline_path)


def validate_longitudinal_data(data: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in data]
    if missing: raise ValueError(f"Longitudinal input missing required columns: {missing}")
    if data[["patient_id", "visit_id"]].isna().any().any(): raise ValueError("Missing patient or visit identifiers")
    if data.duplicated(["patient_id", "visit_id"]).any(): raise ValueError("Duplicate patient_id + visit_id keys")
    if data.visit_id.duplicated().any(): raise ValueError("visit_id must be globally unique")
    out = data.copy(); out["visit_date"] = pd.to_datetime(out.visit_date, errors="coerce")
    if out.visit_date.isna().any(): raise ValueError("visit dates must be parseable")
    days = pd.to_numeric(out.time_since_observed_baseline_days, errors="coerce")
    if days.lt(0).any(): raise ValueError("Negative follow-up time")
    out["time_since_observed_baseline_days"] = days
    out[TIME] = pd.to_numeric(out[TIME], errors="coerce")
    bases = out.is_observed_baseline.fillna(False).astype(bool).groupby(out.patient_id).sum()
    if not bases.eq(1).all(): raise ValueError("Each patient must have exactly one observed baseline")
    ordered = out.sort_values(["patient_id", "visit_number"], kind="stable")
    if ordered.groupby("patient_id").visit_date.diff().dt.days.lt(0).any(): raise ValueError("Visit ordering is not chronological")
    if out.groupby("patient_id").size().ge(2).sum() == 0: raise ValueError("No patients with at least two visits")
    return out.sort_values(["patient_id", "visit_date", "visit_number"], kind="stable").reset_index(drop=True)


def merge_baseline_covariates(visits: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if "patient_id" not in baseline or baseline.patient_id.isna().any(): raise ValueError("Baseline input has missing patient_id")
    if baseline.patient_id.duplicated().any(): raise ValueError("Duplicate baseline patient IDs")
    pop = "baseline_pop" if "baseline_pop" in baseline else "pop_status"
    required = [pop, "age_baseline", "sex", "protocol_group", "time_since_diagnosis_years"]
    missing = [c for c in required if c not in baseline]
    if missing: raise ValueError(f"Baseline input missing required columns: {missing}")
    cov = baseline[["patient_id", *required]].copy().rename(columns={pop:"baseline_pop", "time_since_diagnosis_years":"baseline_time_since_diagnosis_years"})
    # Avoid overwriting an existing visit-level baseline_pop while preserving current pop_status.
    base_cols = [c for c in ("baseline_pop", "age_baseline", "sex", "protocol_group", "baseline_time_since_diagnosis_years") if c in visits]
    merged = visits.drop(columns=base_cols).merge(cov, on="patient_id", how="left", validate="many_to_one")
    if len(merged) != len(visits): raise ValueError("Baseline merge changed longitudinal row count")
    return merged


def valid_observation_mask(data: pd.DataFrame, spec: Mapping[str, Any]) -> pd.Series:
    variable, flag = spec["variable"], spec["availability"]
    if variable not in data or flag not in data: return pd.Series(False, index=data.index)
    availability = data[flag].fillna(False).astype(bool)
    return availability & pd.to_numeric(data[variable], errors="coerce").notna()


def build_outcome_support(data: pd.DataFrame, spec: Mapping[str, Any], min_followup_days: int) -> pd.DataFrame:
    """Return one row per patient with valid-visit count and observed span."""
    mask = valid_observation_mask(data, spec)
    valid = data.loc[mask, ["patient_id", "time_since_observed_baseline_days"]]
    support = valid.groupby("patient_id").time_since_observed_baseline_days.agg(n_valid_visits="count", first="min", last="max")
    support["followup_days"] = support["last"] - support["first"]
    support["meets_visit_rule"] = support.n_valid_visits.ge(2) & support.followup_days.ge(min_followup_days)
    return support


def select_fatigue_outcome(data: pd.DataFrame, requested: str, min_followup_days: int) -> Tuple[str, int, int, str]:
    counts = {}
    for name, spec in FATIGUE_SPECS.items():
        support = build_outcome_support(data, spec, min_followup_days)
        counts[name] = int(support.meets_visit_rule.sum()) if len(support) else 0
    if requested != "auto":
        selected, reason = requested, f"Explicitly requested {requested}; support did not alter selection."
    elif counts["profad_total"] > counts["mdafs_global"]:
        selected, reason = "profad_total", "PROFAD had more eligible patients."
    else:
        selected = "mdafs_global"
        reason = "MDAFS had more eligible patients." if counts["mdafs_global"] > counts["profad_total"] else "Eligible support tied; MDAFS preferred as the fatigue-specific global measure."
    return selected, counts["mdafs_global"], counts["profad_total"], reason


def add_eligibility_flags(data: pd.DataFrame, specs: Mapping[str, Mapping[str, Any]], min_followup_days: int) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    out, supports = data.copy(), {}
    for key, spec in specs.items():
        support = build_outcome_support(out, spec, min_followup_days); supports[key] = support
        nmap = support.n_valid_visits if len(support) else pd.Series(dtype=float)
        fmap = support.followup_days if len(support) else pd.Series(dtype=float)
        out[f"n_valid_{key}_visits_patient"] = out.patient_id.map(nmap).fillna(0).astype(int)
        out[f"valid_{key}_followup_days"] = out.patient_id.map(fmap).fillna(0.0)
        visit_rule = out.patient_id.map(support.meets_visit_rule if len(support) else pd.Series(dtype=bool)).fillna(False)
        out[f"eligible_{key}_model"] = visit_rule & out.baseline_pop.isin(POPS)
    return out, supports


def model_formula(variable: str, reduced: bool = False) -> str:
    return f"{variable} ~ {REDUCED_TERMS if reduced else FULL_TERMS}"


def calculate_linear_contrast(params: pd.Series, covariance: pd.DataFrame, weights: Mapping[str, float]) -> Dict[str, float]:
    names = list(params.index); a = np.array([weights.get(n, 0.0) for n in names], dtype=float)
    cov = covariance.loc[names, names].to_numpy(dtype=float)
    estimate = float(a @ params.to_numpy(dtype=float)); variance = float(a @ cov @ a)
    se = math.sqrt(max(variance, 0.0)); z = estimate / se if se > 0 else np.nan
    return {"estimate":estimate, "standard_error":se, "ci95_low":estimate-1.96*se, "ci95_high":estimate+1.96*se,
            "p_value":float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan}


def _interaction_name(names: Sequence[str], pop: str) -> Optional[str]:
    return next((n for n in names if TIME in n and ":" in n and f"[T.{pop}]" in n), None)


def extract_trajectory_estimates(params: pd.Series, covariance: pd.DataFrame) -> List[Dict[str, Any]]:
    names = list(params.index); rows = []
    for pop in POPS:
        weights = {TIME:1.0}; interaction = _interaction_name(names, pop)
        if interaction: weights[interaction] = 1.0
        rows.append({"estimand_type":"annual_slope", "estimand":f"{pop} annual slope", "baseline_pop":pop, "reference_pop":"Pop3", "contrast":"", **calculate_linear_contrast(params, covariance, weights)})
    for pop in ("Pop1", "Pop2"):
        interaction = _interaction_name(names, pop)
        values = calculate_linear_contrast(params, covariance, {interaction:1.0}) if interaction else {k:np.nan for k in ("estimate","standard_error","ci95_low","ci95_high","p_value")}
        rows.append({"estimand_type":"slope_difference", "estimand":f"{pop} minus Pop3 annual slope difference", "baseline_pop":pop, "reference_pop":"Pop3", "contrast":f"{pop} - Pop3", **values})
    return rows


def _attempt_model(frame: pd.DataFrame, formula: str) -> Tuple[Any, str, bool, float]:
    """Try the prespecified random-intercept LMM, then Gaussian GEE."""
    import statsmodels.formula.api as smf
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Gaussian
    try:
        fit = smf.mixedlm(formula, frame, groups=frame["patient_id"]).fit(reml=False, method="lbfgs", disp=False)
        variance = float(np.asarray(fit.cov_re)[0, 0])
        cov = fit.cov_params().loc[fit.fe_params.index, fit.fe_params.index]
        if not fit.converged or variance <= 1e-8 or not np.isfinite(cov.to_numpy()).all(): raise np.linalg.LinAlgError("unstable mixed model")
        return fit, "Linear mixed model", False, variance
    except Exception:
        fit = smf.gee(formula, "patient_id", frame, family=Gaussian(), cov_struct=Exchangeable()).fit()
        if not np.isfinite(fit.cov_params().to_numpy()).all(): raise np.linalg.LinAlgError("GEE covariance unavailable")
        return fit, "Gaussian GEE fallback", True, np.nan


def fit_longitudinal_model(data: pd.DataFrame, key: str, spec: Mapping[str, Any], min_patients: int, skip_models: bool = False) -> Dict[str, Any]:
    variable = spec["variable"]; valid = valid_observation_mask(data, spec); eligible = data[f"eligible_{key}_model"]
    base = data.loc[eligible].patient_id.drop_duplicates(); pop_counts = base.to_frame().merge(data[["patient_id","baseline_pop"]].drop_duplicates(), on="patient_id").baseline_pop.value_counts()
    status = None
    if skip_models: status = "skipped_by_user"
    elif variable not in data or spec["availability"] not in data: status = "skipped_missing_variable"
    elif base.nunique() < min_patients: status = "skipped_insufficient_patients"
    elif any(pop_counts.get(p, 0) < 5 for p in POPS): status = "skipped_insufficient_pop_support"
    model_data = data.loc[valid & eligible].copy() if variable in data else data.iloc[0:0].copy()
    if status is None and pd.to_numeric(model_data[variable], errors="coerce").nunique() < 2: status = "skipped_no_outcome_variation"
    if status is None and model_data[TIME].nunique() < 2: status = "skipped_insufficient_patients"
    answer = {"model_status":status, "model_type":"", "model_specification":"full_adjustment", "formula":model_formula(variable), "fit":None, "rows":[], "converged":False, "random_intercept_variance":np.nan, "fallback_used":False, "warning":""}
    if status: return answer
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        answer["model_status"] = "skipped_dependency_unavailable"; return answer
    full_required = ["age_baseline","sex","protocol_group","baseline_time_since_diagnosis_years"]
    full = model_data.dropna(subset=[variable, TIME, "baseline_pop", *full_required])
    reduced = model_data.dropna(subset=[variable, TIME, "baseline_pop", "age_baseline", "protocol_group"])
    attempts = [(full, False), (reduced, True)]
    errors = []
    for frame, is_reduced in attempts:
        if frame.patient_id.nunique() < min_patients or any(frame.groupby("baseline_pop").patient_id.nunique().get(p, 0) < 5 for p in POPS): continue
        formula = model_formula(variable, is_reduced)
        try:
            fit, model_type, fallback, variance = _attempt_model(frame, formula)
            params = fit.fe_params if model_type == "Linear mixed model" else fit.params
            covariance = fit.cov_params().loc[params.index, params.index]
            answer.update(fit=fit, rows=extract_trajectory_estimates(params, covariance), model_type=model_type,
                          model_specification="reduced_adjustment" if is_reduced else "full_adjustment", formula=formula,
                          model_status=("fitted_gee_fallback" if fallback else ("fitted_reduced_adjustment" if is_reduced else "fitted")),
                          converged=True, random_intercept_variance=variance, fallback_used=fallback)
            return answer
        except Exception as exc: errors.append(str(exc))
    answer.update(model_status="failed", warning="; ".join(errors)[-500:]); return answer


def _standardization(data: pd.DataFrame, spec: Mapping[str, Any]) -> Tuple[float, str]:
    mask = valid_observation_mask(data, spec) & data.baseline_pop.isin(POPS)
    baseline = pd.to_numeric(data.loc[mask & data.is_observed_baseline.fillna(False).astype(bool), spec["variable"]], errors="coerce")
    sd = baseline.std(ddof=1)
    if len(baseline) >= 20 and pd.notna(sd) and sd > 0: return float(sd), "observed_baseline"
    pooled = pd.to_numeric(data.loc[mask, spec["variable"]], errors="coerce").std(ddof=1)
    return (float(pooled), "pooled_valid_visits_fallback") if pd.notna(pooled) and pooled > 0 else (np.nan, "unavailable")


def standardize_estimates(results: pd.DataFrame, specs: Mapping[str, Mapping[str, Any]], sd_info: Mapping[str, Tuple[float, str]]) -> pd.DataFrame:
    rows = []
    for _, r in results.loc[results.estimand_type.eq("annual_slope") & results.estimate.notna()].iterrows():
        sd, source = sd_info[r.outcome_key]
        if not np.isfinite(sd) or sd <= 0: continue
        multiplier = 1 if specs[r.outcome_key]["higher_burden"] else -1
        lo, hi = r.ci95_low / sd * multiplier, r.ci95_high / sd * multiplier
        rows.append({"outcome_key":r.outcome_key,"outcome_label":r.outcome_label,"source_variable":r.source_variable,"baseline_pop":r.baseline_pop,"estimand":r.estimand,"raw_estimate":r.estimate,"outcome_standard_deviation":sd,"standardization_source":source,"direction_multiplier":multiplier,"standardized_adverse_estimate":r.estimate/sd*multiplier,"standardized_standard_error":r.standard_error/sd,"standardized_ci95_low":min(lo,hi),"standardized_ci95_high":max(lo,hi),"p_value":r.p_value,"n_eligible_patients":r.n_eligible_patients,"model_type":r.model_type,"model_status":r.model_status})
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def _bh(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index); valid = values.dropna().sort_values(); m = len(valid)
    if not m: return out
    adjusted = (valid * m / np.arange(1, m + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    out.loc[adjusted.index] = adjusted; return out


def make_integrated_figure(standardized: pd.DataFrame, specs: Mapping[str, Mapping[str, Any]], fatigue_selected: str, path: Path) -> None:
    keys = list(specs); fig, ax = plt.subplots(figsize=(9, 7)); colors = {"Pop1":"#B2182B","Pop2":"#2166AC","Pop3":"#4D4D4D"}; offsets={"Pop1":.22,"Pop2":0,"Pop3":-.22}
    for i, key in enumerate(keys):
        rows = standardized.loc[standardized.outcome_key.eq(key)]
        if rows.empty: ax.text(0, i, "not estimated", color="gray", va="center")
        for pop in POPS:
            r = rows.loc[rows.baseline_pop.eq(pop)]
            if r.empty: continue
            r=r.iloc[0]; ax.errorbar(r.standardized_adverse_estimate, i+offsets[pop], xerr=[[r.standardized_adverse_estimate-r.standardized_ci95_low],[r.standardized_ci95_high-r.standardized_adverse_estimate]], fmt="o", color=colors[pop], label=pop if i==0 else None, capsize=2)
        if not rows.empty: ax.text(ax.get_xlim()[1], i, f" n={int(rows.n_eligible_patients.iloc[0])}", color="gray", fontsize=8, va="center")
    labels=[specs[k]["label"] for k in keys]; ax.set_yticks(range(len(keys))); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.axvline(0,color="black",lw=.8); ax.set_xlabel("Standardized adverse annual change (baseline SD units)"); ax.legend(title="Baseline Pop", loc="best")
    fig.text(.01,.01,"Estimates are standardized using the outcome-specific baseline standard deviation. Positive values consistently indicate increasing burden or\ndeclining health status. Models are adjusted longitudinal associations, not causal effects.\nBaseline Pop is defined using baseline ESSDAI and ESSPRI; trajectories may partly reflect baseline group definition and regression to the mean.\nFatigue measure: "+fatigue_selected, fontsize=8)
    fig.tight_layout(rect=(0,.13,1,1)); fig.savefig(path); plt.close(fig)


def _planned_paths() -> Dict[str, Path]:
    return {"parquet":common.INTERMEDIATE_DATA_DIR/"25_longitudinal_modeling_patient_visit.parquet", "csv":common.INTERMEDIATE_DATA_DIR/"25_longitudinal_modeling_patient_visit.csv", "results":common.BLOCKB_TABLES_DIR/"25_longitudinal_model_results.csv", "standard":common.BLOCKB_TABLES_DIR/"25_standardized_trajectory_estimates.csv", "diagnostics":common.BLOCKB_TABLES_DIR/"25_model_support_and_diagnostics.csv", "figure":common.BLOCKB_FIGURES_DIR/"25_standardized_annual_change_forestplot.pdf", "qc":common.BLOCKB_QC_DIR/"25_longitudinal_multidomain_models_qc.json", "log":common.BLOCKB_LOGS_DIR/"25_longitudinal_multidomain_models.log"}


def main(argv: Optional[Sequence[str]] = None) -> None:
    args=parse_args(argv); paths=_planned_paths()
    if not args.overwrite:
        existing=[str(p) for p in paths.values() if p.exists()]
        if existing: raise FileExistsError("Planned outputs already exist: " + ", ".join(existing))
    for p in paths.values(): p.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=paths["log"], level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    visits, baseline=load_inputs(args.input,args.baseline_input); visits=validate_longitudinal_data(visits); n_input=len(visits)
    data=merge_baseline_covariates(visits,baseline)
    selected,n_mdafs,n_profad,reason=select_fatigue_outcome(data,args.fatigue_outcome,args.min_followup_days)
    specs=dict(OUTCOME_SPECS); specs["fatigue"] = dict(FATIGUE_SPECS[selected])
    # Stable clinical display order.
    specs={k:specs[k] for k in ("essdai","esspri","pcs","mcs","fatigue","active_domains")}
    data,supports=add_eligibility_flags(data,specs,args.min_followup_days)
    result_rows=[]; diagnostics=[]; statuses={}; sd_info={}
    for key,spec in specs.items():
        support=supports[key]; valid=valid_observation_mask(data,spec); eligible_ids=set(data.loc[data[f"eligible_{key}_model"],"patient_id"]); eligible=data.patient_id.isin(eligible_ids)
        fit=fit_longitudinal_model(data,key,spec,args.min_patients,args.skip_models); statuses[key]=fit["model_status"]
        sd_info[key]=_standardization(data,spec)
        patient_pop=data.loc[eligible,["patient_id","baseline_pop"]].drop_duplicates().baseline_pop.value_counts()
        spans=support.loc[support.index.isin(eligible_ids),"followup_days"] if len(support) else pd.Series(dtype=float)
        stats={"n_valid_visits":int((valid&eligible).sum()),"n_eligible_patients":len(eligible_ids), **{f"n_{p.lower()}_patients":int(patient_pop.get(p,0)) for p in POPS}, "median_valid_visits_per_patient":float(support.loc[support.index.isin(eligible_ids),"n_valid_visits"].median()) if eligible_ids else np.nan,"median_followup_years":float(spans.median()/365.25) if len(spans) else np.nan}
        interpretation="Adjusted longitudinal association; positive raw slope = " + ("increasing measured burden." if spec["higher_burden"] else "improving score and negative raw slope = declining score.")
        estimands = fit["rows"] or [
            *[{"estimand_type":"annual_slope", "estimand":f"{p} annual slope", "baseline_pop":p,
               "reference_pop":"Pop3", "contrast":"", **{x:np.nan for x in ("estimate","standard_error","ci95_low","ci95_high","p_value")}} for p in POPS],
            *[{"estimand_type":"slope_difference", "estimand":f"{p} minus Pop3 annual slope difference", "baseline_pop":p,
               "reference_pop":"Pop3", "contrast":f"{p} - Pop3", **{x:np.nan for x in ("estimate","standard_error","ci95_low","ci95_high","p_value")}} for p in ("Pop1","Pop2")],
        ]
        for row in estimands: result_rows.append({"outcome_key":key,"outcome_label":spec["label"],"source_variable":spec["variable"],"fatigue_measure_used":selected if key=="fatigue" else "","outcome_unit":spec["unit"],"model_type":fit["model_type"],"model_specification":fit["model_specification"],"model_status":fit["model_status"],"formula":fit["formula"],**row,"fdr_bh_q_value":np.nan,**stats,"interpretation":interpretation})
        diagnostics.append({"outcome_key":key,"source_variable":spec["variable"],"availability_flag":spec["availability"],"n_input_visits":n_input,"n_valid_visits":int(valid.sum()),"n_patients_with_any_valid_visit":len(support),"n_patients_with_two_or_more_valid_visits":int(support.n_valid_visits.ge(2).sum()) if len(support) else 0,"n_patients_meeting_followup_gap":int(support.meets_visit_rule.sum()) if len(support) else 0,**stats,"median_valid_visits_per_patient":stats["median_valid_visits_per_patient"],"median_followup_years":stats["median_followup_years"],"maximum_followup_years":float(spans.max()/365.25) if len(spans) else np.nan,"outcome_baseline_sd":sd_info[key][0],"standardization_source":sd_info[key][1],"model_type":fit["model_type"],"model_specification":fit["model_specification"],"model_status":fit["model_status"],"converged":fit["converged"],"random_intercept_variance":fit["random_intercept_variance"],"fallback_used":fit["fallback_used"],"fatigue_measure_requested":args.fatigue_outcome,"fatigue_measure_selected":selected,"selection_reason":reason,"warning":fit["warning"]})
        logging.info("%s eligible patients=%d status=%s",key,len(eligible_ids),fit["model_status"])
    results=pd.DataFrame(result_rows,columns=RESULT_COLUMNS)
    if len(results):
        mask=results.estimand_type.eq("slope_difference"); results.loc[mask,"fdr_bh_q_value"]=_bh(results.loc[mask,"p_value"])
    standardized=standardize_estimates(results,specs,sd_info)
    diagnostics_df=pd.DataFrame(diagnostics,columns=DIAGNOSTIC_COLUMNS)
    keep=REQUIRED+["baseline_pop","age_baseline","sex","protocol_group","baseline_time_since_diagnosis_years"]+[c for c in ["essdai_total","esspri_total","sf36_pcs","sf36_mcs","profad_total","mdafs_global","n_extraglandular_domains_active"] if c in data]
    keep += [c for c in data if c.startswith("avail_") and c in {s["availability"] for s in list(OUTCOME_SPECS.values())+list(FATIGUE_SPECS.values())}]
    keep += [c for c in data if c.startswith("eligible_") or c.startswith("n_valid_") or (c.startswith("valid_") and c.endswith("_followup_days"))]
    model_ready=data[list(dict.fromkeys(keep))]
    model_ready.to_parquet(paths["parquet"],index=False); model_ready.to_csv(paths["csv"],index=False); results.to_csv(paths["results"],index=False); standardized.to_csv(paths["standard"],index=False); diagnostics_df.to_csv(paths["diagnostics"],index=False)
    make_integrated_figure(standardized,specs,selected,paths["figure"])
    fitted=sum(str(s).startswith("fitted") for s in statuses.values()); failed=sum(s=="failed" for s in statuses.values())
    qc={"input_path":str(args.input),"baseline_input_path":str(args.baseline_input),"n_input_visits":n_input,"n_input_patients":int(visits.patient_id.nunique()),"n_baseline_patients":int(baseline.patient_id.nunique()),"n_visits_after_merge":len(data),"n_duplicate_patient_visit_keys":0,"n_negative_followup_times":0,"n_patients_without_baseline_covariates":int(data.loc[data.age_baseline.isna(),"patient_id"].nunique()),"n_unclassifiable_baseline_pop":int(data.loc[~data.baseline_pop.isin(POPS),"patient_id"].nunique()),"min_followup_days":args.min_followup_days,"min_patients":args.min_patients,"fatigue_measure_requested":args.fatigue_outcome,"fatigue_measure_selected":selected,"n_eligible_mdafs":n_mdafs,"n_eligible_profad":n_profad,"fatigue_selection_reason":reason,"n_models_requested":6,"n_models_fitted":fitted,"n_models_using_reduced_adjustment":sum(d["model_specification"]=="reduced_adjustment" and str(d["model_status"]).startswith("fitted") for d in diagnostics),"n_models_using_gee_fallback":sum(bool(d["fallback_used"]) for d in diagnostics),"n_models_skipped":6-fitted-failed,"n_models_failed":failed,"outcome_model_status":statuses,"missing_optional_columns":[c for s in specs.values() for c in (s["variable"],s["availability"]) if c not in data],"warnings":[d["warning"] for d in diagnostics if d["warning"]]}
    paths["qc"].write_text(json.dumps(qc,indent=2,default=str)+"\n"); logging.info("Outputs written; fatigue=%s",selected)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare leak-free visit states and preserve canonical episode adjacency."""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml
import common
from src.studies._shared import (STUDY_CONTRACT_VERSION, create_study_dirs, load_parquet,
    resolve_concept, sha256_file, validate_integrated_dataset, validate_transition_intervals, write_json)

STRUCTURAL = {"patient_id", "clinical_episode_id", "clinical_anchor_date", "clinical_visit_number",
              "clinical_visit", "is_clinical_baseline"}
TIME_NAMES = {"interval_days", "interval_years", "age_at_visit", "demo__age_at_visit",
              "ids__age_at_visit", "visit_order", "time_since_baseline", "time_since_diagnosis"}
STATIC_TOKENS = ("sex", "race", "ethnicity", "protocol", "age_at_diagnosis", "age_dx")
EXTERNAL_CONCEPTS = ("essdai_total", "esspri_total", "pop")
EXTERNAL_PREFIXES = ("essdai__", "esspri__", "pop__", "essdai_", "esspri_", "pop_")
HISTORY_MARKERS = ("ever_positive", "through_episode", "cumulative", "history_through",
                   "history_to_date", "prior_", "past_", "previous_")
BASELINE_MARKERS = ("baseline_", "__baseline_", "baseline__")

def load_config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict): raise ValueError("Longitudinal Graph config must be a mapping")
    return value

def forbidden_reason(column: str, external: set[str], patterns: list[str]) -> str:
    low = column.lower()
    if column in STRUCTURAL or low in TIME_NAMES or low.startswith("time_since_"): return "administrative"
    if column in external or low.startswith(EXTERNAL_PREFIXES): return "external_characterizer"
    if any(marker in low for marker in HISTORY_MARKERS): return "historical_accumulator"
    if any(marker in low for marker in BASELINE_MARKERS): return "baseline_static_copy"
    if any(x in low for x in patterns): return "future_leakage"
    if any(x in low for x in STATIC_TOKENS): return "static_covariate"
    if low.startswith(("admin__", "ids__")): return "administrative"
    return ""

def temporal_scope(column: str, registry: pd.DataFrame | None = None) -> tuple[str, str]:
    """Resolve scope conservatively; unknown variables fail closed."""
    if registry is not None and not registry.empty:
        key = "public_variable" if "public_variable" in registry else "feature"
        match = registry.loc[registry[key].eq(column)] if key in registry else pd.DataFrame()
        if len(match) and "temporal_scope" in match:
            raw = str(match.iloc[0].temporal_scope).strip().lower()
            mapped = {"episode":"episode_resolved", "clinical_episode":"episode_resolved",
                      "clinical_baseline":"baseline_static", "baseline":"baseline_static",
                      "through_episode":"cumulative_history", "as_of_episode":"cumulative_history",
                      "patient_retrospective":"cumulative_history"}.get(raw, raw)
            if mapped in {"episode_resolved","baseline_static","patient_static","cumulative_history",
                          "external_outcome","administrative","unknown"}:
                return mapped, "step10_registry"
    low=column.lower()
    if column in STRUCTURAL or low.startswith(("spine__","admin__","ids__")): return "administrative", "deterministic_name_rule"
    if low.startswith(EXTERNAL_PREFIXES): return "external_outcome", "deterministic_name_rule"
    if any(x in low for x in HISTORY_MARKERS): return "cumulative_history", "deterministic_name_rule"
    if any(x in low for x in BASELINE_MARKERS): return "baseline_static", "deterministic_name_rule"
    if any(x in low for x in STATIC_TOKENS): return "patient_static", "deterministic_name_rule"
    # These are the public namespaces whose Step-10 contract defines row-level measurements.
    if low.startswith(("lab__","pro__","ext__","ovl__")): return "episode_resolved", "upstream_public_contract"
    return "unknown", "unresolved"

def build_consecutive_intervals(master: pd.DataFrame) -> pd.DataFrame:
    order = master.assign(clinical_anchor_date=pd.to_datetime(master.clinical_anchor_date)).sort_values(
        ["patient_id", "clinical_anchor_date", "clinical_visit_number", "clinical_episode_id"])
    rows=[]
    for patient, group in order.groupby("patient_id", sort=False):
        records=group.to_dict("records")
        for left,right in zip(records[:-1], records[1:]):
            days=(right["clinical_anchor_date"]-left["clinical_anchor_date"]).days
            rows.append({"patient_id":patient,"from_clinical_episode_id":left["clinical_episode_id"],
                "to_clinical_episode_id":right["clinical_episode_id"],
                "from_visit_number":left["clinical_visit_number"],"to_visit_number":right["clinical_visit_number"],
                "interval_days":days,"interval_years":days/365.25})
    return pd.DataFrame(rows, columns=["patient_id","from_clinical_episode_id","to_clinical_episode_id",
        "from_visit_number","to_visit_number","interval_days","interval_years"])

def feature_manifest(master, intervals, config, registry=None):
    fs=config["feature_selection"]; patterns=[str(x).lower() for x in fs["forbidden_name_patterns"]]
    external=set()
    for concept in EXTERNAL_CONCEPTS:
        try: external.add(resolve_concept(master, concept))
        except ValueError: pass
    nvis=len(master); npat=master.patient_id.nunique(); rows=[]; values={}
    ends=pd.concat([intervals[["patient_id","from_clinical_episode_id"]].rename(columns={"from_clinical_episode_id":"clinical_episode_id"}),
                    intervals[["patient_id","to_clinical_episode_id"]].rename(columns={"to_clinical_episode_id":"clinical_episode_id"})]).drop_duplicates()
    for column in master.columns:
        reason=forbidden_reason(column, external, patterns)
        scope,scope_source=temporal_scope(column,registry)
        if not reason and fs.get("require_longitudinally_resolved",False) and scope != "episode_resolved":
            reason="unresolved_temporal_scope" if scope=="unknown" else f"not_episode_resolved:{scope}"
        numeric=pd.to_numeric(master[column], errors="coerce")
        observed=master[column].notna(); numeric_observed=numeric.notna()
        patient_obs=master.loc[observed,"patient_id"].nunique()
        both=0
        if observed.any() and len(intervals):
            seen=set(map(tuple,master.loc[observed,["patient_id","clinical_episode_id"]].to_numpy()))
            both=sum((r.patient_id,r.from_clinical_episode_id) in seen and (r.patient_id,r.to_clinical_episode_id) in seen for r in intervals.itertuples())
        if not reason and not pd.api.types.is_numeric_dtype(master[column]): reason="unsupported_encoding"
        if not reason and numeric.nunique(dropna=True)<2: reason="no_variability"
        if not reason and numeric_observed.mean()<float(fs["minimum_visit_coverage"]): reason="below_visit_coverage"
        if not reason and patient_obs/max(npat,1)<float(fs["minimum_patient_coverage"]): reason="below_patient_coverage"
        included=not reason
        if included: values[column]=numeric.astype(float)
        rows.append({"feature":column,"family":column.split("__",1)[0],"source":"integrated_master",
          "public_column":column,"temporal_scope":scope,"temporal_scope_source":scope_source,"n_visits_observed":int(observed.sum()),
          "pct_visits_observed":float(observed.mean()),"n_patients_observed":int(patient_obs),
          "pct_patients_observed":float(patient_obs/max(npat,1)),"n_intervals_observed_both_ends":int(both),
          "pct_intervals_observed_both_ends":float(both/max(len(intervals),1)),"n_unique_values":int(numeric.nunique(dropna=True)),
          "included":included,"exclusion_reason":reason,
          "is_external_characterizer":reason=="external_characterizer","is_historical_accumulator":reason=="historical_accumulator",
          "is_baseline_static":reason=="baseline_static_copy","is_episode_resolved":scope=="episode_resolved"})
    return pd.DataFrame(rows), pd.DataFrame(values,index=master.index)

def visit_type_coverage(master, features, visit_type, included_features=None):
    columns=["visit_type","feature","n_visits","n_observed","pct_observed","n_patients",
             "n_patients_observed","pct_patients_observed","included_feature"]
    if not visit_type: return pd.DataFrame(columns=columns),pd.DataFrame(columns=["visit_type","n_visits","n_patients","median_feature_coverage","q1_feature_coverage","q3_feature_coverage"])
    rows=[]; included=set(features if included_features is None else included_features)
    for kind,group in master.groupby(visit_type,dropna=False):
        for feature in features:
            obs=group[feature].notna(); npat=group.patient_id.nunique(); patient_obs=group.loc[obs,"patient_id"].nunique()
            rows.append({"visit_type":kind,"feature":feature,"n_visits":len(group),"n_observed":int(obs.sum()),
              "pct_observed":float(obs.mean()),"n_patients":npat,"n_patients_observed":patient_obs,
              "pct_patients_observed":float(patient_obs/max(npat,1)),"included_feature":feature in included})
    detail=pd.DataFrame(rows,columns=columns); summary=[]
    for kind,group in master.groupby(visit_type,dropna=False):
        selected=[f for f in features if f in included]
        coverage=group[selected].notna().mean(axis=1) if selected else pd.Series(np.nan,index=group.index)
        summary.append({"visit_type":kind,"n_visits":len(group),"n_patients":group.patient_id.nunique(),
          "median_feature_coverage":coverage.median(),"q1_feature_coverage":coverage.quantile(.25),"q3_feature_coverage":coverage.quantile(.75)})
    return detail,pd.DataFrame(summary)

def preimputation_coverage(state: pd.DataFrame) -> pd.DataFrame:
    """Per-row observability, deliberately computed without filling missing values."""
    total=state.shape[1]
    observed=state.notna().sum(axis=1)
    return pd.DataFrame({"state_features_observed":observed,"state_features_total":total,
                         "state_coverage_fraction":observed/total},index=state.index)

def parse_args(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--integrated",type=Path,default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    p.add_argument("--intervals",type=Path,default=common.POP_TRANSITION_INTERVALS_PARQUET)
    p.add_argument("--config",type=Path,default=Path(__file__).with_name("config.yaml")); p.add_argument("--dry-run",action="store_true")
    return p.parse_args(argv)

def run(args):
    dirs=create_study_dirs("longitudinal_graph"); logging.basicConfig(filename=dirs["logs"]/"01_prepare.log",level=logging.INFO,force=True)
    cfg=load_config(args.config); master=load_parquet(args.integrated); contract=validate_integrated_dataset(master)
    canonical=load_parquet(args.intervals) if args.intervals.exists() else None
    intervals=build_consecutive_intervals(master)
    if canonical is not None:
        validate_transition_intervals(canonical,master)
        base_cols=["patient_id","from_clinical_episode_id","to_clinical_episode_id","interval_days","interval_years"]
        intervals=canonical[base_cols].merge(intervals.drop(columns=["interval_days","interval_years"]),on=base_cols[:3],validate="one_to_one")
    registry_path=args.integrated.with_name("10_variable_registry.csv")
    registry=pd.read_csv(registry_path) if registry_path.exists() else None
    manifest,state=feature_manifest(master,intervals,cfg,registry)
    included=manifest.loc[manifest.included,"feature"].tolist()
    if not included: raise ValueError("No eligible longitudinal state features")
    state=state[included]; coverage_qc=preimputation_coverage(state)
    observed_counts=coverage_qc.state_features_observed; coverage=coverage_qc.state_coverage_fraction
    imputed=int(state.isna().sum().sum()); total_values=len(state)*len(included); state=state.fillna(state.median())
    keys=master[["patient_id","clinical_episode_id"]].reset_index(drop=True); state=pd.concat([keys,state.reset_index(drop=True)],axis=1)
    metadata=master.drop(columns=included).copy(); metadata["state_features_observed"]=observed_counts
    metadata["state_features_total"]=len(included); metadata["state_coverage_fraction"]=coverage
    cover_lookup=dict(zip(map(tuple,keys.to_numpy()),coverage))
    intervals["from_state_coverage"]=[cover_lookup.get((r.patient_id,r.from_clinical_episode_id),np.nan) for r in intervals.itertuples()]
    intervals["to_state_coverage"]=[cover_lookup.get((r.patient_id,r.to_clinical_episode_id),np.nan) for r in intervals.itertuples()]
    intervals["min_endpoint_coverage"]=intervals[["from_state_coverage","to_state_coverage"]].min(axis=1)
    intervals["mean_endpoint_coverage"]=intervals[["from_state_coverage","to_state_coverage"]].mean(axis=1)
    counts=master.groupby("patient_id").size(); positive=intervals.interval_days.gt(0)
    endpoint=set(map(tuple,state[["patient_id","clinical_episode_id"]].to_numpy()))
    both=np.array([(r.patient_id,r.from_clinical_episode_id) in endpoint and (r.patient_id,r.to_clinical_episode_id) in endpoint for r in intervals.itertuples()])
    summary={"n_patients_master":master.patient_id.nunique(),"n_clinical_visits_master":len(master),"n_patients_1_visit":int(counts.eq(1).sum()),
      "n_patients_ge2_visits":int(counts.ge(2).sum()),"n_patients_ge3_visits":int(counts.ge(3).sum()),"n_eligible_visits":len(state),
      "n_eligible_patients":state.patient_id.nunique(),"n_adjacent_intervals_total":len(intervals),"n_positive_time_intervals":int(positive.sum()),
      "n_intervals_with_state_features_both_ends":int((positive&both).sum()),"median_visits_per_patient":counts.median(),"q1_visits_per_patient":counts.quantile(.25),
      "q3_visits_per_patient":counts.quantile(.75),"median_interval_days":intervals.interval_days.median(),"q1_interval_days":intervals.interval_days.quantile(.25),
      "q3_interval_days":intervals.interval_days.quantile(.75),"min_interval_days":intervals.interval_days.min(),"max_interval_days":intervals.interval_days.max(),
      "n_features_candidate":len(manifest),"n_features_included":len(included),"fraction_imputed_values":imputed/total_values,
      "pct_imputed_values":100*imputed/total_values,"median_visit_state_coverage":coverage.median(),
      "q1_visit_state_coverage":coverage.quantile(.25),"q3_visit_state_coverage":coverage.quantile(.75),
      "median_interval_min_endpoint_coverage":intervals.min_endpoint_coverage.median(),
      "q1_interval_min_endpoint_coverage":intervals.min_endpoint_coverage.quantile(.25),
      "q3_interval_min_endpoint_coverage":intervals.min_endpoint_coverage.quantile(.75),
      "feasibility_status":"viable_descriptive" if counts.ge(2).sum()>=8 and (positive&both).sum()>=8 else "sparse_exploratory"}
    state.to_parquet(dirs["analytic"]/"01_longitudinal_visit_state.parquet",index=False); metadata.to_parquet(dirs["analytic"]/"01_longitudinal_visit_metadata.parquet",index=False)
    intervals.to_parquet(dirs["analytic"]/"01_longitudinal_intervals.parquet",index=False); manifest.to_csv(dirs["tables"]/"01_longitudinal_feature_manifest.csv",index=False)
    pd.DataFrame([summary]).to_csv(dirs["tables"]/"01_longitudinal_feasibility_summary.csv",index=False)
    visit_type=next((x for x in ("spine__interval_name","interval_name","visit_type") if x in master),None)
    visit_detail,visit_summary=visit_type_coverage(master,manifest.feature.tolist(),visit_type,included)
    visit_detail.to_csv(dirs["tables"]/"01_longitudinal_visit_type_coverage.csv",index=False)
    visit_summary.to_csv(dirs["tables"]/"01_longitudinal_visit_type_summary.csv",index=False)
    intervals.interval_days.describe().rename_axis("statistic").reset_index(name="interval_days").to_csv(dirs["tables"]/"01_longitudinal_interval_distribution.csv",index=False)
    write_json(dirs["qc"]/"01_longitudinal_data_qc.json",{"status":"completed","contract_version":STUDY_CONTRACT_VERSION,"input_files":[args.integrated,args.intervals],"input_sha256":sha256_file(args.integrated),"config_file":args.config,"config_sha256":sha256_file(args.config),"random_seed":cfg["random_seed"],"contract":contract,**summary})
    print("ORDER TO REVIEW LONGITUDINAL GRAPH OUTPUTS\n1. Feasibility summary\n2. Feature manifest\n3. Data QC")
    return summary
if __name__ == "__main__": run(parse_args())

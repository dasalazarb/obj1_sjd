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

def load_config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict): raise ValueError("Longitudinal Graph config must be a mapping")
    return value

def forbidden_reason(column: str, external: set[str], patterns: list[str]) -> str:
    low = column.lower()
    if column in STRUCTURAL or low in TIME_NAMES or low.startswith("time_since_"): return "administrative"
    if column in external: return "external_characterizer"
    if any(x in low for x in patterns): return "future_leakage"
    if any(x in low for x in STATIC_TOKENS): return "static_covariate"
    if low.startswith(("admin__", "ids__")): return "administrative"
    return ""

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

def feature_manifest(master, intervals, config):
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
        numeric=pd.to_numeric(master[column], errors="coerce") if not reason else pd.Series(np.nan,index=master.index)
        observed=numeric.notna(); patient_obs=master.loc[observed,"patient_id"].nunique()
        both=0
        if observed.any() and len(intervals):
            seen=set(map(tuple,master.loc[observed,["patient_id","clinical_episode_id"]].to_numpy()))
            both=sum((r.patient_id,r.from_clinical_episode_id) in seen and (r.patient_id,r.to_clinical_episode_id) in seen for r in intervals.itertuples())
        if not reason and not pd.api.types.is_numeric_dtype(master[column]): reason="unsupported_encoding"
        if not reason and numeric.nunique(dropna=True)<2: reason="no_variability"
        if not reason and observed.mean()<float(fs["minimum_visit_coverage"]): reason="below_visit_coverage"
        if not reason and patient_obs/max(npat,1)<float(fs["minimum_patient_coverage"]): reason="below_patient_coverage"
        included=not reason
        if included: values[column]=numeric.astype(float)
        rows.append({"feature":column,"family":column.split("__",1)[0],"source":"integrated_master",
          "public_column":column,"temporal_scope":"clinical_episode","n_visits_observed":int(observed.sum()),
          "pct_visits_observed":float(observed.mean()),"n_patients_observed":int(patient_obs),
          "pct_patients_observed":float(patient_obs/max(npat,1)),"n_intervals_observed_both_ends":int(both),
          "pct_intervals_observed_both_ends":float(both/max(len(intervals),1)),"n_unique_values":int(numeric.nunique(dropna=True)),
          "included":included,"exclusion_reason":reason})
    return pd.DataFrame(rows), pd.DataFrame(values,index=master.index)

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
    manifest,state=feature_manifest(master,intervals,cfg)
    included=manifest.loc[manifest.included,"feature"].tolist()
    if not included: raise ValueError("No eligible longitudinal state features")
    state=state[included]; imputed=int(state.isna().sum().sum()); state=state.fillna(state.median())
    keys=master[["patient_id","clinical_episode_id"]].reset_index(drop=True); state=pd.concat([keys,state.reset_index(drop=True)],axis=1)
    metadata=master.drop(columns=included).copy()
    counts=master.groupby("patient_id").size(); positive=intervals.interval_days.gt(0)
    endpoint=set(map(tuple,state[["patient_id","clinical_episode_id"]].to_numpy()))
    both=np.array([(r.patient_id,r.from_clinical_episode_id) in endpoint and (r.patient_id,r.to_clinical_episode_id) in endpoint for r in intervals.itertuples()])
    summary={"n_patients_master":master.patient_id.nunique(),"n_clinical_visits_master":len(master),"n_patients_1_visit":int(counts.eq(1).sum()),
      "n_patients_ge2_visits":int(counts.ge(2).sum()),"n_patients_ge3_visits":int(counts.ge(3).sum()),"n_eligible_visits":len(state),
      "n_eligible_patients":state.patient_id.nunique(),"n_adjacent_intervals_total":len(intervals),"n_positive_time_intervals":int(positive.sum()),
      "n_intervals_with_state_features_both_ends":int((positive&both).sum()),"median_visits_per_patient":counts.median(),"q1_visits_per_patient":counts.quantile(.25),
      "q3_visits_per_patient":counts.quantile(.75),"median_interval_days":intervals.interval_days.median(),"q1_interval_days":intervals.interval_days.quantile(.25),
      "q3_interval_days":intervals.interval_days.quantile(.75),"min_interval_days":intervals.interval_days.min(),"max_interval_days":intervals.interval_days.max(),
      "n_features_candidate":len(manifest),"n_features_included":len(included),"pct_imputed_values":imputed/(len(state)*len(included)),
      "feasibility_status":"viable_descriptive" if counts.ge(2).sum()>=8 and (positive&both).sum()>=8 else "sparse_exploratory"}
    state.to_parquet(dirs["analytic"]/"01_longitudinal_visit_state.parquet",index=False); metadata.to_parquet(dirs["analytic"]/"01_longitudinal_visit_metadata.parquet",index=False)
    intervals.to_parquet(dirs["analytic"]/"01_longitudinal_intervals.parquet",index=False); manifest.to_csv(dirs["tables"]/"01_longitudinal_feature_manifest.csv",index=False)
    pd.DataFrame([summary]).to_csv(dirs["tables"]/"01_longitudinal_feasibility_summary.csv",index=False)
    visit_type=next((x for x in ("spine__interval_name","interval_name","visit_type") if x in master),None)
    (master.groupby(visit_type).agg(n_visits=("patient_id","size"),n_patients=("patient_id","nunique")).reset_index() if visit_type else pd.DataFrame(columns=["visit_type","n_visits","n_patients"])).to_csv(dirs["tables"]/"01_longitudinal_visit_type_coverage.csv",index=False)
    intervals.interval_days.describe().rename_axis("statistic").reset_index(name="interval_days").to_csv(dirs["tables"]/"01_longitudinal_interval_distribution.csv",index=False)
    write_json(dirs["qc"]/"01_longitudinal_data_qc.json",{"status":"completed","contract_version":STUDY_CONTRACT_VERSION,"input_files":[args.integrated,args.intervals],"input_sha256":sha256_file(args.integrated),"config_file":args.config,"config_sha256":sha256_file(args.config),"random_seed":cfg["random_seed"],"contract":contract,**summary})
    print("ORDER TO REVIEW LONGITUDINAL GRAPH OUTPUTS\n1. Feasibility summary\n2. Feature manifest\n3. Data QC")
    return summary
if __name__ == "__main__": run(parse_args())

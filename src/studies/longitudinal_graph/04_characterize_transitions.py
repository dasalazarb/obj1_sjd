#!/usr/bin/env python3
"""Characterize clinical changes accompanying supported observed movement."""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import common
from src.studies._shared import STUDY_CONTRACT_VERSION, create_study_dirs, load_parquet, resolve_concept, sha256_file, validate_integrated_dataset, write_json

def load_config(path): return yaml.safe_load(Path(path).read_text())
def attach_clinical_endpoints(transitions,master,columns):
    left=master[["patient_id","clinical_episode_id",*columns]].rename(columns={"clinical_episode_id":"from_clinical_episode_id",**{c:f"from__{c}" for c in columns}})
    right=master[["patient_id","clinical_episode_id",*columns]].rename(columns={"clinical_episode_id":"to_clinical_episode_id",**{c:f"to__{c}" for c in columns}})
    out=transitions.merge(left,on=["patient_id","from_clinical_episode_id"],how="left",validate="many_to_one").merge(right,on=["patient_id","to_clinical_episode_id"],how="left",validate="many_to_one")
    if len(out)!=len(transitions): raise AssertionError("Clinical endpoint join changed interval count")
    return out

def transition_label(frame):
    a=frame["from__hard_macrostate"]; b=frame["to__hard_macrostate"]
    return np.where(a.notna()&b.notna(),a.astype("Int64").astype(str)+" -> "+b.astype("Int64").astype(str),"unassigned")

def patient_bootstrap_median_delta(group,delta,replicates,seed):
    observed=group.dropna(subset=[delta]); patients=np.array(sorted(observed.patient_id.unique(),key=str)); rng=np.random.default_rng(seed); estimates=[]
    for _ in range(replicates):
        draw=rng.choice(patients,len(patients),replace=True); values=pd.concat([observed.loc[observed.patient_id.eq(p),delta] for p in draw],ignore_index=True); estimates.append(values.median())
    return (np.nan,np.nan) if not estimates else tuple(np.nanquantile(estimates,[.025,.975]))

def characterize_continuous(frame,variables,roles,replicates,seed,minimum_patients):
    rows=[]
    for variable in variables:
      f,t=f"from__{variable}",f"to__{variable}"; frame[f]=pd.to_numeric(frame[f],errors="coerce"); frame[t]=pd.to_numeric(frame[t],errors="coerce"); delta=f"delta__{variable}"; frame[delta]=frame[t]-frame[f]
      for label,g in frame.groupby("transition_group"):
        valid=g.dropna(subset=[f,t]); n_pat=valid.patient_id.nunique(); lo,hi=patient_bootstrap_median_delta(valid,delta,replicates,seed)
        rows.append({"transition_group":label,"variable":variable,"role":roles[variable],"n_intervals":len(valid),"n_unique_patients":n_pat,"median_from":valid[f].median(),"median_to":valid[t].median(),"median_delta":valid[delta].median(),"q1_delta":valid[delta].quantile(.25),"q3_delta":valid[delta].quantile(.75),"patient_bootstrap_ci95_low":lo,"patient_bootstrap_ci95_high":hi,"support_status":"supported" if n_pat>=minimum_patients else "sparse_descriptive"})
    return pd.DataFrame(rows)

def characterize_categorical(frame,variables,roles):
    rows=[]
    for variable in variables:
      f,t=f"from__{variable}",f"to__{variable}"; valid=frame.dropna(subset=[f,t])
      for (label,a,b),g in valid.groupby(["transition_group",f,t],dropna=False): rows.append({"transition_group":label,"variable":variable,"role":roles[variable],"from_category":a,"to_category":b,"n_intervals":len(g),"n_unique_patients":g.patient_id.nunique()})
    return pd.DataFrame(rows)

def parse_args(argv=None):
    p=argparse.ArgumentParser(); d=create_study_dirs("longitudinal_graph"); p.add_argument("--integrated",type=Path,default=common.INTEGRATED_LONGITUDINAL_PARQUET); p.add_argument("--transitions",type=Path,default=d["analytic"]/"03_patient_temporal_transitions.parquet"); p.add_argument("--manifest",type=Path,default=d["tables"]/"01_longitudinal_feature_manifest.csv"); p.add_argument("--config",type=Path,default=Path(__file__).with_name("config.yaml")); p.add_argument("--dry-run",action="store_true"); return p.parse_args(argv)

def run(args):
    d=create_study_dirs("longitudinal_graph"); logging.basicConfig(filename=d["logs"]/"04_characterize.log",level=logging.INFO,force=True); cfg=load_config(args.config); master=load_parquet(args.integrated); contract=validate_integrated_dataset(master); transitions=pd.read_parquet(args.transitions); manifest=pd.read_csv(args.manifest)
    defining=manifest.loc[manifest.included.astype(bool),"feature"].tolist(); external=[]
    for concept in ("essdai_total","esspri_total","pop"):
      try: external.append(resolve_concept(master,concept))
      except ValueError: pass
    admin=[c for c in ("demo__protocol","protocol","spine__interval_name","interval_name","visit_type") if c in master]
    columns=list(dict.fromkeys([c for c in defining+external+admin if c in master])); roles={c:("defining_feature" if c in defining else "external_characterizer" if c in external else "administrative_qc") for c in columns}
    frame=attach_clinical_endpoints(transitions,master,columns); frame=frame.loc[frame.flow_eligible].copy(); frame["transition_group"]=transition_label(frame)
    numeric=[c for c in columns if pd.api.types.is_numeric_dtype(master[c])]; categorical=[c for c in columns if c not in numeric]; reps=10 if args.dry_run else int(cfg["characterization"]["bootstrap_replicates"]); threshold=int(cfg["characterization"]["minimum_transition_patients"])
    continuous=characterize_continuous(frame,numeric,roles,reps,cfg["random_seed"],threshold); categorical_changes=characterize_categorical(frame,categorical,roles)
    external_table=continuous.loc[continuous.role.eq("external_characterizer")].copy(); pop_col=next((c for c in external if "pop" in c.lower()),None); pop=categorical_changes.loc[categorical_changes.variable.eq(pop_col)].copy() if pop_col else pd.DataFrame()
    continuous.to_csv(d["tables"]/"04_transition_continuous_changes.csv",index=False); categorical_changes.to_csv(d["tables"]/"04_transition_binary_changes.csv",index=False); external_table.to_csv(d["tables"]/"04_transition_external_outcomes.csv",index=False); pop.to_csv(d["tables"]/"04_transition_pop_crosswalk.csv",index=False)
    clinical=continuous.loc[continuous.n_unique_patients.ge(threshold)].sort_values(["transition_group","role","variable"]); clinical.to_csv(d["tables"]/"04_transition_clinical_summary.csv",index=False)
    status="completed" if len(continuous) or len(categorical_changes) else "sparse"
    summary={"status":status,"contract_version":STUDY_CONTRACT_VERSION,"input_files":[args.integrated,args.transitions,args.manifest],"config_file":args.config,"config_sha256":sha256_file(args.config),"random_seed":cfg["random_seed"],"row_counts":{"eligible_intervals":len(frame),"continuous_results":len(continuous),"categorical_results":len(categorical_changes)},"patient_counts":{"eligible":frame.patient_id.nunique()},"feature_manifest_reference":args.manifest,"thresholds":cfg["characterization"],"n_outputs":6,"roles":{"defining_feature":defining,"external_characterizer":external,"administrative_qc":admin},"contract":contract}
    write_json(d["tables"]/"04_transition_characterization_summary.json",summary)
    plot=external_table.pivot(index="transition_group",columns="variable",values="median_delta") if not external_table.empty else pd.DataFrame()
    if not plot.empty:
      fig,ax=plt.subplots(figsize=(max(6,len(plot.columns)*1.3),max(4,len(plot)*.45))); im=ax.imshow(plot.fillna(0),cmap="coolwarm"); ax.set_xticks(range(len(plot.columns)),plot.columns,rotation=45,ha="right"); ax.set_yticks(range(len(plot)),plot.index); ax.set_title("Held-out outcome changes accompanying movement"); fig.colorbar(im,ax=ax,label="Median change"); fig.tight_layout(); fig.savefig(d["figures"]/"04_transition_external_outcomes.png",dpi=cfg["figures"]["dpi"]); fig.savefig(d["figures"]/"04_transition_delta_heatmap.png",dpi=cfg["figures"]["dpi"]); plt.close(fig)
    print("LONGITUDINAL GRAPH / MODEL A COMPLETE\n1. Feasibility summary\n2. Feature manifest\n3. Reference visit Mapper\n4. Macrostate summary\n5. Temporal flow table\n6. Transition matrix\n7. Net flow/null results\n8. Clinical characterization")
    return summary
if __name__=="__main__": run(parse_args())

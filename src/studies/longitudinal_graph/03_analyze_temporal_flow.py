#!/usr/bin/env python3
"""Overlay consecutive observed visits on the fixed reference Mapper."""
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
from src.studies._shared import STUDY_CONTRACT_VERSION, benjamini_hochberg, create_study_dirs, sha256_file, write_json

KEYS=["patient_id","clinical_episode_id"]
def load_config(path): return yaml.safe_load(Path(path).read_text())
def weight_columns(frame,prefix): return sorted([c for c in frame if c.startswith(prefix)],key=lambda x:str(x))

def validate_memberships(membership):
    for prefix in ("node_weight__","macrostate_weight__"):
        cols=weight_columns(membership,prefix)
        if cols:
            sums=membership[cols].fillna(0).sum(axis=1); covered=sums>0
            if not np.allclose(sums[covered],1,atol=1e-10): raise ValueError(f"{prefix} membership does not sum to one")

def attach_interval_memberships(intervals,membership,maximum_interval_years=None):
    validate_memberships(membership); base=intervals.copy(); base["flow_exclusion_reason"]=""
    left=membership.rename(columns={"clinical_episode_id":"from_clinical_episode_id",**{c:f"from__{c}" for c in membership if c not in KEYS}})
    right=membership.rename(columns={"clinical_episode_id":"to_clinical_episode_id",**{c:f"to__{c}" for c in membership if c not in KEYS}})
    out=base.merge(left,on=["patient_id","from_clinical_episode_id"],how="left",validate="many_to_one").merge(right,on=["patient_id","to_clinical_episode_id"],how="left",validate="many_to_one")
    out.loc[out.interval_days.le(0),"flow_exclusion_reason"]="nonpositive_time"
    out.loc[out["from__mapper_covered"].isna()&out.flow_exclusion_reason.eq(""),"flow_exclusion_reason"]="from_state_missing"
    out.loc[out["to__mapper_covered"].isna()&out.flow_exclusion_reason.eq(""),"flow_exclusion_reason"]="to_state_missing"
    out.loc[out["from__mapper_covered"].eq(False)&out.flow_exclusion_reason.eq(""),"flow_exclusion_reason"]="from_mapper_uncovered"
    out.loc[out["to__mapper_covered"].eq(False)&out.flow_exclusion_reason.eq(""),"flow_exclusion_reason"]="to_mapper_uncovered"
    if maximum_interval_years is not None: out.loc[out.interval_years.gt(maximum_interval_years)&out.flow_exclusion_reason.eq(""),"flow_exclusion_reason"]="maximum_interval_exceeded"
    out["flow_eligible"]=out.flow_exclusion_reason.eq(""); return out

def compute_temporal_flow(attached,prefix="macrostate_weight__"):
    eligible=attached.loc[attached.flow_eligible]; from_cols=sorted(c for c in attached if c.startswith("from__"+prefix)); rows=[]
    for fc in from_cols:
      state_from=fc.split(prefix,1)[1]; tc="to__"+prefix+state_from
      # all destinations, preserving fractional products
      for dest_col in sorted(c for c in attached if c.startswith("to__"+prefix)):
        state_to=dest_col.split(prefix,1)[1]; contributions=eligible[fc].fillna(0)*eligible[dest_col].fillna(0); mask=contributions.gt(0)
        if not mask.any(): continue
        subset=eligible.loc[mask]
        rows.append({"from_state":state_from,"to_state":state_to,"weighted_interval_mass":float(contributions.sum()),
          "n_contributing_intervals":int(mask.sum()),"n_unique_patients":int(subset.patient_id.nunique()),"median_interval_days":subset.interval_days.median(),
          "q1_interval_days":subset.interval_days.quantile(.25),"q3_interval_days":subset.interval_days.quantile(.75),"min_interval_days":subset.interval_days.min(),"max_interval_days":subset.interval_days.max()})
    result=pd.DataFrame(rows)
    expected=len(eligible); actual=result.weighted_interval_mass.sum() if len(result) else 0
    if not np.isclose(actual,expected,atol=1e-8): raise AssertionError(f"Temporal mass {actual} != mapped intervals {expected}")
    return result

def compute_net_flow(flow):
    lookup={(str(r.from_state),str(r.to_state)):r.weighted_interval_mass for r in flow.itertuples()}; rows=[]
    states=sorted(set(flow.from_state)|set(flow.to_state))
    for i,a in enumerate(states):
      for b in states[i+1:]: rows.append({"from_macrostate":a,"to_macrostate":b,"forward_mass":lookup.get((a,b),0),"reverse_mass":lookup.get((b,a),0),"net_flow":lookup.get((a,b),0)-lookup.get((b,a),0)})
    return pd.DataFrame(rows)

def hard_transition_table(attached):
    valid=attached.loc[attached.flow_eligible & attached["from__hard_macrostate"].notna() & attached["to__hard_macrostate"].notna()]
    if valid.empty: return pd.DataFrame(columns=["from_hard_macrostate","to_hard_macrostate","n_intervals","n_unique_patients"])
    return valid.groupby(["from__hard_macrostate","to__hard_macrostate"],dropna=False).agg(n_intervals=("patient_id","size"),n_unique_patients=("patient_id","nunique")).reset_index().rename(columns={"from__hard_macrostate":"from_hard_macrostate","to__hard_macrostate":"to_hard_macrostate"})

def patient_cluster_bootstrap(attached,replicates,seed):
    valid=attached.loc[attached.flow_eligible]; patients=np.array(sorted(valid.patient_id.unique(),key=str)); rng=np.random.default_rng(seed); rows=[]
    for replicate in range(replicates):
        sampled=rng.choice(patients,len(patients),replace=True); chunks=[valid.loc[valid.patient_id.eq(p)] for p in sampled]; sample=pd.concat(chunks,ignore_index=True) if chunks else valid.iloc[0:0]
        hard=sample["from__hard_macrostate"].notna()&sample["to__hard_macrostate"].notna(); stay=(sample.loc[hard,"from__hard_macrostate"].astype(str)==sample.loc[hard,"to__hard_macrostate"].astype(str)).mean() if hard.any() else np.nan
        rows.append({"replicate":replicate,"overall_retention":stay,"overall_switching":1-stay if pd.notna(stay) else np.nan,"n_patient_draws":len(sampled),"n_unique_patients_drawn":len(set(sampled))})
    return pd.DataFrame(rows)

def temporal_reversal_null(attached,replicates,seed):
    valid=attached.loc[attached.flow_eligible].copy(); patients=sorted(valid.patient_id.unique(),key=str); rng=np.random.default_rng(seed); rows=[]
    pairs=compute_net_flow(compute_temporal_flow(valid))
    for rep in range(replicates):
      reverse=set(np.array(patients)[rng.random(len(patients))<.5]); sample=valid.copy(); mask=sample.patient_id.isin(reverse)
      for stem in ("hard_macrostate",)+tuple(c.replace("from__","") for c in sample if c.startswith("from__macrostate_weight__")):
        a,b="from__"+stem,"to__"+stem
        if a in sample and b in sample: sample.loc[mask,[a,b]]=sample.loc[mask,[b,a]].to_numpy()
      null=compute_net_flow(compute_temporal_flow(sample))
      for row in null.itertuples(): rows.append({"replicate":rep,"from_macrostate":row.from_macrostate,"to_macrostate":row.to_macrostate,"net_flow":row.net_flow})
    return pd.DataFrame(rows)

def next_visit_permutation_null(attached,replicates,seed):
    valid=attached.loc[attached.flow_eligible].copy(); rng=np.random.default_rng(seed); rows=[]; destinations=[c for c in valid if c.startswith("to__macrostate_weight__")]
    # Fixed, predeclared quartile bins; undersized bins merge into one global stratum.
    bins=pd.qcut(valid.interval_days,4,duplicates="drop") if len(valid)>=8 else pd.Series("all",index=valid.index)
    for rep in range(replicates):
      sample=valid.copy()
      for _,idx in bins.groupby(bins,observed=True).groups.items():
        idx=np.asarray(list(idx)); perm=rng.permutation(idx); sample.loc[idx,destinations]=valid.loc[perm,destinations].to_numpy()
      null=compute_net_flow(compute_temporal_flow(sample))
      for row in null.itertuples(): rows.append({"replicate":rep,"from_macrostate":row.from_macrostate,"to_macrostate":row.to_macrostate,"net_flow":row.net_flow})
    return pd.DataFrame(rows)

def parse_args(argv=None):
    p=argparse.ArgumentParser(); d=create_study_dirs("longitudinal_graph"); p.add_argument("--integrated",type=Path,default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    p.add_argument("--intervals",type=Path,default=d["analytic"]/"01_longitudinal_intervals.parquet"); p.add_argument("--membership",type=Path,default=d["analytic"]/"02_visit_mapper_membership.parquet"); p.add_argument("--nodes",type=Path,default=d["analytic"]/"02_mapper_node_membership.parquet"); p.add_argument("--config",type=Path,default=Path(__file__).with_name("config.yaml")); p.add_argument("--dry-run",action="store_true"); return p.parse_args(argv)

def run(args):
    d=create_study_dirs("longitudinal_graph"); logging.basicConfig(filename=d["logs"]/"03_flow.log",level=logging.INFO,force=True); cfg=load_config(args.config); intervals=pd.read_parquet(args.intervals); membership=pd.read_parquet(args.membership)
    attached=attach_interval_memberships(intervals,membership,cfg["temporal"]["maximum_interval_years"]); eligible=attached.loc[attached.flow_eligible]
    if eligible.empty: raise ValueError("No temporal intervals map at both endpoints")
    node=compute_temporal_flow(attached,"node_weight__"); macro=compute_temporal_flow(attached); node=node.rename(columns={"from_state":"from_node","to_state":"to_node"}); macro=macro.rename(columns={"from_state":"from_macrostate","to_state":"to_macrostate"})
    hard=hard_transition_table(attached); net=compute_net_flow(macro.rename(columns={"from_macrostate":"from_state","to_macrostate":"to_state"}))
    threshold=int(cfg["temporal"]["minimum_edge_patients_display"]); macro["support_status"]=np.where(macro.n_unique_patients.ge(threshold),"supported","sparse_descriptive")
    hard_valid=eligible.loc[eligible["from__hard_macrostate"].notna()&eligible["to__hard_macrostate"].notna()]; stay=(hard_valid["from__hard_macrostate"].astype(str)==hard_valid["to__hard_macrostate"].astype(str))
    retention=[{"macrostate_id":"overall","hard_retention":stay.mean(),"hard_switching":1-stay.mean(),"n_intervals":len(stay),"n_unique_patients":hard_valid.patient_id.nunique()}]
    for g,part in hard_valid.groupby("from__hard_macrostate"):
        s=part["to__hard_macrostate"].eq(g); retention.append({"macrostate_id":g,"hard_retention":s.mean(),"hard_switching":1-s.mean(),"n_intervals":len(s),"n_unique_patients":part.patient_id.nunique()})
    retention=pd.DataFrame(retention); reps=10 if args.dry_run else int(cfg["bootstrap"]["replicates"]); null_reps=10 if args.dry_run else int(cfg["null_models"]["replicates"])
    boot=patient_cluster_bootstrap(attached,reps,cfg["random_seed"]); reversal=temporal_reversal_null(attached,null_reps,cfg["random_seed"]); permutation=next_visit_permutation_null(attached,null_reps,cfg["random_seed"]+1)
    inference=net.copy()
    for name,null in (("reversal",reversal),("permutation",permutation)):
      vals=[]
      for row in inference.itertuples():
        x=null.loc[(null.from_macrostate.astype(str)==str(row.from_macrostate))&(null.to_macrostate.astype(str)==str(row.to_macrostate)),"net_flow"]
        vals.append((1+(x.abs()>=abs(row.net_flow)).sum())/(len(x)+1))
      inference[f"{name}_empirical_p"]=vals; inference[f"{name}_q"]=benjamini_hochberg(vals)
    attached.to_parquet(d["analytic"]/"03_patient_temporal_transitions.parquet",index=False)
    outputs={"03_node_temporal_flow.csv":node,"03_macrostate_temporal_flow.csv":macro,"03_hard_macrostate_transition_matrix.csv":hard,"03_macrostate_retention.csv":retention,"03_macrostate_net_flow.csv":net,"03_transition_support.csv":macro[["from_macrostate","to_macrostate","weighted_interval_mass","n_contributing_intervals","n_unique_patients","support_status"]],"03_patient_bootstrap.csv":boot,"03_temporal_reversal_null.csv":reversal,"03_next_visit_permutation_null.csv":permutation,"03_temporal_flow_inference.csv":inference}
    for name,frame in outputs.items(): frame.to_csv(d["tables"]/name,index=False)
    summary={"status":"completed","contract_version":STUDY_CONTRACT_VERSION,"input_files":[args.intervals,args.membership,args.nodes],"config_file":args.config,"config_sha256":sha256_file(args.config),"random_seed":cfg["random_seed"],"n_intervals_total":len(attached),"n_mapped_temporal_intervals":len(eligible),"n_unique_patients":eligible.patient_id.nunique(),"n_supported_macrostates":len(weight_columns(membership,"macrostate_weight__")),"overall_retention":retention.iloc[0].hard_retention,"overall_switching":retention.iloc[0].hard_switching,"n_supported_macrostate_edges":int(macro.support_status.eq("supported").sum()),"directional_imbalance_beyond_null":bool(((inference.reversal_q<.05)&(inference.permutation_q<.05)).any())}
    write_json(d["tables"]/"03_temporal_flow_summary.json",summary)
    pivot=macro.pivot(index="from_macrostate",columns="to_macrostate",values="weighted_interval_mass").fillna(0); fig,ax=plt.subplots(); im=ax.imshow(pivot,cmap="Blues"); ax.set(title="Fractional macrostate temporal flow",xlabel="To macrostate",ylabel="From macrostate"); fig.colorbar(im,ax=ax,label="Weighted interval mass"); fig.tight_layout(); fig.savefig(d["figures"]/"03_macrostate_transition_matrix.png",dpi=cfg["figures"]["dpi"]); plt.close(fig)
    print("ORDER TO REVIEW LONGITUDINAL GRAPH OUTPUTS\n1. Temporal flow\n2. Transition matrix\n3. Net flow and null results")
    return summary
if __name__=="__main__": run(parse_args())

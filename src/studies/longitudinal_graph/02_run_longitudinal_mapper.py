#!/usr/bin/env python3
"""Build the time-blind reference Mapper and patient-supported macrostates."""
from __future__ import annotations
import argparse, logging, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import common
from src.studies._shared import STUDY_CONTRACT_VERSION, create_study_dirs, sha256_file, write_json

def load_config(path): return yaml.safe_load(Path(path).read_text())
def fit_visit_embedding(matrix, config, weights=None):
    scaled=RobustScaler().fit_transform(matrix)
    if weights is not None: scaled=scaled*np.asarray(weights,dtype=float)
    limit=min(int(config["pca"]["max_components"]),*scaled.shape)
    if limit<2: raise ValueError("PCA requires at least two visits and two features")
    probe=PCA(n_components=limit,random_state=config["random_seed"]).fit(scaled)
    hits=np.flatnonzero(np.cumsum(probe.explained_variance_ratio_)>=float(config["pca"]["variance_target"]))
    dimensions=max(2,min(limit,int(hits[0]+1) if len(hits) else limit))
    pca=PCA(n_components=dimensions,random_state=config["random_seed"]); return pca.fit_transform(scaled),pca

def eps_kdist(data,k=4,percentile=65):
    k=max(2,min(int(k),len(data))); distances=NearestNeighbors(n_neighbors=k).fit(data).kneighbors(data)[0][:,-1]
    eps=float(np.percentile(distances,percentile)); return eps if eps>0 else float(np.nextafter(0,1))

def run_mapper(lens, space, config, eps):
    m=config["mapper"]; cubes=int(m["n_cubes"]); overlap=float(m["perc_overlap"]); nodes={}
    axes=[]
    for dim in range(lens.shape[1]):
        lo,hi=float(lens[:,dim].min()),float(lens[:,dim].max()); width=(hi-lo)/(cubes-(cubes-1)*overlap) if hi>lo else 1.0
        step=width*(1-overlap); axes.append([(lo+i*step,lo+i*step+width) for i in range(cubes)])
    for ix,xrange in enumerate(axes[0]):
      for iy,yrange in enumerate(axes[1]):
        mask=(lens[:,0]>=xrange[0])&(lens[:,0]<=xrange[1])&(lens[:,1]>=yrange[0])&(lens[:,1]<=yrange[1]); idx=np.flatnonzero(mask)
        if not len(idx): continue
        labels=DBSCAN(eps=eps,min_samples=int(m["min_samples"])).fit_predict(space[idx])
        for label in sorted(set(labels)-{-1}): nodes[f"cube{ix}_{iy}_cluster{label}"]=set(idx[labels==label])
    links={n:[] for n in nodes}; names=sorted(nodes)
    for i,left in enumerate(names):
      for right in names[i+1:]:
        if nodes[left]&nodes[right]: links[left].append(right); links[right].append(left)
    if not nodes: raise ValueError("Mapper produced no nodes")
    return {"nodes":nodes,"links":links}

def node_support_table(graph, visits, minimum_patients):
    rows=[]
    for node,members in sorted(graph["nodes"].items()):
        patients=visits.iloc[sorted(members)].patient_id.astype(str); counts=patients.value_counts()
        rows.append({"node_id":node,"n_member_visits":len(members),"n_unique_patients":patients.nunique(),
          "max_visits_from_single_patient":int(counts.max()),"max_patient_fraction":float(counts.max()/len(members)),
          "supported":bool(patients.nunique()>=minimum_patients)})
    return pd.DataFrame(rows)

def feature_family(feature):
    low=feature.lower()
    if low.startswith("lab__"): return "LAB"
    if low.startswith("pro__"): return "PRO"
    if low.startswith("sero__"): return "SEROLOGY_CURRENT"
    if low.startswith("ovl__"): return "OVERLAP"
    if low.startswith("ext__"): return "CLINICAL"
    return "OTHER"

def reduce_redundancy(matrix, threshold=.95):
    """Deterministic high-correlation pruning, preferring coverage then simple names."""
    remaining=list(matrix.columns); rows=[]
    coverage=matrix.notna().mean()
    priority=lambda x:(-coverage[x], int(any(t in x.lower() for t in ("percent","estimated"))),x)
    for left in sorted(remaining,key=priority):
        if left not in remaining: continue
        for right in sorted(remaining,key=priority):
            if right==left or right not in remaining or priority(right)<=priority(left): continue
            pair=matrix[[left,right]].dropna()
            if len(pair)<2: continue
            rho=float(pair.corr(method="spearman").iloc[0,1])
            if np.isfinite(rho) and abs(rho)>=threshold:
                kept,removed=sorted((left,right),key=priority)
                remaining.remove(removed)
                rows.append({"feature_a":left,"feature_b":right,"relationship_type":"high_spearman",
                  "correlation":rho,"action":"remove_redundant","retained_feature":kept,
                  "removed_feature":removed,"reason":f"abs_spearman>={threshold:.2f}; coverage_then_clinical_primacy"})
                if removed==left: break
    columns=["feature_a","feature_b","relationship_type","correlation","action","retained_feature","removed_feature","reason"]
    return remaining,pd.DataFrame(rows,columns=columns)

def administrative_qc(membership, metadata, variables=("spine__interval_name","visit_type","protocol","demo__protocol","interval_name"), flag_threshold=.5):
    merged=membership.merge(metadata,on=["patient_id","clinical_episode_id"],how="left")
    assoc=[]; detail=[]
    valid=merged.loc[merged.hard_macrostate.notna()]
    for col in variables:
        if col not in valid: continue
        observed=valid.loc[valid[col].notna()]
        tab=pd.crosstab(observed[col],observed.hard_macrostate); v=np.nan
        if min(tab.shape)>=2 and tab.to_numpy().sum():
            stat=chi2_contingency(tab)[0]; v=math.sqrt(stat/(tab.to_numpy().sum()*(min(tab.shape)-1)))
        assoc.append({"administrative_variable":col,"cramers_v":v,"n_visits":len(observed),
                      "n_patients":observed.patient_id.nunique(),
                      "administrative_confounding_flag":bool(np.isfinite(v) and v>=flag_threshold),
                      "flag_threshold":flag_threshold})
        if col in {"spine__interval_name","visit_type","interval_name"}:
            for (kind,macro),group in observed.groupby([col,"hard_macrostate"]):
                detail.append({"administrative_variable":col,"visit_type":kind,"macrostate_id":macro,
                  "n_visits":len(group),"pct_within_visit_type":len(group)/(observed[col].eq(kind).sum()),
                  "pct_within_macrostate":len(group)/(observed.hard_macrostate.eq(macro).sum()),
                  "n_unique_patients":group.patient_id.nunique()})
    acols=["administrative_variable","cramers_v","n_visits","n_patients","administrative_confounding_flag","flag_threshold"]
    dcols=["administrative_variable","visit_type","macrostate_id","n_visits","pct_within_visit_type","pct_within_macrostate","n_unique_patients"]
    return pd.DataFrame(assoc,columns=acols),pd.DataFrame(detail,columns=dcols)

def build_macrostates(graph, visits, supported_nodes, minimum_macrostate_patients):
    retained={n:graph["nodes"][n] for n in supported_nodes}; nerve=nx.Graph(); nerve.add_nodes_from(retained); edge_rows=[]
    for left in sorted(retained):
      for right in sorted(retained):
        if left>=right or right not in graph["links"].get(left,[]): continue
        lv,rv=retained[left],retained[right]; lp=set(visits.iloc[list(lv)].patient_id); rp=set(visits.iloc[list(rv)].patient_id)
        sv=len(lv&rv); sp=len(lp&rp); vj=sv/len(lv|rv); pj=sp/len(lp|rp)
        nerve.add_edge(left,right,weight=pj); edge_rows.append({"from_node":left,"to_node":right,"n_shared_visits":sv,"n_shared_patients":sp,"visit_jaccard":vj,"patient_jaccard":pj})
    communities=list(nx.community.greedy_modularity_communities(nerve,weight="weight")) if nerve.number_of_edges() else [{n} for n in nerve]
    communities.sort(key=lambda c:(-len(set().union(*(set(visits.iloc[list(retained[n])].patient_id) for n in c))),min(c)))
    mapping={node:i for i,c in enumerate(communities) for node in c}; summaries=[]
    for i,c in enumerate(communities):
        members=set().union(*(retained[n] for n in c)); patients=set(visits.iloc[list(members)].patient_id)
        summaries.append({"macrostate_id":i,"n_nodes":len(c),"n_visits":len(members),"n_unique_patients":len(patients),"macrostate_supported":len(patients)>=minimum_macrostate_patients})
    return mapping,pd.DataFrame(summaries),pd.DataFrame(edge_rows),nerve

def normalized_memberships(graph, visits, node_to_macrostate, macro_summary):
    supported=set(macro_summary.loc[macro_summary.macrostate_supported,"macrostate_id"]); rows=[]
    for i,visit in visits.reset_index(drop=True).iterrows():
        nodes=sorted(n for n,m in graph["nodes"].items() if i in m and node_to_macrostate.get(n) in supported)
        if not nodes: rows.append({"patient_id":visit.patient_id,"clinical_episode_id":visit.clinical_episode_id,"mapper_covered":False,"hard_macrostate":pd.NA,"macrostate_membership_tie":False}); continue
        node_weight=1/len(nodes); macro={}
        for node in nodes: macro[node_to_macrostate[node]]=macro.get(node_to_macrostate[node],0)+node_weight
        maximum=max(macro.values()); winners=[g for g,w in macro.items() if np.isclose(w,maximum,atol=1e-12,rtol=0)]
        base={"patient_id":visit.patient_id,"clinical_episode_id":visit.clinical_episode_id,"mapper_covered":True,
              "hard_macrostate":winners[0] if len(winners)==1 else pd.NA,"macrostate_membership_tie":len(winners)>1}
        for node in nodes: base[f"node_weight__{node}"]=node_weight
        for group,weight in macro.items(): base[f"macrostate_weight__{group}"]=weight
        rows.append(base)
    return pd.DataFrame(rows)

def parse_args(argv=None):
    p=argparse.ArgumentParser(); dirs=create_study_dirs("longitudinal_graph")
    p.add_argument("--integrated",type=Path,default=common.INTEGRATED_LONGITUDINAL_PARQUET); p.add_argument("--state",type=Path,default=dirs["analytic"]/"01_longitudinal_visit_state.parquet")
    p.add_argument("--metadata",type=Path,default=dirs["analytic"]/"01_longitudinal_visit_metadata.parquet"); p.add_argument("--config",type=Path,default=Path(__file__).with_name("config.yaml")); p.add_argument("--dry-run",action="store_true"); return p.parse_args(argv)

def run(args):
    dirs=create_study_dirs("longitudinal_graph"); logging.basicConfig(filename=dirs["logs"]/"02_mapper.log",level=logging.INFO,force=True); cfg=load_config(args.config)
    state=pd.read_parquet(args.state); metadata=pd.read_parquet(args.metadata); eligible=[c for c in state if c not in {"patient_id","clinical_episode_id"}]
    features,audit=reduce_redundancy(state[eligible],float(cfg["representation"].get("redundancy_spearman_abs_threshold",.95)))
    families={f:feature_family(f) for f in features}; counts=pd.Series(families).value_counts()
    weights=[1/math.sqrt(counts[families[f]]) for f in features]
    # S2 is retained as a sensitivity representation; S3 is the pre-specified primary.
    s2_embedding,_=fit_visit_embedding(state[features],cfg)
    embedded,pca=fit_visit_embedding(state[features],cfg,weights); eps=eps_kdist(embedded,cfg["mapper"]["eps_k_neighbors"],cfg["mapper"]["eps_percentile"]); graph=run_mapper(embedded[:,:2],embedded,cfg,eps)
    nodes=node_support_table(graph,state,int(cfg["mapper"]["minimum_node_patients"])); supported=nodes.loc[nodes.supported,"node_id"].tolist()
    if not supported: raise ValueError("Mapper has no patient-supported nodes")
    mapping,macros,edges,nerve=build_macrostates(graph,state,supported,int(cfg["mapper"]["minimum_macrostate_patients"])); membership=normalized_memberships(graph,state,mapping,macros)
    if not membership.mapper_covered.any(): raise ValueError("No visits covered by supported macrostates")
    long_nodes=[]
    for node,members in graph["nodes"].items():
      for i in members: long_nodes.append({"patient_id":state.iloc[i].patient_id,"clinical_episode_id":state.iloc[i].clinical_episode_id,"node_id":node,"node_supported":node in supported,"macrostate_id":mapping.get(node,pd.NA)})
    pd.DataFrame(long_nodes).to_parquet(dirs["analytic"]/"02_mapper_node_membership.parquet",index=False); membership.to_parquet(dirs["analytic"]/"02_visit_mapper_membership.parquet",index=False)
    variance=pd.DataFrame({"component":np.arange(1,pca.n_components_+1),"explained_variance_ratio":pca.explained_variance_ratio_,"cumulative_variance":np.cumsum(pca.explained_variance_ratio_)})
    variance.to_csv(dirs["tables"]/"02_longitudinal_pca_variance.csv",index=False); pd.DataFrame(pca.components_.T,index=features,columns=[f"PC{i+1}" for i in range(pca.n_components_)]).rename_axis("feature").reset_index().to_csv(dirs["tables"]/"02_longitudinal_pca_loadings.csv",index=False)
    nodes.to_csv(dirs["tables"]/"02_mapper_nodes.csv",index=False); edges.to_csv(dirs["tables"]/"02_mapper_topological_edges.csv",index=False); macros.to_csv(dirs["tables"]/"02_mapper_macrostate_summary.csv",index=False); membership.to_csv(dirs["tables"]/"02_mapper_visit_membership.csv",index=False)
    admin,visit_qc=administrative_qc(membership,metadata,flag_threshold=float(cfg["administrative_qc"]["cramers_v_flag_threshold"]))
    admin.to_csv(dirs["tables"]/"02_mapper_administrative_qc.csv",index=False); visit_qc.to_csv(dirs["tables"]/"02_mapper_visit_type_qc.csv",index=False)
    audit.to_csv(dirs["tables"]/"02_longitudinal_redundancy_audit.csv",index=False)
    removed=set(eligible)-set(features); rep=[]
    for scenario,balanced in (("S2-like",False),("S3-like",True)):
      for feature in eligible:
        retained=feature in features
        rep.append({"feature":feature,"family":feature_family(feature),"eligible_from_script01":True,
          "retained_after_redundancy":retained,"used_in_primary_mapper":bool(balanced and retained),
          "scenario":scenario,"balance_weight":(1/math.sqrt(counts[families[feature]]) if balanced and retained else (1.0 if retained else np.nan))})
    pd.DataFrame(rep).to_csv(dirs["tables"]/"02_mapper_representation_manifest.csv",index=False)
    summary={"status":"completed","contract_version":STUDY_CONTRACT_VERSION,"input_files":[args.state,args.metadata],"config_file":args.config,"config_sha256":sha256_file(args.config),"random_seed":cfg["random_seed"],"primary_scenario":"S3-like","sensitivity_scenario":"S2-like","n_visits":len(state),"n_patients":state.patient_id.nunique(),"n_features":len(features),"mapper_parameters":cfg["mapper"],"eps":eps,"n_nodes":len(nodes),"n_supported_nodes":len(supported),"n_macrostates":len(macros),"n_supported_macrostates":int(macros.macrostate_supported.sum())}
    write_json(dirs["tables"]/"02_longitudinal_mapper_summary.json",summary)
    pos={n:np.mean(embedded[list(graph["nodes"][n]),:2],axis=0) for n in supported}; fig,ax=plt.subplots(figsize=(8,6))
    for a,b in nerve.edges: ax.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],color=".75",zorder=1)
    ax.scatter([pos[n][0] for n in supported],[pos[n][1] for n in supported],s=[30+nodes.set_index("node_id").loc[n,"n_unique_patients"]*8 for n in supported],c=[mapping[n] for n in supported],cmap="tab10",zorder=2); ax.set(title="Reference visit-level Mapper",xlabel="PC1 lens",ylabel="PC2 lens"); fig.tight_layout(); fig.savefig(dirs["figures"]/"02_reference_visit_mapper.png",dpi=cfg["figures"]["dpi"]); fig.savefig(dirs["figures"]/"02_reference_visit_mapper_macrostates.png",dpi=cfg["figures"]["dpi"]); plt.close(fig)
    print("ORDER TO REVIEW LONGITUDINAL GRAPH OUTPUTS\n1. PCA variance\n2. Mapper nodes\n3. Macrostate summary\n4. Reference Mapper")
    return summary
if __name__=="__main__": run(parse_args())

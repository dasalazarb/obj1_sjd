#!/usr/bin/env python3
"""Build the time-blind reference Mapper and patient-supported macrostates."""
from __future__ import annotations
import argparse, copy, logging, math, sys
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
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import common
from src.studies._shared import STUDY_CONTRACT_VERSION, create_study_dirs, sha256_file, write_json

def load_config(path): return yaml.safe_load(Path(path).read_text())
KEYS = ["patient_id", "clinical_episode_id"]
HSCRP = "lab__crp_high_sensitivity__value"
URINE_SQUAMOUS = "lab__urine_squamous_cells__value"

def fit_visit_embedding(matrix, config, weights=None):
    scaled=RobustScaler().fit_transform(matrix)
    if weights is not None: scaled=scaled*np.asarray(weights,dtype=float)
    cap=config["pca"]["max_components"]
    limit=min(*scaled.shape) if cap is None else min(int(cap),*scaled.shape)
    if limit<2: raise ValueError("PCA requires at least two visits and two features")
    solver="full" if cap is None else "auto"
    probe=PCA(n_components=limit,svd_solver=solver,random_state=config["random_seed"]).fit(scaled)
    hits=np.flatnonzero(np.cumsum(probe.explained_variance_ratio_)>=float(config["pca"]["variance_target"]))
    dimensions=max(2,min(limit,int(hits[0]+1) if len(hits) else limit))
    pca=PCA(n_components=dimensions,svd_solver=solver,random_state=config["random_seed"]); return pca.fit_transform(scaled),pca

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
    return (mapping,pd.DataFrame(summaries,columns=["macrostate_id","n_nodes","n_visits","n_unique_patients","macrostate_supported"]),
            pd.DataFrame(edge_rows,columns=["from_node","to_node","n_shared_visits","n_shared_patients","visit_jaccard","patient_jaccard"]),nerve)

def normalized_memberships(graph, visits, node_to_macrostate, macro_summary):
    supported=set(macro_summary.loc[macro_summary.macrostate_supported,"macrostate_id"]); rows=[]
    for i,visit in visits.reset_index(drop=True).iterrows():
        all_nodes=sorted(n for n,m in graph["nodes"].items() if i in m)
        nodes=sorted(n for n,m in graph["nodes"].items() if i in m and node_to_macrostate.get(n) in supported)
        coverage={"mapper_graph_covered":bool(all_nodes),
                  "in_unsupported_macrostate":any(n in node_to_macrostate and node_to_macrostate[n] not in supported for n in all_nodes)}
        if not nodes: rows.append({"patient_id":visit.patient_id,"clinical_episode_id":visit.clinical_episode_id,"mapper_covered":False,"hard_macrostate":pd.NA,"macrostate_membership_tie":False,**coverage}); continue
        node_weight=1/len(nodes); macro={}
        for node in nodes: macro[node_to_macrostate[node]]=macro.get(node_to_macrostate[node],0)+node_weight
        maximum=max(macro.values()); winners=[g for g,w in macro.items() if np.isclose(w,maximum,atol=1e-12,rtol=0)]
        base={"patient_id":visit.patient_id,"clinical_episode_id":visit.clinical_episode_id,"mapper_covered":True,
              "hard_macrostate":winners[0] if len(winners)==1 else pd.NA,"macrostate_membership_tie":len(winners)>1,**coverage}
        for node in nodes: base[f"node_weight__{node}"]=node_weight
        for group,weight in macro.items(): base[f"macrostate_weight__{group}"]=weight
        rows.append(base)
    return pd.DataFrame(rows)

def coverage_summary(membership):
    """Graph coverage and supported assignments have separate, explicit denominators.

    Percentages are on the 0--100 scale, relative to all input visits/patients.
    ``mapper_covered`` retains its supported-macrostate meaning for Script 03.
    """
    covered=membership.mapper_graph_covered.astype(bool)
    hard=membership.hard_macrostate.notna()
    ties=membership.macrostate_membership_tie.astype(bool)
    other=covered & ~hard & ~ties
    if not ((hard | ties) <= covered).all() or (hard & ties).any():
        raise AssertionError("Assignments must partition covered visits")
    result={"n_visits":len(membership),"n_patients":int(membership.patient_id.nunique())}
    for stem,mask in (("mapper_covered",covered),("hard_assigned",hard)):
        result[f"n_{stem}_visits"]=int(mask.sum())
        result[f"pct_{stem}_visits"]=100*mask.mean() if len(mask) else 0.0
        result[f"n_{stem}_patients"]=int(membership.loc[mask,"patient_id"].nunique())
        result[f"pct_{stem}_patients"]=100*result[f"n_{stem}_patients"]/max(result["n_patients"],1)
    for stem,mask in (("membership_ties",ties),("uncovered_visits",~covered),
                      ("covered_without_supported_hard_assignment",other)):
        result[f"n_{stem}"]=int(mask.sum())
        result[f"pct_{stem}"]=100*mask.mean() if len(mask) else 0.0
    supported=membership.mapper_covered.astype(bool)
    result["n_supported_macrostate_covered_visits"]=int(supported.sum())
    unsupported=membership.in_unsupported_macrostate.astype(bool)
    result["n_visits_in_unsupported_macrostates"]=int(unsupported.sum())
    result["n_patients_in_unsupported_macrostates"]=int(membership.loc[unsupported,"patient_id"].nunique())
    assert result["n_mapper_covered_visits"]+result["n_uncovered_visits"]==len(membership)
    assert result["n_hard_assigned_visits"]+result["n_membership_ties"]+int(other.sum())==int(covered.sum())
    return result

def sensitivity_features(features, excluded_feature):
    """Drop only the nominated feature from the fixed post-pruning feature set."""
    return [f for f in features if f != excluded_feature]

def run_mapper_scenario(state, features, config, *, family_balance, scenario_name):
    """Run the complete time-blind pipeline; lens is always (PC1, PC2)."""
    if state[KEYS].duplicated().any(): raise ValueError("Visit keys must be unique")
    counts=pd.Series([feature_family(f) for f in features]).value_counts()
    weights=[1/math.sqrt(counts[feature_family(f)]) for f in features] if family_balance else None
    embedding,pca=fit_visit_embedding(state[features],config,weights)
    eps=eps_kdist(embedding,config["mapper"]["eps_k_neighbors"],config["mapper"]["eps_percentile"])
    try:
        graph=run_mapper(embedding[:,:2],embedding,config,eps)
    except ValueError as exc:
        if str(exc) != "Mapper produced no nodes": raise
        graph={"nodes":{},"links":{}}
    nodes=node_support_table(graph,state,int(config["mapper"]["minimum_node_patients"]))
    supported=nodes.loc[nodes.supported,"node_id"].tolist() if len(nodes) else []
    if nodes.empty:
        nodes=pd.DataFrame(columns=["node_id","n_member_visits","n_unique_patients","max_visits_from_single_patient","max_patient_fraction","supported"])
    mapping,macros,edges,nerve=build_macrostates(graph,state,supported,int(config["mapper"]["minimum_macrostate_patients"]))
    membership=normalized_memberships(graph,state,mapping,macros)
    variance=float(pca.explained_variance_ratio_.sum())
    summary={"scenario":scenario_name,"status":"completed" if membership.mapper_covered.any() else "no_supported_coverage",
             "family_balance":family_balance,"n_features":len(features),"n_pcs":int(pca.n_components_),
             "cumulative_variance":variance,"pca_variance_target_reached":bool(variance>=float(config["pca"]["variance_target"])),
             "eps":eps,"n_nodes":len(nodes),"n_supported_nodes":len(supported),
             "n_macrostates":len(macros),"n_supported_macrostates":int(macros.macrostate_supported.sum()),
             **coverage_summary(membership)}
    return {"features":list(features),"embedding":embedding,"pca":pca,"eps":eps,"graph":graph,
            "node_table":nodes,"macrostate_table":macros,"visit_membership":membership,
            "topological_edges":edges,"node_to_macrostate":mapping,"nerve":nerve,"summary":summary}

def macrostate_sets(scenario):
    """Use all visits in the macrostate's nodes, including ambiguous memberships."""
    membership=scenario["visit_membership"]
    sets={}
    for macro in scenario["macrostate_table"].macrostate_id:
        indices=set().union(*(members for node,members in scenario["graph"]["nodes"].items()
                             if scenario["node_to_macrostate"].get(node)==macro))
        visits=membership.iloc[sorted(indices)]
        sets[macro]=(set(map(tuple,visits[KEYS].to_numpy())),set(visits.patient_id))
    return sets

def jaccard(left,right):
    return len(left & right)/len(left | right) if left | right else 0.0

def compare_mapper_scenarios(primary, sensitivity):
    """Match each primary macrostate independently by patient, then visit Jaccard.

    Many-to-one matches are allowed and exposed so mergers are not hidden.
    ARI uses only non-tied supported hard assignments on common visit keys.
    """
    left_sets,right_sets=macrostate_sets(primary),macrostate_sets(sensitivity)
    rows=[]
    for left,(lv,lp) in left_sets.items():
        matches=[(jaccard(lp,rp),jaccard(lv,rv),right) for right,(rv,rp) in right_sets.items()]
        match=max(matches,key=lambda x:(x[0],x[1],-int(x[2]))) if matches else None
        if match is None or match[0]==0:
            right=None; rv,rp=set(),set(); pj=vj=0.0
        else:
            pj,vj,right=match; rv,rp=right_sets[right]
        rows.append({"s3_macrostate_id":left,"matched_s2_macrostate_id":right,
                     "s3_n_visits":len(lv),"s2_n_visits":len(rv),"s3_n_patients":len(lp),"s2_n_patients":len(rp),
                     "visit_jaccard":vj,"patient_jaccard":pj})
    columns=["s3_macrostate_id","matched_s2_macrostate_id","s3_n_visits","s2_n_visits","s3_n_patients","s2_n_patients","visit_jaccard","patient_jaccard"]
    matching=pd.DataFrame(rows,columns=columns)
    def hard(scenario):
        frame=scenario["visit_membership"]
        return frame.loc[frame.hard_macrostate.notna() & ~frame.macrostate_membership_tie,KEYS+["hard_macrostate"]]
    common_visits=hard(primary).merge(hard(sensitivity),on=KEYS,validate="one_to_one",suffixes=("_primary","_sensitivity"))
    enough=len(common_visits)>=2
    ari=float(adjusted_rand_score(common_visits.hard_macrostate_primary.astype(int),common_visits.hard_macrostate_sensitivity.astype(int))) if enough else None
    supported_ids=set(primary["macrostate_table"].loc[primary["macrostate_table"].macrostate_supported,"macrostate_id"])
    supported_matches=matching.loc[matching.s3_macrostate_id.isin(supported_ids)]
    ps=primary["summary"]; ss=sensitivity["summary"]
    summary={"ari":ari,"ari_status":"requires_sample_size_review" if enough else "insufficient_comparable_visits",
             "n_visits_compared":len(common_visits),"n_patients_compared":int(common_visits.patient_id.nunique()),
             "pct_primary_visits_compared":100*len(common_visits)/max(ps["n_visits"],1),
             "n_primary_labels_compared":int(common_visits.hard_macrostate_primary.nunique()),
             "n_sensitivity_labels_compared":int(common_visits.hard_macrostate_sensitivity.nunique()),
             "matching_method":"independent_max_patient_jaccard_then_visit_jaccard; many_to_one_allowed",
             "patient_jaccard_summary":{"mean":float(supported_matches.patient_jaccard.mean()) if len(supported_matches) else None,
                                         "min":float(supported_matches.patient_jaccard.min()) if len(supported_matches) else None,
                                         "n_supported_primary_macrostates":len(supported_matches)},
             "patient_jaccard_vs_primary":float(supported_matches.patient_jaccard.mean()) if len(supported_matches) else None,
             "visit_jaccard_vs_primary":float(supported_matches.visit_jaccard.mean()) if len(supported_matches) else None}
    for prefix,source in (("s3",ps),("s2",ss)):
        for field in ("n_nodes","n_supported_nodes","n_macrostates","n_supported_macrostates"):
            summary[f"{prefix}_{field}"]=source[field]
    return matching,summary

def feature_distribution_diagnostics(raw, features=(HSCRP,URINE_SQUAMOUS), *, source="preimputation_integrated"):
    """Finite observed values only; extremes are strictly outside Q1/Q3 +/- 3 IQR."""
    rows=[]
    for feature in features:
        unavailable=source=="source_unavailable" or feature not in raw
        values=pd.to_numeric(raw[feature],errors="coerce") if not unavailable else pd.Series(np.nan,index=raw.index)
        observed=values.loc[np.isfinite(values)]
        q1,q3=observed.quantile(.25),observed.quantile(.75); iqr=q3-q1
        lower,upper=q1-3*iqr,q3+3*iqr
        rows.append({"feature":feature,"status":"completed" if len(observed) and not unavailable else "not_applicable",
                     "reason":("original_source_unavailable" if source=="source_unavailable" else
                               "feature_unavailable" if feature not in raw else "no_finite_observations" if not len(observed) else ""),
                     "source":source,"n_total_visits":len(raw),"n_observed":pd.NA if unavailable else len(observed),
                     "pct_observed":np.nan if unavailable else 100*len(observed)/max(len(raw),1),"median":observed.median(),"q1":q1,"q3":q3,
                     "iqr":iqr,"p95":observed.quantile(.95),"p99":observed.quantile(.99),"min":observed.min(),"max":observed.max(),
                     "n_zero":pd.NA if unavailable else int(observed.eq(0).sum()),"n_negative":pd.NA if unavailable else int(observed.lt(0).sum()),
                     "n_extreme_iqr":pd.NA if unavailable else int((observed.lt(lower)|observed.gt(upper)).sum()),
                     "extreme_iqr_rule":"x < Q1 - 3*IQR or x > Q3 + 3*IQR",
                     "extreme_lower_bound":lower,"extreme_upper_bound":upper})
    return pd.DataFrame(rows)

def hscrp_log_state(state, features, diagnostics):
    """Fail closed when original observations are absent, negative, or unavailable."""
    row=diagnostics.set_index("feature").loc[HSCRP]
    if HSCRP not in features: return None,"feature_not_retained_after_redundancy"
    if row.status!="completed": return None,"original_observations_unavailable"
    if row.n_negative: return None,"negative_original_values"
    values=state[HSCRP]
    if not np.isfinite(values).all() or values.lt(0).any(): return None,"invalid_state_values"
    transformed=state.copy(); transformed[HSCRP]=np.log1p(values)
    return transformed,None

def pc_feature_diagnostics(scenarios, features=(HSCRP,URINE_SQUAMOUS)):
    """Audit implicated feature loadings in PC1--PC4 for each fitted scenario."""
    rows=[]
    for name,scenario in scenarios.items():
        retained=scenario["features"]; components=scenario["pca"].components_
        for feature in features:
            if feature not in retained: continue
            index=retained.index(feature)
            for i,component in enumerate(components[:4]):
                loading=float(component[index])
                rows.append({"scenario":name,"feature":feature,"component":f"PC{i+1}",
                             "loading":loading,"squared_loading_fraction":loading**2/float(np.square(component).sum()),
                             "absolute_loading_rank":int(1+np.sum(np.abs(component)>abs(loading)))})
    return pd.DataFrame(rows,columns=["scenario","feature","component","loading","squared_loading_fraction","absolute_loading_rank"])

def interval_coverage_summary(intervals, membership, maximum_interval_years=None):
    """Count original consecutive endpoints without calculating temporal flow."""
    summary={"status":"completed","n_intervals_total":len(intervals)}
    positive=intervals.interval_days.gt(0)
    if maximum_interval_years is not None: positive &= intervals.interval_years.le(maximum_interval_years)
    for field,stem in (("mapper_graph_covered","mapper_graph"),("mapper_covered","supported_macrostate")):
        keys=set(map(tuple,membership.loc[membership[field],KEYS].to_numpy()))
        both=pd.Series([(r.patient_id,r.from_clinical_episode_id) in keys and (r.patient_id,r.to_clinical_episode_id) in keys
                        for r in intervals.itertuples()],index=intervals.index)
        summary[f"n_intervals_both_endpoints_{stem}_covered"]=int((positive & both).sum())
    hard=set(map(tuple,membership.loc[membership.hard_macrostate.notna(),KEYS].to_numpy()))
    both=pd.Series([(r.patient_id,r.from_clinical_episode_id) in hard and (r.patient_id,r.to_clinical_episode_id) in hard
                    for r in intervals.itertuples()],index=intervals.index)
    summary["n_intervals_both_endpoints_hard_assigned"]=int((positive & both).sum())
    return summary

def parse_args(argv=None):
    p=argparse.ArgumentParser(); dirs=create_study_dirs("longitudinal_graph")
    p.add_argument("--integrated",type=Path,default=common.INTEGRATED_LONGITUDINAL_PARQUET); p.add_argument("--state",type=Path,default=dirs["analytic"]/"01_longitudinal_visit_state.parquet")
    p.add_argument("--metadata",type=Path,default=dirs["analytic"]/"01_longitudinal_visit_metadata.parquet")
    p.add_argument("--intervals",type=Path,default=dirs["analytic"]/"01_longitudinal_intervals.parquet")
    p.add_argument("--config",type=Path,default=Path(__file__).with_name("config.yaml")); p.add_argument("--dry-run",action="store_true"); return p.parse_args(argv)

def run(args):
    dirs=create_study_dirs("longitudinal_graph")
    logging.basicConfig(filename=dirs["logs"]/"02_mapper.log",level=logging.INFO,force=True)
    cfg=load_config(args.config)
    state=pd.read_parquet(args.state); metadata=pd.read_parquet(args.metadata)
    eligible=[c for c in state if c not in KEYS]
    features,audit=reduce_redundancy(state[eligible],float(cfg["representation"].get("redundancy_spearman_abs_threshold",.95)))

    # Audit original values on exactly the Script-01 cohort, never imputed coverage.
    raw_source="source_unavailable"
    raw=state[KEYS].copy()
    if args.integrated.exists():
        source=pd.read_parquet(args.integrated)
        observed_features=[f for f in (HSCRP,URINE_SQUAMOUS) if f in source]
        raw=raw.merge(source[KEYS+observed_features],on=KEYS,how="left",validate="one_to_one",indicator=True)
        if not raw._merge.eq("both").all(): raise ValueError("Integrated source does not contain every state visit")
        raw=raw.drop(columns="_merge"); raw_source="preimputation_integrated"
    diagnostics=feature_distribution_diagnostics(raw,source=raw_source)
    diagnostics.to_csv(dirs["tables"]/"02_feature_distribution_diagnostics.csv",index=False)

    primary=run_mapper_scenario(state,features,cfg,family_balance=True,scenario_name="S3-primary")
    scenarios={"S3-primary":primary}
    scenarios["S2-no-family-balance"]=run_mapper_scenario(state,features,cfg,family_balance=False,scenario_name="S2-no-family-balance")
    full_cfg=copy.deepcopy(cfg); full_cfg["pca"]["max_components"]=None
    scenarios["S3-full80"]=run_mapper_scenario(state,features,full_cfg,family_balance=True,scenario_name="S3-full80")
    excluded=sensitivity_features(features,URINE_SQUAMOUS)
    skipped={}
    if len(excluded)<len(features):
        scenarios["S3-without-urine-squamous"]=run_mapper_scenario(state,excluded,cfg,family_balance=True,scenario_name="S3-without-urine-squamous")
    else:
        skipped["S3-without-urine-squamous"]="feature_not_retained_after_redundancy"
    log_state,reason=hscrp_log_state(state,features,diagnostics)
    if log_state is not None:
        scenarios["S3-hsCRP-log1p"]=run_mapper_scenario(log_state,features,cfg,family_balance=True,scenario_name="S3-hsCRP-log1p")
    else:
        skipped["S3-hsCRP-log1p"]=reason
    pc_feature_diagnostics(scenarios).to_csv(dirs["tables"]/"02_mapper_pc_feature_diagnostics.csv",index=False)

    comparisons={}; sensitivity_rows=[]
    for name,scenario in scenarios.items():
        matching,comparison=compare_mapper_scenarios(primary,scenario)
        comparisons[name]=comparison
        sensitivity_rows.append({**scenario["summary"],"n_ties":scenario["summary"]["n_membership_ties"],
                                 "ari_vs_primary":comparison["ari"],
                                 "patient_jaccard_vs_primary":comparison["patient_jaccard_vs_primary"],
                                 "visit_jaccard_vs_primary":comparison["visit_jaccard_vs_primary"],
                                 "n_visits_compared":comparison["n_visits_compared"],
                                 "n_patients_compared":comparison["n_patients_compared"],
                                 "ari_status":comparison["ari_status"]})
        # Keep per-macrostate overlap inspectable; scalar means alone hide losses.
        if name=="S2-no-family-balance":
            matching.to_csv(dirs["tables"]/"02_s2_s3_macrostate_stability.csv",index=False)
            write_json(dirs["tables"]/"02_s2_s3_stability_summary.json",comparison)
        elif name!="S3-primary":
            stem={"S3-full80":"full80","S3-without-urine-squamous":"no_urine_squamous","S3-hsCRP-log1p":"hscrp_log1p"}[name]
            matching.rename(columns={"s3_macrostate_id":"primary_macrostate_id","matched_s2_macrostate_id":"matched_sensitivity_macrostate_id",
                                     "s3_n_visits":"primary_n_visits","s2_n_visits":"sensitivity_n_visits",
                                     "s3_n_patients":"primary_n_patients","s2_n_patients":"sensitivity_n_patients"}).to_csv(
                                         dirs["tables"]/f"02_{stem}_macrostate_stability.csv",index=False)
    for name,reason in skipped.items():
        sensitivity_rows.append({"scenario":name,"status":"not_applicable","reason":reason})
        comparisons[name]={"status":"not_applicable","reason":reason,"ari":None}
        stem={"S3-without-urine-squamous":"no_urine_squamous","S3-hsCRP-log1p":"hscrp_log1p"}[name]
        # Overwrite a previous run's comparison rather than leaving stale evidence.
        pd.DataFrame(columns=["primary_macrostate_id","matched_sensitivity_macrostate_id","primary_n_visits","sensitivity_n_visits",
                              "primary_n_patients","sensitivity_n_patients","visit_jaccard","patient_jaccard"]).to_csv(
                                  dirs["tables"]/f"02_{stem}_macrostate_stability.csv",index=False)
    order=["S3-primary","S2-no-family-balance","S3-full80","S3-without-urine-squamous","S3-hsCRP-log1p"]
    sensitivity_table=pd.DataFrame(sensitivity_rows).set_index("scenario").reindex(order).reset_index()
    sensitivity_table.to_csv(dirs["tables"]/"02_mapper_sensitivity_summary.csv",index=False)

    for prefix,scenario in (("s2",scenarios["S2-no-family-balance"]),("s3",primary)):
        scenario["node_table"].to_csv(dirs["tables"]/f"02_{prefix}_mapper_nodes.csv",index=False)
        scenario["macrostate_table"].to_csv(dirs["tables"]/f"02_{prefix}_mapper_macrostate_summary.csv",index=False)
        scenario["visit_membership"].to_csv(dirs["tables"]/f"02_{prefix}_mapper_visit_membership.csv",index=False)

    # Canonical outputs remain S3-primary, with the same supported weights for flow.
    graph=primary["graph"]; nodes=primary["node_table"]; macros=primary["macrostate_table"]
    membership=primary["visit_membership"]; mapping=primary["node_to_macrostate"]; pca=primary["pca"]
    supported=nodes.loc[nodes.supported,"node_id"].tolist()
    long_nodes=[]
    for node,members in graph["nodes"].items():
        for i in sorted(members):
            long_nodes.append({"patient_id":state.iloc[i].patient_id,"clinical_episode_id":state.iloc[i].clinical_episode_id,
                               "node_id":node,"node_supported":node in supported,"macrostate_id":mapping.get(node,pd.NA)})
    pd.DataFrame(long_nodes,columns=KEYS+["node_id","node_supported","macrostate_id"]).to_parquet(dirs["analytic"]/"02_mapper_node_membership.parquet",index=False)
    membership.to_parquet(dirs["analytic"]/"02_visit_mapper_membership.parquet",index=False)
    variance=pd.DataFrame({"component":np.arange(1,pca.n_components_+1),"explained_variance_ratio":pca.explained_variance_ratio_,
                          "cumulative_variance":np.cumsum(pca.explained_variance_ratio_)})
    variance.to_csv(dirs["tables"]/"02_longitudinal_pca_variance.csv",index=False)
    pd.DataFrame(pca.components_.T,index=features,columns=[f"PC{i+1}" for i in range(pca.n_components_)]).rename_axis("feature").reset_index().to_csv(dirs["tables"]/"02_longitudinal_pca_loadings.csv",index=False)
    nodes.to_csv(dirs["tables"]/"02_mapper_nodes.csv",index=False)
    primary["topological_edges"].to_csv(dirs["tables"]/"02_mapper_topological_edges.csv",index=False)
    macros.to_csv(dirs["tables"]/"02_mapper_macrostate_summary.csv",index=False)
    membership.to_csv(dirs["tables"]/"02_mapper_visit_membership.csv",index=False)
    admin,visit_qc=administrative_qc(membership,metadata,flag_threshold=float(cfg["administrative_qc"]["cramers_v_flag_threshold"]))
    admin.to_csv(dirs["tables"]/"02_mapper_administrative_qc.csv",index=False)
    visit_qc.to_csv(dirs["tables"]/"02_mapper_visit_type_qc.csv",index=False)
    audit.to_csv(dirs["tables"]/"02_longitudinal_redundancy_audit.csv",index=False)
    families={f:feature_family(f) for f in features}; counts=pd.Series(families).value_counts(); rep=[]
    for scenario,balanced in (("S2-like",False),("S3-like",True)):
        for feature in eligible:
            retained=feature in features
            rep.append({"feature":feature,"family":feature_family(feature),"eligible_from_script01":True,
                        "retained_after_redundancy":retained,"used_in_primary_mapper":bool(balanced and retained),
                        "scenario":scenario,"balance_weight":(1/math.sqrt(counts[families[feature]]) if balanced and retained else (1.0 if retained else np.nan))})
    pd.DataFrame(rep).to_csv(dirs["tables"]/"02_mapper_representation_manifest.csv",index=False)

    s2=comparisons["S2-no-family-balance"]; full80=scenarios["S3-full80"]["summary"]
    interval_path=args.intervals
    intervals=(interval_coverage_summary(pd.read_parquet(interval_path),membership,cfg["temporal"]["maximum_interval_years"])
               if interval_path.exists() else {"status":"not_applicable","reason":"script01_intervals_unavailable"})
    # Qualitative gate requires review of overlap, sample sizes, loading changes and
    # raw distributions. Do not invent an ARI cutoff or automatically approve flow.
    blockers=[]
    if primary["summary"]["n_supported_macrostates"]<2: blockers.append("primary_has_fewer_than_two_supported_macrostates")
    for name,scenario in scenarios.items():
        if name!="S3-primary" and scenario["summary"]["n_supported_macrostates"]<2:
            blockers.append(f"{name}:fewer_than_two_supported_macrostates")
    if intervals.get("n_intervals_both_endpoints_supported_macrostate_covered")==0: blockers.append("no_supported_mapped_intervals")
    if not primary["summary"]["n_supported_macrostate_covered_visits"]: blockers.append("no_supported_coverage")
    if not full80["pca_variance_target_reached"]: blockers.append("full80_variance_target_not_reached")
    administrative_flag=bool(admin.administrative_confounding_flag.any())
    if administrative_flag: blockers.append("administrative_association_above_configured_threshold")
    input_files=[args.state,args.metadata]
    if args.integrated.exists(): input_files.append(args.integrated)
    if interval_path.exists(): input_files.append(interval_path)
    summary={**primary["summary"],"status":"completed","contract_version":STUDY_CONTRACT_VERSION,
             "input_files":input_files,"input_sha256":{str(path):sha256_file(path) for path in input_files},
             "config_file":args.config,"config_sha256":sha256_file(args.config),"random_seed":cfg["random_seed"],
             "primary_scenario":"S3-like","sensitivity_scenario":"S2-like",
             "n_features_before_redundancy":len(eligible),"n_features_after_redundancy":len(features),
             "n_pca_components":primary["summary"]["n_pcs"],"pca_cumulative_variance":primary["summary"]["cumulative_variance"],
             "n_pcs_primary":primary["summary"]["n_pcs"],"variance_primary":primary["summary"]["cumulative_variance"],
             "n_pcs_full80":full80["n_pcs"],"variance_full80":full80["cumulative_variance"],
             "mapper_parameters":cfg["mapper"],"s2_s3_ari":s2["ari"],"s2_s3_patient_jaccard_summary":s2["patient_jaccard_summary"],
             "full80_ari_vs_primary":comparisons["S3-full80"]["ari"],
             "no_urine_squamous_ari_vs_primary":comparisons["S3-without-urine-squamous"]["ari"],
             "hscrp_log_ari_vs_primary":comparisons["S3-hsCRP-log1p"]["ari"],
             "sensitivity_comparisons":comparisons,"interval_coverage":intervals,
             "administrative_confounding_flag":administrative_flag,
             "administrative_qc_interpretation":"review_variable_associations; absence_of_flag_does_not_establish_no_confounding",
             "coverage_definitions":{"mapper_covered_summary":"at_least_one_graph_node",
                                     "mapper_covered_membership_column":"at_least_one_supported_macrostate; unchanged_for_Script03",
                                     "covered_without_supported_hard_assignment":"graph_covered_and_neither_hard_nor_tied",
                                     "unsupported_macrostate_counts":"any_membership; may_overlap_supported_membership",
                                     "pct_denominators":"all_input_visits_or_unique_patients; percentages_0_to_100"},
             "representation_stability_gate":{"status":"blocked" if blockers else "requires_review","blockers":blockers,
                                              "unavailable_checks":skipped,
                                              "raw_feature_diagnostics_status":raw_source,
                                              "interval_coverage_status":intervals["status"],
                                              "temporal_flow_executed":False}}
    write_json(dirs["tables"]/"02_longitudinal_mapper_summary.json",summary)
    embedded=primary["embedding"]; nerve=primary["nerve"]
    pos={n:np.mean(embedded[list(graph["nodes"][n]),:2],axis=0) for n in supported}
    fig,ax=plt.subplots(figsize=(8,6))
    for a,b in nerve.edges: ax.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],color=".75",zorder=1)
    if supported:
        ax.scatter([pos[n][0] for n in supported],[pos[n][1] for n in supported],
                   s=[30+nodes.set_index("node_id").loc[n,"n_unique_patients"]*8 for n in supported],
                   c=[mapping[n] for n in supported],cmap="tab10",zorder=2)
    ax.set(title="Reference visit-level Mapper",xlabel="PC1 lens",ylabel="PC2 lens")
    fig.tight_layout()
    for name in ("02_reference_visit_mapper.png","02_reference_visit_mapper_macrostates.png"):
        fig.savefig(dirs["figures"]/name,dpi=cfg["figures"]["dpi"])
    plt.close(fig)
    print("ORDER TO REVIEW LONGITUDINAL GRAPH OUTPUTS\n1. Sensitivity summary\n2. S2/S3 macrostate stability\n3. Feature distribution diagnostics\n4. Mapper summary and coverage\nReview the representation-stability gate before Script 03.")
    return summary

if __name__=="__main__": run(parse_args())

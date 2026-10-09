#!/usr/bin/env python3
"""Build the time-blind reference Mapper and patient-supported macrostates."""
from __future__ import annotations
import argparse, copy, json, logging, math, sys
from types import SimpleNamespace
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
from src.studies.longitudinal_graph import validation as qc

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
    return qc.audited_mapper(lens, space, config, eps, eps_kdist)[0]

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

def run_mapper_scenario(state, features, config, *, family_balance, scenario_name, patient_weighted=False):
    """Run the complete time-blind pipeline; lens is always (PC1, PC2)."""
    qc.validate_keys(state, "visit state")
    counts=pd.Series([feature_family(f) for f in features]).value_counts()
    weights=[1/math.sqrt(counts[feature_family(f)]) for f in features] if family_balance else None
    if patient_weighted:
        embedding,pca=qc.weighted_embedding(state[features],config,weights,state.patient_id)
    else:
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
    summary["execution_status"]="completed"
    summary["weight_rule"]="equal_patient_mass" if patient_weighted else "equal_visit_mass"
    summary["weight_method"]="equal_weight_per_supported_node"
    _,cover,noise=qc.audited_mapper(embedding[:,:2],embedding,config,eps,eps_kdist,state.patient_id)
    # Use the actual returned graph for support counts (including test-injected zero-node runs).
    for row in cover.index:
        prefix=f"cube{cover.loc[row,'cube_x']}_{cover.loc[row,'cube_y']}_cluster"
        cover.loc[row,"n_supported_nodes"]=sum(n.startswith(prefix) for n in supported)
    result={"features":list(features),"embedding":embedding,"pca":pca,"eps":eps,"graph":graph,
            "node_table":nodes,"macrostate_table":macros,"visit_membership":membership,
            "topological_edges":edges,"node_to_macrostate":mapping,"nerve":nerve,"summary":summary,
            "cover_qc":cover,"noise_qc":pd.concat([state[KEYS].reset_index(drop=True),noise],axis=1)}
    return qc.topology_identity(result)

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
    if enough and min(summary["n_primary_labels_compared"],summary["n_sensitivity_labels_compared"])<2:
        summary["ari_status"]="degenerate_single_label_not_stability_evidence"
    summary["n_common_visits"]=len(common_visits)
    summary["n_common_patients"]=int(common_visits.patient_id.nunique())
    for metric in ("patient_jaccard","visit_jaccard"):
        values=supported_matches[metric]
        summary[metric+"_distribution"]={name:float(getattr(values,name)()) if len(values) else None for name in ("min","median","max")}
    positive=matching.loc[matching.patient_jaccard.gt(0)]
    summary["many_to_one_matches"]=int(positive.matched_s2_macrostate_id.duplicated(keep=False).sum())
    summary["unmatched_primary_macrostates"]=int(matching.patient_jaccard.eq(0).sum())
    pair_rows=[]
    for left,(lv,lp) in left_sets.items():
        for right,(rv,rp) in right_sets.items():
            if lp & rp: pair_rows.append({"primary":left,"sensitivity":right,"patient_jaccard":jaccard(lp,rp),"visit_jaccard":jaccard(lv,rv)})
    summary["all_nonzero_correspondences"]=pair_rows
    summary["primary_split_candidates"]=int(sum(sum(row["primary"]==left for row in pair_rows)>1 for left in left_sets))
    summary["sensitivity_merge_candidates"]=int(sum(sum(row["sensitivity"]==right for row in pair_rows)>1 for right in right_sets))
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
        rows[-1].update({"unit_validation":"unknown","assay_era_validation":"unknown","detection_limit_coding":"unknown",
                         "prior_transformation_status":"unknown"})
        rows[-1]["n_invalid_observed"]=int((raw[feature].notna() & ~np.isfinite(values)).sum()) if not unavailable else pd.NA
        if "patient_id" in raw and not unavailable:
            extreme=values.lt(lower)|values.gt(upper)
            rows[-1]["n_observed_patients"]=raw.loc[values.index.isin(observed.index),"patient_id"].nunique()
            rows[-1]["n_extreme_patients"]=raw.loc[extreme,"patient_id"].nunique()
    return pd.DataFrame(rows)

def hscrp_log_state(state, features, diagnostics, prior_transform="identity"):
    """Fail closed when original observations are absent, negative, or unavailable."""
    row=diagnostics.set_index("feature").loc[HSCRP]
    if HSCRP not in features: return None,"feature_not_retained_after_redundancy"
    if prior_transform!="identity": return None,"prior_transform_not_verified_identity"
    if row.status!="completed": return None,"original_observations_unavailable"
    if row.get("n_invalid_observed",0): return None,"invalid_original_observations"
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
    p=argparse.ArgumentParser(); dirs=qc.study_paths(common)
    p.add_argument("--integrated",type=Path,default=common.INTEGRATED_LONGITUDINAL_PARQUET); p.add_argument("--state",type=Path,default=dirs["analytic"]/"01_longitudinal_visit_state.parquet")
    p.add_argument("--metadata",type=Path,default=dirs["analytic"]/"01_longitudinal_visit_metadata.parquet")
    p.add_argument("--intervals",type=Path,default=dirs["analytic"]/"01_longitudinal_intervals.parquet")
    p.add_argument("--feature-manifest",type=Path,default=dirs["tables"]/"01_longitudinal_feature_manifest.csv")
    p.add_argument("--output-root",type=Path,help="Isolated output bundle root (analytic/tables/figures/qc/logs)")
    p.add_argument("--config",type=Path,default=Path(__file__).with_name("config.yaml")); p.add_argument("--dry-run",action="store_true"); return p.parse_args(argv)

def run(args):
    from src.studies.longitudinal_graph.runner import execute
    return execute(args, SimpleNamespace(**globals()))

if __name__=="__main__": run(parse_args())

#!/usr/bin/env python3
"""Describe candidate clinical drivers and baseline profiles of observed Pop transitions.

This module consumes the canonical script-23 episode artifact. Concurrent changes
are descriptive components of observed transitions, not causal mechanisms.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402

POPS = ["Pop1", "Pop2", "Pop3"]
DOMAINS = ["constitutional", "lymphadenopathy", "articular", "cutaneous", "pulmonary",
           "renal", "muscular", "pns", "cns", "hematologic", "biological"]
DELTAS = ["delta_essdai", "delta_esspri", "delta_esspri_dryness", "delta_esspri_fatigue",
          "delta_esspri_pain", "delta_pcs", "delta_mcs", "delta_profad", "delta_mdafs",
          "delta_n_extraglandular_domains"]
REQUIRED = ["patient_id", "episode_id", "from_visit_id", "to_visit_id", "from_date", "to_date",
            "interval_days", "interval_years", "from_pop", "to_pop", "pop_transition_evaluable"]
MODEL_COLUMNS = ["model_name", "outcome", "predictor", "contrast", "coefficient", "standard_error",
                 "odds_ratio", "ci95_low", "ci95_high", "p_value", "n_episodes", "n_patients",
                 "n_events", "formula", "model_status", "interpretation"]
INTERPRETATION = "Baseline profile associated with an observed transition; not a causal mechanism."


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=getattr(common, "TRANSITION_EPISODE_PARQUET", common.INTERMEDIATE_DATA_DIR / "23_build_transition_episode_dataset" / "23_transition_episode_dataset.parquet"))
    p.add_argument("--min-transition-n", type=int, default=10)
    p.add_argument("--min-model-events", type=int, default=20)
    g = p.add_mutually_exclusive_group(); g.add_argument("--overwrite", action="store_true", default=True)
    g.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    p.add_argument("--skip-models", action="store_true")
    return p.parse_args(argv)


def load_episodes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Run src/block_B/23_build_transition_episode_dataset.py first.")
    return pd.read_parquet(path)


def validate_episodes(data: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in data]
    if missing: raise ValueError(f"Transition episode input missing required columns: {missing}")
    if data.episode_id.duplicated().any(): raise ValueError("Duplicate episode IDs")
    if data[["episode_id", "patient_id", "from_visit_id", "to_visit_id"]].isna().any().any():
        raise ValueError("Episode, patient, origin, and destination identifiers must be nonmissing")
    out = data.copy()
    out["from_date"] = pd.to_datetime(out.from_date, errors="coerce")
    out["to_date"] = pd.to_datetime(out.to_date, errors="coerce")
    if out[["from_date", "to_date"]].isna().any().any() or out.to_date.le(out.from_date).any():
        raise ValueError("Origin and destination dates must be present and chronological")
    if pd.to_numeric(out.interval_days, errors="coerce").le(0).any() or pd.to_numeric(out.interval_years, errors="coerce").le(0).any():
        raise ValueError("Episode intervals must be positive")
    return out


def classify_transition_family(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy(); valid = out.from_pop.isin(POPS) & out.to_pop.isin(POPS) & out.pop_transition_evaluable.eq(True)
    pairs = {("Pop1", "Pop1"): "stable_pop1", ("Pop2", "Pop2"): "stable_pop2", ("Pop3", "Pop3"): "stable_pop3",
             ("Pop2", "Pop1"): "to_pop1", ("Pop3", "Pop1"): "to_pop1", ("Pop1", "Pop2"): "from_pop1",
             ("Pop1", "Pop3"): "from_pop1", ("Pop2", "Pop3"): "pop2_to_pop3", ("Pop3", "Pop2"): "pop3_to_pop2"}
    out["transition_family"] = [pairs.get((a, b), "unevaluable") if ok else "unevaluable" for a, b, ok in zip(out.from_pop, out.to_pop, valid)]
    out["transition_pair"] = out.from_pop.astype("string").fillna("Missing") + " -> " + out.to_pop.astype("string").fillna("Missing")
    out["event_any_pop_change"] = valid & out.from_pop.ne(out.to_pop)
    out["event_transition_to_pop1"] = valid & out.from_pop.isin(["Pop2", "Pop3"]) & out.to_pop.eq("Pop1")
    return out


def _domain_column(data: pd.DataFrame, domain: str) -> Optional[str]:
    for c in (f"eg_{domain}_active_transition", f"{domain}_transition", f"domain_{domain}_transition"):
        if c in data: return c
    return None


def derive_candidate_drivers(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy(); cols = {d: _domain_column(out, d) for d in DOMAINS}; rows = []
    for _, row in out.iterrows():
        family = row["transition_family"]
        if family == "unevaluable": rows.append(("", 0, "not_evaluable")); continue
        if family not in {"to_pop1", "from_pop1"}:
            rows.append(("", 0, "not_applicable")); continue
        target = "incident" if family == "to_pop1" else "resolved"
        relevant = [c for c in cols.values() if c]
        if not relevant or any(pd.isna(row[c]) or str(row[c]) == "not_evaluable" for c in relevant):
            rows.append(("", 0, "not_evaluable")); continue
        names = [d for d, c in cols.items() if c and str(row[c]) == target]
        klass = "single_domain" if len(names) == 1 else "multiple_domains" if len(names) > 1 else "no_observed_domain_change"
        rows.append(("|".join(names), len(names), klass))
    out[["candidate_driver_domains", "n_candidate_driver_domains", "candidate_driver_class"]] = pd.DataFrame(rows, index=out.index)
    return out


def derive_dominant_component(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy(); fields = ["delta_esspri_dryness", "delta_esspri_fatigue", "delta_esspri_pain"]
    results = []
    for _, row in out.iterrows():
        vals = [pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0] for c in fields]
        if any(pd.isna(v) for v in vals): results.append(("not_evaluable", np.nan)); continue
        absolute = np.abs(vals); maximum = max(absolute)
        if maximum == 0: results.append(("no_change", 0.0)); continue
        winners = np.flatnonzero(np.isclose(absolute, maximum))
        results.append(("tie", maximum) if len(winners) > 1 else (("dryness", "fatigue", "pain")[winners[0]], vals[winners[0]]))
    out[["dominant_esspri_component", "dominant_esspri_component_delta"]] = pd.DataFrame(results, index=out.index)
    return out


def summarize_domain_changes(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in data.loc[data.transition_family.ne("unevaluable")].groupby("transition_family"):
        types = ["incident"] if family == "to_pop1" else ["resolved"] if family == "from_pop1" else (["incident", "resolved"] if family in {"pop2_to_pop3", "pop3_to_pop2"} else [])
        for domain in DOMAINS:
            col = _domain_column(data, domain)
            if not col: continue
            for change in types:
                ev = group[col].notna() & group[col].ne("not_evaluable"); changed = ev & group[col].eq(change)
                for (a, b), exact in group.groupby(["from_pop", "to_pop"]):
                    ee = exact[col].notna() & exact[col].ne("not_evaluable"); cc = ee & exact[col].eq(change); n = int(ee.sum())
                    rows.append({"transition_family": family, "from_pop": a, "to_pop": b, "domain": domain, "change_type": change,
                                 "n_evaluable_episodes": n, "n_with_change": int(cc.sum()), "pct_with_change": 100*cc.sum()/n if n else np.nan,
                                 "n_unique_patients": exact.loc[cc, "patient_id"].nunique(), "median_interval_days": exact.loc[cc, "interval_days"].median()})
    result = pd.DataFrame(rows, columns=["transition_family","from_pop","to_pop","domain","change_type","n_evaluable_episodes","n_with_change","pct_with_change","n_unique_patients","median_interval_days"])
    if not result.empty: result["rank_within_transition"] = result.groupby(["transition_family", "from_pop", "to_pop", "change_type"])["pct_with_change"].rank(method="min", ascending=False)
    if "rank_within_transition" not in result: result["rank_within_transition"] = pd.Series(dtype=float)
    return result


def summarize_pro_changes(data: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (family, a, b), group in data.loc[data.transition_family.ne("unevaluable")].groupby(["transition_family", "from_pop", "to_pop"]):
        for measure in DELTAS:
            if measure not in group: continue
            x=pd.to_numeric(group[measure], errors="coerce").dropna(); n=len(x)
            rows.append({"transition_family":family,"from_pop":a,"to_pop":b,"transition_pair":f"{a} -> {b}","measure":measure.removeprefix("delta_"),"n_evaluable":n,
                         "median_delta":x.median(),"q1_delta":x.quantile(.25),"q3_delta":x.quantile(.75),"mean_delta":x.mean(),"sd_delta":x.std(),
                         "n_increased":int(x.gt(0).sum()),"n_decreased":int(x.lt(0).sum()),"n_unchanged":int(x.eq(0).sum()),
                         "pct_increased":100*x.gt(0).sum()/n if n else np.nan,"pct_decreased":100*x.lt(0).sum()/n if n else np.nan})
    return pd.DataFrame(rows, columns=["transition_family","from_pop","to_pop","transition_pair","measure","n_evaluable","median_delta","q1_delta","q3_delta","mean_delta","sd_delta","n_increased","n_decreased","n_unchanged","pct_increased","pct_decreased"])


def summarize_serology_profiles(data: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for family, group in data.groupby("transition_family"):
        for marker, flag in (("anti_ro_pos", "anti_ro_interpretable"), ("low_c4", "c4_interpretable")):
            if marker not in group: continue
            values=group[marker]; interpretable=values.notna() & (group[flag].eq(True) if flag in group else True)
            positive=interpretable & values.eq(True); n=int(interpretable.sum())
            rows.append({"transition_family":family,"serology_marker":marker,"n_total_episodes":len(group),"n_interpretable":n,"n_positive":int(positive.sum()),
                         "pct_positive":100*positive.sum()/n if n else np.nan,"n_unique_patients":group.loc[interpretable,"patient_id"].nunique(),
                         "denominator_definition":"Episodes with an interpretable baseline result","analysis_type":"baseline_serological_profile"})
    return pd.DataFrame(rows, columns=["transition_family","serology_marker","n_total_episodes","n_interpretable","n_positive","pct_positive","n_unique_patients","denominator_definition","analysis_type"])


def summarize_patient_profiles(data: pd.DataFrame) -> pd.DataFrame:
    continuous={"age_baseline","baseline_time_since_diagnosis_years","n_prespecified_comorbidities"}
    variables=["age_baseline","sex","protocol_group","baseline_pop","baseline_overlap_status","baseline_time_since_diagnosis_years","anti_ro_pos","low_c4","fibromyalgia","depression","anxiety","any_comorbidity","n_prespecified_comorbidities"]
    masks={"stable":data.transition_family.str.startswith("stable_"),"any_pop_change":data.event_any_pop_change.eq(True),
           **{x:data.transition_family.eq(x) for x in ("to_pop1","from_pop1","pop2_to_pop3","pop3_to_pop2")}}
    rows=[]
    for profile, mask in masks.items():
        g=data.loc[mask]
        for var in variables:
            if var not in g: continue
            base={"profile":profile,"variable":var,"n_episodes":len(g),"n_unique_patients":g.patient_id.nunique(),"level":"", "median":np.nan,"q1":np.nan,"q3":np.nan,"n_available":0,"n":np.nan,"denominator":np.nan,"percent":np.nan}
            if var in continuous:
                x=pd.to_numeric(g[var],errors="coerce").dropna(); base.update(median=x.median(),q1=x.quantile(.25),q3=x.quantile(.75),n_available=len(x)); rows.append(base)
            else:
                available=g[var].notna(); denom=int(available.sum())
                for level,n in g.loc[available,var].value_counts(dropna=False).items(): rows.append({**base,"level":level,"n_available":denom,"n":int(n),"denominator":denom,"percent":100*n/denom if denom else np.nan})
    return pd.DataFrame(rows)

def _skip_model(name: str, outcome: str, status: str, frame: pd.DataFrame) -> Dict[str, object]:
    return {**dict.fromkeys(MODEL_COLUMNS, np.nan), "model_name":name,"outcome":outcome,"n_episodes":len(frame),
            "n_patients":frame.patient_id.nunique() if "patient_id" in frame else 0,"n_events":int(frame.get(outcome,pd.Series(dtype=float)).eq(True).sum()),
            "model_status":status,"interpretation":INTERPRETATION}


def fit_profile_models(data: pd.DataFrame, min_events: int = 20, skip: bool = False) -> pd.DataFrame:
    specs=[("any_pop_change","event_any_pop_change",data.from_pop.isin(POPS)&data.to_pop.isin(POPS)),
           ("transition_to_pop1","event_transition_to_pop1",data.from_pop.isin(["Pop2","Pop3"])&data.to_pop.isin(POPS))]
    if skip: return pd.DataFrame([_skip_model(n,o,"skipped_by_user",data.loc[m]) for n,o,m in specs],columns=MODEL_COLUMNS)
    rows=[]; full=["age_baseline","sex","protocol_group","baseline_time_since_diagnosis_years","anti_ro_pos","low_c4","fibromyalgia","depression","n_prespecified_comorbidities","from_pop","interval_years"]
    fallback=["age_baseline","from_pop","interval_years","protocol_group"]
    for name,outcome,mask in specs:
        frame=data.loc[mask].copy(); events=int(frame[outcome].eq(True).sum()); nonevents=len(frame)-events
        if events<min_events or nonevents<min_events or frame.patient_id.nunique()<20:
            rows.append(_skip_model(name,outcome,"skipped_sparse_outcome_or_clusters",frame)); continue
        predictors=[c for c in full if c in frame]
        categorical=[c for c in ("sex","protocol_group","from_pop") if c in predictors]
        bad=any(frame[c].dropna().nunique()<2 for c in predictors) or any(frame[c].value_counts().min()<5 for c in categorical)
        if bad: predictors=[c for c in fallback if c in frame and frame[c].dropna().nunique()>1]
        complete=frame[[outcome,"patient_id",*predictors]].dropna()
        if not predictors or complete[outcome].sum()<min_events or (len(complete)-complete[outcome].sum())<min_events or complete.patient_id.nunique()<20:
            rows.append(_skip_model(name,outcome,"skipped_predictor_or_complete_case_sparsity",complete)); continue
        terms=[f"C({c})" if c in categorical else c for c in predictors]; formula=f"{outcome} ~ "+" + ".join(terms)
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            fit=smf.gee(formula,groups="patient_id",data=complete,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
            for term,coef in fit.params.items():
                se=fit.bse[term]; rows.append({"model_name":name,"outcome":outcome,"predictor":term.split("[")[0],"contrast":term,
                    "coefficient":coef,"standard_error":se,"odds_ratio":math.exp(coef),"ci95_low":math.exp(coef-1.96*se),"ci95_high":math.exp(coef+1.96*se),
                    "p_value":fit.pvalues[term],"n_episodes":len(complete),"n_patients":complete.patient_id.nunique(),"n_events":int(complete[outcome].sum()),
                    "formula":formula,"model_status":"fitted_reduced" if predictors!=[c for c in full if c in frame] else "fitted","interpretation":INTERPRETATION})
        except Exception as exc:
            row=_skip_model(name,outcome,f"skipped_model_failure: {type(exc).__name__}",complete); row["formula"]=formula; rows.append(row)
    return pd.DataFrame(rows,columns=MODEL_COLUMNS)


def build_first_to_last(data: pd.DataFrame) -> pd.DataFrame:
    rows=[]; measures=["essdai","esspri","pcs","mcs","profad","mdafs","n_extraglandular_domains"]
    for pid,g in data.groupby("patient_id",sort=False):
        g=g.sort_values(["from_date","to_date","episode_id"],kind="stable"); first=g.iloc[0]; last=g.iloc[-1]
        first_date=pd.to_datetime(first.from_date); last_date=pd.to_datetime(last.to_date); evaluable=first.from_pop in POPS and last.to_pop in POPS
        row={"patient_id":pid,"first_visit_id":first.from_visit_id,"last_visit_id":last.to_visit_id,"first_date":first_date,"last_date":last_date,
             "followup_days":(last_date-first_date).days,"followup_years":(last_date-first_date).days/365.25,"first_pop":first.from_pop,"last_pop":last.to_pop,
             "first_to_last_transition":f"{first.from_pop} -> {last.to_pop}","first_to_last_evaluable":evaluable,
             "first_to_last_changed":bool(evaluable and first.from_pop!=last.to_pop)}
        for m in measures:
            a=first.get(f"from_{m}",np.nan); b=last.get(f"to_{m}",np.nan); row[f"first_{m}"]=a; row[f"last_{m}"]=b
            row[f"net_delta_{m}"]=pd.to_numeric(pd.Series([b]),errors="coerce").iloc[0]-pd.to_numeric(pd.Series([a]),errors="coerce").iloc[0]
        for d in DOMAINS:
            fc=_domain_endpoint_column(data,d,"from"); lc=_domain_endpoint_column(data,d,"to"); a=_bool(first.get(fc)) if fc else None; b=_bool(last.get(lc)) if lc else None
            row[f"{d}_first_to_last_status"]="not_evaluable" if a is None or b is None else "incident" if not a and b else "resolved" if a and not b else "persistent_active" if a else "persistent_inactive"
        rows.append(row)
    result=pd.DataFrame(rows)
    if not result.empty and result.patient_id.duplicated().any(): raise ValueError("First-to-last construction produced more than one row per patient")
    return result


def _bool(value) -> Optional[bool]:
    if pd.isna(value): return None
    if isinstance(value,(bool,np.bool_)): return bool(value)
    if value in (0,1): return bool(value)
    text=str(value).lower().strip()
    if text in {"true","yes"}: return True
    if text in {"false","no"}: return False
    return None


def _domain_endpoint_column(data: pd.DataFrame, domain: str, side: str) -> Optional[str]:
    for c in (f"{side}_eg_{domain}_active",f"{side}_{domain}_active",f"{side}_{domain}"):
        if c in data: return c
    return None


def compare_transition_scales(data: pd.DataFrame, first_last: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    fl=first_last.loc[first_last.first_to_last_evaluable.eq(True)]
    summary=[]; comparison=[]
    for a in POPS:
        for b in POPS:
            cg=data.loc[data.from_pop.eq(a)&data.to_pop.eq(b)&data.transition_family.ne("unevaluable")]; cden=data.loc[data.from_pop.eq(a)&data.transition_family.ne("unevaluable")].shape[0]
            fg=fl.loc[fl.first_pop.eq(a)&fl.last_pop.eq(b)]; fden=fl.loc[fl.first_pop.eq(a)].shape[0]
            cp=100*len(cg)/cden if cden else np.nan; fp=100*len(fg)/fden if fden else np.nan
            summary.append({"from_pop":a,"to_pop":b,"transition_pair":f"{a} -> {b}","n_patients":len(fg),"pct_within_origin":fp})
            comparison.append({"transition_pair":f"{a} -> {b}","consecutive_n_episodes":len(cg),"consecutive_n_unique_patients":cg.patient_id.nunique(),
                               "consecutive_pct_within_origin":cp,"first_to_last_n_patients":len(fg),"first_to_last_pct_within_origin":fp,
                               "absolute_percentage_point_difference":abs(cp-fp) if pd.notna(cp) and pd.notna(fp) else np.nan})
    return pd.DataFrame(summary),pd.DataFrame(comparison)


def summarize_dominant(data: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    groups={"Pop2 -> Pop3":data.transition_pair.eq("Pop2 -> Pop3"),"Pop3 -> Pop2":data.transition_pair.eq("Pop3 -> Pop2"),
            "to_pop1":data.transition_family.eq("to_pop1"),"from_pop1":data.transition_family.eq("from_pop1")}
    for label,mask in groups.items():
        x=data.loc[mask,"dominant_esspri_component"]; denom=int(x.ne("not_evaluable").sum())
        for component in ["dryness","fatigue","pain","tie","no_change","not_evaluable"]:
            n=int(x.eq(component).sum()); rows.append({"transition_group":label,"dominant_esspri_component":component,"n":n,"denominator_evaluable":denom,"percent":100*n/denom if denom and component!="not_evaluable" else np.nan})
    return pd.DataFrame(rows)

def make_figures(domain: pd.DataFrame, pro: pd.DataFrame, comparison: pd.DataFrame, figure_dir: Path) -> None:
    families=["to_pop1","from_pop1","pop2_to_pop3","pop3_to_pop2"]
    fig,ax=plt.subplots(figsize=(8,6)); matrix=np.full((len(DOMAINS),4),np.nan); annotations=np.full(matrix.shape,"",dtype=object)
    for j,f in enumerate(families):
        preferred="incident" if f in {"to_pop1","pop2_to_pop3"} else "resolved"
        for i,d in enumerate(DOMAINS):
            x=domain.loc[(domain.transition_family==f)&(domain.domain==d)&(domain.change_type==preferred)]
            if len(x):
                n=x.n_evaluable_episodes.sum(); changed=x.n_with_change.sum(); matrix[i,j]=100*changed/n if n else np.nan; annotations[i,j]=f"{matrix[i,j]:.0f}%\nn={int(n)}" if n else ""
    cmap=plt.cm.Blues.copy(); cmap.set_bad("lightgray"); im=ax.imshow(np.ma.masked_invalid(matrix),aspect="auto",cmap=cmap,vmin=0,vmax=100)
    ax.set(xticks=range(4),xticklabels=families,yticks=range(len(DOMAINS)),yticklabels=DOMAINS,title="Domain changes during observed Pop transitions")
    for i in range(len(DOMAINS)):
        for j in range(4): ax.text(j,i,annotations[i,j],ha="center",va="center",fontsize=6)
    fig.colorbar(im,ax=ax,label="Domain-evaluable episodes (%)"); fig.tight_layout(); fig.savefig(figure_dir/"24_domain_drivers_heatmap.pdf"); plt.close(fig)

    shown=["esspri","esspri_dryness","esspri_fatigue","esspri_pain","pcs","mcs","profad","mdafs"]
    fig,axes=plt.subplots(4,2,figsize=(11,13));
    for ax,m in zip(axes.flat,shown):
        x=pro.loc[pro.measure.eq(m)]; labels=x.transition_pair.tolist(); pos=np.arange(len(x)); ax.errorbar(pos,x.median_delta,yerr=[x.median_delta-x.q1_delta,x.q3_delta-x.median_delta],fmt="o"); ax.axhline(0,color="gray",lw=.8); ax.set_title(m); ax.set_xticks(pos); ax.set_xticklabels(labels,rotation=60,ha="right",fontsize=7); ax.set_ylabel("Median change (IQR)")
    fig.suptitle("Descriptive PRO changes by observed transition"); fig.tight_layout(); fig.savefig(figure_dir/"24_pro_changes_by_transition.pdf"); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,prefix,title in ((axes[0],"consecutive","Consecutive episodes"),(axes[1],"first_to_last","First-to-last patients")):
        mat=np.full((3,3),np.nan)
        for i,a in enumerate(POPS):
            for j,b in enumerate(POPS):
                r=comparison.loc[comparison.transition_pair.eq(f"{a} -> {b}")].iloc[0]; pct=r[f"{prefix}_pct_within_origin"]; n=r["consecutive_n_episodes" if prefix=="consecutive" else "first_to_last_n_patients"]; mat[i,j]=pct; ax.text(j,i,f"{int(n)}\n{pct:.1f}%" if pd.notna(pct) else f"{int(n)}\nNA",ha="center",va="center")
        ax.imshow(np.ma.masked_invalid(mat),vmin=0,vmax=100,cmap="Blues"); ax.set(xticks=range(3),xticklabels=POPS,yticks=range(3),yticklabels=POPS,xlabel="Destination",ylabel="Origin",title=title)
    fig.tight_layout(); fig.savefig(figure_dir/"24_consecutive_vs_first_last_transition_matrices.pdf"); plt.close(fig)


def _planned_outputs() -> List[Path]:
    names=["24_transition_mechanism_summary.csv","24_domain_drivers_by_transition.csv","24_pro_changes_by_transition.csv","24_dominant_symptom_component.csv","24_serology_profiles_by_transition.csv","24_patient_profiles_by_transition.csv","24_transition_profile_models.csv","24_first_to_last_transition_summary.csv","24_consecutive_vs_first_last.csv"]
    return [common.INTERMEDIATE_DATA_DIR/"24_transition_mechanisms"/f"24_transition_mechanism_episode.{x}" for x in ("parquet","csv")]+[common.INTERMEDIATE_DATA_DIR/"24_transition_mechanisms"/f"24_first_to_last_transition_patient.{x}" for x in ("parquet","csv")]+[common.BLOCKB_TABLES_DIR/"24_transition_mechanisms"/x for x in names]+[common.BLOCKB_FIGURES_DIR/"24_transition_mechanisms"/x for x in ("24_domain_drivers_heatmap.pdf","24_pro_changes_by_transition.pdf","24_consecutive_vs_first_last_transition_matrices.pdf")]+[common.BLOCKB_QC_DIR/"24_transition_mechanisms"/"24_transition_mechanisms_qc.json",common.BLOCKB_LOGS_DIR/"24_transition_mechanisms"/"24_transition_mechanisms.log"]


def main(argv: Optional[Sequence[str]] = None) -> None:
    args=parse_args(argv); planned=_planned_outputs()
    if not args.overwrite:
        existing=[str(p) for p in planned if p.exists()]
        if existing: raise FileExistsError("Refusing to overwrite planned outputs: "+", ".join(existing))
    episodes=derive_dominant_component(derive_candidate_drivers(classify_transition_family(validate_episodes(load_episodes(args.input)))))
    optional=set(DELTAS+[f"eg_{d}_active_transition" for d in DOMAINS]+["anti_ro_pos","low_c4"]); missing=sorted(optional-set(episodes.columns)); warnings=[f"Optional column unavailable: {c}" for c in missing]
    domain=summarize_domain_changes(episodes); pro=summarize_pro_changes(episodes); serology=summarize_serology_profiles(episodes); profiles=summarize_patient_profiles(episodes)
    models=fit_profile_models(episodes,args.min_model_events,args.skip_models); first_last=build_first_to_last(episodes); flsummary,comparison=compare_transition_scales(episodes,first_last); dominant=summarize_dominant(episodes)
    useful=[c for c in REQUIRED+["transition_pair","transition_family","candidate_driver_domains","n_candidate_driver_domains","candidate_driver_class","dominant_esspri_component","dominant_esspri_component_delta","event_any_pop_change","event_transition_to_pop1"]+DELTAS+[f"from_{m}" for m in ("essdai","esspri","pcs","mcs","profad","mdafs","n_extraglandular_domains")]+[f"to_{m}" for m in ("essdai","esspri","pcs","mcs","profad","mdafs","n_extraglandular_domains")]+[f"from_eg_{d}_active" for d in DOMAINS]+[f"to_eg_{d}_active" for d in DOMAINS] if c in episodes]
    mechanism=episodes[useful].copy()
    if len(mechanism)!=len(episodes): raise ValueError("Intermediate mechanism output changed input episode count")
    summary=episodes.groupby(["transition_family","from_pop","to_pop"],dropna=False).agg(n_episodes=("episode_id","size"),n_unique_patients=("patient_id","nunique"),median_interval_days=("interval_days","median")).reset_index()
    for p in {x.parent for x in planned}: p.mkdir(parents=True,exist_ok=True)
    mechanism.to_parquet(common.INTERMEDIATE_DATA_DIR/"24_transition_mechanisms"/"24_transition_mechanism_episode.parquet",index=False); mechanism.to_csv(common.INTERMEDIATE_DATA_DIR/"24_transition_mechanisms"/"24_transition_mechanism_episode.csv",index=False)
    first_last.to_parquet(common.INTERMEDIATE_DATA_DIR/"24_transition_mechanisms"/"24_first_to_last_transition_patient.parquet",index=False); first_last.to_csv(common.INTERMEDIATE_DATA_DIR/"24_transition_mechanisms"/"24_first_to_last_transition_patient.csv",index=False)
    tables={"transition_mechanism_summary":summary,"domain_drivers_by_transition":domain,"pro_changes_by_transition":pro,"dominant_symptom_component":dominant,"serology_profiles_by_transition":serology,"patient_profiles_by_transition":profiles,"transition_profile_models":models,"first_to_last_transition_summary":flsummary,"consecutive_vs_first_last":comparison}
    for name,frame in tables.items(): frame.to_csv(common.BLOCKB_TABLES_DIR/"24_transition_mechanisms"/f"24_{name}.csv",index=False)
    make_figures(domain,pro,comparison,common.BLOCKB_FIGURES_DIR/"24_transition_mechanisms")
    longitudinal={"from_anti_ro_pos","to_anti_ro_pos","from_low_c4","to_low_c4"}.issubset(episodes)
    qc={"input_path":str(args.input),"n_input_episodes":len(episodes),"n_unique_patients":episodes.patient_id.nunique(),"n_evaluable_pop_episodes":int(episodes.transition_family.ne("unevaluable").sum()),"n_unevaluable_pop_episodes":int(episodes.transition_family.eq("unevaluable").sum()),"n_changed_pop_episodes":int(episodes.event_any_pop_change.sum()),
        **{f"n_{f}":int(episodes.transition_family.eq(f).sum()) for f in ("to_pop1","from_pop1","pop2_to_pop3","pop3_to_pop2")},"n_domain_evaluable_episodes":int(episodes.candidate_driver_class.ne("not_evaluable").sum()),"n_pro_evaluable_episodes":int(episodes[[c for c in DELTAS if c in episodes]].notna().any(axis=1).sum()) if any(c in episodes for c in DELTAS) else 0,
        "n_first_to_last_patients":len(first_last),"n_first_to_last_evaluable":int(first_last.first_to_last_evaluable.sum()),"longitudinal_serology_change_status":"available_visit_aligned" if longitudinal else "unavailable_not_visit_aligned","models_requested":not args.skip_models,"n_models_fitted":int(models.model_status.astype(str).str.startswith("fitted").groupby(models.model_name).any().sum()),"n_models_skipped":int((~models.model_status.astype(str).str.startswith("fitted")).groupby(models.model_name).all().sum()),"missing_optional_columns":missing,"warnings":warnings}
    (common.BLOCKB_QC_DIR/"24_transition_mechanisms"/"24_transition_mechanisms_qc.json").write_text(json.dumps(qc,indent=2,default=str)+"\n")
    log=[f"Input loaded: {args.input}",f"Episodes: {len(episodes)}; patients: {episodes.patient_id.nunique()}",f"Transition families: {episodes.transition_family.value_counts().to_dict()}","Domain and PRO analyses completed",f"First-to-last patients: {len(first_last)}",f"Model status: {models.model_status.value_counts().to_dict()}",f"Longitudinal serology: {qc['longitudinal_serology_change_status']}",*warnings,"Outputs written"]
    (common.BLOCKB_LOGS_DIR/"24_transition_mechanisms"/"24_transition_mechanisms.log").write_text("\n".join(log)+"\n")

if __name__ == "__main__": main()

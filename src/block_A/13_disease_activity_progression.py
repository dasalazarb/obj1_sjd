#!/usr/bin/env python3
"""Canonical Block A analysis of observed disease-activity progression.

Progression is characterized by continuous trajectories and transitions among
ESSDAI Low, Moderate, and High states. Kaplan--Meier analyses use the date on
which worsening is first observed: the biological transition is interval
censored between the last negative and first positive assessment.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import common  # noqa: E402
from config import (CHANGE_THRESHOLDS, ESSDAI_HIGH_CUTOFF,
                    ESSDAI_MODERATE_OR_HIGH_CUTOFF)  # noqa: E402

LOG = logging.getLogger("disease_activity_progression")
STEM = "13_disease_activity_progression"
KEYS = ["patient_id", "clinical_episode_id"]
REQUIRED = KEYS + ["clinical_anchor_date", "clinical_visit_number", "clinical_visit",
    "clinical_baseline_episode_id", "clinical_baseline_date", "is_clinical_baseline",
    "time_since_clinical_baseline_days", "time_since_clinical_baseline_years"]
ESSDAI = "essdai__total"
ESSPRI = {"total": "esspri__total", "dryness": "esspri__dryness",
          "fatigue": "esspri__fatigue", "pain": "esspri__pain"}
DOMAINS = ("constitutional", "lymphadenopathy", "glandular", "articular",
           "cutaneous", "pulmonary", "renal", "muscular", "pns", "cns",
           "hematologic", "biological")
DOMAIN_ALIASES = {
    "constitutional": ("essdai__constitutional_ordinal_score", "eg_constitutional_ordinal_score"),
    "lymphadenopathy": ("essdai__lymphadenopathy_ordinal_score", "eg_lymphadenopathy_ordinal_score"),
    "glandular": ("essdai__glandular_ordinal_score", "eg_glandular_ordinal_score"),
    "articular": ("essdai__articular_ordinal_score", "eg_articular_ordinal_score"),
    "cutaneous": ("essdai__cutaneous_ordinal_score", "eg_cutaneous_ordinal_score"),
    "pulmonary": ("essdai__pulmonary_ordinal_score", "eg_pulmonary_ordinal_score"),
    "renal": ("essdai__renal_ordinal_score", "eg_renal_ordinal_score"),
    "muscular": ("essdai__muscular_ordinal_score", "eg_muscular_ordinal_score"),
    "pns": ("essdai__pns_ordinal_score", "eg_pns_ordinal_score", "eg_neuro_peripheral_ordinal_score"),
    "cns": ("essdai__cns_ordinal_score", "eg_cns_ordinal_score"),
    "hematologic": ("essdai__hematologic_ordinal_score", "eg_hematologic_ordinal_score"),
    "biological": ("essdai__biological_ordinal_score", "eg_biological_ordinal_score"),
}
MIN_REPEATED_PATIENTS = 3
MIN_STATE_CHANGES = 5
STATE_ORDER = ("Low", "Moderate", "High")
STATE_RANK = {x: i for i, x in enumerate(STATE_ORDER)}
LANDMARKS = (0, .5, 1, 2, 3, 5)
EVENT_NOTE = ("First-observed-event analysis: state change occurs between the last "
              "non-event visit and the first event-positive visit; conventional KM "
              "uses the latter as observed event time.")


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, low_memory=False)


def classify_essdai_activity(score: object) -> str:
    """Classify an ESSDAI score without conflating activity and Pop states."""
    if pd.isna(score): return "Missing"
    value = float(score)
    if value < ESSDAI_MODERATE_OR_HIGH_CUTOFF: return "Low"
    if value < ESSDAI_HIGH_CUTOFF: return "Moderate"
    return "High"


def classify_essdai_series(values: pd.Series) -> pd.Series:
    return values.map(classify_essdai_activity).astype(pd.CategoricalDtype([*STATE_ORDER, "Missing"], ordered=True))


def baseline_to_last_status(baseline_state: str, last_state: str) -> str:
    if baseline_state not in STATE_RANK or last_state not in STATE_RANK: return "not_evaluable"
    delta = STATE_RANK[last_state] - STATE_RANK[baseline_state]
    return "worsened" if delta > 0 else "improved" if delta < 0 else "stable"


def validate_inputs(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED + [ESSDAI]) - set(df))
    if missing: raise KeyError(f"Integrated dataset lacks required columns: {missing}")
    x = df.copy()
    x["patient_id"] = x.patient_id.astype("string")
    if x[KEYS].isna().any().any() or x.duplicated(KEYS).any():
        raise ValueError("duplicate or missing patient_id + clinical_episode_id")
    for col in ("clinical_anchor_date", "clinical_baseline_date"):
        x[col] = pd.to_datetime(x[col], errors="coerce")
    x["time_since_clinical_baseline_years"] = pd.to_numeric(x.time_since_clinical_baseline_years, errors="coerce")
    x[ESSDAI] = pd.to_numeric(x[ESSDAI], errors="coerce")
    if x.time_since_clinical_baseline_years.dropna().lt(0).any(): raise ValueError("negative time since baseline")
    bases = x.loc[x.is_clinical_baseline.eq(True)]
    if bases.patient_id.duplicated().any(): raise ValueError("multiple clinical baselines per patient")
    if not bases.clinical_episode_id.astype("string").eq(bases.clinical_baseline_episode_id.astype("string")).all():
        raise ValueError("baseline episode inconsistency")
    if not bases.clinical_anchor_date.eq(bases.clinical_baseline_date).all(): raise ValueError("baseline date inconsistency")
    if (~x[ESSDAI].dropna().between(0, 123)).any(): raise ValueError("ESSDAI outside allowed range 0-123")
    for col in ESSPRI.values():
        if col in x:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            if (~x[col].dropna().between(0, 10)).any(): raise ValueError(f"{col} outside 0-10")
    baseline_ids = baseline.patient_id.astype("string")
    absent = set(baseline_ids.dropna()) - set(x.patient_id.dropna())
    if absent: raise ValueError(f"{len(absent)} canonical baseline patients absent from longitudinal input")
    return x.sort_values(["patient_id", "clinical_anchor_date", "clinical_episode_id"], kind="stable")


def patient_progression(df: pd.DataFrame) -> pd.DataFrame:
    """Patient endpoints anchored only at the official clinical baseline."""
    rows = []
    for pid, g in df.groupby("patient_id", sort=False):
        b = g.loc[g.is_clinical_baseline.eq(True)]
        if len(b) != 1 or pd.isna(b.iloc[0][ESSDAI]): continue
        b = b.iloc[0]; after = g.loc[(g.clinical_anchor_date > b.clinical_anchor_date) & g[ESSDAI].notna()]
        if after.empty: continue
        last = after.iloc[-1]; bs, ls = classify_essdai_activity(b[ESSDAI]), classify_essdai_activity(last[ESSDAI])
        ranks = after[ESSDAI].map(classify_essdai_activity).map(STATE_RANK)
        br = STATE_RANK[bs]
        rows.append({"patient_id": pid, "baseline_essdai": b[ESSDAI], "last_essdai": last[ESSDAI],
            "baseline_state": bs, "last_state": ls, "delta_essdai": last[ESSDAI]-b[ESSDAI],
            "status": baseline_to_last_status(bs, ls),
            "followup_years": (last.clinical_anchor_date-b.clinical_anchor_date).days/365.25,
            "ever_worsened": bool((ranks > br).any()), "ever_improved": bool((ranks < br).any()),
            "maximum_state_reached": STATE_ORDER[int(max([br, *ranks.dropna().tolist()]))],
            "minimum_state_reached": STATE_ORDER[int(min([br, *ranks.dropna().tolist()]))]})
    return pd.DataFrame(rows)


def build_state_transitions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use adjacent clinical episodes only; missing middle ESSDAI breaks a chain."""
    intervals=[]
    for pid, g in df.groupby("patient_id", sort=False):
        g=g.sort_values(["clinical_anchor_date", "clinical_episode_id"], kind="stable").reset_index(drop=True)
        for i in range(1, len(g)):
            a,b=g.iloc[i-1],g.iloc[i]
            if pd.isna(a[ESSDAI]) or pd.isna(b[ESSDAI]): continue
            years=(b.clinical_anchor_date-a.clinical_anchor_date).days/365.25
            if years <= 0: continue
            intervals.append({"patient_id":pid,"from_state":classify_essdai_activity(a[ESSDAI]),
                "to_state":classify_essdai_activity(b[ESSDAI]),"interval_years":years})
    raw=pd.DataFrame(intervals)
    rows=[]
    for fr in STATE_ORDER:
        denom=int((raw.from_state.eq(fr)).sum()) if not raw.empty else 0
        for to in STATE_ORDER:
            z=raw.loc[raw.from_state.eq(fr)&raw.to_state.eq(to)] if not raw.empty else raw
            rows.append({"from_state":fr,"to_state":to,"n_intervals":len(z),
                "n_patients":z.patient_id.nunique() if len(z) else 0,"row_total":denom,
                "row_pct":100*len(z)/denom if denom else np.nan,
                "median_interval_years":z.interval_years.median() if len(z) else np.nan})
    return pd.DataFrame(rows),raw


def build_tte_risk_set(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    """Construct one first-observed-event/censoring row per eligible patient."""
    if origin not in {"low", "moderate"}: raise ValueError("origin must be low or moderate")
    rows=[]
    for pid,g in df.groupby("patient_id", sort=False):
        b=g.loc[g.is_clinical_baseline.eq(True)]
        if len(b)!=1 or pd.isna(b.iloc[0][ESSDAI]): continue
        b=b.iloc[0]; score=float(b[ESSDAI])
        eligible=score < 5 if origin=="low" else 5 <= score < 14
        if not eligible: continue
        after=g.loc[(g.clinical_anchor_date>b.clinical_anchor_date)&g[ESSDAI].notna()].sort_values("clinical_anchor_date")
        if after.empty: continue
        positive=after[ESSDAI].ge(5 if origin=="low" else 14)
        event=bool(positive.any()); endpoint=after.loc[positive].iloc[0] if event else after.iloc[-1]
        prior=after.loc[after.clinical_anchor_date < endpoint.clinical_anchor_date]
        left=(prior.iloc[-1].clinical_anchor_date-b.clinical_anchor_date).days/365.25 if event and len(prior) else 0.0
        time=(endpoint.clinical_anchor_date-b.clinical_anchor_date).days/365.25
        rows.append({"patient_id":pid,"baseline_essdai":score,"event":event,"time_years":time,
            "event_or_censor_date":endpoint.clinical_anchor_date,"last_non_event_time_years":left,
            "event_score":endpoint[ESSDAI] if event else np.nan,"analysis":"first-observed-event"})
    out=pd.DataFrame(rows)
    if not out.empty and (out.time_years<=0).any(): raise AssertionError("KM event/censor time must be >0")
    return out


def km_estimate(tte: pd.DataFrame, landmarks: Iterable[float]=(1,2,5)) -> tuple[pd.DataFrame, dict]:
    if tte.empty:
        return pd.DataFrame(columns=["time","n_at_risk","n_events","survival"]), {f"event_free_{y}y":np.nan for y in landmarks}
    surv=1.; rows=[]
    for t in sorted(tte.time_years.unique()):
        risk=int((tte.time_years>=t).sum()); events=int(((tte.time_years==t)&tte.event).sum())
        if events: surv*=1-events/risk
        rows.append({"time":t,"n_at_risk":risk,"n_events":events,"survival":surv})
    curve=pd.DataFrame(rows)
    probs={}
    for y in landmarks:
        prior=curve.loc[curve.time<=y]
        probs[f"event_free_{y}y"]=float(prior.iloc[-1].survival) if len(prior) else 1.
    reached=curve.loc[curve.survival<=.5]
    probs["median_time_to_event"]=float(reached.iloc[0].time) if len(reached) else "NR"
    return curve,probs


def fit_continuous_model(data: pd.DataFrame, outcome: str, label: str) -> tuple[dict, object|None]:
    d=data[["patient_id","time_since_clinical_baseline_years",outcome]].dropna().rename(columns={outcome:"y","time_since_clinical_baseline_years":"time"})
    d=d.loc[d.time>=0]
    # The primary longitudinal cohort requires two measurements on distinct
    # dates; a subject with one observation must not influence the fit.
    eligible = d.groupby("patient_id").time.nunique().loc[lambda x: x >= 2].index
    d=d.loc[d.patient_id.isin(eligible)]
    result=None; used="MixedLM"; fallback=False; warning_text=""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result=MixedLM.from_formula("y ~ time", groups="patient_id", re_formula="1", data=d).fit(reml=False, method="lbfgs", disp=False)
        converged=bool(result.converged); warning_text="; ".join(str(w.message) for w in caught)
        if not converged: raise RuntimeError("MixedLM did not converge")
    except Exception as exc:
        fallback=True; used="GEE Gaussian"; warning_text=f"MixedLM failed: {exc}"
        try:
            result=GEE.from_formula("y ~ time", groups="patient_id", data=d, family=Gaussian(), cov_struct=Exchangeable()).fit()
            converged=True
        except Exception as exc2:
            result=None; converged=False; warning_text += f"; GEE failed: {exc2}"
    est=se=p=lo=hi=np.nan
    if result is not None:
        est=float(result.params["time"]); se=float(result.bse["time"]); p=float(result.pvalues["time"]); lo=est-1.96*se; hi=est+1.96*se
    spans=d.groupby("patient_id").time.agg(lambda s:s.max()-s.min())
    row={"outcome":label,"model":used,"model_used":used,"n_patients":d.patient_id.nunique(),"n_observations":len(d),
         "annual_change":est,"annual_change_estimate":est,"ci95_low":lo,"ci95_high":hi,"p_value":p,
         "median_followup_years":spans.median() if len(spans) else np.nan,"model_status":"ok" if converged else "failed",
         "converged":converged,"singular":("singular" in warning_text.lower()),"fallback_used":fallback,"warning":warning_text,
         "time_variation":d.time.nunique()}
    return row,result


def bh_fdr(values: pd.Series) -> pd.Series:
    out=pd.Series(np.nan,index=values.index,dtype=float); valid=values.dropna(); m=len(valid)
    if not m:return out
    order=valid.sort_values().index; ranked=valid.loc[order].to_numpy()*m/np.arange(1,m+1)
    adjusted=np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1); out.loc[order]=adjusted
    return out


def resolve_domains(df: pd.DataFrame) -> dict[str,str]:
    resolved={}
    for domain in DOMAINS:
        matches=[c for c in DOMAIN_ALIASES[domain] if c in df]
        if len(matches)>1: raise ValueError(f"Ambiguous ordinal columns for {domain}: {matches}")
        if matches: resolved[domain]=matches[0]
    return resolved


def domain_analyses(df: pd.DataFrame, resolved: dict[str,str]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    models=[]; changes=[]; support=[]
    for domain in DOMAINS:
        col=resolved.get(domain); d=df[["patient_id","time_since_clinical_baseline_years","is_clinical_baseline"]+([col] if col else [])].copy()
        if not col:
            support.append({"domain":domain,"column":pd.NA,"n_patients":0,"n_patients_repeated":0,"n_observations":0,"n_nonzero":0,"n_state_changes":0,"n_unique_levels":0})
            models.append({"domain":domain,"outcome_type":"ordinal","model":"none","model_status":"column_unavailable","fallback_used":False})
            continue
        d[col]=pd.to_numeric(d[col],errors="coerce"); z=d.dropna(subset=[col,"time_since_clinical_baseline_years"])
        if (~z[col].isin([0,1,2,3])).any(): raise ValueError(f"{col} outside ordinal range 0-3")
        counts=z.groupby("patient_id").size(); changes_n=int(z.sort_values(["patient_id","time_since_clinical_baseline_years"]).groupby("patient_id")[col].apply(lambda x:x.diff().ne(0).iloc[1:].sum()).sum())
        sup={"domain":domain,"column":col,"n_patients":z.patient_id.nunique(),"n_patients_repeated":int(counts.ge(2).sum()),"n_observations":len(z),"n_nonzero":int(z[col].gt(0).sum()),"n_state_changes":changes_n,"n_unique_levels":z[col].nunique()};support.append(sup)
        row={**sup,"outcome_type":"ordinal","model":"descriptive","effect":np.nan,"effect_scale":"descriptive_only","ci95_low":np.nan,"ci95_high":np.nan,"p_value":np.nan,"model_status":"insufficient_support","fallback_used":False,"interpretation_note":"Ordinal activity over observed follow-up"}
        if sup["n_patients_repeated"]>=MIN_REPEATED_PATIENTS and changes_n>=MIN_STATE_CHANGES and sup["n_unique_levels"]>=2:
            try:
                # OrderedModel provides the primary ordinal trend; clustered binary GEE is a documented fallback.
                fit=OrderedModel(z[col].astype(int), z[["time_since_clinical_baseline_years"]], distr="logit").fit(method="bfgs",disp=False)
                key="time_since_clinical_baseline_years"; e=float(fit.params[key]); se=float(fit.bse[key])
                row.update(model="ordinal logistic",effect=e,effect_scale="ordinal log-odds per year",ci95_low=e-1.96*se,ci95_high=e+1.96*se,p_value=float(fit.pvalues[key]),model_status="ok")
            except Exception as exc:
                try:
                    q=z.assign(active=z[col].gt(0).astype(int),time=z.time_since_clinical_baseline_years)
                    fit=GEE.from_formula("active ~ time",groups="patient_id",data=q,family=Binomial(),cov_struct=Exchangeable()).fit();e=float(fit.params.time);se=float(fit.bse.time)
                    row.update(outcome_type="binary",model="GEE binomial",effect=float(np.exp(e)),effect_scale="odds ratio active vs inactive per year",ci95_low=float(np.exp(e-1.96*se)),ci95_high=float(np.exp(e+1.96*se)),p_value=float(fit.pvalues.time),model_status="ok",fallback_used=True,interpretation_note=f"Ordinal model failed ({exc}); binary fallback")
                except Exception as exc2: row["interpretation_note"]=f"Models failed: {exc}; {exc2}"
        models.append(row)
        paired=[]
        for pid,g in z.groupby("patient_id"):
            b=g.loc[g.is_clinical_baseline.eq(True),col]
            after=g.loc[~g.is_clinical_baseline.eq(True)].sort_values("time_since_clinical_baseline_years")
            if len(b)==1 and len(after): paired.append((float(b.iloc[0]),float(after.iloc[-1][col])))
        delta=np.array([b-a for a,b in paired])
        changes.append({"domain":domain,"n_paired_patients":len(paired),"baseline_median":np.median([a for a,b in paired]) if paired else np.nan,"last_median":np.median([b for a,b in paired]) if paired else np.nan,"median_delta":np.median(delta) if len(delta) else np.nan,"improved_one_or_more_levels":int((delta<0).sum()),"stable":int((delta==0).sum()),"worsened_one_or_more_levels":int((delta>0).sum())})
    models=pd.DataFrame(models); models["q_value"]=bh_fdr(models.get("p_value",pd.Series(dtype=float)))
    return models,pd.DataFrame(changes),pd.DataFrame(support)


def availability(df: pd.DataFrame, measures: dict[str,str]) -> pd.DataFrame:
    rows=[]
    for name,col in measures.items():
        if col not in df: rows.append({"measure":name,"n_observations":0,"n_patients_any":0,"n_patients_repeated":0});continue
        z=df.loc[df[col].notna(),["patient_id","time_since_clinical_baseline_years"]]
        counts=z.groupby("patient_id").size(); spans=z.groupby("patient_id").time_since_clinical_baseline_years.agg(lambda s:s.max()-s.min())
        rows.append({"measure":name,"n_observations":len(z),"n_patients_any":z.patient_id.nunique(),"n_patients_repeated":int(counts.ge(2).sum()),"median_measurements_per_patient":counts.median(),"median_observed_span_years":spans.median(),"q1_observed_span_years":spans.quantile(.25),"q3_observed_span_years":spans.quantile(.75)})
    return pd.DataFrame(rows)


def plot_trajectory(df,outcome,path,title):
    z=df.dropna(subset=[outcome,"time_since_clinical_baseline_years"]); fig,ax=plt.subplots(figsize=(8,5))
    for _,g in z.groupby("patient_id"): ax.plot(g.time_since_clinical_baseline_years,g[outcome],color="0.75",alpha=.25,lw=.6)
    ax.scatter(z.time_since_clinical_baseline_years,z[outcome],s=8,alpha=.25)
    if len(z)>=2:
        coef=np.polyfit(z.time_since_clinical_baseline_years,z[outcome],1); xx=np.linspace(0,z.time_since_clinical_baseline_years.max(),100); ax.plot(xx,np.polyval(coef,xx),color="#9c2f45",lw=2)
    ax.set(xlabel="Years since clinical baseline",ylabel=title,title=f"{title} trajectory\nn={z.patient_id.nunique()} patients; {len(z)} observations");fig.tight_layout();fig.savefig(path);plt.close(fig)


def plot_km(tte,path,title):
    curve,_=km_estimate(tte);fig,ax=plt.subplots(figsize=(7,5));
    if len(curve): ax.step([0,*curve.time],[1,*curve.survival],where="post")
    ax.set(xlabel="Years since clinical baseline",ylabel="Event-free probability",ylim=(0,1.02),title=title);ax.text(.01,.02,EVENT_NOTE,transform=ax.transAxes,fontsize=7,wrap=True);fig.tight_layout();fig.savefig(path);plt.close(fig)


def save_heatmap(matrix,path,title,fmt=".1f"):
    fig,ax=plt.subplots(figsize=(7,5));a=np.asarray(matrix,dtype=float);im=ax.imshow(a,aspect="auto",cmap="Blues",vmin=0);fig.colorbar(im,ax=ax)
    ax.set(xticks=range(matrix.shape[1]),xticklabels=matrix.columns,yticks=range(matrix.shape[0]),yticklabels=matrix.index,title=title)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if np.isfinite(a[i,j]): ax.text(j,i,format(a[i,j],fmt),ha="center",va="center",fontsize=8)
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--integrated",type=Path,default=common.INTEGRATED_LONGITUDINAL_PARQUET);p.add_argument("--baseline",type=Path,default=common.INTEGRATED_BASELINE_PARQUET);p.add_argument("--output-root",type=Path,default=common.PROJECT_ROOT);p.add_argument("--overwrite",action=argparse.BooleanOptionalAction,default=False);p.add_argument("--dry-run",action="store_true");return p.parse_args()


def main(args=None):
    args=parse_args() if args is None else args
    tables=args.output_root/"outputs/tables/blockA"/STEM; figures=args.output_root/"outputs/figures/blockA"/STEM; qc=args.output_root/"outputs/qc/blockA"/STEM; logs=args.output_root/"outputs/logs"/STEM; analytic=args.output_root/"data/analytic/blockA"/STEM
    for d in (tables,figures,qc,logs,analytic): d.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.FileHandler(logs/f"{STEM}.log"),logging.StreamHandler()])
    LOG.info("Inputs integrated=%s baseline=%s",args.integrated,args.baseline)
    df=validate_inputs(read_table(args.integrated),read_table(args.baseline)); baseline=read_table(args.baseline); resolved=resolve_domains(df)
    df["essdai_activity_state"]=classify_essdai_series(df[ESSDAI]);df["essdai_activity_rank"]=df.essdai_activity_state.map({**STATE_RANK,"Missing":np.nan}).astype("Float64")
    df["previous_essdai"]=df.groupby("patient_id")[ESSDAI].shift();df["delta_essdai_from_previous"]=df[ESSDAI]-df.previous_essdai;df["previous_essdai_state"]=df.groupby("patient_id").essdai_activity_state.shift();df["state_change_from_previous"]=df.essdai_activity_rank-df.groupby("patient_id").essdai_activity_rank.shift()
    measures={"ESSDAI total":ESSDAI,**{f"ESSPRI {k}":v for k,v in ESSPRI.items()},**{f"ESSDAI domain {k}":v for k,v in resolved.items()}}
    avail=availability(df,measures)
    prog=patient_progression(df); transitions,raw_intervals=build_state_transitions(df);low=build_tte_risk_set(df,"low");moderate=build_tte_risk_set(df,"moderate")
    base_rows=df.loc[df.is_clinical_baseline.eq(True)];counts=df.loc[df[ESSDAI].notna()].groupby("patient_id").size()
    flow=[("Overall integrated baseline cohort",baseline.patient_id.nunique()),("Patients with >=1 ESSDAI",counts.size),("Patients with >=2 ESSDAI on distinct dates",sum(df.loc[df[ESSDAI].notna()].groupby("patient_id").clinical_anchor_date.nunique()>=2)),("Patients with >=3 ESSDAI",sum(counts>=3))]
    for state in STATE_ORDER: flow.append((f"Patients with baseline ESSDAI {state}",(base_rows[ESSDAI].map(classify_essdai_activity)==state).sum()))
    for k,col in ESSPRI.items(): flow.append((f"Patients with >=2 ESSPRI {k}",int(df.loc[df.get(col,pd.Series(index=df.index,dtype=float)).notna()].groupby("patient_id").clinical_anchor_date.nunique().ge(2).sum())))
    cohort=pd.DataFrame(flow,columns=["cohort","n_patients"])
    attr=[]
    for name,col in {"ESSDAI":ESSDAI,**ESSPRI}.items():
        n=baseline.patient_id.nunique();c=df.loc[df.get(col,pd.Series(index=df.index,dtype=float)).notna()].groupby("patient_id").size()
        attr += [{"outcome":name,"reason":"eligible_baseline_cohort","n":n},{"outcome":name,"reason":"excluded_no_outcome","n":n-len(c)},{"outcome":name,"reason":"excluded_only_one_measure","n":int(c.eq(1).sum())},{"outcome":name,"reason":"excluded_same_date_duplicates","n":int(df.loc[df.get(col,pd.Series(index=df.index,dtype=float)).notna()].duplicated(["patient_id","clinical_anchor_date"],False).sum())},{"outcome":name,"reason":"excluded_negative_or_invalid_time","n":int(df.time_since_clinical_baseline_years.isna().sum())},{"outcome":name,"reason":"final_modeled_cohort","n":int(c.ge(2).sum())}]
    if args.dry_run:
        print(cohort.to_string(index=False));print("\n",avail.to_string(index=False));LOG.info("Dry run complete; no definitive outputs written");return
    cohort.to_csv(tables/"13_progression_cohort_flow.csv",index=False);avail.to_csv(tables/"13_progression_measure_availability.csv",index=False);pd.DataFrame(attr).to_csv(tables/"13_progression_attrition.csv",index=False)
    primary,fit=fit_continuous_model(df,ESSDAI,"ESSDAI total");pd.DataFrame([primary]).to_csv(tables/"13_essdai_longitudinal_model.csv",index=False)
    sensitivity=pd.DataFrame([{"analysis":"random_slope","model_status":"not_run_unless_stable_support","eligible_patients":int(counts.ge(3).sum()),"note":"Reserved sensitivity; primary random-intercept model retained"},{"analysis":"quadratic_time","model_status":"not_run","eligible_patients":int(counts.ge(3).sum()),"note":"Avoided over-flexible model"}]);sensitivity.to_csv(tables/"13_essdai_model_sensitivity.csv",index=False)
    prog.to_csv(tables/"13_essdai_baseline_to_last_status.csv",index=False)
    summary=[];den=len(prog)
    for s in ("worsened","stable","improved"): summary.append({"definition":f"baseline_to_last_{s}","n":int(prog.status.eq(s).sum()),"denominator":den})
    for label,mask in [("ever_worsened",prog.ever_worsened),("never_worsened",~prog.ever_worsened),("ever_reached_moderate_or_high",prog.maximum_state_reached.isin(["Moderate","High"])),("ever_reached_high",prog.maximum_state_reached.eq("High"))]:summary.append({"definition":label,"n":int(mask.sum()),"denominator":den})
    summary=pd.DataFrame(summary);summary["pct"]=100*summary.n/summary.denominator.replace(0,np.nan);summary.to_csv(tables/"13_essdai_progression_status_summary.csv",index=False);transitions.to_csv(tables/"13_essdai_state_transition_matrix.csv",index=False)
    event_qc=[]
    for name,tte in [("low_to_ge5",low),("moderate_to_high",moderate)]:
        curve,stats=km_estimate(tte); row={"analysis":name,"n_at_risk":len(tte),"n_events":int(tte.event.sum()) if len(tte) else 0,"n_censored":int((~tte.event).sum()) if len(tte) else 0,"median_followup":tte.time_years.median() if len(tte) else np.nan,"event_rate":tte.event.mean() if len(tte) else np.nan,**stats,"interval_censored_sensitivity":"not executed; first-observed-event KM is primary","note":EVENT_NOTE};pd.DataFrame([row]).to_csv(tables/f"13_tte_{name}.csv",index=False);event_qc.append(row)
        tte.to_csv(tables/f"13_tte_{name}_patient_level.csv",index=False)
    dmodels,dchange,dsupport=domain_analyses(df,resolved);dmodels.to_csv(tables/"13_essdai_domain_models.csv",index=False);dchange.to_csv(tables/"13_essdai_domain_change_summary.csv",index=False);dsupport.to_csv(qc/"13_progression_domain_support.csv",index=False)
    pro_rows=[];pro_changes=[]
    for name,col in ESSPRI.items():
        if col not in df: continue
        row,_=fit_continuous_model(df,col,f"ESSPRI {name}");row["measure"]=name;pro_rows.append(row)
        pairs=[]
        for pid,g in df.groupby("patient_id"):
            b=g.loc[g.is_clinical_baseline.eq(True),col];a=g.loc[(~g.is_clinical_baseline.eq(True))&g[col].notna()].sort_values("clinical_anchor_date")
            if len(b)==1 and pd.notna(b.iloc[0]) and len(a):pairs.append((b.iloc[0],a.iloc[-1][col],(a.iloc[-1].clinical_anchor_date-g.loc[g.is_clinical_baseline.eq(True)].iloc[0].clinical_anchor_date).days/365.25))
        delta=np.array([b-a for a,b,_ in pairs]);pro_changes.append({"measure":name,"cohort":"all_repeated","n":len(pairs),"median_change":np.median(delta) if len(delta) else np.nan})
        spaced=[(a,b,t) for a,b,t in pairs if t>=.5]; spaced_delta=np.array([b-a for a,b,_ in spaced])
        pro_changes.append({"measure":name,"cohort":"protocol_spacing_ge_6_months","n":len(spaced),"median_change":np.median(spaced_delta) if len(spaced_delta) else np.nan})
    pro_models=pd.DataFrame(pro_rows);pro_models["q_value"]=np.nan
    if len(pro_models):
        ix=pro_models.measure.isin(["dryness","fatigue","pain"]);pro_models.loc[ix,"q_value"]=bh_fdr(pro_models.loc[ix,"p_value"])
    pro_models.to_csv(tables/"13_esspri_models.csv",index=False);pd.DataFrame(pro_changes).to_csv(tables/"13_esspri_change_summary.csv",index=False)
    sens=[]
    for r in prog.itertuples():
        sens += [{"patient_id":r.patient_id,"outcome":"essdai_improved_ge3","evaluable":True,"met":r.delta_essdai<=CHANGE_THRESHOLDS["essdai_improvement"]},{"patient_id":r.patient_id,"outcome":"essdai_legacy_worsening_ge5_sensitivity","evaluable":True,"met":r.delta_essdai>=CHANGE_THRESHOLDS["essdai_legacy_worsening"]}]
    for name,col in ESSPRI.items():
        if col not in df:continue
        for pid,g in df.groupby("patient_id"):
            b=g.loc[g.is_clinical_baseline.eq(True),col];a=g.loc[(~g.is_clinical_baseline.eq(True))&g[col].notna()].sort_values("clinical_anchor_date")
            if len(b)==1 and pd.notna(b.iloc[0]) and len(a):
                delta=a.iloc[-1][col]-b.iloc[0];sens.append({"patient_id":pid,"outcome":f"esspri_{name}_absolute_improvement","evaluable":True,"met":delta<=CHANGE_THRESHOLDS["esspri_absolute_improvement"]});sens.append({"patient_id":pid,"outcome":f"esspri_{name}_relative_improvement","evaluable":b.iloc[0]!=0,"met":delta/b.iloc[0]<=CHANGE_THRESHOLDS["esspri_relative_improvement"] if b.iloc[0]!=0 else pd.NA})
    pd.DataFrame(sens).to_csv(tables/"13_clinical_change_sensitivity.csv",index=False)
    keep=REQUIRED[:3]+["clinical_baseline_date","time_since_clinical_baseline_years",ESSDAI,*ESSPRI.values(),*resolved.values(),"essdai_activity_state","essdai_activity_rank","previous_essdai","delta_essdai_from_previous","previous_essdai_state","state_change_from_previous"]
    df[[c for c in dict.fromkeys(keep) if c in df]].rename(columns={ESSDAI:"essdai_total",**{v:f"esspri_{k}" for k,v in ESSPRI.items()}}).to_parquet(analytic/"13_progression_episode_level.parquet",index=False)
    model_qc=pd.concat([pd.DataFrame([primary]),pro_models],ignore_index=True);model_qc.to_csv(qc/"13_progression_model_qc.csv",index=False);pd.DataFrame(event_qc).to_csv(qc/"13_progression_event_qc.csv",index=False)
    same_date=df.duplicated(["patient_id","clinical_anchor_date"],False).sum();first_delay=[]
    for pid,g in df.groupby("patient_id"):
        b=g.loc[g.is_clinical_baseline.eq(True)];obs=g.loc[g[ESSDAI].notna()]
        if len(b)==1 and len(obs):first_delay.append((obs.iloc[0].clinical_anchor_date-b.iloc[0].clinical_anchor_date).days)
    pd.DataFrame([{"check":"structural_validation","status":"pass","same_date_distinct_episode_rows":same_date,"baseline_essdai_available":base_rows[ESSDAI].notna().sum(),"baseline_essdai_missing":base_rows[ESSDAI].isna().sum(),"first_essdai_after_baseline":sum(x>0 for x in first_delay),"median_delay_to_first_essdai_days":np.median(first_delay) if first_delay else np.nan}]).to_csv(qc/"13_progression_qc.csv",index=False)
    bins=pd.cut(df.time_since_clinical_baseline_years,[-.001,0,1,2,3,5,np.inf],labels=["baseline",">0-1y",">1-2y",">2-3y",">3-5y",">5y"],include_lowest=True);miss=[]
    for name,col in measures.items():
        for period,g in df.groupby(bins,observed=False):miss.append({"measure":name,"time_window":period,"n_rows":len(g),"n_evaluable":g[col].notna().sum() if col in g else 0,"n_patients":g.loc[g[col].notna(),"patient_id"].nunique() if col in g else 0,"pct_missing":100*g[col].isna().mean() if col in g and len(g) else np.nan})
    pd.DataFrame(miss).to_csv(qc/"13_progression_missingness_by_time.csv",index=False)
    plot_trajectory(df,ESSDAI,figures/"13_essdai_trajectory.pdf","ESSDAI total");plot_km(low,figures/"13_km_low_to_ge5.pdf","Low to ESSDAI >=5");plot_km(moderate,figures/"13_km_moderate_to_high.pdf","Moderate to High ESSDAI")
    mat=transitions.pivot(index="from_state",columns="to_state",values="row_pct").reindex(index=STATE_ORDER,columns=STATE_ORDER);save_heatmap(mat,figures/"13_essdai_state_transition_heatmap.pdf","ESSDAI state transitions (%)")
    active=[]
    for domain,col in resolved.items():
        for period,g in df.groupby(bins,observed=False):active.append({"domain":domain,"period":str(period),"pct":100*g[col].gt(0).sum()/g[col].notna().sum() if g[col].notna().sum() else np.nan})
    ah=pd.DataFrame(active).pivot(index="domain",columns="period",values="pct") if active else pd.DataFrame(index=DOMAINS,columns=["baseline"]);save_heatmap(ah,figures/"13_essdai_domain_activity_heatmap.pdf","ESSDAI domain active (%)")
    fig,ax=plt.subplots(figsize=(7,6));ok=dmodels.loc[dmodels.effect.notna()] if "effect" in dmodels else dmodels.iloc[0:0];ax.errorbar(ok.effect,range(len(ok)),xerr=[ok.effect-ok.ci95_low,ok.ci95_high-ok.effect],fmt="o");ax.set(yticks=range(len(ok)),yticklabels=ok.domain,xlabel="Time effect (model scale)",title="ESSDAI domain effects");fig.tight_layout();fig.savefig(figures/"13_essdai_domain_effects_forest.pdf");plt.close(fig)
    if "esspri__total" in df:plot_trajectory(df,"esspri__total",figures/"13_esspri_trajectory_total.pdf","ESSPRI total")
    fig,axes=plt.subplots(1,3,figsize=(13,4));
    for ax,(name,col) in zip(axes,list(ESSPRI.items())[1:]):
        z=df.dropna(subset=[col]);ax.scatter(z.time_since_clinical_baseline_years,z[col],s=8,alpha=.3);ax.set(title=name,xlabel="Years",ylabel="ESSPRI")
    fig.tight_layout();fig.savefig(figures/"13_esspri_trajectory_components.pdf");plt.close(fig)
    LOG.info("Completed: %d rows, %d patients",len(df),df.patient_id.nunique())
    print("ORDER TO REVIEW DISEASE ACTIVITY PROGRESSION OUTPUTS\n\n1. 13_progression_cohort_flow.csv\n2. 13_essdai_longitudinal_model.csv\n3. 13_essdai_trajectory.pdf\n4. 13_essdai_progression_status_summary.csv\n5. 13_essdai_state_transition_matrix.csv\n6. 13_km_low_to_ge5.pdf\n7. 13_km_moderate_to_high.pdf\n8. 13_essdai_domain_models.csv\n9. 13_essdai_domain_activity_heatmap.pdf\n10. 13_esspri_models.csv\n11. 13_esspri_trajectory_total.pdf\n12. 13_clinical_change_sensitivity.csv\n13. 13_progression_qc.csv")

if __name__=="__main__": main()

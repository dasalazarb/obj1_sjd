"""Audit longitudinal observation and form completeness on the canonical visit spine.

``absent`` means that no valid items or valid score were observed in the
available analytical extract.  It does not establish that a form was not
administered.  Outputs describe observation availability, not disease change
and not causal missing-data mechanisms.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common

TABLES_DIR = common.BLOCKB_TABLES_DIR
FIGURES_DIR = common.BLOCKB_FIGURES_DIR
QC_DIR = common.BLOCKB_QC_DIR
LOGS_DIR = common.BLOCKB_LOGS_DIR
INTERMEDIATE_DIR = common.INTERMEDIATE_DATA_DIR
KEY_COLUMNS = ["patient_id", "visit_id"]
REQUIRED_COLUMNS = ["patient_id", "visit_id", "visit_date", "visit_number", "is_observed_baseline", "time_since_observed_baseline_days", "time_since_observed_baseline_years"]
RECOMMENDED_DESIGN_COLUMNS = ["protocol", "protocol_number", "study_protocol", "interval_name"]
RECOMMENDED_COLUMNS = ["essdai_total", "esspri_total", "pop_status", "pop_missingness_label", "overlap_status", "glandular_evaluable", "extraglandular_evaluable", "sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global", "has_essdai", "has_esspri", "has_sf36_pcs", "has_sf36_mcs", "has_profad", "has_mdafs", "n_data_blocks_available", "time_from_previous_visit_days", "time_from_previous_visit_years", "previous_pop", "previous_overlap_status"]
PRO_DETAIL_COLUMNS = ["patient_id", "visit_id", "visit_date", "esspri_n_components", "esspri_n_components_available", "esspri_scoring_valid", "esspri_scoring_version", "sf36_n_items_available", "sf36_n_domains_available", "sf36_scoring_valid", "sf36_pcs_scoring_status", "sf36_mcs_scoring_status", "sf36_scoring_version", "profad_n_items_answered", "profad_n_items_available", "profad_scoring_valid", "profad_scoring_status", "profad_scoring_version", "mdafs_n_activity_items_answered", "mdafs_n_items_available", "mdafs_scoring_valid", "mdafs_scoring_status", "mdafs_scoring_version"]
COUNT_RANGES = {"esspri_n_components": (0,3), "esspri_n_components_available": (0,3), "sf36_n_items_available": (0,36), "sf36_n_domains_available": (0,8), "profad_n_items_answered": (0,19), "profad_n_items_available": (0,19), "mdafs_n_items_available": (0,16), "mdafs_n_activity_items_answered": (0,11)}
BOOLEAN_COLUMNS = ["is_observed_baseline", "esspri_scoring_valid", "sf36_scoring_valid", "profad_scoring_valid", "mdafs_scoring_valid", "glandular_evaluable", "extraglandular_evaluable", "has_essdai", "has_esspri", "has_sf36_pcs", "has_sf36_mcs", "has_profad", "has_mdafs"]
MEASURES = [("ESSDAI","avail_essdai"),("ESSPRI complete","avail_esspri_complete"),("SF-36 complete","avail_sf36_complete"),("PROFAD complete","avail_profad_complete"),("MDAFS complete","avail_mdafs_complete"),("Overlap evaluable","avail_overlap_evaluable"),("Pop classifiable","avail_pop_classifiable"),("Pop core complete","pop_core_complete"),("Systemic-overlap core complete","systemic_overlap_core_complete"),("PRO core complete","pro_core_complete"),("Full multidomain complete","full_multidomain_complete"),("Extended multidomain complete","extended_multidomain_complete")]
MODEL_COLUMNS = ["predictor","level_or_contrast","coefficient","standard_error","odds_ratio","ci_95_lower","ci_95_upper","p_value","reference_category","n_visits","n_patients","model_name","model_formula","model_status","interpretation_scope"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    p.add_argument("--pros-input",type=Path,default=common.PROS_LONGITUDINAL_PARQUET)
    p.add_argument("--min-gap-days",type=int,default=180)
    p.add_argument("--overwrite",dest="overwrite",action="store_true",default=True)
    p.add_argument("--no-overwrite",dest="overwrite",action="store_false")
    p.add_argument("--skip-models",action="store_true")
    a=p.parse_args(argv)
    if a.min_gap_days < 0: p.error("--min-gap-days must be nonnegative")
    return a


def setup_logging() -> logging.Logger:
    common.ensure_output_dirs(); logger=logging.getLogger("blockB20"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt=logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(LOGS_DIR/"20_missingness_and_observation_process.log",mode="w")):
        h.setFormatter(fmt); logger.addHandler(h)
    return logger


def _read(path: Union[Path,str], columns: Optional[List[str]]=None) -> pd.DataFrame:
    path=Path(path)
    if path.suffix.lower() in (".parquet",".pq"): return pd.read_parquet(path,columns=columns)
    return pd.read_csv(path,usecols=(lambda c: c in columns) if columns else None)


def load_integrated_data(path: Union[Path,str]) -> pd.DataFrame: return _read(path)


def load_pro_detail(path: Union[Path,str]) -> Tuple[pd.DataFrame,List[str]]:
    path=Path(path)
    if not path.exists(): return pd.DataFrame(columns=KEY_COLUMNS), list(PRO_DETAIL_COLUMNS)
    if path.suffix.lower() in (".parquet",".pq"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc: raise RuntimeError("pyarrow is required to inspect Parquet schema") from exc
        names=pq.ParquetFile(path).schema.names
    else: names=pd.read_csv(path,nrows=0).columns.tolist()
    available=[c for c in PRO_DETAIL_COLUMNS if c in names]
    if not set(KEY_COLUMNS).issubset(available): raise ValueError("PRO-detail input lacks patient_id and/or visit_id")
    return _read(path,available), [c for c in PRO_DETAIL_COLUMNS if c not in names]


def _to_boolean(s: pd.Series) -> pd.Series:
    if str(s.dtype)=="boolean": return s
    mapping={"true":True,"false":False,"1":True,"0":False,"yes":True,"no":False}
    return s.map(lambda x: mapping.get(str(x).strip().lower(), pd.NA) if pd.notna(x) else pd.NA).astype("boolean")


def normalize_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    out=df.copy(); violations=[]
    for c in KEY_COLUMNS:
        if c in out: out[c]=out[c].astype("string")
    if "visit_date" in out: out["visit_date"]=pd.to_datetime(out["visit_date"],errors="coerce")
    for c in BOOLEAN_COLUMNS:
        if c in out: out[c]=_to_boolean(out[c])
    for c,(lo,hi) in COUNT_RANGES.items():
        if c not in out: continue
        n=pd.to_numeric(out[c],errors="coerce"); bad=out[c].notna() & (n.isna() | n.mod(1).ne(0) | n.lt(lo) | n.gt(hi))
        for idx in out.index[bad]: violations.append({"row_index":idx,"patient_id":out.at[idx,"patient_id"] if "patient_id" in out else pd.NA,"visit_id":out.at[idx,"visit_id"] if "visit_id" in out else pd.NA,"variable":c,"value":out.at[idx,c],"valid_range":"%d-%d; integer required"%(lo,hi)})
        if bad.any(): continue
        out[c]=n.astype("Int64")
    for c in ["visit_number","time_since_observed_baseline_days","time_since_observed_baseline_years","time_from_previous_visit_days","time_from_previous_visit_years","essdai_total","esspri_total","sf36_pcs","sf36_mcs","profad_total","mdafs_global"]:
        if c in out: out[c]=pd.to_numeric(out[c],errors="coerce")
    return out,pd.DataFrame(violations,columns=["row_index","patient_id","visit_id","variable","value","valid_range"])


def validate_integrated_input(df: pd.DataFrame) -> Dict[str,Any]:
    missing=[c for c in REQUIRED_COLUMNS if c not in df]
    if missing: raise ValueError("Integrated input missing required columns: "+", ".join(missing))
    dup=df.duplicated(KEY_COLUMNS,keep=False); vdup=df.duplicated("visit_id",keep=False)
    baselines=df.groupby("patient_id",dropna=False)["is_observed_baseline"].apply(lambda s:int(s.eq(True).sum()))
    ordered=df.sort_values(["patient_id","visit_date","visit_number"],kind="stable")
    intervals=ordered.groupby("patient_id",dropna=False).visit_date.diff().dt.days
    qc={"n_duplicate_patient_visit_keys":int(dup.sum()),"n_duplicate_visit_ids":int(vdup.sum()),"n_patients_without_baseline":int(baselines.eq(0).sum()),"n_patients_with_multiple_baselines":int(baselines.gt(1).sum()),"n_invalid_visit_dates":int(df.visit_date.isna().sum()),"n_negative_followup_times":int(pd.to_numeric(df.time_since_observed_baseline_days,errors="coerce").lt(0).sum()),"n_negative_visit_intervals":int(intervals.lt(0).sum())}
    failures=[]
    if qc["n_duplicate_patient_visit_keys"]: failures.append("duplicated patient_id + visit_id keys")
    if qc["n_duplicate_visit_ids"]: failures.append("globally duplicated visit_id values")
    if qc["n_patients_without_baseline"]: failures.append("patients without an observed baseline")
    if qc["n_patients_with_multiple_baselines"]: failures.append("patients with multiple observed baselines")
    if qc["n_negative_followup_times"] or qc["n_negative_visit_intervals"]: failures.append("negative follow-up or chronological intervals")
    if failures: raise ValueError("Integrated input validation failed: "+"; ".join(failures))
    return qc


def _equal(a:pd.Series,b:pd.Series)->pd.Series:
    if pd.api.types.is_datetime64_any_dtype(a) or pd.api.types.is_datetime64_any_dtype(b): return pd.to_datetime(a,errors="coerce").eq(pd.to_datetime(b,errors="coerce"))
    return a.astype("string").str.strip().str.lower().eq(b.astype("string").str.strip().str.lower())


def merge_pro_detail(integrated:pd.DataFrame,pro:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    if pro.duplicated(KEY_COLUMNS).any(): raise ValueError("PRO-detail contains duplicate patient-visit keys; one-to-many merge rejected")
    shared=[c for c in pro.columns if c in integrated.columns and c not in KEY_COLUMNS]
    right=pro.rename(columns={c:c+"__pro_detail" for c in shared})
    merged=integrated.merge(right,on=KEY_COLUMNS,how="left",validate="one_to_one",indicator=True)
    discrepancies=[]
    for c in shared:
        pc=c+"__pro_detail"; both=merged[c].notna() & merged[pc].notna(); bad=both & ~_equal(merged[c],merged[pc])
        for _,r in merged.loc[bad,KEY_COLUMNS+[c,pc]].iterrows(): discrepancies.append({"patient_id":r.patient_id,"visit_id":r.visit_id,"field":c,"integrated_value":r[c],"pro_detail_value":r[pc]})
        merged[c]=merged[c].where(merged[c].notna(),merged[pc]); merged.drop(columns=pc,inplace=True)
    summary=pd.DataFrame([{"n_integrated_visits":len(integrated),"n_pro_detail_rows":len(pro),"n_pro_detail_matched_visits":int(merged._merge.eq("both").sum()),"n_pro_detail_unmatched_visits":int(merged._merge.ne("both").sum()),"n_shared_fields":len(shared),"n_shared_field_disagreements":len(discrepancies)}])
    merged.drop(columns="_merge",inplace=True)
    if len(merged)!=len(integrated): raise RuntimeError("Integrated row count changed during PRO-detail merge")
    return merged,summary,pd.DataFrame(discrepancies,columns=["patient_id","visit_id","field","integrated_value","pro_detail_value"])


def _series(df:pd.DataFrame,name:str,dtype:Optional[str]=None)->pd.Series:
    s=df[name] if name in df else pd.Series(pd.NA,index=df.index)
    return s.astype(dtype) if dtype else s


def _pro_state(valid:pd.Series,count:pd.Series,score_present:pd.Series,instrument:str)->Tuple[pd.Series,pd.Series]:
    valid=_to_boolean(valid); count=pd.to_numeric(count,errors="coerce")
    state=pd.Series("unknown_detail",index=valid.index,dtype="string")
    state.loc[valid.eq(True)]="complete"; unresolved=~valid.eq(True).fillna(False)
    state.loc[unresolved & count.eq(0)]="absent"; state.loc[unresolved & count.gt(0)]="partial"
    contradiction=(valid.eq(True)&count.eq(0)) | (valid.eq(True)&~score_present) | (valid.eq(False)&score_present)
    return state,contradiction.astype("boolean")


def derive_observation_states(df:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    out=df.copy(); out["essdai_observation_state"]=np.where(_series(out,"essdai_total").notna(),"complete","absent")
    ec="esspri_n_components_available" if "esspri_n_components_available" in out else "esspri_n_components"
    specs=[("esspri",ec,"esspri_total"),("sf36","sf36_n_items_available",None),("profad","profad_n_items_available" if "profad_n_items_available" in out else "profad_n_items_answered","profad_total"),("mdafs","mdafs_n_items_available","mdafs_global")]
    rows=[]
    for inst,count_col,score in specs:
        score_present=(_series(out,"sf36_pcs").notna() & _series(out,"sf36_mcs").notna()) if inst=="sf36" else _series(out,score).notna()
        state,disc=_pro_state(_series(out,inst+"_scoring_valid"),_series(out,count_col),score_present,inst)
        out[inst+"_observation_state"]=state; out[inst+"_observation_discordance"]=disc
        for _,r in out.loc[disc,KEY_COLUMNS].iterrows(): rows.append({"patient_id":r.patient_id,"visit_id":r.visit_id,"instrument":inst,"reason":"validity, count, and score presence disagree"})
    ge=_to_boolean(_series(out,"glandular_evaluable")); xe=_to_boolean(_series(out,"extraglandular_evaluable")); os=_series(out,"overlap_status").astype("string")
    evaluable=ge.eq(True)&xe.eq(True)&os.notna()&~os.str.lower().isin(["insufficient_info","unclassifiable","unknown"])
    anyinfo=ge.notna()|xe.notna()|os.notna(); out["overlap_observation_state"]=pd.Series(np.select([evaluable,anyinfo],["evaluable","insufficient_info"],default="absent"),index=out.index,dtype="string")
    pop=_series(out,"pop_status").astype("string"); miss=_series(out,"pop_missingness_label").astype("string").str.lower(); ess=_series(out,"essdai_total"); epr=_series(out,"esspri_total")
    ps=pd.Series("other_unclassifiable",index=out.index,dtype="string"); ps.loc[pop.str.contains("pop ?1",case=False,na=False)]="classifiable_pop1"; ps.loc[pop.str.contains("pop ?[23]",case=False,na=False,regex=True)]="classifiable_pop2_or_pop3"
    ps.loc[ess.isna()&epr.isna()]="both_missing"; ps.loc[miss.str.contains("low.*symptom|esspri.*missing",na=False)]="low_essdai_symptom_unknown"; ps.loc[ess.isna()&epr.lt(5)]="essdai_unknown_esspri_low"; ps.loc[ess.isna()&epr.ge(5)]="essdai_unknown_esspri_high"
    out["pop_observation_state"]=ps
    return out,pd.DataFrame(rows,columns=["patient_id","visit_id","instrument","reason"])


def _state_available(s:pd.Series,complete:str="complete")->pd.Series:
    result=pd.Series(pd.NA,index=s.index,dtype="boolean"); result.loc[s.eq(complete)]=True; result.loc[~s.eq("unknown_detail") & s.notna()]=False; return result


def derive_protocol_group(df:pd.DataFrame)->pd.Series:
    result=pd.Series(pd.NA,index=df.index,dtype="string")
    for c in ["protocol","protocol_number","study_protocol"]:
        if c in df:
            v=df[c].astype("string").str.strip(); result=result.fillna(v.where(v.ne("")))
    return result.fillna("Unknown")


def derive_completeness_indicators(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy(); out["protocol_group"]=derive_protocol_group(out)
    out["avail_essdai"]=_state_available(out.essdai_observation_state)
    for inst in ["esspri","sf36","profad","mdafs"]: out["avail_"+inst+"_complete"]=_state_available(out[inst+"_observation_state"])
    out["avail_overlap_evaluable"]=_state_available(out.overlap_observation_state,"evaluable")
    out["avail_pop_classifiable"]=out.pop_observation_state.isin(["classifiable_pop1","classifiable_pop2_or_pop3"]).astype("boolean")
    out["pop_core_complete"]=out.avail_essdai & out.avail_esspri_complete; out["systemic_overlap_core_complete"]=out.avail_essdai & out.avail_overlap_evaluable; out["pro_core_complete"]=out.avail_esspri_complete & out.avail_sf36_complete
    out["full_multidomain_complete"]=out.avail_essdai & out.avail_esspri_complete & out.avail_sf36_complete & out.avail_overlap_evaluable
    out["extended_multidomain_complete"]=out.full_multidomain_complete & out.avail_profad_complete & out.avail_mdafs_complete; out["core_complete"]=out.full_multidomain_complete; out["extended_complete"]=out.extended_multidomain_complete
    procols=["avail_esspri_complete","avail_sf36_complete","avail_profad_complete","avail_mdafs_complete"]
    out["n_pro_instruments_complete"]=out[procols].astype("Int64").sum(axis=1,min_count=1).astype("Int64")
    core=["avail_essdai","avail_esspri_complete","avail_sf36_complete","avail_overlap_evaluable"]
    out["n_core_blocks_complete"]=out[core].astype("Int64").sum(axis=1,min_count=1).astype("Int64")
    out["pro_bundle_state"]=np.select([out[procols].eq(True).all(axis=1),out[procols].eq(False).all(axis=1)],["all_complete","none_complete"],default="mixed_or_unknown")
    code_maps={"essdai":{"complete":1,"absent":0},"esspri":{"complete":1,"absent":0,"partial":2,"unknown_detail":9},"sf36":{"complete":1,"absent":0,"partial":2,"unknown_detail":9},"profad":{"complete":1,"absent":0,"partial":2,"unknown_detail":9},"mdafs":{"complete":1,"absent":0,"partial":2,"unknown_detail":9},"overlap":{"evaluable":1,"absent":0,"insufficient_info":9}}
    labels={"essdai":"ESSDAI","esspri":"ESSPRI","sf36":"SF36","profad":"PROFAD","mdafs":"MDAFS","overlap":"OVERLAP"}
    codes=[]; readable=[]
    for idx in out.index:
        cp=[]; rp=[]
        for inst in ["essdai","esspri","sf36","profad","mdafs","overlap"]:
            state=str(out.at[idx,inst+"_observation_state"]); cp.append("%s=%s"%(labels[inst],code_maps[inst].get(state,9))); rp.append(("SF-36" if inst=="sf36" else inst.upper())+" "+state.replace("_"," "))
        codes.append("|".join(cp)); readable.append("; ".join(rp))
    out["visit_observation_pattern"]=pd.Series(codes,index=out.index,dtype="string"); out["visit_observation_pattern_label"]=pd.Series(readable,index=out.index,dtype="string")
    out=add_time_groups(out); ordered=out.sort_values(["patient_id","visit_date","visit_number"],kind="stable")
    for target in ["esspri","sf36"]: ordered["previous_"+target+"_available"]=ordered.groupby("patient_id")["avail_"+target+"_complete"].shift(1).astype("boolean")
    return ordered.sort_index()


def add_time_groups(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy(); y=out.visit_date.dt.year; out["calendar_year"]=y.astype("Int64").astype("string").fillna("Unknown")
    out["calendar_period"]=pd.Series(np.select([y.between(2011,2015),y.between(2016,2020),y.between(2021,2026),y.gt(2026)],["2011–2015","2016–2020","2021–2026","After 2026"],default="Unknown"),index=out.index,dtype="string")
    t=pd.to_numeric(out.time_since_observed_baseline_years,errors="coerce"); out["followup_time_bin"]=pd.Series(np.select([t.eq(0),t.gt(0)&t.le(1),t.gt(1)&t.le(2),t.gt(2)&t.le(5),t.gt(5)&t.le(10),t.gt(10)],["Baseline",">0–1 year",">1–2 years",">2–5 years",">5–10 years",">10 years"],default="Unknown"),index=out.index,dtype="string")
    v=pd.to_numeric(out.visit_number,errors="coerce"); out["visit_number_group"]=pd.Series(np.select([v.eq(0),v.eq(1),v.eq(2),v.eq(3),v.ge(4)],["0","1","2","3","4+"],default="Unknown"),index=out.index,dtype="string")
    base=out.loc[out.is_observed_baseline.eq(True),["patient_id","pop_status"]].drop_duplicates("patient_id").copy() if "pop_status" in out else pd.DataFrame(columns=["patient_id","pop_status"])
    if len(base):
        p=base.pop_status.astype("string"); base["baseline_pop_group"]=np.select([p.str.contains("pop ?1",case=False,na=False),p.str.contains("pop ?2",case=False,na=False),p.str.contains("pop ?3",case=False,na=False),p.str.contains("unclass",case=False,na=False)],["Pop1","Pop2","Pop3","Unclassifiable"],default="Unknown"); out=out.merge(base[["patient_id","baseline_pop_group"]],on="patient_id",how="left",validate="many_to_one")
    else: out["baseline_pop_group"]="Unknown"
    out["baseline_pop_group"]=out.baseline_pop_group.fillna("Unknown").astype("string"); return out


def summarize_availability(df:pd.DataFrame,group:Optional[str]=None)->pd.DataFrame:
    groups=[("Overall",df)] if group is None else list(df.groupby(group,dropna=False,sort=False))
    rows=[]
    for level,g in groups:
        for label,col in MEASURES:
            available=int(g[col].eq(True).sum()); n=len(g)
            rows.append({**({group:str(level)} if group else {}),"measure":label,"total_canonical_visits":n,"unique_patients_contributing":int(g.patient_id.nunique()),"available_visits":available,"unavailable_visits":n-available,"percentage_available":100*available/n if n else np.nan,"percentage_missing":100*(n-available)/n if n else np.nan,"denominator_definition":"All canonical patient-visits in the stated stratum"})
    return pd.DataFrame(rows)


def build_joint_completeness_tables(df:pd.DataFrame)->Dict[str,pd.DataFrame]:
    def patterns(group:Optional[str]=None)->pd.DataFrame:
        cols=["visit_observation_pattern","visit_observation_pattern_label"]+([group] if group else [])
        x=df.groupby(cols,dropna=False).agg(number_of_visits=("visit_id","size"),number_of_unique_patients=("patient_id","nunique"),baseline_visits=("is_observed_baseline",lambda s:int(s.eq(True).sum()))).reset_index(); x["followup_visits"]=x.number_of_visits-x.baseline_visits; x["percentage_of_canonical_visits"]=100*x.number_of_visits/len(df); x["denominator_definition"]="All canonical visits"; return x
    av=[c for _,c in MEASURES[:6]]; binary=df[KEY_COLUMNS+av].copy()
    pairs=[]
    for a,b in combinations(av,2):
        use=df[a].notna()&df[b].notna(); x=df.loc[use,a].astype(bool); y=df.loc[use,b].astype(bool); both=int((x&y).sum()); onlya=int((x&~y).sum()); onlyb=int((~x&y).sum()); neither=int((~x&~y).sum()); den=math.sqrt((both+onlya)*(onlyb+neither)*(both+onlyb)*(onlya+neither)); phi=((both*neither-onlya*onlyb)/den) if den else np.nan
        pairs.append({"first_measure":a,"second_measure":b,"both_available":both,"only_first_available":onlya,"only_second_available":onlyb,"neither_available":neither,"phi_coefficient":phi,"number_of_visits_used":int(use.sum()),"denominator_definition":"Visits with known availability for both measures"})
    return {"patterns_overall":patterns(),"patterns_protocol":patterns("protocol_group"),"patterns_period":patterns("calendar_period"),"joint":binary,"pairwise":pd.DataFrame(pairs)}


def classify_binary_sequence(values:Sequence[Union[bool,int]])->str:
    x=[int(bool(v)) for v in values]
    if len(x)==1:return "single_visit"
    if all(x):return "always_available"
    if not any(x):return "never_available"
    switches=sum(a!=b for a,b in zip(x,x[1:]))
    if switches==1 and x[0]==0:return "late_entry"
    if switches==1 and x[0]==1:return "monotone_dropout"
    return "intermittent"


def build_patient_level_outputs(df:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for patient,g in df.sort_values(["patient_id","visit_date","visit_number"],kind="stable").groupby("patient_id",sort=False):
        row={"patient_id":patient}
        for target in ["esspri","sf36"]:
            col="avail_"+target+"_complete"; seq=g[col].eq(True).astype(int).tolist(); complete=g.loc[g[col].eq(True),"visit_date"]; incomplete=g.loc[~g[col].eq(True),"visit_date"]
            gaps=complete.sort_values().diff().dt.days
            pre=target+"_"; row.update({pre+"n_canonical_visits":len(g),pre+"n_complete_visits":sum(seq),pre+"proportion_complete":sum(seq)/len(seq),pre+"first_complete_date":complete.min(),pre+"last_complete_date":complete.max(),pre+"first_incomplete_date":incomplete.min(),pre+"last_incomplete_date":incomplete.max(),pre+"longest_gap_between_complete_days":gaps.max() if len(complete)>=2 else np.nan,pre+"baseline_available":bool(seq[0]),pre+"last_visit_available":bool(seq[-1]),pre+"n_availability_switches":sum(a!=b for a,b in zip(seq,seq[1:])),pre+"observation_pattern":classify_binary_sequence(seq)})
        rows.append(row)
    return pd.DataFrame(rows)


def _max_gap(g:pd.DataFrame,mask:pd.Series)->float:
    d=g.loc[mask.fillna(False),"visit_date"].dropna(); return float((d.max()-d.min()).days) if len(d)>=2 else np.nan


def build_analytic_cohort_membership(df:pd.DataFrame,min_gap_days:int=180)->pd.DataFrame:
    rows=[]
    for pid,g in df.sort_values(["visit_date","visit_number"]).groupby("patient_id",sort=False):
        masks={"essdai":g.avail_essdai.eq(True),"esspri":g.avail_esspri_complete.eq(True),"sf36":g.avail_sf36_complete.eq(True),"full":g.full_multidomain_complete.eq(True)}; counts={k:int(v.sum()) for k,v in masks.items()}; gaps={k:_max_gap(g,v) for k,v in masks.items()}
        pop=g.avail_pop_classifiable.eq(True); ov=g.avail_overlap_evaluable.eq(True); poptrans=bool((pop&pop.shift(fill_value=False)).any()); ovtrans=bool((ov&ov.shift(fill_value=False)).any())
        r={"patient_id":pid,"n_canonical_visits":len(g),"n_essdai_assessments":counts["essdai"],"max_essdai_measurement_gap_days":gaps["essdai"],"n_complete_esspri_assessments":counts["esspri"],"max_complete_esspri_gap_days":gaps["esspri"],"n_complete_sf36_assessments":counts["sf36"],"max_complete_sf36_gap_days":gaps["sf36"],"n_full_multidomain_complete_visits":counts["full"],"max_full_multidomain_complete_gap_days":gaps["full"],"in_all_canonical_patients":True,"in_observed_baseline_cohort":bool(g.is_observed_baseline.eq(True).any()),"in_2plus_canonical_visits":len(g)>=2,"in_2plus_essdai_assessments":counts["essdai"]>=2,"in_2plus_essdai_assessments_min_gap":counts["essdai"]>=2 and gaps["essdai"]>=min_gap_days,"in_2plus_complete_esspri_assessments":counts["esspri"]>=2,"in_2plus_complete_esspri_assessments_min_gap":counts["esspri"]>=2 and gaps["esspri"]>=min_gap_days,"in_2plus_complete_sf36_assessments":counts["sf36"]>=2,"in_2plus_complete_sf36_assessments_min_gap":counts["sf36"]>=2 and gaps["sf36"]>=min_gap_days,"in_1plus_evaluable_pop_transition":poptrans,"in_1plus_evaluable_overlap_transition":ovtrans,"in_1plus_full_multidomain_complete_visit":counts["full"]>=1,"in_2plus_full_multidomain_complete_visits":counts["full"]>=2,"in_2plus_full_multidomain_complete_visits_min_gap":counts["full"]>=2 and gaps["full"]>=min_gap_days,"in_1plus_extended_multidomain_complete_visit":bool(g.extended_multidomain_complete.eq(True).any())}; rows.append(r)
    out=pd.DataFrame(rows)
    for c in out.filter(regex="^in_"): out[c]=out[c].astype("boolean")
    return out


def build_analytic_cohort_counts(membership:pd.DataFrame,df:pd.DataFrame,min_gap_days:int=180)->pd.DataFrame:
    defs=[("all_canonical_patients","in_all_canonical_patients","Any canonical visit"),("observed_baseline_cohort","in_observed_baseline_cohort","Observed baseline visit"),("patients_with_2plus_canonical_visits","in_2plus_canonical_visits","At least two canonical visits"),("patients_with_2plus_essdai_assessments","in_2plus_essdai_assessments","At least two complete ESSDAI assessments"),("patients_with_2plus_essdai_assessments_min_gap","in_2plus_essdai_assessments_min_gap","At least two ESSDAI assessments with maximum actual date separation at least minimum gap"),("patients_with_2plus_complete_esspri_assessments","in_2plus_complete_esspri_assessments","At least two complete ESSPRI assessments"),("patients_with_2plus_complete_esspri_assessments_min_gap","in_2plus_complete_esspri_assessments_min_gap","At least two complete ESSPRI assessments separated by minimum gap"),("patients_with_2plus_complete_sf36_assessments","in_2plus_complete_sf36_assessments","At least two complete SF-36 assessments"),("patients_with_2plus_complete_sf36_assessments_min_gap","in_2plus_complete_sf36_assessments_min_gap","At least two complete SF-36 assessments separated by minimum gap"),("patients_with_1plus_evaluable_pop_transition","in_1plus_evaluable_pop_transition","At least one consecutive evaluable Pop transition"),("patients_with_1plus_evaluable_overlap_transition","in_1plus_evaluable_overlap_transition","At least one consecutive evaluable overlap transition"),("patients_with_1plus_full_multidomain_complete_visit","in_1plus_full_multidomain_complete_visit","At least one full multidomain complete-case visit"),("patients_with_2plus_full_multidomain_complete_visits","in_2plus_full_multidomain_complete_visits","At least two full multidomain complete-case visits"),("patients_with_2plus_full_multidomain_complete_visits_min_gap","in_2plus_full_multidomain_complete_visits_min_gap","At least two full multidomain complete visits separated by minimum gap"),("patients_with_1plus_extended_multidomain_complete_visit","in_1plus_extended_multidomain_complete_visit","At least one extended multidomain complete-case visit")]
    n=len(membership); rows=[]
    for name,col,definition in defs:
        ids=set(membership.loc[membership[col].eq(True),"patient_id"]); rows.append({"cohort_name":name,"definition":definition,"number_of_patients":len(ids),"percentage_of_all_canonical_patients":100*len(ids)/n if n else np.nan,"number_of_eligible_visits":int(df.patient_id.isin(ids).sum()),"parent_denominator":"All canonical patients","nesting":"non-nested analytical branch" if name not in ("all_canonical_patients","observed_baseline_cohort") else "root/baseline branch","minimum_measurement_gap_days":min_gap_days if "min_gap" in name else np.nan,"membership_column_name":col})
    return pd.DataFrame(rows)


def model_feature_columns(target:str,history:bool=False)->List[str]:
    base=["protocol_group","calendar_period","is_observed_baseline","time_since_observed_baseline_years"]
    return (["protocol_group","calendar_period","time_since_observed_baseline_years","time_from_previous_visit_years","previous_"+target+"_available","previous_essdai","previous_overlap_status"] if history else base)


def _empty_model(status:str,name:str)->pd.DataFrame:
    x=pd.DataFrame(columns=MODEL_COLUMNS); x.loc[0]=[pd.NA,pd.NA,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,pd.NA,0,0,name,pd.NA,status,"Association with observation availability; not a clinical or causal effect."]; return x


def fit_observation_models(df:pd.DataFrame,skip_models:bool=False)->Tuple[Dict[str,pd.DataFrame],pd.DataFrame,pd.DataFrame]:
    outputs={}; diagnostics=[]; warnings=[]
    for target,outcome in [("esspri","avail_esspri_complete"),("sf36","avail_sf36_complete")]:
        status="skipped_by_user" if skip_models else "pending"; known=df[outcome].notna(); d=df.loc[known].copy(); events=int(d[outcome].eq(True).sum()); nonevents=len(d)-events; predictors=[c for c in model_feature_columns(target) if c in d]; npar=max(1,len(predictors)); varied=d.groupby("patient_id")[outcome].nunique().gt(1).sum()
        if not skip_models:
            if d[outcome].nunique()<2: status="skipped_no_outcome_variation"
            elif d.patient_id.nunique()<20: status="skipped_fewer_than_20_clusters"
            elif events<10 or nonevents<10: status="skipped_insufficient_outcome_counts"
            elif min(events,nonevents)/npar<5: status="skipped_low_events_per_parameter"
        diagnostics.append({"model_name":target+"_design_process_gee","n_visits":len(d),"n_unique_patients":d.patient_id.nunique(),"n_outcome_available":events,"percentage_outcome_available":100*events/len(d) if len(d) else np.nan,"n_outcome_unavailable":nonevents,"n_candidate_parameters":npar,"events_per_parameter":min(events,nonevents)/npar,"predictor_missingness":json.dumps({c:int(d[c].isna().sum()) for c in predictors}),"n_patients_with_within_patient_variation":int(varied),"original_formula":outcome+" ~ "+" + ".join(predictors),"simplified_formula":"","simplification_reason":"","convergence_status":status})
        if status!="pending": outputs[target]=_empty_model(status,target+"_design_process_gee"); continue
        try:
            import statsmodels.api as sm
            from statsmodels.genmod.cov_struct import Exchangeable
            use=d[[outcome,"patient_id"]+predictors].dropna().copy(); cats=[c for c in ["protocol_group","calendar_period"] if c in predictors]; x=pd.get_dummies(use[predictors],columns=cats,drop_first=True,dtype=float); x=sm.add_constant(x.astype(float),has_constant="add"); y=use[outcome].astype(int)
            result=sm.GEE(y,x,groups=use.patient_id,family=sm.families.Binomial(),cov_struct=Exchangeable()).fit(); ci=result.conf_int(); rows=[]
            for p in result.params.index: rows.append({"predictor":p,"level_or_contrast":p,"coefficient":result.params[p],"standard_error":result.bse[p],"odds_ratio":np.exp(result.params[p]),"ci_95_lower":np.exp(ci.loc[p,0]),"ci_95_upper":np.exp(ci.loc[p,1]),"p_value":result.pvalues[p],"reference_category":"First sorted category (categorical predictors)","n_visits":len(use),"n_patients":use.patient_id.nunique(),"model_name":target+"_design_process_gee","model_formula":diagnostics[-1]["original_formula"],"model_status":"fitted","interpretation_scope":"Association with observation availability; not a clinical or causal effect."})
            outputs[target]=pd.DataFrame(rows,columns=MODEL_COLUMNS); diagnostics[-1]["convergence_status"]="fitted"
        except ImportError:
            status="skipped_dependency_unavailable"; outputs[target]=_empty_model(status,target+"_design_process_gee"); diagnostics[-1]["convergence_status"]=status; warnings.append({"model":target,"status":status,"message":"statsmodels unavailable"})
        except Exception as exc:
            status="failed_gee"; outputs[target]=_empty_model(status,target+"_design_process_gee"); diagnostics[-1]["convergence_status"]=status; warnings.append({"model":target,"status":status,"message":str(exc)})
    return outputs,pd.DataFrame(diagnostics),pd.DataFrame(warnings,columns=["model","status","message"])

def planned_output_paths()->List[Path]:
    names=["20_availability_overall.csv","20_availability_by_protocol.csv","20_availability_by_calendar_year.csv","20_availability_by_calendar_period.csv","20_availability_by_followup_bin.csv","20_availability_by_visit_number.csv","20_availability_by_baseline_pop.csv","20_completeness_patterns_overall.csv","20_completeness_patterns_by_protocol.csv","20_completeness_patterns_by_calendar_period.csv","20_joint_availability_matrix.csv","20_pairwise_coavailability.csv","20_patient_observation_patterns_summary.csv","20_analytic_cohort_counts.csv","20_esspri_availability_model.csv","20_sf36_availability_model.csv","20_model_eligibility_and_diagnostics.csv"]
    paths=[TABLES_DIR/n for n in names]; paths += [FIGURES_DIR/n for n in ["20_availability_heatmap_protocol_time.pdf","20_completeness_patterns_bar.pdf","20_patient_observation_patterns.pdf","20_analytic_cohort_flow.pdf"]]
    paths += [QC_DIR/n for n in ["20_observation_process_qc.json","20_pro_detail_merge_summary.csv","20_shared_column_discrepancies.csv","20_indicator_discrepancies.csv","20_observation_state_discrepancies.csv","20_count_range_violations.csv","20_missing_recommended_columns.csv","20_model_failures_or_warnings.csv"]]
    paths += [INTERMEDIATE_DIR/n for n in ["20_observation_process_patient_visit.parquet","20_observation_process_patient_visit.csv","20_observation_process_patient_summary.parquet","20_observation_process_patient_summary.csv","20_analytic_cohort_membership_patient.parquet","20_analytic_cohort_membership_patient.csv"]]; return paths


def make_figures(df:pd.DataFrame,patterns:pd.DataFrame,patient:pd.DataFrame,counts:pd.DataFrame)->None:
    # Protocol x period complete-instrument mean (unknown categories retained).
    heat=df.pivot_table(index="protocol_group",columns="calendar_period",values="n_pro_instruments_complete",aggfunc="mean",dropna=False)
    fig,ax=plt.subplots(figsize=(8,4)); im=ax.imshow(heat.fillna(0),aspect="auto",cmap="Blues"); ax.set_xticks(range(len(heat.columns)),heat.columns,rotation=30,ha="right"); ax.set_yticks(range(len(heat.index)),heat.index); ax.set_title("Mean complete PRO instruments by protocol and period\nDenominator: all canonical visits in each cell"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(FIGURES_DIR/"20_availability_heatmap_protocol_time.pdf"); plt.close(fig)
    top=patterns.nlargest(10,"number_of_visits"); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(range(len(top)),top.number_of_visits); ax.set_yticks(range(len(top)),top.visit_observation_pattern); ax.invert_yaxis(); ax.set_xlabel("Canonical visits"); ax.set_title("Most frequent exact observation patterns"); fig.tight_layout(); fig.savefig(FIGURES_DIR/"20_completeness_patterns_bar.pdf"); plt.close(fig)
    summary=[]
    for t in ["esspri","sf36"]:
        z=patient[t+"_observation_pattern"].value_counts(); summary.extend((t,k,v) for k,v in z.items())
    s=pd.DataFrame(summary,columns=["target","pattern","n"]); fig,ax=plt.subplots(figsize=(9,5)); piv=s.pivot(index="pattern",columns="target",values="n").fillna(0); piv.plot.bar(ax=ax); ax.set_ylabel("Patients"); ax.set_title("Patient observation-pattern classifications"); fig.tight_layout(); fig.savefig(FIGURES_DIR/"20_patient_observation_patterns.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,8)); ax.axis("off"); root=counts.iloc[0]; ax.text(.5,.92,"Analytical-cohort availability\nAll canonical patients: n=%d"%root.number_of_patients,ha="center",va="center",bbox=dict(boxstyle="round",fc="#dceafa"),fontsize=12)
    shown=counts.iloc[1:]; ys=np.linspace(.78,.08,len(shown));
    for y,(_,r) in zip(ys,shown.iterrows()): ax.plot([.5,.18],[.89,y],color="0.7",lw=.7); ax.text(.18,y,"%s\nn=%d (%.1f%%)"%(r.cohort_name.replace("_"," "),r.number_of_patients,r.percentage_of_all_canonical_patients),ha="left",va="center",fontsize=7,bbox=dict(boxstyle="round",fc="white",ec="0.6"))
    fig.tight_layout(); fig.savefig(FIGURES_DIR/"20_analytic_cohort_flow.pdf"); plt.close(fig)


def _write_df(df:pd.DataFrame,path:Path)->None:
    if path.suffix==".parquet": df.to_parquet(path,index=False)
    else: df.to_csv(path,index=False)


def write_outputs(df:pd.DataFrame,patient:pd.DataFrame,membership:pd.DataFrame,tables:Dict[str,pd.DataFrame],qc_tables:Dict[str,pd.DataFrame],qc:Dict[str,Any],overwrite:bool=True)->None:
    paths=planned_output_paths(); conflicts=[str(p) for p in paths if p.exists()]
    if not overwrite and conflicts: raise FileExistsError("--no-overwrite conflicts with planned outputs:\n"+"\n".join(conflicts))
    common.ensure_output_dirs()
    for stem,data in [("20_observation_process_patient_visit",df),("20_observation_process_patient_summary",patient),("20_analytic_cohort_membership_patient",membership)]:
        _write_df(data,INTERMEDIATE_DIR/(stem+".parquet")); _write_df(data,INTERMEDIATE_DIR/(stem+".csv"))
    mapping={"availability_overall":"20_availability_overall.csv","availability_protocol":"20_availability_by_protocol.csv","availability_year":"20_availability_by_calendar_year.csv","availability_period":"20_availability_by_calendar_period.csv","availability_followup":"20_availability_by_followup_bin.csv","availability_visit":"20_availability_by_visit_number.csv","availability_baseline_pop":"20_availability_by_baseline_pop.csv","patterns_overall":"20_completeness_patterns_overall.csv","patterns_protocol":"20_completeness_patterns_by_protocol.csv","patterns_period":"20_completeness_patterns_by_calendar_period.csv","joint":"20_joint_availability_matrix.csv","pairwise":"20_pairwise_coavailability.csv","patient_patterns":"20_patient_observation_patterns_summary.csv","cohort_counts":"20_analytic_cohort_counts.csv","esspri_model":"20_esspri_availability_model.csv","sf36_model":"20_sf36_availability_model.csv","model_diagnostics":"20_model_eligibility_and_diagnostics.csv"}
    for k,n in mapping.items(): tables[k].to_csv(TABLES_DIR/n,index=False)
    qmap={"merge":"20_pro_detail_merge_summary.csv","shared":"20_shared_column_discrepancies.csv","indicator":"20_indicator_discrepancies.csv","state":"20_observation_state_discrepancies.csv","range":"20_count_range_violations.csv","missing":"20_missing_recommended_columns.csv","model_warnings":"20_model_failures_or_warnings.csv"}
    for k,n in qmap.items(): qc_tables[k].to_csv(QC_DIR/n,index=False)
    with open(QC_DIR/"20_observation_process_qc.json","w",encoding="utf-8") as f: json.dump(qc,f,indent=2,default=str)
    make_figures(df,tables["patterns_overall"],patient,tables["cohort_counts"])


def main(argv:Optional[Sequence[str]]=None)->None:
    args=parse_args(argv); logger=setup_logging(); logger.info("Integrated input: %s; PRO detail: %s",args.input,args.pros_input)
    raw=load_integrated_data(args.input); data,range1=normalize_dtypes(raw); base_qc=validate_integrated_input(data)
    pro,missing_pro=load_pro_detail(args.pros_input); pro,range2=normalize_dtypes(pro); ranges=pd.concat([range1,range2],ignore_index=True)
    if len(ranges):
        ranges.to_csv(QC_DIR/"20_count_range_violations.csv",index=False); raise ValueError("Fatal noninteger or out-of-range item counts detected")
    merged,merge_summary,shared=merge_pro_detail(data,pro); states,state_disc=derive_observation_states(merged); visits=derive_completeness_indicators(states)
    if len(visits)!=len(raw): raise RuntimeError("Integrated output row count differs from integrated input")
    patient=build_patient_level_outputs(visits); membership=build_analytic_cohort_membership(visits,args.min_gap_days); cohorts=build_analytic_cohort_counts(membership,visits,args.min_gap_days); joint=build_joint_completeness_tables(visits); models,diagnostics,model_warnings=fit_observation_models(visits,args.skip_models)
    missing=sorted(set(RECOMMENDED_DESIGN_COLUMNS+RECOMMENDED_COLUMNS+PRO_DETAIL_COLUMNS)-set(merged.columns)); missing_df=pd.DataFrame({"column":missing,"classification":["PRO detail" if c in PRO_DETAIL_COLUMNS else "recommended integrated/design" for c in missing]})
    indicator=[]
    pairs=[("has_essdai","essdai_total"),("has_esspri","esspri_total"),("has_sf36_pcs","sf36_pcs"),("has_sf36_mcs","sf36_mcs"),("has_profad","profad_total"),("has_mdafs","mdafs_global")]
    for flag,score in pairs:
        if flag in visits and score in visits:
            bad=visits[flag].notna() & visits[flag].ne(visits[score].notna());
            for _,r in visits.loc[bad,KEY_COLUMNS].iterrows(): indicator.append({"patient_id":r.patient_id,"visit_id":r.visit_id,"indicator":flag,"score":score})
    tables={"availability_overall":summarize_availability(visits),"availability_protocol":summarize_availability(visits,"protocol_group"),"availability_year":summarize_availability(visits,"calendar_year"),"availability_period":summarize_availability(visits,"calendar_period"),"availability_followup":summarize_availability(visits,"followup_time_bin"),"availability_visit":summarize_availability(visits,"visit_number_group"),"availability_baseline_pop":summarize_availability(visits,"baseline_pop_group"),**joint,"patient_patterns":pd.concat([patient.esspri_observation_pattern.value_counts().rename_axis("pattern").reset_index(name="n_patients").assign(target="ESSPRI"),patient.sf36_observation_pattern.value_counts().rename_axis("pattern").reset_index(name="n_patients").assign(target="SF-36")]),"cohort_counts":cohorts,"esspri_model":models["esspri"],"sf36_model":models["sf36"],"model_diagnostics":diagnostics}
    qc={"n_integrated_rows_input":len(raw),"n_integrated_rows_output":len(visits),"n_unique_patients":int(visits.patient_id.nunique()),"n_unique_visit_ids":int(visits.visit_id.nunique()),**base_qc,"n_pro_detail_rows":len(pro),"n_pro_detail_matched_visits":int(merge_summary.iloc[0].n_pro_detail_matched_visits),"n_pro_detail_unmatched_visits":int(merge_summary.iloc[0].n_pro_detail_unmatched_visits),"n_shared_field_disagreements":len(shared),"n_indicator_disagreements":len(indicator),"n_observation_state_disagreements":len(state_disc),"n_count_range_violations":0,"missing_recommended_columns":missing,"n_visits_complete_essdai":int(visits.avail_essdai.eq(True).sum()),"n_visits_complete_esspri":int(visits.avail_esspri_complete.eq(True).sum()),"n_visits_complete_sf36":int(visits.avail_sf36_complete.eq(True).sum()),"n_visits_complete_profad":int(visits.avail_profad_complete.eq(True).sum()),"n_visits_complete_mdafs":int(visits.avail_mdafs_complete.eq(True).sum()),"n_visits_evaluable_overlap":int(visits.avail_overlap_evaluable.eq(True).sum()),"n_visits_pop_core_complete":int(visits.pop_core_complete.eq(True).sum()),"n_visits_systemic_overlap_core_complete":int(visits.systemic_overlap_core_complete.eq(True).sum()),"n_visits_pro_core_complete":int(visits.pro_core_complete.eq(True).sum()),"n_visits_full_multidomain_complete":int(visits.full_multidomain_complete.eq(True).sum()),"n_visits_extended_multidomain_complete":int(visits.extended_multidomain_complete.eq(True).sum()),"n_patients_essdai_longitudinal_eligible":int(membership.in_2plus_essdai_assessments.sum()),"n_patients_essdai_longitudinal_min_gap_eligible":int(membership.in_2plus_essdai_assessments_min_gap.sum()),"n_patients_esspri_longitudinal_eligible":int(membership.in_2plus_complete_esspri_assessments.sum()),"n_patients_esspri_longitudinal_min_gap_eligible":int(membership.in_2plus_complete_esspri_assessments_min_gap.sum()),"n_patients_sf36_longitudinal_eligible":int(membership.in_2plus_complete_sf36_assessments.sum()),"n_patients_sf36_longitudinal_min_gap_eligible":int(membership.in_2plus_complete_sf36_assessments_min_gap.sum()),"min_gap_days":args.min_gap_days,"models_requested":not args.skip_models,"esspri_model_status":str(models["esspri"].model_status.iloc[0]),"sf36_model_status":str(models["sf36"].model_status.iloc[0])}
    qct={"merge":merge_summary,"shared":shared,"indicator":pd.DataFrame(indicator,columns=["patient_id","visit_id","indicator","score"]),"state":state_disc,"range":ranges,"missing":missing_df,"model_warnings":model_warnings}; write_outputs(visits,patient,membership,tables,qct,qc,args.overwrite)
    logger.info("Rows=%d patients=%d; merge matched=%d; warnings=%d; model statuses: ESSPRI=%s SF-36=%s",len(visits),visits.patient_id.nunique(),qc["n_pro_detail_matched_visits"],len(missing)+len(shared)+len(indicator)+len(state_disc)+len(model_warnings),qc["esspri_model_status"],qc["sf36_model_status"])

if __name__=="__main__": main()

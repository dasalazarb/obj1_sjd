#!/usr/bin/env python3
"""Section 5: rheumatological condition status and historical documentation.

The codebook semantics are enforced here: ``rheumatological_comorbidities__``
fields are status-coded at baseline, while ``past_medical_history__`` and
``sjogren's_syndrome_history__`` fields are historical documentation only.
Those historical families are never used to create comorbidity events,
rheumatological prevalence, or progression-model exposures.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - import-time tests do not need parquet I/O.
    pq = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402
import config  # noqa: E402
from src.derivations.visit_dates import add_parsed_visit_dates  # noqa: E402

PATIENT_ID_COL = "ids__patient_record_number"
VISIT_DATE_COL = "ids__visit_date"
BIRTH_DATE_CANDIDATES = ("ids__date_of_birth", "ids__birth_date", "demographics__date_of_birth")
ESSDAI_PRIMARY_COL = config.ESSDAI_TOTAL_RAW
SEVERE_THRESHOLD = config.ESSDAI_SEVERE
SCRIPT_VERSION = "2.0.0"
RANDOM_SEED = 20260728

TABLES_DIR = common.OUTPUTS_DIR / "tables" / "blockA"
FIGURES_DIR = common.OUTPUTS_DIR / "figures" / "blockA"
QC_DIR = common.OUTPUTS_DIR / "qc" / "blockA"
LOG_PATH = common.OUTPUTS_DIR / "logs" / "07_comorbidities.log"
BASELINE_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidities_baseline_patient.parquet"
LONGITUDINAL_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidities_analysis_longitudinal.parquet"
SEVERE_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidity_severe5_survival.parquet"
NEW_DOMAIN_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidity_new_domain_survival.parquet"
DOMAIN_AUDIT_PATH = common.INTERMEDIATE_DATA_DIR / "07_comorbidity_new_domain_patient_domain.parquet"

STATUS_ORDER = ["confirmed_present", "history_only", "documented_status_unspecified", "not_documented"]
STATUS_LABELS = {
    "confirmed_present": "Confirmed/present",
    "history_only": "History only; current status unknown",
    "documented_status_unspecified": "Documented; status unspecified",
}

@dataclass(frozen=True)
class Condition:
    name: str
    label: str
    general: tuple[str, ...]
    history: tuple[str, ...] = ()
    confirmed: tuple[str, ...] = ()
    category: str = "rheumatological"
    model_eligible: bool = False
    notes: str = ""

EXISTING_AND_RHEUMATOLOGICAL_CONDITIONS = [
    Condition("sle", "Systemic lupus erythematosus", ("rheumatological_comorbidities__sle1",), ("rheumatological_comorbidities__sle_hx",), ("rheumatological_comorbidities__sle_confirmed",), "concomitant_said", True),
    Condition("rheumatoid_arthritis", "Rheumatoid arthritis", ("rheumatological_comorbidities__ra",), ("rheumatological_comorbidities__ra_hx",), ("rheumatological_comorbidities__ra_confirm",), "concomitant_said", True),
    Condition("systemic_sclerosis", "Systemic sclerosis", ("rheumatological_comorbidities__systemic_sclerosis",), ("rheumatological_comorbidities__systmc_sclerosis_hx",), ("rheumatological_comorbidities__systmc_sclerosis_confirm",), "concomitant_said", True),
    Condition("mixed_connective_tissue_disease", "Mixed connective tissue disease", ("rheumatological_comorbidities__mixed_connective_tissue_disease",), ("rheumatological_comorbidities__mixed_connect_tissue_hx",), ("rheumatological_comorbidities__mixed_connect_tissue_confirm",), "concomitant_said", True),
    Condition("polymyositis", "Polymyositis", ("rheumatological_comorbidities__polymyositis",), ("rheumatological_comorbidities__polymyositis_hx",), ("rheumatological_comorbidities__polymyositis_confirm",), "concomitant_said", True),
    Condition("dermatomyositis", "Dermatomyositis", ("rheumatological_comorbidities__dermatomyositis",), ("rheumatological_comorbidities__dermatomyositis_hx",), ("rheumatological_comorbidities__dermatomyositis_confirm",), "concomitant_said", True),
    Condition("antiphospholipid_syndrome", "Antiphospholipid syndrome", ("rheumatological_comorbidities__antiphospholipid_syndrome",), ("rheumatological_comorbidities__antiphospholipid_syn_hx",), ("rheumatological_comorbidities__antiphospholipid_syn_confirm",), "concomitant_said", True),
    Condition("fibromyalgia", "Fibromyalgia", ("rheumatological_comorbidities__fibromyalgia1",), ("rheumatological_comorbidities__fibromyalgia1_hx",), ("rheumatological_comorbidities__fibromyalgia1_confirm",), "other_rheumatological_musculoskeletal"),
    Condition("osteoporosis", "Osteoporosis", ("rheumatological_comorbidities__osteoporosis1",), ("rheumatological_comorbidities__osteoporosis1_hx",), ("rheumatological_comorbidities__osteoporosis1_confirm",), "other_rheumatological_musculoskeletal"),
    Condition("osteopenia", "Osteopenia", ("rheumatological_comorbidities__osteopenia",), ("rheumatological_comorbidities__osteopenia_hx",), ("rheumatological_comorbidities__osteopenia_confirm",), "other_rheumatological_musculoskeletal"),
    Condition("osteoarthritis", "Osteoarthritis", ("rheumatological_comorbidities__osteoarthritis",), ("rheumatological_comorbidities__osteoarthritis_hx",), ("rheumatological_comorbidities__osteoarthritis_confirm",), "other_rheumatological_musculoskeletal"),
    Condition("crystalline_arthropathy", "Crystalline arthropathy", ("rheumatological_comorbidities__crystalline_arthropathy",), ("rheumatological_comorbidities__crystalline_arthropathy_hx",), ("rheumatological_comorbidities__crystalline_arthro_confirm",), "other_rheumatological_musculoskeletal"),
    Condition("raynaud", "Raynaud's phenomenon", ("rheumatological_comorbidities__integ_raynds",), ("rheumatological_comorbidities__integ_raynds_hx",), ("rheumatological_comorbidities__integ_raynds_confirm",), "associated_manifestation"),
    Condition("cryoglobulinemia", "Cryoglobulinemia", ("rheumatological_comorbidities__cryoglobulinemia",), ("rheumatological_comorbidities__cryoglobulinemia_hx",), ("rheumatological_comorbidities__cryoglobulinemia_confirm",), "associated_manifestation"),
    Condition("primary_biliary_cholangitis", "Primary biliary cholangitis", ("rheumatological_comorbidities__primary_billiary_cirrhosis",), ("rheumatological_comorbidities__prim_billiary_cirrhosis_hx",), ("rheumatological_comorbidities__prim_billiary_cirrhosis_confirm",), "non_rheumatological_immune_mediated"),
    Condition("inflammatory_bowel_disease", "Inflammatory bowel disease", ("rheumatological_comorbidities__inflam_bowel",), ("rheumatological_comorbidities__inflam_bowel_hx",), ("rheumatological_comorbidities__inflam_bowel_confirm",), "non_rheumatological_immune_mediated"),
    Condition("sarcoidosis", "Sarcoidosis", ("rheumatological_comorbidities__sarcoidosis",), ("rheumatological_comorbidities__sarcoidosis_hx",), ("rheumatological_comorbidities__sarcoidosis_confirm",), "non_rheumatological_immune_mediated"),
    Condition("other_rheumatological_condition", "Other rheumatological condition", ("rheumatological_comorbidities__rheumatological_other",), ("rheumatological_comorbidities__rheumatological_other_hx",), ("rheumatological_comorbidities__rheumatological_other_confirm",), "other_rheumatological_condition", False, "Other text requires manual audit; not automatically counted as SAID."),
]
CONDITIONS = EXISTING_AND_RHEUMATOLOGICAL_CONDITIONS
CONDITION_NAMES = [c.name for c in CONDITIONS]
PROGRESSION_CONDITIONS = [c for c in CONDITIONS if c.model_eligible]
PROGRESSION_CONDITION_NAMES_ORDERED = [c.name for c in PROGRESSION_CONDITIONS]

SJOGREN_HISTORY_TERMS = [
    ("sjogrens_dx", "Sjögren diagnosis history", "diagnostic_history", ("sjogren's_syndrome_history__sjogrens_dx",), ("sjogren's_syndrome_history__sjogrens_dx_date",)),
    ("dry_eyes", "Dry eyes", "historical_glandular_manifestation", ("sjogren's_syndrome_history__dry_eyes",), ("sjogren's_syndrome_history__dry_eye_onset", "sjogren's_syndrome_history__dry_eyes_onset")),
    ("dry_mouth", "Dry mouth", "historical_glandular_manifestation", ("sjogren's_syndrome_history__dry_mouth",), ("sjogren's_syndrome_history__dry_mouth_onset",)),
    ("other_dryness", "Other dryness", "historical_glandular_manifestation", ("sjogren's_syndrome_history__other_dryness",), ("sjogren's_syndrome_history__other_dryness_onset",)),
    ("glandular_swelling", "Glandular swelling", "historical_glandular_manifestation", ("sjogren's_syndrome_history__glandular_swelling",), ()),
]
for n, label in [("arthritis","Arthritis"),("autonomic_dysfunction","Autonomic dysfunction"),("chemosensory_involvement","Chemosensory involvement"),("cholecystitis_cholangitis","Cholecystitis/cholangitis"),("cns_cognitive_involvement","CNS/cognitive involvement"),("fatigue","Fatigue"),("interstitial_cystitis","Interstitial cystitis"),("liver_involvement","Liver involvement"),("lymphoma","Lymphoma"),("myositis_myalgia","Myositis/myalgia"),("cranial_peripheral_neuropathy","Cranial/peripheral neuropathy"),("pancreatitis","Pancreatitis"),("pulmonary_involvement","Pulmonary involvement"),("raynaud_phenom","Raynaud's phenomenon"),("renal_involvement","Renal involvement"),("skin_involvement","Skin involvement"),("thyroiditis","Thyroiditis"),("vasculitis","Vasculitis"),("other_non_sicca","Other non-sicca manifestation")]:
    SJOGREN_HISTORY_TERMS.append((n, label, "historical_systemic_or_extraglandular_manifestation", (f"sjogren's_syndrome_history__{n}",), (f"sjogren's_syndrome_history__{n}_onset", f"sjogren's_syndrome_history__{n}_date")))

_UNRECOGNIZED: list[dict[str, Any]] = []

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    p.add_argument("--rebuild-upstream", action="store_true")
    p.add_argument("--monte-carlo-replicates", type=int, default=2000)
    p.add_argument("--minimum-events", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    p.add_argument("--skip-models", action="store_true")
    return p.parse_args(argv)

def ensure_directories():
    for p in (TABLES_DIR, FIGURES_DIR, QC_DIR, LOG_PATH.parent, common.INTERMEDIATE_DATA_DIR): p.mkdir(parents=True, exist_ok=True)

def setup_logging():
    logger = logging.getLogger("07_comorbidities"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()): h.setFormatter(fmt); logger.addHandler(h)
    return logger

def _schema(path: Path) -> list[str]:
    if pq is None: raise ModuleNotFoundError("pyarrow is required to read parquet inputs")
    return pq.read_schema(path).names

def normalize_binary_flag(series: pd.Series) -> pd.Series:
    pos={"1","1.0","true","yes","y","positive","present","confirmed","history"}; neg={"0","0.0","false","no","n","negative","absent"}; miss={str(x).strip().lower() for x in config.MISSING_STRINGS}|{""}
    out=pd.Series(pd.NA,index=series.index,dtype="boolean",name=series.name)
    for i,v in series.items():
        if v is None or pd.isna(v): continue
        t=str(v).strip().lower()
        if t in pos: out.at[i]=True
        elif t in neg: out.at[i]=False
        elif t not in miss: _UNRECOGNIZED.append({"source_column":str(series.name),"original_value":str(v),"row_index":str(i)})
    return out

def _any_flag(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    flags=[normalize_binary_flag(df[c]) for c in cols if c in df]
    if not flags: return pd.Series(False,index=df.index,dtype="boolean")
    return pd.concat(flags,axis=1).fillna(False).astype(bool).any(axis=1).astype("boolean")

def selected_raw_columns(input_path: Path) -> list[str]:
    schema=set(_schema(input_path)); desired={PATIENT_ID_COL,VISIT_DATE_COL,ESSDAI_PRIMARY_COL,"visit_id","patient_id","visit_number","visit_date","baseline_pop","baseline_pop_status","pop_status","observed_baseline_date","time_since_observed_baseline_years","age_at_visit","sex",*BIRTH_DATE_CANDIDATES}
    for c in CONDITIONS: desired.update(c.general+c.history+c.confirmed)
    desired.add("rheumatological_comorbidities__rheumatological_specify")
    desired.update(x for x in schema if x.startswith("past_medical_history__") or x.startswith("sjogren's_syndrome_history__"))
    return [c for c in desired if c in schema]

def load_selected_raw_columns(input_path: Path) -> pd.DataFrame:
    return pd.read_parquet(input_path, columns=selected_raw_columns(input_path))

def derive_condition_status(raw: pd.DataFrame) -> pd.DataFrame:
    out=pd.DataFrame(index=raw.index)
    for c in CONDITIONS:
        g,h,cf=(_any_flag(raw,c.general), _any_flag(raw,c.history), _any_flag(raw,c.confirmed))
        status=np.select([cf,h & ~cf,g & ~h & ~cf],["confirmed_present","history_only","documented_status_unspecified"],default="not_documented")
        out[f"{c.name}_status"]=pd.Categorical(status,categories=STATUS_ORDER,ordered=True)
        out[c.name]=pd.Series(status,index=raw.index).eq("confirmed_present").astype("boolean")
        out[f"{c.name}_history_and_confirmed_conflict"]=(h & cf).astype("boolean")
        out[f"{c.name}_general_and_no_status_flag"]=(g & ~h & ~cf).astype("boolean")
    if {"polymyositis","dermatomyositis"}.issubset(out.columns):
        out["inflammatory_myopathy_any"]=(out["polymyositis"] | out["dermatomyositis"]).astype("boolean")
    return out

def _prepare_ids(raw: pd.DataFrame) -> pd.DataFrame:
    df=raw.copy()
    if "patient_id" not in df: df=df.rename(columns={PATIENT_ID_COL:"patient_id"})
    if "visit_date" not in df: df=add_parsed_visit_dates(df, PATIENT_ID_COL if PATIENT_ID_COL in df else "patient_id", VISIT_DATE_COL if VISIT_DATE_COL in df else "visit_date")
    df["visit_date"]=pd.to_datetime(df["visit_date"], errors="coerce")
    return df

def build_baseline_comorbidity_dataset(raw: pd.DataFrame, spine: pd.DataFrame|None=None, pop: pd.DataFrame|None=None):
    df=_prepare_ids(raw); status=derive_condition_status(df); df=pd.concat([df,status],axis=1)
    sort_cols=["patient_id"] + (["visit_number"] if "visit_number" in df else ["visit_date"])
    base=df.sort_values(sort_cols).drop_duplicates("patient_id", keep="first").copy()
    if "baseline_pop" not in base:
        base["baseline_pop"]=base.get("baseline_pop_status", base.get("pop_status", pd.NA))
    if "baseline_date" not in base: base["baseline_date"]=base["visit_date"]
    base[CONDITION_NAMES]=base[CONDITION_NAMES].fillna(False).astype("boolean")
    base["n_prespecified_comorbidities"]=base[PROGRESSION_CONDITION_NAMES_ORDERED].astype(int).sum(axis=1)
    base["any_comorbidity"]=(base["n_prespecified_comorbidities"]>0).astype("boolean")
    base["two_or_more_comorbidities"]=(base["n_prespecified_comorbidities"]>=2).astype("boolean")
    return base, pd.DataFrame(columns=["patient_id","visit_date","n_source_rows"]), 0

def summarize_overall_prevalence(base: pd.DataFrame) -> pd.DataFrame:
    rows=[]; N=len(base)
    for c in CONDITIONS:
        if f"{c.name}_status" in base:
            counts=base[f"{c.name}_status"].astype("string").value_counts().to_dict(); pos=int(counts.get("confirmed_present",0))
            hist=int(counts.get("history_only",0)); uns=int(counts.get("documented_status_unspecified",0)); neg=int(counts.get("not_documented",0))
        else:
            s=base[c.name].fillna(False).astype("boolean"); pos=int(s.eq(True).sum()); hist=uns=0; neg=N-pos
        rows.append({"condition":c.name,"display_label":c.label,"category":c.category,"source_columns":"|".join(c.confirmed),"n_baseline_total":N,"n_total_cohort":N,"n_evaluable":N,"n_baseline_positive":pos,"n_positive":pos,"n_negative":neg,"n_missing":0,"pct_baseline":100*pos/N if N else np.nan,"pct_total_cohort":100*pos/N if N else np.nan,"pct_among_evaluable":100*pos/N if N else np.nan,"n_history_only":hist,"n_documented_status_unspecified":uns,"prevalence_status":"confirmed_present_only","notes":c.notes})
    return pd.DataFrame(rows).sort_values(["pct_total_cohort","display_label"],ascending=[False,True]).reset_index(drop=True)

def summarize_prevalence_by_pop(base: pd.DataFrame, replicates:int=2000, seed:int=RANDOM_SEED) -> pd.DataFrame:
    rows=[]; N=len(base)
    for c in CONDITIONS:
        row={"condition":c.name,"display_label":c.label,"category":c.category,"n_positive_total_cohort":int(base[c.name].fillna(False).eq(True).sum()),"pct_total_cohort":100*int(base[c.name].fillna(False).eq(True).sum())/N if N else np.nan,"prevalence_status":"confirmed_present_only"}
        for i,pop_name in enumerate(("Pop1","Pop2","Pop3"),1):
            d=base.loc[base["baseline_pop"].eq(pop_name)] if "baseline_pop" in base else base.iloc[0:0]
            s=d[c.name].fillna(False).astype("boolean") if c.name in d else pd.Series(dtype="boolean")
            n=int(s.eq(True).sum()); den=len(d); row.update({f"n_pop{i}":n,f"N_pop{i}":den,f"pct_pop{i}":100*n/den if den else np.nan})
        row.update({"global_test":"descriptive only","global_p_value":np.nan,"minimum_expected_cell":np.nan,"sparse_table_flag":False,"interpretation_status":"Confirmed/present baseline distribution; descriptive, non-causal."})
        rows.append(row)
    return pd.DataFrame(rows)

PAST_MEDICAL_HISTORY_COLUMNS = ["history_item", "display_label", "clinical_category", "n_patients_with_documented_history", "pct_documented_history_frequency", "n_patients_without_positive_record", "source_columns_used", "interpretation_label"]
SJOGREN_HISTORY_COLUMNS = ["history_item", "display_label", "history_group", "n_patients_with_documented_history", "pct_documented_history_frequency", "n_patients_without_positive_record", "source_columns_used", "date_source_columns", "interpretation_label"]
SJOGREN_DATE_COLUMNS = ["history_item", "display_label", "date_source_column", "n_rows_with_valid_date", "n_missing_or_unparseable_date", "n_dates_before_birth", "n_dates_after_visit", "median_date", "q1_date", "q3_date", "min_date", "max_date"]

def descriptive_past_medical_history(raw: pd.DataFrame) -> pd.DataFrame:
    df=_prepare_ids(raw); cols=[c for c in df if c.startswith("past_medical_history__")]; N=df["patient_id"].nunique(); rows=[]
    for col in sorted(cols):
        patient=_any_flag(df,[col]).groupby(df["patient_id"]).max().reindex(df["patient_id"].drop_duplicates()).fillna(False)
        n=int(patient.sum()); name=col.removeprefix("past_medical_history__")
        rows.append({"history_item":name,"display_label":name.replace("_"," ").title(),"clinical_category":name.split("_hx_")[0].split("_")[0],"n_patients_with_documented_history":n,"pct_documented_history_frequency":100*n/N if N else np.nan,"n_patients_without_positive_record":N-n,"source_columns_used":col,"interpretation_label":"Documented past medical history frequency"})
    return pd.DataFrame(rows, columns=PAST_MEDICAL_HISTORY_COLUMNS)

def descriptive_sjogren_history(raw: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    df=_prepare_ids(raw); N=df["patient_id"].nunique(); rows=[]; date_rows=[]; birth=next((c for c in BIRTH_DATE_CANDIDATES if c in df), None)
    for name,label,cat,cols,date_cols in SJOGREN_HISTORY_TERMS:
        present=[c for c in cols if c in df]; dates=[c for c in date_cols if c in df]
        if not present and not dates: continue
        flag=_any_flag(df,present) if present else pd.Series(False,index=df.index,dtype="boolean")
        patient=flag.groupby(df["patient_id"]).max().reindex(df["patient_id"].drop_duplicates()).fillna(False); n=int(patient.sum())
        rows.append({"history_item":name,"display_label":label,"history_group":cat,"n_patients_with_documented_history":n,"pct_documented_history_frequency":100*n/N if N else np.nan,"n_patients_without_positive_record":N-n,"source_columns_used":"|".join(present),"date_source_columns":"|".join(dates),"interpretation_label":"Manifestation histórica documentada"})
        for dc in dates:
            parsed=pd.to_datetime(df[dc], errors="coerce"); valid=parsed.notna(); before_birth=int((valid & (pd.to_datetime(df[birth],errors="coerce")>parsed)).sum()) if birth else 0; after_visit=int((valid & (parsed>df["visit_date"])).sum())
            vals=parsed.loc[valid & (before_birth==0 if False else True)]
            date_rows.append({"history_item":name,"display_label":label,"date_source_column":dc,"n_rows_with_valid_date":int(valid.sum()),"n_missing_or_unparseable_date":int((~valid).sum()),"n_dates_before_birth":before_birth,"n_dates_after_visit":after_visit,"median_date":vals.median(),"q1_date":vals.quantile(.25) if len(vals) else pd.NaT,"q3_date":vals.quantile(.75) if len(vals) else pd.NaT,"min_date":vals.min() if len(vals) else pd.NaT,"max_date":vals.max() if len(vals) else pd.NaT})
    return pd.DataFrame(rows, columns=SJOGREN_HISTORY_COLUMNS), pd.DataFrame(date_rows, columns=SJOGREN_DATE_COLUMNS)

def source_mapping(raw_columns: Iterable[str]) -> pd.DataFrame:
    raw=set(raw_columns); rows=[]
    for c in CONDITIONS:
        rows.append({"condition":c.name,"display_label":c.label,"conceptual_group":c.category,"general_columns":"|".join(c.general),"history_columns":"|".join(c.history),"confirmed_columns":"|".join(c.confirmed),"derivation_rule":"confirmed_present > history_only > documented_status_unspecified > not_documented; no OR across history and confirmed statuses","contributes_to_rheumatological_prevalence":"confirmed_present only","model_exposure_eligible":c.model_eligible,"available_source_columns":"|".join([x for x in c.general+c.history+c.confirmed if x in raw]),"notes":c.notes})
    rows.append({"condition":"past_medical_history__*","display_label":"Past medical history fields","conceptual_group":"documented_past_history_only","derivation_rule":"one patient row descriptive summary only; not prevalence, event, risk set, or model exposure","contributes_to_rheumatological_prevalence":"no","model_exposure_eligible":False})
    rows.append({"condition":"sjogren's_syndrome_history__*","display_label":"Sjögren history fields","conceptual_group":"historical_sjogren_manifestations_only","derivation_rule":"documented history and date QC only; not prevalence, event, risk set, or model exposure","contributes_to_rheumatological_prevalence":"no","model_exposure_eligible":False})
    return pd.DataFrame(rows)

def condition_status_conflicts(base: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for c in CONDITIONS:
        for pid,r in base.iterrows():
            conflict=bool(r.get(f"{c.name}_history_and_confirmed_conflict", False)); unspecified=bool(r.get(f"{c.name}_general_and_no_status_flag", False))
            if conflict or unspecified:
                rows.append({"patient_id":r.get("patient_id"),"condition":c.name,"documented_status":r.get(f"{c.name}_status"),"history_and_confirmed_positive":conflict,"general_positive_without_history_or_confirmed":unspecified,"audit_action":"manual review"})
    if "rheumatological_comorbidities__rheumatological_specify" in raw:
        txt=raw[["rheumatological_comorbidities__rheumatological_specify"]].dropna().drop_duplicates()
        for _, rr in txt.iterrows(): rows.append({"patient_id":pd.NA,"condition":"other_rheumatological_condition","documented_status":"text_audit","original_text":rr.iloc[0],"audit_action":"manual classification pending"})
    return pd.DataFrame(rows)

def create_status_plot(overall: pd.DataFrame):
    d=overall.sort_values("display_label"); fig,ax=plt.subplots(figsize=(10,max(5,.38*len(d))))
    y=np.arange(len(d)); left=np.zeros(len(d))
    for col,color in [("n_baseline_positive","#2166ac"),("n_history_only","#fdae61"),("n_documented_status_unspecified","#bdbdbd")]:
        vals=d[col].to_numpy(float); ax.barh(y,vals,left=left,label={"n_baseline_positive":STATUS_LABELS["confirmed_present"],"n_history_only":STATUS_LABELS["history_only"],"n_documented_status_unspecified":STATUS_LABELS["documented_status_unspecified"]}[col],color=color); left+=vals
    ax.set_yticks(y,d["display_label"]); ax.set_xlabel("Patients at baseline"); ax.set_title("Documented status of rheumatological conditions at baseline"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES_DIR/"07_rheumatological_conditions_status.pdf",bbox_inches="tight"); plt.close(fig)

def create_sjogren_history_plot(desc: pd.DataFrame):
    fig,ax=plt.subplots(figsize=(10,5))
    if desc.empty or "pct_documented_history_frequency" not in desc:
        ax.text(.5,.5,"No clinically interpretable Sjögren history variables were available in the input.",ha="center",va="center",wrap=True)
        ax.axis("off")
    else:
        d=desc.sort_values("pct_documented_history_frequency").tail(25)
        fig.set_size_inches(10,max(5,.35*len(d)))
        ax.barh(np.arange(len(d)),d["pct_documented_history_frequency"],color="#41ab5d")
        ax.set_yticks(np.arange(len(d)),d["display_label"])
        ax.set_xlabel("Patients with documented history (%)")
        ax.set_title("Documented Sjögren's syndrome history")
    fig.tight_layout(); fig.savefig(FIGURES_DIR/"07_sjogren_history_descriptive.pdf",bbox_inches="tight"); plt.close(fig)

def write_baseline_outputs(base: pd.DataFrame):
    try: base.to_parquet(BASELINE_PATH,index=False)
    except Exception: pass
    base.to_csv(BASELINE_PATH.with_suffix(".csv"),index=False)

def main(argv: Sequence[str] | None = None) -> int:
    args=parse_args(argv); ensure_directories(); logger=setup_logging(); logger.info("Loading canonical integrated patient-visit input")
    raw=load_selected_raw_columns(args.input); base,duplicates,n_pipe=build_baseline_comorbidity_dataset(raw); write_baseline_outputs(base)
    overall=summarize_overall_prevalence(base); by_pop=summarize_prevalence_by_pop(base,args.monte_carlo_replicates,args.random_seed)
    pmh=descriptive_past_medical_history(raw); sj_desc,sj_dates=descriptive_sjogren_history(raw); mapping=source_mapping(raw.columns); conflicts=condition_status_conflicts(base,raw)
    overall.to_csv(TABLES_DIR/"07_rheumatological_conditions_status_overall.csv",index=False); by_pop.to_csv(TABLES_DIR/"07_rheumatological_conditions_status_by_pop.csv",index=False); pmh.to_csv(TABLES_DIR/"07_past_medical_history_descriptive.csv",index=False); sj_desc.to_csv(TABLES_DIR/"07_sjogren_history_descriptive.csv",index=False); sj_dates.to_csv(TABLES_DIR/"07_sjogren_history_dates_summary.csv",index=False); mapping.to_csv(TABLES_DIR/"07_condition_source_mapping.csv",index=False); conflicts.to_csv(TABLES_DIR/"07_condition_status_conflicts.csv",index=False)
    create_status_plot(overall); create_sjogren_history_plot(sj_desc)
    pd.DataFrame(_UNRECOGNIZED).drop_duplicates().to_csv(QC_DIR/"07_unrecognized_condition_values.csv",index=False)
    qc={"input_path":str(args.input),"script_version":SCRIPT_VERSION,"run_timestamp":datetime.now(timezone.utc).isoformat(),"n_baseline_patients":len(base),"pop_denominators":{p:int(base["baseline_pop"].eq(p).sum()) for p in ("Pop1","Pop2","Pop3")},"past_medical_history_used_for_prevalence_or_models":False,"sjogren_history_used_for_prevalence_or_models":False,"comorbidity_incidence_rates_generated":False,"confirmed_present_only_prevalence":True,"no_cummax_for_historical_fields":True,"no_history_confirm_or_combination":True,"depression_in_rheumatological_conditions":False,"symptom_anxiety_in_rheumatological_conditions":False,"no_upstream_recalculation_of_pop_pro_essdai_overlap":True,"n_unrecognized_values":len(_UNRECOGNIZED),"n_status_conflict_rows":len(conflicts),"warnings":["Past-medical-history and Sjögren-history outputs are frequencies of documentation, not prevalence estimates.","Progression models, if enabled downstream, use only confirmed/present baseline rheumatological conditions and remain non-causal associations."]}
    (QC_DIR/"07_comorbidities_qc.json").write_text(json.dumps(qc,indent=2,default=str)+"\n")
    logger.info("Validated source mapping, condition states, and Pop denominators before writing Section 5 outputs")
    print("Generated Section 5 status and historical documentation outputs; no comorbidity incidence outputs were created.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

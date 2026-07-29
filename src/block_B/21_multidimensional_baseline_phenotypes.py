#!/usr/bin/env python3
"""Integrate and describe previously-derived baseline phenotypes by Pop."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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
ALL_POPS = ["Overall", *POPS, "Unclassifiable"]
DOMAINS = [f"eg_{x}_active" for x in ["constitutional", "lymphadenopathy", "articular", "cutaneous", "pulmonary", "renal", "muscular", "pns", "cns", "hematologic", "biological"]]
COMORBIDITIES = ["fibromyalgia", "osteoporosis", "ild", "thyroid_disease", "depression", "anxiety", "raynaud", "peripheral_neuropathy", "renal_tubular_acidosis", "myositis", "cryoglobulinemia", "chronic_bronchitis"]
SEROLOGY = ["anti_ro_pos", "anti_la_pos", "double_ro_la_pos", "ana_pos", "rf_pos", "cryo_pos", "low_c4", "leukopenia"]
SERO_DENOM = {"anti_ro_pos":"anti_ro_interpretable", "anti_la_pos":"anti_la_interpretable", "double_ro_la_pos":"anti_ro_interpretable", "ana_pos":"ana_interpretable", "rf_pos":"rf_interpretable", "cryo_pos":"cryo_interpretable", "low_c4":"c4_interpretable", "leukopenia":"wbc_interpretable"}
BLOCKS = {
    "Demographics and design": ["age_baseline", "sex", "race", "ethnicity", "protocol_group", "time_since_diagnosis_years"],
    "Disease activity and symptoms": ["essdai_total", "esspri_total", "esspri_dryness", "esspri_fatigue", "esspri_pain"],
    "Glandular and extraglandular profile": ["overlap_status", "n_extraglandular_domains_active", "glandular_active", "extraglandular_active", *DOMAINS],
    "Patient-reported and functional outcomes": ["sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global"],
    "Serology": SEROLOGY,
    "Comorbidities": [*COMORBIDITIES, "n_prespecified_comorbidities", "any_comorbidity", "two_or_more_comorbidities"],
    "Observation completeness": ["pop_core_complete", "pro_core_complete", "systemic_overlap_core_complete", "full_multidomain_complete", "extended_multidomain_complete"],
}
CONTINUOUS = {"age_baseline", "time_since_diagnosis_years", "essdai_total", "esspri_total", "esspri_dryness", "esspri_fatigue", "esspri_pain", "n_extraglandular_domains_active", "sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global", "n_prespecified_comorbidities"}
CATEGORICAL = {"sex", "race", "ethnicity", "protocol_group", "overlap_status"}
BOOLS = set(SEROLOGY + COMORBIDITIES + ["any_comorbidity", "two_or_more_comorbidities", "glandular_active", "extraglandular_active", *DOMAINS, "pop_core_complete", "pro_core_complete", "systemic_overlap_core_complete", "full_multidomain_complete", "extended_multidomain_complete", *SERO_DENOM.values()])
MODEL_COLUMNS = ["clinical_block", "outcome", "outcome_type", "contrast", "reference_pop", "estimate", "standard_error", "ci95_low", "ci95_high", "p_value", "effect_measure", "n_complete", "n_pop1", "n_pop2", "n_pop3", "adjustment_variables", "model_status", "interpretation"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-input", type=Path, default=common.INTERMEDIATE_DATA_DIR / "11_integrated_baseline_patient_level.parquet")
    p.add_argument("--observation-input", type=Path, default=common.INTERMEDIATE_DATA_DIR / "20_observation_process_patient_visit.parquet")
    p.add_argument("--comorbidity-input", type=Path, default=common.INTERMEDIATE_DATA_DIR / "07_comorbidities_baseline_patient.parquet")
    p.add_argument("--serology-input", type=Path, default=common.BLOCKA_QC_DIR / "01_serology_patient_level.csv")
    group = p.add_mutually_exclusive_group(); group.add_argument("--overwrite", action="store_true", default=True); group.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    p.add_argument("--skip-models", action="store_true")
    return p.parse_args(argv)


def validate_unique_patients(frame, source):
    if "patient_id" not in frame or frame.patient_id.isna().any():
        raise ValueError(f"{source}: missing patient_id")
    if frame.patient_id.duplicated().any():
        raise ValueError(f"{source}: duplicate patient_id values")


def load_inputs(paths):
    required = {"baseline":"script 11", "observation":"script 20", "comorbidity":"script 07", "serology":"script 01 serology"}
    frames = {}
    for source, path in paths.items():
        if not path.exists(): raise FileNotFoundError(f"Missing {source} input: {path}. Run {required[source]} first.")
        frames[source] = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    obs = frames["observation"]
    if "is_observed_baseline" not in obs: raise ValueError("observation: missing is_observed_baseline")
    frames["observation"] = obs.loc[obs.is_observed_baseline.fillna(False).astype(bool)].copy()
    return frames


def merge_baseline_blocks(primary, observation, comorbidity, serology):
    frames = {"baseline":primary.copy(), "observation":observation.copy(), "comorbidity":comorbidity.copy(), "serology":serology.copy()}
    for name, frame in frames.items():
        frame["patient_id"] = frame["patient_id"].astype("string") if "patient_id" in frame else pd.Series(dtype="string")
        validate_unique_patients(frame, name)
    if "pop_status" not in primary: raise ValueError("baseline: missing pop_status")
    if not primary.pop_status.isin(POPS).any(): raise ValueError("baseline: no Pop1, Pop2, or Pop3 patients")
    result = frames["baseline"].copy(); summaries = []
    primary_ids = set(result.patient_id)
    for name, secondary in frames.items():
        ids = set(secondary.patient_id)
        summaries.append({"source":name, "n_rows":len(secondary), "n_unique_patients":secondary.patient_id.nunique(), "n_matched_to_primary":len(primary_ids & ids), "n_unmatched_primary":len(primary_ids - ids), "n_secondary_only":len(ids - primary_ids)})
        if name == "baseline": continue
        keep = ["patient_id"] + [c for c in secondary if c != "patient_id" and c not in result]
        result = result.merge(secondary[keep], on="patient_id", how="left", validate="one_to_one")
    if len(result) != len(primary): raise ValueError("Primary patient count changed after merging")
    result["protocol_group"] = result.get("protocol", pd.Series(pd.NA, index=result.index)).astype("string").fillna("Unknown")
    for c in BOOLS & set(result.columns): result[c] = result[c].astype("boolean")
    for c in CONTINUOUS & set(result.columns): result[c] = pd.to_numeric(result[c], errors="coerce")
    used = ["patient_id", "pop_status", "protocol"] + [v for values in BLOCKS.values() for v in values] + list(SERO_DENOM.values())
    return result[[c for c in dict.fromkeys(used) if c in result]], pd.DataFrame(summaries)


def _mask(frame, variable):
    if variable in SERO_DENOM:
        flag = SERO_DENOM[variable]
        return frame[flag].eq(True) if flag in frame else pd.Series(False, index=frame.index)
    return frame[variable].notna()


def _denom_text(variable):
    if variable in SERO_DENOM: return f"Patients with {SERO_DENOM[variable]} == True"
    if variable in COMORBIDITIES or variable in {"any_comorbidity", "two_or_more_comorbidities", "n_prespecified_comorbidities"}: return "Patients with upstream script 07 coded value"
    return "Patients with non-missing baseline value"


def build_profile_table(data):
    rows = []
    for block, variables in BLOCKS.items():
        for variable in variables:
            if variable not in data: continue
            for pop in POPS:
                sub = data.loc[data.pop_status.eq(pop)]; available = _mask(sub, variable); values = sub.loc[available, variable]; base = {"clinical_block":block, "variable":variable, "variable_label":variable.replace("_", " ").title(), "pop_status":pop, "n_total_in_pop":len(sub), "n_available":int(available.sum()), "n_missing":int(len(sub)-available.sum()), "pct_available":100*available.mean() if len(sub) else np.nan, "denominator_definition":_denom_text(variable)}
                if variable in CATEGORICAL:
                    categories = (["neither", "glandular_only", "extraglandular_only", "overlap", "insufficient_info"] if variable == "overlap_status" else sorted(values.dropna().astype(str).unique()))
                    for category in categories:
                        n = int(values.astype(str).eq(category).sum()); rows.append({**base, "variable":f"{variable}={category}", "variable_label":f"{base['variable_label']}: {category}", "variable_type":"categorical", "summary":f"{n} ({100*n/len(values):.1f}%)" if len(values) else "NA", "median":np.nan, "q1":np.nan, "q3":np.nan, "n_positive":n, "pct_positive":100*n/len(values) if len(values) else np.nan})
                elif variable in CONTINUOUS:
                    x=pd.to_numeric(values, errors="coerce").dropna(); med=x.median() if len(x) else np.nan; q1=x.quantile(.25) if len(x) else np.nan; q3=x.quantile(.75) if len(x) else np.nan
                    rows.append({**base, "variable_type":"continuous", "summary":f"{med:.2f} ({q1:.2f}–{q3:.2f})" if len(x) else "NA", "median":med, "q1":q1, "q3":q3, "n_positive":np.nan, "pct_positive":np.nan})
                else:
                    n=int(values.eq(True).sum()); pct=100*n/len(values) if len(values) else np.nan
                    rows.append({**base, "variable_type":"binary", "summary":f"{n} / {len(values)} ({pct:.1f}%)" if len(values) else "NA", "median":np.nan, "q1":np.nan, "q3":np.nan, "n_positive":n, "pct_positive":pct})
    return pd.DataFrame(rows)


def build_denominator_table(data):
    rows=[]
    for block, variables in BLOCKS.items():
        for variable in variables:
            if variable not in data: continue
            for pop in ALL_POPS:
                sub=data if pop == "Overall" else data.loc[data.pop_status.eq(pop)]; available=_mask(sub, variable)
                rows.append({"clinical_block":block,"variable":variable,"pop_status":pop,"n_total":len(sub),"n_available":int(available.sum()),"n_missing":int(len(sub)-available.sum()),"pct_available":100*available.mean() if len(sub) else np.nan,"denominator_definition":_denom_text(variable)})
    return pd.DataFrame(rows)


def fit_adjusted_models(data, skip_models=False):
    continuous=["sf36_pcs","sf36_mcs","profad_total","mdafs_global","n_extraglandular_domains_active","n_prespecified_comorbidities"]
    binary=["anti_ro_pos","low_c4","fibromyalgia","depression","any_comorbidity"]
    rows=[]; interpretation="Adjusted association across baseline Pop groups; not a causal effect."; adjust="age_baseline, protocol_group, time_since_diagnosis_years"
    for outcome, kind in [(x,"continuous") for x in continuous]+[(x,"binary") for x in binary]:
        required = {outcome, "pop_status", "age_baseline", "protocol_group", "time_since_diagnosis_years"}
        status="skipped_by_user" if skip_models else "skipped_missing_variable" if not required.issubset(data.columns) else None
        complete=pd.DataFrame()
        if status is None:
            needed=[outcome,"pop_status","age_baseline","protocol_group","time_since_diagnosis_years"]
            complete=data.loc[data.pop_status.isin(POPS), needed].copy()
            if outcome in SERO_DENOM and SERO_DENOM[outcome] in data: complete=complete.loc[data.loc[complete.index,SERO_DENOM[outcome]].eq(True)]
            complete=complete.dropna(); counts=complete.pop_status.value_counts()
            y=pd.to_numeric(complete[outcome],errors="coerce"); complete=complete.loc[y.notna()].copy(); complete[outcome]=y.dropna()
            if len(complete)<30 or any(counts.get(p,0)<10 for p in POPS) or (kind=="binary" and (complete[outcome].sum()<5 or (1-complete[outcome]).sum()<5)): status="skipped_insufficient_data"
            elif complete[outcome].nunique()<2: status="skipped_no_variation"
        fitted=None
        if status is None:
            try:
                import statsmodels.formula.api as smf
                import statsmodels.api as sm
                formula=f'{outcome} ~ C(pop_status, Treatment(reference="Pop3")) + age_baseline + C(protocol_group) + time_since_diagnosis_years'
                fitted=smf.ols(formula,complete).fit(cov_type="HC3") if kind=="continuous" else smf.glm(formula,complete,family=sm.families.Binomial()).fit()
                if kind=="binary" and not getattr(fitted,"converged",True): raise RuntimeError("not converged")
                status="fitted"
            except Exception: status="failed"
        counts=complete.pop_status.value_counts() if len(complete) else {}
        for pop in ["Pop1","Pop2"]:
            estimate=se=lo=hi=p=np.nan
            if fitted is not None:
                term=f'C(pop_status, Treatment(reference="Pop3"))[T.{pop}]'; raw=fitted.params.get(term,np.nan); se_raw=fitted.bse.get(term,np.nan); ci=fitted.conf_int().loc[term]
                if kind=="binary": estimate,se,lo,hi=np.exp(raw),se_raw,np.exp(ci.iloc[0]),np.exp(ci.iloc[1])
                else: estimate,se,lo,hi=raw,se_raw,ci.iloc[0],ci.iloc[1]
                p=fitted.pvalues.get(term,np.nan)
            rows.append({"clinical_block":"Serology" if outcome in SEROLOGY else "Comorbidities" if outcome in COMORBIDITIES or outcome.startswith("n_prespecified") or outcome=="any_comorbidity" else "Glandular and extraglandular profile" if outcome.startswith("n_extraglandular") else "Patient-reported and functional outcomes", "outcome":outcome,"outcome_type":kind,"contrast":f"{pop} vs Pop3","reference_pop":"Pop3","estimate":estimate,"standard_error":se,"ci95_low":lo,"ci95_high":hi,"p_value":p,"effect_measure":"adjusted odds ratio" if kind=="binary" else "adjusted mean difference","n_complete":len(complete),"n_pop1":int(counts.get("Pop1",0)),"n_pop2":int(counts.get("Pop2",0)),"n_pop3":int(counts.get("Pop3",0)),"adjustment_variables":adjust,"model_status":status,"interpretation":interpretation})
    return pd.DataFrame(rows,columns=MODEL_COLUMNS)


def make_heatmap(data, path):
    specs=[("Age at baseline","age_baseline",False),("Time since diagnosis","time_since_diagnosis_years",False),("ESSDAI","essdai_total",False),("ESSPRI","esspri_total",False),("SF-36 PCS","sf36_pcs",False),("SF-36 MCS","sf36_mcs",False),("PROFAD","profad_total",False),("MDAFS","mdafs_global",False),("Extraglandular domain count","n_extraglandular_domains_active",False),("Anti-Ro positive","anti_ro_pos",True),("Low C4","low_c4",True),("Fibromyalgia","fibromyalgia",True),("Depression","depression",True),("Any comorbidity","any_comorbidity",True),("Full multidomain complete","full_multidomain_complete",True)]
    values=[]; annotations=[]; labels=[]
    for label,var,binary in specs:
        if var not in data: continue
        row=[]; ann=[]
        for pop in POPS:
            sub=data.loc[data.pop_status.eq(pop)]; mask=_mask(sub,var); x=sub.loc[mask,var]
            value=100*x.eq(True).mean() if binary and len(x) else pd.to_numeric(x,errors="coerce").median() if len(x) else np.nan
            row.append(value); ann.append((f"{value:.1f}%\n(n={len(x)})" if binary else f"{value:.1f}\n(n={len(x)})") if pd.notna(value) else "NA")
        values.append(row); annotations.append(ann); labels.append(label)
    raw=np.asarray(values,float); z=np.full_like(raw,np.nan)
    for i,row in enumerate(raw):
        if np.isfinite(row).any():
            sd=np.nanstd(row); z[i]=(row-np.nanmean(row))/sd if sd else np.where(np.isfinite(row),0,np.nan)
    cmap=plt.get_cmap("coolwarm").copy(); cmap.set_bad("#bdbdbd")
    fig,ax=plt.subplots(figsize=(7, max(6,.45*len(labels)))); im=ax.imshow(np.ma.masked_invalid(z),aspect="auto",cmap=cmap,vmin=-1.5,vmax=1.5)
    ax.set_xticks(range(3),POPS); ax.set_yticks(range(len(labels)),labels)
    for i in range(len(labels)):
        for j in range(3): ax.text(j,i,annotations[i][j],ha="center",va="center",fontsize=7)
    fig.colorbar(im,ax=ax,label="Within-variable standardized value"); fig.text(.02,.01,"Colors are standardized within each variable and show relative differences across Pop groups.\nColor intensity must not be compared between variables.",fontsize=8); fig.tight_layout(rect=(0,.06,1,1)); fig.savefig(path); plt.close(fig)


def main(argv=None):
    args=parse_args(argv); paths={"baseline":args.baseline_input,"observation":args.observation_input,"comorbidity":args.comorbidity_input,"serology":args.serology_input}
    outputs=[common.INTERMEDIATE_DATA_DIR/"21_multidimensional_baseline_patient.parquet",common.INTERMEDIATE_DATA_DIR/"21_multidimensional_baseline_patient.csv",common.BLOCKB_TABLES_DIR/"21_multidimensional_profile_by_pop.csv",common.BLOCKB_TABLES_DIR/"21_variable_denominators.csv",common.BLOCKB_TABLES_DIR/"21_adjusted_pop_comparisons.csv",common.BLOCKB_FIGURES_DIR/"21_multidimensional_baseline_heatmap.pdf",common.BLOCKB_QC_DIR/"21_multidimensional_baseline_qc.json",common.BLOCKB_QC_DIR/"21_merge_summary.csv"]
    if not args.overwrite and any(p.exists() for p in outputs): raise FileExistsError("Output exists; use --overwrite")
    common.ensure_output_dirs(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.FileHandler(common.BLOCKB_LOGS_DIR/"21_multidimensional_baseline_phenotypes.log",mode="w"),logging.StreamHandler()])
    frames=load_inputs(paths); data,merge=merge_baseline_blocks(frames["baseline"],frames["observation"],frames["comorbidity"],frames["serology"]); profile=build_profile_table(data); denoms=build_denominator_table(data); models=fit_adjusted_models(data,args.skip_models)
    data.to_parquet(outputs[0],index=False); data.to_csv(outputs[1],index=False); profile.to_csv(outputs[2],index=False); denoms.to_csv(outputs[3],index=False); models.to_csv(outputs[4],index=False); make_heatmap(data,outputs[5]); merge.to_csv(outputs[7],index=False)
    qc={"n_primary_baseline_patients":len(frames["baseline"]),"n_integrated_patients":len(data),**{f"n_{p.lower()}":int(data.pop_status.eq(p).sum()) for p in POPS},"n_unclassifiable":int(data.pop_status.eq("Unclassifiable").sum()),"n_missing_age":int(data.get("age_baseline",pd.Series(index=data.index,dtype=float)).isna().sum()),"n_missing_protocol":int(data.protocol_group.eq("Unknown").sum()),"n_missing_disease_duration":int(data.get("time_since_diagnosis_years",pd.Series(index=data.index,dtype=float)).isna().sum()),"n_models_requested":11,"n_models_fitted":int(models.loc[models.model_status.eq("fitted"),"outcome"].nunique()),"n_models_skipped":int(models.loc[~models.model_status.eq("fitted"),"outcome"].nunique()),"input_paths":{k:str(v) for k,v in paths.items()},"warnings":[]}
    outputs[6].write_text(json.dumps(qc,indent=2)+"\n"); logging.info("Integrated %d baseline patients; models fitted=%d skipped=%d",len(data),qc["n_models_fitted"],qc["n_models_skipped"])

if __name__ == "__main__": main()

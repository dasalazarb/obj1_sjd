#!/usr/bin/env python3
"""Describe the observed-baseline integrated clinical profile.

This script consumes only the canonical patient-visit integrated dataset produced
by script 10.  It audits that dataset, selects its official observed baseline,
and summarizes previously-derived Pop, overlap, and PRO variables; no clinical
score or classification is recalculated here.
"""
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
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402

KEY_COLUMNS = ["patient_id", "visit_id"]
REQUIRED_COLUMNS = ["patient_id", "visit_id", "visit_date", "visit_number", "is_observed_baseline", "pop_status", "overlap_status"]
OPTIONAL_OUTCOME_COLUMNS = ["sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global"]
POP_ORDER = ["Pop1", "Pop2", "Pop3", "Unclassifiable"]
POP_CLASSIFIABLE = POP_ORDER[:3]
OVERLAP_ORDER = ["neither", "glandular_only", "extraglandular_only", "overlap", "insufficient_info"]
OVERLAP_CLASSIFIABLE = OVERLAP_ORDER[:4]
POP_COLORS = {"Pop1": "#E66101", "Pop2": "#5E5AA8", "Pop3": "#1B9E77", "Unclassifiable": "#9E9E9E"}
OVERLAP_COLORS = {"neither": "#9E9E9E", "glandular_only": "#4C78A8", "extraglandular_only": "#59A14F", "overlap": "#E15759", "insufficient_info": "#D9D2CE"}
SF36_DOMAINS = ["sf36_physical_functioning", "sf36_role_physical", "sf36_bodily_pain", "sf36_general_health", "sf36_vitality", "sf36_social_functioning", "sf36_role_emotional", "sf36_mental_health"]
DOMAIN_COLUMNS = ["eg_constitutional_active", "eg_lymphadenopathy_active", "eg_articular_active", "eg_cutaneous_active", "eg_pulmonary_active", "eg_renal_active", "eg_muscular_active", "eg_pns_active", "eg_cns_active", "eg_hematologic_active", "eg_biological_active"]
BASELINE_COLUMNS = ["patient_id", "visit_id", "visit_date", "visit_number", "is_observed_baseline", "observed_baseline_date", "protocol", "interval_name", "n_visits_patient", "essdai_total", "esspri_dryness", "esspri_fatigue", "esspri_pain", "esspri_total", "esspri_total_observed", "esspri_scoring_valid", "pop_status", "pop_status_display", "baseline_pop_status", "baseline_pop_status_display", "pop_missingness_label", "glandular_active", "glandular_evaluable", "extraglandular_active", "extraglandular_evaluable", "overlap_active", "overlap_status", "n_extraglandular_domains_active", *DOMAIN_COLUMNS, *SF36_DOMAINS, "sf36_pcs", "sf36_mcs", "sf36_scoring_valid", "profad_total", "profad_scoring_valid", "mdafs_global", "mdafs_scoring_valid", "has_pop_data", "has_overlap_data", "has_pro_data", "has_essdai", "has_esspri", "has_sf36_pcs", "has_sf36_mcs", "has_profad", "has_mdafs", "n_data_blocks_available", "dx_date", "dx_date_precision", "time_since_diagnosis_days", "time_since_diagnosis_years", "integration_version", "integration_run_date"]

# Explicit metadata also drives the downloadable dictionary table.
VARIABLE_SCHEMA = {
    "patient_id": ("Visit spine", "Canonical patient identifier", "string", "non-missing", True, True, "00_build_visit_spine.py"),
    "visit_id": ("Visit spine", "Unique visit-spine identifier", "string", "unique", True, True, "00_build_visit_spine.py"),
    "visit_date": ("Visit spine", "Clinical visit date", "datetime64[ns]", "date or missing", True, True, "00_build_visit_spine.py"),
    "visit_number": ("Visit spine", "Within-patient visit sequence", "Int64", ">=0; baseline=0", True, True, "00_build_visit_spine.py"),
    "is_observed_baseline": ("Visit spine", "Official observed-baseline indicator", "boolean", "True/False/missing", True, True, "10_build_integrated_longitudinal_dataset.py"),
    "pop_status": ("Pop", "Previously derived ESSDAI/ESSPRI phenotype", "string", "Pop1/Pop2/Pop3/Unclassifiable", True, True, "01_pop_distribution.py"),
    "overlap_status": ("Overlap", "Previously derived glandular/extraglandular status", "string", "/".join(OVERLAP_ORDER), True, True, "06_overlap_glandular_followup.py"),
    "essdai_total": ("Pop", "Previously derived ESSDAI total", "float64", "0-123", False, True, "01_pop_distribution.py"),
    "esspri_total": ("Pop", "Previously derived ESSPRI total", "float64", "0-10", False, True, "01_pop_distribution.py"),
    "n_extraglandular_domains_active": ("Overlap", "Official active extraglandular-domain count", "Int64", "0-11 integer", False, True, "06_overlap_glandular_followup.py"),
    "sf36_pcs": ("PROs", "Previously scored SF-36 physical component", "float64", "operational plausibility 0-100", False, True, "09_pros_longitudinal.py"),
    "sf36_mcs": ("PROs", "Previously scored SF-36 mental component", "float64", "operational plausibility 0-100", False, True, "09_pros_longitudinal.py"),
    "profad_total": ("PROs", "Previously scored PROFAD total", "float64", "upstream-defined", False, True, "09_pros_longitudinal.py"),
    "mdafs_global": ("PROs", "Previously scored MDAFS global", "float64", "upstream-defined", False, True, "09_pros_longitudinal.py"),
}


def setup_logging() -> logging.Logger:
    log_dir = common.OUTPUTS_DIR / "logs" / "11_integrated_baseline_profile"; log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_dir / "11_integrated_baseline_profile.log", mode="w"), logging.StreamHandler()])
    return logging.getLogger(__name__)


def normalize_status(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    value = series.astype("string").str.strip()
    return value.map(lambda x: mapping.get(str(x).lower(), x) if pd.notna(x) else pd.NA).astype("string")


def normalize_integrated_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    strings = ["patient_id", "visit_id", "protocol", "interval_name", "pop_status", "pop_status_display", "baseline_pop_status", "baseline_pop_status_display", "pop_missingness_label", "overlap_status", "dx_date_precision", "integration_version", "integration_run_date"]
    dates = ["visit_date", "observed_baseline_date", "dx_date", "previous_visit_date", "next_visit_date"]
    booleans = ["is_observed_baseline", "is_last_observed_visit", "glandular_active", "glandular_evaluable", "extraglandular_active", "extraglandular_evaluable", "overlap_active", "sf36_scoring_valid", "profad_scoring_valid", "mdafs_scoring_valid", "esspri_scoring_valid", "has_pop_data", "has_overlap_data", "has_pro_data", "has_essdai", "has_esspri", "has_sf36_pcs", "has_sf36_mcs", "has_profad", "has_mdafs", *DOMAIN_COLUMNS]
    continuous = ["essdai_total", "esspri_dryness", "esspri_fatigue", "esspri_pain", "esspri_total", "esspri_total_observed", *SF36_DOMAINS, "sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global", "time_since_observed_baseline_days", "time_since_observed_baseline_years", "time_since_diagnosis_days", "time_since_diagnosis_years"]
    for c in strings:
        if c in data: data[c] = data[c].astype("string")
    for c in dates:
        if c in data: data[c] = pd.to_datetime(data[c], errors="coerce")
    for c in booleans:
        if c in data: data[c] = data[c].astype("boolean")
    for c in continuous:
        if c in data: data[c] = pd.to_numeric(data[c], errors="coerce")
    for c in ["visit_number", "n_visits_patient", "n_data_blocks_available", "n_extraglandular_domains_active"]:
        if c in data:
            numeric = pd.to_numeric(data[c], errors="coerce")
            if c == "n_extraglandular_domains_active" and numeric.dropna().mod(1).ne(0).any():
                # Retain numeric values for a transparent range audit; do not silently round.
                data[c] = numeric
            else: data[c] = numeric.astype("Int64")
    if "pop_status" in data: data["pop_status"] = normalize_status(data.pop_status, {"pop1":"Pop1", "pop2":"Pop2", "pop3":"Pop3", "unclassified":"Unclassifiable", "unclassifiable":"Unclassifiable"})
    if "overlap_status" in data: data["overlap_status"] = normalize_status(data.overlap_status, {"unclassifiable":"insufficient_info", "unknown":"insufficient_info", "insufficient information":"insufficient_info", "insufficient_info":"insufficient_info"})
    return data


def require_and_validate_keys(data: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in data]
    if missing: raise ValueError(f"Integrated dataset is missing required columns: {missing}")
    duplicated = data.loc[data.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS]
    if not duplicated.empty: raise ValueError(f"Duplicate patient-visit keys: {duplicated.drop_duplicates().head(10).to_dict('records')}")
    if not data.visit_id.is_unique: raise ValueError("visit_id is not globally unique; examples: " + str(data.loc[data.visit_id.duplicated(keep=False), KEY_COLUMNS].head(10).to_dict("records")))


def range_violations(data: pd.DataFrame) -> pd.DataFrame:
    ranges = {"essdai_total": (0, 123), "esspri_dryness": (0, 10), "esspri_fatigue": (0, 10), "esspri_pain": (0, 10), "esspri_total": (0, 10), "esspri_total_observed": (0, 10), **{c:(0,100) for c in SF36_DOMAINS}, "n_data_blocks_available": (0,3)}
    records=[]
    for col, (low, high) in ranges.items():
        if col not in data: continue
        values=pd.to_numeric(data[col], errors="coerce"); mask=values.notna() & ~values.between(low, high)
        for _, row in data.loc[mask, KEY_COLUMNS].assign(observed_value=values[mask]).iterrows(): records.append({**row[KEY_COLUMNS].to_dict(), "variable":col,"observed_value":row.observed_value,"expected_min":low,"expected_max":high,"violation_type":"outside_range"})
    c="n_extraglandular_domains_active"
    if c in data:
        values=pd.to_numeric(data[c], errors="coerce"); mask=values.notna() & ((values<0) | values.mod(1).ne(0))
        for _, row in data.loc[mask, KEY_COLUMNS].assign(observed_value=values[mask]).iterrows(): records.append({**row[KEY_COLUMNS].to_dict(), "variable":c,"observed_value":row.observed_value,"expected_min":0,"expected_max":11,"violation_type":"negative_or_noninteger"})
    return pd.DataFrame(records, columns=[*KEY_COLUMNS,"variable","observed_value","expected_min","expected_max","violation_type"])


def select_baseline(data: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, pd.Series]:
    discordance = data.is_observed_baseline.fillna(False).ne(data.visit_number.eq(0))
    if discordance.any(): logger.warning("STRONG WARNING: %d baseline indicator/visit-number discordances; official indicator retained.", discordance.sum())
    baseline=data.loc[data.is_observed_baseline.eq(True)].copy()
    counts=baseline.groupby("patient_id").size()
    if counts.gt(1).any(): raise ValueError("Multiple observed-baseline rows per patient: " + str(baseline.loc[baseline.patient_id.isin(counts[counts.gt(1)].index), ["patient_id","visit_id","visit_date"]].head(10).to_dict("records")))
    missing_patients=set(data.patient_id.dropna())-set(baseline.patient_id.dropna())
    if missing_patients: raise ValueError(f"No observed baseline for {len(missing_patients)} patients; examples: {sorted(missing_patients)[:10]}")
    return baseline, discordance


def valid_mask(frame: pd.DataFrame, outcome: str, validity: str, warnings: list[str]) -> pd.Series:
    if outcome not in frame: return pd.Series(False, index=frame.index)
    if validity not in frame:
        warnings.append(f"{validity} absent; {outcome} analyses use non-missing values without upstream validity confirmation.")
        return frame[outcome].notna()
    return frame[outcome].notna() & frame[validity].eq(True)


def summarize(series: pd.Series) -> dict[str, float | int]:
    """Summarize numeric values, coercing nullable booleans to 0/1 safely.

    Pandas considers BooleanDtype numeric in some versions, but its quantile
    implementation cannot interpolate booleans.  A float conversion provides
    stable descriptive output for availability/indicator columns as well.
    """
    x = pd.to_numeric(series, errors="coerce").astype("float64").dropna()
    if x.empty:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan,
                "q1": np.nan, "q3": np.nan, "minimum": np.nan, "maximum": np.nan}
    return {"n": len(x), "mean": x.mean(), "sd": x.std(ddof=1),
            "median": x.median(), "q1": x.quantile(.25), "q3": x.quantile(.75),
            "minimum": x.min(), "maximum": x.max()}


def variable_availability(baseline: pd.DataFrame, valid: dict[str,pd.Series]) -> pd.DataFrame:
    rows=[]
    for col in baseline.columns:
        meta=VARIABLE_SCHEMA.get(col, ("Integrated", "Integrated longitudinal variable", "not specified", "not specified", False, False, "10_build_integrated_longitudinal_dataset.py"))
        s=baseline[col]; numeric=pd.to_numeric(s, errors="coerce") if pd.api.types.is_numeric_dtype(s) else pd.Series(dtype=float)
        mask=valid.get(col, s.notna())
        d={"variable":col,"clinical_block":meta[0],"dtype_expected":meta[2],"dtype_observed":str(s.dtype),"n_baseline_total":len(baseline),"n_nonmissing":int(s.notna().sum()),"pct_nonmissing":100*s.notna().mean(),"n_scoring_valid":int(mask.sum()) if col in valid else np.nan,"pct_scoring_valid":100*mask.mean() if col in valid else np.nan,"n_available_for_analysis":int(mask.sum()),"pct_available_for_analysis":100*mask.mean()}
        d.update({k:(v if len(numeric) else np.nan) for k,v in summarize(numeric).items() if k != "n"}); rows.append(d)
    return pd.DataFrame(rows)


def grouped_summary(baseline: pd.DataFrame, outcome: str, mask: pd.Series, group: str, groups: list[str], overall=True) -> pd.DataFrame:
    rows=[]
    specifications=([("Overall", baseline)] if overall else [])+[(x,baseline.loc[baseline[group].eq(x)]) for x in groups]
    for label, frame in specifications: rows.append({"outcome":outcome,"grouping":group,"group":label,**summarize(frame.loc[mask.reindex(frame.index, fill_value=False), outcome])})
    return pd.DataFrame(rows)


def overlap_by_pop(baseline: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    rows=[]
    for pop in POP_CLASSIFIABLE:
        subset=baseline.loc[baseline.pop_status.eq(pop)]; denom=int(subset.overlap_status.isin(OVERLAP_CLASSIFIABLE).sum())
        for ov in OVERLAP_CLASSIFIABLE:
            n=int(subset.overlap_status.eq(ov).sum()); rows.append({"pop_status":pop,"overlap_status":ov,"n":n,"denominator_within_pop":denom,"pct_within_pop":100*n/denom if denom else np.nan,"n_insufficient_info":int(subset.overlap_status.eq("insufficient_info").sum())})
    table=pd.DataFrame(rows); matrix=table.pivot(index="pop_status",columns="overlap_status",values="n").reindex(index=POP_CLASSIFIABLE, columns=OVERLAP_CLASSIFIABLE).fillna(0).astype(int)
    test=[]
    if matrix.shape[0] > 1 and matrix.to_numpy().sum() and (matrix.sum(axis=1)>0).all() and (matrix.sum(axis=0)>0).all():
        chi,p,dof,expected=stats.chi2_contingency(matrix); test.append({"test":"chi_square","statistic":chi,"p_value":p,"degrees_of_freedom":dof,"minimum_expected":expected.min()})
    return table,matrix.reset_index(),pd.DataFrame(test,columns=["test","statistic","p_value","degrees_of_freedom","minimum_expected"])


def correlations(baseline: pd.DataFrame, valid: dict[str,pd.Series]) -> pd.DataFrame:
    cols=[c for c in ["essdai_total","esspri_total","sf36_pcs","sf36_mcs","profad_total","mdafs_global","n_extraglandular_domains_active"] if c in baseline]
    rows=[]
    for i,x in enumerate(cols):
        for y in cols[i+1:]:
            mask=baseline[x].notna() & baseline[y].notna() & valid.get(x,True) & valid.get(y,True); pair=baseline.loc[mask,[x,y]]
            if len(pair)>=3: sp=stats.spearmanr(pair[x],pair[y]); pe=stats.pearsonr(pair[x],pair[y])
            else: sp=(np.nan,np.nan); pe=(np.nan,np.nan)
            rows.append({"variable_x":x,"variable_y":y,"n_complete_pairs":len(pair),"spearman_rho":sp.statistic if hasattr(sp,'statistic') else sp[0],"spearman_p_value":sp.pvalue if hasattr(sp,'pvalue') else sp[1],"pearson_r":pe.statistic if hasattr(pe,'statistic') else pe[0],"pearson_p_value":pe.pvalue if hasattr(pe,'pvalue') else pe[1]})
    result=pd.DataFrame(rows); p=result.spearman_p_value.to_numpy(dtype=float); order=np.argsort(np.nan_to_num(p,nan=np.inf)); adjusted=np.full(len(p),np.nan); m=np.isfinite(p).sum()
    if m:
        ranked=np.minimum.accumulate((p[order][:m]*m/np.arange(1,m+1))[::-1])[::-1]; adjusted[order[:m]]=np.minimum(ranked,1)
    result["spearman_p_value_bh"]=adjusted
    return result[["variable_x","variable_y","n_complete_pairs","spearman_rho","spearman_p_value","spearman_p_value_bh","pearson_r","pearson_p_value"]]


def missingness(baseline: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for variable in ["has_overlap_data","has_sf36_pcs","has_sf36_mcs","has_profad","has_mdafs","n_data_blocks_available"]:
        if variable not in baseline: continue
        for pop in POP_CLASSIFIABLE:
            s=baseline.loc[baseline.pop_status.eq(pop),variable]; available=s.notna() if variable=="n_data_blocks_available" else s.eq(True)
            rows.append({"variable":variable,"pop_status":pop,"n_total":len(s),"n_available":int(available.sum()),"pct_available":100*available.mean() if len(s) else np.nan,"n_missing":int((~available).sum()),"pct_missing":100*(~available).mean() if len(s) else np.nan})
    return pd.DataFrame(rows)


def make_figures(baseline: pd.DataFrame, overlap: pd.DataFrame, valid: dict[str,pd.Series], correlation: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True,exist_ok=True)
    matrix=overlap.pivot(index="pop_status",columns="overlap_status",values="n").reindex(index=POP_CLASSIFIABLE,columns=OVERLAP_CLASSIFIABLE).fillna(0); denom=matrix.sum(axis=1).replace(0,np.nan)
    ax=(matrix.div(denom,axis=0)*100).plot.bar(stacked=True,color=[OVERLAP_COLORS[x] for x in OVERLAP_CLASSIFIABLE],figsize=(7,4)); ax.set(ylabel="Percent within Pop (classifiable overlap)",xlabel="Pop status"); ax.legend(title="Overlap",bbox_to_anchor=(1.02,1),loc="upper left"); plt.tight_layout(); plt.savefig(fig_dir/"11_overlap_by_pop_stacked_bar.pdf"); plt.close()
    for outcome,name in [("sf36_pcs","11_pcs_by_pop.pdf"),("sf36_mcs","11_mcs_by_pop.pdf"),("profad_total","11_profad_by_pop.pdf"),("mdafs_global","11_mdafs_by_pop.pdf"),("n_extraglandular_domains_active","11_extraglandular_domain_count_by_pop.pdf")]:
        if outcome not in baseline: continue
        values=[baseline.loc[baseline.pop_status.eq(p)&valid.get(outcome,pd.Series(True,index=baseline.index)),outcome].dropna() for p in POP_CLASSIFIABLE]
        fig,ax=plt.subplots(figsize=(7,4)); bp=ax.boxplot(values,labels=POP_CLASSIFIABLE,patch_artist=True)
        for box,p in zip(bp["boxes"],POP_CLASSIFIABLE): box.set_facecolor(POP_COLORS[p])
        ax.set(xlabel="Pop status",ylabel=outcome); plt.tight_layout(); plt.savefig(fig_dir/name); plt.close()
    cols=sorted(set(correlation.variable_x).union(correlation.variable_y))
    if cols:
        matrix=pd.DataFrame(np.eye(len(cols)),index=cols,columns=cols)
        for r in correlation.itertuples(): matrix.loc[r.variable_x,r.variable_y]=matrix.loc[r.variable_y,r.variable_x]=r.spearman_rho
        fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(matrix, vmin=-1,vmax=1,cmap="coolwarm"); ax.set_xticks(range(len(cols)),cols,rotation=45,ha="right"); ax.set_yticks(range(len(cols)),cols); fig.colorbar(im,ax=ax,label="Spearman rho"); plt.tight_layout(); plt.savefig(fig_dir/"11_integrated_baseline_correlation_heatmap.pdf"); plt.close()


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--input",type=Path,default=common.INTEGRATED_LONGITUDINAL_PARQUET); args=parser.parse_args()
    logger=setup_logging(); common.ensure_output_dirs(); tables=common.BLOCKA_TABLES_DIR/"11_integrated_baseline_profile"; qc_dir=common.BLOCKA_QC_DIR/"11_integrated_baseline_profile"; fig_dir=common.OUTPUTS_DIR/"figures"/"blockA"/"11_integrated_baseline_profile"; [p.mkdir(parents=True,exist_ok=True) for p in (tables,qc_dir,fig_dir,common.INTERMEDIATE_DATA_DIR/"11_integrated_baseline_profile")]
    if not args.input.exists(): raise FileNotFoundError(f"Integrated input not found: {args.input}")
    raw=pd.read_parquet(args.input); original_dtypes=raw.dtypes.astype(str).to_dict(); data=normalize_integrated_dtypes(raw); require_and_validate_keys(data)
    unexpected_pop=sorted(set(data.pop_status.dropna())-set(POP_ORDER)); unexpected_overlap=sorted(set(data.overlap_status.dropna())-set(OVERLAP_ORDER))
    if unexpected_pop: raise ValueError(f"Unexpected normalized Pop levels: {unexpected_pop}")
    if unexpected_overlap: raise ValueError(f"Unexpected normalized overlap levels: {unexpected_overlap}")
    if "time_since_observed_baseline_days" in data and data.loc[data.is_observed_baseline.eq(True), "time_since_observed_baseline_days"].dropna().abs().gt(1e-6).any():
        logger.warning("STRONG WARNING: baseline time_since_observed_baseline_days is not approximately zero.")
    violations=range_violations(data)
    fatal=violations.loc[violations.variable.isin(["n_data_blocks_available","n_extraglandular_domains_active"])]
    if not fatal.empty: raise ValueError("Fatal count range violations: " + str(fatal.head(10).to_dict("records")))
    baseline,discordance=select_baseline(data,logger); warnings=[]
    if not violations.empty: warnings.append(f"{len(violations)} defined-range violations were recorded; no values were corrected.")
    for c in OPTIONAL_OUTCOME_COLUMNS:
        if c not in data: warnings.append(f"Optional outcome missing: {c}")
    for c in ["essdai_total","esspri_total","n_extraglandular_domains_active","sf36_scoring_valid","profad_scoring_valid","mdafs_scoring_valid"]:
        if c not in data: warnings.append(f"Recommended variable missing: {c}")
    valid={"sf36_pcs":valid_mask(baseline,"sf36_pcs","sf36_scoring_valid",warnings),"sf36_mcs":valid_mask(baseline,"sf36_mcs","sf36_scoring_valid",warnings),"profad_total":valid_mask(baseline,"profad_total","profad_scoring_valid",warnings),"mdafs_global":valid_mask(baseline,"mdafs_global","mdafs_scoring_valid",warnings)}
    # Official domain count is never replaced; this is only a transparent QC recount.
    present_domains=[c for c in DOMAIN_COLUMNS if c in baseline]
    if present_domains:
        domain_values=baseline[present_domains].astype("boolean"); baseline["n_extraglandular_domains_recount_qc"]=domain_values.sum(axis=1).where(domain_values.notna().any(axis=1)).astype("Int64")
        discrepancies=(baseline.n_extraglandular_domains_active.notna() & baseline.n_extraglandular_domains_recount_qc.notna() & baseline.n_extraglandular_domains_active.ne(baseline.n_extraglandular_domains_recount_qc)) if "n_extraglandular_domains_active" in baseline else pd.Series(False,index=baseline.index)
    else: discrepancies=pd.Series(False,index=baseline.index)
    indicator_discrepancies={}
    for indicator,value in [("has_pop_data","pop_status"),("has_overlap_data","overlap_status"),("has_essdai","essdai_total"),("has_esspri","esspri_total"),("has_sf36_pcs","sf36_pcs"),("has_sf36_mcs","sf36_mcs"),("has_profad","profad_total"),("has_mdafs","mdafs_global")]:
        if indicator in data and value in data: indicator_discrepancies[indicator]=int(data[indicator].ne(data[value].notna()).fillna(True).sum())
    baseline_out=baseline[[c for c in BASELINE_COLUMNS if c in baseline]+(["n_extraglandular_domains_recount_qc"] if "n_extraglandular_domains_recount_qc" in baseline else [])]
    baseline_out.to_parquet(common.INTERMEDIATE_DATA_DIR/"11_integrated_baseline_profile"/"11_integrated_baseline_patient_level.parquet",index=False); baseline_out.to_csv(common.INTERMEDIATE_DATA_DIR/"11_integrated_baseline_profile"/"11_integrated_baseline_patient_level.csv",index=False)
    dictionary=pd.DataFrame([{"variable":k,"clinical_block":v[0],"description":v[1],"expected_dtype":v[2],"allowed_range_or_levels":v[3],"required":v[4],"used_in_primary_analysis":v[5],"source_script":v[6],"recalculated_in_script_11":False} for k,v in VARIABLE_SCHEMA.items()]); dictionary.to_csv(tables/"11_integrated_variable_dictionary.csv",index=False)
    variable_availability(baseline,valid).to_csv(tables/"11_integrated_baseline_variable_availability.csv",index=False)
    over,matrix,test=overlap_by_pop(baseline); over.to_csv(tables/"11_overlap_by_pop.csv",index=False); matrix.to_csv(tables/"11_overlap_by_pop_matrix.csv",index=False); test.to_csv(tables/"11_overlap_by_pop_statistical_test.csv",index=False)
    pcs_mcs_parts=[grouped_summary(baseline,o,valid[o],"pop_status",POP_CLASSIFIABLE) for o in ["sf36_pcs","sf36_mcs"] if o in baseline]
    (pd.concat(pcs_mcs_parts, ignore_index=True) if pcs_mcs_parts else pd.DataFrame(columns=["outcome","grouping","group"])).to_csv(tables/"11_pcs_mcs_by_pop.csv",index=False)
    fatigue_pop_parts=[grouped_summary(baseline,o,valid[o],"pop_status",POP_CLASSIFIABLE) for o in ["profad_total","mdafs_global"] if o in baseline]
    (pd.concat(fatigue_pop_parts, ignore_index=True) if fatigue_pop_parts else pd.DataFrame(columns=["outcome","grouping","group"])).to_csv(tables/"11_fatigue_measures_by_pop.csv",index=False)
    fatigue_overlap_parts=[grouped_summary(baseline,o,valid[o],"overlap_status",OVERLAP_CLASSIFIABLE) for o in ["profad_total","mdafs_global"] if o in baseline]
    fatigue_overlap=pd.concat(fatigue_overlap_parts,ignore_index=True) if fatigue_overlap_parts else pd.DataFrame(columns=["outcome","grouping","group"]); fatigue_overlap.to_csv(tables/"11_fatigue_measures_by_overlap.csv",index=False)
    if "n_extraglandular_domains_active" in baseline:
        domain=pd.concat([grouped_summary(baseline,"n_extraglandular_domains_active",baseline.n_extraglandular_domains_active.notna(),"pop_status",POP_CLASSIFIABLE)],ignore_index=True)
        for label,frame in [("Overall", baseline)] + [(p, baseline.loc[baseline.pop_status.eq(p)]) for p in POP_CLASSIFIABLE]:
            x=frame.n_extraglandular_domains_active.dropna(); mask=domain.group.eq(label); n=len(x)
            domain.loc[mask,"n_zero"]=int(x.eq(0).sum()); domain.loc[mask,"pct_zero"]=100*x.eq(0).mean() if n else np.nan; domain.loc[mask,"n_one"]=int(x.eq(1).sum()); domain.loc[mask,"pct_one"]=100*x.eq(1).mean() if n else np.nan; domain.loc[mask,"n_two_or_more"]=int(x.ge(2).sum()); domain.loc[mask,"pct_two_or_more"]=100*x.ge(2).mean() if n else np.nan; domain.loc[mask,"n_three_or_more"]=int(x.ge(3).sum()); domain.loc[mask,"pct_three_or_more"]=100*x.ge(3).mean() if n else np.nan
        domain.to_csv(tables/"11_extraglandular_domain_count_by_pop.csv",index=False)
    else: pd.DataFrame().to_csv(tables/"11_extraglandular_domain_count_by_pop.csv",index=False)
    corr=correlations(baseline,valid); corr.to_csv(tables/"11_integrated_baseline_correlations.csv",index=False); missingness(baseline).to_csv(tables/"11_missingness_by_pop.csv",index=False)
    audit_source=data.loc[data.is_observed_baseline.eq(True) | discordance].copy()
    audit=audit_source[[c for c in ["patient_id","visit_id","visit_date","visit_number","is_observed_baseline","observed_baseline_date","pop_status","overlap_status","n_data_blocks_available"] if c in audit_source]].copy(); audit["baseline_date_matches_visit_date"]=audit.observed_baseline_date.eq(audit.visit_date) if "observed_baseline_date" in audit else pd.NA; audit["baseline_visit_number_matches"]=audit.visit_number.eq(0); audit.to_csv(qc_dir/"11_baseline_selection_audit.csv",index=False); violations.to_csv(qc_dir/"11_variable_range_violations.csv",index=False)
    claims=("This descriptive baseline profile uses the official is_observed_baseline indicator and previously-derived Pop, overlap, and PRO measures.\nPop is not independent of ESSDAI/ESSPRI, and extraglandular domains may contribute to ESSDAI.\nPROFAD and MDAFS are described numerically; score direction is not asserted.\n") ; (tables/"11_integrated_baseline_profile_claims.txt").write_text(claims)
    make_figures(baseline,over,valid,corr,fig_dir)
    qc={"input_path":str(args.input),"input_shape":list(data.shape),"integration_version":sorted(data.integration_version.dropna().unique().tolist()) if "integration_version" in data else [],"integration_run_date":sorted(data.integration_run_date.dropna().unique().tolist()) if "integration_run_date" in data else [],"n_input_rows":len(data),"n_input_patients":int(data.patient_id.nunique()),"n_unique_visit_ids":int(data.visit_id.nunique()),"n_duplicate_patient_visit_keys":0,"n_baseline_rows":len(baseline),"n_baseline_patients":int(baseline.patient_id.nunique()),"n_patients_with_multiple_baseline_rows":0,"n_baseline_visit_number_discordances":int(discordance.sum()),"n_pop_classifiable":int(baseline.pop_status.isin(POP_CLASSIFIABLE).sum()),"n_pop_unclassifiable":int(baseline.pop_status.eq("Unclassifiable").sum()),"n_overlap_classifiable":int(baseline.overlap_status.isin(OVERLAP_CLASSIFIABLE).sum()),"n_overlap_insufficient_info":int(baseline.overlap_status.eq("insufficient_info").sum()),"n_domain_count_discrepancies":int(discrepancies.sum()),"domain_columns_used_for_qc_recount":present_domains,"dtype_discrepancies":{c:{"expected":VARIABLE_SCHEMA[c][2],"observed":original_dtypes.get(c)} for c in VARIABLE_SCHEMA if c in original_dtypes and VARIABLE_SCHEMA[c][2] not in original_dtypes[c]},"range_violations":violations.variable.value_counts().to_dict(),"unexpected_pop_levels":unexpected_pop,"unexpected_overlap_levels":unexpected_overlap,"availability_indicator_discrepancies":indicator_discrepancies,"missing_requested_variables":[c for c in BASELINE_COLUMNS if c not in data],"small_cell_flags":over.loc[over.n.lt(5),["pop_status","overlap_status","n"]].to_dict("records"),"statistical_test_warnings":[],"general_warnings":warnings}
    for name,outcome,flag in [("pcs","sf36_pcs","sf36_scoring_valid"),("mcs","sf36_mcs","sf36_scoring_valid"),("profad","profad_total","profad_scoring_valid"),("mdafs","mdafs_global","mdafs_scoring_valid")]: qc[f"n_{name}_nonmissing"]=int(baseline[outcome].notna().sum()) if outcome in baseline else 0; qc[f"n_{name}_scoring_valid"]=int(valid.get(outcome,pd.Series(False,index=baseline.index)).sum())
    (qc_dir/"11_integrated_baseline_profile_qc.json").write_text(json.dumps(qc,indent=2,default=str))
    generated=sorted([p.relative_to(PROJECT_ROOT).as_posix() for p in [*tables.glob("11_*"),*qc_dir.glob("11_*"),*fig_dir.glob("11_*"),common.INTERMEDIATE_DATA_DIR/"11_integrated_baseline_profile"/"11_integrated_baseline_patient_level.parquet",common.INTERMEDIATE_DATA_DIR/"11_integrated_baseline_profile"/"11_integrated_baseline_patient_level.csv",common.OUTPUTS_DIR/"logs"/"11_integrated_baseline_profile"/"11_integrated_baseline_profile.log"]])
    print("Integrated baseline profile completed successfully."); print(f"Baseline patients: {len(baseline)}\nPop classifiable: {qc['n_pop_classifiable']}\nOverlap classifiable: {qc['n_overlap_classifiable']}\nValid PCS: {qc['n_pcs_scoring_valid']}\nValid MCS: {qc['n_mcs_scoring_valid']}\nValid PROFAD: {qc['n_profad_scoring_valid']}\nValid MDAFS: {qc['n_mdafs_scoring_valid']}"); print("Generated files:\n"+"\n".join(generated))

if __name__ == "__main__": main()

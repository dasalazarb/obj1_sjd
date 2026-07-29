#!/usr/bin/env python3
"""Observed-baseline cross-domain associations; no causal interpretation."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402

DEFAULT_INPUT = common.INTERMEDIATE_DATA_DIR / "21_multidimensional_baseline_patient.parquet"
CORRELATION_VARIABLES = ["essdai_total", "esspri_total", "esspri_dryness", "esspri_fatigue",
    "esspri_pain", "sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global",
    "n_extraglandular_domains_active", "n_prespecified_comorbidities", "fibromyalgia",
    "depression", "anti_ro_pos", "low_c4"]
BLOCKS = {**dict.fromkeys(["essdai_total", "n_extraglandular_domains_active"], "Systemic activity"),
    **dict.fromkeys(["esspri_total", "esspri_dryness", "esspri_fatigue", "esspri_pain"], "Symptom burden"),
    **dict.fromkeys(["sf36_pcs", "sf36_mcs"], "Functional status"),
    **dict.fromkeys(["profad_total", "mdafs_global"], "Fatigue and discomfort"),
    **dict.fromkeys(["anti_ro_pos", "low_c4"], "Serology"),
    **dict.fromkeys(["n_prespecified_comorbidities", "fibromyalgia", "depression"], "Comorbidities")}
BINARY = {"fibromyalgia", "depression", "anti_ro_pos", "low_c4", "glandular_active",
    "eg_articular_active", "eg_pulmonary_active", "eg_renal_active", "eg_pns_active",
    "eg_cns_active", "eg_hematologic_active", "eg_biological_active"}
DOMAINS = ["glandular_active", "eg_articular_active", "eg_pulmonary_active", "eg_renal_active",
    "eg_pns_active", "eg_cns_active", "eg_hematologic_active", "eg_biological_active"]
MODEL_COLUMNS = ["model_name", "analysis_group", "outcome", "predictor", "predictor_type",
    "contrast_or_unit", "estimate", "standard_error", "ci95_low", "ci95_high", "p_value",
    "fdr_bh_q_value", "n_complete", "n_patients", "formula", "fatigue_measure_used",
    "model_status", "interpretation"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--min-pair-n", type=int, default=20)
    p.add_argument("--min-model-n", type=int, default=50)
    p.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    p.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    p.add_argument("--skip-models", action="store_true")
    return p.parse_args(argv)


def load_baseline(path):
    if not Path(path).exists():
        raise FileNotFoundError("Run src/block_B/21_multidimensional_baseline_phenotypes.py first.")
    data = pd.read_parquet(path)
    missing = [c for c in ("patient_id", "pop_status") if c not in data]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if data.patient_id.isna().any():
        raise ValueError("patient_id must be nonmissing")
    if data.patient_id.duplicated().any():
        raise ValueError("patient_id must be unique (one row per patient)")
    if data.empty:
        raise ValueError("No usable baseline patients")
    return data.copy()


def define_ssa_group(data):
    out = data.copy()
    interpretable = out.get("anti_ro_interpretable", pd.Series(False, index=out.index)).eq(True)
    anti_ro = out.get("anti_ro_pos", pd.Series(pd.NA, index=out.index))
    positive, negative = anti_ro.eq(True), anti_ro.eq(False)
    out["ssa_group"] = np.select([interpretable & positive, interpretable & negative],
                                  ["SSA_positive", "SSA_negative"], default="SSA_unknown")
    return out


def _bh(values):
    values = np.asarray(values, float); order = np.argsort(values); ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values)+1))[::-1])[::-1]
    result = np.empty(len(values)); result[order] = np.minimum(adjusted, 1.0)
    return result


def build_correlations(data, min_pair_n=20, analysis_group="Overall"):
    rows = []
    for x, y in combinations(CORRELATION_VARIABLES, 2):
        row = {"variable_x": x, "variable_y": y, "clinical_block_x": BLOCKS[x],
               "clinical_block_y": BLOCKS[y], "n_complete_pairs": 0, "spearman_rho": np.nan,
               "p_value": np.nan, "fdr_bh_q_value": np.nan, "analysis_group": analysis_group,
               "status": "missing_variable"}
        if x in data and y in data:
            pair = pd.DataFrame({x: pd.to_numeric(data[x], errors="coerce"), y: pd.to_numeric(data[y], errors="coerce")})
            for variable, flag in (("anti_ro_pos", "anti_ro_interpretable"), ("low_c4", "c4_interpretable")):
                if variable in pair and flag in data:
                    pair.loc[~data[flag].eq(True), variable] = np.nan
            pair = pair.dropna(); row["n_complete_pairs"] = len(pair)
            if len(pair) < min_pair_n: row["status"] = "insufficient_pairs"
            elif pair[x].nunique() < 2 or pair[y].nunique() < 2: row["status"] = "no_variation"
            else:
                row["spearman_rho"], row["p_value"] = spearmanr(pair[x], pair[y]); row["status"] = "estimated"
        rows.append(row)
    result = pd.DataFrame(rows)
    mask = result.status.eq("estimated")
    if mask.any(): result.loc[mask, "fdr_bh_q_value"] = _bh(result.loc[mask, "p_value"])
    return result


def fit_ols_model(data, outcome, predictors, model_name, group="Overall", min_n=50,
                  fatigue="", binary_min=5, interpretation="Baseline cross-domain association; no causal interpretation."):
    formula = outcome + " ~ " + " + ".join("C(protocol_group)" if x == "protocol_group" else x for x in predictors)
    base = dict(model_name=model_name, analysis_group=group, outcome=outcome, predictor=pd.NA,
        predictor_type=pd.NA, contrast_or_unit=pd.NA, estimate=np.nan, standard_error=np.nan,
        ci95_low=np.nan, ci95_high=np.nan, p_value=np.nan, fdr_bh_q_value=np.nan, n_complete=0,
        n_patients=0, formula=formula, fatigue_measure_used=fatigue, interpretation=interpretation)
    missing = [c for c in [outcome] + predictors if c not in data]
    if missing:
        return pd.DataFrame([{**base, "model_status": "skipped_missing_variable"}], columns=MODEL_COLUMNS)
    use = data[[outcome] + predictors].copy()
    for c in [outcome] + [p for p in predictors if p != "protocol_group"]:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.dropna(); base.update(n_complete=len(use), n_patients=len(use))
    if len(use) < min_n:
        return pd.DataFrame([{**base, "model_status": "skipped_insufficient_data"}], columns=MODEL_COLUMNS)
    if use[outcome].nunique() < 2 or any(use[p].nunique() < 2 for p in predictors):
        return pd.DataFrame([{**base, "model_status": "skipped_no_variation"}], columns=MODEL_COLUMNS)
    for p in set(predictors) & BINARY:
        if use[p].value_counts().min() < binary_min:
            return pd.DataFrame([{**base, "model_status": "skipped_insufficient_data"}], columns=MODEL_COLUMNS)
    continuous = [p for p in predictors if p not in BINARY and p != "protocol_group"]
    for p in continuous:
        use[p] = (use[p] - use[p].mean()) / use[p].std(ddof=0)
    try:
        import statsmodels.formula.api as smf
        fitted = smf.ols(formula, use).fit(cov_type="HC3")
        if np.linalg.matrix_rank(fitted.model.exog) < fitted.model.exog.shape[1]:
            return pd.DataFrame([{**base, "model_status": "failed"}], columns=MODEL_COLUMNS)
    except ImportError:
        return pd.DataFrame([{**base, "model_status": "skipped_dependency_unavailable"}], columns=MODEL_COLUMNS)
    except Exception:
        return pd.DataFrame([{**base, "model_status": "failed"}], columns=MODEL_COLUMNS)
    rows = []
    for term in fitted.params.index:
        if term == "Intercept": continue
        original = "protocol_group" if term.startswith("C(protocol_group)") else term
        kind = "categorical" if original == "protocol_group" else "binary" if original in BINARY else "continuous"
        unit = "category versus reference" if kind == "categorical" else "present versus absent" if kind == "binary" else "per 1-SD higher predictor"
        ci = fitted.conf_int().loc[term]
        rows.append({**base, "predictor": term, "predictor_type": kind, "contrast_or_unit": unit,
            "estimate": fitted.params[term], "standard_error": fitted.bse[term], "ci95_low": ci.iloc[0],
            "ci95_high": ci.iloc[1], "p_value": fitted.pvalues[term], "model_status": "fitted"})
    return pd.DataFrame(rows, columns=MODEL_COLUMNS)


def make_correlation_heatmap(correlations, path):
    variables = [v for v in CORRELATION_VARIABLES if ((correlations.variable_x == v) | (correlations.variable_y == v)).any()]
    rho = pd.DataFrame(np.nan, index=variables, columns=variables); n = rho.copy()
    np.fill_diagonal(rho.values, 1)
    for row in correlations.loc[correlations.analysis_group.eq("Overall")].itertuples():
        rho.loc[row.variable_x, row.variable_y] = rho.loc[row.variable_y, row.variable_x] = row.spearman_rho
        n.loc[row.variable_x, row.variable_y] = n.loc[row.variable_y, row.variable_x] = row.n_complete_pairs
    cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("lightgray")
    fig, ax = plt.subplots(figsize=(11, 9)); im = ax.imshow(rho, vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(variables)), variables, rotation=60, ha="right", fontsize=7); ax.set_yticks(range(len(variables)), variables, fontsize=7)
    for i in range(len(variables)):
        for j in range(len(variables)):
            if np.isfinite(rho.iloc[i,j]): ax.text(j, i, f"{rho.iloc[i,j]:.2f}\n(n={n.iloc[i,j]:.0f})" if i != j else "1.00", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, label="Spearman rho"); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def make_forestplot(models, path):
    fitted = models.loc[models.model_status.eq("fitted") & models.predictor.notna()].copy()
    outcomes = [x for x in ["sf36_pcs", "esspri_total"] if x in set(fitted.outcome)]
    fig, axes = plt.subplots(max(1, len(outcomes)), 1, figsize=(10, max(3, len(fitted)*.22)), squeeze=False)
    if not outcomes: axes[0,0].text(.5,.5,"No adjusted models fitted",ha="center")
    for ax, outcome in zip(axes[:,0], outcomes):
        d=fitted.loc[fitted.outcome.eq(outcome)].reset_index(drop=True); y=np.arange(len(d))
        ax.errorbar(d.estimate,y,xerr=[d.estimate-d.ci95_low,d.ci95_high-d.estimate],fmt="o"); ax.axvline(0,color="gray",lw=1)
        ax.set_yticks(y,[f"{r.predictor} | {r.analysis_group} (n={r.n_complete})" for r in d.itertuples()],fontsize=7); ax.invert_yaxis(); ax.set_title(outcome+" (outcome-unit coefficients)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main(argv=None):
    args=parse_args(argv); data=define_ssa_group(load_baseline(args.input))
    paths=[common.BLOCKB_TABLES_DIR/n for n in ["22_cross_domain_correlations.csv","22_pcs_multivariable_model.csv","22_esspri_multivariable_model.csv","22_esspri_domain_associations.csv","22_ssa_stratified_associations.csv","22_analysis_denominators.csv"]]
    paths += [common.BLOCKB_FIGURES_DIR/n for n in ["22_cross_domain_correlation_heatmap.pdf","22_adjusted_associations_forestplot.pdf"]]
    if not args.overwrite and any(p.exists() for p in paths): raise FileExistsError("Output exists; use --overwrite")
    for p in (common.BLOCKB_TABLES_DIR,common.BLOCKB_FIGURES_DIR,common.BLOCKB_QC_DIR,common.BLOCKB_LOGS_DIR): p.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=common.BLOCKB_LOGS_DIR/"22_cross_domain_correlations.log",level=logging.INFO,filemode="w",format="%(asctime)s | %(message)s")
    counts=data.ssa_group.value_counts(); logging.info("Input loaded: %d patients; SSA groups %s",len(data),counts.to_dict())
    correlations=pd.concat([build_correlations(data,args.min_pair_n,"Overall")]+[build_correlations(data.loc[data.ssa_group.eq(g)],args.min_pair_n,g) for g in ["SSA_positive","SSA_negative"]],ignore_index=True)
    pcs_predictors=["essdai_total","esspri_total","profad_total","fibromyalgia","depression","age_baseline","protocol_group","time_since_diagnosis_years"]
    fatigue="profad_total"
    if "profad_total" not in data or data[[c for c in ["sf36_pcs"]+pcs_predictors if c in data]].dropna().shape[0] < args.min_model_n:
        pcs_predictors=["mdafs_global" if x=="profad_total" else x for x in pcs_predictors]; fatigue="mdafs_global"
    ess_predictors=["anti_ro_pos","low_c4","fibromyalgia","depression","glandular_active","n_extraglandular_domains_active","age_baseline","protocol_group","time_since_diagnosis_years"]
    model_data=data.copy()
    if "anti_ro_interpretable" in data: model_data.loc[~data.anti_ro_interpretable.eq(True),"anti_ro_pos"]=np.nan
    if "c4_interpretable" in data: model_data.loc[~data.c4_interpretable.eq(True),"low_c4"]=np.nan
    if args.skip_models:
        skip=lambda name,out: pd.DataFrame([{**{c:pd.NA for c in MODEL_COLUMNS},"model_name":name,"analysis_group":"Overall","outcome":out,"model_status":"skipped_by_user","n_complete":0,"n_patients":0}],columns=MODEL_COLUMNS)
        pcs=skip("pcs_primary","sf36_pcs"); ess=skip("esspri_primary","esspri_total"); domains=skip("esspri_domain_specific","esspri_total"); strat=skip("ssa_stratified","multiple")
    else:
        pcs=fit_ols_model(model_data,"sf36_pcs",pcs_predictors,"pcs_primary",min_n=args.min_model_n,fatigue=fatigue)
        ess=fit_ols_model(model_data,"esspri_total",ess_predictors,"esspri_primary",min_n=args.min_model_n)
        domain_rows=[]
        for d in DOMAINS:
            domain_rows.append(fit_ols_model(data,"esspri_total",[d,"age_baseline","protocol_group","time_since_diagnosis_years"],"esspri_domain_specific",group="Overall",min_n=args.min_model_n,binary_min=10,interpretation="exploratory domain-specific association"))
        domains=pd.concat(domain_rows,ignore_index=True)
        ok=domains.model_status.eq("fitted") & domains.predictor.isin(DOMAINS)
        if ok.any(): domains.loc[ok,"fdr_bh_q_value"]=_bh(domains.loc[ok,"p_value"])
        strat_rows=[]
        label="Exploratory SSA-stratified association; comparison between strata is descriptive."
        for g in ["SSA_positive","SSA_negative"]:
            subset=model_data.loc[model_data.ssa_group.eq(g)]
            strat_rows += [fit_ols_model(subset,"sf36_pcs",pcs_predictors,"pcs_ssa_stratified",g,args.min_model_n,fatigue,10,label)]
            ep=[x for x in ess_predictors if x!="anti_ro_pos"]
            if "low_c4" in subset and subset.low_c4.dropna().nunique()<2: ep.remove("low_c4")
            strat_rows += [fit_ols_model(subset,"esspri_total",ep,"esspri_ssa_stratified",g,args.min_model_n,"",10,label)]
        strat=pd.concat(strat_rows,ignore_index=True)
    all_models=pd.concat([pcs,ess,domains,strat],ignore_index=True)
    den=[]
    for d in all_models.drop_duplicates(["model_name","analysis_group","formula"]).itertuples():
        den.append({"analysis_name":d.model_name,"analysis_group":d.analysis_group,"outcome":d.outcome,"n_initial":len(data),"n_complete":d.n_complete,"n_excluded_missing":len(data)-int(d.n_complete or 0),"n_ssa_positive":int(counts.get("SSA_positive",0)),"n_ssa_negative":int(counts.get("SSA_negative",0)),"n_ssa_unknown":int(counts.get("SSA_unknown",0)),"model_status":d.model_status})
    correlations.to_csv(paths[0],index=False); pcs.to_csv(paths[1],index=False); ess.to_csv(paths[2],index=False); domains.to_csv(paths[3],index=False); strat.to_csv(paths[4],index=False); pd.DataFrame(den).to_csv(paths[5],index=False)
    make_correlation_heatmap(correlations,paths[6]); make_forestplot(all_models,paths[7])
    qc={"input_path":str(args.input),"n_input_patients":len(data),"n_unique_patients":data.patient_id.nunique(),"n_ssa_positive":int(counts.get("SSA_positive",0)),"n_ssa_negative":int(counts.get("SSA_negative",0)),"n_ssa_unknown":int(counts.get("SSA_unknown",0)),"n_correlations_requested":len(correlations),"n_correlations_estimated":int(correlations.status.eq("estimated").sum()),"n_primary_models_requested":2,"n_primary_models_fitted":int(pcs.model_status.eq("fitted").any())+int(ess.model_status.eq("fitted").any()),"n_domain_models_fitted":int(domains.loc[domains.model_status.eq("fitted"),"formula"].nunique()),"n_stratified_models_fitted":int(strat.loc[strat.model_status.eq("fitted"),["model_name","analysis_group"]].drop_duplicates().shape[0]),"pcs_model_n":int(pd.to_numeric(pcs.n_complete,errors="coerce").max() or 0),"esspri_model_n":int(pd.to_numeric(ess.n_complete,errors="coerce").max() or 0),"pcs_maximum_vif":None,"esspri_maximum_vif":None,"warnings":[]}
    with open(common.BLOCKB_QC_DIR/"22_cross_domain_qc.json","w") as f: json.dump(qc,f,indent=2)
    logging.info("Correlations estimated: %d; models fitted: %d; outputs written",qc["n_correlations_estimated"],int(all_models.model_status.eq("fitted").sum()))

if __name__ == "__main__": main()

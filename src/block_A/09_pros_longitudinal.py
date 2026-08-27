#!/usr/bin/env python3
"""ITEMS 7.1–7.3: canonical episode-level PRO analysis.

Scoring is delegated exclusively to :mod:`src.derivations.pro_scoring`.  This
entry point consumes upstream episode assignments and clinical timing; it never
matches assessments by date or reconstructs visits/baseline.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import common  # noqa: E402
from src.derivations.pro_scoring import (  # noqa: E402
    PRO_ITEM_COLUMNS, PRO_MEASURES, PRO_PRIMARY_MEASURES, PRO_SCORE_RANGES,
    SF36_MEASURES, score_all_pros,
)

LOG = logging.getLogger("pros_longitudinal")
OUT_STEM = common.INTERMEDIATE_DATA_DIR / "09_pros_episode_level"
TABLES = common.BLOCKA_TABLES_DIR
QC = common.BLOCKA_QC_DIR
FIGURES = common.OUTPUTS_DIR / "figures" / "blockA"
SPINE_COLUMNS = [
    "patient_id", "clinical_episode_id", "clinical_anchor_date",
    "clinical_visit_number", "is_clinical_baseline",
    "clinical_baseline_episode_id", "clinical_baseline_date",
    "time_since_clinical_baseline_days", "time_since_clinical_baseline_years",
]
LABELS = {
    "esspri_dryness": "ESSPRI dryness", "esspri_fatigue": "ESSPRI fatigue",
    "esspri_pain": "ESSPRI pain", "esspri_total": "ESSPRI total",
    "esspri_partial_mean": "ESSPRI partial mean", "profad_total": "PROFAD total",
    "mdafs_global": "MDAFS global", **SF36_MEASURES,
}
NORM = {"sf36_pcs": 50.0, "sf36_mcs": 50.0}
PROTOCOL_COLUMNS = ("parent_protocol", "protocol", "protocol_parent", "source_protocol")
ESSPRI_COMPARISON_NOTE = (
    "ESSPRI contributes directly to baseline Pop classification; inferential "
    "comparison by Pop is not performed to avoid circular interpretation."
)
CLINICAL_VISIT_DESCRIPTIVE_NOTE = (
    "Clinical visit number is used for descriptive summaries only. "
    "Longitudinal inference uses actual elapsed clinical time. Late visit "
    "numbers have small sample sizes and should not be interpreted as "
    "population-level trajectories."
)


def read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--responses", type=Path, default=common.SOURCE_EPISODE_SPINE,
                   help="Canonical episode-assigned dataset containing raw PRO responses")
    p.add_argument("--spine", type=Path, default=common.CLINICAL_VISIT_SPINE_PARQUET)
    p.add_argument("--pop", type=Path, default=common.POP_LONGITUDINAL_PARQUET)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def load_spine(path: Path) -> pd.DataFrame:
    spine = read(path)
    missing = set(SPINE_COLUMNS) - set(spine)
    if missing:
        raise KeyError(f"Clinical spine lacks {sorted(missing)}")
    optional = [c for c in (*PROTOCOL_COLUMNS, "baseline_pop") if c in spine]
    spine = spine[SPINE_COLUMNS + optional].copy()
    for c in ("clinical_anchor_date", "clinical_baseline_date"):
        spine[c] = pd.to_datetime(spine[c], errors="coerce")
    keys = ["patient_id", "clinical_episode_id"]
    if spine[keys].isna().any().any() or spine.duplicated(keys).any():
        raise ValueError("Clinical spine violates patient + clinical episode identity")
    base = spine.loc[spine.is_clinical_baseline.eq(True)]
    if base.patient_id.duplicated().any():
        raise ValueError("More than one clinical baseline for a patient")
    if not base.clinical_episode_id.astype("string").eq(base.clinical_baseline_episode_id.astype("string")).all():
        raise ValueError("Clinical baseline episode mismatch")
    if not base.clinical_anchor_date.eq(base.clinical_baseline_date).all():
        raise ValueError("Clinical baseline date mismatch")
    return spine


def attach_baseline_pop(spine: pd.DataFrame, pop_path: Path) -> pd.DataFrame:
    if "baseline_pop" in spine:
        return spine
    if not pop_path.exists():
        spine["baseline_pop"] = pd.NA
        return spine
    pop = read(pop_path)
    candidate = next((c for c in ("baseline_pop", "baseline_pop_status", "pop_baseline_status") if c in pop), None)
    if candidate is None:
        spine["baseline_pop"] = pd.NA
        return spine
    values = pop[["patient_id", candidate]].dropna(subset=[candidate]).drop_duplicates()
    if values.patient_id.duplicated().any():
        raise ValueError("Upstream baseline Pop is not patient-constant")
    return spine.merge(values.rename(columns={candidate: "baseline_pop"}), on="patient_id", how="left", validate="many_to_one")


def attach_parent_protocol(spine: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical upstream protocol metadata by episode identity."""
    spine_candidate = next((column for column in PROTOCOL_COLUMNS if column in spine), None)
    if spine_candidate is not None and spine_candidate != "parent_protocol":
        spine = spine.rename(columns={spine_candidate: "parent_protocol"})
    if "parent_protocol" in spine and spine["parent_protocol"].notna().all():
        return spine
    candidate = next((column for column in PROTOCOL_COLUMNS if column in source), None)
    if candidate is None:
        if "parent_protocol" not in spine:
            spine["parent_protocol"] = pd.NA
        return spine
    keys = ["patient_id", "clinical_episode_id"]
    protocol = source[keys + [candidate]].dropna(subset=keys).copy()
    conflicting = protocol.dropna(subset=[candidate]).groupby(keys)[candidate].nunique().gt(1)
    if conflicting.any():
        raise ValueError("Upstream parent protocol is not unique by patient + clinical episode")
    protocol = protocol.drop_duplicates(keys).rename(columns={candidate: "_upstream_parent_protocol"})
    result = spine.merge(protocol, on=keys, how="left", validate="one_to_one")
    if "parent_protocol" in result:
        result["parent_protocol"] = result["parent_protocol"].fillna(result.pop("_upstream_parent_protocol"))
    else:
        result = result.rename(columns={"_upstream_parent_protocol": "parent_protocol"})
    return result


def instrument_columns(instrument: str, scored: pd.DataFrame) -> list[str]:
    prefix = {"ESSPRI": "esspri_", "SF-36": "sf36_", "PROFAD": "profad_", "MDAFS": "mdafs_"}[instrument]
    return [c for c in scored if c.startswith(prefix)]


def canonical_episode_data(raw: pd.DataFrame, spine: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    keys = ["patient_id", "clinical_episode_id"]
    missing = set(keys) - set(raw)
    if missing:
        raise KeyError("PRO responses require upstream episode assignment: " + ", ".join(sorted(missing)))
    raw = raw.copy()
    raw["_raw_row"] = np.arange(len(raw))
    known = spine[keys].drop_duplicates().assign(_mapped=True)
    mapped = raw.merge(known, on=keys, how="left", validate="many_to_one")
    outside_spine = mapped["_mapped"].isna()
    valid_identity = mapped[keys].notna().all(axis=1)
    should_be_clinical = mapped.get("clinical_visit", pd.Series(False, index=mapped.index)).eq(True)
    nonclinical = outside_spine & valid_identity & ~should_be_clinical
    true_failure = outside_spine & should_be_clinical
    analysis_raw = mapped.loc[~outside_spine].drop(columns="_mapped").copy()
    scored = score_all_pros(analysis_raw)
    violations = pd.DataFrame(scored.attrs.get("pro_range_violations", []))
    episode = spine.copy()
    conflicts: list[dict[str, Any]] = []
    duplicate_groups = 0
    for instrument, primary in PRO_PRIMARY_MEASURES.items():
        cols = instrument_columns(instrument, scored)
        stem = instrument.lower().replace("-", "")
        selected = []
        for key, group in scored.groupby(keys, dropna=False, sort=False):
            has_any = group[f"{stem}_any_item_present"]
            relevant = group.loc[has_any]
            if relevant.empty:
                continue
            if len(relevant) > 1:
                duplicate_groups += 1
            valid = relevant.loc[relevant[primary].notna()]
            values = valid[primary].dropna().unique()
            conflict = len(values) > 1
            if conflict:
                conflicts.append({"patient_id": key[0], "clinical_episode_id": key[1], "instrument": instrument,
                                  "n_rows": len(relevant), "n_valid": len(valid), "distinct_valid_scores": "|".join(map(str, values)),
                                  "resolution": "scores_set_missing"})
                chosen = relevant.iloc[0].copy()
                chosen[cols] = np.nan
            else:
                chosen = (valid.iloc[0] if len(valid) else relevant.iloc[0]).copy()
            record = {keys[0]: key[0], keys[1]: key[1], **{c: chosen[c] for c in cols}}
            record[f"{stem}_available"] = True
            record[f"{stem}_complete"] = bool(not conflict and pd.notna(chosen[primary]))
            record[f"{stem}_conflict"] = conflict
            selected.append(record)
        if selected:
            episode = episode.merge(pd.DataFrame(selected), on=keys, how="left", validate="one_to_one")
        for suffix in ("available", "complete", "conflict"):
            col = f"{stem}_{suffix}"
            if col not in episode:
                episode[col] = False
            episode[col] = episode[col].fillna(False).astype(bool)
    # Public names requested by the specification.
    episode = episode.rename(columns={"sf36_available": "sf36_available", "sf36_complete": "sf36_complete"})
    episode = episode.sort_values(["patient_id", "clinical_anchor_date", "clinical_episode_id"]).reset_index(drop=True)
    mapping = pd.DataFrame([{
        "n_source_episode_rows": len(raw),
        "n_source_unique_episodes": int(mapped.loc[valid_identity, keys].drop_duplicates().shape[0]),
        "n_clinical_episodes_included": int(mapped.loc[~outside_spine, keys].drop_duplicates().shape[0]),
        "n_nonclinical_episodes_excluded": int(mapped.loc[nonclinical, keys].drop_duplicates().shape[0]),
        "n_true_episode_mapping_failures": int(true_failure.sum()),
        "n_rows_with_patient_id": int(raw.patient_id.notna().sum()),
        "n_rows_with_episode_id": int(raw.clinical_episode_id.notna().sum()),
        "n_multiple_rows_same_patient_episode_instrument": duplicate_groups,
        "n_conflicts_same_patient_episode_instrument": len(conflicts),
    }])
    return episode, pd.DataFrame(conflicts), violations, {**mapping.iloc[0].to_dict()}


def measures() -> list[tuple[str, str]]:
    return [(inst, measure) for inst, cols in PRO_MEASURES.items() for measure in cols]


def describe(values: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return {"n_available": len(x), "mean": x.mean(), "sd": x.std(ddof=1), "median": x.median(),
            "q1": x.quantile(.25), "q3": x.quantile(.75), "min": x.min(), "max": x.max()}


def baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df.loc[df.is_clinical_baseline.eq(True)]
    rows = []
    for inst, measure in measures():
        d = describe(base[measure] if measure in base else pd.Series(dtype=float))
        threshold = 5.0 if measure == "esspri_total" else np.nan
        row = {"instrument": inst, "measure": measure, "measure_label": LABELS.get(measure, measure),
               "n_total_baseline": len(base), **d}
        row.update(n_missing=len(base)-d["n_available"], pct_available=100*d["n_available"]/len(base) if len(base) else np.nan,
                   pct_missing=100*(len(base)-d["n_available"])/len(base) if len(base) else np.nan,
                   clinical_threshold=threshold, n_above_threshold=int(base[measure].ge(threshold).sum()) if pd.notna(threshold) and measure in base else np.nan,
                   pct_above_threshold=100*base[measure].ge(threshold).sum()/d["n_available"] if pd.notna(threshold) and d["n_available"] else np.nan,
                   normative_reference="50 ± 10" if measure in NORM else pd.NA,
                   difference_from_normative_mean=d["mean"]-NORM[measure] if measure in NORM else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def by_pop_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df.loc[df.is_clinical_baseline.eq(True)]
    rows=[]
    for inst, measure in measures():
        groups = [pd.to_numeric(base.loc[base.baseline_pop.eq(p), measure], errors="coerce").dropna() for p in ("Pop1", "Pop2", "Pop3")] if measure in base else []
        definitional = inst == "ESSPRI"
        pval = (kruskal(*groups).pvalue if not definitional and len(groups) == 3
                and all(len(g) >= 2 for g in groups) else np.nan)
        role = "definitional_descriptive" if definitional else "independent_descriptive_inferential"
        for pop, vals in zip(("Pop1", "Pop2", "Pop3"), groups):
            d=describe(vals); rows.append({"baseline_pop":pop,"instrument":inst,"measure":measure,"N":int(base.baseline_pop.eq(pop).sum()),**d,"iqr":d["q3"]-d["q1"],"kruskal_wallis_p_value":pval,
                                           "comparison_role": role,
                                           "comparison_note": ESSPRI_COMPARISON_NOTE if definitional else pd.NA})
    out=pd.DataFrame(rows)
    out["q_value"] = np.nan
    if len(out):
        tests = out.drop_duplicates(["instrument", "measure"]).set_index(["instrument", "measure"])["kruskal_wallis_p_value"]
        adjusted = fdr(tests)
        out["q_value"] = [adjusted.get((inst, measure), np.nan) for inst, measure in zip(out.instrument, out.measure)]
    return out


def by_visit_table(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for visit, group in df.groupby("clinical_visit_number", dropna=False):
        for inst, measure in measures():
            d=describe(group[measure] if measure in group else pd.Series(dtype=float))
            rows.append({"clinical_visit_number":visit,"instrument":inst,"measure":measure,
                         "n_patients_observed":group.patient_id.nunique(),**d,
                         "pct_available":100*d["n_available"]/group.patient_id.nunique() if group.patient_id.nunique() else np.nan,
                         "methodological_note":CLINICAL_VISIT_DESCRIPTIVE_NOTE})
    return pd.DataFrame(rows)


def availability_table(df: pd.DataFrame, grouping: str) -> pd.DataFrame:
    rows=[]
    for value, group in df.groupby(grouping, dropna=False):
        for inst in PRO_PRIMARY_MEASURES:
            stem=inst.lower().replace("-", "")
            any_col, complete_col=f"{stem}_available",f"{stem}_complete"
            any_n=int(group[any_col].sum()); complete_n=int(group[complete_col].sum())
            n_patients=group.patient_id.nunique()
            row={grouping:value,"instrument":inst,"n_patients":n_patients,"n_with_any_item":any_n,"n_with_complete_score":complete_n,
                 "pct_with_any_item":100*any_n/len(group) if len(group) else np.nan,"pct_with_complete_score":100*complete_n/len(group) if len(group) else np.nan}
            if grouping == "parent_protocol":
                patients_any=group.loc[group[any_col], "patient_id"].nunique()
                patients_complete=group.loc[group[complete_col], "patient_id"].nunique()
                row={grouping:value,"instrument":inst,"n_patients":n_patients,
                     "n_patients_with_any_item":patients_any,"n_patients_with_complete_score":patients_complete,
                     "pct_patients_with_any_item":100*patients_any/n_patients if n_patients else np.nan,
                     "pct_patients_with_complete_score":100*patients_complete/n_patients if n_patients else np.nan,
                     "n_episodes":len(group),"n_episodes_with_any_item":any_n,
                     "n_episodes_with_complete_score":complete_n,
                     "pct_episodes_with_any_item":100*any_n/len(group) if len(group) else np.nan,
                     "pct_episodes_with_complete_score":100*complete_n/len(group) if len(group) else np.nan}
            rows.append(row)
    return pd.DataFrame(rows)


def fdr(values: pd.Series) -> pd.Series:
    result=pd.Series(np.nan,index=values.index,dtype=float); valid=values.dropna().sort_values(); m=len(valid)
    if m:
        adjusted=(valid*np.arange(m,0,-1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
        # Standard BH in original rank order.
        ranked=values.dropna().sort_values(); q=(ranked*m/np.arange(1,m+1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
        result.loc[q.index]=q
    return result


def fit_model(data: pd.DataFrame, measure: str) -> dict[str, Any]:
    d=data[["patient_id","time_since_clinical_baseline_years",measure]].dropna()
    if d.patient_id.nunique() < 3 or len(d) < 6:
        return {"model_used":"none","model_status":"insufficient_data","annual_change_estimate":np.nan,"CI95_low":np.nan,"CI95_high":np.nan,"p_value":np.nan}
    y=d[measure].astype(float); x=pd.DataFrame({"const":1.0,"time":d.time_since_clinical_baseline_years.astype(float)},index=d.index)
    try:
        fit=MixedLM(y,x,groups=d.patient_id).fit(reml=False,method="lbfgs",disp=False)
        est,se,p=float(fit.params["time"]),float(fit.bse["time"]),float(fit.pvalues["time"]); used="LME"
    except Exception as exc:  # model failure is explicitly audited before the prespecified fallback
        LOG.warning("LME failed for %s: %s",measure,exc)
        try:
            fit=GEE(y,x,groups=d.patient_id,cov_struct=Exchangeable(),family=Gaussian()).fit()
            est,se,p=float(fit.params["time"]),float(fit.bse["time"]),float(fit.pvalues["time"]); used="GEE"
        except Exception as fallback:
            return {"model_used":"LME then GEE","model_status":f"failed: {fallback}","annual_change_estimate":np.nan,"CI95_low":np.nan,"CI95_high":np.nan,"p_value":np.nan}
    return {"model_used":used,"model_status":"fitted","annual_change_estimate":est,"CI95_low":est-1.96*se,"CI95_high":est+1.96*se,"p_value":p}


def longitudinal_tables(df: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,dict[str,int]]:
    rows=[]; models=[]; eligible_counts={}
    for inst, measure in measures():
        if measure not in df: continue
        valid=df.loc[df[measure].notna() & df.clinical_anchor_date.notna()].copy()
        eligible_ids=[]
        for pid,g in valid.groupby("patient_id"):
            if len(g)>=2 and g.clinical_anchor_date.nunique()>=2: eligible_ids.append(pid)
        eligible=valid.loc[valid.patient_id.isin(eligible_ids)].sort_values(["patient_id","clinical_anchor_date"])
        if measure == PRO_PRIMARY_MEASURES[inst]: eligible_counts[inst]=len(eligible_ids)
        first=eligible.groupby("patient_id",sort=False).first(); last=eligible.groupby("patient_id",sort=False).last()
        changes=last[measure]-first[measure] if len(first) else pd.Series(dtype=float)
        model=fit_model(eligible,measure)
        spans=eligible.groupby("patient_id").time_since_clinical_baseline_years.agg(lambda x:x.max()-x.min()) if len(eligible) else pd.Series(dtype=float)
        counts=eligible.groupby("patient_id").size() if len(eligible) else pd.Series(dtype=float)
        row={"instrument":inst,"measure":measure,"n_patients_longitudinal":len(eligible_ids),"n_observations":len(eligible),
             "n_missing":int(df[measure].isna().sum()),"median_observed_pro_span_years":spans.median(),"median_measurements_per_patient":counts.median(),
             "first_available_mean":first[measure].mean() if len(first) else np.nan,"last_available_mean":last[measure].mean() if len(last) else np.nan,
             "mean_first_to_last_change":changes.mean(),**model}
        rows.append(row); models.append({"instrument":inst,"measure":measure,"n_patients":len(eligible_ids),"n_observations":len(eligible),
                                         "longitudinal_summary_basis":"first_and_last_available_measurements",**model})
    out=pd.DataFrame(rows); model_qc=pd.DataFrame(models)
    if len(out): out["q_value"]=fdr(out.p_value)
    if len(model_qc): model_qc["q_value"]=fdr(model_qc.p_value)
    return out,model_qc,eligible_counts


def scoring_qc(raw: pd.DataFrame, episode: pd.DataFrame, violations: pd.DataFrame) -> pd.DataFrame:
    rows=[]; normalized = score_all_pros(raw)
    for inst, primary in PRO_PRIMARY_MEASURES.items():
        stem=inst.lower().replace("-","")
        rows.append({"instrument":inst,"n_raw_responses":int(normalized[f"{stem}_any_item_present"].sum()),"n_scored":int(episode[primary].notna().sum()),
                     "n_invalid":int((violations.instrument.eq(inst)).sum()) if len(violations) else 0,
                     "n_out_of_range":int((violations.instrument.eq(inst)).sum()) if len(violations) else 0,
                     "n_partial":int((episode[f"{stem}_available"] & ~episode[f"{stem}_complete"]).sum()),"n_complete":int(episode[f"{stem}_complete"].sum())})
    return pd.DataFrame(rows)


def range_violations(df: pd.DataFrame) -> int:
    return sum(int((pd.to_numeric(df[c],errors="coerce").notna() & ~pd.to_numeric(df[c],errors="coerce").between(lo,hi)).sum()) for c,(lo,hi) in PRO_SCORE_RANGES.items() if c in df)


def make_figures(by_visit: pd.DataFrame, by_pop: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True,exist_ok=True)
    primary={"ESSPRI":"esspri_total","SF-36":"sf36_pcs","PROFAD":"profad_total","MDAFS":"mdafs_global"}
    for inst,measure in primary.items():
        d=by_visit.loc[by_visit.measure.eq(measure)].sort_values("clinical_visit_number")
        fig,ax=plt.subplots(figsize=(7,4)); ax.plot(d.clinical_visit_number,d["mean"],marker="o")
        for _,r in d.iterrows(): ax.annotate(f"n={int(r.n_available)}",(r.clinical_visit_number,r["mean"]),xytext=(0,6),textcoords="offset points",ha="center",fontsize=7)
        ax.set(xlabel="Clinical visit number (descriptive only)",ylabel=LABELS.get(measure,measure),title=f"{inst} trajectory")
        fig.tight_layout(); fig.savefig(FIGURES/f"09_{inst.lower().replace('-','')}_trajectory.pdf"); plt.close(fig)
    d=by_pop.loc[by_pop.measure.isin(primary.values())]
    fig,ax=plt.subplots(figsize=(9,5))
    for i,(measure,g) in enumerate(d.groupby("measure")):
        ax.plot(np.arange(3)+i*.06,g.set_index("baseline_pop").reindex(["Pop1","Pop2","Pop3"])["mean"],marker="o",label=LABELS.get(measure,measure))
    ax.set_xticks(range(3),["Pop1","Pop2","Pop3"]); ax.set_ylabel("Baseline mean (instrument scale)"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES/"09_pros_by_pop.pdf"); plt.close(fig)


def write(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite: raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    (df.to_parquet(path,index=False) if path.suffix==".parquet" else df.to_csv(path,index=False))


def main() -> None:
    args=parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    for d in (OUT_STEM.parent,TABLES,QC,FIGURES): d.mkdir(parents=True,exist_ok=True)
    raw=read(args.responses)
    spine=attach_parent_protocol(attach_baseline_pop(load_spine(args.spine),args.pop), raw)
    episode,conflicts,violations,mapping=canonical_episode_data(raw,spine)
    duplicates=int(episode.duplicated(["patient_id","clinical_episode_id"]).sum())
    multiple=int(episode.loc[episode.is_clinical_baseline.eq(True)].patient_id.duplicated().sum())
    score_range=range_violations(episode)
    negative=int(episode.time_since_clinical_baseline_years.lt(0).sum())
    if duplicates or multiple or score_range or negative:
        raise ValueError(f"Hard PRO QC failed: duplicates={duplicates}, multiple_baseline={multiple}, range={score_range}, negative_time={negative}")
    base=baseline_table(episode); pop=by_pop_table(episode); visit=by_visit_table(episode)
    avail_visit=availability_table(episode,"clinical_visit_number")
    avail_protocol=availability_table(episode,"parent_protocol")
    longitudinal,model_qc,eligible=longitudinal_tables(episode)
    score_qc=scoring_qc(raw,episode,violations)
    summary={"n_patients_clinical_spine":spine.patient_id.nunique(),"n_clinical_episodes":len(spine),"n_pro_episode_rows":len(episode),
             "n_baseline_patients":int(episode.is_clinical_baseline.sum()),
             **{f"n_{inst.lower().replace('-','')}_baseline_available":int(episode.loc[episode.is_clinical_baseline.eq(True),primary].notna().sum()) for inst,primary in PRO_PRIMARY_MEASURES.items()},
             **{f"n_{inst.lower().replace('-','')}_longitudinal_eligible":eligible.get(inst,0) for inst in PRO_PRIMARY_MEASURES},
             "n_duplicate_patient_episode":duplicates,"n_multiple_baseline":multiple,
             "n_nonclinical_source_episodes_excluded":mapping["n_nonclinical_episodes_excluded"],
             "n_true_episode_mapping_failures":mapping["n_true_episode_mapping_failures"],
             "n_clinical_episodes_parent_protocol_known":int(episode.parent_protocol.notna().sum()),
             "n_clinical_episodes_parent_protocol_unknown":int(episode.parent_protocol.isna().sum()),
             "n_11d_episodes":int(episode.parent_protocol.astype("string").str.upper().eq("11D").sum()),
             "n_15d_episodes":int(episode.parent_protocol.astype("string").str.upper().eq("15D").sum()),
             "n_11d_patients":int(episode.loc[episode.parent_protocol.astype("string").str.upper().eq("11D"),"patient_id"].nunique()),
             "n_15d_patients":int(episode.loc[episode.parent_protocol.astype("string").str.upper().eq("15D"),"patient_id"].nunique()),
             "n_scoring_range_violations":score_range}
    artifacts=[(episode,OUT_STEM.with_suffix(".parquet")),(episode,OUT_STEM.with_suffix(".csv")),(base,TABLES/"09_pros_baseline.csv"),(pop,TABLES/"09_pros_by_baseline_pop.csv"),(visit,TABLES/"09_pros_by_clinical_visit_number.csv"),(avail_visit,TABLES/"09_pros_availability_by_clinical_visit_number.csv"),(avail_protocol,TABLES/"09_pros_availability_by_protocol.csv"),(longitudinal,TABLES/"09_pros_longitudinal_change.csv"),(pd.DataFrame([summary]),QC/"09_pros_qc_summary.csv"),(score_qc,QC/"09_pros_scoring_qc.csv"),(pd.DataFrame([mapping]),QC/"09_pros_episode_mapping_qc.csv"),(conflicts,QC/"09_pros_duplicate_conflicts.csv"),(model_qc,QC/"09_pros_model_qc.csv")]
    for frame,path in artifacts: write(frame,path,args.overwrite)
    make_figures(visit,pop)
    LOG.info("Wrote canonical PRO analysis for %d clinical episodes",len(episode))

if __name__ == "__main__": main()

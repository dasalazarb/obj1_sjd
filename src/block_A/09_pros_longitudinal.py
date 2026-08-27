#!/usr/bin/env python3
"""Canonical clinical-episode PRO pipeline (Story Map items 7.1--7.3).

Scoring belongs exclusively to ``src.derivations.pro_scoring``. This module
joins responses carrying an upstream episode assignment to the authoritative
clinical spine; it never reconstructs episodes, visits, baseline, or timing.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import common  # noqa: E402
from src.derivations.pro_scoring import (  # noqa: E402
    ESSPRI_COMPONENTS, MDAFS_ACTIVITY_FLAGS, MDAFS_ITEMS, PROFAD_ITEMS,
    PRO_SCORE_RANGES, SF36_ITEMS, SF36_MEASURES, score_all_pros,
)

LOG = logging.getLogger("pros_clinical_episode")
KEYS = ["patient_id", "clinical_episode_id"]
SPINE_COLUMNS = [*KEYS, "clinical_anchor_date", "clinical_visit_number",
    "is_clinical_baseline", "clinical_baseline_episode_id", "clinical_baseline_date",
    "time_since_clinical_baseline_days", "time_since_clinical_baseline_years"]
TABLES = Path(common.BLOCKA_TABLES_DIR)
QC = Path(common.BLOCKA_QC_DIR)
INTERMEDIATE = Path(common.INTERMEDIATE_DATA_DIR)


@dataclass(frozen=True)
class Instrument:
    name: str
    prefix: str
    items: tuple[str, ...]
    primary: str
    measures: tuple[str, ...]


INSTRUMENTS = (
    Instrument("ESSPRI", "esspri", tuple(ESSPRI_COMPONENTS.values()), "esspri_total",
        ("esspri_dryness", "esspri_fatigue", "esspri_pain", "esspri_total", "esspri_partial_mean")),
    Instrument("SF-36", "sf36", tuple(SF36_ITEMS), "sf36_pcs", tuple(SF36_MEASURES)),
    Instrument("PROFAD", "profad", tuple(PROFAD_ITEMS), "profad_total", ("profad_total",)),
    Instrument("MDAFS", "mdafs", tuple(MDAFS_ITEMS + MDAFS_ACTIVITY_FLAGS), "mdafs_global", ("mdafs_global",)),
)
LABELS = {"esspri_dryness": "Dryness", "esspri_fatigue": "Fatigue", "esspri_pain": "Pain",
    "esspri_total": "ESSPRI total", "esspri_partial_mean": "Partial mean (sensitivity)",
    **SF36_MEASURES, "profad_total": "PROFAD total", "mdafs_global": "MDAFS global fatigue index"}


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet": return pd.read_parquet(path)
    if path.suffix.lower() == ".csv": return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".xlsx", ".xls"}: return pd.read_excel(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def validate_spine(spine: pd.DataFrame) -> pd.DataFrame:
    """Validate authoritative fields without deriving replacements."""
    missing = set(SPINE_COLUMNS) - set(spine)
    if missing: raise KeyError(f"Clinical episode spine lacks required columns: {sorted(missing)}")
    out = spine.copy()
    for col in ("clinical_anchor_date", "clinical_baseline_date"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    if out[KEYS].isna().any().any() or out.duplicated(KEYS).any():
        raise ValueError("Clinical spine must have one row per patient_id + clinical_episode_id")
    base = out.loc[out.is_clinical_baseline.eq(True)]
    if base.patient_id.duplicated().any(): raise ValueError("Multiple clinical baselines for a patient")
    if not base.clinical_episode_id.astype("string").eq(base.clinical_baseline_episode_id.astype("string")).all():
        raise ValueError("Clinical baseline episode mismatch")
    if not base.clinical_anchor_date.eq(base.clinical_baseline_date).all():
        raise ValueError("Clinical baseline date mismatch")
    if pd.to_numeric(out.time_since_clinical_baseline_days, errors="coerce").lt(0).any():
        raise ValueError("Negative longitudinal time")
    return out


def any_item(df: pd.DataFrame, items: tuple[str, ...]) -> pd.Series:
    cols = [c for c in items if c in df]
    return df[cols].notna().any(axis=1) if cols else pd.Series(False, index=df.index)


def score_columns(df: pd.DataFrame, spec: Instrument) -> list[str]:
    return [c for c in df if c.startswith(spec.prefix + "_")]


def resolve_episode_duplicates(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Prefer valid assessments, retain identical scores, and exclude conflicts."""
    selected: dict[tuple[Any, Any], dict[str, Any]] = {}
    conflicts, multiple = [], 0
    for spec in INSTRUMENTS:
        subset = scored.loc[any_item(scored, spec.items) & scored[KEYS].notna().all(axis=1)]
        cols = score_columns(scored, spec)
        for key, group in subset.groupby(KEYS, dropna=False, sort=False):
            multiple += int(len(group) > 1)
            valid = group.loc[group[spec.primary].notna()]
            chosen = None
            if len(valid) > 1:
                vectors = valid[list(spec.measures)].astype("string").fillna("<NA>")
                if len(vectors.drop_duplicates()) > 1:
                    conflicts.append({"patient_id": key[0], "clinical_episode_id": key[1],
                        "instrument": spec.name, "n_assessments": len(group),
                        "n_valid_assessments": len(valid), "resolution": "excluded_conflicting_valid_scores",
                        "observed_score_vectors": " | ".join(vectors.agg(";".join, axis=1).unique())})
                else: chosen = valid.iloc[0]
            elif len(valid) == 1: chosen = valid.iloc[0]
            else:
                present = [c for c in spec.items if c in group]
                chosen = group.loc[group[present].notna().sum(axis=1).idxmax()]
            record = selected.setdefault(tuple(key), {})
            record.update(({c: np.nan for c in cols} if chosen is None else chosen[cols].to_dict()))
            record[f"{spec.prefix}_duplicate_conflict"] = chosen is None
    rows = [{"patient_id": k[0], "clinical_episode_id": k[1], **v} for k, v in selected.items()]
    resolved = pd.DataFrame(rows) if rows else pd.DataFrame(columns=KEYS)
    return resolved, pd.DataFrame(conflicts), {
        "n_multiple_rows_same_patient_episode_instrument": multiple,
        "n_conflicts_same_patient_episode_instrument": len(conflicts)}


def attach_baseline_pop(episodes: pd.DataFrame, pop: pd.DataFrame | None) -> pd.DataFrame:
    """Attach a patient-invariant upstream baseline Pop; never classify it here."""
    candidates = ("baseline_pop", "baseline_pop_status", "clinical_baseline_pop_status", "pop_baseline_status")
    source = pop if pop is not None else episodes
    col = next((c for c in candidates if c in source), None)
    if col is None:
        episodes["baseline_pop"] = pd.NA
        return episodes
    if pop is None:
        episodes["baseline_pop"] = episodes[col]
        return episodes
    if "baseline_pop" in episodes:
        episodes = episodes.drop(columns="baseline_pop")
    values = source[["patient_id", col]].dropna(subset=[col]).drop_duplicates()
    if values.groupby("patient_id")[col].nunique().gt(1).any():
        raise ValueError("Upstream baseline Pop is not patient-invariant")
    return episodes.merge(values.drop_duplicates("patient_id").rename(columns={col: "baseline_pop"}),
                          on="patient_id", how="left", validate="many_to_one")


def build_episode_level(spine: pd.DataFrame, responses: pd.DataFrame, pop: pd.DataFrame | None = None):
    spine = validate_spine(spine)
    responses = responses.reset_index(drop=True).copy()
    if set(KEYS) - set(responses):
        raise KeyError("PRO responses require upstream patient_id and clinical_episode_id assignment")
    mapped = responses[KEYS].notna().all(axis=1)
    membership = responses[KEYS].merge(spine[KEYS].assign(_mapped=True), on=KEYS, how="left")._mapped.eq(True)
    included = mapped & membership
    scored = score_all_pros(responses.loc[included].copy())
    violations = pd.DataFrame(scored.attrs.get("pro_range_violations", []))
    resolved, conflicts, duplicate_qc = resolve_episode_duplicates(scored)
    episode = spine.merge(resolved, on=KEYS, how="left", validate="one_to_one")
    episode = attach_baseline_pop(episode, pop)
    for spec in INSTRUMENTS:
        raw_keys = scored.loc[any_item(scored, spec.items), KEYS].drop_duplicates().assign(_available=True)
        episode = episode.merge(raw_keys, on=KEYS, how="left", validate="one_to_one")
        episode[f"{spec.prefix}_available"] = episode.pop("_available").fillna(False).astype(bool)
        episode[f"{spec.prefix}_complete"] = episode.get(spec.primary, pd.Series(np.nan, index=episode.index)).notna()
    mapping_qc = {"n_raw_pro_rows": len(responses), "n_rows_with_patient_id": int(responses.patient_id.notna().sum()),
        "n_rows_with_episode_id": int(responses.clinical_episode_id.notna().sum()),
        "n_rows_without_episode_mapping": int((~included).sum()), **duplicate_qc}
    return episode, conflicts, violations, mapping_qc


def describe(series: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(series, errors="coerce").dropna()
    return {"n_available": len(x), "mean": x.mean(), "sd": x.std(ddof=1), "median": x.median(),
            "q1": x.quantile(.25), "q3": x.quantile(.75), "min": x.min(), "max": x.max()}


def baseline_table(episodes: pd.DataFrame) -> pd.DataFrame:
    base = episodes.loc[episodes.is_clinical_baseline.eq(True)]
    rows = []
    for spec in INSTRUMENTS:
        for measure in spec.measures:
            values = base.get(measure, pd.Series(np.nan, index=base.index)); summary = describe(values)
            n, available = len(base), summary["n_available"]
            threshold = 5.0 if measure == "esspri_total" else np.nan
            normative = "50 ± 10" if measure in {"sf36_pcs", "sf36_mcs"} else pd.NA
            numeric = pd.to_numeric(values, errors="coerce")
            rows.append({"instrument": spec.name, "measure": measure, "measure_label": LABELS[measure],
                "n_total_baseline": n, **summary, "n_missing": n-available,
                "pct_available": 100*available/n if n else np.nan, "pct_missing": 100*(n-available)/n if n else np.nan,
                "clinical_threshold": threshold,
                "n_above_threshold": int(numeric.ge(threshold).sum()) if pd.notna(threshold) else np.nan,
                "pct_above_threshold": 100*numeric.ge(threshold).sum()/available if pd.notna(threshold) and available else np.nan,
                "normative_reference": normative,
                "difference_from_normative_mean": summary["mean"]-50 if pd.notna(normative) else np.nan})
    return pd.DataFrame(rows)


def by_pop_table(episodes: pd.DataFrame) -> pd.DataFrame:
    base, rows, tests = episodes.loc[episodes.is_clinical_baseline.eq(True)], [], {}
    for spec in INSTRUMENTS:
        for measure in spec.measures:
            groups = [pd.to_numeric(base.loc[base.baseline_pop.eq(p), measure], errors="coerce").dropna() for p in ("Pop1","Pop2","Pop3")]
            usable = [g for g in groups if len(g)]
            tests[measure] = stats.kruskal(*usable).pvalue if len(usable) >= 2 else np.nan
            for pop, values in zip(("Pop1","Pop2","Pop3"), groups):
                N = int(base.baseline_pop.eq(pop).sum()); d = describe(values)
                rows.append({"baseline_pop":pop,"instrument":spec.name,"measure":measure,"N":N,**d,
                             "n_missing":N-len(values),"iqr":d["q3"]-d["q1"]})
    out = pd.DataFrame(rows); valid = pd.Series(tests).dropna()
    q = dict(zip(valid.index, multipletests(valid, method="fdr_bh")[1])) if len(valid) else {}
    out["kruskal_wallis_p_value"] = out.measure.map(tests); out["q_value"] = out.measure.map(q)
    return out


def visit_table(episodes: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for visit, group in episodes.groupby("clinical_visit_number", dropna=False):
        for spec in INSTRUMENTS:
            for measure in spec.measures:
                d=describe(group.get(measure,pd.Series(np.nan,index=group.index)))
                rows.append({"clinical_visit_number":visit,"instrument":spec.name,"measure":measure,
                    "n_patients_observed":group.patient_id.nunique(),**d,
                    "pct_available":100*d["n_available"]/len(group) if len(group) else np.nan})
    return pd.DataFrame(rows)


def availability_table(episodes: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows=[]
    for value, group in episodes.groupby(group_column, dropna=False):
        for spec in INSTRUMENTS:
            any_seen=group[f"{spec.prefix}_available"]; complete=group[f"{spec.prefix}_complete"]
            rows.append({group_column:value,"instrument":spec.name,"n_patients":group.patient_id.nunique(),
                "n_episodes":len(group),"n_with_any_item":int(any_seen.sum()),"n_with_complete_score":int(complete.sum()),
                "pct_with_any_item":100*any_seen.mean(),"pct_with_complete_score":100*complete.mean(),
                "pct_available":100*complete.mean()})
    return pd.DataFrame(rows)


def longitudinal_summary(episodes: pd.DataFrame):
    rows, diagnostics, counts = [], [], {}
    time="time_since_clinical_baseline_years"
    for spec in INSTRUMENTS:
        for measure in spec.measures:
            data=episodes[["patient_id","clinical_anchor_date",time,measure]].dropna().copy()
            patient=data.groupby("patient_id").agg(n=(measure,"size"),dates=("clinical_anchor_date","nunique"))
            ids=patient.index[(patient.n>=2)&(patient.dates>=2)]
            eligible=data.loc[data.patient_id.isin(ids)].sort_values(["patient_id","clinical_anchor_date"])
            if measure==spec.primary: counts[spec.prefix]=len(ids)
            first=eligible.groupby("patient_id").first(); last=eligible.groupby("patient_id").last()
            change=last[measure]-first[measure] if len(ids) else pd.Series(dtype=float)
            followup=last[time]-first[time] if len(ids) else pd.Series(dtype=float)
            estimate=low=high=pvalue=np.nan; model="not_fitted"; status="insufficient_data"; warning=""
            if len(ids)>=3 and eligible[time].nunique()>=2:
                try:
                    import statsmodels.formula.api as smf
                    fit=smf.mixedlm(f"{measure} ~ {time}",eligible,groups=eligible.patient_id).fit(reml=False)
                    estimate=fit.params[time]; se=fit.bse[time]; low,high=estimate-1.96*se,estimate+1.96*se
                    pvalue=fit.pvalues[time]; model,status="LME","fitted"
                except Exception as exc:
                    warning=f"LME failed: {type(exc).__name__}: {exc}"
                    try:
                        import statsmodels.api as sm
                        import statsmodels.formula.api as smf
                        fit=smf.gee(f"{measure} ~ {time}",groups="patient_id",data=eligible,family=sm.families.Gaussian()).fit()
                        estimate=fit.params[time]; se=fit.bse[time]; low,high=estimate-1.96*se,estimate+1.96*se
                        pvalue=fit.pvalues[time]; model,status="GEE_fallback","fitted_after_lme_failure"
                    except Exception as exc2: status="model_failed"; warning+=f"; GEE failed: {type(exc2).__name__}: {exc2}"
            rows.append({"instrument":spec.name,"measure":measure,"n_patients_longitudinal":len(ids),
                "n_observations":len(eligible),"n_missing":int(episodes[measure].isna().sum()),
                "median_followup_years":followup.median(),"median_measurements_per_patient":patient.loc[ids,"n"].median() if len(ids) else np.nan,
                "baseline_mean_or_median":first[measure].mean() if len(ids) else np.nan,
                "last_mean_or_median":last[measure].mean() if len(ids) else np.nan,"mean_or_median_change":change.mean(),
                "annual_change_estimate":estimate,"CI95_low":low,"CI95_high":high,"p_value":pvalue,
                "model_used":model,"model_status":status})
            diagnostics.append({"instrument":spec.name,"measure":measure,"n_patients":len(ids),"n_observations":len(eligible),
                "model_used":model,"model_status":status,"fallback_used":model=="GEE_fallback","warning":warning})
    out=pd.DataFrame(rows); out["q_value"]=np.nan; valid=out.p_value.notna()
    if valid.any(): out.loc[valid,"q_value"]=multipletests(out.loc[valid,"p_value"],method="fdr_bh")[1]
    return out,pd.DataFrame(diagnostics),counts



def create_figures(by_visit: pd.DataFrame, by_pop: pd.DataFrame) -> None:
    """Render only pre-aggregated canonical tables (no plotting-time derivation)."""
    import matplotlib.pyplot as plt
    figure_dir = Path(common.OUTPUTS_DIR) / "figures" / "blockA"
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = {"ESSPRI": "esspri_total", "SF-36": "sf36_pcs",
               "PROFAD": "profad_total", "MDAFS": "mdafs_global"}
    stems = {"ESSPRI": "esspri", "SF-36": "sf36", "PROFAD": "profad", "MDAFS": "mdafs"}
    for instrument, measure in primary.items():
        data = by_visit.loc[by_visit.measure.eq(measure)].sort_values("clinical_visit_number")
        fig, ax = plt.subplots(figsize=(7, 4))
        if len(data):
            ax.errorbar(data.clinical_visit_number, data["mean"], yerr=data.sd, marker="o")
            for row in data.itertuples():
                ax.annotate(f"n={row.n_available}", (row.clinical_visit_number, row.mean),
                            xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        ax.set(xlabel="Clinical visit number (descriptive only)", ylabel=LABELS[measure],
               title=f"{instrument} availability and trajectory")
        fig.tight_layout(); fig.savefig(figure_dir / f"09_{stems[instrument]}_trajectory.pdf"); plt.close(fig)
    data = by_pop.loc[by_pop.measure.isin(primary.values())]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (instrument, measure) in zip(axes.flat, primary.items()):
        panel = data.loc[data.measure.eq(measure)].set_index("baseline_pop").reindex(["Pop1","Pop2","Pop3"])
        ax.bar(panel.index, panel["mean"], yerr=panel.sd)
        for i, row in enumerate(panel.itertuples()):
            ax.text(i, 0 if pd.isna(row.mean) else row.mean, f"n={row.n_available}", ha="center", va="bottom", fontsize=8)
        ax.set_title(instrument); ax.set_ylabel(LABELS[measure])
    fig.tight_layout(); fig.savefig(figure_dir / "09_pros_by_pop.pdf"); plt.close(fig)

def scoring_qc(responses, episode, violations):
    rows=[]
    for spec in INSTRUMENTS:
        raw=int(any_item(responses,spec.items).sum()); complete=int(episode[f"{spec.prefix}_complete"].sum())
        partial_col={"esspri":"esspri_partial_mean","sf36":"sf36_n_domains_available","profad":"profad_n_items_available","mdafs":"mdafs_n_items_available"}[spec.prefix]
        partial=episode.get(partial_col,pd.Series(np.nan,index=episode.index)).notna()&~episode[f"{spec.prefix}_complete"]
        out_range=int(violations.instrument.eq(spec.name).sum()) if len(violations) and "instrument" in violations else 0
        rows.append({"instrument":spec.name,"n_raw_responses":raw,"n_scored":complete,"n_invalid":raw-complete,
                     "n_out_of_range":out_range,"n_partial":int(partial.sum()),"n_complete":complete})
    return pd.DataFrame(rows)


def range_violation_count(episodes):
    return sum(int((pd.to_numeric(episodes[m],errors="coerce").notna() &
                    ~pd.to_numeric(episodes[m],errors="coerce").between(lo,hi)).sum())
               for m,(lo,hi) in PRO_SCORE_RANGES.items() if m in episodes)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clinical-spine",type=Path,default=common.CLINICAL_VISIT_SPINE_PARQUET)
    p.add_argument("--responses",type=Path,default=common.CLINICAL_VISIT_SPINE_PARQUET,
                   help="PRO rows carrying upstream patient_id + clinical_episode_id")
    p.add_argument("--pop",type=Path,default=common.POP_LONGITUDINAL_PARQUET)
    p.add_argument("--overwrite",dest="overwrite",action="store_true",default=True)
    p.add_argument("--no-overwrite",dest="overwrite",action="store_false")
    return p.parse_args()


def main():
    args=parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    common.ensure_output_dirs()
    episode,conflicts,violations,mapping=build_episode_level(read_frame(args.clinical_spine),read_frame(args.responses),
                                                             read_frame(args.pop) if args.pop.exists() else None)
    if "parent_protocol" not in episode:
        episode["parent_protocol"]=pd.NA
        LOG.warning("parent_protocol absent upstream; retaining missing values without interval-text inference")
    longitudinal,model_qc,eligible=longitudinal_summary(episode)
    baseline=baseline_table(episode); by_pop=by_pop_table(episode); by_visit=visit_table(episode)
    summary={"n_patients_clinical_spine":episode.patient_id.nunique(),"n_clinical_episodes":len(episode),
        "n_pro_episode_rows":int(episode[[f"{s.prefix}_available" for s in INSTRUMENTS]].any(axis=1).sum()),
        "n_baseline_patients":int(episode.is_clinical_baseline.eq(True).sum()),
        **{f"n_{s.prefix}_baseline_available":int(episode.loc[episode.is_clinical_baseline.eq(True),f"{s.prefix}_complete"].sum()) for s in INSTRUMENTS},
        **{f"n_{s.prefix}_longitudinal_eligible":eligible.get(s.prefix,0) for s in INSTRUMENTS},
        "n_duplicate_patient_episode":int(episode.duplicated(KEYS).sum()),
        "n_multiple_baseline":int(episode.loc[episode.is_clinical_baseline.eq(True),"patient_id"].duplicated().sum()),
        "n_unmapped_pro_assessments":mapping["n_rows_without_episode_mapping"],
        "n_scoring_range_violations":range_violation_count(episode)}
    if summary["n_scoring_range_violations"]: raise ValueError("Scores outside scoring-module ranges")
    outputs={INTERMEDIATE/"09_pros_episode_level.csv":episode,TABLES/"09_pros_baseline.csv":baseline,
        TABLES/"09_pros_by_baseline_pop.csv":by_pop,TABLES/"09_pros_by_clinical_visit_number.csv":by_visit,
        TABLES/"09_pros_availability_by_clinical_visit_number.csv":availability_table(episode,"clinical_visit_number"),
        TABLES/"09_pros_availability_by_protocol.csv":availability_table(episode,"parent_protocol"),
        TABLES/"09_pros_longitudinal_change.csv":longitudinal,
        QC/"09_pros_qc_summary.csv":pd.DataFrame(summary.items(),columns=["metric","value"]),
        QC/"09_pros_scoring_qc.csv":scoring_qc(read_frame(args.responses),episode,violations),
        QC/"09_pros_episode_mapping_qc.csv":pd.DataFrame(mapping.items(),columns=["metric","value"]),
        QC/"09_pros_duplicate_conflicts.csv":conflicts,QC/"09_pros_model_qc.csv":model_qc}
    parquet=INTERMEDIATE/"09_pros_episode_level.parquet"
    if not args.overwrite and (parquet.exists() or any(p.exists() for p in outputs)): raise FileExistsError("Output exists")
    episode.to_parquet(parquet,index=False)
    for path,frame in outputs.items(): frame.to_csv(path,index=False)
    create_figures(by_visit, by_pop)
    LOG.info("Wrote %d canonical patient-episode rows",len(episode))


if __name__ == "__main__": main()

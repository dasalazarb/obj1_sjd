"""Audit BTRIS ORDER DOCUMENTATION proximity to authoritative Step 00 episodes.

Candidate-only matching, never medication exposure/active therapy/ingestion.
The clinical episodes are NOT reconstructed, edited, or redefined. The
restricted HMAC key links a BTRIS patient token to a Step 00 patient without
releasing direct patient identifiers. Every possible episode match is retained;
multiple candidates remain ambiguous rather than silently selecting one.

Primary window: [episode_start_date, episode_end_date].
Exploratory proximity window: expand episode bounds by ONE CALENDAR YEAR on each side.
Do not classify post-episode orders as prior treatment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("_step08_audit_match", HERE / "08_treatment_audit_v2.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

VERSION = "08-btris-episode-candidate-match-1.1.0"
DEFAULT_SPINE = audit.v1.common.CLINICAL_VISIT_SPINE_PARQUET
SOURCE = "medication_documentation_consolidated/08_medications_consolidated_documentation.parquet"
SOURCE_SUMMARY = "medication_documentation_consolidated/08_medications_consolidated_summary.json"

def _require(frame, required, what):
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(what + "_missing:" + ",".join(sorted(missing)))

def _strict_day(value):
    """Only actual full dates; never infer a day from year/month-only values."""
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp,)):
        return value.normalize()
    text = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return pd.NaT
    try:
        return pd.Timestamp(text)
    except (TypeError, OverflowError, ValueError):
        return pd.NaT

def build(documentation, clinical_spine, key, source_summary):
    if not isinstance(key,bytes) or len(key)<32:
        raise ValueError("missing_or_short_pseudonym_key")
    if source_summary.get("version")!="08-medication-documentation-consolidated-1.0.0":
        raise ValueError("unrecognized_consolidation_version")
    _require(documentation, [
        "documentation_record_id","source","source_kind","linked_patient_pseudonymous",
        "date_of_documentation","documentation_dated_usable","actual_exposure_confirmed",
        "clinical_episode_id"],"consolidated")
    if documentation.documentation_record_id.isna().any() or documentation.documentation_record_id.duplicated().any():
        raise ValueError("invalid_documentation_primary_key")
    if documentation.actual_exposure_confirmed.ne(False).any():
        raise ValueError("invalid_confirmed_exposure_state")
    if len(documentation) != source_summary.get("n_total_documentation_records"):
        raise ValueError("consolidation_row_count_mismatch")
    btris = documentation.loc[documentation.source_kind.eq("BTRIS_order")].copy()
    if not btris.source.isin(["11D","15D"]).all():
        raise ValueError("unexpected_btris_source")
    if len(btris)!=source_summary.get("n_btris_order_rows"):
        raise ValueError("btris_count_not_reconciled")
    if btris.clinical_episode_id.fillna("").ne("").any():
        raise ValueError("btris_already_assigned_episode")
    if btris.linked_patient_pseudonymous.isna().any() or btris.linked_patient_pseudonymous.eq("").any():
        raise ValueError("unlinked_btris_patient")
    _require(clinical_spine,[
        "patient_id","clinical_episode_id","clinical_visit",
        "episode_start_date","episode_end_date","clinical_anchor_date"],"step00_clinical_spine")
    spine=clinical_spine.copy()
    spine=spine.loc[spine.clinical_visit.eq(True)].copy()
    if spine[["patient_id","clinical_episode_id"]].isna().any().any() or (
        spine[["patient_id","clinical_episode_id"]].duplicated().any()
    ):
        raise ValueError("invalid_step00_episode_keys")
    spine["patient_token"]=spine.patient_id.map(
        lambda v:audit.pseudonym(key,"patient",str(v).strip()))
    if spine.patient_token.eq("").any() or spine.patient_token.isna().any():
        raise ValueError("invalid_patient_token")
    for col in ("episode_start_date","episode_end_date","clinical_anchor_date"):
        spine[col]=spine[col].map(_strict_day)
    if (spine[["episode_start_date","episode_end_date","clinical_anchor_date"]].isna().any().any()
        or spine.episode_start_date.gt(spine.clinical_anchor_date).any()
        or spine.clinical_anchor_date.gt(spine.episode_end_date).any()):
        raise ValueError("invalid_step00_episode_time_bounds")
    if spine[["patient_token","clinical_episode_id"]].duplicated().any():
        raise ValueError("duplicate_pseudonym_episode")
    # A pseudonym should represent exactly one raw patient, without exporting it.
    if spine.groupby("patient_token").patient_id.nunique(dropna=False).gt(1).any():
        raise ValueError("patient_token_collision")
    grouped={token:part.sort_values(["episode_start_date","clinical_episode_id"])
             for token,part in spine.groupby("patient_token",sort=False)}
    candidates=[]
    statuses=[]
    for r in btris.itertuples(index=False):
        event=_strict_day(r.date_of_documentation)
        usable=bool(r.documentation_dated_usable)
        token=r.linked_patient_pseudonymous
        pool=grouped.get(token)
        episode_matches=[]
        if usable and pd.notna(event) and pool is not None:
            for e in pool.itertuples(index=False):
                start=e.episode_start_date
                end=e.episode_end_date
                extended_start=start-pd.DateOffset(years=1)
                extended_end=end+pd.DateOffset(years=1)
                if event < extended_start or event > extended_end:
                    continue
                inside=start<=event<=end
                relation=("within_episode" if inside else
                          "before_episode" if event<start else "after_episode")
                distance=0 if inside else (
                    int((start-event).days) if event<start else int((event-end).days))
                item=dict(
                    documentation_record_id=r.documentation_record_id,
                    linked_patient_pseudonymous=token,
                    source=r.source,
                    clinical_episode_id=str(e.clinical_episode_id),
                    episode_start_date=start.date().isoformat(),
                    episode_end_date=end.date().isoformat(),
                    clinical_anchor_date=e.clinical_anchor_date.date().isoformat(),
                    btris_documentation_date=event.date().isoformat(),
                    temporal_relation=relation,
                    within_episode=inside,
                    within_one_calendar_year_proximity=not inside,
                    distance_from_episode_days=distance,
                    prior_to_episode_start=event<start,
                    post_episode_end=event>end,
                    date_evidence_kind="btris_order_or_start_date_unvalidated",
                    actual_exposure_confirmed=False,
                    match_version=VERSION,
                )
                episode_matches.append(item)
        n_inside=sum(x["within_episode"] for x in episode_matches)
        n_tolerance=len(episode_matches)-n_inside
        if pd.isna(event):
            # Undated records stay in the protected patient-level catalog;
            # missing date alone is not a pharmacological identity/QC rejection.
            status="missing_exact_order_documentation_day"
        elif not usable:
            status="blocked_by_documentation_qc"
        elif pool is None:
            status="patient_absent_from_clinical_spine"
        elif not episode_matches:
            status="no_candidate_within_year"
        elif n_inside==1:
            status="one_within_episode" if len(episode_matches)==1 else "one_within_episode_plus_tolerance"
        elif n_inside>1:
            status="multiple_within_episode"
        elif n_tolerance==1:
            status="one_tolerance_only"
        else:
            status="multiple_tolerance_only"
        candidates.extend(episode_matches)
        statuses.append(dict(
            documentation_record_id=r.documentation_record_id,
            linked_patient_pseudonymous=token,
            source=r.source,
            btris_documentation_date=event.date().isoformat() if pd.notna(event) else "",
            documentation_dated_usable=usable,
            match_status=status,
            n_candidates=len(episode_matches),
            n_within_episode=n_inside,
            n_tolerance_only=n_tolerance,
            unambiguous_best_candidate=bool(
                status in {"one_within_episode","one_tolerance_only",
                           "one_within_episode_plus_tolerance"}),
            needs_manual_temporal_review=bool(
                status in {"multiple_within_episode","multiple_tolerance_only",
                           "one_within_episode_plus_tolerance"}),
            confirmed_exposure=False,
            match_version=VERSION,
        ))
    columns=[
        "documentation_record_id","linked_patient_pseudonymous","source",
        "clinical_episode_id","episode_start_date","episode_end_date",
        "clinical_anchor_date","btris_documentation_date","temporal_relation",
        "within_episode","within_one_calendar_year_proximity",
        "distance_from_episode_days","prior_to_episode_start","post_episode_end",
        "date_evidence_kind","actual_exposure_confirmed","match_version",
    ]
    candidate_df=pd.DataFrame(candidates,columns=columns)
    status_df=pd.DataFrame(statuses)
    if len(status_df)!=len(btris) or status_df.documentation_record_id.duplicated().any():
        raise ValueError("btris_row_level_status_incomplete")
    if int(status_df.n_candidates.sum())!=len(candidate_df):
        raise ValueError("candidate_count_mismatch")
    if not candidate_df.empty and not candidate_df.apply(
        lambda row:(
            row.within_episode != row.within_one_calendar_year_proximity
            and (row.temporal_relation=="within_episode")==row.within_episode
        ),axis=1).all():
        raise ValueError("temporal_relation_inconsistent")
    breakdown=(status_df.groupby(["source","match_status"],dropna=False)
        .agg(n_order_rows=("documentation_record_id","size"),
             n_patients=("linked_patient_pseudonymous","nunique"),
             n_episode_candidates=("n_candidates","sum"),
             n_exact_candidates=("n_within_episode","sum"),
             n_tolerance_candidates=("n_tolerance_only","sum")).reset_index())
    metrics={
        "version":VERSION,
        "unit":"BTRIS_ORDER_ROW_NOT_CONFIRMED_EXPOSURE",
        "source_consolidation_version":source_summary["version"],
        "n_consolidated_documentation_records":int(len(documentation)),
        "n_btris_orders":int(len(btris)),
        "n_btris_documentation_date_usable":int(btris.documentation_dated_usable.sum()),
        "n_btris_with_one_or_more_candidates":int(status_df.n_candidates.gt(0).sum()),
        "n_btris_without_candidate":int(status_df.n_candidates.eq(0).sum()),
        "n_btris_with_multiple_candidates":int(status_df.n_candidates.gt(1).sum()),
        "n_within_episode_candidates":int(candidate_df.within_episode.sum()),
        "n_tolerance_only_candidates":int(candidate_df.within_one_calendar_year_proximity.sum()),
        "n_orders_with_at_least_one_within_episode":int(status_df.n_within_episode.gt(0).sum()),
        "n_orders_only_with_tolerance_candidates":int(
            (status_df.n_candidates.gt(0)&status_df.n_within_episode.eq(0)).sum()),
        "n_pseudonymous_patients_with_candidate":int(
            status_df.loc[status_df.n_candidates.gt(0),"linked_patient_pseudonymous"].nunique()),
        "n_step00_clinical_episodes":int(len(spine)),
        "n_step00_clinical_patients":int(spine.patient_token.nunique()),
        "status_counts":{str(k):int(v) for k,v in status_df.match_status.value_counts().items()},
        "proximity_relation_is_not_confirmed_treatment":True,
        "one_calendar_year_proximity_before_and_after":True,
        "no_automatic_episode_assignment":True,
        "no_cross_source_deduplication":True,
        "date_window":"episode_start/end plus or minus one calendar year",
        "post_episode_documentation_not_valid_as_prior_exposure":True,
        "causal_treatment_effect":"NO_GO",
        "confirmed_medication_exposure":"NO_GO",
    }
    return candidate_df,status_df,breakdown,metrics

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()

def run(audit_root,spine_path):
    root=Path(audit_root).resolve()
    restricted=root/"restricted"
    data_path=restricted/SOURCE
    summary_path=restricted/SOURCE_SUMMARY
    spine_path=Path(spine_path).resolve()
    if not all(x.is_file() for x in [data_path,summary_path,spine_path]):
        raise FileNotFoundError("consolidation_summary_and_Step00_clinical_spine_required")
    audit.reject_downstream_source(spine_path)
    key=(restricted/".pseudonym_key").read_bytes()
    source_summary=json.loads(summary_path.read_text())
    data=pd.read_parquet(data_path)
    spine=pd.read_parquet(spine_path)
    candidates,status,breakdown,metrics=build(data,spine,key,source_summary)
    metrics["inputs_sha256"]={
        "consolidated_medications":_sha(data_path),
        "consolidation_summary":_sha(summary_path),
        "clinical_visit_spine":_sha(spine_path),
    }
    out=restricted/"btris_episode_window_candidates"
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(out,0o700)
    for filename,frame in [
        ("08_btris_episode_candidate_matches.parquet",candidates),
        ("08_btris_episode_match_status.parquet",status)]:
        path=out/filename
        frame.to_parquet(path,index=False)
        os.chmod(path,0o600)
    qc_path=out/"08_btris_episode_match_qc.csv"
    breakdown.to_csv(qc_path,index=False)
    os.chmod(qc_path,0o600)
    summary_out=out/"08_btris_episode_match_summary.json"
    summary_out.write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n")
    os.chmod(summary_out,0o600)
    return metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root",type=Path,required=True)
    parser.add_argument("--clinical-spine",type=Path,default=DEFAULT_SPINE)
    args=parser.parse_args()
    print(json.dumps(run(args.audit_root,args.clinical_spine),indent=2,sort_keys=True))

if __name__=="__main__":
    main()

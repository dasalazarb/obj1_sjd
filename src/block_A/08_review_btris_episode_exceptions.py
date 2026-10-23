"""Review BTRIS episode-matching exceptions without assigning medication exposure.

Uses ONLY restricted Step 08 candidate/status Parquets, consolidated order
documentation, and Step 00 clinical spine. The 52 multi-candidate orders get a
transparent ranking (not automatic assignment). The 21 no-candidate orders
are separated into QC holds and temporal near misses. No patient identifiers,
raw names or row-level findings are written to aggregate reports.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import pandas as pd

VERSION = "08-btris-episode-exception-review-1.0.0"

def _require(frame, names, label):
    missing=set(names)-set(frame.columns)
    if missing:
        raise ValueError(label+"_missing_columns:"+",".join(sorted(missing)))

def _day(value):
    if pd.isna(value) or str(value).strip()=="":
        return pd.NaT
    value=str(value)[:10]
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return pd.NaT

def _patient_token(secret, identifier):
    value=str(identifier).strip()
    return hmac.new(secret, ("patient\0"+value).encode(), hashlib.sha256).hexdigest()

def _tokens(value):
    if pd.isna(value):
        return []
    return sorted(set(t.strip() for t in str(value).split("|") if t.strip()))

def _interval_distance(day, lo, hi):
    if lo<=day<=hi:
        return 0
    return (lo-day).days if day<lo else (day-hi).days

def build(status, candidates, medications, clinical_spine, secret, match_summary):
    if not isinstance(secret,bytes) or len(secret)<32:
        raise ValueError("invalid_protected_hmac_key")
    if match_summary.get("version")!="08-btris-episode-candidate-match-1.0.0":
        raise ValueError("wrong_upstream_matching_version")
    _require(status,["documentation_record_id","source","linked_patient_pseudonymous",
        "btris_documentation_date","match_status","n_candidates","n_within_episode"],"status")
    _require(candidates,["documentation_record_id","clinical_episode_id",
        "episode_start_date","episode_end_date","clinical_anchor_date",
        "within_episode","distance_from_episode_days"],"candidates")
    _require(medications,["documentation_record_id","source","source_kind",
        "linked_patient_pseudonymous","qc_flags","review_reasons"],"medications")
    _require(clinical_spine,["patient_id","clinical_episode_id","clinical_visit",
        "episode_start_date","episode_end_date","clinical_anchor_date"],"clinical_spine")
    if status.documentation_record_id.isna().any() or status.documentation_record_id.duplicated().any():
        raise ValueError("duplicate_status_documentation_ids")
    if len(status)!=match_summary["n_btris_orders"]:
        raise ValueError("status_total_not_reconciled")
    if len(candidates)!=int(status.n_candidates.sum()):
        raise ValueError("candidate_count_not_reconciled")
    counts=status.match_status.value_counts().to_dict()
    if counts!={k:int(v) for k,v in match_summary["status_counts"].items()}:
        raise ValueError("match_status_counts_changed")
    meds=medications.loc[medications.source_kind.eq("BTRIS_order")].copy()
    if meds.documentation_record_id.duplicated().any() or len(meds)!=len(status):
        raise ValueError("btris_medication_keys_not_reconciled")
    merged=status.merge(meds[["documentation_record_id","source",
                              "linked_patient_pseudonymous","qc_flags","review_reasons"]],
                        on="documentation_record_id",how="left",validate="one_to_one",
                        suffixes=("","_med"),indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("btris_exception_record_missing_from_consolidation")
    if not merged.source.eq(merged.source_med).all() or not merged.linked_patient_pseudonymous.eq(merged.linked_patient_pseudonymous_med).all():
        raise ValueError("source_or_patient_disagrees")
    episode=clinical_spine.loc[clinical_spine.clinical_visit.eq(True)].copy()
    if episode[["patient_id","clinical_episode_id"]].isna().any().any() or episode.duplicated(["patient_id","clinical_episode_id"]).any():
        raise ValueError("invalid_step00_spine_keys")
    episode["token"]=episode.patient_id.map(lambda x:_patient_token(secret,x))
    for k in ["episode_start_date","episode_end_date","clinical_anchor_date"]:
        episode[k]=episode[k].map(_day)
    if episode[["episode_start_date","episode_end_date","clinical_anchor_date"]].isna().any().any():
        raise ValueError("invalid_episode_dates")
    if episode.episode_start_date.gt(episode.episode_end_date).any():
        raise ValueError("reversed_episode")
    pools={k:v for k,v in episode.groupby("token")}
    candidate_pool={k:v.copy() for k,v in candidates.groupby("documentation_record_id")}
    exceptions=merged.loc[merged.n_candidates.ne(1) |
        merged.match_status.eq("one_within_episode_plus_tolerance")].copy()
    review_rows=[]
    rank_rows=[]
    reason_rows=[]
    for r in exceptions.itertuples(index=False):
        rid=r.documentation_record_id
        matches=candidate_pool.get(rid,pd.DataFrame(columns=candidates.columns))
        event=_day(r.btris_documentation_date)
        pool=pools.get(r.linked_patient_pseudonymous)
        qc=_tokens(r.qc_flags)
        review=_tokens(r.review_reasons)
        # Source flags remain separate from possible product identity concerns.
        if r.match_status=="blocked_by_documentation_qc":
            category="qc_hold_no_linkage_inferred"
        elif r.match_status=="no_candidate_within_month":
            category="outside_one_month_window"
        elif r.match_status=="patient_absent_from_clinical_spine":
            category="patient_absent"
        elif r.match_status=="missing_exact_order_documentation_day":
            category="no_order_day"
        elif r.match_status=="one_within_episode_plus_tolerance":
            category="one_direct_preferred_review_secondary"
        elif r.match_status=="multiple_within_episode":
            category="overlapping_episode_intervals_manual_review"
        elif r.match_status=="multiple_tolerance_only":
            category="multiple_month_tolerance_manual_review"
        else:
            category="other_exception"
        nearest_gap=None
        nearest_type="not_evaluated"
        if r.match_status=="no_candidate_within_month" and pd.notna(event) and pool is not None:
            gaps=[]
            for e in pool.itertuples(index=False):
                gaps.append((_interval_distance(event,e.episode_start_date,e.episode_end_date),
                    abs((event-e.clinical_anchor_date).days),
                    str(e.clinical_episode_id),
                    "before" if event<e.episode_start_date else
                    "after" if event>e.episode_end_date else "within"))
            gaps.sort()
            if gaps:
                nearest_gap=gaps[0][0]
                nearest_type=gaps[0][3]
        proposal=""
        margin=None
        anchor_tie=False
        if len(matches):
            scored=[]
            for m in matches.itertuples(index=False):
                anchor=_day(m.clinical_anchor_date)
                anchor_gap=abs((event-anchor).days) if pd.notna(event) and pd.notna(anchor) else None
                priority=0 if bool(m.within_episode) else 1
                boundary=int(m.distance_from_episode_days)
                scored.append((priority,boundary,anchor_gap,str(m.clinical_episode_id),m))
            scored.sort(key=lambda x:(x[0],x[1],x[2] if x[2] is not None else 999999,x[3]))
            best=scored[0]
            if len(scored)>1:
                second=scored[1]
                if best[:3]==second[:3]:
                    anchor_tie=True
                elif best[0]==second[0] and best[1]==second[1] and best[2] is not None and second[2] is not None:
                    margin=second[2]-best[2]
            for rank,entry in enumerate(scored,1):
                m=entry[4]
                rank_rows.append({
                    "documentation_record_id":rid,
                    "linked_patient_pseudonymous":r.linked_patient_pseudonymous,
                    "match_status":r.match_status,
                    "candidate_rank_descriptive":rank,
                    "clinical_episode_id":m.clinical_episode_id,
                    "order_date":r.btris_documentation_date,
                    "episode_start_date":m.episode_start_date,
                    "episode_end_date":m.episode_end_date,
                    "clinical_anchor_date":m.clinical_anchor_date,
                    "within_episode":bool(m.within_episode),
                    "distance_from_episode_days":int(m.distance_from_episode_days),
                    "anchor_distance_days":entry[2],
                    "ranking_only_not_assignment":True,
                })
            if r.match_status=="one_within_episode_plus_tolerance":
                proposal="unique_direct_candidate_other_tolerance_retained"
            elif r.match_status=="multiple_within_episode":
                proposal="unique_nearest_anchor_descriptive_only" if not anchor_tie else "anchor_distance_tie"
            elif r.match_status=="multiple_tolerance_only":
                proposal="unique_nearest_boundary_descriptive_only" if not anchor_tie else "distance_tie"
        if r.match_status=="blocked_by_documentation_qc":
            for flag in qc or ["unspecified_qc_block"]:
                reason_rows.append({"source":r.source,"match_status":r.match_status,
                                    "reason_type":"qc_flag","reason":flag})
            for flag in review:
                reason_rows.append({"source":r.source,"match_status":r.match_status,
                                    "reason_type":"identity_review","reason":flag})
        review_rows.append({
            "documentation_record_id":rid,
            "linked_patient_pseudonymous":r.linked_patient_pseudonymous,
            "source":r.source,
            "match_status":r.match_status,
            "review_category":category,
            "n_episode_candidates":len(matches),
            "n_within_episode":int(r.n_within_episode),
            "descriptive_ranking":proposal,
            "anchor_distance_tie":anchor_tie,
            "nearest_anchor_margin_days":margin,
            "nearest_episode_gap_days":nearest_gap,
            "nearest_episode_direction":nearest_type,
            "qc_flags":r.qc_flags,
            "classification_review_reasons":r.review_reasons,
            "manual_review_required":True,
            "automatic_episode_assignment":False,
            "confirmed_medication_exposure":False,
        })
    review_df=pd.DataFrame(review_rows)
    rank_df=pd.DataFrame(rank_rows)
    reason_df=pd.DataFrame(reason_rows,columns=["source","match_status","reason_type","reason"])
    # Every ambiguous or without-candidate order must be represented exactly
    # once; the lone "direct + tolerance" order is already in the 52.
    expected=merged.loc[(merged.n_candidates.gt(1))|(merged.n_candidates.eq(0))]
    if len(review_df)!=len(expected) or set(review_df.documentation_record_id)!=set(expected.documentation_record_id):
        raise ValueError("exception_review_coverage_failed")
    groups=(review_df.groupby(["source","match_status","review_category","descriptive_ranking"],
                              dropna=False)
        .agg(n_order_rows=("documentation_record_id","size"),
             n_nearest_episode_gap_known=("nearest_episode_gap_days","count"),
             n_anchor_ties=("anchor_distance_tie","sum"))
        .reset_index())
    reason_counts=(reason_df.groupby(["source","match_status","reason_type","reason"],dropna=False)
                   .size().rename("n_flag_occurrences").reset_index())
    nearest=(review_df.loc[review_df.match_status.eq("no_candidate_within_month"),
                          ["nearest_episode_gap_days","nearest_episode_direction"]].copy())
    gaps=nearest.nearest_episode_gap_days.dropna()
    summary={
        "version":VERSION,"upstream_version":match_summary["version"],
        "n_orders_total":int(len(status)),
        "n_review_orders":int(len(review_df)),
        "n_multi_candidate_orders":int(status.n_candidates.gt(1).sum()),
        "n_zero_candidate_orders":int(status.n_candidates.eq(0).sum()),
        "n_qc_blocked":int(review_df.match_status.eq("blocked_by_documentation_qc").sum()),
        "n_no_candidate_within_month":int(review_df.match_status.eq("no_candidate_within_month").sum()),
        "n_one_direct_plus_tolerance":int(review_df.match_status.eq("one_within_episode_plus_tolerance").sum()),
        "n_anchor_ranking_ties":int(review_df.anchor_distance_tie.sum()),
        "n_near_misses_with_known_gap":int(gaps.size),
        "nearest_episode_gap_days_min":int(gaps.min()) if gaps.size else None,
        "nearest_episode_gap_days_median":float(gaps.median()) if gaps.size else None,
        "nearest_episode_gap_days_max":int(gaps.max()) if gaps.size else None,
        "group_counts":{str(k):int(v) for k,v in review_df.review_category.value_counts().items()},
        "all_original_candidates_retained":True,
        "automatic_episode_assignment":False,
        "changed_episode_boundaries":False,
        "date_tolerance_policy_unchanged":True,
        "confirmed_medication_exposure":"NO_GO",
        "documentation_temporal_review":"PROTECTED_ONLY"
    }
    return review_df,rank_df,groups,reason_counts,summary

def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()

def run(audit_root,spine_path):
    root=Path(audit_root).resolve()
    protected=root/"restricted"
    paths={
        "match_status":protected/"btris_episode_window_candidates/08_btris_episode_match_status.parquet",
        "matches":protected/"btris_episode_window_candidates/08_btris_episode_candidate_matches.parquet",
        "match_summary":protected/"btris_episode_window_candidates/08_btris_episode_match_summary.json",
        "medications":protected/"medication_documentation_consolidated/08_medications_consolidated_documentation.parquet",
        "spine":Path(spine_path).resolve(),
    }
    if any(not p.is_file() for p in paths.values()):
        raise FileNotFoundError("required_restricted_Step08_and_Step00_inputs_missing")
    secret=(protected/".pseudonym_key").read_bytes()
    result=build(pd.read_parquet(paths["match_status"]),pd.read_parquet(paths["matches"]),
        pd.read_parquet(paths["medications"]),pd.read_parquet(paths["spine"]),secret,
        json.loads(paths["match_summary"].read_text()))
    review,ranked,groups,reasons,summary=result
    summary["input_fingerprints"]={k:_sha(p) for k,p in paths.items()}
    out=protected/"btris_episode_exception_review"
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(out,0o700)
    for name,frame in [
        ("08_btris_exception_order_review.parquet",review),
        ("08_btris_exception_candidate_ranking.parquet",ranked)
    ]:
        dest=out/name
        frame.to_parquet(dest,index=False)
        os.chmod(dest,0o600)
    for name,frame in [
        ("08_btris_exception_review_qc.csv",groups),
        ("08_btris_exception_blocking_reasons.csv",reasons)
    ]:
        dest=out/name
        frame.to_csv(dest,index=False)
        os.chmod(dest,0o600)
    dest=out/"08_btris_exception_review_summary.json"
    dest.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    os.chmod(dest,0o600)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root",type=Path,required=True)
    parser.add_argument("--clinical-spine",type=Path,
                        default=Path("data/intermediate/00_clinical_visit_spine.parquet"))
    args=parser.parse_args()
    print(json.dumps(run(args.audit_root,args.clinical_spine),indent=2,sort_keys=True))

if __name__=="__main__":
    main()

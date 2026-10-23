"""Restricted provenance audit: BTRIS non-text MRNs and overlapping Step 00 episodes.

Reports QC; does not verify numeric-MRN identity, lift any blocker, choose an
episode for a BTRIS order, or infer drug administration/exposure.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
import json
import os
from itertools import combinations
from pathlib import Path
import pandas as pd

VERSION = "08-btris-exception-provenance-1.0.0"

def require(frame, names, label):
    missing=sorted(set(names)-set(frame.columns))
    if missing:
        raise ValueError(label+"_missing_columns:"+",".join(missing))

def flags(value):
    return set() if pd.isna(value) else set(filter(None,(x.strip() for x in str(value).split("|"))))

def day(value):
    if pd.isna(value):
        return pd.NaT
    try:
        return pd.Timestamp(value).normalize()
    except (TypeError, ValueError, OverflowError):
        return pd.NaT

def patient_token(secret, identifier):
    return hmac.new(secret,("patient\0"+str(identifier).strip()).encode(),hashlib.sha256).hexdigest()

def build(meds,audit,status,matches,exceptions,spine,secret,summary):
    if not isinstance(secret,bytes) or len(secret)<32:
        raise ValueError("invalid_pseudonym_key")
    if summary.get("version")!="08-btris-episode-candidate-match-1.0.0":
        raise ValueError("unexpected_previous_match_version")
    require(meds,["documentation_record_id","source","source_kind",
        "source_audit_record_id","linked_patient_pseudonymous","qc_flags",
        "review_reasons"],"medications")
    require(audit,["audit_record_id","source","source_row",
        "source_patient_pseudonymous","linked_patient_pseudonymous",
        "mrn_identity_quality","linkage_status","qc_flags",
        "eligible_linked"],"original_BTRIS_audit")
    require(status,["documentation_record_id","match_status","n_candidates",
        "source","linked_patient_pseudonymous"],"match_status")
    require(matches,["documentation_record_id","clinical_episode_id",
        "within_episode","episode_start_date","episode_end_date",
        "clinical_anchor_date"],"episode_candidates")
    require(exceptions,["documentation_record_id","match_status"],"exceptions")
    require(spine,["patient_id","clinical_episode_id","clinical_visit",
        "episode_start_date","episode_end_date","clinical_anchor_date"],"step00")
    for label,frame,column in [
        ("medications",meds,"documentation_record_id"),
        ("BTRIS_audit",audit,"audit_record_id"),
        ("status",status,"documentation_record_id"),
        ("exceptions",exceptions,"documentation_record_id")
    ]:
        if frame[column].isna().any() or frame[column].duplicated().any():
            raise ValueError(label+"_missing_or_duplicate_record_id")
    if len(status)!=summary["n_btris_orders"] or len(matches)!=int(status.n_candidates.sum()):
        raise ValueError("candidate_status_count_mismatch")
    if (status.n_candidates.gt(1).sum()!=summary["n_btris_with_multiple_candidates"]
        or status.n_candidates.eq(0).sum()!=summary["n_btris_without_candidate"]):
        raise ValueError("exception_counts_changed")
    expected=status.loc[status.n_candidates.ne(1)]
    if set(expected.documentation_record_id)!=set(exceptions.documentation_record_id):
        raise ValueError("exception_record_set_mismatch")
    source=meds.loc[meds.source_kind.eq("BTRIS_order")].copy()
    orig=audit.loc[audit.source.isin(["11D","15D"])].copy()
    if source.source_audit_record_id.isna().any() or source.source_audit_record_id.duplicated().any():
        raise ValueError("BTRIS_source_record_keys_invalid")
    joined=source.merge(orig[[
        "audit_record_id","source","source_row","source_patient_pseudonymous",
        "linked_patient_pseudonymous","mrn_identity_quality","linkage_status",
        "qc_flags","eligible_linked"
    ]],left_on="source_audit_record_id",right_on="audit_record_id",
      how="left",validate="one_to_one",suffixes=("","_audit"),indicator=True)
    if not joined._merge.eq("both").all() or not joined.source.eq(joined.source_audit).all():
        raise ValueError("BTRIS_original_audit_join_failed")
    if not joined.linked_patient_pseudonymous.eq(joined.linked_patient_pseudonymous_audit).all():
        raise ValueError("BTRIS_original_vs_released_patient_token_mismatch")
    t2p=joined.groupby("source_patient_pseudonymous").linked_patient_pseudonymous.nunique(dropna=True)
    p2t=joined.groupby("linked_patient_pseudonymous").source_patient_pseudonymous.nunique(dropna=True)
    blocked_ids=set(expected.loc[
        expected.match_status.eq("blocked_by_documentation_qc"),"documentation_record_id"])
    blocked=joined.loc[joined.documentation_record_id.isin(blocked_ids)].copy()
    if len(blocked)!=len(blocked_ids):
        raise ValueError("QC_hold_provenance_join_incomplete")
    ident=[]
    for r in blocked.itertuples(index=False):
        combined=flags(r.qc_flags)|flags(r.qc_flags_audit)
        ident.append(dict(
            documentation_record_id=r.documentation_record_id,
            patient_token=r.linked_patient_pseudonymous,
            original_source_mrn_token=r.source_patient_pseudonymous,
            source=r.source,source_row=r.source_row,
            mrn_identity_quality=r.mrn_identity_quality,
            audit_linkage_status=r.linkage_status,
            audit_eligible_linked=bool(r.eligible_linked),
            n_patient_tokens_per_source_mrn=int(t2p.get(r.source_patient_pseudonymous,0)),
            n_mrn_tokens_per_patient=int(p2t.get(r.linked_patient_pseudonymous,0)),
            nontext_normalization_warning=(
                "nontext_mrn_normalized_match_requires_identifier_review" in combined),
            original_mrn_independently_verified=False,
            source_mrn_text_exact=(r.mrn_identity_quality=="text_exact"),
            qc_flags="|".join(sorted(combined)),
            identity_review_reasons=r.review_reasons,
            episode_assignment_authorized=False,confirmed_exposure=False))
    identity=pd.DataFrame(ident)
    identity_qc=identity.groupby(
        ["source","mrn_identity_quality","audit_linkage_status",
         "nontext_normalization_warning"],dropna=False).agg(
        n_orders=("documentation_record_id","size"),
        n_patients=("patient_token","nunique"),
        n_source_mrn_collision_orders=("n_patient_tokens_per_source_mrn",
            lambda x:int(x.gt(1).sum())),
        n_multiple_mrn_token_orders=("n_mrn_tokens_per_patient",
            lambda x:int(x.gt(1).sum()))).reset_index()

    spine=spine.loc[spine.clinical_visit.eq(True)].copy()
    spine["patient_token"]=spine.patient_id.map(lambda v:patient_token(secret,v))
    for col in ("episode_start_date","episode_end_date","clinical_anchor_date"):
        spine[col]=spine[col].map(day)
    if (spine[["patient_token","clinical_episode_id"]].duplicated().any()
        or spine[["episode_start_date","episode_end_date","clinical_anchor_date"]].isna().any().any()):
        raise ValueError("Step00_spine_episode_integrity_failed")
    lookup=spine.set_index(["patient_token","clinical_episode_id"])
    multiple=status.loc[status.n_candidates.gt(1)].copy()
    candidate_map={k:v for k,v in matches.loc[
        matches.documentation_record_id.isin(multiple.documentation_record_id)
    ].groupby("documentation_record_id")}
    pair_rows=[]
    for r in multiple.itertuples(index=False):
        subset=candidate_map.get(r.documentation_record_id)
        if subset is None or len(subset)!=r.n_candidates or subset.clinical_episode_id.duplicated().any():
            raise ValueError("candidate_episode_set_incomplete")
        for l,q in combinations(subset.itertuples(index=False),2):
            values=[]
            for candidate in (l,q):
                k=(r.linked_patient_pseudonymous,candidate.clinical_episode_id)
                if k not in lookup.index:
                    raise ValueError("candidate_not_in_Step00")
                v=lookup.loc[k]
                if (day(candidate.episode_start_date)!=v.episode_start_date
                    or day(candidate.episode_end_date)!=v.episode_end_date):
                    raise ValueError("candidate_bounds_disagree_with_Step00")
                values.append(v)
            a,b=values
            intersection_start=max(a.episode_start_date,b.episode_start_date)
            intersection_end=min(a.episode_end_date,b.episode_end_date)
            overlap=max(0,int((intersection_end-intersection_start).days)+1)
            if a.episode_start_date==b.episode_start_date and a.episode_end_date==b.episode_end_date:
                shape="identical_bounds"
            elif not overlap:
                shape="disjoint_with_tolerance"
            elif ((a.episode_start_date<=b.episode_start_date and a.episode_end_date>=b.episode_end_date)
                or (b.episode_start_date<=a.episode_start_date and b.episode_end_date>=a.episode_end_date)):
                shape="nested_intervals"
            else:
                shape="partially_overlapping_intervals"
            pair_id=sorted([str(l.clinical_episode_id),str(q.clinical_episode_id)])
            pair_token=hmac.new(secret,("episode_pair\0"+r.linked_patient_pseudonymous+
                    "\0"+"\0".join(pair_id)).encode(),hashlib.sha256).hexdigest()
            pair_rows.append(dict(
                documentation_record_id=r.documentation_record_id,
                patient_token=r.linked_patient_pseudonymous,
                episode_pair_token=pair_token,
                episode_id_left=pair_id[0],episode_id_right=pair_id[1],
                match_status=r.match_status,
                both_within_episode=bool(l.within_episode and q.within_episode),
                one_within_episode=bool(l.within_episode != q.within_episode),
                episode_geometry=shape,overlap_days=overlap,
                identical_anchor_dates=bool(a.clinical_anchor_date==b.clinical_anchor_date),
                automatic_episode_assignment=False,confirmed_exposure=False))
    pairs=pd.DataFrame(pair_rows)
    if pairs.empty and len(multiple):
        raise ValueError("multiple_episode_pairs_missing")
    per_pair=pairs.groupby("episode_pair_token",dropna=False).agg(
        n_affected_orders=("documentation_record_id","nunique"),
        n_patient_tokens=("patient_token","nunique"),
        geometry=("episode_geometry","first"),
        overlap_days=("overlap_days","first"),
        same_anchors=("identical_anchor_dates","first")).reset_index()
    per_pair["manual_structure_review"]=True
    per_pair["automatic_assignment"]=False
    overlap_qc=pairs.groupby(
        ["match_status","episode_geometry","identical_anchor_dates"],dropna=False).agg(
        n_order_pair_rows=("documentation_record_id","size"),
        n_distinct_orders=("documentation_record_id","nunique"),
        n_distinct_episode_pairs=("episode_pair_token","nunique"),
        n_pseudonymous_patients=("patient_token","nunique")).reset_index()
    result=dict(
        version=VERSION,
        n_qc_held_orders=int(len(identity)),
        n_qc_held_nontext_mrn=int(identity.mrn_identity_quality.ne("text_exact").sum()),
        n_qc_held_normalization_warnings=int(identity.nontext_normalization_warning.sum()),
        n_qc_held_possible_source_mrn_collisions=int(identity.n_patient_tokens_per_source_mrn.gt(1).sum()),
        n_qc_held_multiple_mrn_tokens=int(identity.n_mrn_tokens_per_patient.gt(1).sum()),
        n_multicandidate_orders=int(len(multiple)),
        n_distinct_episode_pairs=int(len(per_pair)),
        n_identical_interval_pairs=int(per_pair.geometry.eq("identical_bounds").sum()),
        n_nested_interval_pairs=int(per_pair.geometry.eq("nested_intervals").sum()),
        n_partial_interval_pairs=int(per_pair.geometry.eq("partially_overlapping_intervals").sum()),
        n_disjoint_tolerance_pairs=int(per_pair.geometry.eq("disjoint_with_tolerance").sum()),
        max_orders_per_pair=int(per_pair.n_affected_orders.max()),
        n_source_patient_identifiers_independently_verified=0,
        no_qc_blocks_lifted=True,
        no_automatic_episode_assignment=True,
        external_original_identifier_confirmation_required=True,
        actual_exposure="NO_GO")
    return identity,pairs,per_pair,identity_qc,overlap_qc,result

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def run(audit_root,spine_path):
    root=Path(audit_root).resolve()
    restricted=root/"restricted"
    paths=dict(
        meds=restricted/"medication_documentation_consolidated/08_medications_consolidated_documentation.parquet",
        audit=restricted/"08_btris_orders_audit_long.parquet",
        status=restricted/"btris_episode_window_candidates/08_btris_episode_match_status.parquet",
        matches=restricted/"btris_episode_window_candidates/08_btris_episode_candidate_matches.parquet",
        exceptions=restricted/"btris_episode_exception_review/08_btris_exception_order_review.parquet",
        match_summary=restricted/"btris_episode_window_candidates/08_btris_episode_match_summary.json",
        spine=Path(spine_path).resolve())
    if not all(p.is_file() for p in paths.values()):
        raise FileNotFoundError("missing_Step08_or_Step00_provenance_input")
    secret=(restricted/".pseudonym_key").read_bytes()
    identity,pairs,per_pair,identity_qc,overlap_qc,result=build(
        pd.read_parquet(paths["meds"]),pd.read_parquet(paths["audit"]),
        pd.read_parquet(paths["status"]),pd.read_parquet(paths["matches"]),
        pd.read_parquet(paths["exceptions"]),pd.read_parquet(paths["spine"]),
        secret,json.loads(paths["match_summary"].read_text()))
    result["input_fingerprints"]={k:sha256(v) for k,v in paths.items()}
    out=restricted/"btris_exception_provenance"
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(out,0o700)
    for name,frame in [
        ("08_btris_qc_hold_identity_review.parquet",identity),
        ("08_btris_overlap_episode_pairs.parquet",pairs),
        ("08_btris_episode_pair_burden.parquet",per_pair)]:
        path=out/name
        frame.to_parquet(path,index=False)
        os.chmod(path,0o600)
    for name,frame in [
        ("08_btris_qc_hold_identity_qc.csv",identity_qc),
        ("08_btris_overlap_geometry_qc.csv",overlap_qc)]:
        path=out/name
        frame.to_csv(path,index=False)
        os.chmod(path,0o600)
    path=out/"08_btris_exception_provenance_summary.json"
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    os.chmod(path,0o600)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root",type=Path,required=True)
    parser.add_argument("--clinical-spine",type=Path,
        default=Path("data/intermediate/00_clinical_visit_spine.parquet"))
    args=parser.parse_args()
    print(json.dumps(run(args.audit_root,args.clinical_spine),indent=2,sort_keys=True))

if __name__=="__main__":
    main()

"""Prepare protected manual-review packets for unresolved BTRIS source identifiers
and overlapping authoritative clinical episodes. No identity adjudication,
episode reassignment, source-data modification, or treatment-exposure inference.

The resulting review templates contain pseudonymous IDs only. Human approvals
MUST be evidenced independently; this script does not accept or apply them.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import pandas as pd

VERSION = "08-btris-manual-review-packets-1.0.0"

def _require(frame,fields,source):
    missing=sorted(set(fields)-set(frame.columns))
    if missing:
        raise ValueError(f"{source}_required_fields_missing:{','.join(missing)}")

def _token(secret,patient):
    if pd.isna(patient) or str(patient).strip()=="":
        raise ValueError("step00_missing_patient")
    return hmac.new(secret,("patient\0"+str(patient).strip()).encode(),
                    hashlib.sha256).hexdigest()

def _day(value):
    if pd.isna(value):
        return pd.NaT
    try:
        return pd.Timestamp(value).normalize()
    except (TypeError,ValueError,OverflowError):
        return pd.NaT

def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()

def build(burden,episode_pairs,identity,spine,secret,upstream_summary):
    if not isinstance(secret,bytes) or len(secret)<32:
        raise ValueError("pseudonym_key_invalid")
    if upstream_summary.get("version")!="08-btris-exception-provenance-1.0.0":
        raise ValueError("unexpected_source_provenance_version")
    _require(burden,["episode_pair_token","n_affected_orders","geometry",
        "overlap_days","same_anchors"],"episode_burden")
    _require(episode_pairs,["episode_pair_token","patient_token","episode_id_left",
        "episode_id_right","documentation_record_id"],"episode_pairs")
    _require(identity,["documentation_record_id","patient_token","source",
        "source_row","mrn_identity_quality","original_source_mrn_token",
        "nontext_normalization_warning","original_mrn_independently_verified",
        "qc_flags","identity_review_reasons"],"identity")
    _require(spine,["patient_id","clinical_episode_id","clinical_visit",
        "episode_start_date","episode_end_date","clinical_anchor_date",
        "visit_type","clinical_visit_number","is_clinical_baseline"],"step00")
    if identity.documentation_record_id.duplicated().any():
        raise ValueError("identity_review_duplicate_order")
    if burden.episode_pair_token.duplicated().any():
        raise ValueError("episode_burden_duplicate_pair")
    if (len(burden)!=upstream_summary["n_distinct_episode_pairs"]
        or len(identity)!=upstream_summary["n_qc_held_orders"]):
        raise ValueError("source_review_counts_changed")
    lookup=spine.loc[spine.clinical_visit.eq(True)].copy()
    if (lookup[["patient_id","clinical_episode_id"]].isna().any().any()
        or lookup.duplicated(["patient_id","clinical_episode_id"]).any()):
        raise ValueError("step00_patient_episode_keys_invalid")
    lookup["patient_token"]=lookup.patient_id.map(lambda x:_token(secret,x))
    lookup["clinical_episode_id"]=lookup.clinical_episode_id.astype(str)
    lookup=lookup.set_index(["patient_token","clinical_episode_id"])
    # Pair metadata are derived from original candidate rows. Every pair must
    # belong to one pseudonymous patient and have exactly one episode-ID pair.
    pairing=[]
    for pid,part in episode_pairs.groupby("episode_pair_token",sort=False):
        tokens=part.patient_token.dropna().unique()
        left=part.episode_id_left.dropna().astype(str).unique()
        right=part.episode_id_right.dropna().astype(str).unique()
        if len(tokens)!=len(left)!=1:
            pass
        if len(tokens)!=1 or len(left)!=1 or len(right)!=1:
            raise ValueError("pair_patient_or_episode_ids_inconsistent")
        if left[0]==right[0]:
            raise ValueError("identical_episode_ids_in_pair")
        pair=burden.loc[burden.episode_pair_token.eq(pid)]
        if len(pair)!=1:
            raise ValueError("pair_not_present_in_burden")
        if part.documentation_record_id.nunique()!=int(pair.iloc[0].n_affected_orders):
            raise ValueError("pair_affected_order_count_disagrees")
        enriched=[]
        for eid in [left[0],right[0]]:
            key=(tokens[0],eid)
            if key not in lookup.index:
                raise ValueError("episode_pair_not_present_in_step00")
            src=lookup.loc[key]
            if isinstance(src,pd.DataFrame):
                raise ValueError("duplicate_spine_key")
            start,end,anchor=(_day(src[k]) for k in
                ["episode_start_date","episode_end_date","clinical_anchor_date"])
            if any(pd.isna(v) for v in [start,end,anchor]):
                raise ValueError("episode_dates_missing")
            if not(start<=anchor<=end):
                raise ValueError("episode_anchor_not_within_episode")
            enriched.append(dict(
                episode_id=eid,
                start_date=start.date().isoformat(),
                end_date=end.date().isoformat(),
                anchor_date=anchor.date().isoformat(),
                visit_type=str(src.visit_type),
                clinical_visit_number=src.clinical_visit_number,
                is_baseline=bool(src.is_clinical_baseline)))
        a,b=enriched
        if (a["anchor_date"]==b["anchor_date"])!=bool(pair.iloc[0].same_anchors):
            raise ValueError("anchor_equality_disagrees_with_provenance")
        pairing.append(dict(
            episode_pair_token=pid,
            patient_token=tokens[0],
            n_affected_orders=int(pair.iloc[0].n_affected_orders),
            geometry=str(pair.iloc[0].geometry),
            overlap_days=int(pair.iloc[0].overlap_days),
            identical_anchors=bool(pair.iloc[0].same_anchors),
            clinical_visit_number_equal=bool(
                pd.notna(a["clinical_visit_number"])
                and pd.notna(b["clinical_visit_number"])
                and a["clinical_visit_number"]==b["clinical_visit_number"]),
            same_visit_type=a["visit_type"]==b["visit_type"],
            left_episode_id=a["episode_id"],right_episode_id=b["episode_id"],
            left_episode_start=a["start_date"],left_episode_end=a["end_date"],
            right_episode_start=b["start_date"],right_episode_end=b["end_date"],
            left_anchor=a["anchor_date"],right_anchor=b["anchor_date"],
            left_visit_type=a["visit_type"],right_visit_type=b["visit_type"],
            left_visit_number=a["clinical_visit_number"],
            right_visit_number=b["clinical_visit_number"],
            left_baseline=a["is_baseline"],right_baseline=b["is_baseline"],
            review_question="same_clinical_event_or_distinct_events_check_EDA_manifest_and_source_rows",
            review_status="pending_external_episode_provenance_check",
            adjudicated_episode_id="", adjudication_evidence_reference="",
            no_automatic_episode_selection=True))
    pairs=pd.DataFrame(pairing)
    if len(pairs)!=len(burden):
        raise ValueError("not_all_episode_pairs_accounted")
    ids=identity.copy()
    if ids.original_mrn_independently_verified.eq(True).any():
        raise ValueError("unexpected_already_verified_identifiers")
    ids["review_status"]="pending_original_source_text_MRN_verification"
    ids["source_field_to_verify"]="BTRIS MRN raw storage type and exact textual source"
    ids["review_evidence_reference"]=""
    ids["independent_identifier_verified"]=False
    ids["authorization_to_release_QC"]=False
    ids["episode_assignment_authorized"]=False
    ids["confirmed_medication_exposure"]=False
    pair_qc=(pairs.groupby(
        ["geometry","identical_anchors","clinical_visit_number_equal",
         "same_visit_type"],dropna=False)
        .agg(n_episode_pairs=("episode_pair_token","size"),
             n_order_mentions_across_pairs=("n_affected_orders","sum"))
        .reset_index())
    identity_qc=(ids.groupby(["source","mrn_identity_quality","review_status"],
        dropna=False).agg(
            n_orders=("documentation_record_id","size"),
            n_pseudonymous_patients=("patient_token","nunique")).reset_index())
    summary=dict(
        version=VERSION,
        source_version=upstream_summary["version"],
        n_episode_pairs_for_review=int(len(pairs)),
        n_order_mentions_across_pairs=int(pairs.n_affected_orders.sum()),
        n_pairs_same_anchor=int(pairs.identical_anchors.sum()),
        n_pairs_same_visit_number=int(pairs.clinical_visit_number_equal.sum()),
        n_pairs_same_visit_type=int(pairs.same_visit_type.sum()),
        n_pairs_both_baseline=int((pairs.left_baseline & pairs.right_baseline).sum()),
        n_identifier_orders_for_review=int(len(ids)),
        n_identifier_patients_for_review=int(ids.patient_token.nunique()),
        n_identifier_source_tokens_for_review=int(ids.original_source_mrn_token.nunique()),
        n_identifier_independently_verified=0,
        n_episodes_reassigned=0,
        no_change_to_step00=True,no_change_to_medication_data=True,
        confirmed_medication_exposure="NO_GO",
        manual_review_requirements=[
            "Inspect upstream EDA clinical episode manifest and raw row map for each pair.",
            "Verify original BTRIS text MRNs from source owner/crosswalk in authorized environment.",
            "Do not use anchor distance to claim unique visit or treatment exposure.",
            "Document provenance and approval outside automatic analysis before releasing holds.",
        ])
    return pairs,ids,pair_qc,identity_qc,summary

def run(audit_root,spine_path):
    root=Path(audit_root).resolve()
    protected=root/"restricted"
    paths=dict(
        episode_burden=protected/"btris_exception_provenance/08_btris_episode_pair_burden.parquet",
        episode_pairs=protected/"btris_exception_provenance/08_btris_overlap_episode_pairs.parquet",
        identity=protected/"btris_exception_provenance/08_btris_qc_hold_identity_review.parquet",
        provenance_summary=protected/"btris_exception_provenance/08_btris_exception_provenance_summary.json",
        step00_spine=Path(spine_path).resolve())
    if any(not p.is_file() for p in paths.values()):
        raise FileNotFoundError("required_Step08_provenance_or_Step00_input_missing")
    secret=(protected/".pseudonym_key").read_bytes()
    outputs=build(
        pd.read_parquet(paths["episode_burden"]),
        pd.read_parquet(paths["episode_pairs"]),
        pd.read_parquet(paths["identity"]),
        pd.read_parquet(paths["step00_spine"]),
        secret,json.loads(paths["provenance_summary"].read_text()))
    pairs,ids,pair_qc,identity_qc,summary=outputs
    summary["input_sha256"]={k:_sha256(v) for k,v in paths.items()}
    dest=protected/"btris_manual_review_packets"
    dest.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(dest,0o700)
    for name,frame in [
        ("08_btris_episode_pair_review.parquet",pairs),
        ("08_btris_MRN_source_verification_review.parquet",ids)]:
        path=dest/name
        frame.to_parquet(path,index=False)
        os.chmod(path,0o600)
    for name,frame in [
        ("08_btris_episode_pair_context_qc.csv",pair_qc),
        ("08_btris_MRN_review_qc.csv",identity_qc)]:
        path=dest/name
        frame.to_csv(path,index=False)
        os.chmod(path,0o600)
    path=dest/"08_btris_review_packets_summary.json"
    path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    os.chmod(path,0o600)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-root",type=Path,required=True)
    p.add_argument("--clinical-spine",type=Path,
        default=Path("data/intermediate/00_clinical_visit_spine.parquet"))
    args=p.parse_args()
    print(json.dumps(run(args.audit_root,args.clinical_spine),indent=2,sort_keys=True))

if __name__=="__main__":
    main()

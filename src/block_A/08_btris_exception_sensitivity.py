"""Sensitivity audit of Step 08 BTRIS temporal exceptions.

No automatic clinical-episode assignment and no exposure inference.
Recheck *time only* for previously QC-blocked BTRIS order records without
changing any identity/ingredient/identifier QC flags. Original matching
outputs, medication releases and Step 00 are immutable.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location(
    "med_order_episode_match",HERE/"08_match_btris_episode_windows.py")
matcher=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matcher)

VERSION="08-btris-exception-sensitivity-1.0.0"

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def build(medications,spine,key,consolidated_summary,original_status,original_candidates,
          original_match_summary,exception_review):
    if original_match_summary.get("version")!="08-btris-episode-candidate-match-1.0.0":
        raise ValueError("unexpected_prior_match_version")
    if original_match_summary.get("n_btris_orders") != len(original_status):
        raise ValueError("prior_status_row_count_mismatch")
    if len(original_candidates) != int(original_status.n_candidates.sum()):
        raise ValueError("prior_candidate_count_mismatch")
    if original_status.documentation_record_id.duplicated().any():
        raise ValueError("prior_status_duplicate_key")
    if len(exception_review) != int(original_match_summary["n_btris_with_multiple_candidates"] +
                                   original_match_summary["n_btris_without_candidate"]):
        raise ValueError("exception_review_count_mismatch")
    if exception_review.documentation_record_id.duplicated().any():
        raise ValueError("exception_review_duplicate_key")
    exception_ids=set(exception_review.documentation_record_id)
    expected_ids=set(original_status.loc[
        original_status.n_candidates.ne(1),"documentation_record_id"])
    if exception_ids != expected_ids:
        raise ValueError("exception_review_does_not_cover_original_exceptions")
    blocked=original_status.loc[
        original_status.match_status.eq("blocked_by_documentation_qc"),"documentation_record_id"]
    if blocked.empty:
        raise ValueError("no_qc_blocked_orders_to_review")
    flagged=medications.loc[medications.documentation_record_id.isin(blocked)].copy()
    if len(flagged)!=len(blocked) or not flagged.source_kind.eq("BTRIS_order").all():
        raise ValueError("blocked_orders_not_reconciled")
    if flagged.documentation_dated_usable.ne(False).any():
        raise ValueError("expected_qc_hold_not_preserved")
    # Never change the saved medication record. This in-memory copy is ONLY
    # a diagnostic of order-date proximity, not an eligibility reclassification.
    temporary=medications.copy()
    mask=temporary.documentation_record_id.isin(blocked)
    temporary.loc[mask,"documentation_dated_usable"]=True
    alternative,alternative_status,_,_=matcher.build(
        temporary,spine,key,consolidated_summary)
    original=original_status.set_index("documentation_record_id")
    revised=alternative_status.set_index("documentation_record_id")
    unchanged=[x for x in revised.index if x not in set(blocked)]
    if not original.loc[unchanged,["match_status","n_candidates","n_within_episode"]].equals(
        revised.loc[unchanged,["match_status","n_candidates","n_within_episode"]]):
        raise ValueError("nonblocked_matching_changed")
    # Rows below are exploratory. QC and patient ID warnings remain unchanged,
    # and the original blocked records remain blocked in the canonical release.
    rechecked=alternative_status.loc[
        alternative_status.documentation_record_id.isin(blocked)].copy()
    risk=flagged[["documentation_record_id","qc_flags","review_reasons",
                  "classification_confidence","date_of_documentation"]].copy()
    rechecked=rechecked.merge(
        risk,on="documentation_record_id",validate="one_to_one")
    rechecked["original_match_status"]="blocked_by_documentation_qc"
    rechecked["temporal_sensitivity_only"]=True
    rechecked["identity_qc_not_overridden"]=True
    rechecked["patient_identifier_review_required"]=rechecked.qc_flags.fillna("").str.contains(
        "nontext_mrn_normalized_match_requires_identifier_review",regex=False)
    rechecked["manual_review_required"]=True
    rechecked["episode_assignment_authorized"]=False
    recandidates=alternative.loc[
        alternative.documentation_record_id.isin(blocked)].copy()
    recandidates["temporal_sensitivity_only"]=True
    recandidates["episode_assignment_authorized"]=False
    if len(recandidates) != int(rechecked.n_candidates.sum()):
        raise ValueError("sensitivity_candidate_count_mismatch")
    original_exceptions=exception_review.copy()
    direct_plus=original_exceptions.match_status.eq("one_within_episode_plus_tolerance")
    overlaps=original_exceptions.match_status.eq("multiple_within_episode")
    ties=original_exceptions.anchor_distance_tie.fillna(False)
    other_tolerance=original_exceptions.match_status.eq("multiple_tolerance_only")
    beyond=original_exceptions.match_status.eq("no_candidate_within_month")
    original_exceptions["triage_decision"]="manual_review_no_episode_assignment"
    original_exceptions.loc[direct_plus,"triage_decision"]=(
        "direct_candidate_preferred_for_descriptive_linkage_only")
    original_exceptions.loc[overlaps & ~ties,"triage_decision"]=(
        "nearest_anchor_descriptive_rank_not_proof")
    original_exceptions.loc[overlaps & ties,"triage_decision"]=(
        "true_anchor_distance_tie_keep_all_candidates")
    original_exceptions.loc[other_tolerance,"triage_decision"]=(
        "tolerance_only_multiple_keep_all_candidates")
    original_exceptions.loc[beyond,"triage_decision"]=(
        "outside_predefined_month_window_no_link")
    original_exceptions.loc[
        original_exceptions.match_status.eq("blocked_by_documentation_qc"),
        "triage_decision"]="qc_hold_separate_temporal_sensitivity_available"
    original_exceptions["episode_assignment_authorized"]=False
    original_exceptions["confirmed_medication_exposure"]=False
    # Aggregate only; no patient-level information outside protected Parquet.
    status_breakdown=rechecked.groupby(
        ["source","match_status","patient_identifier_review_required"],
        dropna=False).agg(
            n_orders=("documentation_record_id","size"),
            n_candidate_episode_rows=("n_candidates","sum"),
            n_orders_with_direct_candidate=("n_within_episode",
                lambda values:int(values.gt(0).sum()))).reset_index()
    decision_breakdown=original_exceptions.groupby(
        ["source","match_status","triage_decision"],dropna=False
        ).size().rename("n_orders").reset_index()
    if len(original_exceptions)!=len(exception_review):
        raise ValueError("exception_preservation_failure")
    metrics={
        "version":VERSION,
        "n_original_exception_orders":int(len(original_exceptions)),
        "n_multi_candidate_orders":int(original_exceptions.n_episode_candidates.gt(1).sum()),
        "n_prior_zero_candidate_orders":int(
            original_exceptions.n_episode_candidates.eq(0).sum()),
        "n_direct_plus_tolerance":int(direct_plus.sum()),
        "n_overlapping_episode_orders":int(overlaps.sum()),
        "n_nearest_anchor_distinct_descriptive":int((overlaps&~ties).sum()),
        "n_nearest_anchor_ties":int((overlaps&ties).sum()),
        "n_multiple_tolerance_only":int(other_tolerance.sum()),
        "n_beyond_one_month_window":int(beyond.sum()),
        "n_qc_hold_orders_rechecked_time_only":int(len(rechecked)),
        "n_qc_hold_with_temporal_candidate":int(rechecked.n_candidates.gt(0).sum()),
        "n_qc_hold_without_temporal_candidate":int(rechecked.n_candidates.eq(0).sum()),
        "n_qc_hold_with_multiple_candidates":int(rechecked.n_candidates.gt(1).sum()),
        "n_qc_hold_patient_identifier_review":int(
            rechecked.patient_identifier_review_required.sum()),
        "n_qc_hold_date_missing":int(
            rechecked.date_of_documentation.fillna("").eq("").sum()),
        "sensitivity_temporal_candidates":int(len(recandidates)),
        "no_qc_gates_released":True,
        "no_episode_assignments":True,
        "no_tolerance_extension":True,
        "confirmed_medication_exposure":"NO_GO",
        "analysis_scope":"documentation_temporal_sensitivity_PROTECTED_ONLY",
    }
    if metrics["n_multi_candidate_orders"]+metrics["n_prior_zero_candidate_orders"] != len(original_exceptions):
        raise ValueError("exception_partition_inconsistent")
    return original_exceptions,rechecked,recandidates,decision_breakdown,status_breakdown,metrics

def run(audit_root,spine_path):
    root=Path(audit_root).resolve()
    protected=root/"restricted"
    paths={
        "medications":protected/"medication_documentation_consolidated/08_medications_consolidated_documentation.parquet",
        "consolidated_summary":protected/"medication_documentation_consolidated/08_medications_consolidated_summary.json",
        "candidate_matches":protected/"btris_episode_window_candidates/08_btris_episode_candidate_matches.parquet",
        "match_status":protected/"btris_episode_window_candidates/08_btris_episode_match_status.parquet",
        "match_summary":protected/"btris_episode_window_candidates/08_btris_episode_match_summary.json",
        "exception_review":protected/"btris_episode_exception_review/08_btris_exception_order_review.parquet",
        "clinical_spine":Path(spine_path).resolve(),
    }
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError("missing_required_input:"+p.name)
    matcher.audit.reject_downstream_source(paths["clinical_spine"])
    secret=(protected/".pseudonym_key").read_bytes()
    results=build(
        pd.read_parquet(paths["medications"]),
        pd.read_parquet(paths["clinical_spine"]),
        secret,json.loads(paths["consolidated_summary"].read_text()),
        pd.read_parquet(paths["match_status"]),
        pd.read_parquet(paths["candidate_matches"]),
        json.loads(paths["match_summary"].read_text()),
        pd.read_parquet(paths["exception_review"]),
    )
    review,blocked,candidates,triage_qc,blocked_qc,summary=results
    summary["input_sha256"]={label:_sha(path) for label,path in paths.items()}
    output=protected/"btris_exception_sensitivity"
    output.mkdir(exist_ok=True,parents=True,mode=0o700)
    os.chmod(output,0o700)
    for filename,frame in [
        ("08_btris_exceptions_triage.parquet",review),
        ("08_btris_blocked_temporal_review.parquet",blocked),
        ("08_btris_blocked_temporal_candidates.parquet",candidates)]:
        p=output/filename
        frame.to_parquet(p,index=False)
        os.chmod(p,0o600)
    for filename,frame in [
        ("08_btris_exceptions_triage_qc.csv",triage_qc),
        ("08_btris_blocked_temporal_qc.csv",blocked_qc)]:
        p=output/filename
        frame.to_csv(p,index=False)
        os.chmod(p,0o600)
    p=output/"08_btris_exception_sensitivity_summary.json"
    p.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    os.chmod(p,0o600)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root",required=True,type=Path)
    parser.add_argument("--clinical-spine",type=Path,
        default=Path("data/intermediate/00_clinical_visit_spine.parquet"))
    args=parser.parse_args()
    print(json.dumps(run(args.audit_root,args.clinical_spine),indent=2,sort_keys=True))
if __name__=="__main__":
    main()

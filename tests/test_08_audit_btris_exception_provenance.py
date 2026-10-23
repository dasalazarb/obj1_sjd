import importlib.util
from pathlib import Path
import pandas as pd
import pytest

p=Path(__file__).resolve().parents[1]/"src/block_A/08_audit_btris_exception_provenance.py"
spec=importlib.util.spec_from_file_location("provenance",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
KEY=b"synthetic-32-byte-key-for-tests-only"

def inputs():
    token=m.patient_token(KEY,"p1")
    meds=pd.DataFrame([dict(documentation_record_id="B1",source="11D",source_kind="BTRIS_order",
        source_audit_record_id="11D:0",linked_patient_pseudonymous=token,qc_flags="",
        review_reasons="ingredient_unresolved"),
        dict(documentation_record_id="B2",source="11D",source_kind="BTRIS_order",
        source_audit_record_id="11D:1",linked_patient_pseudonymous=token,qc_flags="",
        review_reasons="")])
    audit=pd.DataFrame([dict(audit_record_id=f"11D:{i}",source="11D",source_row=i,
        source_patient_pseudonymous=f"mrn{i}",linked_patient_pseudonymous=token,
        mrn_identity_quality="nontext_unverified" if i==0 else "text_exact",
        linkage_status="linked_one_to_one",qc_flags="",eligible_linked=True) for i in range(2)])
    status=pd.DataFrame([dict(documentation_record_id="B1",match_status="blocked_by_documentation_qc",
        n_candidates=0,source="11D",linked_patient_pseudonymous=token),
        dict(documentation_record_id="B2",match_status="multiple_within_episode",
        n_candidates=2,source="11D",linked_patient_pseudonymous=token)])
    matches=pd.DataFrame([dict(documentation_record_id="B2",clinical_episode_id=e,
        within_episode=True,episode_start_date=start,episode_end_date=end,
        clinical_anchor_date=anchor) for e,start,end,anchor in [
        ("e1","2021-01-01","2021-04-01","2021-02-01"),
        ("e2","2021-02-01","2021-03-01","2021-02-15")]])
    exceptions=status[["documentation_record_id","match_status"]].copy()
    spine=pd.DataFrame([dict(patient_id="p1",clinical_episode_id=e,clinical_visit=True,
        episode_start_date=start,episode_end_date=end,clinical_anchor_date=anchor)
        for e,start,end,anchor in [
        ("e1","2021-01-01","2021-04-01","2021-02-01"),
        ("e2","2021-02-01","2021-03-01","2021-02-15")]])
    summary=dict(version="08-btris-episode-candidate-match-1.0.0",
        n_btris_orders=2,n_btris_with_multiple_candidates=1,n_btris_without_candidate=1)
    return meds,audit,status,matches,exceptions,spine,KEY,summary

def test_identity_and_overlap_stay_unresolved():
    identity,pairs,per_pair,identity_qc,overlap_qc,summary=m.build(*inputs())
    assert len(identity)==1 and len(pairs)==1
    assert summary["n_qc_held_nontext_mrn"]==1
    assert summary["n_nested_interval_pairs"]==1
    assert not identity.episode_assignment_authorized.any()
    assert summary["external_original_identifier_confirmation_required"]

def test_audit_patient_token_mismatch_is_blocked():
    data=list(inputs());data[1].loc[0,"linked_patient_pseudonymous"]="different"
    with pytest.raises(ValueError,match="patient_token_mismatch"):
        m.build(*data)

def test_candidate_episode_bounds_mismatch_is_blocked():
    data=list(inputs());data[5].loc[0,"episode_end_date"]="2021-05-01"
    with pytest.raises(ValueError,match="bounds_disagree"):
        m.build(*data)

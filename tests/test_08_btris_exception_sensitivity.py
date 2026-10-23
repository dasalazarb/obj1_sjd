import importlib.util
from pathlib import Path
import pandas as pd

module=Path(__file__).resolve().parents[1]/"src/block_A/08_btris_exception_sensitivity.py"
spec=importlib.util.spec_from_file_location("sensitivity",module)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_qc_hold_reevaluated_without_changing_original():
    key=b"x"*32
    token=m.matcher.audit.pseudonym(key,"patient","p1")
    meds=pd.DataFrame([
        dict(documentation_record_id="B1",source="11D",source_kind="BTRIS_order",
        linked_patient_pseudonymous=token,date_of_documentation="2021-02-03",
        documentation_dated_usable=False,actual_exposure_confirmed=False,
        clinical_episode_id="",qc_flags="nontext_mrn_normalized_match_requires_identifier_review",
        review_reasons="ingredient_unresolved",classification_confidence="candidate_review")])
    spine=pd.DataFrame([dict(patient_id="p1",clinical_episode_id="e1",clinical_visit=True,
        episode_start_date="2021-02-01",episode_end_date="2021-02-28",clinical_anchor_date="2021-02-15")])
    original_status=pd.DataFrame([dict(
        documentation_record_id="B1",linked_patient_pseudonymous=token,source="11D",
        match_status="blocked_by_documentation_qc",n_candidates=0,n_within_episode=0,
        btris_documentation_date="2021-02-03")])
    original_candidates=pd.DataFrame(columns=["documentation_record_id"])
    exception_review=pd.DataFrame([dict(documentation_record_id="B1",
        match_status="blocked_by_documentation_qc",source="11D",
        linked_patient_pseudonymous=token,n_episode_candidates=0,
        anchor_distance_tie=False)])
    source_summary=dict(version="08-medication-documentation-consolidated-1.0.0",
        n_total_documentation_records=1,n_btris_order_rows=1)
    match_summary=dict(version="08-btris-episode-candidate-match-1.0.0",
        n_btris_orders=1,n_btris_with_multiple_candidates=0,n_btris_without_candidate=1)
    results=m.build(meds,spine,key,source_summary,original_status,original_candidates,
                    match_summary,exception_review)
    review,blocked,candidates,triage,blocked_qc,summary=results
    assert summary["n_qc_hold_with_temporal_candidate"]==1
    assert summary["no_qc_gates_released"]
    assert not blocked.episode_assignment_authorized.any()
    assert meds.documentation_dated_usable.eq(False).all()

def test_required_classification_confidence_reports_clear_schema_error():
    import pytest
    key=b"x"*32
    token=m.matcher.audit.pseudonym(key,"patient","p1")
    meds=pd.DataFrame([
        dict(documentation_record_id="B1",source="11D",source_kind="BTRIS_order",
             linked_patient_pseudonymous=token,date_of_documentation="2021-02-03",
             documentation_dated_usable=False,actual_exposure_confirmed=False,
             clinical_episode_id="",qc_flags="ingredient_product_conflict",
             review_reasons="ingredient_unresolved")])
    spine=pd.DataFrame([dict(patient_id="p1",clinical_episode_id="e1",
        clinical_visit=True,episode_start_date="2021-02-01",
        episode_end_date="2021-02-28",clinical_anchor_date="2021-02-15")])
    status=pd.DataFrame([dict(documentation_record_id="B1",
        linked_patient_pseudonymous=token,source="11D",
        match_status="blocked_by_documentation_qc",n_candidates=0,n_within_episode=0,
        btris_documentation_date="2021-02-03")])
    review=pd.DataFrame([dict(documentation_record_id="B1",
        match_status="blocked_by_documentation_qc",source="11D",
        linked_patient_pseudonymous=token,n_episode_candidates=0,
        anchor_distance_tie=False)])
    with pytest.raises(ValueError,match="missing_required_medication_columns:classification_confidence"):
        m.build(meds,spine,key,
            dict(version="08-medication-documentation-consolidated-1.0.0",
                 n_total_documentation_records=1,n_btris_order_rows=1),
            status,pd.DataFrame(columns=["documentation_record_id"]),
            dict(version="08-btris-episode-candidate-match-1.0.0",
                 n_btris_orders=1,n_btris_with_multiple_candidates=0,
                 n_btris_without_candidate=1),review)

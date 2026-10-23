import importlib.util
from pathlib import Path
import pandas as pd

p=Path(__file__).resolve().parents[1]/"src/block_A/medication_diagnostics/08_review_btris_episode_exceptions.py"
spec=importlib.util.spec_from_file_location("review",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_ambiguous_and_unmatched_are_retained():
    key=b"x"*32
    token=m._patient_token(key,"p1")
    spine=pd.DataFrame([
        dict(patient_id="p1",clinical_episode_id="e1",clinical_visit=True,
             episode_start_date="2021-01-01",episode_end_date="2021-03-01",
             clinical_anchor_date="2021-01-15"),
        dict(patient_id="p1",clinical_episode_id="e2",clinical_visit=True,
             episode_start_date="2021-02-01",episode_end_date="2021-04-01",
             clinical_anchor_date="2021-03-15")])
    status=pd.DataFrame([
        dict(documentation_record_id="B1",source="11D",linked_patient_pseudonymous=token,
             btris_documentation_date="2021-02-15",match_status="multiple_within_episode",
             n_candidates=2,n_within_episode=2),
        dict(documentation_record_id="B2",source="11D",linked_patient_pseudonymous=token,
             btris_documentation_date="2021-07-01",match_status="no_candidate_within_month",
             n_candidates=0,n_within_episode=0)])
    candidates=pd.DataFrame([
        dict(documentation_record_id="B1",clinical_episode_id="e1",
             episode_start_date="2021-01-01",episode_end_date="2021-03-01",
             clinical_anchor_date="2021-01-15",within_episode=True,distance_from_episode_days=0),
        dict(documentation_record_id="B1",clinical_episode_id="e2",
             episode_start_date="2021-02-01",episode_end_date="2021-04-01",
             clinical_anchor_date="2021-03-15",within_episode=True,distance_from_episode_days=0)])
    meds=pd.DataFrame([
        dict(documentation_record_id=k,source="11D",source_kind="BTRIS_order",
             linked_patient_pseudonymous=token,qc_flags="",review_reasons="")
        for k in ["B1","B2"]])
    summary=dict(version="08-btris-episode-candidate-match-1.0.0",
                 n_btris_orders=2,status_counts=dict(
                 multiple_within_episode=1,no_candidate_within_month=1))
    review,ranked,aggregate,reasons,metrics=m.build(status,candidates,meds,spine,key,summary)
    assert len(review)==2
    assert len(ranked)==2
    assert metrics["n_multi_candidate_orders"]==1
    assert metrics["n_zero_candidate_orders"]==1
    assert metrics["n_near_misses_with_known_gap"]==1
    assert not review.automatic_episode_assignment.any()

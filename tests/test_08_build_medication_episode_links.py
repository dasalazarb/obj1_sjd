"""Synthetic checks for many-to-many medication documentation linkage."""
import importlib.util
from pathlib import Path
import pandas as pd
import pytest

p=Path(__file__).resolve().parents[1]/"src/block_A/08_build_medication_episode_links.py"
spec=importlib.util.spec_from_file_location("medlinks",p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
KEY=b"x"*32

def sample():
    patient=m.token(KEY,"p1")
    meds=pd.DataFrame([
        dict(documentation_record_id="B1",source="11D",source_kind="BTRIS_order",
             linked_patient_pseudonymous=patient,date_of_documentation="2021-02-15",
             clinical_episode_id="",documentation_dated_usable=True,
             date_within_authoritative_episode=pd.NA,actual_exposure_confirmed=False),
        dict(documentation_record_id="B2",source="15D",source_kind="BTRIS_order",
             linked_patient_pseudonymous=patient,date_of_documentation="2021-04-14",
             clinical_episode_id="",documentation_dated_usable=True,
             date_within_authoritative_episode=pd.NA,actual_exposure_confirmed=False),
        dict(documentation_record_id="B3",source="11D",source_kind="BTRIS_order",
             linked_patient_pseudonymous=patient,date_of_documentation="2021-02-15",
             clinical_episode_id="",documentation_dated_usable=False,
             date_within_authoritative_episode=pd.NA,actual_exposure_confirmed=False),
        dict(documentation_record_id="C1",source="CTDB_original_visit",
             source_kind="CTDB_reported_current",
             linked_patient_pseudonymous=patient,date_of_documentation="2021-02-15",
             clinical_episode_id="e1",documentation_dated_usable=True,
             date_within_authoritative_episode=True,actual_exposure_confirmed=False),
    ])
    episodes=pd.DataFrame([
        dict(patient_id="p1",clinical_episode_id="e1",clinical_visit=True,
             episode_start_date="2021-01-01",episode_end_date="2021-03-01",
             clinical_anchor_date="2021-02-01"),
        dict(patient_id="p1",clinical_episode_id="e2",clinical_visit=True,
             episode_start_date="2021-02-01",episode_end_date="2021-03-31",
             clinical_anchor_date="2021-02-15")])
    matches=pd.DataFrame([
        dict(documentation_record_id="B1",linked_patient_pseudonymous=patient,
             source="11D",clinical_episode_id=e,within_episode=True,
             within_one_calendar_month_tolerance=False,
             episode_start_date=start,episode_end_date=end,
             clinical_anchor_date=anchor)
        for e,start,end,anchor in [
            ("e1","2021-01-01","2021-03-01","2021-02-01"),
            ("e2","2021-02-01","2021-03-31","2021-02-15")]
    ]+[dict(documentation_record_id="B2",linked_patient_pseudonymous=patient,
             source="15D",clinical_episode_id="e2",within_episode=False,
             within_one_calendar_month_tolerance=True,
             episode_start_date="2021-02-01",episode_end_date="2021-03-31",
             clinical_anchor_date="2021-02-15")])
    status=pd.DataFrame([
        dict(documentation_record_id="B1",linked_patient_pseudonymous=patient,
             source="11D",match_status="multiple_within_episode",n_candidates=2,
             n_within_episode=2,n_tolerance_only=0),
        dict(documentation_record_id="B2",linked_patient_pseudonymous=patient,
             source="15D",match_status="one_tolerance_only",n_candidates=1,
             n_within_episode=0,n_tolerance_only=1),
        dict(documentation_record_id="B3",linked_patient_pseudonymous=patient,
             source="11D",match_status="blocked_by_documentation_qc",n_candidates=0,
             n_within_episode=0,n_tolerance_only=0)])
    cs={"version":"08-medication-documentation-consolidated-1.0.0",
        "n_total_documentation_records":4,"n_btris_order_rows":3,
        "n_ctdb_original_visit_mentions":1}
    ms={"version":"08-btris-episode-candidate-match-1.0.0",
        "n_btris_orders":3,"n_btris_with_multiple_candidates":1}
    return meds,matches,status,episodes,episodes.copy(),KEY,cs,ms

def test_one_order_to_two_episodes_without_duplicate_source_order():
    primary,tolerance,record_status,_,summary=m.build(*sample())
    assert len(primary)==3 # two direct BTRIS + one CTDB
    assert len(tolerance)==1
    assert summary["n_BTRIS_orders_with_multiple_direct_relations"]==1
    assert summary["n_BTRIS_within_episode_relations"]==2
    assert summary["n_BTRIS_QC_blocked_still_unlinked"]==1
    assert len(record_status)==4
    assert record_status.loc[record_status.documentation_record_id.eq("B1"),
                             "n_primary_episode_relations"].iloc[0]==2
    assert primary.loc[primary.documentation_record_id.eq("B1"),
                       "clinical_episode_id"].tolist()==["e1","e2"]
    assert not primary.confirmed_medication_exposure.any()

def test_never_use_wrong_patient_episode():
    x=list(sample())
    x[1].loc[0,"linked_patient_pseudonymous"]="wrong"
    with pytest.raises(ValueError,match="patient_or_source_mismatch"):
        m.build(*x)

def test_spine_date_change_is_blocked():
    x=list(sample())
    x[3].loc[0,"episode_end_date"]="2021-02-20"
    with pytest.raises(ValueError,match="clinical_spine_does_not_match"):
        m.build(*x)

def test_bad_tolerance_cannot_be_primary():
    x=list(sample())
    x[1].loc[2,"within_episode"]=True
    x[1].loc[2,"within_one_calendar_month_tolerance"]=False
    with pytest.raises(ValueError,match="temporal_classification_incorrect"):
        m.build(*x)

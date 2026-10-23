import importlib.util
from pathlib import Path
import pandas as pd

p=Path(__file__).resolve().parents[1]/"src/block_A/08_match_btris_episode_windows.py"
spec=importlib.util.spec_from_file_location("match_btris",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

KEY=b"x"*32

def sample():
    token=m.audit.pseudonym(KEY,"patient","p1")
    med=pd.DataFrame([
        dict(documentation_record_id="BTRIS:1",source="11D",source_kind="BTRIS_order",
             linked_patient_pseudonymous=token,date_of_documentation="2021-02-03",
             documentation_dated_usable=True,actual_exposure_confirmed=False,clinical_episode_id=""),
        dict(documentation_record_id="BTRIS:2",source="15D",source_kind="BTRIS_order",
             linked_patient_pseudonymous=token,date_of_documentation="2021-03-12",
             documentation_dated_usable=True,actual_exposure_confirmed=False,clinical_episode_id=""),
        dict(documentation_record_id="CTDB:1",source="CTDB_original_visit",
             source_kind="CTDB_reported_current",linked_patient_pseudonymous=token,
             date_of_documentation="2021-02-03",documentation_dated_usable=True,
             actual_exposure_confirmed=False,clinical_episode_id="e1")
    ])
    spine=pd.DataFrame([dict(patient_id="p1",clinical_episode_id="e1",clinical_visit=True,
        episode_start_date="2021-02-01",episode_end_date="2021-02-28",clinical_anchor_date="2021-02-15")])
    summary=dict(version="08-medication-documentation-consolidated-1.0.0",
                 n_total_documentation_records=3,n_btris_order_rows=2)
    return med,spine,summary

def test_exact_and_month_window_distinguished():
    med,spine,summary=sample()
    candidates,status,qc,result=m.build(med,spine,KEY,summary)
    assert len(candidates)==2
    assert result["n_btris_orders"]==2
    assert result["n_orders_with_at_least_one_within_episode"]==1
    assert result["n_orders_only_with_tolerance_candidates"]==1
    assert set(status.match_status)=={"one_within_episode","one_tolerance_only"}
    assert not candidates.actual_exposure_confirmed.any()

def test_post_episode_is_not_prior_exposure():
    med,spine,summary=sample()
    candidates,_,_,_=m.build(med,spine,KEY,summary)
    later=candidates[candidates.btris_documentation_date.eq("2021-03-12")].iloc[0]
    assert later.temporal_relation=="after_episode"
    assert later.post_episode_end
    assert not later.prior_to_episode_start

def test_patient_token_without_matching_spine_is_unmatched():
    med,spine,summary=sample()
    spine.loc[0,"patient_id"]="other"
    candidates,status,_,result=m.build(med,spine,KEY,summary)
    assert candidates.empty
    assert result["n_btris_without_candidate"]==2
    assert set(status.match_status)=={"patient_absent_from_clinical_spine"}

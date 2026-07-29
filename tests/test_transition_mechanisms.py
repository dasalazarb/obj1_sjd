"""Focused synthetic tests for transition mechanism derivations."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SPEC=importlib.util.spec_from_file_location("transition_mechanisms",Path("src/block_B/24_transition_mechanisms.py"))
TM=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(TM)


def base():
    return pd.DataFrame({"patient_id":["a"]*3,"episode_id":["e1","e2","e3"],"from_pop":["Pop2","Pop1",None],"to_pop":["Pop1","Pop3","Pop2"],"pop_transition_evaluable":[True,True,False]})


def test_transition_families():
    got=TM.classify_transition_family(base())
    assert got.transition_family.tolist()==["to_pop1","from_pop1","unevaluable"]


def test_incident_resolved_and_multiple_candidate_domains():
    data=TM.classify_transition_family(base().iloc[:2].assign(eg_articular_active_transition=["incident","resolved"],eg_renal_active_transition=["incident","persistent_inactive"]))
    got=TM.derive_candidate_drivers(data)
    assert got.loc[0,"candidate_driver_class"]=="multiple_domains"
    assert got.loc[0,"candidate_driver_domains"]=="articular|renal"
    assert got.loc[1,"candidate_driver_domains"]=="articular"
    assert got.loc[1,"candidate_driver_class"]=="single_domain"


def test_dominant_component_rules():
    x=pd.DataFrame({"delta_esspri_dryness":[3,2,0,np.nan],"delta_esspri_fatigue":[1,-2,0,1],"delta_esspri_pain":[0,1,0,1]})
    got=TM.derive_dominant_component(x)
    assert got.dominant_esspri_component.tolist()==["dryness","tie","no_change","not_evaluable"]
    assert got.loc[0,"dominant_esspri_component_delta"]==3


def test_first_to_last_one_row_per_patient():
    d=pd.DataFrame({"patient_id":["a","a","b"],"episode_id":["1","2","3"],"from_visit_id":["a1","a2","b1"],"to_visit_id":["a2","a3","b2"],
        "from_date":pd.to_datetime(["2020-01-01","2021-01-01","2020-06-01"]),"to_date":pd.to_datetime(["2021-01-01","2022-01-01","2021-06-01"]),"from_pop":["Pop2","Pop1","Pop3"],"to_pop":["Pop1","Pop1","Pop2"]})
    got=TM.build_first_to_last(d)
    assert len(got)==got.patient_id.nunique()==2
    assert got.set_index("patient_id").loc["a","first_to_last_transition"]=="Pop2 -> Pop1"


def test_missing_serology_is_not_negative():
    d=TM.classify_transition_family(base().iloc[:2].assign(anti_ro_pos=[True,np.nan],anti_ro_interpretable=[True,False]))
    got=TM.summarize_serology_profiles(d)
    assert got.n_interpretable.sum()==1
    assert got.n_positive.sum()==1


def test_sparse_models_are_skipped():
    d=TM.classify_transition_family(base())
    got=TM.fit_profile_models(d,min_events=2)
    assert len(got)==2
    assert got.model_status.str.startswith("skipped").all()

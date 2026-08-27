import importlib
import numpy as np
import pandas as pd
import pytest

pros = importlib.import_module("src.block_A.09_pros_longitudinal")


def spine():
    return pd.DataFrame({
        "patient_id":["p1","p1","p2"], "clinical_episode_id":["e1","e2","e3"],
        "clinical_anchor_date":pd.to_datetime(["2020-01-01","2021-01-01","2020-02-01"]),
        "clinical_visit_number":[1,2,1], "is_clinical_baseline":[True,False,True],
        "clinical_baseline_episode_id":["e1","e1","e3"],
        "clinical_baseline_date":pd.to_datetime(["2020-01-01","2020-01-01","2020-02-01"]),
        "time_since_clinical_baseline_days":[0,366,0],
        "time_since_clinical_baseline_years":[0,366/365.25,0], "parent_protocol":["11D","11D","15D"]})


def esspri(pid, eid, values):
    return {"patient_id":pid,"clinical_episode_id":eid,
        "esspri_questionnaire__dryness":values[0],"esspri_questionnaire__fatigue":values[1],
        "esspri_questionnaire__pain":values[2]}


def test_baseline_is_authoritative_episode_subset_and_timing_is_preserved():
    episode, conflicts, _, mapping = pros.build_episode_level(spine(), pd.DataFrame([
        esspri("p1","e1",(3,4,5)), esspri("p1","e2",(4,5,6))]))
    assert len(episode) == 3 and not episode.duplicated(pros.KEYS).any()
    assert pros.baseline_table(episode).query("measure == 'esspri_total'").iloc[0].n_available == 1
    assert episode.loc[episode.clinical_episode_id.eq("e2"), "time_since_clinical_baseline_days"].iat[0] == 366
    assert conflicts.empty and mapping["n_rows_without_episode_mapping"] == 0


def test_conflicting_valid_duplicate_scores_are_not_averaged():
    responses=pd.DataFrame([esspri("p1","e1",(1,2,3)),esspri("p1","e1",(7,8,9))])
    episode, conflicts, _, mapping=pros.build_episode_level(spine(),responses)
    assert np.isnan(episode.loc[episode.clinical_episode_id.eq("e1"),"esspri_total"].iat[0])
    assert len(conflicts)==1
    assert mapping["n_conflicts_same_patient_episode_instrument"]==1


def test_spine_hard_checks_reject_reconstructed_or_inconsistent_baseline():
    bad=spine(); bad.loc[0,"clinical_baseline_episode_id"]="other"
    with pytest.raises(ValueError,match="episode mismatch"):
        pros.validate_spine(bad)

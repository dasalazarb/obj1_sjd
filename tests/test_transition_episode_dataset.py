"""Focused tests for the canonical transition episode builder."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "src/block_B/23_build_transition_episode_dataset.py"
SPEC = importlib.util.spec_from_file_location("transition_episodes", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def visits(pop=("Pop2", "Pop1", "Unclassifiable")):
    dates = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-02-10"])
    return pd.DataFrame({
        "patient_id": ["p1"] * 3, "visit_id": ["v0", "v1", "v2"],
        "visit_date": dates, "visit_number": [0, 1, 2],
        "is_observed_baseline": [True, False, False],
        "time_since_observed_baseline_days": [0, 10, 40],
        "time_since_observed_baseline_years": [0, 10 / 365.25, 40 / 365.25],
        "pop_status": pop, "overlap_status": ["neither", "overlap", "insufficient_info"],
        "essdai_total": [1.0, 4.0, 2.0],
        "eg_articular_active": [False, True, True],
        "eg_renal_active": [True, False, False],
        "eg_cns_active": [True, True, True],
        "eg_pulmonary_active": [False, False, False],
    })


def test_n_visits_create_n_minus_one_correct_consecutive_episodes():
    result = module.build_transition_episodes(visits())
    assert len(result) == 2
    assert result[["from_visit_id", "to_visit_id"]].values.tolist() == [["v0", "v1"], ["v1", "v2"]]
    assert result.interval_days.tolist() == [10, 30]
    assert result.loc[0, "from_pop"] == "Pop2" and result.loc[0, "to_pop"] == "Pop1"


def test_continuous_delta_is_destination_minus_origin():
    result = module.build_transition_episodes(visits())
    assert result.delta_essdai.tolist() == [3.0, -2.0]
    assert result.essdai_change_direction.tolist() == ["increased", "decreased"]


def test_domain_transition_categories_and_counts():
    result = module.build_transition_episodes(visits()).iloc[0]
    assert result.eg_articular_active_transition == "incident"
    assert result.eg_renal_active_transition == "resolved"
    assert result.eg_cns_active_transition == "persistent_active"
    assert result.eg_pulmonary_active_transition == "persistent_inactive"
    assert result.n_domains_incident == 1 and result.n_domains_resolved == 1


def test_unclassifiable_pop_episode_is_retained_but_unevaluable():
    result = module.build_transition_episodes(visits())
    assert len(result) == 2
    assert not bool(result.loc[1, "pop_transition_evaluable"])
    assert not bool(result.loc[1, "event_pop_change"])
    assert bool(result.loc[1, "transition_involves_unclassifiable"])


def test_observation_and_baseline_merges_preserve_episode_count():
    data = visits()
    observation = data[["patient_id", "visit_id"]].assign(
        avail_essdai=True, avail_esspri_complete=True, avail_sf36_complete=True,
        avail_profad_complete=False, avail_mdafs_complete=True, full_multidomain_complete=True)
    merged = module.merge_visit_completeness(module.validate_visit_data(data), observation)
    episodes = module.build_transition_episodes(merged)
    baseline = pd.DataFrame({"patient_id": ["p1"], "pop_status": ["Pop2"], "age_baseline": [50]})
    result = module.merge_baseline_characteristics(episodes, baseline)
    assert len(result) == len(episodes) == 2
    assert result.baseline_pop.eq("Pop2").all() and result.age_baseline.eq(50).all()
    assert result.episode_essdai_change_evaluable.all()


@pytest.mark.parametrize("mutation, message", [
    (lambda x: pd.concat([x, x.iloc[[0]]], ignore_index=True), "Duplicate patient_id"),
    (lambda x: x.assign(visit_date=[x.visit_date.iloc[0], x.visit_date.iloc[0], x.visit_date.iloc[2]]), "positive intervals"),
])
def test_invalid_duplicate_or_nonpositive_visits_raise(mutation, message):
    with pytest.raises(ValueError, match=message):
        module.validate_visit_data(mutation(visits()))

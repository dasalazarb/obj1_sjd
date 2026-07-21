import importlib
import numpy as np
import pandas as pd

from src.derivations.pro_scoring import SF36_ITEMS, score_all_pros


def _complete_sf36_row():
    row = {item: 1 for item in SF36_ITEMS}
    for q in range(3, 13): row[f"sf-36_health_survey__sf36_q{q}"] = 3
    for q in range(13, 20): row[f"sf-36_health_survey__sf36_q{q}"] = 2
    return row


def test_observed_only_esspri_never_uses_proxy_values():
    row = {"esspri_questionnaire__dryness": 3, "esspri_questionnaire__fatigue": 4,
           "esspri_questionnaire__pain": np.nan,
           "profile_of_fatigue_and_discomfort__profad_need_rest": 7}
    scored = score_all_pros(pd.DataFrame([row]))
    assert pd.isna(scored.loc[0, "esspri_total"])
    assert scored.loc[0, "esspri_n_components_available"] == 2
    assert not scored.loc[0, "esspri_scoring_valid"]


def test_shared_scoring_is_the_baseline_algorithm_and_pcs_mcs_are_numeric():
    baseline = importlib.import_module("src.block_A.09_pros_baseline")
    row = _complete_sf36_row() | {"esspri_questionnaire__dryness": 1, "esspri_questionnaire__fatigue": 2, "esspri_questionnaire__pain": 3}
    one = score_all_pros(pd.DataFrame([row]))
    two = baseline.score_all_pros(pd.DataFrame([row]))
    pd.testing.assert_series_equal(one["esspri_total"], two["esspri_total"])
    assert one.loc[0, "sf36_pcs"] == two.loc[0, "sf36_pcs"]
    assert np.isfinite(one.loc[0, "sf36_pcs"])
    assert np.isfinite(one.loc[0, "sf36_mcs"])


def test_visit_timing_keeps_follow_up_visits_with_unique_visit_ids():
    from src.derivations.visit_dates import add_visit_timing
    visits = pd.DataFrame({"patient_id": ["p1", "p1"], "visit_date": pd.to_datetime(["2020-01-01", "2020-06-01"])})
    timed = add_visit_timing(visits)
    assert len(timed) == 2
    assert timed[["patient_id", "visit_id"]].duplicated().sum() == 0
    assert timed["visit_number"].tolist() == [0, 1]

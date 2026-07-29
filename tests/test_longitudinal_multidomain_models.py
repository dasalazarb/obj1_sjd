"""Focused synthetic tests for script 25 (no clinical score derivation)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).parents[1] / "src/block_B/25_longitudinal_multidomain_models.py"
SPEC = importlib.util.spec_from_file_location("longitudinal_models", PATH)
models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(models)


def visits():
    rows = []
    for patient, days in (("a", (0, 200)), ("b", (0, 100)), ("c", (0, 200))):
        for i, day in enumerate(days):
            rows.append({"patient_id":patient, "time_since_observed_baseline_days":day,
                         "baseline_pop":"Pop1", "essdai_total":i + 1,
                         "avail_essdai":patient != "c" or i == 0})
    return pd.DataFrame(rows)


def test_outcome_eligibility_requires_two_valid_visits_and_gap():
    support = models.build_outcome_support(visits(), models.OUTCOME_SPECS["essdai"], 180)
    assert support.loc["a", "meets_visit_rule"]
    assert not support.loc["b", "meets_visit_rule"]
    assert not support.loc["c", "meets_visit_rule"]


def test_outcome_denominators_are_separate():
    frame = visits()
    for variable, flag in (("esspri_total","avail_esspri_complete"), ("sf36_pcs","avail_sf36_complete"),
                           ("sf36_mcs","avail_sf36_complete"), ("mdafs_global","avail_mdafs_complete")):
        frame[variable] = frame.essdai_total
        frame[flag] = frame.patient_id.ne({"esspri_total":"a", "sf36_pcs":"b", "sf36_mcs":"c", "mdafs_global":"b"}[variable])
    specs = {"essdai":models.OUTCOME_SPECS["essdai"], "esspri":models.OUTCOME_SPECS["esspri"],
             "pcs":models.OUTCOME_SPECS["pcs"], "mcs":models.OUTCOME_SPECS["mcs"],
             "fatigue":models.FATIGUE_SPECS["mdafs_global"]}
    marked, _ = models.add_eligibility_flags(frame, specs, 180)
    sets = {key:tuple(sorted(marked.loc[marked[f"eligible_{key}_model"], "patient_id"].unique())) for key in specs}
    assert len(set(sets.values())) > 1


def test_fatigue_auto_uses_support_and_tie_prefers_mdafs():
    frame = visits().drop(columns=["essdai_total", "avail_essdai"])
    for variable, flag in (("mdafs_global","avail_mdafs_complete"), ("profad_total","avail_profad_complete")):
        frame[variable] = 1.0; frame[flag] = True
    selected, n_m, n_p, reason = models.select_fatigue_outcome(frame, "auto", 180)
    assert (selected, n_m, n_p) == ("mdafs_global", 2, 2)
    assert "tied" in reason


def test_primary_formula_has_prespecified_covariates_only():
    formula = models.model_formula("essdai_total")
    for term in (models.TIME, "baseline_pop", "age_baseline", "C(sex)", "C(protocol_group)", "baseline_time_since_diagnosis_years"):
        assert term in formula
    for forbidden in ("pop_status", "next_pop", "transition", "delta", "destination"):
        assert forbidden not in formula


def test_linear_contrasts_include_covariance():
    time = models.TIME
    i1 = f'{time}:C(baseline_pop, Treatment(reference="Pop3"))[T.Pop1]'
    i2 = f'{time}:C(baseline_pop, Treatment(reference="Pop3"))[T.Pop2]'
    params = pd.Series({time:1.0, i1:2.0, i2:-0.5})
    covariance = pd.DataFrame(np.diag([4.0, 9.0, 16.0]), index=params.index, columns=params.index)
    rows = models.extract_trajectory_estimates(params, covariance)
    by_name = {r["estimand"]:r for r in rows}
    assert by_name["Pop1 annual slope"]["estimate"] == 3
    assert by_name["Pop2 annual slope"]["estimate"] == .5
    assert by_name["Pop3 annual slope"]["estimate"] == 1
    assert by_name["Pop1 annual slope"]["standard_error"] == pytest.approx(np.sqrt(13))
    assert by_name["Pop1 minus Pop3 annual slope difference"]["standard_error"] == 3
    assert by_name["Pop2 minus Pop3 annual slope difference"]["estimate"] == -.5


@pytest.mark.parametrize("key,expected", [("essdai",2), ("esspri",2), ("fatigue",2), ("active_domains",2), ("pcs",-2), ("mcs",-2)])
def test_standardized_direction(key, expected):
    specs = dict(models.OUTCOME_SPECS)
    specs["fatigue"] = models.FATIGUE_SPECS["mdafs_global"]
    result = pd.DataFrame([{"outcome_key":key,"outcome_label":"x","source_variable":"x","baseline_pop":"Pop1","estimand":"s","estimand_type":"annual_slope","estimate":4.0,"standard_error":2.0,"ci95_low":0.0,"ci95_high":8.0,"p_value":.1,"n_eligible_patients":20,"model_type":"m","model_status":"fitted"}])
    standardized = models.standardize_estimates(result, specs, {key:(2.0,"observed_baseline")})
    assert standardized.iloc[0].standardized_adverse_estimate == expected
    assert standardized.iloc[0].standardized_ci95_low <= standardized.iloc[0].standardized_ci95_high


def test_model_failure_is_returned_not_raised(monkeypatch):
    frame = pd.DataFrame({"patient_id":[f"p{i}" for i in range(15) for _ in (0,1)],
                          "baseline_pop":[f"Pop{i//5+1}" for i in range(15) for _ in (0,1)],
                          "essdai_total":[0,1]*15, "avail_essdai":True,
                          "eligible_essdai_model":True, models.TIME:[0,1]*15,
                          "age_baseline":40, "sex":"F", "protocol_group":"A",
                          "baseline_time_since_diagnosis_years":2})
    calls=[]
    def fail(*args): calls.append(1); raise np.linalg.LinAlgError("synthetic")
    monkeypatch.setattr(models, "_attempt_model", fail)
    answer = models.fit_longitudinal_model(frame, "essdai", models.OUTCOME_SPECS["essdai"], 10)
    assert answer["model_status"] == "failed"
    assert len(calls) == 2  # full, then the sole reduced adjustment specification

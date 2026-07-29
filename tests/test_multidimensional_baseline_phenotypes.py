"""Focused tests for the integrated multidimensional baseline analysis."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).parents[1] / "src/block_B/21_multidimensional_baseline_phenotypes.py"
spec = importlib.util.spec_from_file_location("baseline21", PATH)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def primary():
    return pd.DataFrame({"patient_id":["1","2","3"], "pop_status":["Pop1","Pop2","Pop3"], "protocol":["A","A",None], "essdai_total":[1,2,3], "esspri_total":[4,5,6]})


def test_merge_preserves_primary_and_does_not_add_secondary_only():
    observation=pd.DataFrame({"patient_id":["1","2","3","4"],"full_multidomain_complete":[True,False,True,True]})
    empty=pd.DataFrame({"patient_id":["1","2","3"]})
    result, summary=m.merge_baseline_blocks(primary(),observation,empty,empty)
    assert len(result)==3 and result.patient_id.is_unique
    assert summary.loc[summary.source.eq("observation"),"n_secondary_only"].item()==1


def test_duplicate_secondary_patient_raises():
    duplicate=pd.DataFrame({"patient_id":["1","1"]})
    empty=pd.DataFrame({"patient_id":["1","2","3"]})
    with pytest.raises(ValueError,match="duplicate patient_id"):
        m.merge_baseline_blocks(primary(),duplicate,empty,empty)


def test_serology_summary_uses_interpretability_denominator():
    data=pd.DataFrame({"pop_status":["Pop1"]*3,"anti_ro_pos":[True,False,True],"anti_ro_interpretable":[True,True,False]})
    row=m.build_profile_table(data).iloc[0]
    assert row.n_available==2 and row.n_positive==1 and row.pct_positive==50


def test_comorbidity_values_are_used_without_rederivation():
    data=pd.DataFrame({"pop_status":["Pop1","Pop1"],"fibromyalgia":[True,False]})
    row=m.build_profile_table(data).iloc[0]
    assert row.summary=="1 / 2 (50.0%)"
    assert "script 07" in row.denominator_definition


def test_models_are_focused_and_pop3_is_reference():
    data=pd.DataFrame({"pop_status":["Pop1","Pop2","Pop3"]*10,"age_baseline":range(30),"protocol_group":["A"]*30,"time_since_diagnosis_years":[1]*30})
    result=m.fit_adjusted_models(data,skip_models=True)
    assert not result.outcome.str.startswith("essdai").any()
    assert not result.outcome.str.startswith("esspri").any()
    assert set(result.reference_pop)=={"Pop3"}
    assert set(result.contrast)=={"Pop1 vs Pop3","Pop2 vs Pop3"}


def test_missing_outcomes_skip_without_blocking_descriptives():
    data=pd.DataFrame({"pop_status":["Pop1","Pop2","Pop3"],"age_baseline":[1,2,3]})
    profile=m.build_profile_table(data)
    models=m.fit_adjusted_models(data)
    assert not profile.empty
    assert set(models.model_status)=={"skipped_missing_variable"}

"""Synthetic contract tests for reusable study helpers."""
import pandas as pd
import pytest

from src.studies._shared import (enrich_transition_intervals, resolve_concept,
    validate_integrated_dataset, validate_predictors, validate_transition_intervals)


def master():
    return pd.DataFrame({"patient_id":["A","A","A"],"clinical_episode_id":["A1","A2","A3"],
      "clinical_anchor_date":["2020-01-01","2021-01-01","2022-01-01"],"clinical_visit_number":[1,2,3],
      "clinical_visit":[True]*3,"is_clinical_baseline":[True,False,False],"pop_status":["Pop1","Pop2","Pop3"],
      "essdai_total":[6,3,2],"esspri_total_observed":[3.,6.,3.],"integration_version":["v2_clinical_episode"]*3,
      "marker":[1,2,3]})


def intervals():
    return pd.DataFrame({"patient_id":["A","A"],"from_clinical_episode_id":["A1","A2"],
      "to_clinical_episode_id":["A2","A3"],"from_pop":["Pop1","Pop2"],"to_pop":["Pop2","Pop3"],
      "interval_days":[366,365],"interval_years":[366/365.25,365/365.25]})


def public_master():
    result = master().drop(columns=["pop_status", "essdai_total",
                                    "esspri_total_observed", "integration_version"])
    result["pop__status"] = ["Pop1", "Pop2", "Pop3"]
    result["essdai__total"] = [6, 3, 2]
    result["esspri__total"] = [3., 6., 3.]
    return result


def test_pharma_accepts_public_v2_schema_and_prefers_public_names():
    both = public_master().assign(pop_status="Pop3", essdai_total=999,
                                  esspri_total_observed=999)
    result = validate_integrated_dataset(both)
    assert result["contract_version"] == "clinical_episode_curated_v2"
    assert resolve_concept(both, "pop") == "pop__status"


def test_pharma_legacy_aliases_are_secondary_only():
    assert resolve_concept(master(), "pop") == "pop_status"
    with pytest.raises(ValueError, match="Missing required concept"):
        validate_integrated_dataset(master(), allow_legacy_aliases=False)


def test_pharma_rejects_missing_required_public_concepts():
    with pytest.raises(ValueError, match="esspri_total"):
        validate_integrated_dataset(public_master().drop(columns="esspri__total"))


def test_master_hard_failures():
    x=master(); x.loc[1,"clinical_episode_id"]="A1"
    with pytest.raises(ValueError,match="Duplicate"): validate_integrated_dataset(x)
    x=master(); x.loc[1,"pop_status"]="Other"
    with pytest.raises(ValueError,match="Unexpected"): validate_integrated_dataset(x)
    x=master(); x.loc[1,"is_clinical_baseline"]=True
    with pytest.raises(ValueError,match="one clinical baseline"): validate_integrated_dataset(x)


def test_interval_hard_failures():
    x=intervals(); x.loc[0,"to_clinical_episode_id"]="missing"
    with pytest.raises(ValueError,match="nonexistent"): validate_transition_intervals(x,master())
    x=intervals().iloc[[0]].copy(); x.loc[x.index[0],"to_clinical_episode_id"]="A3"; x.loc[x.index[0],"to_pop"]="Pop3"
    with pytest.raises(ValueError,match="consecutive"): validate_transition_intervals(x,master())
    x=intervals(); x.loc[0,"from_pop"]="Pop3"
    with pytest.raises(ValueError,match="discordant"): validate_transition_intervals(x,master())
    x=intervals(); x.loc[0,"interval_days"]=0
    with pytest.raises(ValueError,match="positive"): validate_transition_intervals(x,master())


def test_safe_enrichment_and_predictor_guard():
    out=enrich_transition_intervals(intervals(),master(),["marker"],["essdai_total"])
    assert out.from_marker.tolist()==[1,2]
    assert out.to_essdai_total.tolist()==[3,2]
    for bad in (["to_marker"],["next_marker"],["delta_marker"],["lab_patient_consensus_value"]):
      with pytest.raises(ValueError,match="future or consensus"): validate_predictors(bad)

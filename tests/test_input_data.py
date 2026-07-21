"""Tests for the visit merge performed during raw-input preparation."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "00_input data.py"
SPEC = spec_from_file_location("input_data", MODULE_PATH)
input_data = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(input_data)


def test_merge_matching_visits_uses_non_natural_history_values_for_overlapping_dates():
    source = pd.DataFrame(
        {
            "ids__patient_record_number": ["100", "100", "200"],
            "ids__interval_name": [
                "Natural History Protocol 478 Interval",
                "Follow-up interval",
                "Natural History Protocol 478 Interval",
            ],
            "ids__visit_date": ["5/13/2015", "2015-05-13 | 2015-05-18", "2015-05-13"],
            "ids__site_name": ["Natural History site", "Follow-up site", "Other patient site"],
            "visit_summary_form__sjogrens_class": ["1", "4", "2"],
            "vital_signs__pulse": [70, 80, 90],
            "source_file": ["natural.csv", "follow_up.csv", "other.csv"],
        }
    )

    result = input_data.merge_matching_visits(source)

    assert len(result) == 2
    patient_100 = result.loc[result["ids__patient_record_number"].eq("100")].iloc[0]
    assert patient_100["ids__interval_name"] == "Follow-up interval"
    assert patient_100["ids__visit_date"] == "2015-05-13 | 2015-05-18"
    assert patient_100["ids__site_name"] == "Follow-up site"
    assert patient_100["visit_summary_form__sjogrens_class"] == "4"
    assert patient_100["vital_signs__pulse"] == 80
    assert patient_100["source_file"] == "follow_up.csv"


def test_merge_matching_visits_falls_back_to_natural_history_when_preferred_value_missing():
    source = pd.DataFrame(
        {
            "ids__patient_record_number": ["100", "100"],
            "ids__interval_name": ["Natural History Protocol 478 Interval", "Follow-up interval"],
            "ids__visit_date": ["2015-05-13", "2015-05-13 | 2015-05-18"],
            "ids__race": ["Asian", None],
        }
    )

    result = input_data.merge_matching_visits(source)

    assert len(result) == 1
    assert result.loc[0, "ids__interval_name"] == "Follow-up interval"
    assert result.loc[0, "ids__race"] == "Asian"


def test_merge_matching_visits_requires_patient_date_and_interval_columns():
    source = pd.DataFrame({"ids__patient_record_number": ["100"]})

    try:
        input_data.merge_matching_visits(source)
    except KeyError as error:
        assert "ids__visit_date" in str(error)
        assert "ids__interval_name" in str(error)
    else:
        raise AssertionError("Expected a KeyError for missing merge columns")

"""Behavioral tests for the Pharma laboratory inventory."""
import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "src/studies/pharma/00_profile_labs.py"
SPEC = importlib.util.spec_from_file_location("pharma_lab_profile", SCRIPT)
profile_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile_module)


def test_categorical_result_can_come_from_reference_status():
    frame = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(10)],
        "ana__value": [pd.NA] * 10,
        "ana__text": [" ", ""] + [pd.NA] * 8,
        "ana__reference_status": [" positive ", "negative"] * 5,
    })

    row = profile_module.profile_labs(frame).iloc[0]

    assert row.inferred_type == "categorical"
    assert row.result_source == "reference_status"
    assert row.recommended_use == "categorical_longitudinal"
    assert row.n_nonmissing == 0
    assert row.n_text_nonmissing == 0
    assert row.n_reference_status_nonmissing == 10
    assert row.n_any_result == 10
    assert row.n_patients == 10
    assert row.data_sufficiency == "adequate"


def test_text_fallback_and_numeric_sufficiency_are_independent():
    frame = pd.DataFrame({
        "patient_id": ["a", "b", "c"],
        "ana_pattern__value": [pd.NA] * 3,
        "ana_pattern__text": ["speckled", " homogeneous ", "speckled"],
        "beta_hcg__value": [2.5, pd.NA, pd.NA],
        "empty__value": [pd.NA, pd.NA, pd.NA],
    })

    profile = profile_module.profile_labs(frame).set_index("lab")

    assert profile.loc["ana_pattern", "result_source"] == "text"
    assert profile.loc["ana_pattern", "recommended_use"] == "descriptive_only"
    assert profile.loc["beta_hcg", "inferred_type"] == "numeric"
    assert profile.loc["beta_hcg", "data_sufficiency"] == "insufficient"
    assert profile.loc["empty", "result_source"] == "none"
    assert profile.loc["empty", "review_reason"] == "no_valid_result"


def test_multiple_units_preserves_numeric_type_but_requires_review():
    frame = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(5)],
        "glucose__value": [1, 2, 3, 4, 5],
        "glucose__unit": ["mg/dL", " mmol/L ", "mg/dL", "", "  "],
    })

    row = profile_module.profile_labs(frame).iloc[0]

    assert row.inferred_type == "numeric"
    assert row.n_units == 2
    assert row.recommended_use == "review_or_exclude"
    assert "multiple_units" in row.review_reason


def test_numeric_values_do_not_require_the_string_accessor():
    frame = pd.DataFrame({
        "patient_id": ["a", "b", "c", "d"],
        "glucose__value": [1, 2, pd.NA, 4],
    })

    row = profile_module.profile_labs(frame).iloc[0]

    assert row.inferred_type == "numeric"
    assert row.n_nonmissing == 3
    assert row.n_unique == 3
    assert row.sample_values == "1 | 2 | 4"

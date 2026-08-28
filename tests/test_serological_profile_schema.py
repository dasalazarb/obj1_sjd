"""Schema regression tests for the episode laboratory artifacts."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "block_A" / "01_serological_profile.py"
)
SPEC = spec_from_file_location("serological_profile", MODULE_PATH)
serology = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(serology)


def test_analyte_schema_uses_nullable_semantic_dtypes():
    frame = pd.DataFrame(
        {
            "patient_id": ["1", "2"],
            "clinical_visit_number": [1, None],
            "clinical_visit": [True, None],
            "clinical_anchor_date": ["2024-01-01", None],
            "selected_value_numeric": ["2.5", None],
            "selected_value_text": [None, "positive"],
            "result_conflict": [False, True],
        }
    )

    result = serology._coerce_analyte_output_schema(frame)

    assert str(result["clinical_visit_number"].dtype) == "Int64"
    assert str(result["clinical_visit"].dtype) == "boolean"
    assert str(result["clinical_anchor_date"].dtype) == "datetime64[ns]"
    assert str(result["selected_value_numeric"].dtype) == "Float64"
    assert str(result["selected_value_text"].dtype) == "string"
    assert str(result["result_conflict"].dtype) == "boolean"
    assert serology._object_dtype_columns(result) == []


def test_wide_schema_keeps_numeric_and_text_values_separate():
    frame = pd.DataFrame(
        {
            "patient_id": ["1"],
            "clinical_episode_id": ["episode-1"],
            "clinical_visit_number": [None],
            "clinical_visit": [True],
            "wbc__value": ["3.2"],
            "wbc__text": [None],
            "wbc__measurement_date": ["2024-01-01"],
            "wbc__days_from_anchor": [0],
            "wbc__n_measurements": [1],
            "wbc__conflict": [False],
            "wbc__episode_status": ["low"],
        }
    )

    result = serology._coerce_wide_output_schema(frame)

    assert str(result["wbc__value"].dtype) == "Float64"
    assert str(result["wbc__text"].dtype) == "string"
    assert str(result["wbc__measurement_date"].dtype) == "datetime64[ns]"
    assert str(result["wbc__days_from_anchor"].dtype) == "Int64"
    assert str(result["wbc__n_measurements"].dtype) == "Int64"
    assert str(result["wbc__conflict"].dtype) == "boolean"
    assert str(result["wbc__episode_status"].dtype) == "string"
    assert not any(column.endswith("__episode_value") for column in result)
    assert serology._object_dtype_columns(result) == []


def test_build_wide_does_not_create_mixed_episode_value():
    spine = pd.DataFrame(
        {
            "patient_id": ["1"],
            "clinical_episode_id": ["episode-1"],
            "clinical_anchor_date": pd.to_datetime(["2024-01-01"]),
        }
    )
    selected = pd.DataFrame(
        {
            "patient_id": ["1"],
            "clinical_episode_id": ["episode-1"],
            "canonical_analyte": ["anti_ro_ssa"],
            "selected_value_numeric": [pd.NA],
            "selected_value_text": ["positive"],
            "selected_unit": [pd.NA],
            "selected_reference_status": ["positive"],
            "selected_lab_date": pd.to_datetime(["2023-12-31"]),
            "selected_days_from_clinical_anchor": [-1],
            "n_measurements_in_episode": [1],
            "result_conflict": [False],
            "selection_status": ["selected"],
        }
    )
    usable = pd.DataFrame(
        {
            "patient_id": ["1"],
            "canonical_analyte": ["anti_ro_ssa"],
            "lab_family": ["stable_autoimmune"],
            "lab_date": pd.to_datetime(["2023-12-31"]),
            "result_text": ["positive"],
        }
    )

    result, _ = serology.build_wide(spine, selected, usable)

    assert "anti_ro_ssa__episode_value" not in result
    assert result.loc[0, "anti_ro_ssa__text"] == "positive"
    assert result.loc[0, "anti_ro_ssa__episode_status"] == "positive"


def test_invalid_administrative_tokens_are_not_qualitative_results():
    row = pd.Series(
        {
            "result_raw": " Not tested ",
            "result_text": pd.NA,
            "reported_interpretation": pd.NA,
            "result_numeric_exact": pd.NA,
            "result_numeric_bound": pd.NA,
        }
    )

    assert serology._normalize_result_token(row["result_raw"]) == "not tested"
    assert serology._is_invalid_result_token(row["result_raw"])
    assert serology._value_type(row) == "invalid_nonresult"


def test_valid_result_wins_without_conflict_over_invalid_placeholder():
    usable = pd.DataFrame(
        {
            "patient_id": ["1", "1"],
            "clinical_episode_id": ["episode-1", "episode-1"],
            "canonical_analyte": ["ana", "ana"],
            "days_from_clinical_anchor": [0, 0],
            "lab_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "result_raw": ["NEGATIVE", "Not tested"],
            "result_text": [pd.NA, pd.NA],
            "reported_interpretation": [pd.NA, pd.NA],
            "result_numeric_exact": [pd.NA, pd.NA],
            "result_numeric_bound": [pd.NA, pd.NA],
            "result_operator": [pd.NA, pd.NA],
            "unit": [pd.NA, pd.NA],
            "_lab_record_id": [1, 2],
        }
    )
    spine = pd.DataFrame(
        {
            "patient_id": ["1"],
            "clinical_episode_id": ["episode-1"],
            "clinical_anchor_date": pd.to_datetime(["2024-01-01"]),
        }
    )

    selected, conflict_ids = serology.select_episode_analytes(usable, spine)

    assert selected.loc[0, "selected_value_type"] == "qualitative"
    assert selected.loc[0, "selected_value_text"] == "NEGATIVE"
    assert not selected.loc[0, "result_conflict"]
    assert selected.loc[0, "n_measurements_in_episode"] == 2
    assert selected.loc[0, "n_valid_measurements_in_episode"] == 1
    assert selected.loc[0, "conflict_resolved_by_invalid_token_filter"]
    assert conflict_ids == set()


def test_invalid_token_qc_preserves_source_record_counts():
    labs = pd.DataFrame(
        {
            "patient_id": ["1", "1", "2"],
            "matched_clinical_episode_id": ["e1", "e1", "e2"],
            "canonical_analyte": ["ana", "ana", "ana"],
            "result_raw": [":", "POSITIVE", ":"],
            "result_numeric_exact": [pd.NA, pd.NA, pd.NA],
            "result_numeric_bound": [pd.NA, pd.NA, pd.NA],
        }
    )

    result = serology.build_invalid_result_tokens_qc(labs)

    assert result.to_dict("records") == [
        {
            "canonical_analyte": "ana",
            "normalized_token": ":",
            "raw_example": ":",
            "n_records": 2,
            "n_patients": 2,
            "n_episodes": 2,
        }
    ]

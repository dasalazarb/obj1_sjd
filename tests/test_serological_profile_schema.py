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


def test_slide_priority_export_copies_all_available_variants_and_builds_qc(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    numeric_pdf = figures / "wbc.pdf"
    categorical_png = figures / "ana.png"
    ignored_png = figures / "other.png"
    for path in (numeric_pdf, categorical_png, ignored_png):
        path.write_text(path.stem, encoding="utf-8")
    manifest = pd.DataFrame(
        [
            {
                "lab_id": "wbc", "display_label": "WBC", "render_status": "ok",
                "pdf_path": str(numeric_pdf), "png_path": "",
            },
            {
                "lab_id": "ana_status", "display_label": "ANA",
                "render_status": "ok", "categorical_png_path": str(categorical_png),
            },
            {
                "lab_id": "not_requested", "display_label": "Other",
                "render_status": "ok", "png_path": str(ignored_png),
            },
        ]
    )

    exported = serology.export_slide_priority_figures(
        manifest, {"wbc", "ana_status", "missing"}, tmp_path / "slides"
    )
    qc = serology.build_slide_selection_qc(
        manifest, {"wbc", "ana_status", "missing"}
    ).set_index("lab_id")

    assert set(exported["lab_id"]) == {"wbc", "ana_status"}
    assert {path.name for path in (tmp_path / "slides").iterdir()} == {
        "wbc.pdf", "ana.png"
    }
    assert qc.loc["wbc", "has_numeric_figure"]
    assert qc.loc["ana_status", "has_categorical_figure"]
    assert not qc.loc["missing", "present_in_manifest"]


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
            "lab_id": ["anti_ro_ssa"],
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
            "lab_id": ["anti_ro_ssa"],
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


def test_fixed_genetic_asof_ignores_future_and_future_conflict():
    spine = pd.DataFrame({
        "patient_id": ["synthetic", "synthetic", "synthetic"],
        "clinical_episode_id": ["before", "known", "after-conflict"],
        "clinical_anchor_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
    })
    selected = pd.DataFrame()
    usable = pd.DataFrame({
        "patient_id": ["synthetic", "synthetic", "synthetic"],
        "lab_id": ["hla_fictional"] * 3,
        "lab_family": ["fixed_genetic"] * 3,
        "lab_date": pd.to_datetime(["2020-06-01", "2021-06-01", None]),
        "result_text": ["marker-a", "marker-b", "marker-undated"],
        "result_raw": [pd.NA] * 3,
    })
    result, _ = serology.build_wide(spine, selected, usable)
    assert result["hla_fictional__known_through_episode"].tolist() == [False, True, True]
    asof = result["hla_fictional__asof_value"]
    assert pd.isna(asof.iloc[0]) and asof.iloc[1] == "marker-a" and pd.isna(asof.iloc[2])
    # The global retrospective conflict is intentionally separate and does not
    # erase the value that was unambiguous at the middle episode.
    assert result["hla_fictional__patient_consensus_conflict"].all()


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
            "lab_id": ["ana", "ana"],
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


def test_lab_id_uses_canonical_then_complete_slugged_provenance():
    labs = pd.DataFrame(
        {
            "canonical_analyte": ["anti_ro_ssa", pd.NA, pd.NA],
            "order_name_original": ["ignored", "Acute Care Panel", pd.NA],
            "cluster_name_original": ["ignored", "Sodium (Blood)", "Sodium"],
        }
    )

    result = serology._add_lab_id(labs)

    assert result["lab_id"].tolist()[:2] == [
        "anti_ro_ssa",
        "acute_care_panel__sodium_blood",
    ]
    assert pd.isna(result.loc[2, "lab_id"])
    assert pd.isna(result.loc[1, "canonical_analyte"])


def test_build_wide_uses_fallback_lab_id():
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
            "lab_id": ["acute_care_panel__sodium_blood"],
            "selected_value_numeric": [140.0],
            "selected_value_text": [pd.NA],
            "selected_unit": ["mmol/L"],
            "selected_reference_status": ["within_range"],
            "selected_lab_date": pd.to_datetime(["2024-01-01"]),
            "selected_days_from_clinical_anchor": [0],
            "n_measurements_in_episode": [1],
            "result_conflict": [False],
            "selection_status": ["selected_single"],
        }
    )
    usable = pd.DataFrame(
        {
            "patient_id": ["1"],
            "lab_id": ["acute_care_panel__sodium_blood"],
            "lab_family": [pd.NA],
            "lab_date": pd.to_datetime(["2024-01-01"]),
            "result_numeric_exact": [140.0],
        }
    )

    result, _ = serology.build_wide(spine, selected, usable)

    assert result.loc[0, "acute_care_panel__sodium_blood__value"] == 140.0


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

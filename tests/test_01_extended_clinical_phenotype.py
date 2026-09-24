"""Contract tests for the extended clinical phenotype layer."""
import importlib.util
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


extended = load("extended", "src/block_A/01_extended_clinical_phenotype.py")
integrated = load("integrated_extended_test", "src/block_A/10_build_integrated_longitudinal_dataset.py")


def spine():
    frame = pd.DataFrame({
        "patient_id": ["a", "a"], "clinical_episode_id": ["a1", "a2"],
        "clinical_anchor_date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "clinical_visit_number": [1, 2], "clinical_visit": [True, True], "visit_type": ["clinical", "clinical"],
        "episode_start_date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "episode_end_date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "clinical_baseline_episode_id": ["a1", "a1"], "clinical_baseline_date": pd.to_datetime(["2020-01-01"] * 2),
        "is_clinical_baseline": [True, False], "time_since_clinical_baseline_days": [0, 366],
        "time_since_clinical_baseline_years": [0., 366 / 365.25],
        "biopsy_pathology__f_score": ["Not Done", "1.5"],
        "wus_only__flow_whole_unstim": ["0.05", pd.NA],
        "sgus_grading_scale:_omeract_(2021)__r_parotid_grade": [2, pd.NA],
    })
    return frame


def test_preserves_spine_baseline_and_does_not_leak():
    source = spine()
    result, _, _ = extended.build_extended_clinical_longitudinal(source)
    assert len(result) == len(source) and not result.duplicated(extended.KEYS).any()
    pd.testing.assert_frame_equal(result[["is_clinical_baseline", "clinical_baseline_episode_id", "clinical_baseline_date"]],
                                  source[["is_clinical_baseline", "clinical_baseline_episode_id", "clinical_baseline_date"]])
    assert pd.isna(result.loc[0, "biopsy_focus_score"])
    assert result.loc[1, "biopsy_focus_score"] == 1.5


def test_missing_is_not_false_and_scales_remain_separate():
    result, _, _ = extended.build_extended_clinical_longitudinal(spine())
    assert pd.isna(result.loc[0, "ocular_schirmer_abnormal_any"])
    assert result.loc[0, "sgus_omeract_r_parotid_grade"] == 2
    assert pd.isna(result.loc[0, "sgus_theander_final_score_grade"])
    assert not any(c.startswith("sgus_jousse_joulin_") and result.loc[0, c] == 2 for c in result)


def test_flow_provenance_and_step10_contract():
    source = spine()
    result, _, _ = extended.build_extended_clinical_longitudinal(source)
    assert result.loc[0, "salivary_flow_unstimulated_source"] == "wus_only__flow_whole_unstim"
    integrated.validate_source(integrated.validate_spine(source), result, "extended_clinical")


def test_focus_score_parser_examples():
    source = pd.concat([spine().iloc[[0]]] * 4, ignore_index=True)
    source["patient_id"] = ["a", "b", "c", "d"]
    source["clinical_episode_id"] = ["a1", "b1", "c1", "d1"]
    source["clinical_baseline_episode_id"] = source.clinical_episode_id
    source["biopsy_pathology__f_score"] = ["2", "1.5", "Not Done", ""]
    result, _, _ = extended.build_extended_clinical_longitudinal(source)
    assert result.biopsy_focus_score.iloc[:2].tolist() == [2., 1.5]
    assert result.biopsy_focus_score.iloc[2:].isna().all()

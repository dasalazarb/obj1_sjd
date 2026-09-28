"""Contract tests for the integrated clinical-baseline boundary."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "src" / "block_A" / "11_integrated_baseline_characterization.py"
SPEC = importlib.util.spec_from_file_location("baseline_characterization", SCRIPT)
baseline_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(baseline_module)


def integrated_frame():
    return pd.DataFrame({
        "patient_id": ["p1", "p1", "p2"], "clinical_episode_id": ["e1", "e2", "e3"],
        "clinical_anchor_date": ["2020-01-01", "2021-01-01", "2020-02-01"],
        "clinical_visit_number": [1, 2, 1], "clinical_baseline_episode_id": ["e1", "e1", "e3"],
        "clinical_baseline_date": ["2020-01-01", "2020-01-01", "2020-02-01"],
        "is_clinical_baseline": [True, False, True], "clinical_visit": [True, True, True],
        "pop_status": ["Pop1", "Pop2", "Pop3"], "essdai_total": [2.0, 4.0, 8.0],
        "biopsy_focus_score": [1.2, None, 2.1],
    })


def test_selects_only_official_baseline_and_retains_all_columns():
    frame = baseline_module.normalize_integrated_dtypes(integrated_frame())
    baseline = baseline_module.validate_integrated(frame)
    assert list(baseline.clinical_episode_id) == ["e1", "e3"]
    assert set(baseline.columns) == set(frame.columns)


def test_missing_baseline_is_hard_failure():
    frame = integrated_frame()
    frame.loc[frame.patient_id.eq("p2"), "is_clinical_baseline"] = False
    with pytest.raises(ValueError, match="No clinical baseline"):
        baseline_module.validate_integrated(baseline_module.normalize_integrated_dtypes(frame))


def test_duplicate_episode_key_is_hard_failure():
    frame = pd.concat([integrated_frame(), integrated_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate patient_id"):
        baseline_module.validate_integrated(baseline_module.normalize_integrated_dtypes(frame))


def test_availability_respects_upstream_scoring_validity():
    baseline = baseline_module.validate_integrated(baseline_module.normalize_integrated_dtypes(integrated_frame()))
    baseline["sf36_pcs"] = [40.0, 50.0]
    baseline["sf36_scoring_valid"] = [True, False]
    row = baseline_module.variable_availability(baseline).set_index("variable").loc["sf36_pcs"]
    assert row.n_nonmissing == 2
    assert row.n_available_for_analysis == 1


def test_table1_is_manifest_driven_and_excludes_ids_dates_and_sensitivity():
    baseline = baseline_module.validate_integrated(baseline_module.normalize_integrated_dtypes(integrated_frame()))
    baseline["ids__subject_number"] = [101, 102]
    baseline["esspri_total_s1_one_proxy"] = [4.0, 5.0]
    baseline["technical_measure"] = [99.0, 100.0]
    overall, by_pop = baseline_module.build_table1(baseline)
    rendered = " ".join(overall.Variable.astype(str))
    assert "essdai_total" in rendered
    assert "patient_id" not in rendered
    assert "subject_number" not in rendered
    assert "clinical_anchor_date" not in rendered
    assert "s1_one_proxy" not in rendered
    assert "technical_measure" not in rendered
    assert set(["Overall", *baseline_module.POP_ORDER]) <= set(by_pop.columns)


def test_datetime_never_enters_continuous_summary():
    baseline = baseline_module.validate_integrated(baseline_module.normalize_integrated_dtypes(integrated_frame()))
    summary = baseline_module.continuous_summary(baseline)
    assert "clinical_anchor_date" not in set(summary.variable)


def test_reporting_products_have_distinct_contracts():
    baseline = baseline_module.validate_integrated(baseline_module.normalize_integrated_dtypes(integrated_frame()))
    assert {"n_nonmissing", "n_valid"} <= set(baseline_module.variable_availability(baseline))
    assert {"n_missing", "missingness_category"} <= set(baseline_module.missingness_qc(baseline))
    assert {"qc_type", "n_violations", "status"} <= set(baseline_module.variable_qc(baseline))

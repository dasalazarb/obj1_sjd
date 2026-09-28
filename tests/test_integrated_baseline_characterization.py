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


def test_binary_table1_uses_evaluable_denominator_and_one_event_row():
    baseline = baseline_module.validate_integrated(baseline_module.normalize_integrated_dtypes(integrated_frame()))
    # Add two further patients so the denominator regression is explicit.
    extra = pd.DataFrame({column: [baseline.iloc[-1][column], baseline.iloc[-1][column]] for column in baseline})
    extra["patient_id"] = ["p3", "p4"]
    extra["clinical_episode_id"] = extra["clinical_baseline_episode_id"] = ["e4", "e5"]
    baseline = pd.concat([baseline, extra], ignore_index=True)
    baseline["ocular_staining_positive"] = pd.Series([True, True, False, None], dtype="boolean")

    categorical = baseline_module.categorical_summary(baseline, ["ocular_staining_positive"])
    overall_category = categorical.loc[categorical.group.eq("Overall")].iloc[0]
    assert overall_category.denominator == 3
    assert overall_category.n == 2
    assert overall_category.pct == pytest.approx(66.6667)
    assert "Missing" not in set(categorical.level)

    overall, _ = baseline_module.build_table1(baseline)
    row = overall.loc[overall.Variable.eq("Ocular staining positive, n/N (%)")].iloc[0]
    assert row["N available"] == 3
    assert row["N missing"] == 1
    assert row.Summary == "2/3 (66.7%)"


def test_multiclass_categories_exclude_missing_and_use_available_denominator():
    baseline = pd.DataFrame({"sex": ["Female", "Male", None, "Female"]})
    summary = baseline_module.categorical_summary(baseline, ["sex"])
    overall = summary.loc[summary.group.eq("Overall")]
    assert set(overall.level) == {"Female", "Male"}
    assert set(overall.denominator) == {3}
    assert overall.set_index("level").loc["Female", "pct"] == pytest.approx(66.6667)


def test_pop_status_is_overall_only_not_an_outcome_in_by_pop_table():
    baseline = baseline_module.validate_integrated(baseline_module.normalize_integrated_dtypes(integrated_frame()))
    overall, by_pop = baseline_module.build_table1(baseline)
    assert overall.Variable.str.startswith("pop_status:").any()
    assert not by_pop.Variable.str.startswith("pop_status:").any()


def test_race_and_ethnicity_are_separate_canonical_categories():
    baseline = baseline_module.validate_integrated(baseline_module.normalize_integrated_dtypes(integrated_frame()))
    baseline["ids__race"] = [" Asian ", "asian"]
    baseline["ids__ethnicity"] = ["Hispanic", "Not Hispanic"]
    derived = baseline_module.add_demographic_history_derivations(baseline)

    assert derived.race.tolist() == ["Asian", "Asian"]
    assert derived.ethnicity.tolist() == ["Hispanic", "Not Hispanic"]
    assert "race_ethnicity" not in derived
    manifest = baseline_module.resolved_table1_manifest(derived)
    assert {"race", "ethnicity"} <= set(manifest["Cohort / demographics"])
    assert "ids__race" not in sum(manifest.values(), [])
    assert "ids__ethnicity" not in sum(manifest.values(), [])
    assert baseline_module.variable_role(derived, "race") == "clinical_measure"
    assert baseline_module.variable_role(derived, "ethnicity") == "clinical_measure"
    assert baseline_module.variable_role(derived, "ids__race") == "raw"
    assert baseline_module.normalize_demographic_categories(pd.Series(["Unknown"])).iloc[0] == "Unknown"

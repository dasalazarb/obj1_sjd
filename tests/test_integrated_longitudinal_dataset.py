"""Behavioural tests for the clinical-episode integration boundary."""
import importlib.util
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
MODULE_PATH = Path(__file__).resolve().parents[1] / "src/block_A/10_build_integrated_longitudinal_dataset.py"
spec = importlib.util.spec_from_file_location("integrated_builder", MODULE_PATH)
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(builder)


def frames():
    spine = pd.DataFrame({
        "patient_id": ["a", "a", "b"], "clinical_episode_id": ["a1", "a2", "b1"],
        "clinical_anchor_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-06-01"]),
        "clinical_visit_number": [1, 2, 1], "clinical_visit": [True] * 3,
        "visit_type": ["clinical"] * 3, "episode_start_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-06-01"]),
        "episode_end_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-06-01"]),
        "clinical_baseline_episode_id": ["a1", "a1", "b1"],
        "clinical_baseline_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-06-01"]),
        "is_clinical_baseline": [True, False, True],
        "time_since_clinical_baseline_days": [0, 366, 0],
        "time_since_clinical_baseline_years": [0.0, 366 / 365.25, 0.0],
    })
    metadata = spine[[*builder.KEYS, "clinical_anchor_date", "clinical_visit_number"]]
    pop = metadata.assign(pop_status=["Pop2", "Pop1", "Pop3"], essdai_total=[4., 2., 1.], esspri_total_observed=[7., 5., 3.], esspri_total=[7., 5., 3.])
    labs = metadata.assign(**{"crp__value": [1., 2., pd.NA], "crp__text": [pd.NA] * 3,
                              "crp__measurement_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"), pd.NaT],
                              "crp__n_measurements": [1, 1, 0]})
    overlap = metadata.assign(overlap_status=["neither", "overlap", "neither"], overlap_evaluable=[True] * 3,
                              extraglandular_active=[False, True, False])
    pros = metadata.assign(sf36_pcs=[40., 35., 50.], sf36_mcs=[45., 46., 48.], profad_total=[1., 2., 3.], mdafs_global=[2., 3., 4.])
    extended = metadata.assign(
        biopsy_focus_score=[1.0, pd.NA, pd.NA],
        salivary_flow_unstimulated=[pd.NA, 0.2, pd.NA],
        ocular_schirmer_min=[pd.NA, pd.NA, 3.0],
        ocular_staining_positive=[pd.NA, pd.NA, pd.NA],
        sicca_any_symptom=[pd.NA, True, pd.NA],
        sgus_available=[False, False, True],
    )
    return spine, pop, labs, overlap, pros, extended


def test_sources_integrate_on_episode_and_spine_is_preserved():
    inputs = frames()
    result, _ = builder.build_integrated(*inputs)
    assert len(result) == len(inputs[0])
    assert not result.duplicated(builder.KEYS).any()
    a2 = result.set_index("clinical_episode_id").loc["a2"]
    assert (a2.pop_status, a2["crp__value"], a2.overlap_status, a2.sf36_pcs) == ("Pop1", 2., "overlap", 35.)


@pytest.mark.parametrize("source_index", [1, 2, 3, 4, 5])
def test_missing_source_episode_is_hard_failure(source_index):
    inputs = list(frames())
    inputs[source_index] = inputs[source_index].iloc[:-1]
    with pytest.raises(AssertionError, match="contract"):
        builder.build_integrated(*inputs)


def test_extra_extended_episode_is_hard_failure():
    inputs = list(frames())
    extra = inputs[5].iloc[[0]].assign(patient_id="z", clinical_episode_id="z1")
    inputs[5] = pd.concat([inputs[5], extra], ignore_index=True)
    with pytest.raises(AssertionError, match="contract"):
        builder.build_integrated(*inputs)


def test_duplicate_extended_episode_is_hard_failure():
    inputs = list(frames())
    inputs[5] = pd.concat([inputs[5], inputs[5].iloc[[0]]], ignore_index=True)

    with pytest.raises(AssertionError, match="contract"):
        builder.build_integrated(*inputs)


@pytest.mark.parametrize("column,value", [("clinical_anchor_date", pd.Timestamp("1999-01-01")), ("clinical_visit_number", 99)])
def test_structural_discrepancy_is_hard_failure(column, value):
    inputs = list(frames())
    inputs[2].loc[0, column] = value
    with pytest.raises(AssertionError, match="contract"):
        builder.build_integrated(*inputs)


def test_extended_structural_discrepancy_is_hard_failure():
    inputs = list(frames())
    inputs[5].loc[0, "clinical_anchor_date"] = pd.Timestamp("1999-01-01")

    with pytest.raises(AssertionError, match="contract"):
        builder.build_integrated(*inputs)


def test_inherited_raw_spine_columns_are_not_reintegrated_from_pop():
    spine, pop, labs, overlap, pros, extended = frames()
    spine["esspri_questionnaire__dryness"] = [2.0, 4.0, 6.0]
    pop["esspri_questionnaire__dryness"] = [2.0, 4.00001, 6.0]

    result, summaries = builder.build_integrated(spine, pop, labs, overlap, pros, extended)

    assert result["esspri_questionnaire__dryness"].tolist() == [2.0, 4.0, 6.0]
    assert "pop_status" in result.columns
    assert "esspri_total_observed" in result.columns
    assert summaries[0]["n_inherited_spine_columns_dropped"] == 3


def test_true_derived_feature_conflict_is_hard_failure():
    spine, pop, labs, overlap, pros, extended = frames()
    pros["esspri_total_observed"] = [1.0, 2.0, 3.0]

    with pytest.raises(AssertionError, match="conflicting duplicate features"):
        builder.build_integrated(spine, pop, labs, overlap, pros, extended)


def test_identical_extended_feature_collision_keeps_one_copy_without_suffixes():
    spine, pop, labs, overlap, pros, extended = frames()
    pop["biopsy_focus_score"] = extended["biopsy_focus_score"]

    result, _ = builder.build_integrated(spine, pop, labs, overlap, pros, extended)

    assert result.columns.tolist().count("biopsy_focus_score") == 1
    assert not any(column.endswith(("_x", "_y")) for column in result)


def test_conflicting_extended_feature_collision_is_hard_failure():
    spine, pop, labs, overlap, pros, extended = frames()
    pop["biopsy_focus_score"] = [2.0, pd.NA, pd.NA]

    with pytest.raises(AssertionError, match="conflicting duplicate features"):
        builder.build_integrated(spine, pop, labs, overlap, pros, extended)


def test_only_retrospective_features_and_labs_have_semantic_dtypes():
    result, _ = builder.build_integrated(*frames())
    assert not any(column.startswith("next_") for column in result)
    assert result.set_index("clinical_episode_id").loc["a2", "previous_pop_status"] == "Pop2"
    assert result["crp__value"].dtype == "Float64"
    assert result["crp__text"].dtype.name == "string"
    assert result["crp__n_measurements"].dtype == "Int64"
    assert not result.set_index("clinical_episode_id").loc["b1", "has_lab_measurement"]


def test_extended_coverage_and_five_block_count():
    result, summaries = builder.build_integrated(*frames())
    indexed = result.set_index("clinical_episode_id")

    assert indexed.loc["a1", "has_extended_clinical_data"]
    assert indexed.loc["b1", "has_extended_clinical_data"]
    assert indexed.loc["a1", "n_integrated_blocks_available"] == 5
    assert result["n_integrated_blocks_available"].max() <= 5
    assert summaries[-1]["source"] == "extended_clinical"


def test_sgus_false_alone_is_not_coverage_but_true_is():
    inputs = list(frames())
    extended = inputs[5]
    primary_non_sgus = [
        column for column in builder.EXTENDED_CLINICAL_PRIMARY_FEATURES
        if column != "sgus_available"
    ]
    extended[primary_non_sgus] = pd.NA
    extended["sgus_available"] = [False, True, False]

    result, _ = builder.build_integrated(*inputs)
    coverage = result.set_index("clinical_episode_id")["has_extended_clinical_data"]

    assert not coverage.loc["a1"]
    assert coverage.loc["a2"]
    assert not coverage.loc["b1"]


def test_final_keys_and_baseline_metadata_exactly_match_spine():
    inputs = frames()
    result, _ = builder.build_integrated(*inputs)
    spine = builder.validate_spine(inputs[0]).set_index(builder.KEYS)
    actual = result.set_index(builder.KEYS)

    assert set(actual.index) == set(spine.index)
    for column in [
        "is_clinical_baseline",
        "clinical_baseline_episode_id",
        "clinical_baseline_date",
    ]:
        pd.testing.assert_series_equal(
            actual[column].sort_index(), spine[column].sort_index(),
            check_names=False,
        )


def zero_block_inputs():
    spine, pop, labs, overlap, pros, extended = frames()
    # Patient b's sole (baseline) episode has no data in any integrated block.
    pop.loc[2, ["pop_status", "essdai_total", "esspri_total_observed", "esspri_total"]] = pd.NA
    labs.loc[2, ["crp__value", "crp__measurement_date"]] = pd.NA
    labs.loc[2, "crp__n_measurements"] = 0
    overlap.loc[2, "overlap_evaluable"] = False
    pros.loc[2, ["sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global"]] = pd.NA
    extended.loc[2, ["biopsy_focus_score", "salivary_flow_unstimulated",
                     "ocular_schirmer_min", "ocular_staining_positive",
                     "sicca_any_symptom"]] = pd.NA
    extended.loc[2, "sgus_available"] = False
    return spine, pop, labs, overlap, pros, extended


def test_zero_block_episode_is_not_removed():
    inputs = zero_block_inputs()
    result, _ = builder.build_integrated(*inputs)

    assert len(result) == len(inputs[0])
    assert result.set_index("clinical_episode_id").loc["b1", "n_integrated_blocks_available"] == 0


def test_zero_block_baseline_is_identified():
    result, _ = builder.build_integrated(*zero_block_inputs())
    _, summary = builder.build_zero_block_qc(result)

    assert summary["n_zero_block_clinical_baselines"] == 1
    assert summary["zero_block_baseline_present"] is True


def test_patient_with_all_episodes_zero_block_is_identified():
    result, _ = builder.build_integrated(*zero_block_inputs())
    _, summary = builder.build_zero_block_qc(result)

    assert summary["n_patients_with_all_episodes_zero_block"] == 1

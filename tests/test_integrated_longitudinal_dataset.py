"""Behavioural tests for the curated clinical-episode integration boundary."""
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
        "visit_type": ["clinical"] * 3,
        "episode_start_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-06-01"]),
        "episode_end_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-06-01"]),
        "clinical_baseline_episode_id": ["a1", "a1", "b1"],
        "clinical_baseline_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-06-01"]),
        "is_clinical_baseline": [True, False, True],
        "time_since_clinical_baseline_days": [0, 366, 0],
        "time_since_clinical_baseline_years": [0.0, 366 / 365.25, 0.0],
        "n_raw_rows_in_episode": [1, 2, 1],
    })
    metadata = spine[[*builder.KEYS, "clinical_anchor_date", "clinical_visit_number"]]
    pop = metadata.assign(
        pop_status=["Pop2", "Pop1", "Pop3"], essdai_total=[4., 2., 1.],
        esspri_dryness=[7., 5., 3.], esspri_fatigue=[7., 5., 3.],
        esspri_pain=[7., 5., 3.], esspri_total=[7., 5., 3.],
        esspri_total_observed=[7., 5., 3.],
    )
    labs = metadata.assign(**{
        "crp__value": [1., 2., pd.NA], "crp__text": ["1", "2.0", pd.NA],
        "crp__unit": ["mg/L", "mg/L", pd.NA],
        "crp__measurement_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01"), pd.NaT],
        "crp__n_measurements": [1, 1, 0], "crp__conflict": [False, False, False],
        "ana__text": ["positive", "negative", pd.NA],
    })
    overlap = metadata.assign(
        overlap_status=["neither", "overlap", "neither"], overlap_evaluable=[True] * 3,
        extraglandular_active=[False, True, False], eg_pulmonary_active=[False, True, False],
        pulmonary=[False, True, False],
    )
    pros = metadata.assign(sf36_pcs=[40., 35., 50.], sf36_mcs=[45., 46., 48.],
                           profad_total=[1., 2., 3.], mdafs_global=[2., 3., 4.],
                           mdafs_n_activity_items_answered=[4., 5., 6.],
                           sf36_available=[True] * 3, sf36_conflict=[False] * 3,
                           sf36_scoring_version=["v1"] * 3)
    extended = metadata.assign(
        biopsy_focus_score=[1.0, pd.NA, pd.NA], biopsy_evaluable=[True, False, False],
        salivary_flow_unstimulated=[pd.NA, 0.2, pd.NA], ocular_schirmer_min=[pd.NA, pd.NA, 3.0],
        ocular_staining_positive=[pd.NA] * 3, sicca_any_symptom=[pd.NA, True, pd.NA],
        sgus_available=[False, False, True], biopsy_focus_score_source=["x", pd.NA, pd.NA],
    )
    return spine, pop, labs, overlap, pros, extended


def test_curated_outputs_preserve_spine_and_partition_roles():
    inputs = frames()
    result = builder.build_curated(*inputs)
    assert len(result.analytic) == len(inputs[0])
    assert not result.analytic.duplicated(builder.KEYS).any()
    assert set(map(tuple, result.analytic[builder.KEYS].to_numpy())) == set(map(tuple, inputs[0][builder.KEYS].to_numpy()))
    assert "crp__value" in result.analytic
    assert "ana__text" in result.analytic
    assert "crp__text" not in result.analytic  # numeric text is reconstructible
    assert {"crp__unit", "crp__measurement_date", "sf36_available", "biopsy_evaluable"} <= set(result.context)
    assert "mdafs_n_activity_items_answered" in result.context
    assert "mdafs_n_activity_items_answered" not in result.analytic
    assert "crp__conflict" not in result.analytic and "crp__conflict" not in result.context
    assert "biopsy_focus_score_source" not in result.analytic
    assert "esspri_total_observed" not in result.analytic
    assert "pulmonary" not in result.analytic
    assert not any(c.endswith(("_x", "_y")) for c in result.analytic)


@pytest.mark.parametrize("source_index", [1, 2, 3, 4, 5])
def test_missing_source_episode_is_hard_failure(source_index):
    inputs = list(frames())
    inputs[source_index] = inputs[source_index].iloc[:-1]
    with pytest.raises(AssertionError, match="contract"):
        builder.build_curated(*inputs)


def test_dynamic_new_lab_family_is_classified_without_analyte_list():
    _, _, labs, _, _, _ = frames()
    labs["new_marker__value"] = [1.0, pd.NA, 2.0]
    labs["new_marker__unit"] = ["U", pd.NA, "U"]
    labs["new_marker__measurement_date"] = pd.to_datetime(["2020-01-01", None, "2020-06-01"])
    labs["new_marker__conflict"] = [False, False, True]
    roles = {c: builder.classify_column("labs", c, labs).role for c in labs if c.startswith("new_marker")}
    assert roles == {"new_marker__value": "ANALYTIC", "new_marker__unit": "CONTEXT",
                     "new_marker__measurement_date": "CONTEXT", "new_marker__conflict": "QC"}


def test_identical_alias_is_excluded_and_discordant_alias_fails():
    inputs = list(frames())
    result = builder.build_curated(*inputs)
    assert "esspri_total_observed" not in result.analytic
    inputs[1].loc[0, "esspri_total_observed"] = 999
    with pytest.raises(builder.CurationContractError, match="discordant"):
        builder.build_curated(*inputs)


def test_unknown_column_is_fail_closed():
    inputs = list(frames())
    inputs[0]["unknown_clinical_xyz"] = 1
    with pytest.raises(builder.CurationContractError, match="unknown_clinical_xyz"):
        builder.build_curated(*inputs)


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("ids__race", "ANALYTIC"),
        ("ids__ethnicity", "ANALYTIC"),
        ("ids__age_at_visit", "ANALYTIC"),
        ("visit_datetime", "CONTEXT"),
        ("ids__interval_name", "CONTEXT"),
        ("ids__time_24_hour", "CONTEXT"),
        ("ids__visit_date", "PROVENANCE"),
        ("ids__patient_record_number", "PROVENANCE"),
        ("ids__subject_number", "PROVENANCE"),
        ("ids__dob", "PROVENANCE"),
        ("ids__sex", "PROVENANCE"),
        ("ansar__e_i_ratio", "PROVENANCE"),
        ("some_form__some_question", "PROVENANCE"),
    ],
)
def test_spine_curated_exceptions_and_raw_form_default(name, role):
    assert builder.classify_spine_column(name)["role"] == role


def test_unknown_canonical_spine_column_remains_unclassified():
    assert builder.classify_spine_column("new_clinical_score") is None


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("mdafs_n_activity_items_answered", "CONTEXT"),
        ("profad_n_items_answered", "CONTEXT"),
        ("mdafs_global", "ANALYTIC"),
        ("mdafs_conflict", "QC"),
        ("new_unknown_canonical_score", "UNCLASSIFIED"),
    ],
)
def test_pro_score_and_completeness_classification_is_fail_closed(name, role):
    pros = pd.DataFrame({name: [1.0, pd.NA]})
    assert builder.classify_column("pros", name, pros).role == role


def test_pop_proxy_legacy_is_explicit_failure():
    inputs = list(frames())
    inputs[1]["esspri_proxy_total"] = 1.0
    with pytest.raises(builder.CurationContractError, match="observed-only"):
        builder.build_curated(*inputs)


def test_temporal_and_coverage_fields_are_not_in_master():
    result = builder.build_curated(*frames())
    assert "n_integrated_blocks_available" not in result.analytic
    assert not any(c.startswith(("previous_", "delta_")) for c in result.analytic)
    assert "n_integrated_blocks_available" in result.coverage
    _, summary = builder.build_zero_block_qc(result.coverage)
    assert summary["n_zero_block_episodes"] == 0


def test_registry_has_exactly_one_role_for_every_source_column():
    inputs = frames()
    result = builder.build_curated(*inputs)
    assert not result.registry.duplicated(["source", "variable"]).any()
    assert set(result.registry.role) <= builder.ROLES
    assert not result.registry.role.eq("UNCLASSIFIED").any()
    assert len(result.registry) == sum(len(frame.columns) for frame in inputs)

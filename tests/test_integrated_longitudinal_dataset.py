"""Minimal behavioural tests for the integrated longitudinal dataset builder."""
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
    spine = pd.DataFrame({"patient_id": ["a", "a", "b"], "visit_id": ["a0", "a1", "b0"],
                          "visit_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-06-01"]),
                          "visit_number": [0, 1, 0], "time_since_observed_baseline_days": [0, 366, 0]})
    pop = pd.DataFrame({"patient_id": ["a", "a", "b"], "visit_id": ["a0", "a1", "b0"],
                        "visit_date": spine.visit_date, "pop_status": ["Pop2", "Pop1", "Pop3"],
                        "essdai_total": [4.0, 2.0, 1.0], "esspri_total": [7.0, 5.0, 3.0],
                        "baseline_pop_status": ["Pop2", "Pop2", "Pop3"]})
    overlap = pd.DataFrame({"patient_id": ["a", "a", "b"], "visit_id": ["a0", "a1", "b0"],
                            "visit_date": spine.visit_date, "overlap_status": ["neither", "overlap", "neither"],
                            "extraglandular_active": [False, True, True], "extraglandular_evaluable": [True, True, True],
                            "n_extraglandular_domains_active": [0, 2, 1]})
    pros = pd.DataFrame({"patient_id": ["a", "a", "b"], "visit_id": ["a0", "a1", "b0"],
                         "visit_date": spine.visit_date, "sf36_pcs": [40.0, 35.0, 50.0], "sf36_mcs": [45.0, 46.0, 48.0],
                         "profad_total": [1.0, 2.0, 3.0], "profad_fatigue": [1.0, 2.0, 3.0], "mdafs_total": [2.0, 3.0, 4.0]})
    return spine, pop, overlap, pros


def test_merge_preserves_spine_rows_and_unique_patient_visits():
    spine, pop, overlap, pros = frames()
    result, _ = builder.build_integrated(spine, pop, overlap, pros)
    assert len(result) == len(spine)
    assert not result.duplicated(["patient_id", "visit_id"]).any()


def test_lags_deltas_incidence_and_original_scores_are_preserved():
    spine, pop, overlap, pros = frames()
    result, _ = builder.build_integrated(spine, pop, overlap, pros)
    follow_up = result.loc[result.visit_id.eq("a1")].iloc[0]
    assert follow_up.previous_pop == "Pop2"
    assert follow_up.next_pop is pd.NA or pd.isna(follow_up.next_pop)
    assert follow_up.delta_pcs == -5.0
    assert follow_up.incident_extraglandular_from_previous
    assert result.loc[result.visit_id.eq("a0"), "sf36_pcs"].iloc[0] == 40.0
    assert result.loc[result.visit_id.eq("a1"), "essdai_total"].iloc[0] == 2.0

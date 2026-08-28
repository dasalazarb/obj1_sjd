"""Contract tests for canonical clinical-episode Pop transitions."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "src" / "block_A" / "10_pop_transitions.py"
SPEC = importlib.util.spec_from_file_location("pop_transitions", SCRIPT)
transitions = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(transitions)


def episodes(states=("Pop1", "Pop2", "Pop3"), dates=None):
    dates = dates or ("2020-01-01", "2020-07-01", "2021-01-01")
    return pd.DataFrame({
        "patient_id": ["A"] * len(states),
        "clinical_episode_id": [f"A{i}" for i in range(1, len(states) + 1)],
        "clinical_anchor_date": list(dates),
        "clinical_visit_number": range(1, len(states) + 1),
        "pop_status": list(states),
    })


def test_pairs_adjacent_canonical_episodes_and_computes_elapsed_time():
    intervals, qc = transitions.build_intervals(episodes())
    assert list(zip(intervals.from_clinical_episode_id, intervals.to_clinical_episode_id)) == [
        ("A1", "A2"), ("A2", "A3")
    ]
    assert qc["n_adjacent_intervals_expected"] == qc["n_intervals_retained"] == 2
    assert intervals.loc[0, "interval_days"] == (pd.Timestamp("2020-07-01") - pd.Timestamp("2020-01-01")).days
    assert intervals.loc[0, "interval_years"] == pytest.approx(intervals.loc[0, "interval_days"] / 365.25)


def test_unclassifiable_is_described_but_never_bridged_for_model():
    frame = episodes(("Pop1", "Unclassifiable", "Pop2"))
    descriptive, _ = transitions.build_intervals(frame)
    assert descriptive.transition_pair.tolist() == [
        "Pop1 -> Unclassifiable", "Unclassifiable -> Pop2"
    ]
    assert "Pop1 -> Pop2" not in descriptive.transition_pair.tolist()
    model, qc = transitions.prepare_multistate_data(frame)
    assert model.empty
    assert qc["n_adjacent_intervals_excluded_unclassifiable"] == 2


def test_duplicate_episode_key_is_a_hard_failure():
    frame = episodes()
    frame.loc[1, "clinical_episode_id"] = "A1"
    with pytest.raises(ValueError, match="Duplicate"):
        transitions.validate_input(frame)


def test_nonpositive_interval_is_counted_and_excluded():
    frame = episodes(("Pop1", "Pop2"), ("2020-01-01", "2020-01-01"))
    intervals, qc = transitions.build_intervals(frame)
    assert intervals.empty
    assert qc["n_adjacent_intervals_created_before_time_filter"] == 1
    assert qc["n_intervals_nonpositive_time"] == 1


def test_contract_needs_no_legacy_visit_or_baseline_columns():
    frame = episodes()
    assert set(frame.columns) == transitions.REQUIRED_COLUMNS
    validated = transitions.validate_input(frame)
    assert validated.clinical_visit_number.tolist() == [1, 2, 3]

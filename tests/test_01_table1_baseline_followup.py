"""Focused contract tests for the Table 1 longitudinal extension."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "src" / "block_A" / "01_table1_baseline.py"
SPEC = importlib.util.spec_from_file_location("table1_baseline", SCRIPT)
table1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(table1)


def episode_frame(dates, episode_ids=None, protocols=None, baseline="2020-01-01"):
    episode_ids = episode_ids or [f"e{i}" for i in range(len(dates))]
    data = {
        "patient_id": ["p1"] * len(dates),
        "clinical_episode_id": episode_ids,
        "clinical_anchor_date": dates,
        "clinical_baseline_date": [baseline] * len(dates),
        "clinical_visit": [True] * len(dates),
        "clinical_baseline_episode_id": [episode_ids[0]] * len(dates),
        "is_clinical_baseline": [True] + [False] * (len(dates) - 1),
    }
    if protocols is not None:
        data["source_protocol"] = protocols
    return pd.DataFrame(data)


def prepare_metrics(frame):
    episodes, audit = table1.prepare_longitudinal_clinical_episodes(frame, ["p1"])
    return episodes, audit, table1.build_patient_followup_metrics(episodes, ["p1"])


def test_baseline_only():
    _, _, metrics = prepare_metrics(episode_frame(["2020-01-01"]))
    row = metrics.iloc[0]
    assert row.n_clinical_episodes == 1
    assert row.followup_days == row.followup_years == 0
    assert not row.has_followup_6mo


def test_followup_over_one_year():
    _, _, metrics = prepare_metrics(episode_frame(["2020-01-01", "2021-01-02"]))
    assert metrics.iloc[0].n_clinical_episodes == 2
    assert metrics.iloc[0].visits_per_followup_year == pytest.approx(1, abs=0.01)
    assert metrics.iloc[0].has_followup_1yr


def test_same_day_episodes_are_retained_with_zero_gap():
    episodes, _, metrics = prepare_metrics(episode_frame(["2020-01-01"] * 2))
    gaps = table1.build_intervisit_gaps(episodes)
    assert metrics.iloc[0].n_clinical_episodes == 2
    assert gaps.iloc[0].gap_zero_days


def test_duplicate_episode_id_is_hard_failure():
    frame = episode_frame(["2020-01-01"] * 2, ["e1", "e1"])
    with pytest.raises(ValueError, match="Duplicate patient_id"):
        table1.prepare_longitudinal_clinical_episodes(frame)


def test_prebaseline_episode_is_audited_and_excluded():
    frame = episode_frame(["2019-01-01", "2020-01-01"], baseline="2020-01-01")
    frame["is_clinical_baseline"] = [False, True]
    episodes, audit, metrics = prepare_metrics(frame)
    assert audit.iloc[0].is_prebaseline
    assert not audit.iloc[0].included_in_primary_followup
    qc = table1.build_followup_qc(episodes, metrics, table1.build_intervisit_gaps(episodes), None)
    assert qc.set_index("qc_check").loc["followup_n_prebaseline_clinical_episodes", "value"] == 1


def test_missing_anchor_is_audited_without_imputation_or_gap():
    episodes, audit, metrics = prepare_metrics(episode_frame(["2020-01-01", None]))
    assert not audit.iloc[1].has_valid_anchor_date
    assert pd.isna(episodes.iloc[1].clinical_anchor_date)
    assert metrics.iloc[0].n_clinical_episodes == 2
    assert metrics.iloc[0].n_dated_clinical_episodes == 1
    assert table1.build_intervisit_gaps(episodes).empty
    qc = table1.build_followup_qc(episodes, metrics, table1.build_intervisit_gaps(episodes), None)
    assert qc.set_index("qc_check").loc["followup_n_clinical_episodes_missing_anchor_date", "value"] == 1


def test_dual_protocol_membership_has_one_overall_row():
    episodes, _, metrics = prepare_metrics(
        episode_frame(["2020-01-01", "2021-01-01"], protocols=["11D", "15D"])
    )
    assert metrics.iloc[0].in_protocol_11d and metrics.iloc[0].in_protocol_15d
    assert len(metrics) == 1
    assert len(episodes) == 2


@pytest.mark.parametrize("value,expected", [
    ("11D", "11D"), ("11-D", "11D"), ("11 D", "11D"),
    ("15D", "15D"), ("15-D", "15D"), ("15 D", "15D"),
    ("11D | 15D", "11D | 15D"),
])
def test_protocol_normalization(value, expected):
    assert table1.normalize_protocol_membership(value) == expected


def test_protocol_resolution_skips_empty_higher_priority_column():
    frame = episode_frame(["2020-01-01", "2021-01-01"])
    frame["source_protocol"] = [None, ""]
    frame["ids__protocol"] = ["11D", "15D"]
    assert table1.resolve_protocol_column(frame) == "ids__protocol"
    episodes, _ = table1.prepare_longitudinal_clinical_episodes(frame, ["p1"])
    assert set(episodes.protocol_membership) == {"11D", "15D"}


def test_gap_iqr_contains_only_q1_to_q3():
    episodes, _, metrics = prepare_metrics(
        episode_frame(["2020-01-01", "2020-06-29", "2021-06-29"])
    )
    summary, _ = table1.build_followup_summary(metrics, table1.build_intervisit_gaps(episodes))
    value = summary.set_index("Indicator").loc["IQR inter-visit gap, days", "Overall"]
    assert value == "226.2–318.8"
    assert "(" not in value


def test_retention_is_monotonic():
    _, _, metrics = prepare_metrics(episode_frame(["2020-01-01", "2026-01-01"]))
    row = metrics.iloc[0]
    assert row.has_followup_5yr
    assert all(row[column] for column in ["has_followup_3yr", "has_followup_2yr", "has_followup_1yr", "has_followup_6mo"])


def test_table1_and_followup_universe_match():
    _, _, metrics = prepare_metrics(episode_frame(["2020-01-01"]))
    baseline = pd.DataFrame({"patient_id": ["p1"]})
    retention = table1.build_retention_table({"Overall": metrics})
    table1.validate_followup_hard_qc(metrics, baseline, retention)
    assert set(metrics.patient_id) == set(baseline.patient_id)

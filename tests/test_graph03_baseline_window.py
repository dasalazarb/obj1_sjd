"""Synthetic tests for the shared calendar and candidate-selection rules."""
from datetime import date

import pandas as pd
import pytest

from src.studies.graph.stage03.s03_baseline_window import (
    AFTER_BASELINE_WINDOW, BASELINE_DAY0, BASELINE_POST_DAY0_TO_6M, DATE_UNRESOLVED,
    MISMATCH, PREBASELINE, VERSION, baseline_window_end, calendar_day, calendar_windows,
    classify_measurement_day, is_eligible_baseline_measurement, policy, require_policy,
    select_calendar_rows,
)
from src.studies.graph.stage03.s03_longitudinal import after_characterization_window


def test_calendar_months_and_mutually_exclusive_bins():
    anchor = '2024-08-31'
    assert baseline_window_end(anchor) == date(2025, 2, 28)
    assert baseline_window_end('2023-08-31') == date(2024, 2, 29)
    observed = pd.Series(['2024-08-30', anchor, '2024-09-01', '2024-09-30', '2024-10-01',
                          '2024-11-29', '2024-11-30', '2025-02-28', '2025-03-01', None])
    anchors = pd.Series([anchor] * len(observed))
    windows = calendar_windows(anchors, observed)
    assert windows.window_bin.tolist() == ['PRE', 'DAY0', 'DAY1_30', 'DAY1_30', 'DAY31_90',
                                           'DAY31_90', 'DAY91_TO_6M', 'DAY91_TO_6M', 'AFTER_6M', 'UNKNOWN']
    assert windows.baseline_offset_days.iloc[:-1].tolist() == [-1, 0, 1, 30, 31, 90, 91, 181, 182]
    assert pd.isna(windows.baseline_offset_days.iloc[-1])
    assert windows.temporal_class.tolist() == [
        PREBASELINE, BASELINE_DAY0, *([BASELINE_POST_DAY0_TO_6M] * 6), AFTER_BASELINE_WINDOW, DATE_UNRESOLVED]
    assert windows.temporal_class.tolist() == [classify_measurement_day(anchor, value) for value in observed]


def test_calendar_days_preserve_written_timezone_and_validate_provenance():
    assert calendar_day('8/31/2024 23:30:12') == date(2024, 8, 31)
    assert calendar_day(pd.Timestamp('2024-08-31T23:50:00-08:00')) == date(2024, 8, 31)
    assert calendar_day('2024-08-31T23:50-0800') == date(2024, 8, 31)
    assert calendar_day('2024-02-30') is None
    assert classify_measurement_day(pd.NA, '2024-08-31') == DATE_UNRESOLVED
    gates = dict(identity_verified=True, collection_verified=True, assay_comparable=True,
                 result_valid=True, provenance_verified=True)
    assert is_eligible_baseline_measurement('2024-08-31', '2025-02-28', **gates)
    for gate in gates:
        assert not is_eligible_baseline_measurement('2024-08-31', '2024-08-31', **dict(gates, **{gate: False}))


def test_earliest_conflict_does_not_fall_through_to_a_later_day():
    rows = pd.DataFrame(dict(patient_id=['a', 'a', 'a', 'b', 'b'],
                             collection_calendar_day=pd.to_datetime(['2024-09-01', '2024-08-31', '2024-08-31',
                                                                      '2024-09-02', '2024-09-02']),
                             value=[8., 7., 9., 1., 1.]))
    selected, conflict = select_calendar_rows(rows, ['patient_id'])
    assert selected.index.tolist() == [1, 2, 3, 4]
    assert conflict.tolist() == [True, True, False, False]
    assert selected.loc[~conflict].patient_id.unique().tolist() == ['b']


def test_longitudinal_sensitivity_keeps_only_visits_after_the_exact_boundary():
    baseline = pd.DataFrame(dict(patient_id=['a'], clinical_baseline_date=['2024-08-31']))
    observations = pd.DataFrame(dict(patient_id=['a'] * 3,
                                     visit_calendar_day=['2024-09-01', '2025-02-28', '2025-03-01'],
                                     time_years=[.01, .5, .51], value=[1., 2., 3.]))
    sensitivity = after_characterization_window(observations, baseline)
    assert sensitivity.value.tolist() == [3.]
    assert sensitivity.time_years.tolist() == [.51]
    assert observations.value.tolist() == [1., 2., 3.]


def test_legacy_policy_cannot_be_used_as_current():
    require_policy(dict(window_policy_version=VERSION))
    assert policy().metadata()['window_policy_sha256'] == policy().sha256
    for settings in ({}, dict(window_policy_version=VERSION, lookback_days=365)):
        with pytest.raises(ValueError, match=MISMATCH):
            require_policy(settings)

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from secondary_comparison_fixtures import synthetic
from src.block_A.secondary_comparison_config import ComparisonConfig
from src.block_A.secondary_comparison_feasibility import feasibility
from src.block_A.secondary_comparison_descriptive import changes, profiles
from src.block_A.secondary_comparison_contract import inference_frame
from src.block_A.secondary_trajectory_analysis import support_table


def test_support_matches_step15_and_gate_keeps_four_groups():
    frame, groups, spec = synthetic(n=12, unequal=True)
    cfg = ComparisonConfig()
    result = feasibility(frame, groups, spec, 3, 'clinical_baseline', cfg)
    expected = support_table(frame, groups, spec, 3)
    pd.testing.assert_frame_equal(result['support'][expected.columns], expected)
    assert result['gate']['status'] == 'ELIGIBLE'
    assert result['windows'].n_patients.max() <= 12
    sparse = frame.loc[~frame.patient_id.str.startswith('fake_group3')]
    result = feasibility(sparse, groups, spec, 3, 'clinical_baseline', cfg)
    assert result['gate']['status'] == 'NOT_ESTIMABLE'
    assert len(result['support']) == 4
    assert result['support'].N_group_total.eq(12).all()
    assert result['opportunities'].missing_percent_valid_opportunities.isna().all()


def test_change_excludes_reference_and_separates_late_clock():
    frame, groups, spec = synthetic(n=2)
    patient = frame.patient_id.iloc[0]
    late = frame.patient_id.eq(patient)
    frame.loc[late, 'reference_date'] += pd.Timedelta(days=365)
    frame.loc[late, 'reference_delay_days'] = 365
    frame.loc[late, 'reference_source'] = 'first_observed_after_clinical_baseline'
    frame.loc[late, 'time_since_variable_reference_years'] -= 1
    frame = frame.loc[~late | frame.time_since_variable_reference_years.ge(0)].copy()
    frame.loc[late.reindex(frame.index), 'is_delta_followup'] = frame.loc[late.reindex(frame.index), 'time_since_variable_reference_years'].gt(0)
    result = changes(frame, spec, ComparisonConfig())['change']
    assert result.n.sum() == frame.is_delta_followup.sum()
    assert result.change_clock.eq('time_since_variable_reference_years').all()
    assert set(result.reference_source) == {'official_clinical_baseline', 'first_observed_after_clinical_baseline'}
    assert not result.mean_time_since_reference.eq(0).any()


def test_singletons_no_false_change_and_no_auc_extrapolation():
    frame, groups, spec = synthetic(n=2)
    singleton = frame.loc[frame.time.eq(0)].copy()
    singleton['is_delta_followup'], singleton['n_variable_dates'], singleton['delta'] = False, 1, np.nan
    result = changes(singleton, spec, ComparisonConfig())
    assert result['change'].empty and result['slopes'].empty
    late = frame.loc[frame.time.ge(1)].copy()
    result = changes(late, spec, ComparisonConfig())
    assert result['auc']['n'].eq(0).all()


def test_preindex_support_is_recomputed_and_official_profile_is_real():
    frame, groups, spec = synthetic(n=12)
    frame['time'] -= 3
    data = inference_frame(frame, 'essdai_index')
    assert data.time.min() == 0 and data.n_analysis_dates.eq(2).all()
    result = feasibility(frame, groups, spec, 3, 'essdai_index', ComparisonConfig())
    assert result['gate']['status'] == 'NOT_ESTIMABLE'  # Only two distinct times remain.
    assert result['support'].preindex_observations_excluded.eq(36).all()
    profiles_table, _ = profiles(frame.loc[frame.time_from_clinical_baseline_years.gt(0)], spec, ComparisonConfig())
    assert profiles_table.loc[profiles_table.profile.eq('official_baseline'), 'n'].eq(0).all()


def test_cumulative_and_mechanical_outcomes_are_descriptive():
    frame, groups, spec = synthetic()
    for special in (replace(spec, expected_longitudinal=False, observation_semantics='cumulative_through_episode'),
                    replace(spec, is_essdai_derived=True), replace(spec, type='ordinal')):
        result = feasibility(frame, groups, special, 3, 'clinical_baseline', ComparisonConfig())
        assert result['gate']['status'] == 'NOT_ESTIMABLE'

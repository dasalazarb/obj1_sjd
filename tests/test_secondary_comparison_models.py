from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from secondary_comparison_fixtures import synthetic
from src.block_A import secondary_comparison_models as models
from src.block_A.secondary_comparison_config import ComparisonConfig
from src.block_A.secondary_comparison_feasibility import feasibility
from src.block_A.secondary_comparison_sensitivity import adjust_tests, scenarios
from src.block_A.secondary_trajectory_analysis import ContractError


def fit(frame=None, groups=None, spec=None, cfg=None):
    if frame is None:
        frame, groups, spec = synthetic()
    cfg = cfg or ComparisonConfig()
    gate = feasibility(frame, groups, spec, 3, 'clinical_baseline', cfg)['gate']
    return models.longitudinal(frame, spec, gate, cfg)


def test_recovers_known_slopes_without_extrapolation():
    result = fit(cfg=ComparisonConfig(prediction_window_years=.5))
    assert result['global'].status.eq('ESTIMATED').all()
    assert result['global'].df.eq(3).all()
    assert result['global'].p.iloc[0] < .001
    contrasts = result['contrasts'].loc[result['contrasts'].multiplicity_role.eq('prespecified')]
    assert contrasts.estimate.tolist() == pytest.approx([.6, -.6, .3], abs=.04)
    assert result['predictions'].time.between(0, 4).all()
    assert result['predictions'].n_observed_local.min() == 0  # Gap support recorded, not interpolated as data.
    assert result['predictions'].loc[result['predictions'].status.eq('NOT_ESTIMABLE'), 'estimate'].isna().all()
    assert not result['contrasts'].time.eq(5).any()


def test_zero_interaction_is_not_created_by_unequal_visits():
    frame, groups, spec = synthetic(n=40, slopes=(0, 0, 0, 0), unequal=True)
    result = fit(frame, groups, spec)
    assert result['global'].status.iloc[0] == 'ESTIMATED'
    slopes = result['contrasts'].loc[result['contrasts'].contrast_type.eq('slope')]
    assert slopes.estimate.abs().max() < .05
    assert result['global'].n_patients.iloc[0] == 160
    assert result['global'].n_observations.iloc[0] > 160


def test_mixed_failure_falls_back_with_attempt_audit(monkeypatch):
    def failure(*args, **kwargs):
        raise np.linalg.LinAlgError('synthetic failure')
    monkeypatch.setattr(models, 'mixed_fit', failure)
    result = fit()
    assert result['attempts'].status.tolist() == ['MODEL_FAILED', 'ESTIMATED']
    assert result['global'].model_family.iloc[0] == 'GEE_Gaussian_robust'
    assert 'LinAlgError' in result['global'].fallback_reason.iloc[0]


@pytest.mark.parametrize('fault', ['constant', 'sparse', 'all_positive', 'ordinal', 'cumulative'])
def test_nonestimable_has_no_fake_pvalue(fault):
    frame, groups, spec = synthetic(binary=fault == 'all_positive')
    if fault == 'constant' or fault == 'all_positive':
        frame['value'] = 1.
    if fault == 'sparse':
        frame = frame.loc[~frame.patient_id.str.startswith('fake_group3')]
    if fault == 'ordinal':
        spec = replace(spec, type='ordinal')
    if fault == 'cumulative':
        spec = replace(spec, expected_longitudinal=False, observation_semantics='cumulative_through_episode')
    result = fit(frame, groups, spec)
    assert result['global'].status.eq('NOT_ESTIMABLE').all()
    assert result['global'].p.isna().all()
    assert result['coefficients'].empty


def test_binary_probabilities_and_separation():
    frame, groups, spec = synthetic(n=50, slopes=(0, .2, -.2, .1), binary=True)
    result = fit(frame, groups, spec)
    assert result['global'].status.iloc[0] == 'ESTIMATED'
    predictions = result['predictions'].dropna(subset=['estimate'])
    assert predictions.estimate.between(0, 1).all()
    assert predictions.ci95_low.between(0, 1).all()
    frame['value'] = frame.time.ge(2).astype(float)
    result = fit(frame, groups, spec)
    assert result['global'].status.iloc[0] == 'NOT_ESTIMABLE'
    assert 'separation' in result['global'].reason.iloc[0]


def test_mixed_thresholds_block_globally():
    frame, groups, spec = synthetic()
    gate = feasibility(frame, groups, spec, 3, 'clinical_baseline', ComparisonConfig())['gate']
    frame.loc[0, 'threshold_abs'] = 5
    with pytest.raises(ContractError, match='mixed thresholds'):
        models.longitudinal(frame, spec, gate, ComparisonConfig())


def test_fdr_separates_threshold_family_and_invalid_pvalues():
    table = pd.DataFrame(dict(variable=list('abcdef'), threshold_abs=[3, 3, 3, 3, 5, 3],
        time_axis='clinical_baseline', scenario='primary', family=['priority']*5+['mechanical'],
        status=['ESTIMATED']*2+['NOT_ESTIMABLE', 'ESTIMATED', 'ESTIMATED', 'ESTIMATED'],
        p=[.01, .04, np.nan, np.inf, .03, .001], is_essdai_derived=[False]*5+[True]))
    result = adjust_tests(table)
    assert result.q_value.iloc[:2].tolist() == pytest.approx([.02, .04])
    assert result.family_n_tests.iloc[:2].eq(2).all()
    assert result.q_value.iloc[4] == .03
    assert result.q_value.iloc[[2, 3, 5]].isna().all()


def test_sensitivity_subcohorts_preserve_group_and_other_clock():
    frame, _, _ = synthetic(n=2)
    original = frame.groupby('patient_id').observed_pattern.first().to_dict()
    for name, data, axis in scenarios(frame, 'clinical_baseline', ComparisonConfig()):
        assert data.groupby('patient_id').observed_pattern.first().to_dict().items() <= original.items()
        assert axis in ('clinical_baseline', 'essdai_index')

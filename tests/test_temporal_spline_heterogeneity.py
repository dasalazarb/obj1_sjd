from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from temporal_spline_fixtures import fixture, evaluated
from src.block_A.temporal_spline_config import SplineConfig
from src.block_A.temporal_spline_feasibility import evaluate
from src.block_A.temporal_spline_models import fit_family
from src.block_A.temporal_spline_contrasts import characterize
from src.block_A.temporal_spline_inference import adjust_fdr, separation_intervals
from src.block_A.secondary_trajectory_analysis import ContractError


def test_six_pairs_distinct_estimands_and_joint_bands():
    ev, spec, cfg = evaluated('crossing')
    fits, _, _ = fit_family(ev, spec, cfg)
    out = characterize(ev, fits, spec, cfg)
    pairs = out['pairs']
    assert pairs.contrast.nunique() == 6
    assert set(pairs.contrast_type) == {'level', 'change', 'velocity'}
    assert pairs.critical_value.min() >= 1.96
    assert pairs.sim95_low.le(pairs.ci95_low+1e-12).all()
    assert pairs.sim95_high.ge(pairs.ci95_high-1e-12).all()
    assert len(out['changes']) == len(out['derivatives'])
    assert out['changes'].loc[out['changes'].time.eq(0), 'estimate'].eq(0).all()
    assert out['characteristics'].status.eq('NOT_ESTIMABLE_TIMING').all()


def test_sparse_group_support_does_not_invent_C():
    d, groups, spec = fixture()
    group = groups.observed_pattern.unique()[-1]
    d = d.loc[~d.observed_pattern.eq(group) | d.time.isin([0., 3.])]
    ev = evaluate(d, groups, spec, SplineConfig(), 'clinical_baseline', 3, 3)
    assert not ev['gate']['eligible_C']
    assert ev['gate']['status'] == 'NOT_ESTIMABLE_SPLINE_GROUP'


def test_unsupported_reference_uses_first_common_supported_grid_and_gaps_do_not_merge():
    d, groups, spec = fixture()
    d = d.loc[~d.time.eq(0)]
    cfg = SplineConfig(simulations=600, prediction_window_years=.25)
    ev = evaluate(d, groups, spec, cfg, 'clinical_baseline', 3, 3)
    fits, _, _ = fit_family(ev, spec, cfg)
    out = characterize(ev, fits, spec, cfg)
    assert out['changes'].reference_time.eq(.25).all()
    bands = pd.DataFrame([dict(contrast='b-a', contrast_type='level', time=t, estimate=1., sim95_low=.1, sim95_high=2., method='test') for t in (0., .25, 1., 1.25)])
    intervals = separation_intervals(bands, .25)
    assert len(intervals) == 2
    assert intervals.start.tolist() == [0., 1.]


def test_fdr_across_all_independent_variables_not_priority_subfamilies():
    rows = [dict(variable=str(i), threshold_abs=3, time_axis='clinical_baseline', scenario='main',
        test='global_temporal_heterogeneity', domain='labs', is_essdai_derived=False, status='ESTIMATED', p=p)
        for i, p in enumerate([.01, .04, .08, .8])]
    rows.append(dict(rows[0], variable='essdai', is_essdai_derived=True, p=.0001))
    d = adjust_fdr(pd.DataFrame(rows))
    np.testing.assert_allclose(d.q_value.iloc[:4], [.04, .08, .08*4/3, .8])
    assert d.family_n_tests.iloc[:4].eq(4).all()
    assert np.isnan(d.q_value.iloc[-1])


def test_fixed_groups_thresholds_and_duplicate_dates_are_contract_errors():
    d, groups, spec = fixture()
    changed = d.copy()
    changed.loc[changed.index[0], 'threshold_abs'] = 5
    with pytest.raises(ContractError):
        evaluate(changed, groups, spec, SplineConfig(), 'clinical_baseline')
    changed = pd.concat([d, d.iloc[:1]], ignore_index=True)
    with pytest.raises(ContractError):
        evaluate(changed, groups, spec, SplineConfig(), 'clinical_baseline')


@pytest.mark.parametrize('kwargs', [dict(type='ordinal'), dict(unit='count'), dict(observation_semantics='historical_ever'), dict(is_essdai_derived=True)])
def test_unsupported_semantics_inventory_without_models(kwargs):
    d, groups, spec = fixture()
    ev = evaluate(d, groups, replace(spec, **kwargs), SplineConfig(), 'clinical_baseline', 3, 3)
    assert not ev['gate']['eligible_A']


def test_sup_t_null_coverage_and_reproducibility():
    from src.block_A.temporal_spline_inference import simultaneous_critical
    rng = np.random.default_rng(1710)
    gradients = rng.normal(size=(90, 8))
    cov = np.eye(8)
    critical = simultaneous_critical(gradients, cov, 5000, 12)
    assert critical == simultaneous_critical(gradients, cov, 5000, 12)
    se = np.sqrt((gradients**2).sum(axis=1))
    null = rng.normal(size=(1000, 8))
    false_separation = (np.abs(null@gradients.T/se).max(axis=1) > critical).mean()
    assert .02 < false_separation < .09  # Software calibration of Gaussian simulation only.


def test_patient_bootstrap_shortlist_uses_same_family(monkeypatch):
    import src.block_A.temporal_spline_models as models
    from src.block_A.temporal_spline_sensitivity import bootstrap_patients
    # GEE provides a deterministic quick validation; each resample retains full histories.
    def no_mixed(*args, **kwargs):
        raise np.linalg.LinAlgError('use robust family for bootstrap validation')
    monkeypatch.setattr(models, 'mixed_fit', no_mixed)
    ev, spec, cfg = evaluated('different', n=14)
    cfg = replace(cfg, bootstrap_replicates=100)
    fits, _, _ = models.fit_family(ev, spec, cfg)
    result = bootstrap_patients(ev, fits, spec, cfg).iloc[0]
    assert result.status == 'VALIDATION_ESTIMATED'
    assert result.successful_replicates == 100
    assert result.bootstrap_sup_t_critical > 1.96

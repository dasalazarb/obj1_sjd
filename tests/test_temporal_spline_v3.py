"""Synthetic regressions for the A prediction repair and post-hoc C3 alternative."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import chi2
from temporal_spline_fixtures import fixture, evaluated, cli
from src.block_A.secondary_comparison_models import PATTERNS
from src.block_A.temporal_spline_config import SplineConfig, BASIS_SPECIFICATIONS
from src.block_A.temporal_spline_design import FrozenBasis
from src.block_A.temporal_spline_feasibility import evaluate
from src.block_A.temporal_spline_contrasts import characterize, function_and_gradient, model_design_and_derivative
from src.block_A.temporal_spline_models import Fit, fit_family, tests as model_tests


def linear_fit():
    return Fit('A', 'GEE_gaussian_robust', np.array([2., 1., 2., 3., .5, .2, -.3, .8]), np.eye(8)*.01)


def test_A_without_basis_supported_grid_and_export(tmp_path):
    ev, spec, cfg = evaluated()
    ev['basis'] = None
    fit = linear_fit()
    out = characterize(ev, {'A': fit}, spec, cfg)
    times = ev['support'].loc[ev['support'].four_group_supported, 'time'].unique()
    assert len(out['curves']) == 4*len(times)
    assert out['curves'].n_observed_local.ge(5).all()
    assert out['curves'].model_family.eq(fit.family).all()
    assert 'sim95_low' not in out['curves']
    assert all(out[name].empty for name in ('changes', 'derivatives', 'pairs', 'bands'))
    assert set(out['registry'].component).issuperset({'B_temporal_functions', 'C_temporal_contrasts'})
    path = tmp_path/'03_group_curves.csv'
    out['curves'].to_csv(path, index=False)
    assert len(pd.read_csv(path)) == 4*len(times)
    assert characterize(ev, {}, spec, cfg)['curves'].empty
    ev['support']['four_group_supported'] = False
    empty = characterize(ev, {'A': fit}, spec, cfg)
    assert empty['curves'].empty
    assert empty['registry'].reason.tolist() == ['no_four_group_supported_grid']
    assert characterize(ev, {}, spec, cfg)['curves'].empty


@pytest.mark.parametrize('binary', [False, True])
def test_A_known_functions_and_gradients_without_basis(binary):
    fit = linear_fit()
    t, reference, eps = 1.2, .25, 1e-6
    ev, spec, cfg = evaluated()
    out = characterize(dict(ev, basis=None), {'A': fit}, replace(spec, type='binary') if binary else spec, cfg)
    if binary:
        assert out['curves'][['estimate', 'ci95_low', 'ci95_high']].apply(lambda s: s.between(0, 1).all()).all()
    for i, group in enumerate(PATTERNS):
        intercept = 2.+(fit.beta[i] if i else 0.)
        slope = .5+(fit.beta[4+i] if i else 0.)
        expected = expit(intercept+slope*t) if binary else intercept+slope*t
        probability = expit(intercept+slope*t)
        for kind in ('level', 'change', 'velocity'):
            value, gradient = function_and_gradient(None, fit, group, t, kind, reference, binary)
            target = expected if kind == 'level' else (probability*(1-probability)*slope if binary else slope) if kind == 'velocity' else expected-(expit(intercept+slope*reference) if binary else intercept+slope*reference)
            assert value == pytest.approx(target, abs=1e-12)
            for j in range(8):
                beta = fit.beta.copy()
                beta[j] += eps
                high = function_and_gradient(None, replace(fit, beta=beta), group, t, kind, reference, binary)[0]
                beta[j] -= 2*eps
                low = function_and_gradient(None, replace(fit, beta=beta), group, t, kind, reference, binary)[0]
                assert gradient[j] == pytest.approx((high-low)/(2*eps), abs=1e-8)
            assert np.isfinite(gradient@fit.covariance@gradient)


def test_A_B_predictions_equal_original_basis_formula():
    ev, spec, cfg = evaluated()
    basis = ev['basis']
    fits = {'A': linear_fit(), 'B': Fit('B', 'GEE_gaussian_robust', np.arange(10)*.1, np.eye(10)*.01)}
    curves = characterize(ev, fits, spec, cfg)['curves']
    for row in curves.itertuples():
        fit = fits[row.model]
        x = basis.row(row.observed_pattern, row.time, row.model)
        assert row.estimate == pytest.approx(x@fit.beta, abs=1e-12)
        assert row.SE == pytest.approx(np.sqrt(x@fit.covariance@x), abs=1e-12)
    for group in PATTERNS:
        for t in [0., .41, 1.3, 3.]:
            x, dx = model_design_and_derivative(None, fits['A'], group, t)
            np.testing.assert_array_equal(x, basis.row(group, t, 'A'))
            np.testing.assert_array_equal(dx, basis.row(group, t, 'A', 1))


@pytest.mark.parametrize('basis_name,k', [('four_knots', 2), ('three_knots', 1)])
def test_basis_dimensions_nesting_derivatives_and_independent_wald(basis_name, k):
    data, groups, spec = fixture()
    cfg = SplineConfig(simulations=600)
    ev = evaluate(data, groups, spec, cfg, 'clinical_baseline', 3, 3, basis_name)
    basis, matrices = ev['basis'], ev['matrices']
    assert basis.as_dict()['nonlinear_df'] == k
    assert basis.as_dict()['df_total'] == 1+k
    assert basis.as_dict()['knots_years'][0] == 0.
    assert len(basis.knots) == k+2
    if k == 1:
        assert basis.knots == (0., .5, 1.)
    assert [np.linalg.matrix_rank(matrices[m]) for m in 'ABC'] == [8, 8+k, 8+4*k]
    np.testing.assert_array_equal(matrices['A'], matrices['B'][:, :8])
    np.testing.assert_array_equal(matrices['B'], matrices['C'][:, :8+k])
    np.testing.assert_allclose(basis.nonlinear([0., 3.], 2), 0, atol=1e-12)
    np.testing.assert_allclose(np.column_stack([np.ones(len(ev['data'])), ev['data'].time]).T@basis.nonlinear(ev['data'].time), 0, atol=1e-10)
    t, eps = np.array([.14, .81, 1.4, 2.8]), 1e-5
    np.testing.assert_allclose(basis.nonlinear(t, 1), (basis.nonlinear(t+eps)-basis.nonlinear(t-eps))/(2*eps), atol=1e-7, rtol=1e-6)
    fits = {m: Fit(m, 'GEE_gaussian_robust', np.arange(x.shape[1])*.1, np.eye(x.shape[1])*.25) for m, x in matrices.items()}
    tests = {r['test']: r for r in model_tests(fits)}
    # Independent explicit positions and chi-square calculation (not a shared index helper).
    indices = {'common_nonlinearity': [8] if k == 1 else [8, 9],
               'group_specific_curvature': [9, 10, 11] if k == 1 else [10, 11, 12, 13, 14, 15],
               'global_temporal_heterogeneity': [5, 6, 7, 9, 10, 11] if k == 1 else [5, 6, 7, 10, 11, 12, 13, 14, 15]}
    for name, positions in indices.items():
        statistic = sum((j*.1)**2/.25 for j in positions)
        assert tests[name]['df'] == len(positions)
        assert tests[name]['statistic'] == pytest.approx(statistic)
        assert tests[name]['p'] == pytest.approx(chi2.sf(statistic, len(positions)))


def test_bases_freeze_identical_sample_and_four_knots_default():
    data, groups, spec = fixture()
    cfg = SplineConfig()
    evaluations = [evaluate(data, groups, spec, cfg, 'clinical_baseline', 3, 3, name) for name in BASIS_SPECIFICATIONS]
    pd.testing.assert_frame_equal(evaluations[0]['data'], evaluations[1]['data'])
    pd.testing.assert_frame_equal(evaluations[0]['support'], evaluations[1]['support'])
    assert FrozenBasis.freeze(data.time).as_dict() == evaluations[0]['basis'].as_dict()


@pytest.mark.parametrize('failure', ['covariance', 'convergence'])
def test_C3_failure_retains_coherent_AB_without_C_inference(monkeypatch, failure):
    import src.block_A.temporal_spline_models as models
    from src.block_A.temporal_spline_contract import metadata
    from src.block_A.temporal_spline_inference import adjust_fdr
    from src.block_A.temporal_spline_results import classify, qc_summary
    data, groups, spec = fixture(n=14)
    cfg = SplineConfig(simulations=600)
    ev = evaluate(data, groups, spec, cfg, 'clinical_baseline', 3, 3, 'three_knots')
    original = models.gee_fit
    def no_mixed(*args):
        raise np.linalg.LinAlgError('synthetic full-family fallback')
    def reject_C(y, x, ids, binary):
        result = original(y, x, ids, binary)
        if x.shape[1] == 12:
            if failure == 'covariance':
                result.cov_params = lambda: np.zeros((12, 12))
            else:
                result.converged = False
        return result
    monkeypatch.setattr(models, 'mixed_fit', no_mixed)
    monkeypatch.setattr(models, 'gee_fit', reject_C)
    meta = dict(metadata(spec, 3, 'clinical_baseline', cfg), scenario='main', sample_fingerprint='synthetic')
    tables, fits = cli.analyze(ev, spec, cfg, meta, True, True)
    assert set(fits) == {'A', 'B'}
    assert {f.family for f in fits.values()} == {'GEE_gaussian_robust'}
    assert tables['01_model_comparison'].loc[lambda d: d.model.eq('C'), 'fit_status'].eq('FIT_FAILED').all()
    tables['02_global_temporal_tests'] = adjust_fdr(tables['02_global_temporal_tests'])
    classified = classify(tables['00_variable_eligibility'], tables['01_model_comparison'], tables['02_global_temporal_tests'], cfg.c_support_policy)
    assert classified.loc[classified.model.eq('C'), ['p', 'q_value']].isna().all().all()
    assert classified.loc[classified.model.eq('C'), 'status'].eq('C_NOT_ESTIMABLE').all()
    assert tables['04_simultaneous_bands'].empty and tables['03_within_group_change'].empty
    summary = qc_summary(tables)
    assert summary['n_C_failed_'+failure] == 1
    assert summary['n_outcomes_with_supported_A_curves'] == 1
    assert summary['n_outcomes_with_supported_C_curves'] == 0


def test_FDR_and_classification_never_mix_bases():
    from src.block_A.temporal_spline_inference import adjust_fdr
    from src.block_A.temporal_spline_results import classify, TEST_MODELS
    rows, eligibility, comparisons = [], [], []
    cfg = SplineConfig()
    for base, values in [('four_knots', [.01, .04]), ('three_knots', [.2, .8])]:
        for i, p in enumerate(values):
            identity = dict(outcome_id=str(i), variable=str(i), domain='labs', outcome_type='numeric', unit='mg/L',
                            basis_specification=base, threshold_abs=3, time_axis='clinical_baseline', scenario='main')
            eligibility.append(dict(identity, interpretation_group='independent', **{
                key: value for m in 'ABC' for key, value in [('eligibility_status_'+m, 'ELIGIBLE'), ('reason_'+m, '')]}))
            for m in 'ABC':
                comparisons.append(dict(identity, model=m, fit_status='ESTIMATED', reason=''))
            for test, m in TEST_MODELS.items():
                rows.append(dict(identity, test=test, model=m, p=p, reason='', status='ESTIMATED', is_essdai_derived=False))
    tests = adjust_fdr(pd.DataFrame(rows))
    assert tests.family_n_tests.eq(2).all()
    np.testing.assert_allclose(tests.loc[tests.test.eq('common_nonlinearity'), 'q_value'], [.02, .04, .4, .8])
    out = classify(pd.DataFrame(eligibility), pd.DataFrame(comparisons), tests, cfg.c_support_policy)
    assert len(out) == 16
    assert not out.duplicated(['outcome_id', 'basis_specification', 'test']).any()
    assert out.loc[out.basis_specification.eq('three_knots'), 'q_value'].ge(.4).all()


def test_A_only_figure_segments_and_labels(tmp_path, monkeypatch):
    from src.block_A.temporal_spline_figures import render
    from src.block_A.temporal_spline_outputs import TABLES
    import matplotlib.axes
    ev, spec, cfg = evaluated()
    support = ev['support'].copy()
    support.loc[support.time.eq(1.), 'four_group_supported'] = False
    out = characterize(dict(ev, support=support, basis=None), {'A': linear_fit()}, spec, cfg)
    tables = {name: pd.DataFrame() for name in TABLES}
    identity = dict(outcome_id='synthetic', variable=spec.variable_id, domain=spec.domain, basis_specification='three_knots')
    tables['00_variable_eligibility'] = pd.DataFrame([dict(identity, interpretation_group='independent', eligible_C=False)])
    tables['00_local_temporal_support'] = support.assign(**identity)
    tables['03_group_curves'] = out['curves'].assign(**identity)
    calls = []
    original = matplotlib.axes.Axes.plot
    def capture(self, x, y, *args, **kwargs):
        if kwargs.get('ls') == ':':
            calls.append((np.asarray(x), kwargs.get('label')))
        return original(self, x, y, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, 'plot', capture)
    paths = render(tables, tmp_path, cfg, 'all', 200, 'synthetic_validation')
    assert any(name.endswith('_atlas.png') for name in paths)
    assert len(calls) == 8  # Two disconnected segments for each of four A curves.
    assert all(not (x.min() < 1. < x.max()) for x, _ in calls)
    assert sum(label is not None and label.endswith(' A') for _, label in calls) == 4


def test_degenerate_basis_keeps_A_with_explicit_nonlinear_gate():
    # Two group-wide times would fail A's minimum distinct-time requirement.
    # Three distinct times with the same pooled median/boundary instead isolate the spline failure.
    data, groups, spec = fixture()
    data = data.loc[data.time.isin([0., 1.5, 3.])].copy()
    # Isolate a freeze failure while preserving the eligibility/sample contract;
    # the end-to-end test also exercises genuine repeated boundary knots.
    from unittest.mock import patch
    with patch.object(FrozenBasis, 'freeze', side_effect=ValueError('degenerate_nonlinear_basis')):
        ev = evaluate(data, groups, spec, SplineConfig(), 'clinical_baseline', 3, 3, 'three_knots')
    assert ev['basis'] is None
    assert ev['gate']['eligible_A'] and not ev['gate']['eligible_B'] and not ev['gate']['eligible_C']
    assert 'nonestimable_nonlinear_basis' in ev['gate']['reason_B']
    assert set(ev['matrices']) == {'A'}


def test_four_knot_numerics_match_frozen_V2_reference(monkeypatch):
    from pathlib import Path
    import src.block_A.temporal_spline_models as models
    reference = json.loads((Path(__file__).parent/'fixtures/temporal_spline_v2_reference.json').read_text())
    monkeypatch.setattr(models, 'mixed_fit', lambda *a: (_ for _ in ()).throw(ValueError('frozen synthetic GEE reference')))
    data, groups, spec = fixture(n=14)
    cfg = SplineConfig(simulations=600)
    ev = evaluate(data, groups, spec, cfg, 'clinical_baseline', 3, 3, 'four_knots')
    fits, _, tests = fit_family(ev, spec, cfg)
    for key in ('lower', 'upper', 'knots', 'projection', 'scales'):
        np.testing.assert_allclose(ev['basis'].as_dict()[key], reference['basis'][key], atol=1e-12, rtol=1e-12)
    for m in 'ABC':
        np.testing.assert_allclose(fits[m].beta, reference['fits'][m]['beta'], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(fits[m].covariance, reference['fits'][m]['covariance'], atol=1e-9, rtol=1e-9)
    pd.testing.assert_frame_equal(pd.DataFrame(tests), pd.DataFrame(reference['tests']), atol=1e-9, rtol=1e-9)
    out = characterize(ev, fits, spec, cfg)
    for name, records in reference['temporal'].items():
        actual = out[name]
        if 'time' in actual:
            actual = actual.loc[actual.time.isin([0., 1.25, 3.])]
        expected = pd.DataFrame(records)
        actual = actual[list(expected)].reset_index(drop=True)  # V3 adds model_family to group curves.
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-9, rtol=1e-9)


def test_C3_full_family_fallback_after_mixed_C_failure(monkeypatch):
    import src.block_A.temporal_spline_models as models
    data, groups, spec = fixture(n=14)
    cfg = SplineConfig(simulations=600)
    ev = evaluate(data, groups, spec, cfg, 'clinical_baseline', 3, 3, 'three_knots')
    original = models.mixed_fit
    def fail_C(y, x, ids):
        if x.shape[1] == 12:
            raise np.linalg.LinAlgError('synthetic C3 failure')
        return original(y, x, ids)
    monkeypatch.setattr(models, 'mixed_fit', fail_C)
    fits, attempts, tests = fit_family(ev, spec, cfg)
    assert set(fits) == set('ABC')
    assert all(f.family == 'GEE_gaussian_robust' for f in fits.values())
    assert {a['model'] for a in attempts if a['model_family'].startswith('MixedLM')} == set('ABC')
    assert all(r['method'] == 'robust_Wald' for r in tests)

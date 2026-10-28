import numpy as np
import pytest
from temporal_spline_fixtures import evaluated
from src.block_A.temporal_spline_models import fit_family


@pytest.mark.parametrize('shape', ['linear', 'common', 'different', 'crossing', 'early_late'])
def test_prespecified_synthetic_model_shapes(shape):
    ev, spec, cfg = evaluated(shape)
    fits, attempts, rows = fit_family(ev, spec, cfg)
    assert set(fits) == set('ABC'), attempts
    assert len({f.family for f in fits.values()}) == 1
    tests = {r['test']: r for r in rows}
    assert all(r['status'] == 'ESTIMATED' for r in rows)
    assert tests['global_temporal_heterogeneity']['df'] == 9
    if shape == 'linear':
        assert tests['common_nonlinearity']['p'] > .01
        assert tests['group_specific_curvature']['p'] > .01
    elif shape == 'common':
        assert tests['common_nonlinearity']['p'] < .001
        assert tests['group_specific_curvature']['p'] > .01
    elif shape in ('different', 'early_late'):
        assert tests['group_specific_curvature']['p'] < .001
        assert tests['global_temporal_heterogeneity']['p'] < .001
    else:
        assert tests['linear_group_time']['p'] < .001


def test_entire_family_gee_fallback_and_no_likelihood_criteria(monkeypatch):
    import src.block_A.temporal_spline_models as m
    ev, spec, cfg = evaluated()
    original = m.mixed_fit
    def fail_C(y, x, ids):
        if x.shape[1] == 16:
            raise np.linalg.LinAlgError('synthetic C failure')
        return original(y, x, ids)
    monkeypatch.setattr(m, 'mixed_fit', fail_C)
    fits, attempts, rows = fit_family(ev, spec, cfg)
    assert set(fits) == set('ABC')
    assert all(f.family.startswith('GEE') and np.isnan(f.aic) and np.isnan(f.loglik) for f in fits.values())
    assert all(r['method'] == 'robust_Wald' for r in rows)
    assert any(a['model_family'].startswith('MixedLM') for a in attempts)


def test_failed_A_never_jumps_to_C(monkeypatch):
    import src.block_A.temporal_spline_models as m
    ev, spec, cfg = evaluated()
    def fail(*args, **kwargs):
        raise ValueError('synthetic failure')
    monkeypatch.setattr(m, 'mixed_fit', fail)
    monkeypatch.setattr(m, 'gee_fit', fail)
    fits, attempts, rows = m.fit_family(ev, spec, cfg)
    assert not fits and not rows
    assert {r['model'] for r in attempts} == {'A'}


def test_binary_gee_uses_probabilities_and_probability_velocity():
    from src.block_A.temporal_spline_contrasts import function_and_gradient
    from scipy.special import expit
    ev, spec, cfg = evaluated('linear', n=40, binary=True)
    fits, attempts, tests = fit_family(ev, spec, cfg)
    assert 'C' in fits, attempts
    assert fits['C'].family == 'GEE_binomial_robust'
    group = ev['data'].observed_pattern.iloc[0]
    t, eps = 1.3, 1e-5
    value, gradient = function_and_gradient(ev['basis'], fits['C'], group, t, 'velocity', binary=True)
    low, _ = function_and_gradient(ev['basis'], fits['C'], group, t-eps, binary=True)
    high, _ = function_and_gradient(ev['basis'], fits['C'], group, t+eps, binary=True)
    assert value == pytest.approx((high-low)/(2*eps), rel=1e-5)
    fit = fits['C']
    for j in (0, 4, 8, 12):
        previous = fit.beta[j]
        fit.beta[j] = previous+eps
        upper, _ = function_and_gradient(ev['basis'], fit, group, t, 'velocity', binary=True)
        fit.beta[j] = previous-eps
        lower, _ = function_and_gradient(ev['basis'], fit, group, t, 'velocity', binary=True)
        fit.beta[j] = previous
        assert gradient[j] == pytest.approx((upper-lower)/(2*eps), abs=1e-7)

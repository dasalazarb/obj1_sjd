"""V2 acceptance contracts; all inputs are synthetic."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from test_temporal_spline_validation import fake_fit, candidate
from temporal_spline_fixtures import evaluated
from src.block_A import temporal_spline_validation as v
from src.block_A.temporal_spline_sensitivity import resample_patients


@pytest.mark.parametrize('model', list('ABC'))
@pytest.mark.parametrize('loss', [False, True])
def test_paired_policies_share_one_fit_per_draw(model, loss, monkeypatch):
    ev, spec, cfg = evaluated(n=10)
    cfg = replace(cfg, bootstrap_replicates=100)
    fit, _ = fake_fit(ev['data'], ev['basis'], model, 'GEE_gaussian_robust', spec, cfg)
    calls = []
    def fitting(*args):
        calls.append(1)
        return fake_fit(*args)
    monkeypatch.setattr(v, 'fit_fixed_family', fitting)
    if loss:
        monkeypatch.setattr(v, 'supported_times', lambda data, times, cfg: times[:-1])
    result = v.bootstrap_summary(ev, fit, spec, cfg, 7)
    assert len(calls) == 100
    assert result['calibration_valid_conditional']
    assert result['calibration_valid_strict'] == (not loss)
    assert result['n_attempted'] == sum(result[k] for k in
        ('n_accepted', 'n_rejected_refit', 'n_rejected_support', 'n_rejected_estimand', 'n_other'))
    assert result['n_accepted_conditional']+result['n_failed_conditional'] == 100
    assert result['n_rejected_support'] == (100 if loss else 0)


@pytest.mark.parametrize('reason,category', [
    ('FIT_FAILED_DESIGN', 'DESIGN'), ('FIT_FAILED_LONGITUDINAL_SUPPORT', 'LONGITUDINAL_SUPPORT'),
    ('FIT_FAILED_NUMERICAL', 'NUMERICAL'), ('FIT_FAILED_CONVERGENCE', 'CONVERGENCE'),
    ('FIT_FAILED_COVARIANCE', 'COVARIANCE'), ('FIT_FAILED_RESIDUAL_QC', 'RESIDUAL_QC'),
    ('FIT_FAILED_BINARY_SUPPORT', 'BINARY'), ('UNRECOGNIZED', 'OTHER')])
def test_exclusive_fit_failure_category(reason, category, monkeypatch):
    ev, spec, cfg = evaluated(n=10)
    fit, _ = fake_fit(ev['data'], ev['basis'], 'A', 'GEE_gaussian_robust', spec, cfg)
    monkeypatch.setattr(v, 'fit_fixed_family', lambda *args: (None, reason))
    result = v.bootstrap_summary(ev, fit, spec, replace(cfg, bootstrap_replicates=100), 7)
    assert result['n_rejected_refit'] == 100 and result['n_rejected_support'] == 0
    assert json.loads(result['refit_failure_category']) == {category: 100}


@pytest.mark.parametrize('bad', ['family', 'covariance', 'target'])
def test_invalid_replica_rejected_by_both_policies(bad, monkeypatch):
    ev, spec, cfg = evaluated(n=10)
    fit, _ = fake_fit(ev['data'], ev['basis'], 'A', 'GEE_gaussian_robust', spec, cfg)
    def fitting(*args):
        replica, reason = fake_fit(*args)
        if bad == 'family':
            replica.family = 'changed'
        elif bad == 'covariance':
            replica.covariance[0, 0] = np.nan
        return replica, reason
    monkeypatch.setattr(v, 'fit_fixed_family', fitting)
    if bad == 'target':
        monkeypatch.setattr(v, 'target_values', lambda *args: np.full(10, np.nan))
    result = v.bootstrap_summary(ev, fit, spec, replace(cfg, bootstrap_replicates=100))
    assert result['n_accepted'] == result['n_accepted_conditional'] == 0
    assert result['n_rejected_refit']+result['n_rejected_estimand'] == 100


def test_draws_exactly_match_v1_and_distinguish_sources():
    ev, spec, cfg = evaluated(n=10)
    actual, diagnostic = v.diagnostic_resample(ev['data'], np.random.default_rng(7), v.original_times(ev), cfg)
    expected = resample_patients(ev['data'], np.random.default_rng(7))
    pd.testing.assert_frame_equal(actual, expected)
    assert diagnostic[0] < diagnostic[1] <= 1
    assert set(actual.columns) == set(ev['data'].columns)


@pytest.mark.parametrize('fragile,calibrated,expected', [
    (True, False, 'FRAGILE'), (True, True, 'FRAGILE'),
    (False, False, 'INCONCLUSIVE'), (False, True, 'SUPPORTED_EXPLORATORY')])
def test_status_precedence(fragile, calibrated, expected):
    loo = dict(n_requested=100, n_successful=100, n_failed=0, material_fragility_detected=fragile,
        standardized_shift_p95=.1, material_direction_fraction=0., band_agreement=1.)
    boot = dict(calibration_valid=calibrated, calibration_valid_conditional=True, band_agreement=1., critical_ratio=1.)
    assert v.scientific_status(candidate('one'), 1., loo, boot) == expected
    dimensions = v.validation_dimensions(dict(loo=[loo], bootstrap=[boot]))
    assert dimensions['bootstrap_calibration_status'] == ('CALIBRATED_STRICT' if calibrated else 'CONDITIONAL_ONLY')


def test_zero_success_is_not_material_fragility():
    loo = dict(n_requested=100, n_successful=0, n_failed=100)
    assert v.scientific_status(candidate('one'), 1., loo, {}) == 'INCONCLUSIVE'


def test_linked_suppression_hides_flags_reasons_and_classification():
    raw = pd.DataFrame([dict(outcome_id='one', n_patients_analytic=3,
        material_fragility_detected=True, calibration_valid=True, calibration_valid_conditional=True,
        loo_stability_status='MATERIAL_INFLUENCE', bootstrap_calibration_status='CALIBRATED_STRICT',
        scientific_validation_status='FRAGILE', status='FRAGILE', validation_reason_codes='MATERIAL_INFLUENCE',
        refit_failure_category='{"DESIGN":1}', support_loss_histogram='{"1":1}')])
    safe = v.suppress_table(raw).iloc[0]
    assert safe.status == 'INCONCLUSIVE'
    assert safe.loo_stability_status == safe.bootstrap_calibration_status == 'DISCLOSURE_SUPPRESSED'
    assert pd.isna(safe.material_fragility_detected) and pd.isna(safe.calibration_valid)
    assert safe.refit_failure_category == safe.support_loss_histogram == '{}'


@pytest.mark.parametrize('policy', ['v1', 'changed_policy'])
def test_resume_rejects_changed_version_or_policy(tmp_path, policy):
    plan = {'tasks': []}
    v.Checkpoints(tmp_path/'cache', {'version': 'validation_global_v2', 'policy': v.BOOTSTRAP_POLICY}, plan)
    context = {'version': 'validation_global_v1' if policy == 'v1' else v.VALIDATION_VERSION,
               'policy': v.BOOTSTRAP_POLICY if policy == 'v1' else policy}
    with pytest.raises(Exception, match='hash mismatch'):
        v.Checkpoints(tmp_path/'cache', context, plan, resume=True)


def test_new_namespace_preserves_historical_paths(tmp_path):
    paths = v.directories(tmp_path, 'three_knots', 3)
    assert all('temporal-spline-v3/validation_global_v2/' in str(p) for p in paths.values())
    assert '07_bootstrap_failure_diagnostics' in v.TABLES


def test_strict_matches_frozen_v1_algorithm(monkeypatch):
    import frozen_step17_v1_bootstrap as historical
    ev, spec, cfg = evaluated(n=10)
    cfg = replace(cfg, bootstrap_replicates=100)
    fit, _ = fake_fit(ev['data'], ev['basis'], 'C', 'GEE_gaussian_robust', spec, cfg)
    monkeypatch.setattr(historical, 'fit_fixed_family', fake_fit)
    monkeypatch.setattr(v, 'fit_fixed_family', fake_fit)
    old = historical.bootstrap_summary(ev, fit, spec, cfg, 7)
    new = v.bootstrap_summary(ev, fit, spec, cfg, 7)
    for key in ('n_successful', 'n_failed', 'calibration_valid', 'bootstrap_sup_t_critical', 'band_agreement'):
        assert old[key] == new[key]


def test_small_source_cells_never_escape_as_support_rates():
    ev, spec, cfg = evaluated(n=10)
    class RepeatedSource:
        def choice(self, pool, count, replace):
            return np.repeat(pool[0], count)
    draw, diagnostic = v.diagnostic_resample(ev['data'], RepeatedSource(), v.original_times(ev), cfg)
    assert draw.patient_id.nunique() == ev['data'].patient_id.nunique()
    assert np.isnan(diagnostic).all()
    assert set(draw.columns) == set(ev['data'].columns)

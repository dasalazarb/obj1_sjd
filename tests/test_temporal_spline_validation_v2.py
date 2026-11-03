"""Synthetic V2 acceptance: fixed targets, shared draws, disclosure and resume."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from temporal_spline_fixtures import evaluated
from test_temporal_spline_validation import fake_fit, candidate, cli
from src.block_A import temporal_spline_validation as v
from src.block_A.temporal_spline_sensitivity import resample_patients
from src.block_A.secondary_trajectory_analysis import ContractError


@pytest.fixture
def setup():
    ev, spec, cfg = evaluated(n=10)
    cfg = replace(cfg, bootstrap_replicates=100)
    fit, _ = fake_fit(ev['data'], ev['basis'], 'A', 'GEE_gaussian_robust', spec, cfg)
    return ev, spec, cfg, fit


def run_draws(setup, monkeypatch, *, loss=False, refitter=None):
    ev, spec, cfg, fit = setup
    # Keep the real patient sampler; deterministic refits isolate policy behavior.
    monkeypatch.setattr(v, 'fit_fixed_family', refitter or (lambda *args: (fit, '')))
    times = v.original_times(ev)
    monkeypatch.setattr(v, 'supported_times', lambda *args: times[:-1] if loss else times)
    return v.bootstrap_summary(ev, fit, spec, cfg, 7)


def test_v2_namespace_and_work_are_disjoint_from_v1(tmp_path, monkeypatch):
    new = v.directories(tmp_path, 'three_knots', 3)
    assert v.VALIDATION_VERSION == 'validation_global_v2'
    with monkeypatch.context() as old:
        old.setattr(v, 'VALIDATION_VERSION', 'validation_global_v1')
        previous = v.directories(tmp_path, 'three_knots', 3)
    assert all('temporal-spline-v3/validation_global_v2/' in str(p) for p in new.values())
    assert not set(new.values()) & set(previous.values())


def test_v1_checkpoint_cannot_be_resumed_even_with_identical_context(tmp_path, monkeypatch):
    plan = v.frozen_plan([candidate('one')])
    with monkeypatch.context() as old:
        old.setattr(v, 'VALIDATION_VERSION', 'validation_global_v1')
        v.Checkpoints(tmp_path/'old', {'seed': 7}, plan)
    with pytest.raises(ContractError, match='hash mismatch'):
        v.Checkpoints(tmp_path/'old', {'seed': 7}, plan, resume=True)


def test_changed_policy_rejects_resume_and_identical_resume_reuses_hash(tmp_path, monkeypatch):
    plan = v.frozen_plan([candidate('one')])
    first = v.Checkpoints(tmp_path/'cache', {'seed': 7}, plan)
    first.save(plan['tasks'][0]['task_id'], {'n_attempted': 100})
    again = v.Checkpoints(first.path, {'seed': 7}, plan, resume=True)
    assert again.load(plan['tasks'][0]['task_id']) == {'n_attempted': 100}
    monkeypatch.setattr(v, 'BOOTSTRAP_DEFINITION', dict(v.BOOTSTRAP_DEFINITION, sensitivity='changed'))
    with pytest.raises(ContractError, match='hash mismatch'):
        v.Checkpoints(first.path, {'seed': 7}, plan, resume=True)


def test_source_unique_diagnostic_does_not_count_copies_as_independent(setup):
    ev, _, cfg, _ = setup
    rng = np.random.default_rng(7)
    before = deepcopy(rng.bit_generator.state)
    drawn = resample_patients(ev['data'], rng)
    after = deepcopy(rng.bit_generator.state)
    metrics = v.source_support_diagnostic(ev['data'], drawn, before, v.original_times(ev), cfg)
    assert rng.bit_generator.state == after
    assert metrics['bootstrap_draw_count'] == 40
    assert metrics['pseudo_local_fraction'] == 1.
    assert 0 < metrics['source_unique_local_fraction'] < 1.
    assert metrics['group_grid_coverage_fraction'] == 1.
    assert metrics['grid_outside_observed_range_years'] == 0.
    assert all(isinstance(value, (int, float)) for value in metrics.values())
    v.audit_payload(metrics, ev['data'].patient_id)


@pytest.mark.parametrize('reason,category', [
    ('FIT_FAILED_DESIGN', 'DESIGN'), ('FIT_FAILED_LONGITUDINAL_SUPPORT', 'LONGITUDINAL_SUPPORT'),
    ('FIT_FAILED_NUMERICAL', 'NUMERICAL'), ('FIT_FAILED_nonconvergence', 'CONVERGENCE'),
    ('FIT_FAILED_singular_or_indefinite_covariance', 'COVARIANCE'),
    ('FIT_FAILED_residual_asymmetry_exceeds_limit', 'RESIDUAL_QC'),
    ('FIT_FAILED_BINARY_SUPPORT', 'BINARY_SUPPORT'), ('NONCOMPARABLE_FAMILY', 'FAMILY'),
    ('unrecognized failure must not be logged', 'OTHER')])
def test_refit_first_obstacle_is_exclusive_and_allowlisted(setup, monkeypatch, reason, category):
    result = run_draws(setup, monkeypatch, loss=True, refitter=lambda *args: (None, reason))
    assert result['n_rejected_refit'] == result['n_attempted'] == 100
    assert result['n_rejected_support'] == result['n_rejected_estimand'] == 0
    assert json.loads(result['refit_failure_category']) == {category: 100}
    assert json.loads(result['stage_counts'])['REFIT_NOT_ESTIMABLE'] == 100
    assert json.loads(result['support_loss_histogram']) == {}
    assert result['failure_diagnosis'] == 'REFIT_QC_DOMINANT'
    assert reason not in result['failure_reasons']


def test_full_support_accepts_both_without_two_refits_per_draw(setup, monkeypatch):
    calls = []
    def refit(*args):
        calls.append(True)
        return setup[-1], ''
    result = run_draws(setup, monkeypatch, refitter=refit)
    assert len(calls) == result['n_refits_attempted'] == 100
    assert result['calibration_valid_strict'] and result['calibration_valid_conditional']
    assert result['n_accepted'] == result['n_accepted_conditional'] == 100
    assert result['bootstrap_sup_t_critical'] == result['bootstrap_sup_t_critical_conditional']
    assert result['bootstrap_calibration_status'] == 'CALIBRATED_STRICT'
    assert result['failure_diagnosis'] == 'NO_REJECTIONS'


def test_support_lost_only_strict_rejects_conditional_is_separate(setup, monkeypatch):
    result = run_draws(setup, monkeypatch, loss=True)
    assert result['n_accepted'] == 0 and result['n_accepted_conditional'] == 100
    assert result['n_rejected_support'] == 100 and result['n_rejected_support_conditional'] == 0
    assert result['bootstrap_calibration_status'] == 'CONDITIONAL_ONLY'
    assert result['support_loss_conditional'] > 0
    assert json.loads(result['support_loss_histogram']) == {'1': 100}
    assert result['failure_diagnosis'] == 'SUPPORT_FILTER_DOMINANT'
    rows = v.bootstrap_policy_rows(result)
    assert [r['bootstrap_policy'] for r in rows] == [v.BOOTSTRAP_POLICY, v.SENSITIVITY_POLICY]
    assert rows[1]['n_accepted'] == 100 and rows[1]['n_failed'] == 0
    assert rows[1]['status'] == 'CALIBRATED_CONDITIONAL_EXPLORATORY'
    for row in rows:
        hidden = v.suppress_table(pd.DataFrame([dict(outcome_id='one', **row)]), sensitive_outcomes=['one'])
        assert hidden.status.iloc[0] == hidden.reason.iloc[0] == 'DISCLOSURE_SUPPRESSED'


@pytest.mark.parametrize('failure', ['family', 'covariance', 'nonfinite', 'changed_model'])
def test_invalid_fit_or_estimand_rejected_in_both_policies(setup, monkeypatch, failure):
    fit = setup[-1]
    changed = replace(fit)
    if failure == 'family': changed.family = 'MixedLM_ML_random_intercept'
    elif failure == 'covariance': changed.covariance = np.zeros_like(fit.covariance)
    elif failure == 'changed_model': changed.model = 'C'
    else: changed.beta = np.full_like(fit.beta, np.nan)
    result = run_draws(setup, monkeypatch, refitter=lambda *args: (changed, ''))
    assert result['n_accepted'] == result['n_accepted_conditional'] == 0
    assert result['n_rejected_refit'] == result['n_rejected_refit_conditional'] == 100


def test_nonfinite_target_is_counted_as_estimand_failure(setup, monkeypatch):
    monkeypatch.setattr(v, 'target_values', lambda rows, *args: np.full(len(rows), np.nan))
    result = run_draws(setup, monkeypatch)
    assert result['n_rejected_estimand'] == result['n_rejected_estimand_conditional'] == 100
    assert result['n_rejected_refit'] == result['n_rejected_support'] == 0


def test_exclusive_partition_and_secondary_diagnostics_do_not_double_count(setup, monkeypatch):
    counter = iter(range(100))
    fit = setup[-1]
    def refit(*args):
        i = next(counter)
        return (None, 'FIT_FAILED_DESIGN') if i < 30 else (fit, '')
    result = run_draws(setup, monkeypatch, loss=True, refitter=refit)
    for row in v.bootstrap_policy_rows(result):
        assert row['n_attempted'] == row['n_accepted']+row['n_failed'] == 100
        assert row['n_failed'] == sum(row[k] for k in ('n_rejected_refit','n_rejected_support','n_rejected_estimand','n_other'))
    assert json.loads(result['stage_counts'])['FIT_COMPARABLE_SUPPORT_LOST'] == 70
    assert result['failure_diagnosis'] == 'SUPPORT_FILTER_DOMINANT'


def test_draw_failures_are_other_not_refits(setup, monkeypatch):
    import src.block_A.temporal_spline_sensitivity as sensitivity
    def fail(*args): raise ContractError('synthetic source identity must not escape')
    monkeypatch.setattr(sensitivity, 'resample_patients', fail)
    result = run_draws(setup, monkeypatch)
    assert result['n_other'] == result['n_other_conditional'] == 100
    assert result['n_refits_attempted'] == result['n_draw_completed'] == 0
    assert 'source identity' not in json.dumps(result)


@pytest.mark.parametrize('model,size', [('A', 10), ('B', 65), ('C', 234)])
def test_frozen_estimand_family_is_identical_in_both_policies(setup, monkeypatch, model, size):
    ev, spec, cfg, _ = setup
    fit, _ = fake_fit(ev['data'], ev['basis'], model, 'GEE_gaussian_robust', spec, cfg)
    rows, _, _ = v.target_family(ev, fit)
    assert len(rows) == size
    if model == 'A': assert len([r for r in rows if r['contrast']]) == 6
    if model == 'B':
        assert {r['observed_pattern'] for r in rows if r['estimand'] == 'shared_curvature'} == {'shared'}
    if model == 'C':
        assert len({r['contrast'] for r in rows}) == 6
        assert {r['estimand'] for r in rows} == {'level','change','velocity'}
    seen = []
    original = v.target_values
    def targets(target_rows, evaluation, refit):
        assert target_rows == rows and evaluation is ev
        seen.append(True)
        return original(target_rows, evaluation, refit)
    monkeypatch.setattr(v, 'target_values', targets)
    result = run_draws((ev, spec, cfg, fit), monkeypatch, loss=True)
    assert len(seen) == 100 and result['n_accepted_conditional'] == 100


def good_loo():
    return dict(n_requested=100, n_successful=100, n_failed=0, standardized_shift_p95=.1,
                material_direction_fraction=0., band_agreement=1., support_fraction_p25=1.)


@pytest.mark.parametrize('calibration', ['INCONCLUSIVE', 'NOT_ESTIMABLE_METHOD'])
def test_material_loo_takes_priority_over_uncalibrated_bootstrap(calibration):
    loo = dict(good_loo(), material_fragility_detected=True)
    boot = dict(status=calibration, calibration_valid=False)
    result = v.validation_dimensions(candidate('one'), 1., loo, boot)
    assert result['scientific_validation_status'] == 'FRAGILE'
    assert result['loo_stability_status'] == 'MATERIAL_INFLUENCE'
    assert not result['bootstrap_strict_calibration_valid']
    assert result['bootstrap_calibration_status'] in ('INCONCLUSIVE_STRICT','NOT_ESTIMABLE_METHOD')
    assert 'LOO_MATERIAL_EXCLUSION_SHIFT' in result['validation_reason_codes']


def test_conditional_only_never_confers_scientific_support():
    boot = dict(calibration_valid=False, calibration_valid_conditional=True)
    result = v.validation_dimensions(candidate('one'), 1., good_loo(), boot)
    assert result['scientific_validation_status'] == 'INCONCLUSIVE'
    assert result['loo_stability_status'] == 'STABLE_WITHIN_DEFINED_LIMITS'
    assert result['bootstrap_calibration_status'] == 'CONDITIONAL_ONLY'


def test_supported_remains_exploratory_and_bad_metrics_never_pass():
    boot = dict(calibration_valid=True, band_agreement=1., critical_ratio=1.)
    assert v.scientific_status(candidate('one'), 1., good_loo(), boot) == 'SUPPORTED_EXPLORATORY'
    assert v.scientific_status(candidate('one'), 1., good_loo(), dict(boot, critical_ratio=np.nan)) == 'INCONCLUSIVE'
    assert v.scientific_status(candidate('one'), 1., dict(good_loo(), band_agreement=np.nan), boot) == 'INCONCLUSIVE'
    failed = dict(good_loo(), n_successful=0, n_failed=100)
    assert v.scientific_status(candidate('one'), 1., failed, boot) == 'INCONCLUSIVE'


def test_binary_consumes_no_bootstrap_draws_preserves_loo(setup, monkeypatch):
    ev, spec, cfg, fit = setup
    monkeypatch.setattr(v, 'fit_fixed_family', lambda *args: pytest.fail('binary bootstrap refit'))
    result = v.bootstrap_summary(ev, fit, replace(spec, type='binary'), cfg)
    assert result['status'] == 'NOT_ESTIMABLE_METHOD' and result['n_attempted'] == 0
    assert v.scientific_status(candidate('one'), 1., dict(good_loo(), material_fragility_detected=True), result) == 'FRAGILE'


@pytest.mark.parametrize('n', [100, 101, 125, 200, 201])
def test_strict_success_threshold_is_unchanged(setup, monkeypatch, n):
    ev, spec, cfg, fit = setup
    result = run_draws((ev, spec, replace(cfg, bootstrap_replicates=n), fit), monkeypatch)
    assert result['required_successes'] == max(100, math.ceil(.8*n))


def test_suppression_hides_partitions_flags_status_and_complements(tmp_path):
    row = dict(outcome_id='one', n_attempted=200, n_accepted=199, n_failed=1,
        failure_reasons='{"REFIT_DESIGN":1}', refit_failure_category='{"DESIGN":1}',
        calibration_valid=True, bootstrap_policy=v.BOOTSTRAP_POLICY, status='FRAGILE',
        reason='Insufficient comparable strict replicates',
        scientific_validation_status='FRAGILE', loo_material_fragility_detected=True,
        loo_stability_status='MATERIAL_INFLUENCE', bootstrap_calibration_status='CALIBRATED_STRICT',
        validation_reason_codes='LOO_MATERIAL_EXCLUSION_SHIFT', failure_diagnosis='REFIT_QC_DOMINANT')
    safe = v.suppress_table(pd.DataFrame([row]))
    assert safe.n_attempted.isna().all() and safe.n_accepted.isna().all() and safe.n_failed.isna().all()
    assert safe.failure_reasons.iloc[0] == safe.refit_failure_category.iloc[0] == '{}'
    assert safe.calibration_valid.iloc[0] is None and safe.loo_material_fragility_detected.iloc[0] is None
    assert safe.loo_stability_status.iloc[0] == safe.bootstrap_calibration_status.iloc[0] == 'DISCLOSURE_SUPPRESSED'
    assert safe.scientific_validation_status.iloc[0] == 'INCONCLUSIVE'
    assert safe.failure_diagnosis.iloc[0] == 'DISCLOSURE_SUPPRESSED'
    pd.testing.assert_frame_equal(safe, v.suppress_table(safe))
    safe['bootstrap_policy'] = v.BOOTSTRAP_POLICY
    summary = v.diagnostic_summary(safe)
    assert all(n is None for n in summary[v.BOOTSTRAP_POLICY]['aggregate_counts'].values())
    cache = v.Checkpoints(tmp_path/'safe', {}, {'tasks':[]})
    cache.save('a'*64, {'rows':safe.to_dict('records')})
    text = (cache.path/('a'*64+'.json')).read_text()
    assert 'MATERIAL_INFLUENCE' not in text and 'LOO_MATERIAL_EXCLUSION_SHIFT' not in text


@pytest.mark.parametrize('key', ['patient_id','source_patient_ids','pseudopatient_id','bootstrap_draw_indices','replicate_coefficients'])
def test_identifying_bootstrap_state_cannot_enter_checkpoints(tmp_path, key):
    cache = v.Checkpoints(tmp_path/'cache', {}, {'tasks':[]})
    with pytest.raises(ContractError): cache.save('a'*64, {key:['invented']})


def test_same_seed_same_rejections_and_no_raw_draws(setup, monkeypatch):
    a = run_draws(setup, monkeypatch, loss=True)
    b = v.bootstrap_summary(*[setup[i] for i in (0,3,1,2)], seed=7)
    assert v.json_safe(a) == v.json_safe(b)
    v.audit_payload(a, setup[0]['data'].patient_id)
    assert not any(pid in json.dumps(v.json_safe(a)) for pid in setup[0]['data'].patient_id)


@pytest.mark.parametrize('model', list('ABC'))
def test_v1_strict_regression_uses_identical_draws_and_original_se(setup, monkeypatch, model):
    ev, spec, cfg, _ = setup
    fit, _ = fake_fit(ev['data'], ev['basis'], model, 'GEE_gaussian_robust', spec, cfg)
    monkeypatch.setattr(v, 'fit_fixed_family', fake_fit)
    # Independent V1 acceptance oracle, preserving RNG order and support rule.
    rows, gradients, estimates = v.target_family(ev, fit)
    se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', gradients, fit.covariance, gradients), 0))
    valid = se > 1e-12
    rng, maxima = np.random.default_rng(17), []
    rejected = 0
    for _ in range(cfg.bootstrap_replicates):
        draw = resample_patients(ev['data'], rng)
        refit, _ = fake_fit(draw, ev['basis'], model, fit.family, spec, cfg)
        if refit is None or v.supported_times(draw, v.original_times(ev), cfg) != v.original_times(ev):
            rejected += 1
            continue
        values = v.target_values(rows, ev, refit)
        maxima.append(float(np.max(np.abs(values[valid]-estimates[valid])/se[valid])))
    result = v.bootstrap_summary(ev, fit, spec, cfg, 17)
    assert result['n_accepted'] == len(maxima) and result['n_failed'] == rejected
    assert result['bootstrap_sup_t_critical'] == pytest.approx(np.quantile(maxima, .95))
    assert result['standardization'] == 'original_standard_errors_not_replicate_studentized'


@pytest.mark.parametrize('shape,model', [('linear','A'), ('common','B'), ('different','C')])
def test_preregistered_synthetic_pilot_real_fixed_family_200_draws(tmp_path, shape, model):
    """Preregistered A/B/C shapes, no outcome selection after validation or p/q."""
    from src.block_A.temporal_spline_models import fit_fixed_family
    ev, spec, cfg = evaluated(shape=shape, n=12)
    cfg = replace(cfg, bootstrap_replicates=200)
    fit, reason = fit_fixed_family(ev['data'], ev['basis'], model, 'GEE_gaussian_robust', spec, cfg)
    assert fit is not None, reason
    loo, influence = v.loo_summary(ev, fit, spec, cfg, 1701)
    boot = v.bootstrap_summary(ev, fit, spec, cfg, 1701)
    assert loo['n_attempted'] == 48 and loo['n_successful']+loo['n_failed'] == 48
    assert boot['n_attempted'] == boot['n_refits_attempted'] == 200
    assert boot['required_successes'] == 160
    for row in v.bootstrap_policy_rows(boot):
        assert row['n_accepted']+row['n_failed'] == 200
        assert row['n_failed'] == sum(row[k] for k in ('n_rejected_refit','n_rejected_support','n_rejected_estimand','n_other'))
    dimensions = v.validation_dimensions(candidate(shape, model), 1., loo, boot)
    tables = [v.suppress_table(pd.DataFrame([dict(outcome_id=shape, **row)]), ev['data'].patient_id)
              for row in (loo, *v.bootstrap_policy_rows(boot), dict(status=dimensions['scientific_validation_status'], **dimensions))]
    if any(table.disclosure_status.eq('DISCLOSURE_SUPPRESSED').any() for table in tables):
        tables = [v.suppress_table(table, ev['data'].patient_id, [shape]) for table in tables]
    for index, table in enumerate(tables):
        path = tmp_path/f'synthetic_{model}_{index}.csv'
        table.to_csv(path, index=False)
        v.audit_payload(table.to_dict('records'), ev['data'].patient_id)
        assert not any(pid in path.read_text() for pid in ev['data'].patient_id)
    assert dimensions['scientific_validation_status'] in ('SUPPORTED_EXPLORATORY','INCONCLUSIVE','FRAGILE')


def test_summary_does_not_pool_identical_draws_and_labels_each_policy(setup, monkeypatch):
    result = run_draws(setup, monkeypatch, loss=True)
    rows = v.bootstrap_policy_rows(result)
    assert rows[0]['failure_diagnosis'] == 'SUPPORT_FILTER_DOMINANT'
    assert rows[1]['failure_diagnosis'] == 'NO_REJECTIONS'
    assert 'conditional original-grid' in rows[1]['reason']
    summary = v.refit_qc_summary(pd.DataFrame(rows))
    assert 'aggregate_counts' not in summary
    assert set(summary['by_policy']) == {v.BOOTSTRAP_POLICY, v.SENSITIVITY_POLICY}
    assert all(s['aggregate_counts']['n_attempted'] == 100 for s in summary['by_policy'].values())


def test_observed_refits_are_suppressed_with_their_linked_partitions():
    accounting = dict(reference_refits_attempted=5, loo_exclusions_attempted=48,
        bootstrap_draws_attempted=200, bootstrap_refits_attempted=200, reference_failure_categories={})
    visible = v.disclosed_accounting(accounting, False)
    assert visible['bootstrap_refits_attempted'] == visible['bootstrap_draws_attempted'] == 200
    hidden = v.disclosed_accounting(accounting, True)
    assert hidden['bootstrap_refits_attempted'] is None and hidden['reference_refits_attempted'] is None
    hidden = v.disclosed_accounting(dict(accounting, reference_failure_categories={'DESIGN':1}), False)
    assert hidden['reference_failure_categories'] is None and hidden['loo_exclusions_attempted'] is None

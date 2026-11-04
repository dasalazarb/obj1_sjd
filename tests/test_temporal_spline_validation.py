"""Synthetic-only contracts for the global Step 17 validation supplement."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from temporal_spline_fixtures import fixture, evaluated, cli as original_cli, args_for
from test_secondary_trajectories_by_essdai_pattern import upstream
from src.block_A.secondary_trajectory_analysis import ContractError, sha256
from src.block_A.secondary_comparison_models import PATTERNS
from src.block_A.temporal_spline_config import SplineConfig, BASIS_SPECIFICATIONS
from src.block_A.temporal_spline_contract import metadata, sample_fingerprint
from src.block_A.temporal_spline_feasibility import evaluate
from src.block_A.temporal_spline_models import Fit, fit_fixed_family
from src.block_A.temporal_spline_outputs import directories as reference_directories
from src.block_A.temporal_spline_sensitivity import resample_patients, bootstrap_patients
from src.block_A import temporal_spline_validation as v

ROOT = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location('validate_step17', ROOT/'src/block_A/17_validate_temporal_heterogeneity.py')
cli = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = cli
module_spec.loader.exec_module(cli)


def meta_for(spec, ev, threshold=3, basis='four_knots'):
    cfg = SplineConfig()
    meta = {k: val for k, val in metadata(spec, threshold, 'clinical_baseline', cfg).items()
            if k in ('outcome_id', 'domain', 'variable', 'outcome_type', 'unit', 'threshold_abs', 'time_axis')}
    return dict(meta, basis_specification=basis, scenario='main', validation_version=v.VALIDATION_VERSION,
        validation_depth='deep', sample_fingerprint=sample_fingerprint(ev['data']),
        n_patients_analytic=ev['data'].patient_id.nunique(), n_observations_analytic=len(ev['data']), seed=17)


def validation_args(upstream, root, *extra):
    return cli.parse_args(['--integrated', str(upstream['integrated']), '--baseline', str(upstream['baseline']),
        '--patterns', str(upstream['patterns']), '--patterns-manifest', str(upstream['manifest']),
        '--output-root', str(root), '--input-origin', 'synthetic_validation', '--no-progress', *extra])


def fake_fit(data, basis, model, family, spec, cfg):
    """Deterministic fitting double; no fitted estimand depends on a patient ID."""
    from src.block_A.secondary_comparison_models import design
    matrix = design(data) if model == 'A' else basis.matrix(data.observed_pattern, data.time, model)
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        return None, 'FIT_FAILED_DESIGN'
    beta = np.linalg.lstsq(matrix, data.value.to_numpy(float), rcond=None)[0]
    return Fit(model, family, beta, np.eye(len(beta))*.04), ''


def test_default_interface_complete_and_cutoff_not_system_date():
    args = cli.parse_args([])
    assert args.depth == 'systematic' and args.threshold == args.basis == 'both'
    assert args.data_cutoff is None and args.bootstrap_replicates == 200
    assert not args.resume and args.max_outcomes is None
    for flags in (['--resume'], ['--data-cutoff', 'today'], ['--bootstrap-replicates', '99'],
                  ['--seed', '-1'], ['--max-patient-refits', '-1', '--depth', 'deep']):
        with pytest.raises(SystemExit):
            cli.parse_args(flags)


def test_disjoint_windows_unique_dates_partition_and_patient_means():
    data, groups, spec = fixture(n=8)
    selected = []
    # Per group: two both, two early-only, two late-only, two neither.
    for group in PATTERNS:
        ids = groups.loc[groups.observed_pattern.eq(group), 'patient_id'].tolist()
        for i, pid in enumerate(ids):
            times = [0., .25, .5, 1.75, 2., 2.5] if i < 2 else [0., .25, .5] if i < 4 else [1.75, 2., 2.5] if i < 6 else [1.]
            selected.append(data.loc[data.patient_id.eq(pid) & data.time.isin(times)].copy())
    frame = pd.concat(selected, ignore_index=True)
    ev = evaluate(frame, groups, spec, SplineConfig(), 'clinical_baseline', 3., 3.)
    result = v.systematic(ev, groups, spec, meta_for(spec, ev), SplineConfig())
    c = result['coverage']
    assert c.n_unique_early.eq(4).all() and c.n_unique_late.eq(4).all()
    assert c.n_unique_both.eq(2).all() and c.n_only_early.eq(2).all()
    assert c.n_only_late.eq(2).all() and c.n_neither.eq(2).all()
    assert c.n_patients_ge3_dates.eq(6).all() and c.distinct_dates_median.eq(3).all()
    assert c.pct_both_of_group.eq(25).all() and c.pct_both_of_early.eq(50).all()
    assert result['sensitive']
    # Diagnostic patient means give each person exactly one weight.
    first_group = PATTERNS[0]
    paired = frame.loc[frame.observed_pattern.eq(first_group)]
    early = paired.loc[paired.time.between(*v.EARLY)].groupby('patient_id').value.mean()
    late = paired.loc[paired.time.between(*v.LATE)].groupby('patient_id').value.mean()
    assert result['changes'].n_paired.eq(len((late-early).dropna())).all()
    assert v.EARLY[1] < v.LATE[0]
    exported = v.suppress_table(c, groups.patient_id, [c.outcome_id.iloc[0]])
    for col in ('n_unique_early', 'n_unique_both', 'n_group_denominator', 'pct_both_of_group', 'followup_span_years_median'):
        assert exported[col].isna().all()


def test_unequal_visits_do_not_reweight_window_descriptives():
    data, groups, spec = fixture(n=10)
    data['value'] = data.patient_id.map({p: i % 10 for i, p in enumerate(groups.patient_id)})
    ev = evaluate(data, groups, spec, SplineConfig(), 'clinical_baseline', 3., 3.)
    result = v.systematic(ev, groups, spec, meta_for(spec, ev), SplineConfig())
    assert result['distributions']['mean'].eq(4.5).all()
    assert result['changes'].median_change.eq(0).all()
    assert result['coverage'].n_unique_both.eq(10).all()
    assert result['coverage'].n_observations_analytic.eq(130).all()


@pytest.mark.parametrize('changes,expected', [({}, 'ESTIMATED'), ({'type': 'binary'}, 'ESTIMATED'),
    ({'type': 'ordinal'}, 'NOT_APPLICABLE_TYPE'), ({'unit': 'count'}, 'NOT_APPLICABLE_TYPE'),
    ({'observation_semantics': 'historical_ever'}, 'NOT_APPLICABLE_TYPE'),
    ({'is_essdai_derived': True}, 'NOT_APPLICABLE_TYPE')])
def test_numeric_binary_ordinal_historical_types(changes, expected):
    data, groups, spec = fixture(n=10)
    spec = replace(spec, **changes)
    if spec.type == 'binary':
        data.value = np.arange(len(data)) % 2
    ev = evaluate(data, groups, spec, SplineConfig(), 'clinical_baseline', 3., 3.)
    result = v.systematic(ev, groups, spec, meta_for(spec, ev), SplineConfig())
    assert result['changes'].status.eq(expected).all()
    assert len(result['coverage']) == 4
    if spec.type == 'binary':
        assert result['changes'].change_scale.eq('late_minus_early_mean_binary_probability').all()
    if expected == 'NOT_APPLICABLE_TYPE':
        assert result['changes'].median_change.isna().all()


def test_empty_outcome_still_has_all_group_registries():
    data, groups, spec = fixture(n=5)
    ev = evaluate(data.iloc[0:0], groups, spec, SplineConfig(), 'clinical_baseline', 3., 3.)
    result = v.systematic(ev, groups, spec, meta_for(spec, ev), SplineConfig())
    assert len(result['coverage']) == len(result['changes']) == 4
    assert result['coverage'].status.eq('INSUFFICIENT_DATA').all()
    assert result['coverage'].n_neither.eq(5).all()


def candidate(oid, model='C', **kwargs):
    return dict(outcome_id=oid, threshold_abs=3, basis_specification='four_knots', model=model,
        time_axis='clinical_baseline', scenario='main', domain='lab', variable='duplicate_name',
        outcome_type='numeric', model_family='GEE_gaussian_robust', eligible_for_deep=True,
        reference_comparable=True, status='DEEP_PENDING', reference_q_value=.8, **kwargs)


def test_queue_all_models_bases_thresholds_homonyms_and_no_significance_filter():
    rows = []
    for basis in BASIS_SPECIFICATIONS:
        for threshold in (3, 5):
            for model in 'ABC':
                for oid, domain in [('one', 'lab'), ('two', 'pro')]:
                    row = candidate(oid, model)
                    row.update(basis_specification=basis, threshold_abs=threshold, domain=domain)
                    rows.append(row)
    plan = v.frozen_plan(rows, seed=7)
    reverse = v.frozen_plan(list(reversed(rows)), seed=7)
    assert plan == reverse and len(plan['tasks']) == 24
    changed_p = [dict(row, reference_q_value=.00001) for row in rows]
    assert v.frozen_plan(changed_p, seed=7) == plan
    assert len({t['task_id'] for t in plan['tasks']}) == 24
    assert set(t['model'] for t in plan['tasks']) == set('ABC')


def test_checkpoint_atomic_idempotence_context_and_corruption(tmp_path):
    plan = v.frozen_plan([candidate('one')])
    context = dict(code='hash1', sample='hash2')
    c = v.Checkpoints(tmp_path/'cache', context, plan)
    task_id = plan['tasks'][0]['task_id']
    assert c.load(task_id) is None
    c.save(task_id, {'status': 'COMPLETE', 'aggregate': 7})
    with pytest.raises(FileExistsError, match='checkpoint exists; use --resume'):
        v.Checkpoints(c.path, context, plan)
    resumed = v.Checkpoints(c.path, context, plan, resume=True)
    assert resumed.load(task_id)['aggregate'] == 7
    with pytest.raises(FileExistsError):
        resumed.save(task_id, {'status': 'COMPLETE'})
    with pytest.raises(ContractError, match='hash mismatch'):
        v.Checkpoints(c.path, dict(code='changed'), plan, resume=True)
    path = c.path/(task_id+'.json')
    record = json.loads(path.read_text())
    record['payload']['aggregate'] = 8
    path.write_text(json.dumps(record))
    with pytest.raises(ContractError, match='hash mismatch'):
        resumed.load(task_id)
    assert not list(c.path.glob('.aggregate-*'))


def test_checkpoint_interrupted_replace_keeps_prior_plan(tmp_path, monkeypatch):
    plan = v.frozen_plan([candidate('one')])
    c = v.Checkpoints(tmp_path/'cache', {}, plan)
    before = (c.path/'plan.json').read_bytes()
    def fail(*args):
        raise OSError('synthetic interrupted replace')
    monkeypatch.setattr(v.os, 'replace', fail)
    with pytest.raises(OSError):
        c.save(plan['tasks'][0]['task_id'], {'status': 'COMPLETE'})
    assert before == (c.path/'plan.json').read_bytes()
    assert not list(c.path.glob('.aggregate-*'))


def test_privacy_columns_values_nested_json_and_complementary_counts(tmp_path):
    ids = ['synthetic-person-101']
    for payload in ({'patient_id': ids[0]}, {'text': 'omitted '+ids[0]}, {'rows': [{'note': ids[0]}]}):
        with pytest.raises(ContractError):
            v.audit_payload(payload, ids)
    with pytest.raises(ContractError):
        v.suppress_table(pd.DataFrame([dict(patient_id=ids[0])]), ids)
    table = pd.DataFrame([dict(outcome_id='one', n_requested=30, n_successful=29, n_failed=1,
        standardized_shift_p95=.1, failure_reasons='{"FIT_FAILED":1}', seed=17)])
    safe = v.suppress_table(table, ids)
    assert safe[['n_requested', 'n_successful', 'n_failed', 'standardized_shift_p95']].isna().all().all()
    assert safe.failure_reasons.iloc[0] == '{}'
    c = v.Checkpoints(tmp_path/'cache', {}, {'tasks': []}, patient_ids=ids)
    with pytest.raises(ContractError):
        c.save('a'*64, {'note': ids[0]})


@pytest.mark.parametrize('model', list('ABC'))
def test_LOO_removes_entire_patient_freezes_basis_and_family(model, monkeypatch):
    ev, spec, cfg = evaluated(n=10)
    fit, _ = fake_fit(ev['data'], ev['basis'], model, 'GEE_gaussian_robust', spec, cfg)
    seen = []
    def refit(data, basis, name, family, variable, config):
        omitted = set(ev['data'].patient_id)-set(data.patient_id)
        assert len(omitted) == 1
        assert len(data) == len(ev['data'])-13
        assert basis is ev['basis'] and family == fit.family and name == model
        assert data.groupby('patient_id').observed_pattern.nunique().eq(1).all()
        seen.append(True)
        return fake_fit(data, basis, name, family, variable, config)
    monkeypatch.setattr(v, 'fit_fixed_family', refit)
    summary, influence = v.loo_summary(ev, fit, spec, cfg, 7)
    assert len(seen) == 40 and summary['n_requested'] == summary['n_successful'] == 40
    assert summary['n_failed'] == 0 and influence
    assert 'patient_id' not in json.dumps([summary, influence])
    assert not any('max' in key or 'min' in key for row in influence for key in row)
    assert summary['estimand_family'] == v.family_name(model)


def test_fixed_family_failure_never_silently_falls_back(monkeypatch):
    import src.block_A.temporal_spline_models as models
    ev, spec, cfg = evaluated(n=10)
    def failure(*args):
        raise ValueError('do not export this internal message')
    def fallback(*args):
        raise AssertionError('forbidden fallback')
    monkeypatch.setattr(models, 'mixed_fit', failure)
    monkeypatch.setattr(models, 'gee_fit', fallback)
    fit, reason = fit_fixed_family(ev['data'], ev['basis'], 'A', 'MixedLM_ML_random_intercept', spec, cfg)
    assert fit is None and reason == 'FIT_FAILED_NUMERICAL'
    fit, reason = fit_fixed_family(ev['data'], ev['basis'], 'A', 'invalid', spec, cfg)
    assert reason == 'NONCOMPARABLE_FAMILY'


def test_fixed_refit_rebuilds_rank_and_rejects_covariance_and_warnings(monkeypatch):
    import src.block_A.temporal_spline_models as models
    ev, spec, cfg = evaluated(n=10)
    fit, reason = fit_fixed_family(ev['data'].loc[ev['data'].time.eq(0)], ev['basis'], 'C',
        'GEE_gaussian_robust', spec, cfg)
    assert fit is None and reason == 'FIT_FAILED_DESIGN'
    original = models.gee_fit
    def bad_cov(*args):
        result = original(*args)
        result.cov_params = lambda: np.zeros((8, 8))
        return result
    monkeypatch.setattr(models, 'gee_fit', bad_cov)
    fit, reason = fit_fixed_family(ev['data'], ev['basis'], 'A', 'GEE_gaussian_robust', spec, cfg)
    assert fit is None and 'covariance' in reason


def test_LOO_counts_all_failures_and_not_comparable(monkeypatch):
    ev, spec, cfg = evaluated(n=10)
    fit, _ = fake_fit(ev['data'], ev['basis'], 'A', 'GEE_gaussian_robust', spec, cfg)
    monkeypatch.setattr(v, 'fit_fixed_family', lambda *args: (None, 'NONCOMPARABLE_FAMILY'))
    summary, influence = v.loo_summary(ev, fit, spec, cfg, 7)
    assert summary['n_failed'] == summary['n_noncomparable'] == summary['n_requested'] == 40
    assert not influence


def test_patient_bootstrap_whole_trajectories_duplicates_and_reproducibility():
    data, groups, _ = fixture(n=8)
    a = resample_patients(data, np.random.default_rng(7))
    b = resample_patients(data, np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby('patient_id').size().eq(13).all()
    assert a.groupby('patient_id').observed_pattern.nunique().eq(1).all()
    assert set(a.patient_id).isdisjoint(data.patient_id)
    assert a.groupby('observed_pattern').patient_id.nunique().eq(8).all()
    # Whole trajectories (including the outcome sequence) exactly match a
    # source patient; replacement admits duplicated sequences.
    source = {tuple(part.value) for _, part in data.groupby('patient_id')}
    draws = [tuple(part.value) for _, part in a.groupby('patient_id')]
    assert all(seq in source for seq in draws) and len(set(draws)) < len(draws)
    corrupted = data.copy()
    corrupted.loc[corrupted.index[0], 'observed_pattern'] = PATTERNS[1]
    with pytest.raises(ContractError):
        resample_patients(corrupted, np.random.default_rng(7))


@pytest.mark.parametrize('model', list('ABC'))
def test_bootstrap_model_specific_calibration_and_determinism(model, monkeypatch):
    ev, spec, cfg = evaluated(n=10)
    cfg = replace(cfg, bootstrap_replicates=100)
    fit, _ = fake_fit(ev['data'], ev['basis'], model, 'GEE_gaussian_robust', spec, cfg)
    monkeypatch.setattr(v, 'fit_fixed_family', fake_fit)
    a = bootstrap_patients(ev, {model: fit}, spec, cfg, model=model, validation_seed=7)
    b = bootstrap_patients(ev, {model: fit}, spec, cfg, model=model, validation_seed=7)
    assert a == b and a['calibration_valid']
    assert a['n_successful'] == a['n_requested'] == 100 and a['n_failed'] == 0
    assert a['estimand_family'] == v.family_name(model)
    assert a['bootstrap_sup_t_critical'] > 0
    targets, _, _ = v.target_family(ev, fit)
    if model == 'A':
        assert len(targets) == 10
    elif model == 'C':
        assert len(targets) == 6*13*3
        assert len({t['contrast'] for t in targets}) == 6


def test_bootstrap_failures_ceiling_and_binary_method_guard(monkeypatch):
    ev, spec, cfg = evaluated(n=10)
    cfg = replace(cfg, bootstrap_replicates=101)
    fit, _ = fake_fit(ev['data'], ev['basis'], 'A', 'GEE_gaussian_robust', spec, cfg)
    monkeypatch.setattr(v, 'fit_fixed_family', lambda *args: (None, 'FIT_FAILED_NUMERICAL'))
    result = v.bootstrap_summary(ev, fit, spec, cfg, 9)
    assert result['n_failed'] == result['n_attempted'] == 101 and not result['calibration_valid']
    assert result['required_successes'] == 100
    cfg = replace(cfg, bootstrap_replicates=201)
    result = v.bootstrap_summary(ev, fit, spec, cfg, 9)
    assert result['required_successes'] == 161  # ceil, not truncated 160
    binary = v.bootstrap_summary(ev, fit, replace(spec, type='binary'), cfg)
    assert binary['status'] == 'NOT_ESTIMABLE_METHOD' and binary['n_attempted'] == 0


def test_support_never_bridges_gaps_or_extends_original_grid():
    ev, spec, cfg = evaluated(n=10)
    reduced = ev['data'].loc[~ev['data'].time.between(1., 2.)]
    times = v.supported_times(reduced, [0., .25, 1.5, 2.75, 3.], cfg)
    assert times == [0., .25, 2.75, 3.]
    fit, _ = fake_fit(ev['data'], ev['basis'], 'C', 'GEE_gaussian_robust', spec, cfg)
    ev = dict(ev, support=ev['support'].loc[ev['support'].time.isin(times)])
    rows, _, _ = v.target_family(ev, fit)
    assert set(r['time'] for r in rows) == set(times)


def test_scientific_classification_ignores_p_q_and_requires_both_deep_components():
    row = candidate('one')
    good = dict(n_requested=100, n_successful=100, n_failed=0, standardized_shift_p95=.1,
        material_direction_fraction=0., band_agreement=1.)
    boot = dict(calibration_valid=True, band_agreement=1., critical_ratio=1.)
    assert v.scientific_status(row, 1.) == 'DEEP_PENDING_RESOURCE_BUDGET'
    assert v.scientific_status(row, 1., good, boot) == 'SUPPORTED_EXPLORATORY'
    fragile = dict(good, standardized_shift_p95=2.)
    row['reference_q_value'] = .0001
    assert v.scientific_status(row, 1., fragile, boot) == 'FRAGILE'
    assert v.scientific_status(row, .1, good, boot) == 'INCONCLUSIVE'
    assert v.scientific_status(row, 1., good, dict(boot, calibration_valid=False)) == 'INCONCLUSIVE'
    assert v.scientific_status(dict(row, reference_comparable=False), 1., good, boot) == 'NOT_COMPARABLE_REFERENCE'


@pytest.fixture
def complete_reference(upstream, tmp_path):
    root = tmp_path/'reference'
    # No outcome subset: exercise the actual full synthetic source catalog.
    original_cli.run(args_for(upstream, root, '--sensitivity', 'none'))
    return root


@pytest.mark.parametrize('validation_version', [v.VALIDATION_VERSION, 'validation_test_v2'])
def test_systematic_all_catalog_four_bundles_immutable_reference_privacy_and_q(complete_reference, upstream, monkeypatch, validation_version):
    monkeypatch.setattr(v, 'VALIDATION_VERSION', validation_version)
    root = complete_reference
    source = root/'synthetic_validation/outputs'
    before = {p: sha256(p) for p in source.rglob('*') if p.is_file()}
    before.update({p: sha256(p) for p in upstream['root'].rglob('*') if p.is_file()})
    monkeypatch.setattr(cli, 'fit_family', lambda *args: pytest.fail('systematic must not fit models'))
    result = cli.run(validation_args(upstream, root))
    assert result['software_qc_status'] == 'PASS' and result['published_bundles'] == 4
    assert result['data_cutoff'] == '2025-01-01'
    published_tables, stale = {}, []
    for basis in BASIS_SPECIFICATIONS:
        for threshold in (3, 5):
            for area, path in v.directories(root/'synthetic_validation', basis, threshold).items():
                if area == 'tables':
                    published_tables.update({p: sha256(p) for p in path.glob('*.csv')})
                marker = path/'stale.txt'
                marker.write_text('old validation output')
                stale.append(marker)
    second = cli.run(validation_args(upstream, root))
    assert second['software_qc_status'] == 'PASS' and second['published_bundles'] == 4
    assert all(not p.exists() for p in stale)
    assert all(sha256(p) == value for p, value in published_tables.items())
    assert not list(source.rglob('validation-autoreplace-*'))
    for basis in BASIS_SPECIFICATIONS:
        for threshold in (3, 5):
            source_dirs = reference_directories(root/'synthetic_validation', 'clinical_baseline', threshold, 'main', basis_specification=basis)
            paths = v.directories(root/'synthetic_validation', basis, threshold)
            original = pd.read_csv(source_dirs['tables']/'00_variable_eligibility.csv')
            inventory = pd.read_csv(paths['tables']/'07_outcome_validation_inventory.csv')
            assert len(inventory) == 3*len(original)
            assert set(inventory.outcome_id) == set(original.outcome_id)
            assert inventory.reference_comparable.all()
            tests = pd.read_csv(source_dirs['tables']/'02_global_temporal_tests.csv')
            for row in inventory.itertuples():
                name = dict(A='linear_group_time', B='common_nonlinearity', C='global_temporal_heterogeneity')[row.model]
                t = tests.loc[tests.outcome_id.eq(row.outcome_id) & tests.test.eq(name)].iloc[0]
                assert row.reference_q_value == t.q_value or pd.isna(row.reference_q_value) and pd.isna(t.q_value)
                assert row.reference_family_n_tests == t.family_n_tests
            for name in v.TABLES:
                table = pd.read_csv(paths['tables']/(name+'.csv'))
                assert set(v.BASE_COLUMNS) <= set(table)
                v.audit_payload(table.to_dict('records'), upstream['p'].patient_id)
                assert set(table.outcome_id) == set(original.outcome_id)
            manifest = json.loads((paths['qc']/'run_manifest.json').read_text())
            assert manifest['software_qc_status'] == 'PASS' and manifest['code_hashes']
            for artifact in manifest['artifacts']:
                area = 'qc' if artifact['path'].startswith('qc/') else 'tables'
                relative = artifact['path'][3:] if area == 'qc' else artifact['path']
                assert sha256(paths[area]/relative) == artifact['sha256']
            coverage = pd.read_csv(paths['tables']/'07_patient_temporal_coverage.csv')
            assert len(coverage) == 4*len(original)
    assert all(sha256(p) == before[p] for p in before)


def test_invalid_reference_hash_scope_cutoff_and_outcome_sample(complete_reference, upstream):
    root = complete_reference
    with pytest.raises(ContractError, match='data_cutoff'):
        cli.run(validation_args(upstream, root, '--data-cutoff', '2026-01-01'))
    ref = v.authenticate_reference(root/'synthetic_validation', 'four_knots', 3, 'synthetic_validation')
    oid = ref['tables']['00_variable_eligibility'].outcome_id.iloc[0]
    ref['tables']['00_variable_eligibility'].loc[0, 'sample_fingerprint'] = 'wrong'
    data, groups, spec = fixture(n=6)
    ev = evaluate(data, groups, spec, SplineConfig(), 'clinical_baseline', 3, 3)
    meta = meta_for(spec, ev)
    meta['outcome_id'] = oid
    ok, reason = v.reference_comparison(ref, ev, spec, meta, {})
    assert not ok and 'sample_fingerprint' in reason
    paths = reference_directories(root/'synthetic_validation', 'clinical_baseline', 3, 'main')
    path = paths['tables']/'02_global_temporal_tests.csv'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ContractError, match='artifact hash'):
        v.authenticate_reference(root/'synthetic_validation', 'four_knots', 3, 'synthetic_validation')


def test_deep_zero_budget_checkpoint_resume_and_backup_without_rewriting_V3(complete_reference, upstream):
    root = complete_reference
    args = validation_args(upstream, root, '--depth', 'deep', '--max-patient-refits', '0')
    first = cli.run(args)
    checkpoints = list((root/'synthetic_validation/work').rglob('plan.json'))
    assert len(checkpoints) == 1
    plan_before = checkpoints[0].read_bytes()
    resumed = cli.run(validation_args(upstream, root, '--depth', 'deep', '--max-patient-refits', '0', '--resume'))
    assert resumed['n_deep_complete'] == 0
    assert checkpoints[0].read_bytes() == plan_before
    assert not list((root/'synthetic_validation/outputs').rglob('validation-autoreplace-*'))
    backup = root/'synthetic_validation/outputs/backups/17_nonlinear_temporal_heterogeneity/validation_global_v1/test'
    args = validation_args(upstream, root, '--depth', 'deep', '--max-patient-refits', '0', '--resume', '--backup-root', str(backup))
    second = cli.run(args)
    assert first['n_deep_complete'] == second['n_deep_complete'] == 0
    assert checkpoints[0].read_bytes() == plan_before
    assert len(list(backup.rglob('validation_summary.json'))) == 4
    with pytest.raises(FileExistsError, match='checkpoint exists; use --resume'):
        cli.run(validation_args(upstream, root, '--depth', 'deep', '--max-patient-refits', '0'))
    assert checkpoints[0].read_bytes() == plan_before
    args.seed += 1
    with pytest.raises(ContractError, match='hash mismatch'):
        cli.run(args)


def test_deep_budget_completed_task_resume_and_reconstruction_gate(upstream, tmp_path, monkeypatch):
    data, groups, spec = fixture('different', n=10)
    original_load = original_cli.load_inputs
    def load(args):
        x, b, p, specs, catalog, inputs, manifest = original_load(args)
        return x, b, p, {spec.canonical_column: spec}, catalog, inputs, manifest
    def fixed(patterns, threshold):
        return groups.assign(threshold_abs=threshold)
    def measure(x, fixed_groups, *args):
        return data.assign(threshold_abs=int(fixed_groups.threshold_abs.iloc[0])), 'AVAILABLE', [], spec.unit
    for module in (original_cli, cli):
        monkeypatch.setattr(module, 'load_inputs', load)
        monkeypatch.setattr(module, 'fixed_groups', fixed)
        monkeypatch.setattr(module, 'measurements', measure)
    root = tmp_path/'deep'
    original_cli.run(args_for(upstream, root, '--sensitivity', 'none'))
    refs = {p: sha256(p) for p in (root/'synthetic_validation/outputs').rglob('*') if p.is_file()}
    calls = []
    def loo(ev, fit, spec, cfg, seed):
        calls.append(('LOO', fit.model))
        return dict(model=fit.model, status='COMPLETE', reason='', n_requested=40, n_successful=40,
            n_failed=0, standardized_shift_p95=.1, material_direction_fraction=0., band_agreement=1.), []
    def boot(ev, fits, spec, cfg, *, model, validation_seed):
        calls.append(('BOOT', model))
        return dict(model=model, status='CALIBRATED', reason='', n_requested=100, n_successful=100,
            n_failed=0, calibration_valid=True, band_agreement=1., critical_ratio=1.)
    monkeypatch.setattr(v, 'loo_summary', loo)
    monkeypatch.setattr(cli, 'bootstrap_patients', boot)
    first = cli.run(validation_args(upstream, root, '--depth', 'deep', '--bootstrap-replicates', '100',
        '--max-patient-refits', '146'))
    assert first['n_deep_candidates'] == 12
    assert first['n_deep_complete'] == 1 and first['n_deep_pending'] == 11
    assert len(calls) == 2
    backup = root/'synthetic_validation/outputs/backups/17_nonlinear_temporal_heterogeneity/validation_global_v1/test'
    resumed = cli.run(validation_args(upstream, root, '--depth', 'deep', '--bootstrap-replicates', '100',
        '--max-patient-refits', '0', '--resume', '--backup-root', str(backup)))
    assert resumed['n_deep_complete'] == 1 and len(calls) == 2
    final = cli.run(validation_args(upstream, root, '--depth', 'deep', '--bootstrap-replicates', '100',
        '--all-fitted', '--resume', '--backup-root', str(backup)))
    assert final['n_deep_complete'] == 12 and final['n_deep_pending'] == 0
    assert len(calls) == 24
    assert all(sha256(p) == h for p, h in refs.items())
    for basis in BASIS_SPECIFICATIONS:
        for threshold in (3, 5):
            paths = v.directories(root/'synthetic_validation', basis, threshold)
            classified = pd.read_csv(paths['tables']/'07_scientific_validation_classification.csv')
            assert classified.status.eq('SUPPORTED_EXPLORATORY').all()
            manifest = json.loads((paths['qc']/'run_manifest.json').read_text())
            assert manifest['software_qc_status'] == 'PASS'
    # Same-family reference reproduction also checks the published coefficients.
    ref = v.authenticate_reference(root/'synthetic_validation', 'four_knots', 3, 'synthetic_validation')
    ev = evaluate(data, groups, spec, SplineConfig(simulations=600), 'clinical_baseline', 3, 3)
    oid = meta_for(spec, ev)['outcome_id']
    ref['tables']['01_model_coefficients'].loc[0, 'estimate'] += 100
    _, comparable = cli._reproduce_fit(ref, ev, spec, SplineConfig(simulations=600), oid)
    assert not comparable['A']


@pytest.mark.parametrize('field,value', [('version', 'temporal-spline-v4'),
    ('input_origin', 'canonical'), ('c_support_policy', 'wrong'),
    ('fdr_selection_scope', 'explicit_variable_subset'), ('code_hashes', {'bad': {'sha256': 'invalid'}})])
def test_invalid_reference_manifest_contracts_block(complete_reference, field, value):
    root = complete_reference/'synthetic_validation'
    path = reference_directories(root, 'clinical_baseline', 3, 'main')['qc']/'run_manifest.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match='NOT_COMPARABLE_REFERENCE'):
        v.authenticate_reference(root, 'four_knots', 3, 'synthetic_validation')


@pytest.mark.parametrize('persistent_backup', [False, True])
def test_validation_staging_rollback_preserves_source_and_snapshot(complete_reference, upstream, monkeypatch, persistent_backup):
    from src.block_A import secondary_comparison_outputs as outputs
    root = complete_reference
    cli.run(validation_args(upstream, root))
    before = {p: sha256(p) for p in (root/'synthetic_validation/outputs').rglob('*') if p.is_file()}
    before.update({p: sha256(p) for p in upstream['root'].rglob('*') if p.is_file()})
    rename = outputs.os.replace
    installed = []
    def fail_after_first(source, destination):
        if 'step17-validation-stage-' in str(source):
            installed.append(destination)
            if len(installed) == 2:
                raise OSError('synthetic publication interruption')
        return rename(source, destination)
    monkeypatch.setattr(outputs.os, 'replace', fail_after_first)
    backup = root/'synthetic_validation/outputs/backups/17_nonlinear_temporal_heterogeneity/validation_global_v1/rollback'
    flags = ('--backup-root', str(backup)) if persistent_backup else ()
    with pytest.raises(OSError):
        cli.run(validation_args(upstream, root, *flags))
    assert len(installed) == 2 and all(sha256(p) == value for p, value in before.items())
    if not persistent_backup:
        backups = list((root/'synthetic_validation/outputs').rglob('validation-autoreplace-*'))
        assert len(backups) == 1
        backup = backups[0]
    assert len(list(backup.rglob('validation_summary.json'))) == 4
    assert not list((root/'synthetic_validation/work').glob('step17-validation-stage-*'))


def test_A_validation_stays_estimable_without_a_spline_basis():
    ev, spec, cfg = evaluated(n=12)
    fit, reason = fit_fixed_family(ev['data'], None, 'A', 'GEE_gaussian_robust', spec, cfg)
    assert fit is not None, reason
    rows, gradients, values = v.target_family(dict(ev, basis=None), fit)
    assert len(rows) == 10 and gradients.shape == (10, 8)
    assert np.isfinite(values).all()


def test_low_support_or_single_material_deletion_prevents_scientific_support():
    row = candidate('one')
    good = dict(n_requested=100, n_successful=100, n_failed=0, standardized_shift_p95=.1,
        material_direction_fraction=0., band_agreement=1.)
    boot = dict(calibration_valid=True, band_agreement=1., critical_ratio=1.)
    assert v.scientific_status(row, 1., dict(good, support_fraction_p25=.5), boot) == 'INCONCLUSIVE'
    assert v.scientific_status(row, 1., dict(good, material_fragility_detected=True), boot) == 'FRAGILE'


def test_finite_full_family_q_is_unchanged_for_selected_deep_jobs():
    ev, spec, cfg = evaluated(n=10)
    meta = meta_for(spec, ev)
    models = pd.DataFrame([dict(outcome_id=meta['outcome_id'], model=m,
        fit_status='ESTIMATED', model_family='GEE_gaussian_robust') for m in 'ABC'])
    tests = pd.DataFrame([dict(outcome_id=meta['outcome_id'], test=name, q_value=q,
        family_n_tests=287, status='ESTIMATED') for name, q in [
            ('linear_group_time', .72), ('common_nonlinearity', .37),
            ('global_temporal_heterogeneity', .016), ('group_specific_curvature', .032)]])
    reference = {'tables': {'01_model_comparison': models, '02_global_temporal_tests': tests}}
    rows = v.inventory_rows(reference, ev, spec, meta, True, '', 'deep')
    assert [r['reference_q_value'] for r in rows] == [.72, .37, .016]
    assert all(r['reference_family_n_tests'] == 287 for r in rows)
    assert rows[-1]['reference_group_curvature_q'] == .032
    selected = v.frozen_plan(rows, [meta['outcome_id']], seed=7)['tasks'][:1]
    assert selected and rows[-1]['reference_q_value'] == .016
    assert rows[-1]['reference_family_n_tests'] == 287


def test_fixed_family_longitudinal_support_cannot_be_relaxed():
    ev, spec, cfg = evaluated(n=10)
    reduced = ev['data'].loc[ev['data'].patient_id.ne(ev['data'].patient_id.iloc[0])]
    fit, reason = fit_fixed_family(reduced, ev['basis'], 'A', 'GEE_gaussian_robust', spec, cfg)
    assert fit is None and reason == 'FIT_FAILED_LONGITUDINAL_SUPPORT'


def test_bootstrap_failure_tally_is_not_a_patient_cell():
    table = pd.DataFrame([dict(outcome_id='one', n_patients_analytic=50, n_observations_analytic=200,
        n_requested=200, n_attempted=200, n_successful=199, n_failed=1, calibration_valid=True,
        bootstrap_sup_t_critical=3., failure_reasons='{"FIT_FAILED_NUMERICAL":1}')])
    safe = v.suppress_table(table)
    assert safe.n_failed.iloc[0] == 1 and safe.n_requested.iloc[0] == 200
    assert safe.bootstrap_sup_t_critical.iloc[0] == 3.


def test_actual_GEE_LOO_and_bootstrap_use_the_reference_family():
    ev, spec, cfg = evaluated(n=12)
    fit, reason = fit_fixed_family(ev['data'], ev['basis'], 'A', 'GEE_gaussian_robust', spec, cfg)
    assert fit is not None, reason
    loo, influence = v.loo_summary(ev, fit, spec, cfg, 8)
    assert loo['n_requested'] == 48
    assert loo['n_successful']+loo['n_failed'] == 48 and loo['n_successful'] > 30
    assert influence and loo['n_noncomparable'] == 0
    result = v.bootstrap_summary(ev, fit, spec, replace(cfg, bootstrap_replicates=100), 8)
    assert result['n_attempted'] == 100 and result['n_successful']+result['n_failed'] == 100
    assert result['n_successful'] > 80
    assert result['calibration_valid'] == (result['n_successful'] >= result['required_successes'])


def test_absent_diagnostic_window_does_not_invalidate_full_horizon_histories():
    data, groups, spec = fixture(n=12)
    data = data.loc[data.time.ge(.75)]
    ev = evaluate(data, groups, spec, SplineConfig(), 'clinical_baseline', 3., 3.)
    result = v.systematic(ev, groups, spec, meta_for(spec, ev), SplineConfig())
    assert result['coverage'].n_unique_early.eq(0).all()
    assert result['coverage'].n_patients_ge2_dates.eq(12).all()
    assert result['minimum_overlap'] == 0. and result['minimum_continuity_fraction'] == 1.
    good = dict(n_requested=48, n_successful=48, n_failed=0, standardized_shift_p95=.1,
        material_direction_fraction=0., band_agreement=1.)
    boot = dict(calibration_valid=True, band_agreement=1., critical_ratio=1.)
    assert v.scientific_status(candidate('one'), result['minimum_continuity_fraction'], good, boot) == 'SUPPORTED_EXPLORATORY'

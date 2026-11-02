"""Regression contract for the uniform exploratory C support revision (T01–T15)."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from temporal_spline_fixtures import cli, fixture, evaluated
from src.block_A.secondary_trajectory_analysis import ContractError
from src.block_A.temporal_spline_config import SplineConfig, C_SUPPORT_POLICY, VERSION, ANALYSIS_ROLE
from src.block_A.temporal_spline_contract import metadata
from src.block_A.temporal_spline_design import FrozenBasis
from src.block_A.temporal_spline_feasibility import evaluate
from src.block_A.temporal_spline_outputs import annotate, directories, preflight, safe_table
from src.block_A.temporal_spline_results import classify, TEST_MODELS
from src.block_A.temporal_spline_inference import adjust_fdr


def meta_for(spec, cfg, total):
    return dict(metadata(spec, 3, 'clinical_baseline', cfg), scenario='main', n_patients=total,
                analysis_role=ANALYSIS_ROLE, c_support_policy=C_SUPPORT_POLICY)


def test_annotation_local_counts_and_semantic_aliases():
    out = annotate(pd.DataFrame([dict(n_patients=8, reference_policy='common_grid', family='Gaussian')]),
                   dict(n_patients=100, reference_policy='variable_baseline', family='priority_labs'))
    assert out.n_patients.tolist() == [8] and out.n_patients_analysis.tolist() == [100]
    assert out.inference_reference_policy.tolist() == ['common_grid']
    assert out.variable_reference_policy.tolist() == ['variable_baseline']
    assert out.outcome_family.tolist() == ['Gaussian'] and out.inference_family.tolist() == ['priority_labs']
    for frame, meta in [(pd.DataFrame([dict(status='ESTIMATED')]), dict(status='primary')),
                        (pd.DataFrame([dict(n_patients_analysis=8)]), dict(n_patients=100)),
                        (pd.DataFrame([dict(reference_policy='a', inference_reference_policy='b')]), {})]:
        with pytest.raises(ContractError, match='collision'):
            annotate(frame, meta)


def test_four_group_export_counts_and_window_suppression(tmp_path):
    counts = pd.DataFrame(dict(observed_pattern=list('abcd'), n_patients=[89, 19, 27, 15]))
    out = cli.disclosure_table(annotate(counts, dict(n_patients=150)))
    out.to_csv(tmp_path/'counts.csv', index=False)
    saved = pd.read_csv(tmp_path/'counts.csv')
    assert saved.n_patients.tolist() == [89, 19, 27, 15]
    assert saved.n_patients_analysis.eq(150).all()
    small = cli.disclosure_table(annotate(pd.DataFrame([dict(n_patients=3, percent_group_measured=30.,
        proportion_measured=.3)]), dict(n_patients=100)))
    assert small.n_patients.isna().all() and small.percent_group_measured.isna().all()
    assert small.proportion_measured.isna().all()
    safe_table(small)
    with pytest.raises(ContractError):
        safe_table(pd.DataFrame(dict(patient_id=['not-exportable'])))


def two_date_histories(n=26):
    data, groups, spec = fixture(n=n)
    kept = []
    for _, group in data.groupby('observed_pattern', sort=False):
        for j, (_, patient) in enumerate(group.groupby('patient_id', sort=False)):
            kept.append(patient.loc[patient.time.isin([j % 13 / 4, (j+6) % 13 / 4])])
    return pd.concat(kept, ignore_index=True), groups, spec


@pytest.mark.parametrize('threshold,domain', [(3, 'labs'), (5, 'serology')])
def test_short_individual_histories_allow_identifiable_C(threshold, domain, monkeypatch):
    import src.block_A.temporal_spline_models as models
    monkeypatch.setattr(models, 'mixed_fit', lambda *a: (_ for _ in ()).throw(ValueError('GEE validation')))
    data, groups, spec = two_date_histories()
    data['threshold_abs'] = threshold
    spec = replace(spec, domain=domain)
    cfg = SplineConfig(simulations=600)
    ev = evaluate(data, groups, spec, cfg, 'clinical_baseline', 3, 3)
    gate = ev['gate']
    assert ev['counts'].n_three_dates.eq(0).all() and gate['limited_three_date_histories']
    assert gate['eligible_A'] and gate['eligible_B'] and gate['eligible_C'], gate
    assert gate['n_supported_grid_times'] >= 4 and gate['supported_grid_span_years'] >= 1
    assert gate['support_policy'] == C_SUPPORT_POLICY
    assert all(d['design_rank'] == d['n_columns'] for d in ev['designs'])
    fits, attempts, tests = models.fit_family(ev, spec, cfg)
    assert 'C' in fits, attempts
    assert len(tests) == 4 and all(np.isfinite(t['p']) for t in tests)


def test_three_grid_times_rejected_with_specific_reason():
    data, groups, spec = fixture()
    ev = evaluate(data, groups, spec, SplineConfig(grid_step=1.5), 'clinical_baseline', 3, 3)
    assert ev['gate']['eligible_B'] and not ev['gate']['eligible_C']
    assert ev['gate']['n_supported_grid_times'] == 3
    assert 'insufficient_c_supported_grid_points' in ev['gate']['reason_C']


def restricted_support(last):
    data, groups, spec = fixture()
    anchor_ids = data.groupby('observed_pattern').patient_id.first()
    data = data.loc[data.time.le(last) | (data.patient_id.isin(anchor_ids) & data.time.eq(3))]
    cfg = SplineConfig(simulations=600, prediction_window_years=.25)
    return evaluate(data, groups, spec, cfg, 'clinical_baseline', 3, 3), spec, cfg


def test_four_points_with_span_under_year_rejected():
    ev, _, _ = restricted_support(.75)
    gate = ev['gate']
    assert gate['eligible_B'] and gate['n_supported_grid_times'] == 4
    assert gate['supported_grid_span_years'] == .75
    assert 'insufficient_c_supported_grid_span_years' in gate['reason_C']
    assert 'insufficient_c_supported_regions' in gate['reason_C']
    assert 'insufficient_c_supported_annual_windows' in gate['reason_C']


def test_two_regions_two_windows_are_sufficient_and_gaps_unpublished(monkeypatch):
    import src.block_A.temporal_spline_models as models
    from src.block_A.temporal_spline_contrasts import characterize
    monkeypatch.setattr(models, 'mixed_fit', lambda *a: (_ for _ in ()).throw(ValueError('GEE validation')))
    ev, spec, cfg = restricted_support(1.25)
    gate = ev['gate']
    assert gate['n_supported_regions'] == gate['n_supported_annual_windows'] == 2
    assert gate['eligible_C'], gate
    fits, attempts, tests = models.fit_family(ev, spec, cfg)
    assert 'C' in fits, attempts
    out = characterize(ev, fits, spec, cfg)
    supported = set(ev['support'].loc[ev['support'].four_group_supported, 'time'])
    for key in ('curves', 'changes', 'derivatives', 'pairs', 'bands'):
        assert set(out[key].time) <= supported
        assert out[key].time.max() == 1.25


@pytest.mark.parametrize('failure', ['rank', 'condition'])
def test_invalid_C_design_blocks_C_without_blocking_AB(monkeypatch, failure):
    original = FrozenBasis.matrix
    def invalid(self, patterns, times, model):
        matrix = original(self, patterns, times, model)
        if model == 'C':
            if failure == 'rank':
                matrix[:, -1] = matrix[:, -2]
            else:
                matrix[:, -1] *= 1e-12
        return matrix
    monkeypatch.setattr(FrozenBasis, 'matrix', invalid)
    data, groups, spec = fixture()
    gate = evaluate(data, groups, spec, SplineConfig(), 'clinical_baseline', 3, 3)['gate']
    assert gate['eligible_A'] and gate['eligible_B'] and not gate['eligible_C']
    assert 'singular_or_ill_conditioned_C_design' in gate['reason_C']


@pytest.mark.parametrize('failure', ['covariance', 'convergence', 'B_failure'])
def test_fit_failures_never_publish_C_tests(monkeypatch, failure):
    import src.block_A.temporal_spline_models as models
    monkeypatch.setattr(models, 'mixed_fit', lambda *a: (_ for _ in ()).throw(ValueError('GEE validation')))
    original = models.gee_fit
    def bad_fit(y, matrix, ids, binary):
        if failure == 'B_failure' and matrix.shape[1] == 10:
            raise ValueError('invalid lower model')
        result = original(y, matrix, ids, binary)
        if matrix.shape[1] == 16:
            if failure == 'covariance':
                result.cov_params = lambda: np.zeros((16, 16))
            elif failure == 'convergence':
                result.converged = False
        return result
    monkeypatch.setattr(models, 'gee_fit', bad_fit)
    ev, spec, cfg = evaluated(n=14)
    tables, fits = cli.analyze(ev, spec, cfg, meta_for(spec, cfg, ev['gate']['n_patients']), True, True)
    assert 'A' in fits and 'C' not in fits
    tests = adjust_fdr(tables['02_global_temporal_tests'])
    assert tests.loc[tests.model.eq('C'), ['p', 'q_value']].isna().all().all()
    classified = classify(tables['00_variable_eligibility'], tables['01_model_comparison'], tests, C_SUPPORT_POLICY)
    assert classified.loc[classified.model.eq('C'), 'fit_status'].eq('FIT_FAILED').all()
    assert classified.loc[classified.test.eq('group_specific_curvature'), 'status'].eq('C_NOT_ESTIMABLE').all()
    if failure == 'B_failure':
        assert not tables['01_fit_attempts'].model.eq('C').any()


def test_AB_retained_and_numerically_unchanged_by_annotation(monkeypatch):
    import src.block_A.temporal_spline_models as models
    monkeypatch.setattr(models, 'mixed_fit', lambda *a: (_ for _ in ()).throw(ValueError('GEE validation')))
    data, groups, spec = fixture(n=14)
    cfg = SplineConfig(simulations=600, grid_step=1.5)
    ev = evaluate(data, groups, spec, cfg, 'clinical_baseline', 3, 3)
    raw_fits, _, raw_tests = models.fit_family(ev, spec, cfg)
    tables, fitted = cli.analyze(ev, spec, cfg, meta_for(spec, cfg, ev['gate']['n_patients']), True, False)
    assert set(fitted) == {'A', 'B'}
    for model in 'AB':
        np.testing.assert_array_equal(fitted[model].beta, raw_fits[model].beta)
        np.testing.assert_array_equal(fitted[model].covariance, raw_fits[model].covariance)
        coefficients = tables['01_model_coefficients'].loc[lambda d: d.model.eq(model)]
        np.testing.assert_array_equal(coefficients.estimate, raw_fits[model].beta)
        np.testing.assert_array_equal(coefficients.SE, np.sqrt(np.diag(raw_fits[model].covariance)))
    tests = adjust_fdr(tables['02_global_temporal_tests'])
    for raw in raw_tests:
        assert tests.loc[tests.test.eq(raw['test']), 'p'].iloc[0] == raw['p']
    classified = classify(tables['00_variable_eligibility'], tables['01_model_comparison'], tests, C_SUPPORT_POLICY)
    assert classified.loc[classified.model.isin(list('AB')), 'test_status'].eq('ESTIMATED').all()
    assert classified.loc[classified.model.eq('C'), ['p', 'q_value']].isna().all().all()


def test_homonyms_classify_independently_without_domain_FDR_split():
    eligibility, models, tests = [], [], []
    for domain, p in [('labs', .01), ('serology', .20)]:
        identity = dict(outcome_id=domain+'/shared', variable='shared', domain=domain, outcome_type='numeric', unit='mg/L')
        eligibility.append(dict(**identity, interpretation_group='independent', **{
            k: v for m in 'ABC' for k, v in [('eligible_'+m, True), ('eligibility_status_'+m, 'ELIGIBLE'), ('reason_'+m, '')]}))
        for model in 'ABC':
            models.append(dict(**identity, model=model, fit_status='ESTIMATED', reason=''))
        for test, model in TEST_MODELS.items():
            tests.append(dict(**identity, model=model, test=test, status='ESTIMATED', reason='', p=p,
                is_essdai_derived=False, threshold_abs=3, time_axis='clinical_baseline', scenario='main'))
    tests = adjust_fdr(pd.DataFrame(tests))
    assert tests.family_n_tests.eq(2).all()
    out = classify(pd.DataFrame(eligibility), pd.DataFrame(models), tests, C_SUPPORT_POLICY)
    assert len(out) == 8 and not out.duplicated(['outcome_id', 'test']).any()
    specific = out.loc[out.test.eq('group_specific_curvature')].set_index('domain')
    assert specific.loc['labs', 'status'] == 'EXPLORATORY_NONLINEAR_GROUP_HETEROGENEITY'
    assert specific.loc['serology', 'status'] == 'NO_EXPLORATORY_NONLINEAR_EVIDENCE'
    np.testing.assert_allclose(specific.q_value, [.02, .20])
    with pytest.raises(ContractError, match='duplicate'):
        classify(pd.DataFrame(eligibility), pd.DataFrame(models), pd.concat([tests, tests.iloc[:1]]), C_SUPPORT_POLICY)
    wrong_units = tests.copy()
    wrong_units.loc[0, 'unit'] = 'unrelated'
    with pytest.raises(ContractError, match='mismatched'):
        classify(pd.DataFrame(eligibility), pd.DataFrame(models), wrong_units, C_SUPPORT_POLICY)


@pytest.mark.parametrize('kwargs', [dict(minimum_c_supported_regions=3), dict(minimum_c_supported_grid_points=3),
    dict(minimum_prediction_patients=4), dict(minimum_c_supported_grid_span_years=.5), dict(c_support_policy='ad_hoc')])
def test_named_policy_cannot_be_silently_retuned(kwargs):
    with pytest.raises(ValueError, match=C_SUPPORT_POLICY):
        SplineConfig(**kwargs)


def test_version_namespace_and_backup_guard(tmp_path):
    dest = directories(tmp_path, 'clinical_baseline', 3, 'main')
    assert VERSION in dest['tables'].parts
    old = tmp_path/'outputs/tables/blockA/17_nonlinear_temporal_heterogeneity/clinical_baseline/delta3/main'
    old.mkdir(parents=True)
    preflight(tmp_path, dest, {})
    dest['tables'].mkdir(parents=True)
    with pytest.raises(FileExistsError):
        preflight(tmp_path, dest, {})
    with pytest.raises(ContractError, match='backup'):
        preflight(tmp_path, dest, {}, overwrite=True)


def test_suppressed_group_cannot_be_reconstructed_from_analysis_total():
    data, groups, spec = fixture(n=14)
    last_group = groups.observed_pattern.unique()[-1]
    keep = groups.loc[groups.observed_pattern.eq(last_group), 'patient_id'].iloc[:3]
    data = data.loc[~data.observed_pattern.eq(last_group) | data.patient_id.isin(keep)]
    cfg = SplineConfig(simulations=600)
    ev = evaluate(data, groups, spec, cfg, 'clinical_baseline', 3, 3)
    tables, _ = cli.analyze(ev, spec, cfg, meta_for(spec, cfg, ev['gate']['n_patients']), False, False)
    for name in ('00_group_support', '00_variable_eligibility', '00_aggregated_observations'):
        published = cli.disclosure_table(tables[name])
        assert published.n_patients_analysis.isna().all()
        if name == '00_aggregated_observations':
            small_windows = published.loc[published.observed_pattern.eq(last_group)]
            assert small_windows.status.eq('SUPPRESSED_SMALL_CELL').all()
            assert small_windows[['mean', 'median', 'q25', 'q75']].isna().all().all()
        if name == '00_group_support':
            assert published.loc[published.observed_pattern.eq(last_group), 'n_patients'].isna().all()
        if name == '00_variable_eligibility':
            assert published.n_patients.isna().all()
            assert json.loads(published.n_per_group.iloc[0])[last_group] is None
        pd.testing.assert_frame_equal(cli.disclosure_table(published), published)


def test_local_support_counts_distinct_patients_at_every_published_point():
    data, groups, spec = fixture(n=14)
    last_group = groups.observed_pattern.unique()[-1]
    keep = groups.loc[groups.observed_pattern.eq(last_group), 'patient_id'].iloc[:4]
    data = data.loc[~data.observed_pattern.eq(last_group) | data.patient_id.isin(keep) | ~data.time.between(.5, 1.5)]
    ev = evaluate(data, groups, spec, SplineConfig(prediction_window_years=.5), 'clinical_baseline', 3, 3)
    row = ev['support'].loc[lambda d: d.observed_pattern.eq(last_group) & d.time.eq(1.)].iloc[0]
    assert row.n_observed_local == 4 and not row.four_group_supported
    # Those four patients each contribute several rows; observations cannot stand in for patients.
    assert len(data.loc[data.observed_pattern.eq(last_group) & data.time.between(.75, 1.25)]) > 5


def test_homonym_sensitivity_matches_only_same_canonical_outcome():
    from src.block_A.temporal_spline_sensitivity import consistency
    from src.block_A.temporal_spline_outputs import TABLES
    rows = []
    for domain, p in [('labs', .01), ('serology', .20)]:
        for threshold in (3, 5):
            rows.append(dict(outcome_id=domain+'/shared', variable='shared', domain=domain, outcome_type='numeric', unit='mg/L',
                threshold_abs=threshold, time_axis='clinical_baseline', scenario='main', test='linear_group_time', model='A',
                status='ESTIMATED', reason='', p=p, q_value=p, sample_fingerprint=domain,
                n_patients_analysis=100 if domain == 'labs' else 80))
    tables = {name: pd.DataFrame() for name in TABLES}
    tables['02_global_temporal_tests'] = pd.DataFrame(rows)
    out = consistency({(3, 'clinical_baseline', 'main'): tables})
    assert len(out) == 2 and out.same_estimation_rows.all()
    assert out.loc[out.domain.eq('labs'), 'primary_q'].tolist() == [.01]
    assert out.loc[out.domain.eq('serology'), 'primary_q'].tolist() == [.20]

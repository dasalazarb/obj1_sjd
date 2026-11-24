"""Graph03g contracts with synthetic frozen inputs and original lab records."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from src.studies.graph._pipeline import digest
from src.studies.graph.stage03 import s03_n3_model_readiness_audit as G
from src.studies.graph.stage03 import s03_n3_exploratory as E
from src.studies.graph.stage03 import s03_n3_exploratory_models as M
from src.studies.graph.stage03 import s03_n3_exploratory_source as S
from tests.test_graph03_n3_auto import automatic
from tests.test_graph03_n3_calendar_day import calendar_originals, write_btris
from tests.test_graph03_n3_measurement_source import originals
from tests.test_graph03_n3_exploratory import (
    ANA, CELL, HEP2, POLICY, RF, RF_TEXT, URINE, bind_lineage, current, paths_before, update_graph03,
)


def stop_estimation(*args, **kwargs):
    raise AssertionError('03g must never run an estimator or the exploratory orchestrator')


def read_output(directory, name):
    return pd.read_csv(directory / f'03g_{name}.csv', keep_default_na=False)


@pytest.fixture
def published(current, monkeypatch):
    repo, paths, _, phasee = current
    # Frozen registry for RF and ANA plus representation-specific RF lineage.
    # Strip the test review records before 03e to emulate the pending state.
    bind_lineage(repo, paths)
    config = repo / E.CONFIGS / E.VALIDATION
    validation = yaml.safe_load(config.read_text())
    validation['characterization']['independence_reviews'] = []
    validation['n3_validation']['prespecified_outcomes'] = []
    validation['anchors'].extend(dict(source_column=source, type='binary') for source in (HEP2, URINE))
    config.write_text(yaml.safe_dump(validation))
    update_graph03(repo, paths)
    graph03 = paths['graph03_run']
    metadata = graph03 / G.GRAPH03['summary']
    summary = json.loads(metadata.read_text())
    summary['sha256_by_artifact'] = {name: digest(graph03 / name)
                                     for role, name in G.GRAPH03.items() if role != 'summary'}
    metadata.write_text(json.dumps(summary))

    eda = pd.read_parquet(paths['eda_long']).assign(unit='IU/mL', lab_date='2020-01-01')
    hep2 = eda.assign(canonical_analyte=S.LAB_SOURCES[HEP2], lab_date='2020-07-02',
                     order_identifier=eda.order_identifier + '_HEP2', order_name_original='SYNTHETIC_HEP2',
                     result_text='positive')
    eda = pd.concat([eda, hep2], ignore_index=True)
    write_btris(paths, eda, repo.parent)
    eda.to_parquet(paths['eda_long'], index=False)
    result = E.run_exploratory_auto(repo)
    assert result['n_model_pairs_estimable'] == 0
    assert result['graph03_input'] == paths['graph03_run'].name, result['data_issues']
    monkeypatch.setattr(E, '_estimate', stop_estimation)
    monkeypatch.setattr(E, '_analyze', stop_estimation)
    monkeypatch.setattr(M, '_estimate', stop_estimation)
    return repo, paths, phasee, phasee.parent / G.DESTINATION


def test_summaries_use_aggregates_and_preserve_published_zero_and_suppression(published, monkeypatch):
    repo, paths, phasee, target = published
    baseline = paths['baseline']
    manifest = G.validate_release(paths['graph02_release'])[0]
    membership = baseline.parent / G.artifact_paths(manifest)['membership']
    read = pd.read_csv

    def no_clinical_reads(path, *args, **kwargs):
        assert Path(path) not in {baseline, membership}
        return read(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_csv', no_clinical_reads)
    monkeypatch.setattr(pd, 'read_parquet', stop_estimation)
    monkeypatch.setattr(E, 'current_source_inputs', stop_estimation)
    # A published suppressed coverage cell may encode several distinct conditions.
    coverage_path = phasee / '03e_coverage.csv'
    coverage = read(coverage_path)
    mask = (coverage.representation_id.eq('R0') & coverage.outcome.eq(ANA) & coverage.stratum.eq(S.PRIMARY)
            & coverage.provenance.eq(S.ORIGINAL) & coverage.definition_role.eq('PRIMARY'))
    coverage.loc[mask, ['status', 'n_measured', 'n_total', 'n_missing', 'measurement_fraction']] = (
        G.SUPPRESSED, np.nan, np.nan, np.nan, np.nan)
    coverage.to_csv(coverage_path, index=False)

    assert G.run_audit(repo, mode='summaries', strict=True)['estimators_run'] == 0
    matrix = read_output(target, 'readiness_matrix').set_index('outcome')
    assert set(matrix.index) == {RF, RF_TEXT, ANA, HEP2, URINE}
    assert matrix.approval_gate_status.eq('INDEPENDENCE_REVIEW_PENDING').all()
    assert matrix.loc[RF, 'graph03e_reason'] == 'INDEPENDENCE_REVIEW_PENDING'
    assert matrix.loc[RF, 'structural_independence_classification'] == 'STRUCTURALLY_INDEPENDENT_REVIEW_PENDING'
    assert 'independence_review_record' in matrix.loc[RF, 'missing_review_requirements']
    assert 'prespecified_outcome_for_representation_and_policy' in matrix.loc[RF, 'missing_review_requirements']
    assert matrix.loc[ANA, 'primary_measurement_status'] == G.SUPPRESSED
    assert matrix.loc[ANA, 'model_A_support_status'] == G.SUPPRESSED
    assert matrix.loc[HEP2, 'primary_measurement_status'] == 'ZERO_ELIGIBLE_MEASUREMENTS'
    assert matrix.loc[HEP2, 'source_funnel_status'] in {
        G.UNKNOWN, 'NO_LINKED_CANDIDATES_IN_PRIMARY_WINDOW'}
    assert matrix.loc[URINE, 'structural_independence_classification'] == 'LINEAGE_UNRESOLVED'


def test_secure_recompute_is_private_reproducible_and_preserves_inputs(published):
    repo, paths, phasee, target = published
    source_before = digest(paths['step10']), digest(paths['eda_long'])
    before = paths_before(paths), {path: digest(path) for path in phasee.iterdir() if path.is_file()}
    assert G.run_audit(repo, mode='secure-recompute', representation='all', strict=True)['estimators_run'] == 0
    matrix = read_output(target, 'readiness_matrix').set_index(G.PAIRS)
    assert matrix.loc['R0', HEP2].source_funnel_status == 'OUTSIDE_0_TO_6M'
    assert matrix.loc['R2', RF_TEXT].structural_independence_classification == M.DIRECT
    assert matrix.loc['R2', RF].structural_independence_classification == M.PROXY
    assert matrix.loc['R2', RF].topology_status == 'R2_EXPLORATORY_UNSTABLE'
    assert matrix.loc['R0', RF_TEXT].hypothesis_family == matrix.loc['R0', RF].hypothesis_family
    assert matrix.loc['R0', RF_TEXT].rf_primary_priority == 'QUALITATIVE_IF_NUMERIC_NOT_ESTIMABLE'
    assert matrix.loc['R0', RF].model_A_support_status == matrix.loc['R0', RF].model_B_support_status == G.PASS
    assert matrix.loc['R0', RF].readiness_status == 'STATISTICALLY_PLAUSIBLE_PENDING_REVIEW'

    outputs = {path.name: path.read_text() for path in target.iterdir()}
    assert set(outputs) == G.OUTPUT_FILES - {'03g_failure.json'}
    assert target.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in target.iterdir())
    assert all(not any(secret in contents for secret in ('SYNTHETIC_', '2020-01-01', '2020-07-02', str(repo),
                                                          'p_incremental', 'n_measured'))
               for contents in outputs.values())
    summary = json.loads(outputs['03g_summary.json'])
    assert summary['sha256_before'] == summary['sha256_after']
    G.run_audit(repo, mode='secure-recompute', representation='all', strict=True)
    assert outputs == {path.name: path.read_text() for path in target.iterdir()}
    assert before == (paths_before(paths), {path: digest(path) for path in phasee.iterdir() if path.is_file()})
    assert source_before == (digest(paths['step10']), digest(paths['eda_long']))


def run_cli(repo, *args, config=None):
    env = os.environ.copy()
    env.pop('GRAPH03G_CONFIG', None)
    if config is not None:
        env['GRAPH03G_CONFIG'] = str(config)
    script = repo / 'src/studies/graph/03g_graph_n3_model_readiness_audit.py'
    return subprocess.run([sys.executable, str(script), *args], cwd=repo, capture_output=True, text=True, env=env)


def test_cli_runs_without_flags_and_selects_secure_settings_from_yaml(published):
    repo, _, _, target = published
    default = run_cli(repo)
    assert default.returncode == 0, default.stderr
    assert json.loads(default.stdout)['mode'] == 'summaries'
    summary = json.loads((target / '03g_summary.json').read_text())
    assert summary['representations'] == ['R0']
    assert summary['strict'] is True

    config = target.parent / 'graph03g.yaml'
    config.write_text(yaml.safe_dump(dict(mode='secure-recompute', representation='all', output_dir='secure-audit')))
    secure = run_cli(repo, config=config)
    assert secure.returncode == 0, secure.stderr
    assert json.loads(secure.stdout)['mode'] == 'secure-recompute'
    rows = read_output(config.parent / 'secure-audit', 'readiness_matrix').set_index(G.PAIRS)
    assert set(rows.index.get_level_values('representation_id')) == {'R0', 'R1', 'R2', 'R3', 'R4'}
    assert rows.loc['R0', HEP2].source_funnel_status == 'OUTSIDE_0_TO_6M'

    rejected = run_cli(repo, '--mode', 'summaries')
    assert rejected.returncode == 2
    assert 'sin argumentos' in rejected.stderr


@pytest.mark.parametrize(('change', 'reason'), [
    ('window', 'AUDIT_WINDOW_POLICY_MISMATCH'),
    ('release', 'AUDIT_GRAPH03E_ANCESTRY_MISMATCH'),
    ('duplicate', 'AUDIT_DUPLICATE_PUBLISHED_PAIR'),
    ('missing_columns', 'AUDIT_INPUT_SCHEMA_OR_INTEGRITY_ERROR'),
    ('overlap', 'AUDIT_OUTPUT_OVERLAPS_INPUT'),
    ('invalid_config', 'AUDIT_INVALID_CONFIG'),
])
def test_cli_stops_on_tampered_contract_without_echoing_paths(published, change, reason):
    repo, _, phasee, target = published
    path = phasee / '03e_summary.json'
    if change in {'window', 'release'}:
        metadata = json.loads(path.read_text())
        key = 'window_policy_version' if change == 'window' else 'graph02_release_sha256'
        metadata[key] = 'SYNTHETIC_CORRUPT'
        path.write_text(json.dumps(metadata))
    if change in {'duplicate', 'missing_columns'}:
        path = phasee / '03e_independence_audit.csv'
        df = pd.read_csv(path)
        df = pd.concat([df, df.iloc[[0]]]) if change == 'duplicate' else df.drop(columns='classification')
        df.to_csv(path, index=False)
    config = None
    if change in {'overlap', 'invalid_config'}:
        config = repo / 'graph03g.yaml'
        settings = {'output_dir': str(phasee)} if change == 'overlap' else {'strict': 'SYNTHETIC_CORRUPT'}
        config.write_text(yaml.safe_dump(settings))
    run = run_cli(repo, config=config)
    assert run.returncode == 1 and reason in run.stderr
    assert all(secret not in run.stderr + run.stdout for secret in ('SYNTHETIC_', str(repo)))
    assert not target.exists()


@pytest.mark.parametrize(('total', 'group_two', 'events', 'reason'), [
    (29, 14, None, 'INSUFFICIENT_COMPLETE_CASE_COMMUNITY_SUPPORT'),
    (30, 7, None, 'INSUFFICIENT_COMPLETE_CASE_COMMUNITY_SUPPORT'),
    (30, 8, None, ''),
    (30, 15, 4, 'INSUFFICIENT_BINARY_COMMUNITY_SUPPORT'),
    (30, 15, 5, ''),
    (30, 15, 11, 'INSUFFICIENT_BINARY_COMMUNITY_SUPPORT'),
    (30, 15, 10, ''),
])
def test_support_matches_03e_boundaries(total, group_two, events, reason):
    frame = support_frame(total, group_two)
    if events is not None:
        for _, group in frame.groupby('community_id'):
            frame.loc[group.index, 'value'] = (np.arange(len(group)) < events).astype(float)
        frame['demo__age_at_baseline'] = 40.0
    numeric = events is None
    engine, prepared = M._model_support(frame, numeric, POLICY['models'], CELL)
    diagnostics, ready = G._support(frame, numeric, POLICY['models'], CELL, False)
    assert engine.get('reason', '') == reason
    if events in (4, 11):
        assert diagnostics['A']['status'] == diagnostics['B']['status'] == G.SUPPRESSED
        assert diagnostics['A']['reason'] == diagnostics['B']['reason'] == ''
    else:
        assert ready == (prepared is not None)
        assert diagnostics['A']['reason'] == diagnostics['B']['reason'] == reason


def support_frame(total, group_two):
    return pd.DataFrame({'value': np.arange(total, dtype=float),
                         'community_id': ['1'] * (total - group_two) + ['2'] * group_two,
                         'demo__age_at_baseline': np.random.default_rng(79).normal(49, 3, total),
                         'demo__sex': 'female', 'clinical_baseline_date': '2020-01-01'})


def test_separate_design_support_and_residual_df():
    frame = support_frame(30, 15)
    frame['clinical_baseline_date'] = [f'{1890 + 5 * (i % 19)}-01-01' for i in range(30)]
    diagnostics, ready = G._support(frame, True, POLICY['models'], CELL, False)
    engine, _ = M._model_support(frame, True, POLICY['models'], CELL)
    assert not ready and 'INSUFFICIENT_RESIDUAL_DF' in engine['reason']
    assert diagnostics['A']['status'] == G.PASS
    assert diagnostics['B']['reason'] == 'INSUFFICIENT_RESIDUAL_DF'


def test_recompute_unavailable_source_keeps_published_evidence(published, monkeypatch):
    repo, _, _, target = published
    load = E.current_source_inputs

    def without_eda(*args, **kwargs):
        tables, records, issues = load(*args, **kwargs)
        tables.pop('eda_long')
        return tables, records, issues

    monkeypatch.setattr(E, 'current_source_inputs', without_eda)
    G.run_audit(repo, mode='secure-recompute', strict=True)
    rows = read_output(target, 'readiness_matrix').set_index('outcome')
    assert rows.loc[HEP2, 'primary_measurement_status'] == 'ZERO_ELIGIBLE_MEASUREMENTS'
    assert rows.source_funnel_status.eq('SOURCE_SCHEMA_UNAVAILABLE').all()
    assert rows.loc[RF, 'model_A_support_status'] == G.UNKNOWN


@pytest.mark.parametrize(('source', 'changes', 'stage', 'original'), [
    (RF_TEXT, {'result_text': 'borderline'}, 'INVALID_OR_CENSORED_RESULT', 'UNKNOWN_RESULT_TOKEN'),
    (RF, {'unit': 'incorrect_unit'}, 'ASSAY_DEFINITION_UNAVAILABLE_OR_INCOMPARABLE', 'NUMERIC_UNIT_UNAVAILABLE'),
    (RF, {'order_name_original': None}, 'ASSAY_DEFINITION_UNAVAILABLE_OR_INCOMPARABLE',
     'ORIGINAL_TEST_DEFINITION_UNAVAILABLE'),
    (RF, {'lab_date': '2020-07-02T01:00:00+05:00'}, 'OUTSIDE_0_TO_6M', 'AFTER_6M'),
    (RF, {'specimen_date_origin': None}, 'COLLECTION_DATE_UNVERIFIED', S.EDA_ONLY),
    (RF, {'lab_date': None}, 'COLLECTION_DATE_UNVERIFIED', 'DATE_MISSING'),
    (HEP2, {}, 'NO_RECOGNIZED_OUTCOME_MAPPING', ''),
])
def test_funnel_distinguishes_first_losses(calendar_originals, source, changes, stage, original):
    paths, _, _, _ = calendar_originals
    eda = pd.read_parquet(paths['eda_long']).iloc[:20].assign(
        unit='IU/mL', lab_date='2020-07-01T23:45:00-05:00',
        specimen_date_origin='Collected Date Time')
    eda = eda.assign(**changes).assign(specimen_datetime=lambda rows: rows.lab_date)
    tables = {role: pd.read_parquet(paths[role]) for role in ('step10', 'context')}
    tables['eda_long'] = eda
    baseline = pd.read_csv(paths['baseline'], dtype={'patient_id': 'string'})
    candidates = S._candidates(tables, baseline, S._btris_records({}, []), POLICY)
    result = S._outcome(candidates, source, POLICY)
    rows, first = G._funnel(result, pd.DataFrame({'_patient_match': S.NORMALIZE(eda.patient_id)}), source, CELL)
    assert first == stage
    assert not len(S.primary_measurements(result))
    if original:
        assert original in {row.get('original_reason') for row in rows}
    if stage in {'INVALID_OR_CENSORED_RESULT', 'ASSAY_DEFINITION_UNAVAILABLE_OR_INCOMPARABLE'}:
        assert result.inventory.window_bin.eq('DAY91_TO_6M').all()


def test_funnel_distinguishes_empty_source_and_suppresses_small_complements(calendar_originals):
    paths, _, _, _ = calendar_originals
    eda = pd.read_parquet(paths['eda_long']).iloc[:20].assign(
        unit='IU/mL', lab_date='2020-07-01', specimen_date_origin='Collected Date Time')
    tables = {role: pd.read_parquet(paths[role]) for role in ('step10', 'context')}
    baseline = pd.read_csv(paths['baseline'], dtype={'patient_id': 'string'})

    def audit(frame):
        candidates = S._candidates(dict(tables, eda_long=frame), baseline, S._btris_records({}, []), POLICY)
        return G._funnel(S._outcome(candidates, RF, POLICY),
                         pd.DataFrame({'_patient_match': S.NORMALIZE(frame.patient_id)}), RF, CELL)

    _, empty = audit(eda.iloc[0:0])
    assert empty == 'NO_SOURCE_ROWS'
    eda.loc[eda.index[0], 'lab_date'] = '2020-07-02'
    eda['specimen_datetime'] = eda.lab_date
    rows, suppressed = audit(eda)
    assert suppressed == G.SUPPRESSED
    assert all(row['status'] == G.SUPPRESSED and not row.get('reason') and not row.get('original_reason')
               for row in rows)


def test_reviewed_direct_outcome_still_descriptive_and_eligible_pair_ready(published):
    repo, _, _, target = published
    config = repo / E.CONFIGS / E.VALIDATION
    validation = yaml.safe_load(config.read_text())
    for rep, outcome in [('R0', RF), ('R2', RF_TEXT)]:
        scope = dict(representation_id=rep, source_column=outcome,
                     window_policy_version=G.baseline_policy().policy_version)
        validation['characterization']['independence_reviews'].append(dict(
            scope, **G.REVIEWS, reviewer='SYNTHETIC_PERSON',
            source_reference='/SYNTHETIC_PRIVATE_PATH', baseline_window='six months'))
        validation['n3_validation']['prespecified_outcomes'].append(scope)
    config.write_text(yaml.safe_dump(validation))
    G.run_audit(repo, mode='secure-recompute', representation='all', strict=True)
    rows = read_output(target, 'readiness_matrix').set_index(G.PAIRS)
    assert rows.loc['R0', RF].readiness_status == 'READY_FOR_ESTIMATION'
    assert rows.loc['R0', RF].graph03e_reason == 'INDEPENDENCE_REVIEW_PENDING'
    assert rows.loc['R2', RF_TEXT].approval_gate_status == G.PASS
    assert rows.loc['R2', RF_TEXT].readiness_status == 'N1_DESCRIPTIVE'
    assert 'SYNTHETIC_PRIVATE_PATH' not in (target / '03g_independence_lineage.csv').read_text()

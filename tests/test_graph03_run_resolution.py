"""Autonomous Graph03 commands and ancestry selection, synthetic releases only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from src.studies.graph._pipeline import digest
from src.studies.graph.stage03.s03_n3_phasec_adjudication import load_phasec_inputs, run_phasec
from src.studies.graph.stage03 import s03_n3_resolution as D
from src.studies.graph.stage03.s03_run_resolution import (
    graph_paths, resolve_graph03_dependencies, resolve_latest_compatible,
)
from test_graph03_n3_phasec_adjudication import synthetic_bundle

ROOT = Path(__file__).resolve().parents[1]
C_CONFIG = ROOT / 'src/studies/graph/configs/n3_phasec_review.yaml'
D_CONFIG = ROOT / 'src/studies/graph/configs/n3_phase_d_resolution.yaml'


def install_bundle(tmp_path, data_root, day='01'):
    tmp_path.mkdir(parents=True)
    graph03, release, _ = synthetic_bundle(tmp_path)
    paths = graph_paths(ROOT)
    g02 = paths['graph02'] / 'runs' / f'202611{day}T010000000000Z_00000002'
    g03 = paths['graph03'] / 'runs' / f'202611{day}T020000000000Z_00000003'
    g02.parent.mkdir(parents=True, exist_ok=True)
    g03.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(release.parent), g02)
    shutil.move(str(graph03), g03)
    release = g02 / release.name
    manifest = json.loads(release.read_text())
    manifest.update(run_id=g02.name, created_at_utc=f'2026-11-{day}T01:00:00Z')
    release.write_text(json.dumps(manifest))
    for name in ['03_graph_n3_validation_qc.json', '03_graph_release_input_qc.json', '03_graph_results_summary.json']:
        path = g03 / name
        document = json.loads(path.read_text())
        document.update(graph03_run_id=g03.name, graph02_run_id=g02.name,
                        graph02_release_sha256=digest(release))
        if name == '03_graph_results_summary.json':
            document.update(schema_version='graph03_results_v1', execution_status='COMPLETE')
        path.write_text(json.dumps(document))
    refresh_summary(g03)
    return g03, release


def refresh_summary(graph03):
    path = graph03 / '03_graph_results_summary.json'
    summary = json.loads(path.read_text())
    summary['sha256_by_artifact'] = {p.name: digest(p) for p in graph03.iterdir() if p != path}
    path.write_text(json.dumps(summary))


def args(phase='C', **overrides):
    result = dict(graph03_run=None, graph02_release=None, phasec_run=None,
                  config=D_CONFIG if phase == 'D' else C_CONFIG, strict=False)
    return argparse.Namespace(**(result | overrides))


def command(script, data_root, cwd, *flags):
    return subprocess.run([sys.executable, str(ROOT / 'src/studies/graph' / script), *flags],
        cwd=cwd, env=os.environ | {'OBJ1_SJD_ROOT': str(data_root)}, capture_output=True, text=True)


def test_both_commands_zero_flags_outside_checkout_and_repeatable(tmp_path, monkeypatch):
    monkeypatch.setenv('OBJ1_SJD_ROOT', str(tmp_path))
    graph03, release = install_bundle(tmp_path / 'seed', tmp_path)
    before = {p: digest(p) for directory in [graph03, release.parent] for p in directory.iterdir()}
    c = command('03c_graph_n3_clinical_adjudication.py', tmp_path, tmp_path)
    assert c.returncode == 0, c.stderr
    paths = graph_paths(ROOT)
    first_c = next((paths['phasec'] / 'runs').iterdir())
    qc_c = json.loads((first_c / '03_graph_n3_phasec_qc.json').read_text())
    assert qc_c['run_id'] == first_c.name
    assert len(qc_c['dependency_resolution']) == 2
    d = command('03d_graph_n3_resolution_preflight.py', tmp_path, tmp_path)
    assert d.returncode == 0, d.stderr
    assert 'COMPLETE_WITH_BLOCKERS' in d.stdout
    first_d = next((paths['phased'] / 'runs').iterdir())
    qc = json.loads((first_d / (D.PREFIX + 'qc.json')).read_text())
    assert qc['run_id'] == first_d.name and qc['models_estimated'] == 0
    assert len(qc['dependency_resolution']) == 3
    assert qc['blocker_status']['models_abcd'] == 'NOT_ESTIMABLE'
    assert qc['blocker_status']['protocol'] == 'PENDING_HUMAN_REVIEW'
    second = command('03d_graph_n3_resolution_preflight.py', tmp_path, ROOT)
    assert second.returncode == 0, second.stderr
    second_d = next(p for p in (paths['phased'] / 'runs').iterdir() if p != first_d)
    # Compare scientific decisions, excluding per-execution provenance columns.
    columns = ['representation_id', 'source_column', 'model', 'primary_reason',
               'possible_remedy', 'status', 'blocking_reasons', 'r0_stable_only_support_gate']
    pd.testing.assert_frame_equal(
        pd.read_csv(first_d / (D.PREFIX + 'model_preflight.csv'))[columns],
        pd.read_csv(second_d / (D.PREFIX + 'model_preflight.csv'))[columns])
    assert before == {p: digest(p) for p in before}
    for directory in [first_c, first_d]:
        for p in directory.iterdir():
            assert 'SYNTHETIC_0' not in p.read_text() and '2020-01-01' not in p.read_text()


def test_newest_complete_skips_incomplete_and_failed_run(tmp_path, monkeypatch):
    monkeypatch.setenv('OBJ1_SJD_ROOT', str(tmp_path))
    old, release = install_bundle(tmp_path / 'seed1', tmp_path, '01')
    latest, _ = install_bundle(tmp_path / 'seed2', tmp_path, '02')
    selected = resolve_graph03_dependencies(args(), ROOT, phase='C')
    assert selected[0] == latest
    (latest / '03_graph_n3_source_inventory.csv').unlink()
    selected = resolve_graph03_dependencies(args(), ROOT, phase='C')
    assert selected[:2] == (old, release)
    assert selected[3][-1]['rejected_candidates']
    with pytest.raises(RuntimeError, match='archivo requerido ausente'):
        resolve_graph03_dependencies(args(graph03_run=latest), ROOT, phase='C')


def test_phase_d_uses_phasec_ancestry_even_with_newer_a_and_02(tmp_path, monkeypatch):
    monkeypatch.setenv('OBJ1_SJD_ROOT', str(tmp_path))
    old, release = install_bundle(tmp_path / 'seed1', tmp_path, '01')
    result = run_phasec(old, release, None, C_CONFIG, ROOT)
    latest, newer_release = install_bundle(tmp_path / 'seed2', tmp_path, '02')
    selected = resolve_graph03_dependencies(args('D'), ROOT, phase='D')
    assert selected[:3] == (old, release, Path(result['output_dir']))
    with pytest.raises(RuntimeError, match='compatible'):
        resolve_graph03_dependencies(args('D', graph03_run=latest), ROOT, phase='D')
    with pytest.raises(RuntimeError, match='compatible'):
        resolve_graph03_dependencies(args(graph03_run=old, graph02_release=newer_release), ROOT, phase='C')
    explicit = resolve_graph03_dependencies(args(graph03_run=old, graph02_release=release), ROOT, phase='C')
    assert explicit[:2] == (old, release)


def test_incompatible_new_phasec_falls_back_as_a_unit(tmp_path, monkeypatch):
    monkeypatch.setenv('OBJ1_SJD_ROOT', str(tmp_path))
    graph03, release = install_bundle(tmp_path / 'seed', tmp_path)
    old = run_phasec(graph03, release, None, C_CONFIG, ROOT)
    newer = run_phasec(graph03, release, None, C_CONFIG, ROOT)
    qc = Path(newer['output_dir']) / '03_graph_n3_phasec_qc.json'
    value = json.loads(qc.read_text())
    value['input_graph03_run_id'] = 'OTHER_BRANCH'
    qc.write_text(json.dumps(value))
    selected = resolve_graph03_dependencies(args('D'), ROOT, phase='D')
    assert selected[2] == Path(old['output_dir'])
    assert any('compatible' in reason for reason in selected[3][-1]['rejected_candidates'])


def test_normal_code_pin_drift_does_not_bypass_real_integrity(tmp_path, monkeypatch):
    monkeypatch.setenv('OBJ1_SJD_ROOT', str(tmp_path))
    graph03, release = install_bundle(tmp_path / 'seed', tmp_path)
    cfg = yaml.safe_load(C_CONFIG.read_text())
    cfg['input_contract']['implementation_commit_sha'] = 'f' * 40
    assert load_phasec_inputs(graph03, release, cfg)['qc']['implementation_commit_sha'] == 'e' * 40
    with pytest.raises(ValueError, match='PIN'):
        load_phasec_inputs(graph03, release, cfg, strict=True)
    (release.parent / 'baseline_raw.csv').write_text('tampered')
    with pytest.raises(RuntimeError, match='SHA mismatch'):
        resolve_graph03_dependencies(args(), ROOT, phase='C')


@pytest.mark.parametrize('mutation', ['schema', 'phase_a_status', 'denominators', 'summary_failed', 'failure_marker'])
def test_semantically_invalid_run_rejected_even_with_consistent_hashes(tmp_path, monkeypatch, mutation):
    monkeypatch.setenv('OBJ1_SJD_ROOT', str(tmp_path))
    graph03, _ = install_bundle(tmp_path / 'seed', tmp_path)
    qcpath = graph03 / '03_graph_n3_validation_qc.json'
    qc = json.loads(qcpath.read_text())
    if mutation == 'schema':
        qc['schema_version'] = 'unknown'
    elif mutation == 'phase_a_status':
        qc['phase_a_status'] = 'FAILED'
    elif mutation == 'summary_failed':
        path = graph03 / '03_graph_results_summary.json'
        summary = json.loads(path.read_text())
        summary['execution_status'] = 'FAILED'
        path.write_text(json.dumps(summary))
    elif mutation == 'failure_marker':
        (graph03 / '03_graph_failure.json').write_text('{"execution_status":"FAILED"}')
    else:
        inventory = graph03 / '03_graph_n3_source_inventory.csv'
        frame = pd.read_csv(inventory)
        frame.loc[0, 'n_hard_supported'] = 1000
        frame.to_csv(inventory, index=False)
    qcpath.write_text(json.dumps(qc))
    refresh_summary(graph03)
    with pytest.raises(RuntimeError, match='compatible'):
        resolve_graph03_dependencies(args(), ROOT, phase='C')


def test_discovery_never_uses_mtime_as_creation_time(tmp_path):
    for identifier in ['20261101T010000000000Z_00000001', '20261102T010000000000Z_00000002']:
        directory = tmp_path / 'runs' / identifier
        directory.mkdir(parents=True)
        (directory / 'qc.json').write_text(json.dumps({'run_id': identifier}))
    old = tmp_path / 'runs' / '20261101T010000000000Z_00000001'
    os.utime(old, (9999999999, 9999999999))
    selected, _, _ = resolve_latest_compatible('Graph03 Phase A', [tmp_path], ['qc.json'],
                                              'qc.json', lambda p: None)
    assert selected.name.startswith('20261102')


def test_pop_equivalence_requires_signoff_and_reconciles_derived_copy():
    values = pd.Series(['Unclassifiable', 'Unclassified', 'Pop1', None], index=['a', 'b', 'c', 'd'])
    review = dict(verified=False, record=dict(crosswalk={'Unclassifiable': 'Unclassified'},
                                            equivalence_review='DOCUMENTED'))
    derived, applied = D.reviewed_pop_values(values, review)
    assert not applied and derived.equals(values)

    review['verified'] = True
    derived, applied = D.reviewed_pop_values(values, review)
    assert applied and derived.eq('Unclassified').sum() == 2
    assert values.iloc[0] == 'Unclassifiable' and derived.index.equals(values.index)
    review['record']['crosswalk'] = {'Unclassifiable': 'Pop1'}
    derived, applied = D.reviewed_pop_values(values, review)
    assert not applied and derived.equals(values)


def test_r0_sensitivity_selection_is_fixed_before_endpoint_missingness(tmp_path, monkeypatch):
    monkeypatch.setenv('OBJ1_SJD_ROOT', str(tmp_path))
    graph03, release = install_bundle(tmp_path / 'seed', tmp_path)
    c = run_phasec(graph03, release, None, C_CONFIG, ROOT)
    config = yaml.safe_load(D_CONFIG.read_text())
    inputs = D.load_inputs(graph03, release, Path(c['output_dir']), config)
    validation = yaml.safe_load((ROOT / 'src/studies/graph/configs/validation_03.yaml').read_text())
    # Change observations in memory only. Selection must still be the same 50
    # stable supported patients for every endpoint, including an unobserved one.
    inputs['frozen']['baseline_raw']['ext__biopsy_focus_score'] = pd.NA
    tables, _ = D.build_tables(inputs, config, ROOT, validation)
    models = tables['model_preflight']
    r0 = models.loc[models.representation_id.eq('R0')]
    assert r0.n_r0_stable_hard.eq(50).all()
    assert r0.n_r0_stable_common_complete_case.eq(0).all()
    assert r0.common_population_verified.all()
    assert models.status.eq('NOT_ESTIMABLE').all()
    assert not models.loc[models.representation_id.eq('R2'), 'r0_stable_only_executed'].fillna(False).any()
    assert set(models.primary_reason) and not {'p_value', 'effect', 'ci_low'} & set(models)


def test_normal_d_reports_downstream_code_drift_without_adding_a_blocker(tmp_path, monkeypatch):
    monkeypatch.setenv('OBJ1_SJD_ROOT', str(tmp_path))
    graph03, release = install_bundle(tmp_path / 'seed', tmp_path)
    c = run_phasec(graph03, release, None, C_CONFIG, ROOT)
    phasec = Path(c['output_dir'])
    # Simulate a historical documentary code digest. Artifact integrity remains
    # valid; current code is not required to match the historical producer.
    qcpath = phasec / '03_graph_n3_phasec_qc.json'
    qc = json.loads(qcpath.read_text())
    qc['documentary_source_sha256']['src/studies/_shared.py'] = '0' * 64
    qcpath.write_text(json.dumps(qc))
    inventory_path = phasec / '03_graph_n3_phasec_artifact_inventory.csv'
    inventory = pd.read_csv(inventory_path)
    inventory.loc[inventory.relative_path.eq(qcpath.name), 'sha256'] = digest(qcpath)
    inventory.to_csv(inventory_path, index=False)
    result = D.run_resolution(graph03, release, phasec, None, D_CONFIG, ROOT)
    output = Path(result['output_dir'])
    qc = json.loads((output / (D.PREFIX + 'qc.json')).read_text())
    assert qc['code_drift_gate'] == 'REPORTED_CODE_DRIFT'
    models = pd.read_csv(output / (D.PREFIX + 'model_preflight.csv'))
    assert not models.blocking_reasons.str.contains('CODE_DRIFT').any()
    assert models.status.eq('NOT_ESTIMABLE').all() and qc['automatic_approvals'] == 0

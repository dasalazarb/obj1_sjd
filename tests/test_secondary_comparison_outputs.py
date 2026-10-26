import json
import os
from pathlib import Path
import shutil
import pandas as pd
import pytest

from test_secondary_trajectories_by_essdai_pattern import upstream
from secondary_comparison_fixtures import args_for, cli
from src.block_A.secondary_comparison_config import STEM
from src.block_A.secondary_comparison_contract import fingerprints
from src.block_A.secondary_comparison_outputs import directories, install_bundle, preflight, safe_table
from src.block_A.secondary_trajectory_analysis import ContractError, sha256


def test_dry_run_is_fast_feasibility_only_zero_writes(upstream, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run fitted a model or rendered a figure')
    monkeypatch.setattr(cli, 'longitudinal', forbidden)
    monkeypatch.setattr(cli, 'render_figures', forbidden)
    output = tmp_path/'output'
    report = cli.run(args_for(upstream, output, '--dry-run', '--variables', 'esspri_total', 'complement_c3'))
    assert report['dry_run'] and not output.exists()
    assert len(report['thresholds']) == 2
    assert report['thresholds']['3']['group_counts'] != report['thresholds']['5']['group_counts']
    assert all(row['status'] == 'NOT_ESTIMABLE' for row in report['thresholds']['3']['feasibility'])


def test_bundles_manifest_privacy_and_upstream_unchanged(upstream, tmp_path):
    original = {p: sha256(p) for p in upstream['root'].rglob('*') if p.is_file()}
    output = tmp_path/'output'
    args = args_for(upstream, output, '--variables', 'esspri_total', 'complement_c3', 'anti_ro_ssa', 'overlap_status')
    report = cli.run(args)
    assert report['status'].startswith('PRELIMINARY')
    root = output/'synthetic_validation'
    for threshold in (3, 5):
        dirs = directories(root, 'clinical_baseline', threshold, 'all')
        content = json.loads((dirs['qc']/'run_manifest.json').read_text())
        assert content['qc_status'] == 'PASS'
        assert content['clinical_review_status'] == 'NOT_CLINICALLY_REVIEWED'
        assert content['input_origin'] == 'synthetic_validation'
        assert content['inputs']['context']['sha256']
        for artifact in content['artifacts']:
            assert sha256(root/artifact['path']) == artifact['sha256']
        for csv in dirs['tables'].glob('*.csv'):
            data = pd.read_csv(csv)
            safe_table(data, upstream['p'].patient_id)
            assert 'synthetic_000' not in csv.read_text()
        models = pd.read_csv(dirs['tables']/'03_global_group_time_tests.csv')
        assert models.p.isna().all() and models.q_value.isna().all()
        assert not models.threshold_abs.ne(threshold).any()
        manifest = pd.read_csv(dirs['tables']/'plot_manifest.csv')
        assert manifest.status.eq('RENDERED').all()
        assert list(dirs['figures'].glob('*.png'))
    assert {p: sha256(p) for p in original} == original
    with pytest.raises(FileExistsError):
        cli.run(args)
    # Feasibility gets its own honest namespace and can coexist with full output.
    args.analysis = 'feasibility'
    cli.run(args)
    assert not (directories(root, 'clinical_baseline', 3, 'feasibility')['tables']/'03_global_group_time_tests.csv').exists()


def test_input_mutation_aborts_before_install(upstream, tmp_path, monkeypatch):
    # Mutate an independent copy; preserve the shared canonical fixture.
    copied = tmp_path/'copy'
    shutil.copytree(upstream['root'], copied)
    u = dict(upstream)
    for key in ('root', 'integrated', 'baseline', 'patterns', 'manifest'):
        u[key] = copied if key == 'root' else copied/upstream[key].relative_to(upstream['root'])
    original = cli.measurements
    once = [False]
    def mutation(*args, **kwargs):
        result = original(*args, **kwargs)
        if not once[0]:
            once[0] = True
            with u['integrated'].open('ab') as file:
                file.write(b'changed during run')
        return result
    monkeypatch.setattr(cli, 'measurements', mutation)
    output = tmp_path/'out'
    with pytest.raises(ContractError, match='changed during run'):
        cli.run(args_for(u, output, '--analysis', 'feasibility', '--variables', 'esspri_total'))
    assert not list(output.rglob('run_manifest.json'))


def test_transaction_rollback_and_retained_backup(tmp_path, monkeypatch):
    stage, dest = {}, {}
    for area in ('tables', 'figures', 'qc'):
        stage[area], dest[area] = tmp_path/'stage'/area, tmp_path/'dest'/area
        stage[area].mkdir(parents=True)
        dest[area].mkdir(parents=True)
        (stage[area]/'new').write_text('new')
        (dest[area]/'old').write_text('old')
    rename = os.replace
    def failing(source, target):
        if Path(target).name == 'qc':
            raise OSError('synthetic install failure')
        return rename(source, target)
    monkeypatch.setattr(os, 'replace', failing)
    with pytest.raises(OSError, match='synthetic install'):
        install_bundle(stage, dest, tmp_path/'backups')
    assert all((p/'old').read_text() == 'old' and not (p/'new').exists() for p in dest.values())
    assert len(list((tmp_path/'backups').rglob('old'))) == 3


def test_export_failure_leaves_prior_bundle_untouched(upstream, tmp_path, monkeypatch):
    output = tmp_path/'out'
    args = args_for(upstream, output, '--analysis', 'feasibility', '--variables', 'esspri_total', '--threshold', '3')
    cli.run(args)
    root = output/'synthetic_validation'
    original = {p: sha256(p) for p in root.rglob('*') if p.is_file()}
    args.overwrite, args.backup_root = True, root/'outputs/backups'/STEM
    def failure(*args, **kwargs):
        raise OSError('synthetic CSV export failure')
    monkeypatch.setattr(cli, 'write_table', failure)
    with pytest.raises(OSError):
        cli.run(args)
    assert {p: sha256(p) for p in original} == original


def test_overwrite_keeps_other_threshold(upstream, tmp_path):
    output = tmp_path/'out'
    args = args_for(upstream, output, '--analysis', 'feasibility', '--variables', 'esspri_total')
    cli.run(args)
    root = output/'synthetic_validation'
    dirs = directories(root, 'clinical_baseline', 5, 'feasibility')
    original = {p: sha256(p) for d in dirs.values() for p in d.rglob('*') if p.is_file()}
    args.threshold, args.overwrite, args.backup_root = '3', True, root/'outputs/backups'/STEM
    cli.run(args)
    assert {p: sha256(p) for p in original} == original
    assert len(list(args.backup_root.rglob('run_manifest.json'))) == 1


def test_unsafe_outputs_and_identifiers_block(tmp_path):
    with pytest.raises(ContractError, match='identifying columns'):
        safe_table(pd.DataFrame(dict(patient_id=['fake'])))
    with pytest.raises(ContractError, match='patient identity'):
        safe_table(pd.DataFrame(dict(estimate=['fake'])), ['fake'])
    root = tmp_path/'out'
    dest = directories(root, 'clinical_baseline', 3, 'all')
    inputs = fingerprints({})
    with pytest.raises(ContractError, match='requires'):
        preflight(root, dest, inputs, overwrite=True)
    with pytest.raises(ContractError, match='below'):
        preflight(root, dest, inputs, overwrite=True, backup_root=tmp_path/'unsafe')


def test_estimable_end_to_end_models_figures_and_real_hashes(upstream, tmp_path):
    """Run the real v2 producer on fictional data, then consume it without writes upstream."""
    import numpy as np
    from test_secondary_trajectories_by_essdai_pattern import step13
    from src.block_A import observed_patterns_v2 as v2
    from PIL import Image
    rng = np.random.default_rng(1601)
    pieces = []
    for i, scores in enumerate(([2, 3, 2], [2, 8, 9], [12, 6, 5], [2, 9, 2])):
        template = upstream['x'].loc[upstream['x'].patient_id.eq(f'synthetic_{i:03d}')]
        for j in range(15):
            part = template.copy()
            part['patient_id'] = f'fictional_{i}_{j:02d}'
            part['essdai__total'] = scores
            part['esspri__total'] = 5+rng.normal(0, .5)+np.array([0, 2, 4])*[0, .6, -.6, .3][i]+rng.normal(0, .12, 3)
            pieces.append(part)
    x = pd.concat(pieces, ignore_index=True)
    b = x.loc[x.is_clinical_baseline]
    root = tmp_path/'fictional_upstream'
    inputs = root/'inputs'
    inputs.mkdir(parents=True)
    integrated, baseline = inputs/'step10.parquet', inputs/'step11.parquet'
    x.to_parquet(integrated, index=False)
    b.to_parquet(baseline, index=False)
    endpoints = step13.build_patient_absolute_change_endpoints(x)
    v2.write_observed_pattern_outputs_v2(x, b, endpoints, root, input_origin='synthetic_validation',
        input_paths={'step10': integrated, 'step11': baseline})
    dirs = v2.output_directories(root)
    u = dict(integrated=integrated, baseline=baseline,
        patterns=dirs['analytic']/'13_essdai_patient_observed_patterns_v2.parquet',
        manifest=dirs['qc']/'13_essdai_observed_pattern_manifest_v2.json')
    original = {p: sha256(p) for p in root.rglob('*') if p.is_file()}
    output = tmp_path/'result'
    args = args_for(u, output, '--variables', 'esspri_total', '--figures', 'all')
    report = cli.run(args)
    for k in (3, 5):
        assert report['thresholds'][str(k)]['model_status']['ESTIMATED'] == 1
        dest = directories(output/'synthetic_validation', 'clinical_baseline', k, 'all')
        tests = pd.read_csv(dest['tables']/'03_global_group_time_tests.csv')
        assert tests.n_patients.iloc[0] == 60
        assert tests.family_n_tests.iloc[0] == 1
        contrast = pd.read_csv(dest['tables']/'03_pairwise_trajectory_contrasts.csv')
        contrast = contrast.loc[contrast.scenario.eq('primary') & contrast.multiplicity_role.eq('prespecified')]
        assert contrast.estimate.tolist() == pytest.approx([.6, -.6, .3], abs=.06)
        for png in dest['figures'].rglob('*.png'):
            with Image.open(png) as image:
                image.verify()
        plot = pd.read_csv(dest['tables']/'plot_manifest.csv')
        assert plot.status.eq('RENDERED').all()
        assert plot.figure.str.endswith('_adjusted_trajectory').any()
        assert plot.figure.str.endswith('_forest').any()
    assert {p: sha256(p) for p in original} == original

import json
from pathlib import Path
import pandas as pd
import pytest
from test_secondary_trajectories_by_essdai_pattern import upstream
from temporal_spline_fixtures import args_for, cli, fixture
from src.block_A.secondary_trajectory_analysis import sha256, ContractError
from src.block_A.temporal_spline_outputs import directories, TABLES, safe_table
from src.block_A.temporal_spline_sensitivity import scenarios
from src.block_A.temporal_spline_config import SplineConfig, VERSION, C_SUPPORT_POLICY, ANALYSIS_ROLE


def test_dry_run_design_and_support_zero_writes(upstream, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('dry run must not fit or write')
    monkeypatch.setattr(cli, 'fit_family', forbidden)
    monkeypatch.setattr(cli, 'write_table', forbidden)
    out = tmp_path/'output'
    report = cli.run(args_for(upstream, out, '--dry-run', '--variables', 'esspri_total'))
    assert not out.exists()
    assert set(report['thresholds']) == {'3', '5'}
    assert report['thresholds']['3']['group_counts'] != report['thresholds']['5']['group_counts']


def test_bundle_both_thresholds_registry_privacy_manifests_and_no_upstream_changes(upstream, tmp_path):
    before = {p: sha256(p) for p in upstream['root'].rglob('*') if p.is_file()}
    args = args_for(upstream, tmp_path/'output', '--variables', 'esspri_total', 'anti_ro_ssa', '--sensitivity', 'none')
    report = cli.run(args)
    assert report['qc_status'] == 'PASS' and report['published_bundles'] == 2
    for threshold in (3, 5):
        dirs = directories(tmp_path/'output/synthetic_validation', 'clinical_baseline', threshold, 'main')
        manifest = json.loads((dirs['qc']/'run_manifest.json').read_text())
        assert manifest['data_cutoff'] == '2025-01-01' and manifest['input_origin'] == 'synthetic_validation'
        assert manifest['qc_status'] == 'PASS' and manifest['code_hashes']
        assert manifest['version'] == VERSION and manifest['c_support_policy'] == C_SUPPORT_POLICY
        assert manifest['analysis_role'] == ANALYSIS_ROLE
        assert set('ABC').issubset(manifest['model_counts'])
        assert manifest['supported_temporal_coverage']
        for name in TABLES:
            table = pd.read_csv(dirs['tables']/(name+'.csv'))
            safe_table(table, upstream['p'].patient_id)
            if len(table) and 'threshold_abs' in table:
                assert table.threshold_abs.eq(threshold).all()
        tests = pd.read_csv(dirs['tables']/'02_global_temporal_tests.csv')
        assert tests.p.isna().all() and tests.q_value.isna().all()
    assert before == {p: sha256(p) for p in before}
    with pytest.raises(FileExistsError):
        cli.run(args)


def test_one_command_all_with_supported_synthetic_outcome(upstream, tmp_path, monkeypatch):
    # Authenticate real upstream fixtures first, then replace only measurement acquisition
    # with a dense synthetic outcome; never label this a real clinical smoke run.
    data, groups, spec = fixture('different', n=14)
    original = cli.load_inputs
    def load(args):
        x, b, p, specs, catalog, inputs, manifest = original(args)
        return x, b, p, {spec.canonical_column: spec}, catalog, inputs, manifest
    monkeypatch.setattr(cli, 'load_inputs', load)
    monkeypatch.setattr(cli, 'fixed_groups', lambda patterns, threshold: groups.copy())
    monkeypatch.setattr(cli, 'measurements', lambda *a: (data.copy(), 'AVAILABLE', [], spec.unit))
    args = args_for(upstream, tmp_path/'out', '--threshold', '3', '--sensitivity', 'none', '--figures', 'all')
    report = cli.run(args)
    assert report['qc_status'] == 'PASS'
    dirs = directories(tmp_path/'out/synthetic_validation', 'clinical_baseline', 3, 'main')
    tests = pd.read_csv(dirs['tables']/'02_global_temporal_tests.csv')
    assert tests.status.eq('ESTIMATED').all()
    bands = pd.read_csv(dirs['tables']/'04_simultaneous_bands.csv')
    assert bands.contrast.nunique() == 6 and bands.contrast_type.nunique() == 3
    assert len(list(dirs['figures'].glob('*.png'))) >= 8
    reconcile = pd.read_csv(dirs['tables']/'07_step16_reconciliation.csv')
    assert reconcile.status.iloc[0] in ('RECONCILED', 'DIFFERENT_VALIDATED_FAMILY')
    manifest = json.loads((dirs['qc']/'run_manifest.json').read_text())
    for artifact in manifest['artifacts']:
        if artifact['path'].startswith('figures/'):
            path = dirs['figures']/artifact['path'].removeprefix('figures/')
        else:
            path = dirs['tables']/artifact['path']
        assert sha256(path) == artifact['sha256']


def test_zero_argument_full_run_defaults_and_sensitivities_keep_same_full_sample():
    args = cli.parse_args([])
    assert args.data_cutoff == '2026-05-31'
    assert args.threshold == 'both' and args.time_axis == 'clinical_baseline'
    assert args.analysis == 'all' and args.sensitivity == 'standard'
    assert args.figures == 'priority' and args.dry_run is False
    assert args.output_root == cli.common.PROJECT_ROOT
    assert SplineConfig().max_primary_years == 3.
    assert cli.parse_args(['--data-cutoff', '2025-01-01']).data_cutoff == '2025-01-01'
    with pytest.raises(SystemExit):
        cli.parse_args(['--data-cutoff', 'invalid'])
    data, groups, spec = fixture()
    definitions = {name: (d, cap, horizon) for name, d, axis, cap, horizon in scenarios(data, 'clinical_baseline', SplineConfig(), 'standard')}
    a, b = definitions['full_common_horizon'], definitions['full_sample_primary_domain']
    pd.testing.assert_frame_equal(a[0], b[0])
    assert a[1] is None and b[1] is None and a[2] is None and b[2] == 3.


def test_standard_sensitivities_publish_independent_namespaces(upstream, tmp_path):
    report = cli.run(args_for(upstream, tmp_path/'out', '--threshold', '3', '--variables', 'esspri_total'))
    assert report['published_bundles'] == 8
    root = tmp_path/'out/synthetic_validation'
    alternate = directories(root, 'essdai_index', 3, 'alternate_time_axis')
    assert (alternate['qc']/'run_manifest.json').is_file()
    for scenario in ('main', 'full_common_horizon', 'full_sample_primary_domain', 'official_variable_baseline_only',
                     'delayed_variable_baseline_only', 'variable_ge3_dates', 'essdai_ge3_dates'):
        dirs = directories(root, 'clinical_baseline', 3, scenario)
        assert (dirs['qc']/'run_manifest.json').is_file()


def test_upstream_change_aborts_publication(upstream, tmp_path, monkeypatch):
    import shutil
    copied = tmp_path/'copy'
    shutil.copytree(upstream['root'], copied)
    u = dict(upstream)
    for key in ('root', 'integrated', 'baseline', 'patterns', 'manifest'):
        u[key] = copied if key == 'root' else copied/upstream[key].relative_to(upstream['root'])
    original = cli.measurements
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with u['integrated'].open('ab') as handle:
            handle.write(b'changed during run')
        return result
    monkeypatch.setattr(cli, 'measurements', mutate)
    with pytest.raises(ContractError, match='changed during run'):
        cli.run(args_for(u, tmp_path/'out', '--variables', 'esspri_total', '--sensitivity', 'none'))
    assert not list((tmp_path/'out').rglob('run_manifest.json'))

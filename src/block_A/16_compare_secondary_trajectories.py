#!/usr/bin/env python3
"""Step 16: comparative secondary trajectories in fixed observed ESSDAI groups."""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import date, datetime, timezone
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import statsmodels
import common
from src.block_A.secondary_trajectory_analysis import ContractError, WARNINGS, fixed_groups
from src.block_A.secondary_comparison_config import STEM, VERSION, load_config
from src.block_A.secondary_comparison_contract import (
    code_paths, fingerprints, git_identity, inference_frame, load_inputs, measurements, metadata, verify_unchanged,
)
from src.block_A.secondary_comparison_feasibility import feasibility
from src.block_A.secondary_comparison_descriptive import changes, profiles
from src.block_A.secondary_comparison_models import longitudinal
from src.block_A.secondary_comparison_sensitivity import adjust_tests, discovery, scenarios, threshold_consistency
from src.block_A.secondary_comparison_outputs import (
    artifact_manifest, directories, install_bundle, preflight, write_manifest, write_table,
)
from src.block_A import secondary_comparison_figures as figures


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--integrated', type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    p.add_argument('--baseline', type=Path, default=common.INTEGRATED_BASELINE_PARQUET)
    p.add_argument('--patterns', type=Path, default=common.DISEASE_ACTIVITY_PROGRESSION_DIR/'13_essdai_patient_observed_patterns_v2.parquet')
    p.add_argument('--patterns-manifest', type=Path, default=common.BLOCKA_QC_DIR/'13_disease_activity_progression/13_essdai_observed_pattern_manifest_v2.json')
    p.add_argument('--context', type=Path)
    p.add_argument('--source-registry', type=Path)
    p.add_argument('--threshold', choices=('3', '5', 'both'), default='both')
    p.add_argument('--time-axis', choices=('clinical_baseline', 'essdai_index'), default='clinical_baseline')
    p.add_argument('--analysis', choices=('feasibility', 'baseline', 'change', 'longitudinal', 'sensitivity', 'figures', 'all'), default='all')
    p.add_argument('--domains', nargs='+')
    p.add_argument('--variables', nargs='+')
    p.add_argument('--priority-only', action='store_true')
    for name in ('minimum-patients-per-group', 'minimum-repeated-per-group'):
        p.add_argument('--'+name, type=int)
    p.add_argument('--minimum-observed-span-years', type=float)
    p.add_argument('--window-years', type=float)
    p.add_argument('--config', type=Path)
    p.add_argument('--data-cutoff', default=date.today().isoformat())
    p.add_argument('--input-origin', choices=('canonical', 'synthetic_validation'), default='canonical')
    p.add_argument('--output-root', type=Path, default=common.PROJECT_ROOT)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Contracts and feasibility only; no models, figures or writes')
    mode.add_argument('--validate-models', action='store_true', help='Limited explicit model validation; no writes')
    p.add_argument('--validation-limit', type=int, default=3)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--backup-root', type=Path)
    p.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--figures', choices=('priority', 'all', 'none'), default='priority')
    p.add_argument('--format', nargs='+', choices=('png', 'pdf'), default=['png'])
    p.add_argument('--dpi', type=int, default=250)
    args = p.parse_args(argv)
    try:
        date.fromisoformat(args.data_cutoff)
    except ValueError:
        p.error('--data-cutoff requires YYYY-MM-DD')
    if args.dpi < 200 or args.validation_limit < 1:
        p.error('dpi >=200 and validation-limit >=1 required')
    if args.overwrite and not args.backup_root and not (args.dry_run or args.validate_models):
        p.error('--overwrite requires --backup-root')
    return args


def combine(frames):
    nonempty = [f for f in frames if len(f)]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True)
    schemas = [f for f in frames if len(f.columns)]
    return pd.concat(schemas, ignore_index=True) if schemas else pd.DataFrame(columns=['variable', 'status', 'reason'])


def annotated(data, meta, **extra):
    result = data.copy()
    for key, value in {**meta, **extra}.items():
        result[key] = value
    return result


def run(args):
    started = time.monotonic()
    code_hashes = fingerprints(code_paths(ROOT))
    config_snapshot = fingerprints({'config': args.config}) if args.config else {}
    cfg = load_config(args.config, {key: getattr(args, key) for key in
        ('minimum_patients_per_group', 'minimum_repeated_per_group', 'minimum_observed_span_years', 'window_years', 'seed')})
    x, baseline, patterns, specs, catalog, input_hashes, upstream_manifest = load_inputs(args)
    verify_unchanged(config_snapshot)
    ids, domains = {s.variable_id for s in specs.values()}, {s.domain for s in specs.values()}
    if args.variables and not set(args.variables).issubset(ids):
        raise ContractError('unknown requested variables')
    if args.domains and not set(args.domains).issubset(domains):
        raise ContractError('unknown requested domains')
    selected = [s for s in specs.values() if
        (not args.variables or s.variable_id in args.variables) and
        (not args.domains or s.domain in args.domains) and
        (not args.priority_only or s.variable_id in cfg.priority_variables)]
    if not selected:
        raise ContractError('selection contains no registered variables')
    if args.validate_models:
        selected = selected[:args.validation_limit]
    root = args.output_root.resolve()
    if args.input_origin == 'synthetic_validation':
        root /= 'synthetic_validation'
    thresholds = (3, 5) if args.threshold == 'both' else (int(args.threshold),)
    destinations = {f'{area}/delta{k}': path for k in thresholds
        for area, path in directories(root, args.time_axis, k, args.analysis).items()}
    no_writes = args.dry_run or args.validate_models
    if not no_writes:
        preflight(root, destinations, input_hashes, args.overwrite, args.backup_root)
    bundles, report = {}, dict(status='PRELIMINARY / NOT_CLINICALLY_REVIEWED',
        input_origin=args.input_origin, dry_run=args.dry_run, validate_models=args.validate_models, thresholds={})
    for threshold in thresholds:
        groups = fixed_groups(patterns, threshold)
        original_groups = groups.copy(deep=True)
        collected = {}
        def add(name, data, meta=None):
            collected.setdefault(name, []).append(annotated(data, meta) if meta else data)
        cat = catalog.copy(deep=True)
        cat['selected'] = cat.canonical_column.isin({s.canonical_column for s in selected})
        checks, statuses = [], []
        for index, spec in enumerate(selected, 1):
            begin = time.monotonic()
            if args.progress:
                print(f'Step16 ±{threshold} {index}/{len(selected)} {spec.variable_id}: feasibility', flush=True)
            frame, observation_status, qc, unit = measurements(x, groups, spec, args.data_cutoff, args.time_axis, cfg)
            spec = replace(spec, unit=unit)
            meta = metadata(spec, threshold, args.time_axis, cfg)
            checks.extend(dict(**q, threshold_abs=threshold) for q in qc)
            f = feasibility(frame, groups, spec, threshold, args.time_axis, cfg)
            for name, key in (('00_group_observation_support', 'support'), ('00_coverage_by_window', 'windows'),
                ('00_availability_by_protocol', 'protocols'), ('00_followup_observation_opportunities', 'opportunities'),
                ('00_overlap_of_time_support', 'overlap')):
                add(name, f[key], meta)
            add('00_variable_feasibility', pd.DataFrame([dict(**f['gate'], observation_status=observation_status)]), meta)
            if frame.empty or f['gate']['reason']:
                add('00_exclusions_and_reasons', pd.DataFrame([dict(status=f['gate']['status'],
                    reason=observation_status if frame.empty else f['gate']['reason'])]), meta)
            model_seconds = 0.
            if not args.dry_run and args.analysis != 'feasibility':
                if args.analysis in ('baseline', 'all'):
                    description, contrasts = profiles(frame, spec, cfg)
                    add('01_profiles', description, meta)
                    if 'profile' in description:
                        add('01_official_baseline_descriptives', description.loc[description.profile.eq('official_baseline')], meta)
                        add('01_first_available_descriptives', description.loc[description.profile.eq('first_available')], meta)
                    add('01_pairwise_baseline_contrasts', contrasts, meta)
                if args.analysis in ('change', 'all'):
                    change_data = inference_frame(frame, args.time_axis)
                    # ESSDAI-index inference must not use a preindex reference for change.
                    if args.time_axis == 'essdai_index' and len(change_data):
                        change_data = change_data.loc[change_data.reference_date.ge(change_data.essdai_index_date)]
                    d = changes(change_data, spec, cfg)
                    for name, key in (('02_numeric_change_by_window', 'change'), ('02_individual_slope_distribution', 'slopes'),
                        ('02_variability_by_group', 'variability'), ('02_observed_auc', 'auc'), ('02_change_contrasts', 'contrasts')):
                        add(name, d[key], meta)
                if args.analysis in ('longitudinal', 'sensitivity', 'figures', 'all') or args.validate_models:
                    model_start = time.monotonic()
                    models = longitudinal(frame, spec, f['gate'], cfg, args.time_axis)
                    mapping = {'attempts': '03_model_attempts', 'qc': '03_model_qc', 'global': '03_global_group_time_tests',
                        'coefficients': '03_model_coefficients', 'contrasts': '03_pairwise_trajectory_contrasts',
                        'predictions': '03_marginal_predictions', 'support': '03_model_support'}
                    for key, name in mapping.items():
                        add(name, models[key], meta)
                    if args.analysis in ('all', 'sensitivity') and not args.validate_models:
                        for scenario, subset, axis in scenarios(frame, args.time_axis, cfg):
                            sf = feasibility(subset, groups, spec, threshold, axis, cfg)
                            sensitivity = longitudinal(subset, spec, sf['gate'], cfg, axis, scenario)
                            smeta = metadata(spec, threshold, axis, cfg)
                            add('04_sensitivity_comparison', sensitivity['global'], smeta)
                            for key in ('attempts', 'qc', 'contrasts', 'support'):
                                add(mapping[key], sensitivity[key], smeta)
                            add('04_sensitivity_support', annotated(sf['support'], smeta, scenario=scenario))
                        add('04_sensitivity_comparison', pd.DataFrame([dict(scenario='baseline_covariate_adjustment',
                            status='NOT_ESTIMABLE', reason='No clinically approved covariate mapping prespecified', p=np.nan)]), meta)
                    model_seconds = time.monotonic()-model_start
            statuses.append(dict(**meta, observation_status=observation_status,
                longitudinal_status=f['gate']['status'], reason=f['gate']['reason'],
                analysis_seconds=time.monotonic()-begin, model_seconds=model_seconds,
                maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            # Patient-level intermediates are released after each outcome.
            del frame
        pd.testing.assert_frame_equal(groups, original_groups)
        add('variable_registry', cat)
        add('analysis_status', pd.DataFrame(statuses))
        add('qc_checks', pd.DataFrame(checks+[dict(variable='all', check='protected_provenance_and_fixed_groups', status='PASS', n=0)]))
        add('estimand_difference_log', pd.DataFrame([dict(
            analysis='followup_change', status='DOCUMENTED',
            difference='Step15 delta windows can include reference delta=0; Step16 uses is_delta_followup only, own-reference clock and separate reference sources')]))
        tables = {name: combine(items) for name, items in collected.items()}
        if '03_global_group_time_tests' in tables:
            tables['03_global_group_time_tests'] = adjust_tests(tables['03_global_group_time_tests'])
            tables['03_pairwise_trajectory_contrasts'] = adjust_tests(tables['03_pairwise_trajectory_contrasts'], 'pairwise')
            if '04_sensitivity_comparison' in tables:
                tables['04_sensitivity_comparison'] = adjust_tests(tables['04_sensitivity_comparison'])
            tables['04_multiple_testing'] = tables['03_global_group_time_tests'].copy()
            tables['04_discovery_matrix'] = discovery(tables['03_global_group_time_tests'], tables['03_pairwise_trajectory_contrasts'])
        if '01_pairwise_baseline_contrasts' in tables and len(tables['01_pairwise_baseline_contrasts']):
            cross = tables['01_pairwise_baseline_contrasts']
            tables['01_pairwise_baseline_contrasts'] = combine([
                adjust_tests(cross.loc[cross.contrast.eq('global')]),
                adjust_tests(cross.loc[cross.contrast.ne('global')], 'pairwise')])
        bundles[threshold] = tables
        report['thresholds'][str(threshold)] = dict(group_counts=groups.observed_pattern.value_counts().to_dict(),
            feasibility=tables['00_variable_feasibility'][['variable', 'status', 'reason']].to_dict('records'))
        if '03_global_group_time_tests' in tables:
            report['thresholds'][str(threshold)]['model_status'] = tables['03_global_group_time_tests'].status.value_counts().to_dict()
    all_contrasts = combine([t.get('03_pairwise_trajectory_contrasts', pd.DataFrame()) for t in bundles.values()])
    consistency = threshold_consistency(all_contrasts)
    for tables in bundles.values():
        if args.analysis in ('all', 'sensitivity') and not args.dry_run:
            tables['04_threshold_consistency'] = consistency
    verify_unchanged(input_hashes)
    verify_unchanged(code_hashes)
    if no_writes:
        report['analysis_seconds'] = time.monotonic()-started
        return report
    # Stage on the destination filesystem, then publish all thresholds together.
    scratch = root/'work'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='step16-', dir=scratch) as temporary:
        stage = Path(temporary)
        staging = {f'{area}/delta{k}': path for k in thresholds
            for area, path in directories(stage, args.time_axis, k, args.analysis).items()}
        for path in staging.values():
            path.mkdir(parents=True)
        figure_events = []
        for threshold, tables in bundles.items():
            dirs = {area: staging[f'{area}/delta{threshold}'] for area in ('tables', 'figures', 'qc')}
            if args.analysis in ('all', 'figures') and args.figures != 'none':
                events = render_figures(tables, dirs['figures'], threshold, args, cfg)
                figure_events.extend(events)
                tables['plot_manifest'] = pd.DataFrame(events)
            for name, data in tables.items():
                area = 'qc' if name in ('qc_checks', '03_model_qc') else 'tables'
                write_table(data, dirs[area]/(name+'.csv'), patterns.patient_id)
        # Check bytes after all potentially long fitting and rendering work.
        verify_unchanged(input_hashes)
        verify_unchanged(code_hashes)
        artifacts = artifact_manifest(stage)
        for threshold in thresholds:
            content = dict(analysis=STEM, version=VERSION, qc_status='PASS',
                clinical_review_status='NOT_CLINICALLY_REVIEWED', publication_status='RESTRICTED_INTERNAL_PRELIMINARY',
                analysis_mode=args.analysis, input_origin=args.input_origin,
                generated_at=datetime.now(timezone.utc).isoformat(), git=git_identity(ROOT),
                inputs=input_hashes, code=code_hashes, upstream_definition=upstream_manifest['definition_version'],
                config=cfg.as_dict(), threshold_abs=threshold, time_axis=args.time_axis,
                software=dict(python=platform.python_version(), pandas=pd.__version__, statsmodels=statsmodels.__version__, numpy=np.__version__),
                group_counts=report['thresholds'][str(threshold)]['group_counts'],
                artifacts=[a for a in artifacts if f'/delta{threshold}/' in a['path']],
                final_immutability_check='PASS', limitations=WARNINGS,
                interpretation='Retrospective descriptive associations; no causal or prospective predictive estimand',
                observation_process='Informative visits not resolved by MixedLM or GEE; missing opportunity denominator not invented',
                analysis_seconds=time.monotonic()-started,
                maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            write_manifest(staging[f'qc/delta{threshold}']/'run_manifest.json', content)
        verify_unchanged(input_hashes)
        verify_unchanged(code_hashes)
        install_bundle(staging, destinations, args.backup_root)
    report['analysis_seconds'] = time.monotonic()-started
    report['outputs'] = {key: str(path) for key, path in destinations.items()}
    return report


def render_figures(tables, destination, threshold, args, cfg):
    events = []
    selected = tables['variable_registry'].loc[tables['variable_registry'].selected.eq(True)]
    kwargs = dict(formats=args.format, dpi=args.dpi, origin=args.input_origin)
    def draw(name, callback):
        start = time.monotonic()
        try:
            paths = callback()
            for path in paths:
                events.append(dict(figure=name, status='RENDERED', reason='', path=str(path.relative_to(destination)),
                                   figure_seconds=time.monotonic()-start))
        except (ValueError, ArithmeticError, OSError) as error:
            import matplotlib.pyplot as plt
            plt.close('all')
            events.append(dict(figure=name, status='FIGURE_FAILED', reason=type(error).__name__,
                               path='', figure_seconds=time.monotonic()-start))
    draw('observation_support_heatmap', lambda: figures.availability(tables['00_group_observation_support'], threshold,
        destination/'observation_support_heatmap', **kwargs))
    draw('observation_support_by_window', lambda: figures.availability_windows(tables['00_coverage_by_window'], threshold,
        destination/'observation_support_by_window', **kwargs))
    contrasts = tables.get('03_pairwise_trajectory_contrasts', pd.DataFrame())
    if len(contrasts):
        contrasts = contrasts.loc[contrasts.scenario.eq('primary')]
    draw('comparison_effect_size_heatmap', lambda: figures.effects(contrasts, tables['variable_registry'], threshold,
        destination/'comparison_effect_size_heatmap', **kwargs))
    predictions = tables.get('03_marginal_predictions', pd.DataFrame())
    for row in selected.itertuples():
        if args.figures == 'priority' and row.variable_id not in cfg.priority_variables:
            continue
        if len(predictions):
            d = predictions.loc[predictions.variable.eq(row.variable_id) & predictions.scenario.eq('primary')]
            if len(d) and d.status.eq('ESTIMATED').any():
                draw(row.variable_id+'_adjusted_trajectory', lambda d=d, row=row: figures.trajectory(d, row.label,
                    d.effect_scale.iloc[0], threshold, args.time_axis,
                    destination/row.domain/(row.variable_id+'_adjusted_trajectory'), **kwargs))
        if len(contrasts):
            c = contrasts.loc[contrasts.variable.eq(row.variable_id)]
            if len(c):
                draw(row.variable_id+'_forest', lambda c=c, row=row: figures.forest(c, row.label,
                    c.loc[c.contrast_type.eq('slope'), 'effect_scale'].iloc[0], threshold, destination/row.domain/(row.variable_id+'_forest'), **kwargs))
    return events


def main(argv=None):
    try:
        report = run(parse_args(argv))
    except (ContractError, FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()

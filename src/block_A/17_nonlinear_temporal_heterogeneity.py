#!/usr/bin/env python3
"""Step 17: one read-only upstream consumer for nonlinear temporal heterogeneity."""
from __future__ import annotations
import argparse
from itertools import combinations
from dataclasses import replace
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import scipy
import statsmodels
import common
from src.block_A.secondary_trajectory_analysis import ContractError, fixed_groups
from src.block_A.temporal_spline_config import VERSION, STEM, load_config
from src.block_A.temporal_spline_contract import (
    code_paths, fingerprints, git_identity, load_inputs, measurements, metadata, sample_fingerprint, verify_unchanged,
)
from src.block_A.temporal_spline_feasibility import evaluate
from src.block_A.temporal_spline_models import fit_family
from src.block_A.temporal_spline_contrasts import characterize
from src.block_A.temporal_spline_inference import adjust_fdr
from src.block_A.temporal_spline_sensitivity import scenarios, consistency, bootstrap_patients
from src.block_A.temporal_spline_outputs import (
    TABLES, artifact_manifest, directories, install_bundle, preflight, write_manifest, write_table,
)

MAPPING = dict(curves='03_group_curves', changes='03_within_group_change', derivatives='04_temporal_derivatives',
    pairs='04_pairwise_temporal_differences', bands='04_simultaneous_bands', intervals='05_separation_intervals',
    characteristics='05_temporal_characteristics', registry='06_non_estimable_registry')
TEST_MODELS = dict(linear_group_time='A', common_nonlinearity='B', group_specific_curvature='C', global_temporal_heterogeneity='C')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--integrated', type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    p.add_argument('--baseline', type=Path, default=common.INTEGRATED_BASELINE_PARQUET)
    p.add_argument('--patterns', type=Path, default=common.DISEASE_ACTIVITY_PROGRESSION_DIR/'13_essdai_patient_observed_patterns_v2.parquet')
    p.add_argument('--patterns-manifest', type=Path, default=common.BLOCKA_QC_DIR/'13_disease_activity_progression/13_essdai_observed_pattern_manifest_v2.json')
    p.add_argument('--context', type=Path)
    p.add_argument('--source-registry', type=Path)
    p.add_argument('--input-origin', choices=('canonical', 'synthetic_validation'), default='canonical')
    p.add_argument('--threshold', choices=('3', '5', 'both'), default='both')
    p.add_argument('--time-axis', choices=('clinical_baseline', 'essdai_index'), default='clinical_baseline')
    p.add_argument('--analysis', choices=('feasibility', 'models', 'temporal', 'all'), default='all')
    p.add_argument('--sensitivity', choices=('none', 'standard', 'full'), default='standard')
    p.add_argument('--variables', nargs='+')
    p.add_argument('--config', type=Path)
    p.add_argument('--data-cutoff', required=True, help='Frozen YYYY-MM-DD; never defaults to the current date')
    p.add_argument('--max-primary-years', type=float)
    p.add_argument('--seed', type=int)
    p.add_argument('--simulations', type=int)
    p.add_argument('--bootstrap-replicates', type=int)
    p.add_argument('--output-root', type=Path, default=common.PROJECT_ROOT)
    p.add_argument('--dry-run', action='store_true', help='Contracts, support and design matrices; zero model fits or writes')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--backup-root', type=Path)
    p.add_argument('--figures', choices=('none', 'priority', 'all'), default='priority')
    p.add_argument('--dpi', type=int, default=250)
    p.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)
    try:
        date.fromisoformat(args.data_cutoff)
    except ValueError:
        p.error('--data-cutoff requires YYYY-MM-DD')
    if args.dpi < 200:
        p.error('--dpi must be >=200')
    if args.overwrite and not args.backup_root and not args.dry_run:
        p.error('--overwrite requires --backup-root')
    return args


def annotate(frame, meta):
    d = frame.copy()
    for k, v in meta.items():
        d[k] = v
    return d


def combine(parts):
    nonempty = [p for p in parts if len(p)]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=['variable', 'status', 'reason'])


def analyze(evaluation, spec, cfg, meta, do_models, do_temporal):
    collected = {k: [] for k in TABLES}
    def add(name, rows):
        d = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        if len(d):
            collected[name].append(annotate(d, meta))
    gate, basis = evaluation['gate'], evaluation['basis']
    add('00_variable_eligibility', [dict(**gate, observation_semantics=spec.observation_semantics,
        basis=json.dumps(basis.as_dict(), sort_keys=True) if basis else '')])
    add('00_local_temporal_support', evaluation['support'])
    add('00_group_support', evaluation['counts'])
    data = evaluation['data']
    if len(data):
        # One patient contributes once to a window's descriptive aggregate.
        observed = data.assign(annual_window=np.floor(data.time).astype(int))
        patient_windows = observed.groupby(['observed_pattern', 'annual_window', 'patient_id']).agg(
            value=('value', 'mean'), time=('time', 'mean')).reset_index()
        aggregates = []
        for (group, window), part in patient_windows.groupby(['observed_pattern', 'annual_window']):
            enough = len(part) >= cfg.minimum_prediction_patients
            aggregates.append(dict(observed_pattern=group, annual_window=window,
                time=float(part.time.mean()), n_patients=len(part),
                mean=part.value.mean() if enough else np.nan,
                median=part.value.median() if enough else np.nan,
                q25=part.value.quantile(.25) if enough else np.nan,
                q75=part.value.quantile(.75) if enough else np.nan,
                status='DESCRIPTIVE' if enough else 'SUPPRESSED_SMALL_CELL'))
        add('00_aggregated_observations', aggregates)
    fits, attempts, tests = fit_family(evaluation, spec, cfg) if do_models else ({}, [], [])
    add('01_fit_attempts', attempts)
    for model, fit in fits.items():
        for j, coefficient in enumerate(fit.beta):
            add('01_model_coefficients', [dict(model=model, term_index=j, estimate=float(coefficient),
                SE=float(np.sqrt(fit.covariance[j, j])), model_family=fit.family,
                effect_scale='log_odds_basis' if spec.type == 'binary' else spec.unit+'_basis', status='ESTIMATED')])
    if 'A' in fits:
        from src.block_A.secondary_comparison_models import PATTERNS, design_row
        from src.block_A.temporal_spline_contrasts import effect
        for a, b in combinations(PATTERNS, 2):
            vector = (design_row(b, 1)-design_row(b, 0))-(design_row(a, 1)-design_row(a, 0))
            add('01_linear_pairwise_slopes', [dict(contrast=b+' - '+a, model='A', status='ESTIMATED',
                effect_scale='log_odds/year' if spec.type == 'binary' else spec.unit+'/year',
                interpretation='Linear model A slope; complementary horizon diagnostic, no post-selection test',
                **effect(float(vector@fits['A'].beta), vector, fits['A'].covariance))])
    for model in 'ABC':
        fit = fits.get(model)
        reason = gate['reason_'+model] or ('fit_failed_or_lower_model_failed' if do_models else 'models_not_requested')
        add('01_model_comparison', [dict(model=model, status='ESTIMATED' if fit else 'NOT_ESTIMABLE',
            reason='' if fit else reason, model_family=fit.family if fit else '',
            n_fixed_parameters=len(fit.beta) if fit else np.nan,
            logLik=fit.loglik if fit else np.nan, AIC=fit.aic if fit else np.nan, BIC=fit.bic if fit else np.nan,
            n_observations=gate['n_observations'], basis=json.dumps(basis.as_dict()) if basis else '',
            information_criterion_policy='ML only; GEE AIC/BIC are undefined',
            random_structure='patient_random_intercept' if fit and fit.family.startswith('MixedLM') else 'patient_cluster_robust_independence')])
        if not fit:
            add('06_non_estimable_registry', [dict(component=model, status='NOT_ESTIMABLE', reason=reason)])
    if do_models:
        by_test = {r['test']: r for r in tests}
        for test, model in TEST_MODELS.items():
            add('02_global_temporal_tests', [by_test.get(test, dict(test=test, model=model, status='NOT_ESTIMABLE',
                reason=gate['reason_'+model] or 'required_fit_failed', p=np.nan))])
    if do_temporal:
        temporal = characterize(evaluation, fits, spec, cfg)
        for key, name in MAPPING.items():
            add(name, temporal[key])
        if spec.variable_id in cfg.priority_variables and meta['scenario'] == 'main':
            add('06_patient_bootstrap_validation', bootstrap_patients(evaluation, fits, spec, cfg))
    return {name: combine(parts) for name, parts in collected.items()}, fits


def reconcile_step16(evaluation, fits, spec, cfg, axis):
    """Explicitly refit the Step 16 FUNCTION on the Step 17 sample, not prior products."""
    from src.block_A.secondary_comparison_models import longitudinal, design
    if not evaluation['gate']['eligible_A']:
        return pd.DataFrame([dict(status='NOT_COMPARABLE', reason='A is not eligible on shared rows')])
    gate = dict(evaluation['gate'], status='ELIGIBLE', reason='')
    result = longitudinal(evaluation['data'], spec, gate, cfg, axis, 'step17_shared_sample_validation')
    d = result['global']
    if d.empty or not d.iloc[0].status == 'ESTIMATED':
        return pd.DataFrame([dict(status='NOT_COMPARABLE', reason='Step16 reference fit failed QC on shared sample')])
    if 'A' not in fits:
        return pd.DataFrame([dict(status='NOT_COMPARABLE', reason='Step17 A failed QC')])
    # Step 16's columns and fixed estimand are exactly the same by construction.
    old = result['coefficients'].estimate.to_numpy(float)
    difference = float(np.max(np.abs(old-fits['A'].beta)))
    old_family, new_family = d.iloc[0].model_family, fits['A'].family
    same_family = (old_family.startswith('MixedLM') == new_family.startswith('MixedLM'))
    expected_design = design(evaluation['data'])
    np.testing.assert_allclose(expected_design, evaluation['matrices']['A'], rtol=0, atol=0)
    if same_family and difference > 1e-5:
        raise ContractError('Step16/17 A mismatch on identical sample and family')
    return pd.DataFrame([dict(status='RECONCILED' if same_family else 'DIFFERENT_VALIDATED_FAMILY',
        reason='' if same_family else 'Step17 coherent ABC family fallback differs from Step16 A-only choice',
        same_sample=True, same_clock=True, same_scale=True, same_design=True,
        step16_family=old_family, step17_family=new_family, maximum_absolute_coefficient_difference=difference,
        estimand_difference='Previously published Step16 full horizon is not directly comparable with Step17 common-support capped horizon')])


def run(args):
    start = time.monotonic()
    code_hashes = fingerprints(code_paths(ROOT))
    config_snapshot = fingerprints({'config': args.config}) if args.config else {}
    cfg = load_config(args.config, **{k: getattr(args, k) for k in
        ('max_primary_years', 'seed', 'simulations', 'bootstrap_replicates')})
    x, baseline, patterns, specs, catalog, inputs, upstream = load_inputs(args)
    verify_unchanged(config_snapshot)
    identity = git_identity(ROOT)
    if args.variables and not set(args.variables).issubset({s.variable_id for s in specs.values()}):
        raise ContractError('unknown registered --variables identifier')
    selected = [s for s in specs.values() if not args.variables or s.variable_id in args.variables]
    if not selected:
        raise ContractError('no registered variables selected')
    thresholds = (3, 5) if args.threshold == 'both' else (int(args.threshold),)
    root = args.output_root.resolve()
    if args.input_origin == 'synthetic_validation':
        root /= 'synthetic_validation'
    bundles, report, destinations = {}, dict(dry_run=args.dry_run, input_origin=args.input_origin,
        status='PRELIMINARY / NOT_CLINICALLY_REVIEWED', thresholds={}), {}
    for threshold in thresholds:
        groups = fixed_groups(patterns, threshold)
        immutable = groups.copy(deep=True)
        for i, original_spec in enumerate(selected, 1):
            if args.progress:
                print(f'Step17 ±{threshold} {i}/{len(selected)} {original_spec.variable_id}: support/models/temporal', flush=True)
            frame, observation_status, qc, unit = measurements(x, groups, original_spec, args.data_cutoff, args.time_axis, cfg)
            spec = replace(original_spec, unit=unit)
            for scenario, subset, axis, fit_cap, inference_cap in scenarios(frame, args.time_axis, cfg, args.sensitivity):
                ev = evaluate(subset, groups, spec, cfg, axis, fit_cap, inference_cap)
                sample_hash = sample_fingerprint(ev['data'])
                meta = dict(metadata(spec, threshold, axis, cfg), scenario=scenario,
                    analysis_role='primary' if threshold == 3 and scenario == 'main' and axis == 'clinical_baseline' else 'sensitivity',
                    sample_fingerprint=sample_hash, n_patients=ev['gate']['n_patients'],
                    fit_horizon_cap=fit_cap, inference_horizon_cap=inference_cap,
                    observation_status=observation_status)
                result, fits = analyze(ev, spec, cfg, meta,
                    do_models=not args.dry_run and args.analysis != 'feasibility',
                    do_temporal=not args.dry_run and args.analysis in ('temporal', 'all'))
                if not args.dry_run and args.analysis in ('all', 'models') and scenario == 'main':
                    result['07_step16_reconciliation'] = annotate(reconcile_step16(ev, fits, spec, cfg, axis), meta)
                key = (threshold, axis, scenario)
                destination = directories(root, axis, threshold, scenario, args.analysis)
                for area, path in destination.items():
                    destinations[f'{area}/delta{threshold}/{axis}/{scenario}'] = path
                target = bundles.setdefault(key, {name: [] for name in TABLES})
                for name, table in result.items():
                    if len(table):
                        target[name].append(table)
                if scenario == 'main':
                    report['thresholds'].setdefault(str(threshold), dict(group_counts=groups.observed_pattern.value_counts().to_dict(), feasibility=[]))
                    report['thresholds'][str(threshold)]['feasibility'].append(dict(variable=spec.variable_id, **ev['gate']))
            pd.testing.assert_frame_equal(groups, immutable)
    tables_by_key = {key: {name: combine(parts) for name, parts in collected.items()} for key, collected in bundles.items()}
    for tables in tables_by_key.values():
        tests = adjust_fdr(tables['02_global_temporal_tests'])
        tables['02_global_temporal_tests'] = tests
        classified = []
        for row in tables['00_variable_eligibility'].to_dict('records'):
            d = tests.loc[tests.variable.eq(row['variable']) & tests.test.eq('global_temporal_heterogeneity')] if len(tests) else tests
            classification = ('NOT_ESTIMABLE' if d.empty or d.iloc[0].status != 'ESTIMATED' else
                              'DYNAMIC_HETEROGENEITY_DETECTED' if d.iloc[0].q_value < .05 else 'NO_DYNAMIC_DIFFERENCE_DETECTED')
            classified.append(dict(variable=row['variable'], domain=row['domain'], interpretation_group=row['interpretation_group'], status=classification,
                reason=row['reason'], interpretation='Variable-level evidence, never a new patient phenotype'))
        tables['05_heterogeneity_classification'] = pd.DataFrame(classified)
    consistency_table = consistency(tables_by_key)
    for (threshold, axis, scenario), tables in tables_by_key.items():
        if len(consistency_table):
            tables['06_sensitivity_and_threshold_consistency'] = consistency_table.loc[
                consistency_table.threshold_abs.eq(threshold) & consistency_table.time_axis.eq(axis) & consistency_table.scenario.eq(scenario)].copy()
    verify_unchanged(inputs)
    verify_unchanged(code_hashes)
    if args.dry_run:
        report.update(analysis_seconds=time.monotonic()-start, code=identity)
        return report
    preflight(root, destinations, inputs, args.overwrite, args.backup_root)
    scratch = root/'work'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='step17-', dir=scratch) as temporary:
        os.chmod(temporary, 0o700)
        stage, staging = Path(temporary), {}
        for key, tables in tables_by_key.items():
            threshold, axis, scenario = key
            paths = {}
            for area in ('tables', 'figures', 'qc'):
                name = f'{area}/delta{threshold}/{axis}/{scenario}'
                paths[area] = stage/name
                paths[area].mkdir(parents=True, exist_ok=True, mode=0o700)
                staging[name] = paths[area]
            for name, table in tables.items():
                write_table(disclosure_table(table), paths['tables']/(name+'.csv'), patterns.patient_id)
            rendered = []
            if args.figures != 'none':
                from src.block_A.temporal_spline_figures import render
                rendered = render(tables, paths['figures'], cfg, args.figures, args.dpi, args.input_origin)
            write_manifest(paths['qc']/'run_manifest.json', dict(
                version=VERSION, qc_status='PASS', scientific_status='PRELIMINARY',
                clinical_review_status='NOT_CLINICALLY_REVIEWED',
                input_origin=args.input_origin, upstream_status=upstream.get('qc_status'),
                threshold_abs=threshold, time_axis=axis, scenario=scenario, analysis=args.analysis,
                data_cutoff=args.data_cutoff, parameters=cfg.as_dict(), code=identity,
                invocation=vars(args), selected_variables=[s.variable_id for s in selected],
                fdr_selection_scope='explicit_variable_subset' if args.variables else 'all_registered_variables',
                preliminary_dirty_code=identity['worktree_dirty'], inputs=inputs, code_hashes=code_hashes,
                created_at=datetime.now(timezone.utc).isoformat(), figures=rendered,
                observation_process='NOT_ESTIMABLE: no verified visit-opportunity denominator; no intensity weighting',
                privacy='Access-controlled aggregates; small nonzero count cells <5 suppressed. Institutional disclosure review required before release.',
                timing='No onset inference; characteristic timing not estimated without validated stability and timing intervals',
                simultaneous_inference='Asymptotic coefficient sup-t; joint six pairs x supported times x three estimands. Patient bootstrap optional for priority main.',
                real_data_validation='Not asserted by software QC; inspect input_origin and institutional smoke validation',
                software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__),
                artifacts=artifact_manifest(paths['tables'])+[
                    dict(path='figures/'+a['path'], sha256=a['sha256']) for a in artifact_manifest(paths['figures'])]))
        verify_unchanged(inputs)
        verify_unchanged(code_hashes)
        install_bundle(staging, destinations, args.backup_root)
    report.update(analysis_seconds=time.monotonic()-start, published_bundles=len(tables_by_key), qc_status='PASS')
    return report


def disclosure_table(table):
    """Conservative suppression is additional to institutional access controls."""
    d = table.copy()
    small_row = pd.Series(False, index=d.index)
    for col in d:
        if (col.startswith(('n_', 'N_')) and pd.api.types.is_numeric_dtype(d[col])
                and col not in ('n_fixed_parameters', 'n_columns', 'n_supported_regions', 'n_supported_annual_windows')):
            mask = d[col].between(1, 4)
            small_row |= mask
            d.loc[mask, col] = np.nan
    if 'percent_group_measured' in d:
        d.loc[small_row, 'percent_group_measured'] = np.nan
    if 'n_per_group' in d:
        d['n_per_group'] = d.n_per_group.map(lambda value: json.dumps({k: v if v == 0 or v >= 5 else None for k, v in json.loads(value).items()}) if isinstance(value, str) else value)
    return d


def main(argv=None):
    try:
        result = run(parse_args(argv))
    except (ContractError, FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()

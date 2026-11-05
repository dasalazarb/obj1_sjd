#!/usr/bin/env python3
"""Validate every Step 17 V3 outcome without modifying its original bundles."""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
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
from src.block_A.temporal_spline_config import VERSION, STEM, C_SUPPORT_POLICY, BASIS_SPECIFICATIONS, SplineConfig
from src.block_A.temporal_spline_contract import (
    code_paths, fingerprints, git_identity, load_inputs, measurements, metadata, sample_fingerprint, verify_unchanged,
)
from src.block_A.temporal_spline_feasibility import evaluate
from src.block_A.temporal_spline_models import fit_family
from src.block_A.temporal_spline_sensitivity import bootstrap_patients
from src.block_A.temporal_spline_outputs import (
    artifact_manifest, preflight, install_bundle, write_manifest, write_table,
)
from src.block_A import temporal_spline_validation as v


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--integrated', type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    p.add_argument('--baseline', type=Path, default=common.INTEGRATED_BASELINE_PARQUET)
    p.add_argument('--patterns', type=Path, default=common.DISEASE_ACTIVITY_PROGRESSION_DIR/'13_essdai_patient_observed_patterns_v2.parquet')
    p.add_argument('--patterns-manifest', type=Path, default=common.BLOCKA_QC_DIR/'13_disease_activity_progression/13_essdai_observed_pattern_manifest_v2.json')
    p.add_argument('--context', type=Path)
    p.add_argument('--source-registry', type=Path)
    p.add_argument('--input-origin', choices=('canonical', 'synthetic_validation'), default='canonical')
    p.add_argument('--output-root', type=Path, default=common.PROJECT_ROOT)
    p.add_argument('--reference-root', type=Path, help='Project containing complete V3 bundles; default: output-root')
    p.add_argument('--data-cutoff', help='Must match all four reference manifests; default: derive from them')
    p.add_argument('--threshold', choices=('3', '5', 'both'), default='both')
    p.add_argument('--basis', choices=(*BASIS_SPECIFICATIONS, 'both'), default='both')
    p.add_argument('--depth', choices=('systematic', 'deep'), default='systematic')
    p.add_argument('--bootstrap-replicates', type=int, default=200)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--outcomes-file', type=Path, help='JSON list of outcome_id values prioritizing deep tasks only')
    p.add_argument('--max-outcomes', type=int, help='Deep outcome budget per invocation; systematic inventory remains complete')
    p.add_argument('--max-patient-refits', type=int, help='Deep refit budget per invocation; whole model tasks only')
    p.add_argument('--all-fitted', action='store_true', help='Explicitly include every fitted model in the deep queue (also the default)')
    p.add_argument('--resume', action='store_true', help='Reuse hash-verified aggregate checkpoints and publish a new snapshot')
    p.add_argument('--backup-root', type=Path, help='Optional persistent backup of replaced validation outputs')
    p.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)
    args.config = None  # upstream adapter; use authenticated reference parameters
    if args.data_cutoff:
        try:
            date.fromisoformat(args.data_cutoff)
        except ValueError:
            p.error('--data-cutoff requires YYYY-MM-DD')
    if args.bootstrap_replicates < 100:
        p.error('--bootstrap-replicates must be >=100 (default 200)')
    if args.seed < 0:
        p.error('--seed must be nonnegative')
    for name in ('max_outcomes', 'max_patient_refits'):
        if getattr(args, name) is not None and getattr(args, name) < 0:
            p.error('--'+name.replace('_', '-')+' must be nonnegative')
    if args.depth != 'deep' and (args.resume or args.all_fitted or args.max_outcomes is not None or
                               args.max_patient_refits is not None or args.outcomes_file):
        p.error('deep queue controls require --depth deep')
    return args


def _root(path, origin):
    return path.resolve()/('synthetic_validation' if origin == 'synthetic_validation' else '')


def _combine(rows):
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=v.BASE_COLUMNS)
    for col in v.BASE_COLUMNS:
        if col not in frame:
            frame[col] = ''
    return frame


def _reproduce_fit(reference, evaluation, spec, cfg, outcome_id, costs=None):
    """Refit the original coherent family once, then check all published terms."""
    fits, attempts, _ = fit_family(evaluation, spec, cfg)
    if costs is not None:
        costs['reference_reproduction_attempts'] += len(attempts)
    models = reference['tables']['01_model_comparison']
    coefficients = reference['tables']['01_model_coefficients']
    comparable = {}
    for model in 'ABC':
        ref = models.loc[models.outcome_id.eq(outcome_id) & models.model.eq(model)].iloc[0]
        actual = fits.get(model)
        if ref.fit_status != 'ESTIMATED':
            comparable[model] = actual is None
            continue
        terms = coefficients.loc[coefficients.outcome_id.eq(outcome_id) & coefficients.model.eq(model)].sort_values('term_index')
        comparable[model] = bool(actual is not None and actual.family == ref.model_family and
            len(terms) == len(actual.beta) and
            np.array_equal(terms.term_index.to_numpy(), np.arange(len(actual.beta))) and
            np.allclose(terms.estimate.to_numpy(float), actual.beta, rtol=1e-5, atol=1e-6) and
            np.allclose(terms.SE.to_numpy(float), np.sqrt(np.diag(actual.covariance)), rtol=1e-5, atol=1e-6))
    return fits, comparable


def run(args):
    started = time.monotonic()
    output_root = _root(args.output_root, args.input_origin)
    reference_root = _root(args.reference_root or args.output_root, args.input_origin)
    references, reference_snapshot = {}, {}
    # Authenticate every source bundle even when the requested snapshot covers
    # only one threshold/basis. Never silently accept an incomplete denominator.
    for basis in BASIS_SPECIFICATIONS:
        for threshold in (3, 5):
            ref = v.authenticate_reference(reference_root, basis, threshold, args.input_origin)
            references[(basis, threshold)] = ref
            reference_snapshot.update({f'{basis}_delta{threshold}_{k}': item for k, item in ref['snapshot'].items()})
    manifests = [r['manifest'] for r in references.values()]
    cutoffs = {m['data_cutoff'] for m in manifests}
    if len(cutoffs) != 1 or args.data_cutoff and args.data_cutoff not in cutoffs:
        raise ContractError('NOT_COMPARABLE_REFERENCE: data_cutoff')
    args.data_cutoff = next(iter(cutoffs))
    date.fromisoformat(args.data_cutoff)
    parameters = [m['parameters'] for m in manifests]
    if len({v.digest(p) for p in parameters}) != 1:
        raise ContractError('NOT_COMPARABLE_REFERENCE: inconsistent parameters')
    cfg = SplineConfig(**parameters[0])
    if cfg.max_primary_years != 3.:
        raise ContractError('validation requires the original three-year main horizon')
    deep_cfg = replace(cfg, bootstrap_replicates=args.bootstrap_replicates)
    paths = code_paths(ROOT)
    paths['src/block_A/17_validate_temporal_heterogeneity.py'] = Path(__file__).resolve()
    code_snapshot = fingerprints(paths)
    x, _, patterns, specs, _, inputs, upstream = load_inputs(args)
    patient_ids = tuple(patterns.patient_id.unique())
    identities = {metadata(s, 3, 'clinical_baseline', cfg)['outcome_id'] for s in specs.values()}
    if len(identities) != len(specs):
        raise ContractError('duplicate canonical outcome identity')
    for ref in references.values():
        if set(ref['tables']['00_variable_eligibility'].outcome_id) != identities:
            raise ContractError('NOT_COMPARABLE_REFERENCE: complete catalog inventory differs')
    priority = []
    if args.outcomes_file:
        priority = json.loads(args.outcomes_file.read_text())
        if not isinstance(priority, list) or any(not isinstance(o, str) for o in priority) or not set(priority) <= identities:
            raise ContractError('--outcomes-file requires registered outcome_id JSON list')
        inputs.update(fingerprints({'deep_priority_outcomes': args.outcomes_file}))
    thresholds = (3, 5) if args.threshold == 'both' else (int(args.threshold),)
    bases = BASIS_SPECIFICATIONS if args.basis == 'both' else (args.basis,)
    destinations = {f'{area}/{basis}/delta{threshold}': path
        for basis in bases for threshold in thresholds
        for area, path in v.directories(output_root, basis, threshold).items()}
    # Overwrite ONLY validation destinations. Original V3 references are never destinations.
    existing = any(path.exists() for path in destinations.values())
    auto_backup_root = None
    replacement_backup_root = args.backup_root
    if existing and replacement_backup_root is None:
        backup_parent = output_root / 'outputs' / 'backups' / STEM
        backup_parent.mkdir(parents=True, exist_ok=True)
        auto_backup_root = Path(tempfile.mkdtemp(
            prefix='validation-autoreplace-', dir=backup_parent
        ))
        replacement_backup_root = auto_backup_root

    preflight(
        output_root, destinations, dict(inputs, **reference_snapshot),
        overwrite=existing or args.resume,
        backup_root=replacement_backup_root,
    )
    bundles, evaluations, inventory, sensitive, overlaps = {}, {}, [], set(), {}
    for threshold in thresholds:
        groups = fixed_groups(patterns, threshold)
        immutable = groups.copy(deep=True)
        for index, original_spec in enumerate(specs.values(), 1):
            if args.progress:
                print(f'Step17 global validation delta{threshold}: {index}/{len(specs)} systematic', flush=True)
            frame, observation_status, _, unit = measurements(x, groups, original_spec, args.data_cutoff, 'clinical_baseline', cfg)
            spec = replace(original_spec, unit=unit)
            for basis in bases:
                evaluation = evaluate(frame, groups, spec, cfg, 'clinical_baseline', 3., 3., basis)
                meta = {k: val for k, val in metadata(spec, threshold, 'clinical_baseline', cfg).items()
                        if k in ('outcome_id', 'domain', 'variable', 'outcome_type', 'unit', 'threshold_abs', 'time_axis')}
                meta.update(scenario='main', basis_specification=basis, validation_version=v.VALIDATION_VERSION,
                    validation_depth=args.depth, seed=args.seed, sample_fingerprint=sample_fingerprint(evaluation['data']),
                    n_patients_analytic=evaluation['gate']['n_patients'], n_observations_analytic=len(evaluation['data']),
                    observation_status=observation_status)
                ref = references[(basis, threshold)]
                meta.update(data_cutoff=args.data_cutoff,
                    reference_manifest_sha256=ref['snapshot']['reference_manifest']['sha256'],
                    code_sha256=v.digest({k: item['sha256'] for k, item in code_snapshot.items()}),
                    upstream_sha256=v.digest({k: item['sha256'] for k, item in inputs.items()}))
                comparable, reason = v.reference_comparison(ref, evaluation, spec, meta,
                    {k: val for k, val in inputs.items() if k != 'deep_priority_outcomes'})
                diagnostics = v.systematic(evaluation, groups, spec, meta, cfg)
                if diagnostics['sensitive']:
                    sensitive.add(meta['outcome_id'])
                rows = v.inventory_rows(ref, evaluation, spec, meta, comparable, reason, args.depth)
                for row in rows:
                    row.update(n_supported_grid_times=evaluation['gate']['n_supported_grid_times'],
                        supported_grid_span_years=evaluation['gate']['supported_grid_span_years'],
                        n_paired_patients=int(diagnostics['changes'].n_paired.sum()),
                        minimum_longitudinal_overlap=diagnostics['minimum_overlap'],
                        longitudinal_continuity_fraction=diagnostics['minimum_continuity_fraction'],
                        continuity_rule=diagnostics['continuity_rule'],
                        local_support_status='SUPPORTED_GRID' if evaluation['gate']['n_supported_grid_times'] else 'NO_SUPPORTED_GRID')
                inventory.extend(rows)
                bundle = bundles.setdefault((basis, threshold), {name: [] for name in v.TABLES})
                bundle['07_patient_temporal_coverage'].extend(diagnostics['coverage'].to_dict('records'))
                bundle['07_within_patient_change'].extend(diagnostics['changes'].to_dict('records'))
                bundle['07_distribution_diagnostics'].extend(diagnostics['distributions'].to_dict('records'))
                key = (meta['outcome_id'], threshold, basis)
                overlaps[key] = diagnostics['minimum_continuity_fraction']
                # Systematic mode does not retain patient frames. Deep mode keeps
                # only main analytic frames in process memory, never on disk.
                if args.depth == 'deep':
                    evaluations[key] = (evaluation, spec, meta)
            pd.testing.assert_frame_equal(groups, immutable)
    plan = v.frozen_plan(inventory, priority, args.seed)
    context = dict(version=v.VALIDATION_VERSION, cutoff=args.data_cutoff,
        bootstrap_policy=v.BOOTSTRAP_POLICY, sensitivity_policy=v.SENSITIVITY_POLICY,
        calibration_denominator=v.CALIBRATION_DENOMINATOR, original_support_policy=C_SUPPORT_POLICY,
        inputs={k: i['sha256'] for k, i in inputs.items()},
        references={k: i['sha256'] for k, i in reference_snapshot.items()},
        code={k: i['sha256'] for k, i in code_snapshot.items()},
        parameters=parameters[0], validation_seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates, materiality=v.MATERIAL,
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__, statsmodels=statsmodels.__version__))
    cache = None
    if args.depth == 'deep':
        cache_root = output_root/'work'/STEM/VERSION/v.VALIDATION_VERSION
        # Threshold/basis scope is fixed; budgets may grow across resume calls.
        scope = v.digest(dict(thresholds=thresholds, bases=bases))[:16]
        cache = v.Checkpoints(cache_root/scope, context, plan, args.resume, patient_ids)
    lookup = {tuple(row[k] for k in ('outcome_id', 'threshold_abs', 'basis_specification', 'model')): row for row in inventory}
    actual_costs = dict(reference_reproduction_attempts=0, loo_attempts=0, bootstrap_attempts=0)
    consumed, selected_outcomes, task_states, deep_results, reproduced = 0, set(), [], {}, {}
    for task in plan['tasks']:
        key = (task['outcome_id'], task['threshold_abs'], task['basis_specification'])
        model = task['model']
        row = lookup[(*key, model)]
        payload = cache.load(task['task_id']) if cache else None
        source = 'CHECKPOINT' if payload else 'PENDING'
        if cache and payload is None:
            evaluation, spec, meta = evaluations[key]
            # Budget includes original family reconstruction (up to two
            # families x three models), full LOO and requested bootstrap.
            cost = evaluation['data'].patient_id.nunique()+(args.bootstrap_replicates if spec.type != 'binary' else 0)+(6 if key not in reproduced else 0)
            within_outcomes = args.max_outcomes is None or task['outcome_id'] in selected_outcomes or len(selected_outcomes) < args.max_outcomes
            within_refits = args.max_patient_refits is None or consumed+cost <= args.max_patient_refits
            if within_outcomes and within_refits:
                selected_outcomes.add(task['outcome_id'])
                consumed += cost
                if args.progress:
                    print(f'Step17 global validation deep task {task["queue_position"]}/{len(plan["tasks"])}: model {model}', flush=True)
                task_start = time.monotonic()
                if key not in reproduced:
                    reproduced[key] = _reproduce_fit(references[(key[2], key[1])], evaluation, spec, cfg, key[0], actual_costs)
                fits, comparison = reproduced[key]
                if not comparison[model] or model not in fits:
                    loo = dict(model=model, status='NOT_COMPARABLE_REFERENCE', reason='Reference family/coefficients/covariance not reproduced',
                        n_requested=0, n_successful=0, n_failed=0)
                    influence = []
                    boot = dict(model=model, status='NOT_COMPARABLE_REFERENCE', reason=loo['reason'],
                        calibration_valid=False, n_requested=args.bootstrap_replicates, n_successful=0, n_failed=0)
                    scientific = 'NOT_COMPARABLE_REFERENCE'
                else:
                    loo, influence = v.loo_summary(evaluation, fits[model], spec, cfg, args.seed)
                    boot = bootstrap_patients(evaluation, fits, spec, deep_cfg, model=model, validation_seed=args.seed)
                    actual_costs['loo_attempts'] += loo.get('n_attempted', loo['n_requested'])
                    actual_costs['bootstrap_attempts'] += boot.get('n_attempted', boot['n_requested'])
                    scientific = v.scientific_status(row, overlaps[key], loo, boot)
                if not influence:
                    influence = [dict(model=model, status='NOT_ESTIMABLE',
                        reason='No comparable aggregate LOO estimands', n_requested=loo['n_requested'],
                        n_successful=loo['n_successful'], n_failed=loo['n_failed'])]
                safe_loo = v.suppress_table(pd.DataFrame([dict(meta, **loo)]), patient_ids, sensitive).iloc[0]
                safe_boot = v.suppress_table(pd.DataFrame([dict(meta, **boot)]), patient_ids, sensitive).iloc[0]
                if 'DISCLOSURE_SUPPRESSED' in (safe_loo.disclosure_status, safe_boot.disclosure_status):
                    scientific = 'INCONCLUSIVE' if scientific not in ('NOT_COMPARABLE_REFERENCE', 'NOT_ESTIMABLE') else scientific
                payload = dict(scientific_status=scientific, duration_seconds=time.monotonic()-task_start,
                    loo=v.suppress_table(pd.DataFrame([dict(meta, **loo)]), patient_ids, sensitive).to_dict('records'),
                    bootstrap=v.suppress_table(pd.DataFrame([dict(meta, **boot)]), patient_ids, sensitive).to_dict('records'),
                    influence=v.suppress_table(pd.DataFrame([dict(meta, **i) for i in influence]), patient_ids, sensitive).to_dict('records'))
                cache.save(task['task_id'], payload)
                source = 'EXECUTED'
        if payload:
            deep_results[(*key, model)] = payload
            target = bundles[(key[2], key[1])]
            for table_name, field in [('07_fit_stability', 'loo'), ('07_patient_bootstrap_qc', 'bootstrap'),
                                      ('07_influence_summary', 'influence'), ('07_bootstrap_failure_diagnostics', 'bootstrap')]:
                target[table_name].extend(dict(r, duration_seconds=payload['duration_seconds']) for r in payload[field])
        task_states.append(dict(task_id=task['task_id'], queue_position=task['queue_position'],
            status='DEEP_COMPLETE' if payload else 'DEEP_PENDING_RESOURCE_BUDGET', source=source))
        row['queue_position'] = task['queue_position']
    for row in inventory:
        key = tuple(row[k] for k in ('outcome_id', 'threshold_abs', 'basis_specification'))
        result = deep_results.get((*key, row['model']))
        status = result['scientific_status'] if result else v.scientific_status(row, overlaps[key])
        row['validation_coverage'] = ('REFERENCE_MISMATCH' if status == 'NOT_COMPARABLE_REFERENCE' else
            'DEEP_COMPLETE' if result else 'DEEP_PENDING' if row['eligible_for_deep'] and args.depth == 'deep' else row['status'])
        row['status'] = row['validation_coverage']
        row['loo_status'] = result['loo'][0]['status'] if result else 'NOT_RUN'
        row['bootstrap_status'] = result['bootstrap'][0]['status'] if result else 'NOT_RUN'
        row.update(v.validation_dimensions(result, dict(row, status=status)))
        target = bundles[(key[2], key[1])]
        target['07_outcome_validation_inventory'].append(row)
        classification = dict(row, status=status, scientific_validation_status=status,
            software_qc_status='FAIL' if status == 'NOT_COMPARABLE_REFERENCE' else 'PASS',
            reason=row['reason'] or ('Intensive validation has not been completed' if status == 'DEEP_PENDING_RESOURCE_BUDGET' else
                                    'Exploratory descriptive stability; inspect continuous diagnostics'))
        target['07_scientific_validation_classification'].append(classification)
        if result is None:
            for name in ('07_fit_stability', '07_patient_bootstrap_qc', '07_influence_summary', '07_bootstrap_failure_diagnostics'):
                target[name].append(dict(row, status=status, n_requested=0, n_successful=0, n_failed=0,
                    reason='No intensive result; see inventory eligibility and frozen queue'))
    verify_unchanged(inputs)
    verify_unchanged(reference_snapshot)
    verify_unchanged(code_snapshot)
    scratch = output_root/'work'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='step17-validation-stage-', dir=scratch) as temporary:
        os.chmod(temporary, 0o700)
        staging = {}
        for (basis, threshold), collected in bundles.items():
            tables = {name: v.suppress_table(_combine(rows), patient_ids, sensitive) for name, rows in collected.items()}
            paths = {}
            for area in ('tables', 'qc'):
                name = f'{area}/{basis}/delta{threshold}'
                paths[area] = Path(temporary)/name
                paths[area].mkdir(parents=True, mode=0o700)
                staging[name] = paths[area]
            for name, table in tables.items():
                write_table(table, paths['tables']/(name+'.csv'), patient_ids)
            inv = tables['07_outcome_validation_inventory']
            classes = tables['07_scientific_validation_classification']
            bundle_tasks = [task for task in plan['tasks'] if task['basis_specification'] == basis and task['threshold_abs'] == threshold]
            task_ids = {t['task_id'] for t in bundle_tasks}
            complete = classes.loc[classes.validation_coverage.eq('DEEP_COMPLETE')]
            # Do not reconstruct suppressed patient denominators through cost totals.
            hidden_costs = any(record.get('disclosure_status') == 'DISCLOSURE_SUPPRESSED'
                for result in deep_results.values() for field in ('loo', 'bootstrap') for record in result[field])
            summary = dict(validation_version=v.VALIDATION_VERSION,
                n_outcomes_inventory=inv.outcome_id.nunique(), n_outcome_models=len(inv),
                by_model={m: dict(n_inventory=int(inv.model.eq(m).sum()),
                    n_eligible_for_deep=int((inv.model.eq(m) & inv.eligible_for_deep).sum()),
                    n_deep_complete=int(complete.model.eq(m).sum())) for m in 'ABC'},
                by_domain={str(domain): dict(n_inventory=len(part),
                    n_deep_complete=int(part.validation_coverage.eq('DEEP_COMPLETE').sum()))
                    for domain, part in inv.groupby('domain')},
                scientific_status_counts=classes.status.value_counts().to_dict(),
                loo_stability_status_counts=classes.loo_stability_status.value_counts().to_dict(),
                bootstrap_calibration_status_counts=classes.bootstrap_calibration_status.value_counts().to_dict(),
                bootstrap_failure_diagnostics=tables['07_bootstrap_failure_diagnostics'].to_dict('records'),
                stability_denominator=len(complete),
                n_supported_exploratory=int(complete.status.eq('SUPPORTED_EXPLORATORY').sum()),
                stability_fraction=float(complete.status.eq('SUPPORTED_EXPLORATORY').mean()) if len(complete) else None,
                stability_scope='Only intensively completed models; selected subset is not population robustness',
                tasks=[t for t in task_states if t['task_id'] in task_ids],
                refit_qc={name: dict(
                    status_counts=tables[name].status.value_counts().to_dict(),
                    aggregate_counts={col: (int(pd.to_numeric(tables[name][col], errors='coerce').sum())
                        if len(tables[name]) and pd.to_numeric(tables[name][col], errors='coerce').notna().all() else None)
                        for col in ('n_requested', 'n_successful', 'n_failed')},
                    suppression='Counts omitted when any contributing task tally is disclosure-suppressed')
                    for name in ('07_fit_stability', '07_patient_bootstrap_qc')},
                selection=plan, resource_budget=dict(max_patient_refits=args.max_patient_refits,
                    max_outcomes=args.max_outcomes, reserved_refits_this_invocation=None if hidden_costs else consumed,
                    actual_refits_this_invocation={k: None if hidden_costs else value for k, value in actual_costs.items()},
                    disclosure_status='DISCLOSURE_SUPPRESSED' if hidden_costs else 'AGGREGATE'),
                limitations=['Retrospective associations, not causal progression/prediction',
                    'Same-cohort bootstrap is not independent validation',
                    'Three-knot specification introduced post hoc; exploratory inference',
                    'Local overlapping support windows do not establish individual continuity',
                    'Materiality thresholds are descriptive standardized rules, not clinical MCIDs',
                    'Binary bootstrap not validated; binary models cannot be classified supported',
                    'Whole-model checkpoints; interrupted model restarts from its first refit',
                    'Conservative cross-table complementary suppression may hide usable aggregates'],
                relationship_to_reference='Read-only V3 main; original full-family q values unchanged')
            # These are OUTCOME/MODEL task counts, never patient counts.
            v.audit_payload(summary, patient_ids)
            write_manifest(paths['qc']/'validation_summary.json', v.json_safe(summary))
            manifest = dict(version=VERSION, validation_version=v.VALIDATION_VERSION,
                basis_specification=basis, threshold_abs=threshold, time_axis='clinical_baseline', scenario='main',
                bootstrap_policy=v.BOOTSTRAP_POLICY, sensitivity_policy=v.SENSITIVITY_POLICY,
                original_support_policy=C_SUPPORT_POLICY, calibration_denominator=v.CALIBRATION_DENOMINATOR,
                sample_fingerprints=inv[['outcome_id', 'sample_fingerprint']].drop_duplicates().to_dict('records'),
                c_support_policy=C_SUPPORT_POLICY, input_origin=args.input_origin, data_cutoff=args.data_cutoff,
                software_qc_status='FAIL' if classes.status.eq('NOT_COMPARABLE_REFERENCE').any() else 'PASS',
                scientific_status='EXPLORATORY_NOT_CLINICALLY_REVIEWED', validation_depth=args.depth,
                full_inventory_scope='all_registered_variables', parameters=parameters[0], materiality=v.MATERIAL,
                context_hash=v.digest(context), plan_hash=v.digest(plan), selection_plan=plan,
                code=git_identity(ROOT), code_hashes=code_snapshot, inputs=inputs,
                reference_hashes=reference_snapshot, upstream_status=upstream.get('qc_status'),
                software=context['software'], seed=args.seed, created_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=time.monotonic()-started,
                privacy='Aggregate-only; cross-table complementary suppression; NIH disclosure review before release',
                artifacts=artifact_manifest(paths['tables'])+[
                    dict(path='qc/'+a['path'], sha256=a['sha256']) for a in artifact_manifest(paths['qc'])])
            v.audit_payload(manifest, patient_ids)
            write_manifest(paths['qc']/'run_manifest.json', v.json_safe(manifest))
        verify_unchanged(inputs)
        verify_unchanged(reference_snapshot)
        verify_unchanged(code_snapshot)
        install_bundle(staging, destinations, replacement_backup_root)
        if auto_backup_root is not None:
            shutil.rmtree(auto_backup_root)
    mismatch = any(not row['reference_comparable'] for row in inventory) or any(
        result['scientific_status'] == 'NOT_COMPARABLE_REFERENCE' for result in deep_results.values())
    return dict(validation_version=v.VALIDATION_VERSION, input_origin=args.input_origin,
        software_qc_status='FAIL' if mismatch else 'PASS',
        published_bundles=len(bundles), n_inventory_models=len(inventory),
        n_deep_candidates=len(plan['tasks']), n_deep_complete=len(deep_results),
        n_deep_pending=len(plan['tasks'])-len(deep_results),
        data_cutoff=args.data_cutoff, duration_seconds=time.monotonic()-started)


def main(argv=None):
    try:
        result = run(parse_args(argv))
    except (ContractError, FileExistsError, FileNotFoundError, ValueError) as error:
        # Errors are codes/paths from authenticated infrastructure, never model
        # warnings containing patient rows. Do not print data or raw exceptions.
        raise SystemExit(str(error)) from None
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

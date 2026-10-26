"""Step 16 read-only adapter to authenticated Step 13 / Step 15 contracts."""
from dataclasses import replace
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from src.block_A.secondary_trajectory_analysis import (
    ContractError, attach_context, fixed_groups, observations, read_table, sha256,
    validate_inputs, verified_manifest,
)
from src.block_A.secondary_trajectory_catalog import build_catalog


def fingerprints(paths):
    return {role: dict(path=str(Path(p).resolve()), sha256=sha256(p)) for role, p in paths.items()}


def verify_unchanged(snapshot):
    for role, info in snapshot.items():
        if not Path(info['path']).is_file() or sha256(info['path']) != info['sha256']:
            raise ContractError('input/code changed during run: ' + role)


def code_paths(root):
    """Resolve the root first: /data and /vf/users aliases have identical keys."""
    root = Path(root).resolve()
    paths = [root/'common.py', root/'config.py', root/'src/derivations/pro_scoring.py']
    for pattern in ('secondary_comparison_*.py', 'secondary_trajectory_*.py',
                    'observed_patterns_v2*.py', '16_compare_secondary_trajectories.py'):
        paths.extend((root/'src/block_A').glob(pattern))
    return {str(p.resolve().relative_to(root)): p.resolve() for p in sorted(set(paths))}


def load_inputs(args):
    required = (args.integrated, args.baseline, args.patterns, args.patterns_manifest)
    if any(not p.is_file() for p in required):
        raise ContractError('BLOCKED_UPSTREAM: missing canonical Step 10/11/13 files')
    # Capture bytes BEFORE validation; detect changes while reading, too.
    paths = dict(step10=args.integrated, step11=args.baseline, patterns=args.patterns,
                 step13_manifest=args.patterns_manifest)
    initial = fingerprints(paths)
    manifest, protected = verified_manifest(args.patterns_manifest, args.patterns,
        args.integrated, args.baseline, args.input_origin)
    paths.update(protected)
    manifest_root = args.patterns_manifest.resolve().parents[4]
    paths.update({f'step13_artifact_{i}': (manifest_root/entry['path']).resolve()
                  for i, entry in enumerate(manifest['artifacts'])})
    for role, explicit, default in (
        ('context', args.context, args.integrated.with_name('10_integrated_longitudinal_context.parquet')),
        ('source_registry', args.source_registry, args.integrated.with_name('10_variable_registry.csv')),
    ):
        path = explicit or default
        if path.is_file():
            paths[role] = path
        elif explicit:
            raise ContractError('missing explicit ' + role)
    if args.config:
        paths['config'] = args.config
    snapshot = fingerprints(paths)
    verify_unchanged(initial)
    x, b, p = validate_inputs(read_table(args.integrated), read_table(args.baseline),
        read_table(protected['patterns']), read_table(protected['eligibility']),
        read_table(protected['selected_dates']), args.data_cutoff)
    specs, catalog = build_catalog(x, read_table(paths['source_registry']) if 'source_registry' in paths else None)
    x = attach_context(x, read_table(paths['context']) if 'context' in paths else None)
    # Step 15's overlap status is not flagged: classify conservatively here,
    # without changing its catalog or any upstream producer.
    specs = {col: replace(s, is_essdai_derived=True)
             if col.startswith(('essdai__', 'ovl__')) else s for col, s in specs.items()}
    catalog.loc[catalog.canonical_column.str.startswith(('essdai__', 'ovl__')), 'is_essdai_derived'] = True
    verify_unchanged(snapshot)
    # Close the verification-to-fingerprint race for every manifest artifact.
    verified_manifest(args.patterns_manifest, args.patterns, args.integrated, args.baseline, args.input_origin)
    return x, b, p, specs, catalog, snapshot, manifest


def measurements(x, groups, spec, cutoff, axis, cfg):
    """Select minimal columns, reuse Step 15 date/duplicate/unit/reference policy."""
    stem = spec.canonical_column.removesuffix('__value')
    instrument = spec.canonical_column.split('__')[-1].split('_')[0]
    names = {'patient_id', 'clinical_episode_id', 'clinical_baseline_episode_id',
             'clinical_baseline_date', 'clinical_anchor_date', 'is_clinical_baseline',
             'source_protocol', spec.canonical_column, spec.measurement_date_column}
    names.update(c for c in x if c.startswith((stem+'__', 'pro__'+instrument+'_'))
                 and c.endswith(('unit', 'assay', 'platform', 'scoring_version', 'conflict')))
    names.add(spec.canonical_column+'_conflict')
    frame, status, qc, unit = observations(x[[c for c in x if c in names]], groups,
        spec, cutoff, axis, cfg.window_years)
    if not frame.empty:
        frame['time_since_variable_reference_years'] = (frame.variable_measurement_date-frame.reference_date).dt.days/365.25
        frame['threshold_abs'] = int(groups.threshold_abs.iloc[0])
        frame['variable_id'], frame['canonical_column'] = spec.variable_id, spec.canonical_column
        frame['domain'], frame['outcome_type'], frame['unit'] = spec.domain, spec.type, unit
    return frame, status, qc, unit


def inference_frame(frame, axis):
    if frame.empty:
        return frame.copy()
    data = frame.loc[frame.time.ge(0)].copy() if axis == 'essdai_index' else frame.copy()
    # Support must be recomputed after filtering preindex measurements.
    data['n_analysis_dates'] = data.groupby('patient_id').value.transform('size')
    return data


def metadata(spec, threshold, axis, cfg):
    return dict(variable=spec.variable_id, domain=spec.domain, threshold_abs=threshold,
        time_axis=axis, unit=spec.unit, outcome_type=spec.type,
        is_essdai_derived=spec.is_essdai_derived,
        interpretation_group='mechanically_related' if spec.is_essdai_derived else 'independent',
        family=cfg.family(spec), is_priority=spec.variable_id in cfg.priority_variables, reference_policy=spec.reference_policy,
        clinical_review_status='NOT_CLINICALLY_REVIEWED',
        estimand='descriptive association conditional on retrospective fixed ESSDAI group')


def git_identity(root):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    return dict(commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                worktree_dirty=bool(git('status', '--porcelain')))

#!/usr/bin/env python3
"""New restricted, human-approved Graph03 N3 analysis; never refits Mapper."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from uuid import uuid4

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.graph.stage03.s03_n3_controlled import input_pins, run_controlled
from src.studies.graph.stage03.s03_n3_decisions import decision_templates
from src.studies.graph.stage03.s03_n3_phasec_adjudication import CANDIDATES
from src.studies.graph.stage03.s03_n3_resolution import _check_overlap
from src.studies.graph.stage03.s03_run_resolution import (
    graph_paths, resolve_graph03_dependencies,
)
from src.studies.graph._pipeline import config_path


def _configured_inputs():
    """Resolve inputs only from the existing versioned project configs."""
    pipeline_file = ROOT / 'src/studies/graph/configs/pipeline.yaml'
    controlled_file = ROOT / 'src/studies/graph/configs/n3_controlled_inference.yaml'
    pipeline = yaml.safe_load(pipeline_file.read_text())
    controlled = yaml.safe_load(controlled_file.read_text())
    data_root = Path(os.environ.get('OBJ1_SJD_ROOT', ROOT)).resolve()

    def data_location(path):
        resolved = Path(path).resolve()
        if data_root != ROOT and resolved.is_relative_to(ROOT):
            return data_root / resolved.relative_to(ROOT)
        return resolved

    integrated = data_location(config_path(pipeline_file, pipeline['paths']['integrated']))
    # Step10 publishes this alongside the integrated analytic parquet.
    context = integrated.with_name('10_integrated_longitudinal_context.parquet')
    measurement_setting = controlled.get('source_paths', {}).get('measurements')
    if not isinstance(measurement_setting, str) or not measurement_setting.strip():
        raise ValueError('N3_MEASUREMENT_SOURCE_MUST_BE_CONFIGURED')
    measurements = data_location(config_path(controlled_file, measurement_setting))
    return context, measurements


def _source_readiness(output_dir, context, measurements):
    """Report named upstream schemas only; never fabricate/export patient rows."""
    import re
    source_root = ROOT / 'data/intermediate'
    paths = {
        'step10_context': Path(context),
        'step01_lab_analytes': source_root / 'block_A/01_serological_profile/01_labs_episode_analyte_level.parquet',
        'step01_essdai_esspri': source_root / '01_pop_distribution/01_visit_level_classification.parquet',
        'step01_extended_clinical': source_root / 'block_A/01_extended_clinical_phenotype/01_extended_clinical_phenotype_longitudinal.parquet',
        'required_instrument_measurements': Path(measurements),
    }
    inventory = {}
    for label, path in paths.items():
        info = {'path': str(path), 'exists': path.is_file()}
        if path.is_file():
            try:
                import pyarrow.parquet as pq
                columns = set(pq.read_schema(path).names)
                info['has_patient_episode_keys'] = {'patient_id', 'clinical_episode_id'} <= columns
                info['date_columns'] = sorted(x for x in columns
                    if re.search(r'(?:date|datetime|specimen)', x, flags=re.I))
                info['instrument_fields'] = sorted(columns & {
                    'source_column', 'value', 'measurement_date',
                    'assay_id', 'unit', 'sample_id'})
                info['selected_lab_fields'] = sorted(columns & {
                    'lab_id', 'selected_lab_date', 'selected_value_numeric',
                    'selected_value_text', 'selected_unit', 'selection_status',
                    'unit_conflict', 'result_conflict'})
            except (ImportError, OSError, ValueError):
                info['schema_inspection'] = 'UNAVAILABLE'
        inventory[label] = info
    candidate_status = 'NOT_ATTEMPTED'
    candidate_artifact = ''
    if Path(context).is_file() and not Path(measurements).is_file():
        try:
            from src.studies.graph.stage03.s03_n3_source_adapter import build_candidate
            candidate_path = output_dir / 'unapproved_lab_measurement_candidates.parquet'
            candidate = build_candidate(ROOT, Path(context), candidate_path)
            candidate_status = candidate['status']
            candidate_artifact = str(candidate_path)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            # Emit only controlled reason tokens, never source values.
            import re as _re
            token = str(exc)
            candidate_status = token if _re.fullmatch('[A-Z][A-Z0-9_]+', token) else 'ADAPTER_ERROR'
    report = {
        'status': 'SOURCE_CONTRACT_NOT_READY',
        'estimators_run': 0,
        'candidate_status': candidate_status,
        'candidate_artifact': candidate_artifact,
        'candidate_is_approved': False,
        'required_instrument_columns': [
            'patient_id', 'clinical_episode_id', 'source_column', 'value',
            'measurement_date', 'assay_id', 'unit', 'sample_id'],
        'upstream_sources': inventory,
        'action': ('Construct and independently review an instrument-level source. '
                   'Never substitute clinical episode dates for instrument dates, '
                   'invent sample IDs, conflate assays or auto-approve endpoints.'),
    }
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = output_dir / 'source_readiness.json'
    target.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    target.chmod(0o600)
    return {'status': report['status'], 'report': str(target),
            'output_dir': str(output_dir), 'estimators_run': 0}


def _fill_automatic_inputs(args, parser):
    """One-command mode; keep explicit overrides for reproducibility/debugging."""
    if args.graph03_run is None or args.graph02_release is None:
        selection = SimpleNamespace(
            graph03_run=args.graph03_run,
            graph02_release=args.graph02_release,
            config=ROOT / 'src/studies/graph/configs/n3_phasec_review.yaml',
            strict=False,
        )
        graph03, release, _, _ = resolve_graph03_dependencies(selection, ROOT, phase='C')
        args.graph03_run, args.graph02_release = graph03, release

    if args.context is None or args.measurements is None:
        context, measurements = _configured_inputs()
        args.context = args.context or context
        args.measurements = args.measurements or measurements

    output_root = graph_paths(ROOT)['graph03'].parent / '03_graph_n3_controlled_inference'
    unique_run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8]
    if not Path(args.context).is_file() or not Path(args.measurements).is_file():
        args.source_readiness = _source_readiness(
            output_root / 'runs' / unique_run, args.context, args.measurements)
        return args
    if args.output_dir is None:
        args.output_dir = output_root / 'runs' / unique_run
    if args.plan is None:
        active = output_root / 'plans' / 'active.yaml'
        if active.is_file():
            args.plan = active
            if not args.prepare_plan:
                # A deliberately activated, content-pinned plan is execution
                # intent. run_controlled still blocks every unapproved model.
                args.execute = True
        else:
            if args.execute:
                parser.exit(2, '[Graph03] No active approved N3 plan. '
                            'Create and review a draft first.\n')
            args.prepare_plan = True
            args.plan = output_root / 'plans' / ('draft_' + unique_run + '.yaml')
    if args.prepare_plan:
        args.plan.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.decision_root is None:
        args.decision_root = output_root / 'decisions'
    return args



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['graph03-run', 'graph02-release', 'context', 'measurements', 'plan', 'output-dir']:
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--decision-root', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-plan', action='store_true',
                      help='Create an unsigned plan and review templates without reading clinical rows')
    mode.add_argument('--execute', action='store_true', help='Estimate only individually READY approved models')
    args = parser.parse_args(argv)
    try:
        args = _fill_automatic_inputs(args, parser)
        if getattr(args, 'source_readiness', None):
            print(json.dumps(args.source_readiness, sort_keys=True))
            return
        if args.prepare_plan:
            release = args.graph02_release.resolve()
            frozen_roots = [args.graph03_run.resolve(), release if release.is_dir() else release.parent]
            _check_overlap(args.output_dir.resolve(), frozen_roots)
            _check_overlap(args.plan.resolve(), frozen_roots)
            plan = yaml.safe_load((ROOT / 'src/studies/graph/configs/n3_controlled_inference.yaml').read_text())
            plan['input_sha256'] = input_pins(args.graph03_run, args.graph02_release,
                                             args.context, args.measurements, ROOT)
            templates = decision_templates(plan, plan['input_sha256'], ['R0','R1','R2','R3','R4'], list(CANDIDATES))
            if (args.plan.exists() or args.output_dir.exists() or args.output_dir == args.plan or
                    args.output_dir in args.plan.parents or args.plan in args.output_dir.parents):
                raise FileExistsError('DRAFT_PLAN_AND_TEMPLATE_OUTPUT_MUST_BE_NEW_AND_DISTINCT')
            args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            plan_created = False
            try:
                with args.plan.open('x') as handle:
                    plan_created = True
                    yaml.safe_dump(plan, handle, sort_keys=False)
                args.plan.chmod(0o600)
                path = args.output_dir / 'unsigned_decision_templates.yaml'
                path.write_text(yaml.safe_dump(templates, sort_keys=False))
                path.chmod(0o600)
            except BaseException:
                shutil.rmtree(args.output_dir)
                if plan_created:
                    args.plan.unlink()
                raise
            result = dict(status='UNSIGNED_DRAFT_NO_APPROVAL', plan=str(args.plan), output_dir=str(args.output_dir))
        else:
            result = run_controlled(args.graph03_run, args.graph02_release, args.context,
                args.measurements, args.output_dir, args.plan, args.decision_root, ROOT, execute=args.execute)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, yaml.YAMLError):
        # Clinical paths, free values and exception messages stay off console.
        parser.exit(1, '[Graph03] Controlled N3 stopped: invalid input, integrity or review contract.\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

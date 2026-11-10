#!/usr/bin/env python3
"""New restricted, human-approved Graph03 N3 analysis; never refits Mapper."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.graph.stage03.s03_n3_controlled import input_pins, run_controlled
from src.studies.graph.stage03.s03_n3_decisions import decision_templates
from src.studies.graph.stage03.s03_n3_phasec_adjudication import CANDIDATES
from src.studies.graph.stage03.s03_n3_resolution import _check_overlap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['graph03-run', 'graph02-release', 'context', 'measurements', 'plan', 'output-dir']:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--decision-root', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-plan', action='store_true',
                      help='Create an unsigned plan and review templates without reading clinical rows')
    mode.add_argument('--execute', action='store_true', help='Estimate only individually READY approved models')
    args = parser.parse_args(argv)
    try:
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
            if args.decision_root is None:
                parser.error('--decision-root is required for preflight/execute')
            result = run_controlled(args.graph03_run, args.graph02_release, args.context,
                args.measurements, args.output_dir, args.plan, args.decision_root, ROOT, execute=args.execute)
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError):
        # Clinical paths, free values and exception messages stay off console.
        parser.exit(1, '[Graph03] Controlled N3 stopped: invalid input, integrity or review contract.\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

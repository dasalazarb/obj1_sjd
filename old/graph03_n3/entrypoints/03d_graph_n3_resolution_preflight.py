#!/usr/bin/env python3
"""Phase D: check current results and scientific readiness; no estimators."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.graph.stage03.s03_n3_resolution import run_resolution
from src.studies.graph.stage03.s03_run_resolution import current_dependencies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph03-run', type=Path)
    parser.add_argument('--graph02-release', type=Path)
    parser.add_argument('--phasec-run', type=Path,
                        help='Optional existing Phase C directory; contracts are always checked')
    parser.add_argument('--output-dir', type=Path, help='Override the current diagnostic directory')
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/n3_phase_d_resolution.yaml')
    parser.add_argument('--preflight-only', action='store_true',
                        help='Document intent; estimation is prohibited with or without this flag')
    parser.add_argument('--strict', action='store_true', help='Also check optional historical pins and documentary code drift')
    parser.add_argument('--verbose', action='store_true', help='Show traceback on failure')
    args = parser.parse_args(argv)
    try:
        graph03, release, phasec = current_dependencies(args, ROOT, phase='D')
        result = run_resolution(graph03, release, phasec, args.output_dir, args.config, ROOT,
            strict=args.strict)
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        if args.verbose:
            raise
        parser.exit(1, f'[Graph03] {exc}\n')
    print(f'[Graph03] Phase D: {result.get("phased_status", result["status"])}')
    print(f'[Graph03] Resultado: {result["output_dir"]}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Phase C audit with automatic compatible frozen inputs; no estimators."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.studies.graph.stage03.s03_n3_phasec_adjudication import run_phasec
from src.studies.graph.stage03.s03_run_resolution import resolve_graph03_dependencies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph03-run',type=Path,help='Explicit existing Phase A run directory')
    parser.add_argument('--graph02-release',type=Path,help='Explicit frozen Graph02 manifest or directory')
    parser.add_argument('--output-dir',type=Path,help='Optional exact new run directory')
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/n3_phasec_review.yaml')
    parser.add_argument('--strict', action='store_true', help='Also check optional historical pins and documentary code drift')
    parser.add_argument('--verbose', action='store_true', help='Show traceback on failure')
    args = parser.parse_args(argv)
    try:
        graph03, release, phasec, evidence = resolve_graph03_dependencies(args, ROOT, phase='C')
        result = run_phasec(graph03, release, args.output_dir, args.config, ROOT,
            strict=args.strict, dependency_resolution=evidence)
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        if args.verbose:
            raise
        parser.exit(1, f'[Graph03] {exc}\n')
    print(f'[Graph03] Phase C: {result["phasec_status"]}')
    print(f'[Graph03] Resultado: {result["output_dir"]}')


if __name__ == '__main__':
    main()

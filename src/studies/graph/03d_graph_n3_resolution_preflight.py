#!/usr/bin/env python3
"""Resolve documentary blockers using explicit frozen inputs; never estimate models."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.graph.stage03.s03_n3_resolution import run_resolution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph03-run', type=Path, required=True)
    parser.add_argument('--graph02-release', type=Path, required=True)
    parser.add_argument('--phasec-run', type=Path, required=True,
                        help='Explicit Phase C directory, identified by its pinned QC')
    parser.add_argument('--output-dir', type=Path, required=True, help='New restricted run directory')
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/n3_phase_d_resolution.yaml')
    parser.add_argument('--preflight-only', action='store_true',
                        help='Document intent; estimation is prohibited with or without this flag')
    args = parser.parse_args(argv)
    result = run_resolution(args.graph03_run, args.graph02_release, args.phasec_run,
                            args.output_dir, args.config, ROOT)
    print(f'Phase D {result["status"]}; {result["phased_status"]}; run_id={result["run_id"]}')


if __name__ == '__main__':
    main()

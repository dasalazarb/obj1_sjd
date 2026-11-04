#!/usr/bin/env python3
"""Standalone frozen-input Phase C audit; never runs Graph02 or Phase B."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.studies.graph.stage03.s03_n3_phasec_adjudication import run_phasec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph03-run',type=Path,required=True,help='Explicit existing Phase A run directory')
    parser.add_argument('--graph02-release',type=Path,required=True,help='Explicit frozen Graph02 manifest or directory')
    parser.add_argument('--output-dir',type=Path,required=True,help='New separate restricted Phase C directory')
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/n3_phasec_review.yaml')
    args = parser.parse_args(argv)
    result = run_phasec(args.graph03_run,args.graph02_release,args.output_dir,args.config,ROOT)
    print(f'Phase C {result["status"]}; run_id={result["run_id"]}; outputs={result["output_dir"]}')


if __name__ == '__main__':
    main()

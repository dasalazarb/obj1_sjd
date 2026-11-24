#!/usr/bin/env python3
"""Audita las causas de no estimación. Diagnóstico sin autorización clínica."""
import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.graph.stage03.s03_n3_model_readiness_audit import AuditError, VERSION, run_audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('summaries', 'secure-recompute'),
                        help='Agregados publicados o recálculo institucional en memoria')
    parser.add_argument('--representation', default='R0', choices=('R0', 'R1', 'R2', 'R3', 'R4', 'all'))
    for option in ('graph02-release', 'graph03-run', 'graph03e-dir', 'output-dir'):
        parser.add_argument('--' + option, type=Path)
    parser.add_argument('--strict', action='store_true', help='Exige la vinculación registrada y rechaza divergencias')
    parser.add_argument('--version', action='version', version=VERSION)
    try:
        result = run_audit(ROOT, **vars(parser.parse_args(argv)))
    except AuditError as exc:
        parser.exit(1, f'[Graph03g] {exc}.\n')
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RuntimeError, yaml.YAMLError):
        parser.exit(1, '[Graph03g] AUDIT_INPUT_SCHEMA_OR_INTEGRITY_ERROR. '
                    'Verifique los resultados compatibles de Graph02, Graph03 y Graph03e.\n')
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    main()

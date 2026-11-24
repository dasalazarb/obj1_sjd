#!/usr/bin/env python3
"""Audita las causas de no estimación. Diagnóstico sin autorización clínica."""
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.graph._pipeline import config_path
from src.studies.graph.stage03.s03_n3_model_readiness_audit import AuditError, run_audit

DEFAULTS = {'mode': 'summaries', 'representation': 'R0', 'strict': True}
PATHS = ('graph02_release', 'graph03_run', 'graph03e_dir', 'output_dir')


def _settings():
    override = os.environ.get('GRAPH03G_CONFIG')
    if not override:
        return DEFAULTS.copy()
    path = Path(override).expanduser().resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, yaml.YAMLError):
        raise AuditError('AUDIT_CONFIG_UNREADABLE') from None
    if (not isinstance(config, dict) or config.keys() - {*DEFAULTS, *PATHS}
            or any(not isinstance(config[key], type(value)) for key, value in DEFAULTS.items() if key in config)
            or any(not isinstance(config[key], str) or not config[key].strip() for key in PATHS if key in config)):
        raise AuditError('AUDIT_INVALID_CONFIG')
    settings = DEFAULTS | config
    for key in PATHS:
        if key in config:
            settings[key] = config_path(path, config[key])
    return settings


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        sys.stderr.write('[Graph03g] Ejecute el script sin argumentos. '
                         'Use GRAPH03G_CONFIG para una configuración opcional.\n')
        raise SystemExit(2)
    try:
        result = run_audit(ROOT, **_settings())
    except AuditError as exc:
        sys.stderr.write(f'[Graph03g] {exc}.\n')
        raise SystemExit(1) from None
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RuntimeError, yaml.YAMLError):
        sys.stderr.write('[Graph03g] AUDIT_INPUT_SCHEMA_OR_INTEGRITY_ERROR. '
                         'Verifique los resultados compatibles de Graph02, Graph03 y Graph03e.\n')
        raise SystemExit(1) from None
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    main()

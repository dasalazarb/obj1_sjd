#!/usr/bin/env python3
"""Ejecuta la auditoría integral de Graph03g."""
import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        sys.stderr.write('[Graph03g] Ejecute el script sin argumentos.\n')
        raise SystemExit(2)
    previous_umask = os.umask(0o077)
    try:
        # Readers and numerical libraries can include paths or input values in warnings.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from src.studies.graph.stage03.s03_n3_model_readiness_audit import run_audit

            result = run_audit(ROOT)
    except Exception:
        sys.stderr.write('[Graph03g] AUDIT_FAILED. No se pudo iniciar o publicar la auditoría.\n')
        raise SystemExit(1) from None
    finally:
        os.umask(previous_umask)
    print(json.dumps(result, sort_keys=True))
    if result['status'] == 'AUDIT_FAILED':
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    main()

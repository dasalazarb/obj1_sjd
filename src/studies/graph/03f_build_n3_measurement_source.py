#!/usr/bin/env python3
"""Audita fuentes originales de Graph03 N3 y construye evidencia restringida."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.graph.stage03.s03_n3_measurement_source import run_source
from src.studies.graph.stage03.s03_n3_calendar_day import BTRIS_ROLES

PATHS = {
    'eda_long': 'Parquet longitudinal original EDA/BTRIS',
    'step01': 'Parquet Step01 de analitos seleccionados por episodio',
    'step10': 'Parquet analítico longitudinal de Step10',
    'context': 'Parquet longitudinal de contexto de Step10',
    'graph02_release': 'Ruta del manifest o directorio del release Graph02 elegido explícitamente',
    'graph03_run': 'Directorio del run congelado de Graph03 Fase A',
    'baseline': 'CSV original baseline_raw del release Graph02, o copia con SHA-256 idéntico',
    'spine': 'Spine original opcional, inventario de esquema',
    'essdai_original': 'Fuente original opcional de ESSDAI, inventario de esquema',
    'esspri_original': 'Fuente original opcional de ESSPRI, inventario de esquema',
    'biopsy_original': 'Fuente original opcional de biopsia, inventario de esquema',
    'protocol_original': 'Fuente original opcional de asignaciones de protocolo, inventario de esquema',
    **{role: f'Contrato v2: archivo CSV o carpeta con Lab*.csv de {role}' for role in BTRIS_ROLES},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--audit-only', dest='mode', action='store_const', const='audit-only',
                      help='Registra esquemas, agregados suprimidos y bloqueos')
    mode.add_argument('--build-candidate', dest='mode', action='store_const', const='build-candidate',
                      help='Guarda observaciones NO APROBADAS y ledger dentro de un run privado')
    mode.add_argument('--build-reviewed', dest='mode', action='store_const', const='build-reviewed',
                      help='Verifica una revisión externa firmada antes de construir la fuente instrumentada')
    parser.add_argument('--config', type=Path, default=ROOT / 'src/studies/graph/configs/n3_measurement_source.yaml')
    for name, help_text in PATHS.items():
        parser.add_argument('--' + name.replace('_', '-'), type=Path, help=help_text)
    parser.add_argument('--output-root', type=Path, required=True, help='Directorio privado fuera del repositorio')
    parser.add_argument('--run-id', help='Identificador nuevo dentro de runs/; por defecto se genera uno')
    parser.add_argument('--review', type=Path, help='Expediente YAML externo en modo revisado')
    parser.add_argument('--signature', type=Path, help='Firma separada del expediente, SHA-256')
    parser.add_argument('--trust-key', type=Path, help='Clave pública institucional provista por un canal autorizado')
    args = parser.parse_args(argv)

    try:
        paths = {name: getattr(args, name) for name in PATHS if getattr(args, name) is not None}
        result = run_source(args.mode, paths, args.output_root, args.config, ROOT, run_id=args.run_id,
                            review_path=args.review, signature_path=args.signature, trust_key=args.trust_key)
    except (AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeError,
            subprocess.SubprocessError, yaml.YAMLError) as exc:
        message = str(exc)
        token = message if re.fullmatch('[A-Z][A-Z0-9_]+', message) else 'INVALID_SOURCE_OR_REVIEW_CONTRACT'
        parser.exit(1, f'[Graph03] Fuente N3 detenida. Código {token}.\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

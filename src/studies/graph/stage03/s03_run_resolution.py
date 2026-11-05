"""Graph03's single run selector: recorded time, contracts and linked ancestry.

Only configured study collections are searched. Filesystem mtime breaks ties;
it never establishes a run's creation time or scientific eligibility.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

import yaml

from .._pipeline import config_path, digest
from ..stage02.s02_contracts import validate_release
from .s03_n3_phasec_adjudication import INPUT_FILES, load_phasec_inputs


def graph_paths(repository_root):
    repo = Path(repository_root).resolve()
    data_root = Path(os.environ.get('OBJ1_SJD_ROOT', repo)).resolve()
    config = repo / 'src/studies/graph/configs/pipeline.yaml'
    paths = yaml.safe_load(config.read_text())['paths']
    result = {}
    for key, stage in [('graph02', '02'), ('graph03', '03')]:
        configured = config_path(config, paths['output_' + stage])
        result[key] = data_root / configured.relative_to(repo) if configured.is_relative_to(repo) else configured
    result['phasec'] = result['graph03'].parent / '03_graph_n3_clinical_adjudication'
    result['phased'] = result['graph03'].parent / '03_graph_n3_resolution_preflight'
    return result


def resolve_latest_compatible(kind, roots, required_files, metadata_file, validate, *,
                              override=None, expected_run=None):
    """Return (directory, validated payload, selection evidence), newest valid first.

    Explicit overrides receive exactly the same validation and never fall back.
    Historical nonstandard IDs need a recorded ISO creation time for discovery;
    they remain usable via an explicit override. Rejections are bounded in logs.
    """
    candidates = [Path(override).resolve()] if override else sorted({
        p.resolve() for root in roots for p in (Path(root) / 'runs').glob('*') if p.is_dir()})
    failures, ordered = [], []
    for path in candidates:
        try:
            for name in required_files:
                if not (path / name).is_file():
                    raise ValueError('archivo requerido ausente: ' + name)
            metadata = json.loads((path / metadata_file).read_text())
            if not isinstance(metadata, dict):
                raise ValueError('metadatos del run deben ser un objeto JSON')
            run_id = metadata.get('run_id', metadata.get('graph03_run_id', path.name))
            if expected_run is not None and run_id != expected_run:
                if override:
                    raise ValueError(f'run {run_id} incompatible; se esperaba {expected_run}')
                continue
            stamp = metadata.get('created_at_utc')
            match = re.fullmatch(r'(\d{8}T\d{6}\d{6}Z)_[a-f0-9]{8}', str(run_id))
            if match:
                created = datetime.strptime(match[1], '%Y%m%dT%H%M%S%fZ').replace(tzinfo=timezone.utc)
                criterion = 'timestamp del run ID'
            elif stamp:
                if not isinstance(stamp, str):
                    raise ValueError('created_at_utc debe ser una fecha ISO con zona horaria')
                created = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
                if created.tzinfo is None:
                    raise ValueError('fecha de creación sin zona horaria')
                criterion = 'created_at_utc'
            elif override:
                created, criterion = datetime.min.replace(tzinfo=timezone.utc), 'override explícito'
            else:
                raise ValueError('sin fecha de creación verificable; use un override explícito')
            ordered.append((created, path.stat().st_mtime_ns, str(path), path, run_id, criterion))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f'{path.name}: {str(exc)[:600]}')
    for _, _, _, path, run_id, criterion in sorted(ordered, reverse=True):
        try:
            payload = validate(path)
            reason = ('override validado' if override else 'más reciente compatible por ' + criterion)
            evidence = dict(dependency=kind, path=str(path), run_id=run_id, reason=reason,
                            rejected_candidates=failures[:5])
            return path, payload, evidence
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            failures.append(f'{path.name}: {str(exc)[:600]}')
    scripts = {'Graph02': '02_graph_representation_sensitivity.py',
               'Graph03 Phase A': '03_graph_community_characterization.py',
               'Graph03 Phase C': '03c_graph_n3_clinical_adjudication.py'}
    searched = str(override) if override else ', '.join(str(Path(p) / 'runs') for p in roots)
    raise RuntimeError(f'No se encontró un run compatible de {kind}. '
                       f'Ejecute primero src/studies/graph/{scripts[kind]}. '
                       f'Buscado en {searched}. Descartes: ' + '; '.join(failures[:5]))


def resolve_graph03_dependencies(args, repository_root, *, phase):
    """Resolve Phase C's pair or Phase D's linked triple as a unit."""
    paths = graph_paths(repository_root)
    config = yaml.safe_load(args.config.read_text())
    from .s03_n3_resolution import load_inputs, validate_config
    from .s03_n3_phasec_adjudication import validate_phasec_config
    (validate_config if phase == 'D' else validate_phasec_config)(config)
    strict = args.strict

    def select_a(expected=None):
        def validate_a(graph03):
            qc = json.loads((graph03 / '03_graph_n3_validation_qc.json').read_text())

            def validate_02(directory):
                manifest, release = validate_release(directory)
                if (manifest['run_id'] != qc['graph02_run_id'] or
                        digest(release) != qc['graph02_release_sha256']):
                    raise ValueError('Graph02 no pertenece a esta Phase A')
                return release

            override = args.graph02_release
            if override and override.is_file():
                if override.name != '02_graph_release_manifest.json':
                    raise ValueError('El override Graph02 debe ser 02_graph_release_manifest.json o su directorio')
                override = override.parent
            _, release, evidence02 = resolve_latest_compatible('Graph02', [paths['graph02']],
                ['02_graph_release_manifest.json'], '02_graph_release_manifest.json', validate_02,
                override=override, expected_run=qc['graph02_run_id'])
            cconfig = config if phase == 'C' else yaml.safe_load((Path(repository_root) /
                'src/studies/graph/configs/n3_phasec_review.yaml').read_text())
            load_phasec_inputs(graph03, release, cconfig, strict=strict)
            return release, evidence02

        graph03, (release, evidence02), evidence03 = resolve_latest_compatible(
            'Graph03 Phase A', [paths['graph03']], INPUT_FILES, '03_graph_n3_validation_qc.json',
            validate_a, override=args.graph03_run, expected_run=expected)
        return graph03, release, [evidence02, evidence03]

    if phase == 'D':
        def validate_c(phasec):
            qc = json.loads((phasec / '03_graph_n3_phasec_qc.json').read_text())
            graph03, release, evidence = select_a(qc['input_graph03_run_id'])
            load_inputs(graph03, release, phasec, config, strict=strict)
            return graph03, release, evidence

        phasec, (graph03, release, evidence), evidencec = resolve_latest_compatible(
            'Graph03 Phase C', [paths['phasec']],
            ['03_graph_n3_phasec_qc.json', '03_graph_n3_phasec_artifact_inventory.csv'],
            '03_graph_n3_phasec_qc.json', validate_c, override=args.phasec_run)
        evidence.append(evidencec)
    else:
        graph03, release, evidence = select_a()
        phasec = None
    for record in evidence:
        print(f'[Graph03] {record["dependency"]}: {record["path"]}; '
              f'run_id={record["run_id"]}; {record["reason"]}')
        for rejected in record['rejected_candidates']:
            print(f'[Graph03] Descartado: {rejected}')
    print(f'[Graph03] Configuración: {args.config.resolve()}')
    print('[Graph03] Resolución de dependencias: OK')
    return graph03, release, phasec, evidence

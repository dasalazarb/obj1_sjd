"""Configuration, immutable run directories, and artifact utilities (no modeling)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from uuid import uuid4
import numpy as np
import pandas as pd
import yaml
import common

GRAPH = Path(__file__).resolve().parent
DEFAULT_CONFIG = GRAPH / 'configs/pipeline.yaml'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_yaml(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f'Configuration must be a mapping: {path}')
    return value


def config_path(config, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(config).parent / path).resolve()


def load_pipeline(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    cfg = read_yaml(path)
    if cfg.get('schema_version') != 'graph_pipeline_v1':
        raise ValueError('Unsupported pipeline schema')
    cfg['_path'] = str(path)
    hashes = {'pipeline': digest(path)}
    for name, value in cfg['includes'].items():
        p = config_path(path, value)
        cfg[name] = read_yaml(p)
        hashes[name] = digest(p)
    cfg['_hashes'] = hashes
    cfg['_config_sha256'] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if cfg['analysis']['mode'] not in {'exploratory', 'clinical_release'}:
        raise ValueError('Unknown analysis mode')
    if cfg['analysis']['feature_policy'] not in {'phenotype_independent', 'clinical_inclusive'}:
        raise ValueError('Unknown feature policy')
    if cfg['analysis']['reference_scenario'] != 'S3':
        raise ValueError('Protocol fixes S3/R0 as primary; amend the protocol before changing it')
    return cfg


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def write_json(path, value, *, atomic=False):
    path = Path(path)
    payload = json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + '\n'
    if atomic:
        temp = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
        try:
            with temp.open('x') as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    else:
        path.write_text(payload)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def new_run(output_root, run_id=None):
    run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run ID')
    path = Path(output_root) / 'runs' / run_id
    path.mkdir(parents=True, exist_ok=False)
    (path / 'figures').mkdir()
    (path / 'logs').mkdir()
    return path


def output_root(cfg, stage):
    key = f'output_{stage}'
    return config_path(cfg['_path'], cfg['paths'][key])


def version_info():
    import importlib.metadata
    result = {'python': platform.python_version()}
    for package in ('numpy','pandas','scipy','scikit-learn','networkx','kmapper','statsmodels','matplotlib','PyYAML'):
        result[package] = importlib.metadata.version(package)
    return result


def git_commit():
    return subprocess.check_output(['git','rev-parse','HEAD'], cwd=common.PROJECT_ROOT, text=True).strip()


def save_table(path, rows, columns=None):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty and columns:
        frame = pd.DataFrame(columns=columns)
    if not len(frame.columns):
        frame = pd.DataFrame(columns=['status','reason'])
    frame.to_csv(path, index=False)
    return frame


def artifact_inventory(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and p.name not in {'02_graph_release_manifest.json', '03_graph_results_summary.json'}}


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--pipeline-config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--run-id')
    p.add_argument('--output-dir', type=Path, help='Run collection root; outputs use runs/<run-id>')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Preflight only; never publish an incomplete release')
    return p

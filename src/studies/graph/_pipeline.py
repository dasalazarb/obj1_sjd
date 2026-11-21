"""Configuration, current study outputs, and artifact utilities (no modeling)."""
from __future__ import annotations
import argparse
from contextlib import contextmanager, ExitStack
import fcntl
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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


def output_root(cfg, stage):
    key = f'output_{stage}'
    return config_path(cfg['_path'], cfg['paths'][key])


def stage_destinations(root, *, include_figures=True):
    """Mirror the standard table directory for figures; keep custom roots flat."""
    destinations = {'stage': root}
    if include_figures:
        for parent in root.parents:
            if parent.name == 'tables' and parent.parent.name == 'outputs':
                figures = (parent.with_name('figures') / root.relative_to(parent)).resolve()
                if figures.is_relative_to(root) or root.is_relative_to(figures):
                    raise ValueError('Graph figure destination overlaps the table destination')
                destinations['figures'] = figures
                break
    return destinations


def staged_artifact_paths(directory, figure_dir, figure_prefix):
    paths = [path.name for path in sorted(directory.iterdir()) if path.is_file()]
    if figure_dir != directory:
        paths.extend((figure_prefix / path.name).as_posix()
                     for path in sorted(figure_dir.iterdir()) if path.is_file())
    return paths


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


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--pipeline-config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--output-dir', type=Path, help='Override the current stage directory')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Preflight only; never publish an incomplete release')
    return p


@contextmanager
def output_lock(directory, *, shared=False):
    """Coordinate readers and writers of the canonical directory."""
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


@contextmanager
def stage_outputs(destinations, owned, *, failure, completion=None, diagnostic=False, obsolete_dirs=()):
    """Stage files on the destination filesystem and roll back failed publication.

    Each destination maps to a staging directory. Only direct, stage-owned files
    participate in replacement or cleanup. Diagnostics leave other files alone.
    """
    with ExitStack() as stack:
        staging = {}
        for key, destination in sorted(destinations.items(), key=lambda item: str(item[1])):
            destination.mkdir(parents=True, exist_ok=True, mode=0o700)
            stack.enter_context(output_lock(destination))
            temp = Path(stack.enter_context(TemporaryDirectory(prefix='.graph-', dir=destination.parent)))
            staging[key] = temp / 'files'
            staging[key].mkdir(mode=0o700)
        failed = destinations[failure[0]] / failure[1]
        try:
            yield staging
            _publish_outputs(destinations, staging, owned, failed, completion, diagnostic, obsolete_dirs)
        except Exception as exc:
            status = {'execution_status': 'FAILED', 'exception_type': type(exc).__name__,
                      'failed_at_utc': utc_now(), 'preflight_only': diagnostic}
            code = getattr(exc, 'code', str(exc))
            if isinstance(code, str) and re.fullmatch(r'[A-Z][A-Z0-9_]{2,100}', code):
                status['reason'] = code
            write_json(failed, status, atomic=True)
            raise


def _publish_outputs(destinations, staging, owned, failed, completion, diagnostic, obsolete_dirs):
    changes = {}
    backup_roots = {}
    for key, destination in destinations.items():
        staged = {path.name: path for path in staging[key].iterdir() if path.is_file()}
        if any(not owned(key, name) for name in staged):
            raise ValueError('Staged file does not belong to this Graph stage')
        names = set(staged)
        if not diagnostic:
            names.update(path.name for path in destination.iterdir() if path.is_file() and owned(key, path.name))
            for name in obsolete_dirs:
                directory = destination / name
                if directory.is_dir() and not directory.is_symlink():
                    changes.update((path, None) for path in directory.iterdir()
                                   if path.is_file() and owned(key, path.name))
                    backup_roots[directory] = staging[key].parent
        for name in sorted(names):
            changes[destination / name] = staged.get(name)
        backup_roots[destination] = staging[key].parent
    if not diagnostic and completion and changes.get(completion) is None:
        raise ValueError('Graph stage did not produce its completion artifact')
    if not diagnostic and failed.exists():
        changes[failed] = None
    if completion in changes:
        changes[completion] = changes.pop(completion)
    backups = {}
    for index, target in enumerate(changes):
        if target.is_file() or target.is_symlink():
            backup = backup_roots[target.parent] / f'backup-{index}'
            os.link(target, backup, follow_symlinks=False)
            backups[target] = backup
    applied = []
    try:
        for target, source in changes.items():
            if source is None:
                target.unlink(missing_ok=True)
            else:
                source.chmod(0o600)
                os.replace(source, target)
            applied.append(target)
    except BaseException:
        for target in reversed(applied):
            if target in backups:
                os.replace(backups[target], target)
            else:
                target.unlink(missing_ok=True)
        raise

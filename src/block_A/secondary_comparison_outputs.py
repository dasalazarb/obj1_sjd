"""Aggregate-only staging, immutable fingerprints, transactional install/rollback."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from src.block_A.secondary_trajectory_analysis import ContractError, sha256
from src.block_A.secondary_comparison_config import STEM

FORBIDDEN = {'patient_id', 'clinical_episode_id', 'clinical_baseline_episode_id',
             'reference_date', 'variable_measurement_date', 'clinical_baseline_date', 'essdai_index_date'}


def directories(root, axis, threshold, analysis):
    # A feasibility-only run cannot masquerade as a complete inferential bundle.
    suffix = Path(axis)/f'delta{threshold}'/('feasibility' if analysis == 'feasibility' else analysis)
    return {area: Path(root)/'outputs'/area/'blockA'/STEM/suffix for area in ('tables', 'figures', 'qc')}


def preflight(root, destinations, inputs, overwrite=False, backup_root=None):
    root = Path(root).resolve()
    for dest in destinations.values():
        if not dest.resolve().is_relative_to(root/'outputs') or STEM not in dest.parts:
            raise ContractError('unsafe Step 16 destination')
        # No symlink redirection inside publication trees, including nonexistent leaves.
        p = dest
        while p != root:
            if p.is_symlink():
                raise ContractError('symlink in output destination')
            p = p.parent
        if dest.exists() and not overwrite:
            raise FileExistsError('Step 16 destination exists: '+str(dest))
        for item in inputs.values():
            path = Path(item['path']).resolve()
            if path.is_relative_to(dest.resolve()) or dest.resolve().is_relative_to(path):
                raise ContractError('output overlaps protected input')
    if overwrite and not backup_root:
        raise ContractError('--overwrite requires --backup-root')
    if backup_root:
        backup_root = Path(backup_root).resolve()
        if not backup_root.is_relative_to(root/'outputs/backups'/STEM):
            raise ContractError('backup root must be below outputs/backups/'+STEM)
        if any(backup_root.is_relative_to(d.resolve()) or d.resolve().is_relative_to(backup_root) for d in destinations.values()):
            raise ContractError('backup overlaps output')
        if any(Path(item['path']).resolve().is_relative_to(backup_root) for item in inputs.values()):
            raise ContractError('backup overlaps input')


def safe_table(data, patient_ids=()):
    if FORBIDDEN & set(data):
        raise ContractError('identifying columns in aggregate output')
    # Exact cell identity check; no identifying free-text warnings are exported.
    ids = set(map(str, patient_ids))
    for column in data.select_dtypes(include=['object', 'string']):
        if data[column].dropna().astype(str).isin(ids).any():
            raise ContractError('patient identity in aggregate output')
    return data


def write_table(data, path, patient_ids=()):
    safe_table(data, patient_ids)
    # Empty analyses still have readable schema, not an empty CSV file.
    if not len(data.columns):
        data = pd.DataFrame(columns=['variable', 'status', 'reason'])
    data.to_csv(path, index=False)


def artifact_manifest(stage):
    stage = Path(stage)
    return [dict(path=str(p.relative_to(stage)), sha256=sha256(p))
            for p in sorted(stage.rglob('*')) if p.is_file() and p.name != 'run_manifest.json']


def install_bundle(staging, destinations, backup_root=None):
    """Same-filesystem renames; QC last; reverse all installed dirs on failure."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = Path(backup_root)/stamp if backup_root else None
    saved, installed = [], []
    try:
        for name, dest in destinations.items():
            if dest.exists():
                if backup is None:
                    raise FileExistsError('destination appeared during staging')
                copy = backup/name
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(dest, copy)
                saved.append((dest, copy))
                shutil.rmtree(dest)
        for name in sorted(staging, key=lambda n: n.startswith('qc')):
            dest = destinations[name]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise FileExistsError('destination appeared during install')
            os.replace(staging[name], dest)
            installed.append(dest)
    except BaseException:
        for dest in reversed(installed):
            shutil.rmtree(dest)
        for dest, copy in saved:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(copy, dest)
        raise


def write_manifest(path, content):
    path.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False, default=str)+'\n')

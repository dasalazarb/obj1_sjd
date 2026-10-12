"""Read-only historical references and atomic, create-only review bundles.

The seal lives outside the reference. It inventories every file, including code,
figures and environment records, rather than replacing the historical manifest.
It establishes byte integrity, not scientific reproducibility of a dirty run.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value or "") or value in {".", ".."}:
        raise ValueError("Run IDs must be simple names, without paths")
    return value


def inventory(root):
    root = Path(root).resolve(strict=True)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Reference bundles must not contain symlinks")
    return {str(p.relative_to(root)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}


def create_json(path, value):
    """Atomic file publication without an overwrite race (link fails if present)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def seal_reference(root, seal_path, run_id):
    """Audit an existing archive in place; never write inside or duplicate it."""
    root = Path(root).resolve(strict=True)
    seal_path = Path(seal_path).resolve()
    if seal_path.is_relative_to(root):
        raise ValueError("The reference seal must live outside historical artifacts")
    files = inventory(root)
    if not files:
        raise ValueError("Cannot freeze an empty reference")
    summaries = [p for p in root.rglob("02_longitudinal_mapper_summary*.json")]
    if len(summaries) != 1:
        raise ValueError("Reference requires exactly one historical Mapper summary")
    summary = json.loads(summaries[0].read_text())
    manifest = summary.get("run_manifest", {})
    if summary.get("scenario", "S3-primary") != "S3-primary":
        raise ValueError("Only historical S3-primary can be the reference")
    value = {"reference_run_id": safe_id(run_id), "reference_root": str(root),
             "reference_status": "historical_frozen", "files": files,
             "reference_config_sha256": summary.get("config_sha256", manifest.get("config_sha256")),
             "reference_script_sha256": manifest.get("script_sha256"),
             "reference_input_sha256": summary.get("input_sha256", manifest.get("input_sha256", {})),
             "historical_git_dirty": manifest.get("git_dirty"),
             "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
             "reproducibility_status": "requires_code_environment_and_input_provenance_review"}
    create_json(seal_path, value)
    return value


def verify_reference(seal_path):
    value = json.loads(Path(seal_path).read_text())
    if value.get("reference_status") != "historical_frozen":
        raise ValueError("Reference is not frozen")
    current = inventory(value["reference_root"])
    if current != value["files"]:
        changed = sorted(k for k in current.keys() | value["files"].keys() if current.get(k) != value["files"].get(k))
        raise ValueError(f"Historical reference integrity failed: {changed}")
    return {**value, "reference_manifest_sha256": sha256(seal_path), "integrity_status": "passed"}


def protect_path(path, reference_roots=()):
    path = Path(path).resolve()
    if any(part in {"history", "longitudinal_graph_history", "references"} for part in path.parts):
        raise ValueError("Writing historical/reference paths is prohibited")
    for root in reference_roots:
        root = Path(root).resolve()
        if path.is_relative_to(root) or root.is_relative_to(path):
            raise ValueError("Output path overlaps the frozen reference")
    return path


@contextmanager
def new_bundle(path, reference_roots=(), force=False):
    """Publish a completed directory by rename, under an exclusive writer lock.

    Force is intentionally limited to a directory explicitly named development;
    it never permits touching a historical reference. Production runs are create-only.
    """
    path = protect_path(path, reference_roots)
    if force and "development" not in path.parts:
        raise ValueError("--force is permitted only under an explicit development namespace")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f".{path.name}.publishing")
    lock.mkdir()  # Exclusive cooperative lock; concurrent writers fail before work.
    staged = None
    try:
        if path.exists():
            if not force:
                raise FileExistsError(f"Run already exists: {path}")
            if path.is_symlink():
                raise ValueError("Cannot force a symlink output")
        staged = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
        yield staged
        if path.exists():
            if not force:
                raise FileExistsError(f"Run appeared during publication: {path}")
            shutil.rmtree(path)
        staged.rename(path)
        staged = None
    finally:
        if staged is not None:
            shutil.rmtree(staged)
        lock.rmdir()


def reference_artifact(reference, stem, suffix):
    """Resolve exact historical names and researcher export suffixes without guessing."""
    root = Path(reference["reference_root"])
    matches = [root / name for name in reference["files"]
               if Path(name).name.startswith(stem) and Path(name).suffix == suffix]
    exact = [p for p in matches if p.name == stem + suffix]
    if exact:
        return exact[0]
    if len(matches) != 1:
        raise ValueError(f"Reference artifact absent or ambiguous: {stem}{suffix}")
    return matches[0]

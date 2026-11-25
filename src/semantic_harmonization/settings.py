"""Repository-relative configuration and content-addressed provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

MODULE = Path(__file__).resolve().parent
ROOT = MODULE.parents[1]
KEYS = ["patient_id", "clinical_episode_id"]


def digest(value: Any) -> str:
    """Hash canonical UTF-8 JSON."""
    return hashlib.sha256(serialize(value).encode()).hexdigest()


def serialize(value: Any) -> str:
    """Return compact, deterministic JSON without non-standard numbers."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_hash(path: Path) -> str:
    """Hash a file without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_yaml(path: Path) -> dict:
    """Read a YAML mapping."""
    settings = yaml.safe_load(path.read_text())
    if not isinstance(settings, dict):
        raise ValueError(f"Expected a YAML mapping at {path}")
    return settings


def write_json(path: Path, value: Any) -> None:
    """Atomically write a private JSON file."""
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_text(serialize(value) + "\n")
    os.chmod(temp, 0o600)
    temp.replace(path)


def merge(base: dict, overrides: dict) -> dict:
    """Merge nested local overrides into the versioned defaults."""
    for key, value in overrides.items():
        base[key] = merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return base


def pipeline_config(root: Path = ROOT) -> dict:
    """Load repository-owned YAML and resolve paths independently of the caller."""
    root = root.resolve()
    dirs = root / "src" / "semantic_harmonization" / "configs"
    cfg = read_yaml(dirs / "pipeline.yaml")
    for name in ("concepts", "normalization", "resources", "prompts"):
        cfg[name] = read_yaml(dirs / f"{name}.yaml")
    local = dirs / "local.yaml"
    if local.is_file():
        merge(cfg, read_yaml(local))
    paths = cfg["paths"]
    obj1 = (root / paths["obj1_root"]).resolve() if paths["obj1_root"] else root
    if not (obj1 / "src" / "block_A" / "10_build_integrated_longitudinal_dataset.py").is_file():
        raise ValueError(f"Configured obj1_sjd root lacks its Step 10 producer: {obj1}")
    paths["obj1_root"] = str(obj1)
    for name in ("integrated", "context", "analytic", "output", "pharma_registry", "source_events",
                 "gpt_oss", "embedding_model", "knowledge_dir", "audit", "gold", "codebook"):
        if paths.get(name):
            paths[name] = str((obj1 / paths[name]).resolve())
    for source in cfg["resources"]["optional"].values():
        if source["path"]:
            source["path"] = str((obj1 / source["path"]).resolve())
    if Path(paths["analytic"]) == Path(paths["output"]):
        raise ValueError("paths.analytic and paths.output must be distinct")
    return cfg


def load_config() -> dict:
    """Resolve the local codebook for resource preparation or harmonization."""
    cfg = pipeline_config()
    paths = cfg["paths"]
    if not paths["codebook"]:
        root = Path(paths["obj1_root"])
        name = "Consolidated_Codebook_all_columns.xlsx"
        candidates = sorted({p.resolve() for p in root.rglob(name) if ".git" not in p.parts})
        if len(candidates) != 1:
            raise ValueError(f"Expected one {name} in obj1_sjd, found {len(candidates)}. Set paths.codebook in configs/local.yaml")
        paths["codebook"] = str(candidates[0])
    return cfg

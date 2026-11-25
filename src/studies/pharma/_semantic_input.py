"""Isolate dependent Pharma stages and check their shared source provenance."""
import argparse
import json
from pathlib import Path

from src.semantic_harmonization.study_input import StudyInput, active_study_input
from src.semantic_harmonization.settings import file_hash, write_json

PROVENANCE = "study_input_provenance.json"


def versioned(path: Path, source: StudyInput) -> Path:
    """Address the same file in the run-specific study output directory."""
    return path.parent / "semantic_harmonization" / source.run_id / path.name


def follow_up(args: argparse.Namespace, defaults: dict[str, Path]) -> tuple[argparse.Namespace, StudyInput | None]:
    """Read only intermediates produced from the exact same candidate."""
    source = active_study_input()
    if source is None:
        return args, None
    values = vars(args).copy()
    for name, default in defaults.items():
        requested = Path(values[name]).resolve()
        expected = versioned(default, source).resolve()
        if requested not in {default.resolve(), expected}:
            raise ValueError(f"Pharma {name} override conflicts with the selected semantic run")
        manifest = expected.parent / PROVENANCE
        if not manifest.is_file() or not expected.is_file():
            raise FileNotFoundError(f"Run the required preceding Pharma stage for the selected candidate: {name}")
        evidence = json.loads(manifest.read_text())
        if evidence.get("semantic_source") != source.provenance() or evidence.get("sha256", {}).get(expected.name) != file_hash(expected):
            raise ValueError(f"Pharma {name} provenance or hash differs from the selected semantic run")
        values[name] = expected
    return argparse.Namespace(**values), source


def study_name(name: str, source: StudyInput | None) -> str:
    """Use the existing output group with an isolated candidate suffix."""
    return f"{name}/semantic_harmonization/{source.run_id}" if source else name


def publish_provenance(source: StudyInput | None, directory: Path, names: tuple[str, ...]) -> None:
    """Bind outputs for dependent stages to their verified candidate."""
    if source is not None:
        write_json(directory / PROVENANCE, {"semantic_source": source.provenance(),
                                             "sha256": {name: file_hash(directory / name) for name in names}})

"""Resolve the same verified Step 10 candidate and sidecars for each study."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from .settings import MODULE, ROOT, digest, file_hash, merge, pipeline_config, read_yaml, write_json

SOURCE_MANIFEST = "candidate_source_manifest.json"
RUN_MANIFEST = "run_manifest.json"
ANALYTIC_FILES = {
    "integrated": "10_integrated_longitudinal_clinical_episode_harmonized.parquet",
    "context": "10_integrated_longitudinal_context_harmonized.parquet",
    "feature_manifest": "feature_manifest.parquet",
}
PROFILE_DIR = Path("study_inputs/pharma")
PROFILE_FILES = {
    "pharma_profile": "00_pharma_lab_profile.csv",
    "pharma_candidates": "00_pharma_graph_lab_candidates.csv",
    "pharma_selection": "00_pharma_graph_lab_selection_audit.csv",
    "pharma_provenance": "00_pharma_profile_summary.json",
    "pharma_expected": "00_pharma_expected_lab_audit.csv",
    "pharma_impact": "00_pharma_downstream_impact.csv",
}
HEX = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True)
class StudyInput:
    """A single version with stable paths for the candidate and its sidecars."""

    run_id: str
    manifest: Path
    files: dict[str, Path]
    sha256: dict[str, str]

    def select(self, role: str, requested: Path | None = None, *, default: Path | None = None) -> Path | None:
        """Return the verified role, rejecting an override from another run."""
        selected = self.files.get(role)
        if requested is not None and (default is None or Path(requested).resolve() != Path(default).resolve()):
            if selected is None or Path(requested).resolve() != selected:
                raise ValueError(f"{role} override conflicts with the shared semantic candidate")
        return selected

    def require(self, role: str) -> Path:
        """Return a required verified asset or fail with its missing role."""
        if role not in self.files:
            raise ValueError(f"Semantic candidate has no verified {role} asset")
        return self.files[role]

    def provenance(self) -> dict:
        """Supply the candidate version and Step 10 hash for study provenance."""
        return {"run_id": self.run_id, "manifest": str(self.manifest),
                "integrated_sha256": self.sha256["integrated"]}


def selection_id(dirs: Path = MODULE / "configs") -> str | None:
    """Read the shared opt-in selection, leaving existing defaults intact."""
    selection = read_yaml(dirs / "input_selection.yaml")
    local = dirs / "input_selection.local.yaml"
    if local.is_file():
        merge(selection, read_yaml(local))
    run_id = selection.get("run_id")
    if selection.get("mode") == "original" and run_id is None:
        return None
    if selection.get("mode") != "candidate" or not isinstance(run_id, str) or not HEX.fullmatch(run_id):
        raise ValueError("Select original, or set candidate mode with one full run_id")
    return run_id


def write_source_manifest(out: Path, analytic: Path, run_id: str, original: dict) -> None:
    """Describe the validated files relative to their matched run directories."""
    assets = {}
    for group, root, names in (
        ("analytic", analytic, ANALYTIC_FILES),
        ("report", out, {role: (PROFILE_DIR / name).as_posix() for role, name in PROFILE_FILES.items()}),
    ):
        for role, relative in names.items():
            path = root / relative
            if path.is_file():
                assets[role] = {"group": group, "path": relative, "sha256": file_hash(path)}
    if not {"integrated", "feature_manifest"} <= assets.keys():
        raise ValueError("Validated candidate is missing a core asset")
    if ("context" in original) != ("context" in assets):
        raise ValueError("Candidate context disagrees with Step 10 context availability")
    if set(assets) & PROFILE_FILES.keys() and not PROFILE_FILES.keys() <= assets.keys():
        raise ValueError("Candidate has an incomplete Pharma/Graph input bundle")
    write_json(out / SOURCE_MANIFEST, {"schema": "sjd_semantic_source_v1", "run_id": run_id,
                                     "original_step10_sha256": original["integrated"], "assets": assets})


def verify_source(run_id: str, *, analytic_root: Path, report_root: Path) -> StudyInput:
    """Authenticate a single completed candidate and all registered assets."""
    if not isinstance(run_id, str) or not HEX.fullmatch(run_id):
        raise ValueError("Invalid semantic candidate run_id")
    roots = {}
    for group, parent in (("analytic", analytic_root), ("report", report_root)):
        root = parent.resolve() / run_id
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"Missing or indirect semantic run directory: {group}")
        roots[group] = root
    report = roots["report"]
    manifest_path = report / SOURCE_MANIFEST
    for path in (manifest_path, report / RUN_MANIFEST):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Semantic candidate publication is incomplete")
    run = json.loads((report / RUN_MANIFEST).read_text())
    publication = run.get("outputs", {})
    identity = run.get("identity")
    if (not isinstance(identity, dict) or run.get("run_id") != run_id or digest(identity) != run_id
            or run.get("inputs") != identity.get("inputs")
            or publication.get("report", {}).get(SOURCE_MANIFEST) != file_hash(manifest_path)):
        raise ValueError("Semantic source manifest does not belong to the selected completed run")
    manifest = json.loads(manifest_path.read_text())
    original = run.get("inputs", {})
    if (manifest.get("schema") != "sjd_semantic_source_v1" or manifest.get("run_id") != run_id
            or manifest.get("original_step10_sha256") != original.get("integrated")):
        raise ValueError("Semantic source manifest identity mismatch")
    exact_assets = {
        **{role: ("analytic", name) for role, name in ANALYTIC_FILES.items()},
        **{role: ("report", (PROFILE_DIR / name).as_posix()) for role, name in PROFILE_FILES.items()},
    }
    assets = manifest.get("assets", {})
    if not {"integrated", "feature_manifest"} <= assets.keys() or ("context" in original) != ("context" in assets):
        raise ValueError("Semantic source manifest lacks required assets")
    if set(assets) & PROFILE_FILES.keys() and not PROFILE_FILES.keys() <= assets.keys():
        raise ValueError("Semantic source manifest has an incomplete Pharma/Graph bundle")
    files, hashes = {}, {}
    for role, asset in assets.items():
        group, relative = asset["group"], asset["path"]
        if (group, relative) != exact_assets.get(role):
            raise ValueError(f"Unexpected semantic candidate asset: {role}")
        path = roots[group] / relative
        if (path.resolve() != path or not path.is_file()
                or publication.get(group, {}).get(relative) != asset.get("sha256")
                or file_hash(path) != asset["sha256"]):
            raise ValueError(f"Semantic candidate asset hash or path mismatch: {role}")
        files[role], hashes[role] = path, asset["sha256"]
    return StudyInput(run_id, manifest_path, files, hashes)


def active_study_input() -> StudyInput | None:
    """Resolve the opt-in candidate only when configured."""
    run_id = selection_id()
    if run_id is None:
        return None
    paths = pipeline_config(ROOT)["paths"]
    return verify_source(run_id, analytic_root=Path(paths["analytic"]), report_root=Path(paths["output"]))

from dataclasses import dataclass
from pathlib import Path
import re
import yaml
from shared.utils.paths import APP_ROOT, contained, outputs_root, repo_root


class OutputError(ValueError):
    pass


class MissingOutput(OutputError):
    pass


class SchemaError(OutputError):
    pass


PROVENANCE_PATH = "data/analytic/blockA/11_integrated_baseline_characterization/11_integrated_baseline_patient_level.provenance.json"
DENIED = re.compile(r"patient_level\.(csv|parquet)$|patient_audit|patient_followup|intervisit_gaps|episode_audit|zero_day_gap_audit|integrated_longitudinal_clinical_episode|overlap_episode_level|labs_episode_wide|patient_id_crosswalk", re.I)


@dataclass(frozen=True)
class Entry:
    id: str
    spec: dict

    @property
    def path(self):
        return self.spec["path"]

    @property
    def producer(self):
        return self.spec.get("producer", "Not supplied")


class Registry:
    def __init__(self, config=None, outputs=None, repo=None):
        self.outputs = Path(outputs) if outputs is not None else outputs_root()
        self.repo = Path(repo) if repo is not None else repo_root()
        cfg = yaml.safe_load(Path(config or APP_ROOT / "config/output_registry.yml").read_text())
        self.entries = {}
        for key, spec in cfg["outputs"].items():
            self.add(key, spec)

    def add(self, key, spec):
        if key in self.entries:
            raise SchemaError("Duplicate registry id: " + key)
        if spec.get("kind") not in {"csv", "json", "figure", "plotly_json"}:
            raise SchemaError("Unsupported output type")
        path = spec["path"]
        if not isinstance(path,str) or not path:
            raise SchemaError('Output path must be a nonempty string')
        provenance = path == PROVENANCE_PATH and spec.get("path_root") == "repo" and spec["kind"] == "json"
        if not provenance and (spec.get("path_root") == "repo" or Path(path).parts[0] == "data" or DENIED.search(path) or Path(path).suffix == ".parquet"):
            raise SchemaError("Patient-level or data path rejected: " + path)
        try:
            contained(self.repo if provenance else self.outputs, path)
        except ValueError as exc:
            raise SchemaError(str(exc)) from exc
        self.entries[key] = Entry(key, spec)

    def get(self, key):
        try:
            return self.entries[key]
        except KeyError as exc:
            raise MissingOutput("Unregistered output: " + key) from exc

    def path(self, key):
        entry = self.get(key)
        try:
            return contained(self.repo if entry.spec.get("path_root") == "repo" else self.outputs, entry.path)
        except ValueError as exc:
            raise SchemaError(str(exc)) from exc

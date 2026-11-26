"""Generate a private, CPU-only report from immutable published runs."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

from .compare_runs import baseline_files
from .report import report
from .report_progress import DEFINITIONS, VERSION
from .report_sources import INPUTS, RUN_ID, file_manifest, inventory_policy, published
from .settings import MODULE, digest, file_hash, pipeline_config, read_yaml, write_json

TARGET_FILES = ("run_manifest.json", "unique_values.parquet", "semantic_dictionary.parquet",
                "variable_inventory.csv", "semantic_facts.parquet")


def baseline_run(cfg, override=None):
    paths = cfg["paths"]
    if override is None:
        files = baseline_files(cfg)
        folder = files["run_manifest.json"].parent if files else None
        expected = None if paths.get("baseline") else cfg.get("comparison", {}).get("baseline_run_id")
    else:
        folder, expected = Path(paths["output"]) / override, override
    if folder and folder.exists():
        return published(folder, Path(paths["analytic"]), expected)
    return None


def target_run(cfg, baseline, selection):
    root, analytic = (Path(cfg["paths"][key]) for key in ("output", "analytic"))
    if selection:
        folders = [root / selection]
    else:
        folders = sorted(path for path in root.glob("*") if RUN_ID.fullmatch(path.name) and
                         (path / "run_manifest.json").is_file())
    candidates, errors = [], []
    for folder in folders:
        try:
            raw = json.loads((folder / "run_manifest.json").read_text())
            if str(raw.get("schema_version")) != "2.0":
                if selection:
                    errors.append(f"{folder.name}: expected V2 schema 2.0")
                continue
            absent = [name for name in TARGET_FILES if not (folder / name).is_file()]
            if absent:
                errors.append(f"{folder.name}: missing {', '.join(absent)}")
                continue
            candidates.append(published(folder, analytic, folder.name))
        except (OSError, ValueError) as error:
            errors.append(f"{folder.name}: {error}")
    if not candidates:
        details = "; ".join(errors)
        raise FileNotFoundError(f"No complete published V2 run in {root}" + (f" ({details})" if details else ""))
    if len(candidates) == 1:
        return candidates[0]
    if baseline:
        previous = inventory_policy(baseline.rows("variable_inventory.csv"))
        matches = []
        for candidate in candidates:
            current = inventory_policy(candidate.rows("variable_inventory.csv"))
            if previous is not None and current is not None and all(previous[key] == current[key] for key in previous.keys() & current.keys()) and all(
                    baseline.manifest["inputs"].get(key) == candidate.manifest["inputs"].get(key) for key in INPUTS):
                matches.append(candidate)
        if len(matches) == 1:
            return matches[0]
    choices = "\n".join(f'  {r.manifest["run_id"]}  {r.manifest["completed_at"]}  schema {r.manifest["schema_version"]}'
                        for r in candidates)
    raise ValueError(f"Choose report_selection.target_run_id in configs/report_selection.local.yaml. Published V2 runs:\n{choices}")


def generate(cfg, selection, output_root):
    """Build and seal a report alongside the original publication trees."""
    for name in ("target_run_id", "baseline_run_id"):
        value = selection.get(name)
        if value is not None and (not isinstance(value, str) or not RUN_ID.fullmatch(value)):
            raise ValueError(f"report_selection.{name} must be a full run ID")
    for root in (cfg["paths"].get(key) for key in ("output", "analytic", "baseline")):
        if root and output_root.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("Historical report output must be outside published run directories")
    baseline = baseline_run(cfg, selection.get("baseline_run_id"))
    target = target_run(cfg, baseline, selection.get("target_run_id"))
    sources = {key: dict(run_id=run.manifest["run_id"], manifest_sha256=file_hash(run.qc / "run_manifest.json"),
                         outputs=run.manifest["outputs"], inputs=run.manifest["inputs"]) if run else None
               for key, run in (("v1", baseline), ("v2", target))}
    code = {path.relative_to(MODULE).as_posix(): file_hash(path) for path in sorted(MODULE.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".html", ".yaml"} and not path.name.endswith("local.yaml")}
    identity = dict(sources=sources, code=code, rules=VERSION, selection=selection,
                    protection=cfg["concepts"], normalization=cfg["normalization"], report=cfg["report"])
    report_id = digest(identity)
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    final = output_root / report_id
    with (output_root / ".report.lock").open("a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if final.exists():
            sealed = json.loads((final / "report_manifest.json").read_text())
            if sealed.get("report_id") != report_id or sealed.get("identity") != identity or sealed.get("outputs") != file_manifest(
                    final, skip={"report_manifest.json"}):
                raise ValueError(f"Existing historical report failed verification: {final}")
            return final, json.loads((final / "progress_summary.json").read_text())
        with tempfile.TemporaryDirectory(dir=output_root, prefix=".report-") as staging:
            staged = Path(staging)
            model = report(staged, cfg, target.manifest, target=target, baseline=baseline)
            for key, run in (("v1", baseline), ("v2", target)):
                if run and (file_hash(run.qc / "run_manifest.json") != sources[key]["manifest_sha256"]
                            or published(run.qc, run.analytic.parent).manifest != run.manifest):
                    raise ValueError("A published input changed during report generation")
            for path in staged.iterdir():
                path.chmod(0o600)
            write_json(staged / "report_manifest.json", dict(report_id=report_id, identity=identity, sources=sources,
                       scope=model["summary"]["scope"], missing=model["missing"], definitions=DEFINITIONS,
                       denominators={k: model["summary"][k] for k in ("cumulative", "incremental", "historical_denominator")},
                       version=VERSION, completed_at=datetime.now(timezone.utc).isoformat(), outputs=file_manifest(staged)))
            staged.rename(final)
    return final, model["summary"]


def main():
    if len(sys.argv) != 1:
        raise SystemExit("Configure this command through configs/report_selection.local.yaml.")
    try:
        cfg = pipeline_config()
        local = MODULE / "configs" / "report_selection.local.yaml"
        selection = read_yaml(local).get("report_selection", {}) if local.is_file() else {}
        if not isinstance(selection, dict):
            raise ValueError("report_selection must be a YAML mapping.")
        output_root = Path(cfg["paths"]["obj1_root"]) / (selection.get("output_root") or
                          "outputs/tables/qc/semantic_harmonization_reports")
        out, summary = generate(cfg, selection, output_root)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
    print(out.resolve() / "semantic_harmonization_final.html")
    print(f'{summary["variables"]} eligible variables; {summary["initial_variables"]} with confirmed initial work.')
    print(", ".join(f"{key}: {value}" for key, value in summary["variable_states"].items()))


if __name__ == "__main__":
    main()

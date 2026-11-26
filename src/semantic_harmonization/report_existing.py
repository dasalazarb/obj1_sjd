"""Generate a private, CPU-only report from immutable published runs."""
import pyarrow.parquet as pq

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


def previous_run(cfg, override=None, *, reference="previous_v2_run_id"):
    """Load the explicitly pinned previous V2 when it is available."""
    previous_id = override or cfg.get("comparison", {}).get(reference)
    if previous_id is None:
        return None
    if not isinstance(previous_id, str) or not RUN_ID.fullmatch(previous_id):
        raise ValueError(f"{reference} must be a full run ID")
    root = Path(cfg["paths"]["output"]) / previous_id
    if not root.is_dir() and override is None:
        return None
    run = published(root, Path(cfg["paths"]["analytic"]), previous_id)
    if str(run.manifest["schema_version"]) != "2.0":
        raise ValueError("Previous run does not have V2 schema")
    return run


def frozen_work(cfg, output_root, previous, baseline=None, *, report_reference="historical_report_id"):
    """Authenticate a pinned report and read its original work identities."""
    reference = cfg.get("comparison", {}).get(report_reference)
    if previous is None or reference is None:
        return None, None
    if not isinstance(reference, str) or not RUN_ID.fullmatch(reference):
        raise ValueError(f"{report_reference} must be a full report ID")
    path = output_root / reference
    if not path.is_dir():
        return None, None
    manifest = json.loads((path / "report_manifest.json").read_text())
    sources = manifest.get("sources", {})
    source = sources.get("v2") or {}
    v1 = sources.get("v1") or {}
    if (manifest.get("report_id") != reference or digest(manifest.get("identity")) != reference or
            source.get("run_id") != previous.manifest["run_id"] or
            source.get("manifest_sha256") != file_hash(previous.qc / "run_manifest.json") or
            baseline and (v1.get("run_id") != baseline.manifest["run_id"] or
                          v1.get("manifest_sha256") != file_hash(baseline.qc / "run_manifest.json")) or
            manifest.get("outputs") != file_manifest(path, skip={"report_manifest.json"})):
        raise ValueError("Frozen historical work failed report source or content verification")
    snapshot = path / "work_snapshot.parquet"
    if snapshot.is_file():
        rows = pq.read_table(snapshot).to_pylist()
    else:
        html = (path / "semantic_harmonization_final.html").read_text(encoding="utf-8")
        literal = html.split('<script id="payload" type="application/json">', 1)[1].split("</script>", 1)[0]
        rows = json.loads(literal)["work"]
    if len({row["work_id"] for row in rows}) != len(rows):
        raise ValueError("Frozen report contains duplicate work IDs")
    return rows, file_hash(path / "report_manifest.json")


def frozen_references(cfg, output_root, previous, recent, baseline):
    """Carry the original work IDs from either authenticated comparison publication."""
    frozen, historical_hash = frozen_work(cfg, output_root, previous, baseline)
    recent_work, recent_hash = frozen_work(cfg, output_root, recent, baseline, report_reference="recent_report_id")
    if frozen is None and recent_hash:
        crosswalk = output_root / cfg["comparison"]["recent_report_id"] / "historical_work_crosswalk.parquet"
        if crosswalk.is_file():
            current = {item["work_id"]: item for item in recent_work}
            entries = pq.read_table(crosswalk).to_pylist()
            if entries:
                if len({row["work_id"] for row in entries}) != len(entries):
                    raise ValueError("Historical crosswalk contains duplicate work IDs")
                frozen = []
                for row in entries:
                    work = current.get(row["current_work_id"], {})
                    frozen.append(work | dict(
                        work_id=row["work_id"], recent_work_id=row["current_work_id"],
                        mapping_id=work.get("mapping_id") or row["mapping_id"], variable=row["variable"],
                        problem_kind=row["kind"], state_v1=row["state_v1"], state_v2=row["state_previous_v2"],
                        confirmed_initial=True, frozen_reconciliation=row["reconciliation"],
                        frozen_evidence=row["evidence"]))
    return frozen, historical_hash, recent_hash


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
    for name in ("target_run_id", "baseline_run_id", "previous_v2_run_id"):
        value = selection.get(name)
        if value is not None and (not isinstance(value, str) or not RUN_ID.fullmatch(value)):
            raise ValueError(f"report_selection.{name} must be a full run ID")
    for root in (cfg["paths"].get(key) for key in ("output", "analytic", "baseline")):
        if root and output_root.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("Historical report output must be outside published run directories")
    baseline = baseline_run(cfg, selection.get("baseline_run_id"))
    target = target_run(cfg, baseline, selection.get("target_run_id"))
    previous = previous_run(cfg, selection.get("previous_v2_run_id"))
    if previous and previous.manifest["run_id"] == target.manifest["run_id"]:
        if selection.get("previous_v2_run_id"):
            raise ValueError("Previous and target V2 publications must differ")
        previous = None
    recent = previous_run(cfg, reference="recent_v2_run_id")
    if recent and recent.manifest["run_id"] == target.manifest["run_id"]:
        recent = None
    frozen, historical_hash, recent_report_hash = frozen_references(cfg, output_root, previous, recent, baseline)
    references = (("v1", baseline), ("previous_v2", previous), ("recent_v2", recent), ("v2", target))
    sources = {key: dict(run_id=run.manifest["run_id"], manifest_sha256=file_hash(run.qc / "run_manifest.json"),
                         outputs=run.manifest["outputs"], inputs=run.manifest["inputs"]) if run else None
               for key, run in references}
    code = {path.relative_to(MODULE).as_posix(): file_hash(path) for path in sorted(MODULE.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".html", ".yaml"} and not path.name.endswith("local.yaml")}
    identity = dict(sources=sources, frozen_work_sha256=historical_hash, recent_report_sha256=recent_report_hash, code=code, rules=VERSION, selection=selection,
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
            model = report(staged, cfg, target.manifest, target=target, baseline=baseline, previous=previous, recent=recent, frozen=frozen)
            for key, run in references:
                if run and (file_hash(run.qc / "run_manifest.json") != sources[key]["manifest_sha256"]
                            or published(run.qc, run.analytic.parent).manifest != run.manifest):
                    raise ValueError("A published input changed during report generation")
            if historical_hash and frozen_work(cfg, output_root, previous, baseline)[1] != historical_hash:
                raise ValueError("Frozen historical work changed during report generation")
            if recent_report_hash and frozen_work(
                    cfg, output_root, recent, baseline, report_reference="recent_report_id")[1] != recent_report_hash:
                raise ValueError("Recent V2 report changed during report generation")
            for path in staged.iterdir():
                path.chmod(0o600)
            summary = model["summary"]
            denominator_keys = ("cumulative", "incremental", "historical_denominator",
                                "comparable_mappings", "previous_v2_comparable_mappings", "recent_v2_comparable_mappings")
            effective = summary.get("effective_values") or {}
            write_json(staged / "report_manifest.json", dict(report_id=report_id, identity=identity, sources=sources,
                       scope=summary["scope"], assessment=summary["assessment"], missing=model["missing"],
                       definitions=DEFINITIONS, denominators={k: summary[k] for k in denominator_keys} | {
                           k: effective.get(k) for k in ("episode_cells", "source_event_cells", "protected_cells")},
                       version=VERSION, completed_at=datetime.now(timezone.utc).isoformat(), outputs=file_manifest(staged)))
            staged.rename(final)
    return final, model["summary"]


def update_report_pointer(output_root, run_id, report_id):
    """Publish the convenience selector atomically outside all sealed run trees."""
    pointer = output_root / "current_v2.json"
    with tempfile.TemporaryDirectory(dir=output_root, prefix=".current-") as directory:
        staged = Path(directory) / pointer.name
        write_json(staged, {"target_run_id": run_id, "report_id": report_id,
                            "html": f"{report_id}/semantic_harmonization_final.html"})
        staged.chmod(0o600)
        staged.replace(pointer)


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
        pointer = output_root / "current_v2.json"
        if not selection.get("target_run_id") and pointer.is_file():
            selection["target_run_id"] = json.loads(pointer.read_text())["target_run_id"]
        out, summary = generate(cfg, selection, output_root)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
    print(out.resolve() / "semantic_harmonization_final.html")
    print(f'{summary["variables"]} eligible variables; {summary["initial_variables"]} with confirmed initial work.')
    print(", ".join(f"{key}: {value}" for key, value in summary["variable_states"].items()))


if __name__ == "__main__":
    main()

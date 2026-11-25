"""Run the restricted semantic pipeline without command-line configuration."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import fcntl
from importlib.metadata import PackageNotFoundError, distributions, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import monotonic

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "semantic_harmonization"

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .analytic_representation import materialize
from .compare_runs import baseline_identity, review
from .application import apply, rebuild, validate_original, write_candidate, write_parquet
from .evaluation import evaluate
from .inference import LocalModel, apply_extraction, extract_batch, force_offline, profile, routing, validate_extraction, validate_profile
from .inventory import inventory
from .semantic_facts import canonical, from_storage, parse_mapping, verify_contract
from .rag import KnowledgeIndex, documents
from .report import report
from .resources import verify, vocabularies
from .schemas import (DICTIONARY, EXTRACTIONS, FACTS, IDENTITY, INVENTORY, LINEAGE, SEGMENTS, SEMANTIC, TRACE,
                      Extraction, Semantic, SemanticFact, SemanticSegment)
from .settings import MODULE, digest, file_hash, load_config, serialize, write_json
from .study_input import write_source_manifest
from .validation import comparison, downstream, summarize, validate_mappings

DERIVED_SCHEMA = pa.schema([("variable", pa.string()), ("recomputed", pa.bool_()), ("evaluated_episodes", pa.int64()),
                            *[(name, pa.string()) for name in ("before", "after", "transitions", "rule")],
                            ("before_distinct", pa.int64()), ("after_distinct", pa.int64())])
IMPACT_COLUMNS = ["study", "feature", "observed_before", "observed_after", "eligibility_before", "eligibility_after", "status"]


def now() -> str:
    """Return an ISO UTC provenance timestamp."""
    return datetime.now(timezone.utc).isoformat()


def build_identity(cfg: dict, resources: dict, inputs: dict) -> dict:
    """Bind cache identity to original bytes, executable code and dependencies."""
    source = {p.relative_to(MODULE).as_posix(): file_hash(p) for p in sorted(MODULE.rglob("*"))
              if p.is_file() and (p.suffix in {".py", ".yaml", ".html", ".lock", ".sbatch", ".sh"}
                                  and not p.name.endswith("local.yaml"))}
    obj1_root = Path(cfg["paths"]["obj1_root"])
    contracts = ["src/block_A/01_serological_profile.py", "src/block_A/10_build_integrated_longitudinal_dataset.py",
                 "src/studies/pharma/00_profile_labs.py", "src/studies/graph/configs/pipeline.yaml"]
    contracts += ["src/studies/graph/_feature_catalog.py", "src/studies/graph/_lab_contract.py",
                  "src/studies/graph/_pipeline.py", "src/studies/_shared.py"]
    external = {name: file_hash(obj1_root / name) for name in contracts if (obj1_root / name).is_file()}
    packages = {}
    for name in ("pandas", "pyarrow", "pydantic", "numpy", "torch", "transformers", "vllm", "rank-bm25", "rapidfuzz"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    resource_identity = {name: {k: item.get(k) for k in ("revision", "version", "sha256", "status")}
                         for name, item in resources["resources"].items()}
    return dict(inputs=inputs, source=digest(source), obj1_contracts=external, config=digest(cfg),
                resources=resource_identity, dependencies=packages, baseline=baseline_identity(cfg),
                execution_environment={
                    "container_sha256": os.environ.get("SJD_CONTAINER_SHA256"),
                    "runtime_sha256": digest(sorted((item.metadata["Name"], item.version) for item in distributions()
                                                    if item.metadata["Name"])),
                })


def inputs_for(cfg: dict) -> dict:
    """Hash the required Step 10 input and configured optional sources."""
    paths = cfg["paths"]
    if not Path(paths["integrated"]).is_file():
        raise FileNotFoundError(f"Step 10 input missing: {paths['integrated']}")
    hashes = {}
    for name in ("integrated", "context", "source_events", "audit", "gold"):
        path = Path(paths[name]) if paths.get(name) else None
        if path is not None and path.is_file():
            hashes[name] = file_hash(path)
        elif path is not None and name in {"gold", "audit", "source_events"}:
            raise FileNotFoundError(f"Configured input missing: {path}")
    return hashes


def file_manifest(directory: Path, *, skip=()) -> dict:
    """Record all staged files by relative path and checksum."""
    return {path.relative_to(directory).as_posix(): file_hash(path)
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name not in skip}


def completed(out: Path, analytic: Path, identity: dict) -> bool:
    """Verify an existing completed run before reusing it."""
    manifest = json.loads((out / "run_manifest.json").read_text())
    if (manifest["identity"] != identity or manifest["outputs"]["report"] != file_manifest(out, skip={"run_manifest.json"})
            or manifest["outputs"]["analytic"] != file_manifest(analytic)):
        raise ValueError("Published run failed content verification")
    return True


def audit_reference(path: Path) -> dict:
    """Read the marginal audit only as an inventory reference."""
    from html.parser import HTMLParser

    class Payload(HTMLParser):
        reading = False
        content = ""

        def handle_starttag(self, tag, attrs):
            if tag == "script" and dict(attrs).get("id") == "payload":
                self.reading = True

        def handle_data(self, data):
            if self.reading:
                self.content += data

        def handle_endtag(self, tag):
            if tag == "script":
                self.reading = False

    parser = Payload()
    parser.feed(path.read_text())
    decoded = json.loads(parser.content)
    if not isinstance(decoded, dict):
        raise ValueError("Categorical audit payload must be an object")
    return {"status": "loaded as marginal reference", "sha256": file_hash(path)}


def run(cfg: dict, *, encoder=None, model_factory=LocalModel, resource_verify=verify) -> tuple[Path, Path]:
    """Stage and verify a content-addressed candidate and its report."""
    os.umask(0o077)
    force_offline()
    start = now()
    clock = monotonic()
    paths = cfg["paths"]
    hashes = inputs_for(cfg)
    resources = resource_verify(cfg)
    identity = build_identity(cfg, resources, hashes)
    run_id = digest(identity)
    out_root, analytic_root = Path(paths["output"]), Path(paths["analytic"])
    out_final, analytic_final = out_root / run_id, analytic_root / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    analytic_root.mkdir(parents=True, exist_ok=True)
    sessions = out_root / ".staging"
    sessions.mkdir(exist_ok=True)
    with (sessions / ".pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if inputs_for(cfg) != hashes:
            raise ValueError("Source content changed while waiting for the staging lock")
        if out_final.exists():
            completed(out_final, analytic_final, identity)
            return analytic_final, out_final
        resume = sessions / run_id
        resume.mkdir(exist_ok=True)
        with (resume / "progress.log").open("a", buffering=1) as log:
            def stage(name: str) -> None:
                log.write(f"{now()} {name}\n")

            stage("01 inventory")
            original = pq.read_table(paths["integrated"])
            master = original.to_pandas()
            context = pd.read_parquet(paths["context"]) if "context" in hashes else None
            source = pq.read_table(paths["source_events"]) if "source_events" in hashes else None
            inventory_paths = [resume / name for name in ("unique_values.parquet", "row_lineage.parquet", "variable_inventory.csv")]
            inventory_state = resume / "inventory_checkpoint.json"
            if inventory_state.exists() and all(p.is_file() for p in inventory_paths) and json.loads(inventory_state.read_text()) == {p.name: file_hash(p) for p in inventory_paths}:
                pairs = pq.read_table(inventory_paths[0], schema=INVENTORY).to_pylist()
                lineage = pq.read_table(inventory_paths[1], schema=LINEAGE).to_pylist()
                variables = pd.read_csv(inventory_paths[2]).astype(object).where(lambda df: df.notna(), None).to_dict("records")
            else:
                pairs, lineage, variables = inventory(master, context, source.to_pandas() if source is not None else None, hashes, cfg)
                write_parquet(inventory_paths[0], pairs, INVENTORY)
                write_parquet(inventory_paths[1], lineage, LINEAGE)
                pd.DataFrame(variables).to_csv(inventory_paths[2], index=False)
                write_json(inventory_state, {p.name: file_hash(p) for p in inventory_paths})

            stage("02 knowledge")
            docs = documents(cfg, resources)
            index = KnowledgeIndex(cfg, resources, docs, Path(paths["knowledge_dir"]) / "knowledge_index", encoder)
            stage("03 deterministic parsing")
            parsed_schema = pa.schema([*IDENTITY, *SEMANTIC])
            deterministic_path = resume / "deterministic_results.parquet"
            segments_path = resume / "semantic_segments.parquet"
            deterministic_facts_path = resume / "deterministic_facts.parquet"
            deterministic_state = resume / "deterministic_checkpoint.json"
            artifacts = [deterministic_path, segments_path, deterministic_facts_path]
            vocab = vocabularies(resources)
            if (deterministic_state.is_file() and all(p.is_file() for p in artifacts)
                    and json.loads(deterministic_state.read_text()) == {p.name: file_hash(p) for p in artifacts}):
                deterministic = {row["mapping_id"]: Semantic.model_validate({f: row[f] for f in SEMANTIC.names})
                                 for row in pq.read_table(deterministic_path).to_pylist()}
                segments = [SemanticSegment.model_validate(row) for row in pq.read_table(segments_path).to_pylist()]
                m1_facts = [from_storage(row) for row in pq.read_table(deterministic_facts_path).to_pylist()]
            else:
                deterministic, segments, m1_facts = {}, [], []
                for row in pairs:
                    parsed, parts, facts = parse_mapping(row, cfg, vocab)
                    deterministic[row["mapping_id"]] = parsed
                    segments.extend(parts)
                    m1_facts.extend(facts)
                write_parquet(deterministic_path, [dict(row, **deterministic[row["mapping_id"]].model_dump())
                                                  for row in pairs], parsed_schema)
                write_parquet(segments_path, [item.model_dump() for item in segments], SEGMENTS)
                write_parquet(deterministic_facts_path, [item.storage() for item in m1_facts], FACTS)
                write_json(deterministic_state, {p.name: file_hash(p) for p in artifacts})
            verify_contract(pairs, segments, m1_facts)
            parts_by_mapping, facts_by_mapping = defaultdict(list), defaultdict(list)
            for part in segments:
                parts_by_mapping[part.mapping_id].append(part)
            for fact in m1_facts:
                facts_by_mapping[fact.mapping_id].append(fact)
            routes = {}
            for row in pairs:
                key = row["mapping_id"]
                routes[key] = routing(row, deterministic[key], parts_by_mapping[key], facts_by_mapping[key])
            pending = [row for row in pairs if routes[row["mapping_id"]]["send"]]
            stage(f"03 deterministic counts mappings={len(pairs)} segments={len(segments)} facts={len(m1_facts)} residual={len(pending)}")
            stage("04 compatible retrieval")
            index.measure([row["concept"].replace("_", " ") for row in pending])
            retrieved = {row["mapping_id"]: index.retrieve(row) for row in pending}

            stage("05 GPT-OSS GPU smoke and extraction")
            model = model_factory(cfg)
            model.smoke()
            cache = resume / "extractions"
            cache.mkdir(exist_ok=True)
            outputs, missing = {}, []

            for row in pending:
                key = row["mapping_id"]
                target = cache / (key + ".json")
                if target.is_file():
                    cached = json.loads(target.read_text())
                    if cached["mapping_id"] != key or cached["status"] not in {
                        "validated", "context_limit", "unsupported_evidence", "invalid_schema"
                    }:
                        raise ValueError("Invalid extraction cache record")
                    if cached["status"] == "validated":
                        if profile(row) == "finding":
                            validate_extraction(serialize({"findings": cached["findings"]}), row, retrieved[key])
                        else:
                            candidates = [{field: fact[field] for field in (
                                "segment_id", "start", "end", "quote", "payload", "document_ids")} for fact in cached["facts"]]
                            regenerated = validate_profile(serialize({"candidates": candidates}), row, retrieved[key], cfg, vocab)
                            if [fact.model_dump() for fact in regenerated] != cached["facts"]:
                                raise ValueError("Cached proposals failed regeneration")
                    elif cached["findings"] or cached["facts"]:
                        raise ValueError("Rejected cache record contains proposals")
                    outputs[key] = cached
                else:
                    missing.append(row)
            size = cfg["model"]["batch_size"]
            for first in range(0, len(missing), size):
                for result in extract_batch(model, missing[first:first + size], retrieved, cfg["model"]["max_validation_retries"], cfg=cfg, vocab=vocab):
                    key = result["mapping_id"]
                    write_json(cache / (key + ".json"), result)
                    outputs[key] = result
            stage("06 mapping validation")
            final = {key: value.model_copy(deep=True) for key, value in deterministic.items()}
            dictionary, final_facts = [], list(m1_facts)
            for row in pairs:
                key = row["mapping_id"]
                semantic = final[key]
                if key in outputs:
                    item = outputs[key]
                    item["contribution_status"] = item["status"]
                    if item["status"] == "validated":
                        try:
                            if profile(row) == "finding":
                                added = apply_extraction(semantic, Extraction.model_validate({"findings": item["findings"]}),
                                                         row, cfg["prompts"]["version"])
                            else:
                                existing = {canonical(fact) for fact in facts_by_mapping[key]}
                                added = []
                                for candidate in map(SemanticFact.model_validate, item["facts"]):
                                    meaning = canonical(candidate)
                                    if meaning not in existing:
                                        existing.add(meaning)
                                        added.append(candidate)
                        except ValueError:
                            item["contribution_status"] = "conflict"
                        else:
                            final_facts.extend(added)
                            item["contribution_status"] = ("added" if added else "duplicate"
                                                           if item["findings"] or item["facts"] else "empty")
                    elif semantic.strategy == "unchanged":
                        semantic.interpretation_note = ("Full narrative exceeds the model context budget." if item["status"] == "context_limit"
                                                        else "Model extraction did not pass evidence or schema validation.")
                dictionary.append(dict(row, **semantic.model_dump(), knowledge_doc_ids=[item["doc_id"] for item in retrieved.get(key, [])]))
            validate_mappings(dictionary, lineage, variables, len(master))
            verify_contract(pairs, segments, final_facts)

            with tempfile.TemporaryDirectory(dir=analytic_root, prefix=".candidate-") as analytic_temp, \
                 tempfile.TemporaryDirectory(dir=out_root, prefix=".report-") as out_temp:
                analytic, out = Path(analytic_temp), Path(out_temp)
                for artifact in (*inventory_paths, *artifacts):
                    shutil.copyfile(artifact, out / artifact.name)
                (out / "knowledge_documents.jsonl").write_text("".join(serialize(row) + "\n" for row in docs))
                write_json(out / "knowledge_index_manifest.json", json.loads((index.path / "manifest.json").read_text()))
                write_json(out / "resources_manifest.json", resources)
                trace = [dict(item, mapping_id=key) for key, items in retrieved.items() for item in items]
                write_parquet(out / "retrieval_trace.parquet", trace, TRACE)
                write_parquet(out / "llm_extractions.parquet", [
                    dict(item, facts=[SemanticFact.model_validate(fact).storage() for fact in item["facts"]])
                    for item in outputs.values()], EXTRACTIONS)
                write_parquet(out / "semantic_facts.parquet", [fact.storage() for fact in final_facts], FACTS)
                write_parquet(out / "semantic_dictionary.parquet", dictionary, DICTIONARY)
                with (out / "semantic_dictionary.jsonl").open("w") as stream:
                    for row in dictionary:
                        stream.write(serialize(row) + "\n")

                stage("07 apply")
                wide, features, observations = apply(original, source, lineage, dictionary, analytic)
                context_candidate = None
                if context is not None:
                    context_table = pq.read_table(paths["context"])
                    context_candidate = analytic / "10_integrated_longitudinal_context_harmonized.parquet"
                    pq.write_table(context_table, context_candidate)
                    validate_original(context_table, pq.read_table(context_candidate))
                stage("08 derived reconstruction")
                wide, derived = rebuild(master, wide, observations, features)
                write_parquet(out / "derived_comparison.parquet", derived, DERIVED_SCHEMA)
                candidate_path = write_candidate(analytic, wide, features)
                stage("09 dataset validation")
                candidate = pq.read_table(candidate_path)
                validate_original(original, candidate)
                observed_original = candidate.select(original.column_names).to_pandas()
                pd.testing.assert_frame_equal(master, observed_original, check_exact=True)
                if inputs_for(cfg) != hashes:
                    raise ValueError("Source inputs changed during harmonization")
                comparison_rows = comparison(variables, dictionary)
                pd.DataFrame(comparison_rows).to_csv(out / "variable_comparison.csv", index=False)
                impact = downstream(master, candidate.to_pandas(), features, cfg, out, candidate_path,
                                    context_candidate, analytic_final)
                pd.DataFrame(impact, columns=IMPACT_COLUMNS).to_csv(out / "downstream_impact.csv", index=False)
                summary = summarize(master, variables, dictionary, observations, list(outputs.values()), features)
                summary["clinical_evaluation"] = evaluate(Path(paths["gold"]) if "gold" in hashes else None,
                                                          pairs, deterministic, final, retrieved, model, cfg,
                                                          vocab=vocab, routes=routes, segments=segments,
                                                          typed_m1=m1_facts, typed_m4=final_facts)
                summary["v2"] = review(out, cfg, hashes, pairs, segments, m1_facts, final_facts, routes,
                                       outputs, variables, identity["baseline"])
                summary["v2"]["analytic_representation"] = materialize(analytic, out, master, observations,
                                                                       final_facts, features, variables)
                summary["audit_reference"] = audit_reference(Path(paths["audit"])) if "audit" in hashes else {"status": "unavailable"}
                summary["source_events"] = "Read-only measurement input profiled alongside the Step 10 candidate" if source is not None else "Step 10 episode-variable adapter"
                write_json(out / "validation_summary.json", summary)
                if inputs_for(cfg) != hashes:
                    raise ValueError("Source inputs changed before publication")

                stage("10 offline report")
                revision = subprocess.run(["git", "-C", paths["obj1_root"], "rev-parse", "HEAD"], capture_output=True, text=True)
                manifest = dict(run_id=run_id, started_at=start, commit=revision.stdout.strip() or None,
                                schema_version=cfg["schema_version"], inputs=hashes,
                                input_paths={name: paths[name] for name in hashes}, model=identity["resources"]["gpt_oss"],
                                prompt={"version": cfg["prompts"]["version"], "sha256": digest(cfg["prompts"])},
                                identity=identity, patients=summary["patients"], episodes=len(master), columns=len(master.columns))
                report(out, cfg, manifest)
                if inputs_for(cfg) != hashes or baseline_identity(cfg) != identity["baseline"]:
                    raise ValueError("Inputs changed after validation")
                write_source_manifest(out, analytic, run_id, hashes)
                stage("validated and ready to publish")
                (out / "logs").mkdir()
                shutil.copyfile(resume / "progress.log", out / "logs" / "pipeline.log")
                manifest.update(completed_at=now(), elapsed_seconds=round(monotonic() - clock, 3),
                                outputs={"report": file_manifest(out), "analytic": file_manifest(analytic)})
                write_json(out / "run_manifest.json", manifest)
                installed_analytic = False
                if analytic_final.exists():
                    if file_manifest(analytic_final) != manifest["outputs"]["analytic"]:
                        raise ValueError("Incomplete analytic publication differs from this validated run")
                else:
                    analytic.rename(analytic_final)
                    installed_analytic = True
                try:
                    out.rename(out_final)
                except OSError:
                    if installed_analytic:
                        analytic_final.rename(analytic)
                    raise
        shutil.rmtree(resume)
    return analytic_final, out_final


def main() -> None:
    """Run using YAML and surface safe error types in scheduler output."""
    if len(sys.argv) != 1:
        raise SystemExit("Configure this command through YAML.")
    try:
        cfg = load_config()
    except (ValueError, RuntimeError) as error:
        import traceback

        print(f"ERROR TYPE: {type(error).__name__}", file=sys.stderr)

        for frame in traceback.extract_tb(error.__traceback__):
            print(
                f"{frame.filename}:{frame.lineno} "
                f"in {frame.name}",
                file=sys.stderr
            )

        raise SystemExit(1) from None
    try:
        _, out = run(cfg)
    except PermissionError as error:
        raise SystemExit(f"Harmonization stopped because an input or staging path is inaccessible ({error.filename}).") from None
    except FileNotFoundError as error:
        raise SystemExit(f"Harmonization stopped because a required file is missing ({error}).") from None
    except (ValueError, RuntimeError) as error:
        import traceback

        print(f"ERROR TYPE: {type(error).__name__}", file=sys.stderr)

        for frame in traceback.extract_tb(error.__traceback__):
            print(
                f"{frame.filename}:{frame.lineno} in {frame.name}",
                file=sys.stderr,
            )

        raise SystemExit(
            "Harmonization stopped before publication."
        ) from None
    print(f"Validated staging run: {out}")


if __name__ == "__main__":
    main()

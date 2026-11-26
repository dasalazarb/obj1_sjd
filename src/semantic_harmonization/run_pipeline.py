"""Run the restricted semantic pipeline without command-line configuration."""
from __future__ import annotations

from collections import Counter, defaultdict
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

from .analytic_representation import materialize, write_readiness
from .compare_runs import baseline_identity, review
from .application import apply, rebuild, validate_original, write_candidate, write_parquet
from .effective_values import EFFECTIVE, RULE as EFFECTIVE_CONTRACT, build_effective
from .evaluation import evaluate
from .inference import (EvidenceError, InferenceUnavailable, LocalModel, apply_extraction, empty_extraction,
                        extract_batch, force_offline, profile, routing, skipped_smoke, smoke_cases, validate_extraction,
                        validate_profile)
from .inference_audit import SMOKE_AUDIT, SMOKE_SUMMARY, TRANSIENT, summarize_inference
from .inventory import inventory
from .semantic_facts import canonical, from_storage, parse_mapping, verify_contract
from .rag import KnowledgeIndex, documents
from .recovery import historical_narratives, incoming_reuse, proposal_meanings, reconcile_narratives
from .report import report
from .report_existing import baseline_run, frozen_references, generate, previous_run, update_report_pointer
from .report_progress import selector_blockers
from .report_sources import Run, file_manifest
from .resources import verify, vocabularies
from .schemas import (DICTIONARY, EXTRACTIONS, FACTS, IDENTITY, INVENTORY, LINEAGE, SEGMENTS, SEMANTIC, TRACE,
                      Extraction, Semantic, SemanticFact, SemanticSegment)
from .settings import MODULE, digest, file_hash, inference_contract, inference_mode, load_config, serialize, write_json
from .study_input import verify_source, write_source_manifest
from .table_exports import export_csv, pair_directory
from .validation import comparison, downstream, summarize, validate_mappings

DERIVED_SCHEMA = pa.schema([("variable", pa.string()), ("recomputed", pa.bool_()), ("evaluated_episodes", pa.int64()),
                            *[(name, pa.string()) for name in ("before", "after", "transitions", "rule")],
                            ("before_distinct", pa.int64()), ("after_distinct", pa.int64())])
IMPACT_COLUMNS = ["study", "feature", "observed_before", "observed_after", "eligibility_before", "eligibility_after", "status"]


def comparative_root(cfg: dict) -> Path:
    return Path(cfg["paths"]["obj1_root"]) / "outputs/tables/qc/semantic_harmonization_reports"


def now() -> str:
    """Return an ISO UTC provenance timestamp."""
    return datetime.now(timezone.utc).isoformat()


def comparison_identity(cfg):
    """Bind the configured frozen comparisons to their sealed manifests."""
    refs = cfg.get("comparison", {})
    comparison_paths = {
        "previous_v2": Path(cfg["paths"]["output"]) / str(refs.get("previous_v2_run_id")) / "run_manifest.json",
        "recent_v2": Path(cfg["paths"]["output"]) / str(refs.get("recent_v2_run_id")) / "run_manifest.json",
        "historical_work": comparative_root(cfg) / str(refs.get("historical_report_id")) / "report_manifest.json",
        "recent_report": comparative_root(cfg) / str(refs.get("recent_report_id")) / "report_manifest.json",
    }
    return {name: file_hash(path) if path.is_file() else None for name, path in comparison_paths.items()}


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
    return dict(effective_contract=EFFECTIVE_CONTRACT, inference_profile=digest(inference_contract(cfg)), inference_mode=inference_mode(cfg),
                comparison=comparison_identity(cfg), inputs=inputs, source=digest(source), obj1_contracts=external, config=digest(cfg),
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
        baseline = baseline_run(cfg)
        previous = previous_run(cfg)
        recent = previous_run(cfg, reference="recent_v2_run_id")
        frozen, historical_hash, recent_report_hash = frozen_references(cfg, comparative_root(cfg), previous, recent, baseline)
        for reference, expected in (("historical_work", historical_hash), ("recent_report", recent_report_hash)):
            if expected is not None and expected != identity["comparison"][reference]:
                raise ValueError("Pinned report changed before harmonization")

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
                pairs, lineage, variables = inventory(
                    master, context, source.to_pandas() if source is not None else None, hashes, cfg,
                    arrow_episode=original, arrow_source=source)
                write_parquet(inventory_paths[0], pairs, INVENTORY)
                write_parquet(inventory_paths[1], lineage, LINEAGE)
                pd.DataFrame(variables).to_csv(inventory_paths[2], index=False)
                write_json(inventory_state, {p.name: file_hash(p) for p in inventory_paths})

            stage("02 knowledge")
            docs = documents(cfg, resources)
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
            pending = [dict(row, _residual_segment_ids=routes[row["mapping_id"]]["residual_segment_ids"])
                       for row in pairs if routes[row["mapping_id"]]["send"]]
            pending_by_id = {row["mapping_id"]: row for row in pending}
            stage(f"03 deterministic counts mappings={len(pairs)} segments={len(segments)} facts={len(m1_facts)} residual={len(pending)}")
            stage("04 compatible retrieval")
            retrieved, knowledge = {}, {"status": "no eligible residual retrieval"}
            if pending:
                index = KnowledgeIndex(cfg, resources, docs, Path(paths["knowledge_dir"]) / "knowledge_index", encoder)
                index.measure([row["concept"].replace("_", " ") for row in pending])
                retrieved = {row["mapping_id"]: index.retrieve(row) for row in pending}
                knowledge = json.loads((index.path / "manifest.json").read_text())
            stage("05 residual extraction and revalidation")
            cache = resume / "extractions"
            cache.mkdir(exist_ok=True)
            reusable = incoming_reuse((recent, previous, baseline), hashes, pending, lineage, cfg,
                                      identity["resources"], knowledge, retrieved)
            outputs, missing, reused, checkpoints = {}, [], {}, set()

            def revalidate(item, row, *, published_cache=False):
                key = row["mapping_id"]
                if item["mapping_id"] != key or item["status"] != "validated":
                    raise ValueError("Invalid extraction cache record")
                records = item["facts"] or []
                item = dict(item, facts=[])
                if profile(row) == "finding":
                    if records:
                        raise ValueError("Finding cache record contains profile facts")
                    validate_extraction(serialize({"findings": item["findings"]}), row, retrieved[key])
                else:
                    if item["findings"]:
                        raise ValueError("Profile cache record contains narrative findings")
                    facts = [from_storage(fact) if published_cache else SemanticFact.model_validate(fact)
                             for fact in records]
                    proposals = [{field: getattr(fact, field) if field != "payload" else fact.payload.model_dump()
                                  for field in ("segment_id", "start", "end", "quote", "payload", "document_ids")}
                                 for fact in facts]
                    item["facts"] = validate_profile(serialize({"candidates": proposals}), row, retrieved[key], cfg, vocab)
                return item

            for row in pending:
                key = row["mapping_id"]
                target = cache / (key + ".json")
                if target.is_file():
                    cached = json.loads(target.read_text())
                    validated = revalidate(cached, row)
                    outputs[key] = dict(validated, facts=[f.model_dump() for f in validated["facts"]])
                    checkpoints.add(key)
                    continue
                candidates = []
                for source_id, item in reusable.get(key, ()):
                    try:
                        candidates.append((source_id, revalidate(item, row, published_cache=True)))
                    except (KeyError, ValueError):
                        continue
                if candidates and all(proposal_meanings(item) == proposal_meanings(candidates[0][1])
                                      for _, item in candidates):
                    source_id, item = candidates[0]
                    outputs[key] = dict(item, facts=[f.model_dump() for f in item["facts"]])
                    reused[key] = source_id
                else:
                    missing.append(row)
            mode = identity["inference_mode"]
            model = None
            smoke = skipped_smoke("inference_disabled" if mode == "disabled" else "no_residual_inference")
            if mode == "auto" and (missing or pending and "gold" in hashes):
                stage("05 GPT-OSS GPU smoke and extraction")
                fixtures = smoke_cases(cfg)
                try:
                    model = model_factory(cfg)
                except InferenceUnavailable:
                    smoke = skipped_smoke("inference_runtime_error", status="disabled")
                    print("GPT-OSS smoke phase=initialization reason=inference_runtime_error", flush=True)
                else:
                    smoke = model.smoke(fixtures)
            model_initialized = model is not None
            engine_available = smoke.model_available
            enabled = {name for name, result in smoke.profiles.items() if result.status == "passed"}
            groups = defaultdict(list)
            for row in missing:
                key = row["mapping_id"]
                result = smoke.profiles[profile(row)]
                if mode == "disabled":
                    outputs[key] = empty_extraction(row, "not_run", "inference_disabled")
                elif result.status == "disabled" and (engine_available or result.reason_code != "inference_runtime_error"):
                    outputs[key] = empty_extraction(row, "profile_disabled", result.reason_code)
                elif not engine_available:
                    outputs[key] = empty_extraction(row, "model_unavailable", "inference_runtime_error")
                elif result.status == "passed":
                    groups[result.profile].append(row)
                else:
                    raise ValueError("An eligible extraction lacks a smoke outcome")
            size = cfg["model"]["batch_size"]
            sent = set()
            for group in groups.values():
                for first in range(0, len(group), size):
                    batch = group[first:first + size]
                    results = None
                    if engine_available:
                        sent.update(row["mapping_id"] for row in batch)
                        try:
                            results = extract_batch(model, batch, retrieved, cfg["model"]["max_validation_retries"],
                                                    cfg=cfg, vocab=vocab)
                        except InferenceUnavailable:
                            engine_available = False
                            stage("05 optional inference stopped: inference_runtime_error")
                    if results is None:
                        results = [empty_extraction(row, "model_unavailable", "inference_runtime_error") for row in batch]
                    for row, result in zip(batch, results, strict=True):
                        key = row["mapping_id"]
                        if result["mapping_id"] != key:
                            raise ValueError("Extraction result mapping differs from the submitted mapping")
                        if result["status"] == "context_limit":
                            sent.discard(key)
                        if result["status"] == "validated" and (result["findings"] or result["facts"]):
                            write_json(cache / (key + ".json"), result)
                        outputs[key] = result
            if outputs.keys() != pending_by_id.keys():
                raise ValueError("Residual extraction audit does not cover every routed mapping")
            stage("06 mapping validation")
            prior_findings = historical_narratives((recent, previous, baseline), hashes, pairs, lineage, segments)
            final = {key: value.model_copy(deep=True) for key, value in deterministic.items()}
            dictionary, final_facts, historical_facts, added_counts = [], list(m1_facts), {}, Counter()
            for row in pairs:
                key = row["mapping_id"]
                semantic = final[key]
                added = []
                if key in outputs:
                    item = outputs[key]
                    item["contribution_status"] = item["status"]
                    if item["status"] == "validated":
                        try:
                            if profile(row) == "finding":
                                added = apply_extraction(semantic, Extraction.model_validate({"findings": item["findings"]}),
                                                         pending_by_id[key], cfg["prompts"]["version"])
                            else:
                                existing = {canonical(fact) for fact in facts_by_mapping[key]}
                                added = []
                                for candidate in map(SemanticFact.model_validate, item["facts"]):
                                    meaning = canonical(candidate)
                                    if meaning not in existing:
                                        existing.add(meaning)
                                        added.append(candidate)
                        except EvidenceError:
                            added = []
                            item["contribution_status"] = "conflict"
                        else:
                            final_facts.extend(added)
                            added_counts[key] = len(added)
                            if key in reused:
                                historical_facts.update((fact.fact_id, reused[key]) for fact in added)
                            item["contribution_status"] = ("added" if added else "duplicate"
                                                           if item["findings"] or item["facts"] else "empty")
                    elif semantic.strategy == "unchanged":
                        semantic.interpretation_note = {
                            "context_limit": "Full narrative exceeds the model context budget.",
                            "profile_disabled": "The profile did not pass smoke validation.",
                            "model_unavailable": "The local inference engine was unavailable.",
                            "not_run": "Model inference was disabled by configuration.",
                        }.get(item["status"], "Model extraction did not pass evidence or schema validation.")
                recovered, provenance = reconcile_narratives(
                    row, semantic, [*facts_by_mapping[key], *added], prior_findings.get(key),
                    retrieved.get(key, []), cfg["prompts"]["version"])
                final_facts.extend(recovered)
                historical_facts.update(provenance)
                dictionary.append(dict(row, **semantic.model_dump(), knowledge_doc_ids=[item["doc_id"] for item in retrieved.get(key, [])]))
            validate_mappings(dictionary, lineage, variables, len(master))
            verify_contract(pairs, segments, final_facts)

            evaluation = evaluate(Path(paths["gold"]) if "gold" in hashes else None,
                                  pairs, deterministic, final, retrieved, model if engine_available else None, cfg,
                                  vocab=vocab, routes=routes, segments=segments, typed_m1=m1_facts, typed_m4=final_facts,
                                  enabled_profiles=enabled)
            if evaluation.get("engine_failed"):
                engine_available = False
            failures = {key: item["status"] for key, item in outputs.items() if item["status"] in TRANSIENT}
            degraded = (any(result.status == "disabled" for result in smoke.profiles.values()) or bool(failures)
                        or evaluation.get("reason") == "Inference unavailable during holdout")
            if degraded:
                # Keep a recoverable run from permanently occupying the normal
                # identity. An unchanged degraded outcome can still be reused.
                identity["inference_degradation"] = digest({
                    "smoke": smoke.audit, "failures": failures, "evaluation": evaluation,
                    "profiles": {name: result.status for name, result in smoke.profiles.items()},
                    "added_facts": sorted(fact.fact_id for fact in final_facts[len(m1_facts):]),
                })
                run_id = digest(identity)
                out_final, analytic_final = out_root / run_id, analytic_root / run_id
                if out_final.exists():
                    completed(out_final, analytic_final, identity)
                    shutil.rmtree(resume)
                    return analytic_final, out_final

            with tempfile.TemporaryDirectory(dir=analytic_root, prefix=".candidate-") as analytic_temp, \
                 tempfile.TemporaryDirectory(dir=out_root, prefix=".report-") as out_temp:
                analytic, out = Path(analytic_temp), Path(out_temp)
                for artifact in (*inventory_paths, *artifacts):
                    shutil.copyfile(artifact, out / artifact.name)
                (out / "knowledge_documents.jsonl").write_text("".join(serialize(row) + "\n" for row in docs))
                write_json(out / "knowledge_index_manifest.json", knowledge)
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
                wide, features, observations = apply(original, source, lineage, dictionary, analytic, final_facts)
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
                representation, entries, temporal, baseline_counts = materialize(
                    analytic, out, master, observations, final_facts, features, variables, wide)
                stage("09a effective values and Arrow validation")
                effective = build_effective(
                    original, source, lineage, dictionary, final_facts, segments, cfg, run_id, analytic, out,
                    historical=historical_facts, outputs=outputs, routes=routes)
                comparison_rows = comparison(variables, dictionary)
                pd.DataFrame(comparison_rows).to_csv(out / "variable_comparison.csv", index=False)
                impact = downstream(master, candidate.to_pandas(), features, cfg, out, candidate_path,
                                    context_candidate, analytic_final)
                write_readiness(out, entries, temporal, baseline_counts)
                pd.DataFrame(impact, columns=IMPACT_COLUMNS).to_csv(out / "downstream_impact.csv", index=False)
                summary = summarize(master, variables, dictionary, observations, list(outputs.values()), features)
                summary["technical_integrity"] = "passed"
                summary["clinical_evaluation"] = evaluation
                summary["effective_values"] = effective
                audit_batches = pq.ParquetFile(out / "effective_value_audit.parquet").iter_batches(
                    columns=["mapping_id", "effective_origin", "source_grain", "occurrences"])
                profile_summary = summarize_inference(
                    smoke, pending, outputs, reused.keys() | checkpoints, sent, lineage,
                    (row for batch in audit_batches for row in batch.to_pylist()), added_counts)
                inference_tables = {}
                for name, rows, schema in (("inference_smoke_audit", smoke.audit, SMOKE_AUDIT),
                                           ("inference_profile_summary", profile_summary, SMOKE_SUMMARY)):
                    path = out / f"{name}.parquet"
                    write_parquet(path, rows, schema)
                    export_csv(path)
                    sidecar = path.with_suffix(".csv.schema.json")
                    inference_tables[name] = json.loads(sidecar.read_text())["sha256"] | {sidecar.name: file_hash(sidecar)}
                summary["inference"] = {"mode": mode, "model_initialized": model_initialized,
                                        "model_available": engine_available,
                                        "status": "degraded" if degraded else "available" if enabled else "not_run",
                                        "revalidated_historical_mappings": len(reused),
                                        "accepted_historical_facts": len(historical_facts),
                                        "fresh_residual_mappings": len(missing),
                                        "profile_summary": {row["profile"]: row for row in profile_summary},
                                        "artifacts": inference_tables}
                summary["v2"] = review(out, cfg, hashes, pairs, segments, m1_facts, final_facts, routes,
                                       outputs, variables, identity["baseline"])
                summary["v2"]["analytic_representation"] = representation
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
                comparison_model = report(out, cfg, manifest, target=Run(out, analytic, manifest),
                                          baseline=baseline, previous=previous, recent=recent, frozen=frozen, routes=routes)
                for key in ("semantic_completeness", "downstream_activation"):
                    summary[key] = comparison_model["summary"][key]
                write_json(out / "validation_summary.json", summary)
                pair_directory(analytic)
                pair_directory(out)
                if (inputs_for(cfg) != hashes or baseline_identity(cfg) != identity["baseline"]
                        or comparison_identity(cfg) != identity["comparison"]):
                    raise ValueError("Inputs changed after validation")
                for key, config_key, reference in (("previous_v2", "previous_v2_run_id", previous),
                                                   ("recent_v2", "recent_v2_run_id", recent)):
                    if reference:
                        verified = previous_run(cfg, reference=config_key)
                        if (not verified or verified.manifest != reference.manifest
                                or file_hash(verified.qc / "run_manifest.json") != identity["comparison"][key]):
                            raise ValueError("A pinned V2 publication changed before publication")
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
        # Configuration is resolved before reading clinical inputs.
        raise SystemExit(f"Harmonization configuration is incomplete ({error}).") from None
    try:
        analytic, out = run(cfg)
        report_dir, summary = generate(cfg, {"target_run_id": out.name}, comparative_root(cfg))
        verify_source(out.name, analytic_root=analytic.parent, report_root=out.parent)
        blockers = selector_blockers(summary)
        if blockers:
            print("V2 selector retained (" + "; ".join(blockers) + ").")
        else:
            update_report_pointer(comparative_root(cfg), out.name, report_dir.name)
            print("V2 selector updated.")
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
    print(f"V2 run ID: {out.name}")
    print(f"Comparative report ID: {report_dir.name}")
    print(f"Effective Parquet: {analytic.resolve() / EFFECTIVE}")
    print(f"Effective CSV: {(analytic.resolve() / EFFECTIVE).with_suffix('.csv')}")
    print(f"Comparative HTML: {report_dir.resolve() / 'semantic_harmonization_final.html'}")


if __name__ == "__main__":
    main()

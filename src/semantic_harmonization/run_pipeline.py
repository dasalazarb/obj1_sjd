"""Materialize the Step 10 Effective dataset."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
from importlib.metadata import PackageNotFoundError, distributions, version
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from time import monotonic

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.semantic_harmonization"

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import common

from .analytic_representation import materialize, write_readiness
from .application import apply, recalculate_serology, validate_original, write_parquet
from .effective_values import EFFECTIVE, RULE as EFFECTIVE_CONTRACT, EffectiveText, build_effective, update_effective
from .inference import (EvidenceError, InferenceUnavailable, LocalModel, apply_extraction, empty_extraction,
                        extract_batch, force_offline, profile, routing, skipped_smoke, smoke_cases, validate_extraction,
                        validate_profile)
from .inference_audit import SMOKE_AUDIT, SMOKE_SUMMARY, TRANSIENT, summarize_inference
from .inventory import inventory
from .semantic_facts import canonical, from_storage, parse_mapping, verify_contract
from .rag import KnowledgeIndex, documents
from .report_sources import file_manifest, published
from .resources import verify, vocabularies
from .schemas import (DICTIONARY, EXTRACTIONS, FACTS, IDENTITY, INVENTORY, LINEAGE, ROUTES, SEGMENTS, SEMANTIC, TRACE,
                      Extraction, Semantic, SemanticFact, SemanticSegment)
from .settings import MODULE, digest, file_hash, inference_contract, inference_mode, load_config, serialize, write_json
from .study_input import EFFECTIVE_PUBLICATION, write_source_manifest
from .table_exports import export_csv, pair_directory
from .validation import summarize, validate_mappings


def now() -> str:
    """Return an ISO UTC provenance timestamp."""
    return datetime.now(timezone.utc).isoformat()


def build_identity(cfg: dict, resources: dict, inputs: dict) -> dict:
    """Bind cache identity to original bytes, executable code and dependencies."""
    source = {p.relative_to(MODULE).as_posix(): file_hash(p) for p in sorted(MODULE.iterdir())
              if p.is_file() and p.suffix in {".py", ".lock", ".sbatch", ".sh"}}
    effective_config = {name: value for name, value in cfg.items()
                        if name not in {"comparison", "report", "validation", "paths"}}
    effective_config["paths"] = {name: value for name, value in cfg["paths"].items()
                                 if name not in {"baseline", "audit", "gold"}}
    packages = {}
    for name in ("pandas", "pyarrow", "pydantic", "numpy", "torch", "transformers", "vllm", "rank-bm25", "rapidfuzz"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    resource_identity = {name: {k: item.get(k) for k in ("revision", "version", "sha256", "status")}
                         for name, item in resources["resources"].items()}
    return dict(publication=EFFECTIVE_PUBLICATION, effective_contract=EFFECTIVE_CONTRACT,
                inference_profile=digest(inference_contract(cfg)), inference_mode=inference_mode(cfg),
                comparison={"status": "not_used"}, baseline={"status": "not_used"},
                inputs=inputs, source=digest(source), config=digest(effective_config),
                resources=resource_identity, dependencies=packages,
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
    for name in ("integrated", "context", "source_events"):
        path = Path(paths[name]) if paths.get(name) else None
        if path is not None and path.is_file():
            hashes[name] = file_hash(path)
        elif path is not None and name == "source_events":
            raise FileNotFoundError(f"Configured source events missing: {path}")
    return hashes


def completed(out: Path, analytic: Path, identity: dict) -> None:
    """Verify an existing completed run before reusing it."""
    if published(out, analytic.parent, expected_id=analytic.name).manifest["identity"] != identity:
        raise ValueError("Published run failed identity verification")


def run(cfg: dict, *, encoder=None, model_factory=LocalModel, resource_verify=None) -> tuple[Path, Path]:
    """Stage and verify the Effective dataset before publishing it."""
    os.umask(0o077)
    force_offline()
    start = now()
    clock = monotonic()
    paths = cfg["paths"]
    hashes = inputs_for(cfg)
    resources = verify(cfg, require_inference=False) if resource_verify is None else resource_verify(cfg)
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
            mode = identity["inference_mode"]
            model_resource = identity["resources"]["gpt_oss"]
            can_extract = mode == "auto" and model_resource["status"] == "available"
            stage("04 compatible retrieval")
            retrieved, knowledge = {}, {"status": "not_used"}
            if pending and can_extract:
                index = KnowledgeIndex(cfg, resources, docs, Path(paths["knowledge_dir"]) / "knowledge_index", encoder)
                index.measure([row["concept"].replace("_", " ") for row in pending])
                retrieved = {row["mapping_id"]: index.retrieve(row) for row in pending}
                knowledge = json.loads((index.path / "manifest.json").read_text())
            stage("05 residual extraction and revalidation")
            cache = resume / "extractions"
            cache.mkdir(exist_ok=True)
            outputs, missing, checkpoints = {}, [], set()

            def revalidate(item, row):
                key = row["mapping_id"]
                if item["mapping_id"] != key or item["status"] != "validated":
                    raise ValueError("Invalid extraction cache record")
                if profile(row) == "finding":
                    if item["facts"]:
                        raise ValueError("Finding cache record contains profile facts")
                    validate_extraction(serialize({"findings": item["findings"]}), row, retrieved[key])
                    return item
                if item["findings"]:
                    raise ValueError("Profile cache record contains narrative findings")
                facts = [SemanticFact.model_validate(fact) for fact in item["facts"]]
                proposals = [{field: getattr(fact, field) for field in
                              ("segment_id", "start", "end", "quote", "document_ids")} |
                             {"payload": fact.payload.model_dump()} for fact in facts]
                validated = validate_profile(serialize({"candidates": proposals}), row, retrieved[key], cfg, vocab)
                return dict(item, facts=[fact.model_dump() for fact in validated])

            for row in pending:
                key = row["mapping_id"]
                target = cache / (key + ".json")
                if can_extract and target.is_file():
                    try:
                        outputs[key] = revalidate(json.loads(target.read_text()), row)
                    except (OSError, KeyError, TypeError, ValueError):
                        stage("05 ignored an invalid extraction checkpoint")
                    else:
                        checkpoints.add(key)
                        continue
                missing.append(row)
            model = None
            smoke = skipped_smoke("inference_disabled" if mode == "disabled" else "no_residual_inference")
            if mode == "auto" and missing and not can_extract:
                smoke = skipped_smoke("inference_runtime_error", status="disabled")
            if can_extract and missing:
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
            final = {key: value.model_copy(deep=True) for key, value in deterministic.items()}
            dictionary, final_facts, added_counts = [], list(m1_facts), Counter()
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
                            item["contribution_status"] = ("added" if added else "duplicate"
                                                           if item["findings"] or item["facts"] else "empty")
                    elif semantic.strategy == "unchanged":
                        semantic.interpretation_note = {
                            "context_limit": "Full narrative exceeds the model context budget.",
                            "profile_disabled": "The profile did not pass smoke validation.",
                            "model_unavailable": "The local inference engine was unavailable.",
                            "not_run": "Model inference was disabled by configuration.",
                        }.get(item["status"], "Model extraction did not pass evidence or schema validation.")
                dictionary.append(dict(row, **semantic.model_dump(), knowledge_doc_ids=[item["doc_id"] for item in retrieved.get(key, [])]))
            validate_mappings(dictionary, lineage, variables, len(master))
            verify_contract(pairs, segments, final_facts)

            failures = {key: item["status"] for key, item in outputs.items() if item["status"] in TRANSIENT}
            degraded = any(result.status == "disabled" for result in smoke.profiles.values()) or bool(failures)
            if degraded:
                # Keep a recoverable run from permanently occupying the normal
                # identity. An unchanged degraded outcome can still be reused.
                identity["inference_degradation"] = digest({
                    "smoke": smoke.audit, "failures": failures,
                    "profiles": {name: result.status for name, result in smoke.profiles.items()},
                    "added_facts": sorted(fact.fact_id for fact in final_facts[len(m1_facts):]),
                })
                run_id = digest(identity)
                out_final, analytic_final = out_root / run_id, analytic_root / run_id
                if out_final.exists():
                    completed(out_final, analytic_final, identity)
                    shutil.rmtree(resume)
                    return analytic_final, out_final

            with tempfile.TemporaryDirectory(dir=analytic_root, prefix=".effective-") as analytic_temp, \
                 tempfile.TemporaryDirectory(dir=out_root, prefix=".qc-") as out_temp:
                analytic, out = Path(analytic_temp), Path(out_temp)
                for artifact in (*inventory_paths, *artifacts):
                    shutil.copyfile(artifact, out / artifact.name)
                (out / "knowledge_documents.jsonl").write_text("".join(serialize(row) + "\n" for row in docs))
                write_json(out / "knowledge_index_manifest.json", knowledge)
                write_json(out / "resources_manifest.json", resources)
                trace = [dict(item, mapping_id=key) for key, items in retrieved.items() for item in items]
                write_parquet(out / "retrieval_trace.parquet", trace, TRACE)
                write_parquet(out / "llm_extractions.parquet", [
                    dict(item, facts=[SemanticFact.model_validate(fact).storage() for fact in item["facts"]],
                         revalidated_checkpoint=key in checkpoints)
                    for key, item in outputs.items()], EXTRACTIONS)
                write_parquet(out / "semantic_facts.parquet", [fact.storage() for fact in final_facts], FACTS)
                write_parquet(out / "semantic_dictionary.parquet", dictionary, DICTIONARY)
                with (out / "semantic_dictionary.jsonl").open("w") as stream:
                    for row in dictionary:
                        stream.write(serialize(row) + "\n")

                write_parquet(out / "mapping_routing.parquet", [
                    dict(route, mapping_id=key, extraction_status=outputs.get(key, {}).get("status"),
                         contribution_status=outputs.get(key, {}).get("contribution_status"),
                         extraction_reason_code=outputs.get(key, {}).get("reason_code"))
                    for key, route in routes.items()
                ], ROUTES)

                stage("07 materialize typed observations")
                _, _, observations = apply(original, source, lineage, dictionary, analytic, final_facts,
                                           project_wide=False)
                if context is not None:
                    context_table = pq.read_table(paths["context"])
                    context_path = analytic / "10_integrated_longitudinal_context_harmonized.parquet"
                    pq.write_table(context_table, context_path)
                    validate_original(context_table, pq.read_table(context_path))
                representation, entries, temporal, baseline_counts = materialize(
                    analytic, out, master, observations, final_facts, [], variables)
                write_readiness(out, entries, temporal, baseline_counts)

                stage("08 effective values and Arrow validation")
                effective = build_effective(
                    original, source, lineage, dictionary, final_facts, segments, cfg, run_id, analytic, out,
                    outputs=outputs, routes=routes)
                summary = summarize(master, variables, dictionary, observations, list(outputs.values()), [])
                summary["technical_integrity"] = "passed"
                summary["effective_values"] = effective
                summary["analytic_representation"] = representation
                summary["comparison"] = identity["comparison"]
                audit_batches = pq.ParquetFile(out / "effective_value_audit.parquet").iter_batches(
                    columns=["mapping_id", "effective_origin", "source_grain", "occurrences"])
                profile_summary = summarize_inference(
                    smoke, pending, outputs, checkpoints, sent, lineage,
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
                                        "revalidated_checkpoint_mappings": len(checkpoints),
                                        "fresh_residual_mappings": len(missing),
                                        "profile_summary": {row["profile"]: row for row in profile_summary},
                                        "artifacts": inference_tables}
                write_json(out / "validation_summary.json", summary)

                stage("09 verify paired exports and seal manifests")
                pair_directory(analytic)
                pair_directory(out)
                if inputs_for(cfg) != hashes:
                    raise ValueError("Source inputs changed before publication")
                revision = subprocess.run(["git", "-C", paths["obj1_root"], "rev-parse", "HEAD"], capture_output=True, text=True)
                manifest = dict(run_id=run_id, started_at=start, commit=revision.stdout.strip() or None,
                                schema_version=cfg["schema_version"], inputs=hashes,
                                input_paths={name: paths[name] for name in hashes}, model=model_resource,
                                prompt={"version": cfg["prompts"]["version"], "sha256": digest(cfg["prompts"])},
                                identity=identity, patients=summary["patients"], episodes=len(master), columns=len(master.columns))
                write_source_manifest(out, analytic, run_id, hashes, publication=EFFECTIVE_PUBLICATION)
                stage("validated and ready to publish")
                (out / "logs").mkdir()
                shutil.copyfile(resume / "progress.log", out / "logs" / "pipeline.log")
                analytic_hashes = file_manifest(analytic)
                effective_csv = Path(EFFECTIVE).with_suffix(".csv").name
                effective_schema = Path(EFFECTIVE).with_suffix(".csv.schema.json").name
                if json.loads((analytic / effective_schema).read_text())["sha256"] != {
                    name: analytic_hashes[name] for name in (EFFECTIVE, effective_csv)
                }:
                    raise ValueError("Effective CSV and Parquet hashes changed after verification")
                manifest.update(completed_at=now(), elapsed_seconds=round(monotonic() - clock, 3),
                                outputs={"report": file_manifest(out), "analytic": analytic_hashes})
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


def _active_review(original, wide, cfg):
    """Summarize the final episode table against its immutable source, not prior runs."""
    from .effective_values import encode_value
    from .inventory import protected, route

    audit, variables = [], []
    families = defaultdict(Counter)
    measures = ("cells", "source_null", "source_nonnull", "effective_cells",
                "effective_null", "effective_nonnull", "effective_empty",
                "changed_cells", "retained_exact_cells", "removed_cells",
                "new_nulls", "newly_empty_cells")
    for field in original.schema:
        name = field.name
        if protected(name, cfg):
            continue
        _, family = route(name, cfg)
        source_col = original[name]
        retained = name in wide.column_names
        final_col = wide[name] if retained else None
        counts = Counter(cells=len(source_col), source_null=source_col.null_count,
                         source_nonnull=len(source_col) - source_col.null_count)
        pairs = Counter()
        for i, raw in enumerate(source_col):
            raw_json = encode_value(raw, field.type)
            if not retained:
                counts["removed_cells"] += 1
                pairs[(raw_json, None, "dropped_column")] += 1
                continue
            effective = final_col[i]
            effective_json = encode_value(effective, final_col.type)
            counts["effective_cells"] += 1
            counts["effective_null"] += not effective.is_valid
            counts["effective_nonnull"] += effective.is_valid
            counts["effective_empty"] += effective_json == '""'
            counts["new_nulls"] += raw.is_valid and not effective.is_valid
            counts["newly_empty_cells"] += effective_json == '""' and raw_json != '""'
            same = raw_json == effective_json
            counts["retained_exact_cells" if same else "changed_cells"] += 1
            decision = ("original_null" if not raw.is_valid else "original_unchanged") if same else (
                "derived_recomputed" if name.startswith("sero__") else "text_harmonized")
            pairs[(raw_json, effective_json, decision)] += 1
        for (raw_json, effective_json, decision), n in pairs.items():
            audit.append(dict(variable=name, family=family, source_grain="episode_variable",
                              raw_type=str(field.type), raw_json=raw_json,
                              effective_type=str(final_col.type) if retained else None,
                              effective_json=effective_json, effective_origin=decision, occurrences=n))
        variables.append(dict(variable=name, family=family, source_grain="episode_variable",
                              **{key: counts[key] for key in measures}))
        families[family].update(counts)
        families[family]["variables"] += 1
        families[family]["retained_variables"] += retained
    family_rows = [dict(family=family, source_grain="episode_variable",
                        variables=counts["variables"], retained_variables=counts["retained_variables"],
                        **{key: counts[key] for key in measures})
                   for family, counts in sorted(families.items())]
    strings = ("variable", "family", "source_grain", "raw_type", "raw_json",
               "effective_type", "effective_json", "effective_origin")
    audit_schema = pa.schema([(key, pa.string()) for key in strings] + [("occurrences", pa.int64())])
    summary_schema = pa.schema([(key, pa.string()) for key in ("variable", "family", "source_grain")]
                               + [(key, pa.int64()) for key in measures])
    family_schema = pa.schema([(key, pa.string()) for key in ("family", "source_grain")]
                              + [(key, pa.int64()) for key in ("variables", "retained_variables", *measures)])
    return {
        "effective_value_audit": pa.Table.from_pylist(audit, schema=audit_schema),
        "effective_variable_summary": pa.Table.from_pylist(variables, schema=summary_schema),
        "effective_family_summary": pa.Table.from_pylist(family_rows, schema=family_schema),
    }


def _write_active_csv(table, path):
    """Write the current CSV without the legacy sealed-export hash machinery."""
    import csv
    from .table_exports import cell, CSV_ENCODING

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(table.column_names)
        for batch in table.to_batches(max_chunksize=2048):
            writer.writerows([cell(row[name]) for name in table.column_names] for row in batch.to_pylist())
    write_json(path.with_suffix(".csv.schema.json"), {
        "columns": [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in table.schema],
        "rows": table.num_rows, "encoding": CSV_ENCODING,
    })


def _write_active_html(path, original, wide, tables):
    """Render an offline, searchable review of the active output only."""
    from html import escape

    audit = tables["effective_value_audit"].to_pylist()
    variables = tables["effective_variable_summary"].to_pylist()
    families = tables["effective_family_summary"].to_pylist()
    changed = sum(row["changed_cells"] for row in variables)
    emptied = sum(row["newly_empty_cells"] for row in variables)
    dropped = original.num_columns - wide.num_columns

    def render_table(title, rows, columns):
        head = "".join("<th>" + escape(key.replace("_", " ")) + "</th>" for key in columns)
        body = []
        for row in rows:
            body.append("<tr>" + "".join("<td>" + escape(
                "—" if row.get(key) is None else str(row[key])) + "</td>" for key in columns) + "</tr>")
        return ("<section><h2>" + escape(title) + "</h2><div class='scroll'><table><thead><tr>"
                + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div></section>")

    cards = "".join("<div><small>" + label + "</small><strong>" + f"{value:,}" + "</strong></div>"
                    for label, value in (("Episodes", wide.num_rows), ("Final columns", wide.num_columns),
                                         ("Removed columns", dropped), ("Changed retained cells", changed),
                                         ("New empty strings", emptied)))
    summary_fields = ("cells", "effective_cells", "effective_nonnull", "effective_empty",
                      "changed_cells", "removed_cells", "new_nulls", "newly_empty_cells")
    sections = render_table("Clinical families", families, ("family", "variables", "retained_variables", *summary_fields))
    sections += render_table("Variable summary", variables, ("variable", "family", *summary_fields))
    sections += render_table("Value inventory", audit, ("variable", "family", "raw_json", "effective_json",
                                                         "effective_origin", "occurrences"))
    doc = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'">
<title>SjD | Effective clinical dataset</title><style>
*{box-sizing:border-box}body{font:14px system-ui,sans-serif;background:#f5f7fa;color:#172b3a;margin:0}
main{max-width:1600px;margin:auto;padding:32px}h1{font-size:30px;margin:0 0 8px}h2{font-size:19px}
p{line-height:1.6;color:#526475}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.cards div,section{background:white;border:1px solid #dbe3eb;border-radius:10px;padding:20px;margin-bottom:16px}
small{display:block;color:#536879}strong{display:block;font-size:28px;color:#176c65;margin-top:8px}
input{width:min(600px,100%);font:inherit;padding:12px;border:1px solid #8ea3b4;border-radius:7px;margin:12px 0 24px}
.scroll{overflow:auto;max-height:540px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #e3e9ed;vertical-align:top;overflow-wrap:anywhere}
th{position:sticky;top:0;background:#eaf1f5;white-space:nowrap}td{white-space:pre-wrap;min-width:80px}
td:first-child{min-width:170px}tr:nth-child(even){background:#f7fafb}a{color:#176c65;margin-right:12px}
[hidden]{display:none!important}</style></head><body><main>
<h1>Effective clinical dataset</h1>
<p>Current episode-level output after cases 1–8. Original values refer only to Step 10, not a previous harmonization run. Counts exclude protected and identifier columns. Removed columns have no effective value. JSON strings distinguish empty strings from nulls. These counts describe transformations, not independent clinical validation or LLM attribution.</p>
<div class="cards">""" + cards + """</div>
<p><a href="effective_variable_summary.csv">Variable CSV</a><a href="effective_family_summary.csv">Family CSV</a><a href="effective_value_audit.csv">Value CSV</a></p>
<label for="search">Search variables, families and results</label><br><input id="search" type="search" placeholder="Filter all tables..." autocomplete="off">
""" + sections + """<script>
const search=document.getElementById('search');
search.addEventListener('input',()=>{const q=search.value.trim().toLowerCase();
document.querySelectorAll('tbody tr').forEach(row=>{row.hidden=!row.textContent.toLowerCase().includes(q)});});
</script></main></body></html>"""
    path.write_text(doc, encoding="utf-8")


def _refresh_active_outputs(original, wide, cfg, target):
    """Stage and replace only the active wide table and its current review outputs."""
    qc = Path(cfg["paths"]["output"]) / target.parent.name
    qc.mkdir(mode=0o700, parents=True, exist_ok=True)
    tables = _active_review(original, wide, cfg)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".effective-") as analytic_tmp, \
         tempfile.TemporaryDirectory(dir=qc, prefix=".review-") as qc_tmp:
        analytic = Path(analytic_tmp)
        review = Path(qc_tmp)
        pq.write_table(wide, analytic / target.name)
        _write_active_csv(wide, (analytic / target.name).with_suffix(".csv"))
        for name, table in tables.items():
            pq.write_table(table, review / f"{name}.parquet")
            _write_active_csv(table, review / f"{name}.csv")
        _write_active_html(review / "semantic_harmonization_final.html", original, wide, tables)

        # All artifacts are prepared before the active parquet is replaced.
        for folder, destination in ((analytic, target.parent), (review, qc)):
            for staged in sorted(folder.iterdir()):
                if staged.name == target.name:
                    continue
                staged.chmod(0o600)
                staged.replace(destination / staged.name)
        staged = analytic / target.name
        staged.chmod(stat.S_IMODE(target.stat().st_mode))
        staged.replace(target)
    print(f"Effective CSV: {target.with_suffix('.csv')}", flush=True)
    print(f"Effective QC: {qc}", flush=True)
    print(f"Effective HTML: {qc / 'semantic_harmonization_final.html'}", flush=True)


def refresh_active(cfg: dict) -> Path:
    """Refresh the parquet selected by the downstream consumers in place."""
    paths = cfg["paths"]
    root = Path(paths["obj1_root"])
    effective_input = root / common.EFFECTIVE_LONGITUDINAL_PARQUET.relative_to(common.PROJECT_ROOT)
    target = common.resolve_effective_longitudinal_input(effective_input)
    if not target.is_file():
        raise FileNotFoundError(f"No existe el Effective activo en {effective_input}")

    force_offline()
    model = index = ipd = None

    def extract(rows):
        nonlocal model, index, ipd
        if model is None:
            resources = verify(cfg)
            ipd = vocabularies(resources)["ipd"]
            corpus = documents(cfg, resources)
            index = KnowledgeIndex(cfg, resources, corpus, Path(paths["knowledge_dir"]) / "knowledge_index")
            model = LocalModel(cfg)
        return model.extract_compact(rows, index, ipd)

    directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX)
        original = pq.read_table(paths["integrated"])
        context_path = paths.get("context")
        context = pd.read_parquet(context_path) if context_path and Path(context_path).is_file() else None
        text = EffectiveText(cfg, extract)
        wide = update_effective(original, pq.read_table(target), context, text)
        wide = recalculate_serology(
            wide, text, eda_root=common.EDA_SJD_ROOT,
            eda_spine_path=common.EDA_SJD_CLINICAL_EPISODE_SPINE,
            all_spine_path=root / common.EPISODE_SPINE_PARQUET.relative_to(common.PROJECT_ROOT),
        )
        _refresh_active_outputs(original, wide, cfg, target)
    finally:
        os.close(directory)
    return target


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
        effective = refresh_active(cfg)
    except PermissionError as error:
        raise SystemExit(f"Harmonization stopped because an input or staging path is inaccessible ({error.filename}).") from None
    except FileNotFoundError as error:
        raise SystemExit(f"Harmonization stopped because a required file is missing ({error}).") from None
    except InferenceUnavailable as error:
        raise SystemExit(f"Harmonization stopped before publication ({error}).") from None
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
    print(f"Effective Parquet: {effective.resolve()}")


if __name__ == "__main__":
    main()

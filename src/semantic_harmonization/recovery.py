"""Reuse compatible, sealed extractions and revalidate them against current evidence."""
from __future__ import annotations

from collections import defaultdict

from .report_sources import INPUTS, grain
from .semantic_facts import canonical
from .settings import digest


def incoming_reuse(references, inputs, rows, lines, cfg, resources, knowledge, retrieval):
    """Find validated proposals with identical source, prompt, retrieval, model and lineage."""
    if not rows:
        return {}
    candidates = defaultdict(list)
    current = {row["mapping_id"]: row for row in rows}
    occurrence = defaultdict(list)
    for line in lines:
        if line["mapping_id"] in current:
            occurrence[line["mapping_id"]].append(line)
    identity_fields = ("variable", "concept", "family", "raw_type", "raw_json", "context_json")
    document_fields = ("doc_id", "rank", "snippet", "source", "source_version", "lexical_rank", "dense_rank", "exact")
    historical_cfg = {**cfg, "comparison": {key: value for key, value in cfg["comparison"].items()
                                           if key not in {"recent_v2_run_id", "recent_report_id"}}}
    for reference in references:
        if not reference:
            continue
        manifest = reference.manifest
        prior_identity = manifest.get("identity", {})
        same_profile = (prior_identity.get("inference_profile") == digest(cfg["model"])
                        or prior_identity.get("inference_profile") is None
                        and prior_identity.get("config") == digest(historical_cfg))
        if (any(manifest["inputs"].get(key) != inputs.get(key) for key in INPUTS)
                or manifest.get("prompt", {}).get("sha256") != digest(cfg["prompts"])
                or not same_profile
                or manifest.get("model") != resources["gpt_oss"]):
            continue
        prior_knowledge = reference.json("knowledge_index_manifest.json")
        if not prior_knowledge or any(prior_knowledge.get(key) != knowledge.get(key)
                                      for key in ("corpus", "encoder", "rules", "config")):
            continue
        mappings = {row["mapping_id"]: row for row in reference.rows("unique_values.parquet") or ()
                    if row["mapping_id"] in current}
        old_line, old_trace = defaultdict(list), defaultdict(list)
        for line in reference.rows("row_lineage.parquet") or ():
            if line["mapping_id"] in mappings:
                old_line[line["mapping_id"]].append(line)
        for doc in reference.rows("retrieval_trace.parquet") or ():
            if doc["mapping_id"] in mappings:
                old_trace[doc["mapping_id"]].append(doc)
        for item in reference.rows("llm_extractions.parquet") or ():
            key = item["mapping_id"]
            old = mappings.get(key)
            if (old is None or item["status"] != "validated"
                    or item.get("contribution_status") != "added"
                    or any(old.get(name) != current[key].get(name) for name in identity_fields)
                    or not old_line[key] or grain(old_line[key]) != grain(occurrence[key])):
                continue
            trace = lambda items: [tuple(row.get(field) for field in document_fields) for row in items]
            if trace(old_trace[key]) != trace(retrieval[key]):
                continue
            candidates[key].append((manifest["run_id"], item))
    return candidates


def proposal_meanings(item):
    """Compare proposals without provenance strings or provider-generated IDs."""
    findings = sorted(digest({key: value for key, value in item.items() if key != "document_ids"})
                      for item in item["findings"])
    return findings, sorted(canonical(fact) for fact in item["facts"])


def historical_narratives(references, inputs, rows, lines, segments):
    """Find sealed historical findings with identical occurrence lineage and literal evidence."""
    from .report_progress import legacy, v1_spans
    from .semantic_facts import NARRATIVE_TYPES, from_storage

    allowed_families = {"cryoglobulin", "immunofixation"}
    current = {row["mapping_id"]: row for row in rows if row["family"] in allowed_families}
    if not current:
        return {}
    occurrences, parts, output = defaultdict(list), defaultdict(list), defaultdict(list)
    for line in lines:
        if line["mapping_id"] in current:
            occurrences[line["mapping_id"]].append(line)
    for part in segments:
        if part.mapping_id in current:
            parts[part.mapping_id].append(part)
    identity_fields = ("variable", "concept", "family", "raw_type", "raw_json", "context_json")
    for reference in references:
        if not reference or any(reference.manifest["inputs"].get(key) != inputs.get(key) for key in INPUTS):
            continue
        old_inventory = {r["mapping_id"]: r for r in reference.rows("semantic_dictionary.parquet") or ()
                         if r["mapping_id"] in current and
                         all(r.get(field) == current[r["mapping_id"]][field] for field in identity_fields)}
        old_line = defaultdict(list)
        for line in reference.rows("row_lineage.parquet") or ():
            if line["mapping_id"] in old_inventory:
                old_line[line["mapping_id"]].append(line)
        matched = {key for key in old_line if grain(old_line[key]) == grain(occurrences[key])}
        if not matched:
            continue
        if str(reference.manifest["schema_version"]) == "2.0":
            proposals = (from_storage(r) for r in reference.rows("semantic_facts.parquet") or ()
                         if r["mapping_id"] in matched and r["fact_type"] in NARRATIVE_TYPES)
        else:
            def project():
                for key in sorted(matched):
                    semantic, facts, _ = legacy(old_inventory[key])
                    yield from v1_spans(facts, semantic, parts[key]) or ()
            proposals = project()
        for fact in proposals:
            if fact.fact_type not in NARRATIVE_TYPES:
                continue
            if any(part.source_field == fact.source_field and part.start <= fact.start < fact.end <= part.end
                   and part.text[fact.start-part.start:fact.end-part.start] == fact.quote
                   for part in parts[fact.mapping_id]):
                output[fact.mapping_id].append((reference.manifest["run_id"], fact))
    return output


def reconcile_narratives(row, semantic, existing, proposals, retrieval, rule):
    """Accept only independently reproducible, non-conflicting historical findings."""
    from .inference import apply_extraction, validate_extraction
    from .schemas import Extraction
    from .settings import serialize

    if not proposals:
        return [], {}
    meanings = {canonical(fact) for fact in existing}
    verified = {}
    for run_id, old in proposals:
        meaning = canonical(old)
        if meaning in meanings or meaning in verified:
            continue
        trial = semantic.model_copy(deep=True)
        request = Extraction(findings=[old.payload])
        try:
            checked = validate_extraction(serialize(request.model_dump()), row, retrieval)
            additions = apply_extraction(trial, checked, row, rule)
        except ValueError:
            continue
        if len(additions) == 1:
            fact = additions[0]
            if (canonical(fact) == meaning and (fact.source_field, fact.start, fact.end, fact.quote)
                    == (old.source_field, old.start, old.end, old.quote)):
                verified[meaning] = (run_id, old.payload)
    if not verified:
        return [], {}
    # Merge as a set so an unresolved contradiction never wins by reference order.
    try:
        added = apply_extraction(semantic, Extraction(findings=[value[1] for value in verified.values()]), row, rule)
    except ValueError:
        return [], {}
    return added, {fact.fact_id: verified[canonical(fact)][0] for fact in added}

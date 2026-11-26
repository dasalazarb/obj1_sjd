"""Reconcile comment bodies and analytical links using persisted evidence."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import re

import pyarrow as pa

from .report_sources import unique_counts
from .schemas import INT, STR
from .semantic_facts import METADATA_ROLES, clinical, coverage, fact_spans

COMMENT = pa.schema([
    *[(key, STR) for key in ("mapping_id", "variable", "family", "prefix", "prefix_kind", "prefix_source", "raw",
                            "body", "disposition", "source_field", "role", "segment", "reason")],
    *[(key, INT) for key in ("prefix_start", "prefix_end", "start", "end", "occurrences", "episodes", "patients",
                            "source_events")],
    ("m1_fields", pa.list_(STR)), ("m4_fields", pa.list_(STR)), ("strategies", pa.list_(STR)),
    ("evidence", pa.list_(pa.struct([("start", INT), ("end", INT), ("quote", STR), ("strategy", STR)]))),
    ("residual", pa.list_(pa.struct([("start", INT), ("end", INT), ("text", STR)]))),
])
COMMENT_SUMMARY = pa.schema([
    *[(key, STR) for key in ("variable", "family", "prefix_kind", "disposition")],
    *[(key, INT) for key in ("mappings", "occurrences", "episodes", "patients", "source_events")],
])
MATERIALIZATION = pa.schema([
    *[(key, STR) for key in ("mapping_id", "fact_id", "variable", "family", "fact_type", "status", "reason")],
    ("wide_fields", pa.list_(STR)),
    *[(key, INT) for key in ("expected_observations", "linked_observations", "as_of_observations")],
])


def comment_rows(row, parts, deterministic, final, counts):
    """Audit marker-associated text per source, including observation-linked comments."""
    markers = [part for part in parts if part.role_rule == "comment_marker"]
    if not markers:
        return []
    available = [part for part in parts if part.role_rule not in {"comment_marker", "transport"} and part.text.strip()]
    bodies = {}
    for marker in markers:
        next_start = min((other.start for other in markers if other.source_field == marker.source_field
                          and other.start >= marker.end), default=float("inf"))
        body = [part for part in available if part.source_field == marker.source_field
                and marker.end <= part.start < next_start]
        if not body and marker.source_field == "raw_text":
            body = [part for part in available if part.source_field.startswith("context.")]
        bodies[marker.segment_id] = body
    candidates = {part.segment_id: part for body in bodies.values() for part in body}
    by_segment_m1, by_segment_m4 = defaultdict(list), defaultdict(list)
    for indexes, stage in ((by_segment_m1, deterministic), (by_segment_m4, final)):
        for fact in stage:
            for span in fact_spans(fact):
                indexes[span.segment_id].append(span)
    spans = coverage(candidates.values(), final)
    recovered = any(clinical(fact) or fact.temporality in {"historical", "planned"} and
                    fact.fact_type != "observation_metadata" for part_id in candidates for fact in by_segment_m4[part_id])
    incomplete = any(part.role == "unresolved" or part.role in {"patient_result", "historical_reference"}
                     and not (spans[key][0] and not spans[key][1]) for key, part in candidates.items())
    disposition = ("marker_without_body" if not candidates else
                   "partially_structured" if recovered and incomplete else
                   "fully_structured" if recovered else "ambiguous_body" if incomplete else "metadata_only")
    output = []
    for marker in markers:
        prefix = marker.text.strip()
        kind = re.sub(r"[\s:;.\-–—]+$", "", " ".join(prefix.casefold().split()))
        if kind == "see comments":
            kind = "see comment"
        body = bodies[marker.segment_id]
        # Slice each exact source interval so transport separators and whitespace are preserved.
        sources = defaultdict(list)
        for part in parts:
            sources[part.source_field].append(part.text)
        body_sources = defaultdict(list)
        for part in body:
            body_sources[part.source_field].append(part)
        literal_body = "\n".join("".join(sources[source])[segments[0].start:segments[-1].end]
                                 for source, segments in body_sources.items())
        for part in body or [None]:
            key = part.segment_id if part else None
            m1, m4 = by_segment_m1[key], by_segment_m4[key]
            residual = spans[key][1] if part and part.role not in METADATA_ROLES else []
            output.append(dict(
                mapping_id=row["mapping_id"], variable=row["variable"], family=row["family"],
                prefix=prefix, prefix_kind=kind, prefix_source=marker.source_field, prefix_start=marker.start,
                prefix_end=marker.end, raw=(value if isinstance(value := json.loads(row["raw_json"]), str) else row["raw_json"]),
                body=literal_body, disposition=disposition, source_field=part.source_field if part else None,
                role=part.role if part else None, segment=part.text if part else None,
                start=part.start if part else None, end=part.end if part else None,
                reason="missing_referenced_comment" if not part else
                "unresolved_source_scope" if part.role == "unresolved" else "uncovered_result" if residual else None,
                m1_fields=fields(m1), m4_fields=fields(m4), strategies=sorted({fact.strategy for fact in m4}),
                evidence=[{k: getattr(fact, k) for k in ("start", "end", "quote", "strategy")} for fact in m4],
                residual=residual, **{key: counts[key] for key in ("occurrences", "episodes", "patients", "source_events")}
            ))
    return output


def fields(facts):
    """Identify populated typed fields, retaining their fact namespace."""
    return sorted({f"{fact.fact_type}.{field}" for fact in facts for field, value in fact.payload.model_dump().items()
                   if field not in {"quote", "document_ids"} and value is not None and value != []})


def comment_summary(rows, lineage):
    """Aggregate by variable, family and prefix without counting segment rows as mappings."""
    groups = defaultdict(dict)
    for row in rows:
        for variable in (row["variable"], None):
            groups[variable, row["family"], row["prefix_kind"], row["disposition"]][row["mapping_id"]] = row
    return [
        dict(variable=variable, family=family, prefix_kind=prefix, disposition=disposition,
             mappings=len(mappings), **{key: counts[key] for key in ("episodes", "patients", "source_events", "occurrences")})
        for (variable, family, prefix, disposition), mappings in groups.items()
        for lines in [None if lineage is None else [line for key in mappings for line in lineage.get(key, ())]]
        for counts in [unique_counts(lines) if lineage is not None else dict.fromkeys(
            ("episodes", "patients", "source_events", "occurrences"))]
    ]


def materialization_rows(row, facts, lines, links, features):
    """Account for each clinical fact and its verified observation links."""
    if links is None:
        return []
    expected = [line for line in lines or () if line["patient_id"] is not None and line["clinical_episode_id"] is not None]
    linked = Counter(fact_id for link in links for fact_id in link["fact_ids"])
    as_of = Counter(fact_id for link in links for fact_id in link["as_of_fact_ids"])
    out = []
    for fact in facts:
        if not clinical(fact) and fact.fact_type != "medication_context":
            continue
        n = linked[fact.fact_id]
        names = sorted(feature["name"] for feature in features if wide_field(fact, feature.get("semantic_field")))
        status = ("linked" if n and n == len(expected) else "missing_links" if expected else
                  "source_observation_only")
        out.append(dict(mapping_id=row["mapping_id"], fact_id=fact.fact_id, variable=row["variable"],
                        family=row["family"], fact_type=fact.fact_type, status=status,
                        reason="Source observation has no linked episode" if not expected else
                        "Semantics retained at observation grain" if not names else None, wide_fields=names,
                        expected_observations=len(expected), linked_observations=n, as_of_observations=as_of[fact.fact_id]))
    return out


def wide_field(fact, field):
    if field is None:
        return False
    if field in type(fact.payload).model_fields and getattr(fact.payload, field) is not None:
        return True
    if fact.fact_type == "titer_result":
        return field.startswith("titer_")
    if fact.fact_type == "ana_pattern":
        return field == "pattern_terms" and bool(fact.payload.pattern)
    if fact.fact_type in {"morphology_result", "m_spike_result"}:
        numeric = getattr(fact.payload, "percentage", None) or getattr(fact.payload, "concentration", None)
        return bool(numeric and field in type(numeric).model_fields and getattr(numeric, field) is not None)
    if fact.fact_type == "immunofixation_finding":
        from .application import immunofixation_fields
        return field in immunofixation_fields([fact.payload.model_dump()])
    return fact.fact_type == "qualitative_result" and field == "qualitative_result"

REGRESSION = pa.schema([
    *[(key, STR) for key in ("mapping_id", "variable", "family", "reference_version", "change", "fact_type",
                            "canonical_fact", "reason", "evidence")],
    ("occurrences", INT),
])
HISTORICAL_WORK = pa.schema([
    *[(key, STR) for key in ("work_id", "current_work_id", "mapping_id", "variable", "kind", "state_v1",
                            "state_previous_v2", "state_final_v2", "reconciliation", "evidence")],
])
DERIVED = pa.schema([
    *[(key, STR) for key in ("variable", "producer", "disposition", "rule", "availability", "evidence")],
    ("evaluated_episodes", INT),
])


def clinical_meanings(facts):
    """Exclude process metadata, history, and ambiguous HLA from clinical deltas."""
    return {meaning: row for meaning, row in facts.items()
            if row["fact_type"] not in {"observation_metadata", "medication_context"}
            and row["temporality"] in {"current", "unspecified"}
            and not (row["fact_type"] == "hla_result" and row["hla_result"]["ambiguous"])}


def regression_rows(mapping, final, v1, previous, parts):
    """Expose removals, restorations and the exact evidence for scoped corrections."""
    current = clinical_meanings(final)
    legacy = clinical_meanings(v1) if v1 is not None else None
    prior = clinical_meanings(previous) if previous is not None else None
    records = []
    common = {key: mapping[key] for key in ("mapping_id", "variable", "family", "occurrences")}
    for version, before in (("V1", legacy), ("Previous V2", prior)):
        if before is None:
            continue
        for meaning in sorted(before.keys() - current.keys()):
            original = before[meaning]
            category = original["fact_type"]
            reason = None
            evidence = None
            if category == "titer_result" and mapping["family"] == "hla":
                token = next((part for part in parts if part.role == "patient_result" and
                              re.fullmatch(r"\s*\d{2,3}:\d{2,3}\s*", part.text)), None)
                if token:
                    reason, evidence = "HLA locus grammar identifies a reported allele token", token.text
            records.append(common | dict(reference_version=version, fact_type=category, canonical_fact=meaning,
                                         change="justified_removal" if reason else "unexplained_removal",
                                         reason=reason or "Review supporting evidence and field equivalence",
                                         evidence=evidence))
    if legacy is not None and prior is not None:
        for meaning in sorted((legacy.keys() & current.keys()) - prior.keys()):
            records.append(common | dict(reference_version="Previous V2", change="restored",
                                         fact_type=current[meaning]["fact_type"], canonical_fact=meaning,
                                         reason="Equivalent supported fact is present in V1 and final V2",
                                         evidence=mapping["raw_json"]))
    return records


def historical_crosswalk(frozen, work, mappings, segment_index):
    """Preserve original work IDs and reconcile only when source evidence is comparable."""
    if frozen is None:
        return []
    by_id = {item["work_id"]: item for item in work}
    by_mapping = defaultdict(list)
    for item in work:
        by_mapping[item["mapping_id"]].append(item)
    result = []
    for item in frozen:
        if not item.get("confirmed_initial"):
            continue
        mapping = mappings.get(item["mapping_id"])
        target_id = mapping["mapping_id"] if mapping else None
        current = by_id.get(item["work_id"]) if mapping else None
        if current is None and mapping:
            start, end = item.get("segment_start"), item.get("segment_end")
            candidates = [peer for peer in by_mapping[target_id] if peer["problem_kind"] == item["problem_kind"]
                          and peer["source_field"] == item["source_field"] and start is not None and end is not None
                          and peer["segment_start"] is not None and peer["segment_end"] is not None
                          and max(start, peer["segment_start"]) < min(end, peer["segment_end"])]
            if len(candidates) == 1:
                current = candidates[0]
        evidence = None
        reconciliation = "matched" if current else "unmatched"
        state = current["state_v2"] if current else None
        if not current and mapping:
            for part in segment_index.get(target_id, ()):
                if part.source_field != item.get("source_field") or item.get("segment_start") is None or item.get("segment_end") is None:
                    continue
                if not max(part.start, item["segment_start"]) < min(part.end, item["segment_end"]):
                    continue
                if part.role == "reference_rule":
                    reconciliation, state, evidence = "reclassified_reference_rule", "no_correction_required", part.text
                    break
                if mapping["family"] == "hla" and item["problem_kind"] == "structured_field":
                    if re.fullmatch(r"\s*\d{2,3}:\d{2,3}\s*", part.text):
                        reconciliation, state, evidence = "reclassified_hla_grammar", "no_correction_required", part.text
                        break
        result.append(dict(work_id=item["work_id"], current_work_id=current["work_id"] if current else None,
                           mapping_id=item["mapping_id"], variable=item["variable"], kind=item["problem_kind"],
                           state_v1=item.get("state_v1"), state_previous_v2=item.get("state_v2"),
                           state_final_v2=state, reconciliation=reconciliation, evidence=evidence))
    return result


def derived_rows(derived, variables):
    """Record the published rule or the source limitation without guessing a reconstruction."""
    by_variable = {row["variable"]: row for row in derived}
    producers = {
        "sero": "src/block_A/01_serological_profile.py",
        "essdai": "src/block_A/10_build_integrated_longitudinal_dataset.py",
        "esspri": "src/block_A/10_build_integrated_longitudinal_dataset.py",
        "ext": "src/block_A/10_build_integrated_longitudinal_dataset.py",
        "ovl": "src/block_A/10_build_integrated_longitudinal_dataset.py",
        "pop": "src/block_A/10_build_integrated_longitudinal_dataset.py",
        "pro": "src/block_A/10_build_integrated_longitudinal_dataset.py",
    }
    return [
        dict(variable=name, producer=producers.get(name.split("__")[0]),
             disposition="candidate_reconstructed" if record.get("recomputed") else "source_preserved",
             rule=record.get("rule"), evaluated_episodes=record.get("evaluated_episodes"),
             availability="See recorded reconstruction rule" if record.get("recomputed") else "Unverified",
             evidence="derived_comparison.parquet" if record else "Original Step 10 field verified during publication")
        for name in sorted(by_variable.keys() | {r["variable"] for r in variables if r["family"] == "derived"})
        for record in [by_variable.get(name, {})]
    ]

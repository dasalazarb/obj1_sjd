"""Build typed facts and the compatible scalar projection from segmented evidence."""
from __future__ import annotations

from collections import defaultdict
import json
import re

from .parsers import ALLELE, normalize, parse_segment
from .schemas import (
    AnaPattern, Finding, HlaResult, MorphologyResult, NumericResult, ObservationMetadata,
    OrdinalResult, QualitativeResult, Semantic, SemanticFact, SemanticSegment, SpikeResult, TiterResult,
)
from .segmentation import RESULT_LABEL, segment, sources
from .settings import digest, serialize

VERSION = "semantic-facts-v2"
METADATA_ROLES = {"reference_rule", "assay_method", "specimen_information", "administrative_status",
                  "general_interpretation", "historical_reference"}
NUMERIC_FIELDS = tuple(NumericResult.model_fields)
TITER_FIELDS = ("titer_numerator", "titer_denominator", "titer_comparator")
SCALAR_GROUPS = (NUMERIC_FIELDS, TITER_FIELDS, ("ordinal_grade", "ordinal_score", "descriptor"))
NARRATIVE_TYPES = {"immunofixation_finding", "cryoglobulin_finding"}


def make_fact(row: dict, part: SemanticSegment, kind: str, payload, strategy: str, version: str,
              quote: str | None = None, temporality: str = "current") -> SemanticFact:
    """Use literal, uniquely placed evidence and content-based identifiers."""
    if not quote or part.text.count(quote) != 1:
        quote = part.text
    start = part.start + part.text.index(quote)
    end = start + len(quote)
    identity = [part.mapping_id, kind, payload.model_dump(exclude={"quote", "document_ids"}),
                temporality, part.source_field, start, end]
    return SemanticFact(
        mapping_id=part.mapping_id, fact_id=digest(identity), family=row["family"], concept=row["concept"],
        fact_type=kind, payload=payload, temporality=temporality, segment_id=part.segment_id,
        source_field=part.source_field, start=start, end=end, quote=quote, strategy=strategy,
        rule_or_prompt_version=version, document_ids=payload.document_ids if isinstance(payload, Finding) else [],
        schema_version=VERSION)


def hla_payload(allele) -> HlaResult:
    """Represent alternatives without asserting each as a confirmed allele."""
    token, locus = allele.reported, allele.locus
    alternatives = [item.strip() for item in token.split("/")] if "/" in token else []
    if not alternatives or not all(ALLELE.fullmatch(item.upper()) for item in alternatives):
        alternatives = []
    explicit = ALLELE.fullmatch(token.upper())
    reported_locus = explicit["locus"] if explicit and explicit["locus"] else locus
    kind = ("alternatives" if alternatives else "group" if allele.ipd_match in {"g_group", "p_group"}
            or allele.resolution_fields == 1 else "allele" if allele.allele else "unresolved")
    return HlaResult(locus=reported_locus, token=token, nomenclature=kind, allele=allele.allele,
                     resolution=allele.resolution_fields, ipd_match=allele.ipd_match, suffix=allele.suffix,
                     alternatives=alternatives, ambiguous=kind in {"alternatives", "unresolved"},
                     position=allele.source_order)


def part_facts(row: dict, part: SemanticSegment, parsed: Semantic, version: str) -> list[SemanticFact]:
    """Translate supported assay attributes into typed extension records."""
    family, concept = row["family"], row["concept"]
    facts = []

    def add(kind, payload, key=None, quote=None, temporality="current"):
        ev = next((item for item in parsed.evidence if item.field == key and item.source == part.source_field), None)
        facts.append(make_fact(row, part, kind, payload, ev.strategy if ev else parsed.strategy,
                               ev.rule if ev else version, quote or (ev.quote if ev else None), temporality))

    if part.role in METADATA_ROLES:
        add("observation_metadata", ObservationMetadata(
            role=part.role, specimen_state=parsed.specimen_state, result_availability=parsed.result_availability),
            temporality="historical" if part.role == "historical_reference" else "unspecified")
    numeric = NumericResult(**{key: getattr(parsed, key) for key in NUMERIC_FIELDS}) if parsed.comparator else None
    if numeric and family not in {"m_spike", "cbc_morphology"}:
        add("numeric_result", numeric, "comparator")
    if parsed.qualitative_result or family == "urine_nominal" and parsed.category:
        add("qualitative_result", QualitativeResult(
            category=parsed.qualitative_result or parsed.category, descriptor=parsed.descriptor),
            "qualitative_result" if parsed.qualitative_result else "category")
    if parsed.ordinal_grade or family == "urine_graded" and parsed.descriptor:
        add("ordinal_result", OrdinalResult(grade=parsed.ordinal_grade or parsed.descriptor,
                                            descriptor=parsed.descriptor, score=parsed.ordinal_score), "ordinal_grade")
    if parsed.titer_denominator:
        add("titer_result", TiterResult(numerator=parsed.titer_numerator, denominator=parsed.titer_denominator,
                                        comparator=parsed.titer_comparator), "titer_denominator")
    if family in {"ana", "ana_narrative"}:
        state = parsed.qualitative_result if parsed.qualitative_result in {
            "positive", "negative", "equivocal", "indeterminate", "borderline"} else None
        if parsed.pattern_terms or state and concept.endswith("_pattern"):
            for pattern in parsed.pattern_terms or [None]:
                compartment = "cytoplasmic" if "cytoplasmic" in concept or pattern and (
                    "cytoplasmic" in pattern or "reticular" in pattern) else None
                add("ana_pattern", AnaPattern(compartment=compartment, pattern=pattern, state=state), "pattern_terms")
    if family == "m_spike" and (parsed.category or numeric):
        add("m_spike_result", SpikeResult(detection=parsed.category, concentration=numeric),
            "category" if parsed.category else "comparator", quote=part.text)
    if family in {"urine_microscopy", "cbc_morphology"} and (parsed.category or parsed.descriptor or numeric):
        add("morphology_result", MorphologyResult(entity=concept, descriptor=parsed.category or parsed.descriptor,
                                                  percentage=numeric), "category")
    for item in parsed.findings:
        add(f"{family}_finding", item, "findings", item.quote, item.temporality)
    for item in parsed.alleles:
        if item.reported:
            add("hla_result", hla_payload(item), "alleles", item.reported)
    for item in parsed.medications:
        add("medication_context", item, "medications", item.quote, "unspecified")
    return facts


def clinical(fact: SemanticFact) -> bool:
    """Distinguish biological observations from transport and unresolved tokens."""
    return (fact.fact_type not in {"observation_metadata", "medication_context"}
            and fact.temporality in {"current", "unspecified"}
            and (fact.fact_type != "hla_result" or fact.payload.nomenclature != "unresolved"))


def merge_segments(parts: list[Semantic]) -> Semantic:
    """Project compatible scalar groups and retain conflicting values only as facts."""
    fields = list(Semantic.model_fields)
    result = Semantic()
    conflicts = []
    for group in (*SCALAR_GROUPS, *((field,) for field in fields
                    if field not in {"strategy", "interpretation_note", "evidence", "findings", "alleles", "medications"}
                    and all(field not in group for group in SCALAR_GROUPS))):
        populated = [part for part in parts if any(getattr(part, key) not in (None, []) for key in group)]
        if len({serialize([getattr(part, key) for key in group]) for part in populated}) > 1:
            if group == ("result_availability",):
                substantive = [part for part in populated if part.result_availability != "reported"]
                if len({part.result_availability for part in substantive}) == 1:
                    populated = substantive
                else:
                    conflicts.extend(group)
                    continue
            else:
                conflicts.extend(group)
                continue
        if populated:
            part = populated[0]
            for key in group:
                setattr(result, key, getattr(part, key))
            result.evidence.extend(item for item in part.evidence if item.field in group)
    for field in ("findings", "alleles", "medications"):
        for part in parts:
            for item in getattr(part, field):
                if item not in getattr(result, field):
                    getattr(result, field).append(item)
            result.evidence.extend(item for item in part.evidence if item.field == field)
    result.strategy = next((part.strategy for part in parts if part.strategy != "unchanged"), "unchanged")
    if conflicts:
        result.interpretation_note = "Conflicting assertions retained in typed facts: " + ", ".join(conflicts)
    else:
        result.interpretation_note = next((part.interpretation_note for part in parts if part.interpretation_note), None)
    return Semantic.model_validate(result.model_dump())


def parse_mapping(row: dict, cfg: dict, vocab: dict) -> tuple[Semantic, list[SemanticSegment], list[SemanticFact]]:
    """Run deterministic parsing and retain both typed and legacy representations."""
    raw = json.loads(row["raw_json"])
    parts = segment(row)
    if raw is None:
        return Semantic(strategy="identity", interpretation_note="Source null."), parts, []
    if isinstance(raw, str) and not raw.strip():
        return Semantic(strategy="whitespace_to_null", interpretation_note="Whitespace-only source string retained."), parts, []
    parsed_parts, typed = [], {}
    for part in parts:
        if part.role == "unresolved":
            continue
        clean = RESULT_LABEL.sub("", part.text).strip() if part.role == "patient_result" else part.text
        parsed = parse_segment(row, cfg, vocab, clean, part.source_field, part.role)
        for fact in part_facts(row, part, parsed, cfg["normalization"]["version"]):
            typed[fact.fact_id] = fact
        if row["family"] == "hla" and row["concept"] in {"hla_drb", "hla_dpb"}:
            parsed.alleles = [item for item in parsed.alleles
                              if not ("*" in item.reported and item.reported != "*" and not item.allele)]
            if not parsed.alleles:
                parsed.evidence = [item for item in parsed.evidence if item.field not in {"alleles", "result_availability"}]
                parsed.result_availability = None
        parsed_parts.append(parsed)
    return merge_segments(parsed_parts), parts, list(typed.values())


def canonical(fact: SemanticFact) -> str:
    """Compare meaning independently of citations, versions, offsets and lexical variants."""
    payload = fact.payload.model_dump(exclude={"quote", "document_ids"})
    for field in ("descriptor", "pattern"):
        if isinstance(payload.get(field), str):
            payload[field] = normalize(payload[field])
    return serialize([fact.fact_type, fact.temporality, payload])


def verify_contract(rows: list[dict], segments: list[SemanticSegment], facts: list[SemanticFact]) -> None:
    """Fail publication on broken joins, offsets, duplicates or incompatible roles."""
    lookups = {item.segment_id: item for item in segments}
    if len(lookups) != len(segments) or len({item.fact_id for item in facts}) != len(facts):
        raise ValueError("Duplicate semantic segment or fact")
    by_source = defaultdict(list)
    for item in segments:
        by_source[item.mapping_id, item.source_field].append(item)
    source_count = 0
    for row in rows:
        for field, source in sources(row).items():
            source_count += 1
            offset = 0
            for item in by_source[row["mapping_id"], field]:
                if item.start != offset or item.end <= item.start or source[item.start:item.end] != item.text:
                    raise ValueError("Semantic segment failed source reconstruction")
                offset = item.end
            if offset != len(source):
                raise ValueError("Semantic segment lost source text")
    if source_count != len(by_source):
        raise ValueError("Semantic segment has no source")
    for fact in facts:
        item = lookups[fact.segment_id]
        role_compatible = (item.role in METADATA_ROLES if fact.fact_type == "observation_metadata" else
                           item.role == "patient_result" or item.role == "historical_reference"
                           and fact.temporality == "historical" and fact.fact_type in NARRATIVE_TYPES)
        if (item.mapping_id != fact.mapping_id or item.source_field != fact.source_field
                or not item.start <= fact.start < fact.end <= item.end
                or item.text[fact.start-item.start:fact.end-item.start] != fact.quote or not role_compatible):
            raise ValueError("Semantic fact failed evidence contract")


def from_storage(row: dict) -> SemanticFact:
    """Rehydrate the selected Arrow struct without coercing other arms."""
    from .schemas import FACT_PAYLOADS

    return SemanticFact.model_validate(
        {key: row[key] for key in SemanticFact.model_fields if key != "payload"} |
        {"payload": FACT_PAYLOADS[row["fact_type"]].model_validate(row[row["fact_type"]])})

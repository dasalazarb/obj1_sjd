"""Compare published meanings and reconcile work against final source evidence."""
from collections import Counter, defaultdict
import json
import re

from .compare_runs import baseline_meaning
from .inventory import clean
from .parsers import GRADE, TITER, normalize
from .report_sources import (INPUTS, allowed, feature_transitions, grain, grouped, indexed, inventory_policy, lineage,
                             materialization, same_original, unique_counts)
from .schemas import Semantic, SemanticSegment
from .segmentation import GENERIC_UNCERTAINTY, NARRATIVE, NARRATIVE_SIGNAL, PROCESS, RESULT_LABEL, SPECIMEN, segment, sources
from .semantic_facts import METADATA_ROLES, canonical, clinical, from_storage, part_facts
from .settings import digest, serialize

VERSION = "report-progress-1"
STATES = ("resolved", "partial", "pending", "not_assessable", "no_correction_required")
DEFINITIONS = {
    "lexical": "A result with edge whitespace or a recognized qualitative alias requires a structured result with source evidence.",
    "structured_field": "A reported bound, interval, titer or grade requires its corresponding structured components. Evidence and expected components come from the persisted deterministic stage or the existing field grammar.",
    "result_fragment": "A source-scoped result or explicitly historical narrative requires a supported interpretation. Exact final evidence spans are checked within each source segment.",
    "medication": "Documented concatenated medication text requires linked medication entities. An unexplained remainder in other medication text requires verification.",
    "hla_resolution": "An unreported HLA allele is a source limitation. A supplied allele that cannot be verified requires assessment.",
    "verification": "Unknown scope, missing referenced material or unavailable evidence requires verification outside the confirmed-work denominator.",
    "analytic": "Expected typed observation links are checked against actual candidate rows. Missing links are analytical work outside the confirmed initial semantic denominator.",
    "derived": "Published reconstruction is described at the derived feature grain. Missing rule or availability evidence is shown as a verification requirement.",
}
REASONS = {
    "missing_referenced_comment": ("inventory/metadata", "Retrieve the referenced comment from the source.", "The original contains a comment marker and the mapping has no substantive comment or result."),
    "unsupported_parser_profile": ("parser/registry", "Review assay metadata and configs/concepts.yaml.", "Recorded M1 routing indicates no verified model profile."),
    "residual_clinical_statement": ("segmentation/inference", "Review the exact clause against source evidence and the extraction contract.", "Final facts leave an explicitly scoped clinical clause uncovered."),
    "ambiguous_semantics": ("inventory/metadata", "Check the assay or source note to establish meaning.", "The source does not establish a fully interpretable result or component."),
    "hla_resolution_unavailable": ("inventory/metadata", "Check whether higher resolution was actually reported.", "The HLA source supplies a placeholder or the allele could not be verified."),
    "context_limit": ("inference", "Review the recorded prompt context budget for this source.", "The LLM extraction reached its recorded context limit."),
    "invalid_schema": ("inference", "Review the recorded extraction schema and prompt.", "The LLM extraction failed recorded schema validation."),
    "unsupported_evidence": ("inference", "Review the extraction citations against the source segment.", "The LLM response failed recorded evidence validation."),
    "conflict": ("inference", "Review the recorded conflicting extraction against the source.", "The validated proposal was not incorporated because it conflicted with existing facts."),
    "insufficient_artifacts": ("comparison", "Recover the missing published reporting artifacts.", "Evidence needed to evaluate this work was unavailable."),
    "derived_rule_or_availability_missing": ("derivation", "Check the published derivation rule and source result availability.", "The published artifacts do not demonstrate a reconstruction for this field."),
    "not_materialized": ("materialization", "Review observation linkage in analytic_representation.py.", "Final facts have eligible episode-linked observations but the candidate is missing some expected fact links."),
    "reason_not_recorded": ("unknown", "Review the original and the final evidence for this field.", "No individual cause was recorded."),
}
FAILED_LLM = {"context_limit", "invalid_schema", "unsupported_evidence", "conflict"}
COVERAGE_ROLES = {"patient_result", "historical_reference"}
MEANING_FIELDS = tuple(name for name in Semantic.model_fields if name not in {"strategy", "evidence", "interpretation_note"})
IDENTITY_COLUMNS = ("mapping_id", "variable", "concept", "family", "raw_type", "raw_json", "context_json")
NONCLINICAL_GAPS = re.compile(r"\b(?:and|or|a|an|the|result|current|currently|previously|historical|prior|at|was|is|now|patient)\b|[^\w]", re.I)
CLINICAL_STATEMENT = re.compile(r"\b(?:detected|identified|observed|present|seen|demonstrated|revealed|absent|negative|positive|reactive)\b", re.I)
FINDING_SUFFIX = re.compile(r"^\s*(?:band\s+)?(?:was |is )?(?:detected|identified|observed|seen|present)[.\s]*$", re.I)
BOUND = re.compile(r"^\s*(?:[<>≤≥]=?\s*[+-]?(?:\d|\.\d)|[+-]?\d+(?:\.\d+)?\s*[-–]\s*[+-]?\d)")

def raw_key(row):
    return row["variable"], row["raw_type"], serialize(json.loads(row["raw_json"])), serialize(json.loads(row["context_json"]))


def pairs(old, new, paired):
    """Outer join typed raw/context keys with an unambiguous ID fallback."""
    old_keys, new_keys = defaultdict(list), defaultdict(list)
    for index, keys in ((old, old_keys), (new, new_keys)):
        for row in index.values():
            keys[raw_key(row)].append(row["mapping_id"])
    consumed = set()
    for key, after in new.items():
        signature = raw_key(after)
        match = old.get(key)
        if not paired or match and raw_key(match) != signature:
            match = None
        if paired and match is None and len(old_keys[signature]) == len(new_keys[signature]) == 1:
            possible = old[old_keys[signature][0]]
            if possible["mapping_id"] not in new:
                match = possible
        if match:
            consumed.add(match["mapping_id"])
        yield match, after
    for key in sorted(old.keys() - consumed):
        yield old[key], None


def coverage(parts, facts):
    """Return exact unclaimed Unicode intervals for every eligible segment."""
    by_segment = defaultdict(list)
    for fact in facts:
        if fact.fact_type == "observation_metadata" or fact.fact_type == "hla_result" and fact.payload.ambiguous:
            continue
        by_segment[fact.segment_id].append(fact)
    result = {}
    for part in parts:
        offset, gaps = part.start, []
        supported = [fact for fact in by_segment[part.segment_id]
                 if fact.mapping_id == part.mapping_id and fact.source_field == part.source_field
                 and (fact.temporality == "historical") == (part.role == "historical_reference")
                 and part.start <= fact.start < fact.end <= part.end
                 and part.text[fact.start-part.start:fact.end-part.start] == fact.quote]
        for start, end in sorted((fact.start, fact.end) for fact in supported):
            if start > offset:
                gaps.append((offset, start))
            offset = max(offset, end)
        if offset < part.end:
            gaps.append((offset, part.end))
        remaining = []
        for start, end in gaps:
            text = part.text[start-part.start:end-part.start]
            # A positive narrative fact may cite only its entity span. A simple
            # assertion suffix is already represented by the recorded polarity.
            suffix_accounted = end == part.end and FINDING_SUFFIX.fullmatch(text) and any(
                fact.end == start and getattr(fact.payload, "assertion", None) == "present" for fact in supported)
            if NONCLINICAL_GAPS.sub("", text) and not suffix_accounted:
                remaining.append(dict(start=start, end=end, text=text))
        result[part.segment_id] = (bool(supported), remaining)
    return result


def legacy(row):
    """Adapt recorded V1 fields. Attach no reconstructed V1 source coordinates."""
    if row is None or not any(name in row for name in MEANING_FIELDS):
        return None, None, None
    semantic = Semantic.model_validate({name: row[name] for name in Semantic.model_fields if name in row and
                                       (row[name] is not None or name not in {"findings", "alleles", "medications", "evidence", "pattern_terms", "strategy"})})
    projected = []
    parts = segment(row)
    if parts:
        projected = part_facts(row, parts[0], semantic, "legacy-v1")
    old_clinical = baseline_meaning(dict(row, **semantic.model_dump())) if parts else set()
    return semantic, projected, old_clinical


def v1_spans(facts, dictionary, parts):
    """Use only unambiguous literal V1 citations to assess component coverage."""
    if facts is None:
        return None
    evidence = dictionary.evidence if dictionary else []
    spans = []
    for fact in facts:
        payload_quote = getattr(fact.payload, "quote", None) or getattr(fact.payload, "token", None)
        citations = {(ev.source, payload_quote or ev.quote) for ev in evidence
                     if ev.quote and ev.field != "result_availability" and
                     (payload_quote in ev.quote if payload_quote else ev.quote == fact.quote)}
        locations = {(part.segment_id, part.start + part.text.index(quote), quote)
                     for source, quote in citations for part in parts
                     if source == part.source_field and part.text.count(quote) == 1}
        if len(locations) == 1:
            sid, start, quote = locations.pop()
            part = next(p for p in parts if p.segment_id == sid)
            spans.append(fact.model_copy(update=dict(mapping_id=part.mapping_id, source_field=part.source_field,
                                                     segment_id=sid, start=start, end=start+len(quote), quote=quote)))
    return spans


def component_state(required, facts):
    if facts is None:
        return "not_assessable"
    values = [fact.payload.concentration if fact.fact_type == "m_spike_result" else fact.payload.percentage
              if fact.fact_type == "morphology_result" else fact.payload for fact in facts]
    names = list(required)
    matched = [sum(getattr(value, name, None) is not None and (required[name] is None or
                   str(getattr(value, name)) == str(required[name])) for name in names) for value in values]
    best = max(matched, default=0)
    return "resolved" if best == len(names) else "partial" if best else "pending"


def required_components(part, m1):
    text = part.text.strip()
    kinds = {fact.fact_type: fact.payload for fact in (m1 or ()) if fact.segment_id == part.segment_id}
    if titer := kinds.get("titer_result"):
        return {name: getattr(titer, name) for name in ("numerator", "denominator", "comparator")}
    if titer := TITER.fullmatch(normalize(text)):
        return {"numerator": int(titer["num"]), "denominator": int(titer["den"]), "comparator": None}
    number = kinds.get("numeric_result") or getattr(kinds.get("m_spike_result"), "concentration", None)
    if number and number.comparator != "EQ":
        return {name: getattr(number, name) for name in ("comparator", "limit_value", "range_low", "range_high")
                if getattr(number, name) is not None}
    if BOUND.match(text):
        return {"comparator": None, "range_low": None, "range_high": None} if re.match(r"^\s*\d.*[-–]\s*\d", text) else {
            "comparator": None, "limit_value": None}
    if grade := kinds.get("ordinal_result"):
        return {"grade": grade.grade}
    if GRADE.fullmatch(normalize(text)):
        return {"grade": None}
    return {}


def gap_state(entry):
    return "resolved" if entry[0] and not entry[1] else "partial" if entry[0] else "pending"


def work_for(row, parts, m1, final, old_facts, comparable, cfg, route, outcome, old_available):
    """Define original work independently of M4, then evaluate each version."""
    raw = json.loads(row["raw_json"])
    old_coverage = coverage(parts or (), old_facts or ())
    m1_coverage = coverage(parts or (), m1 or ())
    final_coverage = coverage(parts or (), final or ())
    work = []
    notes = set()

    def add(kind, component, part=None, state="not_assessable", before="not_assessable", m1_state="not_assessable",
            confirmed=True, reason="reason_not_recorded", gaps=None, required=None):
        if state == "resolved":
            reason = None
        elif state in {"pending", "partial"}:
            if outcome in FAILED_LLM:
                reason = outcome
            elif route.get("reason") == "no_verified_model_profile":
                reason = "unsupported_parser_profile"
        module, action, detail = REASONS.get(reason, (None, None, None))
        work.append(dict(work_id=digest([row["mapping_id"], kind, component]), problem_kind=kind, component=component,
                         confirmed_initial=confirmed, state_original="pending" if confirmed else "not_assessable",
                         state_v1=before if comparable else None, state_m1=m1_state, state_v2=state,
                         source_field=part.source_field if part else None, segment_start=part.start if part else None,
                         segment_end=part.end if part else None, remaining_fragments=gaps or [],
                         remaining_fragment=" … ".join(gap["text"] for gap in gaps) if gaps else None,
                         role=part.role if part else None, role_rule=part.role_rule if part else None,
                         reason_code=reason, reason_detail=detail, required_component=required or kind,
                         next_action=action, responsible_module=module, llm_outcome=outcome or "not recorded",
                         work_type="verification" if not confirmed else "semantic",
                         evidence_source="published source segment" if part else "published mapping"))

    if row["family"] in {"derived", "other"}:
        if row["family"] == "derived":
            notes.add("derived source; see published reconstruction")
        elif raw is not None and raw != "":
            add("verification", "outside_clinical_profiles", confirmed=False, reason="ambiguous_semantics")
        return work, notes
    if parts is None:
        add("verification", "missing_segments", confirmed=False, reason="insufficient_artifacts")
        return work, notes
    if not parts and (raw is None or raw == "" or type(raw) is bool):
        notes.add("no source result" if raw is None or raw == "" else "typed source preserved")
    result_parts = [p for p in parts if p.role == "patient_result" or
                    p.role == "historical_reference" and row["family"] in {"cryoglobulin", "immunofixation"}
                    and NARRATIVE_SIGNAL.search(p.text)]
    texts = sources(row)
    marker = next((p for p in parts if p.role_rule == "comment_marker"), None)
    if marker and not result_parts and len(texts) == 1 and not any(
            p.role == "unresolved" and p.role_rule != "transport" and p.text.strip() for p in parts):
        add("verification", "referenced_comment", marker, confirmed=False, reason="missing_referenced_comment",
            gaps=[dict(start=marker.start, end=marker.end, text=marker.text)])
    for part in parts:
        text = part.text.strip()
        if not text or part.role_rule == "transport" or part == marker:
            continue
        component = f"{part.source_field}:{part.start}:{part.end}"
        if PROCESS.fullmatch(text) or SPECIMEN.fullmatch(text):
            notes.add("legitimate missing / administrative only")
            continue
        if part not in result_parts:
            if part.role in METADATA_ROLES:
                notes.add("metadata / reference only")
            elif part.role == "unresolved":
                add("verification", component, part, confirmed=False, reason="ambiguous_semantics",
                    gaps=[dict(start=part.start, end=part.end, text=part.text)])
            continue
        if row["family"] == "hla" and text == "*":
            notes.add("hla_resolution_unavailable")
            continue
        present, gaps = final_coverage[part.segment_id]
        state = gap_state((present, gaps)) if final is not None else "not_assessable"
        before = gap_state(old_coverage[part.segment_id]) if old_facts is not None and old_available else "not_assessable"
        # A legacy result with a meaning but no unique citation cannot be localized.
        if old_available and not old_coverage[part.segment_id][0] and old_facts:
            before = "not_assessable"
        m1_state = gap_state(m1_coverage[part.segment_id]) if m1 is not None else "not_assessable"
        legacy_in_part = [fact for fact in old_facts or () if fact.segment_id == part.segment_id]
        typed_in_part = [fact for fact in final or () if fact.segment_id == part.segment_id]
        m1_in_part = [fact for fact in m1 or () if fact.segment_id == part.segment_id]
        actual = text.casefold().rstrip(".")
        qual = cfg["normalization"]["qualitative"].get(normalize(text).rstrip("."))
        lex = part.source_field == "raw_text" and isinstance(raw, str) and (raw != raw.strip() or qual and actual != qual)
        required = required_components(part, m1)
        if lex:
            add("lexical", component, part, "resolved" if present else state, "resolved" if legacy_in_part else before,
                "resolved" if m1_in_part else m1_state, gaps=gaps if not present else None)
        if required:
            typed_state = component_state(required, typed_in_part if final is not None else None)
            add("structured_field", component, part, typed_state,
                component_state(required, legacy_in_part) if legacy_in_part else before,
                component_state(required, m1_in_part if m1 is not None else None), gaps=gaps,
                required=", ".join(required))
        narrative = (row["family"] in NARRATIVE and NARRATIVE_SIGNAL.search(text) and not GENERIC_UNCERTAINTY.search(text)
                     and (CLINICAL_STATEMENT.search(text) or RESULT_LABEL.match(text)
                          or any(f.fact_type.endswith("_finding") for f in m1_in_part)))
        medication = row["family"] == "medication" and any(delimiter in text for delimiter in (";", "|"))
        if narrative or medication:
            if gaps and not any(CLINICAL_STATEMENT.search(g["text"]) or NARRATIVE_SIGNAL.search(g["text"]) for g in gaps) and not medication:
                state = "not_assessable" if present else state
            reason = "residual_clinical_statement" if state in {"pending", "partial"} else "ambiguous_semantics"
            add("medication" if medication else "result_fragment", component, part, state, before, m1_state,
                reason=reason if final is not None else "insufficient_artifacts", gaps=gaps)
        elif not (lex or required) and (not present or gaps):
            hla = row["family"] == "hla"
            add("hla_resolution" if hla else "verification", component, part, confirmed=False,
                reason="hla_resolution_unavailable" if hla else "ambiguous_semantics", gaps=gaps)
    return work, notes


def typed(facts):
    return [from_storage(row) for row in facts] if facts is not None else None


def storage(facts):
    return [fact.storage() for fact in facts] if facts is not None else None


def canonical_storage(facts):
    entries = {}
    for fact in facts or ():
        meaning = canonical(fact)
        kind, when, payload = json.loads(meaning)
        entries[meaning] = {"fact_type": kind, "temporality": when, kind: payload}
    return entries


def nonempty(row):
    raw = json.loads(row["raw_json"])
    return raw is not None and (not isinstance(raw, str) or bool(raw.strip()))


def mapping_inventory(run, cfg, policy):
    dictionary = run.rows("semantic_dictionary.parquet")
    values = run.rows("unique_values.parquet")
    valid = lambda row: allowed(row["variable"], cfg) and (policy or {}).get(row["variable"], "eligible") == "eligible"
    semantics = indexed([row for row in dictionary or () if valid(row)])
    inventory = indexed([{key: row.get(key) for key in IDENTITY_COLUMNS} | {
        key: row.get(key) for key in ("episodes", "patients", "observations", "baseline", "laboratory_source_events")}
                     for row in (values if values is not None else dictionary or ()) if valid(row)])
    for key in inventory.keys() & semantics.keys():
        if raw_key(inventory[key]) != raw_key(semantics[key]):
            raise ValueError("Dictionary mapping disagrees with the typed inventory")
    return inventory, semantics


def aggregate(states):
    counts = Counter(states)
    if counts["resolved"] or counts["partial"]:
        if counts["pending"] or counts["partial"]:
            return "partial"
        return "not_assessable" if counts["not_assessable"] else "resolved"
    if counts["pending"]:
        return "pending"
    return "not_assessable" if counts["not_assessable"] else "no_correction_required"


def percent(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                percent=round(100*numerator/denominator, 2) if denominator else None)


def summarize(mappings, work, new_lines, new_policy, names, scope):
    """Use the same work units for variable, family and global totals."""
    by_var, by_work = defaultdict(list), defaultdict(list)
    for row in mappings:
        by_var[row["variable"]].append(row)
    for item in work:
        by_work[item["variable"]].append(item)
    variables, families = [], defaultdict(list)
    for name, family in sorted(names.items()):
        rows = by_var[name]
        items = [r for r in by_work[name] if r["v2_available"]]
        confirmed = [r for r in items if r["confirmed_initial"]]
        remaining = [r for r in items if r["state_v2"] != "resolved"]
        paired = [r for r in confirmed if r["comparable"]]
        affected = {r["mapping_id"] for r in remaining if r["mapping_id"] is not None}
        selected = [r for r in rows if r["v2_available"]]
        baseline = [r for r in rows if r["v1_available"] and r["old_nonempty"]]
        before = [r for r in rows if r["comparable"] and r["old_nonempty"] and r["nonempty"]
                  and r["clinical_v1"] is not None and r["clinical_v2"] is not None]
        assessed = [r for r in selected if r["nonempty"] and r["clinical_v2"] is not None]
        if new_lines is not None and all(new_lines.get(key) for key in affected):
            count = unique_counts([line for key in affected for line in new_lines[key]])
        else:
            count = unique_counts(None)
        var = dict(variable=name, family=family, mappings=len(selected),
                   distinct=len({(r["raw_type"], r["raw_json"]) for r in selected}),
                   state=aggregate(r["state_v2"] for r in items) if selected or items else "not_assessable",
                   initial_work=len(confirmed) if selected or items else None,
                   resolved_v1=sum(r["state_v1"] == "resolved" for r in paired) if paired else None,
                   resolved_v2=sum(r["state_v2"] == "resolved" for r in confirmed),
                   remaining=sum(r["state_v2"] in {"pending", "partial"} for r in items),
                   not_assessable=sum(r["state_v2"] == "not_assessable" for r in items),
                   affected_episodes=count["episodes"], affected_patients=count["patients"],
                   affected_source_events=count["source_events"], affected_occurrences=count["occurrences"],
                   coverage_v2=percent(sum(r["clinical_v2"] for r in assessed), len(assessed)),
                   coverage_v1=percent(sum(r["clinical_v1"] is True for r in baseline), len([r for r in baseline if r["clinical_v1"] is not None])),
                   paired_coverage={f"v{version}": percent(sum(r[f"clinical_v{version}"] for r in before), len(before))
                                    for version in (1,2)} if scope == "paired" else None,
                   strategies=sorted({s for r in selected for s in r["v2_fact_strategies"]} |
                                     {r["v2_strategy"] for r in selected if r["v2_strategy"]}),
                   changed=any(r["change_kind"] in {"added", "removed", "added_and_removed", "representation_only"} for r in rows),
                   unchanged=bool(rows) and all(r["change_kind"] == "unchanged" for r in rows),
                   materialization={}, causes=sorted({r["reason_code"] for r in remaining if r["reason_code"]}))
        variables.append(var)
        families[family].append(var)
    def tally(rows, key):
        counter = Counter(row[key] for row in rows)
        return {state: counter[state] for state in STATES}
    family_rows = [dict(family=family, variables=len(rows), **tally(rows, "state"),
                        coverage_v2=percent(sum(r["coverage_v2"]["numerator"] for r in rows),
                                            sum(r["coverage_v2"]["denominator"] for r in rows)))
                   for family, rows in sorted(families.items())]
    current = [r for r in work if r["v2_available"]]
    confirmed = [r for r in current if r["confirmed_initial"]]
    paired = [r for r in confirmed if r["comparable"]]
    unresolved = [r for r in paired if r["state_v1"] in {"partial", "pending"}]
    affected = {r["mapping_id"] for r in current if r["state_v2"] != "resolved" and r["mapping_id"] is not None}
    totals = unique_counts([line for key in affected for line in new_lines[key]]) if new_lines is not None and all(
        new_lines.get(key) for key in affected) else unique_counts(None)
    causes = defaultdict(list)
    for row in current:
        if row["reason_code"]:
            causes[row["reason_code"]].append(row)
    summary = dict(variables=len(variables), inventory_total=len(new_policy) if new_policy is not None else None,
                   protected=sum(p != "eligible" for p in new_policy.values()) if new_policy is not None else None,
                   initial_variables=len({r["variable"] for r in confirmed}), variable_states=tally(variables, "state"),
                   work_states=tally(current, "state_v2"), historical={name: tally(paired, f"state_{name}")
                     for name in ("original", "v1", "v2")} if scope == "paired" else None,
                   historical_denominator=len(paired) if scope == "paired" else None,
                   initial_work=len(confirmed),
                   additional_work_outside_initial=sum(not r["confirmed_initial"] and r["state_v2"] in {"pending", "partial"} for r in current),
                   additional_resolutions=sum(r["state_v2"] == "resolved" for r in unresolved) if scope == "paired" else None,
                   comparable_mappings=sum(r["comparable"] for r in mappings),
                   noncomparable_mappings=sum(not r["comparable"] for r in mappings),
                   affected=totals, causes=[dict(reason=k, work=len(v), mappings=len({r["mapping_id"] for r in v if r["mapping_id"]}))
                                            for k,v in sorted(causes.items())],
                   cumulative=percent(sum(r["state_v2"] == "resolved" for r in paired), len(paired)) if scope == "paired" else None,
                   incremental=percent(sum(r["state_v2"] == "resolved" for r in unresolved), len(unresolved)) if scope == "paired" else None,
                   reopenings=sum(r["state_v1"] == "resolved" and r["state_v2"] in {"pending", "partial"} for r in paired) if scope == "paired" else None)
    return variables, family_rows, summary


def compare_runs(target, baseline, cfg, *, routes=None):
    """Build the shared report model from staged or verified published runs."""
    inventory_v2 = target.rows("variable_inventory.csv")
    inventory_v1 = baseline.rows("variable_inventory.csv") if baseline else None
    policy_v2, policy_v1 = inventory_policy(inventory_v2), inventory_policy(inventory_v1)
    new, dictionary_v2 = mapping_inventory(target, cfg, policy_v2)
    old, dictionary_v1 = mapping_inventory(baseline, cfg, policy_v1) if baseline else ({}, {})
    new_lines, old_lines = lineage(target, new), lineage(baseline, old) if baseline else None
    final = grouped(target.rows("semantic_facts.parquet"), new)
    m1 = grouped(target.rows("deterministic_facts.parquet"), new)
    old_typed = grouped(baseline.rows("semantic_facts.parquet"), old) if baseline and str(
        baseline.manifest["schema_version"]).startswith("2") else None
    parts = grouped(target.rows("semantic_segments.parquet"), new)
    if routes is None:
        routes = {r["mapping_id"]: r for r in target.rows("mapping_routing.parquet") or ()}
    extractions = {r["mapping_id"]: r for r in target.rows("llm_extractions.parquet", ("mapping_id", "status"),
                                                         optional_columns=("contribution_status",)) or ()}
    retrieval = grouped(target.rows("retrieval_trace.parquet", ("mapping_id", "doc_id"),
                                   optional_columns=("source", "source_version", "rank")), new)
    new_features, current_features, new_candidate, new_links, new_wide_known = materialization(
        target, new_lines, {k: r["variable"] for k, r in new.items()}, cfg)
    if baseline:
        old_features, previous_features, old_candidate, _, old_wide_known = materialization(
            baseline, old_lines, {k: r["variable"] for k, r in old.items()}, cfg)
    else:
        old_features, previous_features, old_candidate, old_wide_known = {}, [], None, False
    scope, reason, originals = "original_v2", "Published V1 unavailable", None
    if baseline:
        scope, reason = "separate_scenarios", "Clinical input fingerprints differ"
        if all(baseline.manifest["inputs"].get(key) == target.manifest["inputs"].get(key) for key in INPUTS):
            scope, reason = "separate_scenarios", "Inventory policy differs or is unavailable"
            if policy_v1 is not None and policy_v2 is not None and all(
                    policy_v1[key] == policy_v2[key] for key in policy_v1.keys() & policy_v2.keys()):
                scope, reason = "paired", "Identical source fingerprints and shared inventory policy; each mapping requires equivalent lineage"
                originals = same_original(old_candidate, new_candidate)
                if originals is False:
                    scope, reason = "separate_scenarios", "Candidate keys or preserved source values differ"
    feature_changes = feature_transitions(old_candidate, new_candidate, previous_features, current_features) if (
        scope == "paired" and originals and old_wide_known and new_wide_known) else []
    maps, work = [], []
    stage_routes = []
    strategy_counts = defaultdict(Counter)
    for before, after in pairs(old, new, scope == "paired"):
        row = after or before
        key = row["mapping_id"]
        prior_key = before["mapping_id"] if before else None
        this_line = (new_lines or {}).get(key) if after else (old_lines or {}).get(prior_key)
        prior_line = (old_lines or {}).get(prior_key)
        comparable = bool(before and after and this_line and prior_line and grain(this_line) == grain(prior_line))
        comparison_reason = ("Equivalent typed value, context and occurrence lineage" if comparable else
                             "Occurrence lineage unavailable or different" if before and after else
                             "No matching mapping in the other version" if scope == "paired" else reason)
        old_sem, old_projected, old_clinical = legacy(dictionary_v1.get(prior_key))
        if before and old_typed is not None:
            old_projected = typed(old_typed.get(prior_key, []))
            old_clinical = {canonical(f) for f in old_projected if clinical(f)}
        current_sem = dictionary_v2.get(key) if after else None
        typed_final = typed(final.get(key, [])) if after and final is not None else None
        typed_m1 = typed(m1.get(key, [])) if after and m1 is not None else None
        segments = [SemanticSegment.model_validate(p) for p in parts.get(key, ())] if after and parts is not None else None
        previous, current, deterministic = map(canonical_storage, (old_projected, typed_final, typed_m1))
        changes_assessable = comparable and old_projected is not None and typed_final is not None
        removed = sorted(previous.keys() - current.keys()) if changes_assessable else []
        added = sorted(current.keys() - previous.keys()) if changes_assessable else []
        retained = sorted(current.keys() & previous.keys()) if changes_assessable else []
        legacy_before = old_sem.model_dump() if old_sem else None
        legacy_after = {k: current_sem.get(k) for k in Semantic.model_fields} if current_sem else None
        representation_changed = legacy_before is not None and legacy_after is not None and any(
            legacy_before[k] != legacy_after[k] for k in MEANING_FIELDS)
        change = ("not_comparable" if not changes_assessable else "added_and_removed" if added and removed else
                  "added" if added else "removed" if removed else "representation_only" if representation_changed else "unchanged")
        counts = unique_counts(this_line)
        for field, alias in (("episodes","episodes"),("patients","patients"),("occurrences","observations"),
                             ("source_events","laboratory_source_events"),("baseline_episodes","baseline")):
            if counts[field] is None:
                counts[field] = row.get(alias)
        extraction = extractions.get(key, {}) if after else {}
        outcome = extraction.get("contribution_status") or extraction.get("status")
        route = routes.get(key, {}) if after else {}
        prior_spans = (old_projected if old_typed is not None and prior_key == key else
                       v1_spans(old_projected, old_sem, segments or ()))
        items, notes = work_for(row, segments, typed_m1, typed_final, prior_spans, comparable, cfg, route, outcome,
                                old_projected is not None) if after else ([], set())
        v1_features = old_features.get(prior_key, []) if before else []
        v2_features = new_features.get(key, []) if after else []
        actual = ("written" if v2_features else "candidate feature not observed for this mapping"
                  if new_candidate and this_line and new_wide_known and new_links is not None else "unavailable")
        if after and typed_final and new_links is not None and this_line and row["family"] not in {"derived", "other"}:
            expected = sum(line["patient_id"] is not None and line["clinical_episode_id"] is not None for line in this_line)
            fact_ids = {fact.fact_id for fact in typed_final}
            linked = sum(fact_ids <= set(link["fact_ids"]) for link in new_links.get(key, ()))
            if linked < expected:
                code = "not_materialized"
                module, action, detail = REASONS[code]
                items.append(dict(work_id=digest([key, "analytic", "typed links"]), problem_kind="analytic",
                                  component="typed links", confirmed_initial=False, state_original="not_assessable",
                                  state_v1=None, state_m1=None, state_v2="pending", reason_code=code, reason_detail=detail,
                                  required_component=f"Typed facts linked on episode observations ({linked}/{expected})",
                                  next_action=action, responsible_module=module, work_type="analytic",
                                  evidence_source="patient_visit_semantic_observations.parquet"))
                actual = "partially written" if v2_features else "typed links missing"
        result = row | counts | dict(
            comparison_scope=scope, v1_available=before is not None, v2_available=after is not None,
            comparable=comparable, comparison_reason=comparison_reason, baseline_mapping_id=prior_key,
            v1_strategy=old_sem.strategy if old_sem else None, v2_strategy=current_sem.get("strategy") if current_sem else None,
            v1_legacy_fields=legacy_before, v2_legacy_fields=legacy_after,
            v1_canonical_facts=list(previous.values()) if old_projected is not None else None,
            v2_typed_facts=storage(typed_final), m1_canonical_facts=list(deterministic.values()) if typed_m1 is not None else None,
            v2_fact_strategies=sorted({f.strategy for f in typed_final or ()}), facts_added=[current[k] for k in added] if changes_assessable else None,
            facts_removed=[previous[k] for k in removed] if changes_assessable else None,
            facts_retained=[current[k] for k in retained] if changes_assessable else None,
            change_kind=change, actual_application=actual, materialized_features=v2_features,
            baseline_features=v1_features, clinical_v1=bool(old_clinical) if old_clinical is not None else None,
            clinical_v2=any(clinical(f) for f in typed_final) if typed_final is not None else None,
            nonempty=bool(after and nonempty(after)), old_nonempty=bool(before and nonempty(before)),
            initial_disposition=aggregate(r["state_original"] for r in items) if after else None,
            final_disposition=aggregate(r["state_v2"] for r in items) if after else None,
            pending_reason_codes=sorted({item["reason_code"] for item in items if item["reason_code"]}),
            disposition_notes=sorted(notes), evidence=legacy_after["evidence"] if legacy_after else [],
            knowledge_document_ids=sorted({doc for f in typed_final or () for doc in f.document_ids} |
                                          set((current_sem or {}).get("knowledge_doc_ids") or ())),
            retrieval=(retrieval or {}).get(key, []) if after else [], llm_outcome=outcome or "not recorded",
            m1_m4_added=len(current.keys()-deterministic.keys()) if typed_m1 is not None and typed_final is not None else None,
            m1_m4_removed=len(deterministic.keys()-current.keys()) if typed_m1 is not None and typed_final is not None else None)
        if result["v2_available"]:
            methods = result["v2_fact_strategies"]
            strategy = "mixed" if len(methods) > 1 else methods[0] if methods else result["v2_strategy"] or "not recorded"
            strategy_counts[strategy].update(mappings=1, occurrences=counts["occurrences"] or 0,
                                              facts=len(current), resolved_work=sum(r["state_v2"] == "resolved" for r in items))
            stage_routes.append(dict(mapping_id=key, **{name: route.get(name) for name in (
                "profile", "reason", "send", "relevant_segments", "resolved_segments", "residual_segments")},
                                     final_disposition=result["final_disposition"], pending_reason_codes=result["pending_reason_codes"],
                                     extraction_status=extraction.get("status"), contribution_status=extraction.get("contribution_status")))
        for item in items:
            work.append(item | {k: result[k] for k in ("mapping_id", "variable", "family", "raw_type", "raw_json",
                                                        "v2_available", "comparable")} |
                        dict(context_summary=row["context_json"], strategy_attempted=route.get("reason") or result["v2_strategy"],
                             current_structured_result=storage(typed_final), document_ids=result["knowledge_document_ids"],
                             episode_occurrences=counts["episodes"], source_event_occurrences=counts["source_events"],
                             affected_patients=counts["patients"], occurrences=counts["occurrences"]))
        maps.append(result)
    names = {r["variable"]: r.get("family") or "other" for r in inventory_v2 or ()
             if r["policy"] == "eligible" and allowed(r["variable"], cfg)}
    names.update((r["variable"], r["family"]) for r in new.values())
    derived = [r for r in target.rows("derived_comparison.parquet") or () if allowed(r["variable"], cfg)]
    for entry in derived:
        name = entry["variable"]
        names.setdefault(name, "derived")
        if entry.get("recomputed") and entry.get("rule"):
            continue
        reason_code = "derived_rule_or_availability_missing"
        module, action, detail = REASONS[reason_code]
        work.append(dict(work_id=digest([name, "derived", "reconstruction"]), mapping_id=None, variable=name,
                         family=names[name], problem_kind="derived", component="reconstruction",
                         v2_available=True, comparable=False, state_original="not_assessable",
                         state_v1=None, state_m1=None, state_v2="not_assessable", confirmed_initial=False,
                         reason_code=reason_code, reason_detail=detail, required_component="published derivation",
                         next_action=action, responsible_module=module, work_type="derived",
                         evidence_source="derived_comparison.parquet"))
    variables, families, summary = summarize(maps, work, new_lines, policy_v2, names, scope)
    summary.update(scope=scope, comparison_reason=reason, original_columns_verified=originals,
                   baseline_projection=("typed" if old_typed is not None else "legacy" if baseline else None),
                   m4_over_m1=dict(mappings=sum(bool(r["m1_m4_added"]) for r in maps),
                                   facts_added=sum(r["m1_m4_added"] or 0 for r in maps),
                                   facts_removed=sum(r["m1_m4_removed"] or 0 for r in maps)) if m1 is not None and final is not None else None)
    for var in variables:
        name = var["variable"]
        var["materialization"] = {key: [r for r in rows if r["source_variable"] == name] for key, rows in
                                  (("v1", previous_features), ("v2", current_features))}
        feature_names = {r["feature_name"] for rows in var["materialization"].values() for r in rows}
        var["candidate_transitions"] = [r for r in feature_changes if r["feature_name"] in feature_names]
    downstream = [r for r in target.rows("downstream_impact.csv") or ()
                  if not r.get("feature") or allowed(r["feature"].removeprefix("harm__"), cfg)]
    validation = target.json("validation_summary.json") or {}
    resources = (target.json("resources_manifest.json") or {}).get("resources", {})
    legacy_counts = {name: [r for r in target.rows(f"{name}.csv") or ()
                            if not r.get("variable") or allowed(r["variable"], cfg)]
                     for name in ("semantic_coverage_by_family", "extraction_routing_summary")}
    provenance = {key: {k: run.manifest.get(k) for k in ("run_id", "started_at", "completed_at", "commit", "schema_version",
                                                         "inputs", "model", "prompt")} if run else None
                  for key, run in (("v1", baseline), ("v2", target))}
    return clean(dict(mappings=maps, work=work, variables=variables, families=families, summary=summary, routes=stage_routes,
                      strategies=[dict(strategy=k, **v) for k,v in sorted(strategy_counts.items())], derived=derived,
                      downstream=downstream, validation=validation.get("clinical_evaluation"), resources={
                          name: {k: r.get(k) for k in ("status", "version", "revision", "sha256", "license")}
                          for name,r in resources.items()}, provenance=provenance, legacy=legacy_counts,
                      missing={"v1": sorted(baseline.missing) if baseline else None, "v2": sorted(target.missing)},
                      rule_version=VERSION, definitions=DEFINITIONS, page_size=cfg["report"]["page_size"]))

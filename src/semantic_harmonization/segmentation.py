"""Partition literal evidence before applying any clinical parser."""
from __future__ import annotations

import json
import re

from .schemas import SemanticSegment
from .settings import digest

VERSION = "semantic-segmentation-v2.1"
ADMIN = re.compile(r"(?:see\s+(?:note|below|comments?)\b|note(?=\s*:))\s*[:.\-–—]?\s*", re.I)
HEADINGS = re.compile(
    r"(?<![\w/])(?:(?P<reference>reference\s+(?:range|values?|intervals?)|interpretive criteria|cutoff values?)"
    r"|(?P<method>method(?:ology)?(?=\s*:)|assay (?:method|protocol)|additional information|test information|"
    r"test performed (?:at|by)|measures only|signed by|laboratory address)"
    r"|(?P<general>clinical associations?|clinical significance|may be associated with|can be seen in)"
    r"|(?P<history>previously|historical result|prior result)"
    r"|(?P<current>(?:current|patient)\s+result|currently|now))(?=\s|:|$)", re.I)
CUTOFF = re.compile(
    r"^\s*(?:(?:negative|positive|equivocal|borderline|reactive|nonreactive)\s*:?\s*[<>≤≥=]\s*\d+(?![\d:]|\s*:)"
    r"|(?:[<>≤≥]=?\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?)"
    r"(?:\s+(?:units?|U/mL|IU/mL))?\s+(?:(?:weak|strong|moderately)\s+)?"
    r"(?:negative|positive|equivocal|borderline|reactive|nonreactive)\b)", re.I)
CLAUSE_MARKER = re.compile(
    r"(?<=[.!?])\s+(?=see\s+(?:note|below|comments?)\b|note\s*:|(?:current\s+)?(?:patient\s+)?result\s*:)", re.I)
CRYO_CLAUSE = re.compile(
    r"(?<=[.!?])\s+(?=(?:preliminary\s+|final\s+)?(?:negative|positive)\s+at\b|"
    r"(?:will be\s+|to be\s+)?(?:reviewed again|recheck|repeat)\s+(?:at|in)\b)", re.I)
CRYO_READING = re.compile(
    r"^(?:(?:preliminary|final)\s+(?:result\s*:\s*)?)?(?:negative|positive)\s+at\b|"
    r"^(?P<planned>(?:will be\s+|to be\s+)?(?:reviewed again|recheck|repeat)\s+(?:at|in)\s+\d)", re.I)

RESULT_FORM = re.compile(
    r"^(?:(?:<=|>=|[<>≤≥=])\s*)?[+-]?(?:\d|\.\d)|"
    r"^(?:positive|negative|equivocal|indeterminate|borderline|reactive|non[- ]?reactive|"
    r"repeat(?:ed)? reactive|normal|trace|pos|neg|non[- ]hemolyzed|detected|not detected)\b", re.I)
NARRATIVE_SIGNAL = re.compile(
    r"\b(?:igg|igm|iga|igd|ige|kappa|lambda|immunoglobulins?|cryoglobulins?|cryoprecipitate|"
    r"observed pattern|incubat\w*|recheck|reviewed again|monoclonal band|detected|"
    r"morphology|anisocytosis|reticular|speckled|homogeneous|cytoplasmic)\b", re.I)
GENERIC_UNCERTAINTY = re.compile(r"\b(?:if|such as|for example|recommend|may occur|would|should|expected|will be|control)\b", re.I)
DESCRIPTIVE = {"ana", "urine_nominal", "urine_microscopy", "urine_graded", "cbc_morphology"}
PROCESS = re.compile(
    r"(?:not (?:done|performed|processed)|test not performed|(?:sample|specimen) (?:was )?not processed|"
    r"cancelled|pending|done|not applicable|unable to interpret|(?:sent|referred) to (?:an? )?(?:external|reference) (?:lab|laboratory))\.?$", re.I)
SPECIMEN = re.compile(r"(?:(?:specimen )?ha?emolyzed|clotted|specimen rejected|quantity insufficient|qns)\.?$", re.I)
NARRATIVE = {"immunofixation", "cryoglobulin", "ana_narrative", "cbc_morphology"}
RESULT_LABEL = re.compile(r"^\s*(?:current\s+)?(?:patient\s+)?result\s*:\s*", re.I)
HLA_LABEL = re.compile(r"^\s*HLA\s+(?:typing|result)\s*:\s*", re.I)
HLA_TOKEN = re.compile(r"(?:[A-Z][A-Z0-9]*\*)?\d{2,3}(?::\d{2,3}){0,3}[GPNLSQCA]?|\*", re.I)


def sources(row: dict) -> dict[str, str]:
    """Return text sources separately, preserving the numeric scalar spelling."""
    raw = json.loads(row["raw_json"])
    context = json.loads(row["context_json"])
    values = {"raw_text": raw}
    if type(raw) in {int, float}:
        values["raw_text"] = str(raw)
    values.update((f"context.{field}", context.get(field))
                  for field in ("observation_comment", "observation_note"))
    return {key: value for key, value in values.items() if isinstance(value, str) and value}


def boundaries(text: str, family: str) -> list[tuple[int, int]]:
    """Split transport and top-level clauses without cutting values or parentheses."""
    spans = []
    start = depth = 0
    hla_list = family == "hla" and all(HLA_TOKEN.fullmatch(token.strip())
                                     for token in re.split(r"[|,;/]", HLA_LABEL.sub("", text)))
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        if not depth and char in "|\r\n;" and not hla_list:
            if index > start:
                spans.append((start, index))
            spans.append((index, index + 1))
            start = index + 1
    if start < len(text):
        spans.append((start, len(text)))
    clauses = []
    for first, last in spans:
        cuts = {first, last, *(match.end() for match in CLAUSE_MARKER.finditer(text, first, last))}
        if family == "cryoglobulin":
            cuts.update(match.end() for match in CRYO_CLAUSE.finditer(text, first, last))
        cuts = sorted(cuts)
        clauses.extend(zip(cuts, cuts[1:]))
    return clauses


def segment(row: dict) -> list[SemanticSegment]:
    """Separate results, protected context, and transport with exact offsets."""
    family = row["family"]
    key = row.get("mapping_id") or digest([row.get("variable"), row["raw_json"], row["context_json"]])
    output = []
    evidence_sources = sources(row)
    raw = evidence_sources.get("raw_text", "")
    linked_comment = any(ADMIN.match(raw[start:end].lstrip())
                         for start, end in boundaries(raw, family))
    for source, text in evidence_sources.items():
        scoped = None

        def emit(start, end, role, rule):
            if start != end:
                output.append(SemanticSegment(
                    mapping_id=key, segment_id=digest([key, source, start, end, VERSION]),
                    source_field=source, start=start, end=end, text=text[start:end],
                    role=role, role_rule=rule, segmentation_version=VERSION))

        for start, end in ([(0, len(text))] if family == "medication"
                           else boundaries(text, family)):
            clause = text[start:end]
            trimmed = clause.strip()
            if not trimmed or trimmed in {";", "|"}:
                emit(start, end, "unresolved", "transport")
                continue
            if family == "medication":
                emit(start, end, "patient_result" if source == "raw_text" else "unresolved", "documented_medication")
                continue
            marker = ADMIN.match(clause.lstrip())
            if marker:
                scoped = None
                marker_end = start + len(clause) - len(clause.lstrip()) + marker.end()
                emit(start, marker_end, "administrative_status", "comment_marker")
                start = marker_end
                clause = text[start:end]
                trimmed = clause.strip()
                if not trimmed:
                    emit(start, end, "unresolved", "transport")
                    continue
            if family in {"other", "derived"}:
                emit(start, end, "unresolved", "outside_clinical_profiles")
                continue
            matches = [match for match in HEADINGS.finditer(clause)
                       if clause[:match.start()].count("(") == clause[:match.start()].count(")")]
            cuts = [0, *[m.start() for m in matches if m.start()], len(clause)]
            hints = {m.start(): m.lastgroup for m in matches}
            for left, right in zip(cuts, cuts[1:]):
                value = clause[left:right].strip()
                heading = hints.get(left)
                if heading == "current" or RESULT_LABEL.match(value):
                    if scoped not in {"reference_rule", "assay_method", "general_interpretation"} or (
                        left == 0 and RESULT_LABEL.match(value)
                    ):
                        scoped = "patient_result"
                elif heading:
                    scoped = {"reference": "reference_rule", "method": "assay_method",
                              "general": "general_interpretation", "history": "historical_reference"}[heading]
                if family == "administrative":
                    role, rule = "administrative_status", "process_profile"
                elif heading or scoped in {"assay_method", "reference_rule", "general_interpretation", "historical_reference"} and not PROCESS.fullmatch(value):
                    role, rule = scoped, "scoped_clause"
                elif CUTOFF.match(value) and not RESULT_LABEL.match(value):
                    scoped = "reference_rule"
                    role, rule = "reference_rule", "category_threshold"
                elif PROCESS.fullmatch(value):
                    role, rule = "administrative_status", "process_statement"
                elif SPECIMEN.fullmatch(value):
                    role, rule = "specimen_information", "specimen_statement"
                elif family == "cryoglobulin" and (reading := CRYO_READING.match(value)) and (
                    reading["planned"] or not GENERIC_UNCERTAINTY.search(value)
                ):
                    role, rule = "patient_result", "timed_assay_statement"
                elif GENERIC_UNCERTAINTY.search(value):
                    role, rule = "unresolved", "uncertain_scope"
                elif (scoped == "patient_result" or (source == "raw_text" or linked_comment or marker) and
                      (RESULT_FORM.match(value) or family in DESCRIPTIVE and re.fullmatch(r"[\w /.+()%,-]+", value)
                       or family == "hla" and any(HLA_TOKEN.fullmatch(token.strip())
                                                   for token in re.split(r"[,/|]", HLA_LABEL.sub("", value)))) or
                      family in NARRATIVE | {"m_spike", "ana"} and NARRATIVE_SIGNAL.search(value)):
                    role, rule = "patient_result", "assay_source" if source == "raw_text" else "assay_comment"
                else:
                    role, rule = "unresolved", "unqualified_source"
                emit(start + left, start + right, role, rule)
    return output

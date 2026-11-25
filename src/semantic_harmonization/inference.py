"""Offline structured GPT-OSS extraction with source-span verification."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from pydantic import ValidationError

from .parsers import narrative, new_finding, normalize, parse_segment
from .schemas import AnaExtraction, Evidence, Extraction, Finding, HlaExtraction, MorphologyExtraction, Semantic
from .segmentation import segment
from .semantic_facts import canonical, make_fact, part_facts
from .settings import serialize

NARRATIVE = {"cryoglobulin", "immunofixation"}
PROFILES = {
    "ana": (AnaExtraction, "ana_pattern"),
    "ana_narrative": (AnaExtraction, "ana_pattern"),
    "hla": (HlaExtraction, "hla_result"),
    "cbc_morphology": (MorphologyExtraction, "morphology_result"),
}
ELIGIBLE_ROLES = {"patient_result", "historical_reference"}
OBSERVED = re.compile(r"\b(?:detected|identified|observed|present|seen|demonstrated|revealed|absent|negative for)\b", re.I)
NONRESULT = re.compile(r"\b(?:if|such as|for example|recommend|may occur|would|should|expected|will be|rule out|control)\b", re.I)
TEMPORAL_QUAL = re.compile(r"\b(previously|formerly|historical)\s+(positive|negative)\s*[,;]?\s*(?:and\s+)?\b(now|currently)\s+(positive|negative)\b", re.I)
ENTITY_SPAN = re.compile(r"\b(?:(?P<intensity>faint|trace)\s+)?(?:(?P<pattern>monoclonal|oligoclonal|polyclonal)\s+)?(?:(?:free)\s+)?(?:(?P<ig>ig[gmade])(?:\s+(?P<chain>kappa|lambda))?|(?P<light>kappa|lambda))\b", re.I)
NEGATION = re.compile(r"\b(?:no|not|without|absent|negative for)\b", re.I)
UNCERTAINTY = re.compile(r"\b(?:possible|suspected|questionable|cannot exclude|may represent)\b", re.I)


def force_offline() -> None:
    """Disable remote model fetching and runtime telemetry before imports."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", WANDB_MODE="disabled")


def profile(row: dict) -> str | None:
    """Select the concrete schema for an implemented assay profile."""
    family = row["family"]
    return "finding" if family in NARRATIVE else family if family in PROFILES else None


def eligible_segments(row: dict):
    """Keep only source-separated results or explicitly historical evidence."""
    roles = ELIGIBLE_ROLES if row["family"] in NARRATIVE else {"patient_result"}
    return [part for part in segment(row) if part.role in roles]


def routing(row: dict, semantic: Semantic, parts=None, facts=None) -> dict:
    """Measure supported statements and select assertions with residual scope."""
    parts = eligible_segments(row) if parts is None else [
        part for part in parts if part.role == "patient_result" or
        part.role == "historical_reference" and row["family"] in NARRATIVE]
    if facts is None:
        resolved = {part.segment_id for part in parts if any(
            ev.source == part.source_field and ev.quote in part.text and ev.field != "result_availability"
            for ev in semantic.evidence)}
    else:
        resolved = {item.segment_id for item in facts if item.fact_type != "observation_metadata"
                    and not (item.fact_type == "hla_result" and item.payload.ambiguous)}
    uncovered = []
    for part in parts:
        residual = part.text
        evidence = [ev.quote for ev in semantic.evidence if ev.source == part.source_field and
                    ev.field != "result_availability" and ev.quote in part.text]
        if row["family"] in NARRATIVE:
            evidence = [item.quote for item in semantic.findings if item.quote in part.text] + [
                item for item in evidence if item == part.text.strip() and not semantic.findings]
        for quote in evidence:
            residual = residual.replace(quote, " ", 1)
        residual = re.sub(r"\b(?:and|or|result|current)\b|[^\w]", "", residual, flags=re.I)
        if part.segment_id not in resolved or residual or semantic.interpretation_note == "Incomplete concentration expression retained.":
            uncovered.append(part)
    eligible = profile(row)
    raw = json.loads(row["raw_json"])
    if raw is None or not isinstance(raw, str) or not raw.strip():
        reason = "blank_or_nontext"
    elif row["variable"].startswith(("sero__", "pop__", "ovl__", "essdai__", "esspri__", "pro__")):
        reason = "derived_source"
    elif not eligible:
        reason = "no_verified_model_profile"
    elif not parts:
        reason = "no_result_segments"
    elif semantic.result_availability in {"not_performed", "pending", "not_interpretable"}:
        reason = "assay_unavailable"
    elif uncovered:
        reason = "partial_result" if resolved else "unparsed_result"
    else:
        reason = "deterministic_complete"
    send = reason in {"partial_result", "unparsed_result"}
    return dict(profile=eligible, reason=reason, send=send, relevant_segments=len(parts),
                resolved_segments=len(resolved), residual_segments=len(uncovered))


def needs_extraction(row: dict, semantic: Semantic) -> bool:
    """Retain the public routing predicate for existing callers."""
    return routing(row, semantic)["send"]


def supports(finding: Finding, text: str, family: str) -> bool:
    """Verify polarity, entity attributes and scope against the surrounding text."""
    if finding.quote not in text or text.count(finding.quote) != 1 or NONRESULT.search(text):
        return False
    deterministic = narrative(text, family)
    for match in TEMPORAL_QUAL.finditer(text) if family == "cryoglobulin" else ():
        for position, temporal in ((2, "historical"), (4, "current")):
            deterministic.append(new_finding("cryoglobulin", match[0], temporality=temporal,
                                             assertion="present" if match[position].lower() == "positive" else "absent"))
    clinical = finding.model_dump(exclude={"document_ids", "quote"})
    if any(finding.quote in verified.quote and clinical == verified.model_dump(exclude={"document_ids", "quote"})
           for verified in deterministic):
        return True
    if finding.entity not in {"immunoglobulin", "light_chain"}:
        return False
    start = text.index(finding.quote)
    before = re.split(r"[.;\n]", text[:start])[-1]
    after = re.split(r"[.;\n]", text[start + len(finding.quote):], maxsplit=1)[0]
    clause = before + finding.quote + after
    candidates = list(ENTITY_SPAN.finditer(finding.quote))
    if len(candidates) != 1:
        return False
    match = candidates[0]
    norm = normalize(clause)
    if not OBSERVED.search(norm) and not UNCERTAINTY.search(norm):
        return False
    timeline = re.findall(r"\b(previously|historical|prior|currently|now|planned)\b", norm)
    if len(set(timeline)) > 1 or re.search(r"\b(?:but|however|versus|compared)\b", norm):
        return False
    if not timeline and re.search(r"\b(?:previously|historical|prior|currently|now|planned)\b", text, re.I):
        return False
    temporality = {"previously": "historical", "historical": "historical", "prior": "historical",
                   "currently": "current", "now": "current", "planned": "planned"}.get(timeline[0] if timeline else "", "unspecified")
    assertion = "uncertain" if UNCERTAINTY.search(norm) else "absent" if NEGATION.search(norm) else "present"
    if len(list(ENTITY_SPAN.finditer(clause))) != 1:
        return False
    expected = new_finding(
        "immunoglobulin" if match["ig"] else "light_chain", finding.quote,
        assertion=assertion, temporality=temporality,
        immunoglobulin_class="Ig" + match["ig"][-1].upper() if match["ig"] else None,
        light_chain=(match["chain"] or match["light"] or "").lower() or None,
        pattern=match["pattern"].lower() if match["pattern"] else None,
        intensity_descriptor=match["intensity"].lower() if match["intensity"] else None,
    )
    return expected.model_dump(exclude={"document_ids", "quote"}) == clinical


def validate_extraction(payload: str, row: dict, snippets: list[dict]) -> Extraction:
    """Accept only final structured claims with compatible source spans."""
    extracted = Extraction.model_validate_json(payload)
    allowed = {item["doc_id"] for item in snippets}
    for finding in extracted.findings:
        if not set(finding.document_ids).issubset(allowed):
            raise ValueError("Unknown evidence document")
        if len(finding_segments(finding, row)) != 1:
            raise ValueError("Finding lacked supported assertion, temporal scope or source span")
    return extracted


def finding_segments(finding: Finding, row: dict) -> list:
    """Locate one unambiguous source segment with supported scope."""
    return [part for part in eligible_segments(row)
            if (part.role == "historical_reference") == (finding.temporality == "historical")
            and supports(finding, part.text, row["family"])]


def apply_extraction(semantic: Semantic, extracted: Extraction, row: dict, version: str) -> list:
    """Merge only distinct, verified findings and reject explicit contradictions."""
    additions = []
    for finding in extracted.findings:
        if not finding_segments(finding, row):
            raise ValueError("Proposed finding has no supported segment")
        clinical = finding.model_dump(exclude={"quote", "document_ids"})
        if any(item.model_dump(exclude={"quote", "document_ids"}) == clinical for item in [*semantic.findings, *additions]):
            continue
        for item in [*semantic.findings, *additions]:
            identity = ("entity", "immunoglobulin_class", "light_chain", "incubation_hours")
            same_scope = (item.temporality == finding.temporality or
                          {item.temporality, finding.temporality} <= {"current", "unspecified"})
            if same_scope and all(getattr(item, key) == getattr(finding, key) for key in identity):
                attributes = ("assertion", "pattern", "intensity_descriptor", "percentage", "comparator")
                if any(getattr(item, key) is not None and getattr(finding, key) is not None
                       and getattr(item, key) != getattr(finding, key) for key in attributes):
                    raise ValueError("Proposed finding conflicts with existing evidence")
        additions.append(finding)
    facts = []
    for finding in additions:
        part, = finding_segments(finding, row)
        facts.append(make_fact(row, part, f'{row["family"]}_finding', finding, "llm_rag", version,
                               finding.quote, finding.temporality))
        semantic.findings.append(finding)
        semantic.evidence.append(Evidence(field="findings", quote=finding.quote, source=part.source_field,
                                          rule=version, strategy="llm_rag"))
    if facts:
        if semantic.strategy == "unchanged":
            semantic.strategy = "llm_rag"
        semantic.interpretation_note = None
    return facts


def validate_profile(answer: str, row: dict, retrieved: list, cfg: dict, vocab: dict) -> list:
    """Check concrete proposals against the whole result clause and local grammar."""
    schema, kind = PROFILES[row["family"]]
    extraction = schema.model_validate_json(answer)
    allowed_docs = {item["doc_id"] for item in retrieved}
    parts = {part.segment_id: part for part in eligible_segments(row)}
    accepted = []
    for candidate in extraction.candidates:
        part = parts.get(candidate.segment_id)
        if (part is None or not part.start <= candidate.start < candidate.end <= part.end
                or part.text[candidate.start-part.start:candidate.end-part.start] != candidate.quote
                or not set(candidate.document_ids) <= allowed_docs or NONRESULT.search(part.text)):
            raise ValueError("Invalid profile evidence span")
        # A bounded result label can delimit a literal value. Reparse it and
        # independently verify every asserted attribute before accepting it.
        clause = part.text.strip()
        value = re.sub(r"^(?:current |patient )?(?:result|observed pattern|morphology|hla typing)\s*:\s*", "", clause, flags=re.I)
        if candidate.quote.strip() not in {value, clause}:
            raise ValueError("Profile proposal omits assertion context")
        parsed = parse_segment(row, cfg, vocab, value, part.source_field, part.role)
        if not any(item.fact_type == kind and item.payload == candidate.payload
                   for item in part_facts(row, part, parsed, cfg["normalization"]["version"])):
            raise ValueError("Profile proposal is unsupported")
        fact = make_fact(row, part, kind, candidate.payload, "llm_rag", cfg["prompts"]["profile_version"], candidate.quote)
        accepted.append(fact.model_copy(update={"document_ids": candidate.document_ids}))
    return accepted


class LocalModel:
    """Run the local GPT-OSS vLLM backend with schema-constrained final output."""

    def __init__(self, cfg: dict):
        force_offline()
        os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(Path(cfg["paths"]["knowledge_dir"]) / "harmony")
        from openai_harmony import HarmonyEncodingName, Role, load_harmony_encoding
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self.cfg = cfg
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.role = Role.ASSISTANT
        path = cfg["paths"]["gpt_oss"]
        model = cfg["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.engine = LLM(model=path, tokenizer=path, max_model_len=model["max_context_tokens"],
                          gpu_memory_utilization=model["gpu_memory_utilization"], trust_remote_code=False,
                          seed=model["seed"], disable_log_stats=True,
                          dtype="bfloat16",
                          load_format="safetensors",
                          enforce_eager=True,
                          tensor_parallel_size=1,
                          disable_custom_all_reduce=True)
        schemas = {"finding": Extraction} | {family: contract[0] for family, contract in PROFILES.items()}
        self.params = {
            name: SamplingParams(temperature=model["temperature"], max_tokens=model["max_output_tokens"],
                                 seed=model["seed"], skip_special_tokens=False,
                                 structured_outputs=StructuredOutputsParams(json=schema.model_json_schema()))
            for name, schema in schemas.items()
        }

    def messages(self, row: dict, retrieved: list[dict], retry: bool = False) -> list[dict]:
        """Construct a bounded, local-only dialogue from assay context and text."""
        context = json.loads(row["context_json"])
        active = profile(row)
        parts = eligible_segments(row)
        # Finding extraction uses literal source evidence.
        # Profile extraction additionally requires exact segment IDs.
        if active == "finding":
            raw_segments = {
                f"{part.source_field}#{i + 1}": part.text
                for i, part in enumerate(parts)
            }
        else:
            raw_segments = {
                part.segment_id: part.text
                for part in parts
            }
        payload = dict(concept=row["concept"], variable_name=row["variable"], raw_segments=raw_segments,
                       assay_method=context.get("method"), raw_unit=context.get("unit"),
                       schema_version=self.cfg["schema_version"],
                       retrieved_documents=[json.loads(item["snippet"]) for item in retrieved])
        if active != "finding":
            payload["segments"] = [part.model_dump(exclude={"mapping_id", "text"}) for part in parts]
        if retry:
            payload["validation_reminder"] = "Keep only literal findings with supported assertion, temporal scope and exact source span. Return an empty list if unsupported."
        allowed_document_ids = sorted({
            item["doc_id"] for item in retrieved
        })

        payload["allowed_document_ids"] = allowed_document_ids

        system = self.cfg["prompts"][
            "system" if active == "finding" else "profile_system"
        ]

        system += (
            "\n\nEVIDENCE IDENTIFIER CONTRACT:\n"
            "Source segment identifiers and retrieval document "
            "identifiers are different entities.\n"
            "A segment_id identifies a source text span, "
            "not a retrieved document.\n"
            "Never copy a segment_id, mapping_id, source label, "
            "or arbitrary hash into document_ids.\n"
            "Every entry in document_ids MUST appear exactly "
            "in allowed_document_ids.\n"
            "Do not invent document identifiers.\n"
            "Cite a retrieved document only when it actually "
            "supports the proposed interpretation.\n"
            "If allowed_document_ids is empty, every "
            "document_ids field MUST be an empty JSON array [].\n"
            "Literal source evidence alone does not require "
            "a retrieved document citation.\n"
        )

        if not allowed_document_ids:
            system += (
                "For this request no retrieval documents are "
                "available. You MUST return document_ids=[] "
                "for every finding or candidate.\n"
            )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": serialize(payload)}
        ]

    def chat(self, messages: list[list[dict]], active: str = "finding") -> list[str | None]:
        """Pass only fitted dialogues to Harmony and preserve batch alignment."""
        model = self.cfg["model"]
        available = model["max_context_tokens"] - model["max_output_tokens"]
        fitting = [i for i, prompt in enumerate(messages)
                   if len(self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)) <= available]
        result = [None] * len(messages)
        if not fitting:
            return result
        try:
            outputs = self.engine.chat([messages[i] for i in fitting], sampling_params=self.params[active], use_tqdm=False)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or len(messages) == 1:
                raise
            print(f"Reducing local GPT-OSS batch after GPU memory exhaustion ({len(messages)} items).")
            split = len(messages) // 2
            return self.chat(messages[:split], active) + self.chat(messages[split:], active)
        for i, output in zip(fitting, outputs, strict=True):
            final = output.outputs[0]
            if final.finish_reason != "stop":
                result[i] = ""
                continue
            result[i] = self.final_channel(final.token_ids)
        return result

    def final_channel(self, ids: list[int]) -> str:
        """Use the official Harmony parser and retain only one assistant final."""
        from openai_harmony import HarmonyError

        try:
            messages = self.encoding.parse_messages_from_completion_tokens(ids, self.role, strict=False)
        except HarmonyError:
            return ""
        final = [message for message in messages if message.channel == "final"]
        if len(final) != 1:
            return ""
        return "".join(part.text for part in final[0].content if hasattr(part, "text"))

    def smoke(self) -> None:
        """Exercise actual GPU inference, schema, negation and timed readings."""
        examples = [
            ("immunofixation", "No anomalous immunoglobulins."),
            ("cryoglobulin", "negative at 24h, positive at 7d"),
            ("cryoglobulin", "not performed"),
        ]
        rows = [dict(concept=family, variable="lab__" + family + "__text", family=family,
                     raw_json=serialize(raw), context_json="{}") for family, raw in examples]
        reply = self.chat([self.messages(row, []) for row in rows])
        for row, (_, raw), answer in zip(rows, examples, reply, strict=True):
            extracted = validate_extraction(answer or "", row, [])
            meaning = lambda item: serialize(item.model_dump(exclude={"quote", "document_ids"}))
            expected = sorted(meaning(item) for item in narrative(raw, row["family"]))
            if sorted(meaning(item) for item in extracted.findings) != expected:
                raise RuntimeError("GPT-OSS GPU smoke failed complete finding comparison")
        examples = {
            "ana": ("ana_hep2_cytoplasmic_pattern", "Positive"),
            "ana_narrative": ("ana_pattern_comment", "Observed pattern: reticular"),
            "hla": ("hla_a", "99"),
            "cbc_morphology": ("cbc_diff__anisocytosis_blood", "few"),
        }
        vocab = {"ucum": set(), "ipd": {}, "rxnorm": {}}
        for active, (concept, raw) in examples.items():
            schema, kind = PROFILES[active]
            row = dict(concept=concept, variable=f"lab__{concept}__text", family=active, raw_json=serialize(raw),
                       context_json="{}")
            negative = row | {"raw_json": serialize("Method: synthetic reference only")}
            answer, abstain = self.chat([self.messages(row, []), self.messages(negative, [])], active)
            proposed = validate_profile(answer or "", row, [], self.cfg, vocab)
            part, = segment(row)
            expected = [fact for fact in part_facts(row, part, parse_segment(row, self.cfg, vocab, raw, part.source_field,
                                                                            part.role), self.cfg["normalization"]["version"])
                        if fact.fact_type == kind]
            if (not expected or sorted(map(canonical, proposed)) != sorted(map(canonical, expected))
                    or schema.model_validate_json(abstain or "").candidates):
                raise RuntimeError("GPT-OSS GPU smoke failed complete profile comparison")


def extract_batch(model: LocalModel, rows: list[dict], retrieved: dict, retries: int, *, cfg=None, vocab=None) -> list[dict]:
    """Batch requests by concrete schema and retry unsupported proposals."""
    groups = {}
    for row in rows:
        groups.setdefault(profile(row), []).append(row)
    output = {}
    for active, pending in groups.items():
        for round_number in range(retries + 1):
            next_pending = []
            messages = [model.messages(row, retrieved[row["mapping_id"]], bool(round_number)) for row in pending]
            answers = model.chat(messages) if active == "finding" else model.chat(messages, active)
            for row, answer in zip(pending, answers, strict=True):
                key = row["mapping_id"]
                result = dict(mapping_id=key, findings=[], facts=[], status="context_limit")
                if answer is not None:
                    try:
                        if active == "finding":
                            extracted = validate_extraction(answer, row, retrieved[key])
                            result["findings"] = [item.model_dump() for item in extracted.findings]
                        else:
                            result["facts"] = [item.model_dump() for item in validate_profile(answer, row, retrieved[key], cfg, vocab)]
                    except ValidationError:
                        result["status"] = "invalid_schema"
                        next_pending.append(row)
                    except ValueError:
                        result["status"] = "unsupported_evidence"
                        next_pending.append(row)
                    else:
                        result["status"] = "validated"
                output[key] = result
            pending = next_pending
            if not pending:
                break
    return [output[row["mapping_id"]] for row in rows]

"""Offline structured GPT-OSS extraction with source-span verification."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re

from pydantic import BaseModel, StrictStr, ValidationError

from .parsers import ALLELE, hla, narrative, new_finding, normalize, parse_segment
from .schemas import AnaExtraction, Evidence, Extraction, Finding, HlaExtraction, MorphologyExtraction, Semantic
from .segmentation import segment
from .semantic_facts import canonical, coverage, make_fact, part_facts
from .settings import serialize

NARRATIVE = {"cryoglobulin", "immunofixation"}
PROFILES = {
    "ana": (AnaExtraction, "ana_pattern"),
    "ana_narrative": (AnaExtraction, "ana_pattern"),
    "hla": (HlaExtraction, "hla_result"),
    "cbc_morphology": (MorphologyExtraction, "morphology_result"),
}
HLA_COMPUTED = {"lookup_locus", "normalized_report", "group", "ipd_match", "allele"}
ELIGIBLE_ROLES = {"patient_result", "historical_reference"}
OBSERVED = re.compile(r"\b(?:detected|identified|observed|present|seen|demonstrated|revealed|absent|negative for)\b", re.I)
NONRESULT = re.compile(r"\b(?:if|such as|for example|recommend|may occur|would|should|expected|will be|rule out|control)\b", re.I)
TEMPORAL_QUAL = re.compile(r"\b(previously|formerly|historical)\s+(positive|negative)\s*[,;]?\s*(?:and\s+)?\b(now|currently)\s+(positive|negative)\b", re.I)
ENTITY_SPAN = re.compile(r"\b(?:(?P<intensity>faint|trace)\s+)?(?:(?P<pattern>monoclonal|oligoclonal|polyclonal)\s+)?(?:(?:free)\s+)?(?:(?P<ig>ig[gmade])(?:\s+(?P<chain>kappa|lambda))?|(?P<light>kappa|lambda))\b", re.I)
NEGATION = re.compile(r"\b(?:no|not|without|absent|negative for)\b", re.I)
UNCERTAINTY = re.compile(r"\b(?:possible|suspected|questionable|cannot exclude|may represent)\b", re.I)


class EvidenceError(ValueError):
    """An extraction claim failed source or consistency verification."""


class InferenceUnavailable(RuntimeError):
    """The optional engine cannot safely be used again in the current run."""


class CompactResult(BaseModel):
    result: StrictStr


@dataclass(frozen=True)
class Completion:
    answer: str | None
    finish_reason: str | None = None
    has_final_channel: bool | None = None


@dataclass(frozen=True)
class ProfileSmokeResult:
    profile: str
    status: str
    reason_code: str | None
    attempts: int
    cases_passed: int
    cases_total: int


@dataclass(frozen=True)
class SmokeResult:
    profiles: dict[str, ProfileSmokeResult]
    audit: list[dict]
    model_available: bool


def skipped_smoke(reason: str, *, status: str = "not_run") -> SmokeResult:
    return SmokeResult({
        name: ProfileSmokeResult(name, status, reason, 0, 0, count)
        for name, count in (("finding", 3), *((name, 2) for name in PROFILES))
    }, [], False)


def empty_extraction(row, status, reason=None):
    return dict(mapping_id=row["mapping_id"], findings=[], facts=[], status=status, reason_code=reason)


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
    selected = row.get("_residual_segment_ids")
    return [part for part in segment(row) if part.role in roles
            and (selected is None or part.segment_id in selected)]


def routing(row: dict, semantic: Semantic, parts=None, facts=None) -> dict:
    """Measure supported statements and select assertions with residual scope."""
    parts = eligible_segments(row) if parts is None else [
        part for part in parts if part.role == "patient_result" or
        part.role == "historical_reference" and row["family"] in NARRATIVE]
    if facts is None:
        resolved = {part.segment_id for part in parts if any(
            ev.source == part.source_field and ev.quote in part.text and ev.field != "result_availability"
            for ev in semantic.evidence)}
        uncovered = [part for part in parts if part.segment_id not in resolved]
    else:
        spans = coverage(parts, facts)
        resolved = {key for key, (supported, gaps) in spans.items() if supported and not gaps}
        uncovered = [part for part in parts if part.segment_id not in resolved]
    if semantic.interpretation_note == "Incomplete concentration expression retained.":
        uncovered = parts
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
    elif uncovered and row["family"] == "hla":
        reason = "hla_requires_literal_resource_verification"
    elif uncovered:
        reason = "partial_result" if resolved else "unparsed_result"
    else:
        reason = "deterministic_complete"
    send = reason in {"partial_result", "unparsed_result"}
    return dict(profile=eligible, reason=reason, send=send, relevant_segments=len(parts),
                resolved_segments=len(resolved), residual_segments=len(uncovered),
                residual_segment_ids=[part.segment_id for part in uncovered])


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
            raise EvidenceError("Unknown evidence document")
        if len(finding_segments(finding, row)) != 1:
            raise EvidenceError("Finding lacked supported assertion, temporal scope or source span")
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
            raise EvidenceError("Proposed finding has no supported segment")
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
                    raise EvidenceError("Proposed finding conflicts with existing evidence")
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
            raise EvidenceError("Invalid profile evidence span")
        # A bounded result label can delimit a literal value. Reparse it and
        # independently verify every asserted attribute before accepting it.
        clause = part.text.strip()
        value = literal_profile_value(clause)
        if candidate.quote.strip() not in {value, clause}:
            raise EvidenceError("Profile proposal omits assertion context")
        parsed = parse_segment(row, cfg, vocab, value, part.source_field, part.role)
        computed = HLA_COMPUTED if kind == "hla_result" else set()
        verified = next((item for item in part_facts(row, part, parsed, cfg["normalization"]["version"])
                         if item.fact_type == kind
                         and item.payload.model_dump(exclude=computed) == candidate.payload.model_dump(exclude=computed)
                         and all(getattr(candidate.payload, field) in (None, getattr(item.payload, field)) for field in computed)), None)
        if not verified:
            raise EvidenceError("Profile proposal is unsupported")
        fact = make_fact(row, part, kind, verified.payload, "llm_rag", cfg["prompts"]["profile_version"], candidate.quote)
        accepted.append(fact.model_copy(update={"document_ids": candidate.document_ids}))
    return accepted


def literal_profile_value(clause: str) -> str:
    return re.sub(r"^(?:current |patient )?(?:result|observed pattern|morphology|hla typing)\s*:\s*",
                  "", clause, flags=re.I)


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    row: dict
    expected: list[str]


def finding_meaning(item: Finding) -> str:
    return serialize(item.model_dump(exclude={"quote", "document_ids"}))


def smoke_cases(cfg):
    """Check the deterministic side of every synthetic fixture before using the model."""
    def row(family, concept, raw):
        return dict(concept=concept, family=family, variable=f"lab__{concept}__text",
                    raw_json=serialize(raw), context_json="{}")

    def require(condition, active, case_id):
        if not condition:
            details = dict(profile=active, case_id=case_id, phase="fixture", reason_code="fixture_invalid",
                           status="failed", attempt=0, expected_count=None, accepted_count=0, fixture_valid=False,
                           finish_reason=None, has_final_channel=None, abstention_passed=None)
            print("GPT-OSS smoke " + serialize(details), flush=True)
            raise ValueError(f"GPT-OSS fixture_invalid: {case_id}")

    cases = {"finding": []}
    for case_id, family, raw, contract in (
        ("immunofixation_negation", "immunofixation", "No anomalous immunoglobulins.",
         [("anomalous_immunoglobulins", "absent", "unspecified", None)]),
        ("cryoglobulin_timed", "cryoglobulin", "negative at 24h, positive at 7d",
         [("cryoglobulin", "absent", "current", 24), ("cryoglobulin", "present", "current", 168)]),
        ("cryoglobulin_not_performed", "cryoglobulin", "not performed", []),
    ):
        sample = row(family, family, raw)
        findings = narrative(raw, family)
        require(Counter((f.entity, f.assertion, f.temporality, f.incubation_hours) for f in findings) == Counter(contract)
                and all(len(finding_segments(f, sample)) == 1 for f in findings)
                and (bool(contract) or not eligible_segments(sample)), "finding", case_id)
        cases["finding"].append(SmokeCase(case_id, sample, sorted(map(finding_meaning, findings))))

    vocab = {"ucum": set(), "ipd": {}, "rxnorm": {}}
    for active, (concept, raw, contract) in {
        "ana": ("ana_hep2_cytoplasmic_pattern", "Positive",
                {"compartment": "cytoplasmic", "state": "positive"}),
        "ana_narrative": ("ana_pattern_comment", "Observed pattern: reticular",
                          {"compartment": "cytoplasmic", "pattern": "reticular"}),
        "hla": ("hla_a", "99", {"locus": "A", "token": "99", "nomenclature": "group", "resolution": 1,
                                "allele": None, "group": None, "ipd_match": "resource_unavailable"}),
        "cbc_morphology": ("cbc_diff__anisocytosis_blood", "few",
                           {"entity": "cbc_diff__anisocytosis_blood", "descriptor": "few", "percentage": None}),
    }.items():
        sample = row(active, concept, raw)
        positive, negative = f"{active}_literal", f"{active}_method_only"
        parts = eligible_segments(sample)
        require(len(parts) == 1, active, positive)
        part, = parts
        # Independently parse the fixture as written. The validator normalizes
        # the bounded label in the source clause before verifying the answer.
        parsed = parse_segment(sample, cfg, vocab, raw, part.source_field, part.role)
        _, kind = PROFILES[active]
        facts = [f for f in part_facts(sample, part, parsed, cfg["normalization"]["version"]) if f.fact_type == kind]
        require(len(facts) == 1 and all(getattr(facts[0].payload, field) == value
                                        for field, value in contract.items()), active, positive)
        fact, = facts
        payload = fact.payload.model_dump()
        if active == "hla":
            for field in HLA_COMPUTED:
                payload[field] = None
        candidate = dict(segment_id=part.segment_id, start=part.start, end=part.end, quote=part.text,
                         document_ids=[], payload=payload)
        try:
            proposed = validate_profile(serialize({"candidates": [candidate]}), sample, [], cfg, vocab)
        except (ValidationError, EvidenceError):
            require(False, active, positive)
        expected = sorted(map(canonical, facts))
        require(sorted(map(canonical, proposed)) == expected, active, positive)
        abstain = sample | {"raw_json": serialize("Method: synthetic reference only")}
        require(not eligible_segments(abstain), active, negative)
        cases[active] = [SmokeCase(positive, sample, expected), SmokeCase(negative, abstain, [])]
    return cases, vocab


class LocalModel:
    """Run the local GPT-OSS vLLM backend with schema-constrained final output."""

    def __init__(self, cfg: dict):
        force_offline()
        os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(Path(cfg["paths"]["knowledge_dir"]) / "harmony")
        try:
            from openai_harmony import HarmonyEncodingName, Role, load_harmony_encoding
            import torch
            from torch.cuda import OutOfMemoryError
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            import vllm._C
            import vllm._moe_C
            from vllm.sampling_params import StructuredOutputsParams
            from vllm.v1.engine.exceptions import EngineDeadError
        except (ImportError, OSError) as error:
            raise InferenceUnavailable("Local inference dependencies are unavailable") from error
        if not torch.cuda.is_available() or not hasattr(torch.ops._moe_C, "topk_softmax"):
            raise InferenceUnavailable("Local inference CUDA extensions are unavailable")

        self.engine_errors = (EngineDeadError, OutOfMemoryError)
        # The deployed engine runs in-process. Its crash dump can include scheduler token IDs.
        logging.getLogger("vllm.logging_utils.dump_input").disabled = True

        self.cfg = cfg
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.role = Role.ASSISTANT
        path = cfg["paths"]["gpt_oss"]
        model = cfg["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        try:
            self.engine = LLM(model=path, tokenizer=path, max_model_len=model["max_context_tokens"],
                              gpu_memory_utilization=model["gpu_memory_utilization"], trust_remote_code=False,
                              seed=model["seed"], disable_log_stats=True,
                              dtype="bfloat16",
                              load_format="safetensors",
                              enforce_eager=True,
                              tensor_parallel_size=1,
                              disable_custom_all_reduce=True)
        except (EngineDeadError, OutOfMemoryError, TimeoutError) as error:
            raise InferenceUnavailable("Local inference engine is unavailable") from error
        schemas = {"finding": Extraction, "compact": CompactResult} | {
            family: contract[0] for family, contract in PROFILES.items()
        }
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

    def compact_messages(self, row: dict, retrieved: list[dict], ipd: dict) -> list[dict]:
        """Give the local model the literal result and its assay context."""
        payload = dict(variable=row["variable"], result=row["raw_text"],
                       assay_context=json.loads(row["context_json"]),
                       retrieved_documents=[json.loads(item["snippet"]) for item in retrieved])
        if row["family"] == "hla":
            locus = row["concept"].removeprefix("hla_").upper()
            tokens = re.finditer(rf"(?<![\w:*]){ALLELE.pattern}(?![\w:*])", row["raw_text"], re.I)
            payload["ipd_nomenclature"] = [
                {"reported": item.reported, "ipd_match": item.ipd_match}
                for token in tokens for item in hla(token[0], locus, ipd)
                if item.ipd_match in {"allele", "g_group", "p_group", "compatible_resolution"}
            ]
        return [{"role": "system", "content": self.cfg["prompts"]["compact_system"]},
                {"role": "user", "content": serialize(payload)}]

    def chat(self, messages: list[list[dict]], active: str = "finding") -> list[str | None]:
        """Keep the extraction interface and discard unfinished completions."""
        return [reply.answer for reply in self.chat_completions(messages, active)]

    def chat_completions(self, messages: list[list[dict]], active: str) -> list[Completion]:
        """Pass fitted dialogues to Harmony and preserve batch alignment."""
        model = self.cfg["model"]
        available = model["max_context_tokens"] - model["max_output_tokens"]
        template_kwargs = {"reasoning_effort": "low"} if active == "compact" else {}
        fitting = [i for i, prompt in enumerate(messages)
                   if len(self.tokenizer.apply_chat_template(
                       prompt, tokenize=True, add_generation_prompt=True, return_dict=False,
                       **template_kwargs)) <= available]
        result = [Completion(None)] * len(messages)
        if not fitting:
            return result
        try:
            chat_kwargs = {"chat_template_kwargs": template_kwargs} if template_kwargs else {}
            outputs = self.engine.chat(
                [messages[i] for i in fitting], sampling_params=self.params[active],
                use_tqdm=False, **chat_kwargs)
        except self.engine_errors as error:
            raise InferenceUnavailable("Local inference engine is unavailable") from error
        for i, output in zip(fitting, outputs, strict=True):
            completion, = output.outputs
            answer = self.final_channel(completion.token_ids)
            final_answer = (answer or "") if completion.finish_reason == "stop" else ""
            result[i] = Completion(final_answer, completion.finish_reason, answer is not None)
        return result

    def final_channel(self, ids: list[int]) -> str | None:
        """Return only a single assistant final from the official Harmony parser."""
        from openai_harmony import HarmonyError

        try:
            messages = self.encoding.parse_messages_from_completion_tokens(ids, self.role, strict=False)
        except HarmonyError:
            return None
        final = [message for message in messages if message.channel == "final"]
        if len(final) != 1:
            return None
        return "".join(part.text for part in final[0].content if hasattr(part, "text"))

    def extract_compact(self, rows: list[dict], index, ipd: dict) -> list[str]:
        """Retry only unfinished requests; keep the clinical source intact."""
        settings = self.cfg["model"]
        batch_size = settings["batch_size"]
        retries = settings["max_validation_retries"]
        logger = logging.getLogger(__name__)
        outcomes = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            retrieved = [index.retrieve(row) for row in batch]
            remaining = list(range(len(batch)))
            answers = {}
            failures = {}
            for attempt in range(retries + 1):
                # Trim only retrieved documents, never the clinical text or assay context.
                # The last retry is individual and uses literal source evidence alone.
                last_retry = attempt > 0 and attempt == retries
                groups = [[i] for i in remaining] if last_retry else [remaining]
                for group in groups:
                    prompts = []
                    for i in group:
                        docs = retrieved[i]
                        if last_retry:
                            docs = []
                        elif attempt:
                            docs = docs[:max(1, len(docs) // (2 ** attempt))]
                        prompts.append(self.compact_messages(batch[i], docs, ipd))
                    replies = self.chat_completions(prompts, "compact")
                    for i, reply in zip(group, replies, strict=True):
                        if reply.finish_reason is None:
                            failures[i] = "prompt_exceeds_context_budget"
                        elif reply.finish_reason == "length":
                            failures[i] = "output_token_limit"
                        elif reply.finish_reason != "stop":
                            failures[i] = "unfinished_generation"
                        elif not reply.has_final_channel:
                            failures[i] = "missing_harmony_final"
                        elif not reply.answer:
                            failures[i] = "empty_final_payload"
                        else:
                            try:
                                # A JSON object containing result="" is a valid extraction.
                                answers[i] = CompactResult.model_validate_json(reply.answer).result
                            except ValidationError:
                                failures[i] = "invalid_compact_json"
                        if i in answers:
                            failures.pop(i, None)
                remaining = [i for i in remaining if i not in answers]
                if not remaining:
                    break
                counts = dict(Counter(failures[i] for i in remaining))
                logger.warning(
                    "Compact extraction batch_offset=%d attempt=%d/%d remaining=%d reasons=%s",
                    start, attempt + 1, retries + 1, len(remaining), counts)
            if remaining:
                counts = dict(Counter(failures[i] for i in remaining))
                raise InferenceUnavailable(
                    f"Compact clinical extraction did not return a complete result "
                    f"(batch_offset={start}, remaining={len(remaining)}, reasons={counts})")
            outcomes.extend(answers[i] for i in range(len(batch)))
        return outcomes

    def smoke(self, fixtures=None) -> SmokeResult:
        """Run all required synthetic cases independently with one bounded retry."""
        cases, vocab = smoke_cases(self.cfg) if fixtures is None else fixtures
        profiles, audit = {}, []
        model_available = True
        for active, fixtures in cases.items():
            passed, attempts, reason = 0, 0, None
            for case in fixtures:
                if not model_available:
                    reason = reason or "inference_runtime_error"
                    break
                for attempt in (1, 2):
                    completion, accepted = Completion(None), []
                    try:
                        completion, = self.chat_completions([self.messages(case.row, [], attempt == 2)], active)
                    except InferenceUnavailable:
                        model_available = False
                        failure = "inference_runtime_error"
                    except TimeoutError:
                        failure = "inference_runtime_error"
                    else:
                        failure, accepted = self.check_smoke_answer(active, case, completion, vocab)
                    details = dict(profile=active, case_id=case.case_id, attempt=attempt,
                                   status="passed" if failure is None else "failed", reason_code=failure,
                                   phase="generation" if failure in {"inference_runtime_error", "context_limit",
                                                                      "truncated_response", "empty_response"} else "validation",
                                   expected_count=len(case.expected), accepted_count=len(accepted),
                                   abstention_passed=None if case.expected else failure is None, fixture_valid=True,
                                   finish_reason=completion.finish_reason, has_final_channel=completion.has_final_channel)
                    audit.append(details)
                    attempts += 1
                    print("GPT-OSS smoke " + serialize(details), flush=True)
                    if failure is None or not model_available:
                        break
                passed += failure is None
                if failure is not None:
                    reason = reason or failure
            profiles[active] = ProfileSmokeResult(active, "passed" if passed == len(fixtures) else "disabled",
                                                  reason, attempts, passed, len(fixtures))
        return SmokeResult(profiles, audit, model_available)

    def check_smoke_answer(self, active, case, reply, vocab):
        if reply.finish_reason == "length":
            return "truncated_response", []
        if reply.answer is None:
            return "context_limit", []
        if not reply.answer.strip():
            return "empty_response", []
        try:
            if not case.expected:
                schema, field = (Extraction, "findings") if active == "finding" else (PROFILES[active][0], "candidates")
                return ("failed_abstention" if getattr(schema.model_validate_json(reply.answer), field) else None), []
            if active == "finding":
                accepted = [finding_meaning(item) for item in validate_extraction(reply.answer, case.row, []).findings]
            else:
                accepted = [canonical(item) for item in validate_profile(reply.answer, case.row, [], self.cfg, vocab)]
        except ValidationError:
            return "invalid_schema", []
        except EvidenceError:
            return "unsupported_evidence", []
        expected, proposed = Counter(case.expected), Counter(accepted)
        if proposed - expected:
            return "unexpected_fact", accepted
        return ("missing_expected_fact" if expected - proposed else None), accepted


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
            try:
                answers = model.chat(messages) if active == "finding" else model.chat(messages, active)
            except TimeoutError:
                for row in pending:
                    key = row["mapping_id"]
                    output[key] = empty_extraction(row, "timeout", "inference_runtime_error")
                continue
            for row, answer in zip(pending, answers, strict=True):
                key = row["mapping_id"]
                result = empty_extraction(row, "context_limit")
                if answer is not None and not answer.strip():
                    result["status"] = "empty_response"
                    next_pending.append(row)
                elif answer is not None:
                    try:
                        if active == "finding":
                            extracted = validate_extraction(answer, row, retrieved[key])
                            result["findings"] = [item.model_dump() for item in extracted.findings]
                        else:
                            result["facts"] = [item.model_dump() for item in validate_profile(answer, row, retrieved[key], cfg, vocab)]
                    except ValidationError:
                        result["status"] = "invalid_schema"
                        next_pending.append(row)
                    except EvidenceError:
                        result["status"] = "unsupported_evidence"
                        next_pending.append(row)
                    else:
                        result["status"] = "validated"
                if result["status"] != "validated":
                    result["reason_code"] = result["status"]
                output[key] = result
            pending = next_pending
            if not pending:
                break
    return [output[row["mapping_id"]] for row in rows]

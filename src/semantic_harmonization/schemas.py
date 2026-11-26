"""Strict clinical result, child records, and Arrow storage contracts."""
from __future__ import annotations

from typing import Literal
from decimal import Decimal, InvalidOperation

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Finding(Strict):
    entity: Literal["anomalous_immunoglobulins", "immunoglobulin", "light_chain", "cryoglobulin", "cryoprecipitate"]
    assertion: Literal["present", "absent", "uncertain"]
    temporality: Literal["current", "historical", "planned", "unspecified"]
    immunoglobulin_class: Literal["IgG", "IgM", "IgA", "IgD", "IgE"] | None
    light_chain: Literal["kappa", "lambda"] | None
    pattern: Literal["monoclonal", "oligoclonal", "polyclonal"] | None
    intensity_descriptor: Literal["faint", "trace"] | None
    incubation_hours: int | None
    incubation_stage: Literal["preliminary", "final"] | None = None
    percentage: str | None
    comparator: Literal["EQ", "LT", "LE", "GT", "GE"] | None
    quote: str = Field(min_length=1)
    document_ids: list[str]


class Extraction(Strict):
    findings: list[Finding]


class Allele(Strict):
    locus: str
    reported: str
    allele: str | None
    resolution_fields: int | None
    suffix: str | None
    source_order: int
    ipd_match: str | None
    lookup_locus: str | None = None
    normalized_report: str | None = None
    group: str | None = None


class Medication(Strict):
    reported: str
    name: str
    dose: str | None = None
    unit: str | None = None
    route: str | None = None
    frequency: str | None = None
    rxcui: str | None = None
    source: str | None
    documentary_evidence: str
    window: str | None
    temporal_availability: Literal["unknown", "retrospective"]
    quote: str


class Evidence(Strict):
    field: str
    quote: str
    source: str
    rule: str
    strategy: str


class Semantic(Strict):
    qualitative_result: Literal["positive", "negative", "equivocal", "indeterminate", "borderline", "reactive", "nonreactive", "normal", "trace"] | None = None
    exact_value: str | None = None
    limit_value: str | None = None
    comparator: Literal["EQ", "LT", "LE", "GT", "GE", "RANGE"] | None = None
    range_low: str | None = None
    range_high: str | None = None
    raw_unit: str | None = None
    ucum_unit: str | None = None
    titer_numerator: int | None = None
    titer_denominator: int | None = None
    titer_comparator: Literal["EQ", "LT", "LE", "GT", "GE"] | None = None
    pattern_terms: list[str] = Field(default_factory=list)
    category: str | None = None
    ordinal_grade: str | None = None
    ordinal_score: int | None = None
    descriptor: str | None = None
    specimen_state: Literal["acceptable", "hemolyzed", "clotted", "rejected", "insufficient", "other"] | None = None
    result_availability: Literal["reported", "not_performed", "pending", "not_interpretable"] | None = None
    findings: list[Finding] = Field(default_factory=list)
    alleles: list[Allele] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    strategy: str = "unchanged"
    interpretation_note: str | None = None


STR, INT, BOOL = pa.string(), pa.int64(), pa.bool_()
FINDING = pa.schema([
    (name, INT if name == "incubation_hours" else pa.list_(STR) if name == "document_ids" else STR)
    for name in Finding.model_fields
])
ALLELE = pa.schema([(name, INT if name in {"resolution_fields", "source_order"} else STR) for name in Allele.model_fields])
MEDICATION = pa.schema([(name, STR) for name in Medication.model_fields])
EVIDENCE = pa.struct([(name, STR) for name in Evidence.model_fields])
NESTED = {"findings": pa.list_(pa.struct(FINDING)), "alleles": pa.list_(pa.struct(ALLELE)),
          "medications": pa.list_(pa.struct(MEDICATION)), "evidence": pa.list_(EVIDENCE),
          "pattern_terms": pa.list_(STR), "titer_numerator": INT, "titer_denominator": INT, "ordinal_score": INT}
SEMANTIC = pa.schema([(name, NESTED.get(name, STR)) for name in Semantic.model_fields])
IDENTITY = pa.schema([(name, STR) for name in ("mapping_id", "variable", "concept", "family", "raw_type", "raw_json", "context_json")])
INVENTORY = pa.schema([*IDENTITY, ("episodes", INT), ("patients", INT), ("observations", INT), ("baseline", INT),
                       ("laboratory_source_events", INT)])
DICTIONARY = pa.schema([*INVENTORY, *SEMANTIC, ("knowledge_doc_ids", pa.list_(STR))])
LINEAGE = pa.schema([
    ("observation_id", STR), ("mapping_id", STR), ("patient_id", STR), ("clinical_episode_id", STR),
    ("source_grain", STR), ("source_row", INT), ("source_observation_id", STR), ("source_hash", STR),
    ("measurement_date", STR), ("result_available_date", STR), ("paired_numeric_json", STR),
    ("n_measurements", INT), ("source_reference_status", STR), ("result_valid", BOOL), ("baseline", BOOL),
])
OBSERVATIONS = pa.schema([*LINEAGE, *[f for f in IDENTITY if f.name != "mapping_id"], *SEMANTIC])
CHILD_LINK = pa.schema([("observation_id", STR), ("mapping_id", STR), ("ordinal", INT)])
TRACE = pa.schema([("mapping_id", STR), ("doc_id", STR), ("rank", INT), ("lexical_rank", INT),
                   ("dense_rank", INT), ("exact", BOOL), ("snippet", STR), ("source", STR), ("source_version", STR)])
EXTRACTIONS = pa.schema([("mapping_id", STR), ("status", STR), ("findings", pa.list_(pa.struct(FINDING)))])
FEATURES = pa.schema([(name, STR) for name in ("name", "source", "kind", "dtype", "units", "temporal_availability",
                                               "allowed_use", "lineage", "analytic_eligibility_rule")])


Role = Literal["patient_result", "reference_rule", "assay_method", "specimen_information",
               "administrative_status", "general_interpretation", "historical_reference", "unresolved"]
SourceField = Literal["raw_text", "context.observation_comment", "context.observation_note"]
Temporality = Literal["current", "historical", "planned", "unspecified"]


class SemanticSegment(Strict):
    segment_id: str
    mapping_id: str
    source_field: SourceField
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)
    role: Role
    role_rule: str
    segmentation_version: str


class QualitativeResult(Strict):
    category: str
    descriptor: str | None = None


class NumericResult(Strict):
    exact_value: str | None = None
    limit_value: str | None = None
    range_low: str | None = None
    range_high: str | None = None
    comparator: Literal["EQ", "LT", "LE", "GT", "GE", "RANGE"]
    raw_unit: str | None = None
    ucum_unit: str | None = None

    @model_validator(mode="after")
    def check_interval(self):
        fields = (self.exact_value, self.limit_value, self.range_low, self.range_high)
        expected = {"EQ": (True, False, False, False), "RANGE": (False, False, True, True)}
        if tuple(value is not None for value in fields) != expected.get(self.comparator, (False, True, False, False)):
            raise ValueError("Numeric payload disagrees with comparator")
        try:
            decimals = [Decimal(value) for value in fields if value is not None]
        except InvalidOperation as error:
            raise ValueError("Invalid decimal payload") from error
        if any(not value.is_finite() for value in decimals):
            raise ValueError("Nonfinite numeric payload")
        if self.comparator == "RANGE" and Decimal(self.range_low) > Decimal(self.range_high):
            raise ValueError("Numeric range is reversed")
        return self


class OrdinalResult(Strict):
    grade: str
    descriptor: str | None = None
    score: int | None = None


class TiterResult(Strict):
    numerator: int = Field(gt=0)
    denominator: int = Field(gt=0)
    comparator: Literal["EQ", "LT", "LE", "GT", "GE"]


class AnaPattern(Strict):
    compartment: Literal["nuclear", "cytoplasmic", "mitotic"] | None = None
    pattern: str | None = None
    state: Literal["positive", "negative", "equivocal", "indeterminate", "borderline"] | None = None


class SpikeResult(Strict):
    detection: Literal["present", "absent", "uncertain"] | None = None
    concentration: NumericResult | None = None


class HlaResult(Strict):
    locus: str
    token: str
    nomenclature: Literal["allele", "group", "alternatives", "unresolved"]
    allele: str | None = None
    resolution: int | None = None
    ipd_match: str | None = None
    suffix: str | None = None
    lookup_locus: str | None = None
    normalized_report: str | None = None
    group: str | None = None
    ambiguous: bool = False
    alternatives: list[str] = Field(default_factory=list)
    position: int = Field(ge=0)


class MorphologyResult(Strict):
    entity: str
    descriptor: str | None = None
    grade: str | None = None
    percentage: NumericResult | None = None


class ObservationMetadata(Strict):
    role: Role
    descriptor: str | None = None
    specimen_state: str | None = None
    result_availability: str | None = None


FACT_PAYLOADS = {
    "qualitative_result": QualitativeResult, "numeric_result": NumericResult,
    "ordinal_result": OrdinalResult, "titer_result": TiterResult, "ana_pattern": AnaPattern,
    "immunofixation_finding": Finding, "cryoglobulin_finding": Finding, "m_spike_result": SpikeResult,
    "hla_result": HlaResult, "morphology_result": MorphologyResult, "observation_metadata": ObservationMetadata,
    "medication_context": Medication,
}

Payload = (QualitativeResult | NumericResult | OrdinalResult | TiterResult | AnaPattern |
           Finding | SpikeResult | HlaResult | MorphologyResult | ObservationMetadata | Medication)


class FactSourceSpan(Strict):
    segment_id: str
    source_field: SourceField
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)


class SemanticFact(Strict):
    mapping_id: str
    fact_id: str
    family: str
    concept: str
    fact_type: Literal["qualitative_result", "numeric_result", "ordinal_result", "titer_result", "ana_pattern",
                       "immunofixation_finding", "cryoglobulin_finding", "m_spike_result", "hla_result",
                       "morphology_result", "observation_metadata", "medication_context"]
    payload: Payload
    temporality: Temporality
    segment_id: str
    source_field: SourceField
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)
    strategy: str
    rule_or_prompt_version: str
    duplicate_source_spans: list[FactSourceSpan] = Field(default_factory=list)
    document_ids: list[str]
    schema_version: str

    @model_validator(mode="after")
    def match_payload(self):
        if type(self.payload) is not FACT_PAYLOADS[self.fact_type] or self.end - self.start != len(self.quote):
            raise ValueError("Fact type, payload or offsets disagree")
        return self

    def storage(self) -> dict:
        row = self.model_dump(exclude={"payload"})
        payload = self.payload.model_dump()
        return dict(row, **{name: payload if name == self.fact_type else None for name in FACT_PAYLOADS})


SEGMENTS = pa.schema([(name, INT if name in {"start", "end"} else STR) for name in SemanticSegment.model_fields])
NUMERIC_PAYLOAD = pa.struct([(name, STR) for name in NumericResult.model_fields])
PAYLOAD_OVERRIDES = {
    "numeric_result": dict.fromkeys(NumericResult.model_fields, STR),
    "m_spike_result": {"concentration": NUMERIC_PAYLOAD},
    "morphology_result": {"percentage": NUMERIC_PAYLOAD},
    "ordinal_result": {"score": INT},
    "titer_result": {"numerator": INT, "denominator": INT},
    "hla_result": {"resolution": INT, "position": INT, "ambiguous": BOOL, "alternatives": pa.list_(STR)},
    "immunofixation_finding": {field.name: field.type for field in FINDING},
    "cryoglobulin_finding": {field.name: field.type for field in FINDING},
}
FACT_SOURCE_SPANS = pa.list_(pa.struct([(name, INT if name in {"start", "end"} else STR)
                                        for name in FactSourceSpan.model_fields]))
FACTS = pa.schema([
    *[(name, INT if name in {"start", "end"} else pa.list_(STR) if name == "document_ids" else
       FACT_SOURCE_SPANS if name == "duplicate_source_spans" else STR)
      for name in SemanticFact.model_fields if name != "payload"],
    *[(name, pa.struct([(field, PAYLOAD_OVERRIDES.get(name, {}).get(field, STR)) for field in model.model_fields]))
      for name, model in FACT_PAYLOADS.items()],
])


class SpanProposal(Strict):
    segment_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)
    document_ids: list[str]


class AnaProposal(SpanProposal):
    payload: AnaPattern


class HlaProposal(SpanProposal):
    payload: HlaResult


class MorphologyProposal(SpanProposal):
    payload: MorphologyResult


class AnaExtraction(Strict):
    candidates: list[AnaProposal]


class HlaExtraction(Strict):
    candidates: list[HlaProposal]


class MorphologyExtraction(Strict):
    candidates: list[MorphologyProposal]


EXTRACTIONS = EXTRACTIONS.append(pa.field("contribution_status", STR)).append(pa.field("facts", pa.list_(pa.struct(FACTS))))

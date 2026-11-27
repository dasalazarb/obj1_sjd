"""Materialize Effective values from the Step 10 clinical episode table."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

from .inventory import CONTEXT_FIELDS, LAB, WIDE_CONTEXT, check_keys, clean, protected, route
from .parsers import METHOD_OR_COMMENT, PLANNED, format_hla_result, format_urine_result, normalize, standalone_limit
from .schemas import BOOL, INT, LINEAGE, STR
from .segmentation import ADMIN, CUTOFF, HEADINGS, HLA_LABEL, PROCESS, SPECIMEN
from .semantic_facts import clinical, coverage, fact_spans
from .settings import KEYS, serialize
from .table_exports import structured

EFFECTIVE = "10_integrated_longitudinal_clinical_episode_effective.parquet"
REDUNDANT_LAB_TEXT = frozenset("""
lab__acute_care_panel__anion_gap_blood__text
lab__acute_care_panel__chloride_blood__text
lab__acute_care_panel__co2_total_blood__text
lab__acute_care_panel__potassium_blood__text
lab__acute_care_panel__sodium_blood__text
lab__albumin__text
lab__alkaline_phosphatase__text
lab__alpha1_globulin__text
lab__alpha2_globulin__text
lab__alt__text
lab__amylase__text
lab__anc__text
lab__ast__text
lab__basophil_absolute__text
lab__basophil_percent__text
lab__beta1_globulin__text
lab__beta2_globulin__text
lab__beta_globulin__text
lab__calcium__text
lab__cbc_diff__variant_lymphocytes_blood__text
lab__complement_c3__text
lab__creatine_kinase__text
lab__creatinine__text
lab__eosinophil_absolute__text
lab__eosinophil_percent__text
lab__esr__text
lab__estimated_average_glucose__text
lab__ferritin__text
lab__gamma_globulin__text
lab__glucose__text
lab__hdl_cholesterol__text
lab__hematocrit__text
lab__hemoglobin__text
lab__hemoglobin_a1c__text
lab__igg__text
lab__igg_subclass_1__text
lab__igg_subclass_2__text
lab__igg_total__text
lab__igm__text
lab__immature_granulocyte_absolute__text
lab__immature_granulocyte_percent__text
lab__immunofixation_electrophoresis_serum__albumin_ord_electrophoresis_pro_or_ife_blood__text
lab__iron_saturation__text
lab__ldl_cholesterol_calculated__text
lab__lymphocyte_count__text
lab__lymphocyte_percent__text
lab__magnesium__text
lab__mch__text
lab__mchc__text
lab__mcv__text
lab__monocyte_absolute__text
lab__monocyte_percent__text
lab__mpv__text
lab__neutrophil_percent__text
lab__nucleated_rbc_absolute__text
lab__nucleated_rbc_percent__text
lab__phosphorus__text
lab__platelet_count__text
lab__protein_total__text
lab__protein_total_electrophoresis__text
lab__quantiferon_nil__text
lab__quantiferon_tb1_minus_nil__text
lab__quantiferon_tb2_minus_nil__text
lab__rbc__text
lab__rdw__text
lab__serum_iron__text
lab__total_cholesterol__text
lab__transferrin__text
lab__triglycerides__text
lab__uric_acid__text
lab__urine_ph__text
lab__urine_squamous_cells__text
lab__wbc__text
""".split())
RULE = "effective-value-v2"
ORIGINS = ("current_validated", "previous_validated", "original_fallback", "original_partial",
           "original_unchanged", "original_null")
CORRECTED = frozenset(ORIGINS[:2])
EMPTY_RESULTS = {
    "improperly collected specimen not run", "specimen clotted", "invalid specimen",
    "insufficient sample", "insufficient specimen", "specimen not run", "not tested",
    "test canceled", "test cancelled", "canceled", "result pending",
}
URINE_PREFIXES = ("urine_", "urinalysis_includes_microscopic__")
TEXT_TYPES = (pa.types.is_string, pa.types.is_large_string, pa.types.is_string_view)
RESIDUAL = pa.list_(pa.struct([("source_field", STR), ("start", INT), ("end", INT), ("text", STR), ("role", STR)]))
EVIDENCE = pa.list_(pa.struct([("fact_id", STR), ("source_field", STR), ("start", INT), ("end", INT),
                              ("quote", STR), ("rule_or_prompt_version", STR), ("reference_run_id", STR),
                              ("document_ids", pa.list_(STR))]))
VALUE_FIELDS = [
    *[(k, STR) for k in ("variable", "family", "source_type", "raw_type", "raw_json", "effective_type",
                        "effective_json", "effective_origin", "reason", "extraction_status", "temporal_availability",
                        "source_run_id", "rule_version", "decision_fact_id")],
    ("harmonized_fact_ids", pa.list_(STR)), ("accepted_fact_ids", pa.list_(STR)), ("clinical_fact_ids", pa.list_(STR)),
    ("evidence", EVIDENCE), ("residual_segments", RESIDUAL),
]
OBSERVATIONS = pa.schema([*LINEAGE, *VALUE_FIELDS, ("linked_availability_date", STR), ("availability_source", STR),
                          ("as_of_fact_ids", pa.list_(STR))])
AUDIT = pa.schema([("mapping_id", STR), ("source_grain", STR), ("context_json", STR), *VALUE_FIELDS, ("occurrences", INT)])
METRICS = ("cells", "source_null", "source_nonnull", *ORIGINS, "typed_cells", "clinical_fact_cells", "verified_temporal_cells",
           "available_as_of_cells", "retained_exact_cells", "effective_nonnull", "new_nulls", "integrity_failures")
SUMMARY = pa.schema([("variable", STR), ("family", STR), ("source_grain", STR), *[(k, INT) for k in METRICS]])
FEATURES = pa.schema([("variable", STR), ("arrow_type", STR), ("nullable", BOOL), ("role", STR), ("family", STR),
                      ("transformation", STR), ("temporal_availability", STR), ("allowed_use", STR),
                      ("support", STR), *[(k, INT) for k in METRICS]])


def wire_type(dtype):
    """Use Arrow's schema for interpretation, including dictionary values."""
    return dtype.value_type if pa.types.is_dictionary(dtype) else dtype


def scalar_value(value, dtype):
    """Encode Arrow temporal values by exact ticks and preserve compound cell structure."""
    if not value.is_valid:
        return None
    if pa.types.is_dictionary(dtype):
        value = value.value
    dtype = wire_type(dtype)
    if pa.types.is_temporal(dtype):
        bits = pa.int32() if pa.types.is_date32(dtype) or pa.types.is_time32(dtype) else pa.int64()
        return value.cast(bits).as_py()
    if pa.types.is_map(dtype):
        return [[scalar_value(pair[0], dtype.key_type), scalar_value(pair[1], dtype.item_type)]
                for pair in value.values]
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        return [scalar_value(item, dtype.value_type) for item in value.values]
    if pa.types.is_struct(dtype):
        return [scalar_value(value[i], field.type) for i, field in enumerate(dtype)]
    return structured(value.as_py())


def decode_value(encoded, dtype):
    """Recover an Arrow scalar using the original Parquet field type."""
    def decode(value, target):
        if value is None:
            return None
        target = wire_type(target)
        if pa.types.is_temporal(target):
            # Python datetime truncates nanoseconds. Arrow accepts exact ticks.
            return value
        if pa.types.is_map(target):
            return [(decode(key, target.key_type), decode(item, target.item_type)) for key, item in value]
        if pa.types.is_list(target) or pa.types.is_large_list(target) or pa.types.is_fixed_size_list(target):
            return [decode(item, target.value_type) for item in value]
        if pa.types.is_struct(target):
            return {field.name: decode(item, field.type) for field, item in zip(target, value, strict=True)}
        if pa.types.is_decimal(target):
            return Decimal(value)
        if pa.types.is_binary(target) or pa.types.is_large_binary(target) or pa.types.is_fixed_size_binary(target):
            return bytes.fromhex(value)
        if pa.types.is_floating(target) and isinstance(value, str):
            return float(value)
        return value

    return pa.scalar(decode(json.loads(encoded), dtype), type=dtype)


def encode_value(scalar, dtype):
    return serialize(scalar_value(scalar, dtype))


@dataclass(frozen=True)
class Decision:
    origin: str
    reason: str
    value: str | None = None
    fact_id: str | None = None


def lexical_rules(mapping, rules):
    """Return only context-routed, whole-token, lossless replacements."""
    family = mapping["family"]
    if family in {"ana", "ana_narrative", "qualitative", "urine_graded", "cryoglobulin"}:
        return {key: value for key, value in rules["qualitative"].items()
                if key not in {"repeat reactive", "repeated reactive"}}
    if family == "urine_nominal":
        return {term: term for term in rules["nominal"].get(mapping["concept"], ())}
    if family in {"urine_microscopy", "cbc_morphology"}:
        return rules["microscopy"]
    return {}


def choose_effective(raw, dtype, mapping, parts, facts, historical, rules, *, compact=False):
    """Select the same-type Effective rendering of a source cell."""
    if not raw.is_valid:
        return Decision("original_null", "source null")
    if compact:
        text = raw.as_py()
        token = normalize(text).strip(" .:")
        if standalone_limit(text.strip()):
            return Decision("original_unchanged", "literal result", text)
        if not token:
            return Decision("original_unchanged" if not text else "current_validated", "empty result", "")
        attached_comment = ADMIN.match(text.strip()) and any(
            json.loads(mapping["context_json"]).get(name) for name in ("observation_comment", "observation_note")
        )
        if attached_comment:
            return Decision("original_fallback", "requires clinical extraction")
        if (token in EMPTY_RESULTS or token in rules["availability"] or token in rules["specimen"]
                or token in rules["administrative"] or PROCESS.fullmatch(token) or SPECIMEN.fullmatch(token)):
            return Decision("current_validated", "non-result", "")
        if mapping["family"] == "hla" and (value := format_hla_result(text)) is not None:
            return Decision("current_validated", "HLA nomenclature", value)
        if (METHOD_OR_COMMENT.search(text) or ADMIN.match(text.strip()) or CUTOFF.match(text) or PLANNED.search(text)
                or any(match.lastgroup in {"reference", "method", "general"} for match in HEADINGS.finditer(text))
                or mapping["family"] == "hla" and HLA_LABEL.match(text)
                or re.search(r"[.!?]\s+[A-Za-z]|\n\s*\S", text)
                or len(text) > 85 or len(text.split()) > 11):
            return Decision("original_fallback", "requires clinical extraction")
        value = format_effective_text(text, mapping, rules)
        return Decision("original_unchanged" if value == text else "current_validated", "concise result", value)

    populated = [fact for fact in facts if clinical(fact)]
    if any(check(dtype) for check in TEXT_TYPES):
        text = raw.as_py()
        # Scope transformations to complete lexemes. Unicode compatibility folding
        # and comments with metadata are still available as descriptive evidence.
        token = normalize(text) if text.isascii() else text
        if mapping["family"] == "urine_nominal":
            token = token.replace("-", " ")
        target = lexical_rules(mapping, rules).get(token)
        body = [part for part in parts if part.source_field == "raw_text"]
        incompatible_context = any(part.role in {"specimen_information", "historical_reference"}
                                   or part.role == "administrative_status" and part.role_rule != "comment_marker"
                                   for part in parts if part.source_field != "raw_text")
        if (target and len(body) == 1 and body[0].role == "patient_result" and body[0].text == text
                and not incompatible_context and mapping.get("specimen_state") is None
                and mapping.get("result_availability") in (None, "reported")):
            def supports(fact):
                payload = fact.payload
                category = getattr(payload, "category", None) or getattr(payload, "descriptor", None)
                return (fact.fact_type in {"qualitative_result", "morphology_result"} and category == target
                        and fact.temporality == "current" and fact.source_field == "raw_text"
                        and text[fact.start:fact.end] == fact.quote and fact.quote.strip() == text.strip())

            supporting = [fact for fact in populated if supports(fact)]
            if supporting and all(supports(fact) or fact.fact_type == "ana_pattern"
                                  and fact.payload.state == target for fact in populated):
                if target == text:
                    return Decision("original_unchanged", "already canonical")
                current = [fact for fact in supporting if fact.fact_id not in historical]
                fact = (current or supporting)[0]
                return Decision("current_validated" if current else "previous_validated",
                                "complete source lexeme under current assay rule", target, fact.fact_id)

    if mapping["family"] in {"derived", "other", "administrative"}:
        return Decision("original_unchanged", "no effective transformation for this source role")
    if populated:
        scopes = coverage([part for part in parts if part.role in {"patient_result", "unresolved"}
                           and part.role_rule != "transport"], facts)
        if any(not supported or gaps for supported, gaps in scopes.values()):
            return Decision("original_partial", "supported facts leave source text unresolved")
        return Decision("original_fallback", "supported facts retained separately without a lossless same-type rendering")
    return Decision("original_fallback", "no complete validated correction")


def format_effective_text(text, mapping, rules):
    """Apply orthographic rules to a standalone or freshly extracted result."""
    if not text.strip():
        return ""
    if standalone_limit(text.strip()):
        return text
    if mapping["family"] == "hla":
        formatted = format_hla_result(text)
        return text if formatted is None else formatted
    if mapping["concept"].startswith(URINE_PREFIXES):
        return format_urine_result(text, rules["urine_effective"])
    return text


class EffectiveText:
    """Reuse the same decisions across episode cells and source observations."""

    def __init__(self, cfg, extract):
        self.cfg = cfg
        self.extract = extract
        self.values = {}

    def resolve(self, requests):
        """Resolve each variable, original text, and assay context once."""
        rules = self.cfg["normalization"]
        pending = {}
        for key in dict.fromkeys(requests):
            if key in self.values:
                continue
            variable, text, context = key
            concept, family = route(variable, self.cfg)
            row = dict(variable=variable, concept=concept, family=family, context_json=context)
            choice = choose_effective(pa.scalar(text, type=STR), STR, row, (), (), {}, rules, compact=True)
            if text is not None and choice.value is None:
                pending[key] = dict(row, raw_text=text)
            else:
                self.values[key] = choice.value
        if pending:
            for key, row, answer in zip(pending, pending.values(), self.extract(list(pending.values())), strict=True):
                self.values[key] = format_effective_text(answer, row, rules)


def replace_values(table, name, values):
    """Replace a column using its original Arrow field and physical type."""
    index = table.schema.get_field_index(name)
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def update_effective(original, active, context, text):
    """Build the active wide table without republishing historical sidecars."""
    keys = check_keys(original.select(KEYS).to_pandas(), "Step 10")
    if not keys.equals(check_keys(active.select(KEYS).to_pandas(), "Effective")):
        raise ValueError("Effective wide episode keys or row order differ from Step 10")
    if context is not None:
        context = context.set_index(KEYS).reindex(keys)

    def context_values(name):
        if name in original.column_names:
            return original[name].to_pylist()
        if context is not None and name in context:
            return context[name].tolist()
        return [None] * len(active)

    wide = active.select([name for name in active.column_names if name not in REDUNDANT_LAB_TEXT])
    metadata = wide.schema.metadata
    if metadata and b"pandas" in metadata and wide.num_columns != active.num_columns:
        pandas_schema = json.loads(metadata[b"pandas"])
        pandas_schema["columns"] = [column for column in pandas_schema["columns"]
                                    if column["field_name"] in wide.column_names]
        wide = wide.replace_schema_metadata(metadata | {b"pandas": serialize(pandas_schema).encode()})
    requests = {}
    for field in wide.schema:
        match = LAB.fullmatch(field.name)
        if not match or match[2] != "text" or not any(check(wire_type(field.type)) for check in TEXT_TYPES):
            continue
        column_context = {key: context_values(f"lab__{match[1]}__{suffix}") for key, suffix in WIDE_CONTEXT.items()}
        requests[field.name] = [
            (field.name, value, serialize({key: clean(column_context[key][i]) if key in column_context else None
                                          for key in CONTEXT_FIELDS}))
            for i, value in enumerate(wide[field.name].to_pylist())
        ]
    text.resolve(key for column in requests.values() for key in column)
    for name, column in requests.items():
        values = [text.values[key] for key in column]
        if any(value != key[1] for value, key in zip(values, column, strict=True)):
            wide = replace_values(wide, name, values)
    return wide


def residual_segments(parts, facts):
    spans = coverage(parts, facts)
    return [dict(source_field=part.source_field, role=part.role, **gap)
            for part in parts for gap in spans[part.segment_id][1]]


def count_cell(counts, row, source_null):
    counts["cells"] += 1
    counts[row["effective_origin"]] += 1
    counts["source_null" if source_null else "source_nonnull"] += 1
    counts["effective_nonnull"] += not source_null
    counts["retained_exact_cells"] += row["effective_origin"] not in CORRECTED
    counts["typed_cells"] += bool(row["accepted_fact_ids"])
    counts["clinical_fact_cells"] += bool(row["clinical_fact_ids"])
    counts["verified_temporal_cells"] += row["temporal_availability"] in {"verified_result_date", "available_by_episode"}
    counts["available_as_of_cells"] += row["temporal_availability"] == "available_by_episode"


def typed_equal(left, right):
    """Compare Arrow values with a separate null bitmap and NaN-safe equality."""
    if left.equals(right):
        return True
    if not left.is_null().equals(right.is_null()):
        return False
    dtype = left.type
    return all(a.equals(b) or encode_value(a, dtype) == encode_value(b, dtype)
               for a, b in zip(left, right, strict=True))


def build_effective(original, source, lineage, dictionary, facts, segments, cfg, run_id, analytic, qc,
                    *, historical=None, outputs=None, routes=None):
    """Select once per mapping and Arrow value, then write linked wide, long and QC files."""
    historical, outputs, routes = historical or {}, outputs or {}, routes or {}
    mapping_index = {row["mapping_id"]: row for row in dictionary}
    fact_index, segment_index, lines = defaultdict(list), defaultdict(list), defaultdict(list)
    for fact in facts:
        fact_index[fact.mapping_id].append(fact)
    for part in segments:
        segment_index[part.mapping_id].append(part)
    for line in lineage:
        row = mapping_index[line["mapping_id"]]
        group = row["variable"] if line["source_grain"] == "episode_variable" else None
        lines[group].append(line)
    keys = [original[name].to_pylist() for name in KEYS]
    linked = {}
    for batch in pq.ParquetFile(analytic / "patient_visit_semantic_observations.parquet").iter_batches(
        columns=["observation_id", "verified_available_by_episode", "as_of_fact_ids", "result_available_date", "availability_source"]
    ):
        linked.update((row["observation_id"], row) for row in batch.to_pylist()
                      if row["verified_available_by_episode"] or row["as_of_fact_ids"] or row["availability_source"])

    audit, groups, families, totals = {}, defaultdict(Counter), defaultdict(Counter), Counter()
    decisions, manifests, columns = {}, [], []
    with pq.ParquetWriter(analytic / "effective_observations.parquet", OBSERVATIONS) as writer:
        def materialize(column, column_lines, grain, *, fixed_variable=None):
            if [line["source_row"] for line in column_lines] != list(range(len(column))):
                raise ValueError("Effective observation rows do not cover their original source grain")
            dtype = column.type
            changed, batch = {}, []
            for i, (raw, line) in enumerate(zip(column, column_lines, strict=True)):
                key = line["mapping_id"]
                mapping = mapping_index[key]
                variable, family = mapping["variable"], mapping["family"]
                if fixed_variable is not None and (variable != fixed_variable or
                        any(str(values[i]) != line[name] for name, values in zip(KEYS, keys, strict=True))):
                    raise ValueError("Effective observation key differs from the source episode")
                raw_json = encode_value(raw, dtype)
                decision_key = key, str(dtype), raw_json, grain
                if decision_key not in decisions:
                    parts, accepted = segment_index[key], fact_index[key]
                    decision = choose_effective(raw, dtype, mapping, parts, accepted, historical, cfg["normalization"])
                    encoded = raw_json if decision.origin not in CORRECTED else serialize(decision.value)
                    decoded = decode_value(encoded, dtype)
                    if encode_value(decoded, dtype) != encoded or (
                        decision.origin in CORRECTED and (not decoded.is_valid or not raw.is_valid)
                    ):
                        raise ValueError("Effective value failed Arrow round-trip")
                    reference = historical.get(decision.fact_id)
                    details = dict(
                        variable=variable, family=family, source_type="Step 10" if grain == "episode_variable" else "BTRIS",
                        raw_type=str(dtype), raw_json=raw_json, effective_type=str(dtype), effective_json=encoded,
                        effective_origin=decision.origin, reason=decision.reason,
                        rule_version=RULE, decision_fact_id=decision.fact_id, source_run_id=reference or run_id,
                        extraction_status=outputs.get(key, {}).get("contribution_status") or
                                          outputs.get(key, {}).get("status") or routes.get(key, {}).get("reason") or "not recorded",
                        harmonized_fact_ids=[f.fact_id for f in accepted if f.fact_id not in historical],
                        accepted_fact_ids=[f.fact_id for f in accepted],
                        clinical_fact_ids=[f.fact_id for f in accepted if clinical(f)],
                        evidence=[dict(fact_id=f.fact_id, source_field=span.source_field, start=span.start,
                                       end=span.end, quote=span.quote, rule_or_prompt_version=f.rule_or_prompt_version,
                                       reference_run_id=historical.get(f.fact_id), document_ids=f.document_ids)
                                  for f in accepted for span in fact_spans(f)],
                        residual_segments=residual_segments(parts, accepted),
                    )
                    decisions[decision_key] = details, decoded if decision.origin in CORRECTED else None
                details, replacement = decisions[decision_key]
                if replacement is not None:
                    changed[i] = replacement
                timing = linked.get(line["observation_id"], {})
                availability = ("available_by_episode" if timing.get("as_of_fact_ids") else "verified_result_date"
                                if timing.get("verified_available_by_episode") else "unverified")
                row = dict(line, **details, temporal_availability=availability,
                           linked_availability_date=timing.get("result_available_date"),
                           availability_source=timing.get("availability_source"),
                           as_of_fact_ids=timing.get("as_of_fact_ids", []))
                batch.append(row)
                count_cell(groups[(variable, family, grain)], row, not raw.is_valid)
                count_cell(families[(family, grain)], row, not raw.is_valid)
                count_cell(totals, row, not raw.is_valid)
                audit_key = (*decision_key, availability)
                if audit_key not in audit:
                    audit[audit_key] = dict(details, mapping_id=key, source_grain=grain,
                                            context_json=mapping["context_json"], temporal_availability=availability,
                                            occurrences=0)
                audit[audit_key]["occurrences"] += 1
                if len(batch) == 8192:
                    writer.write_table(pa.Table.from_pylist(batch, schema=OBSERVATIONS))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=OBSERVATIONS))
            if not changed:
                return column
            return pa.array([changed.get(i, raw) for i, raw in enumerate(column)], type=dtype)

        for field in original.schema:
            name = field.name
            policy = protected(name, cfg)
            peers = lines.pop(name, [])
            column = original[name]
            if policy:
                if peers:
                    raise ValueError("Protected column was inventoried for effective transformation")
                n = column.null_count
                info = Counter(cells=len(column), source_null=n, source_nonnull=len(column)-n,
                               original_null=n, original_unchanged=len(column)-n, retained_exact_cells=len(column),
                               effective_nonnull=len(column)-n)
                role, family = "protected", None
                totals.update(info)
            else:
                column = materialize(column, peers, "episode_variable", fixed_variable=name)
                family = mapping_index[peers[0]["mapping_id"]]["family"] if peers else None
                role, info = "literal_observation", groups[(name, family, "episode_variable")]
            columns.append(column)
            manifests.append(dict(variable=name, arrow_type=str(field.type), nullable=field.nullable,
                                  role=role, family=family, transformation=RULE if not policy else "identity",
                                  temporal_availability="observation-specific" if not policy else "source contract",
                                  allowed_use="literal/descriptive; use linked validated features for prediction",
                                  support="effective_observations.parquet" if not policy else "Step 10",
                                  **{k: info[k] for k in METRICS}))
        if source is not None:
            materialize(source["result_raw"], lines.pop(None, []), "btris_observation")
        if lines:
            raise ValueError("Effective lineage contains unexpected variables or source grains")

    wide = pa.Table.from_arrays(columns, schema=original.schema)
    pq.write_table(wide, analytic / EFFECTIVE)
    written = pq.read_table(analytic / EFFECTIVE)
    if not original.schema.equals(written.schema, check_metadata=True) or original.num_rows != written.num_rows:
        raise ValueError("Effective wide changed the original Arrow schema or row count")
    for field in original.schema:
        if (not original[field.name].is_null().equals(written[field.name].is_null())
                or not typed_equal(wide[field.name], written[field.name])):
            raise ValueError("Effective wide changed after Parquet round-trip")

    original_columns = {name: original[name] for name in original.column_names}
    written_columns = {name: written[name] for name in written.column_names}
    source_values = source["result_raw"] if source is not None else None
    long_counts = Counter()
    for batch in pq.ParquetFile(analytic / "effective_observations.parquet").iter_batches(
        columns=["source_grain", "source_row", "variable", "raw_json", "effective_json", "effective_origin"]
    ):
        for row in batch.to_pylist():
            name, i, grain = row["variable"], row["source_row"], row["source_grain"]
            if i != long_counts[(grain, name) if grain == "episode_variable" else (grain, None)]:
                raise ValueError("Effective observation count or order differs from source")
            long_counts[(grain, name) if grain == "episode_variable" else (grain, None)] += 1
            old = original_columns[name] if grain == "episode_variable" else source_values
            new = written_columns[name] if grain == "episode_variable" else old
            raw_record = row["raw_json"]
            if (raw_record != encode_value(old[i], old.type)
                    or grain == "episode_variable" and row["effective_json"] != encode_value(new[i], new.type)
                    or row["effective_origin"] not in CORRECTED and raw_record != row["effective_json"]):
                raise ValueError("Effective long disagrees with its raw or wide value")
    expected = sum(item["cells"] for item in manifests if item["role"] == "literal_observation")
    if sum(long_counts.values()) != expected + (source.num_rows if source is not None else 0):
        raise ValueError("Effective wide and long cell denominators differ")

    for folder, filename, rows, schema in (
        (analytic, "effective_feature_manifest", manifests, FEATURES),
        (qc, "effective_value_audit", list(audit.values()), AUDIT),
        (qc, "effective_variable_summary",
         [dict(variable=name, family=family, source_grain=grain, **{k: counts[k] for k in METRICS})
          for (name, family, grain), counts in groups.items()], SUMMARY),
        (qc, "effective_family_summary",
         [dict(family=family, source_grain=grain, **{k: counts[k] for k in METRICS})
          for (family, grain), counts in families.items()], SUMMARY.remove(0)),
    ):
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), folder / f"{filename}.parquet")
    return {key: totals[key] for key in METRICS} | {
        "episode_cells": original.num_rows * original.num_columns,
        "source_event_cells": source.num_rows if source is not None else 0,
        "protected_cells": sum(item["cells"] for item in manifests if item["role"] == "protected"),
    }

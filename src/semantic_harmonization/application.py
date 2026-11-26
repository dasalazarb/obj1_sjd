"""Apply the semantic dictionary without replacing source columns."""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
import json
import math
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .effective_values import typed_equal
from .inventory import clean
from .schemas import ALLELE, CHILD_LINK, FEATURES, FINDING, MEDICATION, OBSERVATIONS, SEMANTIC
from .semantic_facts import clinical
from .settings import KEYS, serialize

WIDE_FIELDS = {
    "qualitative_result": pa.string(), "exact_value": pa.float64(), "limit_value": pa.float64(),
    "comparator": pa.string(), "range_low": pa.float64(), "range_high": pa.float64(),
    "raw_unit": pa.string(), "ucum_unit": pa.string(), "titer_numerator": pa.int64(),
    "titer_denominator": pa.int64(), "titer_comparator": pa.string(), "category": pa.string(),
    "ordinal_grade": pa.string(), "ordinal_score": pa.int64(), "descriptor": pa.string(), "specimen_state": pa.string(),
    "result_availability": pa.string(),
}
SEMANTIC_FACTS = [*WIDE_FIELDS, "pattern_terms", "findings", "alleles", "medications"]


def write_parquet(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    """Write a Parquet artifact with an explicit typed schema."""
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def add_column(table: pa.Table, name: str, values: list, dtype: pa.DataType) -> pa.Table:
    """Append a feature while preventing collision with original columns."""
    if name in table.column_names:
        raise ValueError(f"Harmonized feature collides with an input column: {name}")
    return table.append_column(name, pa.array(values, type=dtype))


def project_float(value: str | None) -> float | None:
    """Project a decimal to wide only if its float round-trip is equivalent."""
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and Decimal(str(numeric)) == Decimal(value) else None


def feature(name: str, source: str, field: str, dtype: pa.DataType, availability: str = "unknown") -> dict:
    """Record the use boundary and provenance of a candidate feature."""
    return dict(name=name, source=source, kind=field, dtype=str(dtype), units="see same-variable unit fields"
                if pa.types.is_floating(dtype) else None, temporal_availability=availability,
                allowed_use="as_of" if availability == "verified_result_date" else "descriptive",
                lineage="harmonized_observations.parquet",
                analytic_eligibility_rule="Candidate field; requires study-specific admissibility")


IMMUNOFIXATION_FIELDS = {
    ("IgG", "kappa"): "igg_kappa_present", ("IgG", "lambda"): "igg_lambda_present",
    ("IgM", "kappa"): "igm_kappa_present",
}
FINDING_WIDE_FIELDS = (*IMMUNOFIXATION_FIELDS.values(), "free_lambda_present", "polyclonal_pattern",
                       "oligoclonal_pattern", "abnormal_band_present")


def immunofixation_fields(findings: list[dict]) -> dict[str, bool]:
    """Project explicit entities and documented absence within their stated scope."""
    values = defaultdict(set)
    for item in findings:
        if item["temporality"] not in {"current", "unspecified"} or item["assertion"] == "uncertain":
            continue
        present = item["assertion"] == "present"
        if field := IMMUNOFIXATION_FIELDS.get((item["immunoglobulin_class"], item["light_chain"])):
            values[field].add(present)
        if item["entity"] == "light_chain" and item["light_chain"] == "lambda" and re.search(r"\bfree\b", item["quote"], re.I):
            values["free_lambda_present"].add(present)
        if item["pattern"] in {"polyclonal", "oligoclonal"}:
            values[f'{item["pattern"]}_pattern'].add(present)
        if item["entity"] == "anomalous_immunoglobulins" or (
            present and item["entity"] in {"immunoglobulin", "light_chain"} and (
                item["pattern"] == "monoclonal" or re.search(r"\babnormal band\b", item["quote"], re.I)
            )
        ):
            values["abnormal_band_present"].add(present)
    return {field: next(iter(status)) for field, status in values.items() if len(status) == 1}


def include_typed_projection(observation: dict, facts) -> None:
    """Materialize profile enrichments where they preserve existing scalar evidence."""
    profiles = defaultdict(list)
    for fact in facts:
        profiles[fact.fact_type].append(fact)
    patterns = list(dict.fromkeys([*observation["pattern_terms"], *(
        fact.payload.pattern for fact in profiles["ana_pattern"] if clinical(fact) and fact.payload.pattern
    )]))
    observation["pattern_terms"] = patterns
    for fact in profiles["hla_result"]:
        if fact.strategy != "llm_rag":
            continue
        payload = fact.payload
        allele = dict(locus=payload.locus, reported=payload.token, allele=payload.allele,
                      resolution_fields=payload.resolution, suffix=payload.suffix,
                      source_order=payload.position, ipd_match=payload.ipd_match, lookup_locus=payload.lookup_locus,
                      normalized_report=payload.normalized_report, group=payload.group)
        if allele not in observation["alleles"]:
            observation["alleles"] = [*observation["alleles"], allele]
    morph = profiles["morphology_result"]
    if any(fact.strategy == "llm_rag" for fact in morph):
        descriptors = {fact.payload.descriptor for fact in morph if fact.payload.descriptor is not None}
        if len(descriptors) == 1 and observation["category"] is None and observation["descriptor"] is None:
            observation["category"] = next(iter(descriptors))
        intervals = {serialize(fact.payload.percentage.model_dump()) for fact in morph if fact.payload.percentage}
        if len(intervals) == 1 and observation["comparator"] is None:
            observation.update(json.loads(next(iter(intervals))))



def apply(input_table: pa.Table, source: pa.Table | None, lineage: list[dict], dictionary: list[dict],
          analytic: Path, facts=(), *, project_wide=True) -> tuple[pa.Table, list[dict], list[dict]]:
    """Join every observation m:1 and materialize the typed sidecars."""
    lookup = {row["mapping_id"]: row for row in dictionary}
    if len(lookup) != len(dictionary) or set(lookup) != {row["mapping_id"] for row in lineage}:
        raise ValueError("Semantic dictionary failed the m:1 mapping contract")
    if len({row["observation_id"] for row in lineage}) != len(lineage):
        raise ValueError("Observation lineage keys must be unique")
    by_mapping = defaultdict(list)
    for fact in facts:
        by_mapping[fact.mapping_id].append(fact)
    observations, source_rows, episode_variables = [], [], defaultdict(list)
    children = {field: [] for field in ("findings", "alleles", "medications")}
    for line in lineage:
        mapping = lookup[line["mapping_id"]]
        values = {name: mapping[name] for name in SEMANTIC.names}
        observation = dict(line, **values, **{name: mapping[name] for name in
                                             ("variable", "concept", "family", "raw_type", "raw_json", "context_json")})
        include_typed_projection(observation, by_mapping[line["mapping_id"]])
        if mapping["family"] == "immunofixation":
            observation.update(immunofixation_fields(observation["findings"]))
        observations.append(observation)
        if line["source_grain"] == "btris_observation":
            source_rows.append(observation)
        else:
            episode_variables[mapping["variable"]].append(observation)
        for field, output in children.items():
            output.extend(dict(observation_id=line["observation_id"], mapping_id=line["mapping_id"],
                               ordinal=i, **item) for i, item in enumerate(observation[field]))
    write_parquet(analytic / "harmonized_observations.parquet", observations, OBSERVATIONS)
    for field, filename, schema in (
        ("findings", "semantic_findings", FINDING), ("alleles", "hla_alleles", ALLELE),
        ("medications", "medication_entities", MEDICATION),
    ):
        write_parquet(analytic / f"{filename}.parquet", children[field], pa.schema([*CHILD_LINK, *schema]))

    if source is not None:
        if [row["source_row"] for row in source_rows] != list(range(source.num_rows)):
            raise ValueError("BTRIS source order or grain changed")
        source_candidate = add_column(source, "harm_mapping_id", [row["mapping_id"] for row in source_rows], pa.string())
        for item in SEMANTIC:
            source_candidate = add_column(source_candidate, "harm_" + item.name, [row[item.name] for row in source_rows], item.type)
        target = analytic / "harmonized_source_events.parquet"
        pq.write_table(source_candidate, target)
        validate_original(source, pq.read_table(target))

    wide, features = input_table, []
    for variable, rows in episode_variables.items():
        if [row["source_row"] for row in rows] != list(range(wide.num_rows)):
            raise ValueError(f"Episode inventory alignment failed for {variable}")
        if not project_wide or rows[0]["family"] in {"other", "derived"} or variable == "spine__interval_name":
            continue
        for field, dtype in {**WIDE_FIELDS, "pattern_terms": pa.list_(pa.string()),
                             **dict.fromkeys(FINDING_WIDE_FIELDS, pa.bool_())}.items():
            values = [row.get(field) for row in rows]
            if field == "pattern_terms":
                values = [value or None for value in values]
            if pa.types.is_floating(dtype):
                values = [project_float(value) for value in values]
            if all(value is None for value in values):
                continue
            name = f"harm__{variable}__{field}"
            wide = add_column(wide, name, values, dtype)
            features.append(feature(name, variable, field, dtype))
    return wide, features, observations


def validate_original(before: pa.Table, after: pa.Table) -> None:
    """Verify row count, order, types, metadata and every original value."""
    names = before.column_names
    if before.num_rows != after.num_rows or after.column_names[:len(names)] != names:
        raise ValueError("Original column order or row count changed")
    after_original = after.select(names)
    if not before.schema.equals(after_original.schema, check_metadata=True):
        raise ValueError("Original Arrow schema or metadata changed")
    for name in names:
        if not typed_equal(before[name], after_original[name]):
            raise ValueError(f"Original Arrow values or null positions changed for {name}")


def timestamp(value):
    """Parse an explicit source timestamp without substituting an anchor date."""
    return pd.to_datetime(value, errors="coerce", utc=True)


def rebuild(master: pd.DataFrame, wide: pa.Table, observations: list[dict],
            features: list[dict]) -> tuple[pa.Table, list[dict]]:
    """Recompute supported longitudinal candidates from dated source evidence."""
    anchor = master["clinical_anchor_date"].map(timestamp) if "clinical_anchor_date" in master else pd.Series(pd.NaT, index=master.index)
    staged = defaultdict(list)
    for row in observations:
        if row["source_grain"] != "btris_observation" or row["result_valid"] is not True:
            continue
        date = timestamp(row["result_available_date"])
        measured = timestamp(row["measurement_date"])
        if pd.isna(date) or pd.isna(measured) or date < measured:
            continue
        staged[(row["patient_id"], row["concept"])].append((date, row))

    output = []
    for col in master:
        if not col.startswith(("sero__", "essdai__", "esspri__", "ext__", "ovl__", "pop__", "pro__")):
            continue
        after, evaluated = None, 0
        before = master[col]
        rule = "Existing producer value retained; required lineage or source components unavailable."
        is_ever = col.startswith("sero__") and col.endswith("__ever_positive_through_episode")
        is_hla = col.startswith("sero__hla_") and col.endswith("__asof_value")
        if is_ever or is_hla:
            concept = col.split("__")[1]
            values = []
            for pid, at in zip(master[KEYS[0]], anchor, strict=True):
                known = [(date, row) for date, row in staged[(str(pid), concept)] if not pd.isna(at) and date <= at]
                assay_keys = {serialize({field: json.loads(row["context_json"]).get(field)
                                         for field in ("assay_id", "method", "unit")}) for _, row in known}
                if len(assay_keys) > 1:
                    values.append(None)
                    continue
                if is_ever:
                    states = {row["qualitative_result"] for _, row in known
                              if row["specimen_state"] is None and row["result_availability"] == "reported"}
                    values.append(True if "positive" in states else False if "negative" in states else None)
                else:
                    dated = [(date, json.loads(row["raw_json"])) for date, row in known
                             if any(item["allele"] is not None for item in row["alleles"])]
                    latest = max((date for date, _ in dated), default=None)
                    alternatives = {raw for date, raw in dated if date == latest}
                    values.append(next(iter(alternatives)) if len(alternatives) == 1 else None)
            if any(value is not None for value in values):
                dtype = pa.bool_() if is_ever else pa.string()
                name = "harm__" + col
                wide = add_column(wide, name, values, dtype)
                after = pd.Series(values, dtype="boolean" if is_ever else "string")
                evaluated = int(after.notna().sum())
                scope = "Tri-state cumulative explicit serology" if is_ever else "Most recent compatible HLA report"
                rule = scope + " through each anchor using verified result availability."
                features.append(feature(name, col, "derived_recompute", dtype, "verified_result_date"))
        candidate = before if after is None else after
        source_values = [clean(value) for value in before]
        candidate_values = source_values if after is None else [clean(value) for value in candidate]

        def states(values):
            return ["null" if value is None else "true" if value is True or value == "True"
                    else "false" if value is False or value == "False" else "observed" for value in values]

        a, b = states(source_values), states(candidate_values)
        transitions = {f"{left} → {right}": n for (left, right), n in Counter(zip(a, b)).items()}
        output.append(dict(variable=col, recomputed=after is not None, evaluated_episodes=evaluated,
                           before=serialize(dict(Counter(a))), after=serialize(dict(Counter(b))),
                           transitions=serialize(transitions), rule=rule,
                           before_distinct=len({serialize(value) for value in source_values if value is not None}),
                           after_distinct=len({serialize(value) for value in candidate_values if value is not None})))
    return wide, output


def write_candidate(analytic: Path, wide: pa.Table, features: list[dict]) -> Path:
    """Write the standalone candidate and its feature manifest."""
    path = analytic / "10_integrated_longitudinal_clinical_episode_harmonized.parquet"
    pq.write_table(wide, path)
    write_parquet(analytic / "feature_manifest.parquet", features, FEATURES)
    return path

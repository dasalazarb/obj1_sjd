"""Write private reporting tables and render their offline explorer."""
from pathlib import Path
import os
from zipfile import ZIP_DEFLATED, ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from .application import write_parquet
from .analytic_representation import READINESS
from .closure_audit import COMMENT, COMMENT_SUMMARY, DERIVED, HISTORICAL_WORK, MATERIALIZATION, REGRESSION
from .effective_values import AUDIT as EFFECTIVE_AUDIT, SUMMARY as EFFECTIVE_SUMMARY
from .inference_audit import SMOKE_AUDIT, SMOKE_SUMMARY
from .report_progress import compare_runs
from .schemas import FACTS, ROUTES, SEMANTIC, STR, INT, BOOL
from .settings import MODULE, serialize, write_json
from .table_exports import export_csv

CANONICAL = pa.struct([FACTS.field("fact_type"), FACTS.field("temporality"),
                       *[f for f in FACTS if pa.types.is_struct(f.type)]])
FACT_LIST = pa.list_(pa.struct(list(FACTS)))
FEATURE = pa.struct([("name", STR), ("kind", STR), ("semantic_field", STR), ("observed", INT), ("values", pa.list_(STR)),
                     ("verified", INT), ("as_of", INT)])
MAPPING = pa.schema([
    *[(k, STR) for k in ("mapping_id", "baseline_mapping_id", "variable", "concept", "family", "raw_type", "raw_json",
                        "context_json", "recent_v2_mapping_id", "previous_v2_mapping_id", "previous_v2_change", "comparison_scope", "comparison_reason", "v1_strategy", "v2_strategy",
                        "change_kind", "actual_application", "initial_disposition", "final_disposition", "llm_outcome")],
    *[(k, BOOL) for k in ("v1_available", "v2_available", "comparable", "nonempty", "old_nonempty", "clinical_v1", "clinical_v2")],
    *[(k, INT) for k in ("episodes", "source_events", "occurrences", "patients", "baseline_episodes", "m1_m4_added", "m1_m4_removed")],
    *[(k, pa.struct(list(SEMANTIC))) for k in ("v1_legacy_fields", "v2_legacy_fields")],
    *[(k, pa.list_(CANONICAL)) for k in ("v1_canonical_facts", "m1_canonical_facts", "facts_added", "facts_removed", "facts_retained")],
    ("v2_typed_facts", FACT_LIST), ("previous_v2_typed_facts", FACT_LIST), ("recent_v2_typed_facts", FACT_LIST),
    ("effective_values", pa.list_(pa.struct(list(EFFECTIVE_AUDIT)))), ("evidence", SEMANTIC.field("evidence").type),
    *[(k, pa.list_(FEATURE)) for k in ("materialized_features", "baseline_features")],
    *[(k, pa.list_(STR)) for k in ("v2_fact_strategies", "pending_reason_codes", "knowledge_document_ids", "disposition_notes")],
    ("retrieval", pa.list_(pa.struct([("doc_id", STR), ("source", STR), ("source_version", STR), ("rank", INT)]))),
])
WORK = pa.schema([
    *[(k, STR) for k in ("work_id", "mapping_id", "variable", "family", "problem_kind", "component", "raw_type", "raw_json",
                        "context_summary", "source_field", "remaining_fragment", "state_original", "state_v1", "state_m1", "state_v2",
                        "reason_code", "reason_detail", "strategy_attempted", "llm_outcome", "required_component",
                        "next_action", "responsible_module", "evidence_source", "role", "role_rule", "work_type")],
    *[(k, INT) for k in ("segment_start", "segment_end", "episode_occurrences", "source_event_occurrences", "affected_patients", "occurrences")],
    *[(k, BOOL) for k in ("confirmed_initial", "comparable", "v2_available")],
    ("remaining_fragments", pa.list_(pa.struct([("start", INT), ("end", INT), ("text", STR)]))),
    ("current_structured_result", FACT_LIST), ("document_ids", pa.list_(STR)),
])
IMPACT = pa.schema([
    *[(key, STR) for key in ("study", "feature", "version", "scope", "eligibility_before", "eligibility_after", "status")],
    ("observed_before", INT), ("observed_after", INT),
])


def impact_rows(rows):
    """Adapt historical CSV numbers to the typed comparative contract."""
    for row in rows:
        yield {**row, **{key: int(float(row[key])) if row.get(key) not in (None, "", r"\N") else None
                         for key in ("observed_before", "observed_after")}}


def flatten(value, prefix=""):
    """Flatten summary metrics while preserving decimal numbers and nulls."""
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            yield from flatten(item, name)
        else:
            yield dict(metric=name, value=serialize(item))


def render(output, model):
    """Embed structured data with JSON-safe escaping in a single offline document."""
    payload = {key: value for key, value in model.items() if key != "routes"}
    private_keys = {"mapping_id", "segment_id", "fact_id"}
    def public_fact(fact):
        return {key: ([{k: v for k, v in span.items() if k != "segment_id"} for span in value]
                      if key == "duplicate_source_spans" and value else value)
                for key, value in fact.items() if key not in private_keys}
    payload["mappings"] = [
        row | {version: [public_fact(fact) for fact in row[version]] if row[version] is not None else None
               for version in ("previous_v2_typed_facts", "recent_v2_typed_facts", "v2_typed_facts")}
        for row in model["mappings"]
    ]
    payload.pop("effective_audit", None)
    payload["work"] = [{key: value for key, value in row.items() if key != "current_structured_result"}
                       for row in model["work"]]
    payload["materialization_audit"] = [{key: value for key, value in row.items() if key != "fact_id"}
                                        for row in model["materialization_audit"]]
    safe = serialize(payload).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    environment = Environment(loader=FileSystemLoader(MODULE / "templates"), autoescape=select_autoescape(["html"]))
    path = output / "semantic_harmonization_final.html"
    path.write_text(environment.get_template("report.html").render(payload=Markup(safe)), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def report(output: Path, cfg: dict, manifest: dict, *, target, baseline=None, routes=None, previous=None, recent=None,
           frozen=None):
    """Generate identical report artifacts for historical and future runs."""
    if target.manifest != manifest:
        raise ValueError("Report provenance differs from the target run")
    model = compare_runs(target, baseline, cfg, routes=routes, previous=previous, recent=recent, frozen=frozen)
    pending = [item for item in model["work"] if item["state_v2"] != "resolved"]
    tables = [
        *((name, model[key], schema) for key, name, schema in (
            ("effective_audit", "effective_value_audit", EFFECTIVE_AUDIT),
            ("effective_variables", "effective_variable_summary", EFFECTIVE_SUMMARY),
            ("effective_families", "effective_family_summary", EFFECTIVE_SUMMARY.remove(0)),
        ) if model[key] is not None and target.qc != output),
        ("mapping_comparison", model["mappings"], MAPPING), ("pending_work", pending, WORK),
        ("work_snapshot", model["work"], WORK), ("mapping_routing", model["routes"], ROUTES),
        ("comment_prefix_audit", model["comments"], COMMENT), ("comment_prefix_summary", model["comment_summary"], COMMENT_SUMMARY),
        ("semantic_materialization_audit", model["materialization_audit"], MATERIALIZATION),
        ("derived_rule_audit", model["derived_audit"], DERIVED), ("downstream_feature_readiness", model["readiness"], READINESS),
        ("regression_comparison", model["regressions"], REGRESSION),
        ("downstream_impact", list(impact_rows(model["downstream"])), IMPACT),
        ("historical_work_crosswalk", model["historical_work"], HISTORICAL_WORK),
        ("progress_summary", list(flatten(model["summary"])), pa.schema([("metric", STR), ("value", STR)])),
    ]
    for name, rows, schema in tables:
        path = output / f"{name}.parquet"
        write_parquet(path, rows, schema)
        export_csv(path)
    for key, name, fallback in (("variables", "variable_progress", ["variable", "state"]),
                                ("families", "family_progress", ["family", "variables"])):
        rows = model[key]
        table = pa.Table.from_pylist(rows) if rows else pa.table({name: pa.array([], type=STR) for name in fallback})
        path = output / f"{name}.parquet"
        pq.write_table(table, path)
        export_csv(path)
    if model["profile_summary"] is not None:
        archive = output / "inference_smoke_qc.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as qc:
            for key, name, schema in (
                ("smoke_audit", "inference_smoke_audit", SMOKE_AUDIT),
                ("profile_summary", "inference_profile_summary", SMOKE_SUMMARY),
            ):
                path = output / f"{name}.parquet"
                if target.qc != output:
                    write_parquet(path, model[key], schema)
                    export_csv(path)
                for artifact in (path, path.with_suffix(".csv"), path.with_suffix(".csv.schema.json")):
                    qc.write(artifact, artifact.name)
        archive.chmod(0o600)
    write_json(output / "progress_summary.json", model["summary"])
    render(output, model)
    return model

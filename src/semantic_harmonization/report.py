"""Write private reporting tables and render their offline explorer."""
from pathlib import Path
import csv
import os
import re

import pyarrow as pa
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from .application import write_parquet
from .report_progress import compare_runs
from .schemas import FACTS, SEMANTIC, STR, INT, BOOL
from .settings import MODULE, serialize, write_json

CANONICAL = pa.struct([FACTS.field("fact_type"), FACTS.field("temporality"),
                       *[f for f in FACTS if pa.types.is_struct(f.type)]])
FACT_LIST = pa.list_(pa.struct(list(FACTS)))
FEATURE = pa.struct([("name", STR), ("kind", STR), ("observed", INT), ("values", pa.list_(STR)),
                     ("verified", INT), ("as_of", INT)])
MAPPING = pa.schema([
    *[(k, STR) for k in ("mapping_id", "baseline_mapping_id", "variable", "concept", "family", "raw_type", "raw_json",
                        "context_json", "comparison_scope", "comparison_reason", "v1_strategy", "v2_strategy",
                        "change_kind", "actual_application", "initial_disposition", "final_disposition", "llm_outcome")],
    *[(k, BOOL) for k in ("v1_available", "v2_available", "comparable", "nonempty", "old_nonempty", "clinical_v1", "clinical_v2")],
    *[(k, INT) for k in ("episodes", "source_events", "occurrences", "patients", "baseline_episodes", "m1_m4_added", "m1_m4_removed")],
    *[(k, pa.struct(list(SEMANTIC))) for k in ("v1_legacy_fields", "v2_legacy_fields")],
    *[(k, pa.list_(CANONICAL)) for k in ("v1_canonical_facts", "m1_canonical_facts", "facts_added", "facts_removed", "facts_retained")],
    ("v2_typed_facts", FACT_LIST), ("evidence", SEMANTIC.field("evidence").type),
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
ROUTES = pa.schema([
    *[(k, STR) for k in ("mapping_id", "profile", "reason", "final_disposition", "extraction_status", "contribution_status")],
    ("send", BOOL), *[(k, INT) for k in ("relevant_segments", "resolved_segments", "residual_segments")],
    ("pending_reason_codes", pa.list_(STR)),
])


def csv_cell(value):
    if isinstance(value, (list, dict)):
        return serialize(value)
    if isinstance(value, str) and re.match(r"^(?:\s*[=+\-@＝＋－＠]|[\t\r\n])", value):
        return "'" + value
    return value


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: csv_cell(row.get(key)) for key in fields} for row in rows)


def render(output, model):
    """Embed structured data with JSON-safe escaping in a single offline document."""
    payload = dict(model)
    payload.pop("routes")
    # Row keys identify mappings rather than individuals. Strip repeated nested join keys.
    for row in payload["mappings"]:
        for fact in row["v2_typed_facts"] or ():
            for key in ("mapping_id", "segment_id", "fact_id"):
                fact.pop(key, None)
    for row in payload["work"]:
        row.pop("current_structured_result", None)
    safe = serialize(payload).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    environment = Environment(loader=FileSystemLoader(MODULE / "templates"), autoescape=select_autoescape(["html"]))
    path = output / "semantic_harmonization_final.html"
    path.write_text(environment.get_template("report.html").render(payload=Markup(safe)))
    os.chmod(path, 0o600)
    return path


def report(output: Path, cfg: dict, manifest: dict, *, target, baseline=None, routes=None):
    """Generate identical report artifacts for historical and future runs."""
    if target.manifest != manifest:
        raise ValueError("Report provenance differs from the target run")
    model = compare_runs(target, baseline, cfg, routes=routes)
    pending = [item for item in model["work"] if item["state_v2"] != "resolved"]
    write_parquet(output / "mapping_comparison.parquet", model["mappings"], MAPPING)
    write_parquet(output / "pending_work.parquet", pending, WORK)
    write_parquet(output / "mapping_routing.parquet", model["routes"], ROUTES)
    write_csv(output / "pending_work.csv", pending, WORK.names)
    for key, name, fallback in (("variables", "variable_progress", ["variable", "state"]),
                                ("families", "family_progress", ["family", "variables"])):
        rows = model[key]
        write_csv(output / f"{name}.csv", rows, list(rows[0]) if rows else fallback)
    write_json(output / "progress_summary.json", model["summary"])
    render(output, model)
    return model

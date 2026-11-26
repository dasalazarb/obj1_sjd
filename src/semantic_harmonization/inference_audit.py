"""Summarize smoke attempts and their impact at mapping and occurrence grain."""
from collections import Counter, defaultdict
from dataclasses import asdict

import pyarrow as pa

from .inference import profile
from .schemas import BOOL, INT, STR

SMOKE_AUDIT = pa.schema([
    *[(key, STR) for key in ("profile", "case_id", "phase", "status", "reason_code", "finish_reason")],
    *[(key, INT) for key in ("attempt", "expected_count", "accepted_count")],
    *[(key, BOOL) for key in ("abstention_passed", "fixture_valid", "has_final_channel")],
])
MAPPING_METRICS = ("eligible", "reused", "sent_to_model", "disabled", "model_unavailable", "not_run",
                   "accepted", "fallback")
CELL_METRICS = ("retained_episode_cells", "retained_source_event_cells", "added_facts", "eligible_patients",
                "fallback_patients")
COUNT_FIELDS = (*CELL_METRICS,
                *(f"{metric}_{grain}" for metric in MAPPING_METRICS for grain in ("mappings", "occurrences")))
SMOKE_SUMMARY = pa.schema([
    *[(key, STR) for key in ("profile", "status", "reason_code")],
    *[(key, INT) for key in ("attempts", "cases_passed", "cases_total", *COUNT_FIELDS)],
])
TRANSIENT = {"profile_disabled", "model_unavailable", "invalid_schema", "unsupported_evidence",
             "empty_response", "timeout"}
FALLBACK = {"original_fallback", "original_partial"}


def summarize_inference(smoke, pending, outputs, reused, sent, lineage, effective_audit, added_counts):
    """Use effective audit counts for retained cells and lineage for distinct patients."""
    by_id = {row["mapping_id"]: profile(row) for row in pending}
    counts = defaultdict(Counter)
    patients, fallback_patients = defaultdict(set), defaultdict(set)
    fallback = set()
    for row in effective_audit:
        key = row["mapping_id"]
        active = by_id.get(key)
        origin = row["effective_origin"]
        if active and origin.startswith("original_"):
            metric = ("retained_episode_cells" if row["source_grain"] == "episode_variable"
                      else "retained_source_event_cells")
            counts[active][metric] += row["occurrences"]
            if origin in FALLBACK:
                fallback.add(key)
    for line in lineage:
        key = line["mapping_id"]
        if key in by_id and line["patient_id"] is not None:
            active = by_id[key]
            patients[active].add(line["patient_id"])
            if key in fallback:
                fallback_patients[active].add(line["patient_id"])
    for row in pending:
        key = row["mapping_id"]
        item = outputs[key]
        tally = counts[by_id[key]]
        for metric, include in (
            ("eligible", True), ("reused", key in reused), ("sent_to_model", key in sent),
            ("disabled", item["status"] == "profile_disabled"), ("model_unavailable", item["status"] == "model_unavailable"),
            ("not_run", item["status"] == "not_run"), ("accepted", item["contribution_status"] == "added"),
            ("fallback", key in fallback),
        ):
            if include:
                tally[f"{metric}_mappings"] += 1
                tally[f"{metric}_occurrences"] += row["observations"]
        tally["added_facts"] += added_counts[key]
    for active in smoke.profiles:
        counts[active].update(eligible_patients=len(patients[active]), fallback_patients=len(fallback_patients[active]))
    return [asdict(result) | {key: counts[active][key] for key in COUNT_FIELDS}
            for active, result in smoke.profiles.items()]

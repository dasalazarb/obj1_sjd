"""Reconcile published V1 mappings with V2 facts on identical clinical inputs."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .inventory import LAB, routing_basis
from .schemas import Semantic
from .segmentation import segment
from .semantic_facts import canonical, clinical, part_facts
from .settings import file_hash, serialize

BASELINE_FILES = ("run_manifest.json", "semantic_dictionary.parquet", "variable_inventory.csv")


def baseline_files(cfg: dict) -> dict[str, Path]:
    """Find the configured baseline without modifying its immutable outputs."""
    named = cfg.get("comparison", {}).get("baseline_run_id")
    explicit = cfg["paths"].get("baseline")
    if not (named or explicit):
        return {}
    directory = Path(explicit) if explicit else Path(cfg["paths"]["output"]) / named
    if not directory.is_dir():
        return {}
    files = {name: directory / name for name in BASELINE_FILES}
    if not all(path.is_file() for path in files.values()):
        raise ValueError("Baseline report directory lacks required artifacts")
    return files


def baseline_identity(cfg: dict) -> dict:
    return {name: file_hash(path) for name, path in baseline_files(cfg).items()}


def nonempty_lab(row: dict) -> bool:
    raw = json.loads(row["raw_json"])
    return LAB.fullmatch(row["variable"]) is not None and raw is not None and (
        not isinstance(raw, str) or bool(raw.strip()))


def disposition(facts) -> str:
    if any(clinical(item) for item in facts):
        return "clinical_fact_present"
    if any(item.fact_type == "observation_metadata" for item in facts):
        return "metadata_or_reference_only"
    return "unresolved"


def clinical_set(facts) -> set[str]:
    return {canonical(item) for item in facts if clinical(item)}


def baseline_meaning(row: dict) -> set[str]:
    """Adapt V1 fields to the V2 comparison vocabulary without re-parsing V1 results."""
    parts = segment(row)
    if not parts:
        return set()
    old = Semantic.model_validate({name: row[name] for name in Semantic.model_fields})
    return clinical_set(part_facts(row, parts[0], old, "legacy-v1"))


def csv(out: Path, name: str, rows: list[dict], columns=None) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(out / f"{name}.csv", index=False)


def review(out: Path, cfg: dict, hashes: dict, rows: list[dict], segments, m1, final, routes: dict,
           extraction: dict, variables: list[dict], expected_baseline: dict) -> dict:
    """Emit complete disjoint coverage, routing, incremental and baseline audits."""
    by_mapping, determ = defaultdict(list), defaultdict(list)
    for fact in final:
        by_mapping[fact.mapping_id].append(fact)
    for fact in m1:
        determ[fact.mapping_id].append(fact)
    coverage_rows = []
    laboratory_routing = Counter()
    routing_counts, concept_routes, incremental = defaultdict(Counter), defaultdict(Counter), Counter()
    for row in rows:
        key = row["mapping_id"]
        routing = routes[key]
        weight = row["observations"]
        status = extraction.get(key, {}).get("contribution_status", "not_sent")
        tally = routing_counts[row["family"], routing["profile"], routing["reason"], status]
        tally.update(mappings=1, occurrences=weight, relevant_segments=routing["relevant_segments"],
                     residual_segments=routing["residual_segments"], resolved_segments=routing["resolved_segments"])
        if LAB.fullmatch(row["variable"]):
            basis = routing_basis(row["variable"], row["family"])
            concept_routes[row["variable"], row["concept"], row["family"], basis].update(mappings=1, occurrences=weight)
            laboratory_routing.update(eligible_mappings=1, profiled_mappings=int(basis != "fallback"),
                                      eligible_occurrences=weight, profiled_occurrences=weight * int(basis != "fallback"))
        if nonempty_lab(row):
            result = disposition(by_mapping[key])
            coverage_rows.append(dict(family=row["family"], concept=row["concept"], disposition=result,
                                      mappings=1, occurrences=weight,
                                      explicit_profile=int(routing_basis(row["variable"], row["family"]) != "fallback"),
                                      partial=int(result == "clinical_fact_present" and routing["residual_segments"] > 0)))
        novel = clinical_set(by_mapping[key]) - clinical_set(determ[key])
        if novel:
            incremental.update(mappings=1, occurrences=weight, facts=len(novel))
    csv(out, "concept_routing", [dict(variable=v, concept=c, family=f, route=r, **t)
                                 for (v, c, f, r), t in sorted(concept_routes.items())],
        ["variable", "concept", "family", "route", "mappings", "occurrences"])
    csv(out, "extraction_routing_summary", [dict(family=f, profile=p, reason=r, outcome=o, **t)
                                            for (f, p, r, o), t in routing_counts.items()])
    coverage = pd.DataFrame(coverage_rows, columns=["family", "concept", "disposition", "mappings", "occurrences",
                                                     "explicit_profile", "partial"])
    for key in ("family", "concept"):
        grouped = coverage.groupby([key, "disposition"], sort=True)[["mappings", "occurrences", "explicit_profile", "partial"]].sum()
        grouped.to_csv(out / f"semantic_coverage_by_{key}.csv")

    baseline, changes, residual = compare(cfg, hashes, rows, by_mapping, routes, variables, expected_baseline)
    csv(out, "baseline_comparison", changes, ["status", "mapping_id", "variable", "family", "observations",
                                              "facts_added", "facts_removed", "reason"])
    csv(out, "baseline_residual_disposition", residual, ["mapping_id", "variable", "family_v1", "family_v2",
                                                           "observations", "disposition", "residual_segments"])
    return dict(version="semantic-qc-v2", baseline=baseline, segment_roles=dict(Counter(part.role for part in segments)),
                fact_types=dict(Counter(item.fact_type for item in final)), incremental_llm=dict.fromkeys(
                    ("mappings", "occurrences", "facts"), 0) | dict(incremental),
                laboratory_routing=dict(laboratory_routing),
                llm_outcomes=dict(Counter(item["contribution_status"] for item in extraction.values())),
                laboratory_coverage={name: dict(zip(("mappings", "occurrences", "partial"), map(int, values)))
                                     for name, values in coverage.groupby("disposition")[["mappings", "occurrences", "partial"]].sum().iterrows()})


def compare(cfg, hashes, rows, by_mapping, routes, variables, expected):
    """Only quantify differences when input, mapping grain and protection agree."""
    files = baseline_files(cfg)
    if {name: file_hash(path) for name, path in files.items()} != expected:
        raise ValueError("Baseline artifacts changed during execution")
    if not files:
        return {"status": "unavailable", "reason": "Published baseline assets absent"}, [], []
    manifest = json.loads(files["run_manifest.json"].read_text())
    configured = cfg.get("comparison", {}).get("baseline_run_id")
    if not cfg["paths"].get("baseline") and configured and manifest["run_id"] != configured:
        raise ValueError("Baseline run ID does not match configured reference")
    for name in set(files) - {"run_manifest.json"}:
        if manifest["outputs"]["report"].get(name) != expected[name]:
            raise ValueError("Baseline artifact checksum differs from its published manifest")
    summary = {"run_id": manifest["run_id"]}
    keys = ("integrated", "context", "source_events")
    if any(hashes.get(key) != manifest["inputs"].get(key) for key in keys):
        return summary | {"status": "separate_scenario", "reason": "Clinical input fingerprints differ"}, [], []
    old_inventory = pd.read_csv(files["variable_inventory.csv"])
    policy = {row["variable"]: row["policy"] for row in variables}
    if old_inventory.set_index("variable")["policy"].to_dict() != policy:
        return summary | {"status": "incompatible", "reason": "Inventory policy differs"}, [], []
    current = {row["mapping_id"]: row for row in rows if nonempty_lab(row)}
    previous = {row["mapping_id"]: row for row in pq.read_table(files["semantic_dictionary.parquet"]).to_pylist()
                if nonempty_lab(row)}
    changes, residual = [], []
    counts, residual_counts = Counter(), defaultdict(Counter)
    for key in sorted(current.keys() | previous.keys()):
        before, after = previous.get(key), current.get(key)
        row = after or before
        record = dict(mapping_id=key, variable=row["variable"], family=row["family"], observations=row["observations"])
        if before is None or after is None or before["observations"] != after["observations"]:
            changes.append(dict(record, status="unmatched", reason="Mapping identity or grain differs"))
            counts["unmatched"] += 1
        else:
            old, new = baseline_meaning(before), clinical_set(by_mapping[key])
            added, removed = new - old, old - new
            counts.update(comparable=1, added_mappings=bool(added), removed_mappings=bool(removed),
                          added_facts=len(added), removed_facts=len(removed),
                          added_occurrences=after["observations"] * bool(added))
            if added or removed:
                changes.append(dict(record, status="changed", facts_added=serialize(sorted(added)),
                                    facts_removed=serialize(sorted(removed)),
                                    reason="Reconciled typed meaning" if before["family"] == after["family"]
                                    else "Assay profile corrected"))
        if before and before["strategy"] == "unchanged" and before["variable"].endswith("__text"):
            comparable = after and before["observations"] == after["observations"]
            outcome = disposition(by_mapping[key]) if comparable else "unmatched"
            residual.append(dict(mapping_id=key, variable=row["variable"], family_v1=before["family"],
                                 family_v2=after["family"] if after else None, observations=before["observations"],
                                 disposition=outcome, residual_segments=routes[key]["residual_segments"] if after else None))
            residual_counts[outcome].update(mappings=1, occurrences=before["observations"])
    return summary | {"status": "reconciled", **counts, "baseline_residual": dict(residual_counts)}, changes, residual

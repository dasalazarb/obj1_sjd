"""Materialize observation-grain longitudinal inputs and projection provenance."""
from __future__ import annotations

from collections import Counter, defaultdict
import json

import pandas as pd
import pyarrow as pa

from .application import timestamp, write_parquet
from .schemas import BOOL, INT, STR
from .semantic_facts import canonical, clinical

ROLE_VERSION = "semantic-representation-v2"
FIELD_ROLES = {
    "unit": "measurement_metadata", "comparator": "measurement_metadata", "specimen_state": "observation_process",
    "result_availability": "observation_process", "derived_recompute": "protected_or_derived",
}
VIEW = pa.schema([
    *[(name, STR) for name in ("patient_id", "clinical_episode_id", "observation_id", "mapping_id", "concept_id",
                              "source_variable", "family", "source_grain", "measurement_date", "result_available_date",
                              "observation_status")],
    ("fact_ids", pa.list_(STR)), ("as_of_fact_ids", pa.list_(STR)), ("verified_available_by_episode", BOOL),
])
MANIFEST = pa.schema([
    *[(name, STR) for name in ("feature_name", "concept_id", "source_variable", "semantic_field", "role", "data_type",
                               "unit_contract", "temporal_availability", "allowed_use", "source_lineage",
                               "projection_rule", "projection_version", "reason")],
    *[(name, INT) for name in ("observed_episodes", "observed_patients", "distinct_values")],
])


def role(fact) -> str:
    if fact.fact_type == "hla_result":
        return "genetic_context"
    if fact.fact_type == "medication_context":
        return "treatment_context"
    if fact.fact_type == "observation_metadata":
        return {"specimen_information": "observation_process", "administrative_status": "observation_process",
                "assay_method": "measurement_metadata", "reference_rule": "unresolved_or_reference",
                "general_interpretation": "unresolved_or_reference", "historical_reference": "unresolved_or_reference"}[
                    fact.payload.role]
    return "biological_value" if clinical(fact) else "unresolved_or_reference"


def observation_status(row, facts) -> str:
    if row["result_availability"] and row["result_availability"] != "reported":
        return row["result_availability"]
    if any(clinical(item) for item in facts) or row["result_availability"] == "reported":
        return "observed"
    if row["specimen_state"]:
        return "specimen_issue"
    if any(item.fact_type == "observation_metadata" and
           item.quote.strip().casefold().rstrip(".") == "not applicable" for item in facts):
        return "not_applicable"
    raw = json.loads(row["raw_json"])
    return "no_source_value" if raw is None or isinstance(raw, str) and not raw.strip() else "unresolved_or_metadata"


def materialize(analytic, out, master, observations, facts, features, variables) -> dict:
    """Write analysis-ready links without spreading facts to different visits."""
    by_mapping, profiles, observation_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for fact in facts:
        by_mapping[fact.mapping_id].append(fact)
    anchors = {(str(row.patient_id), str(row.clinical_episode_id)): timestamp(row.clinical_anchor_date)
               for row in master[["patient_id", "clinical_episode_id", "clinical_anchor_date"]].itertuples(index=False)
               } if "clinical_anchor_date" in master else {}
    view = []
    for row in observations:
        if row["clinical_episode_id"] is None or row["patient_id"] is None:
            continue
        if row["family"] in {"derived", "other", "protected"}:
            continue
        pid, episode = row["patient_id"], row["clinical_episode_id"]
        source_facts = by_mapping[row["mapping_id"]]
        available = timestamp(row["result_available_date"])
        measured = timestamp(row["measurement_date"])
        anchor = anchors.get((pid, episode), pd.NaT)
        verified = not pd.isna(available) and not pd.isna(anchor) and available <= anchor and (
            pd.isna(measured) or available >= measured) and row["result_valid"] is not False
        view.append(dict(patient_id=pid, clinical_episode_id=episode, observation_id=row["observation_id"],
                         mapping_id=row["mapping_id"], concept_id=row["concept"], source_variable=row["variable"],
                         family=row["family"], source_grain=row["source_grain"], measurement_date=row["measurement_date"],
                         result_available_date=row["result_available_date"], observation_status=observation_status(row, source_facts),
                         fact_ids=[item.fact_id for item in source_facts],
                         as_of_fact_ids=[item.fact_id for item in source_facts if verified and clinical(item)
                                         and not (item.fact_type == "hla_result" and item.payload.ambiguous)],
                         verified_available_by_episode=verified))
        observation_groups[row["variable"]].append(row)
        for item in source_facts:
            profiles[row["variable"], row["concept"], item.fact_type, role(item)].append((pid, episode, canonical(item)))
    write_parquet(analytic / "patient_visit_semantic_observations.parquet", view, VIEW)

    entries = []
    for (variable, concept, field, kind), observed in sorted(profiles.items()):
        entries.append(dict(feature_name=f"{variable}/{field}/{kind}", concept_id=concept, source_variable=variable,
                            semantic_field=field, role=kind, data_type="typed_struct", unit_contract="per fact",
                            temporal_availability="per observation result availability",
                            allowed_use="candidate_as_of_if_verified" if kind in {"biological_value", "genetic_context"} else "descriptive",
                            source_lineage="patient_visit_semantic_observations.parquet → semantic_facts.parquet",
                            projection_rule="observation links to canonical mapping facts", projection_version=ROLE_VERSION,
                            reason=None, observed_episodes=len({(p, e) for p, e, _ in observed}),
                            observed_patients=len({p for p, _, _ in observed}),
                            distinct_values=len({value for _, _, value in observed})))
    for item in features:
        field = item["kind"]
        kind = FIELD_ROLES.get(field, "measurement_metadata" if "unit" in field or "comparator" in field
                               else "biological_value")
        values = [(row["patient_id"], row["clinical_episode_id"], row[field])
                  for row in observation_groups[item["source"]] if field in row and row[field] is not None
                  and row["source_grain"] == "episode_variable"]
        entries.append(dict(feature_name=item["name"], concept_id=next((r["concept"] for r in observation_groups[item["source"]]), None),
                            source_variable=item["source"], semantic_field=field, role=kind, data_type=item["dtype"],
                            unit_contract=item["units"], temporal_availability=item["temporal_availability"],
                            allowed_use=item["allowed_use"], source_lineage=item["lineage"],
                            projection_rule=item["analytic_eligibility_rule"], projection_version=ROLE_VERSION,
                            reason=None, observed_episodes=len({(p, e) for p, e, _ in values}),
                            observed_patients=len({p for p, _, _ in values}), distinct_values=len({v for _, _, v in values})))
    for item in variables:
        if item["policy"] != "eligible" or item["family"] in {"derived", "other"}:
            entries.append(dict(feature_name=item["variable"], source_variable=item["variable"], semantic_field="source",
                                data_type=item["dtype"], role="protected_or_derived",
                                projection_rule="retain source column", projection_version=ROLE_VERSION,
                                reason=item["policy"] if item["policy"] != "eligible" else item["family"]))
    write_parquet(analytic / "analytical_representation_manifest.parquet", entries, MANIFEST)
    coverage = pd.DataFrame(entries, columns=MANIFEST.names)
    coverage.to_csv(out / "feature_role_coverage.csv", index=False)
    return dict(observations=len(view), roles=dict(Counter(row["role"] for row in entries)),
                observation_status=dict(Counter(row["observation_status"] for row in view)),
                as_of_observations=sum(bool(row["as_of_fact_ids"]) for row in view))

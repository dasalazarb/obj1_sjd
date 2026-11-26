"""Materialize observation-grain longitudinal inputs and projection provenance."""
from __future__ import annotations

from collections import Counter, defaultdict
import json

from .settings import serialize

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .application import timestamp, write_parquet
from .closure_audit import wide_field
from .schemas import BOOL, INT, STR
from .semantic_facts import canonical, clinical

ROLE_VERSION = "semantic-representation-v2"
FIELD_ROLES = {
    "unit": "measurement_metadata", "comparator": "measurement_metadata", "specimen_state": "observation_process",
    "result_availability": "observation_process", "derived_recompute": "protected_or_derived",
}
VIEW = pa.schema([
    *[(name, STR) for name in ("patient_id", "clinical_episode_id", "observation_id", "mapping_id", "concept_id",
                              "source_variable", "family", "source_grain", "source_observation_id", "measurement_date", "result_available_date",
                              "observation_status", "availability_source", "representative_observation_id")],
    ("fact_ids", pa.list_(STR)), ("as_of_fact_ids", pa.list_(STR)), ("verified_available_by_episode", BOOL),
])
MANIFEST = pa.schema([
    *[(name, STR) for name in ("feature_name", "concept_id", "source_variable", "semantic_field", "role", "data_type",
                               "unit_contract", "temporal_availability", "allowed_use", "source_lineage",
                               "projection_rule", "projection_version", "reason")],
    *[(name, INT) for name in ("observed_episodes", "observed_patients", "distinct_values", "verified_observations", "as_of_observations")],
])


TEMPORAL = pa.schema([
    *[(name, STR) for name in ("feature_name", "source_variable", "availability_rule", "exclusion_reason")],
    *[(name, INT) for name in ("observed_episodes", "observed_patients", "available_episodes", "as_of_episodes")],
])
READINESS = pa.schema([
    *[(name, STR) for name in ("feature_name", "source_variable", "study", "mode", "role", "data_type", "encoding", "status",
                               "reason", "model_consumption")],
    *[(name, INT) for name in ("observed_episodes", "observed_patients", "baseline_episodes", "available_episodes",
                               "as_of_episodes", "distinct_values")],
])


def as_of_licensed(row, fact):
    """Require a current biological assertion with observation-level availability."""
    if not clinical(fact) or row["variable"].startswith("sero__hla_"):
        return False
    if fact.fact_type == "hla_result":
        return fact.payload.nomenclature == "allele" and fact.payload.ipd_match == "allele" and bool(fact.payload.allele)
    return True



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



def observation_identity(row):
    return row["patient_id"], row["clinical_episode_id"], row["concept"], row["source_observation_id"]


def related_observations(observations):
    """Link alternate representations only with an explicit common observation ID."""
    groups = defaultdict(list)
    for row in observations:
        identity = observation_identity(row)
        if all(value is not None and value != "" for value in identity):
            groups[identity].append(row)
    representatives = {}
    for key, rows in groups.items():
        source = [row for row in rows if row["source_grain"] == "btris_observation"]
        if len(source) == 1:
            representatives[key] = source[0]
        elif not source and len(rows) == 2 and {row["variable"].rsplit("__", 1)[-1] for row in rows} == {"text", "value"}:
            representatives[key] = next(row for row in rows if row["variable"].endswith("__text"))
    return representatives

def materialize(analytic, out, master, observations, facts, features, variables, wide=None):
    """Write analysis-ready links without spreading facts to different visits."""
    by_mapping, profiles, observation_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for fact in facts:
        by_mapping[fact.mapping_id].append(fact)
    anchors = {(str(row.patient_id), str(row.clinical_episode_id)): timestamp(row.clinical_anchor_date)
               for row in master[["patient_id", "clinical_episode_id", "clinical_anchor_date"]].itertuples(index=False)
               } if "clinical_anchor_date" in master else {}
    view = []
    temporal, baseline = [], {}
    by_observation = {}
    representatives = related_observations(observations)
    for row in observations:
        observation_groups[row["variable"]].append(row)
        if row["clinical_episode_id"] is None or row["patient_id"] is None:
            continue
        if row["family"] in {"derived", "other", "protected"}:
            continue
        pid, episode = row["patient_id"], row["clinical_episode_id"]
        source_facts = by_mapping[row["mapping_id"]]
        representative = representatives.get(observation_identity(row))
        availability_date = row["result_available_date"]
        measurement_date = row["measurement_date"]
        availability_source = "source observation" if availability_date else None
        eligible_source = (representative is not None and representative["source_grain"] == "btris_observation"
                           and representative["result_valid"] is True)
        if eligible_source and not availability_date:
            availability_date = representative["result_available_date"]
            if availability_date:
                availability_source = "matched BTRIS observation ID"
                measurement_date = measurement_date or representative["measurement_date"]
        available = timestamp(availability_date)
        measured = timestamp(measurement_date)
        anchor = anchors.get((pid, episode), pd.NaT)
        verified = not pd.isna(available) and not pd.isna(anchor) and available <= anchor and (
            pd.isna(measured) or available >= measured) and row["result_valid"] is not False
        observable = not pd.isna(available) and (pd.isna(measured) or available >= measured) and row["result_valid"] is not False
        linked = dict(patient_id=pid, clinical_episode_id=episode, observation_id=row["observation_id"],
                         mapping_id=row["mapping_id"], concept_id=row["concept"], source_variable=row["variable"],
                         family=row["family"], source_grain=row["source_grain"],
                         source_observation_id=row["source_observation_id"], measurement_date=measurement_date,
                         result_available_date=availability_date, availability_source=availability_source,
                         representative_observation_id=representative["observation_id"] if representative else None, observation_status=observation_status(row, source_facts),
                         fact_ids=[item.fact_id for item in source_facts],
                         as_of_fact_ids=[item.fact_id for item in source_facts if verified and as_of_licensed(row, item)],
                         verified_available_by_episode=verified)
        view.append(linked)
        by_observation[row["observation_id"]] = (linked, observable)
        for item in source_facts:
            profiles[row["variable"], row["concept"], item.fact_type, role(item)].append(((pid, episode), canonical(item), observable, item.fact_id in linked["as_of_fact_ids"], row["baseline"] is True))
    write_parquet(analytic / "patient_visit_semantic_observations.parquet", view, VIEW)

    def add_temporal(name, variable, values):
        details = dict(
            feature_name=name, source_variable=variable,
            observed_episodes=len({key for key, *_ in values}),
            observed_patients=len({key[0] for key, *_ in values}),
            available_episodes=len({key for key, _, available, *_ in values if available}),
            as_of_episodes=len({key for key, _, _, as_of, _ in values if as_of}),
            availability_rule="Explicit result availability on the linked observation, by episode anchor",
        )
        details["exclusion_reason"] = (None if details["as_of_episodes"] else
                                       "No eligible evidence by the episode anchor" if details["available_episodes"] else
                                       "No verified result availability")
        temporal.append(details)
        baseline[name] = len({key for key, _, _, _, base in values if base})
        return details

    entries = []
    for (variable, concept, field, kind), observed in sorted(profiles.items()):
        name = f"{variable}/{field}/{kind}"
        timing = add_temporal(name, variable, observed)
        entries.append(dict(feature_name=name, concept_id=concept, source_variable=variable,
                            semantic_field=field, role=kind, data_type="typed_struct", unit_contract="per fact",
                            temporal_availability="per observation result availability",
                            allowed_use="candidate_as_of_if_verified" if kind in {"biological_value", "genetic_context"} else "descriptive",
                            source_lineage="patient_visit_semantic_observations.parquet → semantic_facts.parquet",
                            projection_rule="observation links to canonical mapping facts", projection_version=ROLE_VERSION,
                            reason=timing["exclusion_reason"], observed_episodes=timing["observed_episodes"],
                            observed_patients=timing["observed_patients"],
                            verified_observations=sum(item[2] for item in observed),
                            as_of_observations=sum(item[3] for item in observed),
                            distinct_values=len({value for _, value, *_ in observed})))
    for item in features:
        field = item["kind"]
        kind = FIELD_ROLES.get(field, "measurement_metadata" if "unit" in field or "comparator" in field
                               else "biological_value")
        values = []
        written = wide[item["name"]].to_pylist() if wide is not None else None
        for row in observation_groups[item["source"]]:
            if row["source_grain"] != "episode_variable":
                continue
            value = written[row["source_row"]] if written is not None else row.get(field)
            if value is None:
                continue
            link, available = by_observation.get(row["observation_id"], ({}, False))
            verified = kind == "biological_value" and bool(link.get("as_of_fact_ids")) and any(
                fact.fact_id in link["as_of_fact_ids"] and wide_field(fact, field)
                for fact in by_mapping[row["mapping_id"]])
            # Derived wide fields carry their own reconstruction and availability proof.
            # Their episode-variable source row does not supply that proof per value.
            values.append(((row["patient_id"], row["clinical_episode_id"]), serialize(value), available,
                           verified and field != "derived_recompute", row["baseline"] is True))
        timing = add_temporal(item["name"], item["source"], values)
        entries.append(dict(feature_name=item["name"], concept_id=next((r["concept"] for r in observation_groups[item["source"]]), None),
                            source_variable=item["source"], semantic_field=field, role=kind, data_type=item["dtype"],
                            unit_contract=item["units"], temporal_availability=item["temporal_availability"],
                            allowed_use=item["allowed_use"], source_lineage=item["lineage"],
                            projection_rule=item["analytic_eligibility_rule"], projection_version=ROLE_VERSION,
                            reason=timing["exclusion_reason"], observed_episodes=timing["observed_episodes"],
                            observed_patients=timing["observed_patients"],
                            verified_observations=sum(value[2] for value in values),
                            as_of_observations=sum(value[3] for value in values),
                            distinct_values=len({value for _, value, *_ in values})))
    for item in variables:
        if item["policy"] != "eligible" or item["family"] in {"derived", "other"}:
            entries.append(dict(feature_name=item["variable"], source_variable=item["variable"], semantic_field="source",
                                data_type=item["dtype"], role="protected_or_derived",
                                projection_rule="retain source column", projection_version=ROLE_VERSION,
                                reason=item["policy"] if item["policy"] != "eligible" else item["family"]))
    write_parquet(analytic / "analytical_representation_manifest.parquet", entries, MANIFEST)
    write_parquet(out / "feature_role_coverage.parquet", entries, MANIFEST)
    write_parquet(analytic / "feature_temporal_coverage.parquet", temporal, TEMPORAL)
    summary = dict(observations=len(view), roles=dict(Counter(row["role"] for row in entries)),
                observation_status=dict(Counter(row["observation_status"] for row in view)),
                as_of_observations=sum(bool(row["as_of_fact_ids"]) for row in view))
    return summary, entries, temporal, baseline


def write_readiness(out, entries, temporal, baseline):
    """Attach downstream profile evidence to already validated candidate features."""
    entry_by_name = {entry["feature_name"]: entry for entry in entries}
    readiness = []
    profile = out / "study_inputs/pharma/00_pharma_semantic_feature_profile.parquet"
    audited = {(item["feature_name"], item["study"]): item for item in pq.read_table(profile).to_pylist()} if profile.is_file() else {}
    for timing in temporal:
        entry = entry_by_name[timing["feature_name"]]
        for study in ("Pharma", "Graph", "Economic"):
            evidence = audited.get((timing["feature_name"], study), {})
            for mode in ("descriptive", "as_of_verified"):
                excluded = mode == "as_of_verified" and not timing["as_of_episodes"]
                unavailable = ("No downstream model run" if study == "Economic" else
                               "Analytical representation requires a study-specific encoding and selection contract")
                reason = timing["exclusion_reason"] if excluded else evidence.get("reason") or unavailable
                readiness.append({key: value for key, value in timing.items() if key in READINESS.names} | dict(
                    study=study, mode=mode, role=entry["role"], data_type=entry["data_type"],
                    status="excluded" if excluded else "encoding smoke eligible" if (
                        mode == "descriptive" and evidence.get("smoke_eligible")
                    ) else "profiled" if evidence else "not assessed",
                    encoding=evidence.get("encoding"), reason=reason, model_consumption="not run", baseline_episodes=baseline[timing["feature_name"]],
                    distinct_values=entry["distinct_values"]))
    write_parquet(out / "downstream_feature_readiness.parquet", readiness, READINESS)

"""Read verified reporting inputs without initializing inference or resources."""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
import re

import pyarrow.compute as pc
import pyarrow.parquet as pq

from .inventory import protected
from .settings import KEYS, digest, file_hash

INPUTS = ("integrated", "context", "source_events")
RUN_ID = re.compile("[a-f0-9]{64}")
CANDIDATE = "10_integrated_longitudinal_clinical_episode_harmonized.parquet"
FEATURE_FIELDS = ("feature_name", "source_variable", "semantic_field", "role", "data_type",
                  "temporal_availability", "allowed_use", "observed_episodes", "observed_patients",
                  "verified_observations", "as_of_observations")


def file_manifest(directory: Path, *, skip=()) -> dict:
    """Record published files by relative path and content checksum."""
    return {path.relative_to(directory).as_posix(): file_hash(path)
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name not in skip}


@dataclass
class Run:
    qc: Path
    analytic: Path | None
    manifest: dict
    missing: set[str] = field(default_factory=set)
    _schemas: dict = field(default_factory=dict)

    def path(self, name, *, analytic=False):
        root = self.analytic if analytic else self.qc
        path = root / name if root is not None else None
        if path is None or not path.is_file():
            self.missing.add(name)
            return None
        return path

    def rows(self, name, columns=None, *, analytic=False, optional_columns=()):
        path = self.path(name, analytic=analytic)
        if path is None:
            return None
        if path.suffix == ".csv" and path.with_suffix(".csv.schema.json").is_file():
            path = path.with_suffix(".parquet")
        if path.suffix == ".csv":
            with path.open(newline="") as stream:
                return list(csv.DictReader(stream))
        if columns is not None:
            schema = self._schemas.get(path)
            if schema is None:
                schema = self._schemas[path] = pq.read_schema(path).names
            if set(columns) - set(schema):
                self.missing.add(f"{name} (required columns)")
                return None
            columns = [*columns, *(key for key in optional_columns if key in schema)]
        return pq.read_table(path, columns=columns).to_pylist()

    def json(self, name):
        path = self.path(name)
        return json.loads(path.read_text()) if path else None


def published(qc, analytic_root, expected_id=None):
    """Verify both sealed trees and the documented V1/V2 identity contract."""
    path = qc / "run_manifest.json"
    manifest = json.loads(path.read_text())
    run_id = manifest.get("run_id", "")
    identity = manifest.get("identity", {})
    if (not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id)
            or (expected_id is not None and run_id != expected_id)
            or not manifest.get("completed_at") or not manifest.get("inputs", {}).get("integrated")
            or identity.get("inputs") != manifest["inputs"] or digest(identity) != run_id):
        raise ValueError(f"Invalid or incomplete run identity in {path}")
    analytic = analytic_root / run_id
    outputs = manifest.get("outputs", {})
    for directory, name, skip in ((qc, "report", {"run_manifest.json"}), (analytic, "analytic", ())):
        if not outputs.get(name) or not directory.is_dir() or file_manifest(directory, skip=skip) != outputs[name]:
            raise ValueError(f"Published {name} checksums differ or artifacts are missing for {run_id}")
    return Run(qc, analytic, manifest)


def inventory_policy(rows):
    if rows is None:
        return None
    entries = {row["variable"]: row["policy"] for row in rows}
    if len(entries) != len(rows):
        raise ValueError("Duplicate variable inventory entries")
    return entries


def allowed(variable, cfg):
    return isinstance(variable, str) and not protected(variable, cfg)


def indexed(rows):
    result = {row["mapping_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate inventory or dictionary mapping IDs")
    return result


def grouped(rows, keys):
    if rows is None:
        return None
    groups = defaultdict(list)
    for row in rows:
        if row["mapping_id"] in keys:
            groups[row["mapping_id"]].append(row)
    return groups


LINE_FIELDS = (*KEYS, "mapping_id", "source_grain", "source_row")


def lineage(run, keys):
    return grouped(run.rows(
        "row_lineage.parquet", LINE_FIELDS,
        optional_columns=("source_hash", "source_observation_id", "measurement_date", "result_available_date",
                          "paired_numeric_json", "n_measurements", "source_reference_status", "result_valid", "baseline")), keys)


def grain(rows):
    return sorted(digest({key: value for key, value in row.items() if key != "mapping_id"}) for row in rows)


def unique_counts(rows):
    if rows is None:
        return dict(episodes=None, patients=None, source_events=None, occurrences=None, baseline_episodes=None)
    episode = {(r["patient_id"], r["clinical_episode_id"]) for r in rows if r["patient_id"] is not None
               and r["clinical_episode_id"] is not None and r["source_grain"] == "episode_variable"}
    baseline = {(r["patient_id"], r["clinical_episode_id"]) for r in rows if r.get("baseline") is True
                and (r["patient_id"], r["clinical_episode_id"]) in episode and r["source_grain"] == "episode_variable"}
    return dict(episodes=len(episode), patients=len({r["patient_id"] for r in rows if r["patient_id"] is not None}),
                source_events=sum(r["source_grain"] == "btris_observation" for r in rows), occurrences=len(rows),
                baseline_episodes=len(baseline) if not rows or "baseline" in rows[0] else None)


def materialization(run, lines, mapping_variables, cfg):
    """Measure written fields at episode grain and typed links at observation grain."""
    candidate = run.path(CANDIDATE, analytic=True)
    wide = run.rows("feature_manifest.parquet", analytic=True)
    representation = run.rows("analytical_representation_manifest.parquet", analytic=True)
    links = run.rows("patient_visit_semantic_observations.parquet",
                     ("mapping_id", "fact_ids", "as_of_fact_ids", "verified_available_by_episode"), analytic=True)
    by_mapping, features = defaultdict(list), []
    if candidate and wide is not None:
        schema = pq.read_schema(candidate).names
        wide = [r for r in wide if allowed(r.get("source"), cfg)
                and (r.get("name") == "harm__" + r["source"]
                     or str(r.get("name", "")).startswith("harm__" + r["source"] + "__"))]
        names = [row["name"] for row in wide if row["name"] in schema]
        table = pq.read_table(candidate, columns=[*KEYS, *names])
        keys = list(zip(*(table[name].to_pylist() for name in KEYS), strict=True))
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate keys in analytical candidate")
        by_key = {key: index for index, key in enumerate(keys)}
        indexes = defaultdict(dict)
        for mapping, occurrences in (lines or {}).items():
            for row in occurrences:
                if row["source_grain"] != "episode_variable":
                    continue
                key = (row["patient_id"], row["clinical_episode_id"])
                if key not in by_key:
                    raise ValueError("Episode lineage key absent from candidate")
                if row["source_row"] != by_key[key]:
                    raise ValueError("Episode lineage row differs from candidate")
                indexes[mapping][row["source_row"]] = key
        source_mappings = defaultdict(list)
        for mapping in indexes:
            source_mappings[mapping_variables[mapping]].append(mapping)
        for entry in wide:
            name, variable = entry["name"], entry["source"]
            values = table[name].to_pylist() if name in names else None
            row = dict(feature_name=name, source_variable=variable, semantic_field=entry["kind"], role="wide candidate",
                       data_type=entry["dtype"], temporal_availability=entry["temporal_availability"],
                       allowed_use=entry["allowed_use"])
            features.append(row | dict(observed_episodes=sum(v is not None for v in values) if values is not None else None,
                                       observed_patients=len({keys[i][0] for i,v in enumerate(values) if v is not None})
                                       if values is not None else None))
            if values is not None:
                # A mapping can have no episode instances when it occurs only in BTRIS.
                for mapping in source_mappings[variable]:
                    written = [values[i] for i in indexes[mapping] if values[i] is not None]
                    if written:
                        by_mapping[mapping].append(dict(name=name, kind="wide", semantic_field=entry["kind"], observed=len(written),
                                                        values=sorted({json.dumps(v, default=str) for v in written})))
    for entry in representation or ():
        variable = entry.get("source_variable")
        if allowed(variable, cfg) and entry.get("data_type") == "typed_struct" and links is not None:
            features.append({k: entry.get(k) for k in FEATURE_FIELDS})
    by_link = grouped(links, mapping_variables)
    if by_link is not None:
        for mapping, entries in by_link.items():
            observed = sum(bool(row["fact_ids"]) for row in entries)
            if observed:
                by_mapping[mapping].append(dict(name="Typed observation links", kind="long", observed=observed,
                                                verified=sum(bool(row["verified_available_by_episode"]) for row in entries),
                                                as_of=sum(bool(row["as_of_fact_ids"]) for row in entries)))
    return by_mapping, features, candidate, by_link, wide is not None


def same_original(before, after, original=None):
    """Verify preserved source columns and their values with candidate keys aligned."""
    if before is None or after is None:
        return None
    old = pq.ParquetFile(before)
    new = pq.ParquetFile(after)
    source = pq.ParquetFile(original) if original else old
    names = [name for name in source.schema_arrow.names if not name.startswith("harm__")]
    if (set(names) != {name for name in new.schema_arrow.names if not name.startswith("harm__")}
            or set(names) != {name for name in old.schema_arrow.names if not name.startswith("harm__")}):
        return False
    for first in (old, source) if original else (old,):
        lkeys, rkeys = first.read(columns=KEYS), new.read(columns=KEYS)
        if lkeys.num_rows != rkeys.num_rows:
            return False
        if not lkeys.equals(rkeys):
            lorder = pc.sort_indices(lkeys, sort_keys=[(key, "ascending") for key in KEYS])
            rorder = pc.sort_indices(rkeys, sort_keys=[(key, "ascending") for key in KEYS])
            if not lkeys.take(lorder).equals(rkeys.take(rorder)):
                return False
        else:
            lorder = rorder = None
        for offset in range(0, len(names), 32):
            columns = names[offset:offset+32]
            lhs, rhs = first.read(columns=columns), new.read(columns=columns)
            if lorder is not None:
                lhs, rhs = lhs.take(lorder), rhs.take(rorder)
            if not lhs.equals(rhs):
                return False
    return True


def feature_transitions(before, after, features_v1, features_v2):
    """Count changed candidate values after aligning rows by episode key."""
    if before is None or after is None:
        return []
    names_v1 = {r["feature_name"] for r in features_v1 if r["role"] == "wide candidate"}
    names_v2 = {r["feature_name"] for r in features_v2 if r["role"] == "wide candidate"}
    columns = sorted(names_v1 | names_v2)
    old, new = pq.ParquetFile(before), pq.ParquetFile(after)
    sort_keys = [(key, "ascending") for key in KEYS]
    old_order = pc.sort_indices(old.read(columns=KEYS), sort_keys=sort_keys)
    new_order = pc.sort_indices(new.read(columns=KEYS), sort_keys=sort_keys)
    output = []
    for start in range(0, len(columns), 32):
        batch = columns[start:start+32]
        old_names = set(old.schema_arrow.names) & set(batch)
        new_names = set(new.schema_arrow.names) & set(batch)
        old_values = old.read(columns=sorted(old_names)).take(old_order) if old_names else None
        new_values = new.read(columns=sorted(new_names)).take(new_order) if new_names else None
        for name in batch:
            old_present = name in old_names
            new_present = name in new_names
            def states(values, present):
                return [("observed", json.dumps(v, default=str)) if v is not None else ("null", None)
                        for v in values[name].to_pylist()] if present else [("column absent", None)] * len(old_order)
            transitions = Counter(
                (v1[0], v2[0], v1 != v2)
                for v1, v2 in zip(states(old_values, old_present), states(new_values, new_present), strict=True))
            output.append(dict(feature_name=name, v1_column_present=old_present, v2_column_present=new_present,
                               evaluated_episodes=len(old_order),
                               transitions=[dict(v1=a, v2=b, value_changed=changed, episodes=n)
                                            for (a,b,changed),n in sorted(transitions.items())]))
    return output

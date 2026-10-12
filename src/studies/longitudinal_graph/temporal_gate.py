"""Enforce researcher approval bound to exact reviewed artifacts before Script 03."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from src.studies.longitudinal_graph.reference import sha256


def require_temporal_approval(readiness_path, approval_path, manifest_path, membership_path, *, intervals_path=None, config_path=None):
    if any(path is None or not Path(path).is_file() for path in (readiness_path, approval_path, manifest_path)):
        raise ValueError("Script 03 blocked: missing readiness, run manifest or recorded researcher approval")
    readiness = json.loads(Path(readiness_path).read_text())
    record = json.loads(Path(approval_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if readiness.get("status") not in {"ready_for_researcher_review", "approved-temporal"}:
        raise ValueError("Script 03 blocked: official readiness gate is not ready for approval")
    required = {f"G{i}" for i in range(8)}
    gates = {row["gate"]: row for row in readiness.get("gates", [])}
    if any(gates.get(gate, {}).get("status") != "passed" for gate in required):
        raise ValueError("Script 03 blocked: required scientific gates are incomplete")
    if record.get("approval_status") != "approved-temporal":
        raise ValueError("Script 03 blocked: researcher approval status missing")
    for field in ("responsible", "approved_at", "justification", "selected_scenario", "source_run_id"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Script 03 blocked: approval missing {field}")
    datetime.fromisoformat(record["approved_at"].replace("Z", "+00:00"))
    if record["source_run_id"] != manifest.get("run_id"):
        raise ValueError("Script 03 blocked: approval refers to a different run")
    if record["selected_scenario"] != readiness.get("scenario"):
        raise ValueError("Script 03 blocked: approval refers to a different scenario")
    if record.get("run_manifest_sha256") != sha256(manifest_path):
        raise ValueError("Script 03 blocked: approved run manifest hash changed")
    if record.get("membership_sha256") != sha256(membership_path):
        raise ValueError("Script 03 blocked: approved membership hash changed")
    # The reviewed manifest must inventory the selected membership itself.
    base = Path(manifest_path).parent.resolve()
    member = Path(membership_path).resolve()
    if not member.is_relative_to(base) or manifest.get("artifact_sha256", {}).get(str(member.relative_to(base))) != sha256(member):
        raise ValueError("Script 03 blocked: membership is not in the reviewed artifact inventory")
    if intervals_path is not None and manifest.get("input_sha256", {}).get(str(intervals_path)) != sha256(intervals_path):
        raise ValueError("Script 03 blocked: interval source differs from the reviewed run")
    if config_path is not None and manifest.get("config_sha256") != sha256(config_path):
        raise ValueError("Script 03 blocked: configuration differs from the reviewed run")
    return record

"""Read-only CLI discovery for versioned runs; never choose an ambiguous reference."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from src.studies.longitudinal_graph._review.reference import safe_id
from src.studies.longitudinal_graph.validation import study_paths


class ReviewStartupError(ValueError):
    """Expected CLI setup problem, displayed without an implementation traceback."""
    def __init__(self, message, reason="G0_frozen_reference_unavailable"):
        super().__init__(message)
        self.reason = reason


def _checked_id(value, option):
    try:
        return safe_id(value)
    except (ValueError, TypeError) as error:
        hint = "Omit --run-id to generate a unique ID." if option == "--run-id" else "Provide --reference-root to select an existing archive."
        raise ReviewStartupError(
            f"{option} must be a simple name (letters, digits, '.', '_' or '-'), not a path; "
            f"for example: review-20261012. {hint}",
            reason="invalid_cli_identifier",
        ) from error


def _archives(locations, requested_id=None):
    """Check the two existing layouts, without recursive repository-wide search."""
    found = {}
    for location in locations:
        children = [location / requested_id] if requested_id else sorted(location.glob("*"))
        for child in children:
            if not child.is_dir() or child.is_symlink():
                continue
            summaries = list(child.glob("02_longitudinal_mapper_summary*.json"))
            summaries += list((child / "tables").glob("02_longitudinal_mapper_summary*.json"))
            if summaries:
                found[child.resolve()] = child
    return list(found.values())


def resolve_review_options(args, common):
    """Supply unique run IDs and select only an explicit or unique reference.

    A unique existing archive is reused in place and sealed externally by the
    normal runner. Discovery does not write anything, including in dry runs.
    Existing seals take precedence; multiple choices require an explicit ID/root.
    """
    resolved = copy.copy(args)
    dirs = study_paths(common)
    root = Path(args.output_root) if args.output_root is not None else dirs["tables"].parent / "longitudinal_graph_review"
    resolved.output_root = root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    resolved.run_id = _checked_id(args.run_id if args.run_id is not None else
                                  f"{args.mode}-{stamp}-{uuid4().hex[:8]}", "--run-id")
    resolved.run_id_generated = args.run_id is None
    requested = args.reference_run_id
    if requested is not None:
        requested = _checked_id(requested, "--reference-run-id")
    explicit_root = args.reference_root
    if explicit_root is not None:
        explicit_root = Path(explicit_root)
        resolved.reference_run_id = requested or _checked_id(explicit_root.name, "--reference-run-id")
        resolved.reference_root = explicit_root
    else:
        resolved.reference_run_id = requested
    seals = root / "references"
    if resolved.reference_run_id is None:
        choices = sorted(seals.glob("*.json"))
        if len(choices) > 1:
            raise ReviewStartupError(
                "Several frozen references are available: " + ", ".join(p.stem for p in choices) +
                ". Choose one with --reference-run-id; no reference was selected.",
                reason="ambiguous_historical_reference",
            )
        if choices:
            resolved.reference_run_id = _checked_id(choices[0].stem, "--reference-run-id")
    if resolved.reference_run_id is not None:
        seal = seals / f"{resolved.reference_run_id}.json"
        if seal.exists():
            # Do not hide malformed/stale seals by falling back to another archive.
            try:
                record = json.loads(seal.read_text())
                valid = (isinstance(record, dict) and record.get("reference_status") == "historical_frozen" and
                         record.get("reference_run_id") == resolved.reference_run_id and
                         isinstance(record.get("reference_root"), str) and isinstance(record.get("files"), dict) and bool(record["files"]))
            except (ValueError, OSError):
                valid = False
            if not valid:
                raise ReviewStartupError(f"Invalid reference seal: {seal}. Repair its provenance before running; the seal was not replaced.",
                                         reason="invalid_reference_seal")
            if not Path(record["reference_root"]).is_dir():
                raise ReviewStartupError(f"The archive recorded in {seal} is unavailable: {record['reference_root']}. "
                                         "Restore the archived files or reconcile the reference location before running.",
                                         reason="missing_reference_archive")
            resolved.reference_selection = "explicit_root" if explicit_root is not None else "explicit_id" if requested is not None else "unique_seal"
            if explicit_root is None:
                resolved.reference_root = Path(record["reference_root"])
            return resolved
    if explicit_root is not None:
        if not explicit_root.is_dir():
            raise ReviewStartupError(f"Historical archive not found: {explicit_root}. Point --reference-root to the existing complete S3-primary archive.")
        # Selecting a reference explicitly or unambiguously authorizes the
        # preservation preflight. This is byte sealing, never scientific approval.
        resolved.freeze_reference = True
        resolved.reference_selection = "explicit_root"
        return resolved
    locations = [root / "history", root.parent / "longitudinal_graph_history",
                 dirs["tables"].parent / "longitudinal_graph_history"]
    candidates = _archives(locations, requested)
    if len(candidates) > 1:
        raise ReviewStartupError(
            "Several historical archives are available: " + ", ".join(str(p) for p in candidates) +
            ". Select the intended archive with --reference-root; no reference was selected.",
            reason="ambiguous_historical_reference",
        )
    if candidates:
        resolved.reference_root = candidates[0]
        resolved.reference_run_id = requested or _checked_id(candidates[0].name, "--reference-run-id")
        resolved.freeze_reference = True
        resolved.reference_selection = "explicit_id_archive" if requested is not None else "unique_archive"
        return resolved
    raise ReviewStartupError(
        "No frozen S3-primary reference or historical archive was found. "
        "Use --reference-root /path/to/longitudinal_graph_history/<historical-run-id> "
        "(or --reference-run-id for an existing seal). A unique --run-id is generated automatically. "
        "Historical outputs were not rewritten and Script 03 remains blocked."
    )

#!/usr/bin/env python3
"""Optional, read-only Step15: private ESSDAI/medication documentary review."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import common
from src.block_A.secondary_trajectory_analysis import (
    ContractError, attach_context, read_table, sha256, validate_inputs, verified_manifest,
)
from src.block_A.intermittent_medication_analysis import (
    NAME_COLUMNS, NOTICE, cohort_summary, documentary_history, evidence_for_intervals,
    intervals, medication_bundle, selected_visits, threshold_comparison,
)

STEM = "step15_intermittent_medication_review"

# Internal run configuration: running this file without flags uses these values.
# Keep individual outputs outside the checkout, next to the project directory:
# /data/salazarda/data/obj1_sjd -> /data/salazarda/data/restricted/obj1_sjd
# ROOT is resolved, so equivalent institutional mount aliases use the same path.
# Every setting remains overridable through its corresponding CLI option.
DEFAULT_CONFIG = dict(
    integrated=common.INTEGRATED_LONGITUDINAL_PARQUET,
    baseline=common.INTEGRATED_BASELINE_PARQUET,
    patterns=common.DISEASE_ACTIVITY_PROGRESSION_DIR / "13_essdai_patient_observed_patterns_v2.parquet",
    patterns_manifest=common.BLOCKA_QC_DIR / "13_disease_activity_progression/13_essdai_observed_pattern_manifest_v2.json",
    spine=common.CLINICAL_VISIT_SPINE_PARQUET,
    medication_features=common.MEDICATION_EPISODE_FEATURES_PARQUET,
    medication_audit_root=ROOT / "outputs/audit/08_medications",
    pseudonym_key=None,  # Use medication_audit_root/restricted/.pseudonym_key.
    context=None,  # Use the Step10 context sibling when present.
    restricted_output_root=ROOT.parent / "restricted" / ROOT.name,
    threshold="both",
    input_origin="canonical",
    data_cutoff=datetime.now(timezone.utc).date().isoformat(),
    post_window_days=90,
    measurement_gap_days=365,
    minimum_cell=5,
    format="png",
    dpi=160,
    jump_figures=True,
    no_overwrite=True,
    dry_run=False,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, epilog=NOTICE,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--integrated", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--patterns", type=Path)
    p.add_argument("--patterns-manifest", type=Path)
    p.add_argument("--spine", type=Path)
    p.add_argument("--medication-features", type=Path)
    p.add_argument("--medication-audit-root", type=Path)
    p.add_argument("--pseudonym-key", type=Path, help="Established protected Step08 key, used in memory only; never generated")
    p.add_argument("--context", type=Path, help="Optional exact-key Step10 measurement-date context")
    p.add_argument("--restricted-output-root", type=Path, help="Authorized external directory named/under restricted, mode 0700; never inside a Git repository")
    p.add_argument("--threshold", choices=("3", "5", "both"))
    p.add_argument("--input-origin", choices=("canonical", "synthetic_validation"))
    p.add_argument("--data-cutoff")
    p.add_argument("--post-window-days", type=int)
    p.add_argument("--measurement-gap-days", type=int)
    p.add_argument("--minimum-cell", type=int, help="Local disclosure minimum (>=5); aggregates remain access controlled")
    p.add_argument("--format", choices=("png",))
    p.add_argument("--dpi", type=int)
    p.add_argument("--jump-figures", action=argparse.BooleanOptionalAction)
    p.add_argument("--no-overwrite", action="store_true", help="Existing runs are always preserved")
    p.add_argument("--dry-run", action="store_true", help="Validate and build all tables in memory; create no directories or products")
    p.set_defaults(**DEFAULT_CONFIG)
    args = p.parse_args(argv)
    if args.post_window_days < 0 or args.measurement_gap_days < 1 or args.minimum_cell < 5 or args.dpi < 50:
        p.error("invalid windows, disclosure minimum or DPI")
    try:
        date.fromisoformat(args.data_cutoff)
    except ValueError:
        p.error("data-cutoff must be YYYY-MM-DD")
    return args


def output_destination(path, origin):
    """Reject symlinks, public roots and repositories before any writes."""
    raw = Path(os.path.abspath(path))
    if "restricted" not in raw.parts:
        raise ContractError("output root must be named/under restricted")
    for ancestor in (raw, *raw.parents):
        if ancestor.is_symlink():
            raise ContractError("symlink in restricted output path")
        marker = ancestor / ".git"
        # Managed executors can mount empty .git guard directories in /tmp
        # and /workspace. A real checkout has a HEAD or a worktree gitfile.
        if marker.is_file() or (marker / "HEAD").exists():
            raise ContractError("restricted output cannot be inside a Git repository")
    resolved = raw.resolve()
    if resolved.is_relative_to(ROOT.resolve()):
        raise ContractError("restricted output cannot be inside the publication repository")
    if raw.exists() and (not raw.is_dir() or stat.S_IMODE(raw.stat().st_mode) != 0o700 or raw.stat().st_uid != os.getuid()):
        raise ContractError("existing restricted root must be owned by this user with permissions 0700")
    if origin == "synthetic_validation":
        resolved /= "synthetic_validation"
    return resolved / STEM


@contextmanager
def private_umask():
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


def safe_pseudonym(patient_id):
    # Filenames never contain the original research identifier, even if a
    # raw numeric MRN slipped into an otherwise reconciled upstream spine.
    return hashlib.sha256((STEM + "\0" + patient_id).encode()).hexdigest()[:24]


def run(args):
    required = [args.integrated, args.baseline, args.patterns, args.patterns_manifest,
                args.spine, args.medication_features]
    if any(not p.is_file() for p in required):
        raise ContractError("BLOCKED_UPSTREAM: canonical Step00/08/10/11/13 products are required")
    destination = output_destination(args.restricted_output_root, args.input_origin)
    if destination.exists() and not args.dry_run:
        raise ContractError("restricted run already exists; --no-overwrite protects it")
    manifest13, protected = verified_manifest(args.patterns_manifest, args.patterns, args.integrated, args.baseline, args.input_origin)
    inputs = {"step10": args.integrated, "step11": args.baseline, "step13_manifest": args.patterns_manifest, **protected}
    fingerprints = {role: dict(path=str(p.resolve()), sha256=sha256(p)) for role, p in inputs.items()}
    sequence = read_table(protected["selected_dates"])
    x, _, patterns = validate_inputs(read_table(args.integrated), read_table(args.baseline),
        read_table(protected["patterns"]), read_table(protected["eligibility"]), sequence, args.data_cutoff)
    context = args.context or args.integrated.with_name("10_integrated_longitudinal_context.parquet")
    if context.is_file():
        x = attach_context(x, read_table(context))
        fingerprints["step10_context"] = dict(path=str(context.resolve()), sha256=sha256(context))
    elif args.context:
        raise ContractError("BLOCKED_UPSTREAM: missing explicit Step10 context")
    key_path = args.pseudonym_key or args.medication_audit_root / "restricted/.pseudonym_key"
    features, catalog, links, manifest08, med_hashes = medication_bundle(
        args.medication_features, args.medication_audit_root, args.spine, x, key_path, args.data_cutoff)
    fingerprints.update(med_hashes)
    if any(Path(entry["path"]).is_relative_to(destination) or destination.is_relative_to(Path(entry["path"])) for entry in fingerprints.values()):
        raise ContractError("restricted output overlaps an input")
    thresholds = (3, 5) if args.threshold == "both" else (int(args.threshold),)
    comparison = threshold_comparison(patterns)
    bundles = {}
    for k in thresholds:
        roster, visits = selected_visits(x, patterns, sequence, k, args.data_cutoff)
        # Keep all seven original name fields, including nulls and their semantics.
        visits = visits.drop(columns=[c for c in NAME_COLUMNS if c in visits]).merge(features, on=["patient_id", "clinical_episode_id"], validate="one_to_one")
        jumps, domains, secondary = intervals(visits, k, args.measurement_gap_days, args.data_cutoff)
        history = documentary_history(catalog, links, roster.patient_id, args.data_cutoff)
        jumps, evidence, contrasts = evidence_for_intervals(jumps, history, features, args.post_window_days)
        counts = cohort_summary(jumps, evidence, k, args.minimum_cell)
        tables = dict(intermittent_patient_roster=roster, intermittent_essdai_visits=visits,
            intermittent_essdai_jumps=jumps, intermittent_essdai_domains=domains,
            secondary_endpoint_context=secondary, patient_medication_history=history,
            patient_medication_episode_inventory=features.loc[features.patient_id.isin(roster.patient_id)].copy(),
            patient_medication_episode_links=links.loc[links.patient_id.isin(roster.patient_id)].copy(),
            jump_medication_evidence=evidence, visit_medication_report_contrasts=contrasts,
            threshold_comparison=comparison)
        bundles[k] = (tables, counts)
    # A dry-run never creates the requested root or even a temporary figure.
    if args.dry_run:
        return {"status": "validated_dry_run_no_publication", "thresholds": list(thresholds), "input_origin": args.input_origin}
    # Check for concurrent source mutation before publishing a coherent snapshot.
    if any(sha256(entry["path"]) != entry["sha256"] for entry in fingerprints.values()):
        raise ContractError("upstream changed during analysis; no publication")
    from src.block_A.intermittent_medication_figures import timeline, cohort_figure
    import matplotlib.pyplot as plt
    code_files = [Path(__file__), *Path(__file__).parent.glob("intermittent_medication_*.py")]
    code_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in code_files}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    # Private staging on the destination filesystem; one atomic run rename,
    # preserving existing runs even if a later figure fails.
    with private_umask():
        args.restricted_output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".step15-review-", dir=destination.parent) as tmp:
            stage = Path(tmp) / "run"
            stage.mkdir(mode=0o700)
            for k, (tables, counts) in bundles.items():
                folder = stage / "clinical_baseline" / f"delta{k}"
                figures = folder / "figures"
                figures.mkdir(mode=0o700, parents=True)
                for name, table in tables.items():
                    table.to_parquet(folder / (name + ".parquet"), index=False)
                aggregate = stage / "controlled_aggregates" / f"delta{k}"
                aggregate.mkdir(mode=0o700, parents=True)
                counts.to_csv(aggregate / "cohort_documentation_summary.csv", index=False)
                figure = cohort_figure(counts, k, args.input_origin)
                try:
                    figure.savefig(aggregate / "cohort_documentation.png", dpi=args.dpi)
                finally:
                    plt.close(figure)
                for pid, visits in tables["intermittent_essdai_visits"].groupby("patient_id"):
                    safe = safe_pseudonym(pid)
                    jumps = tables["intermittent_essdai_jumps"].loc[lambda f: f.patient_id.eq(pid)]
                    history = tables["patient_medication_history"].loc[lambda f: f.patient_id.eq(pid)]
                    requests = [(f"patient_{safe}_timeline.png", None)]
                    if args.jump_figures:
                        requests += [(f'jump_{safe}_{r.jump_id}.png', (r.from_date, r.to_date)) for r in jumps.itertuples()]
                    for name, zoom in requests:
                        prefix = "SYNTHETIC VALIDATION | " if args.input_origin == "synthetic_validation" else ""
                        figure = timeline(visits, jumps, history, k, prefix + "Private subject " + safe, zoom)
                        try:
                            figure.savefig(figures / name, dpi=args.dpi)
                        finally:
                            plt.close(figure)
                checks = [dict(check=name, status="PASS") for name in (
                    "step13_all_artifact_hashes", "step10_11_identity", "step00_08_10_identity",
                    "step08_generation_hashes_hmac_scope", "selected_original_dates_and_scores",
                    "document_episode_identity", "consecutive_intervals", "no_confirmed_exposure_inferred",
                    "restricted_only", "no_upstream_writes")]
                pd.DataFrame(checks).to_csv(folder / "qc_checks.csv", index=False)
                metadata = dict(status="PASS", input_origin=args.input_origin, threshold_abs=k,
                    definition_version=manifest13["definition_version"], date_policy="original_selected_dates; (from,to]; no interpolation",
                    post_window_days=args.post_window_days, measurement_gap_days=args.measurement_gap_days,
                    minimum_cell=args.minimum_cell, confirmed_medication_exposure=False,
                    cohort_membership="retrospective_not_baseline_predictor", notice=NOTICE,
                    weighted_domain_attribution="not_certified_not_computed", step08_restricted_generation=manifest08["restricted_generation"],
                    threshold_selection=dict(n_k3=int(comparison.intermittent_k3.sum()), n_k5=int(comparison.intermittent_k5.sum()),
                        n_intersection=int((comparison.intermittent_k3 & comparison.intermittent_k5).sum()),
                        n_k3_only=int((comparison.intermittent_k3 & ~comparison.intermittent_k5).sum()),
                        n_k5_only=int((~comparison.intermittent_k3 & comparison.intermittent_k5).sum())),
                    inputs=fingerprints, code_commit=commit, source_sha256=code_hashes,
                    run_at=datetime.now(timezone.utc).isoformat(),
                    outputs_sha256={str(p.relative_to(stage)): sha256(p) for p in [*folder.rglob("*"), *aggregate.rglob("*")] if p.is_file()})
                (folder / "run_manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            for path in stage.rglob("*"):
                os.chmod(path, 0o700 if path.is_dir() else 0o600)
            if destination.exists():
                raise ContractError("restricted run appeared concurrently; no overwrite")
            os.rename(stage, destination)
    return {"status": "published_restricted", "thresholds": list(thresholds), "input_origin": args.input_origin}


def main(argv=None):
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception as error:
        # Upstream exceptions can contain actual identifiers or medication text.
        # Console output is deliberately limited to fixed diagnostic categories.
        blocked = isinstance(error, ContractError) and str(error).startswith("BLOCKED_UPSTREAM")
        print(json.dumps({"status": "BLOCKED_UPSTREAM" if blocked else "BLOCKED_VALIDATION",
                          "message": "No clinical products published. Check authenticated upstream inputs, identity and protected output policy."}), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

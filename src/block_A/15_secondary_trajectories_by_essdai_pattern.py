#!/usr/bin/env python3
"""Read-only Step 15: observed secondary measurements in fixed ESSDAI v2 groups."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import date, datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import platform

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import common
import config
from src.block_A.secondary_trajectory_analysis import (
    ContractError, DEFINITION, STEM, WARNINGS, attach_context, fixed_groups,
    observations, read_table, sha256, summarize, support_table, validate_inputs, verified_manifest,
)
from src.block_A.secondary_trajectory_catalog import build_catalog
from src.block_A.secondary_trajectory_figures import plot_variable
from src.block_A.observed_patterns_v2_figures import LABELS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "No classification, model fitting, imputation or upstream writes. Missing/unverified upstream blocks the run. "
        "Synthetic bundles are isolated under output-root/synthetic_validation and visibly labeled. "
        "Windows are [start,end) years, one last observed contribution per patient; actual trajectory dates are retained."))
    parser.add_argument("--integrated", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET, help="Unmodified canonical Step 10 input hashed by Step 13")
    parser.add_argument("--baseline", type=Path, default=common.INTEGRATED_BASELINE_PARQUET, help="Unmodified official Step 11 baseline hashed by Step 13")
    parser.add_argument("--patterns", type=Path, default=common.DISEASE_ACTIVITY_PROGRESSION_DIR/"13_essdai_patient_observed_patterns_v2.parquet", help="Manifest-authenticated protected v2 assignment, never recalculated")
    parser.add_argument("--patterns-manifest", type=Path, default=common.BLOCKA_QC_DIR/"13_disease_activity_progression/13_essdai_observed_pattern_manifest_v2.json", help="All listed artifacts and Step 10/11 hashes must match")
    parser.add_argument("--context", type=Path, help="Step 10 context sidecar; by default use sibling 10_integrated_longitudinal_context.parquet when present")
    parser.add_argument("--source-registry", type=Path, help="Step 10 variable registry; by default use sibling 10_variable_registry.csv when present")
    parser.add_argument("--output-root", type=Path, default=common.PROJECT_ROOT, help="Repository-style output root; writes only isolated Step 15 bundles")
    parser.add_argument("--threshold", choices=("3", "5", "both"), default="3", help="Fixed ESSDAI classification threshold; separate bundles for 3 and 5")
    parser.add_argument("--time-axis", choices=("clinical_baseline", "essdai_index"), default="clinical_baseline", help="Actual date clock; ESSDAI-index sensitivity retains negative times in a separate namespace")
    parser.add_argument("--domains", nargs="+", help="Explicit domain selection; other candidates remain inventoried")
    parser.add_argument("--variables", nargs="+", help="Explicit variable IDs; other candidates remain inventoried")
    parser.add_argument("--max-variables", type=int, help="Development-only limit; excluded candidates are recorded")
    parser.add_argument("--format", choices=("png",), default="png", help="PNG only; no PDF output")
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--window-years", type=float, default=1., help="Width of descriptive windows; actual points are never moved")
    parser.add_argument("--minimum-support", type=int, default=5, help="Minimum unique patients for displayed category confidence intervals; sparse estimates flagged")
    parser.add_argument("--data-cutoff", default=datetime.now(timezone.utc).date().isoformat(), help="Latest allowed real measurement date (YYYY-MM-DD)")
    parser.add_argument("--input-origin", choices=("canonical", "synthetic_validation"), default="canonical", help="Must equal authenticated Step 13 manifest origin")
    parser.add_argument("--dry-run", action="store_true", help="Validate provenance, all inputs, catalog and full analysis; zero final writes or directories")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Flush aggregate phase/variable progress to stderr")
    overwrite = parser.add_mutually_exclusive_group()
    overwrite.add_argument("--no-overwrite", action="store_true", help="Default: reject any existing Step 15 bundle before writing")
    overwrite.add_argument("--overwrite", action="store_true", help="Replace only Step 15 after staging; requires explicit --backup-root")
    parser.add_argument("--backup-root", type=Path, help="Explicit backup directory below effective-output-root/outputs/backups/15_secondary_trajectories_by_essdai_pattern")
    args = parser.parse_args(argv)
    if args.window_years <= 0 or not float(args.window_years) < float("inf") or args.minimum_support < 1 or args.dpi < 50:
        parser.error("window-years must be finite and positive, minimum-support >=1, dpi >=50")
    if args.max_variables is not None and args.max_variables < 1:
        parser.error("max-variables must be positive")
    if args.overwrite and not args.backup_root and not args.dry_run:
        parser.error("--overwrite requires --backup-root so the prior bundle is retained")
    try:
        date.fromisoformat(args.data_cutoff)
    except (ValueError, TypeError):
        parser.error("data-cutoff must be a calendar date")
    return args


def directories(root, axis):
    paths = {area: common.script_output_dir(root/"outputs"/area/"blockA", STEM) for area in ("figures", "tables", "qc")}
    return {area: p/"essdai_index" if axis == "essdai_index" else p/"clinical_baseline" for area, p in paths.items()}


def concatenate(frames, columns):
    nonempty = [f for f in frames if not f.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=columns)


def source_code_hashes(source_files, project_root):
    """Record source identities using canonical paths (supports Biowulf /data aliases)."""
    canonical_root = Path(project_root).resolve()
    return {
        str(Path(path).resolve().relative_to(canonical_root)): sha256(path)
        for path in source_files
    }


class Runtime:
    """Aggregate-only diagnostics; never log patient identities or measurements."""
    def __init__(self, progress):
        self.progress = progress
        self.started = self.last = time.perf_counter()
        self.phase = "input provenance"
        self.phases = {}
        self.variables = []

    def log(self, message):
        if self.progress:
            print("[Step15] "+message, file=sys.stderr, flush=True)

    def mark(self, phase):
        now = time.perf_counter()
        self.phases[self.phase] = self.phases.get(self.phase, 0.) + now-self.last
        self.last, self.phase = now, phase
        self.log(phase)

    def metrics(self):
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_mb = rss/(1024*1024 if sys.platform == "darwin" else 1024)
        except (ImportError, AttributeError, OSError):
            rss_mb = "not_available"
        return dict(wall_seconds=time.perf_counter()-self.started, rss_peak_mb=rss_mb,
                    phases_wall_seconds=dict(self.phases), variables=list(self.variables),
                    python=platform.python_version(), pandas=pd.__version__)


def run(args):
    runtime = Runtime(args.progress)
    with ExitStack() as cleanup:
        try:
            report = _run(args, cleanup, runtime)
            runtime.mark("Completed")
            report["runtime"] = runtime.metrics()
            return report
        except BaseException:
            runtime.log("Failed during "+runtime.phase+"; staging discarded")
            raise


def _run(args, cleanup, runtime):
    required = [args.integrated, args.baseline, args.patterns, args.patterns_manifest]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ContractError("BLOCKED_UPSTREAM: missing canonical inputs/artifacts: "+", ".join(missing))
    manifest, protected = verified_manifest(args.patterns_manifest, args.patterns, args.integrated, args.baseline, args.input_origin)
    inputs = {"step10": args.integrated, "step11": args.baseline, "step13_manifest": args.patterns_manifest, **protected}
    context = args.context or args.integrated.with_name("10_integrated_longitudinal_context.parquet")
    registry_path = args.source_registry or args.integrated.with_name("10_variable_registry.csv")
    for role, path, explicit in (("context", context, args.context), ("source_registry", registry_path, args.source_registry)):
        if path.is_file():
            inputs[role] = path
        elif explicit:
            raise ContractError("missing explicit "+role)
    input_hashes = {role: dict(path=str(p), sha256=sha256(p)) for role, p in inputs.items()}
    # Check code provenance before expensive rendering: /data and /vf/users
    # can refer to the same Biowulf directory but differ as lexical paths.
    source_files = [Path(__file__), *Path(__file__).parent.glob("secondary_trajectory_*.py")]
    source_sha256 = source_code_hashes(source_files, ROOT)
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    runtime.mark("Read / validate inputs")
    x, baseline, patterns = validate_inputs(read_table(args.integrated), read_table(args.baseline),
        read_table(protected["patterns"]), read_table(protected["eligibility"]), read_table(protected["selected_dates"]), args.data_cutoff)
    runtime.mark("Context alignment")
    n_integrated_rows, n_integrated_columns = x.shape
    n_patients = x.patient_id.nunique()
    integrated_columns = list(x)
    x = attach_context(x, read_table(context) if "context" in inputs else None)
    registry = read_table(registry_path) if "source_registry" in inputs else None
    # Clinical candidates must exist in analytic Step 10; context adds only
    # dates, units and evaluability, never a previously excluded measurement.
    runtime.mark("Variable catalog")
    specs, catalog = build_catalog(x[list(integrated_columns)], registry)
    available_ids = {s.variable_id for s in specs.values()}
    available_domains = {s.domain for s in specs.values()}
    if args.variables and not set(args.variables).issubset(available_ids):
        raise ContractError("unknown requested variable IDs")
    if args.domains and not set(args.domains).issubset(available_domains):
        raise ContractError("unknown requested domains")
    selected = [s for s in specs.values() if (not args.variables or s.variable_id in args.variables) and (not args.domains or s.domain in args.domains)]
    if args.max_variables:
        selected = selected[:args.max_variables]
    selected_cols = {s.canonical_column for s in selected}
    unselected = catalog.canonical_column.isin(set(specs)-selected_cols)
    catalog.loc[unselected, "status"] = "not_selected"
    catalog.loc[unselected, "exclusion_or_limitation_reason"] = "explicit CLI selection/development limit"
    root = args.output_root.resolve()
    if args.input_origin == "synthetic_validation":
        root /= "synthetic_validation"
    dirs = directories(root, args.time_axis)
    thresholds = [3, 5] if args.threshold == "both" else [int(args.threshold)]
    destinations = {f"{area}/delta{k}": path/f"delta{k}" for area, path in dirs.items() for k in thresholds}
    for path in inputs.values():
        if any(path.resolve().is_relative_to(d.resolve()) for d in dirs.values()):
            raise ContractError("input path overlaps Step 15 output destination")
    if not args.dry_run and not args.overwrite and any(d.exists() for d in destinations.values()):
        raise FileExistsError("Step 15 bundle exists; protected by --no-overwrite")
    if args.backup_root:
        backup_root = args.backup_root.resolve()
        if not backup_root.is_relative_to((root/"outputs/backups"/STEM).resolve()):
            raise ContractError("backup root must be inside this Step 15 bundle's dedicated outputs/backups directory")
        if backup_root == root or any(backup_root.is_relative_to(d.resolve()) or d.resolve().is_relative_to(backup_root) for d in dirs.values()):
            raise ContractError("backup root overlaps final bundle")
        if any(p.resolve().is_relative_to(backup_root) for p in inputs.values()):
            raise ContractError("backup root overlaps protected input storage")
    runtime.log(f"Provenance OK | Step10 rows={n_integrated_rows} cols={n_integrated_columns}")
    n_candidates = int(catalog.loc[catalog.canonical_column.isin(selected_cols), "status"].eq("candidate").sum())
    runtime.log(f"Inventory: eligible selected={n_candidates} thresholds={thresholds}")
    stage, staging = None, None
    if not args.dry_run:
        # Isolated staging on the destination filesystem, cleaned on every exit.
        root.mkdir(parents=True, exist_ok=True)
        stage = Path(cleanup.enter_context(tempfile.TemporaryDirectory(prefix=".step15-", dir=root)))
        staging = directories(stage, args.time_axis)
        for d in staging.values():
            d.mkdir(parents=True)
    bundles, checks = {}, []
    for threshold in thresholds:
        groups = fixed_groups(patterns, threshold)
        original_groups = groups.copy(deep=True)
        cat = catalog.copy(deep=True)
        support, summaries, categories, availability, plotted = [], [], [], [], []
        n_png = 0
        variable_index = 0
        for spec in selected:
            eligible = cat.loc[cat.canonical_column.eq(spec.canonical_column), "status"].iloc[0] == "candidate"
            if not eligible:
                continue
            variable_index += 1
            runtime.mark(f"delta{threshold} [{variable_index}/{n_candidates}] {spec.domain}/{spec.variable_id}: analysis")
            variable_start = time.perf_counter()
            frame, status, qc, unit = observations(x, groups, spec, args.data_cutoff, args.time_axis, args.window_years)
            extraction_seconds = time.perf_counter()-variable_start
            checks += [dict(**row, threshold_abs=threshold) for row in qc]
            mask = cat.canonical_column.eq(spec.canonical_column)
            cat.loc[mask, "status"] = status if status in ("plot_ready", "baseline_only_plotted") else "excluded"
            cat.loc[mask, "unit"] = unit
            if status not in ("plot_ready", "baseline_only_plotted"):
                cat.loc[mask, "exclusion_or_limitation_reason"] = status
            variable_support = support_table(frame, groups, spec, threshold)
            support.append(variable_support)
            for col, value in (("n_any", variable_support.n_any.sum()), ("n_repeated", variable_support.n_repeated.sum()),
                               ("n_groups_supported", variable_support.n_any.gt(0).sum())):
                cat.loc[mask, col] = value
            if frame.empty:
                runtime.variables.append(dict(domain=spec.domain, variable_id=spec.variable_id, threshold_abs=threshold,
                    n_observations_used=0, n_png=0, extraction_seconds=extraction_seconds,
                    analysis_seconds=time.perf_counter()-variable_start, figure_seconds=0.))
                runtime.log(f"delta{threshold} {spec.variable_id}: {status}")
                continue
            analyzed = frame if status == "plot_ready" else frame.groupby("patient_id", sort=False).head(1)
            summary, category, avail = summarize(analyzed, spec, threshold, args.time_axis, unit, args.minimum_support,
                                                 profile=status == "baseline_only_plotted")
            summaries.append(summary)
            categories.append(category)
            availability.append(avail)
            kinds = (["absolute", "delta"] if spec.type == "numeric" else ["prevalence" if spec.type == "binary" else "category_distribution"]) if status == "plot_ready" else ["first_available_profile"]
            analysis_seconds = time.perf_counter()-variable_start
            render_start = time.perf_counter()
            if not args.dry_run:
                runtime.mark(f"delta{threshold} {spec.domain}/{spec.variable_id}: rendering / staging")
                # Historical numeric first-available figures use real windows,
                # whereas their published table pools first measurements.
                plot_summaries = ((summary, category) if not (status == "baseline_only_plotted" and spec.type == "numeric") else
                    summarize(analyzed, spec, threshold, args.time_axis, unit, args.minimum_support)[:2])
                group_counts = {key: int(groups.observed_pattern.eq(key).sum()) for key in LABELS}
                for kind in kinds:
                    destination = staging["figures"]/f"delta{threshold}"/spec.domain/(spec.variable_id+"_"+kind+".png")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    plot_variable(frame, groups, variable_support, spec, threshold, kind, destination,
                                  time_axis=args.time_axis, unit=unit, origin=args.input_origin,
                                  dpi=args.dpi, minimum_support=args.minimum_support, summaries=plot_summaries)
                    plotted.append(dict(path=str(destination.relative_to(stage)), variable=spec.variable_id, type=kind,
                        threshold_abs=threshold, time_axis=args.time_axis, unit=unit, groups_present=json.dumps(list(LABELS)),
                        N_group=json.dumps(group_counts, sort_keys=True), n_any=int(variable_support.n_any.sum()),
                        n_repeated=int(variable_support.n_repeated.sum()), reference_policy=spec.reference_policy,
                        source_producer=spec.source_producer, observation_semantics=spec.observation_semantics,
                        input_origin=args.input_origin, sha256=sha256(destination)))
            n_png += len(kinds)
            figure_seconds = time.perf_counter()-render_start if not args.dry_run else 0.
            runtime.variables.append(dict(domain=spec.domain, variable_id=spec.variable_id, threshold_abs=threshold,
                n_observations_used=len(frame), n_png=0 if args.dry_run else len(kinds),
                extraction_seconds=extraction_seconds, analysis_seconds=analysis_seconds, figure_seconds=figure_seconds))
            runtime.log(f"delta{threshold} {spec.variable_id} | observed={len(frame)} | analysis={analysis_seconds:.3f}s | PNG={0 if args.dry_run else len(kinds)} ({figure_seconds:.3f}s)")
            del analyzed, frame
        # Include zero availability for every explicitly selected clinical
        # candidate, even if the integrated column is absent.
        present = {r.variable.iloc[0] for r in support}
        for spec in selected:
            if spec.variable_id not in present:
                support.append(support_table(pd.DataFrame(), groups, spec, threshold))
        pd.testing.assert_frame_equal(groups, original_groups)
        checks += [dict(variable="all", threshold_abs=threshold, check="fixed_group_assignment_and_denominators", status="PASS", n=0)]
        bundles[threshold] = dict(catalog=cat, support=concatenate(support, ["variable", "observed_pattern", "N_group"]),
            summary=concatenate(summaries, ["variable", "group", "threshold_abs", "window", "n_patients_unique"]),
            category=concatenate(categories, ["variable", "group", "threshold_abs", "window", "n_evaluable", "percentage"]),
            availability=concatenate(availability, ["variable", "group", "threshold_abs", "window", "source_protocol", "n_patients_unique"]),
            plotted=plotted, n_png=n_png, group_counts={key: int(groups.observed_pattern.eq(key).sum()) for key in LABELS})
    checks += [dict(variable="all", threshold_abs="all", check="step13_artifacts_and_input_identity", status="PASS", n=0)]
    if "context" not in inputs:
        checks.append(dict(variable="all", threshold_abs="all", check="context_unavailable; dates/units/protocols may be excluded", status="WARN", n=1))
    report = dict(status="TESTED_SYNTHETIC / NOT_RUN_CANONICAL" if args.input_origin == "synthetic_validation" else "DESCRIPTIVE_CANONICAL_RUN",
                  input_origin=args.input_origin, groups={str(k): b["group_counts"] for k, b in bundles.items()},
                  variables={str(k): b["catalog"].status.value_counts().to_dict() for k, b in bundles.items()},
                  n_png=sum(b["n_png"] for b in bundles.values()), dry_run=args.dry_run)
    report.update(n_variables_selected=n_candidates, n_variables_plotted=sum(
        int(b["catalog"].status.isin(["plot_ready", "baseline_only_plotted"]).sum()) for b in bundles.values()),
        n_patients=int(n_patients), n_integrated_rows=n_integrated_rows, n_integrated_columns=n_integrated_columns)
    if args.dry_run:
        return report
    runtime.mark("Export aggregate tables")
    for threshold, bundle in bundles.items():
        tables = staging["tables"]/f"delta{threshold}"
        tables.mkdir()
        cat = bundle["catalog"]
        for name, data in (("variable_registry", cat), ("group_support_by_variable", bundle["support"]),
                           ("trajectory_summary_by_window", bundle["summary"]), ("binary_category_summary", bundle["category"]),
                           ("availability_by_protocol", bundle["availability"]),
                           ("excluded_variables", cat.loc[~cat.status.eq("plot_ready")])):
            data.to_csv(tables/(name+".csv"), index=False)
        plotted = bundle["plotted"]
        pd.DataFrame(plotted, columns=list(plotted[0]) if plotted else ["path", "variable", "type", "threshold_abs", "sha256"]).to_csv(tables/"plot_manifest.csv", index=False)
    runtime.mark("Final input hashes / provenance verification")
    # Verify bytes again after analysis; a changed input invalidates all output.
    for role, path in inputs.items():
        if sha256(path) != input_hashes[role]["sha256"]:
            raise ContractError("input changed during run: "+role)
    verified_manifest(args.patterns_manifest, args.patterns, args.integrated, args.baseline, args.input_origin)
    if source_code_hashes(source_files, ROOT) != source_sha256:
        raise ContractError("Step 15 source files changed during run")
    for threshold, bundle in bundles.items():
        qc_dir = staging["qc"]/f"delta{threshold}"
        qc_dir.mkdir()
        local_checks = [c for c in checks if c["threshold_abs"] in (threshold, "all")]
        pd.DataFrame(local_checks).to_csv(qc_dir/"qc_checks.csv", index=False)
        files = [p for area in staging.values() for p in (area/f"delta{threshold}").rglob("*") if p.is_file()]
        local_report = dict(report, groups={str(threshold): bundle["group_counts"]},
                            variables={str(threshold): report["variables"][str(threshold)]}, n_png=bundle["n_png"])
        local_report["runtime"] = runtime.metrics()
        run_manifest = dict(**local_report, definition_version=DEFINITION, created_at_utc=datetime.now(timezone.utc).isoformat(),
            code_commit=code_commit, source_sha256=source_sha256,
            inputs=input_hashes, step13_artifacts=manifest["artifacts"], qc_status="PASS_WITH_WARNINGS" if any(c["status"] == "WARN" for c in local_checks) else "PASS",
            parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, random_seed=config.RANDOM_SEED,
            window_policy="[start,end) actual years; last observation per patient/window", duplicate_policy="official episode on its baseline date wins; concordant last episode ID; discordant dates excluded",
            scientific_limitations=WARNINGS, privacy="Controlled research PNGs and aggregates only; no patient IDs, individual dates or mappings in outputs",
            artifacts=[dict(path=str(p.relative_to(stage)), sha256=sha256(p)) for p in sorted(files)])
        (qc_dir/"run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False)+"\n")
    staged_bundles = {key: staging[key.split("/")[0]]/key.split("/")[1] for key in destinations}
    runtime.mark("Validation PASS | installing verified bundle")
    install_bundle(staged_bundles, destinations, args.backup_root)
    return report


def install_bundle(staging, destinations, backup_root):
    """Install complete directories with rollback; retain old bundle on overwrite."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = Path(backup_root)/stamp if backup_root else None
    moved, installed = [], []
    try:
        for area, dest in destinations.items():
            if dest.exists():
                if backup is None:
                    raise FileExistsError("Step 15 destination appeared during staging")
                saved = backup/area
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(dest, saved)
                moved.append((dest, saved))
                shutil.rmtree(dest)
        # QC manifest is the completion marker and is installed last.
        for area in sorted(staging, key=lambda name: name.startswith("qc")):
            dest = destinations[area]
            dest.parent.mkdir(parents=True, exist_ok=True)
            installed.append(dest)
            shutil.move(str(staging[area]), str(dest))
    except BaseException:
        for dest in reversed(installed):
            if dest.exists():
                shutil.rmtree(dest)
        for dest, saved in moved:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(saved, dest)
        raise


def main(argv=None):
    args = parse_args(argv)
    try:
        report = run(args)
    except (ContractError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

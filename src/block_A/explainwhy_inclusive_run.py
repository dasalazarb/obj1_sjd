"""Opt-in, staged writer for Step 14; never writes historical product names."""
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

from src.block_A import explainwhy_inclusive as core
from src.block_A import observed_patterns_v2 as v2
from src.block_A.explainwhy_feasibility import attach_audit_context
from src.block_A.explainwhy_inclusive_figures import make_figures
from src.block_A.explainwhy_inclusive_models import build_tasks, fit_models, feasibility_table, domain_index_support

PREFIX = core.PREFIX


def directories(root):
    stem = "14_risk_factors_progression"
    return {"analytic": Path(root)/"data/analytic/blockA"/stem,
        **{area: Path(root)/"outputs"/area/"blockA"/stem for area in ("tables", "qc", "figures")}}


def snapshot(root, paths, step14):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=step14.ROOT, text=True).strip()
    historical = {str(p.relative_to(root)): core.sha256(p)
        for top in (Path(root)/"data/analytic/blockA", Path(root)/"outputs") if top.exists()
        for p in top.rglob("*") if p.is_file() and p.name.startswith(("13_", "14_"))
        and not p.name.startswith(PREFIX) and "backups" not in p.parts}
    return dict(branch=git("branch", "--show-current"), code_commit=git("rev-parse", "HEAD"),
        scientific_diff=git("diff", "--stat", "--", "src", "config.py", "common.py"),
        working_tree_status=git("status", "--porcelain"),
        runtime_fingerprint=step14.runtime_code_fingerprint(),
        inputs={role: {"path": str(p), "available": p.is_file(),
            "sha256": core.sha256(p) if p.is_file() else None} for role, p in paths.items()},
        historical_products_before=historical,
        packages={name: importlib.metadata.version(name) for name in ("pandas", "numpy", "statsmodels", "lifelines", "pyarrow")})


def install_bundle(stage, root, existing):
    """Back up only this prefix; install manifest last; roll back exceptions."""
    root = Path(root)
    backup = None
    if existing:
        backup = root/"outputs/backups/14_risk_factors_progression"/(
            "inclusive_v2_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        for old in existing:
            dest = backup/old.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, dest)
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    manifest = [p for p in files if p.name == PREFIX+"manifest.json"]
    files = [p for p in files if p not in manifest] + manifest
    installed = []
    try:
        # Stale products from a previous successful bundle cannot survive a
        # blocked rerun and masquerade as current complete results.
        for old in existing:
            old.unlink()
        for source in files:
            dest = root/source.relative_to(stage)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, dest)
            installed.append(dest)
    except BaseException:
        for dest in installed:
            dest.unlink(missing_ok=True)
        if backup:
            for old in existing:
                old.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup/old.relative_to(root), old)
        raise


def _write_csv(frame, directory, name):
    if not len(frame.columns):
        frame = pd.DataFrame(columns=["status", "reason"])
    frame.to_csv(directory/(PREFIX+name+".csv"), index=False)


def _closure(manifest):
    rows = ["# Inclusive ESSDAI extension", "", f"Status: {manifest['status']}",
        f"Input origin: {manifest['input_origin']}", f"Code commit: {manifest['snapshot']['code_commit']}", "",
        "Clinical review: PENDING_CLINICAL_REVIEW. Privacy/institutional review: PENDING.",
        "No institutional closure is certified by software tests.", "",
        "Time zero is the first valid ESSDAI index. No pre-index event-free time is assumed.",
        "Observed detection is interval censored, subject to delayed entry and differential visit opportunity.",
        "Age is age at the official clinical baseline, including for alternative-index subjects.",
        "Treatment is not_implemented; disease duration is unavailable without verified Step 11 evidence.",
        "Biological-domain/biomarker associations may be algebraically coupled to the total ESSDAI.",
        "Retrospective pattern labels are descriptions, never prospective predictors.", ""]
    for name, state in manifest.get("components", {}).items():
        rows.append(f"- {name}: {state}")
    if manifest.get("blocking_reason"):
        rows += ["", manifest["blocking_reason"]]
    return "\n".join(rows)+"\n"


def _sensitivity_legacy_risks(risks, legacy):
    out = {}
    old = legacy.set_index("patient_id")
    for k, risk in risks.items():
        g = risk.loc[risk.index_source.eq(v2.OFFICIAL)].copy()
        if g.empty:
            out[k] = g
            continue
        g["event"] = g.patient_id.map(old[f"ever_worsened_ge{k}"]).astype(int)
        g["first_event_date"] = pd.to_datetime(g.patient_id.map(old[f"first_observed_worsening_ge{k}_date"]))
        end = pd.to_datetime(g.first_event_date.where(g.event.eq(1), g.patient_id.map(old.last_date)))
        g["event_or_censor_time_years"] = (end-g.essdai_index_date).dt.days/365.25
        out[k] = g
    return out


def _observation_strata(risks):
    frames = []
    for n in (3, 4):
        frames.append(core.event_summary({k: d.loc[d.n_distinct_essdai_dates.ge(n)] for k, d in risks.items()}, f"S4:{n}+dates"))
    for label, lo, hi in (("<2y", 0, 2), ("2-<5y", 2, 5), (">=5y", 5, np.inf)):
        frames.append(core.event_summary({k: d.loc[d.last_followup_years.ge(lo) & d.last_followup_years.lt(hi)]
            for k, d in risks.items()}, "S5:"+label))
    return pd.concat(frames, ignore_index=True)


def _attrition(results, risks, intervals):
    cols = ["population", "analysis_type", "outcome", "predictor", "sensitivity", "n_cohort", "n_at_risk",
        "n_at_risk_fallback", "n_with_predictor_available", "n_fallback_with_predictor", "n_complete_cases",
        "n_complete_case_fallback", "n_fitted", "n_fitted_fallback", "n_events", "support_reason"]
    out = results[cols].copy()
    events = risks[3].loc[risks[3].event.eq(1)]
    counts = {}
    for predictor in results.predictor.unique():
        col = "pred__"+predictor
        available = intervals.loc[intervals.outcome_evaluable.eq(True) & intervals.at_risk_first_delta3.eq(True) &
            intervals.event_first_delta3.eq(1) & intervals.get(col, pd.Series(np.nan, index=intervals.index)).notna()]
        counts[predictor] = len(events)-events.patient_id.isin(available.patient_id).sum()
    out["event_not_modelable_due_to_missing_from_or_predictor"] = out.predictor.map(counts).where(
        out.analysis_type.eq("timevarying") & out.outcome.eq("first_delta3") & out.population.ne("official_only"))
    return out


def run_inclusive(args, step14):
    root = Path(args.output_root).resolve()
    dirs = directories(root)
    existing = [p for d in dirs.values() if d.exists() for p in d.glob(PREFIX+"*") if p.is_file()]
    if existing and not args.overwrite and not args.dry_run:
        raise FileExistsError("Inclusive v2 outputs exist; --no-overwrite protects this family")
    options = dict(step10=args.integrated, step11=args.baseline, context=args.context,
        registry=args.variable_registry, patterns=args.patterns_v2, sequence=args.selected_dates_v2,
        eligibility=args.eligibility_v2, manifest=args.pattern_manifest_v2, legacy=args.legacy_endpoints)
    unspecified = [role for role, path in options.items() if path is None]
    paths = {role: Path(path) for role, path in options.items() if path is not None}
    before = snapshot(root, paths, step14)
    manifest = dict(rule_version=core.RULE_VERSION, analysis_family=core.FAMILY, input_origin=args.input_origin,
        status="BLOCKED", qc_status="NOT_RUN", snapshot=before, index_time_origin="essdai_index_date",
        clinical_review="PENDING_CLINICAL_REVIEW", privacy_review="PENDING_INSTITUTIONAL_REVIEW",
        canonical_execution="NOT_RUN", components={"implementation": "IMPLEMENTED", "clinical_results": "BLOCKED"})
    tables, analytic, checks = {}, {}, {}
    try:
        if unspecified:
            raise ValueError("Explicit inclusive input options required: "+", ".join(unspecified))
        missing = [role for role, p in paths.items() if not p.is_file()]
        if missing:
            raise ValueError("NOT RUN — required canonical inputs unavailable: "+", ".join(missing))
        producer = json.loads(paths["manifest"].read_text())
        hashes = core.validate_manifest(producer, paths, args.input_origin)
        raw, baseline = step14.read_table(paths["step10"]), step14.read_table(paths["step11"])
        patterns = pd.read_parquet(paths["patterns"])
        audit, sequence = pd.read_parquet(paths["eligibility"]), pd.read_parquet(paths["sequence"])
        clean, official = core.reconcile_products(raw, baseline, patterns, audit, sequence)
        legacy = pd.read_parquet(paths["legacy"])
        cohort = core.index_cohort(audit, hashes)
        risks = {k: core.first_event_riskset(cohort, sequence, k) for k in (3, 5)}
        checks["legacy_bridge_qc"] = core.legacy_bridge(clean, risks, legacy, step14)
        registry = pd.read_csv(paths["registry"])
        context = step14.read_table(paths["context"])
        clean, official = attach_audit_context(clean, official, context, step14)
        predictors, specs, availability = core.build_index_predictors(clean, official, cohort, step14, registry)
        intervals, composition = core.build_lagged_intervals(clean, cohort, sequence, risks[3], specs, step14, registry)
        checks["temporal_leakage_qc"] = core.temporal_qc(clean, official, cohort, sequence, risks, intervals, predictors, specs, step14, registry)
        checks["index_reconciliation_qc"] = pd.DataFrame([dict(check=name, status="PASS", n_failures=0) for name in (
            "physical_v2_input_artifact_and_definition_hashes", "full_canonical_reconstruction", "unique_complete_step11_universe",
            "two_disjoint_index_routes", "official_clinical_baseline_preserved", "selected_date_event_reconstruction")])
        if args.dry_run:
            print("DRY RUN — physical inputs, index reconciliation, legacy bridge and temporal QC PASS; no fits or files written")
            return checks
        tasks = build_tasks(clean, cohort, sequence, risks, predictors, intervals, specs, step14)
        results, attempts, fdr = fit_models(tasks, cohort, step14, args.minimum_events)
        # S3 reuses the upstream sensitivity index policy, including fallback reselection.
        excluded, excluded_sequence = v2.build_observed_patterns_v2(clean, official, True)
        excluded_cohort = core.index_cohort(excluded, hashes)
        excluded_risks = {k: core.first_event_riskset(excluded_cohort, excluded_sequence, k) for k in (3, 5)}
        ep, es, _ = core.build_index_predictors(clean, official, excluded_cohort, step14, registry)
        ei, _ = core.build_lagged_intervals(clean, excluded_cohort, excluded_sequence, excluded_risks[3], es, step14, registry)
        et = build_tasks(clean, excluded_cohort, excluded_sequence, excluded_risks, ep, ei, es, step14,
                         "exclude_discordant_dates", include_secondary=False)
        er, ea, _ = fit_models(et, excluded_cohort, step14, args.minimum_events, official_comparison=False)
        variants, all_attempts = [results, er], [attempts, ea]
        # S6 changes only the all-episode outcome policy on the official subset.
        lr = _sensitivity_legacy_risks(risks, legacy)
        lt = build_tasks(clean, cohort, sequence, lr, predictors, intervals, specs, step14,
                         "v1_all_episodes_official", include_secondary=False)
        lt = [task for task in lt if task.kind == "cox"]
        lresult, la, _ = fit_models(lt, cohort, step14, args.minimum_events, official_comparison=False, reference_results=results)
        variants.append(lresult)
        all_attempts.append(la)
        for days in args.lab_max_age_days:
            wp, ws, _ = core.build_index_predictors(clean, official, cohort, step14, registry, days)
            wt = build_tasks(clean, cohort, sequence, risks, wp, intervals, ws, step14,
                             f"lab_age_le_{days}d", include_secondary=False)
            # These windows concern index labs; lagged labs keep strict FROM timing.
            wt = [t for t in wt if t.spec.family == "laboratory" and t.analysis_type == "baseline"]
            wr, wa, _ = fit_models(wt, cohort, step14, args.minimum_events, official_comparison=False)
            variants.append(wr)
            all_attempts.append(wa)
        results = pd.concat(variants, ignore_index=True)
        results, fdr_within = step14.apply_fdr_by_family(results)
        results, fdr_wide = step14.apply_outcome_wide_fdr(results)
        summaries = pd.concat([core.event_summary(risks), core.event_summary(
            {k: d.loc[d.index_source.eq(v2.OFFICIAL)] for k, d in risks.items()}, "official_only"),
            core.event_summary(excluded_risks, "S3:exclude_discordant_dates"),
            core.event_summary(lr, "S6:v1_all_episodes_official")], ignore_index=True)
        tables.update(cohort_flow=core.cohort_flow(cohort), first_event_summary=summaries,
            index_source_comparison=core.event_summary(risks), predictor_availability=availability,
            domain_index_support=domain_index_support(clean, cohort, step14),
            model_feasibility=feasibility_table(results, cohort, intervals),
            delta3_models=results.loc[results.outcome.eq("first_delta3") & results.analysis_type.eq("baseline")],
            delta5_sensitivity_models=results.loc[results.outcome.eq("first_delta5")],
            lagged_models=results.loc[results.analysis_type.eq("timevarying")],
            trajectory_models=results.loc[results.outcome.eq("essdai_slope")],
            official_only_comparison=results.loc[results.population.isin(["inclusive_primary", "official_only"])],
            index_state_domain_models=results.loc[~results.outcome.isin(["first_delta3", "first_delta5", "essdai_slope", "subsequent_essdai"])],
            pattern_characterization=v2.pattern_summary_v2(audit, "index_source"),
            observation_sensitivities=_observation_strata(risks))
        analytic.update(index_cohort=cohort, first_delta3_riskset=risks[3], first_delta5_riskset=risks[5],
            lagged_intervals=intervals, index_predictors=predictors)
        checks.update(attrition_qc=_attrition(results, risks, intervals), model_attempts_qc=pd.concat(all_attempts, ignore_index=True),
            fdr_qc=pd.concat([fdr_within.assign(scope="within_new_family"), fdr_wide.assign(scope="outcome_wide_new_family")], ignore_index=True),
            domain_composition_qc=composition)
        # Re-read physical hashes after all fits, prior to publishing anything.
        if any(core.sha256(path) != hashes[role] for role, path in paths.items()):
            raise ValueError("input bytes changed during analysis; bundle blocked")
        manifest.update(status="CONDITIONAL", qc_status="PASS", input_hashes=hashes,
            producer_commit=producer["code_commit"], producer_review=producer.get("closure_status"),
            n_step11=len(cohort), n_eligible=int(cohort.eligible.sum()),
            n_official=int((cohort.eligible & cohort.index_source.eq(v2.OFFICIAL)).sum()),
            n_fallback=int((cohort.eligible & cohort.index_source.eq(v2.FALLBACK)).sum()),
            lab_primary_age_days=0, lab_sensitivity_age_days=args.lab_max_age_days,
            lab_window_authorization="explicit CLI prespecification required; no inferred carry-forward",
            components={"implementation": "IMPLEMENTED", "cohort_and_outcomes": "IMPLEMENTED",
                "models": "IMPLEMENTED; per-model NOT_ESTIMABLE rows retained", "treatment": "NOT_IMPLEMENTED",
                "clinical_review": "CONDITIONAL", "privacy_review": "CONDITIONAL"})
        clean_code = not before["working_tree_status"]
        manifest["canonical_execution"] = "EXECUTED_CANONICAL" if args.input_origin == "canonical" and clean_code else (
            "CONDITIONAL_DIRTY_CODE" if args.input_origin == "canonical" else "SYNTHETIC_VALIDATION_ONLY")
    except (ValueError, KeyError, FileNotFoundError, AssertionError) as exc:
        # All errors here are bounded contract messages, never DataFrame diffs.
        manifest["blocking_reason"] = str(exc) if isinstance(exc, ValueError) else "Input schema/contract rejected; no new clinical results published"
        if args.dry_run:
            raise SystemExit("BLOCKED: "+manifest["blocking_reason"]) from None
        tables, analytic, checks = {}, {}, {}
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step14-inclusive-", dir=root) as temp:
        stage = Path(temp)
        sd = directories(stage)
        for directory in sd.values():
            directory.mkdir(parents=True)
        for name, frame in analytic.items():
            frame.to_parquet(sd["analytic"]/(PREFIX+name+".parquet"), index=False)
        for name, frame in tables.items():
            _write_csv(frame, sd["tables"], name)
        for name, frame in checks.items():
            _write_csv(frame, sd["qc"], name)
        if manifest["qc_status"] == "PASS":
            markers = make_figures(cohort, sequence, patterns, results, summaries, sd["figures"], args.input_origin)
            if markers.status.ne("PASS").any():
                raise ValueError("marker QC failure; no bundle installed")
            _write_csv(markers, sd["qc"], "marker_qc")
        # Historical bytes are checked before any installation.
        for rel, digest in before["historical_products_before"].items():
            if not (root/rel).is_file() or core.sha256(root/rel) != digest:
                raise ValueError("historical output changed during inclusive run")
        (sd["qc"]/(PREFIX+"prechange_snapshot.json")).write_text(json.dumps(before, indent=2))
        (sd["qc"]/(PREFIX+"closure_summary.md")).write_text(_closure(manifest))
        manifest["artifacts"] = [{"path": str(p.relative_to(stage)), "sha256": core.sha256(p)}
            for p in sorted(stage.rglob("*")) if p.is_file()]
        (sd["qc"]/(PREFIX+"manifest.json")).write_text(json.dumps(step14._json_safe(manifest), indent=2, allow_nan=False))
        install_bundle(stage, root, existing)
    if manifest["status"] == "BLOCKED":
        raise SystemExit("BLOCKED: "+manifest["blocking_reason"])
    print("Inclusive v2 complete: CONDITIONAL; clinical/institutional review pending; "+manifest["canonical_execution"])
    return manifest

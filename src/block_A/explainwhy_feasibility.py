"""Read-only scientific feasibility audit; never fit or authorize new models.

Called explicitly by Step 14. Missing canonical inputs produce unknown counts,
not a cohort with zero patients. This module does not change primary tasks/FDR.
"""
from pathlib import Path
from dataclasses import replace
import importlib.metadata
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.block_A.observed_patterns import (
    build_observed_patterns, pattern_summary, verified_baseline_pop, write_manifest,
)

COUNT_COLUMNS = ("n_patients_baseline", "n_patients_repeated", "n_valid_episodes",
    "n_distinct_dates", "n_lagged_eligible_intervals", "n_delta3_events_with_predictor",
    "n_delta5_events_with_predictor", "n_missing", "n_unknown_timing",
    "n_incompatible_units", "n_unknown_units", "n_schema_values", "n_paired_essdai_episodes")
PENDING = "PENDING / NOT RUN — canonical inputs unavailable"


def audit_registry(step14):
    registry = [s for s in step14.build_predictor_registry() if s.primary_role != "adjustment"]
    registry += [step14.PredictorSpec("essdai_total", "Recorded ESSDAI", "essdai", "continuous",
                 "essdai__total", "essdai__total")]
    registry += [step14.PredictorSpec("esspri_"+name, "ESSPRI "+name, "pro", "continuous", col, col)
                 for name, col in step14.step13.ESSPRI.items()]
    registry += [step14.PredictorSpec(name, name, "pro", "continuous", "pro__"+name, "pro__"+name)
                 for name in ("sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global")]
    registry += [step14.PredictorSpec("systemic_treatment_exposure", "Systemic treatment", "treatment",
                 "binary", "systemic_treatment_exposure", "systemic_treatment_exposure")]
    return registry


def source_evidence(variable_registry, column):
    if variable_registry is None or "public_variable" not in variable_registry or not column:
        return "", ""
    evidence = variable_registry.loc[variable_registry.public_variable.eq(column)]
    def unique(name):
        values = evidence[name].dropna().astype(str).unique() if name in evidence else []
        return values[0] if len(values) == 1 else ""
    return unique("producer_script"), unique("temporal_scope")


def attach_audit_context(integrated, baseline, context, step14):
    """Audit-only metadata enrichment; leave primary Step 14 joins unchanged."""
    integrated, baseline = step14.attach_episode_context(integrated, baseline, context)
    context = context.copy()
    for key in step14.KEYS:
        context[key] = context[key].astype("string")
    extra = [c for c in context if c.endswith(("__selection_status", "__conflict", "_scoring_version", "_scoring_valid"))]
    frames = []
    for frame in (integrated, baseline):
        shared = [c for c in extra if c in frame]
        joined = frame.merge(context[step14.KEYS+extra], on=step14.KEYS, validate="one_to_one", how="left", suffixes=("", "_context"))
        for col in shared:
            if not (joined[col].eq(joined[col+"_context"]) | (joined[col].isna() & joined[col+"_context"].isna())).all():
                raise ValueError("conflicting audit context metadata")
        frames.append(joined.drop(columns=[c+"_context" for c in shared]))
    return tuple(frames)


def audit_values(frame, spec, step14, variable_registry, baseline_mode=False):
    """Stricter feasibility only: unit/date uncertainty never certifies support."""
    column = spec.baseline_column if baseline_mode else spec.time_varying_column
    empty = pd.Series(False, index=frame.index)
    if not column or column not in frame:
        return pd.Series(np.nan, index=frame.index), empty, empty, empty, "", ""
    producer, scope = source_evidence(variable_registry, column)
    values = step14.predictor_values(frame, spec, baseline_mode)
    # The treatment audit owns registration, timing, unknown/exposure semantics.
    if spec.family == "treatment":
        _, _, exposures = step14.treatment_adjustment_decision(
            {"baseline" if baseline_mode else "integrated": frame}, variable_registry)
        values = exposures.get("baseline" if baseline_mode else "integrated", pd.Series(np.nan, index=frame.index))
    present = pd.to_numeric(frame[column], errors="coerce").notna()
    unknown_time, incompatible, unknown_unit = empty.copy(), empty.copy(), empty.copy()
    if spec.family == "laboratory":
        stem = column.removesuffix("__value")
        dates = pd.to_datetime(frame.get(stem+"__measurement_date", pd.Series(pd.NaT, index=frame.index)), errors="coerce")
        unknown_time = present & dates.isna()
        dates_valid = dates.notna() & dates.le(frame.clinical_anchor_date)
        unit = frame.get(stem+"__unit", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip().replace("", pd.NA)
        unknown_unit = present & unit.isna()
        # Multiple recorded units require an approved upstream harmonization.
        incompatible = present & unit.notna() & (unit.loc[present].nunique() > 1)
        values = values.where(dates_valid & ~unknown_unit & ~incompatible)
    else:
        # No undocumented use of nominal visit dates or retrospective histories.
        known = bool(producer and scope in {"episode", "clinical_baseline", "baseline", "as_of_episode", "through_episode"})
        unknown_time = present & ~pd.Series(known, index=frame.index)
        if scope in {"baseline", "clinical_baseline"} and not baseline_mode:
            values = values.where(frame.is_clinical_baseline.eq(True))
        if not known:
            values = values.where(empty)
        if spec.family == "pro" and not spec.name.startswith("esspri_"):
            instrument = spec.name.split("_")[0]
            version_col = "pro__"+instrument+"_scoring_version"
            valid_col = "pro__"+instrument+"_scoring_valid"
            version = frame.get(version_col, pd.Series(pd.NA, index=frame.index))
            valid = frame.get(valid_col, pd.Series(False, index=frame.index)).eq(True)
            values = values.where(version.notna() & valid & (version.dropna().nunique() == 1))
    # A unit and date do not establish the producer or temporal scope.
    if not producer or scope not in {"episode", "clinical_baseline", "baseline", "as_of_episode", "through_episode"}:
        values = values.where(empty)
    if scope in {"baseline", "clinical_baseline"} and not baseline_mode:
        values = values.where(frame.is_clinical_baseline.eq(True))
    return values, unknown_time, incompatible, unknown_unit, producer, scope


def modality_feasibility(df, baseline, endpoints, variable_registry, step14):
    """Counts of patients, patient-dates and original adjacent intervals."""
    registered = audit_registry(step14)
    resolved = step14.resolve_predictors(registered, baseline, df)
    # resolve_predictors already preserves exact canonical names for extra PROs.
    pop = verified_baseline_pop(baseline)
    base_state = baseline.set_index("patient_id")[step14.ESSDAI].map(step14.step13.classify_essdai_activity)
    strata = [("overall", "Overall", set(baseline.patient_id))]
    strata += [("baseline_state", s, set(base_state.index[base_state.eq(s)])) for s in ("Low", "Moderate", "High", "Missing")]
    if pop is not None:
        strata += [("baseline_pop", s, set(pop.index[pop.eq(s)])) for s in ("Pop1", "Pop2", "Pop3", "Unclassifiable")]
    rows = []
    ordered = df.sort_values(["patient_id", "clinical_anchor_date", "clinical_episode_id"], kind="stable")
    for original, spec in zip(registered, resolved):
        if spec.family == "treatment":
            decision, _, _ = step14.treatment_adjustment_decision({"integrated": df, "baseline": baseline}, variable_registry)
            columns = decision.set_index("source_checked").selected_column
            spec = replace(spec, baseline_column=columns.get("baseline") or None,
                           time_varying_column=columns.get("integrated") or None)
        values, unknown_time, incompatible, unknown_unit, producer, scope = audit_values(df, spec, step14, variable_registry)
        base_values, _, _, _, baseline_producer, baseline_scope = audit_values(baseline, spec, step14, variable_registry, True)
        valid = values.notna()
        col = spec.time_varying_column
        # Shift the full canonical spine first; never bridge a missing visit.
        adjacency = ordered[["patient_id", "clinical_anchor_date", step14.ESSDAI]].copy()
        adjacency["available"] = valid.reindex(ordered.index)
        nxt_date = adjacency.groupby("patient_id").clinical_anchor_date.shift(-1)
        nxt_score = adjacency.groupby("patient_id")[step14.ESSDAI].shift(-1)
        intervals = adjacency.available & adjacency[step14.ESSDAI].notna() & nxt_date.gt(adjacency.clinical_anchor_date) & nxt_score.notna()
        unit_col = col.removesuffix("__value")+"__unit" if col else ""
        units = sorted(df[unit_col].dropna().astype(str).unique()) if unit_col in df else []
        for stratifier, stratum, ids in strata:
            use = df.patient_id.isin(ids)
            obs = df.loc[use & valid]
            dates_per_patient = obs.groupby("patient_id").clinical_anchor_date.nunique()
            base_ids = set(baseline.loc[baseline.patient_id.isin(ids) & base_values.notna(), "patient_id"])
            row = {"modality": spec.family, "variable": spec.name,
                "source_column": col or spec.baseline_column or original.time_varying_column or original.baseline_column or "",
                "baseline_source_column": spec.baseline_column or "", "schema_present": bool(col or spec.baseline_column),
                "producer": producer or baseline_producer, "temporal_scope": scope or baseline_scope, "units": json.dumps(units),
                "stratifier": stratifier, "stratum": stratum,
                "n_patients_baseline": len(base_ids), "n_patients_repeated": int(dates_per_patient.ge(2).sum()),
                "n_valid_episodes": len(obs), "n_distinct_dates": len(obs[["patient_id", "clinical_anchor_date"]].drop_duplicates()),
                "n_lagged_eligible_intervals": int((intervals & adjacency.patient_id.isin(ids)).sum()),
                "n_missing": int((use & ~valid).sum()), "n_unknown_timing": int((use & unknown_time).sum()),
                "n_incompatible_units": int((use & incompatible).sum()), "n_unknown_units": int((use & unknown_unit).sum()),
                "n_schema_values": int(df.loc[use, col].notna().sum()) if col else 0,
                "n_paired_essdai_episodes": int((use & valid & df[step14.ESSDAI].notna()).sum())}
            for k in (3, 5):
                row[f"n_delta{k}_events_with_predictor"] = int((endpoints.patient_id.isin(base_ids) & endpoints[f"ever_worsened_ge{k}"]).sum())
            row["support_status"] = "PARTIAL" if len(obs) or base_ids else "BLOCKED"
            row["support_reason"] = ("descriptive observations only; new models require G1/G2 review" if len(obs)
                else "baseline-only support; repeated/lagged observations unavailable" if base_ids
                else "absent or no evaluable observations with verified timing/units/producer/instrument")
            rows.append(row)
    return pd.DataFrame(rows)


def input_paths(args):
    return {"step10_integrated": Path(args.integrated), "step11_baseline": Path(args.baseline),
        "step10_context": getattr(args, "context", None) or Path(args.integrated).with_name("10_integrated_longitudinal_context.parquet"),
        "step10_registry": getattr(args, "variable_registry", None) or Path(args.integrated).with_name("10_variable_registry.csv"),
        "step13_episode": Path(args.progression),
        "step13_delta_endpoints": Path(args.output_root)/"data/analytic/blockA/13_disease_activity_progression/13_essdai_patient_delta_endpoints.parquet"}


def run_feasibility_audit(args, step14):
    """Create-only or backed up audit bundle, without altering primary outputs."""
    dirs = step14.output_directories(args.output_root)
    existing = [p for d in dirs.values() if d.exists() for p in d.glob("14_explainwhy_*") if p.is_file()]
    if existing and not args.overwrite:
        raise FileExistsError("Explain-why audit outputs exist; use --overwrite for a backed-up rerun")
    backup = None
    if existing:
        backup = Path(args.output_root)/"outputs/backups"/step14.STEM/("explainwhy_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        for path in existing:
            target = backup/path.relative_to(args.output_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for directory in (dirs["tables"], dirs["qc"]):
        directory.mkdir(parents=True, exist_ok=True)
    paths = input_paths(args)
    inventory = [{"input": name, "filename": p.name, "available": p.is_file(),
                  "sha256": step14.sha256_file(p) if p.is_file() else None} for name, p in paths.items()]
    # Snapshot hashes of existing products; no patient rows or clinical estimates.
    step13_dirs = [Path(args.output_root)/prefix/step14.step13.STEM for prefix in
                   ("outputs/tables/blockA", "outputs/figures/blockA", "outputs/qc/blockA", "data/analytic/blockA")]
    outputs = [{"path": str(p.relative_to(args.output_root)), "sha256": step14.sha256_file(p)}
               for d in [*dirs.values(), *step13_dirs] if d.exists() for p in sorted(d.iterdir()) if p.is_file()]
    code = step14.runtime_code_fingerprint()
    snapshot = {"inputs": inventory, "preexisting_outputs": outputs, "code": code,
                "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "statsmodels", "scipy", "pyarrow")},
                "input_origin": args.input_origin, "backup_created": backup is not None}
    snapshot_path = dirs["qc"]/"14_explainwhy_prechange_snapshot.md"
    snapshot_path.write_text("# Explain-why pre-run snapshot\n\n```json\n"+json.dumps(snapshot, indent=2)+"\n```\n", encoding="utf-8")
    published = [snapshot_path]
    missing = [name for name in ("step10_integrated", "step11_baseline") if not paths[name].is_file()]
    gate, reason = "BLOCKED", PENDING
    feasibility = pd.DataFrame([{**{c: pd.NA for c in COUNT_COLUMNS},
        "modality": s.family, "variable": s.name, "source_column": s.time_varying_column or s.baseline_column,
        "producer": "", "temporal_scope": "", "units": "unknown", "schema_present": pd.NA,
        "stratifier": "overall", "stratum": "Overall", "support_status": "BLOCKED", "support_reason": PENDING}
        for s in audit_registry(step14)])
    outcomes = pd.DataFrame([{"outcome": f"first_observed_baseline_delta{k}", "n_patients": pd.NA,
                             "n_events": pd.NA, "status": "NOT_RUN", "reason": PENDING} for k in (3, 5)])
    treatment = pd.DataFrame([{"decision": "not_implemented", "reason": PENDING}])
    if not missing:
        try:
            integrated = step14.read_table(paths["step10_integrated"])
            baseline = step14.read_table(paths["step11_baseline"])
            if paths["step10_context"].is_file():
                integrated, baseline = attach_audit_context(integrated, baseline, step14.read_table(paths["step10_context"]), step14)
            elif getattr(args, "context", None):
                raise ValueError("explicit context unavailable")
            progression = step14.read_table(paths["step13_episode"]) if paths["step13_episode"].is_file() else None
            df, baseline = step14.validate_inputs(integrated, baseline, progression)
            if len(step14.step13.resolve_domains(df)) != len(step14.DOMAINS):
                raise ValueError("incomplete 12-domain canonical schema")
            variable_registry = pd.read_csv(paths["step10_registry"]) if paths["step10_registry"].is_file() else None
            if getattr(args, "variable_registry", None) and variable_registry is None:
                raise ValueError("explicit variable registry unavailable")
            endpoints = step14.step13.build_patient_absolute_change_endpoints(df)
            if paths["step13_delta_endpoints"].is_file():
                stored = pd.read_parquet(paths["step13_delta_endpoints"])
                pd.testing.assert_frame_equal(stored.reset_index(drop=True), endpoints.reset_index(drop=True))
            feasibility = modality_feasibility(df, baseline, endpoints, variable_registry, step14)
            registry = step14.resolve_predictors(step14.build_predictor_registry(), baseline, df)
            composition = step14.composition_episode_audit(df, registry, variable_registry)
            composition_summary = composition.groupby("composition_category", dropna=False).size().rename("n_episodes").reset_index()
            path = dirs["qc"]/"14_explainwhy_domain_composition_support.csv"
            composition_summary.to_csv(path, index=False)
            published.append(path)
            patterns = build_observed_patterns(df, endpoints)
            pattern_support = pattern_summary(patterns)
            path = dirs["tables"]/"14_explainwhy_pattern_support.csv"
            pattern_support.to_csv(path, index=False)
            published.append(path)
            outcomes = pd.DataFrame([{"outcome": f"first_observed_baseline_delta{k}", "n_patients": len(endpoints),
                "n_events": int(endpoints[f"ever_worsened_ge{k}"].sum()), "status": "descriptive_only",
                "reason": "Step 13 definition; G2 complete-case/support/PH/FDR review not performed"} for k in (3, 5)])
            treatment, _, _ = step14.treatment_adjustment_decision({"integrated": df, "baseline": baseline}, variable_registry)
            gate, reason = "PARTIAL", "canonical contracts validated; descriptive audit only, no new inference authorized by support"
        except (ValueError, KeyError, AssertionError) as exc:
            # Never echo patient rows/values from validators or pandas diffs.
            reason = "canonical structural/temporal/provenance contract rejected ("+type(exc).__name__+"); inspect inputs in authorized environment"
            feasibility["support_status"] = "BLOCKED"
            feasibility["support_reason"] = reason
            outcomes["status"], outcomes["reason"] = "NOT_RUN", reason
    decisions = pd.DataFrame([
        {"gate": "G0", "status": gate, "reason": reason},
        {"gate": "G1", "status": "BLOCKED" if gate == "BLOCKED" else "PARTIAL", "reason": "technical synthetic validation separate from real class/clinical review"},
        {"gate": "G2", "status": "BLOCKED", "reason": "NOT_IMPLEMENTED: no new delta3 models before real feasibility, complete-case support and review"},
        {"gate": "G3", "status": "BLOCKED" if treatment.decision.eq("not_implemented").all() else "PARTIAL", "reason": "upstream dated intervals, indication, start/stop, external use and unknown states require review"},
        {"gate": "G4", "status": "BLOCKED" if gate == "BLOCKED" else "PARTIAL", "reason": "ocular/salivary/PRO measurement pairing, units and instrument versions must support any future model"},
        {"gate": "omics", "status": "NOT_IMPLEMENTED", "reason": "inventory/consent/sample timing outside this ticket"}])
    for directory, name, data in (("qc", "input_availability", pd.DataFrame(inventory)),
            ("tables", "modality_feasibility", feasibility), ("tables", "outcome_support", outcomes),
            ("qc", "feasibility_decisions", decisions), ("qc", "treatment_readiness", treatment)):
        path = dirs[directory]/f"14_explainwhy_{name}.csv"
        data.to_csv(path, index=False)
        published.append(path)
    # Remove stale supported output on a subsequent failed audit (backup retained).
    if gate == "BLOCKED":
        (dirs["tables"]/"14_explainwhy_pattern_support.csv").unlink(missing_ok=True)
        (dirs["qc"]/"14_explainwhy_domain_composition_support.csv").unlink(missing_ok=True)
        published = [p for p in published if p.exists()]
    write_manifest(published, dirs["qc"]/"14_explainwhy_manifest.json", {
        "input_origin": args.input_origin, "G0": gate, "reason": reason,
        "missing_inputs": missing, "clinical_status": "PENDING_CLINICAL_REVIEW",
        "scientific_closure": "CONDITIONAL", "new_models_fitted": 0})
    return decisions

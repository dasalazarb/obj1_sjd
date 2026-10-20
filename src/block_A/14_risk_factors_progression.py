#!/usr/bin/env python3
"""Earlier clinical information associated with subsequent SjD progression.

Analysis consumer of Steps 10/11/13; no raw-data phenotyping or imputation.
Event times are first observed positive assessments, not biological onset times.
Time-updated event inference uses piecewise-exponential Poisson GEE: lifelines'
CoxTimeVaryingFitter does not implement reliable patient-clustered robust variance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import subprocess
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
from scipy.stats import norm
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.genmod.families import Gaussian, Poisson
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import common  # noqa: E402
from config import CHANGE_THRESHOLDS, ESSDAI_DOMAIN_WEIGHTS  # noqa: E402

# Import the canonical definitions and fit validators, never a private copy.
_spec = importlib.util.spec_from_file_location(
    "risk_progression_step13", Path(__file__).with_name("13_disease_activity_progression.py"))
step13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step13)
ESSDAI, TIME, DOMAINS = step13.ESSDAI, step13.TIME, step13.DOMAINS
STEM = "14_risk_factors_progression"
LOG = logging.getLogger(STEM)
KEYS = ["patient_id", "clinical_episode_id"]
MIN_PATIENTS_MODEL = 20
MIN_REPEATED_PATIENTS = 10
MIN_EVENTS_COX = 10
MIN_EXPOSED = MIN_UNEXPOSED = 5
MIN_EXPOSED_EVENTS = 3
MIN_DOMAIN_EVENTS = 10
MIN_CROSS_EXPOSED_EVENTS = 5
MIN_INTERVALS = 20
MIN_PREDICTOR_VARIATION = 2
EVENTS_PER_PARAMETER = 5
SINGULAR_TOL = step13.SINGULAR_TOL
RANDOM_SEED = 20261007
ADJUSTMENT_RULE_VERSION = "hierarchical-v1"
COMPOSITION_RULE_VERSION = "concordant-complete-panel-from-to-v2"
COMPOSITION_AUDIT_RULE_VERSION = COMPOSITION_RULE_VERSION
DURATION_RULE_VERSION = "verified-step11-baseline-duration-v1"
COMPOSITION_TOLERANCE = 1e-8
LEAVE_SOURCE_SENSITIVITY = "leave_source_domain_outcome"
HYPOTHESIS_RULE_VERSION = "canonical-construct-v1"
FDR_RULE_VERSION = "outcome-wide-v1"
ADJUSTMENT_LEVELS = ("full", "reduced_A", "reduced_B", "minimal")
ADJUSTMENT_SELECTION_RULE = "first supported estimable level: Full > Reduced A > Reduced B > Minimal, before fitting"
EVENT_NOTE = step13.EVENT_NOTE
TV_NOTE = ("Piecewise-exponential Poisson GEE with log(interval_years) offset, "
           "patient-clustered robust variance and prespecified time bands; "
           "selected because clustered start-stop Cox variance is unavailable.")
MAIN_DOMAINS = {"glandular", "articular", "pulmonary", "renal", "pns", "cns",
                "hematologic", "lymphadenopathy"}
DOMAIN_LABELS = {**step13.DOMAIN_LABELS,
    "pns": "Peripheral nervous system", "cns": "Central nervous system"}
TABLES = ("predictor_registry", "predictor_feasibility", "baseline_essdai_trajectory_models",
    "baseline_ge5_cox_models", "baseline_high_activity_models", "baseline_new_domain_models",
    "timevarying_essdai_models", "timevarying_ge5_models", "timevarying_new_domain_models",
    "cross_domain_support", "cross_domain_models", "sensitivity_models",
    "model_interpretation_summary")
QC_TABLES = ("structural_qc", "predictor_availability", "baseline_model_attrition",
    "interval_model_attrition", "temporal_leakage_qc", "domain_riskset_qc",
    "model_qc", "multiple_testing_qc", "predictor_redundancy_qc",
    "outcome_wide_fdr_qc", "domain_composition_qc", "adjustment_selection_qc",
    "domain_composition_audit_summary", "domain_composition_discrepancy_distribution",
    "cox_failure_diagnostics", "run_comparison_summary")


@dataclass(frozen=True)
class PredictorSpec:
    name: str
    label: str
    family: str
    variable_type: str
    baseline_column: str | None
    time_varying_column: str | None
    evaluability_column: str | None = None
    direction: str | None = None
    transform: str = "none"
    primary_role: str = "primary"
    notes: str = ""
    hypothesis_group_id: str = ""
    primary_canonical: bool = True
    alternative_to: str = ""
    analysis_role: str = "primary"
    redundancy_evidence: str = ""
    fdr_eligibility: bool = True
    source_definition: str = "canonical upstream predictor"


def read_table(path: Path) -> pd.DataFrame:
    return step13.read_table(Path(path))


def attach_episode_context(integrated, baseline, context):
    """Join canonical Step 10 evaluability/lab/provenance metadata, never predictors.

    Step 10 deliberately partitions its analytic and context products. Step 11
    remains the sole baseline selector; enrich its existing episode keys only.
    """
    frames = [integrated.copy(), baseline.copy(), context.copy()]
    for frame in frames:
        for key in KEYS:
            if key not in frame or frame[key].isna().any():
                raise ValueError("Step 10 context join requires nonmissing episode keys")
            frame[key] = frame[key].astype("string")
        if frame.duplicated(KEYS).any():
            raise ValueError("Step 10 context join has duplicate episode keys")
    integrated, baseline, context = frames
    if set(map(tuple, integrated[KEYS].to_numpy())) != set(map(tuple, context[KEYS].to_numpy())):
        raise ValueError("Step 10 context episode keys disagree with integrated data")
    provenance_fields = {"source_protocol", "_essdai_total_version", "_essdai_version", "_essdai_domain_version",
                         "essdai_total_source", "_essdai_resolution_method"}
    allowed = [c for c in context if c in provenance_fields or c.endswith(("_evaluable", "_valid", "__measurement_date",
                                                "__days_from_anchor", "__unit", "__n_measurements"))]
    allowed += [c for c in ("_essdai_version", "_essdai_total_version", "essdai_total_source")
                if c in context and c not in allowed]
    out = []
    for frame in (integrated, baseline):
        shared = [c for c in allowed + ["clinical_anchor_date"] if c in frame and c in context]
        checked = frame[KEYS+shared].merge(context[KEYS+shared], on=KEYS, how="left", validate="one_to_one", suffixes=("_source", "_context"))
        for col in shared:
            a, b = checked[col+"_source"], checked[col+"_context"]
            if col.endswith("date"):
                a, b = pd.to_datetime(a, errors="coerce"), pd.to_datetime(b, errors="coerce")
            if not (a.eq(b) | (a.isna() & b.isna())).all():
                raise ValueError(f"Step 10 context conflicts with source: {col}")
        additional = [c for c in allowed if c not in frame]
        enriched = frame.merge(context[KEYS+additional], on=KEYS, how="left", validate="one_to_one")
        enriched.attrs = frame.attrs.copy()
        out.append(enriched)
    return tuple(out)


def build_predictor_registry() -> list[PredictorSpec]:
    registry = [PredictorSpec("age", "Age at clinical baseline", "demographic", "continuous",
                    "demo__age_at_baseline", None, primary_role="adjustment"),
                PredictorSpec("sex", "Sex", "demographic", "categorical", "demo__sex",
                    None, primary_role="adjustment"),
                PredictorSpec("duration", "Disease duration (years)", "disease_history",
                    "continuous", "disease_duration", None, primary_role="adjustment",
                    source_definition="Step 11: (official clinical baseline date - diagnosis date).days / 365.25; missing diagnosis remains unknown",
                    notes="No follow-up or symptom-onset duration; unverified pharma aliases and clipped legacy duration are not resolved." )]
    for name, label, typ in (
        ("biopsy_focus_score", "Focus score", "continuous"),
        ("salivary_flow_unstimulated", "Unstimulated salivary flow", "continuous"),
        ("ocular_schirmer_min", "Schirmer test", "continuous"),
        ("ocular_staining_positive", "Ocular staining positivity", "binary"),
        ("sicca_any_symptom", "Any sicca symptom", "binary")):
        evaluability = {"biopsy_focus_score": "ext__biopsy_evaluable",
                        "ocular_schirmer_min": "ext__ocular_schirmer_evaluable"}.get(name)
        registry.append(PredictorSpec(name, label, "extended_clinical", typ,
            "ext__" + name, "ext__" + name, evaluability_column=evaluability,
            transform="zscore" if typ == "continuous" else "none"))
    for name, label, typ in (
        ("glandular_active", "Glandular phenotype activity", "binary"),
        ("n_glandular_manifestations_active", "Number of glandular manifestations", "continuous"),
        ("glandular_objective_eye_active", "Objective eye involvement", "binary"),
        ("glandular_objective_mouth_active", "Objective mouth involvement", "binary"),
        ("glandular_salivary_gland_swelling_active", "Salivary gland swelling", "binary")):
        registry.append(PredictorSpec(name, label, "glandular", typ, "ovl__" + name,
            "ovl__" + name, transform="zscore" if typ == "continuous" else "none"))
    for name, label in (
        ("anti_ro_ssa", "Anti-Ro/SSA positivity"), ("anti_la_ssb", "Anti-La/SSB positivity"),
        ("ana", "ANA positivity"), ("rf", "Rheumatoid factor positivity"),
        ("cryoglobulinemia", "Cryoglobulinemia"), ("low_c4", "Low C4"),
        ("leukopenia", "Leukopenia")):
        registry.append(PredictorSpec(name, label, "serology", "binary",
            "sero__baseline_" + name,
            "sero__" + name + "__ever_positive_through_episode" if name in
            {"anti_ro_ssa", "anti_la_ssb", "ana", "rf"} else None))
    for name, label in (("c3", "C3"), ("c4", "C4"), ("igg", "IgG"),
                        ("esr", "ESR"), ("crp", "CRP"), ("crp_high_sensitivity", "High-sensitivity CRP")):
        analyte = {"c3": "complement_c3", "c4": "complement_c4"}.get(name, name)
        registry.append(PredictorSpec("lab_" + name, label, "laboratory", "continuous",
            f"lab__{analyte}__value", f"lab__{analyte}__value", transform="zscore",
            primary_role="sensitivity", notes="Continuous laboratory sensitivity; no clinical cutoffs."))
    for domain in DOMAINS:
        registry.append(PredictorSpec("domain_" + domain,
            DOMAIN_LABELS[domain] + " ESSDAI domain", "essdai_domain", "ordinal",
            step13.DOMAIN_ALIASES[domain][0], step13.DOMAIN_ALIASES[domain][0],
            primary_role="primary" if domain in MAIN_DOMAINS else "secondary",
            notes="ESSDAI total includes the weighted ordinal contribution; lagged sensitivity adjusts for other domains."))
    out = []
    for spec in registry:
        alternative = spec.name == "glandular_salivary_gland_swelling_active"
        group = "domain_glandular" if alternative else spec.name
        role = "alternate_representation" if alternative else (
            "adjustment" if spec.primary_role == "adjustment" else
            "sensitivity" if spec.primary_role == "sensitivity" else "primary")
        out.append(replace(spec, hypothesis_group_id=group, primary_canonical=not alternative,
            alternative_to="domain_glandular" if alternative else "", analysis_role=role,
            fdr_eligibility=role == "primary",
            redundancy_evidence="Prespecified glandular construct; review snapshot Spearman ~0.999672, 148 complete pairs. Current support in redundancy QC; policy does not depend on rerun correlation."
                if alternative else ""))
    return out


def resolve_predictors(registry, baseline, integrated):
    """Resolve only explicit equivalent names; absent glandular fields stay absent."""
    def resolve(spec, frame, baseline_mode):
        column = spec.baseline_column if baseline_mode else spec.time_varying_column
        if column is None:
            return None
        aliases = [column]
        if spec.family == "serology" and baseline_mode:
            aliases += ["baseline_" + spec.name, "lab__baseline_" + spec.name]
        elif spec.family == "serology" and not baseline_mode:
            aliases += {"rf": ["sero__rheumatoid_factor__ever_positive_through_episode"],
                        "ana": ["sero__ana_status__ever_positive_through_episode"]}.get(spec.name, [])
        elif spec.family == "essdai_domain":
            aliases = list(step13.DOMAIN_ALIASES[spec.name.removeprefix("domain_")])
        elif spec.family == "laboratory":
            analyte = spec.name.removeprefix("lab_")
            aliases += [f"lab__{analyte}__value", f"lab__{analyte.upper()}__value",
                        f"lab__{analyte}__value".replace("igg", "IgG")]
            aliases += {"igg": ["lab__immunoglobulin_g__value"],
                        "esr": ["lab__sedimentation_rate__value"],
                        "crp": ["lab__c_reactive_protein__value"]}.get(analyte, [])
        elif baseline_mode and spec.name in {"age", "sex", "duration"}:
            aliases += {"age": ["age_at_baseline", "age_baseline"], "sex": ["sex"],
                        "duration": []}[spec.name]
        found = next((c for c in aliases if c in frame), None)
        if found and not baseline_mode and "patient_consensus" in found:
            raise ValueError("Retrospective consensus forbidden in time-varying registry")
        return found
    return [replace(s, baseline_column=resolve(s, baseline, True),
                    time_varying_column=resolve(s, integrated, False)) for s in registry]


def _evaluation_columns(frame, column, spec=None):
    candidates = [f"{column}_evaluable", f"{column}__evaluable", f"{column}_valid"]
    if spec and spec.evaluability_column:
        candidates.insert(0, spec.evaluability_column)
    if "ordinal_score" in column:
        candidates.insert(0, column.replace("ordinal_score", "evaluable"))
    if column.endswith("_active"):
        candidates.insert(0, column.removesuffix("_active") + "_evaluable")
    return [c for c in candidates if c in frame]


def predictor_values(frame, spec, baseline_mode):
    column = spec.baseline_column if baseline_mode else spec.time_varying_column
    if not column or column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    if spec.variable_type == "categorical":
        values = frame[column].astype("string").str.strip().replace("", pd.NA)
    else:
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        if spec.variable_type == "binary":
            # Canonical booleans or 0/1 only. Unknown is never negative.
            if (~values.dropna().isin([0, 1])).any():
                raise ValueError(f"Nonbinary canonical predictor: {column}")
    for flag in _evaluation_columns(frame, column, spec):
        values = values.where(frame[flag].eq(True).fillna(False))
    if spec.name == "duration":
        # Consume Step 11 duration only; never derive from later episode history.
        values = values.where(np.isfinite(values) & values.ge(0))
        age_col = next((c for c in ("demo__age_at_baseline", "age_at_baseline", "age_baseline") if c in frame), None)
        if age_col:
            age = pd.to_numeric(frame[age_col], errors="coerce")
            values = values.where(age.isna() | values.le(age))
        if "dx_date" in frame:
            diagnosis = pd.to_datetime(frame.dx_date, errors="coerce")
            anchor = pd.to_datetime(frame.clinical_baseline_date, errors="coerce")
            values = values.where(diagnosis.notna() & diagnosis.le(anchor))
    if spec.family == "laboratory":
        stem = column.removesuffix("__value")
        date_col = stem + "__measurement_date"
        days_col = stem + "__days_from_anchor"
        date = pd.to_datetime(frame[date_col], errors="coerce") if date_col in frame else None
        days = pd.to_numeric(frame[days_col], errors="coerce") if days_col in frame else None
        anchor = pd.to_datetime(frame.clinical_anchor_date, errors="coerce")
        known = date.notna() & date.le(anchor) if date is not None else (
            days.notna() & days.le(0) if days is not None else pd.Series(False, index=frame.index))
        if days is not None:
            known &= days.notna() & days.le(0)
        status_col = stem + "__selection_status"
        if status_col in frame:
            # The upstream selector takes the nearest exact value (median of
            # ties), preferring pre-anchor at equal absolute distance. Tied
            # nearest medians therefore share the selected date's temporal scope.
            known &= frame[status_col].isin(["selected_single", "selected_repeated_numeric_median"])
        conflict = stem + "__conflict"
        if conflict in frame:
            known &= frame[conflict].eq(False).fillna(False)
        values = values.where(known)
    return values


def domain_values(frame, domain):
    col = next((c for c in step13.DOMAIN_ALIASES[domain] if c in frame), None)
    spec = PredictorSpec(domain, domain, "essdai_domain", "ordinal", col, col)
    return predictor_values(frame, spec, False)


def validate_inputs(integrated, baseline, progression=None):
    """Fail on conflicting official baselines, chronology, values or episode keys."""
    df = step13.validate_inputs(integrated, baseline)
    baseline = baseline.copy()
    baseline.patient_id = baseline.patient_id.astype("string")
    baseline.clinical_episode_id = baseline.clinical_episode_id.astype("string")
    for col in ("clinical_anchor_date", "clinical_baseline_date"):
        if col not in baseline:
            raise ValueError(f"Step 11 baseline lacks {col}")
        baseline[col] = pd.to_datetime(baseline[col], errors="coerce")
    actual = df.loc[df.is_clinical_baseline.eq(True)].set_index("patient_id")
    official = baseline.set_index("patient_id").reindex(actual.index)
    for col in ("clinical_anchor_date", "clinical_baseline_date", ESSDAI):
        if col not in official or not actual[col].eq(official[col]).fillna(False).where(
                ~(actual[col].isna() & official[col].isna()), True).all():
            raise ValueError(f"Step 11 official baseline conflicts with Step 10: {col}")
    for spec in resolve_predictors(build_predictor_registry(), baseline, df):
        col = spec.baseline_column
        if col and col in actual and col in official:
            if not (actual[col].eq(official[col]) | (actual[col].isna() & official[col].isna())).all():
                raise ValueError(f"Step 11 official baseline predictor conflicts with Step 10: {col}")
    if df.clinical_anchor_date.isna().any():
        raise ValueError("Missing canonical clinical anchor date; cannot establish adjacency")
    for _, g in df.groupby("patient_id"):
        order = g.sort_values("clinical_visit_number", kind="stable")
        if not order.clinical_anchor_date.is_monotonic_increasing:
            raise ValueError("Clinical dates not monotonic in canonical visit order")
    dates = (df.clinical_anchor_date - df.clinical_baseline_date).dt.total_seconds() / (86400 * 365.25)
    if not np.allclose(dates, df[TIME], atol=1/365.25, equal_nan=False):
        raise ValueError("Canonical follow-up time conflicts with baseline dates")
    for domain in DOMAINS:
        for col in step13.DOMAIN_ALIASES[domain]:
            if col not in df:
                continue
            scores = pd.to_numeric(df[col], errors="coerce")
            if (~scores.dropna().isin([0, 1, 2, 3])).any():
                raise ValueError(f"Domain ordinal score outside 0-3: {col}")
            for flag in _evaluation_columns(df, col):
                if (df[flag].eq(False).fillna(False) & scores.notna()).any():
                    raise ValueError(f"Domain score/evaluability contradiction: {col}")
    if progression is not None:
        p = progression.copy()
        if "essdai_total" in p and ESSDAI not in p:
            p = p.rename(columns={"essdai_total": ESSDAI})
        for col in KEYS:
            if col not in p or p[col].isna().any():
                raise ValueError("Step 13 progression has missing episode keys")
            p[col] = p[col].astype("string")
        if p.duplicated(KEYS).any():
            raise ValueError("Step 13 progression has duplicate keys")
        left = df.assign(clinical_episode_id=df.clinical_episode_id.astype("string")).set_index(KEYS)
        right = p.set_index(KEYS)
        if set(left.index) != set(right.index):
            raise ValueError("Step 13 progression episode keys disagree with Step 10")
        right = right.reindex(left.index)
        for col in (ESSDAI, "clinical_anchor_date", TIME):
            if col in right:
                a, b = left[col], right[col]
                if col == "clinical_anchor_date":
                    b = pd.to_datetime(b, errors="coerce")
                if not (a.eq(b) | (a.isna() & b.isna())).all():
                    raise ValueError(f"Step 13 progression disagrees with Step 10: {col}")
        state_col = next((c for c in ("essdai_activity_state", "activity_state") if c in right), None)
        if state_col:
            if not step13.classify_essdai_series(left[ESSDAI]).astype("string").eq(
                    right[state_col].astype("string")).all():
                raise ValueError("Step 13 activity states disagree with canonical definitions")
    return (df.sort_values(["patient_id", "clinical_anchor_date", "clinical_visit_number", "clinical_episode_id"], kind="stable"),
            baseline.sort_values("patient_id", kind="stable").reset_index(drop=True))


def duration_source_report(baseline, registry):
    spec = next(s for s in registry if s.name == "duration")
    available = predictor_values(baseline, spec, True).notna()
    diagnosis = pd.to_datetime(baseline.get("dx_date", pd.Series(pd.NaT, index=baseline.index)), errors="coerce")
    anchor = pd.to_datetime(baseline.clinical_baseline_date, errors="coerce")
    proposed = spec.baseline_column is None and (diagnosis.notna() & diagnosis.le(anchor)).any()
    status = "verified_source_used" if available.any() else (
        "upstream_change_proposed_pending_approval" if proposed else "not_available")
    return {"duration_source_status": status,
        "duration_definition": "Step 11 official baseline minus confirmed SjD diagnosis, days / 365.25; never follow-up",
        "duration_anchor": "clinical_baseline_date", "duration_n_available": int(available.sum()),
        "duration_exclusion_reason": "" if available.any() else "no_verified_canonical_diagnosis_date",
        "duration_rule_version": DURATION_RULE_VERSION}


def build_predictor_availability(baseline, integrated, registry, intervals=None, prepared=None):
    rows = []
    for spec in registry:
        for mode, frame in ((True, baseline), (False, integrated)):
            values = predictor_values(frame, spec, mode)
            col = spec.baseline_column if mode else spec.time_varying_column
            rows.append({**asdict(spec), "analysis_type": "baseline" if mode else "timevarying",
                "resolved_column": col, "available": col is not None,
                "availability_scope": "official_baseline" if mode else "integrated_raw",
                "origin_product": "Step 11" if mode else "Step 10",
                "n_eligible_intervals": 0, "n_available_intervals": 0,
                "n_total": len(frame), "n_available": int(values.notna().sum()),
                "n_available_patients": frame.loc[values.notna(), "patient_id"].nunique(),
                "n_raw_nonmissing": int(frame[col].notna().sum()) if col else 0,
                "n_temporal_or_evaluability_excluded": int(frame[col].notna().sum() - values.notna().sum()) if col else 0,
                "support_reason": "column_unavailable" if col is None else
                "dated as-of upstream selection only; future/undated/conflicting values excluded"
                if spec.family == "laboratory" else
                "Step 11 baseline diagnosis duration; negative, longer-than-age or future diagnosis excluded, never clipped"
                if spec.name == "duration" else "canonical evaluability respected"})
        if intervals is not None:
            static = spec.primary_role == "adjustment"
            column = spec.name if static else "pred__" + spec.name
            values = intervals.get(column, pd.Series(np.nan, index=intervals.index))
            rows.append({**asdict(spec), "analysis_type": "timevarying", "resolved_column": column,
                "available": bool(values.notna().any()), "availability_scope": "lagged_interval_effective",
                "origin_product": "Step 11 static baseline via patient_id" if static else "Step 10 FROM episode",
                "n_total": len(intervals), "n_available": int(values.notna().sum()),
                "n_raw_nonmissing": int(values.notna().sum()), "n_temporal_or_evaluability_excluded": 0,
                "n_eligible_intervals": len(intervals), "n_available_intervals": int(values.notna().sum()),
                "n_available_patients": intervals.loc[values.notna(), "patient_id"].nunique(),
                "support_reason": "static official baseline propagated; never FROM/TO demographic update" if static else
                    "adjacent intervals before outcome-specific restriction; model complete cases reported separately"})
    if prepared is not None:
        static_specs = [s for s in registry if s.primary_role == "adjustment"]
        for task, data, meta in prepared:
            if task.analysis_type == "baseline":
                continue
            for spec in static_specs:
                values = data[spec.name]
                rows.append({**asdict(spec), "analysis_type": task.analysis_type,
                    "availability_scope": "selected_model_complete_cases", "origin_product": "Step 11 static baseline via patient_id",
                    "resolved_column": spec.name, "available": bool(values.notna().any()),
                    "predictor": task.spec.name, "outcome": task.outcome, "sensitivity": task.sensitivity,
                    "included_in_formula": spec.name in meta["adjustment_covariates"].split(","),
                    "n_total": len(data), "n_available": int(values.notna().sum()),
                    "n_raw_nonmissing": int(values.notna().sum()), "n_temporal_or_evaluability_excluded": 0,
                    "n_eligible_intervals": len(data), "n_available_intervals": int(values.notna().sum()),
                    "n_available_patients": data.loc[values.notna(), "patient_id"].nunique(),
                    "selected_adjustment_level": meta["selected_adjustment_level"],
                    "support_reason": "exact selected/unsupported pre-fit cohort; availability does not imply a fitted estimator"})
    duration = duration_source_report(baseline, registry)
    for row in rows:
        if row["name"] == "duration":
            row.update(duration)
            if duration["duration_source_status"] != "verified_source_used":
                row["support_reason"] = duration["duration_exclusion_reason"]
    return pd.DataFrame(rows)


def build_baseline_analysis_dataset(baseline, registry):
    out = baseline.copy()  # Never select/reconstruct baseline from longitudinal rows.
    out["baseline_essdai"] = pd.to_numeric(out[ESSDAI], errors="coerce")
    out["baseline_activity_state"] = step13.classify_essdai_series(out.baseline_essdai)
    out["eligible_ge5"] = out.baseline_activity_state.eq("Low")
    out["eligible_high"] = out.baseline_activity_state.eq("Moderate")
    for spec in registry:
        out["pred__" + spec.name] = predictor_values(baseline, spec, True)
        if spec.family == "laboratory":
            unit = spec.baseline_column.removesuffix("__value") + "__unit" if spec.baseline_column else ""
            out["pred_unit__" + spec.name] = baseline.get(unit, pd.Series(pd.NA, index=baseline.index, dtype="string"))
    for domain in DOMAINS:
        out["eligible_new_" + domain] = domain_values(baseline, domain).eq(0)
    out["eligible_any_new_domain"] = out[["eligible_new_" + d for d in DOMAINS]].any(axis=1)
    return out


INTERVAL_COLUMNS = ["patient_id", "from_clinical_episode_id", "to_clinical_episode_id",
    "from_date", "to_date", "interval_days", "interval_years", "start_time", "stop_time",
    "from_essdai", "to_essdai", "delta_essdai", "from_activity_state", "to_activity_state",
    "event_cross_ge5", "event_cross_high", "at_risk_first_ge5", "event_first_ge5",
    "at_risk_any_new_domain", "event_any_new_domain", "from_essdai_version",
    "from_essdai_total_version", "from_essdai_total_source"]


def canonical_domain_weights():
    """Reuse config weights with exactly the verified Step 13 domain spellings."""
    aliases = {"gland_swell": "glandular", "neuro_periph": "pns"}
    weights = {aliases.get(name, name): weight for name, weight in ESSDAI_DOMAIN_WEIGHTS.items()}
    if set(weights) != set(DOMAINS) or len(weights) != len(ESSDAI_DOMAIN_WEIGHTS):
        raise ValueError("ESSDAI weight/domain mapping disagrees with canonical Step 13 domains")
    return weights


def derive_domain_composition_side(frame, side, registry):
    """Fail closed on both sides; upstream versions are never inferred from equality.

    No available manifest in the current producer contract certifies partial
    panels. Require all 12 evaluable scores to reconstruct the recorded total.
    Raw contributions remain available for audit, including rejected subtractions.
    """
    if side not in {"from", "to"}:
        raise ValueError("Composition side must be from or to")
    out = frame.copy()
    weights = canonical_domain_weights()
    total = pd.to_numeric(out[side + "_essdai"], errors="coerce").astype(float)
    scores = pd.DataFrame({d: pd.to_numeric(out[side + "_domain_" + d], errors="coerce").astype(float) for d in DOMAINS})
    complete = scores.isin([0, 1, 2, 3]).all(axis=1)
    weighted = scores.mul(pd.Series(weights))
    reconstructed = weighted.sum(axis=1, min_count=len(DOMAINS))
    concordant = complete & total.notna() & np.isclose(total, reconstructed, atol=COMPOSITION_TOLERANCE, rtol=0)
    mismatch = complete & total.notna() & ~concordant
    category = pd.Series("complete_panel_concordant", index=out.index)
    category.loc[mismatch] = "complete_panel_discordant"
    category.loc[~complete] = "incomplete_panel"
    category.loc[total.isna()] = "missing_total"
    out[side + "_complete_domain_panel"] = complete
    out[side + "_reconstructed_essdai"] = reconstructed
    out[side + "_total_minus_reconstructed"] = total - reconstructed
    out[side + "_composition_concordant"] = concordant
    out[side + "_composition_category"] = category
    specs = {s.name.removeprefix("domain_"): s for s in registry if s.family == "essdai_domain"}
    out[side + "_composition_complete_panel"] = complete
    out[side + "_composition_total_delta"] = (total - reconstructed).where(complete & total.notna())
    out[side + "_composition_complete_panel_discordant"] = mismatch
    out[side + "_composition_complete_concordant"] = complete & total.notna() & ~mismatch
    version = out.get(side + "_essdai_version", pd.Series(pd.NA, index=out.index)).astype("string")
    total_version = out.get(side + "_essdai_total_version", pd.Series(pd.NA, index=out.index)).astype("string")
    incompatible = (version.notna() & total_version.notna() & version.ne(total_version)).fillna(False).astype(bool)
    qc, audit = [], []
    for domain, weight in weights.items():
        ordinal = scores[domain]
        contribution = weight * ordinal
        other = total - contribution
        known = total.notna() & ordinal.notna()
        invalid_ordinal = ordinal.notna() & ~ordinal.isin([0, 1, 2, 3])
        invalid_value = known & (~np.isfinite(other) | other.lt(0) | invalid_ordinal)
        valid = known & ~invalid_value & concordant
        domain_category = category.copy()
        domain_category.loc[invalid_value] = "invalid_subtraction"
        domain_category.loc[ordinal.isna()] = "missing_target_domain"
        domain_category.loc[total.isna()] = "missing_total"
        out[side + "_domain_weighted_contribution_raw_" + domain] = contribution
        out[side + "_domain_contribution_" + domain] = contribution.where(valid)
        out[side + "_essdai_other_domains_" + domain] = other.where(valid)
        out[side + "_composition_category_" + domain] = domain_category
        qc.append({"side": side, "domain": domain, "domain_weight": weight,
            "source_columns": json.dumps([ESSDAI, specs[domain].time_varying_column if domain in specs else None]),
            "n_intervals": len(out), "n_available": int(valid.sum()),
            "n_missing_total_or_domain": int((~known).sum()),
            "n_invalid_subtraction": int(invalid_value.sum()),
            "n_complete_panel_total_mismatch": int(mismatch.sum()),
            "n_incomplete_panels": int((~complete).sum()),
            "n_complete_panel_concordant": int(concordant.sum()),
            "rule_version": COMPOSITION_RULE_VERSION,
            "support_reason": "invalid or missing composition excluded; no correction/imputation",
            "composition_provenance_manifest_present": False,
            "comparison_scope": "complete evaluable concordant panels only; partial panels excluded"})
    return out, pd.DataFrame(qc)


def derive_other_domain_adjustments(intervals, registry):
    out, from_qc = derive_domain_composition_side(intervals, "from", registry)
    out, to_qc = derive_domain_composition_side(out, "to", registry)
    return out, pd.concat([from_qc, to_qc], ignore_index=True)


def composition_episode_audit(df, registry, variable_registry=None):
    """Audit unique canonical episodes, never interval copies or corrected totals."""
    fields = KEYS + ["clinical_anchor_date", "clinical_visit_number"]
    fields += [c for c in ("source_protocol", "_essdai_total_version", "_essdai_version", "_essdai_domain_version",
                          "essdai_total_source", "_essdai_resolution_method") if c in df]
    audit = df[fields].copy()
    side = pd.DataFrame({"from_essdai": df[ESSDAI]}, index=df.index)
    for domain in DOMAINS:
        side["from_domain_" + domain] = domain_values(df, domain)
    composition, _ = derive_domain_composition_side(side, "from", registry)
    audit["essdai_total"] = df[ESSDAI]
    for domain in DOMAINS:
        audit["domain_" + domain] = side["from_domain_" + domain]
        audit["weighted_" + domain] = composition["from_domain_weighted_contribution_raw_" + domain]
    for target, source in (("reconstructed_essdai", "reconstructed_essdai"),
                           ("total_minus_reconstructed", "total_minus_reconstructed"),
                           ("complete_domain_panel", "complete_domain_panel"),
                           ("composition_concordant", "composition_concordant"),
                           ("composition_category", "composition_category")):
        audit[target] = composition["from_" + source]
    audit["absolute_difference"] = audit.total_minus_reconstructed.abs()
    negative = pd.DataFrame({d: audit["weighted_" + d].gt(audit.essdai_total) for d in DOMAINS})
    audit["articular_negative_subtraction"] = negative.articular
    audit["any_negative_target_subtraction"] = negative.any(axis=1)
    audit["essdai_total_producer"] = "01_pop_distribution.py"
    audit["domain_producer"] = "06_overlap_glandular.py"
    audit["producer_evidence"] = "Step 10 producer contract; runtime registry unavailable"
    if variable_registry is not None and {"public_variable", "producer_script"}.issubset(variable_registry):
        mapping = variable_registry.dropna(subset=["public_variable", "producer_script"])
        mapping = mapping.groupby("public_variable").producer_script.agg(lambda s: ";".join(sorted(set(map(str, s)))))
        if ESSDAI in mapping.index:
            audit["essdai_total_producer"] = mapping.loc[ESSDAI]
        columns = [s.time_varying_column for s in registry if s.family == "essdai_domain" and s.time_varying_column]
        producers = sorted(set(mapping.loc[c] for c in columns if c in mapping.index))
        if producers:
            audit["domain_producer"] = ";".join(producers)
        audit["producer_evidence"] = "10_variable_registry.csv and Step 10 producer contract"
    # Version fields are retained only when supplied upstream, never inferred.
    for original, target in (("_essdai_total_version", "essdai_total_version"), ("_essdai_domain_version", "domain_version")):
        if original in audit:
            audit[target] = audit[original]
    audit["provenance_status"] = "same_version_not_certified; complete_concordance_required"
    return audit


def composition_discordance_summary(audit):
    rows = []
    dimensions = {"composition_category": audit.composition_category,
        "total_minus_reconstructed": audit.total_minus_reconstructed,
        "absolute_difference": audit.absolute_difference,
        "clinical_visit_number": audit.clinical_visit_number,
        "calendar_year": pd.to_datetime(audit.clinical_anchor_date).dt.year,
        "essdai_total_state": step13.classify_essdai_series(audit.essdai_total),
        "articular_active": audit.domain_articular.gt(0).where(audit.domain_articular.notna()),
        "articular_negative_subtraction": audit.articular_negative_subtraction,
        "any_negative_target_subtraction": audit.any_negative_target_subtraction,
        "essdai_total_producer": audit.essdai_total_producer,
        "domain_producer": audit.domain_producer}
    for column in ("source_protocol", "essdai_total_version", "domain_version"):
        if column in audit:
            dimensions[column] = audit[column]
    for domain in DOMAINS:
        values = audit["domain_" + domain]
        dimensions[domain + "_active"] = values.gt(0).where(values.notna())
    for scope, frame in (("all_episodes", audit), ("discordant_episodes", audit.loc[audit.composition_category.eq("complete_panel_discordant")])):
        for dimension, values in dimensions.items():
            for value, count in values.loc[frame.index].value_counts(dropna=False).items():
                rows.append({"scope": scope, "dimension": dimension, "value": str(value), "n_episodes": int(count)})
    return pd.DataFrame(rows, columns=["scope", "dimension", "value", "n_episodes"])


def composition_snapshot(audit):
    discordant = audit.composition_category.eq("complete_panel_discordant")
    return {"n_episodes_audited": len(audit), "n_complete_panels": int(audit.complete_domain_panel.sum()),
        "n_concordant": int(audit.composition_concordant.sum()), "n_discordant": int(discordant.sum()),
        "n_incomplete_panels": int((~audit.complete_domain_panel).sum()),
        "delta_distribution": {str(k): int(v) for k, v in audit.loc[discordant, "total_minus_reconstructed"].value_counts().sort_index().items()},
        "n_articular_negative_subtractions": int(audit.articular_negative_subtraction.sum())}


def build_leave_source_domain_qc(intervals):
    rows = []
    for domain in DOMAINS:
        fr = intervals["from_essdai_other_domains_" + domain].notna()
        to = intervals["to_essdai_other_domains_" + domain].notna()
        rows.append({"domain": domain, "n_intervals": len(intervals), "n_valid_from": int(fr.sum()),
            "n_valid_to": int(to.sum()), "n_valid_both": int((fr & to).sum()),
            "n_excluded_from": int((~fr).sum()), "n_excluded_to": int((~to).sum()),
            "n_complete_both_patients": int(intervals.loc[fr & to, "patient_id"].nunique()),
            "composition_requirement": "complete evaluable concordant panels on FROM and TO",
            "primary_bh_eligible": False, "rule_version": COMPOSITION_RULE_VERSION})
    return pd.DataFrame(rows)


def build_lagged_interval_dataset(df, baseline, registry):
    """Adjacent episodes only, even when an intermediate outcome is missing.

    Event risk ends at the first observed event, regardless of predictor missingness.
    Incident domains require confirmed inactivity at official baseline and FROM.
    """
    values = {s.name: predictor_values(df, s, False) for s in registry if s.time_varying_column}
    domains = {d: domain_values(df, d) for d in DOMAINS}
    base = baseline.set_index("patient_id")
    rows = []
    for pid, g in df.groupby("patient_id", sort=True):
        g = g.sort_values(["clinical_anchor_date", "clinical_visit_number", "clinical_episode_id"], kind="stable")
        b = base.loc[pid]
        eligible_domains = {d for d in DOMAINS if bool(b["eligible_new_" + d])}
        seen_domains = set()
        seen_ge5 = not bool(b.eligible_ge5)
        for pos in range(len(g)):
            a = g.iloc[pos]
            # Update from all episodes, including invalid/zero-length intervals.
            seen_ge5 |= a[ESSDAI] >= step13.ESSDAI_MODERATE_OR_HIGH_CUTOFF if pd.notna(a[ESSDAI]) else False
            seen_domains.update(d for d in DOMAINS if pd.notna(domains[d].loc[a.name]) and domains[d].loc[a.name] > 0)
            if pos == len(g) - 1:
                continue
            z = g.iloc[pos + 1]
            days = (z.clinical_anchor_date - a.clinical_anchor_date).total_seconds()/86400
            if days <= 0:
                continue
            av, zv = a[ESSDAI], z[ESSDAI]
            known = pd.notna(av) and pd.notna(zv)
            row = {"patient_id": pid, "from_clinical_episode_id": str(a.clinical_episode_id),
                "to_clinical_episode_id": str(z.clinical_episode_id),
                "from_date": a.clinical_anchor_date, "to_date": z.clinical_anchor_date,
                "interval_days": days, "interval_years": days/365.25,
                "start_time": a[TIME], "stop_time": z[TIME], "from_essdai": av, "to_essdai": zv,
                "delta_essdai": zv-av, "from_activity_state": step13.classify_essdai_activity(av),
                "to_activity_state": step13.classify_essdai_activity(zv),
                "event_cross_ge5": float(av < step13.ESSDAI_MODERATE_OR_HIGH_CUTOFF and zv >= step13.ESSDAI_MODERATE_OR_HIGH_CUTOFF) if known else np.nan,
                "event_cross_high": float(step13.classify_essdai_activity(av) == "Moderate" and step13.classify_essdai_activity(zv) == "High") if known else np.nan,
                "at_risk_first_ge5": known and not seen_ge5,
                "from_essdai_version": a.get("_essdai_version", pd.NA),
                "from_essdai_total_version": a.get("_essdai_total_version", pd.NA),
                "from_essdai_total_source": a.get("essdai_total_source", "Step 10 official total; original producer provenance unavailable"),
                "event_first_ge5": float(zv >= step13.ESSDAI_MODERATE_OR_HIGH_CUTOFF) if known and not seen_ge5 else np.nan}
            for name, series in values.items():
                row["pred__" + name] = series.loc[a.name]
            for spec in registry:
                if spec.family == "laboratory" and spec.time_varying_column:
                    unit = spec.time_varying_column.removesuffix("__value") + "__unit"
                    row["pred_unit__" + spec.name] = a.get(unit, pd.NA)
            for d in DOMAINS:
                fr, to = domains[d].loc[a.name], domains[d].loc[z.name]
                risk = d in eligible_domains and d not in seen_domains and fr == 0 and pd.notna(to)
                row["from_domain_" + d], row["to_domain_" + d] = fr, to
                row["at_risk_new_" + d] = bool(risk)
                row["event_new_domain_" + d] = float(to > 0) if risk else np.nan
            remaining = eligible_domains - seen_domains
            # A negative global interval requires evaluation of ALL remaining
            # baseline-inactive domains; partial negative panels are unknown.
            global_start = bool(remaining) and all(row["from_domain_" + d] == 0 for d in remaining)
            new_positive = any(pd.notna(row["to_domain_" + d]) and row["to_domain_" + d] > 0 for d in remaining)
            global_known = all(pd.notna(row["to_domain_" + d]) for d in remaining)
            risk_any = global_start and (new_positive or global_known) and not bool(eligible_domains & seen_domains)
            row["at_risk_any_new_domain"] = risk_any
            row["event_any_new_domain"] = float(new_positive) if risk_any else np.nan
            rows.append(row)
    extra = ["pred__" + s.name for s in registry if s.time_varying_column]
    extra += ["pred_unit__" + s.name for s in registry if s.family == "laboratory" and s.time_varying_column]
    extra += [prefix + d for d in DOMAINS for prefix in
              ("from_domain_", "to_domain_", "at_risk_new_", "event_new_domain_")]
    out = pd.DataFrame(rows, columns=INTERVAL_COLUMNS + extra)
    out = out.merge(baseline[["patient_id", "baseline_essdai", "pred__age", "pred__sex", "pred__duration"]]
                     .rename(columns={"pred__age": "age", "pred__sex": "sex", "pred__duration": "duration"}),
                     on="patient_id", how="left", validate="many_to_one")
    out, qc = derive_other_domain_adjustments(out, registry)
    out.attrs["domain_composition_qc"] = qc.to_dict("records")
    return out


RISK_COLUMNS = ["patient_id", "target_domain", "risk_start_date", "risk_end_date",
    "time_years", "event", "event_date", "baseline_target_domain_score", "last_non_event_time_years"]


def build_domain_risk_sets(df, baseline):
    """First observed incident activity, censor only at an evaluable assessment."""
    domains = {d: domain_values(df, d) for d in DOMAINS}
    rows = []
    for pid, g in df.groupby("patient_id", sort=True):
        b = baseline.loc[baseline.patient_id.eq(pid)].iloc[0]
        after = g.loc[g.clinical_anchor_date.gt(b.clinical_baseline_date)]
        if after.empty:
            continue
        eligible = [d for d in DOMAINS if b["eligible_new_" + d]]
        for d in [*eligible, "any_new_domain"]:
            scores = domains[d].reindex(after.index) if d != "any_new_domain" else None
            positive = scores.gt(0) if scores is not None else pd.DataFrame(
                {k: domains[k].reindex(after.index).gt(0) for k in eligible}).any(axis=1)
            evaluable = scores.notna() if scores is not None else pd.DataFrame(
                {k: domains[k].reindex(after.index).notna() for k in eligible}).all(axis=1)
            if not eligible or not (positive | evaluable).any():
                continue
            event = bool(positive.any())
            end = after.loc[positive].iloc[0] if event else after.loc[evaluable].iloc[-1]
            prior = after.loc[evaluable & ~positive & after.clinical_anchor_date.lt(end.clinical_anchor_date)]
            years = (end.clinical_anchor_date - b.clinical_baseline_date).total_seconds()/(86400*365.25)
            rows.append({"patient_id": pid, "target_domain": d,
                "risk_start_date": b.clinical_baseline_date, "risk_end_date": end.clinical_anchor_date,
                "time_years": years, "event": int(event),
                "event_date": end.clinical_anchor_date if event else pd.NaT,
                "baseline_target_domain_score": 0,
                "last_non_event_time_years": (prior.iloc[-1].clinical_anchor_date - b.clinical_baseline_date).days/365.25 if len(prior) else 0})
    return pd.DataFrame(rows, columns=RISK_COLUMNS).merge(baseline, on="patient_id", how="left", validate="many_to_one")


def build_worsening_sensitivity_risk_set(df, baseline):
    """Only total ESSDAI uses the upstream legacy +5 change sensitivity."""
    rows = []
    cutoff = CHANGE_THRESHOLDS["essdai_legacy_worsening"]
    for pid, g in df.groupby("patient_id"):
        b = baseline.loc[baseline.patient_id.eq(pid)].iloc[0]
        if pd.isna(b.baseline_essdai):
            continue
        after = g.loc[g.clinical_anchor_date.gt(b.clinical_baseline_date) & g[ESSDAI].notna()]
        if after.empty:
            continue
        events = (after[ESSDAI] - b.baseline_essdai).ge(cutoff)
        end = after.loc[events].iloc[0] if events.any() else after.iloc[-1]
        rows.append({"patient_id": pid, "event": int(events.any()),
                     "time_years": (end.clinical_anchor_date-b.clinical_baseline_date).days/365.25})
    return pd.DataFrame(rows, columns=["patient_id", "event", "time_years"]).merge(baseline, on="patient_id", validate="one_to_one")


def run_temporal_leakage_qc(df, baseline, intervals, risk_sets, registry, hard_fail=True):
    """Audit copied predictor values against their exact source episodes."""
    violations = []
    def record(check, mask, frame, detail=""):
        for index in frame.index[pd.Series(mask, index=frame.index).fillna(True)]:
            row = frame.loc[index]
            violations.append({"check": check, "status": "violation", "patient_id": row.get("patient_id"),
                "from_clinical_episode_id": row.get("from_clinical_episode_id"),
                "to_clinical_episode_id": row.get("to_clinical_episode_id"), "detail": detail})
    record("strict_interval_order", ~intervals.from_date.lt(intervals.to_date), intervals)
    record("duplicate_intervals", intervals.duplicated(["patient_id", "from_clinical_episode_id", "to_clinical_episode_id"]), intervals)
    record("negative_followup", intervals.start_time.lt(0) | intervals.stop_time.le(intervals.start_time), intervals)
    indexed = df.assign(clinical_episode_id=df.clinical_episode_id.astype("string")).set_index(KEYS)
    episode_composition = pd.DataFrame({"from_essdai": df[ESSDAI]}, index=df.index)
    for domain in DOMAINS:
        episode_composition["from_domain_" + domain] = domain_values(df, domain)
    episode_composition, _ = derive_domain_composition_side(episode_composition, "from", registry)
    episode_composition.index = pd.MultiIndex.from_frame(df[KEYS].astype("string"))
    def same_value(expected, actual):
        return (pd.isna(expected) and pd.isna(actual)) or (
            pd.notna(expected) and pd.notna(actual) and bool(np.isclose(expected, actual, atol=COMPOSITION_TOLERANCE, rtol=0)))
    for i, row in intervals.iterrows():
        key = (row.patient_id, row.from_clinical_episode_id)
        source = indexed.loc[[key]].reset_index()
        fr = indexed.loc[key]
        record("from_episode_date", [fr.clinical_anchor_date != row.from_date], intervals.loc[[i]])
        for side in ("from", "to"):
            side_key = (row.patient_id, row[side + "_clinical_episode_id"])
            episode = indexed.loc[side_key]
            record(side + "_episode_date", [episode.clinical_anchor_date != row[side + "_date"]], intervals.loc[[i]])
            record(side + "_episode_essdai", [not same_value(episode[ESSDAI], row[side + "_essdai"])], intervals.loc[[i]])
            composition = episode_composition.loc[side_key]
            for domain in DOMAINS:
                for component in ("domain_", "essdai_other_domains_"):
                    column = side + "_" + component + domain
                    record(side + "_composition_source_episode", [not same_value(composition["from_" + component + domain], row[column])], intervals.loc[[i]], domain + ":" + component)
        ordered = df.loc[df.patient_id.eq(row.patient_id)].sort_values(
            ["clinical_anchor_date", "clinical_visit_number", "clinical_episode_id"], kind="stable")
        ids = ordered.clinical_episode_id.astype(str).tolist()
        expected = ids[ids.index(row.from_clinical_episode_id) + 1]
        record("adjacent_canonical_episodes", [expected != row.to_clinical_episode_id], intervals.loc[[i]])
        for spec in registry:
            if not spec.time_varying_column:
                continue
            expected_value = predictor_values(source, spec, False).iloc[0]
            actual = row.get("pred__" + spec.name, np.nan)
            equal = (pd.isna(expected_value) and pd.isna(actual)) or (
                pd.notna(expected_value) and pd.notna(actual) and expected_value == actual)
            record("predictor_from_episode", [not equal], intervals.loc[[i]], spec.name)
            col = spec.time_varying_column
            date_col = col.removesuffix("__value") + "__measurement_date"
            if pd.notna(actual) and date_col in source:
                date = pd.to_datetime(source[date_col].iloc[0], errors="coerce")
                record("no_future_measurement", [pd.isna(date) or date > row.from_date], intervals.loc[[i]], spec.name)
        for domain in DOMAINS:
            risk = bool(row["at_risk_new_" + domain])
            record("inactive_target_at_start", [risk and row["from_domain_" + domain] != 0], intervals.loc[[i]], domain)
    for spec in registry:
        expected = predictor_values(baseline, spec, True)
        actual = baseline["pred__" + spec.name]
        record("official_baseline_predictor", ~(expected.eq(actual) | (expected.isna() & actual.isna())), baseline, spec.name)
    record("riskset_inactive_at_entry", risk_sets.baseline_target_domain_score.ne(0), risk_sets)
    record("positive_risk_followup", risk_sets.time_years.le(0), risk_sets)
    official = baseline.set_index("patient_id")
    for i, row in risk_sets.iterrows():
        entry = official.loc[[row.patient_id]].reset_index()
        domain = row.target_domain
        eligible = [d for d in DOMAINS if domain_values(entry, d).iloc[0] == 0]
        record("riskset_official_inactive_entry", [domain not in eligible if domain != "any_new_domain" else not eligible], risk_sets.loc[[i]], domain)
        endpoint = df.loc[df.patient_id.eq(row.patient_id) & df.clinical_anchor_date.eq(row.risk_end_date)]
        scores = {d: domain_values(endpoint, d) for d in (eligible if domain == "any_new_domain" else [domain])}
        positive = any(s.gt(0).any() for s in scores.values())
        known = bool(scores) and all(s.notna().any() for s in scores.values())
        record("riskset_evaluable_endpoint", [not positive if row.event else not known or positive], risk_sets.loc[[i]], domain)
    out = pd.DataFrame(violations, columns=["check", "status", "patient_id", "from_clinical_episode_id", "to_clinical_episode_id", "detail"])
    if len(out) and hard_fail:
        raise ValueError(f"Structural temporal leakage: {len(out)} violations; {out.to_dict('records')}")
    if out.empty:
        out = pd.DataFrame([{"check": c, "status": "pass", "detail": "0 violations"} for c in (
            "strict_interval_order", "duplicate_intervals", "negative_followup", "adjacent_canonical_episodes",
            "predictor_from_episode", "no_future_measurement", "official_baseline_predictor",
            "inactive_target_at_start", "riskset_inactive_at_entry", "positive_risk_followup",
            "riskset_official_inactive_entry", "riskset_evaluable_endpoint")])
        out = pd.concat([out, pd.DataFrame([{"check": side + "_" + check, "status": "pass", "detail": "0 violations"}
            for side in ("from", "to") for check in ("episode_essdai", "composition_source_episode", "episode_date")])], ignore_index=True)
    return out


@dataclass
class ModelTask:
    spec: PredictorSpec
    data: pd.DataFrame
    analysis_type: str
    outcome: str
    kind: str
    sensitivity: str = "none"
    omit_baseline_essdai: bool = False
    support_exclusion: str = ""
    minimum_exposed_events: int = MIN_EXPOSED_EVENTS
    state_covariate: str = "from_essdai"
    fixed_adjustment_level: str = ""
    extra_covariates: tuple = ()
    comparison_role: str = ""


def _family(task):
    group = {"essdai_domain": "organ_domains", "glandular": "clinical",
             "extended_clinical": "clinical"}.get(task.spec.family, task.spec.family)
    if task.analysis_type == "cross_domain":
        return "cross_domain__incident_domain"
    scope = task.analysis_type
    if task.sensitivity != "none" or task.spec.primary_role == "sensitivity":
        scope = "sensitivity_" + scope + "_" + task.sensitivity
    return f"{scope}_{group}__{task.outcome}"


def _scaled_values(values, spec):
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if spec.transform == "binary_active":
        return numeric.gt(0).astype(float).where(numeric.notna()), np.nan, np.nan
    mean, sd = numeric.mean(), numeric.std(ddof=1)
    if spec.transform == "zscore":
        return (numeric-mean)/sd if pd.notna(sd) and sd > 0 else numeric*np.nan, mean, sd
    return numeric, np.nan, np.nan


def _design_formula(task, adjustment, data=None):
    if task.kind == "trajectory":
        terms = ["x * time"]
    elif task.kind == "lagged":
        terms = ["x"] + [c for c in (task.state_covariate, "interval_years")
                           if data is None or data[c].nunique(dropna=True) > 1]
    elif task.kind == "poisson":
        terms = ["x", "C(time_band)"]
    else:
        terms = ["x"]
    terms += ["C(sex)" if c == "sex" else c for c in adjustment]
    return "y ~ " + " + ".join(terms)


def _support_counts(data, task):
    numeric = pd.to_numeric(data.x, errors="coerce")
    categorical = task.spec.variable_type in {"binary", "ordinal"} or task.spec.transform == "binary_active"
    # Exposure counts for ordinal scores use active >0, but effect is per level.
    positive = numeric.gt(0) if categorical else pd.Series(False, index=data.index)
    negative = numeric.eq(0) if categorical else pd.Series(False, index=data.index)
    events = data.y.eq(1) if task.kind in {"cox", "poisson"} else pd.Series(False, index=data.index)
    repeated = data.groupby("patient_id").size().ge(2).sum()
    return {"n_patients": data.patient_id.nunique(), "n_observations": len(data),
        "n_intervals": len(data) if task.kind in {"lagged", "poisson"} else 0,
        "n_events": int(events.sum()) if task.kind in {"cox", "poisson"} else np.nan,
        "n_exposed": data.loc[positive, "patient_id"].nunique() if categorical else np.nan,
        "n_unexposed": data.loc[negative, "patient_id"].nunique() if categorical else np.nan,
        "n_exposed_events": int((events & positive).sum()) if categorical else np.nan,
        "n_unexposed_events": int((events & negative).sum()) if categorical else np.nan,
        "n_unique_predictor_levels": numeric.nunique(), "n_repeated_patients": int(repeated)}


def _check_support(data, task, minimum_events, formula):
    counts = _support_counts(data, task)
    reasons = []
    if task.support_exclusion:
        reasons.append(task.support_exclusion)
    if counts["n_patients"] < MIN_PATIENTS_MODEL:
        reasons.append(f"fewer than {MIN_PATIENTS_MODEL} complete-case patients")
    if counts["n_unique_predictor_levels"] < MIN_PREDICTOR_VARIATION:
        reasons.append("insufficient predictor variation")
    if task.kind == "trajectory" and counts["n_repeated_patients"] < MIN_REPEATED_PATIENTS:
        reasons.append(f"fewer than {MIN_REPEATED_PATIENTS} repeated patients")
    if task.kind in {"lagged", "poisson"} and len(data) < MIN_INTERVALS:
        reasons.append(f"fewer than {MIN_INTERVALS} complete intervals")
    if pd.notna(counts["n_exposed"]):
        if counts["n_exposed"] < MIN_EXPOSED or counts["n_unexposed"] < MIN_UNEXPOSED:
            reasons.append("insufficient exposed/unexposed patients")
    parameters = 0
    if len(data):
        try:
            _, design = patsy.dmatrices(formula, data, return_type="dataframe")
            parameters = design.shape[1] - (1 if task.kind == "cox" else 0)
            if np.linalg.matrix_rank(design) < design.shape[1]:
                reasons.append("rank-deficient model design")
        except (ValueError, TypeError, patsy.PatsyError) as exc:
            reasons.append(f"design not estimable: {exc}")
    if task.kind in {"cox", "poisson"}:
        minimum = max(minimum_events, MIN_DOMAIN_EVENTS) if "domain" in task.outcome else minimum_events
        if counts["n_events"] < minimum:
            reasons.append(f"fewer than {minimum} events")
        if pd.notna(counts["n_exposed_events"]) and counts["n_exposed_events"] < task.minimum_exposed_events:
            reasons.append(f"fewer than {task.minimum_exposed_events} exposed events")
        if pd.notna(counts["n_unexposed_events"]) and counts["n_unexposed_events"] < MIN_EXPOSED_EVENTS:
            reasons.append(f"fewer than {MIN_EXPOSED_EVENTS} unexposed events")
        if counts["n_events"] / max(parameters, 1) < EVENTS_PER_PARAMETER:
            reasons.append(f"fewer than {EVENTS_PER_PARAMETER} events per parameter")
    return counts, parameters, reasons


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def prepare_model(task, n_source_patients, minimum_events=MIN_EVENTS_COX):
    """One auditable complete-case cohort drives support, attrition and fitting.

    Select the highest supported prespecified level using missingness, design and
    event support only. Numerical failures do not trigger covariate reselection.
    """
    d = task.data.copy()
    xcol = "pred__" + task.spec.name
    if xcol not in d:
        d[xcol] = np.nan
    scaling_cohort = d.drop_duplicates("patient_id")[xcol] if task.analysis_type == "baseline" else d[xcol]
    _, mean, sd = _scaled_values(scaling_cohort, task.spec)
    if task.spec.transform == "zscore":
        numeric = pd.to_numeric(d[xcol], errors="coerce").astype(float)
        d["x"] = (numeric-mean)/sd if pd.notna(sd) and sd > 0 else numeric*np.nan
    else:
        d["x"], _, _ = _scaled_values(d[xcol], task.spec)
    for name in ("age", "sex", "duration"):
        if name not in d:
            d[name] = d.get("pred__" + name, np.nan)
    if "sex" in d:
        d["sex"] = d.sex.astype("string").str.strip().replace("", pd.NA).astype(object)
    d["y"] = pd.to_numeric(d.get("y", pd.Series(np.nan, index=d.index)), errors="coerce")
    required = ["patient_id", "y", "x"]
    if task.kind == "trajectory":
        required += ["time"]
    elif task.kind == "lagged":
        required += [task.state_covariate, "interval_years"]
    elif task.kind == "cox":
        required += ["time_years"]
    elif task.kind == "poisson":
        required += ["interval_years", "start_time"]
        # Fixed bands on FROM time preserve the original canonical intervals.
        d["time_band"] = pd.cut(d.start_time, [-np.inf, 1, 3, np.inf],
                                labels=["<1y", "1-3y", ">=3y"], right=False).astype(object)
        required += ["time_band"]
    core = [] if task.omit_baseline_essdai or task.kind == "lagged" else ["baseline_essdai"]
    core += list(task.extra_covariates)
    desired = core + ["age", "sex", "duration"]
    for name in set(required + desired):
        if name not in d:
            d[name] = np.nan
    d = d.replace([np.inf, -np.inf], np.nan)
    # Nonpositive follow-up/offsets cannot enter a model; source rows stay intact.
    positive_time = "time_years" if task.kind == "cox" else "interval_years" if task.kind in {"lagged", "poisson"} else "time"
    d.loc[~pd.to_numeric(d[positive_time], errors="coerce").gt(0), positive_time] = np.nan
    n_outcome = d.dropna(subset=[c for c in required if c != "x"]).patient_id.nunique()
    n_predictor = d.loc[d.x.notna()].patient_id.nunique()
    levels = [core + extra for extra in (["age", "sex", "duration"], ["age", "sex"], ["age"], [])]
    evaluated, candidate_qc = [], []
    selected = None
    for level, candidate in zip(ADJUSTMENT_LEVELS, levels):
        subset = d.dropna(subset=list(dict.fromkeys(required + candidate))).copy()
        if task.kind == "trajectory":
            ids = subset.groupby("patient_id").time.nunique()
            subset = subset.loc[subset.patient_id.isin(ids.index[ids.ge(2)])].copy()
        unavailable = [c for c in candidate if not d[c].notna().any()]
        constants = [c for c in candidate if c not in unavailable and subset[c].nunique(dropna=True) <= 1]
        usable = [c for c in candidate if c not in constants]
        # Observed categories only. Never turn missing sex into a string/category.
        if "sex" in usable:
            subset["sex"] = subset.sex.astype(str)
        formula = _design_formula(task, usable, subset)
        counts, parameters, reasons = _check_support(subset, task, minimum_events, formula)
        if unavailable:
            reasons.append("unavailable adjustment: " + ",".join(unavailable))
        demographic_constants = [c for c in constants if c not in core]
        if demographic_constants:
            reasons.append("constant adjustment: " + ",".join(demographic_constants))
        for covariate in task.extra_covariates:
            if covariate in constants:
                reasons.append("constant required sensitivity adjustment: " + covariate)
            if covariate == "treatment_exposure":
                treated = subset.loc[subset[covariate].eq(1), "patient_id"].nunique()
                untreated = subset.loc[subset[covariate].eq(0), "patient_id"].nunique()
                if treated < MIN_EXPOSED or untreated < MIN_UNEXPOSED:
                    reasons.append("insufficient treated/untreated patients for sensitivity")
        unit_col = "pred_unit__" + task.spec.name
        units = sorted(map(str, subset[unit_col].dropna().unique())) if unit_col in subset else []
        if task.spec.family == "laboratory" and len(units) > 1:
            reasons.append("mixed laboratory units; no conversion defined in Step 14")
        supported = not reasons
        if supported and selected is None and (not task.fixed_adjustment_level or task.fixed_adjustment_level == level):
            selected = len(evaluated)
        evaluated.append((subset, usable, formula, counts, parameters, reasons, units))
        candidate_qc.append({"level": level, "candidate_covariates": candidate,
            "effective_covariates": usable, "unavailable": unavailable, "constant": constants,
            "n_missing_by_covariate": {c: int(d[c].isna().sum()) for c in candidate},
            "n_complete_cases": len(subset), **counts, "candidate_parameters": parameters,
            "formula": formula, "supported": supported,
            "support_reason": "; ".join(dict.fromkeys(reasons)) or "supported"})
    # Unsupported tasks retain the Minimal attempted design and its exact counts.
    chosen = selected if selected is not None else (
        ADJUSTMENT_LEVELS.index(task.fixed_adjustment_level) if task.fixed_adjustment_level else len(evaluated) - 1)
    full, usable, formula, counts, parameters, reasons, units = evaluated[chosen]
    adjustment_status = ADJUSTMENT_LEVELS[chosen]
    full_support_reason = candidate_qc[0]["support_reason"]
    full_complete_cases = candidate_qc[0]["n_complete_cases"]
    full_parameters = candidate_qc[0]["candidate_parameters"]
    missing = sorted(set(candidate_qc[0]["unavailable"] + candidate_qc[0]["constant"]))
    # pandas extension strings are converted for patsy/statsmodels compatibility.
    if "time_band" in full:
        full["time_band"] = full.time_band.astype(str)
    full["patient_id"] = full.patient_id.astype(str)
    full = full.reset_index(drop=True)
    metadata = {**counts, "n_source_cohort": n_source_patients,
        "n_with_outcome_support": n_outcome, "n_with_predictor_available": n_predictor,
        "n_complete_cases": len(full), "candidate_parameters": parameters,
        "n_full_complete_cases": full_complete_cases, "full_candidate_parameters": full_parameters,
        "full_model_support_reason": full_support_reason,
        "events_per_candidate_parameter": counts["n_events"]/max(parameters, 1),
        "supported_for_model": not reasons, "support_reason": "; ".join(dict.fromkeys(reasons)) or "supported",
        "adjustment_status": adjustment_status, "adjustment_covariates": ",".join(usable),
        "adjustment_status_legacy": "full" if adjustment_status == "full" else "prespecified_reduced",
        "adjustment_selection_rule": ADJUSTMENT_SELECTION_RULE,
        "candidate_adjustment_levels": ",".join(ADJUSTMENT_LEVELS),
        "selected_adjustment_level": adjustment_status if selected is not None else "none",
        "selection_reason": "highest supported estimable prespecified level" if selected is not None else "no candidate level supported",
        "adjustment_rule_version": ADJUSTMENT_RULE_VERSION,
        "candidate_adjustment_qc": json.dumps(_json_safe(candidate_qc), allow_nan=False),
        "unavailable_or_constant_adjustment": ",".join(missing), "formula": formula,
        "constant_state_covariates": ",".join(c for c in (task.state_covariate, "interval_years")
            if task.kind == "lagged" and c in full and full[c].nunique(dropna=True) <= 1),
        "state_covariate": task.state_covariate if task.kind == "lagged" else "baseline_essdai",
        "domain_weight": canonical_domain_weights().get(task.spec.name.removeprefix("domain_"), np.nan),
        "composition_source_columns": json.dumps([ESSDAI, task.spec.time_varying_column]) if task.spec.family == "essdai_domain" else "",
        "sensitivity_numerically_identical_state": bool(task.kind == "lagged" and task.state_covariate != "from_essdai" and len(full) and np.allclose(full[task.state_covariate], full.from_essdai)),
        "laboratory_units": ",".join(units) or "not_available",
        "standardization_mean": mean, "standardization_sd": sd,
        "standardization_cohort": "available as-of predictors in outcome-eligible source cohort, before covariate complete-case restriction",
        "structural_relationship": "ESSDAI total includes predictor domain" if task.spec.family == "essdai_domain" else "none"}
    sample_columns = [c for c in ("patient_id", "from_clinical_episode_id", "to_clinical_episode_id", "clinical_anchor_date", "time_years") if c in full]
    keys = full[sample_columns].astype(str).sort_values(sample_columns, kind="stable").values.tolist()
    metadata["analysis_sample_hash"] = hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
    metadata["matched_sample_hash"] = metadata["analysis_sample_hash"] if task.comparison_role and task.comparison_role != "primary_full_sample" else ""
    metadata["comparison_role"] = task.comparison_role
    metadata["fixed_adjustment_level"] = task.fixed_adjustment_level
    return full, metadata


def result_template(task, meta):
    scale = "per 1 SD higher value" if task.spec.transform == "zscore" else (
        "active vs inactive" if task.spec.transform == "binary_active" else
        "positive vs negative" if task.spec.variable_type == "binary" else "per +1 ordinal level")
    estimand = {"trajectory": "difference in annual ESSDAI slope",
        "lagged": "difference in subsequent ESSDAI conditional on starting ESSDAI",
        "cox": "first-observed-event hazard association", "poisson": "subsequent first-observed-event rate association"}[task.kind]
    if task.sensitivity == LEAVE_SOURCE_SENSITIVITY:
        estimand = "difference in subsequent ESSDAI outside source domain conditional on prior ESSDAI outside source domain"
    return {**meta, "analysis_type": task.analysis_type, "predictor": task.spec.name,
        "predictor_label": task.spec.label, "predictor_family": task.spec.family,
        "predictor_type": task.spec.variable_type, "baseline_column": task.spec.baseline_column,
        "time_varying_column": task.spec.time_varying_column,
        "transform": task.spec.transform, "effect_scale": scale, "outcome": task.outcome,
        "hypothesis_group_id": task.spec.hypothesis_group_id or task.spec.name,
        "primary_canonical": task.spec.primary_canonical,
        "alternative_to": task.spec.alternative_to, "analysis_role": task.spec.analysis_role,
        "redundancy_evidence": task.spec.redundancy_evidence,
        "hypothesis_rule_version": HYPOTHESIS_RULE_VERSION,
        "fdr_eligibility": task.spec.fdr_eligibility and task.sensitivity == "none",
        "claim_status": "sensitivity" if task.sensitivity != "none" else "unsupported", "inference_tier": "unsupported",
        "estimand": estimand, "sensitivity": task.sensitivity,
        "model_attempted": "none", "model_used": "none",
        "effect_measure": "HR" if task.kind == "cox" else "IRR" if task.kind == "poisson" else "beta",
        "estimate": np.nan, "standard_error": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
        "p_value": np.nan, "q_value": np.nan, "fdr_family": _family(task),
        "converged": False, "singular_fit": False, "fallback_used": False,
        "ph_assumption_p_value": np.nan, "ph_assumption_status": "not_tested",
        "model_status": "not_estimable", "interpretation_status": "descriptive_only",
        "valid_for_inference": False, "cluster_variable": "patient_id",
        "warning": (EVENT_NOTE if task.kind in {"cox", "poisson"} else "Association, not a causal effect; not a validated prediction model.") +
            ("; ESSDAI total includes this domain; review weighted other-domains sensitivity" if task.spec.family == "essdai_domain" else "") +
            ("; sensitivity state equals original numerically, not independent evidence" if meta["sensitivity_numerically_identical_state"] else ""),
        "model_selection_note": TV_NOTE if task.kind == "poisson" else "",
        "treatment_adjustment": "canonical as-of systemic treatment sensitivity" if task.extra_covariates else "primary models unchanged; see 14_treatment_adjustment_decision.csv"}


def _extract_fit(row, fit, term, exponential=False):
    e, se, p = float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term])
    vals = np.array([e, e-norm.ppf(.975)*se, e+norm.ppf(.975)*se])
    if exponential:
        with np.errstate(over="ignore"):
            vals = np.exp(vals)
    if not np.isfinite(vals).all() or se <= 0 or not (0 <= p <= 1):
        raise ValueError("invalid coefficient inference")
    row.update(estimate=vals[0], standard_error=se, ci95_low=vals[1], ci95_high=vals[2], p_value=p,
               valid_for_inference=True, converged=True, model_status="ok")
    row["interpretation_status"] = "sensitivity_valid" if row["sensitivity"] != "none" else (
        "exploratory_sparse" if row["adjustment_status"] != "full" else "primary_valid")


def _attempt_statistics(fit, term, exponential=False):
    """Retain attempted coefficients for QC without promoting invalid inference."""
    if fit is None or term not in fit.params:
        return {}
    e, se = float(fit.params[term]), float(fit.bse[term])
    values = np.array([e, e-norm.ppf(.975)*se, e+norm.ppf(.975)*se])
    if exponential:
        with np.errstate(over="ignore"):
            values = np.exp(values)
    return {"estimate": values[0], "ci95_low": values[1], "ci95_high": values[2],
            "standard_error": se, "p_value": float(fit.pvalues[term])}


def fit_baseline_essdai_model(task, data, row, attempts):
    term = "x:time"
    mixed_valid = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mixed = MixedLM.from_formula(row["formula"], groups="patient_id", re_formula="1", data=data).fit(
                reml=False, method="lbfgs", disp=False)
        diag = step13.assess_mixedlm_fit(mixed, caught)
        mixed_valid = diag["valid_for_primary_inference"] and step13._fixed_effect_inference_valid(mixed, term)
        attempts.append({**row, **diag, **_attempt_statistics(mixed, term), "model_attempted": "MixedLM", "selected": mixed_valid,
            "model_status": "ok" if mixed_valid else "invalid_inference",
            "singular_fit": diag["singular"], "valid_for_inference": mixed_valid})
        if mixed_valid:
            _extract_fit(row, mixed, term)
            row.update(model_used="MixedLM random intercept", model_attempted="MixedLM")
            return row
        row["warning"] += "; MixedLM rejected: " + diag["invalid_reason"]
        row["mixedlm_singular"] = diag["singular"]
    except (ValueError, np.linalg.LinAlgError, RuntimeError, FloatingPointError) as exc:
        attempts.append({**row, "model_attempted": "MixedLM", "model_status": "model_failed",
                         "warning": str(exc), "selected": False})
        row["warning"] += "; MixedLM rejected: " + str(exc)
    row.update(fallback_used=True, model_attempted="MixedLM; Gaussian GEE")
    return _fit_gee(task, data, row, attempts, Gaussian(), term, Exchangeable())


def _fit_gee(task, data, row, attempts, family, term="x", correlation=None):
    name = "Poisson GEE piecewise exponential" if isinstance(family, Poisson) else "Gaussian GEE exchangeable"
    if row["model_attempted"] == "none":
        row["model_attempted"] = name
    recorded = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kwargs = {"offset": np.log(data.interval_years)} if isinstance(family, Poisson) else {}
            fit = GEE.from_formula(row["formula"], data=data, groups="patient_id", family=family,
                cov_struct=correlation or Exchangeable(), **kwargs).fit(cov_type="robust")
        text = "; ".join(str(w.message) for w in caught)
        valid = step13._fixed_effect_inference_valid(fit, term)
        if "iteration limit" in text.lower() or "overflow" in text.lower():
            valid = False
        attempts.append({**row, **_attempt_statistics(fit, term, isinstance(family, Poisson)),
            "model_attempted": name, "converged": bool(fit.converged),
            "selected": valid, "valid_for_inference": valid, "warning": text,
            "model_status": "ok" if valid else "invalid_inference"})
        recorded = True
        if not valid:
            raise ValueError("GEE non-converged or invalid robust covariance/inference")
        row.update(model_used=name, model_attempted=row["model_attempted"] if row["fallback_used"] and
                   task.kind != "poisson" else name)
        _extract_fit(row, fit, term, isinstance(family, Poisson))
        row["warning"] += "; " + text if text else ""
    except (ValueError, np.linalg.LinAlgError, RuntimeError, FloatingPointError) as exc:
        row.update(model_status="model_failed", interpretation_status="model_failed",
                   valid_for_inference=False, warning=row["warning"] + "; " + str(exc))
        if not recorded:
            attempts.append({**row, "model_attempted": name, "selected": False})
    return row


def build_cox_design_diagnostics(design, data):
    """Aggregate diagnostics only: no keys, row values or clinical dates."""
    matrix = design.to_numpy(dtype=float)
    finite = np.isfinite(matrix).all()
    rank = int(np.linalg.matrix_rank(matrix)) if finite and len(matrix) else 0
    variances = design.var(ddof=1)
    standardized = (design - design.mean()) / design.std(ddof=1).replace(0, np.nan)
    condition = float(np.linalg.cond(matrix)) if finite and len(matrix) else np.nan
    standard_condition = float(np.linalg.cond(standardized)) if len(matrix) and np.isfinite(standardized).all().all() else np.nan
    summaries, contingency, separation = {}, {}, []
    for column in design:
        groups = {}
        for event in (0, 1):
            values = design.loc[data.y.eq(event), column]
            groups[str(event)] = {"n": len(values), "min": values.min(), "max": values.max(),
                "mean": values.mean(), "sd": values.std(ddof=1), "n_unique": values.nunique()}
        summaries[column] = groups
        if design[column].nunique() == 2:
            table = pd.crosstab(design[column], data.y).reindex(columns=[0, 1], fill_value=0)
            contingency[column] = table.to_dict()
            if (table == 0).any().any():
                separation.append(column)
    times = pd.to_numeric(data.time_years, errors="coerce")
    return {"design_columns": json.dumps(design.columns.tolist()), "design_rank": rank,
        "design_parameters": design.shape[1], "design_finite": bool(finite),
        "design_condition_number": condition, "standardized_design_condition_number": standard_condition,
        "design_variances": json.dumps(_json_safe(variances.to_dict()), allow_nan=False),
        "design_correlations": json.dumps(_json_safe(design.corr().to_dict()), allow_nan=False),
        "event_group_summaries": json.dumps(_json_safe(summaries), allow_nan=False),
        "binary_event_contingencies": json.dumps(_json_safe(contingency), allow_nan=False),
        "separation_pattern_columns": ",".join(separation),
        "separation_note": "zero event/censor cells flag possible separation; not proof of Cox time-order separation",
        "n_tied_event_times": int(times.loc[data.y.eq(1)].duplicated(keep=False).sum()),
        "n_nonpositive_times": int(times.le(0).sum()), "n_invalid_event_values": int((~data.y.isin([0, 1])).sum()),
        "n_time_nonfinite": int((~np.isfinite(times)).sum()),
        "n_sex_levels": data.sex.nunique(dropna=True) if "sex" in data else 0}


def fit_baseline_cox_model(task, data, row, attempts):
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    from lifelines.exceptions import ConvergenceError, ConvergenceWarning
    row["model_attempted"] = "Cox proportional hazards"
    diagnostics = {"cox_failure_reason": "none", "cox_exception_type": "",
        "cox_warning_types": "", "cox_warning_messages": "", "cox_raw_coefficients": "",
        "cox_covariance": "", "covariance_min_eigenvalue": np.nan,
        "information_condition_number": np.nan, "hessian_min_information_eigenvalue": np.nan}
    failure_reason = "design_construction_failed"
    try:
        _, design = patsy.dmatrices(row["formula"], data, return_type="dataframe", NA_action="raise")
        design = design.drop(columns="Intercept")
        if not design.index.equals(data.index):
            failure_reason = "row_alignment_mismatch"
            raise ValueError("Cox design must preserve the exact selected rows and order")
        diagnostics.update(build_cox_design_diagnostics(design, data))
        if not diagnostics["design_finite"]:
            failure_reason = "nonfinite_design"
            raise ValueError("Cox design contains nonfinite values")
        if diagnostics["design_rank"] < diagnostics["design_parameters"]:
            failure_reason = "rank_deficient_design"
            raise ValueError("Cox selected design is rank deficient")
        if diagnostics["n_nonpositive_times"] or diagnostics["n_time_nonfinite"] or diagnostics["n_invalid_event_values"]:
            failure_reason = "invalid_time_or_event_coding"
            raise ValueError("Cox requires positive finite observed times and binary events")
        design["time_years"], design["event"] = data.time_years.to_numpy(), data.y.to_numpy()
        failure_reason = "cox_fit_exception"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitter = CoxPHFitter(penalizer=0.)
            try:
                fitter.fit(design, duration_col="time_years", event_col="event")
            finally:
                diagnostics["cox_warning_types"] = ";".join(w.category.__name__ for w in caught)
                diagnostics["cox_warning_messages"] = "; ".join(str(w.message) for w in caught)
        bad = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        covariance = fitter.variance_matrix_.to_numpy(dtype=float)
        diagnostics["cox_raw_coefficients"] = json.dumps(_json_safe(fitter.summary[["coef", "se(coef)", "p"]].to_dict()), allow_nan=False)
        diagnostics["cox_covariance"] = json.dumps(_json_safe(covariance.tolist()), allow_nan=False)
        covariance_valid = np.isfinite(covariance).all() and np.allclose(covariance, covariance.T)
        if covariance_valid:
            minimum = float(np.linalg.eigvalsh(covariance).min())
            diagnostics["covariance_min_eigenvalue"] = minimum
            covariance_valid = minimum > 0
            diagnostics["information_condition_number"] = float(np.linalg.cond(covariance))
        hessian = getattr(fitter, "_hessian_", None)
        if hessian is not None and np.isfinite(np.asarray(hessian)).all():
            diagnostics["hessian_min_information_eigenvalue"] = float(np.linalg.eigvalsh(-hessian).min())
        if bad or not covariance_valid:
            failure_reason = "cox_convergence_warning" if bad else "invalid_covariance"
            raise ValueError(failure_reason)
        summary = fitter.summary.loc["x"]
        values = summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].astype(float)
        if (not np.isfinite(values).all() or not np.isfinite(summary["se(coef)"]) or
                values.iloc[1] <= 0 or not 0 <= values.iloc[3] <= 1 or summary["se(coef)"] <= 0):
            failure_reason = "invalid_coefficient_inference"
            raise ValueError("Cox invalid coefficient inference")
        row.update(model_used="Cox proportional hazards", converged=True, valid_for_inference=True,
            model_status="ok", estimate=values.iloc[0], ci95_low=values.iloc[1],
            ci95_high=values.iloc[2], p_value=values.iloc[3], standard_error=float(summary["se(coef)"]), interpretation_status=
            "sensitivity_valid" if task.sensitivity != "none" else
            "exploratory_sparse" if row["adjustment_status"] != "full" else "primary_valid")
        try:
            ph = proportional_hazard_test(fitter, design, time_transform="rank")
            p = float(ph.summary.loc["x", "p"])
            row.update(ph_assumption_p_value=p,
                       ph_assumption_status="warning" if p < .05 else "pass" if np.isfinite(p) else "not_estimable")
            if p < .05:
                row["interpretation_status"] = "exploratory_ph_warning"
                row["warning"] += "; Predictor proportional-hazards assumption warning"
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            row.update(ph_assumption_status="not_estimable", warning=row["warning"] + "; PH test: " + str(exc),
                       interpretation_status="exploratory_ph_warning")
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError, RuntimeError, FloatingPointError, ConvergenceError, patsy.PatsyError) as exc:
        diagnostics.update(cox_failure_reason=failure_reason, cox_exception_type=type(exc).__name__)
        row.update(model_status="model_failed", interpretation_status="model_failed", converged=False,
            valid_for_inference=False, warning=row["warning"] + "; " + str(exc))
        for column in ("estimate", "standard_error", "ci95_low", "ci95_high", "p_value", "q_value"):
            row[column] = np.nan
    row["cox_failure_reason"] = diagnostics["cox_failure_reason"]
    attempts.append({**row, **diagnostics, "selected": row["valid_for_inference"]})
    return row


def fit_lagged_essdai_model(task, data, row, attempts):
    return _fit_gee(task, data, row, attempts, Gaussian())


def fit_timevarying_event_model(task, data, row, attempts):
    row["fallback_used"] = True
    return _fit_gee(task, data, row, attempts, Poisson(), correlation=Independence())


def fit_task(task, data, metadata, attempts):
    row = result_template(task, metadata)
    if not metadata["supported_for_model"]:
        if task.analysis_type == "cross_domain":
            row["model_status"] = "descriptive_only"
        LOG.info("Skipped %s / %s / %s: %s", task.analysis_type, task.spec.name, task.outcome, row["support_reason"])
        return row
    LOG.info("Attempt %s / %s / %s (%s)", task.analysis_type, task.spec.name, task.outcome, task.sensitivity)
    row = {"trajectory": fit_baseline_essdai_model, "cox": fit_baseline_cox_model,
           "lagged": fit_lagged_essdai_model, "poisson": fit_timevarying_event_model}[task.kind](task, data, row, attempts)
    status = "failed" if not row["valid_for_inference"] else (
        "alternative_representation" if not task.spec.primary_canonical else
        "sensitivity" if task.sensitivity != "none" else "primary")
    row.update(claim_status="sensitivity" if task.sensitivity != "none" else status, inference_tier=status)
    return row


def apply_fdr_by_family(results):
    results = results.copy()
    results["q_value"] = np.nan
    qc = []
    for family, group in results.groupby("fdr_family", sort=True):
        valid = group.valid_for_inference.fillna(False) & group.p_value.between(0, 1)
        # New leave-source and matched comparisons are reported without BH.
        if "sensitivity" in group:
            valid &= ~group.sensitivity.isin([LEAVE_SOURCE_SENSITIVITY,
                "domain_primary_matched_sample", "domain_other_domains_matched_sample"])
        indices = group.index[valid]
        if len(indices):
            results.loc[indices, "q_value"] = multipletests(group.loc[indices, "p_value"], method="fdr_bh")[1]
        qc.append({"fdr_family": family, "n_planned": len(group), "n_valid_tests": len(indices),
                   "n_excluded": len(group)-len(indices), "method": "Benjamini-Hochberg",
                   "support_reason": "valid inference only; unsupported/failed models excluded"})
    results["q_value_family"] = results.q_value
    return results, pd.DataFrame(qc)


def apply_outcome_wide_fdr(results):
    """Independent BH across prespecified canonical primary contrasts only.

    Family q-values are untouched. Baseline/TV/cross-domain and HR/IRR/beta
    cannot pool. Duplicate representation policy precedes inference filtering.
    """
    out = results.copy()
    out["q_value_outcome_wide"] = np.nan
    out["outcome_wide_fdr_reason"] = "sensitivity_excluded"
    groups = ["analysis_type", "outcome", "effect_measure"]
    qc = []
    for key, group in out.groupby(groups, sort=True, dropna=False):
        primary = group.sensitivity.eq("none")
        canonical = group.primary_canonical.eq(True) & group.fdr_eligibility.eq(True)
        candidates = group.loc[primary & canonical].sort_values("predictor", kind="stable")
        duplicate = candidates.duplicated("hypothesis_group_id", keep="first")
        unique_indices = candidates.index[~duplicate]
        duplicate_indices = group.index[primary & ~canonical].union(candidates.index[duplicate])
        out.loc[duplicate_indices, "outcome_wide_fdr_reason"] = "duplicate_or_nonprimary_representation_excluded"
        out.loc[unique_indices, "outcome_wide_fdr_reason"] = "invalid_inference_or_nonfinite_p"
        valid = group.valid_for_inference.eq(True) & group.p_value.between(0, 1) & np.isfinite(group.p_value)
        indices = unique_indices.intersection(group.index[valid], sort=False)
        if len(indices):
            out.loc[indices, "q_value_outcome_wide"] = multipletests(group.loc[indices, "p_value"], method="fdr_bh")[1]
            out.loc[indices, "outcome_wide_fdr_reason"] = "included_canonical_primary"
        qc.append({"analysis_type": key[0], "outcome": key[1], "effect_measure": key[2],
            "n_planned": int(primary.sum()), "n_valid_primary": int((primary & valid).sum()),
            "n_duplicate_excluded": len(duplicate_indices), "n_tests_bh": len(indices),
            "method": "Benjamini-Hochberg", "scope": "analysis_type x outcome x effect_measure; canonical primary only",
            "included_predictor_ids": json.dumps(group.loc[indices, "predictor"].tolist()),
            "rule_version": FDR_RULE_VERSION, "hypothesis_rule_version": HYPOTHESIS_RULE_VERSION})
    return out, pd.DataFrame(qc)


def build_adjustment_selection_qc(results):
    rows = []
    identifiers = ["analysis_type", "predictor", "outcome", "sensitivity", "transform"]
    for _, row in results.iterrows():
        for candidate in json.loads(row.candidate_adjustment_qc):
            rows.append({**row[identifiers].to_dict(), **candidate,
                "selected": candidate["level"] == row.selected_adjustment_level,
                "rule_version": ADJUSTMENT_RULE_VERSION})
    return pd.DataFrame(rows)


def build_cross_domain_support(tasks, n_source, minimum_events):
    rows = []
    for task in tasks:
        if task.analysis_type == "cross_domain":
            _, metadata = prepare_model(task, n_source, minimum_events)
            rows.append({**metadata, "predictor": task.spec.name,
                "target_domain": task.outcome.removeprefix("new_domain_"), "analysis_type": "cross_domain",
                "model_status": "supported" if metadata["supported_for_model"] else "descriptive_only"})
    return pd.DataFrame(rows)


def fit_cross_domain_models(task, data, row, attempts):
    return fit_timevarying_event_model(task, data, row, attempts)


def build_analysis_tasks(df, baseline, intervals, domain_risks, registry):
    # Audit attrs stay on the saved interval product. Avoid deep-copying a large
    # aggregate audit into every candidate cohort and estimator design.
    intervals = intervals.copy(deep=False)
    intervals.attrs = {}
    tasks = []
    cols = ["patient_id", "baseline_essdai"] + ["pred__" + s.name for s in registry]
    cols += ["pred_unit__" + s.name for s in registry if s.family == "laboratory"]
    long = df[["patient_id", "clinical_anchor_date", TIME, ESSDAI]].merge(
        baseline[cols], on="patient_id", validate="many_to_one")
    # Baseline predictor -> strictly subsequent ESSDAI assessments.
    long = long.loc[long[TIME].gt(0)].rename(columns={TIME: "time", ESSDAI: "y"})
    tte = {}
    for origin, outcome in (("low", "essdai_ge5"), ("moderate", "high_activity")):
        risk = step13.build_tte_risk_set(df, origin).drop(columns="baseline_essdai").rename(columns={"event": "y"})
        tte[outcome] = risk.merge(baseline, on="patient_id", validate="one_to_one")
    legacy = build_worsening_sensitivity_risk_set(df, baseline).rename(columns={"event": "y"})
    for spec in registry:
        if spec.primary_role == "adjustment":
            continue
        lab = spec.primary_role == "sensitivity"
        sens = "continuous_laboratory" if lab else "none"
        tasks.append(ModelTask(spec, long, "baseline", "essdai_slope", "trajectory", sens))
        for outcome, frame in tte.items():
            tasks.append(ModelTask(spec, frame, "baseline", outcome, "cox", sens))
        for domain in ["any_new_domain", *DOMAINS]:
            risk = domain_risks.loc[domain_risks.target_domain.eq(domain)].rename(columns={"event": "y"})
            tasks.append(ModelTask(spec, risk, "baseline", "new_domain_" + domain, "cox", sens,
                support_exclusion="predictor equals target domain" if spec.name == "domain_" + domain else ""))
        subsequent = intervals.rename(columns={"to_essdai": "y"})
        unavailable = "time-varying column unavailable" if not spec.time_varying_column else ""
        tasks.append(ModelTask(spec, subsequent, "timevarying", "subsequent_essdai", "lagged", sens,
                               support_exclusion=unavailable))
        risk = intervals.loc[intervals.at_risk_first_ge5.eq(True)].assign(
            y=lambda d: d.event_first_ge5)
        tasks.append(ModelTask(spec, risk, "timevarying", "essdai_ge5", "poisson", sens,
                               support_exclusion=unavailable))
        for domain in ["any_new_domain", *DOMAINS]:
            risk_col = "at_risk_any_new_domain" if domain == "any_new_domain" else "at_risk_new_" + domain
            event_col = "event_any_new_domain" if domain == "any_new_domain" else "event_new_domain_" + domain
            frame = intervals.loc[intervals[risk_col].eq(True)].assign(y=lambda d: d[event_col])
            tasks.append(ModelTask(spec, frame, "timevarying", "new_domain_" + domain, "poisson", sens,
                support_exclusion="predictor equals target domain" if spec.name == "domain_" + domain else unavailable))
        tasks.append(ModelTask(spec, legacy, "baseline", "essdai_legacy_plus5", "cox", "total_plus5"))
        spans = long.groupby("patient_id").time.agg(lambda s: s.max()-s.min())
        spaced = long.loc[long.patient_id.isin(spans.index[spans.ge(.5)])]
        tasks.append(ModelTask(spec, spaced, "baseline", "essdai_slope", "trajectory", "minimum_followup_0.5y"))
        if spec.family == "essdai_domain":
            binary = replace(spec, transform="binary_active")
            for candidate, sensitivity, omit in ((binary, "domain_binary", False),
                                                  (spec, "without_baseline_total", True)):
                tasks.extend([ModelTask(candidate, long, "baseline", "essdai_slope", "trajectory", sensitivity, omit),
                    ModelTask(candidate, tte["essdai_ge5"], "baseline", "essdai_ge5", "cox", sensitivity, omit)])
            tasks.append(ModelTask(binary, subsequent, "timevarying", "subsequent_essdai", "lagged", "domain_binary"))
            domain = spec.name.removeprefix("domain_")
            tasks.append(ModelTask(spec, subsequent, "timevarying", "subsequent_essdai", "lagged",
                "domain_from_other_domains_adjusted", state_covariate="from_essdai_other_domains_" + domain))
            state, future = "from_essdai_other_domains_" + domain, "to_essdai_other_domains_" + domain
            eligible = intervals.loc[intervals[state].notna() & intervals[future].notna()].assign(y=lambda d: d[future])
            leave = ModelTask(spec, eligible, "timevarying", "subsequent_essdai", "lagged",
                LEAVE_SOURCE_SENSITIVITY, state_covariate=state,
                comparison_role="leave_source_domain_outcome_matched_sample")
            # Select once before fitting and freeze the complete cases and level
            # for all three comparisons. Total = other-state + weighted x makes
            # the two state designs equivalent in rank, but not in estimand.
            matched, selected = prepare_model(leave, len(baseline))
            level = selected["adjustment_status"]
            tasks.extend([
                replace(leave, data=matched, fixed_adjustment_level=level),
                ModelTask(spec, matched.assign(y=matched.to_essdai), "timevarying", "subsequent_essdai", "lagged",
                    "domain_primary_matched_sample", fixed_adjustment_level=level, comparison_role="primary_matched_sample"),
                ModelTask(spec, matched.assign(y=matched.to_essdai), "timevarying", "subsequent_essdai", "lagged",
                    "domain_other_domains_matched_sample", state_covariate=state, fixed_adjustment_level=level,
                    comparison_role="other_domains_matched_sample")])
        if spec.family == "laboratory" and spec.time_varying_column:
            stem = spec.time_varying_column.removesuffix("__value")
            days_col = stem + "__days_from_anchor"
            if days_col in df:
                anchor_days = df.set_index(KEYS)[days_col]
                days = [anchor_days.loc[(r.patient_id, r.from_clinical_episode_id)] for r in intervals.itertuples()]
                # Upstream episode-selected anchor-day evidence; no new universal recency window.
                proximal = subsequent.loc[pd.to_numeric(pd.Series(days, index=intervals.index), errors="coerce").eq(0)]
                tasks.append(ModelTask(spec, proximal, "timevarying", "subsequent_essdai", "lagged", "lab_anchor_day"))
    domain_specs = [s for s in registry if s.family == "essdai_domain"]
    for spec in domain_specs:
        for domain in DOMAINS:
            if spec.name == "domain_" + domain:
                continue
            frame = intervals.loc[intervals["at_risk_new_" + domain].eq(True)].assign(
                y=lambda d: d["event_new_domain_" + domain])
            tasks.append(ModelTask(replace(spec, transform="binary_active"), frame, "cross_domain",
                "new_domain_" + domain, "poisson", minimum_exposed_events=MIN_CROSS_EXPOSED_EVENTS))
    return tasks


def build_domain_sensitivity_comparison(results):
    relevant = results.loc[results.predictor_family.eq("essdai_domain") &
                           results.analysis_type.eq("timevarying") & results.outcome.eq("subsequent_essdai")].copy()
    full = relevant.loc[relevant.sensitivity.eq("none")].assign(comparison_role="primary_full_sample")
    matched = relevant.loc[relevant.comparison_role.isin(["primary_matched_sample", "other_domains_matched_sample",
                                                        "leave_source_domain_outcome_matched_sample"])]
    out = pd.concat([full, matched], ignore_index=True)
    for domain, group in matched.groupby("predictor"):
        if group.matched_sample_hash.nunique() != 1 or group.n_complete_cases.nunique() != 1 or group.n_patients.nunique() != 1:
            raise ValueError(f"Domain comparison samples disagree: {domain}")
    return out


def build_attrition_tables(results):
    cols = ["analysis_type", "predictor", "outcome", "sensitivity", "transform",
            "n_source_cohort", "n_with_outcome_support", "n_with_predictor_available",
            "n_complete_cases", "n_patients", "n_observations", "n_intervals", "n_events",
            "adjustment_covariates", "model_status", "support_reason",
            "adjustment_status", "selected_adjustment_level", "selection_reason",
            "adjustment_selection_rule", "candidate_adjustment_qc", "formula"]
    return (results.loc[results.analysis_type.eq("baseline"), cols],
            results.loc[~results.analysis_type.eq("baseline"), cols])


def build_model_qc(results, attempts):
    history = pd.DataFrame(attempts)
    skipped = results.loc[results.model_attempted.eq("none")].assign(selected=False)
    out = pd.concat([history, skipped], ignore_index=True)
    valid = out.valid_for_inference.eq(True) & out.selected.eq(True)
    out["claim_status"] = np.select([
        out.model_attempted.eq("none"), ~valid,
        ~out.primary_canonical.eq(True), ~out.sensitivity.eq("none")],
        ["unsupported", "failed", "alternative_representation", "sensitivity"], default="primary")
    out["inference_tier"] = out.claim_status
    return out


def predictor_redundancy_qc(baseline, registry, min_pair_n=20):
    if min_pair_n < 20:
        raise ValueError("redundancy QC requires min_pair_n >= 20")
    rows = []
    candidates = [s for s in registry if s.primary_role != "adjustment"]
    for pos, a in enumerate(candidates):
        for b in candidates[pos+1:]:
            d = baseline[["pred__"+a.name, "pred__"+b.name]].apply(pd.to_numeric, errors="coerce").dropna()
            binary = a.variable_type == b.variable_type == "binary"
            variation = d.nunique().gt(1).all()
            enough = len(d) >= min_pair_n and variation
            coefficient = d.corr(method="pearson" if binary else "spearman").iloc[0, 1] if len(d) >= 3 and variation else np.nan
            glandular = {a.name, b.name} == {"domain_glandular", "glandular_salivary_gland_swelling_active"}
            active_a = d.iloc[:, 0].gt(0) if a.variable_type == "ordinal" else d.iloc[:, 0]
            active_b = d.iloc[:, 1].gt(0) if b.variable_type == "ordinal" else d.iloc[:, 1]
            contingency = pd.crosstab(active_a, active_b).to_json() if binary or glandular else ""
            rows.append({"predictor_a": a.name, "predictor_b": b.name,
                "method": "phi" if binary else "Spearman", "n_complete_pairs": len(d),
                "correlation": coefficient, "redundancy_flag": bool(enough and pd.notna(coefficient) and abs(coefficient) >= .9),
                "min_pair_n": min_pair_n, "construct_duplicate_prespecified": glandular,
                "concordance": float(active_a.eq(active_b).mean()) if (binary or glandular) and len(d) else np.nan,
                "contingency": contingency, "status": "evaluable" if enough else "not_evaluable",
                "canonical_predictor": "domain_glandular" if glandular else "",
                "policy": "QC only; correlation never changes prespecified canonical hypotheses",
                "support_reason": "insufficient_pair_support" if len(d) < min_pair_n else
                    "insufficient_pair_variation" if not variation else "supported"})
    return pd.DataFrame(rows)


def _plot_forest(data, path, title, xlabel, exponential=False):
    use = data.loc[data.valid_for_inference.fillna(False)].copy()
    if "primary_canonical" in use:
        use = use.loc[use.primary_canonical.eq(True)]
    use = use.loc[np.isfinite(use[["estimate", "ci95_low", "ci95_high"]]).all(axis=1)]
    use = use.sort_values(["predictor_family", "predictor_label"], kind="stable")
    fig, ax = plt.subplots(figsize=(9, max(3, .33*len(use)+1.8)))
    if use.empty:
        ax.text(.5, .5, "No associations met support and inference checks", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    else:
        labels = []
        for y, row in enumerate(use.itertuples()):
            caution = row.interpretation_status.startswith("exploratory")
            color = "#c67a20" if caution else "#206082"
            ax.plot([row.ci95_low, row.ci95_high], [y, y], color=color, lw=1.6)
            ax.scatter([row.estimate], [y], color=color, marker="o" if caution else "s", s=28)
            counts = "N=" + str(row.n_patients)
            if pd.notna(row.n_events):
                counts += ", events=" + str(int(row.n_events))
            elif row.n_intervals:
                counts += ", intervals=" + str(row.n_intervals)
            labels.append(row.predictor_label + " [" + row.adjustment_status + "; " + counts + "]\n" +
                row.model_used + (" (exploratory/caution)" if caution else ""))
        ax.set_yticks(range(len(use)), labels=labels)
        ax.invert_yaxis()
        if exponential:
            ax.set_xscale("log")
        ax.axvline(1 if exponential else 0, color="gray", ls="--", lw=1)
    ax.set(title=title + "\nCanonical primary representations; associations, not causality", xlabel=xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .01, "q_value_family: family BH; q_value_outcome_wide: separate outcome-wide BH. Display uses support and inference checks.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": STEM, "CreationDate": None})
    plt.close(fig)


def plot_baseline_forest(data, directory):
    _plot_forest(data, directory/"14_baseline_risk_factor_forest.pdf",
                 "Baseline characteristics and subsequent ESSDAI ≥5",
                 "Hazard ratio (95% CI)", True)


def plot_slope_forest(data, directory):
    _plot_forest(data, directory/"14_essdai_slope_difference_forest.pdf",
                 "Baseline characteristics and subsequent ESSDAI trajectory",
                 "β difference in annual ESSDAI slope (95% CI)")


def plot_timevarying_forest(data, directory):
    _plot_forest(data, directory/"14_timevarying_progression_forest.pdf",
                 "FROM-episode characteristics and subsequent ESSDAI ≥5",
                 "Incidence rate ratio (95% CI), Poisson GEE", True)


def plot_cross_domain_heatmap(data, directory):
    valid = data.loc[data.valid_for_inference.fillna(False)]
    if len(valid) < 4:  # Prespecified display gate, never an inference gate.
        LOG.info("Cross-domain heatmap omitted: fewer than four valid pairs")
        return
    values = valid.assign(source_domain=lambda d: d.predictor.str.removeprefix("domain_"),
                          target_domain=lambda d: d.outcome.str.removeprefix("new_domain_"),
                          log_irr=lambda d: np.log(d.estimate))
    matrix = values.pivot(index="source_domain", columns="target_domain", values="log_irr").reindex(index=DOMAINS, columns=DOMAINS)
    limit = max(float(np.nanmax(np.abs(matrix))), .1)
    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(DOMAINS)), [DOMAIN_LABELS[d] for d in DOMAINS], rotation=60, ha="right")
    ax.set_yticks(range(len(DOMAINS)), [DOMAIN_LABELS[d] for d in DOMAINS])
    ax.set(title="Secondary cross-domain associations; gray = unsupported", xlabel="Incident target domain", ylabel="FROM predictor domain")
    fig.colorbar(im, ax=ax, label="log(IRR), patient-clustered Poisson GEE")
    fig.tight_layout()
    fig.savefig(directory/"14_cross_domain_heatmap.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def output_directories(root):
    root = Path(root)
    return {"tables": root/"outputs/tables/blockA"/STEM,
            "figures": root/"outputs/figures/blockA"/STEM,
            "qc": root/"outputs/qc/blockA"/STEM,
            "logs": root/"outputs/logs"/STEM,
            "analytic": root/"data/analytic/blockA"/STEM}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    parser.add_argument("--baseline", type=Path, default=common.INTEGRATED_BASELINE_PARQUET)
    parser.add_argument("--context", type=Path, default=None,
                        help="Step 10 context companion (default: beside --integrated, if present)")
    parser.add_argument("--variable-registry", type=Path, default=None,
                        help="Step 10 10_variable_registry.csv (default: beside --integrated)")
    parser.add_argument("--input-origin", choices=("canonical", "synthetic_validation"), default="canonical",
                        help="Synthetic validation runs can never receive real-cohort closure PASS")
    parser.add_argument("--progression", type=Path, default=common.DISEASE_ACTIVITY_PROGRESSION_PARQUET,
                        help="Optional Step 13 episode product; definitions always imported from Step 13")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--minimum-events", type=int, default=MIN_EVENTS_COX)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--explainwhy-audit-only", action="store_true",
                        help="Write isolated explain-why availability/feasibility QC; never fit models")
    parser.add_argument("--explainwhy-inclusive-main", action="store_true",
                        help="Isolated inclusive ESSDAI-index v2 family; preserve historical outputs")
    parser.add_argument("--patterns-v2", type=Path, help="Explicit v2 eligible patient Parquet")
    parser.add_argument("--selected-dates-v2", type=Path, help="Explicit v2 selected-date Parquet")
    parser.add_argument("--eligibility-v2", type=Path, help="Explicit complete v2 eligibility Parquet")
    parser.add_argument("--pattern-manifest-v2", type=Path, help="Explicit v2 producer manifest")
    parser.add_argument("--legacy-endpoints", type=Path, help="Step 13 original patient delta endpoints for bridge QC")
    parser.add_argument("--lab-max-age-days", type=int, nargs="*", default=[],
                        help="Prespecified nonnegative index-lab age windows for sensitivity only; primary is same-date")
    args = parser.parse_args(argv)
    if args.explainwhy_inclusive_main and args.explainwhy_audit_only:
        parser.error("inclusive-main and audit-only are separate entrypoints")
    if any(days < 0 for days in args.lab_max_age_days) or len(set(args.lab_max_age_days)) != len(args.lab_max_age_days):
        parser.error("lab age windows must be distinct nonnegative days")
    if args.minimum_events < MIN_EVENTS_COX:
        parser.error(f"--minimum-events must be >= {MIN_EVENTS_COX}; safeguards cannot be lowered")
    return args


def _write_csv(frame, path):
    if len(frame.columns) == 0:
        frame = pd.DataFrame(columns=["model_status", "support_reason"])
    frame.to_csv(path, index=False)


def treatment_adjustment_decision(sources, variable_registry=None):
    """Use only a registered binary systemic exposure with dated as-of evidence.

    Names alone, raw drugs, and retrospective ever-exposure are insufficient.
    A future or unknown measurement remains missing, never untreated.
    """
    rows, valid_sources = [], {}
    registry = variable_registry if variable_registry is not None else pd.DataFrame()
    for source_name, frame in sources.items():
        candidates = [c for c in frame if any(word in c.lower() for word in ("treat", "therap", "medication", "drug_exposure"))]
        usable = []
        for column in candidates:
            if not {"public_variable", "canonical_concept", "temporal_scope", "producer_script"}.issubset(registry):
                continue
            evidence = registry.loc[registry.public_variable.eq(column) &
                registry.canonical_concept.eq("systemic_treatment_exposure") &
                registry.temporal_scope.isin(["as_of_episode", "baseline"]) & registry.producer_script.fillna("").ne("")]
            if len(evidence) != 1 or "ever" in column.lower():
                continue
            scope = evidence.temporal_scope.iloc[0]
            if scope == "baseline" and source_name != "baseline":
                continue
            date_column = column + "__measurement_date"
            if date_column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if not values.dropna().isin([0, 1]).all() or (frame[column].notna() & values.isna()).any():
                continue
            dates = pd.to_datetime(frame[date_column], errors="coerce")
            known = dates.notna() & dates.le(frame.clinical_anchor_date) & values.notna()
            values = values.where(known).astype(float)
            if values.notna().any():
                usable.append((column, values, scope))
        accepted = len(usable) == 1
        if accepted:
            column, values, scope = usable[0]
            valid_sources[source_name] = values
        rows.append({"decision": "sensitivity_only" if accepted else "not_implemented",
            "source_checked": source_name, "candidate_columns": json.dumps(candidates),
            "temporal_scope": scope if accepted else "no_verified_baseline_or_asof_episode_exposure",
            "coverage": float(values.notna().mean()) if accepted and len(frame) else 0.,
            "n_available": int(values.notna().sum()) if accepted else 0,
            "selected_column": column if accepted else "",
            "reason": "registered binary systemic exposure known at or before canonical episode" if accepted else
                "no unique canonical temporally valid systemic treatment exposure; names/ever-exposure are insufficient",
            "primary_analysis_action": "unchanged",
            "sensitivity_action": "treatment_adjusted_sensitivity_subject_to_existing_support_gates" if accepted else "not_implemented; no ad hoc derivation",
            "protocol_consistency": 'allowed under "as appropriate"; observational limitation documented'})
    decision = pd.DataFrame(rows)
    metadata = {"status": "sensitivity_only" if valid_sources else "not_implemented",
        "reason": "registered as-of exposure available" if valid_sources else "no canonical temporally valid treatment exposure available to Step 14",
        "source_checked": list(sources), "candidate_columns": {r["source_checked"]: json.loads(r["candidate_columns"]) for r in rows},
        "temporal_validity": "measurement dated at/before canonical episode required; future/unknown excluded",
        "decision_rule": "registered binary systemic exposure, verified producer and temporal_scope; no raw derivation; existing model support gates",
        "producer_contracts_checked": ["10_build_integrated_longitudinal_dataset.py", "11_integrated_baseline_characterization.py"],
        "primary_analysis_action": "unchanged", "causal_interpretation": False}
    return decision, metadata, valid_sources


def add_treatment_sensitivities(tasks, df, baseline, exposures):
    if not exposures:
        return tasks
    baseline_values = exposures.get("baseline")
    if "integrated" in exposures:
        known = df[KEYS].assign(treatment_exposure=exposures["integrated"]).set_index(KEYS).treatment_exposure
        baseline_map = baseline[KEYS].merge(known.reset_index(), on=KEYS, how="left", validate="one_to_one")
        baseline_map = baseline_map.set_index("patient_id").treatment_exposure
    elif baseline_values is not None:
        baseline_map = baseline.assign(treatment_exposure=baseline_values).set_index("patient_id").treatment_exposure
    else:
        baseline_map = None
    extras = []
    for task in tasks:
        if task.sensitivity != "none" or not task.spec.primary_canonical or task.analysis_type == "cross_domain":
            continue
        frame = task.data.copy()
        if task.analysis_type == "baseline" and baseline_map is not None:
            frame["treatment_exposure"] = frame.patient_id.map(baseline_map)
        elif task.analysis_type == "timevarying" and "integrated" in exposures:
            keys = pd.MultiIndex.from_frame(frame[["patient_id", "from_clinical_episode_id"]].rename(columns={"from_clinical_episode_id": "clinical_episode_id"}))
            frame["treatment_exposure"] = known.reindex(keys).to_numpy()
        else:
            continue
        extras.append(replace(task, data=frame, sensitivity="treatment_adjusted", extra_covariates=("treatment_exposure",)))
    return tasks + extras


FINGERPRINT_EXCLUDED = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".ipynb_checkpoints", "data", "outputs", "work", "dist", "build", "site-packages",
    "node_modules", "vendor", ".tox", ".nox", ".cache"}
CODE_SUFFIXES = {".py", ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".js", ".ts", ".tsx", ".css", ".html", ".sh", ".ipynb", ".r"}


def scientific_code_path(path):
    parts = Path(path).parts
    if any(part in FINGERPRINT_EXCLUDED or part.startswith((".venv_", ".venv-", "venv_", "env_")) for part in parts):
        return False
    return Path(path).suffix.lower() in CODE_SUFFIXES or Path(path).name.startswith("requirements")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_code_fingerprint(root=ROOT):
    """Prune environments before traversal; hash scientific code/config only."""
    root = Path(root)
    files = {}
    for directory, subdirs, names in os.walk(root, followlinks=False):
        subdirs[:] = sorted(d for d in subdirs if d not in FINGERPRINT_EXCLUDED and
                           not d.startswith((".venv_", ".venv-", "venv_", "env_")) and not (Path(directory)/d).is_symlink())
        for name in sorted(names):
            path = Path(directory)/name
            relative = path.relative_to(root).as_posix()
            if scientific_code_path(relative) and not path.is_symlink():
                files[relative] = sha256_file(path)
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    commit, branch, status = git("rev-parse", "HEAD"), git("branch", "--show-current"), git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    changed, untracked = [], []
    records = iter(status.stdout.split("\0"))
    for record in records:
        if not record:
            continue
        flag, path = record[:2], record[3:]
        paths = [path]
        if "R" in flag or "C" in flag:
            paths.append(next(records, ""))
        for relative in paths:
            if scientific_code_path(relative):
                changed.append(relative)
                if flag == "??":
                    untracked.append(relative)
    diff = git("diff", "HEAD", "--", *sorted(set(changed))) if changed else None
    identified = commit.returncode == status.returncode == 0
    dirty_code = bool(changed)
    dirty_worktree = bool(status.stdout)
    return {"git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "git_worktree_dirty": dirty_worktree, "git_code_dirty": dirty_code,
        "untracked_scientific_code": sorted(set(untracked)), "changed_scientific_code": sorted(set(changed)),
        "git_diff_sha256": hashlib.sha256((diff.stdout if diff else "").encode()).hexdigest(),
        "code_file_sha256": files,
        "code_tree_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "reproducibility_status": "unknown_git_state" if not identified else "dirty_untracked_code" if untracked else
            "dirty_tracked_code" if dirty_code else "code_identified_noncode_changes" if dirty_worktree else "fully_reproducible",
        "fingerprint_exclusions": sorted(FINGERPRINT_EXCLUDED) + [".venv_*", ".venv-*", "venv_*", "env_*"]}


def write_closure_summary(metadata, dirs, results=None, pending=None):
    pending = list(pending or [])
    if metadata.get("dry_run"):
        pending.append("Dry run: inferential models have not been fitted.")
    if metadata.get("git_code_dirty") or metadata.get("reproducibility_status") == "unknown_git_state":
        pending.append("Final run must use an identified committed scientific code state.")
    if metadata.get("input_origin") != "canonical":
        pending.append("Synthetic validation cannot close the real cohort; rerun using canonical inputs.")
    status = "CONDITIONAL" if pending else "PASS"
    metadata["closure_status"] = {"leave_source_domain_outcome": "implemented; support and inference in sensitivity/QC outputs",
        "treatment_adjustment_decision": metadata.get("treatment_adjustment", {}).get("status", "not_audited"),
        "composition_discordance_audit": "source/provenance inconsistency retained; complete concordance required; no upstream fix inferred",
        "reproducibility": metadata.get("reproducibility_status"), "final_decision": status, "pending": pending}
    metrics = metadata.get("primary_snapshot", {})
    lines = ["# Step 14 closure summary", "", f"Commit SHA: `{metadata.get('git_commit')}`",
        f"Input SHA-256: `{json.dumps(metadata.get('input_sha256', {}), sort_keys=True)}`",
        f"Cohort: {metadata.get('n_baseline_patients')} baseline patients; {metadata.get('n_episodes')} episodes.",
        f"Structural QC: {metadata.get('structural_qc_status', 'not_run')}. Temporal leakage: {metadata.get('temporal_leakage_status', 'not_run')}.",
        "Adjustment: prespecified Full > Reduced A > Reduced B > Minimal; no post-fit covariate reselection.",
        f"Disease duration: {metadata.get('disease_duration_decision')}",
        f"Treatment: {metadata.get('treatment_adjustment')}",
        f"Composition (unique FROM episodes): {metadata.get('composition_snapshot')}",
        "Composition policy: recorded total and all domains preserved; provenance inconsistency excluded from sensitivities.",
        "Leave-source policy: subtract source contribution at TO and FROM; shared matched intervals and fixed adjustment level; no primary BH.",
        f"Primary inference / FDR / failed models: {metrics}",
        f"Reproducibility: {metadata.get('reproducibility_status')}; code tree SHA-256: {metadata.get('code_tree_sha256')}"]
    if results is not None:
        biological = results.loc[results.predictor.eq("domain_biological") & results.comparison_role.ne("")]
        for row in biological.itertuples():
            lines.append(f"Biological {row.comparison_role}: n={row.n_patients}, intervals={row.n_intervals}, beta={row.estimate}, SE={row.standard_error}, CI=[{row.ci95_low}, {row.ci95_high}], p={row.p_value}, status={row.model_status}.")
    lines += ["", *["Pending: " + item for item in pending], "", "STEP 14 CLOSURE STATUS: " + status]
    (dirs["qc"]/"14_closure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_analysis_metadata(args, context_path, progression_consumed, dry_run, backup_path=None):
    inputs = {name: getattr(args, name) for name in ("integrated", "baseline", "progression")}
    inputs["context"] = context_path
    inputs["variable_registry"] = getattr(args, "variable_registry", None) or Path(args.integrated).with_name("10_variable_registry.csv")
    hashes = {}
    for name, path in inputs.items():
        if path and Path(path).is_file():
            hashes[name] = sha256_file(path)
        else:
            hashes[name] = None
    return {"inputs": {name: str(path) if path else None for name, path in inputs.items()},
        "input_sha256": hashes, **runtime_code_fingerprint(),
        "input_origin": getattr(args, "input_origin", "canonical"),
        "run_started_utc": datetime.now(timezone.utc).isoformat(), "dry_run": dry_run,
        "output_backup_path": str(backup_path) if backup_path else None,
        "progression_product_consumed": progression_consumed, "random_seed": RANDOM_SEED,
        "rule_versions": {"adjustment": ADJUSTMENT_RULE_VERSION, "composition": COMPOSITION_RULE_VERSION,
            "hypotheses": HYPOTHESIS_RULE_VERSION, "outcome_wide_fdr": FDR_RULE_VERSION,
            "composition_audit": COMPOSITION_AUDIT_RULE_VERSION, "duration": DURATION_RULE_VERSION},
        "configuration": {"minimum_events": args.minimum_events, "dry_run": dry_run,
            "overwrite": args.overwrite, "output_root": str(args.output_root)},
        "adjustment_selection_rule": ADJUSTMENT_SELECTION_RULE,
        "candidate_adjustment_levels": list(ADJUSTMENT_LEVELS),
        "support_gates": {"patients": MIN_PATIENTS_MODEL, "repeated_patients": MIN_REPEATED_PATIENTS,
            "events": args.minimum_events, "exposed": MIN_EXPOSED, "unexposed": MIN_UNEXPOSED,
            "exposed_events": MIN_EXPOSED_EVENTS, "cross_exposed_events": MIN_CROSS_EXPOSED_EVENTS,
            "intervals": MIN_INTERVALS, "events_per_parameter": EVENTS_PER_PARAMETER},
        "baseline_outcomes": "strictly subsequent assessments; trajectory requires two subsequent distinct times",
        "event_convention": EVENT_NOTE, "timevarying_model": TV_NOTE,
        "trajectory_method": "accepted MixedLM random intercept; prespecified Gaussian GEE fallback with all attempts retained",
        "lab_temporal_policy": "upstream episode-selected values dated at/before FROM only; anchor-day sensitivity",
        "context_path": str(context_path), "context_consumed": Path(context_path).exists(),
        "composition_provenance_manifest_present": False,
        "leave_source_domain_outcome_policy": {"from_state": "ESSDAI total minus weighted source domain at FROM",
            "to_outcome": "future ESSDAI total minus weighted source domain at TO",
            "composition_requirement": "complete evaluable concordant panels on both sides; no partial-panel certification contract available",
            "primary_bh_eligible": False, "sensitivity": LEAVE_SOURCE_SENSITIVITY},
        "cross_domain_heatmap_display_gate": 4,
        "fdr_policy": "q_value/q_value_family: historical family BH; q_value_outcome_wide: separate multiplicity sensitivity, never select the more favorable procedure",
        "versions": {"python": sys.version.split()[0], **{name: importlib.metadata.version(name) for name in
            ("pandas", "numpy", "scipy", "patsy", "statsmodels", "lifelines", "matplotlib", "pyarrow")}}}


def load_run_reference(backup):
    if backup is None:
        return {"status": "reference_unavailable", "metadata": {}, "results": pd.DataFrame()}
    qc = backup/"outputs/qc/blockA"/STEM
    tables = backup/"outputs/tables/blockA"/STEM
    metadata_path = qc/"14_analysis_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    path = tables/"14_model_interpretation_summary.csv"
    if not path.exists():
        path = qc/"14_dry_run_feasibility.csv"
    return {"status": "backed_up_reference" if metadata else "reference_metadata_unavailable",
        "metadata": metadata, "results": pd.read_csv(path) if path.exists() else pd.DataFrame(),
        "results_sha256": sha256_file(path) if path.exists() else None}


def build_run_comparison(reference, metadata, results):
    old = reference["metadata"]
    identical = bool(old.get("input_sha256", {}).get("integrated") and old.get("input_sha256", {}).get("baseline")) and all(
        old.get("input_sha256", {}).get(name) == metadata["input_sha256"].get(name)
        for name in ("integrated", "baseline", "context", "progression"))
    rows = []
    for name in ("n_baseline_patients", "n_episodes", "n_unique_complete_panel_discordant", "n_articular_negative_subtractions",
                 "n_supported_primary_trajectories", "n_primary_trajectory_fallbacks", "n_failed_primary_cox",
                 "n_valid_primary", "n_primary_family_q_below_005", "n_primary_outcome_wide_q_below_005",
                 "duration_n_available", "code_tree_sha256", "git_commit", "git_branch", "git_diff_sha256", "reproducibility_status"):
        before, after = old.get(name), metadata.get(name)
        if identical and name in {"n_baseline_patients", "n_episodes"} and before is not None and before != after:
            raise ValueError("Unchanged canonical input hashes but cohort/episode count changed")
        delta = after-before if isinstance(before, (float, int)) and isinstance(after, (float, int)) else np.nan
        rows.append({"scope": "run", "metric": name, "before": before, "after": after, "delta": delta,
            "comparison_status": "same_inputs" if identical else reference["status"]})
    fields = ["n_patients", "n_intervals", "n_events", "n_complete_cases", "supported_for_model", "model_status",
        "selected_adjustment_level", "adjustment_covariates", "model_used", "estimate", "standard_error", "ci95_low",
        "ci95_high", "p_value", "q_value_family", "q_value_outcome_wide", "support_reason"]
    keys = ["analysis_type", "predictor", "outcome", "sensitivity", "transform"]
    previous = reference["results"]
    if not previous.empty:
        if previous.duplicated(keys).any() or results.duplicated(keys).any():
            raise ValueError("Run comparison requires one selected row per planned hypothesis")
        # Outer join preserves unavailable/failed hypotheses and new sensitivities.
        joined = previous.reindex(columns=keys + fields).merge(results.reindex(columns=keys + fields),
            on=keys, how="outer", validate="one_to_one", suffixes=("_before", "_after"), indicator=True)
        for _, pair in joined.iterrows():
            for field in fields:
                if (old.get("dry_run") or metadata["dry_run"]) and field in {"model_status", "model_used", "estimate", "standard_error",
                    "ci95_low", "ci95_high", "p_value", "q_value_family", "q_value_outcome_wide"}:
                    continue
                before, after = pair[field+"_before"], pair[field+"_after"]
                delta = after-before if isinstance(before, (float, int)) and isinstance(after, (float, int)) else np.nan
                rows.append({"scope": "hypothesis", **pair[keys].to_dict(), "metric": field,
                    "before": before, "after": after, "delta": delta,
                    "comparison_status": "prefit_only" if metadata["dry_run"] else str(pair["_merge"])})
    return pd.DataFrame(rows)


def write_run_comparison(reference, metadata, results, qc_directory):
    comparison = build_run_comparison(reference, metadata, results)
    _write_csv(comparison, qc_directory/"14_run_comparison_summary.csv")
    note = ("Reference outputs were not available; no productive scientific before/after inference is claimed."
        if reference["status"] == "reference_unavailable" else
        "Reference outputs were preserved in the automatic backup; inspect all hypothesis-level changes in the CSV.")
    if reference["metadata"].get("git_worktree_dirty") and not reference["metadata"].get("git_diff_sha256"):
        note += " Reference code was dirty without a diff hash; its commit alone does not identify the generating code."
    (qc_directory/"14_run_comparison.md").write_text(
        "# Step 14 run comparison\n\n" + note + "\n\n" +
        f"Current commit: {metadata['git_commit']}; branch: {metadata['git_branch']}; status: {metadata['reproducibility_status']}.\n\n" +
        f"Code tree SHA-256: {metadata['code_tree_sha256']}. Input hashes and configuration are in 14_analysis_metadata.json.\n\n" +
        ("This is a dry-run: only pre-fit counts/support are compared; no fitted N, effects or p/q changes are asserted.\n\n" if metadata["dry_run"] else
         "All planned hypotheses, including unsupported and failed models, are compared separately from estimator attempts.\n\n") +
        "Review sample/event counts, adjustment, estimator, effect, CI, p, family BH and outcome-wide BH separately. "
        "Other-domains adjustment still predicts subsequent total ESSDAI, not exclusively other-domain progression. "
        "Treatment adjustment is not implemented.\n", encoding="utf-8")


def write_output_manifest(dirs, root):
    records = [{"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for directory in dirs.values() for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "14_output_manifest.csv"]
    _write_csv(pd.DataFrame(records), dirs["qc"]/"14_output_manifest.csv")


def main(args=None):
    args = args or parse_args()
    if getattr(args, "explainwhy_inclusive_main", False):
        from src.block_A.explainwhy_inclusive_run import run_inclusive
        return run_inclusive(args, sys.modules[__name__])
    if getattr(args, "explainwhy_audit_only", False):
        from src.block_A.explainwhy_feasibility import run_feasibility_audit
        decisions = run_feasibility_audit(args, sys.modules[__name__])
        print(decisions.to_string(index=False))
        if decisions.loc[decisions.gate.eq("G0"), "status"].eq("BLOCKED").any():
            raise SystemExit(2)
        return
    if args.minimum_events < MIN_EVENTS_COX:
        raise ValueError(f"minimum_events cannot be below {MIN_EVENTS_COX}")
    dirs = output_directories(args.output_root)
    # Protect ALL completed outputs before creating logs or writing any QC.
    existing = [p for directory in dirs.values() if directory.exists() for p in directory.iterdir() if p.is_file()]
    if existing and not args.overwrite:
        raise FileExistsError("Step 14 outputs exist; use --overwrite to replace them")
    backup = None
    if existing:
        backup = Path(args.output_root)/"outputs/backups"/STEM/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        for source in existing:
            destination = backup/source.relative_to(args.output_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    reference = load_run_reference(backup)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(dirs["logs"]/"14_risk_factors_progression.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.setLevel(logging.INFO)
    LOG.addHandler(handler)
    try:
        context_path = getattr(args, "context", None) or Path(args.integrated).with_name("10_integrated_longitudinal_context.parquet")
        metadata = build_analysis_metadata(args, context_path,
            bool(args.progression and Path(args.progression).exists()), args.dry_run, backup)
        (dirs["qc"]/"14_analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOG.info("Input paths integrated=%s baseline=%s progression=%s", args.integrated, args.baseline, args.progression)
        context_path = getattr(args, "context", None) or Path(args.integrated).with_name("10_integrated_longitudinal_context.parquet")
        missing = [str(path) for path in (args.integrated, args.baseline) if not Path(path).is_file()]
        if missing:
            metadata = build_analysis_metadata(args, context_path, False, args.dry_run, backup)
            metadata["missing_canonical_inputs"] = missing
            metadata["treatment_adjustment"] = {"status": "not_audited", "reason": "canonical inputs unavailable"}
            write_closure_summary(metadata, dirs, pending=["Required canonical inputs unavailable: " + "; ".join(missing)])
            (dirs["qc"]/"14_analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            _write_csv(pd.DataFrame([{"check": "canonical_inputs_available", "status": "fail", "detail": "; ".join(missing)}]), dirs["qc"]/"14_structural_qc.csv")
            write_output_manifest(dirs, args.output_root)
            raise FileNotFoundError("Required canonical inputs unavailable: " + "; ".join(missing))
        integrated, baseline = read_table(args.integrated), read_table(args.baseline)
        progression = read_table(args.progression) if args.progression and Path(args.progression).exists() else None
        if progression is None:
            LOG.info("Step 13 product absent; reuse imported canonical Step 13 definitions directly")
        try:
            if Path(context_path).exists():
                LOG.info("Step 10 context metadata: %s", context_path)
                integrated, baseline = attach_episode_context(integrated, baseline, read_table(context_path))
            elif getattr(args, "context", None):
                raise ValueError(f"Explicit Step 10 context path does not exist: {context_path}")
            else:
                LOG.info("No context companion; use embedded metadata, exclude labs with no as-of timing evidence")
            df, baseline = validate_inputs(integrated, baseline, progression)
        except (ValueError, KeyError) as exc:
            _write_csv(pd.DataFrame([{"check": "input_contract", "status": "fail", "detail": str(exc)}]), dirs["qc"]/"14_structural_qc.csv")
            raise
        LOG.info("Input rows=%s patients=%s baseline rows=%s", len(df), df.patient_id.nunique(), len(baseline))
        registry = resolve_predictors(build_predictor_registry(), baseline, df)
        variable_registry_path = getattr(args, "variable_registry", None) or Path(args.integrated).with_name("10_variable_registry.csv")
        if getattr(args, "variable_registry", None) and not variable_registry_path.exists():
            raise ValueError(f"Explicit variable registry does not exist: {variable_registry_path}")
        variable_registry = pd.read_csv(variable_registry_path) if variable_registry_path.exists() else None
        treatment_qc, treatment_metadata, treatment_exposures = treatment_adjustment_decision(
            {"integrated": df, "baseline": baseline}, variable_registry)
        _write_csv(treatment_qc, dirs["qc"]/"14_treatment_adjustment_decision.csv")
        availability = build_predictor_availability(baseline, df, registry)
        _write_csv(availability, dirs["qc"]/"14_predictor_availability.csv")
        LOG.info("Resolved predictors baseline=%s timevarying=%s", sum(s.baseline_column is not None for s in registry), sum(s.time_varying_column is not None for s in registry))
        LOG.info("Missing predictors: %s", availability.loc[~availability.available, ["name", "analysis_type"]].to_dict("records"))
        base = build_baseline_analysis_dataset(baseline, registry)
        intervals = build_lagged_interval_dataset(df, base, registry)
        _write_csv(pd.DataFrame(intervals.attrs["domain_composition_qc"]), dirs["qc"]/"14_domain_composition_qc.csv")
        episode_audit = composition_episode_audit(df, registry, variable_registry)
        _write_csv(episode_audit.loc[episode_audit.composition_category.eq("complete_panel_discordant")],
                   dirs["qc"]/"14_domain_composition_discordant_episodes.csv")
        _write_csv(composition_discordance_summary(episode_audit), dirs["qc"]/"14_domain_composition_discordance_summary.csv")
        discrepancy_distribution = episode_audit.loc[episode_audit.complete_domain_panel & episode_audit.essdai_total.notna()]
        _write_csv(discrepancy_distribution.groupby("total_minus_reconstructed", dropna=False).size().rename("n_episodes").reset_index(),
                   dirs["qc"]/"14_domain_composition_discrepancy_distribution.csv")
        _write_csv(episode_audit.groupby("composition_category", dropna=False).size().rename("n_episodes").reset_index(),
                   dirs["qc"]/"14_domain_composition_audit_summary.csv")
        _write_csv(build_leave_source_domain_qc(intervals), dirs["qc"]/"14_leave_source_domain_outcome_qc.csv")
        domains = build_domain_risk_sets(df, base)
        leakage = run_temporal_leakage_qc(df, base, intervals, domains, registry, hard_fail=False)
        _write_csv(leakage, dirs["qc"]/"14_temporal_leakage_qc.csv")
        if leakage.status.eq("violation").any():
            raise ValueError("Structural temporal leakage detected; inspect 14_temporal_leakage_qc.csv")
        _write_csv(pd.DataFrame([{"check": c, "status": "pass", "n_rows": len(df), "n_patients": len(base)} for c in (
            "unique_patient_episode_keys", "official_baseline_agreement", "monotonic_canonical_dates",
            "nonnegative_followup", "essdai_and_domain_ranges", "progression_key_agreement" if progression is not None else "step13_definitions_imported")]),
            dirs["qc"]/"14_structural_qc.csv")
        domain_qc = []
        for d in ["any_new_domain", *DOMAINS]:
            risk = domains.loc[domains.target_domain.eq(d)]
            domain_qc.append({"target_domain": d, "n_at_risk": len(risk), "n_events": int(risk.event.sum()),
                "n_baseline_inactive": int(base["eligible_any_new_domain" if d == "any_new_domain" else "eligible_new_"+d].sum()),
                "n_without_evaluable_followup": int(base["eligible_any_new_domain" if d == "any_new_domain" else "eligible_new_"+d].sum())-len(risk),
                "censoring_rule": "last evaluable target assessment; any_new requires all baseline-inactive domains evaluable for negative censoring",
                "status": "pass"})
        _write_csv(pd.DataFrame(domain_qc), dirs["qc"]/"14_domain_riskset_qc.csv")
        _write_csv(predictor_redundancy_qc(base, registry), dirs["qc"]/"14_predictor_redundancy_qc.csv")
        tasks = build_analysis_tasks(df, base, intervals, domains, registry)
        tasks = add_treatment_sensitivities(tasks, df, baseline, treatment_exposures)
        prepared = [(task, *prepare_model(task, len(base), args.minimum_events)) for task in tasks]
        availability = build_predictor_availability(baseline, df, registry, intervals, prepared)
        _write_csv(availability, dirs["qc"]/"14_predictor_availability.csv")
        feasibility = pd.DataFrame([{**result_template(task, meta), "n_total_patients": len(base),
            "n_predictor_available": meta["n_with_predictor_available"],
            "pct_predictor_available": 100*meta["n_with_predictor_available"]/len(base) if len(base) else np.nan,
            "n_predictor_positive": meta["n_exposed"], "n_predictor_negative": meta["n_unexposed"]}
            for task, _, meta in prepared])
        support = build_cross_domain_support(tasks, len(base), args.minimum_events)
        _write_csv(build_adjustment_selection_qc(feasibility), dirs["qc"]/"14_adjustment_selection_qc.csv")
        metadata = build_analysis_metadata(args, context_path, progression is not None, args.dry_run, backup)
        metadata["structural_qc_status"] = "pass"
        metadata["temporal_leakage_status"] = "0 violations"
        metadata["treatment_adjustment"] = treatment_metadata
        duration_n = int(base.pred__duration.notna().sum())
        metadata["disease_duration_decision"] = {"duration_source_status": "available" if duration_n else "not_available",
            "duration_n_available": duration_n,
            "duration_exclusion_reason": "" if duration_n else "no_verified_canonical_diagnosis_date",
            "action": "do not derive from follow-up; preserve age/sex when supported"}
        from_keys = intervals[["patient_id", "from_clinical_episode_id"]].rename(columns={"from_clinical_episode_id": "clinical_episode_id"})
        from_audit = episode_audit.merge(from_keys, on=KEYS, how="inner", validate="one_to_one")
        metadata["composition_snapshot"] = composition_snapshot(from_audit)
        metadata["all_episode_composition_snapshot"] = composition_snapshot(episode_audit)
        metadata["composition_discordance_decision"] = {"status": "source_provenance_inconsistency",
            "upstream_fix_applied": False, "reason": "total directly recorded by Pop producer; domains independently recoded by overlap producer; no row-level evidence establishes an upstream bug",
            "action": "preserve inputs; restrict composition sensitivities to complete concordant panels"}
        metadata["selected_adjustment_level_counts"] = feasibility.selected_adjustment_level.value_counts().to_dict()
        metadata["n_baseline_patients"], metadata["n_episodes"] = len(base), len(df)
        metadata["n_unique_complete_panel_discordant"] = int(intervals.from_composition_complete_panel_discordant.sum())
        metadata["n_articular_negative_subtractions"] = int(intervals.from_essdai.lt(intervals.from_domain_weighted_contribution_raw_articular).sum())
        metadata["model_row_semantics"] = "one selected row per planned hypothesis; counts in unsupported/failed rows are pre-fit, not fitted N"
        metadata["attempt_row_semantics"] = "one row per attempted estimator; matched-comparison attempts labeled separately"
        if args.dry_run:
            write_closure_summary(metadata, dirs)
            (dirs["qc"]/"14_analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            _write_csv(feasibility, dirs["qc"]/"14_dry_run_feasibility.csv")
            print(feasibility[["analysis_type", "predictor", "outcome", "sensitivity", "supported_for_model", "support_reason"]].to_string(index=False))
            print(f"Dry run: {len(tasks)} planned associations; {int(feasibility.supported_for_model.sum())} supported. No models fitted.")
            LOG.info("Dry-run validated contracts and feasibility; no inferential outputs written")
            write_run_comparison(reference, metadata, feasibility, dirs["qc"])
            write_output_manifest(dirs, args.output_root)
            return
        attempts = []
        results = pd.DataFrame([fit_task(task, data, meta, attempts) for task, data, meta in prepared])
        results, fdr_qc = apply_fdr_by_family(results)
        results, outcome_fdr_qc = apply_outcome_wide_fdr(results)
        comparison = build_domain_sensitivity_comparison(results)
        _write_csv(comparison, dirs["tables"]/"14_domain_sensitivity_comparison.csv")
        _write_csv(results.loc[results.sensitivity.eq(LEAVE_SOURCE_SENSITIVITY)], dirs["tables"]/"14_leave_source_domain_outcome_models.csv")
        primary = results.loc[results.sensitivity.eq("none")]
        tables = {
            "predictor_registry": pd.DataFrame([{**asdict(s),
                **(duration_source_report(baseline, registry) if s.name == "duration" else {}),
                "requested_baseline_column": original.baseline_column,
                "requested_time_varying_column": original.time_varying_column}
                for s, original in zip(registry, build_predictor_registry())]),
            "predictor_feasibility": feasibility,
            "cross_domain_support": support,
            "sensitivity_models": results.loc[~results.sensitivity.eq("none")],
            "model_interpretation_summary": results}
        for name, analysis, kind in (
            ("baseline_essdai_trajectory_models", "baseline", "essdai_slope"),
            ("baseline_ge5_cox_models", "baseline", "essdai_ge5"),
            ("baseline_high_activity_models", "baseline", "high_activity"),
            ("timevarying_essdai_models", "timevarying", "subsequent_essdai"),
            ("timevarying_ge5_models", "timevarying", "essdai_ge5")):
            tables[name] = primary.loc[primary.analysis_type.eq(analysis) & primary.outcome.eq(kind)]
        for name, analysis in (("baseline_new_domain_models", "baseline"), ("timevarying_new_domain_models", "timevarying")):
            tables[name] = primary.loc[primary.analysis_type.eq(analysis) & primary.outcome.str.startswith("new_domain_")]
        tables["cross_domain_models"] = primary.loc[primary.analysis_type.eq("cross_domain")]
        for name in TABLES:
            _write_csv(tables[name], dirs["tables"]/f"14_{name}.csv")
        baseline_attrition, interval_attrition = build_attrition_tables(results)
        _write_csv(baseline_attrition, dirs["qc"]/"14_baseline_model_attrition.csv")
        _write_csv(interval_attrition, dirs["qc"]/"14_interval_model_attrition.csv")
        _write_csv(build_model_qc(results, attempts), dirs["qc"]/"14_model_qc.csv")
        cox_attempts = [{**row, "cox_hypothesis_id": ":".join(str(row[c]) for c in (
            "analysis_type", "predictor", "outcome", "sensitivity", "transform"))}
            for row in attempts if row["model_attempted"] == "Cox proportional hazards"]
        cox_columns = list(dict.fromkeys(["cox_hypothesis_id", "cox_failure_reason", "cox_exception_type", "cox_warning_types",
            "cox_warning_messages", "attempt_scope", "formula", "selected_adjustment_level", "n_patients", "n_events",
            "model_status", "selected", "valid_for_inference"] + [column for row in cox_attempts for column in row]))
        _write_csv(pd.DataFrame(cox_attempts, columns=cox_columns), dirs["qc"]/"14_cox_failure_diagnostics.csv")
        _write_csv(fdr_qc, dirs["qc"]/"14_multiple_testing_qc.csv")
        _write_csv(outcome_fdr_qc, dirs["qc"]/"14_outcome_wide_fdr_qc.csv")
        for frame, name in ((base, "baseline_predictor_dataset"), (intervals, "lagged_interval_dataset"),
                            (domains, "domain_incidence_risk_set")):
            frame.to_parquet(dirs["analytic"]/f"14_{name}.parquet", index=False)
        # Remove a stale optional heatmap when rerunning with less support.
        heatmap = dirs["figures"]/"14_cross_domain_heatmap.pdf"
        if heatmap.exists():
            heatmap.unlink()
        plot_baseline_forest(tables["baseline_ge5_cox_models"], dirs["figures"])
        plot_slope_forest(tables["baseline_essdai_trajectory_models"], dirs["figures"])
        plot_timevarying_forest(tables["timevarying_ge5_models"], dirs["figures"])
        plot_cross_domain_heatmap(tables["cross_domain_models"], dirs["figures"])
        trajectories = results.loc[results.analysis_type.eq("baseline") & results.outcome.eq("essdai_slope") & results.sensitivity.eq("none")]
        metadata["trajectory_estimators_used"] = trajectories.model_used.value_counts().to_dict()
        metadata["n_supported_primary_trajectories"] = int(trajectories.supported_for_model.sum())
        metadata["n_primary_trajectory_fallbacks"] = int(trajectories.fallback_used.sum())
        canonical_primary = results.loc[results.sensitivity.eq("none") & results.primary_canonical & results.fdr_eligibility]
        metadata["primary_snapshot"] = {"n_valid_primary": int(canonical_primary.valid_for_inference.sum()),
            "n_failed_primary_cox": int((canonical_primary.model_status.eq("failed") & canonical_primary.effect_measure.eq("HR")).sum()),
            "n_primary_family_q_below_005": int(canonical_primary.q_value_family.lt(.05).sum()),
            "n_primary_outcome_wide_q_below_005": int(canonical_primary.q_value_outcome_wide.lt(.05).sum())}
        metadata["leave_source_domain_outcome_snapshot"] = {"n_planned": int(results.sensitivity.eq(LEAVE_SOURCE_SENSITIVITY).sum()),
            "n_valid": int((results.sensitivity.eq(LEAVE_SOURCE_SENSITIVITY) & results.valid_for_inference).sum())}
        write_closure_summary(metadata, dirs, comparison)
        (dirs["qc"]/"14_analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOG.info("Completed: %s valid of %s associations, %s fallbacks", int(results.valid_for_inference.sum()), len(results), int(results.fallback_used.sum()))
        LOG.info("Output paths: %s", {k: str(v) for k, v in dirs.items()})
        write_run_comparison(reference, metadata, results, dirs["qc"])
        write_output_manifest(dirs, args.output_root)
        print("ORDER TO REVIEW PRIMARY OBJECTIVE 3 OUTPUTS\n"
              "1. Structural / temporal leakage QC\n2. Predictor registry and availability\n"
              "3. Predictor feasibility\n4. Baseline ESSDAI trajectory associations\n"
              "5. Baseline progression-to-ESSDAI>=5 associations\n6. Baseline new-domain associations\n"
              "7. Time-varying lagged ESSDAI associations\n8. Time-varying event associations\n"
              "9. Cross-domain support and models\n10. Sensitivity analyses\n"
              "11. Model QC / FDR / interpretation summary")
    finally:
        LOG.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()

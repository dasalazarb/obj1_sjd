#!/usr/bin/env python3
"""Earlier clinical information associated with subsequent SjD progression.

Analysis consumer of Steps 10/11/13; no raw-data phenotyping or imputation.
Event times are first observed positive assessments, not biological onset times.
Time-updated event inference uses piecewise-exponential Poisson GEE: lifelines'
CoxTimeVaryingFitter does not implement reliable patient-clustered robust variance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import importlib.util
import json
import logging
from pathlib import Path
import sys
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
from config import CHANGE_THRESHOLDS  # noqa: E402

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
    "model_qc", "multiple_testing_qc", "predictor_redundancy_qc")


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


def read_table(path: Path) -> pd.DataFrame:
    return step13.read_table(Path(path))


def attach_episode_context(integrated, baseline, context):
    """Join only canonical Step 10 evaluability/lab metadata, never predictors.

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
    allowed = [c for c in context if c.endswith(("_evaluable", "_valid", "__measurement_date",
                                                "__days_from_anchor", "__unit", "__n_measurements"))]
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
        out.append(frame.merge(context[KEYS+additional], on=KEYS, how="left", validate="one_to_one"))
    return tuple(out)


def build_predictor_registry() -> list[PredictorSpec]:
    registry = [PredictorSpec("age", "Age at clinical baseline", "demographic", "continuous",
                    "demo__age_at_baseline", None, primary_role="adjustment"),
                PredictorSpec("sex", "Sex", "demographic", "categorical", "demo__sex",
                    None, primary_role="adjustment"),
                PredictorSpec("duration", "Disease duration (years)", "disease_history",
                    "continuous", "disease_duration", None, primary_role="adjustment")]
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
            notes="ESSDAI total contains this domain; also analyze without baseline total."))
    return registry


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
                        "duration": ["demo__disease_duration", "disease_duration_years"]}[spec.name]
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


def build_predictor_availability(baseline, integrated, registry):
    rows = []
    for spec in registry:
        for mode, frame in ((True, baseline), (False, integrated)):
            values = predictor_values(frame, spec, mode)
            col = spec.baseline_column if mode else spec.time_varying_column
            rows.append({**asdict(spec), "analysis_type": "baseline" if mode else "timevarying",
                "resolved_column": col, "available": col is not None,
                "n_total": len(frame), "n_available": int(values.notna().sum()),
                "n_raw_nonmissing": int(frame[col].notna().sum()) if col else 0,
                "n_temporal_or_evaluability_excluded": int(frame[col].notna().sum() - values.notna().sum()) if col else 0,
                "support_reason": "column_unavailable" if col is None else
                "dated as-of upstream selection only; future/undated/conflicting values excluded"
                if spec.family == "laboratory" else "canonical evaluability respected"})
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
    "at_risk_any_new_domain", "event_any_new_domain"]


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
    return out.merge(baseline[["patient_id", "baseline_essdai", "pred__age", "pred__sex", "pred__duration"]]
                     .rename(columns={"pred__age": "age", "pred__sex": "sex", "pred__duration": "duration"}),
                     on="patient_id", how="left", validate="many_to_one")


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
    for i, row in intervals.iterrows():
        key = (row.patient_id, row.from_clinical_episode_id)
        source = indexed.loc[[key]].reset_index()
        fr = indexed.loc[key]
        record("from_episode_date", [fr.clinical_anchor_date != row.from_date], intervals.loc[[i]])
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
        terms = ["x"] + [c for c in ("from_essdai", "interval_years")
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


def prepare_model(task, n_source_patients, minimum_events=MIN_EVENTS_COX):
    """One auditable complete-case cohort drives support, attrition and fitting.

    Full adjustment: baseline total, age, sex, duration. Prespecified reduction:
    baseline total only (age only in the no-total sensitivity). No stepwise search.
    Zero-variance adjustment fields are explicitly recorded as non-estimable.
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
        d["sex"] = d.sex.astype(object)
    d["y"] = pd.to_numeric(d.get("y", pd.Series(np.nan, index=d.index)), errors="coerce")
    required = ["patient_id", "y", "x"]
    if task.kind == "trajectory":
        required += ["time"]
    elif task.kind == "lagged":
        required += ["from_essdai", "interval_years"]
    elif task.kind == "cox":
        required += ["time_years"]
    elif task.kind == "poisson":
        required += ["interval_years", "start_time"]
        # Fixed bands on FROM time preserve the original canonical intervals.
        d["time_band"] = pd.cut(d.start_time, [-np.inf, 1, 3, np.inf],
                                labels=["<1y", "1-3y", ">=3y"], right=False).astype(object)
        required += ["time_band"]
    n_outcome = d.dropna(subset=[c for c in required if c != "x"]).patient_id.nunique()
    n_predictor = d.loc[d.x.notna()].patient_id.nunique()
    desired = ["baseline_essdai", "age", "sex", "duration"]
    if task.omit_baseline_essdai:
        desired.remove("baseline_essdai")
    if task.kind == "lagged":
        desired = ["age", "sex", "duration"]
    usable = [c for c in desired if c in d and d[c].nunique(dropna=True) > 1]
    missing = [c for c in desired if c not in usable]
    reduced = ["age"] if task.omit_baseline_essdai or task.kind == "lagged" else ["baseline_essdai"]
    reduced = [c for c in reduced if c in usable]
    # Full model support is checked on its own complete-case cohort first.
    full = d.replace([np.inf, -np.inf], np.nan).dropna(subset=list(dict.fromkeys(required+usable)))
    if task.kind == "trajectory":
        ids = full.groupby("patient_id").time.nunique()
        full = full.loc[full.patient_id.isin(ids.index[ids.ge(2)])]
    formula = _design_formula(task, usable, full)
    counts, parameters, reasons = _check_support(full, task, minimum_events, formula)
    full_support_reason = "; ".join(reasons + (["unavailable/constant adjustment: " + ",".join(missing)] if missing else [])) or "supported"
    full_complete_cases, full_parameters = len(full), parameters
    adjustment_status = "full"
    if missing or reasons:
        adjustment_status = "prespecified_reduced"
        full = d.replace([np.inf, -np.inf], np.nan).dropna(subset=list(dict.fromkeys(required+reduced)))
        if task.kind == "trajectory":
            ids = full.groupby("patient_id").time.nunique()
            full = full.loc[full.patient_id.isin(ids.index[ids.ge(2)])]
        usable = reduced
        formula = _design_formula(task, usable, full)
        counts, parameters, reasons = _check_support(full, task, minimum_events, formula)
    # pandas extension strings are converted for patsy/statsmodels compatibility.
    if "sex" in full:
        full["sex"] = full.sex.astype(str)
    if "time_band" in full:
        full["time_band"] = full.time_band.astype(str)
    full["patient_id"] = full.patient_id.astype(str)
    full = full.reset_index(drop=True)
    unit_col = "pred_unit__" + task.spec.name
    units = sorted(map(str, full[unit_col].dropna().unique())) if unit_col in full else []
    if task.spec.family == "laboratory" and len(units) > 1:
        reasons.append("mixed laboratory units; no conversion defined in Step 14")
    metadata = {**counts, "n_source_cohort": n_source_patients,
        "n_with_outcome_support": n_outcome, "n_with_predictor_available": n_predictor,
        "n_complete_cases": len(full), "candidate_parameters": parameters,
        "n_full_complete_cases": full_complete_cases, "full_candidate_parameters": full_parameters,
        "full_model_support_reason": full_support_reason,
        "events_per_candidate_parameter": counts["n_events"]/max(parameters, 1),
        "supported_for_model": not reasons, "support_reason": "; ".join(dict.fromkeys(reasons)) or "supported",
        "adjustment_status": adjustment_status, "adjustment_covariates": ",".join(usable),
        "unavailable_or_constant_adjustment": ",".join(missing), "formula": formula,
        "constant_state_covariates": ",".join(c for c in ("from_essdai", "interval_years")
            if task.kind == "lagged" and c in full and full[c].nunique(dropna=True) <= 1),
        "laboratory_units": ",".join(units) or "not_available",
        "standardization_mean": mean, "standardization_sd": sd,
        "standardization_cohort": "available as-of predictors in outcome-eligible source cohort, before covariate complete-case restriction",
        "structural_relationship": "ESSDAI total includes predictor domain" if task.spec.family == "essdai_domain" else "none"}
    return full, metadata


def result_template(task, meta):
    scale = "per 1 SD higher value" if task.spec.transform == "zscore" else (
        "active vs inactive" if task.spec.transform == "binary_active" else
        "positive vs negative" if task.spec.variable_type == "binary" else "per +1 ordinal level")
    estimand = {"trajectory": "difference in annual ESSDAI slope",
        "lagged": "difference in subsequent ESSDAI conditional on starting ESSDAI",
        "cox": "first-observed-event hazard association", "poisson": "subsequent first-observed-event rate association"}[task.kind]
    return {**meta, "analysis_type": task.analysis_type, "predictor": task.spec.name,
        "predictor_label": task.spec.label, "predictor_family": task.spec.family,
        "predictor_type": task.spec.variable_type, "baseline_column": task.spec.baseline_column,
        "time_varying_column": task.spec.time_varying_column,
        "transform": task.spec.transform, "effect_scale": scale, "outcome": task.outcome,
        "estimand": estimand, "sensitivity": task.sensitivity,
        "model_attempted": "none", "model_used": "none",
        "effect_measure": "HR" if task.kind == "cox" else "IRR" if task.kind == "poisson" else "beta",
        "estimate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
        "p_value": np.nan, "q_value": np.nan, "fdr_family": _family(task),
        "converged": False, "singular_fit": False, "fallback_used": False,
        "ph_assumption_p_value": np.nan, "ph_assumption_status": "not_tested",
        "model_status": "not_estimable", "interpretation_status": "descriptive_only",
        "valid_for_inference": False, "cluster_variable": "patient_id",
        "warning": EVENT_NOTE if task.kind in {"cox", "poisson"} else "Association, not a causal effect.",
        "model_selection_note": TV_NOTE if task.kind == "poisson" else "",
        "treatment_adjustment": "treatment adjustment not implemented in Step 14"}


def _extract_fit(row, fit, term, exponential=False):
    e, se, p = float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term])
    vals = np.array([e, e-norm.ppf(.975)*se, e+norm.ppf(.975)*se])
    if exponential:
        with np.errstate(over="ignore"):
            vals = np.exp(vals)
    if not np.isfinite(vals).all() or se <= 0 or not (0 <= p <= 1):
        raise ValueError("invalid coefficient inference")
    row.update(estimate=vals[0], ci95_low=vals[1], ci95_high=vals[2], p_value=p,
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


def fit_baseline_cox_model(task, data, row, attempts):
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    from lifelines.exceptions import ConvergenceError, ConvergenceWarning
    row["model_attempted"] = "Cox proportional hazards"
    try:
        _, design = patsy.dmatrices(row["formula"], data, return_type="dataframe")
        design = design.drop(columns="Intercept")
        design["time_years"], design["event"] = data.time_years.to_numpy(), data.y.to_numpy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitter = CoxPHFitter(penalizer=0.)
            fitter.fit(design, duration_col="time_years", event_col="event")
        bad = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        if bad or not np.isfinite(fitter.variance_matrix_.to_numpy()).all() or np.linalg.eigvalsh(
                fitter.variance_matrix_.to_numpy()).min() <= 0:
            raise ValueError("Cox convergence warning or invalid covariance")
        summary = fitter.summary.loc["x"]
        values = summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].astype(float)
        if not np.isfinite(values).all() or values.iloc[1] <= 0:
            raise ValueError("Cox invalid coefficient inference")
        row.update(model_used="Cox proportional hazards", converged=True, valid_for_inference=True,
            model_status="ok", estimate=values.iloc[0], ci95_low=values.iloc[1],
            ci95_high=values.iloc[2], p_value=values.iloc[3], interpretation_status=
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
    except (ValueError, np.linalg.LinAlgError, RuntimeError, ConvergenceError) as exc:
        row.update(model_status="model_failed", interpretation_status="model_failed", warning=row["warning"] + "; " + str(exc))
    attempts.append({**row, "selected": row["valid_for_inference"]})
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
    return {"trajectory": fit_baseline_essdai_model, "cox": fit_baseline_cox_model,
            "lagged": fit_lagged_essdai_model, "poisson": fit_timevarying_event_model}[task.kind](task, data, row, attempts)


def apply_fdr_by_family(results):
    results = results.copy()
    results["q_value"] = np.nan
    qc = []
    for family, group in results.groupby("fdr_family", sort=True):
        valid = group.valid_for_inference.fillna(False) & group.p_value.between(0, 1)
        indices = group.index[valid]
        if len(indices):
            results.loc[indices, "q_value"] = multipletests(group.loc[indices, "p_value"], method="fdr_bh")[1]
        qc.append({"fdr_family": family, "n_planned": len(group), "n_valid_tests": len(indices),
                   "n_excluded": len(group)-len(indices), "method": "Benjamini-Hochberg",
                   "support_reason": "valid inference only; unsupported/failed models excluded"})
    return results, pd.DataFrame(qc)


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
                                                  (spec, "without_baseline_essdai", True)):
                tasks.extend([ModelTask(candidate, long, "baseline", "essdai_slope", "trajectory", sensitivity, omit),
                    ModelTask(candidate, tte["essdai_ge5"], "baseline", "essdai_ge5", "cox", sensitivity, omit),
                    ModelTask(candidate, subsequent, "timevarying", "subsequent_essdai", "lagged", sensitivity, omit)])
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


def build_attrition_tables(results):
    cols = ["analysis_type", "predictor", "outcome", "sensitivity", "transform",
            "n_source_cohort", "n_with_outcome_support", "n_with_predictor_available",
            "n_complete_cases", "n_patients", "n_observations", "n_intervals", "n_events",
            "adjustment_covariates", "model_status", "support_reason"]
    return (results.loc[results.analysis_type.eq("baseline"), cols],
            results.loc[~results.analysis_type.eq("baseline"), cols])


def build_model_qc(results, attempts):
    history = pd.DataFrame(attempts)
    skipped = results.loc[results.model_attempted.eq("none")].assign(selected=False)
    return pd.concat([history, skipped], ignore_index=True)


def predictor_redundancy_qc(baseline, registry):
    rows = []
    candidates = [s for s in registry if s.primary_role != "adjustment"]
    for pos, a in enumerate(candidates):
        for b in candidates[pos+1:]:
            d = baseline[["pred__"+a.name, "pred__"+b.name]].apply(pd.to_numeric, errors="coerce").dropna()
            binary = a.variable_type == b.variable_type == "binary"
            enough = len(d) >= 3 and d.nunique().gt(1).all()
            coefficient = d.corr(method="pearson" if binary else "spearman").iloc[0, 1] if enough else np.nan
            rows.append({"predictor_a": a.name, "predictor_b": b.name,
                "method": "phi" if binary else "Spearman", "n_complete_pairs": len(d),
                "correlation": coefficient, "redundancy_flag": abs(coefficient) >= .9 if pd.notna(coefficient) else False,
                "support_reason": "supported" if enough else "insufficient pair variation"})
    return pd.DataFrame(rows)


def _plot_forest(data, path, title, xlabel, exponential=False):
    use = data.loc[data.valid_for_inference.fillna(False)].copy()
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
            labels.append(row.predictor_label + (" (caution)" if caution else ""))
        ax.set_yticks(range(len(use)), labels=labels)
        ax.invert_yaxis()
        if exponential:
            ax.set_xscale("log")
        ax.axvline(1 if exponential else 0, color="gray", ls="--", lw=1)
    ax.set(title=title, xlabel=xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": STEM, "CreationDate": None})
    plt.close(fig)


def plot_baseline_forest(data, directory):
    _plot_forest(data, directory/"14_baseline_risk_factor_forest.pdf",
                 "Baseline characteristics and subsequent ESSDAI ≥5",
                 "Hazard ratio (95% CI)", True)


def plot_slope_forest(data, directory):
    _plot_forest(data, directory/"14_essdai_slope_difference_forest.pdf",
                 "Baseline characteristics and subsequent ESSDAI trajectory",
                 "Difference in annual ESSDAI slope (95% CI)")


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
    parser.add_argument("--progression", type=Path, default=common.DISEASE_ACTIVITY_PROGRESSION_PARQUET,
                        help="Optional Step 13 episode product; definitions always imported from Step 13")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--minimum-events", type=int, default=MIN_EVENTS_COX)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.minimum_events < MIN_EVENTS_COX:
        parser.error(f"--minimum-events must be >= {MIN_EVENTS_COX}; safeguards cannot be lowered")
    return args


def _write_csv(frame, path):
    if len(frame.columns) == 0:
        frame = pd.DataFrame(columns=["model_status", "support_reason"])
    frame.to_csv(path, index=False)


def main(args=None):
    args = args or parse_args()
    if args.minimum_events < MIN_EVENTS_COX:
        raise ValueError(f"minimum_events cannot be below {MIN_EVENTS_COX}")
    dirs = output_directories(args.output_root)
    # Protect ALL completed outputs before creating logs or writing any QC.
    existing = [p for directory in dirs.values() if directory.exists() for p in directory.iterdir() if p.is_file()]
    if existing and not args.overwrite:
        raise FileExistsError("Step 14 outputs exist; use --overwrite to replace them")
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(dirs["logs"]/"14_risk_factors_progression.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.setLevel(logging.INFO)
    LOG.addHandler(handler)
    try:
        LOG.info("Input paths integrated=%s baseline=%s progression=%s", args.integrated, args.baseline, args.progression)
        integrated, baseline = read_table(args.integrated), read_table(args.baseline)
        context_path = getattr(args, "context", None) or Path(args.integrated).with_name("10_integrated_longitudinal_context.parquet")
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
        availability = build_predictor_availability(baseline, df, registry)
        _write_csv(availability, dirs["qc"]/"14_predictor_availability.csv")
        LOG.info("Resolved predictors baseline=%s timevarying=%s", sum(s.baseline_column is not None for s in registry), sum(s.time_varying_column is not None for s in registry))
        LOG.info("Missing predictors: %s", availability.loc[~availability.available, ["name", "analysis_type"]].to_dict("records"))
        base = build_baseline_analysis_dataset(baseline, registry)
        intervals = build_lagged_interval_dataset(df, base, registry)
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
        prepared = [(task, *prepare_model(task, len(base), args.minimum_events)) for task in tasks]
        feasibility = pd.DataFrame([{**result_template(task, meta), "n_total_patients": len(base),
            "n_predictor_available": meta["n_with_predictor_available"],
            "pct_predictor_available": 100*meta["n_with_predictor_available"]/len(base) if len(base) else np.nan,
            "n_predictor_positive": meta["n_exposed"], "n_predictor_negative": meta["n_unexposed"]}
            for task, _, meta in prepared])
        support = build_cross_domain_support(tasks, len(base), args.minimum_events)
        if args.dry_run:
            _write_csv(feasibility, dirs["qc"]/"14_dry_run_feasibility.csv")
            print(feasibility[["analysis_type", "predictor", "outcome", "sensitivity", "supported_for_model", "support_reason"]].to_string(index=False))
            print(f"Dry run: {len(tasks)} planned associations; {int(feasibility.supported_for_model.sum())} supported. No models fitted.")
            LOG.info("Dry-run validated contracts and feasibility; no inferential outputs written")
            return
        attempts = []
        results = pd.DataFrame([fit_task(task, data, meta, attempts) for task, data, meta in prepared])
        results, fdr_qc = apply_fdr_by_family(results)
        primary = results.loc[results.sensitivity.eq("none")]
        tables = {
            "predictor_registry": pd.DataFrame([{**asdict(s),
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
        _write_csv(fdr_qc, dirs["qc"]/"14_multiple_testing_qc.csv")
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
        metadata = {"inputs": {c: str(getattr(args, c)) for c in ("integrated", "baseline", "progression")},
            "progression_product_consumed": progression is not None, "random_seed": RANDOM_SEED,
            "support_gates": {"patients": MIN_PATIENTS_MODEL, "repeated_patients": MIN_REPEATED_PATIENTS,
                "events": args.minimum_events, "exposed": MIN_EXPOSED, "unexposed": MIN_UNEXPOSED,
                "exposed_events": MIN_EXPOSED_EVENTS, "cross_exposed_events": MIN_CROSS_EXPOSED_EVENTS,
                "intervals": MIN_INTERVALS, "events_per_parameter": EVENTS_PER_PARAMETER},
            "baseline_outcomes": "strictly subsequent assessments; trajectory requires two subsequent distinct times",
            "event_convention": EVENT_NOTE, "timevarying_model": TV_NOTE,
            "lab_temporal_policy": "upstream episode-selected values dated at/before FROM only; anchor-day sensitivity",
            "context_path": str(context_path), "context_consumed": Path(context_path).exists(),
            "treatment_adjustment": "treatment adjustment not implemented in Step 14",
            "cross_domain_heatmap_display_gate": 4,
            "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__}}
        (dirs["qc"]/"14_analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOG.info("Completed: %s valid of %s associations, %s fallbacks", int(results.valid_for_inference.sum()), len(results), int(results.fallback_used.sum()))
        LOG.info("Output paths: %s", {k: str(v) for k, v in dirs.items()})
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

#!/usr/bin/env python3
"""Canonical Block A analysis of observed disease-activity progression.

Progression is characterized by continuous trajectories and transitions among
ESSDAI Low, Moderate, and High states. Kaplan--Meier analyses use the date on
which worsening is first observed: the biological transition is interval
censored between the last negative and first positive assessment.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE, OrdinalGEE
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import common  # noqa: E402
from config import (CHANGE_THRESHOLDS, ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS, ESSDAI_HIGH_CUTOFF,
                    ESSDAI_MODERATE_OR_HIGH_CUTOFF, ESSDAI_MAX_VALID,
                    ESSDAI_FOLLOWUP_DURATION_BANDS_YEARS)  # noqa: E402

LOG = logging.getLogger("disease_activity_progression")
STEM = "13_disease_activity_progression"
KEYS = ["patient_id", "clinical_episode_id"]
REQUIRED = KEYS + ["clinical_anchor_date", "clinical_visit_number", "clinical_visit",
    "clinical_baseline_episode_id", "clinical_baseline_date", "is_clinical_baseline",
    "time_since_clinical_baseline_days", "time_since_clinical_baseline_years"]
ESSDAI = "essdai__total"
ESSPRI = {"total": "esspri__total", "dryness": "esspri__dryness",
          "fatigue": "esspri__fatigue", "pain": "esspri__pain"}
DOMAINS = ("constitutional", "lymphadenopathy", "glandular", "articular",
           "cutaneous", "pulmonary", "renal", "muscular", "pns", "cns",
           "hematologic", "biological")
DOMAIN_ALIASES = {
    "constitutional": ("essdai__constitutional_ordinal_score", "eg_constitutional_ordinal_score"),
    "lymphadenopathy": ("essdai__lymphadenopathy_ordinal_score", "eg_lymphadenopathy_ordinal_score"),
    "glandular": ("essdai__glandular_ordinal_score", "eg_glandular_ordinal_score"),
    "articular": ("essdai__articular_ordinal_score", "eg_articular_ordinal_score"),
    "cutaneous": ("essdai__cutaneous_ordinal_score", "eg_cutaneous_ordinal_score"),
    "pulmonary": ("essdai__pulmonary_ordinal_score", "eg_pulmonary_ordinal_score"),
    "renal": ("essdai__renal_ordinal_score", "eg_renal_ordinal_score"),
    "muscular": ("essdai__muscular_ordinal_score", "eg_muscular_ordinal_score"),
    "pns": ("essdai__pns_ordinal_score", "eg_pns_ordinal_score", "eg_neuro_peripheral_ordinal_score"),
    "cns": ("essdai__cns_ordinal_score", "eg_cns_ordinal_score"),
    "hematologic": ("essdai__hematologic_ordinal_score", "eg_hematologic_ordinal_score"),
    "biological": ("essdai__biological_ordinal_score", "eg_biological_ordinal_score"),
}
MIN_REPEATED_PATIENTS = 3
MIN_STATE_CHANGES = 5
SINGULAR_TOL = 1e-8
MIN_KM_EVENTS = 5
MIN_KM_MEDIAN_RISK = 5
TIME = "time_since_clinical_baseline_years"
TIME_WINDOWS = ("baseline", ">0-1y", ">1-2y", ">2-3y", ">3-5y", ">5y")
WINDOW_SELECTION = ("Official clinical baseline; follow-up closest to window center "
                    "(0.5, 1.5, 2.5, 4 years); >5y first eligible; ties by date then episode ID")
DOMAIN_LABELS = {d: d.capitalize() for d in DOMAINS}
DOMAIN_LABELS.update(pns="PNS", cns="CNS")
STATE_ORDER = ("Low", "Moderate", "High")
STATE_RANK = {x: i for i, x in enumerate(STATE_ORDER)}
LANDMARKS = (0, .5, 1, 2, 3, 5)
EVENT_NOTE = ("First-observed-event analysis: state change occurs between the last "
              "non-event visit and the first event-positive visit; conventional KM "
              "uses the latter as observed event time.")


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, low_memory=False)


def classify_essdai_activity(score: object) -> str:
    """Classify an ESSDAI score without conflating activity and Pop states."""
    if pd.isna(score): return "Missing"
    value = float(score)
    if value < ESSDAI_MODERATE_OR_HIGH_CUTOFF: return "Low"
    if value < ESSDAI_HIGH_CUTOFF: return "Moderate"
    return "High"


def classify_essdai_series(values: pd.Series) -> pd.Series:
    return values.map(classify_essdai_activity).astype(pd.CategoricalDtype([*STATE_ORDER, "Missing"], ordered=True))


def baseline_to_last_status(baseline_state: str, last_state: str) -> str:
    if baseline_state not in STATE_RANK or last_state not in STATE_RANK: return "not_evaluable"
    delta = STATE_RANK[last_state] - STATE_RANK[baseline_state]
    return "worsened" if delta > 0 else "improved" if delta < 0 else "stable"


def validate_inputs(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED + [ESSDAI]) - set(df))
    if missing: raise KeyError(f"Integrated dataset lacks required columns: {missing}")
    x = df.copy()
    x["patient_id"] = x.patient_id.astype("string")
    if x[KEYS].isna().any().any() or x.duplicated(KEYS).any():
        raise ValueError("duplicate or missing patient_id + clinical_episode_id")
    for col in ("clinical_anchor_date", "clinical_baseline_date"):
        x[col] = pd.to_datetime(x[col], errors="coerce")
    x["time_since_clinical_baseline_years"] = pd.to_numeric(x.time_since_clinical_baseline_years, errors="coerce")
    x[ESSDAI] = pd.to_numeric(x[ESSDAI], errors="coerce")
    if x.time_since_clinical_baseline_years.dropna().lt(0).any(): raise ValueError("negative time since baseline")
    bases = x.loc[x.is_clinical_baseline.eq(True)]
    if bases.patient_id.duplicated().any(): raise ValueError("multiple clinical baselines per patient")
    if set(bases.patient_id) != set(x.patient_id):
        raise ValueError("one official clinical baseline required per patient")
    if not bases[TIME].eq(0).all(): raise ValueError("official baseline must have time zero")
    if not bases.clinical_episode_id.astype("string").eq(bases.clinical_baseline_episode_id.astype("string")).all():
        raise ValueError("baseline episode inconsistency")
    if not bases.clinical_anchor_date.eq(bases.clinical_baseline_date).all(): raise ValueError("baseline date inconsistency")
    if (~x[ESSDAI].dropna().between(0, ESSDAI_MAX_VALID)).any(): raise ValueError(f"ESSDAI outside allowed range 0-{ESSDAI_MAX_VALID}")
    for col in ESSPRI.values():
        if col in x:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            if (~x[col].dropna().between(0, 10)).any(): raise ValueError(f"{col} outside 0-10")
    baseline_ids = baseline.patient_id.astype("string")
    if baseline_ids.isna().any() or baseline_ids.duplicated().any():
        raise ValueError("canonical baseline must have one row per patient")
    if "is_clinical_baseline" in baseline and not baseline.is_clinical_baseline.eq(True).all():
        raise ValueError("baseline product contains non-baseline episodes")
    absent = set(baseline_ids.dropna()) - set(x.patient_id.dropna())
    if absent: raise ValueError(f"{len(absent)} canonical baseline patients absent from longitudinal input")
    if set(baseline_ids) != set(bases.patient_id):
        raise ValueError("canonical baseline cohort does not match longitudinal baselines")
    if "clinical_episode_id" in baseline:
        canonical = baseline.assign(patient_id=baseline_ids).set_index("patient_id")
        actual = bases.set_index("patient_id")
        if not actual.clinical_episode_id.astype("string").eq(
                canonical.clinical_episode_id.reindex(actual.index).astype("string")).all():
            raise ValueError("Step 11 baseline episode differs from Step 10")
    return x.sort_values(["patient_id", "clinical_anchor_date", "clinical_episode_id"], kind="stable")


def patient_progression(df: pd.DataFrame) -> pd.DataFrame:
    """Patient endpoints anchored only at the official clinical baseline."""
    rows = []
    for pid, g in df.groupby("patient_id", sort=False):
        b = g.loc[g.is_clinical_baseline.eq(True)]
        if len(b) != 1 or pd.isna(b.iloc[0][ESSDAI]): continue
        b = b.iloc[0]; after = g.loc[(g.clinical_anchor_date > b.clinical_anchor_date) & g[ESSDAI].notna()]
        if after.empty: continue
        last = after.iloc[-1]; bs, ls = classify_essdai_activity(b[ESSDAI]), classify_essdai_activity(last[ESSDAI])
        ranks = after[ESSDAI].map(classify_essdai_activity).map(STATE_RANK)
        br = STATE_RANK[bs]
        rows.append({"patient_id": pid, "baseline_essdai": b[ESSDAI], "last_essdai": last[ESSDAI],
            "baseline_state": bs, "last_state": ls, "delta_essdai": last[ESSDAI]-b[ESSDAI],
            "status": baseline_to_last_status(bs, ls),
            "followup_years": (last.clinical_anchor_date-b.clinical_anchor_date).days/365.25,
            "ever_worsened": bool((ranks > br).any()), "ever_improved": bool((ranks < br).any()),
            "maximum_state_reached": STATE_ORDER[int(max([br, *ranks.dropna().tolist()]))],
            "minimum_state_reached": STATE_ORDER[int(min([br, *ranks.dropna().tolist()]))]})
    return pd.DataFrame(rows, columns=["patient_id", "baseline_essdai", "last_essdai",
        "baseline_state", "last_state", "delta_essdai", "status", "followup_years",
        "ever_worsened", "ever_improved", "maximum_state_reached", "minimum_state_reached"])


# Absolute change complements (and never redefines) categorical progression.
DELTA_STATUSES = ("improved", "stable", "worsened")
DELTA_COLORS = {"improved": "#0072B2", "stable": "#999999", "worsened": "#D55E00"}
DELTA_BASE_COLUMNS = ["patient_id", "baseline_clinical_episode_id", "baseline_date",
    "baseline_essdai", "baseline_state", "last_clinical_episode_id", "last_date",
    "last_essdai", "last_state", "followup_years", "n_essdai_followup_assessments",
    "n_distinct_essdai_followup_dates", "delta_essdai_last",
    "delta_essdai_max_from_baseline", "delta_essdai_min_from_baseline"]


def build_patient_absolute_change_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """One row per eligible patient; call validate_inputs against Step 11 first.

    Retain canonical episodes on tied dates. The last valid episode in canonical
    date/episode-ID order supplies the last score; ever uses all valid episodes.
    Same-day baseline companions are never follow-up. Missing scores are skipped.
    """
    columns = list(DELTA_BASE_COLUMNS)
    for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
        columns += [f"last_status_pm{k}", f"last_stable_pm{k}",
            f"still_worsened_at_last_ge{k}", f"not_worsened_at_last_after_ever_ge{k}"]
        for direction in ("worsened", "improved"):
            columns += [f"ever_{direction}_ge{k}", f"last_{direction}_ge{k}"]
        for direction in ("worsening", "improvement"):
            columns.append(f"first_observed_{direction}_ge{k}_date")
    rows = []
    ordered = df.sort_values(["patient_id", "clinical_anchor_date", "clinical_episode_id"], kind="stable")
    for pid, g in ordered.groupby("patient_id", sort=False):
        base = g.loc[g.is_clinical_baseline.eq(True)]
        if len(base) != 1:
            raise ValueError("absolute change requires exactly one official baseline per patient")
        b = base.iloc[0]
        if pd.isna(b[ESSDAI]):
            continue
        after = g.loc[g.clinical_anchor_date.gt(b.clinical_baseline_date) & g[ESSDAI].notna()]
        if after.empty:
            continue
        last = after.iloc[-1]
        delta = after[ESSDAI] - b[ESSDAI]
        row = {"patient_id": pid, "baseline_clinical_episode_id": b.clinical_episode_id,
            "baseline_date": b.clinical_baseline_date, "baseline_essdai": b[ESSDAI],
            "baseline_state": classify_essdai_activity(b[ESSDAI]),
            "last_clinical_episode_id": last.clinical_episode_id,
            "last_date": last.clinical_anchor_date, "last_essdai": last[ESSDAI],
            "last_state": classify_essdai_activity(last[ESSDAI]),
            "followup_years": (last.clinical_anchor_date - b.clinical_baseline_date).days / 365.25,
            "n_essdai_followup_assessments": len(after),
            "n_distinct_essdai_followup_dates": after.clinical_anchor_date.nunique(),
            "delta_essdai_last": delta.iloc[-1], "delta_essdai_max_from_baseline": delta.max(),
            "delta_essdai_min_from_baseline": delta.min()}
        for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
            worse, better = delta.ge(k), delta.le(-k)
            row[f"last_status_pm{k}"] = "worsened" if worse.iloc[-1] else "improved" if better.iloc[-1] else "stable"
            row[f"last_stable_pm{k}"] = not (worse.iloc[-1] or better.iloc[-1])
            row[f"still_worsened_at_last_ge{k}"] = bool(worse.any() and worse.iloc[-1])
            row[f"not_worsened_at_last_after_ever_ge{k}"] = bool(worse.any() and not worse.iloc[-1])
            for direction, date_label, mask in (("worsened", "worsening", worse), ("improved", "improvement", better)):
                row[f"ever_{direction}_ge{k}"] = bool(mask.any())
                row[f"last_{direction}_ge{k}"] = bool(mask.iloc[-1])
                row[f"first_observed_{date_label}_ge{k}_date"] = after.loc[mask, "clinical_anchor_date"].iloc[0] if mask.any() else pd.NaT
        rows.append(row)
    out = pd.DataFrame(rows, columns=columns)
    for col in columns:
        if col.endswith("_date") or col in ("baseline_date", "last_date"):
            out[col] = pd.to_datetime(out[col])
        elif col.startswith(("ever_", "last_worsened", "last_improved", "last_stable", "still_", "not_worsened")):
            out[col] = out[col].astype(bool)
    return out


def absolute_change_strata(patient_delta):
    yield "overall", "Overall", patient_delta
    for state in STATE_ORDER:
        yield "baseline_state", state, patient_delta.loc[patient_delta.baseline_state.eq(state)]


def _delta_proportion(n_event, n):
    """Patient proportion with Wilson 95% bounds; empty strata are not estimable."""
    if not n:
        return {"pct_patients": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    p, z = n_event / n, norm.ppf(.975)
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {"pct_patients": 100*p, "ci95_low": 100*max(0., center-half), "ci95_high": 100*min(1., center+half)}


def summarize_absolute_change_endpoints(patient_delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stype, stratum, g in absolute_change_strata(patient_delta):
        for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
            for assessment, directions in (("baseline_to_last", ("worsening", "improvement", "stable")),
                                           ("ever_observed", ("worsening", "improvement"))):
                for direction in directions:
                    status = {"worsening": "worsened", "improvement": "improved", "stable": "stable"}[direction]
                    mask = g[f"last_status_pm{k}"].eq(status) if assessment == "baseline_to_last" else g[f"ever_{status}_ge{k}"]
                    n_event, n = int(mask.sum()), len(g)
                    rows.append({"stratum_type": stype, "stratum": stratum, "threshold_abs": k,
                        "direction": direction, "assessment": assessment, "n_patients_evaluable": n,
                        "n_patients_event": n_event, **_delta_proportion(n_event, n)})
    return pd.DataFrame(rows)


def reconcile_absolute_change_endpoints(patient_delta):
    rows = []
    for stype, stratum, g in absolute_change_strata(patient_delta):
        for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
            ever, last = g[f"ever_worsened_ge{k}"], g[f"last_worsened_ge{k}"]
            for category, mask in ((f"never_ge{k}", ~ever), (f"ever_ge{k}_not_last", ever & ~last),
                                   (f"ever_ge{k}_and_last", ever & last)):
                n_event = int(mask.sum())
                rows.append({"stratum_type": stype, "stratum": stratum, "threshold_abs": k,
                    "category": category, "n_patients_evaluable": len(g), "n_patients_event": n_event,
                    **_delta_proportion(n_event, len(g))})
    return pd.DataFrame(rows)


def absolute_change_state_crosstab(patient_delta):
    rows = []
    for state in STATE_ORDER:
        g = patient_delta.loc[patient_delta.baseline_state.eq(state)]
        for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
            for last_state in STATE_ORDER:
                for status in DELTA_STATUSES:
                    n = int((g.last_state.eq(last_state) & g[f"last_status_pm{k}"].eq(status)).sum())
                    rows.append({"baseline_state": state, "last_state": last_state, "threshold_abs": k,
                        "last_status": status, "n_patients": n, "n_from_baseline_state": len(g),
                        "pct_within_baseline_state": 100*n/len(g) if len(g) else np.nan})
    return pd.DataFrame(rows)


def summarize_worsening_by_baseline_state(patient_delta):
    rows = []
    for stype, state, g in absolute_change_strata(patient_delta):
        for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
            row = {"stratum_type": stype, "baseline_state": state, "threshold_abs": k, "n_evaluable": len(g)}
            for assessment, direction in (("ever", "worsened"), ("last", "worsened"), ("last", "improved")):
                n = int(g[f"{assessment}_{direction}_ge{k}"].sum())
                row[f"n_{assessment}_{direction}"] = n
                row[f"pct_{assessment}_{direction}"] = _delta_proportion(n, len(g))["pct_patients"]
            row["median_followup_years"] = g.followup_years.median() if len(g) else np.nan
            row["median_n_followup_visits"] = g.n_essdai_followup_assessments.median() if len(g) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def absolute_change_eligibility_qc(df, patient_delta):
    bases = df.loc[df.is_clinical_baseline.eq(True)]
    later = df.loc[df.clinical_anchor_date.gt(df.clinical_baseline_date) & df[ESSDAI].notna()]
    rows = []
    for stype, state, g in absolute_change_strata(patient_delta):
        b = bases if state == "Overall" else bases.loc[bases[ESSDAI].map(classify_essdai_activity).eq(state)]
        obs = later.loc[later.patient_id.isin(b.patient_id)]
        row = {"stratum_type": stype, "stratum": state, "n_baseline_patients": len(b),
            "n_baseline_essdai_available": int(b[ESSDAI].notna().sum()),
            "n_baseline_essdai_missing": int(b[ESSDAI].isna().sum()),
            "n_with_later_essdai": obs.patient_id.nunique(), "n_patients_evaluable": len(g),
            "n_valid_baseline_without_followup": int((b[ESSDAI].notna() & ~b.patient_id.isin(obs.patient_id)).sum()),
            "n_valid_essdai_episodes": int(df.loc[df.patient_id.isin(b.patient_id), ESSDAI].notna().sum()),
            "n_eligible_followup_assessments": int(g.n_essdai_followup_assessments.sum()),
            "n_eligible_distinct_followup_dates": int(g.n_distinct_essdai_followup_dates.sum()),
            "n_same_date_followup_extra_assessments": int((g.n_essdai_followup_assessments - g.n_distinct_essdai_followup_dates).sum()),
            "n_same_baseline_date_nonbaseline_valid_episodes": int((df.patient_id.isin(b.patient_id) & ~df.is_clinical_baseline.eq(True) & df.clinical_anchor_date.eq(df.clinical_baseline_date) & df[ESSDAI].notna()).sum()),
            "status": "pass" if len(g) else "not_estimable",
            "policy": "canonical episodes retained; last valid episode ordered by date then episode ID; only dates > official baseline"}
        for col, label in (("followup_years", "followup_years"), ("n_essdai_followup_assessments", "n_followup_visits")):
            for stat, q in (("min", 0), ("q25", .25), ("median", .5), ("q75", .75), ("max", 1)):
                row[f"{stat}_{label}"] = g[col].quantile(q) if len(g) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def absolute_change_invariants_qc(df, endpoints, legacy, summary, reconciliation, crosstab, compact, hard_fail=True):
    """Aggregate-only QC. The caller saves failures before raising in production."""
    rows = []
    def record(check, count, reason):
        rows.append({"check": check, "status": "fail" if count else "pass", "n_failures": int(count),
                     "reason": reason if count else "", "n_patients_evaluable": len(endpoints)})
    record("unique_patient_ids", endpoints.patient_id.isna().sum() + endpoints.patient_id.duplicated().sum(), "missing or duplicate patient IDs")
    record("baseline_state_partition", abs(len(endpoints) - endpoints.baseline_state.isin(STATE_ORDER).sum()), "unknown baseline activity state")
    expected = build_patient_absolute_change_endpoints(df)
    # This also verifies episode IDs, official dates, strict follow-up, extrema,
    # first documented dates, same-date selection and all endpoint flags.
    record("canonical_endpoint_rows", int(not endpoints.reset_index(drop=True).equals(expected)), "endpoints differ from canonical observed episodes")
    record("positive_followup", (~endpoints.last_date.gt(endpoints.baseline_date)).sum() + endpoints.followup_years.le(0).sum(), "follow-up must be strictly after baseline")
    cols = ["patient_id", "baseline_essdai", "last_essdai", "baseline_state", "last_state", "delta_essdai"]
    a = endpoints.rename(columns={"delta_essdai_last": "delta_essdai"})[cols].sort_values("patient_id").reset_index(drop=True)
    b = legacy[cols].sort_values("patient_id").reset_index(drop=True)
    record("legacy_patient_progression_agreement", int(not a.equals(b)), "patient population, scores or states differ from legacy progression")
    for direction in ("worsened", "improved"):
        for assessment in ("ever", "last"):
            small, large = sorted(ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS)
            record(f"{assessment}_{direction}_nested_thresholds", (endpoints[f"{assessment}_{direction}_ge{large}"] & ~endpoints[f"{assessment}_{direction}_ge{small}"]).sum(), "larger absolute change must imply smaller")
        for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
            record(f"last_implies_ever_{direction}_{k}", (endpoints[f"last_{direction}_ge{k}"] & ~endpoints[f"ever_{direction}_ge{k}"]).sum(), "last event missing from ever event")
    for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
        partition = endpoints[[f"last_worsened_ge{k}", f"last_stable_pm{k}", f"last_improved_ge{k}"]].sum(axis=1)
        record(f"last_partition_{k}", partition.ne(1).sum(), "last categories must be exclusive and exhaustive")
        e, l = endpoints[f"ever_worsened_ge{k}"], endpoints[f"last_worsened_ge{k}"]
        record(f"ever_reconciliation_{k}", abs(int(e.sum()) - int((e & l).sum()) - int((e & ~l).sum())), "ever must equal ever-and-last plus ever-not-last")
    for name, actual, rebuilt in (("threshold_summary", summary, summarize_absolute_change_endpoints(endpoints)),
            ("ever_last_reconciliation", reconciliation, reconcile_absolute_change_endpoints(endpoints)),
            ("state_crosstab", crosstab, absolute_change_state_crosstab(endpoints)),
            ("compact_summary", compact, summarize_worsening_by_baseline_state(endpoints))):
        record(name, int(not actual.equals(rebuilt)), "table counts, denominators or percentages differ from patient endpoints")
    last = summary.loc[summary.assessment.eq("baseline_to_last")]
    partitions = last.groupby(["stratum", "threshold_abs"]).agg(events=("n_patients_event", "sum"), n=("n_patients_evaluable", "first"))
    record("summary_last_partition", partitions.events.ne(partitions.n).sum(), "last category counts do not sum to evaluable patients")
    record("patient_percentage_bounds", ((summary.pct_patients.notna()) & ~summary.pct_patients.between(0, 100)).sum(), "patient percentages outside 0-100")
    totals = crosstab.groupby("threshold_abs").n_patients.sum()
    record("crosstab_patient_totals", totals.ne(len(endpoints)).sum(), "crosstab must count patients once per threshold")
    out = pd.DataFrame(rows)
    if hard_fail and out.status.eq("fail").any():
        raise ValueError("Absolute-change QC failed: " + ", ".join(out.loc[out.status.eq("fail"), "check"]))
    return out


EVER_WORSENING_CATEGORIES = ("never_observed", "ever_not_last", "ever_and_last")
EVER_WORSENING_COLORS = {"never_observed": "#999999", "ever_not_last": "#0072B2",
                         "ever_and_last": "#D55E00"}
EVER_WORSENING_STYLES = {"never_observed": "-", "ever_not_last": "--", "ever_and_last": "-"}


def ever_worsening_categories(endpoints, k):
    """Categories use existing flags only; never infer remission or persistence."""
    ever, last = endpoints[f"ever_worsened_ge{k}"], endpoints[f"last_worsened_ge{k}"]
    return pd.Series(np.select([~ever, ever & ~last], EVER_WORSENING_CATEGORIES[:2],
                               default="ever_and_last"), index=endpoints.index)


def observation_opportunity_bands(endpoints):
    """Assign every evaluable patient once per dimension, retaining date ties."""
    a, b = ESSDAI_FOLLOWUP_DURATION_BANDS_YEARS
    if not (0 < a < b < np.inf):
        raise ValueError("Follow-up band boundaries must be positive, finite and increasing")
    duration_labels = (f"<{a:g}y", f"{a:g}-<{b:g}y", f">={b:g}y")
    duration = pd.to_numeric(endpoints.followup_years, errors="coerce")
    visits = pd.to_numeric(endpoints.n_essdai_followup_assessments, errors="coerce")
    return {
        "followup_duration": pd.cut(duration.where(duration.gt(0)), [0, a, b, np.inf],
                                    labels=duration_labels, right=False),
        "followup_assessment_count": pd.cut(visits.where(visits.ge(1) & visits.mod(1).eq(0)),
                                           [1, 2, 3, np.inf], labels=("1", "2", ">=3"), right=False),
    }


def summarize_observation_opportunity_sensitivity(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Observed proportions conditional on post-baseline follow-up, not risks."""
    rows = []
    for stratifier, bands in observation_opportunity_bands(endpoints).items():
        for stype, state, group in absolute_change_strata(endpoints):
            for band in bands.cat.categories:
                g = group.loc[bands.loc[group.index].eq(band)]
                for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
                    row = {"stratifier": stratifier, "stratum_type": stype,
                           "baseline_state": state, "band": band, "threshold_abs": k,
                           "n_patients_evaluable": len(g)}
                    for assessment in ("ever", "last"):
                        n = int(g[f"{assessment}_worsened_ge{k}"].sum())
                        prop = _delta_proportion(n, len(g))
                        row.update({f"n_{assessment}_worsened": n,
                                    f"pct_{assessment}_worsened": prop["pct_patients"],
                                    f"ci95_low_{assessment}": prop["ci95_low"],
                                    f"ci95_high_{assessment}": prop["ci95_high"]})
                    row["n_ever_not_last"] = int((g[f"ever_worsened_ge{k}"] & ~g[f"last_worsened_ge{k}"]).sum())
                    for source, label in (("followup_years", "followup_years"),
                            ("n_essdai_followup_assessments", "n_followup_assessments"),
                            ("n_distinct_essdai_followup_dates", "n_distinct_followup_dates")):
                        row[f"median_{label}"] = g[source].median() if len(g) else np.nan
                    # Support annotation only, never a suppression or eligibility rule.
                    row["support_note"] = "not_estimable" if not len(g) else "sparse" if len(g) < 10 else "descriptive_only"
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize_baseline_change_opportunity(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Supplement primary denominators with mathematical floor/ceiling support."""
    rows = []
    for _, state, g in absolute_change_strata(endpoints):
        for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
            row = {"baseline_state": state, "threshold_abs": k, "n_evaluable": len(g)}
            for direction, possible in (("improve", g.baseline_essdai.ge(k)),
                    ("worsen", g.baseline_essdai.add(k).le(ESSDAI_MAX_VALID))):
                status = "improved" if direction == "improve" else "worsened"
                event = g[f"last_{status}_ge{k}"]
                n_possible, n_event = int(possible.sum()), int(event.sum())
                n_among = int((event & possible).sum())
                row.update({f"n_can_{direction}_by_k": n_possible,
                            f"n_cannot_{direction}_by_k": len(g) - n_possible,
                            f"n_last_{status}_all": n_event,
                            f"pct_last_{status}_all": _delta_proportion(n_event, len(g))["pct_patients"],
                            f"n_last_{status}_among_possible": n_among,
                            f"pct_last_{status}_among_possible": _delta_proportion(n_among, n_possible)["pct_patients"]})
            rows.append(row)
    return pd.DataFrame(rows)


def absolute_change_refinement_qc(endpoints, sensitivity, opportunity, reconciliation, hard_fail=True):
    """Aggregate diagnostics only: no identifiers, exact dates or row dumps."""
    rows = []
    def record(check, failures, reason):
        rows.append({"check": check, "status": "fail" if failures else "pass",
                     "n_failures": int(failures), "reason": reason if failures else ""})

    for name, actual, rebuilt in (
            ("sensitivity_table", sensitivity, summarize_observation_opportunity_sensitivity(endpoints)),
            ("baseline_opportunity_table", opportunity, summarize_baseline_change_opportunity(endpoints))):
        record(name, int(not actual.equals(rebuilt)), "aggregate table differs from endpoint-derived summary")
    record("followup_assessments_vs_dates",
           (~(endpoints.n_distinct_essdai_followup_dates.ge(1)
              & endpoints.n_distinct_essdai_followup_dates.le(endpoints.n_essdai_followup_assessments)
              & endpoints.n_distinct_essdai_followup_dates.mod(1).eq(0))).sum(),
           "distinct follow-up dates must be positive integers, no greater than assessment counts; tied episodes retained")
    for stratifier, bands in observation_opportunity_bands(endpoints).items():
        record(f"{stratifier}_exhaustive", bands.isna().sum(), "some evaluable patients have no valid band")
        for _, state, g in absolute_change_strata(endpoints):
            table = sensitivity.loc[sensitivity.stratifier.eq(stratifier) & sensitivity.baseline_state.eq(state)]
            totals = table.groupby("threshold_abs")
            record(f"{stratifier}_{state}_denominators", totals.n_patients_evaluable.sum().ne(len(g)).sum(),
                   "bands must partition the evaluable stratum for each threshold")
            for assessment in ("ever", "last"):
                expected = pd.Series({k: int(g[f"{assessment}_worsened_ge{k}"].sum()) for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS})
                record(f"{stratifier}_{state}_{assessment}_totals",
                       totals[f"n_{assessment}_worsened"].sum().reindex(expected.index).ne(expected).sum(),
                       "band event totals disagree with primary endpoints")
            wide = table.pivot(index="band", columns="threshold_abs", values=["n_patients_evaluable", "n_ever_worsened", "n_last_worsened"])
            small, large = sorted(ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS)
            failures = wide["n_patients_evaluable"][small].ne(wide["n_patients_evaluable"][large]).sum()
            for event in ("n_ever_worsened", "n_last_worsened"):
                failures += wide[event][large].gt(wide[event][small]).sum()
            failures += table.n_last_worsened.gt(table.n_ever_worsened).sum()
            record(f"{stratifier}_{state}_nested", failures, "threshold denominators or event nesting disagree within bands")
        overall = sensitivity.loc[sensitivity.stratifier.eq(stratifier) & sensitivity.baseline_state.eq("Overall")]
        states = sensitivity.loc[sensitivity.stratifier.eq(stratifier) & sensitivity.baseline_state.isin(STATE_ORDER)]
        counts = ["n_patients_evaluable", "n_ever_worsened", "n_last_worsened", "n_ever_not_last"]
        expected = overall.set_index(["band", "threshold_abs"])[counts].sort_index()
        actual = states.groupby(["band", "threshold_abs"])[counts].sum().reindex(expected.index)
        record(f"{stratifier}_states_sum_to_overall", actual.ne(expected).to_numpy().sum(),
               "baseline-state counts must sum to Overall in every band")
    for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
        ever, last = endpoints[f"ever_worsened_ge{k}"], endpoints[f"last_worsened_ge{k}"]
        categories = ever_worsening_categories(endpoints, k)
        record(f"trajectory_last_implies_ever_{k}", (last & ~ever).sum(), "last worsening missing from ever flag")
        for stype, state, g in absolute_change_strata(endpoints):
            part = reconciliation.loc[reconciliation.stratum.eq(state) & reconciliation.threshold_abs.eq(k)].set_index("category")
            counts = categories.loc[g.index].value_counts()
            failures = abs(int(counts.sum()) - len(g))
            for category, original in zip(EVER_WORSENING_CATEGORIES, (f"never_ge{k}", f"ever_ge{k}_not_last", f"ever_ge{k}_and_last")):
                failures += abs(int(counts.get(category, 0)) - int(part.loc[original, "n_patients_event"]))
            record(f"trajectory_reconciliation_{state}_{k}", failures, "plot categories must partition and reconcile within each baseline stratum")
        for direction, possible in (("improve", endpoints.baseline_essdai.ge(k)),
                ("worsen", endpoints.baseline_essdai.add(k).le(ESSDAI_MAX_VALID))):
            status = "improved" if direction == "improve" else "worsened"
            for assessment in ("ever", "last"):
                record(f"{assessment}_{direction}_mathematically_possible_{k}",
                       (endpoints[f"{assessment}_{status}_ge{k}"] & ~possible).sum(),
                       "observed change cannot exceed mathematical baseline opportunity")
            part = opportunity.loc[opportunity.threshold_abs.eq(k)]
            failures = (part[f"n_can_{direction}_by_k"] + part[f"n_cannot_{direction}_by_k"]).ne(part.n_evaluable).sum()
            failures += part[f"n_last_{status}_all"].ne(part[f"n_last_{status}_among_possible"]).sum()
            record(f"{direction}_opportunity_partition_{k}", failures, "possibility must partition evaluable patients and preserve all real events")
    out = pd.DataFrame(rows)
    if hard_fail and out.status.eq("fail").any():
        raise ValueError("Absolute-change refinement QC failed; inspect aggregate check names and reasons")
    return out


def plot_ever_worsening_trajectories(df: pd.DataFrame, endpoints: pd.DataFrame, k: int, path: Path) -> None:
    """Plot observed ESSDAI trajectories by ever/last worsening category."""
    fig, axes = plt.subplots(1, len(STATE_ORDER), figsize=(15, 6), sharey=True, squeeze=False)
    ordered = df.sort_values(["patient_id", "clinical_anchor_date", "clinical_episode_id"], kind="stable")
    groups = {pid: g for pid, g in ordered.groupby("patient_id", sort=False)}
    categories = ever_worsening_categories(endpoints, k)
    labels = {"never_observed": f"Never observed ≥{k}", "ever_not_last": f"No longer ≥{k} at last observed visit",
              "ever_and_last": f"Observed ≥{k} ever and at last visit"}
    for ax, state in zip(axes[0], STATE_ORDER):
        subset = endpoints.loc[endpoints.baseline_state.eq(state)].sort_values("patient_id", kind="stable")
        # Background first; event trajectories remain visible in dense Low panels.
        for zorder, category in enumerate(EVER_WORSENING_CATEGORIES, start=2):
            for row in subset.loc[categories.loc[subset.index].eq(category)].itertuples():
                g = groups[row.patient_id]
                g = g.loc[g[ESSDAI].notna() & (g.clinical_episode_id.eq(row.baseline_clinical_episode_id)
                                              | g.clinical_anchor_date.gt(row.baseline_date))]
                time = (g.clinical_anchor_date - row.baseline_date).dt.total_seconds() / (365.25 * 86400)
                alpha = (.15 if state == "Low" else .35) if category == "never_observed" else .8
                ax.plot(time, g[ESSDAI], color=EVER_WORSENING_COLORS[category],
                        linestyle=EVER_WORSENING_STYLES[category], alpha=alpha,
                        linewidth=.8 if category == "never_observed" else 1.1,
                        marker="o", markersize=2.5, zorder=zorder)
        for cutoff in (ESSDAI_MODERATE_OR_HIGH_CUTOFF, ESSDAI_HIGH_CUTOFF):
            ax.axhline(cutoff, color="#555555", alpha=.45, linestyle=":", linewidth=.7, zorder=1)
            ax.text(.98, cutoff, f"ESSDAI {cutoff}", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
        followup = f"Median follow-up {subset.followup_years.median():.2f} years" if len(subset) else "Follow-up not estimable"
        ax.set(title=f"Baseline {state}: n={len(subset)}\n{followup}", xlabel="Years since official clinical baseline")
        counts = categories.loc[subset.index].value_counts()
        ax.text(.5, -.25, "\n".join(f"{labels[c]}: n={counts.get(c, 0)}" for c in EVER_WORSENING_CATEGORIES),
                transform=ax.transAxes, ha="center", va="top", fontsize=8)
        ax.margins(y=.1)
    axes[0, 0].set_ylabel("Observed ESSDAI total (points)")
    handles = [plt.Line2D([0], [0], color=EVER_WORSENING_COLORS[c], linestyle=EVER_WORSENING_STYLES[c],
                         label=labels[c]) for c in EVER_WORSENING_CATEGORIES]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(.5, .08), fontsize=9)
    fig.suptitle(f"Individual ESSDAI trajectories: ever-observed versus last worsening ≥{k}")
    fig.text(.5, .035, "Markers are intermittent observations; connecting lines do not establish continuous or monotonic change.", ha="center", fontsize=9)
    fig.text(.5, .005, "No longer meeting the threshold does not imply remission; ever and last does not establish persistent worsening.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.06, right=.99, bottom=.36, top=.83, wspace=.10)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_absolute_change_summary(summary, path):
    """Render exactly the published baseline-to-last summary rows."""
    fig, axes = plt.subplots(1, len(ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS), figsize=(12, 7), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS):
        data = summary.loc[summary.threshold_abs.eq(k) & summary.assessment.eq("baseline_to_last") & summary.stratum_type.eq("baseline_state")]
        bottom = np.zeros(len(STATE_ORDER))
        for status, direction in (("improved", "improvement"), ("stable", "stable"), ("worsened", "worsening")):
            part = data.loc[data.direction.eq(direction)].set_index("stratum").reindex(STATE_ORDER)
            heights = part.pct_patients.fillna(0).to_numpy()
            ax.bar(STATE_ORDER, heights, bottom=bottom, color=DELTA_COLORS[status], label=status.capitalize())
            for x, (height, n) in enumerate(zip(heights, part.n_patients_event)):
                if height >= 12:
                    ax.text(x, bottom[x]+height/2, f"{height:.1f}%\n(n={n})", ha="center", va="center", color="white" if status != "stable" else "black", fontsize=9)
            bottom += heights
        denoms = data.groupby("stratum").n_patients_evaluable.first()
        for x, state in enumerate(STATE_ORDER):
            n = denoms.loc[state]
            ax.text(x, 103, f"n={n}" if n else "n=0; not estimable", ha="center", fontsize=9)
        ax.set(title=f"Baseline-to-last change: ±{k} points", xlabel="Baseline ESSDAI state", ylim=(0, 113))
        ax.set_yticks([0, 25, 50, 75, 100])
        cells = []
        for direction in ("improvement", "stable", "worsening"):
            part = data.loc[data.direction.eq(direction)].set_index("stratum").reindex(STATE_ORDER)
            cells.append([f"{r.n_patients_event} ({r.pct_patients:.1f}%)" if r.n_patients_evaluable else "0 (NE)" for r in part.itertuples()])
        table = ax.table(cellText=cells, rowLabels=["Improved", "Stable", "Worsened"],
                         colLabels=STATE_ORDER, cellLoc="center", bbox=(.12, -.50, .88, .30))
        table.auto_set_font_size(False)
        table.set_fontsize(8)
    axes[0, 0].set_ylabel("Evaluable patients (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .07))
    fig.suptitle("Absolute ESSDAI change from official baseline")
    fig.text(.5, .025, "Colors describe final absolute change; activity-state transitions are a separate analysis.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.07, right=.98, bottom=.41, top=.88, wspace=.18)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ever_vs_last_worsening(summary, path):
    fig, axes = plt.subplots(1, len(ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS), figsize=(12, 5), sharey=True, squeeze=False)
    strata = ["Overall", *STATE_ORDER]
    x = np.arange(len(strata))
    visible = summary.loc[summary.direction.eq("worsening"), "pct_patients"].dropna()
    peak = visible.max() if len(visible) else 0.
    ymax = min(100., max(10., peak + max(8., .35 * peak)))
    for ax, k in zip(axes[0], ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS):
        data = summary.loc[summary.threshold_abs.eq(k) & summary.direction.eq("worsening")]
        for shift, assessment, label, color in ((-.2, "ever_observed", "Ever observed", "#0072B2"), (.2, "baseline_to_last", "Last assessment", "#D55E00")):
            part = data.loc[data.assessment.eq(assessment)].set_index("stratum").reindex(strata)
            heights = part.pct_patients.fillna(0).to_numpy()
            ax.bar(x+shift, heights, width=.38, label=label, color=color)
            for xx, height, row in zip(x+shift, heights, part.itertuples()):
                if row.n_patients_evaluable:
                    inside = height > .7 * ymax
                    ax.text(xx, height + (-.02 if inside else .02) * ymax,
                            f"{row.pct_patients:.1f}% ({row.n_patients_event}/{row.n_patients_evaluable})",
                            ha="center", va="top" if inside else "bottom", fontsize=8, rotation=90,
                            color="white" if inside else "black")
        ticks = []
        for state in strata:
            n = int(data.loc[data.stratum.eq(state), "n_patients_evaluable"].iloc[0])
            ticks.append(f"{state}\nn={n}" if n else f"{state}\nn=0; NE")
        ax.set(xticks=x, xticklabels=ticks, ylim=(0, ymax), title=f"Worsening ≥{k} points", xlabel="Baseline ESSDAI state")
    axes[0, 0].set_ylabel("Evaluable patients (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, .07))
    fig.suptitle("Observed worsening from official baseline: ever versus last")
    fig.text(.5, .025, "Identical patient denominators; unequal follow-up and visit frequency affect ever-observed proportions.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .18, 1, .93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_individual_absolute_change_trajectories(df, endpoints, k, path):
    fig, axes = plt.subplots(1, len(STATE_ORDER), figsize=(15, 5), sharey=True, squeeze=False)
    ordered = df.sort_values(["patient_id", "clinical_anchor_date", "clinical_episode_id"], kind="stable")
    groups = {pid: g for pid, g in ordered.groupby("patient_id", sort=False)}
    for ax, state in zip(axes[0], STATE_ORDER):
        subset = endpoints.loc[endpoints.baseline_state.eq(state)]
        for row in subset.itertuples():
            g = groups[row.patient_id]
            g = g.loc[g[ESSDAI].notna() & (g.clinical_episode_id.eq(row.baseline_clinical_episode_id) | g.clinical_anchor_date.gt(row.baseline_date))]
            time = (g.clinical_anchor_date - row.baseline_date).dt.total_seconds() / (365.25*86400)
            ax.plot(time, g[ESSDAI], color=DELTA_COLORS[getattr(row, f"last_status_pm{k}")],
                    alpha=.6 if len(subset) < 30 else .15, linewidth=.8, marker="o", markersize=2)
        for cutoff in (ESSDAI_MODERATE_OR_HIGH_CUTOFF, ESSDAI_HIGH_CUTOFF):
            ax.axhline(cutoff, color="#555555", alpha=.45, linestyle="--", linewidth=.7)
            ax.text(.98, cutoff, f"ESSDAI {cutoff}", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
        counts = subset[f"last_status_pm{k}"].value_counts()
        labels = ", ".join(f"{s}: {counts.get(s, 0)}" for s in DELTA_STATUSES)
        followup = f"Median follow-up {subset.followup_years.median():.2f} years" if len(subset) else "Not estimable"
        ax.set(title=f"Baseline {state}: n={len(subset)}\n{followup}", xlabel="Years since official clinical baseline")
        ax.text(.5, -.32, labels, transform=ax.transAxes, ha="center", fontsize=9)
        ax.margins(y=.1)
    axes[0, 0].set_ylabel("Observed ESSDAI total (points)")
    handles = [plt.Line2D([0], [0], color=DELTA_COLORS[s], label=s.capitalize()) for s in DELTA_STATUSES]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(.5, .065))
    fig.suptitle(f"Individual ESSDAI trajectories; final absolute change ±{k} points")
    fig.text(.5, .02, "All evaluable patients shown at observed times. Color indicates final change; paths need not be monotonic.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.06, right=.99, bottom=.32, top=.80, wspace=.08)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_absolute_change_outputs(df, legacy, tables, figures, qc, analytic):
    endpoints = build_patient_absolute_change_endpoints(df)
    summary = summarize_absolute_change_endpoints(endpoints)
    reconciliation = reconcile_absolute_change_endpoints(endpoints)
    crosstab = absolute_change_state_crosstab(endpoints)
    compact = summarize_worsening_by_baseline_state(endpoints)
    absolute_change_eligibility_qc(df, endpoints).to_csv(qc/"13_essdai_delta_eligibility_qc.csv", index=False)
    checks = absolute_change_invariants_qc(df, endpoints, legacy, summary, reconciliation, crosstab, compact, hard_fail=False)
    checks.to_csv(qc/"13_essdai_delta_invariants_qc.csv", index=False)
    sensitivity = summarize_observation_opportunity_sensitivity(endpoints)
    opportunity = summarize_baseline_change_opportunity(endpoints)
    refinement = absolute_change_refinement_qc(endpoints, sensitivity, opportunity, reconciliation, hard_fail=False)
    refinement.to_csv(qc/"13_essdai_refinement_qc.csv", index=False)
    if checks.status.eq("fail").any():
        raise ValueError("Absolute-change integrity failure; inspect 13_essdai_delta_invariants_qc.csv")
    if refinement.status.eq("fail").any():
        raise ValueError("Absolute-change refinement failure; inspect 13_essdai_refinement_qc.csv")
    endpoints.to_parquet(analytic/"13_essdai_patient_delta_endpoints.parquet", index=False)
    for name, data in (("delta_threshold_summary", summary), ("worsening_by_baseline_state", compact),
                       ("ever_vs_last_reconciliation", reconciliation), ("baseline_last_state_by_delta", crosstab),
                       ("observation_opportunity_sensitivity", sensitivity), ("baseline_change_opportunity", opportunity)):
        data.to_csv(tables/f"13_essdai_{name}.csv", index=False)
    plot_absolute_change_summary(summary, figures/"13_essdai_delta_by_baseline_state.pdf")
    plot_ever_vs_last_worsening(summary, figures/"13_essdai_ever_vs_last_worsening.pdf")
    for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
        plot_individual_absolute_change_trajectories(df, endpoints, k, figures/f"13_essdai_individual_trajectories_delta{k}.pdf")
        plot_ever_worsening_trajectories(df, endpoints, k, figures/f"13_essdai_individual_trajectories_ever_worsening_delta{k}.pdf")


def build_state_transitions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use adjacent clinical episodes only; missing middle ESSDAI breaks a chain."""
    intervals=[]
    for pid, g in df.groupby("patient_id", sort=False):
        g=g.sort_values(["clinical_anchor_date", "clinical_episode_id"], kind="stable").reset_index(drop=True)
        for i in range(1, len(g)):
            a,b=g.iloc[i-1],g.iloc[i]
            if pd.isna(a[ESSDAI]) or pd.isna(b[ESSDAI]): continue
            if pd.isna(a.clinical_anchor_date) or pd.isna(b.clinical_anchor_date): continue
            years=(b.clinical_anchor_date-a.clinical_anchor_date).days/365.25
            if years <= 0: continue
            intervals.append({"patient_id":pid,"from_state":classify_essdai_activity(a[ESSDAI]),
                "to_state":classify_essdai_activity(b[ESSDAI]),"interval_years":years})
    raw=pd.DataFrame(intervals)
    rows=[]
    for fr in STATE_ORDER:
        denom=int((raw.from_state.eq(fr)).sum()) if not raw.empty else 0
        for to in STATE_ORDER:
            z=raw.loc[raw.from_state.eq(fr)&raw.to_state.eq(to)] if not raw.empty else raw
            rows.append({"from_state":fr,"to_state":to,"n_intervals":len(z),
                "n_patients":z.patient_id.nunique() if len(z) else 0,"row_total":denom,
                "row_pct":100*len(z)/denom if denom else np.nan,
                "median_interval_years":z.interval_years.median() if len(z) else np.nan})
    return pd.DataFrame(rows),raw


def build_tte_risk_set(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    """Keep observed-event times and the preceding non-event bound."""
    if origin not in {"low", "moderate"}:
        raise ValueError("origin must be low or moderate")
    rows = []
    columns = ["patient_id", "baseline_essdai", "event", "time_years",
               "event_or_censor_date", "last_non_event_time_years",
               "event_time_years", "event_score", "analysis"]
    for pid, g in df.groupby("patient_id", sort=False):
        b = g.loc[g.is_clinical_baseline.eq(True)]
        if len(b) != 1 or pd.isna(b.iloc[0][ESSDAI]):
            continue
        b = b.iloc[0]
        score = float(b[ESSDAI])
        eligible = score < 5 if origin == "low" else 5 <= score < 14
        if not eligible:
            continue
        after = g.loc[(g.clinical_anchor_date > b.clinical_anchor_date) & g[ESSDAI].notna()]
        after = after.sort_values(["clinical_anchor_date", "clinical_episode_id"], kind="stable")
        if after.empty:
            continue
        positive = after[ESSDAI].ge(5 if origin == "low" else 14)
        event = bool(positive.any())
        endpoint = after.loc[positive].iloc[0] if event else after.iloc[-1]
        prior = after.loc[(after.clinical_anchor_date < endpoint.clinical_anchor_date)
                          & ~positive]
        left = (prior.iloc[-1].clinical_anchor_date - b.clinical_anchor_date).days/365.25 if len(prior) else 0.
        time = (endpoint.clinical_anchor_date - b.clinical_anchor_date).days/365.25
        rows.append({"patient_id": pid, "baseline_essdai": score, "event": event,
                     "time_years": time, "event_or_censor_date": endpoint.clinical_anchor_date,
                     "last_non_event_time_years": left if event else time,
                     "event_time_years": time if event else np.nan,
                     "event_score": endpoint[ESSDAI] if event else np.nan,
                     "analysis": "first-observed-event"})
    out = pd.DataFrame(rows, columns=columns)
    if not out.empty and (out.time_years <= 0).any():
        raise AssertionError("KM event/censor time must be >0")
    return out


def km_estimate(tte: pd.DataFrame, landmarks: Iterable[float] = (1, 2, 5)):
    rows = []
    survival = 1.
    for t in sorted(tte.time_years.unique()) if len(tte) else []:
        risk = int(tte.time_years.ge(t).sum())
        events = int((tte.time_years.eq(t) & tte.event).sum())
        if events:
            survival *= 1 - events/risk
        rows.append({"time": t, "n_at_risk": risk, "n_events": events, "survival": survival})
    curve = pd.DataFrame(rows, columns=["time", "n_at_risk", "n_events", "survival"])
    stats = {}
    for y in landmarks:
        prior = curve.loc[curve.time.le(y)]
        # Do not extrapolate beyond the last observation.
        available = len(tte) and y <= tte.time_years.max()
        stats[f"event_free_{y}y"] = (float(prior.iloc[-1].survival) if len(prior) else 1.) if available else np.nan
        stats[f"n_at_risk_{y}y"] = int(tte.time_years.ge(y).sum()) if len(tte) else 0
    reached = curve.loc[curve.survival.le(.5)]
    events = int(tte.event.sum()) if len(tte) else 0
    reliable = (events >= MIN_KM_EVENTS and len(reached)
                and reached.iloc[0].n_at_risk >= MIN_KM_MEDIAN_RISK)
    stats["median_time_to_event"] = float(reached.iloc[0].time) if reliable else "NR"
    stats["median_time_to_event_status"] = ("estimable" if reliable else "not_reliably_estimable"
        if events < MIN_KM_EVENTS or len(reached) else "not_reached")
    stats["interpretation_status"] = ("descriptive_sparse_events" if events < MIN_KM_EVENTS
                                      else "descriptive_first_observed_event")
    return curve, stats


def tte_summary(tte, analysis):
    _, stats = km_estimate(tte)
    events = int(tte.event.sum()) if len(tte) else 0
    person_years = float(tte.time_years.sum()) if len(tte) else 0.
    return {"analysis": analysis, "n_at_risk": len(tte), "n_events": events,
            "n_censored": len(tte) - events,
            "observed_event_proportion": events/len(tte) if len(tte) else np.nan,
            "total_person_years": person_years,
            "event_rate_per_100_person_years": events/person_years*100 if person_years else np.nan,
            "median_followup": tte.time_years.median() if len(tte) else np.nan,
            **stats, "interval_censored_sensitivity": "not executed; first-observed-event KM is primary",
            "note": EVENT_NOTE + (f" Only {events} observed event(s); estimates are descriptive and unstable."
                                    if events < MIN_KM_EVENTS else "")}


def _warning_text(captured):
    return "; ".join(str(getattr(w, "message", w)) for w in captured)


def assess_mixedlm_fit(fit, captured_warnings) -> dict:
    """Convergence alone is insufficient for primary mixed-model inference."""
    text = _warning_text(captured_warnings)
    lower = text.lower()
    converged = bool(getattr(fit, "converged", False))
    cov = np.asarray(fit.cov_re, dtype=float)
    finite = bool(cov.ndim == 2 and cov.size and np.isfinite(cov).all())
    minimum = float(np.linalg.eigvalsh(cov).min()) if finite else np.nan
    variance = float(cov[0, 0]) if finite else np.nan
    singular = bool("singular" in lower or (finite and minimum <= SINGULAR_TOL))
    boundary = bool("boundary" in lower or (finite and variance <= SINGULAR_TOL))
    hessian = "hessian" in lower and "not positive definite" in lower
    reasons = [name for name, bad in (
        ("not_converged", not converged), ("singular", singular),
        ("boundary", boundary), ("hessian_not_positive_definite", hessian),
        ("nonfinite_random_covariance", not finite)) if bad]
    return {"converged": converged, "singular": singular, "boundary": boundary,
            "hessian_warning": hessian, "covariance_finite": finite,
            "minimum_random_covariance_eigenvalue": minimum,
            "random_effect_variance": variance,
            "random_intercept_variance": variance,
            "valid_for_primary_inference": not reasons,
            "warning_text": text, "invalid_reason": "; ".join(reasons)}


def _fixed_effect_inference_valid(fit, key="time"):
    """Require finite estimates and a usable covariance, including robust GEE."""
    if fit is None or not bool(getattr(fit, "converged", False)):
        return False
    params = getattr(fit, "fe_params", fit.params)
    names = list(params.index)
    cov = np.asarray(fit.cov_params().loc[names, names], dtype=float)
    return bool(key in names and np.isfinite(params).all()
                and np.isfinite(cov).all()
                and np.linalg.eigvalsh(cov).min() >= -SINGULAR_TOL
                and np.isfinite(fit.bse[key]) and fit.bse[key] > 0
                and np.isfinite(fit.pvalues[key])
                and 0 <= fit.pvalues[key] <= 1)


def continuous_model_data(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Exact cohort used for fits, attrition and population prediction plots."""
    if outcome not in data:
        return pd.DataFrame(columns=["patient_id", "clinical_anchor_date", "time", "y"])
    cols = ["patient_id", "clinical_anchor_date", TIME, outcome]
    d = data[cols].copy().rename(columns={outcome: "y", TIME: "time"})
    d["y"] = pd.to_numeric(d.y, errors="coerce")
    d["time"] = pd.to_numeric(d.time, errors="coerce")
    d["clinical_anchor_date"] = pd.to_datetime(d.clinical_anchor_date, errors="coerce")
    d = d.dropna()
    d = d.loc[np.isfinite(d.y) & np.isfinite(d.time) & d.time.ge(0)]
    official = data.loc[data.is_clinical_baseline.eq(True), "patient_id"]
    dates = d.groupby("patient_id").clinical_anchor_date.nunique()
    times = d.groupby("patient_id").time.nunique()
    eligible = dates.index[dates.ge(2) & times.ge(2)]
    return d.loc[d.patient_id.isin(eligible) & d.patient_id.isin(official)]


def _model_attempt(fit, label, analysis, d, diagnostics, valid, reason=""):
    e = se = p = lo = hi = np.nan
    if fit is not None:
        e = float(fit.params.get("time", np.nan))
        se = float(fit.bse.get("time", np.nan))
        p = float(fit.pvalues.get("time", np.nan))
        lo, hi = e - norm.ppf(.975) * se, e + norm.ppf(.975) * se
    return {"outcome": label, "analysis": analysis, "attempted_model": analysis,
            "n_patients": d.patient_id.nunique(), "n_observations": len(d),
            "n_unique_times": d.time.nunique(), "estimate": e,
            "estimate_per_year": e, "ci95_low": lo, "ci95_high": hi,
            "p_value": p, **diagnostics,
            "valid_for_primary_inference": valid,
            "model_status": ("ok" if valid else "singular_not_interpretable"
                             if diagnostics.get("singular") else "failed"),
            "interpretation_status": "sensitivity_valid" if valid else "not_interpretable",
            "failure_reason": reason}


def fit_continuous_model(data, outcome, label, attempts=None):
    """Select a valid random-intercept LMM, otherwise patient-clustered GEE.

    Both attempts are retained for sensitivity/QC, including a converged but
    singular LMM. ESSDAI and all ESSPRI measures share this contract.
    """
    d = continuous_model_data(data, outcome)
    supported = d.patient_id.nunique() >= MIN_REPEATED_PATIENTS and d.time.nunique() >= 2
    mixed = gee = None
    diag = {"converged": False, "singular": False, "boundary": False,
            "hessian_warning": False, "covariance_finite": False,
            "random_intercept_variance": np.nan, "random_effect_variance": np.nan,
            "warning_text": "", "valid_for_primary_inference": False,
            "invalid_reason": "insufficient_support"}
    mixed_attempted = gee_attempted = False
    gee_warning = ""
    gee_reason = "insufficient_support"
    if supported:
        mixed_attempted = True
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mixed = MixedLM.from_formula("y ~ time", groups="patient_id",
                    re_formula="1", data=d).fit(reml=False, method="lbfgs", disp=False)
            diag = assess_mixedlm_fit(mixed, caught)
            if not _fixed_effect_inference_valid(mixed):
                diag["valid_for_primary_inference"] = False
                diag["invalid_reason"] += "; invalid_fixed_effect_inference"
        except Exception as exc:
            text = _warning_text(caught)
            diag.update(valid_for_primary_inference=False,
                        singular="singular" in (text + str(exc)).lower(),
                        boundary="boundary" in (text + str(exc)).lower(),
                        hessian_warning="hessian" in text.lower() and "not positive definite" in text.lower(),
                        warning_text=text,
                        invalid_reason=f"MixedLM error: {exc}")
        # GEE is also a sensitivity comparator when the LMM is valid.
        gee_attempted = True
        caught_gee = []
        try:
            with warnings.catch_warnings(record=True) as caught_gee:
                warnings.simplefilter("always")
                gee = GEE.from_formula("y ~ time", groups="patient_id", data=d,
                    family=Gaussian(), cov_struct=Exchangeable()).fit(cov_type="robust")
            gee_warning = _warning_text(caught_gee)
            gee_reason = "" if _fixed_effect_inference_valid(gee) else "invalid_GEE_inference"
        except Exception as exc:
            gee_reason = f"GEE error: {exc}"
            gee_warning = _warning_text(caught_gee)
    mixed_valid = diag["valid_for_primary_inference"]
    gee_valid = gee_attempted and not gee_reason
    fallback = mixed_attempted and not mixed_valid
    selected = "MixedLM" if mixed_valid else "GEE Gaussian" if gee_valid else "none"
    fit = mixed if mixed_valid else gee if gee_valid else None
    reason = diag["invalid_reason"] if fallback else "" if supported else "insufficient_support"
    if fallback and not gee_valid:
        reason += "; " + gee_reason
    history = [
        _model_attempt(mixed, label, "MixedLM random intercept", d, diag,
                       mixed_valid, diag["invalid_reason"]),
        _model_attempt(gee, label, "GEE Gaussian exchangeable", d,
            {"converged": bool(getattr(gee, "converged", False)), "singular": False,
             "random_intercept_variance": np.nan, "warning_text": gee_warning},
            gee_valid, gee_reason)]
    for item in history:
        item.update(selected_model=selected, fallback_used=fallback, fallback_reason=reason,
                    attempted=mixed_attempted if item["analysis"].startswith("MixedLM") else gee_attempted)
        if not item["attempted"]:
            item.update(model_status="insufficient_support", interpretation_status="descriptive_only")
    if fit is not None:
        e, se, p = float(fit.params.time), float(fit.bse.time), float(fit.pvalues.time)
    else:
        e = se = p = np.nan
    spans = d.groupby("patient_id").time.agg(lambda s: s.max() - s.min())
    row = {"outcome": label, "model": selected, "model_used": selected,
        "selected_model": selected, "primary_model": selected,
        "primary_model_valid": fit is not None,
        "n_patients": d.patient_id.nunique(), "n_observations": len(d),
        "n_unique_times": d.time.nunique(), "annual_change": e,
        "annual_change_estimate": e, "estimate_per_year": e,
        "ci95_low": e - norm.ppf(.975)*se, "ci95_high": e + norm.ppf(.975)*se,
        "p_value": p, "median_followup_years": spans.median(),
        "model_status": "ok" if fit is not None else "failed" if supported else "insufficient_support",
        "interpretation_status": "primary_valid" if fit is not None else "descriptive_only",
        "converged": bool(getattr(fit, "converged", False)),
        "singular": False if fit is not None else diag["singular"],
        "mixedlm_attempted": mixed_attempted, "mixedlm_converged": diag["converged"],
        "mixedlm_singular": diag["singular"], "mixedlm_boundary": diag["boundary"],
        "mixedlm_hessian_warning": diag["hessian_warning"],
        "mixedlm_random_intercept_variance": diag["random_intercept_variance"],
        "mixedlm_warning": diag["warning_text"], "gee_attempted": gee_attempted,
        "gee_converged": bool(getattr(gee, "converged", False)),
        "fallback_used": fallback, "fallback_reason": reason,
        "warning": "; ".join(filter(None, [diag["warning_text"], gee_warning])),
        "time_variation": d.time.nunique()}
    if attempts is not None:
        attempts.extend(history)
    return row, fit


def continuous_sensitivities(data, outcome, label):
    """Actual random-slope and quadratic fits, never used as primary results.

    ML AIC/BIC comparisons are only labeled comparable on identical rows and
    random-effect structure. Gaussian GEE AIC is deliberately not compared.
    """
    all_data = continuous_model_data(data, outcome)
    support = all_data.groupby("patient_id").time.nunique()
    slope_data = all_data.loc[all_data.patient_id.isin(support.index[support.ge(3)])]
    rows = []
    for analysis, d, formula, re_formula in (
        ("random_slope", slope_data, "y ~ time", "~time"),
        ("linear_same_cohort", all_data, "y ~ time", "1"),
        ("quadratic_time", all_data, "y ~ time + I(time ** 2)", "1")):
        fit = None
        diag = {"converged": False, "singular": False, "random_intercept_variance": np.nan}
        row = {"outcome": label, "analysis": analysis, "n_patients": d.patient_id.nunique(),
               "n_observations": len(d), "model_status": "insufficient_support",
               "interpretation_status": "descriptive_only", "AIC": np.nan,
               "BIC": np.nan, "logLik": np.nan, "random_slope_variance": np.nan,
               "random_effect_correlation": np.nan, "quadratic_estimate": np.nan,
               "quadratic_p_value": np.nan, "ic_comparable": False,
               "comparison_note": "No valid comparable ML fit"}
        if d.patient_id.nunique() >= MIN_REPEATED_PATIENTS and d.time.nunique() >= 3:
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fit = MixedLM.from_formula(formula, groups="patient_id", re_formula=re_formula,
                        data=d).fit(reml=False, method="lbfgs", disp=False)
                diag = assess_mixedlm_fit(fit, caught)
                valid = diag["valid_for_primary_inference"] and _fixed_effect_inference_valid(fit)
                row.update(model_status="ok" if valid else "singular_not_interpretable"
                           if diag["singular"] or diag["boundary"] else "unstable",
                           interpretation_status="sensitivity_valid" if valid else "not_interpretable",
                           AIC=float(fit.aic), BIC=float(fit.bic), logLik=float(fit.llf),
                           estimate=float(fit.params.time), ci95_low=float(fit.conf_int().loc["time", 0]),
                           ci95_high=float(fit.conf_int().loc["time", 1]), p_value=float(fit.pvalues.time))
                if analysis == "random_slope":
                    cov = np.asarray(fit.cov_re, dtype=float)
                    row["random_slope_variance"] = cov[1, 1]
                    denom = np.sqrt(cov[0, 0]*cov[1, 1]) if cov[0, 0]*cov[1, 1] > 0 else 0
                    row["random_effect_correlation"] = cov[0, 1]/denom if denom else np.nan
                if analysis == "quadratic_time":
                    row.update(quadratic_estimate=float(fit.params["I(time ** 2)"]),
                               quadratic_p_value=float(fit.pvalues["I(time ** 2)"]),
                               note="Descriptive evidence of non-linear observed trajectory only")
            except Exception as exc:
                row.update(model_status="failed", note=str(exc))
                diag["warning_text"] = _warning_text(caught)
        rows.append({**row, **diag})
    linear, quadratic = rows[1:]
    if linear["model_status"] == quadratic["model_status"] == "ok":
        for row in (linear, quadratic):
            row.update(ic_comparable=True, comparison_note="ML; identical cohort and random intercept")
    return pd.DataFrame(rows)


def bh_fdr(values: pd.Series) -> pd.Series:
    out=pd.Series(np.nan,index=values.index,dtype=float); valid=values.dropna(); m=len(valid)
    if not m:return out
    order=valid.sort_values().index; ranked=valid.loc[order].to_numpy()*m/np.arange(1,m+1)
    adjusted=np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1); out.loc[order]=adjusted
    return out


def resolve_domains(df: pd.DataFrame) -> dict[str,str]:
    resolved={}
    for domain in DOMAINS:
        matches=[c for c in DOMAIN_ALIASES[domain] if c in df]
        if len(matches)>1: raise ValueError(f"Ambiguous ordinal columns for {domain}: {matches}")
        if matches: resolved[domain]=matches[0]
    return resolved


def domain_contract_qc(df, resolved):
    rows = []
    for domain in DOMAINS:
        col = resolved.get(domain)
        valid = df[col].notna() if col else pd.Series(False, index=df.index)
        rows.append({"domain": domain, "expected_public_column": DOMAIN_ALIASES[domain][0],
                     "resolved_column": col, "available": col is not None,
                     "n_nonmissing": int(valid.sum()), "n_patients": df.loc[valid, "patient_id"].nunique(),
                     "status": "pass" if col == DOMAIN_ALIASES[domain][0] else "legacy_alias"
                               if col else "missing", "n_domains_expected": 12,
                     "n_domains_resolved": len(resolved),
                     "missing_domains": ";".join(d for d in DOMAINS if d not in resolved)})
    return pd.DataFrame(rows)


def _valid_gee(fit, caught, key):
    warning = _warning_text(caught).lower()
    return (_fixed_effect_inference_valid(fit, key)
            and not any(s in warning for s in ("iteration limit", "perfect separation", "singular")))


def domain_analyses(df, resolved):
    models, changes, support = [], [], []
    for domain in DOMAINS:
        col = resolved.get(domain)
        z = df[["patient_id", TIME, "is_clinical_baseline"] + ([col] if col else [])].copy()
        if col:
            z[col] = pd.to_numeric(z[col], errors="coerce")
            if (~z[col].dropna().isin([0, 1, 2, 3])).any():
                raise ValueError(f"{col} outside ordinal range 0-3")
            z = z.dropna(subset=[col, TIME])
            z = z.loc[np.isfinite(z[TIME]) & z[TIME].ge(0)].sort_values(["patient_id", TIME], kind="stable")
        else:
            z = z.iloc[:0]
        repeats = z.groupby("patient_id")[TIME].nunique()
        changes_n = 0
        if col:
            for _, group in z.groupby("patient_id"):
                changes_n += int((group[col].diff().ne(0) & group[TIME].diff().gt(0)).sum())
        sup = {"domain": domain, "column": col, "resolved_column": col,
               "n_patients": z.patient_id.nunique(), "n_patients_repeated": int(repeats.ge(2).sum()),
               "n_observations": len(z), "n_nonzero": int(z[col].gt(0).sum()) if col else 0,
               "n_state_changes": changes_n, "n_unique_levels": z[col].nunique() if col else 0,
               "n_active_patients": z.loc[z[col].gt(0), "patient_id"].nunique() if col else 0,
               "n_inactive_patients": z.loc[z[col].eq(0), "patient_id"].nunique() if col else 0}
        support.append(sup)
        row = {**sup, "domain_label": DOMAIN_LABELS[domain], "outcome_type": "ordinal",
               "model": "none", "cluster_variable": "patient_id", "effect": np.nan,
               "effect_scale": "descriptive_only", "ci95_low": np.nan, "ci95_high": np.nan,
               "p_value": np.nan, "model_status": "descriptive_only" if col else "column_unavailable",
               "valid_for_inference": False, "converged": False,
               "fallback_used": False, "fallback_reason": "", "ordinal_warning": "",
               "binary_warning": "", "interpretation_note": "Insufficient repeated-measure support"}
        gate = (sup["n_patients_repeated"] >= MIN_REPEATED_PATIENTS
                and changes_n >= MIN_STATE_CHANGES and sup["n_unique_levels"] >= 2)
        if gate:
            key = TIME
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    # OrdinalGEE inserts threshold intercepts internally. Its
                    # slope models P(score > cut); positive is higher activity.
                    fit = OrdinalGEE(z[col].astype(int), z[[key]].astype(float),
                        groups=z.patient_id.to_numpy(), cov_struct=Exchangeable()).fit(cov_type="robust")
                row["ordinal_warning"] = _warning_text(caught)
                if not _valid_gee(fit, caught, key):
                    raise RuntimeError("OrdinalGEE nonconverged or invalid robust inference")
                e, se = float(fit.params[key]), float(fit.bse[key])
                row.update(model="OrdinalGEE", effect=e, effect_scale="ordinal log-odds per year",
                           ci95_low=e - norm.ppf(.975)*se, ci95_high=e + norm.ppf(.975)*se,
                           p_value=float(fit.pvalues[key]), model_status="ok", converged=True,
                           valid_for_inference=True, interpretation_note="Patient-clustered ordinal time effect")
            except Exception as exc:
                row.update(fallback_used=True, fallback_reason=str(exc), ordinal_warning=_warning_text(caught))
                q = z.assign(active=z[col].gt(0).astype(int), time=z[TIME].astype(float))
                binary_changes = sum(int((g.active.diff().ne(0) & g.time.diff().gt(0)).sum())
                                     for _, g in q.groupby("patient_id"))
                binary_supported = (q.active.nunique() == 2 and binary_changes >= MIN_STATE_CHANGES
                                    and q.time.nunique() >= 2)
                if binary_supported:
                    caught_binary = []
                    try:
                        with warnings.catch_warnings(record=True) as caught_binary:
                            warnings.simplefilter("always")
                            fit = GEE.from_formula("active ~ time", groups="patient_id", data=q,
                                family=Binomial(), cov_struct=Exchangeable()).fit(cov_type="robust")
                        row["binary_warning"] = _warning_text(caught_binary)
                        if not _valid_gee(fit, caught_binary, "time"):
                            raise RuntimeError("Binomial GEE nonconverged or invalid robust inference")
                        e, se = float(fit.params.time), float(fit.bse.time)
                        effect, lo, hi = np.exp([e, e - norm.ppf(.975)*se, e + norm.ppf(.975)*se])
                        if not np.isfinite([effect, lo, hi]).all() or lo <= 0:
                            raise RuntimeError("Nonfinite binary odds-ratio inference")
                        row.update(outcome_type="binary", model="GEE binomial", effect=effect,
                                   effect_scale="odds ratio active vs inactive per year", ci95_low=lo,
                                   ci95_high=hi, p_value=float(fit.pvalues.time), model_status="ok",
                                   converged=True, valid_for_inference=True,
                                   interpretation_note="Patient-clustered active/inactive fallback; descriptive association")
                    except Exception as exc2:
                        row.update(model_status="failed", binary_warning=_warning_text(caught_binary),
                                   interpretation_note=f"Ordinal and binary models failed: {exc}; {exc2}")
                else:
                    row["interpretation_note"] = f"Ordinal model failed ({exc}); insufficient active/inactive variation"
        models.append(row)
        paired = []
        if col:
            for _, g in z.groupby("patient_id"):
                b = g.loc[g.is_clinical_baseline.eq(True), col]
                after = g.loc[~g.is_clinical_baseline.eq(True) & g[TIME].gt(0)]
                if len(b) == 1 and len(after):
                    paired.append((float(b.iloc[0]), float(after.iloc[-1][col])))
        delta = np.array([b - a for a, b in paired])
        changes.append({"domain": domain, "n_paired_patients": len(paired),
            "baseline_median": np.median([a for a, _ in paired]) if paired else np.nan,
            "last_median": np.median([b for _, b in paired]) if paired else np.nan,
            "median_delta": np.median(delta) if len(delta) else np.nan,
            "improved_one_or_more_levels": int((delta < 0).sum()), "stable": int((delta == 0).sum()),
            "worsened_one_or_more_levels": int((delta > 0).sum())})
    models = pd.DataFrame(models)
    models["q_value"] = np.nan
    valid = models.valid_for_inference & models.model_status.eq("ok")
    models.loc[valid, "q_value"] = bh_fdr(models.loc[valid, "p_value"])
    models["fdr_family"] = "ESSDAI domain time effects"
    return models, pd.DataFrame(changes), pd.DataFrame(support)


def availability(df, measures):
    rows = []
    for name, col in measures.items():
        valid = df[col].notna() if col in df else pd.Series(False, index=df.index)
        z = df.loc[valid]
        counts = z.groupby("patient_id").size()
        dates = z.groupby("patient_id").clinical_anchor_date.nunique()
        spans = z.groupby("patient_id")[TIME].agg(lambda s: s.max() - s.min())
        rows.append({"measure": name, "n_observations": len(z),
                     "n_patients_any": z.patient_id.nunique(), "n_patients_repeated": int(counts.ge(2).sum()),
                     "n_patients_distinct_dates_ge2": int(dates.ge(2).sum()),
                     "median_measurements_per_patient": counts.median(),
                     "median_observed_span_years": spans.median(),
                     "q1_observed_span_years": spans.quantile(.25), "q3_observed_span_years": spans.quantile(.75)})
    return pd.DataFrame(rows)


def select_one_episode_per_patient_per_window(df):
    """Select independently of outcome availability, preventing visit weighting."""
    x = df.copy()
    times = pd.to_numeric(x[TIME], errors="coerce")
    x["time_window"] = pd.Series(pd.NA, index=x.index, dtype="string")
    official = x.is_clinical_baseline.eq(True)
    x.loc[official, "time_window"] = "baseline"
    follow = ~official & times.gt(0) & np.isfinite(times)
    x.loc[follow, "time_window"] = pd.cut(times.loc[follow], [0, 1, 2, 3, 5, np.inf],
                                         labels=TIME_WINDOWS[1:]).astype("string")
    x = x.loc[x.time_window.notna()].copy()
    centers = {">0-1y": .5, ">1-2y": 1.5, ">2-3y": 2.5, ">3-5y": 4.}
    x["_distance"] = (x[TIME] - x.time_window.map(centers)).abs()
    x.loc[x.time_window.eq(">5y"), "_distance"] = x.loc[x.time_window.eq(">5y"), TIME]
    x.loc[x.time_window.eq("baseline"), "_distance"] = 0.
    if x.loc[x.time_window.eq("baseline")].patient_id.duplicated().any():
        raise ValueError("multiple official baselines in descriptive windows")
    x = x.sort_values(["patient_id", "time_window", "_distance", "clinical_anchor_date",
                       "clinical_episode_id"], kind="stable")
    x = x.drop_duplicates(["patient_id", "time_window"])
    x["selection_rule"] = WINDOW_SELECTION
    return x.drop(columns="_distance")


def missingness_by_time(selected, measures):
    rows = []
    for name, col in measures.items():
        for period in TIME_WINDOWS:
            g = selected.loc[selected.time_window.eq(period)]
            valid = g[col].notna() if col in g else pd.Series(False, index=g.index)
            eligible = g.patient_id.nunique()
            evaluable = g.loc[valid, "patient_id"].nunique()
            rows.append({"measure": name, "time_window": period,
                         "n_patients_eligible": eligible, "n_patients_evaluable": evaluable,
                         "n_patients_missing": eligible - evaluable,
                         "pct_missing": 100*(eligible - evaluable)/eligible if eligible else np.nan,
                         "n_rows": len(g), "selection_rule": WINDOW_SELECTION})
    return pd.DataFrame(rows)


def domain_activity_by_time(selected, resolved):
    rows = []
    for domain in DOMAINS:
        col = resolved.get(domain)
        for period in TIME_WINDOWS:
            g = selected.loc[selected.time_window.eq(period)]
            z = pd.to_numeric(g[col], errors="coerce").dropna() if col else pd.Series(dtype=float)
            rows.append({"domain": domain, "time_window": period, "n_evaluable": len(z),
                         "pct_active": 100*z.gt(0).mean() if len(z) else np.nan,
                         "selection_rule": WINDOW_SELECTION})
    return pd.DataFrame(rows)


def progression_attrition(df, measures, baseline_ids):
    """Sequential patient exclusions; the last set equals continuous_model_data."""
    rows = []
    initial = set(baseline_ids)
    for name, col in measures.items():
        obs = df.loc[df[col].notna()] if col in df else df.iloc[:0]
        counts = obs.groupby("patient_id").size()
        dates = obs.groupby("patient_id").clinical_anchor_date.nunique()
        eligible = set(continuous_model_data(df, col).patient_id)
        steps = [
            ("canonical_baseline_cohort", "Canonical Step 11 baseline cohort", initial),
            ("at_least_one_outcome", ">=1 observed outcome", set(counts.index)),
            ("at_least_two_outcomes", ">=2 observed outcome rows", set(counts.index[counts.ge(2)])),
            ("distinct_dates", ">=2 distinct valid outcome dates", set(dates.index[dates.ge(2)])),
            ("valid_time", "After dropping invalid/negative/nonfinite times, >=2 dates and times", eligible),
            ("final_modeled_cohort", "Final eligible cohort supplied to longitudinal model", eligible)]
        remaining = initial.copy()
        for order, (step, description, keep) in enumerate(steps, 1):
            previous = len(remaining)
            remaining &= keep
            rows.append({"outcome": name, "step_order": order, "step": step,
                         "description": description, "n_remaining": len(remaining),
                         "n_excluded_at_step": previous - len(remaining)})
    return pd.DataFrame(rows)


def same_date_episode_qc(df):
    valid = df.loc[df.clinical_anchor_date.notna()]
    dup = valid.duplicated(["patient_id", "clinical_anchor_date"], keep=False)
    groups = valid.loc[dup].groupby(["patient_id", "clinical_anchor_date"]).size()
    return pd.DataFrame([{"n_same_date_episode_rows": int(dup.sum()),
        "n_same_date_patient_dates": len(groups), "n_patients_affected": valid.loc[dup, "patient_id"].nunique(),
        "n_zero_interval_pairs": int((groups - 1).sum()),
        "policy": "retained as distinct canonical episodes; zero-duration transition intervals excluded"}])


def marginal_predictions(fit, time_grid):
    """Population fixed-effect prediction and covariance-derived 95% CI."""
    params = getattr(fit, "fe_params", fit.params)
    names = ["Intercept", "time"]
    beta = params.loc[names].to_numpy(dtype=float)
    cov = fit.cov_params().loc[names, names].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(time_grid)), time_grid])
    prediction = design @ beta
    variance = np.einsum("ij,jk,ik->i", design, cov, design)
    if not np.isfinite(variance).all() or (variance < -SINGULAR_TOL).any():
        raise ValueError("invalid prediction covariance")
    se = np.sqrt(np.maximum(variance, 0))
    return pd.DataFrame({"time": time_grid, "prediction": prediction,
                         "ci95_low": prediction - norm.ppf(.975)*se,
                         "ci95_high": prediction + norm.ppf(.975)*se})


def _trajectory_axis(ax, df, outcome, fit, model_type, title):
    z = continuous_model_data(df, outcome)
    for _, g in z.groupby("patient_id"):
        g = g.sort_values("time")
        ax.plot(g.time, g.y, color="0.75", alpha=.3, lw=.6)
    ax.scatter(z.time, z.y, s=8, alpha=.25, color="0.5")
    if fit is not None and len(z):
        pred = marginal_predictions(fit, np.linspace(z.time.min(), z.time.max(), 100))
        ax.plot(pred.time, pred.prediction, color="#9c2f45", lw=2, label="Model marginal trajectory")
        ax.fill_between(pred.time, pred.ci95_low, pred.ci95_high, color="#9c2f45", alpha=.2, label="95% CI")
        ax.legend(fontsize=7)
    ax.set(xlabel="Years since clinical baseline", ylabel=title,
           title=f"{title}\nSelected model: {model_type}; {z.patient_id.nunique()} patients; {len(z)} observations")
    if fit is None:
        ax.text(.02, .95, "No valid inferential model; observed data only", transform=ax.transAxes, fontsize=8)


def plot_continuous_trajectory(df, outcome, fit_result, model_type, path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    _trajectory_axis(ax, df, outcome, fit_result, model_type, title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_km(tte, path, title):
    curve, stats = km_estimate(tte)
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(curve):
        ax.step([0, *curve.time], [1, *curve.survival], where="post")
    events = int(tte.event.sum()) if len(tte) else 0
    if events < MIN_KM_EVENTS:
        title += f"\nOnly {events} observed event{'s' if events != 1 else ''}; descriptive and unstable"
    ax.set(xlabel="Years since clinical baseline", ylabel="Event-free probability", ylim=(0, 1.02), title=title)
    times = [0, 1, 2, 5]
    risk = [int(tte.time_years.ge(t).sum()) if len(tte) else 0 for t in times]
    ax.table(cellText=[times, risk], rowLabels=["Years", "At risk"],
             cellLoc="center", bbox=[0, -.37, 1, .2])
    fig.text(.08, .025, "First-observed-event KM; biological changes are interval censored.\n"
             f"Median: {stats['median_time_to_event']} ({stats['median_time_to_event_status']}). "
             "Small risk sets limit precision.", fontsize=8)
    fig.subplots_adjust(bottom=.34, top=.85)
    fig.savefig(path)
    plt.close(fig)


def save_heatmap(matrix, path, title, fmt=".1f", denominators=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    a = np.asarray(matrix, dtype=float)
    image = ax.imshow(a, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(image, ax=ax, label="Percent")
    labels = [DOMAIN_LABELS.get(d, d) for d in matrix.index]
    ax.set(xticks=range(matrix.shape[1]), xticklabels=matrix.columns,
           yticks=range(matrix.shape[0]), yticklabels=labels, title=title)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            text = format(a[i, j], fmt) + "%" if np.isfinite(a[i, j]) else "NA"
            if denominators is not None:
                text += f"\nn={int(denominators.iloc[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_domain_forest(models, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, model, null, label in (
        (axes[0], "OrdinalGEE", 0, "Ordinal log-odds per year"),
        (axes[1], "GEE binomial", 1, "Active/inactive odds ratio per year")):
        ok = models.loc[models.valid_for_inference & models.model.eq(model)]
        ax.axvline(null, color="0.5", ls="--")
        if len(ok):
            ax.errorbar(ok.effect, range(len(ok)),
                        xerr=[ok.effect - ok.ci95_low, ok.ci95_high - ok.effect], fmt="o")
        else:
            ax.text(.5, .5, "No valid models", transform=ax.transAxes, ha="center")
        if model == "GEE binomial":
            ax.set_xscale("log")
        ax.set(yticks=range(len(ok)), yticklabels=ok.domain_label, xlabel=label,
               title=f"{model}: patient-clustered")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--integrated", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    p.add_argument("--baseline", type=Path, default=common.INTEGRATED_BASELINE_PARQUET)
    p.add_argument("--output-root", type=Path, default=common.PROJECT_ROOT)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                   help="Regenerate existing Step 13 outputs (default); --no-overwrite protects them")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(args=None):
    args = parse_args() if args is None else args
    tables = args.output_root/"outputs/tables/blockA"/STEM
    figures = args.output_root/"outputs/figures/blockA"/STEM
    qc = args.output_root/"outputs/qc/blockA"/STEM
    logs = args.output_root/"outputs/logs"/STEM
    analytic = args.output_root/"data/analytic/blockA"/STEM
    if not args.dry_run and not args.overwrite:
        existing = [p for folder in (tables, figures, qc, analytic) if folder.exists()
                    for p in folder.iterdir() if p.is_file()]
        if existing:
            raise FileExistsError("Step 13 outputs exist; pass --overwrite to regenerate")
    for folder in (tables, figures, qc, logs, analytic):
        folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logs/f"{STEM}.log"), logging.StreamHandler()], force=True)
    LOG.info("Inputs integrated=%s baseline=%s", args.integrated, args.baseline)
    baseline = read_table(args.baseline)
    df = validate_inputs(read_table(args.integrated), baseline)
    resolved = resolve_domains(df)
    domain_qc = domain_contract_qc(df, resolved)
    domain_qc.to_csv(qc/"13_progression_domain_contract_qc.csv", index=False)
    if len(resolved) != len(DOMAINS):
        missing = [d for d in DOMAINS if d not in resolved]
        raise ValueError(f"ESSDAI domain contract: {len(resolved)}/12 domains resolved; missing {missing}. "
                         "Regenerate Step 06 then Step 10; never derive domains inside Step 13.")
    LOG.info("12/12 ESSDAI domains resolved")
    for col in resolved.values():
        values = pd.to_numeric(df[col], errors="coerce")
        if (~values.dropna().isin([0, 1, 2, 3])).any():
            raise ValueError(f"{col} outside ordinal range 0-3")
        df[col] = values
    df["essdai_activity_state"] = classify_essdai_series(df[ESSDAI])
    df["essdai_activity_rank"] = df.essdai_activity_state.map({**STATE_RANK, "Missing": np.nan}).astype("Float64")
    df["previous_essdai"] = df.groupby("patient_id")[ESSDAI].shift()
    df["delta_essdai_from_previous"] = df[ESSDAI] - df.previous_essdai
    df["previous_essdai_state"] = df.groupby("patient_id").essdai_activity_state.shift()
    df["state_change_from_previous"] = df.essdai_activity_rank - df.groupby("patient_id").essdai_activity_rank.shift()
    continuous_measures = {"ESSDAI total": ESSDAI, **{f"ESSPRI {k}": v for k, v in ESSPRI.items()}}
    measures = {**continuous_measures, **{f"ESSDAI domain {k}": v for k, v in resolved.items()}}
    avail = availability(df, measures)
    base_rows = df.loc[df.is_clinical_baseline.eq(True)]
    counts = df.loc[df[ESSDAI].notna()].groupby("patient_id").size()
    flow = [{"cohort": "Overall integrated baseline cohort", "n_patients": baseline.patient_id.nunique()}]
    for row in avail.itertuples():
        flow.extend([{"cohort": f"{row.measure}: >=1 observation", "n_patients": row.n_patients_any},
                     {"cohort": f"{row.measure}: >=2 distinct dates", "n_patients": row.n_patients_distinct_dates_ge2}])
    for state in STATE_ORDER:
        flow.append({"cohort": f"Baseline ESSDAI {state}",
                     "n_patients": int(base_rows[ESSDAI].map(classify_essdai_activity).eq(state).sum())})
    cohort = pd.DataFrame(flow)
    attrition = progression_attrition(df, continuous_measures, baseline.patient_id.astype("string"))
    if args.dry_run:
        print(cohort.to_string(index=False))
        print("\n", avail.to_string(index=False))
        LOG.info("Dry run complete: 12/12 domains; contract QC only, no final analyses")
        return
    cohort.to_csv(tables/"13_progression_cohort_flow.csv", index=False)
    avail.to_csv(tables/"13_progression_measure_availability.csv", index=False)
    attrition.to_csv(tables/"13_progression_attrition.csv", index=False)
    same_date_episode_qc(df).to_csv(qc/"13_same_date_episode_qc.csv", index=False)
    selected_episodes = select_one_episode_per_patient_per_window(df)
    missingness_by_time(selected_episodes, measures).to_csv(qc/"13_progression_missingness_by_time.csv", index=False)
    activity = domain_activity_by_time(selected_episodes, resolved)
    activity.to_csv(tables/"13_essdai_domain_activity_by_time.csv", index=False)

    attempts = []
    primary, fit = fit_continuous_model(df, ESSDAI, "ESSDAI total", attempts)
    pd.DataFrame([primary]).to_csv(tables/"13_essdai_longitudinal_model.csv", index=False)
    sensitivity = pd.concat([pd.DataFrame(attempts), continuous_sensitivities(df, ESSDAI, "ESSDAI total")], ignore_index=True)
    sensitivity.to_csv(tables/"13_essdai_model_sensitivity.csv", index=False)
    plot_continuous_trajectory(df, ESSDAI, fit, primary["selected_model"],
                              figures/"13_essdai_trajectory.pdf", "ESSDAI total")

    prog = patient_progression(df)
    write_absolute_change_outputs(df, prog, tables, figures, qc, analytic)
    prog.to_csv(tables/"13_essdai_baseline_to_last_status.csv", index=False)
    summary = []
    for status in ("worsened", "stable", "improved"):
        summary.append({"definition": f"baseline_to_last_{status}", "n": int(prog.status.eq(status).sum()), "denominator": len(prog)})
    for label, mask in (("ever_worsened", prog.ever_worsened), ("never_worsened", ~prog.ever_worsened),
        ("ever_reached_moderate_or_high", prog.maximum_state_reached.isin(["Moderate", "High"])),
        ("ever_reached_high", prog.maximum_state_reached.eq("High"))):
        summary.append({"definition": label, "n": int(mask.sum()), "denominator": len(prog)})
    summary = pd.DataFrame(summary)
    summary["pct"] = 100*summary.n/summary.denominator.replace(0, np.nan)
    summary.to_csv(tables/"13_essdai_progression_status_summary.csv", index=False)
    transitions, raw_intervals = build_state_transitions(df)
    transitions.to_csv(tables/"13_essdai_state_transition_matrix.csv", index=False)
    matrix = transitions.pivot(index="from_state", columns="to_state", values="row_pct").reindex(index=STATE_ORDER, columns=STATE_ORDER)
    save_heatmap(matrix, figures/"13_essdai_state_transition_heatmap.pdf", "ESSDAI state transitions (%)")
    event_qc = []
    for name, origin, title in (("low_to_ge5", "low", "Low to ESSDAI >=5"),
                               ("moderate_to_high", "moderate", "Moderate to High ESSDAI")):
        tte = build_tte_risk_set(df, origin)
        stats = tte_summary(tte, name)
        pd.DataFrame([stats]).to_csv(tables/f"13_tte_{name}.csv", index=False)
        tte.to_csv(tables/f"13_tte_{name}_patient_level.csv", index=False)
        plot_km(tte, figures/f"13_km_{name}.pdf", title)
        event_qc.append(stats)
    pd.DataFrame(event_qc).to_csv(qc/"13_progression_event_qc.csv", index=False)

    dmodels, dchange, dsupport = domain_analyses(df, resolved)
    dmodels.to_csv(tables/"13_essdai_domain_models.csv", index=False)
    dchange.to_csv(tables/"13_essdai_domain_change_summary.csv", index=False)
    dsupport.to_csv(qc/"13_progression_domain_support.csv", index=False)
    domain_matrix = activity.pivot(index="domain", columns="time_window", values="pct_active").reindex(index=DOMAINS, columns=TIME_WINDOWS)
    domain_n = activity.pivot(index="domain", columns="time_window", values="n_evaluable").reindex(index=DOMAINS, columns=TIME_WINDOWS)
    save_heatmap(domain_matrix, figures/"13_essdai_domain_activity_heatmap.pdf",
                 "ESSDAI domain activity: % of evaluable patients", denominators=domain_n)
    plot_domain_forest(dmodels, figures/"13_essdai_domain_effects_forest.pdf")

    pro_rows, pro_changes, pro_fits, spacing_rows = [], [], {}, []
    for name, col in ESSPRI.items():
        if col not in df:
            continue
        row, pro_fit = fit_continuous_model(df, col, f"ESSPRI {name}", attempts)
        row.update(measure=name, primary_cohort="all_repeated", protocol_spacing_role="sensitivity")
        pro_rows.append(row)
        pro_fits[name] = (pro_fit, row["selected_model"])
        d = continuous_model_data(df, col)
        spans = d.groupby("patient_id").time.agg(lambda s: s.max() - s.min())
        spaced_ids = spans.index[spans.ge(.5)]
        for cohort_name, ids in (("all_repeated", spans.index), ("protocol_spacing_ge_6_months", spaced_ids)):
            use = df.loc[df.patient_id.isin(ids)]
            pairs = []
            for _, g in use.groupby("patient_id"):
                b = g.loc[g.is_clinical_baseline.eq(True), col]
                a = g.loc[g.clinical_anchor_date.gt(g.clinical_baseline_date) & g[col].notna()]
                if len(b) == 1 and pd.notna(b.iloc[0]) and len(a):
                    pairs.append(a.iloc[-1][col] - b.iloc[0])
            pro_changes.append({"measure": name, "cohort": cohort_name, "n": len(pairs),
                "median_change": np.median(pairs) if pairs else np.nan,
                "n_patients_repeated": len(spans), "n_patients_spacing_ge6mo": len(spaced_ids),
                "pct_of_repeated": 100*len(spaced_ids)/len(spans) if len(spans) else np.nan,
                "median_span_years": spans.reindex(ids).median(), "primary_cohort": "all_repeated",
                "spacing_definition": ">=0.5 years between first and last evaluable outcome; sensitivity"})
        spaced_row, _ = fit_continuous_model(df.loc[df.patient_id.isin(spaced_ids)], col, f"ESSPRI {name}")
        spaced_row.update(measure=name, cohort="protocol_spacing_ge_6_months", interpretation_role="sensitivity")
        spacing_rows.append(spaced_row)
    pro_models = pd.DataFrame(pro_rows)
    if len(pro_models):
        pro_models["q_value"] = np.nan
        ix = pro_models.measure.isin(["dryness", "fatigue", "pain"]) & pro_models.primary_model_valid
        pro_models.loc[ix, "q_value"] = bh_fdr(pro_models.loc[ix, "p_value"])
    pro_models.to_csv(tables/"13_esspri_models.csv", index=False)
    pd.DataFrame(pro_changes).to_csv(tables/"13_esspri_change_summary.csv", index=False)
    pd.DataFrame(spacing_rows).to_csv(tables/"13_esspri_protocol_spacing_sensitivity.csv", index=False)
    pro_fit, model = pro_fits.get("total", (None, "none"))
    plot_continuous_trajectory(df, ESSPRI["total"], pro_fit, model,
                              figures/"13_esspri_trajectory_total.pdf", "ESSPRI total")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, name in zip(axes, ("dryness", "fatigue", "pain")):
        pro_fit, model = pro_fits.get(name, (None, "none"))
        _trajectory_axis(ax, df, ESSPRI[name], pro_fit, model, f"ESSPRI {name}")
        ax.set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig(figures/"13_esspri_trajectory_components.pdf")
    plt.close(fig)

    clinical_change = []
    for r in prog.itertuples():
        clinical_change.extend([
            {"patient_id": r.patient_id, "outcome": "essdai_improved_ge3", "evaluable": True,
             "met": r.delta_essdai <= CHANGE_THRESHOLDS["essdai_improvement"]},
            {"patient_id": r.patient_id, "outcome": "essdai_legacy_worsening_ge5_sensitivity", "evaluable": True,
             "met": r.delta_essdai >= CHANGE_THRESHOLDS["essdai_legacy_worsening"]}])
    for name, col in ESSPRI.items():
        if col not in df:
            continue
        for pid, g in df.groupby("patient_id"):
            b = g.loc[g.is_clinical_baseline.eq(True), col]
            a = g.loc[g.clinical_anchor_date.gt(g.clinical_baseline_date) & g[col].notna()]
            if len(b) == 1 and pd.notna(b.iloc[0]) and len(a):
                delta = a.iloc[-1][col] - b.iloc[0]
                clinical_change.extend([
                    {"patient_id": pid, "outcome": f"esspri_{name}_absolute_improvement", "evaluable": True,
                     "met": delta <= CHANGE_THRESHOLDS["esspri_absolute_improvement"]},
                    {"patient_id": pid, "outcome": f"esspri_{name}_relative_improvement", "evaluable": b.iloc[0] != 0,
                     "met": delta/b.iloc[0] <= CHANGE_THRESHOLDS["esspri_relative_improvement"] if b.iloc[0] != 0 else pd.NA}])
    pd.DataFrame(clinical_change, columns=["patient_id", "outcome", "evaluable", "met"]).to_csv(tables/"13_clinical_change_sensitivity.csv", index=False)
    keep = REQUIRED + [ESSDAI, *ESSPRI.values(), *resolved.values(), "essdai_activity_state",
        "essdai_activity_rank", "previous_essdai", "delta_essdai_from_previous", "previous_essdai_state", "state_change_from_previous"]
    df[[c for c in dict.fromkeys(keep) if c in df]].rename(columns={ESSDAI: "essdai_total",
        **{v: f"esspri_{k}" for k, v in ESSPRI.items()}}).to_parquet(analytic/"13_progression_episode_level.parquet", index=False)
    model_qc = pd.DataFrame(attempts)
    model_qc["warning_category"] = np.select(
        [model_qc.singular.eq(True),
         model_qc.get("boundary", pd.Series(False, index=model_qc.index)).eq(True),
         model_qc.get("hessian_warning", pd.Series(False, index=model_qc.index)).eq(True),
         ~model_qc.converged.eq(True), model_qc.warning_text.fillna("").ne("")],
        ["singular", "boundary", "hessian_not_positive_definite", "not_converged", "other"], default="none")
    model_qc.to_csv(qc/"13_progression_model_qc.csv", index=False)
    pop_checks = {}
    if "pop__status" in df:
        known = df[ESSDAI].notna()
        pop_checks = {"pop1_essdai_below5": int((known & df["pop__status"].eq("Pop1") & df[ESSDAI].lt(5)).sum()),
            "pop2_pop3_essdai_ge5": int((known & df["pop__status"].isin(["Pop2", "Pop3"]) & df[ESSDAI].ge(5)).sum())}
    pd.DataFrame([{"check": "structural_validation", "status": "pass", "n_patients": df.patient_id.nunique(),
        "n_domains_expected": 12, "n_domains_resolved": len(resolved), "missing_domains": "",
        "n_duplicate_episode_keys": int(df.duplicated(KEYS).sum()),
        "n_negative_times": int(df[TIME].lt(0).sum()), "n_invalid_times": int(df[TIME].isna().sum()),
        "n_official_baselines": len(base_rows), "baseline_essdai_available": int(base_rows[ESSDAI].notna().sum()),
        "baseline_essdai_missing": int(base_rows[ESSDAI].isna().sum()), **pop_checks}]).to_csv(qc/"13_progression_qc.csv", index=False)
    LOG.info("Completed: %d rows, %d patients, 12/12 domains", len(df), df.patient_id.nunique())
    print("ORDER TO REVIEW DISEASE ACTIVITY PROGRESSION OUTPUTS\n"
          "1. Domain contract / structural QC\n2. Cohort flow / sequential attrition\n"
          "3. Selected continuous models / model QC / sensitivities\n4. Model-derived trajectories\n"
          "5. State transitions / sparse-event KM\n6. Patient-clustered domain models / FDR\n"
          "7. Patient-window activity / missingness\n8. ESSPRI models / spacing sensitivity")


if __name__ == "__main__":
    main()

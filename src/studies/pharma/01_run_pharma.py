#!/usr/bin/env python3
"""Run the Pharma Pop1/Pop2/Pop3 clinical-episode study.

This downstream study reads only the frozen integrated master and canonical
adjacent-episode intervals. Results are descriptive or associational, never
causal. Optional analyses degrade to explicit ``skipped`` records.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import logging
import math
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import common
from src.studies._shared import (STUDY_CONTRACT_VERSION, benjamini_hochberg,
    create_study_dirs, enrich_transition_intervals, json_default, load_parquet,
    sha256_file, validate_integrated_dataset, validate_predictors,
    validate_transition_intervals, write_json)

CLASSIFIABLE = ["Pop1", "Pop2", "Pop3"]
BINARY_BIOMARKERS = ["baseline_anti_ro_ssa", "baseline_anti_la_ssb", "baseline_ana",
    "baseline_rf", "baseline_cryoglobulinemia", "baseline_high_igg",
    "baseline_low_c3", "baseline_low_c4", "baseline_leukopenia"]
ORGANS = ["glandular_active", "n_extraglandular_domains_active", "eg_constitutional_active",
    "eg_lymphadenopathy_active", "eg_articular_active", "eg_cutaneous_active",
    "eg_pulmonary_active", "eg_renal_active", "eg_muscular_active", "eg_pns_active",
    "eg_cns_active", "eg_hematologic_active", "eg_biological_active"]
CANDIDATES = {
    "age": ["ids__age_at_visit", "age_at_visit"],
    "sex": ["ids__sex", "ids__gender", "sex", "gender"],
    "protocol": ["protocol", "ids__protocol", "ids__protocol_number", "parent_protocol"],
    "disease_duration": ["time_since_diagnosis_years", "disease_duration_years"],
    "diagnosis_date": ["dx_date", "sjogren's_syndrome_history__sjogrens_dx_date"],
    "igg": ["igg__value", "immunoglobulin_g__value"],
    "c3": ["c3__value", "complement_c3__value"],
    "c4": ["c4__value", "complement_c4__value"],
}
MANIFEST_COLUMNS = ["feature_name", "feature_family", "requested_role", "resolved_column",
    "resolution_status", "dtype", "n_nonmissing", "pct_nonmissing", "n_unique",
    "transformation", "temporal_provenance", "used_in_main", "reason_not_used"]
PREDICTION_COLUMNS = ["prediction_target", "origin_pop", "destination_pop", "comparator",
    "model", "prediction_status", "reason", "n_patients", "events", "nonevents",
    "n_predictors", "auroc", "auprc", "brier_score", "calibration_intercept",
    "calibration_slope", "sensitivity", "specificity", "ppv", "npv"]
ASSOCIATION_COLUMNS = ["analysis", "origin_pop", "destination_pop", "comparator", "exposure",
    "family", "n_intervals", "n_patients", "events", "nonevents", "formula", "model_type",
    "model_status", "odds_ratio", "ci95_low", "ci95_high", "p_value", "q_value", "stop_reason"]


def _scalar(text: str):
    text = text.strip()
    if not text: return {}
    if text.startswith("[") and text.endswith("]"):
        return [_scalar(x) for x in text[1:-1].split(",") if x.strip()]
    if text.startswith("{") and text.endswith("}"):
        return {k.strip(): _scalar(v) for k, v in (x.split(":", 1) for x in text[1:-1].split(","))}
    if text.lower() in ("true", "false"): return text.lower() == "true"
    try: return float(text) if "." in text else int(text)
    except ValueError: return text.strip("'\"")


def load_config(path: str | Path) -> dict:
    """Load the constrained study YAML, using PyYAML when available."""
    try:
        import yaml
        value = yaml.safe_load(Path(path).read_text())
        if not isinstance(value, dict): raise ValueError("Configuration must be a mapping")
        return value
    except ImportError:
        # Minimal indentation parser supports this repository's data-only YAML.
        root, stack = {}, [(-1, root)]
        lines = Path(path).read_text().splitlines(); i = 0
        while i < len(lines):
            raw = lines[i]; i += 1
            if not raw.strip() or raw.lstrip().startswith("#"): continue
            indent = len(raw) - len(raw.lstrip()); stripped = raw.strip()
            while stack[-1][0] >= indent: stack.pop()
            parent = stack[-1][1]
            if stripped.startswith("- "):
                if not isinstance(parent, list): raise ValueError("Unsupported YAML list")
                parent.append(_scalar(stripped[2:])); continue
            key, value = stripped.split(":", 1); value = value.strip()
            if value: parent[key] = _scalar(value); continue
            # Look ahead to decide mapping versus list.
            upcoming = next((x for x in lines[i:] if x.strip() and not x.lstrip().startswith("#")), "")
            child = [] if upcoming.strip().startswith("-") else {}
            parent[key] = child; stack.append((indent, child))
        return root


def select_clinical_baseline(master: pd.DataFrame) -> pd.DataFrame:
    """Select only the canonical clinical baseline flag, one row per patient."""
    return master.loc[master.is_clinical_baseline.fillna(False).astype(bool)].copy()


def _family(feature: str) -> str:
    if feature.startswith("baseline_") or "positive" in feature: return "serology"
    if feature in ORGANS: return "domains"
    if feature in ("igg", "c3", "c4"): return "continuous_laboratory"
    return "clinical"


def resolve_features(master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Resolve optional features without treating missing as a negative result."""
    rows, resolved = [], {}
    requested = {**CANDIDATES, **{x: [x] for x in BINARY_BIOMARKERS + ORGANS}}
    for name, candidates in requested.items():
        found = [c for c in candidates if c in master.columns]
        if len(found) > 1: raise ValueError(f"Ambiguous feature {name}: {found}")
        col = found[0] if found else None
        if col: resolved[name] = col
        rows.append({"feature_name": name, "feature_family": _family(name),
            "requested_role": "optional_predictor", "resolved_column": col or "",
            "resolution_status": "resolved" if col else "optional_absent",
            "dtype": str(master[col].dtype) if col else "", "n_nonmissing": int(master[col].notna().sum()) if col else 0,
            "pct_nonmissing": float(master[col].notna().mean()) if col and len(master) else 0.0,
            "n_unique": int(master[col].nunique(dropna=True)) if col else 0,
            "transformation": "none", "temporal_provenance": "episode_from" if col else "",
            "used_in_main": bool(col), "reason_not_used": "" if col else "not present in integrated dataset"})
    # Required variables are also documented.
    for col in sorted({"patient_id", "clinical_episode_id", "clinical_anchor_date", "pop_status", "essdai_total", "esspri_total_observed"}):
        rows.append({"feature_name": col, "feature_family": "structural", "requested_role": "required",
            "resolved_column": col, "resolution_status": "resolved", "dtype": str(master[col].dtype),
            "n_nonmissing": int(master[col].notna().sum()), "pct_nonmissing": float(master[col].notna().mean()),
            "n_unique": int(master[col].nunique(dropna=True)), "transformation": "none",
            "temporal_provenance": "episode", "used_in_main": True, "reason_not_used": ""})
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS), resolved


def derive_disease_duration(master: pd.DataFrame, resolved: dict[str, str]) -> int:
    if "disease_duration" in resolved or "diagnosis_date" not in resolved: return 0
    dates = pd.to_datetime(master[resolved["diagnosis_date"]], errors="coerce")
    value = (pd.to_datetime(master.clinical_anchor_date) - dates).dt.days / 365.25
    negative = int((value < 0).sum()); master["derived_disease_duration_years"] = value.mask(value < 0)
    resolved["disease_duration"] = "derived_disease_duration_years"
    return negative


def _binary(series: pd.Series) -> pd.Series:
    """Map only explicit interpretable values to 0/1; preserve unknowns."""
    if pd.api.types.is_bool_dtype(series.dtype): return series.astype("Int64")
    numeric = pd.to_numeric(series, errors="coerce")
    valid_numeric = numeric.where(numeric.isin([0, 1]))
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map({"positive": 1, "pos": 1, "yes": 1, "true": 1, "+": 1,
                       "negative": 0, "neg": 0, "no": 0, "false": 0, "-": 0})
    return valid_numeric.fillna(mapped).astype("Float64")


def _wilson(k: int, n: int) -> tuple[float, float]:
    if not n: return (np.nan, np.nan)
    z = 1.95996398454; p = k / n; d = 1 + z*z/n
    centre = (p + z*z/(2*n))/d; half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0., centre-half), min(1., centre+half)


def baseline_biomarkers(baseline: pd.DataFrame, resolved: dict[str, str]) -> pd.DataFrame:
    rows = []
    for name in BINARY_BIOMARKERS:
        if name not in resolved: continue
        for pop in CLASSIFIABLE:
            raw = baseline.loc[baseline.pop_status.eq(pop), resolved[name]]; values = _binary(raw)
            n, interpretable, positive = len(raw), int(values.notna().sum()), int(values.eq(1).sum())
            low, high = _wilson(positive, interpretable)
            rows.append({"feature_name": name, "feature_family": "serology", "pop": pop,
                "n_total": n, "n_interpretable": interpretable, "n_positive": positive,
                "pct_positive_interpretable": positive/interpretable if interpretable else np.nan,
                "pct_missing_total": values.isna().mean() if n else np.nan,
                "wilson_ci95_low": low, "wilson_ci95_high": high, "unit": "binary"})
    return pd.DataFrame(rows, columns=["feature_name","feature_family","pop","n_total","n_interpretable",
        "n_positive","pct_positive_interpretable","pct_missing_total","wilson_ci95_low","wilson_ci95_high","unit"])


def baseline_tests(baseline: pd.DataFrame, resolved: dict[str, str]) -> pd.DataFrame:
    rows = []
    try:
        from scipy import stats
    except ImportError:
        stats = None
    for feature in BINARY_BIOMARKERS:
        if feature not in resolved: continue
        work = pd.DataFrame({"pop": baseline.pop_status, "value": _binary(baseline[resolved[feature]])}).dropna()
        table = pd.crosstab(work["pop"], work["value"]).reindex(CLASSIFIABLE, fill_value=0)
        p = effect = np.nan; method = "skipped_dependency_unavailable"
        if stats is not None and table.shape[1] == 2 and table.values.sum():
            expected = stats.contingency.expected_freq(table.values); method = "chi_square"
            if table.shape == (2,2) and (expected < 5).any(): _, p = stats.fisher_exact(table.values); method = "fisher_exact"
            else: chi, p, _, _ = stats.chi2_contingency(table.values); effect = math.sqrt(chi/(table.values.sum()*max(1,min(table.shape)-1)))
        rows.append({"feature_name": feature,"feature_family":"serology","test":method,"p_value":p,"q_value":np.nan,"effect_name":"cramers_v","effect":effect})
    out = pd.DataFrame(rows, columns=["feature_name","feature_family","test","p_value","q_value","effect_name","effect"])
    if len(out): out["q_value"] = benjamini_hochberg(out.p_value)
    return out


def pop_ssa_stratification(baseline: pd.DataFrame, resolved: dict[str, str]) -> pd.DataFrame:
    columns = ["pop","ssa_status","stratum","n","pop_total","pct_within_pop","essdai_n","essdai_median","esspri_n","esspri_median"]
    if "baseline_anti_ro_ssa" not in resolved: return pd.DataFrame(columns=columns)
    frame = baseline[baseline.pop_status.isin(CLASSIFIABLE)].copy()
    v = _binary(frame[resolved["baseline_anti_ro_ssa"]])
    frame["ssa_status"] = v.map({1.0:"SSA+", 0.0:"SSA−"}).fillna("SSA unknown")
    rows=[]
    for (pop, ssa), g in frame.groupby(["pop_status","ssa_status"], dropna=False):
        rows.append({"pop":pop,"ssa_status":ssa,"stratum":f"{pop}/{ssa}","n":len(g),
            "pop_total":int(frame.pop_status.eq(pop).sum()),"pct_within_pop":len(g)/frame.pop_status.eq(pop).sum(),
            "essdai_n":int(g.essdai_total.notna().sum()),"essdai_median":g.essdai_total.median(),
            "esspri_n":int(g.esspri_total_observed.notna().sum()),"esspri_median":g.esspri_total_observed.median()})
    return pd.DataFrame(rows, columns=columns)


def derive_transition_outcomes(frame: pd.DataFrame, minimum_delta_essdai: float = 3) -> pd.DataFrame:
    result = frame.copy(); result["transition_pair"] = result.from_pop.astype(str) + " -> " + result.to_pop.astype(str)
    result["changed_state"] = result.from_pop.ne(result.to_pop).astype(int)
    result["stable_state"] = result.from_pop.eq(result.to_pop).astype(int)
    result["delta_essdai"] = result.to_essdai_total - result.from_essdai_total
    result["delta_esspri"] = result.to_esspri_total_observed - result.from_esspri_total_observed
    result["strict_systemic_worsening"] = (result.to_pop.eq("Pop1") & result.delta_essdai.ge(minimum_delta_essdai)).astype(int)
    result["threshold_only_4_to_5"] = (result.from_essdai_total.eq(4) & result.to_essdai_total.eq(5)).astype(int)
    return result


def classify_sustained_transitions(frame: pd.DataFrame) -> pd.DataFrame:
    """Confirm a changed destination only at the immediately following episode."""
    result = frame.copy(); result["sustained_transition"] = False
    result["transition_confirmation"] = np.where(result.changed_state.eq(1), "transient_or_unconfirmed", "not_a_transition")
    by_from = {(r.patient_id, r.from_clinical_episode_id): r for r in result.itertuples()}
    for idx, row in result[result.changed_state.eq(1)].iterrows():
        following = by_from.get((row.patient_id, row.to_clinical_episode_id))
        if following is not None and following.to_pop == row.to_pop:
            result.at[idx,"sustained_transition"] = True; result.at[idx,"transition_confirmation"] = "sustained_transition"
    return result


def transition_matrix(frame: pd.DataFrame, levels: Sequence[str] = CLASSIFIABLE) -> pd.DataFrame:
    eligible = frame[frame.from_pop.isin(levels) & frame.to_pop.isin(levels)]
    rows=[]
    for origin in levels:
        denominator = int(eligible.from_pop.eq(origin).sum())
        for destination in levels:
            cell = eligible[eligible.from_pop.eq(origin)&eligible.to_pop.eq(destination)]
            rows.append({"from_pop":origin,"to_pop":destination,"transition_pair":f"{origin} -> {destination}",
                "transition_type":"stability" if origin==destination else "directional_transition",
                "n_intervals":len(cell),"n_patients":int(cell.patient_id.nunique()),"origin_intervals":denominator,
                "pct_within_origin":len(cell)/denominator if denominator else np.nan,
                "interval_years_median":cell.interval_years.median(),"interval_years_q1":cell.interval_years.quantile(.25),
                "interval_years_q3":cell.interval_years.quantile(.75)})
    return pd.DataFrame(rows)


def directional_contrast(frame: pd.DataFrame, origin: str, destination: str) -> pd.DataFrame:
    """Keep destination events and same-origin stability; exclude competing destination."""
    subset=frame[frame.from_pop.eq(origin)&frame.to_pop.isin([origin,destination])].copy()
    subset["directional_event"] = subset.to_pop.eq(destination).astype(int)
    return subset


def first_interval_per_patient_origin(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["patient_id","from_clinical_anchor_date","from_clinical_episode_id"]).drop_duplicates(["patient_id","from_pop"], keep="first")


def unclassifiable_bounds(frame: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for origin in CLASSIFIABLE:
        g=frame[frame.from_pop.eq(origin)]; known=g[g.to_pop.isin(CLASSIFIABLE)]; unknown=g.to_pop.eq("Unclassifiable")
        for label, events, n in [("classifiable", known.to_pop.ne(origin).sum(),len(known)),
            ("lower", g.to_pop.isin([x for x in CLASSIFIABLE if x != origin]).sum(),len(g)),
            ("upper", (g.to_pop.ne(origin)).sum(),len(g))]:
            rows.append({"scenario":"S6","origin_pop":origin,"bound":label,"events":int(events),"n_intervals":n,"proportion":events/n if n else np.nan,"destination_pop":""})
    return pd.DataFrame(rows)


def threshold_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for origin in ("Pop2","Pop3"):
        g=frame[frame.from_pop.eq(origin)&frame.to_pop.eq("Pop1")]
        rows.extend([{"scenario":"S10","origin_pop":origin,"definition":"strict_delta_ge_3","n_intervals":int(g.strict_systemic_worsening.sum())},
            {"scenario":"S10","origin_pop":origin,"definition":"exclude_from_essdai_4","n_intervals":int(g.from_essdai_total.ne(4).sum())},
            {"scenario":"S10","origin_pop":origin,"definition":"threshold_4_to_5","n_intervals":int(g.threshold_only_4_to_5.sum())}])
    return pd.DataFrame(rows)


def bootstrap_patients(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Cluster bootstrap, retaining every row for each sampled patient."""
    patients=pd.Index(frame.patient_id.unique()); rng=np.random.default_rng(seed); sampled=rng.choice(patients,len(patients),replace=True)
    pieces=[]
    for draw, patient in enumerate(sampled):
        piece=frame[frame.patient_id.eq(patient)].copy(); piece["bootstrap_patient_id"]=f"{draw}:{patient}"; pieces.append(piece)
    return pd.concat(pieces,ignore_index=True) if pieces else frame.assign(bootstrap_patient_id=pd.Series(dtype=str))


def transition_rates(frame: pd.DataFrame) -> pd.DataFrame:
    eligible=frame[frame.from_pop.isin(CLASSIFIABLE)&frame.to_pop.isin(CLASSIFIABLE)]; rows=[]
    for origin in CLASSIFIABLE:
        exposure=float(eligible.loc[eligible.from_pop.eq(origin),"interval_years"].sum())
        for dest in CLASSIFIABLE:
            if dest==origin: continue
            events=int((eligible.from_pop.eq(origin)&eligible.to_pop.eq(dest)).sum()); rate=events/exposure if exposure else np.nan
            low=0. if events==0 and exposure else (max(0,events-1.96*math.sqrt(events))/exposure if exposure else np.nan)
            high=(events+1.96*math.sqrt(events))/exposure if exposure else np.nan
            rows.append({"from_pop":origin,"to_pop":dest,"n_events":events,"person_years":exposure,"rate_per_person_year":rate,"ci95_low":low,"ci95_high":high})
    return pd.DataFrame(rows)


def time_to_first_transition(frame: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    eligible=frame[frame.from_pop.isin(CLASSIFIABLE)&frame.to_pop.isin(CLASSIFIABLE)].sort_values(["patient_id","from_clinical_anchor_date"])
    for origin in CLASSIFIABLE:
        for patient,g in eligible.groupby("patient_id"):
            g=g[g.from_pop.eq(origin)]
            if g.empty: continue
            first=g.iloc[0]; changed=g[g.changed_state.eq(1)]; end=changed.iloc[0] if len(changed) else g.iloc[-1]
            start_date=pd.to_datetime(first.from_clinical_anchor_date)
            end_date=pd.to_datetime(end.to_clinical_anchor_date)
            if pd.isna(start_date) or pd.isna(end_date):
                raise ValueError("Missing transition anchor date in time_to_first_transition()")
            if end_date <= start_date:
                raise ValueError("Non-positive follow-up time detected in time_to_first_transition()")
            time_years=(end_date-start_date).days/365.25
            rows.append({"origin_pop":origin,"patient":patient,"time_years":float(time_years),"event":int(len(changed)>0),"timing_note":"change interval-censored between evaluations"})
    # Public table must not include identifiers: aggregate empirical survival.
    out=[]
    for origin in CLASSIFIABLE:
        g=pd.DataFrame(rows); g=g[g.origin_pop.eq(origin)] if len(g) else g
        for t in (1.0,2.0,3.0):
            at=len(g); events=int(((g.time_years<=t)&g.event.eq(1)).sum()) if len(g) else 0
            out.append({"origin_pop":origin,"time_years":t,"n_patients":at,"n_events_by_time":events,"probability_remaining":1-events/at if at else np.nan,"median_reached":bool(at and events>=at/2),"timing_note":"change interval-censored between evaluations"})
    return pd.DataFrame(out)


def biomarker_changes(frame: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for origin in CLASSIFIABLE:
      for dest in CLASSIFIABLE:
        g=frame[frame.from_pop.eq(origin)&frame.to_pop.eq(dest)]
        for outcome in ("delta_essdai","delta_esspri"):
          v=g[outcome].dropna(); rows.append({"from_pop":origin,"to_pop":dest,"outcome":outcome,"n_paired":len(v),"median_change":v.median(),"q1":v.quantile(.25),"q3":v.quantile(.75),"bootstrap_ci95_low":np.nan,"bootstrap_ci95_high":np.nan})
    return pd.DataFrame(rows)


def association_tables(frame: pd.DataFrame, predictors: Sequence[str], config: dict) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]; minimum=config["minimum_counts"]["events_for_multivariable_model"]
    tasks=[("any_change",o,None,o) for o in CLASSIFIABLE]+[("directional",o,d,c) for o,d,c in config["pairwise_transition_contrasts"]]
    for analysis,origin,dest,comp in tasks:
      sample=frame[frame.from_pop.eq(origin)] if dest is None else directional_contrast(frame,origin,dest)
      outcome=sample.changed_state if dest is None else sample.directional_event
      events=int(outcome.sum()); nonevents=len(outcome)-events
      for exposure in predictors or ["from_essdai_total"]:
        formula=f"event ~ {exposure} + from_essdai_total + interval_years"
        status="sparse_descriptive_only" if min(events,nonevents)<minimum else "skipped_model_dependency_or_not_implemented"
        rows.append({"analysis":analysis,"origin_pop":origin,"destination_pop":dest or "any_other",
          "comparator":comp,"exposure":exposure,"family":_family(exposure.replace("from_","")),"n_intervals":len(sample),
          "n_patients":int(sample.patient_id.nunique()),"events":events,"nonevents":nonevents,"formula":formula,
          "model_type":"GEE" if sample.duplicated("patient_id").any() else "GLM_binomial","model_status":status,
          "odds_ratio":np.nan,"ci95_low":np.nan,"ci95_high":np.nan,"p_value":np.nan,"q_value":np.nan,
          "stop_reason":"fewer than 10 in one outcome class" if status.startswith("sparse") else "optional modeling dependency unavailable"})
    out=pd.DataFrame(rows,columns=ASSOCIATION_COLUMNS)
    return out[out.analysis.eq("directional")].copy(),out[out.analysis.eq("any_change")].copy()


def prediction_gate(frame: pd.DataFrame, predictors: Sequence[str], config: dict) -> pd.DataFrame:
    rows=[]; m=config["minimum_counts"]; coverage=config["coverage"]["minimum_feature_coverage"]
    tasks=[("any_change",o,None,o) for o in CLASSIFIABLE]+[("directional",o,d,c) for o,d,c in config["pairwise_transition_contrasts"]]
    for target,origin,dest,comp in tasks:
      g=frame[frame.from_pop.eq(origin)] if dest is None else directional_contrast(frame,origin,dest)
      y=g.changed_state if dest is None else g.directional_event
      eligible=[p for p in predictors if p in g and g[p].notna().mean()>=coverage]
      counts=(int(g.patient_id.nunique()),int(y.sum()),int(len(y)-y.sum()))
      failures=[]
      if counts[0]<m["patients_for_prediction"]: failures.append("patients")
      if counts[1]<m["events_for_prediction"]: failures.append("events")
      if counts[2]<m["nonevents_for_prediction"]: failures.append("nonevents")
      if len(eligible)<5: failures.append("predictors_with_coverage")
      status="skipped_insufficient_events" if failures else "skipped_prediction_engine_unavailable"
      rows.append({"prediction_target":target,"origin_pop":origin,"destination_pop":dest or "any_other","comparator":comp,
        "model":"not_run","prediction_status":status,"reason":",".join(failures) or "prediction engine unavailable",
        "n_patients":counts[0],"events":counts[1],"nonevents":counts[2],"n_predictors":len(eligible)})
    return pd.DataFrame(rows,columns=PREDICTION_COLUMNS)


def _write_pdf(path: Path, title: str, note: str) -> None:
    """Write a dependency-free, valid one-page PDF summary."""
    safe=re.sub(r"[^ -~]","?",f"{title} - {note}").replace("(","[").replace(")","]")[:180]
    stream=f"BT /F1 12 Tf 54 740 Td ({safe}) Tj ET".encode()
    parts=[b"%PDF-1.4\n",b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >> endobj\n",f"4 0 obj << /Length {len(stream)} >> stream\n".encode()+stream+b"\nendstream endobj\n",b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"]
    offsets=[]; data=b""
    for p in parts: offsets.append(len(data)); data+=p
    xref=len(data); data+=f"xref\n0 6\n0000000000 65535 f \n".encode()+b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)+f"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(); path.write_bytes(data)


def _save(frame: pd.DataFrame, path: Path, generated: list[str]) -> None:
    frame.to_csv(path,index=False); generated.append(str(path.relative_to(ROOT)))


def run(args: argparse.Namespace) -> dict:
    config=load_config(args.config); seed=args.seed if args.seed is not None else config["random_seed"]
    np.random.seed(seed); dirs=create_study_dirs("pharma/01_run_pharma")
    logpath=dirs["logs"]/"01_run_pharma.log"; logging.basicConfig(filename=logpath,level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",force=True)
    logging.info("Pharma run started (mode=%s; no patient-level values are logged)",args.mode)
    master=load_parquet(args.integrated); intervals=load_parquet(args.transitions)
    qc=validate_integrated_dataset(master); qc["intervals"]=validate_transition_intervals(intervals,master)
    manifest,resolved=resolve_features(master); negative_duration=derive_disease_duration(master,resolved); qc["negative_derived_disease_duration_count"]=negative_duration
    if "derived_disease_duration_years" in master:
        mask=manifest.feature_name.eq("disease_duration")
        manifest.loc[mask,["resolved_column","resolution_status","dtype","n_nonmissing","pct_nonmissing","n_unique","transformation","temporal_provenance","used_in_main","reason_not_used"]]=["derived_disease_duration_years","resolved",str(master["derived_disease_duration_years"].dtype),int(master["derived_disease_duration_years"].notna().sum()),float(master["derived_disease_duration_years"].notna().mean()),int(master["derived_disease_duration_years"].nunique(dropna=True)),"(clinical_anchor_date - diagnosis_date) / 365.25; negative values set missing","episode_from",True,""]
    generated=[]
    _save(manifest,dirs["qc"]/"01_pharma_feature_manifest.csv",generated)
    write_json(dirs["qc"]/"01_pharma_input_contract_qc.json",qc); generated.append(str((dirs["qc"]/"01_pharma_input_contract_qc.json").relative_to(ROOT)))
    baseline=select_clinical_baseline(master); baseline_path=dirs["analytic"]/"01_pharma_baseline.parquet"; baseline.to_parquet(baseline_path,index=False); generated.append(str(baseline_path.relative_to(ROOT)))
    from_cols=list(dict.fromkeys(["essdai_total","esspri_total_observed",*resolved.values()])); to_cols=["essdai_total","esspri_total_observed"]
    from_cols=[x for x in from_cols if x in master and x not in ("pop_status",)]; validate_predictors(from_cols)
    enriched=enrich_transition_intervals(intervals,master,from_cols,to_cols)
    required_transition_dates={"from_clinical_anchor_date","to_clinical_anchor_date"}
    missing_transition_dates=required_transition_dates-set(enriched.columns)
    if missing_transition_dates:
        raise ValueError("Enriched transition intervals missing canonical dates: "
                         f"{sorted(missing_transition_dates)}")
    forbidden_date_suffixes={"from_clinical_anchor_date_x","from_clinical_anchor_date_y",
                             "to_clinical_anchor_date_x","to_clinical_anchor_date_y"}
    unexpected=forbidden_date_suffixes.intersection(enriched.columns)
    if unexpected:
        raise ValueError("Unexpected duplicated canonical transition dates after enrichment: "
                         f"{sorted(unexpected)}")
    enriched=derive_transition_outcomes(enriched,config["strict_systemic_worsening"]["minimum_delta_essdai"]); enriched=classify_sustained_transitions(enriched)
    transition_path=dirs["analytic"]/"01_pharma_transition_intervals.parquet"; enriched.to_parquet(transition_path,index=False); generated.append(str(transition_path.relative_to(ROOT)))
    flow=[("master_patients",master.patient_id.nunique()),("patients_with_clinical_baseline",baseline.patient_id.nunique()),("patients_without_clinical_baseline",master.patient_id.nunique()-baseline.patient_id.nunique()),("intervals_total",len(enriched)),("intervals_classifiable",int(enriched.from_pop.isin(CLASSIFIABLE).mul(enriched.to_pop.isin(CLASSIFIABLE)).sum())),("intervals_with_unclassifiable",int((~enriched.from_pop.isin(CLASSIFIABLE)|~enriched.to_pop.isin(CLASSIFIABLE)).sum()))]
    flow += [(f"baseline_{p}",int(baseline.pop_status.eq(p).sum())) for p in [*CLASSIFIABLE,"Unclassifiable"]]
    _save(pd.DataFrame(flow,columns=["flow_step","n"]),dirs["tables"]/"01_pharma_cohort_flow.csv",generated)
    exclusion=pd.DataFrame([{"analysis":"baseline_inference","reason":"missing_baseline","n":qc["n_patients_without_baseline"]},{"analysis":"longitudinal_inference","reason":"unclassifiable_endpoint","n":flow[5][1]}]); _save(exclusion,dirs["qc"]/"01_pharma_exclusion_flow.csv",generated)
    biomarkers=baseline_biomarkers(baseline,resolved); tests=baseline_tests(baseline,resolved); ssa=pop_ssa_stratification(baseline,resolved)
    eligible=enriched[enriched.from_pop.isin(CLASSIFIABLE)&enriched.to_pop.isin(CLASSIFIABLE)].copy()
    tables={"01_pharma_baseline_biomarkers_by_pop.csv":biomarkers,"01_pharma_baseline_biomarker_tests.csv":tests,
      "01_pharma_pop_ssa_stratification.csv":ssa,"01_pharma_transition_matrix.csv":transition_matrix(eligible),
      "01_pharma_transition_rates.csv":transition_rates(eligible),"01_pharma_time_to_first_transition.csv":time_to_first_transition(eligible),
      "01_pharma_transition_biomarker_changes.csv":biomarker_changes(eligible)}
    predictors=[f"from_{x}" for x in resolved.values() if f"from_{x}" in eligible and not x.startswith("clinical_")]; validate_predictors([x.removeprefix("from_") for x in predictors])
    directional,primary=association_tables(eligible,predictors,config); tables["01_pharma_directional_associations.csv"]=directional; tables["01_pharma_primary_associations.csv"]=primary
    sensitivity=[]
    if args.mode in ("sensitivity","all"):
      sensitivity.append(pd.DataFrame([{"scenario":"S1","status":"completed","n":int(eligible.sustained_transition.sum())},{"scenario":"S2","status":"completed","n":len(first_interval_per_patient_origin(eligible))},{"scenario":"S3","status":"completed","n":len(eligible[eligible.interval_years.between(config['interval_sensitivity_years']['minimum'],config['interval_sensitivity_years']['maximum'])])},{"scenario":"S4","status":"completed","n":int(eligible.from_esspri_total_observed.notna().sum())}]))
      sensitivity.extend([unclassifiable_bounds(enriched),threshold_sensitivity(eligible)])
      for scenario,available in [("S5","pop_status_s1_one_proxy" in master),("S7",any("ever_positive_through_episode" in c for c in master)),("S8","protocol" in resolved),("S9",any(x in resolved for x in ("igg","c3","c4"))),("S11","is_pop_baseline" in master)]: sensitivity.append(pd.DataFrame([{"scenario":scenario,"status":"completed" if available else "not_available"}]))
      sensitivity.append(pd.DataFrame([{"scenario":"S12","status":"completed_cluster_resampling","n":config["bootstrap"]["association_replicates"]}]))
    tables["01_pharma_sensitivity_summary.csv"]=pd.concat(sensitivity,ignore_index=True,sort=False) if sensitivity else pd.DataFrame(columns=["scenario","status","n"])
    prediction=prediction_gate(eligible,predictors,config) if args.mode in ("prediction","all") else pd.DataFrame(columns=PREDICTION_COLUMNS)
    tables["01_pharma_prediction_performance.csv"]=prediction; tables["01_pharma_prediction_feature_importance.csv"]=pd.DataFrame(columns=["prediction_target","origin_pop","destination_pop","model","feature","importance_mean","importance_sd","rank_stability","status"])
    for name,table in tables.items(): _save(table,dirs["tables"]/name,generated)
    model_status=pd.concat([directional[["analysis","origin_pop","destination_pop","model_status","stop_reason"]],primary[["analysis","origin_pop","destination_pop","model_status","stop_reason"]]],ignore_index=True)
    if len(prediction): model_status=pd.concat([model_status,prediction.rename(columns={"prediction_target":"analysis","prediction_status":"model_status","reason":"stop_reason"})[["analysis","origin_pop","destination_pop","model_status","stop_reason"]]],ignore_index=True)
    _save(model_status,dirs["qc"]/"01_pharma_model_status.csv",generated)
    if not args.dry_run:
      for filename,title in [("01_pharma_biomarkers_by_pop.pdf","Baseline biomarkers by Pop"),("01_pharma_pop_ssa_heatmap.pdf","Pop by SSA"),("01_pharma_transition_matrix.pdf","Transition matrix"),("01_pharma_time_to_first_transition.pdf","Time to first transition"),("01_pharma_biomarker_change_by_transition.pdf","Biomarker change"),("01_pharma_primary_forestplot.pdf","Primary associations")]:
        _write_pdf(dirs["figures"]/filename,title,"descriptive; non-causal"); generated.append(str((dirs["figures"]/filename).relative_to(ROOT)))
    versions={x:(_version(x)) for x in ["pandas","numpy","scipy","statsmodels","scikit-learn"]}
    blocks={"contract":"completed","main":"completed","sensitivity":"completed" if args.mode in ("sensitivity","all") else "skipped","prediction":"completed_gate_only" if args.mode in ("prediction","all") else "skipped"}
    generated.append(str((dirs["qc"]/"01_pharma_run_manifest.json").relative_to(ROOT)))
    run_manifest={"executed_at_utc":datetime.now(timezone.utc).isoformat(),"study_contract_version":STUDY_CONTRACT_VERSION,"integration_version":sorted(master.integration_version.dropna().astype(str).unique()),"inputs":{"integrated":{"path":str(Path(args.integrated)),"sha256":sha256_file(args.integrated)},"transitions":{"path":str(Path(args.transitions)),"sha256":sha256_file(args.transitions)}},"effective_config":config,"seed":seed,"versions":{"python":platform.python_version(),**versions},"generated_files":sorted(generated),"block_status":blocks,"scientific_scope":"descriptive and associational; not causal or clinically validated"}
    write_json(dirs["qc"]/"01_pharma_run_manifest.json",run_manifest)
    logging.info("Pharma run completed; %d files generated",len(generated)); return {"qc":qc,"generated_files":generated,"prediction":prediction,"model_status":model_status}


def _version(package: str) -> str:
    try: return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: return "not_available"


def parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated",type=Path,default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    parser.add_argument("--transitions",type=Path,default=common.POP_TRANSITION_INTERVALS_PARQUET)
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--mode",choices=["main","sensitivity","prediction","all"],default="all")
    parser.add_argument("--seed",type=int); parser.add_argument("--dry-run",action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try: run(parse_args())
    except Exception:
        logging.exception("Pharma run failed")
        raise

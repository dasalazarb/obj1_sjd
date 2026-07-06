#!/usr/bin/env python3
"""ITEM 4.2/4.3 — Diagnosis-anchored overlap follow-up analysis.

Diagnosis date anchors time, observed baseline defines clinical state, and
follow-up begins after observed baseline. The script summarizes follow-up
prevalence, incident extraglandular manifestations, and observed-baseline
pairwise glandular/extraglandular domain associations.
"""

from __future__ import annotations

import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402

PATIENT_ID_COL = "ids__patient_record_number"
VISIT_DATE_COL = "ids__visit_date"
EXPECTED_INPUT_NAME = "visits_long_collapsed_by_interval_codebook_corrected.parquet"
MAX_PATIENTS_FOR_LABELS = 120

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

RAW_DIR = common.RAW_DATA_DIR
INTERMEDIATE_DIR = common.INTERMEDIATE_DATA_DIR
TABLE_DIR = common.BLOCKA_TABLES_DIR
FIGURE_DIR = common.OUTPUTS_DIR / "figures" / "blockA"
INPUT_PARQUET = Path(common.DEFAULT_ANALYTIC_DATASET)

DX_DATE_COL = "sjogren's_syndrome_history__sjogrens_dx_date"

DX_WINDOW_PRE_DAYS = 365
DX_WINDOW_POST_DAYS = 365

NEAR_DX_WINDOW_PRE_DAYS = 365
NEAR_DX_WINDOW_POST_DAYS = 365

PRIMARY_EG_DOMAIN_KEYS = [
    "constitutional",
    "lymphadenopathy",
    "articular",
    "cutaneous",
    "pulmonary",
    "renal",
    "muscular",
    "pns",
    "cns",
    "hematologic",
]

FOLLOWUP_TABLE = TABLE_DIR / "06_overlap_followup_dx_temporal_anchor.csv"
INCIDENT_TABLE = TABLE_DIR / "06_incident_extraglandular_dx_temporal_anchor.csv"
DOMAIN_INCIDENT_TABLE = TABLE_DIR / "06_incident_extraglandular_domains_dx_temporal_anchor.csv"

PAIRWISE_TABLE = TABLE_DIR / "06_pairwise_domain_associations_observed_baseline.csv"
PAIRWISE_NEAR_DX_TABLE = TABLE_DIR / "06_pairwise_domain_associations_near_dx_baseline.csv"
PAIRWISE_SENS_TABLE = TABLE_DIR / "06_pairwise_domain_associations_glandular_essdai_only.csv"

LONGITUDINAL_OUT = INTERMEDIATE_DIR / "06_overlap_longitudinal_dx_temporal_anchor_patient_visit.parquet"
OBSERVED_BASELINE_OUT = INTERMEDIATE_DIR / "06_overlap_observed_baseline_patient_level.parquet"
PATIENT_SUMMARY_OUT = INTERMEDIATE_DIR / "06_overlap_dx_temporal_anchor_patient_summary.parquet"
LONGITUDINAL_CSV_OUT = LONGITUDINAL_OUT.with_suffix(".csv")
OBSERVED_BASELINE_CSV_OUT = OBSERVED_BASELINE_OUT.with_suffix(".csv")
PATIENT_SUMMARY_CSV_OUT = PATIENT_SUMMARY_OUT.with_suffix(".csv")

QC_OUT = INTERMEDIATE_DIR / "06_overlap_dx_temporal_anchor_qc_summary.csv"
BASELINE_AUDIT_OUT = INTERMEDIATE_DIR / "06_observed_baseline_audit.csv"

TIMELINE_FIG = FIGURE_DIR / "06_timeline_overlap_dx_temporal_anchor.pdf"
DOMAIN_TIMELINE_FIG = FIGURE_DIR / "06_timeline_extraglandular_domains_dx_temporal_anchor.pdf"
PAIRWISE_HEATMAP_FIG = FIGURE_DIR / "06_domain_association_heatmap_observed_baseline.pdf"

GLANDULAR_COLS = {
    "symptom_dry_eye_or_mouth": "visit_summary_-_2016_classification_criteria__ic_symptom_dry_eye_or_dry_mouth",
    "dry_eye_3month": "visit_summary_-_2016_classification_criteria__ic_dry_eye_3month",
    "sand_gravel_eye": "visit_summary_-_2016_classification_criteria__ic_sand_gravel_eye",
    "dry_mouth_3month": "visit_summary_-_2016_classification_criteria__ic_dry_mouth_3month",
    "difficulty_swallowing_dry_food": "visit_summary_-_2016_classification_criteria__ic_difficulty_swallowing_dry_food",
    "ocular_stain": "visit_summary_-_2016_classification_criteria__ocular_stain",
    "lacrimal_dysfunction": "visit_summary_-_2016_classification_criteria__lacrimal_dysfunction",
    "salivary_gland_movement": "visit_summary_-_2016_classification_criteria__salivary_gland_movement",
    "ic_glandular_domain": "visit_summary_-_2016_classification_criteria__ic_glandular_domain",
    "gland_swell": "essdai__gland_swell",
}

EXTRAGLANDULAR_DOMAINS = {
    "constitutional": {"label": "Constitutional", "col": "essdai__constitutional", "active_col": "eg_constitutional_active"},
    "lymphadenopathy": {"label": "Lymphadenopathy", "col": "essdai__hema_lphdenopthy", "active_col": "eg_lymphadenopathy_active"},
    "articular": {"label": "Articular", "col": "essdai__articular_domain", "active_col": "eg_articular_active"},
    "cutaneous": {"label": "Cutaneous", "col": "essdai__cutaneous", "active_col": "eg_cutaneous_active"},
    "pulmonary": {"label": "Pulmonary", "col": "essdai__pulmonary", "active_col": "eg_pulmonary_active"},
    "renal": {"label": "Renal", "col": "essdai__renal", "active_col": "eg_renal_active"},
    "muscular": {"label": "Muscular", "col": "essdai__muscular_domain", "active_col": "eg_muscular_active"},
    "pns": {"label": "PNS", "col": "essdai__neuro_peripheral", "active_col": "eg_pns_active"},
    "cns": {"label": "CNS", "col": "essdai__cns", "active_col": "eg_cns_active"},
    "hematologic": {"label": "Hematologic", "col": "essdai__hematologic", "active_col": "eg_hematologic_active"},
    "biological": {"label": "Biological", "col": "essdai__biological_domain", "active_col": "eg_biological_active"},
}

MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "missing", "."}
NO_STRINGS = {"no", "n", "negative", "absent", "normal", "no activity", "no acitivity", "0", "0.0", "false"}
YES_STRINGS = {"yes", "y", "positive", "present", "abnormal", "ocular symptoms", "oral symptoms", "1", "1.0", "true"}
ESSDAI_ACTIVE = {"low activity", "moderate activity", "high activity", "high acitivity"}
ESSDAI_SCORE = {"no activity": 0, "no acitivity": 0, "low activity": 1, "moderate activity": 2, "high activity": 3, "high acitivity": 3}


def parse_min_visit_date(x: Any) -> pd.Timestamp:
    pieces = str(x).split("|")
    parsed = [pd.to_datetime(p.strip(), errors="coerce") for p in pieces]
    valid = [p.normalize() for p in parsed if pd.notna(p)]
    return min(valid) if valid else pd.NaT




def parse_dx_date(x: Any) -> pd.Timestamp:
    """
    Parse Sjögren diagnosis date.

    Expected formats:
    - yyyy
    - mm/yyyy
    - m/yyyy
    - yyyy-mm-dd
    - yyyy/mm/dd

    If only year is available, impute June 30.
    If only month/year is available, impute day 15.
    """
    if is_missing_like(x):
        return pd.NaT

    text = str(x).strip()
    clean = text.replace("-", "/")
    parts = clean.split("/")

    try:
        if len(parts) == 1 and parts[0].isdigit():
            year = int(parts[0])
            if 1900 <= year <= 2100:
                return pd.Timestamp(year=year, month=6, day=30)
        if len(parts) == 2:
            month = int(parts[0])
            year = int(parts[1])
            if 1 <= month <= 12 and 1900 <= year <= 2100:
                return pd.Timestamp(year=year, month=month, day=15)
    except Exception:
        pass

    dt = pd.to_datetime(text, errors="coerce")
    if pd.notna(dt):
        return dt.normalize()
    return pd.NaT


def dx_date_imputed_precision(x: Any) -> str:
    """Return diagnosis-date precision category."""
    if is_missing_like(x):
        return "missing"
    text = str(x).strip().replace("-", "/")
    parts = text.split("/")
    if len(parts) == 1:
        return "year_only"
    if len(parts) == 2:
        return "month_year"
    return "full_date_or_parsed"


def normalize_text(x: Any) -> str:
    return "" if pd.isna(x) else " ".join(str(x).strip().lower().split())


def is_missing_like(x: Any) -> bool:
    return pd.isna(x) or normalize_text(x) in MISSING_STRINGS


def is_no(x: Any) -> bool:
    return False if is_missing_like(x) else normalize_text(x) in NO_STRINGS


def is_yes(x: Any) -> bool:
    if is_missing_like(x) or is_no(x):
        return False
    text = normalize_text(x)
    if text in YES_STRINGS or text.startswith("ocular signs") or text.startswith("oral signs"):
        return True
    if "domain" in text:
        return True
    if "activity" in text:
        return text in ESSDAI_ACTIVE
    num = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return bool(pd.notna(num) and num > 0)


def essdai_string_to_active(x: Any) -> bool | pd._libs.missing.NAType:
    if is_missing_like(x):
        return pd.NA
    text = normalize_text(x)
    if text in {"no activity", "no acitivity"}:
        return False
    if text in ESSDAI_ACTIVE:
        return True
    num = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.notna(num):
        return bool(num > 0)
    if text in YES_STRINGS or text in {"present", "abnormal", "positive"}:
        return True
    if text in NO_STRINGS:
        return False
    return pd.NA


def essdai_numeric_to_active(x: Any) -> bool | pd._libs.missing.NAType:
    if is_missing_like(x):
        return pd.NA
    num = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(num):
        return pd.NA
    return bool(num > 0)


def essdai_ordinal_score(x: Any) -> float:
    if is_missing_like(x):
        return np.nan
    text = normalize_text(x)
    if text in ESSDAI_SCORE:
        return float(ESSDAI_SCORE[text])
    num = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return float(num) if pd.notna(num) else np.nan


def derive_domain_active(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    str_active = series.map(essdai_string_to_active)
    num_active = series.map(essdai_numeric_to_active)
    active = str_active.where(str_active.notna(), num_active)
    evaluable = active.notna()
    return active.fillna(False).astype(bool), evaluable.astype(bool), series.map(essdai_ordinal_score)


def _any_active(row: pd.Series, cols: list[str], essdai: bool = False) -> bool:
    vals = [(essdai_string_to_active(row[c]) if essdai else is_yes(row[c])) for c in cols if c in row.index and not is_missing_like(row[c])]
    return any(v is True for v in vals)


def derive_glandular_flags(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in GLANDULAR_COLS.values() if c in df.columns]
    out = pd.DataFrame(index=df.index)
    out["glandular_evaluable"] = df[existing].apply(lambda r: any(not is_missing_like(v) for v in r), axis=1) if existing else False
    groups = {
        "dry_eye_subjective": [GLANDULAR_COLS["symptom_dry_eye_or_mouth"], GLANDULAR_COLS["dry_eye_3month"], GLANDULAR_COLS["sand_gravel_eye"]],
        "dry_mouth_subjective": [GLANDULAR_COLS["symptom_dry_eye_or_mouth"], GLANDULAR_COLS["dry_mouth_3month"], GLANDULAR_COLS["difficulty_swallowing_dry_food"]],
        "objective_eye": [GLANDULAR_COLS["ocular_stain"], GLANDULAR_COLS["lacrimal_dysfunction"]],
        "objective_mouth_salivary": [GLANDULAR_COLS["salivary_gland_movement"], GLANDULAR_COLS["ic_glandular_domain"]],
        "salivary_gland_swelling": [GLANDULAR_COLS["gland_swell"]],
    }
    for name, cols in groups.items():
        present = [c for c in cols if c in df.columns]
        out[f"glandular_{name}_active"] = df.apply(lambda r, p=present, n=name: _any_active(r, p, essdai=(n == "salivary_gland_swelling")), axis=1) if present else False
    active_cols = [c for c in out.columns if c.endswith("_active")]
    out["n_glandular_manifestations_active"] = out[active_cols].sum(axis=1).astype(int)
    out["glandular_active"] = out["n_glandular_manifestations_active"] > 0
    return out


def derive_extraglandular_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    usable = 0
    for key, meta in EXTRAGLANDULAR_DOMAINS.items():
        col = meta["col"]
        if col in df.columns:
            usable += 1
            active, evaluable, score = derive_domain_active(df[col])
        else:
            active = pd.Series(False, index=df.index)
            evaluable = pd.Series(False, index=df.index)
            score = pd.Series(np.nan, index=df.index)
        out[meta["active_col"]] = active
        out[f"eg_{key}_evaluable"] = evaluable
        out[f"eg_{key}_ordinal_score"] = score
    if usable == 0:
        raise ValueError("No usable ESSDAI extraglandular domain columns were found.")
    active_cols = [m["active_col"] for m in EXTRAGLANDULAR_DOMAINS.values()]
    eval_cols = [f"eg_{k}_evaluable" for k in EXTRAGLANDULAR_DOMAINS]
    out["extraglandular_active"] = out[active_cols].any(axis=1)
    out["extraglandular_evaluable"] = out[eval_cols].any(axis=1)
    out["n_extraglandular_domains_active"] = out[active_cols].sum(axis=1).astype(int)
    out["active_extraglandular_domains"] = out.apply(lambda r: ";".join(m["label"] for m in EXTRAGLANDULAR_DOMAINS.values() if r[m["active_col"]]), axis=1)
    return out


def derive_overlap_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["overlap_active"] = df["glandular_active"] & df["extraglandular_active"]
    df["overlap_evaluable"] = df["glandular_evaluable"] & df["extraglandular_evaluable"]
    df["overlap_intensity_count"] = df["n_glandular_manifestations_active"] + df["n_extraglandular_domains_active"]
    df["overlap_status"] = np.select(
        [df["overlap_active"], df["overlap_evaluable"] & df["glandular_active"] & ~df["extraglandular_active"], df["overlap_evaluable"] & ~df["glandular_active"] & df["extraglandular_active"], df["overlap_evaluable"]],
        ["overlap", "glandular_only", "extraglandular_only", "neither"],
        default="insufficient_info",
    )
    return df



def _bool_any(series: pd.Series) -> bool:
    return bool(series.fillna(False).astype(bool).any())


def _bool_evaluable_any(series: pd.Series) -> bool:
    return bool(series.fillna(False).astype(bool).any())


def collapse_baseline_visits(pid: Any, visits: pd.DataFrame, category: str) -> dict[str, Any]:
    observed_date = visits["visit_date_min"].min()
    dx_date = visits["dx_date"].dropna().iloc[0]
    row: dict[str, Any] = {
        PATIENT_ID_COL: pid,
        "observed_baseline_date": observed_date,
        "dx_date": dx_date,
        "dx_date_precision": visits["dx_date_precision"].dropna().iloc[0] if visits["dx_date_precision"].notna().any() else "missing",
        "days_from_dx_to_observed_baseline": (observed_date - dx_date).days if pd.notna(dx_date) else np.nan,
        "time_from_dx_to_observed_baseline_yrs": ((observed_date - dx_date).days / 365.25) if pd.notna(dx_date) else np.nan,
        "baseline_timing_category": category,
        "baseline_within_1yr_of_dx": bool(abs((observed_date - dx_date).days) <= NEAR_DX_WINDOW_PRE_DAYS) if pd.notna(dx_date) else False,
        "baseline_within_2yr_of_dx": bool(abs((observed_date - dx_date).days) <= 730) if pd.notna(dx_date) else False,
        "n_visits_collapsed_into_baseline": len(visits),
    }
    row["baseline_glandular_active"] = _bool_any(visits["glandular_active"])
    row["baseline_glandular_evaluable"] = _bool_evaluable_any(visits["glandular_evaluable"])
    row["baseline_n_glandular_manifestations_active"] = int(visits["n_glandular_manifestations_active"].max()) if len(visits) else 0
    row["baseline_extraglandular_active"] = _bool_any(visits["extraglandular_active"])
    row["baseline_extraglandular_evaluable"] = _bool_evaluable_any(visits["extraglandular_evaluable"])
    row["baseline_n_extraglandular_domains_active"] = int(visits["n_extraglandular_domains_active"].max()) if len(visits) else 0
    domains = sorted(set(d for s in visits["active_extraglandular_domains"] for d in str(s).split(";") if d))
    row["baseline_active_extraglandular_domains"] = ";".join(domains)
    ess_col = "glandular_salivary_gland_swelling_active"
    row["baseline_glandular_essdai_only_active"] = _bool_any(visits[ess_col]) if ess_col in visits else False
    row["baseline_glandular_essdai_only_evaluable"] = row["baseline_glandular_evaluable"] if ess_col in visits else False
    for key, meta in EXTRAGLANDULAR_DOMAINS.items():
        row[f"baseline_eg_{key}_active"] = _bool_any(visits[meta["active_col"]])
        row[f"baseline_eg_{key}_evaluable"] = _bool_evaluable_any(visits[f"eg_{key}_evaluable"])
    if not (row["baseline_glandular_evaluable"] and row["baseline_extraglandular_evaluable"]):
        row["baseline_status"] = "insufficient_info"
    elif row["baseline_glandular_active"] and row["baseline_extraglandular_active"]:
        row["baseline_status"] = "overlap"
    elif row["baseline_glandular_active"]:
        row["baseline_status"] = "glandular_only"
    elif row["baseline_extraglandular_active"]:
        row["baseline_status"] = "extraglandular_only"
    else:
        row["baseline_status"] = "neither"
    return row


def select_observed_baseline(long_df: pd.DataFrame) -> pd.DataFrame:
    """Select one observed baseline per patient without inventing a dx-date visit."""
    rows = []
    evaluable = long_df[long_df["glandular_evaluable"] | long_df["extraglandular_evaluable"]].copy()
    for pid, g0 in long_df.groupby(PATIENT_ID_COL, sort=False):
        g = evaluable[evaluable[PATIENT_ID_COL].eq(pid)].sort_values("visit_date_min")
        if g0["dx_date"].isna().all():
            rows.append({PATIENT_ID_COL: pid, "observed_baseline_date": pd.NaT, "dx_date": pd.NaT, "baseline_timing_category": "missing_dx_date", "baseline_status": "insufficient_info"})
            continue
        if g.empty:
            dx_date = g0["dx_date"].dropna().iloc[0]
            rows.append({PATIENT_ID_COL: pid, "observed_baseline_date": pd.NaT, "dx_date": dx_date, "baseline_timing_category": "unclassifiable", "baseline_status": "insufficient_info"})
            continue
        near = g[g["in_near_dx_window"].fillna(False)]
        if len(near):
            rows.append(collapse_baseline_visits(pid, near, "near_dx_observed_baseline"))
            continue
        post = g[g["days_from_dx"] > DX_WINDOW_POST_DAYS]
        if len(post):
            rows.append(collapse_baseline_visits(pid, post.iloc[[0]], "late_observed_baseline"))
            continue
        pre = g[g["days_from_dx"] < -DX_WINDOW_PRE_DAYS]
        if len(pre):
            rows.append(collapse_baseline_visits(pid, pre.sort_values("days_from_dx", ascending=False).iloc[[0]], "pre_dx_observed_baseline"))
            continue
        rows.append({PATIENT_ID_COL: pid, "observed_baseline_date": pd.NaT, "dx_date": g0["dx_date"].dropna().iloc[0], "baseline_timing_category": "unclassifiable", "baseline_status": "insufficient_info"})
    baseline = pd.DataFrame(rows)
    defaults = {
        "baseline_glandular_active": False, "baseline_glandular_evaluable": False,
        "baseline_extraglandular_active": False, "baseline_extraglandular_evaluable": False,
        "baseline_within_1yr_of_dx": False, "baseline_within_2yr_of_dx": False,
        "n_visits_collapsed_into_baseline": 0,
    }
    for k, v in defaults.items():
        if k not in baseline:
            baseline[k] = v
        baseline[k] = baseline[k].fillna(v)
    return baseline


def build_dx_temporal_patient_summary(baseline: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, base in baseline.iterrows():
        pid = base[PATIENT_ID_COL]
        g = long_df[long_df[PATIENT_ID_COL].eq(pid)]
        fu = g[g["is_post_observed_baseline_followup"].fillna(False)]
        fu_eval = fu[fu["overlap_evaluable"]]
        fu_eg_eval = fu[fu["extraglandular_evaluable"]]
        incident_denom = bool(base.get("baseline_glandular_active", False) and not base.get("baseline_extraglandular_active", False) and base.get("baseline_glandular_evaluable", False) and base.get("baseline_extraglandular_evaluable", False) and len(fu_eg_eval) > 0)
        inc_visits = fu_eg_eval[fu_eg_eval["extraglandular_active"]] if incident_denom else fu_eg_eval.iloc[0:0]
        first = inc_visits.iloc[0] if len(inc_visits) else None
        row = base.to_dict()
        row.update({"fu_n_visits": len(fu), "fu_n_evaluable_visits": len(fu_eval), "fu_has_evaluable_overlap": len(fu_eval) > 0, "fu_glandular_ever": _bool_any(fu_eval["glandular_active"]) if len(fu_eval) else False, "fu_extraglandular_ever": _bool_any(fu_eval["extraglandular_active"]) if len(fu_eval) else False, "fu_overlap_ever": _bool_any(fu_eval["overlap_active"]) if len(fu_eval) else False, "fu_active_extraglandular_domains_ever": ";".join(sorted(set(d for s in fu["active_extraglandular_domains"] for d in str(s).split(";") if d))), "incident_denominator": incident_denom, "incident_extraglandular": first is not None, "first_incident_date": first["visit_date_min"] if first is not None else pd.NaT, "time_from_observed_baseline_to_new_domain_yrs": first["time_from_observed_baseline_yrs"] if first is not None else np.nan, "time_from_dx_to_new_domain_yrs": first["time_from_dx_yrs"] if first is not None else np.nan, "incident_domains_at_first_event": first["active_extraglandular_domains"] if first is not None else ""})
        for key, meta in EXTRAGLANDULAR_DOMAINS.items():
            row[f"fu_eg_{key}_ever"] = _bool_any(fu[meta["active_col"]]) if len(fu) else False
        if not row["fu_has_evaluable_overlap"]:
            row["followup_status"] = "insufficient_followup"
        elif row["fu_overlap_ever"]:
            row["followup_status"] = "overlap_followup"
        elif row["fu_glandular_ever"] and not row["fu_extraglandular_ever"]:
            row["followup_status"] = "glandular_only_followup"
        elif row["fu_extraglandular_ever"] and not row["fu_glandular_ever"]:
            row["followup_status"] = "extraglandular_only_followup"
        elif row["fu_glandular_ever"] and row["fu_extraglandular_ever"]:
            row["followup_status"] = "mixed_nonoverlap_followup"
        else:
            row["followup_status"] = "neither_followup"
        rows.append(row)
    return pd.DataFrame(rows)


def make_followup_table(patient: pd.DataFrame) -> pd.DataFrame:
    denom = int(patient["fu_has_evaluable_overlap"].sum())
    rows = []
    for cat in ["overlap_followup", "glandular_only_followup", "extraglandular_only_followup", "neither_followup", "insufficient_followup"]:
        n = int((patient["followup_status"] == cat).sum())
        d = len(patient) if cat == "insufficient_followup" else denom
        rows.append(["followup_prevalence", cat, n, d, (100 * n / d) if d else np.nan, "patient_percent", "Follow-up begins after observed baseline."])
    return pd.DataFrame(rows, columns=["analysis", "category", "n", "denominator", "pct", "pct_type", "notes"])


def make_incident_table(patient: pd.DataFrame) -> pd.DataFrame:
    denom = int(patient["incident_denominator"].sum())
    incident = patient[patient["incident_extraglandular"]].copy()
    n_inc = len(incident)
    return pd.DataFrame([
        ["summary", "n_glandular_only_observed_baseline", denom, np.nan, np.nan, "Strict observed-baseline glandular-only denominator"],
        ["summary", "n_incident_extraglandular", n_inc, denom, (100*n_inc/denom) if denom else np.nan, "First post-observed-baseline extraglandular activity"],
        ["summary", "median_time_from_observed_baseline_to_new_domain_yrs", incident["time_from_observed_baseline_to_new_domain_yrs"].median() if n_inc else np.nan, n_inc, np.nan, "Among incident patients"],
        ["summary", "median_time_from_dx_to_new_domain_yrs", incident["time_from_dx_to_new_domain_yrs"].median() if n_inc else np.nan, n_inc, np.nan, "Among incident patients"],
    ], columns=["section", "domain", "n_incident_first_event", "denominator_incident_patients", "pct_among_incident_patients", "notes"])


def make_domain_incident_table(baseline: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, meta in EXTRAGLANDULAR_DOMAINS.items():
        active_col = meta["active_col"]; eval_col = f"eg_{key}_evaluable"
        at_risk = baseline[(baseline[f"baseline_eg_{key}_evaluable"] == True) & (baseline[f"baseline_eg_{key}_active"] == False)]
        n_at_risk = n_inc = 0; times_ob = []; times_dx = []; event_dates = []; py = 0.0
        for _, b in at_risk.iterrows():
            fu = long_df[(long_df[PATIENT_ID_COL].eq(b[PATIENT_ID_COL])) & (long_df["is_post_observed_baseline_followup"].fillna(False)) & (long_df[eval_col])]
            if fu.empty:
                continue
            n_at_risk += 1
            py += max(float(fu["time_from_observed_baseline_yrs"].max()), 0.0)
            ev = fu[fu[active_col]].head(1)
            if len(ev):
                n_inc += 1; r = ev.iloc[0]
                times_ob.append(r["time_from_observed_baseline_yrs"]); times_dx.append(r["time_from_dx_yrs"]); event_dates.append(r["visit_date_min"])
        rows.append([key, meta["label"], n_at_risk, n_inc, (100*n_inc/n_at_risk) if n_at_risk else np.nan, np.nanmedian(times_ob) if times_ob else np.nan, np.nanmedian(times_dx) if times_dx else np.nan, py, (100*n_inc/py) if py else np.nan, min(event_dates) if event_dates else pd.NaT, max(event_dates) if event_dates else pd.NaT])
    return pd.DataFrame(rows, columns=["domain", "domain_label", "n_at_risk", "n_incident", "pct_incident", "median_time_from_observed_baseline_to_domain_yrs", "median_time_from_dx_to_domain_yrs", "person_years_observed", "incidence_rate_per_100_py", "first_event_date_min", "first_event_date_max"])


def _fisher_exact_p(a:int,b:int,c:int,d:int)->float:
    n=a+b+c+d; r1=a+b; c1=a+c
    def prob(x): return math.comb(c1,x)*math.comb(n-c1,r1-x)/math.comb(n,r1)
    lo=max(0,r1-(n-c1)); hi=min(r1,c1); p_obs=prob(a)
    return min(1.0, sum(prob(x) for x in range(lo,hi+1) if prob(x) <= p_obs + 1e-12))


def _bh(pvals: list[float]) -> list[float]:
    m=len(pvals); order=np.argsort([1 if pd.isna(p) else p for p in pvals]); adj=[np.nan]*m; prev=1.0
    for rank, i in enumerate(order[::-1], start=1):
        p=pvals[i]
        if pd.isna(p): continue
        val=min(prev, p*m/(m-rank+1)); adj[i]=val; prev=val
    return adj


def make_pairwise_table(baseline: pd.DataFrame, glandular_active_col: str="baseline_glandular_active", glandular_eval_col: str="baseline_glandular_evaluable") -> pd.DataFrame:
    rows=[]
    for key in PRIMARY_EG_DOMAIN_KEYS:
        meta=EXTRAGLANDULAR_DOMAINS[key]
        active_col=f"baseline_eg_{key}_active"; eval_col=f"baseline_eg_{key}_evaluable"
        comp=baseline[(baseline[glandular_eval_col] == True) & (baseline[eval_col] == True)]
        gp=comp[glandular_active_col].astype(bool); dp=comp[active_col].astype(bool)
        a=int((gp & dp).sum()); b=int((gp & ~dp).sum()); c=int((~gp & dp).sum()); d=int((~gp & ~dp).sum())
        n=a+b+c+d; miss=len(baseline)-n
        if n:
            exp=[(a+b)*(a+c)/n,(a+b)*(b+d)/n,(c+d)*(a+c)/n,(c+d)*(b+d)/n]
            min_exp=min(exp)
        else: min_exp=np.nan
        fisher = bool(n and min_exp < 5)
        p = _fisher_exact_p(a,b,c,d) if fisher else (math.erfc(math.sqrt((((a*d-b*c)**2*n)/((a+b)*(c+d)*(a+c)*(b+d)))/2)) if all(x>0 for x in [(a+b),(c+d),(a+c),(b+d)]) else np.nan)
        aa,bb,cc,dd = (a,b,c,d) if min(a,b,c,d)>0 else (a+.5,b+.5,c+.5,d+.5)
        risk1=aa/(aa+bb); risk0=cc/(cc+dd); pr=risk1/risk0 if risk0 else np.nan
        se_log_pr=math.sqrt((1/aa)-(1/(aa+bb))+(1/cc)-(1/(cc+dd))) if aa and cc else np.nan
        orv=(aa*dd)/(bb*cc) if bb and cc else np.nan; se_log_or=math.sqrt(1/aa+1/bb+1/cc+1/dd)
        rows.append({"domain":key,"domain_label":meta["label"],"n_complete":n,"n_missing_or_not_evaluable":miss,"glandular_pos_domain_pos":a,"glandular_pos_domain_neg":b,"glandular_neg_domain_pos":c,"glandular_neg_domain_neg":d,"pct_domain_active_if_glandular_pos":100*a/(a+b) if (a+b) else np.nan,"pct_domain_active_if_glandular_neg":100*c/(c+d) if (c+d) else np.nan,"prevalence_ratio":pr,"pr_ci_low":math.exp(math.log(pr)-1.96*se_log_pr) if pr and not pd.isna(se_log_pr) else np.nan,"pr_ci_high":math.exp(math.log(pr)+1.96*se_log_pr) if pr and not pd.isna(se_log_pr) else np.nan,"risk_difference":(a/(a+b)-c/(c+d)) if (a+b) and (c+d) else np.nan,"odds_ratio":orv,"or_ci_low":math.exp(math.log(orv)-1.96*se_log_or) if orv else np.nan,"or_ci_high":math.exp(math.log(orv)+1.96*se_log_or) if orv else np.nan,"min_expected_cell_count":min_exp,"test_used":"fisher_exact" if fisher else "chi_square","p_value":p,"haldane_anscombe_applied": min(a,b,c,d)==0})
    out=pd.DataFrame(rows); out["p_adj_fdr_bh"]=_bh(out["p_value"].tolist()); out["significant_nominal_0_05"]=out["p_value"]<0.05; out["significant_fdr_0_05"]=out["p_adj_fdr_bh"]<0.05
    return out


def make_pairwise_heatmap(pairwise: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(pairwise)+1)))
    vals = np.log2(pairwise["prevalence_ratio"].replace([np.inf, -np.inf], np.nan)).to_numpy().reshape(-1,1)
    im = ax.imshow(vals, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_yticks(range(len(pairwise))); ax.set_yticklabels(pairwise["domain_label"]); ax.set_xticks([0]); ax.set_xticklabels(["log2(PR)"])
    for i,r in pairwise.reset_index(drop=True).iterrows():
        star = "*" if bool(r["significant_fdr_0_05"]) else ""
        ax.text(0, i, f"PR {r['prevalence_ratio']:.2f}\nN {int(r['n_complete'])}\nFDR {r['p_adj_fdr_bh']:.3g}{star}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="log2(prevalence ratio)"); ax.set_title("Observed-baseline pairwise associations")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def make_overlap_timeline_plot(long_df: pd.DataFrame, patient: pd.DataFrame, path: Path) -> None:
    plot_df = long_df[long_df["dx_date"].notna()].copy()
    order = patient.sort_values(["baseline_timing_category", "time_from_dx_to_observed_baseline_yrs"], na_position="last").reset_index(drop=True)
    order["y"] = np.arange(len(order))[::-1]
    plot_df = plot_df.merge(order[[PATIENT_ID_COL,"y"]], on=PATIENT_ID_COL, how="inner")
    fig, ax = plt.subplots(figsize=(14, max(6, min(18, 0.08*len(order)+4))))
    for x in [0,1,2,5,10]: ax.axvline(x, color="#777", ls="--" if x else "-", lw=0.8, alpha=0.6)
    for _, g in plot_df.groupby(PATIENT_ID_COL): ax.hlines(g["y"].iloc[0], g["time_from_dx_yrs"].min(), g["time_from_dx_yrs"].max(), color="#ddd", lw=.5)
    ax.scatter(plot_df["time_from_dx_yrs"], plot_df["y"], c="#999", s=14, label="Observed visits")
    base = plot_df[plot_df["is_observed_baseline"]]; ax.scatter(base["time_from_dx_yrs"], base["y"], c="#1f77b4", s=30, marker="D", label="Observed baseline")
    ov = plot_df[plot_df["overlap_active"]]; ax.scatter(ov["time_from_dx_yrs"], ov["y"], c="#2ca02c", s=28, marker="s", label="Overlap visits")
    inc = patient[patient["incident_extraglandular"]].merge(order[[PATIENT_ID_COL,"y"]], on=PATIENT_ID_COL); ax.scatter(inc["time_from_dx_to_new_domain_yrs"], inc["y"], c="#d62728", s=42, marker="*", label="Incident extraglandular event")
    ax.set_title("Diagnosis-anchored timeline of glandular/extraglandular overlap\nTime zero is reported diagnosis date; baseline is first evaluable observed assessment.")
    ax.set_xlabel("Years from Sjögren diagnosis date"); ax.set_ylabel("Patients"); ax.legend(loc="best"); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def make_domain_timeline_plot(long_df: pd.DataFrame, patient: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8)); y=0; yt=[]; yl=[]
    fu = long_df[long_df["is_post_observed_baseline_followup"].fillna(False)]
    for key in PRIMARY_EG_DOMAIN_KEYS:
        active = fu[fu[EXTRAGLANDULAR_DOMAINS[key]["active_col"]]]
        ax.scatter(active["time_from_dx_yrs"], np.full(len(active), y), s=16, label=EXTRAGLANDULAR_DOMAINS[key]["label"])
        yt.append(y); yl.append(EXTRAGLANDULAR_DOMAINS[key]["label"]); y += 1
    for x in [0,1,2,5,10]: ax.axvline(x, color="#777", ls="--" if x else "-", lw=0.8, alpha=0.6)
    ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_xlabel("Years from Sjögren diagnosis date"); ax.set_title("Post-observed-baseline extraglandular domain activity")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def run_qc(raw: pd.DataFrame, baseline: pd.DataFrame, patient: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    valid_base = baseline[baseline["observed_baseline_date"].notna()]
    q = {
        "n_total_patients": raw[PATIENT_ID_COL].nunique(),
        "n_with_dx_date": raw.assign(dx=raw[DX_DATE_COL].map(parse_dx_date))[raw.assign(dx=raw[DX_DATE_COL].map(parse_dx_date))["dx"].notna()][PATIENT_ID_COL].nunique() if DX_DATE_COL in raw else 0,
        "n_without_dx_date": raw[PATIENT_ID_COL].nunique() - baseline[baseline["dx_date"].notna()][PATIENT_ID_COL].nunique(),
        "n_with_observed_baseline": len(valid_base),
        "n_without_observed_baseline": int(baseline["observed_baseline_date"].isna().sum()),
        "n_near_dx_observed_baseline": int((baseline["baseline_timing_category"]=="near_dx_observed_baseline").sum()),
        "n_late_observed_baseline": int((baseline["baseline_timing_category"]=="late_observed_baseline").sum()),
        "n_pre_dx_observed_baseline": int((baseline["baseline_timing_category"]=="pre_dx_observed_baseline").sum()),
        "median_time_from_dx_to_observed_baseline_yrs": valid_base["time_from_dx_to_observed_baseline_yrs"].median(),
        "iqr_time_from_dx_to_observed_baseline_yrs": valid_base["time_from_dx_to_observed_baseline_yrs"].quantile(.75)-valid_base["time_from_dx_to_observed_baseline_yrs"].quantile(.25),
        "pct_observed_baseline_more_than_1yr_after_dx": 100*(valid_base["time_from_dx_to_observed_baseline_yrs"]>1).mean() if len(valid_base) else np.nan,
        "pct_observed_baseline_more_than_2yr_after_dx": 100*(valid_base["time_from_dx_to_observed_baseline_yrs"]>2).mean() if len(valid_base) else np.nan,
        "pct_observed_baseline_more_than_5yr_after_dx": 100*(valid_base["time_from_dx_to_observed_baseline_yrs"]>5).mean() if len(valid_base) else np.nan,
        "n_with_post_observed_baseline_followup": int(patient["fu_n_visits"].gt(0).sum()),
        "n_incidence_denominator": int(patient["incident_denominator"].sum()),
        "n_incident_extraglandular": int(patient["incident_extraglandular"].sum()),
        "n_pairwise_evaluable_by_domain": ";".join(f"{r.domain}:{int(r.n_complete)}" for r in pairwise.itertuples()),
        "each_patient_one_observed_baseline_row": bool(baseline[PATIENT_ID_COL].is_unique),
        "no_incident_event_on_or_before_observed_baseline": bool((patient.loc[patient["incident_extraglandular"], "time_from_observed_baseline_to_new_domain_yrs"] > 0).all()),
        "fisher_used_when_expected_cell_lt5": bool((pairwise.loc[pairwise["min_expected_cell_count"] < 5, "test_used"] == "fisher_exact").all()),
        "fdr_values_present_for_primary_domains": bool(pairwise["p_adj_fdr_bh"].notna().all()),
    }
    return pd.DataFrame([{"check": k, "value": v, "passed": True} for k,v in q.items()])

def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    if INPUT_PARQUET.name != EXPECTED_INPUT_NAME:
        warnings.warn(f"Input filename from common differs from expected reference: {INPUT_PARQUET}")
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Expected longitudinal parquet from common.DEFAULT_ANALYTIC_DATASET not found: {INPUT_PARQUET}")
    raw = pd.read_parquet(INPUT_PARQUET)
    missing_critical = [c for c in [PATIENT_ID_COL, VISIT_DATE_COL, DX_DATE_COL] if c not in raw.columns]
    if missing_critical:
        raise ValueError(f"Missing critical columns: {missing_critical}")
    optional = [*GLANDULAR_COLS.values(), *[m["col"] for m in EXTRAGLANDULAR_DOMAINS.values()]]
    missing_optional = [c for c in optional if c not in raw.columns]
    if missing_optional:
        warnings.warn("Missing optional columns, continuing: " + ", ".join(missing_optional))
    if not any(m["col"] in raw.columns for m in EXTRAGLANDULAR_DOMAINS.values()):
        raise ValueError("No ESSDAI extraglandular domain column is available.")

    df = raw.copy()
    df["visit_date_min"] = df[VISIT_DATE_COL].map(parse_min_visit_date)
    df["dx_date"] = df[DX_DATE_COL].map(parse_dx_date)
    df["dx_date_precision"] = df[DX_DATE_COL].map(dx_date_imputed_precision)
    df = df[df["visit_date_min"].notna()].sort_values([PATIENT_ID_COL, "visit_date_min"]).copy()
    df["days_from_dx"] = (df["visit_date_min"] - df["dx_date"]).dt.days
    df["time_from_dx_yrs"] = df["days_from_dx"] / 365.25
    df["near_dx_window_start"] = df["dx_date"] - pd.to_timedelta(DX_WINDOW_PRE_DAYS, unit="D")
    df["near_dx_window_end"] = df["dx_date"] + pd.to_timedelta(DX_WINDOW_POST_DAYS, unit="D")
    df["in_near_dx_window"] = df["days_from_dx"].between(-DX_WINDOW_PRE_DAYS, DX_WINDOW_POST_DAYS)

    long_df = pd.concat([
        df[[PATIENT_ID_COL, "visit_date_min", "dx_date", "dx_date_precision", "days_from_dx", "time_from_dx_yrs", "in_near_dx_window", "near_dx_window_start", "near_dx_window_end"]],
        derive_glandular_flags(df),
        derive_extraglandular_flags(df),
    ], axis=1)
    long_df = derive_overlap_flags(long_df)
    dx_long_df = long_df[long_df["dx_date"].notna()].copy()

    baseline = select_observed_baseline(dx_long_df)
    baseline_valid = baseline[baseline["observed_baseline_date"].notna()].copy()
    long_df = long_df.merge(
        baseline[[PATIENT_ID_COL, "observed_baseline_date", "baseline_timing_category"]],
        on=PATIENT_ID_COL,
        how="left",
    )
    long_df["is_observed_baseline"] = long_df["visit_date_min"].eq(long_df["observed_baseline_date"])
    long_df["is_post_observed_baseline_followup"] = long_df["visit_date_min"] > long_df["observed_baseline_date"]
    long_df["time_from_observed_baseline_yrs"] = (long_df["visit_date_min"] - long_df["observed_baseline_date"]).dt.days / 365.25

    patient = build_dx_temporal_patient_summary(baseline_valid, long_df)
    follow = make_followup_table(patient)
    incident = make_incident_table(patient)
    domain_incident = make_domain_incident_table(baseline_valid, long_df)
    pairwise_main = make_pairwise_table(baseline_valid)
    near_dx = baseline_valid[baseline_valid["baseline_within_1yr_of_dx"] == True]
    pairwise_near = make_pairwise_table(near_dx) if len(near_dx) else make_pairwise_table(baseline_valid.iloc[0:0])
    pairwise_sens = make_pairwise_table(baseline_valid, "baseline_glandular_essdai_only_active", "baseline_glandular_essdai_only_evaluable")

    long_df.to_parquet(LONGITUDINAL_OUT, index=False)
    long_df.to_csv(LONGITUDINAL_CSV_OUT, index=False)
    baseline_valid.to_parquet(OBSERVED_BASELINE_OUT, index=False)
    baseline_valid.to_csv(OBSERVED_BASELINE_CSV_OUT, index=False)
    patient.to_parquet(PATIENT_SUMMARY_OUT, index=False)
    patient.to_csv(PATIENT_SUMMARY_CSV_OUT, index=False)
    follow.to_csv(FOLLOWUP_TABLE, index=False)
    incident.to_csv(INCIDENT_TABLE, index=False)
    domain_incident.to_csv(DOMAIN_INCIDENT_TABLE, index=False)
    pairwise_main.to_csv(PAIRWISE_TABLE, index=False)
    pairwise_near.to_csv(PAIRWISE_NEAR_DX_TABLE, index=False)
    pairwise_sens.to_csv(PAIRWISE_SENS_TABLE, index=False)
    baseline.to_csv(BASELINE_AUDIT_OUT, index=False)
    make_overlap_timeline_plot(long_df, patient, TIMELINE_FIG)
    make_domain_timeline_plot(long_df, patient, DOMAIN_TIMELINE_FIG)
    make_pairwise_heatmap(pairwise_main, PAIRWISE_HEATMAP_FIG)
    qc = run_qc(raw, baseline, patient, pairwise_main)
    qc.to_csv(QC_OUT, index=False)

    n_with_dx_date = int(dx_long_df[PATIENT_ID_COL].nunique())
    n_near_dx_observed_baseline = int((baseline_valid["baseline_timing_category"] == "near_dx_observed_baseline").sum())
    n_late_observed_baseline = int((baseline_valid["baseline_timing_category"] == "late_observed_baseline").sum())
    median_time_from_dx_to_baseline = baseline_valid["time_from_dx_to_observed_baseline_yrs"].median()
    n_with_followup = int(patient["fu_n_visits"].gt(0).sum())
    n_incidence_denominator = int(patient["incident_denominator"].sum())
    n_incident = int(patient["incident_extraglandular"].sum())

    print("\nITEM 4.2/4.3 — Diagnosis-anchored temporal analysis")
    print(f"Patients with valid dx_date: {n_with_dx_date}")
    print(f"Patients with observed baseline: {len(baseline_valid)}")
    print(f"Near-dx observed baseline patients: {n_near_dx_observed_baseline}")
    print(f"Late observed baseline patients: {n_late_observed_baseline}")
    print(f"Median years from dx to observed baseline: {median_time_from_dx_to_baseline:.2f}")
    print(f"Patients with post-baseline follow-up: {n_with_followup}")
    print(f"Incident extraglandular denominator: {n_incidence_denominator}")
    print(f"Incident extraglandular events: {n_incident}")
    print(f"Pairwise association table saved to: {PAIRWISE_TABLE}")
    print(f"Follow-up table saved to: {FOLLOWUP_TABLE}")
    print(f"Incident table saved to: {INCIDENT_TABLE}")
    top = pairwise_main.sort_values(["p_adj_fdr_bh", "p_value"]).head(3)
    for _, r in top.iterrows():
        print(
            f"{r['domain_label']}: "
            f"PR={r['prevalence_ratio']:.2f}, "
            f"OR={r['odds_ratio']:.2f}, "
            f"p={r['p_value']:.4g}, "
            f"FDR={r['p_adj_fdr_bh']:.4g}, "
            f"N={int(r['n_complete'])}"
        )


if __name__ == "__main__":
    main()

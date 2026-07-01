#!/usr/bin/env python3
"""ITEM 4.2 — Follow-up prevalence of glandular/extraglandular overlap.

Reads the longitudinal collapsed visit parquet, derives visit-level glandular and
ESSDAI extraglandular activity, summarizes follow-up overlap prevalence and
incident extraglandular manifestations after glandular-only baseline disease,
and writes manuscript tables, intermediate QC files, and timeline figures.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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

RAW_DIR = common.RAW_DATA_DIR
INTERMEDIATE_DIR = common.INTERMEDIATE_DATA_DIR
TABLE_DIR = common.BLOCKA_TABLES_DIR
FIGURE_DIR = common.OUTPUTS_DIR / "figures" / "blockA"
INPUT_PARQUET = Path(common.DEFAULT_ANALYTIC_DATASET)

FOLLOWUP_TABLE = TABLE_DIR / "06_overlap_followup.csv"
INCIDENT_TABLE = TABLE_DIR / "06_incident_extraglandular.csv"
LONGITUDINAL_OUT = INTERMEDIATE_DIR / "06_overlap_longitudinal_patient_visit.parquet"
PATIENT_SUMMARY_OUT = INTERMEDIATE_DIR / "06_overlap_patient_summary.parquet"
QC_OUT = INTERMEDIATE_DIR / "06_overlap_qc_summary.csv"
TIMELINE_FIG = FIGURE_DIR / "06_timeline_overlaping.pdf"
DOMAIN_TIMELINE_FIG = FIGURE_DIR / "06_timeline_overlaping_plus_extraglandular_cats.pdf"

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


def make_followup_table(patient: pd.DataFrame) -> pd.DataFrame:
    denom = int(patient["fu_has_evaluable_overlap"].sum())
    total = int(patient["fu_n_visits"].gt(0).sum())
    rows = []
    for cat in ["overlap_followup", "glandular_only_followup", "extraglandular_only_followup", "neither_followup", "insufficient_followup"]:
        n = int((patient["followup_status"] == cat).sum())
        d = total if cat == "insufficient_followup" else denom
        rows.append(["followup_prevalence", cat, n, d, (100 * n / d) if d else np.nan, "patient_percent", "Follow-up is visit_order > 1."])
    rows += [["followup_denominator", "followup_evaluable_patients", denom, total, (100 * denom / total) if total else np.nan, "patient_percent", "Patients with >=1 evaluable follow-up visit."],
             ["followup_denominator", "followup_total_patients", total, len(patient), (100 * total / len(patient)) if len(patient) else np.nan, "patient_percent", "Patients with >=1 post-baseline visit."]]
    for key, meta in EXTRAGLANDULAR_DOMAINS.items():
        col = f"fu_eg_{key}_ever"
        n = int(patient[col].sum()) if col in patient else 0
        rows.append(["followup_extraglandular_domain", f"{key}_ever_active_followup", n, denom, (100 * n / denom) if denom else np.nan, "patient_percent", meta["label"]])
    return pd.DataFrame(rows, columns=["analysis", "category", "n", "denominator", "pct", "pct_type", "notes"])


def make_incident_table(patient: pd.DataFrame) -> pd.DataFrame:
    denom = int(patient["incident_denominator"].sum())
    incident = patient[patient["incident_extraglandular"]].copy()
    n_inc = len(incident)
    counter = Counter(d for s in incident["incident_domains_at_first_event"].dropna() for d in str(s).split(";") if d)
    max_count = max(counter.values()) if counter else 0
    common = ";".join(sorted([d for d, n in counter.items() if n == max_count])) if counter else "None"
    median_time = incident["time_to_new_domain_yrs"].median() if n_inc else np.nan
    rows = [["summary", "n_glandular_only_at_baseline", denom, np.nan, np.nan, "Strict baseline glandular-only denominator"],
            ["summary", "n_incident_extraglandular", n_inc, denom, (100*n_inc/denom) if denom else np.nan, "Incident post-baseline extraglandular active"],
            ["summary", "pct_incident_extraglandular", n_inc, denom, (100*n_inc/denom) if denom else np.nan, "Percent uses strict denominator"],
            ["summary", "most_common_incident_domain", common, n_inc, np.nan, "Ties separated by semicolon"],
            ["summary", "median_time_to_new_domain_yrs", median_time, n_inc, np.nan, "Among incident patients"]]
    for label in [m["label"] for m in EXTRAGLANDULAR_DOMAINS.values()]:
        n = counter.get(label, 0)
        rows.append(["domain_counts", label, n, n_inc, (100*n/n_inc) if n_inc else np.nan, "First incident visit domains"])
    return pd.DataFrame(rows, columns=["section", "domain", "n_incident_first_event", "denominator_incident_patients", "pct_among_incident_patients", "notes"])


def ordered_patients(long_df: pd.DataFrame) -> list[Any]:
    def key(g: pd.DataFrame) -> tuple[int, float, float]:
        base_overlap = bool(g.loc[g["visit_order"].eq(1), "overlap_active"].any())
        any_overlap = bool(g["overlap_active"].any())
        any_eval = bool(g["overlap_evaluable"].any())
        first = g.loc[g["overlap_active"], "time_from_baseline_yrs"].min()
        maxfu = g["time_from_baseline_yrs"].max()
        group = 0 if base_overlap else 1 if any_overlap else 2 if any_eval else 3
        return (group, float(first) if pd.notna(first) else float(maxfu or 0), -float(maxfu or 0))
    return sorted(long_df[PATIENT_ID_COL].dropna().unique(), key=lambda p: key(long_df[long_df[PATIENT_ID_COL] == p]))


def make_overlap_timeline_plot(long_df: pd.DataFrame, patient: pd.DataFrame, path: Path) -> None:
    pats = ordered_patients(long_df)
    anon = {p: f"P{i+1:03d}" for i, p in enumerate(pats)}
    ymap = {p: i for i, p in enumerate(pats)}
    h = max(6, min(80, 1.2 + 0.18 * len(pats)))
    fig, ax = plt.subplots(figsize=(12, h))
    d = long_df.copy(); d["y"] = d[PATIENT_ID_COL].map(ymap)
    insuff = d["overlap_status"].eq("insufficient_info")
    no = d["overlap_evaluable"] & ~d["overlap_active"]
    ov = d["overlap_active"]
    ax.scatter(d.loc[insuff, "time_from_baseline_yrs"], d.loc[insuff, "y"], c="#eeeeee", edgecolors="#cccccc", s=20, label="insufficient info")
    ax.scatter(d.loc[no, "time_from_baseline_yrs"], d.loc[no, "y"], c="#888888", s=22, label="no overlap")
    sc = ax.scatter(d.loc[ov, "time_from_baseline_yrs"], d.loc[ov, "y"], c=d.loc[ov, "overlap_intensity_count"], cmap="viridis", marker="s", s=36, label="overlap")
    if ov.any():
        fig.colorbar(sc, ax=ax, label="Active glandular + extraglandular count")
    ax.set_yticks(range(len(pats)))
    ax.set_yticklabels([anon[p] for p in pats] if len(pats) <= MAX_PATIENTS_FOR_LABELS else [], fontsize=6)
    ax.invert_yaxis(); ax.set_xlabel("Time from baseline (years)"); ax.set_ylabel("Patients")
    n_eval = int(patient["fu_has_evaluable_overlap"].sum()); n_ov = int(patient["fu_overlap_ever"].sum())
    ax.set_title(f"Glandular/extraglandular overlap timeline\nTotal patients: {len(patient)} | Evaluable follow-up: {n_eval} | Overlap during follow-up: {n_ov}")
    ax.legend(handles=[Line2D([0],[0], marker='o', color='w', markerfacecolor='#888888', label='no overlap', markersize=6), Line2D([0],[0], marker='o', color='w', markerfacecolor='#eeeeee', markeredgecolor='#cccccc', label='insufficient info', markersize=6), Line2D([0],[0], marker='s', color='w', markerfacecolor='#440154', label='overlap', markersize=6)], loc="best")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def make_domain_timeline_plot(long_df: pd.DataFrame, path: Path) -> None:
    pats = ordered_patients(long_df); domains = list(EXTRAGLANDULAR_DOMAINS.items())
    nrows = max(1, len(pats) * len(domains)); h = max(8, min(120, 0.09 * nrows + 2))
    fig, ax = plt.subplots(figsize=(14, h))
    colors = plt.cm.tab20(np.linspace(0, 1, len(domains)))
    for pi, p in enumerate(pats):
        g = long_df[long_df[PATIENT_ID_COL] == p]
        base = pi * len(domains)
        for di, (key, meta) in enumerate(domains):
            y = base + di
            ev = g[f"eg_{key}_evaluable"]
            ac = g[meta["active_col"]]
            ax.scatter(g.loc[ev & ~ac, "time_from_baseline_yrs"], [y] * int((ev & ~ac).sum()), c="#eeeeee", s=5, marker="o")
            ax.scatter(g.loc[ac, "time_from_baseline_yrs"], [y] * int(ac.sum()), c=[colors[di]], s=18, marker="x", label=meta["label"] if pi == 0 else None)
        ax.axhline(base - 0.5, color="#dddddd", lw=0.5)
    ticks = [i * len(domains) + len(domains)/2 - 0.5 for i in range(len(pats))]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"P{i+1:03d}" for i in range(len(pats))] if len(pats) <= MAX_PATIENTS_FOR_LABELS else [], fontsize=5)
    ax.invert_yaxis(); ax.set_xlabel("Time from baseline (years)"); ax.set_ylabel("Patients / extraglandular domains")
    ax.set_title("ESSDAI extraglandular domain activity timeline")
    ax.legend(ncol=3, fontsize=7, loc="upper right")
    fig.text(0.01, 0.01, "X marks active ESSDAI extraglandular domain at a recorded visit.", fontsize=8)
    fig.tight_layout(rect=[0, 0.02, 1, 1]); fig.savefig(path); plt.close(fig)


def build_patient_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in long_df.groupby(PATIENT_ID_COL, sort=False):
        base = g[g["visit_order"] == 1].iloc[0]
        fu = g[g["visit_order"] > 1]
        fu_eval = fu[fu["overlap_evaluable"]]
        fu_eg_eval = fu[fu["extraglandular_evaluable"]]
        incident_denom = bool(base["glandular_active"] and not base["extraglandular_active"] and base["overlap_evaluable"] and len(fu_eg_eval) > 0)
        inc_visits = fu_eg_eval[fu_eg_eval["extraglandular_active"]] if incident_denom else fu_eg_eval.iloc[0:0]
        first = inc_visits.iloc[0] if len(inc_visits) else None
        row = {PATIENT_ID_COL: pid, "baseline_date": base["baseline_date"], "baseline_status": base["overlap_status"], "baseline_glandular_active": bool(base["glandular_active"]), "baseline_extraglandular_active": bool(base["extraglandular_active"]), "fu_n_visits": len(fu), "fu_n_evaluable_visits": len(fu_eval), "fu_has_evaluable_overlap": len(fu_eval) > 0, "fu_glandular_ever": bool(fu_eval["glandular_active"].any()), "fu_extraglandular_ever": bool(fu_eval["extraglandular_active"].any()), "fu_overlap_ever": bool(fu_eval["overlap_active"].any()), "max_overlap_intensity_count": int(fu_eval["overlap_intensity_count"].max()) if len(fu_eval) else 0, "fu_active_extraglandular_domains_ever": ";".join(sorted(set(d for s in fu["active_extraglandular_domains"] for d in str(s).split(";") if d))), "incident_denominator": incident_denom, "incident_extraglandular": first is not None, "first_incident_date": first["visit_date_min"] if first is not None else pd.NaT, "time_to_new_domain_yrs": first["time_from_baseline_yrs"] if first is not None else np.nan, "incident_domains_at_first_event": first["active_extraglandular_domains"] if first is not None else "", "n_visits": len(g), "n_evaluable_visits": int(g["overlap_evaluable"].sum())}
        for key, meta in EXTRAGLANDULAR_DOMAINS.items():
            row[f"fu_eg_{key}_ever"] = bool(fu_eval[meta["active_col"]].any()) if len(fu_eval) else False
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


def run_qc(raw: pd.DataFrame, long_df: pd.DataFrame, patient: pd.DataFrame, missing_optional: list[str], output_paths: list[Path], invalid_visit_dates: int) -> pd.DataFrame:
    checks = []
    def add(name: str, value: Any, passed: bool = True) -> None:
        checks.append({"check": name, "value": value, "passed": bool(passed)})
        print(f"QC {'PASS' if passed else 'FAIL'}: {name} = {value}")
    add("n raw rows", len(raw)); add("n unique patients", raw[PATIENT_ID_COL].nunique())
    add("n rows with invalid visit date", invalid_visit_dates)
    add("n patients without valid baseline date", int(patient["baseline_date"].isna().sum()))
    add("n patients with follow-up", int(patient["fu_n_visits"].gt(0).sum()))
    add("n patients with evaluable follow-up", int(patient["fu_has_evaluable_overlap"].sum()))
    add("n baseline glandular-only", int((patient["baseline_status"] == "glandular_only").sum()))
    add("n baseline glandular-only with evaluable follow-up", int(patient["incident_denominator"].sum()))
    add("n incident extraglandular", int(patient["incident_extraglandular"].sum()))
    baseline_counts = long_df[long_df["visit_order"] == 1].groupby(PATIENT_ID_COL).size()
    add("baseline exactly one visit per patient", int((baseline_counts == 1).sum()), (baseline_counts == 1).all())
    bad_inc = patient[patient["incident_denominator"] & patient["baseline_extraglandular_active"]]
    add("no baseline extraglandular active in incidence denominator", len(bad_inc), len(bad_inc) == 0)
    add("n_incident <= incidence denominator", f"{patient['incident_extraglandular'].sum()}/{patient['incident_denominator'].sum()}", patient["incident_extraglandular"].sum() <= patient["incident_denominator"].sum())
    t = patient.loc[patient["incident_extraglandular"], "time_to_new_domain_yrs"]
    add("time_to_new_domain_yrs >= 0", int((t >= 0).sum()), bool((t >= 0).all()))
    oi = long_df.loc[long_df["overlap_active"], "overlap_intensity_count"]
    add("overlap_intensity_count >= 2 for overlap_active", int((oi >= 2).sum()), bool((oi >= 2).all()))
    add("missing optional columns", ";".join(missing_optional) if missing_optional else "None")
    for col in sorted(set([*GLANDULAR_COLS.values(), *[m["col"] for m in EXTRAGLANDULAR_DOMAINS.values()], PATIENT_ID_COL, VISIT_DATE_COL]) & set(raw.columns)):
        checks.append({"check": f"missingness::{col}", "value": int(raw[col].isna().sum()), "passed": True})
    for path in output_paths:
        add(f"output exists and nonempty::{path.name}", path.stat().st_size if path.exists() else 0, path.exists() and path.stat().st_size > 0)
    qc = pd.DataFrame(checks)
    if not qc["passed"].all():
        failed = qc.loc[~qc["passed"], "check"].tolist()
        raise AssertionError(f"QC failures: {failed}")
    return qc


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True); FIGURE_DIR.mkdir(parents=True, exist_ok=True); INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    if INPUT_PARQUET.name != EXPECTED_INPUT_NAME:
        warnings.warn(f"Input filename from common differs from expected reference: {INPUT_PARQUET}")
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Expected longitudinal parquet from common.DEFAULT_ANALYTIC_DATASET not found: {INPUT_PARQUET}")
    raw = pd.read_parquet(INPUT_PARQUET)
    missing_critical = [c for c in [PATIENT_ID_COL, VISIT_DATE_COL] if c not in raw.columns]
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
    invalid_visit_dates = int(df["visit_date_min"].isna().sum())
    df = df[df["visit_date_min"].notna()].sort_values([PATIENT_ID_COL, "visit_date_min"]).copy()
    df["visit_order"] = df.groupby(PATIENT_ID_COL).cumcount() + 1
    df["baseline_date"] = df.groupby(PATIENT_ID_COL)["visit_date_min"].transform("first")
    df["time_from_baseline_yrs"] = (df["visit_date_min"] - df["baseline_date"]).dt.days / 365.25
    long_df = pd.concat([df[[PATIENT_ID_COL, "visit_date_min", "baseline_date", "visit_order", "time_from_baseline_yrs"]], derive_glandular_flags(df), derive_extraglandular_flags(df)], axis=1)
    long_df = derive_overlap_flags(long_df)
    patient = build_patient_summary(long_df)

    follow = make_followup_table(patient); incident = make_incident_table(patient)
    long_df.to_parquet(LONGITUDINAL_OUT, index=False); patient.to_parquet(PATIENT_SUMMARY_OUT, index=False)
    follow.to_csv(FOLLOWUP_TABLE, index=False); incident.to_csv(INCIDENT_TABLE, index=False)
    make_overlap_timeline_plot(long_df, patient, TIMELINE_FIG); make_domain_timeline_plot(long_df, DOMAIN_TIMELINE_FIG)
    qc = run_qc(raw, long_df, patient, missing_optional, [FOLLOWUP_TABLE, INCIDENT_TABLE, LONGITUDINAL_OUT, PATIENT_SUMMARY_OUT, TIMELINE_FIG, DOMAIN_TIMELINE_FIG], invalid_visit_dates)
    qc.to_csv(QC_OUT, index=False)
    run_qc(raw, long_df, patient, missing_optional, [FOLLOWUP_TABLE, INCIDENT_TABLE, LONGITUDINAL_OUT, PATIENT_SUMMARY_OUT, QC_OUT, TIMELINE_FIG, DOMAIN_TIMELINE_FIG], invalid_visit_dates)

    follow_saved = pd.read_csv(FOLLOWUP_TABLE); incident_saved = pd.read_csv(INCIDENT_TABLE)
    overlap_row = follow_saved.loc[follow_saved["category"] == "overlap_followup"].iloc[0]
    denom_inc_row = incident_saved.loc[incident_saved["domain"] == "n_glandular_only_at_baseline"].iloc[0]
    inc_row = incident_saved.loc[incident_saved["domain"] == "n_incident_extraglandular"].iloc[0]
    common_row = incident_saved.loc[incident_saved["domain"] == "most_common_incident_domain"].iloc[0]
    median_row = incident_saved.loc[incident_saved["domain"] == "median_time_to_new_domain_yrs"].iloc[0]
    median = pd.to_numeric(pd.Series([median_row["n_incident_first_event"]]), errors="coerce").iloc[0]
    print("\nDuring follow-up, "
          f"{float(overlap_row['pct']):.1f}% of evaluable patients had evidence of overlap between glandular and extraglandular involvement "
          f"({int(overlap_row['n'])}/{int(overlap_row['denominator'])}). Among {int(float(denom_inc_row['n_incident_first_event']))} patients with glandular-only disease at baseline and evaluable follow-up, "
          f"{float(inc_row['pct_among_incident_patients']):.1f}% developed new extraglandular manifestations ({int(float(inc_row['n_incident_first_event']))}/{int(float(denom_inc_row['n_incident_first_event']))}), "
          f"most commonly {common_row['n_incident_first_event']}. Median time to new extraglandular domain was {median:.2f} years.")


if __name__ == "__main__":
    main()

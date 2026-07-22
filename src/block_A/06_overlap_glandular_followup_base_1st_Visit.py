#!/usr/bin/env python3
"""ITEM 4.2 — Follow-up prevalence of glandular/extraglandular overlap.

Reads the longitudinal collapsed visit parquet, derives visit-level glandular and
ESSDAI extraglandular activity, summarizes follow-up overlap prevalence and
incident extraglandular manifestations after glandular-only baseline disease,
and writes manuscript tables, intermediate QC files, and timeline figures.

CHANGE LOG (this version):
- Added `make_domain_incident_table`: per-domain "n at risk" (evaluable AND
  inactive at baseline), n incident, % incident, person-years observed, and
  incidence rate per 100 person-years — mirroring the dx-anchored script's
  06_incident_extraglandular_domains_dx_temporal_anchor.csv, but using the
  1st-visit baseline (visit_order == 1) instead of a dx-anchored baseline.
- `build_patient_summary` now also carries per-domain baseline active/evaluable
  flags on the patient row, since the domain table needs them.
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

FOLLOWUP_TABLE = TABLE_DIR / "06_overlap_followup.csv"
INCIDENT_TABLE = TABLE_DIR / "06_incident_extraglandular.csv"
DOMAIN_INCIDENT_TABLE = TABLE_DIR / "06_incident_extraglandular_domains.csv"
PAIRWISE_TABLE = TABLE_DIR / "06_pairwise_domain_associations_baseline.csv"
PAIRWISE_HEATMAP_FIG = FIGURE_DIR / "06_domain_association_heatmap_baseline.pdf"

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


def make_domain_incident_table(patient: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """Per-domain incidence during follow-up, anchored to the 1st-visit baseline.

    Mirrors the dx-anchored script's domain incidence table:
    - "At risk" = baseline (visit_order == 1) evaluable AND NOT active for that domain.
    - Follow-up = visit_order > 1.
    - Person-years = time from baseline (years) to the patient's last evaluable
      follow-up visit for that domain (censored at last observation if no event).
    - Incidence rate = incident cases per 100 person-years.
    """
    rows = []
    for key, meta in EXTRAGLANDULAR_DOMAINS.items():
        active_col = meta["active_col"]
        eval_col = f"eg_{key}_evaluable"
        base_active_col = f"baseline_eg_{key}_active"
        base_eval_col = f"baseline_eg_{key}_evaluable"

        at_risk = patient[(patient[base_eval_col] == True) & (patient[base_active_col] == False)]

        n_at_risk = 0
        n_inc = 0
        times = []
        py = 0.0
        event_dates = []

        for pid in at_risk[PATIENT_ID_COL]:
            fu = long_df[
                (long_df[PATIENT_ID_COL] == pid)
                & (long_df["visit_order"] > 1)
                & (long_df[eval_col])
            ]
            if fu.empty:
                continue
            n_at_risk += 1
            py += max(float(fu["time_from_baseline_yrs"].max()), 0.0)
            ev = fu[fu[active_col]].sort_values("time_from_baseline_yrs").head(1)
            if len(ev):
                n_inc += 1
                times.append(ev.iloc[0]["time_from_baseline_yrs"])
                event_dates.append(ev.iloc[0]["visit_date_min"])

        rows.append([
            key,
            meta["label"],
            n_at_risk,
            n_inc,
            (100 * n_inc / n_at_risk) if n_at_risk else np.nan,
            np.nanmedian(times) if times else np.nan,
            py,
            (100 * n_inc / py) if py else np.nan,
            min(event_dates) if event_dates else pd.NaT,
            max(event_dates) if event_dates else pd.NaT,
        ])

    return pd.DataFrame(rows, columns=[
        "domain", "domain_label", "n_at_risk", "n_incident", "pct_incident",
        "median_time_to_domain_yrs", "person_years_observed",
        "incidence_rate_per_100_py", "first_event_date_min", "first_event_date_max",
    ])


def _fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    r1 = a + b
    c1 = a + c

    def prob(x: int) -> float:
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)

    lo = max(0, r1 - (n - c1))
    hi = min(r1, c1)
    p_obs = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs + 1e-12))


def _bh(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort([1 if pd.isna(p) else p for p in pvals])
    adj = [np.nan] * m
    prev = 1.0
    for rank, i in enumerate(order[::-1], start=1):
        p = pvals[i]
        if pd.isna(p):
            continue
        val = min(prev, p * m / (m - rank + 1))
        adj[i] = val
        prev = val
    return adj


def make_pairwise_table(
    patient: pd.DataFrame,
    glandular_active_col: str = "baseline_glandular_active",
    glandular_eval_col: str = "baseline_glandular_evaluable",
) -> pd.DataFrame:
    """Baseline (1st-visit) 2x2 associations between sicca/glandular status and
    each ESSDAI extraglandular domain: prevalence ratio, risk difference, odds
    ratio (Haldane-Anscombe corrected when any cell is zero), Fisher exact or
    chi-square depending on minimum expected cell count, and BH-FDR across the
    primary domain set. Mirrors the dx-anchored script's make_pairwise_table,
    but computed on the 1st-visit baseline rather than a dx-anchored baseline.
    """
    rows = []
    for key in PRIMARY_EG_DOMAIN_KEYS:
        meta = EXTRAGLANDULAR_DOMAINS[key]
        active_col = f"baseline_eg_{key}_active"
        eval_col = f"baseline_eg_{key}_evaluable"
        comp = patient[(patient[glandular_eval_col] == True) & (patient[eval_col] == True)]
        gp = comp[glandular_active_col].astype(bool)
        dp = comp[active_col].astype(bool)
        a = int((gp & dp).sum()); b = int((gp & ~dp).sum())
        c = int((~gp & dp).sum()); d = int((~gp & ~dp).sum())
        n = a + b + c + d
        miss = len(patient) - n
        if n:
            exp = [(a + b) * (a + c) / n, (a + b) * (b + d) / n, (c + d) * (a + c) / n, (c + d) * (b + d) / n]
            min_exp = min(exp)
        else:
            min_exp = np.nan
        fisher = bool(n and min_exp < 5)
        if fisher:
            p = _fisher_exact_p(a, b, c, d)
        elif all(x > 0 for x in [(a + b), (c + d), (a + c), (b + d)]):
            chi2 = ((a * d - b * c) ** 2 * n) / ((a + b) * (c + d) * (a + c) * (b + d))
            p = math.erfc(math.sqrt(chi2 / 2))
        else:
            p = np.nan
        aa, bb, cc, dd = (a, b, c, d) if min(a, b, c, d) > 0 else (a + .5, b + .5, c + .5, d + .5)
        risk1 = aa / (aa + bb); risk0 = cc / (cc + dd)
        pr = risk1 / risk0 if risk0 else np.nan
        se_log_pr = math.sqrt((1 / aa) - (1 / (aa + bb)) + (1 / cc) - (1 / (cc + dd))) if aa and cc else np.nan
        orv = (aa * dd) / (bb * cc) if bb and cc else np.nan
        se_log_or = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
        rows.append({
            "domain": key, "domain_label": meta["label"],
            "n_complete": n, "n_missing_or_not_evaluable": miss,
            "glandular_pos_domain_pos": a, "glandular_pos_domain_neg": b,
            "glandular_neg_domain_pos": c, "glandular_neg_domain_neg": d,
            "pct_domain_active_if_glandular_pos": 100 * a / (a + b) if (a + b) else np.nan,
            "pct_domain_active_if_glandular_neg": 100 * c / (c + d) if (c + d) else np.nan,
            "prevalence_ratio": pr,
            "pr_ci_low": math.exp(math.log(pr) - 1.96 * se_log_pr) if pr and not pd.isna(se_log_pr) else np.nan,
            "pr_ci_high": math.exp(math.log(pr) + 1.96 * se_log_pr) if pr and not pd.isna(se_log_pr) else np.nan,
            "risk_difference": (a / (a + b) - c / (c + d)) if (a + b) and (c + d) else np.nan,
            "odds_ratio": orv,
            "or_ci_low": math.exp(math.log(orv) - 1.96 * se_log_or) if orv else np.nan,
            "or_ci_high": math.exp(math.log(orv) + 1.96 * se_log_or) if orv else np.nan,
            "min_expected_cell_count": min_exp,
            "test_used": "fisher_exact" if fisher else "chi_square",
            "p_value": p,
            "haldane_anscombe_applied": min(a, b, c, d) == 0,
        })
    out = pd.DataFrame(rows)
    out["p_adj_fdr_bh"] = _bh(out["p_value"].tolist())
    out["significant_nominal_0_05"] = out["p_value"] < 0.05
    out["significant_fdr_0_05"] = out["p_adj_fdr_bh"] < 0.05
    return out


def make_pairwise_heatmap(pairwise: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(pairwise) + 1)))
    vals = np.log2(pairwise["prevalence_ratio"].replace([np.inf, -np.inf], np.nan)).to_numpy().reshape(-1, 1)
    im = ax.imshow(vals, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_yticks(range(len(pairwise))); ax.set_yticklabels(pairwise["domain_label"])
    ax.set_xticks([0]); ax.set_xticklabels(["log2(PR)"])
    for i, r in pairwise.reset_index(drop=True).iterrows():
        star = "*" if bool(r["significant_fdr_0_05"]) else ""
        ax.text(0, i, f"PR {r['prevalence_ratio']:.2f}\nN {int(r['n_complete'])}\nFDR {r['p_adj_fdr_bh']:.3g}{star}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="log2(prevalence ratio)")
    ax.set_title("1st-visit baseline pairwise associations")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def assign_plot_group(row: pd.Series) -> int:
    if row.get("baseline_status") == "overlap":
        return 0
    if bool(row.get("incident_extraglandular", False)):
        return 1
    if bool(row.get("fu_overlap_ever", False)):
        return 2
    if row.get("fu_n_evaluable_visits", 0) > 0:
        return 3
    return 4


def prepare_timeline_plot_data(long_df: pd.DataFrame, patient_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, float]:
    patient_order_df = patient_summary.copy()
    if "max_time_from_baseline_yrs" not in patient_order_df.columns:
        max_followup = long_df.groupby(PATIENT_ID_COL)["time_from_baseline_yrs"].max().rename("max_time_from_baseline_yrs")
        patient_order_df = patient_order_df.merge(max_followup, on=PATIENT_ID_COL, how="left")
    patient_order_df["plot_group"] = patient_order_df.apply(assign_plot_group, axis=1)
    patient_order_df["first_event_sort"] = patient_order_df["time_to_new_domain_yrs"].fillna(999)
    patient_order_df["followup_sort"] = patient_order_df["max_time_from_baseline_yrs"].fillna(0)
    patient_order_df = patient_order_df.sort_values(
        ["plot_group", "first_event_sort", "followup_sort", "n_visits"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    patient_order_df["patient_plot_id"] = [f"P{i+1:03d}" for i in range(len(patient_order_df))]
    patient_order_df["y"] = np.arange(len(patient_order_df))[::-1]
    plot_df = long_df.merge(
        patient_order_df[[PATIENT_ID_COL, "patient_plot_id", "y", "plot_group"]],
        on=PATIENT_ID_COL,
        how="inner",
    )
    patient_ranges = (
        plot_df.groupby(["patient_plot_id", "y"], as_index=False)
        .agg(xmin=("time_from_baseline_yrs", "min"), xmax=("time_from_baseline_yrs", "max"), n_visits=("visit_order", "max"))
    )
    group_breaks = patient_order_df.groupby("plot_group")["y"].min().sort_values().values
    xmax = np.nanmax(plot_df["time_from_baseline_yrs"]) if len(plot_df) else 1
    xmax_plot = max(1, min(float(np.ceil(xmax)), 15))
    return patient_order_df, plot_df, patient_ranges, group_breaks, xmax_plot


def add_time_references(ax: plt.Axes, xmax_plot: float, *, label: bool = True, linewidth: float = 1.2, alpha: float = 0.85) -> None:
    time_refs = [
        (1/52.1775, "1 wk", "#F4A340"),
        (1/12, "1 mo", "#E6C229"),
        (0.5, "6 mo", "#2ECC71"),
        (2, "2 yr", "#3498DB"),
        (4, "4 yr", "#2ECC71"),
        (6, "6 yr", "#9B59B6"),
        (8, "8 yr", "#E74C3C"),
        (10, "10 yr", "#1ABC9C"),
    ]
    for x, text, color in time_refs:
        if x <= xmax_plot:
            ax.axvline(x, color=color, linestyle="--", linewidth=linewidth, alpha=alpha, zorder=0)
            if label:
                ymin, ymax = ax.get_ylim()
                ax.text(x + 0.03, ymax - 0.02 * (ymax - ymin), text, color=color, fontsize=8, fontweight="bold", ha="left", va="top")


def make_overlap_timeline_plot(long_df: pd.DataFrame, patient: pd.DataFrame, path: Path) -> None:
    patient_order_df, plot_df, patient_ranges, group_breaks, xmax_plot = prepare_timeline_plot_data(long_df, patient)
    n_patients = patient_order_df.shape[0]
    fig_width = 14
    fig_height = max(7, min(18, 0.075 * n_patients + 4))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=False)
    ax.set_ylim(-1, n_patients)

    for _, r in patient_ranges.iterrows():
        ax.hlines(y=r["y"], xmin=0, xmax=max(r["xmax"], 0.05), color="#C8CDD2", linewidth=0.55, alpha=0.8, zorder=1)

    d_insuff = plot_df[plot_df["overlap_status"].eq("insufficient_info")]
    ax.scatter(d_insuff["time_from_baseline_yrs"], d_insuff["y"], s=16, marker="o", facecolors="#F4F4F4", edgecolors="#BDBDBD", linewidths=0.35, alpha=0.95, zorder=2, label="Insufficient info")

    d_no = plot_df[plot_df["overlap_evaluable"].eq(True) & plot_df["overlap_active"].eq(False)]
    ax.scatter(d_no["time_from_baseline_yrs"], d_no["y"], s=18, marker="o", color="#8E8E8E", alpha=0.75, zorder=3, label="No overlap")

    d_ov = plot_df[plot_df["overlap_active"].eq(True)]
    norm = Normalize(vmin=max(2, np.nanmin(d_ov["overlap_intensity_count"])) if len(d_ov) else 2, vmax=max(3, np.nanmax(d_ov["overlap_intensity_count"])) if len(d_ov) else 3)
    sc = ax.scatter(d_ov["time_from_baseline_yrs"], d_ov["y"], s=34, marker="s", c=d_ov["overlap_intensity_count"], cmap="viridis", norm=norm, edgecolors="white", linewidths=0.25, alpha=0.95, zorder=4, label="Overlap")

    add_time_references(ax, xmax_plot, label=True)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.grid(axis="y", visible=False)

    tick_every = 1 if n_patients <= 60 else 5 if n_patients <= 120 else 10
    ytick_df = patient_order_df.iloc[::tick_every]
    ax.set_yticks(ytick_df["y"])
    ax.set_yticklabels(ytick_df["patient_plot_id"])
    for yb in group_breaks[:-1]:
        ax.axhline(y=yb - 0.5, color="#555555", linewidth=0.8, alpha=0.35, zorder=0)

    n_eval = int(patient["fu_has_evaluable_overlap"].sum())
    n_ov = int(patient["fu_overlap_ever"].sum())
    ax.set_title(f"Glandular/extraglandular overlap timeline\nTotal patients: {len(patient)} | Evaluable follow-up: {n_eval} | Overlap during follow-up: {n_ov}", pad=12)
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", label="No overlap", markerfacecolor="#8E8E8E", markeredgecolor="#8E8E8E", markersize=5),
        Line2D([0], [0], marker="o", color="none", label="Insufficient info", markerfacecolor="#F4F4F4", markeredgecolor="#BDBDBD", markersize=5),
        Line2D([0], [0], marker="s", color="none", label="Overlap", markerfacecolor="#3B528B", markeredgecolor="white", markersize=6),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True, title="Visit status")
    if len(d_ov) > 0:
        cbar = fig.colorbar(sc, ax=ax, pad=0.012, fraction=0.035)
        cbar.set_label("Active glandular + extraglandular count")
    ax.set_xlabel("Time from baseline (years)"); ax.set_ylabel("Patients")
    ax.set_xlim(-0.05, xmax_plot + 0.1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.18)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    assert path.exists(); assert path.stat().st_size > 10_000
    print(f"[FIGURE SAVED] {path}"); print(f"Patients plotted: {n_patients}"); print(f"X-axis max years: {xmax_plot}")


def make_domain_timeline_plot(long_df: pd.DataFrame, patient: pd.DataFrame, path: Path) -> None:
    patient_order_df, plot_df, patient_ranges, group_breaks, xmax_plot = prepare_timeline_plot_data(long_df, patient)
    n_patients = patient_order_df.shape[0]
    domain_order = ["Constitutional", "Lymphadenopathy", "Articular", "Cutaneous", "Pulmonary", "Renal", "Muscular", "PNS", "CNS", "Hematologic", "Biological"]
    domain_active_cols = {meta["label"]: meta["active_col"] for meta in EXTRAGLANDULAR_DOMAINS.values()}
    domain_eval_cols = {meta["label"]: f"eg_{key}_evaluable" for key, meta in EXTRAGLANDULAR_DOMAINS.items()}
    domain_colors = {"Constitutional": "#1F77B4", "Lymphadenopathy": "#FF7F0E", "Articular": "#2CA02C", "Cutaneous": "#D62728", "Pulmonary": "#9467BD", "Renal": "#8C564B", "Muscular": "#E377C2", "PNS": "#7F7F7F", "CNS": "#BCBD22", "Hematologic": "#17BECF", "Biological": "#004D40"}
    n_domains = len(domain_order)
    n_patient_domain_rows = n_patients * n_domains
    fig_height = max(12, min(24, max(1.25 * n_domains + 5, 0.035 * n_patient_domain_rows + 5)))
    fig, axes = plt.subplots(n_domains, 1, figsize=(16, fig_height), sharex=True, sharey=True, constrained_layout=False)
    if n_domains == 1:
        axes = [axes]

    for ax, domain in zip(axes, domain_order):
        active_col = domain_active_cols[domain]
        eval_col = domain_eval_cols.get(domain)
        for _, r in patient_ranges.iterrows():
            ax.hlines(y=r["y"], xmin=0, xmax=max(r["xmax"], 0.05), color="#D3D7DB", linewidth=0.35, alpha=0.55, zorder=1)
        if eval_col in plot_df.columns:
            d_inactive = plot_df[plot_df[eval_col].eq(True) & plot_df[active_col].eq(False)]
        else:
            d_inactive = plot_df[plot_df[active_col].notna() & plot_df[active_col].eq(False)]
        ax.scatter(d_inactive["time_from_baseline_yrs"], d_inactive["y"], s=5, marker="o", color="#DADADA", alpha=0.35, zorder=2)
        d_active = plot_df[plot_df[active_col].eq(True)]
        ax.scatter(d_active["time_from_baseline_yrs"], d_active["y"], s=24, marker="x", color=domain_colors[domain], linewidths=0.95, alpha=0.95, zorder=3)
        ax.text(0.005, 0.82, domain, transform=ax.transAxes, ha="left", va="center", fontsize=9, fontweight="bold", color=domain_colors[domain], bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#DDDDDD", alpha=0.9))
        ax.grid(axis="x", linestyle=":", linewidth=0.45, alpha=0.30); ax.grid(axis="y", visible=False)
        add_time_references(ax, xmax_plot, label=False, linewidth=0.8, alpha=0.45)
        for yb in group_breaks[:-1]:
            ax.axhline(y=yb - 0.5, color="#555555", linewidth=0.6, alpha=0.25, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC"); ax.spines["bottom"].set_color("#CCCCCC")

    top_ax = axes[0]
    for x, label, color in [(1/52.1775, "1 wk", "#F4A340"), (1/12, "1 mo", "#E6C229"), (0.5, "6 mo", "#2ECC71"), (2, "2 yr", "#3498DB"), (4, "4 yr", "#2ECC71"), (6, "6 yr", "#9B59B6"), (8, "8 yr", "#E74C3C"), (10, "10 yr", "#1ABC9C")]:
        if x <= xmax_plot:
            top_ax.text(x + 0.03, n_patients - 1, label, color=color, fontsize=7, fontweight="bold", ha="left", va="top")

    tick_every = 5 if n_patients <= 60 else 10 if n_patients <= 120 else 20
    ytick_df = patient_order_df.iloc[::tick_every]
    for ax in axes:
        ax.set_yticks(ytick_df["y"]); ax.set_yticklabels(ytick_df["patient_plot_id"])
        ax.set_xlim(-0.05, xmax_plot + 0.1); ax.set_ylim(-1, n_patients)
    fig.suptitle("ESSDAI extraglandular domain activity timeline\nX marks active ESSDAI extraglandular domain at a recorded visit", fontsize=13, fontweight="bold", y=0.985)
    axes[-1].set_xlabel("Time from baseline (years)")
    fig.text(0.015, 0.5, "Patients", va="center", rotation="vertical", fontsize=10)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.945, bottom=0.07, hspace=0.16)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    assert path.exists(); assert path.stat().st_size > 10_000
    print(f"[FIGURE SAVED] {path}"); print(f"Patients plotted: {n_patients}"); print(f"X-axis max years: {xmax_plot}")

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
        row = {PATIENT_ID_COL: pid, "baseline_date": base["baseline_date"], "baseline_status": base["overlap_status"], "baseline_glandular_active": bool(base["glandular_active"]), "baseline_glandular_evaluable": bool(base["glandular_evaluable"]), "baseline_extraglandular_active": bool(base["extraglandular_active"]), "baseline_extraglandular_evaluable": bool(base["extraglandular_evaluable"]), "fu_n_visits": len(fu), "fu_n_evaluable_visits": len(fu_eval), "fu_has_evaluable_overlap": len(fu_eval) > 0, "fu_glandular_ever": bool(fu_eval["glandular_active"].any()), "fu_extraglandular_ever": bool(fu_eval["extraglandular_active"].any()), "fu_overlap_ever": bool(fu_eval["overlap_active"].any()), "max_overlap_intensity_count": int(fu_eval["overlap_intensity_count"].max()) if len(fu_eval) else 0, "fu_active_extraglandular_domains_ever": ";".join(sorted(set(d for s in fu["active_extraglandular_domains"] for d in str(s).split(";") if d))), "incident_denominator": incident_denom, "incident_extraglandular": first is not None, "first_incident_date": first["visit_date_min"] if first is not None else pd.NaT, "time_to_new_domain_yrs": first["time_from_baseline_yrs"] if first is not None else np.nan, "incident_domains_at_first_event": first["active_extraglandular_domains"] if first is not None else "", "n_visits": len(g), "n_evaluable_visits": int(g["overlap_evaluable"].sum())}
        for key, meta in EXTRAGLANDULAR_DOMAINS.items():
            row[f"fu_eg_{key}_ever"] = bool(fu_eval[meta["active_col"]].any()) if len(fu_eval) else False
            # NEW: per-domain baseline active/evaluable flags, needed for the
            # domain-specific incidence table (make_domain_incident_table).
            row[f"baseline_eg_{key}_active"] = bool(base[meta["active_col"]])
            row[f"baseline_eg_{key}_evaluable"] = bool(base[f"eg_{key}_evaluable"])
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

    follow = make_followup_table(patient)
    incident = make_incident_table(patient)
    domain_incident = make_domain_incident_table(patient, long_df)  # NEW
    pairwise = make_pairwise_table(patient)  # NEW

    long_df.to_parquet(LONGITUDINAL_OUT, index=False); patient.to_parquet(PATIENT_SUMMARY_OUT, index=False)
    follow.to_csv(FOLLOWUP_TABLE, index=False); incident.to_csv(INCIDENT_TABLE, index=False)
    domain_incident.to_csv(DOMAIN_INCIDENT_TABLE, index=False)  # NEW
    pairwise.to_csv(PAIRWISE_TABLE, index=False)  # NEW
    make_overlap_timeline_plot(long_df, patient, TIMELINE_FIG); make_domain_timeline_plot(long_df, patient, DOMAIN_TIMELINE_FIG)
    make_pairwise_heatmap(pairwise, PAIRWISE_HEATMAP_FIG)  # NEW
    output_paths = [FOLLOWUP_TABLE, INCIDENT_TABLE, DOMAIN_INCIDENT_TABLE, PAIRWISE_TABLE, LONGITUDINAL_OUT, PATIENT_SUMMARY_OUT, TIMELINE_FIG, DOMAIN_TIMELINE_FIG, PAIRWISE_HEATMAP_FIG]
    qc = run_qc(raw, long_df, patient, missing_optional, output_paths, invalid_visit_dates)
    qc.to_csv(QC_OUT, index=False)
    run_qc(raw, long_df, patient, missing_optional, output_paths + [QC_OUT], invalid_visit_dates)

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

    print("\nPer-domain incidence during follow-up (1st-visit baseline):")
    for _, r in domain_incident.sort_values("pct_incident", ascending=False).iterrows():
        print(
            f"{r['domain_label']}: {r['n_incident']}/{r['n_at_risk']} ({r['pct_incident']:.1f}%), "
            f"{r['incidence_rate_per_100_py']:.2f}/100 PY"
        )

    print("\nBaseline (1st-visit) pairwise associations, sorted by FDR:")
    for _, r in pairwise.sort_values(["p_adj_fdr_bh", "p_value"]).iterrows():
        star = "*" if bool(r["significant_fdr_0_05"]) else ""
        print(
            f"{r['domain_label']}: PR={r['prevalence_ratio']:.2f} "
            f"(95% CI {r['pr_ci_low']:.2f}-{r['pr_ci_high']:.2f}), "
            f"OR={r['odds_ratio']:.2f}, p={r['p_value']:.4g}, "
            f"FDR={r['p_adj_fdr_bh']:.4g}{star}, N={int(r['n_complete'])}"
        )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""ITEM 1.3 — Descriptive ESSDAI/ESSPRI on the clinical episode spine."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402

INPUT_PATH = common.EPISODE_SPINE_PARQUET
ID_COL = "patient_id"
ESSDAI_TOTAL_COL = "essdai__essdai_total_score"
VISIT_NUMBER_COL = "clinical_visit_number"
ESSDAI_DOMAIN_VARS = {
    "Pulmonary": "essdai__pulmonary", "Hematologic": "essdai__hematologic",
    "Lymphadenopathy": "essdai__hema_lphdenopthy", "Constitutional": "essdai__constitutional",
    "Cutaneous": "essdai__cutaneous", "Glandular": "essdai__gland_swell",
    "Renal": "essdai__renal", "Peripheral nervous system": "essdai__neuro_peripheral",
    "Central nervous system": "essdai__cns", "Articular": "essdai__articular_domain",
    "Muscular": "essdai__muscular_domain", "Biological": "essdai__biological_domain",
}
ESSPRI_COMPONENTS = {
    "esspri_questionnaire__dryness": "esspri_dryness",
    "esspri_questionnaire__fatigue": "esspri_fatigue",
    "esspri_questionnaire__pain": "esspri_pain",
}
TABLE_DIR = common.OUTPUTS_DIR / "tables" / "blockA"
FIGURE_DIR = common.OUTPUTS_DIR / "figures" / "blockA"
QC_DIR = common.BLOCKA_QC_DIR
INTERMEDIATE_DIR = common.INTERMEDIATE_DATA_DIR


def ensure_dirs():
    for directory in (TABLE_DIR, FIGURE_DIR, QC_DIR, INTERMEDIATE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _as_bool(series):
    """Interpret upstream booleans without changing their meaning."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})


def require_spine_columns(df):
    required = {
        ID_COL, "clinical_episode_id", "clinical_anchor_date", VISIT_NUMBER_COL,
        "clinical_visit", "is_clinical_baseline", "clinical_baseline_episode_id",
        "clinical_baseline_date", ESSDAI_TOTAL_COL,
    } | set(ESSPRI_COMPONENTS) | set(ESSDAI_DOMAIN_VARS.values())
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Canonical clinical episode spine is missing required columns: {missing}")


def derive_essdai_total(df):
    """Copy only the upstream canonical ESSDAI total, applying existing 0–123 QC."""
    values = pd.to_numeric(df[ESSDAI_TOTAL_COL], errors="coerce")
    invalid = (values < 0) | (values > 123)
    df["essdai_total"] = values.mask(invalid)
    return df, int(invalid.sum())


def derive_esspri_total(df):
    """Calculate observed ESSPRI only from complete dryness/fatigue/pain triples."""
    out_of_range = 0
    for source, target in ESSPRI_COMPONENTS.items():
        values = pd.to_numeric(df[source], errors="coerce")
        invalid = (values < 0) | (values > 10)
        out_of_range += int(invalid.sum())
        df[target] = values.mask(invalid)
    components = list(ESSPRI_COMPONENTS.values())
    df["esspri_n_components"] = df[components].notna().sum(axis=1).astype("int64")
    df["esspri_total"] = df[components].mean(axis=1).where(df["esspri_n_components"].eq(3))
    return df, out_of_range


def derive_domain_activity(df):
    for domain, column in ESSDAI_DOMAIN_VARS.items():
        values = pd.to_numeric(df[column], errors="coerce")
        df[f"essdai_domain_{domain}_score"] = values
        df[f"essdai_domain_{domain}_active"] = np.where(values.notna(), (values > 0).astype(float), np.nan)
    return df


def select_baseline(df):
    """Select the authoritative clinical baseline; never substitute another episode."""
    return df.loc[df["clinical_visit_bool"] & df["is_clinical_baseline_bool"]].copy()


def summarize_continuous(series):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    keys = ("median", "q1", "q3", "mean", "sd", "min", "max")
    if clean.empty:
        return {"n": 0, **dict.fromkeys(keys, np.nan)}
    return {"n": int(clean.count()), "median": clean.median(), "q1": clean.quantile(.25),
            "q3": clean.quantile(.75), "mean": clean.mean(), "sd": clean.std(ddof=1),
            "min": clean.min(), "max": clean.max()}


def _fmt(value, digits=1):
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def _fmt_median_iqr_range(stats):
    return (f"{_fmt(stats['median'])} (IQR {_fmt(stats['q1'])}–{_fmt(stats['q3'])}; "
            f"range {_fmt(stats['min'])}–{_fmt(stats['max'])})")


def summarize_baseline_activity(baseline_df):
    essdai, esspri = (summarize_continuous(baseline_df[c]) for c in ("essdai_total", "esspri_total"))
    n_ge5 = int((baseline_df["essdai_total"] >= 5).sum())
    pct_ge5 = 100 * n_ge5 / essdai["n"] if essdai["n"] else np.nan
    rows = [
        {"section": "Disease activity", "variable": "ESSDAI, median (IQR; range)", "n": essdai["n"], "summary": _fmt_median_iqr_range(essdai), "value_numeric": essdai["median"], "min": essdai["min"], "max": essdai["max"], "denominator": essdai["n"], "note": "Clinical baseline"},
        {"section": "Disease activity", "variable": "ESSDAI ≥5, n (%)", "n": n_ge5, "summary": f"{n_ge5} / {essdai['n']} ({_fmt(pct_ge5)}%)", "value_numeric": pct_ge5, "denominator": essdai["n"], "note": "Systemic disease activity at clinical baseline"},
        {"section": "Disease activity", "variable": "ESSPRI, median (IQR; range)", "n": esspri["n"], "summary": _fmt_median_iqr_range(esspri), "value_numeric": esspri["median"], "min": esspri["min"], "max": esspri["max"], "denominator": esspri["n"], "note": "Mean of dryness, fatigue, and pain; all 3 components required"},
    ]
    return pd.DataFrame(rows), {"essdai": essdai, "esspri": esspri, "n_ge5": n_ge5, "pct_ge5": pct_ge5}


def summarize_domains_baseline(baseline_df):
    rows = []
    for domain, variable in ESSDAI_DOMAIN_VARS.items():
        active = baseline_df[f"essdai_domain_{domain}_active"]
        n_nonmissing, n_active = int(active.notna().sum()), int(active.eq(1).sum())
        rows.append({"domain": domain, "variable": variable, "n_nonmissing": n_nonmissing,
                     "n_active": n_active, "pct_active": 100 * n_active / n_nonmissing if n_nonmissing else np.nan})
    return pd.DataFrame(rows).sort_values("pct_active", ascending=False, na_position="last")


def summarize_by_visit(clinical_df):
    """Sequence summaries; visit number is descriptive and not elapsed time."""
    order = sorted(clinical_df[VISIT_NUMBER_COL].dropna().unique())
    visit_rows, domain_rows = [], []
    for visit_number, group in clinical_df.groupby(VISIT_NUMBER_COL, dropna=False):
        for measure, prefix in (("essdai_total", "essdai"), ("esspri_total", "esspri")):
            stats = summarize_continuous(group[measure]); ge5 = int(group[measure].ge(5).sum())
            visit_rows.append({VISIT_NUMBER_COL: visit_number, "measure": prefix, f"n_{prefix}": stats["n"],
                               **{k: stats[k] for k in ("median", "q1", "q3", "mean", "sd", "min", "max")},
                               f"pct_{prefix}_ge5": 100 * ge5 / stats["n"] if stats["n"] else np.nan})
        for domain in ESSDAI_DOMAIN_VARS:
            active = group[f"essdai_domain_{domain}_active"]
            n_nonmissing, n_active = int(active.notna().sum()), int(active.eq(1).sum())
            domain_rows.append({VISIT_NUMBER_COL: visit_number, "domain": domain, "n_nonmissing": n_nonmissing,
                                "n_active": n_active, "pct_active": 100 * n_active / n_nonmissing if n_nonmissing else np.nan})
    return pd.DataFrame(visit_rows), pd.DataFrame(domain_rows), order


def make_baseline_domain_bar(summary):
    plot_df = summary.sort_values("pct_active")
    fig, ax = plt.subplots(figsize=(9, max(4, .35 * len(plot_df) + 1)))
    ax.barh(plot_df["domain"], plot_df["pct_active"].fillna(0), color="#4C78A8")
    for i, row in enumerate(plot_df.itertuples(index=False)):
        ax.text((0 if pd.isna(row.pct_active) else row.pct_active) + .5, i,
                f"{_fmt(row.pct_active)}% ({row.n_active}/{row.n_nonmissing})", va="center", fontsize=8)
    ax.set(xlabel="% active at clinical baseline", ylabel="ESSDAI domain", title="Baseline ESSDAI domain activity")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "01_essdai_esspri_essdai_domains_baseline_bar.pdf"); plt.close(fig)


def make_distribution_plots(clinical_df, visit_order, domain_by_visit):
    for measure, ylabel, title, filename in (
        ("essdai_total", "ESSDAI total", "ESSDAI total distribution by clinical visit number", "01_essdai_esspri_essdai_total_distribution_by_clinical_visit_number.pdf"),
        ("esspri_total", "ESSPRI total", "ESSPRI distribution by clinical visit number", "01_essdai_esspri_esspri_distribution_by_clinical_visit_number.pdf"),
    ):
        fig, ax = plt.subplots(figsize=(max(8, .45 * len(visit_order)), 5))
        sns.boxplot(data=clinical_df, x=VISIT_NUMBER_COL, y=measure, order=visit_order, ax=ax, color="#9ECAE1")
        ax.axhline(5, color="firebrick", linestyle="--", linewidth=1)
        ax.set(xlabel="Clinical visit number (sequence, not uniform time)", ylabel=ylabel, title=title)
        fig.tight_layout(); fig.savefig(FIGURE_DIR / filename); plt.close(fig)
    heat = domain_by_visit.pivot(index="domain", columns=VISIT_NUMBER_COL, values="pct_active").reindex(columns=visit_order)
    fig, ax = plt.subplots(figsize=(max(8, .45 * len(visit_order)), max(5, .35 * len(heat))))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "% active"}, ax=ax)
    ax.set(title="ESSDAI domain activity by clinical visit number", xlabel="Clinical visit number (sequence, not uniform time)", ylabel="ESSDAI domain")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "01_essdai_esspri_essdai_domain_activity_by_clinical_visit_number.pdf"); plt.close(fig)


def make_qc_report(df, clinical_df, baseline_df, essdai_oor, esspri_oor):
    baseline_counts = baseline_df.groupby(ID_COL).size()
    baseline_episode_match = baseline_df["clinical_episode_id"].astype("string").eq(baseline_df["clinical_baseline_episode_id"].astype("string"))
    baseline_date_match = baseline_df["clinical_anchor_date"].eq(baseline_df["clinical_baseline_date"])
    has_essdai, has_esspri = baseline_df["essdai_total"].notna(), baseline_df["esspri_total"].notna()
    rows = [
        ("n_unique_patients", int(df[ID_COL].nunique(dropna=True))),
        ("n_clinical_episodes", int(df[[ID_COL, "clinical_episode_id"]].drop_duplicates().shape[0])),
        ("n_clinical_visits", len(clinical_df)),
        ("n_duplicate_patient_episode_ids", int(df.duplicated([ID_COL, "clinical_episode_id"]).sum())),
        ("n_patients_with_multiple_clinical_baselines", int(baseline_counts.gt(1).sum())),
        ("n_baseline_episode_mismatches", int((~baseline_episode_match).sum())),
        ("n_baseline_date_mismatches", int((~baseline_date_match).sum())),
        ("n_nonclinical_baselines", int((df["is_clinical_baseline_bool"] & ~df["clinical_visit_bool"]).sum())),
        ("n_clinical_baselines", len(baseline_df)),
        ("n_clinical_visits_with_essdai", int(clinical_df["essdai_total"].notna().sum())),
        ("n_clinical_visits_without_essdai", int(clinical_df["essdai_total"].isna().sum())),
        ("n_baseline_with_essdai", int(has_essdai.sum())), ("n_baseline_without_essdai", int((~has_essdai).sum())),
        ("n_baseline_with_complete_esspri", int(has_esspri.sum())), ("n_baseline_without_complete_esspri", int((~has_esspri).sum())),
        ("n_baseline_with_both", int((has_essdai & has_esspri).sum())),
        ("n_baseline_with_neither", int((~has_essdai & ~has_esspri).sum())),
        ("essdai_out_of_range_values", essdai_oor), ("esspri_component_out_of_range_values", esspri_oor),
    ]
    for n in range(4):
        rows.append((f"n_clinical_visits_with_esspri_{n}_components", int(clinical_df["esspri_n_components"].eq(n).sum())))
        rows.append((f"n_baseline_with_esspri_{n}_components", int(baseline_df["esspri_n_components"].eq(n).sum())))
    return pd.DataFrame(rows, columns=["metric", "value"])


def episode_output_columns(df):
    identifiers = [ID_COL, "clinical_episode_id", "clinical_anchor_date", VISIT_NUMBER_COL,
                   "clinical_baseline_episode_id", "clinical_baseline_date", "is_clinical_baseline",
                   "time_since_clinical_baseline_days", "time_since_clinical_baseline_years"]
    measures = ["essdai_total", "esspri_dryness", "esspri_fatigue", "esspri_pain", "esspri_n_components", "esspri_total"]
    domains = [c for domain in ESSDAI_DOMAIN_VARS for c in (f"essdai_domain_{domain}_score", f"essdai_domain_{domain}_active")]
    return [c for c in identifiers + measures + domains if c in df.columns]


def write_outputs(clinical_df, baseline_df, activity, domains, by_visit, domains_by_visit, qc):
    baseline_df.to_csv(QC_DIR / "01_essdai_esspri_baseline_dataset.csv", index=False)
    clinical_df[episode_output_columns(clinical_df)].to_csv(INTERMEDIATE_DIR / "01_essdai_esspri_episode_level.csv", index=False)
    activity.to_csv(TABLE_DIR / "01_essdai_esspri_disease_activity_summary.csv", index=False)
    domains.to_csv(TABLE_DIR / "01_essdai_esspri_essdai_domain_baseline.csv", index=False)
    by_visit.to_csv(TABLE_DIR / "01_essdai_esspri_by_clinical_visit_number_summary.csv", index=False)
    domains_by_visit.to_csv(TABLE_DIR / "01_essdai_esspri_domain_by_clinical_visit_number_summary.csv", index=False)
    qc.to_csv(QC_DIR / "01_essdai_esspri_qc_report.csv", index=False)


def main():
    ensure_dirs()
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Step 00 episode spine not found: {INPUT_PATH}. "
            "Run src/00_build_visit_spine.py before this analysis."
        )
    df = pd.read_parquet(INPUT_PATH); require_spine_columns(df)
    df["clinical_visit_bool"] = _as_bool(df["clinical_visit"])
    df["is_clinical_baseline_bool"] = _as_bool(df["is_clinical_baseline"])
    df["clinical_anchor_date"] = pd.to_datetime(df["clinical_anchor_date"], errors="coerce")
    df["clinical_baseline_date"] = pd.to_datetime(df["clinical_baseline_date"], errors="coerce")
    if "episode_start_date" in df:
        df["episode_start_date"] = pd.to_datetime(df["episode_start_date"], errors="coerce")
    df, essdai_oor = derive_essdai_total(df); df, esspri_oor = derive_esspri_total(df); df = derive_domain_activity(df)
    sort_columns = [ID_COL, "clinical_anchor_date"] + (["episode_start_date"] if "episode_start_date" in df else []) + ["clinical_episode_id"]
    clinical_df = df.loc[df["clinical_visit_bool"] & df["clinical_anchor_date"].notna()].sort_values(sort_columns).copy()
    baseline_df = select_baseline(df)
    activity, stats = summarize_baseline_activity(baseline_df); domains = summarize_domains_baseline(baseline_df)
    by_visit, domains_by_visit, visit_order = summarize_by_visit(clinical_df)
    qc = make_qc_report(df, clinical_df, baseline_df, essdai_oor, esspri_oor)
    structural = qc.loc[qc.metric.isin(["n_duplicate_patient_episode_ids", "n_patients_with_multiple_clinical_baselines", "n_baseline_episode_mismatches", "n_baseline_date_mismatches", "n_nonclinical_baselines"])]
    if structural.value.ne(0).any():
        raise ValueError(f"Clinical episode spine structural QC failed: {dict(zip(structural.metric, structural.value))}")
    write_outputs(clinical_df, baseline_df, activity, domains, by_visit, domains_by_visit, qc)
    make_baseline_domain_bar(domains); make_distribution_plots(clinical_df, visit_order, domains_by_visit)
    print(qc.to_string(index=False)); print(f"Input: {INPUT_PATH}")
    print(f"Baseline ESSDAI: {_fmt_median_iqr_range(stats['essdai'])}; baseline ESSPRI: {_fmt_median_iqr_range(stats['esspri'])}")


if __name__ == "__main__":
    main()

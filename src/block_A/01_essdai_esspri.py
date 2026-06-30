#!/usr/bin/env python3
"""ITEM 1.3 — Baseline disease activity: ESSDAI and ESSPRI.

Builds patient-level baseline disease activity summaries and longitudinal
visit/interval summaries for the Sjögren Disease Natural History Study.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402

INPUT_PATH = Path("/data/raw/visits_long_collapsed_by_interval_codebook_corrected.parquet")
ID_COL = "ids__patient_record_number"
INTERVAL_COL = "ids__interval_name"
VISIT_DATE_COL = "ids__visit_date"

ESSDAI_TOTAL_CANDIDATES = [
    "essdai-_r__essdai_total_score",
    "essdai__essdai_total_score",
]
ESSDAI_DOMAIN_VARS = {
    "Pulmonary": "essdai__pulmonary",
    "Hematologic": "essdai__hematologic",
    "Lymphadenopathy": "essdai__hema_lphdenopthy",
    "Constitutional": "essdai__constitutional",
    "Cutaneous": "essdai__cutaneous",
    "Glandular": "essdai__gland_swell",
    "Renal": "essdai__renal",
    "Peripheral nervous system": "essdai__neuro_peripheral",
    "Central nervous system": "essdai__cns",
    "Articular": "essdai__articular_domain",
    "Muscular": "essdai__muscular_domain",
    "Biological": "essdai__biological_domain",
}
ESSPRI_COMPONENTS = [
    "esspri_questionnaire__dryness",
    "esspri_questionnaire__fatigue",
    "esspri_questionnaire__pain",
]
BASELINE_PATTERNS = re.compile(r"baseline|screening|\bbl\b|initial", flags=re.IGNORECASE)
TABLE_DIR = common.OUTPUTS_DIR / "tables" / "blockA"
FIGURE_DIR = common.OUTPUTS_DIR / "figures" / "blockA"
QC_DIR = TABLE_DIR / "qc"
INTERMEDIATE_DIR = common.PROJECT_ROOT / "data_intermediate" / "block_A"


def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)


def safe_file_stem(path: Path):
    """Return a compact filesystem-safe stem that identifies the input source."""
    stem = path.stem or "input"
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem.lower()).strip("_")
    return safe[:80] or "input"


def _date_fragments(value):
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def parse_visit_date_min(value):
    dates = [pd.to_datetime(part, errors="coerce") for part in _date_fragments(value)]
    dates = [date for date in dates if pd.notna(date)]
    return min(dates) if dates else pd.NaT


def extract_visit_year(value):
    years = []
    for part in _date_fragments(value):
        parsed = pd.to_datetime(part, errors="coerce")
        if pd.notna(parsed):
            years.append(int(parsed.year))
            continue
        match = re.search(r"(19|20)\d{2}", part)
        if match:
            years.append(int(match.group(0)))
    return min(years) if years else np.nan


def coalesce_numeric(df, columns, out_col, valid_min=None, valid_max=None):
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    out_of_range = 0
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
        invalid = pd.Series(False, index=df.index)
        if valid_min is not None:
            invalid |= values < valid_min
        if valid_max is not None:
            invalid |= values > valid_max
        out_of_range += int(invalid.sum())
        values = values.mask(invalid)
        result = result.combine_first(values)
    df[out_col] = result
    return df, out_of_range


def derive_essdai_total(df):
    return coalesce_numeric(df, ESSDAI_TOTAL_CANDIDATES, "essdai_total", valid_min=0, valid_max=123)


def derive_esspri_total(df):
    out_of_range = 0
    rename = {
        "esspri_questionnaire__dryness": "esspri_dryness",
        "esspri_questionnaire__fatigue": "esspri_fatigue",
        "esspri_questionnaire__pain": "esspri_pain",
    }
    for src, dst in rename.items():
        values = pd.to_numeric(df[src], errors="coerce") if src in df.columns else pd.Series(np.nan, index=df.index)
        invalid = (values < 0) | (values > 10)
        out_of_range += int(invalid.sum())
        df[dst] = values.mask(invalid)
    component_cols = list(rename.values())
    df["esspri_n_components"] = df[component_cols].notna().sum(axis=1)
    df["esspri_total"] = df[component_cols].mean(axis=1).where(df["esspri_n_components"] >= 2)
    return df, out_of_range


def derive_domain_activity(df):
    for domain, col in ESSDAI_DOMAIN_VARS.items():
        score_col = f"essdai_domain_{domain}_score"
        active_col = f"essdai_domain_{domain}_active"
        values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
        df[score_col] = values
        df[active_col] = np.where(values.notna(), (values > 0).astype(float), np.nan)
    return df


def select_baseline(df):
    work = df.copy()
    work["_has_activity_data"] = (
        work["essdai_total"].notna()
        | work["esspri_total"].notna()
        | work[[f"essdai_domain_{d}_score" for d in ESSDAI_DOMAIN_VARS]].notna().any(axis=1)
    )
    work["_is_explicit_baseline"] = work[INTERVAL_COL].astype("string").fillna("").str.contains(BASELINE_PATTERNS)
    work["_sort_date"] = work["visit_date_parsed"]
    work["_sort_year"] = work["visit_year"]
    rows = []
    duplicates_before = 0
    for _, group in work.groupby(ID_COL, dropna=True, sort=True):
        candidates = group[group["_is_explicit_baseline"]]
        if candidates.empty:
            candidates = group[group["_has_activity_data"]]
        if candidates.empty:
            continue
        duplicates_before += max(len(candidates) - 1, 0)
        selected = candidates.sort_values(["_sort_date", "_sort_year"], na_position="last").head(1)
        rows.append(selected)
    baseline = pd.concat(rows, ignore_index=True) if rows else work.head(0).copy()
    return baseline.drop(columns=[c for c in baseline.columns if c.startswith("_")], errors="ignore"), duplicates_before


def summarize_continuous(series):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"n": 0, "median": np.nan, "q1": np.nan, "q3": np.nan, "mean": np.nan, "sd": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(clean.count()), "median": clean.median(), "q1": clean.quantile(0.25), "q3": clean.quantile(0.75),
        "mean": clean.mean(), "sd": clean.std(ddof=1), "min": clean.min(), "max": clean.max(),
    }


def _fmt(x, digits=1):
    return "NA" if pd.isna(x) else f"{x:.{digits}f}"


def summarize_baseline_activity(baseline_df):
    essdai = summarize_continuous(baseline_df["essdai_total"])
    esspri = summarize_continuous(baseline_df["esspri_total"])
    n_ge5 = int((baseline_df["essdai_total"] >= 5).sum())
    pct_ge5 = 100 * n_ge5 / essdai["n"] if essdai["n"] else np.nan
    rows = [
        {"section": "Disease activity", "variable": "ESSDAI, median (IQR)", "n": essdai["n"], "summary": f"{_fmt(essdai['median'])} ({_fmt(essdai['q1'])}–{_fmt(essdai['q3'])})", "value_numeric": essdai["median"], "denominator": essdai["n"], "note": "Baseline"},
        {"section": "Disease activity", "variable": "ESSDAI ≥5, n (%)", "n": n_ge5, "summary": f"{n_ge5} / {essdai['n']} ({_fmt(pct_ge5)}%)", "value_numeric": pct_ge5, "denominator": essdai["n"], "note": "Systemic disease activity"},
        {"section": "Disease activity", "variable": "ESSPRI, median (IQR)", "n": esspri["n"], "summary": f"{_fmt(esspri['median'])} ({_fmt(esspri['q1'])}–{_fmt(esspri['q3'])})", "value_numeric": esspri["median"], "denominator": esspri["n"], "note": "Mean of dryness/fatigue/pain; ≥2 components required"},
    ]
    return pd.DataFrame(rows), {"essdai": essdai, "esspri": esspri, "n_ge5": n_ge5, "pct_ge5": pct_ge5}


def summarize_domains_baseline(baseline_df):
    rows = []
    for domain, var in ESSDAI_DOMAIN_VARS.items():
        active = baseline_df[f"essdai_domain_{domain}_active"]
        n_nonmissing = int(active.notna().sum())
        n_active = int((active == 1).sum())
        rows.append({"domain": domain, "variable": var, "n_nonmissing": n_nonmissing, "n_active": n_active, "pct_active": 100 * n_active / n_nonmissing if n_nonmissing else np.nan})
    return pd.DataFrame(rows).sort_values("pct_active", ascending=False, na_position="last")


def summarize_by_visit(df):
    order = df.groupby(INTERVAL_COL)["visit_year"].median().sort_values().index.tolist()
    visit_rows = []
    for interval, group in df.groupby(INTERVAL_COL, dropna=False):
        for measure, prefix in [("essdai_total", "essdai"), ("esspri_total", "esspri")]:
            stats = summarize_continuous(group[measure])
            ge5 = int((group[measure] >= 5).sum())
            visit_rows.append({"interval_name": interval, "measure": prefix, f"n_{prefix}": stats["n"], "median": stats["median"], "q1": stats["q1"], "q3": stats["q3"], "mean": stats["mean"], "sd": stats["sd"], "min": stats["min"], "max": stats["max"], f"pct_{prefix}_ge5": 100 * ge5 / stats["n"] if stats["n"] else np.nan})
    domain_rows = []
    for interval, group in df.groupby(INTERVAL_COL, dropna=False):
        for domain in ESSDAI_DOMAIN_VARS:
            active = group[f"essdai_domain_{domain}_active"]
            n_nonmissing = int(active.notna().sum())
            n_active = int((active == 1).sum())
            domain_rows.append({"interval_name": interval, "domain": domain, "n_nonmissing": n_nonmissing, "n_active": n_active, "pct_active": 100 * n_active / n_nonmissing if n_nonmissing else np.nan})
    return pd.DataFrame(visit_rows), pd.DataFrame(domain_rows), order


def make_baseline_domain_bar(domain_summary):
    plot_df = domain_summary.sort_values("pct_active", ascending=True)
    height = max(4, 0.35 * len(plot_df) + 1)
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(plot_df["domain"], plot_df["pct_active"].fillna(0), color="#4C78A8")
    for i, row in enumerate(plot_df.itertuples(index=False)):
        label = f"{_fmt(row.pct_active)}% ({row.n_active}/{row.n_nonmissing})"
        ax.text((0 if pd.isna(row.pct_active) else row.pct_active) + 0.5, i, label, va="center", fontsize=8)
    ax.set_xlabel("% active at baseline")
    ax.set_ylabel("ESSDAI domain")
    ax.set_title("Baseline ESSDAI domain activity")
    ax.text(0, -0.16, "Active defined as domain score >0; denominator excludes missing domain values.", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_essdai_domains_baseline_bar.pdf")
    plt.close(fig)


def make_distribution_plots(df, interval_order, domain_by_visit):
    plot_df = df[df[INTERVAL_COL].notna()].copy()
    for measure, ylabel, title, path in [
        ("essdai_total", "ESSDAI total", "ESSDAI total distribution by visit interval", "01_essdai_total_distribution_by_visit.pdf"),
        ("esspri_total", "ESSPRI total", "ESSPRI distribution by visit interval", "01_esspri_distribution_by_visit.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(interval_order)), 5))
        sns.boxplot(data=plot_df, x=INTERVAL_COL, y=measure, order=interval_order, ax=ax, color="#9ECAE1")
        ax.axhline(5, color="firebrick", linestyle="--", linewidth=1)
        ax.set_xlabel("Visit interval")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / path)
        plt.close(fig)
    heat = domain_by_visit.pivot(index="domain", columns="interval_name", values="pct_active").reindex(columns=interval_order)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(interval_order)), max(5, 0.35 * len(heat))))
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "% active"}, ax=ax)
    ax.set_title("ESSDAI domain activity by visit interval")
    ax.set_xlabel("Visit interval")
    ax.set_ylabel("ESSDAI domain")
    ax.text(0, -0.18, "Active = domain score >0.", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_essdai_domain_activity_by_visit.pdf")
    plt.close(fig)


def make_qc_report(df, baseline_df, duplicates_before, essdai_oor, esspri_oor, domain_summary):
    rows = [
        ("original_rows", len(df)),
        ("unique_patients", df[ID_COL].nunique(dropna=True)),
        ("patients_with_baseline_derived", len(baseline_df)),
        ("baseline_essdai_nonmissing", int(baseline_df["essdai_total"].notna().sum())),
        ("baseline_esspri_nonmissing", int(baseline_df["esspri_total"].notna().sum())),
        ("baseline_both_essdai_esspri_nonmissing", int((baseline_df["essdai_total"].notna() & baseline_df["esspri_total"].notna()).sum())),
        ("essdai_out_of_range_values", essdai_oor),
        ("esspri_component_out_of_range_values", esspri_oor),
        ("baseline_duplicate_patient_rows_before_dedup", duplicates_before),
        ("baseline_duplicate_patient_rows_after_dedup", int(baseline_df[ID_COL].duplicated().sum())),
    ]
    for col in ESSDAI_TOTAL_CANDIDATES + ESSPRI_COMPONENTS + list(ESSDAI_DOMAIN_VARS.values()):
        rows.append((f"missing__{col}", int(df[col].isna().sum()) if col in df.columns else len(df)))
    for label, counts in [("interval_distribution", df[INTERVAL_COL].value_counts(dropna=False)), ("visit_year_distribution", df["visit_year"].value_counts(dropna=False).sort_index())]:
        for value, count in counts.items():
            rows.append((f"{label}__{value}", int(count)))
    essdai_clean = baseline_df["essdai_total"].dropna()
    if len(essdai_clean) >= 3 and essdai_clean.skew() <= 0:
        rows.append(("warning__essdai_not_right_skewed", 1))
    if len(baseline_df) and baseline_df["esspri_total"].isna().mean() > 0.20:
        rows.append(("warning__esspri_baseline_missing_gt_20pct", 1))
    if domain_summary["n_active"].sum() == 0:
        rows.append(("warning__no_active_domains", 1))
    top_domains = set(domain_summary.head(3)["domain"])
    if not ({"Glandular", "Articular"} & top_domains):
        rows.append(("warning__glandular_or_articular_not_top_domains", 1))
    if esspri_oor > 0 or essdai_oor > 0:
        rows.append(("warning__out_of_range_values_before_cleaning", 1))
    return pd.DataFrame(rows, columns=["metric", "value"])


def write_metric_intermediates(df, baseline_df, input_path):
    """Save row-level files used to calculate ESSDAI/ESSPRI metrics for review."""
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    source_stem = safe_file_stem(input_path)
    metric_columns = [
        ID_COL,
        INTERVAL_COL,
        VISIT_DATE_COL,
        "visit_date_parsed",
        "visit_year",
        *ESSDAI_TOTAL_CANDIDATES,
        *ESSPRI_COMPONENTS,
        "essdai_total",
        "esspri_dryness",
        "esspri_fatigue",
        "esspri_pain",
        "esspri_n_components",
        "esspri_total",
    ]
    for domain in ESSDAI_DOMAIN_VARS:
        metric_columns.extend([f"essdai_domain_{domain}_score", f"essdai_domain_{domain}_active"])
    metric_columns.extend(ESSDAI_DOMAIN_VARS.values())
    metric_columns = list(dict.fromkeys(metric_columns))
    available_columns = [col for col in metric_columns if col in df.columns]

    outputs = []
    for label, data in (
        ("all_visits_metric_inputs", df),
        ("baseline_patient_metric_inputs", baseline_df),
    ):
        audit = data[available_columns].copy()
        audit.insert(0, "source_file", str(input_path))
        path = INTERMEDIATE_DIR / f"01_essdai_esspri_from_{source_stem}__{label}.csv"
        audit.to_csv(path, index=False)
        outputs.append(path)
    return outputs


def write_outputs(baseline_df, activity_summary, domain_summary, by_visit, domain_by_visit, qc_report):
    baseline_df.to_csv(QC_DIR / "01_item1_3_baseline_dataset.csv", index=False)
    activity_summary.to_csv(TABLE_DIR / "01_item1_3_disease_activity_summary.csv", index=False)
    domain_summary.to_csv(TABLE_DIR / "01_item1_3_essdai_domain_baseline.csv", index=False)
    by_visit.to_csv(TABLE_DIR / "01_item1_3_by_visit_summary.csv", index=False)
    domain_by_visit.to_csv(TABLE_DIR / "01_item1_3_domain_by_visit_summary.csv", index=False)
    qc_report.to_csv(QC_DIR / "01_item1_3_qc_report.csv", index=False)
    table_path = TABLE_DIR / "01_table1_overall.csv"
    if table_path.exists():
        overall = pd.read_csv(table_path)
        overall = overall[overall.get("section", pd.Series(dtype=str)) != "Disease activity"]
        overall = pd.concat([overall, activity_summary], ignore_index=True, sort=False)
    else:
        overall = activity_summary.copy()
    overall.to_csv(table_path, index=False)


def main():
    ensure_dirs()
    df = pd.read_parquet(INPUT_PATH)
    if ID_COL not in df.columns:
        raise ValueError(f"Required patient identifier missing: {ID_COL}")
    if INTERVAL_COL not in df.columns:
        df[INTERVAL_COL] = np.nan
    if VISIT_DATE_COL not in df.columns:
        df[VISIT_DATE_COL] = np.nan
    df["visit_year"] = df[VISIT_DATE_COL].map(extract_visit_year)
    df["visit_date_parsed"] = df[VISIT_DATE_COL].map(parse_visit_date_min)
    df, essdai_oor = derive_essdai_total(df)
    df, esspri_oor = derive_esspri_total(df)
    df = derive_domain_activity(df)
    baseline_df, duplicates_before = select_baseline(df)
    activity_summary, stats = summarize_baseline_activity(baseline_df)
    domain_summary = summarize_domains_baseline(baseline_df)
    by_visit, domain_by_visit, interval_order = summarize_by_visit(df)
    qc_report = make_qc_report(df, baseline_df, duplicates_before, essdai_oor, esspri_oor, domain_summary)
    intermediate_paths = write_metric_intermediates(df, baseline_df, INPUT_PATH)
    write_outputs(baseline_df, activity_summary, domain_summary, by_visit, domain_by_visit, qc_report)
    make_baseline_domain_bar(domain_summary)
    make_distribution_plots(df, interval_order, domain_by_visit)
    essdai, esspri = stats["essdai"], stats["esspri"]
    print(f"At baseline, median ESSDAI was {_fmt(essdai['median'])} (IQR {_fmt(essdai['q1'])}–{_fmt(essdai['q3'])}); {_fmt(stats['pct_ge5'])}% of patients had ESSDAI ≥5 (systemic disease activity). Median ESSPRI was {_fmt(esspri['median'])} (IQR {_fmt(esspri['q1'])}–{_fmt(esspri['q3'])}).")
    top3 = domain_summary.head(3)
    domains = ", ".join(f"{row.domain} ({_fmt(row.pct_active)}%)" for row in top3.itertuples(index=False))
    print(f"Most common active ESSDAI domains at baseline were: {domains}.")
    for intermediate_path in intermediate_paths:
        print(f"Saved metric intermediate: {intermediate_path}")


if __name__ == "__main__":
    main()

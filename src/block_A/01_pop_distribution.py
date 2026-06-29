#!/usr/bin/env python3
"""ITEM 1.4 — Baseline by subpopulation Pop1 / Pop2 / Pop3.

Classifies longitudinal visits using ESSDAI/ESSPRI rules, summarizes the
baseline classifiable cohort by Pop1/Pop2/Pop3, and creates a swimmer plot for
longitudinal feasibility review. Outputs are written under the repository's
standard Block A output folders from ``common.py`` when available.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402

PATIENT_ID_COL = "ids__patient_record_number"
VISIT_DATE_COL = "ids__visit_date"
CODEBOOK_COLUMN = "FORM_NAME__QUESTION_NAME"
DEFAULT_INPUT = Path("/data/salazarda/data/eda_sjd/data_analytic/visits_long_collapsed_by_interval_codebook_corrected.parquet")

ESSDAI_TOTAL_CANDIDATES = ["essdai__essdai_total_score"]
ESSPRI_COMPONENTS = [
    "esspri_questionnaire__dryness",
    "esspri_questionnaire__fatigue",
    "esspri_questionnaire__pain",
]
AGE_CANDIDATES = [
    "ids__age_at_diagnosis",
    "ids__age_at_visit",
    "demographics__age_at_diagnosis",
    "age_at_diagnosis",
    "age",
]
SEX_CANDIDATES = ["ids__sex", "ids__gender", "demographics__sex", "demographics__gender", "sex", "gender"]
RACE_CANDIDATES = ["ids__race", "demographics__race", "race", "ethnicity", "ids__ethnicity"]
POP_ORDER = ["Pop1", "Pop2", "Pop3", "Unclassifiable"]
POP_COLORS = {"Pop1": "#d95f02", "Pop2": "#7570b3", "Pop3": "#1b9e77", "Unclassifiable": "#9e9e9e"}
MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "unknown", "unk", "-99"}


def _common_path(name: str, fallback: Path) -> Path:
    return Path(getattr(common, name, fallback))


OUTPUTS_DIR = _common_path("OUTPUTS_DIR", PROJECT_ROOT / "outputs")
TABLES_DIR = _common_path("TABLES_DIR", OUTPUTS_DIR / "tables")
FIGURES_DIR = _common_path("FIGURES_DIR", OUTPUTS_DIR / "figures")
BLOCKA_TABLES_DIR = _common_path("BLOCKA_TABLES_DIR", TABLES_DIR / "blockA")
BLOCKA_FIGURES_DIR = _common_path("BLOCKA_FIGURES_DIR", FIGURES_DIR / "blockA")
BLOCKA_QC_DIR = OUTPUTS_DIR / "qc" / "blockA"
DEFAULT_CODEBOOK = Path(getattr(common, "DEFAULT_CODEBOOK", PROJECT_ROOT / "metadata" / "Consolidated_Codebook_all_columns.xlsx"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ITEM 1.4 Pop1/Pop2/Pop3 baseline and longitudinal outputs.")
    parser.add_argument("--input", type=Path, default=Path(getattr(common, "DEFAULT_POP_DISTRIBUTION_INPUT", DEFAULT_INPUT)))
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    return parser.parse_args()


def is_missing(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_STRINGS


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input analytic file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def load_codebook(path: Path | None = None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def codebook_columns(codebook: pd.DataFrame | None) -> set[str]:
    if codebook is None or CODEBOOK_COLUMN not in codebook.columns:
        return set()
    return set(codebook[CODEBOOK_COLUMN].dropna().astype(str))


def select_first_available(df: pd.DataFrame, candidates: Iterable[str], codebook: pd.DataFrame | None = None) -> str | None:
    cb_cols = codebook_columns(codebook)
    for col in candidates:
        if col in df.columns and (not cb_cols or col in cb_cols):
            return col
    return None


def validate_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    missing_required = [col for col in (PATIENT_ID_COL, VISIT_DATE_COL) if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    essdai_cols = [col for col in ESSDAI_TOTAL_CANDIDATES if col in df.columns]
    if not essdai_cols:
        raise ValueError(f"No ESSDAI total column found. Tried: {ESSDAI_TOTAL_CANDIDATES}")
    missing_esspri = [col for col in ESSPRI_COMPONENTS if col not in df.columns]
    if missing_esspri:
        raise ValueError(f"Cannot calculate ESSPRI total; missing required components: {missing_esspri}")
    return {"essdai": essdai_cols, "esspri": ESSPRI_COMPONENTS.copy()}


def parse_visit_dates(value: object) -> list[pd.Timestamp]:
    if is_missing(value):
        return []
    dates = []
    for fragment in str(value).split("|"):
        parsed = pd.to_datetime(fragment.strip(), errors="coerce")
        if pd.notna(parsed):
            dates.append(pd.Timestamp(parsed).normalize())
    return dates


def numeric_from_first_number(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def coalesce_essdai(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in ESSDAI_TOTAL_CANDIDATES:
        if col in df.columns:
            result = result.combine_first(numeric_from_first_number(df[col]))
    return result


def compute_esspri(df: pd.DataFrame) -> pd.Series:
    comps = [numeric_from_first_number(df[col]) for col in ESSPRI_COMPONENTS]
    comp_df = pd.concat(comps, axis=1)
    comp_df.columns = ["dryness", "fatigue", "pain"]
    return comp_df.mean(axis=1).where(comp_df.notna().all(axis=1), np.nan)


def classify_pop(essdai_total: object, esspri_total: object) -> str:
    essdai_missing = pd.isna(essdai_total)
    esspri_missing = pd.isna(esspri_total)
    if not essdai_missing and float(essdai_total) >= 5:
        return "Pop1"
    if not essdai_missing and float(essdai_total) < 5 and not esspri_missing and float(esspri_total) >= 5:
        return "Pop2"
    if not essdai_missing and float(essdai_total) < 5 and not esspri_missing and float(esspri_total) < 5:
        return "Pop3"
    return "Unclassifiable"


def build_longitudinal_pop_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    work = df.copy()
    work["patient_id"] = work[PATIENT_ID_COL].astype("string")
    work["row_date_original"] = work[VISIT_DATE_COL]
    parsed_lists = work[VISIT_DATE_COL].map(parse_visit_dates)
    work["row_date_min"] = parsed_lists.map(lambda x: min(x) if x else pd.NaT)
    work["row_date_max"] = parsed_lists.map(lambda x: max(x) if x else pd.NaT)
    all_dates = pd.DataFrame({"patient_id": work["patient_id"], "dates": parsed_lists}).explode("dates").dropna()
    baseline_dates = all_dates.groupby("patient_id")["dates"].min().rename("baseline_date")
    work = work.merge(baseline_dates, on="patient_id", how="left")
    work["event_date"] = work["row_date_max"]
    work["essdai_total"] = coalesce_essdai(work)
    work["esspri_total"] = compute_esspri(work)
    work["pop_status"] = [classify_pop(e, p) for e, p in zip(work["essdai_total"], work["esspri_total"])]
    work["time_years"] = (work["event_date"] - work["baseline_date"]).dt.days / 365.25
    negative = int((work["time_years"] < 0).sum())
    warnings: list[str] = []
    if negative:
        warnings.append(f"Excluded {negative} rows with negative time_years.")
    out_of_range_essdai = int(((work["essdai_total"] < 0) | (work["essdai_total"] > 123)).sum())
    out_of_range_esspri = int(((work["esspri_total"] < 0) | (work["esspri_total"] > 10)).sum())
    if out_of_range_essdai:
        raise ValueError(f"ESSDAI values outside plausible range 0-123: {out_of_range_essdai}")
    if out_of_range_esspri:
        raise ValueError(f"ESSPRI values outside range 0-10: {out_of_range_esspri}")
    work = work[~(work["time_years"] < 0)].copy()
    duplicate_dates = int(work.duplicated(["patient_id", "event_date"]).sum())
    qc_counts = {
        "n_rows_missing_visit_date": int(work["event_date"].isna().sum()),
        "n_rows_missing_essdai": int(work["essdai_total"].isna().sum()),
        "n_rows_missing_esspri": int(work["esspri_total"].isna().sum()),
        "n_rows_negative_time": negative,
        "n_duplicate_patient_event_dates": duplicate_dates,
    }
    cols = ["patient_id", "baseline_date", "event_date", "time_years", "essdai_total", "esspri_total", "pop_status", "row_date_original", "row_date_min", "row_date_max"]
    return work, qc_counts, warnings


def build_baseline_dataset(longitudinal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in longitudinal.groupby("patient_id", dropna=True, sort=True):
        baseline_date = g["baseline_date"].iloc[0]
        baseline_rows = g[g["row_date_min"] == baseline_date].copy()
        if baseline_rows.empty:
            baseline_rows = g.sort_values(["row_date_min", "row_date_max"], na_position="last").head(1).copy()
        baseline_rows["_has_essdai"] = baseline_rows["essdai_total"].notna().astype(int)
        baseline_rows["_has_esspri"] = baseline_rows["esspri_total"].notna().astype(int)
        baseline_rows = baseline_rows.sort_values(["_has_essdai", "_has_esspri", "row_date_max"], ascending=[False, False, True], na_position="last")
        rows.append(baseline_rows.iloc[0].drop(labels=["_has_essdai", "_has_esspri"], errors="ignore"))
    return pd.DataFrame(rows).reset_index(drop=True)


def fmt_median_iqr(values: pd.Series) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    return f"{vals.median():.1f} [{vals.quantile(0.25):.1f}, {vals.quantile(0.75):.1f}]"


def normalize_sex(value: object) -> str | float:
    if is_missing(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"f", "female", "woman", "w", "2", "2.0"} or "female" in s:
        return "female"
    if s in {"m", "male", "man", "1", "1.0"} or "male" in s:
        return "male"
    return s


def run_stat_tests(data: pd.DataFrame, variable: str, kind: str, warnings: list[str]) -> tuple[str, str]:
    groups = [data.loc[data["pop_status"] == pop, variable].dropna() for pop in ["Pop1", "Pop2", "Pop3"]]
    try:
        if kind == "continuous":
            usable = [pd.to_numeric(g, errors="coerce").dropna() for g in groups]
            if sum(len(g) > 0 for g in usable) < 2:
                return "", "Kruskal-Wallis"
            return f"{kruskal(*usable, nan_policy='omit').pvalue:.4g}", "Kruskal-Wallis"
        table = pd.crosstab(data["pop_status"], data[variable]).reindex(["Pop1", "Pop2", "Pop3"]).fillna(0)
        if table.shape[0] < 2 or table.shape[1] < 2:
            return "", "Chi-square test"
        _, pvalue, _, expected = chi2_contingency(table)
        if (expected < 5).any():
            warnings.append(f"Chi-square expected cell count <5 for {variable}.")
        return f"{pvalue:.4g}", "Chi-square test"
    except Exception as exc:  # statistical edge cases should not prevent outputs
        warnings.append(f"Could not run {kind} test for {variable}: {exc}")
        return "", "Kruskal-Wallis" if kind == "continuous" else "Chi-square test"


def summarize_table1_by_pop(baseline: pd.DataFrame, codebook: pd.DataFrame | None, warnings: list[str]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    classifiable = baseline[baseline["pop_status"].isin(["Pop1", "Pop2", "Pop3"])].copy()
    n_class = len(classifiable)
    rows = []
    def counts_avail(var: str | None = None) -> dict[str, int]:
        if var is None:
            return {"n_available_overall": n_class, **{f"n_available_{p.lower()}": int((classifiable["pop_status"] == p).sum()) for p in ["Pop1", "Pop2", "Pop3"]}}
        return {"n_available_overall": int(classifiable[var].notna().sum()), **{f"n_available_{p.lower()}": int(classifiable.loc[classifiable["pop_status"] == p, var].notna().sum()) for p in ["Pop1", "Pop2", "Pop3"]}}
    pop_counts = classifiable["pop_status"].value_counts().reindex(["Pop1", "Pop2", "Pop3"], fill_value=0)
    rows.append({"Variable": "N", "Overall": str(n_class), **{p: str(int(pop_counts[p])) for p in ["Pop1", "Pop2", "Pop3"]}, "p_value": "", "test": "", **counts_avail()})
    rows.append({"Variable": "Percent of baseline classifiable cohort", "Overall": "100.0%" if n_class else "NA", **{p: (f"{100*pop_counts[p]/n_class:.1f}%" if n_class else "NA") for p in ["Pop1", "Pop2", "Pop3"]}, "p_value": "", "test": "", **counts_avail()})
    for var, label in [("essdai_total", "ESSDAI total, median [IQR]"), ("esspri_total", "ESSPRI total, median [IQR]")]:
        p, test = run_stat_tests(classifiable, var, "continuous", warnings)
        rows.append({"Variable": label, "Overall": fmt_median_iqr(classifiable[var]), **{pop: fmt_median_iqr(classifiable.loc[classifiable["pop_status"] == pop, var]) for pop in ["Pop1", "Pop2", "Pop3"]}, "p_value": p, "test": test, **counts_avail(var)})
    selected = {
        "selected_age_column": select_first_available(baseline, AGE_CANDIDATES, codebook),
        "selected_sex_column": select_first_available(baseline, SEX_CANDIDATES, codebook),
        "selected_race_column": select_first_available(baseline, RACE_CANDIDATES, codebook),
    }
    if selected["selected_age_column"]:
        baseline["_age"] = pd.to_numeric(baseline[selected["selected_age_column"]], errors="coerce")
        classifiable = baseline[baseline["pop_status"].isin(["Pop1", "Pop2", "Pop3"])].copy()
        p, test = run_stat_tests(classifiable, "_age", "continuous", warnings)
        rows.append({"Variable": "Age, median [IQR]", "Overall": fmt_median_iqr(classifiable["_age"]), **{pop: fmt_median_iqr(classifiable.loc[classifiable["pop_status"] == pop, "_age"]) for pop in ["Pop1", "Pop2", "Pop3"]}, "p_value": p, "test": test, **counts_avail("_age")})
    else:
        warnings.append("Variable not found: age")
    if selected["selected_sex_column"]:
        baseline["_sex"] = baseline[selected["selected_sex_column"]].map(normalize_sex)
        classifiable = baseline[baseline["pop_status"].isin(["Pop1", "Pop2", "Pop3"])].copy()
        p, test = run_stat_tests(classifiable, "_sex", "categorical", warnings)
        def female_fmt(d: pd.DataFrame) -> str:
            denom = int(d["_sex"].notna().sum()); num = int((d["_sex"] == "female").sum())
            return "NA" if denom == 0 else f"{num} ({100*num/denom:.1f}%)"
        rows.append({"Variable": "Sex, n female (%)", "Overall": female_fmt(classifiable), **{pop: female_fmt(classifiable[classifiable["pop_status"] == pop]) for pop in ["Pop1", "Pop2", "Pop3"]}, "p_value": p, "test": test, **counts_avail("_sex")})
    else:
        warnings.append("Variable not found: sex")
    if selected["selected_race_column"]:
        baseline["_race"] = baseline[selected["selected_race_column"]].where(~baseline[selected["selected_race_column"]].map(is_missing), np.nan)
        classifiable = baseline[baseline["pop_status"].isin(["Pop1", "Pop2", "Pop3"])].copy()
        p, test = run_stat_tests(classifiable, "_race", "categorical", warnings)
        for level in sorted(classifiable["_race"].dropna().astype(str).unique()):
            def lvl_fmt(d: pd.DataFrame, lvl: str = level) -> str:
                denom = int(d["_race"].notna().sum()); num = int((d["_race"].astype(str) == lvl).sum())
                return "NA" if denom == 0 else f"{num} ({100*num/denom:.1f}%)"
            rows.append({"Variable": f"Race, {level}, n (%)", "Overall": lvl_fmt(classifiable), **{pop: lvl_fmt(classifiable[classifiable["pop_status"] == pop]) for pop in ["Pop1", "Pop2", "Pop3"]}, "p_value": p, "test": test, **counts_avail("_race")})
            p = test = ""
    else:
        warnings.append("Variable not found: race")
    columns = ["Variable", "Overall", "Pop1", "Pop2", "Pop3", "p_value", "test", "n_available_overall", "n_available_pop1", "n_available_pop2", "n_available_pop3"]
    return pd.DataFrame(rows)[columns], selected


def make_pop_swimmer_plot(longitudinal: pd.DataFrame, baseline: pd.DataFrame, output_path: Path, warnings: list[str]) -> int:
    plot_df = longitudinal.dropna(subset=["patient_id", "time_years"]).copy()
    if plot_df.empty:
        warnings.append("No valid dated longitudinal rows available for swimmer plot.")
        return 0
    max_time = float(plot_df["time_years"].max())
    x_limit = max_time
    outside = 0
    if max_time > 15:
        x_limit = float(plot_df["time_years"].quantile(0.99))
        outside = int((plot_df["time_years"] > x_limit).sum())
        warnings.append(f"Swimmer plot x-axis limited to 99th percentile ({x_limit:.2f} years); {outside} points outside.")
    summary = plot_df.groupby("patient_id").agg(first=("time_years", "min"), last=("time_years", "max"), visits=("time_years", "count")).reset_index()
    base_status = baseline.set_index("patient_id")["pop_status"].to_dict()
    summary["baseline_pop"] = summary["patient_id"].map(base_status).fillna("Unclassifiable")
    summary["pop_rank"] = summary["baseline_pop"].map({p: i for i, p in enumerate(POP_ORDER)}).fillna(99)
    summary = summary.sort_values(["pop_rank", "last", "visits"], ascending=[True, False, False]).reset_index(drop=True)
    grouped_summaries = {pop: summary[summary["baseline_pop"] == pop].copy() for pop in POP_ORDER}
    for pop, group in grouped_summaries.items():
        grouped_summaries[pop]["y"] = np.arange(len(group), 0, -1)
    y_map = pd.concat(grouped_summaries.values(), ignore_index=True).set_index("patient_id")["y"].to_dict()
    plot_df["y"] = plot_df["patient_id"].map(y_map)
    plot_df["baseline_pop"] = plot_df["patient_id"].map(base_status).fillna("Unclassifiable")
    height = min(28, max(10, len(summary) * 0.045 + 5))
    fig, axes = plt.subplots(4, 1, figsize=(13, height), sharex=True, gridspec_kw={"hspace": 0.28})
    x_ticks = [(0, "baseline"), (0.5, "6 mo"), (1, "1y"), (2, "2y"), (4, "4y"), (6, "6y"), (8, "8y"), (10, "10y")]
    for ax, baseline_pop in zip(axes, POP_ORDER):
        group = grouped_summaries[baseline_pop]
        for _, row in group.iterrows():
            ax.hlines(row["y"], row["first"], row["last"], color="#d0d0d0", linewidth=0.8, zorder=1)
        panel_df = plot_df[plot_df["baseline_pop"] == baseline_pop]
        for pop in POP_ORDER:
            sub = panel_df[panel_df["pop_status"] == pop]
            ax.scatter(sub["time_years"], sub["y"], s=12, color=POP_COLORS[pop], label=pop, alpha=0.9, zorder=2)
        for x, label in x_ticks:
            ax.axvline(x, color="#777777", linestyle="--", linewidth=0.7, alpha=0.6)
            if x <= max(x_limit, 10):
                ax.text(x, 1.01, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="#555555")
        ax.set_xlim(left=-0.05, right=max(x_limit, 10) * 1.02)
        ax.set_ylim(0, max(len(group), 1) + 1)
        ax.set_yticks([])
        ax.set_ylabel("Patients")
        ax.set_title(f"Baseline {baseline_pop} (n={len(group)})", loc="left", fontsize=10, color=POP_COLORS[baseline_pop])
        if group.empty:
            ax.text(0.5, 0.5, "No patients", transform=ax.transAxes, ha="center", va="center", color="#777777")
    axes[-1].set_xlabel("Time since baseline (years)")
    fig.suptitle("Longitudinal Pop1/Pop2/Pop3 classification by patient\nSeparate panels by baseline population; points colored by ESSDAI/ESSPRI-defined population", y=0.995)
    fig.text(0.01, 0.01, "Pop1 = ESSDAI ≥5; Pop2 = ESSDAI <5 and ESSPRI ≥5; Pop3 = ESSDAI <5 and ESSPRI <5; grey = insufficient data.", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=4, bbox_to_anchor=(0.98, 0.99))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return outside


def write_outputs(table1: pd.DataFrame, longitudinal: pd.DataFrame, baseline: pd.DataFrame, qc: dict, claim: str) -> None:
    BLOCKA_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    BLOCKA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    BLOCKA_QC_DIR.mkdir(parents=True, exist_ok=True)
    table1.to_csv(BLOCKA_TABLES_DIR / "01_table1_by_pop.csv", index=False)
    longitudinal[["patient_id", "baseline_date", "event_date", "time_years", "essdai_total", "esspri_total", "pop_status", "row_date_original", "row_date_min", "row_date_max"]].to_csv(BLOCKA_TABLES_DIR / "01_pop_longitudinal_status.csv", index=False)
    counts = longitudinal.groupby(["pop_status"]).size().reindex(POP_ORDER, fill_value=0).rename("n_visits").reset_index()
    baseline_counts = baseline["pop_status"].value_counts().reindex(POP_ORDER, fill_value=0).rename_axis("pop_status").reset_index(name="n_baseline_patients")
    counts.merge(baseline_counts, on="pop_status", how="outer").to_csv(BLOCKA_TABLES_DIR / "01_pop_distribution_counts.csv", index=False)
    (BLOCKA_TABLES_DIR / "01_pop_distribution_claim.txt").write_text(claim + "\n", encoding="utf-8")
    with (BLOCKA_QC_DIR / "01_pop_distribution_qc.json").open("w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2, default=str)


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    codebook = load_codebook(args.codebook)
    selected = validate_columns(df)
    longitudinal, row_qc, warnings = build_longitudinal_pop_dataset(df)
    baseline = build_baseline_dataset(longitudinal)
    table1, selected_demo = summarize_table1_by_pop(baseline, codebook, warnings)
    outside = make_pop_swimmer_plot(longitudinal, baseline, BLOCKA_FIGURES_DIR / "02_pop_distribution_plot.pdf", warnings)
    n_total_patients = int(baseline["patient_id"].nunique())
    baseline_counts = baseline["pop_status"].value_counts().reindex(POP_ORDER, fill_value=0)
    n_classifiable = int(baseline_counts[["Pop1", "Pop2", "Pop3"]].sum())
    if int(baseline_counts["Pop1"] + baseline_counts["Pop2"] + baseline_counts["Pop3"]) != n_classifiable:
        raise AssertionError("Pop1 + Pop2 + Pop3 does not equal n_classifiable_baseline")
    if (longitudinal["time_years"] < 0).any():
        raise AssertionError("Negative time_years remains after filtering")
    pct = {pop: (100 * int(baseline_counts[pop]) / n_classifiable if n_classifiable else 0.0) for pop in ["Pop1", "Pop2", "Pop3"]}
    claim = (
        f"At baseline, {pct['Pop1']:.1f}% of classifiable patients were classified as Pop 1 (ESSDAI ≥5), "
        f"{pct['Pop2']:.1f}% as Pop 2 (ESSDAI <5 and ESSPRI ≥5), and "
        f"{pct['Pop3']:.1f}% as Pop 3 (ESSDAI <5 and ESSPRI <5). "
        f"{int(baseline_counts['Unclassifiable'])} patients were unclassifiable at baseline because of missing ESSDAI/ESSPRI components."
    )
    qc = {
        "n_total_rows": int(len(df)),
        "n_total_patients": n_total_patients,
        "n_patients_classifiable_baseline": n_classifiable,
        "n_pop1_baseline": int(baseline_counts["Pop1"]),
        "n_pop2_baseline": int(baseline_counts["Pop2"]),
        "n_pop3_baseline": int(baseline_counts["Pop3"]),
        "n_unclassifiable_baseline": int(baseline_counts["Unclassifiable"]),
        "pct_pop1": pct["Pop1"],
        "pct_pop2": pct["Pop2"],
        "pct_pop3": pct["Pop3"],
        "pct_unclassifiable_baseline": (100 * int(baseline_counts["Unclassifiable"]) / n_total_patients if n_total_patients else 0.0),
        "selected_essdai_columns": selected["essdai"],
        "selected_esspri_component_columns": selected["esspri"],
        **selected_demo,
        **row_qc,
        "n_plot_points_outside_xlim": outside,
        "warnings": warnings,
        "manuscript_claim": claim,
    }
    write_outputs(table1, longitudinal, baseline, qc, claim)
    print(claim)


if __name__ == "__main__":
    main()

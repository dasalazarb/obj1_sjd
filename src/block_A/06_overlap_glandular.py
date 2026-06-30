#!/usr/bin/env python3
"""ITEM 4.1 — Baseline glandular/extraglandular overlap in SjD.

Calculates baseline prevalence of overlap between IC glandular and IC
extraglandular domains using each patient's first available valid visit date.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402

CODEBOOK_COL = "FORM_NAME__QUESTION_NAME"
PATIENT_ID_CANDIDATES = ["ids__patient_record_number", "ids__subject_number"]
VISIT_DATE_COL = "ids__visit_date"
VALUE_CANDIDATES = [
    "Observation Value",
    "observation_value",
    "OBSERVATION_VALUE",
    "value",
    "Value",
    "RESULT_VALUE",
    "result_value",
]
VARIABLE_CANDIDATES = [CODEBOOK_COL, "Variable", "variable", "variable_name", "field_name", "Field Name"]

GLANDULAR_VAR = "visit_summary_-_2016_classification_criteria__ic_glandular_domain"
EXTRAGLANDULAR_VARS = [
    "visit_summary_-_2016_classification_criteria__ic_constitutional_domain",
    "visit_summary_-_2016_classification_criteria__ic_lymphadenopathy_domain",
    "visit_summary_-_2016_classification_criteria__ic_articular_domain",
    "visit_summary_-_2016_classification_criteria__ic_cutaneous_domain",
    "visit_summary_-_2016_classification_criteria__ic_pulmonary_domain",
    "visit_summary_-_2016_classification_criteria__ic_renal_domain",
    "visit_summary_-_2016_classification_criteria__ic_muscular_domain",
    "visit_summary_-_2016_classification_criteria__ic_peripheral_nervous_system_domain",
    "visit_summary_-_2016_classification_criteria__ic_central_nervous_system_domain",
    "visit_summary_-_2016_classification_criteria__ic_hematological_domain",
    "visit_summary_-_2016_classification_criteria__ic_biological_domain",
]
DOMAIN_VARS = [GLANDULAR_VAR, *EXTRAGLANDULAR_VARS]
DOMAIN_LABELS = {
    GLANDULAR_VAR: "glandular",
    **{v: v.replace("visit_summary_-_2016_classification_criteria__ic_", "").replace("_domain", "") for v in EXTRAGLANDULAR_VARS},
}
MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "unknown", "unk", ".", "-99"}
NEGATIVE_STRINGS = {
    "0",
    "no",
    "false",
    "absent",
    "not present",
    "negative",
    "unchecked",
    "unselected",
    "not selected",
    "no activity",
    "none selected",
}
POSITIVE_STRINGS = {"1", "yes", "true", "present", "positive", "checked", "selected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate baseline IC glandular/extraglandular overlap.")
    parser.add_argument("--input", type=Path, default=Path(common.DEFAULT_ANALYTIC_DATASET))
    parser.add_argument("--intermediate-dir", type=Path, default=Path(common.INTERMEDIATE_DATA_DIR))
    parser.add_argument("--tables-dir", type=Path, default=Path(common.BLOCKA_TABLES_DIR))
    parser.add_argument("--figures-dir", type=Path, default=Path(common.OUTPUTS_DIR) / "figures" / "blockA")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def is_missing(value: object) -> bool:
    return pd.isna(value) or str(value).strip().lower() in MISSING_STRINGS


def parse_visit_date_min(value: object) -> pd.Timestamp:
    if is_missing(value):
        return pd.NaT
    dates = [pd.to_datetime(part.strip(), errors="coerce") for part in str(value).split("|")]
    dates = [pd.Timestamp(d).normalize() for d in dates if pd.notna(d)]
    return min(dates) if dates else pd.NaT


def domain_binary(value: object, domain_label: str) -> float:
    if is_missing(value):
        return np.nan
    text = str(value).strip().lower()
    if text in NEGATIVE_STRINGS:
        return 0.0
    if text in POSITIVE_STRINGS or domain_label.lower().replace("_", " ") in text.replace("_", " "):
        return 1.0
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return 1.0 if numeric == 1 else (0.0 if numeric == 0 else np.nan)
    return 1.0  # non-empty IC domain text is treated as selected/present


def choose_existing(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing required {label}; tried {candidates}")


def load_baseline_wide(df: pd.DataFrame, patient_col: str) -> tuple[pd.DataFrame, int, bool]:
    df = df.copy()
    df["visit_date_parsed"] = df[VISIT_DATE_COL].map(parse_visit_date_min)
    all_patients = set(df[patient_col].dropna())
    valid = df[df["visit_date_parsed"].notna()].copy()
    valid_patients = set(valid[patient_col].dropna())
    invalid_patient_count = len(all_patients - valid_patients)
    min_dates = valid.groupby(patient_col)["visit_date_parsed"].transform("min")
    baseline = valid[valid["visit_date_parsed"].eq(min_dates)].copy()
    records = (
        baseline[[patient_col, "visit_date_parsed"]]
        .drop_duplicates()
        .groupby(patient_col, as_index=False)["visit_date_parsed"]
        .min()
    )

    if set(DOMAIN_VARS).issubset(df.columns):
        columns = [patient_col, "visit_date_parsed", *DOMAIN_VARS]
        longish = baseline[columns].copy()
        for var in DOMAIN_VARS:
            bin_col = f"{DOMAIN_LABELS[var]}_binary"
            longish[bin_col] = longish[var].map(lambda x, label=DOMAIN_LABELS[var]: domain_binary(x, label))
        grouped = longish.groupby(patient_col)[[f"{DOMAIN_LABELS[var]}_binary" for var in DOMAIN_VARS]].max(min_count=1)
    else:
        variable_col = choose_existing(df, VARIABLE_CANDIDATES, "variable-name column")
        value_col = choose_existing(df, VALUE_CANDIDATES, "observation-value column")
        needed = [patient_col, "visit_date_parsed", variable_col, value_col]
        longish = baseline.loc[baseline[variable_col].astype(str).isin(DOMAIN_VARS), needed]
        longish = longish.copy()
        longish["domain_label"] = longish[variable_col].map(DOMAIN_LABELS)
        longish["binary_value"] = longish.apply(
            lambda row: domain_binary(row[value_col], row["domain_label"]),
            axis=1,
        )
        grouped = (
            longish.pivot_table(
                index=patient_col,
                columns="domain_label",
                values="binary_value",
                aggfunc=lambda x: x.max(skipna=True) if x.notna().any() else np.nan,
            )
            .rename(columns=lambda label: f"{label}_binary")
        )

    # Keep all rows tied at baseline date but collapse to one patient profile by maximum binary evidence.
    for var in DOMAIN_VARS:
        bin_col = f"{DOMAIN_LABELS[var]}_binary"
        if bin_col not in grouped.columns:
            records[bin_col] = np.nan
        else:
            records[bin_col] = grouped[bin_col].reindex(records[patient_col]).to_numpy()
    source_min_dates = valid.groupby(patient_col)["visit_date_parsed"].min()
    baseline_is_minimum = records.set_index(patient_col)["visit_date_parsed"].eq(source_min_dates).all()
    return records, invalid_patient_count, bool(baseline_is_minimum)


def pct(n: int | float, denom: int | float) -> float:
    return round((float(n) / float(denom) * 100), 1) if denom else 0.0


def main() -> None:
    args = parse_args()
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.input)
    patient_col = choose_existing(df, PATIENT_ID_CANDIDATES, "patient identifier")
    if VISIT_DATE_COL not in df.columns:
        raise ValueError(f"Missing required visit date column: {VISIT_DATE_COL}")
    if not set(DOMAIN_VARS).issubset(df.columns):
        variable_col = choose_existing(df, VARIABLE_CANDIDATES, "variable-name column")
        input_vars = set(df[variable_col].dropna().astype(str))
        missing_input = sorted(set(DOMAIN_VARS) - input_vars)
    else:
        missing_input = sorted(set(DOMAIN_VARS) - set(df.columns))
    if missing_input:
        raise ValueError("Expected IC domain variables absent from input: " + ", ".join(missing_input))

    patient, invalid_date_patients, baseline_is_minimum = load_baseline_wide(df, patient_col)
    gland_col = f"{DOMAIN_LABELS[GLANDULAR_VAR]}_binary"
    extra_cols = [f"{DOMAIN_LABELS[v]}_binary" for v in EXTRAGLANDULAR_VARS]
    all_domain_cols = [gland_col, *extra_cols]
    patient["glandular_baseline"] = patient[gland_col]
    patient["extraglandular_any_baseline"] = patient[extra_cols].max(axis=1, skipna=True)
    patient.loc[patient[extra_cols].notna().sum(axis=1).eq(0), "extraglandular_any_baseline"] = np.nan
    patient["any_domain_evaluable"] = patient[all_domain_cols].notna().any(axis=1)
    evaluable = patient[patient[["glandular_baseline", "extraglandular_any_baseline"]].notna().all(axis=1)].copy()

    conditions = [
        evaluable["glandular_baseline"].eq(1) & evaluable["extraglandular_any_baseline"].eq(1),
        evaluable["glandular_baseline"].eq(1) & evaluable["extraglandular_any_baseline"].eq(0),
        evaluable["glandular_baseline"].eq(0) & evaluable["extraglandular_any_baseline"].eq(1),
        evaluable["glandular_baseline"].eq(0) & evaluable["extraglandular_any_baseline"].eq(0),
    ]
    labels = ["overlap", "glandular_only", "extraglandular_only", "neither"]
    evaluable["overlap_category"] = np.select(conditions, labels, default=pd.NA)
    patient = patient.merge(evaluable[[patient_col, "overlap_category"]], on=patient_col, how="left")

    denom = len(evaluable)
    glandular_eval = evaluable[evaluable["glandular_baseline"].notna()]
    glandular_present = glandular_eval[glandular_eval["glandular_baseline"].eq(1)]
    rows = []
    vars_used = ";".join(DOMAIN_VARS)
    counts = evaluable["overlap_category"].value_counts().to_dict()
    definitions = {
        "overlap": "glandular_baseline == 1 and extraglandular_any_baseline == 1",
        "glandular_only": "glandular_baseline == 1 and extraglandular_any_baseline == 0",
        "extraglandular_only": "glandular_baseline == 0 and extraglandular_any_baseline == 1",
        "neither": "glandular_baseline == 0 and extraglandular_any_baseline == 0",
    }
    for label in labels:
        n = int(counts.get(label, 0))
        rows.append(["baseline_overlap", "category", label, n, denom, pct(n, denom), definitions[label], vars_used])

    for var in DOMAIN_VARS:
        col = f"{DOMAIN_LABELS[var]}_binary"
        dden = int(evaluable[col].notna().sum())
        n = int(evaluable[col].eq(1).sum())
        rows.append(["baseline_domain_prevalence", "domain", DOMAIN_LABELS[var], n, dden, pct(n, dden), "IC domain present at baseline", var])
    cooccur_rows = []
    for var in EXTRAGLANDULAR_VARS:
        col = f"{DOMAIN_LABELS[var]}_binary"
        n = int(glandular_present[col].eq(1).sum())
        dden = int(glandular_present[col].notna().sum())
        row = ["baseline_domain_cooccurrence", "domain_cooccurring_with_glandular", DOMAIN_LABELS[var], n, dden, pct(n, dden), "Extraglandular IC domain present among patients with glandular_baseline == 1", f"{GLANDULAR_VAR};{var}"]
        cooccur_rows.append(row)
        rows.append(row)

    out = pd.DataFrame(rows, columns=["analysis_section", "measure_type", "category_or_domain", "n", "denominator", "pct", "definition", "variables_used"])
    out.to_csv(args.tables_dir / "06_overlap_baseline.csv", index=False)
    patient.to_parquet(args.intermediate_dir / "06_overlap_baseline_patient_level.parquet", index=False)
    pd.DataFrame({"domain_type": ["glandular", *["extraglandular"] * len(EXTRAGLANDULAR_VARS)], "variable": DOMAIN_VARS, "domain_label": [DOMAIN_LABELS[v] for v in DOMAIN_VARS]}).to_csv(args.intermediate_dir / "06_overlap_domain_variables_used.csv", index=False)

    category_sum = int(evaluable["overlap_category"].notna().sum())
    qc = pd.DataFrame([
        ["all_expected_variables_in_input", not missing_input, ""],
        [
            "baseline_is_minimum_valid_visit_date_per_patient",
            baseline_is_minimum,
            "Baseline selected with group minimum of parsed visit dates.",
        ],
        ["categories_mutually_exclusive", bool(evaluable["overlap_category"].notna().all()), "np.select assigns one category per evaluable patient."],
        ["category_sum_equals_evaluable_denominator", category_sum == denom, f"category_sum={category_sum}; denominator={denom}"],
        ["patients_excluded_total_domain_information_missing", True, str(int((~patient["any_domain_evaluable"]).sum()))],
        ["patients_excluded_invalid_or_missing_visit_date", True, str(int(invalid_date_patients))],
        ["glandular_definition_from_ic_glandular_domain", GLANDULAR_VAR in DOMAIN_VARS, GLANDULAR_VAR],
        ["manuscript_sentence_inputs", True, "At baseline, {0}% overlap; top cooccurring domains: {1}".format(pct(counts.get("overlap", 0), denom), "; ".join(f"{r[2]} ({r[5]}%)" for r in sorted(cooccur_rows, key=lambda x: (-x[3], x[2]))[:2]))],
    ], columns=["qc_check", "passed", "details"])
    qc.to_csv(args.intermediate_dir / "06_overlap_baseline_qc.csv", index=False)

    co = pd.DataFrame(0, index=[DOMAIN_LABELS[GLANDULAR_VAR]], columns=[DOMAIN_LABELS[v] for v in EXTRAGLANDULAR_VARS])
    for var in EXTRAGLANDULAR_VARS:
        col = f"{DOMAIN_LABELS[var]}_binary"
        co.loc["glandular", DOMAIN_LABELS[var]] = int(glandular_present[col].eq(1).sum())

    figure_path = args.figures_dir / "06_overlap_baseline_upset_or_heatmap.pdf"
    import matplotlib.pyplot as plt

    try:
        from upsetplot import UpSet, from_indicators

        plot_data = evaluable[[gland_col, *extra_cols]].fillna(0).astype(bool)
        plot_data = plot_data.rename(
            columns={gland_col: "glandular", **{f"{DOMAIN_LABELS[v]}_binary": DOMAIN_LABELS[v] for v in EXTRAGLANDULAR_VARS}}
        )
        fig = plt.figure(figsize=(13, 7))
        UpSet(from_indicators(plot_data.columns, data=plot_data), subset_size="count", show_counts=True).plot(fig=fig)
        fig.suptitle("Baseline glandular/extraglandular IC domain overlap")
        fig.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        fig, ax = plt.subplots(figsize=(13, 3.2))
        im = ax.imshow(co.values, cmap="Blues")
        ax.set_xticks(range(co.shape[1]), co.columns, rotation=45, ha="right")
        ax.set_yticks(range(co.shape[0]), co.index)
        ax.set_title("Baseline co-occurrence with glandular involvement (n)")
        for i in range(co.shape[0]):
            for j in range(co.shape[1]):
                ax.text(j, i, str(co.iloc[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax, label="n")
        fig.tight_layout()
        fig.savefig(figure_path)
        plt.close(fig)


if __name__ == "__main__":
    main()

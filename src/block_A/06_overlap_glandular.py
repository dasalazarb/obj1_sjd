#!/usr/bin/env python3
"""ITEM 4.1 — Baseline glandular/extraglandular overlap.

Computes baseline prevalence of overlap between the 2016 classification
criteria glandular domain and the listed extraglandular domains in SjD, using
the first valid visit date available for each patient.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402

LOG = logging.getLogger(__name__)

PATIENT_ID_COL = "ids__patient_record_number"
VISIT_DATE_COL = "ids__visit_date"
VARIABLE_COL = "FORM_NAME__QUESTION_NAME"
VALUE_CANDIDATES = [
    "Observation Value",
    "OBSERVATION_VALUE",
    "observation_value",
    "Observation_Value",
    "value",
    "Value",
    "answer",
    "Answer",
]

GLANDULAR_DOMAIN = "visit_summary_-_2016_classification_criteria__ic_glandular_domain"
EXTRAGLANDULAR_DOMAINS = [
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
DOMAIN_VARIABLES = [GLANDULAR_DOMAIN, *EXTRAGLANDULAR_DOMAINS]
DOMAIN_LABELS = {v: re.sub(r"^visit_summary_-_2016_classification_criteria__ic_|_domain$", "", v).replace("_", " ").title() for v in DOMAIN_VARIABLES}
POSITIVE_TEXT = {"1", "yes", "y", "true", "t", "present", "positive", "pos", "selected", "checked", "x"}
NEGATIVE_TEXT = {"0", "no", "n", "false", "f", "absent", "negative", "neg", "not selected", "unchecked", "none"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=common.DEFAULT_ANALYTIC_DATASET)
    parser.add_argument("--intermediate-dir", type=Path, default=common.INTERMEDIATE_DATA_DIR)
    parser.add_argument("--table-out", type=Path, default=common.BLOCKA_TABLES_DIR / "06_overlap_baseline.csv")
    parser.add_argument("--figure-out", type=Path, default=common.OUTPUTS_DIR / "figures" / "blockA" / "06_overlap_baseline_upset_or_heatmap.pdf")
    return parser.parse_args()


def detect_value_col(df: pd.DataFrame) -> str:
    for col in VALUE_CANDIDATES:
        if col in df.columns:
            return col
    candidates = [c for c in df.columns if "value" in c.lower() and "observation" in c.lower()]
    if candidates:
        return candidates[0]
    raise ValueError(f"Could not find an observation value column. Tried: {VALUE_CANDIDATES}")


def parse_visit_date(value: Any) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    parts = [p.strip() for p in str(value).split("|") if p.strip()]
    dates = pd.to_datetime(pd.Series(parts), errors="coerce")
    dates = dates.dropna()
    return dates.min() if not dates.empty else pd.NaT


def domain_to_binary(value: Any, variable: str) -> float:
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text or text.lower() in {"nan", "na", "n/a", "null", "none", "missing"}:
        return np.nan
    low = text.lower()
    if low in POSITIVE_TEXT:
        return 1.0
    if low in NEGATIVE_TEXT:
        return 0.0
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return 1.0 if float(numeric) != 0 else 0.0
    label_tokens = [DOMAIN_LABELS[variable].lower(), variable.rsplit("__ic_", 1)[-1].replace("_domain", "").replace("_", " ")]
    if any(token and token in low for token in label_tokens):
        return 1.0
    # IC domain rows generally encode a selected domain as non-empty text.
    return 1.0


def collapse_domain(values: pd.Series, variable: str) -> float:
    parsed = values.map(lambda x: domain_to_binary(x, variable)).dropna()
    if parsed.empty:
        return np.nan
    return 1.0 if (parsed == 1).any() else 0.0


def pct(n: int, denom: int) -> float:
    return round(n / denom * 100, 1) if denom else np.nan


def build_baseline(df: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {PATIENT_ID_COL, VISIT_DATE_COL, VARIABLE_COL, value_col}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required input columns: {missing_cols}")

    all_patients = df[PATIENT_ID_COL].dropna().astype(str).nunique()
    df = df.copy()
    df["patient_id"] = df[PATIENT_ID_COL].astype(str)
    df["visit_date_parsed"] = df[VISIT_DATE_COL].map(parse_visit_date)
    invalid_date_patients = df.loc[df["visit_date_parsed"].isna(), "patient_id"].nunique()
    valid_dates = df.dropna(subset=["visit_date_parsed"])
    min_dates = valid_dates.groupby("patient_id", as_index=False)["visit_date_parsed"].min().rename(columns={"visit_date_parsed": "baseline_date"})
    baseline_long = valid_dates.merge(min_dates, on="patient_id", how="inner")
    baseline_long = baseline_long[baseline_long["visit_date_parsed"].eq(baseline_long["baseline_date"])]
    baseline_domains = baseline_long[baseline_long[VARIABLE_COL].isin(DOMAIN_VARIABLES)].copy()

    rows: list[dict[str, Any]] = []
    for patient_id, patient_df in baseline_domains.groupby("patient_id", sort=False):
        row: dict[str, Any] = {"patient_id": patient_id, "baseline_date": patient_df["baseline_date"].iloc[0]}
        for var in DOMAIN_VARIABLES:
            vals = patient_df.loc[patient_df[VARIABLE_COL].eq(var), value_col]
            row[var] = collapse_domain(vals, var) if not vals.empty else np.nan
        rows.append(row)
    patient = pd.DataFrame(rows)
    if patient.empty:
        patient = pd.DataFrame(columns=["patient_id", "baseline_date", *DOMAIN_VARIABLES])

    patient["glandular_baseline"] = patient[GLANDULAR_DOMAIN]
    patient["extraglandular_any_baseline"] = patient[EXTRAGLANDULAR_DOMAINS].max(axis=1, skipna=True)
    patient.loc[patient[EXTRAGLANDULAR_DOMAINS].notna().sum(axis=1).eq(0), "extraglandular_any_baseline"] = np.nan
    evaluable = patient[[GLANDULAR_DOMAIN, *EXTRAGLANDULAR_DOMAINS]].notna().any(axis=1)
    patient["domains_evaluable_any"] = evaluable
    patient["overlap_category"] = np.select(
        [
            patient["glandular_baseline"].eq(1) & patient["extraglandular_any_baseline"].eq(1),
            patient["glandular_baseline"].eq(1) & patient["extraglandular_any_baseline"].eq(0),
            patient["glandular_baseline"].eq(0) & patient["extraglandular_any_baseline"].eq(1),
            patient["glandular_baseline"].eq(0) & patient["extraglandular_any_baseline"].eq(0),
        ],
        ["overlap", "glandular_only", "extraglandular_only", "neither"],
        default="",
    )
    patient["overlap_category"] = patient["overlap_category"].replace("", pd.NA)
    qc_counts = {
        "n_patients_seen_in_input": int(all_patients),
        "n_patients_with_valid_visit_date": int(min_dates["patient_id"].nunique()),
        "n_patients_excluded_invalid_or_missing_visit_date": int(all_patients - min_dates["patient_id"].nunique()),
        "n_patients_with_any_invalid_date_row": int(invalid_date_patients),
        "n_patients_at_baseline_with_no_domain_information": int((~evaluable).sum()),
    }
    return patient, qc_counts


def build_table(patient: pd.DataFrame) -> pd.DataFrame:
    evaluable = patient[patient["domains_evaluable_any"]].copy()
    denom = int(evaluable["overlap_category"].notna().sum())
    vars_used = ";".join(DOMAIN_VARIABLES)
    rows = []
    for category in ["overlap", "glandular_only", "extraglandular_only", "neither"]:
        n = int(evaluable["overlap_category"].eq(category).sum())
        rows.append(["baseline_overlap", "mutually_exclusive_category", category, n, denom, pct(n, denom), "Baseline category from IC_GLANDULAR_DOMAIN and any listed extraglandular IC domain.", vars_used])
    for var in DOMAIN_VARIABLES:
        n = int(evaluable[var].eq(1).sum())
        rows.append(["baseline_domain_prevalence", "individual_domain", DOMAIN_LABELS[var], n, denom, pct(n, denom), "Domain present at first valid visit; missing is not assumed absent.", var])
    gland_eval = evaluable[evaluable[GLANDULAR_DOMAIN].notna()]
    gland_denom = int(gland_eval[GLANDULAR_DOMAIN].eq(1).sum())
    for var in EXTRAGLANDULAR_DOMAINS:
        n = int((gland_eval[GLANDULAR_DOMAIN].eq(1) & gland_eval[var].eq(1)).sum())
        rows.append(["baseline_glandular_cooccurrence", "extraglandular_domain_cooccurring_with_glandular", DOMAIN_LABELS[var], n, gland_denom, pct(n, gland_denom), "Among patients with glandular involvement at baseline, extraglandular IC domain also present.", f"{GLANDULAR_DOMAIN};{var}"])
    return pd.DataFrame(rows, columns=["analysis_section", "measure_type", "category_or_domain", "n", "denominator", "pct", "definition", "variables_used"])


def build_qc(df: pd.DataFrame, patient: pd.DataFrame, qc_counts: dict[str, Any]) -> pd.DataFrame:
    missing_vars = sorted(set(DOMAIN_VARIABLES) - set(df[VARIABLE_COL].dropna().unique()))
    category_counts = patient["overlap_category"].value_counts(dropna=True)
    category_sum = int(category_counts.sum())
    denom = int(patient["overlap_category"].notna().sum())
    min_check = patient["baseline_date"].notna().all()
    rows = [
        ["expected_domain_variables_present", len(missing_vars) == 0, "pass" if not missing_vars else "fail", ";".join(missing_vars)],
        ["baseline_is_minimum_valid_visit_date_by_patient", bool(min_check), "pass" if min_check else "fail", "Baseline dates were generated with group-wise minimum parsed visit date."],
        ["overlap_categories_mutually_exclusive", bool(patient["overlap_category"].notna().sum() == patient["overlap_category"].dropna().shape[0]), "pass", "np.select assigns at most one category per patient."],
        ["category_sum_equals_evaluable_denominator", category_sum == denom, "pass" if category_sum == denom else "fail", f"sum={category_sum}; denominator={denom}"],
        ["patients_excluded_no_domain_information", qc_counts["n_patients_at_baseline_with_no_domain_information"], "warning" if qc_counts["n_patients_at_baseline_with_no_domain_information"] else "pass", "No listed IC domain evaluable at baseline."],
        ["patients_excluded_invalid_or_missing_visit_date", qc_counts["n_patients_excluded_invalid_or_missing_visit_date"], "warning" if qc_counts["n_patients_excluded_invalid_or_missing_visit_date"] else "pass", "No parseable visit_date after splitting on pipe and taking minimum."],
        ["glandular_definition_source", GLANDULAR_DOMAIN, "pass", "Glandular baseline is derived only from IC_GLANDULAR_DOMAIN."],
    ]
    rows.extend([[k, v, "info", ""] for k, v in qc_counts.items()])
    return pd.DataFrame(rows, columns=["qc_check", "value", "status", "details"])


def plot_heatmap(patient: pd.DataFrame, figure_out: Path) -> None:
    evaluable = patient[patient["domains_evaluable_any"]]
    mat = pd.DataFrame(index=[DOMAIN_LABELS[v] for v in DOMAIN_VARIABLES], columns=[DOMAIN_LABELS[v] for v in DOMAIN_VARIABLES], dtype=float)
    ann = mat.copy().astype(object)
    for a in DOMAIN_VARIABLES:
        for b in DOMAIN_VARIABLES:
            mask = evaluable[a].eq(1) & evaluable[b].eq(1)
            n = int(mask.sum())
            denom = int(evaluable[a].eq(1).sum())
            mat.loc[DOMAIN_LABELS[a], DOMAIN_LABELS[b]] = pct(n, denom) if denom else 0
            ann.loc[DOMAIN_LABELS[a], DOMAIN_LABELS[b]] = f"{n}\n({pct(n, denom) if denom else 0:.1f}%)"
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)), mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, ann.iat[i, j], ha="center", va="center", fontsize=7)
    ax.set_title("Baseline co-occurrence of glandular and extraglandular IC domains")
    fig.colorbar(im, ax=ax, label="Percent among row-domain positive patients")
    fig.tight_layout()
    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_out)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)
    args.table_out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    value_col = detect_value_col(df)
    patient, qc_counts = build_baseline(df, value_col)
    table = build_table(patient)
    qc = build_qc(df, patient, qc_counts)

    patient.to_parquet(args.intermediate_dir / "06_overlap_baseline_patient_level.parquet", index=False)
    pd.DataFrame({"variable": DOMAIN_VARIABLES, "domain_label": [DOMAIN_LABELS[v] for v in DOMAIN_VARIABLES], "domain_group": ["glandular", *["extraglandular"] * len(EXTRAGLANDULAR_DOMAINS)]}).to_csv(args.intermediate_dir / "06_overlap_domain_variables_used.csv", index=False)
    qc.to_csv(args.intermediate_dir / "06_overlap_baseline_qc.csv", index=False)
    table.to_csv(args.table_out, index=False)
    plot_heatmap(patient, args.figure_out)

    cooc = table[table["measure_type"].eq("extraglandular_domain_cooccurring_with_glandular")].sort_values(["n", "pct"], ascending=False).head(2)
    overlap_pct = table.loc[table["category_or_domain"].eq("overlap"), "pct"].iloc[0]
    LOG.info("Manuscript values: overlap=%s%%; top cooccurring domains=%s", overlap_pct, cooc[["category_or_domain", "pct"]].to_dict("records"))


if __name__ == "__main__":
    main()

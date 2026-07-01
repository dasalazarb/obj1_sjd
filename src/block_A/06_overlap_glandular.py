#!/usr/bin/env python3
"""ITEM 4.1A — Baseline prevalence of glandular/extraglandular overlap.

This script estimates baseline-only overlap prevalence. Follow-up prevalent
overlap, incident overlap, and treatment-response analyses are not handled here.

Builds a first-valid-visit patient-level baseline dataset, classifies baseline
patients by glandular/extraglandular overlap with patient-specific source
coalescence, exports manuscript-ready summary values, and writes QC/manifest
intermediate files.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

# Allow execution as `python src/block_A/06_overlap_glandular.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402

LOG = logging.getLogger(__name__)

EXPECTED_INPUT_NAME = "visits_long_collapsed_by_interval_codebook_corrected.parquet"

PATIENT_ID_COL = "ids__patient_record_number"
SUBJECT_ID_COL = "ids__subject_number"
INTERVAL_COL = "ids__interval_name"
VISIT_DATE_COL = "ids__visit_date"
AGE_COL = "ids__age_at_visit"
SEX_COL = "ids__sex"
RACE_COL = "ids__race"
ETHNICITY_COL = "ids__ethnicity"

OUTPUT_TABLE = common.BLOCKA_TABLES_DIR / "06_overlap_baseline.csv"
OUTPUT_FIGURE = common.OUTPUTS_DIR / "figures" / "blockA" / "06_overlap_baseline_upset_or_heatmap.pdf"
PATIENT_LEVEL_OUTPUT = common.INTERMEDIATE_DATA_DIR / "06_overlap_baseline_patient_level.parquet"
MANIFEST_OUTPUT = common.INTERMEDIATE_DATA_DIR / "06_overlap_baseline_variable_manifest.csv"
QC_OUTPUT = common.INTERMEDIATE_DATA_DIR / "06_overlap_baseline_qc.json"

MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "unknown", "unk", "missing", ".", "-99"}
POSITIVE_STRINGS = {"1", "1.0", "yes", "y", "true", "t", "positive", "pos", "present", "checked", "x"}
NEGATIVE_STRINGS = {"0", "0.0", "no", "n", "false", "f", "negative", "neg", "absent", "unchecked"}
SOURCE_VARIABLE_KEY = {
    "preferred": "preferred",
    "essdai_fallback": "fallback",
    "composite_fallback": "fallback_composite",
}

DOMAINS = [
    {
        "domain": "glandular",
        "label": "Glandular",
        "preferred": "visit_summary_-_2016_classification_criteria__ic_glandular_domain",
        "fallback": "essdai__gland_swell",
        "fallback_composite": [
            "visit_summary_-_2016_classification_criteria__ic_symptom_dry_eye_or_dry_mouth",
            "visit_summary_-_2016_classification_criteria__ic_dry_mouth_3month",
            "visit_summary_-_2016_classification_criteria__ic_dry_eye_3month",
            "visit_summary_-_2016_classification_criteria__salivary_gland_movement",
            "visit_summary_-_2016_classification_criteria__lacrimal_dysfunction",
            "essdai__gland_swell",
        ],
        "indicator": "glandular_baseline",
        "is_glandular": True,
    },
    {"domain": "constitutional", "label": "Constitutional", "preferred": "visit_summary_-_2016_classification_criteria__ic_constitutional_domain", "fallback": "essdai__constitutional"},
    {"domain": "lymphadenopathy", "label": "Lymphadenopathy", "preferred": "visit_summary_-_2016_classification_criteria__ic_lymphadenopathy_domain", "fallback": "essdai__hema_lphdenopthy"},
    {"domain": "articular", "label": "Articular", "preferred": "visit_summary_-_2016_classification_criteria__ic_articular_domain", "fallback": "essdai__articular_domain"},
    {"domain": "cutaneous", "label": "Cutaneous", "preferred": "visit_summary_-_2016_classification_criteria__ic_cutaneous_domain", "fallback": "essdai__cutaneous"},
    {"domain": "pulmonary", "label": "Pulmonary", "preferred": "visit_summary_-_2016_classification_criteria__ic_pulmonary_domain", "fallback": "essdai__pulmonary"},
    {"domain": "renal", "label": "Renal", "preferred": "visit_summary_-_2016_classification_criteria__ic_renal_domain", "fallback": "essdai__renal"},
    {"domain": "muscular", "label": "Muscular", "preferred": "visit_summary_-_2016_classification_criteria__ic_muscular_domain", "fallback": "essdai__muscular_domain"},
    {"domain": "peripheral_nervous_system", "label": "Peripheral nervous system", "preferred": "visit_summary_-_2016_classification_criteria__ic_peripheral_nervous_system_domain", "fallback": "essdai__neuro_peripheral"},
    {"domain": "central_nervous_system", "label": "Central nervous system", "preferred": "visit_summary_-_2016_classification_criteria__ic_central_nervous_system_domain", "fallback": "essdai__cns"},
    {"domain": "hematological", "label": "Hematological", "preferred": "visit_summary_-_2016_classification_criteria__ic_hematological_domain", "fallback": "essdai__hematologic"},
    {"domain": "biological", "label": "Biological", "preferred": "visit_summary_-_2016_classification_criteria__ic_biological_domain", "fallback": "essdai__biological_domain"},
]
for d in DOMAINS:
    d.setdefault("indicator", f"extraglandular_{d['domain']}_baseline")
    d.setdefault("is_glandular", False)


def is_missing_value(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_STRINGS


def parse_visit_date_min(value: Any) -> pd.Timestamp:
    """Parse possibly pipe-delimited visit dates and return the earliest date."""
    if is_missing_value(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return pd.to_datetime(value, errors="coerce")
    parsed = []
    for piece in str(value).split("|"):
        dt = pd.to_datetime(piece.strip(), errors="coerce")
        if pd.notna(dt):
            parsed.append(dt.normalize())
    return min(parsed) if parsed else pd.NaT


def read_input() -> pd.DataFrame:
    path = Path(common.DEFAULT_ANALYTIC_DATASET)
    if path.name != EXPECTED_INPUT_NAME:
        raise ValueError(f"Unexpected input filename from common.DEFAULT_ANALYTIC_DATASET: {path}")
    LOG.info("Loading %s", path)
    return pd.read_parquet(path)


def preferred_flag_to_binary(value: Any) -> float:
    if is_missing_value(value):
        return pd.NA
    text = str(value).strip().lower()
    if text in POSITIVE_STRINGS:
        return 1.0
    if text in NEGATIVE_STRINGS:
        return 0.0
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return 1.0 if numeric == 1 else 0.0 if numeric == 0 else pd.NA
    return pd.NA


def essdai_to_binary(value: Any) -> float:
    if is_missing_value(value):
        return pd.NA
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return pd.NA if pd.isna(numeric) else float(numeric > 0)


def any_positive_composite(row: pd.Series, cols: list[str]) -> float:
    """Return 1.0 if any col in cols is positive, 0.0 if all are 0/missing
    but at least one is non-missing, pd.NA if all are missing."""
    values = [preferred_flag_to_binary(row[c]) for c in cols if c in row.index]
    non_missing = [v for v in values if pd.notna(v)]
    if not non_missing:
        return pd.NA
    return 1.0 if any(v == 1.0 for v in non_missing) else 0.0


def aggregate_binary(values: pd.Series, converter) -> Any:
    """Collapse tied baseline rows: any positive > any explicit negative > missing."""
    converted = [converter(value) for value in values]
    non_missing = [value for value in converted if pd.notna(value)]
    if not non_missing:
        return pd.NA
    return 1.0 if any(value == 1.0 for value in non_missing) else 0.0

def aggregate_composite(group: pd.DataFrame, cols: list[str]) -> Any:
    """Collapse composite variables across all tied baseline rows."""
    values: list[Any] = []
    for col in cols:
        if col in group.columns:
            converter = essdai_to_binary if col.startswith("essdai__") else preferred_flag_to_binary
            values.extend(converter(value) for value in group[col])
    non_missing = [value for value in values if pd.notna(value)]
    if not non_missing:
        return pd.NA
    return 1.0 if any(value == 1.0 for value in non_missing) else 0.0


def classify_domain_for_group(group: pd.DataFrame, domain: dict[str, Any]) -> tuple[Any, str]:
    """Return patient-specific baseline domain indicator and source."""
    preferred = domain["preferred"]
    fallback = domain["fallback"]
    if preferred in group.columns:
        preferred_value = aggregate_binary(group[preferred], preferred_flag_to_binary)
        if pd.notna(preferred_value):
            return preferred_value, "preferred"

    if domain["is_glandular"]:
        composite_cols = [c for c in domain.get("fallback_composite", []) if c in group.columns]
        composite_value = aggregate_composite(group, composite_cols)
        if pd.notna(composite_value):
            return composite_value, "composite_fallback"
    elif fallback in group.columns:
        fallback_value = aggregate_binary(group[fallback], essdai_to_binary)
        if pd.notna(fallback_value):
            return fallback_value, "essdai_fallback"

    return pd.NA, "missing"

def aggregate_composite(group: pd.DataFrame, cols: list[str]) -> Any:
    """Collapse composite variables across all tied baseline rows."""
    values: list[Any] = []
    for col in cols:
        if col in group.columns:
            converter = essdai_to_binary if col.startswith("essdai__") else preferred_flag_to_binary
            values.extend(converter(value) for value in group[col])
    non_missing = [value for value in values if pd.notna(value)]
    if not non_missing:
        return pd.NA
    return 1.0 if any(value == 1.0 for value in non_missing) else 0.0


def classify_domain_for_group(group: pd.DataFrame, domain: dict[str, Any]) -> tuple[Any, str]:
    """Return patient-specific baseline domain indicator and source."""
    preferred = domain["preferred"]
    fallback = domain["fallback"]
    if preferred in group.columns:
        preferred_value = aggregate_binary(group[preferred], preferred_flag_to_binary)
        if pd.notna(preferred_value):
            return preferred_value, "preferred"

    if domain["is_glandular"]:
        composite_cols = [c for c in domain.get("fallback_composite", []) if c in group.columns]
        composite_value = aggregate_composite(group, composite_cols)
        if pd.notna(composite_value):
            return composite_value, "composite_fallback"
    elif fallback in group.columns:
        fallback_value = aggregate_binary(group[fallback], essdai_to_binary)
        if pd.notna(fallback_value):
            return fallback_value, "essdai_fallback"

    return pd.NA, "missing"

def first_nonmissing(values: pd.Series) -> Any:
    for value in values:
        if not is_missing_value(value):
            return value
    return pd.NA


def extraglandular_indicator_columns(include_biological: bool = True) -> list[str]:
    return [
        d["indicator"]
        for d in DOMAINS
        if not d["is_glandular"] and (include_biological or d["domain"] != "biological")
    ]


def assign_overlap_category(baseline: pd.DataFrame, extraglandular_col: str, category_col: str) -> None:
    g = baseline["glandular_baseline"]
    e = baseline[extraglandular_col]
    baseline[category_col] = "unclassifiable"
    baseline.loc[g.eq(1) & e.eq(1), category_col] = "overlap"
    baseline.loc[g.eq(1) & e.eq(0), category_col] = "glandular_only"
    baseline.loc[g.eq(0) & e.eq(1), category_col] = "extraglandular_only"
    baseline.loc[g.eq(0) & e.eq(0), category_col] = "neither"


def add_extraglandular_rollup(
    baseline: pd.DataFrame,
    *,
    include_biological: bool,
    suffix: str = "",
) -> None:
    """Add strict and lenient extraglandular rollups for a domain set."""
    ex_cols = extraglandular_indicator_columns(include_biological=include_biological)
    n_available_col = f"n_extraglandular_domains_available_baseline{suffix}"
    pct_missing_col = f"pct_extraglandular_domains_missing_baseline{suffix}"
    all_missing_col = f"all_extraglandular_domains_missing_baseline{suffix}"
    insufficient_col = f"extraglandular_insufficient_data{suffix}"
    lenient_col = f"any_extraglandular_baseline_lenient{suffix}"
    strict_col = f"any_extraglandular_baseline{suffix}"
    strict_category_col = f"overlap_category{suffix}"
    lenient_category_col = f"overlap_category_lenient{suffix}"
    biological_flag_col = f"biological_included_in_extraglandular_definition{suffix}"

    baseline[biological_flag_col] = bool(include_biological)
    baseline[n_available_col] = baseline[ex_cols].notna().sum(axis=1)
    n_ex_domains = len(ex_cols)
    baseline[pct_missing_col] = (n_ex_domains - baseline[n_available_col]) / n_ex_domains
    baseline[all_missing_col] = baseline[n_available_col].eq(0)
    no_positive = baseline[ex_cols].eq(1).sum(axis=1).eq(0)
    baseline[insufficient_col] = no_positive & baseline[pct_missing_col].gt(0.50)

    any_positive = baseline[ex_cols].eq(1).any(axis=1)
    any_available = baseline[ex_cols].notna().any(axis=1)
    baseline[lenient_col] = pd.NA
    baseline.loc[any_positive, lenient_col] = 1.0
    baseline.loc[~any_positive & any_available, lenient_col] = 0.0

    baseline[strict_col] = baseline[lenient_col]
    baseline.loc[baseline[insufficient_col], strict_col] = pd.NA
    baseline.loc[baseline[all_missing_col], strict_col] = pd.NA

    assign_overlap_category(baseline, strict_col, strict_category_col)
    assign_overlap_category(baseline, lenient_col, lenient_category_col)


def build_baseline(df: pd.DataFrame, strict_missingness: bool = True) -> pd.DataFrame:
    if PATIENT_ID_COL not in df.columns:
        raise ValueError(f"Required patient identifier missing: {PATIENT_ID_COL}")
    if VISIT_DATE_COL not in df.columns:
        raise ValueError(f"Required visit date missing: {VISIT_DATE_COL}")

    work = df.copy()
    work["patient_id"] = work[PATIENT_ID_COL].astype("string")
    work["visit_date_min"] = work[VISIT_DATE_COL].map(parse_visit_date_min)
    work["_had_piped_date"] = work[VISIT_DATE_COL].apply(lambda v: isinstance(v, str) and "|" in v)
    work = work[work["patient_id"].notna() & work["visit_date_min"].notna()].copy()

    earliest = work.groupby("patient_id", dropna=True)["visit_date_min"].transform("min")
    candidates = work[work["visit_date_min"].eq(earliest)].copy()

    raw_vars = []
    for d in DOMAINS:
        raw_vars.extend([d["preferred"], d["fallback"]])
        raw_vars.extend(d.get("fallback_composite", []))
    raw_vars = [c for c in dict.fromkeys(raw_vars) if c in candidates.columns]

    keep_cols = [PATIENT_ID_COL, SUBJECT_ID_COL, INTERVAL_COL, "visit_date_min", "_had_piped_date", AGE_COL, SEX_COL, RACE_COL, ETHNICITY_COL]
    keep_cols = [c for c in keep_cols if c in candidates.columns]
    consolidated_rows = []
    for _patient_id, group in candidates.groupby("patient_id", sort=True):
        row: dict[str, Any] = {"patient_id": _patient_id}
        for col in keep_cols:
            row[col] = first_nonmissing(group[col])
        row["n_tied_baseline_rows"] = int(len(group))
        row["_had_piped_date"] = bool(group["_had_piped_date"].any())
        for col in raw_vars:
            row[col] = first_nonmissing(group[col])
        for d in DOMAINS:
            value, source = classify_domain_for_group(group, d)
            row[d["indicator"]] = value
            row[f"{d['domain']}_baseline_source"] = source
        consolidated_rows.append(row)
    baseline = pd.DataFrame(consolidated_rows)

    add_extraglandular_rollup(baseline, include_biological=True)
    add_extraglandular_rollup(baseline, include_biological=False, suffix="_no_biological")
    return baseline

def make_manifest_and_qc(df: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_rows = []
    missingness = {}
    n_baseline = len(baseline)
    for d in DOMAINS:
        composite_vars = d.get("fallback_composite", [])
        raw_for_missingness = [d["preferred"], d["fallback"], *composite_vars]
        for var in raw_for_missingness:
            missingness[var] = None if var not in df.columns else int(df[var].map(is_missing_value).sum())
        source_counts = baseline[f"{d['domain']}_baseline_source"].value_counts(dropna=False).to_dict()
        n_missing = int(source_counts.get("missing", 0))
        manifest_rows.append({
            "domain_name": d["domain"],
            "preferred_variable": d["preferred"],
            "essdai_fallback_variable": d["fallback"],
            "composite_fallback_variables": "|".join(composite_vars) if composite_vars else "",
            "preferred_found": d["preferred"] in df.columns,
            "essdai_fallback_found": d["fallback"] in df.columns,
            "n_patients_used_preferred": int(source_counts.get("preferred", 0)),
            "n_patients_used_composite_fallback": int(source_counts.get("composite_fallback", 0)),
            "n_patients_used_essdai_fallback": int(source_counts.get("essdai_fallback", 0)),
            "n_patients_missing_domain": n_missing,
            "pct_missing_baseline": pct(n_missing, n_baseline),
        })

    classifiable = baseline[baseline["overlap_category"] != "unclassifiable"]
    category_counts = classifiable["overlap_category"].value_counts().to_dict()
    lenient_classifiable = baseline[baseline["overlap_category_lenient"] != "unclassifiable"]
    lenient_counts = lenient_classifiable["overlap_category_lenient"].value_counts().to_dict()
    no_bio_classifiable = baseline[baseline["overlap_category_no_biological"] != "unclassifiable"]
    no_bio_counts = no_bio_classifiable["overlap_category_no_biological"].value_counts().to_dict()
    no_bio_lenient_classifiable = baseline[baseline["overlap_category_lenient_no_biological"] != "unclassifiable"]
    no_bio_lenient_counts = no_bio_lenient_classifiable["overlap_category_lenient_no_biological"].value_counts().to_dict()
    classifiable_pct = len(classifiable) / len(baseline) if len(baseline) else 0
    qc = {
        "n_raw_rows": int(len(df)),
        "n_unique_patients": int(df[PATIENT_ID_COL].nunique(dropna=True)),
        "n_baseline_patients": int(len(baseline)),
        "n_classifiable_patients": int(len(classifiable)),
        "n_unclassifiable_patients": int((baseline["overlap_category"] == "unclassifiable").sum()),
        "pct_classifiable_patients": round(100 * classifiable_pct, 1) if len(baseline) else None,
        "n_patients_all_extraglandular_domains_missing": int(baseline["all_extraglandular_domains_missing_baseline"].sum()),
        "n_patients_gt50pct_extraglandular_domains_missing": int(baseline["pct_extraglandular_domains_missing_baseline"].gt(0.50).sum()),
        "n_patients_extraglandular_insufficient_data": int(baseline["extraglandular_insufficient_data"].sum()),
        "domain_variable_missingness_raw_rows": missingness,
        "source_used_by_patient_and_domain_counts": {
            d["domain"]: {str(k): int(v) for k, v in baseline[f"{d['domain']}_baseline_source"].value_counts(dropna=False).to_dict().items()}
            for d in DOMAINS
        },
        "overlap_categories_sum_to_classifiable_denominator": int(sum(category_counts.values())) == int(len(classifiable)),
        "each_patient_one_baseline_row": bool(baseline[PATIENT_ID_COL].is_unique),
        "category_counts": {k: int(v) for k, v in category_counts.items()},
        "lenient_sensitivity": {
            "biological_included_in_extraglandular_definition": True,
            "n_classifiable_patients": int(len(lenient_classifiable)),
            "n_unclassifiable_patients": int((baseline["overlap_category_lenient"] == "unclassifiable").sum()),
            "category_counts": {k: int(v) for k, v in lenient_counts.items()},
        },
        "exclude_biological_sensitivity": {
            "biological_included_in_extraglandular_definition": False,
            "main_strict_missingness": {
                "n_classifiable_patients": int(len(no_bio_classifiable)),
                "n_unclassifiable_patients": int((baseline["overlap_category_no_biological"] == "unclassifiable").sum()),
                "category_counts": {k: int(v) for k, v in no_bio_counts.items()},
            },
            "lenient_sensitivity": {
                "n_classifiable_patients": int(len(no_bio_lenient_classifiable)),
                "n_unclassifiable_patients": int((baseline["overlap_category_lenient_no_biological"] == "unclassifiable").sum()),
                "category_counts": {k: int(v) for k, v in no_bio_lenient_counts.items()},
            },
        },
        "warnings": [],
    }
    qc["n_piped_visit_dates_resolved"] = int(baseline.get("_had_piped_date", pd.Series(dtype=bool)).sum())
    qc["notes"] = []
    if baseline["overlap_category"].equals(baseline["overlap_category_lenient"]):
        qc["notes"].append(
            "Strict and lenient sensitivity analyses yielded identical overlap classification."
        )
    if classifiable_pct < 0.80:
        qc["warnings"].append(
            "STRONG WARNING: fewer than 80% of baseline patients are classifiable; "
            "overlap prevalence should not be reported as a definitive result."
        )
    return pd.DataFrame(manifest_rows), qc

def pct(n: int | float, denominator: int | float) -> float | Any:
    return round(100 * n / denominator, 1) if denominator else pd.NA


def format_pct_value(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.1f}"


def build_output_table(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    category_labels = ["overlap", "glandular_only", "extraglandular_only", "neither"]
    analysis_specs = [
        ("main_strict_missingness", "overlap_category", True),
        ("lenient_sensitivity", "overlap_category_lenient", True),
        ("main_strict_missingness_no_biological", "overlap_category_no_biological", False),
        ("lenient_sensitivity_no_biological", "overlap_category_lenient_no_biological", False),
    ]

    for analysis_label, category_col, biological_included in analysis_specs:
        classifiable = baseline[baseline[category_col] != "unclassifiable"].copy()
        n_classifiable = len(classifiable)
        n_baseline = len(baseline)
        n_unclassifiable = n_baseline - n_classifiable
        biological_flag = "included" if biological_included else "excluded"
        common_fields = {
            "analysis": analysis_label,
            "biological_included_in_extraglandular_definition": biological_included,
            "extraglandular_definition_note": f"biological domain {biological_flag}",
            "rank": pd.NA,
            "variable_source": "patient_specific_coalesced",
        }
        for measure, n, denominator, denominator_label in [
            ("n_classifiable_baseline", n_classifiable, n_baseline, "baseline_patients"),
            ("pct_classifiable_baseline", n_classifiable, n_baseline, "baseline_patients"),
            ("n_unclassifiable_baseline", n_unclassifiable, n_baseline, "baseline_patients"),
            ("pct_unclassifiable_baseline", n_unclassifiable, n_baseline, "baseline_patients"),
        ]:
            rows.append({
                **common_fields,
                "section": "classification_denominators",
                "measure": measure,
                "domain": "all_domains",
                "n": int(n),
                "denominator": int(denominator),
                "denominator_label": denominator_label,
                "pct": pct(n, denominator),
            })

        for cat in category_labels:
            n = int((classifiable[category_col] == cat).sum())
            rows.append({
                **common_fields,
                "section": "overlap_categories",
                "measure": f"n_{cat}_baseline",
                "domain": cat,
                "n": n,
                "denominator": n_classifiable,
                "denominator_label": "classifiable_patients_primary",
                "pct": pct(n, n_classifiable),
            })
            rows.append({
                **common_fields,
                "section": "overlap_categories_secondary_denominator",
                "measure": f"n_{cat}_baseline_all_baseline",
                "domain": cat,
                "n": n,
                "denominator": n_baseline,
                "denominator_label": "baseline_patients_secondary",
                "pct": pct(n, n_baseline),
            })

        glandular_positive = classifiable[classifiable["glandular_baseline"] == 1]
        g_denom = len(glandular_positive)
        cooccur_rows = []
        for d in DOMAINS:
            if d["is_glandular"] or (d["domain"] == "biological" and not biological_included):
                continue
            col = d["indicator"]
            n_overall = int((classifiable[col] == 1).sum())
            n_glandular = int((glandular_positive[col] == 1).sum())
            source_counts = classifiable[f"{d['domain']}_baseline_source"].value_counts().to_dict()
            variable_source = ";".join(f"{k}:{int(v)}" for k, v in sorted(source_counts.items()))
            rows.append({
                **common_fields,
                "section": "domain_prevalence_overall",
                "measure": "n_pct_domain_positive",
                "domain": d["label"],
                "n": n_overall,
                "denominator": n_classifiable,
                "denominator_label": "classifiable_patients_primary",
                "pct": pct(n_overall, n_classifiable),
                "variable_source": variable_source,
            })
            cooccur_rows.append({
                **common_fields,
                "section": "domain_prevalence_among_glandular_positive",
                "measure": "n_pct_domain_positive_among_glandular",
                "domain": d["label"],
                "n": n_glandular,
                "denominator": g_denom,
                "denominator_label": "glandular_positive_patients",
                "pct": pct(n_glandular, g_denom),
                "variable_source": variable_source,
            })
        cooccur_rows = sorted(cooccur_rows, key=lambda r: (-r["n"], str(r["domain"])))
        for rank, row in enumerate(cooccur_rows, start=1):
            row["rank"] = rank
            rows.append(row)
        for row in cooccur_rows[:2]:
            top = row.copy()
            top["section"] = "top_cooccurring_domains"
            top["denominator_label"] = "glandular_positive_patients"
            top["measure"] = "top_domain_among_glandular_positive"
            rows.append(top)
    return pd.DataFrame(rows)

def _pdf_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_heatmap_figure(baseline: pd.DataFrame) -> None:
    """Write a lightweight one-page PDF bar chart of extraglandular domain prevalence
    among glandular-positive baseline patients.

    NOTE: This is a proxy figure (bar chart) generated without matplotlib/upsetplot.
    The final manuscript figure (UpSet plot or co-occurrence heatmap) should be
    produced with the full plotting environment once available on Biowulf.
    Target: outputs/figures/blockA/06_overlap_baseline_upset_or_heatmap.pdf
    """
    classifiable = baseline[baseline["overlap_category"] != "unclassifiable"].copy()
    glandular_positive = classifiable[classifiable["glandular_baseline"] == 1]
    labels = [d["label"] for d in DOMAINS if not d["is_glandular"]]
    cols = [d["indicator"] for d in DOMAINS if not d["is_glandular"]]
    values = [pct(int((glandular_positive[col] == 1).sum()), len(glandular_positive)) for col in cols]

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    width, height = 612, 792
    margin_left, top = 72, 700
    row_h, bar_x, bar_w_max = 38, 300, 220
    commands = [
        "BT /F1 14 Tf 72 746 Td (Baseline extraglandular co-occurrence with glandular involvement) Tj ET",
        "BT /F1 10 Tf 72 728 Td (% among glandular-positive baseline patients) Tj ET",
    ]
    max_value = max([v for v in values if pd.notna(v)] + [1.0])
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top - i * row_h
        value = 0.0 if pd.isna(value) else float(value)
        bar_w = 0 if max_value == 0 else bar_w_max * value / max_value
        # Light-blue background and darker prevalence bar.
        commands.append(f"0.90 0.95 1.00 rg {bar_x} {y-12} {bar_w_max} 18 re f")
        commands.append(f"0.18 0.45 0.75 rg {bar_x} {y-12} {bar_w:.2f} 18 re f")
        commands.append(f"0 0 0 rg BT /F1 9 Tf {margin_left} {y-7} Td ({_pdf_escape(label)}) Tj ET")
        commands.append(f"0 0 0 rg BT /F1 9 Tf {bar_x + bar_w_max + 12} {y-7} Td ({value:.1f}%) Tj ET")
    content = "\n".join(commands).encode("latin-1", errors="replace")

    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    OUTPUT_FIGURE.write_bytes(pdf)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    common.ensure_output_dirs()
    common.INTERMEDIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)

    df = read_input()
    baseline = build_baseline(df)
    _subj_per_patient = baseline.groupby(PATIENT_ID_COL)[SUBJECT_ID_COL].nunique()
    assert _subj_per_patient.le(1).all(), (
        "patient_record_number maps to multiple subject_numbers — "
        f"review: {_subj_per_patient[_subj_per_patient > 1].index.tolist()}"
    )
    manifest, qc = make_manifest_and_qc(df, baseline)
    for warning_msg in qc["warnings"]:
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)

    output_table = build_output_table(baseline)
    patient_cols = [PATIENT_ID_COL, SUBJECT_ID_COL, "visit_date_min", "n_tied_baseline_rows"]
    for d in DOMAINS:
        patient_cols += [d["preferred"], d["fallback"], *d.get("fallback_composite", [])]
    patient_cols += [d["indicator"] for d in DOMAINS]
    patient_cols += [f"{d['domain']}_baseline_source" for d in DOMAINS]
    patient_cols += [
        "n_extraglandular_domains_available_baseline",
        "pct_extraglandular_domains_missing_baseline",
        "all_extraglandular_domains_missing_baseline",
        "extraglandular_insufficient_data",
        "biological_included_in_extraglandular_definition",
        "any_extraglandular_baseline",
        "any_extraglandular_baseline_lenient",
        "overlap_category",
        "overlap_category_lenient",
        "biological_included_in_extraglandular_definition_no_biological",
        "n_extraglandular_domains_available_baseline_no_biological",
        "pct_extraglandular_domains_missing_baseline_no_biological",
        "all_extraglandular_domains_missing_baseline_no_biological",
        "extraglandular_insufficient_data_no_biological",
        "any_extraglandular_baseline_no_biological",
        "any_extraglandular_baseline_lenient_no_biological",
        "overlap_category_no_biological",
        "overlap_category_lenient_no_biological",
    ]
    patient_cols = [c for c in dict.fromkeys(patient_cols) if c in baseline.columns]

    baseline[patient_cols].to_parquet(PATIENT_LEVEL_OUTPUT, index=False)
    manifest.to_csv(MANIFEST_OUTPUT, index=False)
    output_table.to_csv(OUTPUT_TABLE, index=False)
    with QC_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2, sort_keys=True)
    make_heatmap_figure(baseline)

    classifiable_n = int(qc["n_classifiable_patients"])
    category_counts = qc["category_counts"]
    top2 = output_table[
        (output_table["analysis"] == "main_strict_missingness")
        & (output_table["section"] == "top_cooccurring_domains")
    ].sort_values("rank")
    print("N_baseline_classifiable:", classifiable_n)
    for cat in ["overlap", "glandular_only", "extraglandular_only", "neither"]:
        n = int(category_counts.get(cat, 0))
        print(f"n_{cat}_baseline: {n}; pct_{cat}_baseline: {format_pct_value(pct(n, classifiable_n))}")
    print("Top 2 extraglandular domains co-occurring with glandular involvement:")
    for _, row in top2.iterrows():
        print(f"  {int(row['rank'])}. {row['domain']} ({format_pct_value(row['pct'])}%)")


if __name__ == "__main__":
    main()

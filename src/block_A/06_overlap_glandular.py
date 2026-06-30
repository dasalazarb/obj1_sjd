#!/usr/bin/env python3
"""ITEM 4.1 — Baseline prevalence of glandular/extraglandular overlap.

Builds a first-valid-visit patient-level baseline dataset, classifies baseline
patients by glandular/extraglandular overlap, exports manuscript-ready summary
values, and writes QC/manifest intermediate files.
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
SOURCE_VARIABLE_KEY = {"preferred": "preferred", "essdai_fallback": "fallback"}

DOMAINS = [
    {
        "domain": "glandular",
        "label": "Glandular",
        "preferred": "visit_summary_-_2016_classification_criteria__ic_glandular_domain",
        "fallback": "essdai__gland_swell",
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


def choose_source(df: pd.DataFrame, preferred: str, fallback: str) -> str | None:
    if preferred in df.columns:
        converted = df[preferred].map(preferred_flag_to_binary)
        if converted.notna().any():
            return "preferred"
    if fallback in df.columns:
        converted = df[fallback].map(essdai_to_binary)
        if converted.notna().any():
            return "essdai_fallback"
    return None


def first_nonmissing(values: pd.Series) -> Any:
    for value in values:
        if not is_missing_value(value):
            return value
    return pd.NA


def build_baseline(df: pd.DataFrame, source_by_domain: dict[str, str | None]) -> pd.DataFrame:
    if PATIENT_ID_COL not in df.columns:
        raise ValueError(f"Required patient identifier missing: {PATIENT_ID_COL}")
    if VISIT_DATE_COL not in df.columns:
        raise ValueError(f"Required visit date missing: {VISIT_DATE_COL}")

    work = df.copy()
    work["patient_id"] = work[PATIENT_ID_COL].astype("string")
    work["visit_date_min"] = work[VISIT_DATE_COL].map(parse_visit_date_min)
    work["_had_piped_date"] = work[VISIT_DATE_COL].apply(lambda v: isinstance(v, str) and "|" in v)
    work = work[work["patient_id"].notna() & work["visit_date_min"].notna()].copy()

    selected_raw_vars = [
        d[SOURCE_VARIABLE_KEY[source_by_domain[d["domain"]]]]
        for d in DOMAINS
        if source_by_domain[d["domain"]] in SOURCE_VARIABLE_KEY
    ]
    for d in DOMAINS:
        source = source_by_domain[d["domain"]]
        if source == "preferred":
            work[d["indicator"]] = work[d["preferred"]].map(preferred_flag_to_binary)
        elif source == "essdai_fallback":
            work[d["indicator"]] = work[d["fallback"]].map(essdai_to_binary)
        else:
            work[d["indicator"]] = pd.NA

    earliest = work.groupby("patient_id", dropna=True)["visit_date_min"].transform("min")
    candidates = work[work["visit_date_min"].eq(earliest)].copy()
    candidates["_domain_completeness"] = candidates[[d["indicator"] for d in DOMAINS]].notna().sum(axis=1)
    candidates = candidates.sort_values(["patient_id", "visit_date_min", "_domain_completeness"], ascending=[True, True, False])

    keep_cols = [PATIENT_ID_COL, SUBJECT_ID_COL, INTERVAL_COL, "visit_date_min", "_had_piped_date", AGE_COL, SEX_COL, RACE_COL, ETHNICITY_COL]
    keep_cols = [c for c in keep_cols if c in candidates.columns]
    baseline = candidates.groupby("patient_id", sort=True, as_index=False).first()

    # Recompute identifiers/demographics with first non-missing values among tied earliest visits.
    consolidated_rows = []
    for patient_id, group in candidates.groupby("patient_id", sort=True):
        best = group.iloc[0].copy()
        for col in keep_cols:
            best[col] = first_nonmissing(group[col])
        for col in selected_raw_vars:
            if col in group:
                best[col] = first_nonmissing(group[col])
        consolidated_rows.append(best)
    baseline = pd.DataFrame(consolidated_rows).drop(columns=["_domain_completeness"], errors="ignore")

    ex_cols = [d["indicator"] for d in DOMAINS if not d["is_glandular"]]
    all_domain_cols = [d["indicator"] for d in DOMAINS]
    baseline["all_domains_missing_baseline"] = baseline[all_domain_cols].isna().all(axis=1)
    baseline["any_extraglandular_baseline"] = pd.NA
    has_any_ex_domain_data = baseline[ex_cols].notna().any(axis=1)
    baseline.loc[has_any_ex_domain_data, "any_extraglandular_baseline"] = baseline.loc[has_any_ex_domain_data, ex_cols].fillna(0).max(axis=1)
    g = baseline["glandular_baseline"]
    e = baseline["any_extraglandular_baseline"]
    baseline["overlap_category"] = "unclassifiable"
    baseline.loc[g.eq(1) & e.eq(1), "overlap_category"] = "overlap"
    baseline.loc[g.eq(1) & e.eq(0), "overlap_category"] = "glandular_only"
    baseline.loc[g.eq(0) & e.eq(1), "overlap_category"] = "extraglandular_only"
    baseline.loc[g.eq(0) & e.eq(0), "overlap_category"] = "neither"
    return baseline


def make_manifest_and_qc(df: pd.DataFrame, baseline: pd.DataFrame, source_by_domain: dict[str, str | None]) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_rows = []
    missingness = {}
    for d in DOMAINS:
        preferred_found = d["preferred"] in df.columns
        fallback_found = d["fallback"] in df.columns
        for var in (d["preferred"], d["fallback"]):
            missingness[var] = None if var not in df.columns else int(df[var].map(is_missing_value).sum())
        manifest_rows.append({
            "domain_name": d["domain"],
            "preferred_variable": d["preferred"],
            "essdai_fallback_variable": d["fallback"],
            "preferred_found": preferred_found,
            "essdai_fallback_found": fallback_found,
            "source_used": source_by_domain[d["domain"]] or "none",
        })
    classifiable = baseline[baseline["overlap_category"] != "unclassifiable"]
    category_counts = classifiable["overlap_category"].value_counts().to_dict()
    preferred_fallback_domains = [d["domain"] for d in DOMAINS if source_by_domain[d["domain"]] == "essdai_fallback"]
    qc = {
        "n_raw_rows": int(len(df)),
        "n_unique_patients": int(df[PATIENT_ID_COL].nunique(dropna=True)),
        "n_baseline_patients": int(len(baseline)),
        "n_classifiable_patients": int(len(classifiable)),
        "n_unclassifiable_patients": int((baseline["overlap_category"] == "unclassifiable").sum()),
        "domain_variable_missingness_raw_rows": missingness,
        "overlap_categories_sum_to_classifiable_denominator": int(sum(category_counts.values())) == int(len(classifiable)),
        "each_patient_one_baseline_row": bool(baseline[PATIENT_ID_COL].is_unique),
        "category_counts": {k: int(v) for k, v in category_counts.items()},
        "warnings": [],
    }
    qc["n_piped_visit_dates_resolved"] = int(baseline.get("_had_piped_date", pd.Series(dtype=bool)).sum())
    ex_cols = [d["indicator"] for d in DOMAINS if not d["is_glandular"]]
    n_high_missingness = int(baseline[ex_cols].isna().mean(axis=1).ge(0.5).sum())
    qc["n_patients_extraglandular_derived_ge50pct_domains_missing"] = n_high_missingness
    if n_high_missingness > 0:
        qc["warnings"].append(
            f"{n_high_missingness} patients had any_extraglandular_baseline derived "
            f"with ≥50% of extraglandular domains missing (missing treated as 0)."
        )

    overlap_pct = 100 * category_counts.get("overlap", 0) / len(classifiable) if len(classifiable) else pd.NA
    if not pd.isna(overlap_pct) and (overlap_pct < 1 or overlap_pct > 90):
        qc["warnings"].append(f"Overlap prevalence is unexpectedly low/high: {overlap_pct:.1f}%")
    if preferred_fallback_domains:
        qc["warnings"].append("ESSDAI fallback used for domains with absent/non-informative preferred flags: " + ", ".join(preferred_fallback_domains))
    return pd.DataFrame(manifest_rows), qc


def pct(n: int | float, denominator: int | float) -> float | Any:
    return round(100 * n / denominator, 1) if denominator else pd.NA


def format_pct_value(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.1f}"


def build_output_table(baseline: pd.DataFrame, source_by_domain: dict[str, str | None]) -> pd.DataFrame:
    classifiable = baseline[baseline["overlap_category"] != "unclassifiable"].copy()
    denom = len(classifiable)
    rows = []
    category_labels = ["overlap", "glandular_only", "extraglandular_only", "neither"]
    for cat in category_labels:
        n = int((classifiable["overlap_category"] == cat).sum())
        rows.append({"section": "overlap_categories", "measure": f"n_pct_{cat}_baseline", "domain": cat, "n": n, "denominator": denom, "denominator_label": "classifiable_patients", "pct": pct(n, denom), "rank": pd.NA, "variable_source": "mixed_by_domain"})

    glandular_positive = classifiable[classifiable["glandular_baseline"] == 1]
    g_denom = len(glandular_positive)
    cooccur_rows = []
    for d in DOMAINS:
        if d["is_glandular"]:
            continue
        col = d["indicator"]
        n_overall = int((classifiable[col] == 1).sum())
        n_glandular = int((glandular_positive[col] == 1).sum())
        rows.append({"section": "domain_prevalence_overall", "measure": "n_pct_domain_positive", "domain": d["label"], "n": n_overall, "denominator": denom, "denominator_label": "classifiable_patients", "pct": pct(n_overall, denom), "rank": pd.NA, "variable_source": source_by_domain[d["domain"]] or "none"})
        cooccur_rows.append({"section": "domain_prevalence_among_glandular_positive", "measure": "n_pct_domain_positive_among_glandular", "domain": d["label"], "n": n_glandular, "denominator": g_denom, "denominator_label": "glandular_positive_patients", "pct": pct(n_glandular, g_denom), "rank": pd.NA, "variable_source": source_by_domain[d["domain"]] or "none"})
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
    source_by_domain = {d["domain"]: choose_source(df, d["preferred"], d["fallback"]) for d in DOMAINS}

    # QC: warn if preferred variable has >80% NA at first visits.
    _first_visit_work = df.copy()
    _first_visit_work["visit_date_min"] = _first_visit_work[VISIT_DATE_COL].map(parse_visit_date_min)
    _first_visits = (
        _first_visit_work.sort_values([PATIENT_ID_COL, "visit_date_min"], na_position="last")
        .groupby(PATIENT_ID_COL, sort=False)
        .first()
    )
    for d in DOMAINS:
        if source_by_domain[d["domain"]] == "preferred":
            na_rate = _first_visits[d["preferred"]].map(preferred_flag_to_binary).isna().mean()
            if na_rate > 0.80:
                warnings.warn(
                    f"Domain '{d['domain']}': preferred variable '{d['preferred']}' "
                    f"is {na_rate*100:.1f}% missing at first visits — "
                    f"consider switching to ESSDAI fallback.",
                    RuntimeWarning, stacklevel=2,
                )

    baseline = build_baseline(df, source_by_domain)
    _subj_per_patient = baseline.groupby(PATIENT_ID_COL)[SUBJECT_ID_COL].nunique()
    assert _subj_per_patient.le(1).all(), (
        "patient_record_number maps to multiple subject_numbers — "
        f"review: {_subj_per_patient[_subj_per_patient > 1].index.tolist()}"
    )
    manifest, qc = make_manifest_and_qc(df, baseline, source_by_domain)
    for warning_msg in qc["warnings"]:
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)

    output_table = build_output_table(baseline, source_by_domain)
    patient_cols = [PATIENT_ID_COL, SUBJECT_ID_COL, "visit_date_min"]
    patient_cols += [
        d[SOURCE_VARIABLE_KEY[source_by_domain[d["domain"]]]]
        for d in DOMAINS
        if source_by_domain[d["domain"]] in SOURCE_VARIABLE_KEY
    ]
    patient_cols += [d["indicator"] for d in DOMAINS] + ["any_extraglandular_baseline", "overlap_category"]
    patient_cols = [c for c in dict.fromkeys(patient_cols) if c in baseline.columns]

    baseline[patient_cols].to_parquet(PATIENT_LEVEL_OUTPUT, index=False)
    manifest.to_csv(MANIFEST_OUTPUT, index=False)
    output_table.to_csv(OUTPUT_TABLE, index=False)
    with QC_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2, sort_keys=True)
    make_heatmap_figure(baseline)

    classifiable_n = int(qc["n_classifiable_patients"])
    category_counts = qc["category_counts"]
    top2 = output_table[output_table["section"] == "top_cooccurring_domains"].sort_values("rank")
    print("N_baseline_classifiable:", classifiable_n)
    for cat in ["overlap", "glandular_only", "extraglandular_only", "neither"]:
        n = int(category_counts.get(cat, 0))
        print(f"n_{cat}_baseline: {n}; pct_{cat}_baseline: {format_pct_value(pct(n, classifiable_n))}")
    print("Top 2 extraglandular domains co-occurring with glandular involvement:")
    for _, row in top2.iterrows():
        print(f"  {int(row['rank'])}. {row['domain']} ({format_pct_value(row['pct'])}%)")


if __name__ == "__main__":
    main()

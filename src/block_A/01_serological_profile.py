#!/usr/bin/env python3
"""ITEM 1.2 — Serological profile.

Reads BTRIS 11D/15D lab parquet files, maps exact serology labs, parses results,
creates patient-level serologic indicators, updates Table 1, plots longitudinal
lab values/statuses, and writes QC artifacts.
"""

from __future__ import annotations

import json
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

INPUT_FILES = {
    "11D": Path("/data/salazarda/data/eda_sjd/data_analytic/BTRIS/11D/lab_records.parquet"),
    "15D": Path("/data/salazarda/data/eda_sjd/data_analytic/BTRIS/15D/lab_records.parquet"),
}
PATIENT_ID_CANDIDATES = ["ids__patient_record_number"]
DATE_CANDIDATES = [
    "Observation Date", "observation_date", "Result Date", "result_date",
    "Specimen Date", "specimen_date", "Collection Date", "collection_date",
    "Date", "date",
]
OPTIONAL_COLS = [
    "Unit", "Units", "Unit of Measure", "Reference Range", "Normal Range", "Reference Low", "Reference High",
    "Abnormal Flag", "Flag", "Result Status", "Observation Note", "Comment",
]
LAB_MARKER_MAP = {
    "SS-A/Ro Ab, IgG (Blood)": "ro_ssa_igg",
    "SS-Ro60 Ab, IgG (Blood)": "ro_ssa_igg",
    "SS-Ro52 Ab, IgG (Blood)": "ro_ssa_igg",
    "SS-B/La Ab, IgG (Blood)": "la_ssb_igg",
    "Antinuclear Antibody (ANA) (Blood)": "ana",
    "Antinuclear Antibody (ANA) HEp-2 Substrate (Blood)": "ana",
    "Antinuclear Antibody (ANA) HEp-2 Substrate Titer (Blood)": "ana_titer",
    "Antinuclear Antibody (ANA) HEp-2 Substrate Pattern (Blood)": "ana_pattern",
    "Rheumatoid Factor (Blood)": "rf",
    "Cryoglobulins (Blood)": "cryoglobulins",
    "Complement C4 (Blood)": "c4",
    "WBC (Blood)": "wbc",
}
FINAL_ROWS = [
    ("anti_ro_pos", "SS-A/Ro IgG positive", "Positive if any exact mapped SS-A/Ro IgG, SS-Ro60 IgG, or SS-Ro52 IgG result is interpretable positive using assay-specific cutoffs; these exact labs are not treated as the broader Anti-SS-A screening label."),
    ("anti_la_pos", "SS-B/La IgG positive", "Positive if any exact mapped SS-B/La IgG result is interpretable positive using assay-specific cutoffs; this exact lab is not treated as the broader Anti-SS-B screening label."),
    ("double_ro_la_pos", "SS-A/Ro IgG and SS-B/La IgG double-positive", "Positive among patients interpretable for both exact SS-A/Ro IgG and SS-B/La IgG testing."),
    ("ana_pos", "ANA positive", "Positive by qualitative ANA, numeric ANA above the negative cutoff, or ANA titer >= configured threshold 80; ANA pattern alone does not define positivity."),
    ("rf_pos", "Rheumatoid factor positive", "Positive by qualitative RF, high flag, value above reference high, or numeric value at/above the configured negative cutoff."),
    ("cryo_pos", "Cryoglobulinemia documented", "Positive if cryoglobulin result says positive, detected, or present; negative if it says negative."),
    ("low_c4", "Low C4", "Low by low flag, value below reference low, or value below the configured C4 lower limit when no reference is available."),
    ("leukopenia", "Leukopenia", "Low by WBC low flag/reference range, otherwise configured WBC lower limit."),
]
ANA_POSITIVE_TITER_MIN = 80
C4_DEFAULT_REFERENCE_RANGES = [(15.0, 57.0), (10.0, 40.0), (15.0, 53.0)]
WBC_DEFAULT_REFERENCE_RANGES_X10E9_L = [(3.98, 10.04), (4.23, 9.07)]
DEFAULT_NEGATIVE_UPPER_LIMITS = {
    "SS-A/Ro Ab, IgG (Blood)": 1.0,
    "SS-Ro60 Ab, IgG (Blood)": 20.0,
    "SS-Ro52 Ab, IgG (Blood)": 20.0,
    "SS-B/La Ab, IgG (Blood)": 1.0,
    "Antinuclear Antibody (ANA) (Blood)": 1.0,
}
INCLUSIVE_NEGATIVE_UPPER_LIMIT_LABS = {"Antinuclear Antibody (ANA) (Blood)"}
RF_DEFAULT_NEGATIVE_UPPER_LIMITS = (13.0, 15.0)
TODAY = pd.Timestamp.today().normalize()

POS_RE = re.compile(r"\b(positive|pos|reactive|detected|present|abnormal|high)\b", re.I)
NEG_RE = re.compile(r"\b(negative|neg|non[- ]?reactive|not detected|absent|none detected)\b", re.I)
AMB_RE = re.compile(r"see note|see comment|comment|borderline|equivocal|indeterminate|inconclusive", re.I)
NUM_RE = re.compile(r"^(<=|>=|<|>)?\s*(-?\d+(?:\.\d+)?)")
TITER_RE = re.compile(r"^1\s*:\s*(\d+)")


def _missing(x: Any) -> bool:
    return pd.isna(x) or str(x).strip().lower() in {"", "nan", "none", "na", "n/a"}


def _detect_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    if label == "date":
        for col in df.columns:
            low = col.lower()
            if "date" in low or "datetime" in low:
                return col
    raise ValueError(f"Missing required {label} column. Tried: {', '.join(candidates)}")


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def _existing_date_candidates(df: pd.DataFrame) -> list[str]:
    """Return all available date-like candidate columns, preserving priority order."""
    cols = [col for col in DATE_CANDIDATES if col in df.columns]
    if cols:
        return cols
    return [col for col in df.columns if "date" in col.lower() or "datetime" in col.lower()]


def _coalesced_datetime(df: pd.DataFrame, date_cols: list[str]) -> pd.Series:
    """Parse available date columns, without requiring dates for lab inclusion."""
    if not date_cols:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    parsed = [pd.to_datetime(df[col], errors="coerce") for col in date_cols]
    return pd.concat(parsed, axis=1).bfill(axis=1).iloc[:, 0]


def _ref_low_high(reference_range: Any, low: Any = None, high: Any = None) -> tuple[float | None, float | None]:
    lo = pd.to_numeric(pd.Series([low]), errors="coerce").iloc[0] if not _missing(low) else np.nan
    hi = pd.to_numeric(pd.Series([high]), errors="coerce").iloc[0] if not _missing(high) else np.nan
    if pd.notna(lo) or pd.notna(hi):
        return (None if pd.isna(lo) else float(lo), None if pd.isna(hi) else float(hi))
    if _missing(reference_range):
        return None, None
    text = str(reference_range).replace("–", "-").replace("—", "-")
    if re.search(r"\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?", text):
        return None, None
    range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:-|to|a)\s*(-?\d+(?:\.\d+)?)", text, re.I)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None, None


def _unit_norm(unit: Any) -> str:
    return "" if _missing(unit) else str(unit).strip()


def _wbc_to_x10e9(value: float | None, unit: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    u = _unit_norm(unit).lower().replace(" ", "")
    if u in {"cells/ul", "cells/µl", "cell/ul", "/ul", "/µl"}:
        return float(value) / 1000.0
    return float(value)


def _negative_limit_match(value: float, operator: str, limit: float) -> bool:
    if operator == "<":
        return value <= limit
    if operator == "<=":
        return value <= limit
    return value < limit


def _positive_limit_match(value: float, operator: str, limit: float) -> bool:
    if operator == "<":
        return False
    if operator == "<=":
        return False
    if operator == ">":
        return value >= limit
    if operator == ">=":
        return value >= limit
    return value >= limit


def _range_low_high_from_defaults(ranges: list[tuple[float, float]]) -> tuple[float, float]:
    return min(lo for lo, _ in ranges), max(hi for _, hi in ranges)


def parse_observation_value(value: Any, marker: str, unit: Any = None, reference_range: Any = None, flag: Any = None, reference_low: Any = None, reference_high: Any = None, lab_name: Any = None) -> dict[str, Any]:
    raw = None if pd.isna(value) else value
    clean = "" if raw is None else re.sub(r"\s+", " ", str(raw).strip())
    low_clean = clean.lower()
    out: dict[str, Any] = {
        "raw_value": raw, "clean_value": clean, "numeric_value": np.nan, "operator": "",
        "is_numeric": False, "is_text_free": False, "qualitative_status": "",
        "classification": "unclassified", "classification_reason": "no recognizable result", "plot_value_type": "text",
    }
    titer = TITER_RE.search(clean)
    num = NUM_RE.search(clean.replace(",", ""))
    if titer:
        out.update(numeric_value=float(titer.group(1)), operator="titer", is_numeric=True, plot_value_type="real number")
        if marker == "ana_titer" and ANA_POSITIVE_TITER_MIN is not None and float(titer.group(1)) >= ANA_POSITIVE_TITER_MIN:
            out.update(qualitative_status="positive", classification="positive", classification_reason=f"ANA titer >= configured threshold {ANA_POSITIVE_TITER_MIN}")
        else:
            out.update(classification="unclassified_numeric_titer", classification_reason="numeric titer without applicable positive threshold")
        return out
    if num:
        op = num.group(1) or ""
        out.update(numeric_value=float(num.group(2)), operator=op, is_numeric=True, plot_value_type="Limit value" if op else "real number")
    flag_low = str(flag).strip().lower() in {"l", "low", "lo", "below low normal"}
    flag_high = str(flag).strip().lower() in {"h", "high", "hi", "above high normal", "abnormal", "a"}
    if AMB_RE.search(low_clean) or (len(clean) > 80 and not POS_RE.search(low_clean) and not NEG_RE.search(low_clean)):
        reason = "ambiguous/free-text result requires manual review"
        status = "free text" if "comment" in low_clean or "note" in low_clean else "ambiguous"
        out.update(is_text_free=True, qualitative_status=status, classification="ambiguous", classification_reason=reason, plot_value_type="text")
        return out
    if NEG_RE.search(low_clean):
        out.update(qualitative_status="negative", classification="negative", classification_reason="negative keyword")
        return out
    if POS_RE.search(low_clean):
        out.update(qualitative_status="positive", classification="positive", classification_reason="positive keyword/high/abnormal text")
        return out
    lo, hi = _ref_low_high(reference_range, reference_low, reference_high)
    val = None if pd.isna(out["numeric_value"]) else float(out["numeric_value"])
    lab_key = "" if _missing(lab_name) else str(lab_name).strip()
    negative_limit = DEFAULT_NEGATIVE_UPPER_LIMITS.get(lab_key)
    if val is not None and negative_limit is not None and marker in {"ro_ssa_igg", "la_ssb_igg", "ana"}:
        is_negative = _negative_limit_match(val, out["operator"], negative_limit)
        if lab_key in INCLUSIVE_NEGATIVE_UPPER_LIMIT_LABS and out["operator"] == "":
            is_negative = val <= negative_limit
        if is_negative:
            out.update(qualitative_status="negative", classification="negative", classification_reason=f"numeric value below assay negative cutoff <{negative_limit:g}")
        elif _positive_limit_match(val, out["operator"], negative_limit):
            out.update(qualitative_status="positive", classification="positive", classification_reason=f"numeric value at/above assay negative cutoff {negative_limit:g}")
        return out
    if marker in {"c4", "wbc"}:
        val2 = _wbc_to_x10e9(val, unit) if marker == "wbc" else val
        if flag_low:
            out.update(qualitative_status="low", classification="low", classification_reason="low flag")
        elif val2 is not None and lo is not None and val2 < lo:
            out.update(qualitative_status="low", classification="low", classification_reason="numeric value below reference low")
        else:
            if marker == "c4" and lo is None and hi is None:
                lo, hi = _range_low_high_from_defaults(C4_DEFAULT_REFERENCE_RANGES)
            elif marker == "wbc" and lo is None and hi is None:
                lo, hi = _range_low_high_from_defaults(WBC_DEFAULT_REFERENCE_RANGES_X10E9_L)
            if val2 is not None and lo is not None and val2 < lo:
                out.update(qualitative_status="low", classification="low", classification_reason="numeric value below configured/reference low")
            elif val2 is not None and lo is not None:
                out.update(qualitative_status="normal_or_not_low", classification="negative", classification_reason="numeric value not below low criterion")
    elif marker == "rf":
        if flag_high:
            out.update(qualitative_status="positive", classification="positive", classification_reason="high/abnormal flag")
        elif val is not None and hi is not None and val > hi:
            out.update(qualitative_status="positive", classification="positive", classification_reason="numeric value above reference high")
        elif val is not None and hi is not None:
            out.update(qualitative_status="negative", classification="negative", classification_reason="numeric value not above reference high")
        elif val is not None:
            rf_limit = max(RF_DEFAULT_NEGATIVE_UPPER_LIMITS)
            if _negative_limit_match(val, out["operator"], rf_limit):
                out.update(qualitative_status="negative", classification="negative", classification_reason=f"numeric value below RF negative cutoff <{rf_limit:g}")
            elif _positive_limit_match(val, out["operator"], rf_limit):
                out.update(qualitative_status="positive", classification="positive", classification_reason=f"numeric value at/above RF negative cutoff {rf_limit:g}")
            else:
                out.update(classification="unclassified_numeric_no_ref", classification_reason="numeric RF without reference range or flag")
    elif val is not None:
        out.update(classification="unclassified_numeric", classification_reason="numeric value without marker-specific rule")
    return out


def _result_missing_or_unusable(row: pd.Series) -> bool:
    """Return True when a lab row has no usable parsed result to classify."""
    raw_value = row.get("Observation Value")
    return _missing(raw_value) or str(row.get("classification", "")).startswith("unclassified")


def _fill_nearest_results_same_year(work: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    For rows without a usable result, borrow another result for the same
    patient and serology marker without excluding rows by calendar year/date.

    This keeps the original row/date in the longitudinal table when present.
    Rows with dates prefer the closest dated donor; rows without dates still
    remain eligible and use input order as a deterministic tie-breaker.
    """
    if work.empty:
        return work, {"nearest_same_year_result_filled_n": 0}

    out = work.copy()
    out["_source_row_order"] = np.arange(len(out))
    out["result_filled_from_nearest_same_year"] = False
    out["result_fill_source_lab_date"] = pd.NaT
    out["result_fill_day_delta"] = np.nan
    out["result_fill_source_value"] = pd.NA

    optional_transfer_cols = [
        col
        for col in [
            "Observation Value", "unit", "normal_range", "Unit of Measure",
            "Unit", "Units", "Normal Range", "Reference Range",
            "Abnormal Flag", "Flag", "Reference Low", "Reference High",
        ]
        if col in out.columns
    ]
    unusable = out.apply(_result_missing_or_unusable, axis=1)
    filled = 0

    for idx, row in out[unusable].iterrows():
        if _missing(row.get("patient_id")) or _missing(row.get("serology_marker")):
            continue
        candidates = out[
            (out["patient_id"].astype(str) == str(row["patient_id"]))
            & (out["serology_marker"] == row["serology_marker"])
            & (out.index != idx)
            & (~unusable)
        ].copy()
        if candidates.empty:
            continue
        if pd.notna(row.get("lab_date")) and candidates["lab_date"].notna().any():
            candidates["_abs_day_delta"] = (candidates["lab_date"] - row["lab_date"]).dt.days.abs()
        else:
            candidates["_abs_day_delta"] = np.nan
        donor = candidates.sort_values(["_abs_day_delta", "lab_date", "_source_row_order"], na_position="last").iloc[0]
        out.at[idx, "result_filled_from_nearest_same_year"] = True
        out.at[idx, "result_fill_source_lab_date"] = donor["lab_date"]
        out.at[idx, "result_fill_day_delta"] = int((donor["lab_date"] - row["lab_date"]).days) if pd.notna(donor["lab_date"]) and pd.notna(row.get("lab_date")) else np.nan
        out.at[idx, "result_fill_source_value"] = donor.get("Observation Value")
        for col in optional_transfer_cols:
            out.at[idx, col] = donor.get(col)

        reparsed = parse_observation_value(
            out.at[idx, "Observation Value"],
            out.at[idx, "serology_marker"],
            *(out.at[idx, col] if col in out.columns else None for col in [
                "unit", "normal_range", "Abnormal Flag", "Reference Low", "Reference High", "Cluster Name"
            ])
        )
        for key, value in reparsed.items():
            out.at[idx, key] = value
        unusable.at[idx] = _result_missing_or_unusable(out.loc[idx])
        filled += 1

    out = out.drop(columns=["_source_row_order"])
    qc = {"nearest_same_year_result_filled_n": filled}
    return out, qc

def load_labs() -> tuple[pd.DataFrame, dict[str, Any], str, list[str]]:
    frames = []
    for source, path in INPUT_FILES.items():
        df = pd.read_parquet(path)
        for required in ["Cluster Name", "Observation Value"]:
            if required not in df.columns:
                raise ValueError(f"{path} is missing required column: {required}")
        pid_col = _detect_col(df, PATIENT_ID_CANDIDATES, "patient identifier")
        df = df.copy(); df["source_folder"] = source
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    pid_col = _detect_col(combined, PATIENT_ID_CANDIDATES, "patient identifier")
    date_cols = _existing_date_candidates(combined)
    qc = {"input_rows": int(len(combined)), "exact_duplicate_rows": int(combined.duplicated().sum()), "date_candidate_columns_evaluated": date_cols}
    return combined, qc, pid_col, date_cols


def prepare_long(df: pd.DataFrame, pid_col: str, date_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    terms = re.compile(r"ro|ssa|ss-a|la|ssb|ana|rheumatoid|cryoglobulin|c4|wbc", re.I)
    possible = df.loc[df["Cluster Name"].astype(str).str.contains(terms, na=False) & ~df["Cluster Name"].isin(LAB_MARKER_MAP), ["Cluster Name"]].drop_duplicates().sort_values("Cluster Name")
    possible["n_records"] = possible["Cluster Name"].map(df["Cluster Name"].value_counts())
    work = df[df["Cluster Name"].isin(LAB_MARKER_MAP)].drop_duplicates().copy()
    work["patient_id"] = work[pid_col].astype("string")
    work["lab_date"] = _coalesced_datetime(work, date_cols)
    work["serology_marker"] = work["Cluster Name"].map(LAB_MARKER_MAP)
    unit_col = _first_existing(work, ["Unit of Measure", "Unit", "Units"]); flag_col = _first_existing(work, ["Abnormal Flag", "Flag"])
    rr_col = _first_existing(work, ["Normal Range", "Reference Range"]); rlo_col = _first_existing(work, ["Reference Low"]); rhi_col = _first_existing(work, ["Reference High"])
    parsed = [parse_observation_value(r["Observation Value"], r["serology_marker"], r.get(unit_col) if unit_col else None, r.get(rr_col) if rr_col else None, r.get(flag_col) if flag_col else None, r.get(rlo_col) if rlo_col else None, r.get(rhi_col) if rhi_col else None, r.get("Cluster Name")) for _, r in work.iterrows()]
    work = pd.concat([work.reset_index(drop=True), pd.DataFrame(parsed)], axis=1)
    work["unit"] = work[unit_col] if unit_col else ""
    work["normal_range"] = work[rr_col] if rr_col else ""
    work, nearest_qc = _fill_nearest_results_same_year(work)
    work["needs_manual_review"] = work["classification"].astype(str).str.contains("unclassified|ambiguous", na=False) | work["is_text_free"].fillna(False)
    qc = {"missing_or_future_dates_not_filtered": int(work["lab_date"].isna().sum() + (work["lab_date"] > TODAY).sum())}
    qc.update(nearest_qc)
    return work, possible, qc


def _patient_marker(g: pd.DataFrame, positive_class: str = "positive") -> dict[str, Any]:
    interp = g[g["classification"].isin([positive_class, "positive", "negative", "low"])]
    pos = interp["classification"].isin([positive_class, "positive", "low"]).any()
    first_pos = interp.loc[interp["classification"].isin([positive_class, "positive", "low"]), "lab_date"].min()
    return {"tested": len(g) > 0, "interpretable": len(interp) > 0, "pos": bool(pos) if len(interp) else pd.NA, "first_date": g["lab_date"].min(), "first_positive_date": first_pos, "n_records": int(len(g))}


def build_patient_level(long: pd.DataFrame) -> pd.DataFrame:
    patients = sorted(long["patient_id"].dropna().astype(str).unique())
    rows = []
    for pid in patients:
        row: dict[str, Any] = {"patient_id": pid}
        pg = long[long["patient_id"].astype(str) == pid]
        specs = {"anti_ro": ["ro_ssa_igg"], "anti_la": ["la_ssb_igg"], "ana": ["ana", "ana_titer"], "rf": ["rf"], "cryo": ["cryoglobulins"], "c4": ["c4"], "wbc": ["wbc"]}
        for name, markers in specs.items():
            mg = pg[pg["serology_marker"].isin(markers)]
            d = _patient_marker(mg)
            row[f"{name}_tested"] = d["tested"]; row[f"{name}_interpretable"] = d["interpretable"]
            out_col = {"anti_ro":"anti_ro_pos","anti_la":"anti_la_pos","cryo":"cryo_pos","c4":"low_c4","wbc":"leukopenia"}.get(name, f"{name}_pos")
            row[out_col] = d["pos"]; row[f"{name}_first_date"] = d["first_date"]; row[f"{name}_first_positive_date"] = d["first_positive_date"]; row[f"{name}_n_records"] = d["n_records"]
        ro_pos_labs = pg[(pg["serology_marker"] == "ro_ssa_igg") & (pg["classification"] == "positive")]["Cluster Name"].dropna().unique()
        row["anti_ro_positive_source_lab"] = "; ".join(map(str, ro_pos_labs))
        row["double_ro_la_pos"] = bool(row["anti_ro_pos"] is True and row["anti_la_pos"] is True) if row["anti_ro_interpretable"] and row["anti_la_interpretable"] else pd.NA
        at = pg[pg["serology_marker"] == "ana_titer"]["numeric_value"].dropna(); row["ana_titer_max"] = at.max() if not at.empty else np.nan
        row["ana_pattern_values"] = "; ".join(pg.loc[pg["serology_marker"] == "ana_pattern", "clean_value"].dropna().astype(str).unique())
        row["rf_max_value"] = pg.loc[pg["serology_marker"] == "rf", "numeric_value"].max()
        row["c4_min_value"] = pg.loc[pg["serology_marker"] == "c4", "numeric_value"].min(); row["c4_units"] = "; ".join(pg.loc[pg["serology_marker"] == "c4", "unit"].dropna().astype(str).unique())
        row["wbc_min_value"] = pg.loc[pg["serology_marker"] == "wbc", "numeric_value"].min(); row["wbc_units"] = "; ".join(pg.loc[pg["serology_marker"] == "wbc", "unit"].dropna().astype(str).unique())
        rows.append(row)
    return pd.DataFrame(rows)


def add_table_block(patient: pd.DataFrame, qc_warnings: list[str]) -> pd.DataFrame:
    n_total = len(patient); rows = []
    for col, label, definition in FINAL_ROWS:
        if col == "double_ro_la_pos":
            denom_mask = patient["anti_ro_interpretable"].fillna(False) & patient["anti_la_interpretable"].fillna(False)
        else:
            prefix = {"anti_ro_pos":"anti_ro", "anti_la_pos":"anti_la", "ana_pos":"ana", "rf_pos":"rf", "cryo_pos":"cryo", "low_c4":"c4", "leukopenia":"wbc"}[col]
            denom_mask = patient[f"{prefix}_interpretable"].fillna(False)
        tested_col = {"anti_ro_pos":"anti_ro_tested", "anti_la_pos":"anti_la_tested", "ana_pos":"ana_tested", "rf_pos":"rf_tested", "cryo_pos":"cryo_tested", "low_c4":"c4_tested", "leukopenia":"wbc_tested"}.get(col)
        n_tested = int(patient[tested_col].fillna(False).sum()) if tested_col else int((patient["anti_ro_tested"].fillna(False)&patient["anti_la_tested"].fillna(False)).sum())
        denom = int(denom_mask.sum()); n = int((patient.loc[denom_mask, col] == True).sum())
        pct = None if denom == 0 else round(n / denom * 100, 1)
        rows.append({"section":"Serologic characteristics", "variable":label, "n":n, "denominator":denom, "percent":pct, "formatted":f"{n}/{denom} ({pct:.1f}%)" if pct is not None else f"{n}/{denom} (NA)", "n_total_cohort":n_total, "n_tested":n_tested, "n_interpretable":denom, "missing_n":n_total-n_tested, "unclassified_n":max(n_tested-denom,0), "denominator_type":"n_interpretable", "definition":definition, "qc_flag":"; ".join(qc_warnings)})
    table_path = common.BLOCKA_TABLES_DIR / "01_table1_overall.csv"
    new = pd.DataFrame(rows)
    if table_path.exists():
        old = pd.read_csv(table_path)
        old = old[old["section"] != "Serologic characteristics"] if "section" in old.columns else old
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new.to_csv(table_path, index=False)
    return pd.DataFrame(rows)


def write_qc(long: pd.DataFrame, patient: pd.DataFrame, possible: pd.DataFrame, summary: dict[str, Any], warnings: list[str]) -> None:
    qc_dir = common.OUTPUTS_DIR / "qc" / "blockA"; qc_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(qc_dir / "01_serology_long_clean.csv", index=False)
    patient.to_csv(qc_dir / "01_serology_patient_level.csv", index=False)
    possible.to_csv(qc_dir / "01_serology_possible_unmapped_cluster_names.csv", index=False)
    keys = ["serology_marker","Cluster Name","Observation Value","clean_value","unit","normal_range","numeric_value","operator","qualitative_status","classification","classification_reason","is_text_free","needs_manual_review"]
    unique = long.groupby(keys, dropna=False).agg(n_records=("patient_id","size"), n_patients=("patient_id","nunique"), example_patient_id=("patient_id","first"), first_date=("lab_date","min"), last_date=("lab_date","max")).reset_index()
    unique.to_csv(qc_dir / "01_serology_unique_values.csv", index=False)
    long[long["needs_manual_review"]].to_csv(qc_dir / "01_serology_unclassified_values.csv", index=False)
    summary["warnings"] = warnings
    (qc_dir / "01_serology_summary_qc.json").write_text(json.dumps(summary, indent=2, default=str))


def _metadata_label(g: pd.DataFrame) -> str:
    """Return compact unit/range metadata for plot titles."""
    unit_values = [v for v in g.get("unit", pd.Series(dtype=object)).dropna().astype(str).str.strip().unique() if v]
    range_values = [v for v in g.get("normal_range", pd.Series(dtype=object)).dropna().astype(str).str.strip().unique() if v]
    parts = []
    if unit_values:
        parts.append("Unit of Measure: " + "; ".join(unit_values[:3]) + ("; ..." if len(unit_values) > 3 else ""))
    if range_values:
        parts.append("Normal Range: " + "; ".join(range_values[:3]) + ("; ..." if len(range_values) > 3 else ""))
    return " | ".join(parts)


def _add_reference_lines(ax: plt.Axes, g: pd.DataFrame) -> None:
    """Add horizontal normal-range dividers when one stable range is available."""
    ranges = [v for v in g.get("normal_range", pd.Series(dtype=object)).dropna().astype(str).unique() if str(v).strip()]
    lows_highs = [_ref_low_high(r) for r in ranges]
    lows = sorted({lo for lo, _ in lows_highs if lo is not None})
    highs = sorted({hi for _, hi in lows_highs if hi is not None})
    for lo in lows[:3]:
        ax.axhline(lo, color="tab:green", linestyle="--", linewidth=.8, alpha=.6, label="_nolegend_")
    for hi in highs[:3]:
        ax.axhline(hi, color="tab:red", linestyle="--", linewidth=.8, alpha=.6, label="_nolegend_")


def _continuous_markers(long: pd.DataFrame, threshold: float = 0.5) -> set[str]:
    """Markers with a majority of numeric records should be plotted as continuous."""
    valid = long[long["serology_marker"].notna()].copy()
    if valid.empty:
        return set()
    numeric_share = valid.groupby("serology_marker")["is_numeric"].mean(numeric_only=False)
    return set(numeric_share[numeric_share > threshold].index.astype(str))


def _limit_label(operator: str, value: float, clean_value: Any) -> str:
    """Use the reported limit as the categorical-axis label."""
    if operator:
        return f"{operator}{value:g}"
    return str(clean_value).strip() or "Other"


def _continuous_timeline_point(row: pd.Series) -> tuple[str, float | None, str | None]:
    """Classify one result for a timeline with numeric and categorical y regions.

    ``continuous`` points retain their measurement on the numeric scale.  The
    remaining point types are assigned categorical labels later, so limits and
    text never distort the numeric range.
    """
    value = row["numeric_value"]
    value = None if pd.isna(value) else float(value)
    operator = str(row["operator"] or "")
    lab_name = str(row.get("Cluster Name", "")).strip()

    if lab_name == "SS-A/Ro Ab, IgG (Blood)":
        if value is None:
            return "category", None, "Other"
        # This assay reports three ordinal upper ranges; only values below 8
        # remain on the continuous portion of the axis.
        if operator or value >= 8:
            if value >= 1300:
                return "category", None, ">1300"
            if value >= 240:
                return "category", None, ">240"
            return "category", None, ">8.0"
        return "continuous", value, None

    if lab_name == "SS-B/La Ab, IgG (Blood)":
        if value is None:
            return "category", None, "Other"
        if operator in {"<", "<="}:
            return "category", None, "<0.2"
        if operator in {">", ">="}:
            return "category", None, ">8.0"
        return "continuous", value, None

    if value is not None and not operator:
        return "continuous", value, None
    if value is not None and operator:
        return "category", None, _limit_label(operator, value, row["clean_value"])
    # Free text is represented by a marker without exposing potentially long
    # clinical notes in the figure.
    return "category", None, "Text"


def _plot_continuous_timeline(ax: plt.Axes, g: pd.DataFrame) -> None:
    """Plot continuous values plus limit/text categories on one y-axis."""
    points = g.apply(_continuous_timeline_point, axis=1, result_type="expand")
    points.columns = ["timeline_type", "timeline_value", "timeline_category"]
    g = pd.concat([g, points], axis=1)
    continuous = g[g["timeline_type"] == "continuous"]
    categorical = g[g["timeline_type"] == "category"]

    if not continuous.empty:
        ax.scatter(continuous.lab_date, continuous.timeline_value, marker="o", alpha=.7, label="Real number")
        lower = float(continuous.timeline_value.min())
        upper = float(continuous.timeline_value.max())
        padding = max((upper - lower) * .08, 0.2)
        category_start = lower - padding
        category_step = max(padding, 0.2)
    else:
        category_start, category_step = 0.0, 1.0

    # Text and lower limits belong below the numeric range, while upper limits
    # are placed above it so the position conveys the direction of the limit.
    categories = list(dict.fromkeys(categorical.timeline_category.dropna()))
    lower_categories = [c for c in categories if c in {"Text", "Other"} or str(c).startswith("<")]
    upper_categories = [c for c in categories if c not in lower_categories]
    y_map = {
        category: category_start - (len(lower_categories) - idx - 1) * category_step
        for idx, category in enumerate(lower_categories)
    }
    upper_start = (upper + padding) if not continuous.empty else len(lower_categories) * category_step
    y_map.update({category: upper_start + idx * category_step for idx, category in enumerate(upper_categories)})
    ordered_categories = lower_categories + upper_categories
    if not categorical.empty:
        for category, s in categorical.groupby("timeline_category", sort=False):
            ax.scatter(s.lab_date, [y_map[category]] * len(s), marker="x", alpha=.8, label=str(category))

    _add_reference_lines(ax, g)
    if ordered_categories:
        numeric_ticks = list(ax.get_yticks()) if not continuous.empty else []
        ax.set_yticks(list(y_map.values()) + numeric_ticks)
        ax.set_yticklabels(ordered_categories + [f"{tick:g}" for tick in numeric_ticks])
    ax.set_ylabel("Value / reported limit")
    ax.legend(loc="best", fontsize=7)


def make_plots(long: pd.DataFrame) -> None:
    out = common.OUTPUTS_DIR / "figures" / "blockA"; out.mkdir(parents=True, exist_ok=True)
    marker_titles = {"ro_ssa_igg":"SS-A/Ro IgG", "la_ssb_igg":"SS-B/La IgG", "ana":"ANA", "ana_titer":"ANA titer", "ana_pattern":"ANA pattern", "rf":"RF", "c4":"C4", "wbc":"WBC", "cryoglobulins":"Cryoglobulins"}
    continuous_markers = _continuous_markers(long)
    default_continuous = ["ro_ssa_igg", "la_ssb_igg", "ana_titer", "rf", "c4", "wbc"]
    panel_markers = [m for m in default_continuous if m in set(long.serology_marker.dropna())]
    panel_markers.extend(sorted(continuous_markers - set(panel_markers)))
    panels = {marker: marker_titles.get(marker, marker.replace("_", " ").title()) for marker in panel_markers}
    if panels:
        fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3 * len(panels)), sharex=False)
        if len(panels) == 1: axes = [axes]
        for ax, (marker, title) in zip(axes, panels.items()):
            g = long[(long.serology_marker == marker) & long.lab_date.notna()].copy()
            _plot_continuous_timeline(ax, g)
            metadata = _metadata_label(g)
            ax.set_title(title + (f" ({metadata})" if metadata else "")); ax.set_xlabel("Time")
        fig.tight_layout(); fig.savefig(out / "01_serological_continuous_timeline.pdf"); plt.close(fig)

    categorical_markers = ["ro_ssa_igg", "la_ssb_igg", "ana", "ana_pattern", "rf", "cryoglobulins"]
    categorical_markers = [m for m in categorical_markers if m not in continuous_markers]
    cat = long[long.serology_marker.isin(categorical_markers)].copy()
    cat = cat[cat.lab_date.notna()].copy()
    cat["observed_category"] = cat["clean_value"].where(~cat["clean_value"].astype(str).str.strip().eq(""), cat["classification"])
    cat["observed_category"] = cat["observed_category"].astype(str).str.slice(0, 60)
    cat_agg = cat.groupby(["serology_marker", "lab_date", "observed_category", "classification"], dropna=False).size().reset_index(name="n")
    cat_panels = list(cat_agg["serology_marker"].dropna().unique())
    if cat_panels:
        fig, axes = plt.subplots(len(cat_panels), 1, figsize=(12, max(3, 2.8*len(cat_panels))), sharex=False)
        if len(cat_panels) == 1: axes = [axes]
        for ax, marker in zip(axes, cat_panels):
            g = cat_agg[cat_agg.serology_marker == marker].copy()
            categories = sorted(g["observed_category"].dropna().unique())
            y_map = {cat_value: idx for idx, cat_value in enumerate(categories)}
            for cls, s in g.groupby("classification", dropna=False):
                ax.scatter(s.lab_date, s["observed_category"].map(y_map), s=(s.n.clip(1, 50) * 12), label=str(cls), alpha=.7)
            for y in np.arange(.5, len(categories), 1):
                ax.axhline(y, color="0.9", linewidth=.6, zorder=0)
            source = cat[cat.serology_marker == marker]
            title = marker_titles.get(marker, marker.replace("_", " ").title())
            metadata = _metadata_label(source)
            ax.set_title(title + (f" ({metadata})" if metadata else ""))
            ax.set_yticks(range(len(categories))); ax.set_yticklabels(categories, fontsize=7)
            ax.set_xlabel("Time"); ax.set_ylabel("Observed category")
            ax.legend(fontsize=7, loc="best")
        fig.tight_layout(); fig.savefig(out / "01_serology_categorical_timeline.pdf"); plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    common.ensure_output_dirs(); (common.OUTPUTS_DIR / "qc" / "blockA").mkdir(parents=True, exist_ok=True)
    df, summary, pid_col, date_col = load_labs()
    long, possible, q2 = prepare_long(df, pid_col, date_col); summary.update(q2)
    patient = build_patient_level(long)
    warnings: list[str] = []
    def pct(col: str, mask: pd.Series) -> float: return float((patient.loc[mask, col] == True).mean()*100) if mask.any() else np.nan
    ro = pct("anti_ro_pos", patient.anti_ro_interpretable.fillna(False)); la = pct("anti_la_pos", patient.anti_la_interpretable.fillna(False)); dbl = pct("double_ro_la_pos", patient.anti_ro_interpretable.fillna(False)&patient.anti_la_interpretable.fillna(False)); cryo = pct("cryo_pos", patient.cryo_interpretable.fillna(False))
    if pd.notna(ro) and pd.notna(la) and ro < la: warnings.append("pct_anti_ro_pos < pct_anti_la_pos")
    if pd.notna(dbl) and pd.notna(ro) and dbl > ro: warnings.append("pct_double_pos > pct_anti_ro_pos")
    if pd.notna(dbl) and pd.notna(la) and dbl > la: warnings.append("pct_double_pos > pct_anti_la_pos")
    for name in ["anti_ro","anti_la","ana","rf","cryo","c4","wbc"]:
        interp = int(patient[f"{name}_interpretable"].fillna(False).sum()); tested = int(patient[f"{name}_tested"].fillna(False).sum())
        if len(patient) and (len(patient)-tested)/len(patient) > .5: warnings.append(f"missingness > 50% for {name}")
        if interp < 20: warnings.append(f"interpretable denominator < 20 for {name}")
    if long["is_text_free"].any(): warnings.append("see note/comment/free-text values present")
    for m in ["c4","rf","wbc"]:
        if long.loc[long.serology_marker == m, "unit"].dropna().astype(str).nunique() > 1: warnings.append(f"multiple units for {m}")
    if summary.get("missing_or_future_dates_not_filtered", 0): warnings.append("missing or future dates present (not filtered)")
    if summary.get("exact_duplicate_rows", 0): warnings.append("exact duplicate rows between/within sources present")
    if not possible.empty: warnings.append("similar but unmapped Cluster Name values present")
    add_table_block(patient, warnings); write_qc(long, patient, possible, summary, warnings); make_plots(long)
    print(f"SS-A/Ro IgG positivity was present in {ro:.1f}% of patients with interpretable testing; {dbl:.1f}% were double-positive for SS-A/Ro IgG and SS-B/La IgG. Cryoglobulinemia was documented in {cryo:.1f}% of patients with interpretable cryoglobulin testing.")
    print(f"La positividad para SS-A/Ro IgG estuvo presente en {ro:.1f}% de los pacientes con prueba interpretable; {dbl:.1f}% fueron doble positivos para SS-A/Ro IgG y SS-B/La IgG. La crioglobulinemia fue documentada en {cryo:.1f}% de los pacientes con prueba interpretable.")


if __name__ == "__main__":
    main()

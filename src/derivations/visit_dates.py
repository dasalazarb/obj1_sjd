"""Canonical patient-visit date derivations.

The official clinical date for a source row is the minimum valid fragment in
``ids__visit_date``.  This deliberately retains rows with conflicting
pipe-delimited dates so they can be audited rather than silently discarded.
"""
from __future__ import annotations

import re
import pandas as pd

_MISSING = {"", "na", "n/a", "nan", "none", "unknown", "unk", "missing", "-99"}


def normalize_patient_id(value: object) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text.lower() in _MISSING:
        return pd.NA
    return re.sub(r"(?<=\d)\.0$", "", text)


def parse_visit_date_fragments(value: object) -> dict[str, object]:
    raw = value
    if value is None or pd.isna(value) or str(value).strip().lower() in _MISSING:
        return {"visit_date_raw": raw, "visit_date": pd.NaT, "visit_date_min": pd.NaT,
                "visit_date_max": pd.NaT, "n_date_fragments": 0,
                "n_valid_date_fragments": 0, "date_parse_status": "missing_date",
                "had_pipe_delimited_date": False}
    fragments = [part.strip() for part in str(value).split("|")]
    valid = []
    for fragment in fragments:
        parsed = pd.to_datetime(fragment, errors="coerce")
        if pd.notna(parsed):
            valid.append(pd.Timestamp(parsed).normalize())
    n_fragments, n_valid = len(fragments), len(valid)
    if not n_valid:
        status = "no_valid_date"
    elif n_valid < n_fragments:
        status = "partially_parsed"
    elif n_valid == 1:
        status = "single_valid_date"
    elif min(valid) == max(valid):
        status = "multiple_valid_dates_same_day"
    else:
        status = "multiple_valid_dates_different_days"
    return {"visit_date_raw": raw, "visit_date": min(valid) if valid else pd.NaT,
            "visit_date_min": min(valid) if valid else pd.NaT,
            "visit_date_max": max(valid) if valid else pd.NaT,
            "n_date_fragments": n_fragments, "n_valid_date_fragments": n_valid,
            "date_parse_status": status, "had_pipe_delimited_date": "|" in str(value)}


def add_parsed_visit_dates(df: pd.DataFrame, patient_id_col: str, visit_date_col: str) -> pd.DataFrame:
    if patient_id_col not in df or visit_date_col not in df:
        raise KeyError(f"Expected columns {patient_id_col!r} and {visit_date_col!r}")
    out = df.copy()
    out["patient_id"] = out[patient_id_col].map(normalize_patient_id).astype("string")
    parsed = pd.DataFrame(list(out[visit_date_col].map(parse_visit_date_fragments)), index=out.index)
    return out.join(parsed)


def collapse_patient_visit_rows(df: pd.DataFrame, *, patient_col: str = "patient_id", date_col: str = "visit_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = df[df[patient_col].notna() & df[date_col].notna()].copy()
    records = []
    for (patient, date), group in valid.groupby([patient_col, date_col], sort=True, dropna=False):
        records.append({patient_col: patient, date_col: date, "n_source_rows": len(group),
                        "source_row_indices": "|".join(map(str, group.index.tolist())),
                        "had_multiple_source_rows": len(group) > 1})
    audit = pd.DataFrame(records, columns=[patient_col, date_col, "n_source_rows", "source_row_indices", "had_multiple_source_rows"])
    return audit[[patient_col, date_col]].copy(), audit


def add_visit_timing(visits: pd.DataFrame, *, patient_col: str = "patient_id", date_col: str = "visit_date") -> pd.DataFrame:
    out = visits.copy()
    if out.duplicated([patient_col, date_col]).any():
        raise ValueError("patient-visit keys must be unique before assigning timing")
    out = out.sort_values([patient_col, date_col]).reset_index(drop=True)
    out["observed_baseline_date"] = out.groupby(patient_col)[date_col].transform("min")
    out["time_since_observed_baseline_days"] = (out[date_col] - out["observed_baseline_date"]).dt.days
    out["time_since_observed_baseline_years"] = out["time_since_observed_baseline_days"] / 365.25
    out["visit_number"] = out.groupby(patient_col).cumcount()
    out["visit_id"] = out[patient_col].astype(str) + "__" + out[date_col].dt.strftime("%Y%m%d")
    if (out["time_since_observed_baseline_days"] < 0).any() or not out["visit_id"].is_unique:
        raise ValueError("invalid canonical visit timing")
    return out

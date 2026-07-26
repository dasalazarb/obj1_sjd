"""Shared glandular/extraglandular overlap derivation logic.

Extracted from 06_overlap_glandular_followup.py and
06_overlap_glandular_followup_base_1st_Visit.py, which previously carried this
exact logic (constants + functions) as two byte-identical copies. Both scripts
now import from here; behavior is unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

MISSING_STRINGS = config.MISSING_STRINGS
NO_STRINGS = {"no", "n", "negative", "absent", "normal", "no activity", "no acitivity", "0", "0.0", "false"}
YES_STRINGS = {"yes", "y", "positive", "present", "abnormal", "ocular symptoms", "oral symptoms", "1", "1.0", "true"}
ESSDAI_ACTIVE = {"low activity", "moderate activity", "high activity", "high acitivity"}
ESSDAI_SCORE = {"no activity": 0, "no acitivity": 0, "low activity": 1, "moderate activity": 2, "high activity": 3, "high acitivity": 3}

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

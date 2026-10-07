"""Shared glandular/extraglandular overlap derivation logic.

Clinical flags in this module use three states: observed active, observed
inactive, and not evaluated.  A negative group requires every applicable
source in that group to be explicitly negative; a positive source is
sufficient for a positive group.  The aggregate sicca assessment is the one
exception: its explicit presence or absence is an authoritative observation.
Absence is data; missingness is lack of data.
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
    "tear_subsit": "visit_summary_-_2016_classification_criteria__ic_tear_subsit",
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


def parse_sicca(x: Any) -> bool | pd._libs.missing.NAType:
    """Parse an aggregate sicca assessment without collapsing missingness.

    In particular, ``sicca absent`` is an observed negative, not a missing
    value.  Generic documented yes/no values are accepted because the source
    column itself supplies the sicca context.
    """
    if is_missing_like(x):
        return pd.NA
    text = normalize_text(x)
    # Exported categorical labels may carry prefixes/suffixes (for example,
    # ``0 - Sicca absent``).  Parse their clinical meaning rather than relying
    # on one exact serialization of the label.
    sicca_negative = (
        "sicca" in text
        and any(marker in text for marker in ("absent", "negative", "not present"))
    ) or text.startswith("no sicca") or text.startswith("absence of sicca")
    sicca_positive = "sicca" in text and any(
        marker in text for marker in ("present", "positive")
    )
    if sicca_negative:
        return False
    if sicca_positive:
        return True
    if is_yes(x):
        return True
    if is_no(x):
        return False
    return pd.NA


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
    score = series.map(essdai_ordinal_score).astype("Float64")
    # Scores outside the producer's four-level scale are not interpretable.
    valid_score = score.isin([0.0, 1.0, 2.0, 3.0])
    active = active.where(valid_score).astype("boolean")
    score = score.where(valid_score)
    evaluable = active.notna().astype("boolean")
    return active, evaluable, score


def _tri_or(frame: pd.DataFrame) -> pd.Series:
    """Kleene OR: true wins; false requires every component to be decided."""
    values = frame.astype("boolean")
    result = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    result.loc[values.eq(True).any(axis=1)] = True  # noqa: E712
    result.loc[values.notna().all(axis=1) & values.eq(False).all(axis=1)] = False  # noqa: E712
    return result


def _any_active(row: pd.Series, cols: list[str], essdai: bool = False):
    """Return tri-state activity; all declared sources must decide a negative."""
    if not cols:
        return pd.NA
    vals = []
    for col in cols:
        if col not in row.index or is_missing_like(row[col]):
            vals.append(pd.NA)
        elif essdai:
            vals.append(essdai_string_to_active(row[col]))
        elif is_yes(row[col]):
            vals.append(True)
        elif is_no(row[col]):
            vals.append(False)
        else:
            vals.append(pd.NA)
    if any(value is True for value in vals):
        return True
    if all(value is False for value in vals):
        return False
    return pd.NA


def derive_glandular_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # ESSDAI glandular activity is exclusively the gland-swelling domain.
    # It is distinct from the aggregate sicca/objective/swelling phenotype.
    source = df.get(
        GLANDULAR_COLS["gland_swell"],
        pd.Series(pd.NA, index=df.index, dtype="object"),
    )
    active, evaluable, score = derive_domain_active(source)
    out["eg_glandular_domain_active"] = active
    out["eg_glandular_domain_evaluable"] = evaluable
    out["eg_glandular_ordinal_score"] = score
    aggregate_col = GLANDULAR_COLS["symptom_dry_eye_or_mouth"]
    aggregate = (
        df[aggregate_col].map(parse_sicca)
        if aggregate_col in df
        else pd.Series(pd.NA, index=df.index, dtype="boolean")
    )
    groups = {
        "eye_dryness": [
            GLANDULAR_COLS["dry_eye_3month"],
            GLANDULAR_COLS["sand_gravel_eye"],
            GLANDULAR_COLS["tear_subsit"],
        ],
        "mouth_dryness": [
            GLANDULAR_COLS["dry_mouth_3month"],
            GLANDULAR_COLS["difficulty_swallowing_dry_food"],
        ],
        "objective_eye": [GLANDULAR_COLS["ocular_stain"], GLANDULAR_COLS["lacrimal_dysfunction"]],
        "objective_mouth": [GLANDULAR_COLS["salivary_gland_movement"]],
        "salivary_gland_swelling": [GLANDULAR_COLS["gland_swell"]],
    }
    for name, cols in groups.items():
        out[f"glandular_{name}_active"] = pd.Series(
            df.apply(lambda r, p=cols, n=name: _any_active(r, p, essdai=(n == "salivary_gland_swelling")), axis=1),
            index=df.index, dtype="boolean",
        )
    detailed_sicca = _tri_or(
        out[
            [
                "glandular_eye_dryness_active",
                "glandular_mouth_dryness_active",
            ]
        ]
    )
    # The aggregate assessment is authoritative when recorded, but never
    # assigns its non-localized result to either detailed dryness dimension.
    aggregate = pd.Series(aggregate, index=df.index, dtype="boolean")
    out["sicca_active"] = aggregate.where(
        aggregate.notna(), detailed_sicca
    ).astype("boolean")
    out["sicca_evaluable"] = out["sicca_active"].notna().astype("boolean")
    active_cols = [f"glandular_{name}_active" for name in groups]
    glandular_sources = pd.DataFrame({
        "sicca": out["sicca_active"],
        "objective_eye": out["glandular_objective_eye_active"],
        "objective_mouth": out["glandular_objective_mouth_active"],
        "gland_swelling": out["glandular_salivary_gland_swelling_active"],
    })
    out["glandular_active"] = _tri_or(glandular_sources)
    out["glandular_evaluable"] = out["glandular_active"].notna().astype("boolean")
    objective = out[
        ["glandular_objective_eye_active", "glandular_objective_mouth_active"]
    ]
    out["objective_glandular_dysfunction_active"] = _tri_or(objective)
    out["objective_glandular_dysfunction_evaluable"] = out[
        "objective_glandular_dysfunction_active"
    ].notna().astype("boolean")
    objective_or_swelling = pd.DataFrame(
        {
            "objective": out["objective_glandular_dysfunction_active"],
            "swelling": out["glandular_salivary_gland_swelling_active"],
        },
        index=out.index,
    )
    out["objective_or_swelling_glandular_active"] = _tri_or(
        objective_or_swelling
    )
    out["objective_or_swelling_glandular_evaluable"] = out[
        "objective_or_swelling_glandular_active"
    ].notna().astype("boolean")
    complete = out[active_cols].notna().all(axis=1)
    out["n_glandular_manifestations_active"] = out[active_cols].sum(axis=1).astype("Int64").where(complete)
    # Completeness is deliberately component-based: aggregate positivity can
    # coexist with unknown individual dimensions.
    out["glandular_phenotype_complete"] = complete
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
            active = pd.Series(pd.NA, index=df.index, dtype="boolean")
            evaluable = pd.Series(False, index=df.index, dtype="boolean")
            score = pd.Series(pd.NA, index=df.index, dtype="Float64")
        out[meta["active_col"]] = active
        out[f"eg_{key}_evaluable"] = evaluable
        out[f"eg_{key}_ordinal_score"] = score
    if usable == 0:
        raise ValueError("No usable ESSDAI extraglandular domain columns were found.")
    active_cols = [m["active_col"] for m in EXTRAGLANDULAR_DOMAINS.values()]
    eval_cols = [f"eg_{k}_evaluable" for k in EXTRAGLANDULAR_DOMAINS]
    out["extraglandular_active"] = _tri_or(out[active_cols])
    out["extraglandular_evaluable"] = out["extraglandular_active"].notna().astype("boolean")
    complete = out[eval_cols].eq(True).all(axis=1)
    out["n_extraglandular_domains_active"] = out[active_cols].sum(axis=1).astype("Int64").where(complete)
    no_bio_heme_keys = [
        key for key in EXTRAGLANDULAR_DOMAINS
        if key not in {"biological", "hematologic"}
    ]
    no_bio_heme_active = [
        EXTRAGLANDULAR_DOMAINS[key]["active_col"] for key in no_bio_heme_keys
    ]
    no_bio_heme_complete = out[
        [f"eg_{key}_evaluable" for key in no_bio_heme_keys]
    ].eq(True).all(axis=1)
    out["n_extraglandular_domains_active_no_bio_heme"] = (
        out[no_bio_heme_active].sum(axis=1).astype("Int64").where(no_bio_heme_complete)
    )
    def active_names(row):
        names = [m["label"] for m in EXTRAGLANDULAR_DOMAINS.values() if pd.notna(row[m["active_col"]]) and bool(row[m["active_col"]])]
        return ";".join(names) if complete.loc[row.name] else pd.NA
    out["active_extraglandular_domains"] = out.apply(active_names, axis=1)
    return out


def derive_overlap_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["overlap_active"] = df["glandular_active"].astype("boolean") & df["extraglandular_active"].astype("boolean")
    df["overlap_evaluable"] = (df["glandular_active"].notna() & df["extraglandular_active"].notna()).astype("boolean")
    df["overlap_intensity_count"] = df["n_glandular_manifestations_active"] + df["n_extraglandular_domains_active"]
    df["overlap_status"] = "unclassifiable"
    known = df["overlap_evaluable"].eq(True)
    g, e = df["glandular_active"], df["extraglandular_active"]
    df.loc[known & g.eq(True) & e.eq(True), "overlap_status"] = "overlap"
    df.loc[known & g.eq(True) & e.eq(False), "overlap_status"] = "glandular_only"
    df.loc[known & g.eq(False) & e.eq(True), "overlap_status"] = "extraglandular_only"
    df.loc[known & g.eq(False) & e.eq(False), "overlap_status"] = "neither"
    return df

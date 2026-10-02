"""Contrato público central de Step 10 y adaptador legacy transitorio.

El Parquet sólo contiene nombres públicos. El adaptador añade aliases exclusivamente
en memoria para consumidores que todavía migran; se retirará al eliminar soporte del
contrato ``clinical_episode_curated_v1``.
"""
from __future__ import annotations

import pandas as pd

PUBLIC_CONCEPTS = {
    "pop_status": "pop__status", "essdai_total": "essdai__total",
    "esspri_dryness": "esspri__dryness", "esspri_fatigue": "esspri__fatigue",
    "esspri_pain": "esspri__pain", "esspri_total": "esspri__total",
    "age_baseline": "demo__age_at_baseline", "sex": "demo__sex",
    "ids__age_at_visit": "demo__age_at_visit", "ids__race": "demo__race",
    "ids__ethnicity": "demo__ethnicity", "ids__interval_name": "spine__interval_name",
}


def public_name(original: str) -> str:
    """Resolve common original concepts without duplicating mapping tables."""
    return PUBLIC_CONCEPTS.get(original, original)


def add_legacy_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Add v1 aliases in memory; never use this when writing the integrated master."""
    if not any(c.startswith(("pop__", "essdai__", "esspri__", "pro__", "lab__", "sero__", "ext__", "ovl__", "demo__")) for c in frame):
        return frame
    out = frame.copy()
    aliases = {new: old for old, new in PUBLIC_CONCEPTS.items()}
    for column in frame:
        old = aliases.get(column)
        if old is None:
            if column.startswith(("lab__", "sero__", "pro__", "ext__")):
                old = column.split("__", 1)[1]
            elif column.startswith("essdai__") and column != "essdai__total":
                old = "eg_" + column.split("__", 1)[1]
            elif column.startswith("ovl__"):
                stem = column.split("__", 1)[1]
                old = "overlap_" + stem if stem in {"status", "active", "evaluable", "intensity_count"} else stem
        if old and old not in out:
            out[old] = frame[column]
    # Historical observed spelling was an in-memory alias, never another Parquet column.
    if "esspri__total" in frame and "esspri_total_observed" not in out:
        out["esspri_total_observed"] = frame["esspri__total"]
    return out

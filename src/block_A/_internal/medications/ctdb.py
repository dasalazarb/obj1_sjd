"""Visit-level CTDB medication reconstruction, linked to Step 00 episodes.
Documented CURRENT medication at visit is not confirmed administration/start.
Writes exclusively to Step 08 restricted folder; never modifies upstream data.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pandas as pd

from . import documentation as audit

VERSION = "08-ctdb-visit-1.1"
EDA = audit.v1.common.EDA_SJD_ROOT
SPINE = audit.v1.common.EPISODE_SPINE_PARQUET

def build(visits, row_map, spine, secret):
    if len(secret) < 32:
        raise ValueError("pseudonym key too short")
    keys = {"row_id_raw"}
    for name, table in (("visits", visits), ("row_map", row_map)):
        if not keys.issubset(table) or table.row_id_raw.isna().any() or table.row_id_raw.duplicated().any():
            raise ValueError(name + " missing/duplicated row_id_raw")
    required_map = {"patient_id", "clinical_episode_id", "collection_date"}
    required_spine = {"patient_id", "clinical_episode_id", "episode_start_date",
                      "episode_end_date", "clinical_anchor_date", "clinical_visit"}
    if not required_map.issubset(row_map) or not required_spine.issubset(spine):
        raise ValueError("missing authoritative episode columns")
    patient_col = next((x for x in ["patient_id", "ids__patient_record_number",
                                   "patient_record_number"] if x in visits), None)
    if patient_col is None:
        raise ValueError("source patient column missing")
    if set(visits.row_id_raw) != set(row_map.row_id_raw):
        raise ValueError("row map does not cover all original rows")
    m = row_map[["row_id_raw", "patient_id", "clinical_episode_id", "collection_date"]].rename(
        columns={"patient_id": "patient_id_map", "clinical_episode_id": "episode_id_map"})
    linked = visits.merge(m, on="row_id_raw", validate="one_to_one")
    if not linked[patient_col].astype(str).str.strip().eq(linked.patient_id_map.astype(str).str.strip()).all():
        raise ValueError("source/map patient disagreement")
    episode = spine[list(required_spine)].copy()
    episode["patient_key"] = episode.patient_id.astype(str).str.strip()
    episode["episode_key"] = episode.clinical_episode_id.astype(str).str.strip()
    if episode[["patient_key", "episode_key"]].isna().any().any() or episode[["patient_key", "episode_key"]].duplicated().any():
        raise ValueError("duplicate Step00 episode keys")
    linked["patient_key"] = linked.patient_id_map.astype(str).str.strip()
    linked["episode_key"] = linked.episode_id_map.astype(str).str.strip()
    linked = linked.merge(episode.drop(columns=["patient_id", "clinical_episode_id"]),
                          on=["patient_key", "episode_key"], how="left",
                          validate="many_to_one", indicator=True)
    cohort = set(episode.patient_key)
    if (linked.patient_key.isin(cohort) & linked._merge.ne("both")).any():
        raise ValueError("eligible patient assigned to missing Step00 episode")
    linked = linked.loc[linked._merge.eq("both")]
    records = []
    for _, r in linked.iterrows():
        day = audit.parse_date(r["collection_date"])
        observed = day["parsed"][:10] if day["status"] == "valid" and day["precision"] in {"day", "datetime"} else ""
        lower = audit.parse_date(r["episode_start_date"])
        upper = audit.parse_date(r["episode_end_date"])
        episode_bounds_valid = (
            lower["status"] == upper["status"] == "valid"
            and lower["precision"] in {"day", "datetime"}
            and upper["precision"] in {"day", "datetime"}
            and lower["lower_bound"] <= upper["upper_bound"]
        )
        report_within_episode = bool(
            observed and episode_bounds_valid
            and lower["lower_bound"] <= observed <= upper["upper_bound"]
        )
        for slot, column in enumerate(audit.CTDB_RX_NAMES, 1):
            if column not in linked or not audit.present(r.get(column)):
                continue
            name = str(r[column]).strip()
            if name.casefold() in audit.v1.NEGATIVE_TOKENS:
                continue
            route_col = audit.CTDB_RX_ROUTES[slot-1]
            route, _ = audit.v1.decode_category(r.get(route_col), audit.v1.ROUTE_CODES)
            start = audit.parse_date(r.get(audit.CTDB_RX_START_BY_SLOT[slot]))
            classification = audit.classification.classify(name, "", route, audit.DRUG_RULES, source_kind="CTDB")
            records.append({
                "patient_pseudonym": audit.pseudonym(secret, "patient", r.patient_key),
                "row_pseudonym": audit.pseudonym(secret, "source_visit", str(r.row_id_raw)),
                "clinical_episode_id": r.episode_key,
                "slot": slot,
                "medication_name_raw": name,
                "source_kind": "CTDB",
                "frequency_raw": audit.raw(r.get(audit.CTDB_RX_FREQ_BY_SLOT[slot])),
                "unassigned_visit_doses_json": json.dumps({c:audit.raw(r.get(c)) for c in audit.CTDB_UNASSIGNED_DOSES if audit.present(r.get(c))}, sort_keys=True),
                "dose_slot_mapping": "unassigned_not_inferred",
                "ingredient_canonical": classification["ingredient_canonical"],
                "clinical_category_primary": classification["clinical_category_primary"],
                "classification_confidence": classification["classification_confidence"],
                "clinical_category_detail": classification["clinical_category_detail"],
                "study_category_rule_status": classification["study_category_rule_status"],
                "route_canonical": classification["route_canonical"],
                "classification_review_reasons": classification["review_reasons"],
                "medication_date_within_episode": report_within_episode,
                "episode_dates_valid": episode_bounds_valid,
                "classification_version": audit.classification.VERSION,
                "visit_documentation_date": observed,
                "visit_report_semantics": "current_medication_reported_at_source_visit",
                "source_reported_start_lower": start["lower_bound"],
                "source_reported_start_upper": start["upper_bound"],
                "source_reported_start_precision": start["precision"],
                "episode_start_date": str(r.episode_start_date)[:10],
                "episode_end_date": str(r.episode_end_date)[:10],
                "clinical_anchor_date": str(r.clinical_anchor_date)[:10],
                "clinical_visit": bool(r.clinical_visit),
                "actual_exposure_confirmed": False,
                "recovery_version": VERSION,
            })
    catalog = pd.DataFrame(records)
    if catalog.empty:
        return catalog, catalog.copy(), {"n_original_visit_medication_mentions": 0}
    dated = catalog[catalog.visit_documentation_date.ne("")].copy()
    summary = {"version": VERSION, "n_original_visit_medication_mentions": len(catalog),
               "n_dated": len(dated), "n_undated": len(catalog)-len(dated),
               "n_episodes": int(catalog.clinical_episode_id.nunique()),
               "n_patients": int(catalog.patient_pseudonym.nunique()),
               "n_nonclinical_episode_mentions": int((~catalog.clinical_visit).sum()),
               "n_report_dates_outside_episode_or_invalid_bounds": int(
                   (~catalog.medication_date_within_episode).sum()),
               "n_report_dates_within_episode": int(catalog.medication_date_within_episode.sum()),
               "n_original_visit_high_exact": int(
                   catalog.classification_confidence.eq("high_exact").sum()),
               "reference": "Step 00 episode spine + EDA row_id_raw map",
               "actual_exposure": "NO_GO",
               "downstream_integration": "episode_documentation_only",
               "note": "Original-visit and collapsed interval slots have different grains."}
    return catalog, dated, summary

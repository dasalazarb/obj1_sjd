#!/usr/bin/env python3
"""Integrate previously-derived Pop, overlap, and PRO data by patient-visit.

This module deliberately only joins existing clinical derivations and creates
longitudinal variables; it never recalculates clinical scores or classifications.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402
from src.derivations.visit_dates import normalize_patient_id  # noqa: E402

KEYS = ["patient_id", "visit_id"]
LEGACY_PATIENT_ID_COLUMN = "ids__patient_record_number"
SHARED_SPINE_COLUMNS = ["visit_date", "visit_number", "observed_baseline_date",
                        "time_since_observed_baseline_days", "time_since_observed_baseline_years",
                        "protocol", "interval_name"]
POP_COLUMNS = ["essdai_total", "esspri_dryness", "esspri_fatigue", "esspri_pain", "esspri_total",
               "esspri_total_observed", "pop_status", "pop_status_display", "pop_missingness_label",
               "baseline_pop_status", "baseline_pop_status_display", "esspri_total_s1_one_proxy",
               "pop_status_s1_one_proxy", "esspri_total_s2_up_to_two_proxies",
               "pop_status_s2_up_to_two_proxies", "esspri_total_s3_all_available", "pop_status_s3_all_available"]
OVERLAP_COLUMNS = ["glandular_active", "glandular_evaluable", "extraglandular_active",
                   "extraglandular_evaluable", "overlap_active", "overlap_status",
                   "n_extraglandular_domains_active", "eg_constitutional_active",
                   "eg_lymphadenopathy_active", "eg_articular_active", "eg_cutaneous_active",
                   "eg_pulmonary_active", "eg_renal_active", "eg_muscular_active", "eg_pns_active",
                   "eg_cns_active", "eg_hematologic_active", "eg_biological_active",
                   "time_since_diagnosis_days", "time_since_diagnosis_years", "dx_date", "dx_date_precision"]
PRO_COLUMNS = ["sf36_physical_functioning", "sf36_role_physical", "sf36_bodily_pain",
               "sf36_general_health", "sf36_vitality", "sf36_social_functioning", "sf36_role_emotional",
               "sf36_mental_health", "sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global",
               "sf36_scoring_valid", "profad_scoring_valid",
               "mdafs_scoring_valid", "esspri_scoring_valid"]


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def canonicalize_patient_id(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return a frame with the canonical patient ID column when possible.

    The diagnosis-anchored overlap export historically retained its raw input
    identifier (``ids__patient_record_number``) instead of ``patient_id``.
    Normalize that identifier with the same routine used to build the visit
    spine so its values remain valid merge keys (for example, ``"123.0"``
    becomes ``"123"``).
    """
    if "patient_id" in frame:
        return frame
    if LEGACY_PATIENT_ID_COLUMN not in frame:
        return frame
    canonical = frame.copy()
    canonical["patient_id"] = canonical[LEGACY_PATIENT_ID_COLUMN].map(normalize_patient_id).astype("string")
    return canonical


def assert_unique_keys(frame: pd.DataFrame, source: str) -> None:
    require_columns(frame, KEYS, source)
    duplicates = frame.loc[frame.duplicated(KEYS, keep=False), KEYS]
    if not duplicates.empty:
        examples = duplicates.drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"{source} has {len(duplicates.drop_duplicates())} duplicate patient_id + visit_id keys; examples: {examples}")


def compare_shared_column(base: pd.DataFrame, other: pd.DataFrame, column: str) -> int:
    """Return the count of non-missing, discordant values for a shared column."""
    if column not in base or column not in other:
        return 0
    compared = base[KEYS + [column]].merge(other[KEYS + [column]], on=KEYS, how="inner", suffixes=("_base", "_other"))
    left, right = compared[f"{column}_base"], compared[f"{column}_other"]
    return int((left.notna() & right.notna() & ~left.eq(right)).sum())


def select_source(frame: pd.DataFrame, columns: list[str], source: str) -> pd.DataFrame:
    require_columns(frame, KEYS + ["visit_date"], source)
    return frame[KEYS + ["visit_date"] + [col for col in columns if col in frame]].copy()


def baseline_value(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame.loc[frame.visit_number.eq(0), ["patient_id", column]].set_index("patient_id")[column]
    return frame.patient_id.map(values)


def build_integrated(visit_spine: pd.DataFrame, pop: pd.DataFrame, overlap: pd.DataFrame, pros: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the canonical longitudinal table, returning it and date discrepancy counts."""
    visit_spine, pop, overlap, pros = [
        canonicalize_patient_id(frame, source)
        for source, frame in [("visit_spine", visit_spine), ("pop", pop), ("overlap", overlap), ("pros", pros)]
    ]
    for source, frame in [("visit_spine", visit_spine), ("pop", pop), ("overlap", overlap), ("pros", pros)]:
        assert_unique_keys(frame, source)
    require_columns(visit_spine, KEYS + ["visit_date", "visit_number", "time_since_observed_baseline_days"], "visit_spine")
    spine = visit_spine.copy()
    spine["visit_date"] = pd.to_datetime(spine["visit_date"])
    discrepancy_counts = {f"{source}_{column}": compare_shared_column(spine, frame, column)
                          for source, frame in [("pop", pop), ("overlap", overlap), ("pros", pros)]
                          for column in SHARED_SPINE_COLUMNS if column in frame}
    pop = select_source(pop, POP_COLUMNS, "pop").drop(columns="visit_date")
    overlap = select_source(overlap, OVERLAP_COLUMNS, "overlap").drop(columns="visit_date")
    pros = select_source(pros, PRO_COLUMNS, "pros").drop(columns="visit_date")
    integrated = spine.merge(pop, on=KEYS, how="left", validate="one_to_one")
    integrated = integrated.merge(overlap, on=KEYS, how="left", validate="one_to_one")
    integrated = integrated.merge(pros, on=KEYS, how="left", validate="one_to_one")
    return derive_longitudinal(integrated), discrepancy_counts


def derive_longitudinal(integrated: pd.DataFrame) -> pd.DataFrame:
    """Add availability, lagged, transition, and change variables to an integrated frame."""
    integrated = integrated.sort_values(["patient_id", "visit_date"]).reset_index(drop=True).copy()
    grouped = integrated.groupby("patient_id", sort=False)
    if grouped.visit_number.diff().dropna().lt(0).any():
        raise ValueError("visit_number is inconsistent with patient visit_date ordering")
    integrated["has_pop_data"] = integrated.pop_status.notna()
    integrated["has_overlap_data"] = integrated.overlap_status.notna()
    integrated["has_pro_data"] = integrated[[c for c in ["sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global"] if c in integrated]].notna().any(axis=1)
    for output, source in [("has_essdai", "essdai_total"), ("has_esspri", "esspri_total"), ("has_sf36_pcs", "sf36_pcs"), ("has_sf36_mcs", "sf36_mcs"), ("has_profad", "profad_total"), ("has_mdafs", "mdafs_global")]:
        integrated[output] = integrated[source].notna()
    integrated["n_data_blocks_available"] = integrated[["has_pop_data", "has_overlap_data", "has_pro_data"]].sum(axis=1)
    integrated["n_visits_patient"] = grouped.visit_id.transform("count")
    integrated["is_observed_baseline"] = integrated.visit_number.eq(0)
    integrated["is_last_observed_visit"] = grouped.visit_date.transform("max").eq(integrated.visit_date)
    for output, source, periods in [("previous_visit_id", "visit_id", 1), ("next_visit_id", "visit_id", -1), ("previous_visit_date", "visit_date", 1), ("next_visit_date", "visit_date", -1), ("previous_pop", "pop_status", 1), ("next_pop", "pop_status", -1), ("previous_overlap_status", "overlap_status", 1), ("next_overlap_status", "overlap_status", -1), ("previous_extraglandular_active", "extraglandular_active", 1)]:
        integrated[output] = grouped[source].shift(periods)
    integrated["time_from_previous_visit_days"] = (integrated.visit_date - integrated.previous_visit_date).dt.days
    integrated["time_to_next_visit_days"] = (integrated.next_visit_date - integrated.visit_date).dt.days
    integrated["time_from_previous_visit_years"] = integrated.time_from_previous_visit_days / 365.25
    integrated["time_to_next_visit_years"] = integrated.time_to_next_visit_days / 365.25
    integrated["baseline_pop"] = integrated.get("baseline_pop_status", pd.Series(pd.NA, index=integrated.index)).combine_first(baseline_value(integrated, "pop_status"))
    integrated["baseline_overlap_status"] = baseline_value(integrated, "overlap_status")
    integrated["baseline_extraglandular_active"] = baseline_value(integrated, "extraglandular_active")
    integrated["pop_transition_from_previous"] = transition(integrated.previous_pop, integrated.pop_status)
    integrated["pop_transition_to_next"] = transition(integrated.pop_status, integrated.next_pop)
    valid_pop = {"Pop1", "Pop2", "Pop3"}
    integrated["pop_transition_evaluable"] = integrated.previous_pop.isin(valid_pop) & integrated.pop_status.isin(valid_pop)
    integrated["changed_pop_from_previous"] = pd.Series(pd.NA, index=integrated.index, dtype="boolean")
    mask = integrated.pop_transition_evaluable
    integrated.loc[mask, "changed_pop_from_previous"] = integrated.loc[mask, "previous_pop"].ne(integrated.loc[mask, "pop_status"])
    integrated["overlap_transition_from_previous"] = transition(integrated.previous_overlap_status, integrated.overlap_status)
    sufficient = lambda s: s.notna() & ~s.astype(str).str.contains("insufficient|unknown|unclass", case=False, na=False)
    integrated["overlap_transition_evaluable"] = sufficient(integrated.previous_overlap_status) & sufficient(integrated.overlap_status)
    evaluable = grouped.extraglandular_evaluable.shift(1).eq(True) & integrated.extraglandular_evaluable.eq(True)
    incident = integrated.previous_extraglandular_active.eq(False) & integrated.extraglandular_active.eq(True) & evaluable
    integrated["incident_extraglandular_from_previous"] = incident.astype("boolean").where(evaluable, pd.NA)
    integrated["first_incident_extraglandular"] = (incident & ~integrated.baseline_extraglandular_active.eq(True) & ~incident.groupby(integrated.patient_id).shift(fill_value=False).groupby(integrated.patient_id).cummax()).astype("boolean").where(evaluable, pd.NA)
    delta_map = {"delta_essdai": "essdai_total", "delta_esspri": "esspri_total", "delta_pcs": "sf36_pcs", "delta_mcs": "sf36_mcs", "delta_profad_total": "profad_total", "delta_mdafs_global": "mdafs_global", "delta_n_extraglandular_domains": "n_extraglandular_domains_active"}
    for output, source in delta_map.items(): integrated[output] = grouped[source].diff()
    baseline_map = {"change_from_baseline_essdai": "essdai_total", "change_from_baseline_esspri": "esspri_total", "change_from_baseline_pcs": "sf36_pcs", "change_from_baseline_mcs": "sf36_mcs", "change_from_baseline_profad_total": "profad_total", "change_from_baseline_mdafs_global": "mdafs_global", "change_from_baseline_n_extraglandular_domains": "n_extraglandular_domains_active"}
    for output, source in baseline_map.items(): integrated[output] = integrated[source] - baseline_value(integrated, source)
    for output, source in [("next_essdai", "essdai_total"), ("next_esspri", "esspri_total"), ("next_pcs", "sf36_pcs"), ("next_mcs", "sf36_mcs"), ("next_profad_total", "profad_total"), ("next_mdafs_global", "mdafs_global"), ("next_extraglandular_active", "extraglandular_active"), ("next_n_extraglandular_domains", "n_extraglandular_domains_active")]: integrated[output] = grouped[source].shift(-1)
    integrated["transition_to_pop1_next_visit"] = integrated.pop_status.isin(["Pop2", "Pop3"]) & integrated.next_pop.eq("Pop1")
    integrated["at_risk_transition_to_pop1"] = integrated.pop_status.isin(["Pop2", "Pop3"]) & integrated.next_pop.isin(valid_pop)
    integrated["pcs_decreased_from_previous"] = integrated.delta_pcs < 0
    integrated["mcs_decreased_from_previous"] = integrated.delta_mcs < 0
    integrated["integration_version"] = "v1_observed_pop_overlap_pros"
    integrated["integration_run_date"] = date.today().isoformat()
    return integrated


def transition(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left.astype("string") + "_to_" + right.astype("string")).where(left.notna() & right.notna(), pd.NA)


def coverage(integrated: pd.DataFrame) -> pd.DataFrame:
    measures = {"pop_status": integrated.has_pop_data, "overlap_status": integrated.has_overlap_data, "essdai_total": integrated.has_essdai, "esspri_total": integrated.has_esspri, "sf36_pcs": integrated.has_sf36_pcs, "sf36_mcs": integrated.has_sf36_mcs, "profad_total": integrated.has_profad, "mdafs_global": integrated.has_mdafs, "complete_pop_overlap": integrated.has_pop_data & integrated.has_overlap_data, "complete_pop_pro": integrated.has_pop_data & integrated.has_pro_data, "complete_overlap_pro": integrated.has_overlap_data & integrated.has_pro_data, "complete_all_three_blocks": integrated.n_data_blocks_available.eq(3)}
    n_patients = integrated.patient_id.nunique()
    return pd.DataFrame([{"measure": name, "n_visits_available": int(mask.sum()), "pct_visits_available": 100 * mask.mean(), "n_patients_available": int(integrated.loc[mask, "patient_id"].nunique()), "pct_patients_available": 100 * integrated.loc[mask, "patient_id"].nunique() / n_patients if n_patients else np.nan} for name, mask in measures.items()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spine", type=Path, default=common.VISIT_SPINE_PARQUET); parser.add_argument("--pop", type=Path, default=common.POP_LONGITUDINAL_PARQUET); parser.add_argument("--overlap", type=Path, default=common.OVERLAP_LONGITUDINAL_PARQUET); parser.add_argument("--pros", type=Path, default=common.PROS_LONGITUDINAL_PARQUET); parser.add_argument("--output", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    args = parser.parse_args(); common.ensure_output_dirs()
    frames = [
        canonicalize_patient_id(pd.read_parquet(path), source)
        for source, path in zip(["visit_spine", "pop", "overlap", "pros"], [args.spine, args.pop, args.overlap, args.pros])
    ]
    integrated, discrepancies = build_integrated(*frames)
    assert integrated.visit_id.is_unique and not integrated.duplicated(KEYS).any()
    assert integrated.time_since_observed_baseline_days.ge(0).all()
    assert integrated.loc[integrated.visit_number.eq(0)].groupby("patient_id").size().eq(1).all()
    assert integrated.time_to_next_visit_days.dropna().ge(0).all()
    args.output.parent.mkdir(parents=True, exist_ok=True); integrated.to_parquet(args.output, index=False); integrated.to_csv(args.output.with_suffix(".csv"), index=False)
    coverage(integrated).to_csv(common.BLOCKA_TABLES_DIR / "10_integrated_longitudinal_coverage.csv", index=False)
    spine_ids = set(frames[0].visit_id); summary = []; unmatched = []
    for name, frame in zip(["visit_spine", "pop", "overlap", "pros"], frames):
        matched = frame.visit_id.isin(spine_ids); summary.append({"source": name, "n_source_rows": len(frame), "n_unique_patients": frame.patient_id.nunique(), "n_matched_visit_ids": int(matched.sum()), "n_unmatched_visit_ids": int((~matched).sum()), "pct_matched_visit_ids": 100 * matched.mean() if len(frame) else np.nan})
        if (~matched).any(): unmatched.append(frame.loc[~matched, ["patient_id", "visit_id", "visit_date"]].assign(source=name))
    pd.DataFrame(summary).to_csv(common.BLOCKA_QC_DIR / "10_integrated_merge_summary.csv", index=False)
    if unmatched: pd.concat(unmatched).loc[:, ["source", "patient_id", "visit_id", "visit_date"]].to_csv(common.BLOCKA_QC_DIR / "10_integrated_unmatched_visits.csv", index=False)
    qc = {"n_rows_integrated": len(integrated), "n_unique_patients": int(integrated.patient_id.nunique()), "n_unique_visit_ids": int(integrated.visit_id.nunique()), "n_baseline_rows": int(integrated.is_observed_baseline.sum()), "n_rows_with_pop": int(integrated.has_pop_data.sum()), "n_rows_with_overlap": int(integrated.has_overlap_data.sum()), "n_rows_with_pro": int(integrated.has_pro_data.sum()), "n_rows_with_all_three_blocks": int(integrated.n_data_blocks_available.eq(3).sum()), "n_pop_transitions_evaluable": int(integrated.pop_transition_evaluable.sum()), "n_overlap_transitions_evaluable": int(integrated.overlap_transition_evaluable.sum()), "n_incident_extraglandular_events": int(integrated.incident_extraglandular_from_previous.sum()), "n_transition_to_pop1_events": int(integrated.transition_to_pop1_next_visit.sum()), "n_negative_visit_intervals": int(integrated.time_to_next_visit_days.dropna().lt(0).sum()), "n_duplicate_visit_ids": int(integrated.visit_id.duplicated().sum()), "shared_column_discrepancies": discrepancies}
    (common.BLOCKA_QC_DIR / "10_integrated_longitudinal_qc.json").write_text(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()

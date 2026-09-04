#!/usr/bin/env python3
"""Integrate validated upstream products on the authoritative clinical spine.

One output row represents one patient and one clinical episode.  This boundary
does not reconstruct episodes, baselines, scores, or clinical classifications.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402

KEYS = ["patient_id", "clinical_episode_id"]
STRUCTURAL_COLUMNS = [
    "patient_id", "clinical_episode_id", "clinical_anchor_date",
    "clinical_visit_number", "clinical_visit", "visit_type",
    "episode_start_date", "episode_end_date", "clinical_baseline_episode_id",
    "clinical_baseline_date", "is_clinical_baseline",
    "time_since_clinical_baseline_days", "time_since_clinical_baseline_years",
]
DATE_COLUMNS = {"clinical_anchor_date", "episode_start_date", "episode_end_date", "clinical_baseline_date"}


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise AssertionError(f"{source} is missing required columns: {missing}")


def _normalise_structure(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in DATE_COLUMNS & set(frame.columns):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ["patient_id", "clinical_episode_id", "clinical_baseline_episode_id"]:
        if column in frame:
            frame[column] = frame[column].astype("string")
    return frame


def _different(left: pd.Series, right: pd.Series) -> pd.Series:
    """Null-safe semantic inequality used for copied spine metadata."""
    return left.isna().ne(right.isna()) | (left.notna() & right.notna() & ~left.eq(right))


def validate_spine(spine: pd.DataFrame) -> pd.DataFrame:
    require_columns(spine, STRUCTURAL_COLUMNS, "clinical_spine")
    spine = _normalise_structure(spine)
    if spine.duplicated(KEYS).any():
        raise AssertionError("clinical_spine has duplicate patient_id + clinical_episode_id keys")
    if not spine["clinical_visit"].fillna(False).astype(bool).all():
        raise AssertionError("clinical_spine contains non-clinical visits")
    if spine["clinical_visit_number"].isna().any() or spine["clinical_visit_number"].lt(1).any():
        raise AssertionError("clinical_visit_number must be >= 1")
    ordered = spine.sort_values(["patient_id", "clinical_anchor_date", "clinical_visit_number", "clinical_episode_id"])
    groups = ordered.groupby("patient_id", sort=False)
    if groups["clinical_visit_number"].diff().dropna().le(0).any():
        raise AssertionError("clinical_visit_number must be strictly increasing within patient")
    if any(not group["clinical_anchor_date"].is_monotonic_increasing for _, group in groups):
        raise AssertionError("clinical_anchor_date must increase monotonically within patient")
    baseline = spine["is_clinical_baseline"].fillna(False).astype(bool)
    if baseline.groupby(spine["patient_id"]).sum().gt(1).any():
        raise AssertionError("a patient has more than one clinical baseline")
    if (spine.loc[baseline, "clinical_episode_id"] != spine.loc[baseline, "clinical_baseline_episode_id"]).any():
        raise AssertionError("clinical baseline episode identity is inconsistent")
    if _different(spine.loc[baseline, "clinical_anchor_date"], spine.loc[baseline, "clinical_baseline_date"]).any():
        raise AssertionError("clinical baseline date is inconsistent")
    has_baseline = spine["clinical_baseline_episode_id"].notna()
    if spine.loc[has_baseline, "time_since_clinical_baseline_days"].dropna().lt(0).any():
        raise AssertionError("negative time since clinical baseline")
    return spine


def validate_source(spine: pd.DataFrame, source: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict, list[dict], list[dict]]:
    source = _normalise_structure(source)
    require_columns(source, KEYS, name)
    duplicates = int(source.duplicated(KEYS).sum())
    spine_keys = set(map(tuple, spine[KEYS].itertuples(index=False, name=None)))
    source_keys = set(map(tuple, source[KEYS].itertuples(index=False, name=None)))
    missing, extra = spine_keys - source_keys, source_keys - spine_keys
    mismatches = ([{"source": name, **dict(zip(KEYS, key)), "mismatch": "missing_from_source"} for key in sorted(missing)]
                  + [{"source": name, **dict(zip(KEYS, key)), "mismatch": "extra_in_source"} for key in sorted(extra)])
    discrepancies: list[dict] = []
    shared_structural = [column for column in STRUCTURAL_COLUMNS if column not in KEYS and column in source]
    compared = spine[KEYS + shared_structural].merge(
        source[KEYS + shared_structural], on=KEYS, suffixes=("_spine", "_source")
    )
    for column in shared_structural:
        mask = _different(compared[f"{column}_spine"], compared[f"{column}_source"])
        for row in compared.loc[mask, KEYS + [f"{column}_spine", f"{column}_source"]].itertuples(index=False, name=None):
            discrepancies.append({"source": name, **dict(zip(KEYS, row[:2])), "variable": column,
                                  "spine_value": row[2], "source_value": row[3]})
    inherited_spine_columns = sorted((set(source.columns) & set(spine.columns)) - set(KEYS))
    summary = {"source": name, "n_rows": len(source), "n_patients": source.patient_id.nunique(),
               "n_unique_patient_episode": len(source_keys), "n_duplicates": duplicates,
               "n_missing_spine_keys": len(missing), "n_extra_keys": len(extra),
               "structural_discrepancies": len(discrepancies),
               "n_inherited_spine_columns_dropped": len(inherited_spine_columns)}
    if duplicates or missing or extra or discrepancies:
        raise AssertionError(f"{name} violates clinical-spine contract: {summary}")
    clean = source.drop(columns=inherited_spine_columns, errors="ignore")
    return clean, summary, mismatches, discrepancies


def _coerce_lab_dtypes(frame: pd.DataFrame, lab_columns: set[str]) -> pd.DataFrame:
    frame = frame.copy()
    suffixes = {"__value": "Float64", "__text": "string", "__unit": "string",
                "__days_from_anchor": "Int64", "__n_measurements": "Int64",
                "__conflict": "boolean", "__episode_status": "string", "__selection_status": "string"}
    for column in lab_columns:
        if column.endswith("__measurement_date"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        else:
            for suffix, dtype in suffixes.items():
                if column.endswith(suffix):
                    frame[column] = frame[column].astype(dtype)
                    break
    return frame


def derive_longitudinal(frame: pd.DataFrame, lab_columns: set[str]) -> pd.DataFrame:
    frame = _coerce_lab_dtypes(frame, lab_columns)
    frame = frame.sort_values(["patient_id", "clinical_anchor_date", "clinical_visit_number", "clinical_episode_id"]).reset_index(drop=True)
    grouped = frame.groupby("patient_id", sort=False)
    frame["has_pop_state"] = frame.get("pop_status", pd.Series(pd.NA, index=frame.index)).isin(["Pop1", "Pop2", "Pop3"])
    frame["has_essdai"] = frame.get("essdai_total", pd.Series(pd.NA, index=frame.index)).notna()
    observed = "esspri_total_observed" if "esspri_total_observed" in frame else "esspri_total"
    frame["has_esspri_observed"] = frame.get(observed, pd.Series(pd.NA, index=frame.index)).notna()
    counts = [c for c in lab_columns if c.endswith("__n_measurements")]
    dates = [c for c in lab_columns if c.endswith("__measurement_date")]
    frame["has_lab_measurement"] = (frame[counts].fillna(0).gt(0).any(axis=1) if counts else
                                    frame[dates].notna().any(axis=1) if dates else False)
    frame["has_overlap_data"] = frame.get("overlap_evaluable", pd.Series(False, index=frame.index)).eq(True)
    primaries = [c for c in ["esspri_total", "sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global"] if c in frame]
    frame["has_pro_data"] = frame[primaries].notna().any(axis=1) if primaries else False
    frame["n_integrated_blocks_available"] = frame[["has_pop_state", "has_lab_measurement", "has_overlap_data", "has_pro_data"]].sum(axis=1).astype("Int64")
    frame["n_clinical_visits_patient"] = grouped["clinical_episode_id"].transform("size").astype("Int64")
    frame["is_last_clinical_visit"] = grouped.cumcount().eq(frame["n_clinical_visits_patient"] - 1)
    frame["previous_clinical_episode_id"] = grouped["clinical_episode_id"].shift()
    frame["previous_clinical_anchor_date"] = grouped["clinical_anchor_date"].shift()
    days = (frame["clinical_anchor_date"] - frame["previous_clinical_anchor_date"]).dt.days
    frame["time_from_previous_clinical_episode_days"] = days.astype("Int64")
    frame["time_from_previous_clinical_episode_years"] = (days / 365.25).astype("Float64")
    lag_map = {"pop_status": "previous_pop_status", "essdai_total": "previous_essdai_total",
               "esspri_total": "previous_esspri_total", "overlap_status": "previous_overlap_status",
               "extraglandular_active": "previous_extraglandular_active", "sf36_pcs": "previous_sf36_pcs",
               "sf36_mcs": "previous_sf36_mcs", "profad_total": "previous_profad_total", "mdafs_global": "previous_mdafs_global"}
    for source, output in lag_map.items():
        if source in frame:
            frame[output] = grouped[source].shift()
    delta_map = {"essdai_total": "delta_essdai_from_previous", "esspri_total": "delta_esspri_from_previous",
                 "sf36_pcs": "delta_sf36_pcs_from_previous", "sf36_mcs": "delta_sf36_mcs_from_previous",
                 "profad_total": "delta_profad_from_previous", "mdafs_global": "delta_mdafs_from_previous"}
    for source, output in delta_map.items():
        if source in frame:
            frame[output] = grouped[source].diff()
    frame["integration_version"] = "v2_clinical_episode"
    frame["integration_run_date"] = date.today().isoformat()
    assert not any(column.startswith("next_") for column in frame)
    return frame


def build_integrated(clinical_spine: pd.DataFrame, pop: pd.DataFrame, labs: pd.DataFrame,
                     overlap: pd.DataFrame, pros: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    spine = validate_spine(clinical_spine)
    integrated, summaries = spine.copy(), []
    for name, source in [("pop", pop), ("labs", labs), ("overlap", overlap), ("pros", pros)]:
        clean, summary, _, _ = validate_source(spine, source, name)
        feature_columns = [column for column in clean if column not in KEYS]
        collisions = set(feature_columns) & (set(integrated) - set(STRUCTURAL_COLUMNS))
        if collisions:
            compared = integrated[KEYS + sorted(collisions)].merge(
                clean[KEYS + sorted(collisions)], on=KEYS, suffixes=("_existing", "_source")
            )
            discordant = [column for column in collisions if _different(
                compared[f"{column}_existing"], compared[f"{column}_source"]
            ).any()]
            if discordant:
                raise AssertionError(f"{name} has conflicting duplicate features: {sorted(discordant)}")
            # Identical duplicated features (notably ESSPRI in Pop and PRO
            # products) are provenance-checked but represented only once.
            clean = clean.drop(columns=sorted(collisions))
        integrated = integrated.merge(clean, on=KEYS, how="left", validate="one_to_one")
        summaries.append(summary)
    integrated = derive_longitudinal(integrated, set(labs) - set(STRUCTURAL_COLUMNS))
    if len(integrated) != len(spine) or integrated.duplicated(KEYS).any():
        raise AssertionError("integrated dataset does not preserve the clinical spine")
    return integrated, summaries


def build_zero_block_qc(integrated: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return zero-block episodes and their aggregate, non-identifying QC."""
    zero_block = integrated.loc[
        integrated["n_integrated_blocks_available"].eq(0)
    ].copy()
    baseline = zero_block["is_clinical_baseline"].fillna(False).astype(bool)
    patient_block_counts = (
        integrated.groupby("patient_id")["n_integrated_blocks_available"]
        .sum(min_count=1)
    )
    summary = {
        "n_zero_block_episodes": int(len(zero_block)),
        "n_zero_block_patients": int(zero_block["patient_id"].nunique()),
        "n_zero_block_clinical_baselines": int(baseline.sum()),
        "n_zero_block_nonbaseline_episodes": int((~baseline).sum()),
        "n_patients_with_all_episodes_zero_block": int(
            patient_block_counts.fillna(0).eq(0).sum()
        ),
        "zero_block_baseline_present": bool(baseline.any()),
    }
    return zero_block, summary


def zero_block_distribution(zero_block: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregate zero-block episodes by a category without exposing episode IDs."""
    output_columns = [column, "n_zero_block_episodes", "pct_zero_block_episodes"]
    if column not in zero_block:
        return pd.DataFrame(columns=output_columns)
    counts = zero_block.groupby(column, dropna=False).size().rename("n_zero_block_episodes")
    distribution = counts.reset_index()
    denominator = len(zero_block)
    distribution["pct_zero_block_episodes"] = (
        distribution["n_zero_block_episodes"].div(denominator).mul(100)
        if denominator else pd.Series(dtype="float64")
    )
    return distribution[output_columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spine", type=Path, default=common.CLINICAL_VISIT_SPINE_PARQUET)
    parser.add_argument("--pop", type=Path, default=common.POP_LONGITUDINAL_PARQUET)
    parser.add_argument("--labs", type=Path, default=common.LABS_EPISODE_WIDE_PARQUET)
    parser.add_argument("--overlap", type=Path, default=common.OVERLAP_LONGITUDINAL_PARQUET)
    parser.add_argument("--pros", type=Path, default=common.PROS_LONGITUDINAL_PARQUET)
    parser.add_argument("--output", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    args = parser.parse_args()
    common.ensure_output_dirs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = ["pop", "labs", "overlap", "pros"]
    spine = pd.read_parquet(args.spine)
    sources = [pd.read_parquet(path) for path in [args.pop, args.labs, args.overlap, args.pros]]
    # Validate separately so QC files can be emitted with stable schemas on successful runs.
    validated_spine = validate_spine(spine)
    details = [validate_source(validated_spine, source, name) for name, source in zip(names, sources)]
    integrated, summaries = build_integrated(spine, *sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    integrated.to_parquet(args.output, index=False)
    integrated.to_csv(args.output.with_suffix(".csv"), index=False)
    qc_dir = common.BLOCKA_QC_DIR / "10_build_integrated_longitudinal_dataset"
    qc_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(qc_dir / "10_integrated_source_summary.csv", index=False)
    mismatch_columns = ["source", *KEYS, "mismatch"]
    discrepancy_columns = ["source", *KEYS, "variable", "spine_value", "source_value"]
    pd.DataFrame([x for detail in details for x in detail[2]], columns=mismatch_columns).to_csv(qc_dir / "10_integrated_key_mismatch_qc.csv", index=False)
    pd.DataFrame([x for detail in details for x in detail[3]], columns=discrepancy_columns).to_csv(qc_dir / "10_integrated_structural_discrepancy_qc.csv", index=False)
    coverage_columns = KEYS + ["has_pop_state", "has_essdai", "has_esspri_observed", "has_lab_measurement", "has_overlap_data", "has_pro_data", "n_integrated_blocks_available"]
    integrated[coverage_columns].to_csv(qc_dir / "10_integrated_longitudinal_coverage.csv", index=False)
    zero_block, zero_summary = build_zero_block_qc(integrated)
    zero_block_distribution(zero_block, "clinical_visit_number").to_csv(
        qc_dir / "10_zero_block_by_clinical_visit_number.csv", index=False
    )
    if "visit_type" in integrated:
        zero_block_distribution(zero_block, "visit_type").to_csv(
            qc_dir / "10_zero_block_by_visit_type.csv", index=False
        )
    zero_block_distribution(zero_block, "is_clinical_baseline").to_csv(
        qc_dir / "10_zero_block_by_baseline_status.csv", index=False
    )
    qc = {"n_rows": len(integrated), "n_patients": int(integrated.patient_id.nunique()),
          "n_unique_patient_episode": int(integrated[KEYS].drop_duplicates().shape[0]),
          "n_duplicate_keys": int(integrated.duplicated(KEYS).sum()), "spine_preserved": True,
          "sources": summaries, "zero_block_qc": zero_summary}
    (qc_dir / "10_integrated_longitudinal_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    if zero_summary["zero_block_baseline_present"]:
        print("WARNING: clinical baseline episodes exist with zero integrated data blocks.\n"
              "Review before Step 11 baseline profiling.")
    print(f"Wrote {len(integrated):,} clinical episodes to {args.output}")


if __name__ == "__main__":
    main()

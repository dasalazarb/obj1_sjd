#!/usr/bin/env python3
"""Filter, validate, and publish the authoritative clinical-episode spine.

This step does not reconstruct, collapse, merge, or split clinical episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import common  # noqa: E402


REQUIRED_COLUMNS = {
    "patient_id",
    "clinical_episode_id",
    "episode_start_date",
    "clinical_anchor_date",
    "episode_end_date",
    "clinical_visit",
    "visit_type",
    "clinical_baseline_episode_id",
    "clinical_baseline_date",
    "is_clinical_baseline",
}
DATE_COLUMNS = (
    "episode_start_date",
    "clinical_anchor_date",
    "episode_end_date",
    "clinical_baseline_date",
)
ORDER_COLUMNS = (
    "patient_id",
    "clinical_anchor_date",
    "episode_start_date",
    "clinical_episode_id",
)
QC_CSV = common.BLOCKA_QC_DIR / "00_episode_spine_qc.csv"
QC_JSON = common.BLOCKA_QC_DIR / "00_episode_spine_qc.json"


def sha256_file(path: Path) -> str:
    """Return the SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_upstream_episode_spine(
    source: Path,
    destination: Path,
    provenance_path: Path,
) -> dict[str, object]:
    """Synchronize and document the authoritative EDA episode-spine output."""
    if not source.exists():
        raise FileNotFoundError(
            f"Authoritative eda_sjd clinical episode spine not found: {source}"
        )
    if not source.is_file():
        raise FileNotFoundError(
            f"Authoritative eda_sjd clinical episode spine is not a file: {source}"
        )

    print(f"[SYNC] Upstream spine:\n       {source}")
    print(f"[SYNC] Local snapshot:\n       {destination}")
    source_hash = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    copy_performed = True
    if destination.exists() and sha256_file(destination) == source_hash:
        copy_performed = False
        print("[SYNC] Local clinical episode spine already matches upstream.")
    else:
        shutil.copy2(source, destination)

    copied_hash = sha256_file(destination)
    if copied_hash != source_hash:
        raise RuntimeError("Clinical episode spine copy failed SHA256 validation.")

    source_stat = source.stat()
    provenance: dict[str, object] = {
        "upstream_repository": "eda_sjd",
        "upstream_pipeline_step": "src/11_build_clinical_episode_spine.py",
        "source_path": str(source),
        "local_snapshot_path": str(destination),
        "sha256": source_hash,
        "source_size_bytes": source_stat.st_size,
        "source_modified_time_utc": datetime.fromtimestamp(
            source_stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "synced_at_utc": datetime.now(timezone.utc).isoformat(),
        "copy_performed": copy_performed,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[SYNC] SHA256:\n       {source_hash}")
    print(f"[SYNC] Copy performed: {copy_performed}")
    return {
        "input_mode": "eda_sjd_auto_sync",
        "upstream_source_path": str(source),
        "local_snapshot_path": str(destination),
        "upstream_sha256": source_hash,
        "copy_performed": copy_performed,
    }


def read_source(path: Path) -> pd.DataFrame:
    """Read a supported episode-spine format without altering its rows."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format {suffix!r}; use .parquet or .csv")


def validate_required_columns(source: pd.DataFrame) -> None:
    """Validate the minimum contract before cohort filtering or derivation."""
    missing = sorted(REQUIRED_COLUMNS.difference(source.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def filter_longitudinal_patients(
    source: pd.DataFrame, id_list: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep source rows whose patient ID occurs in the longitudinal ID list."""
    for label, frame in (("episode spine", source), ("longitudinal ID list", id_list)):
        if "patient_id" not in frame.columns:
            raise ValueError(f"Missing required column patient_id in {label}")

    # Deliberately do not coerce either side: cohort membership is an exact
    # patient_id match against the authoritative list.
    requested_ids = set(id_list["patient_id"].dropna().unique())
    filtered = source[source["patient_id"].isin(requested_ids)].copy()
    output_ids = set(filtered["patient_id"].dropna().unique())

    # Keep the cohort boundary explicit at the point where it is established,
    # before any Step 00 output is constructed or written.
    assert output_ids.issubset(requested_ids)

    metrics = {
        "n_ids_requested": len(requested_ids),
        "n_ids_matched": len(output_ids),
        "n_ids_not_found": len(requested_ids - output_ids),
        "n_patients_after_filter": len(output_ids),
    }
    return filtered, metrics


def id_set(series: pd.Series) -> set[object]:
    """Represent an ID set consistently, including a possible missing ID."""
    return set(series.dropna().tolist()) | ({"<MISSING>"} if series.isna().any() else set())


def build_episode_spines(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool], dict[str, object]]:
    """Validate, order, derive compatibility fields, and make clinical view."""
    validate_required_columns(source)

    episode_spine = source.copy()
    for column in DATE_COLUMNS:
        episode_spine[column] = pd.to_datetime(episode_spine[column], errors="coerce")

    # Compatibility fields are aliases only; the authoritative values are not
    # inferred from ids__visit_date or any other legacy field.
    episode_spine["visit_id"] = episode_spine["clinical_episode_id"]
    episode_spine["visit_date"] = episode_spine["clinical_anchor_date"]
    episode_spine["visit_date_clean"] = episode_spine["clinical_anchor_date"]
    episode_spine["time_since_clinical_baseline_days"] = (
        episode_spine["clinical_anchor_date"]
        - episode_spine["clinical_baseline_date"]
    ).dt.days
    episode_spine["time_since_clinical_baseline_years"] = (
        episode_spine["time_since_clinical_baseline_days"] / 365.25
    )

    episode_spine = episode_spine.sort_values(
        list(ORDER_COLUMNS), kind="stable", na_position="last"
    ).reset_index(drop=True)
    clinical_mask = episode_spine["clinical_visit"].eq(True)  # noqa: E712
    episode_spine["clinical_visit_number"] = pd.Series(
        pd.NA, index=episode_spine.index, dtype="Int64"
    )
    episode_spine.loc[clinical_mask, "clinical_visit_number"] = (
        episode_spine.loc[clinical_mask]
        .groupby("patient_id", sort=False, dropna=False)
        .cumcount()
        .add(1)
        .astype("Int64")
    )
    clinical_spine = episode_spine.loc[clinical_mask].copy()

    baseline_mask = episode_spine["is_clinical_baseline"].eq(True)  # noqa: E712
    baseline_counts = (
        episode_spine.loc[baseline_mask]
        .groupby("patient_id", dropna=False)
        .size()
    )
    duplicate_ids = episode_spine.duplicated(
        ["patient_id", "clinical_episode_id"], keep=False
    )
    source_ids = id_set(source["clinical_episode_id"])
    output_ids = id_set(episode_spine["clinical_episode_id"])
    hard_assertions = {
        "A_episode_count_preserved": len(source) == len(episode_spine),
        "B_episode_id_set_preserved": source_ids == output_ids,
        "C_patient_episode_id_unique": not duplicate_ids.any(),
        "D_visit_id_alias_correct": episode_spine["visit_id"].equals(
            episode_spine["clinical_episode_id"]
        ),
        "E_clinical_baseline_unique_per_patient": not baseline_counts.gt(1).any(),
        "F_all_baselines_are_clinical": not (baseline_mask & ~clinical_mask).any(),
        "G_clinical_spine_contains_only_clinical_visits": clinical_spine[
            "clinical_visit"
        ].eq(True).all(),  # noqa: E712
        "H_clinical_visit_count_preserved": len(clinical_spine)
        == int(source["clinical_visit"].eq(True).sum()),  # noqa: E712
    }
    # Pandas reductions can return numpy.bool_, which the standard JSON encoder
    # cannot serialize.  Normalize every assertion result to a native bool so
    # both QC_JSON and the command-line JSON summary are always writable.
    hard_assertions = {
        name: bool(passed) for name, passed in hard_assertions.items()
    }

    all_patients = set(episode_spine["patient_id"].dropna().tolist())
    baseline_patients = set(
        episode_spine.loc[baseline_mask, "patient_id"].dropna().tolist()
    )
    visit_type_counts = episode_spine["visit_type"].value_counts(dropna=False)
    metrics: dict[str, object] = {
        "n_patients_input": int(source["patient_id"].nunique(dropna=True)),
        "n_episodes_input": len(source),
        "n_patients_episode_spine": len(all_patients),
        "n_episodes_episode_spine": len(episode_spine),
        "n_clinical_visits": len(clinical_spine),
        "n_ambiguous_episodes": int(episode_spine["visit_type"].eq("ambiguous").sum()),
        "n_research_or_procedure_episodes": int(
            episode_spine["visit_type"].eq(
                "research_or_procedure_only_candidate"
            ).sum()
        ),
        "n_duplicate_patient_episode_ids": int(duplicate_ids.sum()),
        "n_patients_with_clinical_baseline": len(baseline_patients),
        "n_patients_without_clinical_baseline": len(all_patients - baseline_patients),
        "n_patients_with_multiple_clinical_baselines": int(baseline_counts.gt(1).sum()),
        "n_missing_clinical_episode_id": int(episode_spine["clinical_episode_id"].isna().sum()),
        "n_missing_patient_id": int(episode_spine["patient_id"].isna().sum()),
        "n_missing_clinical_anchor_date_among_clinical_visits": int(
            episode_spine.loc[clinical_mask, "clinical_anchor_date"].isna().sum()
        ),
        "n_episode_ids_lost": len(source_ids - output_ids),
        "n_episode_ids_added": len(output_ids - source_ids),
        "episode_count_preserved": hard_assertions["A_episode_count_preserved"],
        "episode_id_set_preserved": hard_assertions["B_episode_id_set_preserved"],
    }
    for value, count in visit_type_counts.items():
        label = "<MISSING>" if pd.isna(value) else str(value)
        metrics[f"visit_type_count::{label}"] = int(count)

    failed = [name for name, passed in hard_assertions.items() if not passed]
    if failed:
        raise AssertionError("Hard assertion(s) failed: " + ", ".join(failed))
    return episode_spine, clinical_spine, hard_assertions, metrics


def write_outputs(
    episode_spine: pd.DataFrame,
    clinical_spine: pd.DataFrame,
    hard_assertions: dict[str, bool],
    metrics: dict[str, object],
) -> None:
    """Write both spines and the migration-specific QC artifacts."""
    # Publish the filtered source under stable raw-data names. Scripts that use
    # SOURCE_EPISODE_SPINE therefore consume the selected cohort unchanged.
    episode_spine.to_parquet(common.SOURCE_EPISODE_SPINE, index=False)
    episode_spine.to_csv(common.SOURCE_EPISODE_SPINE_CSV, index=False)
    episode_spine.to_parquet(common.EPISODE_SPINE_PARQUET, index=False)
    episode_spine.to_csv(common.EPISODE_SPINE_CSV, index=False)
    clinical_spine.to_parquet(common.CLINICAL_VISIT_SPINE_PARQUET, index=False)
    clinical_spine.to_csv(common.CLINICAL_VISIT_SPINE_CSV, index=False)
    qc_rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    qc_rows.extend(
        {"metric": f"hard_assertion::{key}", "value": value}
        for key, value in hard_assertions.items()
    )
    pd.DataFrame(qc_rows).to_csv(QC_CSV, index=False)
    QC_JSON.write_text(
        json.dumps(
            {"metrics": metrics, "hard_assertions": hard_assertions}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional explicit episode-spine input. If omitted, Step 00 "
            "synchronizes the canonical spine from eda_sjd automatically."
        ),
    )
    parser.add_argument("--id-list", type=Path, default=common.LONGITUDINAL_ID_LIST)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = parser.parse_args()
    common.ensure_output_dirs()
    if args.input is None:
        sync_metrics = sync_upstream_episode_spine(
            source=common.EDA_SJD_CLINICAL_EPISODE_SPINE,
            destination=common.UNFILTERED_EPISODE_SPINE,
            provenance_path=common.UNFILTERED_EPISODE_SPINE_PROVENANCE,
        )
        input_path = common.UNFILTERED_EPISODE_SPINE
        print("[STEP 00] Reading synchronized episode spine...")
    else:
        input_path = args.input
        sync_metrics = {
            "input_mode": "explicit_cli_override",
            "input_path": str(input_path),
        }
        print(f"[STEP 00] Reading explicit episode-spine input: {input_path}")
    output_paths = (
        common.SOURCE_EPISODE_SPINE,
        common.SOURCE_EPISODE_SPINE_CSV,
        common.EPISODE_SPINE_PARQUET,
        common.EPISODE_SPINE_CSV,
        common.CLINICAL_VISIT_SPINE_PARQUET,
        common.CLINICAL_VISIT_SPINE_CSV,
        QC_CSV,
        QC_JSON,
    )
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Output file(s) already exist: " + ", ".join(existing))

    source = read_source(input_path)
    validate_required_columns(source)
    print("[STEP 00] Filtering longitudinal cohort...")
    id_list = pd.read_csv(args.id_list)
    source, filter_metrics = filter_longitudinal_patients(source, id_list)
    print("[STEP 00] Validating authoritative clinical episodes...")
    episode_spine, clinical_spine, assertions, metrics = build_episode_spines(source)
    metrics = {**sync_metrics, **filter_metrics, **metrics}
    print("[STEP 00] Writing outputs...")
    write_outputs(episode_spine, clinical_spine, assertions, metrics)
    print(json.dumps({"metrics": metrics, "hard_assertions": assertions}, indent=2))


if __name__ == "__main__":
    main()

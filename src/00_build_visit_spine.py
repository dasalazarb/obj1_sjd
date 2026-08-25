#!/usr/bin/env python3
"""Validate and publish the authoritative clinical-episode spine.

This step does not reconstruct, collapse, merge, or split clinical episodes.
"""
from __future__ import annotations

import argparse
import json
import sys
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


def read_source(path: Path) -> pd.DataFrame:
    """Read a supported episode-spine format without altering its rows."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format {suffix!r}; use .parquet or .csv")


def id_set(series: pd.Series) -> set[object]:
    """Represent an ID set consistently, including a possible missing ID."""
    return set(series.dropna().tolist()) | ({"<MISSING>"} if series.isna().any() else set())


def build_episode_spines(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool], dict[str, object]]:
    """Validate, order, derive compatibility fields, and make clinical view."""
    missing = sorted(REQUIRED_COLUMNS.difference(source.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

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
    parser.add_argument("--input", type=Path, default=common.SOURCE_EPISODE_SPINE)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = parser.parse_args()
    common.ensure_output_dirs()
    output_paths = (
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

    source = read_source(args.input)
    episode_spine, clinical_spine, assertions, metrics = build_episode_spines(source)
    write_outputs(episode_spine, clinical_spine, assertions, metrics)
    print(json.dumps({"metrics": metrics, "hard_assertions": assertions}, indent=2))


if __name__ == "__main__":
    main()

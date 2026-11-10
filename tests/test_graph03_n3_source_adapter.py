"""Synthetic-only regression tests for the N3 source candidate adapter."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.studies.graph.stage03.s03_n3_source_adapter import build_candidate


def fixture(tmp_path):
    root = tmp_path
    analytic_dir = root / "data/analytic/10_build_integrated_longitudinal_dataset"
    analytes_dir = root / "data/intermediate/block_A/01_serological_profile"
    analytic_dir.mkdir(parents=True)
    analytes_dir.mkdir(parents=True)
    baseline = pd.DataFrame({
        "patient_id": ["synthetic_one", "synthetic_two"],
        "clinical_episode_id": ["e1", "e2"],
        "is_clinical_baseline": [True, True],
        "lab__ana_status__text": ["positive", "negative"],
    })
    baseline.to_parquet(analytic_dir / "10_integrated_longitudinal_clinical_episode.parquet", index=False)
    context = baseline[["patient_id", "clinical_episode_id"]].copy()
    context["lab__ana_status__measurement_date"] = ["2019-02-01", "2019-03-02"]
    context_path = analytic_dir / "10_integrated_longitudinal_context.parquet"
    context.to_parquet(context_path, index=False)
    selected = baseline[["patient_id", "clinical_episode_id"]].copy()
    selected["lab_id"] = "ana_status"
    selected["selected_lab_date"] = ["2019-02-01", "2019-03-01"]
    selected["selected_unit"] = "synthetic_test_unit"
    selected["unit_conflict"] = [False, False]
    selected["result_conflict"] = [False, False]
    selected["selection_status"] = ["selected_qualitative", "selected_qualitative"]
    selected.to_parquet(analytes_dir / "01_labs_episode_analyte_level.parquet", index=False)
    return root, context_path


def test_never_fabricates_dates_or_assay_or_sample_ids(tmp_path):
    root, context_path = fixture(tmp_path)
    candidate_path = tmp_path / "restricted/candidates.parquet"
    result = build_candidate(root, context_path, candidate_path)
    assert result["status"] == "SOURCE_CANDIDATE_UNAPPROVED"
    rows = pd.read_parquet(candidate_path)
    assert len(rows) == 2
    assert rows.loc[0, "value"] == "positive"
    assert pd.isna(rows.loc[1, "value"])  # date mismatch
    assert rows.loc[1, "date_provenance_status"] == "MISSING_OR_DISCORDANT"
    assert rows["assay_id"].isna().all() and rows["sample_id"].isna().all()
    assert rows["clinical_approval_status"].eq("PENDING_HUMAN_REVIEW").all()
    metadata = json.loads(candidate_path.with_suffix(".provenance.json").read_text())
    assert metadata["eligible_for_inference"] is False
    assert metadata["dates_reused_from_episode_anchors"] is False


def test_existing_candidate_is_immutable(tmp_path):
    root, context_path = fixture(tmp_path)
    output = tmp_path / "restricted/candidates.parquet"
    build_candidate(root, context_path, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        build_candidate(root, context_path, output)
    assert output.read_bytes() == original


def test_conflicting_lab_values_are_not_promoted(tmp_path):
    root, context_path = fixture(tmp_path)
    path = root / "data/intermediate/block_A/01_serological_profile/01_labs_episode_analyte_level.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "result_conflict"] = True
    frame.to_parquet(path, index=False)
    output = tmp_path / "restricted/candidates.parquet"
    build_candidate(root, context_path, output)
    candidates = pd.read_parquet(output)
    assert candidates["value"].isna().all()
    assert candidates.loc[0, "value_provenance_status"] == "CONFLICT_REQUIRES_REVIEW"

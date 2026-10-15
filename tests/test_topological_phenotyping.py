"""Focused tests for the topological phenotyping study."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "src/studies/graph/01_run_topological_phenotyping.py"
SPEC = importlib.util.spec_from_file_location("topological_phenotyping", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("age_column", ["age_dx", "age_at_diagnosis"])
def test_load_diagnosis_age_accepts_block_a_names(tmp_path, age_column):
    path = tmp_path / "baseline.csv"
    pd.DataFrame({"patient_id": ["p1", "p2"], age_column: [42.0, 53.0]}).to_csv(path, index=False)

    result = MODULE.load_age_at_diagnosis(path)

    assert result.columns.tolist() == ["patient_id", "age_dx"]
    assert result.age_dx.tolist() == [42.0, 53.0]


def test_load_diagnosis_age_preserves_existing_alias(tmp_path):
    path = tmp_path / "baseline.csv"
    pd.DataFrame({"patient_id": ["p1"], "age_dx": [42.0],
                  "age_at_diagnosis": [53.0]}).to_csv(path, index=False)

    assert MODULE.load_age_at_diagnosis(path).age_dx.tolist() == [42.0]


@pytest.mark.parametrize("include_empty_age", [False, True])
def test_unavailable_diagnosis_age_is_warned_and_excluded(tmp_path, caplog, include_empty_age):
    path = tmp_path / "baseline.csv"
    baseline = pd.DataFrame({"patient_id": ["p1", "p2"],
                             "demo__age_at_baseline": [50.0, 60.0],
                             "demo__age_at_visit": [51.0, 61.0],
                             "sf36_pcs": [40.0, 50.0],
                             "sf36_mcs": [35.0, 45.0]})
    if include_empty_age:
        baseline["age_dx"] = float("nan")
    baseline.to_csv(path, index=False)

    age = MODULE.load_age_at_diagnosis(path)
    assert age.patient_id.tolist() == ["p1", "p2"]
    assert age.age_dx.isna().all()
    assert "Age at diagnosis unavailable" in caplog.text

    baseline = baseline.drop(columns=["age_dx"], errors="ignore").merge(age, on="patient_id")
    manifest, values, _, _ = MODULE.build_feature_manifest(baseline, pd.DataFrame(), 0.5)
    row = manifest.set_index("feature").loc["age_dx"]
    assert not row.included
    assert row.n_nonmissing_baseline == 0
    assert row.exclusion_reason == "no_observed_baseline_data"
    matrix, _, _ = MODULE.build_baseline_matrix(values)
    assert matrix.columns.tolist() == ["sf36_pcs", "sf36_mcs"]


def test_diagnosis_age_loader_rejects_duplicate_patients_without_age(tmp_path):
    path = tmp_path / "baseline.csv"
    pd.DataFrame({"patient_id": ["p1", "p1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one row per patient"):
        MODULE.load_age_at_diagnosis(path)


def test_diagnosis_age_loader_requires_patient_id(tmp_path):
    path = tmp_path / "baseline.csv"
    pd.DataFrame({"age_dx": [42.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="patient_id"):
        MODULE.load_age_at_diagnosis(path)


def test_diagnosis_age_loader_requires_baseline_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Baseline metrics file not found"):
        MODULE.load_age_at_diagnosis(tmp_path / "missing.csv")


def test_branch_characterization_calculates_boolean_quartiles():
    baseline = pd.DataFrame({
        "patient_id": ["1", "2", "3", "4"],
        "boolean_feature": [False, False, True, True],
    })
    membership = pd.DataFrame({
        "patient_id": ["1", "2", "3", "4"],
        "mapper_covered": [True, True, True, True],
        "hard_branch": [0, 0, 0, 0],
    })

    result = MODULE.branch_characterization(
        baseline, membership, ["boolean_feature"], {})

    row = result.iloc[0]
    assert row["n"] == 4
    assert row["median"] == 0.5
    assert row["q1"] == 0.0
    assert row["q3"] == 1.0
    assert row["mean"] == 0.5

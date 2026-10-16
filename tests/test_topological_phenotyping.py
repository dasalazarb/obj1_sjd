"""Focused tests for the topological phenotyping study."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from graph_contract_fixtures import empty_candidates


SCRIPT = Path(__file__).parents[1] / "src/studies/graph/01_run_topological_phenotyping.py"
SPEC = importlib.util.spec_from_file_location("topological_phenotyping", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_public_features_use_canonical_sources_and_keep_unknown_overlap_missing():
    baseline = pd.DataFrame({
        "pro__sf36_pcs": [35., 45., 55., 65.], "sf36_pcs": [0.] * 4,
        "pro__sf36_mcs": [36., 46., 56., 66.],
        "pro__profad_total": [1., 2., 3., 4.],
        "pro__mdafs_global": [4., 3., 2., 1.],
        "ovl__status": ["overlap", "glandular_only", "unclassifiable", None],
        "overlap_baseline": [1.] * 4,
        "pop__status": ["Pop1", "Pop2", "Pop3", "Pop1"],
        "essdai__total": [1., 2., 3., 4.], "esspri__total": [2., 3., 4., 5.],
    })

    manifest, values, _, _ = MODULE._build_legacy_feature_manifest(baseline, empty_candidates(), .5)

    assert set(values) == {*MODULE.PRO_FEATURES, "overlap_baseline"}
    assert values["sf36_pcs"].tolist() == [35., 45., 55., 65.]
    assert values["overlap_baseline"].iloc[:2].tolist() == [1., 0.]
    assert values["overlap_baseline"].iloc[2:].isna().all()
    sources = manifest.set_index("feature").source_column
    assert sources["sf36_pcs"] == "pro__sf36_pcs"
    assert sources["overlap_baseline"] == "ovl__status"


def test_longitudinal_models_resolve_public_outcomes_and_covariates():
    baseline = pd.DataFrame({"patient_id": ["p1", "p2", "p3", "p4"],
                             "demo__age_at_visit": [40., 50., 60., 70.],
                             "demo__sex": ["Female", "Male", "Female", "Male"]})
    master = pd.DataFrame({"patient_id": baseline.patient_id,
                           "time_since_clinical_baseline_years": [0., 0., 0., 0.],
                           "essdai__total": [1., 2., 3., 4.],
                           "esspri__total": [2., 3., 4., 5.],
                           "pro__sf36_pcs": [35., 45., 55., 65.],
                           "pro__profad_total": [1., 2., 3., 4.],
                           "pro__mdafs_global": [4., 3., 2., 1.]})
    membership = pd.DataFrame({"patient_id": baseline.patient_id,
                               "mapper_covered": True, "community_membership_tie": False,
                               "community_supported": True, "hard_community": [0, 0, 1, 1]})

    models, longitudinal, _ = MODULE.fit_longitudinal_models(master, baseline, membership)

    assert set(models.model_status) == {"insufficient_information"}
    assert models.n_observations.tolist() == [4] * len(MODULE.OUTCOMES)
    assert longitudinal.baseline_age.tolist() == [40., 50., 60., 70.]
    assert longitudinal.baseline_sex.tolist() == ["Female", "Male", "Female", "Male"]
    assert longitudinal.essdai_total.tolist() == [1., 2., 3., 4.]
    assert all("baseline_age" in formula and "baseline_sex" in formula for formula in models.model)


def test_overlap_feature_respects_evaluability():
    baseline = pd.DataFrame({
        "ovl__status": ["overlap", "glandular_only", "extraglandular_only", "neither", "overlap", "overlap"],
        "ovl__evaluable": pd.Series([True, True, True, True, False, None], dtype="boolean"),
    })

    _, values, _, _ = MODULE._build_legacy_feature_manifest(baseline, empty_candidates(), .5)

    assert values["overlap_baseline"].iloc[:4].tolist() == [1., 0., 0., 0.]
    assert values["overlap_baseline"].iloc[4:].isna().all()


def test_run_with_public_episode_schema(tmp_path, monkeypatch, caplog):
    rng = np.random.default_rng(17)
    n = 24
    master = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "clinical_episode_id": [f"e{i}" for i in range(n)],
        "clinical_anchor_date": pd.Timestamp("2020-01-01"),
        "clinical_visit_number": 1, "clinical_visit": True, "is_clinical_baseline": True,
        "time_since_clinical_baseline_years": 0.,
        "pop__status": ["Pop1", "Pop2", "Pop3"] * (n // 3),
        "essdai__total": rng.uniform(0, 10, n), "esspri__total": rng.uniform(0, 10, n),
        "ovl__status": ["overlap", "glandular_only", "neither"] * (n // 3),
        "demo__age_at_visit": rng.uniform(30, 70, n), "demo__sex": "Female",
        **{f"pro__{feature}": rng.uniform(1, 10, n) for feature in MODULE.PRO_FEATURES},
        "lab__test__value": rng.uniform(1, 10, n),
    })
    integrated = tmp_path / "integrated.parquet"
    master.to_parquet(integrated, index=False)
    original_parquet = integrated.read_bytes()
    candidates = tmp_path / "candidates.csv"
    import importlib.util
    spec = importlib.util.spec_from_file_location("pharma_for_graph01", SCRIPT.parents[1] / "pharma/00_profile_labs.py")
    pharma = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pharma)
    pharma.run(pharma.parse_args(["--integrated", str(integrated), "--output", str(tmp_path / "profile.csv")]))
    candidates = tmp_path / pharma.GRAPH_CANDIDATES_NAME
    metrics = tmp_path / "baseline.csv"
    master[["patient_id"]].to_csv(metrics, index=False)
    config = MODULE.load_config(SCRIPT.with_name("config.yaml"))
    config["bootstrap"]["replicates"] = 2
    config["permutation"]["ari_replicates"] = 5
    config["persistent_homology"]["enabled"] = False
    config["mapper"].update(n_cubes=2, min_samples=2, minimum_node_support=2)
    config["community_detection"]["minimum_patients"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    dirs = {name: tmp_path / name for name in ("tables", "logs", "figures")}
    for path in dirs.values():
        path.mkdir()
    monkeypatch.setattr(MODULE, "create_study_dirs", lambda _: dirs)

    MODULE.run(MODULE.parse_args([
        "--integrated", str(integrated), "--graph-lab-candidates", str(candidates),
        "--baseline-metrics", str(metrics), "--config", str(config_path),
        "--feature-policy", "legacy_with_overlap"]))

    embedding = pd.read_csv(dirs["tables"] / "01_graph_patient_embedding.csv")
    assert embedding.baseline_pop.tolist() == master["pop__status"].tolist()
    summary = json.loads((dirs["tables"] / "01_graph_results_summary.json").read_text())
    assert summary["embedding_feature_families"] == {
        "lab": 1, "pro": 4, "clinical": 1, "ext": 0, "sero": 0}
    assert summary["n_embedding_features"] == 6
    assert len(list(dirs["figures"].glob("*.png"))) == 5
    assert integrated.read_bytes() == original_parquet
    expected_warnings = [record.getMessage() for record in caplog.records
                         if "Expected clinically important approved lab" in record.getMessage()]
    assert expected_warnings == []


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
    manifest, values, _, _ = MODULE._build_legacy_feature_manifest(baseline, empty_candidates(), 0.5)
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


def test_diagnosis_age_loader_allows_missing_baseline_file(tmp_path):
    assert MODULE.load_age_at_diagnosis(tmp_path / "missing.csv").empty


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

"""T01–T20: synthetic-only coordinated Graph ingestion and release safeguards."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from graph_contract_fixtures import candidate, empty_candidates, write_bundle
from src.studies.graph import _lab_contract as LAB

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("graph_contract_sensitivity", ROOT / "src/studies/graph/02_graph_representation_sensitivity.py")
G2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G2)
G1 = G2.CANON


def episodes(n=100):
    baseline = pd.DataFrame(dict(patient_id=[f"SYNTHETIC_PATIENT_{i}" for i in range(n)],
        clinical_episode_id=[f"base_{i}" for i in range(n)], clinical_anchor_date=pd.Timestamp("2020-01-01"),
        clinical_visit_number=1, clinical_visit=True, is_clinical_baseline=True,
        **{"pop__status": ["Pop1", "Pop2"] * (n//2), "essdai__total": 1., "esspri__total": 2.,
           "ovl__status": ["overlap", "neither"] * (n//2),
           "lab__glucose__value": np.arange(n, dtype=float),
           "lab__ana__text": ["Positive ", " NEGATIVE"] * (n//2),
           "pro__sf36_pcs": np.random.default_rng(21).normal(size=n),
           "pro__sf36_mcs": np.random.default_rng(23).normal(size=n),
           "sero__hla__patient_consensus_value": "allele",
           "sero__ana__ever_positive_through_episode": True,
           "lab__ana__reference_status": "positive"}))
    followup = baseline.copy()
    followup["clinical_episode_id"] = "follow_" + followup.clinical_episode_id
    followup["clinical_anchor_date"] = pd.Timestamp("2021-01-01")
    followup["clinical_visit_number"] = 2
    followup["is_clinical_baseline"] = False
    return pd.concat([baseline, followup], ignore_index=True)


@pytest.fixture
def bundle(tmp_path):
    master = episodes()
    integrated = tmp_path / "integrated.parquet"
    master.to_parquet(integrated)
    cp = tmp_path / "00_pharma_graph_lab_candidates.csv"
    pd.DataFrame([candidate(), candidate("lab__ana", "include_categorical")]).to_csv(cp, index=False)
    rp, paths, summary, release = write_bundle(integrated, cp)
    return master, cp, rp, paths, summary, release


def test_t01_v1_and_empty_headerless_fail_closed():
    with pytest.raises(LAB.LabContractError, match="CONTRACT_MISMATCH"):
        G1.build_feature_manifest(episodes(), pd.DataFrame([{"lab": "glucose"}]), .7)
    with pytest.raises(LAB.LabContractError, match="CONTRACT_MISMATCH"):
        LAB.validate_candidate_rows(episodes(), pd.DataFrame())


@pytest.mark.parametrize("changes", [dict(clinically_approved="pending"), dict(clinical_review_required=True),
    dict(clinical_review_required="pending"), dict(clinical_review_required=np.nan), dict(graph_candidate_status=np.nan),
    dict(graph_encoding_hint="categorical_one_hot")])
def test_t02_strict_approval_and_critical_states(changes):
    with pytest.raises(LAB.LabContractError):
        LAB.validate_candidate_rows(episodes(), pd.DataFrame([candidate(**changes)]))


def test_t03_identity_is_content_hash_not_path_or_mtime(bundle, tmp_path):
    master, cp, _, paths, summary, _ = bundle
    copied = tmp_path / "identical.parquet"
    copied.write_bytes(paths["integrated"].read_bytes())
    LAB.validate_graph_lab_contract(master, pd.read_csv(cp), summary, integrated_path=copied, candidates_path=cp)
    copied.write_bytes(copied.read_bytes() + b"different")
    with pytest.raises(LAB.LabContractError, match="STALE_PHARMA00_OUTPUT"):
        LAB.validate_graph_lab_contract(master, pd.read_csv(cp), summary, integrated_path=copied, candidates_path=cp)


@pytest.mark.parametrize("changes", [dict(source_column="lab__ana__reference_status"),
    dict(source_column="sero__ana__ever_positive_through_episode"), dict(source_column="lab__ana__text"),
    dict(primary_result_source="value_column"), dict(value_column="lab__other__value")])
def test_t04_unauthorized_sources(changes):
    with pytest.raises(LAB.LabContractError, match="CONTRACT_MISMATCH"):
        LAB.validate_candidate_rows(episodes(), pd.DataFrame([candidate(**changes)]))


def test_t04_missing_source():
    with pytest.raises(LAB.LabContractError, match="MISSING_SOURCE_COLUMN"):
        LAB.validate_candidate_rows(episodes(), pd.DataFrame([candidate("lab__absent")]))


@pytest.mark.parametrize("bad", [np.inf, -np.inf, "<5", "SEE NOTE"])
def test_t04_no_nonfinite_or_silent_numeric_cast(bad):
    frame = episodes()
    frame["lab__glucose__value"] = frame["lab__glucose__value"].astype(object)
    frame.loc[0, "lab__glucose__value"] = bad
    with pytest.raises(LAB.LabContractError, match="CONTRACT_MISMATCH"):
        G1.build_feature_manifest(G1.select_baseline(frame), pd.DataFrame([candidate()]), .7)


def test_t05_review_rows_and_full_profile_are_audit_only():
    baseline = G1.select_baseline(episodes())
    candidates = pd.DataFrame([candidate(), candidate("lab__pending", "review_before_graph")])
    _, values, _, audit = G1.build_feature_manifest(baseline, candidates, .7)
    assert not any("pending" in name for name in values)
    assert audit.loc[audit.lab.eq("lab__pending"), "exclusion_reason"].item() == "review_before_graph_not_approved"


def test_t06_empty_v2_requires_summary_version(bundle):
    master, cp, _, paths, summary, _ = bundle
    empty = empty_candidates()
    _, report = LAB.validate_graph_lab_contract(master, empty, summary, integrated_path=paths["integrated"], candidates_path=cp)
    assert report["n_approved_labs"] == 0
    with pytest.raises(LAB.LabContractError, match="NO_APPROVED_LABS"):
        LAB.validate_release(None, empty, summary, paths)
    with pytest.raises(G2.MappingError, match="NO_LABS_FOR_S4"):
        G2.validate_s4_mapping({}, {}, pd.DataFrame(columns=["lab"]), {}, require_all=True)
    summary["profile_contract_version"] = "v1"
    with pytest.raises(LAB.LabContractError, match="CONTRACT_MISMATCH"):
        LAB.validate_graph_lab_contract(master, empty, summary, integrated_path=paths["integrated"], candidates_path=cp)


@pytest.mark.parametrize("changes", [dict(status="pending_review"), dict(reviewer=None), dict(review_evidence=""),
    dict(pharma_candidates_sha256="incorrect"), dict(approved_lab_stems=["lab__glucose", "lab__pending"]),
    dict(acknowledged_release_recommendation="READY"), dict(reviewed_downstream_impact=False),
    dict(reviewed_expected_lab_audit=False), dict(review_date="not_a_date"), dict(analysis_scope="other")])
def test_t07_release_cannot_override_assay_eligibility(bundle, changes):
    _, cp, _, paths, summary, release = bundle
    original = paths["pharma_summary"].read_bytes()
    release.update(changes)
    with pytest.raises(LAB.LabContractError, match="INCOMPLETE_RELEASE_REVIEW"):
        LAB.validate_release(release, pd.read_csv(cp), summary, paths)
    assert paths["pharma_summary"].read_bytes() == original


def test_t07_valid_review_acknowledges_hold_and_schema_drift(bundle):
    _, cp, _, paths, summary, release = bundle
    LAB.validate_release(release, pd.read_csv(cp), summary, paths)
    summary["schema_drift"] = {"n_columns": {"observed": 17, "expected": 485}}
    release["schema_drift_reviewed"] = False
    with pytest.raises(LAB.LabContractError, match="SCHEMA_DRIFT_REVIEW"):
        LAB.validate_release(release, pd.read_csv(cp), summary, paths)
    release["schema_drift_reviewed"] = True
    LAB.validate_release(release, pd.read_csv(cp), summary, paths)


@pytest.mark.parametrize("observed,included", [(69, False), (70, True)])
def test_t08_symmetric_coverage(observed, included):
    baseline = G1.select_baseline(episodes())
    baseline.loc[observed:, ["lab__glucose__value", "lab__ana__text"]] = np.nan
    candidates = pd.DataFrame([candidate(pct_baseline_any_result=1.), candidate("lab__ana", "include_categorical", pct_baseline_any_result=1.)])
    manifest, values, _, audit = G1.build_feature_manifest(baseline, candidates, .7)
    assert ("lab__glucose__value" in values) == included
    assert ("lab__ana__binary" in values) == included
    assert audit.baseline_coverage.eq(observed / 100).all()
    if not included:
        assert set(audit.exclusion_reason) == {"below_minimum_baseline_coverage"}
    assert "categorical_coverage_exception" not in set(manifest.exclusion_reason)


@pytest.mark.parametrize("bad", ["SEE NOTE", "TRACE"])
def test_t09_bad_category_fails_without_exporting_free_text(bad):
    frame = episodes()
    frame.loc[0, "lab__ana__text"] = bad
    with pytest.raises(LAB.LabContractError) as error:
        LAB.validate_candidate_rows(frame, pd.DataFrame([candidate("lab__ana", "include_categorical")]))
    assert bad not in str(error.value)


def test_t09_unknown_is_missing_case_and_spaces_are_safe():
    baseline = G1.select_baseline(episodes())
    baseline.loc[0, "lab__ana__text"] = " UNKNOWN "
    _, values, _, _ = G1.build_feature_manifest(baseline, pd.DataFrame([candidate("lab__ana", "include_categorical")]), .7)
    assert np.isnan(values["lab__ana__binary"].iloc[0])
    assert values["lab__ana__binary"].iloc[1:4].tolist() == [0., 1., 0.]


def test_t10_baseline_coverage_uses_unique_patients():
    frame = episodes()
    baseline = G1.select_baseline(frame)
    _, _, _, audit = G1.build_feature_manifest(baseline, pd.DataFrame([candidate()]), .7)
    assert audit.baseline_nonmissing.item() == 100
    with pytest.raises(LAB.LabContractError, match="one row per patient"):
        G1.build_feature_manifest(frame, pd.DataFrame([candidate()]), .7)


def test_t11_t12_t13_policies_preserve_patients_and_exclude_annotations():
    baseline = G1.select_baseline(episodes())
    baseline["age_dx"] = np.arange(len(baseline)) + 20
    candidates = pd.DataFrame([candidate(), candidate("lab__ana", "include_categorical")])
    legacy, va, _, _ = G1.build_feature_manifest(baseline, candidates, .7)
    alternate, vb, _, _ = G1.build_feature_manifest(baseline, candidates, .7, "without_overlap")
    pd.testing.assert_frame_equal(pd.DataFrame(va).drop(columns="overlap_baseline"), pd.DataFrame(vb))
    assert alternate.set_index("feature").loc["overlap_baseline", "exclusion_reason"] == "excluded_by_feature_policy_without_overlap"
    assert legacy.set_index("feature").loc["age_dx", "temporal_lineage_status"] == "temporal_lineage_unverified"
    assert not any(name.startswith(("sero__", "essdai", "esspri", "pop__")) or name.endswith("reference_status") for name in va)


def test_t14_t15_s0_and_s3_s4_raw_matrices_match():
    baseline = G1.select_baseline(episodes())
    candidates = pd.DataFrame([candidate(), candidate("lab__ana", "include_categorical")])
    cfg = G1.load_config(ROOT / "src/studies/graph/config_sensitivity.yaml")
    for policy in LAB.FEATURE_POLICIES:
        manifest, canonical, _, _ = G1.build_feature_manifest(baseline, candidates, .7, policy)
        full_manifest, universe, _, _ = G1.build_feature_manifest(baseline, candidates, 0., policy)
        families = {r.feature: G2.family_name(r.family) for r in full_manifest.loc[full_manifest.included].itertuples()}
        s0, _, _ = G2.select_features(canonical, families, "S0", cfg, manifest)
        pd.testing.assert_frame_equal(pd.DataFrame(canonical), pd.DataFrame(s0))
        s3, _, _ = G2.select_features(universe, families, "S3", cfg, full_manifest)
        s4, _, _ = G2.select_features(universe, families, "S4", cfg, full_manifest)
        pd.testing.assert_frame_equal(pd.DataFrame(s3), pd.DataFrame(s4))
        assert LAB.matrix_fingerprint(pd.DataFrame(s3)) == LAB.matrix_fingerprint(pd.DataFrame(s4))


def test_t16_bootstrap_reentry_universe_requires_mapping():
    baseline = G1.select_baseline(episodes())
    baseline.loc[10:, "lab__glucose__value"] = np.nan
    manifest, universe, _, _ = G1.build_feature_manifest(baseline, pd.DataFrame([candidate()]), 0.)
    families = {name: "LAB" if name.startswith("lab__") else "PRO" for name in universe}
    with pytest.raises(G2.MappingError, match="LAB_GROUP_MAPPING_ERROR"):
        G2.validate_s4_mapping(universe, families, pd.DataFrame(columns=["lab", "mapping_status"]),
                               {"lab__glucose__value": "glucose"}, require_all=True)


def test_t17_s4_no_lab_and_zero_signal_are_non_estimable():
    raw = pd.DataFrame({"pro": [1., 2., 3.]})
    with pytest.raises(G2.BlockNotEstimable, match="NO_LABS_FOR_S4"):
        G2.prepare_matrix(raw, {"pro": "PRO"}, True, True, {}, {}, "S4")
    raw["lab__glucose__value"] = 1.
    with pytest.raises(G2.BlockNotEstimable, match="S4_LAB_GROUP_NOT_ESTIMABLE"):
        G2.prepare_matrix(raw, {"pro": "PRO", "lab__glucose__value": "LAB"}, True, True,
                          {"glucose": "synthetic_group"}, {"lab__glucose__value": "glucose"}, "S4")


def test_t18_selection_recomputed_in_subsample_and_pending_never_reenters():
    baseline = G1.select_baseline(episodes())
    baseline.loc[60:, "lab__glucose__value"] = np.nan
    manifest, universe, _, _ = G1.build_feature_manifest(baseline, pd.DataFrame([
        candidate(), candidate("lab__pending", "review_before_graph")]), 0.)
    cfg = G1.load_config(ROOT / "src/studies/graph/config_sensitivity.yaml")
    families = {name: "LAB" if name.startswith("lab__") else "PRO" for name in universe}
    full, _ = G2.coverage_filter(universe, families, cfg)
    sampled, _ = G2.coverage_filter({name: values.iloc[:60] for name, values in universe.items()}, families, cfg)
    assert "lab__glucose__value" not in full and "lab__glucose__value" in sampled
    assert not any("pending" in name for name in sampled)


def test_t19_profile_exclusions_do_not_become_candidates():
    baseline = G1.select_baseline(episodes())
    baseline["lab__descriptive__text"] = "REDACTED"
    baseline["lab__unregistered__value"] = 1.
    candidates = pd.DataFrame([candidate()])
    manifest, universe, _, _ = G1.build_feature_manifest(baseline, candidates, 0.)
    families = {r.feature: G2.family_name(r.family) for r in manifest.loc[manifest.included].itertuples()}
    config = G1.load_config(ROOT / "src/studies/graph/config_sensitivity.yaml")
    selected, coverage, redundancy = G2.select_features(universe, families, "S3", config, manifest)
    rep = G2.Representation(pd.DataFrame(selected), families, np.empty((0,0)), None, list(selected), pd.DataFrame())
    profile = pd.DataFrame([candidate(), dict(lab="lab__descriptive", graph_candidate_status="descriptive_only")])
    mapping = pd.DataFrame([dict(lab="glucose", mapping_status="MAPPED")])
    audit = G2.feature_selection_audit(baseline, baseline, candidates, manifest, universe, coverage, redundancy, rep, mapping, "S3", profile).set_index("feature")
    assert audit.loc["lab__descriptive__text", "exclusion_reason"] == "profiled_but_not_exported_as_graph_candidate"
    assert audit.loc["lab__unregistered__value", "exclusion_reason"] == "not_catalogued_by_pharma00"
    assert audit.loc["lab__glucose__value", "exclusion_reason"] == "selected_in_pca"
    assert not audit.loc["lab__descriptive__text", "present_in_pca"]


@pytest.mark.parametrize("module,prefix", [(G1, "01_graph"), (G2, "02_graph")])
@pytest.mark.parametrize("policy", LAB.FEATURE_POLICIES)
def test_t06_t20_preflight_writes_aggregate_audit_without_fitting(bundle, tmp_path, monkeypatch, module, prefix, policy):
    master, cp, _, paths, _, _ = bundle
    metrics = tmp_path / "metrics.csv"
    master[["patient_id"]].drop_duplicates().to_csv(metrics, index=False)
    mapping = tmp_path / "map.csv"
    pd.DataFrame([dict(lab=lab, s4_group="SYNTHETIC", s4_subgroup="", mapping_status="MAPPED") for lab in ("glucose", "ana")]).to_csv(mapping, index=False)
    created = []
    def create_dirs(name):
        created.append(name)
        dirs = {key: tmp_path / name / key for key in ("tables", "figures", "logs")}
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        return dirs
    monkeypatch.setattr(module, "create_study_dirs", create_dirs)
    monkeypatch.setattr(module, "fit_embedding" if module is G2 else "fit_patient_embedding", lambda *a, **k: pytest.fail("model fit during preflight"))
    argv = ["--integrated", str(paths["integrated"]), "--graph-lab-candidates", str(cp),
            "--baseline-metrics", str(metrics), "--preflight-only", "--feature-policy", policy, "--run-id", "synthetic"]
    if module is G2:
        argv += ["--lab-group-map", str(mapping)]
    args = module.parse_args(argv)
    module.run(args)
    folder = tmp_path / created[0] / "tables"
    report = json.loads((folder / f"{prefix}_preflight.json").read_text())
    assert report["status"] == "BLOCKED"  # Production template remains pending.
    assert "INCOMPLETE_RELEASE_REVIEW" in {x["code"] for x in report["blockers"]}
    assert report["feature_policy"] == policy
    assert policy in str(folder)
    assert report["n_lab_features_after_encoding"] == 2
    assert not list(folder.glob("*patient*"))
    assert "SYNTHETIC_PATIENT_" not in (folder / f"{prefix}_preflight.json").read_text()
    with pytest.raises(LAB.LabContractError, match="already exists"):
        module.run(args)


def test_duplicate_stems_and_strict_false():
    LAB.validate_candidate_rows(episodes(), pd.DataFrame([candidate(clinical_review_required="False")]))
    with pytest.raises(LAB.LabContractError, match="unique canonical"):
        LAB.validate_candidate_rows(episodes(), pd.DataFrame([candidate(), candidate()]))


@pytest.mark.parametrize("module,prefix", [(G1, "01_graph"), (G2, "02_graph")])
def test_standard_run_blocks_pending_release_and_empty_labs(bundle, tmp_path, monkeypatch, module, prefix):
    master, cp, _, paths, _, _ = bundle
    master[["patient_id"]].drop_duplicates().to_csv(tmp_path / "metrics.csv", index=False)
    mapping = tmp_path / "map.csv"
    pd.DataFrame([dict(lab=lab, s4_group="SYNTHETIC", s4_subgroup="", mapping_status="MAPPED")
                  for lab in ("glucose", "ana")]).to_csv(mapping, index=False)
    def create_dirs(name):
        dirs = {key: tmp_path / name / key for key in ("tables", "figures", "logs")}
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        return dirs
    monkeypatch.setattr(module, "create_study_dirs", create_dirs)
    monkeypatch.setattr(module, "fit_embedding" if module is G2 else "fit_patient_embedding",
                        lambda *a, **k: pytest.fail("scientific inference started while blocked"))
    args = module.parse_args(["--integrated", str(paths["integrated"]), "--graph-lab-candidates", str(cp),
                              "--baseline-metrics", str(tmp_path / "metrics.csv")])
    if module is G2:
        args.lab_group_map = mapping
    with pytest.raises(LAB.LabContractError, match="INCOMPLETE_RELEASE_REVIEW"):
        module.run(args)
    empty_candidates().to_csv(cp, index=False)
    write_bundle(paths["integrated"], cp, approved=False)
    args.preflight_only = True
    module.run(args)
    reports = [json.loads(path.read_text()) for path in tmp_path.glob(f"graph/**/{prefix}_preflight.json")]
    empty_report = next(report for report in reports if report["n_approved_labs"] == 0)
    assert empty_report["status"] == "BLOCKED"
    assert "NO_APPROVED_LABS" in {item["code"] for item in empty_report["blockers"]}
    if module is G2:
        assert "NO_LABS_FOR_S4" in {item["code"] for item in empty_report["blockers"]}


def test_canonical_config_mismatch_is_contract_error():
    cfg = G1.load_config(ROOT / "src/studies/graph/config_sensitivity.yaml")
    canonical = G1.load_config(ROOT / "src/studies/graph/config.yaml")
    args = G2.parse_args([])
    G2.validate_canonical_config(cfg, canonical, args)
    cfg["mapper"]["n_cubes"] += 1
    with pytest.raises(LAB.LabContractError, match="S0 Mapper"):
        G2.validate_canonical_config(cfg, canonical, args)


def test_fixed_contract_errors_are_never_bootstrap_failures(monkeypatch):
    values = {"lab__glucose__value": pd.Series(range(10)), "pro": pd.Series(range(10))}
    def fail(*args, **kwargs):
        raise LAB.LabContractError("CONTRACT_MISMATCH", "synthetic fixed error")
    monkeypatch.setattr(G2, "select_features", fail)
    cfg = G1.load_config(ROOT / "src/studies/graph/config_sensitivity.yaml")
    with pytest.raises(LAB.LabContractError, match="CONTRACT_MISMATCH"):
        G2.bootstrap_stability(values, {"lab__glucose__value": "LAB", "pro": "PRO"}, pd.DataFrame(), "S4", [{0, 1}], cfg,
                               {"glucose": "SYNTHETIC"}, {"lab__glucose__value": "glucose"})

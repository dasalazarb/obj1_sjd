"""Synthetic review fixtures; no patient data or scientific approval in CI."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import yaml

from src.studies.longitudinal_graph import review_qc
from src.studies.longitudinal_graph.graph_export import graph_views
from src.studies.longitudinal_graph.reference import (inventory, new_bundle, safe_id, seal_reference, sha256, verify_reference)
from src.studies.longitudinal_graph.review_runner import compare_reference
from src.studies.longitudinal_graph.scenarios import (HSCRP, TrainingPreprocessor, candidate_registry, negative_patient_blocks, observed_pruning)
from src.studies.longitudinal_graph.stability import bootstrap_mapper, patient_draw
from src.studies.longitudinal_graph.temporal_gate import require_temporal_approval

ROOT = Path(__file__).parents[1]
def load(name):
    path = ROOT / "src/studies/longitudinal_graph" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

MAP = load("02_run_longitudinal_mapper.py")
FLOW = load("03_analyze_temporal_flow.py")


def config():
    value = MAP.load_config(ROOT / "src/studies/longitudinal_graph/config.yaml")
    value["mapper_validation"].update(bootstrap_replicates=2, admin_permutations=3)
    value["methodology_review"] = {"coverage_bootstrap_replicates": 2, "negative_control_replicates": 1,
                                    "negative_control_bootstrap_replicates": 1, "hscrp_unit_status": "verified_consistent",
                                    "hscrp_prior_transformation": "identity"}
    return value


def source(n=80):
    rng = np.random.default_rng(182)
    raw = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"lab__fixture_{i}__value" for i in range(6)])
    raw[HSCRP] = rng.lognormal(size=n)
    raw["pro__fixture_total"] = rng.normal(size=n)
    raw.insert(0, "clinical_episode_id", [f"v{i}" for i in range(n)])
    raw.insert(0, "patient_id", [f"fixture_{i // 2}" for i in range(n)])
    raw.loc[::9, "lab__fixture_0__value"] = np.nan
    state = raw.copy()
    state.iloc[:, 2:] = state.iloc[:, 2:].fillna(state.iloc[:, 2:].median())
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n) % 2 * 30, unit="D")
    metadata = raw[MAP.KEYS].assign(clinical_anchor_date=dates, visit_type="routine", protocol="synthetic", era="synthetic")
    intervals = pd.DataFrame({"patient_id": raw.patient_id.iloc[::2].tolist(),
        "from_clinical_episode_id": raw.clinical_episode_id.iloc[::2].tolist(),
        "to_clinical_episode_id": raw.clinical_episode_id.iloc[1::2].tolist(), "interval_days": 30, "interval_years": 30 / 365.25})
    return raw, state, metadata, intervals


def freeze_fixture(tmp_path, state, cfg):
    """A complete, freshly computed SYNTHETIC reference, never the supplied S3 run."""
    features, _ = MAP.reduce_redundancy(state.iloc[:, 2:])
    scenario = MAP.run_mapper_scenario(state, features, cfg, family_balance=True, scenario_name="S3-primary")
    root = tmp_path / "history" / "synthetic-reference"
    root.mkdir(parents=True)
    (root / "02_longitudinal_mapper_summary.json").write_text(json.dumps(scenario["summary"]))
    for stem, key in (("mapper_nodes", "node_table"), ("mapper_macrostate_summary", "macrostate_table"), ("mapper_topological_edges", "topological_edges")):
        scenario[key].to_csv(root / f"02_{stem}.csv", index=False)
    scenario["visit_membership"].to_parquet(root / "02_visit_mapper_membership.parquet", index=False)
    rows = []
    for node, indices in scenario["graph"]["nodes"].items():
        for i in indices:
            rows.append({**state.iloc[i][MAP.KEYS].to_dict(), "node_id": node,
                         "macrostate_id": scenario["node_to_macrostate"].get(node)})
    pd.DataFrame(rows).to_parquet(root / "02_mapper_node_membership.parquet", index=False)
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    return root, scenario


def test_s3_primary_readonly_integrity(tmp_path):
    root = tmp_path / "history/ref"
    root.mkdir(parents=True)
    (root / "02_longitudinal_mapper_summary.json").write_text('{"scenario":"S3-primary"}')
    (root / "artifact").write_text("frozen")
    before = inventory(root)
    seal = tmp_path / "references/ref.json"
    seal_reference(root, seal, "ref")
    with new_bundle(tmp_path / "runs/candidate", [root]) as staged:
        (staged / "result").write_text("new")
        assert verify_reference(seal)["integrity_status"] == "passed"
    assert inventory(root) == before
    for path in (root / "run", tmp_path / "history/another", tmp_path / "longitudinal_graph_history/a"):
        with pytest.raises(ValueError, match="historical|overlaps"):
            with new_bundle(path, [root], force=True):
                pass
    with pytest.raises(FileExistsError):
        with new_bundle(tmp_path / "runs/candidate"):
            pass
    (root / "artifact").write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        verify_reference(seal)


def test_s3_primary_reference_metadata(tmp_path):
    # Frozen values are a specification fixture, never assertions about S4 fits.
    values = {"scenario": "S3-primary", "n_patients": 159, "n_visits": 497, "n_features": 98,
              "n_nodes": 11, "n_connected_components": 3, "n_mapper_covered_visits": 389}
    root = tmp_path / "reference"
    root.mkdir()
    (root / "02_longitudinal_mapper_summary.json").write_text(json.dumps(values))
    seal_reference(root, tmp_path / "seal.json", "specification-fixture")
    assert json.loads((root / "02_longitudinal_mapper_summary.json").read_text()) == values


def test_visit_coverage_partitions():
    total = 497
    frame = pd.DataFrame({"patient_id": [f"p{i // 3}" for i in range(total)], "clinical_episode_id": [f"v{i}" for i in range(total)],
        "hard_macrostate": [0] * 346 + [pd.NA] * 151, "macrostate_membership_tie": [False] * 346 + [True] * 29 + [False] * 122,
        "mapper_graph_covered": [True] * 389 + [False] * 108, "mapper_covered": [True] * 375 + [False] * 122})
    counts = review_qc.coverage_partition(frame).coverage_category.value_counts()
    assert counts.to_dict() == {"hard_assigned": 346, "uncovered": 108, "tie": 29, "covered_unsupported_macrostate_or_node": 14}
    assert counts.sum() == 497


def test_graph_full_vs_filtered_exports():
    # A synthetic graph with the supplied degree histogram and supported 5/8 views.
    pairs = [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9)]
    nodes = {f"n{i}": set() for i in range(11)}
    visits = []
    for a, b in pairs:
        idx = len(visits)
        visits.append({"patient_id": f"shared{idx}", "clinical_episode_id": f"v{idx}"})
        nodes[f"n{a}"].add(idx)
        nodes[f"n{b}"].add(idx)
    for node in nodes:
        n = int(node[1:])
        for j in range(max(0, 3 - len(nodes[node])) if n <= 6 else (1 if n == 10 else 0)):
            idx = len(visits)
            visits.append({"patient_id": f"private{idx}", "clinical_episode_id": f"v{idx}"})
            nodes[node].add(idx)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((f"n{a}", f"n{b}") for a, b in pairs)
    records = MAP.node_support_table({"nodes": nodes}, pd.DataFrame(visits), 3)
    scenario = {"graph": {"nodes": nodes}, "raw_nerve": graph, "node_table": records, "topological_edges": [None] * 5}
    tables, integrity = graph_views(scenario, pd.DataFrame(visits))
    assert integrity["views"]["full"]["degree_distribution"] == {"0": 1, "1": 5, "2": 4, "3": 1}
    assert integrity["views"]["full"]["n_edges"] == 8
    assert integrity["views"]["supported"]["n_edges"] == 5
    assert integrity["views"]["full"]["n_components"] == nx.number_connected_components(graph) == 3
    assert integrity["views"]["full"]["degree_sum"] == 16
    assert tables["mapper_edges_full"].n_shared_visits.gt(0).all()
    assert tables["mapper_edges_full"].patient_jaccard.between(0, 1).all()
    graph.add_edge("n1", "n10")
    with pytest.raises(ValueError, match="shared visits"):
        graph_views(scenario, pd.DataFrame(visits))


def test_patient_resampling_and_equal_patient_mass():
    raw, state, _, _ = source(20)
    draw, _ = patient_draw(raw, np.random.default_rng(1))
    assert draw.groupby(["patient_id", "original_episode_id"]).size().groupby(level=0).nunique().eq(1).all()
    cfg = config()
    scenario = MAP.run_mapper_scenario(state, list(state.columns[2:]), cfg, family_balance=True, scenario_name="S4-patient-mass", patient_weighted=True)
    weights = MAP.qc.representation_qc(state, scenario, MAP.feature_family, raw)["patient_visit_weight_qc"]
    np.testing.assert_allclose(weights.effective_pca_weight, 1 / state.patient_id.nunique(), atol=1e-12)
    # Every bootstrap draw retains mass, including repeat draws of one patient.
    assert draw.bootstrap_draw_id.nunique() == raw.patient_id.nunique()
    fitted = draw[MAP.KEYS + list(state.columns[2:])].copy()
    fitted.iloc[:, 2:] = fitted.iloc[:, 2:].fillna(fitted.iloc[:, 2:].median())
    fitted["fit_patient_id"] = draw.bootstrap_draw_id
    sampled = MAP.run_mapper_scenario(fitted, list(state.columns[2:]), cfg, family_balance=True, scenario_name="bootstrap-weight", patient_weighted=True)
    embedding, _ = MAP.qc.weighted_embedding(fitted[state.columns[2:]], cfg, [1 / np.sqrt(7)] * 7 + [1], fitted.fit_patient_id)
    np.testing.assert_allclose(sampled["embedding"], embedding)


def test_raw_missingness_not_zero_and_low_overlap():
    raw, _, _, _ = source(20)
    raw.loc[0, HSCRP] = 0
    raw.loc[1, HSCRP] = np.nan
    table = review_qc.raw_feature_qc(raw, [HSCRP], "original", pd.DataFrame()).iloc[0]
    assert table.n_zero == 1 and table.n_missing == 1
    assert table.observation_provenance.startswith("unknown")
    raw[HSCRP] = raw[HSCRP].astype(object)
    raw.loc[2, HSCRP] = "malformed"
    invalid = review_qc.raw_feature_qc(raw, [HSCRP], "original", pd.DataFrame()).iloc[0]
    assert invalid.n_invalid == 1 and invalid.n_missing == 1
    raw.loc[2, HSCRP] = np.nan
    features, audit = observed_pruning(raw, list(raw.columns[2:]), .95, 100, 100)
    assert len(features) == len(raw.columns) - 2
    assert audit.status.eq("not_testable_low_overlap").all()


def test_preprocessing_fit_train_only():
    raw, _, _, _ = source(20)
    cfg = config()
    definition = next(row for row in candidate_registry(list(raw.columns[2:]), cfg) if row["name"] == "S4-hscrp-winsor")
    train, test = raw.iloc[:10].copy(), raw.iloc[10:].copy()
    assert not set(train.patient_id) & set(test.patient_id)
    pre = TrainingPreprocessor(definition, cfg, MAP.reduce_redundancy).fit(train, list(raw.columns[2:]))
    bounds, medians, features = pre.bounds_, pre.medians_.copy(), list(pre.features_)
    test[HSCRP] = 1e10
    pre.transform(test)
    assert pre.bounds_ == bounds and pre.features_ == features
    pd.testing.assert_series_equal(pre.medians_, medians)
    assert pre.bounds_[1] < 1e10


def test_feature_role_circularity():
    cfg = config()
    cfg["methodology_review"]["validation_anchors"] = [{"feature": "anchor", "derived_from": [HSCRP], "approved_by": "fixture", "independence_review": "passed", "temporal_scope": "episode_resolved"}]
    roles = review_qc.feature_roles([HSCRP], pd.DataFrame({"anchor": [1]}), cfg)
    assert roles.set_index("feature").loc["anchor", "anchor_eligibility"] == "descriptive_only_circular"


def test_admin_missing_not_negative():
    raw, state, metadata, _ = source(20)
    metadata = metadata.drop(columns=["protocol", "era"])
    joined, availability = review_qc.recover_administration(state[MAP.KEYS], metadata, raw)
    assert availability.set_index("variable").loc["protocol", "status"] == "not_tested_missing_source"
    scenario = MAP.run_mapper_scenario(state, list(state.columns[2:]), config(), family_balance=True, scenario_name="S4-test")
    _, available, _, _ = MAP.qc.administrative_audit(scenario["visit_membership"], joined, scenario["embedding"], config())
    gates = review_qc.decision_gates({"reference_manifest_sha256": "fixture"}, {"status": "passed", "views": {}}, available, {}, "fixture")
    assert gates["status"] == "blocked"
    assert next(g for g in gates["gates"] if g["gate"] == "G5")["status"] == "inconclusive"


def test_administration_recovers_protocol_alias_and_uses_continuous_calendar():
    raw, state, metadata, _ = source(40)
    raw["ids__protocol_number"] = "synthetic"
    metadata = metadata.drop(columns=["protocol"])
    metadata["clinical_anchor_date"] = pd.to_datetime(metadata.clinical_anchor_date) + pd.to_timedelta(np.arange(40) * 365, unit="D")
    joined, recovery = review_qc.recover_administration(state[MAP.KEYS], metadata, raw)
    assert joined.protocol.eq("synthetic").all()
    assert "ids__protocol_number" in recovery.set_index("variable").loc["protocol", "protocol_alias_selectors"]
    scenario = MAP.run_mapper_scenario(state, list(state.columns[2:]), config(), family_balance=True, scenario_name="S4-test")
    scenario["visit_membership"].loc[:, "hard_macrostate"] = np.arange(40) % 2
    association, _, _, _ = review_qc.extended_administration(scenario["visit_membership"], joined, scenario["embedding"], config())
    year = association.set_index("administrative_variable").loc["calendar_year"]
    assert year.binning == "none_continuous_covariate" and pd.isna(year.cramers_v)


def test_atomic_failed_run_and_dry_run_do_not_publish(tmp_path):
    target = tmp_path / "runs/failed"
    with pytest.raises(RuntimeError):
        with new_bundle(target) as staged:
            (staged / "partial").write_text("incomplete")
            raise RuntimeError("QC failed")
    assert not target.exists()
    args = MAP.parse_args(["--run-id", "dry", "--reference-run-id", "absent", "--output-root", str(tmp_path / "review"), "--dry-run"])
    assert MAP.run(args)["reason"] == "G0_frozen_reference_unavailable"
    assert not (tmp_path / "review").exists()


def test_blocked_preview_is_isolated_and_labeled(tmp_path):
    _, _, _, intervals = source(6)
    cfg = config()
    cfg["bootstrap"]["replicates"] = 3
    cfg["null_models"]["replicates"] = 3
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    membership = pd.DataFrame({"patient_id": [f"fixture_{i // 2}" for i in range(6)], "clinical_episode_id": [f"v{i}" for i in range(6)],
        "mapper_covered": True, "hard_macrostate": np.arange(6) % 2, "macrostate_membership_tie": False,
        "node_weight__n0": 1 - np.arange(6) % 2, "node_weight__n1": np.arange(6) % 2,
        "macrostate_weight__0": 1 - np.arange(6) % 2, "macrostate_weight__1": np.arange(6) % 2})
    intervals_path, membership_path = tmp_path / "intervals.parquet", tmp_path / "memberships.parquet"
    intervals.to_parquet(intervals_path)
    membership.to_parquet(membership_path)
    readiness = tmp_path / "official_readiness.json"
    readiness.write_text('{"status":"blocked"}')
    before = readiness.read_bytes()
    args = FLOW.parse_args(["--allow-blocked-preview", "--output-root", str(tmp_path / "review"), "--run-id", "preview", "--config", str(config_path),
                           "--intervals", str(intervals_path), "--membership", str(membership_path), "--readiness", str(readiness)])
    result = FLOW.run(args)
    assert result["status"] == "EXPLORATORY_NOT_FOR_INFERENCE"
    assert (tmp_path / "review/previews/preview/EXPLORATORY_NOT_FOR_INFERENCE").exists()
    assert readiness.read_bytes() == before and not (tmp_path / "review/approved_temporal").exists()


def test_ineligible_sensitivity_and_registry_reproducibility():
    cfg = config()
    features = [HSCRP, "pro__fixture", "lab__other"]
    a, b = candidate_registry(features, cfg), candidate_registry(features, cfg)
    assert a == b
    assert next(r for r in a if r["name"] == "S4-pro-summary")["status"] == "not_applicable"
    assert next(r for r in a if r["name"] == "S4-common-features")["status"] == "not_applicable"
    assert all(r["name"].startswith("S4-") for r in a)
    with pytest.raises(ValueError):
        safe_id("../../history")
    _, state, _, _ = source(30)
    fit_a = MAP.run_mapper_scenario(state, list(state.columns[2:]), cfg, family_balance=True, scenario_name="S4-visit-comparator")
    fit_b = MAP.run_mapper_scenario(state, list(state.columns[2:]), cfg, family_balance=True, scenario_name="S4-visit-comparator")
    assert fit_a["graph"] == fit_b["graph"]
    np.testing.assert_allclose(fit_a["embedding"], fit_b["embedding"], atol=1e-12)


def test_ari_single_label_and_jaccard_universe():
    raw, state, _, _ = source(20)
    scenario = MAP.run_mapper_scenario(state, list(state.columns[2:]), config(), family_balance=True, scenario_name="fixture")
    reference = copy.deepcopy(scenario)
    reference["visit_membership"].loc[:, "hard_macrostate"] = 0
    scenario["visit_membership"].loc[:, "hard_macrostate"] = 1
    reference["visit_membership"].loc[:, "macrostate_membership_tie"] = False
    scenario["visit_membership"].loc[:, "macrostate_membership_tie"] = False
    result, crosswalk, universe = compare_reference(reference, scenario, MAP)
    assert result["ari"] is None and result["nmi"] is None
    assert result["n_visits_compared"] == 20
    assert universe.comparison_eligible.all()
    assert not crosswalk.empty
    assert crosswalk.patient_jaccard.between(0, 1).all()
    assert crosswalk.loc[crosswalk.one_to_one_match, "reference_macrostate_id"].is_unique


def test_interval_temporal_consistency():
    _, _, metadata, intervals = source(20)
    assert len(review_qc.validate_intervals(intervals, metadata)) == len(intervals)
    bad = intervals.copy()
    bad.loc[0, "patient_id"] = "different_patient"
    with pytest.raises(ValueError, match="same patient"):
        review_qc.validate_intervals(bad, metadata)
    bad = intervals.copy()
    bad.loc[0, "interval_days"] = -1
    with pytest.raises(ValueError, match="delta_t"):
        review_qc.validate_intervals(bad, metadata)
    with pytest.raises(ValueError, match="Duplicate"):
        review_qc.validate_intervals(pd.concat([intervals, intervals.iloc[:1]]), metadata)


def test_gate_blocks_script03_before_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(FLOW, "create_study_dirs", lambda *args: pytest.fail("blocked run wrote directories"))
    args = FLOW.parse_args(["--mode", "approved-temporal"])
    with pytest.raises(ValueError, match="blocked"):
        FLOW.run(args)
    args.mode = "reference-only"
    assert FLOW.run(args)["temporal_flow_executed"] is False
    assert not list(tmp_path.iterdir())


def test_approval_is_bound_to_run_and_exact_membership(tmp_path):
    membership = tmp_path / "membership.parquet"
    membership.write_bytes(b"synthetic")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"run_id": "fixture", "artifact_sha256": {membership.name: sha256(membership)}}))
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({"status": "ready_for_researcher_review", "scenario": "S4-fixture", "gates": [{"gate": f"G{i}", "status": "passed"} for i in range(8)]}))
    record = {"approval_status": "approved-temporal", "responsible": "SYNTHETIC_TEST_ONLY", "approved_at": "2026-10-12T00:00:00Z",
              "justification": "test", "selected_scenario": "S4-fixture", "source_run_id": "fixture", "run_manifest_sha256": sha256(manifest), "membership_sha256": sha256(membership)}
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps(record))
    assert require_temporal_approval(readiness, approval, manifest, membership) == record
    membership.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        require_temporal_approval(readiness, approval, manifest, membership)


def test_negative_controls_modify_geometry_preserve_patient_series():
    raw, _, _, _ = source(80)
    null, report = negative_patient_blocks(raw, list(raw.columns[2:]), np.random.default_rng(712))
    pd.testing.assert_frame_equal(null[MAP.KEYS], raw[MAP.KEYS])
    assert report["geometry_changed"]
    assert report["n_exchangeable_patients"] == 40
    # Column distributions are preserved while their joint clinical covariance changes.
    np.testing.assert_allclose(np.sort(null[HSCRP]), np.sort(raw[HSCRP]))
    assert not np.allclose(null.iloc[:, 2:].corr(), raw.iloc[:, 2:].corr(), equal_nan=True)


def test_candidate_end_to_end_preserves_reference_and_blocks_inference(tmp_path):
    raw, state, metadata, intervals = source()
    cfg = config()
    historical, _ = freeze_fixture(tmp_path, state, cfg)
    before = inventory(historical)
    raw_path = tmp_path / "raw.parquet"
    state_path, metadata_path, intervals_path = (tmp_path / name for name in ("state.parquet", "metadata.parquet", "intervals.parquet"))
    raw.to_parquet(raw_path)
    state.to_parquet(state_path)
    metadata.to_parquet(metadata_path)
    intervals.to_parquet(intervals_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    args = MAP.parse_args(["--state", str(state_path), "--metadata", str(metadata_path), "--intervals", str(intervals_path),
        "--integrated", str(raw_path), "--config", str(config_path), "--output-root", str(tmp_path / "review"),
        "--reference-root", str(historical), "--reference-run-id", "synthetic-reference", "--freeze-reference", "--run-id", "synthetic-qc"])
    args.review_config = None
    result = MAP.run(args)
    assert result["gate"] == "blocked" and inventory(historical) == before
    target = Path(result["output"])
    manifest = json.loads((target / "run_manifest.json").read_text())
    assert manifest["reference_integrity_after"] == manifest["reference_manifest_sha256"]
    for name, digest in manifest["artifact_sha256"].items():
        assert sha256(target / name) == digest
    comparison = pd.read_csv(target / "tables/02_scenario_comparison.csv")
    assert comparison.set_index("scenario").loc["S3-primary", "role"] == "reference"
    assert comparison.set_index("scenario").loc["S4-pro-summary", "status"] == "not_applicable"
    assert comparison.set_index("scenario").loc["S4-common-features", "status"] == "not_applicable"
    assert comparison.loc[comparison.status.eq("completed"), "gate"].eq("blocked").all()
    directory = target / "scenarios/S4-patient-mass"
    assert all((directory / name).exists() for name in ("02_PC1_representation_qc.png", "02_PC2_representation_qc.png", "02_hscrp_distribution_qc.png"))
    assert json.loads((directory / "02_bootstrap_summary.json").read_text())["held_out_projection"].startswith("not_run")
    assert pd.read_csv(directory / "02_negative_control_results.csv").geometry_changed.all()
    with pytest.raises(FileExistsError):
        MAP.run(args)
    assert inventory(historical) == before
    args.mode = "legacy-reproduction"
    args.run_id = "parallel-reproduction"
    MAP.run(args)
    diff = json.loads((args.output_root / "runs/parallel-reproduction/qc/02_reproduction_diff.json").read_text())
    assert diff["reference_manifest_sha256_before"] == diff["reference_manifest_sha256_after"]
    assert inventory(historical) == before

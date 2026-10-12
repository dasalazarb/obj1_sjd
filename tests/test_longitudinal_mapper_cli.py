"""Regression for bare Script 02 invocations and safe reference discovery."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.studies.longitudinal_graph._review import review_runner
from src.studies.longitudinal_graph._review.reference import inventory, safe_id, seal_reference
from src.studies.longitudinal_graph._review.review_cli import ReviewStartupError, resolve_review_options

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("methodology_fixtures", ROOT / "tests/test_longitudinal_methodology_review.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
MAP = fixtures.MAP


def common_paths(tmp_path):
    return SimpleNamespace(
        STUDIES_ANALYTIC_DIR=tmp_path / "data/analytic/studies",
        STUDIES_TABLES_DIR=tmp_path / "outputs/tables/studies",
        STUDIES_FIGURES_DIR=tmp_path / "outputs/figures/studies",
        STUDIES_QC_DIR=tmp_path / "outputs/qc/studies",
        STUDIES_LOGS_DIR=tmp_path / "outputs/logs/studies",
        INTEGRATED_LONGITUDINAL_PARQUET=tmp_path / "integrated.parquet",
    )


def mock_common(monkeypatch, paths):
    for name, value in vars(paths).items():
        monkeypatch.setattr(MAP.common, name, value)


def dummy_archive(paths, name):
    root = paths.STUDIES_TABLES_DIR / "longitudinal_graph_history" / name
    root.mkdir(parents=True)
    (root / "02_longitudinal_mapper_summary.json").write_text('{"scenario":"S3-primary"}')
    return root


def test_bare_cli_missing_reference_is_actionable_not_traceback(tmp_path, monkeypatch, capsys):
    paths = common_paths(tmp_path)
    mock_common(monkeypatch, paths)
    assert MAP.main([]) == 2
    error = capsys.readouterr().err
    assert "No frozen S3-primary reference" in error and "--reference-root" in error
    assert "Traceback" not in error and "Run IDs must" not in error
    assert not list(tmp_path.iterdir())


def test_automatic_run_id_unique_and_archive_reused_without_discovery_writes(tmp_path):
    paths = common_paths(tmp_path)
    archive = dummy_archive(paths, "historical-20261009")
    before = inventory(archive)
    args = MAP.parse_args([])
    first = resolve_review_options(args, paths)
    second = resolve_review_options(args, paths)
    assert first.run_id != second.run_id and safe_id(first.run_id) == first.run_id
    assert first.run_id_generated and first.reference_selection == "unique_archive"
    assert first.reference_run_id == archive.name and first.reference_root == archive
    assert first.freeze_reference and args.run_id is None and args.reference_root is None
    assert inventory(archive) == before
    assert not first.output_root.exists()


def test_unique_seal_takes_precedence_over_other_archives(tmp_path):
    paths = common_paths(tmp_path)
    original = dummy_archive(paths, "original")
    dummy_archive(paths, "another")
    seal = paths.STUDIES_TABLES_DIR / "longitudinal_graph_review/references/original.json"
    seal_reference(original, seal, "original")
    resolved = resolve_review_options(MAP.parse_args([]), paths)
    assert resolved.reference_selection == "unique_seal"
    assert resolved.reference_run_id == "original" and resolved.reference_root == original
    assert not resolved.freeze_reference


@pytest.mark.parametrize("kind", ["archive", "seal"])
def test_multiple_references_require_selection_without_freezing(tmp_path, kind):
    paths = common_paths(tmp_path)
    for name in ("first", "second"):
        archive = dummy_archive(paths, name)
        if kind == "seal":
            seal_reference(archive, paths.STUDIES_TABLES_DIR / f"longitudinal_graph_review/references/{name}.json", name)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    with pytest.raises(ReviewStartupError, match="Several"):
        resolve_review_options(MAP.parse_args([]), paths)
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert before == after
    selected = resolve_review_options(MAP.parse_args(["--reference-run-id", "first"]), paths)
    assert selected.reference_run_id == "first"


def test_explicit_missing_reference_id_does_not_fall_back(tmp_path):
    paths = common_paths(tmp_path)
    dummy_archive(paths, "available")
    with pytest.raises(ReviewStartupError, match="No frozen"):
        resolve_review_options(MAP.parse_args(["--reference-run-id", "different"]), paths)


def test_reference_root_supplies_id_and_dry_run_does_not_freeze(tmp_path, monkeypatch):
    paths = common_paths(tmp_path)
    mock_common(monkeypatch, paths)
    archive = dummy_archive(paths, "explicit-reference")
    before = inventory(archive)
    args = MAP.parse_args(["--reference-root", str(archive), "--dry-run"])
    assert MAP.run(args)["status"] == "blocked"  # G0 seal would be created by a full run.
    assert inventory(archive) == before
    assert not (paths.STUDIES_TABLES_DIR / "longitudinal_graph_review").exists()


def test_bad_ids_and_stale_seals_report_startup_errors(tmp_path):
    paths = common_paths(tmp_path)
    archive = dummy_archive(paths, "reference")
    with pytest.raises(ReviewStartupError, match="--run-id"):
        resolve_review_options(MAP.parse_args(["--run-id", "../history", "--reference-root", str(archive)]), paths)
    seal = paths.STUDIES_TABLES_DIR / "longitudinal_graph_review/references/reference.json"
    seal_reference(archive, seal, "reference")
    original = seal.read_bytes()
    archive.rename(archive.with_name("moved"))
    with pytest.raises(ReviewStartupError, match="unavailable"):
        resolve_review_options(MAP.parse_args([]), paths)
    assert seal.read_bytes() == original


def test_bare_cli_executes_default_candidate_with_generated_id(tmp_path, monkeypatch, capsys):
    """Real publication from synthetic default inputs; no IDs or CLI flags supplied."""
    paths = common_paths(tmp_path)
    mock_common(monkeypatch, paths)
    raw, state, metadata, intervals = fixtures.source(40)
    cfg = fixtures.config()
    cfg["mapper_validation"].update(bootstrap_replicates=0, admin_permutations=0)
    historical, _ = fixtures.freeze_fixture(paths.STUDIES_TABLES_DIR, state, cfg)
    historical.parent.rename(paths.STUDIES_TABLES_DIR / "longitudinal_graph_history")
    historical = paths.STUDIES_TABLES_DIR / "longitudinal_graph_history" / historical.name
    before = inventory(historical)
    inputs = paths.STUDIES_ANALYTIC_DIR / "longitudinal_graph"
    inputs.mkdir(parents=True)
    state.to_parquet(inputs / "01_longitudinal_visit_state.parquet")
    metadata.to_parquet(inputs / "01_longitudinal_visit_metadata.parquet")
    intervals.to_parquet(inputs / "01_longitudinal_intervals.parquet")
    raw.to_parquet(paths.INTEGRATED_LONGITUDINAL_PARQUET)
    monkeypatch.setattr(MAP, "load_config", lambda path: copy.deepcopy(cfg))
    full_registry = review_runner.candidate_registry
    # Keep this CLI regression focused; the existing integration test covers all candidates.
    monkeypatch.setattr(review_runner, "candidate_registry", lambda features, config: full_registry(features, config)[:1])
    assert MAP.main([]) == 0
    output = paths.STUDIES_TABLES_DIR / "longitudinal_graph_review"
    runs = list((output / "runs").iterdir())
    assert len(runs) == 1 and runs[0].name.startswith("candidate-qc-")
    manifest = json.loads((runs[0] / "run_manifest.json").read_text())
    assert manifest["reference_run_id"] == historical.name
    assert manifest["cli_resolution"] == {"run_id_generated": True, "reference_selection": "unique_archive"}
    assert "_review/review_cli.py" in manifest["code_sha256"]
    assert json.loads((runs[0] / "02_temporal_readiness.json").read_text())["status"] == "blocked"
    assert inventory(historical) == before
    assert '"stage": "reference_preflight"' in capsys.readouterr().out

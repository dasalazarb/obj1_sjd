"""Unit tests for longitudinal-cohort selection in Step 00."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "00_build_visit_spine.py"
SPEC = importlib.util.spec_from_file_location("build_visit_spine", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_filtered_raw_output_names_are_stable() -> None:
    assert MODULE.common.SOURCE_EPISODE_SPINE.name == "clinical_episode_spine_sjd.parquet"
    assert MODULE.common.SOURCE_EPISODE_SPINE_CSV.name == "clinical_episode_spine_sjd.csv"


def test_filter_longitudinal_patients_keeps_only_listed_ids() -> None:
    source = pd.DataFrame(
        {"patient_id": [10, 20, 20, 30], "clinical_episode_id": [1, 2, 3, 4]}
    )
    id_list = pd.DataFrame({"patient_id": [20, 20, 40]})

    filtered, metrics = MODULE.filter_longitudinal_patients(source, id_list)

    assert filtered["clinical_episode_id"].tolist() == [2, 3]
    assert metrics == {
        "n_ids_requested": 2,
        "n_ids_matched": 1,
        "n_ids_not_found": 1,
        "n_patients_after_filter": 1,
    }


def test_filter_longitudinal_patients_uses_exact_id_matching() -> None:
    source = pd.DataFrame({"patient_id": [10, "10", 20]})
    id_list = pd.DataFrame({"patient_id": ["10"]})

    filtered, _ = MODULE.filter_longitudinal_patients(source, id_list)

    assert filtered["patient_id"].tolist() == ["10"]


@pytest.mark.parametrize("missing_from", ["source", "id_list"])
def test_filter_longitudinal_patients_requires_patient_id(missing_from: str) -> None:
    source = pd.DataFrame({"patient_id": [10]})
    id_list = pd.DataFrame({"patient_id": [10]})
    if missing_from == "source":
        source = source.drop(columns="patient_id")
    else:
        id_list = id_list.drop(columns="patient_id")

    with pytest.raises(ValueError, match="patient_id"):
        MODULE.filter_longitudinal_patients(source, id_list)


def test_first_sync_copies_and_writes_provenance(tmp_path: Path) -> None:
    source = tmp_path / "eda" / "spine.parquet"
    destination = tmp_path / "obj1" / "spine.parquet"
    provenance = tmp_path / "obj1" / "spine.provenance.json"
    source.parent.mkdir()
    source.write_bytes(b"authoritative episode spine")

    metrics = MODULE.sync_upstream_episode_spine(source, destination, provenance)

    assert destination.read_bytes() == source.read_bytes()
    assert MODULE.sha256_file(destination) == MODULE.sha256_file(source)
    assert metrics["copy_performed"] is True
    assert metrics["input_mode"] == "eda_sjd_auto_sync"
    record = json.loads(provenance.read_text(encoding="utf-8"))
    assert record["sha256"] == MODULE.sha256_file(source)
    assert record["copy_performed"] is True


def test_sync_skips_an_identical_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    destination = tmp_path / "destination.parquet"
    provenance = tmp_path / "provenance.json"
    source.write_bytes(b"same")
    destination.write_bytes(b"same")

    metrics = MODULE.sync_upstream_episode_spine(source, destination, provenance)

    assert metrics["copy_performed"] is False
    assert MODULE.sha256_file(destination) == MODULE.sha256_file(source)
    assert json.loads(provenance.read_text())["copy_performed"] is False


def test_sync_replaces_an_outdated_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    destination = tmp_path / "destination.parquet"
    source.write_bytes(b"new upstream")
    destination.write_bytes(b"old local snapshot")

    metrics = MODULE.sync_upstream_episode_spine(
        source, destination, tmp_path / "provenance.json"
    )

    assert metrics["copy_performed"] is True
    assert destination.read_bytes() == source.read_bytes()
    assert MODULE.sha256_file(destination) == MODULE.sha256_file(source)


def test_missing_upstream_never_uses_old_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "missing.parquet"
    destination = tmp_path / "old.parquet"
    destination.write_bytes(b"must not be used")

    with pytest.raises(FileNotFoundError, match=str(source)):
        MODULE.sync_upstream_episode_spine(
            source, destination, tmp_path / "provenance.json"
        )

    assert destination.read_bytes() == b"must not be used"
    assert not (tmp_path / "provenance.json").exists()


def _contract_frame() -> "pd.DataFrame":
    return pd.DataFrame(
        {
            "patient_id": [1],
            "clinical_episode_id": ["episode-1"],
            "episode_start_date": ["2020-01-01"],
            "clinical_anchor_date": ["2020-01-01"],
            "episode_end_date": ["2020-01-01"],
            "clinical_visit": [True],
            "visit_type": ["clinical"],
            "clinical_baseline_episode_id": ["episode-1"],
            "clinical_baseline_date": ["2020-01-01"],
            "is_clinical_baseline": [True],
        }
    )


def test_cli_override_does_not_sync(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    explicit = tmp_path / "frozen.parquet"
    captured: dict[str, object] = {}
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--input", str(explicit)])
    monkeypatch.setattr(MODULE.common, "ensure_output_dirs", lambda: None)
    monkeypatch.setattr(
        MODULE,
        "sync_upstream_episode_spine",
        lambda *args, **kwargs: pytest.fail("CLI override attempted upstream sync"),
    )
    monkeypatch.setattr(MODULE, "read_source", lambda path: _contract_frame())
    monkeypatch.setattr(MODULE.pd, "read_csv", lambda path: pd.DataFrame({"patient_id": [1]}))
    monkeypatch.setattr(
        MODULE,
        "write_outputs",
        lambda episode, clinical, assertions, metrics: captured.update(metrics),
    )

    MODULE.main()

    assert captured["input_mode"] == "explicit_cli_override"
    assert captured["input_path"] == str(explicit)


def test_episode_identity_and_counts_are_invariant() -> None:
    source = pd.concat([_contract_frame(), _contract_frame()], ignore_index=True)
    source.loc[1, "clinical_episode_id"] = "episode-2"
    source.loc[1, "clinical_anchor_date"] = "2021-01-01"
    source.loc[1, "episode_start_date"] = "2021-01-01"
    source.loc[1, "episode_end_date"] = "2021-01-01"
    source.loc[1, "is_clinical_baseline"] = False

    episode_spine, clinical_spine, _, metrics = MODULE.build_episode_spines(source)

    invariant_columns = [
        "patient_id",
        "clinical_episode_id",
        "clinical_anchor_date",
        "clinical_visit",
        "is_clinical_baseline",
        "visit_type",
    ]
    expected = source.copy()
    expected["clinical_anchor_date"] = pd.to_datetime(expected["clinical_anchor_date"])
    pd.testing.assert_frame_equal(
        episode_spine[invariant_columns].reset_index(drop=True),
        expected[invariant_columns].reset_index(drop=True),
    )
    assert metrics["n_patients_input"] == source["patient_id"].nunique()
    assert metrics["n_episodes_input"] == len(source)
    assert metrics["n_clinical_visits"] == len(clinical_spine) == 2
    assert metrics["n_patients_with_clinical_baseline"] == 1

"""Unit tests for longitudinal-cohort selection in Step 00."""
from __future__ import annotations

import importlib.util
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
        "n_rows_before_longitudinal_filter": 4,
        "n_rows_after_longitudinal_filter": 2,
        "n_longitudinal_patient_ids": 2,
        "n_longitudinal_patient_ids_matched": 1,
    }


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

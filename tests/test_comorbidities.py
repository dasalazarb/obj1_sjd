import importlib.util
import sys
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "src" / "block_A" / "07_comorbidities.py"
SPEC = importlib.util.spec_from_file_location("comorbidities", MODULE_PATH)
comorbidities = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = comorbidities
SPEC.loader.exec_module(comorbidities)


def _prevalence_frame() -> pd.DataFrame:
    data = {name: pd.Series([True, False, pd.NA], dtype="boolean")
            for name in comorbidities.CONDITION_NAMES}
    data["baseline_pop"] = ["Pop1", "Pop1", "Pop1"]
    return pd.DataFrame(data)


def test_overall_prevalence_preserves_empty_condition_as_missing():
    result = comorbidities.summarize_overall_prevalence(_prevalence_frame())

    fibromyalgia = result.loc[result["condition"].eq("fibromyalgia")].iloc[0]
    assert fibromyalgia["n_total_cohort"] == 3
    assert fibromyalgia["n_evaluable"] == 2
    assert fibromyalgia["n_positive"] == 1
    assert fibromyalgia["n_negative"] == 1
    assert fibromyalgia["n_missing"] == 1
    assert fibromyalgia["pct_total_cohort"] != fibromyalgia["pct_among_evaluable"]


def test_pop_prevalence_uses_evaluable_denominator():
    result = comorbidities.summarize_prevalence_by_pop(
        _prevalence_frame(), replicates=10, seed=1
    )

    fibromyalgia = result.loc[result["condition"].eq("fibromyalgia")].iloc[0]
    assert fibromyalgia["n_pop1"] == 1
    assert fibromyalgia["N_pop1"] == 2
    assert fibromyalgia["pct_pop1"] == 50


def test_condition_statuses_are_mutually_exclusive_and_prioritized():
    general = pd.Series([True, True, False, pd.NA, pd.NA], dtype="boolean")
    history = pd.Series([True, False, True, False, pd.NA], dtype="boolean")
    confirmed = pd.Series([True, False, False, False, pd.NA], dtype="boolean")
    evaluated = pd.concat([general, history, confirmed], axis=1).notna().any(axis=1)

    result = comorbidities.derive_condition_status(general, history, confirmed, evaluated)

    assert result.tolist() == [
        "confirmed_present",
        "status_uncertain",
        "history_only",
        "not_documented",
        "missing",
    ]


def test_visit_spine_schema_is_read_without_empty_projection(monkeypatch):
    expected = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "visit_id": ["P1_0"],
            "visit_date": pd.to_datetime(["2020-01-01"]),
            "visit_number": [0],
        }
    )
    calls = []

    monkeypatch.setattr(
        comorbidities,
        "available_columns",
        lambda path: set(expected.columns),
    )

    def fake_read_parquet(path, columns=None):
        calls.append(columns)
        return expected.loc[:, columns].copy()

    monkeypatch.setattr(comorbidities.pd, "read_parquet", fake_read_parquet)

    result = comorbidities.load_visit_spine()

    assert result.equals(expected)
    assert calls == [["patient_id", "visit_id", "visit_date", "visit_number"]]

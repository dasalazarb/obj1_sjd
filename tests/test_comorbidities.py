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
            "observed_baseline_date": pd.to_datetime(["2020-01-01"]),
            "time_since_observed_baseline_years": [0.0],
            "age_at_visit": [50.0],
            "sex": ["Female"],
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
    assert calls == [[
        "patient_id", "visit_id", "visit_date", "visit_number",
        "observed_baseline_date", "time_since_observed_baseline_years",
        "age_at_visit", "sex",
    ]]


def test_longitudinal_essdai_falls_back_to_population_derivation():
    visit_date = pd.Timestamp("2020-01-01")
    spine = pd.DataFrame(
        {
            "patient_id": ["P1"], "visit_id": ["P1_0"],
            "visit_date": [visit_date], "visit_number": [0],
        }
    )
    pop = pd.DataFrame(
        {
            "patient_id": ["P1"], "visit_id": ["P1_0"],
            "essdai_total": [7.0], "pop_status": ["Pop1"],
        }
    )
    raw = pd.DataFrame({"patient_id": ["P1"], "visit_date": [visit_date]})
    baseline_data = {
        "patient_id": ["P1"], "baseline_essdai": [7.0],
        "baseline_pop": ["Pop1"], "age_baseline": [50.0], "sex": ["Female"],
    }
    baseline_data.update({name: pd.Series([False], dtype="boolean") for name in comorbidities.CONDITION_NAMES})
    baseline = pd.DataFrame(baseline_data)
    domains = spine.copy()

    result = comorbidities.build_longitudinal_essdai_dataset(
        spine, pop, raw, baseline, domains
    )

    assert result.loc[0, "essdai_total_recoded"] == 7.0
    assert result.loc[0, "essdai_total_source"] == "population_longitudinal__essdai_total"
    assert pd.isna(result.loc[0, "essdai_total_raw_qc"])


def test_population_loader_uses_documented_canonical_columns(monkeypatch):
    columns = [
        "patient_id", "visit_id", "visit_date", "visit_number", "essdai_total",
        "pop_status", "baseline_pop_status",
    ]
    expected = pd.DataFrame(columns=columns)
    calls = []
    monkeypatch.setattr(comorbidities, "available_columns", lambda path: set(columns))

    def fake_read_parquet(path, columns=None):
        calls.append(columns)
        return expected.loc[:, columns].copy()

    monkeypatch.setattr(comorbidities.pd, "read_parquet", fake_read_parquet)

    result = comorbidities.load_pop_classification()

    assert result.columns.tolist() == columns
    assert calls == [columns]


def test_empty_population_table_is_not_sent_to_chi_square(monkeypatch):
    frame = _prevalence_frame()
    frame["baseline_pop"] = "Unclassifiable"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("chi-square must not run without evaluable population rows")

    monkeypatch.setattr(comorbidities.stats, "chi2_contingency", fail_if_called)

    result = comorbidities.summarize_prevalence_by_pop(frame, replicates=10, seed=1)

    assert result["global_test"].eq("not estimable").all()
    assert result["global_p_value"].isna().all()

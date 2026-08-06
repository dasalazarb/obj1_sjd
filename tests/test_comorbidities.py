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
    data = {name: pd.Series([True, False, False], dtype="boolean")
            for name in comorbidities.CONDITION_NAMES}
    data.update({f"{name}_status": ["confirmed_present", "no_comorbidity", "missing"]
                 for name in comorbidities.CONDITION_NAMES})
    data["baseline_pop"] = ["Pop1", "Pop1", "Pop1"]
    return pd.DataFrame(data)


def test_overall_prevalence_codes_blank_comorbidity_as_false():
    result = comorbidities.summarize_overall_prevalence(_prevalence_frame())

    fibromyalgia = result.loc[result["condition"].eq("fibromyalgia")].iloc[0]
    assert fibromyalgia["n_total_cohort"] == 3
    assert fibromyalgia["n_evaluable_primary"] == 3
    assert fibromyalgia["n_confirmed_present"] == 1
    assert fibromyalgia["n_no_comorbidity"] == 2
    assert fibromyalgia["n_missing"] == 0
    assert fibromyalgia["pct_confirmed_total_cohort"] == fibromyalgia["pct_confirmed_among_evaluable"]


def test_pop_prevalence_uses_evaluable_denominator():
    result = comorbidities.summarize_prevalence_by_pop(
        _prevalence_frame(), replicates=10, seed=1
    )

    fibromyalgia = result.loc[result["condition"].eq("fibromyalgia")].iloc[0]
    assert fibromyalgia["n_pop1"] == 1
    assert fibromyalgia["N_pop1"] == 3
    assert fibromyalgia["pct_pop1"] == 100 / 3


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
        "no_comorbidity",
        "no_comorbidity",
    ]


def test_primary_exposure_excludes_history_uncertain_and_codes_missing_false():
    spec = next(c for c in comorbidities.CONDITIONS if c.name == "fibromyalgia")
    frame = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4", "P5"],
        "fibromyalgia_status": [
            "confirmed_present", "no_comorbidity", "history_only",
            "status_uncertain", "missing",
        ],
    })

    result = comorbidities.apply_exposure_definition(
        frame, spec, "confirmed_present_vs_no_comorbidity"
    )

    assert result["exposure"].iloc[0] == 1.0
    assert result["exposure"].iloc[1] == 0.0
    assert result["exposure"].iloc[2:4].isna().all()
    assert result["exposure"].iloc[4] == 0.0


def test_complete_cases_reports_missing_and_group_sizes():
    frame = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3"],
        "exposure": [1.0, 0.0, pd.NA],
        "age": [50.0, pd.NA, 60.0],
        "event": [1, 0, 1],
    })

    complete, audit = comorbidities.prepare_complete_cases(
        frame, ["patient_id", "exposure", "age", "event"], event_col="event"
    )

    assert complete["patient_id"].tolist() == ["P1"]
    assert audit["excluded_observations_missing_exposure"] == 1
    assert audit["excluded_observations_missing_age"] == 1
    assert audit["n_exposed_patients"] == 1
    assert audit["n_reference_patients"] == 0


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
    baseline_data.update({f"{name}_status": ["no_comorbidity"] for name in comorbidities.CONDITION_NAMES})
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


def test_new_domain_audit_is_returned_without_dataframe_attrs():
    visit_dates = pd.to_datetime(["2020-01-01", "2021-01-01"])
    long_data = {
        "patient_id": ["P1", "P1"],
        "visit_date": visit_dates,
        "visit_number": [0, 1],
    }
    for domain in comorbidities.DOMAIN_COLS:
        long_data[domain] = pd.Series([False, domain == "eg_renal_active"], dtype="boolean")
        long_data[comorbidities.DOMAIN_EVALUABLE_COLS[domain]] = pd.Series([True, True], dtype="boolean")
    longdf = pd.DataFrame(long_data)
    baseline_data = {
        "patient_id": ["P1"], "baseline_date": [visit_dates[0]],
        "baseline_essdai": [0.0], "baseline_pop": ["Pop3"],
        "age_baseline": [50.0], "sex": ["Female"],
    }
    baseline_data.update({name: pd.Series([False], dtype="boolean") for name in comorbidities.CONDITION_NAMES})
    baseline_data.update({f"{name}_status": ["no_comorbidity"] for name in comorbidities.CONDITION_NAMES})
    baseline = pd.DataFrame(baseline_data)

    survival, audit = comorbidities.build_new_domain_survival_dataset(
        longdf, baseline, return_audit=True
    )

    assert survival.attrs == {}
    assert audit.attrs == {}
    assert survival.loc[0, "first_new_domain_name"] == "renal"
    assert {"patient_id", "visit_date", "domain", "domain_at_risk", "domain_evaluated", "event_at_visit", "censoring_reason"}.issubset(audit.columns)


def test_new_domain_ignores_followup_with_all_risk_domains_missing():
    dates = pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"])
    data = {"patient_id":["P1"]*3,"visit_date":dates,"visit_number":[0,1,2]}
    for domain in comorbidities.DOMAIN_COLS:
        data[domain]=pd.Series([False,pd.NA,domain=="eg_renal_active"],dtype="boolean")
        data[comorbidities.DOMAIN_EVALUABLE_COLS[domain]]=pd.Series([True,False,True],dtype="boolean")
    longdf=pd.DataFrame(data)
    baseline_data={"patient_id":["P1"],"baseline_date":[dates[0]],"baseline_essdai":[0.0],"baseline_pop":["Pop3"],"age_baseline":[50.0],"sex":["Female"]}
    baseline_data.update({name:pd.Series([False],dtype="boolean") for name in comorbidities.CONDITION_NAMES})
    baseline_data.update({f"{name}_status":["no_comorbidity"] for name in comorbidities.CONDITION_NAMES})

    survival,audit=comorbidities.build_new_domain_survival_dataset(longdf,pd.DataFrame(baseline_data),return_audit=True)

    assert survival.loc[0,"first_new_domain_date"]==dates[2]
    assert survival.loc[0,"last_valid_domain_evaluation_date"]==dates[2]
    assert not audit.loc[audit.visit_date.eq(dates[1]),"domain_evaluated"].any()


def test_grouped_plot_clips_rounding_induced_negative_error_bars(tmp_path):
    row = {"display_label": "Test condition"}
    for pop in ["pop1", "pop2", "pop3"]:
        row.update({
            f"pct_{pop}": 0.0,
            f"ci95_{pop}_low": 1e-15,
            f"ci95_{pop}_high": 0.0,
            f"n_{pop}": 0,
            f"N_{pop}": 10,
        })
    output = tmp_path / "grouped.pdf"

    comorbidities.create_grouped_barplot(pd.DataFrame([row]), output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_essdai_ge5_survival_uses_activity_threshold_five():
    visit_dates = pd.to_datetime(["2020-01-01", "2021-01-01"])
    longdf = pd.DataFrame({
        "patient_id": ["P1", "P1"],
        "visit_date": visit_dates,
        "visit_number": [0, 1],
        "essdai_total_recoded": [4.0, 5.0],
    })
    baseline_data = {
        "patient_id": ["P1"], "baseline_date": [visit_dates[0]],
        "baseline_essdai": [4.0], "baseline_pop": ["Pop3"],
        "age_baseline": [50.0], "sex": ["Female"],
    }
    baseline_data.update({name: pd.Series([False], dtype="boolean") for name in comorbidities.CONDITION_NAMES})
    baseline_data.update({f"{name}_status": ["no_comorbidity"] for name in comorbidities.CONDITION_NAMES})
    baseline = pd.DataFrame(baseline_data)

    result = comorbidities.build_essdai_ge5_survival_dataset(longdf, baseline)

    assert comorbidities.ACTIVITY_THRESHOLD_SECTION5 == 5
    assert result.loc[0, "essdai_ge5_event"] == 1
    assert result.loc[0, "event_date"] == visit_dates[1]

    period = comorbidities.build_essdai_ge5_person_period(longdf, baseline)
    assert len(period) == 1
    assert period.loc[0, "event_interval"] == 1
    assert period.loc[0, "interval_number"] == 1

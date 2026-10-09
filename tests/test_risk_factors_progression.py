"""Temporal and inferential contracts for Primary Objective 3 associations."""
import importlib.util
import json
import subprocess
import warnings
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).parents[1]/"src/block_A/14_risk_factors_progression.py"
spec = importlib.util.spec_from_file_location("risk_factors14", PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def episodes(scores=(2, 3, 6, 1), domains=(0, 0, 1, 0), n=1):
    rows = []
    for i in range(n):
        for j, (score, domain) in enumerate(zip(scores, domains)):
            days = j*365
            rows.append({"patient_id": f"p{i}", "clinical_episode_id": f"p{i}e{j}",
                "clinical_anchor_date": pd.Timestamp("2020-01-01")+pd.Timedelta(days=days),
                "clinical_baseline_date": pd.Timestamp("2020-01-01"),
                "clinical_baseline_episode_id": f"p{i}e0", "is_clinical_baseline": j == 0,
                "clinical_visit_number": j+1, "clinical_visit": True,
                "time_since_clinical_baseline_days": days,
                mod.TIME: days/365.25, mod.ESSDAI: score,
                "essdai__pulmonary_ordinal_score": domain,
                "essdai__articular_ordinal_score": 0,
                "demo__age_at_baseline": 30+i,
                "demo__sex": "Female" if i%2 else "Male",
                "disease_duration": i%7,
                "sero__baseline_low_c4": float(i%2),
                "sero__anti_ro_ssa__ever_positive_through_episode": float(j > 0),
                "ext__biopsy_focus_score": float(i%5+j),
                "lab__c4__value": 15.+i+j,
                "lab__c4__measurement_date": pd.Timestamp("2020-01-01")+pd.Timedelta(days=days-1),
                "lab__c4__days_from_anchor": -1,
                "lab__c4__selection_status": "selected_single"})
    return pd.DataFrame(rows)


def datasets(df):
    baseline = df.loc[df.is_clinical_baseline].copy()
    df, baseline = mod.validate_inputs(df, baseline)
    registry = mod.resolve_predictors(mod.build_predictor_registry(), baseline, df)
    base = mod.build_baseline_analysis_dataset(baseline, registry)
    intervals = mod.build_lagged_interval_dataset(df, base, registry)
    risks = mod.build_domain_risk_sets(df, base)
    return df, base, intervals, risks, registry


def get_spec(registry, name):
    return next(s for s in registry if s.name == name)


def confirmed_composition_metadata():
    return {domain: {"total_column": mod.ESSDAI, "domain_column": mod.step13.DOMAIN_ALIASES[domain][0],
        "weight": weight, "index_version": "synthetic-v1", "total_includes_domain": True,
        "verification_status": "verified", "evidence_reference": "synthetic test composition contract",
        "evidence_sha256": "a"*64} for domain, weight in mod.canonical_domain_weights().items()}


def test_registry_public_names_and_absent_glandular_not_substituted():
    df, base, _, _, registry = datasets(episodes())
    assert get_spec(registry, "low_c4").baseline_column == "sero__baseline_low_c4"
    assert get_spec(registry, "anti_ro_ssa").time_varying_column.endswith("ever_positive_through_episode")
    assert get_spec(registry, "glandular_active").baseline_column is None
    assert len([s for s in registry if s.family == "essdai_domain"]) == 12
    assert not any(s.time_varying_column and "patient_consensus" in s.time_varying_column for s in registry)
    assert base.patient_id.is_unique


def test_step10_context_companion_join_preserves_baseline_and_asof_lab_metadata():
    source = episodes()
    metadata = ["lab__c4__measurement_date", "lab__c4__days_from_anchor"]
    context = source[mod.KEYS + ["clinical_anchor_date"] + metadata].copy()
    context["ext__biopsy_evaluable"] = [True, False, True, True]
    integrated = source.drop(columns=metadata)
    baseline = integrated.loc[integrated.is_clinical_baseline].copy()
    enriched, official = mod.attach_episode_context(integrated, baseline, context)
    assert official.clinical_episode_id.tolist() == baseline.clinical_episode_id.tolist()
    assert enriched.clinical_episode_id.tolist() == integrated.clinical_episode_id.tolist()
    df, base, intervals, risks, registry = datasets(enriched)
    assert intervals["pred__lab_c4"].notna().all()
    assert pd.isna(intervals["pred__biopsy_focus_score"].iloc[1])
    with pytest.raises(ValueError, match="context episode keys disagree"):
        mod.attach_episode_context(integrated, baseline, context.iloc[:-1])
    context.loc[0, "clinical_anchor_date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="context conflicts"):
        mod.attach_episode_context(integrated, baseline, context)


def test_repository_analyte_spellings_resolve_without_substituting_hs_crp():
    source = episodes().rename(columns={"lab__c4__value": "lab__complement_c4__value"})
    source["lab__crp_high_sensitivity__value"] = 1.
    source["sero__rheumatoid_factor__ever_positive_through_episode"] = 1.
    registry = mod.resolve_predictors(mod.build_predictor_registry(), source, source)
    assert get_spec(registry, "lab_c4").baseline_column == "lab__complement_c4__value"
    assert get_spec(registry, "lab_crp").baseline_column is None
    assert get_spec(registry, "lab_crp_high_sensitivity").baseline_column == "lab__crp_high_sensitivity__value"
    assert get_spec(registry, "rf").time_varying_column == "sero__rheumatoid_factor__ever_positive_through_episode"


def test_temporal_copy_is_from_episode_and_no_missing_middle_bridge():
    df, base, intervals, risks, registry = datasets(episodes(scores=(2, np.nan, 6, 1)))
    assert intervals.from_clinical_episode_id.tolist() == ["p0e0", "p0e1", "p0e2"]
    assert intervals.to_clinical_episode_id.tolist() == ["p0e1", "p0e2", "p0e3"]
    assert intervals["pred__anti_ro_ssa"].tolist() == [0., 1., 1.]
    assert intervals.event_cross_ge5.iloc[:2].isna().all()
    assert mod.run_temporal_leakage_qc(df, base, intervals, risks, registry).status.eq("pass").all()


def test_first_event_risk_ends_even_after_missing_predictor_and_domain_recurrence():
    df = episodes(scores=(2, 6, 2, 7), domains=(0, 1, 0, 1))
    df.loc[0, "sero__anti_ro_ssa__ever_positive_through_episode"] = np.nan
    _, _, intervals, _, _ = datasets(df)
    assert intervals.at_risk_first_ge5.tolist() == [True, False, False]
    assert intervals.at_risk_new_pulmonary.tolist() == [True, False, False]
    assert intervals.event_cross_ge5.tolist() == [1., 0., 1.]


def test_domain_at_baseline_active_never_incident_even_if_later_inactive():
    _, _, intervals, risks, _ = datasets(episodes(domains=(1, 0, 1, 0)))
    assert not intervals.at_risk_new_pulmonary.any()
    assert "pulmonary" not in risks.target_domain.tolist()


def test_domain_unknown_and_censor_at_last_evaluable_target_assessment():
    _, _, intervals, risks, _ = datasets(episodes(domains=(0, 0, np.nan, np.nan)))
    risk = risks.loc[risks.target_domain.eq("pulmonary")].iloc[0]
    assert not risk.event
    assert risk.risk_end_date == pd.Timestamp("2020-12-31")
    assert intervals.event_new_domain_pulmonary.iloc[1:].isna().all()
    assert not intervals.at_risk_new_pulmonary.iloc[1:].any()


def test_any_new_domain_partial_negatives_do_not_extend_censor_time():
    df = episodes(domains=(0, 0, np.nan, np.nan))
    df["essdai__articular_ordinal_score"] = [0, 0, 0, 0]
    _, _, _, risks, _ = datasets(df)
    risk = risks.loc[risks.target_domain.eq("any_new_domain")].iloc[0]
    assert risk.risk_end_date == pd.Timestamp("2020-12-31")
    df.loc[3, "essdai__articular_ordinal_score"] = 1
    _, _, _, risks, _ = datasets(df)
    risk = risks.loc[risks.target_domain.eq("any_new_domain")].iloc[0]
    assert risk.event == 1
    assert risk.event_date == pd.Timestamp("2022-12-31")


def test_zero_date_interval_does_not_bridge_canonical_episode():
    df = episodes()
    df.loc[1, "clinical_anchor_date"] = df.loc[0, "clinical_anchor_date"]
    df.loc[1, [mod.TIME, "time_since_clinical_baseline_days"]] = 0
    _, _, intervals, _, _ = datasets(df)
    assert intervals.from_clinical_episode_id.tolist() == ["p0e1", "p0e2"]
    assert "p0e0" not in intervals.from_clinical_episode_id.tolist()


def test_unknown_evaluability_is_missing_not_negative():
    df = episodes()
    df["essdai__pulmonary_evaluable"] = [True, pd.NA, True, True]
    _, base, intervals, _, _ = datasets(df)
    assert pd.isna(intervals.from_domain_pulmonary.iloc[1])
    assert not intervals.at_risk_new_pulmonary.iloc[1]
    assert base.eligible_new_pulmonary.iloc[0]


@pytest.mark.parametrize("change,reason", [
    (lambda d: d.assign(patient_id=None), "missing"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate"),
    (lambda d: d.assign(**{mod.ESSDAI: 124}), "range"),
    (lambda d: d.assign(essdai__pulmonary_ordinal_score=4), "outside"),
    (lambda d: d.assign(**{mod.TIME: -1}), "negative"),
])
def test_input_contract_hard_failures(change, reason):
    df = change(episodes())
    with pytest.raises(ValueError, match=reason):
        mod.validate_inputs(df, df.loc[df.is_clinical_baseline])


def test_official_baseline_dates_and_predictors_cannot_disagree():
    df = episodes()
    baseline = df.loc[df.is_clinical_baseline].copy()
    baseline.loc[0, "clinical_baseline_date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="baseline conflicts"):
        mod.validate_inputs(df, baseline)
    baseline = df.loc[df.is_clinical_baseline].copy()
    baseline.loc[0, "sero__baseline_low_c4"] = 1
    with pytest.raises(ValueError, match="predictor conflicts"):
        mod.validate_inputs(df, baseline)


def test_progression_product_actual_schema_checked_including_states_and_scores():
    df = episodes()
    p = df.rename(columns={mod.ESSDAI: "essdai_total"}).copy()
    p["essdai_activity_state"] = mod.step13.classify_essdai_series(df[mod.ESSDAI])
    mod.validate_inputs(df, df.loc[df.is_clinical_baseline], p)
    with pytest.raises(ValueError, match="episode keys disagree"):
        mod.validate_inputs(df, df.loc[df.is_clinical_baseline], p.iloc[:-1])
    p.loc[1, "essdai_total"] = 10
    with pytest.raises(ValueError, match="disagrees"):
        mod.validate_inputs(df, df.loc[df.is_clinical_baseline], p)


def test_future_undated_labs_excluded_and_nearest_pre_anchor_medians_retained():
    df = episodes()
    df.loc[1, "lab__c4__measurement_date"] += pd.Timedelta(days=10)
    df.loc[2, "lab__c4__selection_status"] = "selected_repeated_numeric_median"
    df.loc[3, ["lab__c4__measurement_date", "lab__c4__days_from_anchor"]] = [pd.NaT, np.nan]
    df, base, intervals, _, registry = datasets(df)
    assert intervals["pred__lab_c4"].notna().tolist() == [True, False, True]
    avail = mod.build_predictor_availability(base, df, registry)
    row = avail.loc[avail.name.eq("lab_c4") & avail.analysis_type.eq("timevarying")].iloc[0]
    assert row.n_temporal_or_evaluability_excluded == 2
    assert df["lab__c4__value"].notna().all()  # Canonical values preserved.


def test_temporal_qc_detects_tampered_from_predictor_and_duplicate_intervals():
    df, base, intervals, risks, registry = datasets(episodes())
    intervals.loc[0, "pred__anti_ro_ssa"] = 1
    with pytest.raises(ValueError, match="temporal leakage"):
        mod.run_temporal_leakage_qc(df, base, intervals, risks, registry)
    qc = mod.run_temporal_leakage_qc(df, base, pd.concat([intervals, intervals.iloc[[0]]], ignore_index=True), risks, registry, False)
    assert {"predictor_from_episode", "duplicate_intervals"} <= set(qc.check)


def task_for(kind="trajectory", n=40):
    rng = np.random.default_rng(95)
    rows = []
    for i in range(n):
        x = i%2
        intercept = rng.normal(0, 1)
        for j in range(1, 4):
            rows.append({"patient_id": str(i), "pred__test": x, "time": float(j),
                "y": 4+intercept+.2*j+.3*x*j+rng.normal(0, .4),
                "baseline_essdai": 2+rng.uniform(), "age": 30+i, "sex": "F" if i%3 else "M",
                "duration": i%7, "from_essdai": 2+rng.uniform(), "interval_years": .5+j/5})
    spec = mod.PredictorSpec("test", "Test", "serology", "binary", "test", "test")
    return mod.ModelTask(spec, pd.DataFrame(rows), "baseline" if kind == "trajectory" else "timevarying",
                         "essdai_slope" if kind == "trajectory" else "subsequent_essdai", kind)


def test_singular_mixed_model_falls_back_with_attempt_preserved(monkeypatch):
    task = task_for()
    data, meta = mod.prepare_model(task, 40)
    names = ["Intercept", "x", "time", "x:time", "baseline_essdai", "age", "duration", "C(sex)[T.M]"]
    params = pd.Series(np.zeros(len(names)), index=names)
    fake = SimpleNamespace(params=params, fe_params=params, converged=True, cov_re=np.array([[0.]]),
        bse=pd.Series(1., index=names), pvalues=pd.Series(.5, index=names),
        cov_params=lambda: pd.DataFrame(np.eye(len(names)), index=names, columns=names))
    monkeypatch.setattr(mod.MixedLM, "from_formula", lambda *a, **k: SimpleNamespace(fit=lambda **k: fake))
    attempts = []
    row = mod.fit_task(task, data, meta, attempts)
    assert row["model_used"] == "Gaussian GEE exchangeable"
    assert row["fallback_used"] and row["valid_for_inference"] and not row["singular_fit"]
    assert attempts[0]["singular_fit"] and not attempts[0]["valid_for_inference"]
    assert row["estimand"] == "difference in annual ESSDAI slope"


def test_failed_gee_keeps_attempt_metadata_without_promoting_an_estimate(monkeypatch):
    task = task_for("lagged")
    data, meta = mod.prepare_model(task, 40)
    def fail(*args, **kwargs):
        raise ValueError("forced invalid model")
    monkeypatch.setattr(mod.GEE, "from_formula", fail)
    attempts = []
    row = mod.fit_task(task, data, meta, attempts)
    assert row["model_status"] == "model_failed"
    assert row["model_attempted"] == "Gaussian GEE exchangeable"
    assert row["model_used"] == "none" and pd.isna(row["estimate"])
    assert len(attempts) == 1 and not row["valid_for_inference"]


def test_real_lagged_gee_and_correct_formula():
    task = task_for("lagged")
    data, meta = mod.prepare_model(task, 40)
    assert "from_essdai" in meta["formula"] and "interval_years" in meta["formula"]
    row = mod.fit_task(task, data, meta, [])
    assert row["valid_for_inference"] and row["model_used"] == "Gaussian GEE exchangeable"
    assert row["cluster_variable"] == "patient_id"


def test_fixed_interval_spacing_does_not_create_redundant_intercept():
    task = task_for("lagged")
    task.data["interval_years"] = 1.
    data, meta = mod.prepare_model(task, 40)
    assert meta["supported_for_model"]
    assert meta["constant_state_covariates"] == "interval_years"
    assert "interval_years" not in meta["formula"]
    assert mod.fit_task(task, data, meta, [])["valid_for_inference"]


def event_task(kind="cox", n=120):
    rng = np.random.default_rng(122)
    spec = mod.PredictorSpec("test", "Test", "serology", "binary", "test", "test")
    x = np.arange(n)%2
    baseline = rng.uniform(0, 4, n)
    duration = rng.exponential(2, n)
    event = rng.binomial(1, .5, n)
    data = pd.DataFrame({"patient_id": [str(i) for i in range(n)], "pred__test": x,
        "y": event, "time_years": duration, "baseline_essdai": baseline, "age": rng.normal(45, 10, n),
        "sex": np.where(np.arange(n)%3, "F", "M"), "duration": rng.uniform(0, 10, n),
        "start_time": 0., "interval_years": duration})
    return mod.ModelTask(spec, data, "baseline" if kind == "cox" else "timevarying", "essdai_ge5", kind)


def test_real_cox_estimate_and_predictor_ph_test():
    task = event_task()
    data, meta = mod.prepare_model(task, 120)
    assert meta["supported_for_model"]
    row = mod.fit_task(task, data, meta, [])
    assert row["valid_for_inference"] and row["effect_measure"] == "HR"
    assert row["ph_assumption_status"] in {"pass", "warning"}
    assert np.isfinite(row["ph_assumption_p_value"])


def test_real_timevarying_event_model_has_offset_cluster_and_irr(monkeypatch):
    task = event_task("poisson")
    data, meta = mod.prepare_model(task, 120)
    assert meta["supported_for_model"]
    real = mod.GEE.from_formula
    seen = []
    def record(*a, **k):
        seen.append(k)
        return real(*a, **k)
    monkeypatch.setattr(mod.GEE, "from_formula", record)
    row = mod.fit_task(task, data, meta, [])
    assert row["valid_for_inference"] and row["effect_measure"] == "IRR"
    assert row["fallback_used"] and "clustered start-stop Cox" in row["model_selection_note"]
    assert seen[0]["groups"] == "patient_id"
    assert np.allclose(seen[0]["offset"], np.log(data.interval_years))


def test_feasibility_prevents_sparse_fit_and_reduced_model_is_prespecified(monkeypatch):
    task = event_task(n=40)
    task.data["y"] = 0
    task.data.loc[:11, "y"] = 1
    data, meta = mod.prepare_model(task, 40)
    assert meta["adjustment_status"] == "minimal"
    assert meta["adjustment_status_legacy"] == "prespecified_reduced"
    assert meta["adjustment_covariates"] == "baseline_essdai"
    assert meta["supported_for_model"]
    task.data.loc[:11, "y"] = 0
    data, meta = mod.prepare_model(task, 40)
    assert not meta["supported_for_model"]
    monkeypatch.setattr(mod, "fit_baseline_cox_model", lambda *a: pytest.fail("sparse model was fitted"))
    row = mod.fit_task(task, data, meta, [])
    assert row["model_status"] == "not_estimable" and not row["valid_for_inference"]


def test_standardization_is_patient_based_for_baseline_and_original_retained():
    task = task_for()
    task.spec = mod.replace(task.spec, variable_type="continuous", transform="zscore")
    task.data["pred__test"] = task.data.patient_id.astype(float)
    extra = task.data.loc[task.data.patient_id.eq("0")].copy()
    task.data = pd.concat([task.data, extra], ignore_index=True)
    _, meta = mod.prepare_model(task, 40)
    assert meta["standardization_mean"] == pytest.approx(19.5)
    assert task.data["pred__test"].max() == 39


def test_incompatible_lab_units_skip_model_without_new_unit_conversion():
    task = task_for("lagged")
    task.spec = mod.replace(task.spec, family="laboratory", variable_type="continuous", transform="zscore")
    task.data["pred_unit__test"] = np.where(task.data.patient_id.astype(int)%2, "mg/dL", "g/L")
    _, meta = mod.prepare_model(task, 40)
    assert not meta["supported_for_model"]
    assert "mixed laboratory units" in meta["support_reason"]


def test_fdr_separate_families_only_valid_inference():
    frame = pd.DataFrame({"fdr_family": ["a", "a", "b", "a"],
                          "valid_for_inference": [True, True, True, False],
                          "p_value": [.01, .04, .04, .001]})
    out, qc = mod.apply_fdr_by_family(frame)
    assert out.q_value.tolist()[:3] == pytest.approx([.02, .04, .04])
    assert pd.isna(out.q_value.iloc[3])
    assert qc.n_valid_tests.sum() == 3


def test_tasks_exclude_concurrent_baseline_outcomes_and_same_cross_domain():
    df, base, intervals, risks, registry = datasets(episodes(n=2))
    tasks = mod.build_analysis_tasks(df, base, intervals, risks, registry)
    trajectories = [t for t in tasks if t.kind == "trajectory"]
    assert all(t.data.time.gt(0).all() for t in trajectories)
    cross = [t for t in tasks if t.analysis_type == "cross_domain"]
    assert len(cross) == 12*11
    assert all(t.spec.name.removeprefix("domain_") != t.outcome.removeprefix("new_domain_") for t in cross)
    assert all(t.minimum_exposed_events == 5 for t in cross)
    assert not any("Pop" in t.outcome or "pop" in t.outcome for t in tasks)


def test_main_outputs_empty_inference_dry_run_and_overwrite(tmp_path):
    df = episodes(n=3)
    source, baseline = tmp_path/"integrated.parquet", tmp_path/"baseline.parquet"
    df.to_parquet(source, index=False)
    df.loc[df.is_clinical_baseline].to_parquet(baseline, index=False)
    args = mod.parse_args(["--integrated", str(source), "--baseline", str(baseline),
                           "--progression", str(tmp_path/"absent.parquet"), "--output-root", str(tmp_path/"dry"), "--dry-run"])
    mod.main(args)
    dirs = mod.output_directories(tmp_path/"dry")
    assert not list(dirs["tables"].glob("*.csv"))
    assert not list(dirs["analytic"].glob("*.parquet"))
    assert (dirs["qc"]/"14_dry_run_feasibility.csv").exists()
    args.output_root, args.dry_run = tmp_path/"full", False
    mod.main(args)
    dirs = mod.output_directories(args.output_root)
    assert all((dirs["tables"]/f"14_{name}.csv").exists() for name in mod.TABLES)
    assert all((dirs["qc"]/f"14_{name}.csv").exists() for name in mod.QC_TABLES)
    assert len(list(dirs["analytic"].glob("*.parquet"))) == 3
    assert len(list(dirs["figures"].glob("*.pdf"))) == 3
    assert all(p.stat().st_size > 1000 for p in dirs["figures"].glob("*.pdf"))
    results = pd.read_csv(dirs["tables"]/"14_model_interpretation_summary.csv")
    assert not results.valid_for_inference.any()
    assert results.estimate.isna().all() and results.q_value.isna().all()
    assert (dirs["logs"]/"14_risk_factors_progression.log").stat().st_size > 0
    args.overwrite = False
    previous = (dirs["tables"]/"14_predictor_registry.csv").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        mod.main(args)
    assert (dirs["tables"]/"14_predictor_registry.csv").read_bytes() == previous


def test_cli_event_safeguard_cannot_be_lowered():
    assert mod.parse_args([]).overwrite
    assert not mod.parse_args(["--no-overwrite"]).overwrite
    with pytest.raises(SystemExit):
        mod.parse_args(["--minimum-events", "1"])


@pytest.mark.parametrize("kind", ["trajectory", "lagged", "cox", "poisson"])
def test_missing_duration_preserves_age_and_sex_and_full_when_observed(kind):
    task = event_task(kind) if kind in {"cox", "poisson"} else task_for(kind)
    n = task.data.patient_id.nunique()
    data, full = mod.prepare_model(task, n)
    assert full["selected_adjustment_level"] == "full"
    task.data["duration"] = np.nan
    data, reduced = mod.prepare_model(task, n)
    assert reduced["selected_adjustment_level"] == "reduced_A"
    assert "age" in reduced["adjustment_covariates"] and "sex" in reduced["adjustment_covariates"]
    assert "C(sex)" in reduced["formula"] and "duration" not in reduced["formula"]
    assert reduced["n_complete_cases"] == len(data)
    levels = json.loads(reduced["candidate_adjustment_qc"])
    assert [level["level"] for level in levels] == list(mod.ADJUSTMENT_LEVELS)
    assert levels[0]["unavailable"] == ["duration"]
    if kind in {"cox", "poisson"}:
        assert reduced["n_events"] == int(data.y.eq(1).sum())
    if kind == "lagged":
        assert "from_essdai" in reduced["formula"] and "interval_years" in reduced["formula"]


def test_outcome_values_do_not_select_adjustment_level():
    task = task_for("lagged")
    task.data["duration"] = np.nan
    _, before = mod.prepare_model(task, 40)
    task.data["y"] = np.random.default_rng(4).normal(70, 12, len(task.data))
    _, after = mod.prepare_model(task, 40)
    assert before["selected_adjustment_level"] == after["selected_adjustment_level"] == "reduced_A"
    assert before["candidate_adjustment_qc"] == after["candidate_adjustment_qc"]


@pytest.mark.parametrize("events,level", [(30, "full"), (20, "reduced_A"), (15, "reduced_B"), (10, "minimal")])
def test_event_per_parameter_selects_prespecified_supported_level(events, level):
    task = event_task()
    task.data["y"] = 0
    task.data.loc[:events-1, "y"] = 1
    data, meta = mod.prepare_model(task, 120)
    assert meta["selected_adjustment_level"] == level
    assert meta["n_events"] == events and meta["n_complete_cases"] == len(data)
    assert meta["events_per_candidate_parameter"] >= 5


@pytest.mark.parametrize("kind", ["cox", "poisson"])
@pytest.mark.parametrize("events", [0, 1, 2])
def test_zero_one_two_events_remain_not_estimable(kind, events):
    task = event_task(kind)
    task.data["y"] = 0
    task.data.iloc[:events, task.data.columns.get_loc("y")] = 1
    data, meta = mod.prepare_model(task, 120)
    row = mod.fit_task(task, data, meta, [])
    assert meta["selected_adjustment_level"] == "none"
    assert meta["n_events"] == events
    assert row["claim_status"] == "unsupported" and not row["valid_for_inference"]
    assert pd.isna(row["estimate"])


def test_sex_missing_is_excluded_and_constant_sex_uses_reduced_b():
    task = task_for("lagged")
    task.data.loc[task.data.patient_id.eq("0"), "sex"] = pd.NA
    data, meta = mod.prepare_model(task, 40)
    assert "0" not in data.patient_id.tolist()
    assert meta["n_patients"] == 39 and set(data.sex) == {"M", "F"}
    task.data["sex"] = "F"
    _, meta = mod.prepare_model(task, 40)
    assert meta["selected_adjustment_level"] == "reduced_B"
    assert "C(sex)" not in meta["formula"]


def test_interval_demographics_come_from_official_baseline_not_later_episodes():
    source = episodes(n=3)
    later = ~source.is_clinical_baseline
    source.loc[later, "demo__age_at_baseline"] = 999
    source.loc[later, "demo__sex"] = "Future"
    source.loc[later, "disease_duration"] = 999
    _, base, intervals, _, _ = datasets(source)
    expected = base.set_index("patient_id")
    for name in ("age", "sex", "duration"):
        assert intervals[name].eq(intervals.patient_id.map(expected["pred__"+name])).all()


def test_duration_does_not_resolve_unverified_aliases_or_clip_impossible_values():
    source = episodes(n=3)
    registry = mod.resolve_predictors(mod.build_predictor_registry(), source, source)
    duration = get_spec(registry, "duration")
    source.loc[0, "disease_duration"] = -2
    source.loc[4, "disease_duration"] = 100
    values = mod.predictor_values(source, duration, True)
    assert pd.isna(values.loc[0]) and pd.isna(values.loc[4])
    source.loc[8, "dx_date"] = pd.Timestamp("2021-01-01")
    assert pd.isna(mod.predictor_values(source, duration, True).loc[8])
    source = source.drop(columns="disease_duration")
    for name in ("time_since_diagnosis_years", "disease_duration_years", "disease_duration_yrs_model", "demo__disease_duration"):
        source[name] = 2
    registry = mod.resolve_predictors(mod.build_predictor_registry(), source, source)
    assert get_spec(registry, "duration").baseline_column is None


def test_weighted_other_domain_adjustments_aliases_missing_and_inconsistent_panels():
    source = episodes(scores=(12, 12, 12, 12), domains=(np.nan,)*4)
    source["essdai__articular_ordinal_score"] = 2
    source["essdai__biological_ordinal_score"] = 2
    source.attrs["essdai_composition_provenance"] = confirmed_composition_metadata()
    _, _, intervals, _, _ = datasets(source)
    assert intervals.from_domain_contribution_articular.eq(8).all()
    assert intervals.from_essdai_other_domains_articular.eq(4).all()
    assert intervals.from_domain_contribution_biological.eq(2).all()
    assert intervals.from_essdai_other_domains_biological.eq(10).all()
    assert intervals.from_essdai_other_domains_glandular.isna().all()
    assert mod.canonical_domain_weights()["glandular"] == 2
    assert mod.canonical_domain_weights()["pns"] == mod.ESSDAI_DOMAIN_WEIGHTS["neuro_periph"] == 5
    source[mod.ESSDAI] = 3
    _, _, intervals, _, _ = datasets(source)
    assert intervals.from_essdai_other_domains_articular.isna().all()
    qc = pd.DataFrame(intervals.attrs["domain_composition_qc"])
    assert qc.loc[qc.domain.eq("articular"), "n_invalid_subtraction"].iloc[0] == 3
    for domain in mod.DOMAINS:
        source[mod.step13.DOMAIN_ALIASES[domain][0]] = 0
    _, _, intervals, _, _ = datasets(source)
    assert intervals.from_essdai_other_domains_biological.isna().all()
    assert all(row["n_complete_panel_total_mismatch"] == 3 for row in intervals.attrs["domain_composition_qc"])


def test_lagged_domain_sensitivity_uses_other_domains_and_no_false_baseline_omission():
    rng = np.random.default_rng(222)
    source = episodes(n=40)
    source.attrs["essdai_composition_provenance"] = confirmed_composition_metadata()
    source["essdai__glandular_ordinal_score"] = rng.integers(0, 3, len(source))
    source[mod.ESSDAI] = 2*source.essdai__glandular_ordinal_score + rng.uniform(1, 4, len(source))
    df, base, intervals, risks, registry = datasets(source)
    tasks = mod.build_analysis_tasks(df, base, intervals, risks, registry)
    lagged = [t for t in tasks if t.kind == "lagged"]
    assert not any(t.sensitivity in {"without_baseline_essdai", "without_baseline_total"} for t in lagged)
    target = [t for t in lagged if t.spec.name == "domain_glandular"]
    primary = next(t for t in target if t.sensitivity == "none")
    sensitivity = next(t for t in target if t.sensitivity == "domain_from_other_domains_adjusted")
    data, meta = mod.prepare_model(sensitivity, 40)
    _, main = mod.prepare_model(primary, 40)
    assert meta["supported_for_model"] and main["supported_for_model"]
    assert meta["formula"] != main["formula"]
    assert "from_essdai_other_domains_glandular" in meta["formula"]
    assert meta["domain_weight"] == 2 and not meta["sensitivity_numerically_identical_state"]
    assert np.allclose(data.from_essdai_other_domains_glandular, data.from_essdai - 2*data.pred__domain_glandular)
    row = mod.fit_task(sensitivity, data, meta, [])
    assert row["claim_status"] == "sensitivity" and row["valid_for_inference"]


def test_outcome_wide_bh_is_independent_and_excludes_alternatives_sensitivities_failures():
    rows = []
    for predictor, p in (("a", .01), ("b", .03), ("c", .2), ("alt", .001), ("sens", .001), ("failed", .001)):
        rows.append({"predictor": predictor, "hypothesis_group_id": "a" if predictor == "alt" else predictor,
            "p_value": p, "fdr_family": predictor, "analysis_type": "baseline", "outcome": "slope",
            "effect_measure": "beta", "sensitivity": "check" if predictor == "sens" else "none",
            "primary_canonical": predictor != "alt", "fdr_eligibility": predictor != "alt",
            "valid_for_inference": predictor != "failed"})
    frame, _ = mod.apply_fdr_by_family(pd.DataFrame(rows))
    before = frame.q_value.copy()
    out, qc = mod.apply_outcome_wide_fdr(frame)
    assert out.q_value.equals(before) and out.q_value_family.equals(before)
    assert out.q_value_outcome_wide.iloc[:3].tolist() == pytest.approx([.03, .045, .2])
    assert out.q_value_outcome_wide.iloc[3:].isna().all()
    assert qc.n_tests_bh.iloc[0] == 3 and qc.n_duplicate_excluded.iloc[0] == 1
    assert json.loads(qc.included_predictor_ids.iloc[0]) == ["a", "b", "c"]
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    out, qc = mod.apply_outcome_wide_fdr(duplicate)
    assert pd.isna(out.q_value_outcome_wide.iloc[-1]) and qc.n_duplicate_excluded.iloc[0] == 2
    for analysis, measure in (("baseline", "HR"), ("timevarying", "IRR"), ("cross_domain", "IRR")):
        extra = frame.iloc[[0]].assign(analysis_type=analysis, effect_measure=measure)
        out, qc = mod.apply_outcome_wide_fdr(pd.concat([frame, extra], ignore_index=True))
        assert out.q_value_outcome_wide.iloc[-1] == .01 and len(qc) == 2


def test_redundancy_small_pairs_are_not_evidence_and_glandular_policy_is_explicit():
    registry = mod.build_predictor_registry()
    focus, crp = get_spec(registry, "biopsy_focus_score"), get_spec(registry, "lab_crp_high_sensitivity")
    data = pd.DataFrame({"pred__biopsy_focus_score": [1., 2., 3.], "pred__lab_crp_high_sensitivity": [1., 2., 3.]})
    qc = mod.predictor_redundancy_qc(data, [focus, crp])
    assert qc.correlation.iloc[0] == 1 and not qc.redundancy_flag.iloc[0]
    assert qc.support_reason.iloc[0] == "insufficient_pair_support"
    gland, alt = get_spec(registry, "domain_glandular"), get_spec(registry, "glandular_salivary_gland_swelling_active")
    data = pd.DataFrame({"pred__domain_glandular": [0, 1]*30, "pred__glandular_salivary_gland_swelling_active": [0, 1]*30})
    qc = mod.predictor_redundancy_qc(data, [gland, alt])
    assert qc.redundancy_flag.iloc[0] and qc.concordance.iloc[0] == 1
    assert qc.construct_duplicate_prespecified.iloc[0] and qc.canonical_predictor.iloc[0] == gland.name
    assert gland.primary_canonical and gland.fdr_eligibility
    assert not alt.primary_canonical and not alt.fdr_eligibility and alt.alternative_to == gland.name
    assert alt.analysis_role == "alternate_representation"


def test_existing_outputs_are_backed_up_and_metadata_records_rules(tmp_path):
    df = episodes(n=2)
    source, baseline = tmp_path/"integrated.parquet", tmp_path/"baseline.parquet"
    df.to_parquet(source, index=False)
    df.loc[df.is_clinical_baseline].to_parquet(baseline, index=False)
    args = mod.parse_args(["--integrated", str(source), "--baseline", str(baseline),
        "--progression", str(tmp_path/"absent.parquet"), "--output-root", str(tmp_path/"run"), "--dry-run"])
    dirs = mod.output_directories(args.output_root)
    dirs["qc"].mkdir(parents=True)
    previous = dirs["qc"]/"14_structural_qc.csv"
    previous.write_text("previous run\n")
    mod.main(args)
    metadata = json.loads((dirs["qc"]/"14_analysis_metadata.json").read_text())
    backups = list(Path(metadata["output_backup_path"]).rglob("14_structural_qc.csv"))
    assert len(backups) == 1 and backups[0].read_text() == "previous run\n"
    assert metadata["dry_run"] and metadata["git_commit"]
    assert metadata["rule_versions"]["composition"] == mod.COMPOSITION_RULE_VERSION
    assert len(metadata["input_sha256"]["integrated"]) == 64
    assert "statsmodels" in metadata["versions"] and metadata["treatment_adjustment"]


def composition_fixture(n=325):
    frame = pd.DataFrame({"patient_id": [f"synthetic-{i}" for i in range(n)],
        "from_clinical_episode_id": [f"episode-{i}" for i in range(n)]})
    for domain in mod.DOMAINS:
        frame["from_domain_"+domain] = 0.
    frame["from_essdai"] = 0.
    frame.loc[:47, "from_domain_articular"] = 2
    frame.loc[:47, "from_essdai"] = 3  # 48 negative targets; contained in the 63 mismatches in THIS fixture only.
    frame.loc[48:62, "from_essdai"] = 1
    frame.loc[304:, "from_essdai"] = np.nan
    return frame


def test_composition_snapshot_counts_exclusive_categories_and_intersection_no_twelvefold_count():
    frame = composition_fixture()
    original = frame.copy()
    out, qc = mod.derive_other_domain_adjustments(frame, mod.build_predictor_registry())
    pd.testing.assert_frame_equal(out[original.columns], original)
    assert qc.n_intervals.eq(325).all() and qc.n_available.eq(241).all()
    assert qc.n_complete_panel_total_mismatch.eq(63).all()
    assert qc.n_missing_total_or_domain.eq(21).all()
    assert qc.loc[qc.domain.eq("articular"), "n_invalid_subtraction"].iloc[0] == 48
    audit = pd.DataFrame(out.attrs["domain_composition_audit_summary"])
    assert audit.groupby("domain").n_category.sum().eq(325).all()
    assert audit.groupby("domain").n_eligible.sum().eq(241).all()
    articular = audit.loc[audit.domain.eq("articular")]
    assert articular.n_discordant_and_negative.sum() == 48
    assert articular.n_discordant_not_negative.sum() == 15
    assert articular.n_negative_not_discordant.sum() == 0
    assert articular.n_neither_flag.sum() == 262
    distribution = pd.DataFrame(out.attrs["domain_composition_discrepancy_distribution"])
    assert distribution.n_unique_from_episodes.sum() == 304
    assert distribution.loc[distribution.from_composition_total_delta.ne(0), "n_unique_from_episodes"].sum() == 63
    assert out.from_essdai_other_domains_articular.iloc[:48].isna().all()
    assert out.from_domain_weighted_contribution_raw_articular.iloc[:48].eq(8).all()
    assert out.from_composition_complete_panel_discordant.sum() == 63


def test_partial_composition_requires_documented_same_version_and_never_fills_missing():
    frame = composition_fixture(3)
    frame["from_essdai"] = 12
    frame["from_domain_biological"] = 2
    frame["from_domain_pulmonary"] = np.nan
    out, qc = mod.derive_other_domain_adjustments(frame, mod.build_predictor_registry())
    assert out.from_essdai_other_domains_articular.isna().all()
    assert qc.loc[qc.domain.ne("pulmonary"), "n_partial_unverified"].eq(3).all()
    assert qc.loc[qc.domain.eq("pulmonary"), "n_missing_total_or_domain"].iloc[0] == 3
    proven, _ = mod.derive_other_domain_adjustments(frame, mod.build_predictor_registry(), confirmed_composition_metadata())
    assert proven.from_essdai_other_domains_articular.eq(4).all()
    assert proven.from_essdai_other_domains_biological.eq(10).all()
    assert proven.from_essdai_other_domains_pulmonary.isna().all()
    assert proven.from_domain_pulmonary.isna().all()
    frame["from_essdai_version"], frame["from_essdai_total_version"] = "old", "new"
    excluded, _ = mod.derive_other_domain_adjustments(frame, mod.build_predictor_registry(), confirmed_composition_metadata())
    assert excluded.from_essdai_other_domains_articular.isna().all()
    assert excluded.from_composition_category_articular.eq("incompatible_provenance_or_scale").all()


def test_complete_composition_rejects_even_small_unexplained_difference_and_duplicate_from():
    frame = composition_fixture(3)
    frame["from_essdai"] = 8
    out, _ = mod.derive_other_domain_adjustments(frame, mod.build_predictor_registry())
    assert out.from_essdai_other_domains_articular.eq(0).all()
    frame["from_essdai"] += .00001
    out, _ = mod.derive_other_domain_adjustments(frame, mod.build_predictor_registry())
    assert out.from_essdai_other_domains_articular.isna().all()
    with pytest.raises(ValueError, match="unique canonical FROM"):
        mod.derive_other_domain_adjustments(pd.concat([frame, frame.iloc[[0]]]), mod.build_predictor_registry())


def test_effective_static_availability_matches_patient_baseline_and_selected_cases():
    source = episodes(n=40)
    source.loc[(source.patient_id.eq("p0")) & source.is_clinical_baseline, "demo__age_at_baseline"] = np.nan
    source.loc[~source.is_clinical_baseline, "demo__age_at_baseline"] = 999
    df, base, intervals, _, registry = datasets(source)
    assert intervals.loc[intervals.patient_id.eq("p0"), "age"].isna().all()
    task = mod.ModelTask(get_spec(registry, "biopsy_focus_score"), intervals.rename(columns={"to_essdai": "y"}),
        "timevarying", "subsequent_essdai", "lagged")
    data, meta = mod.prepare_model(task, 40)
    available = mod.build_predictor_availability(base, df, registry, intervals, [(task, data, meta)])
    age = available.loc[available.name.eq("age")]
    raw = age.loc[age.availability_scope.eq("integrated_raw")].iloc[0]
    effective = age.loc[age.availability_scope.eq("lagged_interval_effective")].iloc[0]
    selected = age.loc[age.availability_scope.eq("selected_model_complete_cases")].iloc[0]
    assert raw.n_available == 0 and effective.n_available_intervals == 117
    assert effective.n_eligible_intervals == 120 and effective.n_available_patients == 39
    assert selected.included_in_formula and selected.n_available_intervals == len(data) == meta["n_complete_cases"]
    assert data.age.max() < 999
    for name in ("age", "sex"):
        assert intervals.groupby("patient_id")[name].nunique(dropna=True).le(1).all()


def test_duration_decision_is_explicit_without_proxy_or_upstream_derivation():
    source = episodes().drop(columns="disease_duration")
    source["time_since_first_visit"] = 10
    registry = mod.resolve_predictors(mod.build_predictor_registry(), source, source)
    report = mod.duration_source_report(source.loc[source.is_clinical_baseline], registry)
    assert report["duration_source_status"] == "not_available"
    assert report["duration_exclusion_reason"] == "no_verified_canonical_diagnosis_date"
    assert report["duration_n_available"] == 0
    source["dx_date"] = pd.Timestamp("2019-01-01")
    report = mod.duration_source_report(source.loc[source.is_clinical_baseline], registry)
    assert report["duration_source_status"] == "upstream_change_proposed_pending_approval"
    assert "disease_duration" not in source


@pytest.mark.parametrize("failure", ["warning", "covariance", "standard_error"])
def test_cox_invalid_attempt_is_diagnosed_and_never_reselects_or_publishes(monkeypatch, failure):
    import lifelines
    from lifelines.exceptions import ConvergenceWarning
    task = event_task()
    data, meta = mod.prepare_model(task, 120)
    calls = []
    class FakeCox:
        def __init__(self, **kwargs):
            assert kwargs["penalizer"] == 0
        def fit(self, design, **kwargs):
            calls.append(design)
            columns = [c for c in design if c not in {"time_years", "event"}]
            self.summary = pd.DataFrame({"coef": .1, "se(coef)": .2, "p": .3, "exp(coef)": 1.1,
                "exp(coef) lower 95%": .8, "exp(coef) upper 95%": 1.4}, index=columns)
            self.variance_matrix_ = pd.DataFrame(np.eye(len(columns))*(-1 if failure == "covariance" else 1), index=columns, columns=columns)
            if failure == "standard_error":
                self.summary.loc["x", "se(coef)"] = np.nan
            if failure == "warning":
                warnings.warn("synthetic separation convergence failure", ConvergenceWarning)
            return self
    monkeypatch.setattr(lifelines, "CoxPHFitter", FakeCox)
    attempts = []
    row = mod.fit_task(task, data, meta, attempts)
    assert len(calls) == len(attempts) == 1
    assert row["model_status"] == "model_failed" and not row["valid_for_inference"]
    assert row["selected_adjustment_level"] == meta["selected_adjustment_level"] == "full"
    assert row["formula"] == meta["formula"] and not row["fallback_used"]
    for name in ("estimate", "standard_error", "ci95_low", "ci95_high", "p_value", "q_value"):
        assert pd.isna(row[name])
    diagnostic = attempts[0]
    assert diagnostic["cox_failure_reason"] == {"warning": "cox_convergence_warning", "covariance": "invalid_covariance",
        "standard_error": "invalid_coefficient_inference"}[failure]
    assert diagnostic["cox_raw_coefficients"] and diagnostic["cox_covariance"]
    if failure == "warning":
        assert diagnostic["cox_warning_types"] == "ConvergenceWarning"
        assert "separation" in diagnostic["cox_warning_messages"]


def test_complete_separation_cox_is_not_rescued_by_dropping_adjustment():
    task = event_task()
    task.spec = mod.replace(task.spec, variable_type="continuous")
    task.data["y"] = task.data.pred__test
    data, meta = mod.prepare_model(task, 120)
    assert meta["selected_adjustment_level"] == "full"
    attempts = []
    row = mod.fit_task(task, data, meta, attempts)
    assert not row["valid_for_inference"] and pd.isna(row["estimate"])
    assert row["selected_adjustment_level"] == "full"
    assert "x" in attempts[0]["separation_pattern_columns"]
    assert attempts[0]["cox_failure_reason"] in {"cox_convergence_warning", "cox_fit_exception"}


@pytest.mark.parametrize("problem,reason", [("collinearity", "rank_deficient_design"), ("extreme_scale", "rank_deficient_design"),
                                          ("nonfinite", "nonfinite_design"), ("zero_time", "invalid_time_or_event_coding")])
def test_cox_design_failure_diagnostic_is_specific(problem, reason):
    task = event_task()
    data, meta = mod.prepare_model(task, 120)
    if problem == "collinearity":
        data["age"] = data.baseline_essdai
    elif problem == "extreme_scale":
        data["age"] *= 1e100
    elif problem == "nonfinite":
        data.loc[0, "age"] = np.inf
    else:
        data.loc[0, "time_years"] = 0
    attempts = []
    row = mod.fit_task(task, data, meta, attempts)
    assert row["cox_failure_reason"] == reason and row["model_status"] == "model_failed"
    assert row["selected_adjustment_level"] == meta["selected_adjustment_level"]
    assert pd.isna(row["estimate"]) and not row["valid_for_inference"]


def test_successful_cox_has_design_covariance_and_raw_qc_without_extra_attempts():
    task = event_task()
    data, meta = mod.prepare_model(task, 120)
    attempts = []
    row = mod.fit_task(task, data, meta, attempts)
    assert row["valid_for_inference"] and row["standard_error"] > 0
    assert len(attempts) == 1 and attempts[0]["cox_failure_reason"] == "none"
    assert attempts[0]["design_rank"] == attempts[0]["design_parameters"]
    assert attempts[0]["covariance_min_eigenvalue"] > 0 and attempts[0]["n_nonpositive_times"] == 0


def test_matched_domain_comparison_freezes_adjustment_uses_identical_keys_and_preserves_primary_results():
    rng = np.random.default_rng(823)
    source = episodes(n=40)
    for domain in mod.DOMAINS:
        source[mod.step13.DOMAIN_ALIASES[domain][0]] = rng.integers(0, 2, len(source))
    source[mod.ESSDAI] = sum(source[mod.step13.DOMAIN_ALIASES[d][0]]*w for d, w in mod.canonical_domain_weights().items())
    # Preserve total, but make one FROM visit per patient incomplete: no provenance assertion.
    source.loc[source.clinical_visit_number.eq(2), mod.step13.DOMAIN_ALIASES["pulmonary"][0]] = np.nan
    df, base, intervals, risks, registry = datasets(source)
    tasks = [task for task in mod.build_analysis_tasks(df, base, intervals, risks, registry)
        if task.kind == "lagged" and task.spec.family == "essdai_domain" and task.sensitivity in {"none", "domain_from_other_domains_adjusted"}]
    prepared = [(task, *mod.prepare_model(task, 40)) for task in tasks]
    attempts = []
    results = pd.DataFrame([mod.fit_task(task, data, meta, attempts) for task, data, meta in prepared])
    saved = results.copy(deep=True)
    compared = mod.build_domain_sensitivity_comparison(prepared, results, 10, attempts)
    pd.testing.assert_frame_equal(results, saved)
    assert len(compared) == 36
    matched = compared.loc[compared.comparison_variant.ne("primary_full_sample")]
    for _, pair in matched.groupby("predictor"):
        assert pair.n_intervals.nunique() == pair.matched_sample_sha256.nunique() == pair.adjustment_covariates.nunique() == 1
        assert pair.n_intervals.iloc[0] == 80
    assert all(row["sensitivity"] != "none" for row in attempts if row["attempt_scope"] == "domain_sensitivity_comparison")
    assert compared.loc[~compared.valid_for_inference, "n_fitted_intervals"].isna().all()


def test_code_fingerprint_identifies_dirty_code_without_exporting_diff_or_secrets(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True)
    git("init")
    source = tmp_path/"analysis.py"
    source.write_text("value = 1\n")
    git("add", "analysis.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    clean = mod.runtime_code_fingerprint(tmp_path)
    assert clean["reproducibility_status"] == "fully_reproducible"
    source.write_text("value = 'sensitive-fixture-string'\n")
    dirty = mod.runtime_code_fingerprint(tmp_path)
    assert dirty["reproducibility_status"] == "dirty_tracked_code_with_hash"
    assert clean["git_commit"] == dirty["git_commit"] and clean["code_tree_sha256"] != dirty["code_tree_sha256"]
    assert clean["git_diff_sha256"] != dirty["git_diff_sha256"]
    assert "sensitive-fixture-string" not in json.dumps(dirty)
    (tmp_path/"helper.py").write_text("untracked = 1\n")
    assert mod.runtime_code_fingerprint(tmp_path)["reproducibility_status"] == "dirty_untracked_code"


def test_run_comparison_checks_same_input_counts_and_keeps_dirty_reference_limitation():
    hashes = {"integrated": "a"*64, "baseline": "b"*64, "context": None, "progression": None}
    previous = {"input_sha256": hashes, "n_baseline_patients": 159, "n_episodes": 497, "git_worktree_dirty": True}
    current = {**previous, "dry_run": True}
    reference = {"status": "backed_up_reference", "metadata": previous, "results": pd.DataFrame()}
    compared = mod.build_run_comparison(reference, current, pd.DataFrame())
    assert compared.loc[compared.metric.eq("n_episodes"), "delta"].iloc[0] == 0
    with pytest.raises(ValueError, match="cohort/episode count changed"):
        mod.build_run_comparison(reference, {**current, "n_episodes": 498}, pd.DataFrame())


def test_six_failed_cox_never_enter_family_or_outcome_wide_bh():
    rows = [{"predictor": "supported-"+str(i), "hypothesis_group_id": "supported-"+str(i),
        "p_value": p, "fdr_family": "cox-organ-domains", "analysis_type": "baseline",
        "outcome": "new_domain_articular", "effect_measure": "HR", "sensitivity": "none",
        "primary_canonical": True, "fdr_eligibility": True, "valid_for_inference": True}
        for i, p in enumerate((.01, .03, .2))]
    for name in ("salivary_flow_unstimulated", "ocular_schirmer_min", "rf", "leukopenia", "domain_hematologic", "domain_biological"):
        rows.append({**rows[0], "predictor": name, "hypothesis_group_id": name,
            "p_value": np.nan, "valid_for_inference": False, "model_status": "model_failed"})
    family, qc = mod.apply_fdr_by_family(pd.DataFrame(rows))
    out, wide = mod.apply_outcome_wide_fdr(family)
    assert out.q_value.equals(out.q_value_family)
    assert out.q_value_outcome_wide.iloc[:3].tolist() == pytest.approx([.03, .045, .2])
    assert out.q_value.iloc[3:].isna().all() and out.q_value_outcome_wide.iloc[3:].isna().all()
    assert qc.n_valid_tests.sum() == wide.n_tests_bh.sum() == 3


def test_source_schema_audit_reads_canonical_registry_dictionary_without_inventing_provenance(tmp_path):
    source = episodes()
    registry = mod.resolve_predictors(mod.build_predictor_registry(), source, source)
    pulmonary_column = next(s.time_varying_column for s in registry if s.name == "domain_pulmonary")
    registry_path, dictionary_path = tmp_path/"10_variable_registry.csv", tmp_path/"11_variable_dictionary.csv"
    pd.DataFrame({"public_variable": [mod.ESSDAI, pulmonary_column, "unrelated"],
        "producer_script": ["official_total.py", "official_domains.py", "other.py"],
        "source": ["pop", "overlap", "other"]}).to_csv(registry_path, index=False)
    pd.DataFrame({"variable": ["dx_date", "disease_duration"], "description": ["diagnosis", "baseline duration"]}).to_csv(dictionary_path, index=False)
    inventory = {"step10_registry": {"status": "available", "path": str(registry_path)},
        "step11_dictionary": {"status": "available", "path": str(dictionary_path)},
        "step11_dictionary_tables": {"status": "source_unavailable"}}
    report = mod.build_source_schema_audit(source.loc[source.is_clinical_baseline], source, registry, inventory)
    assert len(report["upstream_column_producers"]) == 2
    assert report["dictionary_status"] == "inspected"
    assert report["dictionary_duration_variables"] == ["dx_date", "disease_duration"]
    assert not report["composition_provenance_manifest_present"]
    assert "synthetic-" not in json.dumps(report)

"""Temporal and inferential contracts for Primary Objective 3 associations."""
import importlib.util
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
    assert meta["adjustment_status"] == "prespecified_reduced"
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

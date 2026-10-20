"""Unit tests for canonical Step 13 endpoint definitions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

PATH=Path(__file__).parents[1]/"src/block_A/13_disease_activity_progression.py"
spec=importlib.util.spec_from_file_location("progression13",PATH)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def episodes(scores, baseline_score=None):
    dates=pd.to_datetime(["2020-01-01","2021-01-01","2022-01-01"][:len(scores)])
    return pd.DataFrame({"patient_id":["p"]*len(scores),"clinical_episode_id":[f"e{i}" for i in range(len(scores))],"clinical_anchor_date":dates,"clinical_baseline_date":[dates[0]]*len(scores),"is_clinical_baseline":[True]+[False]*(len(scores)-1),"essdai__total":scores})


def test_state_boundaries():
    assert [mod.classify_essdai_activity(x) for x in [0,4,5,13,14,40,None]] == ["Low","Low","Moderate","Moderate","High","High","Missing"]


def test_baseline_to_last_status():
    assert mod.baseline_to_last_status("Low","Moderate")=="worsened"
    assert mod.baseline_to_last_status("Moderate","High")=="worsened"
    assert mod.baseline_to_last_status("Moderate","Low")=="improved"
    assert mod.baseline_to_last_status("High","Moderate")=="improved"
    assert mod.baseline_to_last_status("Low","Low")=="stable"


def test_transient_worsening_is_retained():
    out=mod.patient_progression(episodes([3,7,3])).iloc[0]
    assert bool(out.ever_worsened) and out.status=="stable"


def test_km_risk_sets_exclude_wrong_baseline():
    low=episodes([3,7]); moderate=episodes([6,14]); moderate["patient_id"]="q"
    both=pd.concat([low,moderate],ignore_index=True)
    assert mod.build_tte_risk_set(both,"low").patient_id.tolist()==["p"]
    assert mod.build_tte_risk_set(both,"moderate").patient_id.tolist()==["q"]


def test_missing_middle_breaks_transition_chain():
    matrix,raw=mod.build_state_transitions(episodes([3,None,14]))
    assert raw.empty
    assert matrix.n_intervals.sum()==0


def test_no_baseline_substitution():
    assert mod.patient_progression(episodes([None,3])).empty


def longitudinal_fixture(n=24):
    rng = np.random.default_rng(731)
    rows = []
    for i in range(n):
        intercept = rng.normal(0, 1)
        for j, days in enumerate((0, 270, 620, 1390)):
            time = days / 365.25
            row = {"patient_id": f"p{i}", "clinical_episode_id": f"p{i}e{j}",
                "clinical_anchor_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=days),
                "clinical_baseline_date": pd.Timestamp("2020-01-01"),
                "clinical_baseline_episode_id": f"p{i}e0", "is_clinical_baseline": j == 0,
                "clinical_visit_number": j + 1, "clinical_visit": True,
                "time_since_clinical_baseline_days": days, mod.TIME: time,
                mod.ESSDAI: np.clip(6 + intercept + .2*time + rng.normal(0, .5), 0, 30)}
            for col in mod.ESSPRI.values():
                row[col] = np.clip(4 + .3*intercept + .1*time + rng.normal(0, .4), 0, 10)
            for domain in mod.DOMAINS:
                row[f"essdai__{domain}_ordinal_score"] = rng.integers(0, 4)
            rows.append(row)
    return pd.DataFrame(rows)


def fake_fit(variance=1., converged=True, gee=False):
    params = pd.Series({"Intercept": 2., "time": .4})
    cov = pd.DataFrame([[.3, .02], [.02, .04]], index=params.index, columns=params.index)
    result = SimpleNamespace(params=params, bse=pd.Series({"Intercept": .5, "time": .2}),
        pvalues=pd.Series({"Intercept": .01, "time": .04}), converged=converged,
        cov_re=np.array([[variance]]), cov_params=lambda: cov)
    if not gee:
        result.fe_params = params
    return result


@pytest.mark.parametrize(("variance", "warnings", "converged"), [
    (0., [], True), (1e-10, [], True), (np.nan, [], True),
    (1., ["Random effects covariance is singular"], True),
    (1., ["MLE may be on the boundary"], True),
    (1., ["Hessian matrix is not positive definite"], True), (1., [], False),
])
def test_mixedlm_acceptance_rejects_unreliable_fits(variance, warnings, converged):
    assessment = mod.assess_mixedlm_fit(fake_fit(variance, converged), warnings)
    assert not assessment["valid_for_primary_inference"]
    assert assessment["invalid_reason"]


@pytest.mark.parametrize(("variance", "selected"), [(0., "GEE Gaussian"), (1., "MixedLM")])
def test_continuous_model_selection_and_attempt_retention(monkeypatch, variance, selected):
    seen = []
    def mixed_formula(*args, **kwargs):
        seen.append(("MixedLM", kwargs["groups"]))
        return SimpleNamespace(fit=lambda **kwargs: fake_fit(variance))
    def gee_formula(*args, **kwargs):
        seen.append(("GEE", kwargs["groups"]))
        return SimpleNamespace(fit=lambda **kwargs: fake_fit(gee=True))
    monkeypatch.setattr(mod.MixedLM, "from_formula", mixed_formula)
    monkeypatch.setattr(mod.GEE, "from_formula", gee_formula)
    for col in [mod.ESSDAI, *mod.ESSPRI.values()]:
        attempts = []
        row, fit = mod.fit_continuous_model(longitudinal_fixture(), col, col, attempts)
        assert row["model_used"] == selected
        assert row["primary_model_valid"]
        assert row["fallback_used"] == (variance == 0)
        assert row["mixedlm_singular"] == (variance == 0)
        assert len(attempts) == 2
        if variance == 0:
            assert attempts[0]["interpretation_status"] == "not_interpretable"
            assert attempts[0]["model_status"] != "ok"
            assert "singular" in row["fallback_reason"]
    assert all(group == "patient_id" for _, group in seen)


def test_failed_gee_is_not_reported_as_primary_valid(monkeypatch):
    monkeypatch.setattr(mod.MixedLM, "from_formula", lambda *a, **k: SimpleNamespace(fit=lambda **k: fake_fit(0)))
    monkeypatch.setattr(mod.GEE, "from_formula", lambda *a, **k: SimpleNamespace(fit=lambda **k: fake_fit(converged=False, gee=True)))
    row, fit = mod.fit_continuous_model(longitudinal_fixture(), mod.ESSDAI, "ESSDAI")
    assert fit is None and row["selected_model"] == "none"
    assert not row["primary_model_valid"]
    assert row["interpretation_status"] == "descriptive_only"


def test_ordinal_gee_patient_clustering_and_fdr(monkeypatch):
    data = longitudinal_fixture()
    real = mod.OrdinalGEE
    calls = []
    def recorded(*args, **kwargs):
        calls.append(kwargs["groups"])
        # No added intercept; threshold intercepts are created by OrdinalGEE.
        assert list(args[1].columns) == [mod.TIME]
        return real(*args, **kwargs)
    monkeypatch.setattr(mod, "OrdinalGEE", recorded)
    resolved = {"articular": "essdai__articular_ordinal_score"}
    models, _, support = mod.domain_analyses(data, resolved)
    row = models.set_index("domain").loc["articular"]
    assert row.model == "OrdinalGEE" and row.valid_for_inference
    assert row.cluster_variable == "patient_id"
    assert row.q_value == pytest.approx(row.p_value)  # only valid family member
    expected = data.sort_values(["patient_id", mod.TIME], kind="stable")
    assert list(calls[0]) == expected.patient_id.tolist()
    assert support.set_index("domain").loc["articular", "n_patients_repeated"] == 24
    assert models.loc[models.model_status.eq("column_unavailable"), "q_value"].isna().all()


def test_ordinal_failure_uses_patient_clustered_binary_gee(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("forced ordinal instability")
    monkeypatch.setattr(mod, "OrdinalGEE", unavailable)
    models, _, _ = mod.domain_analyses(longitudinal_fixture(), {"articular": "essdai__articular_ordinal_score"})
    row = models.set_index("domain").loc["articular"]
    assert row.model == "GEE binomial" and row.fallback_used
    assert row.cluster_variable == "patient_id" and row.valid_for_inference
    assert row.effect_scale == "odds ratio active vs inactive per year"
    assert row.ci95_low > 0


def test_constant_domain_is_descriptive_not_in_fdr():
    data = longitudinal_fixture()
    data["essdai__glandular_ordinal_score"] = 0
    models, _, _ = mod.domain_analyses(data, {"glandular": "essdai__glandular_ordinal_score"})
    row = models.set_index("domain").loc["glandular"]
    assert row.model_status == "descriptive_only" and pd.isna(row.q_value)


def test_all_public_domains_resolve_including_glandular():
    resolved = mod.resolve_domains(longitudinal_fixture())
    assert len(resolved) == 12
    assert resolved["glandular"] == "essdai__glandular_ordinal_score"
    qc = mod.domain_contract_qc(longitudinal_fixture(), resolved)
    assert qc.available.all() and qc.status.eq("pass").all()


def window_fixture():
    return pd.DataFrame({"patient_id": ["p"]*6,
        "clinical_episode_id": ["official", "other_zero", "early", "closest", "late", "five_plus"],
        "clinical_anchor_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2021-02-01", "2021-07-01", "2021-09-01", "2026-01-01"]),
        "is_clinical_baseline": [True, False, False, False, False, False],
        mod.TIME: [0., 0., 1.1, 1.5, 1.8, 6.], mod.ESSDAI: [1., 99., 2., np.nan, 3., 4.],
        "essdai__glandular_ordinal_score": [0, 3, 1, 2, 3, 0]})


def test_windows_use_only_official_baseline_and_one_patient():
    selected = mod.select_one_episode_per_patient_per_window(window_fixture())
    assert selected.loc[selected.time_window.eq("baseline"), "clinical_episode_id"].tolist() == ["official"]
    assert selected.loc[selected.time_window.eq(">1-2y"), "clinical_episode_id"].tolist() == ["closest"]
    assert not selected.duplicated(["patient_id", "time_window"]).any()
    assert "other_zero" not in selected.clinical_episode_id.tolist()
    # Selection does not favor available outcomes over missing ones.
    missing = mod.missingness_by_time(selected, {"ESSDAI": mod.ESSDAI}).set_index("time_window")
    assert missing.loc[">1-2y", "n_patients_missing"] == 1
    assert missing.loc[">1-2y", "pct_missing"] == 100
    heat = mod.domain_activity_by_time(selected, {"glandular": "essdai__glandular_ordinal_score"})
    row = heat.query("domain == 'glandular' and time_window == 'baseline'").iloc[0]
    assert row.n_evaluable == 1 and row.pct_active == 0


def test_window_selection_is_independent_of_input_order():
    a = mod.select_one_episode_per_patient_per_window(window_fixture())
    b = mod.select_one_episode_per_patient_per_window(window_fixture().sample(frac=1, random_state=1))
    assert a.clinical_episode_id.tolist() == b.clinical_episode_id.tolist()


def test_sparse_km_median_not_reliably_estimable_and_rate_naming():
    tte = pd.DataFrame({"time_years": [.5, .5, 2.], "event": [False, False, True]})
    curve, stats = mod.km_estimate(tte)
    assert curve.iloc[-1].survival == 0  # mathematical median exists at a risk set of 1
    assert stats["median_time_to_event"] == "NR"
    assert stats["median_time_to_event_status"] == "not_reliably_estimable"
    row = mod.tte_summary(tte, "moderate_to_high")
    assert "event_rate" not in row
    assert row["observed_event_proportion"] == pytest.approx(1/3)
    assert row["event_rate_per_100_person_years"] == pytest.approx(100/3)
    assert row["interpretation_status"] == "descriptive_sparse_events"
    assert pd.isna(row["event_free_5y"])


def test_tte_keeps_event_interval_bounds():
    risk = mod.build_tte_risk_set(episodes([3, 4, 7]), "low").iloc[0]
    assert risk.last_non_event_time_years > 0
    assert risk.event_time_years == risk.time_years
    assert risk.last_non_event_time_years < risk.event_time_years


def test_attrition_reconciles_to_exact_model_cohort():
    data = longitudinal_fixture(6)
    data.loc[data.patient_id.eq("p0"), mod.ESSDAI] = np.nan
    data.loc[data.patient_id.eq("p1") & ~data.is_clinical_baseline, mod.ESSDAI] = np.nan
    data.loc[data.patient_id.eq("p2"), "clinical_anchor_date"] = pd.Timestamp("2020-01-01")
    data.loc[data.patient_id.eq("p3") & ~data.is_clinical_baseline, mod.TIME] = np.nan
    flow = mod.progression_attrition(data, {"ESSDAI": mod.ESSDAI}, data.patient_id.unique())
    assert (flow.n_remaining.iloc[:-1].to_numpy() - flow.n_excluded_at_step.iloc[1:].to_numpy()
            == flow.n_remaining.iloc[1:].to_numpy()).all()
    assert flow.n_remaining.iloc[-1] == 2
    assert flow.n_remaining.iloc[-1] == mod.continuous_model_data(data, mod.ESSDAI).patient_id.nunique()
    assert "excluded_same_date_duplicates" not in flow.step.tolist()


def test_same_date_qc_retains_canonical_episodes():
    data = window_fixture()
    qc = mod.same_date_episode_qc(data).iloc[0]
    assert qc.n_same_date_episode_rows == 2
    assert qc.n_same_date_patient_dates == qc.n_patients_affected == qc.n_zero_interval_pairs == 1
    assert len(data) == 6


def test_population_predictions_use_fixed_effect_covariance():
    fit = fake_fit()
    fit.random_effects = {"p": np.array([999.])}
    grid = np.array([0., 2.])
    pred = mod.marginal_predictions(fit, grid)
    assert pred.prediction.tolist() == pytest.approx([2., 2.8])
    design = np.array([1., 2.])
    expected_se = np.sqrt(design @ fit.cov_params().to_numpy() @ design)
    assert pred.ci95_high.iloc[1] == pytest.approx(2.8 + norm.ppf(.975)*expected_se)


def test_sensitivities_are_attempted_when_supported(monkeypatch):
    calls = []
    def formula(formula, **kwargs):
        calls.append((formula, kwargs["re_formula"]))
        result = fake_fit()
        result.aic, result.bic, result.llf = 20., 25., -8.
        result.conf_int = lambda: pd.DataFrame({0: result.params - 1., 1: result.params + 1.})
        if kwargs["re_formula"] == "~time":
            result.cov_re = np.array([[1., .1], [.1, .4]])
        if "I(time ** 2)" in formula:
            result.params["I(time ** 2)"] = .01
            result.pvalues["I(time ** 2)"] = .2
            result.fe_params = result.params
            cov = pd.DataFrame(np.eye(3)*.2, index=result.params.index, columns=result.params.index)
            result.cov_params = lambda: cov
        return SimpleNamespace(fit=lambda **k: result)
    monkeypatch.setattr(mod.MixedLM, "from_formula", formula)
    results = mod.continuous_sensitivities(longitudinal_fixture(), mod.ESSDAI, "ESSDAI")
    assert ("y ~ time", "~time") in calls
    assert ("y ~ time + I(time ** 2)", "1") in calls
    assert results.model_status.eq("ok").all()
    assert results.set_index("analysis").loc["quadratic_time", "ic_comparable"]


def test_step13_generates_all_outputs_and_qc(tmp_path):
    data = longitudinal_fixture()
    baseline = data.loc[data.is_clinical_baseline]
    source = tmp_path/"integrated.parquet"
    basefile = tmp_path/"baseline.parquet"
    data.to_parquet(source, index=False)
    baseline.to_parquet(basefile, index=False)
    mod.main(SimpleNamespace(integrated=source, baseline=basefile, output_root=tmp_path,
                             overwrite=False, dry_run=False))
    tables = tmp_path/"outputs/tables/blockA"/mod.STEM
    qc = tmp_path/"outputs/qc/blockA"/mod.STEM
    figures = tmp_path/"outputs/figures/blockA"/mod.STEM
    old_figures = {"13_essdai_trajectory.pdf", "13_essdai_state_transition_heatmap.pdf",
        "13_km_low_to_ge5.pdf", "13_km_moderate_to_high.pdf", "13_essdai_domain_activity_heatmap.pdf",
        "13_essdai_domain_effects_forest.pdf", "13_esspri_trajectory_total.pdf", "13_esspri_trajectory_components.pdf"}
    new_figures = {"13_essdai_delta_by_baseline_state.pdf", "13_essdai_ever_vs_last_worsening.pdf",
        "13_essdai_observed_pattern_aggregate.pdf",
        "13_essdai_individual_trajectories_delta3.pdf", "13_essdai_individual_trajectories_delta5.pdf",
        "13_essdai_individual_trajectories_ever_worsening_delta3.pdf",
        "13_essdai_individual_trajectories_ever_worsening_delta5.pdf"}
    v2_figures = {"13_essdai_observed_pattern_aggregate_v2.pdf", "13_essdai_index_support_v2.pdf",
        "13_essdai_pattern_by_index_source_v2.pdf", "13_essdai_pattern_by_index_state_v2.pdf",
        "13_essdai_pattern_by_baseline_pop_v2.pdf", "13_essdai_individual_trajectories_delta3_v2.pdf",
        "13_essdai_individual_trajectories_delta5_v2.pdf"}
    assert {p.name for p in figures.glob("*.pdf")} == old_figures | new_figures | v2_figures
    analytic = tmp_path/"data/analytic/blockA"/mod.STEM
    endpoints = pd.read_parquet(analytic/"13_essdai_patient_delta_endpoints.parquet")
    patterns = pd.read_parquet(analytic/"13_essdai_patient_observed_patterns.parquet")
    assert len(patterns) == 48 and not patterns.duplicated(["patient_id", "threshold_abs"]).any()
    assert not pd.read_csv(qc/"13_essdai_observed_pattern_qc.csv").status.eq("fail").any()
    assert len(endpoints) == 24 and endpoints.patient_id.is_unique
    assert set(mod.DELTA_BASE_COLUMNS).issubset(endpoints.columns)
    assert {"last_status_pm3", "ever_worsened_ge5", "first_observed_worsening_ge3_date"}.issubset(endpoints.columns)
    for name in ("delta_threshold_summary", "worsening_by_baseline_state", "ever_vs_last_reconciliation",
                 "baseline_last_state_by_delta", "observation_opportunity_sensitivity", "baseline_change_opportunity"):
        assert not pd.read_csv(tables/f"13_essdai_{name}.csv").empty
    assert pd.read_csv(qc/"13_essdai_refinement_qc.csv").status.eq("pass").all()
    assert pd.read_csv(qc/"13_essdai_delta_invariants_qc.csv").status.eq("pass").all()
    assert pd.read_csv(qc/"13_essdai_delta_eligibility_qc.csv").n_patients_evaluable.iloc[0] == 24
    assert all(p.stat().st_size > 1000 for p in figures.glob('*.pdf'))
    primary = pd.read_csv(tables/"13_essdai_longitudinal_model.csv")
    assert not (primary.singular & primary.primary_model_valid).any()
    models = pd.read_csv(tables/"13_essdai_domain_models.csv")
    assert len(models) == 12 and models.cluster_variable.eq("patient_id").all()
    assert "ordinal logistic" not in models.model.tolist()
    assert models.loc[~models.valid_for_inference, "q_value"].isna().all()
    sensitivity = pd.read_csv(tables/"13_essdai_model_sensitivity.csv")
    assert set(sensitivity.analysis) >= {"MixedLM random intercept", "GEE Gaussian exchangeable", "random_slope", "quadratic_time"}
    missing = pd.read_csv(qc/"13_progression_missingness_by_time.csv")
    assert missing.loc[missing.time_window.eq("baseline"), "n_patients_eligible"].eq(24).all()
    assert pd.read_csv(qc/"13_progression_domain_contract_qc.csv").available.all()
    attempts = pd.read_csv(qc/"13_progression_model_qc.csv")
    assert not (attempts.singular & attempts.model_status.eq("ok")).any()
    for name in ("low_to_ge5", "moderate_to_high"):
        assert "event_rate" not in pd.read_csv(tables/f"13_tte_{name}.csv")
    # CLI defaults must regenerate a completed run, replacing stale results.
    primary_path = tables/"13_essdai_longitudinal_model.csv"
    primary_path.write_text("stale_output\n", encoding="utf-8")
    rerun_args = mod.parse_args(["--integrated", str(source), "--baseline", str(basefile),
                                "--output-root", str(tmp_path)])
    mod.main(rerun_args)
    regenerated = pd.read_csv(primary_path)
    assert regenerated.outcome.tolist() == ["ESSDAI total"]
    assert regenerated.n_patients.tolist() == [24]
    contents = primary_path.read_text()
    with pytest.raises(FileExistsError, match="overwrite"):
        mod.main(mod.parse_args(["--integrated", str(source), "--baseline", str(basefile),
                                 "--output-root", str(tmp_path), "--no-overwrite"]))
    assert primary_path.read_text() == contents


def test_overwrite_cli_defaults_and_opt_out():
    assert mod.parse_args([]).overwrite is True
    assert mod.parse_args(["--overwrite"]).overwrite is True
    assert mod.parse_args(["--no-overwrite"]).overwrite is False


def test_missing_domain_hard_fails_with_actionable_qc(tmp_path):
    data = longitudinal_fixture().drop(columns="essdai__glandular_ordinal_score")
    source = tmp_path/"integrated.parquet"
    basefile = tmp_path/"baseline.parquet"
    data.to_parquet(source, index=False)
    data.loc[data.is_clinical_baseline].to_parquet(basefile, index=False)
    with pytest.raises(ValueError, match="11/12.*glandular"):
        mod.main(SimpleNamespace(integrated=source, baseline=basefile, output_root=tmp_path,
                                 overwrite=False, dry_run=False))
    qc = pd.read_csv(tmp_path/"outputs/qc/blockA"/mod.STEM/"13_progression_domain_contract_qc.csv")
    assert qc.set_index("domain").loc["glandular", "status"] == "missing"


@pytest.mark.parametrize("delta", [2, 3, 4, 5, -2, -3, -4, -5])
def test_absolute_change_inclusive_boundaries(delta):
    row = mod.build_patient_absolute_change_endpoints(episodes([8, 8+delta])).iloc[0]
    assert row.delta_essdai_last == delta
    for k in (3, 5):
        assert row[f"last_worsened_ge{k}"] == (delta >= k)
        assert row[f"last_improved_ge{k}"] == (delta <= -k)
        assert row[f"last_stable_pm{k}"] == (-k < delta < k)
        assert row[f"ever_worsened_ge{k}"] == row[f"last_worsened_ge{k}"]
        assert row[f"ever_improved_ge{k}"] == row[f"last_improved_ge{k}"]


@pytest.mark.parametrize("scores,ever3,ever5,last3,last5", [
    ([2, 6], True, False, True, False), ([6, 9], True, False, True, False),
    ([3, 7, 3], True, False, False, False), ([8, 4, 12], True, False, True, False),
    ([12, 4], False, False, False, False), ([2, 7, 2], True, True, False, False),
    ([6, 11], True, True, True, True), ([3, 3], False, False, False, False),
])
def test_absolute_change_manual_examples_and_state_independence(scores, ever3, ever5, last3, last5):
    data = episodes(scores)
    row = mod.build_patient_absolute_change_endpoints(data).iloc[0]
    legacy = mod.patient_progression(data).iloc[0]
    assert [row.ever_worsened_ge3, row.ever_worsened_ge5, row.last_worsened_ge3, row.last_worsened_ge5] == [ever3, ever5, last3, last5]
    assert legacy.delta_essdai == row.delta_essdai_last
    if scores in ([6, 9], [6, 11]):
        assert legacy.status == "stable" and not legacy.ever_worsened
        assert row.last_worsened_ge3
    if scores == [8, 4, 12]:
        assert row.ever_improved_ge3 and row.ever_worsened_ge3
        assert row.first_observed_improvement_ge3_date == pd.Timestamp("2021-01-01")
        assert row.first_observed_worsening_ge3_date == pd.Timestamp("2022-01-01")
    if ever3 and not last3:
        assert row.not_worsened_at_last_after_ever_ge3
        assert not row.still_worsened_at_last_ge3
    if not ever5:
        assert pd.isna(row.first_observed_worsening_ge5_date)


def absolute_fixture():
    rows = []
    for i, scores in enumerate(([2, 7, 2], [6, 9], [14, 9], [8, 4, 12])):
        g = episodes(scores)
        g["patient_id"] = f"p{i}"
        g["clinical_episode_id"] = [f"p{i}e{j}" for j in range(len(g))]
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def delta_products(data):
    endpoints = mod.build_patient_absolute_change_endpoints(data)
    return (endpoints, mod.patient_progression(data), mod.summarize_absolute_change_endpoints(endpoints),
            mod.reconcile_absolute_change_endpoints(endpoints), mod.absolute_change_state_crosstab(endpoints),
            mod.summarize_worsening_by_baseline_state(endpoints))


def test_absolute_change_patient_denominators_and_tables():
    data = absolute_fixture()
    endpoints, legacy, summary, reconciliation, crosstab, compact = delta_products(data)
    assert len(endpoints) == 4 and endpoints.patient_id.is_unique
    last = summary.query("stratum == 'Overall' and threshold_abs == 3 and assessment == 'baseline_to_last'").set_index("direction")
    assert last.n_patients_evaluable.eq(4).all()
    assert last.n_patients_event.to_dict() == {"worsening": 2, "improvement": 1, "stable": 1}
    assert last.pct_patients.sum() == 100
    moderate_ever = summary.query("stratum == 'Moderate' and threshold_abs == 3 and assessment == 'ever_observed'")
    assert moderate_ever.n_patients_evaluable.eq(2).all()
    assert moderate_ever.pct_patients.sum() == 150  # independent marginal outcomes
    overall_reconcile = reconciliation.query("stratum == 'Overall' and threshold_abs == 5").set_index("category")
    assert overall_reconcile.n_patients_event.to_dict() == {"never_ge5": 3, "ever_ge5_not_last": 1, "ever_ge5_and_last": 0}
    assert crosstab.groupby("threshold_abs").n_patients.sum().eq(4).all()
    assert compact.query("baseline_state == 'Moderate' and threshold_abs == 3").n_last_worsened.iloc[0] == 2
    checks = mod.absolute_change_invariants_qc(data, endpoints, legacy, summary, reconciliation, crosstab, compact)
    assert checks.status.eq("pass").all() and checks.n_failures.eq(0).all()
    assert summary.loc[summary.n_patients_evaluable.gt(0), "ci95_low"].between(0, 100).all()
    assert summary.loc[summary.n_patients_evaluable.gt(0), "ci95_high"].between(0, 100).all()


def test_absolute_change_order_tied_dates_and_missing_last():
    data = episodes([2, 9, 4])
    data.loc[2, "clinical_anchor_date"] = data.loc[1, "clinical_anchor_date"]
    missing = data.iloc[[2]].assign(clinical_episode_id="e3", clinical_anchor_date=pd.Timestamp("2022-01-01"))
    missing[mod.ESSDAI] = np.nan
    data = pd.concat([data, missing], ignore_index=True)
    expected = mod.build_patient_absolute_change_endpoints(data)
    actual = mod.build_patient_absolute_change_endpoints(data.sample(frac=1, random_state=42))
    pd.testing.assert_frame_equal(actual, expected)
    row = actual.iloc[0]
    assert row.last_clinical_episode_id == "e2" and row.last_essdai == 4
    assert row.ever_worsened_ge5 and not row.last_worsened_ge3
    assert row.n_essdai_followup_assessments == 2 and row.n_distinct_essdai_followup_dates == 1
    qc = mod.absolute_change_eligibility_qc(data, actual).iloc[0]
    assert qc.n_same_date_followup_extra_assessments == 1


@pytest.mark.parametrize("scores", [[None, 3, 8], [3, None], [3]])
def test_absolute_change_excludes_missing_baseline_or_followup(scores):
    data = episodes(scores)
    result = mod.build_patient_absolute_change_endpoints(data)
    assert result.empty
    summary = mod.summarize_absolute_change_endpoints(result)
    assert summary.n_patients_evaluable.eq(0).all() and summary.pct_patients.isna().all()
    qc = mod.absolute_change_eligibility_qc(data, result).iloc[0]
    assert qc.n_baseline_patients == 1 and qc.n_patients_evaluable == 0
    assert qc.n_valid_baseline_without_followup == (scores[0] is not None)


def test_absolute_change_same_baseline_date_is_not_followup():
    data = episodes([3, 10, 6])
    data.loc[1, "clinical_anchor_date"] = data.clinical_baseline_date.iloc[0]
    assert mod.build_patient_absolute_change_endpoints(data.iloc[:2]).empty
    row = mod.build_patient_absolute_change_endpoints(data).iloc[0]
    assert row.n_essdai_followup_assessments == 1
    assert row.ever_worsened_ge3 and not row.ever_worsened_ge5
    assert row.first_observed_worsening_ge3_date == pd.Timestamp("2022-01-01")


def test_absolute_change_missing_middle_is_skipped_without_changing_state_intervals():
    data = episodes([2, None, 7])
    row = mod.build_patient_absolute_change_endpoints(data).iloc[0]
    assert row.ever_worsened_ge5 and row.n_essdai_followup_assessments == 1
    assert mod.build_state_transitions(data)[1].empty


@pytest.mark.parametrize("corruption", ["threshold_flag", "summary", "duplicate_id", "first_event_date"])
def test_absolute_change_qc_fails_explicitly(corruption):
    data = absolute_fixture()
    endpoints, legacy, summary, reconciliation, crosstab, compact = delta_products(data)
    if corruption == "threshold_flag":
        endpoints.loc[0, "ever_worsened_ge3"] = False
    elif corruption == "summary":
        summary.loc[0, "n_patients_event"] = 99
    elif corruption == "duplicate_id":
        endpoints.loc[1, "patient_id"] = endpoints.patient_id.iloc[0]
    else:
        endpoints.loc[0, "first_observed_worsening_ge5_date"] = pd.Timestamp("2019-01-01")
    with pytest.raises(ValueError, match="Absolute-change QC failed"):
        mod.absolute_change_invariants_qc(data, endpoints, legacy, summary, reconciliation, crosstab, compact)
    qc = mod.absolute_change_invariants_qc(data, endpoints, legacy, summary, reconciliation, crosstab, compact, hard_fail=False)
    assert qc.status.eq("fail").any() and qc.loc[qc.status.eq("fail"), "reason"].str.len().gt(0).all()


def test_absolute_change_empty_products_and_figures(tmp_path):
    data = episodes([None, 8])
    endpoints, legacy, summary, reconciliation, crosstab, compact = delta_products(data)
    checks = mod.absolute_change_invariants_qc(data, endpoints, legacy, summary, reconciliation, crosstab, compact)
    assert checks.status.eq("pass").all()
    mod.write_absolute_change_outputs(data, legacy, tmp_path, tmp_path, tmp_path, tmp_path)
    assert len(list(tmp_path.glob("*.pdf"))) == 6
    assert all(p.stat().st_size > 1000 for p in tmp_path.glob("*.pdf"))
    assert pd.read_parquet(tmp_path/"13_essdai_patient_delta_endpoints.parquet").empty


def test_dry_run_preserves_new_outputs(tmp_path):
    data = longitudinal_fixture(3)
    source, base = tmp_path/"source.parquet", tmp_path/"baseline.parquet"
    data.to_parquet(source, index=False)
    data.loc[data.is_clinical_baseline].to_parquet(base, index=False)
    figures = tmp_path/"outputs/figures/blockA"/mod.STEM
    figures.mkdir(parents=True)
    sentinel = figures/"13_essdai_individual_trajectories_ever_worsening_delta3.pdf"
    sentinel.write_bytes(b"existing output")
    mod.main(mod.parse_args(["--integrated", str(source), "--baseline", str(base), "--output-root", str(tmp_path), "--dry-run", "--no-overwrite"]))
    assert sentinel.read_bytes() == b"existing output"
    assert not list(tmp_path.rglob("13_essdai_patient_delta_endpoints.parquet"))
    assert not list(tmp_path.rglob("13_essdai_delta_threshold_summary.csv"))
    for name in ("observation_opportunity_sensitivity", "baseline_change_opportunity", "refinement_qc"):
        assert not list(tmp_path.rglob(f"13_essdai_{name}.csv"))


def refinement_products(endpoints):
    return (mod.summarize_observation_opportunity_sensitivity(endpoints),
            mod.summarize_baseline_change_opportunity(endpoints),
            mod.reconcile_absolute_change_endpoints(endpoints))


def capture_closed_figures(monkeypatch):
    figures = []
    close = mod.plt.close
    def capture(fig=None):
        if hasattr(fig, "axes"):
            figures.append(fig)
        close(fig)
    monkeypatch.setattr(mod.plt, "close", capture)
    return figures


@pytest.mark.parametrize("scores,k,category", [
    ([2, 6, 2], 3, "ever_not_last"), ([1, 7, 8], 5, "ever_and_last"),
    ([3, 4, 3], 3, "never_observed"), ([14, 20], 5, "ever_and_last"),
])
def test_ever_trajectory_category_and_observed_line(scores, k, category, tmp_path, monkeypatch):
    data = episodes(scores)
    endpoints = mod.build_patient_absolute_change_endpoints(data)
    assert mod.ever_worsening_categories(endpoints, k).tolist() == [category]
    figures = capture_closed_figures(monkeypatch)
    mod.plot_ever_worsening_trajectories(data, endpoints, k, tmp_path/"plot.pdf")
    fig = figures[-1]
    ax = fig.axes[mod.STATE_ORDER.index(endpoints.baseline_state.iloc[0])]
    lines = [line for line in ax.lines if line.get_marker() == "o"]
    assert len(lines) == 1
    np.testing.assert_array_equal(lines[0].get_ydata(), scores)
    expected_times = (data.clinical_anchor_date - data.clinical_baseline_date).dt.days / 365.25
    np.testing.assert_allclose(lines[0].get_xdata(), expected_times)
    assert lines[0].get_color() == mod.EVER_WORSENING_COLORS[category]
    assert len(fig.legends[0].texts) == 3  # legend includes absent categories
    assert len({ax.get_ylim() for ax in fig.axes}) == 1
    assert any("n=0" in text.get_text() for text in ax.texts)


def test_ever_trajectories_retain_ties_and_layer_events(tmp_path, monkeypatch):
    event = episodes([2, 9, 4])
    event.loc[2, "clinical_anchor_date"] = event.loc[1, "clinical_anchor_date"]
    never = episodes([3, 4, 3]).assign(patient_id="z-never")
    data = pd.concat([event, never], ignore_index=True)
    endpoints = mod.build_patient_absolute_change_endpoints(data)
    figures = capture_closed_figures(monkeypatch)
    mod.plot_ever_worsening_trajectories(data.sample(frac=1, random_state=10), endpoints, 3, tmp_path/"ties.pdf")
    lines = [line for line in figures[-1].axes[0].lines if line.get_marker() == "o"]
    assert [line.get_color() for line in lines] == [mod.EVER_WORSENING_COLORS[c] for c in ("never_observed", "ever_not_last")]
    np.testing.assert_array_equal(lines[1].get_ydata(), [2, 9, 4])
    assert lines[1].get_xdata()[1] == lines[1].get_xdata()[2]
    sensitivity, opportunity, reconciliation = refinement_products(endpoints)
    row = sensitivity.query("stratifier == 'followup_assessment_count' and baseline_state == 'Overall' and band == '2' and threshold_abs == 3").iloc[0]
    assert row.n_patients_evaluable == 2
    assert row.median_n_followup_assessments == 2
    assert row.median_n_distinct_followup_dates == 1.5
    assert mod.absolute_change_refinement_qc(endpoints, sensitivity, opportunity, reconciliation).status.eq("pass").all()


def test_sensitivity_boundaries_counts_wilson_and_primary_totals():
    endpoints = mod.build_patient_absolute_change_endpoints(absolute_fixture())
    endpoints["followup_years"] = [1.999, 2., 4.999, 5.]
    endpoints["n_essdai_followup_assessments"] = [1, 2, 3, 4]
    endpoints["n_distinct_essdai_followup_dates"] = [1, 2, 2, 4]
    bands = mod.observation_opportunity_bands(endpoints)
    assert bands["followup_duration"].tolist() == ["<2y", "2-<5y", "2-<5y", ">=5y"]
    assert bands["followup_assessment_count"].tolist() == ["1", "2", ">=3", ">=3"]
    sensitivity, opportunity, reconciliation = refinement_products(endpoints)
    assert len(sensitivity) == 48
    checks = mod.absolute_change_refinement_qc(endpoints, sensitivity, opportunity, reconciliation)
    assert checks.status.eq("pass").all()
    overall = sensitivity.query("baseline_state == 'Overall' and threshold_abs == 3")
    for _, g in overall.groupby("stratifier"):
        assert g.n_patients_evaluable.sum() == 4
        assert g.n_ever_worsened.sum() == 3
        assert g.n_last_worsened.sum() == 2
        assert g.n_ever_not_last.sum() == 1
    # Single-patient transient event: Wilson n=1, p=1, not a normal Wald interval.
    row = overall.query("stratifier == 'followup_duration' and band == '<2y'").iloc[0]
    assert row.pct_ever_worsened == 100
    assert row.ci95_low_ever == pytest.approx(20.6549314)
    assert row.ci95_high_ever == pytest.approx(100)
    assert row.pct_last_worsened == 0
    assert row.ci95_high_last == pytest.approx(79.3450686)
    empty = sensitivity.loc[sensitivity.n_patients_evaluable.eq(0)]
    assert empty.filter(regex="^(pct_|ci95_)").isna().all().all()
    assert set(sensitivity.support_note) == {"sparse", "not_estimable"}


def test_followup_duration_bands_are_configurable(monkeypatch):
    endpoints = mod.build_patient_absolute_change_endpoints(absolute_fixture())
    endpoints["followup_years"] = [0.5, 1., 2.999, 3.]
    monkeypatch.setattr(mod, "ESSDAI_FOLLOWUP_DURATION_BANDS_YEARS", (1., 3.))
    assert mod.observation_opportunity_bands(endpoints)["followup_duration"].tolist() == ["<1y", "1-<3y", "1-<3y", ">=3y"]


def test_baseline_opportunity_floor_ceiling_and_unchanged_primary_summaries():
    data = []
    for i, scores in enumerate(([2, 0], [4, 1], [5, 0], [14, 20], [120, 123], [121, 123])):
        data.append(episodes(scores).assign(patient_id=f"synthetic-{i}"))
    endpoints = mod.build_patient_absolute_change_endpoints(pd.concat(data, ignore_index=True))
    summary_before = mod.summarize_absolute_change_endpoints(endpoints)
    compact_before = mod.summarize_worsening_by_baseline_state(endpoints)
    sensitivity, opportunity, reconciliation = refinement_products(endpoints)
    low = opportunity.query("baseline_state == 'Low'").set_index("threshold_abs")
    assert low.loc[3, "n_can_improve_by_k"] == 1
    assert low.loc[3, "pct_last_improved_all"] == 50
    assert low.loc[3, "pct_last_improved_among_possible"] == 100
    assert low.loc[5, "n_can_improve_by_k"] == 0
    assert np.isnan(low.loc[5, "pct_last_improved_among_possible"])
    moderate = opportunity.query("baseline_state == 'Moderate' and threshold_abs == 5").iloc[0]
    assert moderate.n_can_improve_by_k == moderate.n_last_improved_all == 1
    high = opportunity.query("baseline_state == 'High'").set_index("threshold_abs")
    assert high.loc[3, "n_can_worsen_by_k"] == 2  # 14 and max-3; max-2 cannot
    assert high.loc[5, "n_can_worsen_by_k"] == 1
    assert high.loc[5, "n_last_worsened_all"] == 1
    assert mod.absolute_change_refinement_qc(endpoints, sensitivity, opportunity, reconciliation).status.eq("pass").all()
    pd.testing.assert_frame_equal(summary_before, mod.summarize_absolute_change_endpoints(endpoints))
    pd.testing.assert_frame_equal(compact_before, mod.summarize_worsening_by_baseline_state(endpoints))


@pytest.mark.parametrize("corruption", ["sensitivity", "opportunity", "reconciliation", "floor_flag", "ceiling_flag", "no_band", "invalid_visit_count", "distinct_dates"])
def test_refinement_qc_reports_corruption_without_identifiers(corruption):
    endpoints = mod.build_patient_absolute_change_endpoints(absolute_fixture())
    sensitivity, opportunity, reconciliation = refinement_products(endpoints)
    if corruption == "sensitivity":
        sensitivity.loc[0, "n_ever_worsened"] += 1
    elif corruption == "opportunity":
        opportunity.loc[0, "n_can_improve_by_k"] += 1
    elif corruption == "reconciliation":
        reconciliation.loc[0, "n_patients_event"] += 1
    elif corruption == "floor_flag":
        endpoints.loc[0, "ever_improved_ge3"] = True
    elif corruption == "ceiling_flag":
        endpoints.loc[0, "baseline_essdai"] = mod.ESSDAI_MAX_VALID
    elif corruption == "no_band":
        endpoints.loc[0, "followup_years"] = 0
    elif corruption == "invalid_visit_count":
        endpoints.loc[0, "n_essdai_followup_assessments"] = 0
    else:
        endpoints.loc[0, "n_distinct_essdai_followup_dates"] = 99
    qc = mod.absolute_change_refinement_qc(endpoints, sensitivity, opportunity, reconciliation, hard_fail=False)
    failed = qc.loc[qc.status.eq("fail")]
    assert not failed.empty and failed.n_failures.gt(0).all() and failed.reason.str.len().gt(0).all()
    assert set(qc.columns) == {"check", "status", "n_failures", "reason"}
    with pytest.raises(ValueError, match="refinement QC failed"):
        mod.absolute_change_refinement_qc(endpoints, sensitivity, opportunity, reconciliation)


def test_refinement_empty_schemas_and_aggregate_csv_privacy(tmp_path):
    empty = mod.build_patient_absolute_change_endpoints(episodes([None, 8]))
    populated = mod.build_patient_absolute_change_endpoints(absolute_fixture())
    for endpoints in (empty, populated):
        sensitivity, opportunity, reconciliation = refinement_products(endpoints)
        assert len(sensitivity) == 48 and len(opportunity) == 8
        assert list(sensitivity.columns) == list(mod.summarize_observation_opportunity_sensitivity(populated).columns)
        assert list(opportunity.columns) == list(mod.summarize_baseline_change_opportunity(populated).columns)
        qc = mod.absolute_change_refinement_qc(endpoints, sensitivity, opportunity, reconciliation)
        assert qc.status.eq("pass").all()
        if endpoints.empty:
            assert sensitivity.n_patients_evaluable.eq(0).all() and opportunity.n_evaluable.eq(0).all()
            assert sensitivity.filter(regex="^(pct_|ci95_)").isna().all().all()
            assert opportunity.filter(regex="^pct_").isna().all().all()
        for name, table in (("sensitivity", sensitivity), ("opportunity", opportunity), ("qc", qc)):
            table.to_csv(tmp_path/f"{name}.csv", index=False)
            reloaded = pd.read_csv(tmp_path/f"{name}.csv")
            assert not any(c in reloaded for c in ("patient_id", "clinical_episode_id", "baseline_date", "last_date"))
            assert not any(c.endswith("_date") for c in reloaded)
            assert not any(str(pid) in reloaded.to_csv(index=False) for pid in endpoints.patient_id)


@pytest.mark.parametrize("n_events", [0, 2, 10])
def test_ever_last_axis_dynamic_bounded_and_shared(n_events, tmp_path, monkeypatch):
    data = pd.concat([episodes([2, 5 if i < n_events else 2]).assign(patient_id=f"s{i}") for i in range(10)], ignore_index=True)
    summary = mod.summarize_absolute_change_endpoints(mod.build_patient_absolute_change_endpoints(data))
    figures = capture_closed_figures(monkeypatch)
    mod.plot_ever_vs_last_worsening(summary, tmp_path/"summary.pdf")
    fig = figures[-1]
    upper = fig.axes[0].get_ylim()[1]
    assert 100 * n_events / 10 <= upper <= 100
    if n_events <= 2:
        assert upper <= 30
    assert fig.axes[1].get_ylim() == fig.axes[0].get_ylim()
    assert all(p.get_height() <= upper for ax in fig.axes for p in ax.patches)


def test_no_overwrite_protects_refinement_outputs(tmp_path):
    tables = tmp_path/"outputs/tables/blockA"/mod.STEM
    tables.mkdir(parents=True)
    sentinel = tables/"13_essdai_observation_opportunity_sensitivity.csv"
    sentinel.write_text("protected aggregate output")
    args = mod.parse_args(["--output-root", str(tmp_path), "--no-overwrite"])
    with pytest.raises(FileExistsError, match="outputs exist"):
        mod.main(args)
    assert sentinel.read_text() == "protected aggregate output"

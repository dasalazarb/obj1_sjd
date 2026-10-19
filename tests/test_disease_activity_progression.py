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
        "13_essdai_individual_trajectories_delta3.pdf", "13_essdai_individual_trajectories_delta5.pdf"}
    assert {p.name for p in figures.glob("*.pdf")} == old_figures | new_figures
    analytic = tmp_path/"data/analytic/blockA"/mod.STEM
    endpoints = pd.read_parquet(analytic/"13_essdai_patient_delta_endpoints.parquet")
    assert len(endpoints) == 24 and endpoints.patient_id.is_unique
    assert set(mod.DELTA_BASE_COLUMNS).issubset(endpoints.columns)
    assert {"last_status_pm3", "ever_worsened_ge5", "first_observed_worsening_ge3_date"}.issubset(endpoints.columns)
    for name in ("delta_threshold_summary", "worsening_by_baseline_state", "ever_vs_last_reconciliation", "baseline_last_state_by_delta"):
        assert not pd.read_csv(tables/f"13_essdai_{name}.csv").empty
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
    assert len(list(tmp_path.glob("*.pdf"))) == 4
    assert all(p.stat().st_size > 1000 for p in tmp_path.glob("*.pdf"))
    assert pd.read_parquet(tmp_path/"13_essdai_patient_delta_endpoints.parquet").empty


def test_dry_run_preserves_new_outputs(tmp_path):
    data = longitudinal_fixture(3)
    source, base = tmp_path/"source.parquet", tmp_path/"baseline.parquet"
    data.to_parquet(source, index=False)
    data.loc[data.is_clinical_baseline].to_parquet(base, index=False)
    figures = tmp_path/"outputs/figures/blockA"/mod.STEM
    figures.mkdir(parents=True)
    sentinel = figures/"13_essdai_individual_trajectories_delta3.pdf"
    sentinel.write_bytes(b"existing output")
    mod.main(mod.parse_args(["--integrated", str(source), "--baseline", str(base), "--output-root", str(tmp_path), "--dry-run", "--no-overwrite"]))
    assert sentinel.read_bytes() == b"existing output"
    assert not list(tmp_path.rglob("13_essdai_patient_delta_endpoints.parquet"))
    assert not list(tmp_path.rglob("13_essdai_delta_threshold_summary.csv"))

"""Synthetic software evidence only; no real-cohort or institutional closure."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.block_A import explainwhy_inclusive as inc
from src.block_A import explainwhy_inclusive_models as models
from src.block_A import explainwhy_inclusive_run as runner
from src.block_A import explainwhy_inclusive_figures as figures
from src.block_A import observed_patterns_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("inclusive_step14", ROOT/"src/block_A/14_risk_factors_progression.py")
step14 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = step14
SPEC.loader.exec_module(step14)


def episodes(scores, days=None, pid="synthetic"):
    days = np.arange(len(scores))*365 if days is None else np.asarray(days)
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(days, unit="D")
    frame = pd.DataFrame(dict(patient_id=pid, clinical_episode_id=[f"e{i:03d}" for i in range(len(scores))],
        clinical_anchor_date=dates, clinical_baseline_date=dates[0], clinical_baseline_episode_id="e000",
        is_clinical_baseline=[True]+[False]*(len(scores)-1), clinical_visit=True,
        clinical_visit_number=np.arange(1, len(scores)+1), time_since_clinical_baseline_days=days,
        time_since_clinical_baseline_years=days/365.25, essdai__total=scores,
        demo__age_at_baseline=45., demo__sex="Female"))
    for d in step14.DOMAINS:
        frame[step14.step13.DOMAIN_ALIASES[d][0]] = 0.
    return frame


def build(scores=None, days=None, frame=None):
    raw = episodes(scores, days) if frame is None else frame
    baseline = raw.loc[raw.is_clinical_baseline].copy()
    audit, sequence = v2.build_observed_patterns_v2(raw, baseline)
    patterns = audit.loc[audit.n_distinct_essdai_dates.ge(2)].copy()
    clean, official = inc.reconcile_products(raw, baseline, patterns, audit, sequence)
    cohort = inc.index_cohort(audit)
    risks = {k: inc.first_event_riskset(cohort, sequence, k) for k in (3, 5)}
    return clean, official, cohort, sequence, patterns, risks


def all_parts(raw):
    clean, baseline, cohort, sequence, patterns, risks = build(frame=raw)
    registry = pd.DataFrame(columns=["public_variable", "producer_script", "temporal_scope"])
    predictors, specs, availability = inc.build_index_predictors(clean, baseline, cohort, step14, registry)
    intervals, _ = inc.build_lagged_intervals(clean, cohort, sequence, risks[3], specs, step14, registry)
    return clean, baseline, cohort, sequence, patterns, risks, registry, predictors, specs, intervals


@pytest.mark.parametrize("scores,source,event,censor", [
    ([2, 5, 7], v2.OFFICIAL, 1, 365),
    ([None, 2, 5], v2.FALLBACK, 1, 365),
    ([None, 4, 4], v2.FALLBACK, 0, 365),
    ([None, 6, 9], v2.FALLBACK, 1, 365),
    ([2, 2, None], v2.OFFICIAL, 0, 365),
])
def test_index_event_censor_and_two_clocks(scores, source, event, censor):
    raw = episodes(scores)
    before = raw.copy(deep=True)
    _, baseline, cohort, _, _, risks = build(frame=raw)
    r = risks[3].iloc[0]
    assert r.index_source == source and r.event == event
    assert r.event_or_censor_time_years == pytest.approx(censor/365.25)
    assert r.official_clinical_baseline_date == before.clinical_baseline_date.iloc[0]
    if event:
        assert r.last_known_non_event_date < r.first_event_date
    else:
        assert pd.isna(r.first_event_date)
    assert r.index_time_origin == "essdai_index_date" and r.interval_censored_onset
    pd.testing.assert_frame_equal(raw, before)
    assert baseline.clinical_baseline_date.iloc[0] == before.clinical_baseline_date.iloc[0]
    assert cohort.baseline_pop_official.iloc[0] == "Unclassifiable"


def test_fallback_exclusion_and_same_day_companion_not_an_index():
    *_, risks = build([None, 5])
    assert risks[3].empty
    _, _, cohort, sequence, _, risks = build([None, 20, 2, 5], [0, 0, 365, 730])
    assert cohort.essdai_index_score.iloc[0] == 2
    assert sequence[inc.SCORE].to_list() == [2, 5]
    assert risks[3].event.iloc[0] == 1
    assert risks[3].first_event_date.iloc[0] == pd.Timestamp("2021-12-31")


def test_risk_uses_scores_not_retrospective_labels_and_delta5_same_clock():
    _, _, cohort, sequence, patterns, risks = build([None, 4, 7, 9, 4])
    assert patterns.observed_pattern.eq("intermittent_observed").all()
    assert risks[3].event.iloc[0] == 1 and risks[5].event.iloc[0] == 1
    assert risks[3].first_event_date.iloc[0] < risks[5].first_event_date.iloc[0]
    assert risks[3].essdai_index_date.equals(risks[5].essdai_index_date)
    assert risks[5].last_known_non_event_date.iloc[0] == risks[3].first_event_date.iloc[0]


def test_same_date_selected_scores_legacy_bridge_and_whole_date_sensitivity():
    raw, _, cohort, sequence, _, risks = build([2, 8, 4, 5], [0, 365, 365, 730])
    assert sequence[inc.SCORE].to_list() == [2, 4, 5]
    legacy = step14.step13.build_patient_absolute_change_endpoints(raw)
    bridge = inc.legacy_bridge(raw, risks, legacy, step14)
    assert bridge.n_disagreements.gt(0).all() and bridge.n_unexplained.eq(0).all()
    assert risks[3].first_event_date.iloc[0] == pd.Timestamp("2021-12-31")
    excluded, seq = v2.build_observed_patterns_v2(raw, raw.loc[raw.is_clinical_baseline], True)
    sensitivity = inc.first_event_riskset(inc.index_cohort(excluded), seq)
    assert sensitivity.n_distinct_essdai_dates.iloc[0] == 2
    assert sensitivity.event.iloc[0] == 1
    legacy.loc[0, "ever_worsened_ge3"] = False
    with pytest.raises(ValueError, match="legacy endpoints"):
        inc.legacy_bridge(raw, risks, legacy, step14)


def test_missing_middle_visit_and_missing_predictor_stop_event_risk():
    raw = episodes([None, 2, None, 5, 2, 8])
    *_, risks, registry, predictors, specs, intervals = all_parts(raw)
    pairs = list(zip(intervals.from_clinical_episode_id, intervals.to_clinical_episode_id))
    assert ("e001", "e003") not in pairs
    missing = intervals.loc[intervals.to_clinical_episode_id.eq("e003")].iloc[0]
    assert not missing.outcome_evaluable and missing.at_risk_first_delta3
    assert intervals.loc[intervals.from_clinical_episode_id.ge("e003"), "at_risk_first_delta3"].eq(False).all()
    assert risks[3].first_event_date.iloc[0] == raw.clinical_anchor_date.iloc[3]
    assert predictors.index_predictor_lab_igg.isna().all()


@pytest.mark.parametrize("age,unit,scope,available", [(0, "g/L", "episode", True),
    (-1, "g/L", "episode", False), (30, "g/L", "episode", False),
    (0, None, "episode", False), (0, "g/L", "patient_consensus", False)])
def test_index_labs_require_date_units_registry_and_no_implicit_carry_forward(age, unit, scope, available):
    raw = episodes([None, 2, 5])
    raw["lab__igg__value"] = 12.
    raw["lab__igg__measurement_date"] = raw.clinical_anchor_date-pd.to_timedelta(age, unit="D")
    raw["lab__igg__unit"] = unit
    registry = pd.DataFrame([dict(public_variable="lab__igg__value", producer_script="registered_synthetic", temporal_scope=scope)])
    clean, baseline, cohort, *_ = build(frame=raw)
    out, _, _ = inc.build_index_predictors(clean, baseline, cohort, step14, registry)
    assert bool(out.index_predictor_lab_igg.notna().iloc[0]) == available
    if age == 30:
        window, _, _ = inc.build_index_predictors(clean, baseline, cohort, step14, registry, max_age_days=30)
        assert window.index_predictor_lab_igg.notna().all()
    assert out.age_reference.str.contains("not age at ESSDAI index").all()
    assert out.index_predictor_duration.isna().all()


def test_inclusive_official_counts_and_qc_without_fallback_attrition():
    raw = pd.concat([episodes([2, 5, 7], pid="official"), episodes([None, 2, 5, 7], pid="fallback")], ignore_index=True)
    clean, baseline, cohort, sequence, _, risks, registry, predictors, specs, intervals = all_parts(raw)
    assert len(risks[3]) == 2 and risks[3].index_source.eq(v2.FALLBACK).sum() == 1
    qc = inc.temporal_qc(clean, baseline, cohort, sequence, risks, intervals, predictors, specs, step14, registry)
    assert qc.status.eq("PASS").all()
    summary = inc.event_summary(risks)
    assert summary.loc[summary.index_source.eq("all"), "n_events"].to_list() == [2, 2]


def test_index_domain_missing_is_not_inactive_and_event_stop_ignores_from():
    raw = episodes([None, 2, None, 5, 2, 8])
    column = step14.step13.DOMAIN_ALIASES["articular"][0]
    raw[column] = [0, None, None, 1, 0, 1]
    clean, _, cohort, sequence, _, _, _, _, _, intervals = all_parts(raw)
    risk = models.state_and_domain_risks(clean, cohort, sequence, step14)
    assert not risk.outcome.eq("new_domain_articular").any()
    assert not intervals.at_risk_new_articular.any()
    raw.loc[1, column] = 0
    *_, intervals = all_parts(raw)
    assert not intervals.loc[intervals.from_clinical_episode_id.ge("e003"), "at_risk_new_articular"].any()


def test_marker_actual_artists_exactly_one_cross_per_fallback_and_two_clocks():
    raw = pd.concat([episodes([2, 5, 7], pid="official"), episodes([None, 2, 5, 7], pid="fallback")], ignore_index=True)
    _, _, cohort, sequence, *_ = build(frame=raw)
    for clock in (False, True):
        fig, qc = figures.plot_trajectories(cohort, sequence, clock)
        assert qc["n_observed_x"] == qc["n_expected_x"] == 1
        ax = fig.axes[0]
        assert len(ax.collections) == 2
        assert all(len(artist.get_offsets()) == 1 for artist in ax.collections)
        fallback = next(line for line in ax.lines if line.get_color() == "#d28641")
        assert fallback.get_xdata()[0] == (365/365.25 if clock else 0)
        assert any("Alternative ESSDAI index (n=1)" in t.get_text() for t in ax.get_legend().texts)
        plt.close(fig)


def test_official_comparison_freezes_adjustment_and_scaling_and_fdr_is_separate(monkeypatch):
    rng = np.random.default_rng(20261020)
    n = 90
    frame = pd.DataFrame(dict(patient_id=[f"s{i}" for i in range(n)], baseline_essdai=rng.integers(0, 8, n),
        pred__lab_igg=rng.normal(size=n), pred__age=rng.uniform(25, 75, n), pred__sex=np.where(np.arange(n)%2, "M", "F"),
        pred__duration=rng.uniform(1, 10, n), y=rng.integers(0, 2, n), time_years=rng.uniform(.1, 8, n)))
    spec = step14.PredictorSpec("lab_igg", "IgG", "laboratory", "continuous", "x", "x", transform="zscore", hypothesis_group_id="igg")
    cohort = frame[["patient_id"]].assign(eligible=True, index_source=[v2.OFFICIAL]*70+[v2.FALLBACK]*20)
    task = step14.ModelTask(spec, frame, "baseline", "first_delta3", "cox")
    fitted = []
    def capture(task, data, meta, attempts):
        fitted.append((task, data.copy(), meta.copy()))
        row = step14.result_template(task, meta)
        row.update(valid_for_inference=True, p_value=.02, model_status="ok")
        return row
    monkeypatch.setattr(step14, "fit_task", capture)
    result, _, fdr = models.fit_models([task], cohort, step14, 10)
    assert len(fitted) == 2
    primary, official = fitted
    assert primary[2]["adjustment_status"] == official[2]["adjustment_status"]
    a = primary[1].set_index("patient_id").x
    b = official[1].set_index("patient_id").x
    pd.testing.assert_series_equal(a.reindex(b.index), b)
    assert result.n_fitted_fallback.to_list() == [20, 0]
    assert result.fdr_family.str.startswith(inc.FAMILY).all()
    assert result.loc[result.population.eq("official_only"), "q_value_outcome_wide"].isna().all()
    assert result.index_time_origin.eq("essdai_index_date").all()
    legacy_task = step14.replace(task, data=frame.iloc[:70].copy(), sensitivity="v1_all_episodes_official")
    models.fit_models([legacy_task], cohort, step14, 10, official_comparison=False, reference_results=result)
    legacy_x = fitted[-1][1].set_index("patient_id").x
    pd.testing.assert_series_equal(a.reindex(legacy_x.index), legacy_x)
    assert fitted[-1][2]["adjustment_status"] == primary[2]["adjustment_status"]


@pytest.fixture
def bundle(tmp_path):
    raw = pd.concat([episodes([2, 5, 7], pid="official"), episodes([None, 2, 5, 7], pid="fallback")], ignore_index=True)
    baseline = raw.loc[raw.is_clinical_baseline]
    inputs = tmp_path/"inputs"
    inputs.mkdir()
    paths = {"step10": inputs/"step10.parquet", "step11": inputs/"step11.parquet",
             "context": inputs/"context.parquet", "registry": inputs/"registry.csv", "legacy": inputs/"legacy.parquet"}
    raw.to_parquet(paths["step10"], index=False)
    baseline.to_parquet(paths["step11"], index=False)
    raw[v2.KEYS+[inc.DATE]].to_parquet(paths["context"], index=False)
    pd.DataFrame(columns=["public_variable", "producer_script", "temporal_scope"]).to_csv(paths["registry"], index=False)
    legacy = step14.step13.build_patient_absolute_change_endpoints(raw)
    legacy.to_parquet(paths["legacy"], index=False)
    producer = tmp_path/"producer"
    v2.write_observed_pattern_outputs_v2(raw, baseline, legacy, producer,
        input_origin="synthetic_validation", input_paths={r: paths[r] for r in ("step10", "step11")})
    dirs = v2.output_directories(producer)
    paths.update(patterns=dirs["analytic"]/"13_essdai_patient_observed_patterns_v2.parquet",
        sequence=dirs["analytic"]/"13_essdai_pattern_selected_dates_v2.parquet",
        eligibility=dirs["analytic"]/"13_essdai_pattern_patient_eligibility_v2.parquet",
        manifest=dirs["qc"]/"13_essdai_observed_pattern_manifest_v2.json")
    args = step14.parse_args(["--explainwhy-inclusive-main", "--input-origin", "synthetic_validation",
        "--output-root", str(tmp_path/"result"), "--integrated", str(paths["step10"]), "--baseline", str(paths["step11"]),
        "--context", str(paths["context"]), "--variable-registry", str(paths["registry"]), "--legacy-endpoints", str(paths["legacy"]),
        "--patterns-v2", str(paths["patterns"]), "--selected-dates-v2", str(paths["sequence"]),
        "--eligibility-v2", str(paths["eligibility"]), "--pattern-manifest-v2", str(paths["manifest"])])
    return args, paths


def test_physical_manifest_and_canonical_reconciliation_fail_closed(bundle):
    args, paths = bundle
    manifest = json.loads(paths["manifest"].read_text())
    assert inc.validate_manifest(manifest, paths, "synthetic_validation")
    with pytest.raises(ValueError, match="mixed"):
        inc.validate_manifest(manifest, paths, "canonical")
    paths["step10"].write_bytes(paths["step10"].read_bytes()+b"corruption")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inc.validate_manifest(manifest, paths, "synthetic_validation")


def test_tampered_product_cannot_pass_count_only_qc():
    raw = episodes([None, 2, 5])
    baseline = raw.loc[raw.is_clinical_baseline]
    audit, sequence = v2.build_observed_patterns_v2(raw, baseline)
    patterns = audit.copy()
    patterns.loc[patterns.threshold_abs.eq(3), "essdai_index_score"] = 3
    with pytest.raises(ValueError, match="reconstruction mismatch"):
        inc.reconcile_products(raw, baseline, patterns, audit, sequence)


def test_dryrun_no_writes_and_full_synthetic_bundle_preserves_history(bundle, monkeypatch):
    args, paths = bundle
    args.dry_run = True
    runner.run_inclusive(args, step14)
    assert not args.output_root.exists()
    args.dry_run = False
    real_registry = step14.build_predictor_registry
    monkeypatch.setattr(step14, "build_predictor_registry", lambda: [s for s in real_registry()
        if s.primary_role == "adjustment" or s.name == "domain_biological"])
    historic = runner.directories(args.output_root)["tables"]/"14_historical.csv"
    historic.parent.mkdir(parents=True)
    historic.write_bytes(b"historical result bytes\n")
    manifest = runner.run_inclusive(args, step14)
    assert manifest["n_eligible"] == 2 and manifest["n_fallback"] == 1
    assert manifest["canonical_execution"] == "SYNTHETIC_VALIDATION_ONLY"
    assert historic.read_bytes() == b"historical result bytes\n"
    dirs = runner.directories(args.output_root)
    for area in dirs:
        assert list(dirs[area].glob(inc.PREFIX+"*"))
    for artifact in manifest["artifacts"]:
        assert inc.sha256(args.output_root/artifact["path"]) == artifact["sha256"]
    for csv in list(dirs["tables"].glob(inc.PREFIX+"*.csv"))+list(dirs["qc"].glob(inc.PREFIX+"*.csv")):
        header = pd.read_csv(csv).columns
        assert "patient_id" not in header
        assert not any(c in header for c in ("first_event_date", "essdai_index_date", "from_date", "to_date"))
    args.overwrite = False
    with pytest.raises(FileExistsError):
        runner.run_inclusive(args, step14)
    args.overwrite = True
    paths["patterns"].unlink()
    with pytest.raises(SystemExit, match="BLOCKED"):
        runner.run_inclusive(args, step14)
    blocked = json.loads((dirs["qc"]/(inc.PREFIX+"manifest.json")).read_text())
    assert blocked["status"] == "BLOCKED" and "n_eligible" not in blocked
    assert not list(dirs["analytic"].glob(inc.PREFIX+"*"))
    assert list((args.output_root/"outputs/backups").rglob(inc.PREFIX+"first_delta3_riskset.parquet"))
    assert historic.read_bytes() == b"historical result bytes\n"


def test_install_rolls_back_partial_copy(tmp_path, monkeypatch):
    root, stage = tmp_path/"target", tmp_path/"stage"
    root.mkdir()
    stage.mkdir()
    old = root/(inc.PREFIX+"a.csv")
    old.write_bytes(b"old")
    (stage/old.name).write_bytes(b"new")
    (stage/(inc.PREFIX+"b.csv")).write_bytes(b"second")
    original = runner.os.replace
    def fail_second(source, dest):
        if str(source).endswith("b.csv"):
            raise OSError("synthetic install failure")
        return original(source, dest)
    monkeypatch.setattr(runner.os, "replace", fail_second)
    with pytest.raises(OSError):
        runner.install_bundle(stage, root, [old])
    assert old.read_bytes() == b"old"
    assert not (root/(inc.PREFIX+"b.csv")).exists()


def test_historical_scientific_functions_and_inputs_remain_exact():
    base = "dc56d303a2c9379e8b3193accb615fd9ae8f91ee"
    for rel in ("config.py", "src/block_A/10_build_integrated_longitudinal_dataset.py",
        "src/block_A/11_integrated_baseline_characterization.py", "src/block_A/13_disease_activity_progression.py",
        "src/block_A/observed_patterns.py", "src/block_A/observed_patterns_v2.py", "src/block_A/explainwhy_feasibility.py"):
        assert subprocess.check_output(["git", "show", base+":"+rel], cwd=ROOT) == (ROOT/rel).read_bytes()
    rel = "src/block_A/14_risk_factors_progression.py"
    old = ast.parse(subprocess.check_output(["git", "show", base+":"+rel], cwd=ROOT))
    new = ast.parse((ROOT/rel).read_text())
    def scientific(tree):
        return [ast.dump(n) for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name not in {"main", "parse_args"}]
    assert scientific(old) == scientific(new)


def test_cli_modes_and_support_guards():
    default = step14.parse_args([])
    assert default.explainwhy_inclusive_main
    assert not default.historical_main
    assert default.patterns_v2 == step14.ROOT / "data/analytic/blockA/13_disease_activity_progression/13_essdai_patient_observed_patterns_v2.parquet"
    assert default.selected_dates_v2 == step14.ROOT / "data/analytic/blockA/13_disease_activity_progression/13_essdai_pattern_selected_dates_v2.parquet"
    assert default.eligibility_v2 == step14.ROOT / "data/analytic/blockA/13_disease_activity_progression/13_essdai_pattern_patient_eligibility_v2.parquet"
    assert default.pattern_manifest_v2 == step14.ROOT / "outputs/qc/blockA/13_disease_activity_progression/13_essdai_observed_pattern_manifest_v2.json"
    assert default.legacy_endpoints == step14.ROOT / "data/analytic/blockA/13_disease_activity_progression/13_essdai_patient_delta_endpoints.parquet"
    assert default.context == default.integrated.with_name("10_integrated_longitudinal_context.parquet")
    assert default.variable_registry == default.integrated.with_name("10_variable_registry.csv")
    historical = step14.parse_args(["--historical-main"])
    assert historical.historical_main and not historical.explainwhy_inclusive_main
    inclusive = step14.parse_args(["--explainwhy-inclusive-main"])
    assert inclusive.explainwhy_inclusive_main and inclusive.patterns_v2 == default.patterns_v2
    for args in (["--minimum-events", "9"], ["--lab-max-age-days", "-1"],
        ["--lab-max-age-days", "30", "30"], ["--explainwhy-inclusive-main", "--explainwhy-audit-only"],
        ["--historical-main", "--explainwhy-inclusive-main"]):
        with pytest.raises(SystemExit):
            step14.parse_args(args)


def test_actual_inclusive_cox_fit_retains_fallback_and_diagnostics():
    rng = np.random.default_rng(122)
    n = 120
    spec = step14.PredictorSpec("synthetic", "Synthetic predictor", "serology", "binary", "x", "x")
    frame = pd.DataFrame(dict(patient_id=[str(i) for i in range(n)], pred__synthetic=np.arange(n)%2,
        y=rng.binomial(1, .5, n), time_years=rng.exponential(2, n), baseline_essdai=rng.uniform(0, 4, n),
        age=rng.normal(45, 10, n), sex=np.where(np.arange(n)%3, "F", "M"), duration=rng.uniform(0, 10, n)))
    cohort = frame[["patient_id"]].assign(eligible=True, index_source=[v2.OFFICIAL]*100+[v2.FALLBACK]*20)
    result, attempts, _ = models.fit_models([step14.ModelTask(spec, frame, "baseline", "first_delta3", "cox")], cohort, step14, 10)
    primary = result.loc[result.population.eq("inclusive_primary")].iloc[0]
    assert primary.valid_for_inference and primary.n_fitted_fallback == 20
    assert primary.effect_measure == "HR" and np.isfinite(primary.estimate)
    assert primary.ph_assumption_status in {"pass", "warning"}
    assert attempts.analysis_family.eq(inc.FAMILY).all()
    assert "covariance_min_eigenvalue" in attempts


def test_temporal_qc_reconstructs_first_event_not_only_date_order():
    parts = all_parts(episodes([2, 5, 7]))
    clean, baseline, cohort, sequence, _, risks, registry, predictors, specs, intervals = parts
    risks[3].loc[0, "first_event_date"] = sequence.clinical_anchor_date.iloc[-1]
    with pytest.raises(ValueError, match="reconstruction mismatch"):
        inc.temporal_qc(clean, baseline, cohort, sequence, risks, intervals, predictors, specs, step14, registry)


def test_no_eligible_patients_still_export_structured_not_estimable(bundle, monkeypatch):
    args, paths = bundle
    raw = episodes([None, 2])
    baseline = raw.loc[raw.is_clinical_baseline]
    raw.to_parquet(paths["step10"], index=False)
    baseline.to_parquet(paths["step11"], index=False)
    raw[v2.KEYS+[inc.DATE]].to_parquet(paths["context"], index=False)
    legacy = step14.step13.build_patient_absolute_change_endpoints(raw)
    legacy.to_parquet(paths["legacy"], index=False)
    producer = paths["manifest"].parents[4]
    v2.write_observed_pattern_outputs_v2(raw, baseline, legacy, producer,
        input_origin="synthetic_validation", input_paths={r: paths[r] for r in ("step10", "step11")})
    original = step14.build_predictor_registry
    monkeypatch.setattr(step14, "build_predictor_registry", lambda: [s for s in original()
        if s.primary_role == "adjustment" or s.name == "domain_biological"])
    manifest = runner.run_inclusive(args, step14)
    assert manifest["n_step11"] == 1 and manifest["n_eligible"] == manifest["n_fallback"] == 0
    table = pd.read_csv(runner.directories(args.output_root)["tables"]/(inc.PREFIX+"model_feasibility.csv"))
    assert table.model_status.isin(["not_estimable", "descriptive_only"]).all()
    assert table.component_status.eq("NOT_ESTIMABLE").all()
    assert table.n_fitted.eq(0).all()


def test_discordant_fallback_index_reselected_only_in_explicit_sensitivity():
    raw, baseline, cohort, seq, _, risks = build([None, 2, 8, 5, 8], [0, 365, 365, 730, 1095])
    assert cohort.essdai_index_score.iloc[0] == 8 and risks[3].event.iloc[0] == 0
    audit, selected = v2.build_observed_patterns_v2(raw, baseline, True)
    risk = inc.first_event_riskset(inc.index_cohort(audit), selected)
    assert risk.essdai_index_score.iloc[0] == 5 and risk.event.iloc[0] == 1
    assert risk.essdai_index_date.iloc[0] > cohort.essdai_index_date.iloc[0]

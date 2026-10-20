"""Entirely synthetic: software evidence, never clinical cohort results."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.block_A import observed_patterns as v1
from src.block_A import observed_patterns_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
BASE = "e96f18955cc5d41208fdb3aa66122b20f25a249a"
spec = importlib.util.spec_from_file_location("patterns_v2_step13", ROOT/"src/block_A/13_disease_activity_progression.py")
step13 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = step13
spec.loader.exec_module(step13)


def episodes(scores, days=None, pid="synthetic"):
    days = np.arange(len(scores))*365 if days is None else np.asarray(days)
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(days, unit="D")
    data = pd.DataFrame({"patient_id": pid, "clinical_episode_id": [f"e{i:03d}" for i in range(len(scores))],
        "clinical_anchor_date": dates, "clinical_baseline_date": dates[0], "clinical_baseline_episode_id": "e000",
        "is_clinical_baseline": [True]+[False]*(len(scores)-1), "clinical_visit": True,
        "clinical_visit_number": np.arange(1, len(scores)+1), "time_since_clinical_baseline_days": days,
        "time_since_clinical_baseline_years": days/365.25, "essdai__total": scores})
    for domain in step13.DOMAINS:
        data[step13.DOMAIN_ALIASES[domain][0]] = 0
    return data


def build(scores=None, days=None, df=None, exclude=False):
    df = episodes(scores, days) if df is None else df
    baseline = df.loc[df.is_clinical_baseline.eq(True)].copy()
    audit, seq = v2.build_observed_patterns_v2(df, baseline, exclude)
    return df, baseline, audit, seq


def row(scores, days=None, k=3, exclude=False):
    return build(scores, days, exclude=exclude)[2].set_index("threshold_abs").loc[k]


EXAMPLES = [([2], v2.INSUFFICIENT), ([2, 2], "within_threshold_observed"),
    ([2, 4], "within_threshold_observed"), ([2, 5], "worsening_observed"),
    ([8, 5], "improvement_observed"), ([2, 5, 7], "worsening_observed"),
    ([8, 5, 3], "improvement_observed"), ([2, 6, 2], "intermittent_observed"),
    ([8, 3, 8], "intermittent_observed"), ([4, 8, 1], "intermittent_observed"),
    ([2, 4, 3], "within_threshold_observed"), ([2, 6, 5], "worsening_observed")]


@pytest.mark.parametrize("scores,label", EXAMPLES)
def test_twelve_specification_examples(scores, label):
    r = row(scores)
    assert r.observed_pattern == label
    assert r.intermittence_evaluable == (len(scores) >= 3)
    assert r.trajectory_evidence_level == ("insufficient" if len(scores) == 1 else "two_points_direction_only" if len(scores) == 2 else "three_or_more_points_pattern")


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("scores,label", [([0, 0], "within_threshold_observed"),
    ([0, 1], "worsening_observed"), ([2, 1], "improvement_observed"),
    ([0, 1, 0], "intermittent_observed"), ([2, 1, 2], "intermittent_observed"),
    ([1, 2, 0], "intermittent_observed"), ([0, 2, 1], "worsening_observed")])
def test_exact_borders_both_thresholds(k, scores, label):
    assert row([s*k for s in scores], k=k).observed_pattern == label


@pytest.mark.parametrize("scores", [[0, 0], [123, 123], [0, 123], [123, 0], [14, 19, 18], [2, 2, 4, 2], [8, 8, 8]])
def test_floor_ceiling_zero_and_nested_events(scores):
    a, b = row(scores), row(scores, k=5)
    assert not b.ever_up or a.ever_up
    assert not b.ever_down or a.ever_down
    if len(scores) == 2:
        assert not a.intermittence_evaluable and a.observed_pattern != "intermittent_observed"
    if scores[0] == 0:
        assert not a.ever_down
    if scores[0] == 123:
        assert not a.ever_up


@pytest.mark.parametrize("scores,n,source,label", [
    ([None, 2], 1, v2.FALLBACK, v2.INSUFFICIENT),
    ([None, 2, 5], 2, v2.FALLBACK, "worsening_observed"),
    ([2, None, 5], 2, v2.OFFICIAL, "worsening_observed"),
    ([None, None], 0, v2.FALLBACK, v2.INSUFFICIENT),
    ([2, None], 1, v2.OFFICIAL, v2.INSUFFICIENT)])
def test_index_eligibility_baseline_and_missing_visits_preserved(scores, n, source, label):
    df = episodes(scores).assign(lab_before_index=[10]*len(scores), esspri__total=[8]*len(scores))
    original = df.copy(deep=True)
    _, baseline, audit, seq = build(df=df)
    r = audit.iloc[0]
    assert r.essdai_index_source == source and r.n_distinct_essdai_dates == n
    assert r.observed_pattern == label and r.baseline_pop_official == "Unclassifiable"
    assert r.official_baseline_episode_id == "e000"
    assert r.official_clinical_baseline_date == pd.Timestamp("2020-01-01")
    if n:
        first = next(i for i, s in enumerate(scores) if s is not None)
        assert r.essdai_index_episode_id == f"e{first:03d}"
        assert seq.time_since_essdai_index_years.iloc[0] == 0
        assert r.days_from_clinical_baseline_to_essdai_index == first*365
    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_frame_equal(baseline, original.loc[original.is_clinical_baseline])
    if source == v2.FALLBACK:
        assert r.baseline_state_official == "Missing"
        assert step13.build_patient_absolute_change_endpoints(df).empty


def test_same_date_ties_baseline_priority_stable_order_and_exclude_entire_date():
    df, base, audit, seq = build([2, 50, 8, 4, 5], [0, 0, 365, 365, 730])
    r = audit.iloc[0]
    assert r.essdai_index_score == 2 and r.n_distinct_essdai_dates == 3
    assert seq["essdai__total"].to_list() == [2, 4, 5]
    assert r.n_discordant_essdai_dates == 2 and r.n_same_date_extra_episodes == 2
    assert r.n_canonical_essdai_episodes == 5
    assert r.n_official_baseline_same_date_companions == 1
    shuffled = v2.build_observed_patterns_v2(df.sample(frac=1, random_state=7), base)[0]
    pd.testing.assert_frame_equal(audit, shuffled)
    excluded, selected = v2.build_observed_patterns_v2(df, base, True)
    assert excluded.observed_pattern.eq(v2.INSUFFICIENT).all()
    assert excluded.exclusion_reason.eq("discordant_official_index_date").all() and selected.empty
    # Non-index discordance drops that whole date, never selects a favorable score.
    df, base, audit, seq = build([2, 8, 4, 5], [0, 365, 365, 730])
    excluded, selected = v2.build_observed_patterns_v2(df, base, True)
    assert selected["essdai__total"].to_list() == [2, 5]
    assert excluded.n_distinct_essdai_dates.eq(2).all()
    assert excluded.set_index("threshold_abs").loc[3, "observed_pattern"] == "worsening_observed"
    assert excluded.set_index("threshold_abs").loc[5, "observed_pattern"] == "within_threshold_observed"


def test_equal_ties_do_not_add_time_and_fallback_discordance_reindexes_explicitly():
    r = row([2, 5, 5], [0, 365, 365])
    assert r.n_distinct_essdai_dates == 2 and r.n_discordant_essdai_dates == 0
    assert r.n_same_date_extra_episodes == 1 and not r.intermittence_evaluable
    df, base, audit, _ = build([None, 2, 8, 5, 6], [0, 365, 365, 730, 1095])
    assert audit.essdai_index_score.eq(8).all()
    excluded, _ = v2.build_observed_patterns_v2(df, base, True)
    assert excluded.essdai_index_score.eq(5).all()
    assert excluded.essdai_index_episode_id.eq("e003").all()
    assert excluded.sensitivity_flag.eq("exclude_entire_discordant_dates").all()


def test_threshold_crossings_reversions_are_temporal_not_sign_changes():
    r = row([2, 6, 6, 3, 8])
    assert r.n_threshold_crossings == 3 and r.n_direction_changes == 2
    assert r.up_reversion and r.last_up and r.observed_pattern == "intermittent_observed"
    assert row([2, 6, 5]).observed_pattern == "worsening_observed"
    assert row([2, 4, 3]).observed_pattern == "within_threshold_observed"
    assert row([2, 7, 6], k=3).observed_pattern == "worsening_observed"
    assert row([2, 7, 6], k=5).observed_pattern == "intermittent_observed"


def test_missing_real_dates_not_nominally_repaired():
    df = episodes([None, 2, 5])
    df.loc[1, "clinical_anchor_date"] = pd.NaT
    _, _, audit, seq = build(df=df)
    assert audit.n_undated_essdai_episodes.eq(1).all()
    assert audit.n_distinct_essdai_dates.eq(1).all()
    assert audit.essdai_index_episode_id.eq("e002").all()
    assert len(seq) == 1
    df["clinical_baseline_date"] = pd.NaT
    df.loc[0, "clinical_anchor_date"] = pd.NaT
    _, _, audit, _ = build(df=df)
    assert audit.exclusion_reason.eq("unreliable_clinical_baseline_date").all()


@pytest.mark.parametrize("problem", ["score_text", "score_negative", "score_infinite", "score_high", "date_invalid", "time_negative", "time_zero", "duplicate", "two_baselines", "baseline_id", "baseline_date", "pop_conflict", "patient_missing"])
def test_structural_errors_fail_closed(problem):
    df = episodes([2, 5, 8])
    if problem == "score_text": df["essdai__total"] = [2, "bad", 8]
    if problem == "score_negative": df.loc[1, "essdai__total"] = -1
    if problem == "score_infinite": df.loc[1, "essdai__total"] = np.inf
    if problem == "score_high": df.loc[1, "essdai__total"] = 124
    if problem == "date_invalid": df["clinical_anchor_date"] = ["2020-01-01", "bad", "2022-01-01"]
    if problem == "time_negative": df.loc[1, "time_since_clinical_baseline_years"] = -1
    if problem == "time_zero": df.loc[1, "time_since_clinical_baseline_years"] = 0
    if problem == "duplicate": df.loc[1, "clinical_episode_id"] = "e000"
    if problem == "two_baselines": df.loc[1, "is_clinical_baseline"] = True
    if problem == "baseline_id": df["clinical_baseline_episode_id"] = "wrong"
    if problem == "baseline_date": df["clinical_baseline_date"] += pd.Timedelta(days=1)
    if problem == "pop_conflict": df["pop__status"] = "Pop1"
    if problem == "patient_missing": df.loc[1, "patient_id"] = None
    with pytest.raises(ValueError):
        build(df=df)


@pytest.mark.parametrize("column,value", [("clinical_anchor_date", pd.Timestamp("2021-01-01")), ("essdai__total", 6), ("clinical_episode_id", "different"), ("pop__status", "Pop1")])
def test_step11_disagreements_block(column, value):
    df = episodes([2, 5]).assign(**{"pop__status": "Unclassifiable"})
    base = df.loc[df.is_clinical_baseline].copy()
    base.loc[0, column] = value
    with pytest.raises(ValueError, match="conflicts"):
        v2.build_observed_patterns_v2(df, base)


def cohort():
    return pd.concat([episodes([2, 5], pid="official_two"), episodes([2, 6, 2], pid="official_three"),
        episodes([None, 2, 7], pid="fallback_two"), episodes([None, 8, 3, 8], pid="fallback_three"),
        episodes([None, 4], pid="fallback_one"), episodes([None, None], pid="none")], ignore_index=True)


def test_denominators_partition_pop_empty_strata_entry_delay_and_sensitivities():
    df, base, audit, seq = build(df=cohort())
    summary = v2.pattern_summary_v2(audit)
    for _, g in summary.groupby("threshold_abs"):
        assert g.n_total_step11.eq(6).all() and g.n_eligible_extended.eq(4).all()
        assert g.n_legacy_official.eq(2).all() and g.n_fallback_eligible.eq(2).all()
        assert g.n_two_points.eq(2).all() and g.n_three_plus.eq(2).all()
        assert g.n_insufficient.eq(2).all() and g.n_patients.sum() == 4
        intermittent = g.loc[g.observed_pattern.eq("intermittent_observed")].iloc[0]
        assert intermittent.denominator_primary == 2
        assert intermittent.pct_primary == 100*intermittent.n_patients/2
        assert intermittent.pct_of_extended == 100*intermittent.n_patients/4
    bypop = v2.pattern_summary_v2(audit, "baseline_pop")
    assert bypop.loc[bypop.stratum.eq("Pop1"), "pct_primary"].isna().all()
    assert bypop.loc[bypop.stratum.eq("Unclassifiable"), "n_eligible_extended"].eq(4).all()
    bybaseline = v2.pattern_summary_v2(audit, "baseline_state")
    assert bybaseline.loc[bybaseline.stratum.eq("Missing"), "n_fallback_eligible"].eq(2).all()
    eligibility = v2.eligibility_summary_v2(audit)
    delay = eligibility.loc[eligibility.index_source.eq(v2.FALLBACK) & eligibility.metric.eq("fallback_entry_delay_days_median")].iloc[0]
    assert delay.value == 365
    excluded, _ = v2.build_observed_patterns_v2(df, base, True)
    sensitivity = v2.followup_sensitivity_v2(audit, excluded)
    assert {"dates_eq2", "dates_ge3", "dates_ge4", "frequency_ge1_per_year", "exclude_entire_discordant_dates"} <= set(sensitivity.sensitivity)
    assert not v2.observed_pattern_qc_v2(df, base, step13.build_patient_absolute_change_endpoints(df), audit, excluded, seq).status.eq("FAIL").any()


def test_valid_published_pop_consumed_without_reconstruction():
    df = episodes([2, 5]).assign(**{"pop__status": "Pop2", "esspri__total": 8})
    assert build(df=df)[2].baseline_pop_official.eq("Pop2").all()
    df = episodes([None, 2, 5]).assign(**{"pop__status": "Unclassifiable", "esspri__total": 8})
    assert build(df=df)[2].baseline_pop_official.eq("Unclassifiable").all()


def write(tmp_path, df=None, **kwargs):
    df, base, _, _ = build(df=cohort() if df is None else df)
    endpoints = step13.build_patient_absolute_change_endpoints(df)
    return v2.write_observed_pattern_outputs_v2(df, base, endpoints, tmp_path, input_origin="synthetic_validation", **kwargs)


def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_complete_bundle_hashes_controlled_privacy_backup_and_no_overwrite(tmp_path):
    df = cohort()
    input_paths = {}
    for role, data in (("step10", df), ("step11", df.loc[df.is_clinical_baseline])):
        path = tmp_path/f"{role}.parquet"
        data.to_parquet(path, index=False)
        input_paths[role] = path
    checks = write(tmp_path, input_paths=input_paths)
    assert checks.status.isin(["PASS", "INFO"]).all()
    dirs = v2.output_directories(tmp_path)
    manifest = json.loads((dirs["qc"]/"13_essdai_observed_pattern_manifest_v2.json").read_text())
    assert len(manifest["artifacts"]) == 19  # current main includes six companion figures
    for artifact in manifest["artifacts"]:
        assert v2.sha256(tmp_path/artifact["path"]) == artifact["sha256"]
    for source in manifest["inputs"]:
        assert v2.sha256(source["path"]) == source["sha256"]
    assert manifest["clinical_review"] == "PENDING_CLINICAL_REVIEW"
    assert manifest["input_origin"] == "synthetic_validation"
    assert "real-cohort validation pending" in manifest["closure_status"]
    patient = pd.read_parquet(dirs["analytic"]/"13_essdai_patient_observed_patterns_v2.parquet")
    assert len(patient) == 8 and patient.n_distinct_essdai_dates.ge(2).all()
    attrition = pd.read_parquet(dirs["analytic"]/"13_essdai_pattern_patient_eligibility_v2.parquet")
    assert len(attrition) == 12
    for directory in (dirs["tables"], dirs["qc"]):
        for path in directory.glob("*.csv"):
            content = path.read_text()
            assert "patient_id" not in content and "2020-" not in content
            assert "official_two" not in content and "fallback_two" not in content
    snapshot = files(tmp_path)
    with pytest.raises(FileExistsError):
        write(tmp_path, overwrite=False)
    assert files(tmp_path) == snapshot
    write(tmp_path)
    assert list((tmp_path/"outputs/backups").rglob("13_essdai_observed_pattern_manifest_v2.json"))


def test_dryrun_bad_qc_staging_failure_and_install_failure_leave_no_partial_outputs(tmp_path, monkeypatch):
    write(tmp_path, dry_run=True)
    assert not files(tmp_path)
    with monkeypatch.context() as m:
        m.setattr(v2, "observed_pattern_qc_v2", lambda *args: pd.DataFrame([{"status": "FAIL", "check": "forced"}]))
        with pytest.raises(ValueError, match="QC failure"):
            write(tmp_path)
        assert not files(tmp_path)
    with monkeypatch.context() as m:
        m.setattr(v2, "plot_patterns_v2", lambda *args: (_ for _ in ()).throw(RuntimeError("forced staging")))
        with pytest.raises(RuntimeError, match="forced staging"):
            write(tmp_path)
        assert not files(tmp_path)
    write(tmp_path)
    snapshot = files(tmp_path)
    real_replace = v2.os.replace
    calls = []
    def replace(source, target):
        calls.append(target)
        if len(calls) == 4:
            raise OSError("forced install")
        real_replace(source, target)
    monkeypatch.setattr(v2.os, "replace", replace)
    with pytest.raises(OSError, match="forced install"):
        write(tmp_path, df=episodes([0, 5, 0]))
    after = files(tmp_path)
    assert {p: value for p, value in after.items() if not p.startswith("outputs/backups/")} == snapshot


def test_corrupt_qc_labels_and_legacy_bridge_fail_closed():
    df, base, audit, seq = build(df=cohort())
    endpoints = step13.build_patient_absolute_change_endpoints(df)
    excluded, _ = v2.build_observed_patterns_v2(df, base, True)
    audit.loc[audit.n_distinct_essdai_dates.eq(2), "observed_pattern"] = "intermittent_observed"
    qc = v2.observed_pattern_qc_v2(df, base, endpoints, audit, excluded, seq)
    assert qc.status.eq("FAIL").any()
    _, _, audit, _ = build(df=df)
    endpoints.loc[0, "last_worsened_ge3"] = not endpoints.loc[0, "last_worsened_ge3"]
    qc = v2.observed_pattern_qc_v2(df, base, endpoints, audit, excluded, seq)
    assert qc.set_index("check").loc["legacy_ever_last_without_discordance", "status"] == "FAIL"


def test_v1_endpoints_outputs_and_step14_source_are_unchanged(tmp_path):
    df = cohort()
    before = step13.build_patient_absolute_change_endpoints(df)
    original = v1.build_observed_patterns(df, before)
    dirs = v2.output_directories(tmp_path)
    for name, d in dirs.items():
        d.mkdir(parents=True)
        (d/f"13_historical_{name}.csv").write_text("immutable legacy artifact\n")
    sentinels = files(tmp_path)
    write(tmp_path, df=df)
    for path, content in sentinels.items():
        assert (tmp_path/path).read_bytes() == content
    pd.testing.assert_frame_equal(before, step13.build_patient_absolute_change_endpoints(df))
    pd.testing.assert_frame_equal(original, v1.build_observed_patterns(df, before))
    for relative in ("src/block_A/observed_patterns.py",
                     "src/block_A/10_build_integrated_longitudinal_dataset.py", "src/block_A/11_integrated_baseline_characterization.py", "config.py"):
        assert subprocess.check_output(["git", "show", BASE+":"+relative], cwd=ROOT) == (ROOT/relative).read_bytes()
    def funcs(source):
        return {node.name: ast.dump(node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name not in {"main", "parse_args"}}
    # Both scripts have isolated opt-in dispatch. Scientific functions stay exact.
    for relative in ("src/block_A/13_disease_activity_progression.py", "src/block_A/14_risk_factors_progression.py"):
        assert funcs(subprocess.check_output(["git", "show", BASE+":"+relative], cwd=ROOT)) == funcs((ROOT/relative).read_text())


def test_cli_v2_only_dryrun_and_missing_input_do_not_touch_legacy(tmp_path, monkeypatch):
    source, baseline = tmp_path/"integrated.parquet", tmp_path/"baseline.parquet"
    df = cohort()
    df.to_parquet(source, index=False)
    df.loc[df.is_clinical_baseline].to_parquet(baseline, index=False)
    argv = ["--observed-patterns-v2", "--integrated", str(source), "--baseline", str(baseline),
            "--output-root", str(tmp_path/"run"), "--input-origin", "synthetic_validation"]
    monkeypatch.setattr(step13, "fit_continuous_model", lambda *a: pytest.fail("v2 cannot fit legacy models"))
    step13.main(step13.parse_args([*argv, "--dry-run"]))
    assert not (tmp_path/"run").exists()
    step13.main(step13.parse_args(argv))
    assert not list((tmp_path/"run").rglob("13_progression_episode_level.parquet"))
    with pytest.raises(FileExistsError):
        step13.main(step13.parse_args([*argv, "--no-overwrite"]))
    source.unlink()
    snapshot = files(tmp_path)
    with pytest.raises(SystemExit, match="NOT RUN — canonical inputs unavailable"):
        step13.main(step13.parse_args(argv))
    assert files(tmp_path) == snapshot


def test_empty_cohort_and_non_longitudinal_bundle(tmp_path):
    empty = episodes([2]).iloc[:0]
    audit, seq = v2.build_observed_patterns_v2(empty, empty.copy())
    assert audit.empty and seq.empty
    assert v2.pattern_summary_v2(audit).n_eligible_extended.eq(0).all()
    write(tmp_path, df=episodes([None, None]))
    dirs = v2.output_directories(tmp_path)
    assert pd.read_parquet(dirs["analytic"]/"13_essdai_patient_observed_patterns_v2.parquet").empty
    assert v2.pattern_summary_v2(build([123, 118])[2], "index_state").loc[lambda g: g.stratum.eq("High"), "n_eligible_extended"].eq(1).all()

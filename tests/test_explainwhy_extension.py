"""Synthetic software contracts only; no patient results or clinical closure."""
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

from src.block_A import observed_patterns as patterns
from src.block_A import explainwhy_feasibility as audit

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("explainwhy_step14", ROOT/"src/block_A/14_risk_factors_progression.py")
step14 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = step14
SPEC.loader.exec_module(step14)
step13 = step14.step13


def episodes(scores, days=None):
    days = list(range(0, len(scores)*365, 365)) if days is None else days
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(days, unit="D")
    df = pd.DataFrame({"patient_id": "synthetic", "clinical_episode_id": [f"e{j:02d}" for j in range(len(scores))],
        "clinical_anchor_date": dates, "clinical_baseline_date": dates[0],
        "clinical_baseline_episode_id": "e00", "is_clinical_baseline": [True]+[False]*(len(scores)-1),
        "clinical_visit_number": np.arange(1, len(scores)+1), "clinical_visit": True,
        "time_since_clinical_baseline_days": days, step13.TIME: np.array(days)/365.25,
        step13.ESSDAI: scores})
    for domain in step13.DOMAINS:
        df[step13.DOMAIN_ALIASES[domain][0]] = 0
    return df


def build(scores, days=None, exclude=False):
    df = episodes(scores, days)
    endpoints = step13.build_patient_absolute_change_endpoints(df)
    return df, endpoints, patterns.build_observed_patterns(df, endpoints, exclude)


@pytest.mark.parametrize("scores,label", [
    ([0, 4, 0], "intermittent_observed"),
    ([0, 4, 6], "worsening_threshold_observed_no_reversion"),
    ([5, 1, 5], "intermittent_observed"),
    ([5, 9, 1], "intermittent_observed"),
    ([1, 1, 2], "within_threshold_observed"),
    ([8, 4, 3], "improvement_threshold_observed_no_reversion"),
    ([0, 5], "insufficient_observations"),
    ([0, 3, None], "insufficient_observations"),
    ([0, 0, 0], "within_threshold_observed"),
    ([0, 3, 3], "worsening_threshold_observed_no_reversion"),
])
def test_ordered_shapes(scores, label):
    df, endpoints, result = build(scores)
    assert result.set_index("threshold_abs").loc[3, "observed_pattern"] == label
    qc = patterns.observed_pattern_qc(df, endpoints, result, patterns.build_observed_patterns(df, endpoints, True))
    assert not qc.status.eq("fail").any()


@pytest.mark.parametrize("scores", [[None, 3, 8], [0], [0, None, None]])
def test_no_substituted_baseline_or_followup(scores):
    _, endpoints, result = build(scores)
    assert endpoints.empty and result.empty
    assert list(result.columns) == patterns.PATIENT_COLUMNS
    assert len(patterns.pattern_summary(result)) == 10


def test_missing_official_baseline_fails():
    df = episodes([0, 4, 0])
    df["is_clinical_baseline"] = False
    with pytest.raises(ValueError, match="exactly one official baseline"):
        step13.build_patient_absolute_change_endpoints(df)


def test_baseline_same_day_companion_never_followup():
    _, endpoints, result = build([0, 8, 0], [0, 0, 365])
    assert not endpoints.ever_worsened_ge3.any()
    assert result.n_dates.eq(2).all()
    assert result.observed_pattern.eq("insufficient_observations").all()


def test_discordant_ties_audited_not_counted_as_reversions():
    df, endpoints, result = build([0, 5, 1, 1], [0, 365, 365, 730])
    assert endpoints.ever_worsened_ge3.all()
    assert result.observed_pattern.eq("within_threshold_observed").all()
    assert not result.up_reversion.any()
    assert result.n_dates.eq(3).all() and result.n_points.eq(4).all()
    excluded = patterns.build_observed_patterns(df, endpoints, True)
    assert excluded.n_dates.eq(2).all()
    qc = patterns.observed_pattern_qc(df, endpoints, result, excluded)
    assert not qc.status.eq("fail").any()
    row = qc.set_index("check").loc["ever_up_3_audited_tie_disagreement"]
    assert row.n_patients == 1


def test_same_day_later_score_cannot_revert_same_date_crossing():
    _, _, result = build([0, 1, 5, 7], [0, 365, 365, 730])
    assert not result.up_reversion.any()
    assert result.n_direction_changes.eq(0).all()


def test_reentry_plateaus_crossings_and_final_direction_are_separate():
    _, endpoints, result = build([0, 4, 4, 1, 6])
    row = result.set_index("threshold_abs").loc[3]
    assert row.observed_pattern == "intermittent_observed" and row.up_reversion
    assert row.directionality_at_last == "up" and row.n_direction_changes == 2
    assert row.n_threshold_crossings == 3 and row.n_up_crossings == 3
    assert row.first_up_date == pd.Timestamp("2020-12-31")
    assert endpoints.last_worsened_ge3.all()


def test_labels_need_not_nest_but_events_do():
    _, _, result = build([0, 5, 4])
    a, b = result.set_index("threshold_abs").loc[3], result.set_index("threshold_abs").loc[5]
    assert a.observed_pattern == "worsening_threshold_observed_no_reversion"
    assert b.observed_pattern == "intermittent_observed"
    assert a.ever_up and b.ever_up


def test_summary_partitions_empty_states_pop_unknown_and_floor():
    df, endpoints, result = build([0, 4, 0])
    baseline = df.loc[df.is_clinical_baseline].assign(**{"pop__status": "Unclassifiable"})
    pop = patterns.verified_baseline_pop(baseline)
    table = patterns.pattern_summary(result, "baseline_pop", pop)
    assert table.loc[table.stratum.eq("Pop3"), "denominator"].eq(0).all()
    assert table.loc[table.stratum.eq("Unclassifiable"), "denominator"].eq(1).all()
    assert not result.can_improve.any() and not result.ever_down.any()
    state = patterns.pattern_summary(result, "baseline_state")
    assert state.loc[state.stratum.eq("High"), "pct_patients"].isna().all()
    for table in (table, state):
        for _, g in table.groupby(["stratum", "threshold_abs"]):
            assert g.n_patients.sum() == g.denominator.iloc[0]


def test_bad_pop_not_recoded_and_qc_rejects_corruption():
    df, endpoints, result = build([0, 4, 0])
    baseline = df.loc[df.is_clinical_baseline].assign(**{"pop__status": "Pop1"})
    with pytest.raises(ValueError, match="conflicts"):
        patterns.verified_baseline_pop(baseline)
    corrupt = result.copy()
    corrupt.loc[0, "observed_pattern"] = "persistent_worsening"
    assert patterns.observed_pattern_qc(df, endpoints, corrupt, patterns.build_observed_patterns(df, endpoints, True)).status.eq("fail").any()


def test_temporal_label_ignores_labs_treatment_and_input_order():
    df, endpoints, result = build([5, 9, 1])
    changed = df.assign(lab__igg__value=[1, 900, 0], systemic_treatment_exposure=[0, 1, None])
    pd.testing.assert_frame_equal(result, patterns.build_observed_patterns(changed.sample(frac=1, random_state=8), endpoints))


def audit_args(tmp_path, *extra):
    return step14.parse_args(["--integrated", str(tmp_path/"integrated.parquet"),
        "--baseline", str(tmp_path/"baseline.parquet"), "--progression", str(tmp_path/"progression.parquet"),
        "--output-root", str(tmp_path), "--explainwhy-audit-only", "--input-origin", "synthetic_validation", *extra])


def test_missing_inputs_reports_unknown_and_no_models(tmp_path, monkeypatch):
    monkeypatch.setattr(step14, "build_analysis_tasks", lambda *a: pytest.fail("no model task construction permitted"))
    decisions = audit.run_feasibility_audit(audit_args(tmp_path), step14)
    assert decisions.set_index("gate").loc["G0", "status"] == "BLOCKED"
    dirs = step14.output_directories(tmp_path)
    f = pd.read_csv(dirs["tables"]/"14_explainwhy_modality_feasibility.csv")
    assert f[list(audit.COUNT_COLUMNS)].isna().all().all()
    assert f.support_reason.str.contains("NOT RUN").all()
    manifest = json.loads((dirs["qc"]/"14_explainwhy_manifest.json").read_text())
    assert manifest["input_origin"] == "synthetic_validation" and manifest["new_models_fitted"] == 0
    verify_hashes(tmp_path, manifest)
    with pytest.raises(FileExistsError):
        audit.run_feasibility_audit(audit_args(tmp_path, "--no-overwrite"), step14)
    audit.run_feasibility_audit(audit_args(tmp_path), step14)
    assert list((tmp_path/"outputs/backups").rglob("14_explainwhy_manifest.json"))


def registry_for(column, scope="episode"):
    return pd.DataFrame([{"public_variable": column, "producer_script": "synthetic_fixture", "temporal_scope": scope}])


@pytest.mark.parametrize("violation", ["future", "undated", "mixed_units", "missing_units", "unregistered", "retrospective_scope"])
def test_lab_timing_units_provenance_fail_closed(violation):
    df = episodes([0, 3, 4])
    col = "lab__igg__value"
    df[col] = [1., 2., 3.]
    df["lab__igg__measurement_date"] = df.clinical_anchor_date
    df["lab__igg__unit"] = "mg/dL"
    vr = registry_for(col)
    if violation == "future":
        df["lab__igg__measurement_date"] += pd.Timedelta(days=1)
    if violation == "undated":
        df["lab__igg__measurement_date"] = pd.NaT
    if violation == "mixed_units":
        df.loc[1, "lab__igg__unit"] = "g/L"
    if violation == "missing_units":
        df["lab__igg__unit"] = None
    if violation == "unregistered":
        vr = None
    if violation == "retrospective_scope":
        vr = registry_for(col, "patient_retrospective")
    spec = next(s for s in step14.build_predictor_registry() if s.name == "lab_igg")
    values, *_ = audit.audit_values(df, spec, step14, vr)
    assert values.isna().all()


def test_feasibility_counts_original_adjacency_patient_dates_and_delta_events():
    df = episodes([6, None, 9, 12])
    col = "lab__igg__value"
    df[col] = [1., 2., 3., 4.]
    df["lab__igg__measurement_date"] = df.clinical_anchor_date
    df["lab__igg__unit"] = "mg/dL"
    df.loc[1, col] = np.nan
    baseline = df.loc[df.is_clinical_baseline].copy()
    endpoints = step13.build_patient_absolute_change_endpoints(df)
    f = audit.modality_feasibility(df, baseline, endpoints, registry_for(col), step14)
    row = f.loc[f.variable.eq("lab_igg") & f.stratifier.eq("overall")].iloc[0]
    assert row.n_patients_baseline == 1 and row.n_patients_repeated == 1
    assert row.n_valid_episodes == 3 and row.n_distinct_dates == 3
    assert row.n_lagged_eligible_intervals == 1
    assert row.n_delta3_events_with_predictor == 1 and row.n_delta5_events_with_predictor == 1
    assert endpoints.first_observed_worsening_ge3_date.iloc[0] == df.clinical_anchor_date.iloc[2]
    assert step13.classify_essdai_activity(9) == "Moderate"  # delta3 is not category change


def test_pro_instrument_version_and_unknown_treatment():
    df = episodes([0, 3, 4]).assign(pro__sf36_pcs=[40., 45., 47.], systemic_treatment_exposure=[0., None, 1.])
    spec = next(s for s in audit.audit_registry(step14) if s.name == "sf36_pcs")
    values, *_ = audit.audit_values(df, spec, step14, registry_for("pro__sf36_pcs"))
    assert values.isna().all()
    df["pro__sf36_scoring_version"], df["pro__sf36_scoring_valid"] = "v1", True
    values, *_ = audit.audit_values(df, spec, step14, registry_for("pro__sf36_pcs"))
    assert values.notna().all()
    treatment = next(s for s in audit.audit_registry(step14) if s.family == "treatment")
    values, *_ = audit.audit_values(df, treatment, step14, registry_for("systemic_treatment_exposure"))
    assert values.isna().all()


def test_baseline_only_registered_serology_is_not_labeled_absent():
    df = episodes([0, 4, 0]).assign(sero__baseline_low_c4=[1., np.nan, np.nan])
    baseline = df.loc[df.is_clinical_baseline].copy()
    table = audit.modality_feasibility(df, baseline, step13.build_patient_absolute_change_endpoints(df),
                                      registry_for("sero__baseline_low_c4", "clinical_baseline"), step14)
    row = table.loc[table.variable.eq("low_c4") & table.stratifier.eq("overall")].iloc[0]
    assert row.n_patients_baseline == 1 and row.n_patients_repeated == 0
    assert row.schema_present and row.support_status == "PARTIAL"
    assert row.n_delta3_events_with_predictor == 1


def verify_hashes(root, manifest):
    for artifact in manifest["artifacts"]:
        assert hashlib.sha256((root/artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]


def test_pattern_output_manifest_privacy_and_failed_qc_guard(tmp_path, monkeypatch):
    df, endpoints, result = build([0, 4, 0])
    baseline = df.loc[df.is_clinical_baseline].copy()
    dirs = {name: tmp_path/path for name, path in {
        "tables": "outputs/tables/blockA/13_disease_activity_progression",
        "figures": "outputs/figures/blockA/13_disease_activity_progression",
        "qc": "outputs/qc/blockA/13_disease_activity_progression",
        "analytic": "data/analytic/blockA/13_disease_activity_progression"}.items()}
    for d in dirs.values():
        d.mkdir(parents=True)
    patterns.write_observed_pattern_outputs(df, baseline, endpoints, **dirs, input_origin="synthetic_validation")
    manifest = json.loads((dirs["qc"]/"13_essdai_observed_pattern_manifest.json").read_text())
    verify_hashes(tmp_path, manifest)
    assert manifest["pop_status"] == "unavailable"
    for directory in (dirs["tables"], dirs["qc"]):
        for p in directory.glob("*.csv"):
            content = p.read_text()
            assert "patient_id" not in content and "2020-" not in content and "synthetic" not in content
    assert (dirs["figures"]/"13_essdai_observed_pattern_aggregate.pdf").stat().st_size > 1000
    patient = dirs["analytic"]/"13_essdai_patient_observed_patterns.parquet"
    before = patient.read_bytes()
    monkeypatch.setattr(patterns, "observed_pattern_qc", lambda *a: pd.DataFrame([{"status": "fail"}]))
    with pytest.raises(ValueError, match="integrity"):
        patterns.write_observed_pattern_outputs(df, baseline, endpoints, **dirs)
    assert patient.read_bytes() == before


def test_available_synthetic_audit_then_contract_failure_clears_stale_support(tmp_path):
    df = episodes([0, 4, 0])
    df.to_parquet(tmp_path/"integrated.parquet", index=False)
    baseline = df.loc[df.is_clinical_baseline].copy()
    baseline.to_parquet(tmp_path/"baseline.parquet", index=False)
    args = audit_args(tmp_path)
    decisions = audit.run_feasibility_audit(args, step14)
    assert decisions.set_index("gate").loc["G0", "status"] == "PARTIAL"
    support = step14.output_directories(tmp_path)["tables"]/"14_explainwhy_pattern_support.csv"
    assert support.exists()
    baseline[step13.ESSDAI] = 999
    baseline.to_parquet(tmp_path/"baseline.parquet", index=False)
    decisions = audit.run_feasibility_audit(args, step14)
    assert decisions.set_index("gate").loc["G0", "status"] == "BLOCKED"
    assert not support.exists()
    assert "999" not in (step14.output_directories(tmp_path)["qc"]/"14_explainwhy_feasibility_decisions.csv").read_text()


def test_primary_functions_preserved_against_audited_main():
    base = "06b492c2904f59d8ce557e61e186593a90cafd38"
    for filename in ("13_disease_activity_progression.py", "14_risk_factors_progression.py"):
        relative = "src/block_A/"+filename
        original = subprocess.check_output(["git", "show", base+":"+relative], cwd=ROOT, text=True)
        def functions(source):
            return {node.name: ast.dump(node) for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        before, after = functions(original), functions((ROOT/relative).read_text())
        assert {k: v for k, v in before.items() if k not in {"parse_args", "main"}} == {k: after[k] for k in before if k not in {"parse_args", "main"}}


def test_audit_cli_missing_inputs_exit_and_primary_outputs_untouched(tmp_path):
    dirs = step14.output_directories(tmp_path)
    dirs["tables"].mkdir(parents=True)
    sentinel = dirs["tables"]/"14_baseline_ge5_cox_models.csv"
    sentinel.write_text("primary unchanged\n")
    with pytest.raises(SystemExit) as error:
        step14.main(audit_args(tmp_path, "--no-overwrite"))
    assert error.value.code == 2
    assert sentinel.read_text() == "primary unchanged\n"
    assert not (dirs["qc"]/"14_analysis_metadata.json").exists()


def test_step13_dryrun_and_no_overwrite_protect_new_outputs(tmp_path):
    df = episodes([0, 4, 0])
    df.to_parquet(tmp_path/"integrated.parquet", index=False)
    df.loc[df.is_clinical_baseline].to_parquet(tmp_path/"baseline.parquet", index=False)
    target = tmp_path/"data/analytic/blockA"/step13.STEM/"13_essdai_patient_observed_patterns.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"protected pattern output")
    cli = ["--integrated", str(tmp_path/"integrated.parquet"), "--baseline", str(tmp_path/"baseline.parquet"),
           "--output-root", str(tmp_path), "--no-overwrite", "--input-origin", "synthetic_validation"]
    step13.main(step13.parse_args([*cli, "--dry-run"]))
    assert target.read_bytes() == b"protected pattern output"
    with pytest.raises(FileExistsError):
        step13.main(step13.parse_args(cli))


def test_audit_context_preserves_instrument_selection_flags_and_rejects_conflict():
    df = episodes([0, 4, 0])
    baseline = df.loc[df.is_clinical_baseline].copy()
    context = df[step14.KEYS].assign(pro__sf36_scoring_version="v1", pro__sf36_scoring_valid=True,
                                    lab__igg__selection_status="selected_single")
    out, base = audit.attach_audit_context(df, baseline, context, step14)
    assert out.pro__sf36_scoring_version.eq("v1").all()
    assert base.lab__igg__selection_status.eq("selected_single").all()
    out["pro__sf36_scoring_version"] = "v2"
    with pytest.raises(ValueError, match="conflicting audit context"):
        audit.attach_audit_context(out, base, context, step14)


def test_four_dates_duration_and_directional_floor_sensitivity():
    df, endpoints, primary = build([1, 4, 1, 6], [0, 365, 730, 1827])
    sensitivity = patterns.followup_sensitivity(primary, patterns.build_observed_patterns(df, endpoints, True))
    for name in ("dates_ge3", "dates_ge4", ">=5y"):
        group = sensitivity.loc[sensitivity.sensitivity.eq(name) & sensitivity.stratifier.eq("overall")]
        assert group.denominator.eq(1).all()
        assert group.groupby("threshold_abs").n_patients.sum().eq(1).all()
    assert primary.can_improve.eq(False).all()
    assert primary.set_index("threshold_abs").loc[3, "observed_pattern"] == "intermittent_observed"

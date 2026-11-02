"""Synthetic-only interpretability and no-prior-access regression tests."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from test_explainwhy_inclusive_v2 import (step14, bundle, runner, inc, models,
    composition, composition_fixture, composition_parts)
from src.block_A.explainwhy_inclusive_interpretability import pns_support_summary


def pns_task(n_exposed=5, missing=False):
    spec = next(s for s in step14.build_predictor_registry() if s.name == "domain_pns")
    rng = np.random.default_rng(17)
    n = 45
    data = pd.DataFrame(dict(patient_id=[f"PRIVATE_SYNTHETIC_{i}" for i in range(n)],
        y=rng.uniform(0, 15, n), pred__domain_pns=[1]*n_exposed+[0]*(n-n_exposed),
        from_essdai=rng.uniform(0, 10, n), interval_years=rng.uniform(.1, 2, n)))
    data = pd.concat([data, data.iloc[:1], data.iloc[:1]], ignore_index=True)
    if missing:
        data.pred__domain_pns = np.nan
    return step14.ModelTask(spec, data, "timevarying", "subsequent_essdai", "lagged", "none")


@pytest.mark.parametrize("exposed,missing,expected", [(5, False, "FRAGILE_MINIMUM_EXPOSED"),
    (7, False, "SUPPORT_REMAINS"), (0, False, "NOT_ESTIMABLE_OR_NOT_AVAILABLE"),
    (5, True, "NOT_ESTIMABLE_OR_NOT_AVAILABLE")])
def test_pns_cluster_support_no_refit_or_identifiers(exposed, missing, expected, monkeypatch):
    task = pns_task(exposed, missing)
    data, meta = step14.prepare_model(task, 45, 10)
    monkeypatch.setattr(step14, "fit_task", lambda *a: pytest.fail("diagnostics must never refit"))
    row = pns_support_summary(task, data, meta,
        dict(model_status="synthetic", valid_for_inference=meta["supported_for_model"]),
        "inclusive_primary", step14, 10, "synthetic_validation")
    assert row["fragility_status"] == expected
    assert row["n_loo_deletions_evaluated"] == (0 if missing else exposed)
    if exposed == 5 and not missing:
        assert row["min_exposed_after_deletion"] == 4
        assert row["min_intervals_after_deletion"] == len(data)-3
        assert row["deletion_support_status"] == "NOT_ESTIMABLE_AFTER_PATIENT_DELETION"
    serialized = json.dumps(step14._json_safe(row))
    assert "PRIVATE_SYNTHETIC" not in serialized
    assert not {"patient_id", "estimate", "p_value", "q_value", "date"}.intersection(row)


def test_no_prior_access_bundle_current_hashes_and_internal_qc(bundle, monkeypatch):
    args, paths = bundle
    args.no_prior_result_comparison = True
    args.overwrite = False
    monkeypatch.setattr(runner, "_primary_regression", lambda *a: pytest.fail("prior results accessed"))
    real_hash = inc.sha256
    def guarded_hash(path):
        if "DO_NOT_READ" in str(path):
            pytest.fail("prior output hashed")
        return real_hash(path)
    monkeypatch.setattr(inc, "sha256", guarded_hash)
    prior = args.output_root/"outputs/DO_NOT_READ/14_prior.csv"
    prior.parent.mkdir(parents=True)
    prior.write_text("prior output must not be read")
    real_registry = step14.build_predictor_registry
    monkeypatch.setattr(step14, "build_predictor_registry", lambda: [s for s in real_registry()
        if s.primary_role == "adjustment" or s.name in {"domain_pns", "domain_biological"}])
    args.dry_run = True
    checks = runner.run_inclusive(args, step14)
    assert "temporal_leakage_qc" in checks and "legacy_bridge_qc" in checks
    assert not runner.directories(args.output_root)["qc"].exists()
    args.dry_run = False
    manifest = runner.run_inclusive(args, step14)
    assert manifest["prior_result_comparison"] == "NOT_REQUESTED_BY_INVESTIGATOR"
    assert manifest["composition_provenance"]["status"] == "UNVERIFIED"
    assert manifest["snapshot"]["historical_products_before"] == {}
    assert manifest["snapshot"]["inclusive_products_before"] == {}
    qc = runner.directories(args.output_root)["qc"]
    result = pd.read_csv(qc/(inc.PREFIX+"primary_result_regression_qc.csv"))
    assert result.status.eq("NOT_REQUESTED_BY_INVESTIGATOR").all()
    domain = pd.read_csv(qc/(inc.PREFIX+"domain_interpretability_qc.csv"))
    assert domain.interpretation_label.eq("OTHER_DOMAINS_NOT_ESTIMABLE").all()
    for artifact in manifest["artifacts"]:
        assert real_hash(args.output_root/artifact["path"]) == artifact["sha256"]
    for name in ("domain_interpretability_qc.csv", "pns_fragility_qc.csv", "closure_summary.md"):
        text = (qc/(inc.PREFIX+name)).read_text()
        assert "patient_id" not in text and "2020-01-01" not in text
    with pytest.raises(FileExistsError):
        runner.run_inclusive(args, step14)


def test_flag_is_inclusive_only():
    with pytest.raises(SystemExit):
        step14.parse_args(["--no-prior-result-comparison"])
    assert step14.parse_args(["--explainwhy-inclusive-main", "--no-prior-result-comparison"]).no_prior_result_comparison


def test_missing_inputs_dryrun_fail_closed(tmp_path):
    args = step14.parse_args(["--explainwhy-inclusive-main", "--no-prior-result-comparison",
        "--integrated", str(tmp_path/"missing.parquet"), "--output-root", str(tmp_path/"new"), "--dry-run"])
    with pytest.raises(SystemExit, match="required canonical inputs unavailable"):
        runner.run_inclusive(args, step14)
    assert not args.output_root.exists()


def test_verified_composition_pair_stays_matched_and_outside_primary_fdr():
    raw, policy = composition_fixture()
    intervals, _, parts = composition_parts(raw, policy)
    clean, _, cohort, sequence, _, risks, _, predictors, specs = parts
    tasks = models.build_tasks(clean, cohort, sequence, risks, predictors, intervals, specs, step14, include_secondary=False)
    pair = [t for t in tasks if t.spec.name == "domain_articular" and t.comparison_role]
    observed = []
    def observe(task, data, meta, result, population):
        if population != "official_only":
            observed.append((meta["analysis_sample_hash"], meta["adjustment_status"]))
    results, _, _ = models.fit_models(pair, cohort, step14, 10, diagnostic_observer=observe)
    assert observed[0] == observed[1]
    assert results.fdr_eligibility.eq(False).all()
    assert results.q_value_outcome_wide.isna().all()


@pytest.mark.parametrize("field,value", [("source_sha256", "0"*64),
    ("producer_sha256", "0"*64), ("producer_path", "../outside.py"),
    ("producer_commit", "invalid"), ("domain_version_field", "inferred_column"),
    ("domain_semantics", "row_resolution_schema"), ("temporal_policy", "future_allowed"),
    ("weights", {"pns": 0}), ("compatible_versions", [{"domain_version": "A"}])])
def test_invalid_declared_authorization_never_falls_back(bundle, field, value):
    _, paths = bundle
    manifest = json.loads(paths["manifest"].read_text())
    producer = "src/block_A/observed_patterns_v2.py"
    auth = dict(author="synthetic", protocol="synthetic-only", source_role="step10",
        source_sha256=inc.sha256(paths["step10"]), producer_path=producer,
        producer_sha256=inc.sha256(step14.ROOT/producer), producer_commit=manifest["code_commit"],
        domain_version_field="_essdai_version", total_version_field="_essdai_total_version",
        domain_semantics="all_12_domain_scoring_algorithm", total_semantics="recorded_total_scoring_algorithm",
        temporal_policy="episode_as_of_no_future_information", weights=step14.canonical_domain_weights(),
        compatible_versions=[dict(domain_version="A", total_version="B", scoring_system="canonical")])
    manifest["composition_provenance"] = auth
    assert composition.load_authorization(manifest, paths, step14.ROOT, step14)[1]["status"] == "VERIFIED_AUTHORIZATION"
    auth[field] = value
    with pytest.raises(ValueError, match="composition_provenance"):
        composition.load_authorization(manifest, paths, step14.ROOT, step14)


@pytest.mark.parametrize("mutation", ["absent_certificate", "partial", "incompatible", "discordant", "negative"])
def test_composition_never_bypasses_gate(mutation):
    raw, policy = composition_fixture()
    if mutation == "absent_certificate":
        policy = None
    elif mutation == "partial":
        raw.loc[1, step14.step13.DOMAIN_ALIASES["pns"][0]] = np.nan
    elif mutation == "incompatible":
        raw["_essdai_total_version"] = "unknown"
    elif mutation == "discordant":
        raw.loc[1, "essdai__total"] = 8
    elif mutation == "negative":
        raw.loc[1, "essdai__total"] = 1
    intervals, _, _ = composition_parts(raw, policy)
    assert (intervals.from_essdai_other_domains_articular.isna() |
            intervals.to_essdai_other_domains_articular.isna()).all()
    assert not intervals.composition_pair_gate_articular.eq("compatible_verified").any()


def test_deletion_keeps_frozen_design_and_checks_rank():
    task = pns_task(7)
    task.data["rare_covariate"] = task.data.patient_id.eq("PRIVATE_SYNTHETIC_0").astype(float)
    task = replace(task, extra_covariates=("rare_covariate",))
    data, meta = step14.prepare_model(task, 45, 10)
    assert meta["supported_for_model"]
    row = pns_support_summary(task, data, meta, dict(model_status="ok", valid_for_inference=True),
        "inclusive_primary", step14, 10, "synthetic_validation")
    assert row["min_exposed_after_deletion"] == 6
    assert row["fragility_status"] == "FRAGILE_OTHER_SUPPORT"
    assert row["n_loo_deletions_below_support"] == 1


def test_pns_uses_complete_cases_and_transformed_exposure():
    task = pns_task(7)
    task.data.loc[task.data.patient_id.eq("PRIVATE_SYNTHETIC_0"), "from_essdai"] = np.nan
    data, meta = step14.prepare_model(task, 45, 10)
    row = pns_support_summary(task, data, meta, dict(model_status="ok", valid_for_inference=True),
        "inclusive_primary", step14, 10, "synthetic_validation")
    assert row["n_exposed_patients"] == 6
    assert row["n_model_intervals"] == len(task.data)-3

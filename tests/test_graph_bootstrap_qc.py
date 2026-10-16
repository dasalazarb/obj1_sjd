"""Execution QC is distinct from topological evidence in graph sensitivity."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "src/studies/graph/02_graph_representation_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("graph_sensitivity_bootstrap_qc", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def config():
    return M.CANON.load_config(SCRIPT.with_name("config_sensitivity.yaml"))


def mock_bootstrap(monkeypatch, n_success, n_matches, requested=60):
    cfg = config()
    cfg["bootstrap"]["replicates"] = requested
    values = {"a": pd.Series(np.arange(10, dtype=float)),
              "b": pd.Series(np.arange(10, dtype=float)[::-1])}
    calls = []
    def fit(raw, families, scenario, *args, **kwargs):
        calls.append(raw.copy())
        return SimpleNamespace(balance_audit=pd.DataFrame(columns=M.BALANCE_COLUMNS))
    attempts = iter(range(requested))
    def mapper(rep, cfg):
        index = next(attempts)
        if index >= n_success:
            raise ValueError("synthetic Mapper computation failure")
        return {"community_sets": [set(range(8))] if index < n_matches else [],
                "supported": np.array([True] if index < n_matches else [])}
    monkeypatch.setattr(M, "fit_embedding", fit)
    monkeypatch.setattr(M, "analyze_mapper", mapper)
    result = M.bootstrap_stability(values, {name: "LAB" for name in values},
                                   pd.DataFrame(), "S1", [set(range(10))], cfg)
    assert len(calls) == requested  # Full representation refit, no success-driven retries.
    return result


@pytest.mark.parametrize("success,matches,status,conditional,conservative", [
    (27, 27, "BOOTSTRAP_QC_FAIL", 1., .45),
    (60, 50, "BOOTSTRAP_QC_PASS", 50/60, 50/60),
    (0, 0, "INVALID_BOOTSTRAP", np.nan, 0.),
])
def test_execution_denominators_and_gate(monkeypatch, success, matches, status, conditional, conservative):
    result = mock_bootstrap(monkeypatch, success, matches)
    assert result.summary["n_requested"] == result.summary["n_attempted"] == 60
    assert result.summary["n_success"] == success
    assert result.summary["n_failed"] == 60 - success
    assert result.qc_status == status
    assert result.matched_successes.tolist() == [matches]
    if success:
        assert result.conditional[0] == pytest.approx(conditional)
    else:
        assert np.isnan(result.conditional[0])
    assert result.conservative[0] == pytest.approx(conservative)
    if status != "BOOTSTRAP_QC_PASS":
        assert np.isnan(result.primary[0])
    else:
        assert result.primary[0] >= config()["bootstrap"]["stable_community_threshold"]
    assert len(result.replicate_audit) == 60
    assert result.summary["n_failed_MAPPER"] == 60 - success


def test_same_seed_reproduces_draws_and_metrics(monkeypatch):
    a = mock_bootstrap(monkeypatch, 27, 27)
    b = mock_bootstrap(monkeypatch, 27, 27)
    pd.testing.assert_frame_equal(a.replicate_audit, b.replicate_audit)
    assert a.summary == b.summary
    np.testing.assert_equal(a.conditional, b.conditional)
    for indices in a.replicate_audit.sample_indices:
        draw = list(map(int, indices.split(",")))
        assert len(draw) == len(set(draw)) == 8


def test_unexpected_schema_error_propagates(monkeypatch):
    monkeypatch.setattr(M, "select_features", lambda *a, **k: (_ for _ in ()).throw(KeyError("broken schema")))
    values = {"a": pd.Series(range(10)), "b": pd.Series(range(10))}
    with pytest.raises(KeyError, match="broken schema"):
        M.bootstrap_stability(values, {x: "LAB" for x in values}, pd.DataFrame(), "S1", [], config())


@pytest.mark.parametrize("stage", ["FEATURE_SELECTION", "PCA"])
def test_expected_computation_failures_record_stage(monkeypatch, stage):
    cfg = config()
    cfg["bootstrap"]["replicates"] = 2
    def fail(*args, **kwargs):
        raise ValueError("synthetic expected numerical failure")
    monkeypatch.setattr(M, "select_features" if stage == "FEATURE_SELECTION" else "fit_embedding", fail)
    values = {"a": pd.Series(range(10)), "b": pd.Series(range(10))}
    result = M.bootstrap_stability(values, {x: "LAB" for x in values}, pd.DataFrame(), "S1", [], cfg)
    assert result.summary[f"n_failed_{stage}"] == 2
    assert result.replicate_audit.failure_stage.eq(stage).all()


def matrix_fixture():
    raw = pd.DataFrame({"lab__hbv__binary": [1., 1., 1., 1.],
                        "pro": [3., 1., 4., 2.], "lab__albumin__value": [1., 2., 3., 4.]})
    families = {"lab__hbv__binary": "LAB", "pro": "PRO", "lab__albumin__value": "LAB"}
    groups = {"hbv": "INFECTIOUS_SEROLOGY", "albumin": "LIVER_PANCREAS"}
    labs = {"lab__hbv__binary": "lab__hbv", "lab__albumin__value": "lab__albumin"}
    return raw, families, groups, labs


def test_constant_infectious_group_is_zero_and_keeps_column_position():
    raw, families, groups, labs = matrix_fixture()
    before = raw.copy()
    matrix, names, audit = M.prepare_matrix(raw, families, True, True, groups, labs, "S4")
    assert names == raw.columns.tolist()
    assert matrix.shape == raw.shape
    assert np.isfinite(matrix).all()
    np.testing.assert_equal(matrix[:, 0], np.zeros(len(raw)))
    row = audit.loc[audit.block.eq("INFECTIOUS_SEROLOGY")].iloc[0]
    assert row.status == "zero_variance"
    assert row.weight_applied == row.effective_weight == 0
    assert row.n_active_groups == 1
    assert audit.loc[audit.status.eq("active"), "effective_weight"].gt(0).all()
    pd.testing.assert_frame_equal(before, raw)
    _, s3_names, _ = M.prepare_matrix(raw, families, True, scenario="S3")
    assert names == s3_names


def test_small_real_signal_is_not_treated_as_constant():
    matrix, sigma = M.scale_mfa_block(pd.DataFrame({"x": np.arange(4) * 1e-200}), "TINY", allow_zero=True)
    assert sigma > 0
    assert np.isfinite(matrix).all() and np.any(matrix != 0)


def test_all_lab_groups_degenerate_are_non_estimable_and_audited():
    raw, families, groups, labs = matrix_fixture()
    raw["lab__albumin__value"] = 1.
    with pytest.raises(M.BlockNotEstimable, match="all groups are degenerate") as error:
        M.prepare_matrix(raw, families, True, True, groups, labs, "S4")
    audit = error.value.balance_audit
    assert audit.loc[audit.level.eq("LAB_GROUP"), "status"].eq("zero_variance").all()
    assert audit.loc[audit.level.eq("FAMILY"), "status"].eq("invalid").all()


def test_degenerate_lab_replicates_are_not_estimable(monkeypatch):
    raw, families, groups, labs = matrix_fixture()
    raw["lab__albumin__value"] = 1.
    cfg = config()
    cfg["bootstrap"]["replicates"] = 2
    monkeypatch.setattr(M, "select_features", lambda values, *a: (values, pd.DataFrame(), pd.DataFrame()))
    result = M.bootstrap_stability(raw.to_dict("series"), families, pd.DataFrame(), "S4", [{0, 1}], cfg,
                                   groups, labs)
    assert result.qc_status == "INVALID_BOOTSTRAP"
    assert result.replicate_audit.status.eq("not_estimable").all()
    assert result.replicate_audit.failure_type.eq("ZERO_VARIANCE_BLOCK").all()
    assert len(result.balance_audit) == 6


def test_mapping_error_is_blocking_before_any_replicate(monkeypatch):
    values = {"lab__unknown__value": pd.Series(range(10)), "pro": pd.Series(range(10))}
    monkeypatch.setattr(M, "select_features", lambda *a, **k: pytest.fail("selection should not start"))
    with pytest.raises(M.MappingError, match="unknown"):
        M.bootstrap_stability(values, {"lab__unknown__value": "LAB", "pro": "PRO"},
                              pd.DataFrame(), "S4", [], config(), {},
                              {"lab__unknown__value": "lab__unknown"})


def test_approved_ace_and_hepatitis_surface_antibody_mapping():
    mapping = M.load_lab_group_map(M.DEFAULT_LAB_GROUP_MAP).set_index("lab")
    assert mapping.loc["angiotensin_converting_enzyme", "s4_group"] == "INFLAMMATION_TISSUE"
    assert mapping.loc["hepatitis_b_surface_antibody", "s4_group"] == "INFECTIOUS_SCREENING"
    assert mapping.loc[["angiotensin_converting_enzyme", "hepatitis_b_surface_antibody"], "mapping_status"].eq("MAPPED").all()


def test_common_intersection_uses_same_patients_and_rejects_small_samples():
    memberships = {}
    for scenario in M.SCENARIOS:
        memberships[scenario] = pd.DataFrame({"patient_id": list("abcd"), "mapper_covered": True,
            "community_membership_tie": False, "community_supported": [True, True, True, scenario != "S4"],
            "hard_community": [0, 0, 1, 1]})
    cfg = config()
    cfg["comparisons"].update(minimum_common_patients=3)
    result = M.common_intersection_ari(memberships, cfg)
    assert len(result) == 10
    assert result.n_patients_used.eq(3).all()
    assert result.fraction_total_used.eq(.75).all()
    assert result.ari.eq(1.).all()
    cfg["comparisons"]["minimum_common_patients"] = 4
    assert M.common_intersection_ari(memberships, cfg).ari.isna().all()


def test_interpretation_cannot_claim_multiple_branches_from_single_stable_core():
    summary = pd.DataFrame({"scenario": list(M.SCENARIOS), "n_stable_communities": 1,
                            "bootstrap_qc_status": "BOOTSTRAP_QC_PASS"})
    result = M.interpretation_check(summary, config=config())
    assert result["COMPUTATIONAL_QC"] == "COMPUTATIONAL_QC_PASS"
    assert result["GLOBAL_PARTITION_ROBUSTNESS"] == "NOT_EVALUABLE"
    summary.loc[summary.scenario.eq("S4"), "bootstrap_qc_status"] = "BOOTSTRAP_QC_FAIL"
    summary.loc[summary.scenario.eq("S4"), "n_stable_communities"] = np.nan
    result = M.interpretation_check(summary, config=config())
    assert result["COMPUTATIONAL_QC"] == "COMPUTATIONAL_QC_FAIL"
    assert result["CORE_EVIDENCE"] == result["GLOBAL_PARTITION_ROBUSTNESS"] == "NOT_EVALUABLE"


def test_explicit_interpretation_thresholds_need_justification_and_all_qc():
    cfg = config()
    cfg["interpretation"].update(core_jaccard_threshold=.5, global_ari_threshold=.8)
    summary = pd.DataFrame({"scenario": list(M.SCENARIOS), "n_stable_communities": 1,
                            "bootstrap_qc_status": "BOOTSTRAP_QC_PASS"})
    with pytest.raises(ValueError, match="threshold_justification"):
        M.interpretation_check(summary, config=cfg)
    cfg["interpretation"]["threshold_justification"] = "Synthetic test criteria, not clinical cutoffs"
    pairs = []
    for i, left in enumerate(M.SCENARIOS):
        for right in list(M.SCENARIOS)[i+1:]:
            pairs.append({"scenario_a": left, "scenario_b": right, "jaccard": 1.,
                          "n_a": 30, "n_b": 30, "ari": 1., "n_patients_used": 30,
                          "fraction_total_used": 1.})
    pairs = pd.DataFrame(pairs)
    result = M.interpretation_check(summary, pairs, pairs, cfg)
    assert result["CORE_EVIDENCE"] == "SUPPORTED"
    assert result["GLOBAL_PARTITION_ROBUSTNESS"] == "UNSUPPORTED"  # Single stable core, not branches.
    summary.loc[summary.scenario.eq("S4"), "bootstrap_qc_status"] = "BOOTSTRAP_QC_FAIL"
    result = M.interpretation_check(summary, pairs, pairs, cfg)
    assert result["CORE_EVIDENCE"] == result["GLOBAL_PARTITION_ROBUSTNESS"] == "NOT_EVALUABLE"


def test_concat_redundancy_preserves_schema_and_nonempty_rows_without_futurewarning():
    empty = pd.DataFrame(columns=M.REDUNDANCY_DTYPES)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = M.concat_redundancy_audits([empty, empty])
        assert result.empty and list(result) == list(M.REDUNDANCY_DTYPES)
        one = pd.DataFrame([{"scenario": "S2", "feature_kept": "a", "feature_removed": "b",
                             "spearman_rho": np.nan, "n_pairwise": 3, "coverage_kept": 1.,
                             "coverage_removed": .9, "reason": "configured"}])
        result = M.concat_redundancy_audits([empty, one, empty])
        assert len(result) == 1
        assert result.scenario.item() == "S2"
        assert str(result.n_pairwise.dtype) == "Int64"

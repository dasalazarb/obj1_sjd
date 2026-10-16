"""Representation, governance and privacy tests; all records are synthetic."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "src/studies/pharma/00_profile_labs.py"
SPEC = importlib.util.spec_from_file_location("pharma_lab_profile", SCRIPT)
profile_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile_module)


def episode_frame(n=12):
    patients = [f"SYNTHETIC_PATIENT_{i // 2}" for i in range(n)]
    return pd.DataFrame({
        "patient_id": patients, "clinical_episode_id": [f"episode_{i}" for i in range(n)],
        "clinical_anchor_date": pd.to_datetime(["2020-01-01" if i % 2 == 0 else "2021-01-01" for i in range(n)]),
        "clinical_visit_number": [i % 2 + 1 for i in range(n)], "clinical_visit": True,
        "is_clinical_baseline": [i % 2 == 0 for i in range(n)],
        "pop__status": "Pop1", "essdai__total": 0., "esspri__total": 0.,
    })


def approved_registry(*labs, provisional=False, binary=False):
    """Test-only decisions exercise gates; no actual clinical approvals exist."""
    registry = profile_module.load_registry()
    for lab in labs:
        registry["assays"][lab] = {
            "approval_status": "approved", "decision_id": "SYNTHETIC_TEST_ONLY",
            "reviewer": "synthetic_fixture", "evidence": "unit_test_not_a_clinical_decision",
            "numeric_approved": True, "binary_approved": binary,
            "allow_unassessed_units": provisional,
        }
    return registry


def test_default_output_is_absolute_and_context_flags_exclusive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = profile_module.parse_args([])
    assert args.output == SCRIPT.parents[3] / "outputs/tables/studies/pharma/00_profile_labs/00_pharma_lab_profile.csv"
    with pytest.raises(SystemExit):
        profile_module.parse_args(["--context", "context.parquet", "--no-context"])


def test_discovery_is_strict_public_results_only():
    frame = episode_frame()
    for column in ["lab__ana_status__text", "lab__nested__assay__value", "lab__metadata_only__reference_status",
                   "ext__eye__value", "essdai__domain__value", "ovl__foo__text", "pro__score__value",
                   "esspri__total__value", "sero__ana__asof_value", "unprefixed__value"]:
        frame[column] = "POSITIVE"
    result = profile_module.profile_labs(frame)
    assert result.lab.tolist() == ["lab__ana_status", "lab__nested__assay"]
    assert (set(result.public_column) - {""}).issubset(frame.columns)


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "duplicate"), ("baseline", "baseline"), ("null", "null"), ("date", "clinical_anchor_date")])
def test_granularity_rejects_invalid_input(mutation, match):
    frame = episode_frame()
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif mutation == "baseline":
        frame.loc[1, "is_clinical_baseline"] = True
    elif mutation == "null":
        frame.loc[1, "patient_id"] = None
    else:
        frame.loc[1, "clinical_anchor_date"] = pd.NaT
    with pytest.raises(ValueError, match=match):
        profile_module.profile_labs(frame)


@pytest.mark.parametrize("token,direction,operator,bound", [
    ("<5", "left", "<", 5), (">1300", "right", ">", 1300), ("< 0.15", "left", "<", .15),
    ("<=30", "left", "<=", 30), ("≥ 10", "right", ">=", 10), ("<3%", "left", "<", 3)])
def test_censoring_is_not_a_point(token, direction, operator, bound):
    parsed = profile_module.parse_result(token)
    assert parsed["kind"] == f"censored_{direction}"
    assert parsed["operator"] == operator
    assert parsed["bound"] == bound
    frame = episode_frame(2)
    frame["lab__anti_ccp__text"] = [token, "SEE NOTE"]
    row = profile_module.profile_labs(frame).iloc[0]
    assert row.n_numeric_episodes == 0
    assert row.n_categorical_interpretable_episodes == 0
    assert row.n_unstructured_text == 1
    assert not row.model_ready


@pytest.mark.parametrize("token,category", [("POSITIVE", "positive"), ("NEGATIVE", "negative"), ("NEG", "negative"),
                                            ("POS", "positive"), ("REACTIVE", "positive"), ("NONREACTIVE", "negative")])
def test_binary_lexical_codes_do_not_confer_assay_approval(token, category):
    assert profile_module.parse_result(token)["category"] == category
    frame = episode_frame()
    frame["lab__ana_status__text"] = [token, "NEGATIVE"] * 6
    row = profile_module.profile_labs(frame).iloc[0]
    assert row.clinically_approved == "pending"
    assert not row.model_ready


def test_free_text_unknowns_and_ordinals_never_become_negative():
    for token in ["SEE NOTE", "SEE COMMENT", "not positive for this assay", "UNKNOWN", "NOT REPORTED", "TRACE", "1+", "2+", "3+"]:
        assert profile_module.parse_result(token)["category"] is None
    rule = {"approval_status": "approved", "ordinal_categories": ["negative", "trace", "1+", "2+", "3+"]}
    assert profile_module.parse_result("2+", rule)["order"] == 3
    assert profile_module.parse_result("2+", rule)["kind"] == "ordinal_approved"
    rule["approval_status"] = "pending"
    assert profile_module.parse_result("2+", rule)["category"] is None


def test_five_values_do_not_borrow_support_from_443_texts():
    frame = episode_frame(444)
    frame["lab__anticardiolipin_igg__value"] = list(range(5)) + [np.nan] * 439
    frame["lab__anticardiolipin_igg__text"] = ["POSITIVE", "NEGATIVE"] * 221 + ["POSITIVE", None]
    row = profile_module.profile_labs(frame, registry=approved_registry("lab__anticardiolipin_igg", provisional=True)).iloc[0]
    assert row.n_numeric_episodes == 5
    assert row.n_raw_text_episodes == 443
    assert row.n_patients_with_ge2_results == 221
    assert row.n_numeric_patients_ge2_dates == 2
    assert not row.numeric_support_met
    assert row.recommended_use == "descriptive_only"
    assert row.value_support_status == "limited_for_longitudinal"


def test_date_and_episode_repetition_are_distinct():
    frame = episode_frame()
    frame.loc[1, "clinical_anchor_date"] = frame.loc[0, "clinical_anchor_date"]
    frame["lab__glucose__value"] = np.arange(12)
    frame["lab__ana_status__text"] = ["POSITIVE", "NEGATIVE"] * 6
    profile = profile_module.profile_labs(frame).set_index("lab")
    assert profile.loc["lab__glucose", "n_patients_ge2_numeric_episodes"] == 6
    assert profile.loc["lab__glucose", "n_patients_ge2_numeric_dates"] == 5
    assert profile.loc["lab__ana_status", "n_categorical_patients_ge2_dates"] == 5
    frame.loc[0, "is_clinical_baseline"] = False
    row = profile_module.profile_labs(frame).set_index("lab").loc["lab__glucose"]
    assert row.n_baseline_denominator == 5
    assert row.pct_baseline_numeric == 1


def test_unknown_metadata_is_not_no_conflicts_or_single_unit():
    frame = episode_frame()
    frame["lab__glucose__value"] = np.arange(12)
    row = profile_module.profile_labs(frame).iloc[0]
    assert row.unit_assessment_status == "not_assessed"
    assert row.unit_availability_status == "not_available_in_integrated"
    assert row.conflict_assessment_status == "not_evaluable_without_upstream_qc"
    assert pd.isna(row.has_conflict)
    assert pd.isna(row.multiple_units)
    assert row.metadata_source == "none"
    assert row.recommended_use == "review_required"  # Pending approval, not absent units alone.
    provisional = profile_module.profile_labs(frame, registry=approved_registry("lab__glucose", provisional=True)).iloc[0]
    assert provisional.recommended_use == "numeric_longitudinal"
    assert provisional.unit_assessment_status == "not_assessed"
    assert "not_harmonized" in provisional.eligibility_reason


def test_context_preserves_rows_order_and_all_master_values():
    frame = episode_frame().iloc[::-1].copy()
    frame["lab__glucose__value"] = np.arange(12)
    context = frame[["patient_id", "clinical_episode_id"]].copy().iloc[::-1]
    context["lab__glucose__unit"] = "mg/dL"
    context["lab__glucose__reference_status"] = ["HIGH", "NORMAL"] * 6
    context["lab__glucose__measurement_date"] = pd.Timestamp("2020-01-01")
    context["lab__glucose__conflict"] = True  # QC must not be imported from context.
    context["lab__glucose__value"] = -999
    context["sero__future__patient_consensus_value"] = 1
    joined, info = profile_module.join_lab_context(frame, context)
    pd.testing.assert_frame_equal(joined[frame.columns], frame)
    assert "lab__glucose__conflict" not in joined
    assert "sero__future__patient_consensus_value" not in joined
    assert info["n_context_keys_unmatched"] == 0
    row = profile_module.profile_labs(frame, context, registry=approved_registry("lab__glucose")).iloc[0]
    assert row.n_units == 1
    assert row.reference_assessment_status == "available"
    assert row.metadata_source == "context"
    assert row.n_categorical_interpretable_episodes == 0
    assert row.result_source == "value"
    assert row.recommended_use == "numeric_longitudinal"


def test_context_missing_metadata_unmatched_keys_and_multiple_units():
    frame = episode_frame()
    frame["lab__glucose__value"] = np.arange(12)
    context = frame[["patient_id", "clinical_episode_id"]].copy()
    empty = profile_module.profile_labs(frame, context).iloc[0]
    assert empty.unit_availability_status == "not_present_in_context"
    context["lab__glucose__unit"] = ["mg/dL", "mmol/L"] * 6
    row = profile_module.profile_labs(frame, context, registry=approved_registry("lab__glucose")).iloc[0]
    assert row.recommended_use == "numeric_requires_harmonization"
    assert row.graph_candidate_status == "review_before_graph"
    assert row.n_units == 2
    _, info = profile_module.join_lab_context(frame, context.iloc[:-1])
    assert info["n_context_keys_unmatched"] == 1
    row = profile_module.profile_labs(frame, context.iloc[2:], registry=approved_registry("lab__glucose")).iloc[0]
    assert row.graph_candidate_status != "include_numeric"


@pytest.mark.parametrize("mutation", ["duplicate", "null", "missing_key"])
def test_context_rejects_invalid_keys(mutation):
    frame = episode_frame()
    context = frame[["patient_id", "clinical_episode_id"]].copy()
    if mutation == "duplicate":
        context = pd.concat([context, context.iloc[:1]])
    elif mutation == "null":
        context.loc[0, "clinical_episode_id"] = None
    else:
        context = context.drop(columns="clinical_episode_id")
    with pytest.raises(ValueError):
        profile_module.join_lab_context(frame, context)


def test_exact_concepts_never_pool_related_assays_or_suffixes():
    frame = episode_frame()
    stems = ["ana_status", "ana_hep2_status", "crp_high_sensitivity", "anti_ro_ssa", "anti_ro52", "anti_ro60", "igg",
             "anti_dsdna_igg", "iga", "anti_endomysial_iga", "igm", "anticardiolipin_igm", "wbc", "urine_wbc",
             "hemoglobin", "urine_hemoglobin", "glucose", "estimated_average_glucose", "cryoglobulins"]
    for stem in stems:
        frame[f"lab__{stem}__text"] = "SEE COMMENT"
    audit = profile_module.expected_lab_audit(profile_module.profile_labs(frame)).set_index("expected_lab")
    assert len(audit) == 30
    for concept, expected in [("ana", "ana_status"), ("crp", "crp_high_sensitivity"), ("igg", "igg"), ("iga", "iga"),
                              ("igm", "igm"), ("wbc", "wbc"), ("hemoglobin", "hemoglobin"), ("glucose", "glucose")]:
        assert audit.loc[concept, "matched_lab_stem"] == f"lab__{expected}"
        assert audit.loc[concept, "preferred_match_status"] == "exact"
    assert audit.loc["ana", "related_lab_stems"] == "lab__ana_hep2_status"
    assert audit.loc["anti_ro_ssa", "related_lab_stems"] == "lab__anti_ro52 | lab__anti_ro60"
    assert audit.loc["cryoglobulin", "match_status"] == "found"
    frame = frame.drop(columns="lab__anti_ro_ssa__text")
    anti_ssa = profile_module.expected_lab_audit(profile_module.profile_labs(frame)).set_index("expected_lab").loc["anti_ro_ssa"]
    assert anti_ssa.matched_lab_stem == ""
    assert anti_ssa.preferred_match_status == "not_found"
    assert anti_ssa.match_status == "multiple_distinct_assays_review"


def test_unprefixed_legacy_discovery_is_deliberately_replaced_not_guessed():
    frame = episode_frame()
    frame["ana__reference_status"] = "positive"
    frame["ro52__value"] = 1
    assert profile_module.profile_labs(frame).empty


def test_approval_requires_evidence(tmp_path):
    registry = approved_registry("lab__glucose")
    registry["assays"]["lab__glucose"]["evidence"] = ""
    path = tmp_path / "registry.yaml"
    import yaml
    path.write_text(yaml.safe_dump(registry))
    with pytest.raises(ValueError, match="evidence"):
        profile_module.load_registry(path)


def test_mixed_clinical_text_is_visible_and_safe_but_not_model_ready():
    frame = episode_frame()
    for lab in ["ana_status", "cryoglobulins", "anti_ccp", "urine_protein_qualitative"]:
        frame[f"lab__{lab}__text"] = ["POSITIVE", "NEGATIVE", "<5", "SEE NOTE", "TRACE", "UNKNOWN"] * 2
    profile = profile_module.profile_labs(frame)
    assert profile.data_present.all()
    assert profile.status.eq("requires_clinical_text_review").all()
    assert profile.n_categorical_interpretable_episodes.eq(4).all()
    assert profile.n_left_censored.eq(2).all()
    assert not profile.model_ready.any()


def test_approved_raw_binary_contract_and_mixed_aliases_are_gated():
    frame = episode_frame()
    frame["lab__ana_status__text"] = ["POSITIVE", "NEGATIVE"] * 6
    registry = approved_registry("lab__ana_status", binary=True)
    row = profile_module.profile_labs(frame, registry=registry).iloc[0]
    assert row.recommended_use == "categorical_longitudinal"
    frame.loc[1, "lab__ana_status__text"] = "NEG"
    row = profile_module.profile_labs(frame, registry=registry).iloc[0]
    assert row.recommended_use == "review_required"
    assert row.status == "requires_consumer_encoding_review"


def test_genotype_is_not_dynamic_even_if_numeric_and_approved():
    frame = episode_frame()
    frame["lab__hla_drb1__value"] = np.arange(12)
    row = profile_module.profile_labs(frame, registry=approved_registry("lab__hla_drb1", provisional=True)).iloc[0]
    assert row.status == "genetic_descriptive_only"
    assert row.graph_candidate_status == "exclude"


def test_row_level_impact_uses_real_old_rows_and_does_not_invent_history():
    frame = episode_frame()
    frame["lab__glucose__value"] = np.arange(12)
    profile = profile_module.profile_labs(frame)
    old = profile.copy()
    old["recommended_use"] = "numeric_longitudinal"
    old["graph_candidate_status"] = "include_numeric"
    impact = profile_module.downstream_impact(profile, old).iloc[0]
    assert impact.impact_pharma01 == "exits"
    assert impact.impact_graph01 == "exits"
    assert impact.requires_manual_approval
    assert impact.n_numeric_episodes == 12
    unknown = profile_module.downstream_impact(profile, None).iloc[0]
    assert unknown.old_eligible == "unknown"
    assert unknown.impact_graph01.startswith("not_comparable")


def test_reports_redact_arbitrary_free_text_and_identifiers():
    frame = episode_frame()
    frame["lab__ana_status__text"] = ["POSITIVE", "NEGATIVE", "PATIENT NAME SECRET SYNTHETIC", "SEE COMMENT"] * 3
    outputs = [profile_module.profile_labs(frame).to_csv(index=False), profile_module.text_code_audit(frame).to_csv(index=False)]
    for output in outputs:
        assert "PATIENT NAME SECRET SYNTHETIC" not in output
        assert "SYNTHETIC_PATIENT_0" not in output
        assert "patient_id" not in output
        assert "[UNSTRUCTURED_TEXT_REDACTED]" in output


def test_empty_inventory_keeps_readable_csv_headers():
    profile = profile_module.profile_labs(episode_frame())
    assert profile.empty
    assert "result_source" in profile
    assert "lab" in profile_module.graph_candidates(profile)
    assert "lab" in profile_module.downstream_impact(profile, None)


def test_nonfinite_values_cannot_reach_raw_numeric_consumers():
    frame = episode_frame(14)
    frame["lab__glucose__value"] = list(range(13)) + [np.inf]
    row = profile_module.profile_labs(frame, registry=approved_registry("lab__glucose", provisional=True)).iloc[0]
    assert row.n_numeric_episodes == 13
    assert row.n_nonfinite_numeric_values == 1
    assert row.graph_candidate_status == "review_before_graph"
    assert row.status == "requires_numeric_quality_review"


def test_partial_qc_cannot_assert_no_conflicts():
    frame = episode_frame()
    frame["lab__glucose__value"] = np.arange(12)
    frame["lab__glucose__conflict"] = [False] + [None] * 11
    row = profile_module.profile_labs(frame).iloc[0]
    assert pd.isna(row.has_conflict)
    assert row.conflict_assessment_status == "incomplete_qc_review"
    frame["lab__glucose__conflict"] = False
    assert profile_module.profile_labs(frame).iloc[0].has_conflict == False


def test_all_historical_profile_fields_remain_additive():
    frame = episode_frame()
    frame["lab__glucose__value"] = np.arange(12)
    historical = {
        'between_patient_variation',
        'conflict_column',
        'constant_result',
        'data_sufficiency',
        'days_from_anchor_column',
        'discordant_value_vs_reference_status',
        'episode_status_column',
        'exclusion_reason',
        'graph_candidate_status',
        'graph_encoding_hint',
        'has_dual_representation',
        'high_missingness',
        'inferred_type',
        'lab',
        'measurement_date_column',
        'mixed_numeric_text',
        'multiple_candidate_result_sources',
        'multiple_units',
        'n_any_result',
        'n_baseline_any_result',
        'n_baseline_categorical',
        'n_baseline_numeric',
        'n_baseline_patients',
        'n_measurements_column',
        'n_nonmissing',
        'n_patients',
        'n_patients_any_result',
        'n_patients_with_ge1_result',
        'n_patients_with_ge2_results',
        'n_patients_with_ge3_results',
        'n_reference_status_nonmissing',
        'n_reference_status_unique',
        'n_text_nonmissing',
        'n_text_unique',
        'n_unique',
        'n_units',
        'n_value_nonmissing',
        'n_value_unique',
        'numeric_max',
        'numeric_median',
        'numeric_min',
        'numeric_parse_pct',
        'pct_any_result',
        'pct_baseline_any_result',
        'pct_baseline_categorical',
        'pct_baseline_numeric',
        'pct_nonmissing',
        'pct_reference_status_nonmissing',
        'pct_text_nonmissing',
        'pct_value_nonmissing',
        'primary_result_source',
        'public_column',
        'recommended_use',
        'reference_status_column',
        'reference_status_unique',
        'requires_unit_review',
        'result_source',
        'review_reason',
        'sample_values',
        'selection_status_column',
        'source_column',
        'text_column',
        'text_sample',
        'text_unique_sample',
        'unit_column',
        'units_unique',
        'value_column',
        'very_low_availability',
        'within_patient_variation',
    }
    assert historical <= set(profile_module.profile_labs(frame).columns)

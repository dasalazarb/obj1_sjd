"""Synthetic safety and reconciliation tests for medication documentation release."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

MODULE = Path(__file__).resolve().parents[1] / "src/block_A/08_build_medication_dataset.py"
spec = importlib.util.spec_from_file_location("step08_medications_release", MODULE)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def row(**overrides):
    base = dict(
        audit_record_id="11D:0", source_kind="BTRIS", source="11D",
        linked_patient_pseudonymous="pseudo1", eligible_linked=True,
        name_present=True, ingredient_canonical="hydroxychloroquine",
        ingredients_canonical=json.dumps(["hydroxychloroquine"]),
        drug_short_label="Hydroxychloroquine",
        clinical_category_primary="systemic_sjd",
        clinical_category_detail="sjd_systemic_candidate",
        classification_confidence="high_exact",
        study_category_rule_status="candidate_not_adjudicated",
        route_canonical="oral", formulation_canonical="oral_tablet",
        strength_canonical="200 mg", start_status="valid",
        start_precision="datetime", start_lower_bound="2020-06-01",
        start_upper_bound="2020-06-01", start_parsed="2020-06-01T14:20:00",
        end_status="valid", end_precision="datetime",
        end_lower_bound="2020-06-01", end_upper_bound="2020-06-01",
        end_parsed="2020-06-01T14:20:00", order_interval_qc="zero",
        time_semantics="order_or_start_unresolved",
        end_semantics="order_validity_end_not_confirmed_discontinuation",
        review_reasons="", qc_flags="", classification_version="08-classification-2.1.10",
        source_sha256="sha",
        evidence_type="documented_order", visit_raw="",
        visit_semantics="",
    )
    base.update(overrides)
    return base


def source_summary(n):
    return dict(
        run_status="completed_with_review_gates",
        classification_version="08-classification-2.1.10",
        reconciliation={"integrity": True},
        source_preservation={"11D": True, "15D": True, "CTDB": True},
        fingerprints={"11D": {"sha256": "sha"}},
        n_order_rows=n,
    )


def test_btris_order_date_is_documentation_not_exposure():
    rec = pd.DataFrame([row()])
    catalog, dated, normalized, qc, summary = release.build_medication_release(
        rec, source_summary(1))
    assert len(normalized) == len(dated) == len(catalog) == 1
    assert catalog.iloc[0].date_evidence_kind == "btris_order_or_start_date_unvalidated"
    assert catalog.iloc[0].date_of_documentation == "2020-06-01"
    assert not catalog.iloc[0].actual_medication_exposure_confirmed
    assert not catalog.iloc[0].treatment_end_confirmed
    assert summary["gates"]["treatment_episodes_or_active_on_date"] == "NO_GO"


def test_ctdb_uses_visit_observation_when_start_missing_not_exposure_start():
    rec = pd.DataFrame([row(
        audit_record_id="CTDB:0:rx:1", source="CTDB", source_kind="CTDB",
        start_status="missing", start_precision="", start_parsed="",
        start_lower_bound="", start_upper_bound="",
        visit_raw="2020-07-06", visit_semantics="reported_visit_date",
        source_sha256="ctdbsha")])
    summary = source_summary(0)
    summary["fingerprints"] = {"CTDB": {"sha256": "ctdbsha"}}
    catalog, dated, *_ = release.build_medication_release(rec, summary)
    assert catalog.iloc[0].date_evidence_kind == "ctdb_current_medication_observed_at_visit"
    assert catalog.iloc[0].source_start_date_exact == ""
    assert catalog.iloc[0].date_of_documentation == "2020-07-06"
    assert len(dated) == 1


def test_year_only_and_concatenated_visit_never_imputed():
    rec = pd.DataFrame([row(
        audit_record_id="CTDB:0:rx:1", source="CTDB", source_kind="CTDB",
        start_status="valid", start_precision="year", start_parsed="",
        start_lower_bound="2020-01-01", start_upper_bound="2020-12-31",
        visit_raw="2020-01-01 | 2021-02-03",
        visit_semantics="concatenated_visits_no_arbitrary_date",
        source_sha256="ctdbsha")])
    summary = source_summary(0)
    summary["fingerprints"] = {"CTDB": {"sha256": "ctdbsha"}}
    catalog, dated, *_ = release.build_medication_release(rec, summary)
    assert catalog.iloc[0].date_of_documentation == ""
    assert len(dated) == 0
    assert catalog.iloc[0].source_start_lower_bound == "2020-01-01"


def test_conflict_quarantined_without_losing_dated_row():
    rec = pd.DataFrame([row(
        qc_flags="end_definitely_before_start",
        end_parsed="2020-05-01T00:00:00",
        end_lower_bound="2020-05-01", end_upper_bound="2020-05-01")])
    catalog, dated, normalized, *_ = release.build_medication_release(
        rec, source_summary(1))
    assert len(catalog) == len(dated) == 1
    assert len(normalized) == 1
    assert catalog.iloc[0].source_end_date_exact_unvalidated == ""
    assert catalog.iloc[0].documentation_dated_usable
    assert catalog.iloc[0].qc_has_warning_only
    assert not catalog.iloc[0].actual_medication_exposure_confirmed


def test_unmapped_and_undated_kept_in_protected_catalog():
    rec = pd.DataFrame([row(
        start_status="missing", start_parsed="", start_precision="",
        ingredient_canonical="", ingredients_canonical="[]",
        drug_short_label="", classification_confidence="unmapped",
        clinical_category_primary="indeterminate")])
    catalog, dated, normalized, *_ = release.build_medication_release(
        rec, source_summary(1))
    assert len(catalog) == 1
    assert len(dated) == len(normalized) == 0


def test_fingerprint_and_version_fail_closed():
    rec = pd.DataFrame([row()])
    with pytest.raises(ValueError, match="source_fingerprint"):
        release.build_medication_release(rec, {
            **source_summary(1), "fingerprints": {"11D": {"sha256": "wrong"}}})
    with pytest.raises(ValueError, match="classification_version"):
        release.build_medication_release(rec, {
            **source_summary(1), "classification_version": "08-classification-2.1.8"})


def test_patient_id_release_requires_eligibility_and_key():
    rec = pd.DataFrame([row()])
    with pytest.raises(ValueError, match="pseudonym_key_required"):
        release.build_medication_release(
            rec, source_summary(1), eligibility=pd.DataFrame({"patient_id": ["123"]}))


def test_v101_normalized_numeric_linkage_warning_preserves_dated_order():
    rec = pd.DataFrame([row(
        qc_flags="nontext_mrn_normalized_match_requires_identifier_review")])
    catalog, dated, normalized, qc, summary = release.build_medication_release(
        rec, source_summary(1))
    assert len(dated) == len(normalized) == 1
    assert qc.iloc[0].n_warning_only_rows == 1
    assert qc.iloc[0].n_major_conflict_rows == 0
    assert not catalog.iloc[0].actual_medication_exposure_confirmed


def test_v101_true_product_conflict_remains_quarantined():
    rec = pd.DataFrame([row(review_reasons="ingredient_product_conflict")])
    catalog, dated, normalized, qc, summary = release.build_medication_release(
        rec, source_summary(1))
    assert len(dated) == 1
    assert len(normalized) == 0
    assert not catalog.iloc[0].documentation_dated_usable

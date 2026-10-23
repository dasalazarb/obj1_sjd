"""Release a protected, dated MEDICATION DOCUMENTATION dataset from Step 08.

This is NOT a drug-exposure or treatment-era reconstruction. BTRIS dates are
order-or-start dates of unresolved meaning; CTDB dates are reported medication
dates or an observed visit date. No order status is equated to dispensing or
administration, and no end date is used to infer a treatment duration.

Run after 08_treatment_audit_v2.py. Never read/write Step 10, 14, 15, pharma
or graph datasets. An optional explicitly authorized patient-ID release allows
joining the dated documentation to a future analysis, but grants no approval
for causal treatment-effect or actual-exposure claims.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path

import pandas as pd

VERSION = "08-medication-documentation-release-1.0.0"
INPUT_BASENAME = "08_treatment_record_classification_audit.parquet"
SUMMARY_BASENAME = "08_treatment_classification_v21_summary.json"
BAD_QC = {
    "end_definitely_before_start",
    "formulation_route_conflict",
    "ingredient_product_conflict",
    "protocol_folder_not_confirmed_by_value",
    "cross_protocol_possible_duplicate_retained",
    "same_order_id_revision_or_discordance_retained",
    "exact_reexport_duplicate_retained",
    "nontext_mrn_not_linkable",
    "nontext_mrn_normalized_match_requires_identifier_review",
    "ambiguous_key",
    "concatenated_visit_dates_temporal_linkage_unresolved",
    "order_after_documented_death_verify",
}
BAD_CLASSIFICATION = {
    "ingredient_product_conflict",
    "product_components_unresolved",
    "formulation_route_conflict",
}
REQUIRED = {
    "audit_record_id", "source_kind", "source", "linked_patient_pseudonymous",
    "eligible_linked", "name_present", "ingredient_canonical",
    "ingredients_canonical", "drug_short_label", "clinical_category_primary",
    "clinical_category_detail", "classification_confidence",
    "study_category_rule_status", "route_canonical", "formulation_canonical",
    "strength_canonical", "start_status", "start_precision",
    "start_lower_bound", "start_upper_bound", "start_parsed",
    "end_status", "end_precision", "end_lower_bound", "end_upper_bound",
    "end_parsed", "order_interval_qc", "time_semantics", "end_semantics",
    "review_reasons", "qc_flags", "classification_version", "source_sha256",
}

def _has(value):
    return pd.notna(value) and str(value).strip() not in {"", "nan", "None"}

def _machine_tokens(value):
    return set(filter(None, str(value or "").split("|")))

def _exact_date(date_value, status, precision):
    if status != "valid" or precision not in ("day", "datetime") or not _has(date_value):
        return ""
    # Prefix date comes directly from an already validated exact-granularity
    # source date; never impute a day from month/year bounds.
    value = str(date_value)
    day = value[:10]
    try:
        return pd.Timestamp(day).date().isoformat() if len(day) == 10 else ""
    except (ValueError, TypeError, OverflowError):
        return ""

def _visit_day(value):
    """Strict single-date fallback for CTDB visit OBSERVATION only, never start."""
    if not _has(value) or "|" in str(value):
        return ""
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%m/%d/%Y", "%m/%d/%Y %H:%M:%S"):
        try:
            return pd.to_datetime(value, format=fmt, errors="raise").date().isoformat()
        except (ValueError, TypeError):
            continue
    return ""

def _required_data(frame, summary):
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("audit_schema_missing_fields:" + ",".join(sorted(missing)))
    if not frame.audit_record_id.is_unique:
        raise ValueError("duplicate_audit_record_id")
    if not summary.get("reconciliation") or not all(summary["reconciliation"].values()):
        raise ValueError("audit_reconciliation_failed")
    if summary.get("run_status") != "completed_with_review_gates":
        raise ValueError("audit_run_not_completed")
    if any(v is not True for v in summary.get("source_preservation", {}).values()):
        raise ValueError("source_preservation_not_confirmed")
    if not summary.get("classification_version", "").startswith("08-classification-"):
        raise ValueError("unknown_classification_version")
    if frame.classification_version.dropna().nunique() > 1 or (
        len(frame) and frame.classification_version.iloc[0] != summary["classification_version"]
    ):
        raise ValueError("classification_version_mismatch")
    if int((frame.source_kind == "BTRIS").sum()) != summary.get("n_order_rows"):
        raise ValueError("btris_rows_not_reconciled")
    for name, info in summary.get("fingerprints", {}).items():
        if name not in {"11D", "15D", "CTDB"} or not info.get("sha256"):
            continue
        rows = frame[frame.source == name]
        if len(rows) and not rows.source_sha256.eq(info["sha256"]).all():
            raise ValueError("source_fingerprint_mismatch:" + name)

def _row_to_document(row):
    qc = _machine_tokens(row.qc_flags)
    review = _machine_tokens(row.review_reasons)
    conflict = bool(qc & BAD_QC or review & BAD_CLASSIFICATION)
    event_day, event_kind = "", "undated"
    start_day = _exact_date(row.start_parsed, row.start_status, row.start_precision)
    end_day = _exact_date(row.end_parsed, row.end_status, row.end_precision)
    if row.source_kind == "BTRIS":
        if start_day:
            event_day, event_kind = start_day, "btris_order_or_start_date_unvalidated"
    elif row.source_kind == "CTDB":
        if start_day:
            event_day, event_kind = start_day, "ctdb_reported_medication_start_unvalidated"
        else:
            observed_day = _visit_day(getattr(row, "visit_raw", ""))
            visit_uncertain = getattr(row, "visit_semantics", "") == "concatenated_visits_no_arbitrary_date"
            if observed_day and not visit_uncertain:
                event_day, event_kind = observed_day, "ctdb_current_medication_observed_at_visit"
    # BTRIS end date is a validity/order boundary of unknown meaning, not a
    # discontinuation. A contradictory end date must remain quarantined.
    end_reviewed = bool(end_day and start_day and end_day >= start_day and
                        "end_definitely_before_start" not in qc)
    category = str(row.clinical_category_primary or "indeterminate")
    named_drug = _has(row.ingredient_canonical)
    signed_product = row.study_category_rule_status == "approved_by_investigator"
    mapped = named_drug or signed_product or _has(row.drug_short_label)
    date_quality = "exact_day" if event_day else "no_exact_observation_day"
    # Only two descriptive gates. No inferred exposure, treatment period,
    # adherence, disease indication or pharmacological effect.
    documented_date_usable = bool(event_day and not conflict)
    normalized_date_usable = bool(
        documented_date_usable and mapped and category != "indeterminate")
    return {
        "audit_record_id": row.audit_record_id,
        "linked_patient_pseudonymous": row.linked_patient_pseudonymous,
        "source": row.source,
        "source_kind": row.source_kind,
        "evidence_type": getattr(row, "evidence_type", ""),
        "clinical_category_primary": category,
        "clinical_category_detail": row.clinical_category_detail,
        "study_category_rule_status": row.study_category_rule_status,
        "drug_short_label": row.drug_short_label,
        "ingredient_canonical": row.ingredient_canonical,
        "ingredients_canonical": row.ingredients_canonical,
        "classification_confidence": row.classification_confidence,
        "route_canonical": row.route_canonical,
        "formulation_canonical": row.formulation_canonical,
        "strength_canonical": row.strength_canonical,
        "date_of_documentation": event_day,
        "date_evidence_kind": event_kind,
        "date_quality": date_quality,
        "source_start_date_exact": start_day,
        "source_start_precision": row.start_precision,
        "source_start_lower_bound": row.start_lower_bound,
        "source_start_upper_bound": row.start_upper_bound,
        "source_end_date_exact_unvalidated": end_day if end_reviewed else "",
        "source_end_precision": row.end_precision,
        "source_end_lower_bound": row.end_lower_bound,
        "source_end_upper_bound": row.end_upper_bound,
        "source_time_semantics": row.time_semantics,
        "source_end_semantics": row.end_semantics,
        "order_interval_qc": row.order_interval_qc,
        "has_unresolved_status_semantics": row.source_kind == "BTRIS",
        "qc_has_major_conflict": conflict,
        "qc_flags": row.qc_flags,
        "classification_review_reasons": row.review_reasons,
        "documentation_dated_usable": documented_date_usable,
        "normalized_dated_usable": normalized_date_usable,
        "actual_medication_exposure_confirmed": False,
        "treatment_end_confirmed": False,
        "release_version": VERSION,
    }

def build_medication_release(records, audit_summary, *, eligibility=None, secret=None):
    """Pure transform; eligibility contains raw CTDB patient IDs if explicitly used."""
    _required_data(records, audit_summary)
    selected = records.loc[
        records.eligible_linked.eq(True)
        & records.name_present.eq(True)
        & records.linked_patient_pseudonymous.notna()
        & records.linked_patient_pseudonymous.ne("")
    ].copy()
    rows = [_row_to_document(row) for row in selected.itertuples(index=False)]
    catalog = pd.DataFrame(rows)
    if catalog.empty:
        raise ValueError("no_linked_named_medication_records")
    if eligibility is not None:
        if secret is None or len(secret) < 32:
            raise ValueError("pseudonym_key_required_to_release_patient_ids")
        if "patient_id" not in eligibility.columns:
            raise ValueError("eligibility_patient_id_missing")
        names = eligibility.patient_id.map(lambda x: str(x).strip() if _has(x) else "")
        if not names.ne("").all() or names.duplicated().any():
            raise ValueError("eligibility_ids_missing_or_duplicate")
        mapping = {
            hmac.new(secret, ("patient\0" + p).encode(), hashlib.sha256).hexdigest(): p
            for p in names
        }
        catalog["patient_id"] = catalog.linked_patient_pseudonymous.map(mapping)
        if catalog.patient_id.isna().any():
            raise ValueError("patient_join_key_not_in_eligibility")
    dated = catalog.loc[catalog.date_of_documentation.ne("")].copy()
    normalized = dated.loc[dated.normalized_dated_usable].copy()
    # Do not discard absent dates or nonmapped drugs from the protected catalog.
    reasons = []
    for source, part in catalog.groupby("source"):
        reasons.append({
            "source": source, "n_linked_named_rows": len(part),
            "n_dated_rows": int(part.date_of_documentation.ne("").sum()),
            "n_documentation_dated_usable": int(part.documentation_dated_usable.sum()),
            "n_normalized_dated_usable": int(part.normalized_dated_usable.sum()),
            "n_undated_rows": int(part.date_of_documentation.eq("").sum()),
            "n_major_conflict_rows": int(part.qc_has_major_conflict.sum()),
        })
    source_qc = pd.DataFrame(reasons)
    summary = {
        "version": VERSION,
        "source_classification_version": audit_summary["classification_version"],
        "release_scope": "dated_documentation_only_not_confirmed_medication_exposure",
        "n_documented_named": int(len(catalog)),
        "n_dated_observations": int(len(dated)),
        "n_normalized_dated": int(len(normalized)),
        "n_undated_retained": int(catalog.date_of_documentation.eq("").sum()),
        "n_patients_with_documented_medications": int(catalog.linked_patient_pseudonymous.nunique()),
        "n_patients_with_dated_observations": int(dated.linked_patient_pseudonymous.nunique()),
        "date_kind_counts": {str(k): int(v) for k, v in catalog.date_evidence_kind.value_counts().items()},
        "patient_id_join_column_released": eligibility is not None,
        "gates": {
            "documentation_descriptive_timeline": "GO_protected_descriptive_only",
            "medication_identity": "row_specific_with_quality_columns",
            "actual_administration_or_dispensing": "NO_GO",
            "treatment_episodes_or_active_on_date": "NO_GO",
            "treatment_effect_and_causal_inference": "NO_GO",
            "raw_exposure_start_or_discontinuation_dates": "NO_GO",
        },
    }
    return catalog, dated, normalized, source_qc, summary

def _private_write(path, frame):
    frame.to_parquet(path, index=False)
    os.chmod(path, 0o600)

def run(audit_root, *, output_dir=None, eligibility_path=None, release_patient_ids=False):
    audit_root = Path(audit_root).resolve()
    private = audit_root / "restricted"
    input_file = private / INPUT_BASENAME
    summary_file = audit_root / SUMMARY_BASENAME
    if not input_file.is_file() or not summary_file.is_file():
        raise FileNotFoundError("rerun Step 08 first; restricted audit and summary required")
    if release_patient_ids and eligibility_path is None:
        raise ValueError("explicit eligibility path required for patient ID release")
    if eligibility_path is not None and not release_patient_ids:
        raise ValueError("eligibility path requires explicit --release-patient-ids")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    records = pd.read_parquet(input_file)
    eligible, secret = None, None
    if release_patient_ids:
        allowed = Path(eligibility_path).resolve()
        if allowed.suffix.lower() == ".parquet":
            eligible = pd.read_parquet(allowed)
        elif allowed.suffix.lower() == ".csv":
            eligible = pd.read_csv(allowed, dtype=object, keep_default_na=False)
        else:
            raise ValueError("eligibility must be CSV or parquet")
        secret = (private / ".pseudonym_key").read_bytes()
    catalog, dated, normalized, source_qc, result = build_medication_release(
        records, summary, eligibility=eligible, secret=secret)
    dest = Path(output_dir).resolve() if output_dir else (private / "medication_documentation_release")
    if not dest.is_relative_to(private):
        raise ValueError("release must be written within protected Step 08 restricted directory")
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(dest, 0o700)
    for filename, frame in (
        ("08_medications_documented_catalog.parquet", catalog),
        ("08_medications_dated_observations.parquet", dated),
        ("08_medications_normalized_dated.parquet", normalized),
    ):
        _private_write(dest / filename, frame)
    source_qc.to_csv(dest / "08_medications_release_source_qc.csv", index=False)
    os.chmod(dest / "08_medications_release_source_qc.csv", 0o600)
    with (dest / "08_medications_release_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    os.chmod(dest / "08_medications_release_summary.json", 0o600)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eligibility", type=Path)
    parser.add_argument("--release-patient-ids", action="store_true",
                        help="Explicitly include the joinable CTDB patient_id in protected files")
    args = parser.parse_args(argv)
    result = run(args.audit_root, output_dir=args.output_dir,
                 eligibility_path=args.eligibility,
                 release_patient_ids=args.release_patient_ids)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

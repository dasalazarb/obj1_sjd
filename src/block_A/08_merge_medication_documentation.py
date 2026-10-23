"""Merge Step 08 BTRIS order documentation with original-visit CTDB reports.

Restricted, descriptive-only release. Replace the COLLAPSED CTDB records with
original-visit CTDB records; never append both CTDB representations. Use the
authoritative Step 00 episode only for original CTDB visit rows. BTRIS has no
episode assignment until separately authorised temporal matching.

This is not a medication exposure / dispensing / administration dataset.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pandas as pd

VERSION = "08-medication-documentation-consolidated-1.0.0"
AUDIT_SOURCE = "08_treatment_record_classification_audit.parquet"
RELEASE_SOURCE = "medication_documentation_release/08_medications_documented_catalog.parquet"
CTDB_SOURCE = "ctdb_original_visit_recovery/08_ctdb_original_visit_medications.parquet"
CTDB_SUMMARY = "ctdb_original_visit_recovery/08_ctdb_visit_recovery_summary.json"
RELEASE_SUMMARY = "medication_documentation_release/08_medications_release_summary.json"

def _required(frame, columns, label):
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(label + "_missing_columns:" + ",".join(sorted(missing)))

def _str(value):
    return "" if pd.isna(value) else str(value).strip()

def _day(value):
    text = _str(value)
    if len(text) != 10 or text[4:5] != "-" or text[7:8] != "-":
        return ""
    try:
        return pd.Timestamp(text).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return ""

def _sha(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()

def _btris(catalog, audit_records):
    part = catalog.loc[catalog.source.isin(["11D", "15D"])].copy()
    raw = audit_records.loc[audit_records.source.isin(["11D", "15D"]),
                            ["audit_record_id", "drug_raw", "ingredient_raw",
                             "linked_patient_pseudonymous", "source"]].copy()
    if raw.audit_record_id.isna().any() or raw.audit_record_id.duplicated().any():
        raise ValueError("raw_btris_audit_ids_not_unique")
    if part.audit_record_id.isna().any() or part.audit_record_id.duplicated().any():
        raise ValueError("btris_release_ids_not_unique")
    if len(part) != len(raw):
        raise ValueError("btris_released_audit_row_count_mismatch")
    part = part.merge(raw, on="audit_record_id", how="left",
                      validate="one_to_one", suffixes=("", "_raw"),
                      indicator=True)
    if not part._merge.eq("both").all():
        raise ValueError("btris_release_raw_records_mismatch")
    if not part.linked_patient_pseudonymous.eq(
            part.linked_patient_pseudonymous_raw).all():
        raise ValueError("btris_patient_identity_disagreement")
    if not part.source.eq(part.source_raw).all():
        raise ValueError("btris_source_disagreement")
    rows = []
    for r in part.itertuples(index=False):
        event = _day(r.date_of_documentation)
        lo = _day(r.documentation_window_lower_bound)
        hi = _day(r.documentation_window_upper_bound)
        if event and (lo != event or hi != event):
            raise ValueError("btris_event_window_inconsistency")
        rows.append(dict(
            documentation_record_id="BTRIS:" + _str(r.audit_record_id),
            source=_str(r.source), source_kind="BTRIS_order",
            linked_patient_pseudonymous=r.linked_patient_pseudonymous,
            clinical_episode_id="", episode_assignment="not_assigned_temporal_match_pending",
            medication_name_raw=_str(r.drug_raw), ingredient_source_raw=_str(r.ingredient_raw),
            drug_short_label=_str(r.drug_short_label),
            ingredient_canonical=_str(r.ingredient_canonical),
            clinical_category_primary=_str(r.clinical_category_primary),
            clinical_category_detail=_str(r.clinical_category_detail),
            classification_confidence=_str(r.classification_confidence),
            study_category_rule_status=_str(r.study_category_rule_status),
            route_canonical=_str(r.route_canonical),
            date_of_documentation=event,
            documentation_window_lower_bound=lo,
            documentation_window_upper_bound=hi,
            date_precision=_str(r.documentation_window_precision),
            date_evidence_kind="btris_order_or_start_date_unvalidated",
            source_reported_start_lower=_str(r.source_start_lower_bound),
            source_reported_start_upper=_str(r.source_start_upper_bound),
            source_reported_start_precision=_str(r.source_start_precision),
            episode_start_date="", episode_end_date="", clinical_anchor_date="",
            date_within_authoritative_episode=pd.NA,
            normalized_dated_usable=bool(r.normalized_dated_usable),
            documentation_dated_usable=bool(r.documentation_dated_usable),
            review_reasons=_str(r.classification_review_reasons),
            qc_flags=_str(r.qc_flags),
            actual_exposure_confirmed=False,
            source_audit_record_id=_str(r.audit_record_id),
            consolidation_version=VERSION,
        ))
    return pd.DataFrame(rows)

def _ctdb(catalog):
    _required(catalog, ["patient_pseudonym", "clinical_episode_id", "row_pseudonym",
                        "slot", "medication_name_raw", "visit_documentation_date",
                        "medication_date_within_episode", "classification_confidence",
                        "clinical_category_primary", "ingredient_canonical",
                        "clinical_category_detail", "study_category_rule_status",
                        "episode_start_date", "episode_end_date", "clinical_anchor_date",
                        "clinical_visit", "source_reported_start_lower",
                        "source_reported_start_upper", "source_reported_start_precision"],
              "ctdb_original")
    ids = catalog.row_pseudonym.astype(str) + ":rx:" + catalog.slot.astype(str)
    if ids.duplicated().any():
        raise ValueError("ctdb_duplicate_original_visit_slot")
    rows = []
    for r in catalog.itertuples(index=False):
        event = _day(r.visit_documentation_date)
        within = bool(r.medication_date_within_episode)
        if event and not within:
            # Keep the discordant original row in the protected catalog but
            # exclude it from every date-ready subset.
            pass
        rows.append(dict(
            documentation_record_id="CTDB_ORIGINAL:" + _str(r.row_pseudonym) +
                                    ":rx:" + str(r.slot),
            source="CTDB_original_visit", source_kind="CTDB_reported_current",
            linked_patient_pseudonymous=r.patient_pseudonym,
            clinical_episode_id=_str(r.clinical_episode_id),
            episode_assignment="authoritative_eda_row_map_step00",
            medication_name_raw=_str(r.medication_name_raw), ingredient_source_raw="",
            drug_short_label=_str(r.ingredient_canonical),
            ingredient_canonical=_str(r.ingredient_canonical),
            clinical_category_primary=_str(r.clinical_category_primary),
            clinical_category_detail=_str(r.clinical_category_detail),
            classification_confidence=_str(r.classification_confidence),
            study_category_rule_status=_str(r.study_category_rule_status),
            route_canonical=_str(r.route_canonical),
            date_of_documentation=event,
            documentation_window_lower_bound=event,
            documentation_window_upper_bound=event,
            date_precision="day" if event else "unknown",
            date_evidence_kind="ctdb_current_medication_reported_at_original_visit",
            source_reported_start_lower=_str(r.source_reported_start_lower),
            source_reported_start_upper=_str(r.source_reported_start_upper),
            source_reported_start_precision=_str(r.source_reported_start_precision),
            episode_start_date=_str(r.episode_start_date),
            episode_end_date=_str(r.episode_end_date),
            clinical_anchor_date=_str(r.clinical_anchor_date),
            date_within_authoritative_episode=within,
            normalized_dated_usable=bool(
                event and within and r.classification_confidence == "high_exact"
                and _str(r.clinical_category_primary) != "indeterminate"),
            documentation_dated_usable=bool(event and within),
            review_reasons=_str(getattr(r,"classification_review_reasons","")),
            qc_flags="date_outside_authoritative_episode" if event and not within else "",
            actual_exposure_confirmed=False,
            source_audit_record_id="",
            consolidation_version=VERSION,
        ))
    return pd.DataFrame(rows)

def build(btris_release, audit_records, ctdb_original, release_summary, ctdb_summary):
    if release_summary.get("version") != "08-medication-documentation-release-1.1.0":
        raise ValueError("unsupported_btris_release_version")
    if ctdb_summary.get("version") != "08-ctdb-visit-1.1":
        raise ValueError("unsupported_ctdb_recovery_version")
    if not str(release_summary.get("source_classification_version","")).startswith("08-classification-"):
        raise ValueError("classification_version_unverified")
    if release_summary["n_documented_named"] != len(btris_release):
        raise ValueError("btris_release_catalog_rows_mismatch")
    if ctdb_summary["n_original_visit_medication_mentions"] != len(ctdb_original):
        raise ValueError("ctdb_recovery_catalog_rows_mismatch")
    _required(btris_release, ["audit_record_id", "source",
              "linked_patient_pseudonymous", "date_of_documentation",
              "documentation_window_lower_bound", "documentation_window_upper_bound",
              "source_start_lower_bound", "source_start_upper_bound",
              "source_start_precision", "documentation_window_precision",
              "normalized_dated_usable", "documentation_dated_usable",
              "drug_short_label", "ingredient_canonical", "clinical_category_primary",
              "clinical_category_detail", "classification_confidence",
              "study_category_rule_status", "route_canonical",
              "classification_review_reasons", "qc_flags"], "btris_release")
    _required(audit_records, ["audit_record_id","drug_raw","ingredient_raw",
              "linked_patient_pseudonymous","source"],"audit_records")
    _required(ctdb_original, ["patient_pseudonym","clinical_episode_id",
              "visit_documentation_date","medication_date_within_episode"],
              "ctdb_original")
    btris = _btris(btris_release, audit_records)
    ctdb = _ctdb(ctdb_original)
    if len(btris) != sum(btris_release.source.eq(s).sum() for s in ("11D","15D")):
        raise ValueError("btris_rows_reconciliation")
    if len(ctdb) != len(ctdb_original):
        raise ValueError("ctdb_rows_reconciliation")
    combined = pd.concat([btris,ctdb],ignore_index=True)
    if combined.documentation_record_id.duplicated().any():
        raise ValueError("combined_documentation_record_id_duplicate")
    if combined.linked_patient_pseudonymous.isna().any() or (
        combined.linked_patient_pseudonymous.astype(str).str.strip().eq("")
    ).any():
        raise ValueError("combined_unlinked_patient")
    if combined.actual_exposure_confirmed.any():
        raise ValueError("exposure_gate_violation")
    dated = combined.loc[combined.date_of_documentation.ne("")
                        & combined.documentation_dated_usable].copy()
    normalized = dated.loc[dated.normalized_dated_usable].copy()
    by_source = combined.groupby("source",dropna=False).agg(
        n_records=("documentation_record_id","size"),
        n_dated=("date_of_documentation",lambda x: x.ne("").sum()),
        n_dated_usable=("documentation_dated_usable","sum"),
        n_normalized_dated=("normalized_dated_usable","sum"),
        n_patients=("linked_patient_pseudonymous","nunique")).reset_index()
    summary = dict(
        version=VERSION,
        unit="documented_order_row_or_original_ctdb_visit_slot",
        n_btris_order_rows=len(btris), n_ctdb_original_visit_mentions=len(ctdb),
        n_total_documentation_records=len(combined),
        n_dated_observations=int(combined.date_of_documentation.ne("").sum()),
        n_dated_usable=len(dated), n_normalized_dated=len(normalized),
        n_unique_patient_tokens=int(combined.linked_patient_pseudonymous.nunique()),
        n_ctdb_authoritatively_linked_episodes=int(ctdb.clinical_episode_id.nunique()),
        n_ctdb_dates_outside_episode=int((ctdb.date_within_authoritative_episode == False).sum()),
        n_btris_records_with_episode_assignment=int(btris.clinical_episode_id.ne("").sum()),
        legacy_ctdb_collapsed_records_excluded=int(
            btris_release.source.eq("CTDB").sum()),
        source_classification_version=release_summary["source_classification_version"],
        notes=["No cross-source deduplication or medication era inference.",
               "CTDB original visit mentions replace, not append to, the legacy collapsed CTDB audit.",
               "BTRIS episode matching is pending separately authorised temporal alignment.",
               "CTDB current reported use != confirmed administration.",
               "BTRIS order date != confirmed medication start."],
        downstream_integration="NO_GO_exposure_or_treatment_effect",
        documentation_descriptive_timeline="GO_protected_only",
    )
    if len(combined) != len(btris) + len(ctdb):
        raise ValueError("combined_reconciliation_failed")
    return combined,dated,normalized,by_source,summary

def run(audit_root):
    base=Path(audit_root).resolve()
    private=base/"restricted"
    source_files={
        "btris_release":private/RELEASE_SOURCE,
        "btris_audit":private/AUDIT_SOURCE,
        "ctdb_original":private/CTDB_SOURCE,
        "release_summary":private/RELEASE_SUMMARY,
        "ctdb_summary":private/CTDB_SUMMARY,
    }
    if not all(path.is_file() for path in source_files.values()):
        raise FileNotFoundError("required Step08 restricted inputs missing; rerun both producers")
    release_sum=json.loads(source_files["release_summary"].read_text())
    ctdb_sum=json.loads(source_files["ctdb_summary"].read_text())
    result=build(pd.read_parquet(source_files["btris_release"]),
                 pd.read_parquet(source_files["btris_audit"]),
                 pd.read_parquet(source_files["ctdb_original"]),
                 release_sum,ctdb_sum)
    allrows,dated,normalized,qc,summary=result
    summary["input_sha256"]={k:_sha(v) for k,v in source_files.items()}
    output=private/"medication_documentation_consolidated"
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(output,0o700)
    for name,frame in [
        ("08_medications_consolidated_documentation.parquet",allrows),
        ("08_medications_consolidated_dated.parquet",dated),
        ("08_medications_consolidated_normalized_dated.parquet",normalized)]:
        destination=output/name
        frame.to_parquet(destination,index=False)
        os.chmod(destination,0o600)
    qc_file=output/"08_medications_consolidated_source_qc.csv"
    qc.to_csv(qc_file,index=False)
    os.chmod(qc_file,0o600)
    summary_file=output/"08_medications_consolidated_summary.json"
    summary_file.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    os.chmod(summary_file,0o600)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-root",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.audit_root),indent=2,sort_keys=True))
if __name__=="__main__":
    main()

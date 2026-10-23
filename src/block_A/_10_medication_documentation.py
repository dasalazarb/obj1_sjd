"""Step 10 adapter for aggregated medication DOCUMENTATION context.

Source: restricted Step 08 one-row-per-medication catalog and primary
many-to-many medication/clinical-episode links. Aggregate by canonical Step 00
(patient_id, clinical_episode_id). Preserve every episode exactly once.
No row-level drug, pseudonym, order, MRN or raw identifier is exported to Step 10.
No therapy/exposure/administration, no temporal tolerance in the main context.

This module intentionally does not add treatment predictors to analytic master.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pandas as pd

VERSION = "10-medication-documentation-context-1.0.0"
UPSTREAM_VERSION = "08-medication-episode-links-1.0.0"
CATALOG = "medication_documentation_consolidated/08_medications_consolidated_documentation.parquet"
LINKS = "medication_episode_documentation_links/08_medication_episode_primary_links.parquet"
RECORD_STATUS = "medication_episode_documentation_links/08_medication_documentation_record_status.parquet"
SUMMARY = "medication_episode_documentation_links/08_medication_episode_link_summary.json"
CATEGORIES = {
    "systemic_sjd": "meddoc__n_systemic_sjd_candidate_documents",
    "symptomatic_sjd": "meddoc__n_symptomatic_sjd_candidate_documents",
    "concomitant": "meddoc__n_concomitant_candidate_documents",
    "procedure_related": "meddoc__n_procedure_related_candidate_documents",
    "indeterminate": "meddoc__n_indeterminate_documents",
}
COUNT_COLUMNS = [
    "meddoc__n_documentation_records",
    "meddoc__n_btris_11d_order_documents",
    "meddoc__n_btris_15d_order_documents",
    "meddoc__n_ctdb_original_visit_mentions",
    "meddoc__n_high_exact_identity_documents",
    *CATEGORIES.values(),
]
FLAG = "meddoc__has_linked_documentation"

def _require(df, columns, label):
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(label + "_missing_columns:" + ",".join(missing))

def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def _day(value):
    if pd.isna(value):
        return pd.NaT
    try:
        parsed = pd.Timestamp(value)
        return parsed.normalize()
    except (ValueError, TypeError, OverflowError):
        return pd.NaT

def _token(key, identifier):
    if pd.isna(identifier) or not str(identifier).strip():
        raise ValueError("Step10_patient_id_missing")
    return hmac.new(key, ("patient\0"+str(identifier).strip()).encode(),
                    hashlib.sha256).hexdigest()

def aggregate(spine, documentation, primary, status, summary, secret):
    """Return a fully keyed episode-level CONTEXT frame and aggregate QC."""
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("Step08_pseudonym_key_invalid")
    if summary.get("version") != UPSTREAM_VERSION:
        raise ValueError("medication_link_version_mismatch")
    _require(spine, [
        "patient_id","clinical_episode_id","clinical_anchor_date",
        "episode_start_date","episode_end_date","clinical_visit",
    ], "Step10_clinical_spine")
    _require(documentation, [
        "documentation_record_id","linked_patient_pseudonymous","source",
        "source_kind","date_of_documentation","clinical_category_primary",
        "classification_confidence","actual_exposure_confirmed",
        "documentation_dated_usable",
    ], "Step08_documentation")
    _require(primary, [
        "documentation_record_id","linked_patient_pseudonymous","source",
        "source_kind","clinical_episode_id","relation_type",
        "primary_documentation_relation","tolerance_sensitivity_only",
        "date_of_documentation","episode_start_date","episode_end_date",
        "patient_and_episode_verified_against_Step00",
        "confirmed_medication_exposure",
    ], "Step08_primary_links")
    _require(status,[
        "documentation_record_id","n_primary_episode_relations",
        "n_tolerance_sensitivity_relations","original_match_status",
    ], "Step08_documentation_status")
    if spine[["patient_id","clinical_episode_id"]].isna().any().any() or (
        spine.duplicated(["patient_id","clinical_episode_id"]).any()):
        raise ValueError("Step10_duplicate_or_missing_episode_keys")
    if any(x.duplicated("documentation_record_id").any() or
           x.documentation_record_id.isna().any()
           for x in [documentation,status]):
        raise ValueError("Step08_documentation_or_status_grain_invalid")
    if primary.duplicated(["documentation_record_id","clinical_episode_id"]).any():
        raise ValueError("Step08_primary_duplicate_links")
    if not primary.primary_documentation_relation.eq(True).all() or (
        not primary.tolerance_sensitivity_only.eq(False).all()):
        raise ValueError("tolerance_links_forbidden_in_Step10_main")
    if not primary.patient_and_episode_verified_against_Step00.eq(True).all():
        raise ValueError("unverified_episode_relation")
    if (not documentation.actual_exposure_confirmed.eq(False).all()
        or not primary.confirmed_medication_exposure.eq(False).all()):
        raise ValueError("medication_exposure_flag_not_supported")
    if len(documentation) != summary["n_original_documentation_records"]:
        raise ValueError("Step08_original_documentation_count_changed")
    if len(primary) != summary["n_primary_episode_relations"]:
        raise ValueError("Step08_primary_link_count_changed")
    if len(status) != len(documentation) or set(status.documentation_record_id) != set(documentation.documentation_record_id):
        raise ValueError("Step08_status_not_one_per_documentation_record")
    counts = primary.groupby("documentation_record_id").size()
    expected = status.set_index("documentation_record_id")["n_primary_episode_relations"]
    counts = counts.reindex(expected.index, fill_value=0).astype(int)
    if not counts.eq(expected.astype(int)).all():
        raise ValueError("Step08_primary_links_disagree_with_record_status")

    clinical = spine[["patient_id","clinical_episode_id","clinical_anchor_date",
                      "episode_start_date","episode_end_date"]].copy()
    clinical["patient_id"] = clinical.patient_id.astype("string")
    clinical["clinical_episode_id"] = clinical.clinical_episode_id.astype("string")
    clinical["linked_patient_pseudonymous"] = clinical.patient_id.map(
        lambda value: _token(secret, value))
    if clinical[["linked_patient_pseudonymous","clinical_episode_id"]].duplicated().any():
        raise ValueError("Step10_token_episode_collision")
    for col in ["episode_start_date","episode_end_date","clinical_anchor_date"]:
        clinical[col] = clinical[col].map(_day)
    if clinical[["episode_start_date","episode_end_date","clinical_anchor_date"]].isna().any().any():
        raise ValueError("Step10_episode_date_missing")
    if (clinical.episode_start_date.gt(clinical.clinical_anchor_date).any()
        or clinical.clinical_anchor_date.gt(clinical.episode_end_date).any()):
        raise ValueError("Step10_episode_date_inconsistent")

    valid_sources = {"11D","15D","CTDB_original_visit"}
    if not documentation.source.isin(valid_sources).all():
        raise ValueError("unexpected_medication_source")
    if not documentation.source_kind.isin(["BTRIS_order","CTDB_reported_current"]).all():
        raise ValueError("unexpected_medication_source_kind")
    if (documentation.source.isin(["11D","15D"]) !=
        documentation.source_kind.eq("BTRIS_order")).any():
        raise ValueError("source_kind_conflicts_with_source")
    if not documentation.clinical_category_primary.isin(CATEGORIES).all():
        raise ValueError("unrecognized_medication_candidate_category")

    details=primary.merge(documentation[[
        "documentation_record_id","linked_patient_pseudonymous","source",
        "source_kind","date_of_documentation",
        "clinical_category_primary","classification_confidence",
        "documentation_dated_usable",
    ]],on="documentation_record_id",how="left",validate="many_to_one",
        suffixes=("","_catalog"),indicator=True)
    if not details["_merge"].eq("both").all():
        raise ValueError("primary_relation_without_source_document")
    for field in ["linked_patient_pseudonymous","source","source_kind",
                  "date_of_documentation"]:
        if not details[field].astype("string").eq(
                details[field+"_catalog"].astype("string")).all():
            raise ValueError("Step08_primary_link_catalog_" + field + "_mismatch")
    if details.linked_patient_pseudonymous.isna().any():
        raise ValueError("missing_protected_pseudonym")
    if not details.documentation_dated_usable.eq(True).all():
        raise ValueError("QC_blocked_BTRIS_cannot_enter_episode_context")

    joined=details.merge(clinical,on=["linked_patient_pseudonymous",
                                       "clinical_episode_id"],how="left",
        validate="many_to_one",suffixes=("","_spine"),indicator="episode_link")
    if not joined.episode_link.eq("both").all():
        raise ValueError("Step08_episode_missing_from_Step10_spine")
    event=joined.date_of_documentation.map(_day)
    if event.isna().any() or (
        (event<joined.episode_start_date_spine) |
        (event>joined.episode_end_date_spine)).any():
        raise ValueError("documentation_date_outside_episode")
    if (joined.episode_start_date.map(_day)!=
        joined.episode_start_date_spine).any() or (
        joined.episode_end_date.map(_day)!=
        joined.episode_end_date_spine).any():
        raise ValueError("Step08_link_boundaries_changed_since_Step00")

    grouped = joined.groupby(["patient_id","clinical_episode_id"],dropna=False)
    results=[]
    for (pid,eid),part in grouped:
        if part.documentation_record_id.duplicated().any():
            raise ValueError("episode_has_duplicate_documentation_relation")
        values=dict(patient_id=pid,clinical_episode_id=eid,
            meddoc__n_documentation_records=int(part.documentation_record_id.nunique()),
            meddoc__n_btris_11d_order_documents=int(part.source.eq("11D").sum()),
            meddoc__n_btris_15d_order_documents=int(part.source.eq("15D").sum()),
            meddoc__n_ctdb_original_visit_mentions=int(
                part.source.eq("CTDB_original_visit").sum()),
            meddoc__n_high_exact_identity_documents=int(
                part.classification_confidence.eq("high_exact").sum()))
        for category,col in CATEGORIES.items():
            values[col]=int(part.clinical_category_primary.eq(category).sum())
        results.append(values)
    sparse=pd.DataFrame(results,columns=[
        "patient_id","clinical_episode_id",*COUNT_COLUMNS])
    # Zero denotes no *linked documentation observed in included sources*,
    # not that the patient was medication-free.
    output=clinical[["patient_id","clinical_episode_id"]].merge(
        sparse,on=["patient_id","clinical_episode_id"],how="left",
        validate="one_to_one",sort=False)
    for col in COUNT_COLUMNS:
        output[col]=output[col].fillna(0).astype("Int64")
    output[FLAG]=output.meddoc__n_documentation_records.gt(0).astype(bool)
    if len(output)!=len(spine) or output.duplicated(["patient_id","clinical_episode_id"]).any():
        raise ValueError("Step10_episode_grain_not_preserved")
    if int(output.meddoc__n_documentation_records.sum())!=len(primary):
        raise ValueError("Step10_episode_documentation_sum_mismatch")
    if output[list(CATEGORIES.values())].sum().sum()!=len(primary):
        raise ValueError("medication_candidate_categories_not_exhaustive")
    if int(output.meddoc__n_btris_11d_order_documents.sum()+
           output.meddoc__n_btris_15d_order_documents.sum()+
           output.meddoc__n_ctdb_original_visit_mentions.sum())!=len(primary):
        raise ValueError("medication_sources_not_exhaustive")
    qc=dict(
        version=VERSION,source_version=summary["version"],
        n_clinical_episodes=len(output),
        n_clinical_patients=int(output.patient_id.nunique()),
        n_episodes_with_primary_documentation=int(output[FLAG].sum()),
        n_original_documentation_records=int(len(documentation)),
        n_unique_documentation_records_with_primary_link=int(
            primary.documentation_record_id.nunique()),
        n_primary_links=int(len(primary)),
        n_btris_links=int(joined.source_kind.eq("BTRIS_order").sum()),
        n_ctdb_links=int(joined.source_kind.eq("CTDB_reported_current").sum()),
        n_btris_orders_with_multiple_episode_links=int(
            counts.loc[documentation.loc[
                documentation.source_kind.eq("BTRIS_order"),
                "documentation_record_id"]].gt(1).sum()),
        n_tolerance_relations_excluded_from_main=int(
            status.n_tolerance_sensitivity_relations.sum()),
        n_documentation_records_without_primary_episode_link=int(counts.eq(0).sum()),
        original_episodes_preserved=True,
        documentation_records_not_duplicated=True,
        only_episode_aggregated_counts_leave_restricted_08=True,
        exposure_status="NO_GO",
        no_medication_continuity_inferred=True,
        zero_count_meaning="no_linked_documentation_observed_not_no_medication",
        unverified_patient_identifier_QC_not_overridden=True,
        drug_candidate_categories_not_patient_indications=True,
    )
    return output,qc

def load_and_aggregate(spine, audit_root, clinical_spine_path):
    """Read guarded Step 08 files, check upstream hashes, return context and QC."""
    root=Path(audit_root).resolve()/"restricted"
    paths={
        "catalog":root/CATALOG,
        "primary":root/LINKS,
        "record_status":root/RECORD_STATUS,
        "summary":root/SUMMARY,
        "secret":root/".pseudonym_key",
    }
    for k,path in paths.items():
        if not path.is_file():
            raise FileNotFoundError("medication_documentation_input_missing:"+k)
    summary=json.loads(paths["summary"].read_text())
    fingerprints=summary.get("input_sha256",{})
    if fingerprints.get("medications")!=_sha(paths["catalog"]):
        raise ValueError("Step08_medication_catalog_hash_changed")
    if fingerprints.get("clinical_Step00_episodes")!=_sha(clinical_spine_path):
        raise ValueError("Step00_clinical_spine_changed_since_08_linking")
    result,qc=aggregate(
        spine,
        pd.read_parquet(paths["catalog"]),
        pd.read_parquet(paths["primary"]),
        pd.read_parquet(paths["record_status"]),
        summary,paths["secret"].read_bytes(),
    )
    qc["input_sha256"]={key:_sha(path) for key,path in paths.items() if key!="secret"}
    return result,qc

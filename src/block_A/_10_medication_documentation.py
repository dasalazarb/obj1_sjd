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
import os
from pathlib import Path

import pandas as pd

VERSION = "10-medication-documentation-context-1.1.0"
UPSTREAM_VERSION = "08-medication-episode-links-1.1.0"
CATALOG = "medication_documentation_consolidated/08_medications_consolidated_documentation.parquet"
LINKS = "medication_episode_documentation_links/08_medication_episode_primary_links.parquet"
RECORD_STATUS = "medication_episode_documentation_links/08_medication_documentation_record_status.parquet"
SUMMARY = "medication_episode_documentation_links/08_medication_episode_link_summary.json"
CONSOLIDATED_SUMMARY = "medication_documentation_consolidated/08_medications_consolidated_summary.json"
BTRIS_AUDIT = "08_treatment_record_classification_audit.parquet"
BTRIS_RELEASE = "medication_documentation_release/08_medications_documented_catalog.parquet"
DETAIL_DIR = "step10_medication_episode_details"
DETAIL_FILE = "10_medication_documentation_episode_detail.parquet"
NAME_FIELD = "meddoc__medication_names"
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
PROXIMITY_FILE = "medication_episode_documentation_links/08_medication_episode_year_proximity.parquet"
PROXIMITY_COLUMNS = [
    "medprox__n_order_episode_relations_within_1y",
    "medprox__n_prior_order_relations_within_1y",
    "medprox__n_subsequent_order_relations_within_1y",
    "medprox__n_unique_order_documents_within_1y",
    "medprox__medication_names_within_1y",
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

def _canonical_name(value):
    """Publish only mapped ingredient names, not source free text, in master."""
    if pd.isna(value):
        return ""
    name = " ".join(str(value).strip().split())
    if name.lower() in {"", "nan", "none", "null", "unknown", "unmapped", "indeterminate"}:
        return ""
    # Avoid injecting delimiters or line breaks into an episode-level CSV cell.
    return name.replace(";", ",").replace("|", "/")[:160]


def aggregate(spine, documentation, primary, status, summary, secret,
              *, include_details=False):
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
        "documentation_dated_usable", "ingredient_canonical",
        "medication_name_raw", "drug_short_label", "route_canonical",
        "clinical_category_detail", "review_reasons", "qc_flags",
        "date_evidence_kind", "source_audit_record_id",
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
        "ingredient_canonical","medication_name_raw","drug_short_label",
        "route_canonical","clinical_category_detail","review_reasons",
        "qc_flags","date_evidence_kind","source_audit_record_id",
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
        values[NAME_FIELD] = "; ".join(sorted({
            name for name in part.ingredient_canonical.map(_canonical_name)
            if name
        }, key=str.casefold))
        results.append(values)
    sparse=pd.DataFrame(results,columns=[
        "patient_id","clinical_episode_id",*COUNT_COLUMNS,NAME_FIELD])
    # Zero denotes no *linked documentation observed in included sources*,
    # not that the patient was medication-free.
    output=clinical[["patient_id","clinical_episode_id"]].merge(
        sparse,on=["patient_id","clinical_episode_id"],how="left",
        validate="one_to_one",sort=False)
    for col in COUNT_COLUMNS:
        output[col]=output[col].fillna(0).astype("Int64")
    output[FLAG]=output.meddoc__n_documentation_records.gt(0).astype(bool)
    output[NAME_FIELD]=output[NAME_FIELD].replace("",pd.NA).astype("string")
    if len(output)!=len(spine) or output.duplicated(["patient_id","clinical_episode_id"]).any():
        raise ValueError("Step10_episode_grain_not_preserved")
    # Explicit sensitivity fields: order date within one calendar year of the
    # clinical episode, but outside the episode. Never equate with exposure.
    proximity_path = getattr(aggregate, "_proximity_path", None)
    if proximity_path is None or not Path(proximity_path).is_file():
        raise FileNotFoundError("one_year_proximity_Step08_artifact_missing")
    proximity = pd.read_parquet(proximity_path)
    _require(proximity, [
        "documentation_record_id", "linked_patient_pseudonymous",
        "clinical_episode_id", "date_of_documentation",
        "episode_start_date", "episode_end_date", "tolerance_sensitivity_only",
        "primary_documentation_relation", "confirmed_medication_exposure",
        "relation_type"], "one_year_proximity")
    if proximity.duplicated(["documentation_record_id", "clinical_episode_id"]).any():
        raise ValueError("duplicate_one_year_order_episode_links")
    if not proximity.empty and (
        not proximity.tolerance_sensitivity_only.eq(True).all()
        or not proximity.primary_documentation_relation.eq(False).all()
        or not proximity.confirmed_medication_exposure.eq(False).all()
        or not proximity.relation_type.eq("year_proximity_only").all()
    ):
        raise ValueError("one_year_proximity_contains_primary_or_exposure_link")
    lookup = documentation[["documentation_record_id",
                             "linked_patient_pseudonymous", "source_kind",
                             "ingredient_canonical", "documentation_dated_usable"]]
    near = proximity.merge(lookup, on="documentation_record_id",
                           how="left", validate="many_to_one",
                           indicator="_catalog_join", suffixes=("", "_source"))
    if not near["_catalog_join"].eq("both").all():
        raise ValueError("one_year_proximity_document_not_in_catalog")
    if not near.linked_patient_pseudonymous.eq(
        near.linked_patient_pseudonymous_source).all():
        raise ValueError("one_year_proximity_patient_mismatch")
    if not near.source_kind.eq("BTRIS_order").all() or not near.documentation_dated_usable.eq(True).all():
        raise ValueError("one_year_proximity_invalid_source_or_date")
    near = near.merge(
        clinical[["linked_patient_pseudonymous", "clinical_episode_id", "patient_id",
                  "episode_start_date", "episode_end_date"]],
        on=["linked_patient_pseudonymous", "clinical_episode_id"],
        how="left", validate="many_to_one", suffixes=("", "_spine"),
        indicator="_episode_join")
    if not near["_episode_join"].eq("both").all():
        raise ValueError("one_year_proximity_episode_not_in_spine")
    dt = near.date_of_documentation.map(_day)
    st = near.episode_start_date_spine
    en = near.episode_end_date_spine
    earlier = dt.lt(st)
    later = dt.gt(en)
    if not near.empty and (
        dt.isna().any() or
        (~(earlier | later)).any() or
        ((dt < st - pd.DateOffset(years=1)) |
         (dt > en + pd.DateOffset(years=1))).any()
    ):
        raise ValueError("one_year_proximity_temporal_validation_failed")
    near["prior"] = earlier
    near["subsequent"] = later
    proxrows = []
    for (pid, eid), group in near.groupby(["patient_id", "clinical_episode_id"]):
        proxrows.append({
            "patient_id": pid, "clinical_episode_id": eid,
            PROXIMITY_COLUMNS[0]: len(group),
            PROXIMITY_COLUMNS[1]: int(group.prior.sum()),
            PROXIMITY_COLUMNS[2]: int(group.subsequent.sum()),
            PROXIMITY_COLUMNS[3]: int(group.documentation_record_id.nunique()),
            PROXIMITY_COLUMNS[4]: "; ".join(sorted({
                n for n in group.ingredient_canonical.map(_canonical_name) if n
            }, key=str.casefold)),
        })
    near_frame = pd.DataFrame(proxrows, columns=["patient_id", "clinical_episode_id", *PROXIMITY_COLUMNS])
    output = output.merge(near_frame, on=["patient_id", "clinical_episode_id"],
                          how="left", validate="one_to_one")
    for column in PROXIMITY_COLUMNS[:4]:
        output[column] = output[column].fillna(0).astype("Int64")
    output[PROXIMITY_COLUMNS[4]] = output[PROXIMITY_COLUMNS[4]].replace("", pd.NA).astype("string")
    if int(output[PROXIMITY_COLUMNS[0]].sum()) != len(near) or (
        output[PROXIMITY_COLUMNS[1]].sum() +
        output[PROXIMITY_COLUMNS[2]].sum() != len(near)
    ):
        raise ValueError("one_year_proximity_aggregate_disagrees_with_edges")
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
        n_one_year_exploratory_links=int(len(near)),
        n_episodes_with_one_year_exploratory_links=int(output[PROXIMITY_COLUMNS[0]].gt(0).sum()),
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
        medication_name_semantics="unique_mapped_ingredients_from_linked_documents_not_verified_exposure",
        n_episode_relations_without_mapped_ingredient=int(
            joined.ingredient_canonical.map(_canonical_name).eq("").sum()),
        n_episodes_with_any_mapped_medication_name=int(output[NAME_FIELD].notna().sum()),
    )
    if not include_details:
        return output,qc
    protected_detail=joined[[
        "patient_id","clinical_episode_id","documentation_record_id",
        "source","source_kind","relation_type","date_of_documentation",
        "date_evidence_kind","episode_start_date","episode_end_date",
        "clinical_anchor_date","medication_name_raw","drug_short_label",
        "ingredient_canonical","route_canonical","clinical_category_primary",
        "clinical_category_detail","classification_confidence","review_reasons",
        "qc_flags","source_audit_record_id",
        "confirmed_medication_exposure",
    ]].copy()
    protected_detail["documentation_date_relative_to_anchor"] = "on_anchor"
    observed=pd.to_datetime(protected_detail.date_of_documentation,errors="raise")
    anchor=pd.to_datetime(protected_detail.clinical_anchor_date,errors="raise")
    protected_detail.loc[observed.lt(anchor),"documentation_date_relative_to_anchor"]="before_anchor"
    protected_detail.loc[observed.gt(anchor),"documentation_date_relative_to_anchor"]="after_anchor"
    if len(protected_detail)!=len(primary):
        raise ValueError("Step10_detail_primary_relation_count_changed")
    return output,qc,protected_detail

def _attach_btris_prescription_fields(detail, audit, release):
    """Join source-record prescription fields to trusted Step 08 links.

    Only BTRIS gets dose/frequency. CTDB slot doses are not reliably aligned.
    """
    _require(audit, ["audit_record_id","source","dose_raw","uom_raw",
                     "frequency_raw"], "Step08_BTRIS_audit")
    _require(release, ["audit_record_id","source","strength_canonical",
                       "formulation_canonical"], "Step08_BTRIS_release")
    btris_mask=detail.source_kind.eq("BTRIS_order")
    btris=detail.loc[btris_mask]
    if (audit.audit_record_id.isna().any() or
        audit.audit_record_id.duplicated().any() or
        release.audit_record_id.isna().any() or
        release.audit_record_id.duplicated().any()):
        raise ValueError("Step08_BTRIS_audit_or_release_duplicate_ids")
    if (btris.source_audit_record_id.isna().any() or
        not btris.documentation_record_id.eq("BTRIS:"+btris.source_audit_record_id).all()):
        raise ValueError("Step08_BTRIS_source_audit_identifier_mismatch")
    a=audit.loc[audit.source.isin(["11D","15D"]),[
        "audit_record_id","source","dose_raw","uom_raw","frequency_raw"
    ]].rename(columns={"source":"audit_source"})
    b=btris.merge(a,left_on="source_audit_record_id",
                  right_on="audit_record_id",how="left",
                  validate="many_to_one",indicator="_audit_match")
    if not b._audit_match.eq("both").all() or not b.source.eq(b.audit_source).all():
        raise ValueError("BTRIS_audit_document_source_or_id_mismatch")
    rel=release.loc[release.source.isin(["11D","15D"]),[
        "audit_record_id","source","strength_canonical","formulation_canonical"
    ]].rename(columns={"source":"release_source"})
    b=b.merge(rel,left_on="source_audit_record_id",right_on="audit_record_id",
              how="left",validate="many_to_one",indicator="_release_match",
              suffixes=("","_release"))
    if not b._release_match.eq("both").all() or not b.source.eq(b.release_source).all():
        raise ValueError("BTRIS_release_document_source_or_id_mismatch")
    # No dose values are inferred for CTDB from unassigned dose slots.
    cols=["dose_raw","uom_raw","frequency_raw",
          "strength_canonical","formulation_canonical"]
    for col in cols:
        detail[col]=pd.Series(pd.NA,index=detail.index,dtype="string")
        if len(b):
            detail.loc[btris_mask,col]=b[col].replace("",pd.NA).astype("string").to_numpy()
    detail["dose_semantics"]="not_available_for_ctdb_original_visit"
    detail.loc[btris_mask,"dose_semantics"]="btris_order_documentation_unverified"
    if not detail.confirmed_medication_exposure.eq(False).all():
        raise ValueError("confirmed_exposure_must_remain_false")
    if detail.duplicated(["documentation_record_id","clinical_episode_id"]).any():
        raise ValueError("detailed_medication_links_have_duplicate_record_episode_pairs")
    return detail


def write_protected_detail(detail, audit_root):
    root=Path(audit_root).resolve()/"restricted"
    if not root.is_dir():
        raise FileNotFoundError("Step08_protected_root_missing")
    destination=root/DETAIL_DIR
    destination.mkdir(mode=0o700,exist_ok=True)
    os.chmod(destination,0o700)
    target=destination/DETAIL_FILE
    import tempfile
    temp=None
    try:
        with tempfile.NamedTemporaryFile(prefix=".meddoc_",suffix=".parquet",
                                         dir=destination,delete=False) as fp:
            temp=Path(fp.name)
        detail.to_parquet(temp,index=False)
        os.chmod(temp,0o600)
        os.replace(temp,target)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()
    return target


def load_and_aggregate(spine, audit_root, clinical_spine_path, *, include_details=False):
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
    prox_path=root/PROXIMITY_FILE
    if not prox_path.is_file():
        raise FileNotFoundError("one_year_proximity_edges_required")
    aggregate._proximity_path = prox_path
    result=aggregate(
        spine,
        pd.read_parquet(paths["catalog"]),
        pd.read_parquet(paths["primary"]),
        pd.read_parquet(paths["record_status"]),
        summary,paths["secret"].read_bytes(),
        include_details=include_details,
    )
    episode_counts,qc=result[:2]
    qc["input_sha256"]={key:_sha(path) for key,path in paths.items() if key!="secret"}
    if not include_details:
        return episode_counts,qc
    details=result[2]
    consolidation_path=root/CONSOLIDATED_SUMMARY
    audit_path=root/BTRIS_AUDIT
    release_path=root/BTRIS_RELEASE
    for path in [consolidation_path,audit_path,release_path]:
        if not path.is_file():
            raise FileNotFoundError("Step08_missing_prescription_detail_input:"+path.name)
    consolidated=json.loads(consolidation_path.read_text())
    if consolidated.get("version")!="08-medication-documentation-consolidated-1.0.0":
        raise ValueError("Step08_consolidation_version_changed")
    original_hashes=consolidated.get("input_sha256",{})
    if (original_hashes.get("btris_audit")!=_sha(audit_path)
            or original_hashes.get("btris_release")!=_sha(release_path)):
        raise ValueError("Step08_BTRIS_audit_or_release_changed_since_consolidation")
    details=_attach_btris_prescription_fields(
        details,pd.read_parquet(audit_path),pd.read_parquet(release_path))
    qc["detail_rows"]=len(details)
    qc["n_btris_links_with_documented_dose"]=int(
        (details.source_kind.eq("BTRIS_order")&details.dose_raw.notna()).sum())
    qc["n_btris_links_with_documented_frequency"]=int(
        (details.source_kind.eq("BTRIS_order")&details.frequency_raw.notna()).sum())
    qc["detail_dose_rule"]="BTRIS raw order fields only; CTDB original-visit dose not reliably slot-mapped"
    qc["input_sha256"]["btris_audit"]=_sha(audit_path)
    qc["input_sha256"]["btris_release"]=_sha(release_path)
    return episode_counts,qc,details

"""Synthetic, identity-safe tests for Step 10 medication context adapter."""
import importlib.util
from pathlib import Path
import pandas as pd
import pytest

path=Path(__file__).resolve().parents[1]/"src/block_A/_10_medication_documentation.py"
spec=importlib.util.spec_from_file_location("step10_meds",path)
med=importlib.util.module_from_spec(spec)
spec.loader.exec_module(med)
KEY=b"x"*32

def sample():
    token=med._token(KEY,"p1")
    spine=pd.DataFrame([
        dict(patient_id="p1",clinical_episode_id=e,clinical_visit=True,
             episode_start_date="2021-01-01",episode_end_date="2021-03-31",
             clinical_anchor_date="2021-02-01")
        for e in ("e1","e2")]+[
        dict(patient_id="p2",clinical_episode_id="e3",clinical_visit=True,
             episode_start_date="2021-04-01",episode_end_date="2021-04-30",
             clinical_anchor_date="2021-04-15")])
    meds=pd.DataFrame([
        dict(documentation_record_id="B1",linked_patient_pseudonymous=token,
             source="11D",source_kind="BTRIS_order",date_of_documentation="2021-02-15",
             clinical_category_primary="systemic_sjd",classification_confidence="high_exact",
             ingredient_canonical="hydroxychloroquine", medication_name_raw="Plaquenil",
             drug_short_label="hydroxychloroquine",route_canonical="oral",
             clinical_category_detail="systemic",review_reasons="",qc_flags="",
             date_evidence_kind="btris_order_or_start_date_unvalidated",source_audit_record_id="",
             actual_exposure_confirmed=False,documentation_dated_usable=True),
        dict(documentation_record_id="C1",linked_patient_pseudonymous=token,
             source="CTDB_original_visit",source_kind="CTDB_reported_current",
             date_of_documentation="2021-02-20",clinical_category_primary="symptomatic_sjd",
             classification_confidence="candidate_review",
             ingredient_canonical="hydroxychloroquine", medication_name_raw="Plaquenil",
             drug_short_label="hydroxychloroquine",route_canonical="oral",
             clinical_category_detail="systemic",review_reasons="",qc_flags="",
             date_evidence_kind="btris_order_or_start_date_unvalidated",source_audit_record_id="",
             actual_exposure_confirmed=False,documentation_dated_usable=True),
        dict(documentation_record_id="B2",linked_patient_pseudonymous=token,
             source="15D",source_kind="BTRIS_order",date_of_documentation="2021-02-15",
             clinical_category_primary="indeterminate",classification_confidence="unmapped",
             ingredient_canonical="hydroxychloroquine", medication_name_raw="Plaquenil",
             drug_short_label="hydroxychloroquine",route_canonical="oral",
             clinical_category_detail="systemic",review_reasons="",qc_flags="",
             date_evidence_kind="btris_order_or_start_date_unvalidated",source_audit_record_id="",
             actual_exposure_confirmed=False,documentation_dated_usable=False)])
    links=[]
    for eid,record,source,kind,day in [
        ("e1","B1","11D","BTRIS_order","2021-02-15"),
        ("e2","B1","11D","BTRIS_order","2021-02-15"),
        ("e1","C1","CTDB_original_visit","CTDB_reported_current","2021-02-20")]:
        links.append(dict(documentation_record_id=record,
             linked_patient_pseudonymous=token,source=source,source_kind=kind,
             clinical_episode_id=eid,
             relation_type="within_episode" if record=="B1" else "CTDB_original_visit_authoritative",
             primary_documentation_relation=True,tolerance_sensitivity_only=False,
             date_of_documentation=day,episode_start_date="2021-01-01",
             episode_end_date="2021-03-31",
             patient_and_episode_verified_against_Step00=True,
             confirmed_medication_exposure=False))
    status=pd.DataFrame([
        dict(documentation_record_id="B1",n_primary_episode_relations=2,
             n_tolerance_sensitivity_relations=0,
             original_match_status="multiple_within_episode"),
        dict(documentation_record_id="C1",n_primary_episode_relations=1,
             n_tolerance_sensitivity_relations=0,
             original_match_status="original_CTDB_visit_verified"),
        dict(documentation_record_id="B2",n_primary_episode_relations=0,
             n_tolerance_sensitivity_relations=0,
             original_match_status="blocked_by_documentation_qc")])
    summary=dict(version="08-medication-episode-links-1.1.0",
        n_original_documentation_records=3,n_primary_episode_relations=3)
    return spine,meds,pd.DataFrame(links),status,summary,KEY

def test_aggregation_preserves_episode_grain_and_order_ids():
    out,qc=med.aggregate(*sample())
    assert len(out)==3
    assert out.duplicated(["patient_id","clinical_episode_id"]).sum()==0
    assert out.meddoc__n_documentation_records.tolist()==[2,1,0]
    assert out[med.NAME_FIELD].iloc[0] == "hydroxychloroquine"
    assert out.meddoc__n_btris_11d_order_documents.tolist()==[1,1,0]
    assert out.meddoc__n_ctdb_original_visit_mentions.tolist()==[1,0,0]
    assert out.meddoc__n_systemic_sjd_candidate_documents.tolist()==[1,1,0]
    assert qc["n_primary_links"]==3
    assert qc["n_btris_orders_with_multiple_episode_links"]==1
    assert not out.columns.isin(["linked_patient_pseudonymous","documentation_record_id"]).any()

def test_tolerance_cannot_enter_main():
    data=list(sample())
    data[2].loc[0,"tolerance_sensitivity_only"]=True
    with pytest.raises(ValueError,match="tolerance_links_forbidden"):
        med.aggregate(*data)

def test_wrong_patient_link_rejected():
    data=list(sample())
    data[2].loc[0,"linked_patient_pseudonymous"]="different"
    with pytest.raises(ValueError,match="primary_link_catalog_linked_patient_pseudonymous_mismatch"):
        med.aggregate(*data)

def test_qc_blocked_order_cannot_be_joined():
    data=list(sample())
    blocked=dict(data[2].iloc[0])
    blocked["documentation_record_id"]="B2"
    blocked["source"]="15D"
    data[2]=pd.concat([data[2],pd.DataFrame([blocked])],ignore_index=True)
    data[3].loc[data[3].documentation_record_id.eq("B2"),"n_primary_episode_relations"]=1
    data[4]["n_primary_episode_relations"]=4
    with pytest.raises(ValueError,match="QC_blocked_BTRIS"):
        med.aggregate(*data)

def test_protected_detail_preserves_source_documents_and_dose_semantics():
    episode,qc,details=med.aggregate(*sample(),include_details=True)
    assert len(details)==qc["n_primary_links"]==3
    assert details.duplicated(["documentation_record_id","clinical_episode_id"]).sum()==0
    assert details.loc[details.documentation_record_id.eq("B1"),"clinical_episode_id"].nunique()==2
    assert details.confirmed_medication_exposure.eq(False).all()
    assert "meddoc__medication_names" in episode.columns
    audit=pd.DataFrame([dict(audit_record_id="11D:1",source="11D",
        dose_raw="200",uom_raw="mg",frequency_raw="once daily")])
    release=pd.DataFrame([dict(audit_record_id="11D:1",source="11D",
        strength_canonical="200 mg",formulation_canonical="tablet")])
    details.loc[details.documentation_record_id.eq("B1"),"source_audit_record_id"]="11D:1"
    details.loc[details.documentation_record_id.eq("B1"),"documentation_record_id"]="BTRIS:11D:1"
    # The BTRIS order is represented in two episode links but only one source order.
    detail=med._attach_btris_prescription_fields(details,audit,release)
    assert detail.loc[detail.source_kind.eq("BTRIS_order"),"dose_raw"].tolist()==["200","200"]
    assert detail.loc[detail.source_kind.eq("CTDB_reported_current"),"dose_raw"].isna().all()
    assert detail.loc[detail.source_kind.eq("CTDB_reported_current"),"dose_semantics"].eq(
        "not_available_for_ctdb_original_visit").all()

"""Exploratory medication candidates within +/- 1 calendar year of a clinical episode.

Match ANY exact date recorded in BTRIS: order/start, order end, or initial
consent. Preserve date provenance and source/QC status. In particular consent
is not a drug date and matches are NOT evidence of treatment administration.
Does not alter canonical Step 08/10 or existing main medication descriptors.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
import json
import os
import re
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
AUDIT=ROOT/"outputs/audit/08_treatments_full_source"
SPINE=ROOT/"data/intermediate/00_clinical_visit_spine.parquet"
STAGED=AUDIT/"reviewed_step10/10_integrated_longitudinal_clinical_episode.parquet"
SOURCES={
    "11D":AUDIT/"restricted/btris_full_source_inputs/11D_medication_records_full.parquet",
    "15D":AUDIT/"restricted/btris_full_source_inputs/15D_medication_records_full.parquet",
}
DATE_FIELDS={
    "order_start":"Order Date/Start Date",
    "order_end":"Order End Date",
    "initial_consent":"Initial Consent Date",
}
RITUX=re.compile(r"rituximab|rituxan|mabthera|truxima|ruxience|riabni",re.I)


def strict_date(value):
    """Exact dates only. Accept BTRIS timestamps with AM/PM, fractional seconds, timezone."""
    from dateutil import parser as date_parser
    if value is None or pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    value = str(value).strip()
    if not value or value.lower() in {"nat", "nan", "none", "null"}:
        return pd.NaT
    # Must contain a recorded day, month and four-digit year.
    if not (re.match(r"^\d{4}-\d{1,2}-\d{1,2}(?:[T ].*)?$", value)
            or re.match(r"^\d{1,2}/\d{1,2}/\d{4}(?:\s+.*)?$", value)):
        return pd.NaT
    try:
        parsed = date_parser.parse(value, dayfirst=False, yearfirst=value[:4].isdigit()
                                   and value[4:5] == "-", fuzzy=False)
        return pd.Timestamp(parsed.date())
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def text(value):
    return "" if value is None or pd.isna(value) else str(value).strip()


def run(audit_root=AUDIT,spine_path=SPINE,staged_path=STAGED):
    audit_root=Path(audit_root).resolve()
    base=audit_root/"restricted"
    catalog=pd.read_parquet(base/"medication_documentation_consolidated"/
                            "08_medications_consolidated_documentation.parquet")
    spine=pd.read_parquet(spine_path)
    secret=(base/".pseudonym_key").read_bytes()
    if len(secret)<32: raise ValueError("invalid_HMAC_key")
    expected={"documentation_record_id","linked_patient_pseudonymous",
              "source_audit_record_id","source_kind","ingredient_canonical",
              "medication_name_raw","qc_flags","review_reasons",
              "classification_confidence"}
    if not expected.issubset(catalog): raise ValueError("missing_catalog_columns")
    if catalog.documentation_record_id.duplicated().any(): raise ValueError("duplicate_documents")
    candidates=catalog.loc[catalog.source_kind.eq("BTRIS_order")].copy()
    if len(candidates)!=9605: 
        raise ValueError("unexpected_BTRIS_document_count")
    if candidates.source_audit_record_id.duplicated().any(): raise ValueError("duplicate_BTRIS_source_ids")
    files={}
    for source in ("11D","15D"):
        file=audit_root/"restricted/btris_full_source_inputs"/(source+"_medication_records_full.parquet")
        files[source]=pd.read_parquet(file)
    for source,frame in files.items():
        if not set(DATE_FIELDS.values()).issubset(frame): raise ValueError("missing_source_date_fields")
    spine=spine.loc[spine.clinical_visit.eq(True)].copy()
    for field in ("episode_start_date","episode_end_date"):
        spine[field]=spine[field].map(strict_date)
    if spine[["episode_start_date","episode_end_date"]].isna().any().any():
        raise ValueError("invalid_Step00_episode_dates")
    if spine[["patient_id","clinical_episode_id"]].duplicated().any():
        raise ValueError("duplicate_Step00_episode")
    spine["token"]=spine.patient_id.map(
        lambda p:hmac.new(secret,("patient\0"+text(p)).encode(),hashlib.sha256).hexdigest())
    groups={key:part for key,part in spine.groupby("token")}
    rows=[]
    invalid_dates=0
    reference_counts={kind:0 for kind in DATE_FIELDS}
    nonblank_counts={kind:0 for kind in DATE_FIELDS}
    invalid_by_kind={kind:0 for kind in DATE_FIELDS}
    for med in candidates.itertuples(index=False):
        source=med.source
        raw_id=text(med.source_audit_record_id)
        match=re.fullmatch(r"(11D|15D):(\d+)",raw_id)
        if match is None or match.group(1)!=source:raise ValueError("bad_source_record_ID")
        pos=int(match.group(2))
        if pos>=len(files[source]):raise ValueError("source_position_out_of_range")
        record=files[source].iloc[pos]
        # Consent is a protocol enrollment reference, NOT medication initiation.
        dates={}
        for kind,field in DATE_FIELDS.items():
            original=text(record[field])
            day=strict_date(original)
            if original:
                nonblank_counts[kind]+=1
            if original and pd.isna(day):
                invalid_dates+=1
                invalid_by_kind[kind]+=1
            if pd.notna(day):
                dates[kind]=day
                reference_counts[kind]+=1
        patient_episodes=groups.get(med.linked_patient_pseudonymous)
        if patient_episodes is None:raise ValueError("BTRIS_patient_not_in_Step00")
        for ep in patient_episodes.itertuples(index=False):
            start,end=ep.episode_start_date,ep.episode_end_date
            valid={kind:day for kind,day in dates.items()
                   if start-pd.DateOffset(years=1)<=day<=end+pd.DateOffset(years=1)}
            if not valid:continue
            # One row per medication-source-record and clinical episode.
            priority=("order_start","order_end","initial_consent")
            selected=next(kind for kind in priority if kind in valid)
            date=valid[selected]
            relation=("within_episode" if start<=date<=end else
                      "prior_to_episode" if date<start else "after_episode")
            # All dates that match are retained; single row avoids inflation.
            medication=text(med.ingredient_canonical) or text(med.medication_name_raw)
            if medication.lower() in {"unknown","unmapped","indeterminate","nan","none"}:
                medication=text(med.medication_name_raw)
            rows.append({
                "patient_id":ep.patient_id,"clinical_episode_id":ep.clinical_episode_id,
                "documentation_record_id":med.documentation_record_id,
                "source":source,"medication_name_candidate":medication,
                "ingredient_canonical":text(med.ingredient_canonical),
                "date_reference_selected":selected,
                "date_reference_kinds_matching":"|".join(valid),
                "date_reference_selected_value":date.date().isoformat(),
                "relation_to_episode":relation,
                "consent_only_basis":set(valid)=={"initial_consent"},
                "requires_manual_chart_review":True,
                "order_end_not_confirmed_discontinuation":True,
                "classification_confidence":text(med.classification_confidence),
                "qc_flags":text(med.qc_flags),"review_reasons":text(med.review_reasons),
                "actual_exposure_confirmed":False
            })
    if (nonblank_counts["order_start"] and reference_counts["order_start"] == 0):
        raise ValueError("all_BTRIS_order_start_dates_failed_parsing_check_source_format")
    out=base/"anydate_medication_associations"
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(out,0o700)
    table=pd.DataFrame(rows,columns=[
        "patient_id","clinical_episode_id","documentation_record_id","source",
        "medication_name_candidate","ingredient_canonical","date_reference_selected",
        "date_reference_kinds_matching","date_reference_selected_value",
        "relation_to_episode","consent_only_basis","requires_manual_chart_review",
        "order_end_not_confirmed_discontinuation","classification_confidence",
        "qc_flags","review_reasons","actual_exposure_confirmed"])
    if table.duplicated(["documentation_record_id","clinical_episode_id"]).any():
        raise ValueError("candidate_duplicate_record_episode")
    detail=out/"08_anydate_medication_episode_candidates.parquet"
    table.to_parquet(detail,index=False);detail.chmod(0o600)
    grouped=[]
    for (patient,episode),group in table.groupby(["patient_id","clinical_episode_id"]):
        grouped.append({
            "patient_id":patient,"clinical_episode_id":episode,
            "medcand__n_associated_documents_1y_anydate":len(group),
            "medcand__n_unique_documents_1y_anydate":group.documentation_record_id.nunique(),
            "medcand__n_consent_only_documents_1y":int(group.consent_only_basis.sum()),
            "medcand__n_order_start_reference_1y":int(
                group.date_reference_kinds_matching.str.contains("order_start",regex=False).sum()),
            "medcand__n_order_end_reference_1y":int(
                group.date_reference_kinds_matching.str.contains("order_end",regex=False).sum()),
            "medcand__n_rituximab_mentions_1y":int(group.medication_name_candidate.str.contains(
                RITUX,na=False).sum()),
            "medcand__medication_names_1y_anydate":"; ".join(sorted(
                {x for x in group.medication_name_candidate if x},key=str.casefold)),
        })
    episode=pd.DataFrame(grouped,columns=[
        "patient_id","clinical_episode_id","medcand__n_associated_documents_1y_anydate",
        "medcand__n_unique_documents_1y_anydate",
        "medcand__n_consent_only_documents_1y",
        "medcand__n_order_start_reference_1y","medcand__n_order_end_reference_1y",
        "medcand__n_rituximab_mentions_1y","medcand__medication_names_1y_anydate"])
    staged=pd.read_parquet(staged_path)
    if len(staged)!=497 or staged.patient_id.nunique()!=159:raise ValueError("bad_staged_cohort")
    if staged[["patient_id","clinical_episode_id"]].duplicated().any():raise ValueError("bad_staged_keys")
    result=staged.merge(episode,on=["patient_id","clinical_episode_id"],how="left",validate="one_to_one")
    numeric=[c for c in episode if c.startswith("medcand__n_")]
    for col in numeric:result[col]=result[col].fillna(0).astype("Int64")
    result["medcand__medication_names_1y_anydate"]=result[
        "medcand__medication_names_1y_anydate"].astype("string")
    if len(result)!=len(staged):raise ValueError("episode_count_changed")
    output=audit_root/"reviewed_step10/10_integrated_longitudinal_with_anydate_candidates.parquet"
    result.to_parquet(output,index=False)
    output.chmod(0o600)
    qc={
        "policy":"Any full-day BTRIS start/end/initial consent within +/- one calendar year of episode bounds",
        "n_candidate_document_episode_pairs":len(table),
        "n_unique_matched_documents":int(table.documentation_record_id.nunique()),
        "n_episodes_with_candidates":int(episode.shape[0]),
        "n_episodes_with_rituximab_candidates":int(
            result.medcand__n_rituximab_mentions_1y.gt(0).sum()),
        "n_patients_with_rituximab_candidates":int(
            result.loc[result.medcand__n_rituximab_mentions_1y.gt(0),"patient_id"].nunique()),
        "n_consent_only_pairs":int(table.consent_only_basis.sum()),
        "n_unparseable_nonblank_dates":invalid_dates,
        "available_date_reference_counts":reference_counts,
        "nonblank_date_reference_counts":nonblank_counts,
        "unparseable_by_date_kind":invalid_by_kind,
        "n_pairs_with_order_start_or_end_reference":int((~table.consent_only_basis).sum()),
        "patient_count":int(result.patient_id.nunique()),"episode_count":len(result),
        "detail_protected_file":str(detail),"staged_episode_dataset":str(output),
        "original_step10_untouched":True,"confirmed_medication_exposure":False,
        "consent_not_drug_administration_date":True,
    }
    qcfile=out/"08_anydate_medication_association_qc.json"
    qcfile.write_text(json.dumps(qc,indent=2,sort_keys=True)+"\n")
    qcfile.chmod(0o600)
    print(json.dumps(qc,indent=2,sort_keys=True))
    return qc

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--audit-root",type=Path,default=AUDIT)
    parser.add_argument("--spine",type=Path,default=SPINE)
    parser.add_argument("--staged",type=Path,default=STAGED)
    opts=parser.parse_args()
    run(opts.audit_root,opts.spine,opts.staged)

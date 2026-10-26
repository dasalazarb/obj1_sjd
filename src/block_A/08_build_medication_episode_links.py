"""Build many-to-many medication-DOCUMENTATION to clinical-episode links.

Frozen Step 00 episodes are authoritative. One medication documentation record
may link to multiple clinical episodes when its exact documented date is inside
each episode interval. Preserve order-grain and link-grain tables separately.
One-calendar-year proximity is a distinct exploratory association, never a primary
inside-episode relation. Existing BTRIS QC holds remain blocked.

Never infer prescriptions administered, exposure, ingestion, discontinuation,
treatment eras, clinical effects, or a single uniquely correct episode.
Only protected Step 08 artifacts are written.
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

VERSION = "08-medication-episode-links-1.1.0"
ROOT = Path(__file__).resolve().parents[2]
AUDIT_SUBDIR = "outputs/audit/08_treatments_v2"
MED_FILE = "medication_documentation_consolidated/08_medications_consolidated_documentation.parquet"
MED_SUMMARY = "medication_documentation_consolidated/08_medications_consolidated_summary.json"
MATCH_FILE = "btris_episode_window_candidates/08_btris_episode_candidate_matches.parquet"
MATCH_STATUS = "btris_episode_window_candidates/08_btris_episode_match_status.parquet"
MATCH_SUMMARY = "btris_episode_window_candidates/08_btris_episode_match_summary.json"

def required(frame, columns, name):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}_missing_columns:" + ",".join(missing))

def exact_day(value):
    if pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    text = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return pd.NaT
    try:
        return pd.Timestamp(text)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT

def token(secret, value):
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError("missing_Step00_patient_id")
    return hmac.new(secret,("patient\0"+str(value).strip()).encode(),
                    hashlib.sha256).hexdigest()

def columns_for_links():
    return [
        "documentation_record_id","linked_patient_pseudonymous","source",
        "source_kind","clinical_episode_id","relation_type",
        "primary_documentation_relation","tolerance_sensitivity_only",
        "date_of_documentation","episode_start_date","episode_end_date",
        "clinical_anchor_date","match_origin","within_one_calendar_year",
        "patient_and_episode_verified_against_Step00",
        "documentation_record_not_duplicated","confirmed_medication_exposure",
        "linkage_version",
    ]

def _checked_spine(all_spine, clinical_spine, secret):
    fields = ["patient_id","clinical_episode_id","clinical_visit",
              "episode_start_date","episode_end_date","clinical_anchor_date"]
    required(all_spine, fields, "all_episode_spine")
    required(clinical_spine, fields, "clinical_visit_spine")
    def prepared(frame,label):
        out=frame[fields].copy()
        if (out[["patient_id","clinical_episode_id"]].isna().any().any()
            or out.duplicated(["patient_id","clinical_episode_id"]).any()):
            raise ValueError(label+"_duplicate_or_missing_episode_key")
        out["linked_patient_pseudonymous"]=out.patient_id.map(lambda v:token(secret,v))
        out["clinical_episode_id"]=out.clinical_episode_id.astype(str).str.strip()
        for col in ["episode_start_date","episode_end_date","clinical_anchor_date"]:
            out[col]=out[col].map(exact_day)
        if (out[["episode_start_date","episode_end_date",
                  "clinical_anchor_date"]].isna().any().any()
            or out.episode_start_date.gt(out.clinical_anchor_date).any()
            or out.clinical_anchor_date.gt(out.episode_end_date).any()):
            raise ValueError(label+"_invalid_episode_dates")
        if out[["linked_patient_pseudonymous","clinical_episode_id"]].duplicated().any():
            raise ValueError(label+"_duplicate_token_episode")
        return out
    all_rows=prepared(all_spine,"all_spine")
    clinic=prepared(clinical_spine,"clinical_spine")
    from_all=all_rows.loc[all_rows.clinical_visit.eq(True)]
    keys=["linked_patient_pseudonymous","clinical_episode_id"]
    datecols=["episode_start_date","episode_end_date","clinical_anchor_date"]
    cmp=clinic.merge(from_all[keys+datecols],on=keys,how="outer",
                      validate="one_to_one",suffixes=("","_all"),indicator=True)
    if not cmp["_merge"].eq("both").all() or any(
        not cmp[c].eq(cmp[c+"_all"]).all() for c in datecols):
        raise ValueError("clinical_spine_does_not_match_Step00_all_spine")
    return all_rows, clinic

def build(medications, candidates, match_status, all_spine, clinical_spine,
          secret, consolidated_summary, matching_summary):
    if not isinstance(secret,bytes) or len(secret)<32:
        raise ValueError("invalid_protected_pseudonym_key")
    if consolidated_summary.get("version")!="08-medication-documentation-consolidated-1.0.0":
        raise ValueError("unexpected_consolidated_input_version")
    if matching_summary.get("version")!="08-btris-episode-candidate-match-1.1.0":
        raise ValueError("unexpected_matching_input_version")
    required(medications,[
        "documentation_record_id","linked_patient_pseudonymous","source",
        "source_kind","date_of_documentation","clinical_episode_id",
        "documentation_dated_usable","date_within_authoritative_episode",
        "actual_exposure_confirmed"],"medication_documentation")
    required(candidates,[
        "documentation_record_id","linked_patient_pseudonymous","source",
        "clinical_episode_id","within_episode",
        "within_one_calendar_year_proximity","episode_start_date",
        "episode_end_date","clinical_anchor_date"],"BTRIS_candidates")
    required(match_status,[
        "documentation_record_id","linked_patient_pseudonymous",
        "source","match_status","n_candidates","n_within_episode",
        "n_tolerance_only"],"BTRIS_status")
    if (medications.documentation_record_id.isna().any()
        or medications.documentation_record_id.duplicated().any()):
        raise ValueError("duplicate_or_missing_medication_documentation_id")
    if medications.actual_exposure_confirmed.ne(False).any():
        raise ValueError("confirmed_exposure_flag_not_allowed")
    if len(medications)!=consolidated_summary["n_total_documentation_records"]:
        raise ValueError("consolidated_row_count_not_reconciled")
    if candidates.duplicated(["documentation_record_id","clinical_episode_id"]).any():
        raise ValueError("duplicate_BTRIS_order_episode_candidate")
    if (match_status.documentation_record_id.isna().any()
        or match_status.documentation_record_id.duplicated().any()):
        raise ValueError("duplicate_BTRIS_match_status")
    btris=medications.loc[medications.source_kind.eq("BTRIS_order")].copy()
    ctdb=medications.loc[medications.source_kind.eq("CTDB_reported_current")].copy()
    if len(btris)+len(ctdb)!=len(medications):
        raise ValueError("unknown_medication_source_kind")
    if (len(btris)!=matching_summary["n_btris_orders"]
        or len(btris)!=consolidated_summary["n_btris_order_rows"]
        or len(ctdb)!=consolidated_summary["n_ctdb_original_visit_mentions"]):
        raise ValueError("source_documentation_counts_changed")
    if set(match_status.documentation_record_id)!=set(btris.documentation_record_id):
        raise ValueError("BTRIS_match_status_not_one_per_order")
    if int(match_status.n_candidates.sum())!=len(candidates):
        raise ValueError("BTRIS_candidate_total_disagrees_with_status")
    if not set(candidates.documentation_record_id).issubset(set(btris.documentation_record_id)):
        raise ValueError("unknown_BTRIS_candidate_order")
    if (int(match_status.n_candidates.gt(1).sum())!=
            matching_summary["n_btris_with_multiple_candidates"]):
        raise ValueError("BTRIS_multicandidate_reconciliation_failed")

    all_rows,clinical=_checked_spine(all_spine,clinical_spine,secret)
    all_index=all_rows.set_index(["linked_patient_pseudonymous","clinical_episode_id"])
    clinical_keys=set(zip(clinical.linked_patient_pseudonymous,
                          clinical.clinical_episode_id))
    meds=medications.set_index("documentation_record_id",drop=False)
    by_status=match_status.set_index("documentation_record_id")
    rows=[]
    for r in candidates.itertuples(index=False):
        m=meds.loc[r.documentation_record_id]
        if (m.source_kind!="BTRIS_order" or m.source!=r.source
            or m.linked_patient_pseudonymous!=r.linked_patient_pseudonymous):
            raise ValueError("BTRIS_candidate_patient_or_source_mismatch")
        key=(r.linked_patient_pseudonymous,str(r.clinical_episode_id))
        if key not in clinical_keys or key not in all_index.index:
            raise ValueError("BTRIS_candidate_episode_not_in_clinical_Step00")
        s=all_index.loc[key]
        event=exact_day(m.date_of_documentation)
        if pd.isna(event) or not bool(m.documentation_dated_usable):
            raise ValueError("BTRIS_candidate_has_blocked_or_missing_order_date")
        start,end,anchor=(s[x] for x in
                          ["episode_start_date","episode_end_date","clinical_anchor_date"])
        if (exact_day(r.episode_start_date)!=start or
            exact_day(r.episode_end_date)!=end or
            exact_day(r.clinical_anchor_date)!=anchor):
            raise ValueError("BTRIS_candidate_episode_dates_differ_from_Step00")
        inside=start<=event<=end
        year_window=(start-pd.DateOffset(years=1)<=event<=
                      end+pd.DateOffset(years=1))
        if not year_window or inside!=bool(r.within_episode) or (
            (not inside)!=bool(r.within_one_calendar_year_proximity)):
            raise ValueError("BTRIS_candidate_temporal_classification_incorrect")
        rows.append(dict(
            documentation_record_id=r.documentation_record_id,
            linked_patient_pseudonymous=m.linked_patient_pseudonymous,
            source=m.source,source_kind=m.source_kind,
            clinical_episode_id=str(r.clinical_episode_id),
            relation_type="within_episode" if inside else "year_proximity_only",
            primary_documentation_relation=inside,
            tolerance_sensitivity_only=not inside,
            date_of_documentation=event.date().isoformat(),
            episode_start_date=start.date().isoformat(),
            episode_end_date=end.date().isoformat(),
            clinical_anchor_date=anchor.date().isoformat(),
            match_origin="BTRIS_temporal_candidates",
            within_one_calendar_year=year_window,
            patient_and_episode_verified_against_Step00=True,
            documentation_record_not_duplicated=True,
            confirmed_medication_exposure=False,linkage_version=VERSION))
    for r in ctdb.itertuples(index=False):
        patient=r.linked_patient_pseudonymous
        eid=str(r.clinical_episode_id).strip()
        key=(patient,eid)
        if not eid or key not in all_index.index:
            raise ValueError("CTDB_original_visit_episode_not_in_Step00")
        s=all_index.loc[key]
        event=exact_day(r.date_of_documentation)
        if (pd.isna(event) or
            not s.episode_start_date<=event<=s.episode_end_date or
            r.date_within_authoritative_episode is not True and
            r.date_within_authoritative_episode != True):
            raise ValueError("CTDB_original_visit_date_invalid_for_episode")
        if bool(r.actual_exposure_confirmed):
            raise ValueError("CTDB_exposure_not_confirmed")
        rows.append(dict(
            documentation_record_id=r.documentation_record_id,
            linked_patient_pseudonymous=patient,source=r.source,
            source_kind=r.source_kind,clinical_episode_id=eid,
            relation_type="CTDB_original_visit_authoritative",
            primary_documentation_relation=True,
            tolerance_sensitivity_only=False,
            date_of_documentation=event.date().isoformat(),
            episode_start_date=s.episode_start_date.date().isoformat(),
            episode_end_date=s.episode_end_date.date().isoformat(),
            clinical_anchor_date=s.clinical_anchor_date.date().isoformat(),
            match_origin="CTDB_original_row_map_via_Step00",
            within_one_calendar_year=True,
            patient_and_episode_verified_against_Step00=True,
            documentation_record_not_duplicated=True,
            confirmed_medication_exposure=False,linkage_version=VERSION))
    edges=pd.DataFrame(rows,columns=columns_for_links())
    if edges.duplicated(["documentation_record_id","clinical_episode_id"]).any():
        raise ValueError("duplicate_medication_episode_link")
    primary=edges.loc[edges.primary_documentation_relation].copy()
    sensitivity=edges.loc[edges.tolerance_sensitivity_only].copy()
    if len(edges)!=len(candidates)+len(ctdb):
        raise ValueError("source_relation_count_not_reconciled")
    primary_counts=primary.groupby("documentation_record_id").size().to_dict()
    tolerance_counts=sensitivity.groupby("documentation_record_id").size().to_dict()
    raw_counts=edges.groupby("documentation_record_id").size().to_dict()
    status_counts=match_status.set_index("documentation_record_id")
    results=[]
    for r in medications.itertuples(index=False):
        rid=r.documentation_record_id
        nmain=int(primary_counts.get(rid,0))
        ntol=int(tolerance_counts.get(rid,0))
        if r.source_kind=="BTRIS_order":
            previous=status_counts.loc[rid]
            if (nmain!=int(previous.n_within_episode) or
                ntol!=int(previous.n_tolerance_only) or
                nmain+ntol!=int(previous.n_candidates)):
                raise ValueError("BTRIS_original_status_counts_mismatch")
            category=str(previous.match_status)
            qc_block=category=="blocked_by_documentation_qc"
        else:
            if nmain!=1 or ntol!=0:
                raise ValueError("CTDB_must_have_exactly_one_authoritative_link")
            category="original_CTDB_visit_verified"
            qc_block=False
        results.append(dict(
            documentation_record_id=rid,
            linked_patient_pseudonymous=r.linked_patient_pseudonymous,
            source=r.source,source_kind=r.source_kind,
            documentation_date=str(r.date_of_documentation),
            original_order_or_visit_row_count=1,
            n_primary_episode_relations=nmain,
            n_tolerance_sensitivity_relations=ntol,
            n_all_candidate_episode_relations=int(raw_counts.get(rid,0)),
            original_match_status=category,
            BTRIS_QC_blocked=qc_block,
            not_linked_to_any_episode=(nmain+ntol==0),
            multiple_primary_episode_relations=(nmain>1),
            confirmed_medication_exposure=False,linkage_version=VERSION))
    record_status=pd.DataFrame(results)
    if len(record_status)!=len(medications) or record_status.documentation_record_id.duplicated().any():
        raise ValueError("documentation_record_grain_changed")
    if (record_status.n_primary_episode_relations.sum()!=len(primary)
        or record_status.n_tolerance_sensitivity_relations.sum()!=len(sensitivity)):
        raise ValueError("relation_reconciliation_failed")
    aggregate=record_status.groupby(["source","source_kind","original_match_status"],
        dropna=False).agg(
        n_unique_documentation_records=("documentation_record_id","nunique"),
        n_primary_episode_relations=("n_primary_episode_relations","sum"),
        n_tolerance_sensitivity_relations=("n_tolerance_sensitivity_relations","sum"),
        n_pseudonymous_patients=("linked_patient_pseudonymous","nunique")
        ).reset_index()
    summary=dict(
        version=VERSION,
        n_original_documentation_records=int(len(medications)),
        n_BTRIS_order_records=int(len(btris)),
        n_CTDB_original_visit_records=int(len(ctdb)),
        n_distinct_original_documentation_records_in_primary=int(
            primary.documentation_record_id.nunique()),
        n_primary_episode_relations=int(len(primary)),
        n_BTRIS_within_episode_relations=int(
            primary.source_kind.eq("BTRIS_order").sum()),
        n_BTRIS_orders_within_one_or_more_episodes=int(
            record_status.loc[record_status.source_kind.eq("BTRIS_order"),
                "n_primary_episode_relations"].gt(0).sum()),
        n_BTRIS_orders_with_multiple_direct_relations=int(
            record_status.loc[record_status.source_kind.eq("BTRIS_order"),
                "multiple_primary_episode_relations"].sum()),
        n_CTDB_authoritative_relations=int(
            primary.source_kind.eq("CTDB_reported_current").sum()),
        n_BTRIS_year_proximity_relations=int(len(sensitivity)),
        n_BTRIS_orders_with_tolerance_candidates=int(
            record_status.loc[record_status.source_kind.eq("BTRIS_order"),
                "n_tolerance_sensitivity_relations"].gt(0).sum()),
        n_BTRIS_QC_blocked_still_unlinked=int(
            record_status.BTRIS_QC_blocked.sum()),
        n_BTRIS_without_any_candidate=int(
            record_status.loc[record_status.source_kind.eq("BTRIS_order"),
                "not_linked_to_any_episode"].sum()),
        n_original_BTRIS_orders=int(len(btris)),
        no_automatic_unique_episode_selection=True,
        multiple_episode_relations_preserved=True,
        one_original_medication_order_not_multiplied=True,
        date_evidence_not_exposure=True,
        exposure_inference="NO_GO",
        Step00_episode_boundaries_unchanged=True,
        primary_includes_year_proximity=False,
        outputs_protected_only=True,
    )
    return primary,sensitivity,record_status,aggregate,summary

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def run(audit_root,all_spine_path,clinical_spine_path):
    protected=Path(audit_root).resolve()/"restricted"
    sources=dict(
        medications=protected/MED_FILE,
        consolidated_summary=protected/MED_SUMMARY,
        candidates=protected/MATCH_FILE,
        match_status=protected/MATCH_STATUS,
        match_summary=protected/MATCH_SUMMARY,
        all_Step00_episodes=Path(all_spine_path).resolve(),
        clinical_Step00_episodes=Path(clinical_spine_path).resolve(),
    )
    if any(not p.is_file() for p in sources.values()):
        raise FileNotFoundError("missing_step08_or_step00_inputs")
    key=(protected/".pseudonym_key").read_bytes()
    result=build(
        pd.read_parquet(sources["medications"]),
        pd.read_parquet(sources["candidates"]),
        pd.read_parquet(sources["match_status"]),
        pd.read_parquet(sources["all_Step00_episodes"]),
        pd.read_parquet(sources["clinical_Step00_episodes"]),
        key,
        json.loads(sources["consolidated_summary"].read_text()),
        json.loads(sources["match_summary"].read_text()),
    )
    primary,tolerance,status,qc,summary=result
    summary["input_sha256"]={name:sha256(p) for name,p in sources.items()}
    destination=protected/"medication_episode_documentation_links"
    destination.mkdir(parents=True,mode=0o700,exist_ok=True)
    os.chmod(destination,0o700)
    for name,frame in [
        ("08_medication_episode_primary_links.parquet",primary),
        ("08_medication_episode_year_proximity.parquet",tolerance),
        ("08_medication_documentation_record_status.parquet",status)]:
        p=destination/name
        frame.to_parquet(p,index=False)
        os.chmod(p,0o600)
    p=destination/"08_medication_episode_link_qc.csv"
    qc.to_csv(p,index=False)
    os.chmod(p,0o600)
    p=destination/"08_medication_episode_link_summary.json"
    p.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    os.chmod(p,0o600)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-root",type=Path,
                   default=ROOT/AUDIT_SUBDIR)
    p.add_argument("--episode-spine-all",type=Path,
                   default=ROOT/"data/intermediate/00_episode_spine_all.parquet")
    p.add_argument("--clinical-visit-spine",type=Path,
                   default=ROOT/"data/intermediate/00_clinical_visit_spine.parquet")
    opts=p.parse_args()
    print(json.dumps(run(opts.audit_root,opts.episode_spine_all,
                         opts.clinical_visit_spine),indent=2,sort_keys=True))

if __name__=="__main__":
    main()

"""Rebuild full-source Step 08 in isolated protected audit root, with QC gates.

Never overwrites prior Step 08, current Step 10 or frozen Step 00.
Publication of replacement Step 10 must be a separate explicit action.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/"src"/"block_A"
RECOVER=HERE/"08_prepare_full_btris_medication_inputs.py"
STEPS=[
    ("audit","08_treatment_audit_v2.py"),
    ("release","08_build_medication_dataset.py"),
    ("ctdb","08_ctdb_visit_recovery.py"),
    ("merge","08_merge_medication_documentation.py"),
    ("match","08_match_btris_episode_windows.py"),
    ("links","08_build_medication_episode_links.py"),
]
def cmd(script,*arguments):
    command=[sys.executable,str(HERE/script),*map(str,arguments)]
    print("RUN:",script,flush=True)
    subprocess.run(command, cwd=ROOT,check=True)
def check(art,expected,field):
    data=json.loads(art.read_text())
    if data.get(field)!=expected:
        raise ValueError("full_source_QC_gate_failed:"+str(art)+":"+field+
                         ":observed="+str(data.get(field))+
                         ",expected="+str(expected))
    return data
def protected_unassigned(audit_root):
    protected=audit_root/"restricted"
    catalog=pd.read_parquet(protected/"medication_documentation_consolidated"/
                            "08_medications_consolidated_documentation.parquet")
    status=pd.read_parquet(protected/"medication_episode_documentation_links"/
                           "08_medication_documentation_record_status.parquet")
    combined=catalog.merge(status[["documentation_record_id",
                                   "n_primary_episode_relations",
                                   "original_match_status"]],
                           on="documentation_record_id",validate="one_to_one")
    missing=combined.loc[
        combined.n_primary_episode_relations.eq(0),
        ["documentation_record_id","linked_patient_pseudonymous","source",
         "source_kind","medication_name_raw","ingredient_canonical",
         "date_of_documentation","date_precision","classification_confidence",
         "clinical_category_primary","qc_flags","review_reasons",
         "original_match_status"]].copy()
    out=protected/"patient_level_unassigned_medications"
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(out,0o700)
    file=out/"08_unassigned_medication_documentation.parquet"
    missing.to_parquet(file,index=False)
    os.chmod(file,0o600)
    counts={
        "n_unassigned_medication_documents":int(len(missing)),
        "n_btris_unassigned":int(missing.source_kind.eq("BTRIS_order").sum()),
        "n_undated_unassigned":int(
            missing.date_of_documentation.fillna("").astype(str).eq("").sum()),
        "patient_level_only":True,
        "not_a_treatment_exposure":True,
    }
    qc=out/"08_unassigned_medication_documentation_qc.json"
    qc.write_text(json.dumps(counts,indent=2)+"\n")
    os.chmod(qc,0o600)
    return counts
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-root",type=Path,default=ROOT/"outputs/audit/08_treatments_full_source")
    p.add_argument("--eda-root",type=Path,default=Path("/vf/users/salazarda/data/eda_sjd"))
    p.add_argument("--cohort",type=Path,default=ROOT/"data/raw/list_ids_longitudinal.csv")
    p.add_argument("--spine",type=Path,default=ROOT/"data/intermediate/00_clinical_visit_spine.parquet")
    p.add_argument("--all-spine",type=Path,default=ROOT/"data/intermediate/00_episode_spine_all.parquet")
    p.add_argument("--output",type=Path,
                   default=ROOT/"outputs/audit/08_treatments_full_source/reviewed_step10/10_integrated_longitudinal_clinical_episode.parquet")
    p.add_argument("--skip-rebuild",action="store_true",
                   help="Rebuild Step 10 from already regenerated full-source 08 artifacts")
    a=p.parse_args()
    audit=a.audit_root.resolve()
    legacy=(ROOT/"outputs/audit/08_treatments_v2").resolve()
    if audit==legacy or audit.is_relative_to(legacy):
        raise ValueError("cannot_overwrite_legacy_medication_audit")
    if "restricted" in audit.parts:
        raise ValueError("audit_root_must_be_parent_of_restricted")
    if not a.skip_rebuild:
        cmd("08_prepare_full_btris_medication_inputs.py",
            "--eda-root",a.eda_root,"--cohort",a.cohort,"--audit-root",audit)
        candidate=audit/"restricted/btris_full_source_inputs"
        prep=check(candidate/"08_btris_full_source_inputs_qc.json",159,"n_eligible_patients")
        for source in ("11D","15D"):
            if prep["source"][source]["n_rituximab_rows"]==0:
                raise ValueError("rituximab_missing_after_full_source_recovery_"+source)
        cmd("08_treatment_audit_v2.py",
            "--btris-11d",candidate/"11D_medication_records_full.parquet",
            "--btris-15d",candidate/"15D_medication_records_full.parquet",
            "--output-dir",audit)
        summary=json.loads((audit/"08_treatment_classification_v21_summary.json").read_text())
        if summary.get("run_status")!="completed_with_review_gates":
            raise ValueError("full_source_Step08_audit_not_completed")
        for script in ("08_build_medication_dataset.py","08_ctdb_visit_recovery.py",
                       "08_merge_medication_documentation.py"):
            cmd(script,"--audit-root",audit)
        cmd("08_match_btris_episode_windows.py","--audit-root",audit,
            "--clinical-spine",a.spine)
        cmd("08_build_medication_episode_links.py","--audit-root",audit,
            "--episode-spine-all",a.all_spine,"--clinical-visit-spine",a.spine)
    unassigned=protected_unassigned(audit)
    out=a.output.resolve()
    legacy_step10=(ROOT/"data/analytic/10_integrated_longitudinal_clinical_episode.parquet").resolve()
    if out==legacy_step10 or out.is_relative_to(legacy):
        raise ValueError("full_source_Step10_output_must_be_staged_separately")
    out.parent.mkdir(parents=True,exist_ok=True)
    cmd("10_build_integrated_longitudinal_dataset.py",
        "--medication-audit-root",audit,"--output",out)
    analytic=pd.read_parquet(out)
    if len(analytic)!=497 or analytic.patient_id.nunique()!=159:
        raise ValueError("Step10_recovered_medications_changed_frozen_clinical_cohort")
    names=analytic["meddoc__medication_names"].fillna("").astype("string")
    ritux=names.str.contains("rituximab",case=False,regex=False)
    detail=audit/"restricted/step10_medication_episode_details/10_medication_documentation_episode_detail.parquet"
    records=pd.read_parquet(detail)
    if len(records)==0:
        raise ValueError("no_recovered_medication_episode_details")
    final={
        "status":"staged_full_source_only",
        "step10_staged_output":str(out),
        "n_episodes":len(analytic),
        "n_patients":int(analytic.patient_id.nunique()),
        "n_detail_document_episode_links":len(records),
        "n_episodes_with_documented_rituximab":int(ritux.sum()),
        "unassigned_patient_level":unassigned,
        "old_step10_untouched":True,
        "exposure_confirmed":False,
    }
    qc=audit/"restricted/08_full_source_rebuild_qc.json"
    qc.write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    os.chmod(qc,0o600)
    print(json.dumps(final,indent=2,sort_keys=True))

if __name__=="__main__":
    main()

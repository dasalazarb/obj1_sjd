"""Aggregate rituximab coverage and reasons it lacks clinical episode links.

Protected inputs only; output contains aggregated counts, no patient IDs or raw
medication names, source rows, dates, or identifiers. Never imputes dates.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

PATTERN=r"rituximab|rituxan|mabthera|truxima|ruxience|riabni"


def run(audit_root):
    protected=Path(audit_root).resolve()/"restricted"
    catalog=pd.read_parquet(protected/"medication_documentation_consolidated"/
                            "08_medications_consolidated_documentation.parquet")
    status=pd.read_parquet(protected/"medication_episode_documentation_links"/
                           "08_medication_documentation_record_status.parquet")
    match=pd.read_parquet(protected/"btris_episode_window_candidates"/
                          "08_btris_episode_match_status.parquet")
    cols=["medication_name_raw","ingredient_canonical","drug_short_label"]
    if not all(c in catalog for c in cols):
        raise ValueError("missing_name_columns")
    mask=pd.Series(False,index=catalog.index)
    for c in cols:
        mask |= catalog[c].astype("string").str.contains(
            PATTERN,case=False,regex=True,na=False)
    target=catalog.loc[mask].copy()
    if target.documentation_record_id.duplicated().any():
        raise ValueError("duplicate_source_record")
    target=target.merge(status[["documentation_record_id",
        "n_primary_episode_relations","original_match_status"]],
        on="documentation_record_id",how="left",validate="one_to_one",indicator="_status_merge")
    if not target["_status_merge"].eq("both").all():
        raise ValueError("source_status_link_missing")
    target=target.drop(columns="_status_merge")
    matched=target.loc[target.source_kind.eq("BTRIS_order")].merge(
        match[["documentation_record_id","match_status"]],
        on="documentation_record_id",validate="one_to_one",how="left",indicator="_match_merge")
    if not matched["_match_merge"].eq("both").all():
        raise ValueError("BTRIS_match_status_missing")
    def counter(series):
        return {str(k):int(v) for k,v in series.fillna("missing").value_counts().items()}
    report={
        "total_rituximab_documentation_records":int(len(target)),
        "btris_rituximab_documentation_records":int(len(matched)),
        "sources":counter(target.source),
        "n_with_primary_episode_link":int(target.n_primary_episode_relations.gt(0).sum()),
        "n_primary_episode_relations":int(target.n_primary_episode_relations.sum()),
        "n_without_documentation_date":int(
            target.date_of_documentation.fillna("").astype("string").str.strip().eq("").sum()),
        "classification_confidence":counter(target.classification_confidence),
        "link_status":counter(target.original_match_status),
        "btris_match_status":counter(matched.match_status),
        "interpretation":"Order documentation is not confirmed administration or exposure.",
    }
    print(json.dumps(report,indent=2,sort_keys=True))
    out=protected/"btris_source_coverage_audit"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"08_rituximab_linkage_qc.json"
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    path.chmod(0o600)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--audit-root",type=Path,
                   default=Path("outputs/audit/08_treatments_full_source"))
    args=p.parse_args()
    run(args.audit_root)

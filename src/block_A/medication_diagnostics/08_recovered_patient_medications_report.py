"""Protected patient-level summary of medication documentation.

Preserves undated documents; does not infer medication administration.
"""
from pathlib import Path
import argparse
import json
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=Path("outputs/audit/08_treatments_full_source"))
    args = parser.parse_args()
    root = args.audit_root / "restricted"
    source = root / "patient_level_unassigned_medications" / "08_all_patient_medication_documentation.parquet"
    if not source.is_file():
        raise FileNotFoundError(source)
    records = pd.read_parquet(source)
    needed = {"linked_patient_pseudonymous", "medication_name_raw",
              "ingredient_canonical", "date_of_documentation",
              "n_primary_episode_relations", "original_match_status"}
    if not needed.issubset(records):
        raise ValueError("missing_recovered_documentation_fields")
    if records.linked_patient_pseudonymous.isna().any():
        raise ValueError("unlinked_patient_record")
    pattern = r"rituximab|rituxan|mabthera|truxima|ruxience|riabni"
    names = records.medication_name_raw.astype("string").fillna("")
    ingredients = records.ingredient_canonical.astype("string").fillna("")
    rx = names.str.contains(pattern, case=False, regex=True, na=False) | ingredients.str.contains(pattern, case=False, regex=True, na=False)
    records = records.copy()
    records["rituximab_mention"] = rx
    records["undated"] = records.date_of_documentation.astype("string").fillna("").str.strip().eq("")
    records["has_episode"] = records.n_primary_episode_relations.gt(0)
    patient = records.groupby("linked_patient_pseudonymous", as_index=False).agg(
        n_documents=("rituximab_mention", "size"),
        n_rituximab_mentions=("rituximab_mention", "sum"),
        n_undated=("undated", "sum"),
        n_without_primary_episode=("has_episode", lambda s: int((~s).sum())),
    )
    dest = root / "recovered_patient_medications"
    dest.mkdir(mode=0o700, parents=True, exist_ok=True)
    summary_file = dest / "08_patient_medications_anytime_summary.parquet"
    patient.to_parquet(summary_file, index=False)
    summary_file.chmod(0o600)
    selected = records.loc[records.rituximab_mention]
    summary = {
        "n_documentation_records": len(records),
        "n_patients_with_any_documentation": len(patient),
        "n_rituximab_documents": len(selected),
        "n_patients_with_rituximab_mentions": int(selected.linked_patient_pseudonymous.nunique()),
        "n_rituximab_documents_with_primary_episode": int(selected.has_episode.sum()),
        "n_rituximab_documents_without_date": int(selected.undated.sum()),
        "rituximab_unlinked_reasons": {
            str(k): int(v) for k, v in selected.loc[~selected.has_episode, "original_match_status"].value_counts().items()
        },
        "medication_exposure_confirmed": False,
    }
    output = dest / "08_recovered_patient_medications_qc.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    output.chmod(0o600)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

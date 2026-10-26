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
    summary = {"n_documentation_records": len(records)}
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

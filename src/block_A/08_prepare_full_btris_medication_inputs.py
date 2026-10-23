"""Rebuild protected BTRIS medication inputs from existing cohort-filtered EDA CSVs.

This preserves all eligible source rows, including orders with no Order Date.
Never substitutes Initial Consent Date for a medication date. Output is an
alternative, versioned input to Step 08; legacy upstream Parquet is untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
import pandas as pd

SOURCES = {
    "11D": "data_intermediate/BTRIS/11D/Medication288444-1.csv",
    "15D": "data_intermediate/BTRIS/15D/Medication288476-1.csv",
}
NEEDED = ("MRN", "Medication Name", "Drug Ingredient", "Order Date/Start Date",
          "Order End Date", "Initial Consent Date")
RITUX = r"rituximab|rituxan|mabthera|truxima|ruxience|riabni"


def patient_key(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s or s.casefold() in ("nan", "none", "null", "<na>"):
        return ""
    if re.fullmatch(r"[0-9]+[.]0+", s):
        s = s.split(".", 1)[0]
    return re.sub(r"^0+", "", re.sub(r"[-/\\\s]", "", s))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(eda_root, ids_path, audit_root):
    root = Path(audit_root).resolve() / "restricted" / "btris_full_source_inputs"
    if "restricted" not in root.parts:
        raise ValueError("BTRIS_candidate_must_be_protected")
    if root.is_relative_to(Path(eda_root).resolve()):
        raise ValueError("cannot_write_back_into_EDA")
    eligible_frame = pd.read_csv(ids_path, dtype="string", usecols=["patient_id"])
    keys = eligible_frame.patient_id.map(patient_key)
    if keys.eq("").any() or keys.duplicated().any():
        raise ValueError("invalid_or_colliding_longitudinal_patient_ids")
    eligible = set(keys)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    report = {"version": "08-btris-recovered-inputs-1.0",
              "n_eligible_patients": len(eligible), "source": {},
              "no_consent_date_substitution": True,
              "no_medication_administration_inference": True}
    paths = {}
    for protocol, relative in SOURCES.items():
        path = (Path(eda_root) / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        df = pd.read_csv(path, dtype="string", keep_default_na=False, low_memory=False)
        if not df.columns.is_unique:
            raise ValueError("duplicate_BTRIS_source_columns")
        missing = set(NEEDED) - set(df.columns)
        if missing:
            raise ValueError("BTRIS_missing_columns:" + str(sorted(missing)))
        selected = df.loc[df.MRN.map(patient_key).isin(eligible)].copy()
        if selected.empty:
            raise ValueError("no_longitudinal_BTRIS_medications_" + protocol)
        if selected.MRN.map(patient_key).eq("").any():
            raise ValueError("empty_selected_patient_id")
        names = selected["Medication Name"].fillna("")
        ing = selected["Drug Ingredient"].fillna("")
        ritux = (names.str.contains(RITUX, case=False, na=False) |
                 ing.str.contains(RITUX, case=False, na=False))
        start = selected["Order Date/Start Date"].fillna("").str.strip().ne("")
        output = root / (protocol + "_medication_records_full.parquet")
        # Preserve an existing candidate only when content and source are unchanged.
        if output.exists():
            old = pd.read_parquet(output)
            if not old.equals(selected):
                raise ValueError("existing_BTRIS_candidate_differs_do_not_overwrite:" + protocol)
        else:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".medfull_",
                                             suffix=".parquet", delete=False) as fp:
                tmp = Path(fp.name)
            try:
                selected.to_parquet(tmp, index=False)
                os.chmod(tmp, 0o600)
                os.replace(tmp, output)
            finally:
                if tmp.exists():
                    tmp.unlink()
        paths[protocol] = output
        report["source"][protocol] = {
            "filtered_csv": str(path), "filtered_csv_sha256": sha256(path),
            "candidate_parquet": str(output), "candidate_sha256": sha256(output),
            "n_original_filtered_csv": len(df),
            "n_cohort_records": len(selected),
            "n_cohort_patients": int(selected.MRN.map(patient_key).nunique()),
            "n_named": int((names.str.strip().ne("") | ing.str.strip().ne("")).sum()),
            "n_no_order_start": int((~start).sum()),
            "n_rituximab_rows": int(ritux.sum()),
            "n_rituximab_no_order_start": int((ritux & ~start).sum()),
        }
    qc = root / "08_btris_full_source_inputs_qc.json"
    if qc.exists() and json.loads(qc.read_text()) != report:
        raise ValueError("existing_BTRIS_full_source_QC_changed_do_not_overwrite")
    if not qc.exists():
        qc.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.chmod(qc, 0o600)
    return paths, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eda-root", type=Path, default=Path("/vf/users/salazarda/data/eda_sjd"))
    p.add_argument("--cohort", type=Path, default=Path("data/raw/list_ids_longitudinal.csv"))
    p.add_argument("--audit-root", type=Path, default=Path("outputs/audit/08_treatments_full_source"))
    args = p.parse_args()
    paths, qc = prepare(args.eda_root, args.cohort, args.audit_root)
    print(json.dumps(qc, indent=2))
    print("Input Parquets:", {key: str(value) for key, value in paths.items()})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score PROs at every available patient visit.

This script does not select only baseline, does not impute ESSPRI with proxies,
and does not fit longitudinal models.  It preserves the repository's existing
visit-level scoring algorithms for ESSPRI, SF-36, PROFAD, and MDAFS.
"""
from __future__ import annotations
import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402
from src.derivations.pro_scoring import score_all_pros  # noqa: E402
from src.derivations.visit_dates import add_parsed_visit_dates, add_visit_timing  # noqa: E402

baseline_helpers = importlib.import_module("src.block_A.09_pros_baseline")
PATIENT_ID_COL = baseline_helpers.PATIENT_ID_COL
VISIT_DATE_COL = baseline_helpers.VISIT_DATE_COL
collapse_patient_visit_duplicates = baseline_helpers.collapse_patient_visit_duplicates
LOG = logging.getLogger("pros_longitudinal_scoring")
DEFAULT_INPUT = Path(getattr(common, "DEFAULT_ANALYTIC_DATASET"))
OUT = Path(common.INTERMEDIATE_DATA_DIR) / "09_pros_longitudinal_patient_visit"
QC_DIR = Path(common.BLOCKA_QC_DIR); TABLES_DIR = Path(common.BLOCKA_TABLES_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score PROs across all patient visits.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    return parser.parse_args()


def load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet": return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}: return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def add_spine_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Use canonical spine when present, otherwise derive compatible timing."""
    spine_path = Path(common.VISIT_SPINE_PARQUET)
    if spine_path.exists():
        cols = ["patient_id", "visit_date", "visit_id", "observed_baseline_date", "time_since_observed_baseline_days", "time_since_observed_baseline_years", "visit_number"]
        spine = pd.read_parquet(spine_path)[cols].copy(); spine["visit_date"] = pd.to_datetime(spine["visit_date"])
        result = df.merge(spine, on=["patient_id", "visit_date"], how="left", validate="one_to_one")
        if result["visit_id"].isna().any(): raise ValueError("PRO patient-visits are missing from canonical visit spine")
        return result
    return add_visit_timing(df)


def write(path: Path, writer, overwrite: bool) -> None:
    if path.exists() and not overwrite: raise FileExistsError(f"Output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True); writer(path)


def availability(df: pd.DataFrame) -> pd.DataFrame:
    measures = ["esspri_total", "sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global"]
    rows=[]
    for measure in measures:
        available = df[measure].notna() if measure in df else pd.Series(False, index=df.index)
        n=int(available.sum()); patients=int(df.loc[available, "patient_id"].nunique())
        rows.append({"measure": measure, "n_visits_available": n, "pct_visits_available": 100*n/len(df) if len(df) else np.nan, "n_patients_available": patients, "pct_patients_available": 100*patients/df.patient_id.nunique() if df.patient_id.nunique() else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    args=parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw=load(args.input)
    if PATIENT_ID_COL not in raw or VISIT_DATE_COL not in raw: raise ValueError(f"Input must include {PATIENT_ID_COL} and {VISIT_DATE_COL}")
    dated=add_parsed_visit_dates(raw, patient_id_col=PATIENT_ID_COL, visit_date_col=VISIT_DATE_COL)
    valid=dated[dated.patient_id.notna() & dated.visit_date.notna()].copy()
    collapsed, conflicts, metrics=collapse_patient_visit_duplicates(valid)
    collapsed["visit_date"] = pd.to_datetime(collapsed["visit_date"], errors="coerce")
    scored=score_all_pros(collapsed).sort_values(["patient_id", "visit_date"]).reset_index(drop=True)
    output=add_spine_timing(scored).sort_values(["patient_id", "visit_date"]).reset_index(drop=True)
    if output.duplicated(["patient_id", "visit_id"]).any(): raise ValueError("patient_id + visit_id must be unique")
    qc={"n_input_rows":len(raw), "n_unique_patients":int(valid.patient_id.nunique()), "n_patient_visits":len(output), "n_duplicate_patient_visit_rows":int(metrics["n_patient_dates_with_multiple_rows"]), "n_patient_visits_with_esspri":int(output.esspri_total.notna().sum()), "n_patient_visits_with_sf36_pcs":int(output.sf36_pcs.notna().sum()), "n_patient_visits_with_sf36_mcs":int(output.sf36_mcs.notna().sum()), "n_patient_visits_with_profad":int(output.profad_total.notna().sum()), "n_patient_visits_with_mdafs":int(output.mdafs_global.notna().sum()), "n_invalid_values":len(scored.attrs.get("pro_range_violations", []))}
    write(OUT.with_suffix(".parquet"), lambda p: output.to_parquet(p,index=False), args.overwrite)
    write(OUT.with_suffix(".csv"), lambda p: output.to_csv(p,index=False), args.overwrite)
    write(QC_DIR / "09_pros_longitudinal_qc.json", lambda p: p.write_text(json.dumps(qc, indent=2)), args.overwrite)
    write(TABLES_DIR / "09_pros_longitudinal_availability.csv", lambda p: availability(output).to_csv(p,index=False), args.overwrite)
    if not conflicts.empty: write(QC_DIR / "09_pros_longitudinal_duplicate_conflicts.csv", lambda p: conflicts.to_csv(p,index=False), args.overwrite)
    LOG.info("Wrote %d patient-visits", len(output))

if __name__ == "__main__": main()

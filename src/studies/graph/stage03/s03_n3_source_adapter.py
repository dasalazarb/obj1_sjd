"""Build an unapproved, internal N3 lab measurement candidate from frozen Step 01/10.

This is a provenance adapter, NOT instrument/assay verification or clinical
approval. It does not infer questionnaire/biopsy measurement dates from visit
anchors and never manufactures sample or assay identifiers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .._pipeline import digest

KEYS = ["patient_id", "clinical_episode_id"]
# Exact Step10 analytic feature -> documented Step01 lab_id and context date.
LAB_SOURCES = {
    "lab__ana_status__text": "ana_status",
    "lab__ana_hep2_status__text": "ana_hep2_status",
    "lab__rheumatoid_factor__text": "rheumatoid_factor",
    "lab__rheumatoid_factor__value": "rheumatoid_factor",
    "lab__urine_protein_qualitative__text": "urine_protein_qualitative",
}
OUTPUT_COLUMNS = KEYS + [
    "source_column", "value", "measurement_date", "assay_id",
    "unit", "sample_id", "source_lab_id", "source_stage",
    "date_provenance_status", "value_provenance_status",
    "assay_identity_status", "sample_identity_status",
    "clinical_approval_status",
]


def _unique(frame, keys, name):
    if not set(keys) <= set(frame) or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("N3_SOURCE_KEYS_INVALID_" + name)


def _dates(values):
    return pd.to_datetime(values, errors="coerce", utc=True)


def build_candidate(root, context_path, output_path):
    """Create a private, versioned-by-hash source candidate, not an approved assay."""
    root, context_path, output_path = map(Path, (root, context_path, output_path))
    analytic_path = (root / "data/analytic/10_build_integrated_longitudinal_dataset/"
                     "10_integrated_longitudinal_clinical_episode.parquet")
    analytes_path = (root / "data/intermediate/block_A/01_serological_profile/"
                     "01_labs_episode_analyte_level.parquet")
    if any(not p.is_file() for p in (analytic_path, context_path, analytes_path)):
        raise ValueError("N3_UPSTREAM_PARQUET_MISSING")
    if output_path.exists():
        raise FileExistsError("N3_CANDIDATE_EXISTS_DO_NOT_OVERWRITE")

    analytic = pd.read_parquet(analytic_path)
    if "is_clinical_baseline" not in analytic:
        raise ValueError("N3_OFFICIAL_BASELINE_FLAG_MISSING")
    baseline = analytic.loc[analytic.is_clinical_baseline.eq(True)].copy()
    _unique(baseline, KEYS, "BASELINE")
    if baseline.patient_id.duplicated().any():
        raise ValueError("N3_DUPLICATED_BASELINE_PATIENT")
    context = pd.read_parquet(context_path)
    _unique(context, KEYS, "CONTEXT")
    selected = pd.read_parquet(analytes_path)
    _unique(selected, KEYS + ["lab_id"], "SELECTED_ANALYTE")
    needed = {"selected_lab_date", "selected_unit", "unit_conflict", "result_conflict",
              "selection_status"}
    if not needed <= set(selected):
        raise ValueError("N3_SELECTED_LAB_PROVENANCE_INCOMPLETE")
    if not set(KEYS) <= set(analytic):
        raise ValueError("N3_ANALYTIC_KEYS_MISSING")
    baseline[KEYS] = baseline[KEYS].astype(str)
    context[KEYS] = context[KEYS].astype(str)
    selected[KEYS] = selected[KEYS].astype(str)
    base = baseline[KEYS + [s for s in LAB_SOURCES if s in baseline]].merge(
        context, on=KEYS, how="left", validate="one_to_one", indicator="_context_join")
    if not base._context_join.eq("both").all():
        raise ValueError("N3_CONTEXT_EPISODE_MISMATCH")

    tables = []
    report = {}
    for source, lab_id in LAB_SOURCES.items():
        date_field = "lab__" + lab_id + "__measurement_date"
        if source not in baseline or date_field not in context:
            report[source] = "SOURCE_OR_DATE_COLUMN_ABSENT"
            continue
        matching = selected.loc[selected.lab_id.astype("string").eq(lab_id),
                                KEYS + sorted(needed)].copy()
        if matching.empty:
            report[source] = "LAB_ID_NOT_PRESENT_IN_SELECTED_ANALYTES"
            continue
        matching = matching.rename(columns={"selected_lab_date": "_selected_date",
                                            "selected_unit": "_unit",
                                            "unit_conflict": "_unit_conflict",
                                            "result_conflict": "_result_conflict",
                                            "selection_status": "_selection_status"})
        m = base[KEYS + [source, date_field]].merge(
            matching, on=KEYS, how="left", validate="one_to_one")
        left, right = _dates(m[date_field]), _dates(m._selected_date)
        dates_agree = left.notna() & right.notna() & left.eq(right)
        conflicts = m._unit_conflict.eq(True).fillna(False) | m._result_conflict.eq(True).fillna(False)
        selected_ok = m._selection_status.isin(["selected_single", "selected_qualitative"])
        eligible_provenance = dates_agree & ~conflicts & selected_ok
        # Keep raw clinical values in the authorized environment only. Values
        # without verified dates are missing, never substituted by episode dates.
        raw = m[source].astype("string")
        values = raw.where(eligible_provenance, pd.NA)
        observed = values.notna()
        frame = m[KEYS].copy()
        frame["source_column"] = source
        frame["value"] = values
        frame["measurement_date"] = right.where(eligible_provenance & observed)
        # lab_id is a canonical analyte identifier, NOT proof of an assay platform.
        frame["assay_id"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        frame["unit"] = m._unit.astype("string").where(eligible_provenance & observed)
        # No specimen/sample key survives the Step01 selected-analyte contract.
        frame["sample_id"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        frame["source_lab_id"] = lab_id
        frame["source_stage"] = "step01_selected_analyte_and_step10_baseline_context"
        frame["date_provenance_status"] = "STEP01_STEP10_CONCORDANT_WHEN_PRESENT"
        frame.loc[~dates_agree, "date_provenance_status"] = "MISSING_OR_DISCORDANT"
        frame["value_provenance_status"] = "UNVERIFIED_CONSTRUCT"
        frame.loc[~selected_ok, "value_provenance_status"] = "SELECTION_RULE_NOT_SUPPORTED"
        frame.loc[conflicts, "value_provenance_status"] = "CONFLICT_REQUIRES_REVIEW"
        frame["assay_identity_status"] = "NOT_VERIFIED"
        frame["sample_identity_status"] = "NOT_AVAILABLE"
        frame["clinical_approval_status"] = "PENDING_HUMAN_REVIEW"
        tables.append(frame[OUTPUT_COLUMNS])
        report[source] = "DATED_LAB_CANDIDATES_UNAPPROVED"
    if not tables:
        raise ValueError("N3_NO_SOURCE_MAPPED_LAB_CANDIDATES")
    output = pd.concat(tables, ignore_index=True)
    # No duplicate source / episode keys; RF text and value remain distinct.
    _unique(output, KEYS + ["source_column"], "RESULT")
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = output_path.with_suffix(".tmp.parquet")
    if temp.exists():
        raise FileExistsError("N3_TEMP_OUTPUT_EXISTS")
    try:
        output.to_parquet(temp, index=False)
        os.chmod(temp, 0o600)
        temp.rename(output_path)
        manifest = {
            "status": "SOURCE_CANDIDATE_UNAPPROVED",
            "privacy_class": "INTERNAL_RESTRICTED",
            "public_export": False,
            "eligible_for_inference": False,
            "sample_id_real": False,
            "assay_id_verified": False,
            "dates_reused_from_episode_anchors": False,
            "sources": report,
            "source_sha256": {
                "analytic": digest(analytic_path),
                "context": digest(context_path),
                "analytes": digest(analytes_path),
                "candidate": digest(output_path),
            },
            "excluded_endpoints": {
                "essdai__total": "ORIGINAL_INSTRUMENT_DATE_NOT_VERIFIED",
                "esspri__total": "ORIGINAL_INSTRUMENT_DATE_NOT_VERIFIED",
                "ext__biopsy_focus_score": "ORIGINAL_BIOPSY_DATE_NOT_VERIFIED",
            },
        }
        metadata_path = output_path.with_suffix(".provenance.json")
        metadata_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        metadata_path.chmod(0o600)
    except BaseException:
        if temp.exists():
            temp.unlink()
        if output_path.exists():
            output_path.unlink()
        raise
    return {"status": manifest["status"], "path": str(output_path),
            "manifest": str(metadata_path), "estimators_run": 0}

#!/usr/bin/env python3
"""Build the episode-aligned extended clinical phenotype layer.

The clinical visit spine is both the row authority and the source of the raw,
already episode-assigned CTDB fields.  No date matching, filling, imputation,
or baseline selection is performed here.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import common  # noqa: E402

KEYS = ["patient_id", "clinical_episode_id"]
STRUCTURAL_COLUMNS = [
    "patient_id", "clinical_episode_id", "clinical_anchor_date",
    "clinical_visit_number", "clinical_visit", "visit_type",
    "episode_start_date", "episode_end_date", "clinical_baseline_episode_id",
    "clinical_baseline_date", "is_clinical_baseline",
    "time_since_clinical_baseline_days", "time_since_clinical_baseline_years",
]

# 2016 ACR/EULAR criterion: unstimulated whole saliva flow <= 0.1 mL/min.
# The input fields must therefore represent a rate in mL/min (not an unscaled
# collection volume); questionable source units are retained in the audit.
LOW_WUSF_THRESHOLD_ML_PER_MIN = 0.1
# Schirmer <=5 mm/5 min in either eye is the classification threshold.
SCHIRMER_ABNORMAL_THRESHOLD_MM_PER_5_MIN = 5.0

PATHOLOGY_COLUMNS = {"biopsy_focus_score": "biopsy_pathology__f_score"}
SALIVARY_FLOW_COLUMNS = {
    "unstimulated": [
        "salivary_flow_form__flow_whole_unstim",
        "salivary_flow_form__tot_unsim_sal_flow",
        "wus_only__flow_whole_unstim",
    ],
    "stimulated": [
        "salivary_flow_form__flow_whole_stim",
        "salivary_flow_form__tot_sim_sal_flow",
    ],
}
OCULAR_COLUMNS = {
    "ocular_schirmer_right": "eye_examination__sch_r",
    "ocular_schirmer_left": "eye_examination__sch_l",
    "ocular_tbut_right": "eye_examination__tbut_r",
    "ocular_tbut_left": "eye_examination__tbut_l",
    "ocular_van_bijsterveld_right": "eye_examination__vanb_r",
    "ocular_van_bijsterveld_left": "eye_examination__vanb_l",
    "ocular_oxford_right": "eye_examination__oxford_r",
    "ocular_oxford_left": "eye_examination__oxford_l",
}
SICCA_COLUMNS = {
    "sicca_any_symptom": "visit_summary_-_2016_classification_criteria__ic_symptom_dry_eye_or_dry_mouth",
    "sicca_dry_eye_3month": "visit_summary_-_2016_classification_criteria__ic_dry_eye_3month",
    "sicca_sand_gravel_eye": "visit_summary_-_2016_classification_criteria__ic_sand_gravel_eye",
    "sicca_tear_substitute": "visit_summary_-_2016_classification_criteria__ic_tear_subsit",
    "sicca_dry_mouth_3month": "visit_summary_-_2016_classification_criteria__ic_dry_mouth_3month",
    "sicca_difficulty_swallowing_dry_food": "visit_summary_-_2016_classification_criteria__ic_difficulty_swallowing_dry_food",
}
AGGREGATE_BOOLEAN_COLUMNS = {
    "ocular_staining_positive": "visit_summary_-_2016_classification_criteria__ocular_stain",
    "lacrimal_dysfunction": "visit_summary_-_2016_classification_criteria__lacrimal_dysfunction",
}
SGUS_COLUMNS = {
    "sgus_theander_final_score_grade": "sg-us_grading_scale:_theander_&_mandl_(2014)__sg_us_final_score_grade",
    "sgus_omeract_r_parotid_grade": "sgus_grading_scale:_omeract_(2021)__r_parotid_grade",
    "sgus_omeract_l_parotid_grade": "sgus_grading_scale:_omeract_(2021)__l_parotid_grade",
    "sgus_omeract_r_submandibular_grade": "sgus_grading_scale:_omeract_(2021)__r_submandibular_grade",
    "sgus_omeract_l_submandibular_grade": "sgus_grading_scale:_omeract_(2021)__l_submandibular_grade",
}
SJDDI_COLUMNS = {
    "sjddi_neuro_cranial_peripheral": "neuro_cran_periph", "sjddi_lymphoma": "lymphoma",
    "sjddi_nephrocalcinosis": "ct_nephrocalcinosis",
    "sjddi_pulmonary_pleural_fibrosis": "pulm_pleural_fibrosis",
    "sjddi_eye_abnormality": "eyes_abnormality", "sjddi_cns": "cns",
    "sjddi_salivary_flow_impairment": "salivary_flow_impair", "sjddi_teeth_loss": "teeth_loss",
    "sjddi_tear_flow_impairment": "tear_flow_impair",
    "sjddi_interstitial_fibrosis": "fibrosis_interstitial",
    "sjddi_pulmonary_function_damage": "pulm_functn_damage",
    "sjddi_reduced_gfr_creatinine": "gfr_creatinin_reduced",
    "sjddi_tubular_acidosis": "tubular_acidosis",
}
SJDDI_PREFIX = "sjogren's_syndrome_disease_damage_index__"
NULL_TOKENS = {"", "na", "n/a", "nan", "none", "null", "not done", "not_done", "unknown", "unk"}
TRUE_TOKENS = {"yes", "y", "true", "1", "positive", "present", "abnormal"}
FALSE_TOKENS = {"no", "n", "false", "0", "negative", "absent", "normal"}


def read_spine(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, low_memory=False)


def validate_spine(spine: pd.DataFrame) -> None:
    missing = [c for c in STRUCTURAL_COLUMNS if c not in spine]
    if missing:
        raise AssertionError(f"clinical spine missing structural columns: {missing}")
    if spine[KEYS].isna().any().any() or spine.duplicated(KEYS).any():
        raise AssertionError("clinical spine keys must be nonmissing and unique")


def _raw(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column] if column in frame else pd.Series(pd.NA, index=frame.index, dtype="object")


def _numeric(frame: pd.DataFrame, source: str, canonical: str, audit: list[dict],
             minimum: float | None = 0, maximum: float | None = None) -> pd.Series:
    raw = _raw(frame, source)
    text = raw.astype("string").str.strip()
    null = raw.isna() | text.str.lower().isin(NULL_TOKENS)
    value = pd.to_numeric(text.mask(null), errors="coerce").astype("Float64")
    invalid = ~null & value.isna()
    out_range = value.notna() & ((value < minimum) if minimum is not None else False)
    if maximum is not None:
        out_range |= value.notna() & value.gt(maximum)
    for idx in frame.index[invalid | out_range]:
        audit.append({**{k: frame.at[idx, k] for k in KEYS}, "canonical_variable": canonical,
                      "source_variable": source, "raw_value": raw.at[idx],
                      "issue": "non_convertible" if invalid.at[idx] else "outside_expected_range"})
    return value.mask(out_range)


def _boolean(frame: pd.DataFrame, source: str, canonical: str, audit: list[dict]) -> pd.Series:
    raw = _raw(frame, source)
    text = raw.astype("string").str.strip().str.lower()
    answer = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    answer.loc[text.isin(TRUE_TOKENS)] = True
    answer.loc[text.isin(FALSE_TOKENS)] = False
    unexpected = raw.notna() & ~text.isin(TRUE_TOKENS | FALSE_TOKENS | NULL_TOKENS)
    for idx in frame.index[unexpected]:
        audit.append({**{k: frame.at[idx, k] for k in KEYS}, "canonical_variable": canonical,
                      "source_variable": source, "raw_value": raw.at[idx], "issue": "unexpected_boolean_token"})
    return answer


def _any_nullable(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].astype("boolean")
    result = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    result.loc[values.eq(True).any(axis=1)] = True
    result.loc[values.notna().all(axis=1) & values.eq(False).all(axis=1)] = False
    return result


def derive_pathology(source: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(index=source.index)
    raw_name = PATHOLOGY_COLUMNS["biopsy_focus_score"]
    out["biopsy_focus_score_raw"] = _raw(source, raw_name).astype("string")
    out["biopsy_focus_score"] = _numeric(source, raw_name, "biopsy_focus_score", audit)
    out["biopsy_evaluable"] = out.biopsy_focus_score.notna().astype("boolean")
    out["biopsy_focus_score_ge1"] = out.biopsy_focus_score.ge(1).astype("boolean").mask(out.biopsy_focus_score.isna())
    return out


def _coalesce_measurements(source: pd.DataFrame, candidates: list[str], stem: str,
                           audit: list[dict], conflicts: list[dict]) -> pd.DataFrame:
    existing = [c for c in candidates if c in source]
    parsed = pd.DataFrame({c: _numeric(source, c, stem, audit) for c in existing}, index=source.index)
    out = pd.DataFrame(index=source.index)
    if not existing:
        out[stem] = pd.Series(pd.NA, index=source.index, dtype="Float64")
        out[f"{stem}_source"] = pd.Series(pd.NA, index=source.index, dtype="string")
        out[f"{stem}_conflict"] = pd.Series(False, index=source.index, dtype="boolean")
    else:
        n_unique = parsed.nunique(axis=1, dropna=True)
        conflict = n_unique.gt(1)
        out[stem] = parsed.bfill(axis=1).iloc[:, 0].astype("Float64").mask(conflict)
        out[f"{stem}_source"] = parsed.apply(
            lambda row: "|".join(row.index[row.notna()].tolist()) or pd.NA, axis=1).astype("string")
        out[f"{stem}_conflict"] = conflict.astype("boolean")
        for idx in source.index[conflict]:
            record = {**{k: source.at[idx, k] for k in KEYS}, "canonical_variable": stem,
                      "candidate_values": ";".join(f"{c}={parsed.at[idx, c]}" for c in existing if pd.notna(parsed.at[idx, c]))}
            conflicts.append(record)
            audit.append({**record, "source_variable": out.at[idx, f"{stem}_source"],
                          "raw_value": record["candidate_values"], "issue": "discordant_sources"})
    out[f"{stem}_evaluable"] = out[stem].notna().astype("boolean")
    return out


def derive_salivary_function(source: pd.DataFrame, audit: list[dict], conflicts: list[dict]) -> pd.DataFrame:
    unstim = _coalesce_measurements(source, SALIVARY_FLOW_COLUMNS["unstimulated"],
                                    "salivary_flow_unstimulated", audit, conflicts)
    stim = _coalesce_measurements(source, SALIVARY_FLOW_COLUMNS["stimulated"],
                                  "salivary_flow_stimulated", audit, conflicts)
    out = pd.concat([unstim, stim], axis=1)
    value = out["salivary_flow_unstimulated"]
    out["low_unstimulated_salivary_flow"] = value.le(LOW_WUSF_THRESHOLD_ML_PER_MIN).astype("boolean").mask(value.isna())
    return out


def derive_ocular_phenotype(source: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(index=source.index)
    ranges = {"schirmer": (0, None), "tbut": (0, None), "van_bijsterveld": (0, 9), "oxford": (0, 5)}
    for canonical, raw in OCULAR_COLUMNS.items():
        scale = next(k for k in ranges if k in canonical)
        out[canonical] = _numeric(source, raw, canonical, audit, *ranges[scale])
    for scale, reducer in (("schirmer", "min"), ("tbut", "min"), ("van_bijsterveld", "max"), ("oxford", "max")):
        sides = [f"ocular_{scale}_right", f"ocular_{scale}_left"]
        out[f"ocular_{scale}_{reducer}"] = getattr(out[sides], reducer)(axis=1, skipna=True).astype("Float64")
        out[f"ocular_{scale}_evaluable"] = out[sides].notna().any(axis=1).astype("boolean")
    observed = out[["ocular_schirmer_right", "ocular_schirmer_left"]]
    out["ocular_schirmer_abnormal_any"] = observed.le(SCHIRMER_ABNORMAL_THRESHOLD_MM_PER_5_MIN).any(axis=1).astype("boolean")
    out.loc[~observed.notna().any(axis=1), "ocular_schirmer_abnormal_any"] = pd.NA
    for canonical, raw in AGGREGATE_BOOLEAN_COLUMNS.items():
        out[canonical] = _boolean(source, raw, canonical, audit)
    return out


def derive_sicca_phenotype(source: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(index=source.index)
    for canonical, raw in SICCA_COLUMNS.items():
        out[canonical] = _boolean(source, raw, canonical, audit)
    out["sicca_ocular_symptom_active"] = _any_nullable(out, ["sicca_dry_eye_3month", "sicca_sand_gravel_eye", "sicca_tear_substitute"])
    out["sicca_oral_symptom_active"] = _any_nullable(out, ["sicca_dry_mouth_3month", "sicca_difficulty_swallowing_dry_food"])
    return out


def _safe_suffix(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def derive_sgus(source: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    mapping = dict(SGUS_COLUMNS)
    for raw in source:
        low = raw.lower()
        suffix = raw.split("__", 1)[-1]
        if "sg" in low and "jousse" in low:
            mapping.setdefault(f"sgus_jousse_joulin_{_safe_suffix(suffix)}", raw)
        elif "sg" in low and "theander" in low:
            mapping.setdefault(f"sgus_theander_{_safe_suffix(suffix)}", raw)
        elif "sgus" in low and "omeract" in low:
            mapping.setdefault(f"sgus_omeract_{_safe_suffix(suffix)}", raw)
    out = pd.DataFrame(index=source.index)
    for canonical, raw in mapping.items():
        out[canonical] = _numeric(source, raw, canonical, audit, minimum=0)
    systems = {
        "theander": [c for c in out if c.startswith("sgus_theander_")],
        "jousse_joulin": [c for c in out if c.startswith("sgus_jousse_joulin_")],
        "omeract": [c for c in out if c.startswith("sgus_omeract_")],
    }
    for name, columns in systems.items():
        out[f"sgus_{name}_available"] = (out[columns].notna().any(axis=1) if columns else False)
        out[f"sgus_{name}_available"] = out[f"sgus_{name}_available"].astype("boolean")
    out["sgus_available"] = out[[f"sgus_{x}_available" for x in systems]].any(axis=1).astype("boolean")
    return out


def derive_sjddi_components(source: pd.DataFrame, audit: list[dict]) -> pd.DataFrame:
    mapping = {canonical: SJDDI_PREFIX + suffix for canonical, suffix in SJDDI_COLUMNS.items()}
    known_raw = set(mapping.values())
    # Preserve additional genuine index components without pretending to score them.
    for raw in source:
        if raw.startswith(SJDDI_PREFIX) and raw not in known_raw:
            mapping.setdefault("sjddi_" + _safe_suffix(raw[len(SJDDI_PREFIX):]), raw)
    out = pd.DataFrame(index=source.index)
    for canonical, raw in mapping.items():
        out[canonical] = _boolean(source, raw, canonical, audit)
    return out


def build_extended_clinical_longitudinal(spine: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    validate_spine(spine)
    audit: list[dict] = []
    conflicts: list[dict] = []
    blocks = [derive_pathology(spine, audit), derive_salivary_function(spine, audit, conflicts),
              derive_ocular_phenotype(spine, audit), derive_sicca_phenotype(spine, audit),
              derive_sgus(spine, audit), derive_sjddi_components(spine, audit)]
    output = pd.concat([spine[STRUCTURAL_COLUMNS].reset_index(drop=True)] +
                       [block.reset_index(drop=True) for block in blocks], axis=1)
    validate_output_contract(spine, output)
    return output, audit, conflicts


def validate_output_contract(spine: pd.DataFrame, output: pd.DataFrame) -> None:
    if len(output) != len(spine) or output.duplicated(KEYS).any():
        raise AssertionError("extended clinical output does not preserve spine rows/unique keys")
    left = set(map(tuple, spine[KEYS].astype("string").itertuples(index=False, name=None)))
    right = set(map(tuple, output[KEYS].astype("string").itertuples(index=False, name=None)))
    if left != right:
        raise AssertionError("extended clinical output has missing or extra spine keys")
    for column in STRUCTURAL_COLUMNS:
        a, b = spine[column].reset_index(drop=True), output[column]
        if not (a.eq(b) | (a.isna() & b.isna())).all():
            raise AssertionError(f"spine structural column changed: {column}")


def build_source_manifest(columns: Iterable[str] | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    direct = {**PATHOLOGY_COLUMNS, **OCULAR_COLUMNS, **SICCA_COLUMNS, **AGGREGATE_BOOLEAN_COLUMNS, **SGUS_COLUMNS,
              **{k: SJDDI_PREFIX + v for k, v in SJDDI_COLUMNS.items()}}
    for canonical, source in direct.items():
        domain = canonical.split("_", 1)[0]
        rows.append({"canonical_variable": canonical, "clinical_domain": domain,
                     "source_variable": source, "source_form": source.split("__", 1)[0],
                     "source_protocol": "episode-assigned upstream", "derivation": "numeric parse" if domain in {"biopsy", "ocular", "sgus"} else "explicit nullable boolean normalization",
                     "expected_type": "Float64" if domain in {"biopsy", "ocular", "sgus"} else "boolean",
                     "valid_range_or_categories": "documented scale/non-negative" if domain in {"biopsy", "ocular", "sgus"} else "True/False/NA",
                     "time_behavior": "observed in episode only; never filled", "longitudinal_eligible": True,
                     "notes": "Raw source remains episode-specific."})
    for kind, sources in SALIVARY_FLOW_COLUMNS.items():
        for source in sources:
            rows.append({"canonical_variable": f"salivary_flow_{kind}", "clinical_domain": "salivary",
                         "source_variable": source, "source_form": source.split("__", 1)[0],
                         "source_protocol": "episode-assigned upstream", "derivation": "coalesce only concordant numeric candidates; discordance => NA + conflict",
                         "expected_type": "Float64", "valid_range_or_categories": ">=0 mL/min",
                         "time_behavior": "observed in episode only; never filled", "longitudinal_eligible": True,
                         "notes": "Source column(s) recorded in provenance feature."})
    manifest = pd.DataFrame(rows).drop_duplicates(["canonical_variable", "source_variable"])
    if columns is not None:
        requested = set(columns) - set(STRUCTURAL_COLUMNS)
        represented = set(manifest.canonical_variable)
        for canonical in sorted(requested - represented):
            rows.append({"canonical_variable": canonical, "clinical_domain": canonical.split("_", 1)[0],
                         "source_variable": "derived from documented episode-level inputs",
                         "source_form": "derived", "source_protocol": "episode-assigned upstream",
                         "derivation": "see code constants and derivation function", "expected_type": "see dictionary",
                         "valid_range_or_categories": "see dictionary", "time_behavior": "episode-specific; never filled",
                         "longitudinal_eligible": True, "notes": "No cross-episode propagation."})
        manifest = pd.DataFrame(rows).drop_duplicates(["canonical_variable", "source_variable"])
        manifest = manifest.loc[manifest.canonical_variable.isin(requested)]
    return manifest


def build_missingness_qc(output: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in output if c not in STRUCTURAL_COLUMNS]
    baseline = output["is_clinical_baseline"].eq(True)
    n_patients = output.patient_id.nunique()
    rows = []
    for c in features:
        patients = output.loc[output[c].notna(), "patient_id"].nunique()
        rows.append({"variable": c, "n_rows_total": len(output), "n_nonmissing": int(output[c].notna().sum()),
                     "pct_nonmissing": 100 * output[c].notna().mean(), "n_baseline_rows": int(baseline.sum()),
                     "n_nonmissing_baseline": int(output.loc[baseline, c].notna().sum()),
                     "pct_nonmissing_baseline": 100 * output.loc[baseline, c].notna().mean() if baseline.any() else np.nan,
                     "n_patients_with_any_measurement": patients,
                     "pct_patients_with_any_measurement": 100 * patients / n_patients if n_patients else np.nan})
    return pd.DataFrame(rows)


def build_structural_qc(spine: pd.DataFrame, output: pd.DataFrame) -> pd.DataFrame:
    sk = set(map(tuple, spine[KEYS].itertuples(index=False, name=None)))
    ok = set(map(tuple, output[KEYS].itertuples(index=False, name=None)))
    baseline = output.is_clinical_baseline.fillna(False).astype(bool)
    episode_mismatch = baseline & output.clinical_episode_id.astype("string").ne(output.clinical_baseline_episode_id.astype("string"))
    dates_equal = pd.to_datetime(output.clinical_anchor_date, errors="coerce").eq(pd.to_datetime(output.clinical_baseline_date, errors="coerce"))
    metrics = {
        "n_rows_input_spine": len(spine), "n_rows_output": len(output),
        "n_patients_input": spine.patient_id.nunique(), "n_patients_output": output.patient_id.nunique(),
        "n_unique_patient_episode_input": len(sk), "n_unique_patient_episode_output": len(ok),
        "n_duplicate_keys": output.duplicated(KEYS).sum(), "n_missing_from_output": len(sk-ok), "n_extra_output_keys": len(ok-sk),
        "n_clinical_baselines": baseline.sum(),
        "n_patients_with_multiple_baselines": output.loc[baseline].groupby("patient_id").size().gt(1).sum(),
        "n_baseline_episode_mismatches": episode_mismatch.sum(),
        "n_baseline_date_mismatches": (baseline & ~dates_equal).sum(),
    }
    return pd.DataFrame({"metric": metrics.keys(), "value": metrics.values()})


def build_dictionary(output: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    lookup = manifest.groupby("canonical_variable").agg(source=("source_variable", lambda x: "|".join(x)), derivation=("derivation", "first"))
    rows = []
    for c in output:
        domain = "structure" if c in STRUCTURAL_COLUMNS else c.split("_", 1)[0]
        rows.append({"variable": c, "domain": domain, "description": c.replace("_", " "),
                     "dtype": str(output[c].dtype), "source": lookup.at[c, "source"] if c in lookup.index else "clinical visit spine/derived",
                     "derivation": lookup.at[c, "derivation"] if c in lookup.index else "copied or explicitly derived",
                     "valid_values": "see source manifest", "temporal_interpretation": "episode-specific; no filling"})
    return pd.DataFrame(rows)


def write_outputs(spine: pd.DataFrame, output: pd.DataFrame, audit: list[dict], conflicts: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(path, index=False)
    output.to_csv(path.with_suffix(".csv"), index=False)
    qc_dir = common.BLOCKA_QC_DIR / "01_extended_clinical_phenotype"
    table_dir = common.BLOCKA_TABLES_DIR / "01_extended_clinical_phenotype"
    qc_dir.mkdir(parents=True, exist_ok=True); table_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_source_manifest(output.columns)
    build_structural_qc(spine, output).to_csv(qc_dir / "01_extended_clinical_phenotype_qc.csv", index=False)
    build_missingness_qc(output).to_csv(qc_dir / "01_extended_clinical_phenotype_missingness.csv", index=False)
    manifest.to_csv(qc_dir / "01_extended_clinical_phenotype_source_manifest.csv", index=False)
    audit_columns = [*KEYS, "canonical_variable", "source_variable", "raw_value", "issue"]
    pd.DataFrame(audit).reindex(columns=audit_columns).to_csv(qc_dir / "01_extended_clinical_phenotype_value_audit.csv", index=False)
    conflict_columns = [*KEYS, "canonical_variable", "candidate_values"]
    pd.DataFrame(conflicts).reindex(columns=conflict_columns).to_csv(qc_dir / "01_extended_clinical_phenotype_conflicts.csv", index=False)
    build_dictionary(output, manifest).to_csv(table_dir / "01_extended_clinical_phenotype_dictionary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spine", type=Path, default=common.CLINICAL_VISIT_SPINE_PARQUET)
    parser.add_argument("--output", type=Path, default=common.EXTENDED_CLINICAL_LONGITUDINAL_PARQUET)
    args = parser.parse_args()
    spine = read_spine(args.spine)
    output, audit, conflicts = build_extended_clinical_longitudinal(spine)
    write_outputs(spine, output, audit, conflicts, args.output)
    print(f"Wrote {len(output):,} clinical episodes to {args.output}")


if __name__ == "__main__":
    main()

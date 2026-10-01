#!/usr/bin/env python3
"""Curate validated episode products into the longitudinal data contract.

Step 10 is deliberately the only integration policy boundary.  Upstream products
remain the source of detailed QC and provenance; this module classifies every
observed column and emits a small analytic master, scientific context, registry,
and compact audit products.  It never reconstructs clinical facts.
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402

KEYS = ["patient_id", "clinical_episode_id"]
STRUCTURAL_COLUMNS = [
    "patient_id", "clinical_episode_id", "clinical_anchor_date",
    "clinical_visit_number", "clinical_visit", "visit_type",
    "episode_start_date", "episode_end_date", "clinical_baseline_episode_id",
    "clinical_baseline_date", "is_clinical_baseline",
    "time_since_clinical_baseline_days", "time_since_clinical_baseline_years",
]
DATE_COLUMNS = {
    "clinical_anchor_date", "episode_start_date", "episode_end_date",
    "clinical_baseline_date", "visit_date", "visit_date_clean", "event_date", "baseline_date",
    "observed_baseline_date", "row_date_original", "row_date_min", "row_date_max",
}
ROLES = {
    "STRUCTURAL", "ANALYTIC", "CONTEXT", "QC", "PROVENANCE",
    "REGISTRY_METADATA", "DEPRECATED_ALIAS", "LEGACY",
    "DOWNSTREAM_DERIVED", "UNCLASSIFIED",
}
ROLE_DESTINATION = {
    "STRUCTURAL": "analytic", "ANALYTIC": "analytic", "CONTEXT": "context",
    "QC": "none", "PROVENANCE": "none", "REGISTRY_METADATA": "registry",
    "DEPRECATED_ALIAS": "none", "LEGACY": "none",
    "DOWNSTREAM_DERIVED": "downstream", "UNCLASSIFIED": "none",
}
DATASET_CONTRACT_VERSION = "clinical_episode_curated_v1"
INTEGRATION_BUILD_VERSION = "v4_central_curation"

KNOWN_ALIASES = {
    "esspri_dryness_observed": "esspri_dryness",
    "esspri_fatigue_observed": "esspri_fatigue",
    "esspri_pain_observed": "esspri_pain",
    "esspri_total_observed": "esspri_total",
    "visit_id": "clinical_episode_id", "visit_date": "clinical_anchor_date",
    "visit_date_clean": "clinical_anchor_date",
    "event_date": "clinical_anchor_date", "visit_number": "clinical_visit_number",
    "baseline_date": "clinical_baseline_date",
    "observed_baseline_date": "clinical_baseline_date",
    "time_since_observed_baseline_days": "time_since_clinical_baseline_days",
    "time_since_observed_baseline_years": "time_since_clinical_baseline_years",
    "time_since_baseline_days": "time_since_clinical_baseline_days",
    "time_since_baseline_years": "time_since_clinical_baseline_years",
    "time_years": "time_since_clinical_baseline_years",
    "row_date_original": "clinical_anchor_date", "row_date_min": "clinical_anchor_date",
    "row_date_max": "clinical_anchor_date",
    "constitutional": "eg_constitutional_active",
    "lymphadenopathy": "eg_lymphadenopathy_active",
    "articular": "eg_articular_active", "cutaneous": "eg_cutaneous_active",
    "pulmonary": "eg_pulmonary_active", "renal": "eg_renal_active",
    "muscular": "eg_muscular_active", "PNS": "eg_pns_active",
    "CNS": "eg_cns_active", "hematologic": "eg_hematologic_active",
    "biological": "eg_biological_active",
}
DOWNSTREAM_DERIVED = {
    "n_clinical_visits_patient", "is_last_clinical_visit",
    "previous_clinical_episode_id", "previous_clinical_anchor_date",
    "time_from_previous_clinical_episode_days",
    "time_from_previous_clinical_episode_years", "previous_pop_status",
    "previous_essdai_total", "previous_esspri_total", "previous_overlap_status",
    "previous_extraglandular_active", "previous_sf36_pcs", "previous_sf36_mcs",
    "previous_profad_total", "previous_mdafs_global",
    "delta_essdai_from_previous", "delta_esspri_from_previous",
    "delta_sf36_pcs_from_previous", "delta_sf36_mcs_from_previous",
    "delta_profad_from_previous", "delta_mdafs_from_previous",
    "pop_baseline_episode_id", "pop_baseline_date", "pop_baseline_status",
    "is_pop_baseline", "time_since_pop_baseline_days",
    "time_since_pop_baseline_years",
}
EXTENDED_CLINICAL_PRIMARY_FEATURES = [
    "biopsy_focus_score", "salivary_flow_unstimulated", "ocular_schirmer_min",
    "ocular_staining_positive", "sicca_any_symptom", "sgus_available",
]


class CurationContractError(AssertionError):
    """Raised when a new or invalid upstream variable violates the contract."""


@dataclass(frozen=True)
class ColumnClassification:
    variable: str
    source: str
    family: str
    role: str
    destination: str
    canonical_variable: str | None = None
    alias_of: str | None = None
    classification_rule: str = ""
    classification_reason: str = ""
    present_in_input: bool = True
    included_in_analytic: bool = False
    included_in_context: bool = False
    dtype: str = ""
    n_nonmissing: int = 0
    pct_nonmissing: float = 0.0
    n_unique: int = 0
    paired_variable: str | None = None
    notes: str = ""


@dataclass
class CuratedBuild:
    analytic: pd.DataFrame
    context: pd.DataFrame
    coverage: pd.DataFrame
    registry: pd.DataFrame
    source_summaries: list[dict]
    key_mismatches: list[dict]
    structural_discrepancies: list[dict]


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise AssertionError(f"{source} is missing required columns: {missing}")


def _normalise_structure(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in DATE_COLUMNS & set(frame.columns):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ["patient_id", "clinical_episode_id", "clinical_baseline_episode_id"]:
        if column in frame:
            frame[column] = frame[column].astype("string")
    return frame


def _different(left: pd.Series, right: pd.Series) -> pd.Series:
    """Return null-safe semantic inequality (including aligned date values)."""
    return left.isna().ne(right.isna()) | (
        left.notna() & right.notna() & ~left.eq(right)
    )


def validate_spine(spine: pd.DataFrame) -> pd.DataFrame:
    require_columns(spine, STRUCTURAL_COLUMNS, "clinical_spine")
    spine = _normalise_structure(spine)
    if spine.duplicated(KEYS).any():
        raise AssertionError("clinical_spine has duplicate patient_id + clinical_episode_id keys")
    if not spine["clinical_visit"].fillna(False).astype(bool).all():
        raise AssertionError("clinical_spine contains non-clinical visits")
    if spine["clinical_visit_number"].isna().any() or spine["clinical_visit_number"].lt(1).any():
        raise AssertionError("clinical_visit_number must be >= 1")
    ordered = spine.sort_values(
        ["patient_id", "clinical_anchor_date", "clinical_visit_number", "clinical_episode_id"]
    )
    groups = ordered.groupby("patient_id", sort=False)
    if groups["clinical_visit_number"].diff().dropna().le(0).any():
        raise AssertionError("clinical_visit_number must be strictly increasing within patient")
    if any(not group["clinical_anchor_date"].is_monotonic_increasing for _, group in groups):
        raise AssertionError("clinical_anchor_date must increase monotonically within patient")
    baseline = spine["is_clinical_baseline"].fillna(False).astype(bool)
    if baseline.groupby(spine["patient_id"]).sum().gt(1).any():
        raise AssertionError("a patient has more than one clinical baseline")
    if (spine.loc[baseline, "clinical_episode_id"] != spine.loc[baseline, "clinical_baseline_episode_id"]).any():
        raise AssertionError("clinical baseline episode identity is inconsistent")
    if _different(spine.loc[baseline, "clinical_anchor_date"], spine.loc[baseline, "clinical_baseline_date"]).any():
        raise AssertionError("clinical baseline date is inconsistent")
    if spine.loc[spine["clinical_baseline_episode_id"].notna(), "time_since_clinical_baseline_days"].dropna().lt(0).any():
        raise AssertionError("negative time since clinical baseline")
    return spine


def validate_source(spine: pd.DataFrame, source: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict, list[dict], list[dict]]:
    """Validate a complete one-to-one episode product without curating columns."""
    source = _normalise_structure(source)
    require_columns(source, KEYS, name)
    duplicates = int(source.duplicated(KEYS).sum())
    spine_keys = set(map(tuple, spine[KEYS].itertuples(index=False, name=None)))
    source_keys = set(map(tuple, source[KEYS].itertuples(index=False, name=None)))
    missing, extra = spine_keys - source_keys, source_keys - spine_keys
    mismatches = (
        [{"source": name, **dict(zip(KEYS, key)), "mismatch": "missing_from_source"} for key in sorted(missing)]
        + [{"source": name, **dict(zip(KEYS, key)), "mismatch": "extra_in_source"} for key in sorted(extra)]
    )
    discrepancies: list[dict] = []
    shared = [c for c in STRUCTURAL_COLUMNS if c not in KEYS and c in source]
    compared = spine[KEYS + shared].merge(source[KEYS + shared], on=KEYS, suffixes=("_spine", "_source"))
    for column in shared:
        mask = _different(compared[f"{column}_spine"], compared[f"{column}_source"])
        for row in compared.loc[mask, KEYS + [f"{column}_spine", f"{column}_source"]].itertuples(index=False, name=None):
            discrepancies.append({"source": name, **dict(zip(KEYS, row[:2])), "variable": column,
                                  "spine_value": row[2], "source_value": row[3]})
    inherited = sorted((set(source.columns) & set(spine.columns)) - set(KEYS))
    summary = {
        "source": name, "n_rows": len(source), "n_columns": len(source.columns),
        "n_patients": source.patient_id.nunique(),
        "n_unique_patient_episode": len(source_keys), "n_duplicates": duplicates,
        "n_missing_spine_keys": len(missing), "n_extra_keys": len(extra),
        "structural_discrepancies": len(discrepancies),
        "n_inherited_spine_columns": len(inherited),
    }
    if duplicates or missing or extra or discrepancies:
        raise AssertionError(f"{name} violates clinical-spine contract: {summary}")
    return source, summary, mismatches, discrepancies


def _legacy(name: str) -> bool:
    lower = name.lower()
    return ("proxy" in lower or lower.startswith("esspri_total_s")
            or lower.startswith("pop_status_s")
            or lower.startswith("esspri_total_replace_"))


def _lab_text_is_redundant(frame: pd.DataFrame, column: str) -> tuple[bool, str | None]:
    paired = column.removesuffix("__text") + "__value"
    text = frame[column]
    if not text.notna().any():
        return True, paired if paired in frame else None
    if paired not in frame or not frame[paired].notna().any():
        return False, paired if paired in frame else None
    both = text.notna() & frame[paired].notna()
    if not both.any() or (text.notna() & frame[paired].isna()).any():
        return False, paired
    parsed = pd.to_numeric(text.loc[both].astype("string").str.strip(), errors="coerce")
    numeric = pd.to_numeric(frame.loc[both, paired], errors="coerce")
    redundant = parsed.notna().all() and (parsed.astype(float) - numeric.astype(float)).abs().le(1e-12).all()
    return bool(redundant), paired


def _decision(role: str, family: str, rule: str, reason: str, **extra: object) -> dict:
    return {"role": role, "family": family, "classification_rule": rule,
            "classification_reason": reason, **extra}


def classify_lab_column(name: str, frame: pd.DataFrame) -> dict | None:
    if name in {
        "baseline_anti_ro_ssa", "baseline_anti_la_ssb", "baseline_ana", "baseline_rf",
        "baseline_cryoglobulinemia", "baseline_low_c4", "baseline_leukopenia",
    }:
        return _decision("ANALYTIC", "baseline_serology", "labs.baseline_summary", "Curated baseline serology summary")
    if name.endswith("__ever_positive_through_episode"):
        return _decision("ANALYTIC", "serology_history", "labs.ever_positive", "Curated cumulative serology state")
    if name.endswith("__patient_consensus_value"):
        return _decision("ANALYTIC", "hla_consensus", "labs.consensus_value", "Stable patient consensus value")
    if name.endswith("__patient_consensus_conflict"):
        return _decision("QC", "hla_consensus", "labs.consensus_conflict", "Consensus conflict flag")
    if name.endswith("__known_through_episode"):
        return _decision("CONTEXT", "hla_consensus", "labs.known_history", "Availability of history at episode")
    suffix_roles = {
        "__unit": ("CONTEXT", "lab_context"),
        "__reference_status": ("CONTEXT", "lab_context"),
        "__measurement_date": ("CONTEXT", "lab_context"),
        "__days_from_anchor": ("CONTEXT", "lab_context"),
        "__n_measurements": ("CONTEXT", "lab_context"),
        "__conflict": ("QC", "lab_qc"),
        "__selection_status": ("PROVENANCE", "lab_provenance"),
        "__episode_status": ("QC", "lab_qc"),
    }
    for suffix, (role, family) in suffix_roles.items():
        if name.endswith(suffix):
            return _decision(role, family, f"labs.suffix.{suffix}", f"Lab field classified by {suffix} convention")
    if name.endswith("__value"):
        role = "ANALYTIC" if frame[name].notna().any() else "PROVENANCE"
        return _decision(role, "lab_measurement", "labs.value", "Populated numeric measurement" if role == "ANALYTIC" else "Empty measurement field")
    if name.endswith("__text"):
        redundant, paired = _lab_text_is_redundant(frame, name)
        role = "PROVENANCE" if redundant else "ANALYTIC"
        return _decision(role, "lab_measurement", "labs.text_content",
                         "Empty or reconstructible numeric text" if redundant else "Non-reconstructible qualitative result",
                         paired_variable=paired)
    return None


def classify_spine_column(name: str) -> dict | None:
    if name in {"n_raw_rows_in_episode", "n_collection_dates_in_episode", "intervals_involved", "episode_span_days"}:
        return _decision("CONTEXT", "episode_context", "spine.episode_composition", "Episode construction context")
    if name in {"sjd_ever_1_2_4", "sjogrens_class_patient_values"}:
        return _decision("ANALYTIC", "classification", "spine.curated_clinical", "Curated patient classification")
    if name in {"ids__race", "ids__ethnicity", "ids__age_at_visit"}:
        return _decision(
            "ANALYTIC", "demographics", "spine.demographic", "Clinically useful demographic variable"
        )
    if name in {"visit_datetime", "ids__interval_name", "ids__time_24_hour"}:
        return _decision(
            "CONTEXT", "visit_context", "spine.visit_context", "Complementary source timing or visit context"
        )
    if name in {"ids__visit_date", "ids__patient_record_number", "ids__subject_number", "ids__dob", "ids__sex"}:
        return _decision(
            "PROVENANCE",
            "source_metadata",
            "spine.source_metadata",
            "Raw source metadata not promoted to the curated master",
        )
    qc = {"manual_review_required", "episode_has_unresolved_conflict", "essdai_has_unresolved_conflict",
          "essdai_total_consistency", "essdai_total_derived_from_domains"}
    provenance = {"assignment_rule", "manual_review_reason", "_essdai_resolution_method", "_essdai_version",
                  "_essdai_total_version", "preferred_essdai_r_raw_row", "analytic_resolution_quality",
                  "essdai_total_source"}
    if name in qc:
        return _decision("QC", "spine_qc", "spine.qc", "Episode resolution quality control")
    if name in provenance:
        return _decision("PROVENANCE", "spine_provenance", "spine.provenance", "Episode construction provenance")
    if "__" in name:
        return _decision(
            "PROVENANCE",
            "raw_source_field",
            "spine.raw_source_default",
            "Raw source-form field retained upstream; not explicitly curated",
        )
    return None


def classify_pop_column(name: str) -> dict | None:
    if name in {"essdai_total", "esspri_dryness", "esspri_fatigue", "esspri_pain", "esspri_total", "pop_status"}:
        return _decision("ANALYTIC", "pop", "pop.canonical", "Canonical observed ESSDAI/ESSPRI/Pop state")
    if name in {"pop_status_detailed", "pop_missingness_label", "protocol"}:
        return _decision("CONTEXT", "pop", "pop.context", "Interpretive or protocol context for Pop classification")
    if name in {
        "pop_status_display", "clinical_baseline_pop_status",
        "clinical_baseline_pop_status_detailed", "baseline_pop_status",
        "baseline_pop_status_display",
    }:
        return _decision(
            "DOWNSTREAM_DERIVED", "pop", "pop.derived_display_or_baseline",
            "Derivable from canonical Pop state and clinical baseline",
        )
    return None


def classify_overlap_column(name: str) -> dict | None:
    analytic = {
        "glandular_dry_eye_subjective_active", "glandular_dry_mouth_subjective_active",
        "glandular_objective_eye_active", "glandular_objective_mouth_salivary_active",
        "glandular_salivary_gland_swelling_active", "n_glandular_manifestations_active",
        "glandular_active", "extraglandular_active", "n_extraglandular_domains_active",
        "overlap_active", "overlap_intensity_count", "overlap_status",
    }
    context = {"glandular_evaluable", "extraglandular_evaluable", "active_extraglandular_domains", "overlap_evaluable"}
    if name in analytic or (name.startswith("eg_") and name.endswith(("_active", "_ordinal_score"))):
        return _decision("ANALYTIC", "overlap", "overlap.clinical", "Curated glandular or extraglandular phenotype")
    if name in context or (name.startswith("eg_") and name.endswith("_evaluable")):
        return _decision("CONTEXT", "overlap", "overlap.evaluability", "Phenotype evaluability context")
    return None


def classify_pro_column(name: str, frame: pd.DataFrame) -> dict | None:
    scores = {
        "sf36_physical_functioning", "sf36_role_physical", "sf36_bodily_pain",
        "sf36_general_health", "sf36_vitality", "sf36_social_functioning",
        "sf36_role_emotional", "sf36_mental_health", "sf36_pcs", "sf36_mcs",
        "profad_total", "mdafs_global",
    }
    if name in scores or name in {"esspri_dryness", "esspri_fatigue", "esspri_pain", "esspri_total"}:
        return _decision("ANALYTIC", "pro", "pros.score", "Canonical patient-reported score")
    if name in {"age_baseline", "sex"}:
        return _decision("ANALYTIC", "pro_covariate", "pros.baseline_covariate", "Curated baseline covariate")
    if name == "parent_protocol":
        return _decision("CONTEXT", "protocol", "pros.protocol", "Protocol membership/context")
    if name in {"baseline_pop", "overlap_baseline"}:
        return _decision(
            "DOWNSTREAM_DERIVED", "baseline_annotation", "pros.baseline_annotation",
            "Derivable from canonical longitudinal state at clinical baseline",
        )
    if name.endswith("_conflict"):
        return _decision("QC", "pro", "pros.conflict", "Scoring conflict flag")
    if name.endswith("_scoring_version") or name.endswith("_n_items_expected") or name == "sf36_norm_reference":
        role = "REGISTRY_METADATA" if frame[name].dropna().nunique() <= 1 else "CONTEXT"
        return _decision(role, "pro", "pros.scale_metadata", "Scale-level constant" if role == "REGISTRY_METADATA" else "Non-constant scoring context")
    if (name.endswith(("_n_items_answered", "_scoring_status", "_available", "_complete", "_scoring_valid",
                       "_any_item_present", "_n_items_available", "_n_domains_available"))
            or name in {"esspri_n_components_available", "esspri_partial_mean"}):
        return _decision("CONTEXT", "pro", "pros.completeness", "Availability or scoring completeness")
    if name == "esspri_n_components":
        role = "REGISTRY_METADATA" if frame[name].dropna().nunique() <= 1 else "CONTEXT"
        return _decision(role, "pro", "pros.component_count", "Instrument component count")
    return None


def classify_extended_column(name: str) -> dict | None:
    clinical_exact = {
        "biopsy_focus_score", "biopsy_focus_score_ge1",
        "salivary_flow_unstimulated", "salivary_flow_stimulated",
        "salivary_total_unstimulated_flow", "salivary_total_stimulated_flow",
        "low_unstimulated_salivary_flow", "ocular_schirmer_right", "ocular_schirmer_left",
        "ocular_tbut_right", "ocular_tbut_left", "ocular_van_bijsterveld_right",
        "ocular_van_bijsterveld_left", "ocular_oxford_right", "ocular_oxford_left",
        "ocular_schirmer_min", "ocular_tbut_min", "ocular_van_bijsterveld_max",
        "ocular_oxford_max", "ocular_schirmer_abnormal_any", "ocular_staining_positive",
        "lacrimal_dysfunction",
    }
    if name in clinical_exact or name.startswith("sicca_") or name.startswith("sjddi_"):
        return _decision("ANALYTIC", "extended_clinical", "extended.clinical_measure", "Canonical clinical phenotype")
    if name == "biopsy_focus_score_raw":
        return _decision("PROVENANCE", "biopsy", "extended.raw", "Raw source representation")
    if name.startswith("sgus_"):
        if name.endswith(("_available", "_scan_done")):
            return _decision("CONTEXT", "sgus", "extended.sgus_availability", "SGUS availability context")
        if name.endswith(("_text", "_free_text", "_notes")):
            return _decision("PROVENANCE", "sgus", "extended.sgus_text", "Free-text source detail")
        return _decision("ANALYTIC", "sgus", "extended.sgus_measure", "Upstream-defined SGUS component or phenotype")
    if name.startswith(("biopsy_", "salivary_", "ocular_")):
        # Specific generic suffix rules below still take precedence in classify_column.
        return _decision("ANALYTIC", "extended_clinical", "extended.named_family", "Upstream-defined clinical measure")
    return None


def classify_column(source: str, name: str, frame: pd.DataFrame) -> ColumnClassification:
    """Classify one observed column using deterministic, fail-closed precedence."""
    if name in STRUCTURAL_COLUMNS:
        decision = _decision("STRUCTURAL", "structure", "contract.structural", "Authoritative grain or temporal structure")
    elif _legacy(name):
        decision = _decision("LEGACY", "legacy", "contract.legacy_proxy", "Forbidden proxy or sensitivity-era variable")
    elif name in KNOWN_ALIASES:
        decision = _decision("DEPRECATED_ALIAS", "alias", "contract.known_alias", "Exact compatibility alias",
                             canonical_variable=KNOWN_ALIASES[name], alias_of=KNOWN_ALIASES[name])
    elif name in DOWNSTREAM_DERIVED or name.startswith(("previous_", "delta_")):
        decision = _decision("DOWNSTREAM_DERIVED", "temporal", "contract.downstream_derived", "Study-specific derivable temporal state")
    else:
        decision = None
        if source == "labs":
            decision = classify_lab_column(name, frame)
        elif source == "clinical_spine":
            decision = classify_spine_column(name)
        elif source == "pop":
            decision = classify_pop_column(name)
        elif source == "overlap":
            decision = classify_overlap_column(name)
        elif source == "pros":
            decision = classify_pro_column(name, frame)
        elif source == "extended_clinical":
            # Generic technical suffixes must beat the broad named-family rule.
            if name.endswith("_conflict"):
                decision = _decision("QC", "extended_clinical", "generic.conflict", "Clinical conflict flag")
            elif name.endswith("_source"):
                decision = _decision("PROVENANCE", "extended_clinical", "generic.source", "Clinical source provenance")
            elif name.endswith(("_evaluable", "_available", "_scan_done")):
                decision = _decision("CONTEXT", "extended_clinical", "generic.availability", "Clinical availability/evaluability")
            else:
                decision = classify_extended_column(name)
        if decision is None and name.endswith("_conflict"):
            decision = _decision("QC", "technical", "generic.conflict", "Conflict flag")
        if decision is None and name.endswith("_source"):
            decision = _decision("PROVENANCE", "technical", "generic.source", "Source provenance")
        if decision is None and name.endswith(("_evaluable", "_available")):
            decision = _decision("CONTEXT", "clinical_context", "generic.availability", "Availability/evaluability context")
        if decision is None and name in {"integration_version", "integration_run_date"}:
            decision = _decision("REGISTRY_METADATA", "integration", "contract.metadata", "Dataset-level metadata, not row data")
        if decision is None:
            decision = _decision("UNCLASSIFIED", "unknown", "contract.fail_closed", "No approved semantic rule")

    role = str(decision["role"])
    assert role in ROLES
    series = frame[name]
    n_nonmissing = int(series.notna().sum())
    n_unique = int(series.dropna().nunique())
    return ColumnClassification(
        variable=name, source=source, family=str(decision["family"]), role=role,
        destination=ROLE_DESTINATION[role],
        canonical_variable=decision.get("canonical_variable") or (name if role not in {"DEPRECATED_ALIAS", "LEGACY", "UNCLASSIFIED"} else None),
        alias_of=decision.get("alias_of"), classification_rule=str(decision["classification_rule"]),
        classification_reason=str(decision["classification_reason"]), dtype=str(series.dtype),
        n_nonmissing=n_nonmissing, pct_nonmissing=(100.0 * n_nonmissing / len(frame) if len(frame) else 0.0),
        n_unique=n_unique, paired_variable=decision.get("paired_variable"),
        included_in_analytic=role in {"STRUCTURAL", "ANALYTIC"}, included_in_context=role == "CONTEXT",
    )


def build_variable_registry(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [asdict(classify_column(source, column, frame))
            for source, frame in sources.items() for column in frame.columns]
    registry = pd.DataFrame(rows)
    if registry.empty or registry.groupby(["source", "variable"]).size().gt(1).any():
        raise AssertionError("registry must contain exactly one classification per source column")
    return registry


def _aligned(frame: pd.DataFrame, column: str, spine: pd.DataFrame) -> pd.Series:
    """Align one column to authoritative spine order without duplicating key labels."""
    if column not in frame.columns:
        raise CurationContractError(
            f"Column {column!r} not found for alias verification"
        )

    required = list(dict.fromkeys([*KEYS, column]))

    aligned = spine[KEYS].merge(
        frame[required],
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    return aligned[column].reset_index(drop=True)


def verify_aliases(sources: dict[str, pd.DataFrame], spine: pd.DataFrame, registry: pd.DataFrame) -> None:
    """Exclude aliases only after their canonical values have been proved equal."""
    aliases = registry.loc[registry.role.eq("DEPRECATED_ALIAS")]
    locations: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    for source, frame in sources.items():
        for column in frame.columns:
            locations.setdefault(column, []).append((source, frame))
    for row in aliases.itertuples(index=False):
        candidates = locations.get(row.alias_of, [])
        if not candidates:
            raise CurationContractError(f"{row.source}.{row.variable} aliases absent canonical {row.alias_of}")
        alias_values = _aligned(sources[row.source], row.variable, spine)
        if not any(not _different(alias_values, _aligned(frame, row.alias_of, spine)).any()
                   for _, frame in candidates):
            raise CurationContractError(
                f"deprecated alias {row.source}.{row.variable} is discordant with {row.alias_of}"
            )


def _merge_partition(base: pd.DataFrame, sources: dict[str, pd.DataFrame], registry: pd.DataFrame,
                     role: str) -> pd.DataFrame:
    output = base.copy()
    for source_name, frame in sources.items():
        selected = registry.loc[(registry.source == source_name) & (registry.role == role), "variable"].tolist()
        selected = [c for c in selected if c not in STRUCTURAL_COLUMNS and c not in output.columns]
        collisions = [c for c in registry.loc[(registry.source == source_name) & (registry.role == role), "variable"]
                      if c in output.columns and c not in STRUCTURAL_COLUMNS]
        if collisions:
            compared = output[KEYS + collisions].merge(frame[KEYS + collisions], on=KEYS, suffixes=("_existing", "_source"), validate="one_to_one")
            discordant = [c for c in collisions if _different(compared[f"{c}_existing"], compared[f"{c}_source"]).any()]
            if discordant:
                raise AssertionError(f"{source_name} has conflicting duplicate features: {sorted(discordant)}")
        if selected:
            output = output.merge(frame[KEYS + selected], on=KEYS, how="left", validate="one_to_one")
    return output


def build_coverage(analytic: pd.DataFrame, context: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    coverage = analytic[KEYS + ["is_clinical_baseline", "clinical_visit_number", "visit_type"]].copy()
    coverage["has_pop_state"] = analytic.get("pop_status", pd.Series(pd.NA, index=analytic.index)).isin(["Pop1", "Pop2", "Pop3"])
    coverage["has_essdai"] = analytic.get("essdai_total", pd.Series(pd.NA, index=analytic.index)).notna()
    coverage["has_esspri_observed"] = analytic.get("esspri_total", pd.Series(pd.NA, index=analytic.index)).notna()
    counts = [c for c in labs if c.endswith("__n_measurements")]
    dates = [c for c in labs if c.endswith("__measurement_date")]
    lab_evidence = (labs[counts].fillna(0).gt(0).any(axis=1) if counts
                    else labs[dates].notna().any(axis=1) if dates else pd.Series(False, index=labs.index))
    lab_coverage = labs[KEYS].copy()
    lab_coverage["has_lab_measurement"] = lab_evidence.to_numpy()
    coverage = coverage.merge(lab_coverage, on=KEYS, how="left", validate="one_to_one")
    coverage["has_overlap_data"] = context.get("overlap_evaluable", pd.Series(False, index=context.index)).eq(True)
    pro = [c for c in ["esspri_total", "sf36_pcs", "sf36_mcs", "profad_total", "mdafs_global"] if c in analytic]
    coverage["has_pro_data"] = analytic[pro].notna().any(axis=1) if pro else False
    evidence = []
    for column in EXTENDED_CLINICAL_PRIMARY_FEATURES:
        holder = context if column in context else analytic
        if column in holder:
            evidence.append(holder[column].eq(True) if column == "sgus_available" else holder[column].notna())
    coverage["has_extended_clinical_data"] = pd.concat(evidence, axis=1).any(axis=1) if evidence else False
    blocks = ["has_pop_state", "has_lab_measurement", "has_overlap_data", "has_pro_data", "has_extended_clinical_data"]
    coverage["n_integrated_blocks_available"] = coverage[blocks].sum(axis=1).astype("Int64")
    return coverage


def build_curated(clinical_spine: pd.DataFrame, pop: pd.DataFrame, labs: pd.DataFrame,
                  overlap: pd.DataFrame, pros: pd.DataFrame,
                  extended_clinical: pd.DataFrame, *, fail_closed: bool = True) -> CuratedBuild:
    spine = validate_spine(clinical_spine)
    raw_sources = {"clinical_spine": spine, "pop": pop, "labs": labs, "overlap": overlap,
                   "pros": pros, "extended_clinical": extended_clinical}
    validated = {"clinical_spine": spine}
    summaries: list[dict] = [{"source": "clinical_spine", "n_rows": len(spine), "n_columns": len(spine.columns),
                              "n_patients": spine.patient_id.nunique(), "n_unique_patient_episode": len(spine),
                              "n_duplicates": 0, "n_missing_spine_keys": 0, "n_extra_keys": 0,
                              "structural_discrepancies": 0, "n_inherited_spine_columns": 0}]
    mismatches: list[dict] = []
    discrepancies: list[dict] = []
    for name in ["pop", "labs", "overlap", "pros", "extended_clinical"]:
        clean, summary, key_qc, structural_qc = validate_source(spine, raw_sources[name], name)
        validated[name] = clean
        summaries.append(summary)
        mismatches.extend(key_qc)
        discrepancies.extend(structural_qc)

    registry = build_variable_registry(validated)
    pop_legacy = registry.loc[(registry.source == "pop") & (registry.role == "LEGACY"), "variable"].tolist()
    if pop_legacy:
        raise CurationContractError(
            "pop source contains deprecated ESSPRI proxy/sensitivity columns; rerun "
            f"01_pop_distribution.py with the observed-only contract. Examples: {pop_legacy[:10]}"
        )
    unclassified = registry.loc[registry.role.eq("UNCLASSIFIED")]
    if fail_closed and not unclassified.empty:
        details = unclassified[["source", "variable", "dtype", "n_nonmissing"]].to_dict("records")
        raise CurationContractError(f"unclassified upstream variables: {details[:20]}")
    verify_aliases(validated, spine, registry)

    # Only the explicit structural contract seeds the master -- never spine.copy().
    analytic = spine[STRUCTURAL_COLUMNS].copy()
    analytic_order = ["clinical_spine", "pop", "overlap", "extended_clinical", "pros", "labs"]
    analytic_sources = {name: validated[name] for name in analytic_order}
    analytic = _merge_partition(analytic, analytic_sources, registry, "ANALYTIC")
    context = spine[KEYS + ["clinical_anchor_date"]].copy()
    context = _merge_partition(context, validated, registry, "CONTEXT")
    labs_normalised = validated["labs"].reset_index(drop=True)
    coverage = build_coverage(analytic, context, labs_normalised)

    spine_keys = set(map(tuple, spine[KEYS].itertuples(index=False, name=None)))
    for label, frame in {"analytic": analytic, "context": context}.items():
        keys = set(map(tuple, frame[KEYS].itertuples(index=False, name=None)))
        if len(frame) != len(spine) or frame.duplicated(KEYS).any() or keys != spine_keys:
            raise AssertionError(f"{label} dataset does not preserve the clinical spine")
        bad = [c for c in frame if c.endswith(("_x", "_y"))]
        if bad:
            raise AssertionError(f"{label} dataset has unexpected merge columns: {bad}")
    forbidden = [c for c in analytic if c in KNOWN_ALIASES or c.startswith(("previous_", "delta_")) or _legacy(c)]
    if forbidden:
        raise AssertionError(f"analytic master contains excluded variables: {forbidden}")
    return CuratedBuild(analytic, context, coverage, registry, summaries, mismatches, discrepancies)


def build_integrated(clinical_spine: pd.DataFrame, pop: pd.DataFrame, labs: pd.DataFrame,
                     overlap: pd.DataFrame, pros: pd.DataFrame,
                     extended_clinical: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Compatibility API returning the new analytic master and source summaries."""
    build = build_curated(clinical_spine, pop, labs, overlap, pros, extended_clinical)
    return build.analytic, build.source_summaries[1:]


def build_zero_block_qc(coverage: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return zero-block episodes and their aggregate, non-identifying QC."""
    zero_block = coverage.loc[coverage["n_integrated_blocks_available"].eq(0)].copy()
    baseline = zero_block["is_clinical_baseline"].fillna(False).astype(bool)
    patient_counts = coverage.groupby("patient_id")["n_integrated_blocks_available"].sum(min_count=1)
    summary = {
        "n_zero_block_episodes": int(len(zero_block)),
        "n_zero_block_patients": int(zero_block["patient_id"].nunique()),
        "n_zero_block_clinical_baselines": int(baseline.sum()),
        "n_zero_block_nonbaseline_episodes": int((~baseline).sum()),
        "n_patients_with_all_episodes_zero_block": int(patient_counts.fillna(0).eq(0).sum()),
        "zero_block_baseline_present": bool(baseline.any()),
    }
    return zero_block, summary


def zero_block_distribution(zero_block: pd.DataFrame, column: str) -> pd.DataFrame:
    output_columns = [column, "n_zero_block_episodes", "pct_zero_block_episodes"]
    if column not in zero_block:
        return pd.DataFrame(columns=output_columns)
    output = zero_block.groupby(column, dropna=False).size().rename("n_zero_block_episodes").reset_index()
    output["pct_zero_block_episodes"] = output["n_zero_block_episodes"].div(len(zero_block)).mul(100) if len(zero_block) else 0.0
    return output[output_columns]


def _write_audits(build: CuratedBuild, output: Path, qc_dir: Path) -> dict:
    context_path = output.with_name("10_integrated_longitudinal_context.parquet")
    registry_path = output.with_name("10_variable_registry.csv")
    build.analytic.to_parquet(output, index=False)
    build.analytic.to_csv(output.with_suffix(".csv"), index=False)
    build.context.to_parquet(context_path, index=False)
    build.registry.to_csv(registry_path, index=False)
    pd.DataFrame(build.source_summaries).to_csv(qc_dir / "10_integrated_source_summary.csv", index=False)
    pd.DataFrame(build.key_mismatches, columns=["source", *KEYS, "mismatch"]).to_csv(qc_dir / "10_integrated_key_mismatch_qc.csv", index=False)
    pd.DataFrame(build.structural_discrepancies, columns=["source", *KEYS, "variable", "spine_value", "source_value"]).to_csv(qc_dir / "10_integrated_structural_discrepancy_qc.csv", index=False)
    build.coverage[KEYS + [c for c in build.coverage if c.startswith("has_") or c == "n_integrated_blocks_available"]].to_csv(
        qc_dir / "10_integrated_longitudinal_coverage.csv", index=False
    )
    build.registry.loc[build.registry.role.eq("UNCLASSIFIED")].to_csv(qc_dir / "10_unclassified_variables.csv", index=False)
    excluded = build.registry.loc[~build.registry.role.isin(["STRUCTURAL", "ANALYTIC", "CONTEXT"])]
    excluded.to_csv(qc_dir / "10_excluded_variables.csv", index=False)
    summary = build.registry.groupby(["role", "destination"]).size().rename("n_source_variables").reset_index()
    summary.to_csv(qc_dir / "10_curation_summary.csv", index=False)
    zero_block, zero_summary = build_zero_block_qc(build.coverage)
    for variable, filename in [("clinical_visit_number", "10_zero_block_by_clinical_visit_number.csv"),
                               ("visit_type", "10_zero_block_by_visit_type.csv"),
                               ("is_clinical_baseline", "10_zero_block_by_baseline_status.csv")]:
        zero_block_distribution(zero_block, variable).to_csv(qc_dir / filename, index=False)
    counts = build.registry.role.value_counts()
    qc = {
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "integration_build_version": INTEGRATION_BUILD_VERSION,
        "run_date": date.today().isoformat(), "n_rows": len(build.analytic),
        "n_patients": int(build.analytic.patient_id.nunique()),
        "n_unique_patient_episode": int(build.analytic[KEYS].drop_duplicates().shape[0]),
        "n_duplicate_keys": int(build.analytic.duplicated(KEYS).sum()),
        "spine_preserved": True, "n_analytic_columns": len(build.analytic.columns),
        "n_context_columns": len(build.context.columns),
        "n_qc_columns_excluded": int(counts.get("QC", 0) + counts.get("PROVENANCE", 0)),
        "n_aliases_excluded": int(counts.get("DEPRECATED_ALIAS", 0)),
        "n_legacy_columns": int(counts.get("LEGACY", 0)),
        "n_downstream_derived": int(counts.get("DOWNSTREAM_DERIVED", 0)),
        "n_registry_metadata": int(counts.get("REGISTRY_METADATA", 0)),
        "n_unclassified": int(counts.get("UNCLASSIFIED", 0)),
        "input_column_counts": {row["source"]: row["n_columns"] for row in build.source_summaries},
        "n_unique_observed_columns": int(build.registry.variable.nunique()),
        "sources": build.source_summaries, "zero_block_qc": zero_summary,
    }
    (qc_dir / "10_integrated_longitudinal_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spine", type=Path, default=common.CLINICAL_VISIT_SPINE_PARQUET)
    parser.add_argument("--pop", type=Path, default=common.POP_LONGITUDINAL_PARQUET)
    parser.add_argument("--labs", type=Path, default=common.LABS_EPISODE_WIDE_PARQUET)
    parser.add_argument("--overlap", type=Path, default=common.OVERLAP_LONGITUDINAL_PARQUET)
    parser.add_argument("--pros", type=Path, default=common.PROS_LONGITUDINAL_PARQUET)
    parser.add_argument("--extended-clinical", type=Path, default=common.EXTENDED_CLINICAL_LONGITUDINAL_PARQUET)
    parser.add_argument("--output", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    args = parser.parse_args()
    common.ensure_output_dirs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    qc_dir = common.BLOCKA_QC_DIR / "10_build_integrated_longitudinal_dataset"
    qc_dir.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(path) for path in [args.spine, args.pop, args.labs, args.overlap, args.pros, args.extended_clinical]]
    # Classify once without raising so contract failures still leave actionable audit files.
    build = build_curated(*frames, fail_closed=False)
    build.registry.to_csv(args.output.with_name("10_variable_registry.csv"), index=False)
    build.registry.loc[build.registry.role.eq("UNCLASSIFIED")].to_csv(qc_dir / "10_unclassified_variables.csv", index=False)
    unclassified = build.registry.loc[build.registry.role.eq("UNCLASSIFIED")]
    if not unclassified.empty:
        details = unclassified[["source", "variable", "dtype", "n_nonmissing"]].to_dict("records")
        raise CurationContractError(f"unclassified upstream variables: {details[:20]}")
    qc = _write_audits(build, args.output, qc_dir)
    if qc["zero_block_qc"]["zero_block_baseline_present"]:
        print("WARNING: clinical baseline episodes exist with zero integrated data blocks; review before Step 11.")
    print(f"Wrote {len(build.analytic):,} curated clinical episodes to {args.output}")


if __name__ == "__main__":
    main()

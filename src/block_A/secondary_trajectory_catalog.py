"""Conservative clinical metadata for the read-only Step 15 consumer.

Every integrated column is inventoried. Unknown clinical semantics never become
plottable merely because pandas reports a numeric dtype.
"""
from dataclasses import asdict, dataclass
import re

import pandas as pd

from src.derivations.pro_scoring import PRO_SCORE_RANGES, SF36_MEASURES


@dataclass(frozen=True)
class Variable:
    variable_id: str
    domain: str
    label: str
    canonical_column: str
    source_producer: str
    type: str = "numeric"
    unit: str = ""
    scale_min: float | None = None
    scale_max: float | None = None
    clinical_direction: str = "descriptive; no secondary pattern classification"
    measurement_date_column: str = "clinical_anchor_date"
    observation_semantics: str = "measured_at_episode"
    aggregation_policy: str = "last observed date per patient per annual window"
    reference_policy: str = "official episode if observed, else first valid post-baseline"
    expected_longitudinal: bool = True
    is_essdai_derived: bool = False
    raw_aliases: str = ""
    limitation: str = ""


def candidates():
    """Metadata audited against Steps 01, 06, 09, 10 and 11; no raw scoring."""
    out = []
    for component in ("total", "dryness", "fatigue", "pain"):
        out.append(Variable("esspri_"+component, "esspri", "ESSPRI "+component,
                            "esspri__"+component, "01_pop_distribution.py", unit="points",
                            scale_min=0, scale_max=10))
    for name, limits in PRO_SCORE_RANGES.items():
        if name.startswith("esspri_"):
            continue
        out.append(Variable(name, "pro", SF36_MEASURES.get(name, name.replace("_", " ").upper()),
                            "pro__"+name, "09_pros_longitudinal.py", unit="instrument points",
                            scale_min=limits[0], scale_max=limits[1]))
    for analyte in ("complement_c3", "complement_c4", "igg", "esr", "crp", "crp_high_sensitivity"):
        out.append(laboratory("lab__"+analyte+"__value"))
    for name in ("anti_ro_ssa", "anti_la_ssb", "ana", "rf", "cryoglobulinemia"):
        out.append(serology("sero__"+name+"__ever_positive_through_episode"))
    for name in ("anti_ro_ssa", "anti_la_ssb", "ana", "rf", "cryoglobulinemia", "low_c4", "leukopenia"):
        out.append(Variable(name+"_baseline_summary", "serology", name.replace("_", " ")+" (upstream baseline summary)",
                            "sero__baseline_"+name, "01_serological_profile.py", type="binary", unit="% evaluable",
                            measurement_date_column="unavailable_baseline_serology_test_date",
                            observation_semantics="historical_ever", expected_longitudinal=False,
                            limitation="External baseline serology summary has no verified individual test date; not an episode test"))
    for scale, unit, maximum in (("schirmer", "mm/5 min", 50), ("tbut", "seconds", 180),
                                 ("van_bijsterveld", "score", 9), ("oxford", "score", 10)):
        for side in ("right", "left", "min" if scale in ("schirmer", "tbut") else "max"):
            name = f"ocular_{scale}_{side}"
            out.append(Variable(name, "ocular", name.replace("_", " "), "ext__"+name,
                                "01_extended_clinical_phenotype.py", unit=unit, scale_min=0, scale_max=maximum))
    for name in ("salivary_flow_unstimulated", "salivary_flow_stimulated"):
        out.append(Variable(name, "salivary", name.replace("_", " "), "ext__"+name,
                            "01_extended_clinical_phenotype.py", unit="mL/min", scale_min=0,
                            limitation="Upstream rate contract; collection volumes are not substituted"))
    for name, domain in (("ocular_staining_positive", "ocular"), ("lacrimal_dysfunction", "ocular"),
                         ("ocular_schirmer_abnormal_any", "ocular"), ("low_unstimulated_salivary_flow", "salivary"),
                         ("sicca_any_symptom", "other"), ("sgus_abnormal", "sgus")):
        out.append(Variable(name, domain, name.replace("_", " "), "ext__"+name,
                            "01_extended_clinical_phenotype.py", type="binary", unit="% evaluable"))
    out.append(Variable("biopsy_focus_score", "pathology", "Biopsy focus score", "ext__biopsy_focus_score",
                        "01_extended_clinical_phenotype.py", unit="foci/4 mm²", scale_min=0))
    out.append(Variable("overlap_status", "overlap", "Glandular/extraglandular overlap", "ovl__status",
                        "06_overlap_glandular.py", type="categorical", unit="% evaluable"))
    # Explicitly unresolved candidates remain visible even if not integrated.
    for name, domain in (("sgus_summary_score", "sgus"), ("salivary_total_unstimulated_flow", "salivary"),
                         ("salivary_total_stimulated_flow", "salivary")):
        out.append(Variable(name, domain, name.replace("_", " "), "ext__"+name,
                            "01_extended_clinical_phenotype.py", limitation="scale/volume unit requires verification"))
    return out


def laboratory(column):
    stem = column.removesuffix("__value")
    name = stem.removeprefix("lab__")
    return Variable(name, "labs", name.replace("_", " "), column, "01_serological_profile.py",
                    measurement_date_column=stem+"__measurement_date", observation_semantics="measured_on_date",
                    scale_min=0, limitation="requires producer-harmonized unit and actual laboratory date")


def serology(column):
    name = column.removeprefix("sero__").removesuffix("__ever_positive_through_episode")
    label = {"anti_ro_ssa": "Anti-Ro/SSA", "anti_la_ssb": "Anti-La/SSB", "ana": "ANA", "rf": "Rheumatoid factor"}.get(name, name.replace("_", " "))
    return Variable(name, "serology", label+" (cumulative known positivity)", column,
                    "01_serological_profile.py", type="binary", unit="% evaluable",
                    observation_semantics="cumulative_through_episode", expected_longitudinal=False,
                    limitation="First available known history only; not a newly observed test or seroconversion")


def infer_registered(column):
    if re.fullmatch(r"lab__[a-z0-9_]+__value", column):
        return laboratory(column)
    if re.fullmatch(r"sero__[a-z0-9_]+__ever_positive_through_episode", column):
        return serology(column)
    if column.startswith("essdai__") and column.endswith(("_ordinal_score", "_active")):
        numeric = column.endswith("_ordinal_score")
        return Variable(column.replace("__", "_"), "essdai_domains", column.removeprefix("essdai__").replace("_", " "),
                        column, "06_overlap_glandular.py", type="ordinal" if numeric else "binary",
                        unit="ordinal level" if numeric else "% evaluable", scale_min=0, scale_max=3 if numeric else 1,
                        observation_semantics="derived_from_essdai", is_essdai_derived=True,
                        limitation="Partially mechanical association: domains also define ESSDAI groups")
    if column.startswith("ovl__") and column.endswith("_active"):
        return Variable(column.removeprefix("ovl__"), "overlap", column.removeprefix("ovl__").replace("_", " "),
                        column, "06_overlap_glandular.py", type="binary", unit="% evaluable",
                        is_essdai_derived="extraglandular" in column or "overlap" in column)
    if column in {"ovl__n_extraglandular_domains_active", "ovl__n_glandular_manifestations_active", "ovl__intensity_count"}:
        return Variable(column.removeprefix("ovl__"), "overlap", column.removeprefix("ovl__").replace("_", " "),
                        column, "06_overlap_glandular.py", unit="count", scale_min=0, is_essdai_derived=True)
    if re.fullmatch(r"ext__sgus_(theander_final_score_grade|omeract_[rl]_(parotid|submandibular)_grade)", column):
        return Variable(column.removeprefix("ext__"), "sgus", column.removeprefix("ext__").replace("_", " "),
                        column, "01_extended_clinical_phenotype.py", type="ordinal", unit="grade", scale_min=0, scale_max=3)
    return None


def build_catalog(integrated, source_registry=None):
    """Return specs plus complete inclusion/exclusion metadata, without writes."""
    specs = {v.canonical_column: v for v in candidates()}
    columns = set(integrated)
    if source_registry is not None and "public_variable" in source_registry:
        columns.update(source_registry.public_variable.dropna().astype(str))
    for col in sorted(columns):
        if col not in specs:
            spec = infer_registered(col)
            if spec:
                specs[col] = spec
    records = []
    for col in sorted(columns | set(specs)):
        spec = specs.get(col)
        found = col in integrated
        if spec is None:
            reason = ("excluded_identifier_or_time_metadata" if any(t in col for t in ("patient", "episode", "date", "protocol", "record", "subject"))
                      else "unverified_semantics_or_technical_metadata")
            if not found:
                reason = "not_in_integrated; "+reason
            records.append(dict(variable_id=col, canonical_column=col, domain="other", column_found=found,
                                type="unresolved", source_producer="unresolved", observation_semantics="unverified",
                                status="excluded", exclusion_or_limitation_reason=reason))
            continue
        reason = "" if found else "not_in_integrated"
        if found and not spec.unit and spec.domain != "labs":
            reason = "unknown_units"
        records.append(dict(**asdict(spec), column_found=found, status="candidate" if not reason else "excluded",
                            exclusion_or_limitation_reason=reason or spec.limitation))
    # Labels/IDs and filenames are metadata; no patient values enter this table.
    return specs, pd.DataFrame(records)

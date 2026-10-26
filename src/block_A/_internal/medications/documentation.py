#!/usr/bin/env python3
"""Validated v2 normalization, identity reconciliation and order QC functions.

Extracted from the recovered-source audit; no CLI or archived imports.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import hmac
import importlib.util
import json
import math
import numbers
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from . import schema as v1, classification
from .dates import parse_date

RULES_VERSION = "08-treatment-audit-2.1.0"
MAPPING_VERSION = "08-drug-candidates-2.0.0"
BTRIS_PROTOCOL_EXPECTED = {"11D": "11-D-0172", "15D": "15-D-0051"}
BTRIS_MEDICATION_COLUMNS = [
    "Protocol Number", "MRN", "Subject Name", "Date of Birth", "Gender",
    "Race", "Ethnic Group", "Date of Death", "Medication Name", "Drug Ingredient",
    "Drug Type", "Category", "Non Formulary Info", "Order Type", "Order ID",
    "Order Date/Start Date", "Order End Date", "Order Status", "Order Set Name",
    "Order Dose/Strength", "Order UOM", "Order Route", "Order No. of Refills",
    "Order Frequency", "Order Instruction", "Take Home Indicator",
    "Initial Consent Date", "Data Source",
]
BTRIS_PATIENT_KEY = "MRN"
BTRIS_ORDER_KEY = "Order ID"
BTRIS_DRUG_FIELDS = ["Medication Name", "Drug Ingredient", "Non Formulary Info"]
BTRIS_TIME_FIELDS = ["Order Date/Start Date", "Order End Date"]
BTRIS_ORDER_FIELDS = [
    "Order Type", "Order Status", "Order Set Name", "Data Source",
    "Take Home Indicator", "Order No. of Refills",
]
BTRIS_DOSE_FIELDS = [
    "Order Dose/Strength", "Order UOM", "Order Route", "Order Frequency", "Order Instruction",
]
CTDB_RX_NAMES = v1.MEDICATION_NAME_COLUMNS
CTDB_RX_ROUTES = v1.MEDICATION_ROUTE_COLUMNS
CTDB_RX_FREQ_BY_SLOT = v1.MEDICATION_FREQUENCY_BY_SLOT
CTDB_RX_START_BY_SLOT = v1.MEDICATION_START_BY_SLOT
CTDB_RX_STOP_BY_SLOT = v1.MEDICATION_STOP_BY_SLOT
CTDB_UNASSIGNED_DOSES = v1.MEDICATION_DOSE_COLUMNS
CTDB_PMH_RX_NAMES = v1.HISTORICAL_RX_NAME_COLUMNS
CTDB_PMH_ALLERGY = v1.RX_ALLERGY_FIELD
CTDB_PATIENT_ID = "ids__patient_record_number"
CTDB_VISIT_DATE = "ids__visit_date"
ELIGIBILITY_PATIENT_ID = "patient_id"
CTDB_COLUMNS = [c for group in v1.FIELD_GROUPS.values() for c in group]
PROTOCOL_DRUGS = (
    "hydroxychloroquine", "methotrexate", "mycophenolate mofetil", "leflunomide",
    "rituximab", "belimumab",
)

# Candidate rules, not a claim that these products occur in either source.
# All observed vocabulary, including unknowns and combinations, is inventoried.
DRUG_RULES = (
    ("hydroxychloroquine", "nonbiologic_systemic", ("hydroxychloroquine", "hydroxychloroquine sulfate", "plaquenil", "hcq")),
    ("methotrexate", "nonbiologic_systemic", ("methotrexate", "trexall", "mtx", "otrexup", "rasuvo")),
    ("mycophenolate mofetil", "nonbiologic_systemic", ("mycophenolate mofetil", "cellcept", "mmf")),
    ("mycophenolic acid sodium", "nonbiologic_systemic", ("mycophenolic acid sodium", "mycophenolate sodium", "myfortic")),
    ("leflunomide", "nonbiologic_systemic", ("leflunomide", "arava")),
    ("rituximab", "biologic", ("rituximab", "rituxan", "mabthera", "truxima", "ruxience", "riabni", "rituximab-abbs", "rituximab-pvvr", "rituximab-arrx")),
    ("belimumab", "biologic", ("belimumab", "benlysta")),
    ("prednisone", "systemic_glucocorticoid", ("prednisone",)),
    ("prednisolone", "systemic_glucocorticoid", ("prednisolone",)),
    ("methylprednisolone", "systemic_glucocorticoid", ("methylprednisolone", "medrol", "solu-medrol")),
    ("dexamethasone", "systemic_glucocorticoid", ("dexamethasone", "decadron")),
    ("pilocarpine", "symptomatic_sjd_oral", ("pilocarpine", "pilocarpine hydrochloride", "salagen")),
    ("cevimeline", "symptomatic_sjd_oral", ("cevimeline", "cevimeline hydrochloride", "evoxac")),
    ("cyclosporine", "needs_review", ("cyclosporine", "ciclosporin", "restasis", "cequa", "sandimmune", "neoral")),
    ("artificial tears", "symptomatic_sjd_ocular", ("artificial tears", "refresh tears", "systane")),
    ("aspirin", "concomitant_other", ("aspirin",)),
    ("atorvastatin", "concomitant_other", ("atorvastatin", "lipitor")),
    ("acetaminophen", "other_symptomatic", ("acetaminophen", "paracetamol", "tylenol")),
    ("trimethoprim", "concomitant_other", ("trimethoprim",)),
    ("sulfamethoxazole", "concomitant_other", ("sulfamethoxazole",)),
)
SUFFIX = re.compile(
    r"(?:(?:\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|%|units?)?)|"
    r"tablets?|tabs?|capsules?|caps?|oral|ophthalmic|ocular|solution|suspension|"
    r"injection|cream|ointment|drops|delayed|release|sodium|po|iv|sc|im|daily|bid|tid|weekly|prn|[\s/(),.-])+$",
    re.I,
)
EMPTY = {"", "na", "n/a", "nan", "none", "null", "missing", "unknown", "unk", ".", "-99", "not available", "(missing)"}
STATUS_FIELDS = [
    "Order Type", "Order Status", "Order Set Name", "Data Source", "Take Home Indicator",
    "Order No. of Refills", "Drug Type", "Category", "Order Route",
]
DATE_KEYS = ("raw", "parsed", "precision", "lower_bound", "upper_bound", "parse_rule", "status")


def raw(value):
    return "" if value is None or pd.isna(value) else str(value)


def present(value):
    return raw(value).strip().casefold() not in EMPTY


def key(value, *, mrn=False):
    """Only exact text MRNs qualify for validated patient linkage."""
    if not present(value) or (mrn and not isinstance(value, str)):
        return ""
    return raw(value).strip()


def mrn_audit_identity(value, secret):
    """Pseudonymize non-text source MRNs for QC only, NEVER for linkage.

    Numeric parquet values may already have lost leading zeros. Their distinct
    representations must not be advertised as validated unique patients.
    """
    exact = key(value, mrn=True)
    if exact:
        return pseudonym(secret, "mrn", exact), "text_exact"
    if not present(value):
        return "", "missing"
    typed_value = f"{type(value).__name__}:{raw(value).strip()}"
    return pseudonym(secret, "mrn_nontext_unverified", typed_value), "nontext_unverified"


def serology_patient_id_match(value):
    """Match the normalization in 01_serological_profile._normalize_patient_id.

    For Parquet numeric MRNs only, first obtain an exact integral string. This
    is necessary because pandas converts numeric values to strings like '123.0'.
    Never round nonintegral values, use identifiers beyond float exactness, or
    use names/DOB/protocol as patient matching keys.
    """
    if not present(value) or isinstance(value, bool):
        return "", "missing_or_invalid"
    if isinstance(value, numbers.Integral):
        text, representation = str(int(value)), "integral_numeric"
    elif isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer() or abs(numeric) > 2 ** 53:
            return "", "unsafe_numeric_mrn"
        text, representation = str(int(numeric)), "integral_float"
    else:
        text, representation = raw(value), "source_text"
        # Textual float artifacts must only be treated as decimal integers if
        # the exact string has a zero-only fractional part.
        if re.fullmatch(r"\d+\.0+", text.strip()):
            text, representation = text.strip().split(".", 1)[0], "text_integer_decimal"
    match = re.sub(r"^0+", "", re.sub(r"[-/\\\s]", "", text.strip()))
    if match in {"", "nan", "None"}:
        return "", "missing_after_normalization"
    return match, representation


def validate_serology_style_linkage(frames, ctdb, eligible):
    """Audit-only patient matching following Step 01 normalized-ID semantics.

    The normalized eligible ID is the authority; do not require every patient
    to have a BTRIS order. Refuse ambiguous CTDB IDs or unsafe BTRIS MRNs.
    Folder/protocol must never be used as a surrogate person identifier.
    """
    result = dict(status="blocked_linkage", reason="NORMALIZED_ID_LINKAGE_NOT_VALIDATED",
                  mode="serology_normalized_id", mapping={}, normalized_mapping={},
                  ambiguous_mrns=set(), ambiguous_patients=set(), invalid_rows=0,
                  n_crosswalk_pairs=None, complete_eligible_mapping=False,
                  n_btris_rows_matched=0, n_btris_rows_outside_eligibility=0,
                  n_btris_rows_unsafe_id=0, n_btris_distinct_matching_keys=0,
                  n_ctdb_normalization_collisions=0)
    if ctdb is None or CTDB_PATIENT_ID not in ctdb or not eligible:
        result["reason"] = "CTDB_OR_ELIGIBILITY_UNAVAILABLE"
        return result
    canonical = {}
    collisions = set()
    for patient in sorted(eligible):
        norm, quality = serology_patient_id_match(patient)
        if not norm or quality == "unsafe_numeric_mrn":
            result["invalid_rows"] += 1
            continue
        if norm in canonical and canonical[norm] != patient:
            collisions.add(norm)
        canonical[norm] = patient
    source_ids = set(ctdb[CTDB_PATIENT_ID].map(key)) - {""}
    for patient in source_ids:
        norm, _ = serology_patient_id_match(patient)
        if norm in canonical and canonical[norm] != patient:
            collisions.add(norm)
    result["n_ctdb_normalization_collisions"] = len(collisions)
    if collisions or result["invalid_rows"]:
        result["reason"] = "CTDB_NORMALIZATION_COLLISIONS_OR_INVALID_IDS"
        return result
    if not source_ids.issuperset(eligible):
        result["reason"] = "ELIGIBLE_IDS_MISSING_FROM_CTDB"
        return result
    matching = set()
    unsafe = 0
    outside = 0
    rows_matched = 0
    for frame in frames.values():
        if frame is None or "MRN" not in frame:
            result["reason"] = "BTRIS_SOURCE_OR_MRN_UNAVAILABLE"
            return result
        for value in frame["MRN"]:
            norm, quality = serology_patient_id_match(value)
            if not norm:
                unsafe += 1
            elif norm in canonical:
                matching.add(norm)
                rows_matched += 1
            else:
                outside += 1
    result.update(n_btris_rows_matched=rows_matched,
                  n_btris_rows_outside_eligibility=outside,
                  n_btris_rows_unsafe_id=unsafe,
                  n_btris_distinct_matching_keys=len(matching),
                  n_crosswalk_pairs=len(matching),
                  complete_eligible_mapping=True,
                  normalized_mapping={k: canonical[k] for k in matching})
    if unsafe:
        result["reason"] = "UNSAFE_OR_MISSING_BTRIS_MRN_ROWS"
        return result
    if not matching:
        result["reason"] = "NO_BTRIS_KEYS_MATCH_ELIGIBLE_PATIENT_IDS"
        return result
    result.update(status="validated",
                  reason="serology_style_normalization_collision_checked_same_identifier_assumption")
    return result


def pseudonym(secret, namespace, value):
    if not value:
        return ""
    return hmac.new(secret, f"{namespace}\0{value}".encode(), hashlib.sha256).hexdigest()


def protocol(value):
    text = raw(value).strip()
    compact = re.sub(r"[\s-]", "", text).upper()
    for canonical in BTRIS_PROTOCOL_EXPECTED.values():
        if compact == canonical.replace("-", ""):
            return canonical
    return "unverified_protocol" if present(value) else "protocol_missing"






def _drug_match(value):
    text = raw(value).strip().casefold()
    for ingredient, drug_class, aliases in DRUG_RULES:
        for alias in sorted(aliases, key=len, reverse=True):
            if text == alias or (text.startswith(alias) and SUFFIX.fullmatch(text[len(alias):])):
                return ingredient, drug_class, alias
    return None


def _components(value):
    # An unknown multi-ingredient product remains needs_review, not single-agent.
    parts = re.split(r"\s*(?:\+|;|\band\b)\s*|\s*/\s*", raw(value), flags=re.I)
    matches = [_drug_match(part) for part in parts]
    if len(parts) > 1 and all(matches):
        return matches
    match = _drug_match(value)
    return [match] if match else []


def normalize_drug(drug, ingredient="", route=""):
    product_matches = _components(drug)
    ingredient_matches = _components(ingredient)
    product_keys = {m[0] for m in product_matches}
    ingredient_keys = {m[0] for m in ingredient_matches}
    conflict = bool(product_keys and ingredient_keys and product_keys != ingredient_keys)
    chosen = ingredient_matches or product_matches
    flags = []
    if present(ingredient) and not ingredient_matches:
        flags.append("ingredient_unverified_product_fallback")
    if conflict:
        flags.append("ingredient_product_conflict")
    result = dict(drug_raw=raw(drug), ingredient_raw=raw(ingredient), drug_normalized="",
                  ingredient_normalized="", brand="", formulation="unresolved", route=raw(route),
                  drug_class="unmapped", mapping_rule="no_unambiguous_rule",
                  mapping_confidence="unmapped", mapping_version=MAPPING_VERSION,
                  indication="indeterminate", normalization_flags="")
    if not chosen:
        if re.search(r"\+|;|\band\b", raw(ingredient) or raw(drug), re.I):
            result["drug_class"] = "needs_review"
        result["normalization_flags"] = "|".join(flags)
        return result
    normalized = " + ".join(sorted({m[0] for m in chosen}))
    ingredient_normalized = normalized
    classes = {m[1] for m in chosen}
    drug_class = next(iter(classes)) if len(classes) == 1 else "needs_review"
    context = " ".join([raw(drug), raw(ingredient), raw(route)]).casefold()
    route_text = raw(route).strip().casefold()
    ocular = bool(re.search(r"\b(ophthalmic|ocular|eye|restasis|cequa)\b", context))
    topical = bool(re.search(r"\b(topical|cream|ointment|dermal|otic|inhaled|intranasal)\b", context))
    systemic = route_text in {"po", "oral", "iv", "intravenous", "im", "intramuscular", "sc", "subcutaneous"}
    systemic = systemic or bool(re.search(r"\b(oral|tablets?|capsules?|injection)\b", context))
    formulation = "ophthalmic" if ocular else "topical_or_local" if topical else "systemic" if systemic else "unresolved"
    if normalized == "artificial tears":
        formulation = "ophthalmic"
    if normalized == "mycophenolic acid sodium":
        formulation = "mycophenolic_acid_sodium_distinct_from_MMF"
    if normalized == "cyclosporine":
        if ocular:
            normalized, drug_class = "cyclosporine ophthalmic", "symptomatic_sjd_ocular"
        else:
            drug_class = "nonbiologic_systemic" if systemic and not topical else "needs_review"
    if drug_class == "systemic_glucocorticoid" and (ocular or topical or not systemic):
        drug_class = "other_symptomatic" if ocular or topical else "needs_review"
        flags.append("glucocorticoid_route_requires_review" if not (ocular or topical) else "local_glucocorticoid_not_systemic")
    if normalized == "pilocarpine" and (ocular or topical or not systemic):
        drug_class = "symptomatic_sjd_ocular" if ocular else "needs_review"
    if conflict or (present(ingredient) and not ingredient_matches):
        drug_class = "needs_review"
    if (ocular or topical) and route_text in {"po", "oral", "iv", "intravenous", "im", "intramuscular", "sc", "subcutaneous"}:
        drug_class = "needs_review"
        flags.append("formulation_route_conflict")
    result.update(drug_normalized=normalized, ingredient_normalized=ingredient_normalized,
                  brand="|".join(sorted({m[2] for m in product_matches if m[2] != m[0]})),
                  formulation=formulation, drug_class=drug_class,
                  mapping_rule="candidate_dictionary:ingredient" if ingredient_matches else "candidate_dictionary:product",
                  mapping_confidence="candidate_requires_review", normalization_flags="|".join(flags))
    return result











def date_fields(start, end):
    return {f"{prefix}_{name}": value for prefix, parsed in (("start", start), ("end", end))
            for name, value in parsed.items()}


def date_flags(start, end):
    flags = []
    for name, value in (("start", start), ("end", end)):
        if value["status"] not in {"missing", "valid"}:
            flags.append(f"{name}_{value['status']}")
        if value["status"] == "valid" and (value["lower_bound"] < "2011-01-01" or value["upper_bound"] > "2026-12-31"):
            flags.append(f"{name}_outside_2011_2026_review")
    if end["status"] == "missing":
        flags.append("end_missing_open_or_undocumented")
    if start["status"] == end["status"] == "valid":
        # Compare exact timestamps when BOTH source values include time.
        # Day-level uncertainty bounds would miss reverse ordering within a day.
        ordered_at_time = False
        if (start["precision"] == end["precision"] == "datetime"
                and start["parsed"] and end["parsed"]):
            try:
                start_stamp = datetime.fromisoformat(start["parsed"])
                end_stamp = datetime.fromisoformat(end["parsed"])
                if (start_stamp.tzinfo is None) == (end_stamp.tzinfo is None):
                    ordered_at_time = True
                    if end_stamp < start_stamp:
                        flags.append("end_definitely_before_start")
            except (ValueError, TypeError):
                pass
        if not ordered_at_time:
            if end["upper_bound"] < start["lower_bound"]:
                flags.append("end_definitely_before_start")
            elif end["lower_bound"] < start["upper_bound"]:
                flags.append("temporal_order_uncertain")
        if start["parsed"] and start["parsed"] == end["parsed"]:
            flags.append("start_end_identical")
    return flags


def signature(row, secret, *, exclude_protocol=False):
    # Include unexpected columns too, and distinguish null from a blank string.
    fields = sorted(c for c in row.index if not (exclude_protocol and c == "Protocol Number"))
    values = [[c, "null" if pd.isna(row[c]) else type(row[c]).__name__, raw(row[c])] for c in fields]
    return pseudonym(secret, "signature", json.dumps(values, ensure_ascii=False))


def _dose_flags(row):
    text = " ".join(raw(row.get(c)) for c in BTRIS_DOSE_FIELDS).casefold()
    flags = []
    for label, expression in (
        ("prn", r"\bprn\b|as needed"), ("titration_or_variable", r"titrat|variable|as directed|taper"),
        ("loading_dose", r"loading|load dose"), ("infusion", r"infus"),
    ):
        if re.search(expression, text):
            flags.append(label)
    refill = row.get("Order No. of Refills")
    if present(refill):
        try:
            number = float(raw(refill))
            if not (0 <= number <= 99 and number.is_integer()):
                flags.append("refills_nonplausible_review")
        except ValueError:
            flags.append("refills_unparsed")
    return flags


def btris_records(frame, source, linkage, eligible, secret):
    records = []
    if frame is None:
        return records
    for position, (_, row) in enumerate(frame.iterrows()):
        mrn = key(row.get("MRN"), mrn=True)
        mrn_token, mrn_identity_quality = mrn_audit_identity(row.get("MRN"), secret)
        patient = (linkage.get("normalized_mapping", {}).get(
            serology_patient_id_match(row.get("MRN"))[0], "")
            if linkage.get("mode") == "serology_normalized_id"
            else linkage["mapping"].get(mrn, ""))
        start, end = (parse_date(row.get(c)) for c in BTRIS_TIME_FIELDS)
        flags = date_flags(start, end) + _dose_flags(row)
        if not mrn:
            if patient and mrn_token:
                flags.append("nontext_mrn_normalized_match_requires_identifier_review")
            else:
                flags.append("nontext_mrn_not_linkable" if mrn_token else "missing_mrn_not_linkable")
        if mrn in linkage["ambiguous_mrns"]:
            flags.append("ambiguous_key")
        if protocol(row.get("Protocol Number")) != BTRIS_PROTOCOL_EXPECTED[source]:
            flags.append("protocol_folder_not_confirmed_by_value")
        death = parse_date(row.get("Date of Death"))
        consent = parse_date(row.get("Initial Consent Date"))
        if start["status"] == "valid":
            if death["status"] == "valid" and start["lower_bound"] > death["upper_bound"]:
                flags.append("order_after_documented_death_verify")
            if consent["status"] == "valid" and start["upper_bound"] < consent["lower_bound"]:
                flags.append("preconsent_historical_order_retained")
        normalized = normalize_drug(row.get("Medication Name"), row.get("Drug Ingredient"), row.get("Order Route"))
        full = signature(row, secret)
        order_id = key(row.get("Order ID"))
        name_present = present(row.get("Medication Name")) or present(row.get("Drug Ingredient"))
        prescription = (name_present and start["status"] == "valid" and start["precision"] in {"day", "datetime"}
                        and all(present(row.get(c)) for c in ["Order Route", "Order Dose/Strength", "Order UOM", "Order Frequency"]))
        records.append(dict(
            audit_record_id=f"{source}:{position}", source_kind="BTRIS", source=source,
            source_protocol_folder=source, protocol_number_raw=raw(row.get("Protocol Number")),
            protocol=protocol(row.get("Protocol Number")), source_row=position, source_row_or_order_id=position,
            patient_id_pseudonymous=pseudonym(secret, "patient", patient) if patient else mrn_token,
            source_patient_pseudonymous=mrn_token, mrn_identity_quality=mrn_identity_quality,
            linked_patient_pseudonymous=pseudonym(secret, "patient", patient),
            linkage_status=("linked_one_to_one" if patient else
                            "ambiguous_key" if mrn in linkage["ambiguous_mrns"] else
                            "unmatched_btris" if mrn else
                            "nontext_unverified" if mrn_token else "invalid_key"),
            eligible_linked=bool(patient and patient in eligible), order_id_pseudonymous=pseudonym(secret, "order", order_id),
            order_identity=pseudonym(secret, "order_identity", json.dumps([mrn_token, order_id, raw(row.get("Data Source"))])) if mrn_token and order_id else "",
            content_signature=full, cross_protocol_signature=signature(row, secret, exclude_protocol=True),
            order_type_raw=raw(row.get("Order Type")), order_status_raw=raw(row.get("Order Status")),
            data_source_raw=raw(row.get("Data Source")), order_set_raw=raw(row.get("Order Set Name")),
            take_home_raw=raw(row.get("Take Home Indicator")),
            category_raw=raw(row.get("Category")), drug_type_raw=raw(row.get("Drug Type")),
            refills_raw=raw(row.get("Order No. of Refills")),
            order_semantic_class="unvalidated_observed_value", matching_confidence="exact_crosswalk" if patient else "unvalidated",
            time_semantics="order_or_start_unresolved", end_semantics="order_validity_end_not_confirmed_discontinuation",
            dose_raw=raw(row.get("Order Dose/Strength")), uom_raw=raw(row.get("Order UOM")),
            frequency_raw=raw(row.get("Order Frequency")), route_raw=raw(row.get("Order Route")),
            dose_text_normalized=" ".join(raw(row.get("Order Dose/Strength")).split()),
            frequency_text_normalized=" ".join(raw(row.get("Order Frequency")).casefold().split()),
            route_text_normalized=" ".join(raw(row.get("Order Route")).casefold().split()),
            instruction_present=present(row.get("Order Instruction")),
            non_formulary_present=present(row.get("Non Formulary Info")),
            name_present=name_present, prescription_description_complete=prescription,
            evidence_type="documented_order", audit_only=True,
            **normalized, **date_fields(start, end), qc_flags="|".join(flags),
        ))
    return records


def duplicate_qc(orders):
    if orders.empty:
        return orders
    orders = orders.copy()
    flags = {i: set(filter(None, row.qc_flags.split("|"))) for i, row in orders.iterrows()}
    exact = orders.duplicated(["source", "content_signature"], keep=False)
    for i in orders.index[exact]:
        flags[i].add("exact_reexport_duplicate_retained")
    for _, group in orders[orders.order_identity.ne("")].groupby(["source", "order_identity"]):
        if group.content_signature.nunique() > 1:
            for i in group.index:
                flags[i].add("same_order_id_revision_or_discordance_retained")
    # Same valid source MRN and order ID are required for possible cross-export matches.
    validated = orders[orders.order_id_pseudonymous.ne("") & orders.linkage_status.eq("linked_one_to_one")]
    for _, group in validated.groupby(["source_patient_pseudonymous", "order_id_pseudonymous", "data_source_raw"]):
        if group.source.nunique() > 1:
            for i in group.index:
                flags[i].add("cross_protocol_possible_duplicate_retained")
                if group.cross_protocol_signature.nunique() == 1:
                    flags[i].add("cross_protocol_equal_except_protocol")
    orders["qc_flags"] = ["|".join(sorted(flags[i])) for i in orders.index]
    return orders


def interval_qc(records):
    """Flag overlapping documented bounds without deriving clinical courses."""
    if records.empty:
        return records
    records = records.copy()
    flags = {i: set(filter(None, row.qc_flags.split("|"))) for i, row in records.iterrows()}
    bounded = records.loc[records.start_status.eq("valid") & records.end_status.eq("valid") &
                          records.start_upper_bound.le(records.end_lower_bound) &
                          records.source_patient_pseudonymous.ne("") & records.drug_normalized.ne("")]
    for _, group in bounded.groupby(["source", "source_patient_pseudonymous", "drug_normalized", "formulation"]):
        max_end, previous = "", None
        for i, row in group.sort_values("start_lower_bound").iterrows():
            if max_end and row.start_lower_bound <= max_end:
                flags[i].add("documented_intervals_overlap_requires_review")
                flags[previous].add("documented_intervals_overlap_requires_review")
            if row.end_upper_bound >= max_end:
                max_end, previous = row.end_upper_bound, i
    records["qc_flags"] = ["|".join(sorted(flags[i])) for i in records.index]
    return records




CANONICAL_COLUMNS = [
    "audit_record_id", "source_kind", "source", "source_protocol_folder", "protocol",
    "protocol_number_raw", "source_row", "source_row_or_order_id", "patient_id_pseudonymous",
    "source_patient_pseudonymous", "linked_patient_pseudonymous", "mrn_identity_quality", "eligible_linked", "name_present",
    "drug_raw", "ingredient_raw", "drug_normalized", "ingredient_normalized", "brand", "formulation",
    "route", "drug_class", "mapping_rule", "mapping_confidence", "mapping_version", "indication",
    "normalization_flags", "order_status_raw", "time_semantics", "end_semantics", "dose_raw", "uom_raw",
    "order_id_pseudonymous", "order_identity", "content_signature", "cross_protocol_signature", "instruction_present",
    "frequency_raw", "route_raw", "evidence_type", "prescription_description_complete", "qc_flags", "audit_only",
] + [f"{prefix}_{name}" for prefix in ("start", "end") for name in DATE_KEYS]


def records_frame(records):
    if records:
        return pd.DataFrame(records)
    result = pd.DataFrame(columns=CANONICAL_COLUMNS)
    for column in ("eligible_linked", "name_present", "instruction_present", "prescription_description_complete"):
        result[column] = result[column].astype(bool)
    return result

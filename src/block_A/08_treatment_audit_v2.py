#!/usr/bin/env python3
"""Read-only CTDB/BTRIS medication audit. Orders are not confirmed exposure.

Version 1 is deliberately left intact. Run this entry point for version 2.
Only the explicitly configured sources below are read; no downstream imports.
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

# Reuse the literal CTDB schema and category codes, without changing v1's rules.
_spec = importlib.util.spec_from_file_location(
    "_step08_v1", Path(__file__).with_name("08_treatment_audit.py"))
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

RULES_VERSION = "08-treatment-audit-2.0.0"
MAPPING_VERSION = "08-drug-candidates-2.0.0"
DEFAULT_OUTPUT = v1.common.OUTPUTS_DIR / "audit" / "08_treatments_v2"
BTRIS_MEDICATION_PATHS = {
    "11D": Path("/data/salazarda/data/eda_sjd/data_analytic/BTRIS/11D/medication_records.parquet"),
    "15D": Path("/data/salazarda/data/eda_sjd/data_analytic/BTRIS/15D/medication_records.parquet"),
}
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
        if re.fullmatch(r"\\d+\\.0+", text.strip()):
            text, representation = text.strip().split(".", 1)[0], "text_integer_decimal"
    match = re.sub(r"^0+", "", re.sub(r"[-/\\\\\\s]", "", text.strip()))
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


def parse_date(value, *, slash_order="MDY"):
    """Retain uncertainty bounds separately from observed dates; strict formats."""
    result = dict(raw=raw(value), parsed="", precision="", lower_bound="", upper_bound="",
                  parse_rule="", status="missing")
    if not present(value):
        return result
    text = raw(value).strip()
    try:
        if isinstance(value, datetime):
            result.update(parsed=value.isoformat(), precision="datetime", parse_rule="native_datetime")
            lower = upper = value.date()
        elif isinstance(value, date):
            result.update(parsed=value.isoformat(), precision="day", parse_rule="native_date")
            lower = upper = value
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}.*", text):
            stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
            result.update(parsed=stamp.isoformat(), precision="datetime", parse_rule="ISO_datetime")
            lower = upper = stamp.date()
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            lower = upper = date.fromisoformat(text)
            result.update(parsed=lower.isoformat(), precision="day", parse_rule="YYYY-MM-DD")
        elif re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text):
            if slash_order != "MDY":
                result.update(status="ambiguous_format", parse_rule="slash_format_not_authorized")
                return result
            lower = upper = datetime.strptime(text, "%m/%d/%Y").date()
            result.update(parsed=lower.isoformat(), precision="day", parse_rule="MM/DD/YYYY")
        elif re.fullmatch(r"\d{1,2}/\d{4}", text) or re.fullmatch(r"\d{4}-\d{2}", text):
            if "/" in text:
                month, year = map(int, text.split("/"))
                rule = "MM/YYYY"
            else:
                year, month = map(int, text.split("-"))
                rule = "YYYY-MM"
            lower = date(year, month, 1)
            upper = date(year, month, calendar.monthrange(year, month)[1])
            result.update(precision="month", parse_rule=rule)
        elif re.fullmatch(r"\d{4}", text):
            year = int(text)
            lower, upper = date(year, 1, 1), date(year, 12, 31)
            result.update(precision="year", parse_rule="YYYY")
        else:
            result.update(status="unsupported_format", parse_rule="no_explicit_rule")
            return result
        result.update(lower_bound=lower.isoformat(), upper_bound=upper.isoformat(), status="valid")
    except (ValueError, OverflowError):
        result.update(parsed="", precision="", status="invalid_date", parse_rule="invalid_calendar_date")
    return result


def compare_dates(left, right):
    if left["status"] != "valid" or right["status"] != "valid":
        return "indeterminate"
    return ("compatible" if max(left["lower_bound"], right["lower_bound"]) <=
            min(left["upper_bound"], right["upper_bound"]) else "incompatible")


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


def file_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=v1.file_hash(path))


def load_source(path, expected, *, parquet_only=False, required_columns=None):
    """Count rows and audit optional columns without blocking a usable source.

    All headers remain audited; only missing indispensable columns block input.
    """
    required_columns = list(expected) if required_columns is None else list(required_columns)
    path = Path(path)
    info = dict(path=str(path.resolve()), status="BLOCKED_SOURCE_UNAVAILABLE", n_rows=None,
                missing_columns=list(expected), unexpected_columns=[], error="source_not_accessible")
    try:
        info.update(file_fingerprint(path))
        if path.suffix.lower() == ".parquet":
            metadata = pq.ParquetFile(path)
            info.update(n_row_groups=metadata.num_row_groups, metadata_rows=metadata.metadata.num_rows,
                        arrow_schema={f.name: str(f.type) for f in metadata.schema_arrow},
                        physical_schema={metadata.schema.column(i).path: metadata.schema.column(i).physical_type
                                         for i in range(len(metadata.schema))})
            frame = metadata.read().to_pandas()
        elif not parquet_only and path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, dtype=object, keep_default_na=False)
        else:
            raise ValueError("unsupported_source_format")
        if not frame.columns.is_unique:
            raise ValueError("duplicate_column_headers")
        info.update(n_rows=len(frame), missing_columns=[c for c in expected if c not in frame],
                    missing_required_columns=[c for c in required_columns if c not in frame],
                    unexpected_columns=[c for c in frame if c not in expected], error="")
        info["status"] = ("BLOCKED_SCHEMA" if info["missing_required_columns"]
                          else "available_partial_schema" if info["missing_columns"]
                          else "available")
        return frame, info
    except (OSError, ValueError, TypeError, KeyError) as exc:
        info.update(status="BLOCKED_SOURCE_UNAVAILABLE", n_rows=None,
                    error=f"{type(exc).__name__}: source_missing_unreadable_or_corrupt")
        return None, info


def schema_audit(frame, info, expected, source):
    rows = []
    for column in list(expected) + info.get("unexpected_columns", []):
        exists = frame is not None and column in frame
        values = frame[column] if exists else None
        rows.append(dict(
            source=source, column=column, expected_column=column in expected,
            column_present=exists if frame is not None else None,
            schema_status="expected" if exists and column in expected else "unexpected_column" if exists else info["status"],
            impact=("missing_exact_header_no_alias_substituted" if not exists else ""),
            dtype=str(values.dtype) if exists else "",
            physical_dtype=info.get("physical_schema", {}).get(column, ""),
            logical_dtype=info.get("arrow_schema", {}).get(column, ""),
            n_null=int(values.isna().sum()) if exists else None,
            n_empty=int(values.map(lambda v: not raw(v).strip()).sum() - values.isna().sum()) if exists else None,
            n_missing_tokens=int(values.map(lambda v: not present(v)).sum()) if exists else None,
            pct_null=100 * values.isna().mean() if exists and len(values) else None,
            n_unique=int(values.dropna().map(raw).nunique()) if exists else None,
            n_unique_observed=int(values[values.map(present)].map(raw).nunique()) if exists else None,
            n_conversion_failures=(sum(parse_date(v)["status"] not in {"valid", "missing"} for v in values)
                                   if exists and column in BTRIS_TIME_FIELDS else None),
        ))
    return rows


def validate_linkage(crosswalk, eligible, *, mrn_column="MRN", patient_column="patient_id", authorized=False):
    result = dict(status="blocked_linkage", reason="CROSS_SOURCE_LINKAGE_NOT_VALIDATED",
                  mapping={}, ambiguous_mrns=set(), ambiguous_patients=set(), invalid_rows=0,
                  n_crosswalk_pairs=None, complete_eligible_mapping=False)
    if crosswalk is None or not authorized:
        return result
    if not {mrn_column, patient_column}.issubset(crosswalk.columns):
        result["reason"] = "CROSSWALK_EXACT_KEYS_MISSING"
        return result
    pairs = []
    for _, row in crosswalk.iterrows():
        mrn, patient = key(row[mrn_column], mrn=True), key(row[patient_column])
        if not mrn or not patient:
            result["invalid_rows"] += 1
        else:
            pairs.append((mrn, patient))
    links = pd.DataFrame(pairs, columns=["mrn", "patient"]).drop_duplicates()
    many_patients = links.groupby("mrn").patient.nunique()
    many_mrns = links.groupby("patient").mrn.nunique()
    bad_mrn = set(many_patients[many_patients.gt(1)].index)
    bad_patient = set(many_mrns[many_mrns.gt(1)].index)
    # Exclude every edge touching a conflicting identity, including partners.
    invalid = links.mrn.isin(bad_mrn) | links.patient.isin(bad_patient)
    result["ambiguous_mrns"] = set(links.loc[invalid, "mrn"])
    result["ambiguous_patients"] = set(links.loc[invalid, "patient"])
    result["mapping"] = dict(links.loc[~invalid, ["mrn", "patient"]].itertuples(index=False, name=None))
    result["n_crosswalk_pairs"] = len(links)
    mapped = set(result["mapping"].values())
    complete = bool(eligible) and eligible.issubset(mapped) and not result["invalid_rows"]
    result.update(complete_eligible_mapping=complete,
                  status="validated" if complete else "blocked_linkage",
                  reason="complete_one_to_one_eligible_crosswalk" if complete else "incomplete_or_ambiguous_crosswalk")
    return result


def linkage_qc(frames, ctdb, eligible, linkage, secret):
    rows = []
    for source, frame in frames.items():
        if frame is None:
            rows.append(dict(source=source, status="BLOCKED_SOURCE_UNAVAILABLE", n_rows=None))
            continue
        seen = set()
        for value in frame.get("MRN", pd.Series([None] * len(frame))):
            mrn = key(value, mrn=True)
            token, quality = mrn_audit_identity(value, secret)
            if (token, quality) in seen:
                continue
            seen.add((token, quality))
            patient = (linkage.get("normalized_mapping", {}).get(
                serology_patient_id_match(value)[0], "")
                if linkage.get("mode") == "serology_normalized_id"
                else linkage["mapping"].get(mrn, ""))
            status = ("linked_one_to_one" if patient else
                      "ambiguous_key" if mrn in linkage["ambiguous_mrns"] else
                      "unmatched_btris" if mrn else
                      "nontext_unverified" if token else "invalid_key")
            rows.append(dict(source=source, patient_id_pseudonymous=token,
                             mrn_identity_quality=quality,
                             linked_id_pseudonymous=pseudonym(secret, "patient", patient),
                             status=status, eligible=patient in eligible if patient else None,
                             cardinality_status="linked_one_to_many_requires_review"
                             if status == "ambiguous_key" else status))
    ctdb_ids = set(ctdb[CTDB_PATIENT_ID].map(key)) if ctdb is not None and CTDB_PATIENT_ID in ctdb else set()
    for patient in sorted(eligible | ctdb_ids):
        status = ("invalid_key" if not patient else "ambiguous_key" if patient in linkage["ambiguous_patients"]
                  else "linked_one_to_one" if patient in (
                      linkage.get("normalized_mapping", {}).values()
                      if linkage.get("mode") == "serology_normalized_id"
                      else linkage["mapping"].values()) else "unmatched_ctdb")
        rows.append(dict(source="CTDB", patient_id_pseudonymous=pseudonym(secret, "patient", patient),
                         status=status, eligible=patient in eligible, observed_in_ctdb=patient in ctdb_ids))
    return pd.DataFrame(rows)

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
            order_identity=pseudonym(secret, "order_identity", json.dumps([mrn, order_id, raw(row.get("Data Source"))])) if mrn and order_id else "",
            content_signature=full, cross_protocol_signature=signature(row, secret, exclude_protocol=True),
            order_type_raw=raw(row.get("Order Type")), order_status_raw=raw(row.get("Order Status")),
            data_source_raw=raw(row.get("Data Source")), order_set_raw=raw(row.get("Order Set Name")),
            take_home_raw=raw(row.get("Take Home Indicator")),
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
    for _, group in validated.groupby(["source_patient_pseudonymous", "order_id_pseudonymous"]):
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


def ctdb_records(frame, eligible, secret, *, protocol_column=None, index_column=None):
    current, corrections, doses = [], [], []
    if frame is None or CTDB_PATIENT_ID not in frame:
        return current, corrections, doses
    for position, (_, row) in enumerate(frame.iterrows()):
        patient = key(row.get(CTDB_PATIENT_ID))
        if patient not in eligible:
            continue
        patient_token = pseudonym(secret, "patient", patient)
        base = dict(source="CTDB", source_kind="CTDB", source_row=position,
                    patient_id_pseudonymous=patient_token, linked_patient_pseudonymous=patient_token,
                    source_patient_pseudonymous=patient_token, eligible_linked=True,
                    protocol=protocol(row.get(protocol_column)) if protocol_column else "protocol_provenance_unavailable",
                    protocol_number_raw=raw(row.get(protocol_column)) if protocol_column else "",
                    source_protocol_folder="CTDB", audit_only=True)
        visit = raw(row.get(CTDB_VISIT_DATE))
        visit_semantics = "concatenated_visits_no_arbitrary_date" if " | " in visit else "reported_visit_date"
        for slot, column in enumerate(CTDB_RX_NAMES, 1):
            fields = [column, CTDB_RX_ROUTES[slot-1], CTDB_RX_FREQ_BY_SLOT[slot],
                      CTDB_RX_START_BY_SLOT[slot], CTDB_RX_STOP_BY_SLOT[slot]]
            if not any(present(row.get(c)) or raw(row.get(c)).strip().casefold() in v1.NEGATIVE_TOKENS for c in fields):
                continue
            name = row.get(column)
            name_present = present(name) and raw(name).strip().casefold() not in v1.NEGATIVE_TOKENS
            route_raw = row.get(CTDB_RX_ROUTES[slot-1])
            route, route_status = v1.decode_category(route_raw, v1.ROUTE_CODES)
            frequency_raw = row.get(CTDB_RX_FREQ_BY_SLOT[slot])
            frequency, frequency_status = v1.decode_category(frequency_raw, v1.FREQUENCY_CODES)
            start, end = parse_date(row.get(CTDB_RX_START_BY_SLOT[slot])), parse_date(row.get(CTDB_RX_STOP_BY_SLOT[slot]))
            flags = date_flags(start, end)
            if not name_present:
                flags.append("no_observed_name_or_explicit_negative")
            if " | " in visit:
                flags.append("concatenated_visit_dates_temporal_linkage_unresolved")
            index = parse_date(row.get(index_column)) if index_column else parse_date(None)
            if start["status"] == index["status"] == "valid" and start["upper_bound"] < index["lower_bound"]:
                flags.append("pre_index_historical_record_retained")
            item = dict(base, audit_record_id=f"CTDB:{position}:rx:{slot}", slot=slot, source_row_or_order_id=position,
                        name_column=column, name_present=name_present, visit_raw=visit, visit_semantics=visit_semantics,
                        route_raw=raw(route_raw), frequency_raw=raw(frequency_raw), frequency_normalized=frequency,
                        route_parse_status=route_status, frequency_parse_status=frequency_status,
                        time_semantics="reported_medication_dates_unvalidated", end_semantics="reported_stop_date_unvalidated",
                        evidence_type="documented_visit_medication", dose_raw="", uom_raw="",
                        order_status_raw="not_applicable", prescription_description_complete=False,
                        **normalize_drug(name, route=route), **date_fields(start, end), qc_flags="|".join(flags))
            current.append(item)
        allergy = raw(row.get(CTDB_PMH_ALLERGY))
        for column in CTDB_PMH_RX_NAMES:
            if present(row.get(column)):
                name = raw(row[column])
                corrections.append(dict(base, field=column, value_raw=name, allergy_field_raw=allergy,
                                        source_semantics="unverified_possible_allergy", excluded_from_medication_counts=True,
                                        allergy_text_exact_match=present(allergy) and name.casefold().strip() == allergy.casefold().strip(),
                                        decision="manual_source_form_validation_required"))
        for column in CTDB_UNASSIGNED_DOSES:
            if present(row.get(column)):
                doses.append(dict(base, dose_column=column, dose_raw=raw(row[column]),
                                  slot_mapping="unresolved_not_assigned"))
    return current, corrections, doses


CANONICAL_COLUMNS = [
    "audit_record_id", "source_kind", "source", "source_protocol_folder", "protocol",
    "protocol_number_raw", "source_row", "source_row_or_order_id", "patient_id_pseudonymous",
    "source_patient_pseudonymous", "linked_patient_pseudonymous", "eligible_linked", "name_present",
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


def count_patients(records):
    return int(records.loc[records.source_patient_pseudonymous.ne(""), "source_patient_pseudonymous"].nunique())


def grouped_inventory(records, columns):
    rows = []
    if records.empty:
        return pd.DataFrame(columns=columns + ["n_records", "n_patients", "n_eligible_linked_patients"])
    for values, part in records.groupby(columns, dropna=False, sort=True):
        if len(columns) == 1:
            values = (values,) if not isinstance(values, tuple) else values
        rows.append(dict(zip(columns, values), n_records=len(part), n_patients=count_patients(part),
                         n_eligible_linked_patients=int(part.loc[part.eligible_linked, "linked_patient_pseudonymous"].nunique())))
    return pd.DataFrame(rows)


def source_profile(frames, infos, orders):
    rows = []
    for source, frame in frames.items():
        info = infos[source]
        if frame is None:
            rows.append(dict(source=source, status=info["status"], n_rows=None, n_patients=None,
                             n_orders_with_id=None, n_eligible_linked_patients=None))
            continue
        part = orders[orders.source.eq(source)]
        valid_ids = part.order_id_pseudonymous.ne("") if len(part) else pd.Series(dtype=bool)
        starts = part.loc[part.start_status.eq("valid"), "start_lower_bound"]
        ends = part.loc[part.start_status.eq("valid"), "start_upper_bound"]
        end_lowers = part.loc[part.end_status.eq("valid"), "end_lower_bound"]
        end_uppers = part.loc[part.end_status.eq("valid"), "end_upper_bound"]
        rows.append(dict(
            source=source, status=info["status"], n_rows=len(frame), n_patients=count_patients(part),
            patient_count_semantics="distinct_source_MRN_representations_not_validated_patients",
            n_nontext_unverified_mrn_rows=int(part.mrn_identity_quality.eq("nontext_unverified").sum()) if len(part) else 0,
            n_text_mrn_rows=int(part.mrn_identity_quality.eq("text_exact").sum()) if len(part) else 0,
            n_orders_with_id=int(valid_ids.sum()), n_order_ids=int(part.loc[valid_ids, "order_id_pseudonymous"].nunique()) if len(part) else 0,
            n_missing_order_id=int((~valid_ids).sum()),
            n_exact_duplicate_rows=int(part.qc_flags.str.contains("exact_reexport_duplicate_retained").sum()),
            n_order_revision_rows=int(part.qc_flags.str.contains("same_order_id_revision_or_discordance_retained").sum()),
            n_cross_protocol_possible_duplicate_rows=int(part.qc_flags.str.contains("cross_protocol_possible_duplicate_retained").sum()),
            n_eligible_linked_patients=int(part.loc[part.eligible_linked, "linked_patient_pseudonymous"].nunique()),
            n_invalid_identity_rows=int(part.patient_id_pseudonymous.eq("").sum()),
            n_protocol_mismatch_rows=int(part.protocol.ne(BTRIS_PROTOCOL_EXPECTED[source]).sum()),
            n_observed_protocol_values=int(frame["Protocol Number"].map(raw).nunique()) if "Protocol Number" in frame else None,
            start_min=starts.min() if len(starts) else "", start_max=ends.max() if len(ends) else "",
            end_min=end_lowers.min() if len(end_lowers) else "", end_max=end_uppers.max() if len(end_uppers) else "",
            unit="order_row_not_administration",
        ))
    return pd.DataFrame(rows)


def status_dictionary(frames):
    rows = []
    for source, frame in frames.items():
        if frame is None:
            continue
        for field in STATUS_FIELDS:
            if field not in frame:
                continue
            for value, n in frame[field].map(raw).value_counts(dropna=False).items():
                rows.append(dict(source=source, field=field, observed_value=value, n_records=int(n),
                                 tentative_interpretation="missing" if not present(value) else "observed_label_only_no_clinical_semantics",
                                 evidence="source_value_distribution_only", confidence="unvalidated",
                                 needs_validation=True, order_semantic_class="unvalidated_observed_value"))
    return pd.DataFrame(rows, columns=["source", "field", "observed_value", "n_records", "tentative_interpretation",
                                       "evidence", "confidence", "needs_validation", "order_semantic_class"])


def dose_qc(orders):
    rows = []
    for source, part in orders.groupby("source"):
        for metric, mask in [
            ("name", part.name_present), ("date", part.start_status.eq("valid")),
            ("route", part.route_raw.map(present)), ("dose_strength", part.dose_raw.map(present)),
            ("uom", part.uom_raw.map(present)), ("frequency", part.frequency_raw.map(present)),
            ("instruction", part.instruction_present),
            ("prescription_description_complete", part.prescription_description_complete),
            ("prescription_description_incomplete", ~part.prescription_description_complete),
        ]:
            rows.append(dict(source=source, metric=metric, n_records=int(mask.sum()), denominator=len(part),
                             percent=100 * mask.mean(), evidence_type="documented_order", administered_dose="not_established"))
    return pd.DataFrame(rows, columns=["source", "metric", "n_records", "denominator", "percent", "evidence_type", "administered_dose"])


def coverage_table(named, linkage, source_ok, eligible):
    columns = ["source", "protocol", "drug_normalized", "drug_class", "evidence_type"]
    result = grouped_inventory(named, columns)
    if linkage["status"] == "validated" and all(source_ok.values()):
        combined = named.loc[named.eligible_linked.astype(bool)].copy()
        combined["source"] = "CTDB+BTRIS"
        combined["protocol"] = "all_sources_not_protocol_cohort"
        combined["evidence_type"] = "mixed_documentation_records_not_exposure"
        combined["source_patient_pseudonymous"] = combined.linked_patient_pseudonymous
        result = pd.concat([result, grouped_inventory(combined, columns)], ignore_index=True)
    result["cross_source_status"] = ("blocked_linkage" if linkage["status"] != "validated" else
                                       "blocked_source" if not all(source_ok.values()) else "validated_documentation_only")
    return result


def annotate_availability(tables, infos, linkage, eligible):
    """Every aggregate table distinguishes inaccessible sources from no records."""
    for name, table in list(tables.items()):
        if name.endswith(".parquet") or "source" not in table or name in {
            "08_btris_source_schema_audit.csv", "08_treatment_ctdb_schema_audit.csv",
        }:
            continue
        table = table.copy()
        table["source_status"] = table.source.map(lambda s: infos.get(s, {}).get("status", "available"))
        table["linkage_validation_status"] = linkage["status"]
        if "n_eligible_linked_patients" in table:
            # CTDB is linked directly to the configured eligibility key. BTRIS
            # counts are unknown without the complete validated eligible crosswalk.
            unknown = table.source.ne("CTDB") & (linkage["status"] != "validated")
            table.loc[unknown, "n_eligible_linked_patients"] = None
            table["eligible_cohort_denominator"] = len(eligible) if eligible else None
            table["percent_eligible_cohort"] = (100 * table.n_eligible_linked_patients / len(eligible)) if eligible else None
        # Record-level QC tables remain empty when there are no records. Source
        # availability is fully reported in the schema/profile/summary products.
        tables[name] = table


def overlap_summary(ctdb, orders, eligible, linkage, source_ok, secret):
    rows = []
    for scope in ("11D", "15D", "union"):
        needed = [scope] if scope != "union" else ["11D", "15D"]
        if linkage["status"] != "validated" or not eligible or not all(source_ok.get(s) for s in ["CTDB", *needed]):
            rows.append(dict(scope=scope, status="blocked_linkage" if linkage["status"] != "validated" else "blocked_source",
                             n_ctdb_only=None, n_btris_only=None, n_both=None, n_neither=None,
                             denominator=len(eligible) if eligible else None))
            continue
        denominator = {pseudonym(secret, "patient", p) for p in eligible}
        c = set(ctdb.loc[ctdb.name_present, "linked_patient_pseudonymous"]) & denominator
        selected = orders[orders.source.isin(needed) & orders.eligible_linked]
        b = set(selected.linked_patient_pseudonymous) & denominator
        row = dict(scope=scope, status="available", n_ctdb_only=len(c-b), n_btris_only=len(b-c),
                   n_both=len(c & b), n_neither=len(denominator - (c | b)), denominator=len(denominator),
                   ctdb_definition="at_least_one_observed_current_name", btris_definition="at_least_one_order_record",
                   missing_documentation_is_not_untreated=True)
        for category in ("ctdb_only", "btris_only", "both", "neither"):
            row[f"pct_{category}"] = 100 * row[f"n_{category}"] / len(denominator)
        for metric, available in (
            ("full_start_date", lambda f: f.start_precision.isin(["day", "datetime"])),
            ("route", lambda f: f.route_raw.map(present)),
            ("frequency", lambda f: f.frequency_raw.map(present)),
            ("dose", lambda f: f.dose_raw.map(present)),
        ):
            ctokens = set(ctdb.loc[ctdb.name_present & available(ctdb), "linked_patient_pseudonymous"])
            btokens = set(selected.loc[available(selected), "linked_patient_pseudonymous"])
            row[f"n_additional_patients_{metric}"] = len((btokens - ctokens) & denominator)
        ckeys = set(ctdb.loc[ctdb.name_present & ctdb.drug_normalized.ne(""), "drug_normalized"])
        bkeys = set(selected.loc[selected.drug_normalized.ne(""), "drug_normalized"])
        row["n_additional_recognized_drugs"] = len(bkeys - ckeys)
        rows.append(row)
    return pd.DataFrame(rows)


def concordance(ctdb, orders, linkage, source_ok):
    columns = ["ctdb_record_id", "btris_record_id", "patient_id_pseudonymous", "level", "date_compatibility"]
    if linkage["status"] != "validated" or not all(source_ok.values()):
        return pd.DataFrame([dict(level="blocked_linkage" if linkage["status"] != "validated" else "blocked_source")], columns=columns)
    rows = []
    ctdb = ctdb.loc[ctdb.name_present.astype(bool)]
    orders = orders.loc[orders.eligible_linked.astype(bool)]
    matched_orders = set()
    for _, left in ctdb.iterrows():
        same_patient = orders[orders.linked_patient_pseudonymous.eq(left.linked_patient_pseudonymous)]
        same_drug = same_patient[
            same_patient.ingredient_normalized.eq(left.ingredient_normalized) &
            (same_patient.formulation.eq(left.formulation) | same_patient.formulation.eq("unresolved") |
             (left.formulation == "unresolved"))] if left.ingredient_normalized else same_patient.iloc[:0]
        if same_drug.empty:
            rows.append(dict(ctdb_record_id=left.audit_record_id, btris_record_id="",
                             patient_id_pseudonymous=left.linked_patient_pseudonymous,
                             level="source_only" if same_patient.empty else "potential_match_unresolved" if not left.drug_normalized else "discordant_drug",
                             date_compatibility="indeterminate"))
        for _, right in same_drug.iterrows():
            compatible = compare_dates({k: left[f"start_{k}"] for k in DATE_KEYS}, {k: right[f"start_{k}"] for k in DATE_KEYS})
            level = "temporally_compatible" if compatible == "compatible" else "not_comparable_no_time" if compatible == "indeterminate" else "same_patient_same_drug"
            if ("unresolved" in {left.formulation, right.formulation} or
                    "needs_review" in {left.drug_class, right.drug_class}):
                level = "potential_match_unresolved"
            rows.append(dict(ctdb_record_id=left.audit_record_id, btris_record_id=right.audit_record_id,
                             patient_id_pseudonymous=left.linked_patient_pseudonymous, level=level, date_compatibility=compatible))
            matched_orders.add(right.audit_record_id)
    for _, right in orders[~orders.audit_record_id.isin(matched_orders)].iterrows():
        rows.append(dict(ctdb_record_id="", btris_record_id=right.audit_record_id,
                         patient_id_pseudonymous=right.linked_patient_pseudonymous, level="source_only", date_compatibility="indeterminate"))
    return pd.DataFrame(rows, columns=columns)


def feasibility(orders, linkage, infos):
    rows = []
    for source in ("11D", "15D"):
        part = orders[orders.source.eq(source)]
        for gate, status, mask in (
            ("drug_identification", "candidate_dictionary_review_required", part.drug_normalized.ne("") & ~part.drug_class.isin(["unmapped", "needs_review"])),
            ("order_timeline", "blocked_order_or_start_semantics_unresolved", None),
            ("patient_linkage", linkage["status"], part.eligible_linked),
            ("actual_exposure", "pending_direct_evidence", None),
            ("treatment_episode", "not_approved", None),
            ("dose_intensity", "prescription_description_only", part.prescription_description_complete),
            ("indication", "indeterminate_no_authorized_source", None),
            ("clinical_prepost_ESSDAI", "not_evaluated_in_this_audit", None),
        ):
            available = infos[source]["status"] == "available"
            rows.append(dict(source=source, gate=gate, status=status if available else infos[source]["status"], unit="order_row",
                             n_records=int(mask.sum()) if mask is not None and available else None,
                             denominator=len(part) if available else None,
                             percent=100 * mask.mean() if mask is not None and len(part) else None))
    return pd.DataFrame(rows)


def write_private_table(path, table):
    # All vocabularies and row-level files are restricted, even if some are aggregate.
    with path.open("wb") as stream:
        os.chmod(path, 0o600)
        if path.suffix == ".parquet":
            table.to_parquet(stream, index=False)
        else:
            table.to_csv(stream, index=False)


def report_markdown(summary):
    lines = ["# Step 08 CTDB + BTRIS audit 2.0", "", f"Rules: `{RULES_VERSION}`. Audit only.", "",
             "Orders document prescribing records, not administration, dispensing, ingestion or SjD indication.", "",
             f"Eligible-patient denominator: {summary['n_eligible_patients'] if summary['n_eligible_patients'] is not None else 'unavailable'}.", "",
             "| Source | Status | Input rows | Source identity tokens (BTRIS unverified) |", "| --- | --- | ---: | ---: |"]
    for source in ("CTDB", "11D", "15D"):
        item = summary["source_counts"][source]
        lines.append(f"| {source} | {item['status']} | {item.get('n_rows')} | {item.get('n_patients')} |")
    lines += ["", "CTDB uses visit rows and current medication slots. PMH RX names remain possible allergies and are excluded.",
              "BTRIS uses order rows; exact reexports and revisions remain in the inventory. Folder is extraction provenance, not exclusive enrollment.", "",
              f"Current named CTDB slots: {summary['n_ctdb_named_slots']}. Possible-allergy slots quarantined: {summary['n_ctdb_possible_allergy_slots']}.",
              "", "## Cross-source coverage", "", "| Scope | Status | CTDB only | BTRIS only | Both | Neither documented | Denominator |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in summary["overlap"]:
        lines.append("| " + " | ".join(str(item.get(k)) for k in ["scope", "status", "n_ctdb_only", "n_btris_only", "n_both", "n_neither", "denominator"]) + " |")
    lines += ["", "Neither means neither source documents the specified record; it does not establish untreated status.",
              "A partial or ambiguous eligible crosswalk blocks combined denominators; source inventories continue.", "",
              "## Comparison with audit 1.0", "",
              "The investigator's prior CTDB figures (159 eligible patients, 476 visit rows, 321 current slots, 318 candidates, 30 patients with current names and 639 historical slots) are reference figures, not BTRIS findings or fixed denominators.",
              "Version 2 quarantines PMH RX names, parses daily MDY dates and preserves month/year bounds. Differences require review of inputs, eligibility, scope and mapping rules.", ""]
    if summary["v1_comparison"]:
        lines.extend(["| Metric | Supplied v1 | This run | Difference |", "| --- | ---: | ---: | ---: |"])
        for row in summary["v1_comparison"]:
            lines.append(f"| {row['metric']} | {row['previous']} | {row['current']} | {row['difference']} |")
    else:
        lines.append("No prior summary supplied: numeric reconciliation to the earlier run remains pending.")
    lines += ["", "## Priorities and limitations", ""]
    lines.extend(f"- {block}" for block in summary["blockers"])
    lines += ["", "No date of order is promoted to first dose; no end of validity is promoted to clinical discontinuation.",
              "No daily-dose conversion, treatment duration, efficacy, adherence, ESSDAI/ESSPRI or Mapper analysis is performed.",
              "P0: verify sources, exact schemas, identity/protocol provenance and PMH/allergy semantics.",
              "P1: review observed status labels, candidate mappings, date uncertainty and prescription completeness.",
              "P2: review coverage and the gates below before authorizing a separate integration phase.", "",
              "## Decision table", "", "| Component | Audit decision | Minimum condition |", "| --- | --- | --- |",
              "| BTRIS order inventory | Conditional | Readable sources, verified schemas and reviewed observed status semantics |",
              "| CTDB reported medications | Conditional | Validate current names and resolve PMH/allergy semantics |",
              "| Patient coverage across CTDB/BTRIS | Conditional | Authorized complete unambiguous eligible crosswalk and readable complete sources |",
              "| Treatment episodes | Not approved in this phase | Validate continuity, administration and discontinuation |",
              "| Actual first treatment date | No automatic approval | Evidence of first administration or consumption |",
              "| Received dose/adherence | Not approved | Prescription is insufficient evidence |",
              "| ESSDAI / Pop / Mapper association | Not evaluated in this audit | Separate methodological authorization |",
              "", "Recommendation: **NO-GO for downstream integration at this stage**. Resolve the documented gates and request a separate scientific review.", ""]
    return "\n".join(lines)


def run(args):
    output = args.output_dir.resolve()
    # Preliminary audit: reuse this dedicated v2 directory and overwrite only
    # explicitly named v2 artifacts, never input files or v1/master outputs.
    if output.exists() and not output.is_dir():
        raise ValueError("Output path exists and is not a directory")
    if "08_treatment_audit" in output.parts or any(p in output.parts for p in ("raw", "analytic", "intermediate")):
        raise ValueError("Use a separate protected audit 2.0 directory, outside master data and v1 outputs")
    for path in (args.input, args.eligibility, args.btris_11d, args.btris_15d):
        reject_downstream_source(path)
    expected_ctdb = [CTDB_PATIENT_ID, CTDB_VISIT_DATE, *CTDB_COLUMNS]
    for column in (args.ctdb_protocol_column, args.ctdb_index_column):
        if column and column not in expected_ctdb:
            expected_ctdb.append(column)
    frames, infos, schema = {}, {}, []
    for source, path in (("11D", args.btris_11d), ("15D", args.btris_15d)):
        frames[source], infos[source] = load_source(path, BTRIS_MEDICATION_COLUMNS, parquet_only=True)
        schema += schema_audit(frames[source], infos[source], BTRIS_MEDICATION_COLUMNS, source)
    ctdb, infos["CTDB"] = load_source(args.input, expected_ctdb,
                                     required_columns=[CTDB_PATIENT_ID, CTDB_VISIT_DATE])
    if ctdb is not None and not any(col in ctdb.columns for col in CTDB_RX_NAMES):
        infos["CTDB"]["status"] = "BLOCKED_SCHEMA"
        infos["CTDB"]["error"] = "all_current_medication_name_columns_absent"
    eligibility, infos["eligibility"] = load_source(args.eligibility, [ELIGIBILITY_PATIENT_ID])
    eligible = set(eligibility[ELIGIBILITY_PATIENT_ID].map(key)) - {""} if eligibility is not None and ELIGIBILITY_PATIENT_ID in eligibility else set()
    if not eligible:
        infos["eligibility"]["status"] = "BLOCKED_ELIGIBILITY"
    ctdb_schema = schema_audit(ctdb, infos["CTDB"], expected_ctdb, "CTDB")
    crosswalk, authorized, cw_spec = None, False, {}
    if args.crosswalk_manifest:
        manifest, manifest_info = load_manifest(args.crosswalk_manifest)
        infos["crosswalk_manifest"] = manifest_info
        if manifest is not None:
            cw_spec = manifest
            crosswalk, infos["crosswalk"] = load_source(manifest["path"], [manifest["mrn_column"], manifest["patient_column"]])
            authorized = bool(manifest["authorization_evidence"].strip())
    if args.crosswalk_manifest:
        linkage = validate_linkage(crosswalk, eligible, mrn_column=cw_spec.get("mrn_column", "MRN"),
                                   patient_column=cw_spec.get("patient_column", "patient_id"), authorized=authorized)
    elif getattr(args, "linkage_mode", "crosswalk_only") == "serology_normalized_id":
        linkage = validate_serology_style_linkage(frames, ctdb, eligible)
    else:
        linkage = validate_linkage(None, eligible)
    previous = None
    if args.previous_summary:
        infos["previous_summary"] = file_fingerprint(args.previous_summary)
        previous = json.loads(args.previous_summary.read_text())
        if not isinstance(previous, dict):
            raise ValueError("Previous summary must be a JSON object")
    if args.pseudonym_key_file:
        infos["pseudonym_key"] = file_fingerprint(args.pseudonym_key_file)
    key_path = output / "restricted" / ".pseudonym_key"
    if not args.pseudonym_key_file and key_path.exists() and not key_path.is_file():
        raise ValueError("Existing pseudonym key path is not a file")
    # Preserve the original pseudonymization key across preliminary reruns, so
    # the same patients retain the same pseudonymous IDs in compared outputs.
    if args.pseudonym_key_file:
        secret = args.pseudonym_key_file.read_bytes()
    elif key_path.is_file():
        secret = key_path.read_bytes()
    else:
        secret = os.urandom(32)
    if len(secret) < 32:
        raise ValueError("Pseudonym key must contain at least 32 bytes")
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(output, 0o700)
    restricted = output / "restricted"
    restricted.mkdir(mode=0o700, exist_ok=True)
    os.chmod(restricted, 0o700)
    if not args.pseudonym_key_file and not key_path.exists():
        with key_path.open("xb") as stream:
            os.chmod(key_path, 0o600)
            stream.write(secret)
    orders = interval_qc(duplicate_qc(records_frame([r for source in ("11D", "15D")
                                                    for r in btris_records(frames[source], source, linkage, eligible, secret)])))
    currents, corrections, doses = ctdb_records(ctdb, eligible, secret,
                                               protocol_column=args.ctdb_protocol_column, index_column=args.ctdb_index_column)
    current = interval_qc(records_frame(currents))
    for records in (orders, current):
        records["source_file"] = records.source.map(lambda s: infos[s]["path"])
        records["source_sha256"] = records.source.map(lambda s: infos[s].get("sha256", ""))
    all_records = pd.concat([orders, current], ignore_index=True)
    named = all_records.loc[all_records.name_present.fillna(False).astype(bool)].copy()
    source_ok = {source: infos[source]["status"] in {"available", "available_partial_schema"}
                 for source in ("CTDB", "11D", "15D")}
    overlaps = overlap_summary(current, orders, eligible, linkage, source_ok, secret)
    profiles = source_profile(frames, infos, orders)
    tables = {
        "08_btris_source_schema_audit.csv": pd.DataFrame(schema),
        "08_btris_source_profile.csv": profiles,
        "08_btris_order_status_dictionary.csv": status_dictionary(frames),
        "08_btris_drug_inventory.csv": grouped_inventory(orders, ["source", "protocol_number_raw", "drug_raw", "ingredient_raw", "drug_normalized", "formulation", "drug_class"]),
        "08_btris_order_date_qc.csv": orders.reindex(columns=["audit_record_id", "source", "patient_id_pseudonymous", "protocol", "order_status_raw", "time_semantics", "end_semantics", *[f"{p}_{k}" for p in ("start", "end") for k in DATE_KEYS], "qc_flags"]),
        "08_btris_dose_route_qc.csv": dose_qc(orders),
        "08_btris_orders_audit_long.parquet": orders,
        "08_treatment_patient_linkage_qc.csv": linkage_qc(frames, ctdb, eligible, linkage, secret),
        "08_treatment_ctdb_btris_overlap.csv": overlaps,
        "08_treatment_normalization_review.csv": grouped_inventory(named, ["source", "protocol", "drug_raw", "ingredient_raw", "drug_normalized", "formulation", "drug_class", "mapping_rule", "mapping_confidence", "normalization_flags"]),
        "08_treatment_ctdb_corrections_qc.csv": pd.DataFrame(corrections, columns=["source", "source_row", "patient_id_pseudonymous", "field", "value_raw", "allergy_field_raw", "source_semantics", "excluded_from_medication_counts", "allergy_text_exact_match", "decision"]),
        "08_treatment_ctdb_schema_audit.csv": pd.DataFrame(ctdb_schema),
        "08_treatment_ctdb_slots_audit_long.parquet": current,
        "08_treatment_ctdb_unassigned_doses.csv": pd.DataFrame(doses, columns=["source", "source_row", "patient_id_pseudonymous", "dose_column", "dose_raw", "slot_mapping"]),
        "08_treatment_ctdb_btris_concordance.csv": concordance(current, orders, linkage, source_ok),
        "08_treatment_combined_coverage.csv": coverage_table(named, linkage, source_ok, eligible),
        "08_treatment_date_availability.csv": grouped_inventory(named, ["source", "protocol", "patient_id_pseudonymous", "drug_normalized", "order_status_raw", "start_precision", "start_status", "end_precision", "end_status"]),
        "08_treatment_combined_feasibility.csv": feasibility(orders, linkage, infos),
    }
    annotate_availability(tables, infos, linkage, eligible)
    review = tables["08_treatment_normalization_review.csv"]
    review["review_priority"] = review.drug_class.map(lambda c: "P0_review" if c == "needs_review" else "P1_unmapped" if c == "unmapped" else "P2_candidate")
    tables["08_treatment_normalization_review.csv"] = review.sort_values(["review_priority", "n_records"], ascending=[True, False])
    for filename, table in tables.items():
        write_private_table(restricted / filename, table)
    # Hash and mtime are checked again after all source-dependent work.
    changed = []
    for name, info in infos.items():
        if "sha256" in info:
            try:
                after = file_fingerprint(info["path"])
                info["preserved"] = all(info[k] == after[k] for k in ("sha256", "mtime_ns", "size_bytes"))
            except OSError:
                info["preserved"] = False
            if not info["preserved"]:
                changed.append(name)
    blockers = [
        "P0: CROSS_SOURCE_LINKAGE_NOT_VALIDATED (does not block source-level audit)" if linkage["status"] != "validated" else "P0: Complete eligible crosswalk supplied; retain source identity/protocol QC.",
        "P0: PMH RX slots are quarantined as unverified possible allergies; do not count as treatment.",
        "P1: All observed BTRIS statuses require source-owner semantic validation; no status is interpreted as administered exposure.",
        "P1: Candidate drug mappings, ingredient conflicts and unmapped vocabulary require review.",
        "P1: CTDB dose-to-slot mapping is unresolved; doses are exported separately.",
        "P2: Actual exposure, treatment episodes, indication and pre/post clinical analyses are not approved by this audit.",
    ]
    for source in ("CTDB", "11D", "15D", "eligibility"):
        if infos[source]["status"] not in {"available", "available_partial_schema"}:
            blockers.append(f"P0: {source} {infos[source]['status']}; unavailable information is not zero treatment.")
        elif infos[source]["status"] == "available_partial_schema":
            blockers.append(f"P1: {source} optional columns missing; inspect schema QC without treating absence as zero.")
    if any(orders.mrn_identity_quality.eq("nontext_unverified")):
        blockers.append("P0: BTRIS MRN contains non-text representations. Counted for source inventory only; no patient linkage without validated crosswalk and identifier precision review.")
    if changed:
        blockers.append("P0: SOURCE_CHANGED_DURING_AUDIT; all derived results require rerun.")
    source_counts = {source: {k: clean_json(v) for k, v in profiles[profiles.source.eq(source)].iloc[0].to_dict().items()
                               if k in {"status", "n_rows", "n_patients", "n_eligible_linked_patients",
                                        "patient_count_semantics", "n_nontext_unverified_mrn_rows",
                                        "n_text_mrn_rows"}} for source in ("11D", "15D")}
    source_counts["CTDB"] = dict(status=infos["CTDB"]["status"], n_rows=infos["CTDB"]["n_rows"],
                                 n_patients=int(ctdb[CTDB_PATIENT_ID].map(key).replace("", pd.NA).nunique()) if ctdb is not None and CTDB_PATIENT_ID in ctdb else None)
    if linkage["status"] != "validated":
        for source in ("11D", "15D"):
            source_counts[source]["n_eligible_linked_patients"] = None
    named_ctdb = current.loc[current.name_present.astype(bool)]
    comparison = []
    if previous is not None:
        for metric, value in dict(n_eligible_patients=len(eligible) if eligible else None,
                                  n_current_slots=len(current) if ctdb is not None and eligible else None,
                                  n_candidate_observations=len(named_ctdb) if ctdb is not None and eligible else None,
                                  n_distinct_patients_with_names=count_patients(named_ctdb) if ctdb is not None and eligible else None,
                                  n_historical_slots=len(corrections) if ctdb is not None and eligible else None).items():
            old = previous.get(metric)
            comparison.append(dict(metric=metric, previous=old if isinstance(old, (int, float)) else None,
                                   current=value, difference=value-old if value is not None and isinstance(old, (int, float)) else None))
    summary = dict(
        rules_version=RULES_VERSION, mapping_version=MAPPING_VERSION, audit_only=True,
        status="blocked" if changed or not all(source_ok.values()) or not eligible else "completed_with_review_gates",
        n_eligible_patients=len(eligible) if eligible else None, source_counts=source_counts,
        ctdb_schema_qc=dict(missing_optional_columns=infos["CTDB"].get("missing_columns", []),
                            missing_required_columns=infos["CTDB"].get("missing_required_columns", [])),
        audit_completion_note="Source-level audit can complete while clinical linkage remains unvalidated.",
        n_ctdb_named_slots=len(named_ctdb) if ctdb is not None and eligible else None,
        n_ctdb_possible_allergy_slots=len(corrections) if ctdb is not None and eligible else None,
        n_ctdb_patients_with_current_names=count_patients(named_ctdb) if ctdb is not None and eligible else None,
        eligibility_qc=dict(
            n_rows=infos["eligibility"]["n_rows"],
            n_missing_keys=int(eligibility[ELIGIBILITY_PATIENT_ID].map(key).eq("").sum()) if eligibility is not None and ELIGIBILITY_PATIENT_ID in eligibility else None,
            n_duplicate_valid_keys=int(eligibility.loc[eligibility[ELIGIBILITY_PATIENT_ID].map(key).ne(""), ELIGIBILITY_PATIENT_ID].map(key).duplicated().sum()) if eligibility is not None and ELIGIBILITY_PATIENT_ID in eligibility else None,
            n_ctdb_eligible_visit_rows=int(ctdb[CTDB_PATIENT_ID].map(key).isin(eligible).sum()) if ctdb is not None and CTDB_PATIENT_ID in ctdb and eligible else None,
            n_ctdb_rows_outside_eligibility=int((~ctdb[CTDB_PATIENT_ID].map(key).isin(eligible)).sum()) if ctdb is not None and CTDB_PATIENT_ID in ctdb and eligible else None),
        linkage={k: v for k, v in linkage.items()
                 if k not in {"mapping", "normalized_mapping", "ambiguous_mrns", "ambiguous_patients"}},
        n_ambiguous_mrn_keys=len(linkage["ambiguous_mrns"]), n_ambiguous_patient_keys=len(linkage["ambiguous_patients"]),
        overlap=clean_json(overlaps.to_dict(orient="records")), v1_comparison=comparison, blockers=blockers,
        gates=dict(actual_exposure="pending_direct_evidence", treatment_episode="not_approved",
                   clinical_prepost_ESSDAI="not_evaluated_in_this_audit", downstream_integration="NO_GO"),
        units=dict(BTRIS="order_row", CTDB="visit_row_and_current_medication_slot", overlap="unique_eligible_patient"),
        btris_union=dict(
            n_order_rows=len(orders) if all(frames[s] is not None for s in ("11D", "15D")) else None,
            n_distinct_observed_mrn_tokens=int(orders.loc[orders.source_patient_pseudonymous.ne(""), "source_patient_pseudonymous"].nunique()) if all(frames[s] is not None for s in ("11D", "15D")) else None,
            n_eligible_linked_patients=int(orders.loc[orders.eligible_linked, "linked_patient_pseudonymous"].nunique()) if linkage["status"] == "validated" and all(source_ok[s] for s in ("11D", "15D")) else None,
            unit_caution="distinct_MRN_tokens_are_not_a_validated_full_universe_patient_union"),
        input_fingerprints={name: dict(status=info.get("status"), sha256=info.get("sha256"))
                            for name, info in infos.items() if name != "pseudonym_key"},
        restrictions=dict(no_downstream_reads=True, no_downstream_writes=True, no_master_data_changes=True,
                          no_clinical_outcome_analysis=True, no_v1_output_changes=True, row_level_files="restricted/"),
        artifacts={name: dict(classification="restricted", rows=len(table)) for name, table in tables.items()},
        input_manifest="restricted/08_treatment_input_manifest.json",
    )
    write_json(restricted / "08_treatment_input_manifest.json", dict(inputs=infos, crosswalk_configuration=cw_spec,
                ctdb_default_substitution=dict(previous_prefix="/vf/users/salazarda/data/obj1_sjd/data/raw/",
                                               selected=str(args.input.resolve()),
                                               reason="explicit CLI input or common.DEFAULT_ANALYTIC_DATASET; no fallback search")))
    write_json(output / "08_treatment_audit_v2_summary.json", summary)
    with (output / "08_treatment_audit_v2_report.md").open("w") as stream:
        os.chmod(stream.name, 0o600)
        stream.write(report_markdown(summary))
    print(f"Audit 2.0: {summary['status']}; linkage={summary['linkage']['status']}; output (overwrite enabled): {output}")
    return summary


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path, value):
    with path.open("w") as stream:
        os.chmod(path, 0o600)
        json.dump(clean_json(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_manifest(path):
    info = dict(path=str(path.resolve()), status="blocked_linkage", error="crosswalk_manifest_unavailable")
    try:
        info.update(file_fingerprint(path))
        manifest = json.loads(path.read_text())
        for field in ("path", "mrn_column", "patient_column", "authorization_evidence"):
            if not isinstance(manifest.get(field), str) or not manifest[field].strip():
                raise ValueError("Crosswalk manifest requires path, exact key columns and authorization_evidence")
        source = Path(manifest["path"])
        manifest["path"] = str((path.resolve().parent / source).resolve())
        reject_downstream_source(Path(manifest["path"]))
        info["status"] = "available"
        return manifest, info
    except (OSError, ValueError, AttributeError):
        info.update(status="blocked_linkage", error="invalid_explicit_crosswalk_manifest")
        return None, info


def reject_downstream_source(path):
    parts = Path(path).resolve().parts
    if any(re.match(r"^(10_|14_|15_)", part) for part in parts) or "studies" in parts:
        raise ValueError("Audit 2.0 does not read downstream Step 10/14/15 or study sources")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=v1.common.DEFAULT_ANALYTIC_DATASET)
    parser.add_argument("--eligibility", type=Path, default=v1.common.LONGITUDINAL_ID_LIST)
    parser.add_argument("--btris-11d", type=Path, default=BTRIS_MEDICATION_PATHS["11D"])
    parser.add_argument("--btris-15d", type=Path, default=BTRIS_MEDICATION_PATHS["15D"])
    parser.add_argument("--crosswalk-manifest", type=Path, help="Explicit authorized MRN-to-eligible-patient crosswalk configuration; overrides normalized matching")
    parser.add_argument("--linkage-mode", choices=["serology_normalized_id", "crosswalk_only"],
                        default="serology_normalized_id",
                        help="Use Step 01 collision-checked patient ID normalization for audit-only matching (default), or require an explicit crosswalk")
    parser.add_argument("--ctdb-protocol-column", help="Existing documented provenance column; never inferred from form presence")
    parser.add_argument("--ctdb-index-column", help="Existing documented index date; pre-index history is retained")
    parser.add_argument("--previous-summary", type=Path, help="Optional read-only audit 1.0 summary for numeric reconciliation")
    parser.add_argument("--pseudonym-key-file", type=Path, help="Existing protected >=32-byte key; otherwise a run-local key is generated")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Audit blocked: {type(exc).__name__}; verify explicit paths, keys and write permissions.\n")
    return 2 if summary["status"] == "blocked" else 0


if __name__ == "__main__":
    sys.exit(main())

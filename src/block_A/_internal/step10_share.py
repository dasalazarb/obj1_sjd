"""Controlled Step-10 column ordering and pseudonymized, review-only CSV export.

The share candidate is NOT a HIPAA Safe Harbor / Expert Determination release.
Use only after the applicable IRB, DUA, and disclosure-risk review.
"""
from __future__ import annotations

import re
import secrets

import pandas as pd

SHARE_DROP_COLUMNS = {
    "clinical_episode_id", "clinical_visit", "visit_type",
    "clinical_baseline_episode_id", "episode_start_date", "episode_end_date",
    "is_clinical_baseline", "spine__sjd_ever_1_2_4",
}

STRUCTURE_ORDER = [
    "patient_id", "clinical_episode_id", "clinical_visit_number",
    "clinical_anchor_date", "clinical_baseline_date",
    "time_since_clinical_baseline_days", "time_since_clinical_baseline_years",
    "clinical_visit", "visit_type", "clinical_baseline_episode_id",
    "episode_start_date", "episode_end_date", "is_clinical_baseline",
]
BLOCKS = [
    "spine__", "demo__", "pop__", "essdai__", "esspri__", "ovl__",
    "ext__", "sero__", "lab__", "pro__", "medication__",
]
EXTENDED_BLOCKS = ["sicca_", "ocular_", "salivary_", "biopsy_", "sgus_", "sjddi_"]
RESTRICTED_NAME = re.compile(
    r"(?:^|[_])(?:mrn|ssn|dob|birth|death|record_number|subject_number|"
    r"first_name|last_name|full_name|email|phone|address|street|postal|"
    r"zip|zipcode|geocode|latitude|longitude|site|institution|"
    r"hla|genotype|allele|calendar_year)(?:[_]|$)", re.I
)
DATE_NAME = re.compile(r"(?:^|[_])(?:date|datetime|timestamp)(?:[_]|$)", re.I)
SAFE_INTERVAL = re.compile(
    r"^(?:Natural History Protocol 478 Interval|Phase\s*\d+[^|;\r\n]{0,85}|"
    r"baseline|year[_ -]?\d+)$", re.I
)
CATEGORICAL_ALLOWED = {
    "demo__sex": {"f", "m", "female", "male", "other", "unknown", "not reported"},
    "pop__status": {"pop1", "pop2", "pop3", "unclassifiable", "unknown"},
    "ovl__status": {"overlap", "glandular_only", "extraglandular_only", "neither", "unclassifiable"},
    "spine__interval_name": None,
}


def harmonize_interval_names(values: pd.Series) -> pd.Series:
    """Normalize explicit compound labels, without deriving a visit from its date."""
    cleaned = values.astype("string").str.strip().mask(lambda x: x.eq(""))
    def one(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        label = str(value)
        if re.search(r"Natural\s+History\s+Protocol\s+478\s+Interval", label, re.I):
            return "Natural History Protocol 478 Interval"
        if re.search(r"Optional\s+Evaluation", label, re.I):
            without_optional = re.sub(r"Optional\s+Evaluation", "", label, flags=re.I)
            phases = re.findall(r"\bPhase\s*\d+[^|;\r\n]*", without_optional, flags=re.I)
            phases = [p.strip(" -:/|;,") for p in phases if p.strip(" -:/|;,")]
            unique = list(dict.fromkeys(phases))
            if len(unique) == 1:
                return re.sub(r"^phase\b", "Phase", unique[0], flags=re.I)
            # Ambiguous/missing phase: preserve the input rather than invent one.
        return label
    return cleaned.map(one).astype("string")


def _column_order_key(column: str) -> tuple:
    if column in STRUCTURE_ORDER:
        return (0, STRUCTURE_ORDER.index(column), "", column)
    for n, prefix in enumerate(BLOCKS, start=1):
        if column.startswith(prefix):
            suffix = column[len(prefix):]
            if prefix == "essdai__" and suffix == "total":
                return (n, -1, "", column)
            if prefix == "esspri__":
                fixed = {"total": 0, "dryness": 1, "fatigue": 2, "pain": 3}
                if suffix in fixed:
                    return (n, fixed[suffix], "", column)
            if prefix == "pop__" and suffix == "status":
                return (n, -1, "", column)
            if prefix == "ext__":
                group = next((i for i, p in enumerate(EXTENDED_BLOCKS) if suffix.startswith(p)), len(EXTENDED_BLOCKS))
                return (n, group, suffix, column)
            return (n, 0, suffix, column)
    return (99, 0, column, column)


def organize_analytic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic grouping; does not alter data, rows or column names."""
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names in Step-10 analytic data")
    return frame.loc[:, sorted(frame.columns, key=_column_order_key)].copy()


def _sensitive_column(name: str) -> bool:
    low = name.lower()
    if name in SHARE_DROP_COLUMNS:
        return True
    if low != "patient_id" and (low.endswith(("_id", "__id", "_key"))
                                or "episode_id" in low or "identifier" in low):
        return True
    if RESTRICTED_NAME.search(low) or any(t in low for t in ("free_text", "__text", "_notes", "_narrative")):
        return True
    if low in {"demo__race", "demo__ethnicity"}:
        return True  # avoid unnecessarily identifying sparse demographic strata
    if low.startswith("medication__") and low.endswith("names"):
        return True  # potentially rare combinations; may need a reviewed release
    if "protocol" in low and low != "spine__interval_name":
        return True
    if low in {"calendar_year", "year", "study_year"}:
        return True
    return False


def _validated_categorical(name: str, values: pd.Series) -> tuple[pd.Series, int]:
    data = values.astype("string").str.strip()
    if name == "spine__interval_name":
        data = harmonize_interval_names(data)
        valid = data.str.fullmatch(SAFE_INTERVAL.pattern, case=False).fillna(True)
    else:
        valid = data.str.casefold().isin(CATEGORICAL_ALLOWED[name]) | data.isna()
    withheld = int((~valid).sum())
    return data.where(valid, pd.NA), withheld


def build_share_dataset(analytic: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create an isolated, pseudonymized candidate without exposing an ID crosswalk.

    Dates receive a *consistent per-patient* 100-year-plus-random-day offset.
    This preserves within-patient intervals, but DOES NOT make dates HIPAA-safe.
    Text, unreviewed categories, rare genetic fields and source identifiers are
    omitted by default; an approved recipient/risk assessment is still required.
    """
    if "patient_id" not in analytic or analytic.patient_id.isna().any():
        raise ValueError("Share export requires fully populated patient_id")
    if not analytic.columns.is_unique:
        raise ValueError("Share export requires unique column names")
    original_ids = analytic.patient_id.astype("string")
    unique_ids = original_ids.unique().tolist()
    id_map = {old: "SJD_" + secrets.token_hex(16) for old in unique_ids}
    # Reuse one date offset for all rows and date variables of each patient.
    day_map = {old: 36525 + secrets.randbelow(731) for old in unique_ids}
    per_row_shift = original_ids.map(day_map).astype("int64")
    share = pd.DataFrame(index=analytic.index)
    share["patient_id"] = original_ids.map(id_map).astype("string")
    dropped = []
    shifted = []
    category_masked = {}
    age_topcoded = {}

    for column in organize_analytic_columns(analytic).columns:
        if column == "patient_id":
            continue
        if _sensitive_column(column):
            dropped.append(column)
            continue
        values = analytic[column]
        is_date = bool(DATE_NAME.search(column)) or pd.api.types.is_datetime64_any_dtype(values)
        if is_date:
            parsed = pd.to_datetime(values, errors="coerce")
            if parsed.notna().sum() != values.notna().sum():
                dropped.append(column)  # unsafe/unparseable date fragments
                continue
            shifted_values = parsed + pd.to_timedelta(per_row_shift, unit="D")
            share[column] = shifted_values.dt.strftime("%Y-%m-%d").astype("string")
            shifted.append(column)
            continue
        if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
            if column.startswith("demo__age") or re.search(r"(?:^|__)age(?:_|$)", column):
                numeric = pd.to_numeric(values, errors="coerce")
                share[column] = numeric.clip(upper=90)
                age_topcoded[column] = int(numeric.ge(90).sum())
            else:
                share[column] = values
            continue
        if column in CATEGORICAL_ALLOWED:
            cleaned, count = _validated_categorical(column, values)
            share[column] = cleaned
            if count:
                category_masked[column] = count
            continue
        # No uncontrolled strings are exported, even from curated sources.
        dropped.append(column)

    share = organize_analytic_columns(share)
    if len(share) != len(analytic) or share.patient_id.nunique() != len(unique_ids):
        raise AssertionError("Share export lost episodes or merged patients")
    if "clinical_visit_number" in share and share.duplicated(["patient_id", "clinical_visit_number"]).any():
        raise AssertionError("Share export has duplicate patient/visit rows")
    if any(c in share for c in SHARE_DROP_COLUMNS):
        raise AssertionError("Forbidden share columns survived")
    if any(pd.api.types.is_datetime64_any_dtype(share[c]) for c in share):
        raise AssertionError("Unshifted datetime column in share export")
    qc = {
        "status": "PSEUDONYMIZED_CANDIDATE_NOT_ANONYMIZED",
        "requires_irb_dua_and_disclosure_risk_review": True,
        "patient_id_method": "cryptographically random per run; no mapping saved",
        "date_method": "same offset per patient: 36525 plus 0..730 random days; exact day intervals preserved",
        "not_hipaa_safe_harbor": True,
        "n_rows": int(len(share)), "n_patients": int(len(unique_ids)),
        "n_source_columns": int(len(analytic.columns)), "n_share_columns": int(len(share.columns)),
        "removed_columns": sorted(dropped),
        "shifted_date_columns": sorted(shifted),
        "categorical_values_masked_counts": category_masked,
        "age_90_plus_topcoded_counts": age_topcoded,
        "note": "Check small-cell combinations, rare phenotypes and potential linkage before any disclosure.",
    }
    return share, qc

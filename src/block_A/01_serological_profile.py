#!/usr/bin/env python3
"""Build the canonical patient/clinical-episode laboratory layer.

All clinical semantics and episode matching come from EDA steps 20 and 20b.
This module selects among already-matched observations; it never reads raw
BTRIS data, reconstructs visits, invents clinical thresholds, or imputes labs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402

LOG = logging.getLogger(__name__)
KEY = ["patient_id", "clinical_episode_id"]
WIDE_SPINE_COLUMNS = [
    "patient_id",
    "clinical_episode_id",
    "clinical_anchor_date",
    "clinical_visit_number",
    "clinical_visit",
    "visit_type",
    "episode_start_date",
    "episode_end_date",
    "clinical_baseline_episode_id",
    "clinical_baseline_date",
    "is_clinical_baseline",
    "time_since_clinical_baseline_days",
    "time_since_clinical_baseline_years",
]
MATCH_KEY = ["_patient_id_match", "clinical_episode_id"]
ANALYTE_KEY = KEY + ["canonical_analyte"]
DEFAULT_LABS = Path(
    "/data/salazarda/data/eda_sjd/data_analytic/BTRIS/20_btris_lab_records_long.parquet"
)
DEFAULT_BASELINE = Path(
    "/data/salazarda/data/eda_sjd/data_analytic/BTRIS/20b_baseline_labs_patient_level.parquet"
)
BASELINE_FEATURES = [
    "baseline_anti_ro_ssa",
    "baseline_anti_la_ssb",
    "baseline_ana",
    "baseline_rf",
    "baseline_cryoglobulinemia",
    "baseline_low_c4",
    "baseline_leukopenia",
]
POSITIVE = re.compile(r"\b(positive|pos|reactive|detected|present)\b", re.I)
NEGATIVE = re.compile(
    r"\b(negative|neg|non[- ]?reactive|not detected|absent|none detected)\b", re.I
)
LOW = re.compile(r"\b(low|below)\b", re.I)
HIGH = re.compile(r"\b(high|above)\b", re.I)
ABNORMAL = re.compile(r"\b(abnormal)\b", re.I)
NORMAL = re.compile(r"\b(normal|within (?:the )?(?:reference )?range)\b", re.I)
INVALID_MAPPING = re.compile(
    r"unmapped|invalid|excluded|reject|no[_ ]?mapping|ambiguous", re.I
)
QC_DETAIL = [
    "patient_id",
    "clinical_episode_id",
    "clinical_anchor_date",
    "canonical_analyte",
    "lab_family",
    "lab_date",
    "days_from_clinical_anchor",
    "result_raw",
    "result_numeric_exact",
    "result_operator",
    "result_numeric_bound",
    "result_text",
    "unit",
    "reference_range_raw",
    "reported_interpretation",
    "order_identifier",
    "specimen_datetime",
    "assay",
    "reason_for_review",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labs-long", type=Path, default=DEFAULT_LABS)
    parser.add_argument("--baseline-labs", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--all-spine", type=Path, default=common.EPISODE_SPINE_PARQUET)
    parser.add_argument(
        "--spine", type=Path, default=common.CLINICAL_VISIT_SPINE_PARQUET
    )
    parser.add_argument(
        "--analyte-output",
        type=Path,
        default=common.BLOCKA_INTERMEDIATE_DATA_DIR
        / "01_labs_episode_analyte_level.parquet",
    )
    parser.add_argument(
        "--wide-output",
        type=Path,
        default=common.BLOCKA_INTERMEDIATE_DATA_DIR / "01_labs_episode_wide.parquet",
    )
    parser.add_argument(
        "--serology-output",
        type=Path,
        default=common.BLOCKA_INTERMEDIATE_DATA_DIR
        / "01_serology_episode_level.parquet",
    )
    return parser.parse_args(argv)


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    return (
        series.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "y"})
    )


def _normalize_patient_id(series: pd.Series) -> pd.Series:
    """Reproduce the EDA patient-ID normalization used at the integration boundary."""
    normalized = (
        series.astype("string")
        .str.strip()
        .str.replace(r"[-/\\\s]", "", regex=True)
        .str.replace(r"^0+", "", regex=True)
    )
    return normalized.mask(normalized.isin(["", "nan", "None"]))


def _column(frame: pd.DataFrame, name: str, default: Any = pd.NA) -> pd.Series:
    return frame[name] if name in frame else pd.Series(default, index=frame.index)


def _coerce_analyte_output_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the stable, semantic schema for the analyte-level artifact."""
    out = frame.copy()
    integer_columns = [
        "clinical_visit_number",
        "selected_days_from_clinical_anchor",
        "n_measurements_in_episode",
        "n_valid_measurements_in_episode",
    ]
    numeric_columns = [
        "selected_value_numeric",
        "selected_numeric_bound",
        "selected_reference_low",
        "selected_reference_high",
        "selected_reference_bound",
        "episode_numeric_min",
        "episode_numeric_max",
        "episode_numeric_median",
    ]
    boolean_columns = [
        "clinical_visit",
        "same_day_conflict",
        "unit_conflict",
        "result_conflict",
    ]
    datetime_columns = [
        "clinical_anchor_date",
        "episode_start_date",
        "episode_end_date",
        "selected_lab_date",
    ]
    string_columns = [
        "patient_id",
        "clinical_episode_id",
        "canonical_analyte",
        "lab_family",
        "analytic_role",
        "selected_value_type",
        "selected_operator",
        "selected_value_text",
        "selected_reported_interpretation",
        "selected_unit",
        "selected_reference_range_raw",
        "selected_reference_operator",
        "selected_reference_status",
        "selection_status",
        "source_protocol",
        "visit_type",
    ]
    for column in integer_columns:
        if column in out:
            out[column] = (
                pd.to_numeric(out[column], errors="coerce").round().astype("Int64")
            )
    for column in numeric_columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Float64")
    for column in boolean_columns:
        if column in out:
            out[column] = out[column].astype("boolean")
    for column in datetime_columns:
        if column in out:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    for column in string_columns:
        if column in out:
            out[column] = out[column].astype("string")
    return out


def _coerce_wide_output_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit types to structural and dynamically named wide columns."""
    integer_columns = {
        "clinical_visit_number",
        "time_since_clinical_baseline_days",
    }
    boolean_columns = {"clinical_visit", "is_clinical_baseline", *BASELINE_FEATURES}
    datetime_columns = {
        "clinical_anchor_date",
        "episode_start_date",
        "episode_end_date",
        "clinical_baseline_date",
    }
    string_columns = {
        "patient_id",
        "clinical_episode_id",
        "visit_type",
        "clinical_baseline_episode_id",
    }

    def coerce(column: str) -> pd.Series:
        values = frame[column]
        if column in integer_columns or column.endswith(
            ("__days_from_anchor", "__n_measurements")
        ):
            return pd.to_numeric(values, errors="coerce").round().astype("Int64")
        if column == "time_since_clinical_baseline_years" or column.endswith(
            "__value"
        ):
            return pd.to_numeric(values, errors="coerce").astype("Float64")
        if column in boolean_columns or column.endswith(
            (
                "__conflict",
                "__ever_positive_through_episode",
                "__known_through_episode",
            )
        ):
            return values.astype("boolean")
        if column in datetime_columns or column.endswith("__measurement_date"):
            return pd.to_datetime(values, errors="coerce")
        if column in string_columns or column.endswith(
            (
                "__text",
                "__unit",
                "__reference_status",
                "__selection_status",
                "__episode_status",
                "__patient_consensus_value",
            )
        ):
            return values.astype("string")
        return values

    # Construct once from typed Series rather than repeatedly inserting dynamic
    # lab columns, which would recreate the fragmentation this boundary prevents.
    return pd.DataFrame({column: coerce(column) for column in frame.columns})


def _object_dtype_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns whose Parquet representation would require inference."""
    return [column for column in frame.columns if frame[column].dtype == "object"]


def _mapping_valid(labs: pd.DataFrame) -> pd.Series:
    status = _column(labs, "semantic_mapping_status").astype("string").str.strip()
    # Missing mapping status is not evidence of a valid upstream mapping.
    return status.notna() & ~status.str.contains(INVALID_MAPPING, na=True)


def _normal_unit(value: Any) -> Any:
    return (
        pd.NA
        if pd.isna(value) or not str(value).strip()
        else re.sub(r"\s+", " ", str(value).strip()).casefold()
    )


def _text(row: pd.Series, name: str) -> str:
    value = row.get(name)
    return "" if pd.isna(value) else str(value).strip()


def _value_type(row: pd.Series) -> str:
    if pd.notna(
        pd.to_numeric(
            pd.Series([row.get("result_numeric_exact")]), errors="coerce"
        ).iloc[0]
    ):
        return "exact_numeric"
    operator = _text(row, "result_operator")
    bound = pd.to_numeric(
        pd.Series([row.get("result_numeric_bound")]), errors="coerce"
    ).iloc[0]
    if operator in {"<", "<=", ">", ">="} and pd.notna(bound):
        return "censored_numeric"
    if any(
        _text(row, c) for c in ("result_text", "result_raw", "reported_interpretation")
    ):
        return "qualitative"
    return "uninterpretable"


def _reference_status(row: pd.Series) -> Any:
    """Interpret only explicit evidence or the row's contemporaneous range."""
    for candidate in (
        _text(row, "reported_interpretation"),
        _text(row, "result_text"),
        _text(row, "result_raw"),
    ):
        if NEGATIVE.search(candidate):
            return "negative"
        if POSITIVE.search(candidate):
            return "positive"
        if LOW.search(candidate):
            return "low"
        if HIGH.search(candidate):
            return "high"
        if ABNORMAL.search(candidate):
            return "abnormal"
        if NORMAL.search(candidate):
            return "normal"
    value = pd.to_numeric(
        pd.Series([row.get("result_numeric_exact")]), errors="coerce"
    ).iloc[0]
    low = pd.to_numeric(pd.Series([row.get("reference_low")]), errors="coerce").iloc[0]
    high = pd.to_numeric(pd.Series([row.get("reference_high")]), errors="coerce").iloc[
        0
    ]
    if pd.notna(value) and pd.notna(low) and value < low:
        return "low"
    if pd.notna(value) and pd.notna(high) and value > high:
        return "high"
    if pd.notna(value) and (pd.notna(low) or pd.notna(high)):
        return "within_range"
    return "uninterpretable"


def _logical_id(row: pd.Series) -> str:
    order = _text(row, "order_identifier")
    if order:
        return "order:" + order
    specimen = _text(row, "specimen_datetime")
    assay = _text(row, "assay")
    if specimen and assay:
        return f"specimen-assay:{specimen}|{assay}"
    order_name, cluster = (
        _text(row, "order_name_original"),
        _text(row, "cluster_name_original"),
    )
    if specimen and order_name and cluster:
        return f"specimen-order-cluster:{specimen}|{order_name}|{cluster}"
    # No strong identity evidence: retain the row as a separate measurement.
    return f"row:{row.name}"


def _deduplicate(group: pd.DataFrame) -> pd.DataFrame:
    work = group.copy()
    work["_logical_id"] = work.apply(_logical_id, axis=1)
    status = _column(work, "result_status", "").astype("string").str.lower()
    work["_status_rank"] = np.select(
        [
            status.str.contains("final|verified", na=False),
            status.str.contains("prelim", na=False),
        ],
        [0, 2],
        default=1,
    )
    # Stable sorting selects final/verified, while drop_duplicates collapses exact
    # repeated exports sharing strong observation identity.
    work = work.sort_values(["_logical_id", "_status_rank"], kind="stable")
    return work.drop_duplicates("_logical_id", keep="first")


def _signature(row: pd.Series) -> tuple[Any, ...]:
    kind = _value_type(row)
    if kind == "exact_numeric":
        value = pd.to_numeric(
            pd.Series([row.get("result_numeric_exact")]), errors="coerce"
        ).iloc[0]
    elif kind == "censored_numeric":
        value = (
            _text(row, "result_operator"),
            pd.to_numeric(
                pd.Series([row.get("result_numeric_bound")]), errors="coerce"
            ).iloc[0],
        )
    else:
        value = (_text(row, "result_text") or _text(row, "result_raw")).casefold()
    return kind, value, _normal_unit(row.get("unit"))


def select_episode_analytes(
    usable: pd.DataFrame, spine: pd.DataFrame
) -> tuple[pd.DataFrame, set[int]]:
    anchor = spine.set_index(KEY)
    rows, conflict_indexes = [], set()
    for key, original in usable.groupby(ANALYTE_KEY, sort=False, dropna=False):
        group = _deduplicate(original)
        distances = pd.to_numeric(group["days_from_clinical_anchor"], errors="coerce")
        minimum = distances.abs().min() if distances.notna().any() else np.nan
        nearest = (
            group[distances.abs().eq(minimum)].copy()
            if pd.notna(minimum)
            else group.copy()
        )
        pre = pd.to_numeric(nearest["days_from_clinical_anchor"], errors="coerce") <= 0
        if pre.any():
            nearest = nearest[pre]
        signatures = {_signature(row) for _, row in nearest.iterrows()}
        units = (
            {_normal_unit(x) for x in group["unit"].dropna()}
            if "unit" in group
            else set()
        )
        units.discard(pd.NA)
        unit_conflict = len(units) > 1 and any(
            _value_type(r) == "exact_numeric" for _, r in group.iterrows()
        )
        result_conflict = len(signatures) > 1
        conflict = unit_conflict or result_conflict
        if conflict:
            conflict_indexes.update(original.index)
        chosen = nearest.sort_values(["lab_date"], kind="stable").iloc[0]
        value_type = _value_type(chosen)
        selected_numeric = (
            pd.to_numeric(
                pd.Series([chosen.get("result_numeric_exact")]), errors="coerce"
            ).iloc[0]
            if value_type == "exact_numeric" and not conflict
            else np.nan
        )
        text_value = _text(chosen, "result_text") or (
            _text(chosen, "result_raw")
            if value_type in {"qualitative", "uninterpretable"}
            else ""
        )
        exact = pd.to_numeric(_column(group, "result_numeric_exact"), errors="coerce")
        compatible = (
            exact[group["unit"].map(_normal_unit).eq(_normal_unit(chosen.get("unit")))]
            if "unit" in group and not unit_conflict
            else exact.iloc[0:0]
        )
        spine_row = anchor.loc[(key[0], key[1])]
        row = {
            "patient_id": key[0],
            "clinical_episode_id": key[1],
            "clinical_anchor_date": spine_row.get("clinical_anchor_date"),
            "clinical_visit_number": spine_row.get(
                "clinical_visit_number", spine_row.get("visit_number", pd.NA)
            ),
            "clinical_visit": spine_row.get("clinical_visit", pd.NA),
            "visit_type": spine_row.get("visit_type", pd.NA),
            "episode_start_date": spine_row.get("episode_start_date", pd.NaT),
            "episode_end_date": spine_row.get("episode_end_date", pd.NaT),
            "canonical_analyte": key[2],
            "lab_family": chosen.get("lab_family"),
            "analytic_role": chosen.get("analytic_role"),
            "selected_lab_date": chosen.get("lab_date"),
            "selected_days_from_clinical_anchor": chosen.get(
                "days_from_clinical_anchor"
            ),
            "selected_value_type": value_type,
            "selected_value_numeric": selected_numeric,
            "selected_operator": (
                chosen.get("result_operator")
                if value_type == "censored_numeric" and not conflict
                else pd.NA
            ),
            "selected_numeric_bound": (
                chosen.get("result_numeric_bound")
                if value_type == "censored_numeric" and not conflict
                else np.nan
            ),
            "selected_value_text": text_value if not conflict else pd.NA,
            "selected_reported_interpretation": (
                chosen.get("reported_interpretation") if not conflict else pd.NA
            ),
            "selected_unit": chosen.get("unit") if not unit_conflict else pd.NA,
            "selected_reference_range_raw": chosen.get("reference_range_raw"),
            "selected_reference_low": chosen.get("reference_low"),
            "selected_reference_high": chosen.get("reference_high"),
            "selected_reference_operator": chosen.get("reference_operator"),
            "selected_reference_bound": chosen.get("reference_bound"),
            "selected_reference_status": (
                _reference_status(chosen) if not conflict else pd.NA
            ),
            "n_measurements_in_episode": len(group),
            "n_valid_measurements_in_episode": len(group),
            "same_day_conflict": bool(result_conflict and minimum == 0),
            "unit_conflict": bool(unit_conflict),
            "result_conflict": bool(result_conflict),
            "selection_status": (
                "conflict"
                if conflict
                else (
                    "selected"
                    if value_type != "uninterpretable"
                    else "selected_uninterpretable"
                )
            ),
            "source_protocol": chosen.get("source_protocol"),
            "episode_numeric_min": (
                compatible.min() if compatible.notna().any() else np.nan
            ),
            "episode_numeric_max": (
                compatible.max() if compatible.notna().any() else np.nan
            ),
            "episode_numeric_median": (
                compatible.median() if compatible.notna().any() else np.nan
            ),
        }
        rows.append(row)
    selected = _coerce_analyte_output_schema(pd.DataFrame(rows))
    return selected, conflict_indexes


def _status_is_positive(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if value == "positive":
        return True
    if value == "negative":
        return False
    return pd.NA


def build_wide(
    spine: pd.DataFrame, selected: pd.DataFrame, usable: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    missing_keys = set(KEY) - set(spine.columns)
    if missing_keys:
        raise ValueError(f"Clinical spine missing wide keys: {sorted(missing_keys)}")
    base_columns = [c for c in WIDE_SPINE_COLUMNS if c in spine.columns]
    wide_base = spine[base_columns].copy().set_index(KEY)
    wide_index = wide_base.index
    feature_frames: list[pd.DataFrame] = []
    fields = {
        "selected_value_numeric": "value",
        "selected_value_text": "text",
        "selected_unit": "unit",
        "selected_reference_status": "reference_status",
        "selected_lab_date": "measurement_date",
        "selected_days_from_clinical_anchor": "days_from_anchor",
        "n_measurements_in_episode": "n_measurements",
        "result_conflict": "conflict",
        "selection_status": "selection_status",
    }
    for source, suffix in fields.items():
        if selected.empty:
            continue
        pivot = selected.pivot(index=KEY, columns="canonical_analyte", values=source)
        pivot.columns = [f"{c}__{suffix}" for c in pivot.columns]
        feature_frames.append(pivot.reindex(wide_index))
    # Include analytes observed only in nonclinical episodes: their dynamic
    # episode value stays missing, but dated stable/genetic history may still
    # legitimately become known at a later clinical visit.
    families = (
        usable.groupby("canonical_analyte")["lab_family"].first()
        if not usable.empty
        else pd.Series(dtype="object")
    )
    future_violations = 0
    anchors = pd.to_datetime(wide_base["clinical_anchor_date"], errors="coerce")
    lab_dates = pd.to_datetime(usable["lab_date"], errors="coerce")
    derived_columns: dict[str, pd.Series] = {}
    for analyte, family in families.items():
        part = (
            selected[selected.canonical_analyte.eq(analyte)]
            if not selected.empty
            else selected
        )
        if part.empty:
            status_series = pd.Series(pd.NA, index=wide_index, dtype="string")
        else:
            status_series = (
                part.set_index(KEY)["selected_reference_status"]
                .reindex(wide_index)
                .astype("string")
            )
        derived_columns[f"{analyte}__episode_status"] = status_series
        evidence = usable[usable.canonical_analyte.eq(analyte)].copy()
        evidence["_date"] = lab_dates.reindex(evidence.index)
        evidence["_status"] = evidence.apply(
            lambda r: _status_is_positive(_reference_status(r)), axis=1
        )
        if family == "stable_autoimmune" and evidence["_status"].notna().any():
            values = []
            for (pid, _), date in zip(wide_index, anchors):
                prior = evidence[
                    evidence.patient_id.eq(pid)
                    & evidence._date.notna()
                    & evidence._date.le(date)
                ]["_status"].dropna()
                values.append(pd.NA if prior.empty else bool(prior.astype(bool).any()))
            derived_columns[f"{analyte}__ever_positive_through_episode"] = pd.Series(
                pd.array(values, dtype="boolean"), index=wide_index
            )
        if family == "fixed_genetic":
            consensus, consensus_conflict = {}, {}
            for pid, patient in evidence.groupby("patient_id"):
                vals = {
                    (_text(r, "result_text") or _text(r, "result_raw")).casefold()
                    for _, r in patient.iterrows()
                } - {""}
                consensus[pid] = next(iter(vals)) if len(vals) == 1 else pd.NA
                consensus_conflict[pid] = len(vals) > 1
            derived_columns[f"{analyte}__patient_consensus_value"] = pd.Series(
                [consensus.get(pid, pd.NA) for pid, _ in wide_index],
                index=wide_index,
                dtype="string",
            )
            derived_columns[f"{analyte}__patient_consensus_conflict"] = pd.Series(
                pd.array(
                    [consensus_conflict.get(pid, False) for pid, _ in wide_index],
                    dtype="boolean",
                ),
                index=wide_index,
            )
            known = []
            for (pid, _), date in zip(wide_index, anchors):
                prior = evidence[
                    evidence.patient_id.eq(pid)
                    & evidence._date.notna()
                    & evidence._date.le(date)
                ]
                known.append(bool(len(prior)))
            derived_columns[f"{analyte}__known_through_episode"] = pd.Series(
                pd.array(known, dtype="boolean"), index=wide_index
            )
    derived_frame = (
        pd.DataFrame(derived_columns, index=wide_index)
        if derived_columns
        else pd.DataFrame(index=wide_index)
    )
    wide = pd.concat(
        [
            wide_base,
            *(frame.reindex(wide_index) for frame in feature_frames),
            derived_frame.reindex(wide_index),
        ],
        axis=1,
    )
    wide = wide.reset_index().copy()
    return wide, future_violations


def _detail(frame: pd.DataFrame, reason: str, spine: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["clinical_episode_id"] = _column(out, "matched_clinical_episode_id")
    anchors = spine.set_index(KEY)["clinical_anchor_date"]
    out["clinical_anchor_date"] = [
        anchors.get((p, e), pd.NaT)
        for p, e in zip(out.patient_id, out.clinical_episode_id)
    ]
    out["reason_for_review"] = reason
    return out.reindex(columns=QC_DETAIL)


def write_table1(baseline: pd.DataFrame) -> None:
    labels = {
        "baseline_anti_ro_ssa": "Anti-Ro/SSA positive",
        "baseline_anti_la_ssb": "Anti-La/SSB positive",
        "baseline_ana": "ANA positive",
        "baseline_rf": "Rheumatoid factor positive",
        "baseline_cryoglobulinemia": "Cryoglobulinemia",
        "baseline_low_c4": "Low C4",
        "baseline_leukopenia": "Leukopenia",
    }
    rows = []
    for col, label in labels.items():
        values = (
            baseline[col].astype("boolean")
            if col in baseline
            else pd.Series(pd.array([pd.NA] * len(baseline), dtype="boolean"))
        )
        n = int(values.notna().sum())
        rows.append(
            {
                "section": "Serologic characteristics",
                "variable": label,
                "value": (
                    f"{int(values.sum(skipna=True))} ({100 * values.mean():.1f}%)"
                    if n
                    else "NA"
                ),
                "n_total_cohort": len(baseline),
                "n_tested": n,
                "n_interpretable": n,
                "missing_n": int(values.isna().sum()),
                "unclassified_n": 0,
                "denominator_type": "20b baseline interpretable",
            }
        )
    path = common.BLOCKA_TABLES_DIR / "01_table1_overall.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if "section" in old:
        old = old[old.section.ne("Serologic characteristics")]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True, sort=False).to_csv(
        path, index=False
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args(argv)
    common.ensure_output_dirs()
    all_spine, clinical_spine, labs, baseline = (
        pd.read_parquet(args.all_spine),
        pd.read_parquet(args.spine),
        pd.read_parquet(args.labs_long),
        pd.read_parquet(args.baseline_labs),
    )
    required_spine = set(KEY + ["clinical_anchor_date"])
    required_labs = {
        "patient_id",
        "canonical_analyte",
        "semantic_mapping_status",
        "result_valid_for_analysis",
        "matched_clinical_episode_id",
        "episode_match_ambiguous",
        "lab_date",
        "days_from_clinical_anchor",
    }
    for name, frame in (
        ("All-episode spine", all_spine),
        ("Clinical spine", clinical_spine),
    ):
        if required_spine - set(frame):
            raise ValueError(
                f"{name} missing required columns: "
                f"{sorted(required_spine - set(frame))}"
            )
    if required_labs - set(labs):
        raise ValueError(
            f"Labs missing required columns: {sorted(required_labs - set(labs))}"
        )
    for frame in (all_spine, clinical_spine, labs, baseline):
        frame["_patient_id_match"] = _normalize_patient_id(frame["patient_id"])

    patient_crosswalk = all_spine[["patient_id", "_patient_id_match"]].drop_duplicates()
    collision_counts = patient_crosswalk.groupby("_patient_id_match")[
        "patient_id"
    ].nunique()
    normalization_collisions = int(collision_counts.gt(1).sum())
    if normalization_collisions:
        raise AssertionError(
            "Patient-ID normalization creates collisions in authoritative spine"
        )
    if all_spine.duplicated(KEY).any():
        raise AssertionError(
            "Duplicate patient_id + clinical_episode_id in all-episode spine"
        )
    if clinical_spine.duplicated(KEY).any():
        raise AssertionError(
            "Duplicate patient_id + clinical_episode_id in clinical spine"
        )
    if (
        "clinical_visit" in clinical_spine
        and not _bool(clinical_spine["clinical_visit"]).all()
    ):
        raise AssertionError("Clinical spine contains clinical_visit != True")
    labs = labs.copy()
    labs["lab_date"] = pd.to_datetime(labs.lab_date, errors="coerce")
    mapped = labs.canonical_analyte.notna() & _mapping_valid(labs)
    valid_result = _bool(labs.result_valid_for_analysis)
    matched = labs.matched_clinical_episode_id.notna()
    ambiguous = _bool(labs.episode_match_ambiguous)
    eligible = mapped & valid_result & matched & ~ambiguous
    usable_all = labs[eligible].copy()
    usable_all["clinical_episode_id"] = usable_all.matched_clinical_episode_id
    all_spine_match_keys = pd.MultiIndex.from_frame(all_spine[MATCH_KEY])
    usable_all_match_keys = pd.MultiIndex.from_frame(usable_all[MATCH_KEY])
    resolved_lab_rows = usable_all_match_keys.isin(all_spine_match_keys)
    unresolved_lab_rows = int((~resolved_lab_rows).sum())
    if unresolved_lab_rows:
        raise AssertionError(
            "Matched lab episode is absent from authoritative all-episode spine "
            "after patient-ID normalization"
        )
    raw_spine_keys = pd.MultiIndex.from_frame(all_spine[KEY])
    raw_usable_keys = pd.MultiIndex.from_frame(usable_all[KEY])
    rows_resolved_by_normalization = int(
        (resolved_lab_rows & ~raw_usable_keys.isin(raw_spine_keys)).sum()
    )

    obj1_by_normalized = patient_crosswalk.set_index("_patient_id_match")["patient_id"]
    lab_patient_groups = labs.groupby("_patient_id_match", dropna=False)
    crosswalk_qc = lab_patient_groups.agg(
        patient_id_eda=("patient_id", "first"),
        n_lab_records=("patient_id", "size"),
    ).reset_index()
    matched_episode_counts = (
        usable_all.groupby("_patient_id_match")["clinical_episode_id"]
        .nunique()
        .rename("n_matched_episodes")
    )
    crosswalk_qc["patient_id_obj1"] = crosswalk_qc["_patient_id_match"].map(
        obj1_by_normalized
    )
    crosswalk_qc = crosswalk_qc.join(matched_episode_counts, on="_patient_id_match")
    crosswalk_qc["n_matched_episodes"] = (
        crosswalk_qc["n_matched_episodes"].fillna(0).astype(int)
    )
    crosswalk_qc = crosswalk_qc.rename(
        columns={"_patient_id_match": "patient_id_normalized"}
    ).reindex(
        columns=[
            "patient_id_obj1",
            "patient_id_normalized",
            "patient_id_eda",
            "n_lab_records",
            "n_matched_episodes",
        ]
    )
    crosswalk_qc.to_csv(
        common.BLOCKA_QC_DIR / "01_labs_patient_id_crosswalk_qc.csv", index=False
    )

    # Episode metadata belongs to the authoritative spine.  Refuse collisions
    # rather than silently replacing similarly named lab-source columns.
    context_columns = [
        c
        for c in (
            "clinical_anchor_date",
            "clinical_visit",
            "visit_type",
            "episode_start_date",
            "episode_end_date",
            "clinical_baseline_episode_id",
            "clinical_baseline_date",
            "is_clinical_baseline",
        )
        if c in all_spine
    ]
    collisions = set(context_columns).intersection(usable_all.columns)
    usable_all = usable_all.rename(columns={c: f"{c}__lab_source" for c in collisions})
    usable_all = usable_all.rename(columns={"patient_id": "patient_id__eda_source"})
    usable_all = usable_all.merge(
        all_spine[MATCH_KEY + ["patient_id"] + context_columns],
        on=MATCH_KEY,
        how="left",
        validate="many_to_one",
    )
    if usable_all["patient_id"].isna().any():
        raise AssertionError(
            "Lab matched episode still cannot be resolved after patient-ID normalization"
        )
    for column in collisions:
        source = usable_all[f"{column}__lab_source"]
        authoritative = usable_all[column]
        if "date" in column:
            source = pd.to_datetime(source, errors="coerce")
            authoritative = pd.to_datetime(authoritative, errors="coerce")
        disagreement = (
            source.notna() & authoritative.notna() & ~source.eq(authoritative)
        )
        if disagreement.any():
            raise AssertionError(
                f"Lab-source {column} disagrees with authoritative all-episode spine"
            )
        usable_all = usable_all.drop(columns=f"{column}__lab_source")
    selected_all, conflict_indexes = select_episode_analytes(usable_all, all_spine)
    if (
        "clinical_visit_number" in selected_all
        and str(selected_all["clinical_visit_number"].dtype) != "Int64"
    ):
        raise AssertionError(
            "clinical_visit_number did not retain nullable integer schema"
        )
    if not selected_all.empty and selected_all.duplicated(ANALYTE_KEY).any():
        raise AssertionError("Duplicate analyte-level keys")
    clinical_keys = pd.MultiIndex.from_frame(clinical_spine[KEY])
    usable_clinical = usable_all.loc[_bool(usable_all["clinical_visit"])].copy()
    if selected_all.empty:
        selected_clinical = selected_all.copy()
    else:
        selected_all_index = pd.MultiIndex.from_frame(selected_all[KEY])
        selected_clinical = selected_all.loc[
            selected_all_index.isin(clinical_keys)
        ].copy()
    wide, future_violations = build_wide(clinical_spine, selected_clinical, usable_all)
    unexpected_spine_columns = set(clinical_spine.columns) - set(WIDE_SPINE_COLUMNS)
    leaked_columns = [c for c in wide.columns if c in unexpected_spine_columns]
    if leaked_columns:
        raise AssertionError(
            "Lab wide accidentally inherited non-structural clinical-spine columns: "
            + ", ".join(leaked_columns[:50])
        )
    baseline_feature_cols = [c for c in BASELINE_FEATURES if c in baseline]
    baseline_for_merge = baseline[["_patient_id_match"] + baseline_feature_cols].copy()
    if baseline_for_merge["_patient_id_match"].duplicated().any():
        raise AssertionError("20b baseline has duplicate normalized patient IDs")
    wide["_patient_id_match"] = _normalize_patient_id(wide["patient_id"])
    wide = wide.merge(
        baseline_for_merge,
        on="_patient_id_match",
        how="left",
        validate="many_to_one",
    )
    wide = wide.drop(columns="_patient_id_match")
    wide_keys = pd.MultiIndex.from_frame(wide[KEY])
    missing_keys, extra_keys = (
        clinical_keys.difference(wide_keys),
        wide_keys.difference(clinical_keys),
    )
    if (
        wide.duplicated(KEY).any()
        or len(wide) != len(clinical_spine)
        or len(missing_keys)
        or len(extra_keys)
    ):
        raise AssertionError("Wide output does not exactly preserve the clinical spine")
    if future_violations:
        raise AssertionError(f"Detected {future_violations} future-leakage violations")

    # Reapply schemas after filtering and merging so every primary artifact has
    # deterministic Parquet types at its write boundary.
    selected_all = _coerce_analyte_output_schema(selected_all)
    wide = _coerce_wide_output_schema(wide)
    for name, frame in (("analyte-level", selected_all), ("wide", wide)):
        object_columns = _object_dtype_columns(frame)
        if object_columns:
            raise AssertionError(
                f"{name} contains ambiguous object columns: "
                + ", ".join(object_columns[:50])
            )

    for path, frame, csv_copy in (
        (args.analyte_output, selected_all, True),
        (args.wide_output, wide, True),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        if csv_copy:
            frame.to_csv(path.with_suffix(".csv"), index=False)
    serology_cols = list(
        dict.fromkeys(
            KEY
            + [
                c
                for c in clinical_spine.columns
                if c in {"clinical_anchor_date", "clinical_visit_number"}
            ]
            + [
                c
                for c in wide
                if any(
                    x in c
                    for x in (
                        "anti_ro",
                        "anti_la",
                        "ana",
                        "rheumatoid_factor",
                        "cryoglobulin",
                        "complement_c4",
                        "baseline_",
                    )
                )
            ]
        )
    )
    wide[serology_cols].to_parquet(args.serology_output, index=False)

    unmatched_df, ambiguous_df = labs[~matched], labs[ambiguous]
    conflict_df = _detail(
        labs.loc[labs.index.intersection(conflict_indexes)],
        "episode_result_or_unit_conflict",
        all_spine,
    )
    _detail(unmatched_df, "unmatched_episode", all_spine).to_csv(
        common.BLOCKA_QC_DIR / "01_labs_unmatched_records.csv", index=False
    )
    _detail(ambiguous_df, "ambiguous_episode_match", all_spine).to_csv(
        common.BLOCKA_QC_DIR / "01_labs_ambiguous_episode_matches.csv", index=False
    )
    nonclinical = usable_all.loc[~_bool(usable_all["clinical_visit"])].copy()
    nonclinical_columns = [
        "patient_id",
        "matched_clinical_episode_id",
        "clinical_anchor_date",
        "clinical_visit",
        "visit_type",
        "lab_date",
        "days_from_clinical_anchor",
        "canonical_analyte",
        "lab_family",
        "analytic_role",
        "result_raw",
        "result_numeric_exact",
        "result_operator",
        "result_numeric_bound",
        "result_text",
        "unit",
        "reported_interpretation",
        "episode_match_method",
        "source_protocol",
    ]
    nonclinical.reindex(columns=nonclinical_columns).to_csv(
        common.BLOCKA_QC_DIR / "01_labs_matched_nonclinical_episode_records.csv",
        index=False,
    )
    conflict_df.to_csv(
        common.BLOCKA_QC_DIR / "01_labs_episode_conflicts.csv", index=False
    )
    unit_frame = labs[mapped].copy()
    unit_frame["unit"] = _column(unit_frame, "unit").map(_normal_unit)
    unit_frame["clinical_episode_id"] = unit_frame.matched_clinical_episode_id
    inventory = (
        unit_frame.groupby(["canonical_analyte", "unit"], dropna=False)
        .agg(
            n_records=("patient_id", "size"),
            n_patients=("patient_id", "nunique"),
            n_episodes=("clinical_episode_id", "nunique"),
        )
        .reset_index()
    )
    inventory.to_csv(common.BLOCKA_QC_DIR / "01_labs_unit_inventory.csv", index=False)
    unit_conflicts = (
        selected_all[selected_all.unit_conflict]
        if not selected_all.empty
        else selected_all
    )
    unit_conflicts.to_csv(
        common.BLOCKA_QC_DIR / "01_labs_unit_conflicts.csv", index=False
    )

    coverage_rows = []
    total_patients, total_episodes = (
        clinical_spine.patient_id.nunique(),
        len(clinical_spine),
    )
    for analyte, raw in labs[mapped].groupby("canonical_analyte", dropna=False):
        sel = (
            selected_clinical[selected_clinical.canonical_analyte.eq(analyte)]
            if not selected_clinical.empty
            else selected_clinical
        )
        valid_raw = raw[eligible.reindex(raw.index, fill_value=False)]
        types = valid_raw.apply(_value_type, axis=1).value_counts()
        coverage_rows.append(
            {
                "canonical_analyte": analyte,
                "lab_family": (
                    raw.lab_family.dropna().iloc[0]
                    if "lab_family" in raw and raw.lab_family.notna().any()
                    else pd.NA
                ),
                "analytic_role": (
                    raw.analytic_role.dropna().iloc[0]
                    if "analytic_role" in raw and raw.analytic_role.notna().any()
                    else pd.NA
                ),
                "n_input_records": len(raw),
                "n_valid_records": len(valid_raw),
                "n_patients_measured": valid_raw.patient_id.nunique(),
                "pct_patients_measured": (
                    100 * valid_raw.patient_id.nunique() / total_patients
                    if total_patients
                    else np.nan
                ),
                "n_clinical_episodes_measured": len(sel),
                "pct_clinical_episodes_measured": (
                    100 * len(sel) / total_episodes if total_episodes else np.nan
                ),
                "n_exact_numeric": int(types.get("exact_numeric", 0)),
                "n_censored_numeric": int(types.get("censored_numeric", 0)),
                "n_qualitative": int(types.get("qualitative", 0)),
                "n_uninterpretable": int(types.get("uninterpretable", 0)),
                "n_units": raw.get("unit", pd.Series(dtype="object"))
                .map(_normal_unit)
                .nunique(),
                "n_unit_conflicts": int(sel.unit_conflict.sum()) if len(sel) else 0,
                "n_episode_conflicts": (
                    int((sel.result_conflict | sel.unit_conflict).sum())
                    if len(sel)
                    else 0
                ),
                "n_ambiguous_episode_matches": int(
                    ambiguous.reindex(raw.index, fill_value=False).sum()
                ),
                "n_unmatched_records": int(
                    (~matched).reindex(raw.index, fill_value=False).sum()
                ),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(
        common.BLOCKA_TABLES_DIR / "01_labs_episode_coverage.csv", index=False
    )
    nonclinical_visit_type = _column(nonclinical, "visit_type").astype("string")
    ambiguous_visit_type = nonclinical_visit_type.str.casefold().eq("ambiguous")
    research_or_procedure = nonclinical_visit_type.str.casefold().str.contains(
        r"research|procedure", na=False
    )
    qc = {
        "n_patient_ids_obj1": int(all_spine.patient_id.nunique()),
        "n_patient_ids_eda_labs": int(labs.patient_id.nunique()),
        "n_patient_ids_matched_after_normalization": int(
            labs.loc[
                labs["_patient_id_match"].isin(patient_crosswalk["_patient_id_match"]),
                "_patient_id_match",
            ].nunique()
        ),
        "n_patient_ids_unmatched_after_normalization": int(
            labs.loc[
                ~labs["_patient_id_match"].isin(patient_crosswalk["_patient_id_match"]),
                "_patient_id_match",
            ].nunique()
        ),
        "n_patient_id_normalization_collisions": normalization_collisions,
        "n_lab_rows_resolved_by_patient_id_normalization": rows_resolved_by_normalization,
        "n_lab_rows_unresolved_after_patient_id_normalization": unresolved_lab_rows,
        "n_20b_patients_matched_after_normalization": int(
            baseline.loc[
                baseline["_patient_id_match"].isin(clinical_spine["_patient_id_match"]),
                "_patient_id_match",
            ].nunique()
        ),
        "n_20b_patients_unmatched_after_normalization": int(
            baseline.loc[
                ~baseline["_patient_id_match"].isin(
                    clinical_spine["_patient_id_match"]
                ),
                "_patient_id_match",
            ].nunique()
        ),
        "n_input_lab_records": len(labs),
        "n_valid_lab_records": int(eligible.sum()),
        "n_mapped_analytes": int(mapped.sum()),
        "n_unique_canonical_analytes": int(
            labs.loc[mapped, "canonical_analyte"].nunique()
        ),
        "n_unique_lab_families": (
            int(labs.loc[mapped, "lab_family"].nunique()) if "lab_family" in labs else 0
        ),
        "n_rows_analyte_episode": len(selected_all),
        "n_rows_wide": len(wide),
        "n_unique_patients": wide.patient_id.nunique(),
        "n_unique_clinical_episodes": wide[KEY].drop_duplicates().shape[0],
        "n_duplicate_patient_episode_analyte_keys": (
            int(selected_all.duplicated(ANALYTE_KEY).sum()) if len(selected_all) else 0
        ),
        "n_duplicate_patient_episode_keys_wide": int(wide.duplicated(KEY).sum()),
        "n_unmatched_records": int((~matched).sum()),
        "n_ambiguous_matches": int(ambiguous.sum()),
        "n_lab_records_matched_to_any_episode": len(usable_all),
        "n_lab_records_matched_to_clinical_episode": len(usable_clinical),
        "n_lab_records_matched_to_nonclinical_episode": len(nonclinical),
        "n_unique_matched_episodes_all": usable_all[KEY].drop_duplicates().shape[0],
        "n_unique_matched_clinical_episodes": usable_clinical[KEY]
        .drop_duplicates()
        .shape[0],
        "n_unique_matched_nonclinical_episodes": nonclinical[KEY]
        .drop_duplicates()
        .shape[0],
        "n_matched_ambiguous_visit_type": int(ambiguous_visit_type.sum()),
        "n_matched_research_or_procedure_only": int(research_or_procedure.sum()),
        "n_matched_other_nonclinical_visit_type": int(
            (~ambiguous_visit_type & ~research_or_procedure).sum()
        ),
        "n_result_conflicts": (
            int(selected_all.result_conflict.sum()) if len(selected_all) else 0
        ),
        "n_unit_conflicts": (
            int(selected_all.unit_conflict.sum()) if len(selected_all) else 0
        ),
        "n_exact_numeric_selected": (
            int(selected_all.selected_value_type.eq("exact_numeric").sum())
            if len(selected_all)
            else 0
        ),
        "n_censored_numeric_selected": (
            int(selected_all.selected_value_type.eq("censored_numeric").sum())
            if len(selected_all)
            else 0
        ),
        "n_qualitative_selected": (
            int(selected_all.selected_value_type.eq("qualitative").sum())
            if len(selected_all)
            else 0
        ),
        "n_clinical_spine_rows": len(clinical_spine),
        "n_wide_rows": len(wide),
        "n_missing_clinical_episodes_in_wide": len(missing_keys),
        "n_extra_clinical_episodes_in_wide": len(extra_keys),
        "future_leakage_violations": future_violations,
    }
    with open(
        common.BLOCKA_QC_DIR / "01_labs_episode_qc.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(qc, handle, indent=2)
    write_table1(
        baseline[
            baseline["_patient_id_match"].isin(clinical_spine["_patient_id_match"])
        ]
    )
    LOG.info("Wrote %s analyte-episode and %s wide rows", len(selected_all), len(wide))


if __name__ == "__main__":
    main()

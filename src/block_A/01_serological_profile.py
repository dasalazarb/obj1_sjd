#!/usr/bin/env python3
"""Build the episode-level serology dataset from authoritative EDA artifacts.

The three deliberately separate feature families are: observations assigned to
an episode (``episode_*``), stable-marker history known by an episode date
(``*_ever_positive_through_episode``), and baseline covariates copied verbatim
from EDA step 20b (``baseline_*``).  This module never reads raw BTRIS files,
creates episodes, or supplies assay-specific clinical cut-offs.
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
DEFAULT_LABS = Path(
    "/data/salazarda/data/eda_sjd/data_analytic/BTRIS/20_btris_lab_records_long.parquet"
)
DEFAULT_BASELINE = Path(
    "/data/salazarda/data/eda_sjd/data_analytic/BTRIS/20b_baseline_labs_patient_level.parquet"
)
KEY = ["patient_id", "clinical_episode_id"]
CORE = {
    "anti_ro_ssa",
    "anti_la_ssb",
    "ana_status",
    "ana_hep2_status",
    "rheumatoid_factor",
    "cryoglobulins",
    "complement_c4",
    "wbc",
}
MARKERS = {
    "ssa": {"anti_ro_ssa"},
    "ssb": {"anti_la_ssb"},
    "ana": {"ana_status", "ana_hep2_status"},
    "rf": {"rheumatoid_factor"},
    "cryo": {"cryoglobulins"},
    "c4": {"complement_c4"},
    "wbc": {"wbc"},
}
STATUS_COL = {
    "ssa": "episode_anti_ro_ssa_status",
    "ssb": "episode_anti_la_ssb_status",
    "ana": "episode_ana_status",
    "rf": "episode_rf_status",
    "cryo": "episode_cryoglobulinemia",
    "c4": "episode_low_c4",
    "wbc": "episode_leukopenia",
}
EVER_COL = {
    "ssa": "anti_ro_ssa_ever_positive_through_episode",
    "ssb": "anti_la_ssb_ever_positive_through_episode",
    "ana": "ana_ever_positive_through_episode",
    "rf": "rf_ever_positive_through_episode",
}
POS = re.compile(r"\b(positive|pos|reactive|detected|present)\b", re.I)
NEG = re.compile(
    r"\b(negative|neg|non[- ]?reactive|not detected|absent|none detected)\b", re.I
)
LOW = re.compile(r"\b(low|below)\b", re.I)
NORMAL = re.compile(r"\b(normal|within (?:the )?reference)\b", re.I)
REVIEW_COLUMNS = [
    "patient_id",
    "clinical_episode_id",
    "clinical_anchor_date",
    "canonical_analyte",
    "lab_date",
    "days_from_clinical_anchor",
    "result_raw",
    "result_numeric",
    "result_operator",
    "result_numeric_bound",
    "reported_interpretation",
    "reference_range_raw",
    "episode_match_method",
    "episode_match_ambiguous",
    "reason_for_review",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labs-long", type=Path, default=DEFAULT_LABS)
    p.add_argument("--baseline-labs", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--spine", type=Path, default=common.CLINICAL_VISIT_SPINE_PARQUET)
    p.add_argument(
        "--output",
        type=Path,
        default=common.BLOCKA_INTERMEDIATE_DATA_DIR
        / "01_serology_episode_level.parquet",
    )
    return p.parse_args(argv)


def _bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s.dtype):
        return s.fillna(False)
    return s.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _number(row: pd.Series) -> float | None:
    for name in ("result_numeric_exact", "result_numeric", "result_numeric_bound"):
        value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return None


def _definitely_below(value: float, operator: str, threshold: float) -> bool:
    return operator not in {">", ">="} and value < threshold


def _definitely_above(value: float, operator: str, threshold: float) -> bool:
    return operator not in {"<", "<="} and value > threshold


def interpret(row: pd.Series, marker: str) -> bool | pd._libs.missing.NAType:
    """Conservatively interpret upstream fields, without fallback cut-offs."""

    def text_value(name: str) -> str:
        value = row.get(name)
        return "" if pd.isna(value) else str(value)

    reported = text_value("reported_interpretation").strip()
    text = " ".join(text_value(c) for c in ("result_text", "result_raw"))
    # Upstream reported interpretation has precedence over result text.
    for candidate in (reported, text):
        if NEG.search(candidate):
            return False
        if POS.search(candidate):
            return True
        if marker in {"c4", "wbc"}:
            if LOW.search(candidate):
                return True
            if NORMAL.search(candidate):
                return False
    value = _number(row)
    if value is None:
        return pd.NA
    op = str(row.get("result_operator", "") or "").strip()
    low = pd.to_numeric(pd.Series([row.get("reference_low")]), errors="coerce").iloc[0]
    high = pd.to_numeric(pd.Series([row.get("reference_high")]), errors="coerce").iloc[
        0
    ]
    if marker in {"c4", "wbc"} and pd.notna(low):
        if _definitely_below(value, op, float(low)):
            return True
        if op not in {"<", "<="} and value >= float(low):
            return False
    if marker in {"ssa", "ssb", "ana", "rf", "cryo"} and pd.notna(high):
        if _definitely_above(value, op, float(high)):
            return True
        if op not in {">", ">="} and value <= float(high):
            return False
    return pd.NA


def _select_group(g: pd.DataFrame, marker: str) -> dict[str, Any]:
    prefix = f"episode_{marker}"
    out: dict[str, Any] = {
        f"{prefix}_n_measurements": len(g),
        f"{prefix}_conflict": False,
    }
    distance = pd.to_numeric(g["days_from_clinical_anchor"], errors="coerce")
    if distance.notna().any():
        nearest = distance.abs().min()
        chosen = g[distance.abs().eq(nearest)].copy()
        if (
            pd.to_numeric(chosen["days_from_clinical_anchor"], errors="coerce") <= 0
        ).any():
            chosen = chosen[
                pd.to_numeric(chosen["days_from_clinical_anchor"], errors="coerce") <= 0
            ]
    else:
        chosen = g.copy()
    chosen = chosen.sort_values(["lab_date", "canonical_analyte"], kind="stable")
    statuses = chosen["_interpreted"].dropna().astype(bool).unique()
    nums = (
        pd.to_numeric(
            chosen.get("result_numeric_exact", chosen.get("result_numeric")),
            errors="coerce",
        )
        .dropna()
        .unique()
    )
    conflict = len(statuses) > 1 or (marker in {"c4", "wbc"} and len(nums) > 1)
    out[f"{prefix}_conflict"] = conflict
    if conflict:
        out[f"{prefix}_selection_status"] = "conflict_nearest_measurements"
        out[STATUS_COL[marker]] = pd.NA
        if marker in {"c4", "wbc"}:
            out[f"episode_{marker}_value"] = np.nan
        return out
    row = chosen.iloc[0]
    out[STATUS_COL[marker]] = statuses[0] if len(statuses) else pd.NA
    out[f"{prefix}_selection_status"] = (
        "selected" if len(statuses) else "selected_uninterpretable"
    )
    out[f"episode_{marker}_measurement_date"] = row.get("lab_date")
    out[f"episode_{marker}_days_from_anchor"] = row.get("days_from_clinical_anchor")
    if marker in {"c4", "wbc"}:
        out[f"episode_{marker}_value"] = _number(row)
    return out


def derive_features(
    spine: pd.DataFrame, labs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    relevant = labs[labs["canonical_analyte"].isin(CORE)].copy()
    relevant["lab_date"] = pd.to_datetime(relevant["lab_date"], errors="coerce")
    matched = relevant["matched_clinical_episode_id"].notna()
    ambiguous = _bool(
        relevant.get("episode_match_ambiguous", pd.Series(False, index=relevant.index))
    )
    valid = _bool(relevant["result_valid_for_analysis"])
    usable = relevant[valid & matched & ~ambiguous].copy()
    usable["clinical_episode_id"] = usable["matched_clinical_episode_id"]
    pieces = []
    conflict_indexes: set[int] = set()
    for marker, analytes in MARKERS.items():
        m = usable[usable["canonical_analyte"].isin(analytes)].copy()
        m["_interpreted"] = m.apply(interpret, axis=1, marker=marker).astype("boolean")
        for key, group in m.groupby(KEY, dropna=False, sort=False):
            selected = _select_group(group, marker)
            selected.update(dict(zip(KEY, key)))
            pieces.append(selected)
            if selected[f"episode_{marker}_conflict"]:
                conflict_indexes.update(group.index)
        usable.loc[m.index, "_interpreted"] = m["_interpreted"]
    features = pd.DataFrame(pieces)
    if features.empty:
        features = pd.DataFrame(columns=KEY)
    else:
        features = features.groupby(KEY, as_index=False).first()
    out = spine.merge(features, on=KEY, how="left", validate="one_to_one")
    # Stable history uses the actual evidence date, not future episode status.
    anchors = pd.to_datetime(out["clinical_anchor_date"], errors="coerce")
    for marker, ever_col in EVER_COL.items():
        evidence = usable[usable["canonical_analyte"].isin(MARKERS[marker])].copy()
        evidence["_interpreted"] = evidence.apply(
            interpret, axis=1, marker=marker
        ).astype("boolean")
        values = []
        for pid, anchor in zip(out["patient_id"], anchors):
            prior = evidence[
                (evidence["patient_id"] == pid)
                & evidence["lab_date"].notna()
                & (evidence["lab_date"] <= anchor)
            ]["_interpreted"].dropna()
            values.append(pd.NA if prior.empty else bool(prior.any()))
        out[ever_col] = pd.array(values, dtype="boolean")
    review = relevant[
        ambiguous
        | relevant.index.isin(conflict_indexes)
        | (
            valid
            & matched
            & ~ambiguous
            & relevant.apply(
                lambda r: pd.isna(
                    interpret(
                        r,
                        next(
                            k for k, v in MARKERS.items() if r["canonical_analyte"] in v
                        ),
                    )
                ),
                axis=1,
            )
        )
    ].copy()
    review["clinical_episode_id"] = review["matched_clinical_episode_id"]
    review["reason_for_review"] = np.where(
        ambiguous.reindex(review.index, fill_value=False),
        "ambiguous_episode_match",
        np.where(
            review.index.isin(conflict_indexes),
            "discordant_nearest_measurements",
            "unresolvable_interpretation",
        ),
    )
    anchor_map = spine.set_index(KEY)["clinical_anchor_date"]
    review["clinical_anchor_date"] = [
        anchor_map.get((p, e), pd.NaT)
        for p, e in zip(review.patient_id, review.clinical_episode_id)
    ]
    unmatched = relevant[~matched].copy()
    unmatched["clinical_episode_id"] = pd.NA
    return (
        out,
        review.reindex(columns=REVIEW_COLUMNS),
        unmatched.reindex(columns=REVIEW_COLUMNS[:-1]),
    )


def merge_baseline(
    out: pd.DataFrame, baseline: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    if baseline["patient_id"].duplicated().any():
        raise AssertionError("20b baseline has duplicate patient_id rows")
    mismatches = 0
    spine_identity = out.groupby("patient_id")[
        ["clinical_baseline_episode_id", "clinical_baseline_date"]
    ].first()
    base_identity = baseline.set_index("patient_id")[
        ["clinical_baseline_episode_id", "clinical_baseline_date"]
    ]
    common_ids = spine_identity.index.intersection(base_identity.index)
    for col in ("clinical_baseline_episode_id", "clinical_baseline_date"):
        left, right = (
            spine_identity.loc[common_ids, col],
            base_identity.loc[common_ids, col],
        )
        if col.endswith("date"):
            left, right = (
                pd.to_datetime(left, errors="coerce"),
                pd.to_datetime(right, errors="coerce"),
            )
        mismatches += int(
            (
                (left.isna() ^ right.isna())
                | (left.notna() & right.notna() & left.ne(right))
            ).sum()
        )
    if mismatches:
        raise AssertionError(
            f"{mismatches} baseline identity mismatches between spine and 20b"
        )
    copied = baseline.drop(
        columns=["clinical_baseline_episode_id", "clinical_baseline_date"],
        errors="ignore",
    )
    return out.merge(
        copied, on="patient_id", how="left", validate="many_to_one"
    ), mismatches


def count_future_leakage_violations(out: pd.DataFrame, labs: pd.DataFrame) -> int:
    """Independently verify cumulative statuses against evidence dated by anchor."""
    analytes = set().union(*(MARKERS[m] for m in EVER_COL))
    relevant = labs[labs["canonical_analyte"].isin(analytes)].copy()
    valid = _bool(relevant["result_valid_for_analysis"])
    ambiguous = _bool(
        relevant.get("episode_match_ambiguous", pd.Series(False, index=relevant.index))
    )
    relevant = relevant[
        valid & relevant["matched_clinical_episode_id"].notna() & ~ambiguous
    ]
    relevant["lab_date"] = pd.to_datetime(relevant["lab_date"], errors="coerce")
    violations = 0
    for marker, column in EVER_COL.items():
        evidence = relevant[relevant["canonical_analyte"].isin(MARKERS[marker])].copy()
        evidence["status"] = evidence.apply(interpret, axis=1, marker=marker).astype(
            "boolean"
        )
        for patient_id, anchor, observed in out[
            ["patient_id", "clinical_anchor_date", column]
        ].itertuples(index=False, name=None):
            prior = evidence[
                (evidence["patient_id"] == patient_id)
                & evidence["lab_date"].notna()
                & (evidence["lab_date"] <= pd.to_datetime(anchor))
            ]["status"].dropna()
            expected = pd.NA if prior.empty else bool(prior.any())
            if pd.isna(expected) != pd.isna(observed) or (
                not pd.isna(expected) and bool(expected) != bool(observed)
            ):
                violations += 1
    return violations


def write_table1(baseline: pd.DataFrame) -> None:
    rows = []
    labels = {
        "baseline_anti_ro_ssa": "Anti-Ro/SSA positive",
        "baseline_anti_la_ssb": "Anti-La/SSB positive",
        "baseline_ana": "ANA positive",
        "baseline_rf": "Rheumatoid factor positive",
        "baseline_cryoglobulinemia": "Cryoglobulinemia",
        "baseline_low_c4": "Low C4",
        "baseline_leukopenia": "Leukopenia",
    }
    for col, label in labels.items():
        s = (
            baseline[col].astype("boolean")
            if col in baseline
            else pd.Series(pd.array([pd.NA] * len(baseline), dtype="boolean"))
        )
        interpreted = int(s.notna().sum())
        rows.append(
            {
                "section": "Serologic characteristics",
                "variable": label,
                "value": f"{int(s.sum(skipna=True))} ({100 * s.mean():.1f}%)"
                if interpreted
                else "NA",
                "n_total_cohort": len(baseline),
                "n_tested": interpreted,
                "n_interpretable": interpreted,
                "missing_n": int(s.isna().sum()),
                "unclassified_n": 0,
                "denominator_type": "20b baseline interpretable",
            }
        )
    path = common.BLOCKA_TABLES_DIR / "01_table1_overall.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        if "section" in old:
            old = old[old["section"] != "Serologic characteristics"]
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args(argv)
    common.ensure_output_dirs()
    spine, labs, baseline = (
        pd.read_parquet(args.spine),
        pd.read_parquet(args.labs_long),
        pd.read_parquet(args.baseline_labs),
    )
    missing = set(KEY + ["clinical_anchor_date"]) - set(spine)
    if missing:
        raise ValueError(f"Spine missing required columns: {sorted(missing)}")
    if spine.duplicated(KEY).any():
        raise AssertionError("Duplicate patient_id + clinical_episode_id in spine")
    if "clinical_visit" in spine and not _bool(spine["clinical_visit"]).all():
        raise AssertionError("Clinical spine contains non-clinical rows")
    original_keys = pd.MultiIndex.from_frame(spine[KEY])
    out, review, unmatched = derive_features(spine, labs)
    out, baseline_mismatches = merge_baseline(out, baseline)
    if len(out) != len(spine) or not pd.MultiIndex.from_frame(out[KEY]).equals(
        original_keys
    ):
        raise AssertionError("Clinical spine rows/order were not preserved")
    if out.duplicated(KEY).any():
        raise AssertionError("Duplicate output patient-episode keys")
    for col in list(STATUS_COL.values()) + list(EVER_COL.values()):
        if col not in out:
            out[col] = pd.array([pd.NA] * len(out), dtype="boolean")
        else:
            out[col] = out[col].astype("boolean")
    future_leakage_violations = count_future_leakage_violations(out, labs)
    if future_leakage_violations:
        raise AssertionError(
            f"Detected {future_leakage_violations} future-leakage violations"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.joinpath(args.output.stem + ".csv").parent.mkdir(
        parents=True, exist_ok=True
    )
    out.to_parquet(args.output, index=False)
    out.to_csv(args.output.with_suffix(".csv"), index=False)
    relevant = labs[labs.canonical_analyte.isin(CORE)]
    ambiguous = _bool(
        relevant.get("episode_match_ambiguous", pd.Series(False, index=relevant.index))
    )
    coverage = []
    for marker in MARKERS:
        status = STATUS_COL[marker]
        measured = out[status].notna()
        patients_measured = out.loc[measured, "patient_id"].nunique()
        raw_marker = relevant[relevant.canonical_analyte.isin(MARKERS[marker])]
        row = {
            "marker": marker,
            "n_clinical_episodes": len(out),
            "n_episodes_measured": int(measured.sum()),
            "pct_episodes_measured": 100 * measured.mean(),
            "n_patients": out.patient_id.nunique(),
            "n_patients_ever_measured": patients_measured,
            "pct_patients_ever_measured": 100
            * patients_measured
            / out.patient_id.nunique()
            if out.patient_id.nunique()
            else np.nan,
            "n_interpretable_episode_results": int(measured.sum()),
            "n_positive_episode_results": int(out[status].sum(skipna=True)),
            "n_negative_episode_results": int((~out[status]).sum(skipna=True)),
            "n_ambiguous_episode_matches": int(
                ambiguous.reindex(raw_marker.index, fill_value=False).sum()
            ),
            "n_episode_conflicts": int(
                out.get(f"episode_{marker}_conflict", pd.Series(False, index=out.index))
                .fillna(False)
                .sum()
            ),
        }
        if marker in {"c4", "wbc"}:
            row["n_episodes_with_numeric_value"] = int(
                out.get(f"episode_{marker}_value", pd.Series(np.nan, index=out.index))
                .notna()
                .sum()
            )
        coverage.append(row)
    pd.DataFrame(coverage).to_csv(
        common.BLOCKA_TABLES_DIR / "01_serology_episode_coverage.csv", index=False
    )
    review.to_csv(
        common.BLOCKA_QC_DIR / "01_serology_episode_conflicts.csv", index=False
    )
    unmatched.to_csv(
        common.BLOCKA_QC_DIR / "01_serology_unmatched_longitudinal_records.csv",
        index=False,
    )
    any_serology = (
        out[[STATUS_COL[x] for x in ("ssa", "ssb", "ana", "rf")]].notna().any(axis=1)
    )
    any_dynamic = (
        out[[STATUS_COL[x] for x in ("cryo", "c4", "wbc")]].notna().any(axis=1)
    )
    qc = {
        "n_rows_clinical_spine": len(spine),
        "n_rows_serology_episode_level": len(out),
        "n_unique_patients": out.patient_id.nunique(),
        "n_unique_clinical_episode_ids": out.clinical_episode_id.nunique(),
        "n_unique_patient_episode_keys": out[KEY].drop_duplicates().shape[0],
        "n_duplicate_patient_episode_keys": int(out.duplicated(KEY).sum()),
        "n_nonclinical_rows": int((~_bool(out.clinical_visit)).sum())
        if "clinical_visit" in out
        else 0,
        "n_longitudinal_lab_records_input": len(labs),
        "n_valid_lab_records": int(_bool(labs.result_valid_for_analysis).sum()),
        "n_lab_records_with_matched_episode": int(
            labs.matched_clinical_episode_id.notna().sum()
        ),
        "n_ambiguous_episode_matches": int(
            _bool(
                labs.get("episode_match_ambiguous", pd.Series(False, index=labs.index))
            ).sum()
        ),
        "n_unmatched_lab_records": int(labs.matched_clinical_episode_id.isna().sum()),
        "n_episodes_with_any_serology": int(any_serology.sum()),
        "n_episodes_with_any_dynamic_lab": int(any_dynamic.sum()),
        "n_baseline_patients_20b": baseline.patient_id.nunique(),
        "n_baseline_spine_mismatches": baseline_mismatches,
        "future_leakage_violations": future_leakage_violations,
    }
    for marker in MARKERS:
        qc[f"n_{marker}_episode_conflicts"] = int(
            out.get(f"episode_{marker}_conflict", pd.Series(False, index=out.index))
            .fillna(False)
            .sum()
        )
    with open(
        common.BLOCKA_QC_DIR / "01_serology_episode_qc.json", "w", encoding="utf-8"
    ) as f:
        json.dump(qc, f, indent=2)
    write_table1(baseline[baseline["patient_id"].isin(spine["patient_id"])])
    LOG.info("Wrote %s rows to %s", len(out), args.output)


if __name__ == "__main__":
    main()

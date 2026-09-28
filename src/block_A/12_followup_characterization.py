#!/usr/bin/env python3
"""Describe longitudinal follow-up from the integrated clinical-episode dataset."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402
CANONICAL_PATIENT_ID_COL = "patient_id"
CLINICAL_EPISODE_COL = "clinical_episode_id"
CLINICAL_ANCHOR_DATE_COL = "clinical_anchor_date"
CLINICAL_VISIT_COL = "clinical_visit"
CLINICAL_BASELINE_DATE_COL = "clinical_baseline_date"
PROTOCOL_CANDIDATES = ["source_protocol", "ids__protocol", "ids__protocol_number", "ids__study_protocol", "protocol", "protocol_number", "parent_protocol"]
RETENTION_THRESHOLDS = {"6 months":182,"1 year":365,"2 years":730,"3 years":1095,"5 years":1826,"10 years":3652}
RETENTION_COLUMNS = {"6 months":"has_followup_6mo","1 year":"has_followup_1yr","2 years":"has_followup_2yr","3 years":"has_followup_3yr","5 years":"has_followup_5yr","10 years":"has_followup_10yr"}
def is_missing_value(value):
    return pd.isna(value) or str(value).strip().lower() in {"", "na", "n/a", "nan", "none", "null"}
def select_patient_id_col(frame):
    if "patient_id" not in frame: raise ValueError("Integrated dataset has no patient_id")
    return "patient_id"
def n_pct(n, denominator):
    return f"{n} ({100*n/denominator:.1f}%)" if denominator else "0 (NA)"
def median_iqr(series):
    values=pd.to_numeric(series,errors="coerce").dropna()
    if values.empty: return "NA", {}
    raw={"median":values.median(),"q1":values.quantile(.25),"q3":values.quantile(.75)}
    return f"{raw['median']:.1f} ({raw['q1']:.1f}–{raw['q3']:.1f})", raw
def resolve_protocol_column(df: pd.DataFrame) -> str | None:
    """Return the first populated protocol field containing a recognized code."""
    for column in PROTOCOL_CANDIDATES:
        if column not in df.columns:
            continue
        populated = df[column].map(lambda value: not is_missing_value(value))
        if populated.any() and df.loc[populated, column].map(normalize_protocol_membership).ne("").any():
            return column
    return None


def normalize_protocol_membership(value: object) -> str:
    """Normalize 11D/15D labels while retaining dual membership."""
    if is_missing_value(value):
        return ""
    compact = str(value).upper().replace("-", "").replace(" ", "")
    memberships = []
    if "11D" in compact:
        memberships.append("11D")
    if "15D" in compact:
        memberships.append("15D")
    return " | ".join(memberships)


def prepare_longitudinal_clinical_episodes(
    df: pd.DataFrame, baseline_patient_ids: Iterable[object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and audit the authoritative clinical-episode unit."""
    work = df.copy()
    source = select_patient_id_col(work)
    work[CANONICAL_PATIENT_ID_COL] = work[source].astype("string")
    clinical = work.loc[work[CLINICAL_VISIT_COL].eq(True).fillna(False)].copy()  # noqa: E712
    if baseline_patient_ids is not None:
        wanted = pd.Index(pd.Series(list(baseline_patient_ids), dtype="string"))
        clinical = clinical.loc[clinical[CANONICAL_PATIENT_ID_COL].isin(wanted)].copy()
    duplicate = clinical.duplicated([CANONICAL_PATIENT_ID_COL, CLINICAL_EPISODE_COL], keep=False)
    if duplicate.any():
        examples = clinical.loc[duplicate, [CANONICAL_PATIENT_ID_COL, CLINICAL_EPISODE_COL]].head().to_dict("records")
        raise ValueError(f"Duplicate patient_id + clinical_episode_id values: {examples}")
    clinical[CLINICAL_ANCHOR_DATE_COL] = pd.to_datetime(clinical[CLINICAL_ANCHOR_DATE_COL], errors="coerce")
    clinical[CLINICAL_BASELINE_DATE_COL] = pd.to_datetime(clinical[CLINICAL_BASELINE_DATE_COL], errors="coerce")
    protocol_col = resolve_protocol_column(clinical)
    # Episode protocol and patient-ever membership are different concepts.
    clinical["episode_protocol"] = clinical[protocol_col].map(normalize_protocol_membership) if protocol_col else ""
    memberships = clinical.groupby(CANONICAL_PATIENT_ID_COL)["episode_protocol"].transform(
        lambda values: " | ".join(code for code in ("11D", "15D") if values.str.contains(code, na=False).any()))
    clinical["patient_protocol_membership"] = memberships
    # Compatibility alias: historically this field described the episode.
    clinical["protocol_membership"] = clinical["episode_protocol"]
    clinical["has_valid_anchor_date"] = clinical[CLINICAL_ANCHOR_DATE_COL].notna()
    clinical["is_prebaseline"] = (
        clinical["has_valid_anchor_date"] & clinical[CLINICAL_BASELINE_DATE_COL].notna()
        & (clinical[CLINICAL_ANCHOR_DATE_COL] < clinical[CLINICAL_BASELINE_DATE_COL])
    )
    clinical["included_in_primary_followup"] = (
        clinical["has_valid_anchor_date"] & clinical[CLINICAL_BASELINE_DATE_COL].notna()
        & ~clinical["is_prebaseline"]
    )
    audit_columns = [CANONICAL_PATIENT_ID_COL, CLINICAL_EPISODE_COL, CLINICAL_ANCHOR_DATE_COL,
                     CLINICAL_BASELINE_DATE_COL, CLINICAL_VISIT_COL, "episode_protocol",
                     "patient_protocol_membership",
                     "is_prebaseline", "has_valid_anchor_date", "included_in_primary_followup"]
    return clinical, clinical[audit_columns].copy()


def build_intervisit_gaps(episodes: pd.DataFrame) -> pd.DataFrame:
    columns = [CANONICAL_PATIENT_ID_COL, "previous_clinical_episode_id", CLINICAL_EPISODE_COL,
               "previous_visit_date", "visit_date", "gap_days", "gap_order",
               "gap_zero_days", "gap_negative", "protocol_membership"]
    dated = episodes.loc[episodes["included_in_primary_followup"]].copy()
    dated["_episode_sort"] = dated[CLINICAL_EPISODE_COL].astype("string")
    dated = dated.sort_values([CANONICAL_PATIENT_ID_COL, CLINICAL_ANCHOR_DATE_COL, "_episode_sort"], kind="stable")
    rows = []
    for patient_id, group in dated.groupby(CANONICAL_PATIENT_ID_COL, sort=False):
        previous = None
        for _, row in group.iterrows():
            if previous is not None:
                gap = int((row[CLINICAL_ANCHOR_DATE_COL] - previous[CLINICAL_ANCHOR_DATE_COL]).days)
                rows.append([patient_id, previous[CLINICAL_EPISODE_COL], row[CLINICAL_EPISODE_COL],
                             previous[CLINICAL_ANCHOR_DATE_COL], row[CLINICAL_ANCHOR_DATE_COL], gap,
                             len(rows) + 1, gap == 0, gap < 0, row["protocol_membership"]])
            previous = row
        # Gap order is patient-specific, not a global row number.
        start = len(rows) - max(len(group) - 1, 0)
        for order, target in enumerate(range(start, len(rows)), 1):
            rows[target][6] = order
    return pd.DataFrame(rows, columns=columns)


def build_zero_day_gap_audit(episodes: pd.DataFrame, gaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """Expose same-day consecutive episodes for review without collapsing them."""
    gaps = build_intervisit_gaps(episodes) if gaps is None else gaps
    columns = [CANONICAL_PATIENT_ID_COL, "previous_clinical_episode_id", CLINICAL_EPISODE_COL,
               "previous_clinical_anchor_date", CLINICAL_ANCHOR_DATE_COL, "visit_type", "source_protocol"]
    zero = gaps.loc[gaps["gap_zero_days"]].copy()
    if zero.empty:
        return pd.DataFrame(columns=columns)
    metadata = episodes.set_index([CANONICAL_PATIENT_ID_COL, CLINICAL_EPISODE_COL])
    rows = []
    for row in zero.itertuples(index=False):
        current = metadata.loc[(getattr(row, CANONICAL_PATIENT_ID_COL), getattr(row, CLINICAL_EPISODE_COL))]
        rows.append({CANONICAL_PATIENT_ID_COL: getattr(row, CANONICAL_PATIENT_ID_COL),
                     "previous_clinical_episode_id": row.previous_clinical_episode_id,
                     CLINICAL_EPISODE_COL: getattr(row, CLINICAL_EPISODE_COL),
                     "previous_clinical_anchor_date": row.previous_visit_date,
                     CLINICAL_ANCHOR_DATE_COL: row.visit_date,
                     "visit_type": current.get("visit_type", pd.NA),
                     "source_protocol": current.get("episode_protocol", "")})
    return pd.DataFrame(rows, columns=columns)


def build_patient_followup_metrics(
    episodes: pd.DataFrame, patient_ids: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Build exactly one longitudinal metrics row for every requested patient."""
    if patient_ids is None:
        patient_ids = episodes[CANONICAL_PATIENT_ID_COL].dropna().unique()
    ids = pd.Series(list(patient_ids), dtype="string").drop_duplicates()
    gaps = build_intervisit_gaps(episodes)
    rows = []
    for patient_id in ids:
        all_patient = episodes.loc[episodes[CANONICAL_PATIENT_ID_COL].eq(patient_id)]
        # Known pre-baseline episodes remain audit-only.  Undated authoritative
        # episodes cannot be placed on the timeline, but still count as clinical
        # episodes; only dated, non-pre-baseline episodes drive temporal metrics.
        countable = all_patient.loc[~all_patient["is_prebaseline"]]
        dated = all_patient.loc[all_patient["included_in_primary_followup"]]
        dates = dated[CLINICAL_ANCHOR_DATE_COL].dropna()
        baseline_dates = all_patient[CLINICAL_BASELINE_DATE_COL].dropna()
        baseline_date = baseline_dates.iloc[0] if not baseline_dates.empty else pd.NaT
        first_date, last_date = (dates.min(), dates.max()) if not dates.empty else (pd.NaT, pd.NaT)
        followup_days = float((last_date - baseline_date).days) if pd.notna(last_date) and pd.notna(baseline_date) else np.nan
        patient_gaps = gaps.loc[gaps[CANONICAL_PATIENT_ID_COL].eq(patient_id)]
        valid_gaps = patient_gaps.loc[~patient_gaps["gap_negative"], "gap_days"]
        n_episodes = int(len(countable))
        n_dated_episodes = int(len(dates))
        row = {
            CANONICAL_PATIENT_ID_COL: patient_id, CLINICAL_BASELINE_DATE_COL: baseline_date,
            "first_clinical_date": first_date, "last_clinical_date": last_date,
            "n_clinical_episodes": n_episodes, "n_dated_clinical_episodes": n_dated_episodes,
            "followup_days": followup_days, "followup_years": followup_days / 365.25,
            "median_gap_days": valid_gaps.median() if not valid_gaps.empty else np.nan,
            "max_gap_days": valid_gaps.max() if not valid_gaps.empty else np.nan,
            "has_gap_over_180d": bool((valid_gaps > 180).any()),
            "has_gap_over_365d": bool((valid_gaps > 365).any()),
            "has_gap_over_730d": bool((valid_gaps > 730).any()),
            "patient_protocol_membership": all_patient["patient_protocol_membership"].iloc[0] if len(all_patient) else "",
            "in_protocol_11d": bool(all_patient["patient_protocol_membership"].str.contains("11D", na=False).any()),
            "in_protocol_15d": bool(all_patient["patient_protocol_membership"].str.contains("15D", na=False).any()),
        }
        for label, days in RETENTION_THRESHOLDS.items():
            row[RETENTION_COLUMNS[label]] = bool(pd.notna(followup_days) and followup_days >= days)
        rows.append(row)
    metrics = pd.DataFrame(rows)
    followup_years = metrics["followup_years"].to_numpy(dtype=float)
    n_clinical_episodes = metrics["n_clinical_episodes"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        metrics["visits_per_followup_year"] = np.where(
            followup_years > 0,
            (n_clinical_episodes - 1) / followup_years,
            np.nan,
        )
    return metrics


def _summary_values(metrics: pd.DataFrame, gaps: pd.DataFrame) -> dict[str, object]:
    n = len(metrics)
    valid_gaps = pd.to_numeric(gaps.loc[~gaps["gap_negative"], "gap_days"], errors="coerce").dropna()
    def count_text(mask: pd.Series) -> str:
        return n_pct(int(mask.sum()), n)
    follow_text, _ = median_iqr(metrics["followup_years"])
    episodes_text, _ = median_iqr(metrics["n_clinical_episodes"])
    rate_text, _ = median_iqr(metrics["visits_per_followup_year"])
    gap_q1_q3 = (
        f"{valid_gaps.quantile(.25):.1f}–{valid_gaps.quantile(.75):.1f}"
        if not valid_gaps.empty else "NA"
    )
    values = {
        "Clinical episodes": int(metrics["n_clinical_episodes"].sum()), "Unique patients": n,
        "Patients with exactly 1 clinical episode": count_text(metrics["n_clinical_episodes"].eq(1)),
        "Patients with >=2 clinical episodes": count_text(metrics["n_clinical_episodes"].ge(2)),
        "Patients with >=3 clinical episodes": count_text(metrics["n_clinical_episodes"].ge(3)),
        "Patients with >=5 clinical episodes": count_text(metrics["n_clinical_episodes"].ge(5)),
        "Patients with >=10 clinical episodes": count_text(metrics["n_clinical_episodes"].ge(10)),
        "Follow-up, median (IQR), years": follow_text,
        "Clinical episodes per patient, median (IQR)": episodes_text,
        "Clinical episodes per patient, mean": round(metrics["n_clinical_episodes"].mean(), 1) if n else np.nan,
        "Maximum clinical episodes per patient": int(metrics["n_clinical_episodes"].max()) if n else np.nan,
        "Median inter-visit gap, days": round(valid_gaps.median(), 1) if not valid_gaps.empty else np.nan,
        "IQR inter-visit gap, days": gap_q1_q3,
        "P90 inter-visit gap, days": round(valid_gaps.quantile(.9), 1) if not valid_gaps.empty else np.nan,
        "Clinical episodes per follow-up year, median (IQR)": rate_text,
    }
    for percentile in (10, 25, 50, 75, 90):
        values[f"Follow-up P{percentile}, years"] = round(metrics["followup_years"].quantile(percentile / 100), 1) if n else np.nan
    values["Maximum follow-up, years"] = round(metrics["followup_years"].max(), 1) if n else np.nan
    for label, column in RETENTION_COLUMNS.items():
        values[f"Follow-up >={label}"] = count_text(metrics[column])
    for days in (180, 365, 730):
        values[f"Patients with at least one gap >{days} days"] = count_text(metrics[f"has_gap_over_{days}d"])
    return values


def build_followup_summary(
    overall_metrics: pd.DataFrame, overall_gaps: pd.DataFrame,
    protocol_metrics: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohorts = dict(protocol_metrics or {})
    cohorts["Overall"] = (overall_metrics, overall_gaps)
    values = {name: _summary_values(*parts) for name, parts in cohorts.items()}
    indicators = list(values["Overall"])
    wide = pd.DataFrame({"Indicator": indicators, **{name: [value[i] for i in indicators] for name, value in values.items()}})
    long = wide.melt(id_vars="Indicator", var_name="Cohort", value_name="Value")
    return wide, long


def build_retention_table(cohort_metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cohort, metrics in cohort_metrics.items():
        denominator = len(metrics)
        for label, days in RETENTION_THRESHOLDS.items():
            retained = int(metrics[RETENTION_COLUMNS[label]].sum())
            rows.append({"cohort": cohort, "time": label, "days": days, "n_retained": retained,
                         "denominator": denominator, "pct_retained": round(100 * retained / denominator, 1) if denominator else np.nan})
    return pd.DataFrame(rows)


def build_followup_qc(episodes: pd.DataFrame, metrics: pd.DataFrame, gaps: pd.DataFrame, protocol_column: str | None) -> pd.DataFrame:
    missing = episodes[CLINICAL_ANCHOR_DATE_COL].isna()
    pre = episodes["is_prebaseline"]
    negative = gaps["gap_negative"] if not gaps.empty else pd.Series(dtype=bool)
    checks = {
        "followup_n_unique_patients": metrics[CANONICAL_PATIENT_ID_COL].nunique(),
        "followup_duplicate_patient_episode_ids": episodes.duplicated([CANONICAL_PATIENT_ID_COL, CLINICAL_EPISODE_COL]).sum(),
        "followup_n_patients_missing_baseline_date": metrics[CLINICAL_BASELINE_DATE_COL].isna().sum(),
        "followup_n_clinical_episodes_missing_anchor_date": missing.sum(),
        "followup_n_patients_with_missing_anchor_date": episodes.loc[missing, CANONICAL_PATIENT_ID_COL].nunique(),
        "followup_n_prebaseline_clinical_episodes": pre.sum(),
        "followup_n_patients_with_prebaseline_clinical_episodes": episodes.loc[pre, CANONICAL_PATIENT_ID_COL].nunique(),
        "followup_n_zero_day_gaps": gaps["gap_zero_days"].sum() if not gaps.empty else 0,
        "followup_n_negative_gaps": negative.sum(),
        "followup_n_patients_with_negative_gaps": gaps.loc[negative, CANONICAL_PATIENT_ID_COL].nunique() if not gaps.empty else 0,
        "followup_protocol_column_found": bool(protocol_column),
        "followup_protocol_column_name": protocol_column if protocol_column else np.nan,
        "followup_n_protocol_11d_patients": metrics["in_protocol_11d"].sum() if protocol_column else np.nan,
        "followup_n_protocol_15d_patients": metrics["in_protocol_15d"].sum() if protocol_column else np.nan,
        "followup_n_dual_protocol_patients": (metrics["in_protocol_11d"] & metrics["in_protocol_15d"]).sum() if protocol_column else np.nan,
    }
    rows = [{"qc_check": key, "value": value, "status": "pass", "details": "Longitudinal descriptive QC."} for key, value in checks.items()]
    warning_keys = {"followup_n_clinical_episodes_missing_anchor_date", "followup_n_prebaseline_clinical_episodes",
                    "followup_n_zero_day_gaps", "followup_n_negative_gaps"}
    for row in rows:
        if row["qc_check"] in warning_keys and row["value"]:
            row["status"] = "warning"
        if row["qc_check"] == "followup_protocol_column_found" and not row["value"]:
            row.update(status="warning", details="protocol_column_missing")
        if row["qc_check"] == "followup_protocol_column_name" and not protocol_column:
            row.update(status="warning", details="no_candidate_contains_recognized_11D_or_15D_values")
    return pd.DataFrame(rows)


def validate_followup_hard_qc(metrics: pd.DataFrame, baseline: pd.DataFrame, retention: pd.DataFrame) -> None:
    if metrics[CANONICAL_PATIENT_ID_COL].duplicated().any():
        raise ValueError("Patient follow-up metrics must contain one row per patient")
    if set(metrics[CANONICAL_PATIENT_ID_COL]) != set(baseline[CANONICAL_PATIENT_ID_COL]):
        raise ValueError("Longitudinal and Table 1 patient universes differ")
    if (metrics["n_dated_clinical_episodes"] > metrics["n_clinical_episodes"]).any():
        raise ValueError("Dated clinical episode counts cannot exceed all clinical episode counts")
    if (metrics["followup_days"].dropna() < 0).any() or (metrics["last_clinical_date"].dropna() < metrics.loc[metrics["last_clinical_date"].notna(), CLINICAL_BASELINE_DATE_COL]).any():
        raise ValueError("Invalid negative follow-up")
    ordered = [RETENTION_COLUMNS[x] for x in RETENTION_THRESHOLDS]
    for earlier, later in zip(ordered, ordered[1:]):
        if (metrics[later] & ~metrics[earlier]).any():
            raise ValueError("Retention thresholds are not monotonic")
    if not retention["pct_retained"].dropna().between(0, 100).all():
        raise ValueError("Retention percentages outside 0–100")



def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    parser.add_argument("--tables-dir",type=Path,default=common.BLOCKA_TABLES_DIR/"12_followup_characterization")
    parser.add_argument("--qc-dir",type=Path,default=common.BLOCKA_QC_DIR/"12_followup_characterization")
    parser.add_argument("--analytic-dir",type=Path,default=common.ANALYTIC_DATA_DIR/"blockA"/"12_followup_characterization")
    return parser.parse_args()
def main():
    args=parse_args()
    data=pd.read_parquet(args.input)
    required={"patient_id","clinical_episode_id","clinical_anchor_date","clinical_baseline_date","clinical_visit"}
    missing=sorted(required-set(data))
    if missing: raise ValueError(f"Integrated dataset is missing required columns: {missing}")
    episodes,audit=prepare_longitudinal_clinical_episodes(data)
    metrics=build_patient_followup_metrics(episodes)
    gaps=build_intervisit_gaps(episodes)
    summary,long=build_followup_summary(metrics,gaps)
    retention=build_retention_table({"Overall":metrics})
    validate_followup_hard_qc(metrics,pd.DataFrame({"patient_id":data.patient_id.drop_duplicates()}),retention)
    for directory in (args.tables_dir,args.qc_dir,args.analytic_dir): directory.mkdir(parents=True,exist_ok=True)
    summary.to_csv(args.tables_dir/"12_followup_summary.csv",index=False)
    long.to_csv(args.tables_dir/"12_followup_summary_long.csv",index=False)
    retention.to_csv(args.tables_dir/"12_retention.csv",index=False)
    metrics.to_csv(args.analytic_dir/"12_patient_followup_metrics.csv",index=False)
    gaps.to_csv(args.analytic_dir/"12_intervisit_gaps.csv",index=False)
    audit.to_csv(args.qc_dir/"12_followup_episode_audit.csv",index=False)
    build_zero_day_gap_audit(episodes,gaps).to_csv(args.qc_dir/"12_zero_day_gap_audit.csv",index=False)
    build_followup_qc(episodes,metrics,gaps,resolve_protocol_column(data)).to_csv(args.qc_dir/"12_followup_qc.csv",index=False)
    print(f"Wrote follow-up characterization for {len(metrics):,} patients")
if __name__ == "__main__": main()

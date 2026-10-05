"""Small, study-independent helpers for clinical-episode studies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

STUDY_CONTRACT_VERSION = "clinical_episode_curated_v2"
PUBLIC_REQUIRED = {
    "patient_id", "clinical_episode_id", "clinical_anchor_date",
    "clinical_visit_number", "clinical_visit", "is_clinical_baseline",
    "pop__status", "essdai__total", "esspri__total",
}
CONCEPTS: dict[str, tuple[str, ...]] = {
    "pop": ("pop__status", "pop_status"),
    "age_at_visit": ("demo__age_at_visit", "ids__age_at_visit", "age_at_visit", "age"),
    "age_at_baseline": ("demo__age_at_baseline", "ids__age_at_baseline", "age_at_baseline"),
    "sex": ("demo__sex", "ids__sex", "sex", "gender"),
    "essdai_total": ("essdai__total", "essdai_total"),
    "esspri_total": ("esspri__total", "esspri_total_observed", "esspri_total"),
    "esspri_dryness": ("esspri__dryness", "esspri_dryness", "dryness"),
    "esspri_fatigue": ("esspri__fatigue", "esspri_fatigue", "fatigue"),
    "esspri_pain": ("esspri__pain", "esspri_pain", "pain"),
    "protocol": ("demo__protocol", "protocol", "ids__protocol", "ids__protocol_number", "parent_protocol"),
}
INTERVAL_REQUIRED = {
    "patient_id", "from_clinical_episode_id", "to_clinical_episode_id",
    "from_pop", "to_pop", "interval_days", "interval_years",
}
POP_LEVELS = {"Pop1", "Pop2", "Pop3", "Unclassifiable"}


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Read a dataset without mutating the public Step-10 schema."""
    return pd.read_parquet(Path(path))


def resolve_concept(frame: pd.DataFrame, concept: str, *, allow_legacy_aliases: bool = True) -> str:
    """Resolve a public concept, always preferring the v2 spelling.

    An empty result is never guessed.  Legacy aliases are an explicit, optional
    compatibility path so callers can enforce a public-only contract.
    """
    if concept not in CONCEPTS:
        raise KeyError(f"Unknown dataset concept: {concept}")
    candidates = CONCEPTS[concept] if allow_legacy_aliases else CONCEPTS[concept][:1]
    found = next((column for column in candidates if column in frame.columns), None)
    if found is None:
        raise ValueError(f"Missing required concept {concept!r}; expected one of {list(candidates)}")
    return found


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def validate_integrated_dataset(frame: pd.DataFrame, *, allow_legacy_aliases: bool = True) -> dict:
    """Validate and summarize the immutable integrated episode master."""
    structural = PUBLIC_REQUIRED - {"pop__status", "essdai__total", "esspri__total"}
    _require(frame, structural, "Integrated dataset")
    pop_col = resolve_concept(frame, "pop", allow_legacy_aliases=allow_legacy_aliases)
    essdai_col = resolve_concept(frame, "essdai_total", allow_legacy_aliases=allow_legacy_aliases)
    esspri_col = resolve_concept(frame, "esspri_total", allow_legacy_aliases=allow_legacy_aliases)
    if frame.duplicated(["patient_id", "clinical_episode_id"]).any():
        raise ValueError("Duplicate patient_id + clinical_episode_id keys")
    dates = pd.to_datetime(frame["clinical_anchor_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid clinical_anchor_date")
    visits = pd.to_numeric(frame["clinical_visit_number"], errors="coerce")
    if visits.isna().any() or (visits < 1).any():
        raise ValueError("clinical_visit_number must be >= 1")
    ordered = frame.assign(_date=dates, _visit=visits).sort_values(
        ["patient_id", "_date", "_visit", "clinical_episode_id"]
    )
    if ordered.groupby("patient_id")["_visit"].apply(lambda x: not x.is_monotonic_increasing).any():
        raise ValueError("clinical_visit_number must increase within patient")
    if not frame["clinical_visit"].fillna(False).astype(bool).all():
        raise ValueError("All rows must be clinical visits")
    baseline = frame["is_clinical_baseline"].fillna(False).astype(bool)
    counts = baseline.groupby(frame["patient_id"]).sum()
    if (counts > 1).any():
        raise ValueError("At most one clinical baseline is allowed per patient")
    if "clinical_baseline_episode_id" in frame:
        actual = frame.loc[baseline, ["patient_id", "clinical_episode_id"]]
        expected = frame.groupby("patient_id")["clinical_baseline_episode_id"].first()
        for row in actual.itertuples(index=False):
            value = expected.get(row.patient_id)
            if pd.notna(value) and value != row.clinical_episode_id:
                raise ValueError("Baseline inconsistent with clinical_baseline_episode_id")
    unexpected = set(frame[pop_col].dropna().unique()) - POP_LEVELS
    if unexpected:
        raise ValueError(f"Unexpected pop_status values: {sorted(unexpected)}")
    # The current registry does not guarantee a universal ESSDAI maximum.
    for col, low, high in ((essdai_col, 0, None), (esspri_col, 0, 10)):
        numeric = pd.to_numeric(frame[col], errors="coerce")
        invalid_high = high is not None and numeric.dropna().gt(high).any()
        if numeric[frame[col].notna()].isna().any() or numeric.dropna().lt(low).any() or invalid_high:
            raise ValueError(f"{col} outside [{low}, {high if high is not None else 'unbounded'}]")
    forbidden = [c for c in frame if c.startswith("next_")]
    if forbidden:
        raise ValueError(f"Integrated dataset contains future columns: {forbidden}")
    return {
        "status": "passed", "contract_version": STUDY_CONTRACT_VERSION,
        "n_rows": len(frame), "n_patients": int(frame.patient_id.nunique()),
        "n_patients_with_baseline": int((counts == 1).sum()),
        "n_patients_without_baseline": int((counts == 0).sum()),
        "resolved_columns": {"pop": pop_col, "essdai_total": essdai_col, "esspri_total": esspri_col},
    }


validate_master = validate_integrated_dataset


def validate_transition_intervals(intervals: pd.DataFrame, master: pd.DataFrame) -> dict:
    """Fail on invalid, non-adjacent, or master-discordant intervals."""
    _require(intervals, INTERVAL_REQUIRED, "Transition intervals")
    keys = ["patient_id", "from_clinical_episode_id", "to_clinical_episode_id"]
    if intervals.duplicated(keys).any():
        raise ValueError("Duplicate transition interval keys")
    days = pd.to_numeric(intervals.interval_days, errors="coerce")
    years = pd.to_numeric(intervals.interval_years, errors="coerce")
    if days.isna().any() or years.isna().any() or (days <= 0).any() or (years <= 0).any():
        raise ValueError("Transition intervals must be positive")
    pop_col = resolve_concept(master, "pop")
    lookup = master.set_index(["patient_id", "clinical_episode_id"])[pop_col]
    order = master.assign(_date=pd.to_datetime(master.clinical_anchor_date)).sort_values(
        ["patient_id", "_date", "clinical_visit_number", "clinical_episode_id"]
    )
    successor = {}
    for patient, group in order.groupby("patient_id", sort=False):
        ids = group.clinical_episode_id.tolist()
        successor.update({(patient, ids[i]): ids[i + 1] for i in range(len(ids) - 1)})
    for row in intervals.itertuples(index=False):
        fk = (row.patient_id, row.from_clinical_episode_id)
        tk = (row.patient_id, row.to_clinical_episode_id)
        if fk not in lookup.index or tk not in lookup.index:
            raise ValueError("Transition interval references nonexistent episode key")
        if lookup.loc[fk] != row.from_pop or lookup.loc[tk] != row.to_pop:
            raise ValueError("Transition Pop is discordant with integrated master")
        if successor.get(fk) != row.to_clinical_episode_id:
            raise ValueError("Transition does not connect consecutive clinical episodes")
    return {"status": "passed", "n_intervals": len(intervals)}


validate_intervals = validate_transition_intervals


def validate_predictors(columns: Iterable[str]) -> list[str]:
    """Reject outcome/future/consensus leakage from an explicit predictor list."""
    columns = list(columns)
    bad = [c for c in columns if c.startswith(("to_", "next_", "delta_")) or "patient_consensus" in c.lower()]
    if bad:
        raise ValueError(f"Predictors contain future or consensus information: {bad}")
    return columns


def enrich_transition_intervals(
    intervals: pd.DataFrame, master: pd.DataFrame,
    from_columns: Sequence[str], to_columns: Sequence[str],
) -> pd.DataFrame:
    """Attach explicit from predictors and a minimal set of to outcomes."""
    validate_predictors(from_columns)
    validate_transition_intervals(intervals, master)
    base = intervals.copy()
    left = master[["patient_id", "clinical_episode_id", *from_columns]].rename(
        columns={"clinical_episode_id": "from_clinical_episode_id", **{c: f"from_{c}" for c in from_columns}}
    )
    right = master[["patient_id", "clinical_episode_id", *to_columns]].rename(
        columns={"clinical_episode_id": "to_clinical_episode_id", **{c: f"to_{c}" for c in to_columns}}
    )
    result = base.merge(left, on=["patient_id", "from_clinical_episode_id"], how="left", validate="many_to_one")
    result = result.merge(right, on=["patient_id", "to_clinical_episode_id"], how="left", validate="many_to_one")
    if len(result) != len(base):
        raise ValueError("Interval enrichment changed row count")
    return result


enrich_intervals = enrich_transition_intervals


def benjamini_hochberg(values: Sequence[float]) -> np.ndarray:
    """Return BH-adjusted p-values, preserving missing positions."""
    p = np.asarray(values, dtype=float); out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid): return out
    ranked = valid[np.argsort(p[valid])]; m = len(ranked)
    adjusted = np.minimum.accumulate((p[ranked] * m / np.arange(1, m + 1))[::-1])[::-1]
    out[ranked] = np.minimum(adjusted, 1.0)
    return out


bh_adjust = benjamini_hochberg


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def create_study_dirs(study_name: str) -> Mapping[str, Path]:
    """Create and return standard output roots for a named study."""
    import common
    roots = {"analytic": common.STUDIES_ANALYTIC_DIR, "tables": common.STUDIES_TABLES_DIR,
             "figures": common.STUDIES_FIGURES_DIR, "qc": common.STUDIES_QC_DIR,
             "logs": common.STUDIES_LOGS_DIR}
    result = {key: root / study_name for key, root in roots.items()}
    for path in result.values(): path.mkdir(parents=True, exist_ok=True)
    return result


def json_default(value):
    """JSON serializer for NumPy, pandas, Path, and date-like values."""
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)): return pd.Timestamp(value).isoformat()
    if isinstance(value, Path): return str(value)
    if value is pd.NA: return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, sort_keys=True) + "\n")

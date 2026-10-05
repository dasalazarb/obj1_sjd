"""Reproducible longitudinal laboratory figures and supporting statistics.

This module deliberately consumes the governed episode-level selection made by
``01_serological_profile``.  It does not match records to dates, coerce textual
results, or manufacture visits.  Missing/unusable observations live in an
axis-coordinate availability strip and therefore never acquire a laboratory Y
value.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

GROUPS = ("A", "B", "C")
GROUP_LABELS = {
    "A": "Class 1 history (no class 2)",
    "B": "Ever class 2",
    "C": "Class 4 only",
}
COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
CLASS_4_TRAJECTORY_COLOR = "#CC0000"
COMPARABLE_UNITS = {"same_as_canonical", "alias_normalized", "converted"}
EMPTY_CLASS_TOKENS = {"", "na", "nan", "none", "null", "unknown", "<na>"}
RETROSPECTIVE_NOTE = (
    "Groups use any class documented over complete follow-up (retrospective; "
    "class 2 takes precedence over class 1, and class 1/2 takes precedence over "
    "class 4). Red observations identify patients with a later class 4 transition. "
    "Individual trajectory segments are intentionally omitted from the default "
    "figure for speed and readability; unusable values never enter numeric summaries."
)


def _column(frame: pd.DataFrame, name: str, default=pd.NA) -> pd.Series:
    return frame[name] if name in frame else pd.Series(default, index=frame.index)


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    return series.astype("string").str.strip().str.casefold().isin({"1", "true", "yes", "y"})


def normalize_class_history(value: object) -> tuple[str, ...]:
    """Normalize EDA's pipe-separated Sjogren class history without inference."""
    if isinstance(value, (set, list, tuple, np.ndarray, pd.Series)):
        parts: Iterable[object] = value
    elif pd.isna(value):
        parts = []
    else:
        parts = str(value).split("|")
    tokens: set[str] = set()
    for part in parts:
        token = str(part).strip()
        if token.casefold() in EMPTY_CLASS_TOKENS:
            continue
        try:
            number = float(token)
            if math.isfinite(number) and number.is_integer():
                token = str(int(number))
        except ValueError:
            pass
        tokens.add(token)
    return tuple(sorted(tokens))


def assign_exclusive_class_group(tokens: object) -> str:
    """Assign class 2 first, then 1, retaining class 4 as an overlay.

    Thus a patient documented as 4 and later/as well as 1 remains in group A;
    any documented 2 places the patient in B.  Group C is reserved for patients
    whose only known class is 4.  Unexpected codes are never silently folded
    into A or C.
    """
    history = set(normalize_class_history(tokens))
    if "2" in history:
        return "B"
    if "1" in history and history.issubset({"1", "4"}):
        return "A"
    if history == {"4"}:
        return "C"
    return "unclassified_or_other"


def build_classification(all_spine: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one retrospective history/group per patient and auditable QC."""
    if "patient_id" not in all_spine:
        raise ValueError("all_spine lacks patient_id")
    aggregate = "sjogrens_class_patient_values"
    raw = "visit_summary_form__sjogrens_class"
    rows = []
    discrepancies = 0
    for patient_id, patient in all_spine.groupby("patient_id", dropna=False, sort=False):
        agg_values = set()
        if aggregate in patient:
            for value in patient[aggregate].dropna():
                agg_values.update(normalize_class_history(value))
        rebuilt = set()
        transitioned_to_4 = False
        if raw in patient:
            order_columns = [
                column
                for column in ("clinical_anchor_date", "episode_start_date", "clinical_visit_number")
                if column in patient
            ]
            ordered_patient = patient.sort_values(order_columns, kind="stable") if order_columns else patient
            prior_1_or_2 = False
            for value in ordered_patient[raw].dropna():
                row_tokens = set(normalize_class_history(value))
                if "4" in row_tokens and prior_1_or_2:
                    transitioned_to_4 = True
                prior_1_or_2 = prior_1_or_2 or bool(row_tokens.intersection({"1", "2"}))
                rebuilt.update(row_tokens)
        chosen = agg_values if agg_values else rebuilt
        differs = bool(agg_values and rebuilt and agg_values != rebuilt)
        discrepancies += int(differs)
        rows.append({
            "patient_id": patient_id,
            "class_history": "|".join(sorted(chosen)),
            "class_group": assign_exclusive_class_group(chosen),
            "ever_class_4": "4" in chosen,
            "class_4_trajectory_highlight": (
                transitioned_to_4 and assign_exclusive_class_group(chosen) in {"A", "B"}
            ),
            "class_source": aggregate if agg_values else raw if rebuilt else "unavailable",
            "aggregate_reconstruction_discrepancy": differs,
        })
    classes = pd.DataFrame(rows)
    histories = classes.class_history.map(normalize_class_history).map(set)
    counts = {
        "n_patients": len(classes),
        "n_only_1": int(sum(x == {"1"} for x in histories)),
        "n_ever_2": int(sum("2" in x for x in histories)),
        "n_ever_4": int(sum("4" in x for x in histories)),
        "n_ever_2_and_4": int(sum({"2", "4"}.issubset(x) for x in histories)),
        "n_ever_1_and_2": int(sum({"1", "2"}.issubset(x) for x in histories)),
        "n_other": int(sum(bool(x) and assign_exclusive_class_group(x) == "unclassified_or_other" for x in histories)),
        "n_unknown": int(sum(not x for x in histories)),
        "n_group_A": int(classes.class_group.eq("A").sum()),
        "n_group_B": int(classes.class_group.eq("B").sum()),
        "n_group_C": int(classes.class_group.eq("C").sum()),
        "n_group_A_or_B_later_transition_to_4": int(
            classes.class_4_trajectory_highlight.sum()
        ),
        "n_source_discrepancies": discrepancies,
        "groups_are_mutually_exclusive": True,
    }
    qc = pd.DataFrame([counts])
    if classes.patient_id.duplicated().any():
        raise AssertionError("Classification is not patient-unique")
    return classes, qc


def classify_plot_value_status(row: pd.Series) -> tuple[str, str, float]:
    """Classify one selected episode value, retaining strict numeric semantics."""
    value_type = str(row.get("selected_value_type", "") or "").casefold()
    result_conflict = bool(row.get("result_conflict", False))
    unit_conflict = bool(row.get("unit_conflict", False))
    unit_status = str(row.get("unit_harmonization_status", "") or "")
    if result_conflict or unit_conflict or (value_type == "exact_numeric" and unit_status not in COMPARABLE_UNITS):
        return "noncomparable_or_conflict", "result/unit conflict or noncomparable unit", np.nan
    if value_type == "censored_numeric":
        return "censored_numeric", str(row.get("selected_operator", "censored")), np.nan
    if value_type == "exact_numeric":
        raw = row.get("selected_value_numeric_harmonized", np.nan)
        if isinstance(raw, (bool, np.bool_)):
            return "documented_nonnumeric", "boolean is not a laboratory magnitude", np.nan
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = np.nan
        if np.isfinite(value):
            return "valid_numeric", "", value
        return "documented_nonnumeric", "non-finite or missing exact numeric value", np.nan
    if value_type in {"qualitative", "uninterpretable", "invalid_nonresult"} or pd.notna(row.get("selected_value_text")):
        return "documented_nonnumeric", value_type or "textual result", np.nan
    return "documented_nonnumeric", value_type or "selected result is not quantitative", np.nan


def check_unit_comparability(row: pd.Series) -> tuple[bool, str]:
    """Expose the governed unit decision used by quantitative plots."""
    if bool(row.get("unit_conflict", False)):
        return False, "unit_conflict"
    status = str(row.get("unit_harmonization_status", "") or "")
    return (status in COMPARABLE_UNITS, status or "unit_status_missing")


def build_lab_episode_plot_frame(
    clinical_spine: pd.DataFrame,
    selected_clinical: pd.DataFrame,
    classes: pd.DataFrame,
    raw_records: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expand official visits only for patients with clinical evidence per lab."""
    keys = ["patient_id", "clinical_episode_id", "lab_id"]
    if selected_clinical.duplicated(keys).any():
        raise AssertionError("selected_clinical has duplicate patient/episode/lab keys")
    class_columns = classes.copy()
    for column in ("ever_class_4", "class_4_trajectory_highlight"):
        if column not in class_columns:
            class_columns[column] = False
    eligible_classes = class_columns.loc[
        class_columns.class_group.isin(GROUPS),
        ["patient_id", "class_group", "ever_class_4", "class_4_trajectory_highlight"],
    ]
    visits = clinical_spine.merge(eligible_classes, on="patient_id", how="inner", validate="many_to_one")
    frames = []
    lab_ids = set(selected_clinical.lab_id.dropna())
    if raw_records is not None and not raw_records.empty and "lab_id" in raw_records:
        lab_ids.update(raw_records.lab_id.dropna())
    clinical_key_set = set(map(tuple, clinical_spine[["patient_id", "clinical_episode_id"]].to_numpy()))
    for lab_id in sorted(lab_ids, key=str):
        selected_lab = selected_clinical.loc[selected_clinical.lab_id.eq(lab_id)].copy()
        patients = set(selected_lab.patient_id.dropna().unique())
        if raw_records is not None and not raw_records.empty:
            episode_col = "matched_clinical_episode_id" if "matched_clinical_episode_id" in raw_records else "clinical_episode_id"
            raw_lab = raw_records.loc[raw_records.lab_id.eq(lab_id)]
            if "episode_match_ambiguous" in raw_lab:
                raw_lab = raw_lab.loc[~_truthy(raw_lab.episode_match_ambiguous)]
            patients.update(p for p, e in raw_lab[["patient_id", episode_col]].dropna().to_numpy() if (p, e) in clinical_key_set)
        if not patients:
            continue
        base = visits.loc[visits.patient_id.isin(patients)].copy()
        base["lab_id"] = lab_id
        selected_lab = selected_lab.copy()
        collisions = [c for c in selected_lab if c in base and c not in keys]
        selected_lab = selected_lab.drop(columns=collisions)
        merged = base.merge(selected_lab, on=keys, how="left", validate="one_to_one", indicator="_selected")
        statuses, reasons, values = [], [], []
        for _, row in merged.iterrows():
            if row["_selected"] == "left_only":
                statuses.append("not_measured")
                reasons.append("no selected record associated with official clinical episode")
                values.append(np.nan)
            else:
                status, reason, value = classify_plot_value_status(row)
                statuses.append(status); reasons.append(reason); values.append(value)
        merged["value_status"] = statuses
        merged["marker_reason"] = reasons
        merged["plot_value"] = values
        # A rejected source record may explain an otherwise empty visit.  It is
        # used only when its governed episode ID is explicit and unambiguous.
        if raw_records is not None and not raw_records.empty:
            raw = raw_records.copy()
            episode_col = "matched_clinical_episode_id" if "matched_clinical_episode_id" in raw else "clinical_episode_id"
            raw = raw.rename(columns={episode_col: "clinical_episode_id"}) if episode_col != "clinical_episode_id" else raw
            raw = raw.loc[raw.lab_id.eq(lab_id)]
            if "episode_match_ambiguous" in raw:
                raw = raw.loc[~_truthy(raw.episode_match_ambiguous)]
            rejected = raw.loc[~_truthy(_column(raw, "result_valid_for_analysis", False))]
            rejected_keys = set(map(tuple, rejected[["patient_id", "clinical_episode_id"]].dropna().to_numpy()))
            has_selected = merged.value_status.ne("not_measured")
            source_rejected = pd.Series(
                [(p, e) in rejected_keys for p, e in merged[["patient_id", "clinical_episode_id"]].to_numpy()],
                index=merged.index,
            )
            mask = ~has_selected & source_rejected
            merged.loc[mask, "value_status"] = "excluded_raw_nonresult"
            merged.loc[mask, "marker_reason"] = "source result rejected by governed eligibility rule"
        merged["included_in_mean"] = merged.value_status.eq("valid_numeric")
        merged["included_in_model"] = merged["included_in_mean"] & pd.to_numeric(
            _column(merged, "time_since_clinical_baseline_years"), errors="coerce"
        ).notna()
        merged["raw_selected_value"] = _column(merged, "selected_value_text").where(
            _column(merged, "selected_value_text").notna(), _column(merged, "selected_value_numeric")
        ).astype("string")
        merged["plot_unit"] = _column(merged, "selected_unit_harmonized")
        merged["display_label"] = _column(merged, "canonical_analyte").fillna(str(lab_id))
        frames.append(merged.drop(columns="_selected"))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    # Only internal holes get the pale missing marker; endpoint absences remain
    # not_measured in data/QC but are not drawn.
    out["draw_missing_marker"] = False
    for (_, patient), idx in out.groupby(["lab_id", "patient_id"]).groups.items():
        group = out.loc[idx]
        observed = group.loc[group.value_status.ne("not_measured"), "clinical_visit_number"]
        if len(observed):
            out.loc[idx, "draw_missing_marker"] = group.clinical_visit_number.between(observed.min(), observed.max()) & group.value_status.eq("not_measured")
    return out


def patient_cluster_bootstrap_ci(data: pd.DataFrame, reps: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Bootstrap whole patients and return percentile CIs at each official visit.

    This keeps the original patient-cluster bootstrap semantics but avoids
    rebuilding a pandas DataFrame on every replicate. Patient-by-visit sums and
    counts are pre-aggregated once and resampled with NumPy in small chunks.
    """
    valid = data.loc[data.value_status.eq("valid_numeric")].copy()
    valid["plot_value"] = pd.to_numeric(valid["plot_value"], errors="coerce")
    valid = valid.dropna(subset=["patient_id", "clinical_visit_number", "plot_value"])
    visits = sorted(valid.clinical_visit_number.unique())
    patients = valid.patient_id.drop_duplicates().to_numpy()

    if len(patients) < 2 or not visits:
        return pd.DataFrame({
            "clinical_visit_number": visits,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        })

    patient_visit = (
        valid.groupby(["patient_id", "clinical_visit_number"], sort=False)["plot_value"]
        .agg(["sum", "count"])
        .reset_index()
    )
    patient_index = pd.Index(patients, name="patient_id")
    visit_index = pd.Index(visits, name="clinical_visit_number")
    sums = (
        patient_visit.pivot(index="patient_id", columns="clinical_visit_number", values="sum")
        .reindex(index=patient_index, columns=visit_index)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    counts = (
        patient_visit.pivot(index="patient_id", columns="clinical_visit_number", values="count")
        .reindex(index=patient_index, columns=visit_index)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    rng = np.random.default_rng(seed)
    draws = np.full((reps, len(visits)), np.nan, dtype=float)
    chunk_size = min(500, max(1, reps))
    for start in range(0, reps, chunk_size):
        stop = min(start + chunk_size, reps)
        sampled_idx = rng.integers(
            0, len(patients), size=(stop - start, len(patients))
        )
        sampled_sums = sums[sampled_idx].sum(axis=1)
        sampled_counts = counts[sampled_idx].sum(axis=1)
        np.divide(
            sampled_sums,
            sampled_counts,
            out=draws[start:stop],
            where=sampled_counts > 0,
        )

    return pd.DataFrame([
        {
            "clinical_visit_number": visit,
            "ci95_low": np.nanpercentile(draws[:, j], 2.5)
            if np.isfinite(draws[:, j]).any()
            else np.nan,
            "ci95_high": np.nanpercentile(draws[:, j], 97.5)
            if np.isfinite(draws[:, j]).any()
            else np.nan,
        }
        for j, visit in enumerate(visits)
    ])

def summarize_observed_by_group_visit(plot_data: pd.DataFrame, reps: int = 2000, seed: int = 42) -> pd.DataFrame:
    rows = []
    if plot_data.empty:
        return pd.DataFrame()
    for (lab_id, group), group_data in plot_data.groupby(["lab_id", "class_group"], sort=True):
        ci = patient_cluster_bootstrap_ci(group_data, reps=reps, seed=seed).set_index("clinical_visit_number")
        for visit, data in group_data.groupby("clinical_visit_number", sort=True):
            numeric = data.loc[data.value_status.eq("valid_numeric")]
            values = numeric.plot_value
            rows.append({
                "lab_id": lab_id, "class_group": group, "clinical_visit_number": visit,
                "n_total_group": group_data.patient_id.nunique(),
                "n_patients_with_any_lab_record": data.loc[data.value_status.ne("not_measured"), "patient_id"].nunique(),
                "n_patients_numeric": numeric.patient_id.nunique(), "n_numeric_values": len(values),
                "n_documented_nonnumeric": int(data.value_status.isin(["documented_nonnumeric", "excluded_raw_nonresult"]).sum()),
                "n_not_measured": int(data.value_status.eq("not_measured").sum()),
                "n_censored": int(data.value_status.eq("censored_numeric").sum()),
                "n_conflict_or_noncomparable": int(data.value_status.eq("noncomparable_or_conflict").sum()),
                "mean": values.mean(), "median": values.median(), "sd": values.std(),
                "q25": values.quantile(.25), "q75": values.quantile(.75),
                "ci95_low": ci.loc[visit, "ci95_low"] if visit in ci.index and len(values) >= 2 else np.nan,
                "ci95_high": ci.loc[visit, "ci95_high"] if visit in ci.index and len(values) >= 2 else np.nan,
                "unit": _column(numeric, "plot_unit").dropna().iloc[0] if _column(numeric, "plot_unit").notna().any() else pd.NA,
                "display_label": _column(data, "display_label").dropna().iloc[0] if _column(data, "display_label").notna().any() else lab_id,
                "visit_date_min": pd.to_datetime(_column(data, "clinical_anchor_date"), errors="coerce").min(),
                "visit_date_max": pd.to_datetime(_column(data, "clinical_anchor_date"), errors="coerce").max(),
            })
    return pd.DataFrame(rows)


def compute_shared_axes(values: Iterable[float], ci_values: Iterable[float] = ()) -> tuple[float, float]:
    array = np.asarray(list(values) + list(ci_values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return 0.0, 1.0
    low, high = float(array.min()), float(array.max())
    span = high - low
    margin = .05 * span if span else max(abs(low) * .05, .5)
    return low - margin, high + margin


def fit_lab_time_group_mixed_model(data: pd.DataFrame, min_repeats_per_group: int = 10):
    """Fit ML random-intercept LMM and LRT the two interaction terms."""
    valid = data.loc[data.included_in_model].copy()
    valid["time_years"] = pd.to_numeric(valid.time_since_clinical_baseline_years, errors="coerce")
    valid = valid.dropna(subset=["time_years", "plot_value", "patient_id", "class_group"])
    repeats = valid.groupby(["class_group", "patient_id"]).agg(n=("plot_value", "size"), span=("time_years", lambda x: x.max()-x.min()))
    repeated = repeats.loc[(repeats.n >= 2) & (repeats.span > 0)].groupby(level=0).size()
    base = {"n_observations": len(valid), "n_patients": valid.patient_id.nunique(),
            **{f"n_repeated_{g}": int(repeated.get(g, 0)) for g in GROUPS},
            "p_raw": np.nan, "test_method": "ML likelihood-ratio test", "retrospective_group_warning": RETROSPECTIVE_NOTE}
    if valid.empty:
        return {**base, "statistical_status": "all_nonnumeric"}, pd.DataFrame()
    if set(valid.class_group.unique()) != set(GROUPS):
        return {**base, "statistical_status": "missing_group"}, pd.DataFrame()
    if any(repeated.get(g, 0) < min_repeats_per_group for g in GROUPS) or valid.time_years.nunique() < 3:
        return {**base, "statistical_status": "insufficient_repeats"}, pd.DataFrame()
    try:
        import statsmodels.formula.api as smf
        from scipy.stats import chi2, norm
        full = smf.mixedlm("plot_value ~ time_years * C(class_group, Treatment('A'))", valid, groups=valid.patient_id).fit(reml=False, method="lbfgs")
        reduced = smf.mixedlm("plot_value ~ time_years + C(class_group, Treatment('A'))", valid, groups=valid.patient_id).fit(reml=False, method="lbfgs")
        if not full.converged or not reduced.converged:
            return {**base, "statistical_status": "nonconvergent"}, pd.DataFrame()
        statistic = max(0., 2 * (full.llf - reduced.llf))
        p = float(chi2.sf(statistic, 2))
        names = {
            "B": "time_years:C(class_group, Treatment('A'))[T.B]",
            "C": "time_years:C(class_group, Treatment('A'))[T.C]",
        }
        cov = full.cov_params()
        estimates = {g: float(full.params[n]) for g, n in names.items()}
        contrasts = []
        for label, estimate, variance in (
            ("B-A", estimates["B"], cov.loc[names["B"], names["B"]]),
            ("C-A", estimates["C"], cov.loc[names["C"], names["C"]]),
            ("C-B", estimates["C"]-estimates["B"], cov.loc[names["C"], names["C"]]+cov.loc[names["B"], names["B"]]-2*cov.loc[names["C"], names["B"]]),
        ):
            se = math.sqrt(max(variance, 0.)); raw_p = float(2*norm.sf(abs(estimate/se))) if se else np.nan
            contrasts.append({"contrast": label, "effect_per_year": estimate, "std_error": se,
                              "ci95_low": estimate-1.96*se, "ci95_high": estimate+1.96*se, "p_raw": raw_p})
        pairwise = pd.DataFrame(contrasts)
        order = pairwise.p_raw.sort_values().index
        adjusted = np.maximum.accumulate([min(1., pairwise.loc[i, "p_raw"]*(len(pairwise)-rank)) for rank, i in enumerate(order)])
        pairwise.loc[order, "p_holm"] = adjusted
        return {**base, "statistical_status": "ok", "lr_statistic": statistic, "df": 2, "p_raw": p,
                "random_intercept_variance": float(full.cov_re.iloc[0, 0])}, pairwise
    except np.linalg.LinAlgError:
        return {**base, "statistical_status": "singular_design"}, pd.DataFrame()
    except Exception:
        return {**base, "statistical_status": "model_error", "error": traceback.format_exc()}, pd.DataFrame()


def adjust_global_lab_pvalues_bh(models: pd.DataFrame) -> pd.DataFrame:
    out = models.copy(); out["q_BH"] = np.nan
    mask = out.statistical_status.eq("ok") & pd.to_numeric(out.p_raw, errors="coerce").notna()
    p = pd.to_numeric(out.loc[mask, "p_raw"])
    if len(p):
        order = p.sort_values().index
        ranked = p.loc[order].to_numpy(); m = len(ranked)
        q = np.minimum.accumulate((ranked*m/np.arange(1, m+1))[::-1])[::-1]
        out.loc[order, "q_BH"] = np.minimum(q, 1.)
    return out


def safe_lab_slugs(lab_ids: Iterable[object]) -> dict[object, str]:
    result, used = {}, {}
    for lab_id in sorted(set(lab_ids), key=str):
        base = re.sub(r"[^a-z0-9_-]+", "-", str(lab_id).casefold()).strip("-_") or "lab"
        slug = base
        if slug in used and used[slug] != lab_id:
            slug = f"{base}-{hashlib.sha256(str(lab_id).encode()).hexdigest()[:8]}"
        used[slug] = lab_id; result[lab_id] = slug
    return result


def build_lab_plot_manifest(lab_ids: Iterable[object]) -> pd.DataFrame:
    """Create the deterministic one-to-one lab-to-filename mapping."""
    slugs = safe_lab_slugs(lab_ids)
    return pd.DataFrame(
        [{"lab_id": lab_id, "lab_slug": slug} for lab_id, slug in slugs.items()]
    )


def render_lab_trajectory_panels(data: pd.DataFrame, summary: pd.DataFrame):
    """Render a compact visit-level trajectory figure.

    The default figure avoids creating one line artist per patient and segment.
    Raw numeric observations are drawn in at most two scatter collections per
    panel, while the visit-level mean and bootstrap CI remain the visual focus.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2), sharex=True, sharey=True)
    lab_id = str(data.lab_id.iloc[0])
    label = str(data.display_label.dropna().iloc[0])
    visits = sorted(data.clinical_visit_number.dropna().unique())
    ci_values = summary[["ci95_low", "ci95_high"]].to_numpy().ravel() if len(summary) else []
    limits = compute_shared_axes(data.plot_value, ci_values)
    unit_values = data.plot_unit.dropna().astype(str).unique()
    unit = unit_values[0] if len(unit_values) == 1 else "unit unavailable/noncomparable"

    for ax, group, panel in zip(axes, GROUPS, "ABC"):
        gd = data.loc[data.class_group.eq(group)]
        sd = summary.loc[summary.class_group.eq(group)].sort_values("clinical_visit_number")
        numeric = gd.loc[gd.value_status.eq("valid_numeric")].copy()
        highlight = numeric.class_4_trajectory_highlight.fillna(False)

        if (~highlight).any():
            ax.scatter(
                numeric.loc[~highlight, "clinical_visit_number"],
                numeric.loc[~highlight, "plot_value"],
                s=11,
                color=COLORS[group],
                alpha=.16,
                linewidths=0,
                zorder=2,
            )
        if highlight.any():
            ax.scatter(
                numeric.loc[highlight, "clinical_visit_number"],
                numeric.loc[highlight, "plot_value"],
                s=13,
                color=CLASS_4_TRAJECTORY_COLOR,
                alpha=.28,
                linewidths=0,
                zorder=3,
            )

        finite_ci = sd.ci95_low.notna() & sd.ci95_high.notna()
        if finite_ci.any():
            ax.fill_between(
                sd.loc[finite_ci, "clinical_visit_number"].astype(float),
                sd.loc[finite_ci, "ci95_low"].astype(float),
                sd.loc[finite_ci, "ci95_high"].astype(float),
                color=COLORS[group],
                alpha=.14,
                linewidth=0,
            )
        if len(sd):
            ax.plot(
                sd.clinical_visit_number,
                sd["mean"],
                color=COLORS[group],
                lw=2.4,
                marker="o",
                ms=5,
                zorder=5,
            )

        repeated = numeric.groupby("patient_id").size().ge(2).sum()
        ax.set_title(
            f"{panel}. {GROUP_LABELS[group]}\n"
            f"N={gd.patient_id.nunique()} | numeric={numeric.patient_id.nunique()} | repeated={repeated}",
            fontsize=10,
        )
        ax.set_ylim(*limits)
        ax.set_xticks(visits, [str(int(v)) for v in visits])
        ax.set_xlabel("Clinical visit")
        ax.grid(axis="y", color="#E5E5E5", lw=.5)
        ax.set_axisbelow(True)

        for visit in visits:
            n = sd.loc[sd.clinical_visit_number.eq(visit), "n_patients_numeric"]
            if len(n):
                ax.text(
                    visit,
                    -.09,
                    f"n={int(n.iloc[0])}",
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                )

    axes[0].set_ylabel(f"{label} ({unit})")
    fig.suptitle(
        f"{label}: longitudinal laboratory profile\n"
        f"Observed values with group mean and 95% patient-cluster bootstrap CI · lab_id: {lab_id}",
        fontsize=13,
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777",
               markeredgewidth=0, alpha=.35, label="Observed numeric value"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_4_TRAJECTORY_COLOR,
               markeredgewidth=0, alpha=.45, label="Patient later transitioned to class 4"),
        Line2D([0], [0], color="#222222", lw=2.4, marker="o", label="Observed group mean"),
        Patch(facecolor="#777777", alpha=.14, label="95% bootstrap CI"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(.5, .09),
        frameon=False,
        fontsize=8.5,
    )
    fig.text(.5, .025, RETROSPECTIVE_NOTE, ha="center", fontsize=8)
    fig.subplots_adjust(top=.76, bottom=.23, left=.075, right=.99, wspace=.08)
    return fig

def render_lab_elapsed_time_panels(data: pd.DataFrame):
    """Supplementary real-time spaghetti plot; it never labels raw means as model estimates."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8), sharex=True, sharey=True)
    valid = data.loc[data.value_status.eq("valid_numeric")].copy()
    limits = compute_shared_axes(valid.plot_value)
    for ax, group, panel in zip(axes, GROUPS, "ABC"):
        gd = valid.loc[valid.class_group.eq(group)]
        for _, patient in gd.groupby("patient_id"):
            patient = patient.sort_values("time_since_clinical_baseline_years")
            trajectory_color = (
                CLASS_4_TRAJECTORY_COLOR
                if patient.class_4_trajectory_highlight.fillna(False).any()
                else COLORS[group]
            )
            ax.plot(patient.time_since_clinical_baseline_years, patient.plot_value, "o-", color=trajectory_color, alpha=.18, lw=.7, ms=2.5)
        ax.set_title(f"{panel}. {GROUP_LABELS[group]}"); ax.set_ylim(*limits); ax.grid(axis="y", color="#E5E5E5", lw=.5)
        ax.set_xlabel("Years since official clinical baseline")
    label = str(data.display_label.dropna().iloc[0]); axes[0].set_ylabel(label)
    fig.suptitle(f"Longitudinal trajectory of {label} by elapsed time\nIndividual observations; no interpolated visit-wise mean")
    fig.text(.5, .025, RETROSPECTIVE_NOTE, ha="center", fontsize=8.5); fig.subplots_adjust(top=.79, bottom=.18, wspace=.08)
    return fig


def export_figure_pair(fig, pdf_path: Path, png_path: Path, dpi: int = 400) -> None:
    """Write both formats through same-directory temporary files then replace."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True); png_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = []
    try:
        for path, kwargs in ((pdf_path, {}), (png_path, {"dpi": dpi})):
            fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent); os.close(fd)
            tmp = Path(name); temporary.append(tmp)
            fig.savefig(tmp, bbox_inches="tight", facecolor="white", **kwargs); os.replace(tmp, path)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def run_longitudinal_pipeline(*, all_spine: pd.DataFrame, clinical_spine: pd.DataFrame,
                              selected_all: pd.DataFrame, selected_clinical: pd.DataFrame,
                              raw_records: pd.DataFrame | None, intermediate_path: Path,
                              tables_dir: Path, qc_dir: Path, figures_dir: Path,
                              dpi: int = 400, bootstrap_reps: int = 2000, seed: int = 42,
                              min_repeats_per_group: int = 10, include_time_facets: bool = False,
                              plot_labs: set[str] | None = None, max_patients_per_lab: int | None = None) -> dict:
    """Build all longitudinal artifacts without changing Step 01 legacy outputs."""
    import matplotlib.pyplot as plt
    for directory in (intermediate_path.parent, tables_dir, qc_dir, figures_dir): directory.mkdir(parents=True, exist_ok=True)
    classes, class_qc = build_classification(all_spine)
    class_qc.to_csv(qc_dir / "01_labs_longitudinal_class_group_qc.csv", index=False)
    clinical = selected_clinical
    if plot_labs: clinical = clinical.loc[clinical.lab_id.astype(str).isin(plot_labs)]
    plot_data = build_lab_episode_plot_frame(clinical_spine, clinical, classes, raw_records)
    if max_patients_per_lab:
        # Explicit debugging-only deterministic cap; default is no subsampling.
        keep = plot_data.groupby("lab_id").patient_id.transform(lambda x: x.isin(sorted(x.dropna().unique(), key=str)[:max_patients_per_lab]))
        plot_data = plot_data.loc[keep].copy()
    plot_data.to_parquet(intermediate_path, index=False)
    summary = summarize_observed_by_group_visit(plot_data, bootstrap_reps, seed)
    summary.to_csv(tables_dir / "01_labs_longitudinal_summary_by_visit.csv", index=False)
    model_rows, pair_rows = [], []
    for lab_id, data in plot_data.groupby("lab_id", sort=True):
        model, pairs = fit_lab_time_group_mixed_model(data, min_repeats_per_group)
        model["lab_id"] = lab_id; model_rows.append(model)
        if len(pairs): pairs.insert(0, "lab_id", lab_id); pair_rows.append(pairs)
    models = adjust_global_lab_pvalues_bh(pd.DataFrame(model_rows)) if model_rows else pd.DataFrame(columns=["lab_id","statistical_status","p_raw","q_BH"])
    models.to_csv(tables_dir / "01_labs_longitudinal_models.csv", index=False)
    pd.concat(pair_rows, ignore_index=True).to_csv(tables_dir / "01_labs_longitudinal_pairwise_contrasts.csv", index=False) if pair_rows else pd.DataFrame(columns=["lab_id","contrast","effect_per_year","p_raw","p_holm"]).to_csv(tables_dir / "01_labs_longitudinal_pairwise_contrasts.csv", index=False)
    invalid = plot_data.loc[~plot_data.value_status.isin(["valid_numeric", "not_measured"])]
    invalid.to_csv(qc_dir / "01_labs_longitudinal_invalid_values_qc.csv", index=False)
    plot_data.groupby(["lab_id", "plot_unit", "value_status"], dropna=False).size().rename("n").reset_index().to_csv(qc_dir / "01_labs_longitudinal_unit_comparability_qc.csv", index=False)
    catalog = [selected_all.lab_id, clinical.lab_id]
    if raw_records is not None and "lab_id" in raw_records:
        episode_col = "matched_clinical_episode_id" if "matched_clinical_episode_id" in raw_records else "clinical_episode_id"
        all_keys = set(map(tuple, all_spine[["patient_id", "clinical_episode_id"]].to_numpy()))
        safely_matched = pd.Series(
            [(p, e) in all_keys for p, e in raw_records[["patient_id", episode_col]].to_numpy()],
            index=raw_records.index,
        )
        if "episode_match_ambiguous" in raw_records:
            safely_matched &= ~_truthy(raw_records.episode_match_ambiguous)
        catalog.append(raw_records.loc[safely_matched, "lab_id"])
    slugs = safe_lab_slugs(pd.concat(catalog).dropna())
    manifest = []
    clinical_ids = set(plot_data.lab_id.unique()) if len(plot_data) else set()
    git_commit = _git_commit()
    model_lookup = models.set_index("lab_id").to_dict("index") if len(models) else {}
    for lab_id in sorted(slugs, key=str):
        slug = slugs[lab_id]; data = plot_data.loc[plot_data.lab_id.eq(lab_id)]
        model = model_lookup.get(
            lab_id,
            {"statistical_status": "descriptive_only", "p_raw": np.nan, "q_BH": np.nan},
        )
        row = {"lab_id": lab_id, "display_label": data.display_label.dropna().iloc[0] if len(data) and data.display_label.notna().any() else lab_id,
               "unit": "|".join(sorted(data.plot_unit.dropna().astype(str).unique())) if len(data) else "",
               "group_sizes": json.dumps({g: int(data.loc[data.class_group.eq(g), 'patient_id'].nunique()) for g in GROUPS}),
               "n_numeric_total": int(data.value_status.eq("valid_numeric").sum()) if len(data) else 0,
               "n_invalid_total": int((~data.value_status.isin(["valid_numeric","not_measured"])).sum()) if len(data) else 0,
               "model_status": model["statistical_status"], "p_raw": model.get("p_raw"), "q_BH": model.get("q_BH"),
               "pdf_path": "", "png_path": "", "temporal_pdf_path": "", "temporal_png_path": "",
               "render_status": "skipped", "skip_reason": "nonclinical_only_or_no_classified_patient" if lab_id not in clinical_ids else "",
               "git_commit": git_commit, "created_utc": datetime.now(timezone.utc).isoformat()}
        if len(data):
            pdf = figures_dir / f"01_lab_trajectory__{slug}.pdf"; png = figures_dir / f"01_lab_trajectory__{slug}.png"
            fig = None
            try:
                fig = render_lab_trajectory_panels(data, summary.loc[summary.lab_id.eq(lab_id)])
                export_figure_pair(fig, pdf, png, dpi); row.update(pdf_path=str(pdf), png_path=str(png), render_status="ok")
            except Exception:
                row.update(render_status="error", skip_reason=traceback.format_exc())
            finally:
                if fig is not None: plt.close(fig)
            if include_time_facets and data.included_in_model.any():
                tpdf = figures_dir / f"01_lab_trajectory_time__{slug}.pdf"; tpng = figures_dir / f"01_lab_trajectory_time__{slug}.png"; fig = None
                try:
                    fig = render_lab_elapsed_time_panels(data); export_figure_pair(fig, tpdf, tpng, dpi); row.update(temporal_pdf_path=str(tpdf), temporal_png_path=str(tpng))
                finally:
                    if fig is not None: plt.close(fig)
        manifest.append(row)
    manifest_df = pd.DataFrame(manifest); manifest_df.to_csv(tables_dir / "01_labs_longitudinal_figure_manifest.csv", index=False)
    qc = {"n_lab_ids": len(manifest_df), "n_clinical_lab_ids": len(clinical_ids), "n_numeric_lab_ids": int(sum(plot_data.groupby('lab_id').value_status.apply(lambda x: x.eq('valid_numeric').any()))) if len(plot_data) else 0,
          "n_all_nonnumeric_lab_ids": int(sum(plot_data.groupby('lab_id').value_status.apply(lambda x: not x.eq('valid_numeric').any()))) if len(plot_data) else 0,
          "n_complete_figure_pairs": int(manifest_df.render_status.eq('ok').sum()) if len(manifest_df) else 0,
          "n_models_ok": int(models.statistical_status.eq('ok').sum()) if len(models) else 0,
          "retrospective_group_warning": RETROSPECTIVE_NOTE}
    (qc_dir / "01_labs_longitudinal_plot_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    log_dir = figures_dir.parents[2] / "logs" / "01_serological_profile"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "01_labs_longitudinal_plots.log").write_text(
        json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), **qc}, indent=2),
        encoding="utf-8",
    )
    return qc


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"

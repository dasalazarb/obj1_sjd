"""Isolated, retrospective ESSDAI descriptions; never baseline/FROM predictors.

Patient products and unsuppressed aggregates are controlled research artifacts.
No writer publishes to an external service or certifies clinical review.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from config import (ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS, ESSDAI_MAX_VALID,
                    ESSDAI_MODERATE_OR_HIGH_CUTOFF, ESSDAI_HIGH_CUTOFF)
from src.block_A.observed_patterns import verified_baseline_pop, build_observed_patterns

DEFINITION_VERSION = "observed-essdai-pattern-v2"
POLICY = "last_canonical_episode_per_date"
STEM = "13_disease_activity_progression"
ESSDAI = "essdai__total"
DATE = "clinical_anchor_date"
KEYS = ["patient_id", "clinical_episode_id"]
PATTERNS = ("within_threshold_observed", "worsening_observed",
            "improvement_observed", "intermittent_observed")
INSUFFICIENT = "insufficient_observations"
OFFICIAL = "official_baseline"
FALLBACK = "first_observed_after_official_baseline"
REQUIRED = KEYS + [DATE, "clinical_baseline_episode_id", "clinical_baseline_date",
                   "is_clinical_baseline", "time_since_clinical_baseline_years", ESSDAI]
META_COLUMNS = ["patient_id", "definition_version", "date_policy", "sensitivity_flag",
    "official_baseline_episode_id", "official_clinical_baseline_date",
    "official_baseline_essdai_available", "essdai_index_source", "essdai_index_episode_id",
    "essdai_index_date", "essdai_index_score", "days_from_clinical_baseline_to_essdai_index",
    "n_distinct_essdai_dates", "n_essdai_followup_dates", "n_canonical_essdai_episodes",
    "n_same_date_extra_episodes", "n_discordant_essdai_dates", "n_excluded_discordant_dates",
    "n_undated_essdai_episodes", "n_missing_essdai_episodes", "n_prebaseline_essdai_episodes",
    "n_official_baseline_same_date_companions", "followup_years_from_essdai_index",
    "assessment_frequency_per_year", "baseline_state_official", "index_activity_state",
    "baseline_pop_official", "cohort", "eligibility_status", "exclusion_reason"]
CHANGE_COLUMNS = ["threshold_abs", "observed_pattern", "trajectory_evidence_level",
    "intermittence_evaluable", "ever_up", "ever_down", "up_reversion", "down_reversion",
    "first_observed_up_date", "first_observed_down_date", "max_delta_from_essdai_index",
    "min_delta_from_essdai_index", "last_delta_from_essdai_index",
    "observed_change_direction_last", "last_up", "last_down", "n_direction_changes",
    "n_threshold_crossings"]
PATIENT_COLUMNS = META_COLUMNS + CHANGE_COLUMNS
SEQUENCE_COLUMNS = KEYS + [DATE, ESSDAI, "essdai_index_date", "time_since_essdai_index_years"]


def activity(score):
    if pd.isna(score):
        return "Missing"
    return ("Low" if score < ESSDAI_MODERATE_OR_HIGH_CUTOFF else
            "Moderate" if score < ESSDAI_HIGH_CUTOFF else "High")


def validate_pattern_inputs(df, baseline):
    """Copy and validate raw inputs, before legacy coercion can conceal errors.

    Missing reliable dates exclude the affected subject, never use nominal time.
    Malformed dates/scores, inconsistent official records or times block a bundle.
    """
    frames = []
    for name, original in (("Step 10", df), ("Step 11", baseline)):
        missing = set(REQUIRED) - set(original)
        if missing:
            raise ValueError(f"{name}: missing required columns {sorted(missing)}")
        x = original.copy(deep=True)
        for key in KEYS + ["clinical_baseline_episode_id"]:
            x[key] = x[key].astype("string")
            if x[key].isna().any() or x[key].str.strip().eq("").any():
                raise ValueError(f"{name}: missing canonical identifiers")
        if x.duplicated(KEYS).any():
            raise ValueError(f"{name}: duplicate episode keys")
        if not x.is_clinical_baseline.isin([True, False]).all():
            raise ValueError(f"{name}: invalid baseline indicator")
        for col in (DATE, "clinical_baseline_date"):
            parsed = pd.to_datetime(x[col], errors="coerce", format="mixed")
            if (x[col].notna() & parsed.isna()).any():
                raise ValueError(f"{name}: malformed real date in {col}")
            if not pd.api.types.is_datetime64_dtype(parsed):
                raise ValueError(f"{name}: dates must be timezone-naive calendar dates")
            if parsed.dropna().ne(parsed.dropna().dt.normalize()).any():
                raise ValueError(f"{name}: expected calendar dates, not timestamps")
            x[col] = parsed
        values = pd.to_numeric(x[ESSDAI], errors="coerce")
        if (x[ESSDAI].notna() & values.isna()).any() or (~values.dropna().between(0, ESSDAI_MAX_VALID)).any():
            raise ValueError(f"{name}: invalid ESSDAI score (range 0-{ESSDAI_MAX_VALID})")
        x[ESSDAI] = values.astype(float)
        for col, divisor in (("time_since_clinical_baseline_years", 365.25),
                             ("time_since_clinical_baseline_days", 1.)):
            if col not in x:
                continue
            times = pd.to_numeric(x[col], errors="coerce")
            if (x[col].notna() & times.isna()).any() or times.dropna().lt(0).any():
                raise ValueError(f"{name}: invalid or negative clinical interval")
            actual = (x[DATE] - x.clinical_baseline_date).dt.days / divisor
            known = actual.notna()
            if (known & (times.isna() | ~np.isclose(times, actual, atol=1e-7, rtol=1e-7))).any():
                raise ValueError(f"{name}: clinical interval conflicts with real dates")
            x[col] = times
        frames.append(x)
    x, b = frames
    official = x.loc[x.is_clinical_baseline.eq(True)].set_index("patient_id")
    if official.index.duplicated().any() or set(official.index) != set(x.patient_id):
        raise ValueError("exactly one official clinical baseline required per patient")
    if b.patient_id.duplicated().any() or not b.is_clinical_baseline.eq(True).all() or set(b.patient_id) != set(official.index):
        raise ValueError("Step 11 must match the complete unique official baseline cohort")
    b = b.set_index("patient_id").reindex(official.index)
    # Compare protected fields and published Pop/ESSPRI wherever shared.
    for col in list(dict.fromkeys(REQUIRED[1:] + ["esspri__total", "pop__status"])):
        if col not in b or col not in official:
            continue
        equal = official[col].eq(b[col]) | (official[col].isna() & b[col].isna())
        if not equal.fillna(False).all():
            raise ValueError(f"Step 11 baseline conflicts with Step 10: {col}")
    if not official.clinical_episode_id.eq(official.clinical_baseline_episode_id).all():
        raise ValueError("official baseline episode ID mismatch")
    dates_equal = official[DATE].eq(official.clinical_baseline_date) | (official[DATE].isna() & official.clinical_baseline_date.isna())
    if not dates_equal.all() or not official.time_since_clinical_baseline_years.eq(0).all():
        raise ValueError("official baseline date/time mismatch")
    for col in ("clinical_baseline_episode_id", "clinical_baseline_date"):
        expected = x.patient_id.map(official[col])
        if not (x[col].eq(expected) | (x[col].isna() & expected.isna())).all():
            raise ValueError("clinical baseline metadata changed across episodes")
    b = b.reset_index()
    verified_baseline_pop(b)  # rejects conflicting published Pop; never derives it
    return x.sort_values(KEYS[:1] + [DATE, "clinical_episode_id"], kind="stable"), b


def classify_sequence(scores, dates, threshold):
    """Inclusive excursions; reversions require a later distinct observed date."""
    scores = np.asarray(scores, dtype=float)
    dates = pd.DatetimeIndex(dates)
    if len(scores) != len(dates) or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("sequence requires ordered distinct reliable dates")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > ESSDAI_MAX_VALID)).any():
        raise ValueError("sequence requires valid observed ESSDAI")
    if threshold not in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
        raise ValueError("unconfigured absolute threshold")
    delta = scores[1:] - scores[0] if len(scores) else np.array([])
    up, down = delta >= threshold, delta <= -threshold
    ur = bool((np.maximum.accumulate(up) & ~up).any()) if len(up) else False
    dr = bool((np.maximum.accumulate(down) & ~down).any()) if len(down) else False
    eu, ed = bool(up.any()), bool(down.any())
    intermittent = len(scores) >= 3 and ((eu and ed) or ur or dr)
    pattern = (INSUFFICIENT if len(scores) < 2 else "intermittent_observed" if intermittent
               else "worsening_observed" if eu else "improvement_observed" if ed else PATTERNS[0])
    slopes = np.sign(np.diff(scores))
    slopes = slopes[slopes != 0]
    return dict(threshold_abs=threshold, observed_pattern=pattern,
        trajectory_evidence_level=("insufficient" if len(scores) < 2 else "two_points_direction_only"
                                   if len(scores) == 2 else "three_or_more_points_pattern"),
        intermittence_evaluable=len(scores) >= 3, ever_up=eu, ever_down=ed,
        up_reversion=ur, down_reversion=dr,
        first_observed_up_date=dates[1:][up][0] if eu else pd.NaT,
        first_observed_down_date=dates[1:][down][0] if ed else pd.NaT,
        max_delta_from_essdai_index=float(delta.max()) if len(delta) else np.nan,
        min_delta_from_essdai_index=float(delta.min()) if len(delta) else np.nan,
        last_delta_from_essdai_index=float(delta[-1]) if len(delta) else np.nan,
        observed_change_direction_last=("not_evaluable" if not len(delta) else "worsened" if up[-1]
                                        else "improved" if down[-1] else "within_threshold"),
        last_up=bool(up[-1]) if len(up) else False, last_down=bool(down[-1]) if len(down) else False,
        n_direction_changes=int(np.count_nonzero(np.diff(slopes))),
        n_threshold_crossings=sum(int(np.count_nonzero(np.diff(np.r_[False, flag]))) for flag in (up, down)))


def build_observed_patterns_v2(df, baseline, exclude_discordant_dates=False):
    """Return protected all-patient audit and selected-date sequence, without edits.

    Only audit rows with n>=2 belong in the eligible patient pattern product.
    A discordant official index date cannot be replaced in the sensitivity;
    fallback indices are reselected from remaining dates, explicitly versioned.
    """
    x, b = validate_pattern_inputs(df, baseline)
    pop = verified_baseline_pop(b)
    groups = dict(tuple(x.groupby("patient_id", sort=False)))
    rows, sequences = [], []
    for official in b.to_dict("records"):
        pid = official["patient_id"]
        g = groups[pid]
        bd, score = official["clinical_baseline_date"], official[ESSDAI]
        available = pd.notna(score)
        source = OFFICIAL if available else FALLBACK
        valid = g.loc[g[ESSDAI].notna() & g[DATE].notna() & g[DATE].ge(bd)]
        discordant = valid.groupby(DATE)[ESSDAI].nunique().gt(1)
        conflict_dates = discordant.index[discordant]
        n_excluded = len(conflict_dates) if exclude_discordant_dates else 0
        kept = valid.loc[~valid[DATE].isin(conflict_dates)] if exclude_discordant_dates else valid
        after = kept.loc[kept[DATE].gt(bd)]
        selected = after.drop_duplicates(DATE, keep="last").copy()
        # The official score has priority over every companion on the index date.
        if available and pd.notna(bd) and not (exclude_discordant_dates and bd in conflict_dates):
            selected = pd.concat([g.loc[g.is_clinical_baseline.eq(True)], selected])
        index = selected.iloc[0] if len(selected) else None
        undated = int((g[ESSDAI].notna() & g[DATE].isna()).sum())
        reason = ("unreliable_clinical_baseline_date" if pd.isna(bd) else
                  "discordant_official_index_date" if available and exclude_discordant_dates and bd in conflict_dates else "")
        if reason:
            selected = selected.iloc[:0]
            index = None
        n = len(selected)
        if not reason and n < 2:
            reason = "no_valid_essdai_dates" if not n else "only_one_valid_essdai_date"
        eligible = n >= 2
        idx_date = index[DATE] if index is not None else pd.NaT
        idx_score = index[ESSDAI] if index is not None else np.nan
        span = (selected[DATE].iloc[-1] - idx_date).days / 365.25 if n else 0.
        raw_n = len(valid)
        # Counts audit all candidate episodes, even when sensitivity drops a date.
        candidate_dates = valid[DATE].nunique()
        meta = dict(patient_id=pid, definition_version=DEFINITION_VERSION, date_policy=POLICY,
            sensitivity_flag="exclude_entire_discordant_dates" if exclude_discordant_dates else "primary",
            official_baseline_episode_id=official["clinical_episode_id"], official_clinical_baseline_date=bd,
            official_baseline_essdai_available=available, essdai_index_source=source,
            essdai_index_episode_id=index["clinical_episode_id"] if index is not None else pd.NA,
            essdai_index_date=idx_date, essdai_index_score=idx_score,
            days_from_clinical_baseline_to_essdai_index=(idx_date - bd).days if n else np.nan,
            n_distinct_essdai_dates=n, n_essdai_followup_dates=max(0, n-1),
            n_canonical_essdai_episodes=raw_n, n_same_date_extra_episodes=max(0, raw_n-candidate_dates),
            n_discordant_essdai_dates=int(discordant.sum()), n_excluded_discordant_dates=n_excluded,
            n_undated_essdai_episodes=undated, n_missing_essdai_episodes=int(g[ESSDAI].isna().sum()),
            n_prebaseline_essdai_episodes=int((g[ESSDAI].notna() & g[DATE].lt(bd)).sum()),
            n_official_baseline_same_date_companions=int((g[DATE].eq(bd) & ~g.is_clinical_baseline.eq(True)).sum()),
            followup_years_from_essdai_index=span, assessment_frequency_per_year=(n-1)/span if span > 0 else np.nan,
            baseline_state_official=activity(score), index_activity_state=activity(idx_score),
            baseline_pop_official=pop.get(pid, "Unclassifiable") if pop is not None else "Unclassifiable",
            cohort=("official_index_cohort" if available else "fallback_index_cohort") if eligible else "not_longitudinally_evaluable",
            eligibility_status="longitudinally_evaluable" if eligible else "not_longitudinally_evaluable",
            exclusion_reason=reason)
        for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
            rows.append({**meta, **classify_sequence(selected[ESSDAI], selected[DATE], k)})
        if n:
            seq = selected[KEYS + [DATE, ESSDAI]].copy()
            seq["essdai_index_date"] = idx_date
            seq["time_since_essdai_index_years"] = (seq[DATE]-idx_date).dt.days / 365.25
            sequences.append(seq)
    audit = pd.DataFrame(rows, columns=PATIENT_COLUMNS)
    for col in ("patient_id", "official_baseline_episode_id", "essdai_index_episode_id"):
        audit[col] = audit[col].astype("string")
    for col in ("official_clinical_baseline_date", "essdai_index_date", "first_observed_up_date", "first_observed_down_date"):
        audit[col] = pd.to_datetime(audit[col])
    for col in ("official_baseline_essdai_available", "intermittence_evaluable", "ever_up", "ever_down", "up_reversion", "down_reversion", "last_up", "last_down"):
        audit[col] = audit[col].astype(bool)
    for col in ("threshold_abs", "n_distinct_essdai_dates", "n_essdai_followup_dates", "n_discordant_essdai_dates"):
        audit[col] = audit[col].astype(int)
    sequence = pd.concat(sequences, ignore_index=True) if sequences else pd.DataFrame(columns=SEQUENCE_COLUMNS)
    return audit, sequence


def pattern_summary_v2(audit, stratifier="overall"):
    """Four clinical labels only; insufficient belongs in eligibility, not phenotype.

    Intermittence has n>=3 as its primary denominator; all n>=2 is also shown.
    Each stratum reports local denominators and the full Step 11 reference N.
    """
    fields = {"index_source": ("essdai_index_source", (OFFICIAL, FALLBACK)),
              "index_state": ("index_activity_state", ("Low", "Moderate", "High", "Missing")),
              "baseline_state": ("baseline_state_official", ("Low", "Moderate", "High", "Missing")),
              "baseline_pop": ("baseline_pop_official", ("Pop1", "Pop2", "Pop3", "Unclassifiable"))}
    strata = [("Overall", audit)] if stratifier == "overall" else [(s, audit.loc[audit[fields[stratifier][0]].eq(s)]) for s in fields[stratifier][1]]
    rows = []
    for stratum, data in strata:
        for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
            g = data.loc[data.threshold_abs.eq(k)]
            eligible = g.loc[g.n_distinct_essdai_dates.ge(2)]
            n2 = int(eligible.n_distinct_essdai_dates.eq(2).sum())
            n3 = int(eligible.n_distinct_essdai_dates.ge(3).sum())
            for label in PATTERNS:
                matched = eligible.observed_pattern.eq(label)
                n = int(matched.sum())
                denom = n3 if label == "intermittent_observed" else len(eligible)
                rows.append(dict(definition_version=DEFINITION_VERSION, stratifier=stratifier, stratum=stratum,
                    threshold_abs=k, observed_pattern=label, n_patients=n,
                    n_total_step11=audit.patient_id.nunique(), n_step11_in_stratum=len(g),
                    n_legacy_official=int(eligible.essdai_index_source.eq(OFFICIAL).sum()),
                    n_fallback_eligible=int(eligible.essdai_index_source.eq(FALLBACK).sum()),
                    n_eligible_extended=len(eligible), n_pattern_eligible=len(eligible),
                    n_two_points=n2, n_three_plus=n3, n_intermittence_evaluable=n3,
                    n_insufficient=len(g)-len(eligible), n_two_points_in_pattern=int((matched & eligible.n_distinct_essdai_dates.eq(2)).sum()),
                    n_three_plus_in_pattern=int((matched & eligible.n_distinct_essdai_dates.ge(3)).sum()),
                    denominator_primary=denom, denominator_label="three_or_more_dates" if label == "intermittent_observed" else "two_or_more_dates",
                    pct_primary=100*n/denom if denom else np.nan,
                    pct_of_extended=100*n/len(eligible) if len(eligible) else np.nan,
                    support_status="not_estimable" if not len(eligible) else "sparse" if n < 10 else "descriptive_only"))
    return pd.DataFrame(rows)


def eligibility_summary_v2(audit):
    g = audit.loc[audit.threshold_abs.eq(ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS[0])]
    rows = []
    for source, data in [("Overall", g), (OFFICIAL, g.loc[g.essdai_index_source.eq(OFFICIAL)]),
                          (FALLBACK, g.loc[g.essdai_index_source.eq(FALLBACK)])]:
        eligible = data.n_distinct_essdai_dates.ge(2)
        metrics = {"n_total_step11": len(data), "n_baseline_score_observed": int(data.official_baseline_essdai_available.sum()),
            "n_baseline_score_missing": int((~data.official_baseline_essdai_available).sum()),
            "n_eligible": int(eligible.sum()), "n_insufficient": int((~eligible).sum()),
            "n_two_points": int(data.n_distinct_essdai_dates.eq(2).sum())}
        metrics.update({f"n_dates_ge{n}": int(data.n_distinct_essdai_dates.ge(n).sum()) for n in (1, 2, 3, 4)})
        for metric, value in metrics.items():
            rows.append(dict(index_source=source, metric=metric, n_patients=value, denominator=len(data)))
        for reason, subset in data.loc[~eligible].groupby("exclusion_reason"):
            rows.append(dict(index_source=source, metric="exclusion:"+reason, n_patients=len(subset), denominator=len(data)))
        delays = data.loc[eligible & data.essdai_index_source.eq(FALLBACK), "days_from_clinical_baseline_to_essdai_index"]
        for label, value in zip(("min", "q25", "median", "q75", "max"), delays.quantile([0, .25, .5, .75, 1]).to_numpy()):
            rows.append(dict(index_source=source, metric="fallback_entry_delay_days_"+label, value=value, n_patients=len(delays), denominator=len(data)))
    return pd.DataFrame(rows)


def followup_sensitivity_v2(audit, excluded):
    n, span, rate = audit.n_distinct_essdai_dates, audit.followup_years_from_essdai_index, audit.assessment_frequency_per_year
    subsets = [("dates_eq2", audit.loc[n.eq(2)]), ("dates_ge3", audit.loc[n.ge(3)]),
               ("dates_ge4", audit.loc[n.ge(4)]), ("exclude_entire_discordant_dates", excluded)]
    for name, mask in (("duration_<2y", n.ge(2) & span.lt(2)), ("duration_2-<5y", n.ge(2) & span.ge(2) & span.lt(5)),
                       ("duration_ge5y", n.ge(2) & span.ge(5)), ("frequency_<1_per_year", n.ge(2) & rate.lt(1)),
                       ("frequency_ge1_per_year", n.ge(2) & rate.ge(1))):
        subsets.append((name, audit.loc[mask]))
    tables = []
    for name, subset in subsets:
        for stratifier in ("overall", "index_source", "baseline_state"):
            table = pattern_summary_v2(subset, stratifier)
            table["n_total_step11"] = audit.patient_id.nunique()
            table.insert(0, "sensitivity", name)
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def legacy_bridge_v2(df, endpoints, audit):
    """Aggregate v1/v2 cross-tab plus endpoint discrepancies on official cohort."""
    df = df.copy()
    for col in (DATE, "clinical_baseline_date"):
        df[col] = pd.to_datetime(df[col], format="mixed")
    for col in KEYS:
        df[col] = df[col].astype("string")
    df[ESSDAI] = pd.to_numeric(df[ESSDAI])
    old = build_observed_patterns(df, endpoints)
    official = audit.loc[audit.essdai_index_source.eq(OFFICIAL) & audit.n_distinct_essdai_dates.ge(2)]
    merged = official.merge(old[["patient_id", "threshold_abs", "observed_pattern"]],
                            on=["patient_id", "threshold_abs"], suffixes=("_v2", "_v1"), how="outer", indicator=True)
    rows = []
    for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
        g = merged.loc[merged.threshold_abs.eq(k)]
        for (v1, v2), part in g.groupby(["observed_pattern_v1", "observed_pattern_v2"], dropna=False):
            rows.append(dict(threshold_abs=k, metric="definition_crosstab", pattern_v1=v1, pattern_v2=v2,
                             n_patients=len(part), denominator=len(endpoints)))
        rows.append(dict(threshold_abs=k, metric="official_cohort_symmetric_difference",
                         n_patients=int(g._merge.ne("both").sum()), denominator=len(endpoints)))
        p = official.loc[official.threshold_abs.eq(k)].set_index("patient_id")
        e = endpoints.set_index("patient_id").reindex(p.index)
        for flag, legacy in (("ever_up", f"ever_worsened_ge{k}"), ("ever_down", f"ever_improved_ge{k}"),
                             ("last_up", f"last_worsened_ge{k}"), ("last_down", f"last_improved_ge{k}")):
            difference = p[flag].ne(e[legacy])
            ties = p.n_discordant_essdai_dates.gt(0)
            for name, mask in (("discordant_date_difference", difference & ties), ("unexplained_difference", difference & ~ties)):
                rows.append(dict(threshold_abs=k, metric=flag+":"+name, n_patients=int(mask.sum()), denominator=len(p)))
    return pd.DataFrame(rows)


def observed_pattern_qc_v2(df, baseline, endpoints, audit, excluded, sequence):
    rows = []
    def check(name, failures):
        rows.append(dict(check=name, status="FAIL" if failures else "PASS", n_failures=int(failures)))
    expected, expected_seq = build_observed_patterns_v2(df, baseline)
    for name, data, sensitivity in (("primary", audit, False), ("sensitivity", excluded, True)):
        n = data.n_distinct_essdai_dates
        check(name+":unique_patient_threshold", data.duplicated(["patient_id", "threshold_abs"]).sum())
        check(name+":complete_step11_universe", abs(len(data)-len(baseline)*len(ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS)))
        check(name+":labels_and_eligibility", ((n.ge(2) & ~data.observed_pattern.isin(PATTERNS)) | (n.lt(2) & data.observed_pattern.ne(INSUFFICIENT))).sum())
        check(name+":two_point_evidence", (n.eq(2) & (data.trajectory_evidence_level.ne("two_points_direction_only") | data.intermittence_evaluable | data.observed_pattern.eq("intermittent_observed"))).sum())
        check(name+":intermittence_evaluable", data.intermittence_evaluable.ne(n.ge(3)).sum())
        excursion = (data.ever_up & data.ever_down) | data.up_reversion | data.down_reversion
        check(name+":intermittence_definition", data.observed_pattern.eq("intermittent_observed").ne(n.ge(3) & excursion).sum())
        rebuild = build_observed_patterns_v2(df, baseline, sensitivity)[0] if sensitivity else expected
        check(name+":deterministic_rebuild_baseline_dates_pop_index", not data.equals(rebuild))
        official = data.essdai_index_source.eq(OFFICIAL) & n.ge(1)
        fallback = data.essdai_index_source.eq(FALLBACK) & n.ge(1)
        check(name+":official_index_identity", (official & (data.essdai_index_episode_id.ne(data.official_baseline_episode_id) | data.essdai_index_date.ne(data.official_clinical_baseline_date))).sum())
        check(name+":fallback_strictly_after_official", (fallback & ~data.essdai_index_date.gt(data.official_clinical_baseline_date)).sum())
        check(name+":no_inferred_fallback_pop", (fallback & data.baseline_pop_official.ne("Unclassifiable")).sum())
        check(name+":positive_followup", (n.ge(2) & data.followup_years_from_essdai_index.le(0)).sum())
        small = data.loc[data.threshold_abs.eq(3)].set_index("patient_id")
        large = data.loc[data.threshold_abs.eq(5)].set_index("patient_id").reindex(small.index)
        for direction in ("up", "down"):
            check(name+":nested_ever_"+direction, (large["ever_"+direction] & ~small["ever_"+direction]).sum())
        table = pattern_summary_v2(data)
        for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
            s = table.loc[table.threshold_abs.eq(k)]
            check(name+f":partition_{k}", abs(int(s.n_patients.sum()) - int((data.threshold_abs.eq(k) & n.ge(2)).sum())))
    check("selected_sequence_rebuild", not sequence.equals(expected_seq))
    check("selected_dates_unique", sequence.duplicated(["patient_id", DATE]).sum())
    if len(sequence):
        check("no_selected_point_before_index", sequence.time_since_essdai_index_years.lt(0).sum())
    bridge = legacy_bridge_v2(df, endpoints, audit)
    check("legacy_official_cohort_reconciled", bridge.loc[bridge.metric.eq("official_cohort_symmetric_difference"), "n_patients"].sum())
    check("legacy_ever_last_without_discordance", bridge.loc[bridge.metric.str.endswith(":unexplained_difference"), "n_patients"].sum())
    rows.append(dict(check="discordant_dates_audited", status="INFO", n_failures=0,
                     n_dates=int(audit.loc[audit.threshold_abs.eq(3), "n_discordant_essdai_dates"].sum())))
    return pd.DataFrame(rows)


def plot_patterns_v2(audit, path, input_origin="canonical"):
    summary = pattern_summary_v2(audit)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    labels = ["Stable / within threshold", "Worsening observed", "Improvement observed", "Intermittent observed", "Insufficient (eligibility)"]
    for ax, k in zip(axes, ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS):
        g = summary.loc[summary.threshold_abs.eq(k)]
        two, three = g.n_two_points_in_pattern.to_list(), g.n_three_plus_in_pattern.to_list()
        insufficient = int(g.n_insufficient.iloc[0])
        ax.barh(range(4), two, label="2 dates: provisional direction", color="#89b4cb")
        ax.barh(range(4), three, left=two, label="3+ dates: pattern evaluable", color="#356583")
        ax.barh(4, insufficient, color="#bdbdbd")
        ax.axhline(3.5, color="0.6", lw=.7)
        for j, row in enumerate(g.itertuples()):
            pct = f"{row.pct_primary:.1f}%" if pd.notna(row.pct_primary) else "NA"
            ax.text(row.n_patients+.2, j, f"{row.n_patients}/{row.denominator_primary} ({pct})", va="center", fontsize=9)
        ax.text(insufficient+.2, 4, f"{insufficient}/{int(g.n_step11_in_stratum.iloc[0])}", va="center", fontsize=9)
        ax.set(yticks=range(5), yticklabels=labels, xlabel="Patients", title=f"±{k}: Step 11 N={int(g.n_total_step11.iloc[0])}\nextended N={int(g.n_eligible_extended.iloc[0])}; 3+ dates N={int(g.n_three_plus.iloc[0])}")
        ax.set_xlim(0, max([1, insufficient, *g.n_patients.to_list()])*1.65 + .5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="lower right", fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Observed ESSDAI patterns v2 — " +
                 ("SYNTHETIC SOFTWARE VALIDATION" if input_origin == "synthetic_validation"
                  else "controlled research aggregate"))
    fig.text(.5, .02, "Direction denominator: ≥2 dates. Intermittence primary denominator: ≥3 dates.\nSporadic observations do not establish continuity, persistence or remission. Institutional disclosure review required.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .09, 1, .94])
    try:
        fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    finally:
        plt.close(fig)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_directories(root):
    root = Path(root)
    return {"tables": root/"outputs/tables/blockA"/STEM,
            "figures": root/"outputs/figures/blockA"/STEM,
            "qc": root/"outputs/qc/blockA"/STEM,
            "analytic": root/"data/analytic/blockA"/STEM}


def write_observed_pattern_outputs_v2(df, baseline, endpoints, output_root, *,
                                     input_origin="canonical", input_paths=None,
                                     overwrite=True, dry_run=False):
    """Validate, stage a complete bundle, backup only v2, then install/rollback.

    No v1/Step 10/11/14 product is a write target. Dry run performs full QC with
    zero filesystem writes. Manifest is installed last as completion marker.
    """
    if input_origin not in ("canonical", "synthetic_validation"):
        raise ValueError("unknown input_origin")
    root = Path(output_root)
    dirs = output_directories(root)
    existing = [p for d in dirs.values() if d.exists() for p in d.glob("13_essdai_*_v2.*") if p.is_file()]
    if existing and not overwrite and not dry_run:
        raise FileExistsError("Step 13 v2 outputs exist; --no-overwrite protects them")
    audit, sequence = build_observed_patterns_v2(df, baseline)
    excluded, _ = build_observed_patterns_v2(df, baseline, True)
    checks = observed_pattern_qc_v2(df, baseline, endpoints, audit, excluded, sequence)
    if checks.status.eq("FAIL").any():
        failures = checks.loc[checks.status.eq("FAIL"), "check"].to_list()
        raise ValueError(f"Observed patterns v2 QC failure: {failures}")
    if dry_run:
        return checks
    # Inputs are hashed before producing products. No row values enter manifests.
    inputs = [{"role": role, "path": str(path), "sha256": sha256(path)} for role, path in (input_paths or {}).items()]
    code_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_root, text=True).strip()
    except subprocess.CalledProcessError:
        commit = "unavailable"
    historical = [p for d in dirs.values() if d.exists() for p in d.glob("13_*") if p.is_file() and "_v2." not in p.name]
    history = [{"path": str(p.relative_to(root)), "sha256": sha256(p)} for p in sorted(historical)]
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step13-v2-", dir=root) as temp:
        stage = Path(temp)
        staging = output_directories(stage)
        for d in staging.values():
            d.mkdir(parents=True)
        def csv(area, name, data):
            data.to_csv(staging[area]/f"13_essdai_{name}_v2.csv", index=False)
        eligible = audit.loc[audit.n_distinct_essdai_dates.ge(2)].copy()
        eligible.to_parquet(staging["analytic"]/"13_essdai_patient_observed_patterns_v2.parquet", index=False)
        audit.to_parquet(staging["analytic"]/"13_essdai_pattern_patient_eligibility_v2.parquet", index=False)
        sequence.to_parquet(staging["analytic"]/"13_essdai_pattern_selected_dates_v2.parquet", index=False)
        csv("tables", "pattern_eligibility", eligibility_summary_v2(audit))
        csv("tables", "observed_pattern_summary", pattern_summary_v2(audit))
        for name, stratifier in (("index_source", "index_source"), ("index_state", "index_state"),
                                 ("baseline_state", "baseline_state"), ("pop", "baseline_pop")):
            csv("tables", "pattern_by_"+name, pattern_summary_v2(audit, stratifier))
        csv("tables", "pattern_followup_sensitivity", followup_sensitivity_v2(audit, excluded))
        csv("tables", "pattern_legacy_bridge", legacy_bridge_v2(df, endpoints, audit))
        csv("qc", "observed_pattern_qc", checks)
        plot_patterns_v2(audit, staging["figures"]/"13_essdai_observed_pattern_aggregate_v2.pdf", input_origin)
        from src.block_A.observed_patterns_v2_figures import make_companion_figures
        make_companion_figures(audit, sequence, staging["figures"])
        paths = sorted(p for d in staging.values() for p in d.iterdir())
        artifacts = [{"path": str(p.relative_to(stage)), "sha256": sha256(p)} for p in paths]
        manifest = dict(definition_version=DEFINITION_VERSION, input_origin=input_origin, code_commit=commit,
            created_at=datetime.now(timezone.utc).isoformat(), clinical_review="PENDING_CLINICAL_REVIEW",
            closure_status="CONDITIONAL — real-cohort validation pending" if input_origin != "canonical" else "CONDITIONAL — clinical and institutional review pending",
            unit_of_analysis="patient × threshold_abs; selected sequence uses distinct real calendar dates",
            index_policy="official baseline when observed; otherwise first strictly later observed ESSDAI",
            date_policy=POLICY, baseline_tie_policy="official score priority; never replace clinical baseline",
            sensitivity_policy="exclude whole discordant date; no replacement official index; reselect fallback from retained dates",
            privacy="controlled research artifacts; no external disclosure authorization; sparse aggregates require institutional review",
            thresholds=list(ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS), inputs=inputs,
            input_frame_sha256=hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest(),
            baseline_frame_sha256=hashlib.sha256(pd.util.hash_pandas_object(baseline, index=True).values.tobytes()).hexdigest(),
            source_hashes={str(p.relative_to(code_root)): sha256(p) for p in [Path(__file__), Path(__file__).with_name("observed_patterns.py"), Path(__file__).with_name("13_disease_activity_progression.py"), code_root/"config.py"]},
            historical_products_before=history, artifacts=artifacts,
            pop_status="verified_published_only" if "pop__status" in baseline else "unavailable_Unclassifiable",
            qc_status="PASS", closing_criteria=["canonical inputs and input hashes verified", "blocking QC PASS", "clinical review", "institutional disclosure review"])
        manifest_path = staging["qc"]/"13_essdai_observed_pattern_manifest_v2.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(manifest_path)
        backup = None
        if existing:
            backup = root/"outputs/backups"/STEM/("v2_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
            for old in existing:
                dest = backup/old.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, dest)
        installed = []
        try:
            for path in paths:
                target = root/path.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, target)
                installed.append(target)
        except BaseException:
            for target in reversed(installed):
                old = backup/target.relative_to(root) if backup else None
                if old is not None and old.exists():
                    shutil.copy2(old, target)
                else:
                    target.unlink(missing_ok=True)
            raise
    return checks


def run_observed_patterns_v2(args, step13):
    """Separate Step 13 CLI mode, before any legacy output writes or fitting."""
    missing = [p for p in (args.integrated, args.baseline) if not p.is_file()]
    if missing:
        raise SystemExit("NOT RUN — canonical inputs unavailable: "+", ".join(map(str, missing)))
    raw, baseline = step13.read_table(args.integrated), step13.read_table(args.baseline)
    # Strict v2 validation first; unchanged legacy validation/domain contract next.
    clean, _ = validate_pattern_inputs(raw, baseline)
    # Missing real dates are explicitly excluded by v2, never nominally repaired.
    # The legacy validator requires dated baselines; only call it when all are dated.
    if clean[DATE].notna().all() and clean.clinical_baseline_date.notna().all():
        step13.validate_inputs(raw, baseline)
    resolved = step13.resolve_domains(clean)
    if len(resolved) != len(step13.DOMAINS):
        raise ValueError("ESSDAI domain contract: v2 requires all 12 Step 10 domains")
    for col in resolved.values():
        values = pd.to_numeric(clean[col], errors="coerce")
        if (clean[col].notna() & values.isna()).any() or (~values.dropna().isin([0, 1, 2, 3])).any():
            raise ValueError("invalid upstream ESSDAI ordinal domain")
    endpoints = step13.build_patient_absolute_change_endpoints(clean)
    checks = write_observed_pattern_outputs_v2(raw, baseline, endpoints, args.output_root,
        input_origin=args.input_origin, input_paths={"step10": args.integrated, "step11": args.baseline},
        overwrite=args.overwrite, dry_run=args.dry_run)
    print(checks.to_string(index=False))
    print("DRY RUN — no files written" if args.dry_run else "Step 13 v2 complete; PENDING_CLINICAL_REVIEW; controlled artifacts only")
    return checks

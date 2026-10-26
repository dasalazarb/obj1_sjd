"""Observed secondary measurements inside immutable, producer-defined groups.

Patient-level intermediate frames stay in memory. Only aggregate summaries are
returned to the publication layer. This module never calls a classifier.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.block_A.observed_patterns_v2_figures import LABELS

DEFINITION = "observed-essdai-pattern-v2"
KEYS = ["patient_id", "clinical_episode_id"]
STEM = "15_secondary_trajectories_by_essdai_pattern"
WARNINGS = (
    "Retrospective selection using complete ESSDAI history; descriptive, not predictive or causal. "
    "Unequal follow-up and observation opportunities; potentially informative missingness; clinical indication bias. "
    "Baseline regression to the mean and floor/ceiling effects; assay/scoring changes and concomitant treatment. "
    "Unobserved fluctuations between visits; two observations do not establish persistence, remission or seroconversion."
)


class ContractError(ValueError):
    """Blocking input, provenance or temporal identity failure."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    # Preserve leading zeros and literal canonical identities in CSV fixtures.
    return pd.read_csv(path, dtype={k: "string" for k in KEYS+["clinical_baseline_episode_id", "official_baseline_episode_id"]})


def verified_manifest(path, patterns_path, integrated_path, baseline_path, origin):
    """Verify *every* v2 artifact, input bytes and exact consumed file identity."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("definition_version") != DEFINITION or manifest.get("qc_status") != "PASS":
        raise ContractError("Step 13 incompatible definition or failed QC")
    if manifest.get("input_origin") != origin:
        raise ContractError("input_origin disagrees with Step 13 provenance; synthetic cannot be canonical")
    root = path.parents[4]
    artifacts = {}
    for entry in manifest.get("artifacts", []):
        relative = Path(entry["path"])
        target = (root/relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root) or target in artifacts.values():
            raise ContractError("invalid or duplicate Step 13 artifact path")
        if not target.is_file() or sha256(target) != entry["sha256"]:
            raise ContractError("Step 13 artifact SHA256 mismatch or missing artifact: "+relative.name)
        if target.name in artifacts:
            raise ContractError("ambiguous Step 13 artifact identity")
        artifacts[target.name] = target
    expected = {
        "patterns": "13_essdai_patient_observed_patterns_v2.parquet",
        "eligibility": "13_essdai_pattern_patient_eligibility_v2.parquet",
        "selected_dates": "13_essdai_pattern_selected_dates_v2.parquet",
    }
    if not set(expected.values()).issubset(artifacts):
        raise ContractError("missing protected Step 13 v2 artifacts")
    if Path(patterns_path).resolve() != artifacts[expected["patterns"]]:
        raise ContractError("patterns path is not the manifest-authenticated artifact")
    for role, source in (("step10", integrated_path), ("step11", baseline_path)):
        inputs = [e for e in manifest.get("inputs", []) if e.get("role") == role]
        if len(inputs) != 1 or inputs[0].get("sha256") != sha256(source):
            raise ContractError("Step 13 input SHA256 identity mismatch: "+role)
    return manifest, {role: artifacts[name] for role, name in expected.items()}


def require(frame, columns, name):
    missing = set(columns)-set(frame)
    if missing:
        raise ContractError(name+": missing columns "+", ".join(sorted(missing)))


def identities(frame, columns, name):
    frame = frame.copy()
    require(frame, columns, name)
    for col in columns:
        # Numeric identifiers can already have lost significant leading zeros.
        if frame[col].dropna().map(lambda v: not isinstance(v, str)).any():
            raise ContractError(name+": unstable non-string keys")
        frame[col] = frame[col].astype("string")
        if frame[col].isna().any() or frame[col].str.strip().eq("").any():
            raise ContractError(name+": null/empty keys")
    return frame


def dates(values, name, cutoff):
    if values.dropna().map(lambda v: isinstance(v, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T]00:00:00(?:\.0+)?)?", v)).any():
        raise ContractError(name+": partial or unverified calendar dates")
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if (values.notna() & parsed.isna()).any() or not pd.api.types.is_datetime64_dtype(parsed):
        raise ContractError(name+": malformed calendar dates")
    if parsed.dropna().ne(parsed.dropna().dt.normalize()).any():
        raise ContractError(name+": timestamps must be calendar dates")
    if parsed.gt(pd.Timestamp(cutoff)).any():
        raise ContractError(name+": dates after data cutoff")
    return parsed


def validate_inputs(integrated, baseline, patterns, eligibility, sequence, cutoff):
    """Validate structure and protected identities without recomputing ESSDAI."""
    x = identities(integrated, KEYS+["clinical_baseline_episode_id"], "Step 10")
    b = identities(baseline, KEYS+["clinical_baseline_episode_id"], "Step 11")
    structural = ["clinical_anchor_date", "clinical_baseline_date", "is_clinical_baseline"]
    for name, frame in (("Step 10", x), ("Step 11", b)):
        require(frame, structural, name)
        if frame.duplicated(KEYS).any():
            raise ContractError(name+": duplicate episode keys")
        if not frame.is_clinical_baseline.isin([True, False]).all():
            raise ContractError(name+": invalid baseline indicator")
        for col in structural[:2]:
            frame[col] = dates(frame[col], name+":"+col, cutoff)
    if b.patient_id.duplicated().any() or not b.is_clinical_baseline.eq(True).all():
        raise ContractError("Step 11: expected unique official baseline")
    official = x.loc[x.is_clinical_baseline.eq(True)].copy()
    if official.patient_id.duplicated().any() or set(official.patient_id) != set(x.patient_id) or set(b.patient_id) != set(x.patient_id):
        raise ContractError("Step 10/11 baseline cohort identity mismatch")
    for frame in (official, b):
        if not frame.clinical_episode_id.eq(frame.clinical_baseline_episode_id).all():
            raise ContractError("official baseline episode identity mismatch")
        equal = frame.clinical_anchor_date.eq(frame.clinical_baseline_date) | (frame.clinical_anchor_date.isna() & frame.clinical_baseline_date.isna())
        if not equal.all():
            raise ContractError("official baseline date mismatch")
    lookup = b.set_index("patient_id")
    for frame in (x, official):
        cols = ["clinical_baseline_episode_id", "clinical_baseline_date"]
        if frame is official:
            cols += ["clinical_episode_id", "clinical_anchor_date"]
        for col in cols:
            expected = frame.patient_id.map(lookup[col])
            if not (frame[col].eq(expected) | (frame[col].isna() & expected.isna())).all():
                raise ContractError("Step 10/11 baseline metadata conflict: "+col)
    # Compare common published baseline measurements too, with null-safe equality.
    a = official.set_index("patient_id").sort_index()
    bb = lookup.reindex(a.index)
    for col in set(a)&set(bb):
        if not (a[col].eq(bb[col]) | (a[col].isna() & bb[col].isna())).fillna(False).all():
            raise ContractError("Step 10/11 official episode conflict: "+col)
    protected = ["patient_id", "threshold_abs", "observed_pattern", "eligibility_status", "n_distinct_essdai_dates",
                 "trajectory_evidence_level", "essdai_index_date", "essdai_index_source", "essdai_index_episode_id",
                 "official_baseline_episode_id", "official_clinical_baseline_date", "definition_version", "date_policy", "sensitivity_flag"]
    frames = []
    for name, original in (("patterns", patterns), ("eligibility", eligibility)):
        require(original, protected, name)
        p = identities(original, ["patient_id", "official_baseline_episode_id"], name)
        if p.duplicated(["patient_id", "threshold_abs"]).any() or not p.threshold_abs.isin([3, 5]).all():
            raise ContractError("duplicate patient groups or unknown threshold")
        if not p.definition_version.eq(DEFINITION).all() or not p.patient_id.isin(b.patient_id).all():
            raise ContractError("Step 13 definition or patient identity conflict")
        if not p.sensitivity_flag.eq("primary").all() or not p.date_policy.eq("last_canonical_episode_per_date").all():
            raise ContractError("not the primary v2 protected date policy")
        if not p.eligibility_status.isin(["longitudinally_evaluable", "not_longitudinally_evaluable"]).all():
            raise ContractError("unknown eligibility status")
        for col in ("essdai_index_date", "official_clinical_baseline_date"):
            p[col] = dates(p[col], name+":"+col, cutoff)
        for col, source in (("official_baseline_episode_id", "clinical_baseline_episode_id"),
                            ("official_clinical_baseline_date", "clinical_baseline_date")):
            expected = p.patient_id.map(lookup[source])
            if not (p[col].eq(expected) | (p[col].isna() & expected.isna())).all():
                raise ContractError("Step 13 official baseline conflict")
        eligible = p.eligibility_status.eq("longitudinally_evaluable")
        if not p.loc[eligible, "observed_pattern"].isin(LABELS).all() or p.loc[eligible, "n_distinct_essdai_dates"].lt(2).any():
            raise ContractError("invalid eligible ESSDAI pattern or support")
        if p.loc[eligible, ["essdai_index_date", "official_clinical_baseline_date", "essdai_index_episode_id"]].isna().any().any():
            raise ContractError("eligible group lacks dated baseline/index identity")
        if p.loc[eligible, "essdai_index_date"].lt(p.loc[eligible, "official_clinical_baseline_date"]).any():
            raise ContractError("ESSDAI index before baseline")
        two = eligible & p.n_distinct_essdai_dates.eq(2)
        if p.loc[two, "observed_pattern"].eq("intermittent_observed").any() or not p.loc[two, "trajectory_evidence_level"].eq("two_points_direction_only").all():
            raise ContractError("two-date evidence must be provisional, never intermittent")
        if not p.loc[eligible & ~two, "trajectory_evidence_level"].eq("three_or_more_points_pattern").all():
            raise ContractError("unknown trajectory evidence level")
        for source in p.loc[eligible, "essdai_index_source"].unique():
            if source not in ("official_baseline", "first_observed_after_official_baseline"):
                raise ContractError("unknown ESSDAI index source")
        off = eligible & p.essdai_index_source.eq("official_baseline")
        late = eligible & ~off
        if (p.loc[off, "essdai_index_date"].ne(p.loc[off, "official_clinical_baseline_date"]).any()
                or p.loc[off, "essdai_index_episode_id"].ne(p.loc[off, "official_baseline_episode_id"]).any()
                or not p.loc[late, "essdai_index_date"].gt(p.loc[late, "official_clinical_baseline_date"]).all()):
            raise ContractError("ESSDAI index source/date conflict")
        frames.append(p)
    p, e = frames
    if set(e.threshold_abs.unique()) != {3, 5}:
        raise ContractError("v2 eligibility must cover both independent thresholds")
    if not p.eligibility_status.eq("longitudinally_evaluable").all():
        raise ContractError("protected patterns contains ineligible rows")
    expected = e.loc[e.eligibility_status.eq("longitudinally_evaluable"), protected].sort_values(["patient_id", "threshold_abs"]).reset_index(drop=True)
    actual = p[protected].sort_values(["patient_id", "threshold_abs"]).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    except AssertionError as error:
        raise ContractError("patterns disagree with original v2 eligibility assignment") from error
    for k, group in e.groupby("threshold_abs"):
        if set(group.patient_id) != set(b.patient_id):
            raise ContractError("incomplete Step 13 eligibility cohort")
    s = identities(sequence, KEYS, "selected dates")
    require(s, ["clinical_anchor_date", "essdai_index_date"], "selected dates")
    for col in ("clinical_anchor_date", "essdai_index_date"):
        s[col] = dates(s[col], "selected dates:"+col, cutoff)
    if s.duplicated(["patient_id", "clinical_anchor_date"]).any() or s.clinical_anchor_date.isna().any():
        raise ContractError("selected dates not distinct reliable dates")
    joined = s.merge(x[KEYS+["clinical_anchor_date"]], on=KEYS, how="left", validate="one_to_one", suffixes=("", "_step10"))
    if not joined.clinical_anchor_date.eq(joined.clinical_anchor_date_step10).all():
        raise ContractError("selected dates conflict with Step 10 episode identity")
    counts = s.groupby("patient_id").size()
    if p.n_distinct_essdai_dates.ne(p.patient_id.map(counts)).any():
        raise ContractError("selected date support differs from v2 groups")
    essdai = "essdai__total"
    if essdai in s and essdai in x:
        scores = s.merge(x[KEYS+[essdai]], on=KEYS, validate="one_to_one", suffixes=("_v2", "_step10"))
        if not scores[essdai+"_v2"].eq(scores[essdai+"_step10"]).all():
            raise ContractError("selected ESSDAI scores conflict with Step 10")
    for row in p.itertuples():
        selected = s.loc[s.patient_id.eq(row.patient_id)]
        if not selected.essdai_index_date.eq(row.essdai_index_date).all() or not ((selected.clinical_episode_id == row.essdai_index_episode_id) & selected.clinical_anchor_date.eq(row.essdai_index_date)).any():
            raise ContractError("selected ESSDAI index identity conflict")
    return x, b, p


def attach_context(x, context):
    if context is None:
        return x.copy()
    c = identities(context, KEYS, "Step 10 context")
    if c.duplicated(KEYS).any():
        raise ContractError("Step 10 context duplicate keys")
    keys = x[KEYS].merge(c[KEYS], on=KEYS, how="outer", indicator=True, validate="one_to_one")
    if not keys._merge.eq("both").all():
        raise ContractError("Step 10 context episode coverage mismatch")
    for col in (set(c)&set(x))-set(KEYS):
        pair = x[KEYS+[col]].merge(c[KEYS+[col]], on=KEYS, validate="one_to_one", suffixes=("_a", "_b"))
        left, right = pair[col+"_a"], pair[col+"_b"]
        if "date" in col:
            left, right = pd.to_datetime(left), pd.to_datetime(right)
        if not (left.eq(right) | (left.isna() & right.isna())).fillna(False).all():
            raise ContractError("Step 10 context conflicts with integrated: "+col)
    return x.merge(c[KEYS+[col for col in c if col not in x]], on=KEYS, how="left", validate="one_to_one")


def fixed_groups(patterns, threshold):
    if threshold not in (3, 5):
        raise ContractError("threshold must be 3 or 5")
    group = patterns.loc[patterns.threshold_abs.eq(threshold)].copy()
    if group.empty:
        raise ContractError("requested threshold absent from protected v2 patterns")
    if group.patient_id.duplicated().any() or not group.observed_pattern.isin(LABELS).all():
        raise ContractError("invalid immutable group map")
    return group


def observations(x, groups, spec, cutoff, time_axis="clinical_baseline", window_years=1.):
    """Resolve observed dates, same-day conflicts, references and both clocks."""
    empty = pd.DataFrame()
    qc = []
    def audit(check, n):
        qc.append(dict(variable=spec.variable_id, check=check, status="WARN" if n else "PASS", n=int(n)))
    if spec.canonical_column not in x:
        return empty, "not_in_integrated", qc, spec.unit
    if spec.measurement_date_column not in x:
        return empty, "invalid_date: no verified measurement date", qc, spec.unit
    # Keep upstream baseline in x; prefix only immutable group metadata.
    cols = ["patient_id", "observed_pattern", "trajectory_evidence_level", "n_distinct_essdai_dates", "essdai_index_date", "essdai_index_source"]
    frame = x.merge(groups[cols], on="patient_id", how="inner", validate="many_to_one").copy()
    frame["value"] = frame[spec.canonical_column]
    frame = frame.loc[frame.value.notna()].copy()
    unit = spec.unit
    if spec.domain == "labs":
        unit_col = spec.canonical_column.removesuffix("__value")+"__unit"
        if unit_col not in frame or frame[unit_col].isna().any():
            return empty, "unknown_units", qc, unit
        units = frame[unit_col].dropna().astype(str).str.strip().unique()
        if len(units) != 1 or not units[0]:
            return empty, "incompatible_assay_or_units", qc, unit
        unit = units[0]
    if not unit:
        return empty, "unknown_units", qc, unit
    stem = spec.canonical_column.removesuffix("__value")
    instrument = spec.canonical_column.split("__")[-1].split("_")[0]
    version_cols = [c for c in frame if c.startswith((stem+"__", "pro__"+instrument+"_")) and c.endswith(("assay", "platform", "scoring_version"))]
    if any(frame[c].dropna().nunique() > 1 for c in version_cols):
        return empty, "incompatible_assay_or_scoring_version", qc, unit
    for flag in (stem+"__conflict", spec.canonical_column+"_conflict"):
        if flag in frame:
            bad = frame[flag].eq(True).fillna(False)
            audit("producer_conflict_excluded", bad.sum())
            frame = frame.loc[~bad].copy()
    if spec.type in ("numeric", "ordinal"):
        values = pd.to_numeric(frame.value, errors="coerce")
        invalid = values.isna() | ~np.isfinite(values.astype(float))
        if spec.scale_min is not None:
            invalid |= values.lt(spec.scale_min)
        if spec.scale_max is not None:
            invalid |= values.gt(spec.scale_max)
        if spec.type == "ordinal":
            invalid |= values.mod(1).ne(0)
        audit("invalid_scale_excluded", invalid.sum())
        frame["value"] = values
        frame = frame.loc[~invalid].copy()
    elif spec.type == "binary":
        mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
        values = frame.value.astype(str).str.lower().map(mapping)
        if values.isna().any():
            raise ContractError("unrecognized canonical binary encoding: "+spec.variable_id)
        frame["value"] = values
    # Only known canonical overlap levels may be published as categories.
    elif spec.canonical_column == "ovl__status":
        missing = frame.value.isin(["Unclassifiable", "unclassifiable", "not_evaluated"])
        frame = frame.loc[~missing].copy()
        allowed = {"overlap", "glandular_only", "extraglandular_only", "neither", "none", "no_active_manifestations"}
        if not frame.value.isin(allowed).all():
            raise ContractError("unknown canonical overlap level")
    frame["variable_measurement_date"] = dates(frame[spec.measurement_date_column], spec.variable_id, cutoff)
    pre = frame.variable_measurement_date.lt(frame.clinical_baseline_date)
    absent = frame.variable_measurement_date.isna() | frame.clinical_baseline_date.isna()
    audit("prebaseline_measurements_excluded", pre.sum())
    audit("undated_measurements_excluded", absent.sum())
    frame = frame.loc[~pre & ~absent].copy()
    chosen = []
    for _, same in frame.groupby(["patient_id", "variable_measurement_date"], sort=True):
        official = same.loc[same.clinical_episode_id.eq(same.clinical_baseline_episode_id) & same.variable_measurement_date.eq(same.clinical_baseline_date)]
        discordant = same.value.nunique() > 1
        if len(same) > 1:
            audit("same_day_discordant" if discordant else "same_day_concordant", 1)
        if len(official):
            chosen.append(official.iloc[0])
        elif not discordant:
            chosen.append(same.sort_values("clinical_episode_id").iloc[-1])
        # No cross-producer rule justifies choosing among discordant values.
    if not chosen:
        return empty, "no_valid_observed_measurements", qc, unit
    frame = pd.DataFrame(chosen).sort_values(["patient_id", "variable_measurement_date", "clinical_episode_id"]).reset_index(drop=True)
    frame["time_from_clinical_baseline_years"] = (frame.variable_measurement_date-frame.clinical_baseline_date).dt.days/365.25
    frame["time_from_essdai_index_years"] = (frame.variable_measurement_date-frame.essdai_index_date).dt.days/365.25
    frame["time"] = frame["time_from_"+time_axis+"_years"]
    frame["window"] = np.floor(frame.time/window_years).astype(int)
    frame["window_start"] = frame.window*window_years
    frame["window_end"] = frame.window_start+window_years
    frame["source_protocol"] = frame.get("source_protocol", pd.Series("not_available", index=frame.index)).fillna("not_available")
    frame["n_variable_dates"] = frame.groupby("patient_id").value.transform("size")
    refs = frame.groupby("patient_id", sort=False).head(1).copy()
    official_ref = refs.clinical_episode_id.eq(refs.clinical_baseline_episode_id) & refs.variable_measurement_date.eq(refs.clinical_baseline_date)
    refs["reference_source"] = np.where(official_ref, "official_clinical_baseline", "first_observed_after_clinical_baseline")
    refs = refs[["patient_id", "value", "variable_measurement_date", "reference_source"]].rename(columns={"value": "reference_value", "variable_measurement_date": "reference_date"})
    frame = frame.merge(refs, on="patient_id", validate="many_to_one")
    frame["delta"] = frame.value-frame.reference_value if spec.type == "numeric" else np.nan
    # Singletons never enter delta summaries or trajectories.
    frame["delta"] = frame.delta.where(frame.n_variable_dates.ge(2))
    frame["is_delta_followup"] = frame.variable_measurement_date.gt(frame.reference_date) & frame.n_variable_dates.ge(2)
    frame["reference_delay_days"] = (frame.reference_date-frame.clinical_baseline_date).dt.days
    return frame, "plot_ready" if spec.expected_longitudinal and frame.n_variable_dates.ge(2).any() else "baseline_only_plotted", qc, unit


def support_table(frame, groups, spec, threshold):
    rows = []
    for pattern in LABELS:
        group = groups.loc[groups.observed_pattern.eq(pattern)]
        data = frame.loc[frame.observed_pattern.eq(pattern)] if len(frame) else frame
        first = data.groupby("patient_id").head(1) if len(data) else data
        rows.append(dict(threshold_abs=threshold, variable=spec.variable_id, observed_pattern=pattern,
            N_group=len(group), n_any=len(first), n_repeated=int(first.n_variable_dates.ge(2).sum()) if len(first) else 0,
            n_ref_official=int(first.reference_source.eq("official_clinical_baseline").sum()) if len(first) else 0,
            n_ref_delayed=int(first.reference_source.ne("official_clinical_baseline").sum()) if len(first) else 0,
            n_two_essdai_dates=int(group.n_distinct_essdai_dates.eq(2).sum()),
            median_days_to_first_measurement=first.reference_delay_days.median() if len(first) else np.nan,
            q25_days_to_first_measurement=first.reference_delay_days.quantile(.25) if len(first) else np.nan,
            q75_days_to_first_measurement=first.reference_delay_days.quantile(.75) if len(first) else np.nan))
    return pd.DataFrame(rows)


def window_contributions(frame):
    """A single observed value per person per window; never a mean of visits."""
    return frame.sort_values(["patient_id", "variable_measurement_date"]).drop_duplicates(["patient_id", "window"], keep="last")


def summarize(frame, spec, threshold, axis, unit, minimum_support=5, profile=False):
    numeric, category, availability = [], [], []
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    data = window_contributions(frame)
    if profile:
        data = frame.sort_values("variable_measurement_date").groupby("patient_id").head(1).copy()
    availability_data = data.copy()
    if profile:
        data["window"] = 0
        data["window_start"], data["window_end"] = np.nan, np.nan
    common = dict(variable=spec.variable_id, threshold_abs=threshold, time_axis=axis, unit=unit,
                  summary_type="first_available_profile" if profile else "by_window",
                  temporal_rule="first valid available observation per patient; no common time implied" if profile else "[start,end) years; last actual observed date per patient; no imputation")
    for (pattern, window), g in data.groupby(["observed_pattern", "window"], sort=True):
        raw = data.loc[data.observed_pattern.eq(pattern)] if profile else frame.loc[frame.observed_pattern.eq(pattern) & frame.window.eq(window)]
        base = dict(**common, group=pattern, window=window, window_start=float(g.window_start.iloc[0]),
                    window_end=float(g.window_end.iloc[0]), n_patients_unique=len(g), n_observations_raw=len(raw),
                    n_at_window=len(g),
                    n_delta_at_window=int(g.is_delta_followup.sum()), support_status="supported" if len(g) >= minimum_support else "insufficient support")
        if spec.type == "numeric":
            row = dict(base)
            for prefix, values in (("absolute", g.value), ("delta", g.delta.dropna())):
                for label, value in (("median", values.median()), ("q25", values.quantile(.25)), ("q75", values.quantile(.75)),
                                     ("mean", values.mean()), ("sd", values.std()), ("min", values.min()), ("max", values.max())):
                    row[prefix+"_"+label] = value
            row["n_delta_observed_including_reference"] = int(g.delta.notna().sum())
            numeric.append(row)
        else:
            levels = [1] if spec.type == "binary" else sorted(data.value.unique(), key=str)
            for level in levels:
                n, total = int(g.value.eq(level).sum()), len(g)
                # Wilson 95% interval, explicitly patient-denominated.
                p, z = n/total, 1.959963984540054
                center = (p+z*z/(2*total))/(1+z*z/total)
                half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
                category.append(dict(**base, category=str(level), n_events=n, n_evaluable=total,
                                     percentage=100*p, ci95_low=100*(center-half), ci95_high=100*(center+half), ci_method="Wilson"))
    if not profile:
        availability_data = frame.sort_values("variable_measurement_date").drop_duplicates(["patient_id", "window", "source_protocol"], keep="last")
    for (pattern, window, protocol), g in availability_data.groupby(["observed_pattern", "window", "source_protocol"], dropna=False):
        raw = frame.loc[frame.observed_pattern.eq(pattern) & frame.window.eq(window) & frame.source_protocol.eq(protocol)]
        availability.append(dict(**common, group=pattern, window=window, source_protocol=protocol,
                                 n_patients_unique=len(g), n_observations_raw=len(raw)))
    return pd.DataFrame(numeric), pd.DataFrame(category), pd.DataFrame(availability)

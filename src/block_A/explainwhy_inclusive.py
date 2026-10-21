"""Protected ESSDAI-index cohorts for the isolated Step 14 inclusive extension.

The v2 products are the index authority. Rebuilding with the upstream v2
function is a blocking reconciliation, never an alternative index algorithm.
No patient identifiers or individual dates are emitted by aggregate QC.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.block_A import observed_patterns_v2 as v2
from src.block_A.explainwhy_feasibility import audit_registry, audit_values

RULE_VERSION = "explainwhy-inclusive-v2"
FAMILY = "explainwhy_inclusive_v2"
PREFIX = "14_explainwhy_inclusive_v2_"
KEYS = v2.KEYS
DATE, SCORE = v2.DATE, v2.ESSDAI
RISK_EXTRA = ["threshold_abs", "last_evaluable_date", "first_event_date",
    "last_known_non_event_date", "first_event_time_years", "last_non_event_time_years",
    "event_or_censor_time_years", "event", "last_followup_years", "interval_censored_onset"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_frame(actual, expected, keys, label):
    if set(actual.columns) != set(expected.columns):
        raise ValueError(f"{label}: schema does not match current v2 contract")
    if actual.duplicated(keys).any() or actual[keys].isna().any().any():
        raise ValueError(f"{label}: missing or duplicate canonical keys")
    a = actual.sort_values(keys).reset_index(drop=True)
    e = expected.sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(a[e.columns], e, check_dtype=False,
                                      check_exact=True, check_names=True)
    except AssertionError:
        # Do not put a patient/score/date diff in public logs.
        raise ValueError(f"{label}: blocking canonical reconstruction mismatch") from None


def reconcile_products(integrated, baseline, patterns, eligibility, sequence):
    """Validate complete rows, not merely manifest labels or cohort counts."""
    clean, official = v2.validate_pattern_inputs(integrated, baseline)
    expected, expected_sequence = v2.build_observed_patterns_v2(clean, official)
    _same_frame(eligibility, expected, ["patient_id", "threshold_abs"], "v2 eligibility")
    _same_frame(patterns, expected.loc[expected.n_distinct_essdai_dates.ge(2)],
                ["patient_id", "threshold_abs"], "v2 eligible patterns")
    _same_frame(sequence, expected_sequence, KEYS, "v2 selected dates")
    return clean, official


def validate_manifest(manifest, paths, input_origin):
    """Physically hash every supplied input before joining any clinical frames."""
    if manifest.get("definition_version") != v2.DEFINITION_VERSION:
        raise ValueError("v2 manifest: unsupported definition_version")
    if manifest.get("qc_status") != "PASS" or manifest.get("date_policy") != v2.POLICY:
        raise ValueError("v2 manifest: QC/date policy not accepted")
    if manifest.get("input_origin") != input_origin:
        raise ValueError("v2 manifest: mixed canonical/synthetic input_origin")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("code_commit", "")):
        raise ValueError("v2 manifest: missing identifiable producer commit")
    if set(manifest.get("thresholds", [])) != {3, 5}:
        raise ValueError("v2 manifest: incompatible thresholds")
    inputs = manifest.get("inputs", [])
    for role in ("step10", "step11"):
        matches = [r for r in inputs if r.get("role") == role]
        if len(matches) != 1 or matches[0].get("sha256") != sha256(paths[role]):
            raise ValueError(f"v2 manifest: physical {role} SHA-256 mismatch or absent")
    names = {"patterns": "13_essdai_patient_observed_patterns_v2.parquet",
        "eligibility": "13_essdai_pattern_patient_eligibility_v2.parquet",
        "sequence": "13_essdai_pattern_selected_dates_v2.parquet"}
    for role, name in names.items():
        if Path(paths[role]).name != name:
            raise ValueError(f"v2 manifest: explicit {role} path must name {name}")
        matches = [r for r in manifest.get("artifacts", []) if Path(r.get("path", "")).name == name]
        if len(matches) != 1 or matches[0].get("sha256") != sha256(paths[role]):
            raise ValueError(f"v2 manifest: physical {role} artifact hash mismatch or absent")
    root = Path(__file__).resolve().parents[2]
    for rel in ("config.py", "src/block_A/observed_patterns_v2.py",
                "src/block_A/observed_patterns.py", "src/block_A/13_disease_activity_progression.py"):
        if manifest.get("source_hashes", {}).get(rel) != sha256(root / rel):
            raise ValueError(f"v2 manifest: producer definition hash mismatch: {rel}")
    return {role: sha256(path) for role, path in paths.items()}


def index_cohort(eligibility, source_hashes=None):
    """Retain the entire Step 11 universe, including longitudinal exclusions."""
    cohort = eligibility.loc[eligibility.threshold_abs.eq(3), v2.META_COLUMNS].copy()
    if cohort.patient_id.duplicated().any():
        raise ValueError("index cohort: duplicate patient")
    cohort["index_source"] = cohort.essdai_index_source
    cohort["index_delay_days"] = cohort.days_from_clinical_baseline_to_essdai_index
    cohort["n_post_index_valid_essdai_dates"] = cohort.n_essdai_followup_dates
    cohort["eligible"] = cohort.eligibility_status.eq("longitudinally_evaluable")
    cohort["entry_status"] = np.where(cohort.eligible, "observed_essdai_entry", "excluded")
    cohort["index_time_origin"] = "essdai_index_date"
    cohort["observation_policy"] = v2.POLICY
    cohort["rule_version"] = RULE_VERSION
    cohort["source_hashes"] = json.dumps(source_hashes or {}, sort_keys=True)
    return cohort.reset_index(drop=True)


def first_event_riskset(cohort, sequence, threshold=3):
    if threshold not in (3, 5):
        raise ValueError("only prespecified +3/+5 thresholds are supported")
    groups = dict(tuple(sequence.groupby("patient_id", sort=False)))
    rows = []
    for record in cohort.loc[cohort.eligible].to_dict("records"):
        g = groups[record["patient_id"]].sort_values(DATE)
        dates = pd.DatetimeIndex(g[DATE])
        if (dates.has_duplicates or dates.hasnans or len(g) < 2 or
                dates[0] != record["essdai_index_date"] or
                g[SCORE].iloc[0] != record["essdai_index_score"] or g[SCORE].isna().any()):
            raise ValueError("risk set: invalid validated v2 sequence")
        after = g.loc[g[DATE].gt(record["essdai_index_date"])]
        positive = after.loc[after[SCORE].sub(record["essdai_index_score"]).ge(threshold)]
        event = not positive.empty
        detected = positive[DATE].iloc[0] if event else pd.NaT
        end = detected if event else after[DATE].iloc[-1]
        lower = g.loc[g[DATE].lt(detected), DATE].iloc[-1] if event else end
        origin = record["essdai_index_date"]
        years = lambda date: (date - origin).days / 365.25 if pd.notna(date) else np.nan
        rows.append({**record, "threshold_abs": threshold, "last_evaluable_date": after[DATE].iloc[-1],
            "first_event_date": detected, "last_known_non_event_date": lower,
            "first_event_time_years": years(detected), "last_non_event_time_years": years(lower),
            "event_or_censor_time_years": years(end), "event": int(event),
            "last_followup_years": years(after[DATE].iloc[-1]), "interval_censored_onset": True})
    return pd.DataFrame(rows, columns=list(cohort.columns) + RISK_EXTRA)


def strict_predictor_values(frame, spec, step14, registry, baseline_mode=False, max_age_days=0):
    """No implicit carry forward: primary labs require same-date evidence.

    A nonzero age limit is an explicit, prespecified sensitivity, never a
    replacement for the strict primary. Registry provenance is still required.
    """
    if spec.family == "treatment":
        return pd.Series(np.nan, index=frame.index)
    values, _, _, _, _, _ = audit_values(frame, spec, step14, registry, baseline_mode)
    column = spec.baseline_column if baseline_mode else spec.time_varying_column
    if spec.family == "laboratory" and column:
        date = pd.to_datetime(frame.get(column.removesuffix("__value") + "__measurement_date",
                             pd.Series(pd.NaT, index=frame.index)), errors="coerce")
        age = (frame[DATE] - date).dt.days
        values = values.where(age.between(0, max_age_days))
    return values


def build_index_predictors(integrated, baseline, cohort, step14, registry, max_age_days=0):
    registered = [s for s in step14.build_predictor_registry() if s.primary_role == "adjustment"] + audit_registry(step14)
    specs = step14.resolve_predictors(registered, baseline, integrated)
    keyed = cohort.loc[cohort.eligible, ["patient_id", "essdai_index_episode_id"]].rename(
        columns={"essdai_index_episode_id": "clinical_episode_id"})
    episodes = keyed.merge(integrated, on=KEYS, how="left", validate="one_to_one")
    out = cohort.loc[cohort.eligible].copy()
    out = out.merge(episodes[KEYS + [DATE]], left_on=["patient_id", "essdai_index_episode_id"],
                    right_on=KEYS, validate="one_to_one").drop(columns="clinical_episode_id")
    availability = []
    for spec in specs:
        if spec.primary_role == "adjustment":
            continue
        values = strict_predictor_values(episodes, spec, step14, registry, max_age_days=max_age_days)
        out["index_predictor_" + spec.name] = values.to_numpy()
        out["index_predictor_asof_" + spec.name] = out.essdai_index_date.where(values.notna().to_numpy())
        if spec.family == "laboratory" and spec.time_varying_column:
            stem = spec.time_varying_column.removesuffix("__value")
            out["index_predictor_measurement_date_" + spec.name] = pd.to_datetime(
                episodes.get(stem + "__measurement_date", pd.Series(pd.NaT, index=episodes.index)), errors="coerce").to_numpy()
            out["pred_unit__" + spec.name] = episodes.get(stem + "__unit", pd.Series(pd.NA, index=episodes.index)).to_numpy()
        for source in (v2.OFFICIAL, v2.FALLBACK):
            mask = out.index_source.eq(source)
            available = out["index_predictor_" + spec.name].notna()
            availability.append(dict(predictor=spec.name, predictor_family=spec.family, index_source=source,
                n_cohort=int(mask.sum()), n_predictor_available=int((mask & available).sum()),
                n_missing=int((mask & ~available).sum()), max_lab_age_days=max_age_days,
                status="not_implemented" if spec.family == "treatment" else "evaluated",
                temporal_rule="registered as-of index; labs require explicit date, unit, age window"))
    out = out.copy()  # consolidate evidence columns before estimator adapters
    # Static Step 11 demographics retain their actual measurement clock.
    official = baseline.set_index("patient_id").reindex(out.patient_id).reset_index()
    for name in ("age", "sex"):
        spec = next(s for s in specs if s.name == name)
        out["index_predictor_" + name] = step14.predictor_values(official, spec, True).to_numpy()
    out["age_reference"] = "official_clinical_baseline; not age at ESSDAI index"
    out["index_predictor_duration"] = np.nan
    duration = next(s for s in specs if s.name == "duration")
    if "dx_date" in official and duration.baseline_column:
        verified = step14.predictor_values(official, duration, True)
        out["index_predictor_duration"] = verified.where(out.index_source.eq(v2.OFFICIAL).to_numpy()).to_numpy()
    out["duration_status"] = np.where(out.index_predictor_duration.notna(),
        "verified_step11_at_official_index", "not_available_at_index")
    out["baseline_essdai"] = out.essdai_index_score  # adapter field for unchanged estimator API
    adapters = {"pred__"+spec.name: out["index_predictor_"+spec.name]
                for spec in specs if "index_predictor_"+spec.name in out}
    out = pd.concat([out, pd.DataFrame(adapters, index=out.index)], axis=1)
    return out, specs, pd.DataFrame(availability)


def build_lagged_intervals(integrated, cohort, sequence, risks, specs, step14, registry, provenance_policy=None):
    """Shift the full clinical spine before any outcome/predictor filtering."""
    from src.block_A.explainwhy_inclusive_composition import VERSION_FIELDS, side_field, apply_composition_gate
    if integrated[KEYS].isna().any().any() or integrated.duplicated(KEYS).any():
        raise ValueError("inclusive intervals: missing or duplicate canonical episode keys")
    selected = set(map(tuple, sequence[KEYS].astype(str).to_numpy()))
    risk_lookup = risks.set_index("patient_id")
    values = {s.name: strict_predictor_values(integrated, s, step14, registry)
              for s in specs if s.primary_role != "adjustment"}
    domains = {d: step14.domain_values(integrated, d) for d in step14.DOMAINS}
    indexed = integrated.set_index(KEYS)
    rows = []
    for entry in cohort.loc[cohort.eligible].to_dict("records"):
        pid, origin = entry["patient_id"], entry["essdai_index_date"]
        g = integrated.loc[integrated.patient_id.eq(pid)]
        order = [DATE] + (["clinical_visit_number"] if "clinical_visit_number" in g else []) + ["clinical_episode_id"]
        g = g.sort_values(order, kind="stable")
        index_episode = indexed.loc[(pid, entry["essdai_index_episode_id"])]
        index_domains = {d: step14.domain_values(pd.DataFrame([index_episode]), d).iloc[0] for d in step14.DOMAINS}
        selected_patient = sequence.loc[sequence.patient_id.eq(pid), KEYS].merge(integrated, on=KEYS, validate="one_to_one")
        first_domain_event = {}
        for d in step14.DOMAINS:
            positive = step14.domain_values(selected_patient, d).gt(0) & selected_patient[DATE].gt(origin)
            first_domain_event[d] = selected_patient.loc[positive, DATE].min()
        event_date = risk_lookup.loc[pid, "first_event_date"]
        for pos in range(len(g)-1):
            a, z = g.iloc[pos], g.iloc[pos+1]
            if pd.isna(a[DATE]) or pd.isna(z[DATE]) or a[DATE] < origin or z[DATE] <= a[DATE]:
                continue
            known = pd.notna(a[SCORE]) and pd.notna(z[SCORE])
            chosen = (str(pid), str(a.clinical_episode_id)) in selected and (str(pid), str(z.clinical_episode_id)) in selected
            outcome_valid = bool(known and chosen)
            at_risk = pd.isna(event_date) or a[DATE] < event_date
            row = dict(patient_id=pid, index_source=entry["index_source"], index_time_origin="essdai_index_date",
                essdai_index_date=origin, essdai_index_episode_id=entry["essdai_index_episode_id"],
                essdai_index_score=entry["essdai_index_score"], baseline_essdai=entry["essdai_index_score"],
                from_clinical_episode_id=a.clinical_episode_id, to_clinical_episode_id=z.clinical_episode_id,
                from_date=a[DATE], to_date=z[DATE], from_essdai=a[SCORE], to_essdai=z[SCORE],
                interval_years=(z[DATE]-a[DATE]).days/365.25,
                start_time=(a[DATE]-origin).days/365.25, stop_time=(z[DATE]-origin).days/365.25,
                selected_endpoints=chosen, outcome_evaluable=outcome_valid,
                at_risk_first_delta3=bool(at_risk),
                event_first_delta3=float(z[DATE] == event_date) if at_risk and outcome_valid else np.nan,
                rule_version=RULE_VERSION)
            for s in specs:
                if s.primary_role == "adjustment":
                    continue
                row["pred__" + s.name] = values[s.name].loc[a.name]
                if s.family == "laboratory" and s.time_varying_column:
                    stem = s.time_varying_column.removesuffix("__value")
                    row["pred_unit__" + s.name] = a.get(stem + "__unit", pd.NA)
            for d in step14.DOMAINS:
                fr, to = domains[d].loc[a.name], domains[d].loc[z.name]
                first_positive = first_domain_event[d]
                before_first = pd.isna(first_positive) or a[DATE] < first_positive
                risk = pd.notna(index_domains[d]) and index_domains[d] == 0 and before_first and fr == 0 and pd.notna(to) and chosen
                row["from_domain_"+d], row["to_domain_"+d] = fr, to
                row["at_risk_new_"+d] = bool(risk)
                row["event_new_domain_"+d] = float(to > 0) if risk else np.nan
            for side, episode in (("from", a), ("to", z)):
                for field in VERSION_FIELDS:
                    row[side_field(side, field)] = episode.get(field, pd.NA)
            rows.append(row)
    columns = ["patient_id", "index_source", "index_time_origin", "essdai_index_date", "essdai_index_episode_id",
        "essdai_index_score", "baseline_essdai", "from_clinical_episode_id", "to_clinical_episode_id", "from_date", "to_date",
        "from_essdai", "to_essdai", "interval_years", "start_time", "stop_time", "selected_endpoints",
        "outcome_evaluable", "at_risk_first_delta3", "event_first_delta3", "rule_version"]
    columns += ["pred__"+s.name for s in specs if s.primary_role != "adjustment"]
    columns += ["pred_unit__"+s.name for s in specs if s.family == "laboratory"]
    columns += [prefix+d for d in step14.DOMAINS for prefix in ("from_domain_", "to_domain_", "at_risk_new_", "event_new_domain_")]
    columns += [side_field(side, field) for side in ("from", "to") for field in VERSION_FIELDS]
    out = pd.DataFrame(rows, columns=columns)
    out, composition = step14.derive_other_domain_adjustments(out, specs)
    return apply_composition_gate(out, composition, step14.DOMAINS, provenance_policy)


def legacy_bridge(integrated, risksets, legacy, step14):
    """Reconstruct all-episode flags/dates and explain every actual disagreement."""
    expected = step14.step13.build_patient_absolute_change_endpoints(integrated)
    _same_frame(legacy, expected, ["patient_id"], "Step 13 legacy endpoints")
    raw = integrated.loc[integrated[SCORE].notna() & integrated[DATE].notna()]
    conflicts = raw.groupby(["patient_id", DATE])[SCORE].nunique()
    conflicting = set(conflicts.index[conflicts.gt(1)])
    legacy = legacy.set_index("patient_id")
    rows = []
    for k, risk in risksets.items():
        subset = risk.loc[risk.index_source.eq(v2.OFFICIAL)]
        differences, explained = 0, 0
        for r in subset.to_dict("records"):
            old = legacy.loc[r["patient_id"]]
            old_event = bool(old[f"ever_worsened_ge{k}"])
            old_date = pd.to_datetime(old[f"first_observed_worsening_ge{k}_date"])
            new_date = r["first_event_date"]
            date_equal = (pd.isna(old_date) and pd.isna(new_date)) or old_date == new_date
            if old_event == bool(r["event"]) and date_equal:
                continue
            differences += 1
            # The exact earliest all-episode positive must fall on a conflicting
            # date that selected-date policy removed or changed.
            is_tie = (r["patient_id"], old_date) in conflicting
            if not is_tie or (pd.notna(new_date) and (pd.isna(old_date) or new_date < old_date)):
                raise ValueError("legacy bridge: unexplained first-event discrepancy")
            explained += 1
        rows.append(dict(threshold_abs=k, n_official=len(subset), n_disagreements=differences,
            n_explained_same_date=explained, n_unexplained=0, status="PASS",
            policy_v1="all_canonical_episodes", policy_v2=v2.POLICY))
    return pd.DataFrame(rows)


def event_summary(risks, label="inclusive_primary"):
    rows = []
    for k, risk in risks.items():
        for source, g in [("all", risk)] + [(s, risk.loc[risk.index_source.eq(s)]) for s in (v2.OFFICIAL, v2.FALLBACK)]:
            rows.append(dict(analysis=label, threshold_abs=k, index_source=source, n_eligible=len(g),
                n_events=int(g.event.sum()), n_censored=int(len(g)-g.event.sum()),
                observed_person_years=float(g.event_or_censor_time_years.sum()),
                median_followup_years=g.last_followup_years.median(),
                median_n_dates=g.n_distinct_essdai_dates.median(),
                n_selected_dates=int(g.n_distinct_essdai_dates.sum()),
                median_index_delay_days=g.index_delay_days.median(), min_index_delay_days=g.index_delay_days.min(),
                q25_index_delay_days=g.index_delay_days.quantile(.25), q75_index_delay_days=g.index_delay_days.quantile(.75),
                max_index_delay_days=g.index_delay_days.max(),
                note="first observed detection; onset interval censored; no pre-index risk"))
    return pd.DataFrame(rows)


def cohort_flow(cohort):
    rows = []
    for source in (v2.OFFICIAL, v2.FALLBACK):
        g = cohort.loc[cohort.index_source.eq(source)]
        rows.append(dict(index_source=source, n_step11=len(g), n_indexed=int(g.essdai_index_date.notna().sum()),
            n_eligible=int(g.eligible.sum()), n_excluded=int((~g.eligible).sum())))
    for reason, g in cohort.loc[~cohort.eligible].groupby("exclusion_reason"):
        rows.append(dict(index_source="all", exclusion_reason=reason, n_excluded=len(g)))
    return pd.DataFrame(rows)


def temporal_qc(integrated, baseline, cohort, sequence, risks, intervals, predictors, specs, step14, registry, provenance_policy=None):
    """Reuse unchanged legacy QC on its own official-baseline estimand first.

    Then validate the separate index clock and strict predictor rules. Neither
    the historical QC nor its definition of baseline is weakened for fallback.
    """
    old_clean, old_base = step14.validate_inputs(integrated, baseline)
    old_specs = step14.resolve_predictors(step14.build_predictor_registry(), old_base, old_clean)
    old_base = step14.build_baseline_analysis_dataset(old_base, old_specs)
    old_intervals = step14.build_lagged_interval_dataset(old_clean, old_base, old_specs)
    old_risks = step14.build_domain_risk_sets(old_clean, old_base)
    old_checks = step14.run_temporal_leakage_qc(old_clean, old_base, old_intervals, old_risks, old_specs, hard_fail=False)
    if old_checks.status.ne("pass").any():
        raise ValueError("legacy temporal QC failed; no inclusive results may be published")
    checks = [dict(check="legacy:"+r["check"], status="PASS", n_failures=0) for r in old_checks.to_dict("records")]
    def check(name, failures):
        n = int(failures)
        checks.append(dict(check=name, status="FAIL" if n else "PASS", n_failures=n))
        if n:
            raise ValueError(f"inclusive temporal QC failed: {name} ({n} violations)")
    for k, risk in risks.items():
        expected_risk = first_event_riskset(cohort, sequence, k)
        _same_frame(risk, expected_risk, ["patient_id", "threshold_abs"], f"inclusive first delta{k}")
        check(f"delta{k}:unique", risk.duplicated(["patient_id", "threshold_abs"]).sum())
        check(f"delta{k}:positive_time", (~risk.event_or_censor_time_years.gt(0)).sum())
        event = risk.loc[risk.event.eq(1)]
        check(f"delta{k}:strict_post_index_detection", (~event.first_event_date.gt(event.essdai_index_date)).sum())
        check(f"delta{k}:ordered_onset_bounds", (~event.last_known_non_event_date.lt(event.first_event_date)).sum())
    expected_predictors, _, _ = build_index_predictors(integrated, baseline, cohort, step14, registry)
    _same_frame(predictors, expected_predictors, ["patient_id"], "inclusive index-time predictors")
    expected_intervals, _ = build_lagged_intervals(integrated, cohort, sequence, risks[3], specs, step14, registry, provenance_policy)
    _same_frame(intervals, expected_intervals, ["patient_id", "from_clinical_episode_id", "to_clinical_episode_id"], "inclusive intervals")
    check("inclusive:post_index_intervals", (~intervals.from_date.ge(intervals.essdai_index_date)).sum())
    check("inclusive:positive_adjacent_intervals", (~intervals.to_date.gt(intervals.from_date)).sum())
    for spec in specs:
        name = "index_predictor_" + spec.name
        if spec.family == "laboratory" and name in predictors:
            date = predictors.get("index_predictor_measurement_date_" + spec.name, pd.Series(pd.NaT, index=predictors.index))
            check("inclusive:no_future_index_lab:"+spec.name,
                (predictors[name].notna() & (~date.le(predictors.essdai_index_date) | date.isna())).sum())
    check("inclusive:official_clock_preserved", cohort.official_clinical_baseline_date.ne(
        cohort.patient_id.map(baseline.set_index("patient_id").clinical_baseline_date)).sum())
    return pd.DataFrame(checks)

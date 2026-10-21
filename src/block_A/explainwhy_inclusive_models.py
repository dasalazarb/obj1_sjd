"""New model families using unchanged Step 14 support and fit diagnostics.

Only model adapters use the legacy name ``baseline_essdai``: its value here is
the verified ESSDAI index score. Output metadata states the actual clock.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from src.block_A import observed_patterns_v2 as v2
from src.block_A.explainwhy_inclusive import FAMILY, RULE_VERSION, DATE, SCORE


def state_and_domain_risks(integrated, cohort, sequence, step14):
    """Index-inactive domains only; missing initial scores never become zero."""
    selected = sequence[v2.KEYS].merge(integrated, on=v2.KEYS, validate="one_to_one")
    rows = []
    for r in cohort.loc[cohort.eligible].to_dict("records"):
        g = selected.loc[selected.patient_id.eq(r["patient_id"])].sort_values(DATE)
        initial, after = g.iloc[0], g.iloc[1:]
        series = {d: step14.domain_values(g, d) for d in step14.DOMAINS}
        inactive = [d for d, values in series.items() if values.iloc[0] == 0]
        targets = {"new_domain_"+d: (values.iloc[1:].gt(0), values.iloc[1:].notna())
                   for d, values in series.items() if d in inactive}
        if inactive:
            panel = pd.DataFrame({d: series[d].iloc[1:] for d in inactive})
            targets["new_domain_any"] = (panel.gt(0).any(axis=1), panel.notna().all(axis=1))
        if r["index_activity_state"] == "Low":
            targets["essdai_ge5"] = (after[SCORE].ge(5), after[SCORE].notna())
        if r["index_activity_state"] == "Moderate":
            targets["high_activity"] = (after[SCORE].ge(14), after[SCORE].notna())
        for outcome, (positive, known) in targets.items():
            evaluable = positive | known
            if not evaluable.any():
                continue
            event = bool(positive.any())
            date = after.loc[positive, DATE].iloc[0] if event else after.loc[known, DATE].iloc[-1]
            rows.append(dict(patient_id=r["patient_id"], outcome=outcome, y=int(event),
                time_years=(date-r["essdai_index_date"]).days/365.25))
    return pd.DataFrame(rows, columns=["patient_id", "outcome", "y", "time_years"])


def build_tasks(integrated, cohort, sequence, risks, predictors, intervals, specs, step14,
                sensitivity="none", include_secondary=True):
    tasks = []
    index_cols = ["patient_id", "baseline_essdai"]
    index_cols += [c for c in predictors if c.startswith(("pred__", "pred_unit__"))]
    covariates = predictors[index_cols]
    long = sequence.merge(covariates, on="patient_id", validate="many_to_one")
    long = long.loc[long.time_since_essdai_index_years.gt(0)].rename(
        columns={SCORE: "y", "time_since_essdai_index_years": "time"})
    base_risks = {k: frame.rename(columns={"event": "y", "event_or_censor_time_years": "time_years"})
        .merge(covariates.drop(columns="baseline_essdai"), on="patient_id", validate="one_to_one")
        .assign(baseline_essdai=lambda d: d.essdai_index_score) for k, frame in risks.items()}
    lag = intervals.merge(predictors[["patient_id", "pred__age", "pred__sex", "pred__duration"]],
                          on="patient_id", validate="many_to_one")
    state_risks = state_and_domain_risks(integrated, cohort, sequence, step14)
    for spec in specs:
        if spec.primary_role == "adjustment":
            continue
        # Labs retain their prespecified exploratory representation. No assay
        # changes primary status merely because its p-value is small.
        sens = sensitivity if sensitivity != "none" else (
            "continuous_laboratory" if spec.primary_role == "sensitivity" else "none")
        tasks.append(step14.ModelTask(spec, base_risks[3], "baseline", "first_delta3", "cox", sens))
        tasks.append(step14.ModelTask(spec, base_risks[5], "baseline", "first_delta5", "cox",
            sensitivity+":delta5" if sensitivity != "none" else "delta5"))
        tasks.append(step14.ModelTask(spec, long, "baseline", "essdai_slope", "trajectory", sens))
        subsequent = lag.loc[lag.outcome_evaluable.eq(True)].assign(y=lambda d: d.to_essdai)
        tasks.append(step14.ModelTask(spec, subsequent, "timevarying", "subsequent_essdai", "lagged", sens))
        at_risk = lag.loc[lag.at_risk_first_delta3.eq(True) & lag.outcome_evaluable.eq(True)].assign(y=lambda d: d.event_first_delta3)
        tasks.append(step14.ModelTask(spec, at_risk, "timevarying", "first_delta3", "poisson", sens))
        if spec.family == "essdai_domain":
            domain = spec.name.removeprefix("domain_")
            state = "from_essdai_other_domains_"+domain
            target = "to_essdai_other_domains_"+domain
            compositional = subsequent.loc[subsequent[state].notna() & subsequent[target].notna()].assign(y=lambda d: d[target])
            reason = ""
            if compositional.empty:
                gate = subsequent.get("composition_pair_gate_"+domain, pd.Series(dtype="string"))
                failures = sorted(gate.dropna().unique())
                provenance = {"missing_version_evidence", "producer_identity_unverified", "incompatible_scoring_versions"}
                reason = ("not_estimable_version_provenance_unverified" if provenance.intersection(failures)
                          else "not_estimable_composition_support")
                reason += ": " + (", ".join(failures) or "no_adjacent_evaluable_intervals")
            tasks.append(step14.ModelTask(spec, compositional, "timevarying", "subsequent_essdai", "lagged",
                step14.LEAVE_SOURCE_SENSITIVITY if sensitivity == "none" else sensitivity+":"+step14.LEAVE_SOURCE_SENSITIVITY,
                state_covariate=state, support_exclusion=reason,
                comparison_role="other_domains" if not compositional.empty else ""))
            if not compositional.empty:
                tasks.append(step14.ModelTask(spec, compositional.assign(y=lambda d: d.to_essdai),
                    "timevarying", "subsequent_essdai", "lagged",
                    ("" if sensitivity == "none" else sensitivity+":")+"matched_total_leave_source_domain_outcome",
                    state_covariate="from_essdai", comparison_role="matched_total"))
        if not include_secondary:
            continue
        for outcome in ["essdai_ge5", "high_activity", "new_domain_any", *["new_domain_"+d for d in step14.DOMAINS]]:
            target = outcome.removeprefix("new_domain_")
            source = state_risks.loc[state_risks.outcome.eq(outcome)].drop(columns="outcome").merge(
                covariates, on="patient_id", validate="one_to_one")
            tasks.append(step14.ModelTask(spec, source, "baseline", outcome, "cox", sens,
                support_exclusion="predictor equals target domain" if spec.name == "domain_"+target else ""))
    if include_secondary:
        for spec in specs:
            if spec.family != "essdai_domain":
                continue
            for target in step14.DOMAINS:
                if spec.name == "domain_"+target:
                    continue
                g = lag.loc[lag["at_risk_new_"+target].eq(True)].assign(y=lambda d: d["event_new_domain_"+target])
                tasks.append(step14.ModelTask(replace(spec, transform="binary_active"), g, "cross_domain",
                    "new_domain_"+target, "poisson", sensitivity if sensitivity != "none" else "cross_domain_secondary",
                    minimum_exposed_events=step14.MIN_CROSS_EXPOSED_EVENTS))
    return tasks


def fit_models(tasks, cohort, step14, minimum_events, official_comparison=True, reference_results=None):
    """Official-only freezes the inclusive adjustment and predictor scale.

    A failing official-only design stays not estimable at that same level; it
    never selects a different adjustment or rescales the predictor.
    """
    rows, attempts = [], []
    composition_references = {}
    official_ids = set(cohort.loc[cohort.eligible & cohort.index_source.eq(v2.OFFICIAL), "patient_id"])
    fallback_ids = set(cohort.loc[cohort.eligible & cohort.index_source.eq(v2.FALLBACK), "patient_id"])
    for task in tasks:
        task.data = task.data.copy()
        if task.comparison_role == "matched_total":
            source = composition_references[task.spec.name]
            task = replace(task, fixed_adjustment_level=source["adjustment_status"])
        reference = None
        reporting_spec = task.spec
        if reference_results is not None:
            match = reference_results.loc[reference_results.analysis_type.eq(task.analysis_type) &
                reference_results.outcome.eq(task.outcome) & reference_results.predictor.eq(task.spec.name) &
                reference_results.population.ne("official_only")]
            if len(match) != 1:
                raise ValueError("sensitivity: unique inclusive reference specification required")
            reference = match.iloc[0]
            task = replace(task, fixed_adjustment_level=reference.adjustment_status)
            if task.spec.transform == "zscore":
                col = "pred__"+task.spec.name
                task.data[col] = (pd.to_numeric(task.data[col], errors="coerce")-reference.standardization_mean)/reference.standardization_sd
                task = replace(task, spec=replace(task.spec, transform="none"))
        # Trajectory selected-date product has no source column; index metadata
        # are attached once, and never used as an exposure.
        data, meta = step14.prepare_model(task, int(cohort.eligible.sum()), minimum_events)
        if task.comparison_role == "other_domains":
            composition_references[task.spec.name] = meta
        if task.comparison_role == "matched_total" and meta["analysis_sample_hash"] != source["analysis_sample_hash"]:
            raise ValueError("composition matched total: complete-case interval sample differs")
        if reference is not None:
            task.spec = reporting_spec
            meta.update(standardization_mean=reference.standardization_mean, standardization_sd=reference.standardization_sd)
            if meta["formula"] != reference.formula:
                meta.update(supported_for_model=False, support_reason="sensitivity cannot retain inclusive reference formula")
        start_attempt = len(attempts)
        row = step14.fit_task(task, data, meta, attempts)
        for attempt in attempts[start_attempt:]:
            attempt.update(analysis_family=FAMILY, population="inclusive", index_time_origin="essdai_index_date",
                n_complete_case_fallback=int(data.patient_id.isin(fallback_ids).groupby(data.patient_id).any().sum()))
        population = "official_only_v1_all_episodes" if task.sensitivity.startswith("v1_all_episodes_official") else (
            "inclusive_primary" if task.sensitivity in ("none", "continuous_laboratory") else "inclusive_sensitivity")
        variants = [(task, data, meta, row, population)]
        if official_comparison:
            subset = task.data.loc[task.data.patient_id.isin(official_ids)].copy()
            frozen_spec = task.spec
            if frozen_spec.transform == "zscore":
                col = "pred__"+frozen_spec.name
                subset[col] = (pd.to_numeric(subset.get(col), errors="coerce")-meta["standardization_mean"])/meta["standardization_sd"]
                frozen_spec = replace(frozen_spec, transform="none")
            official_task = replace(task, spec=frozen_spec, data=subset, sensitivity="official_only:"+task.sensitivity,
                fixed_adjustment_level=meta["adjustment_status"])
            od, om = step14.prepare_model(official_task, len(official_ids), minimum_events)
            if om["formula"] != meta["formula"]:
                om["supported_for_model"] = False
                om["support_reason"] += "; official-only cannot retain the inclusive formula"
            # Restore the reporting scale after the standardized design is frozen.
            official_task.spec = task.spec
            om.update(standardization_mean=meta["standardization_mean"], standardization_sd=meta["standardization_sd"])
            start_attempt = len(attempts)
            result = step14.fit_task(official_task, od, om, attempts)
            for attempt in attempts[start_attempt:]:
                attempt.update(analysis_family=FAMILY, population="official_only", index_time_origin="essdai_index_date",
                    n_complete_case_fallback=0)
            variants.append((official_task, od, om, result, "official_only"))
        for current, fitted, metadata, result, population in variants:
            ids = set(current.data.patient_id)
            fitted_ids = set(fitted.patient_id)
            result.update(analysis_family=FAMILY, population=population, index_time_origin="essdai_index_date",
                index_state_adjustment="ESSDAI at index; adapter baseline_essdai", age_reference="age at official clinical baseline",
                treatment_adjustment="not_implemented", rule_version=RULE_VERSION,
                n_cohort=int(cohort.eligible.sum()), n_official_cohort=len(official_ids), n_fallback_cohort=len(fallback_ids),
                n_at_risk=len(ids), n_at_risk_fallback=len(ids & fallback_ids),
                n_fitted=metadata["n_patients"] if result["valid_for_inference"] else 0,
                n_complete_case_fallback=len(fitted_ids & fallback_ids),
                n_fitted_fallback=len(fitted_ids & fallback_ids) if result["valid_for_inference"] else 0,
                n_fallback_with_predictor=current.data.loc[current.data.get("pred__"+current.spec.name,
                    pd.Series(np.nan, index=current.data.index)).notna() & current.data.patient_id.isin(fallback_ids), "patient_id"].nunique(),
                structural_note="biological/domain composition can be algebraically coupled to total ESSDAI; association only"
                    if current.spec.family in {"laboratory", "essdai_domain", "serology"} else "association only")
            if step14.LEAVE_SOURCE_SENSITIVITY in current.sensitivity and current.comparison_role != "matched_total":
                result["estimand"] = "observational association with future ESSDAI outside the source domain; not a first-delta3 outcome"
            result["fdr_family"] = FAMILY+"__"+result["fdr_family"]
            rows.append(result)
    out = pd.DataFrame(rows)
    out, within = step14.apply_fdr_by_family(out)
    out, wide = step14.apply_outcome_wide_fdr(out)
    within["scope"] = "within_new_family_only"
    wide["scope"] = "outcome_wide_new_canonical_primary_only"
    return out, pd.DataFrame(attempts), pd.concat([within, wide], ignore_index=True)


def feasibility_table(results, cohort, intervals):
    fields = ["population", "analysis_type", "outcome", "predictor", "sensitivity", "model_status",
        "support_reason", "n_cohort", "n_at_risk", "n_at_risk_fallback", "n_with_predictor_available",
        "n_complete_cases", "n_complete_case_fallback", "n_fitted", "n_fitted_fallback", "n_events",
        "selected_adjustment_level", "formula"]
    table = results[fields].copy()
    table["component_status"] = np.where(results.valid_for_inference, "IMPLEMENTED", "NOT_ESTIMABLE")
    # Count outcome events whose adjacent FROM or predictor cannot be modelled.
    risk = intervals.loc[intervals.at_risk_first_delta3.eq(True)]
    table["event_modelability_note"] = "event_not_modelable_due_to_missing_from_or_predictor is predictor-specific; outcome retained"
    table.attrs["n_candidate_intervals"] = len(risk)
    return table


def domain_index_support(integrated, cohort, step14):
    """Expose unavailable fallback entry states without dropping cohort members."""
    eligible = cohort.loc[cohort.eligible]
    episodes = eligible[["patient_id", "essdai_index_episode_id", "index_source"]].rename(
        columns={"essdai_index_episode_id": "clinical_episode_id"}).merge(integrated, on=v2.KEYS, validate="one_to_one")
    rows = []
    for domain in step14.DOMAINS:
        values = step14.domain_values(episodes, domain)
        for source in (v2.OFFICIAL, v2.FALLBACK):
            mask = episodes.index_source.eq(source)
            known = values.loc[mask].notna()
            inactive = values.loc[mask].eq(0)
            rows.append(dict(domain=domain, index_source=source, n_cohort=int(mask.sum()),
                n_index_evaluable=int(known.sum()), n_index_inactive=int(inactive.sum()), n_index_unknown=int((~known).sum()),
                status="not_estimable_for_fallback" if source == v2.FALLBACK and not inactive.any() else "entry_state_audited",
                reason="Requires observed inactive target in the exact ESSDAI index episode; missing is not zero"))
    return pd.DataFrame(rows)

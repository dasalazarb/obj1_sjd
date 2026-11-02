"""Protected, aggregate-only V2 diagnostics; no estimator or patient exports."""
import numpy as np
import pandas as pd

from src.block_A.explainwhy_inclusive import FAMILY, RULE_VERSION
from src.block_A import observed_patterns_v2 as v2


def pns_support_summary(task, data, metadata, result, population, step14,
                        minimum_events, input_origin):
    """Audit the exact transformed complete cases; delete entire clusters in memory.

    Counts are protected QC pending institutional disclosure review. Never retain
    deletion-level records, identifiers, dates, coefficients or significance tests.
    """
    counts = step14._support_counts(data, task)
    active = pd.to_numeric(data.x, errors="coerce").gt(0)
    exposed_ids = data.loc[active, "patient_id"].unique()
    # Include all intervals of exposed patients, including their inactive visits.
    sizes = data.loc[data.patient_id.isin(exposed_ids)].groupby("patient_id").size()
    episode_keys = ["from_clinical_episode_id", "to_clinical_episode_id"]
    episodes = (pd.concat([data[["patient_id", key]].rename(columns={key: "episode"})
        for key in episode_keys], ignore_index=True).drop_duplicates().shape[0]
        if all(key in data for key in episode_keys) else np.nan)
    deletions = []
    for patient in exposed_ids:
        remaining = data.loc[~data.patient_id.eq(patient)]
        support, _, reasons = step14._check_support(remaining, task, minimum_events, metadata["formula"])
        deletions.append((support["n_exposed"], len(remaining), bool(reasons)))
    below = sum(d[2] for d in deletions)
    minimum = min((d[0] for d in deletions), default=np.nan)
    if not metadata["supported_for_model"] or not result.get("valid_for_inference", False):
        status = "NOT_ESTIMABLE_OR_NOT_AVAILABLE"
    elif minimum < step14.MIN_EXPOSED:
        status = "FRAGILE_MINIMUM_EXPOSED"
    elif below:
        status = "FRAGILE_OTHER_SUPPORT"
    else:
        status = "SUPPORT_REMAINS"
    return dict(analysis_family=FAMILY, population=population, predictor="domain_pns",
        outcome="subsequent_essdai", model_status=result["model_status"],
        n_model_patients=counts["n_patients"], n_exposed_patients=counts["n_exposed"],
        n_unexposed_patients=counts["n_unexposed"], n_model_intervals=len(data), n_model_episodes=episodes,
        n_exposed_intervals=int(active.sum()),
        n_intervals_in_exposed_patient_clusters=int(sizes.sum()),
        min_intervals_per_exposed_patient=sizes.min() if len(sizes) else np.nan,
        median_intervals_per_exposed_patient=sizes.median() if len(sizes) else np.nan,
        max_intervals_per_exposed_patient=sizes.max() if len(sizes) else np.nan,
        max_exposed_patient_interval_share=sizes.max()/sizes.sum() if len(sizes) else np.nan,
        n_loo_deletions_evaluated=len(deletions), n_loo_deletions_below_support=below,
        min_exposed_after_deletion=minimum,
        min_intervals_after_deletion=min((d[1] for d in deletions), default=np.nan),
        deletion_support_status="NOT_ESTIMABLE_AFTER_PATIENT_DELETION" if below else (
            "SUPPORT_REMAINS" if deletions else "NOT_AVAILABLE"),
        fragility_status=status, support_guard_passed=bool(metadata["supported_for_model"]),
        n_unique_predictor_levels=counts["n_unique_predictor_levels"],
        selected_adjustment_level=metadata["adjustment_status"],
        interpretation_note="Exploratory association; scarce exposure may limit support. No causal or protective interpretation; no deletion refits.",
        disclosure_status="PROTECTED_PENDING_INSTITUTIONAL_REVIEW",
        input_origin=input_origin, rule_version=RULE_VERSION)


class InterpretabilityAudit:
    """Observe prepared designs once; retain aggregates only."""
    def __init__(self, cohort, intervals, decision, step14, minimum_events, input_origin):
        self.cohort, self.intervals, self.decision = cohort, intervals, decision
        self.step14, self.minimum_events, self.input_origin = step14, minimum_events, input_origin
        self.pns, self.domains = [], []

    def observe(self, task, data, meta, result, population):
        if (task.spec.name == "domain_pns" and task.analysis_type == "timevarying"
                and task.outcome == "subsequent_essdai"
                and task.sensitivity in {"none", "official_only:none"}):
            row = pns_support_summary(task, data, meta, result, population,
                self.step14, self.minimum_events, self.input_origin)
            sources = self.cohort.set_index("patient_id").index_source
            selected = data.patient_id.map(sources)
            row.update(n_complete_case_official_patients=data.loc[selected.eq(v2.OFFICIAL), "patient_id"].nunique(),
                       n_complete_case_alternative_patients=data.loc[selected.eq(v2.FALLBACK), "patient_id"].nunique(),
                       n_complete_case_official_intervals=int(selected.eq(v2.OFFICIAL).sum()),
                       n_complete_case_alternative_intervals=int(selected.eq(v2.FALLBACK).sum()))
            self.pns.append(row)
        if task.spec.family != "essdai_domain" or task.analysis_type != "timevarying" or task.outcome != "subsequent_essdai":
            return
        if task.sensitivity not in {self.step14.LEAVE_SOURCE_SENSITIVITY,
                                    "official_only:"+self.step14.LEAVE_SOURCE_SENSITIVITY}:
            return
        domain = task.spec.name.removeprefix("domain_")
        for source in ("all", v2.OFFICIAL, v2.FALLBACK):
            candidates = self.intervals.loc[self.intervals.outcome_evaluable.eq(True)]
            if population == "official_only":
                candidates = candidates.loc[candidates.index_source.eq(v2.OFFICIAL)]
            if source != "all":
                candidates = candidates.loc[candidates.index_source.eq(source)]
            eligible = candidates.loc[candidates["composition_pair_gate_"+domain].eq("compatible_verified")]
            fitted = data if source == "all" else data.loc[data.patient_id.isin(
                self.cohort.loc[self.cohort.index_source.eq(source), "patient_id"])]
            verified = self.decision["status"] == "VERIFIED_AUTHORIZATION"
            estimable = verified and result["valid_for_inference"] and not fitted.empty
            label = ("OTHER_DOMAINS_SENSITIVITY_ESTIMABLE" if estimable else
                     "OTHER_DOMAINS_NOT_ESTIMABLE" if not verified or eligible.empty else
                     "INSUFFICIENT_MODEL_SUPPORT")
            self.domains.append(dict(domain=domain, index_source=source, population=population,
                analysis_type=task.analysis_type, outcome=task.outcome,
                n_candidate_intervals=len(candidates), n_eligible_composition_intervals=len(eligible),
                n_patients_eligible=eligible.patient_id.nunique(), n_modelable_intervals=len(fitted),
                n_modelable_patients=fitted.patient_id.nunique(),
                version_authorization_status=self.decision["status"],
                gate_reason="; ".join(sorted(candidates["composition_pair_gate_"+domain].dropna().unique())),
                model_status=result["model_status"],
                # The existing fit loop asserts paired sample hashes and freezes adjustments.
                n_supporting_matched_intervals=len(fitted) if task.comparison_role else 0,
                n_supporting_matched_patients=fitted.patient_id.nunique() if task.comparison_role else 0,
                matched_sample_hash=meta["matched_sample_hash"] if source == "all" else "",
                interpretation_label=label, input_origin=self.input_origin, rule_version=RULE_VERSION))

    def outputs(self):
        if not self.pns:
            self.pns.append(dict(analysis_family=FAMILY, population="inclusive_primary",
                predictor="domain_pns", outcome="subsequent_essdai", model_status="NOT_AVAILABLE",
                fragility_status="NOT_ESTIMABLE_OR_NOT_AVAILABLE", input_origin=self.input_origin,
                rule_version=RULE_VERSION, disclosure_status="PROTECTED_PENDING_INSTITUTIONAL_REVIEW"))
        return dict(pns_fragility_qc=pd.DataFrame(self.pns),
                    domain_interpretability_qc=pd.DataFrame(self.domains))

    def safe_summary(self):
        primary = next((r for r in self.pns if r["population"] == "inclusive_primary"), None)
        status = primary["fragility_status"] if primary else "NOT_ESTIMABLE_OR_NOT_AVAILABLE"
        return ("PNS is exploratory. Support diagnostic: "+status+
                ". Scarce exposure can make patient deletion invalidate support. "
                "No deletion refits or causal/protective interpretation. Detailed aggregate QC remains "
                "protected pending institutional disclosure review; clinical review is independent.")

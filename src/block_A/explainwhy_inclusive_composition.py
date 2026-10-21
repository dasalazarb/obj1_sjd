"""Fail-closed composition certification and aggregate observability.

An upstream authorization must bind field semantics, scoring weights and an
explicit compatibility matrix to physical source/producer hashes. Arithmetic
alone, a registry role, or equal version strings cannot authorize inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

QC_RULE_VERSION = "inclusive-composition-provenance-v1"
UPSTREAM_REVIEW = {
    "repository": "dasalazarb/eda_sjd",
    "commit": "ffe59ecc5711b9e2e0bf4e611e32e8feffc5e2b4",
    "path": "src/09d_finalize_clinical_episode_dataset.py",
    "git_blob_sha": "eeb9467e602612c85d8e990e2e19c38728eaeee4",
    "lines": "588-642",
    "finding": "_essdai_version is overwritten per resolved variable with form source r/legacy; total_version is total form source, not an all-domain scoring certificate",
    "scope": "source-code review only; physical input producer/hash linkage still required",
}
FIELD_SEMANTICS = {
    "_essdai_version": "last resolved ESSDAI variable form source; not all-domain scoring algorithm",
    "_essdai_total_version": "recorded total form source; clinical scoring equivalence unverified",
    "_essdai_domain_version": "no all-domain scoring producer established by reviewed source",
}
VERSION_FIELDS = ("_essdai_version", "_essdai_total_version", "_essdai_domain_version",
                  "essdai_total_source", "_essdai_resolution_method", "source_protocol")
FAILURES = ("missing_total_or_domain", "incomplete_domain_panel", "nonconcordant_total",
            "missing_version_evidence", "incompatible_scoring_versions",
            "producer_identity_unverified", "compatible_verified")


def side_field(side, field):
    return side+field if field.startswith("_") else side+"_"+field


def load_authorization(producer_manifest, paths, root, step14):
    """Only an explicitly authorized, hash-bound upstream contract can certify.

    The consumer does not discover/trust arbitrary sidecars. External recovery
    requires its own producer regeneration; it is never merged opportunistically.
    """
    from src.block_A.explainwhy_inclusive import sha256
    decision = dict(status="UNVERIFIED", reason="no_authorized_scoring_contract",
        authorization_source="Step 13 v2 producer manifest: composition_provenance",
        evidence_files=[], compatibility_rule=[], author=None, protocol=None,
        upstream_source_review=UPSTREAM_REVIEW,
        limitations="Arithmetic concordance is diagnostic only; no scoring version inferred.")
    contract = producer_manifest.get("composition_provenance")
    if contract is None:
        return None, decision
    required = {"author", "protocol", "source_role", "source_sha256", "producer_path",
        "producer_sha256", "producer_commit", "domain_version_field", "total_version_field",
        "domain_semantics", "total_semantics", "temporal_policy", "weights", "compatible_versions"}
    if not required.issubset(contract) or any(not contract[k] for k in required):
        raise ValueError("composition_provenance: incomplete upstream authorization")
    if (contract["source_role"] not in {"step10", "context"} or
            contract["domain_version_field"] not in {"_essdai_version", "_essdai_domain_version"} or
            contract["total_version_field"] != "_essdai_total_version" or
            contract["domain_semantics"] != "all_12_domain_scoring_algorithm" or
            contract["total_semantics"] != "recorded_total_scoring_algorithm" or
            contract["temporal_policy"] != "episode_as_of_no_future_information" or
            contract["weights"] != step14.canonical_domain_weights()):
        raise ValueError("composition_provenance: incompatible semantics, weights or temporal policy")
    source = paths[contract["source_role"]]
    producer = (Path(root)/contract["producer_path"]).resolve()
    if not producer.is_relative_to(Path(root).resolve()) or not producer.is_file():
        raise ValueError("composition_provenance: producer path unavailable or outside repository")
    if (sha256(source) != contract["source_sha256"] or
            sha256(producer) != contract["producer_sha256"] or
            producer_manifest.get("source_hashes", {}).get(contract["producer_path"]) != contract["producer_sha256"] or
            contract["producer_commit"] != producer_manifest.get("code_commit")):
        raise ValueError("composition_provenance: source/producer hash or commit mismatch")
    matrix = contract["compatible_versions"]
    if not isinstance(matrix, list) or not matrix or any(
            set(r) != {"domain_version", "total_version", "scoring_system"} or
            not all(isinstance(v, str) and v.strip() for v in r.values()) for r in matrix):
        raise ValueError("composition_provenance: explicit compatibility matrix required")
    pairs = [(r["domain_version"], r["total_version"]) for r in matrix]
    if len(pairs) != len(set(pairs)):
        raise ValueError("composition_provenance: ambiguous compatibility matrix")
    policy = dict(contract)
    policy["source_fields"] = list(step14.read_table(source).columns)
    decision.update(status="VERIFIED_AUTHORIZATION", reason="hash_bound_upstream_scoring_contract",
        author=contract["author"], protocol=contract["protocol"], compatibility_rule=matrix,
        evidence_files=[dict(path=str(source), sha256=sha256(source)),
                        dict(path=str(producer), sha256=sha256(producer))],
        producer_commit=contract["producer_commit"], source_role=contract["source_role"])
    return policy, decision


def certify_versions(frame, side, policy=None):
    """Return episode certification; absent evidence and unknown identity differ."""
    def strings(field):
        return frame.get(side_field(side, field), pd.Series(pd.NA, index=frame.index)).astype("string").str.strip().replace("", pd.NA)
    field = policy["domain_version_field"] if policy else "_essdai_version"
    domain, total = strings(field), strings("_essdai_total_version")
    # Dedicated domain evidence remains visible even without an authorization.
    if policy is None:
        domain = domain.fillna(strings("_essdai_domain_version"))
    status = pd.Series("missing_version_evidence", index=frame.index, dtype="string")
    present = domain.notna() & total.notna()
    status.loc[present] = "producer_identity_unverified"
    system = pd.Series(pd.NA, index=frame.index, dtype="string")
    if policy is not None:
        bound = {field, "_essdai_total_version"}.issubset(policy["source_fields"])
        if bound:
            status.loc[present] = "incompatible_scoring_versions"
            for entry in policy["compatible_versions"]:
                match = present & domain.eq(entry["domain_version"]) & total.eq(entry["total_version"])
                status.loc[match] = "compatible_verified"
                system.loc[match] = entry["scoring_system"]
    return status, system


def apply_composition_gate(frame, composition, domains, policy=None):
    out = frame.copy()
    for side in ("from", "to"):
        status, system = certify_versions(out, side, policy)
        out[side+"_composition_version_status"] = status
        out[side+"_composition_scoring_system"] = system
    both = out.from_composition_version_status.eq("compatible_verified") & out.to_composition_version_status.eq("compatible_verified")
    incompatible_pair = both & out.from_composition_scoring_system.ne(out.to_composition_scoring_system)
    for domain in domains:
        categories = []
        for side in ("from", "to"):
            category = out[side+"_composition_version_status"].copy()
            category.loc[incompatible_pair] = "incompatible_scoring_versions"
            category.loc[~out[side+"_composition_concordant"]] = "nonconcordant_total"
            category.loc[~out[side+"_complete_domain_panel"]] = "incomplete_domain_panel"
            missing = out[side+"_essdai"].isna() | out[side+"_domain_"+domain].isna()
            category.loc[missing] = "missing_total_or_domain"
            out[side+"_composition_gate_"+domain] = category
            categories.append(category)
            valid = category.eq("compatible_verified")
            col = side+"_essdai_other_domains_"+domain
            out[col] = out[col].where(valid)
            mask = composition.side.eq(side) & composition.domain.eq(domain)
            composition.loc[mask, "n_post_version_eligible"] = int(valid.sum())
            composition.loc[mask, "n_missing_version_evidence"] = int(category.eq("missing_version_evidence").sum())
            composition.loc[mask, "n_producer_identity_unverified"] = int(category.eq("producer_identity_unverified").sum())
            composition.loc[mask, "n_incompatible_scoring_versions"] = int(category.eq("incompatible_scoring_versions").sum())
        pair = pd.Series("compatible_verified", index=out.index, dtype="string")
        for reason in reversed(FAILURES[:-1]):
            pair.loc[categories[0].eq(reason) | categories[1].eq(reason)] = reason
        out["composition_pair_gate_"+domain] = pair
        composition.loc[composition.domain.eq(domain), "n_pair_post_version_eligible"] = int(pair.eq("compatible_verified").sum())
    composition["provenance_authorization_present"] = policy is not None
    composition["legacy_manifest_flag_scope"] = "historical hardcoded flag; not a disk inventory"
    composition["status"] = np.where(composition.n_post_version_eligible.gt(0), "PASS", "NOT_ESTIMABLE")
    return out, composition


def source_audit(paths, registry, intervals, step14, decision):
    """Inventory physical fields and source code without exporting row values."""
    from src.block_A.explainwhy_inclusive import sha256
    rows = []
    fields = list(VERSION_FIELDS)+["essdai__total"]
    domain_fields = {s.time_varying_column: s.name.removeprefix("domain_")
        for s in step14.build_predictor_registry() if s.family == "essdai_domain"}
    fields += list(domain_fields)
    for role in ("step10", "context", "step11"):
        path = paths[role]
        frame = step14.read_table(path)
        for field in dict.fromkeys(fields):
            found = field in frame
            registered = pd.Series(False, index=registry.index)
            for col in ("public_variable", "variable", "original_variable"):
                if col in registry:
                    registered |= registry[col].eq(field).fillna(False)
            entry = registry.loc[registered]
            def values(col):
                return " | ".join(sorted(entry[col].dropna().astype(str).unique())) if col in entry else ""
            coverage = {}
            # Values are already joined with collision-checked canonical keys.
            for side in ("from", "to"):
                col = (side+"_essdai" if field == "essdai__total" else
                       side+"_domain_"+domain_fields[field] if field in domain_fields else side_field(side, field))
                coverage["n_"+side+"_interval_values"] = int(intervals[col].notna().sum()) if col in intervals and found else 0
            rows.append(dict(source_role=role, source_path=str(path), source_sha256=sha256(path), field=field,
                exists=found, n_nonnull=int(frame[field].notna().sum()) if found else 0,
                dtype=str(frame[field].dtype) if found else "MISSING", registry_role=values("role"),
                producer=values("producer_script"), temporal_policy=values("temporal_scope"),
                key_policy="patient_id + clinical_episode_id; collision-checked one_to_one",
                semantic_scope=FIELD_SEMANTICS.get(field, "unverified" if field in VERSION_FIELDS else "observed analytic value"),
                valid_states="documented scoring-algorithm tokens required" if "version" in field else "producer contract",
                version_decision=decision["status"], status="PRESENT" if found else "MISSING", **coverage))
    for role, path in paths.items():
        if role in {"registry", "name_mapping"}:
            rows.append(dict(source_role=role, source_path=str(path), exists=path.is_file(),
                source_sha256=sha256(path) if path.is_file() else None,
                status="PRESENT" if path.is_file() else "MISSING"))
    for rel in ("src/00_build_visit_spine.py", "src/block_A/01_essdai_esspri.py", "src/block_A/06_overlap_glandular.py",
                "src/block_A/10_build_integrated_longitudinal_dataset.py"):
        path = step14.ROOT/rel
        rows.append(dict(source_role="producer_code", source_path=str(path), exists=path.is_file(),
            source_sha256=sha256(path) if path.is_file() else None,
            producer="source-code inventory; scoring-version semantics not certified",
            status="PRESENT" if path.is_file() else "MISSING"))
    return pd.DataFrame(rows)


def interval_gate_qc(intervals, integrated, cohort, domains, risks=None):
    """Aggregate Boolean gates without repeatedly copying the wide clinical frame."""
    from src.block_A import observed_patterns_v2 as v2
    rows = [dict(index_source="all", stage="step10_full_unfiltered_spine", side="both", domain="all",
        n_patients=integrated.patient_id.nunique(), n_episodes=len(integrated), n_intervals=np.nan, status="PASS")]
    fallback_ids = set(cohort.loc[cohort.index_source.eq(v2.FALLBACK), "patient_id"])
    predictor_columns = [c for c in intervals if c.startswith("pred__")]
    for source in ("all", v2.OFFICIAL, v2.FALLBACK):
        eligible = cohort.loc[cohort.eligible & (True if source == "all" else cohort.index_source.eq(source))]
        g = intervals.loc[intervals.patient_id.isin(eligible.patient_id)]
        pid = g.patient_id
        fallback = pid.isin(fallback_ids)
        available = g[predictor_columns].notna().any(axis=1) if predictor_columns else pd.Series(False, index=g.index)
        event = g.event_first_delta3.eq(1)
        observed = risks.loc[risks.patient_id.isin(eligible.patient_id)] if risks is not None else None
        n_events = int(observed.event.sum()) if observed is not None else np.nan
        n_fallback_events = int(observed.loc[observed.patient_id.isin(fallback_ids), "event"].sum()) if observed is not None else np.nan
        def count(mask, **labels):
            rows.append(dict(index_source=source, n_patients=pid.loc[mask].nunique(), n_intervals=int(mask.sum()),
                n_intervals_any_predictor=int((mask & available).sum()), n_modelable_event_intervals=int((mask & event).sum()),
                n_observed_events=n_events,
                n_fallback_patients=pid.loc[mask & fallback].nunique(), n_fallback_intervals=int((mask & fallback).sum()),
                n_fallback_intervals_any_predictor=int((mask & fallback & available).sum()),
                n_fallback_modelable_event_intervals=int((mask & fallback & event).sum()),
                n_fallback_observed_events=n_fallback_events, status="PASS", **labels))
        spine = integrated.loc[integrated.patient_id.isin(eligible.patient_id)]
        rows.append(dict(index_source=source, stage="full_episode_spine", side="both", domain="all",
            n_patients=spine.patient_id.nunique(), n_episodes=len(spine), n_intervals=np.nan, status="PASS"))
        all_intervals = pd.Series(True, index=g.index)
        for domain in domains:
            for side in ("from", "to", "both"):
                count(all_intervals, domain=domain, side=side, stage="adjacent_post_index_intervals")
                selected = g.outcome_evaluable.astype(bool)
                count(selected, domain=domain, side=side, stage="selected_endpoints_outcome_evaluable")
                sides = ("from", "to") if side == "both" else (side,)
                for stage, suffix in (("complete_panels", "_complete_domain_panel"),
                                      ("arithmetic_concordance", "_composition_concordant")):
                    for endpoint in sides:
                        selected = selected & g[endpoint+suffix]
                    count(selected, domain=domain, side=side, stage=stage)
                gate = g["composition_pair_gate_"+domain if side == "both" else side+"_composition_gate_"+domain]
                count(selected & gate.eq("compatible_verified"), domain=domain, side=side, stage="verified_source_version_outcome")
                for reason in FAILURES:
                    count(gate.eq(reason), domain=domain, side=side, stage="exclusive_failure_category", category=reason)
    return pd.DataFrame(rows)


def leave_source_support(results, intervals):
    """Actual fitted counts from estimator rows, pre-fit gates by index source."""
    from src.block_A import observed_patterns_v2 as v2
    rows = []
    for result in results.loc[results.sensitivity.str.contains("leave_source_domain_outcome", regex=False)].to_dict("records"):
        domain = result["predictor"].removeprefix("domain_")
        for source in ("all", v2.OFFICIAL, v2.FALLBACK):
            g = intervals.loc[intervals.outcome_evaluable]
            if result["population"] == "official_only":
                g = g.loc[g.index_source.eq(v2.OFFICIAL)]
            if source != "all":
                g = g.loc[g.index_source.eq(source)]
            valid = g["composition_pair_gate_"+domain].eq("compatible_verified")
            pred = g.get("pred__"+result["predictor"], pd.Series(np.nan, index=g.index)).notna()
            fitted = result["n_fitted"] if source == "all" else (
                result["n_fitted_fallback"] if source == v2.FALLBACK else result["n_fitted"]-result["n_fitted_fallback"])
            reasons = g.loc[~valid, "composition_pair_gate_"+domain].value_counts().to_dict()
            rows.append(dict(domain=domain, index_source=source, population=result["population"],
                analysis_type=result["analysis_type"], outcome=result["outcome"], sensitivity=result["sensitivity"],
                n_candidate_patients=g.patient_id.nunique(), n_candidate_intervals=len(g),
                n_verified_patients=g.loc[valid, "patient_id"].nunique(),
                n_verified_predictor_patients=g.loc[valid & pred, "patient_id"].nunique(),
                n_verified_intervals=int(valid.sum()), n_verified_predictor_intervals=int((valid & pred).sum()),
                n_fitted=fitted, n_model_intervals=result["n_intervals"] if source == "all" else np.nan,
                n_complete_cases=result["n_complete_cases"] if source == "all" else np.nan,
                formula=result["formula"], adjustment=result["selected_adjustment_level"],
                model_status=result["model_status"], support_reason=result["support_reason"],
                failure_counts=json.dumps(reasons, sort_keys=True),
                status="PASS" if result["valid_for_inference"] else "NOT_ESTIMABLE"))
    return pd.DataFrame(rows)

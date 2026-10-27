"""Read-only review of fixed ESSDAI v2 intervals and medication documentation.

All returned tables are restricted row-level research data. No function estimates
exposure, treatment efficacy, or unobserved activity between clinical visits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.block_A.secondary_trajectory_analysis import (
    ContractError, KEYS, dates, identities, require, sha256,
)
from src.block_A._internal.medications import contracts
from src.block_A._internal.medications.dates import exact_day, parse_date, present
from src.block_A._internal.medications.documentation import pseudonym
from src.block_A._internal.medications.persistence import manifest_path, publication_lock
from src.block_A._internal.step10_medications import load_contract

DOMAINS = ("constitutional", "lymphadenopathy", "glandular", "articular",
           "cutaneous", "pulmonary", "renal", "muscular", "pns", "cns",
           "hematologic", "biological")
NAME_COLUMNS = contracts.COLUMNS
NOTICE = ("Retrospective, descriptive chart review. Documentation is not verified "
          "exposure. Orders do not establish administration, dispensing or adherence; "
          "order_end is not a confirmed discontinuation. Missing is not untreated. "
          "CTDB reports apply only to their original visit. No causal attribution.")
CATALOG = "08_medication_documentation_long.parquet"
LINKS = "08_medication_candidate_episode_links.parquet"
REVIEW_FILES = ("08_medication_unassigned.parquet", "08_medication_unresolved_name_review.parquet")


def medication_bundle(features_path, audit_root, spine_path, integrated, key_path, cutoff):
    """Authenticate the production generation and reconcile HMACs to Step00 IDs.

    The established key is used only for equality checks in memory. Neither the
    key nor a raw/HMAC crosswalk is returned, written, or logged.
    """
    features_path, audit_root, spine_path = map(Path, (features_path, audit_root, spine_path))
    restricted = audit_root / "restricted"
    required = [features_path, manifest_path(features_path), spine_path, Path(key_path),
                restricted / "08_key_compatibility.json", features_path.with_suffix(".lock")]
    if any(not p.is_file() for p in required):
        raise ContractError("BLOCKED_UPSTREAM: missing Step08 identity/publication inputs")
    with publication_lock(features_path):
        spine = identities(pd.read_parquet(spine_path), KEYS, "Step00")
        features, _ = load_contract(features_path, spine, spine_path)
        metadata = json.loads(manifest_path(features_path).read_text())
        generation = metadata.get("restricted_generation", "")
        if not generation or Path(generation).name != generation or generation in (".", ".."):
            raise ContractError("invalid Step08 restricted generation")
        directory = restricted / "generations" / generation
        if (not directory.is_dir() or directory.is_symlink()
                or not directory.resolve().is_relative_to(restricted.resolve())
                or (restricted / "current").resolve() != directory.resolve()):
            raise ContractError("Step08 restricted current/generation mismatch")
        hashes = metadata.get("restricted_outputs_sha256", {})
        if not {CATALOG, LINKS, *REVIEW_FILES, "08_medication_reconciliation_qc.json"}.issubset(hashes):
            raise ContractError("Step08 restricted artifacts are not authenticated")
        paths = {}
        for name, digest in hashes.items():
            target = directory / name
            if Path(name).name != name or target.is_symlink() or not target.is_file() or sha256(target) != digest:
                raise ContractError("Step08 restricted artifact hash/path mismatch")
            paths[name] = target
        qc = json.loads(paths["08_medication_reconciliation_qc.json"].read_text())
        if qc != metadata.get("qc") or qc.get("confirmed_medication_exposure") is not False:
            raise ContractError("Step08 QC/manifest mismatch")
        secret = Path(key_path).read_bytes()
        if len(secret) < 32 or json.loads((restricted / "08_key_compatibility.json").read_text()).get("scope") != pseudonym(secret, "compatibility", "08-medications"):
            raise ContractError("Step08 HMAC scope mismatch")
        x = identities(integrated, KEYS, "Step10")
        if contracts.key_set(x) != contracts.key_set(spine):
            raise ContractError("Step00/10 episode identity mismatch")
        aligned = x.set_index(KEYS).reindex(pd.MultiIndex.from_frame(spine[KEYS]))
        for col in contracts.STRUCTURE[2:]:
            require(x, [col], "Step10 structural identity")
            left, right = spine[col].reset_index(drop=True), aligned[col].reset_index(drop=True)
            if col.endswith("_date"):
                left, right = dates(left, "Step00 dates", cutoff), dates(right, "Step10 dates", cutoff)
            if not (left.eq(right) | (left.isna() & right.isna())).fillna(False).all():
                raise ContractError("Step00/10 structural identity mismatch")
        # Step10 may omit descriptive medication fields; when present they must
        # be exactly the authenticated Step08 product, including nulls.
        f = features.set_index(KEYS).reindex(pd.MultiIndex.from_frame(x[KEYS]))
        for col in set(NAME_COLUMNS) & set(x):
            a, b = x[col].reset_index(drop=True), f[col].reset_index(drop=True)
            if not (a.eq(b) | (a.isna() & b.isna())).fillna(False).all():
                raise ContractError("Step08/10 medication names mismatch")
        catalog = identities(pd.read_parquet(paths[CATALOG]),
                             ["documentation_record_id", "linked_patient_pseudonymous"], "Step08 catalog")
        require(catalog, ["source_kind", "canonical_medication_name", "medication_name", "qc_flags",
                          "name_requires_review", "original_episode_id", "visit_documentation_date",
                          "order_start_raw", "order_end_raw", "initial_consent_raw"], "Step08 catalog")
        if catalog.documentation_record_id.duplicated().any() or not catalog.source_kind.isin(["CTDB", "BTRIS"]).all():
            raise ContractError("invalid Step08 document identity/source")
        mapping = {pseudonym(secret, "patient", p): p for p in spine.patient_id.unique()}
        if not catalog.linked_patient_pseudonymous.isin(mapping).all():
            raise ContractError("Step08 catalog patient HMAC cannot be reconciled to Step00/10")
        catalog["patient_id"] = catalog.linked_patient_pseudonymous.map(mapping).astype("string")
        links = identities(pd.read_parquet(paths[LINKS]), KEYS + ["documentation_record_id"], "Step08 links")
        require(links, ["selected_reference_kind", "selected_date", "evidence_level", "source_kind",
                        "canonical_medication_name"], "Step08 links")
        if links.duplicated(KEYS + ["documentation_record_id"]).any():
            raise ContractError("duplicate Step08 document/episode links")
        if not set(links[KEYS].itertuples(index=False, name=None)).issubset(contracts.key_set(spine)):
            raise ContractError("Step08 link episode mismatch")
        lookup = catalog.set_index("documentation_record_id")
        for link in links.to_dict("records"):
            if link["documentation_record_id"] not in lookup.index:
                raise ContractError("Step08 link absent from catalog")
            doc = lookup.loc[link["documentation_record_id"]]
            if any(link[c] != doc[c] for c in ("patient_id", "source_kind", "canonical_medication_name")):
                raise ContractError("Step08 document/link identity mismatch")
            kind = link["selected_reference_kind"]
            if kind not in ("ctdb_visit", "order_start", "order_end", "initial_consent"):
                raise ContractError("unknown Step08 date reference")
            if (kind == "ctdb_visit") != (doc.source_kind == "CTDB"):
                raise ContractError("Step08 source/reference mismatch")
            raw = doc.visit_documentation_date if kind == "ctdb_visit" else doc[kind + "_raw"]
            observed, selected = exact_day(raw), exact_day(link["selected_date"])
            # An original CTDB report can legitimately be undated. It stays
            # undated in review instead of borrowing the clinical anchor.
            if not ((pd.isna(observed) and pd.isna(selected)) or observed == selected):
                raise ContractError("Step08 selected date differs from original source")
            if pd.notna(selected) and selected > pd.Timestamp(cutoff):
                raise ContractError("Step08 selected date after data cutoff")
            expected = "reported_at_original_visit" if kind == "ctdb_visit" else "consent_only_review" if kind == "initial_consent" else "order_date_proximity"
            if link["evidence_level"] != expected or (kind == "ctdb_visit" and link["clinical_episode_id"] != doc.original_episode_id):
                raise ContractError("Step08 original visit/evidence mismatch")
        # Review queues must refer back to this generation's same documents.
        for name in REVIEW_FILES:
            queue = pd.read_parquet(paths[name])
            require(queue, ["documentation_record_id"], "Step08 review queue")
            if not queue.documentation_record_id.isin(lookup.index).all():
                raise ContractError("Step08 review queue/catalog mismatch")
        inputs = {"step08_features": features_path, "step08_manifest": manifest_path(features_path),
                  "step00_spine": spine_path, "step08_scope": restricted / "08_key_compatibility.json",
                  **{"step08_" + name: p for name, p in paths.items()}}
        fingerprints = {role: dict(path=str(p.resolve()), sha256=sha256(p)) for role, p in inputs.items()}
    return features, catalog, links, metadata, fingerprints


def selected_visits(x, patterns, sequence, k, cutoff):
    roster = patterns.loc[patterns.threshold_abs.eq(k) & patterns.observed_pattern.eq("intermittent_observed")
                          & patterns.eligibility_status.eq("longitudinally_evaluable")].copy()
    if roster.n_distinct_essdai_dates.lt(3).any():
        raise ContractError("intermittent groups require at least three dates")
    s = identities(sequence, KEYS, "selected dates")
    require(s, ["clinical_anchor_date", "essdai__total", "essdai_index_date"], "selected dates")
    s["clinical_anchor_date"] = dates(s.clinical_anchor_date, "selected dates", cutoff)
    # Verify the producer's physical chronology before any downstream sorting.
    if any(not g.clinical_anchor_date.is_monotonic_increasing for _, g in s.groupby("patient_id", sort=False)):
        raise ContractError("selected dates are not in original chronological order")
    v = s.loc[s.patient_id.isin(roster.patient_id), KEYS + ["clinical_anchor_date", "essdai__total"]].merge(
        x.drop(columns=["clinical_anchor_date", "essdai__total"]), on=KEYS, validate="one_to_one")
    v = v.sort_values(["patient_id", "clinical_anchor_date"]).reset_index(drop=True)
    meta = roster.set_index("patient_id")
    for col in ("essdai_index_date", "essdai_index_score", "essdai_index_source"):
        require(roster, [col], "patterns")
        v[col] = v.patient_id.map(meta[col])
    numeric = pd.to_numeric(v["essdai__total"], errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all() or numeric.lt(0).any():
        raise ContractError("selected ESSDAI contains invalid scores")
    for pid, g in v.groupby("patient_id"):
        if len(g) != meta.loc[pid, "n_distinct_essdai_dates"] or g.iloc[0]["essdai__total"] != meta.loc[pid, "essdai_index_score"]:
            raise ContractError("selected support/index score mismatch")
    v["threshold_abs"] = k
    v["years_from_essdai_index"] = (v.clinical_anchor_date - pd.to_datetime(v.essdai_index_date)).dt.days / 365.25
    return roster, v


def direction(before, after):
    if pd.isna(before) or pd.isna(after):
        return "not_comparable"
    return "increased" if after > before else "decreased" if after < before else "unchanged_documented"


def intervals(visits, k, gap_days=365, cutoff=None):
    jumps, domains, context = [], [], []
    for pid, g in visits.groupby("patient_id", sort=False):
        rows = g.to_dict("records")
        for ordinal, (a, b) in enumerate(zip(rows, rows[1:]), 1):
            jump_id = f"interval_{ordinal:04d}"
            da, db = a["essdai__total"] - a["essdai_index_score"], b["essdai__total"] - a["essdai_index_score"]
            delta = b["essdai__total"] - a["essdai__total"]
            row = dict(patient_id=pid, threshold_abs=k, jump_id=jump_id,
                       from_episode_id=a["clinical_episode_id"], to_episode_id=b["clinical_episode_id"],
                       from_date=a["clinical_anchor_date"], to_date=b["clinical_anchor_date"],
                       days_between=(b["clinical_anchor_date"]-a["clinical_anchor_date"]).days,
                       years_from_essdai_index_before=a["years_from_essdai_index"],
                       years_from_essdai_index_after=b["years_from_essdai_index"],
                       essdai_from=a["essdai__total"], essdai_to=b["essdai__total"],
                       delta_essdai_interval=delta, delta_from_index_before=da, delta_from_index_after=db,
                       jump_direction="improvement" if delta < 0 else "worsening" if delta > 0 else "no_difference",
                       abs_change_ge_k_in_interval=abs(delta) >= k,
                       threshold_up_entered=da < k <= db, threshold_up_exited=db < k <= da,
                       threshold_down_entered=da > -k >= db, threshold_down_exited=db > -k >= da,
                       sign_reversal_from_index=da * db < 0, from_essdai_index_source=a["essdai_index_source"])
            row["measurement_gap_warning"] = row["days_between"] > gap_days
            row["explains_observed_pattern"] = any(row[c] for c in ("threshold_up_entered", "threshold_up_exited", "threshold_down_entered", "threshold_down_exited", "sign_reversal_from_index"))
            available = False
            for domain in DOMAINS:
                col = f"essdai__{domain}_ordinal_score"
                before, after = a.get(col, np.nan), b.get(col, np.nan)
                comparable = pd.notna(before) and pd.notna(after)
                available |= comparable
                domains.append(dict(patient_id=pid, threshold_abs=k, jump_id=jump_id, domain=domain,
                    from_date=row["from_date"], to_date=row["to_date"], ordinal_before=before, ordinal_after=after,
                    direction=direction(before, after), missing_before=pd.isna(before), missing_after=pd.isna(after),
                    scoring_comparability="same_upstream_ordinal_field" if comparable else "not_comparable_missing",
                    weighted_contribution_available=False, weighted_point_change=np.nan))
            row["clinical_domain_information_available"] = available
            jumps.append(row)
            # Context retains original measurement dates; no borrowed dates for labs.
            for endpoint, original in (("from", a), ("to", b)):
                for col in visits.columns:
                    if not (col == "esspri__total" or (col.startswith("lab__") and col.endswith("__value"))):
                        continue
                    date_col = col.removesuffix("__value") + "__measurement_date"
                    when = original.get(date_col, original["clinical_anchor_date"] if col == "esspri__total" else pd.NaT)
                    day = exact_day(when)
                    context.append(dict(patient_id=pid, threshold_abs=k, jump_id=jump_id, endpoint=endpoint,
                        variable=col, value=original[col], measurement_date=day,
                        available=pd.notna(original[col]) and pd.notna(day) and (cutoff is None or day <= pd.Timestamp(cutoff)),
                        temporal_note="original_measurement_date_required; descriptive_only"))
    jump_cols = ["patient_id", "threshold_abs", "jump_id", "jump_direction", "from_episode_id", "to_episode_id", "from_date", "to_date"]
    return (pd.DataFrame(jumps) if jumps else pd.DataFrame(columns=jump_cols),
            pd.DataFrame(domains, columns=["patient_id", "threshold_abs", "jump_id", "domain", "from_date", "to_date",
                "ordinal_before", "ordinal_after", "direction", "missing_before", "missing_after", "scoring_comparability",
                "weighted_contribution_available", "weighted_point_change"]),
            pd.DataFrame(context, columns=["patient_id", "threshold_abs", "jump_id", "endpoint", "variable", "value", "measurement_date", "available", "temporal_note"]))


def documentary_history(catalog, links, patient_ids, cutoff):
    """Retain every source reference, including undated/unassigned documents.

    A document may have several references. Counts must use unique document IDs,
    never the number of references or links to clinical episodes.
    """
    selected = catalog.loc[catalog.patient_id.isin(patient_ids)]
    alternatives = {}
    for docid, g in links.groupby("documentation_record_id"):
        alternatives[docid] = json.dumps([{**r, "selected_date": str(r["selected_date"])} for r in
            g[["clinical_episode_id", "selected_reference_kind", "selected_date"]].to_dict("records")], sort_keys=True)
    events = []
    for doc in selected.to_dict("records"):
        ctdb = doc["source_kind"] == "CTDB"
        refs = [("ctdb_visit", doc["visit_documentation_date"])] if ctdb else [
            (kind, doc[kind + "_raw"]) for kind in ("order_start", "order_end", "initial_consent") if present(doc[kind + "_raw"])]
        if not refs:
            refs = [("unknown", "")]
        start, end = exact_day(doc.get("order_start_raw")), exact_day(doc.get("order_end_raw"))
        contradictory = pd.notna(start) and pd.notna(end) and end < start
        alts = alternatives.get(doc["documentation_record_id"], "[]")
        distinct_refs = {(r["selected_reference_kind"], r["selected_date"]) for r in json.loads(alts)}
        for kind, raw in refs:
            parsed, day = parse_date(raw), exact_day(raw)
            future = pd.notna(day) and day > pd.Timestamp(cutoff)
            consent = kind == "initial_consent"
            usable = pd.notna(day) and not consent and not future and not contradictory
            # Preserve a safe, explicit catalogue subset; source free text or
            # unrelated CTDB visit-level doses must never become a drug dose.
            row = {c: doc.get(c, pd.NA) for c in (
                "patient_id", "documentation_record_id", "source_kind", "source", "canonical_medication_name",
                "medication_name", "name_requires_review", "qc_flags", "clinical_category_primary",
                "clinical_category_detail", "classification_confidence", "route_canonical", "formulation",
                "original_episode_id", "indication", "dose_slot_mapping")}
            row.update(reference_kind=kind, date_raw=parsed["raw"], selected_date=day,
                date_precision=parsed["precision"], date_parse_status=parsed["status"],
                date_lower_bound=parsed["lower_bound"], date_upper_bound=parsed["upper_bound"],
                temporally_usable=usable, future_date=future, contradictory_dates=contradictory,
                selected_reference_alternatives=alts, selected_reference_discordance=len(distinct_refs) > 1,
                evidence_level="reported_at_original_visit" if ctdb else "consent_only_review" if consent else "order_date_proximity",
                confirmed_exposure="unknown", confirmed_exposure_event=pd.NA,
                dose_raw=doc.get("dose_raw", pd.NA) if not ctdb else pd.NA,
                dose_unit=doc.get("uom_raw", pd.NA) if not ctdb else pd.NA,
                indication_verified=False,
                temporal_status="consent_only" if consent else "dated_documentation" if usable else "date_unknown_or_partial")
            events.append(row)
    if not events:
        return pd.DataFrame(columns=["patient_id", "documentation_record_id", "source_kind", "canonical_medication_name",
            "medication_name", "reference_kind", "selected_date", "temporally_usable", "temporal_status",
            "name_requires_review", "selected_reference_discordance", "confirmed_exposure"])
    return pd.DataFrame(events).sort_values(["patient_id", "selected_date", "documentation_record_id", "reference_kind"], na_position="last").reset_index(drop=True)


def evidence_for_intervals(jumps, history, features, post_days=90):
    evidence, contrasts, enriched = [], [], []
    lookup = features.set_index(KEYS)
    for jump in jumps.to_dict("records"):
        pid = jump["patient_id"]
        docs = history.loc[history.patient_id.eq(pid)]
        from_names = lookup.loc[(pid, jump["from_episode_id"]), NAME_COLUMNS[0]]
        to_names = lookup.loc[(pid, jump["to_episode_id"]), NAME_COLUMNS[0]]
        left = set(from_names.split("; ")) if pd.notna(from_names) else set()
        right = set(to_names.split("; ")) if pd.notna(to_names) else set()
        names = sorted(left | right) or [pd.NA]
        for name in names:
            status = "not_documented_either" if pd.isna(name) else "reported_at_both" if name in left & right else "reported_only_at_from" if name in left else "reported_only_at_to"
            contrasts.append(dict(patient_id=pid, threshold_abs=jump["threshold_abs"], jump_id=jump["jump_id"],
                canonical_medication_name=name, report_contrast=status,
                from_list_documented=pd.notna(from_names), to_list_documented=pd.notna(to_names),
                confirmed_exposure="unknown", start_or_stop_inferred=False))
        inside = unresolved = post_order = False
        for event in docs.to_dict("records"):
            if event["temporal_status"] == "consent_only":
                relation = "consent_only"
                unresolved = True
            elif not event["temporally_usable"]:
                relation = "date_unknown_or_partial"
                unresolved = True
            elif event["selected_date"] <= jump["from_date"]:
                relation = "reported_or_ordered_before_from"
            elif event["selected_date"] <= jump["to_date"]:
                relation = "documented_in_interval"
                inside = True
            elif event["selected_date"] <= jump["to_date"] + pd.Timedelta(days=post_days):
                relation = "documented_after_to"
                post_order |= event["source_kind"] == "BTRIS" and jump["jump_direction"] == "worsening"
            else:
                continue
            evidence.append(dict(**event, threshold_abs=jump["threshold_abs"], jump_id=jump["jump_id"],
                                 temporal_relation=relation))
        jump.update(treatment_documentation_in_interval="yes" if inside else "unknown" if unresolved or docs.empty else "no",
            sjd_systemic_drug_documented_near_interval="unknown", indication_verification="not_established_by_step08",
            post_jump_order_possible_reverse_causation=post_order,
            temporal_status_unresolved=unresolved or docs.empty,
            high_priority_chart_review=inside or unresolved or bool(jump.get("abs_change_ge_k_in_interval"))
                or bool(jump.get("clinical_domain_information_available")) or len(jumps.loc[jumps.patient_id.eq(pid)]) > 1)
        enriched.append(jump)
    return (pd.DataFrame(enriched) if enriched else jumps.copy(),
            pd.DataFrame(evidence) if evidence else pd.DataFrame(columns=list(history.columns) + ["threshold_abs", "jump_id", "temporal_relation"]),
            pd.DataFrame(contrasts, columns=["patient_id", "threshold_abs", "jump_id", "canonical_medication_name", "report_contrast",
                "from_list_documented", "to_list_documented", "confirmed_exposure", "start_or_stop_inferred"]))


def cohort_summary(jumps, evidence, k, minimum_cell=5):
    """Controlled-access aggregates; suppress the whole row for sparse cells.

    Never pool thresholds or provide row totals/margins for suppressed cells.
    No table or plot here is approved for public release.
    """
    rows = []
    for direction_ in ("improvement", "worsening", "no_difference", "threshold_reversal"):
        subset = jumps.loc[jumps.jump_direction.eq(direction_)] if direction_ != "threshold_reversal" else jumps.loc[
            jumps.get("threshold_up_exited", pd.Series(False, index=jumps.index)) | jumps.get("threshold_down_exited", pd.Series(False, index=jumps.index))]
        keys = set(subset[["patient_id", "jump_id"]].itertuples(index=False, name=None))
        ev = evidence.loc[[key in keys for key in evidence[["patient_id", "jump_id"]].itertuples(index=False, name=None)]] if not evidence.empty else evidence
        for source in ("CTDB", "BTRIS"):
            for relation in ("reported_or_ordered_before_from", "documented_in_interval", "documented_after_to", "date_unknown_or_partial", "consent_only", "no_source_documentation_in_review_window"):
                selected = ev.loc[ev.source_kind.eq(source) & ev.temporal_relation.eq(relation)]
                hit = set(selected[["patient_id", "jump_id"]].itertuples(index=False, name=None))
                if relation == "no_source_documentation_in_review_window":
                    source_events = ev.loc[ev.source_kind.eq(source)]
                    hit = keys - set(source_events[["patient_id", "jump_id"]].itertuples(index=False, name=None))
                counts = [len(keys), subset.patient_id.nunique(), len(hit), len({p for p, _ in hit}),
                          selected[["patient_id", "documentation_record_id"]].drop_duplicates().shape[0], len(keys)-len(hit)]
                suppressed = any(0 < n < minimum_cell for n in counts)
                rows.append(dict(threshold_abs=k, jump_direction=direction_, source_kind=source, temporal_relation=relation,
                    n_intervals_total=None if suppressed else counts[0], n_patients_total=None if suppressed else counts[1],
                    n_intervals_documented=None if suppressed else counts[2], n_patients_documented=None if suppressed else counts[3],
                    n_documents_unique=None if suppressed else counts[4],
                    proportion_intervals=None if suppressed or not counts[0] else counts[2]/counts[0],
                    suppressed=suppressed, minimum_cell=minimum_cell, access="controlled_review_only"))
    return pd.DataFrame(rows)


def threshold_comparison(patterns):
    table = patterns.pivot(index="patient_id", columns="threshold_abs", values="observed_pattern").reindex(columns=[3, 5])
    table.columns = ["pattern_k3", "pattern_k5"]
    table["intermittent_k3"] = table.pattern_k3.eq("intermittent_observed")
    table["intermittent_k5"] = table.pattern_k5.eq("intermittent_observed")
    table["pattern_changed"] = table.pattern_k3.ne(table.pattern_k5)
    return table.loc[table.intermittent_k3 | table.intermittent_k5].reset_index()

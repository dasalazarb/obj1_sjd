"""Pure, representation-specific N3 audit of verified frozen inputs.

Candidate screening never changes historical feature roles or grants approval.
Only a complete human review backed by the caller's verified evidence catalog
can authorize a new analysis. Free review text and clinical dates are not exported.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd

from .s03_construct_validation import _dependencies, _paths
from .s03_profiles import observed_values
from .s03_release_reader import hard_members

VERSION = 'graph03_n3_audit_v1'
REVIEW_FIELDS = (
    'lineage_review_status', 'clinical_review_status', 'origin_review_status',
    'proxy_review_status', 'discovery_exclusion_review_status',
    'temporal_baseline_status', 'reviewer', 'source_reference', 'baseline_window',
)
INSPECTION_SOURCES = {
    'essdai__total', 'esspri__total', 'ext__biopsy_focus_score',
    'lab__ana_hep2_status__text', 'lab__rheumatoid_factor__value',
    'lab__igg_subclass_1__value', 'lab__crp_high_sensitivity__value', 'ovl__status',
}
KEYS = ['representation_id', 'source_column']
INTERNAL = dict(schema_version=VERSION, privacy_class='INTERNAL_RESTRICTED',
                shareable_export_allowed=False,
                analysis_population='covered_untied_supported_hard_assignment')
SUPPORT_COLUMNS = KEYS + [
    'tier_original', 'review_state', 'n_baseline_total', 'n_hard_supported',
    'n_observed_Y', 'n_missing_Y', 'fraction_observed_Y',
    'n_Y_nonmissing_with_all_covariates', 'n_excluded_missing_covariates',
    'n_positive', 'n_negative', 'n_tested', 'n_unknown', 'n_not_tested',
    'observation_denominator', 'coverage_by_community', 'availability_min',
    'availability_max', 'n_stable_hard', 'n_provisional_hard', 'missing_adjustments',
    'baseline_era_status', 'measurement_status', 'unit', 'type',
    'status', 'reason', *INTERNAL,
]
PROFILE_COLUMNS = KEYS + [
    'community_id', 'tier', 'unit', 'type', 'n_group', 'n_observed', 'n_missing',
    'median', 'q25', 'q75', 'n_positive', 'n_negative', 'proportion',
    'status', 'reason', 'uncertainty_scope', *INTERNAL,
]


def fingerprint(value):
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()


def _text(value):
    return value.strip() if isinstance(value, str) else ''


def verified_reference(record, evidence_catalog):
    """Catalog entries are created from actual repository files by the IO layer."""
    reference = _text(record.get('source_reference'))
    evidence = evidence_catalog.get(reference, {})
    return bool(reference and evidence.get('verified') is True
                and re.fullmatch(r'[0-9a-f]{64}', evidence.get('sha256', '')))


def scoped_records(config, key, identifier, source=None):
    records = config.get('n3_validation', {}).get(key, [])
    return [r for r in records if r.get('representation_id') == identifier
            and (source is None or r.get('source_column') == source)]


def _one(records):
    if len(records) > 1:
        raise ValueError('AMBIGUOUS_REPRESENTATION_SOURCE_POLICY')
    return records[0] if records else {}


def measurement_policy(config, identifier, source, catalog):
    policy = _one(scoped_records(config, 'measurement_policies', identifier, source))
    if policy.get('origin_review_status') != 'APPROVED' or not verified_reference(policy, catalog):
        return {}
    return policy


def values_for(series, declared_type='source_type', policy=None):
    """Preserve missing observation states; never infer a laboratory convention."""
    policy = policy or {}
    raw = series.copy()
    tokens = raw.astype('string').str.strip().str.lower()
    clean = observed_values(raw)
    convention = policy.get('category_convention', {})
    convention_verified = policy.get('category_review_status') == 'VERIFIED' and bool(convention)
    states = {k: np.nan for k in ['n_tested', 'n_positive', 'n_negative', 'n_unknown', 'n_not_tested']}
    if convention_verified:
        groups = {k: {str(v).strip().lower() for v in convention.get(k, [])}
                  for k in ['positive', 'negative', 'unknown', 'not_tested']}
        if any(groups[a] & groups[b] for a in groups for b in groups if a != b):
            raise ValueError('OVERLAPPING_OBSERVATION_CONVENTIONS')
        states.update(n_unknown=int(tokens.isin(groups['unknown']).sum()),
                      n_not_tested=int(tokens.isin(groups['not_tested']).sum()))
        clean = clean.mask(tokens.isin(groups['unknown'] | groups['not_tested']))
        if groups['positive'] and groups['negative']:
            values = pd.Series(np.nan, index=raw.index, dtype=float)
            values.loc[tokens.isin(groups['positive'])] = 1.
            values.loc[tokens.isin(groups['negative'])] = 0.
            states.update(n_positive=int(values.eq(1).sum()), n_negative=int(values.eq(0).sum()),
                          n_tested=int(values.notna().sum()))
            return values, 'binary', states
    boolean = pd.api.types.is_bool_dtype(raw.dtype)
    numbers = pd.to_numeric(clean, errors='coerce').astype('float64').replace([np.inf, -np.inf], np.nan)
    numeric_kind = (boolean or declared_type in {'numeric', 'binary'}
                    or pd.api.types.is_numeric_dtype(raw.dtype)
                    or (clean.notna().any() and numbers.notna().sum() == clean.notna().sum()))
    numeric_binary = numbers.nunique() == 2 and set(numbers.dropna().unique()) <= {0., 1.}
    if boolean or declared_type == 'binary' or (numeric_kind and numeric_binary):
        # Numeric binary encodings still require measurement evidence for inference.
        return numbers.where(numbers.isin([0., 1.])), 'binary', states
    if numeric_kind:
        return numbers, 'numeric', states
    return clean, 'categorical', states


def adjustment_data(frame, config, identifier, catalog):
    """No imputation. Era is derived only from a separately reviewed basal date."""
    data = pd.DataFrame(index=frame.index)
    missing = []
    for i, source in enumerate(config.get('confounders', [])):
        if source not in frame:
            missing.append(source)
            data[f'covariate_{i}'] = np.nan
            continue
        values = observed_values(frame[source])
        numbers = pd.to_numeric(values, errors='coerce').astype('float64').replace([np.inf, -np.inf], np.nan)
        is_number = numbers.notna().sum() == values.notna().sum()
        data[f'covariate_{i}'] = numbers if is_number else values.astype('object')
        if data[f'covariate_{i}'].isna().all():
            missing.append(source)
    era = _one(scoped_records(config, 'baseline_era_reviews', identifier))
    era_status = 'BASELINE_ERA_REVIEW_PENDING'
    if (era.get('date_semantics_review_status') == 'VERIFIED'
            and era.get('baseline_window') and verified_reference(era, catalog)):
        date_source = era.get('date_column')
        if date_source in frame:
            dates = pd.to_datetime(frame[date_source], errors='coerce')
            # Fixed conventional five-year strata, never selected using outcomes.
            data['baseline_era'] = (dates.dt.year // 5 * 5).astype('Int64').astype('string').astype('object')
            data['baseline_era'] = data.baseline_era.where(dates.notna(), np.nan)
            era_status = 'VERIFIED' if dates.notna().any() else 'BASELINE_ERA_UNAVAILABLE'
    if era_status != 'VERIFIED':
        data['baseline_era'] = np.nan
        missing.append('baseline_era')
    return data, missing, era_status


def _exclusion(source, role, config):
    tokens = source.lower().split('__')
    if (source in config.get('confounders', []) or source.startswith('demo__')
            or source in {config.get('date_column'), config.get('baseline_flag')}
            or role.get('adjustment_role')):
        return 'ADJUSTMENT_OR_ADMINISTRATIVE_CONTEXT_ONLY'
    if (source.endswith(('_id', '_date', '_datetime', '_timestamp'))
            or any(t in {'patient', 'visit', 'episode', 'community', 'provenance', 'observation'} for t in tokens)
            or source.startswith(('is_', 'missing__', 'observed__', 'source__', 'admin__'))):
        return 'IDENTIFIER_ADMINISTRATIVE_OR_OBSERVATION_CONTEXT'
    if source == config.get('pop_column'):
        return 'DOWNSTREAM_CLINICAL_LABEL'
    return ''


def _proxy_alerts(source, defining, config):
    """Unproven risks go to review; domain equality never establishes dependence."""
    alerts = []
    stem = re.sub(r'__(value|text|status)$', '', source)
    variants = sorted(c for c in defining if c != source and re.sub(r'__(value|text|status)$', '', c) == stem)
    if variants:
        alerts.append('POSSIBLE_ASSAY_ENCODING_PROXY:' + '|'.join(variants))
    if source.startswith('esspri__') and any(c.startswith(('pro__sf36_', 'pro__profad_', 'pro__mdafs_')) for c in defining):
        alerts.append('ESSPRI_PAIN_DRYNESS_FATIGUE_VS_DISCOVERY_PRO_REVIEW')
    if (source.startswith(('essdai__', 'eg_')) or source == 'ovl__status') and any(
            c.startswith(('essdai__', 'eg_')) or c == 'ovl__status' for c in defining):
        alerts.append('ESSDAI_DOMAINS_OVERLAP_ACTIVITY_CONSTRUCT_REVIEW')
    if source == 'ext__biopsy_focus_score':
        alerts.append('GLANDULAR_CONSTRUCT_AND_PROCEDURE_SELECTION_REVIEW')
    if source in {'lab__igg_subclass_1__value', 'lab__igg_subclass_2__value', 'lab__igg_subclass_3__value', 'lab__igg_subclass_4__value'} and 'lab__igg__value' in defining:
        alerts.append('IGG_SUBCLASS_VS_TOTAL_IGG_PROXY_REVIEW')
    for relation in config.get('characterization', {}).get('relationships', []):
        pair = {relation.get('source_column'), relation.get('target_source')}
        if source in pair and bool((pair - {source}) & defining):
            if relation.get('relationship_type') == 'clinical_proxy_reviewed':
                alerts.append('DECLARED_CLINICAL_PROXY_REQUIRES_VERIFIED_REVIEW')
    return sorted(set(alerts))


def _review_record(config, identifier, source, catalog):
    record = _one([r for r in config.get('characterization', {}).get('independence_reviews', [])
                   if r.get('representation_id') == identifier and r.get('source_column') == source])
    complete = (all(record.get(k) == 'APPROVED' for k in REVIEW_FIELDS[:5])
                and record.get('temporal_baseline_status') == 'VERIFIED'
                and all(_text(record.get(k)) for k in REVIEW_FIELDS[6:])
                and verified_reference(record, catalog))
    return record, complete


def _coverage(frame, source, kind, policy):
    rows = []
    for community, group in frame.groupby('community_id', sort=True):
        raw = group[source] if source in group else pd.Series(np.nan, index=group.index)
        values, _, states = values_for(raw, kind, policy)
        rows.append(dict(community_id=str(community), n_hard=len(group),
                         n_observed=int(values.notna().sum()), n_missing=int(values.isna().sum()),
                         n_stable=int(group.structural_stability_status.eq('stable').sum()), **states))
    return rows


def n3_source_inventory(baseline, membership, roles, feature_sets, registry, manifest, config, evidence_catalog=None):
    """One row per frozen source and representation, counting unique patients."""
    catalog = evidence_catalog or {}
    policy = config.get('n3_validation', {})
    if policy.get('schema_version', VERSION) != VERSION:
        raise ValueError('UNSUPPORTED_N3_AUDIT_SCHEMA')
    if policy.get('auto_approve_independence', False) or policy.get('allow_undocumented_source_promotion', False):
        raise ValueError('AUTOMATIC_N3_APPROVAL_FORBIDDEN')
    if baseline.patient_id.isna().any() or baseline.patient_id.duplicated().any():
        raise ValueError('NONUNIQUE_FROZEN_BASELINE')
    if roles.duplicated(KEYS).any():
        raise ValueError('DUPLICATE_SOURCE_REPRESENTATION_ROLE')
    sources = sorted(set(roles.source_column) | (set(baseline) - {'patient_id'}))
    identifiers = manifest['eligible_representation_ids']
    expected = policy.get('expected_source_count')
    if expected is not None and len(sources) != int(expected):
        raise ValueError('FROZEN_SOURCE_COUNT_CONTRACT_CHANGED')
    # This explicitly audited release has a known 505-source / 159-patient contract.
    if manifest.get('run_id') == '20261103T141918515545Z_fd43fde0' and (len(sources) != 505 or len(baseline) != 159):
        raise ValueError('REFERENCE_RELEASE_SOURCE_POPULATION_CONTRACT_CHANGED')
    metadata = {r['source_column']: r for r in registry.to_dict('records')}
    graph = _dependencies(feature_sets, registry, config)
    for edges in graph.values(): edges.sort(key=lambda edge: tuple(str(value) for value in edge))
    rows = []
    for identifier in identifiers:
        members = hard_members(membership, identifier)
        frame = members.merge(baseline, on='patient_id', how='left', validate='one_to_one')
        if not set(members.patient_id) <= set(baseline.patient_id):
            raise ValueError('HARD_MEMBERS_OUTSIDE_FROZEN_BASELINE')
        role_map = roles.loc[roles.representation_id.eq(identifier)].set_index('source_column').to_dict('index')
        if set(role_map) != set(sources):
            raise ValueError('INCOMPLETE_REPRESENTATION_SOURCE_ROLES')
        defining = set(feature_sets.loc[feature_sets.representation_id.eq(identifier) & feature_sets.included.eq(True), 'source_column'])
        adjustment, missing, era = adjustment_data(frame, config, identifier, catalog)
        for source in sources:
            role = role_map[source]
            record, review_complete = _review_record(config, identifier, source, catalog)
            measurement = measurement_policy(config, identifier, source, catalog)
            raw = frame[source] if source in frame else pd.Series(np.nan, index=frame.index)
            values, kind, states = values_for(raw, role.get('type'), measurement)
            base = baseline[source] if source in baseline else pd.Series(np.nan, index=baseline.index)
            base_values, _, _ = values_for(base, role.get('type'), measurement)
            paths = _paths(graph, source)
            conflicts = sorted((set(paths) & defining) - {source})
            alerts = _proxy_alerts(source, defining, config)
            exclusion = _exclusion(source, role, config)
            domain = _text(role.get('clinical_domain')) or _text(metadata.get(source, {}).get('clinical_domain')) or 'UNMAPPED'
            clinical = domain in {f'D{i:02d}' for i in range(1, 8)} or bool(role.get('is_prespecified_anchor')) or (record.get('clinical_outcome_eligible') is True and review_complete)
            candidate = bool(clinical and not exclusion and role['tier'] not in {'N1_DEFINING', 'N2_RELATED'})
            unit_verified = measurement.get('unit_review_status') == 'VERIFIED' and bool(measurement.get('unit'))
            category_verified = measurement.get('category_review_status') == 'VERIFIED' and bool(measurement.get('category_convention'))
            unparsed = int((observed_values(raw).notna() & values.isna()).sum()) if kind == 'numeric' else 0
            measurement_verified = bool(measurement and (unit_verified if kind == 'numeric' else category_verified) and not unparsed)
            coverage = _coverage(frame, source, role.get('type'), measurement)
            observation_fractions = [r['n_observed']/r['n_hard'] for r in coverage if r['n_hard']]
            risks = []
            if exclusion: risks.append(exclusion)
            if domain in {'D08', 'D09', 'UNMAPPED'} and record.get('clinical_outcome_eligible') is not True:
                risks.append('CLINICAL_DOMAIN_ELIGIBILITY_PENDING')
            if role['tier'] in {'N1_DEFINING', 'N2_RELATED'} or conflicts or role.get('direct_in_discovery'):
                review_state = 'BLOCKED_DEPENDENT'
                risks.append('DISCOVERY_VARIABLE' if role['tier'] == 'N1_DEFINING' else 'DEPENDENT_OR_PROXY')
            elif source not in baseline or not base_values.notna().any():
                review_state = 'SOURCE_UNAVAILABLE'
                risks.append('MISSING_SOURCE' if source not in baseline else 'NO_OBSERVED_BASELINE_VALUES')
            else:
                if not review_complete: risks.append('PENDING_CLINICAL_REVIEW')
                if record and not verified_reference(record, catalog): risks.append('REVIEW_REFERENCE_UNVERIFIED')
                if record.get('temporal_baseline_status') != 'VERIFIED': risks.append('BASELINE_MEASUREMENT_WINDOW_UNVERIFIED')
                if not measurement_verified: risks.append('UNIT_OR_CATEGORY_PENDING_REVIEW')
                review_state = 'N3_REVIEWED' if candidate and review_complete and not risks else 'PENDING_REVIEW'
            if alerts and not review_complete: risks.append('CLINICAL_PROXY_REVIEW_PENDING')
            if unparsed: risks.append('UNPARSED_NUMERIC_VALUES_REQUIRE_REVIEW')
            n = int(values.notna().sum())
            support = (n >= int(config['minimum_model_patients']) and len(coverage) >= 2
                       and all(r['n_observed'] >= int(config['minimum_group_patients']) for r in coverage))
            if review_state == 'BLOCKED_DEPENDENT': screen = 'DISCOVERY_VARIABLE' if role['tier'] == 'N1_DEFINING' else 'DEPENDENT_OR_PROXY'
            elif review_state == 'SOURCE_UNAVAILABLE': screen = 'MISSING_SOURCE'
            elif review_state == 'N3_REVIEWED': screen = 'N3_REVIEWED'
            elif not verified_reference(record, catalog): screen = 'NEEDS_LINEAGE_REVIEW'
            elif record.get('temporal_baseline_status') != 'VERIFIED': screen = 'NEEDS_TEMPORAL_BASELINE_REVIEW'
            elif alerts and not review_complete: screen = 'NEEDS_CLINICAL_PROXY_REVIEW'
            elif not measurement_verified: screen = 'NEEDS_UNIT_OR_CATEGORY_REVIEW'
            elif not support: screen = 'INSUFFICIENT_OBSERVED_SUPPORT'
            else: screen = 'N3_CANDIDATE_REVIEWABLE'
            common = values.notna() & adjustment.notna().all(axis=1)
            reason = '|'.join(sorted(set(risks)))
            row = {k: role.get(k, '') for k in ['validation_role', 'direct_in_discovery', 'matched_encoded_features',
                   'relationship_type', 'lineage_evidence', 'conflicting_defining_sources', 'independence_status',
                   'subdomain', 'instrument']}
            canonical_evidence = [dict(defining_source=s, path=paths[s]) for s in sorted(set(paths) & defining)]
            if canonical_evidence or source != config.get('pop_column'):
                row['lineage_evidence'] = json.dumps(canonical_evidence, sort_keys=True)
            row.update(representation_id=identifier, source_column=source, clinical_domain=domain,
                       independence_status_original=role.get('independence_status',''),
                       validation_role_original=role.get('validation_role',''),
                       tier_original=role['tier'], tier='N3_REVIEWED' if review_state == 'N3_REVIEWED' else role['tier'],
                       unit=measurement.get('unit') if unit_verified else 'UNIT_PENDING_REVIEW', type=kind,
                       source_exists=source in baseline, n_baseline_total=len(baseline),
                       n_baseline_observed=int(base_values.notna().sum()), n_hard=len(frame), n_hard_supported=len(frame),
                       n_observed_unique=n, n_missing=len(frame)-n, fraction_available=n/len(frame) if len(frame) else np.nan,
                       n_valid_categories=int(values.nunique()) if kind != 'numeric' else np.nan,
                       observation_class='VERIFIED_CONVENTIONS' if measurement_verified else 'SOURCE_CONVENTIONS_PENDING_REVIEW',
                       observation_denominator='verified_tested_only' if pd.notna(states['n_tested']) else 'observed_unique_patients_conventions_pending',
                       **states, review_state=review_state, screen_status=screen,
                       analysis_status='NOT_ESTIMABLE', status=review_state, reason=reason,
                       blocking_reasons=reason, is_outcome_candidate_for_review=candidate,
                       is_inspection_source=source in INSPECTION_SOURCES,
                       outcome_candidate_reason='CLINICAL_SOURCE_REQUIRES_DOCUMENTED_REVIEW' if candidate else exclusion or 'DISCOVERY_OR_NONCLINICAL_CONTEXT',
                       clinical_proxy_alerts='|'.join(alerts), n_unparsed_numeric_values=unparsed,
                       numeric_quality_status='UNPARSED_VALUES_REQUIRE_REVIEW' if unparsed else 'NO_AUTOMATIC_OUTLIER_RECODING',
                       pop_construct_dependency=bool({'essdai__total','esspri__total'} & set(paths)),
                       baseline_source='verified_frozen_baseline_raw',
                       baseline_definition_id=manifest.get('baseline_definition_id', ''),
                       baseline_episode_key_status='REQUIRES_SOURCE_REVIEW' if not review_complete else 'DOCUMENTED_IN_REVIEW',
                       temporal_baseline_status=record.get('temporal_baseline_status', 'PENDING_REVIEW'),
                       baseline_window='DOCUMENTED_IN_REVIEW' if record.get('baseline_window') else 'PENDING_REVIEW',
                       baseline_window_evidence_sha256=fingerprint(record['baseline_window']) if record.get('baseline_window') else '',
                       coverage_by_community=json.dumps(coverage, sort_keys=True),
                       availability_min=min(observation_fractions) if observation_fractions else np.nan,
                       availability_max=max(observation_fractions) if observation_fractions else np.nan,
                       sampling_caution='TEST_AVAILABILITY_AND_PROCEDURE_SELECTION_MAY_DIFFER_BY_COMMUNITY',
                       n_stable_hard=int(frame.structural_stability_status.eq('stable').sum()),
                       n_provisional_hard=int(frame.structural_stability_status.ne('stable').sum()),
                       n_Y_nonmissing_with_all_covariates=int(common.sum()), n_excluded_missing_covariates=int((values.notna() & ~common).sum()),
                       missing_adjustments='|'.join(missing), baseline_era_status=era,
                       measurement_status='VERIFIED' if measurement_verified else 'PENDING_REVIEW',
                       review_complete=review_complete, reviewer_recorded=bool(record.get('reviewer')),
                       reviewer_evidence_sha256=fingerprint(record['reviewer']) if record.get('reviewer') else '',
                       source_reference=record['source_reference'] if verified_reference(record, catalog) else '',
                       reference_verified=verified_reference(record, catalog),
                       review_history_sha256=fingerprint(json.dumps(record.get('history', []), sort_keys=True)), **INTERNAL)
            if review_state == 'N3_REVIEWED':
                row.update(independence_status='REVIEWED_NO_CONFLICT', validation_role='independent_baseline_anchor')
            # Only controlled status fields, never free review notes or identifying windows.
            row.update({k: record.get(k, 'NOT_REVIEWED') if record.get(k) in {'APPROVED', 'VERIFIED', 'REJECTED', 'PENDING_REVIEW'} else 'NOT_REVIEWED'
                        for k in REVIEW_FIELDS[:6]})
            rows.append(row)
    inventory = pd.DataFrame(rows)
    if len(inventory) != len(sources)*len(identifiers) or inventory.duplicated(KEYS).any():
        raise ValueError('N3_INVENTORY_GRAIN_FAILED')
    return inventory


def n3_candidate_screen(inventory):
    result = inventory.copy()
    # Fixed documentary priority, independent of association estimates and p values.
    result['review_priority'] = np.where(result.source_column.isin(INSPECTION_SOURCES), 1,
                                         np.where(result.is_outcome_candidate_for_review, 2, 3))
    result['clinical_question'] = 'Does this independently measured baseline manifestation associate with frozen communities?'
    return result.sort_values(['review_priority', 'representation_id', 'source_column']).reset_index(drop=True)


def n3_review_status(inventory):
    columns = KEYS + ['tier_original', 'tier', 'review_state', 'review_complete',
        *REVIEW_FIELDS[:6], 'reviewer_recorded', 'reviewer_evidence_sha256', 'source_reference',
        'reference_verified', 'baseline_window', 'baseline_window_evidence_sha256',
        'review_history_sha256', 'status', 'reason', *INTERNAL]
    result = inventory[columns].copy()
    result['approval_summary'] = 'REVIEWED_SOURCES_PRESENT' if result.review_state.eq('N3_REVIEWED').any() else 'NONE_APPROVED'
    return result


def n3_endpoint_support(inventory, config):
    prespecified = {r if isinstance(r, str) else r.get('source_column')
                    for r in config.get('n3_validation', {}).get('prespecified_outcomes', [])}
    selected = inventory.loc[inventory.is_outcome_candidate_for_review
                             | inventory.source_column.isin(INSPECTION_SOURCES | prespecified)].copy()
    selected = selected.rename(columns={'n_observed_unique': 'n_observed_Y', 'n_missing': 'n_missing_Y',
                                        'fraction_available': 'fraction_observed_Y'})
    result = selected.reindex(columns=SUPPORT_COLUMNS)
    result['status'] = 'NOT_ESTIMABLE'
    result['reason'] = selected.apply(lambda r: r.reason or 'PRESPECIFIED_MODEL_AND_STATISTICAL_GATES_REQUIRED', axis=1)
    return result.reset_index(drop=True)


def n3_audit_tables(baseline, membership, roles, feature_sets, registry, manifest, config, evidence_catalog=None):
    inventory = n3_source_inventory(baseline, membership, roles, feature_sets, registry, manifest, config, evidence_catalog)
    queue = inventory.loc[inventory.review_state.ne('N3_REVIEWED')].copy()
    queue['required_action'] = [
        'RETAIN_INTERNAL_OR_RELATED_CHARACTERIZATION' if state == 'BLOCKED_DEPENDENT' else
        'VERIFY_FROZEN_SOURCE_AVAILABILITY' if state == 'SOURCE_UNAVAILABLE' else
        'DOCUMENT_LINEAGE_PROXY_ORIGIN_BASELINE_AND_MEASUREMENT_REVIEW_FOR_THIS_REPRESENTATION'
        for state in queue.review_state]
    return dict(source_inventory=inventory, candidate_screen=n3_candidate_screen(inventory),
                lineage_review_queue=queue, review_status=n3_review_status(inventory),
                endpoint_support=n3_endpoint_support(inventory, config))

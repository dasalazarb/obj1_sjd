"""Versioned source-level characterization; legacy roles are not approval gates.

Edges carry documented semantics. A reverse derivation cannot be followed by a
forward derivation: two inputs of a composite (notably Pop) are not thereby the
same construct. Domain, family and registry instrument/subdomain fallbacks are
never lineage evidence. No clinical approval is inferred from an absent edge.
"""
import json
from collections import defaultdict, deque
import pandas as pd
from .s03_profiles import observed_values, numeric

POLICY_VERSION = 'graph03_characterization_v1'
EVIDENCE = {'N1_DEFINING': 'INTERNAL_STRUCTURE', 'N2_RELATED': 'RELATED_DESCRIPTIVE',
            'N3_CANDIDATE': 'INDEPENDENCE_CANDIDATE_NOT_VALIDATED',
            'N3_REVIEWED': 'INDEPENDENT_EXPLORATORY_CONDITIONAL',
            'PENDING_REVIEW': 'INDEPENDENCE_NOT_ESTABLISHED', 'UNAVAILABLE': 'UNAVAILABLE'}


def _tokens(value):
    if isinstance(value, (list, tuple, set)): return {str(v) for v in value if str(v)}
    if value is None or pd.isna(value): return set()
    text = str(value).strip()
    if not text: return set()
    if text.startswith('['):
        try: return _tokens(json.loads(text))
        except (ValueError, TypeError): pass
    return {v.strip() for v in text.replace(';', '|').split('|') if v.strip()}


def _dependencies(feature_sets, registry, validation):
    graph = defaultdict(list)
    def edge(source, target, kind, evidence, reverse=False):
        if source != target:
            graph[source].append((target, kind, evidence, reverse))
    def pair(source, target, kind, evidence):
        edge(source, target, kind, evidence); edge(target, source, kind, evidence)
    def derived(source, target, evidence):
        edge(source, target, 'derived_from', evidence)
        edge(target, source, 'component_of', evidence, True)
    grouped = defaultdict(set)
    for table_name, table in [('frozen_feature_sets', feature_sets), ('frozen_registry', registry)]:
        for row in table.to_dict('records'):
            source = row.get('source_column')
            if not isinstance(source, str): continue
            for feature in _tokens(row.get('feature')):
                pair(source, feature, 'encoded_variant', table_name + ':exact_feature_source')
            for key in ('derived_from', 'source_columns'):
                for token in _tokens(row.get(key)):
                    derived(source, token, table_name + ':' + key)
            # Qualified construct identifiers only. Generic LAB/D04/family
            # tokens and inherited subdomain instrument IDs cannot join sources.
            for key in ('semantic_feature_id', 'derivation_group', 'instrument_id'):
                for token in _tokens(row.get(key)):
                    qualified = '__' in token
                    explicit = row.get('lineage_semantics') == 'documented_construct' or (
                        row.get('lineage_status') == 'source_code_verified' and bool(_tokens(row.get('lineage_evidence'))))
                    if key == 'instrument_id' and token == row.get('subdomain'): continue
                    if qualified or explicit:
                        grouped[(key, token)].add(source)
    for (key, token), sources in grouped.items():
        kind = 'same_instrument' if key == 'instrument_id' else 'same_raw_source'
        for source in sources:
            pair(source, 'lineage:' + key + ':' + token, kind, 'frozen_registry:' + key)
    for anchor in validation.get('anchors', []):
        for token in _tokens(anchor.get('derived_from')):
            derived(anchor['source_column'], token, 'prespecified_anchor:derived_from')
    policy = validation.get('characterization', {})
    for instrument in policy.get('instruments', []):
        if not instrument.get('source_reference'): continue
        for source in instrument['source_columns']:
            pair(source, 'instrument:' + instrument['instrument'], 'same_instrument', instrument['source_reference'])
    for relation in policy.get('relationships', []):
        if not relation.get('source_reference'): continue
        source, target, kind = relation['source_column'], relation['target_source'], relation['relationship_type']
        if kind in {'derived_from', 'composite_component'}:
            derived(source, target, relation['source_reference'])
        elif kind in {'same_raw_source', 'encoded_variant', 'same_instrument'}:
            pair(source, target, kind, relation['source_reference'])
        elif kind == 'clinical_proxy_reviewed' and relation.get('review_status') == 'APPROVED' and relation.get('reviewer'):
            pair(source, target, kind, relation['source_reference'])
    return graph


def _paths(graph, source):
    """Finite typed paths, including explicit cycles, without composite bridges."""
    paths = {source: []}; queue = deque([(source, False, [])]); visited = set()
    while queue:
        node, downstream, path = queue.popleft()
        if (node, downstream) in visited: continue
        visited.add((node, downstream))
        for target, kind, evidence, reverse in graph[node]:
            if downstream and kind == 'derived_from': continue
            new = path + [dict(source=node, target=target, relationship_type=kind, source_reference=evidence)]
            paths.setdefault(target, new)
            if (target, downstream or reverse) not in visited:
                queue.append((target, downstream or reverse, new))
    return paths


def _review(validation, identifier, source):
    matches = [r for r in validation.get('characterization', {}).get('independence_reviews', [])
               if r.get('representation_id') == identifier and r.get('source_column') == source]
    if len(matches) > 1: raise ValueError('Ambiguous independence review')
    return matches[0] if matches else {}


def feature_roles(feature_sets, baseline, validation, identifiers, clinical_registry=None):
    policy=validation.get('characterization',{})
    if policy.get('schema_version',POLICY_VERSION)!=POLICY_VERSION:
        raise ValueError('Unsupported characterization policy')
    registry = clinical_registry if clinical_registry is not None else pd.DataFrame()
    graph = _dependencies(feature_sets, registry, validation)
    rows = []; anchors = {r['source_column']: r for r in validation.get('anchors', [])}
    markers = {r['source_column']: r for r in validation.get('clinical_marker_panels', [])}
    metadata = {}
    for record in registry.to_dict('records'):
        metadata.setdefault(record.get('source_column'), record)
    if feature_sets.groupby('feature').source_column.nunique().gt(1).any():
        raise ValueError('Ambiguous frozen encoded-feature source lineage')
    columns = (set(baseline) - {'patient_id'}) | set(anchors) | set(markers)
    for identifier in identifiers:
        selected = feature_sets.loc[feature_sets.representation_id.eq(identifier) & feature_sets.included.eq(True)]
        defining = set(selected.source_column.dropna())
        for c in sorted(columns):
            anchor = anchors.get(c, {}); meta = metadata.get(c, {}); review = _review(validation, identifier, c)
            paths = _paths(graph, c); conflicts = sorted(set(paths) & defining)
            direct = c in defining or c in set(selected.feature.dropna())
            matched = sorted(set(selected.loc[selected.source_column.eq(c) | selected.feature.eq(c), 'feature'].dropna()))
            clean = (numeric(baseline[c]) if anchor.get('type') in {'numeric','binary'} or pd.api.types.is_numeric_dtype(baseline[c]) else observed_values(baseline[c])) if c in baseline else pd.Series(dtype=float)
            observed = bool(clean.notna().any())
            adjustment = c in validation.get('confounders', []) or c == validation.get('date_column')
            dependent = c == validation.get('pop_column')
            lineage_status = review.get('lineage_review_status', 'NOT_REVIEWED')
            clinical_status = review.get('clinical_review_status', 'NOT_REVIEWED')
            temporal = review.get('temporal_baseline_status', 'PENDING_REVIEW')
            approved = (lineage_status == clinical_status == 'APPROVED' and temporal == 'VERIFIED'
                        and review.get('origin_review_status') == 'APPROVED'
                        and review.get('proxy_review_status') == 'APPROVED'
                        and review.get('discovery_exclusion_review_status') == 'APPROVED'
                        and all(review.get(k) for k in ['reviewer', 'source_reference', 'baseline_window']))
            role = 'unavailable'; reason = 'not_prespecified_validation_variable'
            if direct: role, reason = 'defining', 'frozen_embedding_or_lens_feature'
            elif conflicts: role, reason = 'derived_from_defining', 'frozen_source_or_construct_lineage:' + '|'.join(conflicts)
            elif c in anchors and not adjustment and not dependent:
                if not observed: reason = 'SOURCE_UNAVAILABLE'
                else: role, reason = 'independent_baseline_anchor', 'held_out_with_no_declared_defining_proxy'
            elif adjustment: role, reason = 'administrative_confounder', 'prespecified_adjustment'
            if direct: tier, independence, blocking = 'N1_DEFINING', 'NOT_INDEPENDENT', ['CIRCULAR_DEFINING_OR_DERIVED_CONSTRUCT']
            elif conflicts or dependent:
                tier, independence, blocking = 'N2_RELATED', 'NOT_INDEPENDENT', ['DEPENDENT_POP_CROSSWALK_ONLY' if dependent else 'CIRCULAR_DEFINING_OR_DERIVED_CONSTRUCT']
            elif not observed: tier, independence, blocking = 'UNAVAILABLE', 'NOT_ESTABLISHED', ['SOURCE_UNAVAILABLE']
            elif c in anchors and not adjustment:
                tier = 'N3_REVIEWED' if approved else 'N3_CANDIDATE'
                independence = 'REVIEWED_NO_CONFLICT' if approved else 'CANDIDATE_PENDING_REVIEW'
                blocking = [] if approved else ['INDEPENDENCE_REVIEW_PENDING']
            else: tier, independence, blocking = 'PENDING_REVIEW', 'NOT_ESTABLISHED', ['ADJUSTMENT_CONTEXT_ONLY' if adjustment else 'NOT_PRESPECIFIED_ANCHOR']
            if not observed and 'SOURCE_UNAVAILABLE' not in blocking: blocking.append('SOURCE_UNAVAILABLE')
            evidence = [dict(defining_source=s, path=paths[s]) for s in conflicts]
            relation_types = sorted({edge['relationship_type'] for item in evidence for edge in item['path']})
            if dependent and not evidence:
                evidence = [dict(dependent_source=c, source_reference='validation_03:Pop_protocol_ESSDAI_ESSPRI',
                                 relationship_type='composite_component', source_columns=['essdai__total', 'esspri__total'])]
            rows.append(dict(representation_id=identifier, source_column=c, validation_role=role, reason=reason,
                legacy_validation_role=role, tier=tier, independence_status=independence,
                clinical_domain=meta.get('clinical_domain', markers.get(c, {}).get('clinical_domain', '')),
                subdomain=meta.get('subdomain', ''), instrument=(meta.get('instrument_id','') if meta.get('instrument_id')!=meta.get('subdomain') else ''),
                semantic_feature_id=meta.get('semantic_feature_id', ''), direct_in_discovery=direct,
                matched_encoded_features='|'.join(matched), relationship_type='direct_in_discovery' if direct else '|'.join(relation_types) or ('composite_component' if dependent else 'no_declared_relation'),
                conflicting_defining_sources='|'.join(conflicts), lineage_evidence=json.dumps(evidence, sort_keys=True),
                lineage_review_status=lineage_status, clinical_review_status=clinical_status,
                temporal_baseline_status=temporal, observation_status='OBSERVED_FROZEN_BASELINE' if observed else 'UNAVAILABLE',
                is_prespecified_anchor=c in anchors, adjustment_role='prespecified_adjustment' if adjustment else '',
                descriptive_eligible=bool(observed), inferential_eligible=bool(tier == 'N3_REVIEWED' and observed),
                inferential_eligibility_scope='independence_gate_only_requires_statistical_gates',
                blocking_reasons='|'.join(blocking), reviewer=review.get('reviewer', ''),
                source_reference=review.get('source_reference', ''), baseline_window=review.get('baseline_window', ''),
                classification_policy_version=POLICY_VERSION, evidence_level=EVIDENCE[tier],
                unit=anchor.get('unit', markers.get(c, {}).get('unit', meta.get('unit', 'source_units'))),
                type=anchor.get('type', markers.get(c, {}).get('type', 'source_type'))))
    return pd.DataFrame(rows)


def independence_block(role):
    """Fail closed for historical callers that lack the new reviewed contract."""
    if role.get('tier') in {'N1_DEFINING', 'N2_RELATED'}: return 'CIRCULAR_DEFINING_OR_DERIVED_CONSTRUCT'
    if role.get('tier') == 'N3_CANDIDATE': return 'INDEPENDENCE_REVIEW_PENDING'
    if role.get('tier') != 'N3_REVIEWED' or role.get('inferential_eligible') is not True:
        return role.get('blocking_reasons') or 'INDEPENDENCE_NOT_ESTABLISHED'
    return ''


def anchor_lineage_audit(roles, validation):
    names = {a['source_column'] for a in validation['anchors']} | {validation['pop_column']}
    audit = roles.loc[roles.source_column.isin(names)].copy()
    audit['anchor'] = audit.source_column
    audit['independence_validation'] = audit.independence_status
    audit['decision'] = ['BLOCKED_CIRCULAR' if r.tier in {'N1_DEFINING', 'N2_RELATED'} else
                         'ELIGIBLE_CONDITIONAL_ON_OTHER_GATES' if r.inferential_eligible else 'NOT_ESTIMABLE' for r in audit.itertuples()]
    audit['justification'] = audit.blocking_reasons
    audit['anchor_source'] = audit.source_column
    return audit


def assert_independent(rows):
    if rows.empty: return
    if not rows.validation_role.eq('independent_baseline_anchor').all():
        raise ValueError('Circular feature in independent anchor validation')
    # Pending hypotheses may be reported, but must never contain estimates.
    for row in rows.to_dict('records'):
        if independence_block(row):
            if row.get('status') != 'NOT_ESTIMABLE' or any(pd.notna(row.get(k)) for k in ['p_value', 'q_value', 'effect', 'ci_low', 'ci_high']):
                raise ValueError('Unreviewed or circular feature in independent anchor validation')

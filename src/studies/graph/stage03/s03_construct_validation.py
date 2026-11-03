"""Per-representation roles and auditable frozen-source dependency gates."""
import json
from collections import defaultdict
import pandas as pd
from .s03_profiles import observed_values


def _tokens(value):
    if isinstance(value,(list,tuple,set)): return {str(v) for v in value if str(v)}
    if value is None or pd.isna(value): return set()
    text=str(value).strip()
    if not text: return set()
    if text.startswith('['):
        try: return _tokens(json.loads(text))
        except (ValueError,TypeError): pass
    return {v.strip() for v in text.replace(';','|').split('|') if v.strip()}


def _dependencies(feature_sets,registry,validation):
    graph=defaultdict(set)
    for table in (feature_sets,registry):
        for row in table.to_dict('records'):
            source=row.get('source_column')
            if not isinstance(source,str): continue
            for key in ('feature','semantic_feature_id','derivation_group','derived_from','lineage','source_columns'):
                for token in _tokens(row.get(key)):
                    graph[source].add(token); graph[token].add(source)
    for anchor in validation['anchors']:
        for token in _tokens(anchor.get('derived_from')):
            graph[anchor['source_column']].add(token); graph[token].add(anchor['source_column'])
    # Whole instruments and their indices/components share a construct even if
    # a historical registry lacks explicit derivation metadata. No domain-wide
    # equivalence or clinical approval is inferred from technical eligibility.
    families=defaultdict(set)
    for column in set(graph) | {a['source_column'] for a in validation['anchors']}:
        if column.startswith(('essdai__','ovl__')): families['ESSDAI_overlap'].add(column)
        elif column.startswith('esspri__'): families['ESSPRI'].add(column)
        elif column.startswith(('pro__sf36_','sf36__')): families['SF36'].add(column)
    for family,columns in families.items():
        for column in columns:
            graph[column].add('construct:'+family); graph['construct:'+family].add(column)
    return graph


def _reachable(graph,source):
    visited=set(); todo=[source]
    while todo:
        node=todo.pop()
        if node in visited: continue
        visited.add(node); todo.extend(graph[node]-visited)
    return visited


def feature_roles(feature_sets,baseline,validation,identifiers,clinical_registry=None):
    registry=clinical_registry if clinical_registry is not None else pd.DataFrame()
    graph=_dependencies(feature_sets,registry,validation)
    rows=[]; anchors={r['source_column']:r for r in validation['anchors']}
    for identifier in identifiers:
        selected=feature_sets.loc[feature_sets.representation_id.eq(identifier) & feature_sets.included]
        defining=set(selected.source_column) | set(selected.feature)
        for c in sorted((set(baseline.columns)-{'patient_id'}) | set(anchors)):
            anchor=anchors.get(c,{})
            conflicts=sorted(_reachable(graph,c) & defining)
            role='unavailable'; reason='not_prespecified_validation_variable'
            if c in defining:
                role='defining'; reason='frozen_embedding_or_lens_feature'
            elif conflicts:
                role='derived_from_defining'; reason='frozen_source_or_construct_lineage:'+ '|'.join(conflicts)
            elif c in anchors:
                if c not in baseline or observed_values(baseline[c]).isna().all(): reason='SOURCE_UNAVAILABLE'
                else:
                    role='independent_baseline_anchor'; reason='held_out_with_no_declared_defining_proxy'
            elif c in validation['confounders'] or c==validation['date_column']:
                role='administrative_confounder'; reason='prespecified_adjustment'
            rows.append({'representation_id':identifier,'source_column':c,'validation_role':role,'reason':reason,
                'conflicting_defining_sources':'|'.join(conflicts),'unit':anchor.get('unit','source_units'),
                'type':anchor.get('type','source_type')})
    return pd.DataFrame(rows)


def anchor_lineage_audit(roles,validation):
    names={a['source_column'] for a in validation['anchors']} | {validation['pop_column']}
    audit=roles.loc[roles.source_column.isin(names)].copy()
    audit['anchor']=audit.source_column
    audit['relationship_type']=audit.validation_role
    audit['independence_validation']=audit.validation_role.map({
        'independent_baseline_anchor':'NO_DECLARED_CONFLICT','defining':'FAILED',
        'derived_from_defining':'FAILED'}).fillna('NOT_ESTABLISHED')
    audit['decision']=audit.validation_role.map({'independent_baseline_anchor':'ELIGIBLE_CONDITIONAL_ON_OTHER_GATES',
        'defining':'BLOCKED_CIRCULAR','derived_from_defining':'BLOCKED_CIRCULAR'}).fillna('NOT_ESTIMABLE')
    audit['justification']=audit.reason
    return audit


def assert_independent(rows):
    if not rows.empty and not rows.validation_role.eq('independent_baseline_anchor').all():
        raise ValueError('Circular feature in independent anchor validation')

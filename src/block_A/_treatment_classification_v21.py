"""Conservative Step 08 product axes and descriptive QC; no exposure inference.

Private implementation helper for 08_treatment_audit_v2.py, never an entrypoint.
All source row units survive; ingredient lists are never exploded for order counts.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

import pandas as pd

VERSION = '08-classification-2.1.8'
RULE_SOURCE = 'investigator specification 2026-10-21 §3; candidate nomenclature, not institutional validation'
SYSTEMIC = 'sjd_systemic_candidate'
OCULAR = 'sjd_ocular_symptomatic_candidate'
ORAL = 'sjd_oral_symptomatic_candidate'
GENERAL = 'concomitant_or_general'
PROCEDURE = 'procedure_or_diagnostic_candidate'
DENTAL = 'dental_or_local_support'
INDETERMINATE = 'indeterminate'

# Five-category clinical grouping requested by the investigator. These are
# potential pharmacological roles, NOT patient-specific SjD indications.
PRIMARY_SYSTEMIC = 'systemic_sjd'
PRIMARY_SYMPTOMATIC = 'symptomatic_sjd'
PRIMARY_CONCOMITANT = 'concomitant'
PRIMARY_PROCEDURE = 'procedure_related'
PRIMARY_INDETERMINATE = 'indeterminate'
PRIMARY_CATEGORIES = (
    PRIMARY_SYSTEMIC, PRIMARY_SYMPTOMATIC, PRIMARY_CONCOMITANT,
    PRIMARY_PROCEDURE, PRIMARY_INDETERMINATE,
)
# Investigator-approved STUDY CATEGORY rules, not patient-specific clinical
# indication, procedure confirmation, product composition or exposure evidence.
# Keep classification_confidence and review_reasons on their separate axes.
CATEGORY_RULE_SIGNOFF = {
    'citric acid monohydrate': (PRIMARY_PROCEDURE, 'investigator_decision_2026-10-22'),
    'mouth kote dry mouth treatment': (PRIMARY_SYMPTOMATIC, 'investigator_decision_2026-10-22'),
}
PRIMARY_FROM_RELEVANCE = {
    SYSTEMIC: PRIMARY_SYSTEMIC,
    OCULAR: PRIMARY_SYMPTOMATIC,
    ORAL: PRIMARY_SYMPTOMATIC,
    GENERAL: PRIMARY_CONCOMITANT,
    PROCEDURE: PRIMARY_PROCEDURE,
    # Dental fluoride without verified clinical indication is local supportive
    # care, NOT automatically SjD symptomatic treatment.
    DENTAL: PRIMARY_INDETERMINATE,
    INDETERMINATE: PRIMARY_INDETERMINATE,
    'device_or_non_drug': PRIMARY_INDETERMINATE,
}

# Only explicit ingredient/brand aliases. Strengths and formulation tokens are
# anchored and cannot swallow an arbitrary unrecognized second ingredient.
EXTRA_RULES = (
    ('citric acid monohydrate', 'diagnostic_or_local_agent', ('citric acid monohydrate',)),
    ('sodium fluoride', 'dental_fluoride', ('sodium fluoride',)),
    ('ibuprofen', 'nsaid', ('ibuprofen', 'advil', 'motrin')),
    ('carboxymethylcellulose', 'lubricant', ('carboxymethylcellulose', 'carboxymethylcellulose sodium', 'refresh plus',
                                          'carboxymethylcellulose ophth drops 1% (celluvisc)')),
    ('mineral oil', 'lubricant', ('mineral oil',)),
    ('petrolatum', 'lubricant', ('petrolatum', 'white petrolatum')),
    ('codeine', 'opioid_analgesic', ('codeine', 'codeine phosphate')),
    ('fluconazole', 'antifungal', ('fluconazole', 'diflucan')),
    ('nystatin', 'antifungal', ('nystatin',)),
    ('iopamidol', 'radiologic_contrast', ('iopamidol', 'isovue')),
)
# Explicit BTRIS ingredient-field annotations: the trailing terms in these
# source strings are pharmacologic class / formulation descriptors, not
# additional active ingredients. Never strip arbitrary comma-separated text.
# Exact matches provide ingredient identity candidates, not indication,
# medication administration, product use or regulatory validation.
BTRIS_INGREDIENT_ANNOTATIONS = {
    'fluoride, sodium fluoride': ('sodium fluoride',),
    'carboxymethylcellulose, ophthalmic lubricant': ('carboxymethylcellulose',),
    'white petrolatum/mineral oil combination, ophthalmic lubricant, mineral oil': ('petrolatum', 'mineral oil'),
}

SUFFIX = re.compile(
    r'(?:\s|[(),.\-]|\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|%|units?)|'
    r'tablets?|tabs?|capsules?|caps?|oral|ophthalmic|ophth|ocular|topical|'
    r'solution|suspension|emulsions?|injection|cream|ointment|oint|drops|gel|delayed|release|'
    r'po|iv|im|sc|hfa|\(refresh plus\))+', re.I)


def rules(legacy_rules):
    return tuple(legacy_rules) + EXTRA_RULES


def family_label(family):
    return {'needs_review': 'calcineurin_inhibitor',
            'symptomatic_sjd_oral': 'secretagogue',
            'symptomatic_sjd_ocular': 'ocular_lubricant_product',
            'other_symptomatic': 'general_analgesic',
            'concomitant_other': 'concomitant_medication'}.get(family, family)


def dictionary(legacy_rules):
    rows = []
    for ingredient, family, aliases in rules(legacy_rules):
        for alias in aliases:
            rows.append(dict(classification_rule_id=f'{VERSION}:{ingredient}:{alias}',
                             exact_alias=alias, ingredient_canonical=ingredient,
                             pharmacologic_family=family_label(family), route_requirement='explicit_or_unequivocal_brand',
                             formulation_requirement='retain_observed_tokens',
                             relevance_rule='route_and_family_axes_v21',
                             rule_source=RULE_SOURCE, classification_version=VERSION,
                             confidence_policy='high_exact_identity_only; clinical_relevance_is_candidate',
                             pending_clinical_review=True))
    for source_term, components in sorted(BTRIS_INGREDIENT_ANNOTATIONS.items()):
        rows.append(dict(classification_rule_id=f'{VERSION}:source_ingredient_annotation:{source_term}',
                         exact_alias=source_term, ingredient_canonical=' + '.join(components),
                         pharmacologic_family='source_ontology_descriptor',
                         route_requirement='source_exact_only',
                         formulation_requirement='source_exact_only',
                         relevance_rule='identity_only_no_clinical_indication',
                         rule_source=RULE_SOURCE, classification_version=VERSION,
                         confidence_policy='exact_annotation_match; subject_to_product_crosscheck',
                         pending_clinical_review=True))
    for alias, family, relevance in (
        ('mouth kote dry mouth treatment', 'oral_moisturizer', ORAL),
        ('oral balance gel', 'oral_moisturizer', ORAL),
        ('aerochamber', 'device', 'device_or_non_drug'),
    ):
        rows.append(dict(classification_rule_id=f'{VERSION}:{alias}', exact_alias=alias,
                         ingredient_canonical='', pharmacologic_family=family,
                         route_requirement='unresolved', formulation_requirement='unresolved',
                         relevance_rule=relevance, rule_source=RULE_SOURCE,
                         classification_version=VERSION, confidence_policy='candidate_review',
                         pending_clinical_review=True))
    result = pd.DataFrame(rows)
    result['study_category_rule_status'] = [
        'approved_by_investigator' if (
            row.ingredient_canonical == 'citric acid monohydrate' or
            row.exact_alias == 'mouth kote dry mouth treatment'
        ) else 'candidate_not_adjudicated'
        for row in result.itertuples()
    ]
    result['study_category_rule_decision_ref'] = [
        'investigator_decision_2026-10-22'
        if approved else ''
        for approved in result.study_category_rule_status.eq('approved_by_investigator')
    ]
    return result


def _text(value):
    return '' if value is None or pd.isna(value) else str(value)


def _matches(value, vocabulary):
    text = _text(value).strip().casefold()
    for ingredient, family, aliases in vocabulary:
        for alias in sorted(aliases, key=len, reverse=True):
            if text == alias or (text.startswith(alias) and SUFFIX.fullmatch(text[len(alias):])):
                return [(ingredient, family, alias)]
    # Split component names, never strength/unit expressions (mg/mL, Units/gram).
    parts = re.split(
        r'\s*(?:\+|;|\band\b)\s*|(?<!\d)\s*/\s*(?!\d|ml\b|gram\b|g\b|l\b)',
        text,
    )
    if len(parts) > 1:
        matches = [_matches(part, vocabulary) for part in parts]
        if all(matches):
            return [item for match in matches for item in match]
    return []


def _source_annotated_ingredients(value, vocabulary):
    """Interpret only reviewed, exact BTRIS source annotation strings.

    Produce the same ingredient tuples as the existing matcher, and reject
    any annotation whose component has no recognized rule. Product and source
    identities must still agree through the existing conflict gate.
    """
    source = _text(value).strip().casefold()
    if source not in BTRIS_INGREDIENT_ANNOTATIONS:
        return []
    canonical = BTRIS_INGREDIENT_ANNOTATIONS[source]
    matched = []
    for expected in canonical:
        rules_for_component = [
            (ingredient, family, source)
            for ingredient, family, _aliases in vocabulary if ingredient == expected
        ]
        if len(rules_for_component) != 1:
            return []
        matched.extend(rules_for_component)
    return matched


def _route(value):
    text = re.sub(r'\s+', ' ', _text(value).strip().casefold())
    exact = {'po': 'oral', 'oral': 'oral', 'by mouth': 'oral',
             'ophthalmic': 'ophthalmic', 'ophth': 'ophthalmic',
             'ocular': 'ophthalmic', 'eye': 'ophthalmic',
             'in both eyes': 'ophthalmic', 'in left eye': 'ophthalmic',
             'in right eye': 'ophthalmic',
             'topical': 'topical', 'topically': 'topical', 'dental': 'topical',
             'iv': 'iv', 'intravenous': 'iv', 'im': 'other', 'sc': 'other',
             'inhaled': 'other', 'intranasal': 'other'}
    return exact.get(text, 'unresolved')


def _drug_short_label(ingredients, route, original_name, families, *, uncertain=False):
    """Display only; never an exposure identity or medication grouping key.

    Preserve the full original name and the separate ingredient/route/formulation
    fields. Unmapped and contradictory identities have no short label.
    """
    if uncertain:
        return ''
    if ingredients:
        label = ' + '.join(i[:1].upper() + i[1:] for i in ingredients)
        route_sensitive = bool(set(ingredients) & {
            'cyclosporine', 'pilocarpine', 'prednisolone', 'dexamethasone',
            'methylprednisolone', 'sodium fluoride',
        })
        if route in {'ophthalmic', 'topical', 'iv', 'other'}:
            return f'{label} ({route})'
        if route == 'oral' and route_sensitive:
            return f'{label} (oral)'
        if route == 'unresolved' and route_sensitive:
            return f'{label} (route unresolved)'
        return label
    # A product can be identifiable while its precise ingredient composition is
    # not established. Its branded display label must carry that uncertainty.
    aliases = {
        'mouth kote dry mouth treatment': 'Mouth Kote',
        'oral balance gel': 'Oral Balance Gel',
        'artificial tears': 'Artificial tears',
        'systane': 'Systane',
        'refresh tears': 'Refresh Tears',
    }
    name = _text(original_name).strip().casefold()
    if families and name in aliases:
        return aliases[name] + ' (composition unverified)'
    return ''


def classify(drug, ingredient, route, legacy_rules):
    vocabulary = rules(legacy_rules)
    product = _matches(drug, vocabulary)
    annotated_components = _source_annotated_ingredients(ingredient, vocabulary)
    components = annotated_components or _matches(ingredient, vocabulary)
    annotation_rule_id = (f'{VERSION}:source_ingredient_annotation:{_text(ingredient).strip().casefold()}'
                          if annotated_components else '')
    flags = []
    product_ids, ingredient_ids = {x[0] for x in product}, {x[0] for x in components}
    conflict = bool(product_ids and ingredient_ids and product_ids != ingredient_ids)
    if conflict:
        flags.append('ingredient_product_conflict')
    if _text(ingredient).strip() and not components:
        flags.append('ingredient_unresolved')
    if not product and re.search(r'\+|;|\band\b|(?<!\d)\s*/\s*(?!\d|ml\b|gram\b|g\b|l\b)', _text(drug), re.I):
        flags.append('product_components_unresolved')
    chosen = components or product
    # Ingredient ontology descriptors such as "Glycerin Topical" do not
    # override explicitly documented product formulation or administration route.
    text = _text(drug).casefold()
    special = text.strip()
    explicit_route = _route(route)
    hints = set()
    if re.search(r'\b(ophthalmic|ophth|ocular|restasis|cequa|refresh plus|artificial tears|refresh tears|systane)\b', text):
        hints.add('ophthalmic')
    if re.search(r'\b(oral|tablet|tablets|capsule|capsules|salagen|evoxac|neoral|sandimmune|myfortic|cellcept)\b', text):
        hints.add('oral')
    if re.search(r'\b(topical|dental|cream)\b', text):
        hints.add('topical')
    # Oral Balance Gel is applied to the oral cavity. A recorded topical route
    # is not proof of an oral-swallowed medication or a conflicting route.
    if special == 'oral balance gel' and explicit_route == 'topical':
        hints.discard('oral')
    generic_topical_eye = hints == {'ophthalmic'} and explicit_route == 'topical'
    route_conflict = len(hints) > 1 or bool(
        hints and explicit_route != 'unresolved' and explicit_route not in hints
        and not generic_topical_eye)
    if route_conflict:
        flags.append('formulation_route_conflict')
    if generic_topical_eye:
        flags.append('ophthalmic_formulation_generic_topical_route_review')
    canonical_route = 'unresolved' if route_conflict else explicit_route
    if generic_topical_eye:
        canonical_route = 'ophthalmic'
    elif canonical_route == 'unresolved' and len(hints) == 1 and not route_conflict:
        canonical_route = next(iter(hints))
    formulation = 'unresolved'
    for expression, label in (
        (r'\b(?:ointment|oint)\b', 'ointment'), (r'\bdrops?\b', 'drops'),
        (r'\bsolution\b', 'solution'), (r'\bsuspension\b', 'suspension'),
        (r'\bemulsions?\b', 'emulsion'),
        (r'\bgel\b', 'gel'), (r'\bcream\b', 'cream'),
        (r'\btablets?\b', 'tablet'), (r'\bcapsules?\b', 'capsule'),
        (r'\binjection\b', 'injection'),
    ):
        if re.search(expression, text):
            formulation = label
            break
    if formulation in {'drops', 'ointment', 'solution', 'gel', 'emulsion'} and canonical_route != 'unresolved':
        formulation = canonical_route + '_' + formulation
    ingredients = sorted({x[0] for x in chosen}) if not conflict else []
    families = sorted({x[1] for x in chosen})
    relevance = INDETERMINATE
    if ingredients:
        relevance = GENERAL
    if any(f in {'nonbiologic_systemic', 'biologic'} for f in families):
        relevance = SYSTEMIC
    if set(ingredients) & {'cyclosporine', 'pilocarpine'}:
        relevance = (OCULAR if canonical_route == 'ophthalmic' and 'cyclosporine' in ingredients
                     else ORAL if canonical_route == 'oral' and 'pilocarpine' in ingredients
                     else SYSTEMIC if canonical_route == 'oral' and 'cyclosporine' in ingredients
                     else INDETERMINATE)
        if canonical_route == 'unresolved':
            flags.append('route_required')
    if 'cevimeline' in ingredients:
        relevance = ORAL if canonical_route == 'oral' else INDETERMINATE
        if canonical_route == 'unresolved':
            flags.append('route_required')
    if any(f in {'lubricant', 'symptomatic_sjd_ocular'} for f in families):
        relevance = OCULAR if canonical_route == 'ophthalmic' else GENERAL
    if 'systemic_glucocorticoid' in families:
        families = [f for f in families if f != 'systemic_glucocorticoid'] + [
            'systemic_glucocorticoid' if canonical_route in {'oral', 'iv'} else
            'ophthalmic_glucocorticoid' if canonical_route == 'ophthalmic' else
            'local_glucocorticoid' if canonical_route == 'topical' else 'glucocorticoid_route_unresolved']
        relevance = SYSTEMIC if canonical_route in {'oral', 'iv'} else INDETERMINATE
        if canonical_route == 'unresolved':
            flags.append('glucocorticoid_route_requires_review')
    if 'sodium fluoride' in ingredients:
        relevance = DENTAL if canonical_route == 'topical' else INDETERMINATE
    if 'citric acid monohydrate' in ingredients:
        relevance = PROCEDURE
        flags.append('citric_acid_use_pending_validation')
    if 'iopamidol' in ingredients:
        relevance = PROCEDURE
    if 'antifungal' in families:
        relevance = INDETERMINATE
    if special == 'mouth kote dry mouth treatment':
        families, relevance = ['oral_moisturizer'], ORAL
        flags.append('composition_pending_validation')
    if special == 'oral balance gel':
        families, relevance = ['oral_moisturizer'], ORAL
        flags.append('composition_pending_validation')
    if special == 'aerochamber':
        families, relevance = ['device'], 'device_or_non_drug'
        flags.append('device_product_review')
    if ingredients == ['artificial tears']:
        ingredients = []
        flags.append('composition_pending_validation')
    families = [family_label(family) for family in families]
    if conflict or route_conflict or 'product_components_unresolved' in flags:
        relevance = INDETERMINATE
    if not ingredients and not families:
        flags.append('no_unambiguous_identity_rule')
    identity = ' + '.join(ingredients)
    strengths = re.findall(r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b|\b\d+(?:\.\d+)?\s*%', _text(drug), re.I)
    strength = ' | '.join(re.sub(r'\s+', ' ', value.casefold()).strip() for value in strengths)
    product_canonical = ' | '.join([identity or special if families else '', canonical_route, formulation, strength]) if families else ''
    confidence = 'unmapped' if not families else 'candidate_review' if flags else 'high_exact'
    primary_category = PRIMARY_FROM_RELEVANCE.get(relevance, PRIMARY_INDETERMINATE)
    approved_key = (
        'citric acid monohydrate' if ingredients == ['citric acid monohydrate']
        else 'mouth kote dry mouth treatment' if special == 'mouth kote dry mouth treatment'
        else ''
    )
    approved = CATEGORY_RULE_SIGNOFF.get(approved_key)
    # A rule-level decision must never override contradictory patient records.
    category_rule_approved = bool(
        approved and approved[0] == primary_category and not conflict and
        not route_conflict and 'product_components_unresolved' not in flags)
    short_label = _drug_short_label(
        ingredients, canonical_route, drug, families,
        uncertain=bool(conflict or route_conflict or
                       'product_components_unresolved' in flags))
    return dict(source_ingredient_annotation_rule_id=annotation_rule_id,
                study_category_rule_status=('approved_by_investigator' if category_rule_approved
                                            else 'candidate_not_adjudicated'),
                study_category_rule_decision_ref=(approved[1] if category_rule_approved else ''),
                drug_short_label=short_label,
                clinical_category_primary=primary_category,
                clinical_category_detail=relevance,
                clinical_category_validation='candidate_only_indication_unconfirmed',
                ingredient_canonical=identity, ingredients_canonical=json.dumps(ingredients),
                product_canonical=product_canonical, route_canonical=canonical_route,
                formulation_canonical=formulation, strength_canonical=strength, pharmacologic_family='|'.join(sorted(families)) or 'unmapped',
                clinical_relevance_candidate=relevance, clinical_indication_confirmed=False,
                classification_rule_id='|'.join(sorted(f'{VERSION}:{x[0]}:{x[2]}' for x in chosen)) or
                (f'{VERSION}:{special}' if families else f'{VERSION}:unmapped'),
                classification_confidence=confidence, requires_manual_review=bool(flags),
                review_reasons='|'.join(sorted(flags)), classification_version=VERSION)


def _interval(row):
    if row.start_status == 'missing' or row.end_status == 'missing':
        return 'missing'
    if row.start_status != 'valid' or row.end_status != 'valid':
        return 'uncertain'
    if row.start_precision not in {'datetime', 'day'} or row.end_precision not in {'datetime', 'day'}:
        return 'uncertain'
    start, end = datetime.fromisoformat(row.start_parsed), datetime.fromisoformat(row.end_parsed)
    try:
        seconds = (end - start).total_seconds()
    except TypeError:
        return 'uncertain'
    if seconds < 0:
        return 'reverse'
    if seconds == 0:
        return 'zero'
    return 'short' if seconds < 86400 else 'plausible'


def enrich(records, legacy_rules, linkage):
    result = records.copy()
    axes = [classify(row.drug_raw if row.name_present else '', row.ingredient_raw if row.name_present else '',
                     row.route, legacy_rules) for row in result.itertuples()]
    # Stable schema also when every source is unavailable.
    empty = classify('', '', '', legacy_rules)
    for column in empty:
        result[column] = [item[column] for item in axes]
    result['status_code_raw'] = result.order_status_raw
    result['status_semantic_interpretation'] = 'unvalidated'
    result['status_semantic_evidence'] = 'none'
    result['administration_evidence'] = 'not_established'
    result['dispense_evidence'] = 'not_established'
    result['evidence_source_type'] = result.source_kind.map({
        'CTDB': 'ctdb_reported_current_medication', 'BTRIS': 'btris_documented_order'})
    result['order_context_candidate'] = result.get('order_type_raw', pd.Series('', index=result.index)).fillna('')
    result['order_interval_qc'] = [_interval(row) for row in result.itertuples()]
    result['order_start_semantics'] = result.time_semantics
    result['order_end_semantics'] = result.end_semantics
    result['linkage_method'] = linkage.get('mode', 'authorized_crosswalk')
    result['id_precision_qc'] = result.get('mrn_identity_quality', pd.Series('', index=result.index)).fillna('ctdb_exact_key')
    result['linkage_semantic_validation'] = 'external_validation_pending'
    for column in ('order_type_raw', 'order_set_raw', 'take_home_raw', 'data_source_raw', 'category_raw', 'drug_type_raw'):
        if column not in result:
            result[column] = ''
        result[column] = result[column].fillna('')
    # Generic "Core Visit"/"Focused Visits" describe visits, not necessarily
    # drug administration or a diagnostic procedure.
    procedure = result.order_set_raw.str.contains(
        r'\b(?:procedure|test|biopsy|sialometry|pft|pulmonary function tests?)\b',
        case=False, regex=True,
    )
    visit_only = (~procedure & result.order_set_raw.str.contains(
        r'\bvisits?\b', case=False, regex=True))
    result['order_set_procedure_context'] = [
        'candidate' if p else 'visit_context_only' if v else 'unresolved'
        for p, v in zip(procedure, visit_only)
    ]
    result['order_set_context_rule_id'] = [
        f'{VERSION}:order_set_explicit_procedure_token' if p
        else f'{VERSION}:visit_only_not_procedure' if v else 'none'
        for p, v in zip(procedure, visit_only)
    ]
    result['clinical_relevance_secondary'] = [json.dumps([PROCEDURE] if p and r != PROCEDURE else [])
                                              for p, r in zip(procedure, result.clinical_relevance_candidate)]
    for column, token in (
        ('possible_duplicate_order', 'exact_reexport_duplicate_retained'),
        ('possible_revision', 'same_order_id_revision_or_discordance_retained'),
        ('possible_cross_protocol_reexport', 'cross_protocol_possible_duplicate_retained'),
    ):
        result[column] = result.qc_flags.str.contains(token, regex=False)
    return result


COUNT_COLUMNS = ['n_order_rows', 'n_distinct_order_ids', 'n_distinct_eligible_patients', 'n_orders_with_valid_start_date']


def cross_tab(orders, columns, linkage_ok):
    rows = []
    for values, part in orders.groupby(columns, dropna=False, sort=True):
        if len(columns) == 1 and not isinstance(values, tuple):
            values = (values,)
        rows.append(dict(zip(columns, values), n_order_rows=len(part),
                         n_distinct_order_ids=part.loc[part.order_id_pseudonymous.ne(''), 'order_id_pseudonymous'].nunique(),
                         n_distinct_eligible_patients=part.loc[part.eligible_linked, 'linked_patient_pseudonymous'].replace('', pd.NA).nunique() if linkage_ok else None,
                         n_orders_with_valid_start_date=int(part.start_status.eq('valid').sum())))
    return pd.DataFrame(rows, columns=columns + COUNT_COLUMNS)


def coverage(records, eligible, linkage, source_ok):
    rows = []
    scopes = {'CTDB': ['CTDB'], '11D': ['11D'], '15D': ['15D'],
              'BTRIS': ['11D', '15D'], 'union': ['CTDB', '11D', '15D']}
    metrics = {
        'any_documentation': lambda f: f.source_kind.eq('BTRIS') | f.name_present.astype(bool),
        'any_identifiable_drug': lambda f: f.ingredient_canonical.ne(''),
        'any_sjd_systemic_candidate_documentation': lambda f: f.clinical_relevance_candidate.eq(SYSTEMIC),
        'any_sjd_ocular_candidate_documentation': lambda f: f.clinical_relevance_candidate.eq(OCULAR),
        'any_sjd_oral_candidate_documentation': lambda f: f.clinical_relevance_candidate.eq(ORAL),
        'any_procedure_candidate_order': lambda f: f.source_kind.eq('BTRIS') & (f.clinical_relevance_candidate.eq(PROCEDURE) | f.order_set_procedure_context.eq('candidate')),
        'any_unmapped_order': lambda f: f.source_kind.eq('BTRIS') & f.classification_confidence.eq('unmapped'),
        **{f'any_primary_{category}_candidate_documentation':
           (lambda frame, category=category: frame.clinical_category_primary.eq(category)
            & (frame.source_kind.eq('BTRIS') | frame.name_present.astype(bool)))
           for category in PRIMARY_CATEGORIES},
        'any_record_with_full_order_datetime': lambda f: f.source_kind.eq('BTRIS') & f.start_precision.eq('datetime') & f.start_status.eq('valid'),
        'any_identified_route_and_valid_start': lambda f: f.ingredient_canonical.ne('') & f.route_canonical.ne('unresolved') & f.start_status.eq('valid'),
    }
    for scope, sources in scopes.items():
        available = bool(eligible) and all(source_ok[s] for s in sources)
        valid = available and (scope == 'CTDB' or linkage['status'] == 'validated')
        status = 'available_technical_linkage_semantics_pending' if valid else 'blocked_source' if not available else 'blocked_linkage'
        part = records[records.source.isin(sources)]
        for metric, selector in metrics.items():
            selected = part.loc[selector(part)]
            n = selected.loc[selected.eligible_linked, 'linked_patient_pseudonymous'].replace('', pd.NA).nunique() if valid else None
            if valid and n > len(eligible):
                raise ValueError('Coverage exceeds eligible denominator')
            rows.append(dict(scope=scope, metric=metric, n_patients=n, denominator=len(eligible) or None,
                             percent_eligible=100*n/len(eligible) if valid else None,
                             n_record_rows=len(selected) if available else None, status=status,
                             population_stratum='population_stratum_not_available',
                             definition='unique eligible IDs; documentation only; categories overlap'))
        for route in ('oral', 'ophthalmic', 'topical', 'iv', 'other', 'unresolved'):
            selected = part[(part.source_kind.eq('BTRIS') | part.name_present.astype(bool)) & part.route_canonical.eq(route)]
            rows.append(dict(scope=scope, metric=f'any_documentation_route_{route}',
                             n_patients=selected.loc[selected.eligible_linked, 'linked_patient_pseudonymous'].replace('', pd.NA).nunique() if valid else None,
                             denominator=len(eligible) or None, percent_eligible=None,
                             n_record_rows=len(selected) if available else None, status=status,
                             population_stratum='population_stratum_not_available', definition='documented route; no administration inference'))
    return pd.DataFrame(rows)


def _restricted_vocabulary_display(value):
    """Avoid copying obvious free-text contact instructions into review tables.

    Preserve the untouched original in the restricted row-level audit. These
    heuristics reduce accidental disclosure; they do NOT certify that any
    aggregate output is anonymous or safe for unrestricted distribution.
    """
    text = _text(value)
    if (re.search(r'[\r\n]|@|\b(?:please contact|research nurse|phone number)\b',
                  text, re.I)
            or re.search(r'\b\d{3}[-.]\d{3}[-.]\d{4}\b', text)):
        return '[redacted free-text medication entry]', True
    return text, False


def _group_count_no_ids(frame, columns):
    """Aggregate without person identifiers, source dates or free-text fields."""
    return (frame.groupby(columns, dropna=False).size()
            .rename('n_record_rows').reset_index())


def artifacts(records, legacy_rules, eligible, linkage, source_ok):
    orders = records[records.source_kind.eq('BTRIS')].copy()
    ok = linkage['status'] == 'validated'
    dimensions = {
        'folder_protocol_status_type': ['source', 'protocol_number_raw', 'status_code_raw', 'order_type_raw'],
        'status_type_takehome_source': ['status_code_raw', 'order_type_raw', 'take_home_raw', 'data_source_raw'],
        'status_type_category_drugtype': ['status_code_raw', 'order_type_raw', 'category_raw', 'drug_type_raw'],
        'status_dates_qc': ['status_code_raw', 'start_precision', 'end_precision', 'order_interval_qc', 'qc_flags'],
        'type_relevance_family': ['order_type_raw', 'clinical_relevance_candidate', 'pharmacologic_family'],
    }
    matrices = []
    for name, columns in dimensions.items():
        table = cross_tab(orders, columns, ok)
        table['matrix'] = name
        table['status_semantic_interpretation'] = 'unvalidated'
        table['status_semantic_evidence'] = 'none'
        matrices.append(table)
    context = cross_tab(orders, ['order_set_raw', 'order_type_raw', 'category_raw', 'drug_type_raw',
                                'drug_raw', 'ingredient_raw', 'order_set_procedure_context', 'order_set_context_rule_id'], ok)
    unresolved = records[records.name_present.astype(bool) & records.classification_confidence.ne('high_exact')].copy()
    unresolved['source_text_redacted'] = False
    for text_col in ('drug_raw', 'ingredient_raw'):
        replaced = unresolved[text_col].map(_restricted_vocabulary_display)
        unresolved[text_col] = replaced.map(lambda item: item[0])
        unresolved['source_text_redacted'] = (
            unresolved['source_text_redacted'] | replaced.map(lambda item: item[1]))
    priority_rows = []
    dims = ['drug_raw', 'ingredient_raw', 'route_canonical',
            'clinical_relevance_candidate', 'review_reasons', 'source_text_redacted']
    for values, part in unresolved.groupby(dims, dropna=False, sort=True):
        priority_rows.append(dict(zip(dims, values), source_provenance='|'.join(sorted(set(part.source))), n_order_rows=int(part.source_kind.eq('BTRIS').sum()),
                                  n_record_rows=len(part), n_distinct_eligible_patients=part.loc[part.eligible_linked, 'linked_patient_pseudonymous'].replace('', pd.NA).nunique() if ok else None,
                                  proposed_ingredient=part.ingredient_canonical.iloc[0],
                                  drug_short_label=part.drug_short_label.iloc[0],
                                  suggested_action='verify exact source ingredient/formulation; no automatic mapping approval',
                                  proposal_status='pending_review'))
    priority = pd.DataFrame(priority_rows, columns=dims + ['source_provenance', 'n_order_rows', 'n_record_rows', 'n_distinct_eligible_patients', 'proposed_ingredient', 'drug_short_label', 'suggested_action', 'proposal_status'])
    priority = priority.sort_values(['n_order_rows', 'n_record_rows', 'drug_raw', 'ingredient_raw'], ascending=[False, False, True, True])
    # Preserve the pre-existing broad QC inventory for backward compatibility.
    # An order-level identity/status warning affects every BTRIS row and must
    # NOT turn the targeted pharmacological conflict list into the entire cohort.
    conflicts = records[records.requires_manual_review.astype(bool) | records.source_kind.eq('BTRIS') | records.qc_flags.ne('')].copy()
    conflicts['status_review_reason'] = conflicts.source_kind.map({'BTRIS': 'status_semantics_unvalidated', 'CTDB': 'not_applicable'})
    important_qc = (r'end_definitely_before_start|ingredient_product_conflict|'
                    r'formulation_route_conflict|protocol_folder_not_confirmed_by_value|'
                    r'cross_protocol_possible_duplicate_retained|'
                    r'same_order_id_revision_or_discordance_retained')
    # Ordinary unknown ingredients remain in the frequency-weighted unmapped
    # review file. This table is for actionable inconsistencies only.
    critical_reasons = (
        r'ingredient_product_conflict|formulation_route_conflict|'
        r'route_required|glucocorticoid_route_requires_review')
    # Empty input frames can give these all-empty columns a float64 dtype.
    # Normalize only the QC selectors to nullable strings before using .str.
    # Keep the original audit columns, values and row grain unchanged.
    review_text = records["review_reasons"].astype("string").fillna("")
    qc_text = records["qc_flags"].astype("string").fillna("")
    targeted = records.loc[
        review_text.str.contains(critical_reasons, regex=True, na=False)
        | qc_text.str.contains(important_qc, regex=True, na=False)
    ].copy()
    targeted['targeted_review_reason'] = [
        '|'.join(filter(None, [
            '|'.join(x for x in row.review_reasons.split('|')
                     if re.fullmatch(critical_reasons, x)),
            '|'.join(x for x in row.qc_flags.split('|')
                     if re.fullmatch(important_qc, x)),
        ])) for row in targeted.itertuples()
    ]
    targeted['review_priority'] = [
        ('high' if any(flag in reason for flag in (
            'ingredient_product_conflict', 'formulation_route_conflict',
            'end_definitely_before_start')) else 'medium')
        for reason in targeted.targeted_review_reason
    ]
    # Future review can use these compact QC-only tables rather than sending
    # pseudonymous row-level records or raw medication vocabulary.
    targeted_aggregate = _group_count_no_ids(
        targeted, ['source', 'targeted_review_reason', 'review_priority',
                   'clinical_category_primary', 'classification_confidence',
                   'drug_short_label'])
    pending_aggregate = _group_count_no_ids(
        records[records.name_present.astype(bool)
                & records.classification_confidence.ne('high_exact')],
        ['source', 'classification_confidence', 'clinical_category_primary',
         'drug_short_label', 'review_reasons'])
    reference = orders[['status_code_raw', 'order_type_raw', 'data_source_raw']].drop_duplicates().rename(columns={'data_source_raw': 'source_system'})
    for column, value in {
        'effective_from': '', 'effective_to': '', 'meaning_in_system': 'unvalidated',
        'supports_prescribed_order': 'unknown', 'supports_dispense': 'not_established',
        'supports_administered': 'not_established', 'supports_discontinued': 'not_established',
        'verified_by': '', 'documentation_reference': 'none', 'review_date': '', 'status': 'pending_review',
    }.items():
        reference[column] = value
    return {
        '08_treatment_classification_dictionary.csv': dictionary(legacy_rules),
        '08_treatment_record_classification_audit.parquet': records,
        '08_treatment_unmapped_priority.csv': priority,
        '08_treatment_pending_summary_aggregate.csv': pending_aggregate,
        '08_treatment_review_summary_aggregate.csv': targeted_aggregate,
        '08_btris_status_type_matrix.csv': pd.concat(matrices, ignore_index=True),
        '08_btris_order_set_context_qc.csv': context,
        '08_treatment_coverage_by_evidence.csv': coverage(records, eligible, linkage, source_ok),
        '08_treatment_classification_conflicts_qc.csv': conflicts,
        '08_treatment_targeted_review_qc.csv': targeted,
        '08_btris_status_semantic_reference.csv': reference,
    }


def summary(records, tables, source_ok, linkage, fingerprints, drift):
    orders = records[records.source_kind.eq('BTRIS')]
    counts = []
    for source in ('CTDB', '11D', '15D'):
        part = records[records.source.eq(source)]
        counts.append(dict(source=source, n_record_rows=len(part) if source_ok[source] else None,
                           n_named_rows=int(part.name_present.astype(bool).sum()) if source_ok[source] else None,
                           n_identified_rows=int(part.ingredient_canonical.ne('').sum()) if source_ok[source] else None,
                           n_unmapped_rows=int(part.classification_confidence.eq('unmapped').sum()) if source_ok[source] else None,
                           n_distinct_order_ids=part.loc[part.order_id_pseudonymous.ne(''), 'order_id_pseudonymous'].nunique() if source_ok[source] and source != 'CTDB' else None,
                           n_distinct_eligible_patients=part.loc[part.eligible_linked, 'linked_patient_pseudonymous'].replace('', pd.NA).nunique() if source_ok[source] and (source == 'CTDB' or linkage['status'] == 'validated') else None))
    c = tables['08_treatment_coverage_by_evidence.csv']
    lookup = {(row.scope, row.metric): row.n_patients for row in c.itertuples()}
    metrics = {'n_patients_any_ctdb_current_name': lookup['CTDB', 'any_documentation'],
               'n_patients_any_btris_order': lookup['BTRIS', 'any_documentation'],
               'n_patients_any_documentation_union': lookup['union', 'any_documentation']}
    for row in c[c.scope.eq('union')].itertuples():
        if row.metric != 'any_documentation':
            metrics['n_patients_' + row.metric] = row.n_patients
    complete = all(source_ok[s] for s in ('11D', '15D'))
    classified = int(orders.classification_confidence.eq('high_exact').sum())
    matrix = tables['08_btris_status_type_matrix.csv']
    reconciliation = {name: int(part.n_order_rows.sum()) == len(orders) for name, part in matrix.groupby('matrix')}
    reconciliation['classified_plus_pending_equals_order_rows'] = classified + int(orders.classification_confidence.ne('high_exact').sum()) == len(orders)
    reconciliation['source_rows_sum_equals_union'] = sum(len(orders[orders.source.eq(s)]) for s in ('11D', '15D')) == len(orders)
    reconciliation['audit_record_ids_unique'] = records.audit_record_id.is_unique
    return dict(classification_version=VERSION, rule_source=RULE_SOURCE, audit_only=True, downstream_integration='NO_GO',
                clinical_category_primary_definitions={
                    PRIMARY_SYSTEMIC: 'candidate systemic SjD-relevant pharmacology; indication unconfirmed',
                    PRIMARY_SYMPTOMATIC: 'candidate oral or ocular SjD symptoms; indication unconfirmed',
                    PRIMARY_CONCOMITANT: 'general/concomitant documented medication',
                    PRIMARY_PROCEDURE: 'procedure/diagnostic candidate only; context unvalidated',
                    PRIMARY_INDETERMINATE: 'unknown role, unvalidated dental support or non-drug device',
                },
                targeted_review_rows=len(tables['08_treatment_targeted_review_qc.csv']),
                study_category_rule_decisions={
                    'citric_acid_monohydrate': 'procedure_related: approved category rule only',
                    'mouth_kote_dry_mouth_treatment': 'symptomatic_sjd: oral: approved category rule only',
                    'scope': 'not patient-specific indication, composition, administration or confirmed exposure',
                    'n_btris_orders_with_approved_category_rule': int(
                        orders.study_category_rule_status.eq('approved_by_investigator').sum()),
                },
                short_label_policy='display_only; canonical identity and original name retained; empty if unresolved or contradictory',
                review_aggregates='restricted/08_treatment_review_summary_aggregate.csv; restricted/08_treatment_pending_summary_aggregate.csv',
                free_text_precaution='free-text contact information masked in priority review; untouched originals restricted to row-level audit; not deidentified',
                source_counts=counts, coverage_metrics=metrics,
                n_order_rows=len(orders) if complete else None,
                n_orders_classified=classified if complete else None,
                n_orders_pending=int(orders.classification_confidence.ne('high_exact').sum()) if complete else None,
                n_ingredient_mentions=int(orders.ingredients_canonical.map(json.loads).map(len).sum()) if complete else None,
                classification_count_definition='classified means high_exact identity; pending includes candidate_review and unmapped; no clinical approval; ingredient mentions can exceed order rows',
                order_interval_qc=orders.order_interval_qc.value_counts().to_dict() if complete else None,
                status_priority_location='restricted/08_btris_status_type_matrix.csv', status_semantics='all_unvalidated',
                reconciliation=reconciliation, reconciliation_scope='loaded_records_only; missing sources do not establish completeness',
                fingerprints=fingerprints, fingerprint_drift=drift,
                top20_unresolved_location='restricted/08_treatment_unmapped_priority.csv (first 20 rows)',
                population_stratum='population_stratum_not_available',
                clinical_evidence='no confirmed indication, prescription semantics, administration, dispensing, adherence or discontinuation')


def report(summary):
    lines = ['# Step 08 pharmacological classification v2.1', '',
             'Documentation of products and orders only. Potential SjD relevance is a candidate, never an individual indication or exposure.', '',
             f"Downstream integration: **{summary['downstream_integration']}**.", '',
             '| Source | Record rows | Named rows | Identified rows | Unmapped rows | Eligible patients (technical linkage) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summary['source_counts']:
        lines.append('| ' + ' | '.join(str(row[k]) for k in ('source', 'n_record_rows', 'n_named_rows', 'n_identified_rows', 'n_unmapped_rows', 'n_distinct_eligible_patients')) + ' |')
    lines += ['', 'Missing sources produce null clinical counts. No real-data reconciliation is claimed from synthetic validation.', '',
              '## Coverage of pharmacological documentation of any nature', '',
              'Unique eligible patients; systemic, ocular, oral and procedure candidates overlap and must not be summed.', '',
              '| Metric | Patients |', '| --- | ---: |']
    for key, value in summary['coverage_metrics'].items():
        lines.append(f'| {key} | {value} |')
    lines += ['', '## Five primary clinical categories (candidate only)', '',
              'systemic_sjd, symptomatic_sjd, concomitant, procedure_related and indeterminate are mutually exclusive at record level, but patient coverage overlaps across categories. Detailed oral, ocular and dental labels remain preserved.', '',
              'Dental/local support remains indeterminate until its SjD-specific purpose is supported. Core Visit is a visit context, not in itself a procedure.', '',
              'Investigator-approved study CATEGORY rules: citric acid monohydrate -> procedure_related; Mouth Kote -> symptomatic_sjd (oral). These do not validate patient-specific indication, composition or exposure. Product identity review flags remain unchanged.', '',
              'drug_short_label is display-only. It does not determine grouping, route equivalence, actual exposure or SjD indication. Empty labels require source review.', '',
              f"Targeted review rows: {summary['targeted_review_rows']} (restricted/08_treatment_targeted_review_qc.csv).", '',
              '## Pending review and QC', '',
              'All observed status codes and order types remain unvalidated. No institutional dictionary is available to this implementation.', '',
              '| Observed status | Order rows |', '| --- | ---: |']
    # Raw vocabularies remain restricted; status priority is available in the private matrix.
    lines += ['| See restricted status/type matrix | See matrix |', '',
              'The first 20 rows of restricted/08_treatment_unmapped_priority.csv provide observed vocabulary, row weights, patient counts and pending proposals. They are intentionally restricted.', '',
              'Interval QC: reverse < 0 seconds; zero = 0; short > 0 and < 24 hours; plausible >= 24 hours; missing = absent endpoint; uncertain = partial/unparsed endpoint or incompatible timezone awareness. These are administrative interval labels, not clinically validated durations.', '',
              'Original rows, dates and all duplicate/revision candidates are retained. No therapeutic episodes are constructed.',
              'Route, product and ingredient conflicts require review; Citric Acid use and Mouth Kote composition remain pending.', '',
              f"Fingerprint comparison: `{summary['fingerprint_drift']}`.",
              f"Reconciliation (loaded records only): `{summary['reconciliation']}`.", '',
              '| Decision | Status |', '| --- | --- |',
              '| General inventory | Report with source traceability |',
              '| Ingredient/product identity | Report with confidence and conflicts |',
              '| Potential clinical relevance | Candidate only |',
              '| Prescription | Source semantics require institutional validation |',
              '| Administration, received dose, duration, adherence, interruption, response, causality | Not approved |',
              '| Integration to Step 10/14/15 or studies | NO_GO |', '']
    return '\n'.join(lines)


def fingerprint_drift(previous_manifest, infos, dictionary_sha256):
    if previous_manifest is None:
        return dict(status='no_previous_manifest', changed_inputs=[], dictionary_changed=None)
    previous = previous_manifest.get('inputs', {})
    changed = [name for name, info in infos.items() if name in previous and
               previous[name].get('sha256') != info.get('sha256')]
    prior_dictionary = previous_manifest.get('classification_dictionary_sha256')
    return dict(status='changed' if changed or (prior_dictionary and prior_dictionary != dictionary_sha256) else 'unchanged' if prior_dictionary else 'previous_dictionary_unavailable',
                changed_inputs=changed, dictionary_changed=prior_dictionary != dictionary_sha256 if prior_dictionary else None)


def dictionary_hash(legacy_rules):
    payload = dictionary(legacy_rules).to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()

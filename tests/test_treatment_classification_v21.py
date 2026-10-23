"""Synthetic scientific/privacy contracts for the isolated Step 08 extension."""
import json
import stat

import pandas as pd
import pytest

from test_treatment_audit_v2 import audit, args_for, btris, complete_ctdb, ctdb, link, records, ROOT, SECRET

classification = audit.classification


def classify(name, ingredient='', route=''):
    return classification.classify(name, ingredient, route, audit.DRUG_RULES)


@pytest.mark.parametrize('name,route,ingredient,relevance', [
    ('Citric Acid Monohydrate 2% Oral Solution', 'oral', 'citric acid monohydrate', classification.PROCEDURE),
    ('Sodium Fluoride Topical Gel 1.1%', 'topical', 'sodium fluoride', classification.DENTAL),
    ('Ibuprofen 400 mg tablet', 'oral', 'ibuprofen', classification.GENERAL),
    ('Carboxymethylcellulose Ophth Drops 0.5% (REFRESH PLUS)', 'ophthalmic', 'carboxymethylcellulose', classification.OCULAR),
    ('Cevimeline 30 mg capsule', 'oral', 'cevimeline', classification.ORAL),
    ('fluconazole', 'oral', 'fluconazole', classification.INDETERMINATE),
    ('iopamidol', 'iv', 'iopamidol', classification.PROCEDURE),
])
def test_observed_priority_vocabulary_is_specific_and_never_confirms_indication(name, route, ingredient, relevance):
    result = classify(name, route=route)
    assert result['ingredient_canonical'] == ingredient
    assert result['clinical_relevance_candidate'] == relevance
    assert not result['clinical_indication_confirmed']
    assert result['classification_rule_id'].startswith(classification.VERSION)


def test_citric_acid_and_mouth_kote_stay_pending():
    citric = classify('Citric Acid Monohydrate 2% Oral Solution')
    assert citric['classification_confidence'] == 'candidate_review'
    assert citric['clinical_relevance_candidate'] != classification.SYSTEMIC
    assert citric['requires_manual_review']
    mouth = classify('Mouth Kote Dry Mouth Treatment')
    assert mouth['ingredient_canonical'] == ''
    assert json.loads(mouth['ingredients_canonical']) == []
    assert mouth['clinical_relevance_candidate'] == classification.ORAL
    assert 'composition_pending_validation' in mouth['review_reasons']
    tears = classify('Systane')
    assert tears['ingredient_canonical'] == ''
    assert tears['classification_confidence'] == 'candidate_review'
    assert tears['clinical_relevance_candidate'] == classification.OCULAR


def test_routes_local_steroids_and_non_equivalent_molecules():
    eye = classify('carboxymethylcellulose ophthalmic drops')
    oral = classify('carboxymethylcellulose oral solution')
    generic = classify('carboxymethylcellulose')
    assert len({x['product_canonical'] for x in (eye, oral, generic)}) == 3
    assert oral['clinical_relevance_candidate'] == classification.GENERAL
    assert generic['route_canonical'] == 'unresolved'
    eye_c = classify('Restasis', 'cyclosporine')
    systemic_c = classify('Neoral', 'cyclosporine')
    assert eye_c['product_canonical'] != systemic_c['product_canonical']
    assert eye_c['clinical_relevance_candidate'] == classification.OCULAR
    assert systemic_c['clinical_relevance_candidate'] == classification.SYSTEMIC
    assert classify('Myfortic')['ingredient_canonical'] != classify('CellCept')['ingredient_canonical']
    assert classify('prednisolone drops', route='ophthalmic')['pharmacologic_family'] == 'ophthalmic_glucocorticoid'
    assert classify('prednisone')['requires_manual_review']
    assert classify('pilocarpine ophthalmic')['clinical_relevance_candidate'] != classification.ORAL


def test_combinations_conflicts_and_exact_suffix_boundaries():
    combo = classify('acetaminophen/codeine', route='oral')
    assert json.loads(combo['ingredients_canonical']) == ['acetaminophen', 'codeine']
    conflict = classify('Plaquenil', 'methotrexate', 'oral')
    assert conflict['ingredient_canonical'] == ''
    assert conflict['clinical_relevance_candidate'] == classification.INDETERMINATE
    assert 'ingredient_product_conflict' in conflict['review_reasons']
    assert classify('Restasis', 'cyclosporine', 'oral')['route_canonical'] == 'unresolved'
    incomplete = classify('hydroxychloroquine + unknown agent', 'hydroxychloroquine', 'oral')
    assert incomplete['classification_confidence'] == 'candidate_review'
    assert incomplete['clinical_relevance_candidate'] == classification.INDETERMINATE
    assert classify('ibuprofen 200 mg tablet')['product_canonical'] != classify('ibuprofen 400 mg tablet')['product_canonical']
    for text in ('IBUPROFEN 400 MG TABLET', 'ibuprofen 10 units', 'ibuprofen 2%'):
        assert classify(text)['ingredient_canonical'] == 'ibuprofen'
    for text in ('not ibuprofen', 'ibuprofen-like agent', 'ibuprofen + unknown agent', 'ibuprofen/unknown', 'ibuprofen 4 magic'):
        assert classify(text)['classification_confidence'] == 'unmapped'


@pytest.mark.parametrize('code', ['AUC1', 'CANP', 'OPCMP', 'DISC', 'OPODC', 'OPOA', 'AUA1', 'COMPT', 'NEW_UNKNOWN'])
@pytest.mark.parametrize('order_type', ['OnSite', 'Take Home', 'Home Med'])
def test_all_status_codes_and_contexts_remain_unvalidated(code, order_type):
    frame = records([btris(**{'Order Status': code, 'Order Type': order_type, 'Take Home Indicator': 'Y'})])
    enriched = classification.enrich(frame, audit.DRUG_RULES, link())
    row = enriched.iloc[0]
    assert row.status_code_raw == code
    assert row.order_context_candidate == order_type
    assert row.status_semantic_interpretation == 'unvalidated'
    assert row.status_semantic_evidence == 'none'
    assert row.administration_evidence == row.dispense_evidence == 'not_established'


def test_intraday_reverse_zero_short_uncertain_and_missing_are_qc_not_corrections():
    cases = [('03/27/2015 09:30', '03/27/2015 08:30', 'reverse'),
             ('03/27/2015 09:30:00', '03/27/2015 09:30:00', 'zero'),
             ('03/27/2015 09:30', '03/27/2015 10:30', 'short'),
             ('03/27/2015 09:30', '03/28/2015 09:30', 'plausible'),
             ('03/2015', '04/2015', 'uncertain'), ('bad', '2015', 'uncertain'),
             ('03/27/2015', '', 'missing')]
    data = records([btris(**{'Order Date/Start Date': start, 'Order End Date': end}) for start, end, _ in cases])
    result = classification.enrich(data, audit.DRUG_RULES, link())
    assert result.order_interval_qc.tolist() == [item[2] for item in cases]
    assert result.start_raw.tolist() == [item[0] for item in cases]
    assert 'end_definitely_before_start' in result.iloc[0].qc_flags
    assert result.order_end_semantics.eq('order_validity_end_not_confirmed_discontinuation').all()


def test_numeric_order_ids_use_patient_and_source_context_without_deleting_rows():
    eligible = {'123', '456'}
    frame = pd.DataFrame([btris(**{'MRN': 123}), btris(**{'MRN': 123, 'Order Status': 'DISC'}),
                          btris(**{'MRN': 456}), btris(**{'MRN': 123, 'Data Source': 'other'})])
    visits = pd.DataFrame({audit.CTDB_PATIENT_ID: ['123', '456']})
    linkage = audit.validate_serology_style_linkage({'11D': frame}, visits, eligible)
    result = audit.duplicate_qc(records(frame.to_dict('records'), linkage=linkage, eligible=eligible))
    assert result.order_identity.ne('').all()
    assert result.qc_flags.str.contains('same_order_id_revision_or_discordance_retained').tolist() == [True, True, False, False]
    assert len(result) == 4


def test_1208_synthetic_order_rows_reconcile_without_exploding_ingredients(tmp_path):
    args = args_for(tmp_path)
    names = ['acetaminophen/codeine', 'Mouth Kote Dry Mouth Treatment', 'unresolved product']
    left = [btris(**{'Medication Name': names[i % 3], 'Drug Ingredient': None, 'Order ID': f'order-{i}',
                     'Order Status': 'AUC1', 'Order Type': 'OnSite', 'Order Set Name': 'Study visit',
                     'Category': 'medication', 'Drug Type': 'drug'}) for i in range(752)]
    right = [btris(**{'Protocol Number': '15-D-0051', 'Order ID': f'order-{i}', 'Order Status': 'DISC',
                      'Order Type': 'Take Home', 'Order Date/Start Date': '06/08/2015 10:30',
                      'Order End Date': '06/08/2015 09:30' if i < 20 else '06/09/2015 10:30'}) for i in range(456)]
    pd.DataFrame(left).to_parquet(args.btris_11d)
    pd.DataFrame(right).to_parquet(args.btris_15d)
    before = {p: audit.v1.file_hash(p) for p in (args.btris_11d, args.btris_15d, args.input)}
    result = audit.run(args)['classification_v21']
    assert result['n_order_rows'] == 1208
    assert result['n_orders_classified'] + result['n_orders_pending'] == 1208
    assert all(result['reconciliation'].values())
    assert result['order_interval_qc']['reverse'] == 20
    private = args.output_dir / 'restricted'
    classified = pd.read_parquet(private / '08_treatment_record_classification_audit.parquet')
    orders = classified[classified.source_kind.eq('BTRIS')]
    assert len(orders) == 1208 and orders.audit_record_id.is_unique
    assert orders.groupby('source').size().to_dict() == {'11D': 752, '15D': 456}
    assert orders[orders.drug_raw.eq('acetaminophen/codeine')].ingredients_canonical.map(json.loads).map(len).eq(2).all()
    assert orders[orders.drug_raw.eq('unresolved product')].classification_confidence.eq('unmapped').all()
    matrix = pd.read_csv(private / '08_btris_status_type_matrix.csv')
    assert matrix.groupby('matrix').n_order_rows.sum().eq(1208).all()
    context = pd.read_csv(private / '08_btris_order_set_context_qc.csv')
    assert context.n_order_rows.sum() == 1208
    cov = pd.read_csv(private / '08_treatment_coverage_by_evidence.csv')
    assert cov.n_patients.max() <= 4
    assert cov.query("scope == 'union' and metric == 'any_documentation'").n_patients.item() == 1
    assert result['downstream_integration'] == 'NO_GO'
    assert all(audit.v1.file_hash(p) == sha for p, sha in before.items())


def test_rerun_retains_other_outputs_and_key_and_reports_input_dictionary_drift(tmp_path):
    args = args_for(tmp_path)
    first = audit.run(args)
    extra = args.output_dir / 'unrelated.txt'
    extra.write_text('keep unrelated audit product')
    private = args.output_dir / 'restricted'
    ids = pd.read_parquet(private / '08_treatment_record_classification_audit.parquet').linked_patient_pseudonymous.tolist()
    same = audit.run(args)
    assert same['classification_v21']['fingerprint_drift']['status'] == 'unchanged'
    assert same['classification_v21']['fingerprint_drift']['previous_dictionary_file_modified'] is False
    dictionary_path = private / '08_treatment_classification_dictionary.csv'
    dictionary_path.write_text(dictionary_path.read_text() + 'tampered row\n')
    tampered = audit.run(args)['classification_v21']['fingerprint_drift']
    assert tampered['previous_dictionary_file_modified'] is True
    assert extra.read_text() == 'keep unrelated audit product'
    assert ids == pd.read_parquet(private / '08_treatment_record_classification_audit.parquet').linked_patient_pseudonymous.tolist()
    data = pd.read_parquet(args.btris_11d)
    data['Medication Name'] = 'unmapped new product'
    data.to_parquet(args.btris_11d)
    manifest_path = private / '08_treatment_input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['classification_dictionary_sha256'] = 'different-dictionary'
    manifest_path.write_text(json.dumps(manifest))
    changed = audit.run(args)['classification_v21']['fingerprint_drift']
    assert changed['changed_inputs'] == ['11D'] and changed['dictionary_changed'] is True
    assert first['downstream_integration'] == 'NO_GO'


def test_all_new_outputs_private_no_phi_free_text_or_master_changes(tmp_path):
    args = args_for(tmp_path)
    data = pd.read_parquet(args.btris_11d)
    forbidden = ['IDENTIFIABLE SUBJECT XYZ', '1900-12-31', 'INSTRUCTION IDENTIFIABLE', 'NONFORMULARY IDENTIFIABLE']
    for column, value in zip(['Subject Name', 'Date of Birth', 'Order Instruction', 'Non Formulary Info'], forbidden):
        data[column] = value
    data.to_parquet(args.btris_11d)
    protected = [ROOT / 'src/block_A/08_treatment_audit.py', ROOT / 'common.py']
    protected += list((ROOT / 'src').rglob('10_*.py')) + list((ROOT / 'src').rglob('14_*.py')) + list((ROOT / 'src').rglob('15_*.py'))
    protected += list((ROOT / 'src/studies').rglob('*.py')) + list((ROOT / 'src/block_B').rglob('*.py'))
    before = {p: audit.v1.file_hash(p) for p in protected}
    audit.run(args)
    private = args.output_dir / 'restricted'
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    for path in args.output_dir.rglob('*'):
        if not path.is_file():
            continue
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        if path.name == '.pseudonym_key':
            continue
        if path.suffix == '.parquet':
            frame = pd.read_parquet(path)
            assert not {'Subject Name', 'Date of Birth', 'MRN', 'Order Instruction', 'Non Formulary Info'} & set(frame.columns)
            text = frame.to_json()
        else:
            text = path.read_text()
        assert not any(value in text for value in forbidden)
        assert '00123' not in text
    assert all(audit.v1.file_hash(p) == sha for p, sha in before.items())


def test_unsafe_and_colliding_identifiers_block_evidence_coverage(tmp_path):
    args = args_for(tmp_path, crosswalk=False)
    args.linkage_mode = 'serology_normalized_id'
    complete_ctdb([ctdb('123', **{'medications__rx_1_name': 'HCQ'})]).to_parquet(args.input)
    pd.DataFrame({'patient_id': ['123']}).to_csv(args.eligibility, index=False)
    for value in (float(2**54), 12.5):
        pd.DataFrame([btris(**{'MRN': value})]).to_parquet(args.btris_11d)
        pd.DataFrame([btris(**{'MRN': 123})]).to_parquet(args.btris_15d)
        result = audit.run(args)
        assert result['linkage']['status'] == 'blocked_linkage'
        cov = pd.read_csv(args.output_dir / 'restricted/08_treatment_coverage_by_evidence.csv')
        assert cov[cov.scope.eq('union')].n_patients.isna().all()
    pd.DataFrame({'patient_id': ['123', '00123']}).to_csv(args.eligibility, index=False)
    result = audit.run(args)
    assert result['linkage']['n_ctdb_normalization_collisions'] == 1
    assert result['classification_v21']['coverage_metrics']['n_patients_any_documentation_union'] is None or pd.isna(result['classification_v21']['coverage_metrics']['n_patients_any_documentation_union'])


def test_ctdb_slots_and_allergies_stay_separate_and_conflicts_keep_originals(tmp_path):
    args = args_for(tmp_path)
    complete_ctdb([ctdb(**{'medications__rx_1_name': 'Ibuprofen 400 mg tablet',
                          'medications__med2_strt_date': '2015',
                          'past_medical_history__rx_1_name': 'Latex'})]).to_parquet(args.input)
    data = pd.read_parquet(args.btris_11d)
    data['Medication Name'], data['Drug Ingredient'] = 'Plaquenil', 'methotrexate'
    data.to_parquet(args.btris_11d)
    audit.run(args)
    private = args.output_dir / 'restricted'
    source = pd.read_parquet(private / '08_treatment_ctdb_slots_audit_long.parquet')
    classified = pd.read_parquet(private / '08_treatment_record_classification_audit.parquet')
    assert set(source.audit_record_id) == set(classified[classified.source_kind.eq('CTDB')].audit_record_id)
    assert source.name_present.sum() == 1 and len(source) == 2
    assert not classified.drug_raw.eq('Latex').any()
    conflict = classified[classified.source.eq('11D')].iloc[0]
    assert conflict.drug_raw == 'Plaquenil' and conflict.ingredient_raw == 'methotrexate'
    assert 'ingredient_product_conflict' in conflict.review_reasons


@pytest.mark.parametrize('name,route,primary', [
    ('Hydroxychloroquine Sulfate 200 mg tablet', 'oral', classification.PRIMARY_SYSTEMIC),
    ('Cevimeline 30 mg capsule', 'oral', classification.PRIMARY_SYMPTOMATIC),
    ('cycloSPORINE Ophthalmic Emulsion 0.05%', 'ophthalmic', classification.PRIMARY_SYMPTOMATIC),
    ('Ibuprofen 400 mg tablet', 'oral', classification.PRIMARY_CONCOMITANT),
    ('Citric Acid Monohydrate 2% Oral Solution', 'oral', classification.PRIMARY_PROCEDURE),
    ('Sodium Fluoride Topical Gel 1.1%', 'topical', classification.PRIMARY_INDETERMINATE),
    ('unknown product', 'oral', classification.PRIMARY_INDETERMINATE),
])
def test_five_primary_candidate_categories_preserve_detail(name, route, primary):
    out = classify(name, route=route)
    assert out['clinical_category_primary'] == primary
    assert out['clinical_category_detail'] == out['clinical_relevance_candidate']
    assert out['clinical_category_validation'] == 'candidate_only_indication_unconfirmed'
    assert not out['clinical_indication_confirmed']


def test_generic_visit_not_marked_procedure_and_targeted_qc_excludes_generic_warning():
    visits = records([btris(**{'Order Set Name': '11-D-0172 Core Visit',
                               'Medication Name': 'Ibuprofen 400 mg tablet',
                               'Drug Ingredient': 'ibuprofen'}),
                      btris(**{'Order Set Name': 'PFT - Pulmonary Function Tests',
                               'Medication Name': 'Ibuprofen 400 mg tablet',
                               'Drug Ingredient': 'ibuprofen'})])
    enriched = classification.enrich(visits, audit.DRUG_RULES, link())
    assert enriched.order_set_procedure_context.tolist() == ['visit_context_only', 'candidate']
    assert enriched.clinical_category_primary.eq(classification.PRIMARY_CONCOMITANT).all()
    artifacts = classification.artifacts(enriched, audit.DRUG_RULES, {'p1'}, link(),
                                         {'CTDB': True, '11D': True, '15D': True})
    assert len(artifacts['08_treatment_targeted_review_qc.csv']) == 0
    assert len(artifacts['08_treatment_classification_conflicts_qc.csv']) == 2


def test_ophthalmic_emulsion_suffix_exact_matching_and_adversarial_names():
    eye = classify('cycloSPORINE Ophthalmic Emulsion 0.05%', route='ophthalmic')
    assert eye['ingredient_canonical'] == 'cyclosporine'
    assert eye['clinical_category_primary'] == classification.PRIMARY_SYMPTOMATIC
    assert eye['route_canonical'] == 'ophthalmic'
    assert eye['formulation_canonical'] == 'ophthalmic_emulsion'
    assert not eye['clinical_indication_confirmed']
    systemic = classify('cyclosporine oral emulsion', route='oral')
    assert systemic['clinical_category_primary'] == classification.PRIMARY_SYSTEMIC
    assert systemic['formulation_canonical'] == 'oral_emulsion'
    for name in ('cyclosporine ophthalmic emulsifying agent',
                 'cyclosporine ophthalmic emulsion + unknown',
                 'cyclosporine ophthalmic emulsion 0.05% plus unknown'):
        assert classify(name, route='ophthalmic')['classification_confidence'] == 'unmapped'


def test_v213_ophthalmic_routes_and_review_scope():
    eye = classify('Restasis eye drops', 'cyclosporine', 'Topical')
    assert eye['clinical_category_primary'] == classification.PRIMARY_SYMPTOMATIC
    assert eye['route_canonical'] == 'ophthalmic'
    assert 'formulation_route_conflict' not in eye['review_reasons']
    assert 'ophthalmic_formulation_generic_topical_route_review' in eye['review_reasons']
    conflict = classify('Refresh plus', '', 'PO')
    assert conflict['clinical_category_primary'] == classification.PRIMARY_INDETERMINATE
    assert 'formulation_route_conflict' in conflict['review_reasons']
    assert classification._route('in both EYEs') == 'ophthalmic'
    assert classification._route('by mouth') == 'oral'
    glycerin = classify(
        'Carboxymethylcellulose/Glycerin/Polysorbate 80 Ophth Drops 0.5%/1%/0.5%',
        'Ophthalmic Lubricant, Polysorbate 80, Glycerin, Carboxymethylcellulose, Glycerin Topical',
        'in both EYEs')
    assert glycerin['route_canonical'] == 'ophthalmic'
    assert 'formulation_route_conflict' not in glycerin['review_reasons']
    assert 'product_components_unresolved' not in classify(
        'Nystatin Topical Ointment 100,000 Units/gram',
        'Nystatin', 'topical')['review_reasons']


def test_v213_targeted_only_actionable_conflicts_not_every_mapping_candidate():
    rows = records([
        btris(**{'Medication Name': 'unresolved product', 'Drug Ingredient': '',
                 'Order End Date': '06/09/2015',
                 'Order Date/Start Date': '06/08/2015'}),
        btris(**{'Medication Name': 'Plaquenil', 'Drug Ingredient': 'methotrexate',
                 'Order End Date': '06/08/2015 08:30',
                 'Order Date/Start Date': '06/08/2015 09:30'}),
    ])
    enriched = classification.enrich(rows, audit.DRUG_RULES, link())
    tables = classification.artifacts(enriched, audit.DRUG_RULES, {'p1'}, link(),
                                      {'CTDB': True, '11D': True, '15D': True})
    review = tables['08_treatment_targeted_review_qc.csv']
    assert len(review) == 1
    assert review.review_priority.iloc[0] == 'high'
    assert 'ingredient_product_conflict' in review.targeted_review_reason.iloc[0]
    assert 'end_definitely_before_start' in review.targeted_review_reason.iloc[0]
    assert tables['08_treatment_unmapped_priority.csv'].drug_raw.eq(
        'unresolved product').any()


def test_v214_oral_balance_gel_is_mucosal_review_candidate():
    out = classify('Oral Balance Gel', route='topically')
    assert out['clinical_category_primary'] == classification.PRIMARY_SYMPTOMATIC
    assert out['route_canonical'] == 'topical'
    assert 'formulation_route_conflict' not in out['review_reasons']
    assert 'composition_pending_validation' in out['review_reasons']
    assert out['ingredient_canonical'] == ''
    assert not out['clinical_indication_confirmed']


def test_v214_ophthalmic_ointment_abbreviation_not_a_component_separator():
    out = classify('White Petrolatum/Mineral Oil Ophth Oint -', route='ophthalmic')
    assert out['route_canonical'] == 'ophthalmic'
    assert out['clinical_indication_confirmed'] is False
    assert out['clinical_category_primary'] in classification.PRIMARY_CATEGORIES


def test_v215_short_labels_do_not_collapse_routes_or_unverified_products():
    eye = classify('cycloSPORINE Ophthalmic Emulsion 0.05%', route='ophthalmic')
    oral = classify('cyclosporine oral emulsion', route='oral')
    assert eye['drug_short_label'] == 'Cyclosporine (ophthalmic)'
    assert oral['drug_short_label'] == 'Cyclosporine (oral)'
    assert eye['drug_short_label'] != oral['drug_short_label']
    assert classify('Ibuprofen 400 mg tablet', route='oral')['drug_short_label'] == 'Ibuprofen'
    assert classify('Plaquenil', ingredient='methotrexate', route='oral')['drug_short_label'] == ''
    assert classify('unknown product')['drug_short_label'] == ''
    assert classify('Mouth Kote Dry Mouth Treatment')['drug_short_label'] == 'Mouth Kote (composition unverified)'


def test_v215_ophthalmic_oint_abbreviation_preserves_combination_identity():
    out = classify('White Petrolatum/Mineral Oil Ophth Oint -', route='ophthalmic')
    assert out['formulation_canonical'] == 'ophthalmic_ointment'
    assert out['ingredient_canonical'] == 'mineral oil + petrolatum'
    assert out['clinical_category_primary'] == classification.PRIMARY_SYMPTOMATIC
    assert out['drug_short_label'] == 'Mineral oil + Petrolatum (ophthalmic)'
    assert not out['clinical_indication_confirmed']


@pytest.mark.parametrize('product,annotation,expected', [
    ('Sodium Fluoride Topical Gel 1.1%', 'Fluoride, Sodium Fluoride', 'sodium fluoride'),
    ('Carboxymethylcellulose Ophth Drops 0.5% (REFRESH PLUS)',
     'Carboxymethylcellulose, Ophthalmic Lubricant', 'carboxymethylcellulose'),
    ('White Petrolatum/Mineral Oil Ophth Oint -',
     'White Petrolatum/Mineral Oil Combination, Ophthalmic Lubricant, Mineral Oil',
     'mineral oil + petrolatum'),
])
def test_v216_exact_ontology_annotations_reconcile_product_components(product, annotation, expected):
    out = classify(product, ingredient=annotation)
    assert out['ingredient_canonical'] == expected
    assert 'ingredient_unresolved' not in out['review_reasons']
    assert 'ingredient_product_conflict' not in out['review_reasons']
    assert out['source_ingredient_annotation_rule_id'].startswith(classification.VERSION)
    assert not out['clinical_indication_confirmed']


def test_v216_unreviewed_annotation_and_conflicting_product_not_silently_accepted():
    unknown = classify('Sodium Fluoride Topical Gel 1.1%',
                       'Fluoride, Sodium Fluoride, Unknown Compound', 'topical')
    assert 'ingredient_unresolved' in unknown['review_reasons']
    contradictory = classify('Ibuprofen 400 mg tablet',
                             'Carboxymethylcellulose, Ophthalmic Lubricant', 'oral')
    assert 'ingredient_product_conflict' in contradictory['review_reasons']
    assert contradictory['drug_short_label'] == ''
    assert contradictory['clinical_category_primary'] == classification.PRIMARY_INDETERMINATE


def test_v217_priority_masks_contact_details_and_keeps_source_row_original():
    original = 'Non-Formulary Request - product\\nPlease contact Research Nurse at 301-555-0101'
    source = records([btris(**{'Medication Name': original, 'Drug Ingredient': ''})])
    enhanced = classification.enrich(source, audit.DRUG_RULES, link())
    tables = classification.artifacts(enhanced, audit.DRUG_RULES, {'p1'}, link(),
                                      {'CTDB': True, '11D': True, '15D': True})
    priority = tables['08_treatment_unmapped_priority.csv']
    assert priority.source_text_redacted.any()
    assert not priority.drug_raw.astype(str).str.contains('301-555').any()
    assert enhanced.drug_raw.iloc[0] == original
    assert 'linked_patient_pseudonymous' not in tables['08_treatment_pending_summary_aggregate.csv']
    assert 'linked_patient_pseudonymous' not in tables['08_treatment_review_summary_aggregate.csv']


def test_v217_freetext_masking_is_selective():
    safe, flagged = classification._restricted_vocabulary_display(
        'Hydroxychloroquine sulfate tablet 200 mg')
    assert safe == 'Hydroxychloroquine sulfate tablet 200 mg' and not flagged
    masked, flagged = classification._restricted_vocabulary_display(
        'Non-formulary request\\nPlease contact Research Nurse')
    assert flagged and masked.startswith('[redacted')


@pytest.mark.parametrize('name,ingredient,route,category', [
    ('Citric Acid Monohydrate 2% Oral Solution', 'citric acid monohydrate',
     'oral', classification.PRIMARY_PROCEDURE),
    ('Mouth Kote Dry Mouth Treatment', '', 'oral',
     classification.PRIMARY_SYMPTOMATIC),
])
def test_v218_category_rule_signed_off_without_exposure_claims(name, ingredient, route, category):
    out = classify(name, ingredient=ingredient, route=route)
    assert out['clinical_category_primary'] == category
    assert out['study_category_rule_status'] == 'approved_by_investigator'
    assert out['study_category_rule_decision_ref'] == 'investigator_decision_2026-10-22'
    assert out['clinical_category_validation'] == 'candidate_only_indication_unconfirmed'
    assert not out['clinical_indication_confirmed']
    assert out['classification_confidence'] == 'candidate_review'
    assert out['requires_manual_review']


def test_v218_approval_never_overrides_ingredient_conflict_or_other_drug():
    bad = classify('Citric Acid Monohydrate 2% Oral Solution', 'ibuprofen', 'oral')
    assert bad['clinical_category_primary'] == classification.PRIMARY_INDETERMINATE
    assert bad['study_category_rule_status'] == 'candidate_not_adjudicated'
    ordinary = classify('Ibuprofen 400 mg tablet', 'ibuprofen', 'oral')
    assert ordinary['study_category_rule_status'] == 'candidate_not_adjudicated'


def test_v218_dictionary_signoff_keeps_composition_validation_separate():
    dictionary = classification.dictionary(audit.DRUG_RULES)
    citric = dictionary.loc[dictionary.ingredient_canonical.eq('citric acid monohydrate')]
    mouth = dictionary.loc[dictionary.exact_alias.eq('mouth kote dry mouth treatment')]
    assert citric.study_category_rule_status.eq('approved_by_investigator').all()
    assert mouth.study_category_rule_status.eq('approved_by_investigator').all()
    assert mouth.pending_clinical_review.all()


@pytest.mark.parametrize('name,ingredient,category', [
    ('Synthroid', 'levothyroxine', classification.PRIMARY_CONCOMITANT),
    ('Celebrex', 'celecoxib', classification.PRIMARY_CONCOMITANT),
    ('Restasis eye drops', 'cyclosporine', classification.PRIMARY_SYMPTOMATIC),
    ('Azathioprine', 'azathioprine', classification.PRIMARY_SYSTEMIC),
])
def test_v219_ctdb_exact_rules_restricted_to_ctdb(name, ingredient, category):
    ctdb = classification.classify(name, '', '', audit.DRUG_RULES, source_kind='CTDB')
    btris = classification.classify(name, '', '', audit.DRUG_RULES, source_kind='BTRIS')
    assert ctdb['ingredient_canonical'] == ingredient
    assert ctdb['clinical_category_primary'] == category
    assert ctdb['ctdb_exact_alias_rule_id'].startswith(classification.VERSION)
    assert not ctdb['clinical_indication_confirmed']
    assert btris['ctdb_exact_alias_rule_id'] == ''


def test_v219_unreviewed_spelling_is_candidate_not_confirmed():
    typo = classification.classify('Synthyroid', '', '', audit.DRUG_RULES, source_kind='CTDB')
    assert typo['ingredient_canonical'] == 'levothyroxine'
    assert typo['classification_confidence'] == 'candidate_review'
    assert 'ctdb_spelling_variant_requires_review' in typo['review_reasons']
    assert classification.classify('Synthyroid', '', '', audit.DRUG_RULES,
                                   source_kind='BTRIS')['ingredient_canonical'] == ''


def test_v219_generic_ctdb_entries_not_mapped():
    for name in ('eye drops', 'supplements', 'calcium and vitamin D',
                 'Restasis unknown combination'):
        out = classification.classify(name, '', '', audit.DRUG_RULES, source_kind='CTDB')
        assert out['ctdb_exact_alias_rule_id'] == ''


def test_v219_ctdb_whitespace_normalization_is_exact():
    out = classification.classify('  Synthroid   ', '', '', audit.DRUG_RULES,
                                  source_kind='CTDB')
    assert out['ingredient_canonical'] == 'levothyroxine'

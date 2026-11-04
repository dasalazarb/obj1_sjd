"""Invented patients and attestations only; never approve production sources."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.studies.graph._pipeline import load_pipeline
from src.studies.graph.stage03.s03_construct_validation import feature_roles
from src.studies.graph.stage03.s03_n3_candidate_audit import (
    REVIEW_FIELDS, n3_audit_tables, n3_source_inventory, values_for,
)
from src.studies.graph.stage03.s03_n3_validation import evidence_catalog, n3_validation_qc
from src.studies.graph.stage03.s03_n3_incremental import n3_incremental_models

REFERENCE = 'tests/test_graph03_n3_candidate_audit.py:synthetic_n3'
CATALOG = {REFERENCE: dict(verified=True, sha256='a'*64)}


def synthetic_n3(n=100, approved=False, representations=('R0',)):
    rng = np.random.default_rng(213)
    config = deepcopy(load_pipeline()['validation'])
    config['effect_ci_replicates'] = 12
    community = np.repeat([0, 1], n//2)
    baseline = pd.DataFrame(dict(patient_id=[f'SYNTHETIC_ONLY_{i:04d}' for i in range(n)],
        clinical_anchor_date='2020-01-01', is_clinical_baseline=True,
        demo__age_at_baseline=rng.uniform(25, 80, n), demo__sex=rng.choice(['F', 'M'], n),
        demo__protocol=rng.choice(['protocol_a', 'protocol_b'], n),
        pop__status=rng.choice(['Pop1', 'Pop2', 'Pop3'], n),
        essdai__total=rng.normal(size=n), esspri__total=rng.normal(size=n),
        n1=rng.normal(size=n), independent=rng.normal(size=n)+community*1.5,
        lab__invented__text=['positive', 'negative', 'unknown', 'not tested', 'not reported']*(n//5),
        unmapped=rng.normal(size=n)))
    membership = pd.concat([pd.DataFrame(dict(patient_id=baseline.patient_id, representation_id=rep,
        community_id=community, mapper_covered=True, hard_assignment=True, membership_tie=False,
        supported=True, structural_stability_status='stable')) for rep in representations], ignore_index=True)
    features = pd.DataFrame([dict(representation_id=rep, source_column='n1', feature='n1_encoded', included=True)
                             for rep in representations])
    registry = pd.DataFrame([dict(source_column=s, clinical_domain='D07', unit='synthetic_units')
                             for s in ['n1', 'independent', 'lab__invented__text']])
    config['anchors'].append(dict(source_column='independent', type='numeric', unit='synthetic_units'))
    config['n3_validation']['baseline_era_reviews'] = [dict(representation_id=rep, date_column='clinical_anchor_date',
        date_semantics_review_status='VERIFIED', baseline_window='SYNTHETIC_TEST_ONLY', source_reference=REFERENCE)
        for rep in representations]
    config['n3_validation']['measurement_policies'] = [dict(representation_id=rep, source_column='independent',
        origin_review_status='APPROVED', unit_review_status='VERIFIED', unit='synthetic_units', source_reference=REFERENCE)
        for rep in representations]
    if approved:
        for rep in representations: approve_synthetic(config, 'independent', rep)
    manifest = dict(eligible_representation_ids=list(representations), release_status='PARTIAL_OR_EXPLORATORY',
                    run_id='SYNTHETIC_RELEASE', random_seed=456, baseline_definition_id='SYNTHETIC_BASELINE')
    return config, baseline, membership, manifest, features, registry


def approve_synthetic(config, source, rep='R0'):
    config['characterization']['independence_reviews'].append(dict(
        representation_id=rep, source_column=source, lineage_review_status='APPROVED',
        clinical_review_status='APPROVED', origin_review_status='APPROVED', proxy_review_status='APPROVED',
        discovery_exclusion_review_status='APPROVED', temporal_baseline_status='VERIFIED',
        reviewer='SYNTHETIC_REVIEWER_ONLY', source_reference=REFERENCE, baseline_window='SYNTHETIC_TEST_ONLY'))


def audit(fixture, catalog=CATALOG):
    config, baseline, membership, manifest, features, registry = fixture
    roles = feature_roles(features, baseline, config, manifest['eligible_representation_ids'], registry)
    return roles, n3_audit_tables(baseline, membership, roles, features, registry, manifest, config, catalog)


def row(tables, source, rep='R0'):
    return tables['source_inventory'].set_index(['representation_id', 'source_column']).loc[(rep, source)]


def test_complete_505_source_contract_all_representations_and_unmapped():
    f = synthetic_n3(representations=('R0','R1','R2','R3','R4'))
    config, baseline, membership, manifest, features, registry = f
    before = feature_roles(features, baseline, config, manifest['eligible_representation_ids'], registry)
    needed = 505-before.source_column.nunique()
    expanded = pd.concat([baseline, pd.DataFrame({f'unmapped_{i}':np.ones(len(baseline)) for i in range(needed)})], axis=1)
    config['n3_validation']['expected_source_count'] = 505
    roles, tables = audit((config, expanded, membership, manifest, features, registry))
    inventory = tables['source_inventory']
    assert len(inventory) == 2525 and inventory.groupby('representation_id').size().eq(505).all()
    assert not inventory.duplicated(['representation_id','source_column']).any()
    assert inventory.loc[inventory.source_column.str.startswith('unmapped'),'clinical_domain'].eq('UNMAPPED').all()
    pd.testing.assert_frame_equal(roles, feature_roles(features, expanded, config, manifest['eligible_representation_ids'], registry))
    with pytest.raises(ValueError, match='COUNT_CONTRACT'):
        audit((config, baseline, membership, manifest, features, registry))


def test_n1_n2_and_absent_relation_never_promote_and_reviews_are_representation_specific():
    f = synthetic_n3(approved=True, representations=('R0','R2'))
    config, baseline, membership, manifest, features, registry = f
    baseline['lab__anti_ro_ssa__text'] = 'positive'
    baseline['lab__anti_la_ssb__text'] = 'negative'
    baseline['related'] = 1.
    features = pd.concat([features,pd.DataFrame([dict(representation_id='R0',source_column=c,feature=c,included=True)
                                                for c in ['lab__anti_ro_ssa__text','lab__anti_la_ssb__text']])])
    registry = pd.concat([registry,pd.DataFrame([dict(source_column='related',derived_from='n1',clinical_domain='D07')])])
    for s in ['n1','related']: approve_synthetic(config,s)
    _, tables = audit((config, baseline, membership, manifest, features, registry))
    for source in ['n1','related']:
        assert row(tables,source).review_state == 'BLOCKED_DEPENDENT'
    for source in ['lab__anti_ro_ssa__text','lab__anti_la_ssb__text']:
        assert row(tables,source).tier_original == 'N1_DEFINING'
        assert row(tables,source,'R2').tier_original == 'PENDING_REVIEW'
        assert row(tables,source,'R2').review_state != 'N3_REVIEWED'
    assert row(tables,'unmapped').relationship_type == 'no_declared_relation'
    assert row(tables,'unmapped').review_state == 'PENDING_REVIEW'


@pytest.mark.parametrize('field', REVIEW_FIELDS)
def test_partial_or_foreign_review_does_not_authorize(field):
    f = synthetic_n3(approved=True)
    del f[0]['characterization']['independence_reviews'][0][field]
    _, tables = audit(f)
    assert row(tables,'independent').review_state != 'N3_REVIEWED'


def test_unverifiable_reference_and_wrong_representation_fail_closed():
    f = synthetic_n3(approved=True)
    for catalog in [{}, {REFERENCE:dict(verified=True,sha256='invalid')}]:
        _, tables = audit(f,catalog)
        assert row(tables,'independent').review_state == 'PENDING_REVIEW'
    f[0]['characterization']['independence_reviews'][0]['representation_id']='R2'
    _, tables = audit(f)
    assert row(tables,'independent').review_state == 'PENDING_REVIEW'


def test_verified_new_anchor_uses_existing_review_contract_without_mutating_roles():
    f = synthetic_n3(approved=True)
    f[0]['anchors'] = [a for a in f[0]['anchors'] if a['source_column'] != 'independent']
    roles, tables = audit(f)
    assert roles.set_index('source_column').loc['independent','tier']=='PENDING_REVIEW'
    assert row(tables,'independent').tier_original=='PENDING_REVIEW'
    assert row(tables,'independent').review_state=='N3_REVIEWED'


def test_assays_composites_proxies_and_clinical_alerts_are_bidirectional():
    f = synthetic_n3()
    config, baseline, membership, manifest, features, registry = f
    for s in ['lab__igg__value','lab__igg__text','lab__igg_subclass_1__value','pro__sf36_pcs','component','total','ovl__status']:
        baseline[s]=1.
    features = pd.concat([features,pd.DataFrame([dict(representation_id='R0',source_column=c,feature=c,included=True)
                                                for c in ['lab__igg__value','pro__sf36_pcs','component']])])
    registry = pd.concat([registry,pd.DataFrame([
        dict(source_column='lab__igg__text',semantic_feature_id='lab__igg'),
        dict(source_column='lab__igg__value',semantic_feature_id='lab__igg'),
        dict(source_column='total',derived_from='component')])])
    _, tables = audit((config, baseline, membership, manifest, features, registry))
    assert row(tables,'lab__igg__text').review_state=='BLOCKED_DEPENDENT'
    assert row(tables,'total').review_state=='BLOCKED_DEPENDENT'
    assert 'IGG_SUBCLASS' in row(tables,'lab__igg_subclass_1__value').clinical_proxy_alerts
    assert 'ESSPRI_PAIN_DRYNESS_FATIGUE' in row(tables,'esspri__total').clinical_proxy_alerts
    assert row(tables,'ovl__status').review_state=='PENDING_REVIEW'  # Absence of a discovery edge is not approval.
    config['characterization']['relationships'].append(dict(source_column='independent',target_source='n1',
        relationship_type='clinical_proxy_reviewed',review_status='APPROVED',reviewer='SYNTHETIC_ONLY',source_reference=REFERENCE))
    _, tables = audit((config, baseline, membership, manifest, features, registry))
    assert row(tables,'independent').review_state=='BLOCKED_DEPENDENT'


def test_unverified_lab_categories_do_not_invent_test_denominators():
    f = synthetic_n3()
    _, tables = audit(f)
    r=row(tables,'lab__invented__text')
    assert r.n_observed_unique==40 and r.n_missing==60 and r.n_valid_categories==2
    assert pd.isna(r.n_tested) and pd.isna(r.n_negative) and pd.isna(r.n_not_tested)
    f[0]['n3_validation']['measurement_policies'].append(dict(representation_id='R0',source_column='lab__invented__text',
        origin_review_status='APPROVED',category_review_status='VERIFIED',source_reference=REFERENCE,
        category_convention=dict(positive=['positive'],negative=['negative'],unknown=['unknown','not reported'],not_tested=['not tested'])))
    _, tables = audit(f)
    r=row(tables,'lab__invented__text')
    assert (r.n_tested,r.n_positive,r.n_negative,r.n_unknown,r.n_not_tested)==(40,20,20,40,20)
    assert r.n_observed_unique==40
    cells=json.loads(r.coverage_by_community)
    assert sum(c['n_observed'] for c in cells)==40  # Count once per unique patient, before categories.


@pytest.mark.parametrize('dtype',['bool','boolean'])
def test_numeric_declared_booleans_are_binary_and_missing_is_preserved(dtype):
    raw=pd.Series([True,False,True,False] if dtype=='bool' else [True,False,pd.NA,False],dtype=dtype)
    values, kind, states=values_for(raw,'numeric')
    assert kind=='binary' and values.dtype=='float64'
    assert values.eq(0).sum()==2
    assert values.isna().sum()==(1 if dtype=='boolean' else 0)
    assert pd.isna(states['n_negative'])


def test_baseline_date_name_and_unit_name_are_not_approvals_and_no_sensitive_text_leaks():
    f=synthetic_n3(approved=True)
    f[0]['n3_validation']['measurement_policies']=[]
    _, tables=audit(f)
    assert row(tables,'independent').review_state=='PENDING_REVIEW'
    f=synthetic_n3(approved=True)
    record=f[0]['characterization']['independence_reviews'][0]
    record['baseline_window']='2020-01-01 SYNTHETIC_ONLY_0000'
    record['reviewer']='SYNTHETIC_ONLY_0001'
    roles, tables=audit(f)
    result=n3_incremental_models(f[1],f[2],roles,tables['source_inventory'],f[3],f[0],CATALOG)
    qc=n3_validation_qc({**tables,**result},f[1],f[2],f[3],{},True)
    assert qc['no_patient_identifiers_or_exact_dates_exported']
    for table in [*tables.values(), *result.values()]:
        payload=table.to_csv(index=False)
        assert 'patient_id' not in table and '2020-01-01' not in payload
        assert not any(patient in payload for patient in f[1].patient_id)


def test_inventory_deterministic_under_row_order_and_rejects_duplicate_patients():
    f=synthetic_n3(representations=('R0','R2'))
    _, first=audit(f)
    config, baseline, membership, manifest, features, registry=f
    _, second=audit((config,baseline.sample(frac=1,random_state=1),membership.sample(frac=1,random_state=2),manifest,features,registry))
    pd.testing.assert_frame_equal(first['source_inventory'],second['source_inventory'])
    with pytest.raises(ValueError,match='NONUNIQUE'):
        audit((config,pd.concat([baseline,baseline.iloc[:1]]),membership,manifest,features,registry))


def test_evidence_catalog_checks_files_symbols_and_rejects_arbitrary_paths(tmp_path):
    root=tmp_path
    (root/'docs').mkdir(); (root/'docs/review.md').write_text('documented_measurement')
    config={'reviews':[{'source_reference':r} for r in ['docs/review.md:documented_measurement',
        'docs/review.md:missing_symbol','/etc/passwd','docs/../../etc/passwd','https://example.com/review']]}
    catalog=evidence_catalog(config,root)
    assert set(catalog)=={'docs/review.md:documented_measurement'}
    assert len(next(iter(catalog.values()))['sha256'])==64


def test_no_hard_support_preserves_audit_and_missing_measurements_are_not_negative():
    f=synthetic_n3(approved=True)
    f[2]['supported']=False
    _,tables=audit(f)
    assert tables['source_inventory'].n_hard.eq(0).all()
    assert tables['source_inventory'].n_observed_unique.eq(0).all()
    assert tables['endpoint_support'].status.eq('NOT_ESTIMABLE').all()
    assert tables['source_inventory'].n_baseline_total.eq(100).all()


def test_unparsed_continuous_units_block_reviewed_models_without_imputation():
    f=synthetic_n3(approved=True)
    f[1]['independent']=f[1]['independent'].astype(object)
    f[1].loc[0,'independent']='12 mixed_units'
    _,tables=audit(f)
    r=row(tables,'independent')
    assert r.n_unparsed_numeric_values==1 and r.review_state=='PENDING_REVIEW'
    assert 'UNPARSED_NUMERIC_VALUES' in r.blocking_reasons


def test_invalid_documentary_review_cannot_bypass_new_gate_through_legacy_consumers():
    from src.studies.graph.stage03.s03_n3_validation import enforce_n3_review_gates
    from src.studies.graph.stage03.s03_profiles import profile_tables
    from src.studies.graph.stage03.s03_incremental_value import incremental_models
    f=synthetic_n3(approved=True)
    f[0]['characterization']['independence_reviews'][0]['source_reference']='nonexistent_reference'
    roles,tables=audit(f)
    assert roles.set_index('source_column').loc['independent','tier']=='N3_REVIEWED'
    gated=enforce_n3_review_gates(roles,tables['source_inventory'])
    assert gated.tier.equals(roles.tier)
    assert not gated.set_index('source_column').loc['independent','inferential_eligible']
    anchors=profile_tables(f[1],f[2],gated,f[3],f[0],f[3]['random_seed'])['anchors']
    incremental=incremental_models(f[1],f[2],gated,f[3],f[0])
    for result in [anchors,incremental]:
        assert result.p_value.isna().all() and result.status.eq('NOT_ESTIMABLE').all()


def test_inspection_name_without_clinical_metadata_does_not_grant_candidate_eligibility():
    f=synthetic_n3()
    f[1]['ext__biopsy_focus_score']=2.
    _,tables=audit(f)
    r=row(tables,'ext__biopsy_focus_score')
    assert r.is_inspection_source and not r.is_outcome_candidate_for_review
    assert r.review_state=='PENDING_REVIEW'
    assert 'CLINICAL_DOMAIN_ELIGIBILITY_PENDING' in r.blocking_reasons
    assert 'ext__biopsy_focus_score' in set(tables['endpoint_support'].source_column)

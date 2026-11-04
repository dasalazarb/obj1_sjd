"""Synthetic source-level acceptance: no records or approvals for real patients."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
from src.studies.graph import _pipeline as pipe
from src.studies.graph.stage03.s03_construct_validation import feature_roles, assert_independent
from src.studies.graph.stage03.s03_characterization import characterization_tables, statistical_role_gates, characterization_qc
from src.studies.graph.stage03.s03_profiles import profile_tables
from src.studies.graph.stage03.s03_incremental_value import incremental_models
from src.studies.graph.stage03.s03_uncertainty import discovery_sensitivity
from src.studies.graph.stage03.s03_reporting import characterization_figures, safe_characterization_panels


def approve_synthetic(config, source, representation='R0'):
    """Attest invented observations only, never production config or real data."""
    config['characterization']['independence_reviews'].append(dict(source_column=source,representation_id=representation,
        lineage_review_status='APPROVED',clinical_review_status='APPROVED',origin_review_status='APPROVED',
        proxy_review_status='APPROVED',discovery_exclusion_review_status='APPROVED',temporal_baseline_status='VERIFIED',
        reviewer='synthetic_fixture_reviewer',source_reference='SYNTHETIC_TEST_ONLY:invented_independent_measurement',
        baseline_window='SYNTHETIC_TEST_ONLY:at_frozen_index'))


def fixture():
    cfg=deepcopy(pipe.load_pipeline()['validation']); cfg['effect_ci_replicates']=10
    n=60; rng=np.random.default_rng(42)
    baseline=pd.DataFrame({'patient_id':[f'invented_{i}' for i in range(n)],
        'clinical_anchor_date':'2020-01-01','pop__status':['Pop1','Pop2']*(n//2),
        'essdai__total':rng.normal(size=n),'esspri__total':rng.normal(size=n),
        'independent':rng.normal(size=n),'unused':rng.normal(size=n)})
    membership=pd.DataFrame(dict(patient_id=baseline.patient_id,representation_id='R0',community_id=np.repeat([0,1],n//2),
        hard_assignment=True,mapper_covered=True,supported=True,membership_tie=False,structural_stability_status='stable'))
    cfg['anchors'].append(dict(source_column='independent',type='numeric',unit='synthetic_units'))
    meta=dict(eligible_representation_ids=['R0'],release_status='PARTIAL_OR_EXPLORATORY')
    features=pd.DataFrame([dict(representation_id='R0',source_column='unused',feature='unused',included=True)])
    return cfg,baseline,membership,meta,features


def test_frozen_panel_n1_examples_are_not_hardcoded_and_sources_are_unique():
    cfg,baseline,membership,meta,features=fixture()
    sources=['ext__ocular_schirmer_min','ext__ocular_oxford_max','ext__salivary_flow_unstimulated',
             'lab__igg__value','lab__complement_c3__value','pro__sf36_pcs','pro__profad_total','pro__mdafs_global']
    for source in sources: baseline[source]=1.
    features=pd.DataFrame([dict(representation_id=rep,source_column=c,feature=c+'__encoded',included=True)
                           for rep in ['R0','R2'] for c in sources])
    roles=feature_roles(features,baseline,cfg,['R0','R2'])
    assert roles.loc[roles.source_column.isin(sources),'tier'].eq('N1_DEFINING').all()
    assert roles.loc[roles.source_column.isin(sources),'matched_encoded_features'].str.endswith('__encoded').all()
    assert not roles.duplicated(['representation_id','source_column']).any()
    features.loc[features.representation_id.eq('R2') & features.source_column.eq(sources[0]),'included']=False
    roles=feature_roles(features,baseline,cfg,['R0','R2'])
    assert roles.query('representation_id=="R2" and source_column==@sources[0]').tier.iloc[0]=='PENDING_REVIEW'


def test_typed_lineage_real_assay_variants_components_cycles_and_generic_groups():
    cfg,baseline,membership,meta,features=fixture()
    for source in ['lab__igg__value','lab__igg__text','unrelated','component','total','cycle_a','cycle_b']:
        baseline[source]=2.
    registry=pd.DataFrame([
        dict(source_column='lab__igg__value',semantic_feature_id='lab__igg',clinical_domain='D04',family='LAB',derivation_group='LAB',instrument_id='LAB',subdomain='LAB'),
        dict(source_column='lab__igg__text',semantic_feature_id='lab__igg',clinical_domain='D04'),
        dict(source_column='unrelated',semantic_feature_id='LAB',clinical_domain='D04',family='LAB',derivation_group='LAB',instrument_id='LAB',subdomain='LAB'),
        dict(source_column='total',derived_from='component'),dict(source_column='cycle_a',derived_from='cycle_b'),
        dict(source_column='cycle_b',derived_from='cycle_a|lab__igg__text')])
    features=pd.DataFrame([dict(representation_id='R0',source_column=c,feature=c+'__binary',included=True) for c in ['lab__igg__value','component']])
    roles=feature_roles(features,baseline,cfg,['R0'],registry).set_index('source_column')
    assert roles.loc['lab__igg__value','tier']=='N1_DEFINING'
    for c in ['lab__igg__text','total','cycle_a','cycle_b']: assert roles.loc[c,'tier']=='N2_RELATED'
    assert roles.loc['unrelated','tier']=='PENDING_REVIEW'
    assert 'same_raw_source' in roles.loc['lab__igg__text','lineage_evidence']
    assert 'derived_from' in roles.loc['cycle_a','lineage_evidence']


def test_composite_pop_does_not_join_its_independent_inputs():
    cfg,baseline,membership,meta,features=fixture()
    features=features.assign(source_column='essdai__total',feature='essdai__total')
    roles=feature_roles(features,baseline,cfg,['R0']).set_index('source_column')
    assert roles.loc['pop__status','tier']=='N2_RELATED'
    assert roles.loc['esspri__total','tier']=='N3_CANDIDATE'
    assert roles.loc['esspri__total','conflicting_defining_sources']==''


def test_pending_review_blocks_every_inferential_consumer_even_with_stable_support():
    cfg,baseline,membership,meta,features=fixture()
    roles=feature_roles(features,baseline,cfg,['R0'])
    tables=profile_tables(baseline,membership,roles,meta,cfg,9)
    anchors=tables['anchors']
    assert anchors.tier.eq('N3_CANDIDATE').all()
    assert anchors.reason.eq('INDEPENDENCE_REVIEW_PENDING').all()
    assert anchors.status.eq('NOT_ESTIMABLE').all()
    assert anchors[['effect','ci_low','ci_high','p_value','q_value']].isna().all().all()
    assert_independent(anchors)
    result=incremental_models(baseline,membership,roles,meta,cfg)
    assert result.reason.eq('INDEPENDENCE_REVIEW_PENDING').all() and result.p_value.isna().all()
    sensitivity=discovery_sensitivity(pd.DataFrame(),baseline,roles,anchors,pipe.load_pipeline()['experiments'],cfg)
    assert sensitivity.reason.eq('INDEPENDENCE_REVIEW_PENDING').all()
    with pytest.raises(ValueError,match='Unreviewed'): assert_independent(anchors.assign(status='COMPLETE',p_value=.01))
    with pytest.raises(ValueError,match='Circular'): assert_independent(roles.loc[roles.tier.eq('N1_DEFINING')])


def test_reviewed_anchor_passes_only_with_support_stability_and_no_new_proxy():
    cfg,baseline,membership,meta,features=fixture()
    approve_synthetic(cfg,'independent')
    roles=feature_roles(features,baseline,cfg,['R0'])
    assert roles.set_index('source_column').loc['independent','tier']=='N3_REVIEWED'
    tables=profile_tables(baseline,membership,roles,meta,cfg,2)
    tested=tables['anchors'].loc[tables['anchors'].source_column.eq('independent')]
    assert tested.status.eq('COMPLETE').all() and tested.p_value.notna().all()
    assert_independent(tables['anchors'])
    provisional=membership.assign(structural_stability_status='provisional')
    gated=statistical_role_gates(roles,baseline,provisional,meta,cfg)
    assert not gated.set_index('source_column').loc['independent','inferential_eligible']
    tested=profile_tables(baseline,provisional,gated,meta,cfg,2)['anchors']
    assert tested.status.eq('NOT_ESTIMABLE').all() and tested.p_value.isna().all()
    low=baseline.copy();low.loc[2:,'independent']=np.nan
    tested=profile_tables(low,membership,feature_roles(features,low,cfg,['R0']),meta,cfg,2)['anchors']
    assert tested.loc[tested.source_column.eq('independent'),'reason'].eq('INSUFFICIENT_OBSERVED_GROUP_OR_REST').all()
    cfg['characterization']['relationships'].append(dict(source_column='independent',target_source='unused',
        relationship_type='clinical_proxy_reviewed',review_status='APPROVED',reviewer='synthetic_only',source_reference='SYNTHETIC_PROXY'))
    roles=feature_roles(features,baseline,cfg,['R0']).set_index('source_column')
    assert roles.loc['independent','tier']=='N2_RELATED' and not roles.loc['independent','inferential_eligible']


@pytest.mark.parametrize('missing',['reviewer','source_reference','baseline_window','origin_review_status','proxy_review_status',
                                      'discovery_exclusion_review_status','temporal_baseline_status','lineage_review_status','clinical_review_status'])
def test_partial_review_never_grants_independence(missing):
    cfg,baseline,membership,meta,features=fixture();approve_synthetic(cfg,'independent')
    del cfg['characterization']['independence_reviews'][0][missing]
    row=feature_roles(features,baseline,cfg,['R0']).set_index('source_column').loc['independent']
    assert row.tier=='N3_CANDIDATE' and not row.inferential_eligible


def test_absent_all_missing_adjustments_and_representation_specific_reviews():
    cfg,baseline,membership,meta,features=fixture()
    baseline['demo__sex']='female';baseline['independent']=np.nan
    approve_synthetic(cfg,'independent');approve_synthetic(cfg,'esspri__total','R0')
    features=pd.concat([features,pd.DataFrame([dict(representation_id='R0',source_column='demo__sex',feature='sex=F',included=True)])])
    roles=feature_roles(features,baseline,cfg,['R0','R2']).set_index(['representation_id','source_column'])
    assert roles.loc[('R0','demo__sex'),'tier']=='N1_DEFINING'
    assert roles.loc[('R2','demo__sex'),'tier']=='PENDING_REVIEW'
    assert roles.loc[('R0','independent'),'tier']=='UNAVAILABLE'
    assert roles.loc[('R0','esspri__total'),'tier']=='N3_REVIEWED'
    assert roles.loc[('R2','esspri__total'),'tier']=='N3_CANDIDATE'
    assert roles.loc[('R0','ext__ocular_schirmer_min'),'tier']=='UNAVAILABLE'


def test_observation_categories_denominators_and_unique_source_counts():
    cfg,baseline,membership,meta,features=fixture()
    baseline['assay']=['negative','positive','unknown','not tested',None]*12
    cfg['clinical_marker_panels'].append(dict(source_column='assay',label='Invented assay',unit='category',clinical_domain='D04'))
    features=pd.DataFrame([dict(representation_id='R0',source_column='assay',feature=f'assay_{i}',included=True) for i in range(3)])
    roles=feature_roles(features,baseline,cfg,['R0'])
    anchors=profile_tables(baseline,membership,roles,meta,cfg,2)['anchors']
    tables=characterization_tables(baseline,membership,roles,meta,cfg,anchors)
    matrix=tables['clinical_matrix'];assay=matrix.loc[matrix.source_column.eq('assay')]
    assert not matrix.source_column.eq('pop__status').any()
    assert not matrix.clinical_domain.isin(['D08','D09']).any()
    assert (matrix.n_observed+matrix.n_missing).eq(matrix.n_group).all()
    assert assay.n_group.eq(30).all() and assay.n_observed.eq(12).all()
    assert assay.n_pos.eq(6).all() and assay.n_neg.eq(6).all() and assay.n_tested.eq(12).all()
    assert assay.n_unknown.eq(6).all() and assay.n_not_tested.eq(6).all()
    count=tables['tier_counts'].query('tier=="N1_DEFINING" and clinical_domain=="D04"').iloc[0]
    assert count.n_sources==1 and count.n_encoded_features==3 and count.n_patients_evaluable_hard==24
    qc=characterization_qc(tables,baseline,membership,{'graph02_run_id':'synthetic','graph02_release_sha256':'synthetic_sha'},True)
    assert qc['status']=='PASS'
    for table in tables.values():
        assert 'patient_id' not in table
        assert not any(p in table.to_csv(index=False) for p in baseline.patient_id)


def test_whole_panel_suppression_and_separate_tier_figures(tmp_path):
    cfg,baseline,membership,meta,features=fixture()
    cfg['clinical_marker_panels']=[dict(source_column='independent',label='Invented independent',unit='synthetic'),
                                  dict(source_column='unused',label='Invented discovery',unit='synthetic')]
    cfg['characterization']['figure_sources']=['unused']
    roles=feature_roles(features,baseline,cfg,['R0'])
    anchors=profile_tables(baseline,membership,roles,meta,cfg,2)['anchors']
    matrix=characterization_tables(baseline,membership,roles,meta,cfg,anchors)['clinical_matrix']
    matrix.loc[matrix.source_column.eq('unused') & matrix.community_id.eq('0'),['n_observed','n_missing','median']]=[3,27,99999]
    safe=safe_characterization_panels(matrix,5)
    assert not safe.source_column.eq('unused').any()
    characterization_figures(matrix,roles,tmp_path,cfg,5)
    for name in ['N1_R0','N1_R2','N2_R0','N2_R2','N3_R0','N3_R2','role_crosswalk','availability']:
        for ext in ['csv','png','pdf']: assert (tmp_path/f'03_graph_characterization_{name}.{ext}').is_file()
    n3=pd.read_csv(tmp_path/'03_graph_characterization_N3_R0.csv')
    assert n3.inference_status.eq('REVIEW_PENDING').all()
    availability=pd.read_csv(tmp_path/'03_graph_characterization_availability.csv')
    assert not availability.source_column.eq('unused').any()
    assert pd.read_csv(tmp_path/'03_graph_characterization_N1_R0.csv').empty
    for path in tmp_path.glob('*.csv'): assert '99999' not in path.read_text() and 'patient_id' not in path.read_text()


def test_no_supported_hard_assignments_retains_tiers_and_empty_products(tmp_path):
    cfg,baseline,membership,meta,features=fixture()
    membership['supported']=False
    roles=feature_roles(features,baseline,cfg,['R0'])
    tables=profile_tables(baseline,membership,roles,meta,cfg,2)
    assert tables['anchors'].status.eq('NOT_ESTIMABLE').all()
    assert_independent(tables['anchors'])
    products=characterization_tables(baseline,membership,roles,meta,cfg,tables['anchors'])
    assert products['clinical_matrix'].empty and products['evidence_summary'].empty
    assert products['tier_counts'].n_hard_supported.eq(0).all()
    assert products['tier_counts'].n_patients_evaluable_hard.eq(0).all()
    safe=safe_characterization_panels(products['clinical_matrix'],5)
    assert safe[['representation_id','source_column']].empty
    characterization_figures(products['clinical_matrix'],roles,tmp_path,cfg,5)
    assert len(list(tmp_path.glob('*.png')))==8


def test_ambiguous_encoded_features_fail_closed():
    cfg,baseline,membership,meta,features=fixture()
    ambiguous=pd.concat([features,features.assign(source_column='independent')])
    with pytest.raises(ValueError,match='Ambiguous frozen'): feature_roles(ambiguous,baseline,cfg,['R0'])


def test_broad_metadata_and_unreviewed_proxy_are_not_evidence():
    cfg,baseline,membership,meta,features=fixture()
    registry=pd.DataFrame([dict(source_column=source,semantic_feature_id='D04',derivation_group='LAB',
        clinical_domain='D04',family='LAB',instrument_id='D04',subdomain='D04') for source in ['unused','independent']])
    cfg['characterization']['relationships'].append(dict(source_column='independent',target_source='unused',
        relationship_type='clinical_proxy_reviewed',review_status='NOT_REVIEWED',source_reference='synthetic_pending_proxy'))
    roles=feature_roles(features,baseline,cfg,['R0'],registry).set_index('source_column')
    assert roles.loc['independent','tier']=='N3_CANDIDATE'
    assert roles.loc['independent','conflicting_defining_sources']==''


@pytest.mark.parametrize('values,dtype,declared,expected_type,expected_counts', [
    pytest.param([True,False], 'bool', 'numeric', 'binary', (6,6,0,0), id='bool-declared-numeric'),
    pytest.param([True,False], 'bool', 'binary', 'binary', (6,6,0,0), id='bool-declared-binary'),
    pytest.param([True,False], 'bool', 'source_type', 'binary', (6,6,0,0), id='bool-source-type'),
    pytest.param([True,False,pd.NA], 'boolean', 'numeric', 'binary', (4,4,0,0), id='nullable-declared-numeric'),
    pytest.param([True,False,pd.NA], 'boolean', 'binary', 'binary', (4,4,0,0), id='nullable-declared-binary'),
    pytest.param([pd.NA], 'boolean', 'numeric', 'binary', (0,0,0,0), id='nullable-all-missing'),
    pytest.param(['positive','negative','unknown','not tested'], 'string', 'source_type', 'binary', (3,3,3,3), id='text-assay'),
    pytest.param(['positive','negative','unknown','not tested'], 'string', 'binary', 'binary', (3,3,3,3), id='text-declared-binary'),
    pytest.param([1.,2.,3.,4.], 'float64', 'numeric', 'numeric', None, id='continuous'),
    pytest.param([1.,2.,3.,4.], 'float64', 'source_type', 'numeric', None, id='inferred-continuous'),
    pytest.param([np.nan], 'float64', 'numeric', 'numeric', None, id='continuous-all-missing'),
    pytest.param(['true','false'], 'string', 'source_type', 'categorical', None, id='unvalidated-text-booleans'),
])
def test_characterization_value_types_and_observation_states(values,dtype,declared,expected_type,expected_counts):
    cfg,baseline,membership,meta,features=fixture()
    baseline=baseline.iloc[:24].copy()
    membership=membership.iloc[:24].copy().assign(community_id=np.repeat([0,1],12))
    baseline['assay']=pd.Series(values*(24//len(values)),dtype=dtype)
    cfg['clinical_marker_panels'].append(dict(source_column='assay',label='Invented assay',
        unit='synthetic_units',clinical_domain='D04',type=declared))
    features=features.assign(source_column='assay',feature='assay__encoded')
    roles=feature_roles(features,baseline,cfg,['R0'])
    original_roles=roles.copy(deep=True)
    tables=characterization_tables(baseline,membership,roles,meta,cfg,pd.DataFrame())
    assay=tables['clinical_matrix'].loc[lambda rows:rows.source_column.eq('assay')]
    assert not assay.empty and assay.data_type.eq(expected_type).all()
    assert (assay.n_observed+assay.n_missing).eq(assay.n_group).all()
    if expected_counts is not None:
        pos,neg,unknown,not_tested=expected_counts
        assert assay.n_pos.eq(pos).all() and assay.n_neg.eq(neg).all()
        assert assay.n_tested.eq(pos+neg).all() and assay.n_observed.eq(pos+neg).all()
        assert assay.n_unknown.eq(unknown).all() and assay.n_not_tested.eq(not_tested).all()
        assert assay[['median','q25','q75']].isna().all().all()
        if pos+neg: assert assay.proportion.eq(pos/(pos+neg)).all()
        else: assert assay.proportion.isna().all()
    elif expected_type=='numeric' and not baseline.assay.isna().all():
        np.testing.assert_allclose(assay[['median','q25','q75']], [[2.5,1.75,3.25]]*len(assay))
        assert assay.n_observed.eq(12).all()
        assert assay[['n_pos','n_neg','n_tested','proportion']].isna().all().all()
    if baseline.assay.isna().all():
        assert assay.status.eq('NOT_ESTIMABLE').all() and assay.n_observed.eq(0).all()
        assert assay[['median','q25','q75']].isna().all().all()
    expected_note='BOOLEAN_SOURCE_DECLARED_NUMERIC' if pd.api.types.is_bool_dtype(dtype) and declared=='numeric' else ''
    assert assay.type_consistency_note.eq(expected_note).all()
    pd.testing.assert_frame_equal(roles,original_roles)
    pd.testing.assert_frame_equal(tables['role_inventory'],original_roles)
    assert characterization_qc(tables,baseline,membership,{'graph02_run_id':'synthetic'},True)['status']=='PASS'
    safe=safe_characterization_panels(assay,5)
    if expected_counts is not None and any(0<n<5 for n in expected_counts): assert safe.empty
    for table in [*tables.values(),safe]:
        assert 'patient_id' not in table
        assert not any(identifier in table.to_csv(index=False) for identifier in baseline.patient_id)


def test_boolean_summaries_preserve_frozen_tiers_memberships_and_figure_privacy(tmp_path):
    cfg,baseline,membership,meta,features=fixture()
    baseline['unused']=pd.Series([True,False]*30,dtype=bool)
    baseline['related']=pd.Series([True,False,pd.NA]*20,dtype='boolean')
    baseline['independent']=pd.Series([True,False]*30,dtype=bool)
    representations=['R0','R2']
    meta['eligible_representation_ids']=representations
    membership=pd.concat([membership,membership.assign(representation_id='R2')],ignore_index=True)
    features=pd.concat([features,features.assign(representation_id='R2')],ignore_index=True)
    cfg['clinical_marker_panels']=[dict(source_column=s,label='Invented '+s,unit='synthetic_units',
        clinical_domain='D04',type='numeric') for s in ['unused','related','independent']]
    cfg['characterization']['figure_sources']=['unused']
    cfg['characterization']['relationships'].append(dict(source_column='related',target_source='unused',
        relationship_type='derived_from',source_reference='SYNTHETIC_TEST_ONLY:invented_derivation'))
    approve_synthetic(cfg,'independent','R0')
    roles=feature_roles(features,baseline,cfg,representations)
    expected={('R0','unused'):'N1_DEFINING',('R2','unused'):'N1_DEFINING',
        ('R0','related'):'N2_RELATED',('R2','related'):'N2_RELATED',
        ('R0','independent'):'N3_REVIEWED',('R2','independent'):'N3_CANDIDATE'}
    rolemap=roles.set_index(['representation_id','source_column'])
    for key,tier in expected.items(): assert rolemap.loc[key,'tier']==tier
    frozen_baseline=baseline.copy(deep=True); frozen_membership=membership.copy(deep=True)
    frozen_roles=roles.copy(deep=True); frozen_config=deepcopy(cfg); frozen_meta=deepcopy(meta)
    tables=characterization_tables(baseline,membership,roles,meta,cfg,pd.DataFrame())
    matrix=tables['clinical_matrix'].loc[lambda rows:rows.source_column.isin(['unused','related','independent'])]
    assert len(matrix)==12 and matrix.data_type.eq('binary').all()
    assert matrix.type_consistency_note.eq('BOOLEAN_SOURCE_DECLARED_NUMERIC').all()
    assert (matrix.n_pos+matrix.n_neg).eq(matrix.n_tested).all()
    assert matrix.n_tested.eq(matrix.n_observed).all()
    assert matrix[['median','q25','q75']].isna().all().all()
    assert matrix.loc[matrix.representation_id.eq('R2'),'status'].eq('DESCRIPTIVE_PROVISIONAL').all()
    qc=characterization_qc(tables,baseline,membership,{'graph02_release_sha256':'synthetic_frozen_sha'},True)
    assert qc['status']=='PASS' and qc['graph02_unchanged']
    pd.testing.assert_frame_equal(baseline,frozen_baseline)
    pd.testing.assert_frame_equal(membership,frozen_membership)
    pd.testing.assert_frame_equal(roles,frozen_roles)
    pd.testing.assert_frame_equal(tables['role_inventory'],frozen_roles)
    assert cfg==frozen_config and meta==frozen_meta
    # A rare positive suppresses the full source panel, including its complement.
    sensitive=baseline.copy(); sensitive['unused']=False; sensitive.loc[0,'unused']=True
    suppressed=characterization_tables(sensitive,membership,roles,meta,cfg,pd.DataFrame())['clinical_matrix']
    characterization_figures(suppressed,roles,tmp_path,cfg,5)
    availability=pd.read_csv(tmp_path/'03_graph_characterization_availability.csv')
    assert not availability.source_column.eq('unused').any()
    assert set(availability.source_column)=={'related','independent'}
    for rep in representations:
        assert pd.read_csv(tmp_path/f'03_graph_characterization_N1_{rep}.csv').empty
    for path in tmp_path.glob('*.csv'):
        content=path.read_text()
        assert 'patient_id' not in content
        assert not any(identifier in content for identifier in baseline.patient_id)

"""Synthetic acceptance tests: fixed panels, no leakage, support and release safety."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.studies.graph._pipeline import load_pipeline, digest
from src.studies.graph.stage02.s02_missingness_registry import build_panel, panel_metadata, NotEstimable
from src.studies.graph.stage02.s02_imputation import fit_transform, LowRankImputer
from src.studies.graph.stage02.s02_pairwise_distance import observed_distance, require_complete
from src.studies.graph.stage02.s02_missingness_variants import build_variant, mask_distance
from src.studies.graph.stage02.s02_missingness_study import geometry_comparison, cache_signature, run_missingness_study
from src.studies.graph.stage02.s02_representations import build_space
from src.studies.graph.stage02.s02_mapper_engine import analyze
from src.studies.graph.stage02.s02_resampling import resample


@pytest.fixture
def synthetic():
    rng=np.random.default_rng(927); n=40; p=12
    factors=rng.normal(size=(n,3)); matrix=factors@rng.normal(size=(3,p))+.1*rng.normal(size=(n,p))
    raw=pd.DataFrame(matrix,columns=[f'feature_{i}' for i in range(p)])
    raw=raw.mask(rng.random(raw.shape)<.12)
    registry=pd.DataFrame([dict(feature=c,source_column=c,semantic_feature_id=c,derivation_group=c,instrument_id='instrument_'+str(j//2),
        clinical_domain='D0'+str(1+j//4),subdomain='sub_'+str(j//2),technical_family=['LAB','PRO','CLINICAL'][j//4],
        family=['LAB','PRO','CLINICAL'][j//4],role='defining',temporality='baseline',temporal_status='baseline',
        temporal_validity='baseline',encoding='numeric',representation_type='numeric',exploratory_technical_eligible=True,
        mapping_eligibility='ELIGIBLE',unit='synthetic',included=True) for j,c in enumerate(raw)])
    baseline=raw.copy(); baseline['patient_id']=[f'SYNTHETIC_{i}' for i in range(n)]
    baseline['clinical_anchor_date']=pd.Timestamp('2020-01-01')
    cfg=load_pipeline(); config=deepcopy(cfg['sensitivity']); config['analysis_mode']='exploratory'
    config['mapper'].update(n_cubes=2,perc_overlap=.5,min_samples=2,minimum_node_support=2)
    config['community_detection']['minimum_patients']=2
    policy=deepcopy(cfg['missingness_methods']); policy['iterative_regularized'].update(candidate_rank=[2,3],masked_validation_repeats=2)
    families=dict(zip(registry.feature,registry.technical_family))
    return raw,registry,baseline,config,policy,families


def test_zero_argument_defaults_include_all_six_variants_and_hashes():
    cfg=load_pipeline(); policy=cfg['missingness_methods']
    assert policy['enabled'] and policy['references']==['R0','R2']
    assert 'missingness_methods' in cfg['_hashes']
    assert policy['resampling']['main_subsampling_replicates_per_variant']==60
    assert policy['resampling']['iterative_bootstrap_replicates']==20
    assert cfg['experiments']['final_subsampling_replicates']==200
    assert cfg['experiments']['bootstrap_replicates']==100


def test_original_availability_states_and_nominal_lineage(synthetic):
    raw,meta,baseline,*_=synthetic
    raw['level_a']=[1.,0.]*(len(raw)//2); raw['level_b']=1-raw.level_a
    raw.loc[0,['level_a','level_b']]=np.nan
    nominal=pd.concat([meta.iloc[[0]].assign(feature='level_a',source_column='category',semantic_feature_id='category',derivation_group='category',encoding='one_hot'),
        meta.iloc[[0]].assign(feature='level_b',source_column='category',semantic_feature_id='category',derivation_group='category',encoding='one_hot')])
    meta=pd.concat([meta,nominal],ignore_index=True)
    baseline['category__observation_status']='observed'; baseline.loc[0,'category__observation_status']='unknown_detail'
    panel=build_panel(raw,meta,'R0',baseline)
    key=panel.lineage.loc[panel.lineage.source_column.eq('category'),'observation_unit'].iloc[0]
    assert panel.mask.loc[0,key]==0 and panel.states.loc[0,key]=='unknown_detail'
    assert panel.metadata.loc[key,'n_encoded_columns']==2
    assert len(panel.metadata)==raw.shape[1]-1
    filled=raw.fillna(0)
    # Availability was frozen before completing values.
    assert panel.mask.loc[0,key]==0 and filled.loc[0,'level_a']==0


def test_partial_bilateral_and_explicit_missing_reasons(synthetic):
    raw,meta,baseline,*_=synthetic
    meta.loc[:1,'derivation_group']='bilateral'; raw.loc[0,raw.columns[:2]]=[1.,np.nan]
    baseline[raw.columns[2]+'__observation_status']='not_requested'
    panel=build_panel(raw,meta,'R0',baseline)
    key=panel.lineage.loc[panel.lineage.source_column.eq(raw.columns[0]),'observation_unit'].iloc[0]
    assert panel.states.loc[0,key]=='partial'
    key2=panel.lineage.loc[panel.lineage.source_column.eq(raw.columns[2]),'observation_unit'].iloc[0]
    assert panel.states.loc[0,key2]=='not_requested' and panel.mask.loc[0,key2]==0


@pytest.mark.parametrize('field,value',[('temporality','post_baseline'),('source_column','essdai__total')])
def test_no_postbaseline_or_outcome_predictors(synthetic,field,value):
    raw,meta,*_=synthetic; meta.loc[0,field]=value
    with pytest.raises(NotEstimable): panel_metadata(raw,meta,'R0')


def test_panel_specific_masks_and_r2_context_guard(synthetic):
    raw,meta,baseline,*_=synthetic
    a=build_panel(raw,meta,'R0',baseline)
    b=build_panel(raw.iloc[:,4:],meta,'R2',baseline)
    assert a.panel_id=='R0_fixed' and b.panel_id=='R2_fixed'
    assert a.mask.shape[1]>b.mask.shape[1]
    meta.loc[4,'clinical_domain']='D09'
    with pytest.raises(ValueError,match='RESERVED'): build_panel(raw.iloc[:,4:],meta,'R2',baseline)


def test_future_measurement_is_rejected(synthetic):
    raw,meta,baseline,*_=synthetic
    baseline[raw.columns[0]+'__measurement_date']=pd.Timestamp('2021-01-01')
    with pytest.raises(ValueError,match='FUTURE_MEASUREMENT'): build_panel(raw,meta,'R0',baseline)


def test_iterative_preserves_observed_and_is_deterministic(synthetic):
    raw,meta,_,_,policy,_=synthetic; metadata=meta.set_index('feature')
    a=fit_transform(raw,metadata,101,policy['iterative_regularized'])
    b=fit_transform(raw,metadata,101,policy['iterative_regularized'])
    np.testing.assert_array_equal(a.completed,b.completed)
    observed=raw.notna().to_numpy()
    np.testing.assert_array_equal(a.completed.to_numpy()[observed],raw.to_numpy()[observed])
    assert np.isfinite(a.completed.to_numpy()).all() and a.audit['converged']
    assert a.validation.split.str.contains('patient_holdout').all()
    assert a.validation.n_hidden.min()>0


def test_training_parameters_are_unchanged_by_heldout_patients(synthetic):
    raw,meta,_,_,policy,_=synthetic; metadata=meta.set_index('feature')
    model=LowRankImputer(2,policy['iterative_regularized'],'R2').fit(raw.iloc[:30],metadata)
    before=[x.copy() for x in [model.center,model.scale,model.weights,model.loadings]]
    test=raw.iloc[30:].copy(); test.iloc[0,0]=1e12
    predicted=model.transform(test)
    for a,b in zip(before,[model.center,model.scale,model.weights,model.loadings]): np.testing.assert_array_equal(a,b)
    assert predicted.iloc[0,0]==1e12  # Never overwrite even an outlying observed value.


@pytest.mark.parametrize('case',['nonconvergence','memory','nominal','support'])
def test_iterative_not_estimable_has_no_silent_median_fallback(synthetic,case):
    raw,meta,_,_,policy,_=synthetic; p=policy['iterative_regularized']
    if case=='nonconvergence': p.update(max_iterations=1,convergence_tolerance=1e-30)
    if case=='memory': p['max_matrix_cells']=1
    if case=='nominal': meta.loc[0,'encoding']='one_hot'
    if case=='support': raw.iloc[:,0]=np.nan
    with pytest.raises(NotEstimable): fit_transform(raw,meta.set_index('feature'),101,p)


def test_pair_without_shared_evidence_is_nan_not_zero_or_max(synthetic):
    raw,meta,baseline,_,policy,_=synthetic
    raw.loc[0,:]=np.nan; raw.loc[0,raw.columns[:4]]=1.
    raw.loc[1,:]=np.nan; raw.loc[1,raw.columns[4:]]=2.
    evidence=observed_distance(raw,meta,'R2',policy['observed_pairwise_distance'],baseline)
    assert np.isnan(evidence.distance[0,1]) and evidence.distance[0,0]==0
    row=evidence.support.query('patient_index_a==0 and patient_index_b==1').iloc[0]
    assert row.n_shared_constructs==0 and not row.distance_estimable
    with pytest.raises(NotEstimable,match='NO_FILLING'): require_complete(evidence)


def test_synthetic_missingness_increases_unsupported_pairs_without_altering_observed(synthetic):
    raw,meta,baseline,_,policy,_=synthetic
    complete=raw.fillna(0)
    original=observed_distance(complete,meta,'R0',policy['observed_pairwise_distance'],baseline)
    induced=complete.copy(); induced.loc[:9,:]=np.nan
    changed=observed_distance(induced,meta,'R0',policy['observed_pairwise_distance'],baseline)
    assert changed.summary['n_estimable_pairs']<original.summary['n_estimable_pairs']
    pd.testing.assert_frame_equal(induced.iloc[10:],complete.iloc[10:])


def test_nominal_levels_count_once_and_remain_coherent(synthetic):
    raw,meta,baseline,_,policy,_=synthetic
    raw=raw.fillna(0)
    for j in [0,1]:
        meta.loc[j,['source_column','semantic_feature_id','derivation_group']]='category'
        meta.loc[j,'encoding']='one_hot'
    raw.iloc[:,0]=np.arange(len(raw))%2; raw.iloc[:,1]=1-raw.iloc[:,0]
    evidence=observed_distance(raw,meta,'R0',policy['observed_pairwise_distance'],baseline)
    assert evidence.support.n_shared_constructs.max()==len(raw.columns)-1
    assert np.isfinite(evidence.distance).all()


def test_mask_zero_reproduces_reference_and_mixture_components_sum(synthetic):
    raw,meta,baseline,config,policy,families=synthetic
    reference=build_space('R0',raw,families,config,meta,meta,baseline,{}, {})
    zero,audit=build_variant('R0-M',raw,families,config,meta,meta,baseline,{}, {},policy,reference_space=reference,mask_weight=0)
    assert zero is reference
    mixed,audit=build_variant('R0-M',raw,families,config,meta,meta,baseline,{}, {},policy,reference_space=reference)
    np.testing.assert_allclose(mixed.distance**2,audit['value_squared_component']+audit['mask_squared_component'],atol=1e-14)
    pd.testing.assert_frame_equal(reference.raw,raw)
    assert mixed.features==reference.features and audit['mask_weight']==.25


def test_missingness_cache_keys_separate_method_seed_panel_and_states(synthetic):
    raw,meta,baseline,config,policy,_=synthetic
    def key(identifier='R0-I',r=raw,b=baseline,c=config):
        return cache_signature(r,meta,b,identifier,policy,c,'config','source')
    a=key(); assert a!=key('R0-M') and a!=key(r=raw.iloc[:,:-1])
    c=deepcopy(config); c['random_seed']+=1; assert a!=key(c=c)
    b=baseline.copy(); b['feature_0__observation_status']='unknown_detail'; assert a!=key(b=b)


def test_geometry_uses_same_valid_pairs_and_explicit_denominators():
    a=np.array([[0.,1.,2.],[1.,0.,3.],[2.,3.,0.]])
    b=a.copy(); b[0,1]=b[1,0]=np.nan
    result=geometry_comparison(a,b)
    assert result['n_valid_distance_pairs']==2 and result['n_total_distance_pairs']==3
    assert result['geometry_status']=='NOT_ESTIMABLE'


def test_cpu_real_size_159_by_98_and_memory_fallback(synthetic):
    _,_,_,_,policy,_=synthetic; rng=np.random.default_rng(80); n,p=159,98
    raw=pd.DataFrame(rng.normal(size=(n,3))@rng.normal(size=(3,p)),columns=[f'x_{i}' for i in range(p)])
    raw=raw.mask(rng.random(raw.shape)<.12)
    meta=pd.DataFrame(dict(feature=list(raw),source_column=list(raw),derivation_group=list(raw),encoding='numeric',
        technical_family=['LAB','PRO','CLINICAL','SERO','EXT']*(p//5)+['LAB']*(p%5),clinical_domain='D01')).set_index('feature')
    pcy=policy['iterative_regularized']; pcy['candidate_rank']=[3]; pcy['masked_validation_repeats']=1
    result=fit_transform(raw,meta,80,pcy)
    assert result.completed.shape==(n,p) and result.audit['converged']
    with pytest.raises(NotEstimable,match='MEMORY'):
        fit_transform(raw,meta,80,{**pcy,'max_matrix_cells':100})


def test_full_sensitivity_suite_refits_every_variant_and_emits_sources(synthetic,tmp_path):
    raw,meta,baseline,config,policy,families=synthetic
    cfg=load_pipeline(); cfg['missingness_methods']=policy
    policy['resampling'].update(main_subsampling_replicates_per_variant=2,iterative_bootstrap_replicates=2,
        parameter_sensitivity_subsampling_replicates=2)
    cfg['experiments']['minimum_success_fraction']=.5
    cfg['execution']['resume']=False
    results={}
    for reference in ['R0','R2']:
        space=build_space(reference,raw,families,config,meta,meta,baseline,{}, {})
        results[reference]=dict(space=space,mapped=analyze(space,config))
    before={r:(results[r]['space'].distance.copy(),results[r]['mapped']['hard'].copy()) for r in results}
    report=run_missingness_study(results,meta,meta,baseline,families,config,cfg,tmp_path,{}, {},'synthetic_source')
    registry=pd.read_csv(tmp_path/'02_graph_missingness_method_registry.csv')
    main=registry.loc[~registry.experiment_id.str.contains('/')]
    assert set(main.experiment_id)=={r+'-'+m for r in ['R0','R2'] for m in ['I','D','M']}
    assert main.loc[main.experiment_id.str.endswith('-M'),'status'].eq('COMPLETE').all()
    assert main.loc[main.experiment_id.str.endswith('-I'),'status'].eq('COMPLETE').all()
    assert report['graph03_defining'] is False
    attempts=pd.read_csv(tmp_path/'02_graph_missingness_resampling_replications.csv')
    assert set(attempts.loc[attempts.representation_id.str.endswith('-I'),'method'])=={'repeated_subsampling','classic_bootstrap'}
    assert set(attempts.n_draws)<={32,40}
    panels=pd.read_csv(tmp_path/'02_graph_missingness_panel_comparability.csv')
    assert panels.identical_panel.all() and panels.n_excluded_sources.eq(0).all()
    for ext in ['png','pdf','csv']: assert len(list((tmp_path/'figures').glob('*.'+ext)))==7
    for r in results:
        np.testing.assert_array_equal(before[r][0],results[r]['space'].distance)
        np.testing.assert_array_equal(before[r][1],results[r]['mapped']['hard'])
    # No raw clinical values or patient identifiers in aggregate figure sources.
    for path in (tmp_path/'figures').glob('*.csv'):
        frame=pd.read_csv(path)
        assert 'patient_id' not in frame and not set(raw)&set(frame)
        if 'n_support' in frame: assert frame.n_support.dropna().ge(5).all()


def test_resampling_passes_original_missingness_to_each_refit(synthetic):
    raw,meta,baseline,config,policy,families=synthetic
    observed=raw.notna().to_numpy().copy(); seen=[]
    def fit(draw):
        sample=raw.iloc[draw].reset_index(drop=True)
        np.testing.assert_array_equal(sample.notna(),observed[draw])
        seen.append(draw.copy())
        return build_variant('R2-M',sample,families,config,meta,meta,baseline.iloc[draw].reset_index(drop=True),{}, {},policy)[0]
    space,_=build_variant('R2-M',raw,families,config,meta,meta,baseline,{}, {},policy)
    mapped=analyze(space,config)
    settings=load_pipeline()['experiments']; settings['minimum_success_fraction']=.5
    sampled=resample(mapped['community_sets'],len(raw),fit,lambda s:analyze(s,config),settings,seed=101,identifier='R2-M',replicates=2)
    assert len(seen)==2 and sampled['n_requested']==2


@pytest.mark.parametrize('exception,status',[(NotEstimable('SYNTHETIC_SUPPORT_LIMIT'),'NOT_ESTIMABLE'),(ValueError('SYNTHETIC_ESTIMATOR_FAILURE'),'FAILED')])
def test_secondary_method_failure_is_limited_and_statuses_are_distinct(synthetic,tmp_path,monkeypatch,exception,status):
    from src.studies.graph.stage02 import s02_missingness_study as study
    from src.studies.graph.stage02 import s02_reporting as reporting
    raw,meta,baseline,config,policy,families=synthetic
    cfg=load_pipeline(); cfg['missingness_methods']=policy; cfg['execution']['resume']=False
    policy['resampling'].update(main_subsampling_replicates_per_variant=1,iterative_bootstrap_replicates=1,parameter_sensitivity_subsampling_replicates=1)
    policy['iterative_regularized'].update(candidate_rank=[2],masked_validation_repeats=1)
    policy['values_plus_mask']['sensitivity_mask_weights']=[]
    policy['observed_pairwise_distance']['sensitivity_shared_weight_fraction']=[]
    cfg['experiments']['minimum_success_fraction']=.5
    results={}
    for reference in ['R0','R2']:
        space=build_space(reference,raw,families,config,meta,meta,baseline,{}, {})
        results[reference]=dict(space=space,mapped=analyze(space,config))
    original=study.build_variant
    def failing(identifier,*args,**kwargs):
        if identifier=='R2-I': raise exception
        return original(identifier,*args,**kwargs)
    monkeypatch.setattr(study,'build_variant',failing)
    monkeypatch.setattr(reporting,'missingness_method_figures',lambda *args:[])
    report=study.run_missingness_study(results,meta,meta,baseline,families,config,cfg,tmp_path,{}, {},'synthetic')
    assert report['statuses']['R2-I']['status']==status
    assert report['statuses']['R0-M']['status']=='COMPLETE' and report['statuses']['R2-M']['status']=='COMPLETE'
    assert set(results)=={'R0','R2'} and report['graph03_defining'] is False


def test_defining_symptom_components_keep_their_upstream_role(synthetic):
    raw,meta,baseline,_,policy,_=synthetic
    # The upstream catalog explicitly separates defining components from the
    # reserved total outcome. Prefix matching would silently disable R2-M too.
    meta.loc[0,'source_column']='esspri__pain'
    panel=build_panel(raw,meta,'R2',baseline)
    assert 'esspri__pain' in set(panel.lineage.source_column)
    reconstructed=fit_transform(raw,meta.set_index('feature'),101,policy['iterative_regularized'])
    assert reconstructed.completed.shape==raw.shape
    meta.loc[0,'source_column']='esspri__total'
    with pytest.raises(NotEstimable,match='OUTCOME'):
        fit_transform(raw,meta.set_index('feature'),101,policy['iterative_regularized'])

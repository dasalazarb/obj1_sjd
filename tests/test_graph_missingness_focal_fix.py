"""Independent synthetic acceptance of the focal I/D/DG correction."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
from src.studies.graph._pipeline import load_pipeline
from src.studies.graph.stage02.s02_imputation import (MixedIterativeImputer,LowRankImputer,fit_transform,reconstruction_validation)
from src.studies.graph.stage02.s02_missingness_registry import build_panel,NotEstimable
from src.studies.graph.stage02.s02_pairwise_distance import observed_distance,require_complete,PairwiseEvidence
from src.studies.graph.stage02.s02_sparse_observed_graph import build_sparse_graph,graph_resampling


@pytest.fixture
def mixed():
    rng=np.random.default_rng(119); n=90
    x=rng.normal(size=n); codes=np.digitize(x,[-.45,.45])
    raw=pd.DataFrame({'x':x,'y':2*x+rng.normal(0,.05,n),'z':rng.normal(size=n)})
    for j in range(3): raw[f'cat_{j}']=(codes==j).astype(float)
    raw.loc[[0,4,9],['cat_0','cat_1','cat_2']]=np.nan
    registry=pd.DataFrame([dict(feature=c,source_column='category' if c.startswith('cat') else c,
        semantic_feature_id='category' if c.startswith('cat') else c,derivation_group='category' if c.startswith('cat') else c,
        encoding='one_hot' if c.startswith('cat') else 'numeric',temporality='baseline',temporal_status='baseline',
        technical_family='LAB' if c in ['x','y'] else 'CLINICAL',family='LAB' if c in ['x','y'] else 'CLINICAL',
        clinical_domain='D01' if c in ['x','y'] else 'D02',subdomain='synthetic',instrument_id='category' if c.startswith('cat') else c,
        role='defining') for c in raw])
    policy=deepcopy(load_pipeline()['missingness_methods']['iterative_regularized'])
    policy.update(masked_validation_repeats=2,training_mask_fraction=0.)
    return raw,registry,policy


def test_one_nominal_source_is_imputed_coherently_without_changing_panel_or_observed(mixed):
    raw,registry,policy=mixed; meta=registry.set_index('feature')
    a=fit_transform(raw,meta,117,policy); b=fit_transform(raw,meta,117,policy)
    pd.testing.assert_frame_equal(a.completed,b.completed)
    assert list(a.completed)==list(raw)
    assert (a.completed.filter(like='cat_').sum(axis=1)==1).all()
    assert np.isin(a.completed.filter(like='cat_'),[0,1]).all()
    observed=raw.notna().to_numpy()
    np.testing.assert_array_equal(a.completed.to_numpy()[observed],raw.to_numpy()[observed])
    assert a.audit['estimator']=='mixed_iterative_v1_ridge_softmax'
    assert a.validation.seed.eq(117).all()
    assert a.validation.model_seed.eq(117+a.validation['repeat']).all()
    nominal=a.validation.query("variable_type=='nominal' and n_hidden>0")
    assert nominal.accuracy.between(0,1).all() and nominal.baseline_accuracy.notna().all()
    assert nominal.normalized_mae.isna().all()
    assert a.audit['n_source_variables']==4 and a.audit['n_encoded_features']==6


def test_rare_or_absent_nominal_category_has_explicit_source_reason(mixed):
    raw,registry,policy=mixed; raw=raw.copy()
    raw.loc[raw.cat_2.eq(1),['cat_0','cat_1','cat_2']]=[1.,0.,0.]
    with pytest.raises(NotEstimable,match='NOMINAL_CATEGORY_WITHOUT_TRAINING_SUPPORT:category'):
        MixedIterativeImputer(policy,12).fit(raw,registry.set_index('feature'))


def test_mixed_transform_never_refits_and_honors_partial_observations(mixed):
    raw,registry,policy=mixed
    model=MixedIterativeImputer(policy,12).fit(raw.iloc[:70],registry.set_index('feature'))
    before=[[step[-1][-1].coef_.copy() for step in models] for models in model.rounds]
    heldout=raw.iloc[70:].copy(); heldout.iloc[0,0]=1e12
    heldout.loc[heldout.index[1],['cat_0','cat_1','cat_2']]=[0,np.nan,np.nan]
    result=model.transform(heldout)
    assert result.iloc[0,0]==1e12 and result.loc[heldout.index[1],'cat_0']==0
    assert result.loc[heldout.index[1],['cat_0','cat_1','cat_2']].sum()==1
    for previous,models in zip(before,model.rounds):
        for coefficients,step in zip(previous,models): np.testing.assert_array_equal(coefficients,step[-1][-1].coef_)
    with pytest.raises(ValueError,match='FIXED_PANEL_MISMATCH'): model.transform(heldout.iloc[:,::-1])


def test_numeric_low_rank_recovers_estimable_fixture_and_records_each_failed_attempt(mixed):
    raw,registry,policy=mixed
    numeric=raw[['x','y','z']].copy(); numeric['z']=-.5*numeric.x; numeric.loc[::8,'y']=np.nan
    meta=registry.set_index('feature').loc[list(numeric)]
    policy.update(candidate_rank=[1],masked_validation_repeats=2,masked_fraction_per_replicate=.3)
    result=fit_transform(numeric,meta,23,policy)
    assert result.audit['converged'] and result.audit['iterations_run']<=policy['max_iterations']
    assert set(result.validation.validation_stage)=={'rank_selection','evaluation'}
    failed=reconstruction_validation(numeric,meta,23,{**policy,'max_iterations':1,'convergence_tolerance':1e-30},'R0')
    assert len(failed)==2*3
    assert failed.normalized_mae.isna().all()
    assert failed.reason_code.str.contains('NONCONVERGENCE|NO_HIDDEN').all()
    assert failed.iterations_run.eq(1).all()


def test_no_artificial_mask_support_is_not_zero_error_or_success(mixed):
    raw,registry,policy=mixed
    with pytest.raises(NotEstimable) as caught:
        fit_transform(raw,registry.set_index('feature'),12,{**policy,'masked_fraction_per_replicate':1e-20})
    rows=caught.value.validation
    assert rows.n_hidden.eq(0).all() and rows.normalized_mae.isna().all()
    assert rows.status.eq('NOT_ESTIMABLE').all()


def test_hidden_test_values_do_not_select_rank_or_change_training_model(mixed):
    raw,registry,policy=mixed; raw=raw[['x','y','z']].copy(); raw.loc[::7,'y']=np.nan
    meta=registry.set_index('feature').loc[list(raw)]; policy.update(candidate_rank=[1,2])
    order=np.random.default_rng(4).permutation(len(raw)); cut=int(len(raw)*policy['validation_patient_fraction'])
    a=fit_transform(raw,meta,4,policy)
    changed=raw.copy(); changed.iloc[order[:cut]]=changed.iloc[order[:cut]]*7+3
    b=fit_transform(changed,meta,4,policy)
    pd.testing.assert_frame_equal(a.validation.query("validation_stage=='rank_selection'").reset_index(drop=True),
                                  b.validation.query("validation_stage=='rank_selection'").reset_index(drop=True))
    assert a.audit['rank']==b.audit['rank']


def graph_policy():
    policy=deepcopy(load_pipeline()['missingness_methods']['sparse_observed_graph'])
    policy.update(minimum_connected_patients=2,minimum_community_patients=2,
                  minimum_connected_fraction=.4,minimum_largest_component_fraction=.4,minimum_neighbors=1)
    return policy


def artificial_evidence():
    d=np.array([[0.,.1,np.nan,np.nan],[.1,0.,.2,np.nan],[np.nan,.2,0.,np.nan],[np.nan,np.nan,np.nan,0.]])
    a,b=np.triu_indices(4,1)
    support=pd.DataFrame(dict(patient_index_a=a,patient_index_b=b,distance_estimable=np.isfinite(d[a,b]),
        n_shared_constructs=2,n_shared_domains=2,shared_weight_fraction=.8,distance=d[a,b]))
    return PairwiseEvidence(d,support,dict(n_pairs=6))


def test_sparse_edges_are_observed_supported_pairs_and_isolate_never_assigned():
    evidence=artificial_evidence(); policy=graph_policy()
    a=build_sparse_graph(evidence,list('abcd'),policy,k=2,seed=1)
    b=build_sparse_graph(evidence,list('abcd'),policy,k=2,seed=1)
    assert set(a.graph.edges)=={(0,1),(1,2)}==set(b.graph.edges)
    assert all(np.isfinite(evidence.distance[i,j]) for i,j in a.graph.edges)
    assert np.isnan(a.mapped['hard'][3]) and not a.mapped['covered'][3]
    assert a.summary['n_eligibility_components']==2 and a.summary['n_isolated']==1
    assert a.summary['method_type']=='sparse_observed_graph' and a.summary['is_mapper'] is False
    assert a.summary['graph03_defining'] is False
    with pytest.raises(NotEstimable,match='NO_FILLING'): require_complete(evidence)


def test_finite_pair_without_support_cannot_enter_graph():
    evidence=artificial_evidence(); evidence.support.loc[0,'shared_weight_fraction']=.1
    result=build_sparse_graph(evidence,list('abcd'),graph_policy(),k=2,seed=1)
    assert (0,1) not in result.graph.edges and np.isnan(result.mapped['hard'][0])


def test_graph_connectivity_gate_is_explicit_and_not_adaptively_relaxed():
    policy=graph_policy(); policy['minimum_connected_fraction']=1.
    result=build_sparse_graph(artificial_evidence(),list('abcd'),policy,k=2,seed=1)
    assert result.summary['status']=='NOT_ESTIMABLE' and result.summary['n_connected']==3
    assert not result.mapped['covered'].any()


def test_graph_resampling_recalculates_from_original_values_and_records_mask_hash(mixed,monkeypatch):
    import src.studies.graph.stage02.s02_sparse_observed_graph as module
    raw,registry,_=mixed; baseline=pd.DataFrame(index=raw.index)
    distance_policy=load_pipeline()['missingness_methods']['observed_pairwise_distance']; policy=graph_policy()
    evidence=observed_distance(raw,registry,'R0',distance_policy,baseline)
    reference=build_sparse_graph(evidence,list(range(len(raw))),policy,k=5,seed=42,panel_hash=build_panel(raw,registry,'R0',baseline).fingerprint)
    seen=[]; original=module.observed_distance
    def checked(sample,*args,**kwargs):
        seen.append(sample.copy()); assert len(sample)==int(.8*len(raw))
        assert sample.filter(like='cat_').isna().any().any()
        return original(sample,*args,**kwargs)
    monkeypatch.setattr(module,'observed_distance',checked)
    settings=load_pipeline()['experiments']; settings['patient_fraction']=.8
    rows,attempts,success=graph_resampling(raw,registry,baseline,'R0',policy,distance_policy,settings,
        k=5,seed=42,replicates=3,reference_graph=reference)
    assert len(seen)==3 and len(attempts)==3
    assert all(len(row['mask_sha256'])==64 and not row['cache_used'] for row in attempts)
    assert len(set(row['sample_sha256'] for row in attempts))==3
    assert all(row['n_successful_refits']==success for row in rows)


def test_panel_fingerprint_includes_order_and_lineage_and_original_config_is_frozen(mixed):
    raw,registry,_=mixed
    panel=build_panel(raw,registry,'R0')
    assert panel.fingerprint!=build_panel(raw.iloc[:,::-1],registry,'R0').fingerprint
    modified=registry.copy(); modified.loc[0,'source_column']='other_source'
    assert panel.fingerprint!=build_panel(raw,modified,'R0').fingerprint
    policy=load_pipeline()['missingness_methods']
    assert policy['schema_version']=='graph_missingness_methods_v1'
    assert policy['values_plus_mask']['primary_mask_weight']==.25
    assert policy['values_plus_mask']['sensitivity_mask_weights']==[.1,.5]
    assert policy['iterative_regularized']['convergence_tolerance']==1e-5

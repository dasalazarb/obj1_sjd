"""Exact maps, independent review gates, refit invariance and cache isolation."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from src.studies.graph import _pipeline as pipe
from src.studies.graph._feature_catalog import build_catalog
from src.studies.graph.stage02.s02_feature_registry import clinical_registry,eligible_domain_map,verify_observed_baseline_dates
from src.studies.graph.stage02.s02_representations import hierarchical_mfa,build_space
from src.studies.graph.stage02.s02_mapper_engine import analyze
from src.studies.graph.stage02.s02_resampling import resample

SOURCES=['ext__salivary_flow_unstimulated','ext__salivary_total_unstimulated_flow',
'ext__salivary_total_stimulated_flow','ext__low_unstimulated_salivary_flow',
'ext__ocular_tbut_right','ext__ocular_van_bijsterveld_right','ext__ocular_oxford_right','ext__ocular_schirmer_abnormal_any']+[
'pro__sf36_'+s for s in ['physical_functioning','role_physical','bodily_pain','general_health','vitality','social_functioning','role_emotional','mental_health']]


def fixture():
    cfg=pipe.load_pipeline(); rng=np.random.default_rng(71); n=30
    baseline=pd.DataFrame({s:rng.normal(size=n) for s in SOURCES})
    baseline['ext__low_unstimulated_salivary_flow']=[True,False]*15
    baseline['patient_id']=[str(i) for i in range(n)]
    manifest,values,_,_=build_catalog(baseline,pd.DataFrame())
    registry=clinical_registry(manifest,cfg['clinical_domains'],values)
    return cfg,baseline,manifest,pd.DataFrame(values),registry


def test_exact_sixteen_sources_and_binary_original_join():
    cfg,_,_,raw,registry=fixture()
    sources=pd.Series([r['source_column'] for r in cfg['clinical_domains']['features']])
    assert sources.is_unique and sources.isin(SOURCES).sum()==16
    selected=registry.loc[registry.feature.isin(raw)]
    assert not selected.clinical_domain.eq('UNMAPPED').any()
    binary=selected.loc[selected.source_column.eq('ext__low_unstimulated_salivary_flow')]
    assert binary.feature.tolist()==['ext__low_unstimulated_salivary_flow__binary']
    assert binary.derivation_group.iloc[0]=='unstimulated_whole_saliva'


def test_pending_mapping_runs_provisionally_but_never_approves():
    _,_,_,raw,registry=fixture()
    matrix,audit,features=hierarchical_mfa(raw,registry,purpose='exploratory')
    assert matrix.shape[0]==len(raw) and np.isfinite(matrix).all()
    assert set(audit.loc[audit.level.eq('DOMAIN'),'block'])=={'D01','D02','D06'}
    assert registry.approval_status.isin(['pending','unmapped']).all()
    with pytest.raises(ValueError,match='UNAPPROVED'): eligible_domain_map(registry,list(raw),purpose='clinical_release')


@pytest.mark.parametrize('mutation',['duplicate_source','unknown_domain','ambiguous_feature','unit','temporal'])
def test_invalid_metadata_cannot_pass_exploratory_gate(mutation):
    cfg,_,manifest,raw,registry=fixture(); mapping=deepcopy(cfg['clinical_domains'])
    source=SOURCES[0]
    if mutation=='duplicate_source':
        mapping['features'].append(next(r for r in mapping['features'] if r['source_column']==source).copy())
        with pytest.raises(ValueError,match='Duplicate'): clinical_registry(manifest,mapping,raw)
    elif mutation=='unknown_domain':
        next(r for r in mapping['features'] if r['source_column']==source)['clinical_domain']='D99'
        with pytest.raises(ValueError,match='Unknown'): clinical_registry(manifest,mapping,raw)
    elif mutation=='ambiguous_feature':
        with pytest.raises(ValueError,match='Ambiguous'): clinical_registry(pd.concat([manifest,manifest.iloc[:1]]),mapping,raw)
    else:
        if mutation=='unit': manifest.loc[manifest.source_column.eq(source),'unit_review_status']='incompatible'
        else: manifest.loc[manifest.source_column.eq(source),'temporal_status']='future'
        result=clinical_registry(manifest,mapping,raw)
        row=result.loc[result.source_column.eq(source)].iloc[0]
        assert not row.exploratory_technical_eligible and row.exclusion_reason
        assert source not in eligible_domain_map(result,list(raw),purpose='exploratory').source_column.tolist()


def test_r0_unchanged_by_mapping_and_coverage_audit():
    cfg,baseline,manifest,raw,registry=fixture(); config=deepcopy(cfg['sensitivity'])
    config['mapper'].update(n_cubes=2,min_samples=2,minimum_node_support=2)
    families={c:'EXT' if c.startswith('ext__') else 'PRO' for c in raw}
    a=build_space('R0',raw,families,config,registry,manifest,baseline,{},{}); changed=registry.copy(); changed['clinical_domain']='D09'
    b=build_space('R0',raw,families,config,changed,manifest,baseline,{},{})
    assert np.array_equal(a.cluster,b.cluster) and np.array_equal(a.distance,b.distance)
    m1=analyze(a,config); m2=analyze(b,config,audit=True)
    assert m1['community_sets']==m2['community_sets']
    assert np.array_equal(m1['hard'],m2['hard'],equal_nan=True)
    assert len(m2['coverage_audit'])==len(raw)


def test_cache_invalidates_mapping_features_and_seed(tmp_path):
    settings=pipe.load_pipeline()['experiments']; keys=[]
    def failure(draw): raise ValueError('FIXED_SYNTHETIC_FAILURE_NO_RETRIES')
    for mapping,features,seed in [('release_a','a,b',1),('release_b','a,b',1),('release_a','a,c',1),('release_a','a,b',2)]:
        r=resample([{0,1}],4,failure,lambda x:x,settings,seed=seed,identifier='R2',replicates=2,cache_dir=tmp_path,cache_key=mapping+features)
        keys.append(r['attempts'][0]['cache_key']); assert r['n_success']==0 and len(r['attempts'])==2
    assert len(set(keys))==4

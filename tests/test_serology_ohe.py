"""Synthetic-only fixed-schema, leakage, legacy invariance and fail-closed tests."""
import copy
import importlib
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.studies.longitudinal_graph.serology_ohe import *
PREP = importlib.import_module('src.studies.longitudinal_graph.01_prepare_longitudinal_data')
MAP = importlib.import_module('src.studies.longitudinal_graph.02_run_longitudinal_mapper')

@pytest.fixture
def inputs():
    cfg = PREP.load_config(PREP.Path(PREP.__file__).with_name('config.yaml'))
    master = pd.DataFrame({'patient_id': ['a','a','b','b'], 'clinical_episode_id': ['a0','a1','b0','b1'],
                           'clinical_anchor_date': pd.to_datetime(['2020-01-01']*4), 'lab__x__value': [1.,2.,3.,4.]})
    for source in SOURCES.values(): master[source] = ['P', 'N', 'I', None]
    registry = pd.DataFrame({'public_variable': list(SOURCES.values()), 'temporal_scope': ['episode_resolved']*3})
    mapping = {'version': 'synthetic_v1', 'approved_by': 'synthetic_test_only', 'approved_on': '2020-01-01', 'rationale': 'fixture only',
               'markers': {m: {label: {'level': level, 'approved_by': 'synthetic_test_only', 'rationale': 'fixture only'} for label, level in zip(['P','N','I'], LEVELS)} for m in SOURCES}}
    return master, cfg, registry, mapping

def test_fixed_schema_and_lineage(inputs):
    master,cfg,registry,mapping = inputs
    original = master.copy(deep=True)
    state,manifest,_,qc = build_serology_ohe(*inputs)
    assert state.columns.tolist() == KEYS + COLUMNS
    assert len(state) == len(master)
    assert state.loc[3,'sero_ohe__ssa__missing'] == 1
    assert state.loc[3,'sero_ohe__ssa__negative'] == 0
    assert set(manifest.source) == set(SOURCES.values())
    assert not manifest.used_in_S3.any()
    assert (qc.n_source_nonnull == 3).all()
    pd.testing.assert_frame_equal(master,original)

@pytest.mark.parametrize('value', ['not positive', '', 'some private comment', '1:160'])
def test_unknown_closed_and_no_text_leakage(inputs,value):
    master,cfg,registry,mapping = inputs
    master.loc[0,SOURCES['ssa']] = value
    with pytest.raises(SerologyBlocked) as exc: build_serology_ohe(*inputs)
    assert exc.value.report['reason'] == 'unrecognized_source_value'
    assert value not in str(exc.value.report['vocabulary']) if value else True

def test_no_unapproved_dictionary(inputs):
    inputs[3]['approved_by'] = None
    with pytest.raises(SerologyBlocked): build_serology_ohe(*inputs)

def test_no_registry_no_prefix_bypass(inputs):
    with pytest.raises(SerologyBlocked): build_serology_ohe(inputs[0],inputs[1],None,inputs[3])
    assert PREP.temporal_scope('sero_ohe__anything')[0] == 'unknown'

def test_temporal_future_blocked(inputs):
    master,cfg,registry,mapping = inputs
    master['sample_date'] = pd.Timestamp('2020-01-02')
    mapping['temporal_review'] = {'date_columns': {'ssa': 'sample_date'}}
    with pytest.raises(SerologyBlocked, match='future_or_unverifiable'): build_serology_ohe(*inputs)

def test_join_one_to_one_and_order(inputs):
    ohe = build_serology_ohe(*inputs)[0]
    state = inputs[0][KEYS+['lab__x__value']].iloc[::-1].reset_index(drop=True)
    result = join_ohe(state,ohe)
    assert result[KEYS].equals(state[KEYS])
    with pytest.raises(ValueError): join_ohe(state,pd.concat([ohe,ohe.iloc[:1]]))
    with pytest.raises(ValueError): join_ohe(state,ohe.iloc[:-1])

def test_bootstrap_schema_constants_and_training_only(inputs):
    from src.studies.longitudinal_graph.stability import patient_draw
    ohe=build_serology_ohe(*inputs)[0]
    raw=join_ohe(inputs[0][KEYS+['lab__x__value']],ohe)
    sampled,_=patient_draw(raw,np.random.default_rng(3))
    for _,group in sampled.groupby('bootstrap_draw_id'): assert len(group)==2
    sampled=raw.iloc[[0,3]].copy(); sampled.loc[sampled.index[0],'lab__x__value']=np.nan
    state,features=s4_preprocessor(['lab__x__value'])(sampled,['lab__x__value']+COLUMNS)
    assert features == ['lab__x__value']+COLUMNS
    assert state['sero_ohe__ssa__indeterminate'].eq(0).all()
    assert state['lab__x__value'].eq(4).all()
    assert state['sero_ohe__ssa__negative'].eq(0).all()

def test_s3_embedding_invariance_and_ohe_identity(inputs):
    cfg=inputs[1]; master=inputs[0]
    matrix=pd.DataFrame({'lab__x__value':[1.,4.,7.,9.], 'pro__x':[2.,5.,1.,8.]})
    disabled=copy.deepcopy(cfg); disabled['serology_ohe']['enabled']=False
    a,pa=MAP.fit_visit_embedding(matrix,cfg,[1.,1.])
    b,pb=MAP.fit_visit_embedding(matrix,disabled,[1.,1.])
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(pa.components_,pb.components_)
    original_features=MAP.reduce_redundancy(matrix)[0]
    assert not any(f.startswith('sero_ohe__') for f in original_features)
    ohe=build_serology_ohe(*inputs)[0]
    joined=pd.concat([matrix,ohe[COLUMNS]],axis=1)
    _,pca=MAP.fit_visit_embedding(joined,cfg,np.ones(14),COLUMNS)
    assert pca.components_.shape[1]==14

def test_dry_prepare_no_writes_and_blocking(inputs,tmp_path):
    master,cfg,registry,mapping=inputs
    args=SimpleNamespace(integrated=tmp_path/'absent',config=tmp_path/'config')
    result=PREP.prepare_serology(master,cfg,registry,args)
    assert result['status']=='blocked_needs_mapping_review'
    assert list(tmp_path.iterdir())==[]

def test_output_root_legacy_and_ohe_coexist(inputs,tmp_path,monkeypatch):
    master,cfg,registry,mapping=inputs
    integrated=tmp_path/'integrated.parquet'; master.to_parquet(integrated)
    config=tmp_path/'config.yaml'; config.write_text('synthetic')
    map_path=tmp_path/'map.yaml'; map_path.write_text('synthetic')
    import src.studies.longitudinal_graph.serology_ohe as module
    monkeypatch.setattr(module,'load_mapping',lambda *a:(mapping,map_path))
    dirs={k:tmp_path/k for k in ['analytic','tables','qc']}
    for p in dirs.values(): p.mkdir()
    legacy=dirs['analytic']/'01_longitudinal_visit_state.parquet'
    master[KEYS+['lab__x__value']].to_parquet(legacy)
    before=legacy.read_bytes()
    result=PREP.prepare_serology(master,cfg,registry,SimpleNamespace(integrated=integrated,config=config),dirs)
    assert result['status']=='completed'
    assert legacy.read_bytes()==before
    raw=pd.read_parquet(dirs['analytic']/'01_longitudinal_serology_ohe_raw.parquet')
    pd.testing.assert_frame_equal(raw,pd.read_parquet(dirs['analytic']/'01_longitudinal_serology_ohe_state.parquet'))
    assert len(raw.columns)==14

def test_s4_runner_synthetic_full_pipeline(inputs,tmp_path,monkeypatch):
    import src.studies.longitudinal_graph.serology_runner as runner
    from src.studies.longitudinal_graph.stability import bootstrap_mapper
    master,cfg,registry,mapping=inputs
    # Larger fixture for topology, still strictly synthetic.
    rng=np.random.default_rng(10); n=24
    master=pd.DataFrame({'patient_id':np.repeat([f'p{i}' for i in range(12)],2),
                         'clinical_episode_id':[f'e{i}' for i in range(n)],
                         'clinical_anchor_date':pd.date_range('2020-01-01',periods=n),
                         'clinical_visit_number':np.tile([1,2],12),
                         'lab__x__value':rng.normal(size=n), 'pro__x':rng.normal(size=n)})
    for source in SOURCES.values(): master[source]=np.tile(['P','N','I',None],6)
    registry=pd.DataFrame({'public_variable':list(SOURCES.values()),'temporal_scope':['episode_resolved']*3})
    ohe=build_serology_ohe(master,cfg,registry,mapping)[0]
    dirs={k:tmp_path/k for k in ['analytic','tables','figures']}
    for p in dirs.values(): p.mkdir()
    state=master[KEYS+['lab__x__value','pro__x']]
    state_path=dirs['analytic']/'01_longitudinal_visit_state.parquet';state.to_parquet(state_path)
    for kind in ['state','raw']: ohe.to_parquet(dirs['analytic']/f'01_longitudinal_serology_ohe_{kind}.parquet')
    config_path=tmp_path/'approved-config.yaml'; config_path.write_text('synthetic')
    mapping_path=tmp_path/'approved-map.yaml'; mapping_path.write_text('synthetic')
    import src.studies.longitudinal_graph.serology_ohe as module
    monkeypatch.setattr(module,'load_mapping',lambda *a:(mapping,mapping_path))
    provenance={'status':'completed','config_sha256':MAP.sha256_file(config_path),'mapping_sha256':MAP.sha256_file(mapping_path),'mapping_version':'synthetic','artifact_sha256':{f'01_longitudinal_serology_ohe_{kind}.parquet':MAP.sha256_file(dirs['analytic']/f'01_longitudinal_serology_ohe_{kind}.parquet') for kind in ['state','raw']}}
    MAP.write_json(dirs['analytic']/'01_serology_ohe_provenance.json',provenance)
    metadata_path=tmp_path/'metadata.parquet'; master.to_parquet(metadata_path)
    args=SimpleNamespace(state=state_path,metadata=metadata_path,config=config_path,intervals=tmp_path/'absent')
    # Exercise real patient refits with a reduced TEST ONLY budget; production insists on >=200.
    def test_bootstrap(*a,**kw):
        settings=copy.deepcopy(a[2]);settings['mapper_validation']['bootstrap_replicates']=2
        return bootstrap_mapper(a[0],a[1],settings,*a[3:],**kw)
    monkeypatch.setattr(runner,'bootstrap_mapper',test_bootstrap)
    cfg['mapper'].update(n_cubes=2,min_samples=2,minimum_node_patients=1,minimum_macrostate_patients=1)
    primary=MAP.run_mapper_scenario(state,list(state.drop(columns=KEYS)),cfg,family_balance=True,scenario_name='S3-primary')
    before=state_path.read_bytes()
    result=runner.execute_s4(args,MAP,state,state,master,list(state.drop(columns=KEYS)),primary,dirs,cfg)
    assert result['status']=='candidate_requires_review'
    assert state_path.read_bytes()==before
    rows=result['comparison']
    assert [r['n_features'] for r in rows]==[14,11,2]
    manifest=result['scenarios'][0]
    assert manifest['feature_list']==['lab__x__value','pro__x']+COLUMNS
    assert sum(manifest['feature_weights'][f]**2 for f in COLUMNS)==pytest.approx(1.)
    assert manifest['bootstrap']['replicates_completed']+manifest['bootstrap']['replicates_failed']+manifest['bootstrap']['replicates_no_nodes']==2
    assert result['representation_stability_gate']['status']=='blocked'
    cir=pd.read_csv(dirs['tables']/'02_s4_mapper_circularity_audit.csv')
    assert not cir.loc[cir.used_in_S4,'independent_external_validator'].any()


def test_anticorrelated_ohe_protected(inputs):
    master,cfg,registry,mapping=inputs
    for source in SOURCES.values(): master[source]=['P','N','P','N']
    ohe=build_serology_ohe(*inputs)[0]
    retained,audit=MAP.reduce_redundancy(ohe[COLUMNS])
    assert retained==COLUMNS
    assert audit.empty


def test_ohe_cannot_enter_s3(inputs):
    from src.studies.longitudinal_graph.runner import validate_inputs
    state=join_ohe(inputs[0][KEYS+['lab__x__value']],build_serology_ohe(*inputs)[0])
    with pytest.raises(ValueError,match='separate S4 artifact'):
        validate_inputs(state,inputs[0],inputs[1],MAP)

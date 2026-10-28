"""Public synthetic acceptance and failure-injection tests for the frozen study."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import numpy as np
import pandas as pd
import pytest
import yaml
from src.studies.graph import _pipeline as pipe
from src.studies.graph.stage02 import s02_legacy as legacy
from src.studies.graph.stage02.s02_contracts import resolve_sources,validate_release,resolve_latest,SCHEMA
from src.studies.graph.stage02.s02_pipeline import run as run02
from src.studies.graph.stage02.s02_feature_registry import clinical_registry,eligible_domain_map
from src.studies.graph.stage02.s02_representations import gower_distance,hierarchical_mfa,famd_matrix
from src.studies.graph.stage02.s02_resampling import match_sets,draw_indices,collapse_bootstrap,resample
from src.studies.graph.stage02.s02_representations import Space
from src.studies.graph.stage02.s02_perturbations import permute_observed
from src.studies.graph.stage02.s02_decision_rules import release_decision
from src.studies.graph.stage03.s03_construct_validation import feature_roles,assert_independent
from src.studies.graph.stage03.s03_release_reader import read_release
from src.studies.graph.stage03.s03_longitudinal import post_index_data
from src.studies.graph.stage03.s03_pipeline import run as run03
from src.studies.graph.run_graph_study import run as run_study

GRAPH=Path(__file__).parents[1]/'src/studies/graph'


@pytest.fixture
def configuration(tmp_path):
    """Pharma00 real contract on synthetic patients; never relax source governance."""
    rng=np.random.default_rng(919); n=36
    master=pd.DataFrame({'patient_id':[f'synthetic_{i}' for i in range(n)],
        'clinical_episode_id':[f'baseline_{i}' for i in range(n)],'clinical_anchor_date':pd.Timestamp('2020-01-01'),
        'clinical_visit_number':1,'clinical_visit':True,'is_clinical_baseline':True,
        'pop__status':['Pop1','Pop2','Pop3']*(n//3),'essdai__total':rng.uniform(0,10,n),'esspri__total':rng.uniform(0,10,n),
        'pro__sf36_pcs':rng.normal(size=n),'pro__sf36_mcs':rng.normal(size=n),
        'lab__albumin__value':rng.normal(4,.4,n),'lab__hemoglobin__value':rng.normal(12,2,n),
        'demo__protocol':['a','b']*(n//2),'demo__age_at_baseline':rng.uniform(30,70,n),'demo__sex':['female','male']*(n//2)})
    # Distinct irregular visits, before and after index; topology always baseline-only.
    follow=master.copy(); follow['clinical_episode_id']='follow_'+follow.clinical_episode_id
    follow['clinical_anchor_date']=pd.Timestamp('2021-02-01'); follow['is_clinical_baseline']=False; follow['clinical_visit_number']=2
    source=tmp_path/'source.parquet'; pd.concat([master,follow],ignore_index=True).to_parquet(source,index=False)
    spec=importlib.util.spec_from_file_location('synthetic_pharma_for_graph',GRAPH.parent/'pharma/00_profile_labs.py')
    pharma=importlib.util.module_from_spec(spec); spec.loader.exec_module(pharma)
    profile=tmp_path/'profile'/'00_pharma_lab_profile.csv'
    pharma.run(pharma.parse_args(['--integrated',str(source),'--output',str(profile)]))
    pipeline=yaml.safe_load(pipe.DEFAULT_CONFIG.read_text())
    for name,relative in list(pipeline['includes'].items()):
        p=(GRAPH/'configs'/relative).resolve()
        data=yaml.safe_load(p.read_text())
        if name in {'canonical','sensitivity'}:
            data['mapper'].update(n_cubes=2,perc_overlap=.5,min_samples=2,minimum_node_support=2)
            data['community_detection']['minimum_patients']=2
        if name=='experiments':
            for key in ['final_subsampling_replicates','secondary_subsampling_replicates','bootstrap_replicates','preflight_replicates','negative_control_replicates','comparator_replicates']:
                data[key]=2
            data['minimum_mapper_coverage']=0.
            data['minimum_success_fraction']=.5
            data['mapper_grid']['n_cubes']=[2,3]
            data['mapper_grid']['perc_overlap']=[.4,.5]
            data['minimum_stratum_patients']=8
            data['comparator_k']=[2,3]
        if name=='validation':
            data['minimum_group_patients']=2; data['effect_ci_replicates']=20
        out=tmp_path/f'{name}.yaml'; out.write_text(yaml.safe_dump(data)); pipeline['includes'][name]=str(out)
    pipeline['paths'].update(integrated=str(source),pharma_candidates=str(profile.with_name(pharma.GRAPH_CANDIDATES_NAME)),
        baseline_metrics=str(tmp_path/'absent_baseline.csv'),lab_group_map=str(legacy.DEFAULT_LAB_GROUP_MAP),clinical_release=str(GRAPH/'graph_lab_release.yaml'))
    for stage in ['02','03','study']: pipeline['paths']['output_'+stage]=str(tmp_path/('output_'+stage))
    path=tmp_path/'pipeline.yaml'; path.write_text(yaml.safe_dump(pipeline))
    return pipe.load_pipeline(path)


def test_default_full_protocol_and_no_required_flags():
    cfg=pipe.load_pipeline()
    assert cfg['analysis']=={'mode':'exploratory','feature_policy':'phenotype_independent','reference_scenario':'S3'}
    assert cfg['experiments']['final_subsampling_replicates']==200
    assert set(cfg['experiments']['legacy_scenarios'])==set(legacy.SCENARIOS)
    assert cfg['experiments']['mapper_grid']['n_cubes']==[6,8,10]
    assert cfg['experiments']['mapper_grid']['perc_overlap']==[.3,.4,.5]
    assert len(cfg['clinical_domains']['domains'])==9
    for name in ['run_graph_study.py','02_graph_representation_sensitivity.py','03_graph_community_characterization.py']:
        source=(GRAPH/name).read_text()
        assert 'required=True' not in source


def test_zero_argument_full_study_and_standalone_entries(configuration,monkeypatch):
    cfg=configuration
    monkeypatch.setattr(pipe,'DEFAULT_CONFIG',Path(cfg['_path']))
    for executable in ['run_graph_study.py','02_graph_representation_sensitivity.py','03_graph_community_characterization.py']:
        monkeypatch.setattr(sys,'argv',[str(GRAPH/executable)])
        runpy.run_path(str(GRAPH/executable),run_name='__main__')
    manifest,path=resolve_latest(pipe.output_root(cfg,'02'))
    assert manifest['finalized'] and manifest['primary_reference_id']=='R0'
    assert manifest['study_mode']=='exploratory'
    assert manifest['release_status'] in {'NO_STABLE_STRUCTURE','PARTIAL_OR_EXPLORATORY'}
    registry=pd.read_csv(path.parent/'02_graph_experiment_registry.csv')
    for identifier in ['S0','S1','S2','S3','S4','R0','R1','R2','R3','R4','D1','D2','D3','D4','D5']:
        assert identifier in set(registry.experiment_id)
    assert registry.loc[registry.experiment_id.eq('R2'),'status'].iloc[0]=='NOT_ESTIMABLE'
    assert registry.experiment_id.str.startswith('A1/').sum()==9
    assert registry.loc[registry.kind.eq('mapper_grid')].shape[0]==4
    methods=pd.read_csv(path.parent/'02_graph_resampling_replications.csv')
    assert set(methods.method)=={'repeated_subsampling','classic_bootstrap'}
    assert methods.loc[methods.method.eq('repeated_subsampling')].n_draws.eq(methods.loc[methods.method.eq('repeated_subsampling')].n_unique_patients).all()
    assert len(list((path.parent/'figures').glob('*.png')))==4
    assert len(list((path.parent/'figures').glob('*.pdf')))==4
    study=list(pipe.output_root(cfg,'study').glob('runs/*'))[0]
    report=json.loads((study/'graph_study_execution_manifest.json').read_text())
    assert report['execution_status'] in {'COMPLETE_WITH_LIMITATIONS','PARTIAL'}
    linked=(study/report['stages']['02']['manifest_path']).resolve()
    assert pipe.digest(linked)==report['stages']['02']['manifest_sha256']
    assert len(pd.read_csv(study/'graph_study_output_index.csv'))>30
    for run in pipe.output_root(cfg,'03').glob('runs/*'):
        roles=pd.read_csv(run/'03_graph_clinical_feature_roles.csv')
        anchors=pd.read_csv(run/'03_graph_independent_anchor_validation.csv')
        assert anchors.validation_role.eq('independent_baseline_anchor').all()
        summary=json.loads((run/'03_graph_results_summary.json').read_text())
        assert summary['graph02_run_id']
        assert list((run/'figures').glob('*.pdf'))


def test_release_tamper_failure_does_not_refit_or_replace_pointer(configuration,monkeypatch):
    result=run02(configuration)
    root=pipe.output_root(configuration,'02'); pointer=(root/'latest_completed_02.json').read_bytes()
    monkeypatch.setattr(legacy,'fit_embedding',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Graph03 must not refit')))
    run03(configuration,graph02_release=result['manifest'])
    artifact=result['run_dir']/'02_graph_patient_membership.csv'; artifact.write_text(artifact.read_text()+'\n')
    with pytest.raises(ValueError,match='SHA mismatch'): run03(configuration,graph02_release=result['manifest'])
    assert (root/'latest_completed_02.json').read_bytes()==pointer
    with pytest.raises(AssertionError): run02(configuration)
    assert (root/'latest_completed_02.json').read_bytes()==pointer


def test_source_ambiguity_requires_pin(configuration,tmp_path):
    cfg=configuration
    sources=resolve_sources(cfg); candidate=sources['pharma_candidates']
    other=candidate.parent/'runs'/'distinct'; other.mkdir(parents=True)
    import shutil
    for path in candidate.parent.iterdir():
        if path.is_file(): shutil.copyfile(path,other/path.name)
    summary_path=other/'00_pharma_profile_summary.json'; summary=json.loads(summary_path.read_text())
    summary['run_note']='distinct compatible clinical source snapshot'; summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='AMBIGUOUS'): resolve_sources(cfg)
    pin=tmp_path/'pin.json'; pipe.write_json(pin,{'schema_version':'graph_source_v1','feature_policy':cfg['analysis']['feature_policy'],
        **{role:{'path':str(path),'sha256':pipe.digest(path)} for role,path in sources.items()}})
    cfg['paths']['source_manifest']=str(pin)
    assert resolve_sources(cfg)==sources


def test_gower_properties_and_missingness():
    frame=pd.DataFrame({'continuous':[0.,5.,10.],'category':['a','b','a'],'binary':[0,1,0]})
    d=gower_distance(frame,{'continuous':'numeric','category':'nominal','binary':'numeric'})
    assert np.allclose(d,d.T) and np.allclose(np.diag(d),0)
    assert ((d>=0)&(d<=1)).all()
    assert d[0,2]==pytest.approx(1/3)
    frame.loc[0,'continuous']=np.nan
    assert np.isfinite(gower_distance(frame,{'continuous':'numeric','category':'nominal','binary':'numeric'})).all()
    with pytest.raises(ValueError,match='NO_SHARED'):
        gower_distance(pd.DataFrame({'a':[1.,np.nan],'b':[np.nan,2.]}))


def approved_registry():
    return pd.DataFrame({'feature':['a','b','constant'],'source_column':['a','b','constant'],
        'clinical_domain':['D01','D02','D03'],'subdomain':['x','y','z'],'family':['EXT']*3,
        'technical_family':['EXT']*3,'mapping_eligibility':['ELIGIBLE']*3,
        'temporal_validity':['baseline']*3,'temporality':['baseline']*3})


def test_clinical_mfa_zero_blocks_and_strict_mapping():
    raw=pd.DataFrame({'a':[1.,2.,3.,4.],'b':[4.,1.,3.,2.],'constant':[1.]*4})
    registry=approved_registry(); x,audit,features=hierarchical_mfa(raw,registry)
    assert np.isfinite(x).all() and x.shape==(4,2)
    assert audit.loc[audit.block.eq('D03'),'status'].iloc[0]=='zero_variance_excluded'
    for mutation in ['UNMAPPED','NOT_APPROVED']:
        bad=registry.copy(); bad.loc[0,'mapping_eligibility']=mutation
        with pytest.raises(ValueError,match='UNAPPROVED_OR_UNMAPPED'): hierarchical_mfa(raw,bad)
    bad=registry.copy(); bad.loc[0,'temporal_validity']='future'
    with pytest.raises(ValueError,match='TEMPORAL'): hierarchical_mfa(raw,bad)
    with pytest.raises(ValueError,match='Incomplete or duplicate'): hierarchical_mfa(raw,pd.concat([registry,registry.iloc[:1]]))


def test_matching_one_to_one_splits_merges_and_label_permutation():
    reference=[{1,2},{3,4}]; candidate=[{3,4},{1,2}]
    matches,_=match_sets(reference,candidate)
    assert {(r['reference'],r['candidate'],r['jaccard']) for r in matches}=={(0,1,1.),(1,0,1.)}
    split,_=match_sets([{1,2,3,4}],[{1,2},{3,4}])
    assert len(split)==1 and split[0]['split_candidates']==2


def test_draws_and_bootstrap_unique_patient_mapping():
    first=draw_indices(20,.8,123,'repeated_subsampling')
    assert len(first)==len(set(first))==16
    assert np.array_equal(first,draw_indices(20,.8,123,'repeated_subsampling'))
    draw=np.array([0,0,2,2,3]); x=np.arange(10).reshape(5,2)
    space=Space(pd.DataFrame(x),x,x,np.zeros((5,5)),pd.DataFrame(),'synthetic',['a','b'])
    unique,indices=collapse_bootstrap(space,draw)
    assert indices.tolist()==[0,2,3] and len(unique.raw)==3


def test_null_preserves_marginals_and_masks():
    raw=pd.DataFrame({'a':[1.,np.nan,2.,3.,4.],'b':[2.,3.,4.,5.,np.nan]})
    null=permute_observed(raw,44)
    pd.testing.assert_frame_equal(null.isna(),raw.isna())
    for c in raw: assert sorted(null[c].dropna())==sorted(raw[c].dropna())


def test_circularity_includes_derived_proxies():
    features=pd.DataFrame({'representation_id':['R0','R2'],'source_column':['esspri__fatigue','pro__sf36_pcs'],
                           'feature':['esspri__fatigue','pro__sf36_pcs'],'included':[True,True]})
    baseline=pd.DataFrame({'patient_id':['synthetic_a'],'esspri__total':[5.],'esspri__fatigue':[4.],'pro__sf36_pcs':[35.]})
    cfg=pipe.load_pipeline()['validation']; roles=feature_roles(features,baseline,cfg,['R0','R2'])
    assert roles.query("representation_id=='R0' and source_column=='esspri__total'").validation_role.iloc[0]=='derived_from_defining'
    assert roles.query("representation_id=='R2' and source_column=='esspri__total'").validation_role.iloc[0]=='independent_baseline_anchor'
    with pytest.raises(ValueError,match='Circular'): assert_independent(roles.query("validation_role=='derived_from_defining'"))


def test_longitudinal_uses_strict_actual_post_index_dates():
    cfg=pipe.load_pipeline()['validation']
    baseline=pd.DataFrame({'patient_id':['a'],'clinical_anchor_date':['2020-01-01']})
    master=pd.DataFrame({'patient_id':['a']*4,'clinical_anchor_date':['2019-12-31','2020-01-01','2020-02-01','2021-07-01'],
        'is_clinical_baseline':[False,True,False,False],'essdai__total':[99.,88.,1.,2.]})
    members=pd.DataFrame({'patient_id':['a'],'community_id':['0']})
    data=post_index_data(master,baseline,members,'essdai__total',cfg)
    assert data.value.tolist()==[1.,2.] and data.time_years.gt(0).all()
    assert data.time_years.iloc[1]>1


@pytest.mark.parametrize('primary_qc,stable,mode,limitations,expected',[
    (False,1,'exploratory',[],'QC_FAILED'),(True,0,'exploratory',[],'NO_STABLE_STRUCTURE'),
    (True,1,'exploratory',[],'PARTIAL_OR_EXPLORATORY'),(True,1,'clinical_release',[],'VALIDATED_TOPOLOGICAL_STRUCTURE')])
def test_release_status_rules(primary_qc,stable,mode,limitations,expected):
    status,_=release_decision(primary_built=True,primary_qc=primary_qc,stable_count=stable,mode=mode,limitations=limitations,artifact_risk=False)
    assert status==expected


def test_approved_synthetic_clinical_mfa_is_released_and_incomplete_s4_isolated(configuration,tmp_path):
    cfg=configuration
    mapping=deepcopy(cfg['clinical_domains']); mapping['mapping_review_status']='approved'
    for row in mapping['features']:
        row.update(approval='approved',temporal_validity='baseline',unit='SYNTHETIC_UNITS_ONLY',release_id='SYNTHETIC_TEST_ONLY')
    cfg['clinical_domains']=mapping
    # A missing historical S4 mapping blocks R1/S4, while R0/R2/Gower finish.
    map_path=tmp_path/'empty_s4.csv'
    pd.DataFrame(columns=['lab','s4_group','s4_subgroup','mapping_status']).to_csv(map_path,index=False)
    cfg['paths']['lab_group_map']=str(map_path)
    result=run02(cfg)
    manifest,_=validate_release(result['manifest'])
    assert 'R2' in manifest['eligible_representation_ids']
    registry=pd.read_csv(result['run_dir']/'02_graph_experiment_registry.csv')
    assert registry.query("experiment_id=='S4'").status.iloc[0]=='NOT_ESTIMABLE'
    assert registry.query("experiment_id=='R2'").status.iloc[0]=='COMPLETE'
    weights=pd.read_csv(result['run_dir']/'02_graph_domain_weight_audit.csv')
    assert set(weights.query("representation_id=='R2'").level)=={'SUBDOMAIN','DOMAIN'}
    assert registry.query("experiment_id=='R4'").status.iloc[0]=='COMPLETE'  # Nominal sex is one source variable.
    read_release(cfg,result['manifest'])


def clinical_frames(n=90):
    rng=np.random.default_rng(932)
    community=np.tile(np.repeat(np.arange(3),10),3); pop=np.repeat(['Pop1','Pop2','Pop3'],30)
    baseline=pd.DataFrame({'patient_id':[f'oracle_synthetic_{i}' for i in range(n)],'clinical_anchor_date':pd.Timestamp('2020-01-01'),
        'pop__status':pop,'essdai__total':2*community+rng.normal(size=n),
        'esspri__total':community+rng.normal(size=n)})
    membership=pd.DataFrame({'patient_id':baseline.patient_id,'representation_id':'R0','community_id':community,
        'hard_assignment':True,'mapper_covered':True,'membership_tie':False,'supported':True,'structural_stability_status':'stable'})
    manifest={'eligible_representation_ids':['R0'],'release_status':'PARTIAL_OR_EXPLORATORY'}
    config=deepcopy(pipe.load_pipeline()['validation']); config['confounders']=[]; config['effect_ci_replicates']=30
    features=pd.DataFrame({'representation_id':['R0'],'source_column':['pro__sf36_pcs'],'feature':['pro__sf36_pcs'],'included':[True]})
    roles=feature_roles(features,baseline,config,['R0'])
    return baseline,membership,manifest,config,roles


def test_estimable_anchors_incremental_and_discovery_sensitivity():
    from src.studies.graph.stage03.s03_profiles import profile_tables
    from src.studies.graph.stage03.s03_incremental_value import incremental_models
    from src.studies.graph.stage03.s03_uncertainty import discovery_sensitivity
    baseline,membership,manifest,config,roles=clinical_frames()
    tables=profile_tables(baseline,membership,roles,manifest,config,331)
    anchors=tables['anchors']; assert set(anchors.status)=={'COMPLETE'}
    assert anchors.effect.between(-1,1).all() and anchors.ci_low.le(anchors.ci_high).all()
    assert anchors.q_value.between(0,1).all()
    increment=incremental_models(baseline,membership,roles,manifest,config)
    assert increment.status.eq('COMPLETE').all() and increment.delta_r2.gt(.1).all()
    refits=pd.concat([pd.DataFrame({'representation_id':'R0','replicate_id':r,'patient_id':baseline.patient_id,
        'reference_community_id':membership.community_id,'matching_jaccard':1.}) for r in range(3)],ignore_index=True)
    result=discovery_sensitivity(refits,baseline,roles,anchors,pipe.load_pipeline()['experiments'],config)
    assert result.status.eq('COMPLETE').all() and result.n_estimable_refits.eq(3).all()
    assert result.effect_p025.le(result.effect_p975).all()


def test_estimable_real_time_mixed_model():
    from src.studies.graph.stage03.s03_longitudinal import longitudinal
    baseline,membership,manifest,config,roles=clinical_frames(); config['longitudinal_endpoints']=['essdai__total']
    rng=np.random.default_rng(822); rows=[]; intercept=rng.normal(size=len(baseline))
    for i,p in enumerate(baseline.patient_id):
        community=int(membership.community_id.iloc[i])
        for t in [.4,.9,1.8]:
            rows.append({'patient_id':p,'clinical_anchor_date':pd.Timestamp('2020-01-01')+pd.Timedelta(days=t*365.25+i%15),
                'is_clinical_baseline':False,'essdai__total':3+intercept[i]+(.5+.4*community)*t+rng.normal(0,.2)})
    eligibility,models,trajectories=longitudinal(pd.DataFrame(rows),baseline,membership,manifest,config)
    assert eligibility.status.eq('COMPLETE').all()
    assert models.status.eq('COMPLETE').all() and models.estimate.gt(0).all()
    assert models.ci_low.le(models.ci_high).all()
    assert trajectories.n_patients.le(len(baseline)).all()


def test_empty_mapper_is_a_negative_scientific_result(configuration,monkeypatch):
    from src.studies.graph.stage02 import s02_pipeline
    original=s02_pipeline.analyze
    def empty(space,config,**kwargs):
        mapped=original(space,config,**kwargs)
        n=len(space.raw)
        mapped.update(nodes={},communities=[],community_sets=[],covered=np.zeros(n,dtype=bool),
            hard=np.full(n,np.nan),probabilities=np.empty((n,0)),ties=np.zeros(n,dtype=bool),supported=np.zeros(0,dtype=bool))
        return mapped
    monkeypatch.setattr(s02_pipeline,'analyze',empty)
    cfg=configuration; cfg['execution']['resume']=False
    cfg['experiments']['minimum_mapper_coverage']=.5
    result=run02(cfg)
    manifest,_=validate_release(result['manifest'])
    assert manifest['release_status']=='NO_STABLE_STRUCTURE'
    assert manifest['n_covered']==0
    run03(cfg,graph02_release=result['manifest'])


def test_pointer_schema_paths_and_missing_release(tmp_path):
    from src.studies.graph.stage02.s02_contracts import safe_artifact
    with pytest.raises(ValueError,match='MISSING_COMPLETED'): resolve_latest(tmp_path)
    pipe.write_json(tmp_path/'latest_completed_02.json',{'schema_version':'unsupported'})
    with pytest.raises(ValueError,match='pointer schema'): resolve_latest(tmp_path)
    for path in ['../escape.csv','/tmp/escape.csv']:
        with pytest.raises(ValueError,match='remain inside'): safe_artifact(tmp_path,path)


def test_fixed_contract_error_is_not_resampling_instability():
    def bad_fit(draw):
        raise legacy.LAB.LabContractError('CONTRACT_MISMATCH','synthetic fixed error')
    with pytest.raises(legacy.LAB.LabContractError):
        resample([{0,1,2}],5,bad_fit,lambda x:x,pipe.load_pipeline()['experiments'],seed=1,identifier='R0',replicates=2)


def test_exact_measurement_dates_reject_future_baseline_results():
    from src.studies.graph.stage02.s02_feature_registry import verify_observed_baseline_dates
    baseline=pd.DataFrame({'clinical_anchor_date':['2020-01-01'],'lab__albumin__value':[4.],
                           'lab__albumin__measurement_date':['2020-01-02']})
    manifest=pd.DataFrame({'source_column':['lab__albumin__value'],'included':[True]})
    with pytest.raises(ValueError,match='FUTURE_MEASUREMENT'): verify_observed_baseline_dates(baseline,manifest)
    baseline['lab__albumin__measurement_date']='2019-12-31'
    verify_observed_baseline_dates(baseline,manifest)


def test_domain_unit_review_and_reserved_anchor_role():
    registry=approved_registry(); registry['unit']='SYNTHETIC_UNIT'; registry['role']='discovery_candidate'
    registry.loc[0,'unit']='requires_source_review'
    with pytest.raises(ValueError,match='UNIT_REVIEW'): eligible_domain_map(registry,['a','b'])
    registry.loc[0,'unit']='SYNTHETIC_UNIT'; registry.loc[0,'role']='independent_baseline_anchor'
    eligible=eligible_domain_map(registry,['a','b'])
    assert list(eligible.index)==['b']

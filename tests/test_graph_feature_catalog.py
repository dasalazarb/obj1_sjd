"""Expanded Step-10 contract; synthetic observations only, no clinical releases."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.studies.graph import _feature_catalog as C
from src.studies.graph import _lab_contract as L

ROOT = Path(__file__).parents[1]

def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

P = module('catalog_pharma', 'src/studies/pharma/00_profile_labs.py')
G = module('catalog_graph01', 'src/studies/graph/01_run_topological_phenotyping.py')
S = module('catalog_graph02', 'src/studies/graph/02_graph_representation_sensitivity.py')


def frame(n=36):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        'patient_id': [f'SYNTHETIC_PATIENT_{i}' for i in range(n)],
        'clinical_episode_id': [f'baseline_{i}' for i in range(n)],
        'clinical_anchor_date': pd.Timestamp('2020-01-01'),
        'clinical_visit_number': 1, 'clinical_visit': True, 'is_clinical_baseline': True,
        'time_since_clinical_baseline_years': 0.,
        'pop__status': ['Pop1', 'Pop2', 'Pop3'] * (n//3),
        'essdai__total': rng.uniform(0, 20, n), 'esspri__total': rng.uniform(0, 10, n),
        'esspri__pain': rng.normal(size=n), 'essdai__articular_active': [0, 1]* (n//2),
        'ovl__status': ['neither', 'overlap'] * (n//2),
        'demo__age_at_baseline': rng.uniform(30, 70, n),
        'demo__sex': ['Female', 'Male'] * (n//2),
        'lab__complement_c3__value': rng.normal(100, 10, n),
        'lab__cbc_diff__variant_lymphocytes_blood__value': rng.normal(size=n),
        'lab__anti_ro_ssa__text': ['POSITIVE', 'NEGATIVE'] * (n//2),
        'lab__anti_ro_ssa__reference_status': 'normal',
        'lab__cryoglobulins__text': ['SEE NOTE', 'TRACE', '1+', '<3%'] * (n//4),
        'sero__baseline_anti_ro_ssa': [0, 1] * (n//2),
        'sero__ana__ever_positive_through_episode': [0, 1] * (n//2),
        'sero__hla_a__asof_value': ['A', 'B'] * (n//2),
        'sero__hla_a__patient_consensus_value': ['A', 'B'] * (n//2),
        **{f'pro__score_{i}': rng.normal(size=n) for i in range(12)},
        **{f'ext__ocular_schirmer_{side}': rng.uniform(0, 20, n) for side in ('right','left','min')},
        'ext__ocular_tbut_min': rng.uniform(0, 15, n),
        'ext__sgus_omeract_r_parotid_grade': rng.integers(0,4,n),
        'ext__salivary_flow_unstimulated': rng.uniform(0,.5,n),
        'ext__empty': np.nan,
    })


def test_all_columns_and_exact_names_177_stems_485_columns():
    from test_pharma_lab_profile_integration import schema_shaped_frame
    master = schema_shaped_frame()
    baseline = master.loc[master.is_clinical_baseline].reset_index(drop=True)
    manifest, values, _, _ = C.build_catalog(baseline, pd.DataFrame())
    assert manifest.source_column.nunique() == 485
    assert manifest.lab.replace('',np.nan).nunique() == 177
    assert len(manifest.loc[manifest.source_namespace.eq('sero')]) == 36
    assert manifest.loc[manifest.source_namespace.eq('ext'), 'source_column'].nunique() == 96
    assert not any(c.startswith('sero__') for c in values)


def test_discovery_policy_encoding_and_trace_are_exhaustive():
    baseline = frame()
    manifest, values, _, _ = G.build_feature_manifest(baseline, pd.DataFrame(), .70)
    assert set(manifest.source_column) == set(baseline)
    assert sum(k.startswith('pro__') for k in values) == 12
    assert all(f'ext__ocular_schirmer_{s}' in values for s in ('right','left','min'))
    assert 'lab__cbc_diff__variant_lymphocytes_blood__value' in values
    assert 'lab__anti_ro_ssa__text__binary' in values
    assert not any(c.startswith(('sero__','essdai__','ovl__','esspri__')) for c in values)
    assert not manifest.loc[manifest.included,'clinical_approval_status'].eq('approved').any()
    clinical, clinical_values, _, _ = C.build_catalog(baseline, pd.DataFrame(), analysis_mode='clinical_release')
    assert clinical.loc[clinical.source_column.eq('lab__complement_c3__value'),'technical_eligible'].item()
    assert not any(c.startswith('lab__') for c in clinical_values)
    _, inclusive, _, _ = C.build_catalog(baseline,pd.DataFrame(),feature_policy='clinical_inclusive')
    assert 'essdai__articular_active' in inclusive and 'esspri__pain' in inclusive
    assert not any(c in inclusive for c in C.OUTCOMES)
    row = manifest.set_index('source_column').loc['lab__cryoglobulins__text']
    assert not row.parseable and row.exclusion_reason == 'unverified_text_encoding'
    assert manifest.loc[manifest.source_column.eq('ext__empty'),'exclusion_reason'].item() == 'no_observed_baseline_data'


def test_representation_coverage_units_and_no_double_counting():
    b = frame()
    b['lab__anti_ro52__value'] = [1.,2.,3.] + [np.nan]*33
    b['lab__anti_ro52__text'] = ['POSITIVE','NEGATIVE']*18
    b['lab__complement_c3__text'] = ['POSITIVE','NEGATIVE']*18
    m,v,_,_ = C.build_catalog(b,pd.DataFrame())
    assert 'lab__anti_ro52__text__binary' in v and 'lab__anti_ro52__value' not in v
    assert 'lab__complement_c3__text__binary' not in v
    assert m.loc[m.source_column.eq('lab__complement_c3__text'),'exclusion_reason'].str.startswith('alternate_assay_representation').all()
    b['lab__complement_c3__unit'] = ['mg/dL','g/L']*18
    m,v,_,_ = C.build_catalog(b,pd.DataFrame())
    assert not any(c.startswith('lab__complement_c3__') for c in v)
    assert m.loc[m.source_column.eq('lab__complement_c3__value'),'exclusion_reason'].item() == 'multiple_units_not_harmonized'


def test_temporal_validation_is_exact_and_cannot_promote_consensus():
    b=frame()
    validation = {'no_future_data':True,'no_criteria_circularity':True,'knowledge_date_verified':True,'decision_id':'SYNTHETIC_ONLY'}
    metadata={c:{'temporal_validation':validation} for c in b if c.startswith('sero__')}
    m,v,_,_ = C.build_catalog(b,pd.DataFrame(),metadata=metadata)
    assert 'sero__baseline_anti_ro_ssa' in v
    assert not any('patient_consensus' in c for c in v)
    assert m.loc[m.source_column.str.endswith('patient_consensus_value'),'temporal_status'].eq('future_information').all()


def test_thresholds_safe_tokens_nan_and_missing_nominal():
    b=frame()
    b.loc[:11,'lab__anti_ro_ssa__text']=None
    b.loc[0,'demo__sex']=None
    m,v,_,_=C.build_catalog(b,pd.DataFrame())
    assert 'lab__anti_ro_ssa__text__binary' not in v
    assert m.loc[m.source_column.eq('lab__anti_ro_ssa__text'),'exclusion_reason'].item() == 'below_minimum_baseline_coverage'
    assert all(s.isna().iloc[0] for c,s in v.items() if c.startswith('demo__sex'))
    for bad in ['NaN',np.nan,None]:
        enc,_,_=C.encode(pd.Series([1.,2.,bad]),'ext__value')
        assert next(iter(enc.values())).isna().iloc[-1]
    for bad in ['TRACE','1+','<3%','SEE NOTE']:
        enc,_,reason=C.encode(pd.Series(['POSITIVE','NEGATIVE',bad]),'lab__x__text')
        assert not enc and reason
    enc,_,_=C.encode(pd.Series(['TRACE','1+','2+']), 'lab__urine__text', {'ordinal_values':{'trace':0,'1+':1,'2+':2}})
    assert next(iter(enc.values())).tolist()==[0.,1.,2.]


def bundle(tmp_path):
    master=frame(); ip=tmp_path/'master.parquet'; master.to_parquet(ip)
    pp=tmp_path/'profile'/'00_pharma_lab_profile.csv'
    P.run(P.parse_args(['--integrated',str(ip),'--output',str(pp)]))
    cp=pp.with_name(P.GRAPH_CANDIDATES_NAME)
    mp=tmp_path/'map.csv'
    pd.DataFrame([{'lab':C.lab_stem(c).removeprefix('lab__'),'s4_group':'SYNTHETIC','s4_subgroup':'test','mapping_status':'MAPPED'}
                  for c in master if C.lab_stem(c) and 'cryoglobulins' not in c]).to_csv(mp,index=False)
    return master,ip,cp,mp


@pytest.mark.parametrize('script,prefix',[(G,'01_graph'),(S,'02_graph')])
def test_real_three_script_boundary_preflight_and_privacy(tmp_path,script,prefix,monkeypatch):
    master,ip,cp,mp=bundle(tmp_path)
    output=tmp_path/'graph'
    argv=['--integrated',str(ip),'--graph-lab-candidates',str(cp),'--baseline-metrics',str(tmp_path/'missing.csv'),
          '--preflight-only','--output-dir',str(output)]
    if script is S: argv+=['--lab-group-map',str(mp)]
    monkeypatch.setattr(script,'fit_embedding' if script is S else 'fit_patient_embedding',lambda *a,**k:pytest.fail('fit during preflight'))
    script.run(script.parse_args(argv))
    trace=pd.read_csv(output/'tables'/f'{prefix}_all_source_columns_audit.csv')
    assert set(trace.source_column)==set(master)
    assert trace.loc[trace.included,'family'].isin(['lab','pro','ext','clinical']).all()
    summary=json.loads((output/'tables'/f'{prefix}_selection_summary.json').read_text())
    assert summary['analysis_mode']=='exploratory' and summary['clinical_validation']=='not_established'
    provenance=json.loads((output/'tables'/f'{prefix}_input_provenance.json').read_text())
    assert provenance['input_hashes']['integrated']['sha256']==C.file_hash(ip)
    assert sum(r['selected'] for r in summary['family_counts'] if r['family']=='EXT')==6
    for p in (output/'tables').glob('*'):
        assert 'SYNTHETIC_PATIENT_' not in p.read_text()
    with pytest.raises(FileExistsError): script.run(script.parse_args(argv))


def test_provenance_stale_outputs_source_and_no_lab_guard(tmp_path):
    master,ip,cp,_=bundle(tmp_path)
    G.load_real_inputs(ip,cp,'exploratory')
    old=cp.read_bytes(); cp.write_bytes(old+b'\n')
    with pytest.raises(ValueError,match='OUTPUT_HASH'): G.load_real_inputs(ip,cp,'exploratory')
    cp.write_bytes(old); ip.write_bytes(ip.read_bytes()+b'x')
    with pytest.raises(ValueError,match='SOURCE_HASH'): L.validate_provenance(ip,cp,'exploratory')
    m,_,_,_=C.build_catalog(frame().drop(columns=[c for c in frame() if c.startswith('lab__')]),pd.DataFrame())
    with pytest.raises(ValueError,match='NO_TECHNICALLY_ELIGIBLE_LABS'): L.enforce_lab_mode(m,analysis_mode='exploratory')
    assert L.enforce_lab_mode(m,analysis_mode='exploratory',allow_without_lab=True)['lab_modality']=='EXPLICIT_NO_LAB_SENSITIVITY'
    with pytest.raises(ValueError): L.enforce_lab_mode(m,analysis_mode='clinical_release')


def test_s0_graph01_alignment_s3_s4_and_seeded_embedding(tmp_path):
    master,ip,cp,mp=bundle(tmp_path)
    _,c,contract=G.load_real_inputs(ip,cp,'exploratory')
    m,v,_,_=G.build_feature_manifest(master,c,.70,analysis_mode='exploratory',source_hash=contract['input_hash'])
    cfg=G.load_config(ROOT/'src/studies/graph/config_sensitivity.yaml')
    fam={r.feature:S.family_name(r.family) for r in m.loc[m.included].itertuples()}
    s0,_,_=S.select_features(v,fam,'S0',cfg,m)
    assert C.frame_hash(pd.DataFrame(s0))==C.frame_hash(pd.DataFrame(v))
    s3,_,_=S.select_features(v,fam,'S3',cfg,m); s4,_,_=S.select_features(v,fam,'S4',cfg,m)
    pd.testing.assert_frame_equal(pd.DataFrame(s3),pd.DataFrame(s4))
    matrix,_,_=G.build_baseline_matrix(v)
    fit_cfg=G.load_config(ROOT/"src/studies/graph/config.yaml")
    a,_,_=G.fit_patient_embedding(matrix,fit_cfg); b,_,_=G.fit_patient_embedding(matrix,fit_cfg)
    np.testing.assert_allclose(a,b)


def test_s4_missing_only_blocks_s4_and_preserves_other_preflights(tmp_path):
    _,ip,cp,mp=bundle(tmp_path)
    pd.read_csv(mp).iloc[:1].to_csv(mp,index=False)
    out=tmp_path/'run'
    with pytest.raises(S.MappingError,match='only S4 blocked'):
        S.run(S.parse_args(['--integrated',str(ip),'--graph-lab-candidates',str(cp),'--lab-group-map',str(mp),
            '--preflight-only','--output-dir',str(out)]))
    audit=pd.read_csv(out/'tables'/'02_graph_S4_lab_group_mapping_audit.csv')
    assert 'MISSING' in set(audit.mapping_status)
    assert set(pd.read_csv(out/'tables'/'02_graph_multimodal_family_flow.csv').scenario)==set(S.SCENARIOS)


@pytest.mark.parametrize("script,prefix", [(G,"01_graph"), (S,"02_graph")])
def test_cli_no_readable_labs_fails_with_complete_audit(tmp_path,script,prefix):
    master=frame()
    for column in master:
        if C.lab_stem(column): master[column]="SEE NOTE"
    ip=tmp_path/"master.parquet";master.to_parquet(ip)
    output=tmp_path/"profile"/"00_pharma_lab_profile.csv"
    P.run(P.parse_args(["--integrated",str(ip),"--output",str(output)]))
    out=tmp_path/"graph"
    argv=["--integrated",str(ip),"--graph-lab-candidates",str(output.with_name(P.GRAPH_CANDIDATES_NAME)),
          "--preflight-only","--output-dir",str(out)]
    with pytest.raises(ValueError,match="NO_TECHNICALLY_ELIGIBLE_LABS"):
        script.run(script.parse_args(argv))
    audit=pd.read_csv(out/"tables"/f"{prefix}_all_source_columns_audit.csv")
    assert set(audit.source_column)==set(master)
    assert not audit.loc[audit.family.eq("lab"),"included_pca"].any()
    assert json.loads((out/"tables"/f"{prefix}_preflight.json").read_text())["status"]=="BLOCKED"


def test_pharma_new_fields_preserve_longitudinal_semantics_and_runs(tmp_path,monkeypatch):
    master=frame();ip=tmp_path/"master.parquet";master.to_parquet(ip)
    destination=tmp_path/"pharma"/"00_pharma_lab_profile.csv"
    old=P.profile_labs(master)
    profile=P.run(P.parse_args(["--integrated",str(ip),"--output",str(destination)]))
    pd.testing.assert_series_equal(old.recommended_use, profile.recommended_use)
    assert not profile.longitudinal_eligibility.any()
    assert profile.baseline_graph_eligibility.any()
    assert profile.clinical_approval_status.eq("pending").all()
    before={p.name:p.read_bytes() for p in destination.parent.iterdir()}
    with pytest.raises(FileExistsError): P.run(P.parse_args(["--integrated",str(ip),"--output",str(destination)]))
    assert before=={p.name:p.read_bytes() for p in destination.parent.iterdir()}
    summary=destination.with_name(P.SUMMARY_NAME)
    payload=json.loads(summary.read_text());payload["feature_contract_version"]="obsolete"
    summary.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="STALE_FEATURE_CONTRACT"):
        G.load_real_inputs(ip,destination.with_name(P.GRAPH_CANDIDATES_NAME),"exploratory")

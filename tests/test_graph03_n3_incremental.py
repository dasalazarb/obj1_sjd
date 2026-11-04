"""Negative controls and matched-population exploratory N3 models, all invented."""
import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from tests.test_graph03_n3_candidate_audit import synthetic_n3, audit, approve_synthetic, CATALOG, REFERENCE
from src.studies.graph.stage03.s03_n3_incremental import n3_incremental_models, _pop_status, METRICS
from src.studies.graph.stage03.s03_n3_validation import run_n3_validation, n3_validation_qc


def run(f):
    roles, tables=audit(f)
    result=n3_incremental_models(f[1],f[2],roles,tables['source_inventory'],f[3],f[0],CATALOG)
    return result, tables, roles


def specify(f,source='independent',**kwargs):
    f[0]['n3_validation']['prespecified_outcomes']=[dict(source_column=source,**kwargs)]


def test_empty_plan_and_pending_outcomes_never_emit_estimates():
    f=synthetic_n3()
    result, tables, roles=run(f)
    assert result['incremental_models'].status.eq('NOT_ESTIMABLE').all()
    assert result['model_comparison'].empty and result['descriptive_profiles'].empty
    for source in ['independent','n1','essdai__total']:
        specify(f,source)
        result,_,_=run(f)
        assert result['incremental_models'][METRICS].isna().all().all()
        assert result['incremental_models'].effects_json.eq('').all()
        assert result['model_comparison'].empty
    assert n3_validation_qc({**tables,**result},f[1],f[2],f[3],{},True)['no_unapproved_estimates']


def test_reviewed_descriptives_without_plan_do_not_select_outcome():
    f=synthetic_n3(approved=True)
    result,_,_=run(f)
    assert not result['descriptive_profiles'].empty
    assert result['incremental_models'].status.eq('NOT_ESTIMABLE').all()
    assert result['incremental_models'].reason.eq('NO_PRESPECIFIED_OUTCOMES').all()


def test_stable_reviewed_endpoint_estimates_a_b_on_common_patients_and_conditional_intervals():
    f=synthetic_n3(approved=True); specify(f)
    result,tables,roles=run(f)
    models=result['incremental_models'].set_index('model')
    assert models.loc[['A','B'],'status'].eq('COMPLETE').all()
    assert models.loc[['A','B'],'n_patients'].eq(100).all()
    assert models.loc['C','reason']=='N1_COMPARATOR_NOT_PRESPECIFIED'
    comparison=result['model_comparison'].iloc[0]
    assert comparison.contrast=='B_vs_A' and comparison.delta_r2>0
    assert comparison.delta_adjusted_r2==pytest.approx(models.loc['B','adjusted_r2']-models.loc['A','adjusted_r2'])
    assert pd.notna(comparison.delta_adjusted_r2_ci_low)
    effects=json.loads(models.loc['B','effects_json'])
    assert 'community_id_level_1' in effects and all('ci_low' in v for v in effects.values())
    assert models.loc['B','p_value']==comparison.p_value
    assert models.loc['B','q_value']==comparison.q_value
    qc=n3_validation_qc({**tables,**result},f[1],f[2],f[3],{},True)
    assert qc['n_models_complete']==2 and qc['common_model_population_verified']


def test_a_b_c_d_share_exact_population_even_with_different_missing_sources(monkeypatch):
    f=synthetic_n3(approved=True); specify(f)
    f[0]['n3_validation']['source_comparators']=[dict(representation_id='R0',source_columns=['n1'],
        design_review_status='APPROVED',source_reference=REFERENCE,missingness_policy='common_complete_case',include_model_d=True)]
    f[1].loc[:3,'independent']=np.nan
    f[1].loc[4:8,'n1']=np.nan
    f[1].loc[9:10,'demo__age_at_baseline']=np.nan
    import src.studies.graph.stage03.s03_n3_incremental as module
    fit=module._fit; captured=[]
    def capture(y,design,family):
        captured.append((tuple(y.index),tuple(y.to_numpy())))
        return fit(y,design,family)
    monkeypatch.setattr(module,'_fit',capture)
    result,_,_=run(f)
    models=result['incremental_models']
    assert models.status.eq('COMPLETE').all() and models.n_patients.eq(89).all()
    assert len(captured)==4 and all(captured[0]==item for item in captured)
    assert result['model_comparison'].n_patients.eq(89).all()
    assert set(result['model_comparison'].contrast)=={'B_vs_A','C_vs_A','D_vs_C'}
    assert 'patient_id' not in models


@pytest.mark.parametrize('failure,reason',[
    ('unstable','TOPOLOGY_NOT_STABLE'),('release','TOPOLOGY_NOT_STABLE'),
    ('missing_covariate','COVARIATES_UNAVAILABLE'),('era','BASELINE_ERA_UNVERIFIED'),
    ('singular','RANK_DEFICIENT'),('few','INSUFFICIENT_MODEL_SUPPORT'),
    ('constant_y','INSUFFICIENT_MODEL_SUPPORT')])
def test_no_delta_when_any_required_gate_fails(failure,reason):
    f=synthetic_n3(approved=True); specify(f)
    if failure=='unstable': f[2]['structural_stability_status']='provisional'
    if failure=='release': f[3]['release_status']='NO_STABLE_STRUCTURE'
    if failure=='missing_covariate': f[1].drop(columns='demo__sex',inplace=True)
    if failure=='era': f[0]['n3_validation']['baseline_era_reviews']=[]
    if failure=='singular': f[1]['demo__age_at_baseline']=1.
    if failure=='few': f[1].loc[10:,'independent']=np.nan
    if failure=='constant_y': f[1]['independent']=1.
    result,_,_=run(f)
    assert result['incremental_models'].reason.str.contains(reason).all()
    assert result['incremental_models'][METRICS].isna().all().all()
    assert result['model_comparison'].empty


def test_pop_cannot_compare_essdai_esspri_or_documented_derivative():
    f=synthetic_n3()
    for source in ['essdai__total','esspri__total','esspri__pain','eg_domain','ovl__status']:
        status,eligible=_pop_status(source,f[0],'R0',CATALOG)
        assert status=='POP_BASELINE_COMPARATOR_INELIGIBLE' and not eligible
    f[0]['characterization']['relationships'].append(dict(source_column='different_name',target_source='essdai__total',
        relationship_type='derived_from',source_reference=REFERENCE))
    assert not _pop_status('different_name',f[0],'R0',CATALOG)[1]
    assert not _pop_status('independent',f[0],'R0',CATALOG)[1]


def binary_fixture(n=200):
    f=synthetic_n3(n=n,approved=True)
    rng=np.random.default_rng(93)
    f[1]['independent']=rng.binomial(1,.5,n).astype(bool)
    f[0]['n3_validation']['measurement_policies'][0].update(category_review_status='VERIFIED',
        category_convention=dict(positive=['true'],negative=['false'],unknown=[],not_tested=[]))
    specify(f,model_family='binomial_logit',statistical_review_status='APPROVED',source_reference=REFERENCE,
            minimum_events_per_parameter=5)
    return f


def test_binary_models_require_review_event_support_and_reject_separation():
    f=binary_fixture()
    result,_,_=run(f)
    assert result['incremental_models'].set_index('model').loc[['A','B'],'status'].eq('COMPLETE').all()
    assert result['model_comparison'].delta_r2.isna().all()
    assert result['incremental_models'].set_index('model').loc['B','auc']>=.5
    f[0]['n3_validation']['prespecified_outcomes'][0].pop('minimum_events_per_parameter')
    result,_,_=run(f)
    assert result['incremental_models'].reason.str.contains('REVIEW_PENDING').all()
    f=binary_fixture(); f[1]['independent']=False; f[1].loc[:4,'independent']=True
    result,_,_=run(f)
    assert result['incremental_models'].reason.eq('INSUFFICIENT_EVENTS').all()
    f=binary_fixture(); f[1]['independent']=np.repeat([False,True],100)
    result,_,_=run(f)
    assert result['incremental_models'].reason.eq('SEPARATION').all()
    assert result['incremental_models'][METRICS].isna().all().all()


def test_prespecified_fdr_family_counts_include_blocked_tests():
    f=synthetic_n3(approved=True); specify(f)
    f[0]['n3_validation']['prespecified_outcomes'].append(dict(source_column='essdai__total'))
    result,_,_=run(f)
    comparison=result['model_comparison'].iloc[0]
    assert comparison.family_size==2
    assert comparison.q_value==pytest.approx(min(1,comparison.p_value*2))


def test_phase_b_error_keeps_phase_a_and_all_schemas_in_new_run(tmp_path,monkeypatch):
    f=synthetic_n3(); roles,_=audit(f)
    import src.studies.graph.stage03.s03_n3_validation as module
    monkeypatch.setattr(module,'n3_incremental_models',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('sensitive exception details')))
    manifest={**f[3],'graph02_release_sha256':'b'*64}
    tables,qc=run_n3_validation(tmp_path,dict(baseline_raw=f[1],membership=f[2],feature_sets=f[4],clinical_registry=f[5]),
        roles,f[3],f[0],dict(graph02_run_id='synthetic',graph02_release_sha256='b'*64),'c'*64)
    assert qc['phase_a_status']=='COMPLETE' and qc['phase_b_status']=='FAILED_WITH_AUDIT_RETAINED'
    assert len(pd.read_csv(tmp_path/'03_graph_n3_source_inventory.csv'))==len(tables['source_inventory'])
    for name in tables:
        assert len(pd.read_csv(tmp_path/f'03_graph_n3_{name}.csv').columns)>1
    assert 'sensitive exception details' not in (tmp_path/'03_graph_n3_validation_qc.json').read_text()
    assert not tables['source_inventory'].empty


def test_unapproved_source_comparator_remains_in_prespecified_family():
    f=synthetic_n3(approved=True); specify(f)
    f[0]['n3_validation']['source_comparators']=[dict(representation_id='R0',source_columns=['n1'],design_review_status='PENDING_REVIEW')]
    result,_,_=run(f)
    assert result['model_comparison'].family_size.eq(2).all()
    assert result['incremental_models'].set_index('model').loc['C','status']=='NOT_ESTIMABLE'


def test_registry_derived_outcome_cannot_use_pop_even_with_comparator_review():
    f=synthetic_n3(approved=True); specify(f)
    f[5].loc[f[5].source_column.eq('independent'),'derived_from']='essdai__total'
    f[0]['n3_validation']['pop_comparator_reviews']=[dict(representation_id='R0',source_column='independent',eligible=True,
        semantics_review_status='VERIFIED',outcome_dependency_review_status='APPROVED',source_reference=REFERENCE)]
    result,_,_=run(f)
    assert result['incremental_models'].pop_comparator_status.eq('POP_BASELINE_COMPARATOR_INELIGIBLE').all()

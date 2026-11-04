"""Phase C contracts: invented aggregate inputs, never clinical signoffs."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from src.studies.graph._pipeline import digest
from src.studies.graph.stage03.s03_n3_phasec_adjudication import (
    CANDIDATES, COMMON, audit_phasec_baseline_date, audit_phasec_candidate,
    audit_phasec_covariates, audit_phasec_pop_dictionary, build_phasec_tables,
    load_phasec_inputs, run_phasec, validate_phasec_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'src/studies/graph/configs/n3_phasec_review.yaml'


def config():
    return yaml.safe_load(CONFIG.read_text())


def inventory(reps=('R0', 'R2')):
    return pd.DataFrame([dict(representation_id=rep, source_column=source,
        tier_original='N3_CANDIDATE', review_state='PENDING_REVIEW', source_exists=True,
        n_baseline_total=100, n_hard=80, n_hard_supported=80,
        n_observed_unique=60, n_missing=20, n_stable_hard=50, n_provisional_hard=30,
        n_Y_nonmissing_with_all_covariates=0, missing_adjustments='demo__protocol|baseline_era',
        type='numeric' if not source.endswith('__text') else 'categorical',
        measurement_status='PENDING_REVIEW', clinical_proxy_alerts='',
        coverage_by_community=json.dumps([dict(n_observed=30), dict(n_observed=30)]))
        for rep in reps for source in CANDIDATES])


def test_essdai_dependence_and_esspri_conceptual_proxy_are_distinct():
    row = inventory().iloc[0].to_dict()
    row.update(source_column='essdai__total', tier_original='N2_RELATED')
    result = audit_phasec_candidate(row, {'eg_renal'}, {}, {})
    assert result['decision_status'] == 'BLOCKED_RELATED_OR_PROXY'
    assert result['clinical_status'] == 'NOT_ELIGIBLE'
    row.update(source_column='esspri__total', tier_original='N3_CANDIDATE')
    result = audit_phasec_candidate(row, {'pro__sf36_pcs'}, {}, {})
    assert result['evidence_status'] == 'PROXY_REVIEW_PENDING'
    assert result['clinical_status'] == 'PENDING_REVIEW'
    assert 'ESSPRI' in result['proxy_review_flag']
    assert result['pop_comparator_status'] == 'POP_BASELINE_COMPARATOR_INELIGIBLE'


def test_six_families_assay_variants_and_selection_denominators():
    inv = inventory()
    tables = build_phasec_tables(inv, pd.DataFrame(columns=['representation_id','source_column','tier']),
                                pd.DataFrame(columns=['representation_id','source_column','included']),
                                ROOT, config())
    candidates = tables['candidate_adjudication']
    assert candidates.candidate_family.nunique() == 6
    assert len(candidates) == 16
    assert len(tables['model_readiness']) == 64
    assert candidates.groupby(['representation_id','source_column']).size().eq(1).all()
    biopsy = candidates.loc[candidates.source_column.eq('ext__biopsy_focus_score')].iloc[0]
    assert biopsy.n_observed_unique == 60 and biopsy.n_hard_supported == 80
    assert 'SELECTION' in biopsy.sampling_bias_note
    ana = candidates.loc[candidates.candidate_family.eq('ana')]
    rf = candidates.loc[candidates.candidate_family.eq('rheumatoid_factor')]
    assert len(ana) == len(rf) == 4
    assert tables['measurement_review'].observation_eligibility.eq('PENDING_REVIEW').all()
    assert not set(['p_value','effect','ci_low','ci_high']) & set(tables['model_readiness'])
    assert tables['model_readiness'].decision_status.eq('NOT_ESTIMABLE').all()
    assert tables['model_readiness'].topology_gate.eq('BLOCKED_PROVISIONAL_TOPOLOGY').all()


def test_proteinuria_tokens_never_get_recoded_and_documents_do_not_approve():
    row = inventory().query("source_column == 'lab__urine_protein_qualitative__text'").iloc[0].to_dict()
    original = deepcopy(row)
    result = audit_phasec_candidate(row, set(), {'source_code_reference':'src/existing.py',
                                  'source_code_sha256':'a'*64}, {'status':'APPROVED'})
    assert row == original
    assert result['decision_status'] != 'APPROVED_ONLY_AFTER_HUMAN_SIGNOFF'
    assert result['clinical_status'] == 'PENDING_REVIEW'
    assert result['observation_eligibility'] == 'PENDING_REVIEW'
    assert 'NOT_SJD_RENAL_DIAGNOSIS' in result['sampling_bias_note']


def test_source_unavailable_no_evidence_and_proxy_pending_are_distinguishable():
    row = inventory().iloc[0].to_dict()
    assert audit_phasec_candidate(row, set(), {}, {})['evidence_status'] == 'NO_EVIDENCE'
    row['source_exists'] = False
    result = audit_phasec_candidate(row, set(), {}, {})
    assert result['evidence_status'] == 'SOURCE_UNAVAILABLE'
    assert result['decision_status'] == 'BLOCKED_UNAVAILABLE_OR_INVALID'
    row.update(source_exists=True, n_observed_unique=0)
    assert audit_phasec_candidate(row, set(), {}, {})['decision_status'] == 'BLOCKED_UNAVAILABLE_OR_INVALID'


def test_protocol_alias_does_not_fill_or_claim_equivalence():
    baseline = pd.DataFrame({'ids__protocol':['A','B',None]})
    before = baseline.copy(deep=True)
    result = audit_phasec_covariates(baseline, {}, {})
    assert result['decision_status'] == 'ABSENT_IN_FROZEN_BASELINE'
    assert result['alias_review_status'] == 'NOT_REVIEWED'
    assert result['available_aliases'] == 'ids__protocol'
    pd.testing.assert_frame_equal(before, baseline)
    baseline['demo__protocol'] = pd.NA
    assert audit_phasec_covariates(baseline, {}, {})['canonical_field_status'] == 'ALL_NULL'
    baseline['demo__protocol'] = ['A','B','A']
    result = audit_phasec_covariates(baseline, {}, {})
    assert result['decision_status'] == 'NOT_REVIEWED'


def test_baseline_era_is_separate_from_endpoint_and_requires_actual_review_evidence():
    baseline = pd.DataFrame({'clinical_anchor_date':['2020-01-01','2022-01-01'],
                             'clinical_baseline_date':['2020-01-01','2022-01-01']})
    result = audit_phasec_baseline_date(baseline, {'status':'APPROVED'}, {})
    assert result['decision_status'] == 'PENDING_REVIEW'
    assert result['endpoint_measurement_date_status'] == 'PENDING_REVIEW'
    assert result['longitudinal_status'] == 'BLOCKED'
    assert result['n_anchor_baseline_mismatches'] == 0
    baseline.loc[0, 'clinical_baseline_date'] = '2021-01-01'
    assert audit_phasec_baseline_date(baseline, {}, {})['n_anchor_baseline_mismatches'] == 1


def test_pop_exact_categories_and_other_tokens_never_leak_or_rewrite():
    series = pd.Series(['Pop1','Pop2','Pop3','Unclassifiable','Unclassified',None,
                        'PRIVATE_FREE_TEXT_2020-01-01'])
    before = series.copy()
    result = audit_phasec_pop_dictionary(series)
    assert result['n_total'] == 7
    assert result['n_unclassifiable'] == result['n_unclassified'] == result['n_other_invalid'] == 1
    assert result['reconciled'] and result['recoded'] is False
    assert 'PRIVATE_FREE_TEXT' not in json.dumps(result)
    pd.testing.assert_series_equal(series, before)


def test_config_disallows_automatic_approvals_and_unscoped_or_duplicate_reviews():
    cfg = config()
    validate_phasec_config(cfg)
    cfg['auto_approve_independence'] = True
    with pytest.raises(ValueError, match='AUTOMATIC'):
        validate_phasec_config(cfg)
    cfg = config()
    cfg['candidate_reviews'] = [dict(representation_id='R0', source_column='essdai__total')]*2
    with pytest.raises(ValueError, match='DUPLICATE'):
        validate_phasec_config(cfg)


def synthetic_bundle(tmp_path):
    """Minimal immutable frozen contract plus independent Phase A manifest."""
    from src.studies.graph.stage02.s02_contracts import CRITICAL, SCHEMA
    graph02 = tmp_path / 'SYNTHETIC_GRAPH02'
    graph02.mkdir()
    baseline = pd.DataFrame(dict(patient_id=[f'SYNTHETIC_{i}' for i in range(100)],
        ids__protocol=['p']*100, pop__status=['Unclassifiable']*100,
        clinical_anchor_date=['2020-01-01']*100))
    for source in CANDIDATES:
        baseline[source]=([1.0]*60+[None]*40) if not source.endswith('__text') else (['negative','trace']*30+[None]*40)
    membership = pd.DataFrame([dict(patient_id=p, representation_id=rep,community_id=0,
        hard_assignment=True, mapper_covered=True, membership_tie=False, supported=i<80,
        structural_stability_status='stable' if i<50 else 'provisional')
        for rep in ['R0','R2'] for i,p in enumerate(baseline.patient_id)])
    features = pd.DataFrame([dict(representation_id=rep,source_column='pro__sf36_pcs',included=True)
                             for rep in ['R0','R2']])
    paths = {role:role+'.csv' for role in CRITICAL | {'baseline_raw','membership','feature_sets'}}
    paths['release_qc']='release_qc.json'
    for role, filename in paths.items():
        target = graph02 / filename
        if role == 'release_qc':
            target.write_text(json.dumps(dict(release_status='PARTIAL_OR_EXPLORATORY',primary_qc='PASS')))
        else:
            {'baseline_raw':baseline,'membership':membership,'feature_sets':features}.get(
                role,pd.DataFrame({'synthetic':[1]})).to_csv(target,index=False)
    manifest = dict(schema_version=SCHEMA,finalized=True,run_id=graph02.name,git_commit='a'*40,
        study_mode='clinical_release',feature_policy='phenotype_independent',data_sha256='b'*64,
        pharma_profile_sha256='c'*64,config_sha256='d'*64,features_sha256=digest(graph02/paths['feature_sets']),
        baseline_definition_id='SYNTHETIC_ONLY',n_patients=100,n_features=1,n_covered=100,
        method_list=[],primary_reference_id='R0',eligible_representation_ids=['R0','R2'],
        release_status='PARTIAL_OR_EXPLORATORY',created_at_utc='SYNTHETIC',relative_paths=paths,
        sha256_by_artifact={p:digest(graph02/p) for p in paths.values()},source_inputs={},random_seed=1)
    release = graph02/'02_graph_release_manifest.json'
    release.write_text(json.dumps(manifest))
    graph03=tmp_path/'SYNTHETIC_GRAPH03'
    graph03.mkdir()
    inv=inventory()
    roles=inv[['representation_id','source_column','tier_original']].rename(columns={'tier_original':'tier'})
    tables = {'source_inventory':inv,'candidate_screen':inv,'lineage_review_queue':inv,
              'review_status':inv,'endpoint_support':inv.rename(columns={'n_observed_unique':'n_observed_Y'})}
    for name, table in tables.items(): table.to_csv(graph03/f'03_graph_n3_{name}.csv',index=False)
    roles.to_csv(graph03/'03_graph_clinical_feature_roles.csv',index=False)
    meta=dict(graph03_run_id=graph03.name, graph02_run_id=graph02.name,
              graph02_release_sha256=digest(release),implementation_commit_sha='e'*40)
    (graph03/'03_graph_n3_validation_qc.json').write_text(json.dumps(dict(**meta,
        phase_a_status='COMPLETE',schema_version='graph03_n3_audit_v1',status='PASS')))
    (graph03/'03_graph_release_input_qc.json').write_text(json.dumps(dict(**meta,status='PASS')))
    (graph03/'03_graph_n3_results_summary.md').write_text('Synthetic audit only')
    summary=dict(graph02_run_id=graph02.name,graph02_release_sha256=digest(release),
                 sha256_by_artifact={p.name:digest(p) for p in graph03.iterdir()})
    (graph03/'03_graph_results_summary.json').write_text(json.dumps(summary))
    cfg=config()
    cfg['input_contract']=dict(graph03_run_id=graph03.name, graph02_run_id=graph02.name,
        graph02_release_sha256=digest(release), implementation_commit_sha='e'*40,
        n3_schema_version='graph03_n3_audit_v1')
    cfg_path=tmp_path/'review.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))
    return graph03, release, cfg_path


def test_tampered_release_fails_before_output_creation(tmp_path):
    graph03, release, cfg = synthetic_bundle(tmp_path)
    release.write_text(release.read_text()+' ')
    out = tmp_path/'must_not_exist'
    with pytest.raises(ValueError, match='SHA'):
        run_phasec(graph03,release,out,cfg,ROOT)
    assert not out.exists()


def test_explicit_release_contract_and_phase_a_tampering(tmp_path):
    graph03, release, cfg = synthetic_bundle(tmp_path)
    settings=yaml.safe_load(cfg.read_text())
    data=load_phasec_inputs(graph03,release,settings)
    assert data['qc']['phase_a_status']=='COMPLETE'
    target=graph03/'03_graph_n3_source_inventory.csv'
    target.write_text(target.read_text()+'\n')
    with pytest.raises(ValueError,match='SHA'):
        load_phasec_inputs(graph03,release,settings)


def test_full_audit_no_approved_sources_and_no_overwrites(tmp_path):
    graph03, release, cfg = synthetic_bundle(tmp_path)
    protected={p:digest(p) for root in [graph03,release.parent] for p in root.iterdir()}
    out=tmp_path/'audit'
    result=run_phasec(graph03,release,out,cfg,ROOT)
    assert result['status']=='PASS'
    assert len(list(out.iterdir()))==11
    assert all(digest(p)==sha for p,sha in protected.items())
    qc=json.loads((out/'03_graph_n3_phasec_qc.json').read_text())
    assert qc['n_models_complete']==qc['n_automatic_approvals']==0
    assert qc['graph02_unchanged'] and qc['graph03_unchanged']
    for path in out.glob('*.csv'):
        frame=pd.read_csv(path)
        assert set(COMMON)<=set(frame)
        assert not {'patient_id','clinical_episode_id','measurement_date'} & set(frame)
        assert 'SYNTHETIC_0' not in path.read_text()
        assert '2020-01-01' not in path.read_text()
    artifacts=pd.read_csv(out/'03_graph_n3_phasec_artifact_inventory.csv')
    assert all(digest(out/r.relative_path)==r.sha256 for r in artifacts.itertuples())
    with pytest.raises(FileExistsError):
        run_phasec(graph03,release,out,cfg,ROOT)


def test_cli_requires_explicit_run_and_never_imports_pipeline_runner():
    cli=ROOT/'src/studies/graph/03c_graph_n3_clinical_adjudication.py'
    result=subprocess.run([sys.executable,str(cli),'--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--graph03-run' in result.stdout
    assert 's03_pipeline' not in cli.read_text()


def test_supplied_human_era_signoff_never_verifies_endpoint_dates():
    baseline = pd.DataFrame({'clinical_anchor_date':['2020-01-01']*10})
    signed = dict(status='DOCUMENTED', human_signoff=True, reviewer='SYNTHETIC_ONLY',
                  source_sha256='a'*64, date_semantics='official_baseline_anchor',
                  timezone_review_status='DOCUMENTED',lookahead_review_status='DOCUMENTED',
                  official_baseline_contract_review_status='DOCUMENTED')
    result = audit_phasec_baseline_date(baseline,signed,dict(verified=True,sha256='a'*64))
    assert result['decision_status']=='BASELINE_ERA_VERIFIED'
    assert result['endpoint_measurement_date_status']=='PENDING_REVIEW'
    assert result['longitudinal_status']=='BLOCKED'
    del signed['human_signoff']
    assert audit_phasec_baseline_date(baseline,signed,dict(verified=True,sha256='a'*64))['decision_status']=='PENDING_REVIEW'


def test_signed_protocol_non_equivalence_and_mapping_do_not_modify_baseline():
    baseline = pd.DataFrame({'ids__protocol':['A','B']*5})
    before = baseline.copy(deep=True)
    review = dict(status='DOCUMENTED',human_signoff=True,reviewer='SYNTHETIC_ONLY',source_sha256='a'*64,
                  canonical_source='ids__protocol',equivalent_to_prespecified_confounder=False)
    evidence = dict(verified=True,sha256='a'*64)
    result = audit_phasec_covariates(baseline,evidence,{'review':review})
    assert result['decision_status']=='AVAILABLE_BUT_NOT_EQUIVALENT'
    review['equivalent_to_prespecified_confounder']=True
    assert audit_phasec_covariates(baseline,evidence,{'review':review})['decision_status']=='AVAILABLE_WITH_VERIFIED_CANONICAL_MAPPING'
    assert audit_phasec_covariates(baseline,{}, {'review':review})['decision_status']=='ABSENT_IN_FROZEN_BASELINE'
    pd.testing.assert_frame_equal(baseline,before)


def test_documentary_package_only_requests_human_review_and_never_promotes():
    row = inventory().query("source_column == 'esspri__total'").iloc[0].to_dict()
    review = dict(status='DOCUMENTED',document_verified=True,baseline_definition='baseline_observed',
        date_semantics='instrument_measurement',derived_from_future='false',
        independence_status='DOCUMENTED',unit_status='DOCUMENTED')
    result = audit_phasec_candidate(row,set(),{},review)
    assert result['decision_status']=='HUMAN_REVIEW_REQUIRED'
    assert result['clinical_status']=='ELIGIBLE_FOR_HUMAN_REVIEW'
    assert result['original_review_state']=='PENDING_REVIEW'
    review['derived_from_future']='true'
    assert audit_phasec_candidate(row,set(),{},review)['clinical_status']=='NOT_ELIGIBLE'


def test_small_cells_and_derived_totals_are_suppressed_in_runtime_outputs(tmp_path):
    graph03, release, cfg = synthetic_bundle(tmp_path)
    target=graph03/'03_graph_n3_source_inventory.csv'
    inv=pd.read_csv(target)
    select=inv.source_column.eq('ext__biopsy_focus_score')
    inv.loc[select,'n_observed_unique']=3
    inv.loc[select,'n_missing']=77
    inv.to_csv(target,index=False)
    support=pd.read_csv(graph03/'03_graph_n3_endpoint_support.csv')
    support.loc[select,'n_observed_Y']=3
    support.to_csv(graph03/'03_graph_n3_endpoint_support.csv',index=False)
    summary_path=graph03/'03_graph_results_summary.json'
    summary=json.loads(summary_path.read_text())
    for path in [target,graph03/'03_graph_n3_endpoint_support.csv']:
        summary['sha256_by_artifact'][path.name]=digest(path)
    summary_path.write_text(json.dumps(summary))
    out=tmp_path/'suppressed_audit'
    run_phasec(graph03,release,out,cfg,ROOT)
    for filename in ['candidate_adjudication','source_lineage','measurement_review','human_review_queue','model_readiness']:
        frame=pd.read_csv(out/f'03_graph_n3_phasec_{filename}.csv')
        rows=frame.loc[frame.source_column.eq('ext__biopsy_focus_score')]
        assert rows.small_cells_suppressed.all()
        assert rows[[c for c in rows if c.startswith('n_')]].isna().all().all()


@pytest.mark.parametrize('change', ['wrong_run','wrong_commit','incomplete_phase_a','wrong_schema','wrong_denominator'])
def test_bad_phase_a_contracts_fail_before_writes(tmp_path, change):
    graph03, release, cfg = synthetic_bundle(tmp_path)
    target=graph03/'03_graph_n3_validation_qc.json'
    qc=json.loads(target.read_text())
    if change=='wrong_run': qc['graph03_run_id']='OTHER'
    if change=='wrong_commit': qc['implementation_commit_sha']='f'*40
    if change=='incomplete_phase_a': qc['phase_a_status']='PENDING'
    if change=='wrong_schema': qc['schema_version']='unknown'
    target.write_text(json.dumps(qc))
    summary_path=graph03/'03_graph_results_summary.json'
    summary=json.loads(summary_path.read_text())
    summary['sha256_by_artifact'][target.name]=digest(target)
    if change=='wrong_denominator':
        target=graph03/'03_graph_n3_source_inventory.csv'
        inv=pd.read_csv(target)
        inv.loc[0,'n_hard_supported']=99
        inv.to_csv(target,index=False)
        summary['sha256_by_artifact'][target.name]=digest(target)
    summary_path.write_text(json.dumps(summary))
    out=tmp_path/'invalid_audit'
    with pytest.raises(ValueError): run_phasec(graph03,release,out,cfg,ROOT)
    assert not out.exists()


def test_missing_source_produces_documentary_blocker_and_no_patient_record():
    inv=inventory().query("source_column != 'ext__biopsy_focus_score'")
    empty_roles=pd.DataFrame(columns=['representation_id','source_column','tier'])
    features=pd.DataFrame(columns=['representation_id','source_column','included'])
    tables=build_phasec_tables(inv,empty_roles,features,ROOT,config())
    missing=tables['candidate_adjudication'].query("source_column == 'ext__biopsy_focus_score'")
    assert len(missing)==2 and missing.evidence_status.eq('SOURCE_UNAVAILABLE').all()
    assert missing.n_observed_unique.isna().all()


def test_reference_release_invariants_are_reconciled_and_not_applied_to_future_runs():
    from src.studies.graph.stage03.s03_n3_phasec_adjudication import _reconcile_reference
    sources=list(CANDIDATES)+[f'SYNTHETIC_FIELD_{i}' for i in range(497)]
    coverage=[(86,84),(66,67),(58,55),(63,66),(42,45),(46,48),(44,37),(88,82)]
    rows=[]
    for rep in ['R0','R1','R2','R3','R4']:
        for i, source in enumerate(sources):
            selected=i<84 if rep!='R4' else i<80 or 84<=i<88
            n=coverage[i][rep=='R2'] if i<8 and rep in {'R0','R2'} else 0
            rows.append(dict(representation_id=rep,source_column=source,
                is_outcome_candidate_for_review=selected,n_baseline_total=159,
                n_observed_unique=n,n_hard_supported=92 if rep=='R0' else 87,
                n_stable_hard=62 if rep=='R0' else 0,n_Y_nonmissing_with_all_covariates=0))
    frame=pd.DataFrame(rows)
    qc=dict(n_reviewed_independent_sources=0,n_models_complete=0)
    _reconcile_reference(frame,qc)
    frame.loc[0,'n_observed_unique']=85
    with pytest.raises(ValueError,match='COVERAGE_DIVERGED'):
        _reconcile_reference(frame,qc)


def test_proteinuria_categories_preserved_in_frozen_source_without_exports(tmp_path):
    graph03, release, cfg=synthetic_bundle(tmp_path)
    baseline_path=release.parent/'baseline_raw.csv'
    baseline=pd.read_csv(baseline_path)
    source='lab__urine_protein_qualitative__text'
    baseline[source]=['negative','trace','unknown','not tested']*25
    baseline.to_csv(baseline_path,index=False)
    manifest=json.loads(release.read_text())
    manifest['sha256_by_artifact'][baseline_path.name]=digest(baseline_path)
    release.write_text(json.dumps(manifest))
    settings=yaml.safe_load(cfg.read_text())
    settings['input_contract']['graph02_release_sha256']=digest(release)
    cfg.write_text(yaml.safe_dump(settings))
    for name in ['03_graph_n3_validation_qc.json','03_graph_release_input_qc.json','03_graph_results_summary.json']:
        path=graph03/name
        document=json.loads(path.read_text())
        document['graph02_release_sha256']=digest(release)
        path.write_text(json.dumps(document))
    summary_path=graph03/'03_graph_results_summary.json'
    summary=json.loads(summary_path.read_text())
    for name in ['03_graph_n3_validation_qc.json','03_graph_release_input_qc.json']:
        summary['sha256_by_artifact'][name]=digest(graph03/name)
    summary_path.write_text(json.dumps(summary))
    original=digest(baseline_path)
    out=tmp_path/'token_audit'
    run_phasec(graph03,release,out,cfg,ROOT)
    assert digest(baseline_path)==original
    measurement=pd.read_csv(out/'03_graph_n3_phasec_measurement_review.csv')
    assert measurement.category_mapping_applied.eq(False).all()
    assert measurement.query('source_column == @source').observation_eligibility.eq('PENDING_REVIEW').all()

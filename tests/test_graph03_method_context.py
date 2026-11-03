"""Synthetic Graph03 acceptance tests; no NIH data or discovery refits."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
from src.studies.graph import _pipeline as pipe
from src.studies.graph.stage03.s03_release_reader import read_sensitivity_artifacts
from src.studies.graph.stage03.s03_construct_validation import feature_roles,anchor_lineage_audit
from src.studies.graph.stage03.s03_profiles import profile_tables,clinical_domains,numeric
from src.studies.graph.stage03.s03_method_context import method_context,missingness_interpretation,representation_evidence
from src.studies.graph.stage03.s03_longitudinal import longitudinal
from src.studies.graph.stage03.s03_reporting import figures
from tests.test_graph_pipeline import configuration,clinical_frames


def aggregate_release(tmp_path):
    artifact='02_graph_missingness_method_registry.csv'
    frame=pd.DataFrame([dict(experiment_id='R0-M',reference_id='R0',status='COMPLETE',reason='',
        n_patients=159,n_covered=118,mask_weight=.25),
        dict(experiment_id='R0-I',reference_id='R0',status='NOT_ESTIMABLE',reason='VALIDATION_FAILED',n_patients=159)])
    frame.to_csv(tmp_path/artifact,index=False)
    # Deliberately unreadable private sensitivity data. It must never be parsed.
    private='02_graph_missingness_patient_membership.csv'
    (tmp_path/private).write_bytes(b'\xff\x00PRIVATE')
    entries=[dict(artifact=name,sha256=pipe.digest(tmp_path/name),analysis_role='sensitivity',
                  graph03_defining=False,estimator_version='synthetic_v1') for name in (artifact,private)]
    manifest=dict(run_id='synthetic',study_mode='exploratory',feature_policy='phenotype_independent',
        eligible_representation_ids=['R0','R2'],sensitivity_artifacts=entries,
        missingness_sensitivity=dict(enabled=True,graph03_defining=False,estimator_version='synthetic_v1'),
        sha256_by_artifact={r['artifact']:r['sha256'] for r in entries})
    path=tmp_path/'02_graph_release_manifest.json'; pipe.write_json(path,manifest)
    return manifest,path


def test_aggregate_allowlist_hashes_guarantees_and_historical_release(tmp_path):
    manifest,path=aggregate_release(tmp_path)
    tables,qc=read_sensitivity_artifacts(manifest,path)
    assert set(tables)=={'method_registry'} and qc['status']=='AVAILABLE'
    assert qc['sources'][0]['sha256']==pipe.digest(tmp_path/manifest['sensitivity_artifacts'][0]['artifact'])
    assert qc['eligible_representation_ids']==['R0','R2']
    for obj in (manifest['missingness_sensitivity'],manifest['sensitivity_artifacts'][0]):
        obj.pop('graph03_defining')
        with pytest.raises(ValueError,match='non-defining guarantee'): read_sensitivity_artifacts(manifest,path)
        obj['graph03_defining']=False
    artifact=tmp_path/manifest['sensitivity_artifacts'][0]['artifact']
    artifact.write_text(artifact.read_text()+'\n')
    with pytest.raises(ValueError,match='SHA mismatch'): read_sensitivity_artifacts(manifest,path)
    manifest.pop('missingness_sensitivity')
    tables,qc=read_sensitivity_artifacts(manifest,path)
    assert not tables and qc['status']=='NOT_AVAILABLE_IN_RELEASE'
    context=method_context(tables,qc,manifest)
    assert set(context.status)=={'UNAVAILABLE_IN_RELEASE'}
    assert context.n_patients.isna().all() and context.n_common.isna().all()


def test_reject_patient_identifiers_in_aggregate(tmp_path):
    manifest,path=aggregate_release(tmp_path)
    entry=manifest['sensitivity_artifacts'][0]
    pd.DataFrame({'patient_id':['synthetic_patient']}).to_csv(tmp_path/entry['artifact'],index=False)
    entry['sha256']=manifest['sha256_by_artifact'][entry['artifact']]=pipe.digest(tmp_path/entry['artifact'])
    with pytest.raises(ValueError,match='Patient-level'): read_sensitivity_artifacts(manifest,path)


def test_method_context_preserves_actual_denominators_statuses_and_stability(tmp_path):
    manifest,path=aggregate_release(tmp_path)
    tables,qc=read_sensitivity_artifacts(manifest,path)
    tables['cross_method_comparisons']=pd.DataFrame([dict(experiment_id='R0-M',n_common=17,ari_primary=.3)])
    tables['method_stability']=pd.DataFrame([dict(representation_id='R0-M',n_evaluable=9,stability=.2)])
    tables['distance_support_summary']=pd.DataFrame([dict(experiment_id='R0-D',n_pairs=12561,n_estimable_pairs=11111)])
    tables['sparse_graph_comparisons']=pd.DataFrame([dict(experiment_id='R0-DG',n_common=12,ari_primary=.2)])
    manifest['missingness_sensitivity']['statuses']={'R0-D':dict(status='NOT_ESTIMABLE'), 'R0-DG':dict(status='COMPLETE')}
    context=method_context(tables,qc,manifest)
    row=context.loc[context.source_table.eq('cross_method_comparisons')].iloc[0]
    assert row.n_common==17 and row.concordance_denominator=='n_common'
    coverage=context.loc[context.source_table.eq('method_registry') & context.experiment_id.eq('R0-M')].iloc[0]
    assert coverage.n_patients==159 and coverage.n_covered==118 and coverage.mask_weight==.25
    assert context.loc[context.source_table.eq('method_stability'),'n_evaluable'].iloc[0]==9
    assert context.loc[context.experiment_id.eq('R0-I'),'status'].eq('NOT_ESTIMABLE').all()
    assert context.loc[context.experiment_id.eq('R0-D'),'status'].eq('NOT_ESTIMABLE').all()
    dg=context.loc[context.experiment_id.eq('R0-DG')].iloc[0]
    assert dg.mapper_or_sparse_graph=='sparse_graph' and 'does_not_establish' in dg.interpretation
    assert not context.graph03_defining.any()


def test_missingness_denominators_remain_separate():
    audit=pd.DataFrame([dict(representation_id='R0',group_type='global',group='all',n_patients=10,n_encoded_features=8,missing_fraction=.25),
        dict(representation_id='R0',group_type='clinical_domain',group='D01',n_patients=10,n_observation_units=2,observed_fraction=.6,missing_fraction=.25)])
    output=missingness_interpretation({'audit_all_representations':audit},pd.DataFrame())
    assert set(output.measurement_basis)=={'encoded_cells','source_observation_units'}
    assert output.loc[output.measurement_basis.eq('source_observation_units'),'missing_fraction'].isna().all()
    assert output.loc[output.measurement_basis.eq('encoded_cells'),'observed_fraction'].isna().all()


def test_transitive_lineage_blocks_instruments_and_proxies_per_representation():
    cfg=deepcopy(pipe.load_pipeline()['validation'])
    cfg['anchors'] += [dict(source_column='proxy',derived_from=['bridge'],type='numeric'),
                       dict(source_column='pro__sf36_mcs',type='numeric')]
    baseline=pd.DataFrame({c:[1.,2.] for c in ['essdai__total','esspri__total','proxy','pro__sf36_mcs','pop__status']})
    features=pd.DataFrame([dict(representation_id=rep,source_column=c,feature=c,included=True)
        for rep,cols in [('R0',['essdai__activity','esspri__pain','pro__sf36_pcs','discovery_lab']),('R2',['independent_lab'])] for c in cols])
    registry=pd.DataFrame([dict(source_column='bridge',derived_from='discovery_lab'),
                           dict(source_column='pop__status',derived_from='essdai__total')])
    roles=feature_roles(features,baseline,cfg,['R0','R2'],registry)
    for source in ['essdai__total','esspri__total','proxy','pro__sf36_mcs','pop__status']:
        assert roles.query("representation_id=='R0' and source_column==@source").validation_role.iloc[0]=='derived_from_defining'
    assert roles.query("representation_id=='R2' and source_column=='essdai__total'").validation_role.iloc[0]=='independent_baseline_anchor'
    audit=anchor_lineage_audit(roles,cfg)
    assert audit.query("representation_id=='R0'").decision.eq('BLOCKED_CIRCULAR').all()
    assert audit.query("representation_id=='R0'").conflicting_defining_sources.ne('').all()


def test_unknown_values_flow_domains_provisional_evidence_and_stability_gate():
    baseline,membership,manifest,cfg,roles=clinical_frames()
    # Unknown values must be missing even for categorical columns.
    baseline['category']='observed'; baseline.loc[:9,'category']=' Unknown '
    baseline.loc[10:14,'category']='not tested'
    baseline.loc[0,'essdai__total']=np.nan
    features=pd.DataFrame([dict(representation_id='R0',source_column='category',feature='category',included=True)])
    roles=feature_roles(features,baseline,cfg,['R0'])
    membership['structural_stability_status']='provisional'
    tables=profile_tables(baseline,membership,roles,manifest,cfg,9)
    counts=tables['baseline']
    assert (counts.n_observed+counts.n_missing==counts.n_patients).all()
    assert counts.loc[counts.source_column.eq('category'),'n_missing'].sum()==15
    assert numeric(pd.Series(['unknown','not tested','negative','positive'])).tolist()[2:]==[0.,1.]
    assert tables['flow'].n_provisional_hard.iloc[0]==len(baseline)
    assert tables['flow'].n_inference.iloc[0]==0 and tables['flow'].n_descriptive.iloc[0]==len(baseline)
    assert tables['anchors'].status.eq('NOT_ESTIMABLE').all() and tables['anchors'].p_value.isna().all()
    registry=pd.DataFrame([dict(source_column='essdai__total',clinical_domain='D03',derivation_group='activity',unit='points')])
    domains=clinical_domains(baseline,membership,roles,registry,manifest,5)
    assert set(domains.clinical_domain)=={'D03'}
    assert domains.discovery_relation.eq('external_descriptive_source_not_clinical_approval').all()
    assert domains.loc[domains.status.eq('DESCRIPTIVE'),'n_observed'].add(domains.n_missing).dropna().eq(domains.n_group.dropna()).all()
    flow=tables['flow'].assign(representation_id='R2')
    meta={**manifest,'eligible_representation_ids':['R2'],'primary_reference_id':'R0',
          'representations':{'R2':dict(evidence_level='EXPLORATORY_PROVISIONAL')},
          'clinical_mapping':dict(review_status='pending_clinical_review')}
    evidence=representation_evidence(meta,membership,flow,'frozen_sha').iloc[0]
    assert evidence.evidence_level=='EXPLORATORY_PROVISIONAL'
    assert evidence.clinical_mapping_status=='pending_clinical_review'
    assert evidence.analysis_role=='clinical_comparison_provisional'


def test_postbaseline_counts_two_visits_and_unavailable_endpoints():
    baseline,membership,manifest,cfg,roles=clinical_frames()
    cfg['longitudinal_endpoints']=['essdai__total','absent_endpoint']
    before=baseline.assign(clinical_anchor_date=pd.Timestamp('2019-01-01'))
    after=baseline.assign(clinical_anchor_date=pd.Timestamp('2021-01-01'))
    after.loc[:4,'essdai__total']=np.nan
    master=pd.concat([before,baseline,after],ignore_index=True)
    eligibility,models,trajectories=longitudinal(master,baseline,membership,manifest,cfg)
    known=eligibility.loc[eligibility.endpoint.eq('essdai__total')].iloc[0]
    assert known.n_post_index_visits==90 and known.n_missing_endpoint_visits==5
    assert known.n_patients==0 and known.n_visits==0  # Only one postbasal visit.
    assert known.status=='NOT_ESTIMABLE' and models.p_value.isna().all()
    absent=eligibility.loc[eligibility.endpoint.eq('absent_endpoint')].iloc[0]
    assert np.isnan(absent.n_patients) and absent.reason=='ENDPOINT_OR_INDEX_DATE_UNAVAILABLE'
    assert (trajectories.n_visits+trajectories.n_missing_visits==trajectories.n_total_visits).all()


def test_missing_covariates_do_not_silently_reduce_adjustments():
    from src.studies.graph.stage03.s03_incremental_value import incremental_models
    baseline,membership,manifest,cfg,roles=clinical_frames()
    cfg['confounders']=['missing_adjustment']
    result=incremental_models(baseline,membership,roles,manifest,cfg)
    assert result.reason.eq('PRESPECIFIED_COVARIATES_UNAVAILABLE').all()
    assert result.p_value.isna().all()


def test_every_endpoint_gets_a_source_backed_figure_and_small_cells_are_suppressed(tmp_path):
    trajectory=pd.DataFrame([dict(representation_id='R0',community_id='0',endpoint=endpoint,
        time_bin_year=t,n_patients=n,n_missing_patients=0,n_visits=n,median=median)
        for endpoint in ['endpoint_one','endpoint_two'] for t,n,median in [(1,10,2.),(2,3,99999.)]])
    pop=pd.DataFrame([dict(representation_id='R0',community_id='0',population='Pop1',n_patients=3),
                      dict(representation_id='R0',community_id='0',population='Pop2',n_patients=10)])
    figures(pd.DataFrame(),trajectory,pop,tmp_path,minimum_cell=5)
    for endpoint in ['endpoint_one','endpoint_two']:
        for extension in ['csv','png','pdf']: assert (tmp_path/f'03_graph_longitudinal_{endpoint}.{extension}').is_file()
        source=pd.read_csv(tmp_path/f'03_graph_longitudinal_{endpoint}.csv')
        assert source.n_patients.tolist()==[10] and source['median'].tolist()==[2.]
    crosswalk=pd.read_csv(tmp_path/'03_graph_pop_crosswalk.csv')
    assert crosswalk.empty
    for path in tmp_path.glob('*.csv'):
        assert 'patient_id' not in path.read_text() and '99999' not in path.read_text()


def test_pipeline_frozen_release_and_historical_compatibility(configuration,tmp_path,monkeypatch):
    from src.studies.graph.stage02.s02_pipeline import run as run02
    from src.studies.graph.stage03.s03_pipeline import run as run03
    from src.studies.graph.stage03.s03_release_reader import read_release
    from src.studies.graph.stage02 import s02_mapper_engine
    cfg=configuration
    for key in ['run_mapper_grid','run_domain_and_family_ablations','run_negative_controls',
                'run_structural_comparators','run_lens_distance_sensitivity','run_observation_diagnostics']:
        cfg['experiments'][key]=False
    result=run02(cfg)
    manifest,path,frames=read_release(cfg,result['manifest'])
    frozen_hash=pipe.digest(path)
    pointer=pipe.output_root(cfg,'02')/'latest_completed_02.json'
    pointer_hash=pipe.digest(pointer)
    monkeypatch.setattr(s02_mapper_engine,'analyze',lambda *a,**k:pytest.fail('Graph03 cannot refit'))
    output=run03(cfg,graph02_release=path)['run_dir']
    assert pipe.digest(path)==frozen_hash and pipe.digest(pointer)==pointer_hash
    evidence=pd.read_csv(output/'03_graph_representation_evidence.csv')
    r2=evidence.loc[evidence.representation_id.eq('R2')].iloc[0]
    assert r2.evidence_level=='EXPLORATORY_PROVISIONAL' and r2.clinical_mapping_status=='pending_clinical_review'
    for name in ['patient_population_flow','longitudinal_models']:
        table=pd.read_csv(output/f'03_graph_{name}.csv')
        assert not table.representation_id.str.contains('-M|-DG',regex=True).any()
    context=pd.read_csv(output/'03_graph_method_sensitivity_context.csv')
    assert {'R0-M','R2-M','R0-DG','R2-DG'}<=set(context.experiment_id)
    for name in ['clinical_domains_by_community','anchor_lineage_audit','missingness_interpretation','validation_status']:
        assert (output/f'03_graph_{name}.csv').is_file()
    summary=json.loads((output/'03_graph_results_summary.json').read_text())
    assert not summary['confirmatory_validation'] and not summary['external_prediction_validated']
    inventory=pd.read_csv(output/'03_graph_artifact_inventory.csv')
    assert not inventory.shareable_export_allowed.any() and inventory.privacy_class.eq('INTERNAL_RESTRICTED').all()
    for figure in (output/'figures').glob('*.csv'):
        text=figure.read_text()
        assert 'patient_id' not in text
        assert not any(patient in text for patient in frames['baseline_raw'].patient_id)
    # A copy of the synthetic historical release lacks the new optional keys.
    historical=tmp_path/'historical'/path.parent.name
    shutil.copytree(path.parent,historical)
    old=deepcopy(manifest)
    old.pop('missingness_sensitivity'); old.pop('sensitivity_artifacts')
    old_path=historical/path.name; pipe.write_json(old_path,old)
    old_output=run03(cfg,graph02_release=old_path)['run_dir']
    old_qc=json.loads((old_output/'03_graph_release_input_qc.json').read_text())
    assert old_qc['missingness_sensitivity']['status']=='NOT_AVAILABLE_IN_RELEASE'
    assert set(pd.read_csv(old_output/'03_graph_method_sensitivity_context.csv').status)=={'UNAVAILABLE_IN_RELEASE'}
    # Private sensitivity memberships cannot be relabelled as the primary input.
    bad=deepcopy(manifest)
    bad['relative_paths']['membership']=next(e['artifact'] for e in manifest['sensitivity_artifacts'] if e['artifact'].endswith('_patient_membership.csv'))
    pipe.write_json(old_path,bad)
    with pytest.raises(ValueError,match='Sensitivity artifacts cannot enter primary'):
        read_release(cfg,old_path)
    # Each requested optional source is checked, even after release validation.
    entry=manifest['sensitivity_artifacts'][0]
    tampered=path.parent/entry['artifact']; tampered.write_text(tampered.read_text()+'\n')
    with pytest.raises(ValueError,match='SHA mismatch'): read_release(cfg,path)
    assert pipe.digest(pointer)==pointer_hash


def test_common_longitudinal_comparison_uses_intersection_only():
    from src.studies.graph.stage03.s03_longitudinal import comparable_longitudinal_summary
    baseline,membership,manifest,cfg,roles=clinical_frames()
    r2=membership.copy().assign(representation_id='R2')
    r2.loc[:19,'mapper_covered']=False
    r2.loc[:19,'hard_assignment']=False
    master=baseline.assign(clinical_anchor_date=pd.Timestamp('2021-01-01'))
    output=comparable_longitudinal_summary(master,baseline,pd.concat([membership,r2]),
        {**manifest,'eligible_representation_ids':['R0','R2']},cfg)
    assert output.n_common.eq(70).all()
    for identifier in ['R0','R2']:
        part=output.loc[output.representation_id.eq(identifier) & output.endpoint.eq('essdai__total')]
        assert part.n_patients.sum()==70 and part.n_visits.sum()==70
    assert 'p_value' not in output


def test_a_provisional_rest_blocks_an_otherwise_stable_anchor_community():
    baseline,membership,manifest,cfg,roles=clinical_frames()
    membership.loc[membership.community_id.eq(2),'structural_stability_status']='provisional'
    anchors=profile_tables(baseline,membership,roles,manifest,cfg,33)['anchors']
    assert anchors.status.eq('NOT_ESTIMABLE').all()
    assert anchors.p_value.isna().all()


def test_missing_longitudinal_adjustments_are_explicit():
    baseline,membership,manifest,cfg,roles=clinical_frames()
    cfg['confounders']=['absent_adjustment']
    cfg['longitudinal_endpoints']=['essdai__total']
    master=pd.concat([baseline.assign(clinical_anchor_date=pd.Timestamp('2021-01-01')),
                      baseline.assign(clinical_anchor_date=pd.Timestamp('2022-01-01'))])
    eligibility,models,trajectories=longitudinal(master,baseline,membership,manifest,cfg)
    assert eligibility.reason.eq('PRESPECIFIED_COVARIATES_UNAVAILABLE').all()
    assert models.p_value.isna().all()



def test_small_missingness_cells_are_not_disclosed_as_rates(tmp_path):
    missingness=pd.DataFrame([dict(representation_id='R0',measurement_basis='source_observation_units',
        group_type='clinical_domain',group='D01',n_patients=100,n_observation_units=1,observed_fraction=.99)])
    figures(pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),tmp_path,missingness=missingness,minimum_cell=5)
    source=pd.read_csv(tmp_path/'03_graph_missingness_domain.csv')
    assert source.empty

"""Invented records only: source grain, time, support, privacy and provenance."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from src.studies.graph import _pipeline as pipe
from src.studies.graph.stage03.s03_episode_audit import resolve_endpoint_episodes, resolve_all_endpoints
from src.studies.graph.stage03.s03_longitudinal import post_index_data,longitudinal,pop_transitions,comparable_longitudinal_summary


def synthetic_policy(config, keys=None):
    """An attestation about invented fixtures, never an approval of real data."""
    config=deepcopy(config)
    config.setdefault('temporal_resolution',{})['endpoint_policies']={
        endpoint:dict(date_field=config['date_column'],date_semantics_review_status='VERIFIED',
            date_semantics='endpoint_measurement_date',evidence='synthetic fixture: dates and observation identities assigned explicitly',
            identity_review_status='VERIFIED',episode_key_fields=keys or ['patient_id',config['date_column']])
        for endpoint in [*config['longitudinal_endpoints'],config['pop_column']]}
    return config


def fixture():
    cfg=synthetic_policy(pipe.load_pipeline()['validation'],['clinical_episode_id'])
    cfg['longitudinal_endpoints']=['essdai__total','esspri__total']; cfg['confounders']=[]
    baseline=pd.DataFrame(dict(patient_id=['invented_a','invented_b'],clinical_anchor_date=['2020-01-01']*2,pop__status=['Pop1','Pop2']))
    master=pd.DataFrame(dict(patient_id=['invented_a']*5+['invented_b'],
        clinical_episode_id=['pre','base','v1','v1','v2','v3'],
        clinical_anchor_date=['2019-01-01','2020-01-01','2021-01-01','2021-01-01','2022-07-01','2021-04-01'],
        essdai__total=[9,8,2,2,3,np.nan],esspri__total=[7,6,4,4,5,np.nan],
        pop__status=['Pop1','Pop1','Pop2','Pop2','Pop3','Unclassified']))
    members=pd.DataFrame(dict(patient_id=baseline.patient_id,representation_id='R0',community_id=[0,1],
        mapper_covered=True,hard_assignment=True,membership_tie=False,supported=True,structural_stability_status='provisional'))
    return master,baseline,members,dict(eligible_representation_ids=['R0'],release_status='PARTIAL_OR_EXPLORATORY'),cfg


def test_exact_duplicates_reconcile_before_observation_filter_and_membership():
    master,baseline,members,meta,cfg=fixture()
    result=resolve_endpoint_episodes(master,baseline,'essdai__total',cfg)
    assert len(result.observations)==3
    assert result.audits['resolution'].n_collapsed_technical_records.iloc[0]==1
    assert result.audits['collision_summary'].classification.tolist()==['EXACT_DUPLICATE_SAME_EPISODE']
    selected=members[['patient_id','community_id','structural_stability_status']]
    data=post_index_data(master,baseline,selected,'essdai__total',cfg,resolution=result)
    assert data.value.tolist()==[2,3]
    assert data.time_years.iloc[0]==pytest.approx(366/365.25)
    assert len(post_index_data(master,baseline,selected,'essdai__total',cfg,observed_only=False,resolution=result))==3
    for table in result.audits.values(): assert 'patient_id' not in table


def test_conflict_excludes_only_affected_date_not_patient_or_other_endpoint():
    master,baseline,members,meta,cfg=fixture();master.loc[3,'essdai__total']=99
    result=resolve_endpoint_episodes(master,baseline,'essdai__total',cfg)
    assert result.status=='PARTIALLY_RECONCILED'
    assert result.observations.loc[result.observations.patient_id.eq('invented_a'),'value'].tolist()==[3]
    assert 'UNRESOLVED_CONFLICT' in set(result.unresolved.reason)
    # Endpoint identities are audited separately: another instrument's conflict
    # must not invalidate this unchanged measurement.
    other=resolve_endpoint_episodes(master,baseline,'esspri__total',cfg)
    assert len(other.observations)==3 and other.unresolved.empty
    master=master.drop(index=3)
    both=resolve_all_endpoints(master,baseline,cfg)
    assert len(both['essdai__total'].observations)==len(both['esspri__total'].observations)==3


def test_distinct_same_day_episodes_not_independent_visits_or_technical_repeats():
    master,baseline,members,meta,cfg=fixture();master.loc[3,'clinical_episode_id']='different_real_episode'
    result=resolve_endpoint_episodes(master,baseline,'essdai__total',cfg)
    assert 'DISTINCT_EPISODES_SAME_DAY' in set(result.audits['collision_summary'].classification)
    assert result.audits['resolution'].n_collapsed_technical_records.iloc[0]==0
    assert not result.observations.visit_date.eq(pd.Timestamp('2021-01-01',tz='UTC')).any()


def test_unverified_anchor_does_not_fabricate_followup():
    master,baseline,members,meta,cfg=fixture();cfg['temporal_resolution']['endpoint_policies']={}
    master['unapproved_real_date']=master.clinical_anchor_date
    master['clinical_anchor_date']='2020-01-01'
    result=resolve_endpoint_episodes(master,baseline,'essdai__total',cfg)
    assert result.reason=='UNVERIFIED_FOLLOWUP_MEASUREMENT_DATE' and result.observations.empty
    eligibility,models,trajectories=longitudinal(master,baseline,members,meta,cfg)
    assert eligibility.reason.eq(result.reason).all() and trajectories.empty and models.p_value.isna().all()


def test_two_postbaseline_measurements_required_not_two_rows():
    master,baseline,members,meta,cfg=fixture();master=master.loc[master.clinical_episode_id.ne('v2')]
    eligibility,models,trajectories=longitudinal(master,baseline,members,meta,cfg)
    assert eligibility.n_patients.eq(0).all() and eligibility.n_visits.eq(0).all()
    assert eligibility.n_observed_endpoint_visits.eq(1).all()
    assert eligibility.n_missing_endpoint_visits.eq(1).all()
    assert eligibility.descriptive_status.eq('DESCRIPTIVE_PROVISIONAL').all()
    assert models.p_value.isna().all() and not trajectories.empty
    assert (trajectories.n_visits+trajectories.n_missing_visits==trajectories.n_total_visits).all()
    assert trajectories.bin_upper_years.sub(trajectories.bin_lower_years).eq(1).all()


def test_missing_endpoint_and_invalid_date_do_not_block_other_analyses():
    master,baseline,members,meta,cfg=fixture();cfg['longitudinal_endpoints'].append('absent')
    master.loc[4,'clinical_anchor_date']='malformed'
    eligibility,models,trajectories=longitudinal(master,baseline,members,meta,cfg)
    assert eligibility.loc[eligibility.endpoint.eq('absent'),'reason'].eq('ENDPOINT_OR_INDEX_DATE_UNAVAILABLE').all()
    assert 'essdai__total' in set(trajectories.endpoint)
    result=resolve_endpoint_episodes(master,baseline,'essdai__total',cfg)
    assert 'INVALID_OR_MISSING_DATE' in set(result.unresolved.reason)


def test_bad_cardinality_fails_explicitly():
    master,baseline,members,meta,cfg=fixture()
    with pytest.raises(ValueError,match='BASELINE_NOT_ONE_TO_ONE'):
        resolve_endpoint_episodes(master,pd.concat([baseline,baseline.iloc[:1]]),'essdai__total',cfg)
    with pytest.raises(ValueError,match='MEMBERSHIP_NOT_ONE_TO_ONE'):
        post_index_data(master,baseline,pd.concat([members,members]),'essdai__total',cfg)


def test_pop_first_and_all_followup_have_distinct_additivity():
    master,baseline,members,meta,cfg=fixture()
    transitions,eligibility=pop_transitions(master,baseline,members,meta,cfg,return_eligibility=True)
    first=transitions.loc[transitions.transition_definition.eq('baseline_to_first_valid_followup')]
    all_visits=transitions.loc[transitions.transition_definition.eq('baseline_to_all_followups')]
    assert first.n_patients.sum()==1 and first.additive_patient_counts.all()
    assert all_visits.n_patient_visit_pairs.sum()==2 and all_visits.n_patients.sum()==2
    assert all_visits.n_unique_patients_total.eq(1).all() and not all_visits.additive_patient_counts.any()
    assert eligibility.n_unclassified_followup_pop.eq(1).all()
    master.loc[3,'pop__status']='Pop3'
    tied=pop_transitions(master,baseline,members,meta,cfg)
    first=tied.loc[tied.transition_definition.eq('baseline_to_first_valid_followup')]
    assert first.status.eq('NOT_ESTIMABLE').all() and first.reason.eq('AMBIGUOUS_FOLLOWUP_ORDER').all()
    # A later verified date remains usable only in the all-followups sensitivity.
    assert tied.loc[tied.transition_definition.eq('baseline_to_all_followups'),'n_patient_visit_pairs'].sum()==1


def test_topology_barrier_survives_successful_temporal_resolution():
    from tests.test_graph_pipeline import clinical_frames
    baseline,membership,meta,cfg,roles=clinical_frames()
    membership['structural_stability_status']='provisional'
    cfg['longitudinal_endpoints']=['essdai__total','esspri__total']
    master=pd.concat([baseline.assign(clinical_anchor_date='2021-01-01'),baseline.assign(clinical_anchor_date='2022-01-01')],ignore_index=True)
    eligibility,models,trajectories=longitudinal(master,baseline,membership,meta,cfg)
    assert eligibility.reason.eq('TOPOLOGY_NOT_STABLE').all() and models.p_value.isna().all()
    assert not trajectories.empty and trajectories.status.eq('DESCRIPTIVE_PROVISIONAL').all()


def test_one_identity_cannot_change_measurement_date():
    master,baseline,members,meta,cfg=fixture()
    master.loc[4,'clinical_episode_id']='v1'
    result=resolve_endpoint_episodes(master,baseline,'essdai__total',cfg)
    assert 'CONFLICTING_DATES_SAME_EPISODE' in set(result.unresolved.reason)
    assert result.observations.loc[result.observations.patient_id.eq('invented_a')].empty


def test_later_unresolved_pop_does_not_remove_an_earlier_verified_first_visit():
    master,baseline,members,meta,cfg=fixture()
    extra=master.loc[[4]].copy();extra['pop__status']='Pop1'
    master=pd.concat([master,extra],ignore_index=True)
    transitions=pop_transitions(master,baseline,members,meta,cfg)
    first=transitions.loc[transitions.transition_definition.eq('baseline_to_first_valid_followup')]
    assert first.n_patients.sum()==1 and first.followup_pop.tolist()==['Pop2']


def test_figures_keep_domain_availability_and_instrument_magnitudes_separate(tmp_path):
    from src.studies.graph.stage03.s03_reporting import figures
    domains=pd.DataFrame([dict(representation_id='R0',community_id='0',source_column=f'source_{i}',
        clinical_domain='D01',n_group=30,n_observed=20,n_missing=10,observed_fraction=2/3,
        observed_value=10**i,unit=f'unit_{i}',status='DESCRIPTIVE') for i in range(8)])
    baseline=pd.DataFrame([dict(representation_id=rep,community_id='0',source_column=endpoint,
        validation_role='defining',n_patients=30,n_observed=20,n_missing=10,median=value,q25=value-1,q75=value+2)
        for rep in ['R0','R2'] for endpoint,value in [('essdai__total',5),('esspri__total',3)]])
    markers=[dict(source_column='essdai__total',label='ESSDAI',unit='ESSDAI points'),
             dict(source_column='esspri__total',label='ESSPRI',unit='ESSPRI points')]
    figures(pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),tmp_path,domains=domains,baseline_table=baseline,marker_config=markers)
    domain=pd.read_csv(tmp_path/'03_graph_domains_heatmap.csv')
    assert len(domain)==1 and domain.n_observed.iloc[0]==160 and domain.n_denominator.iloc[0]==240
    assert domain.observed_fraction.iloc[0]==pytest.approx(2/3) and 'observed_value' not in domain
    for marker,value in zip(markers,[5,3]):
        name='03_graph_clinical_marker_panels_'+marker['source_column']
        source=pd.read_csv(tmp_path/f'{name}.csv')
        assert source['median'].eq(value).all() and source.unit.eq(marker['unit']).all()
        assert source.loc[source.representation_id.eq('R2'),'status'].eq('DESCRIPTIVE_PROVISIONAL').all()
        assert all((tmp_path/f'{name}.{ext}').exists() for ext in ['csv','png','pdf'])
    for path in tmp_path.glob('*.csv'): assert 'patient_id' not in path.read_text()


def test_unreviewed_schema_has_no_active_hypothetical_field_policy():
    config=pipe.load_pipeline()['validation']
    assert config['temporal_resolution']['endpoint_policies']=={}
    assert not config['temporal_resolution']['allow_arbitrary_first_last_mean']
    assert config['minimum_visits_per_patient']==2


@pytest.mark.parametrize('location,column',[('baseline','clinical_anchor_date'),('master','clinical_anchor_date')])
def test_unavailable_date_field_is_not_estimable_not_an_unhandled_timezone_error(location,column):
    master,baseline,members,meta,cfg=fixture()
    if location=='baseline': baseline=baseline.drop(columns=column)
    else: master=master.drop(columns=column)
    result=resolve_endpoint_episodes(master,baseline,'essdai__total',cfg)
    assert result.reason=='ENDPOINT_OR_INDEX_DATE_UNAVAILABLE' and result.observations.empty

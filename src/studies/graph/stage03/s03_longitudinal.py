"""Real-time post-index eligibility and parsimonious random-intercept trajectories."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from .s03_release_reader import hard_members
from .s03_profiles import numeric,observed_values
from .s03_statistics import bh_qvalues
from .s03_episode_audit import resolve_endpoint_episodes, resolve_all_endpoints, validate_unique_patients


def post_index_data(master,baseline,members,endpoint,config, *, observed_only=True,resolution=None):
    validate_unique_patients(members, 'MEMBERSHIP')
    resolution=resolution or resolve_endpoint_episodes(master,baseline,endpoint,config)
    if resolution.reason: raise ValueError(resolution.reason)
    data=resolution.observations[['patient_id','visit_date','index_date','time_years','value']].merge(members,on='patient_id',validate='many_to_one')
    if observed_only: data=data.loc[data.value.notna()].copy()
    return data


def longitudinal(master,baseline,membership,manifest,config, *, resolutions=None):
    resolutions=resolutions or resolve_all_endpoints(master,baseline,config)
    eligibility=[]; models=[]; trajectories=[]
    for identifier in manifest['eligible_representation_ids']:
        members=hard_members(membership,identifier)
        for endpoint in config['longitudinal_endpoints']:
            row={'representation_id':identifier,'endpoint':endpoint,'status':'NOT_ESTIMABLE','n_patients':np.nan,'n_visits':np.nan,
                 'time_unit':'years_since_frozen_clinical_baseline','analysis_population':'strictly_post_index_observed_values',
                 'n_baseline':len(baseline),'n_mapper_covered':membership.loc[membership.representation_id.eq(identifier) & membership.mapper_covered,'patient_id'].nunique(),
                 'n_hard_supported':len(members),'n_model_patients':0,'n_model_visits':0,
                 'n_stable_hard':int(members.structural_stability_status.eq('stable').sum()),
                 'n_excluded_from_inference_by_stability':int(members.structural_stability_status.ne('stable').sum()),
                 'stability_interpretation':'frozen_Graph02_label_not_clinical_bootstrap_robustness',
                 'temporal_resolution_status':resolutions[endpoint].status,
                 'date_semantics_review_status':resolutions[endpoint].date_review,
                 'descriptive_status':'NOT_ESTIMABLE','model_status':'NOT_ESTIMABLE',
                 **{k:np.nan for k in ['n_post_index_patients','n_post_index_visits','n_observed_endpoint_patients',
                     'n_observed_endpoint_visits','n_missing_endpoint_visits','n_observed_post_index_visits',
                     'n_observed_post_index_patients','n_patients_with_2plus_observed_post_index_visits',
                     'min_followup_years','median_followup_years','max_followup_years']},
                 'n_verified_post_index_visits':0}
            rejected=resolutions[endpoint].unresolved
            rejected=rejected.loc[rejected.patient_id.isin(members.patient_id)]
            row['n_unresolved_episodes']=len(rejected[['patient_id','visit_date']].drop_duplicates())
            row['n_excluded_ambiguous_visits']=len(rejected.loc[rejected.post_index,['patient_id','visit_date']].drop_duplicates()) if 'post_index' in rejected else 0
            row['temporal_exclusion_reasons']='|'.join(sorted(set(rejected.reason)))
            row['n_raw_records']=int(resolutions[endpoint].audits['resolution'].n_raw_records.iloc[0])
            model={**row,'p_value':np.nan,'hypothesis_family':f'{identifier}/longitudinal',
                   'inference_scope':'in_sample_exploratory_conditional_trajectory','community_selection_uncertainty':'not_included_conditional_intervals'}
            try:
                all_visits=post_index_data(master,baseline,members,endpoint,config,observed_only=False,resolution=resolutions[endpoint])
                data=all_visits.loc[all_visits.value.notna()].copy()
                row.update(n_post_index_patients=all_visits.patient_id.nunique(),n_post_index_visits=len(all_visits),
                    n_observed_post_index_visits=len(data),n_missing_endpoint_visits=int(all_visits.value.isna().sum()),
                    n_observed_post_index_patients=data.patient_id.nunique(),
                    n_verified_post_index_visits=len(all_visits),n_observed_endpoint_patients=data.patient_id.nunique(),
                    n_observed_endpoint_visits=len(data),
                    min_followup_years=all_visits.time_years.min(),median_followup_years=all_visits.time_years.median(),
                    max_followup_years=all_visits.time_years.max(),
                    descriptive_status='DESCRIPTIVE_PROVISIONAL' if identifier=='R2' or not members.structural_stability_status.eq('stable').all() else 'DESCRIPTIVE')
                if all_visits.empty: row['descriptive_status']='NOT_ESTIMABLE'
                for (community,timebin),part in all_visits.groupby(['community_id',np.floor(all_visits.time_years)]):
                    observed=part.loc[part.value.notna()]
                    trajectories.append({'representation_id':identifier,'endpoint':endpoint,'community_id':community,
                        'time_bin_year':timebin,'bin_lower_years':timebin,'bin_upper_years':timebin+1,
                        'bin_interval':'[lower,upper)','time_median_years':float(part.time_years.median()),
                        'n_verified_visits':len(part),
                        'n_stable_patients':part.loc[part.structural_stability_status.eq('stable'),'patient_id'].nunique(),
                        'n_provisional_patients':part.loc[part.structural_stability_status.ne('stable'),'patient_id'].nunique(),
                        'stability_interpretation':'frozen_label_not_clinical_bootstrap_robustness','date_semantics_review_status':resolutions[endpoint].date_review,'n_visits':len(observed),'n_patients':observed.patient_id.nunique(),
                        'n_total_visits':len(part),'n_missing_visits':int(part.value.isna().sum()),
                        'n_total_patients':part.patient_id.nunique(),
                        'n_missing_patients':len(set(part.patient_id)-set(observed.patient_id)),
                        'median':float(observed.value.median()) if len(observed) else np.nan,
                        'q25':float(observed.value.quantile(.25)) if len(observed) else np.nan,
                        'q75':float(observed.value.quantile(.75)) if len(observed) else np.nan,
                        'time_min_years':float(part.time_years.min()),'time_max_years':float(part.time_years.max()),
                        'status':row['descriptive_status'],'inference_scope':'descriptive_no_confirmatory_p_values'})
                counts=data.groupby('patient_id').size()
                eligible=counts.loc[counts.ge(int(config['minimum_visits_per_patient']))].index
                row['n_patients_with_2plus_observed_post_index_visits']=int(counts.ge(2).sum())
                data=data.loc[data.patient_id.isin(eligible)].copy()
                row.update(n_patients=data.patient_id.nunique(),n_visits=len(data),n_communities=data.community_id.nunique(),
                    followup_min_years=float(data.time_years.min()) if len(data) else np.nan,
                    followup_max_years=float(data.time_years.max()) if len(data) else np.nan)
                if data.patient_id.nunique()<int(config['minimum_model_patients']) or data.community_id.nunique()<2:
                    raise ValueError('INSUFFICIENT_POST_INDEX_PATIENTS_VISITS_OR_COMMUNITIES')
                sizes=data.groupby('community_id').patient_id.nunique()
                if sizes.min()<int(config['minimum_patients_per_community']): raise ValueError('INSUFFICIENT_COMMUNITY_SUPPORT')
                if manifest['release_status']=='NO_STABLE_STRUCTURE' or not members.structural_stability_status.eq('stable').all():
                    raise ValueError('TOPOLOGY_NOT_STABLE')
                terms=[]; missing_covariates=[]
                frozen_covariates=baseline[['patient_id']+[c for c in config['confounders'] if c in baseline]].copy()
                data=data.merge(frozen_covariates,on='patient_id',validate='many_to_one')
                for i,c in enumerate(config['confounders']):
                    if c not in data or data[c].isna().all(): missing_covariates.append(c); continue
                    key=f'covariate_{i}'; values=observed_values(data[c]); s=numeric(values)
                    if values.isna().all(): missing_covariates.append(c); continue
                    is_numeric=s.notna().sum()==values.notna().sum()
                    data[key]=s if is_numeric else values.astype('object')
                    terms.append(key if is_numeric else f'C({key})')
                if missing_covariates: raise ValueError('PRESPECIFIED_COVARIATES_UNAVAILABLE')
                data=data.dropna(subset=[f'covariate_{i}' for i,c in enumerate(config['confounders']) if c not in missing_covariates])
                # Complete-case losses can remove a patient's second visit or community support.
                counts=data.groupby('patient_id').visit_date.nunique()
                data=data.loc[data.patient_id.isin(counts[counts.ge(int(config['minimum_visits_per_patient']))].index)]
                if data.patient_id.nunique()<int(config['minimum_model_patients']): raise ValueError('INSUFFICIENT_COMPLETE_COVARIATE_PATIENTS')
                if data.community_id.nunique()<2 or data.groupby('community_id').patient_id.nunique().min()<int(config['minimum_patients_per_community']):
                    raise ValueError('INSUFFICIENT_COMMUNITY_SUPPORT_AFTER_COVARIATES')
                formula='value ~ time_years * C(community_id)'+''.join(' + '+x for x in terms)
                with warnings.catch_warnings(record=True) as caught:
                    fitted=smf.mixedlm(formula,data,groups=data.patient_id).fit(reml=False,method='lbfgs',disp=False)
                if any('singular' in str(w.message).lower() or 'converg' in str(w.message).lower() for w in caught):
                    raise ValueError('MODEL_NONCONVERGENCE_OR_SINGULARITY')
                if not fitted.converged or np.linalg.matrix_rank(fitted.model.exog)<fitted.model.exog.shape[1]:
                    raise ValueError('MIXED_MODEL_NONCONVERGENCE_OR_RANK_DEFICIENCY')
                if not np.isfinite(fitted.fe_params).all() or not np.isfinite(fitted.cov_re).all().all() or np.asarray(fitted.cov_re).min()<=0:
                    raise ValueError('MODEL_SINGULARITY_OR_NONFINITE_ESTIMATE')
                ci=fitted.conf_int()
                if not np.isfinite(ci).all().all(): raise ValueError('MODEL_NONFINITE_INTERVAL')
                selected=[x for x in fitted.fe_params.index if 'time_years:' in x]
                if not selected: raise ValueError('NO_ESTIMABLE_COMMUNITY_TIME_INTERACTION')
                row.update(status='COMPLETE',model_status='COMPLETE',reason='',n_model_patients=data.patient_id.nunique(),n_model_visits=len(data))
                for term in selected:
                    models.append({**model,**row,'term':term,'estimate':float(fitted.fe_params[term]),
                        'ci_low':float(ci.loc[term,0]),'ci_high':float(ci.loc[term,1]),'p_value':float(fitted.pvalues[term]),
                        'formula':formula,'missing_adjustments':','.join(missing_covariates),
                        'n_fit_warnings':len(caught),'random_effect':'patient_intercept','followup_unit':'real_elapsed_years'})
            except (ValueError,np.linalg.LinAlgError) as exc:
                row.update(reason='MODEL_SINGULARITY' if isinstance(exc,np.linalg.LinAlgError) else str(exc).split(':')[0]); models.append({**model,**row})
            row['model_status']=row['status']
            eligibility.append(row)
    models=pd.DataFrame(models); models['q_value']=np.nan
    if 'hypothesis_family' in models:
        for _,idx in models.groupby('hypothesis_family').groups.items(): models.loc[idx,'q_value']=bh_qvalues(models.loc[idx,'p_value'])
    return pd.DataFrame(eligibility),models,pd.DataFrame(trajectories)


def pop_transitions(master,baseline,membership,manifest,config, *, resolutions=None,return_eligibility=False):
    """Two explicit descriptive estimands on the same audited source-visit contract."""
    pop=config['pop_column']; rows=[]; eligibility=[]
    resolutions=resolutions or resolve_all_endpoints(master,baseline,config)
    resolution=resolutions[pop]
    labels=['Pop1','Pop2','Pop3']
    for representation in manifest['eligible_representation_ids']:
        members=hard_members(membership,representation)
        validate_unique_patients(members,'MEMBERSHIP')
        starts=baseline[['patient_id']].copy()
        starts['baseline_pop']=baseline[pop] if pop in baseline else pd.NA
        data=resolution.observations.merge(starts,on='patient_id',validate='many_to_one').merge(members,on='patient_id',validate='many_to_one')
        rejected=resolution.unresolved.loc[resolution.unresolved.patient_id.isin(members.patient_id)]
        # An unresolved earlier or undated observation prevents identifying the
        # first valid follow-up for that patient. Later verified visits can still
        # contribute to the all-followups sensitivity.
        ambiguous=rejected.loc[rejected.visit_date.isna() | rejected.get('post_index',pd.Series(False,index=rejected.index))]
        uncertain=set()
        hard_starts=starts.loc[starts.patient_id.isin(members.patient_id)]
        before=data.copy()
        data=data.loc[data.baseline_pop.isin(labels) & data.value.isin(labels)].copy()
        for definition,additive in [('baseline_to_first_valid_followup',True),('baseline_to_all_followups',False)]:
            first=data.sort_values('visit_date').groupby('patient_id',sort=False).head(1)
            first_dates=first.set_index('patient_id').visit_date
            for patient,group in ambiguous.groupby('patient_id'):
                if group.visit_date.isna().any() or patient not in first_dates or group.visit_date.min()<=first_dates[patient]: uncertain.add(patient)
            selected=first.loc[~first.patient_id.isin(uncertain)] if additive else data
            n_unique=selected.patient_id.nunique()
            base=dict(representation_id=representation,transition_definition=definition,
                denominator_population='frozen_supported_hard_baseline_with_classified_Pop_and_verified_followup',
                additive_patient_counts=additive,n_excluded_ambiguous=len(uncertain),
                n_unique_patients_total=n_unique,n_patient_visit_pairs_total=len(selected),
                eligibility_exclusion_counts='separate_audit_dimensions_not_additive',
                interpretation='observed_population_labels_not_transit_through_graph')
            eligibility.append({**base,'n_baseline':len(baseline),'n_hard_supported':len(members),
                'n_missing_baseline_pop':int(hard_starts.baseline_pop.isna().sum()),
                'n_unclassified_baseline_pop':int(hard_starts.baseline_pop.eq('Unclassified').sum()),
                'n_invalid_baseline_pop':int((hard_starts.baseline_pop.notna() & ~hard_starts.baseline_pop.isin(labels+['Unclassified'])).sum()),
                'n_verified_followup_patients':before.patient_id.nunique(),
                'n_without_verified_followup':len(members)-before.patient_id.nunique(),
                'n_missing_followup_pop':int(before.value.isna().sum()),
                'n_unclassified_followup_pop':int(before.value.eq('Unclassified').sum()),
                'n_invalid_followup_pop':int((before.value.notna() & ~before.value.isin(labels+['Unclassified'])).sum()),
                'n_patients':n_unique,'n_visits':len(selected),
                'date_semantics_review_status':resolution.date_review,
                'status':'DESCRIPTIVE' if n_unique else 'NOT_ESTIMABLE',
                'reason':resolution.reason or ('' if n_unique else 'AMBIGUOUS_FOLLOWUP_ORDER' if uncertain else 'NO_VALID_POST_INDEX_POP_LABELS')})
            for (community,start,end),part in selected.groupby(['community_id','baseline_pop','value']):
                rows.append({**base,'community_id':community,'baseline_pop':start,'followup_pop':end,
                    'n_patients':part.patient_id.nunique(),'n_unique_patients':part.patient_id.nunique(),
                    'n_visits':len(part),'n_patient_visit_pairs':len(part),'status':'DESCRIPTIVE','reason':'',
                    'time_min_years':float(part.time_years.min()),'time_max_years':float(part.time_years.max())})
            if selected.empty:
                rows.append({**base,'status':'NOT_ESTIMABLE','reason':eligibility[-1]['reason'],
                    'n_patients':0,'n_unique_patients':0,'n_visits':0,'n_patient_visit_pairs':0})
    result=pd.DataFrame(rows)
    return (result,pd.DataFrame(eligibility)) if return_eligibility else result


def comparable_longitudinal_summary(master,baseline,membership,manifest,config, *, resolutions=None):
    resolutions=resolutions or resolve_all_endpoints(master,baseline,config)
    """R0/R2 descriptions on the same frozen supported-hard patient subset."""
    if not {'R0','R2'}<=set(manifest['eligible_representation_ids']):
        return pd.DataFrame([{'status':'NOT_ESTIMABLE','reason':'R0_R2_COMPARISON_UNAVAILABLE'}])
    reference=hard_members(membership,'R0'); comparison=hard_members(membership,'R2')
    common=set(reference.patient_id)&set(comparison.patient_id)
    rows=[]
    for identifier,members in [('R0',reference),('R2',comparison)]:
        for endpoint in config['longitudinal_endpoints']:
            row=dict(representation_id=identifier,endpoint=endpoint,n_common=len(common),
                analysis_population='common_R0_R2_supported_hard_patients',status='NOT_ESTIMABLE',
                n_patients=np.nan,n_visits=np.nan,inference_scope='descriptive_conditional_no_temporal_graph_progression')
            try:
                visits=post_index_data(master,baseline,members.loc[members.patient_id.isin(common)],endpoint,config,observed_only=False,resolution=resolutions[endpoint])
                if visits.empty:
                    rows.append({**row,'reason':'NO_COMMON_POST_INDEX_OBSERVATIONS'}); continue
                for community,group in visits.groupby('community_id'):
                    observed=group.loc[group.value.notna()]
                    rows.append({**row,'community_id':community,'status':'DESCRIPTIVE_PROVISIONAL','reason':'',
                        'n_patients':observed.patient_id.nunique(),'n_visits':len(observed),'n_total_visits':len(group),
                        'n_common_without_observed_endpoint':len(common)-visits.loc[visits.value.notna(),'patient_id'].nunique(),
                        'community_alignment':'labels_are_representation_specific_not_equivalent',
                        'n_missing_visits':int(group.value.isna().sum()),
                        'median':float(observed.value.median()) if len(observed) else np.nan,
                        'followup_min_years':float(group.time_years.min()),'followup_max_years':float(group.time_years.max())})
            except ValueError as exc: rows.append({**row,'reason':str(exc)})
    return pd.DataFrame(rows)

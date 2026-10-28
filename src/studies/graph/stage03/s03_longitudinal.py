"""Real-time post-index eligibility and parsimonious random-intercept trajectories."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from .s03_release_reader import hard_members
from .s03_profiles import numeric
from .s03_statistics import bh_qvalues


def post_index_data(master,baseline,members,endpoint,config):
    date=config['date_column']
    if endpoint not in master or date not in master or date not in baseline:
        raise ValueError('ENDPOINT_OR_INDEX_DATE_UNAVAILABLE')
    index=baseline[['patient_id',date]].rename(columns={date:'index_date'}).copy()
    index['index_date']=pd.to_datetime(index.index_date,errors='coerce')
    if index.index_date.isna().any(): raise ValueError('UNVERIFIED_BASELINE_INDEX_DATE')
    keep=['patient_id',date,endpoint]
    flag=config['baseline_flag']
    if flag in master: keep.append(flag)
    data=master[keep].copy(); data['patient_id']=data.patient_id.astype('string')
    data=data.merge(index,on='patient_id',validate='many_to_one').merge(members,on='patient_id',validate='many_to_one')
    data['visit_date']=pd.to_datetime(data[date],errors='coerce')
    data['time_years']=(data.visit_date-data.index_date).dt.total_seconds()/(365.25*86400)
    data['value']=numeric(data[endpoint])
    # Strictly later than t0, regardless of ordering or visit-number labels.
    data=data.loc[data.time_years.gt(0) & data.value.notna()].copy()
    if flag in data: data=data.loc[~data[flag].fillna(False).astype(bool)]
    if data.duplicated(['patient_id','visit_date']).any():
        raise ValueError('DUPLICATE_PATIENT_DATES_REQUIRE_EPISODE_REVIEW')
    return data


def longitudinal(master,baseline,membership,manifest,config):
    eligibility=[]; models=[]; trajectories=[]
    for identifier in manifest['eligible_representation_ids']:
        members=hard_members(membership,identifier)
        for endpoint in config['longitudinal_endpoints']:
            row={'representation_id':identifier,'endpoint':endpoint,'status':'NOT_ESTIMABLE','n_patients':0,'n_visits':0,
                 'time_unit':'years_since_frozen_clinical_baseline','analysis_population':'strictly_post_index_observed_values'}
            model={**row,'p_value':np.nan,'hypothesis_family':f'{identifier}/longitudinal',
                   'inference_scope':'in_sample_exploratory_conditional_trajectory','community_selection_uncertainty':'not_included_conditional_intervals'}
            try:
                data=post_index_data(master,baseline,members,endpoint,config)
                counts=data.groupby('patient_id').size()
                eligible=counts.loc[counts.ge(int(config['minimum_visits_per_patient']))].index
                data=data.loc[data.patient_id.isin(eligible)].copy()
                row.update(n_patients=data.patient_id.nunique(),n_visits=len(data),n_communities=data.community_id.nunique(),
                    followup_min_years=float(data.time_years.min()) if len(data) else np.nan,
                    followup_max_years=float(data.time_years.max()) if len(data) else np.nan)
                for (community,timebin),part in data.groupby(['community_id',data.time_years.round(0)]):
                    trajectories.append({'representation_id':identifier,'endpoint':endpoint,'community_id':community,
                        'time_bin_year':timebin,'n_visits':len(part),'n_patients':part.patient_id.nunique(),
                        'median':float(part.value.median()),'q25':float(part.value.quantile(.25)),'q75':float(part.value.quantile(.75)),
                        'time_min_years':float(part.time_years.min()),'time_max_years':float(part.time_years.max())})
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
                    key=f'covariate_{i}'; s=numeric(data[c])
                    is_numeric=s.notna().sum()==data[c].notna().sum()
                    data[key]=s if is_numeric else data[c].astype('object')
                    terms.append(key if is_numeric else f'C({key})')
                data=data.dropna(subset=[f'covariate_{i}' for i,c in enumerate(config['confounders']) if c not in missing_covariates])
                if data.patient_id.nunique()<int(config['minimum_model_patients']): raise ValueError('INSUFFICIENT_COMPLETE_COVARIATE_PATIENTS')
                formula='value ~ time_years * C(community_id)'+''.join(' + '+x for x in terms)
                with warnings.catch_warnings(record=True) as caught:
                    fitted=smf.mixedlm(formula,data,groups=data.patient_id).fit(reml=False,method='lbfgs',disp=False)
                if not fitted.converged or np.linalg.matrix_rank(fitted.model.exog)<fitted.model.exog.shape[1]:
                    raise ValueError('MIXED_MODEL_NONCONVERGENCE_OR_RANK_DEFICIENCY')
                ci=fitted.conf_int(); selected=[x for x in fitted.fe_params.index if 'time_years:' in x]
                if not selected: raise ValueError('NO_ESTIMABLE_COMMUNITY_TIME_INTERACTION')
                row.update(status='COMPLETE',reason='',n_model_patients=data.patient_id.nunique(),n_model_visits=len(data))
                for term in selected:
                    models.append({**model,**row,'term':term,'estimate':float(fitted.fe_params[term]),
                        'ci_low':float(ci.loc[term,0]),'ci_high':float(ci.loc[term,1]),'p_value':float(fitted.pvalues[term]),
                        'formula':formula,'missing_adjustments':','.join(missing_covariates),
                        'n_fit_warnings':len(caught),'random_effect':'patient_intercept','followup_unit':'real_elapsed_years'})
            except (ValueError,np.linalg.LinAlgError) as exc:
                row.update(reason=str(exc).split(':')[0]); models.append({**model,**row})
            eligibility.append(row)
    models=pd.DataFrame(models); models['q_value']=np.nan
    for _,idx in models.groupby('hypothesis_family').groups.items(): models.loc[idx,'q_value']=bh_qvalues(models.loc[idx,'p_value'])
    return pd.DataFrame(eligibility),models,pd.DataFrame(trajectories)


def pop_transitions(master,baseline,membership,manifest,config):
    """Observed label transitions at actual post-index visits, not graph motion."""
    pop=config['pop_column']; date=config['date_column']; rows=[]
    if any(c not in master for c in [pop,date]) or any(c not in baseline for c in [pop,date]):
        return pd.DataFrame([{'status':'NOT_ESTIMABLE','reason':'POP_OR_INDEX_DATE_UNAVAILABLE'}])
    observed=master[['patient_id',date,pop]].copy(); observed['patient_id']=observed.patient_id.astype('string')
    observed['visit_date']=pd.to_datetime(observed[date],errors='coerce')
    for representation in manifest['eligible_representation_ids']:
        members=hard_members(membership,representation)
        starts=baseline[['patient_id',date,pop]].rename(columns={date:'index_date',pop:'baseline_pop'})
        data=observed.merge(starts,on='patient_id',validate='many_to_one').merge(members,on='patient_id',validate='many_to_one')
        data['index_date']=pd.to_datetime(data.index_date,errors='coerce')
        data=data.loc[data.visit_date.gt(data.index_date) & data[pop].isin(['Pop1','Pop2','Pop3']) & data.baseline_pop.isin(['Pop1','Pop2','Pop3'])]
        for (community,start,end),part in data.groupby(['community_id','baseline_pop',pop]):
            rows.append({'representation_id':representation,'community_id':community,'baseline_pop':start,'followup_pop':end,
                'n_patients':part.patient_id.nunique(),'n_visits':len(part),'status':'DESCRIPTIVE',
                'time_min_years':float(((part.visit_date-part.index_date).dt.days/365.25).min()),
                'time_max_years':float(((part.visit_date-part.index_date).dt.days/365.25).max()),
                'interpretation':'observed_population_labels_not_transit_through_graph'})
        if data.empty: rows.append({'representation_id':representation,'status':'NOT_ESTIMABLE','reason':'NO_VALID_POST_INDEX_POP_LABELS'})
    return pd.DataFrame(rows)

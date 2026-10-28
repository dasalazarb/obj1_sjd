"""Exploratory conditional incremental associations; no predictive claims."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import statsmodels.formula.api as smf
from .s03_release_reader import hard_members
from .s03_profiles import numeric
from .s03_statistics import bh_qvalues


def pop_crosswalk(baseline,membership,identifiers,config):
    rows=[]; comparisons=[]; pop=config['pop_column']
    for identifier in identifiers:
        frame=hard_members(membership,identifier).merge(baseline,on='patient_id',validate='one_to_one')
        if pop not in frame:
            rows.append({'representation_id':identifier,'status':'NOT_ESTIMABLE','reason':'POP_UNAVAILABLE'}); continue
        part=frame.loc[frame[pop].isin(['Pop1','Pop2','Pop3'])]
        for (community,population),count in part.groupby(['community_id',pop]).size().items():
            rows.append({'representation_id':identifier,'community_id':community,'population':population,'n_patients':int(count),
                'n_common':len(part),'status':'COMPLETE','interpretation':'descriptive_overlap_not_incremental_value'})
        comparisons.append({'representation_id':identifier,'n_common':len(part),
            'ari':adjusted_rand_score(part.community_id,part[pop]) if len(part)>1 else np.nan,
            'nmi':normalized_mutual_info_score(part.community_id,part[pop]) if len(part)>1 else np.nan})
    return pd.DataFrame(rows),pd.DataFrame(comparisons)


def incremental_models(baseline,membership,roles,manifest,config):
    rows=[]
    for identifier in manifest['eligible_representation_ids']:
        members=hard_members(membership,identifier)
        frame=members.merge(baseline,on='patient_id',validate='one_to_one')
        role_map=roles.loc[roles.representation_id.eq(identifier)].set_index('source_column').validation_role.to_dict()
        for anchor in config['anchors']:
            column=anchor['source_column']; row={'representation_id':identifier,'endpoint':column,'status':'NOT_ESTIMABLE',
                'inference_scope':'in_sample_exploratory_conditional_on_frozen_labels','hypothesis_family':f'{identifier}/incremental',
                'community_selection_uncertainty':'not_included_conditional_intervals','p_value':np.nan}
            if role_map.get(column)!='independent_baseline_anchor': row['reason']='ENDPOINT_UNAVAILABLE_OR_CIRCULAR'
            elif manifest['release_status']=='NO_STABLE_STRUCTURE' or not frame.structural_stability_status.eq('stable').all(): row['reason']='TOPOLOGY_NOT_STABLE'
            elif config['pop_column'] not in frame: row['reason']='POP_UNAVAILABLE'
            else:
                data=pd.DataFrame({'y':numeric(frame[column]),'pop':frame[config['pop_column']], 'community':frame.community_id})
                covariates=[]; missing=[]
                for i,c in enumerate(config['confounders']):
                    if c not in frame or frame[c].isna().all(): missing.append(c); continue
                    key=f'confounder_{i}'
                    values=frame[c]
                    num=numeric(values)
                    is_numeric=num.notna().sum()==values.notna().sum()
                    data[key]=num if is_numeric else values.astype('object')
                    covariates.append(key if is_numeric else f'C({key})')
                date=config['date_column']
                if date in frame:
                    data['baseline_era']=pd.to_datetime(frame[date],errors='coerce').dt.year//5*5
                    covariates.append('C(baseline_era)')
                data=data.dropna()
                row.update(n_patients=len(data),n_communities=data.community.nunique(),missing_adjustments=','.join(missing))
                if len(data)<int(config['minimum_model_patients']) or data.community.nunique()<2 or data['pop'].nunique()<2:
                    row['reason']='INSUFFICIENT_PATIENTS_OR_GROUP_VARIATION'
                elif data.groupby('community').size().min()<int(config['minimum_group_patients']): row['reason']='INSUFFICIENT_COMMUNITY_SUPPORT'
                else:
                    try:
                        terms='C(pop)' + ''.join(' + '+c for c in covariates)
                        reduced_spec=smf.ols('y ~ '+terms,data)
                        full_spec=smf.ols('y ~ '+terms+' + C(community)',data)
                        if np.linalg.matrix_rank(full_spec.exog)<full_spec.exog.shape[1]:
                            raise ValueError('RANK_DEFICIENT_MODEL')
                        reduced=reduced_spec.fit(cov_type='HC3')
                        full=full_spec.fit(cov_type='HC3')
                        if np.linalg.matrix_rank(full.model.exog)<full.model.exog.shape[1] or full.df_resid<10:
                            raise ValueError('RANK_DEFICIENT_OR_INSUFFICIENT_RESIDUAL_DF')
                        names=[name for name in full.params.index if name.startswith('C(community)')]
                        wald=full.wald_test(' , '.join(name+' = 0' for name in names),scalar=True)
                        ci=full.conf_int()
                        row.update(status='COMPLETE',reason='',r2_pop=float(reduced.rsquared),r2_pop_topology=float(full.rsquared),
                            delta_r2=float(full.rsquared-reduced.rsquared),p_value=float(wald.pvalue),
                            n_parameters=len(full.params),covariance='HC3',adjustments=terms,
                            conditional_effects={name:{'estimate':float(full.params[name]),'ci_low':float(ci.loc[name,0]),'ci_high':float(ci.loc[name,1])} for name in names})
                    except (ValueError,np.linalg.LinAlgError) as exc: row['reason']=str(exc).split(':')[0]
            rows.append(row)
    result=pd.DataFrame(rows); result['q_value']=np.nan
    for _,idx in result.groupby('hypothesis_family').groups.items(): result.loc[idx,'q_value']=bh_qvalues(result.loc[idx,'p_value'])
    return result

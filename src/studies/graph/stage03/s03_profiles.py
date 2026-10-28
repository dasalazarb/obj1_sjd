"""Observed-value baseline summaries and exploratory independent-anchor tests."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
from .s03_statistics import bh_qvalues, cliffs_delta, odds_ratio_2x2
from .s03_release_reader import hard_members


def numeric(series):
    # Unknown/not-tested are missing, never negative or zero.
    values=series.replace(['unknown','not tested','not_tested','not reported','indeterminate'],np.nan)
    binary={'positive':1.,'negative':0.,'pos':1.,'neg':0.,'reactive':1.,'nonreactive':0.,'detected':1.,'not detected':0.}
    tokens=values.dropna().astype(str).str.strip().str.lower()
    if len(tokens) and set(tokens)<=binary.keys():
        values=values.astype('string').str.strip().str.lower().map(binary)
    return pd.to_numeric(values,errors='coerce').replace([np.inf,-np.inf],np.nan)


def effect_interval(x,y,kind,replicates,seed):
    rng=np.random.default_rng(seed); values=[]
    for _ in range(int(replicates)):
        a=rng.choice(x,len(x),replace=True); b=rng.choice(y,len(y),replace=True)
        if kind=='binary': values.append(odds_ratio_2x2(int(a.sum()),len(a)-int(a.sum()),int(b.sum()),len(b)-int(b.sum())))
        else: values.append(cliffs_delta(a,b))
    return tuple(np.quantile(values,[.025,.975])) if values else (np.nan,np.nan)


def profile_tables(baseline,membership,roles,manifest,validation,seed):
    baseline_rows=[]; signatures=[]; anchor_rows=[]; missing=[]; flow=[]
    for identifier in manifest['eligible_representation_ids']:
        members=hard_members(membership,identifier)
        frame=members.merge(baseline,on='patient_id',how='left',validate='one_to_one')
        part_roles=roles.loc[roles.representation_id.eq(identifier)]
        populations=membership.loc[membership.representation_id.eq(identifier)].drop_duplicates('patient_id')
        flow.append({'representation_id':identifier,'n_baseline':len(baseline),'n_mapper_covered':int(populations.mapper_covered.sum()),
            'n_ties':int(populations.membership_tie.sum()),'n_hard_supported':len(members),'n_not_in_analysis':len(baseline)-len(members),
            'analysis_population':'covered_untied_supported_hard_assignment'})
        if members.empty:
            for source in validation['anchors']:
                declared=part_roles.loc[part_roles.source_column.eq(source['source_column']),'validation_role']
                if len(declared) and declared.iloc[0] in {'defining','derived_from_defining'}: continue
                anchor_rows.append({'representation_id':identifier,'source_column':source['source_column'],
                    'validation_role':'independent_baseline_anchor','status':'NOT_ESTIMABLE','reason':'NO_SUPPORTED_HARD_ASSIGNMENTS'})
            continue
        for role in part_roles.itertuples():
            column=role.source_column
            if role.validation_role=='unavailable' or column not in frame: continue
            series=numeric(frame[column]); numeric_kind=series.notna().sum()==frame[column].notna().sum() and series.notna().any()
            for community,group in frame.groupby('community_id',sort=True):
                observed=series.loc[group.index].dropna() if numeric_kind else group[column].dropna()
                baseline_rows.append({'representation_id':identifier,'community_id':community,'source_column':column,
                    'validation_role':role.validation_role,'n_patients':len(group),'n_observed':len(observed),
                    'n_missing':len(group)-len(observed),'unit':role.unit,'data_type':'numeric' if numeric_kind else 'categorical',
                    'median':float(observed.median()) if numeric_kind and len(observed) else np.nan,
                    'q25':float(observed.quantile(.25)) if numeric_kind and len(observed) else np.nan,
                    'q75':float(observed.quantile(.75)) if numeric_kind and len(observed) else np.nan,
                    'category_counts':observed.value_counts().to_json() if not numeric_kind else None})
                missing.append({'representation_id':identifier,'community_id':community,'source_column':column,
                    'n_patients':len(group),'n_missing':len(group)-len(observed),'missing_fraction':1-len(observed)/len(group)})
                if role.validation_role not in {'defining','derived_from_defining','independent_baseline_anchor'}: continue
                row={'representation_id':identifier,'community_id':community,'source_column':column,
                    'validation_role':role.validation_role,'unit':role.unit,'analysis_population':'covered_untied_supported_hard_assignment',
                    'status':'DESCRIPTIVE','reason':'discovery_features_are_not_independent_validation',
                    'n_group':len(observed),'n_rest':int(series.loc[frame.community_id.ne(community)].notna().sum()),
                    'effect':np.nan,'ci_low':np.nan,'ci_high':np.nan,'p_value':np.nan,'q_value':np.nan}
                if role.validation_role in {'defining','derived_from_defining'} and numeric_kind:
                    x=series.loc[group.index].dropna().to_numpy(float)
                    y=series.loc[frame.community_id.ne(community)].dropna().to_numpy(float)
                    if len(x) and len(y):
                        row.update(effect=cliffs_delta(x,y),effect_type='cliffs_delta',direction='higher' if cliffs_delta(x,y)>0 else 'lower')
                if role.validation_role=='independent_baseline_anchor':
                    row.update(status='COMPLETE',reason='',hypothesis_family=f'{identifier}/independent_baseline_anchors')
                    x=series.loc[group.index].dropna().to_numpy(float)
                    y=series.loc[frame.community_id.ne(community)].dropna().to_numpy(float)
                    if manifest['release_status']=='NO_STABLE_STRUCTURE' or not group.structural_stability_status.eq('stable').all():
                        row.update(status='NOT_ESTIMABLE',reason='TOPOLOGY_NOT_STABLE_FOR_ANCHOR_INFERENCE')
                    elif not numeric_kind or min(len(x),len(y))<int(validation['minimum_group_patients']):
                        row.update(status='NOT_ESTIMABLE',reason='INSUFFICIENT_OBSERVED_GROUP_OR_REST')
                    else:
                        binary=set(np.r_[x,y])<={0.,1.}
                        kind='binary' if binary else 'continuous'
                        effect=odds_ratio_2x2(int(x.sum()),len(x)-int(x.sum()),int(y.sum()),len(y)-int(y.sum())) if binary else cliffs_delta(x,y)
                        p=fisher_exact([[int(x.sum()),len(x)-int(x.sum())],[int(y.sum()),len(y)-int(y.sum())]])[1] if binary else mannwhitneyu(x,y,alternative='two-sided').pvalue
                        lo,hi=effect_interval(x,y,kind,validation['effect_ci_replicates'],seed)
                        row.update(effect=effect,effect_type='odds_ratio' if binary else 'cliffs_delta',ci_low=lo,ci_high=hi,p_value=float(p),
                            inference_scope='in_sample_exploratory_conditional_on_frozen_labels',community_selection_uncertainty='not_included_conditional_intervals')
                    anchor_rows.append(row.copy())
                signatures.append(row)
        # Unavailable anchors are explicit rows, rather than disappearing.
        for role in part_roles.loc[part_roles.validation_role.eq('unavailable')].itertuples():
            if role.source_column in {x['source_column'] for x in validation['anchors']}:
                anchor_rows.append({'representation_id':identifier,'source_column':role.source_column,'validation_role':'independent_baseline_anchor',
                                   'status':'NOT_ESTIMABLE','reason':role.reason,'p_value':np.nan,'q_value':np.nan})
    anchors=pd.DataFrame(anchor_rows)
    if not anchors.empty:
        if 'hypothesis_family' not in anchors: anchors['hypothesis_family']='unavailable'
        if 'p_value' not in anchors: anchors['p_value']=np.nan
        anchors['q_value']=np.nan
        for _,idx in anchors.groupby('hypothesis_family').groups.items():
            anchors.loc[idx,'q_value']=bh_qvalues(anchors.loc[idx,'p_value'])
        anchors['passes_fdr']=anchors.q_value.le(float(validation['fdr_alpha']))
        if 'effect_type' not in anchors: anchors['effect_type']='not_estimable'
        if 'effect' not in anchors: anchors['effect']=np.nan
        anchors['statistically_characteristic']=anchors.passes_fdr & (
            (anchors.effect_type.eq('cliffs_delta') & anchors.effect.abs().ge(validation.get('minimum_abs_cliffs_delta',.147))) |
            (anchors.effect_type.eq('odds_ratio') & ((anchors.effect>=validation.get('minimum_odds_ratio',1.5)) | (anchors.effect<=1/validation.get('minimum_odds_ratio',1.5)))))
    signatures=pd.DataFrame(signatures)
    if not signatures.empty and not anchors.empty:
        qlookup={(r.representation_id,str(getattr(r,'community_id','')),r.source_column):r.q_value for r in anchors.itertuples()}
        signatures['q_value']=[qlookup.get((r.representation_id,str(r.community_id),r.source_column),np.nan) for r in signatures.itertuples()]
    return {'baseline':pd.DataFrame(baseline_rows),'signatures':signatures,'anchors':anchors,'missingness':pd.DataFrame(missing),'flow':pd.DataFrame(flow)}

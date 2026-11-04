"""Observed-value baseline summaries and exploratory independent-anchor tests."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
from .s03_statistics import bh_qvalues, cliffs_delta, odds_ratio_2x2
from .s03_release_reader import hard_members


def observed_values(series):
    tokens=series.astype('string').str.strip().str.lower()
    unknown=tokens.isin(['unknown','not tested','not_tested','not reported','indeterminate',
                         'not available','unavailable','missing','na','n/a','unknown_detail','uninterpretable',
                         'not_measured','not measured','not_requested','not requested',
                         'outside_window','source_absent','structural',''])
    return series.mask(unknown)


def numeric(series):
    # Unknown/not-tested are missing, never negative or zero.
    values=observed_values(series)
    binary={'positive':1.,'negative':0.,'pos':1.,'neg':0.,'reactive':1.,'nonreactive':0.,'detected':1.,'not detected':0.}
    values=values.map(lambda value: binary.get(str(value).strip().lower(),value))
    return pd.to_numeric(values,errors='coerce').replace([np.inf,-np.inf],np.nan)


def effect_interval(x,y,kind,replicates,seed):
    rng=np.random.default_rng(seed); values=[]
    for _ in range(int(replicates)):
        a=rng.choice(x,len(x),replace=True); b=rng.choice(y,len(y),replace=True)
        if kind=='binary': values.append(odds_ratio_2x2(int(a.sum()),len(a)-int(a.sum()),int(b.sum()),len(b)-int(b.sum())))
        else: values.append(cliffs_delta(a,b))
    return tuple(np.quantile(values,[.025,.975])) if values else (np.nan,np.nan)


def profile_tables(baseline,membership,roles,manifest,validation,seed):
    from .s03_construct_validation import independence_block
    baseline_rows=[]; signatures=[]; anchor_rows=[]; missing=[]; flow=[]
    for identifier in manifest['eligible_representation_ids']:
        members=hard_members(membership,identifier)
        frame=members.merge(baseline,on='patient_id',how='left',validate='one_to_one')
        part_roles=roles.loc[roles.representation_id.eq(identifier)]
        populations=membership.loc[membership.representation_id.eq(identifier)].drop_duplicates('patient_id')
        flow.append({'representation_id':identifier,'n_baseline':len(baseline),'n_mapper_covered':int(populations.mapper_covered.sum()),
            'n_ties':int(populations.membership_tie.sum()),'n_hard_supported':len(members),'n_not_in_analysis':len(baseline)-len(members),
            'analysis_population':'covered_untied_supported_hard_assignment',
            'n_uncovered':int((~populations.mapper_covered).sum()),'n_supported_hard':len(members),
            'n_stable_hard':int(members.structural_stability_status.eq('stable').sum()),
            'n_provisional_hard':int(members.structural_stability_status.ne('stable').sum()),
            'n_descriptive':len(members),
            'n_inference':len(members) if len(members) and members.structural_stability_status.eq('stable').all() and manifest['release_status']!='NO_STABLE_STRUCTURE' else 0,
            'inference_eligibility':'topology_gate_only_endpoint_support_and_covariates_required',
            'n_unclassifiable':np.nan,
            'n_unclassifiable_definition':'DEPRECATED_historical_missing_community_count_not_Pop',
            'n_missing_community_id':int(membership.loc[membership.representation_id.eq(identifier)].groupby('patient_id').community_id.count().eq(0).sum()),
            'stability_interpretation':'frozen_Graph02_label_not_clinical_bootstrap_robustness',
            **{k:v for k,v in pop_counts(frame,validation['pop_column']).items() if k!='n_denominator'},
            **{'n_all_baseline_'+k[2:]:v for k,v in pop_counts(baseline,validation['pop_column']).items() if k!='n_denominator'},
            'denominator_n_patients':len(baseline)})
        if members.empty:
            for source in validation['anchors']:
                declared=part_roles.loc[part_roles.source_column.eq(source['source_column']),'validation_role']
                if len(declared) and declared.iloc[0] in {'defining','derived_from_defining'}: continue
                anchor_rows.append({'representation_id':identifier,'source_column':source['source_column'],
                    'validation_role':'independent_baseline_anchor','status':'NOT_ESTIMABLE','reason':independence_block(part_roles.loc[part_roles.source_column.eq(source['source_column'])].iloc[0].to_dict()) or 'NO_SUPPORTED_HARD_ASSIGNMENTS'})
            continue
        for role in part_roles.itertuples():
            column=role.source_column
            if role.validation_role=='unavailable' or column not in frame: continue
            clean=observed_values(frame[column]); series=numeric(clean)
            numeric_kind=role.type=='numeric' or (series.notna().sum()==clean.notna().sum() and series.notna().any())
            for community,group in frame.groupby('community_id',sort=True):
                observed=series.loc[group.index].dropna() if numeric_kind else clean.loc[group.index].dropna()
                baseline_rows.append({'representation_id':identifier,'community_id':community,'source_column':column,
                    'validation_role':role.validation_role,'n_patients':len(group),'n_observed':len(observed),
                    'n_missing':len(group)-len(observed),'unit':role.unit,'data_type':'numeric' if numeric_kind else 'categorical',
                    'median':float(observed.median()) if numeric_kind and len(observed) else np.nan,
                    'q25':float(observed.quantile(.25)) if numeric_kind and len(observed) else np.nan,
                    'q75':float(observed.quantile(.75)) if numeric_kind and len(observed) else np.nan,
                    'category_counts':observed.value_counts().to_json() if not numeric_kind else None})
                missing.append({'representation_id':identifier,'community_id':community,'source_column':column,
                    'n_patients':len(group),'n_observed':len(observed),'n_missing':len(group)-len(observed),'missing_fraction':1-len(observed)/len(group)})
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
                    block=independence_block(role._asdict())
                    if block:
                        row.update(status='NOT_ESTIMABLE',reason=block)
                    elif manifest['release_status']=='NO_STABLE_STRUCTURE' or not frame.structural_stability_status.eq('stable').all():
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
                row.update(n_group=len(group),n_observed=len(observed),n_missing=len(group)-len(observed),
                    n_stable_hard=int(group.structural_stability_status.eq('stable').sum()),
                    n_provisional_hard=int(group.structural_stability_status.ne('stable').sum()))
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
    return {'baseline':enrich_roles(pd.DataFrame(baseline_rows),roles),'signatures':enrich_roles(signatures,roles),
            'anchors':enrich_roles(anchors,roles),'missingness':pd.DataFrame(missing),'flow':pd.DataFrame(flow)}



def clinical_domains(baseline,membership,roles,registry,manifest,minimum_cell):
    """Describe frozen, observed clinical sources without adding discovery inputs."""
    rows=[]
    if registry.empty or not {'source_column','clinical_domain'}<=set(registry):
        return pd.DataFrame([{'status':'NOT_ESTIMABLE','reason':'FROZEN_CLINICAL_DOMAIN_MAP_UNAVAILABLE'}])
    metadata=registry.loc[registry.clinical_domain.isin([f'D{i:02d}' for i in range(1,8)])].drop_duplicates('source_column')
    for identifier in manifest['eligible_representation_ids']:
        members=hard_members(membership,identifier)
        data=members.merge(baseline,on='patient_id',validate='one_to_one')
        rolemap=roles.loc[roles.representation_id.eq(identifier)].set_index('source_column')
        for meta in metadata.to_dict('records'):
            source=meta['source_column']
            for community,group in data.groupby('community_id'):
                values=observed_values(group[source]) if source in group else pd.Series(np.nan,index=group.index)
                n=int(values.notna().sum()); absent=len(group)-n
                role=rolemap.loc[source,'validation_role'] if source in rolemap.index else 'unavailable'
                small=len(group)<minimum_cell or 0<n<minimum_cell or 0<absent<minimum_cell
                num=numeric(values)
                is_numeric=n>0 and num.notna().sum()==n
                rows.append(dict(representation_id=identifier,community_id=community,source_column=source,
                    construct=meta.get('derivation_group',meta.get('semantic_feature_id',source)),
                    clinical_domain=meta['clinical_domain'],validation_role=role,
                    discovery_relation='selected_or_dependent' if role in {'defining','derived_from_defining'} else 'external_descriptive_source_not_clinical_approval',
                    n_group=np.nan if small else len(group),n_observed=np.nan if small else n,n_missing=np.nan if small else absent,
                    observed_fraction=np.nan if small else n/len(group),
                    observed_value=float(num.median()) if is_numeric and not small else np.nan,
                    category_counts=values.dropna().value_counts().to_json() if not is_numeric and not small and (values.dropna().value_counts()>=minimum_cell).all() else None,
                    value_summary='median_original_observed_source_units' if is_numeric else 'category_counts',
                    unit=meta.get('unit','source_units'),status='SUPPRESSED_SMALL_CELL' if small else 'DESCRIPTIVE',
                    reason='NO_OBSERVED_SOURCE_VALUES' if n==0 else '',inferential_scope='descriptive_not_independent_validation'))
    return enrich_roles(pd.DataFrame(rows),roles) if rows else pd.DataFrame([{'status':'NOT_ESTIMABLE','reason':'NO_SUPPORTED_COMMUNITIES'}])


def uncovered_profiles(baseline,membership,roles,manifest,minimum_cell):
    """Coverage audit only: the uncovered population is never a community."""
    rows=[]
    for identifier in manifest['eligible_representation_ids']:
        part=membership.loc[membership.representation_id.eq(identifier)].drop_duplicates('patient_id')
        ids=part.loc[~part.mapper_covered,'patient_id']
        group=baseline.loc[baseline.patient_id.isin(ids)]
        for role in roles.loc[roles.representation_id.eq(identifier)].itertuples():
            if role.source_column not in group: continue
            values=observed_values(group[role.source_column]); num=numeric(values)
            n=int(values.notna().sum()); missing=len(group)-n
            small=len(group)<minimum_cell or 0<n<minimum_cell or 0<missing<minimum_cell
            rows.append(dict(representation_id=identifier,analysis_population='uncovered_coverage_audit_not_a_community',
                source_column=role.source_column,n_patients=np.nan if small else len(group),
                n_observed=np.nan if small else n,n_missing=np.nan if small else missing,
                median=float(num.median()) if not small and n and num.notna().sum()==n else np.nan,
                status='SUPPRESSED_SMALL_CELL' if small else 'DESCRIPTIVE'))
    return pd.DataFrame(rows)



def pop_counts(frame,column):
    """Use the published label exactly; missing and invalid are separate states."""
    labels=frame[column] if column in frame else pd.Series(pd.NA,index=frame.index)
    counts=dict(n_denominator=len(frame),n_pop1=int(labels.eq('Pop1').sum()),
        n_pop2=int(labels.eq('Pop2').sum()),n_pop3=int(labels.eq('Pop3').sum()),
        n_unclassified_pop=int(labels.eq('Unclassified').sum()),n_missing_pop=int(labels.isna().sum()),
        n_invalid_pop=int((labels.notna() & ~labels.isin(['Pop1','Pop2','Pop3','Unclassified'])).sum()))
    assert sum(v for k,v in counts.items() if k!='n_denominator')==len(frame)
    counts['n_unclassifiable_pop']=counts['n_unclassified_pop']
    return counts


def pop_denominator_audit(baseline,membership,manifest,config):
    from .s03_episode_audit import validate_unique_patients
    validate_unique_patients(baseline,'BASELINE')
    rows=[]
    for identifier in manifest['eligible_representation_ids']:
        part=membership.loc[membership.representation_id.eq(identifier)]
        # Multiple Mapper memberships are legitimate; patient flags must agree.
        if part.groupby('patient_id').mapper_covered.nunique().gt(1).any():
            raise ValueError('INCONSISTENT_PATIENT_COVERAGE')
        covered=set(part.loc[part.mapper_covered,'patient_id'])
        hard=set(hard_members(membership,identifier).patient_id)
        for universe,frame in [('all_baseline',baseline),('mapper_covered',baseline.loc[baseline.patient_id.isin(covered)]),
                               ('hard_supported',baseline.loc[baseline.patient_id.isin(hard)])]:
            counts=pop_counts(frame,config['pop_column'])
            rows.append(dict(representation_id=identifier,population_universe=universe,**counts,
                status='DATA_QUALITY_INVALID_LABELS' if counts['n_invalid_pop'] else 'DESCRIPTIVE',
                reason='UNKNOWN_PUBLISHED_POP_LABEL' if counts['n_invalid_pop'] else ''))
    return pd.DataFrame(rows)


def clinical_marker_summaries(baseline,membership,roles,manifest,validation):
    """Prespecified observed instrument panels, including non-discovery sources.

    This additional description does not alter historical baseline tables or
    grant independence to a source that was not an approved validation anchor.
    """
    rows=[]
    for identifier in ['R0','R2']:
        if identifier not in manifest['eligible_representation_ids']: continue
        frame=hard_members(membership,identifier).merge(baseline,on='patient_id',validate='one_to_one')
        rolemap=roles.loc[roles.representation_id.eq(identifier)].set_index('source_column').validation_role.to_dict()
        for marker in validation.get('clinical_marker_panels',[]):
            column=marker['source_column']
            for community,group in frame.groupby('community_id'):
                values=numeric(group[column]).dropna() if column in group else pd.Series(dtype=float)
                rows.append(dict(representation_id=identifier,community_id=community,source_column=column,
                    validation_role=rolemap.get(column,'unavailable'),unit=marker['unit'],
                    n_patients=len(group),n_observed=len(values),n_missing=len(group)-len(values),
                    median=values.median(),q25=values.quantile(.25),q75=values.quantile(.75),
                    status=('DESCRIPTIVE_PROVISIONAL' if identifier=='R2' or not group.structural_stability_status.eq('stable').all() else 'DESCRIPTIVE') if len(values) else 'NOT_ESTIMABLE',
                    reason='' if len(values) else 'NO_OBSERVED_NUMERIC_MARKER',
                    inference_scope='observed_baseline_description_not_independent_validation'))
    return enrich_roles(pd.DataFrame(rows),roles)


def enrich_roles(frame,roles):
    """Add the versioned gate without changing historical values or roles."""
    if frame.empty or not {'representation_id','source_column'}<=set(frame): return frame
    keys=['representation_id','source_column']
    columns=[c for c in ['tier','independence_status','evidence_level','inferential_eligible',
                        'blocking_reasons','relationship_type','classification_policy_version'] if c in roles and c not in frame]
    frame=frame.merge(roles[keys+columns],on=keys,how='left',validate='many_to_one')
    if 'n_patients' in frame and 'n_group' not in frame: frame['n_group']=frame.n_patients
    return frame

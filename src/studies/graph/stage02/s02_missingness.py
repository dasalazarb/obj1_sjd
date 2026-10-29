"""Baseline observation process diagnostics; never infer disease or MCAR/MAR/MNAR."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_distances, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from .s02_representations import Space, gower_distance, reduce_matrix
from .s02_resampling import match_sets

ADMIN_ALIASES={'demo__protocol':('ids__protocol','ids__protocol_number','parent_protocol')}


def supported_hard(mapped):
    hard=np.asarray(mapped['hard'],float)
    valid=np.asarray(mapped['covered'],bool) & np.isfinite(hard) & ~np.asarray(mapped['ties'],bool)
    supported=np.asarray(mapped['supported'],bool)
    idx=np.flatnonzero(valid)
    valid[idx]=supported[hard[idx].astype(int)]
    return valid


def compare_coverage(reference, candidate, minimum=10):
    n=len(reference['covered'])
    if len(candidate['covered'])!=n: raise ValueError('PATIENT_UNIVERSE_MISMATCH')
    a=supported_hard(reference); b=supported_hard(candidate); common=a & b
    count=int(common.sum())
    enough=count>=minimum and len(np.unique(reference['hard'][common]))>=2 and len(np.unique(candidate['hard'][common]))>=2
    full,_=match_sets(reference['community_sets'],candidate['community_sets'])
    universe=set(np.flatnonzero(common))
    restricted,_=match_sets([s & universe for s in reference['community_sets']], [s & universe for s in candidate['community_sets']])
    return dict(status='COMPLETE' if enough else 'NOT_ESTIMABLE',reason='' if enough else 'INSUFFICIENT_COMMON_SUPPORTED_ASSIGNMENTS_OR_CLASSES',
        n_total=n,n_covered_reference=int(np.sum(reference['covered'])),n_covered_control=int(np.sum(candidate['covered'])),
        n_covered_both=int(np.sum(reference['covered'] & candidate['covered'])),n_hard_both=count,n_common=count,
        fraction_of_total=count/n if n else np.nan,n_hard_reference=int(a.sum()),n_hard_control=int(b.sum()),
        n_tied=int(np.sum(candidate['ties'])),n_unsupported=int(np.sum(candidate['covered'] & ~candidate['ties'] & ~b)),
        n_uncovered=int(np.sum(~candidate['covered'])),coverage_difference=float(np.mean(candidate['covered'])-np.mean(reference['covered'])),
        ari_primary=float(adjusted_rand_score(reference['hard'][common],candidate['hard'][common])) if enough else np.nan,
        nmi_primary=float(normalized_mutual_info_score(reference['hard'][common],candidate['hard'][common])) if enough else np.nan,
        jaccard_full_mean=float(np.mean([r['jaccard'] for r in full])) if full else np.nan,
        jaccard_common_mean=float(np.mean([r['jaccard'] for r in restricted])) if enough and restricted else np.nan,
        community_sizes_reference='|'.join(str(int(np.sum(reference['hard'][common]==c))) for c in np.unique(reference['hard'][common])),
        community_sizes_control='|'.join(str(int(np.sum(candidate['hard'][common]==c))) for c in np.unique(candidate['hard'][common])))


def build_observation_mask(raw, registry, baseline=None):
    """One binary observation unit per documented construct, with encoded lineage.

    Any measured primary source observes a construct; partial bilateral/instrument
    measurement is retained as a state. SF36 subscales remain distinct sources;
    PCS/MCS are omitted when constituent subscales are selected.
    """
    if not raw.index.is_unique or not raw.columns.is_unique: raise ValueError('AMBIGUOUS_OBSERVATION_INPUT')
    meta=registry.loc[registry.feature.isin(raw)].copy()
    if len(meta)!=raw.shape[1]: raise ValueError('INCOMPLETE_OBSERVATION_SOURCE_MAP')
    masks={}; rows=[]; states={}
    selected=set(meta.source_column)
    has_sf36_subscales=any(s.startswith('pro__sf36_') and not s.endswith(('_pcs','_mcs')) for s in selected)
    for source,part in meta.groupby('source_column',sort=True):
        row=part.iloc[0]; group=row.get('derivation_group',source)
        group=source if pd.isna(group) else group
        derived_sf36=has_sf36_subscales and source in {'pro__sf36_pcs','pro__sf36_mcs'}
        administrative=row.clinical_domain=='D09' or row.get('role')=='administrative_confounder'
        excluded=derived_sf36 or administrative
        if group=='SF36_INSTRUMENT': group=source if has_sf36_subscales else 'SF36_SUMMARIES'
        key='observed__'+str(group)
        observed=raw[part.feature].notna().any(axis=1).astype(float)
        if not excluded: masks.setdefault(key,[]).append(observed)
        state_column=next((c for c in (source+'__observation_status',source+'_status',source+'_evaluable') if baseline is not None and c in baseline),None)
        state=pd.Series(np.where(observed.eq(1),'measured','unknown_detail'),index=raw.index)
        if baseline is not None and source in baseline:
            tokens=baseline[source].astype('string').str.lower().str.strip()
            explicit=tokens.isin(['not_measured','not_tested','not tested','partial','invalid','unknown','unknown_detail','indeterminate','uninterpretable'])
            state.loc[explicit]=tokens.loc[explicit]
        if state_column:
            supplied=baseline[state_column].astype('string')
            state.loc[supplied.notna()]=supplied.loc[supplied.notna()]
        states[source]=state
        rows.append(dict(source_column=source,observation_unit=key,encoded_columns='|'.join(part.feature),
            n_encoded_columns=len(part),family=row.technical_family,clinical_domain=row.clinical_domain,
            instrument_id=row.get('instrument_id',row.subdomain),derivation_group=group,
            included_in_mask=not excluded,exclusion_reason='ADMINISTRATIVE_CONTEXT_EXCLUDED_FROM_CLINICAL_MASK' if administrative else 'DERIVED_SF36_SUMMARY_CONSTITUENTS_SELECTED' if derived_sf36 else '',
            observation_rule='any_primary_source_observed;partial_retained_separately',
            state_source=state_column or 'SOURCE_UNAVAILABLE',
            explicit_states='|'.join(sorted(state.dropna().astype(str).unique())),
            masks_source='baseline_R0_raw_selected_features'))
    mask=pd.DataFrame({key:pd.concat(series,axis=1).max(axis=1) for key,series in masks.items()},index=raw.index)
    manifest=pd.DataFrame(rows)
    mask.attrs['partial_states']={key:pd.concat(series,axis=1).mean(axis=1).to_numpy() for key,series in masks.items()}
    mask.attrs['source_states']=states
    return mask,manifest


def observation_features(mask, source_manifest):
    features=pd.DataFrame({'observed_fraction':mask.mean(axis=1)},index=mask.index)
    meta=source_manifest.loc[source_manifest.included_in_mask]
    for field in ['family','clinical_domain','instrument_id']:
        for key,part in meta.groupby(field):
            cols=list(dict.fromkeys(part.observation_unit))
            features[f'{field}__{key}']=mask[cols].mean(axis=1)
    domain_cols=[c for c in features if c.startswith('clinical_domain__')]
    features['n_blocks_complete']=features[domain_cols].eq(1).sum(axis=1)
    features['n_blocks_zero']=features[domain_cols].eq(0).sum(axis=1)
    return features


def summarize_missingness(mask, source_manifest):
    rows=[]; meta=source_manifest.loc[source_manifest.included_in_mask]
    groups=[('global','R0_fixed',list(mask))]
    for field in ['family','clinical_domain','instrument_id']:
        groups += [(field,str(key),list(dict.fromkeys(part.observation_unit))) for key,part in meta.groupby(field)]
    for field,key,cols in groups:
        block=mask[cols]; missing=int(block.size-block.to_numpy().sum())
        rows.append(dict(panel='R0_fixed',group_type=field,group=key,n_patients=len(mask),n_observation_units=len(cols),
            n_cells=block.size,n_missing=missing,missing_fraction=missing/block.size if block.size else np.nan,
            n_any_observed=int(block.gt(0).any(axis=1).sum()),unit='original_source_or_documented_construct',status='COMPLETE'))
    return pd.DataFrame(rows)


def administrative_frame(baseline, columns, date_column):
    """Exact verified structural aliases only; protocol is never inferred from era."""
    frame=pd.DataFrame(index=baseline.index); audit=[]; kinds={}
    for requested in columns:
        source=requested if requested in baseline else next((c for c in ADMIN_ALIASES.get(requested,()) if c in baseline),None)
        if source is None:
            audit.append(dict(requested=requested,source='SOURCE_UNAVAILABLE',status='NOT_ESTIMABLE',reason='SOURCE_UNAVAILABLE')); continue
        value=baseline[source]
        if requested in {'demo__age_at_baseline','age_dx'}:
            frame[requested]=pd.to_numeric(value,errors='coerce'); kinds[requested]='numeric'
        else:
            frame[requested]=value.astype('string'); kinds[requested]='nominal'
        audit.append(dict(requested=requested,source=source,status='COMPLETE',reason='',n_observed=int(value.notna().sum())))
    if date_column in baseline:
        date=pd.to_datetime(baseline[date_column],errors='coerce')
        frame['graph__era']=(date.dt.year//5*5).astype('Int64').astype('string'); kinds['graph__era']='nominal'
        audit.append(dict(requested='graph__era',source=date_column,status='COMPLETE',reason='prespecified_5_year_calendar_bands'))
    frame=frame.loc[:,frame.nunique(dropna=True)>1]
    return frame,{c:kinds[c] for c in frame},pd.DataFrame(audit)


def administrative_strata(series, column, minimum):
    if column=='demo__age_at_baseline':
        # Fixed decade bins, never one stratum per exact continuous value.
        numeric=pd.to_numeric(series,errors='coerce')
        return pd.cut(numeric,[-np.inf,30,40,50,60,70,np.inf],right=False).astype('string')
    result=series.astype('string')
    counts=result.value_counts(); rare=counts[counts<minimum].index
    return result.mask(result.isin(rare),'Other_rare_prespecified')


def _lens(distance):
    n=len(distance); center=np.eye(n)-np.ones((n,n))/n
    gram=-.5*center @ distance**2 @ center
    values,vectors=np.linalg.eigh(gram); order=np.argsort(values)[::-1]
    positive=order[values[order]>np.finfo(float).eps*max(1,np.linalg.norm(gram))]
    if not len(positive): raise ValueError('NO_OBSERVATION_VARIABILITY')
    lens=vectors[:,positive[:2]]*np.sqrt(values[positive[:2]])
    if lens.shape[1]==1: lens=np.column_stack([lens,np.zeros(n)])
    return lens


def run_missingness_only_mapper(mask, source_manifest, mode='N2a', admin=None, kinds=None, config=None, value_space=None):
    """Build a refittable space; Mapper runs through the common engine.

    Binary mask distance is family-balanced Hamming. All-one/all-zero and
    duplicate masks remain patients, with zero mutual distance, never jittered.
    """
    active=mask.loc[:,mask.nunique()>1]
    meta=source_manifest.loc[source_manifest.included_in_mask].drop_duplicates('observation_unit')
    audits=[]; blocks=[]
    if mode in {'N2a','N2c','N2d','M1'}:
        if active.empty: raise ValueError('NO_OBSERVATION_VARIABILITY')
        if mode=='N2d':
            features=observation_features(mask,source_manifest)
            active=features[[c for c in features if c.startswith(('clinical_domain__','instrument_id__'))]]
            active=active.loc[:,active.nunique()>1]
            if active.empty: raise ValueError('NO_DOMAIN_OBSERVATION_VARIABILITY')
            weights=np.ones(active.shape[1])/active.shape[1]
            # Equal domain and instrument contributions, each balanced internally.
            prefixes=[prefix for prefix in ('clinical_domain__','instrument_id__') if any(c.startswith(prefix) for c in active)]
            for prefix in prefixes:
                idx=[i for i,c in enumerate(active) if c.startswith(prefix)]
                for i in idx: weights[i]=1/(len(prefixes)*len(idx))
        else:
            lookup=meta.set_index('observation_unit').family.to_dict()
            counts=pd.Series([lookup[c] for c in active]).value_counts()
            weights=np.array([1/(len(counts)*counts[lookup[c]]) for c in active])
        d=pairwise_distances(active.to_numpy()*weights,metric='manhattan')
        blocks.append(d)
        audits.append(dict(block='mask',weight_applied=.5 if mode in {'N2c','M1'} else 1.,metric='family_balanced_hamming' if mode!='N2d' else 'balanced_domain_instrument_L1',
            n_units=active.shape[1],n_duplicate_masks=int(mask.duplicated().sum()),n_all_observed=int(mask.eq(1).all(axis=1).sum()),n_all_missing=int(mask.eq(0).all(axis=1).sum())))
    if mode in {'N2b','N2c'}:
        if admin is None or admin.empty: raise ValueError('NO_VARIABLE_VERIFIED_ADMINISTRATIVE_SOURCES')
        blocks.append(gower_distance(admin,kinds)); audits.append(dict(block='administration',weight_applied=.5 if mode=='N2c' else 1.,metric='Gower_one_original_source'))
    if mode=='M1':
        if value_space is None: raise ValueError('VALUE_SPACE_UNAVAILABLE')
        # Euclidean concatenation with normalized equal energy blocks; all fitted within draw.
        clinical=value_space.cluster
        norm=np.linalg.norm(clinical)
        if norm==0: raise ValueError('NO_VALUE_VARIABILITY')
        clinical=clinical/norm
        x=active.to_numpy()*np.sqrt(weights); x=x-x.mean(axis=0)
        norm=np.linalg.norm(x)
        if norm==0: raise ValueError('NO_OBSERVATION_VARIABILITY')
        x=x/norm
        embedded=reduce_matrix(np.column_stack([clinical,x])/np.sqrt(2),config)
        return Space(value_space.raw,embedded[:,:2],embedded,pairwise_distances(embedded),pd.DataFrame(audits),mode,list(value_space.raw))
    distance=sum(blocks)/len(blocks)
    return Space(mask if mode!='N2b' else admin,_lens(distance),distance,distance,pd.DataFrame(audits),mode,list(mask if mode!='N2b' else admin))


def fit_adjusted_observation_model(features, covered, seed, repeats=5, minimum_class=5):
    y=np.asarray(covered,int); counts=np.bincount(y,minlength=2); n=len(y)
    common=dict(n_patients=n,n_covered=int(counts[1]),n_uncovered=int(counts[0]),majority_accuracy=float(counts.max()/n),baseline_auc=.5,
        target='Mapper_covered_algorithmic_property',preprocessing='imputer_scaler_fitted_inside_each_training_fold')
    x=features.loc[:,features.nunique()>1]
    if counts.min()<minimum_class or x.empty:
        return [{**common,'status':'NOT_ESTIMABLE','reason':'INSUFFICIENT_CLASS_SUPPORT_OR_PREDICTORS'}]
    cv=RepeatedStratifiedKFold(n_splits=min(5,int(counts.min())),n_repeats=repeats,random_state=seed)
    rows=[]; scores=[]
    for fold,(train,test) in enumerate(cv.split(x,y)):
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=seed))
        model.fit(x.iloc[train],y[train]); score=roc_auc_score(y[test],model.predict_proba(x.iloc[test])[:,1]); scores.append(score)
        rows.append({**common,'status':'COMPLETE','reason':'','fold':fold,'n_train':len(train),'n_test':len(test),'auc':score})
    for row in rows: row.update(auc_mean=float(np.mean(scores)),auc_p025=float(np.quantile(scores,.025)),auc_p975=float(np.quantile(scores,.975)),interval_type='repeated_CV_distribution_not_independent_confidence_interval')
    return rows


def _association(x, labels):
    labels=np.asarray(labels); x=np.asarray(x,float)
    total=np.sum((x-x.mean(axis=0))**2)
    if total==0: return np.nan
    return sum(np.sum(labels==c)*np.sum((x[labels==c].mean(axis=0)-x.mean(axis=0))**2) for c in np.unique(labels))/total


def mask_permutation_test(mask, labels, seed, replicates=199, strata=None, minimum=5):
    labels=pd.Series(labels,index=mask.index); valid=labels.notna()
    x=mask.loc[valid].to_numpy(); y=labels.loc[valid].to_numpy(); counts=pd.Series(y).value_counts()
    common=dict(n_patients=len(x),n_permutations=replicates,statistic='multivariate_observation_eta_squared',
        preserves='whole_mask_rows_and_joint_within_patient_pattern',breaks='mask_to_label_patient_alignment',universe='nonmissing_eligible_labels')
    if len(counts)<2 or counts.min()<minimum or not np.isfinite(_association(x,y)):
        return [{**common,'status':'NOT_ESTIMABLE','reason':'INSUFFICIENT_LABEL_SUPPORT_OR_MASK_VARIANCE'}]
    groups=None
    if strata is not None:
        z=pd.Series(strata,index=mask.index).loc[valid]
        if z.isna().any() or z.value_counts().min()<minimum:
            return [{**common,'status':'NOT_ESTIMABLE','reason':'INSUFFICIENT_ADMINISTRATIVE_STRATUM_SUPPORT'}]
        groups=[np.flatnonzero(z.to_numpy()==c) for c in z.unique()]
    rng=np.random.default_rng(seed); observed=_association(x,y); null=[]
    for _ in range(replicates):
        order=rng.permutation(len(x)) if groups is None else np.arange(len(x))
        if groups is not None:
            for idx in groups: order[idx]=rng.permutation(idx)
        null.append(_association(x[order],y))
    p=(1+np.count_nonzero(np.array(null)>=observed))/(replicates+1)
    return [{**common,'status':'COMPLETE','reason':'','replicate':i,'null_statistic':v,'observed_statistic':observed,
        'p_exploratory':p,'null_p025':float(np.quantile(null,.025)),'null_p975':float(np.quantile(null,.975)),
        'stratified':strata is not None} for i,v in enumerate(null)]


def grouped_observation(mask, labels, minimum=5):
    rows=[]; labels=pd.Series(labels,index=mask.index).astype('string').fillna('Unknown')
    for key,idx in labels.groupby(labels).groups.items():
        n=len(idx); block=mask.loc[idx]
        rows.append(dict(group=str(key),n_patients=n,status='COMPLETE' if n>=minimum else 'NOT_ESTIMABLE',
            reason='' if n>=minimum else 'SMALL_PUBLICATION_CELL',observed_fraction=float(block.to_numpy().mean()) if n>=minimum else np.nan))
    return pd.DataFrame(rows)


def run_observation_diagnostics(raw, registry, baseline, reference, config, settings, validation,
                                run_dir, perturbation, record, fit_values):
    """Default in-process execution; no extra CLI commands or outcome selection."""
    from .._pipeline import save_table
    from .s02_mapper_engine import analyze
    from .s02_coverage_diagnostics import diagnose_uncovered, compare_groups
    mask,source=build_observation_mask(raw,registry,baseline)
    features=observation_features(mask,source)
    admin,kinds,admin_audit=administrative_frame(baseline,validation['confounders'],validation['date_column'])
    seed=int(config['random_seed']); minimum=int(settings.get('minimum_publication_cell',5))
    permutations=int(settings.get('observation_permutations',199)); controls={}; comparisons=[]
    save_table(run_dir/'02_graph_missingness_source_manifest.csv',source)
    save_table(run_dir/'02_graph_missingness_administrative_sources.csv',admin_audit)
    patient_masks=mask.copy(); patient_masks.insert(0,'patient_id',baseline.patient_id.to_numpy())
    save_table(run_dir/'02_graph_patient_observation_masks.csv',patient_masks)
    patient_features=features.copy(); patient_features.insert(0,'patient_id',baseline.patient_id.to_numpy())
    for name,values in mask.attrs.get('partial_states',{}).items(): patient_features['partial__'+name]=values
    for name,values in mask.attrs.get('source_states',{}).items(): patient_features['source_state__'+name]=values.to_numpy()
    # Explicit source observation states stay internal, including unknown_detail.
    for name in source.state_source.unique():
        if name in baseline: patient_features['state__'+name]=baseline[name].to_numpy()
    save_table(run_dir/'02_graph_patient_missingness_features.csv',patient_features)
    summary=summarize_missingness(mask,source)
    save_table(run_dir/'02_graph_missingness_summary.csv',summary.query("group_type=='global'"))
    save_table(run_dir/'02_graph_missingness_by_family_domain.csv',summary.query("group_type!='global'"))
    labels=pd.Series('uncovered',index=mask.index)
    valid=supported_hard(reference)
    labels.loc[reference['covered']]='covered_no_supported_hard_assignment'
    labels.loc[valid]=['C'+str(int(h)) for h in reference['hard'][valid]]
    save_table(run_dir/'02_graph_missingness_by_community.csv',grouped_observation(mask,labels,minimum))
    save_table(run_dir/'02_graph_missingness_by_coverage.csv',grouped_observation(mask,pd.Series(reference['covered'],index=mask.index).map({True:'covered',False:'uncovered'}),minimum))
    pop=next((c for c in ['pop__status','pop_status'] if c in baseline),None)
    save_table(run_dir/'02_graph_missingness_by_pop.csv',grouped_observation(mask,baseline[pop],minimum) if pop else [dict(status='NOT_ESTIMABLE',reason='SOURCE_UNAVAILABLE')])
    patterns=mask.astype(int).astype(str).agg(''.join,axis=1).value_counts()
    save_table(run_dir/'02_graph_missingness_pattern_summary.csv',[
        dict(n_patients=int(count),n_patterns=int((patterns==count).sum()),pattern_identity='suppressed_patient_privacy',status='COMPLETE' if count>=minimum else 'NOT_ESTIMABLE',reason='' if count>=minimum else 'SMALL_PUBLICATION_CELL') for count in sorted(patterns.unique())])

    def fit_control(mode,draw,panel=None):
        subset=raw.iloc[draw].reset_index(drop=True); base=baseline.iloc[draw].reset_index(drop=True)
        value_space=fit_values(draw) if mode=='M1' else None
        if mode=='M1': subset=value_space.raw
        m,s=build_observation_mask(subset,registry,base)
        if panel is not None:
            s=s.loc[s.clinical_domain.isin(panel)]
            cols=list(dict.fromkeys(s.loc[s.included_in_mask,'observation_unit']))
            m=m[cols]
            if m.empty: raise ValueError('PANEL_HAS_NO_OBSERVATION_UNITS')
        a,k,_=administrative_frame(base,validation['confounders'],validation['date_column'])
        return run_missingness_only_mapper(m,s,mode,a,k,config,value_space)

    for mode,name in [('N2a','missingness_only'),('N2b','administrative_only'),('N2c','missingness_plus_admin'),('N2d','domain_missingness_only'),('M1','value_plus_mask_sensitivity')]:
        identifier=mode+'/'+name
        try:
            space=fit_control(mode,np.arange(len(raw))); mapped=analyze(space,config)
            row=perturbation(identifier,space,mapped,kind='observation_control',fit=lambda draw,m=mode:fit_control(m,draw))
            controls[mode]={'space':space,'mapped':mapped}
            comparisons.append({**row,'control_type':name,'universe':'R0_fixed_baseline','n_observation_units':len(mask.columns),
                'block_weights':space.weights.to_json(orient='records'),'mask_panel':'R0_fixed'})
        except (ValueError,np.linalg.LinAlgError) as exc:
            comparisons.append(record(identifier,'observation_control','NOT_ESTIMABLE',str(exc)))
    # Same fixed baseline universe, with explicit alternative panels; these do
    # not replace the principal R0-selected clinical mask or choose a graph.
    clinical_domains={f'D{i:02d}' for i in range(1,8)}
    panels=[('clinical_core',clinical_domains)]+[(domain,{domain}) for domain in sorted(set(source.clinical_domain)&clinical_domains)]
    panel_rows=[]
    for name,domains in panels:
        identifier='N2a/panel_'+name
        try:
            space=fit_control('N2a',np.arange(len(raw)),domains)
            panel_rows.append(perturbation(identifier,space,kind='observation_panel_sensitivity'))
        except (ValueError,np.linalg.LinAlgError) as exc:
            panel_rows.append(record(identifier,'observation_panel_sensitivity','NOT_ESTIMABLE',str(exc)))
    save_table(run_dir/'02_graph_missingness_panel_sensitivity.csv',panel_rows)
    save_table(run_dir/'02_graph_missingness_control_comparisons.csv',[r for r in comparisons if not r['experiment_id'].startswith('M1')])
    save_table(run_dir/'02_graph_missingness_sensitivity_results.csv',[r for r in comparisons if r['experiment_id'].startswith('M1')])
    diagnostic=[]
    for name,x in [('family_domain_loads',features[[c for c in features if c.startswith(('family__','clinical_domain__'))]]),
                   ('administration_only',pd.get_dummies(admin,dtype=float,dummy_na=True))]:
        diagnostic.extend(dict(model=name,**r) for r in fit_adjusted_observation_model(x,reference['covered'],seed,int(settings.get('observation_cv_repeats',5)),minimum))
    save_table(run_dir/'02_graph_missingness_coverage_diagnostic.csv',diagnostic)
    tests=[]
    targets=[('Mapper_covered',pd.Series(reference['covered'],index=mask.index)),('R0_supported_community',pd.Series(reference['hard'],index=mask.index).where(valid))]
    if pop: targets.append(('Pop_external_annotation',baseline[pop]))
    targets += [('administrative_annotation/'+c,administrative_strata(admin[c],c,int(settings['minimum_stratum_patients']))) for c in admin]
    for target,y in targets:
        tests.extend(dict(experiment_id='N5/mask_row_permutation',target=target,**r) for r in mask_permutation_test(mask,y,seed,permutations,minimum=minimum))
    strata=next((admin[c] for c in ['demo__protocol','graph__era'] if c in admin and admin[c].value_counts().min()>=int(settings['minimum_stratum_patients'])),None)
    if strata is None: tests.append(dict(experiment_id='N6/administrative_stratified_permutation',status='NOT_ESTIMABLE',reason='NO_SUFFICIENT_VERIFIED_PROTOCOL_OR_ERA_STRATA',n_patients=len(mask)))
    else:
        tests.extend(dict(experiment_id='N6/administrative_stratified_permutation',target='R0_supported_community',**r) for r in mask_permutation_test(mask,targets[1][1],seed,permutations,strata,minimum))
    for identifier in ['N5/mask_row_permutation','N6/administrative_stratified_permutation']:
        part=[r for r in tests if r['experiment_id']==identifier]
        complete=any(r['status']=='COMPLETE' for r in part)
        record(identifier,'observation_permutation','COMPLETE' if complete else 'NOT_ESTIMABLE','' if complete else '|'.join(sorted({r['reason'] for r in part})),n_permutations=permutations)
    save_table(run_dir/'02_graph_missingness_test_registry.csv',tests)
    coverage,reason_summary=diagnose_uncovered(fit_values(np.arange(len(raw))),reference,baseline.patient_id,features)
    # Administration and Pop are descriptive, retained internally; aggregate small cells below.
    for c in admin: coverage[c]=admin[c].to_numpy()
    if pop: coverage['external_pop_annotation']=baseline[pop].to_numpy()
    save_table(run_dir/'02_graph_patient_coverage_diagnostics.csv',coverage)
    save_table(run_dir/'02_graph_coverage_reason_summary.csv',reason_summary)
    group_comparison=compare_groups(coverage,seed,permutations,minimum)
    save_table(run_dir/'02_graph_coverage_group_comparison.csv',group_comparison)
    pro=[c for c in features if c=='family__PRO']
    save_table(run_dir/'02_graph_missingness_PRO_ablation_diagnostic.csv',[
        dict(status='COMPLETE' if pro else 'NOT_ESTIMABLE',reason='' if pro else 'PRO_SOURCE_UNAVAILABLE',
             n_patients=len(mask),observed_fraction=float(features[pro].to_numpy().mean()) if pro else np.nan,
             interpretation='Compare descriptively with A2/PRO; block removal and observation association do not identify causality')])
    return dict(mask=mask,source=source,features=features,controls=controls,comparisons=comparisons,
        summary=summary,coverage=coverage,reason_summary=reason_summary,group_comparison=group_comparison,
        test_registry=pd.DataFrame(tests),diagnostic=pd.DataFrame(diagnostic),community_labels=labels,
        reason_audit_complete=not coverage.coverage_reason.eq('reason_unresolved').any())

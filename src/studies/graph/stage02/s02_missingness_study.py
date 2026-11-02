"""Run prespecified missingness sensitivities without changing defining releases."""
from pathlib import Path
import hashlib
import json
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .._pipeline import save_table, digest
from .._feature_catalog import frame_hash
from .s02_missingness_registry import NotEstimable, build_panel, observation_summary
from .s02_missingness_variants import build_variant
from .s02_pairwise_distance import observed_distance
from .s02_missingness import compare_coverage
from .s02_mapper_engine import analyze, frozen_membership
from .s02_resampling import resample


def geometry_comparison(a,b,k=10):
    if a.shape!=b.shape: raise ValueError('GEOMETRY_PATIENT_UNIVERSE_MISMATCH')
    upper=np.triu_indices(len(a),1); common=np.isfinite(a[upper])&np.isfinite(b[upper])
    av=a[upper][common]; bv=b[upper][common]
    correlation=float(spearmanr(av,bv).statistic) if len(av)>=3 and np.ptp(av)>0 and np.ptp(bv)>0 else np.nan
    scores=[]
    for i in range(len(a)):
        valid=np.isfinite(a[i])&np.isfinite(b[i]); valid[i]=False; ids=np.flatnonzero(valid)
        count=min(k,len(ids))
        if count<2: continue
        left=set(ids[np.argsort(a[i,ids],kind='stable')[:count]])
        right=set(ids[np.argsort(b[i,ids],kind='stable')[:count]])
        scores.append(len(left&right)/count)
    return dict(n_valid_distance_pairs=int(common.sum()),n_total_distance_pairs=len(common),
        distance_spearman=correlation,neighbor_preservation=float(np.mean(scores)) if scores else np.nan,
        n_neighbor_evaluable_patients=len(scores),neighbor_k=k,
        geometry_status='COMPLETE' if np.isfinite(correlation) else 'NOT_ESTIMABLE',
        geometry_reason='' if np.isfinite(correlation) else 'INSUFFICIENT_NONCONSTANT_COMMON_DISTANCE_PAIRS')


def cache_signature(raw,registry,baseline,identifier,policy,config,config_hash,source_hash):
    """Estimator/code/version/panel/state provenance all participate in cache keys."""
    code=''.join(digest(p) for p in sorted(Path(__file__).parent.glob('s02_*.py')))
    meta=registry.loc[registry.feature.isin(raw)].fillna('').astype(str)
    data=dict(identifier=identifier,policy=policy,config=config,config_hash=config_hash,
        source_hash=source_hash,panel=list(raw),raw_hash=frame_hash(raw),metadata=meta.to_dict('records'),
        baseline_hash=frame_hash(baseline),estimator_version=policy['estimator_version'],code_hash=code)
    return hashlib.sha256(json.dumps(data,sort_keys=True,default=str).encode()).hexdigest()


def run_missingness_study(results,registry,manifest,baseline,families,config,cfg,run_dir,lab_groups,feature_to_lab,source_hash):
    policy=cfg.get('missingness_methods',{})
    if not policy.get('enabled',False): return {'enabled':False,'experiments':[],'artifacts':[]}
    if policy.get('schema_version')!='graph_missingness_methods_v1': raise ValueError('UNKNOWN_MISSINGNESS_PROTOCOL')
    if (policy['references']!=['R0','R2'] or not policy['freeze_original_representations']
            or not policy['freeze_source_feature_panels'] or not policy['no_outcome_based_method_selection']
            or not policy['resampling']['refit_all_within_resample']):
        raise ValueError('MISSINGNESS_PROTOCOL_CONTRACT_CHANGED')
    n=len(baseline); seed=int(config['random_seed']); settings={**cfg['experiments'],**policy['resampling']}
    tables={name:[] for name in ['method_registry','panel_comparability','observation_units','imputation_validation',
        'distance_support_summary','cross_method_comparisons','method_stability','method_coverage','weight_sensitivity',
        'method_qc','resampling_replications','resampling_matching','coassignment','patient_membership',
        'pairwise_support_internal','distance_components_internal','audit_all_representations',
        'imputation_attempts','imputation_by_type','imputation_rank_summary','iterative_method_audit',
        'sparse_graph_registry','sparse_graph_support_summary','sparse_graph_stability','sparse_graph_comparisons',
        'sparse_graph_refits','focal_fix_qc']}
    spaces={}; experiments=[]
    # Audit every historical panel, including Gower. Failures are explicit and
    # cannot change a historical space or R0's release gate.
    for identifier,item in results.items():
        raw=item['space'].raw
        row=dict(representation_id=identifier,n_patients=n,n_encoded_features=raw.shape[1],
            missing_fraction=float(raw.isna().to_numpy().mean()),mapper_coverage=float(item['mapped']['covered'].mean()))
        try:
            panel=build_panel(raw,registry,'R2' if identifier=='R2' else 'R0',baseline)
            row.update(n_constructs=len(panel.metadata),status='COMPLETE',reason='')
            tables['observation_units'].extend(dict(experiment_id=identifier,**r) for r in observation_summary(panel))
            for group_type in ('family','clinical_domain'):
                for group,part in panel.metadata.groupby(group_type):
                    block=panel.mask[list(part.index)]
                    tables['audit_all_representations'].append(dict(**row,group_type=group_type,group=group,
                        n_observation_units=len(part),observed_fraction=float(block.to_numpy().mean())))
            evidence=observed_distance(raw,registry,'R2' if identifier=='R2' else 'R0',policy['observed_pairwise_distance'],baseline)
            tables['distance_support_summary'].append(dict(experiment_id=identifier,analysis_role='audit',**evidence.summary))
        except NotEstimable as exc:
            row.update(status='NOT_ESTIMABLE',reason=str(exc))
        except (ValueError,np.linalg.LinAlgError) as exc:
            row.update(status='FAILED',reason=type(exc).__name__)
        tables['audit_all_representations'].append(dict(**row,group_type='global',group='all'))

    def record(identifier,reference,status,reason,raw=None,**extra):
        source_meta=registry.loc[registry.feature.isin(raw)] if raw is not None else registry.iloc[:0]
        row=dict(experiment_id=identifier,reference_id=reference,input_panel_id=reference+'_fixed',
            method={'I':'iterative_regularized','D':'observed_pairwise_distance','DG':'sparse_observed_graph','M':'values_plus_mask'}[identifier.split('-')[1].split('/')[0]],
            n_patients=n,n_source_variables=source_meta.source_column.nunique(),n_constructs=source_meta.derivation_group.nunique(),
            missing_fraction=float(raw.isna().to_numpy().mean()) if raw is not None else np.nan,
            status=status,reason=reason,seed=seed,config_hash=cfg['_config_sha256'],source_hash=source_hash,
            estimator_version=policy['estimator_version'],cache_version=policy.get('cache_version','legacy_v1'),
            graph03_defining=False,analysis_role='sensitivity',
            evidence_level='EXPLORATORY_PROVISIONAL' if reference=='R2' else 'TECHNICAL_SENSITIVITY')
        if raw is not None:
            row['n_encoded_features']=raw.shape[1]
            try: row['panel_hash']=build_panel(raw,registry,reference,baseline).fingerprint
            except (NotEstimable,ValueError): row['panel_hash']=''
            if identifier.split('-')[1].split('/')[0]=='D':
                support=next((r for r in tables['distance_support_summary'] if r.get('experiment_id')==identifier),{})
                row.update(n_valid_pair_distances=support.get('n_estimable_pairs',np.nan),estimator='observed_pairwise_dense_MDS')
        row.update(extra)
        tables['method_registry'].append(row)
        tables['method_qc'].append(dict(experiment_id=identifier,reference_id=reference,status=status,reason=reason,
            primary_release_gate=False,analysis_role='sensitivity'))
        if status!='COMPLETE':
            if not any(r['experiment_id']==identifier for r in tables['method_coverage']):
                tables['method_coverage'].append(dict(experiment_id=identifier,reference_id=reference,n_patients=n,
                    status=status,reason=reason,coverage_fraction=np.nan,n_covered=np.nan,n_uncovered=np.nan))
            if not any(r.get('experiment_id')==identifier for r in tables['imputation_validation']) and identifier.split('-')[1][0]=='I':
                tables['imputation_validation'].append(dict(experiment_id=identifier,reference_id=reference,
                    status=status,reason=reason,normalized_mae=np.nan,n_hidden=0))
            if not any(r.get('representation_id')==identifier for r in tables['method_stability']):
                requested=policy['resampling']['main_subsampling_replicates_per_variant'] if '/' not in identifier else policy['resampling']['parameter_sensitivity_subsampling_replicates']
                tables['method_stability'].append(dict(representation_id=identifier,reference_id=reference,
                    method='repeated_subsampling',n_requested=requested,n_attempted=0,n_success=0,n_evaluable=0,
                    stability=np.nan,status=status,reason=reason))
        experiments.append(dict(**row,kind='missingness_sensitivity'))
        logging.info('Graph02 missingness %s: %s %s',identifier,status,reason)
        return row

    def collect_imputation(experiment,reference,validation,audit,panel):
        from .s02_imputation import rank_summary
        if validation.empty:
            validation=pd.DataFrame([dict(status=audit.get('status','NOT_ESTIMABLE'),reason=audit.get('reason','NO_VALIDATION'),
                reason_code=audit.get('reason','NO_VALIDATION'),n_hidden=0,normalized_mae=np.nan,baseline_mae=np.nan,
                reference_id=reference,rank=np.nan,repeat=np.nan,split='not_started',n_train=0,n_test=0,
                iterations_run=0,convergence_status='NOT_STARTED',final_relative_change=np.nan,observed_loss=np.nan,
                penalty=np.nan,scale_diagnostic='not_fitted',seed=seed)])
        rows=[dict(experiment_id=experiment,**r) for r in validation.to_dict('records')]
        tables['imputation_validation'].extend(rows); tables['imputation_attempts'].extend(rows)
        tables['imputation_by_type'].extend(rows)
        if 'rank' in validation and validation['rank'].notna().any():
            groups=validation.groupby('validation_stage') if 'validation_stage' in validation else [('not_started',validation)]
            for stage,part in groups:
                tables['imputation_rank_summary'].extend(dict(experiment_id=experiment,validation_stage=stage,**r) for r in
                    rank_summary(part,int(policy['iterative_regularized']['masked_validation_repeats'])).to_dict('records'))
        raw=results[reference]['space'].raw
        nominal=registry.loc[registry.feature.isin(raw)&registry.encoding.astype(str).str.contains('nominal|one_hot',regex=True),'source_column'].nunique()
        audit=dict(audit)
        audit.setdefault('estimator','mixed_iterative_v1_ridge_softmax' if nominal else 'weighted_low_rank_v2_not_imputeMFA')
        tables['iterative_method_audit'].append(dict(experiment_id=experiment,reference_id=reference,panel_hash=panel.fingerprint if panel is not None else '',
            identical_panel=True,panel_changed=False,n_reference_features=raw.shape[1],n_nominal_sources=nominal,
            source_lineage='|'.join(panel.lineage.source_column) if panel is not None else '',policy=json.dumps(policy['iterative_regularized'],sort_keys=True),
            estimator_version=policy['estimator_version'],graph03_defining=False,**{k:v for k,v in audit.items() if k!='n_nominal_sources'}))

    for reference in policy['references']:
        for method,method_name in [('I','iterative_regularized'),('D','observed_pairwise_distance'),('M','values_plus_mask')]:
            identifier=reference+'-'+method
            if reference not in results:
                record(identifier,reference,'NOT_ESTIMABLE','REFERENCE_NOT_ESTIMABLE'); continue
            raw=results[reference]['space'].raw.copy()
            parameters=[(identifier,None)]
            if method=='M': parameters += [(identifier+f'/weight_{w:g}',float(w)) for w in policy['values_plus_mask']['sensitivity_mask_weights']]
            if method=='D': parameters += [(identifier+f'/shared_{w:g}',float(w)) for w in policy['observed_pairwise_distance']['sensitivity_shared_weight_fraction']]
            for experiment,parameter in parameters:
                if not policy[method_name]['enabled']:
                    record(experiment,reference,'NOT_ESTIMABLE','METHOD_DISABLED',raw); continue
                space=None; evidence=None; panel=None
                try:
                    def fit(draw=None):
                        sample=raw if draw is None else raw.iloc[draw].reset_index(drop=True)
                        sample_baseline=baseline if draw is None else baseline.iloc[draw].reset_index(drop=True)
                        candidate,audit=build_variant(identifier,sample,families,config,registry,manifest,sample_baseline,lab_groups,feature_to_lab,
                            policy,reference_space=results[reference]['space'] if draw is None else None,
                            mask_weight=parameter if method=='M' else None,distance_threshold=parameter if method=='D' else None)
                        if list(candidate.features)!=list(raw) or audit['panel_hash']!=panel.fingerprint:
                            raise ValueError('FIXED_PANEL_MISMATCH')
                        return candidate,audit
                    panel=build_panel(raw,registry,reference,baseline)
                    # Save support before attempting Mapper, even for incomplete matrices.
                    if method=='D':
                        distance_policy=dict(policy[method_name])
                        if parameter is not None: distance_policy['minimum_shared_weight_fraction']=parameter
                        evidence=observed_distance(raw,registry,reference,distance_policy,baseline)
                        tables['distance_support_summary'].append(dict(experiment_id=experiment,analysis_role='sensitivity',**evidence.summary))
                        tables['pairwise_support_internal'].extend(dict(experiment_id=experiment,**r) for r in evidence.support.to_dict('records'))
                    tables['panel_comparability'].append(dict(experiment_id=experiment,reference_id=reference,
                        input_panel_id=panel.panel_id,identical_panel=True,n_reference_sources=registry.loc[registry.feature.isin(raw),'source_column'].nunique(),
                        n_candidate_sources=registry.loc[registry.feature.isin(raw),'source_column'].nunique(),n_excluded_sources=0,
                        sources_shared='|'.join(sorted(registry.loc[registry.feature.isin(raw),'source_column'].unique())),
                        excluded_sources='',panel_hash=panel.fingerprint,comparison='within_panel_missingness_only'))
                    space,audit=fit()
                    if 'validation' in audit:
                        collect_imputation(experiment,reference,audit['validation'],audit['imputation'],panel)
                    mapped=analyze(space,config,audit=True)
                    spaces[experiment]={'space':space,'mapped':mapped}
                    if method=='M':
                        a,b=np.triu_indices(n,1)
                        tables['distance_components_internal'].extend(dict(experiment_id=experiment,patient_index_a=int(i),patient_index_b=int(j),
                            value_squared_component=float(audit['value_squared_component'][i,j]),mask_squared_component=float(audit['mask_squared_component'][i,j])) for i,j in zip(a,b))
                    reference_map=results[reference]['mapped']
                    comparison=compare_coverage(reference_map,mapped,int(settings['minimum_comparison_patients']))
                    tables['cross_method_comparisons'].append(dict(experiment_id=experiment,reference_id=reference,
                        input_panel_id=panel.panel_id,**comparison,**geometry_comparison(results[reference]['space'].distance,space.distance)))
                    counts=pd.Series([r['coverage_reason'] for r in mapped['coverage_audit']]).value_counts()
                    for reason,count in counts.items():
                        tables['method_coverage'].append(dict(experiment_id=experiment,reference_id=reference,n_patients=n,
                            n_covered=int(mapped['covered'].sum()),n_uncovered=int((~mapped['covered']).sum()),coverage_fraction=float(mapped['covered'].mean()),
                            n_reference_covered=int(reference_map['covered'].sum()),coverage_difference=float(mapped['covered'].mean()-reference_map['covered'].mean()),
                            coverage_reason=reason,n_reason=int(count),status='COMPLETE',reason=''))
                    budgets=[('repeated_subsampling',int(policy['resampling']['main_subsampling_replicates_per_variant'] if parameter is None else policy['resampling']['parameter_sensitivity_subsampling_replicates']))]
                    if method=='I': budgets.append(('classic_bootstrap',int(policy['resampling']['iterative_bootstrap_replicates'])))
                    qc=True; main_stability=[]; successful_refits=0
                    key=cache_signature(raw,registry,baseline,experiment,policy,config,cfg['_config_sha256'],source_hash)
                    cache_dir=run_dir.parent.parent/'.cache'/policy['estimator_version'] if cfg['execution']['resume'] else None
                    for resampling_method,budget in budgets:
                        sampled=resample(mapped['community_sets'],n,lambda draw:fit(draw)[0],lambda s:analyze(s,config),settings,
                            seed=seed,identifier=experiment,method=resampling_method,replicates=budget,cache_dir=cache_dir,cache_key=key)
                        qc &= sampled['qc_pass']
                        successful_refits+=sampled['n_success']
                        tables['method_stability'].extend(dict(reference_id=reference,stability_threshold=settings['stable_threshold'],**r) for r in sampled['stability'])
                        if not sampled['stability']:
                            tables['method_stability'].append(dict(representation_id=experiment,reference_id=reference,method=resampling_method,
                                n_requested=budget,n_success=sampled['n_success'],n_evaluable=0,status='NOT_ESTIMABLE',
                                reason='NO_REFERENCE_COMMUNITIES',stability=np.nan))
                        from .s02_resampling import draw_indices
                        for attempt in sampled['attempts']:
                            draw=draw_indices(n,float(settings['patient_fraction']),attempt['seed'],resampling_method)
                            attempt.update(mask_sha256=hashlib.sha256(raw.iloc[draw].notna().to_numpy().tobytes()).hexdigest(),
                                panel_hash=panel.fingerprint,estimator_version=policy['estimator_version'])
                        tables['resampling_replications'].extend(dict(reference_id=reference,**r) for r in sampled['attempts'])
                        tables['resampling_matching'].extend(dict(reference_id=reference,**r) for r in sampled['alignments'])
                        if resampling_method=='repeated_subsampling':
                            main_stability=sampled['stability']
                            a,b=np.triu_indices(n,1)
                            for i,j in zip(a,b):
                                denominator=int(sampled['pair_denom'][i,j]); numerator=int(sampled['pair_same'][i,j])
                                tables['coassignment'].append(dict(experiment_id=experiment,patient_index_a=int(i),patient_index_b=int(j),
                                    n_evaluable=denominator,n_coassigned=numerator,coassignment_fraction=numerator/denominator if denominator else np.nan))
                    statuses={r['community_id']:'stable' if r['stable'] else 'provisional' for r in main_stability}
                    tables['patient_membership'].extend(frozen_membership(experiment,baseline,mapped,statuses))
                    row=record(experiment,reference,'COMPLETE' if qc else 'FAILED','' if qc else 'INSUFFICIENT_SUCCESSFUL_REFITS',raw,
                        panel_hash=panel.fingerprint,n_encoded_features=raw.shape[1],n_observation_units=len(panel.metadata),
                        estimator=audit.get('imputation',{}).get('estimator',space.method),
                        n_shared=comparison.get('n_common',0),n_successful_refits=successful_refits,n_valid_pair_distances=int(np.isfinite(space.distance[np.triu_indices(n,1)]).sum()),
                        mask_weight=audit.get('mask_weight'),n_covered=int(mapped['covered'].sum()),n_communities=len(mapped['community_sets']))
                    if parameter is not None: tables['weight_sensitivity'].append(dict(**row,parameter=parameter))
                except NotEstimable as exc:
                    if method=='I' and not any(r['experiment_id']==experiment for r in tables['imputation_attempts']):
                        collect_imputation(experiment,reference,getattr(exc,'validation',pd.DataFrame()),
                            dict(status='NOT_ESTIMABLE',reason=str(exc)),panel)
                    row=record(experiment,reference,'NOT_ESTIMABLE',str(exc),raw)
                    if evidence is not None:
                        tables['cross_method_comparisons'].append(dict(experiment_id=experiment,reference_id=reference,
                            status='NOT_ESTIMABLE',reason=str(exc),n_total=n,**geometry_comparison(results[reference]['space'].distance,evidence.distance)))
                    if parameter is not None: tables['weight_sensitivity'].append(dict(**row,parameter=parameter))
                except MemoryError:
                    record(experiment,reference,'NOT_ESTIMABLE','METHOD_MEMORY_ALLOCATION_FAILED',raw)
                except Exception as exc:
                    # Invalid code/configuration is a method failure, never a
                    # successful fit or an unnoticed median fallback.
                    logging.exception('Graph02 sensitivity failed: %s',experiment)
                    record(experiment,reference,'FAILED',type(exc).__name__+':'+str(exc).split(':')[0],raw)
    # DG never enters `spaces`, Mapper analysis, or Graph03 membership exports.
    from .s02_sparse_observed_graph import build_sparse_graph,graph_resampling
    graph_policy=policy.get('sparse_observed_graph',{})
    if graph_policy.get('enabled',False):
        for reference in policy['references']:
            identifier=reference+'-DG'
            if reference not in results:
                record(identifier,reference,'NOT_ESTIMABLE','REFERENCE_NOT_ESTIMABLE'); continue
            raw=results[reference]['space'].raw.copy()
            panel=build_panel(raw,registry,reference,baseline)
            for k in [graph_policy['primary_k'],*graph_policy['sensitivity_k']]:
                main=k==graph_policy['primary_k']; experiment=identifier if main else identifier+f'/k_{k}'
                try:
                    evidence=observed_distance(raw,registry,reference,policy['observed_pairwise_distance'],baseline)
                    graph=build_sparse_graph(evidence,list(range(n)),graph_policy,k=k,seed=seed,panel_hash=panel.fingerprint)
                    stats=graph.summary
                    tables['sparse_graph_support_summary'].append(dict(experiment_id=experiment,reference_id=reference,**stats))
                    success=0; comparison={}
                    if stats['status']=='COMPLETE':
                        budget=policy['resampling']['main_subsampling_replicates_per_variant'] if main else policy['resampling']['parameter_sensitivity_subsampling_replicates']
                        stability,attempts,success=graph_resampling(raw,registry,baseline,reference,graph_policy,
                            policy['observed_pairwise_distance'],settings,k=k,seed=seed,replicates=budget,reference_graph=graph)
                        tables['sparse_graph_stability'].extend(dict(experiment_id=experiment,**r) for r in stability)
                        tables['sparse_graph_refits'].extend(dict(experiment_id=experiment,**r) for r in attempts)
                        comparison=compare_coverage(results[reference]['mapped'],graph.mapped,int(settings['minimum_comparison_patients']))
                        tables['sparse_graph_comparisons'].append(dict(experiment_id=experiment,reference_id=reference,
                            comparison_warning='cross_algorithm_not_mapper',n_pairs=evidence.summary['n_pairs'],
                            **comparison,**geometry_comparison(results[reference]['space'].distance,evidence.distance)))
                        if success/budget<settings['minimum_success_fraction']:
                            stats={**stats,'status':'FAILED','reason':'INSUFFICIENT_SUCCESSFUL_GRAPH_REFITS'}
                    else:
                        tables['sparse_graph_stability'].append(dict(experiment_id=experiment,k=k,status='NOT_ESTIMABLE',
                            reason=stats['reason'],n_successful_refits=0,n_evaluable=0,stability=np.nan))
                        tables['sparse_graph_comparisons'].append(dict(experiment_id=experiment,reference_id=reference,
                            status='NOT_ESTIMABLE',reason=stats['reason'],comparison_warning='cross_algorithm_not_mapper',n_common=0,n_pairs=evidence.summary['n_pairs']))
                    stats={**stats,'n_successful_refits':success}
                    tables['sparse_graph_registry'].append(dict(experiment_id=experiment,reference_id=reference,**stats))
                    record(experiment,reference,stats['status'],stats['reason'],raw,
                        **{key:stats[key] for key in ['method_type','is_mapper','n_connected','n_covered','n_communities',
                            'n_valid_pair_distances','n_successful_refits']},n_shared=comparison.get('n_common',0),estimator='sparse_observed_graph_v1',k=k)
                except (NotEstimable,ValueError,np.linalg.LinAlgError) as exc:
                    status='NOT_ESTIMABLE' if isinstance(exc,NotEstimable) else 'FAILED'
                    record(experiment,reference,status,str(exc),raw,method_type='sparse_observed_graph',is_mapper=False,k=k)
                    for name in ['sparse_graph_registry','sparse_graph_support_summary','sparse_graph_stability','sparse_graph_comparisons']:
                        tables[name].append(dict(experiment_id=experiment,status=status,reason=str(exc),k=k,method_type='sparse_observed_graph'))
    for row in tables['method_registry']:
        if row['experiment_id'].endswith(('-I','-D','-DG')):
            tables['focal_fix_qc'].append(dict(experiment_id=row['experiment_id'],status='PASS' if row['status']=='COMPLETE' else row['status'],
                reason=row['reason'],panel_hash=row.get('panel_hash',''),graph03_defining=False,primary_release_gate=False))
    # Stable additive schemas exist even when no model can be fitted.
    schemas={
        'imputation_attempts':['experiment_id','reference_id','rank','repeat','source_column','split','n_train','n_test','n_hidden',
            'n_observed_per_feature','iterations_run','convergence_status','reason_code','final_relative_change','observed_loss','penalty',
            'scale_diagnostic','normalized_mae','baseline_mae','seed','status','reason'],
        'imputation_by_type':['experiment_id','reference_id','variable_type','n_hidden','normalized_mae','baseline_mae','accuracy',
            'balanced_accuracy','baseline_accuracy','baseline_balanced_accuracy','class_support','validation_stage','status','reason'],
        'iterative_method_audit':['experiment_id','reference_id','estimator','estimator_version','panel_hash','identical_panel','n_source_variables',
            'n_encoded_features','n_nominal_sources','n_observed_preserved','seed','policy','status','reason'],
        'sparse_graph_registry':['experiment_id','reference_id','method_type','is_mapper','graph03_defining','status','reason','k','n_patients',
            'n_pairs','n_valid_pair_distances','n_connected','n_covered','n_communities','n_successful_refits','panel_hash'],
        'sparse_graph_support_summary':['experiment_id','reference_id','status','reason','k','n_patients','n_pairs','n_estimable_pairs',
            'n_edges','n_connected','n_isolated','n_components','n_eligibility_components','degree_mean'],
        'sparse_graph_stability':['experiment_id','status','reason','k','n_requested','n_successful_refits','n_evaluable','stability','ci_low','ci_high'],
        'sparse_graph_comparisons':['experiment_id','reference_id','status','reason','comparison_warning','n_common','n_covered_reference',
            'n_covered_control','n_pairs','n_valid_distance_pairs','ari_primary'],
        'focal_fix_qc':['experiment_id','status','reason','panel_hash','graph03_defining','primary_release_gate']}
    for name,rows in tables.items():
        frame=pd.DataFrame(rows)
        if name in schemas: frame=frame.reindex(columns=list(dict.fromkeys([*schemas[name],*frame.columns])))
        save_table(run_dir/f'02_graph_missingness_{name}.csv',frame)
    from .s02_reporting import missingness_method_figures
    figures=missingness_method_figures(run_dir,spaces,tables,int(settings.get('minimum_publication_cell',5)),results)
    from .s02_reporting import focal_fix_figures
    figures+=focal_fix_figures(run_dir,tables,int(settings.get('minimum_publication_cell',5)))
    primary=[r for r in tables['method_registry'] if '/' not in r['experiment_id']]
    summary=['# Graph02: sensibilidad transversal a datos faltantes',
        'Análisis técnico exploratorio. R0 y R2 conservan su definición y membresías históricas. Las variantes son sensibilidades no definitorias para Graph03.',
        'Panel fijo: fuentes y pacientes de la referencia; refit completo dentro de cada réplica. La comparación R0/R2 combina selección, ponderación y geometría y no aísla missingness.',
        '| Experimento | Estado | Causa |','| --- | --- | --- |']
    summary += [f'| {r["experiment_id"]} | {r["status"]} | {r["reason"] or "—"} |' for r in primary]
    summary += ['La estabilidad entre métodos indica robustez metodológica, no valida fenotipos. La disponibilidad no identifica MCAR/MAR/MNAR ni causalidad. Las comunidades no indican progresión temporal.',
        'Las distancias incompletas conservan NA y no se entregan a Mapper/MDS. La validación de reconstrucción oculta celdas observadas artificialmente y separa pacientes de entrenamiento y evaluación.',
        'Los CSV por paciente, soporte de pares y componentes son internos/restringidos. No hay exportación automática. Las figuras agregadas suprimen celdas con soporte inferior al mínimo de publicación.',
        'M1 y N2a–N2d son controles históricos independientes; R0-M no es un alias de M1. Las variantes de máscara incluyen estados distintos de ausencia parcial o desconocida.']
    (run_dir/'02_graph_missingness_integrated_summary.md').write_text('\n\n'.join(summary)+'\n')
    focal=['# Graph02 focal correction',
        'DG is a sparse observed graph, not Mapper; all I/D/DG results are non-defining sensitivities. R2 remains exploratory provisional.',
        'Nominal sources: chained ridge/softmax on source-level categories. Numeric panels: weighted_low_rank_v2_not_imputeMFA. Neither implementation is missMDA::imputeMFA.',
        'D retains unknown pairs as NA. DG uses only supported observed pairs; absent edges do not indicate large distances. Comparisons are cross_algorithm_not_mapper.',
        'Training/evaluation patients and training-only median/mode baselines are identical for each method comparison; metrics of different variable types are not pooled.',
        'All artifacts remain INTERNAL_RESTRICTED. Aggregate figures suppress small cells. No automatic sharing package is produced.',
        'No biological phenotype, causality, absence of MNAR bias, or temporal progression is established.',
        '| Method | Status | Reason | Patients | Covered | Groups | Valid pairs | Connected | Refits | Estimator | Panel hash |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in primary:
        focal.append('| '+' | '.join(str(r.get(c,'NA')) for c in ['experiment_id','status','reason','n_patients','n_covered','n_communities',
            'n_valid_pair_distances','n_connected','n_successful_refits','estimator','panel_hash'])+' |')
    (run_dir/'02_graph_missingness_focal_fix_summary.md').write_text('\n\n'.join(focal)+'\n')
    artifacts=[p.relative_to(run_dir).as_posix() for p in sorted(run_dir.glob('02_graph_missingness_*')) if p.is_file()]+figures
    return dict(enabled=True,schema_version=policy['schema_version'],estimator_version=policy['estimator_version'],
        analysis_role='sensitivity',references=policy['references'],panel_policy='frozen_reference_sources_and_patients',
        graph03_defining=False,experiments=experiments,artifacts=artifacts,
        statuses={r['experiment_id']:dict(status=r['status'],reason=r['reason']) for r in tables['method_registry']},
        counts={status:sum(r['status']==status for r in primary) for status in ['COMPLETE','FAILED','NOT_ESTIMABLE']})

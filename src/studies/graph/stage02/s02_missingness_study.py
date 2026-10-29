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
        'pairwise_support_internal','distance_components_internal','audit_all_representations']}
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
            method={'I':'iterative_regularized','D':'observed_pairwise_distance','M':'values_plus_mask'}[identifier.split('-')[1][0]],
            n_patients=n,n_source_variables=source_meta.source_column.nunique(),n_constructs=source_meta.derivation_group.nunique(),
            missing_fraction=float(raw.isna().to_numpy().mean()) if raw is not None else np.nan,
            status=status,reason=reason,seed=seed,config_hash=cfg['_config_sha256'],source_hash=source_hash,
            estimator_version=policy['estimator_version'],analysis_role='sensitivity',
            evidence_level='EXPLORATORY_PROVISIONAL' if reference=='R2' else 'TECHNICAL_SENSITIVITY',**extra)
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
                space=None; evidence=None
                try:
                    def fit(draw=None):
                        sample=raw if draw is None else raw.iloc[draw].reset_index(drop=True)
                        sample_baseline=baseline if draw is None else baseline.iloc[draw].reset_index(drop=True)
                        return build_variant(identifier,sample,families,config,registry,manifest,sample_baseline,lab_groups,feature_to_lab,
                            policy,reference_space=results[reference]['space'] if draw is None else None,
                            mask_weight=parameter if method=='M' else None,distance_threshold=parameter if method=='D' else None)
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
                    mapped=analyze(space,config,audit=True)
                    spaces[experiment]={'space':space,'mapped':mapped}
                    if 'validation' in audit:
                        tables['imputation_validation'].extend(dict(experiment_id=experiment,**r) for r in audit['validation'].to_dict('records'))
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
                    qc=True; main_stability=[]
                    key=cache_signature(raw,registry,baseline,experiment,policy,config,cfg['_config_sha256'],source_hash)
                    cache_dir=run_dir.parent.parent/'.cache'/policy['estimator_version'] if cfg['execution']['resume'] else None
                    for resampling_method,budget in budgets:
                        sampled=resample(mapped['community_sets'],n,lambda draw:fit(draw)[0],lambda s:analyze(s,config),settings,
                            seed=seed,identifier=experiment,method=resampling_method,replicates=budget,cache_dir=cache_dir,cache_key=key)
                        qc &= sampled['qc_pass']
                        tables['method_stability'].extend(dict(reference_id=reference,stability_threshold=settings['stable_threshold'],**r) for r in sampled['stability'])
                        if not sampled['stability']:
                            tables['method_stability'].append(dict(representation_id=experiment,reference_id=reference,method=resampling_method,
                                n_requested=budget,n_success=sampled['n_success'],n_evaluable=0,status='NOT_ESTIMABLE',
                                reason='NO_REFERENCE_COMMUNITIES',stability=np.nan))
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
                        mask_weight=audit.get('mask_weight'),n_covered=int(mapped['covered'].sum()),n_communities=len(mapped['community_sets']))
                    if parameter is not None: tables['weight_sensitivity'].append(dict(**row,parameter=parameter))
                except NotEstimable as exc:
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
    for name,rows in tables.items():
        save_table(run_dir/f'02_graph_missingness_{name}.csv',rows)
    from .s02_reporting import missingness_method_figures
    figures=missingness_method_figures(run_dir,spaces,tables,int(settings.get('minimum_publication_cell',5)),results)
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
    artifacts=[p.relative_to(run_dir).as_posix() for p in sorted(run_dir.glob('02_graph_missingness_*')) if p.is_file()]+figures
    return dict(enabled=True,schema_version=policy['schema_version'],estimator_version=policy['estimator_version'],
        analysis_role='sensitivity',references=policy['references'],panel_policy='frozen_reference_sources_and_patients',
        graph03_defining=False,experiments=experiments,artifacts=artifacts,
        statuses={r['experiment_id']:dict(status=r['status'],reason=r['reason']) for r in tables['method_registry']},
        counts={status:sum(r['status']==status for r in primary) for status in ['COMPLETE','FAILED','NOT_ESTIMABLE']})

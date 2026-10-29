"""Full Graph02 execution: prespecified experiments, QC, and immutable release."""
from copy import deepcopy
from functools import partial
from pathlib import Path
import hashlib
import json
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from .._pipeline import (load_pipeline, config_path, output_root, new_run, save_table, write_json,
    digest, utc_now, git_commit, version_info, artifact_inventory, base_parser)
from .. import _feature_catalog as catalog
from . import s02_legacy as legacy
from .s02_contracts import SCHEMA, resolve_sources, publish_pointer
from .s02_feature_registry import clinical_registry, domain_coverage, eligible_domain_map, verify_observed_baseline_dates, mapping_counts, RESERVED, CLINICAL
from .s02_representations import build_space, Space, hierarchical_mfa
from .s02_mapper_engine import analyze, frozen_membership
from .s02_resampling import resample, match_sets
from .s02_perturbations import permute_observed, mask_control
from .s02_comparators import comparators
from .s02_decision_rules import release_decision
from .s02_reporting import figures, save_figure
from .s02_missingness import compare_coverage, run_observation_diagnostics, administrative_frame, administrative_strata
from .s02_reporting import diagnostic_figures, results_summary, privacy_inventory


def prepare_inputs(cfg):
    sources=resolve_sources(cfg)
    master,candidates,contract=legacy.CANON.load_real_inputs(sources['integrated'],sources['pharma_candidates'],
        cfg['analysis']['mode'],resolve_profile=False)
    baseline=legacy.CANON.select_baseline(master)
    if baseline.patient_id.isna().any(): raise ValueError('Missing baseline patient IDs')
    baseline['patient_id']=baseline.patient_id.astype(str)
    age_path=config_path(cfg['_path'],cfg['paths']['baseline_metrics'])
    age=legacy.CANON.load_age_at_diagnosis(age_path)
    age['patient_id']=age.patient_id.astype(str)
    baseline=baseline.merge(age,on='patient_id',how='left',validate='one_to_one',suffixes=('','_blockA'))
    if 'age_dx_blockA' in baseline: baseline['age_dx']=baseline.pop('age_dx_blockA')
    config=deepcopy(cfg['sensitivity'])
    config['analysis_mode']=cfg['analysis']['mode']; config['feature_policy']=cfg['analysis']['feature_policy']
    canonical=deepcopy(cfg['canonical']); canonical['feature_policy']=config['feature_policy']
    from types import SimpleNamespace
    legacy.validate_canonical_config(config,canonical,SimpleNamespace(canonical_config='pipeline canonical'))
    manifest,values,_,_=legacy.CANON.build_feature_manifest(baseline,candidates,
        float(config['feature_selection']['minimum_baseline_coverage']),analysis_mode=cfg['analysis']['mode'],
        feature_policy=cfg['analysis']['feature_policy'],metadata=config.get('feature_metadata',{}),source_hash=contract['input_hash'])
    verify_observed_baseline_dates(baseline,manifest,cfg['validation']['date_column'])
    governance=legacy.CANON.enforce_lab_mode(manifest,analysis_mode=cfg['analysis']['mode'],
        release_path=config_path(cfg['_path'],cfg['paths']['clinical_release']),
        integrated_path=sources['integrated'],candidates_path=sources['pharma_candidates'])
    for name in values:
        if np.isinf(pd.to_numeric(values[name]).to_numpy(float)).any():
            raise ValueError('Infinite feature input')
    return master,baseline,candidates,manifest,values,config,sources,governance


def run(cfg, *, root=None, run_id=None, preflight_only=False):
    root=Path(root) if root else output_root(cfg,'02')
    run_dir=new_run(root,run_id)
    started=utc_now()
    try:
        return _execute(cfg,root,run_dir,started,preflight_only)
    except Exception as exc:
        write_json(run_dir/'02_graph_release_qc.json',{'release_status':'QC_FAILED','execution_status':'FAILED',
            'exception_type':type(exc).__name__,'reason':str(exc).split(':')[0], 'created_at_utc':utc_now()})
        # Failed runs never replace the last completed pointer.
        raise


def _execute(cfg,root,run_dir,started,preflight_only):
    master,baseline,candidates,manifest,values,config,sources,governance=prepare_inputs(cfg)
    settings=cfg['experiments']; seed=int(config['random_seed']); n=len(baseline)
    families={row.feature:legacy.family_name(row.family) for row in manifest.loc[manifest.included].itertuples()}
    primary,_,_=legacy.select_features(values,families,'S3',config,manifest)
    registry=clinical_registry(manifest,cfg['clinical_domains'],primary)
    save_table(run_dir/'02_graph_clinical_domain_registry.csv',registry)
    write_json(run_dir/'02_graph_preflight.json',{'status':'PASS','n_patients':n,'n_features':len(values),
        'analysis_mode':cfg['analysis']['mode'],'feature_policy':cfg['analysis']['feature_policy'],**governance})
    if preflight_only:
        return {'run_dir':run_dir,'execution_status':'PREFLIGHT_ONLY','manifest':None}
    mapping_error=None
    try:
        lab_frame=legacy.load_lab_group_map(config_path(cfg['_path'],cfg['paths']['lab_group_map']))
        lab_groups=legacy.lab_group_lookup(lab_frame)
    except (OSError,ValueError) as exc:
        mapping_error=str(exc); lab_groups={}
        lab_frame=pd.DataFrame(columns=['lab','s4_group','s4_subgroup','mapping_status'])
    feature_to_lab={r.feature:legacy.canonical_lab_name(r.lab) for r in manifest.loc[manifest.family.eq('lab')].itertuples()}
    experiments=[]; results={}; selection=[]; feature_sets=[]; weights=[]; coverage=[]
    attempts=[]; stability=[]; matches=[]; refit_membership=[]; legacy_members=[]; member_rows=[]; negative=[]; comparator_rows=[]
    nodes=[]; edges=[]; dictionary=[]; pair_rows=[]; logs=[]; qc_by_id={}
    cache_dir=root/'.cache'/settings['cache_version'] if cfg['execution']['resume'] else None
    map_path=config_path(cfg['_path'],cfg['paths']['lab_group_map'])
    cache_base=digest(sources['integrated'])+cfg['_config_sha256']+settings['cache_version']+catalog.frame_hash(pd.DataFrame(values))
    cache_base+=digest(map_path) if map_path.is_file() else 'NO_S4_MAP'
    cache_base+=json.dumps(cfg['clinical_domains'],sort_keys=True)+json.dumps(settings,sort_keys=True)
    cache_base+=SCHEMA+catalog.CONTRACT_VERSION+'graph02_algorithm_v3/'+str(seed)
    stable_ids={}; memberships={}

    def record(identifier,kind,status='COMPLETE',reason='',**more):
        row={'experiment_id':identifier,'kind':kind,'status':status,'reason':reason,'n_patients':n,**more}
        experiments.append(row)
        logs.append({'event':'experiment_completed','time_utc':utc_now(),**row})
        return row

    def select_and_fit(identifier,draw=None,exclude=None,custom_values=None,effective_config=None):
        effective_config=effective_config or config
        scenario={'R0':'S3','R1':'S4','R2':'S3','R3':'S3','R4':'S3'}.get(identifier,identifier)
        source=custom_values if custom_values is not None else values
        sample_baseline=baseline
        if draw is not None:
            source={name:s.iloc[draw].reset_index(drop=True) for name,s in source.items()}
            sample_baseline=baseline.iloc[draw].reset_index(drop=True)
        if exclude: source={name:s for name,s in source.items() if name not in exclude}
        if identifier=='R2':
            gate='clinically_approved' if cfg['analysis']['mode']=='clinical_release' else 'exploratory_technical_eligible'
            allowed=set(registry.loc[registry[gate],'feature'])
            source={name:s for name,s in source.items() if name in allowed}
        chosen,cov,red=legacy.select_features(source,families,scenario,effective_config,manifest)
        raw=pd.DataFrame(chosen,index=sample_baseline.index)
        if len(raw.columns)<2: raise ValueError('INSUFFICIENT_SELECTED_FEATURES')
        if scenario=='S4':
            if mapping_error: raise ValueError('S4_MAPPING_NOT_AVAILABLE')
            legacy.validate_s4_mapping(source,families,lab_frame,feature_to_lab,set(chosen),require_all=True)
        space=build_space(identifier,raw,families,effective_config,registry,manifest,sample_baseline,lab_groups,feature_to_lab)
        return space,cov,red

    def refit(identifier,**kwargs):
        return lambda draw:select_and_fit(identifier,draw,**kwargs)[0]

    def evaluate_resampling(identifier,space,mapped, *, method='repeated_subsampling',budget=None,
                            preliminary=False,effective_config=None,fit=None,experiment_id=None):
        effective_config=effective_config or config
        result=resample(mapped['community_sets'],n,fit or refit(identifier,effective_config=effective_config),
            partial(analyze,config=effective_config),settings,seed=seed,identifier=experiment_id or identifier,
            method=method,replicates=budget,cache_dir=cache_dir,
            cache_key=cache_base+json.dumps(effective_config,sort_keys=True)+'|'.join(sorted(space.features)),preliminary=preliminary)
        attempts.extend(result['attempts']); stability.extend(result['stability']); matches.extend(result['alignments'])
        for row in result['assignments']:
            patient=baseline.patient_id.iloc[row.pop('patient_index')]
            refit_membership.append({**row,'patient_id':patient})
        if not preliminary and method=='repeated_subsampling':
            for a in range(n):
                for b in range(a+1,n):
                    denom=int(result['pair_denom'][a,b])
                    pair_rows.append({'representation_id':identifier,'patient_a':baseline.patient_id.iloc[a],
                        'patient_b':baseline.patient_id.iloc[b],'n_evaluable':denom,'n_coassigned':int(result['pair_same'][a,b]),
                        'coassignment_fraction':result['pair_same'][a,b]/denom if denom else np.nan})
        record((experiment_id or identifier)+'/'+method,'resampling','COMPLETE' if result['qc_pass'] else 'FAILED',
               '' if result['qc_pass'] else 'INSUFFICIENT_SUCCESSFUL_REFITS',n_requested=result['n_requested'],n_success=result['n_success'],phase='preliminary' if preliminary else 'final')
        return result

    identifiers=[*settings['legacy_scenarios'],*settings['representations']]
    for identifier in identifiers:
        try:
            if identifier=='R0' and 'S3' in results:
                space=results['S3']['space']; cov=results['S3']['coverage']; red=results['S3']['redundancy']
            elif identifier=='R1' and 'S4' in results:
                space=results['S4']['space']; cov=results['S4']['coverage']; red=results['S4']['redundancy']
            else:
                space,cov,red=select_and_fit(identifier)
            mapped=analyze(space,config,audit=identifier=='R0')
            results[identifier]={'space':space,'mapped':mapped,'coverage':cov,'redundancy':red}
            coverage.extend(domain_coverage(space.raw,registry,identifier))
            weights.extend({'representation_id':identifier,**r} for r in space.weights.to_dict('records'))
            selected=set(space.features)
            for r in manifest.itertuples():
                feature_sets.append({'scenario':identifier,'representation_id':identifier,'feature':r.feature,
                    'source_column':r.source_column,'family':r.family,'included':r.feature in selected,
                    'role':'defining' if r.feature in selected else 'not_in_embedding',
                    'semantic_feature_id':r.source_column,'encoding':r.encoding})
            audit=manifest.copy(); audit['scenario']=identifier; audit['included_pca']=audit.feature.isin(selected)
            audit['coverage_pass']=audit.feature.map(cov.set_index('feature').included_after_coverage_filter).eq(True)
            audit['redundancy_pass']=~audit.feature.isin(red.feature_removed)
            selection.append(audit)
            evidence='APPROVED' if identifier=='R2' and registry.loc[registry.feature.isin(space.features),'clinically_approved'].all() else 'EXPLORATORY_PROVISIONAL' if identifier=='R2' else 'TECHNICAL_REFERENCE_OR_SENSITIVITY'
            record(identifier,'representation',method=space.method,n_features=len(space.raw.columns),evidence_level=evidence,
                **mapping_counts(registry,space.features),
                mapper_coverage=float(mapped['covered'].mean()),n_nodes=len(mapped['nodes']),n_communities=len(mapped['community_sets']))
            # R0 and R1 reuse identical full fits, not duplicate stability estimates.
            if identifier in {'S3','S4'}: continue
            primary_budget=identifier in settings['final_representation_ids'] and (identifier!='R3' or (mapped['covered'].mean()>=settings['minimum_mapper_coverage'] and len(mapped['communities'])>0))
            budget=int(settings['final_subsampling_replicates'] if primary_budget else settings['secondary_subsampling_replicates'])
            boot=evaluate_resampling(identifier,space,mapped,budget=budget)
            qc_by_id[identifier]=boot['qc_pass']
            statuses={row['community_id']:'stable' if row['stable'] else 'provisional' if boot['qc_pass'] else 'QC_FAILED' for row in boot['stability']}
            member_rows.extend(frozen_membership(identifier,baseline,mapped,statuses))
            for c,group in enumerate(mapped['community_sets']):
                dictionary.append({'representation_id':identifier,'community_id':c,'n_patients_node_union':len(group),
                    'supported':bool(mapped['supported'][c]),'structural_stability_status':statuses.get(c,'not_evaluated'),
                    'object_type':'community','branch_algorithm':None})
            legacy_frame=legacy.patient_membership(identifier,baseline,mapped)
            memberships[identifier]=legacy_frame
            legacy_members.append(legacy_frame)
            for node,patients in mapped['nodes'].items():
                component=next(i for i,s in enumerate(mapped['components']) if node in s)
                community=next(i for i,s in enumerate(mapped['communities']) if node in s)
                for i in sorted(patients):
                    nodes.append({'representation_id':identifier,'node_id':node,'patient_id':baseline.patient_id.iloc[i],
                        'node_support':len(patients),'connected_component':component,'community_id':community})
            for a,b,attr in mapped['nerve'].edges(data=True):
                edges.append({'representation_id':identifier,'source_node':a,'target_node':b,'weight':attr.get('weight',np.nan),'weight_type':'patient_jaccard'})
            if identifier in {'R0','R2'} and settings['run_separate_classic_bootstrap_sensitivity']:
                evaluate_resampling(identifier,space,mapped,method='classic_bootstrap',budget=int(settings['bootstrap_replicates']))
        except (ValueError,np.linalg.LinAlgError) as exc:
            if identifier=='R0': raise ValueError('PRIMARY_REFERENCE_NOT_ESTIMABLE') from exc
            record(identifier,'representation','NOT_ESTIMABLE',str(exc))
    if 'R0' not in results: raise ValueError('PRIMARY_REFERENCE_NOT_BUILT')
    # S3/R0 and S4/R1 are exact aliases; retain historical identifiers and schemas.
    for old,new in [('S3','R0'),('S4','R1')]:
        if new in memberships:
            frame=memberships[new].copy(); frame['scenario']=old; legacy_members.append(frame); memberships[old]=frame
            for row in list(member_rows):
                if row['representation_id']==new: member_rows.append({**row,'representation_id':old})
    reference=results['R0']; base_space=reference['space']; base_map=reference['mapped']

    comparisons=[]; perturbations={}
    def perturbation(identifier,space,mapped=None,kind='perturbation',fit=None):
        mapped=mapped or analyze(space,config)
        comparison=compare_coverage(base_map,mapped,int(settings.get('minimum_comparison_patients',10)))
        comparison['comparison_status']=comparison.pop('status'); comparison['comparison_reason']=comparison.pop('reason')
        row=record(identifier,kind,mapper_coverage=float(mapped['covered'].mean()),
                   n_nodes=len(mapped['nodes']),fragmentation=len(mapped['components'])/max(1,len(mapped['nodes'])),
                   n_communities=len(mapped['community_sets']),**comparison)
        comparisons.append(row); perturbations[identifier]={'space':space,'mapped':mapped}
        if fit is not None and settings['run_resampling']:
            evaluate_resampling(identifier,space,mapped,fit=fit,budget=int(settings['secondary_subsampling_replicates']))
        return row

    # Context exclusions happen before every selection/imputation/scaling/PCA fit.
    contextual=set(registry.loc[registry.clinical_domain.isin(['D08','D09']) | registry.role.isin(RESERVED),'feature'])
    try:
        space,_,_=select_and_fit('R0',exclude=contextual)
        perturbation('A5/context_excluded',space,kind='context_exclusion',fit=refit('R0',exclude=contextual))
    except (ValueError,np.linalg.LinAlgError) as exc:
        record('A5/context_excluded','context_exclusion','NOT_ESTIMABLE',str(exc))

    if settings['run_lens_distance_sensitivity']:
        perturbation('D1',base_space,base_map,'lens_distance')
        for identifier in ('D2','D3','D4','D5','D6'):
            try:
                if identifier=='D2':
                    meta=eligible_domain_map(registry,list(base_space.raw),purpose=cfg['analysis']['mode'])
                    scores=[]
                    for _,part in meta.groupby('clinical_domain'):
                        block=base_space.raw[part.index]
                        scores.append(block.sub(block.mean()).div(block.std().replace(0,np.nan)).mean(axis=1).fillna(0).to_numpy())
                    if len(scores)<2: raise ValueError('DOMAIN_LENS_REQUIRES_TWO_ELIGIBLE_DOMAINS')
                    mapped=analyze(base_space,config,lens=np.column_stack(scores)[:,:2])
                elif identifier=='D3':
                    full,_,_=legacy.prepare_matrix(base_space.raw,families,True)
                    mapped=analyze(base_space,config,distance=pairwise_distances(full))
                elif identifier=='D4':
                    if 'R3' not in results: raise ValueError('GOWER_NOT_ESTIMABLE')
                    mapped=results['R3']['mapped']
                elif identifier=='D5':
                    try: import umap
                    except ImportError as exc: raise ValueError('OPTIONAL_UMAP_DEPENDENCY_UNAVAILABLE') from exc
                    for lens_seed in settings['umap_seeds']:
                        lens=umap.UMAP(n_components=2,random_state=int(lens_seed)).fit_transform(base_space.cluster)
                        perturbation(f'D5/seed_{lens_seed}',base_space,analyze(base_space,config,lens=lens),'lens_distance')
                    continue
                else:
                    for dimension in settings['pca_dimensions']:
                        matrix,_,_=legacy.prepare_matrix(base_space.raw,families,True)
                        dim=min(int(dimension),len(matrix)-1,matrix.shape[1])
                        x=PCA(n_components=dim,random_state=seed).fit_transform(matrix)
                        space=Space(base_space.raw,x[:,:2],x,pairwise_distances(x),pd.DataFrame(),'PCA_fixed',list(base_space.raw))
                        perturbation(f'D6/components_{dimension}',space,kind='lens_distance')
                    continue
                perturbation(identifier,base_space,mapped,'lens_distance')
            except (ValueError,np.linalg.LinAlgError) as exc:
                record(identifier,'lens_distance','NOT_ESTIMABLE',str(exc))
    if settings['run_mapper_grid']:
        grid=settings['mapper_grid']
        for representation in grid['representations']:
            for cubes in grid['n_cubes']:
                for overlap in grid['perc_overlap']:
                    identifier=f'T/{representation}/c{cubes}/o{overlap}'
                    if representation not in results:
                        record(identifier,'mapper_grid','NOT_ESTIMABLE','REPRESENTATION_UNAVAILABLE'); continue
                    effective=deepcopy(config); effective['mapper'].update(n_cubes=int(cubes),perc_overlap=float(overlap))
                    space=results[representation]['space']
                    try:
                        mapped=analyze(space,effective)
                        row=perturbation(identifier,space,mapped,'mapper_grid')
                        boot=evaluate_resampling(representation,space,mapped,budget=int(settings['preflight_replicates']),
                            preliminary=True,effective_config=effective,experiment_id=identifier)
                        row['preliminary_qc']='PASS' if boot['qc_pass'] else 'QC_FAILED'
                        row['eligible_for_final']='prespecified_reference_only' # no outcome-driven selection
                    except (ValueError,np.linalg.LinAlgError) as exc:
                        record(identifier,'mapper_grid','NOT_ESTIMABLE',str(exc))
        for i,local in enumerate(settings['local_sensitivity']):
            effective=deepcopy(config); effective['mapper'].update(local)
            try: perturbation(f'T/local_{i}',base_space,analyze(base_space,effective),'mapper_local')
            except ValueError as exc: record(f'T/local_{i}','mapper_local','NOT_ESTIMABLE',str(exc))
    if settings['run_domain_and_family_ablations']:
        for family in legacy.FAMILY_ORDER:
            excluded={c for c in values if families[c]==family}
            try:
                if not excluded: raise ValueError('FAMILY_ABSENT')
                space,_,_=select_and_fit('R0',exclude=excluded)
                perturbation(f'A2/{family}',space,kind='family_ablation',fit=refit('R0',exclude=excluded))
            except (ValueError,np.linalg.LinAlgError) as exc: record(f'A2/{family}','family_ablation','NOT_ESTIMABLE',str(exc))
        for domain in cfg['clinical_domains']['domains']:
            excluded=set(registry.loc[registry.clinical_domain.eq(domain),'feature'])
            try:
                if not excluded: raise ValueError('DOMAIN_ABSENT_OR_UNMAPPED')
                if domain not in CLINICAL: raise ValueError('CONTEXT_DOMAIN_AUDITED_IN_A5')
                eligible_domain_map(registry,sorted(excluded),purpose=cfg['analysis']['mode'])
                space,_,_=select_and_fit('R0',exclude=excluded)
                perturbation(f'A1/{domain}',space,kind='domain_ablation',fit=refit('R0',exclude=excluded))
            except (ValueError,np.linalg.LinAlgError) as exc: record(f'A1/{domain}','domain_ablation','NOT_ESTIMABLE',str(exc))
        for construct in ['unstimulated_whole_saliva','ocular_schirmer','SF36_INSTRUMENT']:
            excluded=set(registry.loc[registry.derivation_group.eq(construct),'feature'])
            try:
                if not excluded.intersection(base_space.features): raise ValueError('CONSTRUCT_ABSENT_FROM_REFERENCE')
                space,_,_=select_and_fit('R0',exclude=excluded)
                perturbation('A6/'+construct,space,kind='construct_ablation',fit=refit('R0',exclude=excluded))
            except (ValueError,np.linalg.LinAlgError) as exc: record('A6/'+construct,'construct_ablation','NOT_ESTIMABLE',str(exc))
        try:
            valid_units=set(manifest.loc[manifest.unit_review_status.isin(['valid','harmonized','approved']),'feature'])
            reduced={c:v for c,v in values.items() if c in valid_units and v.notna().all()}
            space,_,_=select_and_fit('R0',custom_values=reduced)
            perturbation('A3/unit_valid_complete_panel',space,kind='panel_ablation')
        except (ValueError,np.linalg.LinAlgError) as exc: record('A3/unit_valid_complete_panel','panel_ablation','NOT_ESTIMABLE',str(exc))
        diagnostic=set(registry.loc[registry.subdomain.eq('DIAGNOSTIC_CLASSIFICATION'),'feature'])
        if diagnostic:
            try: perturbation('A4/diagnostic_features',select_and_fit('R0',exclude=diagnostic)[0],kind='diagnostic_ablation')
            except ValueError as exc: record('A4/diagnostic_features','diagnostic_ablation','NOT_ESTIMABLE',str(exc))
        else: record('A4/diagnostic_features','diagnostic_ablation','NOT_ESTIMABLE','NO_APPROVED_DIAGNOSTIC_FEATURE_ROLE_MAP')
    if settings['run_negative_controls']:
        for r in range(int(settings['negative_control_replicates'])):
            identifier=f'N1/permutation_{r}'
            try:
                # Permute exact source variables jointly across their encoded levels.
                raw=pd.DataFrame(values); rng=np.random.default_rng(seed+r)
                for source,part in manifest.loc[manifest.feature.isin(raw)].groupby('source_column'):
                    cols=part.feature.tolist(); observed=raw[cols].notna().all(axis=1)
                    raw.loc[observed,cols]=raw.loc[observed,cols].to_numpy()[rng.permutation(int(observed.sum()))]
                space,_,_=select_and_fit('R0',custom_values={c:raw[c] for c in raw})
                negative.append(perturbation(identifier,space,kind='negative_control'))
            except ValueError as exc: negative.append(record(identifier,'negative_control','NOT_ESTIMABLE',str(exc)))
        try:
            date=cfg['validation']['date_column']
            if date in baseline:
                baseline['graph__era']=pd.to_datetime(baseline[date],errors='coerce').dt.year//5*5
            raw=mask_control(base_space.raw,baseline,[*cfg['validation']['confounders'],'graph__era'])
            if len(raw.columns)<2: raise ValueError('NO_VARIABLE_OBSERVATION_PROCESS')
            rep=legacy.fit_embedding(raw,{c:'CLINICAL' for c in raw},'S2',config)
            space=Space(raw,rep.embedding[:,:2],rep.embedding,pairwise_distances(rep.embedding),pd.DataFrame(),'missingness_admin',list(raw),rep)
            negative.append(perturbation('N2/missingness_protocol_era',space,kind='legacy_mixed_observation_admin_control'))
        except ValueError as exc: negative.append(record('N2/missingness_protocol_era','negative_control','NOT_ESTIMABLE',str(exc)))
        date=cfg['validation']['date_column']
        admin,_,admin_sources=administrative_frame(baseline,cfg['validation']['confounders'],date)
        for item in admin_sources.loc[admin_sources.status.eq('NOT_ESTIMABLE')].to_dict('records'):
            negative.append(record('N3/'+item['requested'],'administrative_stratum','NOT_ESTIMABLE',item['reason']))
        for column in admin:
            strata=administrative_strata(admin[column],column,int(settings['minimum_stratum_patients']))
            for level,idx in strata.groupby(strata,dropna=True).groups.items():
                # Logs use a level index, never a patient identifier or raw protocol value.
                identifier=f'N3/{column}/stratum_{len(negative)}'
                if len(idx)<int(settings['minimum_stratum_patients']):
                    negative.append(record(identifier,'administrative_stratum','NOT_ESTIMABLE','INSUFFICIENT_STRATUM_PATIENTS',n_stratum=len(idx))); continue
                try:
                    positions=np.array(list(idx),int); space=refit('R0')(positions); mapped=analyze(space,config)
                    negative.append(record(identifier,'administrative_stratum',n_stratum=len(idx),
                        mapper_coverage=float(mapped['covered'].mean()),mean_pairwise_distance=float(space.distance[np.triu_indices(len(idx),1)].mean())))
                except ValueError as exc: negative.append(record(identifier,'administrative_stratum','NOT_ESTIMABLE',str(exc)))
        record('N4/selection_knowledge_date','administrative_control','NOT_ESTIMABLE','NO_APPROVED_BASELINE_SELECTION_KNOWLEDGE_DATE_CONTRAST')
    observation=run_observation_diagnostics(base_space.raw,registry,baseline,base_map,config,settings,
        cfg['validation'],run_dir,perturbation,record,refit('R0'))
    for identifier,item in results.items():
        if identifier=='R0': continue
        comparison=compare_coverage(base_map,item['mapped'],int(settings.get('minimum_comparison_patients',10)))
        comparisons.append({'experiment_id':identifier,**comparison})
    save_table(run_dir/'02_graph_comparison_denominators.csv',comparisons)
    save_table(run_dir/'02_graph_domain_ablation_summary.csv',[r for r in experiments if r['experiment_id'].startswith(('A1/','A2/','A6/'))])
    save_table(run_dir/'02_graph_context_exclusion_comparison.csv',[r for r in experiments if r['experiment_id']=='A5/context_excluded'])
    registry['included_R2']=registry.feature.isin(results['R2']['space'].features if 'R2' in results else [])
    registry.loc[registry.exploratory_technical_eligible & ~registry.included_R2,'exclusion_reason']='NOT_SELECTED_IN_R2_REFIT_OR_REPRESENTATION_NOT_ESTIMABLE'
    save_table(run_dir/'02_graph_clinical_domain_registry.csv',registry)
    save_table(run_dir/'02_graph_clinical_feature_mapping_audit.csv',registry)
    unmapped=registry.loc[registry.selected_in_primary & registry.clinical_domain.eq('UNMAPPED')].copy()
    save_table(run_dir/'02_graph_unmapped_selected_features.csv',unmapped)
    save_table(run_dir/'02_graph_domain_contributions.csv',[r for r in weights if r['representation_id']=='R2'])
    save_table(run_dir/'02_graph_clinical_domain_map_summary.csv',[
        dict(clinical_domain=domain,n_original_sources=part.source_column.nunique(),n_encoded_features=len(part),
            n_unique_constructs=part.derivation_group.nunique(),n_approved_sources=part.loc[part.clinically_approved,'source_column'].nunique(),
            n_technical_eligible_sources=part.loc[part.exploratory_technical_eligible,'source_column'].nunique(),
            n_selected_R0=int(part.selected_in_primary.sum()),n_selected_R2=int(part.included_R2.sum()))
        for domain,part in registry.groupby('clinical_domain')])
    if settings['run_structural_comparators']:
        comparator_rows=comparators(base_space,base_map,settings,seed,refit('R0'))
        for row in comparator_rows: record(f'C/{row["method"]}/k{row["k"]}','structural_comparator',**{k:v for k,v in row.items() if k not in {'method','k','n_patients'}})
    # Global one-to-one alignment on a common, explicitly frozen patient universe.
    alignments=[]
    for identifier,item in results.items():
        if identifier=='R0': continue
        matched,scores=match_sets(base_map['community_sets'],item['mapped']['community_sets'])
        alignments.extend({'reference_representation_id':'R0','representation_id':identifier,**row,'n_patients_common':n} for row in matched)
    primary_rows=[r for r in stability if r['representation_id']=='R0' and r['method']=='repeated_subsampling' and r['phase']=='final']
    stable_count=sum(r['stable'] and base_map['supported'][r['community_id']] for r in primary_rows)
    primary_qc=qc_by_id.get('R0',False)
    limitations=sorted({r['reason'] for r in experiments if r['status']=='NOT_ESTIMABLE'})
    limitations.extend('FAILED_EXPERIMENT/'+r['experiment_id'] for r in experiments if r['status']=='FAILED')
    if len(unmapped): limitations.append('SELECTED_UNMAPPED_FEATURES/'+str(len(unmapped)))
    if 'R2' in results and not registry.loc[registry.included_R2,'clinically_approved'].all(): limitations.append('R2_EXPLORATORY_PROVISIONAL_PENDING_CLINICAL_APPROVAL')
    if registry.loc[registry.included_R2,'technical_exceptions'].ne('').any(): limitations.append('R2_TECHNICAL_EXCEPTIONS_RECORDED_IN_MAPPING_AUDIT')
    if not observation['reason_audit_complete']: limitations.append('UNRESOLVED_COVERAGE_REASONS')
    if float(base_map['covered'].mean())<float(settings['minimum_mapper_coverage']):
        limitations.append('LOW_PRIMARY_MAPPER_COVERAGE')
    fragmentation=len(base_map['components'])/max(1,len(base_map['nodes']))
    if fragmentation>float(settings['maximum_fragmentation']):
        limitations.append('HIGH_PRIMARY_FRAGMENTATION')
    null_metrics=[r for r in negative if r.get('status')=='COMPLETE' and 'n_communities' in r]
    artifact_risk=any(r.get('ari_primary',0)>=.8 or (r['experiment_id'].startswith('N1') and r.get('n_communities',0)>=len(base_map['community_sets']) and r.get('mapper_coverage',0)>=float(base_map['covered'].mean())) for r in null_metrics)
    release_status,reasons=release_decision(primary_built=True,primary_qc=primary_qc,stable_count=stable_count,
        mode=cfg['analysis']['mode'],limitations=limitations,artifact_risk=artifact_risk)
    if release_status=='QC_FAILED': raise ValueError('PRIMARY_REFERENCE_QC_FAILED')
    eligible=[i for i in ['R0','R2','R1','R3','R4'] if i in results and qc_by_id.get(i,False)]
    execution_status='PARTIAL' if any(r['status']=='FAILED' for r in experiments) else 'COMPLETE_WITH_LIMITATIONS' if limitations else 'COMPLETE'
    # Freeze raw observed baseline sources and encoded values, without refitting in 03.
    snapshot=baseline.copy()
    for c,s in values.items(): snapshot[c]=s.to_numpy()
    save_table(run_dir/'02_graph_baseline_raw.csv',snapshot)
    save_table(run_dir/'02_graph_feature_sets.csv',feature_sets)
    role_rows=[]
    for row in feature_sets:
        role_rows.append({**row,'validation_role':'defining' if row['included'] else 'unavailable',
            'feature_policy':cfg['analysis']['feature_policy']})
    save_table(run_dir/'02_graph_feature_role_manifest.csv',role_rows)
    save_table(run_dir/'02_graph_feature_selection_audit.csv',pd.concat(selection,ignore_index=True))
    save_table(run_dir/'02_graph_domain_weight_audit.csv',weights)
    save_table(run_dir/'02_graph_domain_coverage.csv',coverage)
    save_table(run_dir/'02_graph_patient_membership.csv',member_rows)
    save_table(run_dir/'02_graph_mapper_nodes.csv',nodes,['representation_id','node_id','patient_id','node_support','connected_component','community_id'])
    save_table(run_dir/'02_graph_mapper_edges.csv',edges,['representation_id','source_node','target_node','weight','weight_type'])
    save_table(run_dir/'02_graph_community_dictionary.csv',dictionary,['representation_id','community_id','n_patients_node_union','supported','structural_stability_status','object_type','branch_algorithm'])
    save_table(run_dir/'02_graph_resampling_replications.csv',attempts)
    stable_frame=save_table(run_dir/'02_graph_community_stability.csv',stability,['representation_id','community_id','method','phase','n_requested','n_success','n_evaluable','stability','ci_low','ci_high','qc_status','stable'])
    save_table(run_dir/'02_graph_resampling_matching.csv',matches)
    save_table(run_dir/'02_graph_refit_membership.csv',refit_membership,['representation_id','replicate_id','seed','patient_id','candidate_community_id','reference_community_id','matching_jaccard'])
    save_table(run_dir/'02_graph_coassignment.csv',pair_rows)
    save_table(run_dir/'02_graph_cross_representation_alignment.csv',alignments)
    save_table(run_dir/'02_graph_negative_controls.csv',negative)
    save_table(run_dir/'02_graph_method_comparators.csv',comparator_rows)
    save_table(run_dir/'02_graph_experiment_registry.csv',experiments)
    legacy_observation=[]
    for i,p in enumerate(baseline.patient_id):
        for family in legacy.FAMILY_ORDER:
            cols=[c for c in base_space.raw if families[c]==family]
            legacy_observation.append({'patient_id':p,'representation_id':'R0','family':family,'n_features':len(cols),
                'n_observed':int(base_space.raw.iloc[i][cols].notna().sum()),'mapper_covered':bool(base_map['covered'][i])})
    save_table(run_dir/'02_graph_coverage_and_missingness.csv',legacy_observation)
    decision=[{'gate':'primary_refit_qc','passed':primary_qc},{'gate':'primary_coverage','passed':base_map['covered'].mean()>=settings['minimum_mapper_coverage']},
        {'gate':'stable_supported_communities','passed':stable_count>0},{'gate':'administrative_artifact_caution','passed':not artifact_risk}]
    save_table(run_dir/'02_graph_decision_table.csv',decision)
    # Preserve historical names alongside the normalized release tables.
    summary=[]
    for identifier,item in results.items():
        row={'scenario':identifier,'n_baseline_patients':n,'n_final_features':len(item['space'].raw.columns),
             'mapper_coverage':float(item['mapped']['covered'].mean()),'n_communities':len(item['mapped']['community_sets']),
             'n_missing_before_imputation':int(item['space'].raw.isna().to_numpy().sum()),
             'analysis_mode':cfg['analysis']['mode'],'feature_policy':cfg['analysis']['feature_policy']}
        summary.append(row)
        if item['space'].legacy_rep is not None:
            save_table(run_dir/f'02_graph_{identifier}_pca_variance.csv',legacy.pca_table(item['space'].legacy_rep))
    save_table(run_dir/'02_graph_representation_sensitivity_summary.csv',summary)
    save_table(run_dir/'02_graph_scenario_patient_membership.csv',pd.concat(legacy_members,ignore_index=True))
    save_table(run_dir/'02_graph_scenario_feature_sets.csv',feature_sets)
    old_stability=stable_frame.loc[(stable_frame.method=='repeated_subsampling') & stable_frame.phase.eq('final')].copy()
    old_stability['scenario']=old_stability.representation_id; old_stability['community']=old_stability.community_id
    old_stability['bootstrap_stability']=old_stability.stability; old_stability['stability_status']=np.where(old_stability.stable,'stable','provisional')
    alias=[]
    for old,new in [('S3','R0'),('S4','R1')]:
        part=old_stability.loc[old_stability.scenario.eq(new)].copy(); part['scenario']=old; alias.append(part)
    save_table(run_dir/'02_graph_scenario_community_stability.csv',pd.concat([old_stability,*alias],ignore_index=True))
    if 'S4' in results: save_table(run_dir/'02_graph_S4_balance_weights.csv',results['S4']['space'].weights)
    qc={'release_status':release_status,'primary_qc':'PASS','execution_status':execution_status,
        'n_stable_supported_communities':int(stable_count),'artifact_risk':artifact_risk,'qc_reasons':reasons,
        'limitations':limitations,'clinical_validation':governance['clinical_validation']}
    write_json(run_dir/'02_graph_release_qc.json',qc)
    embedding_rows=[]; distance_rows=[]
    for identifier,item in results.items():
        space=item['space']
        for i,patient in enumerate(baseline.patient_id):
            row={'representation_id':identifier,'patient_id':patient,'lens_1':float(space.lens[i,0]),
                 'lens_2':float(space.lens[i,1]),'method':space.method}
            if not space.method.startswith('Gower'):
                row.update({f'component_{j+1}':float(value) for j,value in enumerate(space.cluster[i])})
            embedding_rows.append(row)
        for a in range(n):
            for b in range(a+1,n):
                distance_rows.append({'representation_id':identifier,'patient_a':baseline.patient_id.iloc[a],
                    'patient_b':baseline.patient_id.iloc[b],'distance':float(space.distance[a,b]),'method':space.method})
    save_table(run_dir/'02_graph_patient_embedding.csv',embedding_rows)
    save_table(run_dir/'02_graph_patient_distances.csv',distance_rows)
    figures({k:v for k,v in results.items() if k.startswith('R')},stable_frame,run_dir/'figures',registry,coverage,alignments)
    with (run_dir/'logs/02_graph_events.jsonl').open('w') as stream:
        for row in logs: stream.write(json.dumps(row,default=str)+'\n')
    figure_inventory=diagnostic_figures(observation,comparisons,experiments,run_dir/'figures',registry,int(settings.get('minimum_publication_cell',5)))
    inventoried={r['artifact'] for r in figure_inventory}
    figure_inventory.extend(dict(artifact=p.relative_to(run_dir).as_posix(),status='COMPLETE',reason='') for p in sorted((run_dir/'figures').glob('*')) if p.relative_to(run_dir).as_posix() not in inventoried)
    save_table(run_dir/'02_graph_figure_inventory.csv',figure_inventory)
    results_summary(run_dir/'02_graph_results_summary.md',results,perturbations,experiments,comparisons,registry,observation,stable_frame,qc)
    save_table(run_dir/'02_graph_privacy_inventory.csv',privacy_inventory(run_dir))
    relative={'membership':'02_graph_patient_membership.csv','feature_roles':'02_graph_feature_role_manifest.csv',
        'baseline_raw':'02_graph_baseline_raw.csv','feature_sets':'02_graph_feature_sets.csv','stability':'02_graph_community_stability.csv',
        'experiment_registry':'02_graph_experiment_registry.csv','nodes':'02_graph_mapper_nodes.csv','edges':'02_graph_mapper_edges.csv',
        'community_dictionary':'02_graph_community_dictionary.csv','clinical_registry':'02_graph_clinical_domain_registry.csv',
        'mapping_audit':'02_graph_clinical_feature_mapping_audit.csv','comparison_denominators':'02_graph_comparison_denominators.csv',
        'missingness_controls':'02_graph_missingness_control_comparisons.csv','coverage_reasons':'02_graph_coverage_reason_summary.csv',
        'refit_membership':'02_graph_refit_membership.csv','release_qc':'02_graph_release_qc.json','alignment':'02_graph_cross_representation_alignment.csv'}
    pharma=sources['pharma_candidates']; summary_path=pharma.with_name('00_pharma_profile_summary.json')
    source_info={role:{'path':str(p),'sha256':digest(p)} for role,p in sources.items()}
    source_info['pharma_summary']={'path':str(summary_path),'sha256':digest(summary_path)}
    profile_path=pharma.with_name(json.loads(summary_path.read_text()).get('profile_output_name','00_pharma_lab_profile.csv'))
    extra_sources={'pharma_profile':profile_path,'pharma_selection_audit':pharma.with_name('00_pharma_graph_lab_selection_audit.csv'),
                   's4_lab_group_map':config_path(cfg['_path'],cfg['paths']['lab_group_map']),
                   'baseline_metrics':config_path(cfg['_path'],cfg['paths']['baseline_metrics'])}
    if cfg['analysis']['mode']=='clinical_release':
        extra_sources['clinical_release']=config_path(cfg['_path'],cfg['paths']['clinical_release'])
    for role,source in extra_sources.items():
        if source.is_file(): source_info[role]={'path':str(source),'sha256':digest(source)}
    manifest_release={'schema_version':SCHEMA,'finalized':True,'run_id':run_dir.name,'git_commit':git_commit(),
        'study_mode':cfg['analysis']['mode'],'feature_policy':cfg['analysis']['feature_policy'],'data_sha256':digest(sources['integrated']),
        'pharma_profile_sha256':digest(profile_path),'clinical_release_id':(legacy.CANON.load_config(config_path(cfg['_path'],cfg['paths']['clinical_release'])).get('release_id') if cfg['analysis']['mode']=='clinical_release' else None),
        'config_sha256':cfg['_config_sha256'],'config_hashes':cfg['_hashes'],
        'features_sha256':digest(run_dir/'02_graph_feature_sets.csv'),'baseline_definition_id':'Step10_is_clinical_baseline_clinical_episode_curated_v2',
        'n_patients':n,'n_features':len(base_space.raw.columns),'n_covered':int(base_map['covered'].sum()),
        'method_list':sorted({item['space'].method for item in results.values()}),'primary_reference_id':'R0',
        'eligible_representation_ids':eligible,'release_status':release_status,'execution_status':execution_status,
        'qc_reasons':reasons,'created_at_utc':utc_now(),'started_at_utc':started,'relative_paths':relative,
        'clinical_mapping':{'mapping_release_id':cfg['clinical_domains'].get('mapping_release_id'),
            'review_status':cfg['clinical_domains']['mapping_review_status'],
            'technical_exploratory_status':'not_estimable' if 'R2' not in results else 'eligible' if not len(unmapped) and registry.loc[registry.selected_in_primary & registry.clinical_domain.isin(CLINICAL),'exploratory_technical_eligible'].all() else 'partial',
            'selected_unmapped_count':len(unmapped)},
        'representations':{'R2':next((dict(status=r['status'],evidence_level=r.get('evidence_level','NOT_ESTIMABLE'),reason=r['reason']) for r in experiments if r['experiment_id']=='R2'),{'status':'NOT_ESTIMABLE','evidence_level':'NOT_ESTIMABLE'})},
        'missingness':{'masks_source':'baseline_R0_raw_selected_features','legacy_N2_mixed':True,
            'standalone_controls':['N2a','N2b','N2c','N2d'],'diagnostic_complete':all(any(r['experiment_id'].startswith(i+'/') for r in experiments) for i in ['N2a','N2b','N2c','N2d','M1'])},
        'coverage':{'n_total':n,'n_covered_R0':int(base_map['covered'].sum()),'n_uncovered_R0':int((~base_map['covered']).sum()),'reason_audit_complete':bool(observation['reason_audit_complete'])},
        'artifact_inventory':privacy_inventory(run_dir),
        'sha256_by_artifact':artifact_inventory(run_dir),'source_inputs':source_info,'versions':version_info(),
        'protocol_id':settings['protocol_id'],'random_seed':seed,'clinical_validation':governance['clinical_validation']}
    manifest_path=run_dir/'02_graph_release_manifest.json'
    write_json(manifest_path,manifest_release)
    publish_pointer(root,manifest_path)
    print(f'Graph02 {release_status}; {execution_status}; release: {manifest_path}')
    return {'run_dir':run_dir,'manifest':manifest_path,'execution_status':execution_status}


def main(argv=None):
    parser=base_parser(__doc__)
    args=parser.parse_args(argv)
    return run(load_pipeline(args.pipeline_config),root=args.output_dir,run_id=args.run_id,
               preflight_only=args.preflight_only or args.dry_run)

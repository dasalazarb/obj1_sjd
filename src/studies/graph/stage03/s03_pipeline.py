"""Clinical interpretation from a verified release; never reconstruct discovery."""
from pathlib import Path
import json
import pandas as pd
from .._pipeline import (load_pipeline,output_root,new_run,save_table,write_json,digest,utc_now,
                        artifact_inventory,version_info,base_parser)
from .s03_release_reader import read_release
from .s03_construct_validation import feature_roles,assert_independent,anchor_lineage_audit
from .s03_profiles import profile_tables,clinical_domains,uncovered_profiles,pop_denominator_audit,clinical_marker_summaries
from .s03_incremental_value import pop_crosswalk,incremental_models
from .s03_longitudinal import longitudinal,pop_transitions,comparable_longitudinal_summary
from .s03_episode_audit import resolve_all_endpoints, VERSION
from .s03_rare_events import rare_event_table
from .s03_reporting import figures,characterization_figures,safe_characterization_panels
from .s03_figure_qc import figure_qc,validate_figure_policy
from .s03_characterization import statistical_role_gates,characterization_tables,characterization_qc
from .s03_uncertainty import discovery_sensitivity
from .s03_method_context import method_context,missingness_interpretation,representation_evidence


def run(cfg, *, graph02_release=None,root=None,run_id=None,preflight_only=False):
    validate_figure_policy(cfg['validation'])
    manifest,path,frames=read_release(cfg,graph02_release)
    pointer=output_root(cfg,'02')/'latest_completed_02.json'
    pointer_sha=digest(pointer) if pointer.exists() else None
    root=Path(root) if root else output_root(cfg,'03')
    run_dir=new_run(root,run_id)
    provenance={'graph02_run_id':manifest['run_id'],'graph02_release_sha256':digest(path),
                'baseline_definition_id':manifest['baseline_definition_id'],'analysis_mode':manifest['study_mode'],
                'feature_policy':manifest['feature_policy']}
    write_json(run_dir/'03_graph_release_input_qc.json',{**provenance,'status':'PASS','read_mode':'frozen_artifacts_no_refit',
        'missingness_sensitivity':frames['method_context_qc'],'eligible_representation_ids':manifest['eligible_representation_ids']})
    if preflight_only: return {'run_dir':run_dir,'execution_status':'PREFLIGHT_ONLY','manifest':None}
    try:
        baseline=frames['baseline_raw']; membership=frames['membership']; validation=cfg['validation']
        roles=feature_roles(frames['feature_sets'],baseline,validation,manifest['eligible_representation_ids'],frames.get('clinical_registry',pd.DataFrame()))
        roles=statistical_role_gates(roles,baseline,membership,manifest,validation)
        lineage=anchor_lineage_audit(roles,validation)
        tables=profile_tables(baseline,membership,roles,manifest,validation,int(manifest['random_seed']))
        characterization=characterization_tables(baseline,membership,roles,manifest,validation,tables['anchors'])
        markers=clinical_marker_summaries(baseline,membership,roles,manifest,validation)
        minimum_cell=int(cfg['experiments'].get('minimum_publication_cell',5))
        domains=clinical_domains(baseline,membership,roles,frames.get('clinical_registry',pd.DataFrame()),manifest,minimum_cell)
        uncovered=uncovered_profiles(baseline,membership,roles,manifest,minimum_cell)
        context=method_context(frames['method_context_inputs'],frames['method_context_qc'],manifest)
        missingness=missingness_interpretation(frames['method_context_inputs'],tables['missingness'])
        evidence=representation_evidence(manifest,membership,tables['flow'],digest(path))
        assert_independent(tables['anchors'])
        pop,concordance=pop_crosswalk(baseline,membership,manifest['eligible_representation_ids'],validation)
        incremental=incremental_models(baseline,membership,roles,manifest,validation)
        master=pd.read_parquet(manifest['source_inputs']['integrated']['path'])
        resolutions=resolve_all_endpoints(master,baseline,validation)
        temporal_tables={key:pd.concat([r.audits[key] for r in resolutions.values()],ignore_index=True)
            for key in ['temporal_source','collision_summary','resolution','unresolved_reasons']}
        pop_denominators=pop_denominator_audit(baseline,membership,manifest,validation)
        # Crosswalk reconciliation is a runtime invariant, not a reference-run constant.
        for row in pop_denominators.loc[pop_denominators.population_universe.eq('hard_supported')].itertuples():
            actual=pop.loc[pop.representation_id.eq(row.representation_id),'n_patients'].sum() if 'n_patients' in pop else 0
            if actual!=row.n_pop1+row.n_pop2+row.n_pop3: raise ValueError('POP_CROSSWALK_DENOMINATOR_MISMATCH')
        eligibility,models,trajectories=longitudinal(master,baseline,membership,manifest,validation,resolutions=resolutions)
        comparison=comparable_longitudinal_summary(master,baseline,membership,manifest,validation,resolutions=resolutions)
        transitions,transition_eligibility=pop_transitions(master,baseline,membership,manifest,validation,resolutions=resolutions,return_eligibility=True)
        rare=rare_event_table(baseline,validation)
        uncertainty=discovery_sensitivity(frames.get('refit_membership',pd.DataFrame()),baseline,roles,tables['anchors'],cfg['experiments'],validation)
        statuses=[]
        for name,frame in [('independent_anchors',tables['anchors']),('incremental_value',incremental),('longitudinal',eligibility),('rare_events',rare),('pop_transitions',transitions),('discovery_refit_sensitivity',uncertainty)]:
            if frame.empty: statuses.append({'analysis':name,'status':'NOT_ESTIMABLE','reason':'NO_ELIGIBLE_HYPOTHESES'})
            else:
                for row in frame.to_dict('records'):
                    statuses.append({'analysis':name,**{k:row[k] for k in ['representation_id','source_column','endpoint','event','status','reason','n_patients','n_visits','temporal_resolution_status','descriptive_status','model_status','date_semantics_review_status','temporal_exclusion_reasons','tier','independence_status','inferential_eligible'] if k in row}})
        for role in roles.loc[roles.is_prespecified_anchor].to_dict('records'):
            statuses.append({'analysis':'characterization_independence_gate',**{k:role[k] for k in ['representation_id','source_column','tier','independence_status','inferential_eligible']},
                'status':'COMPLETE' if role['inferential_eligible'] else 'NOT_ESTIMABLE','reason':role['blocking_reasons']})
        statuses.extend([{'analysis':'baseline_characterization','status':'COMPLETE' if not tables['baseline'].empty else 'NOT_ESTIMABLE','reason':'' if not tables['baseline'].empty else 'NO_SUPPORTED_COMMUNITIES'},
            {'analysis':'pop_crosswalk','status':'COMPLETE' if not pop.empty and 'n_patients' in pop else 'NOT_ESTIMABLE','reason':'' if not pop.empty and 'n_patients' in pop else 'NO_SUPPORTED_POP_PATIENTS'},
            {'analysis':'external_projection','status':'NOT_ESTIMABLE','reason':validation['projection']['reason']},
            {'analysis':'confirmatory_inference','status':'NOT_ESTIMABLE','reason':'Same-cohort exploratory discovery; no independent confirmatory cohort'}])
        for row in evidence.to_dict('records'):
            for reason in filter(None,row['limitation_reason'].split('|')):
                statuses.append({'analysis':'representation_evidence','representation_id':row['representation_id'],
                    'status':'LIMITATION','reason':reason})
        for row in context.loc[context.source_table.isin(['method_registry','unavailable'])].to_dict('records'):
            if row['status']!='COMPLETE':
                statuses.append({'analysis':'method_sensitivity_context','representation_id':row['experiment_id'],
                    'status':row['status'],'reason':'MISSINGNESS_METHOD_NONESTIMABLE' if row['status']=='NOT_ESTIMABLE' else row.get('reason','SENSITIVITY_SOURCE_UNAVAILABLE'),
                    'source_reason':row.get('method_reason',row.get('reason',''))})
        for row in statuses:
            reason=row.get('reason','')
            row['reason_category']=('UNVERIFIED_DATE' if any(t in reason for t in ['UNVERIFIED','INDEX_DATE_UNAVAILABLE']) else
                'UNRESOLVED_EPISODE_CONFLICT' if any(t in reason for t in ['CONFLICT','AMBIGUOUS','SAME_DAY']) else
                'TOPOLOGY_NOT_STABLE' if 'TOPOLOGY' in reason else
                'INSUFFICIENT_COVARIATES' if any(t in reason for t in ['COVARIATE','ADJUSTMENT']) else
                'MODEL_NONCONVERGENCE' if any(t in reason for t in ['NONCONVERGENCE','SINGULAR','RANK_DEFICIENCY','NONFINITE']) else
                'INSUFFICIENT_LONGITUDINAL_SUPPORT' if row['analysis']=='longitudinal' and 'INSUFFICIENT' in reason else reason)
        anchor_names={a['source_column'] for a in validation['anchors']}
        for role in roles.loc[roles.source_column.isin(anchor_names) & roles.validation_role.isin(['defining','derived_from_defining'])].itertuples():
            statuses.append({'analysis':'independent_anchors','representation_id':role.representation_id,'source_column':role.source_column,'status':'NOT_ESTIMABLE','reason':'CIRCULAR_DEFINING_OR_DERIVED_CONSTRUCT'})
        table_map={'03_graph_clinical_marker_summary.csv':markers,
            '03_graph_temporal_source_audit.csv':temporal_tables['temporal_source'],
            '03_graph_episode_collision_summary.csv':temporal_tables['collision_summary'],
            '03_graph_episode_resolution_audit.csv':temporal_tables['resolution'],
            '03_graph_longitudinal_unresolved_reasons.csv':temporal_tables['unresolved_reasons'],
            '03_graph_longitudinal_denominator_audit.csv':eligibility.copy(),
            '03_graph_pop_denominator_audit.csv':pop_denominators,
            '03_graph_pop_transition_eligibility.csv':transition_eligibility,
            '03_graph_representation_evidence.csv':evidence,'03_graph_clinical_domains_by_community.csv':domains,
            '03_graph_anchor_lineage_audit.csv':lineage,'03_graph_method_sensitivity_context.csv':context,
            '03_graph_missingness_interpretation.csv':missingness,'03_graph_uncovered_baseline_audit.csv':uncovered,
            '03_graph_patient_population_flow.csv':tables['flow'],'03_graph_community_baseline_table.csv':tables['baseline'],
            '03_graph_clinical_feature_roles.csv':roles,'03_graph_clinical_signatures.csv':tables['signatures'],
            '03_graph_independent_anchor_validation.csv':tables['anchors'],'03_graph_pop_crosswalk.csv':pop,
            '03_graph_pop_concordance.csv':concordance,'03_graph_pop_transitions.csv':transitions,'03_graph_incremental_value.csv':incremental,
            '03_graph_longitudinal_representation_comparison.csv':comparison,'03_graph_longitudinal_eligibility.csv':eligibility,'03_graph_longitudinal_models.csv':models,
            '03_graph_community_trajectory_summary.csv':trajectories,'03_graph_missingness_by_community.csv':tables['missingness'],
            '03_graph_rare_events.csv':rare,'03_graph_anchor_discovery_sensitivity.csv':uncertainty,'03_graph_validation_status.csv':pd.DataFrame(statuses),
            '03_graph_cross_representation_alignment.csv':frames.get('alignment',pd.DataFrame())}
        table_map.update({f'03_graph_characterization_{name}.csv':frame for name,frame in characterization.items()})
        frozen_unchanged=(digest(pointer) if pointer.exists() else None)==pointer_sha and digest(path)==provenance['graph02_release_sha256'] and all(
            digest(path.parent / relative)==manifest['sha256_by_artifact'][relative]
            for relative in manifest['relative_paths'].values() if relative in manifest['sha256_by_artifact'])
        tier_qc=characterization_qc(characterization,baseline,membership,provenance,frozen_unchanged)
        write_json(run_dir/'03_graph_characterization_qc.json',tier_qc)
        for name,frame in table_map.items():
            frame=frame.copy()
            for key,value in provenance.items(): frame[key]=value
            save_table(run_dir/name,frame)
        safe_matrix=safe_characterization_panels(characterization['clinical_matrix'],minimum_cell)
        safe_keys=safe_matrix[['representation_id','source_column']].drop_duplicates()
        safe_markers=markers.merge(safe_keys,on=['representation_id','source_column'],how='inner') if not markers.empty else markers
        safe_domains=domains.merge(safe_keys,on=['representation_id','source_column'],how='inner') if 'source_column' in domains else domains
        safe_anchors=tables['anchors'].merge(safe_keys,on=['representation_id','source_column'],how='inner') if not tables['anchors'].empty else tables['anchors']
        figure_sources=figures(safe_anchors,trajectories,pop,run_dir/'figures',flow=tables['flow'],domains=safe_domains,
            context=context,missingness=missingness,minimum_cell=minimum_cell,primary_reference_id=manifest['primary_reference_id'],
            baseline_table=safe_markers,marker_config=validation.get('clinical_marker_panels',[]),
            raw_matrix=characterization['clinical_matrix'],roles=roles)
        presentation=characterization_figures(characterization['clinical_matrix'],roles,run_dir/'figures',validation,minimum_cell,provenance=provenance)
        save_table(run_dir/'03_graph_characterization_N2_source_selection.csv',presentation['selection'])
        figure_sources.update(presentation['sources'])
        presentation_qc=figure_qc(run_dir/'figures',figure_sources,{**provenance,'graph03_run_id':run_dir.name},protected_matrix=safe_matrix)
        write_json(run_dir/'03_graph_figure_qc.json',presentation_qc)
        if presentation_qc['status']=='FAILED': raise ValueError('GRAPH03_FIGURE_QC_FAILED')
        (run_dir/'figures/README.md').write_text('Aggregate internal figures and source CSVs. Publication review is required; no automatic public export. Small cells are suppressed using the configured publication minimum.\n')
        limited=any(row['status'] not in {'COMPLETE','DESCRIPTIVE'} for row in statuses) or presentation_qc['status']!='PASS'
        status='COMPLETE_WITH_LIMITATIONS' if limited else 'COMPLETE'
        (run_dir/'03_graph_results_summary.md').write_text(f'# Graph03 exploratory interpretation\n\nEXPLORATORY: no confirmatory clinical or predictive claims.\n\nFrozen Graph02 run: {manifest["run_id"]}; release SHA: {digest(path)}.\n\nPatients: {len(baseline)}; execution: {status}. Raw observed baseline values only; no imputation or discovery refit.\n\nCommunity and time effects are conditional on estimated frozen labels; intervals do not incorporate discovery uncertainty. See validation status for every unavailable or circular hypothesis.\n\nR0 is the frozen primary reference; R2 clinical comparison retains the release mapping and evidence status; other eligible representations are technical comparators.\n\n'
            + evidence[['representation_id','analysis_role','clinical_mapping_status','evidence_level','limitation_reason']].to_csv(index=False)
            + '\nFrozen method status inventory:\n'
            + context.loc[context.source_table.isin(['method_registry','unavailable'])].reindex(columns=['experiment_id','status','method_reason','n_patients','n_common']).to_csv(index=False)
            + '\nMethod sensitivity states (M/I/D/DG) preserve original denominators. DG is a sparse graph, not Mapper; cross-algorithm ARI does not establish equivalence. None of these methods defines clinical memberships.\n\nMissingness is descriptive: encoded cells and source observation units use different denominators. MCAR/MAR/MNAR, causality and latent disease progression are not established. No method or mask weight is selected using clinical associations.\n\nInference requires the existing stability and support gates; descriptive provisional profiles do not relax those gates. In-sample delta R2 is not external predictive utility. All artifacts remain INTERNAL_RESTRICTED pending publication review.\n')
        n_descriptive=int(eligibility.descriptive_status.isin(['DESCRIPTIVE','DESCRIPTIVE_PROVISIONAL']).sum())
        n_models=int(eligibility.status.eq('COMPLETE').sum())
        n_unresolved=int(temporal_tables['resolution'].n_unresolved_groups.sum())
        with (run_dir/'03_graph_results_summary.md').open('a') as stream:
            stream.write(f'\nCharacterization table QC: {tier_qc["status"]}. Figure QC: {presentation_qc["status"]}. '
                f'R0/R2 clinical figures displayed: {presentation_qc["n_r0_r2_clinical_figures_displayed"]}.\n\n')
            stream.write('COMPLETE_WITH_LIMITATIONS does not imply clinically shareable figures. '
                'Disclosure withholding is a presentation restriction, not absent clinical evidence or scientific non-estimability. '
                'Candidate independence review and display status are separate. See 03_graph_figure_qc.json and figures/03_graph_figure_status.csv.\n\n')
            stream.write(presentation['inventory'][['figure','display_status','inference_status','reason']].to_csv(index=False))
            stream.write('\nN2 selections and documented N1-to-N2 lineage (metadata only):\n\n')
            stream.write(presentation['selection'].to_csv(index=False))
            stream.write('\nSource characterization levels: N1 explains internal structure; N2 is related descriptive evidence; N3 candidates are not validated. Only documented reviewed N3 sources may enter exploratory conditional inference after statistical gates. Pending review is NOT_ESTIMABLE / INDEPENDENCE_REVIEW_PENDING.\n\n')
            stream.write(characterization['tier_counts'].to_csv(index=False))
            stream.write('\nSee characterization_independence_audit.csv and characterization_qc.json for exact sources requiring clinical/lineage/baseline review. Longitudinal endpoint approvals are separate and unchanged.\n')
            stream.write(f'\nTemporal resolution: {VERSION}. Endpoint analyses: {len(eligibility)}; descriptive: {n_descriptive}; estimable models: {n_models}; unresolved endpoint/source groups: {n_unresolved}.\n\n')
            stream.write('Measurement dates require endpoint-specific reviewed evidence; a clinical spine anchor alone is insufficient. Unresolved units are excluded without inventing dates or outcomes. Descriptions use verified visits; models additionally require repeated observations, stability, support and all prespecified covariates. Frozen stability labels are not clinical bootstrap validation.\n\n')
            stream.write(temporal_tables['temporal_source'][['endpoint','date_field_used','date_semantics_review_status','status','reason']].to_csv(index=False))
            stream.write('\nPop counts below separate classified, Unclassified, missing and invalid labels in each universe. Community absence is a separate quantity. First-followup transitions count patients once; all-followup cell patient counts are nonadditive.\n\n')
            stream.write(pop_denominators.to_csv(index=False))
        with (run_dir/'logs/03_graph_events.jsonl').open('w') as stream:
            for row in statuses: stream.write(json.dumps({**provenance,**row},default=str)+'\n')
        inventory=[dict(artifact=p.relative_to(run_dir).as_posix(),sha256=digest(p),
            contains_patient_identifiers=False,privacy_class='INTERNAL_RESTRICTED',shareable_export_allowed=False,
            publication_review_required=True,minimum_publication_cell=minimum_cell)
            for p in sorted(run_dir.rglob('*')) if p.is_file()]
        save_table(run_dir/'03_graph_artifact_inventory.csv',pd.DataFrame(inventory))
        summary={**provenance,'schema_version':'graph03_results_v1','execution_status':status,'created_at_utc':utc_now(),
            'characterization_policy_version':tier_qc['classification_policy_version'],
            'characterization_tier_counts':tier_qc['tier_counts'],
            'characterization_qc_status':tier_qc['status'],
            'figure_qc_status':presentation_qc['status'],
            'figure_qc_artifact':'03_graph_figure_qc.json',
            'n_r0_r2_clinical_figures_displayed':presentation_qc['n_r0_r2_clinical_figures_displayed'],
            'r0_r2_display_limitation':presentation_qc['r0_r2_display_limitation'],
            'n_review_pending_sources':int(roles.tier.eq('N3_CANDIDATE').sum()),
            'n_reviewed_independent_sources':int(roles.tier.eq('N3_REVIEWED').sum()),
            'temporal_resolution_version':VERSION,'n_endpoint_analyses':len(eligibility),
            'n_longitudinal_descriptive':n_descriptive,'n_longitudinal_models_complete':n_models,
            'n_longitudinal_not_estimable':int(eligibility.status.eq('NOT_ESTIMABLE').sum()),
            'n_unresolved_episode_groups':n_unresolved,
            'pop_denominator_qc':{'status':'PASS','rows':pop_denominators.to_dict('records'),'crosswalk_reconciled':True},
            'n_patients':len(baseline),'n_validation_tests':len(statuses),'n_not_estimable':sum(r['status']=='NOT_ESTIMABLE' for r in statuses),
            'config_sha256':cfg['_config_sha256'],'graph02_config_sha256':manifest['config_sha256'],
            'analysis_levels':['observed_baseline_description','verified_longitudinal_description','exploratory_conditional_clinical_analysis','aggregate_method_sensitivity_context'],
            'clinical_mapping':manifest.get('clinical_mapping',{}),'representation_evidence':evidence.to_dict('records'),
            'sensitivity_input_qc':frames['method_context_qc'],
            'method_statuses':context.loc[context.source_table.isin(['method_registry','unavailable']),['experiment_id','status']].to_dict('records'),
            'confirmatory_validation':False,'external_prediction_validated':False,'causal_or_latent_progression_claims':False,
            'privacy_class':'INTERNAL_RESTRICTED','shareable_export_allowed':False,
            'versions':version_info(),'sha256_by_artifact':artifact_inventory(run_dir),'inferential_scope':'exploratory_conditional_on_frozen_labels'}
        summary_path=run_dir/'03_graph_results_summary.json'; write_json(summary_path,summary)
        print(f'Graph03 {status}; results: {summary_path}')
        return {'run_dir':run_dir,'manifest':summary_path,'execution_status':status}
    except Exception as exc:
        write_json(run_dir/'03_graph_failure.json',{**provenance,'execution_status':'FAILED','exception_type':type(exc).__name__})
        raise


def main(argv=None):
    parser=base_parser(__doc__)
    parser.add_argument('--graph02-release',type=Path)
    args=parser.parse_args(argv)
    return run(load_pipeline(args.pipeline_config),graph02_release=args.graph02_release,root=args.output_dir,
               run_id=args.run_id,preflight_only=args.preflight_only or args.dry_run)

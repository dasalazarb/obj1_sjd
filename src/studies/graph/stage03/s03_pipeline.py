"""Clinical interpretation from a verified release; never reconstruct discovery."""
from pathlib import Path
import json
import pandas as pd
from .._pipeline import (load_pipeline,output_root,new_run,save_table,write_json,digest,utc_now,
                        artifact_inventory,version_info,base_parser)
from .s03_release_reader import read_release
from .s03_construct_validation import feature_roles,assert_independent,anchor_lineage_audit
from .s03_profiles import profile_tables,clinical_domains,uncovered_profiles
from .s03_incremental_value import pop_crosswalk,incremental_models
from .s03_longitudinal import longitudinal,pop_transitions,comparable_longitudinal_summary
from .s03_rare_events import rare_event_table
from .s03_reporting import figures
from .s03_uncertainty import discovery_sensitivity
from .s03_method_context import method_context,missingness_interpretation,representation_evidence


def run(cfg, *, graph02_release=None,root=None,run_id=None,preflight_only=False):
    manifest,path,frames=read_release(cfg,graph02_release)
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
        lineage=anchor_lineage_audit(roles,validation)
        tables=profile_tables(baseline,membership,roles,manifest,validation,int(manifest['random_seed']))
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
        eligibility,models,trajectories=longitudinal(master,baseline,membership,manifest,validation)
        comparison=comparable_longitudinal_summary(master,baseline,membership,manifest,validation)
        transitions=pop_transitions(master,baseline,membership,manifest,validation)
        rare=rare_event_table(baseline,validation)
        uncertainty=discovery_sensitivity(frames.get('refit_membership',pd.DataFrame()),baseline,roles,tables['anchors'],cfg['experiments'],validation)
        statuses=[]
        for name,frame in [('independent_anchors',tables['anchors']),('incremental_value',incremental),('longitudinal',eligibility),('rare_events',rare),('pop_transitions',transitions),('discovery_refit_sensitivity',uncertainty)]:
            if frame.empty: statuses.append({'analysis':name,'status':'NOT_ESTIMABLE','reason':'NO_ELIGIBLE_HYPOTHESES'})
            else:
                for row in frame.to_dict('records'):
                    statuses.append({'analysis':name,**{k:row[k] for k in ['representation_id','source_column','endpoint','event','status','reason','n_patients','n_visits'] if k in row}})
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
        anchor_names={a['source_column'] for a in validation['anchors']}
        for role in roles.loc[roles.source_column.isin(anchor_names) & roles.validation_role.isin(['defining','derived_from_defining'])].itertuples():
            statuses.append({'analysis':'independent_anchors','representation_id':role.representation_id,'source_column':role.source_column,'status':'NOT_ESTIMABLE','reason':'CIRCULAR_DEFINING_OR_DERIVED_CONSTRUCT'})
        table_map={'03_graph_representation_evidence.csv':evidence,'03_graph_clinical_domains_by_community.csv':domains,
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
        for name,frame in table_map.items():
            frame=frame.copy()
            for key,value in provenance.items(): frame[key]=value
            save_table(run_dir/name,frame)
        figures(tables['anchors'],trajectories,pop,run_dir/'figures',flow=tables['flow'],domains=domains,
            context=context,missingness=missingness,minimum_cell=minimum_cell,primary_reference_id=manifest['primary_reference_id'])
        (run_dir/'figures/README.md').write_text('Aggregate internal figures and source CSVs. Publication review is required; no automatic public export. Small cells are suppressed using the configured publication minimum.\n')
        limited=any(row['status'] not in {'COMPLETE','DESCRIPTIVE'} for row in statuses)
        status='COMPLETE_WITH_LIMITATIONS' if limited else 'COMPLETE'
        (run_dir/'03_graph_results_summary.md').write_text(f'# Graph03 exploratory interpretation\n\nEXPLORATORY: no confirmatory clinical or predictive claims.\n\nFrozen Graph02 run: {manifest["run_id"]}; release SHA: {digest(path)}.\n\nPatients: {len(baseline)}; execution: {status}. Raw observed baseline values only; no imputation or discovery refit.\n\nCommunity and time effects are conditional on estimated frozen labels; intervals do not incorporate discovery uncertainty. See validation status for every unavailable or circular hypothesis.\n\nR0 is the frozen primary reference; R2 clinical comparison retains the release mapping and evidence status; other eligible representations are technical comparators.\n\n'
            + evidence[['representation_id','analysis_role','clinical_mapping_status','evidence_level','limitation_reason']].to_csv(index=False)
            + '\nFrozen method status inventory:\n'
            + context.loc[context.source_table.isin(['method_registry','unavailable'])].reindex(columns=['experiment_id','status','method_reason','n_patients','n_common']).to_csv(index=False)
            + '\nMethod sensitivity states (M/I/D/DG) preserve original denominators. DG is a sparse graph, not Mapper; cross-algorithm ARI does not establish equivalence. None of these methods defines clinical memberships.\n\nMissingness is descriptive: encoded cells and source observation units use different denominators. MCAR/MAR/MNAR, causality and latent disease progression are not established. No method or mask weight is selected using clinical associations.\n\nInference requires the existing stability and support gates; descriptive provisional profiles do not relax those gates. In-sample delta R2 is not external predictive utility. All artifacts remain INTERNAL_RESTRICTED pending publication review.\n')
        with (run_dir/'logs/03_graph_events.jsonl').open('w') as stream:
            for row in statuses: stream.write(json.dumps({**provenance,**row},default=str)+'\n')
        inventory=[dict(artifact=p.relative_to(run_dir).as_posix(),sha256=digest(p),
            contains_patient_identifiers=False,privacy_class='INTERNAL_RESTRICTED',shareable_export_allowed=False,
            publication_review_required=True,minimum_publication_cell=minimum_cell)
            for p in sorted(run_dir.rglob('*')) if p.is_file()]
        save_table(run_dir/'03_graph_artifact_inventory.csv',pd.DataFrame(inventory))
        summary={**provenance,'schema_version':'graph03_results_v1','execution_status':status,'created_at_utc':utc_now(),
            'n_patients':len(baseline),'n_validation_tests':len(statuses),'n_not_estimable':sum(r['status']=='NOT_ESTIMABLE' for r in statuses),
            'config_sha256':cfg['_config_sha256'],'graph02_config_sha256':manifest['config_sha256'],
            'analysis_levels':['observed_baseline_description','exploratory_conditional_clinical_analysis','aggregate_method_sensitivity_context'],
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

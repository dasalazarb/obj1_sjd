"""Clinical interpretation from a verified release; never reconstruct discovery."""
from pathlib import Path
import json
import pandas as pd
from .._pipeline import (load_pipeline,output_root,new_run,save_table,write_json,digest,utc_now,
                        artifact_inventory,version_info,base_parser)
from .s03_release_reader import read_release
from .s03_construct_validation import feature_roles,assert_independent
from .s03_profiles import profile_tables
from .s03_incremental_value import pop_crosswalk,incremental_models
from .s03_longitudinal import longitudinal,pop_transitions
from .s03_rare_events import rare_event_table
from .s03_reporting import figures
from .s03_uncertainty import discovery_sensitivity


def run(cfg, *, graph02_release=None,root=None,run_id=None,preflight_only=False):
    manifest,path,frames=read_release(cfg,graph02_release)
    root=Path(root) if root else output_root(cfg,'03')
    run_dir=new_run(root,run_id)
    provenance={'graph02_run_id':manifest['run_id'],'graph02_release_sha256':digest(path),
                'baseline_definition_id':manifest['baseline_definition_id'],'analysis_mode':manifest['study_mode']}
    write_json(run_dir/'03_graph_release_input_qc.json',{**provenance,'status':'PASS','read_mode':'frozen_artifacts_no_refit'})
    if preflight_only: return {'run_dir':run_dir,'execution_status':'PREFLIGHT_ONLY','manifest':None}
    try:
        baseline=frames['baseline_raw']; membership=frames['membership']; validation=cfg['validation']
        roles=feature_roles(frames['feature_sets'],baseline,validation,manifest['eligible_representation_ids'])
        tables=profile_tables(baseline,membership,roles,manifest,validation,int(manifest['random_seed']))
        assert_independent(tables['anchors'])
        pop,concordance=pop_crosswalk(baseline,membership,manifest['eligible_representation_ids'],validation)
        incremental=incremental_models(baseline,membership,roles,manifest,validation)
        master=pd.read_parquet(manifest['source_inputs']['integrated']['path'])
        eligibility,models,trajectories=longitudinal(master,baseline,membership,manifest,validation)
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
        anchor_names={a['source_column'] for a in validation['anchors']}
        for role in roles.loc[roles.source_column.isin(anchor_names) & roles.validation_role.isin(['defining','derived_from_defining'])].itertuples():
            statuses.append({'analysis':'independent_anchors','representation_id':role.representation_id,'source_column':role.source_column,'status':'NOT_ESTIMABLE','reason':'CIRCULAR_DEFINING_OR_DERIVED_CONSTRUCT'})
        table_map={'03_graph_patient_population_flow.csv':tables['flow'],'03_graph_community_baseline_table.csv':tables['baseline'],
            '03_graph_clinical_feature_roles.csv':roles,'03_graph_clinical_signatures.csv':tables['signatures'],
            '03_graph_independent_anchor_validation.csv':tables['anchors'],'03_graph_pop_crosswalk.csv':pop,
            '03_graph_pop_concordance.csv':concordance,'03_graph_pop_transitions.csv':transitions,'03_graph_incremental_value.csv':incremental,
            '03_graph_longitudinal_eligibility.csv':eligibility,'03_graph_longitudinal_models.csv':models,
            '03_graph_community_trajectory_summary.csv':trajectories,'03_graph_missingness_by_community.csv':tables['missingness'],
            '03_graph_rare_events.csv':rare,'03_graph_anchor_discovery_sensitivity.csv':uncertainty,'03_graph_validation_status.csv':pd.DataFrame(statuses),
            '03_graph_cross_representation_alignment.csv':frames.get('alignment',pd.DataFrame())}
        for name,frame in table_map.items():
            frame=frame.copy()
            for key,value in provenance.items(): frame[key]=value
            save_table(run_dir/name,frame)
        figures(tables['anchors'],trajectories,pop,run_dir/'figures')
        limited=any(row['status']=='NOT_ESTIMABLE' for row in statuses)
        status='COMPLETE_WITH_LIMITATIONS' if limited else 'COMPLETE'
        (run_dir/'03_graph_results_summary.md').write_text(f'# Graph03 exploratory interpretation\n\nEXPLORATORY: no confirmatory clinical or predictive claims.\n\nFrozen Graph02 run: {manifest["run_id"]}; release SHA: {digest(path)}.\n\nPatients: {len(baseline)}; execution: {status}. Raw observed baseline values only; no imputation or discovery refit.\n\nCommunity and time effects are conditional on estimated frozen labels; intervals do not incorporate discovery uncertainty. See validation status for every unavailable or circular hypothesis.\n')
        with (run_dir/'logs/03_graph_events.jsonl').open('w') as stream:
            for row in statuses: stream.write(json.dumps({**provenance,**row},default=str)+'\n')
        summary={**provenance,'schema_version':'graph03_results_v1','execution_status':status,'created_at_utc':utc_now(),
            'n_patients':len(baseline),'n_validation_tests':len(statuses),'n_not_estimable':sum(r['status']=='NOT_ESTIMABLE' for r in statuses),
            'config_sha256':cfg['_config_sha256'],'graph02_config_sha256':manifest['config_sha256'],
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

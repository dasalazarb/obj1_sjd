"""Clinical interpretation from a verified release; never reconstruct discovery."""
from contextlib import ExitStack
from pathlib import Path
from os.path import relpath
import json
import pandas as pd
from .._pipeline import (digest,load_pipeline,output_root,output_lock,stage_destinations,stage_outputs,
                         staged_artifact_paths,save_table,write_json,base_parser)
from ..stage02.s02_contracts import CURRENT_SCHEMA as GRAPH02_SCHEMA, MANIFEST
from .s03_release_reader import RESULTS, RESULTS_SCHEMA, read_release
from .s03_n3_validation import run_n3_validation,n3_validation_qc,enforce_n3_review_gates
from .s03_construct_validation import feature_roles,assert_independent,anchor_lineage_audit
from .s03_profiles import profile_tables,clinical_domains,uncovered_profiles,pop_denominator_audit,clinical_marker_summaries
from .s03_incremental_value import pop_crosswalk,incremental_models
from .s03_longitudinal import longitudinal,pop_transitions,comparable_longitudinal_summary,temporal_overlap_audit
from .s03_baseline_window import FROZEN_SNAPSHOT, WARNING, calendar_day, policy as baseline_policy, require_policy
from .s03_n3_exploratory import (
    _analyze, _empty_outputs, _policy, current_source_inputs, frozen_fingerprints, write_baseline_artifacts,
)
from .s03_episode_audit import resolve_all_endpoints, VERSION
from .s03_n3_approved_view import validate_baseline
from .s03_rare_events import rare_event_table
from .s03_reporting import figures,characterization_figures,safe_characterization_panels
from .s03_figure_qc import figure_qc,validate_figure_policy
from .s03_characterization import statistical_role_gates,characterization_tables,characterization_qc
from .s03_uncertainty import discovery_sensitivity
from .s03_method_context import method_context,missingness_interpretation,representation_evidence


def run(cfg, *, graph02_release=None, root=None, preflight_only=False):
    release = Path(graph02_release).resolve() if graph02_release else output_root(cfg, '02') / MANIFEST
    release_dir = release if release.is_dir() else release.parent
    root = Path(root).resolve() if root else output_root(cfg, '03')
    destinations = stage_destinations(root, include_figures=not preflight_only)
    if any(path.is_relative_to(source) or source.is_relative_to(path)
           for path in destinations.values() for source in stage_destinations(release_dir).values()):
        raise ValueError('Graph03 destination overlaps Graph02')
    figure_prefix = Path(relpath(destinations.get('figures', root), root))
    with ExitStack() as stack:
        if release_dir.is_dir():
            stack.enter_context(output_lock(release_dir, shared=True))
        staging = stack.enter_context(stage_outputs(
            destinations, lambda _, name: name.startswith('03_graph_'),
            failure=('stage', '03_graph_failure.json'), completion=root / RESULTS, diagnostic=preflight_only,
            obsolete_dirs=('figures', 'logs')))
        status = _execute(cfg, release, staging['stage'], preflight_only, graph02_release is not None,
                          staging.get('figures', staging['stage']), figure_prefix)
    summary = None if preflight_only else root / RESULTS
    print(f'Graph03 {status}; results: {root}')
    return {'run_dir': root, 'manifest': summary, 'execution_status': status}


def _execute(cfg, release, run_dir, preflight_only, explicit_release, figure_dir, figure_prefix):
    require_policy(cfg)
    require_policy(cfg['validation'])
    validate_figure_policy(cfg['validation'])
    manifest,path,frames=read_release(cfg,release if explicit_release else None)
    frozen_before = frozen_fingerprints(path, manifest)
    provenance={**baseline_policy().metadata(), 'graph02_release_sha256':frozen_before['release_manifest'],
                'graph02_directory':str(path.parent),
                'baseline_definition_id':manifest['baseline_definition_id'],'analysis_mode':manifest['study_mode'],
                'feature_policy':manifest['feature_policy']}
    input_qc = {**provenance, 'status': 'PASS', 'read_mode': 'published_artifacts_no_refit',
                'missingness_sensitivity': frames['method_context_qc'],
                'eligible_representation_ids': manifest['eligible_representation_ids']}
    if (path.parent / '02_graph_failure.json').is_file():
        input_qc['graph02_last_attempt_status'] = 'FAILED_USING_LAST_COMPLETE_RESULT'
    if preflight_only:
        frozen_after = frozen_fingerprints(path, manifest)
        if frozen_before != frozen_after:
            raise ValueError('GRAPH02_CHANGED_DURING_PREFLIGHT')
        input_qc.update(graph02_sha256_before=frozen_before, graph02_sha256_after=frozen_after)
        write_json(run_dir / '03_graph_preflight.json', input_qc)
        return 'PREFLIGHT_ONLY'
    write_json(run_dir / '03_graph_release_input_qc.json', input_qc)
    baseline=frames['baseline_raw']; membership=frames['membership']; validation=cfg['validation']
    roles=feature_roles(frames['feature_sets'],baseline,validation,manifest['eligible_representation_ids'],frames.get('clinical_registry',pd.DataFrame()))
    roles=statistical_role_gates(roles,baseline,membership,manifest,validation)
    window_roles = roles.copy()
    n3_tables,n3_qc=run_n3_validation(run_dir,frames,roles,manifest,validation,
                                     dict(provenance, snapshot_basis=FROZEN_SNAPSHOT, source_provenance_tier='FROZEN_RELEASE_VALUES'), None, snapshot_only=True)
    roles=enforce_n3_review_gates(roles,n3_tables['source_inventory'])
    snapshot_n3 = roles.tier.eq('N3_REVIEWED')
    roles.loc[snapshot_n3, 'inferential_eligible'] = False
    roles.loc[snapshot_n3, 'blocking_reasons'] = 'USE_VERIFIED_BASELINE_WINDOW_FOR_N3'
    lineage=anchor_lineage_audit(roles,validation)
    tables=profile_tables(baseline,membership,roles,manifest,validation,int(manifest['random_seed']))
    characterization=characterization_tables(baseline,membership,roles,manifest,validation,tables['anchors'])
    markers=clinical_marker_summaries(baseline,membership,roles,manifest,validation)
    minimum_cell=int(cfg['experiments'].get('minimum_publication_cell',5))
    domains=clinical_domains(baseline,membership,roles,frames.get('clinical_registry',pd.DataFrame()),manifest,minimum_cell)
    uncovered=uncovered_profiles(baseline,membership,roles,manifest,minimum_cell)
    context=method_context(frames['method_context_inputs'],frames['method_context_qc'],manifest)
    missingness=missingness_interpretation(frames['method_context_inputs'],tables['missingness'])
    evidence=representation_evidence(manifest,membership,tables['flow'],None)
    assert_independent(tables['anchors'])
    pop,concordance=pop_crosswalk(baseline,membership,manifest['eligible_representation_ids'],validation)
    incremental=incremental_models(baseline,membership,roles,manifest,validation)
    master=pd.read_parquet(manifest['source_inputs']['integrated']['path'])
    if manifest['schema_version'] == GRAPH02_SCHEMA:
        keys = ['patient_id', 'clinical_episode_id']
        source_baseline = master.loc[master.is_clinical_baseline.eq(True)].copy()
        source_baseline[keys] = source_baseline[keys].astype('string')
        for frame in (baseline, source_baseline):
            validate_baseline(frame)
        published = baseline.set_index(keys).clinical_baseline_date.map(calendar_day).sort_index()
        observed = source_baseline.set_index(keys).clinical_baseline_date.map(calendar_day).sort_index()
        if not published.index.equals(observed.index) or not published.eq(observed).all():
            raise ValueError('GRAPH02_STEP10_BASELINE_MISMATCH')
    resolutions=resolve_all_endpoints(master,baseline,validation)
    selected = {}
    repository_root = Path(__file__).resolve().parents[4]
    window_config = _policy(repository_root)
    if manifest['schema_version'] == GRAPH02_SCHEMA:
        source_tables, records, issues = current_source_inputs(
            repository_root, step10=manifest['source_inputs']['integrated']['path'])
        study = dict(frozen=frames, registry=frames.get('clinical_registry', pd.DataFrame()),
                     manifest=manifest, roles=window_roles)
        window_outputs = _analyze(source_tables, study, records, validation, window_config,
                                  selected_measurements=selected, estimate_models=False)
    else:
        window_outputs = _empty_outputs(window_config)
        issues = [dict(artifact='graph02_release', reason='CURRENT_RELEASE_REQUIRED_FOR_BASELINE_WINDOW')]
    overlap = temporal_overlap_audit(resolutions, baseline, membership, manifest, selected, minimum_cell)
    window_qc = write_baseline_artifacts(run_dir, window_outputs, issues, frozen_before,
                                         frozen_fingerprints(path, manifest), temporal_overlap=overlap,
                                         roles=window_roles)
    temporal_tables={key:pd.concat([r.audits[key] for r in resolutions.values()],ignore_index=True)
        for key in ['temporal_source','collision_summary','resolution','unresolved_reasons']}
    pop_denominators=pop_denominator_audit(baseline,membership,manifest,validation)
    # Crosswalk reconciliation is a runtime invariant, not a reference-run constant.
    for row in pop_denominators.loc[pop_denominators.population_universe.eq('hard_supported')].itertuples():
        actual=pop.loc[pop.representation_id.eq(row.representation_id),'n_patients'].sum() if 'n_patients' in pop else 0
        if actual!=row.n_pop1+row.n_pop2+row.n_pop3: raise ValueError('POP_CROSSWALK_DENOMINATOR_MISMATCH')
    eligibility,models,trajectories=longitudinal(master,baseline,membership,manifest,validation,resolutions=resolutions)
    window_longitudinal = longitudinal(master, baseline, membership, manifest, validation,
                                       resolutions=resolutions, post_characterization=True)
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
    for row in n3_tables['incremental_eligibility'].to_dict('records'):
        statuses.append({'analysis':'n3_incremental',**{k:row[k] for k in ['representation_id','source_column','model','status','reason','n_patients']}})
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
    table_map.update({f'03_graph_longitudinal_post_characterization_{name}.csv':frame
                      for name, frame in zip(('eligibility', 'models', 'trajectories'), window_longitudinal)})
    frozen_tables = {'clinical_marker_summary', 'clinical_domains_by_community', 'community_baseline_table',
                     'clinical_feature_roles', 'clinical_signatures', 'independent_anchor_validation',
                     'missingness_by_community', 'uncovered_baseline_audit', 'anchor_lineage_audit'}
    table_map.update({f'03_graph_characterization_{name}.csv':frame for name,frame in characterization.items()})
    tier_qc=characterization_qc(characterization,baseline,membership,provenance,True)
    write_json(run_dir/'03_graph_characterization_qc.json',tier_qc)
    n3_qc.update(n3_validation_qc(n3_tables,baseline,membership,manifest,
        provenance,True,characterization['clinical_matrix']))
    write_json(run_dir/'03_graph_n3_validation_qc.json',n3_qc)
    for name,frame in table_map.items():
        frame=frame.copy()
        for key,value in provenance.items(): frame[key]=value
        if name.removeprefix('03_graph_').removesuffix('.csv') in frozen_tables or name.startswith('03_graph_characterization_'):
            frame['snapshot_basis'] = FROZEN_SNAPSHOT
            frame['source_provenance_tier'] = 'FROZEN_RELEASE_VALUES'
        elif 'longitudinal' in name or 'trajectory' in name or 'temporal_source' in name:
            frame['source_provenance_tier'] = 'VERIFIED_ENDPOINT_SPECIFIC'
        save_table(run_dir/name,frame)
    safe_matrix=safe_characterization_panels(characterization['clinical_matrix'],minimum_cell)
    safe_keys=safe_matrix[['representation_id','source_column']].drop_duplicates()
    safe_markers=markers.merge(safe_keys,on=['representation_id','source_column'],how='inner') if not markers.empty else markers
    safe_domains=domains.merge(safe_keys,on=['representation_id','source_column'],how='inner') if 'source_column' in domains else domains
    safe_anchors=tables['anchors'].merge(safe_keys,on=['representation_id','source_column'],how='inner') if not tables['anchors'].empty else tables['anchors']
    figure_sources=figures(safe_anchors,trajectories,pop,figure_dir,flow=tables['flow'],domains=safe_domains,
        context=context,missingness=missingness,minimum_cell=minimum_cell,primary_reference_id=manifest['primary_reference_id'],
        baseline_table=safe_markers,marker_config=validation.get('clinical_marker_panels',[]),
        raw_matrix=characterization['clinical_matrix'],roles=roles, source_suffix='_source')
    presentation=characterization_figures(characterization['clinical_matrix'],roles,figure_dir,validation,minimum_cell,provenance=provenance, source_suffix='_source')
    save_table(run_dir/'03_graph_characterization_N2_source_selection.csv',presentation['selection'])
    figure_sources.update(presentation['sources'])
    presentation_qc=figure_qc(figure_dir,figure_sources,provenance,protected_matrix=safe_matrix, source_suffix='_source')
    for item in presentation_qc['figures']:
        item['artifacts'] = {ext: (figure_prefix / name).as_posix() for ext, name in item['artifacts'].items()}
    write_json(run_dir/'03_graph_figure_qc.json',presentation_qc)
    if presentation_qc['status']=='FAILED': raise ValueError('GRAPH03_FIGURE_QC_FAILED')
    (figure_dir/'03_graph_figures_README.md').write_text('Aggregate internal figures and source CSVs. Publication review is required; no automatic public export. Small cells are suppressed using the configured publication minimum.\n')
    limited=any(row['status'] not in {'COMPLETE','DESCRIPTIVE'} for row in statuses) or presentation_qc['status']!='PASS' or 'graph02_last_attempt_status' in input_qc or window_qc['status'] != 'PASS'
    status='COMPLETE_WITH_LIMITATIONS' if limited else 'COMPLETE'
    (run_dir/'03_graph_results_summary.md').write_text(f'# Graph03 exploratory N1/N2 characterization\n\nEXPLORATORY: no confirmatory clinical or predictive claims.\n\nGraph02 input: {path.parent.name}.\n\nPatients: {len(baseline)}; execution: {status}. Raw observed baseline values only; no imputation or discovery refit.\n\nCommunity and time effects are conditional on estimated frozen labels; intervals do not incorporate discovery uncertainty. See validation status for every unavailable or circular hypothesis.\n\nR0 is the frozen primary reference; R2 clinical comparison retains the release mapping and evidence status; other eligible representations are technical comparators.\n\n'
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
            'Candidate independence review and display status are separate. See 03_graph_figure_qc.json and '
            f'[03_graph_figure_status.csv]({(figure_prefix / "03_graph_figure_status.csv").as_posix()}).\n\n')
        stream.write(presentation['inventory'][['figure','display_status','inference_status','reason']].to_csv(index=False))
        stream.write('\nN2 selections and documented N1-to-N2 lineage (metadata only):\n\n')
        stream.write(presentation['selection'].to_csv(index=False))
        stream.write('\nN1 describes the internal structure of Mapper. N2 describes related clinical constructs. '
                     'Both are dependent on the discovery features. The N3 inference workflow is archived. '
                     'The retained audit documents review and support requirements for independent evidence. '
                     'The available evidence does not establish defensible N3 inference under the current contract.\n\n')
        stream.write(characterization['tier_counts'].to_csv(index=False))
        stream.write('\n' + WARNING + ' See 03_graph_baseline_window_qc.json for the verified-source analysis. '
                     'The longitudinal sensitivity begins strictly after the calendar window. '
                     'Overlapping measurements do not supply independent evidence.\n')
        stream.write('\nSee characterization_independence_audit.csv and characterization_qc.json for exact sources requiring clinical/lineage/baseline review. Longitudinal endpoint approvals are separate and unchanged.\n')
        stream.write(f'\nTemporal resolution: {VERSION}. Endpoint analyses: {len(eligibility)}; descriptive: {n_descriptive}; estimable models: {n_models}; unresolved endpoint/source groups: {n_unresolved}.\n\n')
        stream.write('Measurement dates require endpoint-specific reviewed evidence; a clinical spine anchor alone is insufficient. Unresolved units are excluded without inventing dates or outcomes. Descriptions use verified visits; models additionally require repeated observations, stability, support and all prespecified covariates. Frozen stability labels are not clinical bootstrap validation.\n\n')
        stream.write(temporal_tables['temporal_source'][['endpoint','date_field_used','date_semantics_review_status','status','reason']].to_csv(index=False))
        stream.write('\nPop counts below separate classified, Unclassified, missing and invalid labels in each universe. Community absence is a separate quantity. First-followup transitions count patients once; all-followup cell patient counts are nonadditive.\n\n')
        stream.write(pop_denominators.to_csv(index=False))
    with (run_dir/'03_graph_events.jsonl').open('w') as stream:
        for row in statuses: stream.write(json.dumps({**provenance,**row},default=str)+'\n')
    inventory=[dict(artifact=artifact,
        contains_patient_identifiers=False,privacy_class='INTERNAL_RESTRICTED',shareable_export_allowed=False,
        publication_review_required=True,minimum_publication_cell=minimum_cell)
        for artifact in staged_artifact_paths(run_dir, figure_dir, figure_prefix)]
    save_table(run_dir/'03_graph_artifact_inventory.csv',pd.DataFrame(inventory))
    summary={**provenance,'schema_version':RESULTS_SCHEMA,'execution_status':status,
        'n3_inference_status':'ARCHIVED',
        'n3_validation':{key:n3_qc[key] for key in ['schema_version','status','phase_a_status','phase_b_status',
            'n_inventory_rows','source_counts_by_representation','n_candidates_for_review','n_reviewed_independent_sources',
            'n_models_complete','n_outcomes_modeled','n_not_estimable_models','n_comparisons_complete']},
        'baseline_window_qc_status':window_qc['status'],
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
        'analysis_levels':['observed_baseline_description','verified_longitudinal_description','exploratory_conditional_clinical_analysis','aggregate_method_sensitivity_context'],
        'clinical_mapping':manifest.get('clinical_mapping',{}),'representation_evidence':evidence.to_dict('records'),
        'sensitivity_input_qc':frames['method_context_qc'],
        'method_statuses':context.loc[context.source_table.isin(['method_registry','unavailable']),['experiment_id','status']].to_dict('records'),
        'confirmatory_validation':False,'external_prediction_validated':False,'causal_or_latent_progression_claims':False,
        'privacy_class':'INTERNAL_RESTRICTED','shareable_export_allowed':False,
        'inferential_scope':'exploratory_conditional_on_frozen_labels'}
    summary['sha256_by_artifact'] = {file.name: digest(file) for file in sorted(run_dir.glob('03_graph_*')) if file.is_file()}
    write_json(run_dir / RESULTS, summary)
    return status


def main(argv=None):
    parser=base_parser(__doc__)
    parser.add_argument('--graph02-release',type=Path)
    args=parser.parse_args(argv)
    return run(load_pipeline(args.pipeline_config),graph02_release=args.graph02_release,root=args.output_dir,
               preflight_only=args.preflight_only or args.dry_run)

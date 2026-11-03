"""Frozen aggregate sensitivity context; no membership or method selection."""
import numpy as np
import pandas as pd

METHODS={'M':'mapper_values_plus_mask','I':'iterative_imputation',
         'D':'observed_dense_distance','DG':'sparse_observed_graph'}
SCOPE='EXPLORATORY / conditional on frozen labels; no clinical validation'


def method_context(inputs,qc,manifest):
    rows=[]
    registry=inputs.get('method_registry',pd.DataFrame())
    sources={r['artifact']:r['sha256'] for r in qc['sources']}
    for table,frame in inputs.items():
        if table=='audit_all_representations': continue
        for original in frame.to_dict('records'):
            experiment=str(original.get('experiment_id',original.get('representation_id','')))
            base=experiment.split('/')[0]
            if '-' not in base or base.split('-',1)[1] not in METHODS: continue
            reference,kind=base.split('-',1)
            row=dict(original)
            row['experiment_id']=experiment
            # A supporting table's COMPLETE status describes that diagnostic,
            # not success of a method whose registry says NOT_ESTIMABLE.
            row['diagnostic_status']=row.get('status','UNKNOWN')
            match=registry.loc[registry.experiment_id.eq(experiment)] if 'experiment_id' in registry else registry
            status=match.status.iloc[0] if len(match) and 'status' in match else manifest.get('missingness_sensitivity',{}).get('statuses',{}).get(experiment,{}).get('status','UNAVAILABLE_IN_RELEASE')
            reason=match.reason.iloc[0] if len(match) and 'reason' in match else row.get('reason','SOURCE_STATUS_UNAVAILABLE')
            row.update(reference_id=reference,method_family=METHODS[kind],status=status,
                method_reason=reason,source_table=table,graph03_defining=False,
                mapper_or_sparse_graph='sparse_graph' if kind=='DG' else 'mapper',
                estimator_version=qc['estimator_version'],analysis_policy=qc['analysis_policy'],
                source_sha256=sources.get(f'02_graph_missingness_{table}.csv','UNKNOWN'),
                concordance_denominator='n_common',coverage_denominator='n_patients',stability_denominator='n_evaluable_frozen_refits',
                interpretation='cross_algorithm_ARI_does_not_establish_method_equivalence' if kind=='DG' else 'robustness_diagnostic_not_clinical_validation')
            rows.append(row)
    # Missing sources remain visible per reference/method, including historical
    # releases and partially available aggregate inventories.
    for reference in ('R0','R2'):
        for suffix,family in METHODS.items():
            experiment=reference+'-'+suffix
            if any(r['experiment_id'].split('/')[0]==experiment for r in rows): continue
            declared=manifest.get('missingness_sensitivity',{}).get('statuses',{}).get(experiment,{})
            rows.append(dict(experiment_id=experiment,reference_id=reference,method_family=family,
                status=declared.get('status','UNAVAILABLE_IN_RELEASE'),reason=declared.get('reason','NOT_AVAILABLE_IN_RELEASE'),
                diagnostic_status='UNAVAILABLE_IN_RELEASE',source_table='unavailable',graph03_defining=False,
                mapper_or_sparse_graph='sparse_graph' if suffix=='DG' else 'mapper',
                n_common=np.nan,n_patients=np.nan,estimator_version=qc['estimator_version'],
                concordance_denominator='n_common',coverage_denominator='n_patients',stability_denominator='n_evaluable_frozen_refits',
                interpretation='aggregate_source_unavailable_no_membership_substitution'))
    return pd.DataFrame(rows)


def missingness_interpretation(inputs,profiles):
    """Keep coded-cell and observation-unit denominators in separate rows."""
    rows=[]
    audit=inputs.get('audit_all_representations',pd.DataFrame())
    for row in audit.to_dict('records'):
        common={k:row[k] for k in ('representation_id','group_type','group','n_patients','status','reason') if k in row}
        # Graph02 copies the global encoded-cell rate onto domain rows. Only
        # label it global; domain availability belongs to source units instead.
        if row.get('group_type')=='global':
            rows.append({**common,'measurement_basis':'encoded_cells','missing_fraction':row.get('missing_fraction',np.nan),
                'n_encoded_features':row.get('n_encoded_features',np.nan),'denominator_definition':'n_patients * n_encoded_features',
                'source_table':'audit_all_representations','observed_fraction':np.nan})
        elif row.get('group_type') in {'clinical_domain','family'}:
            rows.append({**common,'measurement_basis':'source_observation_units','observed_fraction':row.get('observed_fraction',np.nan),
                'n_observation_units':row.get('n_observation_units',np.nan),'denominator_definition':'n_patients * n_observation_units',
                'source_table':'audit_all_representations','missing_fraction':np.nan})
    for row in profiles.to_dict('records'):
        rows.append({**row,'measurement_basis':'original_clinical_source','source_table':'03_graph_missingness_by_community',
            'denominator_definition':'n_patients per source and community','observed_fraction':1-row['missing_fraction'],'status':'DESCRIPTIVE'})
    if not rows: rows=[{'status':'UNAVAILABLE_IN_RELEASE','reason':'NO_AGGREGATE_MISSINGNESS_SOURCE',
                         'measurement_basis':'UNKNOWN','missing_fraction':np.nan,'observed_fraction':np.nan}]
    result=pd.DataFrame(rows)
    result['interpretation']='descriptive_observation_process_no_MCAR_MAR_MNAR_or_causal_inference'
    return result


def representation_evidence(manifest,membership,flow,release_sha):
    rows=[]
    for identifier in manifest['eligible_representation_ids']:
        meta=manifest.get('representations',{}).get(identifier,{})
        counts=flow.loc[flow.representation_id.eq(identifier)].iloc[0]
        pending=manifest.get('clinical_mapping',{}).get('review_status','UNKNOWN')
        evidence=meta.get('evidence_level','TECHNICAL_REFERENCE_OR_SENSITIVITY')
        role='primary_reference' if identifier==manifest['primary_reference_id'] else 'clinical_comparison_provisional' if identifier=='R2' and evidence!='APPROVED' else 'clinical_comparison' if identifier=='R2' else 'technical_comparator'
        reasons=[]
        if identifier=='R2' and pending!='approved': reasons.append('R2_CLINICAL_MAPPING_PENDING')
        if counts.n_stable_hard < counts.n_supported_hard or counts.n_supported_hard==0: reasons.append('COMMUNITY_STABILITY_NOT_ESTABLISHED')
        rows.append(dict(representation_id=identifier,analysis_role=role,mapper_or_sparse_graph='mapper',
            clinical_mapping_status=pending,evidence_level=evidence,release_status=manifest['release_status'],
            n_patients=int(counts.n_baseline),n_covered=int(counts.n_mapper_covered),
            n_supported_hard=int(counts.n_supported_hard),n_stable_hard=int(counts.n_stable_hard),
            n_provisional_hard=int(counts.n_provisional_hard),limitation_reason='|'.join(reasons),
            graph02_release_sha256=release_sha,inferential_scope=SCOPE))
    return pd.DataFrame(rows)

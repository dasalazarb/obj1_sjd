"""Read only verified frozen artifacts. No discovery/modeling imports."""
from pathlib import Path
import numpy as np
import pandas as pd
from .._pipeline import digest, output_root
from ..stage02.s02_contracts import validate_release, resolve_latest, safe_artifact


def read_release(cfg,path=None):
    manifest,path=validate_release(path) if path else resolve_latest(output_root(cfg,'02'))
    if manifest['feature_policy']!=cfg['analysis']['feature_policy'] or manifest['study_mode']!=cfg['analysis']['mode']:
        raise ValueError('Graph02 release policy/mode differs from the requested pipeline')
    if manifest['data_sha256']!=manifest['source_inputs']['integrated']['sha256']:
        raise ValueError('Frozen clinical source hash disagrees with release')
    sensitivity_paths={entry.get('artifact') for entry in manifest.get('sensitivity_artifacts',[])}
    main_roles={'baseline_raw','membership','nodes','edges','feature_sets','feature_roles',
                'clinical_registry','mapping_audit','refit_membership','alignment','stability',
                'experiment_registry','community_dictionary','comparison_denominators',
                'missingness_controls','coverage_reasons'}
    clinical_roles={'baseline_raw','membership','nodes','feature_sets','refit_membership'}
    if any(relative in sensitivity_paths for role,relative in manifest['relative_paths'].items() if role in clinical_roles):
        raise ValueError('Sensitivity artifacts cannot enter primary clinical release inputs')
    frames={role:pd.read_csv(safe_artifact(path.parent,relative),dtype={'patient_id':'string'})
            for role,relative in manifest['relative_paths'].items() if role in main_roles and relative.endswith('.csv')}
    frames['method_context_inputs'],frames['method_context_qc']=read_sensitivity_artifacts(manifest,path)
    baseline=frames['baseline_raw']; membership=frames['membership']
    if any('-' in identifier for identifier in manifest['eligible_representation_ids']):
        raise ValueError('Sensitivity methods cannot define Graph03 clinical membership')
    if baseline.patient_id.isna().any() or baseline.patient_id.duplicated().any() or len(baseline)!=manifest['n_patients']:
        raise ValueError('Frozen baseline IDs do not match release population')
    required={'patient_id','representation_id','community_id','mapper_covered','hard_assignment','membership_tie',
              'supported','structural_stability_status','membership_score','membership_score_type','community_membership'}
    if not required<=set(membership): raise ValueError('Frozen membership schema incomplete')
    for c in ['mapper_covered','hard_assignment','membership_tie','supported','community_membership']:
        if membership[c].isna().any() or not membership[c].isin([True,False]).all():
            raise ValueError('Membership flags require explicit booleans')
    if membership.duplicated(['patient_id','representation_id','community_id']).any():
        raise ValueError('Duplicate frozen patient-community rows')
    ids=set(baseline.patient_id)
    for identifier in manifest['eligible_representation_ids']:
        part=membership.loc[membership.representation_id.eq(identifier)]
        if set(part.patient_id)!=ids: raise ValueError('Frozen membership patient IDs differ from baseline')
        if part.loc[part.hard_assignment].patient_id.duplicated().any():
            raise ValueError('Multiple hard communities for the same patient')
        if (part.hard_assignment & (~part.mapper_covered | part.membership_tie | ~part.community_membership)).any():
            raise ValueError('Inconsistent frozen hard membership')
    primary=membership.loc[membership.representation_id.eq(manifest['primary_reference_id'])].drop_duplicates('patient_id')
    if int(primary.mapper_covered.sum())!=manifest['n_covered']:
        raise ValueError('Frozen primary coverage differs from manifest')
    if not set(frames['nodes'].patient_id.dropna())<=ids:
        raise ValueError('Frozen Mapper nodes contain IDs outside baseline')
    finite_scores=membership.membership_score.dropna()
    if not finite_scores.between(0,1).all():
        raise ValueError('Invalid descriptive membership score')
    if digest(safe_artifact(path.parent,manifest['relative_paths']['feature_sets']))!=manifest['features_sha256']:
        raise ValueError('Frozen feature SHA differs from manifest')
    for role,entry in manifest['source_inputs'].items():
        if digest(Path(entry['path']))!=entry['sha256']:
            raise ValueError('Clinical source has changed since frozen Graph02 release: '+role)
    return manifest,path,frames


def hard_members(membership,identifier, *, stable_only=False):
    valid=(membership.representation_id.eq(identifier) & membership.hard_assignment & membership.mapper_covered
           & ~membership.membership_tie & membership.supported & membership.community_id.notna())
    if stable_only: valid &= membership.structural_stability_status.eq('stable')
    frame=membership.loc[valid,['patient_id','community_id','structural_stability_status']].copy()
    if frame.patient_id.duplicated().any(): raise ValueError('Nonunique eligible patient membership')
    frame['community_id']=frame.community_id.astype(int).astype(str)
    return frame


# Exact allowlist: never load sensitivity patient membership, pairwise support,
# coassignment, refits, or distance components into clinical/publication tables.
AGGREGATE_SENSITIVITY_TABLES = {
    name: f'02_graph_missingness_{name}.csv' for name in (
        'method_registry', 'cross_method_comparisons', 'method_coverage',
        'method_stability', 'distance_support_summary', 'sparse_graph_registry',
        'sparse_graph_support_summary', 'sparse_graph_stability',
        'sparse_graph_comparisons', 'focal_fix_qc', 'audit_all_representations',
        'imputation_by_type')}


def read_sensitivity_artifacts(manifest, path):
    """Read only explicitly inventoried, hash-verified aggregate diagnostics."""
    path=Path(path)
    suite=manifest.get('missingness_sensitivity')
    qc={'status':'NOT_AVAILABLE_IN_RELEASE','sources':[],
        'graph02_run_id':manifest['run_id'],'graph02_release_sha256':digest(path),
        'estimator_version':'UNKNOWN','analysis_policy':'aggregate_only_non_defining_no_refit',
        'study_mode':manifest['study_mode'],'feature_policy':manifest['feature_policy'],
        'eligible_representation_ids':manifest['eligible_representation_ids']}
    if not suite or not suite.get('enabled',False): return {},qc
    if suite.get('graph03_defining') is not False:
        raise ValueError('Sensitivity lacks explicit non-defining guarantee')
    qc['estimator_version']=suite.get('estimator_version','UNKNOWN')
    entries=manifest.get('sensitivity_artifacts',[])
    tables={}
    for role,filename in AGGREGATE_SENSITIVITY_TABLES.items():
        matches=[entry for entry in entries if entry.get('artifact')==filename]
        if not matches: continue
        if len(matches)!=1: raise ValueError('Ambiguous sensitivity aggregate inventory')
        entry=matches[0]
        if entry.get('graph03_defining') is not False or entry.get('analysis_role')!='sensitivity':
            raise ValueError('Sensitivity artifact lacks explicit non-defining guarantee')
        source=safe_artifact(path.parent,entry['artifact'])
        actual=digest(source)
        if actual!=entry.get('sha256') or actual!=manifest['sha256_by_artifact'].get(entry['artifact']):
            raise ValueError('Sensitivity aggregate SHA mismatch')
        try: frame=pd.read_csv(source)
        except pd.errors.EmptyDataError: frame=pd.DataFrame()
        private_fields=('patient_id','patient_index','patient_membership','subject_id','participant_id',
                        'coassignment','pairwise_support_internal','distance_components_internal')
        if any(token in str(column).lower() for column in frame for token in private_fields):
            raise ValueError('Patient-level data in sensitivity aggregate')
        tables[role]=frame
        qc['sources'].append({'artifact':entry['artifact'],'sha256':actual,
            'estimator_version':entry.get('estimator_version',qc['estimator_version']),
            'graph03_defining':False})
    qc['status']='AVAILABLE' if tables else 'NOT_AVAILABLE_IN_RELEASE'
    qc['missing_aggregate_tables']=[name for name in AGGREGATE_SENSITIVITY_TABLES if name not in tables]
    return tables,qc

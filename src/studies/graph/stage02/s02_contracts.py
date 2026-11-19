"""Current Graph02 artifact contract and explicit historical release validation."""
from pathlib import Path
import json
from .._pipeline import digest, config_path
from .. import _lab_contract as LAB

SCHEMA='graph02_release_v1'
STATUSES={'VALIDATED_TOPOLOGICAL_STRUCTURE','PARTIAL_OR_EXPLORATORY','NO_STABLE_STRUCTURE','QC_FAILED'}
CRITICAL={'membership','feature_roles','baseline_raw','feature_sets','stability','experiment_registry',
          'nodes','edges','community_dictionary','clinical_registry','release_qc'}

CURRENT_SCHEMA = 'graph02_current_v1'
MANIFEST = '02_graph_release_manifest.json'
CURRENT_ARTIFACTS = {'membership':'02_graph_patient_membership.csv','feature_roles':'02_graph_feature_role_manifest.csv',
        'baseline_raw':'02_graph_baseline_raw.csv','feature_sets':'02_graph_feature_sets.csv','stability':'02_graph_community_stability.csv',
        'experiment_registry':'02_graph_experiment_registry.csv','nodes':'02_graph_mapper_nodes.csv','edges':'02_graph_mapper_edges.csv',
        'community_dictionary':'02_graph_community_dictionary.csv','clinical_registry':'02_graph_clinical_domain_registry.csv',
        'mapping_audit':'02_graph_clinical_feature_mapping_audit.csv','comparison_denominators':'02_graph_comparison_denominators.csv',
        'missingness_controls':'02_graph_missingness_control_comparisons.csv','coverage_reasons':'02_graph_coverage_reason_summary.csv',
        'refit_membership':'02_graph_refit_membership.csv','release_qc':'02_graph_release_qc.json','alignment':'02_graph_cross_representation_alignment.csv'}

def source_provenance(integrated,candidates,cfg):
    report=LAB.current_profile(integrated,candidates,cfg['analysis']['mode'])
    summary=json.loads(Path(candidates).with_name('00_pharma_profile_summary.json').read_text())
    required=cfg['sensitivity']['input_governance']['required_pharma_profile_contract']
    if summary.get('profile_contract_version')!=required:
        raise ValueError('INCOMPATIBLE_PHARMA_PROFILE_CONTRACT')
    return report


def resolve_sources(cfg):
    paths=cfg['paths']; anchor=paths.get('source_manifest')
    if anchor:
        manifest=config_path(cfg['_path'],anchor)
        source=json.loads(manifest.read_text())
        if source.get('schema_version')!='graph_source_v1' or source.get('feature_policy')!=cfg['analysis']['feature_policy']:
            raise ValueError('Incompatible pinned source manifest')
        resolved={}
        for role in ('integrated','pharma_candidates'):
            entry=source[role]; path=config_path(manifest,entry['path'])
            if digest(path)!=entry['sha256']: raise ValueError('Pinned source SHA mismatch')
            resolved[role]=path
        source_provenance(resolved['integrated'],resolved['pharma_candidates'],cfg)
        return resolved
    integrated=config_path(cfg['_path'],paths['integrated'])
    if not integrated.is_file():
        raise FileNotFoundError('Missing integrated baseline source; run the authorized upstream pipeline first')
    candidate=config_path(cfg['_path'],paths['pharma_candidates'])
    # Pharma00 still publishes bundles independently of Graph. Its legacy
    # resolver accepts one unambiguous bundle when no canonical file exists.
    candidate = LAB.resolve_candidates(candidate)
    source_provenance(integrated, candidate, cfg)
    return {'integrated': integrated, 'pharma_candidates': candidate}


def safe_artifact(root,relative):
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Artifact path must remain inside frozen release')
    target=(Path(root)/path).resolve()
    if not target.is_relative_to(Path(root).resolve()) or not target.is_file():
        raise ValueError('Missing or escaping release artifact')
    return target


def artifact_paths(manifest):
    return CURRENT_ARTIFACTS if manifest['schema_version'] == CURRENT_SCHEMA else manifest['relative_paths']


def validate_release(path, *, expected_sha=None, baseline_only=False):
    path=Path(path).resolve()
    if path.is_dir(): path=path/MANIFEST
    if expected_sha and digest(path)!=expected_sha:
        raise ValueError('Release manifest SHA mismatch')
    manifest=json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError('Graph02 descriptor must be a JSON object')
    if manifest.get('schema_version') == CURRENT_SCHEMA:
        if expected_sha:
            raise ValueError('A current Graph02 result cannot serve as a historical pin')
        if manifest.get('finalized') is not True or manifest.get('release_status') not in STATUSES - {'QC_FAILED'}:
            raise ValueError('Graph02 is incomplete or QC_FAILED')
        qc = json.loads(safe_artifact(path.parent, CURRENT_ARTIFACTS['release_qc']).read_text())
        if qc.get('primary_qc') != 'PASS' or qc.get('release_status') != manifest['release_status']:
            raise ValueError('Current Graph02 QC mismatch')
        safe_artifact(path.parent, CURRENT_ARTIFACTS['baseline_raw'])
        if not baseline_only:
            for role in CRITICAL:
                safe_artifact(path.parent, CURRENT_ARTIFACTS[role])
            if manifest['primary_reference_id'] not in manifest['eligible_representation_ids']:
                raise ValueError('Current Graph02 primary representation missing')
            coverage = manifest['coverage']
            if (coverage['n_total'] != manifest['n_patients'] or coverage['n_covered_R0'] != manifest['n_covered']
                    or coverage['n_covered_R0'] + coverage['n_uncovered_R0'] != coverage['n_total']):
                raise ValueError('Current Graph02 coverage denominator mismatch')
        return manifest, path
    required={'schema_version','finalized','run_id','git_commit','study_mode','feature_policy','data_sha256',
              'pharma_profile_sha256','config_sha256','features_sha256','baseline_definition_id','n_patients',
              'n_features','n_covered','method_list','primary_reference_id','eligible_representation_ids',
              'release_status','created_at_utc','relative_paths','sha256_by_artifact','source_inputs','random_seed'}
    if required-set(manifest):
        raise ValueError('Graph02 release manifest is incomplete')
    if manifest.get('schema_version')!=SCHEMA:
        raise ValueError('Unsupported Graph02 release schema')
    if manifest.get('finalized') is not True or manifest.get('release_status') not in STATUSES-{'QC_FAILED'}:
        raise ValueError('Graph02 release is incomplete or QC_FAILED')
    if manifest.get('run_id')!=path.parent.name:
        raise ValueError('Graph02 run ID/path mismatch')
    if not CRITICAL<=manifest.get('relative_paths',{}).keys():
        raise ValueError('Graph02 release lacks critical artifacts')
    hashes=manifest.get('sha256_by_artifact',{})
    for rel in manifest['relative_paths'].values():
        if rel not in hashes: raise ValueError('Critical artifact missing hash')
    for rel,expected in hashes.items():
        if digest(safe_artifact(path.parent,rel))!=expected:
            raise ValueError('Frozen release artifact SHA mismatch: '+rel)
    qc=json.loads(safe_artifact(path.parent,manifest['relative_paths']['release_qc']).read_text())
    if qc.get('release_status')!=manifest['release_status'] or qc.get('primary_qc')!='PASS':
        raise ValueError('Release QC/manifest mismatch')
    if manifest['primary_reference_id'] not in manifest['eligible_representation_ids']:
        raise ValueError('Frozen primary representation missing')
    # Additive v2 diagnostics remain optional for historical v1 releases.
    if 'coverage' in manifest:
        coverage=manifest['coverage']
        if (coverage.get('n_total')!=manifest['n_patients'] or
            coverage.get('n_covered_R0')!=manifest['n_covered'] or
            coverage.get('n_covered_R0',0)+coverage.get('n_uncovered_R0',0)!=coverage.get('n_total')):
            raise ValueError('Coverage denominator/manifest mismatch')
    if 'representations' in manifest and 'R2' in manifest['representations']:
        r2=manifest['representations']['R2']
        if r2.get('status') not in {'COMPLETE','NOT_ESTIMABLE','FAILED'}:
            raise ValueError('Invalid R2 execution status')
        if r2.get('evidence_level') not in {'APPROVED','EXPLORATORY_PROVISIONAL','NOT_ESTIMABLE'}:
            raise ValueError('Invalid R2 evidence level')
        if r2.get('evidence_level')=='APPROVED' and manifest.get('clinical_mapping',{}).get('review_status')!='approved':
            raise ValueError('Clinical mapping approval is not established')
    if manifest.get('missingness_sensitivity',{}).get('enabled'):
        sensitivity=manifest['missingness_sensitivity']
        if sensitivity.get('analysis_role')!='sensitivity' or sensitivity.get('graph03_defining') is not False:
            raise ValueError('Invalid missingness sensitivity role')
        if any(i in manifest['eligible_representation_ids'] for i in sensitivity['statuses']):
            raise ValueError('Missingness sensitivity cannot become a defining representation')
        for item in manifest.get('sensitivity_artifacts',[]):
            if item.get('analysis_role')!='sensitivity' or hashes.get(item['artifact'])!=item['sha256']:
                raise ValueError('Sensitivity artifact checksum/role mismatch')
        if set(sensitivity['artifacts'])!={item['artifact'] for item in manifest.get('sensitivity_artifacts',[])}:
            raise ValueError('Missing sensitivity artifact inventory')
    if 'artifact_inventory' in manifest:
        for item in manifest['artifact_inventory']:
            if hashes.get(item['artifact'])!=item.get('sha256'):
                raise ValueError('Diagnostic inventory SHA mismatch')
            if item.get('shareable_export_allowed') is not False:
                raise ValueError('Graph02 artifact export requires publication review')
    return manifest,path

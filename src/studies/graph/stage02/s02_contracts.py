"""Frozen release schema, strict source selection, hashes, and atomic publication."""
from pathlib import Path
import json
import os
from .._pipeline import digest, write_json, utc_now, config_path
from .. import _lab_contract as LAB

SCHEMA='graph02_release_v1'
STATUSES={'VALIDATED_TOPOLOGICAL_STRUCTURE','PARTIAL_OR_EXPLORATORY','NO_STABLE_STRUCTURE','QC_FAILED'}
CRITICAL={'membership','feature_roles','baseline_raw','feature_sets','stability','experiment_registry',
          'nodes','edges','community_dictionary','clinical_registry','release_qc'}


def source_provenance(integrated,candidates,cfg):
    report=LAB.validate_provenance(integrated,candidates,cfg['analysis']['mode'])
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
    # A configured versioned directory is exact. Otherwise inventory all versions,
    # retaining only validated, source-compatible bundles. Never select by mtime.
    choices=[candidate] if candidate.parent.parent.name=='runs' else [candidate,*sorted((candidate.parent/'runs').glob('*/'+candidate.name))]
    valid={}; failures=[]
    for path in choices:
        if not path.is_file(): continue
        try:
            source_provenance(integrated,path,cfg)
            signature=tuple(digest(path.with_name(name)) for name in (path.name,'00_pharma_profile_summary.json','00_pharma_graph_lab_selection_audit.csv'))
            valid.setdefault(signature,path)
        except (ValueError,OSError,KeyError) as exc:
            failures.append(type(exc).__name__)
    if not valid:
        raise ValueError('NO_COMPATIBLE_PHARMA00_PROFILE: regenerate a source-compatible Pharma00 bundle; no valid candidate')
    if len(valid)>1:
        raise ValueError('AMBIGUOUS_PHARMA00_PROFILES: pin an approved source_manifest; distinct compatible runs exist')
    return {'integrated':integrated,'pharma_candidates':next(iter(valid.values()))}


def safe_artifact(root,relative):
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Artifact path must remain inside frozen release')
    target=(Path(root)/path).resolve()
    if not target.is_relative_to(Path(root).resolve()) or not target.is_file():
        raise ValueError('Missing or escaping release artifact')
    return target


def validate_release(path, *, expected_sha=None):
    path=Path(path).resolve()
    if path.is_dir(): path=path/'02_graph_release_manifest.json'
    if expected_sha and digest(path)!=expected_sha:
        raise ValueError('Release manifest SHA mismatch')
    manifest=json.loads(path.read_text())
    required={'schema_version','finalized','run_id','git_commit','study_mode','feature_policy','data_sha256',
              'pharma_profile_sha256','config_sha256','features_sha256','baseline_definition_id','n_patients',
              'n_features','n_covered','method_list','primary_reference_id','eligible_representation_ids',
              'release_status','created_at_utc','relative_paths','sha256_by_artifact','source_inputs','random_seed'}
    if not isinstance(manifest,dict) or required-set(manifest):
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


def resolve_latest(root):
    pointer_path=Path(root)/'latest_completed_02.json'
    if not pointer_path.is_file():
        raise ValueError('MISSING_COMPLETED_GRAPH02_RELEASE: execute Graph02 before Graph03')
    pointer=json.loads(pointer_path.read_text())
    if pointer.get('schema_version')!='graph02_pointer_v1':
        raise ValueError('Unsupported Graph02 latest pointer schema')
    path=safe_artifact(root,pointer['manifest_path'])
    manifest,path=validate_release(path,expected_sha=pointer['manifest_sha256'])
    if manifest['run_id']!=pointer['run_id'] or pointer['release_status']!=manifest['release_status']:
        raise ValueError('Latest release pointer metadata mismatch')
    return manifest,path


def publish_pointer(root,manifest_path):
    manifest,path=validate_release(manifest_path)
    write_json(Path(root)/'latest_completed_02.json',{'schema_version':'graph02_pointer_v1',
        'run_id':manifest['run_id'],'manifest_path':path.relative_to(root).as_posix(),
        'manifest_sha256':digest(path),'release_status':manifest['release_status'],'created_at_utc':utc_now()},atomic=True)

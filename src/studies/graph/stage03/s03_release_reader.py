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
    frames={role:pd.read_csv(safe_artifact(path.parent,relative),dtype={'patient_id':'string'})
            for role,relative in manifest['relative_paths'].items() if relative.endswith('.csv')}
    baseline=frames['baseline_raw']; membership=frames['membership']
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

"""Additive sensitivity spaces; historical R0/R2 and N2/M1 remain untouched."""
import inspect
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from .s02_missingness_registry import NotEstimable, build_panel, panel_metadata
from .s02_imputation import fit_transform
from .s02_pairwise_distance import observed_distance, require_complete


def mask_distance(panel,reference):
    """Equal construct shares within families or clinical subdomains/domains.

    Each unit contributes one state mismatch, keeping partial and unknown distinct
    from observed and negative. Administrative sources stay in historical controls.
    """
    meta=panel.metadata.loc[~panel.metadata.administrative]
    if meta.empty: raise NotEstimable('NO_NONADMINISTRATIVE_OBSERVATION_UNITS')
    top='clinical_domain' if reference=='R2' else 'family'
    d=np.zeros((len(panel.mask),len(panel.mask)))
    for _,part in meta.groupby(top,sort=True):
        groups=list(part.groupby('subdomain')) if reference=='R2' else [('family',part)]
        for _,sub in groups:
            for c in sub.index:
                states=panel.states[c].to_numpy()
                d+=(states[:,None]!=states[None,:])/(meta[top].nunique()*len(groups)*len(sub))
    return d


def build_variant(identifier,raw,families,config,registry,manifest,baseline,lab_groups,feature_to_lab,
                  policy,*,reference_space=None,mask_weight=None,distance_threshold=None):
    from .s02_representations import Space,build_space
    reference,method=identifier.split('-')
    if reference not in {'R0','R2'} or method not in {'I','D','M'}: raise ValueError('UNKNOWN_MISSINGNESS_VARIANT')
    meta=panel_metadata(raw,registry,reference)
    panel=build_panel(raw,registry,reference,baseline)
    audit={'panel':panel,'input_panel_id':panel.panel_id,'panel_hash':panel.fingerprint,
           'analysis_role':'sensitivity','reference_id':reference}
    if method=='I':
        imputation_policy={**policy['iterative_regularized'],'reference_id':reference}
        reconstructed=fit_transform(raw,meta,int(config['random_seed']),imputation_policy)
        space=build_space(reference,reconstructed.completed,families,config,registry,manifest,baseline,lab_groups,feature_to_lab)
        # Preserve original availability while retaining the reference's exact
        # PCA/lens/Mapper implementation on the reconstructed refit.
        space.raw=raw.copy(); space.method=reference+'_iterative_weighted_low_rank'
        audit.update(imputation=reconstructed.audit,validation=reconstructed.validation)
    elif method=='D':
        distance_policy=dict(policy['observed_pairwise_distance'])
        if distance_threshold is not None: distance_policy['minimum_shared_weight_fraction']=float(distance_threshold)
        evidence=observed_distance(raw,registry,reference,distance_policy,baseline)
        audit['pairwise']=evidence
        d=require_complete(evidence)
        options={'metric':'precomputed','metric_mds':True,'init':'random'} if 'metric_mds' in inspect.signature(MDS).parameters else {'metric':True,'dissimilarity':'precomputed'}
        x=MDS(n_components=2,random_state=int(config['random_seed']),n_init=4,**options).fit_transform(d)
        space=Space(raw.copy(),x,d,d,pd.DataFrame(),'observed_pairwise_nonmetric_possible_MDS_lens',list(raw))
    else:
        weight=float(policy['values_plus_mask']['primary_mask_weight'] if mask_weight is None else mask_weight)
        if not 0<=weight<=1: raise ValueError('INVALID_MASK_WEIGHT')
        base=reference_space or build_space(reference,raw,families,config,registry,manifest,baseline,lab_groups,feature_to_lab)
        if base.features!=list(raw): raise ValueError('FIXED_PANEL_MISMATCH')
        if weight==0:
            # Exact reproduction, including lens and legacy Mapper dispatch.
            return base,{**audit,'mask_weight':0.,'reference_identity':True}
        clinical=base.distance.copy(); scale=np.sqrt(np.mean(clinical**2))
        if not scale>0: raise NotEstimable('ZERO_CLINICAL_DISTANCE_SCALE')
        clinical=clinical**2/(scale**2); masks=mask_distance(panel,reference)
        # Classical scaling provides a deterministic lens for the convex
        # squared-distance combination. DBSCAN sees its actual distance.
        clinical_component=(1-weight)*clinical; mask_component=weight*masks
        squared=clinical_component+mask_component; d=np.sqrt(np.maximum(0,squared))
        centering=np.eye(len(d))-np.ones_like(d)/len(d)
        gram=-.5*centering@squared@centering
        eigenvalues,eigenvectors=np.linalg.eigh(gram); ids=np.argsort(eigenvalues)[::-1][:2]
        if np.sum(eigenvalues[ids]>0)<2: raise NotEstimable('MASK_MIXTURE_INSUFFICIENT_RANK')
        x=eigenvectors[:,ids]*np.sqrt(eigenvalues[ids])
        space=Space(raw.copy(),x,d,d,pd.DataFrame(),'values_plus_panel_state_mask',list(raw))
        audit.update(mask_weight=weight,value_squared_component=clinical_component,mask_squared_component=mask_component,
            distance_normalization='clinical_RMS_within_refit;mask_weighted_state_disagreement',legacy_M1_alias=False)
    if space.features!=list(raw): raise ValueError('FIXED_PANEL_MISMATCH')
    space.missingness_audit=audit
    return space,audit

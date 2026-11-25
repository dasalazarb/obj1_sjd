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
        space.raw=raw.copy(); space.method=reference+'_'+reconstructed.audit['estimator']
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


def supported_mask(reasons, reference, layer, policy, *, states=None):
    """Pair denominators count comparable opportunities, weighted by the fixed panel."""

    states = reasons.masks[layer] if states is None else states
    meta = reasons.observation.metadata.loc[states.columns]
    top = 'clinical_domain' if reference == 'R2' else 'family'
    n = len(states)
    numerator, denominator = np.zeros((n, n)), np.zeros((n, n))
    domain_support = {}
    varying = False
    for _, group in meta.groupby(top, sort=True):
        parts = list(group.groupby('subdomain', sort=True)) if reference == 'R2' else [('family', group)]
        for _, part in parts:
            weight = 1/(meta[top].nunique()*len(parts)*len(part))
            for unit, row in part.iterrows():
                values = states[unit].to_numpy()
                measured = pd.notna(values)
                comparable = measured[:, None] & measured[None, :]
                varying |= states[unit].nunique() > 1
                denominator += weight*comparable
                numerator += weight*(comparable & (values[:, None]!=values[None, :]))
                if row.clinical_domain not in ('UNMAPPED', 'D09'):
                    domain = domain_support.setdefault(row.clinical_domain, np.zeros((n, n), dtype=bool))
                    domain |= comparable
    domains = sum(domain_support.values(), start=np.zeros((n, n), dtype=int))
    supported = (domains >= policy['minimum_shared_domains']) & (
        denominator + 1e-12 >= policy['minimum_shared_weight_fraction'])
    distance = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=supported)
    np.fill_diagonal(distance, 0.)
    upper = np.triu_indices(n, 1)
    pairs = int(supported[upper].sum())
    audit = dict(mask_layer=layer, n_units=len(meta), n_patients=n, n_pairs=len(upper[0]), n_supported_pairs=pairs,
                 shared_weight_min=float(denominator[upper].min()) if len(upper[0]) else np.nan,
                 shared_domains_min=int(domains[upper].min()) if len(upper[0]) else 0,
                 minimum_shared_domains=policy['minimum_shared_domains'],
                 minimum_shared_weight_fraction=policy['minimum_shared_weight_fraction'])
    reason = ('NO_DOCUMENTED_CLINICAL_DISAGREEMENT' if layer=='clinical_documented' else 'NO_MASK_DISAGREEMENT') if not varying else (
        'INCOMPLETE_COMPARABLE_MASK_PAIRS' if pairs!=len(upper[0]) else '')
    return distance, denominator, domains, {**audit, 'status': 'NOT_ESTIMABLE' if reason else 'COMPLETE', 'reason': reason}


def classical_lens(distance):
    if (not np.isfinite(distance).all() or np.any(distance < 0)
            or not np.allclose(distance, distance.T) or not np.allclose(np.diag(distance), 0.)):
        raise NotEstimable('INVALID_OR_INCOMPLETE_DISSIMILARITY')
    squared = distance**2
    gram = -.5*(squared-squared.mean(axis=0)-squared.mean(axis=1)[:, None]+squared.mean())
    eigenvalues, vectors = np.linalg.eigh(gram)
    tolerance = np.finfo(float).eps*len(distance)*max(1., np.max(np.abs(eigenvalues)))
    positive = np.flatnonzero(eigenvalues>tolerance)
    negative = eigenvalues[eigenvalues < -tolerance]
    if len(positive)<2:
        raise NotEstimable('MDS_LENS_INSUFFICIENT_POSITIVE_RANK')
    order = positive[-2:][::-1]
    lens = vectors[:, order]*np.sqrt(eigenvalues[order])
    upper = np.triu_indices(len(distance), 1)
    stress = np.linalg.norm((pairwise_distances(lens)-distance)[upper])/np.linalg.norm(distance[upper])
    return lens, dict(positive_rank=len(positive), n_negative_eigenvalues=len(negative),
                      negative_inertia_fraction=float(-negative.sum()/np.abs(eigenvalues).sum()),
                      lens_relative_stress=float(stress), eigenvalues='|'.join(map(str, eigenvalues[::-1])))


def factorial_mds(base, *, mask=None, weight=0., lens_only=False):
    from .s02_representations import Space
    if not 0<=weight<=1:
        raise ValueError('INVALID_MASK_WEIGHT')
    if weight == 0 and not lens_only:
        return base, dict(reference_identity=True, mask_weight=0.)
    upper = np.triu_indices(len(base.raw), 1)
    if lens_only:
        distance = base.distance
        audit = {}
    else:
        rms = np.sqrt(np.mean(base.distance[upper]**2))
        if not np.isfinite(rms) or rms<=0:
            raise NotEstimable('ZERO_CLINICAL_DISTANCE_SCALE')
        clinical_component = (1-weight)*(base.distance/rms)**2
        mask_component = weight*mask
        distance = np.sqrt(clinical_component+mask_component)
        audit = dict(clinical_rms=float(rms), values_squared_energy=float(clinical_component[upper].mean()),
                     mask_squared_energy=float(mask_component[upper].mean()))
    lens, lens_audit = classical_lens(distance)
    method = 'clinical_distance_MDS_control' if lens_only else 'clinical_values_plus_observation_MDS'
    return Space(base.raw, lens, distance, distance, base.weights, method, base.features), {
        **audit, **lens_audit, 'mask_weight': weight,
    }


def reconstructed_space(reference, original, completed, families, config, registry, manifest,
                        baseline, lab_groups, feature_to_lab, method):
    from .s02_representations import build_space
    if (not original.index.equals(completed.index) or list(original)!=list(completed)
            or not np.isfinite(completed.to_numpy()).all()):
        raise ValueError('INVALID_COMPLETED_FIXED_PANEL')
    space = build_space(reference, completed, families, config, registry, manifest,
                        baseline, lab_groups, feature_to_lab)
    if space.features != list(original):
        raise ValueError('FIXED_PANEL_MISMATCH')
    space.raw = original
    space.method = reference+'_'+method
    space.weights = space.weights.assign(imputation=method)
    return space

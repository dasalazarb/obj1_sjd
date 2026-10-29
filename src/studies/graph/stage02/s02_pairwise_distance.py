"""Observed-construct distances with explicit support; no invented pair distances."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .s02_missingness_registry import NotEstimable, build_panel, panel_metadata


@dataclass
class PairwiseEvidence:
    distance: np.ndarray
    support: pd.DataFrame
    summary: dict


def observed_distance(raw,registry,reference,policy,baseline=None):
    meta=panel_metadata(raw,registry,reference)
    panel=build_panel(raw,registry,reference,baseline)
    n=len(raw); per_unit={}
    for key,unit in panel.metadata.iterrows():
        # All encoded levels of a nominal source form one categorical result.
        source_distances=[]
        for source in unit.sources:
            part=meta.loc[meta.source_column.eq(source)]
            encoding=part.get('encoding',pd.Series('numeric',index=part.index)).astype(str)
            block=raw[part.index]
            valid=block.notna().all(axis=1).to_numpy()
            if encoding.str.contains('nominal|one_hot',regex=True).any():
                if not np.isin(block.to_numpy()[np.isfinite(block.to_numpy(float))],[0,1]).all():
                    raise NotEstimable('INVALID_NOMINAL_ENCODING')
                valid &= block.fillna(0).sum(axis=1).eq(1).to_numpy()
                codes=block.fillna(0).to_numpy().argmax(axis=1)
                dist=(codes[:,None]!=codes[None,:]).astype(float)
            else:
                distances=[]
                for c in block:
                    x=block[c].to_numpy(float); ok=np.isfinite(x)
                    if np.isinf(x).any(): raise ValueError('INFINITE_DISTANCE_INPUT')
                    enc=str(meta.loc[c].get('encoding','numeric'))
                    if enc=='ordinal':
                        # Encoding is accepted only after the upstream reviewed
                        # ordinal map, never by guessing the order of raw labels.
                        if not c.endswith('__ordinal'): raise NotEstimable('UNVERIFIED_ORDINAL_ORDER')
                        levels=np.sort(x[ok]); levels=np.unique(levels)
                        x=np.where(ok,np.searchsorted(levels,x),np.nan)
                        scale=max(1,len(levels)-1)
                    elif 'binary' in enc:
                        if not np.isin(x[ok],[0,1]).all(): raise NotEstimable('INVALID_BINARY_ENCODING')
                        scale=1.
                    else:
                        scale=np.nanpercentile(x,75)-np.nanpercentile(x,25) if ok.any() else 0.
                        if not scale>0: scale=np.nanmax(x)-np.nanmin(x) if ok.any() else 0.
                    if not scale>0: continue  # No discriminatory evidence.
                    d=np.minimum(1.,np.abs(x[:,None]-x[None,:])/scale)
                    shared=ok[:,None]&ok[None,:]
                    distances.append(np.where(shared,d,np.nan))
                if not distances: continue
                stack=np.stack(distances); count=np.isfinite(stack).sum(axis=0)
                dist=np.divide(np.nansum(stack,axis=0),count,out=np.full((n,n),np.nan),where=count>0)
                valid=block.notna().any(axis=1).to_numpy()
            shared=valid[:,None]&valid[None,:]
            # Explicit unknown states never create observed evidence.
            known=panel.mask[key].eq(1).to_numpy()
            source_distances.append(np.where(shared & known[:,None] & known[None,:],dist,np.nan))
        if source_distances:
            stack=np.stack(source_distances); count=np.isfinite(stack).sum(axis=0)
            per_unit[key]=np.divide(np.nansum(stack,axis=0),count,out=np.full((n,n),np.nan),where=count>0)
    if not per_unit: raise NotEstimable('NO_DISCRIMINATORY_OBSERVED_CONSTRUCTS')
    numerator=np.zeros((n,n)); weight=np.zeros((n,n)); domains=np.zeros((n,n),int); constructs=np.zeros((n,n),int)
    metadata=panel.metadata.loc[list(per_unit)]
    top='clinical_domain' if reference=='R2' else 'family'
    # Fixed full-panel denominators include inactive constructs; their absence
    # must reduce support instead of artificially inflating comparability.
    all_meta=panel.metadata
    for domain,part in all_meta.groupby(top,sort=True):
        domain_observed=np.zeros((n,n),bool)
        subgroups=list(part.groupby('subdomain',sort=True)) if reference=='R2' else [('family',part)]
        for _,sub in subgroups:
            for key in sub.index:
                if key not in per_unit: continue
                d=per_unit[key]; valid=np.isfinite(d)
                w=1/(all_meta[top].nunique()*len(subgroups)*len(sub))
                numerator+=np.where(valid,d*w,0.); weight+=valid*w
                constructs+=valid; domain_observed|=valid
        domains+=domain_observed
    threshold=float(policy['minimum_shared_weight_fraction']); minimum=int(policy['minimum_shared_domains'])
    estimable=(domains>=minimum)&(weight>=threshold)&(constructs>0)
    d=np.divide(numerator,weight,out=np.full((n,n),np.nan),where=estimable)
    np.fill_diagonal(d,0.)  # Identity is the only unsupported zero exception.
    a,b=np.triu_indices(n,1)
    support=pd.DataFrame(dict(patient_index_a=a,patient_index_b=b,n_shared_constructs=constructs[a,b],
        n_shared_domains=domains[a,b],shared_weight_fraction=weight[a,b],distance_estimable=estimable[a,b],
        distance=d[a,b],reason=np.where(estimable[a,b],'','INSUFFICIENT_SHARED_OBSERVED_EVIDENCE')))
    neighbors=estimable.copy(); np.fill_diagonal(neighbors,False)
    summary=dict(n_patients=n,n_pairs=len(a),n_estimable_pairs=int(estimable[a,b].sum()),
        estimable_pair_fraction=float(estimable[a,b].mean()) if len(a) else np.nan,
        no_shared_evidence_fraction=float((constructs[a,b]==0).mean()) if len(a) else np.nan,
        n_patients_without_reliable_neighbors=int((~neighbors.any(axis=1)).sum()),
        minimum_shared_domains=minimum,minimum_shared_weight_fraction=threshold,
        shared_domains_mean=float(domains[a,b].mean()) if len(a) else np.nan,
        shared_weight_fraction_mean=float(weight[a,b].mean()) if len(a) else np.nan,
        geometry='partial_observation_dissimilarity_triangle_inequality_not_guaranteed')
    return PairwiseEvidence(d,support,summary)


def require_complete(evidence):
    if not np.isfinite(evidence.distance).all():
        raise NotEstimable('INCOMPLETE_OBSERVED_DISTANCE_MATRIX_NO_FILLING_ALLOWED')
    return evidence.distance

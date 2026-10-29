"""Representations separate the Mapper lens from its local clustering space."""
from dataclasses import dataclass
import inspect
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from . import s02_legacy as legacy
from .s02_feature_registry import eligible_domain_map


@dataclass
class Space:
    raw: pd.DataFrame
    lens: np.ndarray
    cluster: np.ndarray
    distance: np.ndarray
    weights: pd.DataFrame
    method: str
    features: list[str]
    legacy_rep: object = None


def gower_distance(frame, kinds=None, weights=None):
    """Pairwise observed-feature Gower; no shared observation is unestimable."""
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError('Gower requires distinct nonempty features')
    n=len(frame)
    total=np.zeros((n,n)); denominator=np.zeros((n,n))
    kinds=kinds or {c:'numeric' for c in frame}
    for c in frame:
        s=frame[c]
        ok=s.notna().to_numpy()
        valid=ok[:,None] & ok[None,:]
        w=float((weights or {}).get(c,1.))
        if not np.isfinite(w) or w<=0:
            raise ValueError('Invalid Gower feature weight')
        if kinds[c]=='nominal':
            x=s.astype('string').fillna('').to_numpy()
            d=(x[:,None]!=x[None,:]).astype(float)
        else:
            x=pd.to_numeric(s,errors='raise').to_numpy(dtype=float,na_value=np.nan)
            if np.isinf(x).any():
                raise ValueError('Infinite Gower input')
            span=np.nanmax(x)-np.nanmin(x) if ok.any() else 0
            if span<=0:
                continue  # Constants provide no similarity evidence.
            d=np.abs(x[:,None]-x[None,:])/span
        total += np.where(valid,d*w,0.)
        denominator += valid*w
    if np.any((denominator==0) & ~np.eye(n,dtype=bool)):
        raise ValueError('GOWER_NO_SHARED_OBSERVATIONS')
    d=np.divide(total,denominator,out=np.zeros_like(total),where=denominator>0)
    np.fill_diagonal(d,0.)
    return np.clip(d,0,1)


def mixed_source_frame(raw, manifest, baseline):
    """One exact source per mixed variable; never count one-hot levels separately."""
    meta=manifest.loc[manifest.feature.isin(raw)]
    frame=pd.DataFrame(index=raw.index); kinds={}
    from .._feature_catalog import clean
    for source, group in meta.groupby('source_column',sort=True):
        if group.representation_type.astype(str).str.startswith(('nominal','one_hot')).any() or group.encoding.astype(str).str.startswith(('nominal','one_hot')).any():
            frame[source]=clean(baseline.loc[raw.index,source])
            kinds[source]='nominal'
        else:
            name=group.feature.iloc[0]
            frame[source]=raw[name]
            kinds[source]='numeric'
    if not len(frame.columns):
        raise ValueError('No mixed source variables')
    return frame,kinds


def reduce_matrix(matrix, config, *, dimensions=None):
    limit=min(len(matrix)-1,matrix.shape[1])
    if limit<2 or not np.isfinite(matrix).all() or np.linalg.norm(matrix-matrix.mean(axis=0))==0:
        raise ValueError('Representation lacks two active dimensions')
    probe=PCA(n_components=limit,random_state=int(config['random_seed'])).fit(matrix)
    target=float(config['pca']['variance_target'])
    count=max(2,int(np.searchsorted(np.cumsum(probe.explained_variance_ratio_),target)+1))
    count=min(limit,dimensions or count)
    return probe.transform(matrix)[:,:count]


def hierarchical_mfa(raw, registry):
    meta=eligible_domain_map(registry,list(raw))
    raw=raw[meta.index]
    blocks=[]; audit=[]
    for domain, group in meta.groupby('clinical_domain',sort=True):
        nested=[]; subrows=[]
        for sub, part in group.groupby('subdomain',sort=True):
            cols=part.index.tolist()
            block,sigma=legacy.scale_mfa_block(raw[cols],f'{domain}/{sub}',allow_zero=True)
            nested.append(block)
            row={'level':'SUBDOMAIN','block':f'{domain}/{sub}','n_features':len(cols),
                 'sigma1_before_weighting':sigma,'weight_applied':1/sigma if sigma else 0.,
                 'effective_weight':1/sigma if sigma else 0.,'status':'active' if sigma else 'zero_variance'}
            subrows.append(row)
        combined=np.column_stack(nested)
        sigma=legacy._first_singular_value(combined,domain)
        if not sigma:
            # Empty/constant domains are audited and do not create an artificial biological zero block.
            audit.extend(subrows)
            audit.append({'level':'DOMAIN','block':domain,'n_features':len(group),
                          'sigma1_before_weighting':0.,'weight_applied':0.,'effective_weight':0.,'status':'zero_variance_excluded'})
            continue
        for row in subrows:
            row['effective_weight']/=sigma
        audit.extend(subrows)
        audit.append({'level':'DOMAIN','block':domain,'n_features':len(group),
                      'sigma1_before_weighting':sigma,'weight_applied':1/sigma,'effective_weight':1/sigma,'status':'active'})
        blocks.append(combined/sigma)
    if not blocks:
        raise ValueError('ZERO_VARIANCE_CLINICAL_DOMAINS')
    return np.column_stack(blocks),pd.DataFrame(audit),list(raw)


def famd_matrix(frame,kinds):
    if not any(k=='nominal' for k in kinds.values()):
        raise ValueError('FAMD_NOT_APPLICABLE_NO_NOMINAL_VARIABLE')
    blocks=[]
    for c in frame:
        if kinds[c]=='nominal':
            observed=frame[c].dropna().astype(str)
            if observed.empty:
                raise ValueError('FAMD_EMPTY_CATEGORICAL_BLOCK')
            s=frame[c].fillna(observed.mode().iloc[0]).astype(str)
            dummy=pd.get_dummies(s,dtype=float)
            frequencies=dummy.mean().to_numpy()
            blocks.append((dummy.to_numpy()-frequencies)/np.sqrt(frequencies))
        else:
            s=pd.to_numeric(frame[c],errors='raise')
            x=s.fillna(s.median()).to_numpy(float)
            std=x.std()
            if std>0: blocks.append(((x-x.mean())/std)[:,None])
    return np.column_stack(blocks)


def build_space(identifier, raw, families, config, registry, manifest, baseline, lab_groups, feature_to_lab):
    if identifier in legacy.SCENARIOS or identifier in {'R0','R1'}:
        scenario={'R0':'S3','R1':'S4'}.get(identifier,identifier)
        rep=legacy.fit_embedding(raw,families,scenario,config,lab_groups,feature_to_lab)
        x=rep.embedding
        return Space(raw,x[:,:2],x,pairwise_distances(x),rep.balance_audit,'PCA',list(raw),rep)
    if identifier=='R2':
        matrix,audit,features=hierarchical_mfa(raw,registry)
        x=reduce_matrix(matrix,config)
        return Space(raw[features],x[:,:2],x,pairwise_distances(x),audit,'hierarchical_clinical_MFA',features)
    frame,kinds=mixed_source_frame(raw,manifest,baseline)
    if identifier=='R3':
        d=gower_distance(frame,kinds)
        # Metric MDS is a lens only; DBSCAN receives the actual Gower matrix.
        mds_options={'metric':'precomputed','metric_mds':True,'init':'random'} if 'metric_mds' in inspect.signature(MDS).parameters else {'metric':True,'dissimilarity':'precomputed'}
        x=MDS(n_components=2,random_state=int(config['random_seed']),n_init=4,**mds_options).fit_transform(d)
        return Space(raw,x,d,d,pd.DataFrame(),'Gower_metric_MDS_lens',list(raw))
    if identifier=='R4':
        x=reduce_matrix(famd_matrix(frame,kinds),config)
        return Space(raw,x[:,:2],x,pairwise_distances(x),pd.DataFrame(),'FAMD',list(raw))
    raise ValueError('Unknown representation')

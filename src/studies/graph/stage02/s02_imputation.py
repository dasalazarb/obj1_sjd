"""Refit-local weighted low-rank reconstruction, not missMDA::imputeMFA.

Minimize a hierarchy-weighted observed squared residual with a nuclear-norm
penalty. Soft singular-value thresholding alternates with restoration of observed
cells. The rank cap is selected by artificial masking on held-out patients.
Nominal blocks are explicitly unsupported; no continuous category imputation.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .s02_missingness_registry import NotEstimable


@dataclass
class Reconstruction:
    completed: pd.DataFrame
    audit: dict
    validation: pd.DataFrame


def _numeric(raw, meta):
    from .s02_feature_registry import BASELINE
    for field in ('temporality','temporal_status'):
        if field in meta and not meta[field].isin(BASELINE).all():
            raise NotEstimable('IMPUTATION_POSTBASELINE_PREDICTOR_FORBIDDEN')
    from .._feature_catalog import OUTCOMES
    forbidden=OUTCOMES | {'pop','essdai','esspri','pop_status','essdai_total','esspri_total'}
    if 'source_column' in meta and meta.source_column.str.lower().isin(forbidden).any():
        raise NotEstimable('OUTCOME_PREDICTORS_FORBIDDEN')
    if 'role' in meta and meta.role.eq('post_baseline_outcome').any():
        raise NotEstimable('IMPUTATION_POSTBASELINE_PREDICTOR_FORBIDDEN')
    encoding=meta.get('encoding',pd.Series('numeric',index=meta.index)).astype(str)
    if encoding.str.contains('nominal|one_hot',regex=True).any():
        raise NotEstimable('ITERATIVE_NOMINAL_BLOCK_UNSUPPORTED_FIXED_PANEL_RETAINED')
    x=raw.to_numpy(float)
    if np.isinf(x).any(): raise ValueError('INFINITE_IMPUTATION_INPUT')
    if np.sum(np.isfinite(x),axis=0).min()<3:
        raise NotEstimable('INSUFFICIENT_OBSERVED_TRAINING_SOURCE_SUPPORT')
    return x


def hierarchy_weights(meta,reference,matrix):
    """Fit nested singular normalization solely on this refit's training data.

    Weights are frozen after median initialization so the regularized completion
    has one fixed weighted objective. Final MFA weights are fitted again on the
    completed refit, using the unchanged historical representation algorithm.
    """
    weights=np.ones(len(meta)); normalized=matrix.copy()
    def normalize(columns):
        indices=meta.index.get_indexer(columns)
        singular=np.linalg.svd(normalized[:,indices],compute_uv=False)
        sigma=float(singular[0]) if len(singular) else 0.
        if sigma>0:
            weights[indices]/=sigma; normalized[:,indices]/=sigma
    top='clinical_domain' if reference=='R2' else 'technical_family'
    if top not in meta: top='family'
    with threadpool_limits(limits=1):
        for _,domain in meta.groupby(top,sort=True):
            if reference=='R2':
                for _,sub in domain.groupby('subdomain',sort=True):
                    for _,construct in sub.groupby('derivation_group',sort=True):
                        normalize(construct.index)
                    normalize(sub.index)
            normalize(domain.index)
    if not np.isfinite(weights).all() or np.any(weights<=0):
        raise NotEstimable('INVALID_TRAINING_HIERARCHY_WEIGHTS')
    return weights


class LowRankImputer:
    def __init__(self,rank,policy,reference):
        self.rank=int(rank); self.policy=policy; self.reference=reference

    def fit_transform(self,raw,meta):
        x=_numeric(raw,meta); n,p=x.shape
        if x.size>int(self.policy.get('max_matrix_cells',2000000)):
            raise NotEstimable('IMPUTATION_MEMORY_BUDGET_EXCEEDED')
        if self.rank<1 or self.rank>=min(n,p):
            raise NotEstimable('IMPUTATION_RANK_EXCEEDS_TRAINING_SUPPORT')
        self.center=np.nanmedian(x,axis=0)
        self.scale=np.nanpercentile(x,75,axis=0)-np.nanpercentile(x,25,axis=0)
        self.scale=np.where(self.scale>0,self.scale,1.)
        standardized=(x-self.center)/self.scale
        self.weights=hierarchy_weights(meta,self.reference,np.where(np.isfinite(standardized),standardized,0.))
        self.levels={j:np.sort(raw.iloc[:,j].dropna().unique()) for j in range(raw.shape[1])}
        self.lower=np.nanmin(x,axis=0); self.upper=np.nanmax(x,axis=0)
        z=(x-self.center)/self.scale*self.weights
        observed=np.isfinite(z); filled=np.where(observed,z,0.)
        tolerance=float(self.policy['convergence_tolerance']); converged=observed.all()
        shrink=float(self.policy.get('singular_value_shrinkage',.1))
        if not 0<shrink<1: raise ValueError('INVALID_SINGULAR_SHRINKAGE')
        self.penalty=None
        with threadpool_limits(limits=1):
            for iteration in range(int(self.policy['max_iterations'])):
                u,s,v=np.linalg.svd(filled,full_matrices=False)
                if self.penalty is None: self.penalty=shrink*s[0]
                shrunk=np.maximum(s[:self.rank]-self.penalty,0.)
                reconstructed=(u[:,:self.rank]*shrunk)@v[:self.rank]
                updated=np.where(observed,z,reconstructed)
                change=np.linalg.norm((updated-filled)[~observed])/max(1.,np.linalg.norm(filled[~observed]))
                filled=updated
                if change<=tolerance:
                    converged=True; break
        if not converged: raise NotEstimable('ITERATIVE_NONCONVERGENCE')
        if not np.isfinite(filled).all(): raise NotEstimable('ITERATIVE_DIVERGENCE')
        self.loadings=v[:self.rank].T
        self.meta=meta.copy(); self.iterations=iteration+1
        completed=filled/self.weights*self.scale+self.center
        completed=self._constrain(completed)
        return pd.DataFrame(np.where(observed,x,completed),columns=raw.columns,index=raw.index)

    def _constrain(self,x):
        x=np.clip(x,self.lower,self.upper)
        for j,(_,row) in enumerate(self.meta.iterrows()):
            if 'binary' in str(row.get('encoding','')):
                x[:,j]=(x[:,j]>=.5).astype(float)
            elif str(row.get('encoding',''))=='ordinal':
                # Preserve reviewed encoded levels observed in the training fold.
                levels=self.levels[j]
                x[:,j]=levels[np.abs(x[:,j,None]-levels).argmin(axis=1)]
        return x

    def fit(self,raw,meta):
        self.levels={j:np.sort(raw.iloc[:,j].dropna().unique()) for j in range(raw.shape[1])}
        self.fit_transform(raw,meta)
        return self

    def transform(self,raw):
        x=raw.to_numpy(float); observed=np.isfinite(x)
        z=(x-self.center)/self.scale*self.weights
        result=np.empty_like(x)
        ridge=float(self.policy.get('score_ridge',.1))
        if ridge<=0: raise ValueError('INVALID_SCORE_RIDGE')
        for i in range(len(x)):
            if not observed[i].any(): raise NotEstimable('VALIDATION_PATIENT_WITHOUT_PREDICTORS')
            load=self.loadings[observed[i]]
            score=np.linalg.solve(load.T@load+ridge*np.eye(self.rank),load.T@z[i,observed[i]])
            result[i]=self.loadings@score
        result=self._constrain(result/self.weights*self.scale+self.center)
        return pd.DataFrame(np.where(observed,x,result),columns=raw.columns,index=raw.index)


def reconstruction_validation(raw,meta,seed,policy,reference):
    """Train on separate patients; hide observed source/instrument blocks in test.

    The held-out patient's remaining values estimate only that patient's scores;
    scalers, bounds, factors, weights and penalty are fitted on training patients.
    """
    rng=np.random.default_rng(seed); rows=[]; n=len(raw)
    if n<10: raise NotEstimable('INSUFFICIENT_PATIENTS_FOR_MASKED_VALIDATION')
    for repeat in range(int(policy['masked_validation_repeats'])):
        order=rng.permutation(n); cut=max(2,int(n*float(policy.get('validation_patient_fraction',.2))))
        train=raw.iloc[order[cut:]]; test=raw.iloc[order[:cut]].copy()
        hidden=np.zeros(test.shape,bool)
        # Alternate random source blocks and complete instrument blocks, with
        # independent patient holdout in both designs.
        grouping='instrument_id' if repeat%2 and 'instrument_id' in meta else 'source_column'
        for _,part in meta.groupby(grouping,sort=True):
            columns=raw.columns.get_indexer(part.index)
            chosen=rng.random(len(test))<float(policy['masked_fraction_per_replicate'])
            hidden[np.ix_(chosen,columns)]=True
        hidden &= test.notna().to_numpy()
        if not hidden.any():
            options=np.argwhere(test.notna().to_numpy())
            if not len(options): continue
            i,j=options[rng.integers(len(options))]; hidden[i,j]=True
        masked=test.mask(hidden)
        for rank in policy['candidate_rank']:
            status='COMPLETE'; reason=''; prediction=None; model=None
            try:
                model=LowRankImputer(rank,policy,reference).fit(train,meta)
                prediction=model.transform(masked)
            except NotEstimable as exc: status='NOT_ESTIMABLE'; reason=str(exc)
            for j,c in enumerate(raw):
                ids=hidden[:,j]
                if not ids.any(): continue
                error=np.nan
                if prediction is not None:
                    error=float(np.mean(np.abs(prediction[c].to_numpy()[ids]-test[c].to_numpy()[ids]))/model.scale[j])
                rows.append(dict(repeat=repeat,rank=int(rank),feature=c,source_column=meta.loc[c,'source_column'],
                    clinical_domain=meta.loc[c,'clinical_domain'],split=grouping+'_mask_patient_holdout',
                    n_train_patients=len(train),n_validation_patients=len(test),n_hidden=int(ids.sum()),normalized_mae=error,
                    status=status,reason=reason,iterations=model.iterations if prediction is not None else np.nan,
                    mechanism='artificial_MCAR_mask_not_inference_about_real_missingness'))
    return pd.DataFrame(rows)


def fit_transform(train_data,feature_metadata,seed,policy):
    meta=feature_metadata.loc[list(train_data)]
    if (int(policy['max_iterations'])<1 or float(policy['convergence_tolerance'])<=0
            or int(policy['masked_validation_repeats'])<1
            or not 0<float(policy['masked_fraction_per_replicate'])<1
            or not policy['candidate_rank']):
        raise ValueError('INVALID_ITERATIVE_POLICY')
    if train_data.size>int(policy.get('max_matrix_cells',2000000)):
        raise NotEstimable('IMPUTATION_MEMORY_BUDGET_EXCEEDED')
    _numeric(train_data,meta)
    reference=policy.get('reference_id','R0')
    validation=reconstruction_validation(train_data,meta,seed,policy,reference)
    if validation.empty: raise NotEstimable('NO_ARTIFICIAL_MASKING_SUPPORT')
    valid=[]
    for rank,part in validation.groupby('rank'):
        if part.status.eq('COMPLETE').all() and part.normalized_mae.notna().all():
            valid.append((float(np.average(part.normalized_mae,weights=part.n_hidden)),int(rank)))
    if not valid: raise NotEstimable('NO_CONVERGED_VALIDATED_IMPUTATION_RANK')
    error,rank=min(valid)
    model=LowRankImputer(rank,policy,reference)
    model.levels={j:np.sort(train_data.iloc[:,j].dropna().unique()) for j in range(train_data.shape[1])}
    completed=model.fit_transform(train_data,meta)
    validation['selected_rank']=rank
    return Reconstruction(completed,dict(rank=rank,normalized_mae=error,iterations=model.iterations,
        converged=True,objective='hierarchy_weighted_observed_L2_plus_nuclear_penalty',
        singular_penalty=model.penalty,hierarchy_weights='training_only_nested_singular_normalization',estimator='weighted_low_rank_v1_not_imputeMFA'),validation)

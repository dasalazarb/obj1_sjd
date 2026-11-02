"""Refit-local weighted low-rank reconstruction, not missMDA::imputeMFA.

Minimize a hierarchy-weighted observed squared residual with a nuclear-norm
penalty. Soft singular-value thresholding alternates with restoration of observed
cells. The rank cap is selected by artificial masking on held-out patients.
Nominal sources use a separate chained ridge/logistic estimator, never ordinal codes.
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


def _guard(raw, meta):
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


def _numeric(raw, meta):
    _guard(raw,meta)
    encoding=meta.get('encoding',pd.Series('numeric',index=meta.index)).astype(str)
    if encoding.str.contains('nominal|one_hot',regex=True).any():
        raise NotEstimable('ITERATIVE_NOMINAL_BLOCK_UNSUPPORTED_FIXED_PANEL_RETAINED')
    x=raw.to_numpy(float)
    if np.isinf(x).any(): raise ValueError('INFINITE_IMPUTATION_INPUT')
    if np.sum(np.isfinite(x),axis=0).min()<3:
        raise NotEstimable('INSUFFICIENT_OBSERVED_TRAINING_SOURCE_SUPPORT')
    for j,(column,row) in enumerate(meta.iterrows()):
        levels=np.unique(x[np.isfinite(x[:,j]),j])
        if len(levels)<2: raise NotEstimable('SOURCE_WITHOUT_TRAINING_VARIATION:'+str(row.source_column))
        if 'binary' in str(row.encoding) and not np.isin(levels,[0,1]).all(): raise NotEstimable('INVALID_BINARY_ENCODING')
        if str(row.encoding)=='ordinal' and not str(column).endswith('__ordinal'): raise NotEstimable('UNVERIFIED_ORDINAL_ORDER')
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
        self.diagnostics=dict(iterations_run=0, convergence_status="NOT_STARTED", final_relative_change=np.nan,
            observed_loss=np.nan, penalty=np.nan, scale_diagnostic="not_fitted")
        x=_numeric(raw,meta); n,p=x.shape
        self.columns=list(raw)
        if x.size>int(self.policy.get('max_matrix_cells',2000000)):
            raise NotEstimable('IMPUTATION_MEMORY_BUDGET_EXCEEDED')
        if self.rank<1 or self.rank>=min(n,p):
            raise NotEstimable('IMPUTATION_RANK_EXCEEDS_TRAINING_SUPPORT')
        self.center=np.nanmedian(x,axis=0)
        self.scale=np.nanpercentile(x,75,axis=0)-np.nanpercentile(x,25,axis=0)
        zero_iqr=self.scale<=0
        fallback=np.nanstd(x,axis=0)
        self.scale=np.where(self.scale>0,self.scale,np.where(fallback>0,fallback,1.))
        self.diagnostics["scale_diagnostic"]=f"zero_iqr={zero_iqr.sum()};training_std_fallback"
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
        changes=[]
        with threadpool_limits(limits=1):
            for iteration in range(int(self.policy['max_iterations'])):
                u,s,v=np.linalg.svd(filled,full_matrices=False)
                if self.penalty is None: self.penalty=shrink*s[0]
                shrunk=np.maximum(s[:self.rank]-self.penalty,0.)
                reconstructed=(u[:,:self.rank]*shrunk)@v[:self.rank]
                updated=np.where(observed,z,reconstructed)
                change=np.linalg.norm((updated-filled)[~observed])/max(1.,np.linalg.norm(filled[~observed]))
                filled=updated
                changes.append(float(change))
                self.diagnostics.update(iterations_run=iteration+1,final_relative_change=float(change),
                    observed_loss=float(np.mean((reconstructed[observed]-z[observed])**2)),penalty=float(self.penalty),
                    convergence_status='ITERATING')
                if not np.isfinite(filled).all() or not np.isfinite(change):
                    self.diagnostics['convergence_status']='DIVERGED'
                    raise NotEstimable('ITERATIVE_DIVERGENCE')
                if change<=tolerance:
                    converged=True; break
        if not converged:
            plateau=len(changes)>=20 and np.ptp(changes[-20:])<=tolerance
            reason='NONCONVERGENCE_PLATEAU' if plateau else 'ITERATIVE_NONCONVERGENCE'
            self.diagnostics['convergence_status']=reason
            raise NotEstimable(reason)
        self.diagnostics['convergence_status']='CONVERGED'
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
        if list(raw)!=self.columns: raise ValueError('FIXED_PANEL_MISMATCH')
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



def source_groups(meta):
    """Nominal one-hot levels are one target; numeric encoded columns stay numeric."""
    for source,part in meta.groupby('source_column',sort=True):
        nominal=part.encoding.astype(str).str.contains('nominal|one_hot',regex=True).any()
        if nominal:
            yield source,list(part.index),'nominal'
        else:
            for column,row in part.iterrows():
                encoding=str(row.get('encoding','numeric'))
                kind='binary' if 'binary' in encoding else 'ordinal' if encoding=='ordinal' else 'continuous'
                yield source,[column],kind


class MixedIterativeImputer:
    """Deterministic chained ridge/softmax models on source-level categorical targets.

    Predictors retain their one-hot columns. Missingness indicators are not added.
    Training-only median/mode initialization, standardization and model fits are
    frozen for held-out patients. Nominal targets never enter a regression as codes.
    """
    def __init__(self,policy,seed):
        self.policy=policy; self.seed=int(seed)
        self.diagnostics=dict(iterations_run=0,convergence_status='NOT_STARTED',final_relative_change=np.nan,
            observed_loss=np.nan,penalty=policy.get('mixed_ridge',1.),scale_diagnostic='training_only_standard_scaler')

    def _constrain(self,values,original,columns,kind,levels=None):
        if kind=='nominal':
            block=original[columns].to_numpy(float)
            allowed=~(np.isfinite(block)&(block==0))
            active=np.isfinite(block)&(block==1)
            allowed=np.where(active.any(axis=1)[:,None],active,allowed)
            if not allowed.any(axis=1).all(): raise NotEstimable('NOMINAL_PARTIAL_ENCODING_INCONSISTENT')
            values=np.asarray(values,float)
            choice=np.argmax(np.where(allowed,values,-np.inf),axis=1)
            return np.eye(len(columns))[choice]
        if kind in {'ordinal','binary'}:
            return levels[np.abs(values[:,0,None]-levels).argmin(axis=1),None]
        return np.clip(values,self.bounds[columns[0]][0],self.bounds[columns[0]][1])

    def _initial(self,raw):
        if list(raw)!=self.columns: raise ValueError('FIXED_PANEL_MISMATCH')
        x=raw.copy().astype(float)
        for source,columns,kind in self.groups:
            block=raw[columns].to_numpy(float)
            if kind=='nominal':
                finite=np.isfinite(block)
                if not np.isin(block[finite],[0,1]).all() or (np.nansum(block,axis=1)>1).any():
                    raise NotEstimable('INVALID_NOMINAL_ENCODING:'+str(source))
                complete=finite.all(axis=1)
                if (block[complete].sum(axis=1)!=1).any(): raise NotEstimable('INVALID_NOMINAL_ENCODING:'+str(source))
                guesses=np.tile(self.initial[columns[0]],(len(raw),1))
                completed=self._constrain(guesses,raw,columns,kind)
                x[columns]=np.where(finite,block,completed)
            else:
                x[columns]=x[columns].fillna(self.initial[columns[0]])
        return x

    def fit_transform(self,raw,meta):
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        _guard(raw,meta)
        if raw.size>int(self.policy.get('max_matrix_cells',2000000)): raise NotEstimable('IMPUTATION_MEMORY_BUDGET_EXCEEDED')
        if np.isinf(raw.to_numpy(float)).any(): raise ValueError('INFINITE_IMPUTATION_INPUT')
        self.columns=list(raw); self.groups=list(source_groups(meta)); self.initial={}; self.bounds={}; self.levels={}
        minimum=int(self.policy.get('minimum_category_support',3))
        for source,columns,kind in self.groups:
            block=raw[columns]; complete=block.notna().all(axis=1); observed=block.loc[complete]
            if len(observed)<3: raise NotEstimable('INSUFFICIENT_OBSERVED_TRAINING_SOURCE_SUPPORT:'+str(source))
            if kind=='nominal':
                counts=observed.sum(axis=0).to_numpy()
                if (counts<minimum).any(): raise NotEstimable('NOMINAL_CATEGORY_WITHOUT_TRAINING_SUPPORT:'+str(source))
                self.initial[columns[0]]=counts/counts.sum()
            else:
                values=observed.iloc[:,0].to_numpy(); levels=np.unique(values)
                if len(levels)<2: raise NotEstimable('SOURCE_WITHOUT_TRAINING_VARIATION:'+str(source))
                if kind=='ordinal' and not columns[0].endswith('__ordinal'): raise NotEstimable('UNVERIFIED_ORDINAL_ORDER')
                if kind=='binary' and not np.isin(levels,[0,1]).all(): raise NotEstimable('INVALID_BINARY_ENCODING')
                if kind=='binary' and observed.iloc[:,0].value_counts().min()<minimum:
                    raise NotEstimable('BINARY_CATEGORY_WITHOUT_TRAINING_SUPPORT:'+str(source))
                self.levels[columns[0]]=levels
                self.initial[columns[0]]=float(observed.iloc[:,0].mode().iloc[0] if kind in {'binary','ordinal'} else np.median(values))
                self.bounds[columns[0]]=(values.min(),values.max())
        filled=self._initial(raw); observed=raw.notna().to_numpy(); self.rounds=[]
        scales=np.nanstd(raw.to_numpy(float),axis=0); scales=np.where(scales>0,scales,1.)
        numeric=[self.columns.index(c) for _,cols,k in self.groups if k=='continuous' for c in cols]
        for iteration in range(int(self.policy['max_iterations'])):
            before=filled.to_numpy().copy(); models=[]
            for source,columns,kind in self.groups:
                predictors=[c for c in self.columns if c not in columns]
                measured=raw[columns].notna().all(axis=1).to_numpy()
                if not predictors: raise NotEstimable('NO_INDEPENDENT_IMPUTATION_PREDICTORS')
                if kind in {'nominal','binary'}:
                    y=raw.loc[measured,columns].to_numpy().argmax(axis=1) if kind=='nominal' else raw.loc[measured,columns[0]].to_numpy()
                    estimator=LogisticRegression(C=float(self.policy.get('mixed_logistic_c',1.)),solver='lbfgs',max_iter=500,random_state=self.seed)
                else:
                    y=raw.loc[measured,columns[0]].to_numpy()
                    estimator=Ridge(alpha=float(self.policy.get('mixed_ridge',1.)))
                model=make_pipeline(StandardScaler(),estimator)
                with threadpool_limits(limits=1): model.fit(filled.loc[measured,predictors],y)
                if getattr(estimator,'n_iter_',None) is not None and np.max(estimator.n_iter_)>=500:
                    raise NotEstimable('MIXED_CLASSIFIER_NONCONVERGENCE:'+str(source))
                models.append((columns,kind,predictors,model))
                self._update(filled,raw,columns,kind,predictors,model)
            self.rounds.append(models)
            after=filled.to_numpy(); missing=~observed
            delta=(after-before)/scales
            change=float(np.linalg.norm(delta[missing])/max(1.,np.linalg.norm((before/scales)[missing])))
            categorical=[self.columns.index(c) for _,cols,k in self.groups if k!='continuous' for c in cols]
            category_stable=np.array_equal(before[:,categorical],after[:,categorical])
            numeric_change=np.linalg.norm(delta[:,numeric][missing[:,numeric]]) if numeric else 0.
            self.diagnostics.update(iterations_run=iteration+1,final_relative_change=change,convergence_status='ITERATING')
            if category_stable and (change<=float(self.policy['convergence_tolerance']) or numeric_change==0):
                self.diagnostics['convergence_status']='CONVERGED'; self.iterations=iteration+1
                return filled
        self.diagnostics['convergence_status']='ITERATIVE_NONCONVERGENCE'
        raise NotEstimable('MIXED_ITERATIVE_NONCONVERGENCE')

    def _update(self,filled,raw,columns,kind,predictors,model):
        with threadpool_limits(limits=1):
            if kind=='nominal':
                predicted=model.predict_proba(filled[predictors])
            else:
                predicted=model.predict(filled[predictors])[:,None]
        values=self._constrain(predicted,raw,columns,kind,self.levels.get(columns[0]))
        original=raw[columns].to_numpy(float)
        filled[columns]=np.where(np.isfinite(original),original,values)

    def fit(self,raw,meta):
        self.fit_transform(raw,meta)
        return self

    def transform(self,raw):
        filled=self._initial(raw)
        if raw.notna().sum(axis=1).eq(0).any(): raise NotEstimable('VALIDATION_PATIENT_WITHOUT_PREDICTORS')
        # Replay the frozen training chain. No fitted state changes on test.
        for models in self.rounds:
            for columns,kind,predictors,model in models:
                self._update(filled,raw,columns,kind,predictors,model)
        return filled


def _mask_blocks(frame,meta,rng,fraction,grouping):
    hidden=np.zeros(frame.shape,bool)
    for _,part in meta.groupby(grouping,sort=True):
        columns=frame.columns.get_indexer(part.index)
        chosen=rng.random(len(frame))<fraction
        hidden[np.ix_(chosen,columns)]=True
    hidden &= frame.notna().to_numpy()
    # Do not invent an evaluation cell when the declared mask has no support.
    return frame.mask(hidden),hidden


def reconstruction_validation(raw,meta,seed,policy,reference,evaluation_data=None):
    """Every source/rank/repeat is recorded, including failed fits and empty masks."""
    rng=np.random.default_rng(seed); rows=[]; n=len(raw)
    if n<10: raise NotEstimable('INSUFFICIENT_PATIENTS_FOR_MASKED_VALIDATION')
    mixed=meta.encoding.astype(str).str.contains('nominal|one_hot',regex=True).any()
    ranks=[0] if mixed else policy['candidate_rank']
    for repeat in range(int(policy['masked_validation_repeats'])):
        order=rng.permutation(n); cut=max(2,int(n*float(policy.get('validation_patient_fraction',.2))))
        train=raw.iloc[order[cut:]] if evaluation_data is None else raw
        test=raw.iloc[order[:cut]].copy() if evaluation_data is None else evaluation_data.copy()
        grouping='instrument_id' if repeat%2 and 'instrument_id' in meta else 'source_column'
        masked,hidden=_mask_blocks(test,meta,rng,float(policy['masked_fraction_per_replicate']),grouping)
        train,_=_mask_blocks(train,meta,rng,float(policy.get('training_mask_fraction',0.)),grouping)
        for rank in ranks:
            status='COMPLETE'; reason=''; prediction=None
            model=MixedIterativeImputer(policy,seed+repeat) if mixed else LowRankImputer(rank,policy,reference)
            try:
                model.fit(train,meta)
                prediction=model.transform(masked)
            except NotEstimable as exc:
                status='NOT_ESTIMABLE'; reason=str(exc)
            for source,columns,kind in source_groups(meta):
                js=raw.columns.get_indexer(columns)
                ids=hidden[:,js].all(axis=1)&test[columns].notna().all(axis=1).to_numpy()
                count=int(ids.sum()); error=baseline=accuracy=balanced=baseline_accuracy=baseline_balanced=np.nan
                row_status=status if count else 'NOT_ESTIMABLE'
                row_reason=reason if count else 'NO_HIDDEN_EVALUABLE_CELLS'
                if prediction is not None and count:
                    actual=test.loc[ids,columns].to_numpy(); predicted=prediction.loc[ids,columns].to_numpy()
                    if kind in {'nominal','binary'}:
                        actual=actual.argmax(axis=1) if kind=='nominal' else actual[:,0]
                        predicted=predicted.argmax(axis=1) if kind=='nominal' else predicted[:,0]
                        observed=train[columns].dropna()
                        levels=observed.to_numpy().argmax(axis=1) if kind=='nominal' else observed.iloc[:,0].to_numpy()
                        mode=pd.Series(levels).mode().iloc[0]
                        accuracy=float(np.mean(actual==predicted)); balanced=balanced_accuracy(actual,predicted)
                        baseline_accuracy=float(np.mean(actual==mode))
                        baseline_balanced=balanced_accuracy(actual,np.full(len(actual),mode))
                    else:
                        c=columns[0]; observed=train[c].dropna().to_numpy()
                        scale=np.percentile(observed,75)-np.percentile(observed,25)
                        if scale<=0: scale=np.std(observed)
                        if scale<=0: scale=1.
                        if kind=='ordinal': scale=1.  # Error on reviewed encoded levels.
                        error=float(np.mean(np.abs(predicted[:,0]-actual[:,0]))/scale)
                        center=pd.Series(observed).mode().iloc[0] if kind=='ordinal' else np.median(observed)
                        baseline=float(np.mean(np.abs(actual[:,0]-center))/scale)
                diag=getattr(model,'diagnostics',{})
                rows.append(dict(reference_id=reference,repeat=repeat,rank=int(rank),feature='|'.join(columns),source_column=source,
                    clinical_domain=meta.loc[columns[0]].get('clinical_domain','UNKNOWN'),variable_type=kind,
                    split=grouping+'_mask_patient_holdout',n_train=len(train),n_test=len(test),
                    n_train_patients=len(train),n_validation_patients=len(test),n_hidden=count,
                    n_observed_per_feature=json_counts(train[columns]),
                    normalized_mae=error,baseline_mae=baseline,accuracy=accuracy,balanced_accuracy=balanced,
                    baseline_accuracy=baseline_accuracy,baseline_balanced_accuracy=baseline_balanced,class_support=json_counts(test.loc[ids,columns],kind),
                    status=row_status,reason=row_reason,reason_code=row_reason,seed=seed,model_seed=seed+repeat,
                    rng_policy='sequential_repeat_stream_from_seed',
                    iterations=getattr(model,'iterations',np.nan),**diag,
                    mechanism='artificial_MCAR_mask_not_inference_about_real_missingness'))
    return pd.DataFrame(rows)


def balanced_accuracy(actual,predicted):
    # Mean recall on the explicitly reported test classes, including single-class folds.
    return float(np.mean([np.mean(predicted[actual==level]==level) for level in np.unique(actual)]))


def json_counts(frame,kind=None):
    import json
    if kind=='nominal': counts=frame.sum(axis=0).astype(int).to_dict()
    elif kind=='binary': counts={str(k):int(v) for k,v in frame.iloc[:,0].value_counts().items()}
    elif kind is not None: counts={}
    else: counts=frame.notna().sum().to_dict()
    return json.dumps(counts,sort_keys=True)


def rank_summary(validation,repeats):
    rows=[]
    for rank,part in validation.groupby('rank'):
        evaluated=part.loc[part.n_hidden.gt(0)]
        passed=evaluated.loc[evaluated.status.eq('COMPLETE')]
        valid_repeats=sum(p.status.eq('COMPLETE').all() and p.n_hidden.sum()>0 for _,p in evaluated.groupby('repeat'))
        excluded=valid_repeats!=repeats or not evaluated.status.eq('COMPLETE').all() or evaluated.empty
        numeric=passed.dropna(subset=['normalized_mae'])
        rows.append(dict(rank=rank,n_requested=repeats,n_successful=valid_repeats,n_hidden=int(passed.n_hidden.sum()),
            normalized_mae=float(np.average(numeric.normalized_mae,weights=numeric.n_hidden)) if len(numeric) else np.nan,
            nonconvergence_rate=float(evaluated.convergence_status.astype(str).str.contains('NONCONVERGENCE|PLATEAU').mean()) if len(evaluated) else np.nan,
            status='NOT_ESTIMABLE' if excluded else 'COMPLETE',
            reason='|'.join(sorted(set(part.loc[part.status.ne('COMPLETE'),'reason'].dropna()))) if excluded else ''))
    return pd.DataFrame(rows)


def fit_transform(train_data,feature_metadata,seed,policy):
    meta=feature_metadata.loc[list(train_data)]
    if (int(policy['max_iterations'])<1 or float(policy['convergence_tolerance'])<=0
            or int(policy['masked_validation_repeats'])<1 or not 0<float(policy['masked_fraction_per_replicate'])<1
            or not policy['candidate_rank']): raise ValueError('INVALID_ITERATIVE_POLICY')
    if train_data.size>int(policy.get('max_matrix_cells',2000000)): raise NotEstimable('IMPUTATION_MEMORY_BUDGET_EXCEEDED')
    _guard(train_data,meta)
    mixed=meta.encoding.astype(str).str.contains('nominal|one_hot',regex=True).any()
    reference=policy.get('reference_id','R0')
    if mixed:
        selection=pd.DataFrame()
        validation=reconstruction_validation(train_data,meta,seed,policy,reference)
    else:
        # Lock rank using only development patients. Outer test patients cannot
        # alter the fitted rank, scalers, hyperparameters or training predictions.
        order=np.random.default_rng(seed).permutation(len(train_data))
        cut=max(2,int(len(train_data)*float(policy.get('validation_patient_fraction',.2))))
        development=train_data.iloc[order[cut:]]; heldout=train_data.iloc[order[:cut]]
        selection=reconstruction_validation(development,meta,seed+1000,policy,reference)
        selection['validation_stage']='rank_selection'
        chosen=rank_summary(selection,int(policy['masked_validation_repeats']))
        accepted=chosen.loc[chosen.status.eq('COMPLETE')]
        if accepted.empty:
            exc=NotEstimable('NO_CONVERGED_VALIDATED_IMPUTATION_RANK')
            exc.validation=selection; exc.rank_summary=chosen
            raise exc
        selected=int(accepted.sort_values(['normalized_mae','rank']).iloc[0]['rank'])
        validation=reconstruction_validation(development,meta,seed+10000,{**policy,'candidate_rank':[selected]},reference,evaluation_data=heldout)
    validation['validation_stage']='evaluation'
    summary=rank_summary(validation,int(policy['masked_validation_repeats']))
    valid=summary.loc[summary.status.eq('COMPLETE')]
    if valid.empty:
        exc=NotEstimable('NO_CONVERGED_VALIDATED_IMPUTATION_RANK')
        exc.validation=pd.concat([selection,validation],ignore_index=True); exc.rank_summary=summary
        raise exc
    # Mixed has no rank and no cross-type optimization; numeric ranks use masked MAE.
    rank=0 if mixed else int(valid.sort_values(['normalized_mae','rank']).iloc[0]['rank'])
    model=MixedIterativeImputer(policy,seed) if mixed else LowRankImputer(rank,policy,reference)
    try:
        completed=model.fit_transform(train_data,meta)
    except NotEstimable as exc:
        exc.validation=pd.concat([selection,validation],ignore_index=True); exc.rank_summary=summary
        raise
    observed=train_data.notna().to_numpy()
    if list(completed)!=list(train_data) or not np.array_equal(completed.to_numpy()[observed],train_data.to_numpy()[observed]):
        raise ValueError('IMPUTATION_OBSERVED_OR_PANEL_CHANGED')
    validation=pd.concat([selection,validation],ignore_index=True)
    validation['selected_rank']=rank
    estimator='mixed_iterative_v1_ridge_softmax' if mixed else 'weighted_low_rank_v2_not_imputeMFA'
    audit=dict(rank=rank,iterations=model.iterations,converged=True,estimator=estimator,
        objective='chained_source_ridge_softmax' if mixed else 'hierarchy_weighted_observed_L2_plus_nuclear_penalty',
        hierarchy_weights='not_applicable_mixed' if mixed else 'training_only_nested_singular_normalization',
        singular_penalty=getattr(model,'penalty',np.nan),n_source_variables=meta.source_column.nunique(),
        n_encoded_features=len(meta),n_nominal_sources=sum(k=='nominal' for _,_,k in source_groups(meta)),
        n_observed_preserved=int(train_data.notna().to_numpy().sum()),seed=seed,**model.diagnostics)
    return Reconstruction(completed,audit,validation)

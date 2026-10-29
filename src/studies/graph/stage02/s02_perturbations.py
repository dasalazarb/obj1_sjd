"""Observation-mask-preserving nulls and explicit perturbation plans."""
import numpy as np
import pandas as pd


def permute_observed(raw,seed):
    result=raw.copy(); rng=np.random.default_rng(seed)
    for c in result:
        observed=result[c].notna()
        result.loc[observed,c]=rng.permutation(result.loc[observed,c].to_numpy())
    return result


def mask_control(raw,baseline,admin_columns):
    result=raw.isna().astype(float).add_prefix('missing__')
    for c in admin_columns:
        if c in baseline:
            dummy=pd.get_dummies(baseline[c].astype('string'),prefix=c,dummy_na=True,dtype=float)
            result=pd.concat([result,dummy],axis=1)
    return result.loc[:,result.nunique()>1]

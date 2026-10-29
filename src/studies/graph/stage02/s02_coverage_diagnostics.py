"""Operational coverage, local support, and patient-bootstrap descriptive effects."""
import numpy as np
import pandas as pd
from .s02_missingness import supported_hard


def diagnose_uncovered(space, mapped, patient_ids, observation):
    n=len(space.raw)
    if len(patient_ids)!=n or len(observation)!=n: raise ValueError('PATIENT_UNIVERSE_MISMATCH')
    d=space.distance.copy(); np.fill_diagonal(d,np.inf)
    ordered=np.sort(d,axis=1); k=min(5,n-1)
    covered=np.asarray(mapped['covered'],bool); valid=supported_hard(mapped)
    table=pd.DataFrame(dict(patient_id=list(patient_ids),patient_index=np.arange(n),mapper_covered=covered,
        hard_supported=valid,membership_tie=mapped['ties'],community_unsupported=covered & ~mapped['ties'] & ~valid,
        nearest_neighbor_distance=ordered[:,0],knn_mean_distance=ordered[:,:k].mean(axis=1)))
    table['local_density_percentile']=table.knn_mean_distance.rank(pct=True,ascending=False)
    table['distance_to_covered']=np.min(d[:,covered],axis=1) if covered.any() else np.nan
    table['relative_distance_range']=np.divide(ordered[:,n-2]-ordered[:,0],ordered[:,n-2],out=np.zeros(n),where=ordered[:,n-2]>0)
    audit=pd.DataFrame(mapped.get('coverage_audit',[]))
    if audit.empty:
        table['coverage_reason']=np.where(covered,'covered','reason_unresolved')
    else: table=table.merge(audit,on='patient_index',validate='one_to_one')
    if 'eps' in table:
        table['n_neighbors_at_eps']=np.sum(d<=table.eps.to_numpy()[:,None],axis=1)
    table=pd.concat([table,observation.reset_index(drop=True)],axis=1)
    summary=table.groupby('coverage_reason',dropna=False).size().rename('n_patients').reset_index()
    summary['n_total']=n; summary['fraction_of_total']=summary.n_patients/n
    summary['status']=np.where(summary.coverage_reason.eq('reason_unresolved'),'NOT_ESTIMABLE','COMPLETE')
    return table,summary


def compare_groups(table, seed, replicates=199, minimum=5):
    rows=[]; covered=table.mapper_covered.to_numpy(bool); rng=np.random.default_rng(seed)
    columns=[c for c in table.select_dtypes(include='number') if c not in {'patient_index'}]
    for column in columns:
        a=table.loc[covered,column].dropna().to_numpy(float); b=table.loc[~covered,column].dropna().to_numpy(float)
        common=dict(metric=column,n_covered=len(a),n_uncovered=len(b),n_total=len(table),
            effect_definition='mean_uncovered_minus_covered',uncertainty='independent_patient_bootstrap_95_percentile')
        if min(len(a),len(b))<minimum:
            rows.append({**common,'status':'NOT_ESTIMABLE','reason':'INSUFFICIENT_GROUP_SUPPORT'}); continue
        effects=[rng.choice(b,len(b),replace=True).mean()-rng.choice(a,len(a),replace=True).mean() for _ in range(replicates)]
        rows.append({**common,'status':'COMPLETE','reason':'','mean_covered':float(a.mean()),'mean_uncovered':float(b.mean()),
            'effect':float(b.mean()-a.mean()),'ci_low':float(np.quantile(effects,.025)),'ci_high':float(np.quantile(effects,.975))})
    return pd.DataFrame(rows)

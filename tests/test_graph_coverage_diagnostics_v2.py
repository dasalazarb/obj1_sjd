"""Exact coverage reasons and separated Mapper/assignment denominators."""
import numpy as np
import pandas as pd
from src.studies.graph.stage02.s02_coverage_diagnostics import diagnose_uncovered,compare_groups
from src.studies.graph.stage02.s02_representations import Space
from src.studies.graph.stage02.s02_mapper_engine import analyze
from src.studies.graph import _pipeline as pipe


def test_159_patient_snapshot_counts_are_derived_and_complete():
    n=159; x=np.arange(n,dtype=float)[:,None]; d=np.abs(x-x.T)
    raw=pd.DataFrame({'a':x[:,0]}); space=Space(raw,np.column_stack([x,x]),x,d,pd.DataFrame(),'synthetic',['a'])
    covered=np.arange(n)<118; hard=np.where(covered,np.arange(n)%2,np.nan)
    mapped=dict(covered=covered,hard=hard,ties=np.zeros(n,bool),supported=np.array([True,True]),coverage_audit=[
        dict(patient_index=i,coverage_reason='covered' if covered[i] else 'all_cubes_below_min_samples') for i in range(n)])
    table,summary=diagnose_uncovered(space,mapped,[f'synthetic_{i}' for i in range(n)],pd.DataFrame({'observed_fraction':np.linspace(.5,1,n)}))
    assert int(table.mapper_covered.sum())==118 and int((~table.mapper_covered).sum())==41
    assert summary.n_patients.sum()==159 and summary.fraction_of_total.sum()==1
    groups=compare_groups(table,711,replicates=19)
    row=groups.loc[groups.metric.eq('observed_fraction')].iloc[0]
    assert row.effect>0 and row.n_uncovered==41 and row.n_covered==118


def test_exact_cover_audit_retains_isolated_noise_without_forced_assignment():
    config=pipe.load_pipeline()['sensitivity']; config['mapper'].update(n_cubes=2,min_samples=3,minimum_node_support=3)
    x=np.array([[0.,0.],[.01,.02],[.02,.01],[10.,10.]])
    d=np.sqrt(((x[:,None]-x[None,:])**2).sum(axis=2)); raw=pd.DataFrame(x)
    mapped=analyze(Space(raw,x,x,d,pd.DataFrame(),'synthetic',list(raw)),config,audit=True)
    audit=mapped['coverage_audit']
    assert len(audit)==4 and audit[-1]['coverage_reason']!='covered'
    assert not mapped['covered'][-1] and np.isnan(mapped['hard'][-1])
    assert audit[-1]['coverage_reason'] in {'all_cubes_below_min_samples','dbscan_noise_in_all_eligible_cubes'}

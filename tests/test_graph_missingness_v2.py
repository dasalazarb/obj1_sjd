"""Observation units, control isolation, valid comparison denominators and CV."""
import numpy as np
import pandas as pd
import pytest
from src.studies.graph.stage02.s02_missingness import (build_observation_mask,run_missingness_only_mapper,
    compare_coverage,administrative_frame,administrative_strata,mask_permutation_test,fit_adjusted_observation_model)


def inputs():
    raw=pd.DataFrame({'sex_a':[1.,0.,np.nan,1.], 'sex_b':[0.,1.,np.nan,0.],
                      'value':[1.,2.,np.nan,4.], 'threshold':[0.,0.,np.nan,1.], 'other':[np.nan,2.,1.,np.nan]})
    registry=pd.DataFrame({'feature':raw.columns,'source_column':['sex','sex','value','threshold','other'],
        'derivation_group':['sex','sex','assay','assay','other'],'technical_family':['EXT']*5,
        'clinical_domain':['D01']*5,'instrument_id':['x']*5,'subdomain':['x']*5})
    return raw,registry


def test_onehot_and_derivatives_have_single_observation_unit():
    raw,registry=inputs(); original=raw.copy()
    mask,manifest=build_observation_mask(raw,registry)
    assert mask.shape==(4,3) and set(np.unique(mask))=={0,1}
    assert mask['observed__sex'].tolist()==[1,1,0,1]
    assert manifest.source_column.is_unique
    pd.testing.assert_frame_equal(raw,original)


def test_pure_mask_and_admin_controls_are_separate_and_balanced():
    raw,registry=inputs(); mask,manifest=build_observation_mask(raw,registry)
    admin=pd.DataFrame({'demo__age_at_baseline':[30.,40.,50.,60.]})
    a=run_missingness_only_mapper(mask,manifest,'N2a')
    b=run_missingness_only_mapper(mask,manifest,'N2b',admin,{'demo__age_at_baseline':'numeric'})
    c=run_missingness_only_mapper(mask,manifest,'N2c',admin,{'demo__age_at_baseline':'numeric'})
    assert all(x.startswith('observed__') for x in a.raw)
    assert list(b.raw)==['demo__age_at_baseline']
    assert np.allclose(c.distance,(a.distance+b.distance)/2)
    assert c.weights.weight_applied.tolist()==[.5,.5]
    with pytest.raises(ValueError,match='VERIFIED_ADMINISTRATIVE'): run_missingness_only_mapper(mask,manifest,'N2b')
    with pytest.raises(ValueError,match='VARIABILITY'): run_missingness_only_mapper(mask*0+1,manifest,'N2a')


def mapped(n,covered,hard_count):
    hard=np.full(n,np.nan); hard[:hard_count]=np.arange(hard_count)%2
    return dict(covered=np.arange(n)<covered,hard=hard,ties=np.zeros(n,bool),supported=np.array([True,True]),
        community_sets=[set(np.arange(hard_count)[np.arange(hard_count)%2==c]) for c in range(2)])


def test_69_common_of_159_never_counts_uncovered_as_a_class():
    a=mapped(159,118,100); b=mapped(159,120,69)
    row=compare_coverage(a,b)
    assert row['n_common']==row['n_hard_both']==69
    assert row['fraction_of_total']==pytest.approx(69/159)
    assert row['n_covered_both']==118 and row['ari_primary']==1
    b['supported'][1]=False
    row=compare_coverage(a,b)
    assert row['n_common']==35 and np.isnan(row['ari_primary'])
    assert row['status']=='NOT_ESTIMABLE'


def test_age_uses_fixed_bins_and_protocol_never_inferred_from_era():
    baseline=pd.DataFrame({'demo__age_at_baseline':np.arange(20,80),'clinical_anchor_date':['2020-01-01']*60})
    admin,kinds,audit=administrative_frame(baseline,['demo__protocol','demo__age_at_baseline'],'clinical_anchor_date')
    assert audit.loc[audit.requested.eq('demo__protocol'),'reason'].iloc[0]=='SOURCE_UNAVAILABLE'
    assert administrative_strata(baseline.demo__age_at_baseline,'demo__age_at_baseline',10).nunique()<=6
    assert 'demo__protocol' not in admin
    baseline['ids__protocol']=['a','b']*30
    _,_,audit=administrative_frame(baseline,['demo__protocol'],'clinical_anchor_date')
    assert audit.loc[audit.requested.eq('demo__protocol'),'source'].iloc[0]=='ids__protocol'


def test_joint_mask_permutation_deterministic_and_small_n_not_fabricated():
    x=pd.DataFrame({'a':[1.]*10+[0.]*10,'b':[0.]*10+[1.]*10})
    y=pd.Series([0]*10+[1]*10)
    rows=mask_permutation_test(x,y,101,19)
    assert rows==mask_permutation_test(x,y,101,19)
    assert rows[0]['observed_statistic']==1
    assert rows[0]['p_exploratory']>=1/20
    assert len(rows)==19
    failed=mask_permutation_test(x.iloc[:3],y.iloc[:3],101,19)[0]
    assert failed['status']=='NOT_ESTIMABLE' and 'p_exploratory' not in failed


def test_cv_retrains_preprocessing_inside_each_fold_and_reports_support():
    x=pd.DataFrame({'load':np.r_[np.zeros(15),np.ones(15)]}); x.loc[3,'load']=np.nan
    rows=fit_adjusted_observation_model(x,[False]*15+[True]*15,321,repeats=2)
    assert len(rows)==10 and all(r['status']=='COMPLETE' for r in rows)
    assert all(r['n_train']+r['n_test']==30 for r in rows)
    assert rows[0]['auc_mean']>.9
    failed=fit_adjusted_observation_model(x,[True]*30,1)[0]
    assert failed['status']=='NOT_ESTIMABLE' and 'auc' not in failed

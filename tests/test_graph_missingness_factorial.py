"""Synthetic tests for the non-defining Graph02 factorial and its denominators."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.studies.graph._pipeline import load_pipeline
from src.studies.graph.stage02.s02_contracts import validate_factorial_inventory
from src.studies.graph.stage02.s02_factorial_imputation import (
    CompletionModel, artificial_mask, complete_panel, factorial_metadata, grouped_split, validation,
)
from src.studies.graph.stage02.s02_factorial_study import (
    Accumulator, Factorial, comparison_registry, historical_failure_audit, item, run_factorial,
)
from src.studies.graph.stage02.s02_missingness_registry import NotEstimable, classify_panel, panel_metadata
from src.studies.graph.stage02.s02_missingness_variants import classical_lens, factorial_mds, supported_mask
from src.studies.graph.stage02.s02_representations import build_space


@pytest.fixture
def cohort():
    cfg = load_pipeline()
    protocol = cfg['missingness_methods']['factorial_missingness_v3']
    protocol['resampling'].update(main_subsamples=2, main_bootstrap=2, parameter_subsamples=1)
    protocol['validation'].update(repeats=1, tuning_repeats=1, patient_ci_replicates=49)
    protocol['estimators'].update(candidate_rank=[1], chain_sweeps=2)
    protocol['uncertainty'].update(draws=2)
    cfg['experiments'].update(observation_permutations=3, minimum_stratum_patients=4)
    cfg['sensitivity'].update(analysis_mode='exploratory')
    cfg['sensitivity']['mapper'].update(n_cubes=3, min_samples=2, minimum_node_support=2)
    rng = np.random.default_rng(923)
    n = 159
    latent = rng.normal(size=(n, 2))
    raw = pd.DataFrame(latent@rng.normal(size=(2, 6))+.08*rng.normal(size=(n,6)), columns=list('abcdef'))
    raw.loc[0, 'a'] = 0.
    raw.loc[1, 'a'] = -1.
    affected = np.arange(2,n,11)
    raw.loc[affected, 'b'] = np.nan
    raw.loc[3::13, 'e'] = np.nan
    baseline = raw.assign(patient_id=[f'synthetic_{i}' for i in range(n)],
                          clinical_anchor_date=pd.Timestamp('2020-01-01'),
                          demo__protocol=np.where(np.arange(n)%2, 'synthetic_protocol_1','synthetic_protocol_2'))
    baseline.loc[affected, 'b__missingness_reason'] = 'clinical_not_requested'
    baseline.loc[3::13, 'e__observation_status'] = 'not_requested'
    registry = pd.DataFrame([
        dict(feature=c, source_column=c, semantic_feature_id=c, derivation_group=c, instrument_id='instrument_'+str(i//3),
             clinical_domain='D01' if i<3 else 'D02', subdomain='synthetic', family='PRO' if i<3 else 'CLINICAL',
             technical_family='PRO' if i<3 else 'CLINICAL', role='defining', temporality='baseline', temporal_status='baseline',
             temporal_validity='baseline', encoding='numeric', representation_type='numeric', exploratory_technical_eligible=True,
             mapping_eligibility='ELIGIBLE', unit='synthetic', included=True) for i,c in enumerate(raw)
    ])
    families = dict(zip(registry.feature,registry.family))
    return raw, baseline, registry, cfg, families


def make_study(cohort, reference='R0'):
    raw, baseline, registry, cfg, families = cohort
    config = cfg['sensitivity']
    raw = build_space(reference, raw, families, config, registry, registry, baseline, {}, {}).raw
    base = item(build_space(reference, raw, families, config, registry, registry, baseline, {}, {}), config, audit=True)
    protocol = cfg['missingness_methods']['factorial_missingness_v3']
    return Factorial(reference, base, registry, registry, baseline, families, config, cfg, {}, {}, protocol, defaultdict(list))


def test_provenance_keeps_unknown_structural_and_administrative_distinct(cohort):
    raw, baseline, registry, cfg, _ = cohort
    baseline.loc[3,'e__observation_status'] = 'not_scheduled_by_protocol'
    baseline.loc[16,'e__observation_status'] = 'not_applicable'
    baseline.loc[0,'a__missingness_reason'] = 'not_applicable'
    reasons = classify_panel(raw, registry, 'R0', baseline)
    cols = reasons.cells.set_index(['source_column','patient_key'])
    assert cols.loc[('a','synthetic_0'),'reason_class']=='contradiction'
    assert cols.loc[('b','synthetic_2'),'reason_class']=='clinical_documented'
    assert cols.loc[('e','synthetic_3'),'reason_class']=='administrative'
    assert cols.loc[('e','synthetic_16'),'reason_class']=='structural'
    assert cols.loc[('e','synthetic_29'),'reason_class']=='unknown'
    assert not reasons.eligible.loc[[3,16],'e'].any()
    assert reasons.eligible.loc[29,'e']
    e = reasons.observation.lineage.set_index('source_column').loc['e','observation_unit']
    assert np.isnan(reasons.masks['clinical_documented'].loc[29,e])
    assert reasons.masks['administrative_only'].loc[3,e]==1


@pytest.mark.parametrize('reference', ['R0','R2'])
def test_factorial_shares_completed_panel_and_mask_and_preserves_reference(cohort, reference):
    study = make_study(cohort, reference)
    study.method = 'mixed_regularized'
    full, lenses = study.full()
    assert set(full)=={'A','B','C','D'}
    assert full['A']['space'] is study.base['space']
    assert full['A']['mapped'] is study.base['mapped']
    assert full['B']['completed_hash']==full['D']['completed_hash']
    assert full['C']['mask_hash']==full['D']['mask_hash']
    assert full['D']['space'].weights.imputation.eq('mixed_regularized').all()
    assert lenses[reference+'-A-MDS-v3'] is not None
    assert factorial_mds(full['A']['space'], weight=0)[0] is full['A']['space']
    assert factorial_mds(full['B']['space'], weight=0)[0] is full['B']['space']
    study.resample(full)
    attempts = pd.DataFrame(study.tables['factorial_refit_attempts'])
    for _, group in attempts.groupby(['resampling','replicate_id']):
        assert group.split_hash.nunique()==group.mask_hash.nunique()==1
        assert group.loc[group.condition.isin(['B','D']), 'completed_hash'].nunique()==1
    assert (attempts.n_unique<attempts.n_draws).any()


def test_grouped_holdout_and_artificial_masks_do_not_change_observations(cohort):
    raw, baseline, registry, cfg, _ = cohort
    ids = baseline.patient_id.to_numpy()
    boot = np.repeat(ids, 2)
    train, held = grouped_split(boot, 119)
    assert not set(boot[train]) & set(boot[held])
    assert len(set(boot[held]))==len(held)
    meta = panel_metadata(raw,registry,'R0')
    policy = cfg['missingness_methods']['factorial_missingness_v3']['validation']
    for mechanism in policy['masking_mechanisms']:
        h1 = artificial_mask(raw,meta,.20,mechanism,8)
        assert np.array_equal(h1,artificial_mask(raw,meta,.20,mechanism,8))
        assert not (h1 & raw.isna().to_numpy()).any()
        assert h1.any()
    study = make_study(cohort)
    rows, support, folds, _ = validation(raw, meta, ids, study.impute_policy, 21, 'R0')
    folds = pd.DataFrame(folds)
    for _, group in folds.groupby(['repeat','mechanism','masking_fraction']):
        assert group.n_hidden.nunique()==group.split_sha256.nunique()==1
    assert {row['stage'] for row in rows}=={'inner_rank_tuning','outer_evaluation'}
    assert support


def test_mixed_draws_and_delta_preserve_legal_encodings_and_observed(cohort):
    raw, baseline, registry, cfg, _ = cohort
    study = make_study(cohort)
    raw = raw.copy()
    category = np.digitize(raw.a, [-1,1])
    for i in range(3):
        raw[f'cat{i}'] = (category==i).astype(float)
    raw['order__ordinal'] = category.astype(float)
    raw.loc[3::18,['cat0','cat1','cat2','order__ordinal']] = np.nan
    raw.loc[8,['cat0','cat1','cat2']] = [0., np.nan, np.nan]
    raw['constant'] = 1.
    source = registry.iloc[0].to_dict()
    registry = pd.concat([registry, pd.DataFrame([{**source, 'feature':c,
        'source_column':'category' if c.startswith('cat') else c,
        'semantic_feature_id':'category' if c.startswith('cat') else c,
        'derivation_group':'category' if c.startswith('cat') else c,
        'encoding':'one_hot' if c.startswith('cat') else 'ordinal' if c.endswith('__ordinal') else 'numeric'}
        for c in raw.columns[6:]])],ignore_index=True)
    reason = classify_panel(raw,registry,'R0',baseline)
    meta = panel_metadata(raw,registry,'R0')
    ids = baseline.patient_id.to_numpy()
    args = (raw, reason, meta, ids, 'mice_pmm_regularized', study.impute_policy, 794, 'R0')
    a, _ = complete_panel(*args, stochastic=True)
    b, _ = complete_panel(*args, stochastic=True)
    pd.testing.assert_frame_equal(a,b)
    observed = raw.notna().to_numpy()
    np.testing.assert_array_equal(a.to_numpy()[observed],raw.to_numpy()[observed])
    assert a.filter(like='cat').sum(axis=1).eq(1).all()
    assert a.order__ordinal.isin([0.,1.,2.]).all()
    assert a.constant.eq(1.).all()
    base, _ = complete_panel(*args)
    shifted, _ = complete_panel(*args, delta=.5, delta_reason='unknown')
    assert shifted.b.equals(base.b)
    assert not shifted.e.equals(base.e)
    np.testing.assert_array_equal(shifted.to_numpy()[observed],raw.to_numpy()[observed])


def test_structural_and_unvalidated_derivations_block_fixed_panel(cohort):
    study = make_study(cohort)
    raw, baseline, registry, *_ = cohort
    baseline.loc[3,'e__missingness_reason'] = 'not_applicable'
    reasons = classify_panel(raw,registry,'R0',baseline)
    with pytest.raises(NotEstimable,match='INELIGIBLE_MISSING'):
        complete_panel(raw, reasons, study.meta, baseline.patient_id.to_numpy(), 'mixed_regularized',
                       study.impute_policy, 10, 'R0')
    baseline.loc[3,'e__missingness_reason'] = None
    registry.loc[registry.feature.isin(['b','a']),'derivation_group'] = 'combined'
    reasons = classify_panel(raw,registry,'R0',baseline)
    with pytest.raises(NotEstimable,match='DERIVATION'):
        complete_panel(raw, reasons, study.meta, baseline.patient_id.to_numpy(), 'mixed_regularized',
                       study.impute_policy, 10, 'R0')


def test_missing_pairs_stay_undefined_and_lens_validates_geometry(cohort):
    study = make_study(cohort)
    states = study.reasons.masks['clinical_documented'].copy()
    states.iloc[0] = np.nan
    distance, weight, domains, audit = supported_mask(study.reasons,'R0','clinical_documented',
                                                     study.policy['mask'], states=states)
    assert audit['reason']=='INCOMPLETE_COMPARABLE_MASK_PAIRS'
    assert np.isfinite(distance[1:,1:]).all() and np.isnan(distance[0,1:]).all()
    assert not weight[0].any() and not domains[0].any()
    with pytest.raises(NotEstimable,match='DISSIMILARITY'):
        classical_lens(distance)
    # A valid, non-Euclidean cycle distance reports its negative eigenvalue.
    d = np.array([[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,1,0]],dtype=float)
    lens, audit = classical_lens(d)
    assert np.isfinite(lens).all() and audit['n_negative_eigenvalues']==1
    assert audit['positive_rank']==2
    with pytest.raises(NotEstimable,match='RANK'):
        classical_lens(np.abs(np.arange(4)[:,None]-np.arange(4)[None,:]))


def test_membership_and_matching_use_evaluable_supported_patients():
    membership = dict(hard=np.array([0.,1.,np.nan,1.]), covered=np.array([True,True,False,True]),
                      ties=np.zeros(4,bool), supported=[True,True],community_sets=[{0,1},{1,3}])
    accumulator = Accumulator(4)
    accumulator.collect(np.arange(4), membership, {0:1,1:0})
    rows, people = accumulator.rows('R0-I-v3','unit')
    pairs = {(r['patient_index_a'],r['patient_index_b']):r for r in rows}
    assert pairs[0,1]['n_same']==0 and pairs[0,1]['n_same_union']==1
    assert pairs[0,2]['n_evaluable']==0 and np.isnan(pairs[0,2]['coassignment'])
    assert people[2]['n_aligned_assignments']==0
    registry = comparison_registry()
    assert len(registry)==40 and len(set(registry))==40


def test_normal_orchestration_inventory_and_unsupported_candidate_are_explicit(cohort,tmp_path):
    raw, baseline, registry, cfg, families = cohort
    reduced = slice(0,36)
    raw, baseline = raw.iloc[reduced], baseline.iloc[reduced]
    config = cfg['sensitivity']
    base = {r: item(build_space(r,raw,families,config,registry,registry,baseline,{}, {}),config) for r in ('R0','R2')}
    prior = historical_failure_audit(tmp_path)
    assert next(r for r in prior if r['experiment_id']=='R0-M')['status']=='SOURCE_UNAVAILABLE'
    report = run_factorial(base,registry,registry,baseline,families,config,cfg,tmp_path,{}, {},prior,{}, (),tmp_path,Path('.'), {})
    assert len(report['statuses'])==6 and report['graph03_defining'] is False
    assert len(pd.read_csv(tmp_path/'02_graph_missingness_factorial_comparisons.csv'))==40
    assert any(f['status']=='NOT_ESTIMABLE' for f in report['figure_inventory'])
    files = {p.name:p for p in tmp_path.iterdir()}
    validate_factorial_inventory(report,['R0','R2'],files)
    candidate = tmp_path/'02_graph_missingness_candidate_selection.csv'
    candidate.write_text('altered\n')
    with pytest.raises(ValueError,match='SHA'):
        validate_factorial_inventory(report,['R0','R2'],files)


def test_rare_target_and_missing_target_are_audited_without_blocking_numeric_fit(cohort):
    raw, baseline, registry, _, _ = cohort
    study = make_study(cohort)
    rng = np.random.default_rng(87)
    train = raw.assign(rare=(np.arange(len(raw))==4).astype(float), absent=np.nan,
                       positive_control=rng.normal(size=len(raw)))
    row = registry.iloc[0].to_dict()
    for name, encoding in [('rare','binary'),('absent','numeric'),('positive_control','numeric')]:
        registry = pd.concat([registry, pd.DataFrame([{**row, 'feature':name, 'source_column':name,
                              'encoding':encoding, 'semantic_feature_id':name, 'derivation_group':name}])], ignore_index=True)
    meta = panel_metadata(train,registry,'R0')
    fit = train.iloc[:100]
    needed = set(train.columns)
    model = CompletionModel(fit,meta,'mice_pmm_regularized',study.impute_policy,
                            baseline.patient_id.iloc[:100].to_numpy(),needed,45,'R0')
    assert model.reasons['rare']=='INSUFFICIENT_CATEGORY_SUPPORT'
    assert model.reasons['absent']=='ALL_MISSING_TRAINING_TARGET'
    held = train.iloc[100:].copy()
    held['positive_control'] = 1e6
    held['rare'] = np.nan
    coefficients = [step[4].coef_.copy() for sweep in model.models for step in sweep]
    predicted, _ = model.predict(held,seed=42)
    assert predicted.rare.isna().all() and predicted.absent.isna().all()
    assert predicted.b.notna().all()
    assert predicted.positive_control.eq(1e6).all()
    for before, step in zip(coefficients, (step for sweep in model.models for step in sweep)):
        np.testing.assert_array_equal(before,step[4].coef_)
        if step[5] is not None:
            _, donors = step[5]
            assert (np.abs(donors) < 1e6).all()


def test_nominal_projection_uses_only_documented_omitted_levels(cohort):
    raw, baseline, registry, _, _ = cohort
    study = make_study(cohort)
    raw = raw.assign(category_low=(raw.a<0).astype(float))
    raw.loc[::17,'category_low'] = np.nan
    row = registry.iloc[0].to_dict()
    entry = {**row, 'source_column':'projected','encoding':'one_hot','semantic_feature_id':'projected',
             'derivation_group':'projected'}
    registry = pd.concat([registry,pd.DataFrame([{**entry,'feature':'category_low'},
                                                {**entry,'feature':'category_high'}])],ignore_index=True)
    ids = baseline.patient_id.to_numpy()
    with pytest.raises(NotEstimable,match='INVALID_NOMINAL_ENCODING'):
        CompletionModel(raw,panel_metadata(raw,registry,'R0'),'mixed_regularized',study.impute_policy,
                        ids,{'category_low'},32,'R0')
    model = CompletionModel(raw,factorial_metadata(raw,registry,'R0'),'mixed_regularized',study.impute_policy,
                            ids,{'category_low'},32,'R0')
    predicted, _ = model.predict(raw)
    assert predicted.category_low.isin([0.,1.]).all()
    observed = raw.category_low.notna()
    pd.testing.assert_series_equal(raw.loc[observed,'category_low'],predicted.loc[observed,'category_low'])

"""Boundary cases for the Stage05 estimands and descriptive tables."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.studies.pharma.stage05 import (
    absolute_risk, biomarker_correlations, clinical_profiles, contracts, eligibility, patient_trajectories,
    progression, quality_control, specimen_feasibility, transition_mechanisms, treatment_context,
)

PHARMA = Path(__file__).resolve().parents[1] / 'src' / 'studies' / 'pharma'


def settings():
    result, _ = contracts.load_settings(PHARMA)
    result.update(bootstrap_replicates=10)
    return result


def trajectories():
    patterns = {
        'a': ('Pop2', 'Pop2', 'Pop1'), 'b': ('Pop2', 'Pop2', 'Pop2'),
        'c': ('Pop2', 'Pop1', 'Pop2', 'Pop1'),
        'd': ('Pop2', 'Pop2', 'Pop3'), 'e': ('Pop2', 'Unclassifiable'),
        'f': ('Pop1', 'Pop2', 'Pop1'), 'g': ('Unclassifiable', 'Pop3'),
        'h': ('Pop2', 'Pop1', 'Unclassifiable', 'Pop1'),
        'i': ('Pop2', 'Pop1', 'Pop1'), 'j': ('Pop2',),
    }
    rows = []
    for p, pops in patterns.items():
        for j, pop in enumerate(pops):
            rows.append(dict(patient_id=p, clinical_episode_id=str(j), is_clinical_baseline=j == 0,
                             clinical_visit_number=j + 1,
                             clinical_anchor_date=pd.Timestamp('2010-01-01') + pd.Timedelta(j*400, unit='D'),
                             pop=pop, feature__essdai_total=6. if pop == 'Pop1' else
                             np.nan if pop == 'Unclassifiable' else 2.,
                             feature__esspri_total=np.nan if pop in ('Pop1', 'Unclassifiable') else
                             5. if pop == 'Pop2' else 3.))
    return pd.DataFrame(rows)


def test_a_documented_event_censoring_and_midpoint():
    master = trajectories()
    data = progression.event_history(master).set_index('patient_id')
    assert data.loc['a', 'event']
    assert np.isnan(data.loc['f', 'time']) and data.loc['f', 'ever']
    assert np.isnan(data.loc['e', 'time']) and not data.loc['e', 'has_followup']
    assert np.isnan(data.loc['j', 'time'])
    assert data.loc['b', 'time'] == 800/365.25
    mid = progression.event_history(master, midpoint=True).set_index('patient_id')
    assert mid.loc['a', 'time'] == pytest.approx(600/365.25)
    changed = master.copy()
    changed.loc[changed.patient_id.eq('a') & changed.clinical_visit_number.eq(3), 'feature__essdai_total'] = 4.
    assert not progression.event_history(changed).set_index('patient_id').loc['a', 'event']
    _, table, _ = progression.risk_tables(master, settings())
    assert not table.loc[table.years.eq(5), 'estimate'].notna().any()


def test_a_tied_events_and_censoring_and_degenerate_bounds():
    curves = progression.product_limit(np.array([1., 1., 2.]), np.array([1, 0, 0]))
    assert curves.iloc[0].n_at_risk == 3
    assert curves.iloc[0].estimate == pytest.approx(1/3)
    assert 0 < curves.iloc[0].ci95_low < 1/3 < curves.iloc[0].ci95_high < 1
    for events in [np.array([0, 0]), np.array([1, 1])]:
        boundary = progression.product_limit(np.ones(2), events).iloc[0]
        assert boundary.estimate == events[0]
        assert np.isnan(boundary.ci95_low)
    cif = progression.product_limit(np.array([1., 1., 2.]), np.array([1, 2, 1]), competing=True)
    assert cif.estimate.to_list() == pytest.approx([1/3, 2/3])


def test_g_trajectory_patterns_missing_and_recurrence():
    data = patient_trajectories.sequence_history(trajectories(), 2)
    base = data.loc[data.index_definition.eq('clinical_baseline')].set_index('patient_id')
    names = {p: patient_trajectories.classify(row, 2, 2) for p, row in base.iterrows()}
    assert names == {'a': 'other_conversion', 'b': 'observed_baseline_persistence', 'c': 'recurrent_conversion',
                     'd': 'other_changes_without_pop1', 'e': 'insufficient_followup', 'f': 'pop1_exit_and_return',
                     'g': 'insufficient_followup', 'h': 'other_conversion', 'i': 'sustained_conversion',
                     'j': 'insufficient_followup'}
    assert base.loc['i', 'confirmation_date'] > base.loc['i', 'sustained_entry_date']
    entry = data.loc[data.index_definition.eq('first_observed_pop2')].set_index('patient_id')
    assert entry.loc['f', 'history'] == 'prior_pop1'
    assert not entry.loc['f', 'first_conversion'] and entry.loc['f', 'recurrent_pop1']


def test_b_origin_only_and_nine_routes_with_unknown_binary():
    intervals = pd.DataFrame(dict(patient_id=['p', 'p', 'q'], from_pop=['Pop2']*3,
                                  to_pop=['Pop1']*3, from_clinical_anchor_date=pd.date_range('2010-01-01', periods=3),
                                  from_visit_number=[1, 2, 1], from_feature__x=[1., 0., np.nan],
                                  to_feature__x=[0., np.nan, 1.]))
    features = {'x': dict(group='clinical', status='available', type='binary', unit='',
                          direction='not_applicable', temporal_source='episode', reason='')}
    first = clinical_profiles.clinical_profiles(intervals, features)
    assert set(zip(first.from_pop, first.to_pop)) == set(contracts.ROUTES)
    assert first.loc[first.level.eq('unknown_or_missing'), 'n_level'].tolist() == [1, 1]
    assert first.loc[first.level.eq('negative'), 'n_level'].tolist() == [1, 0]
    copied = intervals.assign(to_feature__x=1000)
    pd.testing.assert_frame_equal(clinical_profiles.clinical_profiles(copied, features), first)


def test_c_composition_rejects_discordance_and_incomplete_pairs():
    frame = pd.DataFrame(dict(patient_id=['a', 'b', 'c'], from_pop=['Pop2']*3, to_pop=['Pop1']*3,
                              from_feature__essdai_total=[0., 1., 0.], to_feature__essdai_total=[5., 6., 5.]))
    markers = (*transition_mechanisms.DOMAINS, *transition_mechanisms.PRO)
    for name in markers:
        frame[f'from_feature__{name}'] = [0., 0., 0.]
        frame[f'to_feature__{name}'] = [0., 0., 0.]
    frame['to_feature__renal'] = [1., 1., np.nan]
    features = {name: dict(status='available', unit='points', reason='') for name in markers}
    domain, co, _, decomp, leave = transition_mechanisms.mechanisms(frame, features)
    route = lambda data: data.loc[data.from_pop.eq('Pop2') & data.to_pop.eq('Pop1')]
    total = route(decomp).query("marker == 'total'").set_index('scheme')
    assert total.n_evaluable.to_dict() == {'concordant_pair': 1, 'discordant_pair': 1, 'incomplete_pair': 1}
    assert route(domain).query("marker == 'renal'").n_new_active.item() == 2
    assert route(co).query("first_domain == 'renal' and second_domain == 'renal'").estimate.item() == 1.
    assert route(leave).n_lost_for_composition.to_list() == [2, 2]


def test_d_fixed_index_repeated_visits_and_fdr():
    config = settings()
    config.update(minimum_correlation_patients=4, correlation_markers=['x', 'y', 'z'], correlation_clinical_pairs={})
    rows = []
    for i in range(7):
        for visit in range(3):
            rows.append(dict(patient_id=str(i), pop='Pop2', clinical_anchor_date=pd.Timestamp('2010-01-01') +
                             pd.Timedelta(visit, unit='D'), feature__x=i + visit,
                             feature__y=i * 2 + visit, feature__z=np.nan if visit == 0 else i - visit))
    data = pd.DataFrame(rows)
    spec = lambda: dict(unit='units', direction='not_applicable', status='available')
    features = {a: spec() for a in ('x', 'y', 'z')}
    primary, repeated = biomarker_correlations.correlations(data, features, config)
    at_index = primary.query("pop == 'Pop2' and marker == 'x' and paired_marker == 'y'").iloc[0]
    assert at_index.n_patients_evaluable == 7 and at_index.estimate == pytest.approx(1.)
    assert at_index.p_value == at_index.q_value
    unavailable = primary.query("pop == 'Pop2' and paired_marker == 'z'")
    assert unavailable.n_patients_evaluable.eq(0).all()
    within = repeated.query("scheme == 'within_patient_centered' and pop == 'Pop2' and marker == 'x' and paired_marker == 'y'").iloc[0]
    assert within.n_paired_observations == 21 and within.estimate == pytest.approx(1.)
    assert np.isnan(within.p_value) and within.n_bootstrap_successful == 10


def test_e_generator_four_vs_six_routes_and_failed_bootstrap():
    method = absolute_risk
    params = pd.Series({**{f'intercept::{name}': -1. for name in method.ROUTES},
                        **{f'biomarker::{name}': .1 for name in method.ROUTES}, 'age': 0.})
    occupied = method.probabilities(method.generator(params, 0., 50., absorbing=False), [1, 2, 5], absorbing=False)
    reduced = params.drop(index=[col for col in params.index if 'Pop1→' in col])
    q = method.generator(reduced, 0, 50., absorbing=True)
    assert np.all(q[0] == 0) and np.allclose(q.sum(axis=1), 0.)
    first = method.probabilities(q, [1, 2, 5], absorbing=True)
    assert (np.diff(first[:, 1:, 0], axis=0) >= 0).all()
    assert (first[:, 1:, 0] >= occupied[:, 1:, 0]).all()
    with pytest.raises(KeyError):
        method.generator(reduced, 0., 50., absorbing=False)

    class Broken:
        def prepare(self, data, assay, minimum_events):
            assert list(data.patient_id) == [0, 0, 1, 1]
            return data, None, None, np.nan, np.nan, 'synthetic_failure'

    data = pd.DataFrame({'patient_id': ['a', 'a', 'b', 'b']})
    draws, failures = method.bootstrap(Broken(), data, 'lab__esr', list(method.ARRIVAL), {}, settings(),
                                       np.random.default_rng(3))
    assert not draws['first_arrival']
    assert failures['first_arrival']['synthetic_failure'] == 10
    assert not failures['occupancy']


def test_f_medication_documentation_and_eligibility_unknown_are_distinct():
    master = pd.DataFrame({
        'patient_id': ['p', 'p', 'q'], 'clinical_episode_id': ['a', 'b', 'a'],
        'is_clinical_baseline': [True, False, True], 'pop': ['Pop2', 'Pop1', 'Pop2'],
        'feature__ssa': [0., 1., np.nan], 'raw_status__ssa': ['known', 'known', 'indeterminate'],
        treatment_context.CTDB: [None, 'methotrexate', None],
        treatment_context.AFTER: ['methotrexate', None, None],
    })
    intervals = pd.DataFrame(dict(patient_id=['p'], from_clinical_episode_id=['a'], to_clinical_episode_id=['b'],
                                  from_pop=['Pop2'], to_pop=['Pop1']))
    medication = treatment_context.medication_summary(master, intervals)
    mtx = medication.query("from_pop == 'Pop2' and to_pop == 'Pop1' and medication == 'methotrexate'").iloc[0]
    assert mtx.n_later_order_documented == 1
    assert mtx.n_ctdb_reported == 0 and mtx.n_known == 0 and np.isnan(mtx.n_not_exposed)
    known = treatment_context.medication_summary(
        master, intervals.assign(_verified_exposure__methotrexate='not_exposed'))
    mtx_known = known.query("from_pop == 'Pop2' and to_pop == 'Pop1' and medication == 'methotrexate'").iloc[0]
    assert mtx_known.n_known == 1 and mtx_known.n_not_exposed == 1 and mtx_known.estimate == 0
    proxy = eligibility.eligibility(master, {'ssa': dict(status='available', reason='')})
    baseline = proxy.query("baseline_pop == 'Pop2' and not includes_potential_future").iloc[0]
    retrospective = proxy.query("baseline_pop == 'Pop2' and includes_potential_future").iloc[0]
    assert baseline.n_negative_known == 1 and baseline.n_indeterminate == 1
    assert retrospective.n_positive == 1 and retrospective.n_indeterminate == 1


def test_h_denominators_and_no_authorized_inventory():
    master = trajectories()
    intervals = pd.DataFrame({'patient_id': ['a', 'b']})
    reference = pd.DataFrame({'patient_id': ['a', 'outside'], 'clinical_episode_id': ['0', '0']})
    supported = quality_control.baseline_reconciliation(master, intervals, reference)
    assert supported.query("cohort == 'step11_shared_with_clinical_baseline'").n_patients.item() == 1
    assert supported.query("cohort == 'step11_only_absent_from_master'").n_patients.item() == 1
    specimen = specimen_feasibility.specimen_availability(None, master, pd.DataFrame(), settings())
    assert specimen.status.item() == 'not_available' and specimen.n_samples.isna().all()


def test_f_same_sample_model_with_independently_verified_exposures():
    rng = np.random.default_rng(87)
    n = 120
    intervals = pd.DataFrame({
        'patient_id': [f'p{i // 2}' for i in range(n)], 'from_pop': 'Pop2',
        'to_pop': np.where(rng.random(n) < .4, 'Pop1', 'Pop2'), 'interval_years': rng.uniform(.8, 1.8, n),
        'from_feature__mcv': rng.normal(90, 4, n), 'from_feature__age': rng.normal(53, 10, n),
        '_verified_exposure__methotrexate': np.where(rng.random(n) < .5, 'active', 'not_exposed'),
        '_verified_exposure__azathioprine': np.where(rng.random(n) < .4, 'active', 'past'),
    })
    intervals.loc[:3, '_verified_exposure__methotrexate'] = 'unknown'
    table = treatment_context.treatment_confounding(
        intervals, settings(), {'mcv': dict(unit='fL', status='available', reason='')})
    models = table.loc[table.from_pop.eq('Pop2') & table.scheme.str.endswith('same_sample')]
    assert models.n_evaluable.tolist() == [116, 116]
    assert models.status.eq('estimated').all()
    assert models.estimate.notna().all()


def test_h_specimen_linkage_counts_samples_and_aliquots_separately():
    master = trajectories()
    history = patient_trajectories.sequence_history(master, settings()['sustained_visits'])
    index = master.loc[master.patient_id.eq('f') & master.is_clinical_baseline].iloc[0]
    inventory = pd.DataFrame(dict(patient_id=['f', 'f'], clinical_episode_id=['0', '0'],
                                  specimen_id=['test-specimen', 'test-specimen'], aliquot_id=['a', 'b'],
                                  collection_date=index.clinical_anchor_date, inventory_date=index.clinical_anchor_date,
                                  available_at_inventory_date=True, specimen_type='serum', preservation='frozen'))
    summary = specimen_feasibility.specimen_availability(inventory, master, history, settings())
    observed = summary.query("cohort == 'baseline_pop1' and timing == 'available_at_index'").iloc[0]
    assert (observed.n_patients, observed.n_samples, observed.n_aliquots) == (1, 1, 2)
    assert not {'patient_id', 'specimen_id', 'aliquot_id'}.intersection(summary.columns)

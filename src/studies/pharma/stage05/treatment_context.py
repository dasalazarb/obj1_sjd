"""Separate episode documentation from verified exposure states."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .contracts import KEYS, route_groups, support_status

CTDB = 'medication__ctdb_reported_names'
BEFORE = 'medication__btris_order_before_episode_1y_names'
AFTER = 'medication__btris_order_after_episode_1y_names'
MEDICATIONS = {
    'hydroxychloroquine': ('hydroxychloroquine',),
    'methotrexate': ('methotrexate',),
    'azathioprine': ('azathioprine',),
    'mycophenolate': ('mycophenolate mofetil', 'mycophenolic acid sodium'),
    'leflunomide': ('leflunomide',),
    'corticosteroids': ('prednisone', 'prednisolone', 'methylprednisolone', 'dexamethasone'),
    'rituximab': ('rituximab',),
    'belimumab': ('belimumab',),
}
STATES = ('active', 'past', 'not_exposed', 'unknown')


def evidence_names(values: pd.Series) -> pd.Series:
    return values.fillna('').map(lambda value: frozenset(str(value).split('; ')) - {''})


def medication_summary(master: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fields = [column for column in (CTDB, BEFORE, AFTER) if column in master]
    from_source = master[KEYS + fields].rename(columns={'clinical_episode_id': 'from_clinical_episode_id'})
    to_source = master[KEYS + [c for c in [CTDB] if c in master]].rename(
        columns={'clinical_episode_id': 'to_clinical_episode_id', CTDB: 'destination_ctdb'})
    evidence = intervals.merge(from_source, on=['patient_id', 'from_clinical_episode_id'], validate='many_to_one')
    evidence = evidence.merge(to_source, on=['patient_id', 'to_clinical_episode_id'], validate='many_to_one')
    medicines = MEDICATIONS.copy()
    canonical_names = {name for aliases in medicines.values() for name in aliases}
    for field in fields:
        evidence[field] = evidence_names(evidence[field])
        for names in evidence[field]:
            for name in names - canonical_names:
                medicines.setdefault(name, (name,))
    if CTDB in fields:
        evidence['destination_ctdb'] = evidence_names(evidence['destination_ctdb'])
    for origin, target, part in route_groups(evidence):
        for drug, codes in sorted(medicines.items()):
            identified = {column: part[column].map(lambda names: not names.isdisjoint(codes)).astype(bool)
                          for column in fields}
            exposure_column = f'_verified_exposure__{drug}'
            states = (part[exposure_column].where(part[exposure_column].isin(STATES), 'unknown')
                      if exposure_column in part else pd.Series('unknown', index=part.index))
            n_known = int(states.ne('unknown').sum())
            n_active = int(states.eq('active').sum())
            start = identified.get(CTDB)
            if start is not None:
                end = part['destination_ctdb'].map(lambda names: not names.isdisjoint(codes)).astype(bool)
                from_list = part[CTDB].map(bool).astype(bool)
                to_list = part['destination_ctdb'].map(bool).astype(bool)
                comparable = from_list & to_list
            rows.append(dict(analysis_unit='interval', cohort='pharma01_intervals', index_definition='origin_episode',
                             from_pop=origin, to_pop=target, medication=drug, n_intervals=len(part),
                             n_patients=part.patient_id.nunique(), n_known=n_known, n_missing=len(part) - n_known,
                             n_active=n_active if n_known else np.nan,
                             n_past=int(states.eq('past').sum()) if n_known else np.nan,
                             n_not_exposed=int(states.eq('not_exposed').sum()) if n_known else np.nan,
                             estimate=n_active/n_known if n_known else np.nan,
                             n_ctdb_reported=int(start.sum()) if start is not None else np.nan,
                             n_patients_ctdb_reported=part.loc[start, 'patient_id'].nunique() if start is not None else np.nan,
                             n_with_ctdb_medication_names=int(from_list.sum()) if start is not None else np.nan,
                             n_prior_order_documented=int(identified[BEFORE].sum()) if BEFORE in fields else np.nan,
                             n_later_order_documented=int(identified[AFTER].sum()) if AFTER in fields else np.nan,
                             n_ctdb_lists_at_both_endpoints=int(comparable.sum()) if start is not None else np.nan,
                             n_documentation_appeared=int((comparable & ~start & end).sum()) if start is not None else np.nan,
                             n_documentation_disappeared=int((comparable & start & ~end).sum()) if start is not None else np.nan,
                             **support_status(n_known,
                                              reason='step08_episode_documentation_does_not_establish_anchor_exposure')))
    return pd.DataFrame(rows)


def treatment_confounding(intervals: pd.DataFrame, settings: dict, features: dict) -> pd.DataFrame:
    """Compare MCV on the same sample if independently verified MTX/AZA states exist.

    Step 08/10's public episode lists alone cannot populate these internal states.
    Events are modeled via interval Poisson offset and patient-clustered covariance.
    """
    rows = []
    for origin in ('Pop1', 'Pop2', 'Pop3'):
        cohort = intervals.loc[intervals.from_pop.eq(origin)].copy()
        meta = dict(analysis_unit='interval', cohort='pharma01_intervals', index_definition='origin_episode',
                    from_pop=origin, to_pop='Pop1', marker='mcv', unit=features['mcv']['unit'],
                    n_intervals=len(cohort), n_patients=cohort.patient_id.nunique())
        cohort['mcv'] = cohort['from_feature__mcv']
        cohort['age'] = cohort['from_feature__age']
        for name in ('methotrexate', 'azathioprine'):
            column = f'_verified_exposure__{name}'
            cohort[name] = cohort[column].where(cohort[column].isin(STATES), 'unknown') if column in cohort else 'unknown'
            for state in STATES:
                sample = cohort.loc[cohort[name].eq(state)]
                observed = sample.mcv.dropna()
                rows.append(dict(**meta, scheme='descriptive_distribution', medication=name, exposure=state,
                                 n_group=len(sample), n_evaluable=len(observed),
                                 n_patients_evaluable=sample.loc[sample.mcv.notna(), 'patient_id'].nunique(),
                                 n_missing=len(sample) - len(observed),
                                 estimate=observed.median() if len(observed) else np.nan,
                                 p25=observed.quantile(.25) if len(observed) else np.nan,
                                 p75=observed.quantile(.75) if len(observed) else np.nan,
                                 **support_status(len(observed), features['mcv']['status'] == 'available',
                                                  features['mcv']['reason'])))
        known = cohort.methotrexate.ne('unknown') & cohort.azathioprine.ne('unknown')
        sample = cohort.loc[known].dropna(subset=['mcv', 'age']).copy()
        sample['event'] = sample.to_pop.eq('Pop1').astype(int)
        sample['mtx'] = sample.methotrexate.eq('active').astype(int)
        sample['aza'] = sample.azathioprine.eq('active').astype(int)
        n_events = int(sample.event.sum())
        event_patients = sample.loc[sample.event.eq(1), 'patient_id'].nunique()
        n = len(sample)
        reason = 'stable_pop1_is_not_an_incident_transition' if origin == 'Pop1' else ''
        if not reason and not known.any():
            reason = 'no_verified_origin_methotrexate_azathioprine_exposure'
        if not reason and (n_events < settings['minimum_events'] or
                           n - n_events < settings['minimum_events'] or
                           event_patients < settings['minimum_event_patients'] or
                           sample.patient_id.nunique() < settings['minimum_model_patients'] or
                           any(sample[name].value_counts().reindex([0, 1], fill_value=0).min()
                               < settings['minimum_treatment_cell'] for name in ('mtx', 'aza')) or
                           sample.mcv.nunique() < 2):
            reason = 'insufficient_events_patients_or_exposure_variation'
        for version, cols in (('crude_same_sample', ['mcv']), ('age_and_treatment_adjusted_same_sample',
                                                             ['mcv', 'age', 'mtx', 'aza'])):
            status = reason
            estimate = low = high = p = np.nan
            if not reason:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', RuntimeWarning)
                        fit = sm.GLM(sample.event, sm.add_constant(sample[cols]),
                                     family=sm.families.Poisson(), offset=np.log(sample.interval_years)).fit(
                                         cov_type='cluster', cov_kwds={'groups': sample.patient_id})
                    if not fit.converged or not np.isfinite(fit.bse['mcv']):
                        raise ValueError('nonfinite_or_nonconvergent')
                    estimate = np.exp(fit.params['mcv'])
                    low, high = np.exp(fit.conf_int().loc['mcv'])
                    p = fit.pvalues['mcv']
                    if not np.isfinite([estimate, low, high, p]).all():
                        raise ValueError('nonfinite_estimate')
                except (ValueError, ArithmeticError, np.linalg.LinAlgError, PerfectSeparationError, RuntimeWarning):
                    estimate = low = high = p = np.nan
                    status = 'clustered_model_failed'
            rows.append(dict(**meta, scheme=version, medication='methotrexate_and_azathioprine', exposure='known',
                             n_group=int(known.sum()), n_evaluable=n,
                             n_patients_evaluable=sample.patient_id.nunique(), n_missing=len(cohort) - n,
                             n_events=n_events, n_event_patients=event_patients,
                             estimate=estimate, ci95_low=low, ci95_high=high, p_value=p,
                             **support_status(0 if status else n, reason=status)))
    return pd.DataFrame(rows)

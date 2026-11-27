"""Aggregate coverage, cohort reconciliation and complete Step 09 families."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.studies._shared import benjamini_hochberg

from .contracts import BASELINE_POPS, KEYS, check_keys, route_groups, support_status


def coverage(master: pd.DataFrame, intervals: pd.DataFrame, features: dict) -> pd.DataFrame:
    rows = []
    biomarkers = {marker: spec for marker, spec in features.items() if spec['group'] in ('laboratory', 'serology')}
    for origin, target, part in route_groups(intervals):
        for marker, spec in biomarkers.items():
            start = part[f'from_feature__{marker}'].notna()
            paired = start & part[f'to_feature__{marker}'].notna()
            n = int(start.sum())
            rows.append(dict(analysis_unit='interval', cohort='pharma01_intervals', index_definition='origin_episode',
                             scheme='route', from_pop=origin, to_pop=target, marker=marker, unit=spec['unit'],
                             n_patients=part.patient_id.nunique(), n_intervals=len(part),
                             n_evaluable=n, n_paired=int(paired.sum()),
                             n_patients_measured=part.loc[start, 'patient_id'].nunique(),
                             n_patients_paired=part.loc[paired, 'patient_id'].nunique(),
                             n_missing=len(part) - n, estimate=n/len(part) if len(part) else np.nan,
                             **support_status(n, spec['status'] == 'available', spec['reason'])))
    episodes = master.assign(visit=master.clinical_visit_number.astype(int))
    strata = [('pop', ['pop']), ('pop_and_visit', ['pop', 'visit'])]
    if features['protocol']['status'] == 'available':
        episodes = episodes.assign(protocol=episodes.feature__protocol.fillna('Unknown'))
        strata.append(('pop_and_protocol', ['pop', 'protocol']))
    for scheme, grouping in strata:
        group_key = grouping[0] if len(grouping) == 1 else grouping
        for key, part in episodes.groupby(group_key, sort=True):
            key_values = dict(zip(grouping, key if isinstance(key, tuple) else [key]))
            for marker, spec in biomarkers.items():
                valid = part[f'feature__{marker}'].notna()
                n = int(valid.sum())
                rows.append(dict(analysis_unit='episode', cohort='effective_master', index_definition='observed_episode',
                                 scheme=scheme, marker=marker, unit=spec['unit'], **key_values,
                                 n_patients=part.patient_id.nunique(), n_episodes=len(part),
                                 n_patients_measured=part.loc[valid, 'patient_id'].nunique(), n_evaluable=n,
                                 n_missing=len(part) - n, estimate=n/len(part),
                                 **support_status(n, spec['status'] == 'available', spec['reason'])))
    fields = ['analysis_unit', 'cohort', 'index_definition', 'scheme', 'from_pop', 'to_pop', 'pop', 'visit',
              'protocol', 'marker', 'unit', 'n_patients', 'n_intervals', 'n_episodes', 'n_evaluable',
              'n_paired', 'n_patients_measured', 'n_patients_paired', 'n_missing', 'estimate', 'status', 'reason']
    return pd.DataFrame(rows, columns=fields)


def baseline_reconciliation(master: pd.DataFrame, intervals: pd.DataFrame,
                            reference: pd.DataFrame | None) -> pd.DataFrame:
    baseline = master.loc[master.is_clinical_baseline]
    master_ids = set(master.patient_id)
    baseline_ids = set(baseline.patient_id)
    interval_ids = set(intervals.patient_id)
    observed_ids = set(master.loc[master.feature__essdai_total.notna(), 'patient_id'])
    first_classifiable = master.loc[master['pop'].isin(BASELINE_POPS[:3])].drop_duplicates('patient_id')
    distinct_index = first_classifiable.merge(baseline[KEYS], on='patient_id', suffixes=('', '_baseline'))
    rows = []
    for cohort, count in (
        ('effective_master', len(master_ids)), ('clinical_baseline', len(baseline_ids)),
        ('no_clinical_baseline', len(master_ids - baseline_ids)),
        ('essdai_ever_evaluable', len(observed_ids)),
        ('no_observed_essdai', len(master_ids - observed_ids)),
        ('pop_longitudinal_classifiable', len(first_classifiable)), ('pharma01_interval_patients', len(interval_ids)),
        ('clinical_baseline_without_interval', len(baseline_ids - interval_ids)),
        ('first_classifiable_differs_from_clinical_baseline',
         int(distinct_index.clinical_episode_id.ne(distinct_index.clinical_episode_id_baseline).sum())),
    ):
        rows.append(dict(cohort=cohort, n_patients=count, status='estimated', reason=''))
    for pop in BASELINE_POPS[:-1]:
        rows.append(dict(cohort='clinical_baseline', baseline_pop=pop,
                         n_patients=int(baseline['pop'].eq(pop).sum()), status='estimated', reason=''))
    if reference is not None:
        check_keys(reference, ['patient_id'], 'Step 11 baseline reference')
        reference_ids = set(reference.patient_id)
        for cohort, ids in [('step11_baseline_reference', reference_ids),
                            ('step11_shared_with_clinical_baseline', reference_ids & baseline_ids),
                            ('step11_only_absent_from_master', reference_ids - master_ids),
                            ('step11_in_master_without_baseline', (reference_ids & master_ids) - baseline_ids),
                            ('clinical_baseline_only_not_in_step11', baseline_ids - reference_ids)]:
            rows.append(dict(cohort=cohort, n_patients=len(ids), status='estimated', reason=''))
        if 'clinical_episode_id' in reference:
            common = baseline[KEYS].merge(reference[KEYS], on='patient_id', suffixes=('', '_reference'))
            rows.append(dict(cohort='shared_patient_different_baseline_episode',
                             n_patients=int(common.clinical_episode_id.ne(common.clinical_episode_id_reference).sum()),
                             status='estimated', reason=''))
    else:
        rows.append(dict(cohort='step11_baseline_reference', n_patients=np.nan, status='unavailable',
                         reason='historical_148_and_159_counts_cannot_be_linked_without_canonical_reference'))
    return pd.DataFrame(rows, columns=['cohort', 'baseline_pop', 'n_patients', 'status', 'reason'])


def pro_slopes_fdr(source: pd.DataFrame | None) -> pd.DataFrame:
    fields = ['instrument', 'measure', 'model', 'effect', 'model_status', 'n_patients',
              'n_observations', 'p_value', 'q_value_original', 'q_value', 'family',
              'n_family_tests', 'n_family_estimable', 'status', 'reason']
    required = set(fields[:8]) | {'q_value'}
    if source is None or not required.issubset(source) or source.empty:
        return pd.DataFrame([dict(status='unavailable', reason='complete_step09_model_family_unavailable')], columns=fields)
    check_keys(source, ['instrument', 'measure', 'model', 'effect'], 'Step 09 slope family')
    result = source.rename(columns={'q_value': 'q_value_original'}).copy()
    result['family'] = result.model + '_all_step09_time_coefficients'
    groups = result.groupby('family').p_value
    result['q_value'] = groups.transform(benjamini_hochberg)
    result['n_family_tests'] = groups.transform('size')
    result['n_family_estimable'] = groups.transform('count')
    if not np.allclose(result.q_value, result.q_value_original, atol=1e-8, rtol=0, equal_nan=True):
        return pd.DataFrame([dict(status='not_estimable', reason='step09_family_does_not_reproduce_published_fdr')], columns=fields)
    result['status'] = np.where(result.p_value.notna(), 'estimated', 'not_estimable')
    result['reason'] = np.where(result.p_value.notna(), '', result.model_status)
    return result[fields]


def evidence_by_transition(intervals, clinical, lab, coverage_table, mechanism, support, models):
    rows = []
    for origin, target, part in route_groups(intervals):
        selected = lambda table: table.from_pop.eq(origin) & table.to_pop.eq(target)
        local_profiles = pd.concat([clinical, lab], ignore_index=True)
        profiles = local_profiles.loc[selected(local_profiles) & local_profiles.scheme.eq('all_intervals')]
        cov = coverage_table.loc[selected(coverage_table) & coverage_table.scheme.eq('route')]
        domains = mechanism.loc[selected(mechanism)]
        fit = models.loc[models.transition_id.eq(f'{origin}→{target}') &
                         models.model_version.eq('age_adjusted_clustered')]
        risk = support.loc[support.route.eq(f'{origin}→{target}')]
        rows.append(dict(analysis_unit='interval', cohort='pharma01_intervals', from_pop=origin, to_pop=target,
                         priority='primary_contrast' if (origin, target) == ('Pop2', 'Pop1') else 'context',
                         n_intervals=len(part), n_patients=part.patient_id.nunique(),
                         n_profile_markers_evaluable=profiles.loc[profiles.status.eq('estimated'), 'marker'].nunique(),
                         n_biomarkers_measured=int(cov.n_evaluable.gt(0).sum()),
                         n_domains_with_pairs=int(domains.n_evaluable.gt(0).sum()),
                         n_interpretable_step03_biomarkers=int(fit.interpretability_status.eq('interpretable').sum()),
                         n_supported_stage05_risk_markers=int(risk.status.eq('estimated').sum()),
                         **support_status(len(part))))
    return pd.DataFrame(rows)

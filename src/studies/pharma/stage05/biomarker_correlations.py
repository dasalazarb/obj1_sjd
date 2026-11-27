"""Index and longitudinal Spearman associations with patient-level resampling."""
from __future__ import annotations

from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.studies._shared import benjamini_hochberg

from .contracts import POPS, support_status


def rho(values: np.ndarray) -> float:
    a, b = rankdata(values, axis=0).T
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _bootstrap(clusters: list[np.ndarray], settings: dict, rng) -> tuple[float, float, int]:
    if len(clusters) < settings['minimum_correlation_patients']:
        return np.nan, np.nan, 0
    values = []
    for _ in range(settings['bootstrap_replicates']):
        indices = rng.integers(len(clusters), size=len(clusters))
        sampled = np.concatenate([clusters[i] for i in indices])
        estimate = rho(sampled)
        if np.isfinite(estimate):
            values.append(estimate)
    if len(values) < settings['bootstrap_replicates'] * settings['minimum_bootstrap_success']:
        return np.nan, np.nan, len(values)
    low, high = np.quantile(values, [.025, .975])
    return low, high, len(values)


def _row(sample: pd.DataFrame, columns: list[str], features: dict, meta: dict, settings: dict, rng) -> dict:
    x, y = (name.removeprefix('feature__') for name in columns)
    sample = sample.dropna(subset=columns + ['patient_id'])
    clusters = []
    for _, part in sample.groupby('patient_id', sort=False):
        values = part[columns].to_numpy(dtype=float)
        if meta['scheme'] == 'patient_means':
            clusters.append(values.mean(axis=0, keepdims=True))
        elif meta['scheme'] == 'within_patient_centered':
            if len(part) >= 2 and part.clinical_anchor_date.nunique() >= 2:
                clusters.append(values - values.mean(axis=0))
        else:
            clusters.append(values)
    n = len(clusters)
    observations = sum(map(len, clusters))
    available = all(features[marker]['status'] == 'available' for marker in (x, y))
    reason = 'mapped_marker_unavailable' if not available else (
        'insufficient_paired_patients' if n < settings['minimum_correlation_patients'] else 'constant_or_invalid_pairs')
    estimate, p, low, high, successful = np.nan, np.nan, np.nan, np.nan, 0
    if available and n >= settings['minimum_correlation_patients']:
        values = np.concatenate(clusters)
        estimate = rho(values)
        if np.isfinite(estimate):
            if meta['scheme'] != 'within_patient_centered':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    p = float(spearmanr(values[:, 0], values[:, 1]).pvalue)
            low, high, successful = _bootstrap(clusters, settings, rng)
    return dict(**meta, marker=x, paired_marker=y, unit=features[x]['unit'], paired_unit=features[y]['unit'],
                direction=features[x]['direction'], paired_direction=features[y]['direction'],
                n_patients_evaluable=n, n_paired_observations=observations,
                n_patients_missing=meta['n_patients']-n,
                n_bootstrap_successful=successful, estimate=estimate, ci95_low=low, ci95_high=high, p_value=p,
                **support_status(int(np.isfinite(estimate)), available, reason))


def correlations(master: pd.DataFrame, features: dict, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(settings['seed'])
    markers = list(dict.fromkeys(settings['correlation_markers'] + [
        marker for marker, spec in features.items() if spec.get('reviewed_candidate')]))
    pairs = [(x, y, 'biomarker') for x, y in combinations(markers, 2)]
    pairs.extend((x, y, 'clinical_cross_domain') for x, targets in settings['correlation_clinical_pairs'].items()
                 for y in targets)
    primary, longitudinal = [], []
    for pop in POPS:
        visits = master.loc[master['pop'].eq(pop)]
        index = visits.drop_duplicates('patient_id')
        for x, y, family in pairs:
            columns = [f'feature__{x}', f'feature__{y}']
            for scheme in ('first_episode_in_pop', 'patient_means', 'within_patient_centered'):
                rows = primary if scheme == 'first_episode_in_pop' else longitudinal
                sample = index if scheme == 'first_episode_in_pop' else visits
                test_family = family + ('_pop2' if pop == 'Pop2' else '_other_pops')
                meta = dict(analysis_unit='patient' if scheme != 'within_patient_centered' else 'paired_visit',
                            cohort='effective_master', index_definition='first_episode_in_pop'
                            if scheme == 'first_episode_in_pop' else 'all_episodes_in_pop',
                            pop=pop, scheme=scheme, family=test_family,
                            n_patients=visits.patient_id.nunique(), n_visits=len(sample))
                rows.append(_row(sample, columns, features, meta, settings, rng))
    output = []
    for rows in (primary, longitudinal):
        frame = pd.DataFrame(rows)
        frame['q_value'] = frame.groupby(['scheme', 'family']).p_value.transform(benjamini_hochberg)
        output.append(frame)
    return tuple(output)

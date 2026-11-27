"""Descriptive origin profiles in all nine canonical transition routes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import route_groups, support_status

PROFILE_FIELDS = ['marker', 'unit', 'type', 'direction', 'temporal_source', 'n_patients',
                  'n_evaluable', 'n_patients_measured', 'n_missing', 'level', 'n_level', 'estimate',
                  'median', 'p25', 'p75', 'mean', 'sd', 'status', 'reason']


def describe(part: pd.DataFrame, column: str, marker: str, spec: dict) -> list[dict]:
    values = part[column]
    measured = values.dropna()
    n = len(measured)
    available = spec['status'] == 'available'
    result = dict(marker=marker, unit=spec['unit'], type=spec['type'], direction=spec['direction'],
                  temporal_source=spec['temporal_source'], n_patients=part.patient_id.nunique(),
                  n_evaluable=n, n_patients_measured=part.loc[values.notna(), 'patient_id'].nunique(),
                  n_missing=len(part) - n,
                  **support_status(n, available, spec['reason'] if not available else ''))
    if spec['type'] in ('continuous', 'ordinal'):
        return [dict(**result, median=measured.median() if n else np.nan,
                     p25=measured.quantile(.25) if n else np.nan, p75=measured.quantile(.75) if n else np.nan,
                     mean=measured.mean() if n else np.nan, sd=measured.std() if n > 1 else np.nan)]
    levels = [('negative', int(measured.eq(0).sum())), ('positive', int(measured.eq(1).sum()))] if (
        spec['type'] == 'binary'
    ) else list(measured.value_counts().sort_index().items())
    if not available or not len(part):
        return [result]
    rows = [{**result, 'level': level, 'n_level': count, 'estimate': count/n if n else np.nan}
            for level, count in levels]
    rows.append({**result, 'level': 'unknown_or_missing', 'n_level': len(part) - n,
                 'estimate': (len(part) - n)/len(part)})
    return rows


def interval_profiles(intervals: pd.DataFrame, features: dict, groups: set[str]) -> pd.DataFrame:
    rows = []
    for origin, target, part in route_groups(intervals):
        ordered = part.sort_values(['from_clinical_anchor_date', 'from_visit_number'])
        for scheme, sample in [('all_intervals', ordered), ('first_interval_per_patient_route',
                                                          ordered.drop_duplicates('patient_id'))]:
            meta = dict(analysis_unit='interval', cohort='pharma01_intervals', index_definition='origin_episode',
                        scheme=scheme, from_pop=origin, to_pop=target, n_intervals=len(sample))
            for marker, spec in features.items():
                if spec['group'] in groups:
                    rows.extend({**meta, **row} for row in describe(sample, f'from_feature__{marker}', marker, spec))
    return pd.DataFrame(rows, columns=['analysis_unit', 'cohort', 'index_definition', 'scheme',
                                      'from_pop', 'to_pop', 'n_intervals', *PROFILE_FIELDS])


def clinical_profiles(intervals: pd.DataFrame, features: dict) -> pd.DataFrame:
    return interval_profiles(intervals, features, {'clinical', 'domain'})

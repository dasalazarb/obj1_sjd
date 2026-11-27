"""Paired ordinal changes and arithmetic ESSDAI decomposition."""
from __future__ import annotations

from importlib import import_module
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from .contracts import THRESHOLD, route_groups, support_status

DOMAINS = ('constitutional', 'lymphadenopathy', 'glandular', 'articular', 'cutaneous',
           'pulmonary', 'renal', 'muscular', 'pns', 'cns', 'hematologic', 'biological')
PRO = ('esspri_total', 'dryness', 'pain', 'fatigue')


def composition_contract() -> tuple[dict, float]:
    """Reuse Step 14's checked weight aliases and score tolerance."""
    module = import_module('src.block_A.14_risk_factors_progression')
    return module.canonical_domain_weights(), module.COMPOSITION_TOLERANCE


def _pairs(part: pd.DataFrame, marker: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    start, end = part[f'from_feature__{marker}'], part[f'to_feature__{marker}']
    valid = start.notna() & end.notna()
    return start.where(valid), end.where(valid), valid


def _median(values: pd.Series) -> float:
    return values.dropna().median() if values.notna().any() else np.nan


def _pair_support(part: pd.DataFrame, valid: pd.Series) -> dict:
    n = int(valid.sum())
    return dict(n_evaluable=n, n_patients_evaluable=part.loc[valid, 'patient_id'].nunique(),
                n_missing=len(part) - n)


def mechanisms(intervals: pd.DataFrame, features: dict) -> tuple[pd.DataFrame, ...]:
    weights, tolerance = composition_contract()
    drivers, cooccurrence, pro, decomposition, leave_out = [], [], [], [], []
    if set(weights) != set(DOMAINS):
        raise ValueError('Stage05 domain names differ from canonical ESSDAI weights')
    for origin, target, part in route_groups(intervals):
        meta = dict(analysis_unit='interval', cohort='pharma01_intervals', index_definition='origin_episode',
                    from_pop=origin, to_pop=target, n_intervals=len(part), n_patients=part.patient_id.nunique())
        scores, changes = {}, {}
        for marker in (*DOMAINS, *PRO):
            start, end, valid = _pairs(part, marker)
            delta = end - start
            changes[marker] = delta
            support = _pair_support(part, valid)
            n = support['n_evaluable']
            available = features[marker]['status'] == 'available'
            row = dict(**meta, **support, marker=marker, unit=features[marker]['unit'],
                       n_worsened=int(delta.gt(0).sum()), n_improved=int(delta.lt(0).sum()),
                       n_no_change=int(delta.eq(0).sum()), estimate=_median(delta),
                       **support_status(n, available, features[marker]['reason']))
            if marker in DOMAINS:
                scores[marker] = start, end
                row.update(n_new_active=int((start.eq(0) & end.gt(0)).sum()),
                           n_resolved=int((start.gt(0) & end.eq(0)).sum()),
                           n_persistently_active=int((start.gt(0) & end.gt(0)).sum()),
                           n_persistently_inactive=int((start.eq(0) & end.eq(0)).sum()),
                           proportion_active_from=start.gt(0).sum()/n if n else np.nan,
                           proportion_active_to=end.gt(0).sum()/n if n else np.nan)
                for change in range(-3, 4):
                    row[f'n_delta_{change}'] = int(delta.eq(change).sum())
                drivers.append(row)
            else:
                pro.append(row)
        components = pd.DataFrame({marker: changes[marker] for marker in PRO[1:]})
        complete = components.notna().all(axis=1)
        magnitudes = components.abs()
        largest = magnitudes.max(axis=1)
        winners = magnitudes.eq(largest, axis=0).sum(axis=1)
        details = _pair_support(part, complete)
        for marker in (*PRO[1:], 'tie', 'no_change'):
            if marker == 'tie':
                best = winners.gt(1) & largest.gt(0)
            elif marker == 'no_change':
                best = largest.eq(0)
            else:
                best = winners.eq(1) & magnitudes[marker].eq(largest) & largest.gt(0)
            n = details['n_evaluable']
            n_best = int((best & complete).sum())
            pro.append(dict(**meta, **details, marker=marker, unit='proportion',
                            scheme='dominant_absolute_component_change', n_dominant=n_best,
                            estimate=n_best/n if n else np.nan, **support_status(n)))
        for first, second in combinations_with_replacement(DOMAINS, 2):
            a, b = changes[first], changes[second]
            valid = a.notna() & b.notna()
            pair = _pair_support(part, valid)
            n = pair['n_evaluable']
            concurrent = int((a.gt(0) & b.gt(0)).sum())
            cooccurrence.append(dict(**meta, **pair, first_domain=first, second_domain=second,
                                     n_both_worsened=concurrent, estimate=concurrent/n if n else np.nan,
                                     **support_status(n)))

        total_start, total_end, totals_valid = _pairs(part, 'essdai_total')
        reconstructed = [pd.DataFrame({domain: scores[domain][side] * weights[domain]
                                       for domain in DOMAINS}).sum(axis=1, min_count=len(DOMAINS))
                         for side in (0, 1)]
        observed_delta = total_end - total_start
        reconstructed_delta = reconstructed[1] - reconstructed[0]
        complete = totals_valid & reconstructed[0].notna() & reconstructed[1].notna()
        concordant = complete & np.isclose(total_start, reconstructed[0], rtol=0, atol=tolerance) & np.isclose(
            total_end, reconstructed[1], rtol=0, atol=tolerance)
        reason = 'requires_complete_and_endpoint_concordant_composition'
        for scheme, valid in (('concordant_pair', concordant), ('discordant_pair', complete & ~concordant),
                              ('incomplete_pair', ~complete)):
            details = _pair_support(part, valid)
            decomposition.append(dict(**meta, **details, scheme=scheme, marker='total',
                                      unit='ESSDAI points',
                                      observed_delta_median=_median(observed_delta.where(valid)),
                                      reconstructed_delta_median=_median(reconstructed_delta.where(valid)),
                                      residual_median=_median((observed_delta - reconstructed_delta).where(valid)),
                                      **support_status(details['n_evaluable'])))
        for domain in DOMAINS:
            weighted = changes[domain].where(concordant) * weights[domain]
            decomposition.append(dict(**meta, **_pair_support(part, concordant), scheme='concordant_pair',
                                      marker=domain, unit='ESSDAI points', weight=weights[domain],
                                      estimate=_median(weighted), **support_status(int(concordant.sum()), reason=reason)))
        for domain in ('biological', 'hematologic'):
            start, end = [(total - weights[domain] * scores[domain][side]).where(concordant)
                          for side, total in enumerate((total_start, total_end))]
            n = int(concordant.sum())
            leave_out.append(dict(**meta, **_pair_support(part, concordant),
                                  marker=domain, unit='ESSDAI points', threshold=THRESHOLD,
                                  n_total_score_pairs=int(totals_valid.sum()),
                                  n_lost_for_composition=int(totals_valid.sum()) - n,
                                  n_crossing_ge5_full_same_sample=int((concordant & total_start.lt(THRESHOLD) &
                                                                       total_end.ge(THRESHOLD)).sum()),
                                  n_crossing_ge5_excluding_domain=int((start.lt(THRESHOLD) & end.ge(THRESHOLD)).sum()),
                                  full_delta_median_same_sample=_median(observed_delta.where(concordant)),
                                  estimate=_median(end - start), **support_status(n, reason=reason)))
    return tuple(pd.DataFrame(rows) for rows in (drivers, cooccurrence, pro, decomposition, leave_out))

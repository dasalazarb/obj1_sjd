"""Reproducible Gaussian coefficient sup-t inference and explicit BH families."""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from src.block_A.secondary_trajectory_analysis import ContractError


def simultaneous_critical(gradients, covariance, simulations, seed):
    gradients = np.asarray(gradients)
    se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', gradients, covariance, gradients), 0))
    valid = se > 1e-12
    if not valid.any():
        return 1.959963984540054
    eig, vectors = np.linalg.eigh(covariance)
    if not np.isfinite(eig).all() or eig.min() <= 0:
        raise ValueError('invalid coefficient covariance for bands')
    rng = np.random.default_rng(seed)
    root = vectors*np.sqrt(eig)
    maximum = []
    # Bound peak memory for long full-horizon grids.
    for start in range(0, simulations, 250):
        error = rng.normal(size=(min(250, simulations-start), len(eig)))@root.T
        maximum.extend(np.max(np.abs(error@gradients[valid].T/se[valid]), axis=1))
    return max(1.959963984540054, float(np.quantile(maximum, .95)))


def adjust_fdr(table):
    d = table.copy().reset_index(drop=True)
    for col in ('q_value', 'q_domain'):
        d[col] = np.nan
    d['family_n_tests'], d['domain_family_n_tests'] = 0, 0
    d['multiplicity_method'] = 'BH-FDR_across_independent_variables_per_test_scenario_axis_threshold'
    if d.empty or 'p' not in d:
        return d
    valid = d.status.eq('ESTIMATED') & d.p.between(0, 1) & ~d.is_essdai_derived.astype(bool)
    keys = ['threshold_abs', 'time_axis', 'scenario', 'test']
    if 'outcome_id' in d and d.duplicated(keys+['outcome_id']).any():
        raise ContractError('duplicate canonical outcome in FDR family')
    for extra, q, n in [([], 'q_value', 'family_n_tests'), (['domain'], 'q_domain', 'domain_family_n_tests')]:
        for _, indices in d.loc[valid].groupby(keys+extra, dropna=False).groups.items():
            d.loc[indices, q] = multipletests(d.loc[indices, 'p'], method='fdr_bh')[1]
            d.loc[indices, n] = len(indices)
    return d


def separation_intervals(bands, step):
    rows = []
    if bands.empty:
        return pd.DataFrame()
    for (pair, kind), part in bands.groupby(['contrast', 'contrast_type'], sort=False):
        part = part.sort_values('time')
        significant = part.sim95_low.gt(0) | part.sim95_high.lt(0)
        segment, previous, sign = [], None, None
        def save():
            if not segment:
                return
            a = pd.DataFrame(segment)
            rows.append(dict(contrast=pair, contrast_type=kind, start=float(a.time.min()), end=float(a.time.max()),
                n_supported_windows=len(a), estimate_min=a.estimate.min(), estimate_max=a.estimate.max(),
                band_low_min=a.sim95_low.min(), band_high_max=a.sim95_high.max(),
                status='DETECTED_TEMPORAL_SEPARATION_WITHIN_OBSERVED_SUPPORT',
                method=a.method.iloc[0], timing_uncertainty='grid/model/domain dependent; no onset inference'))
        for row, ok in zip(part.to_dict('records'), significant):
            s = np.sign(row['estimate'])
            if not ok or (previous is not None and (row['time']-previous > step*1.01 or s != sign)):
                save()
                segment = []
            if ok:
                segment.append(row)
            previous, sign = row['time'], s
        save()
    return pd.DataFrame(rows)

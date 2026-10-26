"""Frozen cohort/time sensitivities and separate, valid-only multiplicity families."""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def scenarios(frame, axis, cfg):
    if frame.empty:
        return
    yield 'official_variable_baseline', frame.loc[frame.reference_source.eq('official_clinical_baseline')].copy(), axis
    from src.block_A.secondary_comparison_contract import inference_frame
    observed = inference_frame(frame, axis)
    yield 'variable_ge3_dates', observed.loc[observed.n_analysis_dates.ge(3)].copy(), axis
    yield 'essdai_ge3_dates', frame.loc[frame.n_distinct_essdai_dates.ge(3)].copy(), axis
    yield 'essdai_official_index', frame.loc[frame.essdai_index_source.eq('official_baseline')].copy(), axis
    yield 'essdai_delayed_index', frame.loc[frame.essdai_index_source.ne('official_baseline')].copy(), axis
    other_axis = 'essdai_index' if axis == 'clinical_baseline' else 'clinical_baseline'
    other = frame.copy()
    other['time'] = other['time_from_'+other_axis+'_years']
    other['window'] = np.floor(other.time/cfg.window_years).astype(int)
    yield 'alternate_time_axis', other, other_axis
    yield 'common_horizon_truncation', frame.loc[frame.time.between(*cfg.auc_horizon)].copy(), axis


def adjust_tests(table, kind='global'):
    if table.empty:
        return table.assign(q_value=pd.Series(dtype=float), family_n_tests=pd.Series(dtype=int))
    data = table.copy().reset_index(drop=True)
    data['q_value'], data['family_n_tests'] = np.nan, 0
    data['multiplicity_method'] = 'BH-FDR' if kind == 'global' else 'Holm'
    keys = [k for k in ('threshold_abs', 'time_axis', 'scenario', 'family', 'profile') if k in data]
    if kind == 'pairwise':
        keys += [k for k in ('variable', 'contrast_type', 'multiplicity_role', 'time') if k in data]
    valid = data.status.eq('ESTIMATED') & pd.to_numeric(data.p, errors='coerce').between(0, 1)
    if 'is_essdai_derived' in data:
        valid &= ~data.is_essdai_derived.fillna(True).astype(bool)
    for _, indices in data.loc[valid].groupby(keys, dropna=False, sort=False).groups.items():
        data.loc[indices, 'q_value'] = multipletests(data.loc[indices, 'p'].astype(float),
            method='fdr_bh' if kind == 'global' else 'holm')[1]
        data.loc[indices, 'family_n_tests'] = len(indices)
    return data


def discovery(global_tests, contrasts):
    if global_tests.empty:
        return global_tests.copy()
    result = global_tests.copy()
    if len(contrasts) and 'contrast_type' in contrasts:
        primary = contrasts.loc[contrasts.contrast_type.eq('slope') & contrasts.multiplicity_role.eq('prespecified')]
        effects = primary.groupby(['variable', 'threshold_abs', 'time_axis', 'scenario'], dropna=False).agg(
            maximum_absolute_slope_difference=('estimate', lambda x: x.abs().max()),
            maximum_standardized_slope_difference=('standardized_effect', lambda x: x.abs().max())).reset_index()
        result = result.merge(effects, on=['variable', 'threshold_abs', 'time_axis', 'scenario'], how='left')
    # Domain/priority ordering, never p-value selection or automatic refitting.
    keys = (['is_priority'] if 'is_priority' in result else []) + ['family', 'domain', 'variable', 'threshold_abs', 'scenario']
    return result.sort_values(keys, ascending=[False]+[True]*(len(keys)-1) if 'is_priority' in result else True).reset_index(drop=True)


def threshold_consistency(contrasts):
    if contrasts.empty or 'contrast_type' not in contrasts:
        return pd.DataFrame()
    d = contrasts.loc[contrasts.contrast_type.eq('slope') & contrasts.scenario.eq('primary') & contrasts.status.eq('ESTIMATED')]
    keys = ['variable', 'contrast', 'time_axis']
    a, b = [d.loc[d.threshold_abs.eq(k), keys+['estimate', 'ci95_low', 'ci95_high', 'n_patients']] for k in (3, 5)]
    if a.empty or b.empty:
        return pd.DataFrame()
    merged = a.merge(b, on=keys, suffixes=('_delta3', '_delta5'))
    merged['same_sign'] = np.sign(merged.estimate_delta3) == np.sign(merged.estimate_delta5)
    merged['interpretation'] = 'Different retrospective group memberships; sign consistency only, not population equivalence'
    return merged

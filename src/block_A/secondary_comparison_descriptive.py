"""Patient-denominated profiles, true follow-up change and observed variability."""
from itertools import combinations
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid

from src.block_A.observed_patterns_v2_figures import LABELS
from src.block_A.secondary_trajectory_analysis import window_contributions


def numeric_summary(values):
    values = pd.Series(values, dtype=float).dropna()
    return dict(n=len(values), mean=values.mean(), sd=values.std(), median=values.median(),
                q25=values.quantile(.25), q75=values.quantile(.75))


def profiles(frame, spec, cfg):
    descriptions, contrasts = [], []
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    first = frame.sort_values('variable_measurement_date').groupby('patient_id').head(1)
    official = frame.loc[frame.variable_measurement_date.eq(frame.clinical_baseline_date)
                         & frame.clinical_episode_id.eq(frame.clinical_baseline_episode_id)]
    for profile, data in (('official_baseline', official), ('first_available', first)):
        for pattern in LABELS:
            g = data.loc[data.observed_pattern.eq(pattern)]
            base = dict(profile=profile, observed_pattern=pattern, n_evaluable=len(g),
                        reference_delay_days_median=g.reference_delay_days.median())
            if spec.type == 'numeric':
                descriptions.append(dict(**base, **numeric_summary(g.value), category='', percentage=np.nan))
            else:
                levels = [0, 1] if spec.type == 'binary' else sorted(data.value.unique(), key=str)
                for level in levels:
                    events, n = int(g.value.eq(level).sum()), len(g)
                    lo, hi = (np.nan, np.nan)
                    if n:
                        from statsmodels.stats.proportion import proportion_confint
                        lo, hi = proportion_confint(events, n, method='wilson')
                    descriptions.append(dict(**base, category=str(level), n_events=events,
                        percentage=100*events/n if n else np.nan, ci95_low=100*lo, ci95_high=100*hi))
        # Contrasts compare one measured value per patient, no repeated-row tests.
        if spec.type != 'numeric' or spec.is_essdai_derived:
            contrasts.append(dict(profile=profile, contrast='global', status='NOT_ESTIMABLE',
                reason='categorical or mechanical profiles descriptive; no independent cross-sectional test prespecified', p=np.nan))
            continue
        series = [data.loc[data.observed_pattern.eq(p), 'value'].astype(float) for p in LABELS]
        if all(len(s) >= cfg.minimum_patients_per_group for s in series) and data.value.nunique() > 1:
            result = stats.kruskal(*series)
            contrasts.append(dict(profile=profile, contrast='global', method='Kruskal-Wallis',
                status='ESTIMATED', statistic=result.statistic, p=result.pvalue, reason=''))
        else:
            contrasts.append(dict(profile=profile, contrast='global', status='NOT_ESTIMABLE', reason='insufficient_support_or_constant', p=np.nan))
        for a, b in combinations(range(4), 2):
            x, y = series[a], series[b]
            base = dict(profile=profile, contrast=list(LABELS)[b]+' - '+list(LABELS)[a],
                        n_reference=len(x), n_comparison=len(y), method='Welch mean difference')
            if (min(len(x), len(y)) < cfg.minimum_patients_per_group
                    or x.var()+y.var() <= 0):
                contrasts.append(dict(**base, status='NOT_ESTIMABLE', reason='insufficient_support_or_constant', p=np.nan))
                continue
            vx, vy = x.var()/len(x), y.var()/len(y)
            se = np.sqrt(vx+vy)
            df = (vx+vy)**2/(vx**2/(len(x)-1)+vy**2/(len(y)-1))
            estimate = y.mean()-x.mean()
            critical = stats.t.ppf(.975, df)
            pooled = np.sqrt(((len(x)-1)*x.var()+(len(y)-1)*y.var())/(len(x)+len(y)-2))
            contrasts.append(dict(**base, status='ESTIMATED', reason='', estimate=estimate, SE=se,
                ci95_low=estimate-critical*se, ci95_high=estimate+critical*se,
                standardized_effect=estimate/pooled if pooled else np.nan,
                p=2*stats.t.sf(abs(estimate/se), df)))
    return pd.DataFrame(descriptions), pd.DataFrame(contrasts)


def changes(frame, spec, cfg):
    results = {name: [] for name in ('change', 'slopes', 'variability', 'auc', 'contrasts')}
    if frame.empty or spec.type != 'numeric' or not spec.expected_longitudinal:
        return {k: pd.DataFrame(v) for k, v in results.items()}
    followup = frame.loc[frame.is_delta_followup].copy()
    # Late references use their own clock, and remain separate from official ones.
    followup['window'] = np.floor(followup.time_since_variable_reference_years/cfg.window_years).astype(int)
    selected = window_contributions(followup)
    for (pattern, source, window), g in selected.groupby(['observed_pattern', 'reference_source', 'window']):
        results['change'].append(dict(observed_pattern=pattern, reference_source=source, window=int(window),
            window_start=window*cfg.window_years, window_end=(window+1)*cfg.window_years,
            change_clock='time_since_variable_reference_years', reference_delay_days_median=g.reference_delay_days.median(),
            mean_time_since_reference=g.time_since_variable_reference_years.mean(),
            **numeric_summary(g.delta), status='DESCRIPTIVE'))
    individual = []
    for _, g in frame.groupby('patient_id'):
        g = g.sort_values('variable_measurement_date')
        if len(g) < 2:
            continue
        t, values = g.time.to_numpy(float), g.value.to_numpy(float)
        span = t[-1]-t[0]
        if span <= 0:
            continue
        row = dict(observed_pattern=g.observed_pattern.iloc[0], n_dates=len(g), span=span,
            slope=np.polyfit(t-t[0], values, 1)[0], within_sd=np.nan, value_range=np.nan,
            successive_abs_change=np.nan, successive_rate=np.nan, reversals=np.nan, residual_sd=np.nan, auc=np.nan)
        if len(g) >= cfg.minimum_variability_dates and span >= cfg.minimum_variability_span_years:
            diffs = np.diff(values)
            signs = np.sign(diffs[diffs != 0])
            residuals = values-np.polyval(np.polyfit(t-t[0], values, 1), t-t[0])
            row.update(within_sd=np.std(values, ddof=1), value_range=np.ptp(values),
                successive_abs_change=np.mean(np.abs(diffs)), successive_rate=np.mean(np.abs(diffs)/np.diff(t)),
                reversals=int(np.sum(signs[1:] != signs[:-1])), residual_sd=np.sqrt(np.sum(residuals**2)/(len(g)-2)))
        lo, hi = cfg.auc_horizon
        if (len(g) >= cfg.minimum_auc_dates and t[0] <= lo and t[-1] >= hi
                and np.max(np.diff(t)) <= cfg.maximum_auc_gap_years):
            inside = (t > lo) & (t < hi)
            grid = np.r_[lo, t[inside], hi]
            vv = np.interp(grid, t, values)  # Interpolate only inside actual observed support.
            if len(grid) >= cfg.minimum_auc_dates:
                row['auc'] = trapezoid(vv, grid)/(hi-lo)
        individual.append(row)
    patient = pd.DataFrame(individual)
    if not patient.empty:
        # Stratification prevents crude pooling of different visit counts and spans.
        patient['span_band'] = pd.cut(patient.span, [0, 1, 3, 5, np.inf], right=False).astype(str)
        for (pattern, n, band), g in patient.groupby(['observed_pattern', 'n_dates', 'span_band']):
            base = dict(observed_pattern=pattern, n_dates=int(n), span_band=band,
                        span_median=g.span.median(), status='DESCRIPTIVE_EXPLORATORY')
            results['slopes'].append(dict(**base, **numeric_summary(g.slope),
                method='estimated individual OLS slope; two-date slopes provisional; no slope-as-observed inference'))
            for metric in ('within_sd', 'value_range', 'successive_abs_change', 'successive_rate', 'reversals', 'residual_sd'):
                results['variability'].append(dict(**base, metric=metric, **numeric_summary(g[metric])))
            results['auc'].append(dict(**base, horizon_start=cfg.auc_horizon[0], horizon_end=cfg.auc_horizon[1],
                **numeric_summary(g.auc), method='trapezoid with within-support boundary interpolation; common fixed horizon'))
    # Do not pretend late-reference deltas at heterogeneous exact times are aligned tests.
    results['contrasts'].append(dict(status='NOT_ESTIMABLE', p=np.nan,
        reason='Window changes descriptive; primary inference uses repeated native values and actual dates'))
    return {k: pd.DataFrame(v) for k, v in results.items()}

"""Aggregate support before inference; all denominators are unique patients."""
import json
import numpy as np
import pandas as pd

from src.block_A.observed_patterns_v2_figures import LABELS
from src.block_A.secondary_trajectory_analysis import support_table, window_contributions
from src.block_A.secondary_comparison_contract import inference_frame


def feasibility(frame, groups, spec, threshold, axis, cfg):
    data = inference_frame(frame, axis)
    # Preserve exact Step 15 support for reconciliation, including preindex data.
    upstream = support_table(frame, groups, spec, threshold)
    rows, windows, protocols, opportunities = [], [], [], []
    for pattern in LABELS:
        g = groups.loc[groups.observed_pattern.eq(pattern)]
        d = data.loc[data.observed_pattern.eq(pattern)] if len(data) else data
        original = frame.loc[frame.observed_pattern.eq(pattern)] if len(frame) else frame
        counts = d.groupby('patient_id').size() if len(d) else pd.Series(dtype=int)
        spans = d.groupby('patient_id').time.agg(lambda s: s.max()-s.min()) if len(d) else pd.Series(dtype=float)
        first = d.sort_values('variable_measurement_date').groupby('patient_id').head(1) if len(d) else d
        official = (original.variable_measurement_date.eq(original.clinical_baseline_date)
                    & original.clinical_episode_id.eq(original.clinical_baseline_episode_id)) if len(original) else pd.Series(dtype=bool)
        inherited = upstream.loc[upstream.observed_pattern.eq(pattern)].iloc[0].to_dict()
        rows.append(dict(**inherited, time_axis=axis, N_group_total=len(g), N_any=len(counts),
            N_official_baseline=int(original.loc[official, 'patient_id'].nunique()) if len(original) else 0,
            N_first_after_baseline=int(first.reference_delay_days.gt(0).sum()) if len(first) else 0,
            N_repeated_ge2=int(counts.ge(2).sum()), N_repeated_ge3=int(counts.ge(3).sum()),
            n_observations=len(d), n_distinct_calendar_dates=d.variable_measurement_date.nunique() if len(d) else 0,
            n_distinct_times=d.time.nunique() if len(d) else 0,
            span_years_q25=spans.quantile(.25), span_years_median=spans.median(), span_years_q75=spans.quantile(.75),
            time_min=d.time.min() if len(d) else np.nan, time_max=d.time.max() if len(d) else np.nan,
            essdai_assessments_median=g.n_distinct_essdai_dates.median(),
            essdai_index_delayed_n=int(g.essdai_index_source.ne('official_baseline').sum()),
            preindex_observations_excluded=len(original)-len(d),
            observation_rate_per_year_median=(counts/spans.replace(0, np.nan)).median(),
            reference_delay_days_median=first.reference_delay_days.median() if len(first) else np.nan))
        opportunities.append(dict(observed_pattern=pattern, N_group_total=len(g),
            n_patients_with_measurement=len(counts), n_observed_measurements=len(d),
            valid_measurement_opportunities=np.nan, missing_percent_valid_opportunities=np.nan,
            status='NOT_ESTIMABLE', reason='No explicit producer-verified measurement-opportunity denominator; absence is not negative'))
        if len(d):
            wd = window_contributions(d)
            for window, part in wd.groupby('window'):
                windows.append(dict(observed_pattern=pattern, window=int(window),
                    window_start=window*cfg.window_years, window_end=(window+1)*cfg.window_years,
                    n_patients=part.patient_id.nunique(), N_group_total=len(g),
                    percent_group_measured=100*part.patient_id.nunique()/len(g)))
            for (window, protocol), part in d.groupby(['window', 'source_protocol']):
                protocols.append(dict(observed_pattern=pattern, window=int(window), source_protocol=protocol,
                    n_patients=part.patient_id.nunique(), n_observations=len(part)))
    support = pd.DataFrame(rows)
    lower, upper = support.time_min.max(), support.time_max.min()
    overlap = bool(support.time_min.notna().all() and upper-lower >= cfg.minimum_observed_span_years)
    reasons = []
    if support.N_any.lt(cfg.minimum_patients_per_group).any():
        reasons.append('insufficient_independent_patients')
    if support.N_repeated_ge2.lt(cfg.minimum_repeated_per_group).any():
        reasons.append('insufficient_repeated_patients')
    if support.n_distinct_times.lt(cfg.minimum_distinct_times).any():
        reasons.append('insufficient_distinct_times')
    if not overlap:
        reasons.append('insufficient_common_time_support')
    if not spec.expected_longitudinal or spec.observation_semantics in ('cumulative_through_episode', 'historical_ever'):
        reasons.append('historical_or_nonlongitudinal_measurement')
    if spec.is_essdai_derived:
        reasons.append('mechanically_related_descriptive_only')
    if spec.type not in ('numeric', 'binary') or spec.unit == 'count':
        reasons.append('no_prespecified_validated_repeated_model_for_type')
    gate = dict(status='NOT_ESTIMABLE' if reasons else 'ELIGIBLE', reason='; '.join(reasons),
        common_time_min=float(lower), common_time_max=float(upper),
        n_patients=int(support.N_any.sum()), n_observations=int(support.n_observations.sum()),
        n_per_group=json.dumps(dict(zip(support.observed_pattern, support.N_any.astype(int))), sort_keys=True),
        support_rule=f'all four groups: N>={cfg.minimum_patients_per_group}, repeated>={cfg.minimum_repeated_per_group}, '
                     f'times>={cfg.minimum_distinct_times}, common span>={cfg.minimum_observed_span_years}y')
    # Include zero-coverage groups for every observed window.
    if windows:
        by = {(r['observed_pattern'], r['window']): r for r in windows}
        for window in sorted({r['window'] for r in windows}):
            for pattern in LABELS:
                if (pattern, window) not in by:
                    total = int(groups.observed_pattern.eq(pattern).sum())
                    windows.append(dict(observed_pattern=pattern, window=window,
                        window_start=window*cfg.window_years, window_end=(window+1)*cfg.window_years,
                        n_patients=0, N_group_total=total, percent_group_measured=0. if total else np.nan))
    return dict(gate=gate, support=support, windows=pd.DataFrame(windows), protocols=pd.DataFrame(protocols),
                opportunities=pd.DataFrame(opportunities), overlap=pd.DataFrame([dict(
                    common_time_min=lower, common_time_max=upper, sufficient_overlap=overlap)]))

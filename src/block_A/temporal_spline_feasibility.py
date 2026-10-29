"""Frozen common sample and stricter local four-group spline support."""
import json
import numpy as np
import pandas as pd
from src.block_A.secondary_comparison_contract import inference_frame
from src.block_A.secondary_comparison_models import PATTERNS
from src.block_A.secondary_trajectory_analysis import ContractError
from src.block_A.temporal_spline_design import FrozenBasis


def evaluate(frame, groups, spec, cfg, axis, fit_horizon=None, report_horizon=None):
    data = inference_frame(frame, axis)
    if len(data):
        if (data.threshold_abs.nunique() != 1 or data.groupby('patient_id').observed_pattern.nunique().gt(1).any()
                or data.duplicated(['patient_id', 'variable_measurement_date']).any()):
            raise ContractError('threshold/group/date immutability violation')
        expected = groups.set_index('patient_id').observed_pattern
        if not data.observed_pattern.eq(data.patient_id.map(expected)).all():
            raise ContractError('model groups differ from fixed upstream groups')
        data = data.loc[data.time.ge(0)].copy()
        if fit_horizon is not None:
            data = data.loc[data.time.le(fit_horizon)].copy()
    # Crop to common range, then recompute every count on EXACT model rows.
    limits = [data.loc[data.observed_pattern.eq(g), 'time'] if len(data) else pd.Series(dtype=float) for g in PATTERNS]
    lo = max((s.min() for s in limits), default=np.nan)
    hi = min((s.max() for s in limits), default=np.nan)
    valid_bounds = all(len(s) for s in limits) and np.isfinite([lo, hi]).all() and hi > lo
    if valid_bounds:
        data = data.loc[data.time.between(lo, hi)].sort_values(['patient_id', 'time']).reset_index(drop=True)
    else:
        data = data.iloc[0:0].copy()
    rows, reasons = [], []
    for g in PATTERNS:
        d = data.loc[data.observed_pattern.eq(g)] if len(data) else data
        sizes = d.groupby('patient_id').time.nunique() if len(d) else pd.Series(dtype=int)
        spans = d.groupby('patient_id').time.agg(lambda s: s.max()-s.min()) if len(d) else pd.Series(dtype=float)
        rows.append(dict(observed_pattern=g, n_patients=len(sizes), n_repeated=int(sizes.ge(2).sum()),
            n_three_dates=int((sizes.ge(3) & spans.ge(cfg.minimum_observed_span_years)).sum()),
            n_distinct_times=d.time.nunique() if len(d) else 0, n_observations=len(d)))
    counts = pd.DataFrame(rows)
    if counts.n_patients.lt(cfg.minimum_patients_per_group).any():
        reasons.append('insufficient_independent_patients')
    if counts.n_repeated.lt(cfg.minimum_repeated_per_group).any():
        reasons.append('insufficient_repeated_patients')
    if counts.n_distinct_times.lt(cfg.minimum_distinct_times).any():
        reasons.append('insufficient_distinct_times')
    if not valid_bounds or hi-lo < cfg.minimum_observed_span_years:
        reasons.append('insufficient_common_time_span')
    if spec.is_essdai_derived or spec.canonical_column.startswith(('essdai__', 'ovl__')):
        reasons.append('mechanically_related_descriptive_only')
    if not spec.expected_longitudinal or spec.observation_semantics in ('historical_ever', 'cumulative_through_episode'):
        reasons.append('historical_or_nonlongitudinal')
    if (spec.type not in ('numeric', 'binary') or spec.unit in ('count', 'score', 'foci/4 mm²')
            or spec.variable_id.startswith(('ocular_oxford', 'ocular_van_bijsterveld', 'sgus_summary'))):
        reasons.append('no_validated_family_for_ordinal_count_or_complex_score')
    # Categorical outcomes such as ovl__status are inventoried but cannot
    # enter numeric spline families or numeric-only finite-value checks.
    if len(data) and spec.type in ('numeric', 'binary'):
        values = data.value.to_numpy(dtype=float)
        if not np.isfinite(values).all() or data.value.nunique() < 2:
            reasons.append('constant_or_nonfinite_outcome')
    if spec.type == 'binary' and len(data):
        if not set(data.value.unique()).issubset({0, 1}):
            reasons.append('invalid_binary_values')
        for g in PATTERNS:
            d = data.loc[data.observed_pattern.eq(g)]
            if (d.loc[d.value.eq(1), 'patient_id'].nunique() < cfg.minimum_events_per_group or
                    d.loc[d.value.eq(0), 'patient_id'].nunique() < cfg.minimum_nonevents_per_group):
                reasons.append('insufficient_binary_event_or_nonevent_patients')
                break
    grid_hi = min(hi, report_horizon) if report_horizon is not None else hi
    grid = (np.arange(np.ceil(lo/cfg.grid_step)*cfg.grid_step, grid_hi+1e-9, cfg.grid_step)
            if valid_bounds and grid_hi >= lo else np.array([]))
    local = []
    for t in grid:
        by_group = []
        for g in PATTERNS:
            d = data.loc[data.observed_pattern.eq(g) & data.time.sub(t).abs().le(cfg.prediction_window_years/2)]
            n = d.patient_id.nunique()
            denominator = int(groups.observed_pattern.eq(g).sum())
            by_group.append(dict(observed_pattern=g, time=float(t), n_observed_local=n,
                percent_group_measured=100*n/denominator if denominator else np.nan,
                observed_span_years=float(d.time.max()-d.time.min()) if len(d) else np.nan,
                local_supported=n >= cfg.minimum_prediction_patients,
                local_window_width_years=cfg.prediction_window_years))
        supported = all(r['local_supported'] for r in by_group)
        local.extend(dict(**r, four_group_supported=supported) for r in by_group)
    support = pd.DataFrame(local, columns=['observed_pattern', 'time', 'n_observed_local', 'percent_group_measured',
        'observed_span_years', 'local_supported', 'local_window_width_years', 'four_group_supported'])
    # Three distributed regions and three annual windows must ALL be represented.
    ok_times = support.loc[support.four_group_supported, 'time'].unique()
    region_n = len(set(np.clip(np.floor((ok_times-lo)/(grid_hi-lo)*3), 0, 2).astype(int))) if grid_hi > lo else 0
    annual_n = len(set(np.floor(ok_times).astype(int)))
    c_reasons = []
    if counts.n_three_dates.lt(cfg.minimum_three_date_patients).any():
        c_reasons.append('insufficient_three_date_patients_with_span')
    if region_n < 3 or annual_n < 3:
        c_reasons.append('insufficient_distributed_local_four_group_support')
    basis, matrices, designs = None, {}, []
    if not reasons:
        try:
            basis = FrozenBasis.freeze(data.time)
        except ValueError as error:
            c_reasons.append(str(error))
        for model in ('A', 'B', 'C'):
            # A remains possible even if spline knots are degenerate.
            if basis is None and model != 'A':
                continue
            if basis is None:
                from src.block_A.secondary_comparison_models import design
                m = design(data)
            else:
                m = basis.matrix(data.observed_pattern, data.time, model)
            rank, condition = np.linalg.matrix_rank(m), np.linalg.cond(m)
            designs.append(dict(model=model, design_rank=int(rank), n_columns=m.shape[1], design_condition=float(condition)))
            if rank == m.shape[1] and condition <= cfg.maximum_design_condition:
                matrices[model] = m
    a_reason = '; '.join(reasons or ([] if 'A' in matrices else ['singular_or_ill_conditioned_A_design']))
    b_reason = a_reason or ('' if 'B' in matrices else 'nonestimable_nonlinear_basis_or_B_design')
    c_reason = b_reason or '; '.join(c_reasons or ([] if 'C' in matrices else ['singular_or_ill_conditioned_C_design']))
    gate = dict(status='ELIGIBLE' if not c_reason else 'NOT_ESTIMABLE_SPLINE_GROUP',
        reason=c_reason, eligible_A=not bool(a_reason), eligible_B=not bool(b_reason), eligible_C=not bool(c_reason),
        reason_A=a_reason, reason_B=b_reason, reason_C=c_reason,
        common_time_min=float(lo), common_time_max=float(hi), n_patients=int(counts.n_patients.sum()),
        n_observations=len(data), n_supported_regions=region_n, n_supported_annual_windows=annual_n,
        n_per_group=json.dumps(dict(zip(counts.observed_pattern, counts.n_patients)), sort_keys=True))
    return dict(data=data, gate=gate, support=support, counts=counts, basis=basis, matrices=matrices, designs=designs)

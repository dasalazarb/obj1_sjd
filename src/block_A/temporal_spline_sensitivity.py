"""Prespecified independent scenario fits and composition-aware comparisons."""
import numpy as np
import pandas as pd
from src.block_A.secondary_comparison_contract import inference_frame
from src.block_A.temporal_spline_results import OUTCOME_KEYS
from src.block_A.secondary_trajectory_analysis import ContractError


def scenarios(frame, axis, cfg, mode):
    # name, rows, clock, estimation cap, inference cap
    yield 'main', frame, axis, cfg.max_primary_years, cfg.max_primary_years
    if mode == 'none':
        return
    yield 'full_common_horizon', frame, axis, None, None
    # Identical full-horizon estimation rows and knots, only inference domain differs.
    yield 'full_sample_primary_domain', frame, axis, None, cfg.max_primary_years
    if frame.empty:
        for name in ('official_variable_baseline_only', 'delayed_variable_baseline_only',
                     'variable_ge3_dates', 'essdai_ge3_dates', 'alternate_time_axis'):
            other_axis = 'essdai_index' if axis == 'clinical_baseline' else 'clinical_baseline'
            yield name, frame.copy(), other_axis if name == 'alternate_time_axis' else axis, cfg.max_primary_years, cfg.max_primary_years
        if mode == 'full':
            for name in ('essdai_official_index', 'essdai_delayed_index'):
                yield name, frame.copy(), axis, cfg.max_primary_years, cfg.max_primary_years
        return
    yield 'official_variable_baseline_only', frame.loc[frame.reference_source.eq('official_clinical_baseline')].copy(), axis, cfg.max_primary_years, cfg.max_primary_years
    yield 'delayed_variable_baseline_only', frame.loc[frame.reference_source.ne('official_clinical_baseline')].copy(), axis, cfg.max_primary_years, cfg.max_primary_years
    observed = inference_frame(frame, axis)
    if len(observed):
        counts = observed.groupby('patient_id').variable_measurement_date.transform('nunique')
        yield 'variable_ge3_dates', observed.loc[counts.ge(3)].copy(), axis, cfg.max_primary_years, cfg.max_primary_years
    yield 'essdai_ge3_dates', frame.loc[frame.n_distinct_essdai_dates.ge(3)].copy(), axis, cfg.max_primary_years, cfg.max_primary_years
    other_axis = 'essdai_index' if axis == 'clinical_baseline' else 'clinical_baseline'
    other = frame.copy()
    other['time'] = other['time_from_'+other_axis+'_years']
    yield 'alternate_time_axis', other, other_axis, cfg.max_primary_years, cfg.max_primary_years
    if mode == 'full':
        yield 'essdai_official_index', frame.loc[frame.essdai_index_source.eq('official_baseline')].copy(), axis, cfg.max_primary_years, cfg.max_primary_years
        yield 'essdai_delayed_index', frame.loc[frame.essdai_index_source.ne('official_baseline')].copy(), axis, cfg.max_primary_years, cfg.max_primary_years


def match_reference(reference, comparison, keys, suffixes):
    """Each comparison row matches at most one semantically identical reference."""
    try:
        joined = reference.merge(comparison, on=keys, suffixes=suffixes, validate='one_to_many')
    except pd.errors.MergeError as error:
        raise ContractError('nonunique sensitivity reference outcome') from error
    expected = comparison[keys].merge(reference[keys].drop_duplicates(), on=keys, validate='many_to_one')
    if len(joined) != len(expected):
        raise ContractError('sensitivity join changed matched row count')
    return joined


def consistency(tables):
    """Match identical tests and pair/time/estimands; thresholds are not independent."""
    rows = []
    slope_tables = [t['01_linear_pairwise_slopes'] for t in tables.values() if len(t['01_linear_pairwise_slopes'])]
    if slope_tables:
        slopes = pd.concat(slope_tables, ignore_index=True)
        keys = OUTCOME_KEYS+['time_axis', 'contrast']
        primary = slopes.loc[slopes.threshold_abs.eq(3) & slopes.scenario.eq('main')]
        others = slopes.loc[~(slopes.threshold_abs.eq(3) & slopes.scenario.eq('main'))]
        matched = match_reference(primary, others, keys, ('_primary', '_comparison'))
        for r in matched.to_dict('records'):
            rows.append(dict(**{k: r[k] for k in keys}, test='linear_A_slope_horizon_diagnostic',
                scenario=r['scenario_comparison'], threshold_abs=r['threshold_abs_comparison'],
                estimate_primary=r['estimate_primary'], estimate_comparison=r['estimate_comparison'],
                same_sign=np.sign(r['estimate_primary']) == np.sign(r['estimate_comparison']),
                same_estimation_rows=r['sample_fingerprint_primary'] == r['sample_fingerprint_comparison'],
                n_patients_primary=r['n_patients_analysis_primary'], n_patients_comparison=r['n_patients_analysis_comparison'],
                interpretation='Slope/horizon/composition diagnostic for every variable; IgA receives no special selection rule'))
    global_tables = [t['02_global_temporal_tests'] for t in tables.values() if len(t['02_global_temporal_tests'])]
    if global_tables:
        tests = pd.concat(global_tables, ignore_index=True)
        full = tests.loc[tests.scenario.eq('full_common_horizon')]
        restricted = tests.loc[tests.scenario.eq('full_sample_primary_domain')]
        shared = match_reference(full, restricted, OUTCOME_KEYS+['time_axis', 'test', 'threshold_abs'], ('_full', '_restricted'))
        for r in shared.to_dict('records'):
            rows.append(dict(**{k: r[k] for k in OUTCOME_KEYS}, time_axis=r['time_axis'], test=r['test'],
                threshold_abs=r['threshold_abs'], scenario='full_sample_primary_domain',
                reference_scenario='full_common_horizon',
                same_estimation_rows=r['sample_fingerprint_full'] == r['sample_fingerprint_restricted'],
                primary_status=r['status_full'], comparison_status=r['status_restricted'],
                primary_q=r['q_value_full'], comparison_q=r['q_value_restricted'],
                interpretation='Reporting-domain sensitivity on identical full estimation rows; check eligible family and support separately'))
        for identity, d in tests.groupby(OUTCOME_KEYS+['time_axis', 'test']):
            outcome = dict(zip(OUTCOME_KEYS, identity[:-2]))
            axis, test = identity[-2:]
            primary = d.loc[d.threshold_abs.eq(3) & d.scenario.eq('main')]
            if primary.empty:
                continue
            a = primary.iloc[0]
            for b in d.to_dict('records'):
                if b['threshold_abs'] == 3 and b['scenario'] == 'main':
                    continue
                rows.append(dict(**outcome, time_axis=axis, test=test, scenario=b['scenario'],
                    threshold_abs=b['threshold_abs'], primary_status=a.status, comparison_status=b['status'],
                    primary_q=a.q_value, comparison_q=b['q_value'],
                    n_patients_primary=a.n_patients_analysis, n_patients_comparison=b['n_patients_analysis'],
                    same_estimation_rows=a.sample_fingerprint == b['sample_fingerprint'],
                    interpretation='Sensitivity; overlapping patients and retrospectively different threshold memberships'))
    # IgA and every other variable use exactly the same six-pair matching rule.
    pair_tables = [t['04_pairwise_temporal_differences'] for t in tables.values() if len(t['04_pairwise_temporal_differences'])]
    if pair_tables:
        pairs = pd.concat(pair_tables, ignore_index=True)
        keys = OUTCOME_KEYS+['time_axis', 'contrast', 'contrast_type', 'time']
        primary = pairs.loc[pairs.threshold_abs.eq(3) & pairs.scenario.eq('main')]
        other = pairs.loc[~(pairs.threshold_abs.eq(3) & pairs.scenario.eq('main'))]
        merged = match_reference(primary, other, keys, ('_primary', '_comparison'))
        for r in merged.to_dict('records'):
            rows.append(dict(**{k: r[k] for k in keys}, scenario=r['scenario_comparison'],
                threshold_abs=r['threshold_abs_comparison'], estimate_primary=r['estimate_primary'],
                estimate_comparison=r['estimate_comparison'], same_sign=np.sign(r['estimate_primary']) == np.sign(r['estimate_comparison']),
                interpretation='Response-scale contrast sign consistency; no independent replication'))
    return pd.DataFrame(rows)


def bootstrap_patients(evaluation, fits, spec, cfg):
    """Optional shortlist validation of C covariance bands by whole-patient resampling.

    Freeze sample, knots, projection, grid, and model family. Failed replicates
    remain failures. Outputs are aggregate quantiles only; no individual rows.
    """
    from src.block_A.temporal_spline_models import fit_family
    from src.block_A.temporal_spline_contrasts import function_and_gradient
    from src.block_A.secondary_comparison_models import PATTERNS
    from itertools import combinations
    if not cfg.bootstrap_replicates or 'C' not in fits:
        return pd.DataFrame([dict(status='NOT_RUN', reason='Optional frozen-specification shortlist patient bootstrap',
                                  requested_replicates=cfg.bootstrap_replicates, successful_replicates=0)])
    if spec.type == 'binary':
        return pd.DataFrame([dict(status='NOT_ESTIMABLE', reason='Response-scale bootstrap calibration not validated for binary timing')])
    original = fits['C']
    data, basis = evaluation['data'], evaluation['basis']
    times = sorted(evaluation['support'].loc[evaluation['support'].four_group_supported, 'time'].unique())
    if not times:
        return pd.DataFrame([dict(status='NOT_ESTIMABLE', reason='No supported grid')])
    targets = [(b, a, t, kind) for a, b in combinations(PATTERNS, 2) for kind in ('level', 'change', 'velocity') for t in times]
    vectors = []
    for b, a, t, kind in targets:
        _, ga = function_and_gradient(basis, original, a, t, kind, times[0])
        _, gb = function_and_gradient(basis, original, b, t, kind, times[0])
        vectors.append(gb-ga)
    gradients = np.asarray(vectors)
    se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', gradients, original.covariance, gradients), 0))
    valid = se > 1e-12
    rng, maxima = np.random.default_rng(cfg.seed+2), []
    by_patient = {pid: d for pid, d in data.groupby('patient_id')}
    pools = {g: data.loc[data.observed_pattern.eq(g), 'patient_id'].unique() for g in PATTERNS}
    for _ in range(cfg.bootstrap_replicates):
        parts = []
        for g in PATTERNS:
            for index, pid in enumerate(rng.choice(pools[g], len(pools[g]), replace=True)):
                part = by_patient[pid].copy()
                part['patient_id'] = g+'_bootstrap_'+str(index)
                parts.append(part)
        replicate = pd.concat(parts, ignore_index=True)
        ev = dict(evaluation, data=replicate,
                  matrices={m: basis.matrix(replicate.observed_pattern, replicate.time, m) for m in fits})
        boot, _, _ = fit_family(ev, spec, cfg)
        if 'C' not in boot or boot['C'].family != original.family:
            continue
        maxima.append(float(np.max(np.abs(gradients[valid]@(boot['C'].beta-original.beta))/se[valid])))
    enough = len(maxima) >= max(100, int(.8*cfg.bootstrap_replicates))
    return pd.DataFrame([dict(status='VALIDATION_ESTIMATED' if enough else 'NOT_ESTIMABLE',
        reason='' if enough else 'Insufficient successful same-family patient bootstrap replicates',
        requested_replicates=cfg.bootstrap_replicates, successful_replicates=len(maxima),
        bootstrap_sup_t_critical=float(np.quantile(maxima, .95)) if enough else np.nan,
        seed=cfg.seed+2, bootstrap_family='six_pairs_x_supported_grid_x_level_change_velocity',
        method='stratified_whole_patient_fixed_basis_bootstrap_sensitivity')])

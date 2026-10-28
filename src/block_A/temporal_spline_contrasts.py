"""Separate levels, reference changes and velocities with supported six-pair bands."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.special import expit
from src.block_A.secondary_comparison_models import PATTERNS
from src.block_A.temporal_spline_inference import simultaneous_critical, separation_intervals


def function_and_gradient(basis, fit, group, t, kind='level', reference=None, binary=False):
    x = basis.row(group, t, fit.model)
    dx = basis.row(group, t, fit.model, derivative=1)
    beta = fit.beta
    if binary:
        probability = expit(x@beta)
        slope = probability*(1-probability)
        value, gradient = probability, slope*x
        if kind == 'velocity':
            value = slope*(dx@beta)
            gradient = slope*dx+slope*(1-2*probability)*(dx@beta)*x
    else:
        value, gradient = float(x@beta), x
        if kind == 'velocity':
            value, gradient = float(dx@beta), dx
    if kind == 'change':
        initial, initial_gradient = function_and_gradient(basis, fit, group, reference, binary=binary)
        value, gradient = value-initial, gradient-initial_gradient
    return float(value), gradient


def effect(value, gradient, cov):
    variance = float(gradient@cov@gradient)
    if not np.isfinite(variance) or variance < -1e-10:
        raise ValueError('invalid contrast variance')
    se = np.sqrt(max(0., variance))
    return dict(estimate=value, SE=se, ci95_low=value-1.959963984540054*se, ci95_high=value+1.959963984540054*se)


def characterize(evaluation, fits, spec, cfg):
    output = {k: [] for k in ('curves', 'changes', 'derivatives', 'pairs', 'bands', 'intervals', 'characteristics', 'registry')}
    support, basis = evaluation['support'], evaluation['basis']
    times = sorted(support.loc[support.four_group_supported, 'time'].unique())
    if basis is None or not times:
        output['registry'].append(dict(component='temporal_functions', status='NOT_ESTIMABLE', reason='no_supported_grid_or_basis'))
        return {k: pd.DataFrame(v) for k, v in output.items()}
    binary = spec.type == 'binary'
    # Deterministic common reference, first supported prespecified grid point (0 preferred).
    reference = float(times[0])
    for model, fit in fits.items():
        for t in times:
            for group in PATTERNS:
                value, gradient = function_and_gradient(basis, fit, group, t, binary=binary)
                local_n = int(support.loc[support.observed_pattern.eq(group) & support.time.eq(t), 'n_observed_local'].iloc[0])
                row = dict(model=model, observed_pattern=group, time=float(t), n_observed_local=local_n,
                    status='ESTIMATED', effect_scale='probability' if binary else spec.unit, **effect(value, gradient, fit.covariance))
                if binary:
                    # CI for a probability via the linear predictor; differences use delta method.
                    x = basis.row(group, t, model)
                    eta = effect(float(x@fit.beta), x, fit.covariance)
                    row.update(ci95_low=expit(eta['ci95_low']), ci95_high=expit(eta['ci95_high']))
                output['curves'].append(row)
    if 'C' not in fits:
        output['registry'].append(dict(component='C_temporal_contrasts', status='NOT_ESTIMABLE_SPLINE_GROUP',
                                       reason=evaluation['gate']['reason_C'] or 'C_fit_failed'))
        return {k: pd.DataFrame(v) for k, v in output.items()}
    fit = fits['C']
    group_rows, group_gradients, pair_rows, pair_gradients = [], [], [], []
    for kind in ('level', 'change', 'velocity'):
        for t in times:
            functions = {g: function_and_gradient(basis, fit, g, t, kind, reference, binary) for g in PATTERNS}
            for group, (value, gradient) in functions.items():
                row = dict(model='C', observed_pattern=group, time=float(t), contrast_type=kind,
                    reference_time=reference, reference_policy='first_four_group_supported_grid_point',
                    status='ESTIMATED', effect_scale=('probability' if binary else spec.unit)+('/year' if kind == 'velocity' else ''),
                    **effect(value, gradient, fit.covariance))
                group_rows.append(row)
                group_gradients.append(gradient)
            for a, b in combinations(PATTERNS, 2):
                va, ga = functions[a]
                vb, gb = functions[b]
                row = dict(model='C', contrast=b+' - '+a, contrast_type=kind, time=float(t), reference_time=reference,
                    multiplicity_role='prespecified_vs_reference' if a == PATTERNS[0] else 'exploratory',
                    status='ESTIMATED', effect_scale=('probability difference' if binary else spec.unit)+('/year' if kind == 'velocity' else ''),
                    **effect(vb-va, gb-ga, fit.covariance))
                pair_rows.append(row)
                pair_gradients.append(gb-ga)
    # Joint over ALL six pairs, ALL supported times and all three estimands.
    # For binomial GEE this is an explicitly asymptotic response-scale delta sup-t band.
    critical = simultaneous_critical(pair_gradients, fit.covariance, cfg.simulations, cfg.seed)
    method = 'coefficient_simulation_sup_t' + ('_response_delta_asymptotic' if binary else '_gaussian')
    for row in pair_rows:
        row.update(sim95_low=row['estimate']-critical*row['SE'], sim95_high=row['estimate']+critical*row['SE'],
            critical_value=critical, method=method, simulations=cfg.simulations, seed=cfg.seed,
            simultaneous_family='six_pairs_x_supported_grid_x_level_change_velocity')
    group_critical = simultaneous_critical(group_gradients, fit.covariance, cfg.simulations, cfg.seed+1)
    for row in group_rows:
        row.update(sim95_low=row['estimate']-group_critical*row['SE'], sim95_high=row['estimate']+group_critical*row['SE'],
                   method=method, critical_value=group_critical)
        if row['contrast_type'] == 'change':
            output['changes'].append(row)
        elif row['contrast_type'] == 'velocity':
            output['derivatives'].append(row)
        else:
            for curve in output['curves']:
                if curve['model'] == 'C' and curve['observed_pattern'] == row['observed_pattern'] and curve['time'] == row['time']:
                    curve.update(sim95_low=max(0., row['sim95_low']) if binary else row['sim95_low'],
                                 sim95_high=min(1., row['sim95_high']) if binary else row['sim95_high'])
    output['pairs'], output['bands'] = pair_rows, pair_rows
    bands = pd.DataFrame(pair_rows)
    output['intervals'] = separation_intervals(bands, cfg.grid_step).to_dict('records')
    # Timing requires stable patient-bootstrap validation; never label a grid extremum as biological timing.
    for group in PATTERNS:
        output['characteristics'].append(dict(observed_pattern=group, status='NOT_ESTIMABLE_TIMING',
            reason='No validated interior maximum / monotone T50 with patient-bootstrap timing uncertainty',
            maximum_velocity_time=np.nan, T50=np.nan))
    return {k: pd.DataFrame(v) for k, v in output.items()}

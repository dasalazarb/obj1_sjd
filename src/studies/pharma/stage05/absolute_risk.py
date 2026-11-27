"""Exploratory Q projections from equivalent Step 03 models."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.linalg import expm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .contracts import DAYS_PER_YEAR, POPS, flag
from .step03_adapter import Step03

VERSION = 'age_adjusted_clustered'
ROUTES = tuple(f'{a}→{b}' for a in POPS for b in POPS if a != b)
ARRIVAL = tuple(route for route in ROUTES if not route.startswith('Pop1'))
ESTIMANDS = {'first_arrival': ARRIVAL, 'occupancy': ROUTES}
FIT_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError, PerfectSeparationError)
RISK_FIELDS = ['analysis_unit', 'cohort', 'index_definition', 'marker', 'assay', 'unit', 'from_pop',
               'to_pop', 'scheme', 'years', 'n_patients', 'n_reference_patients', 'n_at_risk_reference',
               'p25', 'p75', 'reference_age', 'probability_p25', 'probability_p75',
               'probability_p25_ci95_low', 'probability_p25_ci95_high',
               'probability_p75_ci95_low', 'probability_p75_ci95_high', 'estimate',
               'ci95_low', 'ci95_high', 'n_bootstrap_successful', 'extrapolation', 'status', 'reason']
SUPPORT_FIELDS = ['marker', 'assay', 'unit', 'scheme', 'model_version', 'route', 'n_intervals', 'n_patients',
                  'n_events', 'n_event_patients', 'biomarker_mean', 'biomarker_sd',
                  'max_abs_log_coefficient_delta', 'status', 'reason']


def generator(params: pd.Series, z: float, age: float, *, absorbing: bool) -> np.ndarray:
    required = ARRIVAL if absorbing else ROUTES
    matrix = np.zeros((3, 3))
    for route in required:
        origin, target = route.split('→')
        eta = params[f'intercept::{route}'] + params[f'biomarker::{route}'] * z + params['age'] * age
        with np.errstate(over='ignore', under='ignore'):
            intensity = np.exp(eta)
        if not np.isfinite(intensity) or intensity <= 0:
            raise ValueError('nonfinite_or_nonpositive_intensity')
        matrix[POPS.index(origin), POPS.index(target)] = intensity
    np.fill_diagonal(matrix, -matrix.sum(axis=1))
    if not np.allclose(matrix.sum(axis=1), 0, atol=1e-9):
        raise ValueError('invalid_generator_row_sums')
    return matrix


def probabilities(matrix: np.ndarray, horizons: list, *, absorbing: bool) -> np.ndarray:
    result = np.array([expm(year * matrix) for year in horizons])
    if not (np.isfinite(result).all() and np.allclose(result.sum(axis=2), 1, atol=1e-9)
            and result.min() >= -1e-9 and result.max() <= 1 + 1e-9):
        raise ValueError('invalid_transition_probabilities')
    if absorbing and (np.diff(result[:, 1:, 0], axis=0) < -1e-9).any():
        raise ValueError('nonmonotone_first_arrival')
    return result.clip(0, 1)


def _projection(params, percentiles, age, mean, sd, horizons, schemes):
    output = {}
    for scheme in schemes:
        absorbing = scheme == 'first_arrival'
        output[scheme] = np.stack([
            probabilities(generator(params, (value - mean) / sd, age, absorbing=absorbing),
                          horizons, absorbing=absorbing)[:, :, 0]
            for value in percentiles])
    if set(output) == set(ESTIMANDS) and (output['first_arrival'] + 1e-9 < output['occupancy']).any():
        raise ValueError('occupancy_exceeds_first_arrival')
    return output


def check_upstream(inputs, adapter):
    """Cross-check the full candidate trace and published Step 03 support keys."""
    tables = inputs.tables
    expected = adapter.candidates(tables['sensitivity'])
    actual = tables['candidates']
    if actual.exposure.isna().any() or actual.exposure.duplicated().any():
        raise ValueError('Canonical Pharma 03 candidate keys are invalid')
    if dict(zip(expected.exposure, flag(expected.eligible_for_03))) != dict(zip(actual.exposure, flag(actual.eligible_for_03))):
        raise ValueError('Canonical Pharma 02/03 candidate trace mismatch')
    eligible = set(actual.loc[flag(actual.eligible_for_03), 'exposure'])
    expected_support = {(assay, route) for assay in eligible for route in ROUTES}
    expected_models = {(assay, route, version) for assay, route in expected_support
                       for version in ('same_sample_clustered', VERSION)}
    support = tables['support']
    models = tables['models']
    for table, keys, intended in (
        (support, ['biomarker', 'transition_id'], expected_support),
        (models, ['biomarker', 'transition_id', 'model_version'], expected_models),
    ):
        if table.duplicated(keys).any() or set(table[keys].itertuples(index=False, name=None)) != intended:
            raise ValueError('Canonical Pharma 03 model/support keys disagree with candidate trace')
    if (tables['model_summary'].biomarker.duplicated().any() or
            set(tables['model_summary'].biomarker) != eligible):
        raise ValueError('Canonical Pharma 03 summary keys disagree with candidate trace')
    prepared = {}
    for assay in sorted(eligible):
        if f'from_lab__{assay}' not in inputs.intervals:
            raise ValueError(f'Canonical Pharma 01 lacks the mapped Step 03 assay: {assay}')
        data, stacked, recomputed, mean, sd, reason = adapter.prepare(
            inputs.intervals, assay, inputs.preflight['minimum_events'])
        reported = support.loc[support.biomarker.eq(assay)].set_index('transition_id').reindex(ROUTES)
        rebuilt = recomputed.set_index('transition_id').reindex(ROUTES)
        for col in ('n_intervals_at_risk', 'n_patients_at_risk', 'n_events', 'n_patients_with_event'):
            if not np.array_equal(pd.to_numeric(reported[col]), rebuilt[col]):
                raise ValueError(f'Canonical Pharma 03 support and Pharma 01 disagree: {assay}, {col}')
        if not np.array_equal(flag(reported.supported_for_model), flag(rebuilt.supported_for_model)):
            raise ValueError(f'Canonical Pharma 03 support gate differs: {assay}')
        for col, value in [('biomarker_mean', mean), ('biomarker_sd', sd)]:
            upstream = pd.to_numeric(models.loc[models.biomarker.eq(assay), col])
            if not np.allclose(upstream, value, atol=1e-8, rtol=1e-6, equal_nan=True):
                raise ValueError(f'Canonical Pharma 03 scaling and Pharma 01 disagree: {assay}, {col}')
        for _, row in models.loc[models.biomarker.eq(assay)].iterrows():
            counts = rebuilt.loc[row.transition_id]
            if (row.n_intervals, row.n_patients, row.n_events) != (
                counts.n_intervals_at_risk, counts.n_patients_at_risk, counts.n_events
            ):
                raise ValueError(f'Canonical Pharma 03 model and support denominators disagree: {assay}')
        prepared[assay] = (data, stacked, rebuilt, mean, sd, reason)
    return prepared


def refit_equivalence(adapter, stacked, routes, published, settings):
    if not routes:
        raise ValueError('no_supported_routes')
    params = adapter.fit(stacked, routes)
    observed, expected = [], []
    for route in routes:
        row = published.loc[route]
        if row.model_status != 'success':
            continue
        observed.extend([params[f'biomarker::{route}'], params['age']])
        expected.extend(np.log([row.estimate_per_sd, row.age_estimate]))
    if not observed or not np.isfinite(expected).all():
        raise ValueError('no_finite_upstream_coefficients')
    delta = float(np.max(np.abs(np.array(observed) - expected)))
    if not np.allclose(observed, expected, rtol=settings['refit_relative_tolerance'],
                       atol=settings['refit_absolute_tolerance']):
        raise ValueError('step03_refit_coefficient_mismatch')
    return params, delta


def bootstrap(adapter, data, assay, routes, references, settings, rng):
    """Keep reference ages/quantiles fixed while refitting and rescaling each draw."""
    draws = {scheme: {pop: [] for pop in references} for scheme in ESTIMANDS}
    attempts = {scheme: Counter() for scheme in ESTIMANDS}
    clusters = [group for _, group in data.groupby('patient_id', sort=False)]
    horizons = settings['horizons_years']
    attempted = {scheme for scheme, required in ESTIMANDS.items() if set(required).issubset(routes)}
    if references:
        attempted &= {scheme for _, _, schemes, _ in references.values() for scheme in schemes}
    for _ in range(settings['bootstrap_replicates']):
        chosen = rng.integers(len(clusters), size=len(clusters))
        sampled = pd.concat([clusters[i].assign(patient_id=j) for j, i in enumerate(chosen)], ignore_index=True)
        _, stack, support, mean, sd, reason = adapter.prepare(sampled, assay, settings['minimum_events'])
        if reason:
            for scheme in attempted:
                attempts[scheme][reason] += 1
            continue
        by_route = support.set_index('transition_id')
        supported = _supported_routes(by_route, settings)
        # Keep the model specification fixed. Additional patient-level gates
        # apply to the routes required by each estimand.
        if not set(routes).issubset(by_route.index[flag(by_route.supported_for_model)]):
            for scheme in attempted:
                attempts[scheme]['bootstrap_route_support_failed'] += 1
            continue
        usable = {scheme for scheme in attempted if set(ESTIMANDS[scheme]).issubset(supported)}
        for scheme in attempted - usable:
            attempts[scheme]['bootstrap_estimand_patient_support_failed'] += 1
        if not usable:
            continue
        try:
            params = adapter.fit(stack, routes)
            projections = {
                pop: _projection(params, pct, age, mean, sd, horizons, [scheme for scheme in schemes if scheme in usable])
                for pop, (pct, age, schemes, _) in references.items()
            }
        except FIT_ERRORS as exc:
            for scheme in usable:
                attempts[scheme][f'fit_or_projection:{type(exc).__name__}'] += 1
            continue
        for pop, output in projections.items():
            for scheme, values in output.items():
                index = POPS.index(pop)
                draws[scheme][pop].append(values[:, :, index])
    return draws, attempts


def _supported_routes(support, settings):
    return set(support.index[flag(support.supported_for_model) &
                             support.n_patients_with_event.ge(settings['minimum_event_patients']) &
                             support.n_patients_at_risk.ge(settings['minimum_model_patients'])])


def absolute_risk(inputs, settings):
    adapter = Step03()
    prepared = check_upstream(inputs, adapter)
    rng = np.random.default_rng(settings['seed'])
    rows, audit, boot_qc = [], [], {}
    horizons = settings['horizons_years']
    for marker in settings['risk_markers']:
        spec = inputs.features[marker]
        assay = spec['assay']
        reason = ''
        data, supported, fit_routes, ready = pd.DataFrame(), set(), [], set()
        delta = mean = sd = np.nan
        references, projections = {}, {}
        if assay not in prepared:
            reason = 'not_an_eligible_step03_candidate'
        elif spec['status'] != 'available':
            reason = spec['reason']
        else:
            data, stacked, support, mean, sd, reason = prepared[assay]
            safe = data[f'from_feature__{marker}']
            if not reason and not np.allclose(safe, data.biomarker_value, equal_nan=False, atol=1e-8, rtol=0):
                reason = 'step03_origin_has_unverified_or_postanchor_measurements'
            supported = _supported_routes(support, settings)
            fit_routes = list(support.index[flag(support.supported_for_model)])
            published = inputs.tables['models'].loc[
                inputs.tables['models'].biomarker.eq(assay) &
                inputs.tables['models'].model_version.eq(VERSION)].set_index('transition_id')
            interpretable = set(published.index[published.model_status.eq('success') &
                                                published.interpretability_status.eq('interpretable')])
            supported &= interpretable
            ready = {scheme for scheme, required in ESTIMANDS.items() if set(required).issubset(supported)}
            if ready and not reason:
                try:
                    params, delta = refit_equivalence(adapter, stacked, fit_routes, published, settings)
                    for pop in ('Pop2', 'Pop3'):
                        at_origin = data.loc[data.from_pop.eq(pop)].sort_values('from_clinical_anchor_date')
                        first = at_origin.drop_duplicates('patient_id')
                        if len(first) < settings['minimum_model_patients']:
                            continue
                        pct = first.biomarker_value.quantile([.25, .75]).to_numpy()
                        age = first.age.median()
                        end = data.groupby('patient_id').to_clinical_anchor_date.max().reindex(first.patient_id)
                        followup = (end.to_numpy() - first.from_clinical_anchor_date.to_numpy()) / np.timedelta64(1, 'D')
                        schemes = [scheme for scheme in ESTIMANDS if scheme in ready]
                        references[pop] = (pct, age, schemes, followup / DAYS_PER_YEAR)
                        projections[pop] = _projection(params, pct, age, mean, sd, horizons, schemes)
                except FIT_ERRORS as exc:
                    if str(exc) == 'step03_refit_coefficient_mismatch':
                        raise ValueError(f'Canonical Pharma 03/01 refit mismatch: {assay}') from exc
                    reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                    references, projections = {}, {}
        boot_counts, failures = ({}, {})
        if references:
            boot_counts, failures = bootstrap(adapter, data, assay, fit_routes, references, settings, rng)
        boot_qc[marker] = {'attempted_replicates': settings['bootstrap_replicates'] if references else 0,
                           'successful_by_estimand_and_pop': {scheme: {p: len(v) for p, v in counts.items()}
                                                             for scheme, counts in boot_counts.items()},
                           'failure_counts_by_estimand': failures,
                           'failure_fraction_by_estimand_and_pop': {
                               scheme: {pop: 1 - len(values)/settings['bootstrap_replicates']
                                        for pop, values in counts.items() if scheme in references[pop][2]}
                               for scheme, counts in boot_counts.items()},
                           'reason': reason,
                           'reference_quantiles_and_age': 'fixed', 'minimum_success_fraction': settings['minimum_bootstrap_success']}
        for route in ROUTES:
            counts = prepared[assay][2].loc[route] if assay in prepared else None
            audit.append(dict(marker=marker, assay=assay, unit=spec['unit'], scheme='route',
                              model_version=VERSION, route=route,
                              n_intervals=counts.n_intervals_at_risk if counts is not None else np.nan,
                              n_patients=counts.n_patients_at_risk if counts is not None else np.nan,
                              n_events=counts.n_events if counts is not None else np.nan,
                              n_event_patients=counts.n_patients_with_event if counts is not None else np.nan,
                              biomarker_mean=mean, biomarker_sd=sd, max_abs_log_coefficient_delta=delta,
                              status='estimated' if route in supported and not reason and ready else 'not_estimable',
                              reason=reason or ('' if route in supported and ready else 'route_or_projection_gate_failed')))
        for scheme in ESTIMANDS:
            for pop in ('Pop2', 'Pop3'):
                ref = references.get(pop)
                point = projections.get(pop, {}).get(scheme)
                boot = boot_counts.get(scheme, {}).get(pop, [])
                enough_draws = len(boot) >= settings['minimum_bootstrap_success'] * settings['bootstrap_replicates']
                draw_array = np.asarray(boot)
                probability_ci = (np.quantile(draw_array, [.025, .975], axis=0) if enough_draws else
                                  np.full((2, 2, len(horizons)), np.nan))
                ci = (np.quantile(100 * (draw_array[:, 1] - draw_array[:, 0]), [.025, .975], axis=0)
                      if enough_draws else np.full((2, len(horizons)), np.nan))
                for i, year in enumerate(horizons):
                    n_followup = int(np.sum(ref[3] >= year)) if ref is not None else 0
                    estimated = point is not None and n_followup >= settings['minimum_at_risk_warning']
                    low, high = point[:, i, POPS.index(pop)] if estimated else (np.nan, np.nan)
                    gate = (reason or ('insufficient_routes_for_estimand' if scheme not in ready else
                                       'insufficient_reference_patients' if ref is None else
                                       'horizon_exceeds_reference_followup_support'))
                    rows.append(dict(analysis_unit='model_projection', cohort='step03_age_complete_case',
                                     index_definition='first_eligible_origin_per_patient_and_pop',
                                     marker=marker, assay=assay, unit=spec['unit'], from_pop=pop, to_pop='Pop1',
                                     scheme=scheme, years=year, n_patients=data.patient_id.nunique() if len(data) else 0,
                                     n_reference_patients=len(ref[3]) if ref is not None else 0,
                                     n_at_risk_reference=n_followup, p25=ref[0][0] if ref is not None else np.nan,
                                     p75=ref[0][1] if ref is not None else np.nan,
                                     reference_age=ref[1] if ref is not None else np.nan, probability_p25=low,
                                     probability_p75=high, estimate=(high - low) * 100,
                                     probability_p25_ci95_low=probability_ci[0, 0, i] if estimated else np.nan,
                                     probability_p25_ci95_high=probability_ci[1, 0, i] if estimated else np.nan,
                                     probability_p75_ci95_low=probability_ci[0, 1, i] if estimated else np.nan,
                                     probability_p75_ci95_high=probability_ci[1, 1, i] if estimated else np.nan,
                                     ci95_low=ci[0, i] if estimated else np.nan,
                                     ci95_high=ci[1, i] if estimated else np.nan, n_bootstrap_successful=len(boot),
                                     extrapolation=point is not None and not estimated,
                                     status='estimated' if estimated else 'not_estimable', reason='' if estimated else gate))
    return pd.DataFrame(rows, columns=RISK_FIELDS), pd.DataFrame(audit, columns=SUPPORT_FIELDS), boot_qc

"""Conservative four-group repeated models with auditable MixedLM -> GEE fallback."""
from itertools import combinations
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog
from scipy.special import expit
import statsmodels.api as sm

from src.block_A.observed_patterns_v2_figures import LABELS
from src.block_A.secondary_comparison_contract import inference_frame

PATTERNS = tuple(LABELS)
TERMS = ('intercept', 'group_1', 'group_2', 'group_3', 'time', 'interaction_1', 'interaction_2', 'interaction_3')


def design(data):
    t = data.time.to_numpy(float)
    group = data.observed_pattern
    dummy = np.column_stack([group.eq(p).to_numpy(float) for p in PATTERNS[1:]])
    return np.column_stack([np.ones(len(data)), dummy, t, dummy*t[:, None]])


def design_row(pattern, time):
    d = np.array([float(pattern == p) for p in PATTERNS[1:]])
    return np.r_[1., d, time, d*time]


def mixed_fit(y, matrix, ids):
    return sm.MixedLM(y, matrix, groups=ids).fit(reml=False, method='lbfgs', disp=False)


def gee_fit(y, matrix, ids, binary=False):
    return sm.GEE(y, matrix, groups=ids, family=sm.families.Binomial() if binary else sm.families.Gaussian(),
                  cov_struct=sm.cov_struct.Independence()).fit(maxiter=200, cov_type='robust')


def _covariance(result, mixed):
    params = np.asarray(result.fe_params if mixed else result.params, dtype=float)
    cov = np.asarray(result.cov_params(), dtype=float)[:8, :8]
    return params, cov


def _fit_qc(result, y, matrix, patterns, cfg, mixed=False, binary=False):
    beta, cov = _covariance(result, mixed)
    reasons = []
    if not getattr(result, 'converged', False):
        reasons.append('nonconvergence')
    if not (np.isfinite(beta).all() and np.isfinite(cov).all()):
        reasons.append('nonfinite_coefficient_or_covariance')
        return reasons, {}
    eig = np.linalg.eigvalsh((cov+cov.T)/2)
    if eig.min() <= 0 or np.linalg.cond(cov) > cfg.maximum_design_condition:
        reasons.append('singular_or_indefinite_fixed_effect_covariance')
    if mixed:
        random_variance = float(np.asarray(result.cov_re).ravel()[0])
        if random_variance <= cfg.random_intercept_boundary*max(float(result.scale), 1.):
            reasons.append('random_intercept_at_boundary')
    else:
        random_variance = np.nan
    fitted = expit(matrix@beta) if binary else matrix@beta
    residual = y-fitted
    skew = float(stats.skew(residual)) if np.std(residual) > 0 else 0.
    group_var = [np.var(residual[patterns == p], ddof=1) for p in PATTERNS]
    ratio = max(group_var)/min(group_var) if min(group_var) > 0 else np.inf
    if binary:
        if np.max(np.abs(beta)) > cfg.maximum_logistic_coefficient:
            reasons.append('unstable_logistic_coefficient')
        if np.any((fitted < 1e-6) | (fitted > 1-1e-6)):
            reasons.append('near_deterministic_probabilities')
    else:
        if abs(skew) > cfg.maximum_residual_skewness:
            reasons.append('residual_asymmetry_exceeds_prespecified_limit')
        if ratio > cfg.maximum_residual_variance_ratio:
            reasons.append('residual_variance_ratio_exceeds_prespecified_limit')
    return reasons, dict(residual_skewness=skew, residual_variance_ratio=ratio,
                         random_intercept_variance=random_variance,
                         fixed_covariance_min_eigenvalue=float(eig.min()))


def separation(y, matrix):
    """Detect complete or quasi separation, including time-dependent separation."""
    signed = matrix*(2*y-1)[:, None]
    # Maximize total nonnegative margin within bounded coefficient space.
    fit = linprog(-signed.sum(axis=0), A_ub=-signed, b_ub=np.zeros(len(y)),
                  bounds=[(-1., 1.)]*matrix.shape[1], method='highs')
    return bool(fit.success and -fit.fun > 1e-7)


def _effect(vector, beta, cov):
    estimate = float(vector@beta)
    variance = float(vector@cov@vector)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('invalid contrast variance')
    se = np.sqrt(variance)
    return dict(estimate=estimate, SE=se, ci95_low=estimate-1.959963984540054*se,
                ci95_high=estimate+1.959963984540054*se, p=float(2*stats.norm.sf(abs(estimate/se))))


def longitudinal(frame, spec, gate, cfg, axis='clinical_baseline', scenario='primary'):
    output = {name: [] for name in ('attempts', 'qc', 'global', 'coefficients', 'contrasts', 'predictions', 'support')}
    base = dict(scenario=scenario, model_family='', adjustment_set='none',
                analysis_role='primary' if scenario == 'primary' else 'sensitivity',
                **{k: v for k, v in gate.items() if k not in ('status', 'reason')})
    def blocked(reason, status='NOT_ESTIMABLE'):
        row = dict(**base, status=status, reason=reason, p=np.nan,
                   fit_warnings='', fallback_reason='', effect_scale=spec.unit)
        output['global'] = [row]
        return {k: pd.DataFrame(v) for k, v in output.items()}
    if gate['status'] != 'ELIGIBLE':
        return blocked(gate['reason'])
    data = inference_frame(frame, axis).sort_values(['patient_id', 'time'])
    if data.threshold_abs.nunique() != 1 or data.groupby('patient_id').observed_pattern.nunique().gt(1).any():
        # Contract failures must not be swallowed as local model failures.
        from src.block_A.secondary_trajectory_analysis import ContractError
        raise ContractError('mixed thresholds or mutable patient group in model')
    y, matrix = data.value.to_numpy(float), design(data)
    ids = data.patient_id.to_numpy(str)
    pattern_values = data.observed_pattern.to_numpy(str)
    if (not np.isfinite(y).all() or not np.isfinite(matrix).all() or np.ptp(y) <= 1e-12):
        return blocked('constant_or_nonfinite_outcome')
    if np.linalg.matrix_rank(matrix) != 8 or np.linalg.cond(matrix) > cfg.maximum_design_condition:
        return blocked('singular_or_ill_conditioned_design')
    binary = spec.type == 'binary'
    if binary:
        if not set(np.unique(y)).issubset({0., 1.}):
            return blocked('invalid_binary_values')
        for p in PATTERNS:
            g = data.loc[data.observed_pattern.eq(p)]
            # Event support is counted by patients, not visits.
            event_n = g.loc[g.value.eq(1), 'patient_id'].nunique()
            nonevent_n = g.loc[g.value.eq(0), 'patient_id'].nunique()
            output['support'].append(dict(scenario=scenario, observed_pattern=p,
                n_event_patients=event_n, n_nonevent_patients=nonevent_n))
            if event_n < cfg.minimum_events_per_group or nonevent_n < cfg.minimum_nonevents_per_group:
                return blocked('insufficient_binary_event_or_nonevent_patients')
        if separation(y, matrix):
            return blocked('complete_or_quasi_logistic_separation')
    fitted = None
    failures = []
    for family in (['GEE_logistic'] if binary else ['MixedLM_ML_random_intercept', 'GEE_Gaussian_robust']):
        is_mixed = family.startswith('MixedLM')
        captured = []
        diagnostics = {}
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                result = mixed_fit(y, matrix, ids) if is_mixed else gee_fit(y, matrix, ids, binary)
                reasons, diagnostics = _fit_qc(result, y, matrix, pattern_values, cfg, is_mixed, binary)
            # Warning classes are safe metadata; warning text may contain identifiers.
            warning_classes = sorted({w.category.__name__ for w in captured})
            if is_mixed and any(w.category.__name__ in ('ConvergenceWarning', 'RuntimeWarning') for w in captured):
                reasons.append('mixed_fit_warning')
            if not is_mixed and any(w.category.__name__ in ('ConvergenceWarning', 'IterationLimitWarning', 'RuntimeWarning', 'PerfectSeparationWarning') for w in captured):
                reasons.append('gee_fit_warning')
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            reasons = ['fit_exception_'+type(error).__name__]
            warning_classes = sorted({w.category.__name__ for w in captured})
        output['attempts'].append(dict(**{**base, 'model_family': family},
            status='MODEL_FAILED' if reasons else 'ESTIMATED', reason='; '.join(reasons),
            fit_warnings='; '.join(warning_classes), fallback_reason='; '.join(failures)))
        output['qc'].append(dict(scenario=scenario, model_family=family,
            status='FAIL' if reasons else 'PASS', reason='; '.join(reasons),
            design_rank=int(np.linalg.matrix_rank(matrix)), design_condition=float(np.linalg.cond(matrix)), **diagnostics))
        if not reasons:
            fitted = result
            base.update(model_family=family, fit_warnings='; '.join(warning_classes), fallback_reason='; '.join(failures))
            break
        failures.append(family+': '+ '; '.join(reasons))
    if fitted is None:
        return blocked('; '.join(failures))
    beta, covariance = _covariance(fitted, base['model_family'].startswith('MixedLM'))
    interaction_cov = covariance[5:8, 5:8]
    if np.linalg.matrix_rank(interaction_cov) != 3:
        return blocked('singular_global_interaction_covariance')
    wald = float(beta[5:8]@np.linalg.solve(interaction_cov, beta[5:8]))
    if not np.isfinite(wald) or wald < 0:
        return blocked('invalid_global_wald')
    effect_scale = 'log odds/year' if binary else spec.unit+'/year'
    output['global'].append(dict(**base, status='ESTIMATED', reason='', statistic=wald,
        df=3, p=float(stats.chi2.sf(wald, 3)), test='joint group x continuous time Wald', effect_scale=effect_scale,
        reference_group=PATTERNS[0]))
    for index, term in enumerate(TERMS):
        vector = np.eye(8)[index]
        output['coefficients'].append(dict(**base, term=term, status='ESTIMATED',
            theoretical_intercept_not_observed_baseline=True, **_effect(vector, beta, covariance)))
    baseline_values = data.loc[data.variable_measurement_date.eq(data.clinical_baseline_date), 'value']
    standard_sd = baseline_values.std() if len(baseline_values) > 1 else np.nan
    for a, b in combinations(PATTERNS, 2):
        slope_vector = (design_row(b, 1)-design_row(b, 0))-(design_row(a, 1)-design_row(a, 0))
        effect = _effect(slope_vector, beta, covariance)
        output['contrasts'].append(dict(**base, contrast=b+' - '+a, contrast_type='slope', time=np.nan,
            multiplicity_role='prespecified' if a == PATTERNS[0] else 'exploratory',
            status='ESTIMATED', effect_scale=effect_scale, **effect,
            standardized_effect=effect['estimate']/standard_sd if not binary and standard_sd > 0 else np.nan,
            standardization='pooled actually measured official-baseline SD; slope units SD/year'))
    lo, hi = gate['common_time_min'], gate['common_time_max']
    times = sorted(set(np.linspace(lo, hi, 41).tolist()+[t for t in cfg.report_times if lo <= t <= hi]))
    for t in times:
        observed_n = {p: int(data.loc[data.observed_pattern.eq(p) &
            data.time.sub(t).abs().le(cfg.prediction_window_years/2), 'patient_id'].nunique()) for p in PATTERNS}
        supported = min(observed_n.values()) >= cfg.minimum_prediction_patients
        for p in PATTERNS:
            row = dict(**base, observed_pattern=p, time=t, n_observed_local=observed_n[p],
                n_observed_by_group=json.dumps(observed_n, sort_keys=True),
                local_window_width_years=cfg.prediction_window_years, status='ESTIMATED' if supported else 'NOT_ESTIMABLE',
                reason='' if supported else 'insufficient_local_four_group_support',
                effect_scale='probability' if binary else spec.unit,
                estimate=np.nan, SE=np.nan, ci95_low=np.nan, ci95_high=np.nan)
            if supported:
                effect = _effect(design_row(p, t), beta, covariance)
                effect.pop('p')
                if binary:
                    eta, se = effect['estimate'], effect['SE']
                    effect.update(estimate=expit(eta), SE=expit(eta)*(1-expit(eta))*se,
                                  ci95_low=expit(effect['ci95_low']), ci95_high=expit(effect['ci95_high']))
                row.update(effect)
            output['predictions'].append(row)
        if supported and any(abs(t-r) < 1e-9 for r in cfg.report_times):
            for a, b in combinations(PATTERNS, 2):
                xa, xb = design_row(a, t), design_row(b, t)
                effect = _effect(xb-xa, beta, covariance)
                if binary:
                    pa, pb = expit(xa@beta), expit(xb@beta)
                    gradient = pb*(1-pb)*xb-pa*(1-pa)*xa
                    se = np.sqrt(gradient@covariance@gradient)
                    diff = pb-pa
                    effect = dict(estimate=diff, SE=se, ci95_low=max(-1, diff-1.96*se),
                        ci95_high=min(1, diff+1.96*se), p=float(2*stats.norm.sf(abs(diff/se))))
                output['contrasts'].append(dict(**base, contrast=b+' - '+a, contrast_type='mean_at_supported_time',
                    time=t, status='ESTIMATED', multiplicity_role='secondary',
                    effect_scale='probability difference' if binary else spec.unit, **effect))
    return {k: pd.DataFrame(v) for k, v in output.items()}

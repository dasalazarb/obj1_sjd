"""Coherent ML random-intercept or robust GEE model families, never mixed LRTs."""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy import stats
from src.block_A.secondary_comparison_models import mixed_fit, gee_fit, separation, PATTERNS


@dataclass
class Fit:
    model: str
    family: str
    beta: np.ndarray
    covariance: np.ndarray
    loglik: float = np.nan
    aic: float = np.nan
    bic: float = np.nan


def fit_qc(result, y, matrix, patterns, cfg, mixed, binary):
    p = matrix.shape[1]
    beta = np.asarray(result.fe_params if mixed else result.params, float)
    cov = np.asarray(result.cov_params(), float)[:p, :p]
    cov = (cov+cov.T)/2
    reasons, qc = [], {}
    if not getattr(result, 'converged', False):
        reasons.append('nonconvergence')
    if not np.isfinite(beta).all() or not np.isfinite(cov).all():
        return beta, cov, ['nonfinite_coefficients_or_covariance'], qc
    eigen = np.linalg.eigvalsh(cov)
    qc['covariance_min_eigenvalue'] = float(eigen.min())
    if eigen.min() <= 0 or np.linalg.cond(cov) > cfg.maximum_design_condition:
        reasons.append('singular_or_indefinite_covariance')
    if mixed:
        full_covariance = np.asarray(result.cov_params(), float)
        if (not np.isfinite(full_covariance).all() or
                np.linalg.eigvalsh((full_covariance+full_covariance.T)/2).min() <= 0):
            reasons.append('nonregular_full_information_matrix')
        variance = float(np.asarray(result.cov_re).ravel()[0])
        qc['random_intercept_variance'] = variance
        if variance <= cfg.random_intercept_boundary*max(float(result.scale), 1.):
            reasons.append('random_intercept_at_boundary')
        if not np.isfinite(result.llf):
            reasons.append('nonfinite_ML_likelihood')
    if binary:
        from scipy.special import expit
        fitted = expit(matrix@beta)
        if np.max(np.abs(beta)) > cfg.maximum_logistic_coefficient or np.any((fitted < 1e-6) | (fitted > 1-1e-6)):
            reasons.append('unstable_logistic_fit')
    else:
        # Conditional residuals for MixedLM; marginal residuals for GEE.
        residual = y-np.asarray(result.fittedvalues if mixed else matrix@beta)
        skew = float(stats.skew(residual)) if np.std(residual) > 0 else 0.
        variances = [np.var(residual[patterns == g], ddof=1) for g in PATTERNS]
        ratio = max(variances)/min(variances) if min(variances) > 0 else np.inf
        qc.update(residual_skewness=skew, residual_variance_ratio=float(ratio))
        if abs(skew) > cfg.maximum_residual_skewness:
            reasons.append('residual_asymmetry_exceeds_limit')
        if ratio > cfg.maximum_residual_variance_ratio:
            reasons.append('residual_variance_ratio_exceeds_limit')
    return beta, cov, reasons, qc


def fit_family(evaluation, spec, cfg):
    data, matrices, gate = evaluation['data'], evaluation['matrices'], evaluation['gate']
    wanted = [m for m in 'ABC' if gate['eligible_'+m]]
    attempts, selected = [], {}
    if not wanted:
        return selected, attempts, []
    y, ids = data.value.to_numpy(float), data.patient_id.to_numpy(str)
    patterns = data.observed_pattern.to_numpy(str)
    binary = spec.type == 'binary'
    families = ['GEE_binomial_robust'] if binary else ['MixedLM_ML_random_intercept', 'GEE_gaussian_robust']
    for family in families:
        fits = {}
        mixed = family.startswith('MixedLM')
        for model in wanted:
            matrix = matrices[model]
            reasons, qc, captured = [], {}, []
            if binary and separation(y, matrix):
                reasons = ['complete_or_quasi_logistic_separation']
            else:
                try:
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter('always')
                        result = mixed_fit(y, matrix, ids) if mixed else gee_fit(y, matrix, ids, binary)
                        beta, cov, reasons, qc = fit_qc(result, y, matrix, patterns, cfg, mixed, binary)
                    if any(w.category.__name__ in ('ConvergenceWarning', 'RuntimeWarning', 'IterationLimitWarning',
                                                   'PerfectSeparationWarning') for w in captured):
                        reasons.append('fit_warning_requires_rejection')
                    if mixed and fits and float(result.llf) < list(fits.values())[-1].loglik-1e-6:
                        reasons.append('nonmonotone_nested_ML_likelihood')
                    if not reasons:
                        fits[model] = Fit(model, family, beta, cov,
                            float(result.llf) if mixed else np.nan,
                            float(result.aic) if mixed else np.nan, float(result.bic) if mixed else np.nan)
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                    reasons = ['fit_exception_'+type(error).__name__]
            attempts.append(dict(model=model, model_family=family, status='MODEL_FAILED' if reasons else 'ESTIMATED',
                reason='; '.join(reasons), warnings='; '.join(sorted({w.category.__name__ for w in captured})),
                n_observations=len(data), n_patients=data.patient_id.nunique(),
                design_rank=int(np.linalg.matrix_rank(matrix)), design_condition=float(np.linalg.cond(matrix)),
                n_columns=matrix.shape[1], estimation='ML' if mixed else 'GEE_robust', **qc))
            if reasons:
                break  # No jumping over a failed A or B.
        if len(fits) == len(wanted) or not mixed:
            selected = fits
            break
        # Refit the WHOLE family with GEE; discard partial ML fits.
    return selected, attempts, tests(selected)


def wald(fit, indices):
    sub = fit.covariance[np.ix_(indices, indices)]
    beta = fit.beta[indices]
    if np.linalg.matrix_rank(sub) != len(indices) or np.linalg.eigvalsh(sub).min() <= 0:
        return dict(status='NOT_ESTIMABLE', reason='singular_test_covariance', p=np.nan)
    statistic = float(beta@np.linalg.solve(sub, beta))
    return dict(status='ESTIMATED', reason='', statistic=statistic, df=len(indices),
                p=float(stats.chi2.sf(statistic, len(indices))), method='robust_Wald' if fit.family.startswith('GEE') else 'ML_Wald')


def tests(fits):
    rows = []
    if 'A' in fits:
        rows.append(dict(test='linear_group_time', model='A', **wald(fits['A'], [5, 6, 7])))
    for lower, upper, name, indices in [('A', 'B', 'common_nonlinearity', [8, 9]),
                                        ('B', 'C', 'group_specific_curvature', list(range(10, 16)))]:
        if lower not in fits or upper not in fits:
            continue
        a, b = fits[lower], fits[upper]
        if a.family != b.family:
            raise ValueError('incoherent model families')
        if b.family.startswith('MixedLM'):
            statistic = 2*(b.loglik-a.loglik)
            result = (dict(status='ESTIMATED', reason='', statistic=max(0., statistic), df=len(indices),
                           p=float(stats.chi2.sf(max(0., statistic), len(indices))), method='ML_LRT')
                      if statistic >= -1e-6 else dict(status='NOT_ESTIMABLE', reason='nonmonotone_ML_likelihood', p=np.nan))
        else:
            result = wald(b, indices)
        rows.append(dict(test=name, model=upper, **result))
    if 'C' in fits:
        rows.append(dict(test='global_temporal_heterogeneity', model='C', **wald(fits['C'], [5, 6, 7, *range(10, 16)])))
    for row in rows:
        row['model_family'] = fits[row['model']].family
    return rows

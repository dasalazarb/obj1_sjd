"""Frozen V1 bootstrap algorithm for synthetic policy regression only.

Copied from 6217c3c; never used by production validation.
"""
import json
import math
from collections import Counter
import numpy as np
from src.block_A.temporal_spline_validation import (
    target_family, target_values, original_times, supported_times, family_name,
    simultaneous_critical, fit_fixed_family,
)

def bootstrap_summary(evaluation, fit, spec, cfg, seed=None):
    """Fixed-family stratified patient bootstrap, joint over model-specific family."""
    from src.block_A.temporal_spline_sensitivity import resample_patients
    seed = cfg.seed if seed is None else seed
    requested = cfg.bootstrap_replicates
    base = dict(model=fit.model, n_requested=requested, n_attempted=0, n_successful=0, n_failed=0,
        seed=seed, estimand_family=family_name(fit.model), failure_reasons='{}',
        required_successes=max(100, math.ceil(.8*requested)), calibration_valid=False)
    if spec.type == 'binary':
        return dict(base, status='NOT_ESTIMABLE_METHOD', reason='Binary response bootstrap calibration is not validated')
    rows, gradients, estimates = target_family(evaluation, fit)
    if not len(rows):
        return dict(base, status='NOT_ESTIMABLE', reason='No supported grid')
    se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', gradients, fit.covariance, gradients), 0))
    valid = se > 1e-12
    if not valid.any():
        return dict(base, status='NOT_ESTIMABLE', reason='No nondegenerate estimands')
    analytic = simultaneous_critical(gradients, fit.covariance, cfg.simulations, cfg.seed)
    rng, maxima, failures = np.random.default_rng(seed), [], Counter()
    for _ in range(requested):
        replicate = resample_patients(evaluation['data'], rng)
        refit, reason = fit_fixed_family(replicate, evaluation['basis'], fit.model, fit.family, spec, cfg)
        if refit is None:
            failures[reason] += 1
            continue
        # Calibration requires the entire ORIGINAL family, never a smaller
        # data-dependent set of supported contrasts in a successful replicate.
        if supported_times(replicate, original_times(evaluation), cfg) != original_times(evaluation):
            failures['ORIGINAL_SUPPORT_LOST'] += 1
            continue
        values = target_values(rows, evaluation, refit)
        maxima.append(float(np.max(np.abs(values[valid]-estimates[valid])/se[valid])))
    enough = len(maxima) >= base['required_successes']
    critical = float(np.quantile(maxima, .95)) if enough else np.nan
    analytic_sign = np.where(estimates-analytic*se > 0, 1, np.where(estimates+analytic*se < 0, -1, 0))
    bootstrap_sign = np.where(estimates-critical*se > 0, 1, np.where(estimates+critical*se < 0, -1, 0))
    return dict(base, status='CALIBRATED' if enough else 'INCONCLUSIVE', reason='' if enough else 'Insufficient comparable replicates',
        n_attempted=requested, n_successful=len(maxima), n_failed=requested-len(maxima),
        failure_reasons=json.dumps(failures, sort_keys=True), success_fraction=len(maxima)/requested if requested else 0.,
        calibration_valid=enough, bootstrap_sup_t_critical=critical, analytic_sup_t_critical=analytic,
        analytic_seed=cfg.seed,
        critical_difference=critical-analytic, critical_ratio=critical/analytic,
        global_sup_t_p50=float(np.quantile(maxima, .5)) if enough else np.nan,
        global_sup_t_p90=float(np.quantile(maxima, .9)) if enough else np.nan,
        band_agreement=float(np.mean(analytic_sign[valid] == bootstrap_sign[valid])) if enough else np.nan,
        inference='Exploratory stability within the same cohort; not independent validation')

"""Prespecified N3 comparisons conditional on frozen discovery memberships.

No Mapper training, imputation, outcome selection, external prediction, or changes
to historical incremental estimators. A/B/C/D share one complete-case population.
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .s03_n3_candidate_audit import (
    INTERNAL, KEYS, PROFILE_COLUMNS, adjustment_data, fingerprint, measurement_policy,
    scoped_records, values_for, verified_reference,
)
from .s03_release_reader import hard_members
from .s03_profiles import observed_values
from .s03_statistics import bh_qvalues

SCOPE = 'in_sample_exploratory_conditional_on_frozen_memberships'
METRICS = ['r2', 'adjusted_r2', 'rmse', 'aic', 'auc', 'brier', 'calibration_in_large',
           'p_value', 'q_value', 'effect', 'ci_low', 'ci_high']
BASE_COLUMNS = KEYS + ['model', 'model_family', 'hypothesis_family', 'family_size',
    'status', 'reason', 'n_patients', 'n_parameters', 'residual_df', 'common_population_verified',
    'n_excluded_missing', 'adjustments', 'pop_comparator_status', 'source_comparator_status',
    'independence_gate', 'topology_gate', 'measurement_gate', 'covariate_gate', 'support_gate',
    'design_gate', 'effects_json', 'covariance', 'inference_scope', 'uncertainty_scope', *METRICS, *INTERNAL]
COMPARISON_COLUMNS = KEYS + ['contrast', 'hypothesis_family', 'family_size', 'status', 'reason',
    'n_patients', 'common_population_verified', 'delta_r2', 'delta_adjusted_r2',
    'delta_adjusted_r2_ci_low', 'delta_adjusted_r2_ci_high', 'bootstrap_valid_replicates',
    'delta_aic', 'delta_brier', 'p_value', 'q_value', 'inference_scope', 'uncertainty_scope', *INTERNAL]


def _outcomes(config, identifiers):
    result = []
    for item in config.get('n3_validation', {}).get('prespecified_outcomes', []):
        record = {'source_column': item} if isinstance(item, str) else dict(item)
        source = record.get('source_column')
        if not isinstance(source, str) or not source:
            raise ValueError('INVALID_PRESPECIFIED_N3_OUTCOME')
        reps = [record['representation_id']] if record.get('representation_id') else identifiers
        for identifier in reps:
            if identifier not in identifiers:
                raise ValueError('PRESPECIFIED_N3_REPRESENTATION_NOT_IN_RELEASE')
            result.append({**record, 'representation_id': identifier})
    keys = [(r['representation_id'], r['source_column']) for r in result]
    if len(keys) != len(set(keys)):
        raise ValueError('DUPLICATE_PRESPECIFIED_N3_OUTCOME')
    return sorted(result, key=lambda r: (r['representation_id'], r['source_column']))


def _single(records):
    if len(records) > 1:
        raise ValueError('AMBIGUOUS_N3_COMPARATOR_REVIEW')
    return records[0] if records else {}


def _pop_status(source, config, identifier, catalog):
    # Every documented component/derivative of the ESSDAI/ESSPRI construct is ineligible.
    related = {'essdai__total', 'esspri__total'}
    relations = config.get('characterization', {}).get('relationships', [])
    changed = True
    while changed:
        changed = False
        for r in relations:
            if r.get('relationship_type') in {'derived_from', 'composite_component', 'same_instrument', 'same_raw_source', 'encoded_variant'}:
                a, b = r.get('source_column'), r.get('target_source')
                # Do not traverse Pop to bridge unrelated inputs.
                if config.get('pop_column') in {a, b}: continue
                if a in related or b in related:
                    before = len(related); related.update([a, b]); changed |= len(related) != before
    if source in related or source.startswith(('essdai__', 'esspri__', 'eg_')) or source == 'ovl__status':
        return 'POP_BASELINE_COMPARATOR_INELIGIBLE', False
    review = _single(scoped_records(config, 'pop_comparator_reviews', identifier, source))
    if not review:
        return 'COVARIATES_ONLY_POP_NOT_REQUESTED', False
    valid = (review.get('eligible') is True and review.get('semantics_review_status') == 'VERIFIED'
             and review.get('outcome_dependency_review_status') == 'APPROVED'
             and verified_reference(review, catalog))
    return ('VERIFIED_ELIGIBLE' if valid else 'POP_COMPARATOR_REVIEW_PENDING'), valid


def _source_basis(identifier, roles, config, catalog):
    review = _single(scoped_records(config, 'source_comparators', identifier))
    if not review:
        return [], 'N1_COMPARATOR_NOT_PRESPECIFIED', False
    columns = review.get('source_columns', [])
    defining = set(roles.loc[roles.representation_id.eq(identifier) & roles.tier.eq('N1_DEFINING'), 'source_column'])
    approved = (review.get('design_review_status') == 'APPROVED' and verified_reference(review, catalog)
                and isinstance(columns, list) and bool(columns) and len(columns) == len(set(columns))
                and set(columns) <= defining and review.get('missingness_policy') == 'common_complete_case')
    return (columns if approved else []), ('VERIFIED_REDUCED_N1_BASIS' if approved else 'N1_COMPARATOR_REVIEW_PENDING'), review.get('include_model_d') is True


def _design(data):
    """Deterministic dummy names never export patient-provided category strings."""
    result = pd.DataFrame({'intercept': np.ones(len(data))}, index=data.index)
    for name in data:
        values = data[name]
        if pd.api.types.is_numeric_dtype(values.dtype):
            result[name] = values.astype(float)
        else:
            categories = sorted(values.astype(str).unique())
            for i, category in enumerate(categories[1:], 1):
                result[f'{name}_level_{i}'] = values.astype(str).eq(category).astype(float)
    return result.astype(float)


def _blank(identifier, source, model, reason):
    return dict(representation_id=identifier, source_column=source, model=model,
        model_family='', hypothesis_family=f'{identifier}/n3_incremental', family_size=0,
        status='NOT_ESTIMABLE', reason=reason, n_patients=0, n_parameters=np.nan,
        residual_df=np.nan, common_population_verified=False, n_excluded_missing=0,
        adjustments='prespecified_confounders_and_reviewed_baseline_era',
        pop_comparator_status='', source_comparator_status='', independence_gate='BLOCKED',
        topology_gate='NOT_EVALUATED', measurement_gate='NOT_EVALUATED',
        covariate_gate='NOT_EVALUATED', support_gate='NOT_EVALUATED', design_gate='NOT_EVALUATED',
        effects_json='', covariance='', inference_scope=SCOPE,
        uncertainty_scope='conditional_effect_intervals_exclude_Mapper_selection_uncertainty',
        **{k: np.nan for k in METRICS}, **INTERNAL)


def blocked_results(manifest, config, reason='NO_REVIEWED_INDEPENDENT_ANCHORS'):
    outcomes = _outcomes(config, manifest['eligible_representation_ids'])
    if not outcomes:
        outcomes = [dict(representation_id=r, source_column='') for r in manifest['eligible_representation_ids']]
    rows = [_blank(o['representation_id'], o['source_column'], m, reason)
            for o in outcomes for m in ['A', 'B', 'C', 'D']]
    models = pd.DataFrame(rows, columns=BASE_COLUMNS)
    return dict(descriptive_profiles=pd.DataFrame(columns=PROFILE_COLUMNS),
                incremental_eligibility=models.copy(), incremental_models=models,
                model_comparison=pd.DataFrame(columns=COMPARISON_COLUMNS))


def _descriptives(baseline, membership, inventory, config, catalog):
    rows = []
    for role in inventory.loc[inventory.review_state.eq('N3_REVIEWED')].to_dict('records'):
        identifier, source = role['representation_id'], role['source_column']
        frame = hard_members(membership, identifier).merge(baseline, on='patient_id', validate='one_to_one')
        policy = measurement_policy(config, identifier, source, catalog)
        for community, group in frame.groupby('community_id', sort=True):
            values, kind, states = values_for(group[source], role['type'], policy)
            valid = values.dropna()
            row = dict(representation_id=identifier, source_column=source, community_id=str(community),
                tier='N3_REVIEWED', unit=role['unit'], type=kind, n_group=len(group),
                n_observed=len(valid), n_missing=len(group)-len(valid),
                median=np.nan, q25=np.nan, q75=np.nan, n_positive=states['n_positive'],
                n_negative=states['n_negative'], proportion=np.nan,
                status=('DESCRIPTIVE_PROVISIONAL' if identifier == 'R2' or not group.structural_stability_status.eq('stable').all() else 'DESCRIPTIVE') if len(valid) else 'NOT_ESTIMABLE',
                reason='' if len(valid) else 'NO_OBSERVED_ENDPOINT_VALUES',
                uncertainty_scope='development_sample_description_excludes_Mapper_selection_uncertainty', **INTERNAL)
            if kind == 'numeric' and len(valid):
                row.update(median=float(valid.median()), q25=float(valid.quantile(.25)), q75=float(valid.quantile(.75)))
            elif kind == 'binary' and len(valid):
                row['proportion'] = float(valid.mean())
            # No category levels or free values are exported from restricted profiles.
            rows.append(row)
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def _fit(y, design, family):
    if family == 'gaussian_ols':
        return sm.OLS(y, design).fit(cov_type='HC3')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        fitted = sm.GLM(y, design, family=sm.families.Binomial()).fit(cov_type='HC3')
    if any('separation' in str(w.message).lower() for w in captured):
        raise ValueError('SEPARATION')
    if not fitted.converged:
        raise ValueError('MODEL_NONCONVERGENCE')
    if np.any((fitted.fittedvalues < 1e-10) | (fitted.fittedvalues > 1-1e-10)):
        raise ValueError('SEPARATION_OR_EXTREME_FITTED_PROBABILITIES')
    return fitted


def _model_metrics(fit, y, family):
    ci = fit.conf_int()
    if not (np.isfinite(fit.params).all() and np.isfinite(ci.to_numpy()).all()):
        raise ValueError('NONFINITE_MODEL_ESTIMATES')
    transform = np.exp if family == 'binomial_logit' else float
    effects = {str(name): dict(estimate=float(transform(fit.params[name])), ci_low=float(transform(ci.loc[name, 0])),
                              ci_high=float(transform(ci.loc[name, 1])),
                              effect_scale='odds_ratio' if family == 'binomial_logit' else 'source_units')
               for name in fit.params.index}
    if any(not np.isfinite(v[k]) for v in effects.values() for k in ['estimate','ci_low','ci_high']):
        raise ValueError('NONFINITE_MODEL_ESTIMATES')
    metrics = dict(n_parameters=len(fit.params), residual_df=float(fit.df_resid),
                   aic=float(fit.aic), effects_json=json.dumps(effects, sort_keys=True), covariance='HC3')
    if family == 'gaussian_ols':
        metrics.update(r2=float(fit.rsquared), adjusted_r2=float(fit.rsquared_adj),
                       rmse=float(np.sqrt(np.mean((y-fit.fittedvalues)**2))))
    else:
        metrics.update(auc=float(roc_auc_score(y, fit.fittedvalues)),
                       brier=float(np.mean((y-fit.fittedvalues)**2)),
                       calibration_in_large=float(np.mean(fit.fittedvalues)-np.mean(y)))
    return metrics


def _delta_interval(y, reduced, full, replicates, seed):
    """Paired patient bootstrap; frozen memberships/design coding are held fixed."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        index = rng.integers(0, len(y), len(y))
        a, b = reduced.iloc[index], full.iloc[index]
        if any(np.linalg.matrix_rank(d) != d.shape[1] for d in [a, b]): continue
        fit_a = sm.OLS(y.iloc[index].to_numpy(), a.to_numpy()).fit()
        fit_b = sm.OLS(y.iloc[index].to_numpy(), b.to_numpy()).fit()
        delta = fit_b.rsquared_adj-fit_a.rsquared_adj
        if np.isfinite(delta): values.append(delta)
    if len(values) < max(2, int(np.ceil(.8*replicates))):
        return np.nan, np.nan, len(values)
    lo, hi = np.quantile(values, [.025, .975])
    return float(lo), float(hi), len(values)


def n3_incremental_models(baseline, membership, roles, inventory, manifest, config, evidence_catalog=None):
    catalog = evidence_catalog or {}
    outcomes = _outcomes(config, manifest['eligible_representation_ids'])
    profiles = _descriptives(baseline, membership, inventory, config, catalog)
    if not outcomes:
        result = blocked_results(manifest, config, 'NO_PRESPECIFIED_OUTCOMES|NO_REVIEWED_INDEPENDENT_ANCHORS'
                                 if not inventory.review_state.eq('N3_REVIEWED').any() else 'NO_PRESPECIFIED_OUTCOMES')
        result['descriptive_profiles'] = profiles
        return result
    model_rows, comparisons = [], []
    family_sizes = {}
    for outcome in outcomes:
        identifier, source = outcome['representation_id'], outcome['source_column']
        columns, basis_status, include_d = _source_basis(identifier, roles, config, catalog)
        planned_basis = _single(scoped_records(config, 'source_comparators', identifier))
        family_sizes[identifier] = family_sizes.get(identifier, 0) + 1 + bool(planned_basis) + bool(planned_basis and include_d)
    for outcome in outcomes:
        identifier, source = outcome['representation_id'], outcome['source_column']
        selected = inventory.loc[inventory.representation_id.eq(identifier) & inventory.source_column.eq(source)]
        rows = {m: _blank(identifier, source, m, '') for m in ['A', 'B', 'C', 'D']}
        columns, basis_status, include_d = _source_basis(identifier, roles, config, catalog)
        pop_status, include_pop = _pop_status(source, config, identifier, catalog)
        if len(selected) == 1 and selected.iloc[0].get('pop_construct_dependency', False):
            pop_status, include_pop = 'POP_BASELINE_COMPARATOR_INELIGIBLE', False
        block = ''
        if len(selected) != 1: block = 'MISSING_SOURCE_OR_SOURCE_ROLE'
        elif selected.iloc[0].review_state != 'N3_REVIEWED':
            block = selected.iloc[0].blocking_reasons or 'PENDING_CLINICAL_REVIEW'
        for row in rows.values():
            row.update(family_size=family_sizes[identifier], pop_comparator_status=pop_status,
                       source_comparator_status=basis_status, independence_gate='PASS' if not block else 'BLOCKED')
        if block:
            for row in rows.values(): row['reason'] = block
            model_rows.extend(rows.values()); continue
        role = selected.iloc[0]
        frame = hard_members(membership, identifier).merge(baseline, on='patient_id', validate='one_to_one').sort_values('patient_id').reset_index(drop=True)
        policy = measurement_policy(config, identifier, source, catalog)
        y, kind, _ = values_for(frame[source], role['type'], policy)
        adjustment, missing, era_status = adjustment_data(frame, config, identifier, catalog)
        if include_pop:
            if config['pop_column'] not in frame:
                missing.append('pop_comparator')
                adjustment['pop_comparator'] = np.nan
            else:
                adjustment['pop_comparator'] = frame[config['pop_column']].where(frame[config['pop_column']].isin(['Pop1', 'Pop2', 'Pop3']))
        basis = pd.DataFrame(index=frame.index)
        for i, column in enumerate(columns):
            if column not in frame:
                basis[f'n1_{i}'] = np.nan
            else:
                value = observed_values(frame[column])
                if pd.api.types.is_numeric_dtype(value):
                    value = pd.to_numeric(value, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
                basis[f'n1_{i}'] = value
        # All actually planned model comparisons use the union of required columns.
        common = y.notna() & adjustment.notna().all(axis=1) & basis.notna().all(axis=1)
        patients = frame.loc[common, 'patient_id'].reset_index(drop=True)
        y = y.loc[common].astype(float).reset_index(drop=True) if kind in {'numeric', 'binary'} else y.loc[common].reset_index(drop=True)
        community = frame.loc[common, ['community_id']].reset_index(drop=True)
        x_a = _design(adjustment.loc[common].reset_index(drop=True))
        comm = _design(community).drop(columns='intercept')
        x_b = pd.concat([x_a, comm], axis=1)
        basis_design = _design(basis.loc[common].reset_index(drop=True)).drop(columns='intercept')
        x_c = pd.concat([x_a, basis_design], axis=1)
        designs = {'A': x_a, 'B': x_b}
        if columns:
            designs['C'] = x_c
            if include_d: designs['D'] = pd.concat([x_c, comm], axis=1)
        family = outcome.get('model_family', 'gaussian_ols' if kind == 'numeric' else '')
        topology = bool(len(frame) and manifest['release_status'] != 'NO_STABLE_STRUCTURE'
                        and frame.structural_stability_status.eq('stable').all())
        counts = community.groupby('community_id').size()
        support = bool(len(y) >= int(config['minimum_model_patients']) and len(counts) >= 2
                       and counts.min() >= int(config['minimum_group_patients']) and y.nunique() >= 2)
        for row in rows.values():
            row.update(n_patients=len(y), n_excluded_missing=len(frame)-len(y), model_family=family,
                       common_population_verified=True, topology_gate='PASS' if topology else 'BLOCKED',
                       measurement_gate='PASS' if role.measurement_status == 'VERIFIED' else 'BLOCKED',
                       covariate_gate='PASS' if not missing and era_status == 'VERIFIED' else 'BLOCKED',
                       support_gate='PASS' if support else 'BLOCKED')
        if not topology: block = 'TOPOLOGY_NOT_STABLE'
        elif role.measurement_status != 'VERIFIED': block = 'UNIT_OR_CATEGORY_PENDING_REVIEW'
        elif missing or era_status != 'VERIFIED': block = 'PRESPECIFIED_COVARIATES_UNAVAILABLE_OR_BASELINE_ERA_UNVERIFIED'
        elif not support: block = 'INSUFFICIENT_MODEL_SUPPORT'
        elif kind == 'categorical': block = 'MODEL_FORM_NOT_APPROVED_FOR_CATEGORICAL_ENDPOINT'
        elif kind == 'binary' and (family != 'binomial_logit'
                or outcome.get('statistical_review_status') != 'APPROVED'
                or not verified_reference(outcome, catalog)
                or isinstance(outcome.get('minimum_events_per_parameter'), bool)
                or not isinstance(outcome.get('minimum_events_per_parameter'), (int, float))
                or outcome['minimum_events_per_parameter'] <= 0):
            block = 'BINARY_MODEL_FORM_OR_EVENT_SUPPORT_REVIEW_PENDING'
        elif kind == 'numeric' and family != 'gaussian_ols': block = 'UNSUPPORTED_PRESPECIFIED_MODEL_FAMILY'
        if not block:
            for model, design in designs.items():
                if len(design) != len(patients) or not design.index.equals(y.index):
                    raise ValueError('COMMON_COMPLETE_CASE_POPULATION_MISMATCH')
                if np.linalg.matrix_rank(design) < design.shape[1]:
                    block = 'RANK_DEFICIENT_MODEL'; break
                if len(y)-design.shape[1] < 10:
                    block = 'INSUFFICIENT_RESIDUAL_DF'; break
            if kind == 'binary' and not block:
                cross = pd.crosstab(community.community_id, y).reindex(columns=[0., 1.], fill_value=0)
                max_parameters = max(d.shape[1] for d in designs.values())
                required = outcome['minimum_events_per_parameter'] * max_parameters
                if min(y.eq(1).sum(), y.eq(0).sum()) < required: block = 'INSUFFICIENT_EVENTS'
                elif cross.eq(0).any().any(): block = 'SEPARATION'
        for row in rows.values(): row['design_gate'] = 'PASS' if not block else 'BLOCKED'
        rows['C']['reason'] = basis_status if 'C' not in designs else ''
        rows['D']['reason'] = 'OPTIONAL_MODEL_D_NOT_PRESPECIFIED' if 'D' not in designs else ''
        if block:
            for row in rows.values(): row['reason'] = block
            model_rows.extend(rows.values()); continue
        # Fit transactionally: no partial successful comparisons if a planned model fails.
        try:
            fits = {m: _fit(y, d, family) for m, d in designs.items()}
            metrics = {m: _model_metrics(f, y, family) for m, f in fits.items()}
            planned = [('A', 'B')]
            if 'C' in fits: planned.append(('A', 'C'))
            if 'D' in fits: planned.append(('C', 'D'))
            local = []
            for reduced, full in planned:
                names = [c for c in designs[full] if c not in designs[reduced]]
                if not names: raise ValueError('NO_ADDITIONAL_MODEL_TERMS')
                restriction = np.zeros((len(names), designs[full].shape[1]))
                for i, name in enumerate(names): restriction[i, designs[full].columns.get_loc(name)] = 1.
                test = fits[full].wald_test(restriction, scalar=True)
                if not np.isfinite(test.pvalue): raise ValueError('NONFINITE_MODEL_TEST')
                comparison = dict(representation_id=identifier, source_column=source,
                    contrast=f'{full}_vs_{reduced}', hypothesis_family=f'{identifier}/n3_incremental',
                    family_size=family_sizes[identifier], status='COMPLETE', reason='', n_patients=len(y),
                    common_population_verified=True, delta_r2=np.nan, delta_adjusted_r2=np.nan,
                    delta_adjusted_r2_ci_low=np.nan, delta_adjusted_r2_ci_high=np.nan,
                    bootstrap_valid_replicates=0, delta_aic=float(fits[full].aic-fits[reduced].aic),
                    delta_brier=np.nan, p_value=float(test.pvalue), q_value=np.nan,
                    inference_scope=SCOPE, uncertainty_scope='conditional_on_frozen_Mapper_labels', **INTERNAL)
                if family == 'gaussian_ols':
                    comparison.update(delta_r2=float(fits[full].rsquared-fits[reduced].rsquared),
                                      delta_adjusted_r2=float(fits[full].rsquared_adj-fits[reduced].rsquared_adj))
                    salt = int(fingerprint(f'{identifier}/{source}/{full}/{reduced}')[:8], 16)
                    seed = int(manifest['random_seed']) + salt
                    lo, hi, n_valid = _delta_interval(y, designs[reduced], designs[full], int(config.get('effect_ci_replicates', 500)), seed)
                    comparison.update(delta_adjusted_r2_ci_low=lo, delta_adjusted_r2_ci_high=hi,
                                      bootstrap_valid_replicates=n_valid,
                                      reason='' if np.isfinite(lo) else 'CONDITIONAL_DELTA_INTERVAL_NOT_ESTIMABLE')
                else:
                    comparison['delta_brier'] = metrics[full]['brier']-metrics[reduced]['brier']
                local.append(comparison)
            for model in fits:
                rows[model].update(status='COMPLETE', reason='', **metrics[model])
            for comp in local:
                model = comp['contrast'].split('_')[0]
                rows[model]['p_value'] = comp['p_value']
            comparisons.extend(local)
        except (ValueError, np.linalg.LinAlgError, PerfectSeparationError, FloatingPointError) as exc:
            reason = 'SEPARATION' if isinstance(exc, PerfectSeparationError) else (
                str(exc) if str(exc) in {'SEPARATION', 'SEPARATION_OR_EXTREME_FITTED_PROBABILITIES', 'MODEL_NONCONVERGENCE', 'NONFINITE_MODEL_ESTIMATES', 'NONFINITE_MODEL_TEST', 'NO_ADDITIONAL_MODEL_TERMS'} else 'MODEL_FIT_FAILURE')
            for row in rows.values():
                row.update(status='NOT_ESTIMABLE', reason=reason, **{k: np.nan for k in METRICS})
        model_rows.extend(rows.values())
    models = pd.DataFrame(model_rows, columns=BASE_COLUMNS)
    comparison = pd.DataFrame(comparisons, columns=COMPARISON_COLUMNS)
    # The denominator includes every prespecified contrast, including blocked tests.
    for family_name, group in comparison.groupby('hypothesis_family'):
        total = int(group.family_size.iloc[0])
        p = pd.Series(group.p_value.tolist() + [1.]*(total-len(group)))
        comparison.loc[group.index, 'q_value'] = bh_qvalues(p).iloc[:len(group)].to_numpy()
    for comp in comparison.to_dict('records'):
        mask = (models.representation_id.eq(comp['representation_id']) & models.source_column.eq(comp['source_column'])
                & models.model.eq(comp['contrast'].split('_')[0]))
        models.loc[mask, 'q_value'] = comp['q_value']
    return dict(descriptive_profiles=profiles, incremental_eligibility=models.copy(),
                incremental_models=models, model_comparison=comparison)

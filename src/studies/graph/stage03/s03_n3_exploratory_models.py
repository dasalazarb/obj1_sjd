"""Technical lineage and retrospective baseline model support for exploratory 03e."""

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .s03_construct_validation import _paths, _tokens
from .s03_n3_candidate_audit import _proxy_alerts
from .s03_n3_exploratory_source import LAB_SOURCES, RF_VALUE, RF_TEXT, _clean, _collection_days
from .s03_n3_incremental import _design, _fit
from .s03_n3_resolution_support import support_diagnostics

DIRECT = 'DIRECTLY_DEFINING'
PROXY = 'RELATED_PROXY'
INDEPENDENT = 'INDEPENDENT_WITH_DOCUMENTED_LINEAGE'
UNRESOLVED = 'INDEPENDENCE_UNRESOLVED'
ESTIMATED = 'EXPLORATORY_ESTIMATED'
ANA_PAIR = {'lab__ana_status__text', 'lab__ana_hep2_status__text'}


def _independence(rep, source, roles, features, registry, dependencies, validation, policy):
    def audit(status, reason, blockers=()):
        return dict(representation_id=rep, outcome=source, classification=status, reason=reason,
                    defining_sources='|'.join(sorted(defining)), blocking_sources='|'.join(sorted(blockers)))

    selected = features.loc[features.representation_id.eq(rep) & features.included.eq(True)]
    defining = set(selected.source_column.dropna())
    encoded = set(selected.get('feature', pd.Series(dtype='string')).dropna())
    role = roles.get((rep, source), {})
    tier = role.get('tier')
    lineage = _paths(dependencies, source)
    exact_types = {'encoded_variant', 'derived_from', 'component_of'}
    direct = {target for target in defining | encoded if target in lineage and
              all(edge['relationship_type'] in exact_types for edge in lineage[target])}
    if direct or role.get('direct_in_discovery') is True or tier == 'N1_DEFINING':
        return audit(DIRECT, 'FROZEN_SOURCE_OR_DERIVATION_DEFINES_MAPPER', direct or {source})

    prefix = policy['clinical_proxy_prefixes'].get(source, [])
    related = ANA_PAIR if source in ANA_PAIR else {RF_VALUE, RF_TEXT} if source in {RF_VALUE, RF_TEXT} else {source}
    semantic_column = registry.get('semantic_feature_id', pd.Series(index=registry.index, dtype='string'))
    sources = registry.get('source_column', pd.Series(index=registry.index, dtype='string'))
    constructs = set(semantic_column.loc[sources.eq(source)].dropna())
    semantic_proxies = set(sources.loc[semantic_column.isin(constructs)].dropna())
    proxies = ((set(lineage) | related | semantic_proxies |
                {name for name in defining if any(name.startswith(p) for p in prefix)}) & defining)
    conflicts = _tokens(role.get('conflicting_defining_sources'))
    alerts = _proxy_alerts(source, defining, validation)
    if proxies or conflicts or alerts or tier == 'N2_RELATED':
        reason = '|'.join(['DOCUMENTED_LINEAGE_OR_CLINICAL_PROXY', *alerts])
        return audit(PROXY, reason, proxies | conflicts)
    if (not defining or not encoded or selected[['feature', 'source_column']].isna().any().any()
            or selected.groupby('feature').source_column.nunique().gt(1).any()):
        return audit(UNRESOLVED, 'FROZEN_DEFINING_FEATURE_LINEAGE_INCOMPLETE')
    if not constructs or len(constructs) != 1 or not any('__' in term for term in constructs):
        return audit(UNRESOLVED, 'OUTCOME_CONSTRUCT_LINEAGE_MISSING_OR_AMBIGUOUS')
    documented = set(sources.loc[semantic_column.notna()]) | {
        name for name in defining if any(edge[2].startswith(('prespecified_anchor:', 'frozen_registry:'))
                                         for edge in dependencies[name])}
    missing = defining - documented
    if missing:
        return audit(UNRESOLVED, 'DEFINING_SOURCE_REGISTRY_LINEAGE_MISSING', missing)
    if not role:
        return audit(UNRESOLVED, 'OUTCOME_ROLE_UNAVAILABLE')
    return audit(INDEPENDENT, 'FROZEN_FEATURE_REGISTRY_ROLE_AND_PROXY_CHECKS')


def _adjustment(frame):
    age = pd.to_numeric(frame['demo__age_at_baseline'], errors='coerce')
    sex = _clean(frame['demo__sex']).str.casefold().map(
        {'f': 'female', 'female': 'female', 'm': 'male', 'male': 'male'})
    year = pd.to_datetime(_collection_days(frame.clinical_baseline_date).astype('string'), errors='coerce').dt.year
    era = (year // 5 * 5).astype('Int64').astype('string')
    return pd.DataFrame({'age_at_baseline': age.where(age.between(0, 120)), 'sex': sex, 'baseline_era_5y': era})


def _model_support(frame, numeric, policy, cell, *, audit=None):
    """Shared 03e mask and support gates, optionally exposing private diagnostics."""
    observed = frame.value.notna()
    covariates = _adjustment(frame)
    complete = observed & covariates.notna().all(axis=1)
    counts = frame.loc[complete, 'community_id'].value_counts()
    keep = counts.index[counts.ge(policy['minimum_patients_per_group'])]
    common = complete & frame.community_id.isin(keep)
    y = frame.loc[common, 'value'].astype(float)
    adjustment = covariates.loc[common]
    adjustment = adjustment.loc[:, adjustment.nunique().gt(1)]
    info = dict(covariates='|'.join(adjustment.columns), n_complete_case=int(complete.sum()),
                n_model_eligible=len(y))
    if len(y) >= cell and not 0 < int(observed.sum()) - len(y) < cell:
        info['n'] = len(y)
    if audit is not None:
        audit.update(complete=complete, common=common, community_counts=counts, covariates=covariates)
    cohort_reason = ''
    if len(y) < policy['minimum_complete_cases'] or len(keep) < policy['minimum_groups']:
        cohort_reason = 'INSUFFICIENT_COMPLETE_CASE_COMMUNITY_SUPPORT'
        if audit is None:
            return dict(info, reason=cohort_reason), None
    community = frame.loc[common, 'community_id']
    binary_reason = ''
    if not numeric:
        cells = y.groupby(community).agg(['sum', 'size'])
        if (cells['sum'].lt(policy['minimum_binary_events_per_group']) |
                (cells['size'] - cells['sum']).lt(policy['minimum_binary_nonevents_per_group'])).any():
            binary_reason = 'INSUFFICIENT_BINARY_COMMUNITY_SUPPORT'
            if audit is None:
                return dict(info, reason=binary_reason), None
    support = dict(minimum_model_patients=policy['minimum_complete_cases'],
                   minimum_group_patients=policy['minimum_patients_per_group'],
                   minimum_patients_per_community=policy['minimum_patients_per_group'])
    diagnostics = support_diagnostics(
        y, adjustment, pd.DataFrame(index=y.index), community, 'numeric' if numeric else 'binary',
        support, categories_verified=True, minimum_events_per_parameter=policy['minimum_events_per_parameter'])
    reasons_by_model = {}
    for model in ('A', 'B'):
        design = diagnostics[model]
        reasons = set(filter(None, design['support_blocking_reasons'].split('|')))
        residual = design['n_residual_df']
        if residual is not None and residual < policy['minimum_residual_degrees_of_freedom']:
            reasons.add('INSUFFICIENT_RESIDUAL_DF')
        reasons_by_model[model] = reasons
    if audit is not None:
        audit.update(designs=diagnostics, reasons_by_model={
            model: reasons | set(filter(None, (cohort_reason, binary_reason)))
            for model, reasons in reasons_by_model.items()
        }, binary_community_blocked=bool(binary_reason))
    reasons = reasons_by_model['A'] | reasons_by_model['B']
    if cohort_reason or binary_reason or reasons:
        return dict(info, reason=cohort_reason or binary_reason or '|'.join(sorted(reasons))), None
    a = _design(adjustment)
    b = pd.concat([a, _design(community.to_frame()).drop(columns='intercept')], axis=1)
    return info, (y, a, b, sorted(community.unique()))


def _logit_hc3(y, design, fit):
    # Statsmodels GLM's generic HC3 option can fall back to HC0. Compute
    # the logistic hat diagonal and apply its HC3 adjustment to the scores.
    x = design.to_numpy(dtype=float)
    probability = np.asarray(fit.fittedvalues)
    weighted = x * np.sqrt(probability * (1 - probability))[:, None]
    bread = np.linalg.inv(weighted.T @ weighted)
    leverage = np.einsum('ij,jk,ik->i', weighted, bread, weighted)
    if (leverage >= 1 - 1e-10).any():
        raise ValueError('UNUSABLE_LEVERAGE')
    adjusted_scores = x * ((y.to_numpy() - probability) / (1 - leverage))[:, None]
    return bread @ (adjusted_scores.T @ adjusted_scores) @ bread


def _covariance(y, design, fit, numeric):
    if numeric:
        matrix = design.to_numpy()
        leverage = np.einsum('ij,ji->i', matrix, np.linalg.pinv(matrix))
        if np.max(leverage) >= 1 - 1e-10:
            raise ValueError('UNUSABLE_LEVERAGE')
        covariance = fit.cov_params().to_numpy()
    else:
        covariance = _logit_hc3(y, design, fit)
    if (not np.isfinite(covariance).all() or
            np.any(np.linalg.eigvalsh(covariance) <= 0) or
            np.linalg.matrix_rank(covariance) != len(covariance)):
        raise ValueError('UNUSABLE_COVARIANCE')
    return covariance


def _estimate(prepared, numeric, policy):
    y, a, b, levels = prepared
    family = 'gaussian_ols' if numeric else 'binomial_logit'
    label = policy['numeric_family' if numeric else 'binary_family']
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            fit_a, fit_b = (_fit(y, design, family) for design in (a, b))
            _covariance(y, a, fit_a, numeric)
            covariance = _covariance(y, b, fit_b, numeric)
            offset = a.shape[1]
            restriction = np.eye(b.shape[1])[offset:]
            p = float(fit_b.wald_test(restriction, cov_p=covariance, use_f=numeric, scalar=True).pvalue)
            center = fit_b.params.iloc[offset:].to_numpy()
            se = np.sqrt(np.diag(covariance)[offset:])
            z = norm.ppf(.975)
            effects = np.column_stack([center, center - z * se, center + z * se])
            if not numeric:
                effects = np.exp(effects)
            if not np.isfinite(effects).all() or not 0 <= p <= 1:
                raise ValueError('UNUSABLE_EFFECTS')
            contrasts = [dict(community_id=level, reference_community_id=levels[0],
                              scale='mean_difference' if numeric else 'odds_ratio',
                              effect=float(effect), ci95_lower=float(lo), ci95_upper=float(hi))
                         for level, (effect, lo, hi) in zip(levels[1:], effects)]
            result = dict(status=ESTIMATED, family=label, p_incremental=p, effects_json=json.dumps(contrasts),
                          n_parameters_a=a.shape[1], n_parameters_b=b.shape[1], residual_df_b=len(y) - b.shape[1],
                          diagnostics='HC3_WALD_F' if numeric else 'HC3_WALD_CHI2_NO_SEPARATION_DETECTED')
            if numeric:
                result.update(r2_a=float(fit_a.rsquared), r2_b=float(fit_b.rsquared),
                              observed_skewness=float(y.skew()))
            return result
    except (ValueError, RuntimeWarning, np.linalg.LinAlgError, OverflowError, PerfectSeparationError) as exc:
        reason = str(exc)
        if reason not in {'UNUSABLE_LEVERAGE', 'UNUSABLE_EFFECTS', 'UNUSABLE_COVARIANCE',
                          'MODEL_NONCONVERGENCE', 'SEPARATION', 'SEPARATION_OR_EXTREME_FITTED_PROBABILITIES'}:
            reason = 'UNUSABLE_EXPLORATORY_FIT'
        return dict(status='MODEL_DIAGNOSTIC_FAILURE', family=label, reason=reason)


def _family(source):
    if source in {RF_VALUE, RF_TEXT}:
        return 'rheumatoid_factor'
    return 'ana_related' if source in ANA_PAIR else LAB_SOURCES[source]


def _fdr(rows):
    tested = [row for row in rows if row['status'] == ESTIMATED]
    if not tested:
        return
    adjusted = multipletests([row['p_incremental'] for row in tested], method='fdr_bh')[1]
    families = '|'.join(sorted({row['hypothesis_family'] for row in tested}))
    for row, q in zip(tested, adjusted):
        row.update(q_bh=float(q), fdr_reject_05=bool(q <= .05), fdr_test_count=len(tested), fdr_families=families)

"""Cardinality-checked outcome/test classification and aggregate software QC."""
import numpy as np
import pandas as pd
from src.block_A.secondary_trajectory_analysis import ContractError
from src.block_A.temporal_spline_config import ANALYSIS_ROLE

TEST_MODELS = dict(linear_group_time='A', common_nonlinearity='B',
                   group_specific_curvature='C', global_temporal_heterogeneity='C')
OUTCOME_KEYS = ['outcome_id', 'variable', 'domain', 'outcome_type', 'unit']


def checked_merge(left, right, keys, validate):
    """A left join must preserve every row and match semantic outcome identity."""
    try:
        result = left.merge(right, on=keys, how='left', validate=validate)
    except pd.errors.MergeError as error:
        raise ContractError('nonunique outcome/result join') from error
    if len(result) != len(left):
        raise ContractError('outcome/result join changed row count')
    return result


def classify(eligibility, comparisons, tests, policy):
    if eligibility.empty:
        return pd.DataFrame()
    if eligibility.outcome_id.duplicated().any():
        raise ContractError('duplicate canonical outcome in eligibility')
    if len(tests) and tests.duplicated(['outcome_id', 'test']).any():
        raise ContractError('duplicate canonical outcome/test')
    if comparisons.duplicated(['outcome_id', 'model']).any():
        raise ContractError('duplicate canonical outcome/model')
    expected = []
    for row in eligibility.to_dict('records'):
        for test, model in TEST_MODELS.items():
            expected.append(dict(**{k: row[k] for k in OUTCOME_KEYS},
                interpretation_group=row['interpretation_group'], test=test, model=model,
                eligibility_status=row['eligibility_status_'+model], eligibility_reason=row['reason_'+model]))
    d = checked_merge(pd.DataFrame(expected), comparisons[OUTCOME_KEYS+['model', 'fit_status', 'reason']].rename(
        columns={'reason': 'fit_reason'}), OUTCOME_KEYS+['model'], 'many_to_one')
    if len(tests):
        d = checked_merge(d, tests[OUTCOME_KEYS+['test', 'status', 'reason', 'p', 'q_value', 'family_n_tests']].rename(
            columns={'status': 'test_status', 'reason': 'test_reason', 'family_n_tests': 'fdr_denominator'}),
            OUTCOME_KEYS+['test'], 'one_to_one')
    else:
        d = d.assign(test_status='NOT_REQUESTED', test_reason='tests_not_requested',
                     p=np.nan, q_value=np.nan, fdr_denominator=0)
    if d.fit_status.isna().any() or d.test_status.isna().any():
        raise ContractError('missing or semantically mismatched outcome/model/test')
    d['support_policy'], d['inference_role'] = policy, ANALYSIS_ROLE
    evidence = {
        'linear_group_time': ('LINEAR_GROUP_HETEROGENEITY', 'NO_LINEAR_EVIDENCE', 'A_NOT_ESTIMABLE'),
        'common_nonlinearity': ('COMMON_NONLINEARITY', 'NO_COMMON_NONLINEAR_EVIDENCE', 'B_NOT_ESTIMABLE'),
        'group_specific_curvature': ('EXPLORATORY_NONLINEAR_GROUP_HETEROGENEITY', 'NO_EXPLORATORY_NONLINEAR_EVIDENCE', 'C_NOT_ESTIMABLE'),
        'global_temporal_heterogeneity': ('EXPLORATORY_GLOBAL_TEMPORAL_HETEROGENEITY', 'NO_EXPLORATORY_GLOBAL_EVIDENCE', 'C_NOT_ESTIMABLE'),
    }
    statuses, reasons = [], []
    for row in d.to_dict('records'):
        valid = (row['fit_status'] == 'ESTIMATED' and row['test_status'] == 'ESTIMATED'
                 and np.isfinite(row['p']) and np.isfinite(row['q_value']))
        yes, no, unavailable = evidence[row['test']]
        statuses.append((yes if row['q_value'] < .05 else no) if valid else unavailable)
        reasons.append(row['eligibility_reason'] or row['fit_reason'] or row['test_reason'])
    d['status'], d['reason'] = statuses, reasons
    d['interpretation'] = 'Outcome/test evidence conditional on fixed retrospective groups; shared curvature is not group heterogeneity'
    return d


def qc_summary(tables):
    eligibility, models, tests = (tables[k] for k in
        ('00_variable_eligibility', '01_model_comparison', '02_global_temporal_tests'))
    summary = {}
    for model in 'ABC':
        rows = models.loc[models.model.eq(model)]
        summary[model] = dict(eligible=int(eligibility['eligible_'+model].sum()),
            estimated=int(rows.fit_status.eq('ESTIMATED').sum()),
            failed=int(rows.fit_status.eq('FIT_FAILED').sum()),
            not_estimable=int(rows.fit_status.eq('NOT_ESTIMABLE').sum()),
            not_requested=int(rows.fit_status.eq('NOT_REQUESTED').sum()),
            ineligibility_reasons=eligibility.loc[~eligibility['eligible_'+model], 'reason_'+model].value_counts().to_dict(),
            failure_reasons=rows.loc[rows.fit_status.eq('FIT_FAILED'), 'reason'].value_counts().to_dict(),
            estimators=rows.loc[rows.fit_status.eq('ESTIMATED'), 'model_family'].value_counts().to_dict())
    summary['valid_C_tests'] = int((tests.model.eq('C') & tests.status.eq('ESTIMATED') & tests.p.between(0, 1)).sum()) if len(tests) else 0
    return summary

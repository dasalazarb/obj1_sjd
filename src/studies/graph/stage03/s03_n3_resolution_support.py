"""Design/support diagnostics only. No estimators or discovery imports.

Even a numerically adequate design remains an unapproved proposal. Category
coding and endpoint conventions require content-pinned, scoped human reviews.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def design_matrix(data):
    """Use deterministic internal dummy names; clinical levels never leave memory."""
    parts = [np.ones((len(data), 1))]
    for column in data:
        values = data[column]
        if pd.api.types.is_numeric_dtype(values):
            parts.append(values.to_numpy(dtype=float).reshape(-1, 1))
        else:
            codes, levels = pd.factorize(values, sort=True)
            parts.extend((codes == k).astype(float).reshape(-1, 1) for k in range(1, len(levels)))
    return np.concatenate(parts, axis=1)


def support_diagnostics(y, adjustments, basis, community, kind, validation, *,
                        categories_verified=False, basis_requested=False,
                        include_d=False, minimum_events_per_parameter=None):
    """All planned designs share the union complete-case mask; no coefficient fit.

    Binary complete/quasi separation is a linear feasibility problem. LP status
    uncertainty is blocking, never interpreted as evidence of adequate support.
    """
    common = y.notna() & adjustments.notna().all(axis=1) & basis.notna().all(axis=1)
    common &= community.notna()
    n = int(common.sum())
    raw_complete = int((y.notna() & adjustments.notna().all(axis=1)).sum())
    rows = {}
    counts = community.loc[common].value_counts()
    reasons = []
    if n < validation['minimum_model_patients']:
        reasons.append('INSUFFICIENT_COVARIATE_COMPLETE_PATIENTS')
    if len(counts) < 2:
        reasons.append('INSUFFICIENT_COMMUNITIES')
    if len(counts) and counts.min() < max(validation['minimum_group_patients'],
                                         validation['minimum_patients_per_community']):
        reasons.append('LOW_COMMUNITY_SUPPORT')
    if y.loc[common].nunique() < 2:
        reasons.append('DEGENERATE_OUTCOME')
    reviewed = categories_verified and kind in {'numeric', 'binary'}
    designs = {}
    if reviewed and n:
        a = design_matrix(adjustments.loc[common])
        comm = design_matrix(community.loc[common].astype(str).to_frame()).T[1:].T
        designs = {'A': a, 'B': np.column_stack([a, comm])}
        if basis_requested:
            reduced = design_matrix(basis.loc[common]).T[1:].T
            designs['C'] = np.column_stack([a, reduced])
            if include_d:
                designs['D'] = np.column_stack([a, reduced, comm])
    binary = kind == 'binary' and reviewed
    events = int(y.loc[common].eq(1).sum()) if binary else None
    nonevents = int(y.loc[common].eq(0).sum()) if binary else None
    for model in ['A', 'B', 'C', 'D']:
        blocked = list(reasons)
        if model == 'C' and not basis_requested:
            blocked.append('N1_BASIS_REQUIRES_PRESPECIFICATION')
        if model == 'D' and not (basis_requested and include_d):
            blocked.append('MODEL_D_NOT_PRESPECIFIED')
        matrix = designs.get(model)
        parameters = rank = residual = None
        separation = 'UNKNOWN_BLOCKED' if binary else 'NOT_APPLICABLE' if reviewed else 'UNKNOWN_BLOCKED'
        if matrix is None:
            blocked.append('DESIGN_NOT_EVALUATED_UNREVIEWED_OR_UNAVAILABLE')
        else:
            parameters = matrix.shape[1]
            rank = int(np.linalg.matrix_rank(matrix))
            residual = n - parameters
            if rank < parameters:
                blocked.append('RANK_DEFICIENT_DESIGN')
            if residual < 10:
                blocked.append('INSUFFICIENT_RESIDUAL_DF')
            if binary:
                if not isinstance(minimum_events_per_parameter, (int, float)) or isinstance(minimum_events_per_parameter, bool) or not np.isfinite(minimum_events_per_parameter) or minimum_events_per_parameter <= 0:
                    blocked.append('EVENT_SUPPORT_RULE_REQUIRES_REVIEW')
                elif min(events, nonevents) < minimum_events_per_parameter * parameters:
                    blocked.append('INSUFFICIENT_EVENTS_PER_PARAMETER')
                signs = np.where(y.loc[common].to_numpy() == 1, 1., -1.)
                signed = signs[:, None] * matrix
                try:
                    # Nonnegative margins with at least one positive margin
                    # detect both complete and quasi-complete separation.
                    solution = linprog(np.zeros(parameters),
                        A_ub=np.vstack([-signed, -signed.sum(axis=0)]),
                        b_ub=np.r_[np.zeros(n), -1.], bounds=[(None, None)]*parameters,
                        method='highs')
                    separation = ('SEPARATION' if solution.status == 0 else
                                  'NO_SEPARATION_DETECTED' if solution.status == 2 else 'UNKNOWN_BLOCKED')
                except (ValueError, RuntimeError):
                    separation = 'UNKNOWN_BLOCKED'
                if separation != 'NO_SEPARATION_DETECTED':
                    blocked.append(separation)
        rows[model] = dict(model=model, n_Y_observed=int(y.notna().sum()),
            n_Y_covariate_complete=raw_complete, n_common_complete_case=n,
            n_complete_case_exclusion=len(y)-n, n_communities=len(counts),
            n_minimum_community=int(counts.min()) if len(counts) else 0,
            n_parameters=parameters, n_design_rank=rank, n_residual_df=residual,
            n_events=events, n_nonevents=nonevents, separation_status=separation,
            n_vulnerable_community_cells=int(counts.lt(validation['minimum_patients_per_community']).sum()),
            common_population_verified=True, design_support_gate='BLOCKED' if blocked else 'PREFLIGHT_READY_FOR_REVIEW',
            support_blocking_reasons='|'.join(sorted(set(blocked))))
    return rows

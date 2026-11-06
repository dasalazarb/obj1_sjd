"""New, fail-closed N3 release. Phase C/D and historical estimators stay immutable.

This module consumes verified frozen releases and scoped human decisions. It
reuses the N3 statistical primitives without changing their historical caller.
No discovery, threshold search, imputation or automated approval is performed.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from .._pipeline import digest
from ..stage02.s02_contracts import safe_artifact
from .s03_construct_validation import _dependencies, _paths
from .s03_n3_approved_view import protocol_view, measurement_values, validate_baseline
from .s03_n3_decisions import (
    VERSION, details_for, decision_templates, plan_sha256, verify_decisions,
)
from .s03_n3_incremental import _design, _fit, _model_metrics, _delta_interval, _pop_status
from .s03_n3_phasec_adjudication import CANDIDATES, load_phasec_inputs
from .s03_n3_resolution import _check_overlap, _snapshot
from .s03_n3_resolution_support import support_diagnostics
from .s03_release_reader import hard_members
from .s03_statistics import bh_qvalues
from .s03_profiles import observed_values

PASS = 'PASS'
FAIL = 'FAILED_WITH_EVIDENCE'
UNKNOWN = 'UNASSESSED_BECAUSE_UPSTREAM_BLOCKER'
GATES = ('protocol', 'baseline', 'measurement', 'temporal', 'independence', 'pop',
         'topology', 'design', 'support')
CONFOUNDERS = ['demo__protocol', 'demo__age_at_baseline', 'demo__sex']
SAFE_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_/.-]{0,100}\Z')


def validate_plan(plan, validation):
    if not isinstance(plan, dict) or plan.get('schema_version') != VERSION:
        raise ValueError('UNSUPPORTED_CONTROLLED_N3_PLAN')
    if plan.get('privacy_class') != 'INTERNAL_RESTRICTED' or plan.get('public_export') is not False:
        raise ValueError('RESTRICTED_NEW_RELEASE_REQUIRED')
    if plan.get('require_human_signoff') is not True or plan.get('allow_discovery_refit') is not False:
        raise ValueError('HUMAN_SIGNOFF_AND_FROZEN_TOPOLOGY_REQUIRED')
    if validation.get('confounders') != CONFOUNDERS:
        raise ValueError('PRESPECIFIED_CONFOUNDERS_CHANGED_AMENDMENT_REQUIRED')
    for field, floor in [('minimum_model_patients', 30), ('minimum_group_patients', 8),
                         ('minimum_patients_per_community', 8)]:
        if type(validation.get(field)) is not int or validation[field] < floor:
            raise ValueError('SUPPORT_THRESHOLD_WEAKENED')
    alpha = validation.get('fdr_alpha')
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha <= .05:
        raise ValueError('INVALID_FDR_ALPHA')
    if type(validation.get('effect_ci_replicates')) is not int or validation['effect_ci_replicates'] < 2:
        raise ValueError('BOOTSTRAP_REPLICATES_REQUIRED')
    if not isinstance(plan.get('outcomes'), list):
        raise ValueError('EXPLICIT_OUTCOME_PLAN_REQUIRED')
    keys = []
    for outcome in plan['outcomes']:
        rep, source = outcome.get('representation_id'), outcome.get('source_column')
        if rep not in {'R0', 'R1', 'R2', 'R3', 'R4'} or source not in CANDIDATES:
            raise ValueError('INVALID_PLANNED_REPRESENTATION_OR_SOURCE')
        models = outcome.get('requested_models', [])
        if (not isinstance(models, list) or not models or len(models) != len(set(models))
                or not set(models) <= set('ABCD') or 'A' not in models
                or 'D' in models and 'C' not in models):
            raise ValueError('INVALID_NESTED_MODEL_PLAN')
        basis = outcome.get('n1_basis', [])
        if (not isinstance(basis, list) or len(basis) != len(set(basis)) or
                any(not isinstance(c, str) or not c for c in basis) or
                bool(basis) != ('C' in models)):
            raise ValueError('REDUCED_N1_BASIS_MUST_BE_EXPLICITLY_REQUESTED')
        if type(outcome.get('use_pop')) is not bool:
            raise ValueError('EXPLICIT_POP_COMPARATOR_DECISION_REQUIRED')
        if (not isinstance(outcome.get('hypothesis_family'), str) or
                not SAFE_NAME.fullmatch(outcome['hypothesis_family'])):
            raise ValueError('PRESPECIFIED_HYPOTHESIS_FAMILY_REQUIRED')
        if outcome.get('topology_scope') not in {'full', 'R0_stable_only'}:
            raise ValueError('EXPLICIT_TOPOLOGY_SCOPE_REQUIRED')
        if outcome['topology_scope'] == 'R0_stable_only' and rep != 'R0':
            raise ValueError('STABLE_ONLY_IS_R0_SENSITIVITY')
        if outcome.get('model_family') not in {'gaussian_ols', 'binomial_logit'}:
            raise ValueError('UNSUPPORTED_MODEL_FAMILY')
        keys.append((rep, source))
    if len(keys) != len(set(keys)):
        raise ValueError('DUPLICATE_PLANNED_OUTCOME')
    # Text/numeric RF encodings cannot create independent hypothesis units.
    for rep in {r for r, _ in keys}:
        if sum(r == rep and CANDIDATES[s] == 'rheumatoid_factor' for r, s in keys) > 1:
            raise ValueError('RF_SAME_ASSAY_ENCODINGS_NOT_INDEPENDENT')
    if not isinstance(plan.get('authorized_reviewers'), list) or any(
            not isinstance(r, str) or not r.strip() for r in plan['authorized_reviewers']):
        raise ValueError('EXPLICIT_AUTHORIZED_REVIEWER_LIST_REQUIRED')


def _gate(ok, reason, *, assessed=True):
    return (PASS, '') if ok else (FAIL if assessed else UNKNOWN, reason)


def _failure(call):
    try:
        return call(), ''
    except ValueError as exc:
        # Only this module's controlled error codes can leave the IO boundary.
        reason = str(exc)
        if not re.fullmatch('[A-Z][A-Z0-9_]+', reason):
            reason = 'INVALID_SOURCE_CONTRACT'
        return None, reason


def prepare_pair(baseline, context, measurements, membership, roles, inventory,
                 features, registry, manifest, validation, outcome, decisions):
    """Evaluate independent gates together. No failed gate becomes an approval."""
    rep, source = outcome['representation_id'], outcome['source_column']
    review = {k: details_for(decisions, k, rep, source) for k in
              ('protocol', 'baseline', 'measurement', 'independence', 'pop', 'topology', 'design')}
    gates = {}
    viewed, reason = _failure(lambda: protocol_view(baseline, context, review['protocol']))
    gates['protocol'] = _gate(viewed is not None, reason)
    if viewed is None:
        viewed = baseline.copy(deep=True)
        # Missing/invalid evidence must not authorize even an existing field.
        viewed['demo__protocol'] = np.nan
    baseline_ok = bool(review['baseline']) and all(review['baseline'].get(k) is True for k in (
        'official_baseline_verified', 'episode_anchor_verified', 'era_date_semantics_verified'))
    gates['baseline'] = _gate(baseline_ok, 'OFFICIAL_BASELINE_AND_ERA_REVIEW_REQUIRED')
    measured, reason = _failure(lambda: measurement_values(baseline, measurements, source, review['measurement']))
    gates['measurement'] = _gate(measured is not None, reason)
    temporal_ok = (bool(review['measurement']) and review['measurement'].get('instrument_date_verified') is True
                   and review['measurement'].get('date_semantics') == 'instrument_measurement'
                   and measured is not None)
    gates['temporal'] = _gate(temporal_ok, reason or 'INSTRUMENT_BASELINE_WINDOW_UNVERIFIED')
    y, kind = measured if measured is not None else (pd.Series(np.nan, index=baseline.index), 'unknown')
    viewed[source] = y

    selected = roles.loc[roles.representation_id.eq(rep) & roles.source_column.eq(source)]
    defining = set(features.loc[features.representation_id.eq(rep) & features.included.eq(True), 'source_column'])
    graph = _dependencies(features, registry, validation)
    dependencies = set(_paths(graph, source)) & defining
    independent = (len(selected) == 1 and not selected.iloc[0].tier in {'N1_DEFINING', 'N2_RELATED'}
                   and source not in defining and not dependencies and bool(review['independence'])
                   and all(review['independence'].get(k) is True for k in (
                       'clinically_independent', 'source_lineage_verified', 'proxy_reviewed',
                       'discovery_exclusion_verified', 'same_assay_dependencies_reviewed')))
    gates['independence'] = _gate(independent, 'DEPENDENT_OR_UNAPPROVED_FOR_THIS_ESTIMAND')

    data = pd.DataFrame(index=baseline.index)
    for i, name in enumerate(CONFOUNDERS):
        values = viewed[name] if name in viewed else pd.Series(np.nan, index=baseline.index)
        if name == 'demo__age_at_baseline':
            values = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
        else:
            values = observed_values(values).astype('string').replace('', pd.NA).astype(object)
        data[f'covariate_{i}'] = values
    if baseline_ok:
        dates = pd.to_datetime(baseline.clinical_baseline_date, errors='coerce', utc=True)
        data['baseline_era'] = (dates.dt.year // 5 * 5).astype('Int64').astype('string').astype(object)
    else:
        data['baseline_era'] = np.nan
    pop_ok = True
    if outcome['use_pop']:
        crosswalk = review['pop'].get('crosswalk', {})
        pop_ok = (bool(review['pop']) and review['pop'].get('outcome_dependency_reviewed') is True
                  and review['pop'].get('semantics_verified') is True and isinstance(crosswalk, dict)
                  and set(crosswalk) == {'Pop1', 'Pop2', 'Pop3', 'Unclassifiable', 'Unclassified'}
                  and all(crosswalk.get(k) == k for k in ['Pop1', 'Pop2', 'Pop3'])
                  and all(crosswalk.get(k) in {'Unclassifiable', 'Unclassified'}
                          for k in ['Unclassifiable', 'Unclassified']))
        # Use the existing dependency guard, including documented derivatives.
        pop_status, _ = _pop_status(source, validation, rep, {})
        original = inventory.loc[inventory.representation_id.eq(rep) & inventory.source_column.eq(source)]
        if pop_status == 'POP_BASELINE_COMPARATOR_INELIGIBLE' or (
                len(original) == 1 and bool(original.iloc[0].get('pop_construct_dependency', False))):
            pop_ok = False
        pop = baseline.get('pop__status', pd.Series(pd.NA, index=baseline.index))
        mapped = pop.map(crosswalk) if isinstance(crosswalk, dict) else pd.Series(np.nan, index=baseline.index)
        data['pop_comparator'] = mapped.where(mapped.isin(['Pop1', 'Pop2', 'Pop3']))
    gates['pop'] = _gate(pop_ok, 'POP_CROSSWALK_OR_OUTCOME_INDEPENDENCE_UNAPPROVED')

    members = hard_members(membership, rep, stable_only=outcome['topology_scope'] == 'R0_stable_only')
    topology_ok = (rep != 'R2' and bool(review['topology']) and len(members) > 0
                   and members.structural_stability_status.eq('stable').all())
    if outcome['topology_scope'] == 'R0_stable_only':
        topology_ok &= all(review['topology'].get(k) is True for k in (
            'selected_before_outcomes', 'unchanged_thresholds', 'exclusion_bias_reviewed', 'sensitivity_design_approved'))
    else:
        topology_ok &= (manifest['release_status'] != 'NO_STABLE_STRUCTURE' and
                        review['topology'].get('full_structure_stable') is True)
    gates['topology'] = _gate(topology_ok, 'R2_EXPLORATORY' if rep == 'R2' else 'TOPOLOGY_UNSTABLE_OR_UNAPPROVED')
    community = baseline[['patient_id']].merge(members[['patient_id', 'community_id']], on='patient_id',
                                              how='left', validate='one_to_one').community_id
    community.index = baseline.index
    requested = set(outcome['requested_models'])
    basis_names = outcome.get('n1_basis', [])
    basis = pd.DataFrame(index=baseline.index)
    for i, name in enumerate(basis_names):
        basis[f'n1_{i}'] = observed_values(baseline[name]) if name in baseline else np.nan
        if pd.api.types.is_numeric_dtype(basis[f'n1_{i}']):
            basis[f'n1_{i}'] = pd.to_numeric(basis[f'n1_{i}'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    design_ok = (bool(review['design']) and review['design'].get('requested_models') == outcome['requested_models']
                 and review['design'].get('n1_basis', []) == basis_names
                 and review['design'].get('hypothesis_family') == outcome['hypothesis_family']
                 and review['design'].get('model_family') == outcome['model_family']
                 and review['design'].get('multiplicity_prespecified') is True
                 and review['design'].get('common_complete_case') is True
                 and review['design'].get('covariate_semantics_verified') is True
                 and set(basis_names) <= defining
                 and (not basis_names or (review['design'].get('n1_basis_source_semantics_verified') is True
                     and review['design'].get('reduced_n1_basis_verified') is True))
                 and (kind == 'unknown' or (kind == 'numeric' and outcome['model_family'] == 'gaussian_ols') or
                      (kind == 'binary' and outcome['model_family'] == 'binomial_logit')))
    gates['design'] = _gate(design_ok, 'DESIGN_MODEL_FAMILY_OR_REDUCED_N1_BASIS_UNAPPROVED')
    common = y.notna() & data.notna().all(axis=1) & basis.notna().all(axis=1)
    graph_cohort = topology_ok and bool(requested & {'B', 'D'})
    if graph_cohort:
        common &= community.notna()
    designs = {}
    if common.any() and kind in {'numeric', 'binary'}:
        a = _design(data.loc[common].reset_index(drop=True))
        c = pd.concat([a, _design(basis.loc[common].reset_index(drop=True)).drop(columns='intercept')], axis=1)
        designs = {'A': a, 'C': c}
        if graph_cohort:
            comm = _design(community.loc[common].to_frame().reset_index(drop=True)).drop(columns='intercept')
            designs.update(B=pd.concat([a, comm], axis=1), D=pd.concat([c, comm], axis=1))
    selected_y = y.loc[common].reset_index(drop=True)
    selected_y = pd.to_numeric(selected_y, errors='coerce')
    # Reuse the reviewed Phase D rank/event/separation diagnostic for graph models.
    diagnostics = support_diagnostics(y, data, basis, community, kind, validation,
        categories_verified=kind in {'numeric', 'binary'}, basis_requested='C' in requested,
        include_d='D' in requested, minimum_events_per_parameter=outcome.get('minimum_events_per_parameter'))
    upstream = all(gates[k][0] == PASS for k in ('protocol', 'baseline', 'measurement', 'temporal', 'pop', 'design'))
    rows = []
    for model in 'ABCD':
        row_gates = dict(gates)
        if model in {'A', 'C'}:
            row_gates['topology'] = ('NOT_APPLICABLE', '')
        matrix = designs.get(model)
        reasons = []
        graph_model = model in {'B', 'D'}
        assessed = upstream and (not graph_model or topology_ok)
        if assessed:
            if len(selected_y) < validation['minimum_model_patients']:
                reasons.append('INSUFFICIENT_MODEL_PATIENTS')
            if selected_y.nunique() < 2:
                reasons.append('DEGENERATE_OUTCOME')
            if matrix is None:
                reasons.append('DESIGN_UNAVAILABLE')
            else:
                if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
                    reasons.append('RANK_DEFICIENT_DESIGN')
                if len(selected_y) - matrix.shape[1] < 10:
                    reasons.append('INSUFFICIENT_RESIDUAL_DF')
                if kind == 'binary':
                    event_rule = outcome.get('minimum_events_per_parameter')
                    if (isinstance(event_rule, bool) or not isinstance(event_rule, (int, float)) or
                            not np.isfinite(event_rule) or event_rule <= 0):
                        reasons.append('BINARY_EVENT_RULE_UNAPPROVED')
                    elif min(selected_y.eq(0).sum(), selected_y.eq(1).sum()) < event_rule * matrix.shape[1]:
                        reasons.append('INSUFFICIENT_EVENTS_PER_PARAMETER')
                    # LP separation diagnostic with base-only designs, too.
                    if not graph_model:
                        from scipy.optimize import linprog
                        signs = np.where(selected_y.to_numpy() == 1, 1., -1.)
                        signed = signs[:, None] * matrix.to_numpy()
                        solution = linprog(np.zeros(matrix.shape[1]), A_ub=np.vstack([-signed, -signed.sum(axis=0)]),
                            b_ub=np.r_[np.zeros(len(signed)), -1.], bounds=[(None, None)] * matrix.shape[1], method='highs')
                        if solution.status != 2:
                            reasons.append('SEPARATION' if solution.status == 0 else 'SEPARATION_UNKNOWN')
            if graph_model:
                reasons.extend(filter(None, diagnostics[model]['support_blocking_reasons'].split('|')))
                if matrix is not None and matrix.shape[1] == designs['A' if model == 'B' else 'C'].shape[1]:
                    reasons.append('NO_ADDITIONAL_COMMUNITY_TERMS')
        row_gates['support'] = _gate(assessed and not reasons,
            '|'.join(sorted(set(reasons))) if assessed else 'UPSTREAM_SOURCE_OR_DESIGN_UNRESOLVED', assessed=assessed)
        if model not in requested:
            status = 'NOT_REQUESTED'
        else:
            status = 'READY' if all(s[0] in {PASS, 'NOT_APPLICABLE'} for s in row_gates.values()) else 'NOT_ESTIMABLE'
        row = dict(representation_id=rep, source_column=source, model=model, status=status,
                   topology_scope=outcome['topology_scope'], hypothesis_family=outcome['hypothesis_family'],
                   analysis_population='common_frozen_hard_membership' if graph_cohort else 'common_official_baseline',
                   n_patients=len(selected_y), n_parameters=matrix.shape[1] if matrix is not None else None,
                   common_population_verified=True, inference_scope='exploratory_in_sample_frozen_memberships')
        for name, (state, why) in row_gates.items():
            row[name + '_gate'] = state if model in requested else 'NOT_REQUESTED'
            row[name + '_reason'] = why if model in requested else ''
        rows.append(row)
    return rows, dict(y=selected_y, designs=designs, view=viewed,
                      patient_ids=baseline.loc[common, 'patient_id'].reset_index(drop=True))


def estimate_pair(rows, prepared, outcome, manifest, validation, *, execute):
    """Reuse HC3 fits, paired bootstrap and the existing conditional Wald tests."""
    models = deepcopy(rows)
    fits, metrics = {}, {}
    for row in models:
        if row['status'] != 'READY' or not execute:
            continue
        model = row['model']
        try:
            fit = _fit(prepared['y'], prepared['designs'][model], outcome['model_family'])
            result = _model_metrics(fit, prepared['y'], outcome['model_family'])
            fits[model], metrics[model] = fit, result
            row.update(result, status='COMPLETE')
        except Exception:
            # No raw exception text, patient labels or category levels in reports.
            row.update(status='NOT_ESTIMABLE', fit_reason='MODEL_FIT_FAILURE')
    comparisons = []
    for reduced, full in [('A', 'B'), ('A', 'C'), ('C', 'D')]:
        if full not in outcome['requested_models']:
            continue
        comp = dict(representation_id=outcome['representation_id'], source_column=outcome['source_column'],
                    contrast=full + '_vs_' + reduced, hypothesis_family=outcome['hypothesis_family'],
                    status='NOT_ESTIMABLE' if execute else 'NOT_EXECUTED', reason='REQUIRED_MODEL_UNAVAILABLE',
                    p_value=np.nan, q_value=np.nan, delta_adjusted_r2=np.nan, delta_aic=np.nan,
                    delta_brier=np.nan, delta_adjusted_r2_ci_low=np.nan, delta_adjusted_r2_ci_high=np.nan,
                    common_population_verified=True, inference_scope='exploratory_in_sample_frozen_memberships')
        if full in fits and reduced in fits:
            try:
                a, b = prepared['designs'][reduced], prepared['designs'][full]
                if not a.index.equals(b.index) or len(a) != len(prepared['patient_ids']):
                    raise ValueError('COMMON_POPULATION_MISMATCH')
                added = [name for name in b if name not in a]
                if not added:
                    raise ValueError('NO_ADDITIONAL_TERMS')
                restriction = np.zeros((len(added), b.shape[1]))
                for i, name in enumerate(added):
                    restriction[i, b.columns.get_loc(name)] = 1.
                p = float(fits[full].wald_test(restriction, scalar=True).pvalue)
                if not np.isfinite(p):
                    raise ValueError('NONFINITE_TEST')
                comp.update(status='COMPLETE', reason='', p_value=p,
                            delta_aic=float(fits[full].aic - fits[reduced].aic))
                if outcome['model_family'] == 'gaussian_ols':
                    from .s03_n3_candidate_audit import fingerprint
                    salt = int(fingerprint('/'.join([outcome['representation_id'], outcome['source_column'], full, reduced]))[:8], 16)
                    lo, hi, _ = _delta_interval(prepared['y'], a, b, validation['effect_ci_replicates'],
                                               int(manifest['random_seed']) + salt)
                    comp.update(delta_adjusted_r2=float(fits[full].rsquared_adj - fits[reduced].rsquared_adj),
                                delta_adjusted_r2_ci_low=lo, delta_adjusted_r2_ci_high=hi,
                                reason='' if np.isfinite(lo) else 'CONDITIONAL_INTERVAL_NOT_ESTIMABLE')
                else:
                    comp['delta_brier'] = metrics[full]['brier'] - metrics[reduced]['brier']
            except Exception:
                comp.update(status='NOT_ESTIMABLE', reason='COMPARISON_FIT_FAILURE', p_value=np.nan,
                            delta_aic=np.nan, delta_brier=np.nan, delta_adjusted_r2=np.nan,
                            delta_adjusted_r2_ci_low=np.nan, delta_adjusted_r2_ci_high=np.nan)
        comparisons.append(comp)
    return models, comparisons


def correct_multiplicity(comparisons, alpha):
    """Blocked planned contrasts stay in the fixed family denominator."""
    for family in sorted({r['hypothesis_family'] for r in comparisons}):
        rows = [r for r in comparisons if r['hypothesis_family'] == family]
        p = pd.Series([r['p_value'] if np.isfinite(r['p_value']) else 1. for r in rows])
        q = bh_qvalues(p)
        for row, value in zip(rows, q):
            row['family_size'] = len(rows)
            row['q_value'] = float(value) if np.isfinite(row['p_value']) else np.nan
            row['reject_fdr'] = bool(np.isfinite(row['q_value']) and row['q_value'] <= alpha)


def blocker_ledger(rows, decision_states, decision_pointers=()):
    evidence_by_scope = {(p.get('section'), p.get('representation_id'), p.get('source_column')): p.get('sha256', '')
                         for p in decision_pointers if re.fullmatch('[a-f0-9]{64}', str(p.get('sha256', '')))}
    result = []
    for row in rows:
        if row['status'] == 'NOT_REQUESTED':
            continue
        for gate in GATES:
            state = row[gate + '_gate']
            decision_section = 'measurement' if gate == 'temporal' else gate
            result.append(dict(representation_id=row['representation_id'], source_column=row['source_column'],
                model=row['model'], root_cause=gate, blocking_reason=row[gate + '_reason'],
                source_evidence=evidence_by_scope.get((decision_section, row['representation_id'], row['source_column']), ''),
                owner='statistical_reviewer' if gate in {'design', 'support'} else 'clinical_reviewer',
                action='RETAIN_VERIFIED_CONTRACT' if state in {PASS, 'NOT_APPLICABLE'} else 'RESOLVE_SOURCE_OR_SCOPED_DECISION',
                review_status=decision_states.get((decision_section, row['representation_id'], row['source_column']),
                                                   'NOT_APPLICABLE' if gate == 'support' else 'PENDING_REVIEW'),
                technical_status=state, resolved_on_version=VERSION if state in {PASS, 'NOT_APPLICABLE'} else ''))
    return result


def _safe_table(rows):
    frame = pd.DataFrame(rows)
    # Counts and df are complementary suppressed across the entire report.
    for name in frame:
        if name.startswith('n_') or name in {'residual_df', 'family_size'}:
            if name != 'family_size':  # Hypothesis counts contain no patients.
                frame[name] = None
    frame['privacy_class'] = 'INTERNAL_RESTRICTED'
    frame['shareable_export_allowed'] = False
    frame['patient_count_disclosure'] = 'WITHHELD_WHOLE_RUN'
    return frame


def input_pins(graph03_run, graph02_release, context_path, measurements_path, repository_root):
    """Hash explicit inputs to draft a plan; hashing never grants clinical approval."""
    release = Path(graph02_release)
    if release.is_dir():
        release = safe_artifact(release, '02_graph_release_manifest.json')
    return dict(graph02_release=digest(release),
        graph03_summary=digest(safe_artifact(Path(graph03_run), '03_graph_results_summary.json')),
        context=digest(Path(context_path)), measurements=digest(Path(measurements_path)),
        validation=digest(Path(repository_root) / 'src/studies/graph/configs/validation_03.yaml'),
        implementation=_implementation_pin(repository_root))


def _implementation_hashes(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted((root / 'src/studies/graph').rglob('*.py'))}


def _implementation_pin(root):
    from .s03_n3_candidate_audit import fingerprint
    return fingerprint(json.dumps(_implementation_hashes(root), sort_keys=True))


def _write_aggregate_figures(output, comparisons):
    """Restricted, source-backed effect plots; never a public export route."""
    complete = [r for r in comparisons if r['status'] == 'COMPLETE']
    if not complete:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric, label in [('delta_adjusted_r2', 'Change in adjusted R²'),
                          ('delta_brier', 'Change in Brier score')]:
        selected = [r for r in complete if np.isfinite(r[metric])]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(10, max(3, .45 * len(selected) + 1)))
        for i, row in enumerate(selected):
            value = row[metric]
            ax.plot(value, i, 'o', color='#24536b')
            if metric == 'delta_adjusted_r2':
                lo, hi = row['delta_adjusted_r2_ci_low'], row['delta_adjusted_r2_ci_high']
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.plot([lo, hi], [i, i], color='#24536b')
        ax.set_yticks(range(len(selected)), [r['representation_id'] + ' / ' + r['source_column'] +
                                           ' / ' + r['contrast'] for r in selected])
        ax.axvline(0, color='grey', linewidth=.8)
        ax.set_xlabel(label)
        ax.set_title('Exploratory conditional comparisons · INTERNAL RESTRICTED')
        fig.tight_layout()
        fig.savefig(output / (metric + '.png'), dpi=150)
        plt.close(fig)


def run_controlled(graph03_run, graph02_release, context_path, measurements_path, output_dir,
                   plan_path, decision_root, repository_root, *, execute=False):
    """Publish a new private release, or blocked preflight, with frozen SHA checks.

    All execution inputs are explicit and independently pinned. No compatible-run
    discovery changes an approved estimand. A preflight can generate templates
    with unresolved decisions; --execute still runs only individually READY rows.
    """
    root = Path(repository_root).resolve()
    graph03, release = Path(graph03_run).resolve(), Path(graph02_release).resolve()
    if release.is_dir():
        release = safe_artifact(release, '02_graph_release_manifest.json')
    frozen_roots = [graph03, release.parent]
    output = Path(output_dir).resolve()
    _check_overlap(output, frozen_roots)
    if not all(r.is_dir() for r in frozen_roots) or any(
            a.is_relative_to(b) or b.is_relative_to(a) for a, b in [(graph03, release.parent)]):
        raise ValueError('DISTINCT_FROZEN_INPUTS_REQUIRED')
    plan_path = Path(plan_path).resolve()
    validation_path = root / 'src/studies/graph/configs/validation_03.yaml'
    plan = yaml.safe_load(plan_path.read_text())
    validation = yaml.safe_load(validation_path.read_text())
    validate_plan(plan, validation)
    context_path, measurements_path = Path(context_path).resolve(), Path(measurements_path).resolve()
    if any(output == p or output in p.parents for p in [plan_path, validation_path, context_path, measurements_path]):
        raise ValueError('OUTPUT_OVERLAPS_SOURCE_INPUT')
    protected = {p: digest(p) for p in [plan_path, validation_path, context_path, measurements_path]}
    before = _snapshot(frozen_roots)
    input_hashes = dict(graph02_release=digest(release),
        graph03_summary=digest(safe_artifact(graph03, '03_graph_results_summary.json')),
        context=protected[context_path], measurements=protected[measurements_path],
        validation=protected[validation_path], implementation=_implementation_pin(root))
    if plan.get('input_sha256') != input_hashes:
        raise ValueError('EXPLICIT_INPUT_HASH_PINS_REQUIRED_OR_MISMATCHED')
    created = False
    try:
        # Reuse the frozen release validator, not a new permissive CSV reader.
        from .s03_n3_phasec_adjudication import VERSION as PHASEC_VERSION
        contract = dict(schema_version=PHASEC_VERSION, auto_approve_independence=False,
            privacy_class='INTERNAL_RESTRICTED', minimum_disclosure_cell=5,
            input_contract=dict(n3_schema_version='graph03_n3_audit_v1',
                                graph02_release_sha256=input_hashes['graph02_release']),
            families={f: dict(sources=[s for s in CANDIDATES if CANDIDATES[s] == f]) for f in set(CANDIDATES.values())})
        inputs = load_phasec_inputs(graph03, release, contract, strict=True)
        # Verify the complete Graph03 inventory, not just the Phase A subset.
        summary = json.loads((graph03 / '03_graph_results_summary.json').read_text())
        for name, sha in summary['sha256_by_artifact'].items():
            if digest(safe_artifact(graph03, name)) != sha:
                raise ValueError('GRAPH03_ARTIFACT_SHA_MISMATCH')
        baseline = inputs['frozen']['baseline_raw'].reset_index(drop=True)
        validate_baseline(baseline)
        manifest = inputs['manifest']
        if not {o['representation_id'] for o in plan['outcomes']} <= set(manifest['eligible_representation_ids']):
            raise ValueError('OUTCOME_REPRESENTATION_OUTSIDE_FROZEN_RELEASE')
        registry = pd.read_csv(safe_artifact(release.parent, manifest['relative_paths']['clinical_registry']))
        context, measurements = pd.read_parquet(context_path), pd.read_parquet(measurements_path)
        decisions, states, documents = verify_decisions(plan, decision_root, input_hashes)
        protected.update(documents)
        # Every implementation file used by this new release is content-pinned.
        code_hashes = _implementation_hashes(root)
        from .s03_n3_candidate_audit import fingerprint
        if fingerprint(json.dumps(code_hashes, sort_keys=True)) != input_hashes['implementation']:
            raise ValueError('IMPLEMENTATION_CHANGED_DURING_INPUT_VERIFICATION')
        protected.update({root / name: sha for name, sha in code_hashes.items()})
        preflight, models, comparisons, views, preparations = [], [], [], {}, {}
        for outcome in plan['outcomes']:
            rows, prepared = prepare_pair(baseline, context, measurements, inputs['frozen']['membership'],
                inputs['roles'], inputs['tables']['source_inventory'], inputs['frozen']['feature_sets'],
                registry, manifest, validation, outcome, decisions)
            pair = (outcome['representation_id'], outcome['source_column'])
            views[pair] = prepared['view']
            preparations[pair] = prepared
            preflight.extend(rows)
        # Two approved assignments cannot give contradictory adjustment views.
        approved_views = [views[(r['representation_id'], r['source_column'])]['demo__protocol']
                          for r in preflight if r['model'] == 'A' and r['protocol_gate'] == PASS]
        if any(not approved_views[0].equals(v) for v in approved_views[1:]):
            raise ValueError('INCONSISTENT_APPROVED_PROTOCOL_VIEWS')
        # Distinct ANA names cannot hide a shared original assay in the plan.
        ana_units = []
        for outcome in plan['outcomes']:
            if CANDIDATES[outcome['source_column']] == 'ana':
                detail = details_for(decisions, 'measurement', outcome['representation_id'], outcome['source_column'])
                if detail.get('assay_id'):
                    ana_units.append((outcome['representation_id'], detail['assay_id']))
        if len(ana_units) != len(set(ana_units)):
            raise ValueError('ANA_SAME_ASSAY_HYPOTHESES_REQUIRE_AMENDMENT')
        for outcome in plan['outcomes']:
            pair = (outcome['representation_id'], outcome['source_column'])
            rows = [r for r in preflight if (r['representation_id'], r['source_column']) == pair]
            fitted, contrasts = estimate_pair(rows, preparations[pair], outcome, manifest, validation, execute=execute)
            models.extend(fitted)
            comparisons.extend(contrasts)
        correct_multiplicity(comparisons, validation['fdr_alpha'])
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8]
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        tables = dict(preflight=_safe_table(preflight), models=_safe_table(models),
                      comparisons=_safe_table(comparisons), blockers=_safe_table(blocker_ledger(preflight, states, plan.get('decisions', []))))
        # Patient artifacts are isolated from the report, restricted and never exported.
        payloads = {name + '.csv': frame.to_csv(index=False) for name, frame in tables.items()}
        from .s03_n3_validation import _privacy_check
        _privacy_check(tables, baseline)
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        created = True
        for name, text in payloads.items():
            path = output / name
            path.write_text(text)
            path.chmod(0o600)
        _write_aggregate_figures(output, comparisons)
        templates = decision_templates(plan, input_hashes, manifest['eligible_representation_ids'], list(CANDIDATES))
        (output / 'unsigned_decision_templates.yaml').write_text(yaml.safe_dump(templates, sort_keys=False))
        private = output / 'private'
        private.mkdir(mode=0o700)
        for i, outcome in enumerate(plan['outcomes']):
            pair = (outcome['representation_id'], outcome['source_column'])
            row = next(r for r in preflight if (r['representation_id'], r['source_column'], r['model']) == (*pair, 'A'))
            if row['protocol_gate'] == PASS and row['measurement_gate'] == PASS and row['baseline_gate'] == PASS:
                path = private / ('approved_view_%03d.parquet' % i)
                views[pair].to_parquet(path, index=False)
                path.chmod(0o600)
            # Exact common-case patient lists stay internal for reproducibility.
            if any(r['status'] == 'READY' for r in preflight if (r['representation_id'], r['source_column']) == pair):
                path = private / ('common_cohort_%03d.parquet' % i)
                preparations[pair]['patient_ids'].to_frame().to_parquet(path, index=False)
                path.chmod(0o600)
        completed = any(r['status'] == 'COMPLETE' for r in comparisons)
        metadata = dict(schema_version=VERSION, run_id=run_id, implementation_commit_sha=commit,
            implementation_sha256=code_hashes, plan_sha256=plan_sha256(plan), plan_file_sha256=digest(plan_path),
            input_sha256=input_hashes, decision_sha256=sorted(set(documents.values())),
            privacy_class='INTERNAL_RESTRICTED', public_export=False, no_refit=True,
            graph02_unchanged=True, graph03_unchanged=True, automatic_approvals=0,
            institutional_signature_authentication='EXTERNAL_HUMAN_RESPONSIBILITY',
            status='COMPLETE_WITH_ESTIMATES' if completed else 'COMPLETE_WITH_BLOCKERS' if execute else 'PREFLIGHT_COMPLETE',
            inference_scope='exploratory_in_sample_conditional_on_frozen_memberships',
            patient_count_disclosure='WITHHELD_WHOLE_RUN',
            clinical_feasibility='AT_LEAST_ONE_APPROVED_CONTRAST' if completed else 'NO_APPROVED_ESTIMABLE_CONTRAST_IN_THIS_RUN',
            nonestimability_does_not_establish_insufficient_power_after_upstream_repairs=True)
        frozen_inventory = {('graph03/' if path.is_relative_to(graph03) else 'graph02/') +
                            path.relative_to(graph03 if path.is_relative_to(graph03) else release.parent).as_posix(): sha
                            for path, sha in before.items()}
        metadata['frozen_sha256_before'] = frozen_inventory
        metadata['frozen_sha256_after'] = {name: digest(
            (graph03 if name.startswith('graph03/') else release.parent) / name.split('/', 1)[1])
            for name in frozen_inventory}
        from importlib.metadata import version
        import sys
        metadata['runtime_versions'] = dict(python=sys.version.split()[0], **{
            package: version(package) for package in ('numpy', 'pandas', 'pyarrow', 'scipy',
                'statsmodels', 'scikit-learn', 'matplotlib', 'PyYAML')})
        (output / 'release.json').write_text(json.dumps(metadata, sort_keys=True, indent=2) + '\n')
        (output / 'README.md').write_text(
            '# Restricted controlled N3 release\n\n' + metadata['status'] + '\n\n'
            'Use preflight.csv and blockers.csv for simultaneous gate states. '
            'NOT_REQUESTED is distinct from NOT_ESTIMABLE. A/C do not require '
            'topological stability; community contrasts do. All requested models '
            'for a pair use the same complete-case cohort. R0 stable-only is a '
            'sensitivity, never evidence for stability of full R0. R2 is exploratory. '
            'Blocked hypotheses remain in the prespecified BH denominator. '
            'Negative and nonestimable comparisons are retained.\n\n'
            'private/ contains patient-level derived views and cohort lists. Keep '
            'the entire run on approved institutional infrastructure. Counts and '
            'degrees of freedom are withheld across reports. No automatic export, '
            'ZIP or publication is permitted. Source review and disclosure approval '
            'remain institutional responsibilities. No clinical validity, external '
            'prediction or prognosis is established.\n\n'
            'SHA inventories bind inputs, implementation, decisions and outputs; '
            'frozen Graph02/03 files are verified before and after execution.\n')
        for path in output.rglob('*'):
            if path.is_file():
                path.chmod(0o600)
        inventory_out = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()}
        (output / 'artifact_sha256.json').write_text(json.dumps(inventory_out, sort_keys=True, indent=2) + '\n')
        (output / 'artifact_sha256.json').chmod(0o600)
        return dict(status=metadata['status'], output_dir=str(output), run_id=run_id)
    except BaseException:
        if created:
            shutil.rmtree(output)
        raise
    finally:
        if _snapshot(frozen_roots) != before or any(not p.is_file() or digest(p) != sha for p, sha in protected.items()):
            if created and output.exists():
                shutil.rmtree(output)
            raise ValueError('PROTECTED_INPUT_CHANGED_DURING_CONTROLLED_N3')

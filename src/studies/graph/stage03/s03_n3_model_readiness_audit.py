"""Read-only Graph03g diagnostics over the exploratory Graph03e contract."""

from contextlib import ExitStack
import json
import os
import stat
from pathlib import Path
import re

import pandas as pd
import yaml

from .._pipeline import digest, output_lock, stage_destinations, stage_outputs, write_json
from ..stage02.s02_contracts import MANIFEST, artifact_paths, safe_artifact, validate_release
from . import s03_n3_exploratory as E
from . import s03_n3_exploratory_models as M
from . import s03_n3_exploratory_reporting as R
from . import s03_n3_exploratory_source as S
from . import s03_n3_readiness_reporting as REPORT
from .s03_baseline_window import DAY_BINS, policy as baseline_policy, require_policy
from .s03_construct_validation import _dependencies, _paths, _review, feature_roles, independence_review_approved
from .s03_release_reader import RESULTS, RESULTS_SCHEMA, hard_members
from .s03_run_resolution import graph_paths

VERSION = 'graph03g_model_readiness_v2'
DESTINATION = '03_graph_n3_model_readiness_audit'
UNKNOWN = 'NOT_ASSESSABLE_FROM_PUBLISHED_AGGREGATES'
UNAVAILABLE = 'DATA_UNAVAILABLE'
SUPPRESSED = 'SUPPRESSED'
PASS = 'PASS'
PAIRS = ['representation_id', 'outcome']
GRAPH03 = {
    'summary': RESULTS,
    'release_qc': '03_graph_release_input_qc.json',
    'window_qc': '03_graph_baseline_window_qc.json',
    'roles': '03_graph_clinical_feature_roles.csv',
    'inventory': '03_graph_n3_source_inventory.csv',
}
GRAPH03E = {
    'summary': '03e_summary.json',
    'window_qc': '03_graph_baseline_window_qc.json',
    'coverage': '03e_coverage.csv',
    'models': '03e_models.csv',
    'independence': '03e_independence_audit.csv',
    'flow': '03e_cohort_flow.csv',
    'exclusions': '03e_exclusions.csv',
    'linkage': '03e_source_linkage_audit.csv',
}
CSV_OUTPUTS = ('readiness_matrix', 'independence_lineage', 'measurement_funnel', 'statistical_support',
               'review_queue', 'source_assay_audit', 'pipeline_checks')
OUTPUT_FILES = {f'03g_{name}.csv' for name in CSV_OUTPUTS} | {
    '03g_summary.json', '03g_privacy_qc.json', '03g_RESULTADOS.md', '03g_REVIEW_PACKET.md', '03g_failure.json'}
REVIEWS = {
    'lineage_review_status': 'APPROVED', 'clinical_review_status': 'APPROVED',
    'temporal_baseline_status': 'VERIFIED', 'origin_review_status': 'APPROVED',
    'proxy_review_status': 'APPROVED', 'discovery_exclusion_review_status': 'APPROVED',
}
REVIEW_METADATA = ('reviewer', 'source_reference', 'baseline_window')
TOPOLOGIES = {'FROZEN_COMMUNITIES', 'R2_EXPLORATORY', 'R2_EXPLORATORY_UNSTABLE',
              'R2_EXPLORATORY_NO_HARD_ASSIGNMENTS', 'NO_SUPPORTED_HARD_ASSIGNMENTS', 'PROVISIONAL_COMMUNITIES'}
STABLE = {'FROZEN_COMMUNITIES', 'R2_EXPLORATORY'}
PUBLISHED_STATES = TOPOLOGIES | {
    M.DIRECT, M.PROXY, M.INDEPENDENT, M.UNRESOLVED, M.ESTIMATED, UNAVAILABLE, SUPPRESSED,
    'NOT_ESTIMABLE', 'INSUFFICIENT_SUPPORT', 'ALTERNATIVE_RF_ENCODING_DESCRIPTIVE_ONLY',
    'MODEL_DIAGNOSTIC_FAILURE', 'REPRESENTATION_UNAVAILABLE', 'DESCRIPTIVE',
}
DOCUMENTARY_PUBLISHED_REASONS = {
    'INDEPENDENCE_REVIEW_PENDING', 'TOPOLOGY_NOT_STABLE', 'REPRESENTATION_UNAVAILABLE', UNAVAILABLE,
    'FROZEN_BASELINE_UNAVAILABLE', 'FROZEN_SOURCE_OR_DERIVATION_DEFINES_MAPPER',
    'DOCUMENTED_LINEAGE_OR_CLINICAL_PROXY', 'FROZEN_DEFINING_FEATURE_LINEAGE_INCOMPLETE',
    'OUTCOME_CONSTRUCT_LINEAGE_MISSING_OR_AMBIGUOUS', 'FROZEN_FEATURE_REGISTRY_ROLE_AND_PROXY_CHECKS',
    'DEFINING_SOURCE_REGISTRY_LINEAGE_MISSING', 'OUTCOME_ROLE_UNAVAILABLE',
}
PUBLISHED_REASONS = PUBLISHED_STATES | DOCUMENTARY_PUBLISHED_REASONS | {
    'NO_VALID_MEASUREMENTS_IN_SCOPE', 'RF_NUMERIC_MODEL_ESTIMATED', 'COVERAGE_DISCLOSURE_RESTRICTION',
    'INSUFFICIENT_COMPLETE_CASE_COMMUNITY_SUPPORT', 'INSUFFICIENT_BINARY_COMMUNITY_SUPPORT',
    'INSUFFICIENT_RESIDUAL_DF', 'INSUFFICIENT_COVARIATE_COMPLETE_PATIENTS', 'INSUFFICIENT_COMMUNITIES',
    'LOW_COMMUNITY_SUPPORT', 'DEGENERATE_OUTCOME', 'RANK_DEFICIENT_DESIGN',
    'INSUFFICIENT_EVENTS_PER_PARAMETER', 'SEPARATION', 'UNKNOWN_BLOCKED', 'UNUSABLE_EXPLORATORY_FIT',
    'UNUSABLE_LEVERAGE', 'UNUSABLE_EFFECTS', 'UNUSABLE_COVARIANCE', 'MODEL_NONCONVERGENCE',
    'SEPARATION_OR_EXTREME_FITTED_PROBABILITIES',
}
SOURCE_ROLES = ('step01', 'step10', 'context', 'eda_long', *E.cal.BTRIS_ROLES)
REASONS = {
    'ANALYTE_MAPPING_UNVERIFIED': 'NO_RECOGNIZED_OUTCOME_MAPPING',
    S.EXACT: 'VERIFIED_PATIENT_AND_EPISODE_LINK',
    S.PATIENT_ONLY: 'VERIFIED_PATIENT_LINK_EPISODE_UNKNOWN',
    **dict.fromkeys(('PATIENT_ID_AMBIGUOUS', 'CROSS_PATIENT_CONFLICT', 'AMBIGUOUS_IDENTITY',
                     'PATIENT_NOT_RECONCILED_WITH_FROZEN_BASELINE', 'SOURCE_EPISODE_NOT_VERIFIED'),
                    'PATIENT_OR_EPISODE_LINKAGE_UNRESOLVED'),
    **dict.fromkeys((*S.DATE_FAILURES, S.EDA_ONLY), 'COLLECTION_DATE_UNVERIFIED'),
    **dict.fromkeys(('CENSORED_RESULT', 'ADMINISTRATIVE_RESULT', 'NOT_EXACT_NUMERIC_RESULT',
                     'UNKNOWN_RESULT_TOKEN', 'INVALID_RESULT'), 'INVALID_OR_CENSORED_RESULT'),
    **dict.fromkeys(('ORIGINAL_TEST_DEFINITION_UNAVAILABLE', 'NUMERIC_UNIT_UNAVAILABLE'),
                    'ASSAY_DEFINITION_UNAVAILABLE_OR_INCOMPARABLE'),
    **dict.fromkeys(('IDENTICAL_DUPLICATE', 'AMBIGUOUS_EARLIEST_DAY', 'LATER_ELIGIBLE_DAY',
                     'CONSISTENT_EARLIEST_DAY_REPEAT'), 'FIRST_COMPARABLE_MEASUREMENT_SELECTION'),
}
COMPARABLE = {'', 'IDENTICAL_DUPLICATE', 'AMBIGUOUS_EARLIEST_DAY', 'LATER_ELIGIBLE_DAY',
              'CONSISTENT_EARLIEST_DAY_REPEAT'}
STAGE_REASONS = {
    'source_rows': 'NO_SOURCE_ROWS',
    'outcome_candidates': 'NO_RECOGNIZED_OUTCOME_MAPPING',
    'recognized_mapping': 'NO_RECOGNIZED_OUTCOME_MAPPING',
    'verified_identity': 'PATIENT_OR_EPISODE_LINKAGE_UNRESOLVED',
    'verified_collection': 'COLLECTION_DATE_UNVERIFIED',
    'primary_window': 'OUTSIDE_0_TO_6M',
    'valid_result': 'INVALID_OR_CENSORED_RESULT',
    'comparable_assay': 'ASSAY_DEFINITION_UNAVAILABLE_OR_INCOMPARABLE',
    'first_comparable': 'NO_ELIGIBLE_PRIMARY_DEFINITION',
    'primary_definition': 'NO_ELIGIBLE_PRIMARY_DEFINITION',
}


class AuditError(ValueError):
    """A fixed diagnostic safe to expose on the command line."""


def _require(condition, code):
    if not condition:
        raise AuditError(code)


def _artifacts(directory, names):
    return {role: safe_artifact(directory, name) for role, name in names.items()}


def _table(path, fields):
    return pd.read_csv(path, usecols=fields, keep_default_na=False)


def _window(metadata):
    window = baseline_policy()
    _require(metadata['window_policy_version'] == window.policy_version
             and metadata['window_policy_sha256'] == window.sha256, 'AUDIT_WINDOW_POLICY_MISMATCH')


def _published(files, graph03, fingerprints, reps, outcomes, manifest, strict, cell):
    summary = json.loads(files['summary'].read_text())
    qc = json.loads(files['window_qc'].read_text())
    _window(summary)
    _window(qc)
    _require(summary.get('mode') == E.MODE
             and summary.get('privacy_class') == 'INTERNAL_RESTRICTED'
             and summary.get('public_export') is False
             and summary.get('graph02_release_sha256') == fingerprints['release_manifest']
             and qc.get('graph02_sha256_before') == qc.get('graph02_sha256_after') == fingerprints
             and qc.get('graph02_frozen_unchanged') is True,
             'AUDIT_GRAPH03E_ANCESTRY_MISMATCH')
    _require(summary.get('graph03_input') in {None, graph03.name}
             and (not strict or summary.get('graph03_input') == graph03.name),
             'AUDIT_GRAPH03E_CHARACTERIZATION_MISMATCH')
    _require(summary.get('n_frozen_cohort') in {None, manifest['n_patients']},
             'AUDIT_GRAPH03E_COHORT_MISMATCH')
    _require(summary.get('status') in {'COMPLETE_DESCRIPTIVE_NO_ESTIMABLE_MODELS',
                                       'PARTIAL_WITH_DATA_ISSUES', 'COMPLETE_WITH_EXPLORATORY_MODELS'},
             'AUDIT_GRAPH03E_INCOMPLETE')
    data = {}
    window_fields = ['window_policy_version', 'window_policy_sha256']
    columns = {
        'coverage': ['stratum', 'provenance', 'definition_role', 'definition_id', 'status',
                     'topology_status', 'n_measured', 'n_total'],
        'models': ['stratum', 'provenance', 'status', 'not_estimable_reason'],
        'independence': ['classification', 'reason'],
    }
    for table, fields in columns.items():
        frame = _table(files[table], PAIRS + fields + window_fields)
        _require(frame.window_policy_version.eq(baseline_policy().policy_version).all()
                 and frame.window_policy_sha256.eq(baseline_policy().sha256).all(),
                 'AUDIT_WINDOW_POLICY_MISMATCH')
        if table == 'coverage':
            scope_keys = PAIRS + ['stratum', 'provenance', 'definition_role']
            primary_rows = frame.loc[frame.definition_role.eq('PRIMARY')]
            _require(not primary_rows.duplicated(scope_keys).any()
                     and not frame.duplicated(scope_keys + ['definition_id']).any(),
                     'AUDIT_DUPLICATE_PUBLISHED_PAIR')
            for row in frame.to_dict('records'):
                _coverage_state(row, cell)
                if row['status'] != 'DESCRIPTIVE':
                    _require(row['n_total'] == '', 'AUDIT_INVALID_PUBLISHED_COVERAGE')
                    continue
                _require(_valid_count(row['n_total'], cell)
                         and 0 <= float(row['n_measured']) <= float(row['n_total'])
                         and not R._small(float(row['n_total']) - float(row['n_measured']), cell),
                         'AUDIT_INVALID_PUBLISHED_COVERAGE')
            data['coverage_scopes'] = frame.to_dict('records')
        if table != 'independence':
            scope = frame.stratum.eq(S.PRIMARY) & frame.provenance.eq(S.ORIGINAL)
            if table == 'coverage':
                scope &= frame.definition_role.eq('PRIMARY')
            frame = frame.loc[scope]
        _require(not frame.duplicated(PAIRS).any(), 'AUDIT_DUPLICATE_PUBLISHED_PAIR')
        _require(set(zip(frame.representation_id, frame.outcome)) == {
            (rep, outcome) for rep in reps for outcome in outcomes}, 'AUDIT_INCOMPLETE_PUBLISHED_PAIRS')
        data[table] = frame.set_index(PAIRS).to_dict('index')
    return data


def _read_graph03e(directory, graph03, frozen, reps, outcomes, manifest, strict, cell):
    _inputs_exist(directory, GRAPH03E.values(), 'GRAPH03E')
    files = _artifacts(directory, GRAPH03E)
    published = _published(files, graph03, frozen, reps, outcomes, manifest, strict, cell)
    published['flow'], published['flow_counts'] = _published_funnel(files['flow'], outcomes, cell)
    published['source'] = _published_source(files, outcomes, cell)
    return files, published


def _unavailable_graph03e(reps, outcomes, state):
    pairs = [(rep, source) for rep in reps for source in outcomes]
    return dict(
        coverage={pair: dict(status=UNAVAILABLE, topology_status='', n_measured='') for pair in pairs},
        models={pair: dict(status=state, not_estimable_reason=state) for pair in pairs},
        independence={pair: dict(classification=state, reason=state) for pair in pairs},
        flow={source: ([], state) for source in outcomes}, source={source: [] for source in outcomes},
        flow_counts={}, coverage_scopes=[], state=state,
    )


def _published_funnel(path, outcomes, cell):
    # Graph03e already suppresses the removed cells and cohort complements.
    # Project those authorized stages to states without copying any counts.
    stages = {
        'n_source_rows': 'NO_RECOGNIZED_OUTCOME_MAPPING',
        'n_patients_with_any_candidate': 'PATIENT_OR_EPISODE_LINKAGE_UNRESOLVED',
        'n_temporally_eligible': 'NO_LINKED_CANDIDATES_IN_PRIMARY_WINDOW',
        'n_date_verified': 'COLLECTION_DATE_UNVERIFIED',
        'n_assay_comparable': 'NO_ELIGIBLE_COMPARABLE_RESULT',
        'n_earliest_unambiguous': 'NO_ELIGIBLE_PRIMARY_DEFINITION',
        **dict.fromkeys(('n_pre', 'n_day0', 'n_day1_30', 'n_day31_90', 'n_day91_to_6m', 'n_after_6m',
                         'n_unknown'), ''),
    }
    fields = PAIRS + ['stratum', 'stage', 'status', 'n_remaining', 'window_policy_version', 'window_policy_sha256']
    frame = _table(path, fields)
    frame = frame.loc[frame.representation_id.eq('') & frame.stratum.eq('ALL') & frame.stage.isin(stages)]
    _require(not frame.duplicated(['outcome', 'stage']).any(), 'AUDIT_DUPLICATE_PUBLISHED_STAGE')
    _require(frame.window_policy_version.eq(baseline_policy().policy_version).all()
             and frame.window_policy_sha256.eq(baseline_policy().sha256).all(),
             'AUDIT_WINDOW_POLICY_MISMATCH')
    projected, comparable = {}, {}
    for outcome in outcomes:
        evidence = frame.loc[frame.outcome.eq(outcome)].set_index('stage').to_dict('index')
        output, stop = [], UNKNOWN
        preceding_present = True
        for stage, reason in stages.items():
            row = evidence.get(stage, {})
            state = row.get('status', UNKNOWN)
            if state == 'DESCRIPTIVE':
                count = float(row['n_remaining'])
                _require(_valid_count(count, cell), 'AUDIT_INVALID_PUBLISHED_FLOW')
                comparable[(outcome, stage)] = int(count)
                state = 'PRESENT' if count else 'EMPTY'
            elif state == SUPPRESSED:
                _require(row['n_remaining'] == '', 'AUDIT_INVALID_PUBLISHED_FLOW')
            else:
                state = UNAVAILABLE if state == UNAVAILABLE else UNKNOWN
            if reason and preceding_present and state == 'EMPTY':
                stop = reason
            if reason:
                preceding_present &= state == 'PRESENT'
            output.append(dict(outcome=outcome, stage=stage, status=state,
                               reason=reason if state == 'EMPTY' else '', provenance='PUBLISHED_03E_COHORT_FLOW'))
        projected[outcome] = (output, stop)
    return projected, comparable


def _graph03_metadata(files, manifest, fingerprints, strict):
    summary, release_qc, window_qc = (json.loads(files[name].read_text())
                                      for name in ('summary', 'release_qc', 'window_qc'))
    _window(summary)
    _window(window_qc)
    _require(summary.get('schema_version') == RESULTS_SCHEMA
             and summary.get('execution_status') in {'COMPLETE', 'COMPLETE_WITH_LIMITATIONS'}
             and summary.get('n_patients') == manifest['n_patients']
             and summary.get('graph02_release_sha256') == fingerprints['release_manifest']
             and window_qc.get('graph02_sha256_after') == fingerprints
             and release_qc.get('status') == PASS
             and release_qc.get('eligible_representation_ids') == manifest['eligible_representation_ids'],
             'AUDIT_GRAPH03_ANCESTRY_MISMATCH')
    pins = summary.get('sha256_by_artifact', {})
    for role, path in files.items():
        if role == 'summary':
            continue
        if strict or path.name in pins:
            _require(pins.get(path.name) == digest(path), 'AUDIT_GRAPH03_ARTIFACT_PIN_MISMATCH')
    if 'graph02_sha256_before' in window_qc:
        _require(window_qc['graph02_sha256_before'] == fingerprints, 'AUDIT_GRAPH03_ANCESTRY_MISMATCH')
    fields = ['representation_id', 'source_column', 'tier', 'direct_in_discovery', 'conflicting_defining_sources']
    roles = _table(files['roles'], fields)
    _require(not roles.duplicated(fields[:2]).any(), 'AUDIT_DUPLICATE_CHARACTERIZATION_ROLE')
    _require(roles.direct_in_discovery.isin([True, False]).all(), 'AUDIT_INVALID_CHARACTERIZATION_ROLE')
    return roles


def _validate_reviews(validation):
    for section, key in [('characterization', 'independence_reviews'), ('n3_validation', 'prespecified_outcomes')]:
        records = validation.get(section, {}).get(key, [])
        _require(isinstance(records, list), 'AUDIT_INVALID_REVIEW_CONTRACT')
        keys = []
        for row in records:
            _require(isinstance(row, dict) and all(isinstance(row.get(field), str) and row[field]
                     for field in ('representation_id', 'source_column')), 'AUDIT_INVALID_REVIEW_CONTRACT')
            keys.append((row['representation_id'], row['source_column']) if section == 'characterization'
                        else (row['representation_id'], row['source_column'], row.get('window_policy_version')))
        _require(len(keys) == len(set(keys)), 'AUDIT_DUPLICATE_REVIEW_CONTRACT')


def _metadata_tokens(value):
    # Expose code column names only. Free documentary references may contain private paths.
    return '|'.join(token for token in value.split('|')
                    if re.fullmatch(r'[a-z][a-z0-9_]*__[a-z0-9_]+', token))


def _reason(value, allowed=PUBLISHED_REASONS):
    return '|'.join(part for part in value.split('|') if part in allowed) or UNKNOWN


def _independence(rep, source, roles, features, registry, dependencies, validation, policy, available):
    role = roles.get((rep, source), {})
    audit = (M._independence(rep, source, roles, features, registry, dependencies, validation, policy) if available
             else dict(classification=M.UNRESOLVED, reason='REPRESENTATION_UNAVAILABLE',
                       defining_sources='', blocking_sources=''))
    version = baseline_policy().policy_version
    review = _review(validation, rep, source)
    prespecified = E._prespecified(validation, rep, source, version)
    approved = E._independence_approved(validation, rep, source, review, version)
    checks = {key: PASS if review.get(key) == expected else 'MISSING' if not review.get(key) else 'NOT_VERIFIED'
              for key, expected in REVIEWS.items()}
    checks.update({key + '_status': 'RECORDED' if review.get(key) else 'MISSING' for key in REVIEW_METADATA})
    checks['review_window_policy_status'] = (PASS if review.get('window_policy_version') == version
                                            else 'MISMATCH' if review.get('window_policy_version') else 'MISSING')
    missing = [key for key, state in checks.items() if state not in {PASS, 'RECORDED'}]
    if not review:
        missing.insert(0, 'independence_review_record')
    if not prespecified:
        missing.append('prespecified_outcome_for_representation_and_policy')
    structural = audit['classification']
    interpreted = {M.UNRESOLVED: 'LINEAGE_UNRESOLVED', M.INDEPENDENT: (
        'STRUCTURALLY_INDEPENDENT_REVIEW_RECORDED' if approved else 'STRUCTURALLY_INDEPENDENT_REVIEW_PENDING')}.get(structural, structural)
    paths = _paths(dependencies, source)
    included = features.loc[features.representation_id.eq(rep) & features.included.eq(True)]
    # Encoded nominal feature labels may embed rare clinical values. Project
    # source names and literal source-preserving feature names only.
    features_named = included.loc[included.feature.eq(included.source_column), 'feature']
    named_features = _metadata_tokens('|'.join(sorted(set(features_named))))
    types = sorted({edge['relationship_type'] for target in audit['blocking_sources'].split('|')
                    for edge in paths.get(target, [])})
    fields = dict(structural_independence_classification=interpreted, structural_reason=_reason(audit['reason']),
                  human_review_status=PASS if independence_review_approved(review) and
                  checks['review_window_policy_status'] == PASS else 'PENDING',
                  prespecified_status=PASS if prespecified else 'PENDING',
                  approval_gate_status=PASS if approved else 'INDEPENDENCE_REVIEW_PENDING',
                  missing_review_requirements='|'.join(missing))
    lineage = dict(fields, **checks, frozen_role=role.get('tier', '') if role.get('tier') in {
        'N1_DEFINING', 'N2_RELATED', 'N3_CANDIDATE', 'N3_REVIEWED', 'UNAVAILABLE', 'PENDING_REVIEW'} else UNKNOWN,
        defining_sources=_metadata_tokens(audit['defining_sources']),
        blocking_sources=_metadata_tokens(audit['blocking_sources']),
        defining_features=named_features, relationship_types='|'.join(types),
        feature_label_disclosure=('SOURCE_PRESERVING_NAMES_ONLY' if available else UNAVAILABLE))
    fields.update(structural_class=structural,
                  review_status='CONTRACT_RECORD_PRESENT' if approved else 'PENDING',
                  interpretation='EXPLORATORY_PRESPECIFICATION_RECORDED' if prespecified else 'POST_HOC_EXPLORATORY')
    lineage.update(fields)
    return fields, lineage, structural


def _funnel(result, eda, source, cell, *, audit=None):
    rows = result.inventory
    mapped = rows.loc[rows.mapping_candidate]
    linked = mapped.loc[mapped.identity_status.isin(S.LINKED)]
    verified = linked.loc[linked.provenance.eq(S.ORIGINAL)]
    window = verified.loc[verified.stratum.eq(S.PRIMARY)]
    valid = window.loc[window.first_reason.isin(COMPARABLE | {
        'ORIGINAL_TEST_DEFINITION_UNAVAILABLE', 'NUMERIC_UNIT_UNAVAILABLE'})]
    comparable = window.loc[window.first_reason.isin(COMPARABLE)]
    first = result.selected.loc[result.selected.stratum.eq(S.PRIMARY) & result.selected.provenance.eq(S.ORIGINAL)]
    selected = S.primary_measurements(result)
    stages = dict(zip(STAGE_REASONS, (eda, rows, mapped, linked, verified, window, valid, comparable, first, selected)))
    # Use nested patient sets for the chain. A count of repeated rows never
    # certifies the disclosure of a category involving fewer distinct patients.
    populations = [set(stages['source_rows']._patient_match.dropna())]
    populations.extend(set(frame._patient_match.dropna()) for key, frame in stages.items()
                       if key not in {'source_rows', 'first_comparable', 'primary_definition'})
    populations.extend(set(comparable.loc[comparable.patient_id.isin(frame.patient_id), '_patient_match'].dropna())
                       for frame in (first, selected))
    if audit is not None:
        audit['stages'] = {name: len(patients) for name, patients in zip(stages, populations)}
        audit['by_collection_bin'] = linked.groupby('window_bin')._patient_match.nunique().to_dict()
        audit['by_exclusion_stage'] = rows.groupby(['stratum', 'provenance', 'first_reason'],
                                                   dropna=False)._patient_match.nunique().to_dict()
    safe = not any(frame._patient_match.isna().any() for key, frame in stages.items()
                   if key not in {'first_comparable', 'primary_definition'})
    safe &= not any(R._small(len(group), cell) for group in populations)
    safe &= not any(R._small(len(left - right), cell) for left, right in zip(populations, populations[1:]))
    output, first_empty = [], ''
    for stage, frame in stages.items():
        reason = STAGE_REASONS[stage]
        first_loss = frame.empty and not first_empty
        if first_loss:
            first_empty = reason
        output.append(dict(outcome=source, stage=stage, provenance='SECURE_SOURCE_RECOMPUTE',
                           status=SUPPRESSED if not safe else 'EMPTY' if frame.empty else 'PRESENT',
                           reason=reason if safe and first_loss else ''))
    # Apply existing partitions before reporting controlled categories. Withhold
    # the entire related panel when any exclusion or date/identity cell is protected.
    bins = R._partition(linked.groupby('window_bin')._patient_match.nunique().to_dict(), cell)
    bins_per_patient = linked.groupby('_patient_match').window_bin.nunique()
    if R._small(int(bins_per_patient.gt(1).sum()), cell):
        bins = None
    audits = R._source_audit(rows, source, cell) + R._exclusions(rows, {'outcome': source}, cell)
    if safe and bins is not None and not any(row['status'] == SUPPRESSED for row in audits):
        for bin_name in DAY_BINS:
            output.append(dict(outcome=source, stage='collection_window_bin', original_reason=bin_name,
                               status='PRESENT' if bins.get(bin_name) else 'EMPTY',
                               reason='OUTSIDE_0_TO_6M' if bin_name in (DAY_BINS[0], DAY_BINS[5])
                               and bins.get(bin_name) else '', provenance='SECURE_SOURCE_RECOMPUTE'))
        for audit in audits:
            original = audit.get('reason') or audit.get('category')
            if original in REASONS:
                output.append(dict(outcome=source, stage='source_disposition', original_reason=original,
                                   reason=REASONS[original], status='PRESENT', provenance='SECURE_SOURCE_RECOMPUTE'))
    else:
        output.append(dict(outcome=source, stage='source_disposition', status=SUPPRESSED,
                           provenance='SECURE_SOURCE_RECOMPUTE'))
    state = SUPPRESSED if not safe else first_empty or 'PRIMARY_DEFINITION_SELECTED'
    return output, state


def _valid_count(count, cell):
    value = float(count)
    return value.is_integer() and (value == 0 or value >= cell)


def _coverage_state(row, cell):
    status = row['status']
    _require(status in {'DESCRIPTIVE', SUPPRESSED, UNAVAILABLE, 'REPRESENTATION_UNAVAILABLE'},
             'AUDIT_INVALID_PUBLISHED_COVERAGE')
    count = row['n_measured']
    if status == 'DESCRIPTIVE':
        count = float(count)
        _require(_valid_count(count, cell), 'AUDIT_INVALID_PUBLISHED_COVERAGE')
        return 'ZERO_ELIGIBLE_MEASUREMENTS' if count == 0 else 'MEASUREMENTS_OBSERVED'
    _require(count == '', 'AUDIT_INVALID_PUBLISHED_COVERAGE')
    return status


def _support(frame, numeric, policy, cell, protected, *, private=None):
    audit = {}
    _, prepared = M._model_support(frame, numeric, policy, cell, audit=audit)
    for _, group in frame.assign(complete=audit['complete'], common=audit['common']).groupby('community_id'):
        observed = group.value.dropna()
        complete = group.loc[group.complete, 'value']
        counts = [len(group), len(observed), len(group) - len(observed),
                  len(complete), len(observed) - len(complete)]
        if not numeric:
            for values in (observed, complete):
                counts.extend((int(values.eq(1).sum()), int(values.eq(0).sum())))
        protected |= any(R._small(count, cell) for count in counts)
    minimum_group = policy['minimum_patients_per_group']
    supported_groups = int(audit['community_counts'].ge(minimum_group).sum())
    common_n = int(audit['common'].sum())
    details = {}
    for model in ('A', 'B'):
        design = audit['designs'][model]
        reason = '|'.join(sorted(audit['reasons_by_model'][model]))
        evaluated = design['n_parameters'] is not None
        if private is not None:
            private[model] = dict(support_computed=True, support_feasible=not reason,
                                  support_disclosable=not protected, reasons=audit['reasons_by_model'][model],
                                  valid_covariates=audit['covariates'].notna().sum().to_dict(),
                                  observed_covariates=frame[['demo__age_at_baseline', 'demo__sex',
                                                            'clinical_baseline_date']].notna().sum().to_dict(),
                                  **design)
        details[model] = dict(
            status=SUPPRESSED if protected else 'BLOCKED' if reason else PASS,
            reason='' if protected else reason,
            complete_case_status=SUPPRESSED if protected else
            'MEETS_MINIMUM' if common_n >= policy['minimum_complete_cases'] else 'BELOW_MINIMUM',
            community_balance_status=SUPPRESSED if protected else
            'MEETS_MINIMUM_GROUP_COUNT' if supported_groups >= policy['minimum_groups'] else 'BELOW_MINIMUM_GROUP_COUNT',
            binary_community_support_status=SUPPRESSED if protected else 'NOT_APPLICABLE' if numeric else
            'BELOW_MINIMUM' if audit['binary_community_blocked'] else PASS if evaluated else 'NOT_EVALUATED',
            event_parameter_status=SUPPRESSED if protected else 'NOT_APPLICABLE' if numeric else
            'BELOW_MINIMUM' if 'INSUFFICIENT_EVENTS_PER_PARAMETER' in reason else PASS if evaluated else 'NOT_EVALUATED',
            residual_df_status=SUPPRESSED if protected else
            'BELOW_MINIMUM' if 'INSUFFICIENT_RESIDUAL_DF' in reason else PASS if evaluated else 'NOT_EVALUATED',
            parameter_rank_status=SUPPRESSED if protected else
            'RANK_DEFICIENT' if 'RANK_DEFICIENT_DESIGN' in reason else 'FULL_RANK' if evaluated else 'NOT_EVALUATED',
            outcome_variation_status=(SUPPRESSED if protected else
                                      'VARIES' if frame.loc[audit['common'], 'value'].nunique() > 1
                                      else 'INSUFFICIENT_VARIATION_OR_NO_CASES'),
            diagnostic_provenance='SECURE_03E_COMMON_MASK',
        )
    return details, prepared is not None and not protected


def _not_computed(status, reason, provenance):
    return {model: dict(status=status, reason=reason, diagnostic_provenance=provenance) for model in ('A', 'B')}


def _published_source(files, outcomes, cell):
    """Project only categories already visible in the Graph03e source panels."""
    evidence = {outcome: [] for outcome in outcomes}
    window_fields = ['window_policy_version', 'window_policy_sha256']
    for name, fields, code, count in (
        ('exclusions', ['stratum', 'provenance', 'reason', 'n_patients'], 'reason', 'n_patients'),
        ('linkage', ['dimension', 'category', 'n_count'], 'category', 'n_count'),
    ):
        frame = _table(files[name], ['outcome', 'status', *fields, *window_fields])
        _require(frame.window_policy_version.eq(baseline_policy().policy_version).all()
                 and frame.window_policy_sha256.eq(baseline_policy().sha256).all(),
                 'AUDIT_WINDOW_POLICY_MISMATCH')
        for row in frame.loc[frame.outcome.isin(outcomes)].to_dict('records'):
            token = row[code]
            if row['status'] != 'DESCRIPTIVE' or token not in REASONS:
                continue
            _require(_valid_count(row[count], cell), 'AUDIT_INVALID_PUBLISHED_SOURCE_PANEL')
            scope = row.get('stratum', 'ALL')
            tier = row.get('provenance', 'ALL')
            if scope not in {'ALL', *S.STRATA, S.UNRESOLVED} or tier not in {
                    'ALL', S.ORIGINAL, S.EDA_ONLY, 'UNRESOLVED_DATE_PROVENANCE'}:
                continue
            evidence[row['outcome']].append(dict(outcome=row['outcome'], check=name, scope=scope,
                                                provenance=tier, status='PUBLISHED_03E_CATEGORY',
                                                category=token, reason=REASONS[token]))
    return evidence


def _source_states(data):
    tables, records, issues, *_ = data
    unavailable = {row.get('artifact') for row in issues if row.get('status') == UNAVAILABLE}
    states = {role: 'AVAILABLE' if role in tables else UNAVAILABLE for role in SOURCE_ROLES[:4]}
    for role in E.cal.BTRIS_ROLES:
        state = 'AVAILABLE'
        if role in unavailable or not any(schema['source_role'] == role for schema in records.schemas):
            state = UNAVAILABLE
        states[role] = state
    return states


def _audit_measurements(candidates, source, settings, eda, baseline, step01, published):
    private = {}
    result = S._outcome(candidates, source, settings, audit=private)
    selection = S.primary_measurements(result)
    _require(not selection.patient_id.duplicated().any()
             and set(selection.patient_id) <= set(baseline.patient_id)
             and selection.window_bin.isin(DAY_BINS[1:5]).all()
             and selection.baseline_offset_days.ge(0).all(),
             'AUDIT_INVALID_PRIMARY_SELECTION')
    _funnel(result, eda, source, settings['minimum_disclosure_cell'], audit=private)
    rows = result.inventory
    unknown = rows.loc[rows.first_reason.eq('UNKNOWN_RESULT_TOKEN')]
    invalid_text = S._clean(unknown.result_text).fillna(S._clean(unknown.result_raw))
    if source == 'lab__ana_hep2_status__text':
        # Classifies only the shape of an unrecognized token for source review.
        # It does not extend the clinical crosswalk.
        titres = invalid_text.str.contains(r'\b\d+\s*:\s*\d+\b', regex=True, na=False)
        patterns = invalid_text.str.contains(
            r'\b(?:pattern|speckled|homogeneous|nucleolar|centromere|cytoplasmic)\b', case=False, na=False)
        private['unknown_result_kinds'] = {
            'TITER_OR_PATTERN': unknown.loc[titres | patterns, '_patient_match'].nunique(),
            'UNRECOGNIZED_TEXT': unknown.loc[~(titres | patterns) & invalid_text.notna(), '_patient_match'].nunique(),
            'MISSING_RESULT': unknown.loc[invalid_text.isna(), '_patient_match'].nunique(),
        }
    if source == S.RF_TEXT:
        private['rf_result_kinds'] = {
            'NO_TEXT_OR_RAW_RESULT': unknown.loc[invalid_text.isna(), '_patient_match'].nunique(),
            'UNMAPPED_RESULT_TOKEN': unknown.loc[invalid_text.notna(), '_patient_match'].nunique(),
        }
    private['assay_support'] = result.selected.groupby(
        ['stratum', 'provenance', 'definition_id']).patient_id.nunique().to_dict()
    private['duplicate_rows'] = len(result.duplicates)
    private['sensitivity_selected_patients'] = {scope: selected.patient_id.nunique()
                                               for scope, selected in result.sensitivities.items()}
    private['eda_only_selected_patients'] = result.selected.loc[
        result.selected.provenance.eq(S.EDA_ONLY), 'patient_id'].nunique()

    fresh_flow = dict(n_source_rows=len(rows), **E._eligibility_counts(rows, result))
    linked = rows.loc[rows.identity_status.isin(S.LINKED)]
    bin_labels = ('n_pre', 'n_day0', 'n_day1_30', 'n_day31_90', 'n_day91_to_6m', 'n_after_6m', 'n_unknown')
    fresh_flow.update({label: linked.loc[linked.window_bin.eq(bin_name), 'patient_id'].nunique()
                       for bin_name, label in zip(DAY_BINS, bin_labels)})
    drift = any(fresh_flow[stage] != previous for (outcome, stage), previous in published['flow_counts'].items()
                if source == outcome)
    # Compare only with the official baseline episode's selection. A shared
    # collection day cannot prove exact specimen reuse across stages.
    if step01 is not None:
        keys = ['patient_id', 'clinical_episode_id']
        prior = step01.loc[step01.lab_id.eq(S.LAB_SOURCES[source]), keys + ['selected_lab_date']].copy()
        prior[keys] = prior[keys].astype('string')
        anchors = baseline[keys].astype('string').merge(prior, on=keys, how='inner')
        anchors['collection_calendar_day'] = S._collection_days(anchors.selected_lab_date)
        same_day = selection[['patient_id', 'collection_calendar_day']].merge(
            anchors[['patient_id', 'collection_calendar_day']].dropna().drop_duplicates(),
            on=['patient_id', 'collection_calendar_day'])
        private['same_day_baseline_patients'] = same_day.patient_id.nunique()
    return result, selection, private, drift


def _reconcile_coverage(scopes, result, members):
    """Reconcile every published primary/sensitivity/assay aggregate with visible coverage."""
    slices = {(stratum, provenance, role, definition if role == 'ALTERNATIVE' else ''): (definition, rows)
              for stratum, provenance, role, definition, rows in E._lab_slices(result)}
    for row in scopes:
        if row['status'] != 'DESCRIPTIVE':
            continue
        key = (row['stratum'], row['provenance'], row['definition_role'],
               row['definition_id'] if row['definition_role'] == 'ALTERNATIVE' else '')
        definition, selected = slices.get(key, (None, None))
        ids = members[row['representation_id']].patient_id
        if (selected is None or float(row['n_total']) != len(ids)
                or (row['definition_id'] and definition != row['definition_id'])
                or float(row['n_measured']) != selected.patient_id.isin(ids).sum()):
            return False
    return True


def _topology_scope(rep, topology):
    if topology in STABLE:
        return {'R0': 'PRIMARY_PREFLIGHT', 'R2': 'EXPLORATORY_PREFLIGHT'}.get(rep, 'SENSITIVITY_PREFLIGHT')
    if topology in TOPOLOGIES | {'NO_STABLE_STRUCTURE', 'REPRESENTATION_UNAVAILABLE'}:
        return 'DESCRIPTIVE_ONLY'
    return UNKNOWN


def _support_public(source_status, measured, mode):
    # The exploratory contract does not authorize a new support bit or any of
    # its components. Even a pass/fail can complete a published small partition.
    if source_status in {UNAVAILABLE, 'INTEGRITY_BLOCKED', 'REPRESENTATION_UNAVAILABLE'}:
        return _not_computed(source_status, '', 'GRAPH03G_INPUT_CONTRACT')
    if measured == 'ZERO_ELIGIBLE_MEASUREMENTS':
        return _not_computed('BLOCKED', 'NO_VALID_MEASUREMENTS_IN_SCOPE', 'PUBLISHED_03E_COVERAGE')
    if mode == 'secure-recompute' or measured == SUPPRESSED:
        return _not_computed(SUPPRESSED, '', 'GRAPH03G_DISCLOSURE_POLICY')
    return _not_computed(UNKNOWN, '', 'PUBLISHED_03E_AGGREGATES')


def _build(reps, outcomes, settings, validation, manifest, features, registry, roles, published,
           data=None, controlled_reviews=None):
    dependencies = _dependencies(features, registry, validation)
    controlled_reviews = controlled_reviews or {}
    role_map = {(row['representation_id'], row['source_column']): row for row in roles.to_dict('records')}
    cell = settings['minimum_disclosure_cell']
    matrices, lineages, funnel, support, assays = [], [], [], [], []
    eligible = set(manifest['eligible_representation_ids'])
    source_states = _source_states(data) if data else {}
    source_available = {'eda_long', 'step10'} <= {role for role, state in source_states.items() if state == 'AVAILABLE'}
    if source_available:
        tables, records, _, baseline, membership = data
        candidates = S._candidates(tables, baseline, records, settings)
        eda = pd.DataFrame({'_patient_match': S.NORMALIZE(tables['eda_long'].patient_id)})
    outcome_data = {}
    published_block = published.get('state')
    drift = False
    if source_available:
        for source in outcomes:
            result, selection, private_measurement, flow_drift = _audit_measurements(
                candidates, source, settings, eda, baseline, tables.get('step01'), published)
            scopes = [row for row in published['coverage_scopes'] if row['outcome'] == source
                      and row['representation_id'] in eligible]
            # A missing original source can explain any apparent discrepancy.
            # Keep the published panel and record incomplete reconciliation.
            source_drift = not published_block and all(state == 'AVAILABLE' for state in source_states.values()) and (
                flow_drift or not _reconcile_coverage(scopes, result, membership))
            disposition = S._test_disposition(result, baseline.patient_id)
            outcome_data[source] = selection, private_measurement, source_drift, disposition
            drift |= source_drift
    for source in outcomes:
        if not drift:
            assays.extend(published['source'][source])
        if data:
            for role, state in source_states.items():
                assays.append(dict(outcome=source, check='SOURCE_SCHEMA', scope=role, status=state))
            for scope in (S.PRIMARY, *S.SENSITIVITIES, *S.STRATA[1:], S.EDA_ONLY):
                assays.append(dict(outcome=source, check='ASSAY_AND_TIMING', scope=scope,
                                   status=SUPPRESSED if source_available else UNAVAILABLE))
            assays.append(dict(outcome=source, check='BASELINE_REUSE', scope='PATIENT_ANALYTE_COLLECTION_DAY',
                               status=SUPPRESSED if source_available and 'step01' in tables else UNAVAILABLE))
            assays.append(dict(outcome=source, check='SPECIMEN_IDENTITY', scope='EXACT_SAMPLE',
                               status='EXACT_SAMPLE_LINK_NOT_ESTABLISHED'))
            if source in {S.RF_TEXT, 'lab__ana_hep2_status__text'}:
                assays.append(dict(outcome=source, check='UNKNOWN_RESULT_REVIEW', scope=S.PRIMARY,
                                   status=SUPPRESSED if source_available else UNAVAILABLE))
        transitions, source_state = published['flow'][source]
        if drift:
            source_state = 'INTEGRITY_BLOCKED'
        else:
            funnel.extend(transitions)
        if data:
            for stage in STAGE_REASONS:
                funnel.append(dict(outcome=source, stage=stage, status=SUPPRESSED if source_available else UNAVAILABLE,
                                   provenance='GRAPH03G_SECURE_FUNNEL'))

        for rep in reps:
            label = dict(representation_id=rep, outcome=source)
            pair = (rep, source)
            coverage, model, prior = (published[name][pair] for name in ('coverage', 'models', 'independence'))
            prior_measured = _coverage_state(coverage, cell)
            measured = 'INTEGRITY_BLOCKED' if drift and rep in eligible else prior_measured
            fields, lineage, structural = _independence(
                rep, source, role_map, features, registry, dependencies, validation, settings, rep in eligible)
            fields['controlled_review_evidence'] = controlled_reviews.get(pair, 'PENDING')
            lineage.update(fields)
            topology = coverage['topology_status']
            topology = topology if topology in TOPOLOGIES else UNKNOWN
            if rep not in eligible:
                topology = 'REPRESENTATION_UNAVAILABLE'
            elif manifest['release_status'] == 'NO_STABLE_STRUCTURE':
                topology = 'NO_STABLE_STRUCTURE'

            private_support = {}
            if rep not in eligible:
                step = 'REPRESENTATION_UNAVAILABLE'
            elif drift:
                # Do not identify the outcome/pair that changed.
                step = 'INTEGRITY_BLOCKED'
            elif published_block:
                step = published_block
            elif data and not source_available:
                step = UNAVAILABLE
            elif data:
                step = SUPPRESSED
            else:
                step = UNKNOWN
            if source_available and rep in eligible and not outcome_data[source][2]:
                selection, private_measurement, _, disposition = outcome_data[source]
                members = membership[rep]
                _require(topology in {UNKNOWN, 'NO_STABLE_STRUCTURE', E._topology(rep, members)},
                         'AUDIT_TOPOLOGY_CONTRACT_MISMATCH')
                frame = members.merge(selection[['patient_id', 'value']], on='patient_id', how='left',
                                      validate='one_to_one')
                numeric = source == S.RF_VALUE
                profiles = R._profiles(frame, {}, numeric, cell)
                private_coverage = R._coverage(frame, {}, profiles, cell, disposition)
                protected = (prior_measured == SUPPRESSED or private_coverage['status'] == SUPPRESSED
                             or R._small(len(baseline) - len(frame), cell))
                _support(frame, numeric, settings['models'], cell, protected, private=private_support)
                # Reconciliation decisions are kept separate from the public
                # disposition. Never turn a private pass into a public action.
                for record in private_support.values():
                    record['support_disclosable'] = False
                private_measurement[rep] = dict(n_hard_matched=int(frame.value.notna().sum()), models=private_support)

            diagnostics = _support_public(step, measured, 'secure-recompute' if data else 'summaries')
            disclosure = (SUPPRESSED if diagnostics['A']['status'] == SUPPRESSED else 'PUBLISHED_COVERAGE_ONLY'
                          if measured in {'ZERO_ELIGIBLE_MEASUREMENTS', 'MEASUREMENTS_OBSERVED'} else 'NOT_DISCLOSABLE')
            for name, diagnostic in diagnostics.items():
                support.append(dict(label, model=name, **diagnostic))
            funnel.append(dict(label, stage='primary_hard_community_coverage', status=measured,
                               provenance='PUBLISHED_03E_COVERAGE'))
            if data:
                funnel.append(dict(label, stage='complete_cases_in_supported_communities',
                                   status=diagnostics['A']['status'], provenance='GRAPH03G_DISCLOSURE_POLICY'))
            assay, time, hard_scope = (('COMPARABLE_PRIMARY_DEFINITION', 'PRIMARY_WINDOW_VERIFIED',
                                        'HARD_COMMUNITY_MATCH') if measured == 'MEASUREMENTS_OBSERVED'
                                       else (measured, measured, measured) if measured == SUPPRESSED
                                       else (UNKNOWN, UNKNOWN, UNKNOWN))
            blockers = []
            if step in {UNAVAILABLE, 'INTEGRITY_BLOCKED', 'REPRESENTATION_UNAVAILABLE'}:
                blockers.append(step)
            if structural != M.INDEPENDENT:
                blockers.append(fields['structural_reason'])
            if structural == M.INDEPENDENT and fields['approval_gate_status'] != PASS:
                blockers.append(fields['approval_gate_status'])
            if topology not in STABLE:
                blockers.append('TOPOLOGY_NOT_STABLE' if topology != UNKNOWN else 'TOPOLOGY_UNVERIFIED')
            if measured != 'MEASUREMENTS_OBSERVED':
                blockers.append('COVERAGE_SUPPRESSED' if measured == SUPPRESSED else measured)
            if disclosure == SUPPRESSED:
                blockers.append('DISCLOSURE_RESTRICTED')
            if step in {'INTEGRITY_BLOCKED', 'REPRESENTATION_UNAVAILABLE'}:
                decision = step
            elif structural in {M.DIRECT, M.PROXY}:
                decision = 'N1_DESCRIPTIVE' if structural == M.DIRECT else 'N2_DESCRIPTIVE'
            elif structural != M.INDEPENDENT:
                decision = 'LINEAGE_UNRESOLVED'
            elif fields['approval_gate_status'] != PASS:
                decision = 'N3_REVIEW_PENDING'
            else:
                decision = 'N3_REVIEW_RECORDED_READINESS_UNVERIFIED'

            graph03e_reason = _reason(model['not_estimable_reason'])
            graph03e_status = model['status'] if model['status'] in PUBLISHED_STATES else UNKNOWN
            if ((prior_measured == SUPPRESSED or drift)
                    and not set(graph03e_reason.split('|')) <= DOCUMENTARY_PUBLISHED_REASONS):
                graph03e_reason = graph03e_status = SUPPRESSED
            priority = {S.RF_VALUE: 'NUMERIC_FIRST', S.RF_TEXT: 'QUALITATIVE_IF_NUMERIC_NOT_ESTIMABLE'}
            blockers = list(dict.fromkeys(blockers))
            matrices.append(dict(label, **fields, primary_measurement_status=measured, source_funnel_status=source_state,
                                 data_availability_status=step if step == UNAVAILABLE else measured,
                                 assay_comparability_status=assay, time_eligibility_status=time,
                                 community_support_status=hard_scope, topology_status=topology,
                                 topology_use=_topology_scope(rep, topology),
                                 model_A_support_status=diagnostics['A']['status'],
                                 model_B_support_status=diagnostics['B']['status'], disclosure_status=disclosure,
                                 graph03e_reason=graph03e_reason, graph03e_model_status=graph03e_status,
                                 graph03e_independence_status=prior['classification'] if
                                 prior['classification'] in PUBLISHED_STATES else UNKNOWN,
                                 graph03e_independence_reason=_reason(prior['reason'], DOCUMENTARY_PUBLISHED_REASONS),
                                 hypothesis_family=M._family(source), rf_primary_priority=priority.get(source, ''),
                                 primary_blocker=blockers[0] if blockers else '',
                                 additional_blockers='|'.join(blockers[1:]), readiness_status=decision))
            lineages.append(dict(label, **lineage))
    return matrices, lineages, funnel, support, assays, source_states, drift


def _secure_data(root, release, manifest, paths, btris_files, graph03, validation):
    loaded = E._load_graph02(release.parent, validated_release=(manifest, release))
    _require(not loaded['data_issues'], 'AUDIT_FROZEN_MEMBERSHIP_INVALID')
    frozen = loaded['frozen']
    baseline = frozen['baseline_raw']
    features = frozen['feature_sets']
    registry = loaded['registry']
    roles = feature_roles(features, baseline, validation, manifest['eligible_representation_ids'], registry)
    try:
        E._graph03_qc(graph03, loaded, roles)
    except INPUT_ERRORS:
        raise AuditError('AUDIT_GRAPH03_FROZEN_CONTRACT_MISMATCH') from None
    baseline_fields = baseline.reindex(columns=['patient_id', 'clinical_baseline_date',
                                               'demo__age_at_baseline', 'demo__sex'])
    membership = {rep: hard_members(frozen['membership'], rep).merge(
        baseline_fields, on='patient_id', validate='one_to_one') for rep in manifest['eligible_representation_ids']}
    tables, records, issues = E.current_source_inputs(root, paths=paths, btris_files=btris_files)
    if 'step10' in tables:
        keys = ['patient_id', 'clinical_episode_id']
        dates = [*keys, 'clinical_anchor_date', 'clinical_baseline_date']
        step10 = tables['step10']
        anchored = step10.loc[step10.is_clinical_baseline.eq(True), dates].astype('string')
        frozen_anchor = baseline[dates].astype('string')
        joined = frozen_anchor.merge(anchored, on=keys, how='outer', indicator=True, validate='one_to_one')
        _require(joined['_merge'].eq('both').all() and all(
            S._collection_days(joined[field + '_x']).eq(S._collection_days(joined[field + '_y'])).fillna(False).all()
            for field in dates[len(keys):]), 'AUDIT_STEP10_BASELINE_MISMATCH')
    return (tables, records, issues, baseline, membership), features, registry, roles


def _paths_overlap(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)


class AuditUnavailable(AuditError):
    """An indispensable artifact is unavailable."""


INPUT_ERRORS = (OSError, ValueError, TypeError, AttributeError, KeyError, RuntimeError, yaml.YAMLError)
DOCUMENTS = (
    'configs/baseline_window_03.yaml', 'configs/n3_exploratory_defaults.yaml',
    'stage03/s03_construct_validation.py', 'stage03/s03_n3_exploratory_source.py',
    'stage03/s03_n3_exploratory_models.py', 'stage03/s03_n3_model_readiness_audit.py',
    'stage03/s03_n3_readiness_reporting.py',
)
SUPPORT_DIAGNOSTICS = [
    'complete_case_status', 'community_balance_status', 'binary_community_support_status', 'event_parameter_status',
    'residual_df_status', 'parameter_rank_status', 'outcome_variation_status',
]
COLUMNS = {
    'measurement_funnel': ['representation_id', 'outcome', 'stage', 'source_role', 'status', 'reason',
                           'original_reason', 'provenance'],
    'statistical_support': ['representation_id', 'outcome', 'model', 'status', 'reason',
                            *SUPPORT_DIAGNOSTICS, 'diagnostic_provenance', 'readiness_status', 'review_action'],
    'review_queue': REPORT.REVIEW_COLUMNS,
    'source_assay_audit': ['representation_id', 'outcome', 'check', 'scope', 'provenance',
                           'status', 'category', 'reason'],
    'pipeline_checks': ['check', 'status', 'reason'],
}


def _inputs_exist(directory, files, stage):
    if not all((directory / filename).is_file() and os.access(directory / filename, os.R_OK)
               for filename in files):
        raise AuditUnavailable(f'AUDIT_{stage}_ARTIFACT_UNAVAILABLE')


def _original_files(paths):
    return {role: E.cal.btris_files(paths[role]) for role in E.cal.BTRIS_ROLES}


def _snapshot_sources(paths, btris):
    files = {Path(path).resolve() for role, path in paths.items() if role not in E.cal.BTRIS_ROLES}
    files.update(path.resolve() for group in btris.values() for path in group)
    snapshot = {}
    for path in sorted(files):
        try:
            snapshot[path] = digest(path) if path.is_file() else None
        except OSError:
            snapshot[path] = None
    return snapshot


def _published_overlap(path, reps, outcomes):
    columns = ['representation_id', 'endpoint', 'status', 'reason', 'n_same_source_day_candidate_overlap']
    frame = _table(path, columns)
    rows = []
    for row in frame.loc[frame.representation_id.isin(reps) & frame.endpoint.isin(outcomes)].to_dict('records'):
        reason = row['reason']
        status = row['status']
        if status == 'DESCRIPTIVE' and reason == 'MATCH_ON_PATIENT_OUTCOME_AND_COLLECTION_DAY':
            status = 'PUBLISHED_SAME_DAY_MATCH' if float(row['n_same_source_day_candidate_overlap']) else 'PUBLISHED_ZERO_MATCH'
        elif status != SUPPRESSED:
            status = 'NOT_ASSESSED'
        rows.append(dict(representation_id=row['representation_id'], outcome=row['endpoint'],
                         check='LONGITUDINAL_REUSE', scope='PATIENT_ANALYTE_COLLECTION_DAY',
                         provenance='PUBLISHED_GRAPH03_OVERLAP', status=status))
    return rows


def _verified_controlled_reviews(root, private, graph03, release, sources, validation):
    """Verify overlapping documentary evidence from the existing signed 03e route."""
    from .s03_n3_auto import _signed_yaml, _trust_keys
    from .s03_n3_controlled import input_pins, validate_plan
    from .s03_n3_decisions import verify_decisions

    inference = private / 'institution' / 'inference'
    plan_path = inference / 'active.yaml'
    signature = inference / 'active.sig'
    if not plan_path.exists() and not signature.exists():
        return {}, {}, 'PENDING'
    try:
        key = _trust_keys(private)['pair']
        protected = {path: digest(path) for path in (plan_path, signature, key)}
        _, plan = _signed_yaml(plan_path, signature, key, 'PLAN')
        validate_plan(plan, validation)
        expected = plan['input_sha256']
        stages = private / 'stages' / 'build-reviewed'
        patterns = '*/source/runs/*/reviewed_instrument_measurements.parquet'
        for candidate in sorted(stages.glob(patterns)):
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(stages.resolve()):
                continue
            measurement_sha = digest(candidate)
            if expected.get('measurements') != measurement_sha:
                continue
            pins = input_pins(graph03, release, sources['context'], candidate, root)
            if pins != expected:
                break
            records, _, documents = verify_decisions(plan, inference / 'decisions', pins,
                                                      measurement_trust_key=key)
            planned = {(item['representation_id'], item['source_column']) for item in plan['outcomes']}
            controlled = {(rep, source): 'CONTROLLED_RECORD_CONTENT_VERIFIED'
                          for section, rep, source in records if section == 'independence' and (rep, source) in planned}
            protected.update(documents)
            protected[candidate] = measurement_sha
            return controlled, protected, 'VERIFIED_CONTROLLED_PLAN'
    except INPUT_ERRORS:
        pass
    return {}, {}, 'CONTROLLED_REVIEW_EVIDENCE_UNVERIFIED'


def _blocked_tables(reps, outcomes, state, reason):
    matrices, lineages, funnel, support = [], [], [], []
    for rep in reps:
        for outcome in outcomes:
            label = dict(representation_id=rep, outcome=outcome)
            evidence = dict(structural_class=M.UNRESOLVED, structural_reason=reason, review_status='PENDING',
                            interpretation='POST_HOC_EXPLORATORY', missing_review_requirements='',
                            topology_status=state, topology_use=UNKNOWN, primary_measurement_status=state,
                            data_availability_status=state, assay_comparability_status=state,
                            time_eligibility_status=state, source_funnel_status=state, disclosure_status=SUPPRESSED)
            matrices.append(dict(label, **evidence, primary_blocker=state, additional_blockers=reason,
                                 model_A_support_status=state, model_B_support_status=state, readiness_status=state))
            lineages.append(dict(label, **evidence))
            funnel.append(dict(label, stage='source_rows', status=state, provenance='GRAPH03G_INPUT_CONTRACT'))
            support.extend(dict(label, model=model, status=state, diagnostic_provenance='GRAPH03G_INPUT_CONTRACT')
                           for model in ('A', 'B'))
    return matrices, lineages, funnel, support, []


def _publish(target, tables, checks, summary, settings, references):
    _require(not target.is_symlink(), 'AUDIT_OUTPUT_DIRECTORY_PERMISSIONS')
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = target.stat()
    _require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
             and info.st_uid == os.geteuid(), 'AUDIT_OUTPUT_DIRECTORY_PERMISSIONS')
    matrices, lineages, funnel, support, assays = tables
    dispositions = {(row['representation_id'], row['outcome']): row['readiness_status'] for row in matrices}
    for row in support:
        row['readiness_status'] = dispositions[(row['representation_id'], row['outcome'])]
        if row['status'] == SUPPRESSED:
            row['review_action'] = 'AUTHORIZED_STATISTICAL_REVIEW_REQUIRED'
            row.update(dict.fromkeys(SUPPORT_DIAGNOSTICS, SUPPRESSED))
    queue = REPORT.review_queue(matrices, summary['window_policy_version'])
    privacy = dict(status=PASS, privacy_class='INTERNAL_RESTRICTED', public_export=False,
                   minimum_disclosure_cell=settings['minimum_disclosure_cell'],
                   complementary_suppression='PUBLISHED_CATEGORIES_ONLY_SECURE_RESULTS_WITHHELD',
                   patient_rows_exported=False, source_values_exported=False, free_text_exported=False,
                   private_paths_exported=False, **baseline_policy().metadata())
    with stage_outputs({'stage': target}, lambda _, name: name in OUTPUT_FILES,
                       failure=('stage', '03g_failure.json'), completion=target / '03g_summary.json') as staged:
        directory = staged['stage']
        for name, rows in zip(CSV_OUTPUTS, (matrices, lineages, funnel, support, queue, assays, checks)):
            frame = pd.DataFrame(rows, columns=COLUMNS.get(name))
            frame['window_policy_version'] = summary['window_policy_version']
            frame['graph02_release_sha256'] = summary['graph02_release_sha256']
            frame.to_csv(directory / f'03g_{name}.csv', index=False)
        write_json(directory / '03g_summary.json', summary)
        write_json(directory / '03g_privacy_qc.json', privacy)
        (directory / '03g_RESULTADOS.md').write_text(REPORT.report(matrices, summary, queue), encoding='utf-8')
        (directory / '03g_REVIEW_PACKET.md').write_text(
            REPORT.review_packet(queue, references, summary['graph02_release_sha256']), encoding='utf-8')


def run_audit(root, *, mode='secure-recompute', representation='all', graph02_release=None, graph03_run=None,
              graph03e_dir=None, output_dir=None, strict=True):
    """Audit immutable inputs and publish the safe comparison and technical disposition."""
    root = Path(root).resolve()
    settings = E._policy(root)
    _require(settings['privacy_class'] == 'INTERNAL_RESTRICTED', 'AUDIT_INVALID_PRIVACY_POLICY')
    _require(mode in {'summaries', 'secure-recompute'}, 'AUDIT_INVALID_MODE')
    all_reps = settings['representations']
    _require(representation in {'all', *all_reps}, 'AUDIT_INVALID_REPRESENTATION')
    reps = all_reps if representation == 'all' else [representation]
    outcomes = [S.RF_VALUE, S.RF_TEXT, *(outcome for outcome in settings['measurements']['outcomes']
                                      if outcome not in {S.RF_VALUE, S.RF_TEXT})]
    data_root, sources = E._source_settings(root)
    defaults = graph_paths(root, data_root=data_root)
    release = Path(graph02_release or defaults['graph02'] / MANIFEST).resolve()
    _require(release.name == MANIFEST, 'AUDIT_GRAPH02_MANIFEST_REQUIRED')
    graph03 = Path(graph03_run or defaults['graph03']).resolve()
    phasee = Path(graph03e_dir or defaults['phasee']).resolve()
    destination = Path(output_dir or defaults['phasee'].parent / DESTINATION)
    _require(not destination.is_symlink(), 'AUDIT_OUTPUT_DIRECTORY_PERMISSIONS')
    target = destination.resolve()
    frozen_roots = {release.parent, graph03, phasee, *defaults.values()}
    protected = {p.resolve() for folder in frozen_roots for p in stage_destinations(folder).values()}
    protected.update(Path(path).resolve() for path in sources.values())
    auto_config = yaml.safe_load((root / E.CONFIGS / 'n3_auto.yaml').read_text())
    private = (root / auto_config['private_root']).resolve()
    protected.update((root / E.CONFIGS, root / '.git', private))
    _require(not any(_paths_overlap(target, item) for item in protected), 'AUDIT_OUTPUT_OVERLAPS_INPUT')
    validation_path = root / E.CONFIGS / E.VALIDATION
    references = {f'src/studies/graph/{name}': digest(root / 'src/studies/graph' / name) for name in DOCUMENTS}
    config_inputs = {root / path for path in references} | {
        validation_path, root / E.CONFIGS / 'pipeline.yaml', root / E.CONFIGS / 'n3_auto.yaml',
    }

    limitations, checks, manifest_sha, verified, eligible = [], [], None, False, []
    failure = None
    check = 'GRAPH02_RELEASE'
    with ExitStack() as stack:
        try:
            if not release.is_file():
                raise AuditUnavailable('AUDIT_GRAPH02_MANIFEST_UNAVAILABLE')
            if not graph03.is_dir():
                raise AuditUnavailable('AUDIT_GRAPH03_AGGREGATES_UNAVAILABLE')
            for directory in sorted({release.parent, graph03, phasee}):
                if directory.is_dir():
                    stack.enter_context(output_lock(directory, shared=True))
            manifest, release = validate_release(release)
            frozen = E.frozen_fingerprints(release, manifest)
            _require(set(frozen) == {'release_manifest', 'baseline_raw', 'membership', 'feature_sets'},
                     'AUDIT_INCOMPLETE_FROZEN_FINGERPRINTS')
            eligible = manifest['eligible_representation_ids']
            _require(isinstance(eligible, list) and len(eligible) == len(set(eligible)) and set(eligible) <= set(all_reps),
                     'AUDIT_INVALID_REPRESENTATION_INVENTORY')
            artifacts = artifact_paths(manifest)
            lineage_files = _artifacts(release.parent, {role: artifacts[role]
                                       for role in ('feature_sets', 'clinical_registry', 'release_qc')})
            checks.append(dict(check=check, status=PASS))
            check = 'GRAPH03_CONTRACT'
            _inputs_exist(graph03, GRAPH03.values(), 'GRAPH03')
            graph03_files = _artifacts(graph03, GRAPH03)
            roles = _graph03_metadata(graph03_files, manifest, frozen, strict)
            overlap_name = f"03_graph_{E.BASELINE_ARTIFACTS['temporal_overlap']}.csv"
            overlap_path = graph03 / overlap_name
            if overlap_path.is_file():
                overlap_path = safe_artifact(graph03, overlap_name)
                expected = json.loads(graph03_files['summary'].read_text()).get('sha256_by_artifact', {}).get(overlap_name)
                _require(expected == digest(overlap_path), 'AUDIT_GRAPH03_ARTIFACT_PIN_MISMATCH')
                graph03_files['overlap'] = overlap_path
            checks.append(dict(check=check, status=PASS))
            check = 'GRAPH03E_CONTRACT'
            cell = settings['minimum_disclosure_cell']
            try:
                phasee_files, published = _read_graph03e(phasee, graph03, frozen, all_reps, outcomes, manifest,
                                                        strict, cell)
            except INPUT_ERRORS as exc:
                failure = str(exc) if isinstance(exc, AuditError) else 'AUDIT_INPUT_SCHEMA_OR_INTEGRITY_ERROR'
                state = UNAVAILABLE if isinstance(exc, (AuditUnavailable, FileNotFoundError, PermissionError)) else 'INTEGRITY_BLOCKED'
                checks.append(dict(check=check, status=state, reason=failure))
                phasee_files, published = {}, _unavailable_graph03e(all_reps, outcomes, state)
                if state == UNAVAILABLE:
                    limitations.append(UNAVAILABLE)
            else:
                checks.append(dict(check=check, status=PASS))

            check = 'POLICY_AND_INPUT_PINS'
            validation = yaml.safe_load(validation_path.read_text())
            require_policy(validation)
            _validate_reviews(validation)
            integrated = manifest.get('source_inputs', {}).get('integrated', {})
            if integrated.get('path') and Path(integrated['path']).resolve() != sources['step10']:
                sources['step10'] = Path(integrated['path']).resolve()
                sources['context'] = sources['step10'].with_name('10_integrated_longitudinal_context.parquet')
            _require(not any(_paths_overlap(target, Path(path).resolve()) for path in sources.values()),
                     'AUDIT_OUTPUT_OVERLAPS_INPUT')
            files = {*lineage_files.values(), *graph03_files.values(), *phasee_files.values(), *config_inputs}
            before = {path: digest(path) for path in files}
            original_snapshot, btris = {}, {}
            data = None
            if mode == 'secure-recompute':
                btris = _original_files(sources)
                original_snapshot = _snapshot_sources(sources, btris)
                source_pin = original_snapshot.get(sources['step10'].resolve())
                if strict and integrated.get('sha256') and source_pin is not None:
                    _require(source_pin == integrated['sha256'], 'AUDIT_STEP10_PIN_MISMATCH')
                data, features, registry, roles = _secure_data(
                    root, release, manifest, sources, btris, graph03, validation)
            else:
                features = pd.read_csv(lineage_files['feature_sets'])
                registry = pd.read_csv(lineage_files['clinical_registry'])
                limitations.append('PUBLISHED_AGGREGATES_ONLY')
            _require({'representation_id', 'source_column', 'feature', 'included'} <= set(features)
                     and features.included.isin([True, False]).all()
                     and not features.duplicated(['representation_id', 'feature']).any()
                     and 'source_column' in registry, 'AUDIT_FROZEN_LINEAGE_SCHEMA_INVALID')
            checks.append(dict(check=check, status=PASS))
            controlled_reviews, review_pins = {}, {}
            if data:
                controlled_reviews, review_pins, review_state = _verified_controlled_reviews(
                    root, private, graph03, release, sources, validation)
                checks.append(dict(check='CONTROLLED_REVIEW_EVIDENCE', status=review_state))
            check = 'MEASUREMENT_RECONCILIATION'
            matrices, lineages, funnel, support, assays, source_states, drift = _build(
                reps, outcomes, settings, validation, manifest, features, registry, roles, published,
                data, controlled_reviews)
            if not set(reps) <= set(eligible):
                limitations.append('REPRESENTATION_UNAVAILABLE')
            if any(state == UNAVAILABLE for state in source_states.values()):
                limitations.append(UNAVAILABLE)
            if drift:
                failure = 'PUBLISHED_SELECTION_OR_COVERAGE_DRIFT'
                checks.append(dict(check=check, status='INTEGRITY_BLOCKED', reason=failure))
            elif published.get('state'):
                checks.append(dict(check=check, status=published['state']))
            elif data and all(state == 'AVAILABLE' for state in source_states.values()):
                checks.append(dict(check=check, status='AGGREGATE_RECONCILED',
                                   reason='PATIENT_LEVEL_SELECTION_NOT_IN_PUBLISHED_CONTRACT'))
            else:
                checks.append(dict(check=check, status=UNAVAILABLE if data else UNKNOWN))
            for role, state in source_states.items():
                checks.append(dict(check=role, status=state))
            longitudinal = set(validation.get('longitudinal_endpoints', [])) & set(outcomes)
            overlap = (_published_overlap(graph03_files['overlap'], reps, longitudinal)
                       if longitudinal and 'overlap' in graph03_files else [])
            assays.extend(overlap)
            overlap_seen = {(row['representation_id'], row['outcome']) for row in overlap}
            for rep in reps:
                for source in outcomes:
                    if (rep, source) not in overlap_seen:
                        status = (UNAVAILABLE if source in longitudinal else 'NO_COMPARABLE_ENDPOINT_CONFIGURED')
                        assays.append(dict(representation_id=rep, outcome=source, check='LONGITUDINAL_REUSE',
                                           scope='PATIENT_ANALYTE_COLLECTION_DAY', status=status))
            overlap_state = ('PUBLISHED_GRAPH03_REFERENCE' if overlap else UNAVAILABLE if longitudinal else
                             'NO_COMPARABLE_ENDPOINT_CONFIGURED')
            checks.append(dict(check='LONGITUDINAL_REUSE', status=overlap_state))
            if longitudinal and any((rep, source) not in overlap_seen for rep in eligible for source in longitudinal):
                limitations.append('LONGITUDINAL_REUSE_EVIDENCE_UNAVAILABLE')
            checks.extend([
                dict(check='TEMPORAL_POLICY', status=PASS),
                dict(check='TOPOLOGY', status='GRAPH03_FROZEN_CONTRACT_VERIFIED' if data else 'PUBLISHED_GRAPH03_REFERENCE'),
                dict(check='MODEL_A_B_DISCLOSURE', status=SUPPRESSED if data else UNKNOWN),
                dict(check='PHI_AND_COMPLEMENTARY_SUPPRESSION', status=PASS),
            ])
            check = 'FROZEN_INPUT_INVARIANCE'
            _require(before == {path: digest(path) for path in files}
                     and frozen == E.frozen_fingerprints(release, manifest),
                     'AUDIT_INPUTS_CHANGED_DURING_ANALYSIS')
            _require(all(digest(path) == sha for path, sha in review_pins.items()),
                     'AUDIT_REVIEW_EVIDENCE_CHANGED_DURING_ANALYSIS')
            if data:
                _require(original_snapshot == _snapshot_sources(sources, _original_files(sources)),
                         'AUDIT_SOURCE_INPUTS_CHANGED_DURING_ANALYSIS')
            tables = matrices, lineages, funnel, support, assays
            verified, manifest_sha = True, frozen['release_manifest']
            checks.append(dict(check=check, status=PASS))
        except INPUT_ERRORS as exc:
            failure = str(exc) if isinstance(exc, AuditError) else 'AUDIT_INPUT_SCHEMA_OR_INTEGRITY_ERROR'
            if failure == 'AUDIT_OUTPUT_OVERLAPS_INPUT':
                raise
            state = UNAVAILABLE if isinstance(exc, (AuditUnavailable, FileNotFoundError, PermissionError)) else 'INTEGRITY_BLOCKED'
            checks.append(dict(check=check, status=state, reason=failure))
            tables = _blocked_tables(reps, outcomes, state, failure)
            if state == UNAVAILABLE:
                limitations.append(UNAVAILABLE)

        review_pending = any(row['readiness_status'].startswith('N3_') or row['structural_class'] == M.UNRESOLVED
                             for row in tables[0])
        if review_pending:
            limitations.append('HUMAN_REVIEW_REQUIRED')
        if mode == 'secure-recompute':
            limitations.extend(['AUTHORIZED_STATISTICAL_REVIEW_REQUIRED',
                                'PATIENT_LEVEL_SELECTION_NOT_IN_PUBLISHED_CONTRACT'])
        if failure:
            limitations.append(failure)
        status = 'AUDIT_FAILED' if failure else 'AUDIT_COMPLETE_WITH_LIMITATIONS' if limitations else 'AUDIT_COMPLETE'
        summary = dict(version=VERSION, status=status, mode=mode, strict=strict, estimators_run=0,
                       privacy_class='INTERNAL_RESTRICTED', public_export=False,
                       representations=reps, representations_available=[rep for rep in reps if rep in eligible],
                       outcomes=outcomes, **baseline_policy().metadata(),
                       graph02_release_sha256=manifest_sha, frozen_inputs_unchanged=verified,
                       review_pending=review_pending, limitations=list(dict.fromkeys(limitations)),
                       next_action=('REVIEW_PIPELINE_CHECKS' if failure or UNAVAILABLE in limitations else
                                    'REVIEW_03G_REVIEW_QUEUE'))
        _publish(target, tables, checks, summary, settings, references)
    result = dict(status=status, mode=mode, representations_requested=reps, estimators_run=0,
                  human_review_required=review_pending, privacy_class=summary['privacy_class'])
    if failure:
        result['reason'] = failure
    return result

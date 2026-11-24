"""Read-only Graph03g diagnostics over the exploratory Graph03e contract."""

from collections import Counter
from contextlib import ExitStack
import json
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
from .s03_baseline_window import DAY_BINS, policy as baseline_policy, require_policy
from .s03_construct_validation import _dependencies, _paths, _review, feature_roles, independence_review_approved
from .s03_release_reader import RESULTS, RESULTS_SCHEMA, hard_members
from .s03_run_resolution import graph_paths

VERSION = 'graph03g_model_readiness_v1'
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
}
CSV_OUTPUTS = ('readiness_matrix', 'independence_lineage', 'measurement_funnel', 'statistical_support')
OUTPUT_FILES = {f'03g_{name}.csv' for name in CSV_OUTPUTS} | {
    '03g_summary.json', '03g_privacy_qc.json', '03g_RESULTADOS.md', '03g_failure.json'}
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
PUBLISHED_REASONS = PUBLISHED_STATES | {
    'INDEPENDENCE_REVIEW_PENDING', 'TOPOLOGY_NOT_STABLE', 'FROZEN_BASELINE_UNAVAILABLE',
    'FROZEN_SOURCE_OR_DERIVATION_DEFINES_MAPPER', 'DOCUMENTED_LINEAGE_OR_CLINICAL_PROXY',
    'NO_VALID_MEASUREMENTS_IN_SCOPE', 'RF_NUMERIC_MODEL_ESTIMATED', 'COVERAGE_DISCLOSURE_RESTRICTION',
    'FROZEN_DEFINING_FEATURE_LINEAGE_INCOMPLETE', 'OUTCOME_CONSTRUCT_LINEAGE_MISSING_OR_AMBIGUOUS',
    'FROZEN_FEATURE_REGISTRY_ROLE_AND_PROXY_CHECKS',
    'DEFINING_SOURCE_REGISTRY_LINEAGE_MISSING', 'OUTCOME_ROLE_UNAVAILABLE',
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


def _published(files, graph03, fingerprints, reps, outcomes, manifest, strict):
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
    data = {}
    window_fields = ['window_policy_version', 'window_policy_sha256']
    columns = {
        'coverage': ['stratum', 'provenance', 'definition_role', 'status', 'topology_status', 'n_measured'],
        'models': ['stratum', 'provenance', 'status', 'not_estimable_reason'],
        'independence': ['classification', 'reason'],
    }
    for table, fields in columns.items():
        frame = _table(files[table], PAIRS + fields + window_fields)
        if table != 'independence':
            scope = frame.stratum.eq(S.PRIMARY) & frame.provenance.eq(S.ORIGINAL)
            if table == 'coverage':
                scope &= frame.definition_role.eq('PRIMARY')
            frame = frame.loc[scope]
        _require(not frame.duplicated(PAIRS).any(), 'AUDIT_DUPLICATE_PUBLISHED_PAIR')
        _require(set(zip(frame.representation_id, frame.outcome)) == {
            (rep, outcome) for rep in reps for outcome in outcomes}, 'AUDIT_INCOMPLETE_PUBLISHED_PAIRS')
        _require(frame.window_policy_version.eq(baseline_policy().policy_version).all()
                 and frame.window_policy_sha256.eq(baseline_policy().sha256).all(),
                 'AUDIT_WINDOW_POLICY_MISMATCH')
        data[table] = frame.set_index(PAIRS).to_dict('index')
    return summary, data


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
    }
    fields = PAIRS + ['stratum', 'stage', 'status', 'n_remaining', 'window_policy_version', 'window_policy_sha256']
    frame = _table(path, fields)
    frame = frame.loc[frame.representation_id.eq('') & frame.stratum.eq('ALL') & frame.stage.isin(stages)]
    _require(not frame.duplicated(['outcome', 'stage']).any(), 'AUDIT_DUPLICATE_PUBLISHED_STAGE')
    _require(frame.window_policy_version.eq(baseline_policy().policy_version).all()
             and frame.window_policy_sha256.eq(baseline_policy().sha256).all(),
             'AUDIT_WINDOW_POLICY_MISMATCH')
    projected = {}
    for outcome in outcomes:
        evidence = frame.loc[frame.outcome.eq(outcome)].set_index('stage').to_dict('index')
        output, stop = [], UNKNOWN
        preceding_present = True
        for stage, reason in stages.items():
            row = evidence.get(stage, {})
            state = row.get('status', UNKNOWN)
            if state == 'DESCRIPTIVE':
                count = float(row['n_remaining'])
                _require(count.is_integer() and (count == 0 or count >= cell), 'AUDIT_INVALID_PUBLISHED_FLOW')
                state = 'PRESENT' if count else 'EMPTY'
            elif state == SUPPRESSED:
                _require(row['n_remaining'] == '', 'AUDIT_INVALID_PUBLISHED_FLOW')
            else:
                state = UNAVAILABLE if state == UNAVAILABLE else UNKNOWN
            first_loss = preceding_present and state == 'EMPTY'
            if first_loss:
                stop = reason
            preceding_present &= state == 'PRESENT'
            output.append(dict(outcome=outcome, stage=stage, status=state,
                               reason=reason if first_loss else '', provenance='PUBLISHED_03E_COHORT_FLOW'))
        projected[outcome] = (output, stop)
    return projected


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


def _reason(value):
    return '|'.join(part for part in value.split('|') if part in PUBLISHED_REASONS) or UNKNOWN


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
        'INDEPENDENT_REVIEWED' if approved else 'STRUCTURALLY_INDEPENDENT_REVIEW_PENDING')}.get(structural, structural)
    paths = _paths(dependencies, source)
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
        blocking_sources=_metadata_tokens(audit['blocking_sources']), relationship_types='|'.join(types))
    return fields, lineage, structural


def _funnel(result, eda, source, cell):
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


def _coverage_state(row, cell):
    status = row['status']
    _require(status in {'DESCRIPTIVE', SUPPRESSED, UNAVAILABLE, 'REPRESENTATION_UNAVAILABLE'},
             'AUDIT_INVALID_PUBLISHED_COVERAGE')
    count = row['n_measured']
    if status == 'DESCRIPTIVE':
        count = float(count)
        _require(count.is_integer() and (count == 0 or count >= cell), 'AUDIT_INVALID_PUBLISHED_COVERAGE')
        return 'ZERO_ELIGIBLE_MEASUREMENTS' if count == 0 else 'MEASUREMENTS_OBSERVED'
    _require(count == '', 'AUDIT_INVALID_PUBLISHED_COVERAGE')
    return status


def _support(frame, numeric, policy, cell, protected):
    audit = {}
    info, prepared = M._model_support(frame, numeric, policy, cell, audit=audit)
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
        design = audit.get('designs', {}).get(model)
        reason = info.get('reason', '') if design is None else design['support_blocking_reasons']
        if design and design['n_residual_df'] < policy['minimum_residual_degrees_of_freedom']:
            reason = '|'.join(sorted(set(filter(None, reason.split('|'))) | {'INSUFFICIENT_RESIDUAL_DF'}))
        details[model] = dict(
            status=SUPPRESSED if protected else 'BLOCKED' if reason else PASS,
            reason='' if protected else reason,
            complete_case_status=SUPPRESSED if protected else
            'MEETS_MINIMUM' if common_n >= policy['minimum_complete_cases'] else 'BELOW_MINIMUM',
            community_balance_status=SUPPRESSED if protected else
            'MEETS_MINIMUM_GROUP_COUNT' if supported_groups >= policy['minimum_groups'] else 'BELOW_MINIMUM_GROUP_COUNT',
            binary_community_support_status=SUPPRESSED if protected else 'NOT_APPLICABLE' if numeric else
            'BELOW_MINIMUM' if reason == 'INSUFFICIENT_BINARY_COMMUNITY_SUPPORT' else PASS if design else 'NOT_EVALUATED',
            event_parameter_status=SUPPRESSED if protected else 'NOT_APPLICABLE' if numeric else
            'BELOW_MINIMUM' if 'INSUFFICIENT_EVENTS_PER_PARAMETER' in reason else PASS if design else 'NOT_EVALUATED',
            residual_df_status=SUPPRESSED if protected else
            'BELOW_MINIMUM' if 'INSUFFICIENT_RESIDUAL_DF' in reason else PASS if design else 'NOT_EVALUATED',
            parameter_rank_status=SUPPRESSED if protected else
            'RANK_DEFICIENT' if 'RANK_DEFICIENT_DESIGN' in reason else 'FULL_RANK' if design else 'NOT_EVALUATED',
            outcome_variation_status=(SUPPRESSED if protected else
                                      'VARIES' if frame.loc[audit['common'], 'value'].nunique() > 1
                                      else 'INSUFFICIENT_VARIATION_OR_NO_CASES'),
            diagnostic_provenance='SECURE_03E_COMMON_MASK',
        )
    return details, prepared is not None and not protected


def _not_computed(status, reason, provenance):
    return {model: dict(status=status, reason=reason, diagnostic_provenance=provenance) for model in ('A', 'B')}


def _report(rows, summary):
    lines = [
        '# Auditoría Graph03g de independencia y estimabilidad', '',
        'La auditoría aplica la ventana cerrada desde el día basal hasta seis meses calendario después. '
        'La cobertura primaria exige procedencia original verificada y asignación de comunidad hard.', '',
        'Un cero indica que ninguna medición cumplió los filtros de la cobertura publicada. '
        'Una celda suprimida deja indeterminado el conteo. Los diagnósticos de soporte usan la población '
        'común de Graph03e para A y B. Graph03g ejecutó cero estimadores.', '',
        '| Representación | Desenlace | Motivo publicado en 03e | Cobertura primaria | Embudo | A | B | Decisión |',
        '|---|---|---|---|---|---|---|---|',
    ]
    fields = ['representation_id', 'outcome', 'graph03e_reason', 'primary_measurement_status',
              'source_funnel_status', 'model_A_support_status', 'model_B_support_status', 'readiness_status']
    for row in rows:
        lines.append('| ' + ' | '.join(row[field].replace('|', ' / ') for field in fields) + ' |')
    lines.extend(['', '## Revisión y evidencia pendiente', ''])
    for row in rows:
        missing = row['missing_review_requirements'].replace('|', ', ')
        if missing:
            lines.append(f"- **{row['representation_id']} · {row['outcome']}**. Requisitos pendientes. {missing}.")
        if row['primary_measurement_status'] == 'ZERO_ELIGIBLE_MEASUREMENTS':
            lines.append(f"- **{row['representation_id']} · {row['outcome']}**. Cero mediciones elegibles. "
                         f"Estado de la causa en el embudo. {row['source_funnel_status']}.")
    lines.extend(['',
        'Las categorías evaluables solo con fuentes protegidas requieren el modo `secure-recompute`. '
        'Las revisiones acreditadas y la preespecificación se registran en los contratos vigentes de Graph03. '
        'El estado `READY_FOR_ESTIMATION` permite remitir el desenlace a Graph03e para comprobar el ajuste. '
        'La ruta controlada conserva sus requisitos de firma.', '',
        'Los desenlaces N1 y N2 se interpretan como descriptivos por su relación con Mapper. '
        'Un bloqueo de soporte tras la aprobación conserva la clasificación de no estimable. '
        'Un dato pendiente de verificar mantiene abierta la decisión.', '',
        f"Modo de auditoría `{summary['mode']}`.", '',
    ])
    return '\n'.join(lines)


def _build(reps, outcomes, policy, validation, manifest, features, registry, roles, published, data=None):
    dependencies = _dependencies(features, registry, validation)
    role_map = {(row['representation_id'], row['source_column']): row for row in roles.to_dict('records')}
    cell = policy['minimum_disclosure_cell']
    matrices, lineages, funnel, support = [], [], [], []
    eligible = set(manifest['eligible_representation_ids'])
    if data:
        tables, records, issues, baseline, membership = data
        source_available = {'eda_long', 'step10'} <= set(tables)
        unavailable_roles = {row.get('artifact') for row in issues if row.get('status') == UNAVAILABLE}
        if source_available:
            candidates = S._candidates(tables, baseline, records, policy)
            eda = pd.DataFrame({'_patient_match': S.NORMALIZE(tables['eda_long'].patient_id)})
    for source in outcomes:
        source_state = UNKNOWN
        selection = result = None
        if data:
            for role in SOURCE_ROLES:
                if role in SOURCE_ROLES[:4]:
                    state = ('SOURCE_SCHEMA_UNAVAILABLE' if role not in tables else
                             'NO_SOURCE_ROWS' if tables[role].empty else 'AVAILABLE')
                else:
                    compatible_schema = any(row['source_role'] == role
                                            and row['collection_column'] != 'COLLECTION_COLUMN_MISSING'
                                            and row['original_discriminators'] != 'DISCRIMINATOR_MISSING'
                                            for row in records.schemas)
                    state = ('PARTIAL_SOURCE_AVAILABILITY' if compatible_schema and role in unavailable_roles else
                             'AVAILABLE' if compatible_schema else 'SOURCE_SCHEMA_UNAVAILABLE')
                funnel.append(dict(outcome=source, stage='source_inventory', source_role=role,
                                   status=state, provenance='SECURE_SOURCE_SCHEMA'))
            if source_available:
                result = S._outcome(candidates, source, policy)
                selection = S.primary_measurements(result)
                _require(not selection.patient_id.duplicated().any()
                         and set(selection.patient_id) <= set(baseline.patient_id)
                         and selection.window_bin.isin(DAY_BINS[1:5]).all(),
                         'AUDIT_INVALID_PRIMARY_SELECTION')
                transitions, source_state = _funnel(result, eda, source, cell)
                funnel.extend(transitions)
            else:
                source_state = 'SOURCE_SCHEMA_UNAVAILABLE'
        else:
            transitions, source_state = published['flow'][source]
            funnel.extend(transitions)

        for rep in reps:
            label = dict(representation_id=rep, outcome=source)
            pair = (rep, source)
            coverage = published['coverage'][pair]
            model = published['models'][pair]
            prior = published['independence'][pair]
            measured = _coverage_state(coverage, cell)
            disclosure = SUPPRESSED if measured == SUPPRESSED else 'PUBLISHED_COVERAGE_ONLY'
            fields, lineage, structural = _independence(
                rep, source, role_map, features, registry, dependencies, validation, policy, rep in eligible)
            graph03e_reason = _reason(model['not_estimable_reason'])
            topology = coverage['topology_status']
            topology = topology if topology in TOPOLOGIES else UNKNOWN
            if rep not in eligible:
                topology = 'REPRESENTATION_UNAVAILABLE'
            if manifest['release_status'] == 'NO_STABLE_STRUCTURE':
                topology = 'NO_STABLE_STRUCTURE'
            assay = time = hard_scope = UNKNOWN
            if measured == 'MEASUREMENTS_OBSERVED':
                assay, time, hard_scope = 'COMPARABLE_PRIMARY_DEFINITION', 'PRIMARY_WINDOW_VERIFIED', 'HARD_COMMUNITY_MATCH'
            adequate = False
            if measured == 'ZERO_ELIGIBLE_MEASUREMENTS':
                diagnostics = _not_computed('BLOCKED', 'NO_VALID_MEASUREMENTS_IN_SCOPE', 'PUBLISHED_03E_COVERAGE')
            else:
                diagnostics = _not_computed(UNKNOWN, '', 'PUBLISHED_03E_AGGREGATES')
            if measured == SUPPRESSED:
                diagnostics = _not_computed(SUPPRESSED, '', 'PUBLISHED_03E_COVERAGE')
            if data and rep in eligible:
                members = membership[rep]
                topology = E._topology(rep, members) if manifest['release_status'] != 'NO_STABLE_STRUCTURE' else topology
                if result is not None:
                    frame = members.merge(selection[['patient_id', 'value']], on='patient_id', how='left',
                                          validate='one_to_one')
                    numeric = source == S.RF_VALUE
                    # Keep the existing 03e coverage checks, including complements
                    # against the frozen cohort and within the disposition partition.
                    private_label = {}
                    profiles = R._profiles(frame, private_label, numeric, cell)
                    private_coverage = R._coverage(frame, private_label, profiles, cell,
                                                   S._test_disposition(result, baseline.patient_id))
                    protected = (measured == SUPPRESSED or private_coverage['status'] == SUPPRESSED
                                 or R._small(len(baseline) - len(frame), cell))
                    fresh_count = int(frame.value.notna().sum())
                    if coverage['status'] == 'DESCRIPTIVE' and fresh_count != float(coverage['n_measured']):
                        measured = 'PUBLISHED_COVERAGE_DIFFERS_FROM_RECOMPUTE'
                        protected = True
                    diagnostics, adequate = _support(frame, numeric, policy['models'], cell, protected)
                    disclosure = SUPPRESSED if diagnostics['A']['status'] == SUPPRESSED else 'CATEGORICAL_ONLY'
                    if not protected and (fresh_count or source_state != SUPPRESSED):
                        assay = ('COMPARABLE_PRIMARY_DEFINITION' if not selection.empty
                                 else 'NO_ELIGIBLE_PRIMARY_DEFINITION')
                        time = ('PRIMARY_WINDOW_VERIFIED' if not selection.empty else
                                'NO_VERIFIED_IN_WINDOW' if source_state in {
                                    'COLLECTION_DATE_UNVERIFIED', 'OUTSIDE_0_TO_6M'} else UNKNOWN)
                        hard_scope = ('HARD_COMMUNITY_MATCH' if fresh_count else
                                      'NO_HARD_COMMUNITY_MATCH' if not selection.empty else
                                      'NO_ELIGIBLE_PRIMARY_DEFINITION')
                        if measured == UNAVAILABLE:
                            measured = 'MEASUREMENTS_OBSERVED' if fresh_count else 'ZERO_ELIGIBLE_MEASUREMENTS'
                    elif protected:
                        assay = time = hard_scope = SUPPRESSED
            funnel.append(dict(label, stage='primary_hard_community_coverage', status=measured,
                               reason='COVERAGE_SUPPRESSED' if measured == SUPPRESSED else
                               hard_scope if hard_scope == 'NO_HARD_COMMUNITY_MATCH' else '',
                               provenance='PUBLISHED_03E_COVERAGE_AND_SECURE_CHECK' if data else
                               'PUBLISHED_03E_COVERAGE'))
            if data:
                funnel.append(dict(label, stage='complete_cases_in_supported_communities',
                                   status=diagnostics['A'].get('complete_case_status', diagnostics['A']['status']),
                                   provenance=diagnostics['A']['diagnostic_provenance']))
            for name, diagnostic in diagnostics.items():
                support.append(dict(label, model=name, **diagnostic))

            blockers = []
            if structural != M.INDEPENDENT:
                blockers.append(fields['structural_reason'])
            if fields['approval_gate_status'] != PASS:
                blockers.append(fields['approval_gate_status'])
            if topology not in STABLE:
                blockers.append('TOPOLOGY_NOT_STABLE' if topology != UNKNOWN else 'TOPOLOGY_UNVERIFIED')
            if measured != 'MEASUREMENTS_OBSERVED':
                blockers.append('COVERAGE_SUPPRESSED' if measured == SUPPRESSED else measured)
            if assay != 'COMPARABLE_PRIMARY_DEFINITION':
                blockers.append('ASSAY_COMPARABILITY_UNVERIFIED')
            if time != 'PRIMARY_WINDOW_VERIFIED':
                blockers.append('TIME_ELIGIBILITY_UNVERIFIED')
            for name, diagnostic in diagnostics.items():
                if diagnostic['status'] != PASS:
                    blockers.extend(diagnostic['reason'].split('|') if diagnostic['reason'] else
                                    [f'MODEL_{name}_SUPPORT_{diagnostic["status"]}'])
            if disclosure == SUPPRESSED:
                blockers.append('DISCLOSURE_RESTRICTED')
            if structural in {M.DIRECT, M.PROXY}:
                decision = 'N1_DESCRIPTIVE' if structural == M.DIRECT else 'N2_DESCRIPTIVE'
            elif structural != M.INDEPENDENT:
                decision = 'LINEAGE_UNRESOLVED'
            elif fields['approval_gate_status'] != PASS:
                decision = ('STATISTICALLY_PLAUSIBLE_PENDING_REVIEW' if adequate and topology in STABLE
                            and assay == 'COMPARABLE_PRIMARY_DEFINITION' and time == 'PRIMARY_WINDOW_VERIFIED'
                            and measured == 'MEASUREMENTS_OBSERVED' else 'N3_REVIEW_PENDING')
            elif not blockers:
                decision = 'READY_FOR_ESTIMATION'
            elif topology not in STABLE | {UNKNOWN} or any(d['status'] == 'BLOCKED' for d in diagnostics.values()):
                decision = 'N3_APPROVED_NOT_ESTIMABLE'
            else:
                decision = 'N3_APPROVED_READINESS_UNVERIFIED'
            if source in {S.RF_VALUE, S.RF_TEXT}:
                rf_priority = 'NUMERIC_FIRST' if source == S.RF_VALUE else 'QUALITATIVE_IF_NUMERIC_NOT_ESTIMABLE'
            else:
                rf_priority = ''
            if source == S.RF_TEXT and any(row['representation_id'] == rep and row['outcome'] == S.RF_VALUE
                    and row['readiness_status'] in {'READY_FOR_ESTIMATION', 'STATISTICALLY_PLAUSIBLE_PENDING_REVIEW'}
                    for row in matrices):
                blockers.insert(0, 'RF_NUMERIC_HAS_PRIMARY_PRIORITY')
                if decision == 'READY_FOR_ESTIMATION':
                    decision = 'N3_APPROVED_RF_ALTERNATIVE_CONDITIONAL'
            blockers = list(dict.fromkeys(blockers))
            matrices.append(dict(label, **fields, primary_measurement_status=measured, source_funnel_status=source_state,
                                 data_availability_status=('SOURCE_SCHEMA_UNAVAILABLE' if source_state ==
                                     'SOURCE_SCHEMA_UNAVAILABLE' else measured),
                                 assay_comparability_status=assay, time_eligibility_status=time,
                                 community_support_status=hard_scope, topology_status=topology,
                                 model_A_support_status=diagnostics['A']['status'],
                                 model_B_support_status=diagnostics['B']['status'], disclosure_status=disclosure,
                                 graph03e_reason=graph03e_reason, graph03e_model_status=(
                                     model['status'] if model['status'] in PUBLISHED_STATES else UNKNOWN),
                                 graph03e_independence_status=prior['classification'] if
                                 prior['classification'] in PUBLISHED_STATES else UNKNOWN,
                                 graph03e_independence_reason=_reason(prior['reason']),
                                 hypothesis_family=M._family(source), rf_primary_priority=rf_priority,
                                 primary_blocker=blockers[0] if blockers else '',
                                 additional_blockers='|'.join(blockers[1:]), readiness_status=decision))
            lineages.append(dict(label, **lineage))
    return matrices, lineages, funnel, support


def _secure_data(root, release, manifest, paths, graph03, validation, strict):
    loaded = E._load_graph02(release.parent)
    _require(loaded['release'] == release and not loaded['data_issues'], 'AUDIT_FROZEN_MEMBERSHIP_INVALID')
    frozen = loaded['frozen']
    baseline = frozen['baseline_raw']
    roles = feature_roles(frozen['feature_sets'], baseline, validation,
                          manifest['eligible_representation_ids'], loaded['registry'])
    E._graph03_qc(graph03, loaded, roles)
    baseline_fields = baseline.reindex(columns=['patient_id', 'clinical_baseline_date',
                                               'demo__age_at_baseline', 'demo__sex'])
    membership = {rep: hard_members(frozen['membership'], rep).merge(
        baseline_fields, on='patient_id', validate='one_to_one') for rep in manifest['eligible_representation_ids']}
    integrated = manifest.get('source_inputs', {}).get('integrated', {})
    if strict and integrated.get('sha256'):
        _require(digest(Path(integrated.get('path') or paths['step10'])) == integrated['sha256'],
                 'AUDIT_STEP10_PIN_MISMATCH')
    tables, records, issues = E.current_source_inputs(root, paths=paths)
    return (tables, records, issues, baseline, membership), roles


def _paths_overlap(left, right):
    return left.is_relative_to(right) or right.is_relative_to(left)


def run_audit(root, *, mode, representation='R0', graph02_release=None, graph03_run=None, graph03e_dir=None,
              output_dir=None, strict=False):
    """Audit the pinned inputs and publish only allowlisted categorical outputs."""
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
    target = Path(output_dir or defaults['phasee'].parent / DESTINATION).resolve()
    frozen_roots = {release.parent, graph03, phasee, *defaults.values()}
    protected = {p.resolve() for folder in frozen_roots for p in stage_destinations(folder).values()}
    protected.update(Path(path).resolve() for path in sources.values())
    protected.update((root / E.CONFIGS, root / '.git'))
    _require(not any(_paths_overlap(target, item) for item in protected), 'AUDIT_OUTPUT_OVERLAPS_INPUT')
    validation_path = root / E.CONFIGS / E.VALIDATION
    validation = yaml.safe_load(validation_path.read_text())
    require_policy(validation)
    _validate_reviews(validation)

    _require(release.is_file(), 'AUDIT_GRAPH02_MANIFEST_UNAVAILABLE')
    _require(graph03.is_dir(), 'AUDIT_GRAPH03_AGGREGATES_UNAVAILABLE')
    _require(phasee.is_dir(), 'AUDIT_GRAPH03E_AGGREGATES_UNAVAILABLE')
    with ExitStack() as stack:
        for directory in sorted({release.parent, graph03, phasee}):
            stack.enter_context(output_lock(directory, shared=True))
        manifest, release = validate_release(release)
        artifacts = artifact_paths(manifest)
        lineage_files = {role: safe_artifact(release.parent, artifacts[role])
                         for role in ('feature_sets', 'clinical_registry', 'release_qc')}
        graph03_files = _artifacts(graph03, GRAPH03)
        phasee_files = _artifacts(phasee, GRAPH03E)
        frozen = E.frozen_fingerprints(release, manifest)
        _require(set(frozen) == {'release_manifest', 'baseline_raw', 'membership', 'feature_sets'},
                 'AUDIT_INCOMPLETE_FROZEN_FINGERPRINTS')
        integrated = manifest.get('source_inputs', {}).get('integrated', {}).get('path')
        if integrated:
            sources['step10'] = Path(integrated).resolve()
            sources['context'] = sources['step10'].with_name('10_integrated_longitudinal_context.parquet')
        _require(not any(_paths_overlap(target, Path(item).resolve()) for item in sources.values()),
                 'AUDIT_OUTPUT_OVERLAPS_INPUT')
        inputs = {f'graph02_{role}': path for role, path in lineage_files.items()}
        inputs.update({f'graph03_{role}': path for role, path in graph03_files.items()})
        inputs.update({f'graph03e_{role}': path for role, path in phasee_files.items()})
        inputs.update(validation_config=validation_path, exploratory_config=root / E.CONFIGS / E.DEFAULTS)
        before = {role: digest(path) for role, path in inputs.items()} | frozen
        roles = _graph03_metadata(graph03_files, manifest, frozen, strict)
        published_summary, published = _published(phasee_files, graph03, frozen, all_reps, outcomes, manifest, strict)
        published['flow'] = _published_funnel(phasee_files['flow'], outcomes, settings['minimum_disclosure_cell'])
        features = pd.read_csv(lineage_files['feature_sets'])
        registry = pd.read_csv(lineage_files['clinical_registry'])
        _require({'representation_id', 'source_column', 'feature', 'included'} <= set(features)
                 and features.included.isin([True, False]).all()
                 and not features.duplicated(['representation_id', 'feature']).any()
                 and 'source_column' in registry, 'AUDIT_FROZEN_LINEAGE_SCHEMA_INVALID')
        data = None
        if mode == 'secure-recompute':
            data, roles = _secure_data(root, release, manifest, sources, graph03, validation, strict)
        tables = _build(reps, outcomes, settings, validation, manifest, features, registry, roles, published, data)
        if strict:
            _require(not any(row['primary_measurement_status'] == 'PUBLISHED_COVERAGE_DIFFERS_FROM_RECOMPUTE'
                             for row in tables[0]), 'AUDIT_PUBLISHED_COVERAGE_DRIFT')
        after = {role: digest(path) for role, path in inputs.items()} | E.frozen_fingerprints(release, manifest)
        _require(before == after, 'AUDIT_INPUTS_CHANGED_DURING_ANALYSIS')

        metadata = baseline_policy().metadata()
        privacy = dict(status=PASS, privacy_class='INTERNAL_RESTRICTED', public_export=False,
                       minimum_disclosure_cell=settings['minimum_disclosure_cell'],
                       complementary_suppression='APPLIED_NO_NEW_COUNTS_OR_SUBGROUP_STATISTICS',
                       patient_rows_exported=False, source_values_exported=False,
                       free_text_exported=False, private_paths_exported=False, **metadata)
        summary = dict(version=VERSION, status='AUDIT_COMPLETE', mode=mode, strict=strict, estimators_run=0,
                       privacy_class=privacy['privacy_class'], public_export=False,
                       representations=reps, outcomes=outcomes, **metadata,
                       graph02_release_sha256=frozen['release_manifest'], frozen_inputs_unchanged=True,
                       sha256_before=before, sha256_after=after,
                       graph03e_status=published_summary['status'] if published_summary['status'] in {
                           'COMPLETE_DESCRIPTIVE_NO_ESTIMABLE_MODELS', 'PARTIAL_WITH_DATA_ISSUES',
                           'COMPLETE_WITH_EXPLORATORY_MODELS'} else UNKNOWN,
                       primary_blockers=dict(Counter(row['primary_blocker'] for row in tables[0])),
                       readiness=dict(Counter(row['readiness_status'] for row in tables[0])))
        _require(not target.exists() or (target.is_dir() and target.stat().st_mode & 0o077 == 0),
                 'AUDIT_OUTPUT_DIRECTORY_PERMISSIONS')
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        with stage_outputs({'stage': target}, lambda _, name: name in OUTPUT_FILES,
                           failure=('stage', '03g_failure.json'), completion=target / '03g_summary.json') as staged:
            directory = staged['stage']
            for name, rows in zip(CSV_OUTPUTS, tables):
                frame = pd.DataFrame(rows)
                if name == 'measurement_funnel':
                    frame = frame.reindex(columns=['representation_id', 'outcome', 'stage', 'source_role', 'status',
                                                   'reason', 'original_reason', 'provenance'])
                if name == 'statistical_support':
                    frame = frame.reindex(columns=['representation_id', 'outcome', 'model', 'status', 'reason',
                                                   'complete_case_status', 'community_balance_status',
                                                   'binary_community_support_status', 'event_parameter_status',
                                                   'residual_df_status', 'parameter_rank_status',
                                                   'outcome_variation_status', 'diagnostic_provenance'])
                frame['window_policy_version'] = metadata['window_policy_version']
                frame['graph02_release_sha256'] = frozen['release_manifest']
                frame.to_csv(directory / f'03g_{name}.csv', index=False)
            write_json(directory / '03g_summary.json', summary)
            write_json(directory / '03g_privacy_qc.json', privacy)
            (directory / '03g_RESULTADOS.md').write_text(_report(tables[0], summary), encoding='utf-8')
    return dict(status=summary['status'], mode=mode, estimators_run=0)

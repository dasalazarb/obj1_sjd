"""Zero-argument 03e analysis of private, frozen Graph02/03 results."""

from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import pandas as pd
import yaml

from .._pipeline import digest
from ..stage02.s02_contracts import SCHEMA as RELEASE_SCHEMA, STATUSES, safe_artifact, validate_release
from . import s03_n3_calendar_day as cal
from .s03_construct_validation import _dependencies
from .s03_n3_approved_view import validate_baseline
from .s03_n3_auto import CONFIGS, PHASEC, VALIDATION, _artifact_inventory, _cached, _locked, _settings
from .s03_n3_exploratory_models import (
    ESTIMATED, INDEPENDENT, UNRESOLVED as LINEAGE_UNRESOLVED, _estimate, _family, _fdr, _independence,
    _model_support,
)
from .s03_n3_exploratory_reporting import (
    SCOPE, TABLE_COLUMNS, _baseline_profiles, _chain, _coverage, _exclusions, _figures, _markdown, _profiles,
    _readme, _small, _source_audit,
)
from .s03_n3_exploratory_source import (
    DATE_FAILURES, EARLY, EDA_ONLY, EXACT, LATE, LINKED, OLDER, ORIGINAL, PATIENT_ONLY, PRE, READ_ERRORS,
    RF_TEXT, RF_VALUE, STRATA, UNRESOLVED, _btris_records, _candidates, _inputs, _outcome, _test_disposition,
)
from .s03_n3_measurement_source import LAB_INPUTS, LAB_SOURCES, _private_directory, _read_frozen
from .s03_n3_resolution import _snapshot
from .s03_release_reader import hard_members
from .s03_run_resolution import graph_paths, resolve_graph03_dependencies, resolve_latest_compatible

MODE = 'EXPLORATORY_ONLY'
DEFAULTS = 'n3_exploratory_defaults.yaml'
LIMITATIONS = [
    'Las asociaciones son exploratorias y están condicionadas a las comunidades congeladas. '
    'Los resultados no constituyen evidencia confirmatoria, causal ni clínicamente validada.',
    'Los intervalos omiten la incertidumbre de Mapper y la selección de comunidades. '
    'R1–R4 son sensibilidades de representación, con R2 de alcance exploratorio.',
    'La indicación clínica de las pruebas, el seguimiento selectivo y los casos completos pueden sesgar los perfiles '
    'y asociaciones. El ajuste se limita a covariables basales verificables.',
    'La fecha de EDA por sí sola tiene menor certeza. ANA y HEp-2 son resultados relacionados y las dos '
    'codificaciones de RF representan una sola familia de prueba.',
]
SCOPES = {
    'models': (ORIGINAL, 'NEW_EXPLORATORY_ASSOCIATION'),
    'sensitivity_models': (EDA_ONLY, 'LOWER_DATE_PROVENANCE_SENSITIVITY'),
    'linkage_sensitivity_models': (ORIGINAL, 'EXACT_EPISODE_LINK_SENSITIVITY'),
}


def _policy(root):
    policy = yaml.safe_load((root / CONFIGS / DEFAULTS).read_text())
    if (policy['schema_version'] != 'graph03_n3_exploratory_v1'
            or set(policy['measurements']['outcomes']) != set(LAB_SOURCES)
            or policy['representations'] != ['R0', 'R1', 'R2', 'R3', 'R4']
            or policy['primary_representation'] != 'R0'
            or policy['minimum_disclosure_cell'] < 5
            or policy['mode'] != 'exploratory_only' or policy['public_export'] is not False):
        raise ValueError('EXPLORATORY_CONFIG_INVALID')
    return policy


def _topology(rep, frame):
    if frame.empty:
        return 'R2_EXPLORATORY_NO_HARD_ASSIGNMENTS' if rep == 'R2' else 'NO_SUPPORTED_HARD_ASSIGNMENTS'
    stable = frame.structural_stability_status.eq('stable').all()
    if rep == 'R2':
        return 'R2_EXPLORATORY' if stable else 'R2_EXPLORATORY_UNSTABLE'
    return 'FROZEN_COMMUNITIES' if stable else 'PROVISIONAL_COMMUNITIES'


def _unavailable(outputs, rep, source, reason):
    for stratum in STRATA:
        for provenance in (ORIGINAL, EDA_ONLY):
            outputs['coverage'].append(dict(representation_id=rep, outcome=source, stratum=stratum,
                                            provenance=provenance, status=reason, definition_role='PRIMARY'))
    for table, (provenance, interpretation) in SCOPES.items():
        outputs[table].append(dict(representation_id=rep, outcome=source, stratum=PRE, provenance=provenance,
                                    interpretation=interpretation, status=reason, not_estimable_reason=reason,
                                    hypothesis_family=_family(source), model_a='A', model_b='B'))


def _patient_flow(outputs, result, source, baseline, cell):
    label = dict(outcome=source, stratum='ALL')
    inventory, selected = result.inventory, result.selected
    linked = inventory.loc[inventory.identity_status.isin(LINKED)]
    dated = linked.loc[~linked.date_evidence.isin(DATE_FAILURES) & linked.stratum.ne(UNRESOLVED)]
    in_window = dated.loc[dated.stratum.ne(OLDER)]
    value_reasons = {'ANALYTE_MAPPING_UNVERIFIED', 'CENSORED_RESULT', 'ADMINISTRATIVE_RESULT',
                     'NOT_EXACT_NUMERIC_RESULT', 'UNKNOWN_RESULT_TOKEN', 'INVALID_RESULT'}
    definition_reasons = {'ORIGINAL_TEST_DEFINITION_UNAVAILABLE', 'NUMERIC_UNIT_UNAVAILABLE'}
    valid = in_window.loc[~in_window.first_reason.isin(value_reasons)]
    comparable = valid.loc[~valid.first_reason.isin(definition_reasons)]
    n_unique = inventory._patient_match.nunique()
    rows = _chain(label, dict(n_source_rows=len(inventory)), cell, unit='source_row')
    if len(inventory) and (n_unique < cell or inventory._patient_match.isna().any()):
        rows[0].update(n_remaining=None, status='SUPPRESSED')
    outputs['cohort_flow'].extend(rows)
    outputs['cohort_flow'].extend(_chain(label, dict(n_frozen_cohort=len(baseline)), cell))
    outputs['cohort_flow'].extend(_chain(label, dict(
        n_distinct_source_patients=inventory.source_patient_id.nunique(),
        n_normalized_source_patients=n_unique, n_identity_linked=linked.patient_id.nunique(),
        n_collection_day_attributable=dated.patient_id.nunique(), n_inside_report_window=in_window.patient_id.nunique(),
        n_value_valid=valid.patient_id.nunique(), n_test_comparable=comparable.patient_id.nunique(),
        n_latest_unambiguous=selected.patient_id.nunique()), cell))
    keys = {
        'identity': {
            'n_exact_episode_links': linked.loc[linked.identity_status.eq(EXACT), 'patient_id'].nunique(),
            'n_unique_patient_only_links': linked.loc[linked.identity_status.eq(PATIENT_ONLY), 'patient_id'].nunique(),
        },
        'provenance': {
            'n_date_original_verified': dated.loc[dated.provenance.eq(ORIGINAL), 'patient_id'].nunique(),
            'n_eda_date_only': dated.loc[dated.provenance.eq(EDA_ONLY), 'patient_id'].nunique(),
        },
        'windows': {name: dated.loc[dated.stratum.eq(stratum), 'patient_id'].nunique() for stratum, name in
                    zip(STRATA, ('n_prebaseline_1_365', 'n_day0', 'n_followup_1_365', 'n_followup_gt365',
                                 'n_older_than_365'))},
    }
    for key, counts in keys.items():
        total = linked.patient_id.nunique() if key == 'identity' else dated.patient_id.nunique()
        for stage, count in counts.items():
            outputs['cohort_flow'].extend(_chain(label, {'denominator': total, stage: count}, cell)[1:])
    ambiguous = inventory.loc[inventory.identity_status.isin(
        {'PATIENT_ID_AMBIGUOUS', 'CROSS_PATIENT_CONFLICT', 'AMBIGUOUS_IDENTITY'})]
    audit = _chain(label, {'denominator': n_unique, 'n_ambiguous': ambiguous._patient_match.nunique()}, cell)[1]
    if ambiguous._patient_match.isna().any():
        audit.update(status='SUPPRESSED', n_previous=None, n_remaining=None, n_excluded=None)
    outputs['cohort_flow'].append(audit)

    for (stratum, provenance), group in dated.groupby(['stratum', 'provenance']):
        if stratum == OLDER:
            outputs['cohort_flow'].extend(_chain(dict(outcome=source, stratum=stratum, provenance=provenance),
                                                  dict(n_outside_window=group.patient_id.nunique()), cell))
            continue
        good = group.loc[~group.first_reason.isin(value_reasons)]
        comparable = good.loc[~good.first_reason.isin(definition_reasons)]
        candidates = selected.loc[selected.stratum.eq(stratum) & selected.provenance.eq(provenance)]
        primary = candidates.loc[candidates.definition_id.eq(result.primary.get((stratum, provenance)))]
        counts = dict(n_collection_day_attributable=group.patient_id.nunique(), n_value_valid=good.patient_id.nunique(),
                      n_test_comparable=comparable.patient_id.nunique(), n_latest_unambiguous=candidates.patient_id.nunique(),
                      n_primary_definition=primary.patient_id.nunique())
        outputs['cohort_flow'].extend(_chain(dict(outcome=source, stratum=stratum, provenance=provenance), counts, cell))


def _lab_slices(result):
    for stratum in STRATA:
        for provenance in (ORIGINAL, EDA_ONLY):
            rows = result.selected.loc[result.selected.stratum.eq(stratum) & result.selected.provenance.eq(provenance)]
            definition = result.primary.get((stratum, provenance))
            if stratum == OLDER:
                rows = result.inventory.loc[result.inventory.stratum.eq(OLDER) &
                                            result.inventory.provenance.eq(provenance)].drop_duplicates('patient_id')
                rows = rows.assign(value=1.)
            main = rows.loc[rows.definition_id.eq(definition)] if definition is not None else rows.iloc[0:0]
            if stratum == OLDER:
                main = rows
            yield stratum, provenance, 'PRIMARY', definition, main
            if definition:
                for alternative, group in rows.loc[rows.definition_id.ne(definition)].groupby('definition_id'):
                    yield stratum, provenance, 'ALTERNATIVE', alternative, group


def _label(rep, source, stratum, provenance, topology, **definition):
    interpretation = SCOPE[stratum]
    if stratum == PRE and provenance == EDA_ONLY:
        interpretation = 'LOWER_DATE_PROVENANCE_SENSITIVITY'
    finding_class = ('EXPLORATORY_SENSITIVITY' if stratum == PRE and (provenance == EDA_ONLY or rep != 'R0')
                     else interpretation)
    return dict(representation_id=rep, outcome=source, stratum=stratum, provenance=provenance,
                interpretation=interpretation, finding_class=finding_class, topology_status=topology, **definition)


def _model(label, frame, independence, prior_rf, table, policy, cell):
    numeric = label['outcome'] == RF_VALUE
    outcome = label['outcome']
    model = dict(**label, hypothesis_family=_family(outcome), model_a='A', model_b='B',
                 family=policy['numeric_family' if numeric else 'binary_family'])
    eligible = independence['classification'] == INDEPENDENT
    support = {}
    model_n = 0
    if table == 'linkage_sensitivity_models':
        using_recovered = frame.value.notna() & frame.identity_status.eq(PATIENT_ONLY)
        if not using_recovered.any():
            return dict(model, status='NOT_APPLICABLE', reason='EXACT_SUBSET_IDENTICAL'), support, model_n
        if _small(int(using_recovered.sum()), cell):
            return dict(model, status='SUPPRESSED', reason='LINKAGE_COMPARISON_SUPPRESSED'), support, model_n
        frame = frame.assign(value=frame.value.where(frame.identity_status.eq(EXACT)))
    if not eligible:
        return dict(model, status=independence['classification'], reason=independence['reason']), support, model_n
    if prior_rf and outcome == RF_TEXT:
        return dict(model, status='ALTERNATIVE_RF_ENCODING_DESCRIPTIVE_ONLY', reason='RF_NUMERIC_MODEL_ESTIMATED'), support, model_n
    if frame.value.notna().sum() == 0:
        return dict(model, status='INSUFFICIENT_SUPPORT', reason='NO_VALID_MEASUREMENTS_IN_SCOPE'), support, model_n
    support, prepared = _model_support(frame, numeric, policy, cell)
    model['covariates'] = support['covariates']
    if 'n' in support:
        model['n'] = support['n']
    if prepared is None:
        return dict(model, status='INSUFFICIENT_SUPPORT', reason=support['reason']), support, model_n
    model_n = len(prepared[0])
    model.update(_estimate(prepared, numeric, policy))
    return model, support, model_n


def _analyze(tables, loaded, records, validation, policy):
    outputs = {name: [] for name in TABLE_COLUMNS}
    frozen = loaded['frozen']
    baseline, members, features = (frozen[key] for key in ('baseline_raw', 'membership', 'feature_sets'))
    registry = loaded['registry']
    roles = {(row['representation_id'], row['source_column']): row for row in loaded['roles'].to_dict('records')}
    dependencies = _dependencies(features, registry, validation)
    reps = policy['representations']
    cell = policy['minimum_disclosure_cell']
    baseline_fields = baseline.reindex(columns=['patient_id', 'clinical_baseline_date', 'demo__age_at_baseline', 'demo__sex'])
    membership = {rep: hard_members(members, rep).merge(baseline_fields, on='patient_id', validate='one_to_one')
                  for rep in loaded['manifest']['eligible_representation_ids']}
    topology = {rep: _topology(rep, frame) for rep, frame in membership.items()}
    protected_hard = {rep for rep, frame in membership.items() if _small(len(baseline) - len(frame), cell)}
    outputs['frozen_baseline_profiles'] = _baseline_profiles(
        baseline, members, {rep: frame[['patient_id', 'community_id', 'structural_stability_status']]
                            for rep, frame in membership.items()}, roles, features, reps, topology, cell,
        bool(loaded.get('qc')))
    outputs['source_linkage_audit'].extend(dict(row, dimension='BTRIS_SCHEMA', status='INVENTORY', count_unit='file')
                                           for row in records.schemas)
    outcomes = [RF_VALUE, RF_TEXT, *(outcome for outcome in policy['measurements']['outcomes']
                                    if outcome not in {RF_VALUE, RF_TEXT})]
    available = {'eda_long', 'step10'} <= set(tables)
    candidates = _candidates(tables, baseline, records, policy) if available else None
    rf_tested = set()
    for source in outcomes:
        result = _outcome(candidates, source, policy) if available else None
        if result is not None:
            if not set(result.selected.patient_id) <= set(baseline.patient_id):
                raise ValueError('EXPLORATORY_MEASUREMENT_EXCEEDS_FROZEN_COHORT')
            _patient_flow(outputs, result, source, baseline, cell)
            outputs['source_linkage_audit'].extend(_source_audit(result.inventory, source, cell))
            for (stratum, provenance), group in result.inventory.groupby(['stratum', 'provenance']):
                outputs['exclusions'].extend(_exclusions(group, dict(outcome=source, stratum=stratum,
                                                                     provenance=provenance), cell))
            disposition = _test_disposition(result, baseline.patient_id)
        else:
            outputs['cohort_flow'].append(dict(outcome=source, stratum='ALL', stage='n_source_rows',
                                               count_unit='source_row', status='DATA_UNAVAILABLE'))
            outputs['cohort_flow'].extend(_chain(dict(outcome=source, stratum='ALL'),
                                                  dict(n_frozen_cohort=len(baseline)), cell))
        for rep in reps:
            if rep not in membership:
                audit = dict(representation_id=rep, outcome=source, classification=LINEAGE_UNRESOLVED,
                             reason='REPRESENTATION_UNAVAILABLE')
            else:
                audit = _independence(rep, source, roles, features, registry, dependencies, validation, policy)
            outputs['independence_audit'].append(audit)
            if rep not in membership or result is None:
                _unavailable(outputs, rep, source, 'REPRESENTATION_UNAVAILABLE' if rep not in membership else
                             'DATA_UNAVAILABLE')
                continue
            for stratum, provenance, definition_role, definition_id, selection in _lab_slices(result):
                label = _label(rep, source, stratum, provenance, topology[rep],
                               definition_role=definition_role, definition_id=definition_id)
                frame = membership[rep].merge(selection[['patient_id', 'value', 'identity_status']], on='patient_id',
                                              how='left', validate='one_to_one')
                observed = int(frame.value.notna().sum())
                if definition_role == 'ALTERNATIVE' and observed < cell:
                    continue
                if observed < cell:
                    label['definition_id'] = None
                elif source == RF_VALUE and definition_id:
                    label['unit'] = selection.unit.iloc[0]
                profiles = _profiles(frame, label, source == RF_VALUE, cell) if stratum != OLDER else []
                if rep in protected_hard and stratum != OLDER:
                    profiles = [dict(**label, status='SUPPRESSED')]
                outputs['community_profiles'].extend(profiles)
                if stratum in {EARLY, LATE}:
                    outputs['followup_profiles'].extend(profiles)
                partition = disposition if (stratum, provenance, definition_role) == (PRE, ORIGINAL, 'PRIMARY') else None
                coverage = _coverage(frame, label, profiles, cell, partition)
                if stratum == OLDER:
                    # Internal subgroup counts still determine whether a total
                    # could expose a small community cell through subtraction.
                    coverage = (dict(**label, status='SUPPRESSED') if rep in protected_hard else
                                _coverage(frame, label, _profiles(frame, label, True, cell), cell))
                outputs['coverage'].append(coverage)
                if stratum != PRE or definition_role != 'PRIMARY':
                    continue
                for table, (model_provenance, interpretation) in SCOPES.items():
                    if provenance != model_provenance:
                        continue
                    model_label = dict(label, interpretation=interpretation)
                    if table != 'models':
                        model_label['finding_class'] = 'EXPLORATORY_SENSITIVITY'
                    model, support, n_model = _model(model_label, frame, audit, (rep, table) in rf_tested,
                                                     table, policy['models'], cell)
                    if model['status'] == ESTIMATED and source in {RF_VALUE, RF_TEXT}:
                        rf_tested.add((rep, table))
                    if coverage['status'] == 'SUPPRESSED' or 'n' not in model:
                        for key in ('n', 'residual_df_b'):
                            model.pop(key, None)
                    model['not_estimable_reason'] = model.get('reason')
                    outputs[table].append(model)
                    selected_n = len(selection)
                    hard_n = int(frame.value.notna().sum())
                    if table == 'linkage_sensitivity_models':
                        hard_n = int((frame.value.notna() & frame.identity_status.eq(EXACT)).sum())
                    independent_n = hard_n if audit['classification'] == INDEPENDENT else 0
                    complete_n = support.get('n_complete_case', 0)
                    stages = dict(n_primary_definition=selected_n, n_hard_community=hard_n,
                                  n_independence_eligible=independent_n)
                    if support or model.get('reason') == 'NO_VALID_MEASUREMENTS_IN_SCOPE':
                        stages.update(n_complete_case=complete_n, n_model_eligible=n_model)
                    model_flow = _chain(model_label, stages, cell)
                    if coverage['status'] == 'SUPPRESSED' or model['status'] == 'SUPPRESSED':
                        for row in model_flow:
                            row.update(status='SUPPRESSED', n_previous=None, n_remaining=None, n_excluded=None,
                                       all_removed_reason=None)
                    outputs['cohort_flow'].extend(model_flow)
    _fdr([row for table in SCOPES for row in outputs[table]])
    _mask_cohort_complements(outputs, len(baseline), cell)
    return outputs


def _mask_cohort_complements(outputs, cohort_n, cell):
    # A frozen cohort count also appears in the summary and baseline figure.
    # Suppress nearby counts even when they are in separate output tables.
    sensitive = set(range(cohort_n - cell + 1, cohort_n)) - {0}
    for row in outputs['cohort_flow']:
        if row['count_unit'] == 'patient' and any(row.get(key) in sensitive for key in ('n_remaining', 'n_previous')):
            row.update(status='SUPPRESSED', n_previous=None, n_remaining=None, n_excluded=None, all_removed_reason=None)
    blocked = {(row.get('outcome'), row.get('source_role'), row['dimension']) for row in outputs['source_linkage_audit']
               if row.get('n_count') in sensitive or row.get('n_total') in sensitive}
    if blocked:
        audit = [row for row in outputs['source_linkage_audit']
                 if (row.get('outcome'), row.get('source_role'), row['dimension']) not in blocked]
        audit.extend(dict(outcome=outcome, source_role=role, dimension=dimension, status='SUPPRESSED')
                     for outcome, role, dimension in sorted(blocked))
        outputs['source_linkage_audit'] = audit
    for row in outputs['exclusions']:
        if any(row.get(key) in sensitive for key in ('n_patients', 'denominator_patients')):
            row.update(status='SUPPRESSED', n_rows=None, n_patients=None, denominator_rows=None, denominator_patients=None)
    for table in SCOPES:
        for row in outputs[table]:
            if row.get('n') in sensitive:
                row.pop('n', None)
                row.pop('residual_df_b', None)


def _empty_outputs(policy):
    outputs = {name: [] for name in TABLE_COLUMNS}
    for rep in policy['representations']:
        for source in policy['measurements']['outcomes']:
            _unavailable(outputs, rep, source, 'DATA_UNAVAILABLE')
            outputs['independence_audit'].append(dict(representation_id=rep, outcome=source,
                                                      classification=LINEAGE_UNRESOLVED,
                                                      reason='FROZEN_BASELINE_UNAVAILABLE'))
    return outputs


def _write_run(staging, target, base, outputs, n, policy):
    files = {name: target / (name + '.csv') for name in TABLE_COLUMNS}
    frames = {name: pd.DataFrame(outputs[name], columns=columns.split()) for name, columns in TABLE_COLUMNS.items()}
    for name, frame in frames.items():
        frame.to_csv(staging / files[name].name, index=False)
    figures = _figures(staging / 'figures', frames)
    models = {table: int(frames[table].status.eq(ESTIMATED).sum()) for table in SCOPES}
    if n is None:
        status = 'INPUTS_UNUSABLE'
    elif base['data_issues']:
        status = 'PARTIAL_WITH_DATA_ISSUES'
    elif any(models.values()):
        status = 'COMPLETE_WITH_EXPLORATORY_MODELS'
    else:
        status = 'COMPLETE_DESCRIPTIVE_NO_ESTIMABLE_MODELS'
    coverage = frames['coverage']
    r0 = coverage.loc[coverage.representation_id.eq(policy['primary_representation']) &
                      coverage.definition_role.eq('PRIMARY')]
    visible = r0.loc[r0.n_measured.notna() & r0.stratum.ne(OLDER)]
    n_outcomes = visible.loc[visible.n_measured.astype(float).gt(0), 'outcome'].nunique()
    by_slice = r0[['outcome', 'stratum', 'provenance', 'status', 'n_measured']]
    summary = dict(**base, status=status, output_dir=str(target), run_id=target.name,
                   privacy_class=policy['privacy_class'], public_export=policy['public_export'],
                   n_frozen_cohort=n if n is None or
                   not _small(n, policy['minimum_disclosure_cell']) else None,
                   n_outcomes_with_data=int(n_outcomes), outcomes_count_scope='R0_DISCLOSABLE_MEASUREMENTS',
                   n_model_pairs_estimable=models['models'],
                   models_by_scope=models, r0_outcomes_by_stratum=json.loads(by_slice.to_json(orient='records')),
                   summary=str(target / 'summary.json'), readme=str(target / 'README.md'),
                   report=str(target / 'RESULTADOS_N3.md'), manifest=str(target / 'results_manifest.json'),
                   figures=[str(target / 'figures' / name) for name in figures],
                   **{name: str(path) for name, path in files.items()})
    (staging / 'summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2) + '\n')
    (staging / 'RESULTADOS_N3.md').write_text(_markdown(summary, frames))
    (staging / 'README.md').write_text(_readme())
    manifest = {'schema_version': policy['schema_version'], 'status': status, 'run_id': target.name,
                'graph02_run': base['graph02_run'], 'graph03_run': base['graph03_run'],
                'provenance': base['provenance'], 'sha256_by_artifact': _artifact_inventory(staging)}
    (staging / 'results_manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')


def _readable_digest(path):
    try:
        return digest(path) if path.is_file() else None
    except OSError:
        return None


def _source_hashes(paths):
    return {role: _readable_digest(paths[role]) for role in LAB_INPUTS}


def _btris_hashes(paths):
    return {role: {str(file): _readable_digest(file) for file in cal.btris_files(paths[role])}
            for role in cal.BTRIS_ROLES}


def _implementation(root):
    graph = root / 'src/studies/graph'
    paths = [*graph.rglob('*.py'), root / 'src/block_A/01_serological_profile.py',
             root / 'common.py', root / 'config.py']
    return cal.fingerprint({path.relative_to(root).as_posix(): digest(path) for path in paths})


def _baseline_artifact(release, manifest):
    filename = manifest['relative_paths']['baseline_raw']
    path = safe_artifact(release.parent, filename)
    if digest(path) != manifest['sha256_by_artifact'][filename]:
        raise ValueError('EXPLORATORY_FROZEN_BASELINE_HASH_MISMATCH')
    baseline = pd.read_csv(path, dtype={'patient_id': 'string', 'clinical_episode_id': 'string'})
    validate_baseline(baseline)
    if baseline.empty or len(baseline) != manifest['n_patients']:
        raise ValueError('EXPLORATORY_FROZEN_BASELINE_INCOMPLETE')
    return baseline


def _load_baseline_only(directory):
    release = directory / '02_graph_release_manifest.json'
    manifest = json.loads(release.read_text())
    if (manifest['schema_version'] != RELEASE_SCHEMA or manifest['finalized'] is not True
            or manifest['run_id'] != directory.name or manifest['release_status'] not in STATUSES - {'QC_FAILED'}):
        raise ValueError('EXPLORATORY_FROZEN_BASELINE_MANIFEST_INVALID')
    qc_file = manifest['relative_paths']['release_qc']
    qc_path = safe_artifact(directory, qc_file)
    if digest(qc_path) != manifest['sha256_by_artifact'][qc_file]:
        raise ValueError('EXPLORATORY_FROZEN_BASELINE_QC_HASH_MISMATCH')
    qc = json.loads(qc_path.read_text())
    if qc.get('primary_qc') != 'PASS' or qc.get('release_status') != manifest['release_status']:
        raise ValueError('EXPLORATORY_FROZEN_BASELINE_QC_FAILED')
    baseline = _baseline_artifact(release, manifest)
    return dict(frozen=dict(baseline_raw=baseline, membership=pd.DataFrame(),
                            feature_sets=pd.DataFrame(columns=['representation_id', 'source_column', 'included'])),
                registry=pd.DataFrame(), roles=pd.DataFrame(), release=release,
                manifest=dict(manifest, eligible_representation_ids=[]),
                data_issues=[dict(status='DATA_UNAVAILABLE', artifact='frozen_graph02_inventory',
                                  reason='ONLY_BASELINE_AND_RELEASE_QC_VERIFIED')])


def _load_graph02(directory):
    manifest, release = validate_release(directory)
    baseline = _baseline_artifact(release, manifest)
    frozen = {role: pd.read_csv(safe_artifact(release.parent, manifest['relative_paths'][role]),
                                dtype={'patient_id': 'string', 'clinical_episode_id': 'string'})
              for role in ('membership', 'feature_sets')}
    frozen['baseline_raw'] = baseline
    members = frozen['membership']
    features = frozen['feature_sets']
    registry = pd.read_csv(safe_artifact(release.parent, manifest['relative_paths']['clinical_registry']))
    # The baseline can still be reported if a frozen membership is unusable.
    flags = ['hard_assignment', 'mapper_covered', 'membership_tie', 'supported']
    required = {*flags, 'patient_id', 'representation_id', 'community_id', 'structural_stability_status'}
    accepted = []
    issues = []
    if required <= set(members) and {'included', 'representation_id', 'source_column'} <= set(features):
        for rep in manifest['eligible_representation_ids']:
            part = members.loc[members.representation_id.eq(rep)]
            if (set(part.patient_id) == set(baseline.patient_id) and
                    part[flags].isin([True, False]).all().all() and
                    not part.duplicated(['patient_id', 'community_id']).any() and
                    not part.loc[part.hard_assignment, 'patient_id'].duplicated().any() and
                    not (part.hard_assignment & (~part.mapper_covered | part.membership_tie)).any()):
                accepted.append(rep)
            else:
                issues.append(dict(status='DATA_UNAVAILABLE', artifact='frozen_membership', reason=rep))
    else:
        issues.append(dict(status='DATA_UNAVAILABLE', artifact='frozen_membership_or_features'))
        features = pd.DataFrame(columns=['representation_id', 'source_column', 'included'])
    return dict(frozen=dict(frozen, feature_sets=features), registry=registry, roles=pd.DataFrame(),
                release=release, manifest=dict(manifest, eligible_representation_ids=accepted), data_issues=issues)


def _discover(root, data, phasec, issues):
    args = SimpleNamespace(graph03_run=None, graph02_release=None, strict=False, config=phasec)
    try:
        graph03, release, _, selection = resolve_graph03_dependencies(
            args, root, phase='C', data_root=data, quiet=True, verify_inventory=True)
    except RuntimeError:
        issues.append(dict(status='DATA_UNAVAILABLE', artifact='compatible_graph02_graph03_releases'))
        for reader in (_load_graph02, _load_baseline_only):
            try:
                _, bundle, evidence = resolve_latest_compatible(
                    'Graph02', [graph_paths(root, data_root=data)['graph02']], ['02_graph_release_manifest.json'],
                    '02_graph_release_manifest.json', reader)
                return bundle['release'], None, evidence['run_id'], None
            except RuntimeError:
                continue
        issues.append(dict(status='DATA_UNAVAILABLE', artifact='frozen_graph02_baseline',
                           reason='NO_VERIFIED_GRAPH02_BASELINE'))
        return None, None, None, None
    return release, graph03, selection[0]['run_id'], selection[1]['run_id']


def _load(release, graph03, phasec, issues):
    if release is None:
        return None
    if graph03 is not None:
        try:
            loaded = _read_frozen(dict(graph03_run=graph03, graph02_release=release, baseline=None),
                                  yaml.safe_load(phasec.read_text()))
            if not loaded['frozen']['baseline_raw'].empty:
                return loaded
        except READ_ERRORS:
            issues.append(dict(status='DATA_UNAVAILABLE', artifact='frozen_graph03_inventory'))
    for reader in (_load_graph02, _load_baseline_only):
        try:
            loaded = reader(release.parent)
            issues.extend(loaded.pop('data_issues'))
            return loaded
        except READ_ERRORS:
            continue
    issues.append(dict(status='DATA_UNAVAILABLE', artifact='frozen_graph02_baseline',
                       reason='NO_VERIFIED_GRAPH02_BASELINE'))
    return None


def _latest(folder, output):
    payload = {'run_id': output.name, 'relative_path': 'runs/' + output.name,
               'results_manifest_sha256': digest(output / 'results_manifest.json')}
    with NamedTemporaryFile(mode='w', prefix='.latest-', dir=folder, delete=False) as handle:
        temp = Path(handle.name)
        try:
            json.dump(payload, handle, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
            temp.replace(folder / 'latest.json')
        finally:
            temp.unlink(missing_ok=True)


def run_exploratory_auto(root):
    """Publish one verified, private aggregate release using automatic discovery."""
    root = Path(root).resolve()
    issues = []
    base = dict(mode=MODE, graph02_run=None, graph03_run=None, data_issues=issues, limitations=LIMITATIONS)
    try:
        data, private, paths = _settings(root)
        _private_directory(private)
        folder = private / 'exploratory'
        with _locked(folder):
            phasec, validation = (root / CONFIGS / name for name in (PHASEC, VALIDATION))
            config_paths = [phasec, validation, *(root / CONFIGS / name for name in
                                                 (DEFAULTS, 'n3_auto.yaml', 'pipeline.yaml'))]
            config_sha = {file.name: digest(file) for file in config_paths}
            policy = _policy(root)
            release, graph03, base['graph02_run'], base['graph03_run'] = _discover(root, data, phasec, issues)
            frozen_roots = [path for path in (release.parent if release else None, graph03) if path]
            if any(private.is_relative_to(path) or path.is_relative_to(private) for path in frozen_roots):
                raise ValueError('EXPLORATORY_OUTPUT_OVERLAPS_FROZEN_INPUT')
            frozen_sha = _snapshot(frozen_roots)
            sources_sha, btris_sha = _source_hashes(paths), _btris_hashes(paths)
            code_sha = _implementation(root)
            versions = {name: version(name) for name in ('pandas', 'numpy', 'scipy', 'statsmodels', 'pyarrow', 'matplotlib', 'PyYAML')}
            versions['python'] = platform.python_version()
            provenance = dict(frozen_manifest_sha256={path.name: sha for path, sha in frozen_sha.items()
                              if path.name in {'02_graph_release_manifest.json', '03_graph_results_summary.json'}},
                              frozen_inventory_sha256=cal.fingerprint([
                                  {path.relative_to(directory).as_posix(): sha for path, sha in frozen_sha.items()
                                   if path.is_relative_to(directory)} for directory in frozen_roots]),
                              source_sha256=sources_sha, btris_sha256={
                                  role: sorted(sha for sha in files.values() if sha is not None)
                                  for role, files in btris_sha.items()},
                              config_sha256=config_sha, implementation_sha256=code_sha, versions=versions)
            base['provenance'] = provenance

            def verify():
                if (frozen_sha != _snapshot(frozen_roots) or sources_sha != _source_hashes(paths)
                        or btris_sha != _btris_hashes(paths) or code_sha != _implementation(root)
                        or any(config_sha[file.name] != digest(file) for file in config_paths)):
                    raise ValueError('EXPLORATORY_INPUT_CHANGED_DURING_RUN')

            def build(staging, target):
                tables = _inputs(paths, issues, sources_sha)
                loaded = _load(release, graph03, phasec, issues)
                if loaded is None:
                    _write_run(staging, target, base, _empty_outputs(policy), None, policy)
                    return
                integrated = loaded['manifest']['source_inputs'].get('integrated')
                if integrated and integrated['sha256'] != sources_sha['step10']:
                    issues.append(dict(status='DATA_UNAVAILABLE', artifact='step10',
                                       reason='FROZEN_GRAPH02_INPUT_SHA_MISMATCH'))
                    tables.pop('step10', None)
                btris_files = {role: [Path(file) for file, sha in files.items() if sha is not None]
                               for role, files in btris_sha.items()}
                for role, files in btris_sha.items():
                    if files and btris_files[role] and None in files.values():
                        issues.append(dict(status='DATA_UNAVAILABLE', artifact=role, reason='SOURCE_HASH_UNAVAILABLE'))
                btris = _btris_records(btris_files, issues, tables.get('eda_long', pd.DataFrame()))
                outputs = _analyze(tables, loaded, btris, yaml.safe_load(validation.read_text()), policy)
                _write_run(staging, target, base, outputs, len(loaded['frozen']['baseline_raw']), policy)

            cache_key = dict(provenance=provenance, graph02_run=base['graph02_run'], graph03_run=base['graph03_run'],
                             btris_inventory=cal.fingerprint(btris_sha))
            output = _cached(folder / 'runs', cache_key, build, verify, verify_reuse=True)
            summary = json.loads((output / 'summary.json').read_text())
            if summary['status'] != 'INPUTS_UNUSABLE':
                _latest(folder, output)
            keys = ('status', 'mode', 'graph02_run', 'graph03_run', 'n_outcomes_with_data', 'n_model_pairs_estimable',
                    'models_by_scope', 'output_dir', 'summary', 'models', 'report', 'manifest', 'figures',
                    'limitations', 'data_issues')
            return {key: summary[key] for key in keys}
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        token = str(exc)
        reason = token if re.fullmatch(r'(?:EXPLORATORY_[A-Z_]+|UNSAFE_AUTO_[A-Z_]+|AUTO_RUN_IN_PROGRESS)', token) else (
            'EXPLORATORY_RUN_UNAVAILABLE')
        return dict(**base, status='EXPLORATORY_RUN_FAILED', output_dir=None, reason=reason)

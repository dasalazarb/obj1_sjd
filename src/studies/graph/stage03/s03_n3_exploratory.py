"""Calendar-day exploratory analysis of current Graph02/03 results."""

from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import re

import pandas as pd
import yaml

from .._pipeline import config_path, digest, output_lock, output_root, stage_destinations, stage_outputs, write_json
from ..stage02.s02_contracts import CURRENT_SCHEMA, MANIFEST, artifact_paths, safe_artifact, validate_release
from . import s03_n3_calendar_day as cal
from .s03_baseline_window import (
    AFTER_BASELINE_WINDOW, DAY_BINS, FROZEN_SNAPSHOT, MISMATCH, PRIMARY, WARNING, WINDOW_SNAPSHOT,
    policy as baseline_policy, require_policy,
)
from .s03_construct_validation import _dependencies, feature_roles
from .s03_n3_approved_view import validate_baseline
from .s03_n3_auto import CONFIGS, VALIDATION, _source_settings
from .s03_n3_exploratory_models import (
    ESTIMATED, INDEPENDENT, UNRESOLVED as LINEAGE_UNRESOLVED, _estimate, _family, _fdr, _independence,
    _adjustment, _model_support,
)
from .s03_n3_exploratory_reporting import (
    SCOPE, TABLE_COLUMNS, _baseline_profiles, _chain, _coverage, _exclusions, _figures, _markdown, _partition, _profiles,
    _small, _source_audit,
)
from .s03_n3_exploratory_source import (
    DATE_FAILURES, EDA_ONLY, EXACT, EXACT_WINDOW, LINKED, ORIGINAL, PATIENT_ONLY, READ_ERRORS,
    RF_TEXT, RF_VALUE, SENSITIVITIES, STRATA, UNRESOLVED, _btris_records, _candidates, _inputs,
    _outcome, _test_disposition, primary_measurements,
)
from .s03_n3_measurement_source import LAB_SOURCES
from .s03_release_reader import RESULTS, RESULTS_SCHEMA, hard_members
from .s03_run_resolution import graph_paths

MODE = 'calendar_day_exploratory'
BASELINE_ARTIFACTS = {
    'baseline_eligibility': 'baseline_eligibility',
    'measurement_delay': 'baseline_measurement_delay',
    'baseline_inference': 'n3_baseline_inference',
    'independence_audit': 'n3_role_lineage_audit',
    'baseline_profiles': 'n3_baseline_community_profiles',
    'temporal_overlap': 'temporal_overlap_audit',
}
DEFAULTS = 'n3_exploratory_defaults.yaml'
LIMITATIONS = [
    WARNING,
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
    require_policy(policy)
    if policy.get('time') or 'main_window' in policy.get('measurements', {}):
        raise ValueError(MISMATCH)
    if (policy['schema_version'] != 'graph03_n3_exploratory_v2'
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
    label = dict(representation_id=rep, outcome=source, stratum=PRIMARY, provenance=ORIGINAL,
                 status=reason, definition_role='PRIMARY')
    for name in ('community_profiles', 'baseline_eligibility', 'measurement_delay'):
        outputs[name].append(label.copy())
    for table, (provenance, interpretation) in SCOPES.items():
        outputs[table].append(dict(representation_id=rep, outcome=source, stratum=PRIMARY, provenance=provenance,
                                    interpretation=interpretation, status=reason, not_estimable_reason=reason,
                                    hypothesis_family=_family(source), model_a='A', model_b='B'))


def _patient_flow(outputs, result, source, baseline, cell):
    label = dict(outcome=source, stratum='ALL')
    inventory = result.inventory
    linked = inventory.loc[inventory.identity_status.isin(LINKED)]
    counts = _eligibility_counts(inventory, result)
    n_unique = inventory._patient_match.nunique()
    rows = _chain(label, dict(n_source_rows=len(inventory)), cell, unit='source_row')
    if len(inventory) and (n_unique < cell or inventory._patient_match.isna().any()):
        rows[0].update(n_remaining=None, status='SUPPRESSED')
    outputs['cohort_flow'].extend(rows)
    outputs['cohort_flow'].extend(_chain(label, dict(n_frozen_cohort=len(baseline)), cell))
    outputs['cohort_flow'].extend(_chain(label, dict(
        n_distinct_source_patients=inventory.source_patient_id.nunique(),
        n_normalized_source_patients=n_unique, **counts), cell))
    for stratum, name in zip(DAY_BINS, ('pre', 'day0', 'day1_30', 'day31_90', 'day91_to_6m', 'after_6m', 'unknown')):
        count = linked.loc[linked.window_bin.eq(stratum), 'patient_id'].nunique()
        outputs['cohort_flow'].extend(_chain(label, {'denominator': linked.patient_id.nunique(),
                                                    'n_' + name: count}, cell)[1:])

    ambiguous = inventory.loc[inventory.identity_status.isin(
        {'PATIENT_ID_AMBIGUOUS', 'CROSS_PATIENT_CONFLICT', 'AMBIGUOUS_IDENTITY'})]
    audit = _chain(label, {'denominator': n_unique, 'n_ambiguous': ambiguous._patient_match.nunique()}, cell)[1]
    if ambiguous._patient_match.isna().any():
        audit.update(status='SUPPRESSED', n_previous=None, n_remaining=None, n_excluded=None)
    outputs['cohort_flow'].append(audit)


def _eligibility_counts(inventory, result):
    linked = inventory.loc[inventory.identity_status.isin(LINKED)]
    temporal = linked.loc[linked.stratum.eq(PRIMARY)]
    verified = temporal.loc[temporal.provenance.eq(ORIGINAL)]
    selected = primary_measurements(result)
    comparable = verified.loc[verified.first_reason.isin(('', 'IDENTICAL_DUPLICATE', 'LATER_ELIGIBLE_DAY',
                                                         'AMBIGUOUS_EARLIEST_DAY', 'CONSISTENT_EARLIEST_DAY_REPEAT'))]
    return dict(n_patients_with_any_candidate=linked.patient_id.nunique(),
                n_temporally_eligible=temporal.patient_id.nunique(),
                n_date_verified=verified.patient_id.nunique(),
                n_assay_comparable=comparable.patient_id.nunique(),
                n_earliest_unambiguous=selected.loc[selected.patient_id.isin(inventory.patient_id), 'patient_id'].nunique())


def _baseline_support(outputs, result, members, label, cell, coverage, models, audit):
    selected = primary_measurements(result)
    counts_by_community, delays = [], []
    for community, group in members.groupby('community_id'):
        ids = group.patient_id
        candidates = result.inventory.loc[result.inventory.patient_id.isin(ids)]
        measurements = group.merge(selected[['patient_id', 'value', 'window_bin']], on='patient_id',
                                   how='left', validate='one_to_one')
        counts = dict(n_representation_patients=len(group), **_eligibility_counts(candidates, result))
        counts['n_hard_supported'] = counts['n_earliest_unambiguous']
        complete = _adjustment(measurements).notna().all(axis=1) & measurements.value.notna()
        counts['n_complete_case'] = int(complete.sum())
        # Model eligibility is assessed at the representation level, since communities
        # must pass the same support and covariate gates on a common population.
        counts_by_community.extend(_chain(dict(label, community_id=community), counts, cell))
        bins = measurements.window_bin.fillna('NO_PRIMARY_MEASUREMENT')
        partition = _partition(bins.value_counts().to_dict(), cell)
        if partition is None or len(group) < cell:
            delays.append(dict(label, community_id=community, count_unit='patient', status='SUPPRESSED'))
        else:
            delays.extend(dict(label, community_id=community, window_bin=bin_name, status='DESCRIPTIVE',
                               count_unit='patient', n_total=len(group), n_patients=count)
                          for bin_name, count in sorted(partition.items()))
    if coverage['status'] == 'SUPPRESSED' or any(row['status'] == 'SUPPRESSED' for row in counts_by_community + delays):
        # Totals and complementary communities can reconstruct suppressed losses.
        withheld = [dict(label, count_unit='patient', status='SUPPRESSED')]
        outputs['baseline_eligibility'].extend(withheld)
        outputs['measurement_delay'].extend(withheld)
        return
    counts = _eligibility_counts(result.inventory.loc[result.inventory.patient_id.isin(members.patient_id)], result)
    counts = dict(n_representation_patients=len(members), **counts)
    counts['n_hard_supported'] = counts['n_earliest_unambiguous']
    frame = members.merge(selected[['patient_id', 'value']], on='patient_id', how='left', validate='one_to_one')
    support, prepared = _model_support(frame, label['outcome'] == RF_VALUE, models, cell)
    counts.update(n_complete_case=support['n_complete_case'],
                  n_model_eligible=len(prepared[0]) if prepared is not None and audit['classification'] == INDEPENDENT else 0)
    outputs['baseline_eligibility'].extend(_chain(label, counts, cell) + counts_by_community)
    outputs['measurement_delay'].extend(delays)


def _lab_slices(result):
    for stratum in STRATA:
        for provenance in (ORIGINAL, EDA_ONLY):
            rows = result.selected.loc[result.selected.stratum.eq(stratum) & result.selected.provenance.eq(provenance)]
            definition = result.primary.get((stratum, provenance))
            main = rows.loc[rows.definition_id.eq(definition)] if definition is not None else rows.iloc[0:0]
            yield stratum, provenance, 'PRIMARY', definition, main
            if definition:
                for alternative, group in rows.loc[rows.definition_id.ne(definition)].groupby('definition_id'):
                    yield stratum, provenance, 'ALTERNATIVE', alternative, group
    for stratum, rows in result.sensitivities.items():
        yield stratum, ORIGINAL, 'PRIMARY', result.primary.get((PRIMARY, ORIGINAL)), rows


def _label(rep, source, stratum, provenance, topology, **definition):
    interpretation = SCOPE[stratum]
    if stratum == PRIMARY and provenance == EDA_ONLY:
        interpretation = 'LOWER_DATE_PROVENANCE_SENSITIVITY'
    finding_class = ('EXPLORATORY_SENSITIVITY' if stratum == PRIMARY and (provenance == EDA_ONLY or rep != 'R0')
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


def _analyze(tables, loaded, records, validation, policy, *, audit_only=False, selected_measurements=None):
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
            selection = primary_measurements(result)
            if (selection.patient_id.duplicated().any() or selection.baseline_offset_days.lt(0).any()
                    or not selection.window_bin.isin(DAY_BINS[1:5]).all()):
                raise ValueError('EXPLORATORY_BASELINE_SELECTION_INVALID')
            if selected_measurements is not None:
                selected_measurements[source] = selection
        else:
            outputs['cohort_flow'].append(dict(outcome=source, stratum='ALL', stage='n_source_rows',
                                               count_unit='source_row', status='DATA_UNAVAILABLE'))
            outputs['cohort_flow'].extend(_chain(dict(outcome=source, stratum='ALL'),
                                                  dict(n_frozen_cohort=len(baseline)), cell))
        if audit_only:
            continue
        for rep in reps:
            if rep not in membership:
                audit = dict(representation_id=rep, outcome=source, classification=LINEAGE_UNRESOLVED,
                             reason='REPRESENTATION_UNAVAILABLE')
            else:
                audit = _independence(rep, source, roles, features, registry, dependencies, validation, policy)
            outputs['independence_audit'].append(audit)
            role = roles.get((rep, source), {})
            reviews = validation.get('characterization', {}).get('independence_reviews', [])
            prespecs = validation.get('n3_validation', {}).get('prespecified_outcomes', [])
            in_scope = lambda row: row.get('representation_id') == rep and row.get('source_column') == source
            approved = (role.get('tier') == 'N3_REVIEWED' and any(
                in_scope(row) and row.get('window_policy_version') == baseline_policy().policy_version
                for row in reviews) and any(
                in_scope(row) and row.get('window_policy_version') == baseline_policy().policy_version
                for row in prespecs))
            if audit['classification'] == INDEPENDENT and not approved:
                audit = dict(audit, classification=LINEAGE_UNRESOLVED, reason='INDEPENDENCE_REVIEW_PENDING')
                outputs['independence_audit'][-1] = audit
            if audit['classification'] == INDEPENDENT and (
                    topology.get(rep) not in {'FROZEN_COMMUNITIES', 'R2_EXPLORATORY'} or
                    loaded['manifest'].get('release_status') == 'NO_STABLE_STRUCTURE'):
                audit = dict(audit, classification='NOT_ESTIMABLE', reason='TOPOLOGY_NOT_STABLE')
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
                protected_sensitivity = False
                if stratum in SENSITIVITIES:
                    compared = primary_measurements(result)
                    keys = ['patient_id', 'collection_calendar_day', 'value']
                    compared = compared[keys].merge(selection[keys], how='outer', on=keys, indicator=True)
                    changed = compared.loc[compared['_merge'].ne('both'), 'patient_id']
                    protected_sensitivity = _small(changed.loc[changed.isin(frame.patient_id)].nunique(), cell)
                if definition_role == 'ALTERNATIVE' and observed < cell:
                    continue
                if observed < cell:
                    label['definition_id'] = None
                elif source == RF_VALUE and definition_id:
                    label['unit'] = selection.unit.iloc[0]
                profiles = _profiles(frame, label, source == RF_VALUE, cell)
                if rep in protected_hard or protected_sensitivity:
                    profiles = [dict(**label, status='SUPPRESSED')]
                outputs['community_profiles'].extend(profiles)
                if stratum == AFTER_BASELINE_WINDOW:
                    outputs['followup_profiles'].extend(profiles)
                partition = disposition if (stratum, provenance, definition_role) == (PRIMARY, ORIGINAL, 'PRIMARY') else None
                coverage = _coverage(frame, label, profiles, cell, partition)
                outputs['coverage'].append(coverage)
                if stratum == PRIMARY and provenance == ORIGINAL and definition_role == 'PRIMARY':
                    _baseline_support(outputs, result, membership[rep], label, cell, coverage, policy['models'], audit)
                if definition_role != 'PRIMARY':
                    continue
                for table, (model_provenance, interpretation) in SCOPES.items():
                    model_stratum = EXACT_WINDOW if table == 'linkage_sensitivity_models' else PRIMARY
                    if provenance != model_provenance or stratum != model_stratum:
                        continue
                    if table == 'linkage_sensitivity_models' and selection[
                            ['patient_id', 'collection_calendar_day', 'value']].reset_index(drop=True).equals(
                            primary_measurements(result)[['patient_id', 'collection_calendar_day', 'value']].reset_index(drop=True)):
                        result_label = dict(label, interpretation=interpretation, status='NOT_APPLICABLE',
                                            reason='EXACT_SUBSET_IDENTICAL', not_estimable_reason='EXACT_SUBSET_IDENTICAL',
                                            hypothesis_family=_family(source))
                        outputs[table].append(result_label)
                        continue
                    model_label = dict(label, interpretation=interpretation)
                    if table != 'models':
                        model_label['finding_class'] = 'EXPLORATORY_SENSITIVITY'
                    model, support, n_model = _model(model_label, frame, audit, (rep, table) in rf_tested,
                                                     table, policy['models'], cell)
                    if protected_sensitivity or (coverage['status'] == 'SUPPRESSED' and model['status'] == ESTIMATED):
                        model = dict(model_label, status='SUPPRESSED', reason='COVERAGE_DISCLOSURE_RESTRICTION',
                                     hypothesis_family=_family(source))
                    if model['status'] == ESTIMATED and source in {RF_VALUE, RF_TEXT}:
                        rf_tested.add((rep, table))
                    if coverage['status'] == 'SUPPRESSED' or 'n' not in model:
                        for key in ('n', 'residual_df_b'):
                            model.pop(key, None)
                    model['not_estimable_reason'] = model.get('reason')
                    outputs[table].append(model)
                    selected_n = len(selection)
                    hard_n = int(frame.value.notna().sum())
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
    for table in SCOPES:
        _fdr(outputs[table])
    _mask_cohort_complements(outputs, len(baseline), cell)
    outputs['baseline_inference'] = [
        dict(row, status=row['status'] if row['status'] == ESTIMATED else 'NOT_ESTIMABLE')
        for row in outputs['models']]
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
    outputs['baseline_inference'] = [dict(row, status='NOT_ESTIMABLE') for row in outputs['models']]
    return outputs


def current_source_inputs(root, *, step10=None, context=None, paths=None):
    """Read the same immutable clinical anchor and original sources for every lab caller."""
    if paths is None:
        _, paths = _source_settings(root)
    else:
        paths = dict(paths)
    if step10 is not None:
        configured_step10 = paths['step10']
        paths['step10'] = Path(step10)
        if paths['step10'] != configured_step10:
            paths['context'] = paths['step10'].with_name('10_integrated_longitudinal_context.parquet')
    if context is not None:
        paths['context'] = Path(context)
    issues = []
    tables = _inputs(paths, issues)
    btris_files = {role: cal.btris_files(paths[role]) for role in cal.BTRIS_ROLES}
    records = _btris_records(btris_files, issues, tables.get('eda_long', pd.DataFrame()))
    return tables, records, issues


def frozen_fingerprints(release, manifest):
    artifacts = artifact_paths(manifest)
    values = {'release_manifest': digest(release)}
    for role in ('baseline_raw', 'membership', 'feature_sets'):
        if role in artifacts:
            artifact = release.parent / artifacts[role]
            if artifact.is_file():
                values[role] = digest(safe_artifact(release.parent, artifacts[role]))
    return values


def write_baseline_artifacts(staging, outputs, issues, frozen_before, frozen_after, *, temporal_overlap=None,
                             roles=None):
    if frozen_before != frozen_after:
        raise ValueError('EXPLORATORY_GRAPH02_CHANGED_DURING_ANALYSIS')
    metadata = baseline_policy().metadata()
    write_json(staging / '03_graph_baseline_window_policy.json', {
        **asdict(baseline_policy()), **metadata, 'warning': WARNING})
    profiles = [row for row in outputs['community_profiles'] if row.get('stratum') == PRIMARY and
                row.get('provenance') == ORIGINAL]
    if temporal_overlap is None:
        temporal_overlap = pd.DataFrame([dict(status='NOT_ASSESSED', reason='SEE_GRAPH03_LONGITUDINAL_AUDIT')])
    for name, suffix in BASELINE_ARTIFACTS.items():
        if name == 'temporal_overlap':
            table = temporal_overlap.copy()
        elif name == 'baseline_profiles':
            table = pd.DataFrame(profiles, columns=TABLE_COLUMNS['community_profiles'].split())
        else:
            table = pd.DataFrame(outputs[name], columns=TABLE_COLUMNS[name].split())
        if name == 'independence_audit' and roles is not None and not roles.empty:
            fields = ['representation_id', 'source_column', 'tier', 'direct_in_discovery',
                      'conflicting_defining_sources', 'lineage_review_status', 'clinical_review_status',
                      'temporal_baseline_status', 'lineage_evidence', 'source_reference']
            lineage = roles.reindex(columns=fields).rename(columns={'source_column': 'outcome'})
            table = table.merge(lineage, on=['representation_id', 'outcome'], how='outer', validate='one_to_one')
            table['snapshot_basis'] = table.tier.map(
                lambda tier: FROZEN_SNAPSHOT if tier in ('N1_DEFINING', 'N2_RELATED') else WINDOW_SNAPSHOT)
        else:
            table['snapshot_basis'] = WINDOW_SNAPSHOT
        for key, value in metadata.items():
            table[key] = value
        if 'provenance' in table:
            table['source_provenance_tier'] = table.provenance
        table['graph02_release_sha256'] = frozen_before.get('release_manifest')
        table.to_csv(staging / ('03_graph_' + suffix + '.csv'), index=False)
    qc = dict(**metadata, status='FAILED' if not frozen_before else 'PARTIAL' if issues else 'PASS',
              graph02_sha256_before=frozen_before, graph02_sha256_after=frozen_after,
              graph02_frozen_unchanged=bool(frozen_before and frozen_before == frozen_after),
              n_estimated_primary_pairs=sum(row['status'] == ESTIMATED for row in outputs['baseline_inference']),
              data_issues=issues, privacy_class='INTERNAL_RESTRICTED', public_export=False)
    write_json(staging / '03_graph_baseline_window_qc.json', qc)
    return qc


def _write_run(staging, target, base, outputs, n, policy, figure_dir, figure_target):
    files = {name: target / ('03e_' + name + '.csv') for name in TABLE_COLUMNS}
    frames = {name: pd.DataFrame(outputs[name], columns=columns.split()) for name, columns in TABLE_COLUMNS.items()}
    for name, frame in frames.items():
        for key, value in baseline_policy().metadata().items():
            frame[key] = value
        if 'provenance' in frame:
            frame['source_provenance_tier'] = frame.provenance
        if name == 'frozen_baseline_profiles':
            frame['snapshot_basis'] = FROZEN_SNAPSHOT
        frame.to_csv(staging / files[name].name, index=False)
    figures = _figures(figure_dir, frames)
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
    visible = r0.loc[r0.n_measured.notna() & r0.stratum.eq(PRIMARY) & r0.provenance.eq(ORIGINAL)]
    n_outcomes = visible.loc[visible.n_measured.astype(float).gt(0), 'outcome'].nunique()
    summary = dict(**base, status=status, output_dir=str(target),
                   privacy_class=policy['privacy_class'], public_export=policy['public_export'],
                   n_frozen_cohort=n if n is None or
                   not _small(n, policy['minimum_disclosure_cell']) else None,
                   n_outcomes_with_data=int(n_outcomes), outcomes_count_scope='R0_DISCLOSABLE_MEASUREMENTS',
                   n_model_pairs_estimable=models['models'],
                   models_by_scope=models,
                   summary=str(target / '03e_summary.json'),
                   report=str(target / '03e_RESULTADOS.md'),
                   figures=[str(figure_target / name) for name in figures],
                   baseline_window_qc=str(target / '03_graph_baseline_window_qc.json'),
                   baseline_window_policy=str(target / '03_graph_baseline_window_policy.json'),
                   **{name: str(path) for name, path in files.items()})
    write_json(staging / '03e_summary.json', summary)
    (staging / '03e_RESULTADOS.md').write_text(_markdown(summary, frames))
    return summary



def _baseline_artifact(release, manifest):
    path = safe_artifact(release.parent, artifact_paths(manifest)['baseline_raw'])
    baseline = pd.read_csv(path, dtype={'patient_id': 'string', 'clinical_episode_id': 'string',
                                        'clinical_baseline_episode_id': 'string'})
    validate_baseline(baseline)
    if baseline.empty or len(baseline) != manifest['n_patients']:
        raise ValueError('EXPLORATORY_FROZEN_BASELINE_INCOMPLETE')
    return baseline


def _load_baseline_only(directory):
    manifest, release = validate_release(directory / MANIFEST, baseline_only=True)
    if manifest['schema_version'] != CURRENT_SCHEMA:
        raise ValueError('CURRENT_GRAPH02_BASELINE_REQUIRED')
    baseline = _baseline_artifact(release, manifest)
    return dict(frozen=dict(baseline_raw=baseline, membership=pd.DataFrame(),
                            feature_sets=pd.DataFrame(columns=['representation_id', 'source_column', 'feature', 'included'])),
                registry=pd.DataFrame(), roles=pd.DataFrame(), release=release,
                manifest=dict(manifest, eligible_representation_ids=[]),
                data_issues=[dict(status='DATA_UNAVAILABLE', artifact='frozen_graph02_inventory',
                                  reason='ONLY_BASELINE_AND_RELEASE_QC_VERIFIED')])


def _load_graph02(directory):
    manifest, release = validate_release(directory / MANIFEST)
    if manifest['schema_version'] != CURRENT_SCHEMA:
        raise ValueError('CURRENT_GRAPH02_REQUIRED')
    artifacts = artifact_paths(manifest)
    baseline = _baseline_artifact(release, manifest)
    frozen = {role: pd.read_csv(safe_artifact(release.parent, artifacts[role]),
                                dtype={'patient_id': 'string', 'clinical_episode_id': 'string'})
              for role in ('membership', 'feature_sets')}
    frozen['baseline_raw'] = baseline
    members = frozen['membership']
    features = frozen['feature_sets']
    registry = pd.read_csv(safe_artifact(release.parent, artifacts['clinical_registry']))
    # The baseline can still be reported if a frozen membership is unusable.
    flags = ['hard_assignment', 'mapper_covered', 'membership_tie', 'supported']
    required = {*flags, 'patient_id', 'representation_id', 'community_id', 'structural_stability_status'}
    accepted = []
    issues = []
    if (required <= set(members) and {'included', 'representation_id', 'source_column', 'feature'} <= set(features)
            and features.included.isin([True, False]).all()):
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
        features = pd.DataFrame(columns=['representation_id', 'source_column', 'feature', 'included'])
    return dict(frozen=dict(frozen, feature_sets=features), registry=registry, roles=pd.DataFrame(),
                release=release, manifest=dict(manifest, eligible_representation_ids=accepted), data_issues=issues)



def _graph03_qc(directory, loaded, roles):
    """Check current characterization against discovery before using its context."""
    summary = json.loads(safe_artifact(directory, RESULTS).read_text())
    require_policy(summary)
    if (summary.get('window_policy_sha256') != baseline_policy().sha256
            or summary.get('graph02_release_sha256') != digest(loaded['release'])):
        raise ValueError(MISMATCH)
    if (not isinstance(summary, dict) or summary.get('schema_version') != RESULTS_SCHEMA
            or summary.get('execution_status') not in {'COMPLETE', 'COMPLETE_WITH_LIMITATIONS'}
            or summary.get('n_patients') != len(loaded['frozen']['baseline_raw'])):
        raise ValueError('GRAPH03_COHORT_OR_COMPLETION_MISMATCH')
    qc = json.loads(safe_artifact(directory, '03_graph_release_input_qc.json').read_text())
    if (not isinstance(qc, dict) or qc.get('status') != 'PASS'
            or qc.get('eligible_representation_ids') != loaded['manifest']['eligible_representation_ids']):
        raise ValueError('GRAPH03_REPRESENTATION_MISMATCH')
    baseline_hashes = frozen_fingerprints(loaded['release'], loaded['manifest'])
    window_qc = json.loads(safe_artifact(directory, '03_graph_baseline_window_qc.json').read_text())
    if (window_qc.get('window_policy_sha256') != baseline_policy().sha256
            or window_qc.get('graph02_sha256_after') != baseline_hashes):
        raise ValueError(MISMATCH)
    keys = ['representation_id', 'source_column', 'direct_in_discovery', 'conflicting_defining_sources']
    previous = pd.read_csv(safe_artifact(directory, '03_graph_clinical_feature_roles.csv'), keep_default_na=False)
    if sorted(previous[keys].itertuples(index=False, name=None)) != sorted(roles[keys].itertuples(index=False, name=None)):
        raise ValueError('GRAPH03_FEATURE_LINEAGE_MISMATCH')
    inventory = pd.read_csv(safe_artifact(directory, '03_graph_n3_source_inventory.csv'))
    for rep in loaded['manifest']['eligible_representation_ids']:
        members = hard_members(loaded['frozen']['membership'], rep)
        rows = inventory.loc[inventory.representation_id.eq(rep)]
        stable = members.structural_stability_status.eq('stable').sum()
        if rows.empty or not rows.n_hard_supported.eq(len(members)).all() or not rows.n_stable_hard.eq(stable).all():
            raise ValueError('GRAPH03_TOPOLOGY_DENOMINATOR_MISMATCH')
    return qc


def run_exploratory_auto(root, cfg=None, *, audit_only=False):
    """Publish current aggregates from the configured inputs without refitting Mapper."""
    root = Path(root).resolve()
    issues = []
    base = dict(mode=MODE, graph02_input=None, graph03_input=None, data_issues=issues,
                limitations=LIMITATIONS, **baseline_policy().metadata())
    try:
        data, paths = _source_settings(root)
        if cfg is not None:
            paths['step10'] = config_path(cfg['_path'], cfg['paths']['integrated'])
        defaults = graph_paths(root, data_root=data)
        graph02 = output_root(cfg, '02') if cfg else defaults['graph02']
        graph03 = output_root(cfg, '03') if cfg else defaults['graph03']
        target = graph03.parent / defaults['phasee'].name
        destinations = stage_destinations(target, include_figures=not audit_only)
        inputs = [*paths.values(), *stage_destinations(graph02).values(), *stage_destinations(graph03).values()]
        if any(destination.is_relative_to(path) or path.is_relative_to(destination)
               for destination in destinations.values() for path in inputs):
            raise ValueError('EXPLORATORY_OUTPUT_OVERLAPS_INPUT')
        prefix = '03f_' if audit_only else '03e_'
        completion = target / (prefix + 'summary.json')
        with ExitStack() as stack:
            for directory in dict.fromkeys((graph02, graph03)):
                if directory.is_dir():
                    stack.enter_context(output_lock(directory, shared=True))
            staged = stack.enter_context(stage_outputs(
                destinations, lambda _, name: name.startswith(prefix) or (not audit_only and name.startswith('03_graph_')),
                failure=('stage', prefix + 'failure.json'), completion=completion))
            staging = staged['stage']
            for artifact, directory, marker in (
                    ('graph02', graph02, '02_graph_failure.json'),
                    ('graph03', graph03, '03_graph_failure.json')):
                if (directory / marker).is_file():
                    issues.append(dict(status='PARTIAL', artifact=artifact,
                                       reason='LAST_ATTEMPT_FAILED_USING_LAST_COMPLETE_RESULT'))
            policy = _policy(root)
            validation = cfg['validation'] if cfg else yaml.safe_load((root / CONFIGS / VALIDATION).read_text())
            require_policy(validation)
            loaded = None
            frozen_before = frozen_after = {}
            for reader in (_load_graph02, _load_baseline_only):
                try:
                    loaded = reader(graph02)
                    issues.extend(loaded.pop('data_issues'))
                    break
                except READ_ERRORS:
                    continue
            if loaded is None:
                issues.append(dict(status='DATA_UNAVAILABLE', artifact='graph02_baseline',
                                   reason='NO_VALID_CURRENT_BASELINE'))
                if completion.exists() and json.loads(completion.read_text()).get('status') != 'INPUTS_UNUSABLE':
                    raise ValueError('EXPLORATORY_CURRENT_BASELINE_UNAVAILABLE')
                outputs, n = _empty_outputs(policy), None
            else:
                base['graph02_input'] = graph02.name
                frozen_before = frozen_fingerprints(loaded['release'], loaded['manifest'])
                base['graph02_release_sha256'] = frozen_before['release_manifest']
                baseline = loaded['frozen']['baseline_raw']
                features = loaded['frozen']['feature_sets']
                reps = loaded['manifest']['eligible_representation_ids']
                if reps:
                    try:
                        loaded['roles'] = feature_roles(features, baseline, validation, reps, loaded['registry'])
                    except ValueError:
                        issues.append(dict(status='DATA_UNAVAILABLE', artifact='graph02_feature_lineage',
                                           reason='CURRENT_DEFINING_FEATURE_LINEAGE_INVALID'))
                    try:
                        loaded['qc'] = _graph03_qc(graph03, loaded, loaded['roles'])
                        base['graph03_input'] = graph03.name
                    except READ_ERRORS as exc:
                        issues.append(dict(status='DATA_UNAVAILABLE', artifact='graph03_characterization',
                                           reason=MISMATCH if str(exc) == MISMATCH else
                                           'CURRENT_GRAPH03_MISSING_OR_INCONSISTENT'))
                integrated = loaded['manifest'].get('source_inputs', {}).get('integrated', {}).get('path')
                tables, records, source_issues = current_source_inputs(root, step10=integrated, paths=paths)
                issues.extend(source_issues)
                outputs = _analyze(tables, loaded, records, validation, policy, audit_only=audit_only)
                n = len(baseline)
                frozen_after = frozen_fingerprints(loaded['release'], loaded['manifest'])
                if frozen_before != frozen_after:
                    raise ValueError('EXPLORATORY_GRAPH02_CHANGED_DURING_ANALYSIS')
            if audit_only:
                schema = TABLE_COLUMNS['source_linkage_audit'].split()
                audit = pd.DataFrame(outputs['source_linkage_audit'], columns=schema)
                audit.to_csv(staging / '03f_source_linkage_audit.csv', index=False)
                result = dict(status='AUDITED' if n is not None else 'INPUTS_UNUSABLE',
                              output_dir=str(target), data_issues=issues, **baseline_policy().metadata(),
                              source_linkage_audit=str(target / '03f_source_linkage_audit.csv'))
                write_json(staging / '03f_summary.json', result)
            else:
                write_baseline_artifacts(staging, outputs, issues, frozen_before, frozen_after,
                                         roles=loaded['roles'] if loaded is not None else None)
                result = _write_run(staging, target, base, outputs, n, policy,
                                    staged.get('figures', staging), destinations.get('figures', target))
        return result
    except READ_ERRORS + (RuntimeError, yaml.YAMLError) as exc:
        token = str(exc)
        reason = token if token == MISMATCH or re.fullmatch(r'EXPLORATORY_[A-Z_]+', token) else 'EXPLORATORY_RUN_UNAVAILABLE'
        return dict(**base, status='EXPLORATORY_RUN_FAILED', output_dir=None, reason=reason)

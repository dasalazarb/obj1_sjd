"""Phase D: immutable evidence reconciliation and non-estimating preflight.

Integrity PASS is not clinical approval. Human decisions are content-pinned,
scoped documentary observations, never edits to the current analysis contract.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from .._pipeline import digest
from ..stage02.s02_contracts import safe_artifact
from .s03_construct_validation import _dependencies, _paths
from .s03_n3_candidate_audit import _proxy_alerts
from .s03_n3_phasec_adjudication import (
    CANDIDATES, DOCUMENTARY_FILES, LINEAGE, LAB_REFERENCE,
    load_phasec_inputs,
)
from .s03_n3_resolution_support import support_diagnostics
from .s03_release_reader import hard_members
from .s03_profiles import observed_values

VERSION = 'graph03_n3_phased_v1'
PREFIX = '03_graph_n3_phased_'
REPS = {'R0', 'R1', 'R2', 'R3', 'R4'}
REVIEW_KEYS = {'protocol', 'baseline_date', 'baseline_era', 'measurement',
              'candidate_independence', 'pop_semantics', 'topology_sensitivity', 'statistical_design'}
SOURCE_SCOPED = {'measurement', 'candidate_independence', 'statistical_design'}
POLICY = dict(require_human_signoff=True, **{key: False for key in (
    'allow_auto_approval', 'allow_source_recode', 'allow_discovery_refit',
    'allow_model_estimation', 'allow_endpoint_policy_changes', 'allow_public_export',
    'allow_implicit_latest_release', 'allow_covariate_drop', 'allow_constant_imputation')})
ALIASES = ['demo__protocol', 'protocol', 'ids__protocol', 'ids__protocol_number',
           'parent_protocol', 'source_protocol', 'spine__source_protocol',
           'pop__protocol', 'pro__parent_protocol']
CODE_FILES = sorted(set(DOCUMENTARY_FILES + [
    'src/block_A/01_pop_distribution.py', 'src/studies/graph/stage03/s03_release_reader.py',
    'src/studies/graph/stage02/s02_contracts.py',
    'src/studies/graph/stage03/s03_n3_resolution.py',
    'src/studies/graph/stage03/s03_n3_resolution_support.py',
    'src/studies/graph/03d_graph_n3_resolution_preflight.py']))
PHASEC_TABLES = ['candidate_adjudication', 'source_lineage', 'measurement_review',
                'model_readiness', 'human_review_queue', 'covariate_audit',
                'baseline_era_audit', 'pop_vocabulary_audit']


def _sha(value, size=64):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{%d}' % size, value) is not None


def validate_config(config):
    if not isinstance(config, dict) or config.get('schema_version') != VERSION:
        raise ValueError('UNSUPPORTED_PHASED_SCHEMA')
    if config.get('privacy_class') != 'INTERNAL_RESTRICTED' or type(config.get('minimum_disclosure_cell')) is not int or config['minimum_disclosure_cell'] < 5:
        raise ValueError('PHASED_PRIVACY_CONTRACT_REQUIRED')
    policy = config.get('policy', {})
    if any(policy.get(key) is not value for key, value in POLICY.items()):
        raise ValueError('UNSAFE_PHASED_POLICY')
    if policy.get('code_drift') not in {'reject', 'report_and_block'}:
        raise ValueError('EXPLICIT_CODE_DRIFT_POLICY_REQUIRED')
    pins = config.get('input_contract', {})
    for key in ['graph03_run_id', 'graph02_run_id', 'phasec_run_id']:
        if not isinstance(pins.get(key), str) or not re.fullmatch('[A-Za-z0-9_]+', pins[key]):
            raise ValueError('MISSING_OR_INVALID_RUN_PIN')
    for key in ['graph02_release_sha256', 'graph03_summary_sha256',
                'phasec_qc_sha256', 'phasec_inventory_sha256']:
        if not _sha(pins.get(key)):
            raise ValueError('MISSING_OR_INVALID_INDEPENDENT_HASH_PIN_' + key)
    for key in ['graph03_implementation_commit_sha', 'phasec_implementation_commit_sha']:
        if not _sha(pins.get(key), 40):
            raise ValueError('MISSING_OR_INVALID_IMPLEMENTATION_PIN')
    if pins.get('n3_schema_version') != 'graph03_n3_audit_v1':
        raise ValueError('UNSUPPORTED_PHASE_A_SCHEMA')
    reviews = config.get('proposed_reviews', {})
    if set(reviews) != REVIEW_KEYS:
        raise ValueError('EXPLICIT_SCOPED_REVIEW_SECTIONS_REQUIRED')
    for section, records in reviews.items():
        if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
            raise ValueError('INVALID_REVIEW_RECORDS')
        keys = []
        for record in records:
            rep, source = record.get('representation_id'), record.get('source_column', '')
            if rep not in REPS or section in SOURCE_SCOPED and source not in CANDIDATES:
                raise ValueError('INVALID_REVIEW_SCOPE')
            if record.get('status', 'PENDING_REVIEW') not in {'PENDING', 'PENDING_REVIEW', 'DOCUMENTED', 'REJECTED'}:
                raise ValueError('PHASED_CANNOT_GRANT_APPROVAL')
            keys.append((rep, source if section in SOURCE_SCOPED else ''))
        if len(keys) != len(set(keys)):
            raise ValueError('DUPLICATE_PHASED_REVIEW')


def _snapshot(roots):
    """Protect every frozen file, including uninventoried files; reject symlinks."""
    snapshot = {}
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                raise ValueError('SYMLINK_IN_FROZEN_INPUT')
            if path.is_file():
                snapshot[path] = digest(path)
    return snapshot


def _check_overlap(output, roots):
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError('PHASED_OUTPUT_OVERLAPS_FROZEN_INPUTS')
    if output.exists():
        raise FileExistsError('PHASED_OUTPUT_MUST_BE_NEW')


def _json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('INVALID_DOCUMENT')
    return value


def load_inputs(graph03, release, phasec, config):
    """Validate three frozen releases, including a pinned Phase C hash inventory."""
    validate_config(config)
    pins = config['input_contract']
    summary_path = safe_artifact(graph03, '03_graph_results_summary.json')
    if digest(summary_path) != pins['graph03_summary_sha256']:
        raise ValueError('GRAPH03_SUMMARY_SHA_MISMATCH')
    # Reuse the verified Phase A reader, without reading/mutating Phase C config.
    phasec_contract = dict(schema_version='graph03_n3_phasec_v1', auto_approve_independence=False,
        privacy_class='INTERNAL_RESTRICTED', minimum_disclosure_cell=config['minimum_disclosure_cell'],
        input_contract={key: pins[key] for key in ['graph03_run_id', 'graph02_run_id',
                        'graph02_release_sha256', 'n3_schema_version']},
        families={family: dict(sources=[s for s, f in CANDIDATES.items() if f == family])
                  for family in set(CANDIDATES.values())})
    phasec_contract['input_contract']['implementation_commit_sha'] = pins['graph03_implementation_commit_sha']
    inputs = load_phasec_inputs(graph03, release, phasec_contract)
    summary = _json(summary_path)
    for name, sha in summary['sha256_by_artifact'].items():
        if digest(safe_artifact(graph03, name)) != sha:
            raise ValueError('GRAPH03_ARTIFACT_SHA_MISMATCH')
    qc_path = safe_artifact(phasec, '03_graph_n3_phasec_qc.json')
    inventory_path = safe_artifact(phasec, '03_graph_n3_phasec_artifact_inventory.csv')
    if digest(qc_path) != pins['phasec_qc_sha256'] or digest(inventory_path) != pins['phasec_inventory_sha256']:
        raise ValueError('PHASEC_TRUST_ANCHOR_SHA_MISMATCH')
    qc = _json(qc_path)
    if qc.get('status') != 'PASS' or qc.get('phasec_status') != 'COMPLETE_WITH_DOCUMENTARY_BLOCKERS':
        raise ValueError('PHASEC_QC_NOT_PASS_WITH_BLOCKERS')
    required_true = ['no_inference', 'no_refit', 'graph02_unchanged', 'graph03_unchanged',
                     'no_source_recoding', 'longitudinal_policies_unchanged']
    if any(qc.get(k) is not True for k in required_true) or any(qc.get(k) is not False for k in ['clinical_approval_granted', 'shareable_export_allowed']):
        raise ValueError('UNSAFE_PHASEC_EVIDENCE')
    if qc.get('n_automatic_approvals') != 0 or qc.get('n_models_complete') != 0 or qc.get('privacy_class') != 'INTERNAL_RESTRICTED':
        raise ValueError('PHASEC_APPROVAL_OR_ESTIMATION_NOT_ALLOWED')
    expected = dict(run_id=pins['phasec_run_id'], input_graph03_run_id=pins['graph03_run_id'],
        graph02_run_id=pins['graph02_run_id'], graph02_release_sha256=pins['graph02_release_sha256'],
        implementation_commit_sha=pins['phasec_implementation_commit_sha'],
        input_implementation_commit_sha=pins['graph03_implementation_commit_sha'], schema_version='graph03_n3_phasec_v1')
    if any(qc.get(k) != v for k, v in expected.items()):
        raise ValueError('PHASEC_INPUT_OR_IMPLEMENTATION_PROVENANCE_MISMATCH')
    if qc.get('input_sha256_by_artifact') != inputs['input_hashes'] or qc.get('frozen_sha256_by_artifact') != inputs['manifest']['sha256_by_artifact']:
        raise ValueError('PHASEC_FROZEN_INPUT_HASHES_NOT_RECONCILED')
    artifacts = pd.read_csv(inventory_path, keep_default_na=False)
    required_files = {'03_graph_n3_phasec_' + n + '.csv' for n in PHASEC_TABLES} | {qc_path.name, '03_graph_n3_phasec_summary.md'}
    if 'relative_path' not in artifacts or artifacts.relative_path.duplicated().any() or set(artifacts.relative_path) != required_files:
        raise ValueError('INCOMPLETE_OR_DUPLICATE_PHASEC_INVENTORY')
    for row in artifacts.to_dict('records'):
        path = safe_artifact(phasec, row['relative_path'])
        if digest(path) != row.get('sha256'):
            raise ValueError('PHASEC_ARTIFACT_SHA_MISMATCH')
        if any(row.get(k) != v for k, v in expected.items() if k in {'run_id', 'input_graph03_run_id', 'graph02_run_id', 'graph02_release_sha256', 'implementation_commit_sha', 'schema_version'}):
            raise ValueError('PHASEC_INVENTORY_PROVENANCE_MISMATCH')
    tables = {name: pd.read_csv(safe_artifact(phasec, '03_graph_n3_phasec_' + name + '.csv')) for name in PHASEC_TABLES}
    for table in tables.values():
        for key in ['run_id', 'input_graph03_run_id', 'graph02_run_id',
                    'graph02_release_sha256', 'implementation_commit_sha', 'schema_version']:
            if key not in table or not table[key].eq(expected[key]).all():
                raise ValueError('PHASEC_TABLE_PROVENANCE_MISMATCH')
    candidates = tables['candidate_adjudication']
    models = tables['model_readiness']
    reps = inputs['manifest']['eligible_representation_ids']
    keys = {(rep, source) for rep in reps for source in CANDIDATES}
    for name in ['candidate_adjudication', 'source_lineage', 'measurement_review', 'human_review_queue']:
        table = tables[name]
        if table.duplicated(['representation_id', 'source_column']).any() or set(map(tuple, table[['representation_id', 'source_column']].to_numpy())) != keys:
            raise ValueError('PHASEC_CANDIDATE_KEYS_NOT_RECONCILED')
    if (models.duplicated(['representation_id', 'source_column', 'model']).any() or
            set(map(tuple, models[['representation_id', 'source_column', 'model']].to_numpy())) != {(r, s, m) for r, s in keys for m in 'ABCD'} or
            not models.decision_status.eq('NOT_ESTIMABLE').all()):
        raise ValueError('PHASEC_PREPARATION_ROWS_NOT_RECONCILED')
    if candidates.decision_status.isin(['APPROVED', 'N3_REVIEWED']).any() or qc.get('n_candidate_rows') != len(keys) or qc.get('n_model_readiness_rows') != len(keys)*4:
        raise ValueError('PHASEC_DECISION_OR_ROW_COUNTS_NOT_RECONCILED')
    inputs.update(phasec_qc=qc, phasec_tables=tables, phasec_inventory=artifacts)
    # The frozen clinical registry carries actual lineage; never replace it with
    # guesses based only on source names.
    inputs['registry'] = pd.read_csv(safe_artifact(inputs['release'].parent,
                                    inputs['manifest']['relative_paths']['clinical_registry']))
    inputs['missing_candidate_keys'] = sorted(keys - set(map(tuple, inputs['tables']['source_inventory'][['representation_id', 'source_column']].to_numpy())))
    return inputs


def _reference(root, reference):
    if not isinstance(reference, str) or not re.fullmatch(r'(?:src|docs|tests)/[A-Za-z0-9_./-]+(?::[A-Za-z_][A-Za-z0-9_]*)?', reference):
        return None
    name, _, symbol = reference.partition(':')
    try:
        path = safe_artifact(root, name)
        relative = Path(name)
        if any((Path(root) / Path(*relative.parts[:i])).is_symlink()
               for i in range(1, len(relative.parts) + 1)):
            return None
    except (OSError, ValueError):
        return None
    if path.stat().st_size > 2_000_000 or symbol and symbol not in path.read_text():
        return None
    return path


def review_record(config, section, rep, source, root):
    """Verify an explicitly supplied signoff against its immutable decision file.

    This checks documentary consistency, not the identity/authenticity of a
    medical signature. That remains an institutional responsibility.
    """
    matches = [r for r in config['proposed_reviews'][section]
               if r['representation_id'] == rep and (section not in SOURCE_SCOPED or r['source_column'] == source)]
    record = dict(matches[0]) if matches else {}
    signed, document_sha, state = False, '', 'PENDING_REVIEW'
    try:
        path = _reference(root, record.get('source_reference', ''))
        if path and _sha(record.get('source_sha256')) and digest(path) == record['source_sha256']:
            document_sha = digest(path)
            document = yaml.safe_load(path.read_text())
            payload = {k: v for k, v in record.items() if k not in {'source_reference', 'source_sha256'}}
            consistent = isinstance(document, dict) and document == dict(review_section=section, **payload)
            signed = bool(consistent and record.get('human_signoff') is True and
                isinstance(record.get('reviewer'), str) and record['reviewer'].strip() and
                record.get('review_id') and record.get('status') in {'DOCUMENTED', 'REJECTED'})
            state = record.get('status') if signed else 'INVALID_SIGNOFF'
        elif record:
            state = 'INVALID_OR_MODIFIED_DOCUMENT'
    except (ValueError, OSError, yaml.YAMLError):
        state = 'INVALID_OR_MODIFIED_DOCUMENT'
    return dict(record=record, verified=signed and record.get('status') == 'DOCUMENTED',
                rejected=signed and record.get('status') == 'REJECTED',
                document_sha256=document_sha, human_review_status=state)


def protocol_inventory(baseline, review, rep):
    record, verified = review['record'], review['verified']
    canonical = 'demo__protocol'
    canonical_state = 'ABSENT' if canonical not in baseline else 'ALL_NULL' if baseline[canonical].isna().all() else 'PRESENT_UNVERIFIED'
    present_aliases = [a for a in ALIASES[1:] if a in baseline and baseline[a].notna().any()]
    mapping = (verified and record.get('equivalent_to_prespecified_confounder') is True and
        record.get('canonical_target') == canonical and record.get('original_source') in ALIASES and
        record.get('original_source') in baseline and baseline[record['original_source']].notna().any() and
        record.get('cohort_scope') == rep and record.get('code_dictionary_review') == 'DOCUMENTED' and
        record.get('multiple_protocol_rule_review') == 'DOCUMENTED' and
        record.get('meaning') == 'research_protocol')
    state = 'MAPPING_HUMAN_VERIFIED' if mapping else 'ALIAS_PRESENT_UNVERIFIED' if present_aliases else canonical_state
    rows = []
    for source in ALIASES:
        rows.append(dict(representation_id=rep, source_column=source, canonical_target=canonical,
            canonical_field_status=canonical_state, decision_status=state,
            field_status='ABSENT' if source not in baseline else 'ALL_NULL' if baseline[source].isna().all() else 'PRESENT',
            n_baseline=len(baseline), n_observed=int(baseline[source].notna().sum()) if source in baseline else 0,
            original_source=record.get('original_source') if mapping else '', human_signoff=bool(mapping),
            document_sha256=review['document_sha256'], cohort_scope=rep,
            grain='patient_x_official_baseline;upstream_patient_x_episode',
            original_table='Step10_context_candidate;Step11_analytic_availability_only',
            code_reference='src/block_A/10_build_integrated_longitudinal_dataset.py:build_curated',
            code_dictionary_status='DOCUMENTED' if mapping else 'PENDING_REVIEW',
            multiple_protocol_adjudication='DOCUMENTED' if mapping else 'PENDING_REVIEW',
            proposed_derived_dataset=bool(mapping), mapping_applied=False,
            blocking_reasons='NEW_CONTRACT_AND_RELEASE_REQUIRED' if mapping else 'PROTOCOL_MAPPING_REQUIRES_REVIEW'))
    for source in ['demo__age_at_baseline', 'demo__sex', 'baseline_era']:
        rows.append(dict(representation_id=rep, source_column=source, canonical_target=source,
            decision_status='PENDING_REVIEW', canonical_field_status='AVAILABLE' if source in baseline else 'ABSENT',
            n_baseline=len(baseline), n_observed=int(baseline[source].notna().sum()) if source in baseline else 0,
            mapping_applied=False, blocking_reasons='PRESPECIFIED_COVARIATE_SEMANTICS_REQUIRED'))
    return rows


def date_readiness(review):
    record = review['record']
    future = record.get('derived_from_future')
    fields = ['baseline_definition', 'lookback_rule', 'measurement_source_reference', 'repeated_measurement_rule']
    complete = (review['verified'] and record.get('date_semantics') == 'instrument_measurement' and
        record.get('instrument_measurement_date_verified') is True and future is False and
        record.get('timezone_review') == 'DOCUMENTED' and
        record.get('baseline_temporal_order_review') == 'DOCUMENTED' and
        all(isinstance(record.get(k), str) and bool(record[k].strip()) for k in fields))
    return dict(instrument_measurement_date_verified=bool(complete),
        derived_from_future='true' if future is True else 'false' if future is False else 'unknown',
        baseline_definition='HUMAN_DOCUMENTED' if complete else 'PENDING_REVIEW',
        lookback_rule='HUMAN_DOCUMENTED_SOURCE_SPECIFIC' if complete else 'PENDING_REVIEW',
        timezone_review='DOCUMENTED' if complete else 'PENDING_REVIEW',
        measurement_source_reference='CONTENT_PINNED_HUMAN_DOCUMENT' if complete else 'PENDING_REVIEW',
        repeated_measurement_rule='HUMAN_DOCUMENTED' if complete else 'PENDING_REVIEW',
        human_review_status=review['human_review_status'], document_sha256=review['document_sha256'],
        temporal_eligibility='BLOCKED_FUTURE_DERIVED' if future is True else 'READY_FOR_HUMAN_REVIEW' if complete else 'PENDING_REVIEW',
        longitudinal_status='BLOCKED_SEPARATE_EPISODE_AUDIT_REQUIRED')


def dependency_evidence(source, defining, features, registry, validation, root):
    """Traverse typed, code-referenced dependencies using the existing graph rules."""
    clean = json.loads(json.dumps(validation))
    clean['characterization']['relationships'] = [r for r in clean['characterization'].get('relationships', []) if _reference(root, r.get('source_reference', ''))]
    clean['characterization']['instruments'] = [r for r in clean['characterization'].get('instruments', []) if _reference(root, r.get('source_reference', ''))]
    paths = _paths(_dependencies(features, registry, clean), source)
    related = sorted(defining & set(paths))
    kinds = {edge['relationship_type'] for name in related for edge in paths[name]}
    if source in defining:
        status = 'DIRECT_DEPENDENCE'
    elif related:
        status = ('SAME_RAW_ASSAY' if kinds & {'same_raw_source', 'encoded_variant'} else
                  'COMPOSITE_COMPONENT' if kinds & {'derived_from', 'component_of'} else 'DIRECT_DEPENDENCE')
    elif _proxy_alerts(source, defining, validation):
        status = 'CONCEPTUAL_PROXY_REVIEW'
    else:
        status = 'NO_DOCUMENTED_DEPENDENCE'
    return status, related, {name: paths[name] for name in related}


def pop_dependency(source, validation):
    """Keep the existing _pop_status defense without importing its fit module."""
    related = {'essdai__total', 'esspri__total'}
    relations = validation['characterization'].get('relationships', [])
    while True:
        before = set(related)
        for r in relations:
            a, b = r.get('source_column'), r.get('target_source')
            if validation['pop_column'] in {a, b}:
                continue
            if r.get('relationship_type') in {'derived_from', 'composite_component', 'same_instrument', 'same_raw_source', 'encoded_variant'} and {a, b} & related:
                related.update([a, b])
        if related == before:
            break
    return source in related or source.startswith(('essdai__', 'esspri__', 'eg_')) or source == 'ovl__status'


def _clean_values(series, numeric=False):
    series = observed_values(series)
    if numeric or pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    # Availability is not a category convention or clinical recoding.
    return series.where(series.notna() & series.astype('string').str.strip().ne(''))


def _approved_date(review, baseline):
    record = review['record']
    source = record.get('date_column')
    # Neither an existing date field nor a supplied era approval proves t0.
    return bool(review['verified'] and source in {'clinical_baseline_date', 'clinical_anchor_date'} and
        source in baseline and pd.to_datetime(baseline[source], errors='coerce', utc=True).notna().all() and
        record.get('official_baseline_definition_review') == 'DOCUMENTED' and
        record.get('timezone_review') == 'DOCUMENTED' and record.get('lookahead_review') == 'DOCUMENTED')


def build_tables(inputs, config, root, validation):
    baseline, membership, features = (inputs['frozen'][k] for k in ['baseline_raw', 'membership', 'feature_sets'])
    inv = inputs['tables']['source_inventory'].set_index(['representation_id', 'source_column'])
    phasec = inputs['phasec_tables']['candidate_adjudication'].set_index(['representation_id', 'source_column'])
    historical_models = inputs['phasec_tables']['model_readiness'].set_index(['representation_id', 'source_column', 'model'])
    tables = {k: [] for k in ['protocol_source_inventory', 'baseline_date_lineage', 'endpoint_date_readiness',
        'candidate_evidence_matrix', 'circularity_dependency_report', 'pop_crosswalk_proposal',
        'topology_stability_audit', 'model_preflight', 'human_review_queue']}
    pop_reconciled = True
    for rep in inputs['manifest']['eligible_representation_ids']:
        get_review = lambda section, source='': review_record(config, section, rep, source, root)
        protocol = get_review('protocol')
        tables['protocol_source_inventory'].extend(protocol_inventory(baseline, protocol, rep))
        official_review, era_review = get_review('baseline_date'), get_review('baseline_era')
        official = _approved_date(official_review, baseline)
        era = official and _approved_date(era_review, baseline) and era_review['record'].get('era_rule') == 'fixed_five_year_strata'
        for source in ['clinical_baseline_date', 'clinical_anchor_date', 't0']:
            dates = pd.to_datetime(baseline[source], errors='coerce', utc=True) if source in baseline else pd.Series(pd.NaT, index=baseline.index)
            tables['baseline_date_lineage'].append(dict(representation_id=rep, source_column=source,
                n_baseline=len(baseline), n_valid=int(dates.notna().sum()), n_missing=int(dates.isna().sum()),
                official_baseline_status='HUMAN_DOCUMENTED' if official else 'PENDING_REVIEW',
                administrative_era_status='HUMAN_DOCUMENTED' if era else 'PENDING_REVIEW',
                instrument_measurement_status='PENDING_REVIEW', timezone_review='DOCUMENTED' if official else 'PENDING_REVIEW',
                code_reference='src/block_A/11_integrated_baseline_characterization.py:validate_integrated',
                upstream_reference='src/00_build_visit_spine.py:build_episode_spines',
                date_semantics='episode_anchor_does_not_verify_instrument_date',
                human_review_status=official_review['human_review_status'],
                document_sha256=official_review['document_sha256'], era_document_sha256=era_review['document_sha256'],
                longitudinal_status='BLOCKED_SEPARATE_ENDPOINT_REVIEW'))
        part = membership.loc[membership.representation_id.eq(rep)]
        hard = hard_members(membership, rep)
        frame = hard.merge(baseline, on='patient_id', validate='one_to_one')
        topology = bool(len(hard) and hard.structural_stability_status.eq('stable').all() and rep != 'R2' and
                        inputs['manifest']['release_status'] != 'NO_STABLE_STRUCTURE')
        tables['topology_stability_audit'].append(dict(representation_id=rep,
            n_baseline=len(baseline), n_mapper_covered=part.loc[part.mapper_covered, 'patient_id'].nunique(),
            n_hard_assignment=part.loc[part.hard_assignment, 'patient_id'].nunique(),
            n_membership_tie=part.loc[part.membership_tie, 'patient_id'].nunique(),
            n_supported=part.loc[part.supported, 'patient_id'].nunique(), n_hard_supported=len(hard),
            n_stable_hard=int(hard.structural_stability_status.eq('stable').sum()),
            n_provisional_hard=int(hard.structural_stability_status.ne('stable').sum()),
            decision_status='not_eligible' if not len(hard) else 'descriptive_only' if rep == 'R2' else 'stable_support' if topology else 'provisional',
            topology_gate='READY_FOR_HUMAN_REVIEW' if topology else 'BLOCKED_PROVISIONAL_TOPOLOGY',
            r2_exploratory=rep == 'R2', r0_stable_only_executed=False,
            sensitivity_review_status=get_review('topology_sensitivity')['human_review_status'],
            membership_changed=False, denominator_reconciled=True))
        pop_review = get_review('pop_semantics')
        proposed_crosswalk = pop_review['record'].get('crosswalk', {})
        crosswalk_valid = bool(pop_review['verified'] and
            pop_review['record'].get('equivalence_review') == 'DOCUMENTED' and
            isinstance(proposed_crosswalk, dict) and proposed_crosswalk and
            set(proposed_crosswalk) <= {'Unclassifiable', 'Unclassified'} and
            set(proposed_crosswalk.values()) <= {'Unclassifiable', 'Unclassified'})
        universes = {'baseline': baseline, 'mapper_covered': baseline.loc[baseline.patient_id.isin(part.loc[part.mapper_covered, 'patient_id'])], 'hard_supported': frame}
        for universe, subset in universes.items():
            pop = subset.get('pop__status', pd.Series(pd.NA, index=subset.index))
            labels = ['Pop1', 'Pop2', 'Pop3', 'Unclassifiable', 'Unclassified', 'MISSING', 'INVALID']
            counts = [int(pop.eq(label).fillna(False).sum()) for label in labels[:5]]
            counts += [int(pop.isna().sum()), len(pop)-sum(counts)-int(pop.isna().sum())]
            reconciled = sum(counts) == len(subset)
            pop_reconciled &= reconciled
            for label, count in zip(labels, counts):
                tables['pop_crosswalk_proposal'].append(dict(representation_id=rep, universe=universe,
                    source_column='pop__status', original_category=label,
                    proposed_category=proposed_crosswalk.get(label, label) if crosswalk_valid else label,
                    n_category=count, n_universe=len(subset), denominator_reconciled=reconciled,
                    n_proposed_category=sum(c for l, c in zip(labels, counts)
                        if (proposed_crosswalk.get(l, l) if crosswalk_valid else l) ==
                           (proposed_crosswalk.get(label, label) if crosswalk_valid else label)),
                    comparison_eligible=label in {'Pop1', 'Pop2', 'Pop3'}, descriptive_universe_retained=True,
                    semantic_review_status=pop_review['human_review_status'],
                    document_sha256=pop_review['document_sha256'], mapping_applied=False,
                    decision_status='SIGNED_PROPOSAL_NEW_RELEASE_REQUIRED' if crosswalk_valid else 'PENDING_REVIEW',
                    code_reference='src/block_A/01_pop_distribution.py:DISPLAY',
                    verification_reference='src/block_A/observed_patterns.py:verified_baseline_pop'))
        defining = set(features.loc[features.representation_id.eq(rep) & features.included, 'source_column'])
        rep_features = features.loc[features.representation_id.eq(rep)]
        adjustments = pd.DataFrame(index=frame.index)
        for source in validation['confounders']:
            adjustments[source] = _clean_values(frame[source], numeric=source == 'demo__age_at_baseline') if source in frame else np.nan
        if era:
            dates = pd.to_datetime(frame[era_review['record']['date_column']], errors='coerce', utc=True)
            adjustments['baseline_era'] = (dates.dt.year // 5).astype('Int64').astype('string').where(dates.notna())
        else:
            adjustments['baseline_era'] = np.nan
        for source, family in CANDIDATES.items():
            measurement, independence, stats = (get_review(s, source) for s in ['measurement', 'candidate_independence', 'statistical_design'])
            date = date_readiness(measurement)
            record = measurement['record']
            measure_ready = (measurement['verified'] and date['instrument_measurement_date_verified'] and
                record.get('unit_review') == 'DOCUMENTED' and record.get('category_review') == 'DOCUMENTED')
            independent = independence['verified'] and all(independence['record'].get(k) == 'DOCUMENTED' for k in
                ['clinical_review', 'origin_review', 'proxy_review', 'discovery_exclusion_review'])
            dep, related, paths = dependency_evidence(source, defining, rep_features, inputs['registry'], validation, root)
            role = inputs['roles'].loc[inputs['roles'].representation_id.eq(rep) & inputs['roles'].source_column.eq(source)]
            dependent = dep in {'DIRECT_DEPENDENCE', 'SAME_RAW_ASSAY', 'COMPOSITE_COMPONENT'} or (len(role) == 1 and role.iloc[0].tier in {'N1_DEFINING', 'N2_RELATED'})
            historical = phasec.loc[(rep, source)]
            raw = inv.loc[(rep, source)].to_dict() if (rep, source) in inv.index else {}
            blocked = ['NEW_APPROVED_CONTRACT_AND_RELEASE_REQUIRED', 'HUMAN_SIGNOFF_REQUIRED']
            if not raw:
                blocked.append('SOURCE_ABSENT_FROM_PHASE_A_INVENTORY')
            if dependent:
                blocked.append('DOCUMENTED_DISCOVERY_DEPENDENCE')
            if not independent:
                blocked.append('CONCEPTUAL_INDEPENDENCE_REVIEW_REQUIRED')
            if not measure_ready:
                blocked.append('INSTRUMENT_MEASUREMENT_REVIEW_REQUIRED')
            if date['derived_from_future'] == 'true':
                blocked.append('FUTURE_DERIVED_ENDPOINT')
            if measurement['rejected'] or independence['rejected'] or stats['rejected']:
                blocked.append('HUMAN_REJECTED')
            if str(historical.decision_status).startswith('BLOCKED'):
                blocked.append('HISTORICAL_PHASEC_BLOCK_REQUIRES_NEW_RELEASE')
            lineage_ref, original, transformations = LINEAGE.get(family, (LAB_REFERENCE, 'ORIGINAL_ASSAY_FIELD_PENDING_REVIEW', 'closest_absolute_anchor_distance;pre_anchor_tie;conflict_masking;possible_post_index'))
            row = dict(representation_id=rep, source_column=source, candidate_family=family,
                original_phasec_decision=historical.decision_status, original_phasec_evidence=historical.evidence_status,
                dependency_status=dep, N1_defining='|'.join(sorted(defining)), related_N1_columns='|'.join(related),
                origin=original, transformations=transformations, code_reference=lineage_ref,
                measurement_evidence='HUMAN_DOCUMENTED' if measure_ready else 'PENDING_REVIEW',
                clinical_independence_evidence='HUMAN_DOCUMENTED' if independent else 'PENDING_REVIEW',
                independence_document_sha256=independence['document_sha256'],
                n_Y_observed=int(frame[source].notna().sum()) if source in frame else 0,
                n_hard_supported=len(hard), unit_category_status='DOCUMENTED' if measure_ready else 'PENDING_REVIEW',
                pop_comparator_status='POP_BASELINE_COMPARATOR_INELIGIBLE' if pop_dependency(source, validation) else 'POP_COMPARATOR_REQUIRES_SEPARATE_REVIEW',
                decision_status='PENDING_REVIEW', blocking_reasons='|'.join(sorted(blocked)), **date)
            tables['candidate_evidence_matrix'].append(row)
            tables['endpoint_date_readiness'].append(dict(representation_id=rep, source_column=source, **date,
                timeline='serology' if source.startswith('lab__') else 'questionnaire' if family == 'symptoms' else 'source_specific_clinical',
                t0_status='HUMAN_DOCUMENTED' if official else 'PENDING_REVIEW',
                treatment_timeline_status='SEPARATE_AUDIT_REQUIRED', code_reference=lineage_ref))
            tables['circularity_dependency_report'].append(dict(representation_id=rep, source_column=source,
                dependency_status=dep, related_N1_columns='|'.join(related),
                dependency_paths_json=json.dumps(paths, sort_keys=True),
                proxy_review_status='HUMAN_DOCUMENTED' if independent else 'PENDING_REVIEW',
                independence_approved=False, pop_comparator_status=row['pop_comparator_status'],
                assay_equivalence_status='PENDING_REVIEW', alternative_assay_merge=False))
            kind = record.get('outcome_type', raw.get('type', 'unknown'))
            series = frame.get(source, pd.Series(np.nan, index=frame.index))
            y = _clean_values(series, numeric=kind in {'numeric', 'binary'})
            if kind == 'binary':
                # Only an explicit reviewed numeric encoding is handled here;
                # text assays require a future versioned measurement contract.
                y = y.where(y.isin([0., 1.])) if measure_ready and record.get('binary_encoding') == 'numeric_0_1' else pd.Series(np.nan, index=frame.index)
            design_record = stats['record']
            basis_columns = design_record.get('n1_source_columns', [])
            basis_ready = bool(stats['verified'] and isinstance(basis_columns, list) and basis_columns and
                len(set(basis_columns)) == len(basis_columns) and set(basis_columns) <= defining and
                set(basis_columns) <= set(frame) and design_record.get('n1_basis_review') == 'DOCUMENTED')
            basis = frame[basis_columns].apply(_clean_values) if basis_ready else pd.DataFrame(index=frame.index)
            adjusted = adjustments.copy()
            pop_requested = design_record.get('include_pop') is True
            pop_ready = bool(pop_requested and stats['verified'] and not pop_dependency(source, validation) and
                pop_review['verified'] and design_record.get('pop_dependency_review') == 'DOCUMENTED')
            if pop_requested:
                adjusted['pop_comparator'] = frame['pop__status'].where(frame['pop__status'].isin(['Pop1', 'Pop2', 'Pop3'])) if pop_ready and 'pop__status' in frame else np.nan
            stat_ready = stats['verified'] and all(design_record.get(k) == 'DOCUMENTED' for k in
                ['covariate_category_review', 'hypothesis_family_review', 'multiplicity_review', 'uncertainty_review'])
            stat_ready = bool(stat_ready and design_record.get('hypothesis_family') and
                design_record.get('multiplicity_method') == 'fdr_bh' and
                design_record.get('uncertainty_plan') and
                design_record.get('missingness_policy') == 'common_complete_case')
            diag = support_diagnostics(y, adjusted, basis, frame.community_id, kind, validation,
                categories_verified=measure_ready and stat_ready, basis_requested=basis_ready,
                include_d=stats['verified'] and design_record.get('include_model_d') is True,
                minimum_events_per_parameter=design_record.get('minimum_events_per_parameter'))
            for model, result in diag.items():
                reasons = list(blocked)
                if not topology:
                    reasons.append('BLOCKED_PROVISIONAL_TOPOLOGY')
                if not era or 'demo__protocol' not in frame or frame.get('demo__protocol', pd.Series(dtype=float)).isna().all():
                    reasons.append('PRESPECIFIED_COVARIATE_GATE_BLOCKED')
                if not stat_ready:
                    reasons.append('STATISTICAL_DESIGN_REVIEW_REQUIRED')
                if pop_requested and not pop_ready:
                    reasons.append('POP_BASELINE_COMPARATOR_INELIGIBLE' if pop_dependency(source, validation) else 'POP_COMPARATOR_REVIEW_PENDING')
                result.update(representation_id=rep, source_column=source,
                    historical_phasec_status=historical_models.loc[(rep, source, model), 'decision_status'],
                    decision_status='ESTIMATION_BLOCKED', status='NOT_ESTIMABLE',
                    documentary_gate='READY_FOR_HUMAN_REVIEW' if independent and measure_ready and not dependent else 'BLOCKED',
                    topology_gate='READY_FOR_HUMAN_REVIEW' if topology else 'BLOCKED_PROVISIONAL_TOPOLOGY',
                    covariate_gate='READY_FOR_HUMAN_REVIEW' if era and protocol_inventory(baseline, protocol, rep)[0]['decision_status'] == 'MAPPING_HUMAN_VERIFIED' and adjusted.notna().all(axis=1).any() else 'BLOCKED',
                    minimum_group_patients=validation['minimum_group_patients'], minimum_model_patients=validation['minimum_model_patients'],
                    minimum_patients_per_community=validation['minimum_patients_per_community'], minimum_residual_df=10,
                    fdr_alpha=validation['fdr_alpha'], proposed_contrasts='B-A|C-A|D-C',
                    model_contract={'A': 'covariates_plus_eligible_reviewed_Pop', 'B': 'A_plus_frozen_community',
                        'C': 'A_plus_prespecified_reviewed_N1', 'D': 'C_plus_community_only_if_requested'}[model],
                    blocking_reasons='|'.join(sorted(set(reasons + result['support_blocking_reasons'].split('|')) - {''})),
                    inference_scope='development_cohort_conditional_on_frozen_membership_not_external_validation')
                tables['model_preflight'].append(result)
            tables['human_review_queue'].append(dict(representation_id=rep, source_column=source,
                candidate_family=family, decision_status='PENDING_REVIEW', blocking_reasons=row['blocking_reasons'],
                required_evidence='signed_original_form_assay_date_units_categories_conceptual_independence_and_design',
                protocol_action='RESEARCH_PROTOCOL_DICTIONARY_MULTIPLE_PROTOCOL_RULE_AND_COHORT_SCOPED_SIGNOFF',
                baseline_action='OFFICIAL_BASELINE_AND_ERA_SEPARATE_SIGNOFFS_WITHOUT_ENDPOINT_APPROVAL',
                topology_action='PRESPECIFIED_STABLE_SUBSET_DESIGN_OR_NEW_APPROVED_RELEASE',
                design_action='COMMON_CASE_NESTING_RANK_EVENTS_COMMUNITIES_MULTIPLICITY_AND_UNCERTAINTY',
                reviewer_signoff_recorded=bool(independent and measure_ready),
                document_sha256=measurement['document_sha256']))
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}, pop_reconciled


def _disclosure_boundary(tables):
    """Whole-run complementary suppression across overlapping patient universes.

    Endpoint, covariate, Pop, topology and complete-case partitions overlap.
    Row-only suppression permits reconstruction across tables. This first
    version withholds *all* patient counts, including zeros and denominators.
    Exact reconciliations and decisions are computed before this boundary.
    Design dimensions/df are also withheld to avoid recovering sample size.
    """
    outputs = {}
    for name, table in tables.items():
        output = table.copy()
        for column in output:
            if column.startswith('n_'):
                output[column] = None
        output['patient_count_disclosure'] = 'WITHHELD_WHOLE_RUN_COMPLEMENTARY_SUPPRESSION'
        output['small_cells_suppressed'] = True
        outputs[name] = output
    return outputs


def _privacy_check(payload, baseline):
    # Review prose, arbitrary category labels, exact clinical dates, identifiers
    # and patient records have no route into the serialized evidence package.
    if re.search(r'\b(?:19|20)\d{2}-\d{2}-\d{2}(?:T|\b)', payload):
        raise ValueError('EXACT_DATE_IN_PHASED_OUTPUT')
    for value in baseline.patient_id.dropna().astype(str):
        if len(value) >= 4 and re.search(r'(?<![A-Za-z0-9_])' + re.escape(value) + r'(?![A-Za-z0-9_])', payload):
            raise ValueError('IDENTIFIER_IN_PHASED_OUTPUT')


def _code_provenance(root, inputs):
    recorded = inputs['phasec_qc'].get('documentary_source_sha256', {})
    rows = []
    for name in sorted(set(CODE_FILES) | set(recorded)):
        path = safe_artifact(root, name)
        sha = digest(path)
        rows.append(dict(source_kind='implementation_code', relative_path=name, sha256=sha,
            phasec_recorded_sha256=recorded.get(name, ''),
            code_drift_status='UNCHANGED' if recorded.get(name) == sha else 'CHANGED' if name in recorded else 'ADDED_SINCE_PHASEC'))
    return rows


def _proposal_text(metadata):
    header = '# Proposed Graph03 contract changes — HUMAN REVIEW REQUIRED\n\n'
    return header + '```json\n' + json.dumps(metadata, indent=2, sort_keys=True) + '\n```\n\n' + '''No change is applied to validation_03.yaml or temporal_resolution.endpoint_policies.

Review characterization.independence_reviews separately for each representation
and source. Code provenance and NO_DOCUMENTED_DEPENDENCE never establish
conceptual independence. Preserve every historical PENDING_REVIEW, BLOCKED_*
and NOT_ESTIMABLE record in the frozen releases.

Review n3_validation.measurement_policies with original instrument/form, assay,
unit, category convention, source-specific lookback and repeated-measurement
rules. The official baseline date, administrative era and endpoint date require
three separate decisions. Episode clinical_anchor_date does not prove the
instrument date. Future-derived measurements are ineligible for baseline
inference. Treatment chronology needs a separate episode audit.

For protocol, confirm the research-protocol meaning, original source,
canonical target, code dictionary, multiple-protocol rule, cohort scope and
content-pinned human signoff. Only a new versioned derived dataset may apply a
verified mapping. If absent, request a formal methodological amendment; do not
drop the covariate, fill a constant or change frozen inputs.

Pop proposal: retain Pop1, Pop2, Pop3, Unclassifiable, Unclassified, missing and
invalid as distinct. Step01's DISPLAY maps presentation labels only; it does
not authorize canonical equivalence. Any signed crosswalk requires a new
release, before/after denominator reconciliation and unchanged descriptive
universe. ESSDAI/ESSPRI and their derivatives remain ineligible with Pop,
even after semantic reconciliation. Never combine ANA assays or sum RF text
and numeric codings without independent same-sample/date/assay evidence.

R0 stable-only sensitivity is a proposal, not executed here: prespecify covered,
untied, supported hard assignments with stable status, patient selection before
outcome inspection, comparator population, denominators, community minimums,
power and uncertainty. Quantify post hoc selection and generalizability bias.
It cannot establish stability of the full structure. R2 remains exploratory.

Prespecify n3_validation.prespecified_outcomes, hypothesis family and correlated
assay handling, pop_comparator_reviews, source_comparators and a reduced reviewed
N1 basis. A contains mandatory covariates and reviewed eligible Pop; B adds
community; C adds reviewed N1 to A; D adds community to C only if requested.
B-A, C-A and D-C require nested designs on the same union complete-case cohort.
Retain minimum group/community 8, model 30, residual df 10 and FDR alpha 0.05
(or stricter existing thresholds). Prespecify uncertainty and bootstrap before
analysis. Unknown category semantics, rank, separation or support block
estimation. Observed Y alone does not establish complete-case feasibility.

A subsequent approved contract and release are mandatory before any model.
These are development-cohort conditional comparisons; independent clinical
incremental validity, external validity and prognosis remain unestablished.
'''


def _review_template(metadata, kind):
    return '# ' + kind + ' — pending human decision\n\n```json\n' + json.dumps(metadata, indent=2, sort_keys=True) + '\n```\n\n' + '''Create a separate YAML/JSON decision document in an institutionally approved
repository location. Its content must exactly match the config record (except
source_reference/source_sha256), plus review_section. Hash it independently;
record the exact representation_id and, for candidate reviews, source_column.
Use PENDING, DOCUMENTED or REJECTED, never an inferred APPROVED value. Record
review_id, reviewer and human_signoff; document presence alone is insufficient.
This tool checks consistency, not reviewer identity or medical competence.

Clinical review: document original instrument, source, transformations,
units/categories, actual date/timezone, lookback, repeated measures, future
measurement exclusion, proxy/construct overlap, indication selection and
missingness. Baseline definition, administrative era and endpoint time are
separate decisions. Assay equivalence must be established independently.

Pop review: adjudicate Unclassifiable vs Unclassified explicitly, retain missing
and invalid, reconcile each universe and preserve ineligible patients in the
descriptive population. Proposed mappings are not executed by Phase D.

Protocol review: establish research-protocol meaning, code dictionary,
multiple-protocol adjudication and exact cohort scope. Any mapping or amendment
requires a new contract and derived release. No signature permits modifying
frozen evidence or estimating a model in this phase.
'''


def run_resolution(graph03_run, graph02_release, phasec_run, output_dir, config_path, repository_root):
    """Emit a new restricted evidence package; never discover runs or estimate.

    Frozen checks execute in finally on success and on every failure after the
    initial snapshot. An incomplete output package is removed on failure.
    """
    root = Path(repository_root).resolve()
    graph03, phasec = Path(graph03_run).resolve(), Path(phasec_run).resolve()
    release = Path(graph02_release).resolve()
    if release.is_dir():
        release = safe_artifact(release, '02_graph_release_manifest.json')
    roots = [graph03, release.parent, phasec]
    if any(not r.is_dir() for r in roots):
        raise ValueError('MISSING_EXPLICIT_FROZEN_INPUT_DIRECTORY')
    if any(a != b and (a.is_relative_to(b) or b.is_relative_to(a)) for a in roots for b in roots) or len(set(roots)) != 3:
        raise ValueError('FROZEN_INPUT_DIRECTORIES_MUST_BE_DISTINCT')
    output = Path(output_dir).resolve()
    _check_overlap(output, roots)
    cfg_path = Path(config_path).resolve()
    config = yaml.safe_load(cfg_path.read_text())
    validate_config(config)
    before = _snapshot(roots)
    protected = {cfg_path: digest(cfg_path)}
    created = False
    try:
        inputs = load_inputs(graph03, release, phasec, config)
        validation_path = safe_artifact(root, 'src/studies/graph/configs/validation_03.yaml')
        validation = yaml.safe_load(validation_path.read_text())
        # Threshold drift is not silently allowed to weaken the historical gates.
        for key, floor in [('minimum_group_patients', 8), ('minimum_model_patients', 30),
                           ('minimum_patients_per_community', 8)]:
            if validation.get(key, 0) < floor:
                raise ValueError('PRESPECIFIED_SUPPORT_THRESHOLD_WEAKENED')
        if validation.get('fdr_alpha', 1) > .05 or validation.get('confounders') != ['demo__protocol', 'demo__age_at_baseline', 'demo__sex']:
            raise ValueError('PRESPECIFIED_VALIDATION_CONTRACT_CHANGED')
        code_rows = _code_provenance(root, inputs)
        protected.update({safe_artifact(root, r['relative_path']): r['sha256'] for r in code_rows})
        for records in config['proposed_reviews'].values():
            for record in records:
                try:
                    path = _reference(root, record.get('source_reference', ''))
                    if path:
                        protected[path] = digest(path)
                except (ValueError, OSError):
                    pass  # Invalid evidence remains a blocker in review_record.
        drift = any(r['code_drift_status'] == 'CHANGED' for r in code_rows)
        if drift and config['policy']['code_drift'] == 'reject':
            raise ValueError('DOCUMENTARY_CODE_DRIFT_REJECTED')
        tables, pop_reconciled = build_tables(inputs, config, root, validation)
        if not pop_reconciled:
            raise ValueError('POP_DENOMINATORS_NOT_RECONCILED')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8]
        pins = config['input_contract']
        metadata = dict(run_id=run_id, schema_version=VERSION,
            graph02_run_id=pins['graph02_run_id'], graph02_release_sha256=pins['graph02_release_sha256'],
            graph03_run_id=pins['graph03_run_id'], graph03_summary_sha256=pins['graph03_summary_sha256'],
            phasec_run_id=pins['phasec_run_id'], phasec_qc_sha256=pins['phasec_qc_sha256'],
            phasec_inventory_sha256=pins['phasec_inventory_sha256'],
            graph03_implementation_commit_sha=pins['graph03_implementation_commit_sha'],
            phasec_implementation_commit_sha=pins['phasec_implementation_commit_sha'],
            implementation_commit_sha=commit, config_sha256=digest(cfg_path),
            privacy_class='INTERNAL_RESTRICTED', shareable_export_allowed=False,
            no_inference=True, no_refit=True, clinical_approval_granted=False,
            integrity_gate='PASS', clinical_gate='HUMAN_SIGNOFF_REQUIRED',
            covariate_gate='NEW_CONTRACT_REQUIRED', temporal_gate='SEPARATE_REVIEW_REQUIRED',
            topology_gate='PRESERVE_FROZEN_GATE', statistical_gate='ESTIMATION_BLOCKED',
            code_drift_gate='BLOCKED_CODE_DRIFT' if drift else 'SOURCE_HASHES_RECONCILED',
            policy_gate='NEW_APPROVED_CONTRACT_AND_RELEASE_REQUIRED')
        tables['model_preflight']['code_drift_gate'] = metadata['code_drift_gate']
        if drift:
            tables['model_preflight']['blocking_reasons'] += '|DOCUMENTARY_CODE_DRIFT_REQUIRES_REVIEW'
        source_rows = []
        for path, sha in before.items():
            owner = next(i for i, r in enumerate(roots) if path.is_relative_to(r))
            source_rows.append(dict(source_kind=['graph03_phasea', 'graph02', 'phasec'][owner],
                relative_path=path.relative_to(roots[owner]).as_posix(), sha256=sha))
        source_rows.extend(code_rows)
        source_rows.append(dict(source_kind='phase_d_config', relative_path='explicit_config', sha256=digest(cfg_path)))
        tables['source_manifest'] = pd.DataFrame(source_rows)
        # Whole-run suppression includes source, model, Pop and topology tables.
        tables = _disclosure_boundary(tables)
        for table in tables.values():
            for key, value in metadata.items():
                # Global gate metadata must not overwrite row-specific gates.
                if key in table:
                    table['run_' + key] = value
                else:
                    table[key] = value
        candidates, models = tables['candidate_evidence_matrix'], tables['model_preflight']
        reps = inputs['manifest']['eligible_representation_ids']
        expected_keys = {(r, s) for r in reps for s in CANDIDATES}
        if (candidates.duplicated(['representation_id', 'source_column']).any() or
                set(map(tuple, candidates[['representation_id', 'source_column']].to_numpy())) != expected_keys or
                len(models) != len(expected_keys)*4 or not models.status.eq('NOT_ESTIMABLE').all()):
            raise ValueError('PHASED_KEYS_OR_ESTIMATION_GUARANTEE_FAILED')
        tests = dict(**metadata, status='PASS', checks=dict(
            explicit_frozen_release_hashes='PASS', phase_a_complete='PASS', phasec_qc_and_inventory='PASS',
            unique_candidate_model_keys='PASS', pop_denominators='PASS', no_estimation='PASS',
            no_recode_or_refit='PASS', no_frozen_writes='PASS', privacy_boundary='PASS'),
            automated_suite='NOT_EXECUTED_BY_ANALYSIS_COMMAND_RUN_PYTEST_SEPARATELY',
            original_release_regression='VERIFIED_SUPPLIED_RELEASES_READ_ONLY',
            patient_counts='WITHHELD_WHOLE_RUN_COMPLEMENTARY_SUPPRESSION')
        qc = dict(**metadata, status='PASS', phased_status='COMPLETE_WITH_BLOCKERS',
            graph02_unchanged=True, graph03_unchanged=True, phasec_unchanged=True,
            source_keys_unique=True, model_preparation_rows_reconciled=True,
            missing_candidate_keys=[dict(representation_id=r, source_column=s) for r, s in inputs['missing_candidate_keys']],
            historical_model_status='NOT_ESTIMABLE', automatic_approvals=0, models_estimated=0,
            source_recoding_applied=False, longitudinal_policies_unchanged=True,
            count_disclosure_policy='whole_run_complementary_suppression_including_denominators_and_df',
            minimum_disclosure_cell=config['minimum_disclosure_cell'])
        pop_qc = dict(**metadata, status='PASS', universes_reconciled=pop_reconciled,
            categories_kept_distinct=True, mapping_applied=False, clinical_equivalence='PENDING_REVIEW')
        chronology_qc = dict(**metadata, status='PENDING_REVIEW', exact_dates_exported=False,
            official_baseline_does_not_approve_era_or_endpoint=True, longitudinal_status='BLOCKED')
        payloads = {PREFIX + name + '.csv': table.to_csv(index=False) for name, table in tables.items()}
        for name, document in [('qc', qc), ('tests_summary', tests), ('pop_reconciliation_qc', pop_qc), ('chronology_qc', chronology_qc)]:
            payloads[PREFIX + name + '.json'] = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n'
        payloads[PREFIX + 'proposed_contract_changes.md'] = _proposal_text(metadata)
        payloads[PREFIX + 'clinical_review_template.md'] = _review_template(metadata, 'Clinical review template')
        payloads[PREFIX + 'pop_reviewer_decision_template.md'] = _review_template(metadata, 'Pop reviewer decision template')
        payloads[PREFIX + 'summary.md'] = '# Graph03 Phase D — restricted preflight\n\n```json\n' + json.dumps(metadata, indent=2, sort_keys=True) + '\n```\n\n' + (
            'Integrity QC: PASS. Phase D: COMPLETE_WITH_BLOCKERS. All historical preparation rows remain '
            'NOT_ESTIMABLE. No clinical approval, estimation, recoding, Mapper refit or frozen edit. '
            'Code evidence, instrument evidence and human review remain separate. '
            'Required evidence is listed per representation/source in the review queue and proposals. '
            'Original forms, assay dates, protocol semantics and expert independence decisions cannot '
            'be inferred from this code. R2 is exploratory; R0 stable-only was not executed. '
            'Patient counts and denominators are withheld across the entire package to prevent '
            'cross-table reconstruction. No external sharing or ZIP packaging is authorized.\n')
        for name, payload in payloads.items():
            _privacy_check(payload, inputs['frozen']['baseline_raw'])
        if _snapshot(roots) != before or any(digest(path) != sha for path, sha in protected.items()):
            raise ValueError('PROTECTED_INPUT_CHANGED_DURING_PHASED')
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        created = True
        for name, payload in payloads.items():
            (output / name).write_text(payload)
            (output / name).chmod(0o600)
        artifacts = pd.DataFrame([dict(**metadata, relative_path=name, sha256=digest(output / name)) for name in sorted(payloads)])
        artifacts.to_csv(output / (PREFIX + 'artifact_inventory.csv'), index=False)
        (output / (PREFIX + 'artifact_inventory.csv')).chmod(0o600)
        return dict(status='PASS', phased_status='COMPLETE_WITH_BLOCKERS', run_id=run_id,
                    output_dir=str(output), models_estimated=0)
    except BaseException:
        if created:
            import shutil
            shutil.rmtree(output)
        raise
    finally:
        if _snapshot(roots) != before or any(not path.is_file() or digest(path) != sha for path, sha in protected.items()):
            if created and output.exists():
                import shutil
                shutil.rmtree(output)
            raise ValueError('PROTECTED_INPUT_CHANGED_DURING_PHASED')

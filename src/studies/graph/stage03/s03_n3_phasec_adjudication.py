"""Additive Phase C documentary audit. No estimators, recoding, or approvals.

Only verified, explicitly selected frozen inputs enter this boundary. Evidence
of code existence is distinct from human review and measurement provenance.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4

import pandas as pd
import yaml

from .._pipeline import digest, git_commit, write_json
from ..stage02.s02_contracts import safe_artifact, validate_release
from .s03_n3_candidate_audit import _proxy_alerts
from .s03_release_reader import hard_members

VERSION = 'graph03_n3_phasec_v1'
REFERENCE_RUN = '20261104T180702279234Z_961ffc6f'
CANDIDATES = {
    'essdai__total': 'systemic_activity',
    'esspri__total': 'symptoms',
    'ext__biopsy_focus_score': 'glandular_histopathology',
    'lab__ana_status__text': 'ana',
    'lab__ana_hep2_status__text': 'ana',
    'lab__rheumatoid_factor__text': 'rheumatoid_factor',
    'lab__rheumatoid_factor__value': 'rheumatoid_factor',
    'lab__urine_protein_qualitative__text': 'qualitative_proteinuria',
}
COMMON = ['decision_status', 'evidence_status', 'reviewer_action', 'blocking_reasons',
          'representation_id', 'source_column', 'input_graph03_run_id',
          'graph02_release_sha256', 'privacy_class']
INPUT_FILES = [f'03_graph_n3_{name}.csv' for name in (
    'source_inventory', 'candidate_screen', 'lineage_review_queue', 'review_status', 'endpoint_support')
] + ['03_graph_n3_validation_qc.json', '03_graph_n3_results_summary.md',
     '03_graph_results_summary.json', '03_graph_release_input_qc.json',
     '03_graph_clinical_feature_roles.csv']
PREFIX = '03_graph_n3_phasec_'
# Actual context namespaces retained by Step10; presence is not equivalence.
PROTOCOL_ALIASES = ['demo__protocol', 'protocol', 'ids__protocol', 'ids__protocol_number',
    'parent_protocol', 'source_protocol', 'spine__source_protocol', 'pop__protocol', 'pro__parent_protocol']
# These are documentary mappings established by inspecting code, not clinical
# measurement assertions. Laboratory original fields/assay selection stay pending.
LINEAGE = {
    'systemic_activity': ('src/block_A/01_essdai_esspri.py:derive_essdai_total',
        'essdai__essdai_total_score', 'canonical_upstream_total;numeric_parse;range_0_123'),
    'symptoms': ('src/block_A/01_essdai_esspri.py:derive_esspri_total',
        'esspri_questionnaire__dryness|esspri_questionnaire__fatigue|esspri_questionnaire__pain',
        'complete_three_components;range_0_10;arithmetic_mean'),
    'glandular_histopathology': ('src/block_A/01_extended_clinical_phenotype.py:derive_pathology',
        'biopsy_pathology__f_score', 'numeric_parse;discordant_delimited_results_masked'),
}
LAB_REFERENCE = 'src/block_A/01_serological_profile.py:select_episode_analytes'
DOCUMENTARY_FILES = [
    'src/studies/graph/stage03/s03_construct_validation.py',
    'src/studies/graph/stage03/s03_n3_candidate_audit.py',
    'src/studies/graph/stage03/s03_n3_validation.py',
    'src/studies/graph/stage03/s03_n3_incremental.py',
    'src/studies/graph/configs/validation_03.yaml',
    'src/block_A/01_essdai_esspri.py',
    'src/block_A/01_extended_clinical_phenotype.py',
    'src/block_A/01_serological_profile.py',
    'src/block_A/10_build_integrated_longitudinal_dataset.py',
    'src/block_A/11_integrated_baseline_characterization.py',
    'src/block_A/observed_patterns.py', 'src/00_build_visit_spine.py',
    'src/studies/_shared.py', 'src/studies/graph/stage02/s02_missingness.py',
    'src/studies/graph/stage03/s03_profiles.py',
    'src/studies/graph/stage03/s03_n3_phasec_adjudication.py',
    'src/studies/graph/stage03/s03_run_resolution.py',
    'src/studies/graph/03c_graph_n3_clinical_adjudication.py',
]
ACTION = 'DOCUMENT_LINEAGE_MEASUREMENT_DATE_UNITS_CATEGORIES_AND_CONCEPTUAL_INDEPENDENCE'


def _present(value):
    return value is True or isinstance(value, str) and value == 'True'


def _tokens(value):
    return set(str(value).split('|')) - {'', 'nan', 'None'}


def _number(row, field):
    value = row.get(field)
    return int(value) if pd.notna(value) else None


def validate_phasec_config(config):
    if not isinstance(config, dict) or config.get('schema_version') != VERSION:
        raise ValueError('UNSUPPORTED_PHASEC_SCHEMA')
    if config.get('auto_approve_independence') is not False:
        raise ValueError('AUTOMATIC_PHASEC_APPROVAL_FORBIDDEN')
    if config.get('privacy_class') != 'INTERNAL_RESTRICTED':
        raise ValueError('PHASEC_REQUIRES_INTERNAL_RESTRICTED')
    if int(config.get('minimum_disclosure_cell', 0)) < 5:
        raise ValueError('PHASEC_DISCLOSURE_THRESHOLD_TOO_LOW')
    contract = config.get('input_contract', {})
    if not isinstance(contract, dict):
        raise ValueError('PHASEC_INPUT_CONTRACT_MUST_BE_MAPPING')
    if contract.get('n3_schema_version') != 'graph03_n3_audit_v1':
        raise ValueError('UNSUPPORTED_PHASE_A_SCHEMA')
    for field, size in [('graph02_release_sha256', 64), ('implementation_commit_sha', 40)]:
        if field in contract and not re.fullmatch('[a-f0-9]{%d}' % size, str(contract[field])):
            raise ValueError('INVALID_PHASEC_PIN_' + field)
    declared = {source: family for family, record in config.get('families', {}).items()
                for source in record.get('sources', [])}
    if declared != CANDIDATES:
        raise ValueError('PHASEC_SIX_FAMILY_CONTRACT_CHANGED')
    if sum(len(r['sources']) for r in config['families'].values()) != len(CANDIDATES):
        raise ValueError('DUPLICATE_PHASEC_FAMILY_SOURCE')
    for key in ['candidate_reviews', 'protocol_reviews', 'baseline_era_reviews']:
        records = config.get(key, [])
        keys = [(r.get('representation_id'), r.get('source_column', '') if key == 'candidate_reviews' else '') for r in records]
        if len(keys) != len(set(keys)):
            raise ValueError('DUPLICATE_PHASEC_REVIEW')
        if any(not r.get('representation_id') for r in records):
            raise ValueError('UNSCOPED_PHASEC_REVIEW')
        if any(r['representation_id'] not in {'R0','R1','R2','R3','R4'} for r in records):
            raise ValueError('INVALID_PHASEC_REVIEW_REPRESENTATION')
        if key == 'candidate_reviews' and any(r.get('source_column') not in CANDIDATES for r in records):
            raise ValueError('INVALID_PHASEC_REVIEW_SOURCE')
        if any(r.get('status', 'PENDING_REVIEW') not in {
                'PENDING_REVIEW', 'DOCUMENTED', 'REJECTED'} for r in records):
            raise ValueError('FINAL_APPROVAL_NOT_SUPPORTED_BY_PHASEC')


def _reference(root, reference, expected_sha=None):
    if not re.fullmatch(r'(?:src|docs|tests)/[A-Za-z0-9_./-]+(?::[A-Za-z_][A-Za-z0-9_]*)?', reference):
        return {}
    relative, _, symbol = reference.partition(':')
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        return {}
    if symbol and symbol not in path.read_text():
        return {}
    sha = digest(path)
    if expected_sha is not None and sha != expected_sha:
        return {}
    return dict(source_code_reference=reference, source_code_sha256=sha)


def _lineage(root, family):
    reference, original, transform = LINEAGE.get(family, (
        LAB_REFERENCE, 'ORIGINAL_ASSAY_FIELD_PENDING_REVIEW',
        'upstream_20_20b_episode_match;closest_absolute_anchor_distance;pre_anchor_tie_priority;'
        'discordance_and_unit_conflicts_masked;harmonize_selected_units;build_wide'))
    evidence = _reference(root, reference)
    if not evidence:
        return dict(source_code_reference='', source_code_sha256='', source_table_or_form='',
                    original_field='', upstream_transformations='', evidence_status='NO_EVIDENCE')
    return dict(**evidence, source_table_or_form='episode_spine' if family in LINEAGE else
                'canonical_episode_analytes_from_EDA20_20b', original_field=original,
                upstream_transformations=transform, evidence_status='CODE_DOCUMENTED_MEASUREMENT_PENDING')


def _signed_document(record, evidence):
    """Observe a supplied human signoff; file existence alone is insufficient."""
    return bool(record.get('status') == 'DOCUMENTED' and record.get('human_signoff') is True
        and isinstance(record.get('reviewer'),str) and record['reviewer'].strip()
        and evidence.get('verified') is True and record.get('source_sha256') == evidence.get('sha256')
        and re.fullmatch('[a-f0-9]{64}',evidence.get('sha256','')))


def audit_phasec_candidate(inventory_row, frozen_roles, lineage_docs, reviews):
    """Recommend documentary disposition; never create N3 or clinical approval."""
    row, defining = inventory_row, set(frozen_roles)
    source = row['source_column']
    family = CANDIDATES[source]
    alerts = _proxy_alerts(source, defining, {})
    alerts += sorted(_tokens(row.get('clinical_proxy_alerts', '')))
    if family == 'ana' and any(c.startswith('lab__') for c in defining):
        alerts.append('ANA_VS_DISCOVERY_SEROLOGY_REVIEW')
    if family == 'qualitative_proteinuria' and any('renal' in c or 'urine' in c for c in defining):
        alerts.append('RENAL_URINARY_CONSTRUCT_REVIEW')
    related = sorted(defining & _tokens(row.get('conflicting_defining_sources', '')))
    dependent = (row.get('tier_original') in {'N1_DEFINING', 'N2_RELATED'} or
                 row.get('review_state') == 'BLOCKED_DEPENDENT' or source in defining or bool(related))
    unavailable = not _present(row.get('source_exists')) or not _number(row, 'n_observed_unique')
    reasons = ['PHASEC_DOCUMENTARY_ONLY_NO_INFERENCE']
    if dependent:
        decision, evidence, clinical = 'BLOCKED_RELATED_OR_PROXY', 'NOT_ELIGIBLE', 'NOT_ELIGIBLE'
        reasons.append('DOCUMENTED_DISCOVERY_DEPENDENCE')
    elif unavailable:
        decision, evidence, clinical = 'BLOCKED_UNAVAILABLE_OR_INVALID', 'SOURCE_UNAVAILABLE', 'NOT_ELIGIBLE'
        reasons.append('NO_SUPPORTED_OBSERVED_SOURCE')
    else:
        decision, clinical = 'PENDING_DOCUMENTARY_EVIDENCE', 'PENDING_REVIEW'
        evidence = 'PROXY_REVIEW_PENDING' if alerts else lineage_docs.get('evidence_status', 'NO_EVIDENCE')
        reasons.extend(['HUMAN_CLINICAL_SIGNOFF_REQUIRED', 'ENDPOINT_MEASUREMENT_DATE_PENDING',
                        'UNIT_OR_CATEGORY_REVIEW_PENDING'])
        if alerts: reasons.append('CONCEPTUAL_PROXY_REVIEW_PENDING')
        if not lineage_docs.get('source_code_sha256'): reasons.append('SOURCE_CODE_EVIDENCE_MISSING')
        # A document must be content-pinned and structurally complete. Even this
        # state requests expert review and cannot authorize any downstream model.
        complete = (reviews.get('document_verified') is True and reviews.get('status') == 'DOCUMENTED'
            and reviews.get('baseline_definition') in {'baseline_observed','lookback_baseline','ever_through_episode'}
            and reviews.get('date_semantics') == 'instrument_measurement'
            and reviews.get('derived_from_future') == 'false'
            and reviews.get('independence_status') == 'DOCUMENTED'
            and reviews.get('unit_status' if row.get('type') == 'numeric' else 'category_status') == 'DOCUMENTED')
        if reviews.get('derived_from_future') == 'true':
            decision, clinical, evidence = 'BLOCKED_UNAVAILABLE_OR_INVALID', 'NOT_ELIGIBLE', 'NOT_ELIGIBLE'
            reasons.append('DOCUMENTED_FUTURE_DERIVED_ENDPOINT')
        elif complete:
            decision, clinical = 'HUMAN_REVIEW_REQUIRED', 'ELIGIBLE_FOR_HUMAN_REVIEW'
            evidence = 'DOCUMENTARY_PACKAGE_COMPLETE_HUMAN_DECISION_PENDING'
            reasons = ['HUMAN_CLINICAL_SIGNOFF_REQUIRED', 'PHASEC_DOCUMENTARY_ONLY_NO_INFERENCE']
        if reviews.get('status') == 'REJECTED':
            decision, clinical, evidence = 'BLOCKED_UNAVAILABLE_OR_INVALID', 'NOT_ELIGIBLE', 'NOT_ELIGIBLE'
            reasons.append('DOCUMENTED_HUMAN_REJECTION')
    caution = {
        'glandular_histopathology': 'BIOPSY_INDICATION_SELECTION_AND_MISSING_BY_DESIGN_REQUIRE_REVIEW',
        'qualitative_proteinuria': 'NOT_SJD_RENAL_DIAGNOSIS;REPEATED_SAMPLES_CONTAMINATION_ALTERNATIVE_CAUSES',
        'ana': 'DISTINCT_ASSAYS_NO_MERGE_OR_COVERAGE_BASED_SELECTION',
        'rheumatoid_factor': 'TEXT_VALUE_SAME_SAMPLE_EQUIVALENCE_UNVERIFIED_NO_SUMMATION',
    }.get(family, 'BASELINE_AVAILABILITY_AND_CONSTRUCT_OVERLAP_REQUIRE_REVIEW')
    possible = sorted(c for c in defining if
        family == 'symptoms' and c.startswith(('pro__sf36_', 'pro__profad_', 'pro__mdafs_')) or
        family == 'systemic_activity' and (c.startswith(('essdai__','eg_')) or c == 'ovl__status') or
        family == 'ana' and c.startswith('lab__') or
        family == 'qualitative_proteinuria' and ('renal' in c or 'urine' in c))
    return dict(representation_id=row['representation_id'], source_column=source, candidate_family=family,
        decision_status=decision, evidence_status=evidence, clinical_status=clinical,
        reviewer_action=ACTION, blocking_reasons='|'.join(sorted(set(reasons))),
        proxy_review_flag='|'.join(sorted(set(alerts))) or 'NO_AUTOMATIC_INDEPENDENCE_CLAIM',
        related_N1_columns='|'.join(related), possible_proxy_N1_columns='|'.join(possible),
        dependency_path='FROZEN_PHASEA_LINEAGE' if dependent else 'NOT_ESTABLISHED',
        sampling_bias_note=caution, observation_eligibility='PENDING_REVIEW',
        pop_comparator_status='POP_BASELINE_COMPARATOR_INELIGIBLE' if family in {
            'systemic_activity','symptoms'} else 'POP_COMPARATOR_REVIEW_PENDING',
        original_tier=row.get('tier_original', ''), original_review_state=row.get('review_state', ''),
        **{key:_number(row,key) for key in ['n_hard_supported','n_observed_unique','n_missing',
                                           'n_stable_hard','n_provisional_hard']})


def audit_phasec_covariates(frozen_baseline_schema, upstream_code, config):
    """Inspect availability only; never join aliases into the frozen baseline."""
    frame = frozen_baseline_schema
    canonical = 'demo__protocol'
    aliases = [c for c in PROTOCOL_ALIASES[1:] if c in frame]
    present = canonical in frame
    status = 'ABSENT' if not present else 'ALL_NULL' if frame[canonical].isna().all() else 'AVAILABLE'
    decision = 'ABSENT_IN_FROZEN_BASELINE' if not present else 'NOT_REVIEWED'
    alias_status = 'NOT_REVIEWED'
    review = config.get('review',{})
    reviewed_source = review.get('canonical_source')
    if _signed_document(review,upstream_code):
        if review.get('equivalent_to_prespecified_confounder') is False:
            decision = alias_status = 'AVAILABLE_BUT_NOT_EQUIVALENT'
        elif (review.get('equivalent_to_prespecified_confounder') is True and
              reviewed_source in [canonical,*aliases] and reviewed_source in frame and
              frame[reviewed_source].notna().any()):
            decision = alias_status = 'AVAILABLE_WITH_VERIFIED_CANONICAL_MAPPING'
    return dict(source_column=canonical, decision_status=decision,
        evidence_status='SUPPLIED_HUMAN_MAPPING_SIGNOFF_VERIFIED' if alias_status != 'NOT_REVIEWED' else
                        'SOURCE_UNAVAILABLE' if status in {'ABSENT','ALL_NULL'} else 'CANONICAL_SEMANTICS_PENDING',
        reviewer_action='VERIFY_RESEARCH_PROTOCOL_EQUIVALENCE_OR_PROPOSE_PROTOCOL_AMENDMENT',
        blocking_reasons='NO_ALIAS_FILL_NO_IMPUTATION_NO_ADJUSTMENT_REMOVAL', canonical_field_status=status,
        available_aliases='|'.join(aliases), alias_review_status=alias_status,
        review_evidence_sha256=review.get('source_sha256','') if _signed_document(review,upstream_code) else '',
        n_baseline=len(frame), n_canonical_nonmissing=int(frame[canonical].notna().sum()) if present else 0,
        n_canonical_unique=int(frame[canonical].nunique()) if present else 0,
        alias_coverage_json=json.dumps({c:int(frame[c].notna().sum()) for c in aliases}, sort_keys=True),
        source_code_reference='src/studies/_shared.py|src/studies/graph/stage02/s02_missingness.py')


def audit_phasec_baseline_date(step10_contract, step11_contract, evidence):
    """Date quality is documentary evidence; era never verifies endpoint dates."""
    frame = step10_contract
    anchor = pd.to_datetime(frame.get('clinical_anchor_date', pd.Series(pd.NaT,index=frame.index)),
                            errors='coerce', utc=True)
    official = pd.to_datetime(frame.get('clinical_baseline_date', pd.Series(pd.NaT,index=frame.index)),
                              errors='coerce', utc=True)
    comparable = anchor.notna() & official.notna()
    verified = (_signed_document(step11_contract,evidence) and
                step11_contract.get('date_semantics') == 'official_baseline_anchor' and
                step11_contract.get('timezone_review_status') == 'DOCUMENTED' and
                step11_contract.get('lookahead_review_status') == 'DOCUMENTED' and
                step11_contract.get('official_baseline_contract_review_status') == 'DOCUMENTED' and
                len(anchor) > 0 and anchor.notna().all() and
                not (anchor.ne(official) & comparable).any())
    return dict(source_column='baseline_era', decision_status='BASELINE_ERA_VERIFIED' if verified else 'PENDING_REVIEW',
        evidence_status='SUPPLIED_HUMAN_ERA_SIGNOFF_VERIFIED' if verified else 'BASELINE_ERA_REVIEW_PENDING',
        reviewer_action='REVIEW_BASELINE_ANCHOR_TIMEZONE_LOOKAHEAD_AND_ERA_CONTRACT_SEPARATELY',
        blocking_reasons='ENDPOINT_DATE_INDEPENDENT_REVIEW_REQUIRED' if verified else 'ERA_DATE_SEMANTICS_AND_HUMAN_REVIEW_REQUIRED',
        anchor_field_status='AVAILABLE' if anchor.notna().any() else 'SOURCE_UNAVAILABLE',
        n_baseline=len(frame), n_valid_anchor=int(anchor.notna().sum()),
        n_invalid_or_missing_anchor=int(anchor.isna().sum()), n_distinct_eras=int((anchor.dt.year//5).nunique()),
        n_anchor_baseline_comparable=int(comparable.sum()),
        n_anchor_baseline_mismatches=int((anchor.ne(official) & comparable).sum()),
        n_missing_official_baseline_date=int(official.isna().sum()),
        temporal_order_status='NOT_ASSESSABLE_FROM_ONE_FROZEN_BASELINE_PER_PATIENT',
        timezone_status='DOCUMENTED_IN_SUPPLIED_HUMAN_REVIEW' if verified else
                        'SOURCE_TIMEZONE_NOT_VERIFIED_UTC_USED_FOR_AUDIT_PARSING_ONLY',
        era_rule='fixed_five_year_strata_existing_adjustment_data_contract',
        endpoint_measurement_date_status='PENDING_REVIEW', longitudinal_status='BLOCKED',
        review_evidence_sha256=step11_contract.get('source_sha256','') if verified else '',
        source_code_reference='src/block_A/10_build_integrated_longitudinal_dataset.py|src/block_A/11_integrated_baseline_characterization.py')


def audit_phasec_pop_dictionary(source_contract, frozen_values=None, reporting_contract=None):
    """Exact, allowlisted category reconciliation. Arbitrary raw tokens stay private."""
    series = source_contract if frozen_values is None else frozen_values
    counts = {key:int(series.eq(label).fillna(False).sum()) for key,label in [
        ('n_pop1','Pop1'),('n_pop2','Pop2'),('n_pop3','Pop3'),
        ('n_unclassifiable','Unclassifiable'),('n_unclassified','Unclassified')]}
    missing = int(series.isna().sum())
    other = len(series)-sum(counts.values())-missing
    return dict(source_column='pop__status', decision_status='PENDING_REVIEW',
        evidence_status='VOCABULARY_MISMATCH_REQUIRES_REVIEW' if counts['n_unclassifiable'] else 'AUDITED_NO_SEMANTIC_APPROVAL',
        reviewer_action='CONFIRM_UNCLASSIFIABLE_UNCLASSIFIED_EQUIVALENCE_IN_SEPARATE_PATCH',
        blocking_reasons='NO_POP_RECODING_IN_PHASEC', **counts, n_missing=missing,
        n_other_invalid=other, n_total=len(series), reconciled=sum(counts.values())+missing+other==len(series),
        recoded=False, source_vocabulary='Pop1|Pop2|Pop3|Unclassifiable',
        graph03_vocabulary='Pop1|Pop2|Pop3|Unclassified')


def build_phasec_tables(inventory, roles, feature_sets, repository_root, config):
    """One documentary row per representation/family/source; A/B/C/D stay blocked."""
    validate_phasec_config(config)
    if not inventory.empty:
        _check_keys(inventory,'inventory')
    reps = sorted(inventory.representation_id.unique())
    validation = yaml.safe_load((Path(repository_root) / 'src/studies/graph/configs/validation_03.yaml').read_text())
    reviews = {(r['representation_id'],r['source_column']):r for r in config.get('candidate_reviews', [])}
    rows, lineages, measurements, models = [], [], [], []
    for rep in reps:
        defining = set(feature_sets.loc[feature_sets.representation_id.eq(rep) &
                       feature_sets.included.eq(True), 'source_column'])
        for source, family in CANDIDATES.items():
            found = inventory.loc[inventory.representation_id.eq(rep) & inventory.source_column.eq(source)]
            if found.empty:
                # This is an audit of a missing source, not an invented patient record.
                raw = dict(representation_id=rep, source_column=source, source_exists=False)
            else:
                raw = found.iloc[0].to_dict()
            docs = _lineage(repository_root, family)
            review = dict(reviews.get((rep,source), {}))
            reference = review.get('source_reference', '')
            review['document_verified'] = bool(review.get('source_sha256') and
                _reference(repository_root,reference,review['source_sha256']))
            row = audit_phasec_candidate(raw, defining, docs, review)
            role = roles.loc[roles.representation_id.eq(rep) & roles.source_column.eq(source)]
            if len(role) and role.iloc[0].tier in {'N1_DEFINING','N2_RELATED'}:
                row.update(decision_status='BLOCKED_RELATED_OR_PROXY',clinical_status='NOT_ELIGIBLE',
                           evidence_status='NOT_ELIGIBLE',blocking_reasons='FROZEN_ROLE_DEPENDENT_OR_PROXY')
            rows.append(row)
            lineages.append(dict(**{**docs, **row}, source_grain='patient_x_clinical_episode;assay_selection_pending' if
                source.startswith('lab__') else 'patient_x_clinical_episode',
                baseline_definition='official_is_clinical_baseline_frozen_source;instrument_window_unknown',
                measurement_date_field='selected_lab_date_upstream_not_verified_for_endpoint' if
                source.startswith('lab__') else 'INSTRUMENT_DATE_NOT_ESTABLISHED',
                date_semantics='clinical_anchor_is_episode_date_not_instrument_date',
                baseline_temporal_evidence='PENDING_REVIEW', derived_from_future='unknown',
                data_type=raw.get('type','unknown'), unit_status='PENDING_REVIEW',
                category_status='PENDING_REVIEW', reviewer_required=True, reason=row['blocking_reasons'],
                integrated_mapping_reference='src/block_A/10_build_integrated_longitudinal_dataset.py',
                review_evidence_sha256=review.get('source_sha256','') if review['document_verified'] else ''))
            measurements.append(dict(**row, data_type=raw.get('type','unknown'), unit_status='PENDING_REVIEW',
                category_status='PENDING_REVIEW', category_mapping_applied=False, alternative_assay_merge=False,
                measurement_date_status='PENDING_REVIEW', negative_unknown_not_tested_distinct=True,
                assay_equivalence_status='PENDING_REVIEW', thresholds_status='PENDING_REVIEW'))
            for model in ['A','B','C','D']:
                complete_covariates = _number(raw,'n_Y_nonmissing_with_all_covariates')
                support = 'COMMON_POPULATION_NOT_EVALUATED'
                if complete_covariates is not None and complete_covariates < validation['minimum_model_patients']:
                    support = 'INSUFFICIENT_COVARIATE_COMPLETE_PATIENTS'
                models.append(dict(**{**row, 'decision_status':'NOT_ESTIMABLE'}, model=model,
                    documentary_gate='HUMAN_SIGNOFF_REQUIRED',
                    topology_gate='BLOCKED_PROVISIONAL_TOPOLOGY' if _number(raw,'n_provisional_hard') else
                                  'TOPOLOGY_REQUIRES_RELEASE_REVIEW',
                    support_gate=support,
                    covariate_gate='BLOCKED' if _tokens(raw.get('missing_adjustments','')) else 'REQUIRES_REVIEW',
                    missing_adjustments='|'.join(sorted(_tokens(raw.get('missing_adjustments','')))),
                    n_Y_nonmissing_with_all_covariates=complete_covariates,
                    n_common_complete_case=None,
                    common_population_verified=False, design_gate='DF_COLLINEARITY_NOT_EVALUATED',
                    minimum_model_patients=validation['minimum_model_patients'],
                    minimum_group_patients=validation['minimum_group_patients'],
                    minimum_residual_df=10, outcome_class=raw.get('type','unknown'),
                    binary_event_support_status='EVENTS_PER_PARAMETER_REQUIRES_OUTCOME_REVIEW',
                    n1_basis_columns='', adjusted_covariates='|'.join(validation['confounders'])+'|baseline_era',
                    source_comparator_status='REDUCED_N1_BASIS_REQUIRES_PRESPECIFICATION' if model in {'C','D'} else 'NOT_APPLICABLE',
                    model_contract={'A':'covariates_plus_eligible_reviewed_Pop',
                                    'B':'A_plus_frozen_community', 'C':'A_plus_reviewed_reduced_N1',
                                    'D':'C_plus_frozen_community_if_requested'}[model],
                    multiplicity_plan_status='REQUIRES_PRESPECIFICATION',
                    uncertainty_scope='conditional_on_frozen_memberships_excludes_Mapper_selection'))
    tables = dict(candidate_adjudication=pd.DataFrame(rows), source_lineage=pd.DataFrame(lineages),
                  measurement_review=pd.DataFrame(measurements), model_readiness=pd.DataFrame(models))
    tables['human_review_queue'] = tables['candidate_adjudication'].copy()
    tables['human_review_queue']['next_action'] = ACTION
    return tables


def _check_keys(frame, label):
    keys = ['representation_id','source_column']
    if not set(keys) <= set(frame) or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('DUPLICATE_OR_MISSING_PHASEC_KEYS_'+label)


def _reconcile_reference(inventory, qc):
    """Reference-only invariants, never constraints on a subsequent release."""
    candidates = inventory.loc[inventory.is_outcome_candidate_for_review.eq(True)]
    if (len(inventory) != 2525 or inventory.source_column.nunique() != 505 or
        set(inventory.representation_id) != {'R0','R1','R2','R3','R4'} or
        not inventory.groupby('representation_id').size().eq(505).all() or
        len(candidates) != 420 or candidates.source_column.nunique() != 88 or
        not inventory.n_baseline_total.eq(159).all() or
        qc.get('n_reviewed_independent_sources') != 0 or qc.get('n_models_complete') != 0):
        raise ValueError('REFERENCE_RELEASE_DISTRIBUTION_DIVERGED')
    expected = dict(zip(CANDIDATES, [(86,84),(66,67),(58,55),(63,66),(42,45),
                                    (46,48),(44,37),(88,82)]))
    indexed = inventory.set_index(['representation_id','source_column'])
    for source, counts in expected.items():
        for rep, count, hard, stable in zip(['R0','R2'], counts, [92,87], [62,0]):
            row = indexed.loc[(rep,source)]
            if (row.n_observed_unique != count or row.n_hard_supported != hard or
                row.n_stable_hard != stable or row.n_Y_nonmissing_with_all_covariates != 0):
                raise ValueError('REFERENCE_ENDPOINT_COVERAGE_DIVERGED')


def load_phasec_inputs(graph03_run, graph02_release, config, *, strict=False):
    """Verify all contracts and hashes before generating conclusions or outputs."""
    validate_phasec_config(config)
    contract = config['input_contract']
    manifest, release = validate_release(graph02_release,
        expected_sha=contract.get('graph02_release_sha256') if strict else None)
    release_sha = digest(release)
    root = Path(graph03_run).resolve()
    if strict and (contract.get('graph02_run_id', manifest['run_id']) != manifest['run_id'] or
                   contract.get('graph03_run_id', root.name) != root.name):
        raise ValueError('FROZEN_RUN_PIN_MISMATCH')
    paths = {name:safe_artifact(root,name) for name in INPUT_FILES}
    summary = json.loads(paths['03_graph_results_summary.json'].read_text())
    qc = json.loads(paths['03_graph_n3_validation_qc.json'].read_text())
    input_qc = json.loads(paths['03_graph_release_input_qc.json'].read_text())
    if any(not isinstance(doc, dict) for doc in [summary, qc, input_qc]):
        raise ValueError('PHASE_A_METADATA_MUST_BE_JSON_OBJECT')
    if summary.get('schema_version') != 'graph03_results_v1':
        raise ValueError('UNSUPPORTED_GRAPH03_SUMMARY_SCHEMA')
    if (summary.get('execution_status') not in {'COMPLETE', 'COMPLETE_WITH_LIMITATIONS'} or
            (root / '03_graph_failure.json').exists()):
        raise ValueError('GRAPH03_FAILED_OR_INCOMPLETE_RUN')
    if not isinstance(summary.get('sha256_by_artifact'), dict):
        raise ValueError('GRAPH03_ARTIFACT_HASH_INVENTORY_MUST_BE_MAPPING')
    input_hashes = {name:digest(path) for name,path in paths.items()}
    for name in paths:
        if name != '03_graph_results_summary.json' and summary.get('sha256_by_artifact',{}).get(name) != input_hashes[name]:
            raise ValueError('GRAPH03_INPUT_SHA_MISMATCH')
    for document in [summary,qc,input_qc]:
        if (document.get('graph02_release_sha256') != release_sha or
                document.get('graph02_run_id') != manifest['run_id']):
            raise ValueError('GRAPH03_FROZEN_RELEASE_SHA_OR_PROVENANCE_MISMATCH')
    if (qc.get('graph03_run_id') != root.name or qc.get('phase_a_status') != 'COMPLETE' or
        qc.get('schema_version') != contract['n3_schema_version'] or
        qc.get('status') not in {'PASS','COMPLETE_WITH_LIMITATIONS'} or input_qc.get('status') != 'PASS'):
        raise ValueError('PHASE_A_SCHEMA_OR_COMPLETION_MISMATCH')
    if not re.fullmatch('[a-f0-9]{40}', str(qc.get('implementation_commit_sha', ''))):
        raise ValueError('INVALID_PHASE_A_CODE_PROVENANCE')
    if strict and qc['implementation_commit_sha'] != contract.get('implementation_commit_sha', qc['implementation_commit_sha']):
        raise ValueError('PHASE_A_IMPLEMENTATION_PIN_MISMATCH')
    tables = {name.removeprefix('03_graph_n3_').removesuffix('.csv'):pd.read_csv(path)
              for name,path in paths.items() if name.startswith('03_graph_n3_') and name.endswith('.csv')}
    roles = pd.read_csv(paths['03_graph_clinical_feature_roles.csv'])
    for name, table in {**tables,'roles':roles}.items(): _check_keys(table,name)
    inventory = tables['source_inventory']
    required = {'n_observed_unique','n_missing','n_hard_supported','n_baseline_total',
                'n_stable_hard','n_provisional_hard','n_Y_nonmissing_with_all_covariates',
                'source_exists','tier_original','review_state','missing_adjustments'}
    if not required <= set(inventory):
        raise ValueError('INCOMPLETE_PHASE_A_INVENTORY')
    reps = manifest['eligible_representation_ids']
    if (set(inventory.representation_id) != set(reps) or
            not set(reps) <= {'R0','R1','R2','R3','R4'} or
            not inventory.groupby('representation_id').size().nunique() == 1 or
            len(inventory) != inventory.source_column.nunique()*len(reps)):
        raise ValueError('PHASE_A_SOURCE_REPRESENTATION_GRAIN_MISMATCH')
    for key, table in [('candidate_screen',tables['candidate_screen']),('review_status',tables['review_status']),('roles',roles)]:
        if set(map(tuple,table[['representation_id','source_column']].to_numpy())) != set(
                map(tuple,inventory[['representation_id','source_column']].to_numpy())):
            raise ValueError('PHASE_A_TABLE_KEYS_NOT_RECONCILED')
    support = tables['endpoint_support']
    if not support.empty:
        joined = support.merge(inventory,on=['representation_id','source_column'],suffixes=('_support','_inventory'),validate='one_to_one')
        if len(joined) != len(support) or not joined.n_observed_Y.eq(joined.n_observed_unique).all():
            raise ValueError('PHASE_A_ENDPOINT_SUPPORT_NOT_RECONCILED')
    frozen = {role:pd.read_csv(safe_artifact(release.parent,manifest['relative_paths'][role]),
                              dtype={'patient_id':'string'})
              for role in ['baseline_raw','membership','feature_sets']}
    baseline, membership = frozen['baseline_raw'], frozen['membership']
    if (baseline.patient_id.isna().any() or baseline.patient_id.duplicated().any() or
            len(baseline) != manifest['n_patients']):
        raise ValueError('NONUNIQUE_OR_INCOMPLETE_FROZEN_BASELINE')
    for source, rows in inventory.groupby('source_column'):
        if not rows.source_exists.isin([True,False]).all() or not rows.source_exists.eq(source in baseline).all():
            raise ValueError('FROZEN_SOURCE_AVAILABILITY_NOT_RECONCILED')
    for column in ['hard_assignment','mapper_covered','membership_tie','supported']:
        if column not in membership or membership[column].isna().any() or not membership[column].isin([True,False]).all():
            raise ValueError('INVALID_FROZEN_MEMBERSHIP_FLAGS')
    if not set(membership.patient_id) <= set(baseline.patient_id):
        raise ValueError('FROZEN_MEMBERSHIP_OUTSIDE_BASELINE')
    if (membership.hard_assignment & (~membership.mapper_covered | membership.membership_tie)).any():
        raise ValueError('INCONSISTENT_FROZEN_HARD_ASSIGNMENT')
    if (membership.duplicated(['patient_id','representation_id','community_id']).any() or
        membership.patient_id.isna().any()):
        raise ValueError('NONUNIQUE_FROZEN_MEMBERSHIP')
    if not frozen['feature_sets'].included.isin([True,False]).all():
        raise ValueError('INVALID_FROZEN_FEATURE_INCLUSION_FLAGS')
    for rep in reps:
        rows = inventory.loc[inventory.representation_id.eq(rep)]
        if set(membership.loc[membership.representation_id.eq(rep),'patient_id']) != set(baseline.patient_id):
            raise ValueError('INCOMPLETE_FROZEN_MEMBERSHIP_POPULATION')
        members = hard_members(membership,rep)
        if not rows.n_hard_supported.eq(len(members)).all() or not rows.n_baseline_total.eq(len(baseline)).all():
            raise ValueError('FROZEN_PATIENT_DENOMINATORS_NOT_RECONCILED')
        stable = int(members.structural_stability_status.eq('stable').sum())
        if (not rows.n_stable_hard.eq(stable).all() or
            not rows.n_provisional_hard.eq(len(members)-stable).all() or
            not (rows.n_observed_unique+rows.n_missing).eq(len(members)).all() or
            rows.n_observed_unique.lt(0).any() or rows.n_missing.lt(0).any()):
            raise ValueError('FROZEN_SUPPORT_OR_STABILITY_NOT_RECONCILED')
    if root.name == REFERENCE_RUN:
        _reconcile_reference(inventory,qc)
    protected = {path:input_hashes[name] for name,path in paths.items()}
    protected[release] = release_sha
    for rel, sha in manifest['sha256_by_artifact'].items():
        protected[safe_artifact(release.parent,rel)] = sha
    return dict(tables=tables,roles=roles,manifest=manifest,release=release,qc=qc,
                frozen=frozen,input_hashes=input_hashes,protected=protected)


def _suppress(frame, minimum):
    """Suppress small aggregates and redundant totals that permit subtraction."""
    result = frame.copy()
    columns = [c for c in result if c.startswith('n_')]
    result['small_cells_suppressed'] = False
    for idx,row in result.iterrows():
        # Complementary suppression of every count on the row is conservative;
        # do not retain an exact denominator that reveals a suppressed cell.
        if any(pd.notna(row[c]) and 0 < float(row[c]) < minimum for c in columns):
            result.loc[idx,columns] = None
            result.loc[idx,'small_cells_suppressed'] = True
    # Alias JSON contains only schema keys and counts, never category strings.
    if 'alias_coverage_json' in result:
        result['alias_coverage_json'] = result.alias_coverage_json.map(lambda value: json.dumps(
            {k:(v if v == 0 or v >= minimum else None) for k,v in json.loads(value).items()},sort_keys=True))
    return result


def _privacy_check(tables, baseline):
    private = {'patient_id','clinical_episode_id','episode_id','clinical_anchor_date','measurement_date'}
    identifiers = [str(v) for v in baseline.patient_id if len(str(v)) >= 8]
    for frame in tables.values():
        if set(frame) & private: raise ValueError('PRIVATE_FIELD_IN_PHASEC_OUTPUT')
        payload = frame.to_csv(index=False)
        if any(identifier in payload for identifier in identifiers):
            raise ValueError('PRIVATE_VALUE_IN_PHASEC_OUTPUT')
        if re.search(r'\b(?:19|20)\d{2}-\d{2}-\d{2}\b',payload):
            raise ValueError('EXACT_DATE_IN_PHASEC_OUTPUT')


def run_phasec(graph03_run, graph02_release, output_dir, config_path, repository_root, *,
               strict=False, dependency_resolution=None):
    """Produce eleven restricted artifacts in a new directory, preserving inputs."""
    cfg_path, repo = Path(config_path).resolve(), Path(repository_root).resolve()
    config = yaml.safe_load(cfg_path.read_text())
    inputs = load_phasec_inputs(graph03_run,graph02_release,config,strict=strict)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid4().hex[:8]
    if output_dir is None:
        from .s03_run_resolution import graph_paths
        output_dir = graph_paths(repository_root)['phasec'] / 'runs' / run_id
    output = Path(output_dir).resolve()
    roots = [Path(graph03_run).resolve(),inputs['release'].parent]
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError('PHASEC_OUTPUT_OVERLAPS_FROZEN_INPUTS')
    if output.exists(): raise FileExistsError('PHASEC_OUTPUT_MUST_BE_NEW')
    frozen, inventory = inputs['frozen'], inputs['tables']['source_inventory']
    tables = build_phasec_tables(inventory, inputs['roles'], frozen['feature_sets'],repo,config)
    reps = inputs['manifest']['eligible_representation_ids']
    baseline, membership = frozen['baseline_raw'], frozen['membership']
    covariates, era, pop = [], [], []
    for rep in reps:
        def scoped_document(key):
            records = [r for r in config.get(key,[]) if r['representation_id'] == rep]
            record = records[0] if records else {}
            found = _reference(repo,record.get('source_reference',''),record.get('source_sha256',''))
            return record, dict(verified=bool(found),sha256=found.get('source_code_sha256',''))
        protocol_review, protocol_evidence = scoped_document('protocol_reviews')
        era_review, era_evidence = scoped_document('baseline_era_reviews')
        covariates.append(dict(**audit_phasec_covariates(baseline,protocol_evidence,{'review':protocol_review}),representation_id=rep))
        era.append(dict(**audit_phasec_baseline_date(baseline,era_review,era_evidence),representation_id=rep))
        masks = {'baseline':baseline.patient_id,
            'mapper_covered':membership.loc[membership.representation_id.eq(rep) & membership.mapper_covered,'patient_id'].drop_duplicates(),
            'hard_supported':hard_members(membership,rep).patient_id}
        for universe, ids in masks.items():
            frame = baseline.loc[baseline.patient_id.isin(ids)]
            if 'pop__status' in frame:
                row = audit_phasec_pop_dictionary(frame['pop__status'])
            else:
                row = dict(source_column='pop__status',decision_status='PENDING_REVIEW',
                    evidence_status='SOURCE_UNAVAILABLE',reviewer_action='LOCATE_FROZEN_POP_SOURCE',
                    blocking_reasons='POP_SOURCE_UNAVAILABLE')
            pop.append(dict(**row,representation_id=rep,universe=universe,
                           step11_raw_value_comparison='SOURCE_UNAVAILABLE_CODE_CONTRACT_ONLY'))
    tables.update(covariate_audit=pd.DataFrame(covariates), baseline_era_audit=pd.DataFrame(era),
                  pop_vocabulary_audit=pd.DataFrame(pop))
    metadata = dict(run_id=run_id, input_graph03_run_id=Path(graph03_run).name,
        graph02_run_id=inputs['manifest']['run_id'],
        input_implementation_commit_sha=inputs['qc']['implementation_commit_sha'],
        input_config_sha256=inputs['qc'].get('config_sha256','NOT_RECORDED_IN_INPUT'),
        graph02_release_sha256=inputs['qc']['graph02_release_sha256'],
        implementation_commit_sha=git_commit(), config_sha256=digest(cfg_path),
        schema_version=VERSION, privacy_class='INTERNAL_RESTRICTED',shareable_export_allowed=False)
    for name, table in tables.items():
        for column in COMMON:
            if column not in table: table[column] = ''
        for column,value in metadata.items(): table[column] = value
        tables[name] = _suppress(table,int(config['minimum_disclosure_cell']))
    _privacy_check(tables,baseline)
    candidates = tables['candidate_adjudication']
    unique = not candidates.duplicated(['representation_id','candidate_family','source_column']).any()
    no_approvals = not candidates.decision_status.isin(['APPROVED','APPROVED_ONLY_AFTER_HUMAN_SIGNOFF','N3_REVIEWED']).any()
    no_models = tables['model_readiness'].decision_status.eq('NOT_ESTIMABLE').all()
    if not (unique and no_approvals and no_models): raise ValueError('PHASEC_ADJUDICATION_QC_FAILED')
    # Check again immediately before the first write, including provenance code.
    code_hashes = {p:digest(repo/p) for p in DOCUMENTARY_FILES if (repo/p).is_file()}
    if any(digest(path) != sha for path,sha in inputs['protected'].items()):
        raise ValueError('FROZEN_INPUT_CHANGED_DURING_PHASEC')
    qc = dict(**metadata, status='PASS', phasec_status='COMPLETE_WITH_DOCUMENTARY_BLOCKERS',
        dependency_resolution=dependency_resolution or [], validation_mode='strict' if strict else 'normal',
        source_keys_unique=unique, graph02_unchanged=True, graph03_unchanged=True,
        n_candidate_rows=len(candidates), n_candidate_families=candidates.candidate_family.nunique(),
        n_models_complete=0,n_automatic_approvals=0, n_model_readiness_rows=len(tables['model_readiness']),
        n_inventory_rows=len(inventory), n_inventory_sources=inventory.source_column.nunique(),
        input_sha256_by_artifact=inputs['input_hashes'],documentary_source_sha256=code_hashes,
        missing_documentary_sources=[p for p in DOCUMENTARY_FILES if p not in code_hashes],
        frozen_sha256_by_artifact=inputs['manifest']['sha256_by_artifact'],
        reference_reconciliation='RECONCILED' if Path(graph03_run).name == REFERENCE_RUN else 'NOT_REFERENCE_RELEASE',
        no_patient_identifiers_or_exact_dates_exported=True, clinical_approval_granted=False,
        small_cell_policy='row_count_complementary_suppression',minimum_disclosure_cell=config['minimum_disclosure_cell'],
        empty_table_status={k:'NO_EVIDENCE' for k,v in tables.items() if v.empty},
        no_refit=True, no_source_recoding=True, no_inference=True, longitudinal_policies_unchanged=True)
    # All validation succeeds before making the output directory. Inputs are never
    # opened for writing; exclusive directory creation prevents overwriting runs.
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    for name, table in tables.items():
        table.to_csv(output/f'{PREFIX}{name}.csv',index=False)
        (output/f'{PREFIX}{name}.csv').chmod(0o600)
    write_json(output/f'{PREFIX}qc.json',qc)
    summary = [
        '# Graph03 Phase C — documentary endpoint audit', '',
        f'Run: {run_id}. Input Graph03: {metadata["input_graph03_run_id"]}.',
        f'Frozen Graph02 SHA-256: {metadata["graph02_release_sha256"]}.',
        f'Implementation commit: {metadata["implementation_commit_sha"]}.', '',
        'Audit QC: PASS. Clinical decisions: PENDING HUMAN REVIEW. Models: NOT_ESTIMABLE.',
        'No approvals, model estimates, recoding, source changes or discovery refit were produced.', '',
        '| Representation | Family | Source | Decision | Observed / hard supported |',
        '|---|---|---|---|---|',
    ]
    for r in candidates.itertuples():
        n = 'SUPPRESSED' if r.small_cells_suppressed else f'{r.n_observed_unique} / {r.n_hard_supported}'
        summary.append(f'| {r.representation_id} | {r.candidate_family} | {r.source_column} | {r.decision_status} | {n} |')
    summary += ['',
        'Documentary blockers: actual assay/form lineage, instrument dates, baseline availability, units/categories, '
        'conceptual independence and research-protocol equivalence require separate expert decisions. Code hashes '
        'establish provenance only. Step11 raw values and original laboratory forms were not available in this audit.',
        'Support blockers: observed Y is distinct from complete-case support; missing protocol/era cannot be '
        'filled with constants or removed. A/B/C/D must use one common patient population with adequate events, '
        'community support, degrees of freedom and a reviewed multiplicity plan.',
        'Topology blockers: any provisional hard assignments retain the existing stability gate. R2 remains '
        'exploratory; a stable-only R0 sensitivity requires a separate prespecified amendment.',
        'Model contracts match _pop_status, _source_basis and n3_incremental_models: A contains clinical '
        'covariates and reviewed eligible Pop; B adds frozen community; C adds reviewed reduced N1 to A; '
        'D adds community to C only when requested. ESSDAI/ESSPRI and derivatives cannot use Pop. '
        'Prespecification and human reviews must precede any future contrasts B−A, C−A, D−C.',
        'Required signoffs before editing existing configuration: characterization.independence_reviews; '
        'n3_validation.measurement_policies; baseline_era_reviews; pop_comparator_reviews; source_comparators. '
        'Phase C does not edit these contracts or temporal_resolution.endpoint_policies.',
        'Pop Unclassifiable/Unclassified is diagnosed without recoding. Baseline-era review does not verify '
        'endpoint dates or enable longitudinal analyses.',
        'Mapper communities describe exploratory multidimensional SjD heterogeneity. Independent incremental '
        'clinical utility has not been demonstrated. No external validation, prognosis or causal claim is supported.',
        'INTERNAL_RESTRICTED. No automatic sharing or packaging. Even suppressed aggregates require '
        'institutional disclosure review; consult the protocol for the post-review packaging command.', '']
    (output/f'{PREFIX}summary.md').write_text('\n'.join(summary))
    artifacts = []
    for path in sorted(output.iterdir()):
        artifacts.append({**metadata,**{k:'' for k in COMMON if k not in metadata},
            'decision_status':'INTERNAL_RESTRICTED','evidence_status':'AUDIT_ARTIFACT',
            'reviewer_action':'INSTITUTIONAL_DISCLOSURE_REVIEW_BEFORE_SHARING',
            'blocking_reasons':'NOT_AUTOMATICALLY_SHAREABLE','relative_path':path.name,
            'sha256':digest(path),'size_bytes':path.stat().st_size})
    # A hash inventory cannot contain its own hash. It covers the other ten files.
    pd.DataFrame(artifacts).to_csv(output/f'{PREFIX}artifact_inventory.csv',index=False)
    for path in output.iterdir():
        path.chmod(0o600)
    return dict(status='PASS',phasec_status='COMPLETE_WITH_DOCUMENTARY_BLOCKERS',
                run_id=run_id,output_dir=str(output),n_models_complete=0)

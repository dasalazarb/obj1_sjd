"""N3 artifact IO boundary: commit Phase A before attempting Phase B."""
import json
from pathlib import Path
import re

import pandas as pd

from .._pipeline import digest, git_commit, save_table, write_json
from .s03_n3_candidate_audit import VERSION, INTERNAL, n3_audit_tables
from .s03_n3_incremental import METRICS, blocked_results, n3_incremental_models


def evidence_catalog(config, repository_root):
    """Verify repository references without opening arbitrary paths or remote URLs.

    File existence establishes provenance, not scientific correctness; the human
    review is independently mandatory. Symbol references must exist in that file.
    """
    references = set()
    def visit(value):
        if isinstance(value, dict):
            reference = value.get('source_reference')
            if isinstance(reference, str): references.add(reference)
            for nested in value.values(): visit(nested)
        elif isinstance(value, list):
            for nested in value: visit(nested)
    visit(config)
    root = Path(repository_root).resolve()
    catalog = {}
    for reference in sorted(references):
        if not re.fullmatch(r'(?:src|docs|tests)/[A-Za-z0-9_./-]+(?::[A-Za-z_][A-Za-z0-9_]*)?', reference):
            continue
        relative, _, symbol = reference.partition(':')
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.suffix not in {'.py', '.md', '.yaml', '.yml', '.json', '.csv'}:
            continue
        if path.stat().st_size > 2_000_000: continue
        if symbol and symbol not in path.read_text(): continue
        catalog[reference] = dict(verified=True, sha256=digest(path))
    return catalog


def _privacy_check(tables, baseline):
    # Long identifiers cannot occur incidentally as numeric aggregate counts.
    identifiers = [str(x) for x in baseline.patient_id.dropna() if len(str(x)) >= 8]
    private_fields = {'patient_id', 'clinical_episode_id', 'episode_id', 'clinical_anchor_date', 'measurement_date'}
    for table in tables.values():
        if set(table) & private_fields:
            raise ValueError('PRIVATE_IDENTIFIER_FIELD_IN_N3_OUTPUT')
        payload = table.to_csv(index=False)
        if any(identifier in payload for identifier in identifiers):
            raise ValueError('PRIVATE_IDENTIFIER_VALUE_IN_N3_OUTPUT')
        # Clinical dates cannot be copied through free review windows or notes.
        if re.search(r'\b(?:19|20)\d{2}-\d{2}-\d{2}\b', payload):
            raise ValueError('EXACT_CLINICAL_DATE_IN_N3_OUTPUT')


def n3_validation_qc(tables, baseline, membership, manifest, provenance, frozen_unchanged, classical_matrix=None):
    inventory = tables['source_inventory']
    models = tables['incremental_models']
    counts = inventory.groupby('representation_id').size().to_dict()
    unique = not inventory.duplicated(['representation_id', 'source_column']).any()
    count_ok = (len(counts) == len(manifest['eligible_representation_ids']) and len(set(counts.values())) == 1)
    denominators = bool((inventory.n_observed_unique + inventory.n_missing).eq(inventory.n_hard).all())
    blocked = models.loc[models.status.ne('COMPLETE')]
    no_unapproved_estimates = bool(blocked[METRICS].isna().all().all() and blocked.effects_json.fillna('').eq('').all())
    complete = models.loc[models.status.eq('COMPLETE')]
    approved_keys = set(map(tuple, inventory.loc[inventory.review_state.eq('N3_REVIEWED'), ['representation_id', 'source_column']].to_numpy()))
    inference_ok = set(map(tuple, complete[['representation_id', 'source_column']].to_numpy())) <= approved_keys
    common = bool(complete.empty or (complete.common_population_verified.all()
                  and complete.groupby(['representation_id', 'source_column']).n_patients.nunique().eq(1).all()))
    _privacy_check(tables, baseline)
    if not all([unique, count_ok, denominators, no_unapproved_estimates, inference_ok, common, frozen_unchanged]):
        raise ValueError('N3_VALIDATION_QC_FAILED')
    reconciliations = []
    if classical_matrix is not None and not classical_matrix.empty:
        # Categories repeat the source denominator. Never sum these repeated rows.
        keys = ['representation_id', 'source_column', 'community_id']
        spread = classical_matrix.groupby(keys).n_observed.nunique()
        if spread.gt(1).any(): raise ValueError('CLASSICAL_CATEGORY_DENOMINATORS_INCONSISTENT')
        classical = classical_matrix.drop_duplicates(keys).set_index(keys).n_observed.to_dict()
        for role in inventory.to_dict('records'):
            for cell in json.loads(role['coverage_by_community']):
                key = (role['representation_id'], role['source_column'], str(cell['community_id']))
                if key in classical and cell['n_observed'] != classical[key]:
                    reconciliations.append(dict(representation_id=key[0], source_column=key[1],
                        community_id=key[2], status='REQUIRES_REVIEW', reason='SOURCE_CONVENTION_OR_VALUE_TYPE_RECONCILIATION_REQUIRED'))
    return {**provenance, **INTERNAL, 'status':'PASS' if not reconciliations else 'COMPLETE_WITH_LIMITATIONS',
        'frozen_input_sha256_by_role':{role:manifest.get('sha256_by_artifact',{}).get(relative)
            for role,relative in manifest.get('relative_paths',{}).items() if role in {'membership','feature_sets','baseline_raw'}},
        'source_counts_by_representation':counts, 'n_inventory_rows':len(inventory), 'n_baseline':len(baseline),
        'source_keys_unique':unique, 'source_population_contract_reconciled':count_ok,
        'unique_patient_denominators_reconciled':denominators, 'common_model_population_verified':common,
        'no_unapproved_estimates':no_unapproved_estimates, 'inference_requires_review':inference_ok,
        'graph02_unchanged':frozen_unchanged, 'no_refit':True, 'no_outcome_selection':True,
        'longitudinal_policies_unchanged':True, 'no_patient_identifiers_or_exact_dates_exported':True,
        'classical_coverage_reconciliation_status':'REQUIRES_REVIEW' if reconciliations else 'RECONCILED' if classical_matrix is not None else 'PENDING_CLASSICAL_CHARACTERIZATION',
        'classical_coverage_reconciliation_issues':reconciliations,
        'n_candidates_for_review':int(inventory.is_outcome_candidate_for_review.sum()),
        'n_reviewed_independent_sources':int(inventory.review_state.eq('N3_REVIEWED').sum()),
        'n_models_complete':int(models.status.eq('COMPLETE').sum()),
        'n_outcomes_modeled':len(complete[['representation_id','source_column']].drop_duplicates()),
        'n_not_estimable_models':int(models.status.eq('NOT_ESTIMABLE').sum()),
        'n_comparisons_complete':len(tables['model_comparison']),
        'interpretation':'development_sample_conditional_association_not_external_predictive_utility'}


def enforce_n3_review_gates(roles, inventory):
    """Keep tiers/estimators, but protect every legacy consumer from invalid evidence."""
    result = roles.copy()
    reviewed = inventory.set_index(['representation_id', 'source_column'])
    for index, role in result.loc[result.tier.eq('N3_REVIEWED')].iterrows():
        audit = reviewed.loc[(role.representation_id, role.source_column)]
        if audit.review_state != 'N3_REVIEWED':
            reasons = list(filter(None, str(role.blocking_reasons).split('|')))
            reasons.extend(filter(None, audit.blocking_reasons.split('|')))
            result.at[index, 'inferential_eligible'] = False
            result.at[index, 'blocking_reasons'] = '|'.join(sorted(set(reasons))) or 'PENDING_CLINICAL_REVIEW'
    return result


def _save_tables(run_dir, tables, provenance):
    for name, table in tables.items():
        frame = table.copy()
        for key, value in provenance.items(): frame[key] = value
        save_table(run_dir / f'03_graph_n3_{name}.csv', frame)


def run_n3_validation(run_dir, frames, roles, manifest, config, provenance, config_sha256):
    """Persist complete audit even if a later model implementation fails."""
    run_dir = Path(run_dir)
    catalog = evidence_catalog(config, Path(__file__).resolve().parents[4])
    metadata = {**provenance, 'graph03_run_id':run_dir.name,
                'implementation_commit_sha':git_commit(), 'config_sha256':config_sha256}
    baseline, membership = frames['baseline_raw'], frames['membership']
    audit = n3_audit_tables(baseline, membership, roles, frames['feature_sets'],
                          frames.get('clinical_registry', pd.DataFrame()), manifest, config, catalog)
    _privacy_check(audit, baseline)
    _save_tables(run_dir, audit, metadata)
    phase_b_error = ''
    try:
        results = n3_incremental_models(baseline, membership, roles, audit['source_inventory'], manifest, config, catalog)
    except Exception as exc:
        # Record only exception type: exception messages may contain sensitive values.
        phase_b_error = type(exc).__name__
        results = blocked_results(manifest, config, 'PHASE_B_FAILURE_AUDIT_RETAINED')
    tables = {**audit, **results}
    qc = n3_validation_qc(tables, baseline, membership, manifest, metadata, True)
    qc.update(phase_a_status='COMPLETE', phase_b_status='FAILED_WITH_AUDIT_RETAINED' if phase_b_error else
              'COMPLETE' if qc['n_models_complete'] else 'NOT_ESTIMABLE', phase_b_exception_type=phase_b_error,
              evidence_sha256_by_reference={r: e['sha256'] for r, e in catalog.items()})
    _save_tables(run_dir, results, metadata)
    write_json(run_dir / '03_graph_n3_validation_qc.json', qc)
    (run_dir / '03_graph_n3_results_summary.md').write_text(
        '# N3 clinical audit and conditional exploratory comparisons\n\n'
        f'Execution: COMPLETE_WITH_LIMITATIONS. Phase A: COMPLETE. Phase B: {qc["phase_b_status"]}.\n\n'
        f'Frozen Graph02: {metadata["graph02_run_id"]}; release SHA-256: {metadata["graph02_release_sha256"]}. '
        f'Graph03 run: {run_dir.name}. Contract: {VERSION}.\n\n'
        f'Inventory rows: {len(audit["source_inventory"])}. Candidates requiring review: {qc["n_candidates_for_review"]}. '
        f'Reviewed independent sources: {qc["n_reviewed_independent_sources"]}. '
        f'Completed models: {qc["n_models_complete"]}; comparisons: {qc["n_comparisons_complete"]}.\n\n'
        'NONE_APPROVED / NO_REVIEWED_INDEPENDENT_ANCHORS applies when no documentary review passes. '
        'NOT_ESTIMABLE is insufficient evidence or support, never evidence of no association.\n\n'
        'Next documentary steps: representation-specific lineage, clinical/proxy and discovery exclusion review; '
        'source provenance, verified basal measurement window, units/category conventions, and separate era-date review. '
        'Prespecify outcomes and any reduced N1 comparator before modeling. Pop cannot compare outcomes that define it.\n\n'
        'All estimates are exploratory and in-sample, conditional on frozen memberships. Effect and paired patient '
        'bootstrap intervals omit Mapper selection uncertainty. A positive change in adjusted R2 is not external '
        'prediction or new information beyond X. R2 retains provisional interpretation. '
        'Longitudinal date policies and Graph02 are unchanged.\n\n'
        'INTERNAL_RESTRICTED. Community aggregates may include small cells. No automatic public appendix, '
        'figure or ZIP export; use the existing reviewed suppression route for shareable panels.\n')
    return tables, qc

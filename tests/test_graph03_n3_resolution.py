"""Phase D regression boundary: synthetic data only, no clinical approvals."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from src.studies.graph._pipeline import digest
from src.studies.graph.stage03 import s03_n3_resolution as D
from src.studies.graph.stage03.s03_n3_phasec_adjudication import run_phasec
from src.studies.graph.stage03.s03_n3_resolution_support import support_diagnostics
from test_graph03_n3_phasec_adjudication import synthetic_bundle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'src/studies/graph/configs/n3_phase_d_resolution.yaml'
PYTHON = sys.executable


def bundle(tmp_path, *, all_reps=False, missing_source=False):
    graph03, release, phasec_cfg = synthetic_bundle(tmp_path)
    if missing_source:
        for path in graph03.glob('*.csv'):
            table = pd.read_csv(path)
            if 'source_column' in table:
                table.loc[table.source_column.ne('ext__biopsy_focus_score')].to_csv(path, index=False)
        path = graph03 / '03_graph_results_summary.json'
        summary = json.loads(path.read_text())
        summary['sha256_by_artifact'] = {p.name: digest(p) for p in graph03.iterdir() if p != path}
        path.write_text(json.dumps(summary))
    if all_reps:
        for path in [release.parent / 'membership.csv', release.parent / 'feature_sets.csv',
                     *graph03.glob('*.csv')]:
            table = pd.read_csv(path)
            if 'representation_id' not in table:
                continue
            tables = [table]
            for rep in ['R1', 'R3', 'R4']:
                clone = table.loc[table.representation_id.eq('R0')].copy()
                clone['representation_id'] = rep
                tables.append(clone)
            pd.concat(tables).to_csv(path, index=False)
        manifest = json.loads(release.read_text())
        manifest['eligible_representation_ids'] = ['R0', 'R1', 'R2', 'R3', 'R4']
        manifest['sha256_by_artifact'] = {p: digest(release.parent / p) for p in manifest['sha256_by_artifact']}
        manifest['features_sha256'] = digest(release.parent / 'feature_sets.csv')
        release.write_text(json.dumps(manifest))
        for path in graph03.glob('*.json'):
            document = json.loads(path.read_text())
            document['graph02_release_sha256'] = digest(release)
            path.write_text(json.dumps(document))
        summary_path = graph03 / '03_graph_results_summary.json'
        summary = json.loads(summary_path.read_text())
        summary['sha256_by_artifact'] = {p.name: digest(p) for p in graph03.iterdir() if p != summary_path}
        summary_path.write_text(json.dumps(summary))
        cfg = yaml.safe_load(phasec_cfg.read_text())
        cfg['input_contract']['graph02_release_sha256'] = digest(release)
        phasec_cfg.write_text(yaml.safe_dump(cfg))
    phasec = tmp_path / 'explicit_phasec_directory_not_the_run_id'
    run_phasec(graph03, release, phasec, phasec_cfg, ROOT)
    qc = json.loads((phasec / '03_graph_n3_phasec_qc.json').read_text())
    config = yaml.safe_load(CONFIG.read_text())
    config['input_contract'] = dict(graph03_run_id=graph03.name,
        graph02_run_id=release.parent.name, graph02_release_sha256=digest(release),
        phasec_run_id=qc['run_id'], graph03_implementation_commit_sha='e'*40,
        phasec_implementation_commit_sha=qc['implementation_commit_sha'],
        n3_schema_version='graph03_n3_audit_v1', graph03_summary_sha256=digest(graph03 / '03_graph_results_summary.json'),
        phasec_qc_sha256=digest(phasec / '03_graph_n3_phasec_qc.json'),
        phasec_inventory_sha256=digest(phasec / '03_graph_n3_phasec_artifact_inventory.csv'))
    cfg = tmp_path / 'phased.yaml'
    cfg.write_text(yaml.safe_dump(config))
    return graph03, release, phasec, cfg


def run(paths, output, root=ROOT):
    return D.run_resolution(*paths[:3], output, paths[3], root, strict=True)


def hashes(paths):
    return {p: digest(p) for root in [paths[0], paths[1].parent, paths[2]] for p in root.rglob('*') if p.is_file()}


def edit_cfg(paths, change):
    cfg = yaml.safe_load(paths[3].read_text())
    change(cfg)
    paths[3].write_text(yaml.safe_dump(cfg))


def repin_phasec(paths):
    """Independent fixture repinning simulates a signed new evidence bundle."""
    directory = paths[2]
    inv_path = directory / '03_graph_n3_phasec_artifact_inventory.csv'
    inv = pd.read_csv(inv_path)
    inv['sha256'] = [digest(directory / name) for name in inv.relative_path]
    inv.to_csv(inv_path, index=False)
    edit_cfg(paths, lambda cfg: cfg['input_contract'].update(
        phasec_qc_sha256=digest(directory / '03_graph_n3_phasec_qc.json'),
        phasec_inventory_sha256=digest(inv_path)))


def signed_review(tmp_path, section, **extra):
    root = tmp_path / 'review_repository'
    docs = root / 'docs'
    docs.mkdir(parents=True, exist_ok=True)
    payload = dict(review_section=section, representation_id='R0', status='DOCUMENTED',
        review_id='SYNTHETIC_ONLY', reviewer='SYNTHETIC_REVIEWER', human_signoff=True, **extra)
    path = docs / (section + '.yaml')
    path.write_text(yaml.safe_dump(payload))
    record = {k: v for k, v in payload.items() if k != 'review_section'}
    record.update(source_reference='docs/' + path.name, source_sha256=digest(path))
    config = yaml.safe_load(CONFIG.read_text())
    config['proposed_reviews'][section] = [record]
    return root, config, record, path


def test_complete_run_all_five_reps_40_candidates_160_preparation_rows(tmp_path):
    paths = bundle(tmp_path, all_reps=True)
    before = hashes(paths)
    contracts = {p: digest(ROOT / p) for p in ['src/studies/graph/configs/validation_03.yaml', 'src/studies/graph/configs/n3_phasec_review.yaml']}
    output = tmp_path / 'phased'
    result = run(paths, output)
    assert result['status'] == 'PASS' and result['phased_status'] == 'COMPLETE_WITH_BLOCKERS'
    assert before == hashes(paths)
    assert all(digest(ROOT / p) == sha for p, sha in contracts.items())
    candidates = pd.read_csv(output / (D.PREFIX + 'candidate_evidence_matrix.csv'))
    models = pd.read_csv(output / (D.PREFIX + 'model_preflight.csv'))
    assert len(candidates) == 40 and len(models) == 160
    assert not candidates.duplicated(['representation_id', 'source_column']).any()
    assert models.status.eq('NOT_ESTIMABLE').all()
    assert models.historical_phasec_status.eq('NOT_ESTIMABLE').all()
    assert models.decision_status.eq('ESTIMATION_BLOCKED').all()
    assert not {'p_value', 'q_value', 'effect', 'ci_low', 'ci_high', 'prediction'} & set(models)
    assert set(models.model) == set('ABCD')
    assert candidates.clinical_independence_evidence.eq('PENDING_REVIEW').all()
    inventory = pd.read_csv(output / (D.PREFIX + 'artifact_inventory.csv'))
    assert all(digest(output / r.relative_path) == r.sha256 for r in inventory.itertuples())
    assert len(inventory) == len(list(output.iterdir())) - 1
    for path in output.iterdir():
        payload = path.read_text()
        assert 'SYNTHETIC_0' not in payload and '2020-01-01' not in payload
        if path.suffix == '.csv':
            table = pd.read_csv(path)
            assert not {'patient_id', 'clinical_episode_id', 'measurement_date'} & set(table)
            assert {'run_id', 'schema_version', 'graph02_release_sha256', 'graph03_summary_sha256',
                    'phasec_qc_sha256', 'no_inference', 'no_refit', 'implementation_commit_sha'} <= set(table)
            assert table[[c for c in table if c.startswith('n_')]].isna().all().all()
            assert table.privacy_class.eq('INTERNAL_RESTRICTED').all()
    assert output.stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        run(paths, output)


@pytest.mark.parametrize('key', ['graph02_release_sha256', 'graph03_summary_sha256', 'phasec_qc_sha256', 'phasec_inventory_sha256'])
def test_independent_hash_anchor_required_and_checked(tmp_path, key):
    paths = bundle(tmp_path)
    edit_cfg(paths, lambda cfg: cfg['input_contract'].update({key: '0'*64}))
    before = hashes(paths)
    with pytest.raises(ValueError, match='SHA'):
        run(paths, tmp_path / 'invalid')
    assert before == hashes(paths) and not (tmp_path / 'invalid').exists()


@pytest.mark.parametrize('key', ['graph02_run_id', 'graph03_run_id', 'phasec_run_id', 'graph03_implementation_commit_sha', 'phasec_implementation_commit_sha'])
def test_wrong_run_or_commit_pin(tmp_path, key):
    paths = bundle(tmp_path)
    edit_cfg(paths, lambda cfg: cfg['input_contract'].update({key: '0'*40 if 'commit' in key else 'WRONG'}))
    before = hashes(paths)
    with pytest.raises(ValueError):
        run(paths, tmp_path / 'invalid')
    assert before == hashes(paths)


@pytest.mark.parametrize('field,value', [('status', 'FAIL'), ('phasec_status', 'COMPLETE'),
    ('input_graph03_run_id', 'WRONG'), ('input_sha256_by_artifact', {}),
    ('clinical_approval_granted', True), ('no_inference', False), ('n_models_complete', 1),
    ('frozen_sha256_by_artifact', {})])
def test_qc_and_incongruent_phasec_evidence_rejected_even_when_repinned(tmp_path, field, value):
    paths = bundle(tmp_path)
    target = paths[2] / '03_graph_n3_phasec_qc.json'
    qc = json.loads(target.read_text())
    qc[field] = value
    target.write_text(json.dumps(qc))
    repin_phasec(paths)
    with pytest.raises(ValueError):
        run(paths, tmp_path / 'invalid')


def test_phase_a_incomplete_and_corrupt_phasec_file(tmp_path):
    paths = bundle(tmp_path)
    target = paths[0] / '03_graph_n3_validation_qc.json'
    qc = json.loads(target.read_text())
    qc['phase_a_status'] = 'PENDING'
    target.write_text(json.dumps(qc))
    with pytest.raises(ValueError):
        run(paths, tmp_path / 'invalid')
    # Distinct unpinned corruption in Phase C cannot pass its inventory.
    target = paths[2] / '03_graph_n3_phasec_candidate_adjudication.csv'
    target.write_text(target.read_text() + '\n')
    cfg = yaml.safe_load(paths[3].read_text())
    with pytest.raises(ValueError):
        D.load_inputs(paths[0], paths[1], paths[2], cfg)


@pytest.mark.parametrize('location', ['graph02', 'graph03', 'phasec', 'ancestor'])
def test_output_cannot_overlap_any_frozen_tree(tmp_path, location):
    paths = bundle(tmp_path)
    target = {'graph02': paths[1].parent / 'child', 'graph03': paths[0] / 'child',
              'phasec': paths[2] / 'child', 'ancestor': tmp_path}[location]
    before = hashes(paths)
    with pytest.raises(ValueError, match='OVERLAPS'):
        run(paths, target)
    assert before == hashes(paths)


def test_symlink_output_overlap_and_frozen_symlinks_rejected(tmp_path):
    paths = bundle(tmp_path)
    link = tmp_path / 'alias'
    link.symlink_to(paths[2], target_is_directory=True)
    with pytest.raises(ValueError, match='OVERLAPS'):
        run(paths, link / 'output')
    (paths[2] / 'escaped').symlink_to(paths[1])
    with pytest.raises(ValueError, match='SYMLINK'):
        run(paths, tmp_path / 'output')


def test_frozen_checks_execute_on_exception_and_failed_output_removed(tmp_path, monkeypatch):
    paths = bundle(tmp_path)
    before = hashes(paths)
    def fail(*args):
        raise RuntimeError('synthetic failure')
    monkeypatch.setattr(D, 'build_tables', fail)
    with pytest.raises(RuntimeError):
        run(paths, tmp_path / 'output')
    assert before == hashes(paths)
    def concurrent_change(*args):
        # Simulate an external writer; the workflow itself must detect it.
        (paths[2] / 'external_change').write_text('synthetic')
        raise RuntimeError('synthetic failure')
    monkeypatch.setattr(D, 'build_tables', concurrent_change)
    with pytest.raises(ValueError, match='PROTECTED_INPUT_CHANGED'):
        run(paths, tmp_path / 'output')


def test_protocol_alias_null_and_signed_mapping_are_proposals_only(tmp_path):
    baseline = pd.DataFrame({'ids__protocol': ['A', 'B']*10})
    before = baseline.copy(deep=True)
    review = dict(record={}, verified=False, document_sha256='')
    rows = D.protocol_inventory(baseline, review, 'R0')
    assert rows[0]['decision_status'] == 'ALIAS_PRESENT_UNVERIFIED'
    assert rows[0]['canonical_field_status'] == 'ABSENT'
    baseline['demo__protocol'] = pd.NA
    assert D.protocol_inventory(baseline, review, 'R0')[0]['canonical_field_status'] == 'ALL_NULL'
    baseline = before.copy(deep=True)
    root, cfg, record, document = signed_review(tmp_path, 'protocol',
        original_source='ids__protocol', canonical_target='demo__protocol', cohort_scope='R0',
        equivalent_to_prespecified_confounder=True, code_dictionary_review='DOCUMENTED',
        multiple_protocol_rule_review='DOCUMENTED', meaning='research_protocol')
    verified = D.review_record(cfg, 'protocol', 'R0', '', root)
    rows = D.protocol_inventory(baseline, verified, 'R0')
    assert rows[0]['decision_status'] == 'MAPPING_HUMAN_VERIFIED'
    assert rows[0]['proposed_derived_dataset'] and not rows[0]['mapping_applied']
    pd.testing.assert_frame_equal(baseline, before)
    document.write_text(document.read_text() + '\n')
    assert not D.review_record(cfg, 'protocol', 'R0', '', root)['verified']


@pytest.mark.parametrize('state', ['PENDING', 'PENDING_REVIEW', 'DOCUMENTED', 'REJECTED'])
def test_partial_decisions_require_matching_signed_content(tmp_path, state):
    root, cfg, record, document = signed_review(tmp_path, 'candidate_independence', source_column='esspri__total')
    record['status'] = state
    payload = {k: v for k, v in record.items() if k not in {'source_reference', 'source_sha256'}}
    document.write_text(yaml.safe_dump(dict(review_section='candidate_independence', **payload)))
    record['source_sha256'] = digest(document)
    result = D.review_record(cfg, 'candidate_independence', 'R0', 'esspri__total', root)
    assert result['verified'] == (state == 'DOCUMENTED')
    assert result['rejected'] == (state == 'REJECTED')
    record['reviewer'] = 'DIFFERENT_REVIEWER'
    assert not D.review_record(cfg, 'candidate_independence', 'R0', 'esspri__total', root)['verified']


def test_invalid_signature_no_auto_approval_or_unscoped_review(tmp_path):
    paths = bundle(tmp_path)
    cfg = yaml.safe_load(paths[3].read_text())
    for key in D.POLICY:
        changed = deepcopy(cfg)
        changed['policy'][key] = not D.POLICY[key]
        with pytest.raises(ValueError, match='UNSAFE'):
            D.validate_config(changed)
    cfg['proposed_reviews']['measurement'] = [dict(status='APPROVED', source_column='esspri__total', representation_id='R0')]
    with pytest.raises(ValueError, match='CANNOT_GRANT'):
        D.validate_config(cfg)
    cfg['proposed_reviews']['measurement'] = [dict(status='PENDING', source_column='esspri__total')]
    with pytest.raises(ValueError, match='SCOPE'):
        D.validate_config(cfg)


def test_anchor_not_instrument_date_and_future_always_blocks(tmp_path):
    root, cfg, _, _ = signed_review(tmp_path, 'measurement', source_column='esspri__total',
        date_semantics='episode_anchor', derived_from_future=False)
    review = D.review_record(cfg, 'measurement', 'R0', 'esspri__total', root)
    date = D.date_readiness(review)
    assert not date['instrument_measurement_date_verified'] and date['temporal_eligibility'] == 'PENDING_REVIEW'
    record = review['record']
    record.update(date_semantics='instrument_measurement', instrument_measurement_date_verified=True,
        timezone_review='DOCUMENTED', baseline_temporal_order_review='DOCUMENTED', baseline_definition='official',
        lookback_rule='source_specific', measurement_source_reference='original_form', repeated_measurement_rule='prespecified')
    assert D.date_readiness(review)['temporal_eligibility'] == 'READY_FOR_HUMAN_REVIEW'
    record['derived_from_future'] = True
    assert D.date_readiness(review)['temporal_eligibility'] == 'BLOCKED_FUTURE_DERIVED'
    record['derived_from_future'] = False
    record.pop('timezone_review')
    assert D.date_readiness(review)['temporal_eligibility'] == 'PENDING_REVIEW'
    assert date['longitudinal_status'].startswith('BLOCKED')


def test_pop_dictionary_unclassifiable_and_assays_not_merged(tmp_path):
    paths = bundle(tmp_path)
    before = hashes(paths)
    output = tmp_path / 'output'
    run(paths, output)
    cross = pd.read_csv(output / (D.PREFIX + 'pop_crosswalk_proposal.csv'))
    assert {'Pop1', 'Pop2', 'Pop3', 'Unclassifiable', 'Unclassified', 'MISSING', 'INVALID'} == set(cross.original_category)
    assert cross.original_category.eq(cross.proposed_category).all()
    assert not cross.mapping_applied.any() and cross.descriptive_universe_retained.all()
    candidates = pd.read_csv(output / (D.PREFIX + 'candidate_evidence_matrix.csv'))
    assert len(candidates.loc[candidates.candidate_family.eq('ana')]) == 4
    assert len(candidates.loc[candidates.candidate_family.eq('rheumatoid_factor')]) == 4
    circular = pd.read_csv(output / (D.PREFIX + 'circularity_dependency_report.csv'))
    assert not circular.alternative_assay_merge.any()
    assert not circular.independence_approved.any()
    assert before == hashes(paths)


def test_pop_definitional_derivative_guard_matches_existing_defense():
    # Import existing _pop_status only inside this parity test; Phase D never
    # imports its estimators.
    from src.studies.graph.stage03.s03_n3_incremental import _pop_status
    validation = yaml.safe_load((ROOT / 'src/studies/graph/configs/validation_03.yaml').read_text())
    validation['characterization']['relationships'].append(dict(source_column='custom_derivative', target_source='esspri__total', relationship_type='derived_from'))
    for source in ['essdai__total', 'esspri__total', 'esspri__dryness', 'eg_renal', 'ovl__status', 'custom_derivative']:
        assert D.pop_dependency(source, validation)
        assert _pop_status(source, validation, 'R0', {})[0] == 'POP_BASELINE_COMPARATOR_INELIGIBLE'


def test_real_dependency_paths_distinct_from_conceptual_independence():
    validation = yaml.safe_load((ROOT / 'src/studies/graph/configs/validation_03.yaml').read_text())
    features = pd.DataFrame([dict(source_column='esspri__dryness', included=True)])
    status, related, paths = D.dependency_evidence('esspri__total', {'esspri__dryness'}, features, pd.DataFrame(), validation, ROOT)
    assert status == 'COMPOSITE_COMPONENT' and related and paths
    status, _, _ = D.dependency_evidence('lab__rheumatoid_factor__value', set(), features, pd.DataFrame(), validation, ROOT)
    assert status == 'NO_DOCUMENTED_DEPENDENCE'


def diagnostics(*, binary=False, no_covariates=False, collinear=False, small_community=False, separate=False):
    n = 100
    rng = np.random.default_rng(22)
    x = pd.DataFrame({'age': rng.normal(size=n), 'sex': ['a', 'b']*50})
    if no_covariates:
        x['protocol'] = np.nan
    if collinear:
        x['duplicate_age'] = x.age
    community = pd.Series(['0']*50 + ['1']*50)
    if small_community:
        community = pd.Series(['0']*99 + ['1'])
    y = pd.Series(rng.normal(size=n))
    if binary:
        y = pd.Series([0, 1]*50 if separate else rng.integers(0, 2, n), dtype=float)
    validation = dict(minimum_model_patients=30, minimum_group_patients=8, minimum_patients_per_community=8)
    return support_diagnostics(y, x, pd.DataFrame(index=x.index), community,
        'binary' if binary else 'numeric', validation, categories_verified=True, minimum_events_per_parameter=1)


def test_observed_y_distinct_from_common_covariate_complete_population():
    rows = diagnostics(no_covariates=True)
    assert rows['A']['n_Y_observed'] == 100 and rows['A']['n_common_complete_case'] == 0
    assert all(r['design_support_gate'] == 'BLOCKED' for r in rows.values())
    assert 'INSUFFICIENT_COVARIATE_COMPLETE_PATIENTS' in rows['A']['support_blocking_reasons']


@pytest.mark.parametrize('kwargs,reason', [({'collinear': True}, 'RANK_DEFICIENT_DESIGN'),
    ({'small_community': True}, 'LOW_COMMUNITY_SUPPORT'), ({'binary': True, 'separate': True}, 'SEPARATION')])
def test_design_support_rank_small_community_and_separation_block(kwargs, reason):
    rows = diagnostics(**kwargs)
    assert reason in rows['B']['support_blocking_reasons']
    assert rows['B']['design_support_gate'] == 'BLOCKED'


def test_numerically_ready_is_review_only_never_estimation():
    rows = diagnostics()
    assert rows['A']['design_support_gate'] == rows['B']['design_support_gate'] == 'PREFLIGHT_READY_FOR_REVIEW'
    assert rows['C']['design_support_gate'] == rows['D']['design_support_gate'] == 'BLOCKED'


def test_disclosure_suppresses_complementary_cells_across_all_tables():
    raw = {'one': pd.DataFrame({'n_total': [100], 'n_small': [3]}),
           'another': pd.DataFrame({'n_large': [97], 'n_other': [0]})}
    result = D._disclosure_boundary(raw)
    assert all(frame.filter(regex='^n_').isna().all().all() for frame in result.values())
    assert raw['one'].n_small.iloc[0] == 3


def test_code_drift_reports_and_blocks_or_rejects(tmp_path):
    paths = bundle(tmp_path)
    target = paths[2] / '03_graph_n3_phasec_qc.json'
    qc = json.loads(target.read_text())
    qc['documentary_source_sha256']['src/studies/_shared.py'] = '0'*64
    target.write_text(json.dumps(qc))
    repin_phasec(paths)
    run(paths, tmp_path / 'reported')
    output_qc = json.loads((tmp_path / 'reported' / (D.PREFIX + 'qc.json')).read_text())
    assert output_qc['code_drift_gate'] == 'BLOCKED_CODE_DRIFT'
    edit_cfg(paths, lambda cfg: cfg['policy'].update(code_drift='reject'))
    with pytest.raises(ValueError, match='DRIFT_REJECTED'):
        run(paths, tmp_path / 'rejected')


def test_no_fit_refit_projection_or_pipeline_imports_and_cli_explains_missing_dependency():
    script = ROOT / 'src/studies/graph/03d_graph_n3_resolution_preflight.py'
    result = subprocess.run([PYTHON, str(script), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and '--phasec-run' in result.stdout and '--preflight-only' in result.stdout
    result = subprocess.run([PYTHON, str(script)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'No se encontró un run compatible' in result.stderr
    assert 'Traceback' not in result.stderr
    for path in [script, Path(D.__file__), ROOT / 'src/studies/graph/stage03/s03_n3_resolution_support.py']:
        code = path.read_text()
        for forbidden in ['.fit(', '.predict(', 'import statsmodels', 'import sklearn', 's03_pipeline import', 's02_mapper_engine import']:
            assert forbidden not in code


def test_default_config_requires_no_operator_hashes():
    cfg = yaml.safe_load(CONFIG.read_text())
    D.validate_config(cfg)
    assert cfg['input_contract'] == {'n3_schema_version': 'graph03_n3_audit_v1'}


def test_phasec_corruption_reaches_independent_inventory_boundary(tmp_path):
    paths = bundle(tmp_path)
    target = paths[2] / '03_graph_n3_phasec_source_lineage.csv'
    target.write_text(target.read_text() + '\n')
    with pytest.raises(ValueError, match='PHASEC_ARTIFACT_SHA'):
        run(paths, tmp_path / 'invalid')


def test_incomplete_phase_a_rejected_with_valid_summary_hash(tmp_path):
    paths = bundle(tmp_path)
    target = paths[0] / '03_graph_n3_validation_qc.json'
    document = json.loads(target.read_text())
    document['phase_a_status'] = 'PENDING'
    target.write_text(json.dumps(document))
    summary_path = paths[0] / '03_graph_results_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['sha256_by_artifact'][target.name] = digest(target)
    summary_path.write_text(json.dumps(summary))
    edit_cfg(paths, lambda cfg: cfg['input_contract'].update(graph03_summary_sha256=digest(summary_path)))
    with pytest.raises(ValueError, match='PHASE_A_SCHEMA_OR_COMPLETION'):
        run(paths, tmp_path / 'invalid')


@pytest.mark.parametrize('bad', ['duplicate_key', 'wrong_metadata', 'estimable'])
def test_phasec_table_reconciliation_rejects_repinned_bad_evidence(tmp_path, bad):
    paths = bundle(tmp_path)
    target = paths[2] / '03_graph_n3_phasec_model_readiness.csv'
    table = pd.read_csv(target)
    if bad == 'duplicate_key':
        table = pd.concat([table, table.iloc[:1]], ignore_index=True)
    elif bad == 'wrong_metadata':
        table.loc[0, 'input_graph03_run_id'] = 'OTHER'
    else:
        table.loc[0, 'decision_status'] = 'COMPLETE'
    table.to_csv(target, index=False)
    repin_phasec(paths)
    with pytest.raises(ValueError, match='PHASEC_TABLE_PROVENANCE|PREPARATION_ROWS'):
        run(paths, tmp_path / 'invalid')


def test_review_traversal_and_internal_symlink_never_authorize(tmp_path):
    root, cfg, record, document = signed_review(tmp_path, 'measurement', source_column='esspri__total')
    record['source_reference'] = 'docs/../../outside.yaml'
    assert not D.review_record(cfg, 'measurement', 'R0', 'esspri__total', root)['verified']
    alias = document.parent / 'alias.yaml'
    alias.symlink_to(document)
    record['source_reference'] = 'docs/alias.yaml'
    assert not D.review_record(cfg, 'measurement', 'R0', 'esspri__total', root)['verified']


def test_signed_pop_crosswalk_uses_derived_copy_and_r2_stays_exploratory(tmp_path, monkeypatch):
    paths = bundle(tmp_path)
    cfg = yaml.safe_load(paths[3].read_text())
    inputs = D.load_inputs(*paths[:3], cfg)
    validation = yaml.safe_load((ROOT / 'src/studies/graph/configs/validation_03.yaml').read_text())
    original = inputs['frozen']['baseline_raw'].copy(deep=True)
    original_review = D.review_record
    def records(config, section, rep, source, root):
        if section == 'pop_semantics':
            return dict(verified=True, rejected=False, document_sha256='a'*64, human_review_status='DOCUMENTED',
                        record=dict(crosswalk={'Unclassifiable': 'Unclassified'}, equivalence_review='DOCUMENTED'))
        return original_review(config, section, rep, source, root)
    monkeypatch.setattr(D, 'review_record', records)
    tables, reconciled = D.build_tables(inputs, cfg, ROOT, validation)
    cross = tables['pop_crosswalk_proposal']
    assert cross.loc[cross.original_category.eq('Unclassifiable'), 'proposed_category'].eq('Unclassified').all()
    assert cross.mapping_applied.all() and reconciled
    for _, rows in cross.groupby(['representation_id', 'universe']):
        assert rows.n_category.sum() == rows.n_proposed_category.sum() == rows.n_universe.iloc[0]
    pd.testing.assert_frame_equal(original, inputs['frozen']['baseline_raw'])
    topo = tables['topology_stability_audit']
    assert topo.loc[topo.representation_id.eq('R2'), 'decision_status'].eq('descriptive_only').all()
    assert topo.loc[topo.representation_id.eq('R0'), 'r0_stable_only_executed'].all()
    assert not topo.loc[topo.representation_id.eq('R2'), 'r0_stable_only_executed'].any()


def test_unknown_categories_do_not_construct_design_and_all_models_share_union():
    n = 100
    x = pd.DataFrame({'age': np.arange(n, dtype=float)})
    basis = pd.DataFrame({'reviewed_n1': np.r_[np.arange(70, dtype=float), [np.nan]*30]})
    y = pd.Series(np.arange(n) % 3, dtype=float)
    community = pd.Series(['a', 'b']*50)
    v = dict(minimum_model_patients=30, minimum_group_patients=8, minimum_patients_per_community=8)
    rows = support_diagnostics(y, x, basis, community, 'numeric', v, categories_verified=False,
                               basis_requested=True, include_d=True)
    assert {r['n_common_complete_case'] for r in rows.values()} == {70}
    assert all(r['n_parameters'] is None and r['design_support_gate'] == 'BLOCKED' for r in rows.values())


def test_endpoint_future_derivation_blocks_despite_documented_other_flags(tmp_path, monkeypatch):
    paths = bundle(tmp_path)
    cfg = yaml.safe_load(paths[3].read_text())
    inputs = D.load_inputs(*paths[:3], cfg)
    validation = yaml.safe_load((ROOT / 'src/studies/graph/configs/validation_03.yaml').read_text())
    def records(config, section, rep, source, root):
        return dict(verified=True, rejected=False, document_sha256='a'*64, human_review_status='DOCUMENTED',
            record=dict(derived_from_future=True, instrument_measurement_date_verified=True,
                        date_semantics='instrument_measurement', unit_review='DOCUMENTED', category_review='DOCUMENTED',
                        clinical_review='DOCUMENTED', origin_review='DOCUMENTED', proxy_review='DOCUMENTED',
                        discovery_exclusion_review='DOCUMENTED'))
    monkeypatch.setattr(D, 'review_record', records)
    tables, _ = D.build_tables(inputs, cfg, ROOT, validation)
    assert tables['candidate_evidence_matrix'].temporal_eligibility.eq('BLOCKED_FUTURE_DERIVED').all()
    assert tables['model_preflight'].blocking_reasons.str.contains('FUTURE_DERIVED_ENDPOINT').all()
    assert tables['model_preflight'].status.eq('NOT_ESTIMABLE').all()


def test_exact_date_and_identifier_privacy_boundaries():
    baseline = pd.DataFrame({'patient_id': ['PRIVATE_IDENTIFIER_123']})
    for payload in ['PRIVATE_IDENTIFIER_123', '2020-01-01', '2020-01-01T12:00:00']:
        with pytest.raises(ValueError, match='IN_PHASED_OUTPUT'):
            D._privacy_check(payload, baseline)


def test_missing_inventory_candidate_has_explicit_qc_explanation(tmp_path):
    paths = bundle(tmp_path, missing_source=True)
    output = tmp_path / 'output'
    run(paths, output)
    qc = json.loads((output / (D.PREFIX + 'qc.json')).read_text())
    assert qc['missing_candidate_keys'] == [
        dict(representation_id='R0', source_column='ext__biopsy_focus_score'),
        dict(representation_id='R2', source_column='ext__biopsy_focus_score')]
    candidates = pd.read_csv(output / (D.PREFIX + 'candidate_evidence_matrix.csv'))
    missing = candidates.loc[candidates.source_column.eq('ext__biopsy_focus_score')]
    assert len(missing) == 2
    assert missing.blocking_reasons.str.contains('SOURCE_ABSENT_FROM_PHASE_A_INVENTORY').all()


def test_partial_output_removed_when_publication_fails(tmp_path, monkeypatch):
    paths = bundle(tmp_path)
    before = hashes(paths)
    output = tmp_path / 'output'
    original = Path.write_text
    def fail(path, *args, **kwargs):
        if path.name == D.PREFIX + 'model_preflight.csv':
            raise OSError('synthetic publication failure')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', fail)
    with pytest.raises(OSError, match='publication failure'):
        run(paths, output)
    assert before == hashes(paths) and not output.exists()


def test_low_residual_df_and_unknown_event_rule_block():
    n = 35
    rng = np.random.default_rng(11)
    y = pd.Series(rng.normal(size=n))
    x = pd.DataFrame(rng.normal(size=(n, 26)))
    community = pd.Series(['a']*17 + ['b']*18)
    v = dict(minimum_model_patients=30, minimum_group_patients=8, minimum_patients_per_community=8)
    rows = support_diagnostics(y, x, pd.DataFrame(index=x.index), community, 'numeric', v, categories_verified=True)
    assert 'INSUFFICIENT_RESIDUAL_DF' in rows['B']['support_blocking_reasons']
    y = pd.Series([0, 1]*17 + [0], dtype=float)
    x = pd.DataFrame({'age': rng.normal(size=n)})
    rows = support_diagnostics(y, x, pd.DataFrame(index=x.index), community, 'binary', v,
                               categories_verified=True, minimum_events_per_parameter=float('nan'))
    assert 'EVENT_SUPPORT_RULE_REQUIRES_REVIEW' in rows['B']['support_blocking_reasons']


def test_binary_nonseparated_support_is_review_only():
    rows = diagnostics(binary=True)
    assert rows['B']['separation_status'] == 'NO_SEPARATION_DETECTED'
    assert rows['B']['design_support_gate'] == 'PREFLIGHT_READY_FOR_REVIEW'

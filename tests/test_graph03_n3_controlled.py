"""Controlled N3 contracts, using invented patients and attestations ONLY."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from graph03_schema_inventory import schema_inventory
from src.studies.graph._pipeline import digest
from src.studies.graph.stage03 import s03_n3_controlled as E
from src.studies.graph.stage03.s03_n3_approved_view import protocol_view, measurement_values
from src.studies.graph.stage03.s03_n3_decisions import (
    VERSION, APPROVED, SECTIONS, verify_decisions, decision_templates, plan_sha256,
)
from tests.test_graph03_n3_candidate_audit import synthetic_n3, audit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'ext__biopsy_focus_score'
PYTHON = sys.executable


def outcome(rep='R0', **updates):
    return dict(representation_id=rep, source_column=SOURCE, requested_models=['A','B'],
                n1_basis=[], use_pop=False, topology_scope='full', model_family='gaussian_ols',
                hypothesis_family='SYNTHETIC_ONLY', **updates)


def reviews(spec):
    """Synthetic decisions must never be copied into a production plan."""
    return dict(
        protocol=dict(equivalent_to_research_protocol=True, canonical_target='demo__protocol',
            assignment_scope='patient_research_protocol', multiple_protocol_rule='reject',
            code_dictionary_verified=True, assignment_date_verified=True, dates_normalized_to_utc=True,
            source_column='spine__source_protocol', assignment_date_column='assigned_on',
            code_dictionary={'protocol_a':'protocol_a', 'protocol_b':'protocol_b'}),
        baseline=dict(official_baseline_verified=True, episode_anchor_verified=True, era_date_semantics_verified=True),
        measurement=dict(instrument_date_verified=True, date_semantics='instrument_measurement',
            selection_rule='latest_prebaseline', duplicate_rule='identical_only', baseline_window_verified=True,
            source_semantics_verified=True, missingness_and_indication_reviewed=True, lookback_days=30,
            timezone_reviewed=True, dates_normalized_to_utc=True,
            type='numeric', assay_id='SYNTHETIC_ASSAY', unit='SYNTHETIC_UNIT'),
        independence=dict(clinically_independent=True, source_lineage_verified=True, proxy_reviewed=True,
            discovery_exclusion_verified=True, same_assay_dependencies_reviewed=True),
        pop=dict(outcome_dependency_reviewed=True, semantics_verified=True,
            crosswalk={x:x for x in ['Pop1','Pop2','Pop3','Unclassifiable','Unclassified']}),
        topology=dict(full_structure_stable=True, selected_before_outcomes=True, unchanged_thresholds=True,
            exclusion_bias_reviewed=True, sensitivity_design_approved=True),
        design=dict(requested_models=spec['requested_models'], n1_basis=spec['n1_basis'],
            hypothesis_family=spec['hypothesis_family'], model_family=spec['model_family'],
            multiplicity_prespecified=True, common_complete_case=True, covariate_semantics_verified=True,
            n1_basis_source_semantics_verified=True, reduced_n1_basis_verified=True))


def fixture(rep='R0'):
    f = list(synthetic_n3(representations=(rep,)))
    f[1].rename(columns={'independent':SOURCE}, inplace=True)
    f[1]['clinical_episode_id'] = [f'SYNTHETIC_EP_{i}' for i in range(len(f[1]))]
    f[1]['clinical_baseline_date'] = f[1].clinical_anchor_date
    f[1]['clinical_baseline_episode_id'] = f[1].clinical_episode_id
    baseline = f[1]
    context = baseline[['patient_id','clinical_episode_id','demo__protocol']].rename(
        columns={'demo__protocol':'spine__source_protocol'})
    context['assigned_on'] = '2019-12-01'
    measurements = baseline[['patient_id','clinical_episode_id',SOURCE]].rename(columns={SOURCE:'value'})
    measurements = measurements.assign(source_column=SOURCE, measurement_date='2019-12-31',
        assay_id='SYNTHETIC_ASSAY', unit='SYNTHETIC_UNIT', sample_id='SYNTHETIC_SAMPLE')
    roles, tables = audit(f)
    spec = outcome(rep)
    decisions = {(section, rep, SOURCE):dict(details=details) for section, details in reviews(spec).items()}
    return f, context, measurements, roles, tables['source_inventory'], spec, decisions


def prepare(f, context, measurements, roles, inventory, spec, decisions):
    return E.prepare_pair(f[1], context, measurements, f[2], roles, inventory,
                          f[4], f[5], f[3], f[0], spec, decisions)


def test_protocol_approved_view_preserves_frozen_population_and_columns():
    f, context, _, _, _, spec, _ = fixture()
    before = f[1].copy(deep=True)
    viewed = protocol_view(f[1], context, reviews(spec)['protocol'])
    pd.testing.assert_frame_equal(viewed, before)
    pd.testing.assert_frame_equal(f[1], before)
    with pytest.raises(ValueError, match='REVIEW_REQUIRED'):
        protocol_view(f[1], context, {})


@pytest.mark.parametrize('failure,reason', [
    ('absent','SOURCE_OR_DATE_UNAVAILABLE'), ('incompatible','SEMANTIC_REVIEW'),
    ('dictionary','OUTSIDE_REVIEWED_DICTIONARY'), ('multi','MULTIPLE_PROTOCOLS'),
    ('duplicate','NOT_ONE_TO_ONE'), ('future','AFTER_BASELINE'),
    ('mismatch','DISAGREES_WITH_FROZEN'), ('unmatched','EPISODE_UNMATCHED')])
def test_protocol_failures_never_fill_defaults(failure, reason):
    f, context, _, _, _, spec, _ = fixture()
    review = reviews(spec)['protocol']
    if failure == 'absent': context.drop(columns='spine__source_protocol', inplace=True)
    if failure == 'incompatible': review['assignment_scope'] = 'visit_origin'
    if failure == 'dictionary': context.loc[0,'spine__source_protocol'] = 'unknown'
    if failure == 'multi': context.loc[0,'spine__source_protocol'] = 'protocol_a|protocol_b'
    if failure == 'duplicate': context = pd.concat([context, context.iloc[:1]])
    if failure == 'future': context.loc[0,'assigned_on'] = '2021-01-01'
    if failure == 'mismatch': f[1].loc[0,'demo__protocol'] = 'other'
    if failure == 'unmatched': context = context.iloc[1:]
    before = f[1].copy(deep=True)
    with pytest.raises(ValueError, match=reason):
        protocol_view(f[1], context, review)
    pd.testing.assert_frame_equal(f[1], before)


def test_future_measurements_are_never_basal_and_ties_require_adjudication():
    f, _, measurements, _, _, spec, _ = fixture()
    review = reviews(spec)['measurement']
    future = measurements.copy()
    future['measurement_date'] = '2021-01-01'
    with pytest.raises(ValueError, match='NO_PREBASELINE'):
        measurement_values(f[1], future, SOURCE, review)
    mixed = pd.concat([future, measurements])
    y, _ = measurement_values(f[1], mixed, SOURCE, review)
    pd.testing.assert_series_equal(y, f[1][SOURCE], check_names=False)
    duplicate = measurements.iloc[:1].copy()
    duplicate['value'] += 1
    with pytest.raises(ValueError, match='TIE_REQUIRES'):
        measurement_values(f[1], pd.concat([measurements,duplicate]), SOURCE, review)


def test_ana_assays_are_not_interchangeable_and_not_tested_is_not_negative():
    f, _, measurements, _, _, spec, _ = fixture()
    review = reviews(spec)['measurement']
    measurements['source_column'] = 'lab__ana_status__text'
    measurements.loc[0,'assay_id'] = 'OTHER_ASSAY'
    with pytest.raises(ValueError, match='NON_EQUIVALENT_ASSAY'):
        measurement_values(f[1], measurements, 'lab__ana_status__text', review)
    measurements['assay_id'] = 'SYNTHETIC_ASSAY'
    measurements['value'] = ['positive','negative','unknown','not_tested']*25
    review.update(type='binary', category_convention=dict(positive=['positive'], negative=['negative'],
                  unknown=['unknown'], not_tested=['not_tested']))
    y, kind = measurement_values(f[1], measurements, 'lab__ana_status__text', review)
    assert kind == 'binary' and y.notna().sum() == 50 and y.eq(0).sum() == 25


def test_multigating_and_unrequested_models_remain_independent():
    args = fixture()
    args[-1].clear()
    rows, _ = prepare(*args)
    a, b, c, d = rows
    assert a['protocol_gate'] == a['measurement_gate'] == a['independence_gate'] == E.FAIL
    assert a['support_gate'] == E.UNKNOWN and b['topology_gate'] == E.FAIL
    assert c['status'] == d['status'] == 'NOT_REQUESTED'
    assert a['topology_gate'] == 'NOT_APPLICABLE'


def test_ready_models_use_existing_estimators_and_common_patients():
    args = fixture()
    f, _, measurements, _, _, spec, _ = args
    measurements.loc[:3,'value'] = np.nan
    rows, prepared = prepare(*args)
    assert [r['status'] for r in rows] == ['READY','READY','NOT_REQUESTED','NOT_REQUESTED']
    models, comparisons = E.estimate_pair(rows, prepared, spec, f[3], f[0], execute=True)
    assert [r['status'] for r in models[:2]] == ['COMPLETE','COMPLETE']
    assert comparisons[0]['status'] == 'COMPLETE'
    assert comparisons[0]['delta_adjusted_r2'] > 0
    assert len(prepared['y']) == len(prepared['patient_ids']) == 96
    assert all(d.index.equals(prepared['y'].index) for d in prepared['designs'].values())


def test_a_c_do_not_require_topology_and_c_d_share_common_n1_population():
    args = list(fixture())
    f, _, measurements, _, _, spec, decisions = args
    spec.update(requested_models=['A','B','C','D'], n1_basis=['n1'])
    decisions[('design','R0',SOURCE)]['details'] = reviews(spec)['design']
    f[1].loc[:4,'n1'] = np.nan
    measurements.loc[5:8,'value'] = np.nan
    rows, prepared = prepare(*args)
    assert all(r['status'] == 'READY' for r in rows)
    assert len(prepared['y']) == 91
    models, comparisons = E.estimate_pair(rows, prepared, spec, f[3], f[0], execute=True)
    assert all(r['status'] == 'COMPLETE' for r in models)
    assert {r['contrast'] for r in comparisons} == {'B_vs_A','C_vs_A','D_vs_C'}
    f[2]['structural_stability_status'] = 'provisional'
    rows, prepared = prepare(*args)
    assert [r['status'] for r in rows] == ['READY','NOT_ESTIMABLE','READY','NOT_ESTIMABLE']
    assert rows[2]['topology_gate'] == 'NOT_APPLICABLE'


def test_r0_stable_only_excludes_ties_uncovered_and_never_approves_full_r0():
    args = list(fixture())
    f, _, _, _, _, spec, decisions = args
    f[2].loc[0, 'membership_tie'] = True
    f[2].loc[1, 'mapper_covered'] = False
    f[2].loc[[0,1], 'hard_assignment'] = False
    f[2].loc[2:9, 'structural_stability_status'] = 'provisional'
    rows, _ = prepare(*args)
    assert rows[1]['status'] == 'NOT_ESTIMABLE'
    spec['topology_scope'] = 'R0_stable_only'
    rows, prepared = prepare(*args)
    assert rows[1]['status'] == 'READY'
    assert len(prepared['y']) == 90
    assert rows[1]['topology_scope'] == 'R0_stable_only'
    decisions[('topology','R0',SOURCE)]['details']['selected_before_outcomes'] = False
    assert prepare(*args)[0][1]['status'] == 'NOT_ESTIMABLE'


def test_r2_keeps_exploratory_status_but_base_model_can_run():
    args = fixture('R2')
    rows, _ = prepare(*args)
    assert rows[0]['status'] == 'READY' and rows[1]['status'] == 'NOT_ESTIMABLE'
    assert rows[1]['topology_reason'] == 'R2_EXPLORATORY'


@pytest.mark.parametrize('failure,reason', [('singular','RANK_DEFICIENT'),('few','INSUFFICIENT_MODEL_PATIENTS'),
                                         ('groups','LOW_COMMUNITY_SUPPORT')])
def test_numerical_nonestimability_requires_evidence(failure, reason):
    args = fixture()
    f, _, measurements, _, _, _, _ = args
    if failure == 'singular': f[1]['demo__age_at_baseline'] = 1.
    if failure == 'few': measurements.loc[15:,'value'] = np.nan
    if failure == 'groups': f[2]['community_id'] = [0]*95 + [1]*5
    rows, _ = prepare(*args)
    assert rows[1]['support_gate'] == E.FAIL
    assert reason in rows[1]['support_reason']


def test_pop_needs_explicit_crosswalk_and_cannot_validate_essdai():
    args = list(fixture())
    _, _, _, _, _, spec, decisions = args
    spec['use_pop'] = True
    decisions.pop(('pop','R0',SOURCE))
    assert prepare(*args)[0][0]['pop_gate'] == E.FAIL
    decisions[('pop','R0',SOURCE)] = dict(details=reviews(spec)['pop'])
    assert prepare(*args)[0][0]['pop_gate'] == E.PASS
    decisions[('pop','R0',SOURCE)]['details']['crosswalk']['Unclassifiable'] = 'Pop1'
    assert prepare(*args)[0][0]['pop_gate'] == E.FAIL
    # The existing construct guard is applied even with a signed Pop decision.
    spec['source_column'] = 'essdai__total'
    for section, details in reviews(spec).items():
        decisions[(section,'R0','essdai__total')] = dict(details=details)
    assert prepare(*args)[0][0]['pop_gate'] == E.FAIL


def test_rf_duplicate_hypotheses_rejected_and_thresholds_never_weakened():
    args = fixture()
    validation = args[0][0]
    plan = dict(schema_version=VERSION, privacy_class='INTERNAL_RESTRICTED', public_export=False,
        require_human_signoff=True, allow_discovery_refit=False, authorized_reviewers=[], outcomes=[args[-2]])
    E.validate_plan(plan, validation)
    plan['outcomes'] = [{**args[-2], 'source_column':s} for s in
        ['lab__rheumatoid_factor__text','lab__rheumatoid_factor__value']]
    with pytest.raises(ValueError, match='RF_SAME_ASSAY'):
        E.validate_plan(plan, validation)
    plan['outcomes'] = [args[-2]]
    validation['minimum_model_patients'] = 29
    with pytest.raises(ValueError, match='THRESHOLD_WEAKENED'):
        E.validate_plan(plan, validation)


def test_bh_keeps_blocked_planned_hypotheses_and_negative_results():
    comparisons = [dict(hypothesis_family='test',p_value=p) for p in [.02, .8, np.nan]]
    E.correct_multiplicity(comparisons, .05)
    assert comparisons[0]['q_value'] == pytest.approx(.06)
    assert not comparisons[0]['reject_fdr'] and comparisons[1]['q_value'] > .05
    assert np.isnan(comparisons[2]['q_value']) and comparisons[2]['family_size'] == 3


def test_binary_event_rules_and_separation_include_base_models():
    args = fixture()
    f, _, measurements, _, _, spec, decisions = args
    spec.update(model_family='binomial_logit', minimum_events_per_parameter=3)
    decisions[('design','R0',SOURCE)]['details'] = reviews(spec)['design']
    policy = decisions[('measurement','R0',SOURCE)]['details']
    policy.update(type='binary', category_convention=dict(positive=['pos'],negative=['neg'],unknown=[],not_tested=[]))
    rng = np.random.default_rng(851)
    measurements['value'] = rng.choice(['pos','neg'],len(measurements))
    rows, prepared = prepare(*args)
    assert rows[0]['status'] == rows[1]['status'] == 'READY'
    models, comparisons = E.estimate_pair(rows, prepared, spec, f[3], f[0], execute=True)
    assert models[0]['status'] == models[1]['status'] == comparisons[0]['status'] == 'COMPLETE'
    spec.pop('minimum_events_per_parameter')
    assert 'BINARY_EVENT_RULE_UNAPPROVED' in prepare(*args)[0][0]['support_reason']
    spec['minimum_events_per_parameter'] = 3
    measurements['value'] = np.where(f[1]['demo__sex'].eq('F'),'pos','neg')
    rows, _ = prepare(*args)
    assert rows[0]['status'] == rows[1]['status'] == 'NOT_ESTIMABLE'
    assert 'SEPARATION' in rows[0]['support_reason'] and 'SEPARATION' in rows[1]['support_reason']


def attestations(tmp_path, plan, spec):
    evidence = tmp_path / 'source_evidence.md'
    evidence.write_text('SYNTHETIC_ONLY_SOURCE_DOCUMENTATION')
    for section, details in reviews(spec).items():
        payload = dict(schema_version=VERSION, section=section, representation_id=spec['representation_id'],
            source_column=spec['source_column'], plan_sha256=plan_sha256(plan), input_sha256=plan['input_sha256'],
            decision=APPROVED, reviewer='SYNTHETIC_ONLY_REVIEWER', review_id='SYNTHETIC_ONLY',
            human_signoff=True, institutional_signature_verified=True, details=details,
            evidence=[dict(path=evidence.name, sha256=digest(evidence))])
        path = tmp_path / (section + '.yaml')
        path.write_text(yaml.safe_dump(payload))
        plan['decisions'].append(dict(section=section, representation_id=spec['representation_id'],
            source_column=spec['source_column'], path=path.name, sha256=digest(path)))


def test_templates_unsigned_and_evidence_tamper_scope_and_authorization_fail_closed(tmp_path):
    plan = dict(schema_version=VERSION, input_sha256={'frozen':'a'*64}, outcomes=[outcome()],
                authorized_reviewers=['SYNTHETIC_ONLY_REVIEWER'], decisions=[])
    templates = decision_templates(plan, plan['input_sha256'], ['R0','R2'], [SOURCE])
    assert len(templates) == 2*len(SECTIONS)
    assert all(r['decision'] == 'PENDING_REVIEW' and r['human_signoff'] is False for r in templates)
    attestations(tmp_path, plan, outcome())
    verified, states, _ = verify_decisions(plan, tmp_path, plan['input_sha256'])
    assert len(verified) == len(SECTIONS)
    plan['authorized_reviewers'] = []
    assert not verify_decisions(plan, tmp_path, plan['input_sha256'])[0]
    plan['authorized_reviewers'] = ['SYNTHETIC_ONLY_REVIEWER']
    plan['outcomes'][0]['topology_scope'] = 'R0_stable_only'
    assert not verify_decisions(plan, tmp_path, plan['input_sha256'])[0]
    plan['outcomes'][0]['topology_scope'] = 'full'
    (tmp_path / 'source_evidence.md').write_text('TAMPERED')
    assert not verify_decisions(plan, tmp_path, plan['input_sha256'])[0]


def test_schema_inventory_never_reads_patient_rows_or_exports_counts(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    directory = tmp_path / 'data/analytic/10_build_integrated_longitudinal_dataset'
    directory.mkdir(parents=True)
    master = directory / '10_integrated_longitudinal_clinical_episode.parquet'
    pd.DataFrame(dict(patient_id=['PRIVATE_SECRET_ID'], measurement_date=['2020-01-01'])).to_parquet(master)
    (directory / '10_variable_registry.csv').write_text(
        'variable,role,n_nonmissing,classification_reason\nspine__source_protocol,CONTEXT,1,PRIVATE_SECRET\n')
    (tmp_path / '02_graph_baseline_raw.csv').write_text('patient_id,clinical_episode_id\nPRIVATE_SECRET_ID,PRIVATE_EP\n')
    def forbidden(*a, **kw):
        raise AssertionError('Patient rows must never be read')
    monkeypatch.setattr(pq, 'read_table', forbidden)
    result = schema_inventory(tmp_path, tmp_path)
    payload = json.dumps(result)
    assert result['sources']['Step10_analytic_master']['file_exists']
    assert not result['sources']['Step10_context']['file_exists']
    assert 'PRIVATE_SECRET' not in payload and '2020-01-01' not in payload
    assert 'n_nonmissing' not in result['sources']['Step10_variable_registry']['variable_metadata'][0]


def release_fixture(tmp_path, *, approved=False):
    from tests.test_graph03_n3_phasec_adjudication import synthetic_bundle
    graph03, release, _ = synthetic_bundle(tmp_path)
    manifest = json.loads(release.read_text())
    base_path = release.parent / manifest['relative_paths']['baseline_raw']
    baseline = pd.read_csv(base_path)
    rng = np.random.default_rng(252)
    baseline['clinical_episode_id'] = [f'SYNTHETIC_EP_{i}' for i in range(100)]
    baseline['clinical_baseline_episode_id'] = baseline.clinical_episode_id
    baseline['clinical_baseline_date'] = baseline.clinical_anchor_date
    baseline['is_clinical_baseline'] = True
    baseline['demo__age_at_baseline'] = rng.uniform(25,75,100)
    baseline['demo__sex'] = rng.choice(['F','M'],100)
    baseline[SOURCE] = rng.normal(size=100)
    baseline.loc[60:,SOURCE] = np.nan
    baseline.to_csv(base_path,index=False)
    membership_path = release.parent / manifest['relative_paths']['membership']
    member = pd.read_csv(membership_path)
    member['community_id'] = np.tile(np.repeat([0,1],50),2)
    # R0 stable-only: 25 per community, 50 total, with the original QC counts.
    member['structural_stability_status'] = np.tile(
        ['stable']*25+['provisional']*25+['stable']*25+['provisional']*25,2)
    # Preserve supported/stable partition expected by Phase A: 50 stable in 80 supported.
    member['supported'] = np.tile([True]*25+[True]*25+[True]*25+[True]*5+[False]*20,2)
    # Last 20 are already provisional, so 50 stable supported remain.
    member.to_csv(membership_path,index=False)
    manifest['sha256_by_artifact'] = {p:digest(release.parent/p) for p in manifest['sha256_by_artifact']}
    release.write_text(json.dumps(manifest))
    # Reconcile the synthetic Phase A denominator to 50 stable supported rows.
    for path in graph03.glob('*.json'):
        document = json.loads(path.read_text())
        document['graph02_release_sha256'] = digest(release)
        path.write_text(json.dumps(document))
    summary_path = graph03 / '03_graph_results_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['sha256_by_artifact'] = {p.name:digest(p) for p in graph03.iterdir() if p != summary_path}
    summary_path.write_text(json.dumps(summary))
    context = baseline[['patient_id','clinical_episode_id']].assign(
        spine__source_protocol=rng.choice(['protocol_a','protocol_b'],100), assigned_on='2019-12-01')
    context_path = tmp_path / 'context.parquet'
    context.to_parquet(context_path,index=False)
    measurements = baseline[['patient_id','clinical_episode_id']].assign(source_column=SOURCE,
        value=rng.normal(size=100) + np.repeat([0.,2.],50), measurement_date='2019-12-31',
        assay_id='SYNTHETIC_ASSAY', unit='SYNTHETIC_UNIT', sample_id='SYNTHETIC_SAMPLE')
    measurements_path = tmp_path / 'measurements.parquet'
    measurements.to_parquet(measurements_path,index=False)
    spec = outcome()
    spec['topology_scope'] = 'R0_stable_only'
    plan = yaml.safe_load((ROOT / 'src/studies/graph/configs/n3_controlled_inference.yaml').read_text())
    plan.update(outcomes=[spec], authorized_reviewers=['SYNTHETIC_ONLY_REVIEWER'],
                input_sha256=E.input_pins(graph03, release, context_path, measurements_path, ROOT))
    docs = tmp_path / 'decisions'
    docs.mkdir()
    if approved:
        attestations(docs, plan, spec)
    plan_path = tmp_path / 'plan.yaml'
    plan_path.write_text(yaml.safe_dump(plan))
    return graph03, release, context_path, measurements_path, plan_path, docs


def run_release(paths, output, execute=True):
    return E.run_controlled(*paths[:4], output, paths[4], paths[5], ROOT, execute=execute)


@pytest.mark.parametrize('approved', [False, True])
def test_immutable_release_full_boundary_and_no_disclosure(tmp_path, approved):
    paths = release_fixture(tmp_path, approved=approved)
    before = E._snapshot([paths[0], paths[1].parent])
    output = tmp_path / 'new_release'
    result = run_release(paths, output)
    assert E._snapshot([paths[0], paths[1].parent]) == before
    assert result['status'] == ('COMPLETE_WITH_ESTIMATES' if approved else 'COMPLETE_WITH_BLOCKERS')
    assert bool(list((output/'private').iterdir())) == approved
    for path in output.glob('*.csv'):
        frame = pd.read_csv(path)
        assert not {'patient_id','clinical_episode_id','measurement_date'} & set(frame)
        assert 'SYNTHETIC_EP_' not in path.read_text() and '2020-01-01' not in path.read_text()
        for c in frame:
            if c.startswith('n_') or c == 'residual_df':
                assert frame[c].isna().all()
    hashes = json.loads((output/'artifact_sha256.json').read_text())
    assert all(digest(output/name) == sha for name, sha in hashes.items())
    assert (output.stat().st_mode & 0o777) == 0o700
    assert all((p.stat().st_mode & 0o777) == 0o600 for p in output.rglob('*') if p.is_file())
    with pytest.raises(FileExistsError):
        run_release(paths, output)
    if approved:
        view = pd.read_parquet(next((output/'private').glob('approved_view*')))
        original = pd.read_csv(paths[1].parent/'baseline_raw.csv', dtype={'patient_id':'string'})
        pd.testing.assert_frame_equal(view[['patient_id','clinical_episode_id']], original[['patient_id','clinical_episode_id']])


def test_input_tampering_and_midrun_failure_do_not_publish(tmp_path, monkeypatch):
    paths = release_fixture(tmp_path, approved=True)
    target = paths[1].parent / 'membership.csv'
    original = target.read_text()
    target.write_text(original+'\n')
    with pytest.raises(ValueError):
        run_release(paths, tmp_path/'bad_sha')
    assert not (tmp_path/'bad_sha').exists()
    target.write_text(original)
    def fail(*a, **k):
        raise RuntimeError('PRIVATE_ID_AND_DATE')
    monkeypatch.setattr(E, 'estimate_pair', fail)
    with pytest.raises(RuntimeError):
        run_release(paths, tmp_path/'failed')
    assert not (tmp_path/'failed').exists()


def test_one_approved_pair_advances_without_approving_others(tmp_path):
    paths = release_fixture(tmp_path, approved=False)
    plan = yaml.safe_load(paths[4].read_text())
    plan['outcomes'].append(outcome('R2'))
    attestations(paths[5], plan, plan['outcomes'][0])
    paths[4].write_text(yaml.safe_dump(plan))
    result = run_release(paths, tmp_path/'partial')
    assert result['status'] == 'COMPLETE_WITH_ESTIMATES'
    models = pd.read_csv(tmp_path/'partial/models.csv')
    assert models.loc[models.representation_id.eq('R0') & models.model.isin(['A','B']), 'status'].eq('COMPLETE').all()
    assert models.loc[models.representation_id.eq('R2') & models.model.isin(['A','B']), 'status'].eq('NOT_ESTIMABLE').all()
    comparisons = pd.read_csv(tmp_path/'partial/comparisons.csv')
    assert comparisons.family_size.eq(2).all()
    assert list((tmp_path/'partial').glob('*.png'))


def test_frozen_input_change_during_failure_is_detected(tmp_path, monkeypatch):
    paths = release_fixture(tmp_path, approved=True)
    target = paths[0]/'03_graph_n3_results_summary.md'
    def mutate(*a, **kw):
        target.write_text('SYNTHETIC_MUTATION')
        raise RuntimeError('synthetic failure')
    monkeypatch.setattr(E, 'estimate_pair', mutate)
    with pytest.raises(ValueError, match='PROTECTED_INPUT_CHANGED'):
        run_release(paths, tmp_path/'must_not_exist')
    assert not (tmp_path/'must_not_exist').exists()


def test_cli_help_has_explicit_prepare_preflight_execute_and_no_discovery():
    path = ROOT / 'src/studies/graph/03e_graph_n3_controlled_inference.py'
    process = subprocess.run([PYTHON, str(path), '--help'], text=True, capture_output=True)
    assert process.returncode == 0
    assert '--prepare-plan' in process.stdout and '--execute' in process.stdout
    assert 'run_graph_study' not in path.read_text()


def test_cli_drafts_unsigned_plan_but_rejects_frozen_output_and_overwrites(tmp_path):
    paths = release_fixture(tmp_path)
    script = ROOT / 'src/studies/graph/03e_graph_n3_controlled_inference.py'
    command = [PYTHON, str(script), '--prepare-plan', '--graph03-run', str(paths[0]),
        '--graph02-release', str(paths[1]), '--context', str(paths[2]), '--measurements', str(paths[3])]
    plan_path, output = tmp_path/'draft.yaml', tmp_path/'templates'
    process = subprocess.run(command+['--plan',str(plan_path),'--output-dir',str(output)],
                             text=True,capture_output=True)
    assert process.returncode == 0, process.stderr
    plan = yaml.safe_load(plan_path.read_text())
    assert plan['outcomes'] == plan['decisions'] == plan['authorized_reviewers'] == []
    assert plan['input_sha256'] == E.input_pins(*paths[:4], ROOT)
    templates = yaml.safe_load((output/'unsigned_decision_templates.yaml').read_text())
    assert len(templates) == 40*len(SECTIONS) and all(t['human_signoff'] is False for t in templates)
    original = digest(plan_path)
    process = subprocess.run(command+['--plan',str(plan_path),'--output-dir',str(tmp_path/'other')],
                             text=True,capture_output=True)
    assert process.returncode != 0 and digest(plan_path) == original
    frozen_before = E._snapshot([paths[0], paths[1].parent])
    process = subprocess.run(command+['--plan',str(paths[0]/'bad.yaml'),'--output-dir',str(tmp_path/'bad_templates')],
                             text=True,capture_output=True)
    assert process.returncode != 0 and E._snapshot([paths[0], paths[1].parent]) == frozen_before

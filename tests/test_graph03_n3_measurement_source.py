"""Original measurement source contract using exclusively synthetic inputs."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from src.studies.graph._pipeline import digest
from src.studies.graph.stage03 import s03_n3_measurement_source as F
from src.studies.graph.stage03.s03_n3_approved_view import measurement_values
from src.studies.graph.stage03.s03_n3_measurement_review import AFFIRMATIONS, REVIEW_VERSION
from tests.test_graph03_n3_controlled import release_fixture, fixture, reviews

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'src/studies/graph/03f_build_n3_measurement_source.py'
INFERENCE = ROOT / 'src/studies/graph/03e_graph_n3_controlled_inference.py'
RF = 'lab__rheumatoid_factor__value'
RF_TEXT = 'lab__rheumatoid_factor__text'
ANA = 'lab__ana_status__text'
HEP2 = 'lab__ana_hep2_status__text'
BEFORE = '2019-12-27T14:00:00-05:00'
AFTER = '2020-01-02T14:00:00-05:00'


@pytest.fixture
def originals(tmp_path):
    graph03, release, context, _, _, decisions = release_fixture(tmp_path)
    baseline_path = release.parent / 'baseline_raw.csv'
    baseline = pd.read_csv(baseline_path, dtype={'patient_id': 'string'})
    step10 = baseline.assign(**{RF: 5.0, RF_TEXT: 'positive'})
    step10.to_parquet(tmp_path / 'step10.parquet', index=False)
    ctx = pd.read_parquet(context).assign(
        lab__rheumatoid_factor__measurement_date='2020-01-02',
        lab__rheumatoid_factor__unit='SYNTHETIC_SCALE')
    ctx.to_parquet(context, index=False)
    selected = baseline[F.KEYS].assign(
        lab_id='rheumatoid_factor', selected_lab_date='2020-01-02',
        selected_value_numeric=5., selected_value_text='positive', selected_unit='SYNTHETIC_SCALE',
        selection_status='selected_single', unit_conflict=False, result_conflict=False)
    selected.to_parquet(tmp_path / 'step01.parquet', index=False)
    lab = baseline[F.KEYS].rename(columns={'clinical_episode_id': 'matched_clinical_episode_id'})
    lab = lab.assign(canonical_analyte='rheumatoid_factor', semantic_mapping_status='mapped',
                     episode_match_ambiguous=False, result_valid_for_analysis=True,
                     lab_date=BEFORE, assay='SYNTHETIC_PLATFORM', unit='SYNTHETIC_SCALE',
                     result_numeric_exact=3., result_text='positive', result_raw='3',
                     result_status='final', result_operator='=')
    lab['specimen_id'] = ['SYNTHETIC_SAMPLE_' + str(idx) for idx in lab.index]
    lab['order_identifier'] = ['SYNTHETIC_ORDER_' + str(idx) for idx in lab.index]
    lab = pd.concat([lab, lab.iloc[:12].assign(lab_date=AFTER, result_numeric_exact=5.,
                   result_raw='5', specimen_id=lambda frame: frame.specimen_id + '_REPEAT')],
                   ignore_index=True)
    lab['_lab_record_id'] = lab.index
    lab['unrelated_sensitive_column'] = 'SYNTHETIC_DO_NOT_COPY'
    lab.to_parquet(tmp_path / 'eda.parquet', index=False)
    cfg = yaml.safe_load((ROOT / 'src/studies/graph/configs/n3_measurement_source.yaml').read_text())
    cfg['eda_specimen_id_column'] = 'specimen_id'
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    paths = dict(graph02_release=release, graph03_run=graph03, baseline=baseline_path,
                 eda_long=tmp_path / 'eda.parquet', step01=tmp_path / 'step01.parquet',
                 step10=tmp_path / 'step10.parquet', context=context)
    return paths, config_path, tmp_path / 'private', decisions


def paths_before(paths):
    return F._snapshot([paths['graph03_run'], paths['graph02_release'].parent])


def run(originals, mode='build-candidate', run_id='candidate', **options):
    paths, config, output, _ = originals
    result = F.run_source(mode, paths, output, config, ROOT, run_id=run_id, **options)
    return result, output / 'runs' / run_id


def cli(originals, mode, name, *options):
    paths, config, output, _ = originals
    args = [sys.executable, str(CLI), '--' + mode, '--run-id', name,
            '--config', str(config), '--output-root', str(output)]
    for key, path in paths.items():
        args.extend(['--' + key.replace('_', '-'), str(path)])
    return subprocess.run([*args, *options], capture_output=True, text=True)


def manifest(directory):
    return json.loads((directory / 'provenance.json').read_text())


def mutate(paths, role, change):
    frame = pd.read_parquet(paths[role])
    change(frame)
    frame.to_parquet(paths[role], index=False)


def test_candidate_audits_lineage_and_preserves_original_repetitions(originals):
    paths, _, _, _ = originals
    before = paths_before(paths)
    result, run_dir = run(originals)
    assert result['status'] == 'SOURCE_CANDIDATE_UNAPPROVED'
    assert result['estimators_run'] == 0 and result['candidate_is_approved'] is False
    table = pd.read_parquet(run_dir / 'unapproved_observations.parquet')
    rf = table.loc[table.source_column.eq(RF)]
    assert set(rf.temporal_status) == {'PREBASELINE_OFFSET_CANDIDATE', 'POST_BASELINE_OFFSET_COMPARISON'}
    assert rf.groupby(F.KEYS).size().max() == 2
    assert rf.step01_selected_after_baseline.all()
    assert rf.step01_context_date_status.eq('STEP01_STEP10_DATE_CONCORDANT').all()
    assert rf.step01_step10_value_status.eq('STEP01_STEP10_VALUE_CONCORDANT').all()
    assert rf.loc[rf.lab_date.eq(AFTER), 'eda_matches_selected_value'].all()
    assert rf.loc[rf.lab_date.eq(BEFORE), 'eda_matches_selected_value'].eq(False).all()
    assert table[['measurement_date', 'assay_id', 'sample_id']].isna().all().all()
    assert table.original_sample_key.notna().all() and table.order_identifier.ne(table.original_sample_key).all()
    assert 'unrelated_sensitive_column' not in table
    rf_text = table.loc[table.source_column.eq(RF_TEXT)]
    assert rf_text.eda_row_offset.tolist() == rf.eda_row_offset.tolist()
    metadata = manifest(run_dir)
    assert metadata['inputs']['baseline']['status'] == 'VERIFIED'
    assert metadata['outcomes'][RF]['status'] == 'CANDIDATE_UNAPPROVED'
    assert metadata['aggregates']['status'] == 'RESTRICTED_AGGREGATES'
    assert all(digest(run_dir / name) == sha for name, sha in metadata['output_sha256'].items())
    assert set(metadata['review_binding']['frozen_sha256'].values()) == set(before.values())
    assert paths_before(paths) == before
    assert run_dir.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in run_dir.iterdir())
    text = (run_dir / 'provenance.json').read_text() + (run_dir / 'readiness.csv').read_text()
    assert not any(secret in text for secret in ('SYNTHETIC_SAMPLE_', BEFORE, AFTER, 'SYNTHETIC_PLATFORM'))
    readiness = pd.read_csv(run_dir / 'readiness.csv')
    assert len(readiness) == 16
    assert readiness.loc[readiness.source_column.isin(['essdai__total', 'esspri__total']),
                         'pop'].eq('BLOCKED_OUTCOME_DEFINES_POP').all()
    assert readiness.protocol.eq('RESEARCH_PROTOCOL_ASSIGNMENT_NOT_VERIFIED').all()
    assert all(metadata['outcomes'][s]['search_status'] == 'NOT_VERIFIED' for s in F.NONLAB)
    with pytest.raises(FileExistsError, match='ALREADY_EXISTS'):
        run(originals)


def test_candidate_blocks_uncertain_times_crossed_ids_and_small_cells(originals):
    paths, _, _, _ = originals

    def change(frame):
        frame.loc[0, 'patient_id'] = '00-' + frame.loc[0, 'patient_id']
        frame.loc[0, 'lab_date'] = '2020-01-01T08:30:00Z'
        frame.loc[1, 'lab_date'] = '2019-12-31T12:40:00'
        frame.loc[2, 'lab_date'] = '2019-12-31'
        frame.loc[3, 'lab_date'] = None
        frame.loc[4, 'specimen_datetime'] = BEFORE
        frame.loc[4, 'specimen_id'] = None
        frame.loc[5, 'matched_clinical_episode_id'] = frame.loc[6, 'matched_clinical_episode_id']
        frame.loc[7, 'patient_id'] = 'UNRECONCILED'
        frame.loc[8, 'episode_match_ambiguous'] = True
        frame.loc[9, 'result_raw'] = '<2'
        frame.loc[10, 'result_operator'] = '>'
        frame.loc[11, 'result_text'] = 'not tested'
        frame.loc[12, 'result_text'] = 'unknown'

    mutate(paths, 'eda_long', change)
    mutate(paths, 'context', lambda frame: frame.__setitem__('lab__rheumatoid_factor__measurement_date',
                                                           '2020-01-03'))
    _, run_dir = run(originals)
    table = pd.read_parquet(run_dir / 'unapproved_observations.parquet')
    rf = table.loc[table.source_column.eq(RF)].set_index('eda_row_offset')
    assert rf.loc[0, 'temporal_status'] == 'SAME_DAY_ORDER_UNVERIFIED'
    assert rf.loc[0, 'eda_patient_id'] == '00-' + rf.loc[0, 'patient_id']
    assert rf.loc[1, 'date_precision'] == 'LOCAL_TIME_ZONE_UNVERIFIED'
    assert rf.loc[2, 'date_precision'] == 'DAY_OR_MIDNIGHT_ONLY'
    assert rf.loc[3, 'date_precision'] == 'MISSING'
    assert pd.isna(rf.loc[4, 'original_sample_key'])
    assert rf.loc[4, 'specimen_datetime'] == BEFORE
    assert rf.loc[[9, 10], 'censored'].all() and rf.loc[[9, 10], 'value'].isna().all()
    text = table.loc[table.source_column.eq(RF_TEXT)].set_index('eda_row_offset')
    assert text.loc[[11, 12], 'value'].tolist() == ['not tested', 'unknown']
    assert rf.step01_context_date_status.eq('STEP01_STEP10_DATE_DISCORDANT').all()
    ledger = pd.read_parquet(run_dir / 'private_ledger.parquet').set_index('eda_row_offset')
    assert ledger.loc[5, 'identity_status'] == 'EDA_EPISODE_ASSOCIATED_WITH_OTHER_PATIENT'
    assert ledger.loc[7, 'identity_status'] == 'EDA_PATIENT_NOT_RECONCILED'
    assert ledger.loc[8, 'identity_status'] == 'EDA_EPISODE_MATCH_AMBIGUOUS_OR_UNDOCUMENTED'
    assert not {5, 7, 8} & set(rf.index)
    assert manifest(run_dir)['aggregates'] == {'status': 'WITHHELD_SMALL_CELLS', 'by_outcome': {}}


@pytest.mark.parametrize('case,blocker', [
    ('missing', 'UPSTREAM_MISSING_OR_UNREADABLE'), ('corrupt', 'UPSTREAM_MISSING_OR_UNREADABLE'),
    ('schema', 'UPSTREAM_REQUIRED_COLUMNS_MISSING'), ('frozen', 'FROZEN_RELEASES_NOT_AVAILABLE'),
])
def test_blocked_cli_audit_writes_only_aggregate_evidence(originals, case, blocker):
    paths, _, output, _ = originals
    if case == 'missing':
        paths['eda_long'].unlink()
    if case == 'corrupt':
        paths['eda_long'].write_bytes(b'synthetic corrupt parquet')
    if case == 'schema':
        pd.DataFrame({'unrelated': [0]}).to_parquet(paths['eda_long'])
    if case == 'frozen':
        paths['graph03_run'] = paths['graph03_run'].with_name('missing_release')
    proc = cli(originals, 'audit-only', 'blocked')
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)['status'] == 'SOURCE_CONTRACT_NOT_READY'
    run_dir = output / 'runs/blocked'
    assert set(p.name for p in run_dir.iterdir()) == {'provenance.json', 'readiness.csv'}
    assert blocker in manifest(run_dir)['blockers']


def test_audit_only_with_sources_never_writes_patient_artifacts(originals):
    proc = cli(originals, 'audit-only', 'audit')
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)['status'] == 'SOURCE_AUDIT_COMPLETE'
    assert {p.name for p in (originals[2] / 'runs/audit').iterdir()} == {'provenance.json', 'readiness.csv'}


def test_nonlab_and_protocol_schema_columns_cannot_approve_dates_or_pop(originals, tmp_path):
    paths, _, _, _ = originals
    optional = {
        'essdai_original': {'essdai__essdai_total_score': [3], 'assessment_date': [BEFORE]},
        'esspri_original': {'esspri_questionnaire__pain': [3]},
        'biopsy_original': {'biopsy_pathology__f_score': [2]},
        'protocol_original': {'demo__protocol': ['SYNTHETIC_A|SYNTHETIC_B']},
    }
    for role, rows in optional.items():
        paths[role] = tmp_path / (role + '.parquet')
        pd.DataFrame(rows).to_parquet(paths[role])
    _, run_dir = run(originals, 'audit-only')
    metadata = manifest(run_dir)
    assert metadata['inputs']['essdai_original']['fields']['assessment_date']['status'] == 'CANDIDATE'
    assert metadata['inputs']['protocol_original']['fields']['assignment_date']['status'] == 'NOT_AVAILABLE'
    assert all(metadata['outcomes'][s]['reason'] == 'ORIGINAL_INSTRUMENT_DATE_NOT_VERIFIED' for s in F.NONLAB)


@pytest.mark.parametrize('case', ['unsafe_root', 'unsafe_runs', 'frozen_output', 'baseline', 'tampering', 'duplicate_key'])
def test_rejects_unsafe_outputs_and_invalid_pins(originals, tmp_path, case):
    paths, config, output, _ = originals
    if case == 'unsafe_root':
        output.mkdir(mode=0o755)
        output.chmod(0o755)
    if case == 'unsafe_runs':
        output.mkdir(mode=0o700)
        (output / 'runs').mkdir(mode=0o755)
        (output / 'runs').chmod(0o755)
    if case == 'frozen_output':
        output = paths['graph03_run'] / 'injected'
    if case == 'baseline':
        paths['baseline'] = tmp_path / 'wrong.csv'
        paths['baseline'].write_text('wrong')
    if case == 'tampering':
        with (paths['graph03_run'] / '03_graph_n3_source_inventory.csv').open('a') as handle:
            handle.write('\n')
    if case == 'duplicate_key':
        selected = pd.read_parquet(paths['step01'])
        pd.concat([selected, selected.iloc[:1]]).to_parquet(paths['step01'], index=False)
    with pytest.raises((ValueError, OSError)) as exc:
        F.run_source('build-candidate', paths, output, config, ROOT, run_id='blocked')
    assert str(exc.value)
    assert not (output / 'runs/blocked').exists()


def test_fails_atomically_if_source_changes_during_run(originals, monkeypatch):
    real = F._outcome_audit

    def changed(*args):
        with originals[0]['eda_long'].open('ab') as handle:
            handle.write(b'tampered')
        return real(*args)

    monkeypatch.setattr(F, '_outcome_audit', changed)
    with pytest.raises(ValueError, match='INPUT_CHANGED_DURING_RUN'):
        run(originals)
    assert list((originals[2] / 'runs').iterdir()) == []


def review_rule(source=RF, **updates):
    kind = 'numeric' if source == RF else 'binary'
    inspection = reviews(fixture()[-2])['measurement']
    inspection.update(assay_id='SYNTHETIC_REVIEWED_PLATFORM', unit='SYNTHETIC_SCALE', type=kind)
    if kind == 'binary':
        inspection['category_convention'] = dict(positive=['positive'], negative=['negative'],
                                                 unknown=['unknown'], not_tested=['not tested'])
    rule = dict.fromkeys(AFFIRMATIONS, True)
    rule.update(sample_field='specimen_id', assay_field='assay', unit_field='unit', date_field='lab_date',
                original_assay='SYNTHETIC_PLATFORM', value_field='result_numeric_exact' if kind == 'numeric'
                else 'result_text', measurement_review=inspection)
    rule.update(updates)
    return rule


def signed_review(tmp_path, binding, rules, *, version=REVIEW_VERSION, **metadata):
    # Ephemeral test trust anchor and signed synthetic evidence.
    review = tmp_path / 'source_review.yaml'
    attachment = tmp_path / 'synthetic_evidence.txt'
    attachment.write_text('Synthetic fixture. No clinical approval.\n')
    key, public, sig = (tmp_path / name for name in ('key.pem', 'public.pem', 'signature.bin'))
    review.write_text(yaml.safe_dump(dict(schema_version=version, decision='APPROVED_SOURCE_PROVENANCE',
                                         reviewer='SYNTHETIC', review_id='SYNTHETIC', **metadata,
                                         input_sha256=binding, outcomes=rules,
                                         evidence=[dict(path=attachment.name, sha256=digest(attachment))])))
    def openssl(*args):
        subprocess.run(['openssl', *args], check=True, capture_output=True)

    openssl('genpkey', '-algorithm', 'RSA', '-pkeyopt', 'rsa_keygen_bits:2048', '-out', str(key))
    openssl('pkey', '-in', str(key), '-pubout', '-out', str(public))
    openssl('dgst', '-sha256', '-sign', str(key), '-out', str(sig), str(review))
    return dict(review_path=review, signature_path=sig, trust_key=public)


def test_reviewed_cli_and_03e_prepare_plan_preflight_require_separate_decisions(originals, tmp_path):
    paths, _, output, decision_root = originals
    before = paths_before(paths)
    _, audit = run(originals, 'audit-only', 'audit')
    documents = signed_review(tmp_path, manifest(audit)['review_binding'], {RF: review_rule()})
    reviewed = cli(originals, 'build-reviewed', 'reviewed',
                   '--review', str(documents['review_path']), '--signature', str(documents['signature_path']),
                   '--trust-key', str(documents['trust_key']))
    assert reviewed.returncode == 0, reviewed.stderr
    assert json.loads(reviewed.stdout)['status'] == 'SOURCE_REVIEWED_PENDING_03E_DECISIONS'
    run_dir = output / 'runs/reviewed'
    measurements = run_dir / 'reviewed_instrument_measurements.parquet'
    result = pd.read_parquet(measurements)
    assert list(result) == F.CORE
    assert len(result) == 100 and set(result.value) == {'3.0'}
    assert (result.measurement_date < pd.Timestamp('2020-01-01', tz='UTC')).all()
    metadata = manifest(run_dir)
    assert metadata['outcomes'][RF]['status'] == 'VERIFIED'
    assert metadata['eligible_for_inference'] is False

    ledger = pd.read_parquet(run_dir / 'private_reviewed_ledger.parquet')
    assert set(ledger.reviewed_disposition) == {'RETAINED_FOR_03E_SELECTION', 'AFTER_BASELINE'}
    assert ledger.reviewed_row_offset.dropna().tolist() == list(range(len(result)))

    plan, templates, preflight = (tmp_path / name for name in ('draft-plan.yaml', 'templates', 'preflight'))
    base = [sys.executable, str(INFERENCE), '--graph03-run', str(paths['graph03_run']),
            '--graph02-release', str(paths['graph02_release']), '--context', str(paths['context']),
            '--measurements', str(measurements), '--plan', str(plan), '--decision-root', str(decision_root)]
    draft = subprocess.run([*base, '--prepare-plan', '--output-dir', str(templates)], capture_output=True, text=True)
    assert draft.returncode == 0, draft.stderr
    assert json.loads(draft.stdout)['status'] == 'UNSIGNED_DRAFT_NO_APPROVAL'
    amended = yaml.safe_load(plan.read_text())
    amended['outcomes'] = [dict(representation_id='R0', source_column=RF, requested_models=['A', 'B'],
                               n1_basis=[], use_pop=False, topology_scope='full', model_family='gaussian_ols',
                               hypothesis_family='SYNTHETIC')]
    plan.write_text(yaml.safe_dump(amended))
    check = subprocess.run([*base, '--output-dir', str(preflight)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    result = json.loads(check.stdout)
    assert result['status'] == 'PREFLIGHT_COMPLETE'
    assert json.loads((preflight / 'release.json').read_text())['clinical_feasibility'] == (
        'NO_APPROVED_ESTIMABLE_CONTRAST_IN_THIS_RUN')
    assert 'COMPLETE' not in set(pd.read_csv(preflight / 'models.csv').status)
    assert paths_before(paths) == before


@pytest.mark.parametrize('case,reason', [
    ('untrusted', 'SIGNATURE_INVALID'), ('attachment', 'EVIDENCE_INVALID'), ('source_pin', 'INPUT_PINS_INVALID'),
    ('order', 'SAMPLE_IDENTITY_UNVERIFIED'), ('missing_sample', 'SAMPLE_IDENTITY_UNVERIFIED'),
    ('duplicate_sample_other_patient', 'SAMPLE_IDENTITY_UNVERIFIED'), ('assay_alias', 'ASSAY_IDENTITY_REQUIRED'),
    ('missing_assay', 'NON_EQUIVALENT_ASSAY'), ('mixed_assay', 'NON_EQUIVALENT_ASSAY'),
    ('mixed_unit', 'NON_EQUIVALENT_ASSAY_OR_UNIT'), ('censored', 'VALUE_CENSORED'),
    ('date_only', 'TIME_PRECISION'), ('same_day', 'TIME_PRECISION'), ('timezone', 'TIME_PRECISION'),
    ('tie', 'TIE_REQUIRES'), ('shared_ana', 'SHARE_ORIGINAL_ASSAY'), ('nonresult', 'TOKEN_NOT_MISSING'),
    ('lineage', 'STEP01_STEP10_LINEAGE_DISCORDANT'), ('episode', 'EDA_EPISODE_LINKAGE_UNRESOLVED'),
])
def test_signed_review_fails_closed_for_tampering_or_unresolved_evidence(originals, tmp_path, case, reason):
    paths, _, output, _ = originals

    def change(frame):
        if case == 'order':
            frame.loc[0, 'specimen_id'] = frame.loc[0, 'order_identifier']
        if case == 'missing_sample':
            frame.loc[0, 'specimen_id'] = None
        if case == 'duplicate_sample_other_patient':
            frame.loc[0, 'specimen_id'] = frame.loc[1, 'specimen_id']
        if case == 'missing_assay':
            frame.loc[0, 'assay'] = None
        if case == 'mixed_assay':
            frame.loc[0, 'assay'] = 'UNVERIFIED_DIFFERENT_ASSAY'
        if case == 'mixed_unit':
            frame.loc[0, 'unit'] = 'DIFFERENT_UNIT'
        if case == 'censored':
            frame.loc[0, 'result_operator'] = '>'
        if case in {'date_only', 'same_day', 'timezone'}:
            frame.loc[0, 'lab_date'] = {
                'date_only': '2019-12-31', 'same_day': '2020-01-01T10:00:00Z',
                'timezone': '2019-12-31T10:00:00'}[case]
        if case == 'tie':
            frame.loc[len(frame)] = frame.iloc[0]
            frame.loc[len(frame) - 1, 'result_numeric_exact'] = 3.5
        if case == 'nonresult':
            frame.loc[0, 'result_text'] = 'not tested'
        if case == 'episode':
            frame.loc[0, 'episode_match_ambiguous'] = True

    mutate(paths, 'eda_long', change)
    if case == 'lineage':
        mutate(paths, 'context', lambda frame: frame.__setitem__('lab__rheumatoid_factor__measurement_date', '2020-01-03'))
    rules = {RF: review_rule()}
    if case == 'shared_ana':
        frame = pd.read_parquet(paths['eda_long'])
        pd.concat([frame.assign(canonical_analyte=F.LAB_SOURCES[source]) for source in (ANA, HEP2)]).to_parquet(
            paths['eda_long'], index=False)
        rules = {source: review_rule(source) for source in (ANA, HEP2)}
    if case == 'nonresult':
        rules = {RF_TEXT: review_rule(RF_TEXT)}
        rules[RF_TEXT]['measurement_review']['category_convention']['not_tested'] = []
    if case == 'assay_alias':
        rules[RF]['measurement_review']['assay_id'] = 'Rheumatoid Factor'
    _, audit = run(originals, 'audit-only')
    binding = manifest(audit)['review_binding']
    if case == 'source_pin':
        binding['source_sha256']['eda_long'] = '0' * 64
    documents = signed_review(tmp_path, binding, rules)
    if case == 'untrusted':
        with documents['review_path'].open('a') as handle:
            handle.write('\n')
    if case == 'attachment':
        (tmp_path / 'synthetic_evidence.txt').write_text('changed')
    with pytest.raises(ValueError, match=reason):
        run(originals, 'build-reviewed', 'rejected', **documents)
    assert not (output / 'runs/rejected').exists()


def test_duplicate_original_rows_can_survive_review_without_changing_selection(originals, tmp_path):
    frame = pd.read_parquet(originals[0]['eda_long'])
    pd.concat([frame, frame.iloc[:1]]).to_parquet(originals[0]['eda_long'], index=False)
    _, audit = run(originals, 'audit-only')
    rule = review_rule()
    documents = signed_review(tmp_path, manifest(audit)['review_binding'], {RF: rule})
    _, result = run(originals, 'build-reviewed', 'reviewed', **documents)
    observations = pd.read_parquet(result / 'reviewed_instrument_measurements.parquet')
    baseline = pd.read_csv(originals[0]['baseline'], dtype={'patient_id': 'string'})
    values, kind = measurement_values(baseline, observations, RF, rule['measurement_review'])
    assert len(observations) == 101 and kind == 'numeric' and values.eq(3).all()


@pytest.mark.parametrize('field,reason', [
    ('sample_id', 'SAMPLE_IDENTITY_MISSING'), ('assay_id', 'NON_EQUIVALENT_ASSAY'),
    ('unit', 'NON_EQUIVALENT_ASSAY'), ('source_artifact_status', 'UNAPPROVED_MEASUREMENT_CANDIDATE'),
])
def test_03e_rejects_unapproved_candidates_and_nullable_identifiers(field, reason):
    frozen, _, measurements, _, _, spec, _ = fixture()
    measurements[field] = pd.Series(
        'SOURCE_CANDIDATE_UNAPPROVED' if field == 'source_artifact_status' else pd.NA,
        index=measurements.index, dtype='string')
    with pytest.raises(ValueError, match=reason):
        measurement_values(frozen[1], measurements, spec['source_column'], reviews(spec)['measurement'])

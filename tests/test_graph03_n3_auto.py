"""Exercise the signed baseline window runner with synthetic inputs and keys."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from src.studies.graph._pipeline import digest
from src.studies.graph.stage03 import s03_n3_calendar_day as cal
from src.studies.graph.stage03.s03_baseline_window import VERSION as WINDOW_VERSION, policy
from src.studies.graph.stage03.s03_n3_exploratory_source import DEFINITION
from src.studies.graph.stage03.s03_n3_measurement_review import CALENDAR_AFFIRMATIONS
from src.studies.graph.stage03.s03_run_resolution import graph_paths
from tests.test_graph03_n3_calendar_day import calendar_originals, inspection, source_rule, write_btris
from tests.test_graph03_n3_controlled import attestations
from tests.test_graph03_n3_measurement_source import RF, ROOT, originals, paths_before, signed_review


def write_yaml(path, doc):
    path.write_text(yaml.safe_dump(doc))


def run_command(*args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, **kwargs)


def sign(document, signature, key):
    run_command('openssl', 'dgst', '-sha256', '-sign', str(key), '-out', str(signature), str(document))


@pytest.fixture
def automatic(calendar_originals, tmp_path):
    paths, prespec_config, private, _ = calendar_originals
    repo = tmp_path / 'checkout'
    repo.mkdir()
    shutil.copytree(ROOT / 'src', repo / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('common.py', 'config.py'):
        shutil.copy(ROOT / name, repo / name)
    run_command('git', 'init', '-q', str(repo))
    run_command('git', '-c', 'user.name=Synthetic', '-c', 'user.email=synthetic@example.com',
                'commit', '--allow-empty', '-qm', 'Synthetic base', cwd=repo)

    data = tmp_path / 'study-data'
    folders = graph_paths(repo, data_root=data)
    for role, source in (('graph02', paths['graph02_release'].parent), ('graph03', paths['graph03_run'])):
        target = folders[role] / 'runs' / source.name
        target.parent.mkdir(parents=True)
        target.symlink_to(source, target_is_directory=True)
    release = paths['graph02_release']
    manifest = json.loads(release.read_text())
    manifest['created_at_utc'] = '2026-01-01T00:00:00Z'
    release.write_text(json.dumps(manifest))
    graph03 = paths['graph03_run']
    for path in graph03.glob('*.json'):
        metadata = json.loads(path.read_text())
        metadata.update(created_at_utc='2026-01-01T00:01:00Z', graph02_release_sha256=digest(release))
        path.write_text(json.dumps(metadata))
    summary = graph03 / '03_graph_results_summary.json'
    metadata = json.loads(summary.read_text())
    metadata['sha256_by_artifact'] = {p.name: digest(p) for p in graph03.iterdir() if p != summary}
    summary.write_text(json.dumps(metadata))

    cfg = repo / 'src/studies/graph/configs/n3_auto.yaml'
    config = yaml.safe_load(cfg.read_text())
    config.update(data_root=str(data), private_root=str(private), eda_root=str(tmp_path / 'eda-checkout'),
                  source_paths={role: str(paths[role]) for role in config['source_paths']})
    write_yaml(cfg, config)
    # A legacy adapter file is available to catch accidental fallback.
    legacy = data / 'data/analytic/10_build_integrated_longitudinal_dataset'
    legacy.mkdir(parents=True)
    (legacy / '10_n3_instrument_measurements.parquet').write_bytes(b'SYNTHETIC_LEGACY_SENTINEL')

    eda = pd.read_parquet(paths['eda_long']).assign(lab_date='2020-01-01', unit='IU/mL')
    write_btris(paths, eda, repo.parent)
    eda['result_numeric_exact'] = 2 + (eda.index % 13) / 3
    eda['result_raw'] = eda['result_numeric_exact'].astype(str)
    eda.to_parquet(paths['eda_long'], index=False)
    return repo, paths, private, prespec_config


def auto(automatic):
    repo, _, _, _ = automatic
    code = ('import json; from pathlib import Path; '
            'from src.studies.graph.stage03.s03_n3_auto import run_auto; '
            'print(json.dumps(run_auto(Path.cwd())))')
    proc = run_command(sys.executable, '-c', code, cwd=repo)
    assert not proc.stderr
    for secret in ('SYNTHETIC_0', 'SYNTHETIC_EP_', '2019-12-31', 'SYNTHETIC_LEGACY_SENTINEL'):
        assert secret not in proc.stdout
    result = json.loads(proc.stdout)
    assert result['mode'] == 'baseline_calendar_window'
    if result['report_path']:
        report = Path(result['report_path'])
        assert report.stat().st_mode & 0o777 == 0o600
        assert report.parent.stat().st_mode & 0o777 == 0o700
        assert 'SYNTHETIC_0' not in report.read_text() and '2019-12-31' not in report.read_text()
    return result


def report(result):
    return json.loads(Path(result['report_path']).read_text())


def provision_prespecification(automatic, *, complete=True, **updates):
    _, _, private, config = automatic
    institution = private / 'institution'
    evidence_folder = institution / 'prespecification_evidence'
    evidence_folder.mkdir(parents=True, mode=0o700)
    private.chmod(0o700)
    evidence = evidence_folder / 'synthetic_dictionary.txt'
    evidence.write_text('Synthetic linkage and collection format review.')
    trust = private / 'trust'
    trust.mkdir(mode=0o700)
    key = private.parent / 'prespec_test_key.pem'
    run_command('openssl', 'genpkey', '-algorithm', 'RSA', '-pkeyopt', 'rsa_keygen_bits:2048', '-out', str(key))
    run_command('openssl', 'pkey', '-in', str(key), '-pubout', '-out', str(trust / 'prespecification_public.pem'))
    doc = yaml.safe_load(config.read_text())
    doc.update(schema_version=cal.BASELINE_SOURCE_VERSION, window_policy_version=WINDOW_VERSION)
    doc['calendar_day'].update(contract=cal.CONTRACT, windows={RF: dict(window_policy_version=WINDOW_VERSION)})
    doc['source_paths'] = {}
    doc['calendar_day']['date_formats']['intermediate'] = 'ISO8601'
    if not complete:
        doc['calendar_day']['windows'] = {}
    doc['calendar_day'].update(updates)
    doc.update(decision='APPROVED_CALENDAR_DAY_PRESPECIFICATION', reviewer='SYNTHETIC',
               review_id='SYNTHETIC', evidence=[dict(path=str(evidence.relative_to(institution)), sha256=digest(evidence))])
    target = institution / 'prespecification.yaml'
    write_yaml(target, doc)
    sign(target, institution / 'prespecification.sig', key)
    return key


def baseline_inspection(paths, **details):
    sample = pd.read_parquet(paths['eda_long']).iloc[0]
    definition = {key: sample.get(key) if pd.notna(sample.get(key)) else None for key in DEFINITION}
    definition_id = cal.fingerprint(json.dumps(list(definition.values()), ensure_ascii=False))
    old = inspection(**details)
    old.pop('lookback_days')
    old.pop('minimum_days_before_baseline')
    old.update(measurement_contract=cal.CONTRACT, window_policy_version=WINDOW_VERSION,
               selection_rule=policy().selection_primary, test_definition_id=definition_id, unit=sample.unit)
    return old, definition


def provision_source(automatic, pending):
    _, paths, private, _ = automatic
    audit = Path(report(pending)['artifacts']['source_audit'])
    binding = json.loads(audit.read_text())['review_binding']
    signing = private.parent / 'synthetic_source_signing'
    signing.mkdir()
    inspection, definition = baseline_inspection(paths)
    rule = dict.fromkeys(CALENDAR_AFFIRMATIONS, True)
    rule.update(exact_original_test=definition, measurement_review=inspection)
    files = signed_review(signing, binding, {RF: rule}, version=cal.BASELINE_REVIEW_VERSION,
                          window_policy_version=WINDOW_VERSION)
    folder = private / 'institution' / 'source'
    folder.mkdir(mode=0o700)
    shutil.copy(files['review_path'], folder / 'review.yaml')
    shutil.copy(signing / 'synthetic_evidence.txt', folder)
    return files, folder


def provision_plan(automatic, pending, key, sections, requested=None, *, sign_plan=True):
    private = automatic[2]
    src = private / 'institution' / 'source' / 'review.yaml'
    inference = private / 'institution' / 'inference'
    decisions = inference / 'decisions'
    decisions.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copy(private / 'trust' / 'prespecification_public.pem', private / 'trust' / 'pair_public.pem')
    plan = yaml.safe_load(Path(report(pending)['artifacts']['unsigned_plan']).read_text())
    spec = dict(representation_id='R0', source_column=RF, requested_models=requested or ['A', 'B'],
                n1_basis=[], use_pop=False, topology_scope='full', model_family='gaussian_ols',
                hypothesis_family='SYNTHETIC')
    plan.update(outcomes=[spec], authorized_reviewers=['SYNTHETIC_ONLY_REVIEWER'])
    attestations(decisions, plan, spec)
    plan['decisions'] = [pointer for pointer in plan['decisions'] if pointer['section'] in sections]
    for pointer in plan['decisions']:
        if pointer['section'] == 'measurement':
            path = decisions / pointer['path']
            doc = yaml.safe_load(path.read_text())
            doc['details'], _ = baseline_inspection(automatic[1], source_review_sha256=digest(src))
            write_yaml(path, doc)
            pointer.update(sha256=digest(path), signature_path='measurement.sig')
            sign(path, decisions / 'measurement.sig', key)
    plan_path = inference / 'active.yaml'
    write_yaml(plan_path, plan)
    if sign_plan:
        sign(plan_path, inference / 'active.sig', key)
    return inference


def test_missing_inputs_and_prespecification_stay_on_v2(automatic):
    _, paths, private, _ = automatic
    provision_prespecification(automatic, complete=False)
    missing = paths['eda_long']
    backup = missing.with_suffix('.saved')
    missing.rename(backup)
    result = auto(automatic)
    assert result['status'] == 'SOURCE_INPUTS_MISSING' and result['estimators_run'] == 0
    assert 'SOURCE_INPUT_EDA_LONG_MISSING' in result['blockers']
    assert Path(report(result)['artifacts']['unsigned_prespecification']).is_file()
    backup.rename(missing)
    result = auto(automatic)
    assert result['status'] == 'PENDING_PRESPECIFICATION'
    assert result['blockers'] == ['PRESPECIFICATION_WINDOWS_REQUIRED']
    assert result['estimators_run'] == 0
    assert not (private / 'stages' / 'build-candidate').exists()
    assert 'SOURCE_CONTRACT_NOT_READY' not in json.dumps(result)


def test_date_format_checked_against_btris_without_leaking_values(automatic):
    provision_prespecification(automatic, date_formats={'raw': 'ISO8601', 'intermediate': 'ISO8601'})
    result = auto(automatic)
    assert result['status'] == 'PENDING_PRESPECIFICATION'
    assert result['blockers'] == ['BTRIS_COLLECTION_DATE_FORMAT_UNVERIFIED_BTRIS_RAW_11D']
    assert result['estimators_run'] == 0


def test_audit_candidate_reuse_corruption_and_lock(automatic):
    provision_prespecification(automatic)
    first = auto(automatic)
    assert first['source_status'] == 'SOURCE_CANDIDATE_UNAPPROVED'
    assert first['source_review_status'] == 'SOURCE_REVIEW_PENDING'
    assert 'SOURCE_REVIEW_TRUST_KEY_MISSING' in first['blockers']
    assert first['estimators_run'] == 0
    details = report(first)
    artifacts = details['artifacts']
    assert details['history'] == ['SOURCE_AUDIT_COMPLETE', 'SOURCE_CANDIDATE_UNAPPROVED', 'SOURCE_REVIEW_PENDING']
    candidate = Path(artifacts['source_candidate'])
    assert candidate.is_file()
    assert json.loads(candidate.read_text())['schema_version'] == cal.BASELINE_SOURCE_VERSION
    assert (candidate.parent / 'unapproved_observations.parquet').stat().st_mode & 0o777 == 0o600
    assert auto(automatic) == first
    ledger = candidate.parent / 'private_ledger.parquet'
    ledger.write_bytes(b'synthetic damaged artifact')
    repaired = auto(automatic)
    assert repaired['source_run'] != first['source_run'] and repaired['report_path'] != first['report_path']
    assert report(repaired)['artifacts']['source_candidate'] != artifacts['source_candidate']
    assert ledger.read_bytes() == b'synthetic damaged artifact'
    with (automatic[2] / '.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = auto(automatic)
    assert result['status'] == 'AUTO_RUN_IN_PROGRESS' and result['estimators_run'] == 0


def test_signed_source_plan_and_pair_gates_use_only_ready_models(automatic):
    _, paths, private, _ = automatic
    key = provision_prespecification(automatic)
    before = paths_before(paths) | {paths[role]: digest(paths[role]) for role in ('step10', 'context')}
    pending = auto(automatic)
    source_files, folder = provision_source(automatic, pending)
    shutil.copy(source_files['trust_key'], private / 'trust' / 'source_public.pem')
    unsigned = auto(automatic)
    assert unsigned['status'] == 'SOURCE_REVIEW_PENDING'
    assert unsigned['blockers'] == ['SOURCE_REVIEW_SIGNATURE_MISSING']
    (folder / 'review.sig').write_bytes(b'invalid synthetic signature')
    invalid = auto(automatic)
    assert invalid['status'] == 'SOURCE_REVIEW_INVALID'
    assert invalid['blockers'] == ['SOURCE_REVIEW_SIGNATURE_INVALID']
    shutil.copy(source_files['signature_path'], folder / 'review.sig')
    reviewed = auto(automatic)
    assert reviewed['status'] == 'PLAN_REVIEW_PENDING' and reviewed['estimators_run'] == 0
    assert reviewed['source_review_status'] == 'SOURCE_REVIEWED_PENDING_03E_DECISIONS'
    inference = provision_plan(automatic, reviewed, key, (), sign_plan=False)
    result = auto(automatic)
    assert result['status'] == 'PLAN_REVIEW_PENDING' and result['blockers'] == ['PLAN_SIGNATURE_MISSING']
    sign(inference / 'active.yaml', inference / 'active.sig', key)
    result = auto(automatic)
    assert result['status'] == 'PAIR_REVIEW_PENDING' and result['estimators_run'] == 0
    assert 'PROTOCOL_SEMANTIC_REVIEW_REQUIRED' in result['blockers']
    assert not (private / 'stages' / 'inference').exists()

    sections = {'protocol', 'baseline', 'measurement', 'independence', 'design'}
    provision_plan(automatic, reviewed, key, sections)
    pair_signature = inference / 'decisions' / 'measurement.sig'
    signature_bytes = pair_signature.read_bytes()
    pair_signature.unlink()
    missing_pair_signature = auto(automatic)
    assert missing_pair_signature['status'] == 'PAIR_REVIEW_PENDING'
    assert missing_pair_signature['estimators_run'] == 0
    assert 'CALENDAR_DAY_MEASUREMENT_SIGNATURE_REQUIRED' in missing_pair_signature['blockers']
    pair_signature.write_bytes(signature_bytes)
    result = auto(automatic)
    assert result['status'] == 'COMPLETE_WITH_LIMITATIONS'
    assert result['estimators_run'] == result['n_ready'] == 1
    assert {r['model']: r['status'] for r in report(result)['models']} == {'A': 'COMPLETE', 'B': 'NOT_ESTIMABLE'}
    assert auto(automatic) == result

    provision_plan(automatic, reviewed, key, sections, requested=['A'])
    complete = auto(automatic)
    assert complete['status'] == 'COMPLETE' and complete['estimators_run'] == 1
    assert complete['blockers'] == []
    assert (paths_before(paths) | {paths[r]: digest(paths[r]) for r in ('step10', 'context')}) == before
    eda = pd.read_parquet(paths['eda_long'])
    eda.loc[0, 'result_numeric_exact'] = 9.0
    eda.to_parquet(paths['eda_long'], index=False)
    changed = auto(automatic)
    assert changed['status'] == 'SOURCE_REVIEW_INVALID' and changed['estimators_run'] == 0
    assert changed['blockers'] == ['SOURCE_REVIEW_SCOPE_OR_INPUT_PINS_INVALID']
    assert changed['source_run'] != result['source_run']
    assert Path(complete['report_path']).is_file()

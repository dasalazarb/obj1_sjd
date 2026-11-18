"""Private, content-addressed Calendar Day v2 operation for 03e."""
from contextlib import contextmanager
import csv
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import warnings

import yaml

from .._pipeline import digest
from . import s03_n3_calendar_day as cal
from .s03_n3_controlled import GATES, input_pins, run_controlled, validate_plan, _implementation_hashes
from .s03_n3_decisions import decision_templates, evidence_path, verify_detached_signature
from .s03_n3_measurement_source import AUDITED_FILES, LAB_INPUTS, ORIGINALS, run_source
from .s03_n3_phasec_adjudication import CANDIDATES
from .s03_n3_resolution import _snapshot
from .s03_run_resolution import graph_paths, resolve_graph03_dependencies

CONFIGS = Path('src/studies/graph/configs')
PRESPEC_APPROVAL = 'APPROVED_CALENDAR_DAY_PRESPECIFICATION'
PHASEC = 'n3_phasec_review.yaml'
CONTROLLED = 'n3_controlled_inference.yaml'
VALIDATION = 'validation_03.yaml'
EXPECTED_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError,
                   yaml.YAMLError, subprocess.SubprocessError)
TOKEN = re.compile(r'[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\Z')
SUMMARY_FIELDS = ('mode', 'status', 'graph02_run', 'graph03_run', 'source_status', 'source_run',
                  'source_review_status', 'plan_status', 'preflight_status', 'n_ready',
                  'estimators_run', 'next_required_action', 'report_path')


def _json(path):
    return json.loads(path.read_text())


def _write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n')


def _write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _error_code(exc, fallback):
    message = str(exc)
    return message if len(message) < 120 and TOKEN.fullmatch(message) else fallback


def _directory(path):
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.stat()
    if path.is_symlink() or not path.is_dir() or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ValueError('UNSAFE_AUTO_WORK_DIRECTORY')


@contextmanager
def _locked(root):
    previous = os.umask(0o077)
    try:
        _directory(root)
        fd = os.open(root / '.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        with os.fdopen(fd, 'rb') as lock:
            info = os.fstat(lock.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError('UNSAFE_AUTO_LOCK')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('AUTO_RUN_IN_PROGRESS') from None
            yield
    finally:
        os.umask(previous)


def _artifact_inventory(directory):
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink() or path.stat().st_mode & 0o077:
            raise ValueError('UNSAFE_AUTO_ARTIFACT')
        if path.is_file() and path != directory / 'receipt.json':
            files[path.relative_to(directory).as_posix()] = digest(path)
        elif not path.is_file() and not path.is_dir():
            raise ValueError('UNSAFE_AUTO_ARTIFACT')
    return files


def _cached(parent, inputs, build, verify, *, verify_reuse=False):
    """Publish complete directories only. Keep damaged entries for investigation."""
    _directory(parent)
    key = cal.fingerprint(inputs)
    target = parent / key
    attempt = 0
    while target.exists() or target.is_symlink():
        try:
            if (target.is_symlink() or target.stat().st_mode & 0o077
                    or (target / 'receipt.json').is_symlink()):
                raise ValueError('UNSAFE_AUTO_ARTIFACT')
            intact = _json(target / 'receipt.json') == {'key': key, 'files': _artifact_inventory(target)}
        except (OSError, ValueError):
            intact = False
        if intact:
            if verify_reuse:
                verify()
            return target
        attempt += 1
        target = parent / f'{key}-{attempt}'
    with TemporaryDirectory(prefix='.staging-', dir=parent) as temporary:
        staging = Path(temporary)
        build(staging, target)
        _write_json(staging / 'receipt.json', {'key': key, 'files': _artifact_inventory(staging)})
        verify()
        staging.rename(target)
    return target


def _hashes(paths):
    return {name: digest(path) if path.is_file() else None for name, path in paths.items()}


def _deposits(directory):
    # Hash names as well as bytes, including detached signatures and attachments.
    if directory.is_symlink() or directory.exists() and not directory.is_dir():
        raise ValueError('UNSAFE_INSTITUTIONAL_DEPOSIT')
    return {path.relative_to(directory).as_posix(): sha for path, sha in _snapshot([directory]).items()}


def _unchanged(snapshot, current):
    if snapshot != current():
        raise ValueError('AUTO_INPUTS_CHANGED_DURING_RUN')


def _relative(root, setting):
    if not isinstance(setting, str) or not setting.strip():
        raise ValueError('AUTO_PATH_CONFIGURATION_INVALID')
    return (root / setting).resolve()


def _settings(root):
    cfg = yaml.safe_load((root / CONFIGS / 'n3_auto.yaml').read_text())
    data = _relative(root, cfg['data_root'])
    eda = _relative(root, cfg['eda_root'])
    private = _relative(root, cfg['private_root'])
    definitions = cfg['source_paths']
    required = {*LAB_INPUTS, *cal.BTRIS_ROLES}
    if not isinstance(definitions, dict) or not required <= set(definitions) <= required | set(ORIGINALS):
        raise ValueError('AUTO_SOURCE_PATHS_INVALID')
    paths = {role: _relative(eda if role.startswith(('eda_', 'btris_')) else data, value)
             for role, value in definitions.items()}
    protected = [root, *paths.values(), *graph_paths(root, data_root=data).values()]
    if any(private.is_relative_to(path) or path.is_relative_to(private) for path in protected):
        raise ValueError('AUTO_WORK_DIRECTORY_OVERLAPS_INPUT')
    return data, private, paths


def _trust_keys(root):
    trust = root / 'trust'
    if (trust.is_symlink() or trust.exists() and
            (not trust.is_dir() or trust.stat().st_mode & 0o022)):
        raise ValueError('UNSAFE_INSTITUTIONAL_TRUST_DIRECTORY')
    keys = {kind: trust / (kind + '_public.pem') for kind in ('prespecification', 'source', 'pair')}
    if any(path.is_symlink() or path.exists() and (not path.is_file() or path.stat().st_mode & 0o022)
           for path in keys.values()):
        raise ValueError('UNSAFE_INSTITUTIONAL_TRUST_KEY')
    return keys


def _missing(files, prefix):
    return [{'code': f'{prefix}_{name.upper()}_MISSING', 'path': str(path)}
            for name, path in files.items() if not path.is_file()]


def _signed_yaml(path, signature, key, code):
    if not 0 < path.stat().st_size <= 2_000_000:
        raise ValueError(code + '_DOCUMENT_SIZE_INVALID')
    payload = path.read_bytes()
    if not verify_detached_signature(payload, signature, key):
        raise ValueError(code + '_SIGNATURE_INVALID')
    return payload, yaml.safe_load(payload)


def _prespecification(files):
    _, cfg = _signed_yaml(files['document'], files['signature'], files['trust_key'], 'PRESPECIFICATION')
    if (not isinstance(cfg, dict) or cfg.get('schema_version') != cal.SOURCE_VERSION
            or cfg.get('source_paths') or cfg.get('privacy_class') != 'INTERNAL_RESTRICTED'
            or cfg.get('public_export') is not False
            or type(cfg.get('minimum_disclosure_cell')) is not int or cfg['minimum_disclosure_cell'] < 5
            or cfg.get('decision') != PRESPEC_APPROVAL
            or any(not isinstance(cfg.get(field), str) or not cfg[field].strip()
                   for field in ('reviewer', 'review_id'))):
        raise ValueError('PRESPECIFICATION_SCOPE_OR_SIGNOFF_INVALID')
    settings = cfg.get('calendar_day')
    if not isinstance(settings, dict) or not settings.get('windows'):
        raise ValueError('PRESPECIFICATION_WINDOWS_REQUIRED')
    cal.validate_settings(settings)
    if any(w['minimum_days_before_baseline'] != 1 for w in settings['windows'].values()):
        raise ValueError('PRESPECIFICATION_PRIMARY_WINDOW_REQUIRES_ONE_DAY_MINIMUM')
    if set(settings.get('date_formats', {})) != set(cal.BTRIS_ORIGINS):
        raise ValueError('PRESPECIFICATION_BTRIS_DATE_FORMATS_REQUIRED')
    if any(settings['linkage_fields']['original_observation_name'][origin] != 'Observation Name'
           for origin in cal.BTRIS_ORIGINS):
        raise ValueError('PRESPECIFICATION_ORIGINAL_OBSERVATION_NAME_REQUIRED')
    evidence = cfg.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('PRESPECIFICATION_COLUMN_AND_DATE_EVIDENCE_REQUIRED')
    for attachment in evidence:
        relative = attachment['path']
        if not isinstance(relative, str) or not relative.startswith('prespecification_evidence/'):
            raise ValueError('PRESPECIFICATION_EVIDENCE_DIRECTORY_REQUIRED')
        if digest(evidence_path(files['document'].parent, relative)) != attachment.get('sha256'):
            raise ValueError('PRESPECIFICATION_EVIDENCE_SHA_MISMATCH')


def _prespecification_template():
    return dict(schema_version=cal.SOURCE_VERSION, decision='PENDING_REVIEW', reviewer=None, review_id=None,
                privacy_class='INTERNAL_RESTRICTED', public_export=False, minimum_disclosure_cell=5,
                evidence=[], calendar_day=dict(contract=cal.CONTRACT, windows={}, date_formats=dict(
                    raw=None, intermediate=None), linkage_fields={
                    field: dict(eda=eda, raw=btris, intermediate=btris) for field, eda, btris in zip(
                        cal.KEY_FIELDS, ('patient_id', 'order_identifier', 'order_name_original', None),
                        ('Patient ID', 'Order ID', 'Order Name', 'Observation Name'))}))


def _report():
    return dict(mode='calendar_day_v2', status=None, graph02_run=None, graph03_run=None,
                source_status='NOT_STARTED', source_run=None, source_review_status='NOT_STARTED',
                plan_status='NOT_STARTED', preflight_status='NOT_STARTED', n_ready=0, estimators_run=0,
                blockers=[], next_required_action=None, report_path=None, history=[], artifacts={})


def _transition(report, field, status):
    report[field] = status
    report['history'].append(status)


def _stop(report, field, status, blockers, action):
    _transition(report, field, status)
    report.update(status=status, next_required_action=action)
    report['blockers'].extend(blockers)


def _source_stage(stages, root, mode, paths, config, provenance, verify, **documents):
    def build(staging, target):
        run_source(mode, paths, staging / 'source', config, root, run_id=target.name,
                   verify_calendar_formats=True, **documents)

    directory = _cached(stages / mode, provenance, build, verify)
    run = directory / 'source' / 'runs' / directory.name
    return run, _json(run / 'provenance.json')


def _draft(staging, root, pins, representations):
    plan = yaml.safe_load((root / CONFIGS / CONTROLLED).read_text())
    plan.pop('source_paths', None)
    plan['input_sha256'] = pins
    _write_yaml(staging / 'unsigned_plan.yaml', plan)
    _write_yaml(staging / 'unsigned_decision_templates.yaml',
                decision_templates(plan, plan['input_sha256'], representations, list(CANDIDATES)))


def _table(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _assess_models(report, preflight, reviewed_sources):
    rows = [row for row in _table(preflight / 'preflight.csv') if row['status'] != 'NOT_REQUESTED']
    by_model = {(r['representation_id'], r['source_column'], r['model']): r for r in rows}
    for row in _table(preflight / 'blockers.csv'):
        if row['technical_status'] in {'PASS', 'NOT_APPLICABLE'}:
            continue
        identity = (row['representation_id'], row['source_column'], row['model'])
        outcome = dict(zip(('representation_id', 'source_column', 'model'), identity))
        gate = row['root_cause']
        reason = row['blocking_reason']
        review_status = row['review_status']
        code = reason
        if gate in {'measurement', 'temporal'} and row['source_column'] not in reviewed_sources:
            code = 'CALENDAR_DAY_SOURCE_OUTCOME_UNREVIEWED'
        elif review_status.startswith('CALENDAR_DAY_MEASUREMENT_SIGNATURE_'):
            code = review_status
        report['blockers'].append(dict(outcome, gate=gate, review_status=review_status,
                                       technical_status=row['technical_status'], code=code, technical_reason=reason,
                                       model_status=by_model[identity]['status']))
    report['models'] = [{key: row[key] for key in ('representation_id', 'source_column', 'model', 'status')}
                        for row in rows]
    report['n_ready'] = sum(row['status'] == 'READY' for row in rows)
    pending = any(
        b['technical_status'] == 'FAILED_WITH_EVIDENCE' and b['gate'] != 'support'
        and b['review_status'] not in {'HUMAN_APPROVED_CONTENT_VERIFIED', 'HUMAN_EXCLUDED'}
        and b['code'] != 'CALENDAR_DAY_SOURCE_OUTCOME_UNREVIEWED' for b in report['blockers'])
    return pending


def _inference(report, root, private, stages, graph03, release, paths, reviewed, approvals, verify):
    measurements = reviewed / 'reviewed_instrument_measurements.parquet'
    source_manifest = _json(reviewed / 'provenance.json')
    context = paths['context']
    inference = private / 'institution' / 'inference'
    decisions = inference / 'decisions'
    plan_files = dict(plan=inference / 'active.yaml', signature=inference / 'active.sig',
                      trust_key=private / 'trust' / 'pair_public.pem')
    pins = input_pins(graph03, release, context, measurements, root)
    representations = _json(release)['eligible_representation_ids']
    draft = _cached(stages / 'draft', dict(pins=pins, template_sha256=digest(root / CONFIGS / CONTROLLED)),
                    lambda staging, _target: _draft(staging, root, pins, representations), verify)
    report['artifacts']['unsigned_plan'] = str(draft / 'unsigned_plan.yaml')
    report['artifacts']['unsigned_pair_decisions'] = str(draft / 'unsigned_decision_templates.yaml')

    action = f'Deposite el plan y la firma institucional en {inference}. Use los hashes del borrador.'
    missing = _missing(plan_files, 'PLAN')
    if missing:
        _stop(report, 'plan_status', 'PLAN_REVIEW_PENDING', missing, action)
        return
    try:
        payload, plan = _signed_yaml(*plan_files.values(), 'PLAN')
        validate_plan(plan, yaml.safe_load((root / CONFIGS / VALIDATION).read_text()))
        if plan.get('input_sha256') != pins:
            raise ValueError('EXPLICIT_INPUT_HASH_PINS_REQUIRED_OR_MISMATCHED')
        if not plan['outcomes']:
            _stop(report, 'plan_status', 'PLAN_REVIEW_PENDING', [{'code': 'PLAN_OUTCOMES_REQUIRED'}], action)
            return
        if any(o['representation_id'] not in representations for o in plan['outcomes']):
            raise ValueError('OUTCOME_REPRESENTATION_OUTSIDE_FROZEN_RELEASE')
    except EXPECTED_ERRORS as exc:
        _stop(report, 'plan_status', 'PLAN_REVIEW_PENDING',
              [{'code': _error_code(exc, 'PLAN_DOCUMENT_INVALID'), 'path': str(plan_files['plan'])}], action)
        return
    _transition(report, 'plan_status', 'PLAN_REVIEWED')

    def controlled(staging, *, execute):
        verify()
        signed_plan = staging / 'verified_plan.yaml'
        signed_plan.write_bytes(payload)
        run_controlled(graph03, release, context, measurements, staging / '03e', signed_plan, decisions, root,
                       execute=execute, measurement_trust_key=plan_files['trust_key'])

    key = dict(pins=pins, reviewed_path=str(reviewed), approvals=approvals,
               plan_sha256=hashlib.sha256(payload).hexdigest())
    try:
        preflight = _cached(stages / 'preflight', key, lambda staging, _: controlled(staging, execute=False),
                            verify) / '03e'
    except EXPECTED_ERRORS as exc:
        _stop(report, 'preflight_status', 'NOT_ESTIMABLE',
              [{'code': _error_code(exc, 'PREFLIGHT_INVALID'), 'path': str(decisions)}],
              'Revise el contrato, las decisiones y los hashes de 03e.')
        return
    report['artifacts']['preflight'] = str(preflight)
    report['artifacts']['unsigned_pair_decisions'] = str(preflight / 'unsigned_decision_templates.yaml')
    pending = _assess_models(report, preflight, source_manifest['reviewed_source']['sources'])
    if not report['n_ready']:
        status = 'PAIR_REVIEW_PENDING' if pending else 'NOT_ESTIMABLE'
        action = (f'Deposite las decisiones por representación y outcome en {decisions} y actualice la firma del plan.'
                  if pending else 'Revise los bloqueos por modelo y gate en el informe privado.')
        _stop(report, 'preflight_status', status, [], action)
        return
    _transition(report, 'preflight_status', 'READY')
    try:
        executed = _cached(stages / 'inference', key, lambda staging, _: controlled(staging, execute=True),
                           verify) / '03e'
    except EXPECTED_ERRORS as exc:
        _stop(report, 'preflight_status', 'INFERENCE_INTERRUPTED',
              [{'code': _error_code(exc, 'INFERENCE_RELEASE_FAILED')}],
              'Compruebe la integridad de insumos y aprobaciones y ejecute nuevamente el comando.')
        report['estimators_run'] = None
        return
    report['artifacts']['inference'] = str(executed)
    model_status = {(r['representation_id'], r['source_column'], r['model']): r
                    for r in _table(executed / 'models.csv')}
    for row in report['models']:
        result = model_status[(row['representation_id'], row['source_column'], row['model'])]
        if row['status'] == 'READY':
            row['status'] = result['status']
            if result.get('fit_reason'):
                report['blockers'].append(dict(row, gate='fit', code=result['fit_reason']))
    report['estimators_run'] = report['n_ready']
    for row in _table(executed / 'comparisons.csv'):
        if row.get('reason'):
            report['blockers'].append({key: row[key] for key in ('representation_id', 'source_column', 'contrast')}
                                      | {'gate': 'comparison', 'code': row['reason']})
    report['status'] = 'COMPLETE_WITH_LIMITATIONS' if report['blockers'] else 'COMPLETE'
    report['history'].append(report['status'])
    report['next_required_action'] = ('Revise las limitaciones por modelo en el informe privado.'
                                      if report['blockers'] else 'Consulte la salida privada de inferencia.')


def _operate(report, root, private, stages, paths, release, graph03, engine, approvals, verify):
    institution = private / 'institution'
    keys = _trust_keys(private)
    prespec = dict(document=institution / 'prespecification.yaml',
                   signature=institution / 'prespecification.sig', trust_key=keys['prespecification'])
    source_files = dict(review=institution / 'source' / 'review.yaml',
                        signature=institution / 'source' / 'review.sig', trust_key=keys['source'])
    missing = [{'code': f'SOURCE_INPUT_{role.upper()}_MISSING', 'path': str(path)}
               for role, path in paths.items() if role in {*LAB_INPUTS, *cal.BTRIS_ROLES}
               and not (cal.btris_files(path) if role in cal.BTRIS_ROLES else path.is_file())]
    missing += report['blockers']
    report['blockers'] = []
    prespec_blockers = _missing(prespec, 'PRESPECIFICATION')
    if not prespec_blockers:
        try:
            _prespecification(prespec)
        except EXPECTED_ERRORS as exc:
            prespec_blockers.append(dict(code=_error_code(exc, 'PRESPECIFICATION_DOCUMENT_INVALID'),
                                         path=str(prespec['document'])))
    if missing or prespec_blockers:
        status = 'SOURCE_INPUTS_MISSING' if missing else 'PENDING_PRESPECIFICATION'
        action = ('Restaure los insumos o releases indicados en el informe privado.' if missing
                  else f'Deposite la prespecificación revisada y su firma en {institution}.')
        _stop(report, 'source_status', status, missing + prespec_blockers, action)
        return

    paths = dict(paths, graph02_release=release, graph03_run=graph03)
    config = prespec['document']
    source_key = dict(engine=engine, prespecification=approvals['prespecification'])
    try:
        audit, manifest = _source_stage(stages, root, 'audit-only', paths, config, source_key, verify)
    except EXPECTED_ERRORS as exc:
        _stop(report, 'source_status', 'SOURCE_AUDIT_BLOCKED',
              [{'code': _error_code(exc, 'SOURCE_AUDIT_INVALID')}],
              'Revise integridad, formatos y configuración de los insumos en el informe privado.')
        return
    report['source_run'] = manifest['run_id']
    report['artifacts']['source_audit'] = str(audit / 'provenance.json')
    if manifest['status'] != 'SOURCE_AUDIT_COMPLETE':
        blockers = [{'code': code} for code in manifest['blockers']]
        if any(any(reason in b['code'] for reason in ('DATE_FORMAT_UNVERIFIED', 'COLUMNS_REQUIRED',
                                                     'OBSERVATION_MAPPING_CONFLICT')) for b in blockers):
            status = 'PENDING_PRESPECIFICATION'
        else:
            status = 'SOURCE_AUDIT_BLOCKED'
        _stop(report, 'source_status', status, blockers,
              'Revise los códigos de 03f y el inventario de la auditoría privada.')
        return
    _transition(report, 'source_status', 'SOURCE_AUDIT_COMPLETE')
    try:
        candidate, manifest = _source_stage(stages, root, 'build-candidate', paths, config, source_key, verify)
        if manifest['status'] != 'SOURCE_CANDIDATE_UNAPPROVED':
            raise ValueError('SOURCE_CANDIDATE_NOT_AVAILABLE')
    except EXPECTED_ERRORS as exc:
        _stop(report, 'source_status', 'SOURCE_AUDIT_BLOCKED',
              [{'code': _error_code(exc, 'SOURCE_CANDIDATE_INVALID')}],
              'Revise la auditoría privada y los insumos de 03f.')
        return
    report['source_run'] = manifest['run_id']
    report['artifacts']['source_candidate'] = str(candidate / 'provenance.json')
    _transition(report, 'source_status', 'SOURCE_CANDIDATE_UNAPPROVED')
    action = f'Deposite el expediente de fuente y su firma en {source_files["review"].parent}.'
    missing = _missing(source_files, 'SOURCE_REVIEW')
    if missing:
        _stop(report, 'source_review_status', 'SOURCE_REVIEW_PENDING', missing, action)
        return
    try:
        reviewed_key = dict(source_key, source_review=approvals['source_review'])
        reviewed, manifest = _source_stage(stages, root, 'build-reviewed', paths, config, reviewed_key, verify,
            review_path=source_files['review'], signature_path=source_files['signature'], trust_key=source_files['trust_key'])
        if manifest['status'] != 'SOURCE_REVIEWED_PENDING_03E_DECISIONS':
            raise ValueError('SOURCE_REVIEWED_ARTIFACT_REQUIRED')
    except EXPECTED_ERRORS as exc:
        _stop(report, 'source_review_status', 'SOURCE_REVIEW_INVALID',
              [{'code': _error_code(exc, 'SOURCE_REVIEW_DOCUMENT_INVALID'), 'path': str(source_files['review'])}], action)
        return
    report['artifacts']['source_reviewed'] = str(reviewed / 'provenance.json')
    report['source_run'] = manifest['run_id']
    report['source_status'] = manifest['status']
    _transition(report, 'source_review_status', 'SOURCE_REVIEWED_PENDING_03E_DECISIONS')
    _inference(report, root, private, stages, graph03, release, paths, reviewed, approvals, verify)


def _engine(root, paths, frozen):
    code = _implementation_hashes(root)
    code.update(_hashes({name: root / name for name in AUDITED_FILES}))
    configs = _hashes({name: root / CONFIGS / name for name in
                       ('n3_auto.yaml', 'pipeline.yaml', PHASEC, CONTROLLED, VALIDATION)})
    sources = {role: str(path) for role, path in paths.items()}
    shared_code = ('common.py', 'config.py', 'src/studies/_shared.py')
    code.update(_hashes({name: root / name for name in shared_code}))
    return dict(code_sha256=code, config_sha256=configs, paths=sources,
                inputs_sha256=_hashes({role: path for role, path in paths.items() if role not in cal.BTRIS_ROLES}),
                btris_sha256={role: cal.file_hashes(paths[role]) for role in cal.BTRIS_ROLES},
                frozen_sha256={str(path): sha for path, sha in _snapshot(frozen).items()},
                implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                runtime=dict(python=platform.python_version(), **{
                    package: version(package) for package in ('numpy', 'pandas', 'pyarrow', 'scipy',
                        'statsmodels', 'scikit-learn', 'matplotlib', 'PyYAML')}))


def _approval_hashes(private):
    institution = private / 'institution'
    if institution.is_symlink() or institution.exists() and not institution.is_dir():
        raise ValueError('UNSAFE_INSTITUTIONAL_DEPOSIT')
    keys = _hashes(_trust_keys(private))
    prespec = {name: institution / name for name in ('prespecification.yaml', 'prespecification.sig')}
    if any(path.is_symlink() or path.exists() and not path.is_file() for path in prespec.values()):
        raise ValueError('UNSAFE_INSTITUTIONAL_DEPOSIT')
    evidence = _deposits(institution / 'prespecification_evidence')
    return dict(prespecification=dict(key=keys['prespecification'], files=_hashes(prespec) |
                                     {'prespecification_evidence/' + name: sha for name, sha in evidence.items()}),
                source_review=dict(key=keys['source'], files=_deposits(institution / 'source')),
                inference=dict(key=keys['pair'], files=_deposits(institution / 'inference')))


def _unsigned_source_review(binding):
    return dict(schema_version=cal.REVIEW_VERSION, decision='PENDING_REVIEW', reviewer=None,
                review_id=None, input_sha256=binding, evidence=[], outcomes={})


def _save_report(staging, target, report, engine, approvals, selection):
    if report['source_status'] in {'SOURCE_INPUTS_MISSING', 'PENDING_PRESPECIFICATION'}:
        name = 'unsigned_prespecification.yaml'
        _write_yaml(staging / name, _prespecification_template())
        report['artifacts']['unsigned_prespecification'] = str(target / name)
    if report['source_review_status'] in {'SOURCE_REVIEW_PENDING', 'SOURCE_REVIEW_INVALID'}:
        name = 'unsigned_source_review.yaml'
        audit = _json(Path(report['artifacts']['source_audit']))
        _write_yaml(staging / name, _unsigned_source_review(audit['review_binding']))
        report['artifacts']['unsigned_source_review'] = str(target / name)
    report['report_path'] = str(target / 'report.json')
    _write_json(staging / 'report.json', dict(report, provenance=dict(
        inputs=engine, approvals=approvals, release_selection=selection)))


def _run(root, report):
    data, private, paths = _settings(root)
    with _locked(private):
        stage_root = private / 'stages'
        _directory(stage_root)
        release = graph03 = None
        frozen, selection = [], []
        try:
            args = SimpleNamespace(graph03_run=None, graph02_release=None, strict=False,
                                   config=root / CONFIGS / PHASEC)
            graph03, release, _, selection = resolve_graph03_dependencies(
                args, root, phase='C', data_root=data, quiet=True, verify_inventory=True)
            frozen = [release.parent, graph03]
            if any(private.is_relative_to(path) or path.is_relative_to(private) for path in frozen):
                raise ValueError('AUTO_WORK_DIRECTORY_OVERLAPS_INPUT')
            report['graph02_run'], report['graph03_run'] = [record['run_id'] for record in selection]
        except RuntimeError:
            report['blockers'].append(dict(code='FROZEN_COMPATIBLE_RELEASES_UNAVAILABLE',
                                           paths={name: str(path) for name, path in
                                                  graph_paths(root, data_root=data).items()
                                                  if name in {'graph02', 'graph03'}}))
        engine = _engine(root, paths, frozen)
        approvals = _approval_hashes(private)

        def verify():
            _unchanged(approvals, lambda: _approval_hashes(private))
            _unchanged(engine, lambda: _engine(root, paths, frozen))

        _operate(report, root, private, stage_root, paths, release, graph03, engine, approvals, verify)
        report_inputs = dict(engine=engine, approvals=approvals, report=report.copy())
        saved = _cached(private / 'runs', report_inputs, lambda staging, target: _save_report(
            staging, target, report, engine, approvals, selection), verify, verify_reuse=True)
        return _json(saved / 'report.json')


def run_auto(repository_root):
    """Return a single aggregate-only JSON summary for the operator."""
    report = _report()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            report = _run(Path(repository_root).resolve(), report)
    except EXPECTED_ERRORS as exc:
        code = _error_code(exc, 'AUTO_CONFIGURATION_OR_INTEGRITY_ERROR')
        report.update(status=code, report_path=None,
                      next_required_action='Revise n3_auto.yaml, permisos o integridad de los insumos y reintente.')
        report['blockers'].append({'code': code})
    summary = {key: report[key] for key in SUMMARY_FIELDS}
    summary['blockers'] = sorted({b['code'] for b in report['blockers'] if b.get('code')})
    return summary

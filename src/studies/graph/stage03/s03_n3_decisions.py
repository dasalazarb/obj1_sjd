"""Content-pinned human decisions for a NEW estimand, never historical approvals.

Human attestations are checked against an explicitly authorized reviewer list,
exact scope, input/plan hashes and source evidence. This is not a digital
signature verifier: institutional identity/signature verification remains human.
No command in this module supplies signoffs or approves a decision.
"""
from pathlib import Path
import hashlib
import json
import re

import yaml

from .._pipeline import digest

VERSION = 'graph03_controlled_n3_v1'
APPROVED = 'APPROVED_FOR_THIS_ESTIMAND'
SECTIONS = ('protocol', 'baseline', 'measurement', 'independence', 'pop', 'topology', 'design')


def plan_sha256(plan):
    """Decision locations/reviewer list cannot change the scientific plan hash."""
    payload = {k: v for k, v in plan.items() if k not in {'decisions', 'authorized_reviewers'}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def evidence_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('EVIDENCE_PATH_MUST_BE_RELATIVE')
    root = Path(root).resolve()
    parts = Path(relative).parts
    if '..' in parts or any((root / Path(*parts[:i])).is_symlink()
                            for i in range(1, len(parts) + 1)):
        raise ValueError('UNSAFE_EVIDENCE_PATH')
    path = root / relative
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError('EVIDENCE_MISSING_OR_TOO_LARGE')
    return path


def verify_decisions(plan, root, input_hashes):
    """Return verified records and controlled states; never emit review prose."""
    decisions, states, protected = {}, {}, {}
    for pointer in plan.get('decisions', []):
        key = (pointer.get('section'), pointer.get('representation_id'), pointer.get('source_column'))
        if key in states or key[0] not in SECTIONS:
            raise ValueError('DUPLICATE_OR_INVALID_DECISION_SCOPE')
        states[key] = 'PENDING_REVIEW'
        try:
            path = evidence_path(root, pointer.get('path'))
            actual = digest(path)
            protected[path] = actual
            if not re.fullmatch('[a-f0-9]{64}', str(pointer.get('sha256', ''))) or actual != pointer['sha256']:
                states[key] = 'INVALID_OR_MODIFIED_DECISION'
                continue
            document = yaml.safe_load(path.read_text())
            if not isinstance(document, dict):
                raise ValueError('INVALID_DECISION')
            identity = (document.get('section'), document.get('representation_id'), document.get('source_column'))
            bound = (identity == key and document.get('schema_version') == VERSION
                     and document.get('plan_sha256') == plan_sha256(plan)
                     and document.get('input_sha256') == input_hashes)
            signed = (document.get('human_signoff') is True and
                      document.get('reviewer') in plan.get('authorized_reviewers', []) and
                      isinstance(document.get('review_id'), str) and bool(document['review_id'].strip()) and
                      document.get('institutional_signature_verified') is True)
            attachments = document.get('evidence', [])
            evidence_ok = isinstance(attachments, list) and bool(attachments)
            for attachment in attachments:
                evidence = evidence_path(root, attachment['path'])
                protected[evidence] = digest(evidence)
                evidence_ok &= protected[evidence] == attachment.get('sha256')
            if not bound or not signed or not evidence_ok:
                states[key] = 'INVALID_SIGNOFF_SCOPE_OR_EVIDENCE'
            elif document.get('decision') == APPROVED:
                decisions[key] = document
                states[key] = 'HUMAN_APPROVED_CONTENT_VERIFIED'
            elif document.get('decision') == 'EXCLUDED_FOR_THIS_ESTIMAND':
                states[key] = 'HUMAN_EXCLUDED'
            else:
                states[key] = 'PENDING_REVIEW'
        except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError):
            states[key] = 'INVALID_SIGNOFF_SCOPE_OR_EVIDENCE'
    return decisions, states, protected


def decision_templates(plan, input_hashes, representations, sources):
    """One unsigned template per section and representation/source; no PHI."""
    common = dict(schema_version=VERSION, plan_sha256=plan_sha256(plan),
                  input_sha256=input_hashes, decision='PENDING_REVIEW', reviewer=None,
                  review_id=None, human_signoff=False, institutional_signature_verified=False,
                  evidence=[], details={})
    templates = []
    for rep in representations:
        for source in sources:
            for section in SECTIONS:
                templates.append(dict(common, section=section, representation_id=rep,
                                      source_column=source))
    return templates


def details_for(decisions, section, rep, source):
    document = decisions.get((section, rep, source), {})
    details = document.get('details', {})
    return details if isinstance(details, dict) else {}

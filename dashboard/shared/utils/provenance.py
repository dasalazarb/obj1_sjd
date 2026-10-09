import hashlib
import json
from shared.loaders.refs import resolve


def evidence_hash(refs, run):
    return hashlib.sha256(json.dumps([resolve(ref, run).raw for ref in refs], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def claim_is_reviewed(claim, run):
    return bool(claim.get("reviewed_by") and claim.get("status") == "reviewed" and claim.get("reviewed_hash") == evidence_hash(claim["evidence_refs"], run))

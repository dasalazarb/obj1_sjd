"""Reuse upstream authentication and dates; add Step 17 code fingerprints."""
import hashlib
from src.block_A.secondary_comparison_contract import (  # noqa: F401
    code_paths as upstream_code_paths, fingerprints, git_identity, load_inputs,
    measurements, metadata, verify_unchanged,
)


def code_paths(root):
    paths = upstream_code_paths(root)
    for pattern in ('temporal_spline_*.py', '17_nonlinear_temporal_heterogeneity.py'):
        for p in (root/'src/block_A').glob(pattern):
            paths[str(p.resolve().relative_to(root.resolve()))] = p.resolve()
    return paths


def sample_fingerprint(data):
    # Only a one-way digest is exported, never the row keys.
    columns = ['patient_id', 'variable_measurement_date', 'observed_pattern', 'time', 'value']
    if data.empty:
        return hashlib.sha256(b'empty').hexdigest()
    encoded = data[columns].sort_values(columns[:2]).to_csv(index=False).encode()
    return hashlib.sha256(encoded).hexdigest()

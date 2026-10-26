from pathlib import Path
import shutil
import pandas as pd
import pytest

from test_secondary_trajectories_by_essdai_pattern import upstream
from secondary_comparison_fixtures import args_for, cli
from src.block_A.secondary_comparison_config import ComparisonConfig, load_config
from src.block_A.secondary_comparison_contract import (
    code_paths, fingerprints, load_inputs, measurements, verify_unchanged,
)
from src.block_A.secondary_trajectory_analysis import ContractError, fixed_groups, sha256


def test_consumes_protected_contract_without_reclassifying(upstream, tmp_path):
    args = args_for(upstream, tmp_path/'output', '--dry-run')
    x, b, p, specs, registry, snapshot, _ = load_inputs(args)
    cfg = ComparisonConfig()
    spec = specs['esspri__total']
    frame, status, _, _ = measurements(x, fixed_groups(p, 3), spec, args.data_cutoff, args.time_axis, cfg)
    assert frame.groupby('patient_id').observed_pattern.first().to_dict() == fixed_groups(p, 3).set_index('patient_id').observed_pattern.to_dict()
    assert frame.groupby('patient_id').head(1).time_since_variable_reference_years.eq(0).all()
    assert frame.threshold_abs.eq(3).all()
    verify_unchanged(snapshot)


def test_hash_mutation_and_origin_are_blocking(upstream, tmp_path):
    args = args_for(upstream, tmp_path/'output', '--dry-run')
    args.input_origin = 'canonical'
    with pytest.raises(ContractError, match='origin'):
        load_inputs(args)
    copy = tmp_path/'input'
    copy.write_bytes(b'unchanged')
    snapshot = fingerprints({'input': copy})
    copy.write_bytes(b'changed')
    with pytest.raises(ContractError, match='changed during run'):
        verify_unchanged(snapshot)


def test_catalog_flags_counts_and_cumulative_history(upstream, tmp_path):
    _, _, _, specs, _, _, _ = load_inputs(args_for(upstream, tmp_path/'output', '--dry-run'))
    assert specs['ovl__status'].is_essdai_derived
    assert not specs['sero__anti_ro_ssa__ever_positive_through_episode'].expected_longitudinal
    from src.block_A.secondary_trajectory_catalog import infer_registered
    assert infer_registered('ovl__n_glandular_manifestations_active').type == 'numeric'


def test_symlink_code_keys_are_canonical(tmp_path):
    root = Path(__file__).resolve().parents[1]
    alias = tmp_path/'alias'
    alias.symlink_to(root, target_is_directory=True)
    actual, linked = code_paths(root), code_paths(alias)
    assert actual == linked
    assert all(not Path(key).is_absolute() for key in actual)
    assert 'src/block_A/secondary_comparison_models.py' in actual


@pytest.mark.parametrize('changes', [dict(minimum_patients_per_group=0), dict(window_years=float('inf')),
    dict(minimum_distinct_times=2.5), dict(auc_horizon=(3, 1)), dict(report_times=(-1,)), dict(version='unknown'), dict(seed=-1)])
def test_config_rejects_invalid_support(changes):
    with pytest.raises((ValueError, TypeError)):
        ComparisonConfig(**changes)


def test_artifact_mutation_between_verification_and_snapshot_is_blocked(upstream, tmp_path, monkeypatch):
    from src.block_A import secondary_comparison_contract as contract
    copied = tmp_path/'copy'
    shutil.copytree(upstream['root'], copied)
    u = dict(upstream)
    for key in ('integrated', 'baseline', 'patterns', 'manifest'):
        u[key] = copied/upstream[key].relative_to(upstream['root'])
    original = contract.fingerprints
    mutated = [False]
    def racing_snapshot(paths):
        if 'step13_artifact_3' in paths and not mutated[0]:
            mutated[0] = True
            artifact = paths['step13_artifact_3']
            artifact.write_bytes(artifact.read_bytes()+b'changed before snapshot')
        return original(paths)
    monkeypatch.setattr(contract, 'fingerprints', racing_snapshot)
    with pytest.raises(ContractError, match='SHA256'):
        contract.load_inputs(args_for(u, tmp_path/'out', '--dry-run'))

"""Step 17 isolated aggregate bundle publication; reuse audited rollback/export."""
from pathlib import Path
from src.block_A.secondary_trajectory_analysis import ContractError
from src.block_A.secondary_comparison_outputs import (  # noqa: F401
    artifact_manifest, install_bundle, safe_table, write_manifest, write_table,
)
from src.block_A.temporal_spline_config import STEM, VERSION

TABLES = (
    '00_variable_eligibility', '00_local_temporal_support', '00_group_support', '00_aggregated_observations',
    '01_model_comparison', '01_fit_attempts', '01_model_coefficients', '01_linear_pairwise_slopes', '02_global_temporal_tests',
    '03_group_curves', '03_within_group_change', '04_temporal_derivatives',
    '04_pairwise_temporal_differences', '04_simultaneous_bands', '05_separation_intervals',
    '05_temporal_characteristics', '05_heterogeneity_classification',
    '06_sensitivity_and_threshold_consistency', '06_non_estimable_registry',
    '06_patient_bootstrap_validation', '07_step16_reconciliation',
)


def directories(root, axis, threshold, scenario, analysis='all'):
    # Partial modes have separate namespaces and never replace a complete run.
    suffix = Path(VERSION)/axis/f'delta{threshold}'/scenario
    if analysis != 'all':
        suffix /= analysis
    return {area: Path(root)/'outputs'/area/'blockA'/STEM/suffix for area in ('tables', 'figures', 'qc')}


def annotate(frame, meta):
    """Resolve known semantic aliases, then reject every unexpected collision.

    Metadata's legacy `family` is the configured multiplicity family. A row's
    family describes its outcome; producers of bands/bootstrap use named fields.
    """
    d = frame.copy()
    row_aliases = {'reference_policy': 'inference_reference_policy', 'family': 'outcome_family'}
    meta_aliases = {'n_patients': 'n_patients_analysis', 'family': 'inference_family',
                    'reference_policy': 'variable_reference_policy'}
    for old, new in row_aliases.items():
        if old in d:
            if new in d:
                raise ContractError(f'annotation column collision: {old} -> {new}')
            d = d.rename(columns={old: new})
    for key, value in meta.items():
        target = meta_aliases.get(key, key)
        if target in d:
            raise ContractError(f'annotation metadata collision: {key} -> {target}')
        d[target] = value
    return d


def preflight(root, destinations, inputs, overwrite=False, backup_root=None):
    root = Path(root).resolve()
    for dest in destinations.values():
        if not dest.resolve().is_relative_to(root/'outputs') or STEM not in dest.parts:
            raise ContractError('unsafe Step 17 destination')
        p = dest
        while p != root:
            if p.is_symlink():
                raise ContractError('symlink in Step 17 output destination')
            p = p.parent
        if dest.exists() and not overwrite:
            raise FileExistsError('Step 17 destination exists: '+str(dest))
        for item in inputs.values():
            path = Path(item['path']).resolve()
            if path.is_relative_to(dest.resolve()) or dest.resolve().is_relative_to(path):
                raise ContractError('output overlaps protected input')
    if overwrite and not backup_root:
        raise ContractError('--overwrite requires --backup-root')
    if backup_root:
        backup = Path(backup_root).resolve()
        if not backup.is_relative_to(root/'outputs/backups'/STEM):
            raise ContractError('backup root must be below outputs/backups/'+STEM)
        if any(backup.is_relative_to(d.resolve()) or d.resolve().is_relative_to(backup) for d in destinations.values()):
            raise ContractError('backup overlaps output')
        if any(Path(i['path']).resolve().is_relative_to(backup) for i in inputs.values()):
            raise ContractError('backup overlaps input')

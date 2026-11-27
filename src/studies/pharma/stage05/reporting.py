"""Fixed aggregate output contracts and rollback-safe publication."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import fcntl
from pathlib import Path
import os
from tempfile import TemporaryDirectory

import pandas as pd

import common
from src.studies._shared import sha256_file, write_json

from .contracts import PREFIX, STAGE
from .figures import FIGURE_SOURCES, figures

MODULE_OUTPUTS = {
    'A': ('ever_pop1_by_baseline_pop', 'pop1_cumulative_risk', 'pop1_risk_curve_source'),
    'B': ('patient_profiles_by_transition', 'serology_profiles_by_transition', 'pop2_converter_profiles'),
    'C': ('domain_drivers_by_transition', 'domain_cooccurrence', 'pro_changes_by_transition',
          'essdai_decomposition', 'leave_domain_out_sensitivity'),
    'D': ('biomarker_correlations_by_pop', 'pop2_biomarker_correlations', 'longitudinal_biomarker_correlations'),
    'E': ('absolute_risk_by_biomarker', 'absolute_risk_support'),
    'F': ('medication_by_transition', 'treatment_confounding', 'trial_eligibility_proxy'),
    'G': ('patient_trajectory_summary', 'sustained_conversion', 'patient_converter_profiles',
          'pretransition_biomarker_trends'),
    'H': ('biomarker_coverage', 'baseline_reconciliation', 'transition_evidence_summary',
          'specimen_availability', 'pro_slopes_fdr'),
}
TABLES = tuple(name for group in MODULE_OUTPUTS.values() for name in group)
MANIFEST = f'{PREFIX}output_manifest.json'
PRODUCER = 'src/studies/pharma/05_pharma_deck_evidence.py'
SCHEMA = 'pharma_stage05_aggregate_v1'
QC_FILES = (f'{PREFIX}preflight.json', f'{PREFIX}risk_bootstrap_qc.json',
            f'{PREFIX}candidate_inventory.csv', MANIFEST)
EVIDENCE = f'{PREFIX}evidence_summary.json'
FIGURE_MANIFEST = f'{PREFIX}figure_manifest.csv'
LOG = f'{PREFIX}deck_evidence.log'
IDENTIFIERS = {'patient_id', 'clinical_episode_id', 'clinical_anchor_date', 'index_date', 'specimen_id',
               'aliquot_id', 'visit_id', 'person_id', 'subject_id'}
SOURCE_BY_MODULE = {
    'A': ['master'], 'B': ['master', 'intervals', 'profile', 'registry', 'lab_availability'],
    'C': ['master', 'intervals', 'registry'], 'D': ['master', 'profile', 'registry', 'candidates'],
    'E': ['master', 'profile', 'intervals', 'sensitivity', 'models', 'support', 'candidates', 'model_summary'],
    'F': ['master', 'intervals', 'registry'],
    'G': ['master', 'profile', 'registry'],
    'H': ['master', 'intervals', 'profile', 'models', 'baseline_reference', 'pro_models', 'specimens'],
}
LIMITATIONS = [
    'First Pop1 documentation is detected at clinical visits and subject to interval censoring.',
    'KM uses last valid ESSDAI evaluation and assumes noninformative censoring.',
    'All interval profiles use origin episodes. Repeated measures and routes may share patients.',
    'ESSDAI domain subtraction and correlation are descriptive and cannot identify causal mechanisms.',
    'Risk projections inherit Step 03 observed-jump Poisson intensities and assume constant covariates and rates.',
    'Absolute probabilities are exploratory scenarios without established calibration or clinical utility.',
    'Prescribed or reported medication names alone do not establish as-of exposure or nonuse.',
    'SSA is an observational proxy and does not establish trial eligibility.',
    'Observed persistence depends on the available follow-up and prespecified stability window.',
    'Internal aggregate outputs may contain small cells and require institutional review before external sharing.',
]


def output_dirs() -> dict[str, Path]:
    return {
        'tables': common.STUDIES_TABLES_DIR / STAGE,
        'figures': common.STUDIES_FIGURES_DIR / STAGE,
        'qc': common.STUDIES_QC_DIR / STAGE,
        'logs': common.STUDIES_LOGS_DIR / STAGE,
    }


@contextmanager
def publication_lock(root: Path, *, shared=False):
    # Lock the persistent outputs directory inode; no separate lock file is necessary.
    root.mkdir(parents=True, exist_ok=True)
    handle = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(handle, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield
    finally:
        os.close(handle)


def _owned_files() -> dict[str, tuple]:
    return {
        'tables': tuple(f'{PREFIX}{table}.csv' for table in TABLES) + (EVIDENCE, FIGURE_MANIFEST),
        'figures': tuple(f'{PREFIX}{figure}.pdf' for figure in FIGURE_SOURCES),
        'qc': QC_FILES,
        'logs': (LOG,),
    }


def _commit_files(staging, destination, transaction: Path) -> None:
    """Install manifest last and restore the previous published set on failure."""
    owned = _owned_files()
    paths = [(group, name) for group, names in owned.items() for name in names if name != MANIFEST]
    paths.append(('qc', MANIFEST))
    previous = transaction / 'previous'
    previous.mkdir()
    for directory in destination.values():
        for parent in (directory, *directory.parents):
            if parent.is_symlink():
                raise ValueError(f'Unsafe Stage05 output directory: {parent}')
        directory.mkdir(parents=True, exist_ok=True)
    changed = []
    try:
        for group, name in paths:
            target, prepared = destination[group] / name, staging[group] / name
            backup = previous / f'{group}-{name}'
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError(f'Unsafe Stage05 output target: {target}')
            if target.exists():
                os.link(target, backup)
            changed.append((target, backup))
            if prepared.is_file():
                os.replace(prepared, target)
            elif target.is_file():
                target.unlink()
    except BaseException:
        for target, backup in reversed(changed):
            if backup.is_file():
                os.replace(backup, target)
            elif target.is_file():
                target.unlink()
        raise


def _quality_check(tables: dict) -> dict:
    disclosure = {}
    for name, frame in tables.items():
        if IDENTIFIERS & set(frame):
            raise ValueError(f'Individual-level columns in aggregate output {name}')
        counts = frame.filter(regex=r'^n_').select_dtypes(include='number')
        positives = counts.where(counts.gt(0)).min().dropna()
        disclosure[name] = {'minimum_positive_by_count_column': positives.to_dict()}
        if {'n_patients', 'n_patients_evaluable'}.issubset(frame) and (
            frame.n_patients_evaluable.gt(frame.n_patients).any()
        ):
            raise ValueError(f'Patient numerator exceeds denominator: {name}')
    return disclosure


def _summarize_table(frame) -> dict:
    statuses = frame.status.value_counts().to_dict() if 'status' in frame else {}
    reasons = frame.reason.loc[frame.reason.ne('')].value_counts().to_dict() if 'reason' in frame else {}
    return {'rows': len(frame), 'row_statuses': statuses, 'reasons': reasons,
            'status': 'estimated' if statuses.get('estimated') or (not statuses and len(frame)) else 'not_estimable'}


def _entry(path: Path, group: str, status: str, columns: list, sources: list,
           estimand: str, reason: str = '') -> dict:
    return dict(path=f'outputs/{group}/studies/{STAGE}/{path.name}', status=status, reason=reason,
                sha256=sha256_file(path) if path.is_file() else None,
                producer=PRODUCER, schema=SCHEMA, columns=columns, sources=sources, estimand=estimand)


def publish(tables, settings, preflight, inventory, bootstrap_qc, optional_status) -> None:
    if set(tables) != set(TABLES):
        raise ValueError('Stage05 output table inventory differs from the configured contract')
    disclosure = _quality_check(tables)
    target = output_dirs()
    root = common.OUTPUTS_DIR
    with publication_lock(root), TemporaryDirectory(prefix='.pharma05-', dir=root) as transaction:
        work = Path(transaction)
        staging = {group: work / group for group in target}
        for folder in staging.values():
            folder.mkdir()
        summaries, entries = {}, []
        for module, output_names in MODULE_OUTPUTS.items():
            summaries[module] = {}
            for name in output_names:
                df = tables[name]
                path = staging['tables'] / f'{PREFIX}{name}.csv'
                df.to_csv(path, index=False)
                summary = _summarize_table(df)
                summaries[module][name] = summary
                reason = ' | '.join(sorted(summary['reasons'])) if summary['status'] == 'not_estimable' else ''
                entries.append(_entry(path, 'tables', summary['status'], list(df.columns),
                                      SOURCE_BY_MODULE[module], name, reason))
        figure_manifest = figures(tables, settings, staging['figures'])
        for row in figure_manifest.itertuples(index=False):
            entries.append(_entry(staging['figures'] / row.figure, 'figures', row.status, [],
                                  row.source_tables.split(' | '), row.figure, row.reason))
        figure_path = staging['tables'] / FIGURE_MANIFEST
        figure_manifest.to_csv(figure_path, index=False)
        entries.append(_entry(figure_path, 'tables', 'estimated', list(figure_manifest.columns),
                              list(FIGURE_SOURCES), 'figure_status_and_source_tables'))

        module_status = {module: ('estimated' if any(v['status'] == 'estimated' for v in outputs.values())
                                  else 'not_estimable') for module, outputs in summaries.items()}
        notes = LIMITATIONS.copy()
        if not settings.get('death_date_column'):
            notes.append('A validated death source was unavailable for competing-risk estimation.')
        evidence = {'status': 'complete_with_reported_gates', 'schema': SCHEMA,
                    'cohorts': {k: v for k, v in preflight.items() if k.startswith('n_')},
                    'modules': {module: {'status': module_status[module], 'outputs': value}
                                for module, value in summaries.items()},
                    'sources': preflight['sources'], 'limitations': notes, 'optional_context': optional_status}
        outputs = [
            ('tables', EVIDENCE, evidence, 'aggregate_evidence_summary'),
            ('qc', QC_FILES[0], {**preflight, 'disclosure': {
                'requires_institutional_review': True,
                'configured_cell_threshold': settings['disclosure_minimum_cell'],
                'aggregate_counts': disclosure}}, 'structural_preflight'),
            ('qc', QC_FILES[1], bootstrap_qc, 'patient_cluster_bootstrap'),
        ]
        for group, filename, payload, name in outputs:
            path = staging[group] / filename
            write_json(path, payload)
            entries.append(_entry(path, group, 'estimated', [], list(preflight['sources']), name))
        inventory_path = staging['qc'] / QC_FILES[2]
        inventory.to_csv(inventory_path, index=False)
        entries.append(_entry(inventory_path, 'qc', 'estimated', list(inventory.columns),
                              ['master', 'registry'], 'verified_mapping_and_candidate_columns'))
        log = staging['logs'] / LOG
        statuses = Counter(entry['status'] for entry in entries)
        log.write_text(f'Pharma 05 completed. Artifact status counts: {dict(statuses)}\n')
        entries.append(_entry(log, 'logs', 'estimated', [], [], 'run_status'))
        write_json(staging['qc'] / MANIFEST, {'schema': SCHEMA, 'producer': PRODUCER, 'artifacts': entries})
        _commit_files(staging, target, work)

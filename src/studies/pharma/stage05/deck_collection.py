"""Package reviewed canonical aggregate paths using only the Python standard library."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import fcntl
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

STAGE05 = 'studies/pharma/05_pharma_deck_evidence'
QC_MANIFEST = f'outputs/qc/{STAGE05}/05_pharma_output_manifest.json'
SCHEMA = 'pharma_stage05_aggregate_v1'
PRODUCER = 'src/studies/pharma/05_pharma_deck_evidence.py'
ARCHIVE = 'pharma_deck_outputs.zip'
BLOCKED_PARTS = {'blockb', 'block_b', 'raw', 'analytic', 'intermediate', 'logs', 'backup', 'backups'}
SENSITIVE_HEADER = re.compile(
    r'(?:^|_)(?:patient|person|subject|participant|clinical_episode|episode|visit|specimen|aliquot)_'
    r'(?:id|ids|key|identifier|identifiers|date)(?:$|_)|^(?:mrn|nih_id|dob|date_of_birth|full_name)$',
    re.IGNORECASE,
)
IDENTIFIER_TOKEN = re.compile(r'(?i)^\s*(?:[a-z0-9][a-z0-9._-]*@[^ ]+\.[a-z]{2,}|[a-f0-9]{32,64})\s*$')
MANIFEST_FIELDS = ('slide', 'required', 'path', 'status', 'included', 'sha256', 'bytes',
                   'minimum_positive_count', 'reason')


@dataclass(frozen=True)
class Requested:
    path: str
    slide: str
    required: bool


def _group(kind, directory, slide, names, *, required=False):
    return [Requested(f'outputs/{kind}/{directory}/{name}', slide, required) for name in names]


def allowlist():
    requests = []
    requests += _group('tables', 'blockA/01_pop_distribution', '2', [
        '01_pop_distribution_counts.csv', '01_pop_distribution_by_visit.csv', '01_table1_by_pop.csv',
        '01_pop_unclassifiable_baseline_reason_counts.csv'], required=True)
    requests += _group('tables', 'blockA/10_pop_transitions', '3', [
        '10_transition_matrix_consecutive_episodes.csv', '10_multistate_transition_rates.csv',
        '10_multistate_intensity_matrix.csv', '10_multistate_sojourn_times.csv',
        '10_multistate_transition_probabilities_1y.csv', '10_multistate_transition_probabilities_2y.csv',
        '10_multistate_transition_probabilities_5y.csv'])
    requests += _group('figures', 'blockA/10_pop_transitions', '3', [
        '10_transition_heatmap.pdf', '10_transition_sankey.pdf', '10_multistate_model_diagram.pdf',
        '10_multistate_probability_heatmap_1y.pdf', '10_multistate_probability_heatmap_2y.pdf'])
    requests += _group('tables', 'blockA/09_pros_longitudinal', '5', [
        '09_pros_by_baseline_pop.csv', '09_pros_longitudinal_change.csv',
        '09_pro_longitudinal_models_crude_adjusted.csv', '09_pro_adjusted_slopes_by_pop.csv'])
    requests += _group('tables', 'blockA/14_risk_factors_progression', '5', [
        '14_leave_source_domain_outcome_models.csv'])
    base = 'studies/pharma'
    upstream = {
        '00_profile_labs': ('6', ['00_pharma_lab_profile.csv', '00_pharma_expected_lab_audit.csv',
                                 '00_pharma_serology_profile.csv', '00_pharma_downstream_impact.csv']),
        '01_run_pharma_main': ('3–5', [
            f'01_pharma_{name}.csv' for name in (
                'feature_availability', 'transition_counts', 'transition_associations',
                'transition_model_diagnostics', 'essdai_change_by_transition',
                'essdai_domains_change_by_transition', 'esspri_change_by_transition',
                'esspri_components_change_by_transition', 'lab_availability', 'lab_changes_by_transition',
                'lab_transition_associations', 'lab_transition_model_diagnostics', 'lab_reference_transitions',
                'serology_status_by_transition', 'ml_feasibility', 'longitudinal_models')]),
        '02_pharma_sensitivity': ('4–6', ['02_pharma_sensitivity_candidates.csv', '02_pharma_sensitivity_models.csv',
                                        '02_pharma_sensitivity_summary.csv']),
        '03_pharma_multistate_biomarkers': ('4–6', [
            '03_pharma_multistate_candidates.csv', '03_pharma_multistate_models.csv',
            '03_pharma_multistate_support.csv', '03_pharma_multistate_summary.csv']),
        '03b_pharma_history_feasibility': ('6', ['03b_pharma_history_support.csv',
                                               '03b_pharma_previous_pop_cells.csv', '03b_pharma_history_summary.csv']),
        '04_pharma_history_aware': ('6', ['04_pharma_history_models.csv', '04_pharma_history_comparison.csv',
                                        '04_pharma_history_summary.csv']),
    }
    for stage, (slide, names) in upstream.items():
        requests += _group('tables', f'{base}/{stage}', slide, names, required=stage[:3] not in {'03b', '04_'})
    requests += _group('figures', f'{base}/01_run_pharma_main', '4–5', [
        '01_pharma_transition_associations_forest.pdf', '01_pharma_essdai_change_by_transition.pdf',
        '01_pharma_essdai_domains_heatmap.pdf', '01_pharma_esspri_change_by_transition.pdf',
        '01_pharma_esspri_components_heatmap.pdf', '01_pharma_lab_change_heatmap.pdf'])
    requests += _group('figures', f'{base}/02_pharma_sensitivity', '4', ['02_pharma_sensitivity_forest.pdf'])
    requests += _group('figures', f'{base}/03_pharma_multistate_biomarkers', '4', ['03_pharma_multistate_forest.pdf'])
    requests += _group('tables', 'studies/graph/01_run_topological_phenotyping', '5–6', [
        '01_graph_feature_manifest.csv', '01_graph_lab_ingestion_audit.csv', '01_graph_pca_variance.csv',
        '01_graph_pca_top_loadings.csv', '01_graph_community_size_summary.csv', '01_graph_community_pop_crosswalk.csv',
        '01_graph_community_characterization.csv', '01_graph_longitudinal_models.csv'])

    stage05 = {
        '2': ('trial_eligibility_proxy',),
        '3': ('ever_pop1_by_baseline_pop', 'pop1_cumulative_risk', 'pop1_risk_curve_source',
              'patient_trajectory_summary', 'sustained_conversion'),
        '4': ('patient_profiles_by_transition', 'serology_profiles_by_transition', 'pop2_converter_profiles',
              'patient_converter_profiles', 'absolute_risk_by_biomarker'),
        '5': ('domain_drivers_by_transition', 'domain_cooccurrence', 'pro_changes_by_transition',
              'essdai_decomposition', 'leave_domain_out_sensitivity', 'biomarker_correlations_by_pop',
              'pop2_biomarker_correlations', 'longitudinal_biomarker_correlations'),
        '6': ('absolute_risk_support', 'biomarker_coverage', 'transition_evidence_summary', 'baseline_reconciliation'),
        '7': ('specimen_availability',),
        'supp': ('medication_by_transition', 'treatment_confounding', 'pretransition_biomarker_trends', 'pro_slopes_fdr'),
    }
    for slide, names in stage05.items():
        requests += _group('tables', STAGE05, slide, [f'05_pharma_{name}.csv' for name in names],
                           required=slide not in {'7', 'supp'})
    requests += _group('tables', STAGE05, 'all', [
        '05_pharma_evidence_summary.json', '05_pharma_figure_manifest.csv'], required=True)
    figure_stems = {
        '2': ('baseline_pop_distribution',),
        '3': ('pop1_cumulative_risk', 'patient_trajectory_patterns'),
        '4': ('absolute_risk_by_biomarker',),
        '5': ('pop2_biomarker_correlations', 'domain_drivers_heatmap', 'pro_changes_by_transition'),
        '7': ('specimen_availability',),
    }
    for slide, names in figure_stems.items():
        requests += _group('figures', STAGE05, slide, [f'05_pharma_{name}.pdf' for name in names],
                           required=slide not in {'7'})
    requests += _group('figures', STAGE05, '4', ['05_pharma_converter_profiles.pdf'])
    return [Requested(QC_MANIFEST, 'all', True), *requests]


def _read_canonical(root: Path, name: str) -> bytes:
    """Open each path component relative to a directory descriptor without following links."""
    path = PurePosixPath(name)
    if (path.is_absolute() or path.parts[0] != 'outputs' or '..' in path.parts or
            set(part.casefold() for part in path.parts) & BLOCKED_PARTS):
        raise ValueError('path_is_outside_public_output_contract')
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[:-1]:
            cursor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = cursor
        leaf = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(leaf, 'rb') as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError('artifact_is_not_a_regular_file')
            return handle.read()
    finally:
        os.close(fd)


def _csv_review(content: bytes) -> str:
    """Check identifiers and record minimum positive count for institutional review."""
    reader = csv.DictReader(StringIO(content.decode('utf-8-sig')))
    if not reader.fieldnames:
        raise ValueError('csv_has_no_header')
    if any(SENSITIVE_HEADER.search(name.strip()) for name in reader.fieldnames):
        raise ValueError('csv_has_individual_identifier_or_date_column')
    columns = [name for name in reader.fieldnames if name.lower().startswith(('n_', 'count', 'number_of')) or
               name.lower() in {'n', 'events', 'patients', 'denominator', 'numerator'}]
    minimum = None
    for row in reader:
        if None in row:
            raise ValueError('csv_has_inconsistent_columns')
        if any(IDENTIFIER_TOKEN.fullmatch(value or '') for value in row.values()):
            raise ValueError('csv_has_possible_direct_identifier')
        for name in columns:
            try:
                count = float(row[name])
            except (TypeError, ValueError):
                continue
            if count > 0 and count.is_integer():
                minimum = count if minimum is None else min(minimum, count)
    return str(int(minimum)) if minimum is not None else ''


def _validate_artifact(path, content) -> str:
    if path.endswith('.csv'):
        return _csv_review(content)
    if path.endswith('.pdf'):
        if not content.startswith(b'%PDF-'):
            raise ValueError('invalid_pdf_signature')
        return ''
    if path.endswith('.json'):
        parsed = json.loads(content)
        def check_json(value):
            if isinstance(value, dict):
                if any(SENSITIVE_HEADER.search(str(key)) for key in value):
                    raise ValueError('json_has_individual_identifier_or_date_field')
                for item in value.values():
                    check_json(item)
            elif isinstance(value, list):
                for item in value:
                    check_json(item)
        check_json(parsed)
        return ''
    raise ValueError('artifact_type_not_permitted')


def _stage_manifest(root):
    try:
        content = _read_canonical(root, QC_MANIFEST)
        manifest = json.loads(content)
        if manifest.get('schema') != SCHEMA or manifest.get('producer') != PRODUCER:
            raise ValueError('unrecognized_stage05_manifest')
        paths = defaultdict(list)
        for entry in manifest['artifacts']:
            paths[entry['path']].append(entry)
        return paths, ''
    except FileNotFoundError:
        return {}, 'stage05_manifest_missing'
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        return {}, f'stage05_manifest_incompatible:{type(exc).__name__}'


def _classify(root, request, stage_manifest, manifest_error):
    path = request.path
    record = dict(slide=request.slide, required='yes' if request.required else 'no', path=path,
                  status='', included='no', sha256='', bytes='', minimum_positive_count='', reason='')
    def result(status, reason='', content=None, minimum=''):
        record.update(status=status, reason=reason)
        if content is not None:
            record.update(included='yes', sha256=sha256(content).hexdigest(), bytes=len(content),
                          minimum_positive_count=minimum)
        return record, content

    expected = None
    if f'/{STAGE05}/' in path and path != QC_MANIFEST and not manifest_error:
        entries = stage_manifest.get(path, [])
        if len(entries) > 1:
            return result('MULTIPLE_REVIEW', 'multiple_records_for_canonical_path')
        if len(entries) == 1:
            expected = entries[0]
            if (expected.get('producer') != PRODUCER or expected.get('schema') != SCHEMA or
                    expected.get('status') not in {'estimated', 'not_estimable'}):
                return result('STALE_OR_INCOMPATIBLE', 'manifest_artifact_schema_or_status_unrecognized')
    try:
        content = _read_canonical(root, path)
    except FileNotFoundError:
        if expected and expected.get('status') == 'not_estimable' and expected.get('sha256') is None:
            return result('NOT_ESTIMABLE', expected.get('reason', ''))
        if expected and expected.get('sha256'):
            return result('STALE_OR_INCOMPATIBLE', 'manifest_declares_an_artifact_that_is_missing')
        return result('MISSING_REQUIRED' if request.required else 'OPTIONAL_MISSING',
                      'canonical_file_absent')
    except (ValueError, OSError):
        return result('STALE_OR_INCOMPATIBLE', 'canonical_path_is_unsafe_or_unreadable')
    if path == QC_MANIFEST and manifest_error:
        return result('STALE_OR_INCOMPATIBLE', manifest_error)
    if f'/{STAGE05}/' in path and path != QC_MANIFEST:
        if not expected:
            return result('STALE_OR_INCOMPATIBLE', manifest_error or 'artifact_missing_from_recognized_manifest')
        if expected.get('sha256') != sha256(content).hexdigest():
            return result('STALE_OR_INCOMPATIBLE', 'manifest_hash_disagrees')
    try:
        minimum = _validate_artifact(path, content)
    except (ValueError, UnicodeError, csv.Error):
        return result('STALE_OR_INCOMPATIBLE', 'aggregate_content_validation_failed')
    if expected and expected['status'] == 'not_estimable':
        return result('NOT_ESTIMABLE', expected.get('reason', ''), content, minimum)
    return result('FOUND', content=content, minimum=minimum)


def collect(root: Path) -> tuple[Path, dict]:
    """Replace the deck only after confirming complete ZIP contents and hashes."""
    root = root.resolve()
    outputs = root / 'outputs'
    if not outputs.exists() and not outputs.is_symlink():
        outputs.mkdir()
    destination = root / ARCHIVE
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    lock = os.open(outputs, flags)
    try:
        fcntl.flock(lock, fcntl.LOCK_SH)
        stage, error = _stage_manifest(root)
        rows = []
        with NamedTemporaryFile(prefix='.pharma-deck-', suffix='.zip', dir=root, delete=False) as temp:
            candidate = Path(temp.name)
        try:
            with ZipFile(candidate, 'w', compression=ZIP_DEFLATED) as archive:
                requests = allowlist()
                if len({request.path for request in requests}) != len(requests):
                    raise ValueError('collector_allowlist_has_duplicates')
                for request in requests:
                    record, content = _classify(root, request, stage, error)
                    rows.append(record)
                    if content is not None:
                        archive.writestr(request.path, content)
                summary = Counter(row['status'] for row in rows)
                manifest = StringIO(newline='')
                writer = csv.DictWriter(manifest, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                archive.writestr('MANIFEST.csv', manifest.getvalue())
                archive.writestr('README.md', '\n'.join([
                    '# Pharma deck outputs',
                    '',
                    'Internal aggregate package. Source paths and SHA-256 hashes are in MANIFEST.csv.',
                    'The minimum positive count is a review aid. Institutional disclosure review is required before external sharing.',
                    'PDFs and JSON still require content review. This archive does not authorize sponsor distribution.',
                    'The baseline distribution visual is the aggregated Pharma 05 figure.',
                    '',
                    *[f'- {status} — {count}' for status, count in sorted(summary.items())], '',
                ]))
                archive.writestr('PENDING.md', '\n'.join([
                    '# Pending and unavailable outputs', '',
                    *[f"- {r['status']} · slide {r['slide']} · `{r['path']}` · {r['reason']}"
                      for r in rows if r['status'] != 'FOUND'], '',
                ]))
            with ZipFile(candidate) as archive:
                expected = {row['path']: row['sha256'] for row in rows if row['included'] == 'yes'}
                wanted = {*expected, 'README.md', 'PENDING.md', 'MANIFEST.csv'}
                if (len(archive.namelist()) != len(wanted) or set(archive.namelist()) != wanted or
                        archive.testzip() is not None or
                        any(sha256(archive.read(path)).hexdigest() != digest for path, digest in expected.items())):
                    raise ValueError('new_archive_failed_integrity_check')
            os.replace(candidate, destination)
        finally:
            candidate.unlink(missing_ok=True)
    finally:
        os.close(lock)
    return destination, dict(summary)


if __name__ == '__main__':
    path, statuses = collect(Path(__file__).resolve().parents[4])
    print(f'{path.name} — {statuses}')

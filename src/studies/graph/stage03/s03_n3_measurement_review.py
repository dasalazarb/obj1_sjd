"""Verify externally signed source adjudication and apply its exact scope."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import re
import subprocess

import pandas as pd
import yaml

from .._pipeline import digest
from .s03_n3_approved_view import measurement_values
from .s03_n3_decisions import evidence_path
from .s03_n3_measurement_source import CORE, EDA_FIELDS, KEYS, LAB_SOURCES, UNAVAILABLE_RESULT, _text, _utc

REVIEW_VERSION = 'graph03_n3_measurement_review_v1'
AFFIRMATIONS = (
    'official_episode_linkage_verified', 'assay_identity_verified', 'sample_identity_verified',
    'source_unit_and_scale_verified', 'raw_result_field_verified', 'instrument_time_semantics_verified',
    'baseline_cutoff_utc_verified', 'unavailable_results_reviewed',
)
# A configured candidate source may reveal a specimen field for audit. The
# signed source review must still verify that field's identity semantics.
SOURCE_FIELD = re.compile(r'[a-z][a-z0-9_]*', re.I)
REPRESENTATION_FIELDS = ['result_raw', 'result_text', 'result_numeric_exact', 'result_operator',
                         'result_numeric_bound', 'reported_interpretation', 'result_status']


def _identity(value):
    return re.sub(r'[^a-z0-9]', '', value.casefold()) if isinstance(value, str) else ''


def validate_specimen_column(field):
    if field is None:
        return
    if (not isinstance(field, str) or not SOURCE_FIELD.fullmatch(field)
            or field.casefold() in {name.casefold() for name in EDA_FIELDS}
            or set(field.casefold().split('_')) & {'order', 'patient', 'episode', 'row', 'record', 'assay',
                                                   'time', 'date', 'datetime', 'timestamp'}):
        raise ValueError('SPECIMEN_KEY_MUST_BE_A_SEPARATE_ORIGINAL_FIELD')


def _verify_document(documents, binding):
    review_file, signature, key = (documents[name] for name in ('review', 'signature', 'trust_key'))
    for path in documents.values():
        if not path.is_file() or not 0 < path.stat().st_size <= 2_000_000:
            raise ValueError('SOURCE_REVIEW_DOCUMENT_UNAVAILABLE')
    if key.stat().st_mode & 0o022:
        raise ValueError('UNSAFE_SOURCE_REVIEW_TRUST_KEY')
    hashes = {name: digest(path) for name, path in documents.items()}
    payload = review_file.read_bytes()
    if hashlib.sha256(payload).hexdigest() != hashes['review']:
        raise ValueError('SOURCE_REVIEW_DOCUMENT_CHANGED')
    result = subprocess.run(
        ['openssl', 'dgst', '-sha256', '-verify', str(key), '-signature', str(signature)],
        input=payload, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode != 0:
        raise ValueError('SOURCE_REVIEW_SIGNATURE_INVALID')
    document = yaml.safe_load(payload)
    if (not isinstance(document, dict) or document.get('schema_version') != REVIEW_VERSION
            or document.get('decision') != 'APPROVED_SOURCE_PROVENANCE'
            or any(not isinstance(document.get(key), str) or not document[key].strip()
                   for key in ('reviewer', 'review_id'))
            or document.get('input_sha256') != binding):
        raise ValueError('SOURCE_REVIEW_SCOPE_OR_INPUT_PINS_INVALID')
    links = document.get('evidence')
    if not isinstance(links, list) or not links:
        raise ValueError('SOURCE_REVIEW_EVIDENCE_REQUIRED')
    evidence = {}
    for attachment in links:
        path = evidence_path(review_file.parent, attachment['path'])
        sha = digest(path)
        if sha != attachment.get('sha256') or path in documents.values() or attachment['path'] in evidence:
            raise ValueError('SOURCE_REVIEW_EVIDENCE_INVALID')
        evidence[attachment['path']] = sha
    return document, hashes, evidence


def _finite(value):
    try:
        return Decimal(value).is_finite()
    except (InvalidOperation, TypeError, ValueError):
        return False


def _check_result_type(rows, values, review):
    if review.get('type') == 'numeric':
        if rows.censored.any() or not values.map(_finite).all():
            raise ValueError('REVIEWED_NUMERIC_VALUE_CENSORED_OR_UNPARSEABLE')
        if not rows.eda_result_valid.all():
            raise ValueError('REVIEWED_EDA_RESULT_VALIDITY_UNRESOLVED')
    else:
        if rows.censored.any():
            raise ValueError('REVIEWED_QUALITATIVE_VALUE_CENSORED')
        nonresult = values.str.fullmatch(UNAVAILABLE_RESULT, na=False)
        # Administrative tokens can only be mapped to missingness.
        conventions = review.get('category_convention', {})
        nonresult_categories = set(conventions.get('unknown', [])) | set(conventions.get('not_tested', []))
        if not set(values[nonresult]) <= nonresult_categories:
            raise ValueError('REVIEWED_ADMINISTRATIVE_TOKEN_NOT_MISSING')
        if (~rows.eda_result_valid & ~nonresult).any():
            raise ValueError('REVIEWED_EDA_RESULT_VALIDITY_UNRESOLVED')


def _build_outcome(candidates, baseline, source, rule, specimen_id_column):
    if (not isinstance(rule, dict) or not all(rule.get(flag) is True for flag in AFFIRMATIONS)
            or not specimen_id_column or rule.get('sample_field') != specimen_id_column
            or rule.get('assay_field') != 'assay' or rule.get('date_field') != 'lab_date'
            or rule.get('unit_field') != 'unit'):
        raise ValueError('SOURCE_REVIEW_FIELD_SEMANTICS_REQUIRED')
    inspection = rule.get('measurement_review')
    if not isinstance(inspection, dict):
        raise ValueError('REVIEWED_MEASUREMENT_CONTRACT_REQUIRED')
    assay = rule.get('original_assay')
    reviewed_assay = inspection.get('assay_id')
    aliases = set(LAB_SOURCES) | set(LAB_SOURCES.values()) | {'ANA', 'RF', 'HEp-2', 'urine protein'}
    if any(not _identity(value) or _identity(value) in {_identity(alias) for alias in aliases}
           for value in (assay, reviewed_assay)):
        raise ValueError('VERIFIED_ASSAY_IDENTITY_REQUIRED')
    value_field = rule.get('value_field')
    kind = 'numeric' if source.endswith('__value') else 'binary'
    if (inspection.get('type') != kind
            or value_field not in ({'result_numeric_exact'} if kind == 'numeric' else {'result_text', 'result_raw'})):
        raise ValueError('REVIEWED_SOURCE_VALUE_FIELD_OR_TYPE_INVALID')
    lookback = inspection.get('lookback_days')
    unit = inspection.get('unit')
    if type(lookback) is not int or lookback <= 0 or not isinstance(unit, str) or not unit.strip():
        raise ValueError('REVIEWED_UNIT_OR_WINDOW_REQUIRED')
    rows = candidates.loc[candidates.source_column.eq(source)].copy()
    if rows.empty or not rows.mapping_candidate.all():
        raise ValueError('REVIEWED_ORIGINAL_OBSERVATIONS_UNAVAILABLE_OR_MAPPING_INVALID')
    if (rows.step01_context_date_status.eq('STEP01_STEP10_DATE_DISCORDANT').any()
            or rows.step01_step10_value_status.eq('STEP01_STEP10_VALUE_DISCORDANT').any()):
        raise ValueError('REVIEWED_STEP01_STEP10_LINEAGE_DISCORDANT')
    if (not _text(rows.assay).eq(assay).fillna(False).all()
            or not _text(rows.unit).eq(unit).fillna(False).all()):
        raise ValueError('REVIEWED_NON_EQUIVALENT_ASSAY_OR_UNIT')
    if (rows.original_sample_key.isna().any()
            or any(rows.original_sample_key.eq(_text(rows[field])).fillna(False).any()
                   for field in ('order_identifier', 'specimen_datetime', '_lab_record_id', 'eda_row_offset',
                                 'patient_id', 'clinical_episode_id', 'eda_patient_id', 'assay'))
            or rows.groupby('original_sample_key').patient_id.nunique().gt(1).any()):
        raise ValueError('REVIEWED_SAMPLE_IDENTITY_UNVERIFIED')
    valid_times = rows.temporal_status.isin({
        'PREBASELINE_OFFSET_CANDIDATE', 'POST_BASELINE_OFFSET_COMPARISON',
    })
    if not valid_times.all():
        raise ValueError('REVIEWED_TIME_PRECISION_OR_TIMEZONE_INSUFFICIENT')
    age = (_utc(rows.clinical_baseline_date) - rows.instrument_utc_with_reported_offset).dt.total_seconds() / 86400
    before = rows.temporal_status.eq('PREBASELINE_OFFSET_CANDIDATE')
    retained = before & age.between(0, lookback)
    linkage = rows[['eda_row_offset', 'source_column']].copy()
    linkage['reviewed_disposition'] = 'AFTER_BASELINE'
    linkage.loc[before, 'reviewed_disposition'] = 'OUTSIDE_REVIEWED_LOOKBACK'
    linkage.loc[retained, 'reviewed_disposition'] = 'RETAINED_FOR_03E_SELECTION'
    rows = rows.loc[retained]
    if rows.empty:
        raise ValueError('REVIEWED_NO_PREBASELINE_OBSERVATIONS_IN_WINDOW')
    values = _text(rows[value_field])
    if values.isna().any():
        raise ValueError('REVIEWED_SOURCE_VALUE_MISSING')
    _check_result_type(rows, values, inspection)

    output = rows[KEYS + ['source_column']].assign(
        value=values, measurement_date=rows.instrument_utc_with_reported_offset,
        assay_id=reviewed_assay, unit=unit, sample_id=rows.original_sample_key,
    )
    output = output.astype({key: baseline[key].dtype for key in KEYS})
    latest_time = output.groupby(KEYS).measurement_date.transform('max')
    latest = output.loc[output.measurement_date.eq(latest_time)]
    variants = pd.concat([latest, rows.loc[latest.index, REPRESENTATION_FIELDS]], axis=1).drop_duplicates()
    if variants.duplicated(KEYS).any():
        raise ValueError('REVIEWED_REPEAT_TIE_REQUIRES_ADJUDICATION')
    # Exercise the same contract that 03e will apply to this specific artifact.
    measurement_values(baseline, output, source, inspection)
    return output[CORE], linkage, inspection, assay


def reviewed_measurements(candidates, ledger, baseline, binding, specimen_id_column, documents):
    """Consume a signature from an institutional trust anchor supplied out of band."""
    document, hashes, evidence = _verify_document(documents, binding)
    rules = document.get('outcomes')
    if not isinstance(rules, dict) or not rules or not set(rules) <= set(LAB_SOURCES):
        raise ValueError('SOURCE_REVIEW_OUTCOME_SCOPE_INVALID')
    output, row_links, windows, ana_assays = [], [], {}, []
    offset = 0
    for source, rule in sorted(rules.items()):
        if ledger.loc[ledger.canonical_analyte.eq(LAB_SOURCES[source]), 'baseline_linkage_conflict'].any():
            raise ValueError('REVIEWED_EDA_EPISODE_LINKAGE_UNRESOLVED')
        measurements, linkage, inspection, original_assay = _build_outcome(
            candidates, baseline, source, rule, specimen_id_column)
        linkage['reviewed_row_offset'] = pd.Series(range(offset, offset + len(measurements)),
                                                  index=measurements.index, dtype='Int64')
        offset += len(measurements)
        output.append(measurements)
        row_links.append(linkage)
        windows[source] = inspection['lookback_days']
        if LAB_SOURCES[source].startswith('ana'):
            ana_assays.append((_identity(original_assay), _identity(inspection['assay_id'])))
    if len(ana_assays) == 2 and any(left == right for left, right in zip(*ana_assays)):
        raise ValueError('REVIEWED_ANA_OUTCOMES_SHARE_ORIGINAL_ASSAY')
    combined = pd.concat(output, ignore_index=True)
    if combined.groupby('sample_id').patient_id.nunique().gt(1).any():
        raise ValueError('REVIEWED_SAMPLE_IDENTITY_UNVERIFIED')
    # The route emits observations only. Representation-specific decisions,
    # research protocol and support stay within 03e.
    return combined, pd.concat(row_links, ignore_index=True), dict(
        sources=sorted(windows), lookback_days_by_source=windows, document_sha256=hashes, evidence_sha256=evidence)

"""Verify externally signed source adjudication and apply its exact scope."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import re

import pandas as pd
import yaml

from .._pipeline import digest
from . import s03_n3_calendar_day as cal
from .s03_n3_approved_view import measurement_values, measurement_values_calendar_day, validate_calendar_day_review
from .s03_n3_decisions import evidence_path, verify_detached_signature
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


def _verify_document(documents, binding, version):
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
    if not verify_detached_signature(payload, signature, key):
        raise ValueError('SOURCE_REVIEW_SIGNATURE_INVALID')
    document = yaml.safe_load(payload)
    if (not isinstance(document, dict) or document.get('schema_version') != version
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


CALENDAR_AFFIRMATIONS = ('official_episode_linkage_verified', 'raw_result_field_verified',
                         'btris_eda_linkage_and_step20_truncation_reviewed',
                         'comparable_test_crosswalk_reviewed', 'unavailable_results_reviewed')
TEST_FIELDS = ['original_order_name', 'original_observation_name', 'assay']


def _calendar_rule(source, rule, window):
    if (not isinstance(rule, dict) or not all(rule.get(flag) is True for flag in CALENDAR_AFFIRMATIONS)
            or rule.get('unit_field') != 'unit' or rule.get('date_field') != 'lab_date'
            or any(flag in rule for flag in (
                'assay_identity_verified', 'sample_identity_verified', 'sample_field',
                'assay_field', 'baseline_cutoff_utc_verified', 'instrument_time_semantics_verified'))):
        raise ValueError('CALENDAR_DAY_SOURCE_REVIEW_FIELD_SEMANTICS_REQUIRED')
    inspection = rule.get('measurement_review')
    groups = validate_calendar_day_review(inspection)
    if any(inspection.get(key) != window[key] for key in cal.SELECTOR_FIELDS):
        raise ValueError('CALENDAR_DAY_WINDOW_OUTSIDE_PRESPECIFIED_SCOPE')
    value_field = rule.get('value_field')
    kind = 'numeric' if source.endswith('__value') else 'binary'
    if (inspection['type'] != kind or value_field not in (
            {'result_numeric_exact'} if kind == 'numeric' else {'result_text', 'result_raw'})):
        raise ValueError('REVIEWED_SOURCE_VALUE_FIELD_OR_TYPE_INVALID')
    definitions = rule.get('comparable_original_tests')
    if (not isinstance(definitions, list) or not definitions or any(
            not isinstance(test, dict) or set(test) != set(TEST_FIELDS)
            or any(not isinstance(test[key], str) or not test[key].strip() for key in TEST_FIELDS[:2])
            or (test['assay'] is not None and (not isinstance(test['assay'], str) or not test['assay'].strip()))
            for test in definitions)):
        raise ValueError('CALENDAR_DAY_REVIEWED_CROSSWALK_REQUIRED')
    return inspection, groups, pd.DataFrame(definitions, columns=TEST_FIELDS).astype('string').drop_duplicates()


def _build_calendar_outcome(candidates, baseline, source, rule, window, source_sha):
    inspection, groups, definitions = _calendar_rule(source, rule, window)
    rows = candidates.loc[candidates.source_column.eq(source)].copy()
    if rows.empty:
        raise ValueError('REVIEWED_ORIGINAL_OBSERVATIONS_UNAVAILABLE_OR_MAPPING_INVALID')
    signatures = rows[TEST_FIELDS].astype('string')
    comparable = pd.MultiIndex.from_frame(signatures).isin(pd.MultiIndex.from_frame(definitions))
    comparable &= (rows.unit.eq(inspection['unit']).fillna(False) & rows.mapping_candidate).to_numpy(dtype=bool)
    linkage = rows[['eda_row_offset', 'source_column', 'source_episode_id', 'btris_source',
                    'calendar_date_status']].copy()
    linkage['test_definition_verified'] = comparable & rows.collection_calendar_day.notna()
    linkage['reviewed_disposition'] = rows.calendar_disposition
    latest = rows.calendar_disposition.eq('LATEST_ELIGIBLE_DAY')
    rows['value'] = _text(rows[rule['value_field']])
    administrative = rows[['result_raw', 'result_text', 'result_status', 'reported_interpretation']].apply(
        lambda column: _text(column).str.fullmatch(cal.ADMINISTRATIVE_RESULT, na=False)).any(axis=1)
    if groups is None:
        result_valid = rows.value.map(_finite) & rows.eda_result_valid & ~administrative
    else:
        nonresult = rows.value.isin(groups['unknown'] | groups['not_tested'])
        result_valid = rows.value.isin(set().union(*groups.values())) & (rows.eda_result_valid | nonresult)
        result_valid &= ~administrative | nonresult
    clean = result_valid & ~rows.censored
    linkage.loc[latest & ~comparable, 'reviewed_disposition'] = 'TEST_DEFINITION_OR_UNIT_OUTSIDE_REVIEWED_SCOPE'
    linkage.loc[latest & comparable & ~clean, 'reviewed_disposition'] = 'UNUSABLE_OR_UNREVIEWED_RESULT'
    retained = latest & comparable & clean
    linkage.loc[retained, 'reviewed_disposition'] = 'RETAINED_FOR_03E_SELECTION'
    selected = rows.loc[retained].drop_duplicates(KEYS + cal.OBSERVATION_FIELDS)
    linkage.loc[retained & ~linkage.index.isin(selected.index), 'reviewed_disposition'] = 'IDENTICAL_DUPLICATE_COLLAPSED'
    fingerprints = selected[cal.OBSERVATION_FIELDS].astype('string').to_dict('records')
    output = selected.assign(
        test_definition_id=inspection['test_definition_id'], review_contract_version=cal.REVIEW_VERSION,
        source_review_sha256=source_sha, measurement_contract_sha256=cal.fingerprint(inspection),
        observation_sha256=[cal.fingerprint({key: value if pd.notna(value) else None for key, value in record.items()})
                            for record in fingerprints],
        source_artifact_status='SOURCE_REVIEWED',
    )[cal.CORE].astype({key: baseline[key].dtype for key in KEYS})
    if not output.empty:
        measurement_values_calendar_day(baseline, output, source, inspection, source_review_sha256=source_sha)
    return output, linkage, inspection, set(definitions.fillna('').itertuples(index=False, name=None))


def reviewed_measurements(candidates, ledger, baseline, binding, specimen_id_column, documents, *, calendar=None):
    """Consume a signature from an institutional trust anchor supplied out of band."""
    version = REVIEW_VERSION if calendar is None else cal.REVIEW_VERSION
    document, hashes, evidence = _verify_document(documents, binding, version)
    rules = document.get('outcomes')
    if not isinstance(rules, dict) or not rules or not set(rules) <= set(LAB_SOURCES):
        raise ValueError('SOURCE_REVIEW_OUTCOME_SCOPE_INVALID')
    output, row_links, windows, ana_assays = [], [], {}, []
    offset = 0
    for source, rule in sorted(rules.items()):
        if calendar is None:
            if ledger.loc[ledger.canonical_analyte.eq(LAB_SOURCES[source]), 'baseline_linkage_conflict'].any():
                raise ValueError('REVIEWED_EDA_EPISODE_LINKAGE_UNRESOLVED')
            measurements, linkage, inspection, original_assay = _build_outcome(
                candidates, baseline, source, rule, specimen_id_column)
            ana_identity = (_identity(original_assay), _identity(inspection['assay_id']))
        else:
            if source not in calendar['windows']:
                raise ValueError('SOURCE_REVIEW_OUTCOME_SCOPE_INVALID')
            measurements, linkage, inspection, definitions = _build_calendar_outcome(
                candidates, baseline, source, rule, calendar['windows'][source], hashes['review'])
            ana_identity = (definitions, inspection['test_definition_id'])
        linkage['reviewed_row_offset'] = pd.Series(range(offset, offset + len(measurements)),
                                                  index=measurements.index, dtype='Int64')
        if calendar is not None:
            excluded = ledger.loc[ledger.canonical_analyte.eq(LAB_SOURCES[source]) &
                                  ~ledger.eda_row_offset.isin(linkage.eda_row_offset), [
                                      'eda_row_offset', 'source_episode_id', 'btris_source',
                                      'calendar_date_status', 'identity_status']]
            if not excluded.empty:
                excluded = excluded.assign(source_column=source, test_definition_verified=False,
                                           reviewed_disposition=excluded.identity_status)
                linkage = pd.concat([linkage, excluded.drop(columns='identity_status')], ignore_index=True)
        offset += len(measurements)
        output.append(measurements)
        row_links.append(linkage)
        windows[source] = inspection['lookback_days']
        if LAB_SOURCES[source].startswith('ana'):
            ana_assays.append(ana_identity)
    if len(ana_assays) == 2:
        (original_a, reviewed_a), (original_b, reviewed_b) = ana_assays
        if calendar is None and (original_a == original_b or reviewed_a == reviewed_b):
            raise ValueError('REVIEWED_ANA_OUTCOMES_SHARE_ORIGINAL_ASSAY')
        if calendar is not None and (document.get('ana_hypotheses_distinct_verified') is not True
                                    or original_a & original_b or reviewed_a == reviewed_b):
            raise ValueError('REVIEWED_ANA_DEFINITIONS_NOT_DISTINCT')
    combined = pd.concat(output, ignore_index=True)
    if combined.groupby('sample_id').patient_id.nunique().gt(1).any():
        raise ValueError('REVIEWED_SAMPLE_IDENTITY_UNVERIFIED')
    # 03e evaluates representation-specific decisions, research protocol and support.
    manifest = dict(
        sources=sorted(windows), lookback_days_by_source=windows, document_sha256=hashes, evidence_sha256=evidence)
    if calendar is not None:
        manifest.update(review_contract_version=version, minimum_days_before_baseline_by_source={
            source: calendar['windows'][source]['minimum_days_before_baseline'] for source in windows})
    return combined, pd.concat(row_links, ignore_index=True), manifest

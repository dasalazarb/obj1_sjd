"""Calendar-day provenance and selection for the optional N3 lab contract."""
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import pyarrow as pa

from .._pipeline import digest
from .s03_n3_source_adapter import KEYS, LAB_SOURCES

CONTRACT = 'calendar_day_prebaseline'
SOURCE_VERSION = 'graph03_n3_measurement_source_v2'
REVIEW_VERSION = 'graph03_n3_measurement_review_v2'
IDENTITY = 'reviewed_test_definition_without_specimen_id'
DATE_ORIGIN = 'BTRIS_COLLECTED_DATE_TIME_RAW_INTERMEDIATE_EDA_LAB_DATE'
BTRIS_ORIGINS = ('raw', 'intermediate')
BTRIS_PROTOCOLS = ('11d', '15d')
BTRIS_ROLES = tuple(f'btris_{origin}_{protocol}' for origin in BTRIS_ORIGINS for protocol in BTRIS_PROTOCOLS)
ORIGINAL_IDENTITY = ['original_order_id', 'original_order_name', 'original_observation_name']
CORE = KEYS + ['source_episode_id', 'source_column', 'value', 'collection_calendar_day',
               'test_definition_id', 'unit', 'assay_id', 'sample_id', *ORIGINAL_IDENTITY,
               'date_origin', 'date_resolution', 'review_contract_version', 'source_review_sha256',
               'measurement_contract_sha256', 'observation_sha256', 'source_artifact_status']
KEY_FIELDS = ('patient_id', *ORIGINAL_IDENTITY)
SELECTOR_FIELDS = ('lookback_days', 'minimum_days_before_baseline')
ISO_DATETIME = re.compile(
    r'\d{4}-\d{2}-\d{2}[T ](?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d(?:\.\d{1,9})?)?'
    r'(?:Z|[+-](?:[01]\d|2[0-3]):?[0-5]\d)?')
ISO_DAY = re.compile(r'\d{4}-\d{2}-\d{2}')
DATE_DTYPE = pd.ArrowDtype(pa.date32())
SHA256 = re.compile(r'[a-f0-9]{64}')
UNAVAILABLE_RESULT = (r'(?i)\s*(?:unknown|not[ _-](?:tested|available|done|performed)|'
                      r'unavailable|pending|cancelled|canceled|insufficient[ _-](?:specimen|sample))\s*')
ADMINISTRATIVE_RESULT = UNAVAILABLE_RESULT.replace('[ _-]', '[ _-]+')
RESULT_FIELDS = ['result_raw', 'result_text', 'result_numeric_exact', 'result_operator',
                 'result_numeric_bound', 'reported_interpretation', 'result_status']
OBSERVATION_FIELDS = [*ORIGINAL_IDENTITY, 'source_episode_id', 'btris_source', 'lab_date',
                      'specimen_datetime', 'record_id', 'assay', 'unit', 'value',
                      'mapping_candidate', 'eda_result_valid', 'censored', *RESULT_FIELDS]


def calendar_day(value, date_format='ISO8601', *, allow_time=False):
    """Read the date written by the source without shifting it between zones."""
    try:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, str):
            if ISO_DAY.fullmatch(value) and date_format == 'ISO8601':
                return date.fromisoformat(value)
            if allow_time and date_format == 'ISO8601' and ISO_DATETIME.fullmatch(value):
                return date.fromisoformat(value[:10])
            if allow_time and date_format != 'ISO8601':
                return datetime.strptime(value, date_format).date()
    except ValueError:
        pass
    return None


def calendar_days(series, date_format='ISO8601', *, allow_time=False):
    if allow_time:
        series = series.astype('string')
    dates = series.map(lambda value: calendar_day(value, date_format, allow_time=allow_time))
    return pd.Series(pd.array(dates, dtype=DATE_DTYPE), index=series.index)


def validate_settings(settings):
    if (not isinstance(settings, dict) or settings.get('contract') != CONTRACT
            or not isinstance(settings.get('windows'), dict) or not settings['windows']
            or not set(settings['windows']) <= set(LAB_SOURCES)):
        raise ValueError('CALENDAR_DAY_CONFIG_INVALID')
    for window in settings['windows'].values():
        if not isinstance(window, dict) or set(window) != set(SELECTOR_FIELDS):
            raise ValueError('CALENDAR_DAY_WINDOW_REQUIRED')
        validate_window(window)
    fields = settings.get('linkage_fields')
    if not isinstance(fields, dict) or set(fields) not in (set(KEY_FIELDS), set(KEY_FIELDS) | {'record_id'}):
        raise ValueError('BTRIS_LINKAGE_FIELDS_REQUIRED')
    for origin in ('eda', *BTRIS_ORIGINS):
        names = [value.get(origin) for value in fields.values() if isinstance(value, dict)]
        if (len(names) != len(fields) or any(not isinstance(name, str) or not name.strip() or
                                           name == 'Collected Date Time' for name in names)
                or len(set(names)) != len(fields)):
            raise ValueError('BTRIS_LINKAGE_FIELDS_REQUIRED')
    if any(fields[field]['eda'] != name for field, name in zip(
            KEY_FIELDS[:3], ('patient_id', 'order_identifier', 'order_name_original'))):
        raise ValueError('BTRIS_ORIGINAL_EDA_IDENTITY_FIELDS_REQUIRED')
    if (fields['original_observation_name']['eda'] not in {'observation_name_original', 'cluster_name_original'}
            or any(fields[field]['raw'] != name for field, name in zip(
                ORIGINAL_IDENTITY, ('Order ID', 'Order Name', 'Observation Name')))):
        raise ValueError('BTRIS_ORIGINAL_ORDER_OBSERVATION_FIELDS_REQUIRED')
    settings = dict(settings)
    date_formats = settings.get('date_formats', {})
    if (not isinstance(date_formats, dict) or set(date_formats) - set(BTRIS_ORIGINS)
            or any(not isinstance(fmt, str) or (fmt != 'ISO8601' and
                   not all(part in fmt for part in ('%Y', '%m', '%d')))
                   for fmt in date_formats.values())):
        raise ValueError('BTRIS_DATE_FORMAT_INVALID')
    return settings


def validate_window(window):
    lookback, minimum = (window.get(key) for key in SELECTOR_FIELDS)
    if type(minimum) is not int or minimum not in {1, 2} or type(lookback) is not int or lookback < minimum:
        raise ValueError('CALENDAR_DAY_WINDOW_REQUIRED')


def calendar_age(baseline_days, collection_days):
    return (baseline_days - collection_days).dt.days


def latest_calendar_rows(rows, identity):
    """Return latest rows and a flag for each latest-day row in a discordant group."""
    latest = rows.loc[rows.collection_calendar_day.eq(
        rows.groupby(KEYS).collection_calendar_day.transform('max'))].copy()
    variants = latest.drop_duplicates(KEYS + identity)
    ambiguous = pd.MultiIndex.from_frame(variants.loc[variants.duplicated(KEYS, keep=False), KEYS])
    return latest, pd.Series(pd.MultiIndex.from_frame(latest[KEYS]).isin(ambiguous), index=latest.index)


def calendar_dispositions(rows, window):
    """Classify every row before applying the signed test definition and unit."""
    age = rows.calendar_age_days
    included = age.between(window['minimum_days_before_baseline'], window['lookback_days'])
    result = rows.temporal_status.copy()
    result.loc[age.gt(0)] = 'OUTSIDE_REVIEWED_LOOKBACK'
    result.loc[included] = 'EARLIER_ELIGIBLE_DAY'
    # A BTRIS key ambiguous on the putative latest EDA day cannot make the
    # other result on that day safe, or cause selection of an older day.
    uncertain = rows.calendar_date_status.eq('BTRIS_ORIGINAL_IDENTITY_AMBIGUOUS')
    cutoff = calendar_days(rows.clinical_baseline_date, allow_time=True)
    uncertain &= calendar_age(cutoff, rows.eda_calendar_day).between(
        window['minimum_days_before_baseline'], window['lookback_days'])
    pool = rows.loc[included | uncertain].copy()
    pool['collection_calendar_day'] = pool.collection_calendar_day.fillna(pool.eda_calendar_day)
    latest, ambiguous = latest_calendar_rows(pool, OBSERVATION_FIELDS)
    unclear = pd.MultiIndex.from_frame(latest.loc[uncertain.reindex(latest.index), KEYS])
    ambiguous |= pd.MultiIndex.from_frame(latest[KEYS]).isin(unclear)
    result.loc[latest.index] = 'LATEST_ELIGIBLE_DAY'
    result.loc[latest.index[ambiguous]] = 'AMBIGUOUS_SAME_DAY'
    return result


def btris_files(path):
    if path.is_dir():
        return [candidate for candidate in sorted(path.glob('Lab*.csv')) if candidate.is_file()]
    return [path] if path.is_file() and path.suffix.lower() == '.csv' else []


def file_hashes(path):
    return {file.name: digest(file) for file in btris_files(path)} if path is not None else {}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def btris_columns(settings, origin):
    return {value[origin]: field for field, value in settings['linkage_fields'].items()}


def _btris_days(path, settings, origin):
    columns = btris_columns(settings, origin)
    fmt = settings.get('date_formats', {}).get(origin, 'ISO8601')
    # Deduplicate only identical original rows, even if other columns differ
    # outside the fields projected into EDA. Return only the linkage projection.
    fields = list(columns.values())
    tables = []
    for file in btris_files(path):
        frame = pd.read_csv(file, dtype='string', keep_default_na=False)
        if not {*columns, 'Collected Date Time'} <= set(frame):
            raise ValueError('BTRIS_COLLECTION_COLUMNS_REQUIRED')
        tables.append(frame)
    frame = pd.concat(tables, ignore_index=True).drop_duplicates()
    frame = frame.rename(columns=columns)
    frame = frame[fields + ['Collected Date Time']].copy()
    frame = frame.loc[frame[fields].apply(lambda s: s.notna() & s.str.strip().ne('')).all(axis=1)]
    frame['_btris_day'] = calendar_days(frame['Collected Date Time'], fmt, allow_time=True)
    # Multiple distinct raw rows can reuse an order and observation key.
    frame['_source_ambiguous'] = frame.duplicated(fields, keep=False)
    return frame.drop_duplicates(fields).drop(columns='Collected Date Time')


def btris_provenance(paths, settings):
    fields = list(settings['linkage_fields'])
    tables = []
    for protocol in BTRIS_PROTOCOLS:
        raw, intermediate = (_btris_days(paths[f'btris_{origin}_{protocol}'], settings, origin)
                             for origin in BTRIS_ORIGINS)
        table = raw.merge(intermediate, on=fields, how='outer', suffixes=('_raw', '_intermediate'),
                          validate='one_to_one')
        table['btris_source'] = protocol.upper()
        tables.append(table)
    records = pd.concat(tables, ignore_index=True)
    overlap = records.duplicated(fields, keep=False)
    records.loc[overlap, '_source_ambiguous_raw'] = True
    records.loc[overlap, 'btris_source'] = pd.NA
    return records.drop_duplicates(fields)


def attach_collection_days(rows, records, settings):
    cols = btris_columns(settings, 'eda')
    for eda_field, field in cols.items():
        rows[field if field != 'patient_id' else '_btris_patient'] = rows[eda_field].astype('string')
    keys = [field if field != 'patient_id' else '_btris_patient' for field in cols.values()]
    rows = rows.merge(records.rename(columns={'patient_id': '_btris_patient'}), on=keys,
                      how='left', validate='many_to_one')
    eda_day = calendar_days(rows.lab_date, allow_time=True)
    specimen_day = calendar_days(rows.specimen_datetime, allow_time=True)
    raw_day, intermediate_day = rows._btris_day_raw, rows._btris_day_intermediate
    ambiguous = rows._source_ambiguous_raw.eq(True) | rows._source_ambiguous_intermediate.eq(True)
    matched = eda_day.notna() & raw_day.notna() & intermediate_day.notna()
    conflict = (matched & (~raw_day.eq(eda_day) | ~raw_day.eq(intermediate_day))).fillna(False)
    conflict |= (specimen_day.notna() & eda_day.notna() & ~specimen_day.eq(eda_day)).fillna(False)
    verified = matched & ~conflict & ~ambiguous
    rows['calendar_date_status'] = 'BTRIS_COLLECTION_DAY_UNAVAILABLE'
    rows.loc[eda_day.isna(), 'calendar_date_status'] = 'EDA_COLLECTION_DAY_INVALID_OR_MISSING'
    rows.loc[conflict, 'calendar_date_status'] = 'BTRIS_EDA_COLLECTION_DAY_DISCORDANT'
    rows.loc[ambiguous, 'calendar_date_status'] = 'BTRIS_ORIGINAL_IDENTITY_AMBIGUOUS'
    rows.loc[verified, 'calendar_date_status'] = 'COLLECTION_DAY_CONCORDANT'
    rows['collection_calendar_day'] = eda_day.where(verified)
    # Keep a day solely to audit potentially ambiguous same-day source keys.
    rows['eda_calendar_day'] = eda_day
    rows['date_origin'] = pd.Series(DATE_ORIGIN, index=rows.index, dtype='string').where(verified)
    rows['date_resolution'] = pd.Series('calendar_day', index=rows.index, dtype='string').where(verified)
    if 'record_id' not in rows:
        rows['record_id'] = pd.NA
    return rows.drop(columns=['_btris_patient', '_source_ambiguous_raw', '_source_ambiguous_intermediate'])

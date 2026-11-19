"""Patient and calendar-day source reconciliation for exploratory 03e only."""

from dataclasses import dataclass
from importlib import import_module
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import s03_n3_calendar_day as cal
from .s03_n3_measurement_source import (
    BASE, EDA_FIELDS, EDA_REQUIRED, KEYS, LAB_CONTEXT, LAB_SOURCES, RESULT_LIMIT,
    SELECTED_FIELDS, _text,
)

READ_ERRORS = (OSError, ValueError, TypeError, KeyError)
STEP01 = import_module('src.block_A.01_serological_profile')
NORMALIZE = STEP01._normalize_patient_id
RF_VALUE = 'lab__rheumatoid_factor__value'
RF_TEXT = 'lab__rheumatoid_factor__text'
DEFINITION = ['order_name_original', 'cluster_name_original', 'observation_name_original', 'assay', 'unit']
ORIGINAL = 'ORIGINAL_COLLECTION_VERIFIED'
EDA_ONLY = 'EDA_DATE_ONLY'
PRE = 'PREBASELINE_PRIMARY'
DAY0 = 'BASELINE_SAME_DAY'
EARLY = 'FOLLOWUP_1_365'
LATE = 'FOLLOWUP_GT365'
OLDER = 'OLDER_THAN_365'
UNRESOLVED = 'UNATTRIBUTED_SOURCE_INVENTORY'
STRATA = (PRE, DAY0, EARLY, LATE, OLDER)
EXACT = 'EXACT_EPISODE_LINK'
PATIENT_ONLY = 'UNIQUE_PATIENT_ONLY_EPISODE_UNKNOWN'
LINKED = (EXACT, PATIENT_ONLY)
TRUE = {'true', '1', 'yes', 'y'}
FALSE = {'false', '0', 'no', 'n'}
DATE_FAILURES = {'DATE_DISCORDANT', 'SOURCE_LINK_AMBIGUOUS', 'DATE_MISSING'}
BTRIS_KEYS = ['_patient_match', 'order_identifier', 'order_name_original']
NAME_FIELDS = {
    'cluster_name_original': ('Cluster Name', 'cluster_name', 'cluster_name_original'),
    'observation_name_original': ('Observation Name', 'observation_name', 'observation_name_original'),
    'observation_identifier': ('Observation ID', 'Observation Identifier', 'observation_identifier'),
}


def _column(frame, name):
    return frame[name] if name in frame else pd.Series(pd.NA, index=frame.index, dtype='string')


def _clean(series):
    text = _text(series).str.strip()
    return text.mask(text.eq(''))


def _inputs(paths, issues):
    fields = {
        'eda_long': (EDA_REQUIRED - {'matched_clinical_episode_id', 'episode_match_ambiguous'}, EDA_FIELDS | {
            *NAME_FIELDS, 'specimen_date_origin', 'specimen_type', 'source_protocol'}),
        'step01': (set(KEYS + ['lab_id', 'selected_lab_date']), SELECTED_FIELDS),
        'step10': (set(BASE), set(BASE) | {'clinical_baseline_episode_id'}),
        'context': (set(KEYS), set(KEYS) | LAB_CONTEXT | {'clinical_anchor_date'}),
    }
    tables = {}
    for name, (required, projected) in fields.items():
        try:
            columns = set(pq.read_schema(paths[name]).names)
            if not required <= columns:
                raise ValueError('REQUIRED_COLUMNS_MISSING')
            tables[name] = pd.read_parquet(paths[name], columns=sorted(projected & columns))
        except READ_ERRORS:
            issues.append(dict(status='DATA_UNAVAILABLE', artifact=name))
    return tables


def _collisions(ids):
    keys = pd.DataFrame({'raw': _text(ids), 'normalized': NORMALIZE(ids)}).dropna().drop_duplicates()
    counts = keys.groupby('normalized').raw.nunique()
    return set(counts.index[counts.gt(1)])


def _episode_keys(table):
    return table.reindex(columns=KEYS).apply(_clean).dropna()


def _identity_tables(baseline, analytic, context, btris_collisions, eda_ids):
    """Retain an original-to-normalized crosswalk in memory only."""
    analytic = analytic.copy()
    official = baseline[BASE].copy()
    authority = pd.concat([official.patient_id, analytic.patient_id, _column(context, 'patient_id')])
    unsafe = _collisions(authority) | _collisions(eda_ids) | btris_collisions
    official['frozen_patient_id'] = official.patient_id
    official[KEYS] = _episode_keys(official).reindex(official.index)
    analytic[KEYS] = _episode_keys(analytic).reindex(analytic.index)

    # Global episode ownership includes otherwise unverified/duplicate rows.
    episodes = pd.concat([_episode_keys(table) for table in (baseline, analytic, context)])
    episodes['_patient_match'] = NORMALIZE(episodes.patient_id)
    owners = episodes.groupby('clinical_episode_id')._patient_match.agg(lambda keys: frozenset(keys.dropna()))
    duplicate_episodes = set(analytic.loc[analytic.duplicated(KEYS, keep=False), 'clinical_episode_id'])
    flagged_all = _clean(analytic.loc[analytic.is_clinical_baseline.eq(True), 'patient_id'])
    multiple_baselines = set(flagged_all.loc[flagged_all.duplicated(keep=False)])
    unsafe |= set(NORMALIZE(pd.Series(list(multiple_baselines), dtype='string')).dropna())
    if not context.empty:
        context_keys = _episode_keys(context)
        duplicate_episodes |= set(context_keys.loc[context_keys.duplicated(KEYS, keep=False), 'clinical_episode_id'])

    if 'clinical_anchor_date' in context:
        anchors = context[KEYS + ['clinical_anchor_date']].copy()
        anchors[KEYS] = _episode_keys(context).reindex(context.index)
        anchors = anchors.drop_duplicates(KEYS, keep=False)
        analytic = analytic.merge(anchors, on=KEYS, how='left', suffixes=('', '_context'), validate='many_to_one')
        context_day = _collection_days(analytic.clinical_anchor_date_context)
        agree = _collection_days(analytic.clinical_anchor_date).eq(context_day).fillna(False)
        analytic = analytic.loc[analytic.clinical_anchor_date_context.isna() | agree]

    valid = analytic.loc[analytic[KEYS].notna().all(axis=1) &
                         ~analytic.clinical_episode_id.isin(duplicate_episodes)]
    flagged = valid.loc[valid.is_clinical_baseline.eq(True)]
    if 'clinical_baseline_episode_id' in flagged:
        declared = _clean(flagged.clinical_baseline_episode_id)
        flagged = flagged.loc[declared.isna() | declared.eq(flagged.clinical_episode_id)]
    official = official.loc[~NORMALIZE(official.patient_id).isin(unsafe)].dropna(subset=KEYS)
    pairs = official.merge(flagged[BASE], on=KEYS, suffixes=('', '_step10'), validate='one_to_one')
    day = _collection_days(pairs.clinical_baseline_date)
    same = day.notna() & pairs.clinical_episode_id.map(owners).map(lambda keys: len(keys) == 1)
    for name in ('clinical_anchor_date', 'clinical_baseline_date_step10', 'clinical_anchor_date_step10'):
        same &= day.eq(_collection_days(pairs[name])).fillna(False)
    pairs = pairs.loc[same, BASE + ['frozen_patient_id']].assign(_patient_match=lambda f: NORMALIZE(f.patient_id))
    pairs = pairs.loc[pairs._patient_match.notna() & ~pairs._patient_match.isin(unsafe)]
    episode_keys = pd.MultiIndex.from_frame(
        valid.assign(_patient_match=NORMALIZE(valid.patient_id))[['_patient_match', 'clinical_episode_id']])
    return pairs, unsafe, owners, episode_keys, duplicate_episodes


def _collection_days(series):
    """Read written ISO/US calendar days without making timezone assumptions."""
    days = cal.calendar_days(series, allow_time=True)
    us = series.astype('string').str.extract(r'^(\d{1,2}/\d{1,2}/\d{4})(?:[ T].*)?$', expand=False)
    fallback = pd.to_datetime(us, format='%m/%d/%Y', errors='coerce').dt.date
    return days.fillna(pd.Series(pd.array(fallback, dtype=cal.DATE_DTYPE), index=series.index))


@dataclass
class CollectionRecords:
    table: pd.DataFrame
    schemas: list
    collisions: set


def _btris_records(files, issues, eda=None):
    """Preserve the original field types; Cluster Name and Observation Name are distinct."""
    aliases = {
        'patient_id': ('patient_id', 'ids__patient_record_number', 'MRN', 'Patient ID'),
        'order_identifier': ('Order ID', 'Order Identifier', 'Order Number'),
        'order_name_original': ('Order Name', 'order_name'),
        'day': ('Collected Date Time', 'Collected Date'),
        **NAME_FIELDS,
    }
    available, schemas, collisions = [], [], set()
    order = BTRIS_KEYS[1:]
    wanted = None
    if eda is not None:
        target = eda.loc[eda.canonical_analyte.isin(LAB_SOURCES.values())] if 'canonical_analyte' in eda else eda
        keys = pd.DataFrame({name: _clean(_column(target, name)) for name in order}).dropna().drop_duplicates()
        wanted = pd.MultiIndex.from_frame(keys)
    for role, paths in files.items():
        schema_counts = {}
        source_ids = set()
        for path in paths:
            try:
                header = pd.read_csv(path, nrows=0, encoding='utf-8-sig').columns
                names = {key: next((name for name in options if name in header), None)
                         for key, options in aliases.items()}
                shape = (names['day'] or 'COLLECTION_COLUMN_MISSING',
                         '+'.join(field for field in NAME_FIELDS if names[field]) or 'DISCRIMINATOR_MISSING')
                schema_counts[shape] = schema_counts.get(shape, 0) + 1
                if (not all(names[key] for key in ('patient_id', 'order_identifier', 'order_name_original', 'day'))
                        or not any(names[field] for field in NAME_FIELDS)):
                    raise ValueError('COLLECTION_FIELDS_MISSING')
                projection = {name: key for key, name in names.items() if name}
                if wanted is not None and wanted.empty:
                    continue
                file_records = []
                with pd.read_csv(path, usecols=list(projection), dtype='string', keep_default_na=False,
                                 encoding='utf-8-sig', chunksize=200_000) as chunks:
                    for chunk in chunks:
                        rows = chunk.rename(columns=projection)
                        source_ids.update(rows.patient_id.unique())
                        for field in order:
                            rows[field] = _clean(rows[field])
                        if wanted is not None:
                            rows = rows.loc[pd.MultiIndex.from_frame(rows[order]).isin(wanted)].copy()
                        if rows.empty:
                            continue
                        rows['_patient_match'] = NORMALIZE(rows.pop('patient_id'))
                        rows['day'] = _collection_days(rows.day)
                        for field in NAME_FIELDS:
                            rows[field] = _clean(_column(rows, field))
                        file_records.append(rows.dropna(subset=BTRIS_KEYS).assign(origin=role.split('_')[1]))
                available.extend(file_records)
            except READ_ERRORS:
                issue = dict(status='DATA_UNAVAILABLE', artifact=role)
                if issue not in issues:
                    issues.append(issue)
        if not paths:
            issues.append(dict(status='DATA_UNAVAILABLE', artifact=role))
        if source_ids:
            collisions |= _collisions(pd.Series(list(source_ids), dtype='string'))
        for (collection, discriminators), count in schema_counts.items():
            schemas.append(dict(source_role=role, collection_column=collection,
                                original_discriminators=discriminators, n_source_files=count))
    columns = [*BTRIS_KEYS, *NAME_FIELDS, 'day', 'origin']
    table = pd.concat(available, ignore_index=True) if available else pd.DataFrame(columns=columns)
    return CollectionRecords(table, schemas, collisions)


def _original_collection(rows, records):
    """Join on exact order and available original discriminators with field semantics."""
    order = BTRIS_KEYS[1:]
    measures = records.table
    evidence = pd.DataFrame(index=rows.index)
    for origin in cal.BTRIS_ORIGINS:
        evidence[origin] = pd.Series(pd.array([None] * len(rows), dtype=cal.DATE_DTYPE), index=rows.index)
    evidence['ambiguous'] = False
    evidence['cross_patient'] = False
    evidence['order_without_discriminator_match'] = False
    if measures.empty or rows.empty:
        return evidence
    query = rows.reset_index(names='_source_row')
    # Restrict to measured orders first to avoid a source-wide many-to-many merge.
    keys = query[order].dropna().drop_duplicates()
    measures = measures.merge(keys, on=order, how='inner', validate='many_to_one').drop_duplicates()
    joined = query[['_source_row', *BTRIS_KEYS, *NAME_FIELDS]].dropna(subset=order).merge(
        measures, on=order, suffixes=('_eda', '_original'))
    known, mismatch = [], []
    for name in NAME_FIELDS:
        eda, original = joined[name + '_eda'], joined[name + '_original']
        both = eda.notna() & original.notna()
        known.append(both)
        mismatch.append(both & ~eda.eq(original).fillna(False))
    compatible = pd.concat(known, axis=1).any(axis=1) & ~pd.concat(mismatch, axis=1).any(axis=1)
    same_patient = joined._patient_match_eda.eq(joined._patient_match_original).fillna(False)
    matched = joined.loc[compatible & same_patient]
    # Shared order identifiers alone are common across namespaces. An exact
    # discriminator for a different owner with no match for this owner is a conflict.
    cross = joined.loc[compatible & ~same_patient, '_source_row']
    evidence.loc[cross[~cross.isin(matched._source_row)].unique(), 'cross_patient'] = True
    unmatched = joined.loc[same_patient & ~compatible, '_source_row']
    evidence.loc[unmatched[~unmatched.isin(matched._source_row)].unique(),
                 'order_without_discriminator_match'] = True
    for origin, group in matched.groupby('origin'):
        dates = group.groupby('_source_row').day.agg(['first', 'nunique'])
        evidence.loc[dates.index, origin] = dates['first'].astype(cal.DATE_DTYPE)
        evidence.loc[dates.index[dates['nunique'].gt(1)], 'ambiguous'] = True
    return evidence


def _selection_comparison(rows, tables):
    """Audit differing Step01/context selections; they can describe different tests."""
    discrepancies = pd.Series(False, index=rows.index)
    selected, context = tables.get('step01'), tables.get('context')
    if selected is None or context is None or rows.empty:
        return discrepancies
    selected = selected.drop_duplicates(KEYS + ['lab_id'], keep=False)
    context = context.drop_duplicates(KEYS, keep=False)
    columns = ['patient_id', 'source_episode_id']
    for lab in set(LAB_SOURCES.values()):
        context_day = f'lab__{lab}__measurement_date'
        if context_day not in context:
            continue
        other = selected.loc[selected.lab_id.eq(lab), KEYS + ['selected_lab_date']].merge(
            context[KEYS + [context_day]], on=KEYS, validate='one_to_one')
        day1, day2 = (_collection_days(other[name]) for name in ('selected_lab_date', context_day))
        pairs = other.loc[day1.notna() & day2.notna() & day1.ne(day2), KEYS]
        mask = rows.canonical_analyte.eq(lab) & pd.MultiIndex.from_frame(rows[columns]).isin(
            pd.MultiIndex.from_frame(pairs))
        discrepancies.loc[mask] = True
    return discrepancies


def _candidates(tables, baseline, records, policy):
    eda = tables['eda_long']
    rows = eda.loc[eda.canonical_analyte.isin(LAB_SOURCES.values())].copy().reset_index(drop=True)
    # Keep the original source ID, episode and uncertainty in memory only.
    rows['source_patient_id'] = _text(rows.pop('patient_id'))
    rows['_patient_match'] = NORMALIZE(rows.source_patient_id)
    for field in {*DEFINITION, *NAME_FIELDS, 'order_identifier'}:
        rows[field] = _clean(_column(rows, field))
    for field in set(EDA_FIELDS) - set(rows) - {'patient_id'}:
        rows[field] = _column(rows, field)
    rows['source_episode_id'] = _clean(rows.matched_clinical_episode_id)
    rows['source_episode_unknown'] = rows.source_episode_id.isna()
    dates = _original_collection(rows, records)

    official, unsafe, owners, episodes, duplicate_episodes = _identity_tables(
        baseline, tables.get('step10', pd.DataFrame(columns=BASE)),
        tables.get('context', pd.DataFrame()), records.collisions, eda.patient_id)
    pairs = official.set_index('_patient_match')
    rows['patient_id'] = rows._patient_match.map(pairs.frozen_patient_id)
    rows['clinical_baseline_date'] = rows._patient_match.map(pairs.clinical_baseline_date)
    episode_owners = rows.source_episode_id.map(owners)
    cross_episode = pd.Series(
        [isinstance(ids, frozenset) and bool(ids) and key not in ids
         for key, ids in zip(rows._patient_match, episode_owners)], index=rows.index)
    ambiguous_episode = (rows.source_episode_id.isin(duplicate_episodes) |
                         episode_owners.map(lambda ids: isinstance(ids, frozenset) and len(ids) > 1))
    reported = ~rows.source_episode_unknown
    link = _clean(rows.episode_match_ambiguous).str.lower().isin(FALSE)
    verified = pd.MultiIndex.from_frame(rows[['_patient_match', 'source_episode_id']]).isin(episodes)
    rows['identity_status'] = np.select(
        [rows._patient_match.isna(), cross_episode | dates.cross_patient,
         ambiguous_episode | (reported & ~link), rows._patient_match.isin(unsafe),
         rows.patient_id.isna(), reported & ~verified, reported],
        ['PATIENT_ID_AMBIGUOUS', 'CROSS_PATIENT_CONFLICT', 'AMBIGUOUS_IDENTITY', 'PATIENT_ID_AMBIGUOUS',
         'PATIENT_NOT_RECONCILED_WITH_FROZEN_BASELINE', 'SOURCE_EPISODE_NOT_VERIFIED', EXACT],
        default=PATIENT_ONLY)

    day = _collection_days(rows.lab_date)
    specimen = _collection_days(_column(rows, 'specimen_datetime')).where(
        _clean(_column(rows, 'specimen_date_origin')).isin(policy['specimen_collection_origins']))
    raw, intermediate = dates.raw, dates.intermediate
    discordant = pd.concat([(value.notna() & day.notna() & value.ne(day)).fillna(False)
                            for value in (raw, intermediate, specimen)], axis=1).any(axis=1)
    rows['date_evidence'] = np.select(
        [dates.ambiguous, discordant, day.isna(),
         *[value.eq(day).fillna(False).to_numpy(dtype=bool) for value in (raw, intermediate, specimen)]],
        ['SOURCE_LINK_AMBIGUOUS', 'DATE_DISCORDANT', 'DATE_MISSING',
         'RAW_AND_EDA_AGREE', 'INTERMEDIATE_AND_EDA_AGREE', 'SPECIMEN_AND_EDA_AGREE'], default=EDA_ONLY)
    rows['btris_discriminator_unmatched'] = dates.order_without_discriminator_match
    rows['step01_context_date_disagreement'] = _selection_comparison(rows, tables)
    valid_temporal = ~rows.date_evidence.isin(DATE_FAILURES) & rows.identity_status.isin(LINKED)
    rows['collection_calendar_day'] = day
    delta = cal.calendar_age(_collection_days(rows.clinical_baseline_date), day)
    rows['delta_days'] = delta.where(valid_temporal)
    rows['stratum'] = _strata(rows.delta_days, policy['time'])
    rows['provenance'] = np.select([rows.date_evidence.eq(EDA_ONLY), ~rows.date_evidence.isin(DATE_FAILURES)],
                                   [EDA_ONLY, ORIGINAL], default='UNRESOLVED_DATE_PROVENANCE')
    rows['mapping_candidate'] = STEP01._mapping_valid(rows) & _clean(rows.semantic_mapping_status).notna()
    rows['eda_result_valid'] = _clean(rows.result_valid_for_analysis).str.lower().isin(TRUE)
    return rows


def _strata(delta, time):
    """delta = baseline day − collection day. Follow-up has a negative delta."""
    primary, early = time['prebaseline_primary'], time['followup_early']
    masks = [delta.between(primary['min_day_difference'], primary['max_day_difference']),
             delta.eq(time['same_day']['day_difference']),
             (-delta).between(early['min_day_after_baseline'], early['max_day_after_baseline']),
             (-delta).ge(time['followup_late']['min_day_after_baseline']),
             delta.ge(time['earlier_than_prebaseline']['min_day_difference'])]
    return np.select([mask.fillna(False).to_numpy(dtype=bool) for mask in masks],
                     STRATA, default=UNRESOLVED)


@dataclass
class Outcome:
    inventory: pd.DataFrame
    selected: pd.DataFrame
    primary: dict
    duplicates: pd.DataFrame


def _outcome(candidates, source, policy):
    """Resolve exact definitions and values before selecting each latest day."""
    rows = candidates.loc[candidates.canonical_analyte.eq(LAB_SOURCES[source])].copy()
    raw, operator = _clean(rows.result_raw), _clean(rows.result_operator)
    censored = (raw.str.match(RESULT_LIMIT, na=False) | operator.str.match(RESULT_LIMIT, na=False) |
                operator.str.upper().isin({'LT', 'LE', 'GT', 'GE', 'LTE', 'GTE'}) |
                _clean(rows.result_numeric_bound).notna())
    admin = _clean(rows.result_text).fillna(raw).str.fullmatch(cal.ADMINISTRATIVE_RESULT, na=False)
    numeric = source == RF_VALUE
    if numeric:
        rows['value'] = pd.to_numeric(rows.result_numeric_exact, errors='coerce').astype(float)
        known_units = {unit.casefold() for unit in policy['rf_numeric_units']}
        unit = rows.unit.str.casefold().str.replace(r'\s+', '', regex=True)
        bad_unit = ~unit.isin(known_units)
        exact_operator = operator.isna() | operator.str.upper().isin({'=', '==', 'EQ'})
        raw_number = pd.to_numeric(raw, errors='coerce').astype(float)
        numeric_disagreement = raw_number.notna() & ~np.isclose(raw_number, rows.value, rtol=1e-10, atol=0)
        bad_value = ~np.isfinite(rows.value) | ~exact_operator | numeric_disagreement
    else:
        crosswalk = {token: code for code, words in enumerate(
            (policy['qualitative_crosswalk']['negative'], policy['qualitative_crosswalk']['positive']))
                     for token in words}
        text = _clean(rows.result_text).fillna(raw).str.casefold().str.replace(r'[\s_-]+', ' ', regex=True)
        rows['value'] = text.map(crosswalk).astype(float)
        bad_value = rows.value.isna()
        bad_unit = pd.Series(False, index=rows.index)
    missing_definition = (rows.order_name_original.isna() |
                          rows[['cluster_name_original', 'observation_name_original']].isna().all(axis=1))
    rows['first_reason'] = np.select(
        [~rows.identity_status.isin(LINKED), rows.date_evidence.isin(DATE_FAILURES),
         rows.stratum.eq(UNRESOLVED), rows.stratum.eq(OLDER), ~rows.mapping_candidate, censored, admin,
         bad_value, ~rows.eda_result_valid, missing_definition, bad_unit],
        [rows.identity_status, rows.date_evidence, 'DATE_MISSING', 'OUTSIDE_365_DAY_LOOKBACK',
         'ANALYTE_MAPPING_UNVERIFIED', 'CENSORED_RESULT', 'ADMINISTRATIVE_RESULT',
         'NOT_EXACT_NUMERIC_RESULT' if numeric else 'UNKNOWN_RESULT_TOKEN',
         'INVALID_RESULT', 'ORIGINAL_TEST_DEFINITION_UNAVAILABLE', 'NUMERIC_UNIT_UNAVAILABLE'],
        default='')
    valid = rows.loc[rows.first_reason.eq('')].copy()
    partitions = ['stratum', 'provenance']
    assay_fields = DEFINITION[:-1]
    same_day = ['patient_id', *partitions, 'collection_calendar_day']
    signatures = valid[DEFINITION].astype(object).where(valid[DEFINITION].notna(), None)
    valid['definition'] = [json.dumps(fields, ensure_ascii=False) for fields in
                           signatures.itertuples(index=False, name=None)]
    valid['definition_id'] = valid.definition.map(cal.fingerprint)
    group_keys = ['patient_id', *partitions, 'definition']
    duplicates = valid.duplicated(same_day + DEFINITION + ['source_episode_id', 'order_identifier',
                                  'observation_identifier', 'date_evidence', *cal.RESULT_FIELDS])
    rows.loc[valid.index[duplicates], 'first_reason'] = 'IDENTICAL_DUPLICATE'
    duplicate_records = valid.loc[duplicates].copy()
    valid = valid.loc[~duplicates].copy()
    latest = valid.collection_calendar_day.eq(valid.groupby(group_keys).collection_calendar_day.transform('max'))
    rows.loc[valid.index[~latest], 'first_reason'] = 'EARLIER_ELIGIBLE_DAY'
    on_day = valid.loc[latest].copy()
    conflict = on_day.groupby(group_keys).value.transform('nunique').gt(1)
    if numeric:
        unit_groups = same_day + assay_fields
        mixed_units = valid.groupby(unit_groups, dropna=False).unit.transform('nunique').gt(1)
        conflict |= mixed_units.loc[on_day.index]
    rows.loc[on_day.index[conflict], 'first_reason'] = 'AMBIGUOUS_LATEST_DAY'
    safe = on_day.loc[~conflict].sort_values(['date_evidence', 'order_identifier'], kind='stable')
    selected = safe.drop_duplicates(group_keys).copy()
    rows.loc[safe.index.difference(selected.index), 'first_reason'] = 'CONSISTENT_LATEST_DAY_REPEAT'
    primary = {}
    for partition, group in selected.groupby(partitions):
        coverage = group.groupby('definition').patient_id.nunique()
        definition = min(coverage.index, key=lambda key: (-coverage[key], key))
        primary[partition] = cal.fingerprint(definition)
    return Outcome(rows, selected, primary, duplicate_records)


def _test_disposition(result, patients):
    """One disposition per frozen patient/outcome; measurement windows stay separate."""
    ledger = result.inventory.loc[result.inventory.identity_status.isin(LINKED)]
    disposition = pd.Series('NO_TEST', index=pd.Index(patients, name='patient_id'), dtype='string')
    disposition.loc[disposition.index.intersection(ledger.patient_id)] = 'TESTED_INVALID'
    for strata, status in [({OLDER}, 'TESTED_PREBASELINE_NO_MATCH'),
                           ({EARLY, LATE}, 'TESTED_POSTBASELINE_ONLY'),
                           ({DAY0}, 'TESTED_DAY0_ONLY'), ({PRE}, 'TESTED_PREBASELINE_NO_MATCH')]:
        scoped = ledger.loc[ledger.stratum.isin(strata), 'patient_id']
        disposition.loc[disposition.index.intersection(scoped)] = status
    original = result.selected.loc[result.selected.stratum.eq(PRE) & result.selected.provenance.eq(ORIGINAL)]
    match = result.primary.get((PRE, ORIGINAL))
    disposition.loc[disposition.index.intersection(original.loc[original.definition_id.eq(match), 'patient_id'])] = (
        'TESTED_PREBASELINE_MATCHED')
    return disposition

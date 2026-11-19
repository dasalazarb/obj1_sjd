"""Restricted calendar-day N3 summaries of frozen Graph02/03 communities."""

import json
from importlib import import_module
from pathlib import Path
import re
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import yaml

from .._pipeline import digest
from . import s03_n3_calendar_day as cal
from .s03_construct_validation import _dependencies, _paths, _tokens
from .s03_n3_auto import CONFIGS, PHASEC, VALIDATION, _cached, _locked, _settings
from .s03_n3_candidate_audit import _proxy_alerts
from .s03_n3_incremental import _design, _fit
from .s03_n3_measurement_source import (
    BASE, EDA_FIELDS, EDA_REQUIRED, KEYS, LAB_CONTEXT, LAB_INPUTS, LAB_SOURCES,
    SELECTED_FIELDS, _audited_rows, _private_directory, _read_frozen, _text,
)
from .s03_n3_resolution import _snapshot
from .s03_n3_resolution_support import support_diagnostics
from .s03_release_reader import hard_members
from .s03_run_resolution import resolve_graph03_dependencies

MODE = 'calendar_day_exploratory'
REPS = ('R0', 'R1', 'R2', 'R3', 'R4')
WINDOW = {'minimum_days_before_baseline': 1, 'lookback_days': 365}
CELL = 5
SUPPORT = dict(minimum_model_patients=30, minimum_group_patients=8, minimum_patients_per_community=8)
EVENTS_PER_PARAMETER = 5
BINARY = {'positive': 1, 'detected': 1, 'negative': 0, 'not detected': 0}
DEFINITION = ['order_name_original', 'cluster_name_original', 'observation_name_original', 'assay', 'unit']
RF_VALUE = 'lab__rheumatoid_factor__value'
RF_TEXT = 'lab__rheumatoid_factor__text'
ANA_PAIR = {'lab__ana_status__text', 'lab__ana_hep2_status__text'}
OUTCOMES = (RF_VALUE, RF_TEXT, *sorted(set(LAB_SOURCES) - {RF_VALUE, RF_TEXT}))
CIRCULAR = 'DESCRIPTIVE_ONLY_POTENTIAL_CIRCULARITY'
LIMITATIONS = [
    'Exploratory associations conditional on frozen communities and available measurements.',
    'Step20 lab_date has calendar-day precision. Collection evidence may be limited to the specimen field carried by EDA.',
    'Clinical testing and complete-case selection may introduce bias. Models are partially adjusted and omit research protocol.',
    'R2 is exploratory. Intervals omit uncertainty from Mapper and community selection.',
]
TABLE_COLUMNS = {
    'coverage': 'representation_id outcome status n_total n_measured n_missing measurement_fraction missing_reasons_json',
    'community_profiles': 'representation_id outcome community_id topology_status status n_total n_measured n_missing '
                          'measurement_fraction n_positive n_negative positive_fraction median q1 q3',
    'models': 'representation_id outcome hypothesis_family topology_status status reason n covariates family '
              'p_incremental q_bh r2_a r2_b effects_json diagnostics',
    'exclusions': 'outcome reason n_rows n_patients',
}
READ_ERRORS = (OSError, ValueError, TypeError, KeyError)
STEP01 = import_module('src.block_A.01_serological_profile')


def _inputs(paths, issues):
    fields = {
        'eda_long': (EDA_REQUIRED, EDA_FIELDS | {
            'observation_name_original', 'observation_identifier', 'specimen_type', 'source_protocol'}),
        'step01': (set(KEYS + ['lab_id', 'selected_lab_date']), SELECTED_FIELDS),
        'step10': (set(BASE), set(BASE) | set(LAB_SOURCES) | {'clinical_baseline_episode_id'}),
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


def _unique_rows(frame, keys):
    frame = frame.copy()
    for key in keys:
        frame[key] = _text(frame[key])
    return frame.loc[frame[keys].notna().all(axis=1) & ~frame.duplicated(keys, keep=False)]


def _reconciled_tables(baseline, analytic, context, selected):
    """Pass only reproducible Step10 episode keys to the existing row audit."""
    baseline = baseline.copy()
    flags = analytic.loc[analytic.is_clinical_baseline.eq(True), 'patient_id'].astype('string')
    analytic = _unique_rows(analytic, KEYS)
    context = _unique_rows(context, KEYS)
    for frame in (baseline, analytic, context):
        for field in set(BASE[2:4]) & set(frame):
            frame[field] = cal.calendar_days(frame[field], allow_time=True).astype('string')
    matched = analytic.merge(context[KEYS], on=KEYS, how='inner', validate='one_to_one')
    if 'clinical_anchor_date' in context:
        matched = matched.merge(context[KEYS + ['clinical_anchor_date']], on=KEYS, validate='one_to_one',
                                suffixes=('', '_context'))
        matched = matched.loc[matched.clinical_anchor_date.eq(matched.clinical_anchor_date_context)]
        matched = matched.drop(columns='clinical_anchor_date_context')

    # Step20 and Step01 use this same documented ID normalization. Exclude
    # collisions across the full official ID set before using the exact episode key.
    identities = pd.concat([baseline.patient_id, analytic.patient_id]).drop_duplicates()
    normalized = STEP01._normalize_patient_id(identities)
    safe = identities.loc[normalized.notna() & ~normalized.duplicated(keep=False)]
    pairs = baseline[BASE].merge(
        matched.loc[matched.is_clinical_baseline.eq(True) & matched.patient_id.isin(safe)
                    & ~matched.patient_id.isin(flags[flags.duplicated(keep=False)]), BASE],
        on=KEYS, suffixes=('', '_step10'), validate='one_to_one')
    days = pairs.clinical_baseline_date
    same = days.eq(pairs.clinical_anchor_date) & days.eq(pairs.clinical_anchor_date_step10) & days.eq(
        pairs.clinical_baseline_date_step10)
    official = pairs.loc[same, BASE]
    matched = matched.loc[matched.patient_id.isin(official.patient_id)]
    context = context.merge(matched[KEYS], on=KEYS, how='inner', validate='one_to_one')
    selected = _unique_rows(selected, KEYS + ['lab_id'])
    return official, matched, context, selected


def _collection_days(series):
    """Read the written ISO or US collection day without shifting timezones."""
    days = cal.calendar_days(series, allow_time=True)
    us = series.astype('string').str.extract(r'^(\d{1,2}/\d{1,2}/\d{4})(?:[ T].*)?$', expand=False)
    fallback = pd.to_datetime(us, format='%m/%d/%Y', errors='coerce').dt.date
    return days.fillna(pd.Series(pd.array(fallback, dtype=cal.DATE_DTYPE), index=series.index))


def _btris_records(files, issues):
    """Match original order/observation identity and audit the collection day only."""
    aliases = [
        ('patient_id', 'ids__patient_record_number', 'MRN', 'Patient ID'),
        ('Order ID', 'Order Identifier', 'Order Number'),
        ('Order Name', 'order_name'),
        ('Cluster Name', 'cluster_name', 'Observation Name'),
        ('Collected Date Time', 'Collected Date', 'Specimen Date Time'),
    ]
    keys = list(cal.KEY_FIELDS)
    columns = [*keys, 'btris_source', '_btris_day_raw', '_btris_day_intermediate',
               '_source_ambiguous_raw', '_source_ambiguous_intermediate']
    available = []
    for role, paths in files.items():
        for path in paths:
            try:
                header = pd.read_csv(path, nrows=0, encoding='utf-8-sig').columns
                names = [next((name for name in group if name in header), None) for group in aliases]
                if None in names:
                    raise ValueError('COLLECTION_FIELDS_MISSING')
                rows = pd.read_csv(path, usecols=names, dtype='string', keep_default_na=False, encoding='utf-8-sig')
                rows = rows.rename(columns=dict(zip(names, keys + ['day'])))
                rows['patient_id'] = STEP01._normalize_patient_id(rows.patient_id)
                rows['day'] = _collection_days(rows.day)
                rows = rows.loc[rows[keys].notna().all(axis=1) & rows[keys].ne('').all(axis=1)]
                available.append(rows.assign(origin=role.split('_')[1]))
            except READ_ERRORS:
                issue = dict(status='DATA_UNAVAILABLE', artifact=role)
                if issue not in issues:
                    issues.append(issue)
    records = pd.DataFrame(columns=columns)
    if available:
        observations = pd.concat(available, ignore_index=True)
        for origin in cal.BTRIS_ORIGINS:
            scoped = observations.loc[observations.origin.eq(origin)]
            day = scoped.groupby(keys).day.agg(['first', 'nunique']).reset_index()
            day['_source_ambiguous_' + origin] = day.pop('nunique').gt(1)
            day = day.rename(columns={'first': '_btris_day_' + origin})
            records = (day if origin == 'raw' else records.merge(day, on=keys, how='outer', validate='one_to_one'))
        records['btris_source'] = 'BTRIS'
    # Arrow date type is needed even when no CSV could be read.
    for origin in cal.BTRIS_ORIGINS:
        records['_btris_day_' + origin] = records['_btris_day_' + origin].astype(cal.DATE_DTYPE)
    return records


def _candidates(tables, baseline, records):
    eda = tables['eda_long'].copy().reset_index(drop=True)
    eda['patient_id'] = STEP01._normalize_patient_id(eda.patient_id)
    for field in ('order_identifier', *DEFINITION, 'specimen_datetime'):
        eda[field] = _text(eda[field]) if field in eda else pd.NA
    eda['cluster_name_original'] = _text(eda.cluster_name_original).fillna(_text(eda.observation_name_original))
    # The source audit compares calendar days. Convert source date syntax
    # while retaining the original field for exact duplicate checking.
    eda['specimen_datetime_original'] = eda.specimen_datetime
    eda['specimen_datetime'] = _collection_days(eda.specimen_datetime)
    official, analytic, context, selected = _reconciled_tables(
        baseline, tables['step10'], tables['context'], tables['step01'])
    if official.empty:
        ledger = eda.rename(columns={'patient_id': '_patient_match'})[['canonical_analyte', '_patient_match']].copy()
        return pd.DataFrame(), ledger.assign(identity_status='OFFICIAL_EPISODE_UNVERIFIED')
    calendar = dict(windows=dict.fromkeys(LAB_SOURCES, WINDOW), linkage_fields={
        key: {'eda': field} for key, field in zip(
            cal.KEY_FIELDS, ('patient_id', 'order_identifier', 'order_name_original', 'cluster_name_original'))
    })
    return _audited_rows(eda, selected, analytic, context, official, None,
                         calendar=calendar, records=records)


def _outcome(candidates, ledger, source):
    """Select one unambiguous result on the latest collection day per patient."""
    exclusions = []

    def record(patients, reason):
        if len(patients):
            exclusions.append(pd.DataFrame({'patient_key': patients, 'reason': reason}))

    valid_identity = {'OFFICIAL_BASELINE_EPISODE_MATCHED', 'SAME_PATIENT_OTHER_EPISODE_MATCHED'}
    unmatched = ledger.loc[ledger.canonical_analyte.eq(LAB_SOURCES[source])
                           & ~ledger.identity_status.isin(valid_identity)]
    for reason, group in unmatched.groupby('identity_status'):
        record(group._patient_match, reason)
    empty = pd.DataFrame(columns=['patient_id', 'value', 'date_evidence'])

    def result(rows, definition=None):
        rejected = pd.concat(exclusions, ignore_index=True) if exclusions else pd.DataFrame(
            columns=['patient_key', 'reason'])
        return rows, rejected, definition

    if candidates.empty:
        return result(empty)
    rows = candidates.loc[candidates.source_column.eq(source)].copy()
    rows['patient_key'] = STEP01._normalize_patient_id(rows.patient_id)

    def exclude(mask, reason):
        nonlocal rows
        mask = mask.reindex(rows.index, fill_value=False).fillna(False)
        record(rows.loc[mask, 'patient_key'], reason)
        rows = rows.loc[~mask].copy()

    day = rows.eda_calendar_day
    specimen = rows.specimen_datetime
    raw, intermediate = rows._btris_day_raw, rows._btris_day_intermediate
    selected_day = cal.calendar_days(rows.step01_date, allow_time=True)
    context_day = cal.calendar_days(rows.step10_context_date, allow_time=True)
    selection_conflict = (selected_day.notna() & context_day.notna() & selected_day.ne(context_day)
                          & (day.eq(selected_day) | day.eq(context_day)))
    discordant = selection_conflict.fillna(False)
    for evidence in (raw, intermediate, specimen):
        discordant |= (evidence.notna() & day.notna() & evidence.ne(day)).fillna(False)
    exclude(rows.calendar_date_status.eq('BTRIS_ORIGINAL_IDENTITY_AMBIGUOUS'), 'AMBIGUOUS_BTRIS_COLLECTION')
    exclude(discordant, 'COLLECTION_DAY_DISCORDANT')
    exclude(day.isna(), 'COLLECTION_DAY_MISSING_OR_INVALID')
    corroborated = (raw.eq(day) | intermediate.eq(day) | specimen.eq(day)).fillna(False)
    exclude(~corroborated, 'COLLECTION_DAY_UNVERIFIED')
    rows['date_evidence'] = np.select(
        [raw.eq(day).reindex(rows.index).fillna(False).to_numpy(dtype=bool),
         intermediate.eq(day).reindex(rows.index).fillna(False).to_numpy(dtype=bool)],
        ['RAW_BTRIS_COLLECTION_DAY', 'INTERMEDIATE_BTRIS_COLLECTION_DAY'],
        default='EDA_SPECIMEN_COLLECTION_DAY')
    rows['collection_calendar_day'] = rows.eda_calendar_day
    age = cal.calendar_age(cal.calendar_days(rows.clinical_baseline_date, allow_time=True),
                           rows.collection_calendar_day)
    exclude(age.eq(0), 'SAME_DAY_AS_BASELINE')
    exclude(age.lt(0), 'AFTER_BASELINE')
    exclude(age.gt(WINDOW['lookback_days']), 'OUTSIDE_365_DAY_LOOKBACK')
    exclude(age.isna(), 'OFFICIAL_BASELINE_DAY_UNAVAILABLE')
    exclude(~rows.mapping_candidate, 'ANALYTE_MAPPING_UNVERIFIED')
    exclude(rows.collection_calendar_day.ne(
        rows.groupby('patient_id').collection_calendar_day.transform('max')), 'EARLIER_ELIGIBLE_DAY')

    observation = [*KEYS, *DEFINITION, 'lab_date', 'collection_calendar_day', 'source_episode_id', 'order_identifier',
                   'specimen_datetime_original', 'specimen_type', 'source_protocol', 'observation_identifier',
                   *cal.RESULT_FIELDS]
    duplicates = rows.duplicated([field for field in observation if field in rows])
    exclude(duplicates, 'IDENTICAL_DUPLICATE')
    exclude(rows.duplicated('patient_id', keep=False), 'DISCORDANT_LATEST_DAY')
    exclude(rows.censored, 'CENSORED_RESULT')
    admin = rows.nonresult_token | _text(rows.value).str.fullmatch(cal.ADMINISTRATIVE_RESULT, na=False)
    exclude(admin, 'ADMINISTRATIVE_RESULT')
    exclude(~rows.eda_result_valid, 'INVALID_RESULT')
    exclude(rows.value.isna(), 'MISSING_RESULT')
    exclude(rows[DEFINITION[:2]].isna().any(axis=1), 'ORIGINAL_TEST_DEFINITION_UNAVAILABLE')
    if source == RF_VALUE:
        rows['value'] = pd.to_numeric(rows.value, errors='coerce')
        exclude(~np.isfinite(rows.value), 'NOT_EXACT_NUMERIC_RESULT')
        # Retain original unit text for conventional short units only.
        exclude(~_text(rows.unit).str.fullmatch(r'[A-Za-zµμ%0-9][A-Za-zµμ%0-9 ./^()\-]{0,23}', na=False),
                'NUMERIC_UNIT_UNAVAILABLE')
    else:
        rows['value'] = _text(rows.value).str.strip().str.casefold().map(BINARY)
        exclude(rows.value.isna(), 'UNINTERPRETABLE_QUALITATIVE_RESULT')
    if rows.empty:
        return result(empty)
    # Count patients after latest-day selection. The lexical tie-break never
    # uses communities or outcome values. Missing assay and unit are distinct.
    definitions = [tuple(None if pd.isna(x) else str(x) for x in values)
                   for values in rows[DEFINITION].itertuples(index=False, name=None)]
    rows['definition_key'] = [json.dumps(value, ensure_ascii=False) for value in definitions]
    chosen = min(rows.definition_key.value_counts().items(), key=lambda item: (-item[1], item[0]))[0]
    exclude(rows.definition_key.ne(chosen), 'NON_PRIMARY_TEST_DEFINITION')
    return result(rows[['patient_id', 'value', 'date_evidence']].copy(), json.loads(chosen))


def _small(count):
    return 0 < count < CELL


def _coverage_counts(total, measured):
    missing = total - measured
    if any(_small(count) for count in (total, measured, missing)):
        return {}
    return dict(n_total=total, n_measured=measured, n_missing=missing,
                measurement_fraction=measured / total if total else None)


def _profiles(frame, label, numeric):
    profiles = []
    for community, group in frame.groupby('community_id', sort=True):
        if len(group) < CELL:
            # Small communities share one withheld row, with no community ID.
            continue
        values = group.value.dropna()
        counts = _coverage_counts(len(group), len(values))
        row = dict(**label, community_id=community, status='DESCRIPTIVE' if counts else 'SUPPRESSED', **counts)
        if counts and numeric and len(values) >= CELL:
            row.update(zip(('q1', 'median', 'q3'), values.quantile([.25, .5, .75]).to_numpy()))
        if counts and not numeric:
            positive = int(values.eq(1).sum())
            negative = len(values) - positive
            if not (_small(positive) or _small(negative)):
                row.update(n_positive=positive, n_negative=negative,
                           positive_fraction=positive / len(values) if len(values) else None)
        profiles.append(row)
    if frame.community_id.value_counts().lt(CELL).any():
        profiles.append(dict(**label, status='SUPPRESSED'))
    return profiles


def _coverage(frame, rejected, label, profiles):
    row = dict(representation_id=label['representation_id'], outcome=label['outcome'], status='DESCRIPTIVE')
    if any(profile['status'] == 'SUPPRESSED' for profile in profiles):
        return dict(row, status='SUPPRESSED')
    missing = frame.loc[frame.value.isna(), 'patient_id']
    row.update(_coverage_counts(len(frame), len(frame) - len(missing)))
    if len(missing):
        # Each patient contributes to one disposition so no hidden reason can
        # be recovered by subtracting other reasons from n_missing.
        reasons = rejected.groupby('patient_key').reason.agg(
            lambda values: values.iloc[0] if values.nunique() == 1 else 'MULTIPLE_EXCLUSION_REASONS')
        normalized = STEP01._normalize_patient_id(missing)
        causes = normalized.map(reasons).fillna('NO_ELIGIBLE_LINKED_MEASUREMENT').value_counts()
        if not causes.map(_small).any():
            row['missing_reasons_json'] = json.dumps(causes.to_dict(), sort_keys=True)
    return row


def _safe_exclusions(rejected, source):
    for reason, group in rejected.groupby('reason', sort=True):
        unique = group.patient_key.nunique()
        # A large number of records from a few patients is also suppressed.
        if unique >= CELL and group.patient_key.notna().all():
            yield dict(outcome=source, reason=reason, n_rows=len(group), n_patients=unique)


def _independence(rep, source, roles, features, registry, dependencies, validation):
    role = roles.get((rep, source), {})
    known = (role.get('tier') == 'N3_REVIEWED' and role.get('independence_status') == 'REVIEWED_NO_CONFLICT'
             and role.get('inferential_eligible') is True and source in registry)
    if not known or role.get('direct_in_discovery') is True or _tokens(role.get('conflicting_defining_sources')):
        return False
    defining = set(features.loc[features.representation_id.eq(rep) & features.included.eq(True),
                                'source_column'].dropna())
    if not defining:
        return False
    # The two RF encodings and the two ANA tests can be proxies for each other
    # even when the feature metadata describes them as distinct endpoints.
    related = ANA_PAIR if source in ANA_PAIR else {RF_VALUE, RF_TEXT} if source in {RF_VALUE, RF_TEXT} else {source}
    return all(not set(_paths(dependencies, name)) & defining
               and not _proxy_alerts(name, defining, validation)
               and roles.get((rep, name), {}).get('tier') not in {'N1_DEFINING', 'N2_RELATED'}
               for name in related)


def _adjustment(frame):
    age = pd.to_numeric(frame['demo__age_at_baseline'], errors='coerce')
    sex = _text(frame['demo__sex']).str.strip().str.casefold().map(
        {'f': 'female', 'female': 'female', 'm': 'male', 'male': 'male'})
    day = cal.calendar_days(frame.clinical_baseline_date, allow_time=True)
    year = pd.to_datetime(day.astype('string'), errors='coerce').dt.year
    era = (year // 5 * 5).astype('Int64').astype('string')
    covariates = pd.DataFrame({'age': age.where(age.between(0, 120)), 'sex': sex, 'baseline_era': era})
    return covariates.dropna(axis=1, how='all')


def _model_support(frame, numeric):
    covariates = _adjustment(frame)
    complete = frame.value.notna() & covariates.notna().all(axis=1)
    counts = frame.loc[complete, 'community_id'].value_counts()
    keep = counts.index[counts.ge(SUPPORT['minimum_group_patients'])]
    common = complete & frame.community_id.isin(keep)
    y = frame.loc[common, 'value'].astype(float)
    # Publish N only when it cannot uncover a small complete-case exclusion.
    n = len(y) if len(y) >= CELL and not _small(int(frame.value.notna().sum()) - len(y)) else None
    adjustment = covariates.loc[common]
    adjustment = adjustment.loc[:, adjustment.nunique().gt(1)]
    names = '|'.join(adjustment.columns)
    if len(y) < SUPPORT['minimum_model_patients'] or len(keep) < 2:
        return dict(reason='INSUFFICIENT_COMPLETE_CASE_COMMUNITY_SUPPORT', n=n, covariates=names), None
    community = frame.loc[common, 'community_id']
    if not numeric:
        cells = y.groupby(community).agg(['sum', 'size'])
        if (cells['sum'].lt(CELL) | (cells['size'] - cells['sum']).lt(CELL)).any():
            return dict(reason='INSUFFICIENT_BINARY_COMMUNITY_SUPPORT', n=n, covariates=names), None
    diagnostics = support_diagnostics(
        y, adjustment, pd.DataFrame(index=y.index), community, 'numeric' if numeric else 'binary',
        SUPPORT, categories_verified=True, minimum_events_per_parameter=EVENTS_PER_PARAMETER)
    reasons = sorted(set(filter(None, '|'.join(diagnostics[model]['support_blocking_reasons']
                                               for model in ('A', 'B')).split('|'))))
    if reasons:
        return dict(reason='|'.join(reasons), n=n, covariates=names), None
    a = _design(adjustment)
    b = pd.concat([a, _design(community.to_frame()).drop(columns='intercept')], axis=1)
    return dict(n=n, covariates=names), (y, a, b, sorted(community.unique()))


def _estimate(prepared, numeric):
    y, a, b, levels = prepared
    family = 'gaussian_ols' if numeric else 'binomial_logit'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            fit_a, fit_b = (_fit(y, design, family) for design in (a, b))
            covariance = fit_b.cov_params().iloc[a.shape[1]:, a.shape[1]:]
            if not np.isfinite(covariance.to_numpy()).all() or np.linalg.matrix_rank(covariance) != len(covariance):
                raise ValueError('UNUSABLE_COVARIANCE')
            restriction = np.eye(b.shape[1])[a.shape[1]:]
            p = float(fit_b.wald_test(restriction, use_f=numeric, scalar=True).pvalue)
            terms = b.columns[a.shape[1]:]
            effects = fit_b.conf_int().loc[terms].copy()
            effects['estimate'] = fit_b.params.loc[terms]
            if not numeric:
                effects = np.exp(effects)
            if not np.isfinite(effects.to_numpy()).all() or not 0 <= p <= 1:
                raise ValueError('UNUSABLE_EFFECTS')
            contrasts = [dict(community_id=level, reference_community_id=levels[0],
                              scale='mean_difference' if numeric else 'odds_ratio',
                              effect=row[2], ci95_lower=row[0], ci95_upper=row[1])
                         for level, row in zip(levels[1:], effects.itertuples(index=False, name=None))]
            result = dict(status='EXPLORATORY_ESTIMATED', p_incremental=p, effects_json=json.dumps(contrasts),
                          diagnostics='HC3_WALD_F' if numeric else 'HC3_WALD_CHI2_NO_SEPARATION_DETECTED')
            if numeric:
                matrix = b.to_numpy()
                leverage = np.einsum('ij,ji->i', matrix, np.linalg.pinv(matrix))
                if np.max(leverage) >= 1 - 1e-10 or not all(np.isfinite(f.resid).all() for f in (fit_a, fit_b)):
                    raise ValueError('UNUSABLE_RESIDUALS_OR_LEVERAGE')
                result.update(r2_a=float(fit_a.rsquared), r2_b=float(fit_b.rsquared))
            return dict(result, family=family)
    except (ValueError, RuntimeWarning, np.linalg.LinAlgError, OverflowError, PerfectSeparationError):
        return dict(status='MODEL_DIAGNOSTIC_FAILURE', family=family, reason='UNUSABLE_EXPLORATORY_FIT')


def _analyze(tables, loaded, records, validation):
    frozen, roles, registry = loaded['frozen'], loaded['roles'], loaded['registry']
    baseline, members, features = (frozen[key] for key in ('baseline_raw', 'membership', 'feature_sets'))
    candidates, ledger = _candidates(tables, baseline, records)
    dependencies = _dependencies(features, registry, validation)
    roles = {(row['representation_id'], row['source_column']): row for row in roles.to_dict('records')}
    registry_sources = set(registry.get('source_column', ()))
    baseline_fields = baseline.reindex(columns=KEYS + ['clinical_baseline_date', 'demo__age_at_baseline', 'demo__sex'])
    membership = {rep: hard_members(members, rep).merge(baseline_fields, on='patient_id', validate='one_to_one')
                  for rep in loaded['manifest']['eligible_representation_ids']}
    outputs = {name: [] for name in TABLE_COLUMNS}
    definitions, rf_tested = {}, set()
    for source in OUTCOMES:
        selected, rejected, definition = _outcome(candidates, ledger, source)
        outputs['exclusions'].extend(_safe_exclusions(rejected, source))
        numeric = source == RF_VALUE
        rf = source in {RF_VALUE, RF_TEXT}
        for rep in REPS:
            label = dict(representation_id=rep, outcome=source)
            model = dict(**label, hypothesis_family='rheumatoid_factor' if rf else source)
            if rep not in membership:
                outputs['coverage'].append(dict(**label, status='REPRESENTATION_UNAVAILABLE'))
                outputs['models'].append(dict(**model, status='REPRESENTATION_UNAVAILABLE'))
                continue
            frame = membership[rep].merge(selected, on='patient_id', how='left', validate='one_to_one')
            provisional = not frame.structural_stability_status.eq('stable').all()
            topology = ('R2_EXPLORATORY_UNSTABLE' if provisional else 'R2_EXPLORATORY') if rep == 'R2' else (
                'PROVISIONAL_COMMUNITIES' if provisional else 'FROZEN_COMMUNITIES')
            label['topology_status'] = model['topology_status'] = topology
            profiles = _profiles(frame, label, numeric)
            outputs['community_profiles'].extend(profiles)
            coverage = _coverage(frame, rejected, label, profiles)
            outputs['coverage'].append(coverage)

            if not _independence(rep, source, roles, features, registry_sources, dependencies, validation):
                model['status'] = CIRCULAR
            elif rf and rep in rf_tested:
                model['status'] = 'ALTERNATIVE_RF_ENCODING_DESCRIPTIVE_ONLY'
            else:
                support, prepared = _model_support(frame, numeric)
                if coverage['status'] == 'SUPPRESSED':
                    support.pop('n', None)
                model.update(support)
                if prepared is None:
                    model['status'] = 'INSUFFICIENT_SUPPORT'
                else:
                    if rf:
                        rf_tested.add(rep)
                    model.update(_estimate(prepared, numeric))
            outputs['models'].append(model)

        if definition and any(row['outcome'] == source and row.get('n_measured', 0) >= CELL
                              for row in outputs['community_profiles']):
            evidence = selected.date_evidence.value_counts()
            # Raw names can contain unstructured source text. Record a stable
            # fingerprint of the exact definition and the numeric unit only.
            definitions[source] = dict(original_definition_sha256=cal.fingerprint(definition),
                                       unit=definition[-1] if numeric else None,
                                       collection_evidence=sorted(evidence.index[evidence.ge(CELL)]))

    tested = [row for row in outputs['models'] if row['status'] == 'EXPLORATORY_ESTIMATED']
    if tested:
        adjusted = multipletests([row['p_incremental'] for row in tested], method='fdr_bh')[1]
        for row, q in zip(tested, adjusted):
            row['q_bh'] = float(q)
    return outputs, definitions


def _empty_outputs():
    outputs = {name: [] for name in TABLE_COLUMNS}
    pairs = [dict(representation_id=rep, outcome=source, status='DATA_UNAVAILABLE')
             for rep in REPS for source in OUTCOMES]
    outputs['coverage'] = pairs
    outputs['models'] = pairs
    return outputs


def _figures(folder, profiles):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    folder.mkdir(mode=0o700)
    names = []
    for source in OUTCOMES:
        subset = profiles.loc[profiles.outcome.eq(source) & profiles.n_measured.fillna(0).gt(0)]
        if subset.empty:
            continue
        numeric = source == RF_VALUE
        measure = 'median' if numeric else 'positive_fraction'
        usable = subset.loc[subset[measure].notna()]
        plot = usable if not usable.empty else subset
        fig = Figure(figsize=(8, max(2.5, len(plot) * .24 + 1.5)), layout='constrained')
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        positions = np.arange(len(plot))
        if usable.empty:
            ax.barh(positions, plot.measurement_fraction.astype(float))
            ax.set(xlabel='Fraction with measurement', xlim=(0, 1))
        elif numeric:
            center = plot['median'].astype(float)
            ax.errorbar(center, positions, xerr=[center - plot.q1.astype(float), plot.q3.astype(float) - center],
                        fmt='o', capsize=3)
            ax.set(xlabel='Median and interquartile range (original unit)')
        else:
            ax.barh(positions, plot.positive_fraction.astype(float))
            ax.set(xlabel='Fraction positive among measured', xlim=(0, 1))
        ax.set_yticks(positions, plot.representation_id + ' / ' + plot.community_id.astype(str))
        ax.invert_yaxis()
        ax.set_title(source.removeprefix('lab__').replace('__', ' / ').replace('_', ' '))
        filename = source + '.png'
        fig.savefig(folder / filename, dpi=130)
        names.append(filename)
    if not names:
        fig = Figure(figsize=(6, 2.2), layout='constrained')
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.set_title('Exploratory N3 measurement coverage')
        ax.text(.5, .5, 'No displayable community coverage', transform=ax.transAxes, ha='center')
        ax.set_axis_off()
        fig.savefig(folder / 'coverage.png', dpi=130)
        names.append('coverage.png')
    return names


def _readme(summary):
    provenance = json.dumps(summary['provenance'], sort_keys=True, indent=2)
    return f"""# Graph03 N3 exploratory results

Status **{summary['status']}**. The analysis uses the frozen Graph02 baseline and supported hard community assignments from the compatible Graph02/03 releases. Exploratory findings require separate confirmatory evaluation under the institutional workflow.

## Measurements

- The outcome definitions are general ANA, ANA HEp-2, qualitative RF, quantitative RF and qualitative urine protein. Profiles retain ANA and HEp-2 separately.
- The day written in EDA Step20 `lab_date` must agree with specimen collection evidence. The cutoff is the frozen baseline day. Days 1 through 365 before baseline qualify, inclusive.
- Step10 keys and context corroborate the official patient and source episode. Step01 and context selections can identify date conflicts. Raw and intermediate BTRIS corroboration is recorded when an exact order and observation link is available. Order ID is order provenance.
- The latest eligible calendar day is used per patient and outcome. Identical records collapse. Distinct tied observations on that day exclude the patient/outcome. Recorded assay and unit are part of the exact test definition. The most frequent definition among selected patients wins, with a lexical tie-break. The definition fingerprint covers the original order name, observation name, assay and unit. Missing assay or qualitative unit stays missing.
- Qualitative coding accepts positive or detected and negative or not detected, after case and whitespace normalization. Other categories, invalid results and censored results are excluded.

## Models

- A includes available verified age at baseline, sex and five-year baseline era. B adds frozen community on the same complete cases. Constant covariates have no model term. Adjustment is partial and includes no research-protocol term.
- Models require at least 30 complete cases in at least two communities with at least eight each, full rank and at least ten residual degrees of freedom. Binary models require at least five events and five nonevents per parameter, at least five of each per modeled community and an absence of diagnosed separation.
- The joint HC3 Wald comparison tests the community terms. Contrasts are adjusted odds ratios or mean differences with 95% confidence intervals. BH-FDR covers all estimable representation–outcome tests. Each representation contributes at most one RF test with priority given to the numeric encoding when it passes support and independence checks.
- Uncertain lineage or potential proxies yield descriptives only. R2 is exploratory. Testing indications, missing data, residual confounding, test-definition selection and uncertainty in the original community assignments limit interpretation.

## Disclosure

Counts of one through four, corresponding aggregates, and complementary totals are withheld. Small communities have an unlabeled suppressed row. Missing-reason partitions are released only when all reported cells meet the threshold. The figures use the same masked profiles as the CSVs. The output contains aggregate data and is restricted to its owner.

## Provenance

The frozen artifacts and source files were hashed before publication. Hashes for optional BTRIS files are grouped by source role.
```json
{provenance}
```
"""


def _write_run(staging, target, base, outputs, definitions):
    filenames = {name: target / (name + '.csv') for name in TABLE_COLUMNS}
    frames = {name: pd.DataFrame(outputs[name], columns=columns.split())
              for name, columns in TABLE_COLUMNS.items()}
    for name, frame in frames.items():
        frame.to_csv(staging / filenames[name].name, index=False)
    figures = _figures(staging / 'figures', frames['community_profiles'])
    n_outcomes = len(definitions)
    n_models = sum(row['status'] == 'EXPLORATORY_ESTIMATED' for row in outputs['models'])
    status = 'NO_ELIGIBLE_DATA' if n_outcomes == 0 else 'EXPLORATORY_PARTIAL'
    if (n_outcomes == len(OUTCOMES) and n_models and not base['data_issues']
            and not frames['coverage'].status.eq('REPRESENTATION_UNAVAILABLE').any()):
        status = 'EXPLORATORY_COMPLETE'
    summary = dict(**base, status=status, output_dir=str(target), run_id=target.name,
                   n_outcomes_processed=0 if outputs['coverage'][0]['status'] == 'DATA_UNAVAILABLE' else len(OUTCOMES),
                   n_outcomes_with_data=n_outcomes, n_model_pairs_estimable=n_models,
                   summary=str(target / 'summary.json'), readme=str(target / 'README.md'),
                   figures=[str(target / 'figures' / filename) for filename in figures],
                   outcome_definitions=definitions, **{name: str(path) for name, path in filenames.items()})
    (staging / 'summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2) + '\n')
    (staging / 'README.md').write_text(_readme(summary))


def _source_hashes(paths):
    return {role: digest(paths[role]) if paths[role].is_file() else None for role in LAB_INPUTS}


def _btris_hashes(paths):
    return {role: cal.file_hashes(paths[role]) for role in cal.BTRIS_ROLES}


def run_exploratory_auto(root):
    """Discover immutable sources and publish a private, aggregate-only run."""
    root = Path(root).resolve()
    issues = []
    base = dict(mode=MODE, graph02_run=None, graph03_run=None, data_issues=issues, limitations=LIMITATIONS)
    try:
        data, private, paths = _settings(root)
        _private_directory(private)
        with _locked(private / 'exploratory'):
            phasec = root / CONFIGS / PHASEC
            validation = root / CONFIGS / VALIDATION
            args = SimpleNamespace(graph03_run=None, graph02_release=None, strict=False, config=phasec)
            frozen_roots = []
            try:
                graph03, release, _, selection = resolve_graph03_dependencies(
                    args, root, phase='C', data_root=data, quiet=True, verify_inventory=True)
                frozen_roots = [release.parent, graph03]
                if any(private.is_relative_to(path) or path.is_relative_to(private) for path in frozen_roots):
                    raise ValueError('EXPLORATORY_OUTPUT_OVERLAPS_FROZEN_INPUT')
                base['graph02_run'], base['graph03_run'] = [record['run_id'] for record in selection]
            except RuntimeError:
                issues.append(dict(status='DATA_UNAVAILABLE', artifact='compatible_graph02_graph03_releases'))
            frozen_sha = _snapshot(frozen_roots)
            sources_sha = _source_hashes(paths)
            btris_sha = _btris_hashes(paths)
            configs = {file.name: digest(file) for file in (phasec, validation, root / CONFIGS / 'n3_auto.yaml')}
            provenance = dict(
                frozen_manifest_sha256={
                    path.name: sha for path, sha in frozen_sha.items() if path.name in {
                        '02_graph_release_manifest.json', '03_graph_results_summary.json'}},
                source_sha256=sources_sha,
                btris_sha256={role: cal.fingerprint(files) if files else None for role, files in btris_sha.items()},
                config_sha256=configs,
                implementation_sha256=digest(Path(__file__)),
            )
            base['provenance'] = provenance

            def verify():
                if (frozen_sha != _snapshot(frozen_roots) or sources_sha != _source_hashes(paths)
                        or btris_sha != _btris_hashes(paths)
                        or any(sha != digest(root / CONFIGS / name) for name, sha in configs.items())):
                    raise ValueError('EXPLORATORY_INPUT_CHANGED_DURING_RUN')

            def build(staging, target):
                tables = _inputs(paths, issues)
                outputs, definitions = _empty_outputs(), {}
                if len(tables) == len(LAB_INPUTS) and frozen_roots:
                    try:
                        loaded = _read_frozen(dict(graph03_run=graph03, graph02_release=release, baseline=None),
                                              yaml.safe_load(phasec.read_text()))
                    except READ_ERRORS:
                        issues.append(dict(status='DATA_UNAVAILABLE', artifact='frozen_graph02_graph03_inventory'))
                    else:
                        integrated = loaded['manifest']['source_inputs'].get('integrated')
                        if integrated and integrated['sha256'] != sources_sha['step10']:
                            issues.append(dict(status='DATA_UNAVAILABLE', artifact='step10',
                                               reason='FROZEN_GRAPH02_INPUT_SHA_MISMATCH'))
                        else:
                            btris_files = {role: cal.btris_files(paths[role]) for role in cal.BTRIS_ROLES}
                            btris = _btris_records(btris_files, issues)
                            outputs, definitions = _analyze(tables, loaded, btris, yaml.safe_load(validation.read_text()))
                _write_run(staging, target, base, outputs, definitions)

            cache_key = dict(provenance=provenance, graph02_run=base['graph02_run'], graph03_run=base['graph03_run'],
                             frozen_inventory_sha256=cal.fingerprint(sorted(frozen_sha.values())))
            output = _cached(private / 'exploratory' / 'runs', cache_key, build, verify, verify_reuse=True)
            summary = json.loads((output / 'summary.json').read_text())
            keys = ('status', 'mode', 'graph02_run', 'graph03_run', 'n_outcomes_with_data', 'n_model_pairs_estimable',
                    'output_dir', 'summary', 'models', 'figures', 'limitations', 'data_issues')
            return {key: summary[key] for key in keys}
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        # Publish controlled error codes, never source paths or values from exceptions.
        token = str(exc)
        reason = token if re.fullmatch(r'(?:EXPLORATORY_[A-Z_]+|UNSAFE_AUTO_[A-Z_]+|AUTO_RUN_IN_PROGRESS)', token) else (
            'EXPLORATORY_RUN_UNAVAILABLE')
        return dict(**base, status='EXPLORATORY_PARTIAL', n_outcomes_with_data=0, n_model_pairs_estimable=0,
                    output_dir=None, summary=None, models=None, figures=[], reason=reason)

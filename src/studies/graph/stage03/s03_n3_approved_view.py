"""Restricted derived views; source values and frozen populations are preserved."""
import re

import numpy as np
import pandas as pd

from . import s03_n3_calendar_day as cal

KEYS = ['patient_id', 'clinical_episode_id']


def validate_baseline(baseline):
    if (not set(KEYS + ['clinical_baseline_date', 'clinical_anchor_date']) <= set(baseline)
            or baseline[KEYS].isna().any().any() or baseline.patient_id.duplicated().any()):
        raise ValueError('OFFICIAL_BASELINE_IDENTITY_REQUIRED')
    if 'is_clinical_baseline' not in baseline or not baseline.is_clinical_baseline.eq(True).all():
        raise ValueError('OFFICIAL_BASELINE_FLAG_REQUIRED')
    if 'clinical_baseline_episode_id' in baseline and not baseline.clinical_episode_id.eq(
            baseline.clinical_baseline_episode_id).all():
        raise ValueError('OFFICIAL_BASELINE_EPISODE_MISMATCH')
    dates = pd.to_datetime(baseline.clinical_baseline_date, errors='coerce', utc=True)
    anchors = pd.to_datetime(baseline.clinical_anchor_date, errors='coerce', utc=True)
    if dates.isna().any() or not dates.eq(anchors).all():
        raise ValueError('OFFICIAL_BASELINE_DATE_MISMATCH')


def protocol_view(baseline, context, review):
    """Only a reviewed research-protocol assignment can populate demo__protocol.

    This v1 supports a reviewed single assignment, with an explicit code
    dictionary, one context row per patient/episode and an as-of assignment date.
    Multi-protocol adjudication needs a separate reviewed upstream table; no
    delimiter guessing, first/last selection or constant imputation is allowed.
    """
    validate_baseline(baseline)
    required = {'equivalent_to_research_protocol': True, 'canonical_target': 'demo__protocol',
                'assignment_scope': 'patient_research_protocol',
                'multiple_protocol_rule': 'reject', 'code_dictionary_verified': True,
                'assignment_date_verified': True, 'dates_normalized_to_utc': True}
    if not review or any((review.get(k) is not True if v is True else review.get(k) != v)
                         for k, v in required.items()):
        raise ValueError('PROTOCOL_SEMANTIC_REVIEW_REQUIRED')
    source, date = review.get('source_column'), review.get('assignment_date_column')
    dictionary = review.get('code_dictionary')
    if (not isinstance(source, str) or source in KEYS or not isinstance(date, str) or date in KEYS or source == date or
            not isinstance(dictionary, dict) or not dictionary or
            any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or
                not v.strip() or any(c in k + v for c in '|;,') for k, v in dictionary.items())):
        raise ValueError('PROTOCOL_DICTIONARY_REQUIRED')
    if not set(KEYS + [source, date]) <= set(context):
        raise ValueError('PROTOCOL_SOURCE_OR_DATE_UNAVAILABLE')
    if context[KEYS].isna().any().any() or context.duplicated(KEYS).any():
        raise ValueError('PROTOCOL_CONTEXT_NOT_ONE_TO_ONE')
    # A claimed patient assignment cannot vary by episode in this contract.
    codes = context[source].astype('string')
    if codes.str.contains(r'[|;,]', regex=True, na=False).any():
        raise ValueError('MULTIPLE_PROTOCOLS_REQUIRE_ADJUDICATION')
    if context.assign(_code=codes).groupby('patient_id')._code.nunique().gt(1).any():
        raise ValueError('MULTIPLE_PROTOCOLS_REQUIRE_ADJUDICATION')
    if not set(codes.dropna()) <= set(dictionary):
        raise ValueError('PROTOCOL_CODE_OUTSIDE_REVIEWED_DICTIONARY')
    selected = baseline[KEYS].merge(context[KEYS + [source, date]], on=KEYS,
                                   how='left', validate='one_to_one', indicator=True)
    if not selected['_merge'].eq('both').all():
        raise ValueError('PROTOCOL_BASELINE_EPISODE_UNMATCHED')
    assigned = pd.to_datetime(selected[date], errors='coerce', utc=True)
    cutoff = pd.to_datetime(baseline.clinical_baseline_date, errors='coerce', utc=True).reset_index(drop=True)
    present = selected[source].notna()
    if (present & (assigned.isna() | assigned.gt(cutoff))).any():
        raise ValueError('PROTOCOL_ASSIGNMENT_AFTER_BASELINE_OR_UNDATED')
    mapped = selected[source].astype('string').map(dictionary)
    if not mapped.notna().any():
        raise ValueError('PROTOCOL_ALL_MISSING')
    result = baseline.copy(deep=True)
    if 'demo__protocol' in result:
        existing = result['demo__protocol'].reset_index(drop=True).astype('string')
        conflict = existing.notna() & (mapped.isna() | ~existing.eq(mapped).fillna(False))
        if conflict.any():
            raise ValueError('PROTOCOL_DISAGREES_WITH_FROZEN_CANONICAL')
    result['demo__protocol'] = mapped.to_numpy()
    pd.testing.assert_frame_equal(result[KEYS], baseline[KEYS])
    return result


def measurement_values(baseline, measurements, source, review):
    """Use actual, reviewed pre-baseline instrument times and exact assay/unit.

    Measurements use the official baseline episode key. Select latest eligible
    pre-baseline time within the approved lookback; discordant ties fail closed.
    No administrative/anchor date is substituted for an instrument date.
    """
    required = {'instrument_date_verified': True, 'date_semantics': 'instrument_measurement',
                'selection_rule': 'latest_prebaseline', 'duplicate_rule': 'identical_only',
                'baseline_window_verified': True, 'source_semantics_verified': True,
                'missingness_and_indication_reviewed': True,
                'timezone_reviewed': True, 'dates_normalized_to_utc': True}
    if not review or any((review.get(k) is not True if v is True else review.get(k) != v)
                         for k, v in required.items()):
        raise ValueError('MEASUREMENT_SOURCE_DATE_OR_SEMANTICS_REVIEW_REQUIRED')
    lookback = review.get('lookback_days')
    kind = review.get('type')
    if (type(lookback) is not int or lookback < 0 or kind not in {'numeric', 'binary'} or
            not isinstance(review.get('assay_id'), str) or not review['assay_id'] or
            not isinstance(review.get('unit'), str) or not review['unit']):
        raise ValueError('MEASUREMENT_TYPE_ASSAY_UNIT_OR_WINDOW_REQUIRED')
    columns = KEYS + ['source_column', 'value', 'measurement_date', 'assay_id', 'unit', 'sample_id']
    if not set(columns) <= set(measurements):
        raise ValueError('MEASUREMENT_SCHEMA_UNAVAILABLE')
    selected_source = measurements.loc[measurements.source_column.eq(source)]
    if ('source_artifact_status' in selected_source and
            selected_source.source_artifact_status.eq('SOURCE_CANDIDATE_UNAPPROVED').any()):
        raise ValueError('UNAPPROVED_MEASUREMENT_CANDIDATE')
    rows = selected_source[columns].copy()
    if rows[KEYS].isna().any().any():
        raise ValueError('MEASUREMENT_IDENTITY_MISSING')
    if rows.sample_id.astype('string').fillna('').str.strip().eq('').any():
        raise ValueError('MEASUREMENT_SAMPLE_IDENTITY_MISSING')
    # Never combine different ANA assays, units or RF encodings under one source.
    if (not rows.assay_id.eq(review['assay_id']).fillna(False).all()
            or not rows.unit.eq(review['unit']).fillna(False).all()):
        raise ValueError('NON_EQUIVALENT_ASSAY_OR_UNIT')
    rows = rows.merge(baseline[KEYS + ['clinical_baseline_date']], on=KEYS,
                      how='inner', validate='many_to_one')
    rows['_time'] = pd.to_datetime(rows.measurement_date, errors='coerce', utc=True)
    cutoff = pd.to_datetime(rows.clinical_baseline_date, errors='coerce', utc=True)
    if (rows.value.notna() & rows['_time'].isna()).any():
        raise ValueError('INSTRUMENT_DATE_MISSING_OR_INVALID')
    # Post-baseline measurements stay ineligible, even when closer to an anchor.
    age = (cutoff - rows['_time']).dt.total_seconds() / 86400
    rows = rows.loc[age.between(0, lookback)].drop_duplicates(columns)
    if rows.empty:
        raise ValueError('NO_PREBASELINE_INSTRUMENT_MEASUREMENTS')
    latest = rows.groupby(KEYS)['_time'].transform('max')
    rows = rows.loc[rows['_time'].eq(latest)]
    if rows.duplicated(KEYS).any():
        raise ValueError('REPEATED_MEASUREMENT_TIE_REQUIRES_ADJUDICATION')
    selected = baseline[KEYS].merge(rows[KEYS + ['value']], on=KEYS, how='left', validate='one_to_one')
    values = selected.value
    if kind == 'numeric':
        numeric = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
        if (values.notna() & numeric.isna()).any():
            raise ValueError('UNPARSED_NUMERIC_MEASUREMENT')
        return pd.Series(numeric.to_numpy(), index=baseline.index), kind
    convention = review.get('category_convention', {})
    groups = {k: set(convention.get(k, [])) for k in ['positive', 'negative', 'unknown', 'not_tested']}
    if (not groups['positive'] or not groups['negative'] or
            any(groups[a] & groups[b] for a in groups for b in groups if a != b)):
        raise ValueError('BINARY_CONVENTIONS_UNREVIEWED_OR_OVERLAPPING')
    tokens = values.astype('string')
    if not set(tokens.dropna()) <= set().union(*groups.values()):
        raise ValueError('UNREVIEWED_MEASUREMENT_CATEGORY')
    output = pd.Series(np.nan, index=baseline.index)
    output.loc[tokens.isin(groups['positive']).to_numpy()] = 1.
    output.loc[tokens.isin(groups['negative']).to_numpy()] = 0.
    return output, kind


def validate_calendar_day_review(review):
    required = {
        'measurement_contract': cal.CONTRACT, 'time_resolution': 'calendar_day',
        'identity_contract': cal.IDENTITY, 'date_origin': cal.DATE_ORIGIN,
        'selection_rule': 'latest_prebaseline_calendar_day', 'duplicate_rule': 'identical_only',
    }
    approvals = (
        'calendar_day_origin_verified', 'baseline_calendar_day_verified', 'baseline_window_verified',
        'selection_and_duplicates_reviewed', 'test_definition_verified', 'source_unit_and_scale_verified',
        'unidentified_assay_and_specimen_risk_reviewed', 'missingness_and_indication_reviewed',
    )
    if (not isinstance(review, dict) or any(review.get(key) != value for key, value in required.items())
            or any(review.get(key) is not True for key in approvals)
            or any(key in review for key in ('assay_id', 'sample_id', 'timezone_reviewed', 'dates_normalized_to_utc'))):
        raise ValueError('CALENDAR_DAY_PROVENANCE_AND_SEMANTICS_REVIEW_REQUIRED')
    cal.validate_window(review)
    if (review.get('type') not in {'numeric', 'binary'}
            or any(not isinstance(review.get(key), str) or not review[key].strip()
                   for key in ('test_definition_id', 'unit'))):
        raise ValueError('CALENDAR_DAY_TEST_DEFINITION_UNIT_OR_TYPE_REQUIRED')
    if review['type'] == 'numeric':
        return None
    categories = review.get('category_convention')
    names = ('positive', 'negative', 'unknown', 'not_tested')
    if (not isinstance(categories, dict) or set(categories) != set(names)
            or any(not isinstance(group, list) or any(not isinstance(token, str) or
                not token.strip() for token in group) for group in categories.values())):
        raise ValueError('BINARY_CONVENTIONS_UNREVIEWED_OR_OVERLAPPING')
    groups = {name: set(categories[name]) for name in names}
    if (not groups['positive'] or not groups['negative'] or
            sum(map(len, groups.values())) != len(set().union(*groups.values()))):
        raise ValueError('BINARY_CONVENTIONS_UNREVIEWED_OR_OVERLAPPING')
    if any(re.fullmatch(cal.ADMINISTRATIVE_RESULT, token) for token in groups['positive'] | groups['negative']):
        raise ValueError('REVIEWED_ADMINISTRATIVE_TOKEN_NOT_MISSING')
    return groups


def measurement_values_calendar_day(baseline, measurements, source, review, *, source_review_sha256):
    """Apply a signed calendar-day contract to a pinned, reviewed artifact."""
    groups = validate_calendar_day_review(review)
    if source not in cal.LAB_SOURCES:
        raise ValueError('CALENDAR_DAY_LAB_SOURCE_REQUIRED')
    if not set(cal.CORE) <= set(measurements):
        raise ValueError('CALENDAR_DAY_MEASUREMENT_SCHEMA_UNAVAILABLE')
    rows = measurements.loc[measurements.source_column.eq(source), cal.CORE].copy()
    if rows.empty:
        raise ValueError('NO_PREBASELINE_INSTRUMENT_MEASUREMENTS')
    if not rows.source_artifact_status.eq('SOURCE_REVIEWED').fillna(False).all():
        raise ValueError('UNAPPROVED_MEASUREMENT_CANDIDATE')
    if not cal.SHA256.fullmatch(str(source_review_sha256)):
        raise ValueError('SIGNED_CALENDAR_DAY_SOURCE_REVIEW_REQUIRED')
    requirements = {
        'source_review_sha256': source_review_sha256, 'measurement_contract_sha256': cal.fingerprint(review),
        'review_contract_version': cal.REVIEW_VERSION, 'date_origin': cal.DATE_ORIGIN, 'date_resolution': 'calendar_day',
        'test_definition_id': review['test_definition_id'], 'unit': review['unit'],
    }
    if (any(not rows[key].eq(value).fillna(False).all() for key, value in requirements.items())
            or rows[['assay_id', 'sample_id']].notna().any().any()
            or not rows.observation_sha256.astype('string').str.fullmatch(cal.SHA256, na=False).all()
            or any(rows[key].astype('string').fillna('').str.strip().eq('').any()
                   for key in KEYS + ['source_episode_id', *cal.ORIGINAL_IDENTITY, 'value'])):
        raise ValueError('CALENDAR_DAY_ARTIFACT_OUTSIDE_SIGNED_SCOPE')
    validate_baseline(baseline)
    cutoffs = baseline[KEYS].assign(_cutoff=cal.calendar_days(baseline.clinical_baseline_date, allow_time=True))
    anchors = cal.calendar_days(baseline.clinical_anchor_date, allow_time=True)
    if cutoffs._cutoff.isna().any() or not cutoffs._cutoff.eq(anchors).fillna(False).all():
        raise ValueError('OFFICIAL_BASELINE_CALENDAR_DAY_MISMATCH')
    rows = rows.merge(cutoffs, on=KEYS, how='left', validate='many_to_one')
    rows['collection_calendar_day'] = cal.calendar_days(rows.collection_calendar_day)
    if rows._cutoff.isna().any() or rows.collection_calendar_day.isna().any():
        raise ValueError('CALENDAR_DAY_OR_BASELINE_IDENTITY_INVALID')
    ages = cal.calendar_age(rows._cutoff, rows.collection_calendar_day)
    identity = [field for field in cal.CORE if field not in KEYS]
    latest, discordant = cal.latest_calendar_rows(
        rows.loc[ages.between(review['minimum_days_before_baseline'], review['lookback_days'])], identity)
    chosen = latest.loc[~discordant].drop_duplicates(KEYS + identity)
    if chosen.empty:
        raise ValueError('AMBIGUOUS_SAME_DAY' if discordant.any() else 'NO_PREBASELINE_INSTRUMENT_MEASUREMENTS')
    values = baseline[KEYS].merge(chosen[KEYS + ['value']], on=KEYS, how='left', validate='one_to_one').value
    if groups is None:
        numeric = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
        if (values.notna() & numeric.isna()).any():
            raise ValueError('UNPARSED_NUMERIC_MEASUREMENT')
        return pd.Series(numeric.to_numpy(dtype=float, na_value=np.nan), index=baseline.index), 'numeric'
    tokens = values.astype('string')
    if not set(tokens.dropna()) <= set().union(*groups.values()):
        raise ValueError('UNREVIEWED_MEASUREMENT_CATEGORY')
    output = pd.Series(np.nan, index=baseline.index)
    output.loc[tokens.isin(groups['positive']).to_numpy()] = 1.
    output.loc[tokens.isin(groups['negative']).to_numpy()] = 0.
    return output, 'binary'

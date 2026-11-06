"""Restricted derived views; source values and frozen populations are preserved."""
import numpy as np
import pandas as pd

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
    rows = measurements.loc[measurements.source_column.eq(source), columns].copy()
    if rows[KEYS].isna().any().any():
        raise ValueError('MEASUREMENT_IDENTITY_MISSING')
    # Never combine different ANA assays, units or RF encodings under one source.
    if not rows.assay_id.eq(review['assay_id']).all() or not rows.unit.eq(review['unit']).all():
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

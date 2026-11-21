"""The shared retrospective Graph03 window, independent of clinical approval."""
from dataclasses import asdict, dataclass
from datetime import date, datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from dateutil.relativedelta import relativedelta
import pandas as pd
import yaml

VERSION = 'graph03_baseline_window_v2'
MISMATCH = 'BASELINE_WINDOW_POLICY_VERSION_MISMATCH'
PRIMARY = 'BASELINE_0_TO_6M_PRIMARY'
PREBASELINE = 'PREBASELINE'
BASELINE_DAY0 = 'BASELINE_DAY0'
BASELINE_POST_DAY0_TO_6M = 'BASELINE_POST_DAY0_TO_6M'
AFTER_BASELINE_WINDOW = 'AFTER_BASELINE_WINDOW'
DATE_UNRESOLVED = 'DATE_UNRESOLVED'
DAY_BINS = ('PRE', 'DAY0', 'DAY1_30', 'DAY31_90', 'DAY91_TO_6M', 'AFTER_6M', 'UNKNOWN')
FROZEN_SNAPSHOT = 'FROZEN_GRAPH02_BASELINE_SNAPSHOT'
WINDOW_SNAPSHOT = 'GRAPH03_BASELINE_0_TO_6_CALENDAR_MONTHS'
WARNING = ('Mediciones colectadas hasta seis meses después del índice. Caracterización retrospectiva '
           'condicionada a las comunidades congeladas. Las mediciones pueden no estar disponibles en el día 0.')
POLICY_PATH = Path(__file__).resolve().parents[1] / 'configs/baseline_window_03.yaml'

ISO_STAMP = re.compile(
    r'(\d{4}-\d{2}-\d{2})(?:[T ](?:[01]\d|2[0-3]):[0-5]\d'
    r'(?::[0-5]\d(?:\.\d{1,9})?)?(?:Z|[+-](?:[01]\d|2[0-3]):?[0-5]\d)?)?')
US_STAMP = re.compile(
    r'(\d{1,2}/\d{1,2}/\d{4})(?:[T ]\d{1,2}:[0-5]\d(?::[0-5]\d)?(?: [AP]M)?)?')


@dataclass(frozen=True)
class BaselineWindowPolicy:
    policy_version: str
    anchor: str
    start_offset_days: int
    duration_calendar_months: int
    include_start: bool
    include_end: bool
    selection_primary: str
    date_semantics: str
    primary_date_provenance: str

    def __post_init__(self):
        if self.policy_version != VERSION or (self.start_offset_days, self.duration_calendar_months) != (0, 6):
            raise ValueError(MISMATCH)
        if self.include_start is not True or self.include_end is not True:
            raise ValueError(MISMATCH)

    @property
    def sha256(self):
        payload = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode()).hexdigest()

    def metadata(self):
        return dict(window_policy_version=self.policy_version, window_policy_sha256=self.sha256,
                    baseline_anchor_source=self.anchor, window_start_end_semantics='closed_calendar_days',
                    selection_rule=self.selection_primary, source_provenance_tier=self.primary_date_provenance)


@lru_cache(maxsize=1)
def policy():
    return BaselineWindowPolicy(**yaml.safe_load(POLICY_PATH.read_text()))


def require_policy(settings):
    if (not isinstance(settings, dict) or settings.get('window_policy_version') != policy().policy_version
            or any(key in settings for key in ('lookback_days', 'minimum_days_before_baseline', 'prebaseline_primary'))):
        raise ValueError(MISMATCH)


@lru_cache(maxsize=4096)
def _parse_day(value):
    iso = ISO_STAMP.fullmatch(value)
    us = US_STAMP.fullmatch(value)
    try:
        if iso:
            return date.fromisoformat(iso[1])
        if us:
            return datetime.strptime(us[1], '%m/%d/%Y').date()
    except ValueError:
        return None
    return None


def calendar_day(value):
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return _parse_day(str(value).strip())


@lru_cache(maxsize=4096)
def _end(day):
    return day + relativedelta(months=policy().duration_calendar_months)


def baseline_window_end(t0):
    day = calendar_day(t0)
    return _end(day) if day is not None else None


def classify_measurement_day(t0, day):
    start, observed = calendar_day(t0), calendar_day(day)
    if start is None or observed is None:
        return DATE_UNRESOLVED
    if observed < start:
        return PREBASELINE
    if observed == start:
        return BASELINE_DAY0
    return BASELINE_POST_DAY0_TO_6M if observed <= _end(start) else AFTER_BASELINE_WINDOW


def temporally_eligible(t0, day):
    return classify_measurement_day(t0, day) in (BASELINE_DAY0, BASELINE_POST_DAY0_TO_6M)


def is_eligible_baseline_measurement(t0, day, *, identity_verified, collection_verified, assay_comparable,
                                     result_valid, provenance_verified):
    return all((identity_verified, collection_verified, assay_comparable, result_valid, provenance_verified,
                temporally_eligible(t0, day)))


def calendar_windows(anchors, measurements):
    """Return aligned temporal classes, mutually exclusive bins and signed day offsets."""
    if not anchors.index.equals(measurements.index):
        raise ValueError('BASELINE_WINDOW_UNALIGNED_INPUTS')
    start = pd.to_datetime(anchors.map(calendar_day))
    observed = pd.to_datetime(measurements.map(calendar_day))
    end = pd.to_datetime(anchors.map(baseline_window_end))
    offset = (observed - start).dt.days.astype('Int64')
    classes = pd.Series(DATE_UNRESOLVED, index=anchors.index, dtype='string')
    bins = pd.Series('UNKNOWN', index=anchors.index, dtype='string')
    for mask, category, label in (
        (offset.lt(0), PREBASELINE, 'PRE'),
        (offset.eq(0), BASELINE_DAY0, 'DAY0'),
        (observed.gt(end), AFTER_BASELINE_WINDOW, 'AFTER_6M'),
    ):
        classes.loc[mask] = category
        bins.loc[mask] = label
    inside = offset.gt(0) & observed.le(end)
    classes.loc[inside] = BASELINE_POST_DAY0_TO_6M
    bins.loc[inside & offset.le(30)] = 'DAY1_30'
    bins.loc[inside & offset.between(31, 90)] = 'DAY31_90'
    bins.loc[inside & offset.ge(91)] = 'DAY91_TO_6M'
    return pd.DataFrame(dict(temporal_class=classes, window_bin=bins, baseline_offset_days=offset))


def select_calendar_rows(rows, keys, *, value='value', latest=False):
    """Keep the chosen date per key and mark conflicting values on that date."""
    day = rows.groupby(keys, dropna=False).collection_calendar_day.transform('max' if latest else 'min')
    chosen = rows.loc[rows.collection_calendar_day.eq(day)].copy()
    conflicting = chosen.groupby(keys, dropna=False)[value].transform('nunique').gt(1)
    return chosen, conflicting

"""Versioned episode-name contract, with exact Step 00 identity and structure."""
import hashlib
import json
import pandas as pd

VERSION = '08-medication-episode-features-1.0'
PRODUCER_VERSION = '08-medications-1.0'
KEYS = ['patient_id', 'clinical_episode_id']
STRUCTURE = KEYS + ['clinical_anchor_date', 'episode_start_date', 'episode_end_date', 'clinical_visit']
COLUMNS = ['medication__' + s for s in (
    'ctdb_reported_names', 'btris_order_within_episode_names',
    'btris_order_before_episode_1y_names', 'btris_order_after_episode_1y_names',
    'btris_order_nearby_1y_names', 'btris_consent_only_review_names',
    'chart_review_candidate_names')]
# The specification lists seven name fields (eight including the episode-key group).

def key_set(frame):
    if not set(KEYS).issubset(frame):
        raise ValueError('missing episode keys')
    keys = frame[KEYS].astype('string')
    if keys.isna().any().any() or keys.apply(lambda c: c.str.strip().eq('').any()).any() or keys.duplicated().any():
        raise ValueError('null/empty/duplicate episode keys')
    return set(keys.itertuples(index=False, name=None))


def spine_signature(spine):
    key_set(spine)
    if not set(STRUCTURE).issubset(spine):
        raise ValueError('missing Step00 structural columns')
    frame = spine[STRUCTURE].copy()
    for col in STRUCTURE:
        if col.endswith('_date'):
            from .dates import exact_day
            dates = frame[col].map(exact_day)
            if dates.isna().any():
                raise ValueError('invalid Step00 exact dates')
            frame[col] = dates.map(lambda x: x.date().isoformat())
        else:
            frame[col] = frame[col].astype('string')
    payload = frame.sort_values(KEYS).to_dict('records')
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate(features, spine):
    if list(features.columns) != KEYS + COLUMNS:
        raise ValueError('incompatible medication schema')
    if key_set(features) != key_set(spine):
        raise ValueError('medication episode keys differ from Step00')
    for col in KEYS + COLUMNS:
        if not isinstance(features[col].dtype, pd.StringDtype):
            raise ValueError('medication keys and names must use nullable string dtype')
        if col in KEYS:
            continue
        for cell in features[col].dropna():
            names = cell.split('; ')
            if not cell or cell.casefold() in {'none','null','nan','0'} or names != sorted(set(names)):
                raise ValueError('noncanonical medication name list')
    return features

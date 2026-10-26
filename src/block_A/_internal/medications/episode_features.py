"""Deterministic, nullable drug-name lists by documentary evidence."""
import re
import pandas as pd
from .contracts import KEYS, COLUMNS, validate


def review_name(canonical, raw, confidence):
    text = '' if pd.isna(canonical) else str(canonical).strip()
    trusted = confidence in {'high_exact', 'high', 'exact'}
    if not trusted or text.casefold() in {'', 'unknown', 'unmapped', 'indeterminate', 'nan', 'none'}:
        text = '' if pd.isna(raw) else str(raw)
    from .classification import _restricted_vocabulary_display
    text, redacted = _restricted_vocabulary_display(text)
    if redacted:
        return ''  # Raw evidence remains protected; never publish contact instructions.
    # Source drug-name fields only; remove control characters and list delimiters.
    return re.sub(r'\s+', ' ', re.sub(r'[\x00-\x1f\x7f;]', ' ', text)).strip().casefold()


def canonical_name(canonical, confidence):
    """Only a validated canonical ingredient may enter the Step10 name contract."""
    if confidence not in {'high_exact', 'high', 'exact'}:
        return ''
    text = '' if pd.isna(canonical) else str(canonical).strip()
    if text.casefold() in {'', 'unknown', 'unmapped', 'indeterminate', 'nan', 'none', 'null'}:
        return ''
    from .classification import _restricted_vocabulary_display
    text, redacted = _restricted_vocabulary_display(text)
    if redacted:
        return ''
    return re.sub(r'\\s+', ' ', re.sub(r'[\\x00-\\x1f\\x7f;]', ' ', text)).strip().casefold()


def build(spine, links):
    result = spine[KEYS].astype('string').copy()
    links = links.copy()
    if 'canonical_medication_name' not in links.columns:
        raise ValueError('canonical medication names required for Step10')
    for col in KEYS:
        links[col] = links[col].astype('string')
    masks = [links.source_kind.eq('CTDB')]
    order = links.evidence_level.eq('order_date_proximity')
    masks += [order & links.relation_to_episode.eq(k) for k in ('within_episode','before_episode','after_episode')]
    masks += [order, links.evidence_level.eq('consent_only_review'), pd.Series(True, index=links.index)]
    for col, mask in zip(COLUMNS, masks):
        subset = links.loc[mask & links.canonical_medication_name.fillna('').ne('')]
        grouped = subset.groupby(KEYS).canonical_medication_name.agg(lambda names: '; '.join(sorted(set(names))))
        result[col] = pd.Series([grouped.get(tuple(key), pd.NA) for key in result[KEYS].itertuples(index=False, name=None)], index=result.index, dtype='string')
    result = result.sort_values(KEYS).reset_index(drop=True)
    return validate(result, spine)

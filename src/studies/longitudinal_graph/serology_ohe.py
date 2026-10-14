"""Fixed-schema episode serology; explicit clinical approval, never inferred labels."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.studies.longitudinal_graph.validation import KEYS, validate_keys, align_source

LEVELS = ('positive', 'negative', 'indeterminate', 'missing')
SOURCES = {'ssa': 'lab__anti_ro_ssa__text', 'ssb': 'lab__anti_la_ssb__text', 'ana': 'lab__ana_status__text'}
COLUMNS = [f'sero_ohe__{marker}__{level}' for marker in SOURCES for level in LEVELS]

class SerologyBlocked(ValueError):
    def __init__(self, reason, audit=None):
        super().__init__(reason)
        self.report = {'status': 'blocked_needs_mapping_review', 'reason': reason,
                       'vocabulary': audit or [], 'outputs_written': False}

def normalize_format(value):
    return ' '.join(str(value).split()).casefold()

def vocabulary(master):
    """Only hashes and counts are exportable; arbitrary source comments stay private."""
    rows = []
    for marker, source in SOURCES.items():
        if source not in master:
            continue
        keys = master[source].map(lambda x: '<true-null>' if pd.isna(x) else normalize_format(x))
        for key, count in keys.value_counts(dropna=False).items():
            rows.append({'biomarker': marker, 'source_value_sha256': hashlib.sha256(key.encode()).hexdigest(),
                         'n_visits': int(count), 'true_null': key == '<true-null>'})
    return rows

def load_mapping(config, root):
    section = config['serology_ohe']
    path = Path(section['mapping_file'])
    if not path.is_absolute(): path = Path(root) / path
    if not path.exists(): raise SerologyBlocked('missing_mapping_file')
    mapping = yaml.safe_load(path.read_text())
    return mapping, path

def build_serology_ohe(master, config, registry, mapping):
    validate_keys(master, 'serology source')
    section = config['serology_ohe']; audit = vocabulary(master)
    if (section.get('drop_first') is not False or section.get('categories') != list(LEVELS)
            or section.get('sources') != SOURCES or not section.get('strict_unknown_values')):
        raise SerologyBlocked('invalid_fixed_schema', audit)
    if any(source not in master for source in SOURCES.values()): raise SerologyBlocked('missing_source_columns', audit)
    if not mapping.get('version') or not all(mapping.get(k) for k in ('approved_by', 'approved_on', 'rationale')):
        raise SerologyBlocked('clinical_mapping_not_approved', audit)
    if registry is None or not {'public_variable', 'temporal_scope'} <= set(registry):
        raise SerologyBlocked('missing_temporal_registry', audit)
    state = master[KEYS].reset_index(drop=True).copy(); manifest = []; qc = []; decisions = []
    for marker, source in SOURCES.items():
        entry = registry.loc[registry.public_variable.eq(source)]
        if len(entry) != 1 or entry.iloc[0].temporal_scope != 'episode_resolved':
            raise SerologyBlocked('source_not_episode_resolved', audit)
        approved = mapping.get('markers', {}).get(marker, {})
        clinical_map = {}
        for label, decision in approved.items():
            key = normalize_format(label)
            if (key in clinical_map or not isinstance(decision, dict) or decision.get('level') not in LEVELS
                    or not decision.get('rationale') or not decision.get('approved_by')):
                raise SerologyBlocked('invalid_mapping_decision', audit)
            clinical_map[key] = decision['level']
            decisions.append({'biomarker': marker, 'source': source, 'source_value_sha256': hashlib.sha256(key.encode()).hexdigest(),
                              'level': decision['level'], 'version': mapping['version'], 'approved_by': decision['approved_by'],
                              'rationale': decision['rationale'], 'approved_on': mapping['approved_on']})
        canonical = []
        for value in master[source]:
            if pd.isna(value): canonical.append('missing')
            else:
                key = normalize_format(value)
                if key not in clinical_map: raise SerologyBlocked('unrecognized_source_value', audit)
                canonical.append(clinical_map[key])
        values = pd.Series(canonical)
        temporal = 'temporal_provenance_needs_review'
        review = mapping.get('temporal_review', {})
        date_column = review.get('date_columns', {}).get(marker)
        if date_column:
            if date_column not in master: raise SerologyBlocked('missing_lab_date_column', audit)
            dates = pd.to_datetime(master[date_column], errors='coerce')
            anchor = pd.to_datetime(master.clinical_anchor_date, errors='coerce')
            observed = values.ne('missing').to_numpy()
            if (dates[observed].isna().any() or anchor[observed].isna().any()
                    or (dates[observed] > anchor[observed]).any()):
                raise SerologyBlocked('future_or_unverifiable_lab_measurement', audit)
            if all(review.get(k) for k in ('approved_by', 'approved_on', 'rationale')): temporal = 'verified_at_or_before_anchor'
        observed = values.ne('missing').to_numpy()
        for level in LEVELS:
            name = f'sero_ohe__{marker}__{level}'
            state[name] = values.eq(level).astype('int8')
            manifest.append({'feature': name, 'parent_biomarker': marker, 'level': level, 'source': source,
                             'temporal_scope': 'episode_resolved', 'temporal_scope_source': 'step10_registry',
                             'temporal_provenance': temporal, 'schema_version': section['schema_version'],
                             'mapping_version': mapping['version'], 'used_in_S3': False, 'used_in_S4': True,
                             'missingness_origin': 'true_null_or_explicit_approved_source_label'})
            qc.append({'biomarker': marker, 'level': level, 'n_visits': len(master), 'n_patients': master.patient_id.nunique(),
                       'n_source_nonnull': int(master[source].notna().sum()), 'n_clinically_observed': int(observed.sum()),
                       'n_patients_observed': master.loc[observed, 'patient_id'].nunique(),
                       'count': int(state[name].sum()), 'constant': state[name].nunique() < 2,
                       'one_hot_valid': True, 'unknown_count': 0, 'temporal_provenance': temporal})
    validate_ohe(state)
    return state, pd.DataFrame(manifest), pd.DataFrame(decisions), pd.DataFrame(qc)

def validate_ohe(state):
    validate_keys(state, 'serology OHE')
    if list(state.columns) != KEYS + COLUMNS: raise ValueError('Unexpected OHE schema/order')
    if not state[COLUMNS].isin([0, 1]).all().all(): raise ValueError('OHE must be finite binary')
    for marker in SOURCES:
        if not state[[f'sero_ohe__{marker}__{level}' for level in LEVELS]].sum(axis=1).eq(1).all():
            raise ValueError('Invalid one-hot block')

def join_ohe(state, ohe):
    validate_ohe(ohe)
    if len(state) != len(ohe): raise ValueError('OHE cohort differs')
    aligned = align_source(state[KEYS], ohe, COLUMNS)
    return state.merge(aligned, on=KEYS, how='left', validate='one_to_one', sort=False)

def s4_preprocessor(legacy):
    def preprocess(sampled, features):
        numeric = sampled[legacy].astype(float).replace([np.inf, -np.inf], np.nan)
        medians = numeric.median()
        if medians.isna().any(): raise ValueError('bootstrap_legacy_feature_all_missing')
        indicators = [f for f in features if f in COLUMNS]
        if not sampled[indicators].isin([0, 1]).all().all(): raise ValueError('invalid_bootstrap_OHE')
        return pd.concat([sampled[KEYS], numeric.fillna(medians), sampled[indicators]], axis=1), list(features)
    return preprocess

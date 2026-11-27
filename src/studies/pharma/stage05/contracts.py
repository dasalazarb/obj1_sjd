"""Canonical inputs, structural preflight and time-safe feature materialization."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import common
from config import ESSDAI_MODERATE_OR_HIGH_CUTOFF, ESSPRI_THRESHOLD
from src.studies._shared import (
    STUDY_CONTRACT_VERSION, resolve_concept, sha256_file,
    validate_integrated_dataset, validate_predictors, validate_transition_intervals,
)

STAGE = 'pharma/05_pharma_deck_evidence'
PREFIX = '05_pharma_'
POPS = ('Pop1', 'Pop2', 'Pop3')
BASELINE_POPS = (*POPS, 'Unclassifiable', 'No baseline')
ROUTES = tuple(product(POPS, repeat=2))
KEYS = ['patient_id', 'clinical_episode_id']
DAYS_PER_YEAR = 365.25
ORDER = ['patient_id', 'clinical_anchor_date', 'clinical_visit_number']
THRESHOLD = ESSDAI_MODERATE_OR_HIGH_CUTOFF
SOURCE_FILES = {
    'profile': ('00_profile_labs', '00_pharma_lab_profile.csv'),
    'profile_summary': ('00_profile_labs', '00_pharma_profile_summary.json'),
    'feature_availability': ('01_run_pharma_main', '01_pharma_feature_availability.csv'),
    'lab_availability': ('01_run_pharma_main', '01_pharma_lab_availability.csv'),
    'sensitivity': ('02_pharma_sensitivity', '02_pharma_sensitivity_summary.csv'),
    'models': ('03_pharma_multistate_biomarkers', '03_pharma_multistate_models.csv'),
    'support': ('03_pharma_multistate_biomarkers', '03_pharma_multistate_support.csv'),
    'candidates': ('03_pharma_multistate_biomarkers', '03_pharma_multistate_candidates.csv'),
    'model_summary': ('03_pharma_multistate_biomarkers', '03_pharma_multistate_summary.csv'),
    'history_feasibility': ('03b_pharma_history_feasibility', '03b_pharma_history_summary.csv'),
    'history_models': ('04_pharma_history_aware', '04_pharma_history_summary.csv'),
}
CSV_REQUIRED = {
    'profile': {'lab', 'result_source', 'value_column', 'unit_assessment_status'},
    'registry': {'public_variable', 'role'},
    'feature_availability': {'feature', 'resolved_column'},
    'lab_availability': {'lab', 'transition_pair', 'n_from', 'n_to', 'n_paired'},
    'sensitivity': {'exposure', 'exposure_family', 'model_status', 'interpretability_status', 'direction_consistent'},
    'models': {'biomarker', 'transition_id', 'model_version', 'biomarker_mean', 'biomarker_sd',
               'estimate_per_sd', 'age_estimate', 'model_status', 'interpretability_status',
               'n_intervals', 'n_patients', 'n_events'},
    'support': {'biomarker', 'transition_id', 'supported_for_model', 'n_intervals_at_risk',
                'n_patients_at_risk', 'n_events', 'n_patients_with_event'},
    'candidates': {'exposure', 'eligible_for_03'},
    'model_summary': {'biomarker'},
}
OPTIONAL = {'context', 'history_feasibility', 'history_models', 'baseline_reference', 'pro_models', 'specimens'}


@dataclass
class Inputs:
    master: pd.DataFrame
    intervals: pd.DataFrame
    tables: dict
    features: dict
    preflight: dict
    inventory: pd.DataFrame


def source_paths() -> dict[str, Path]:
    master = common.resolve_effective_longitudinal_input(common.EFFECTIVE_LONGITUDINAL_PARQUET)
    sidecar = common.step10_sidecar_base(master)
    pharma = common.STUDIES_TABLES_DIR / 'pharma'
    return {
        'master': master,
        'intervals': common.STUDIES_ANALYTIC_DIR / 'pharma' / '01_run_pharma_main' /
        '01_pharma_transition_intervals.parquet',
        'registry': sidecar.with_name('10_variable_registry.csv'),
        'context': sidecar.with_name('10_integrated_longitudinal_context.parquet'),
        'baseline_reference': common.INTEGRATED_BASELINE_PARQUET,
        'pro_models': common.BLOCKA_TABLES_DIR / '09_pros_longitudinal' /
        '09_pro_longitudinal_models_crude_adjusted.csv',
        'specimens': common.DATA_DIR / 'restricted' / 'pharma' / 'ctdb_specimen_inventory.parquet',
        **{key: pharma / folder / filename for key, (folder, filename) in SOURCE_FILES.items()},
    }


def load_settings(directory: Path) -> tuple[dict, dict]:
    settings = yaml.safe_load((directory / 'config_stage05.yaml').read_text())
    upstream = yaml.safe_load((directory / 'config.yaml').read_text())
    mapping = json.loads((directory / '05_pharma_mapping.json').read_text())
    if settings.get('contract_version') != upstream.get('contract_version') or (
        settings['contract_version'] != STUDY_CONTRACT_VERSION
    ):
        raise ValueError('Incompatible Pharma dataset contracts')
    settings.update(seed=upstream['random_seed'],
                    minimum_events=upstream['minimum_counts']['events_for_multivariable_model'],
                    sustained_visits=upstream['sustained_transition']['required_consecutive_destination_episodes'])
    horizons = settings['horizons_years']
    if (sorted(set(horizons)) != horizons or min(horizons) <= 0 or
            not 0 < settings['minimum_bootstrap_success'] <= 1 or
            min(settings['bootstrap_replicates'], settings['minimum_events'],
                settings['minimum_event_patients'], settings['minimum_correlation_patients']) < 1 or
            settings['sustained_visits'] < 2 or settings['minimum_trend_measurements'] < 3 or
            settings['minimum_classifiable_visits'] < 2):
        raise ValueError('Invalid fixed Stage05 settings')
    if not {'essdai_total', 'esspri_total'}.issubset(mapping):
        raise ValueError('Stage05 mapping requires ESSDAI and ESSPRI')
    for marker, item in mapping.items():
        if item.get('type') not in {'continuous', 'categorical', 'binary', 'ordinal'} or (
            item.get('temporal_source') not in {'episode', 'dated_episode_lab',
                                               'history_through_anchor', 'unavailable'}
        ):
            raise ValueError(f'Invalid typed mapping: {marker}')
        if item['temporal_source'] in {'episode', 'history_through_anchor'} and not (
            item.get('shared_concept') or item.get('columns')
        ):
            raise ValueError(f'Missing source mapping: {marker}')
        if item['temporal_source'] == 'dated_episode_lab' and not item.get('assay', '').startswith('lab__'):
            raise ValueError(f'Invalid laboratory assay mapping: {marker}')
        if item['type'] == 'binary' and (not item.get('positive') or not item.get('negative') or
                                        set(item['positive']) & set(item['negative'])):
            raise ValueError(f'Invalid binary encoding: {marker}')
        validate_predictors(item.get('columns', []))
    for marker in ('essdai_total', 'esspri_total'):
        item = mapping[marker]
        if (item.get('shared_concept') != marker or item['type'] != 'continuous' or
                item['temporal_source'] != 'episode' or not item.get('required')):
            raise ValueError(f'Invalid required mapping: {marker}')
    return settings, mapping


def numeric(values: pd.Series) -> pd.Series:
    result = pd.to_numeric(values, errors='coerce').astype(float)
    return result.where(np.isfinite(result))


def normalized(values: pd.Series) -> pd.Series:
    return values.astype('string').str.strip().str.lower().replace('', pd.NA)


def flag(values: pd.Series) -> pd.Series:
    return normalized(values).isin(['1', '1.0', 'true'])


def check_keys(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if not set(keys).issubset(frame) or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f'{label} requires unique, non-null keys {keys}')


def same_values(left: pd.Series, right: pd.Series) -> pd.Series:
    return ((left.isna() & right.isna()) | left.eq(right)).fillna(False)


def validate_inputs(master: pd.DataFrame, intervals: pd.DataFrame, settings: dict) -> tuple:
    """Respect Step 10 visit ordering, including same-date episodes."""
    check_keys(master, KEYS, 'Effective master')
    if master.empty:
        raise ValueError('Effective master contains no episodes')
    summary = validate_integrated_dataset(master)
    master = master.copy()
    master.clinical_anchor_date = pd.to_datetime(master.clinical_anchor_date)
    master.clinical_visit_number = numeric(master.clinical_visit_number)
    if master.clinical_visit_number.isna().any() or master.clinical_visit_number.mod(1).ne(0).any():
        raise ValueError('Clinical visit numbers must be finite integers')
    for column in ('clinical_visit', 'is_clinical_baseline'):
        allowed = normalized(master[column]).isin(['1', '1.0', 'true', '0', '0.0', 'false'])
        if not allowed.all() or (column == 'clinical_visit' and not flag(master[column]).all()):
            raise ValueError(f'Invalid canonical flag: {column}')
        master[column] = flag(master[column])
    master = master.sort_values(ORDER).reset_index(drop=True)
    if master.groupby('patient_id').clinical_visit_number.diff().dropna().le(0).any():
        raise ValueError('Clinical visit numbers must increase strictly within patient')
    scores = summary['resolved_columns']
    for key in ('essdai_total', 'esspri_total'):
        column = scores[key]
        values = numeric(master[column])
        if (master[column].notna() & values.isna()).any():
            raise ValueError(f'Non-finite observed score: {column}')
        master[column] = values
    dai, pri = master[scores['essdai_total']], master[scores['esspri_total']]
    master['pop'] = master[scores['pop']].astype('string')
    expected_pop = np.select(
        [dai.ge(THRESHOLD), dai.lt(THRESHOLD) & pri.ge(ESSPRI_THRESHOLD),
         dai.lt(THRESHOLD) & pri.lt(ESSPRI_THRESHOLD)], POPS, default='Unclassifiable')
    if not master['pop'].eq(expected_pop).fillna(False).all():
        raise ValueError('Effective master Pop disagrees with observed ESSDAI/ESSPRI thresholds')

    intervals = intervals.copy().reset_index(drop=True)
    interval_keys = ['patient_id', 'from_clinical_episode_id', 'to_clinical_episode_id']
    check_keys(intervals, interval_keys, 'Pharma 01 intervals')
    for column in ('interval_days', 'interval_years'):
        if column not in intervals:
            raise ValueError(f'Missing Pharma 01 interval field: {column}')
        intervals[column] = numeric(intervals[column])
    validate_transition_intervals(intervals, master)
    if not (intervals.from_pop.isin(POPS) & intervals.to_pop.isin(POPS)).all():
        raise ValueError('Pharma 01 contains an unclassifiable transition endpoint')

    indexed = master.set_index(KEYS)
    for side in ('from', 'to'):
        endpoint_keys = ['patient_id', f'{side}_clinical_episode_id']
        aligned = indexed.reindex(pd.MultiIndex.from_frame(intervals[endpoint_keys])).reset_index(drop=True)
        date_column = f'{side}_clinical_anchor_date'
        if date_column not in intervals:
            raise ValueError(f'Missing Pharma 01 anchor: {date_column}')
        intervals[date_column] = pd.to_datetime(intervals[date_column], errors='coerce')
        if not same_values(intervals[date_column], aligned.clinical_anchor_date).all():
            raise ValueError(f'Pharma 01 discordant anchor: {date_column}')
        for concept in ('essdai_total', 'esspri_total', 'age_at_visit'):
            try:
                source = resolve_concept(master, concept)
            except ValueError:
                continue
            for name in dict.fromkeys((f'{side}_{source}', f'{side}_{concept}')):
                if name in intervals and not np.allclose(
                    numeric(intervals[name]), numeric(aligned[source]), equal_nan=True, rtol=0, atol=1e-8
                ):
                    raise ValueError(f'Pharma 01 and Effective master disagree on {name}')
    elapsed = (intervals.to_clinical_anchor_date - intervals.from_clinical_anchor_date).dt.days
    if not elapsed.gt(0).all() or not intervals.interval_days.eq(elapsed).all() or not np.allclose(
        intervals.interval_years, elapsed / DAYS_PER_YEAR, rtol=0, atol=1e-9
    ):
        raise ValueError('Pharma 01 duration inconsistent with canonical dates')

    by_patient = master.groupby('patient_id', sort=False)
    next_id = by_patient.clinical_episode_id.shift(-1)
    next_date = by_patient.clinical_anchor_date.shift(-1)
    next_pop = by_patient['pop'].shift(-1)
    positive = next_date.gt(master.clinical_anchor_date)
    eligible = positive & master['pop'].isin(POPS) & next_pop.isin(POPS)
    expected = master.loc[eligible, KEYS].assign(next_id=next_id[eligible])
    if set(expected.itertuples(index=False, name=None)) != set(intervals[interval_keys].itertuples(index=False, name=None)):
        raise ValueError('Canonical Pharma 01 interval cohort disagrees with Effective master adjacency')
    classifiable = master.loc[master['pop'].isin(POPS)]
    next_class = classifiable.groupby('patient_id').clinical_visit_number.shift(-1)
    n_gaps = int(next_class.sub(classifiable.clinical_visit_number).gt(1).sum())
    summary.update(n_same_date_adjacencies=int(next_date.eq(master.clinical_anchor_date).sum()),
                   n_positive_adjacencies_touching_unclassifiable=int((positive & ~eligible).sum()),
                   n_classifiable_gaps=n_gaps, n_intervals=len(intervals),
                   n_interval_patients=intervals.patient_id.nunique(),
                   n_classifiable_patients=classifiable.patient_id.nunique())
    death_column = settings.get('death_date_column')
    if death_column:
        if death_column not in master:
            raise ValueError('Configured death date column is absent')
        death = pd.to_datetime(master[death_column], errors='coerce')
        if (master[death_column].notna() & death.isna()).any():
            raise ValueError('Invalid documented death date')
        if death.groupby(master.patient_id).nunique().gt(1).any():
            raise ValueError('Conflicting death dates within a patient')
        master['_death'] = death.groupby(master.patient_id).transform('min')
        if master.clinical_anchor_date.gt(master['_death']).any():
            raise ValueError('Clinical visit after documented death')

    return master, intervals, summary


def exact_columns(frame: pd.DataFrame, expected: list[str], label: str) -> str | None:
    found = [name for name in expected if name in frame]
    if len(found) > 1:
        for name in found[1:]:
            if not same_values(frame[found[0]], frame[name]).all():
                raise ValueError(f'Conflicting official aliases for {label}')
    return found[0] if found else None


def _episode_metadata(master: pd.DataFrame, context: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if set(columns).issubset(master):
        return master[columns]
    if set(columns).issubset(context):
        return master[KEYS].merge(context[KEYS + columns], on=KEYS, how='left',
                                  validate='one_to_one')[columns]
    return pd.DataFrame(index=master.index)


def _lab_values(master: pd.DataFrame, context: pd.DataFrame, profile: pd.DataFrame,
                registry: set[str], item: dict) -> tuple[pd.Series | None, str, str, str]:
    assay = item['assay']
    if assay not in profile.index:
        return None, '', '', 'assay_not_in_pharma00_profile'
    row = profile.loc[assay]
    column = row.get('value_column')
    if row.result_source != 'value' or column != f'{assay}__value' or column not in master or column not in registry:
        return None, '', '', 'verified_numeric_source_unavailable'
    if row.unit_assessment_status != 'available':
        return None, column, '', 'unit_harmonization_unverified'
    unit_column, date_column = row.get('unit_column'), row.get('measurement_date_column')
    if unit_column != f'{assay}__unit' or date_column != f'{assay}__measurement_date':
        return None, column, '', 'canonical_lab_metadata_mapping_unavailable'
    source = _episode_metadata(master, context, [unit_column, date_column])
    if source.empty:
        return None, column, '', 'dated_measurement_or_unit_metadata_unavailable'
    values = numeric(master[column])
    units = source[unit_column].astype('string').str.strip().replace('', pd.NA)
    unique_units = units[values.notna()].dropna().unique()
    if len(unique_units) != 1 or (values.notna() & units.isna()).any():
        return None, column, '', 'mixed_or_incomplete_units'
    dates = pd.to_datetime(source[date_column], errors='coerce')
    return values.where(dates.le(master.clinical_anchor_date)), column, str(unique_units[0]), ''


def materialize(master: pd.DataFrame, intervals: pd.DataFrame, tables: dict, mapping: dict) -> tuple:
    """Resolve verified columns and attach endpoint features by episode key."""
    features = {}
    registry = set(tables['registry'].public_variable.dropna())
    profile = tables['profile'].set_index('lab', verify_integrity=True)
    context = tables.get('context', pd.DataFrame())
    available_labs = set(tables['lab_availability'].lab.dropna())
    available_features = set(tables['feature_availability'].resolved_column.dropna())
    if 'context' in tables:
        check_keys(context, KEYS, 'Step 10 context')
        master_keys = set(master[KEYS].itertuples(index=False, name=None))
        if master_keys != set(context[KEYS].itertuples(index=False, name=None)):
            raise ValueError('Step 10 context keys disagree with Effective master')
        if 'clinical_anchor_date' in context:
            anchors = master[KEYS].merge(context[KEYS + ['clinical_anchor_date']], on=KEYS, validate='one_to_one')
            if not same_values(master.clinical_anchor_date, pd.to_datetime(anchors.clinical_anchor_date)).all():
                raise ValueError('Step 10 context anchors disagree with Effective master')
    mapping = mapping.copy()
    represented = {item.get('assay') for item in mapping.values()}
    candidates = tables['candidates']
    reviewed = candidates.loc[flag(candidates.eligible_for_03), 'exposure'].dropna().unique()
    for assay in sorted(reviewed):
        if assay in represented or not re.fullmatch(r'lab__[a-z0-9_]+', str(assay)):
            continue
        if assay in profile.index and profile.loc[assay].get('recommended_use') == 'numeric_longitudinal':
            mapping[assay] = dict(group='laboratory', type='continuous', temporal_source='dated_episode_lab',
                                  assay=assay, unit='', direction='not_applicable', reviewed_candidate=True)
    for marker, item in mapping.items():
        rule = item['temporal_source']
        column, unit, reason = '', item.get('unit', ''), item.get('reason', '')
        values = None
        if rule == 'dated_episode_lab':
            values, column, unit, reason = _lab_values(master, context, profile, registry, item)
        elif rule != 'unavailable':
            if item.get('shared_concept'):
                try:
                    column = resolve_concept(master, item['shared_concept'])
                except ValueError:
                    if item.get('required'):
                        raise
            else:
                column = exact_columns(master, item['columns'], marker)
            if column and (column in registry or item.get('required')):
                values = master[column]
            elif item.get('context_column') in context and item['context_column'] in registry:
                column = item['context_column']
                values = _episode_metadata(master, context, [column])[column]
            else:
                reason = 'column_absent_or_not_in_step10_registry'
        if item['type'] == 'binary':
            raw = values if values is not None else pd.Series(pd.NA, index=master.index)
            text = normalized(raw)
            encoded = text.map({**dict.fromkeys(item['positive'], 1.), **dict.fromkeys(item['negative'], 0.)})
            observed = raw.notna()
            if item.get('assay'):
                assay = item['assay']
                status_column, date_column = f'{assay}__reference_status', f'{assay}__measurement_date'
                metadata = _episode_metadata(master, context, [status_column, date_column])
                if not metadata.empty and {status_column, date_column}.issubset(registry):
                    dates = pd.to_datetime(metadata[date_column], errors='coerce')
                    current = normalized(metadata[status_column]).where(dates.le(master.clinical_anchor_date))
                    encoded = encoded.fillna(current.map({'positive': 1., 'negative': 0.}))
                    observed |= current.notna()
                    if values is None and observed.any():
                        column, values, reason = status_column, encoded, ''
            master[f'raw_status__{marker}'] = np.select(
                [encoded.notna(), observed], ['known', 'indeterminate'], default='not_measured')
            values = encoded if values is not None else None
        if values is not None and item['type'] in {'continuous', 'ordinal'}:
            values = numeric(values)
            if 'levels' in item:
                values = values.where(values.isin(item['levels']))
        if values is not None and item['type'] == 'categorical':
            values = values.astype('string').str.strip().replace('', pd.NA)
            values = values.mask(normalized(values).isin(['unknown', 'not reported', 'indeterminate']))
        master[f'feature__{marker}'] = values if values is not None else np.nan
        features[marker] = {**item, 'column': column or '', 'unit': unit,
                            'status': 'available' if values is not None else 'unavailable', 'reason': reason,
                            'in_pharma01_availability': column in available_features or item.get('assay') in available_labs,
                            'n_excluded_after_or_undated': (
                                int(numeric(master[column]).notna().sum() - values.notna().sum())
                                if rule == 'dated_episode_lab' and values is not None else 0)}
        if rule == 'dated_episode_lab' and column:
            for side in ('from', 'to'):
                enriched = f'{side}_lab__{item["assay"]}'
                if enriched in intervals:
                    endpoint = master.set_index(KEYS)[column].reindex(pd.MultiIndex.from_frame(
                        intervals[['patient_id', f'{side}_clinical_episode_id']]))
                    if not np.allclose(numeric(intervals[enriched]), numeric(endpoint), rtol=0, atol=1e-8, equal_nan=True):
                        raise ValueError(f'Pharma 01 laboratory values disagree with Effective master: {enriched}')

    columns = [f'feature__{marker}' for marker in features]
    for side in ('from', 'to'):
        right = master[KEYS + ['clinical_visit_number', *columns]].rename(
            columns={'clinical_episode_id': f'{side}_clinical_episode_id',
                     'clinical_visit_number': f'{side}_visit_number',
                     **{name: f'{side}_{name}' for name in columns}})
        intervals = intervals.merge(right, on=['patient_id', f'{side}_clinical_episode_id'],
                                    how='left', validate='many_to_one')
    inventory = tables['registry'].loc[tables['registry'].public_variable.notna()].copy()
    inventory = inventory.reindex(columns=['public_variable', 'role', 'source', 'clinical_domain',
                                           'classification_rule', 'destination']).drop_duplicates()
    inventory['present_in_effective'] = inventory.public_variable.isin(master.columns)
    inventory['selected_stage05'] = inventory.public_variable.isin(
        {spec['column'] for spec in features.values() if spec['status'] == 'available'})
    return master, intervals, features, inventory


def validate_availability(intervals: pd.DataFrame, availability: pd.DataFrame) -> None:
    check_keys(availability, ['lab', 'transition_pair'], 'Pharma 01 laboratory availability')
    routes = {f'{origin} -> {target}': part for origin, target, part in route_groups(intervals)}
    for row in availability.itertuples(index=False):
        origin, destination = f'from_lab__{row.lab}', f'to_lab__{row.lab}'
        if row.transition_pair not in routes or not {origin, destination}.issubset(intervals):
            raise ValueError('Pharma 01 availability names disagree with its canonical interval schema')
        part = routes[row.transition_pair]
        start, end = part[origin].notna(), part[destination].notna()
        if (int(start.sum()), int(end.sum()), int((start & end).sum())) != (row.n_from, row.n_to, row.n_paired):
            raise ValueError(f'Pharma 01 laboratory availability counts disagree: {row.lab}, {row.transition_pair}')


def load_inputs(settings: dict, mapping: dict) -> Inputs:
    paths = source_paths()
    missing = [str(path) for role, path in paths.items() if role not in OPTIONAL and not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing canonical Stage05 inputs. Publish upstream at fixed paths before running:\n'
                                + '\n'.join(missing))
    hashes = {role: sha256_file(path) for role, path in paths.items() if path.is_file()}
    tables, optional_errors = {}, {}
    for role in hashes:
        path = paths[role]
        if role in {'master', 'intervals'}:
            continue
        try:
            if path.suffix == '.parquet':
                tables[role] = pd.read_parquet(path)
            elif path.suffix == '.csv':
                tables[role] = pd.read_csv(path)
            else:
                tables[role] = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            if role not in OPTIONAL or role == 'context':
                raise
            optional_errors[role] = f'incompatible_optional_source:{type(exc).__name__}'
            continue
        if role in CSV_REQUIRED and not CSV_REQUIRED[role].issubset(tables[role]):
            raise ValueError(f'Incompatible {role} schema: {sorted(CSV_REQUIRED[role] - set(tables[role]))}')
    profile_sources = tables['profile_summary'].get('input_hashes', [])
    profile_source = [row.get('sha256') for row in profile_sources if row.get('role') == 'integrated']
    if profile_source != [hashes['master']]:
        raise ValueError('Canonical Pharma 00 profile does not bind to the resolved Effective master hash')
    profile_context = [row.get('sha256') for row in profile_sources if row.get('role') == 'context']
    if profile_context and profile_context != [hashes.get('context')]:
        raise ValueError('Canonical Pharma 00 context hash differs from the companion for Effective master')
    master, intervals, summary = validate_inputs(pd.read_parquet(paths['master']),
                                                 pd.read_parquet(paths['intervals']), settings)
    # Some canonical legacy writers have no provenance sidecar. Check those
    # against actual interval keys, endpoint values and the Step03 refit.
    provenance = {}
    for role in ('intervals', 'sensitivity', 'models', 'support', 'candidates', 'model_summary'):
        path = paths[role]
        sidecar = path.parent / 'study_input_provenance.json'
        if sidecar.is_file():
            item = json.loads(sidecar.read_text())
            semantic = item.get('semantic_source', {})
            source = semantic.get('integrated_sha256') or semantic.get('effective_integrated_sha256')
            if item.get('sha256', {}).get(path.name) != hashes[role] or source != hashes['master']:
                raise ValueError(f'Canonical input provenance mismatch: {role}')
            provenance[role] = 'hash_verified'
        else:
            provenance[role] = 'canonical_content_checks; upstream sidecar absent'
    validate_availability(intervals, tables['lab_availability'])
    master, intervals, features, inventory = materialize(master, intervals, tables, mapping)
    if any(sha256_file(paths[role]) != digest for role, digest in hashes.items()):
        raise ValueError('A canonical input changed during preflight')
    summary.update(sources={role: {'path': str(path), 'sha256': hashes.get(role),
                                   'status': 'found' if role in hashes and role not in optional_errors else 'unavailable',
                                   'reason': optional_errors.get(role, '')}
                            for role, path in paths.items()},
                   provenance=provenance, mappings=features, minimum_events=settings['minimum_events'])
    return Inputs(master, intervals, tables, features, summary, inventory)


def route_groups(intervals: pd.DataFrame):
    for origin, destination in ROUTES:
        yield origin, destination, intervals.loc[
            intervals.from_pop.eq(origin) & intervals.to_pop.eq(destination)]


def support_status(n: int, available: bool = True, reason: str = '') -> dict:
    if not available:
        return {'status': 'unavailable', 'reason': reason}
    if not n:
        return {'status': 'not_estimable', 'reason': reason or 'no_evaluable_observations'}
    return {'status': 'estimated', 'reason': ''}

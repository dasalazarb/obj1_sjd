#!/usr/bin/env python3
"""Non-destructive medication availability audit; no confirmed exposure outputs.

All codebook fields are literal constants. Only explicit source metadata is used;
missing documentation never establishes absence of treatment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402
import config  # noqa: E402

RULES_VERSION = "08-treatment-audit-1.0.0"
DEFAULT_OUTPUT = common.OUTPUTS_DIR / "qc" / "blockA" / "08_treatment_audit"
PROTOCOLS = ("11-D-0172", "15-D-0051")

MEDICATION_NAME_COLUMNS = [
    'medications__rx_1_name', 'medications__rx_2_name',
    'medications__rx_3_name', 'medications__rx_4_name',
    'medications__rx_5_name', 'medications__rx_6_name',
    'medications__rx_7_name', 'medications__rx_8_name',
    'medications__rx_9_name', 'medications__rx_10_name',
]
MEDICATION_ROUTE_COLUMNS = [
    'medications__rx_1_route', 'medications__rx_2_route',
    'medications__rx_3_route', 'medications__rx_4_route',
    'medications__rx_5_route', 'medications__rx_6_route',
    'medications__rx_7_route', 'medications__rx_8_route',
    'medications__rx_9_route', 'medications__rx_10_route',
]
# IMPORTANTE: tres nombres llevan underscore final y no siguen un patrón uniforme.
MEDICATION_FREQUENCY_BY_SLOT = {
    1: 'medications__rx_1_frequency',
    2: 'medications__rx_2_frequency',
    3: 'medications__rx_3_frequency_',
    4: 'medications__rx_4_frequency',
    5: 'medications__rx_5_frequency_',
    6: 'medications__rx_6_frequency_',
    7: 'medications__rx_7_frequency',
    8: 'medications__rx_8_frequency',
    9: 'medications__rx_9_frequency',
    10: 'medications__rx_10_frequency',
}
MEDICATION_START_BY_SLOT = {
    1:'medications__med1_strt_date', 2:'medications__med2_strt_date',
    3:'medications__med3_strt_date', 4:'medications__med4_strt_date',
    5:'medications__med5_strt_date', 6:'medications__med6_strt_date',
    7:'medications__med7_strt_date', 8:'medications__med8_strt_date',
    9:'medications__med9_strt_date', 10:'medications__med10_strt_date',
}
MEDICATION_STOP_BY_SLOT = {
    1:'medications__med1_stop_date', 2:'medications__med2_stop_date',
    3:'medications__med3_stop_date', 4:'medications__med4_stop_date',
    5:'medications__med5_stop_date', 6:'medications__med6_stop_date',
    7:'medications__med7_stop_date', 8:'medications__med8_stop_date',
    9:'medications__med9_stop_date', 10:'medications__med10_stop_date',
}
# No se deben asignar automáticamente a slots RX sin evidencia:
MEDICATION_DOSE_COLUMNS = [
    'medications__antiplatelet_1_dose',
    'medications__anticogaulant_1_dose',  # typo existente: conservar
    'medications__cholesterol_med_dose',
    'medications__fibrates_dose',
    'medications__diabetes_dose',
    'medications__dose_6',
    'medications__dose_7',
    'medications__dose_8',
    'medications__dose_9',
    'medications__dose_10',
]
HISTORICAL_RX_NAME_COLUMNS = [
    'past_medical_history__rx_1_name',
    'past_medical_history__rx_2_name',
    'past_medical_history__rx_3_name',
    'past_medical_history__rx_4_name',
    'past_medical_history__rx_5_name',
]
RX_ALLERGY_FIELD = 'past_medical_history__rx_allergies' # NO es medicamento activo

ROUTE_CODES = {
    1: 'PO', 2: 'IM', 3: 'IV', 4: 'SC', 5: 'Topical',
    6: 'Transdermal', 7: 'Per rectum', 8: 'Intranasal', 9: 'Other',
}
FREQUENCY_CODES = {
    1: 'Daily in AM', 2: 'Daily in PM', 3: 'At bedtime', 4: 'Every 4 hours',
    5: 'Every 6 hours', 6: 'Every 8 hours', 7: 'Every 12 hours',
    8: '2 times a day', 9: '3 times a day', 10: '4 times a day',
    11: 'Every other day', 12: 'Weekly', 13: 'Every 2 weeks',
    14: 'Monthly', 15: 'PRN', 16: 'Other',
}
# Exact aliases; only a documented strength/form suffix may follow an alias.
DRUG_RULES = (
    ('hydroxychloroquine', 'non_biologic_systemic', ('hydroxychloroquine', 'plaquenil', 'hcq')),
    ('methotrexate', 'non_biologic_systemic', ('methotrexate', 'trexall', 'mtx')),
    ('mycophenolate mofetil', 'non_biologic_systemic', ('mycophenolate mofetil', 'cellcept', 'mmf')),
    ('leflunomide', 'non_biologic_systemic', ('leflunomide', 'arava')),
    ('rituximab', 'biologic', ('rituximab', 'rituxan', 'mabthera')),
    ('belimumab', 'biologic', ('belimumab', 'benlysta')),
    ('prednisone', 'glucocorticoid', ('prednisone',)),
    ('prednisolone', 'glucocorticoid', ('prednisolone',)),
    ('methylprednisolone', 'glucocorticoid', ('methylprednisolone', 'medrol', 'solu-medrol')),
    ('dexamethasone', 'glucocorticoid', ('dexamethasone', 'decadron')),
    ('pilocarpine', 'symptomatic', ('pilocarpine', 'salagen')),
    ('cevimeline', 'symptomatic', ('cevimeline', 'evoxac')),
    ('artificial tears', 'symptomatic', ('artificial tears',)),
    ('aspirin', 'other', ('aspirin',)),
    ('atorvastatin', 'other', ('atorvastatin', 'lipitor')),
)
NEGATIVE_TOKENS = {'none', 'no', 'no medications', 'none reported', 'no medication', 'nil'}
SUFFIX = re.compile(
    r'(?:(?:\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|%)?)|'
    r'tablets?|tabs?|capsules?|caps?|oral|solution|injection|cream|drops|po|iv|sc|im|daily|bid|tid|weekly|prn|\s)+$',
    re.I,
)
FIELD_GROUPS = {
    'current_name': MEDICATION_NAME_COLUMNS,
    'route': MEDICATION_ROUTE_COLUMNS,
    'frequency': list(MEDICATION_FREQUENCY_BY_SLOT.values()),
    'start': list(MEDICATION_START_BY_SLOT.values()),
    'stop': list(MEDICATION_STOP_BY_SLOT.values()),
    'unassigned_dose': MEDICATION_DOSE_COLUMNS,
    'historical_name': HISTORICAL_RX_NAME_COLUMNS,
    'allergy_not_exposure': [RX_ALLERGY_FIELD],
}
LONG_COLUMNS = [
    'record_id', 'source', 'source_path', 'source_sha256', 'source_row',
    'source_index_raw', 'patient_id', 'visit_raw', 'visit_column', 'protocol',
    'slot', 'name_column', 'name_raw', 'name_state', 'normalized_name',
    'drug_class', 'normalization_rule', 'confidence', 'form_raw',
    'start_column', 'start_raw', 'start_month', 'start_precision', 'start_status',
    'stop_column', 'stop_raw', 'stop_month', 'stop_precision', 'stop_status',
    'route_raw', 'route_label', 'route_representation', 'route_column',
    'frequency_raw', 'frequency_label', 'frequency_representation', 'frequency_column',
    'index_raw', 'index_month', 'index_precision', 'indication', 'indication_source',
    'qc_flags', 'audit_only',
]
QC_COLUMNS = ['record_id', 'source', 'source_row', 'patient_id', 'protocol', 'slot', 'flag']
COVERAGE_COLUMNS = [
    'source', 'protocol', 'level', 'treatment', 'n_eligible_patients',
    'n_patients_with_names', 'n_patients_mapped', 'n_patients_with_start_month',
    'n_patients_with_stop_month', 'n_patients_duration_potential', 'n_records',
    'clinical_feasibility',
]


def raw(value) -> str:
    return '' if value is None or pd.isna(value) else str(value)


def state(value, *, name=False) -> str:
    token = raw(value).strip().casefold()
    if name and token in NEGATIVE_TOKENS:
        return 'explicit_negative'
    if token in config.MISSING_STRINGS or token == '(missing)':
        return 'missing_cell'
    return 'observed'


def parse_month(value) -> dict:
    """Keep observed precision; never turn a month into an observed day."""
    text = raw(value).strip()
    if state(value) == 'missing_cell':
        return {'month': '', 'precision': '', 'status': 'missing'}
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return {'month': value.strftime('%Y-%m'), 'precision': 'day', 'status': 'valid'}
    match = re.fullmatch(r'(\d{1,2})/(\d{4})', text)
    if match:
        month, year = map(int, match.groups())
        precision = 'month'
    else:
        match = re.fullmatch(r'(\d{4})-(\d{2})(?:-(\d{2}))?', text)
        if not match:
            if re.fullmatch(r'\d{4}', text) and 1 <= int(text) <= 9999:
                return {'month': '', 'precision': 'year', 'status': 'partial_year'}
            return {'month': '', 'precision': '', 'status': 'invalid_or_ambiguous'}
        year, month = int(match[1]), int(match[2])
        precision = 'day' if match[3] else 'month'
        if match[3]:
            try:
                date(year, month, int(match[3]))
            except ValueError:
                return {'month': '', 'precision': '', 'status': 'invalid_or_ambiguous'}
    if not 1 <= year <= 9999 or not 1 <= month <= 12:
        return {'month': '', 'precision': '', 'status': 'invalid_or_ambiguous'}
    return {'month': f'{year:04d}-{month:02d}', 'precision': precision, 'status': 'valid'}


def decode_category(value, codes) -> tuple[str, str]:
    text = raw(value).strip()
    if state(value) == 'missing_cell':
        return '', 'missing'
    if re.fullmatch(r'\d+(?:\.0)?', text):
        return (codes[int(float(text))], 'code') if int(float(text)) in codes else ('', 'unmapped')
    for code, label in codes.items():
        if text.casefold() == label.casefold():
            return label, 'label'
        if re.fullmatch(rf'{code}\s+{re.escape(label)}', text, re.I):
            return label, 'code_and_label'
    return '', 'unmapped'


def normalize_name(value) -> dict:
    text = raw(value).strip().casefold()
    if state(value, name=True) != 'observed':
        return dict(normalized_name='', drug_class='', normalization_rule='no_name',
                    confidence='not_applicable', form_raw='')
    for name, drug_class, aliases in DRUG_RULES:
        for alias in aliases:
            suffix = text[len(alias):].strip() if text.startswith(alias) else None
            if suffix is not None and (not suffix or SUFFIX.fullmatch(suffix)):
                form = re.search(r'\b(tablets?|tabs?|capsules?|caps?|solution|injection|cream|drops)\b', suffix)
                return dict(normalized_name=name, drug_class=drug_class,
                            normalization_rule=f'exact_alias:{alias}' + (':strength_form_suffix' if suffix else ''),
                            confidence='dictionary_match_review_required',
                            form_raw=form[0] if form else '')
    return dict(normalized_name='unknown/unmapped', drug_class='unmapped',
                normalization_rule='no_unambiguous_rule', confidence='unmapped', form_raw='')


def field_inventory(frame, source):
    frame = frame.reset_index(drop=True)
    rows = []
    protocols = sorted(set(frame['_audit_protocol'])) or ['unknown']
    for protocol in sorted(set(protocols) | set(PROTOCOLS)):
        part = frame[frame['_audit_protocol'].eq(protocol)]
        for group, columns in FIELD_GROUPS.items():
            for column in columns:
                present = column in frame
                values = part[column] if present else pd.Series(dtype=object)
                states = values.map(lambda x: state(x, name=group.endswith('name')))
                observed = values[states.eq('observed')]
                parse_errors = None
                if group in {'start', 'stop'}:
                    parse_errors = sum(parse_month(x)['status'] != 'valid' for x in observed)
                elif group in {'route', 'frequency'}:
                    codes = ROUTE_CODES if group == 'route' else FREQUENCY_CODES
                    parse_errors = sum(decode_category(x, codes)[1] == 'unmapped' for x in observed)
                rows.append(dict(
                    source=source['name'], protocol=protocol, field=column, field_group=group,
                    column_present=present, source_protocol_rows=len(part),
                    availability=('no_rows_for_protocol' if part.empty else 'column_present' if present else 'column_unavailable'),
                    dtype=str(frame[column].dtype) if present else '',
                    n_missing_cells=int(states.eq('missing_cell').sum()) if present else None,
                    pct_missing_cells=100 * states.eq('missing_cell').mean() if present and len(part) else None,
                    n_explicit_negative=int(states.eq('explicit_negative').sum()) if present else None,
                    n_unique_observed=int(observed.map(raw).nunique()) if present else None,
                    n_patients_observed=int(part.loc[observed.index, '_audit_patient'].nunique()) if present else None,
                    n_visits_observed=int(part.loc[observed.index, source['visit_column']].map(raw).nunique()) if present else None,
                    n_patients_in_source_protocol=int(part['_audit_patient'].nunique()),
                    n_parse_errors=parse_errors,
                    dose_slot_mapping='unresolved' if group == 'unassigned_dose' else '',
                ))
    return pd.DataFrame(rows)


def extract_slots(frame, source):
    current, historical, doses = [], [], []
    for position, (index, row) in enumerate(frame.iterrows()):
        base = dict(source=source['name'], source_path=source.get('path', ''),
                    source_sha256=source.get('sha256', ''), source_row=position,
                    source_index_raw=str(index), patient_id=row['_audit_patient'],
                    visit_raw=raw(row[source['visit_column']]), visit_column=source['visit_column'],
                    protocol=row['_audit_protocol'], audit_only=True)
        for slot, name_column in enumerate(MEDICATION_NAME_COLUMNS, 1):
            fields = dict(name=name_column, route=MEDICATION_ROUTE_COLUMNS[slot-1],
                          frequency=MEDICATION_FREQUENCY_BY_SLOT[slot],
                          start=MEDICATION_START_BY_SLOT[slot], stop=MEDICATION_STOP_BY_SLOT[slot])
            if not any(state(row.get(c), name=k == 'name') != 'missing_cell' for k, c in fields.items()):
                continue
            item = dict(base, record_id=f"{source['name']}:{position}:rx:{slot}", slot=slot,
                        name_column=name_column, name_raw=raw(row.get(name_column)),
                        name_state=state(row.get(name_column), name=True) if name_column in row else 'column_unavailable',
                        indication='indeterminate', indication_source='')
            item.update(normalize_name(row.get(name_column)))
            for kind in ('start', 'stop'):
                column = fields[kind]
                parsed = parse_month(row.get(column))
                if column not in row:
                    parsed['status'] = 'column_unavailable'
                item.update({f'{kind}_column': column, f'{kind}_raw': raw(row.get(column)),
                             **{f'{kind}_{key}': value for key, value in parsed.items()}})
            for kind, codes in [('route', ROUTE_CODES), ('frequency', FREQUENCY_CODES)]:
                value = row.get(fields[kind])
                label, representation = decode_category(value, codes)
                item.update({f'{kind}_raw': raw(value), f'{kind}_label': label,
                             f'{kind}_representation': representation, f'{kind}_column': fields[kind]})
            index_value = row.get(source.get('index_column', ''))
            parsed_index = parse_month(index_value)
            item.update(index_raw=raw(index_value), index_month=parsed_index['month'],
                        index_precision=parsed_index['precision'])
            item['qc_flags'] = ''
            current.append(item)
        for slot, column in enumerate(HISTORICAL_RX_NAME_COLUMNS, 1):
            if state(row.get(column), name=True) == 'missing_cell':
                continue
            historical.append(dict(base, record_id=f"{source['name']}:{position}:history:{slot}",
                                   slot=slot, name_column=column, name_raw=raw(row[column]),
                                   name_state=state(row[column], name=True),
                                   **normalize_name(row[column]), exposure_status='historical_only'))
        for column in MEDICATION_DOSE_COLUMNS:
            if state(row.get(column)) != 'missing_cell':
                value = raw(row[column])
                doses.append(dict(base, dose_column=column, dose_raw=value,
                                  contains_number=bool(re.search(r'\d', value)),
                                  has_explicit_unit=bool(re.search(r'\b(?:mg|mcg|g|ml|units?)\b', value, re.I)),
                                  slot_mapping='unresolved_not_assigned'))
    base_columns = ['source', 'source_path', 'source_sha256', 'source_row', 'source_index_raw',
                    'patient_id', 'visit_raw', 'visit_column', 'protocol', 'audit_only']
    history_columns = base_columns + [
        'record_id', 'slot', 'name_column', 'name_raw', 'name_state', 'normalized_name',
        'drug_class', 'normalization_rule', 'confidence', 'form_raw', 'exposure_status',
    ]
    dose_columns = base_columns + [
        'dose_column', 'dose_raw', 'contains_number', 'has_explicit_unit', 'slot_mapping',
    ]
    return (pd.DataFrame(current, columns=LONG_COLUMNS), pd.DataFrame(historical, columns=history_columns),
            pd.DataFrame(doses, columns=dose_columns))


def temporal_qc(slots):
    slots = slots.copy()
    flags = {i: set() for i in slots.index}
    for i, row in slots.iterrows():
        found = flags[i]
        if row.name_state != 'observed':
            found.add('no_observed_medication_name')
        if row.drug_class == 'unmapped':
            found.add('unmapped_medication_name')
        for kind in ('start', 'stop'):
            if row[f'{kind}_status'] == 'invalid_or_ambiguous':
                found.add(f'{kind}_parse_error')
            if row[f'{kind}_status'] == 'partial_year':
                found.add(f'{kind}_year_only')
            if row[f'{kind}_precision'] == 'month':
                found.add(f'{kind}_monthly_precision')
            if row[f'{kind}_precision'] == 'day':
                found.add(f'{kind}_daily_precision_requires_source_review')
        if not row.start_month:
            found.add('start_unavailable')
        if not row.stop_month:
            found.add('stop_unavailable_open_or_undocumented')
        if row.start_month and row.stop_month:
            if row.start_month > row.stop_month:
                found.add('start_after_stop')
            elif row.start_month == row.stop_month:
                found.add('same_month_order_unresolved')
        if row.start_month and row.index_month:
            if row.start_month < row.index_month:
                found.add('pre_index_prevalent_candidate_valid')
            elif row.start_month == row.index_month:
                found.add('index_same_month_order_unresolved')
        for kind in ('route', 'frequency'):
            if row[f'{kind}_representation'] == 'unmapped':
                found.add(f'{kind}_unmapped')
        if not row.visit_raw:
            found.add('visit_unavailable')
    named = slots[slots.name_state.eq('observed')].copy()
    named['_drug_key'] = named.apply(
        lambda r: r.name_raw.casefold() if r.drug_class == 'unmapped' else r.normalized_name, axis=1)
    for _, group in named.groupby(['source', 'source_row'], dropna=False):
        for _, duplicate in group.groupby('_drug_key'):
            if len(duplicate) > 1:
                for i in duplicate.index:
                    flags[i].add('duplicate_drug_same_source_row')
        if group['_drug_key'].nunique() > 1:
            for i in group.index:
                flags[i].add('multiple_drugs_same_visit_concomitance_unconfirmed')
    for _, group in named[named.visit_raw.str.strip().ne('')].groupby(
            ['source', 'patient_id', 'visit_raw', '_drug_key'], dropna=False):
        if len(group) > 1:
            for i in group.index:
                flags[i].add('duplicate_drug_same_visit_requires_review')
    for _, group in named.groupby(['patient_id', '_drug_key'], dropna=False):
        if group[['source', 'source_row']].drop_duplicates().shape[0] > 1:
            for i in group.index:
                flags[i].add('repeated_drug_observation_not_new_start')
    # Strict overlap of bounded month intervals; open/unknown courses are not inferred.
    bounded = named[named.start_month.ne('') & named.stop_month.ne('') &
                    named.start_month.lt(named.stop_month)]
    for _, group in bounded.groupby('patient_id'):
        records = list(group.iterrows())
        for offset, (i, left) in enumerate(records):
            for j, right in records[offset+1:]:
                if left._drug_key == right._drug_key:
                    continue
                if max(left.start_month, right.start_month) < min(left.stop_month, right.stop_month):
                    flags[i].add('bounded_courses_overlap_candidate')
                    flags[j].add('bounded_courses_overlap_candidate')
    rows = []
    for i, found in flags.items():
        slots.at[i, 'qc_flags'] = '|'.join(sorted(found))
        for flag in sorted(found):
            rows.append({**slots.loc[i, ['record_id', 'source', 'source_row', 'patient_id', 'protocol', 'slot']].to_dict(),
                         'flag': flag})
    return slots, pd.DataFrame(rows, columns=QC_COLUMNS)


def summarize_coverage(slots, cohort):
    rows = []
    for (source, protocol), patients in cohort.groupby(['source', 'protocol']):
        subset = slots[slots.source.eq(source) & slots.protocol.eq(protocol) & slots.name_state.eq('observed')]
        for level, column in [('drug', 'normalized_name'), ('group', 'drug_class')]:
            for treatment, records in subset.groupby(column):
                valid = records.start_month.ne('') & records.stop_month.ne('') & records.start_month.lt(records.stop_month)
                rows.append(dict(
                    source=source, protocol=protocol, level=level, treatment=treatment,
                    n_eligible_patients=patients.patient_id.nunique(),
                    n_patients_with_names=records.patient_id.nunique(),
                    n_patients_mapped=records.loc[records.drug_class.ne('unmapped'), 'patient_id'].nunique(),
                    n_patients_with_start_month=records.loc[records.start_month.ne(''), 'patient_id'].nunique(),
                    n_patients_with_stop_month=records.loc[records.stop_month.ne(''), 'patient_id'].nunique(),
                    n_patients_duration_potential=records.loc[valid, 'patient_id'].nunique(),
                    n_records=len(records), clinical_feasibility='pending_integration',
                ))
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def audit_sources(sources, eligible_ids, *, inventory_callback=None):
    """Pure audit over supplied frames; caller handles paths, output and hashing."""
    eligible_ids = {raw(x).strip() for x in eligible_ids if state(x) != 'missing_cell'}
    if not eligible_ids:
        raise ValueError('Eligibility list is empty; no audit denominator can be established')
    inventories, currents, histories, doses, cohorts, diagnostics = [], [], [], [], [], []
    prepared = []
    seen_sources = set()
    for source in sources:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', source['name']) or source['name'] in seen_sources:
            raise ValueError('Source names must be unique simple identifiers')
        seen_sources.add(source['name'])
        frame = source['frame'].copy()
        required = [source['patient_column'], source['visit_column']]
        required += [source[k] for k in ('protocol_column', 'index_column') if source.get(k)]
        if source.get('protocol_column') and source.get('protocol'):
            raise ValueError('Specify protocol_column or protocol, not both')
        missing = [c for c in required if c not in frame]
        if missing:
            raise ValueError(f"{source['name']}: missing configured source keys: {missing}")
        frame['_audit_patient'] = frame[source['patient_column']].map(lambda x: raw(x).strip())
        if source.get('protocol_column'):
            frame['_audit_protocol'] = frame[source['protocol_column']].map(
                lambda x: raw(x).strip() if state(x) != 'missing_cell' else 'unknown')
        else:
            frame['_audit_protocol'] = source.get('protocol') or 'unknown'
        excluded = ~frame['_audit_patient'].isin(eligible_ids)
        diagnostics.append(dict(source=source['name'], n_rows_input=len(frame),
                                n_rows_outside_eligibility=int(excluded.sum()),
                                n_rows_missing_patient_id=int(frame['_audit_patient'].eq('').sum())))
        # Keep original positional row numbers even after eligibility selection.
        inventory = field_inventory(frame[~excluded], source)
        inventories.append(inventory)
        prepared.append((source, frame))
        cohorts.append(pd.DataFrame(dict(source=source['name'], protocol=frame.loc[~excluded, '_audit_protocol'],
                                         patient_id=frame.loc[~excluded, '_audit_patient'])).drop_duplicates())
    if not sources:
        raise ValueError('At least one source is required')
    all_inventory = pd.concat(inventories, ignore_index=True)
    if inventory_callback:
        inventory_callback(all_inventory)
    for source, frame in prepared:
        current, historical, dose = extract_slots(frame, source)
        for table in (current, historical, dose):
            table.drop(table.index[~table.patient_id.isin(eligible_ids)], inplace=True)
        currents.append(current)
        histories.append(historical)
        doses.append(dose)
    current, qc = temporal_qc(pd.concat(currents, ignore_index=True))
    cohort = pd.concat(cohorts, ignore_index=True).drop_duplicates()
    observed_ids = set(cohort.patient_id)
    coverage = summarize_coverage(current, cohort)
    candidates = current[current.name_state.eq('observed')].copy()
    candidates['candidate_status'] = 'audit_only_unconfirmed_observation'
    historical = pd.concat(histories, ignore_index=True)
    mapping_columns = ['source', 'protocol', 'name_raw', 'name_state', 'normalized_name',
                       'drug_class', 'normalization_rule', 'confidence']
    mapping = pd.concat([
        current.reindex(columns=mapping_columns).assign(inventory_kind='current'),
        historical.reindex(columns=mapping_columns).assign(inventory_kind='historical'),
    ], ignore_index=True).drop_duplicates()
    protocol_status = {}
    for protocol in sorted(set(PROTOCOLS) | {'unknown'} | set(cohort.protocol)):
        selected = cohort[cohort.protocol.eq(protocol)]
        inv = all_inventory[(all_inventory.protocol == protocol) & (all_inventory.field_group == 'current_name')]
        meds = current[current.protocol.eq(protocol) & current.name_state.eq('observed')]
        protocol_status[protocol] = dict(
            n_patients_in_sources=int(selected.patient_id.nunique()),
            n_patients_with_names=int(meds.patient_id.nunique()),
            n_present_name_columns=int(inv.loc[inv.source_protocol_rows.gt(0) & inv.column_present, 'field'].nunique()),
            source_gap='additional_EHR_BTRIS_or_notes_required' if protocol == '15-D-0051' else 'assess_observed_sources',
        )
    summary = dict(
        rules_version=RULES_VERSION, audit_only=True,
        n_eligible_patients=len(eligible_ids), n_eligible_patients_in_sources=len(observed_ids),
        n_eligible_patients_absent_from_sources=len(eligible_ids - observed_ids),
        n_current_slots=len(current), n_candidate_observations=len(candidates),
        n_historical_slots=sum(len(x) for x in histories), n_qc_flags=len(qc),
        n_distinct_patients_with_names=int(candidates.patient_id.nunique()),
        protocol_status=protocol_status, source_diagnostics=diagnostics,
        gates=dict(record_description='available' if len(candidates) else 'blocked_no_names',
                   episode_reconstruction='pending_manual_validation',
                   temporal_analysis='not_approved_requires_date_and_episode_validation',
                   clinical_feasibility='pending_integration'),
        blockers=['Indication is not documented by configured medication fields.',
                  'Dose-to-slot mapping is unresolved; doses are never assigned to RX slots.',
                  'Repeated observations are not confirmed starts or episodes.',
                  'No clinical outcomes imported; clinical feasibility pending integration.',
                  'Month precision cannot establish day-level ordering or duration.'] +
                 (['Some eligible patients have no rows in supplied sources; treatment status unknown.']
                  if eligible_ids - observed_ids else []) +
                 (['Protocol metadata unavailable for some rows; no protocol inferred.']
                  if cohort.protocol.eq('unknown').any() else []),
    )
    feasibility = coverage.copy()
    feasibility['name_gate'] = feasibility['treatment'].map(
        lambda x: 'blocked_unmapped' if x in {'unknown/unmapped', 'unmapped'} else 'dictionary_review_required')
    feasibility['episode_gate'] = 'not_approved_manual_validation_required'
    feasibility['temporal_gate'] = feasibility.apply(
        lambda r: 'blocked_no_bounded_courses' if r.n_patients_duration_potential == 0
        else 'month_resolution_only_requires_validation', axis=1)
    feasibility['indication_gate'] = 'blocked_no_indication_source'
    return {
        'field_inventory': all_inventory, 'raw_slots_long': current,
        'historical_inventory': historical,
        'unassigned_doses': pd.concat(doses, ignore_index=True),
        'normalization_map': mapping, 'episode_candidates': candidates,
        'patient_coverage': coverage, 'temporal_qc': qc,
        'feasibility': feasibility, 'audit_summary': summary,
    }


def report_markdown(summary):
    lines = ['# Step 08 treatment audit — audit artifacts only', '',
             f"Rules: `{summary['rules_version']}`. No confirmed exposure or effectiveness analysis.", '',
             f"Eligibility: {summary['n_eligible_patients']} patients; "
             f"{summary['n_eligible_patients_in_sources']} represented in supplied sources; "
             f"{summary['n_eligible_patients_absent_from_sources']} absent (treatment unknown).", '',
             f"Current nonempty slots: {summary['n_current_slots']}; "
             f"candidate observations: {summary['n_candidate_observations']}; "
             f"historical slots: {summary['n_historical_slots']}; QC flags: {summary['n_qc_flags']}.", '',
             '## Limitations and blockers', '']
    lines.extend(f'- {blocker}' for blocker in summary['blockers'])
    lines.extend(['', '## Pending decision matrix', '',
                  '| Decision | Observed evidence / limitation | Condition for a later phase |',
                  '| --- | --- | --- |'])
    for protocol, counts in summary['protocol_status'].items():
        lines.append(f"| Medication availability: {protocol} | "
                     f"{counts['n_patients_in_sources']} patients in sources; "
                     f"{counts['n_patients_with_names']} with names; "
                     f"{counts['n_present_name_columns']} name columns present | "
                     'Validate source coverage; missing records remain unknown |')
    lines.extend([
        '| Name reliability | Exact versioned aliases; unknown values retained | Review normalization map and combinations |',
        '| Date reliability | Raw dates and month precision retained; QC flags exported | Resolve missing, invalid and same-month dates |',
        '| SjD indication | Indeterminate; no indication source configured | Obtain explicit indication evidence |',
        '| Episode reconstruction | One candidate per observation; no visits collapsed | Validate repeats, bounded overlaps, starts and stops |',
        '| Dose mapping | Fields profiled separately, no RX association | Document dose-slot correspondence |',
        '| Clinical feasibility | Pending integration; no clinical data imported | Separate approved phase with validated outcomes |',
        '| Additional 15-D-0051 sources | No equivalent form established by codebook | Audit EHR/BTRIS/notes; never infer never-treated |',
    ])
    return '\n'.join(lines) + '\n'


def read_table(path):
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix == '.csv':
        return pd.read_csv(path, dtype=object, keep_default_na=False)
    raise ValueError(f'Only CSV and Parquet inputs supported: {path}')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    output = args.output_dir.resolve()
    # A fresh directory prevents overwrites and stale artifacts from a previous run.
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be new or empty; use a separate audit run directory')
    specs = json.loads(args.sources_json.read_text()) if args.sources_json else [dict(
        name='visits', path=str(args.input), patient_column=config.PATIENT_ID,
        visit_column=config.VISIT_DATE,
    )]
    eligibility = read_table(args.eligibility)
    if args.eligibility_id_column not in eligibility:
        raise ValueError(f'Missing eligibility key: {args.eligibility_id_column}')
    sources = []
    for spec in specs:
        path = Path(spec['path']).resolve()
        if args.sources_json and not Path(spec['path']).is_absolute():
            path = (args.sources_json.resolve().parent / spec['path']).resolve()
        sources.append(dict(spec, path=str(path), sha256=file_hash(path), frame=read_table(path)))
    def publish_inventory(inventory):
        output.mkdir(parents=True, exist_ok=True)
        inventory.to_csv(output / '08_treatment_field_inventory.csv', index=False)
        print(inventory.groupby(['source', 'protocol', 'availability']).size().to_string())

    result = audit_sources(sources, eligibility[args.eligibility_id_column],
                           inventory_callback=publish_inventory)
    result['audit_summary']['inputs'] = [
        {key: value for key, value in spec.items() if key != 'frame'} for spec in sources]
    result['audit_summary']['eligibility'] = dict(
        path=str(args.eligibility.resolve()), sha256=file_hash(args.eligibility),
        patient_column=args.eligibility_id_column,
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        if isinstance(table, pd.DataFrame) and name != 'field_inventory':
            table.to_csv(output / f'08_treatment_{name}.csv', index=False)
    (output / '08_treatment_audit_summary.json').write_text(
        json.dumps(result['audit_summary'], indent=2, ensure_ascii=False) + '\n')
    (output / '08_treatment_audit_report.md').write_text(report_markdown(result['audit_summary']))
    print(f'Audit-only outputs: {output}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=common.DEFAULT_ANALYTIC_DATASET)
    parser.add_argument('--sources-json', type=Path, help='Explicit source/key/protocol manifest; replaces --input')
    parser.add_argument('--eligibility', type=Path, default=common.LONGITUDINAL_ID_LIST)
    parser.add_argument('--eligibility-id-column', default='patient_id')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(2, f'Audit blocked: {exc}\n')


if __name__ == '__main__':
    main()

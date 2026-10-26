#!/usr/bin/env python3
"""Literal CTDB medication slots and code decoding, extracted from the validated audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
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

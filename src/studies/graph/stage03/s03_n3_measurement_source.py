"""Audit original EDA laboratory records against pinned Graph03 inputs."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import import_module
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from tempfile import mkdtemp
from uuid import uuid4

import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import yaml

from .._pipeline import config_path as resolve_config_path, digest
from ..stage02.s02_contracts import safe_artifact
from .s03_construct_validation import _dependencies, _paths
from .s03_n3_approved_view import validate_baseline
from . import s03_n3_calendar_day as cal
from .s03_n3_decisions import evidence_path
from .s03_n3_incremental import _pop_status
from .s03_n3_phasec_adjudication import CANDIDATES, LINEAGE, PROTOCOL_ALIASES, load_phasec_inputs
from .s03_n3_resolution import _snapshot
from .s03_n3_source_adapter import KEYS, LAB_SOURCES

SCHEMA = 'graph03_n3_measurement_source_v1'
CORE = KEYS + ['source_column', 'value', 'measurement_date', 'assay_id', 'unit', 'sample_id']
BASE = KEYS + ['clinical_anchor_date', 'clinical_baseline_date', 'is_clinical_baseline']
EDA_REQUIRED = {
    'patient_id', 'matched_clinical_episode_id', 'canonical_analyte', 'semantic_mapping_status',
    'episode_match_ambiguous', 'result_valid_for_analysis', 'lab_date',
}
EDA_FIELDS = EDA_REQUIRED | {
    'assay', 'unit', 'order_identifier', 'specimen_datetime', '_lab_record_id',
    'result_raw', 'result_text', 'result_numeric_exact', 'result_operator', 'result_numeric_bound',
    'result_status', 'reported_interpretation', 'days_from_clinical_anchor',
    'order_name_original', 'cluster_name_original',
}
RESULT_LIMIT = r'(?i)\s*(?:[<>≤≥]|less than\b|greater than\b|below\b|above\b)'
SELECTED_FIELDS = set(KEYS) | {
    'lab_id', 'selected_lab_date', 'selected_unit', 'selected_unit_harmonized', 'selection_status',
    'unit_conflict', 'result_conflict', 'selected_value_numeric', 'selected_value_text',
    'selected_value_numeric_harmonized',
}
LAB_CONTEXT = {f'lab__{lab}__{suffix}' for lab in set(LAB_SOURCES.values())
               for suffix in ('measurement_date', 'unit')}
LAB_INPUTS = ('eda_long', 'step01', 'step10', 'context')
ORIGINALS = {
    'essdai_original': {'essdai__essdai_total_score', 'assessment_datetime', 'assessment_date'},
    'esspri_original': {f'esspri_questionnaire__{item}' for item in ('dryness', 'fatigue', 'pain')}
    | {'questionnaire_datetime', 'questionnaire_date'},
    'biopsy_original': {'biopsy_pathology__f_score', 'collection_datetime', 'report_datetime'},
    'protocol_original': set(PROTOCOL_ALIASES) | {'assignment_date', 'assignment_datetime'},
    'spine': set(BASE) | {'clinical_baseline_episode_id'},
}
NONLAB = {'essdai__total': 'essdai_original', 'esspri__total': 'esspri_original',
          'ext__biopsy_focus_score': 'biopsy_original'}
AUDITED_FILES = (
    'src/block_A/01_serological_profile.py', 'src/block_A/01_essdai_esspri.py',
    'src/block_A/01_extended_clinical_phenotype.py', 'src/block_A/10_build_integrated_longitudinal_dataset.py',
    'src/00_build_visit_spine.py',
)
DAY = r'\d{4}-\d{2}-\d{2}'
CLOCK = r'[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?'
OFFSET = r'(?:Z|[+-]\d{2}:?\d{2})'
UNAVAILABLE_RESULT = cal.UNAVAILABLE_RESULT


def _text(series):
    return series.astype('string').replace(r'^\s*$', pd.NA, regex=True)


def _present(series):
    return _text(series).notna()


def _utc(series):
    return pd.to_datetime(series, format='mixed', errors='coerce', utc=True)


def _date_evidence(series):
    raw = _text(series)
    # A Parquet timestamp stored at midnight does not establish intraday precision.
    day = raw.str.fullmatch(DAY, na=False) | raw.str.fullmatch(
        DAY + r'[T ]00:00(?::00(?:\.0+)?)?(?:' + OFFSET + r')?', na=False)
    explicit_offset = raw.str.fullmatch(DAY + CLOCK + OFFSET, na=False)
    local_time = raw.str.fullmatch(DAY + CLOCK, na=False)
    parsed = _utc(raw.where(day | explicit_offset | local_time))
    precision = pd.Series('INVALID_OR_UNRECOGNIZED', index=raw.index, dtype='string')
    precision.loc[raw.isna()] = 'MISSING'
    precision.loc[parsed.notna() & local_time] = 'LOCAL_TIME_ZONE_UNVERIFIED'
    precision.loc[parsed.notna() & explicit_offset] = 'EXPLICIT_OFFSET_TIMESTAMP'
    precision.loc[parsed.notna() & day] = 'DAY_OR_MIDNIGHT_ONLY'
    calendar_day = pd.to_datetime(raw.str[:10].where(parsed.notna()), errors='coerce')
    return precision, calendar_day, parsed.where(precision.eq('EXPLICIT_OFFSET_TIMESTAMP'))


def _unique(frame, keys, token):
    if not set(keys) <= set(frame):
        raise ValueError(token)
    frame = frame.copy()
    for key in keys:
        frame[key] = _text(frame[key])
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(token)
    return frame


def _reconcile(baseline, analytic, context, normalize, *, require_calendar_day=False):
    official = _unique(baseline, KEYS, 'INVALID_FROZEN_BASELINE')
    validate_baseline(official)
    step10 = _unique(analytic, KEYS, 'INVALID_STEP10_KEYS')
    flagged = step10.loc[step10.is_clinical_baseline.eq(True)]
    validate_baseline(flagged)
    context = _unique(context, KEYS, 'INVALID_STEP10_CONTEXT_KEYS')
    comparison = official[BASE].merge(flagged[BASE], on=KEYS, how='left', validate='one_to_one',
                                      indicator=True, suffixes=('', '_step10'))
    if official.empty or not comparison['_merge'].eq('both').all() or any(
        not _utc(comparison[field]).eq(_utc(comparison[field + '_step10'])).all()
        for field in BASE[2:4]
    ):
        raise ValueError('FROZEN_STEP10_BASELINE_MISMATCH')
    if require_calendar_day:
        for field in BASE[2:4]:
            day = cal.calendar_days(comparison[field], allow_time=True)
            if day.isna().any() or not day.eq(cal.calendar_days(
                    comparison[field + '_step10'], allow_time=True)).fillna(False).all():
                raise ValueError('FROZEN_STEP10_BASELINE_CALENDAR_DAY_MISMATCH')
        if not cal.calendar_days(comparison.clinical_anchor_date, allow_time=True).eq(
                cal.calendar_days(comparison.clinical_baseline_date, allow_time=True)).all():
            raise ValueError('OFFICIAL_BASELINE_CALENDAR_DAY_MISMATCH')
    keys = step10[KEYS].merge(context[KEYS], on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not keys['_merge'].eq('both').all():
        raise ValueError('STEP10_ANALYTIC_CONTEXT_EPISODE_MISMATCH')
    if 'clinical_anchor_date' in context:
        anchors = step10[KEYS + ['clinical_anchor_date']].merge(
            context[KEYS + ['clinical_anchor_date']], on=KEYS, validate='one_to_one',
            suffixes=('', '_context'))
        if not _utc(anchors.clinical_anchor_date).eq(_utc(anchors.clinical_anchor_date_context)).all():
            raise ValueError('STEP10_ANALYTIC_CONTEXT_ANCHOR_MISMATCH')
    crosswalk = step10[KEYS].assign(_patient_match=normalize(step10.patient_id))
    if (crosswalk._patient_match.isna().any()
            or crosswalk.groupby('_patient_match').patient_id.nunique().gt(1).any()):
        raise ValueError('STEP10_EDA_PATIENT_NORMALIZATION_COLLISION')
    return official, step10, context, crosswalk


def _audited_rows(eda, selected, analytic, context, baseline, specimen_id_column, *, calendar=None, records=None):
    step01_code = import_module('src.block_A.01_serological_profile')
    baseline, analytic, context, crosswalk = _reconcile(
        baseline, analytic, context, step01_code._normalize_patient_id, require_calendar_day=calendar is not None)
    selected = _unique(selected, KEYS + ['lab_id'], 'INVALID_STEP01_ANALYTE_KEYS')
    rows = eda.loc[eda.canonical_analyte.isin(LAB_SOURCES.values())].copy()
    sample_key = (_text(rows.pop(specimen_id_column)) if specimen_id_column in rows
                  else pd.Series(pd.NA, index=rows.index, dtype='string'))
    rows['original_sample_key'] = sample_key
    # Physical row position ties the ledger back to the exact hashed EDA file.
    rows['eda_row_offset'] = rows.index
    if calendar is not None:
        if 'specimen_datetime' not in rows:
            rows['specimen_datetime'] = pd.NA
        rows = cal.attach_collection_days(rows, records, calendar)
    rows['eda_patient_id'] = _text(rows.pop('patient_id'))
    rows['_patient_match'] = step01_code._normalize_patient_id(rows.eda_patient_id)
    rows['clinical_episode_id'] = _text(rows.matched_clinical_episode_id)
    rows = rows.merge(crosswalk, on=['_patient_match', 'clinical_episode_id'], how='left',
                      validate='many_to_one')
    if calendar is None:
        rows = rows.merge(baseline[KEYS + ['clinical_baseline_date']], on=KEYS, how='left',
                          validate='many_to_one')
    else:
        rows['source_episode_id'] = rows.clinical_episode_id
        rows = rows.merge(baseline[KEYS + ['clinical_baseline_date']].rename(
            columns={'clinical_episode_id': '_official_episode_id'}), on='patient_id', how='left',
            validate='many_to_one')
    ambiguous = _text(rows.episode_match_ambiguous).str.lower()
    linked = ambiguous.isin({'false', '0', 'no', 'n'})
    rows['identity_status'] = 'OFFICIAL_BASELINE_EPISODE_MATCHED'
    rows.loc[rows.clinical_baseline_date.isna(), 'identity_status'] = 'STEP10_NON_BASELINE_EPISODE'
    rows.loc[rows.patient_id.isna(), 'identity_status'] = 'EDA_EPISODE_NOT_IN_STEP10'
    wrong_patient = rows.patient_id.isna() & rows.clinical_episode_id.isin(crosswalk.clinical_episode_id)
    rows.loc[wrong_patient, 'identity_status'] = 'EDA_EPISODE_ASSOCIATED_WITH_OTHER_PATIENT'
    rows.loc[~rows._patient_match.isin(crosswalk._patient_match), 'identity_status'] = 'EDA_PATIENT_NOT_RECONCILED'
    rows.loc[rows.clinical_episode_id.isna(), 'identity_status'] = 'EDA_EPISODE_ID_MISSING'
    rows.loc[~linked, 'identity_status'] = 'EDA_EPISODE_MATCH_AMBIGUOUS_OR_UNDOCUMENTED'
    known_patient = rows._patient_match.isin(crosswalk._patient_match)
    rows['baseline_linkage_conflict'] = (wrong_patient & known_patient) | (
        rows.clinical_baseline_date.notna() & ~linked)
    if calendar is not None:
        other_episode = (rows.identity_status.eq('OFFICIAL_BASELINE_EPISODE_MATCHED') &
                         rows.clinical_episode_id.ne(rows._official_episode_id).fillna(False))
        rows.loc[other_episode, 'identity_status'] = 'SAME_PATIENT_OTHER_EPISODE_MATCHED'
    rows['mapping_candidate'] = step01_code._mapping_valid(rows) & _present(rows.semantic_mapping_status)
    rows['eda_result_valid'] = _text(rows.result_valid_for_analysis).str.lower().isin({'true', '1', 'yes', 'y'})

    for field in EDA_FIELDS - set(rows) - {'patient_id'}:
        rows[field] = pd.Series(pd.NA, index=rows.index, dtype='string')
    if calendar is None:
        rows['date_precision'], instrument_day, instant = _date_evidence(rows.lab_date)
        rows['instrument_utc_with_reported_offset'] = instant
        rows['reported_date_offset'] = _text(rows.lab_date).str.extract(
            r'(' + OFFSET + r')$', expand=False).where(instant.notna())
        baseline_utc = _utc(rows.clinical_baseline_date)
        baseline_day = pd.to_datetime(_text(rows.clinical_baseline_date).str[:10], errors='coerce')
        rows['temporal_status'] = 'DATE_OR_BASELINE_UNAVAILABLE'
        rows.loc[instrument_day.lt(baseline_day), 'temporal_status'] = 'EARLIER_CALENDAR_DAY_UNVERIFIED'
        rows.loc[instrument_day.gt(baseline_day), 'temporal_status'] = 'AFTER_BASELINE_CALENDAR_DAY'
        rows.loc[instrument_day.eq(baseline_day), 'temporal_status'] = 'SAME_DAY_ORDER_UNVERIFIED'
        comparable = instant.notna() & baseline_utc.notna() & ~instrument_day.eq(baseline_day)
        rows.loc[comparable & instant.lt(baseline_utc) & instrument_day.lt(baseline_day),
                 'temporal_status'] = 'PREBASELINE_OFFSET_CANDIDATE'
        rows.loc[comparable & instant.gt(baseline_utc), 'temporal_status'] = 'POST_BASELINE_OFFSET_COMPARISON'
        rows.loc[comparable & instant.eq(baseline_utc), 'temporal_status'] = 'BASELINE_ORDER_UNRESOLVED'
    else:
        rows['date_precision'] = rows.date_resolution
        rows['instrument_utc_with_reported_offset'] = pd.Series(pd.NaT, index=rows.index, dtype='datetime64[ns, UTC]')
        rows['calendar_age_days'] = cal.calendar_age(
            cal.calendar_days(rows.clinical_baseline_date, allow_time=True), rows.collection_calendar_day)
        rows['temporal_status'] = rows.calendar_date_status
        rows.loc[rows.calendar_age_days.gt(0), 'temporal_status'] = 'PREBASELINE_CALENDAR_DAY'
        rows.loc[rows.calendar_age_days.eq(0), 'temporal_status'] = 'SAME_DAY_EXCLUDED'
        rows.loc[rows.calendar_age_days.lt(0), 'temporal_status'] = 'POSTBASELINE_EXCLUDED'

    raw, operator = _text(rows.result_raw), _text(rows.result_operator)
    rows['censored'] = (raw.str.match(RESULT_LIMIT, na=False)
                        | operator.str.match(RESULT_LIMIT, na=False)
                        | operator.str.strip().str.upper().isin({'LT', 'LE', 'GT', 'GE', 'LTE', 'GTE'})
                        | _present(rows.result_numeric_bound))
    text_value = _text(rows.result_text).fillna(raw)
    rows['nonresult_token'] = text_value.str.fullmatch(UNAVAILABLE_RESULT, na=False)
    ledger_columns = ['eda_row_offset', 'canonical_analyte', 'identity_status', 'baseline_linkage_conflict',
                      'mapping_candidate', 'eda_result_valid', '_patient_match']
    eligible_identity = {'OFFICIAL_BASELINE_EPISODE_MATCHED'}
    if calendar is not None:
        ledger_columns += ['source_episode_id', 'btris_source', 'calendar_date_status', 'temporal_status']
        eligible_identity.add('SAME_PATIENT_OTHER_EPISODE_MATCHED')
    ledger = rows[ledger_columns].copy()
    tables = []
    sources = LAB_SOURCES if calendar is None else {source: LAB_SOURCES[source] for source in calendar['windows']}
    for source, lab in sources.items():
        scoped = rows.loc[rows.canonical_analyte.eq(lab)
                          & rows.identity_status.isin(eligible_identity)].copy()
        table = _compare_selected(scoped, selected.loc[selected.lab_id.eq(lab)], analytic, context, source, lab)
        table['source_column'] = source
        table['value'] = (_text(table.result_numeric_exact).where(~table.censored)
                          if source.endswith('__value') else _text(table.result_text).fillna(_text(table.result_raw)))
        table['measurement_date'] = pd.Series(pd.NaT, index=table.index, dtype='datetime64[ns, UTC]')
        table['assay_id'] = pd.Series(pd.NA, index=table.index, dtype='string')
        table['sample_id'] = pd.Series(pd.NA, index=table.index, dtype='string')
        table['unit'] = _text(table.unit)
        table['source_artifact_status'] = 'SOURCE_CANDIDATE_UNAPPROVED'
        table['clinical_approval_status'] = 'PENDING_HUMAN_REVIEW'
        if calendar is not None:
            table['clinical_episode_id'] = table.pop('_official_episode_id')
            table['test_definition_id'] = pd.Series(pd.NA, index=table.index, dtype='string')
        tables.append(table)
    candidates = pd.concat(tables, ignore_index=True).drop(columns=['_patient_match'])
    return candidates, ledger


def _compare_selected(rows, selected, analytic, context, source, lab):
    date, unit = f'lab__{lab}__measurement_date', f'lab__{lab}__unit'
    selected_numeric = ('selected_value_numeric_harmonized'
                        if 'selected_value_numeric_harmonized' in selected else 'selected_value_numeric')
    selected_unit = 'selected_unit_harmonized' if 'selected_unit_harmonized' in selected else 'selected_unit'
    selected_columns = {
        'selected_lab_date': 'step01_date', 'selection_status': 'step01_selection_status',
        'unit_conflict': 'step01_unit_conflict', 'result_conflict': 'step01_result_conflict',
        selected_unit: 'step01_unit',
        (selected_numeric if source.endswith('__value') else 'selected_value_text'): 'step01_value',
    }

    def view(frame, fields):
        return frame.reindex(columns=KEYS + list(fields)).rename(columns=fields)

    linked = rows.merge(view(selected, selected_columns), on=KEYS, how='left', validate='many_to_one')
    linked = linked.merge(view(context, {date: 'step10_context_date', unit: 'step10_context_unit'}),
                          on=KEYS, how='left', validate='many_to_one')
    linked = linked.merge(view(analytic, {source: 'step10_value'}), on=KEYS, how='left', validate='many_to_one')
    # Calendar agreement is evidence about selection lineage only.
    _, original_day, _ = _date_evidence(linked.lab_date)
    _, selected_day, _ = _date_evidence(linked.step01_date)
    selected_date = _text(linked.step01_date)
    context_date = _text(linked.step10_context_date)
    same_kind = selected_date.str.contains(OFFSET + '$', na=False).eq(
        context_date.str.contains(OFFSET + '$', na=False))
    same_date = same_kind & _utc(selected_date).eq(_utc(context_date))
    linked['step01_context_date_status'] = 'STEP01_STEP10_DATE_DISCORDANT'
    linked.loc[same_date, 'step01_context_date_status'] = 'STEP01_STEP10_DATE_CONCORDANT'
    linked.loc[selected_date.isna() | context_date.isna(), 'step01_context_date_status'] = 'SELECTED_DATE_MISSING'
    linked['eda_matches_selected_day'] = original_day.eq(selected_day)
    linked['step01_selected_after_baseline'] = selected_day.gt(
        pd.to_datetime(_text(linked.clinical_baseline_date).str[:10], errors='coerce'))
    left, right = linked.step01_value, linked.step10_value
    if source.endswith('__value'):
        left, right = (pd.to_numeric(series, errors='coerce') for series in (left, right))
        raw = pd.to_numeric(linked.result_numeric_exact, errors='coerce').where(~linked.censored)
    else:
        left, right = _text(left), _text(right)
        raw = _text(linked.result_text).fillna(_text(linked.result_raw))
    linked['step01_step10_value_status'] = 'STEP01_STEP10_VALUE_DISCORDANT'
    linked.loc[left.eq(right).fillna(False), 'step01_step10_value_status'] = 'STEP01_STEP10_VALUE_CONCORDANT'
    linked.loc[left.isna() | right.isna(), 'step01_step10_value_status'] = 'SELECTED_VALUE_MISSING'
    linked['eda_matches_selected_value'] = raw.eq(left).fillna(False)
    linked['step01_value'] = _text(linked.step01_value)
    linked['step10_value'] = _text(linked.step10_value)
    original_unit, selected_units, context_units = (
        _text(linked[field]) for field in ('unit', 'step01_unit', 'step10_context_unit'))
    linked['eda_matches_selected_unit'] = original_unit.eq(selected_units).fillna(False)
    linked['step01_matches_context_unit'] = selected_units.eq(context_units).fillna(False)
    return linked


def _outcome_audit(candidates, ledger, inventory, minimum):
    outcomes = {}
    counts = {}
    for source, family in CANDIDATES.items():
        if source not in LAB_SOURCES:
            upstream = NONLAB[source]
            outcomes[source] = dict(status='BLOCKED', reason='ORIGINAL_INSTRUMENT_DATE_NOT_VERIFIED',
                                    search_status='NOT_VERIFIED', source=upstream,
                                    original_fields=LINEAGE[family][1], value_rule=LINEAGE[family][2])
            continue
        lab = LAB_SOURCES[source]
        available = candidates is not None
        outcomes[source] = dict(
            status='CANDIDATE_UNAPPROVED' if available else 'UNAVAILABLE',
            reason='INSTRUMENT_PROVENANCE_AND_SIGNOFF_PENDING' if available else 'UPSTREAM_NOT_READY',
            lab_id=lab, original_fields='result_numeric_exact' if source.endswith('__value')
            else 'result_text; result_raw como respaldo candidato',
            value_rule='exacto sin censura ni armonización' if source.endswith('__value')
            else 'categoría original sin recodificar',
            eda_schema_status=inventory['eda_long']['status'],
            field_status={field: 'UNRESOLVED' if available else 'NOT_AVAILABLE' for field in CORE})
        if not available:
            continue
        rows = candidates.loc[candidates.source_column.eq(source)]
        if rows.empty:
            outcomes[source].update(status='BLOCKED', reason='NO_RECONCILED_ORIGINAL_OBSERVATIONS')
        else:
            outcomes[source]['field_status'].update(patient_id='VERIFIED', clinical_episode_id='VERIFIED',
                source_column='CANDIDATE', value='CANDIDATE', unit='CANDIDATE')
        # A repeated source row cannot increase a disclosure denominator.
        identities = ledger.loc[ledger.canonical_analyte.eq(lab)]
        by_patient = rows.groupby('patient_id')
        partitions = {
            'eda_identity_patients': identities.groupby('identity_status')._patient_match.nunique().to_dict(),
            'date_precision_patients': rows.groupby('date_precision').patient_id.nunique().to_dict(),
            'time_status_patients': rows.groupby('temporal_status').patient_id.nunique().to_dict(),
            'date_lineage_patients': rows.groupby('step01_context_date_status').patient_id.nunique().to_dict(),
            'value_lineage_patients': rows.groupby('step01_step10_value_status').patient_id.nunique().to_dict(),
            'assay_count_per_patient': by_patient.assay.nunique().clip(upper=2).value_counts().to_dict(),
            'unit_count_per_patient': by_patient.unit.nunique().clip(upper=2).value_counts().to_dict(),
        }
        raw_status = _text(rows.result_status)
        status_groups = pd.Series('OTHER', index=rows.index)
        status_groups.loc[raw_status.isna()] = 'MISSING'
        status_groups.loc[raw_status.str.fullmatch(r'(?i)final|verified|corrected', na=False)] = 'FINAL_OR_CORRECTED'
        status_groups.loc[raw_status.str.fullmatch(r'(?i)preliminary|prelim|pending', na=False)] = 'PRELIMINARY_OR_PENDING'
        partitions['result_status_patients'] = rows.groupby(status_groups).patient_id.nunique().to_dict()
        duplicate_key = KEYS + ['original_sample_key']
        repeated = rows.original_sample_key.notna() & rows.duplicated(duplicate_key, keep=False)
        time_key = KEYS + ['instrument_utc_with_reported_offset']
        same_time = rows.instrument_utc_with_reported_offset.notna() & rows.duplicated(time_key, keep=False)
        exact_columns = ['value', 'assay', 'unit', 'original_sample_key', 'censored', 'result_operator',
                         'result_numeric_bound', 'result_raw', 'result_text', 'result_numeric_exact']
        variants = rows.drop_duplicates(time_key + exact_columns)
        tie_keys = pd.MultiIndex.from_frame(variants.loc[variants.duplicated(time_key, keep=False), time_key])
        discordant = same_time & pd.MultiIndex.from_frame(rows[time_key]).isin(tie_keys)
        flags = rows[['censored', 'nonresult_token', 'step01_selected_after_baseline',
                      'eda_matches_selected_day', 'eda_matches_selected_value', 'eda_matches_selected_unit',
                      'step01_matches_context_unit', 'mapping_candidate', 'eda_result_valid']].copy()
        flags['assay_label_present'] = _present(rows.assay)
        flags['sample_key_present'] = _present(rows.original_sample_key)
        flags['repeated_sample_key'] = repeated
        flags['repeated_precise_time'] = same_time
        flags['discordant_precise_time'] = discordant
        flags['exact_precise_duplicate'] = same_time & rows.duplicated(time_key + exact_columns, keep=False)
        for flag in ('step01_unit_conflict', 'step01_result_conflict'):
            flags[flag] = _text(rows[flag]).str.lower().map({'true': True, 'false': False})
        for field in flags:
            presence = flags[field].astype('boolean').astype('string').fillna('unknown')
            partitions[field + '_patients'] = rows.groupby(presence).patient_id.nunique().to_dict()
        counts[source] = partitions
    # Withhold all linked partitions if any small cell could be reconstructed
    # across columns or across the two encodings of one assay.
    hidden = ((ledger is not None and ledger._patient_match.isna().any()) or
              any(0 < n < minimum for tables in counts.values() for cells in tables.values() for n in cells.values()))
    status = 'NOT_ASSESSED' if candidates is None else 'WITHHELD_SMALL_CELLS' if hidden else 'RESTRICTED_AGGREGATES'
    return outcomes, dict(status=status,
                          by_outcome={} if hidden else counts)


def _calendar_report(candidates, settings, inputs, minimum, aggregates, reviewed_ledger):
    metrics = ('day_available', 'eligible_prebaseline', 'same_day_excluded', 'postbaseline_excluded',
               'ambiguity_excluded', 'test_definition_verified')
    report = []
    if candidates is None:
        return [dict(source_column=source, disclosure_status='NOT_ASSESSED', **dict.fromkeys(metrics))
                for source in settings['windows']]
    patients = inputs['frozen']['baseline_raw'].patient_id.astype('string')
    partitions = {}
    for source in settings['windows']:
        rows = candidates.loc[candidates.source_column.eq(source)]
        reviewed = (reviewed_ledger.loc[reviewed_ledger.source_column.eq(source)]
                    if reviewed_ledger is not None else None)
        flags = {
            'day_available': rows.collection_calendar_day.notna(),
            'eligible_prebaseline': rows.calendar_disposition.eq('LATEST_ELIGIBLE_DAY'),
            'same_day_excluded': rows.temporal_status.eq('SAME_DAY_EXCLUDED'),
            'postbaseline_excluded': rows.temporal_status.eq('POSTBASELINE_EXCLUDED'),
            'ambiguity_excluded': rows.calendar_disposition.eq('AMBIGUOUS_SAME_DAY'),
        }
        if reviewed is not None and not reviewed.empty:
            flags['eligible_prebaseline'] = rows.eda_row_offset.isin(
                reviewed.loc[reviewed.reviewed_disposition.eq('RETAINED_FOR_03E_SELECTION'), 'eda_row_offset'])
            flags['test_definition_verified'] = rows.eda_row_offset.isin(
                reviewed.loc[reviewed.test_definition_verified, 'eda_row_offset'])
        for name, flag in flags.items():
            partitions[source, name] = patients.isin(rows.loc[flag, 'patient_id'])
        report.append(dict(source_column=source, disclosure_status='RESTRICTED_AGGREGATES', **{
            name: int(partitions[source, name].sum()) if name in flags else None for name in metrics}))
    # Cross-outcome and within-outcome differences can disclose small subsets.
    small = pd.DataFrame(partitions).value_counts().lt(minimum).any()
    if small or aggregates['status'] == 'WITHHELD_SMALL_CELLS':
        for row in report:
            row.update(disclosure_status='WITHHELD_SMALL_CELLS', **dict.fromkeys(metrics))
    return report


def _readiness(inputs, outcomes, validation, reviewed_sources, *, calendar=None):
    representations = inputs['manifest']['eligible_representation_ids'] if inputs else ['R0', 'R1', 'R2', 'R3', 'R4']
    if inputs:
        features = inputs['frozen']['feature_sets']
        graph = _dependencies(features, inputs['registry'], validation)
        roles = inputs['roles'].set_index(['representation_id', 'source_column']).tier.to_dict()
        inventory = inputs['tables']['source_inventory'].set_index(
            ['representation_id', 'source_column']).to_dict('index')
    rows = []
    for rep in representations:
        defining = set(features.loc[features.representation_id.eq(rep) & features.included, 'source_column']) if inputs else set()
        for source in CANDIDATES:
            independence = 'FROZEN_REPRESENTATION_NOT_VERIFIED'
            if inputs:
                tier = roles.get((rep, source))
                dependent = tier in {'N1_DEFINING', 'N2_RELATED'} or bool(set(_paths(graph, source)) & defining)
                independence = ('BLOCKED_DISCOVERY_FEATURE_OR_PROXY' if dependent else
                                'PENDING_REPRESENTATION_SPECIFIC_REVIEW' if tier else 'FROZEN_ROLE_UNAVAILABLE')
            pop_inventory = inventory.get((rep, source), {}) if inputs else {}
            pop = (_pop_status(source, validation, rep, {})[0] == 'POP_BASELINE_COMPARATOR_INELIGIBLE'
                   or pop_inventory.get('pop_construct_dependency') is True)
            instrument_reviewed = source in reviewed_sources
            rows.append(dict(
                representation_id=rep, source_column=source, availability=outcomes[source]['status'],
                baseline='VERIFIED_FROZEN_BASELINE' if inputs else 'FROZEN_BASELINE_NOT_VERIFIED',
                provenance='SIGNED_SOURCE_PROVENANCE_VERIFIED' if instrument_reviewed else outcomes[source]['reason'],
                temporal='SIGNED_WINDOW_CHECKED_BY_PRODUCER' if instrument_reviewed else 'INSTRUMENT_WINDOW_NOT_VERIFIED',
                independence=independence, protocol='RESEARCH_PROTOCOL_ASSIGNMENT_NOT_VERIFIED',
                pop='BLOCKED_OUTCOME_DEFINES_POP' if pop else 'SEPARATE_POP_REVIEW_REQUIRED',
                topology='R2_EXPLORATORY' if rep == 'R2' else 'PENDING_TOPOLOGY_REVIEW',
                clinical_review='PENDING_03E_PAIR_DECISIONS' if calendar is None else 'PAIR_REVIEW_PENDING',
                support='NOT_ASSESSED_BEFORE_03E_PREFLIGHT',
            ))
    return rows


def _file_hash(path):
    return digest(path) if path is not None and path.is_file() else None


def _inventory(paths, specimen_id_column, calendar):
    lab_fields = EDA_FIELDS | ({specimen_id_column} if specimen_id_column else set())
    if calendar is not None:
        lab_fields |= set(cal.btris_columns(calendar, 'eda'))
    fields = dict(eda_long=lab_fields, step01=SELECTED_FIELDS,
                  step10=set(BASE) | set(CANDIDATES) | {'clinical_baseline_episode_id'},
                  context=set(KEYS) | LAB_CONTEXT | set(PROTOCOL_ALIASES) | {'clinical_anchor_date'}, **ORIGINALS)
    info, available = {}, {}
    for role, path in paths.items():
        if role in cal.BTRIS_ROLES:
            files = cal.file_hashes(path)
            info[role] = dict(path=str(path) if path is not None else None, file_sha256=files,
                              sha256=cal.fingerprint(files) if files else None,
                              status='CANDIDATE' if files else 'NOT_AVAILABLE')
            continue
        sha = _file_hash(path)
        record = dict(path=str(path) if path is not None else None,
                      sha256=sha, status='NOT_AVAILABLE' if sha is None else 'CANDIDATE')
        info[role] = record
        if sha is not None and role in fields:
            try:
                if path.suffix.lower() == '.csv' and role in ORIGINALS:
                    schema = {name: 'csv_unspecified' for name in pd.read_csv(path, nrows=0)}
                else:
                    schema = {field.name: str(field.type) for field in pq.read_schema(path)}
                available[role] = set(schema)
                record['fields'] = {
                    field: dict(status='CANDIDATE', dtype=schema[field]) if field in schema
                    else dict(status='NOT_AVAILABLE') for field in sorted(fields[role])
                }
            except (ValueError, OSError, TypeError):
                record['status'] = 'UNRESOLVED'
                record['reason'] = 'SCHEMA_UNREADABLE'
    return info, available


def _read_inputs(paths, available, specimen_id_column, calendar):
    required = dict(eda_long=EDA_REQUIRED, step01=set(KEYS + ['lab_id', 'selected_lab_date']),
                    step10=set(BASE), context=set(KEYS))
    if any(role not in available for role in LAB_INPUTS):
        return None, 'UPSTREAM_MISSING_OR_UNREADABLE'
    if any(not columns <= available[role] for role, columns in required.items()):
        return None, 'UPSTREAM_REQUIRED_COLUMNS_MISSING'
    projected = dict(
        eda_long=EDA_FIELDS | ({specimen_id_column} if specimen_id_column else set()),
        step01=SELECTED_FIELDS,
        step10=set(BASE) | set(LAB_SOURCES) | {'clinical_baseline_episode_id'},
        context=set(KEYS) | LAB_CONTEXT | {'clinical_anchor_date'},
    )
    if calendar is not None:
        fields = set(cal.btris_columns(calendar, 'eda'))
        if not fields <= available['eda_long']:
            return None, 'BTRIS_EDA_LINKAGE_COLUMNS_REQUIRED'
        projected['eda_long'] |= fields
    try:
        return [pd.read_parquet(paths[role], columns=sorted(projected[role] & available[role]))
                for role in LAB_INPUTS], None
    except (ValueError, OSError, TypeError):
        return None, 'UPSTREAM_PARQUET_UNREADABLE'


def _read_frozen(paths, phasec_config):
    inputs = load_phasec_inputs(paths['graph03_run'], paths['graph02_release'], phasec_config)
    if inputs['current']:
        raise ValueError('SIGNED_SOURCE_REQUIRES_EXPLICIT_HISTORICAL_RELEASES')
    release = inputs['release']
    manifest = inputs['manifest']
    actual_baseline = safe_artifact(release.parent, manifest['relative_paths']['baseline_raw'])
    if paths['baseline'] is not None and digest(paths['baseline']) != digest(actual_baseline):
        raise ValueError('EXPLICIT_BASELINE_DISAGREES_WITH_FROZEN_RELEASE')
    summary = json.loads((paths['graph03_run'] / '03_graph_results_summary.json').read_text())
    for name, sha in summary['sha256_by_artifact'].items():
        if digest(safe_artifact(paths['graph03_run'], name)) != sha:
            raise ValueError('GRAPH03_ARTIFACT_SHA_MISMATCH')
    inputs['registry'] = pd.read_csv(safe_artifact(release.parent, manifest['relative_paths']['clinical_registry']))
    inputs['baseline_path'] = actual_baseline
    validate_baseline(inputs['frozen']['baseline_raw'])
    return inputs


def _private_directory(path):
    path.mkdir(mode=0o700, exist_ok=True)
    stat = path.stat()
    if path.is_symlink() or not path.is_dir() or stat.st_uid != os.geteuid() or stat.st_mode & 0o077:
        raise ValueError('OUTPUT_DIRECTORY_PERMISSIONS_UNSAFE')


def _configured_sources(overrides, config, config_path, root, calendar):
    roles = (*LAB_INPUTS, *ORIGINALS, 'baseline', 'graph02_release', 'graph03_run')
    if calendar is not None:
        roles += cal.BTRIS_ROLES
    settings = config.get('source_paths', {})
    if (not isinstance(settings, dict) or not set(settings) <= set(roles)
            or not set(overrides) <= set(roles)
            or any(value is not None and (not isinstance(value, str) or not value.strip())
                   for value in settings.values())):
        raise ValueError('INVALID_MEASUREMENT_SOURCE_PATHS')
    data_root = Path(os.environ.get('OBJ1_SJD_ROOT', root)).resolve()
    paths = {}
    for role in roles:
        if role in overrides:
            paths[role] = Path(overrides[role]).resolve() if overrides[role] is not None else None
        else:
            value = settings.get(role)
            path = resolve_config_path(config_path, value) if value is not None else None
            paths[role] = (data_root / path.relative_to(root)).resolve() if path and path.is_relative_to(root) else path
    return paths


@contextmanager
def _staged_run(root, run_id):
    previous_umask = os.umask(0o077)
    lock = temporary = None
    try:
        _private_directory(root)
        runs = root / 'runs'
        _private_directory(runs)
        output = runs / run_id
        guard = runs / ('.' + run_id + '.lock')
        guard.mkdir(mode=0o700)
        lock = guard
        if output.exists() or output.is_symlink():
            raise FileExistsError('MEASUREMENT_RUN_ALREADY_EXISTS')
        temporary = Path(mkdtemp(prefix='.' + run_id + '.', dir=runs))
        yield temporary, output
        # Both writers and readers see either a complete run or no run.
        if output.exists() or output.is_symlink():
            raise FileExistsError('MEASUREMENT_RUN_ALREADY_EXISTS')
        temporary.rename(output)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        if lock is not None:
            lock.rmdir()
        os.umask(previous_umask)


def run_source(mode, paths, output_root, config_path, repository_root, *, run_id=None,
               review_path=None, signature_path=None, trust_key=None, verify_calendar_formats=False):
    """Publish a private audit, candidate, or externally signed evidence release."""
    from .s03_n3_measurement_review import reviewed_measurements, validate_specimen_column

    root = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    validation_path = root / 'src/studies/graph/configs/validation_03.yaml'
    phasec_path = root / 'src/studies/graph/configs/n3_phasec_review.yaml'
    code_paths = set(root / name for name in AUDITED_FILES) | set((root / 'src/studies/graph').rglob('*.py'))
    code = {path.relative_to(root).as_posix(): path for path in sorted(code_paths)}
    rules = {'source_config': config_path, 'validation': validation_path, 'phasec_config': phasec_path}
    code_hashes = {name: digest(path) for name, path in code.items()}
    rules_hashes = {name: digest(path) for name, path in rules.items()}
    config, validation, phasec = (yaml.safe_load(path.read_text()) for path in rules.values())
    minimum = config.get('minimum_disclosure_cell')
    schema = config.get('schema_version')
    if (schema not in {SCHEMA, cal.SOURCE_VERSION} or config.get('privacy_class') != 'INTERNAL_RESTRICTED'
            or config.get('public_export') is not False or type(minimum) is not int or minimum < 5):
        raise ValueError('INVALID_MEASUREMENT_SOURCE_CONFIG')
    calendar = cal.validate_settings(config.get('calendar_day')) if schema == cal.SOURCE_VERSION else None
    specimen_id_column = None if calendar is not None else config.get('eda_specimen_id_column')
    if calendar is None:
        validate_specimen_column(specimen_id_column)
    if mode not in {'audit-only', 'build-candidate', 'build-reviewed'}:
        raise ValueError('UNKNOWN_MEASUREMENT_SOURCE_MODE')
    documents = {name: Path(path).resolve() for name, path in (
        ('review', review_path), ('signature', signature_path), ('trust_key', trust_key)) if path is not None}
    if (mode == 'build-reviewed' and set(documents) != {'review', 'signature', 'trust_key'}) or (
        mode != 'build-reviewed' and documents
    ):
        raise ValueError('SIGNED_REVIEW_INPUTS_REQUIRED_ONLY_IN_REVIEWED_MODE')

    paths = _configured_sources(paths, config, config_path, root, calendar)
    output_root = Path(output_root).absolute()
    if output_root.is_symlink():
        raise ValueError('OUTPUT_DIRECTORY_PERMISSIONS_UNSAFE')
    output_root = output_root.resolve()
    graph02, graph03 = paths['graph02_release'], paths['graph03_run']
    if graph02 is not None and (graph02.is_dir() or not graph02.exists() and graph02.suffix != '.json'):
        graph02 = paths['graph02_release'] = graph02 / '02_graph_release_manifest.json'
    rel_roots = [p for p in (graph02.parent if graph02 else None, graph03) if p is not None]
    if len(rel_roots) == 2 and any(a.is_relative_to(b) for a, b in (rel_roots, rel_roots[::-1])):
        raise ValueError('DISTINCT_FROZEN_INPUTS_REQUIRED')
    if (output_root.is_relative_to(root)
            or any(output_root.is_relative_to(p) or p.is_relative_to(output_root)
                   for p in [*rel_roots, *filter(None, paths.values()), *rules.values(), *documents.values()])):
        raise ValueError('OUTPUT_OVERLAPS_REPOSITORY_OR_INPUT')
    run_id = run_id if run_id is not None else datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('INVALID_MEASUREMENT_RUN_ID')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()

    with _staged_run(output_root, run_id) as (staging, output):
        active_frozen = [p for p in rel_roots if p.is_dir()]
        frozen_before = _snapshot(active_frozen)
        inventory, available = _inventory(paths, specimen_id_column, calendar)
        source_hashes = {role: item['sha256'] for role, item in inventory.items()}
        inputs = candidates = ledger = None
        blocker = None
        if graph02 and graph02.is_file() and graph03 and graph03.is_dir():
            inputs = _read_frozen(paths, phasec)
            inventory['graph02_release']['status'] = inventory['graph03_run']['status'] = 'VERIFIED'
            inventory['baseline'] = dict(path=str(inputs['baseline_path']),
                                         sha256=digest(inputs['baseline_path']), status='VERIFIED')
        else:
            blocker = 'FROZEN_RELEASES_NOT_AVAILABLE'
        frames, upstream_blocker = _read_inputs(paths, available, specimen_id_column, calendar)
        blockers = list(dict.fromkeys(b for b in (blocker, upstream_blocker) if b))
        records = None
        if calendar is not None:
            if all(source_hashes[role] for role in cal.BTRIS_ROLES):
                try:
                    records = cal.btris_provenance(paths, calendar, verify_formats=verify_calendar_formats)
                except (ValueError, OSError) as exc:
                    reason = str(exc)
                    if verify_calendar_formats and re.fullmatch(
                            r'BTRIS_COLLECTION_(COLUMNS_REQUIRED|DATE_FORMAT_UNVERIFIED|OBSERVATION_MAPPING_CONFLICT)'
                            r'_BTRIS_(RAW|INTERMEDIATE)_(11D|15D)',
                            reason):
                        blockers.append(reason)
                    else:
                        blockers.append('BTRIS_COLLECTION_PROVENANCE_UNRESOLVED')
            else:
                blockers.append('BTRIS_COLLECTION_FILES_NOT_AVAILABLE')
        if not blockers:
            candidates, ledger = _audited_rows(*frames, inputs['frozen']['baseline_raw'], specimen_id_column,
                                               calendar=calendar, records=records)
            if calendar is not None:
                candidates['calendar_disposition'] = pd.Series(index=candidates.index, dtype='string')
                for source, window in calendar['windows'].items():
                    rows = candidates.loc[candidates.source_column.eq(source)]
                    candidates.loc[rows.index, 'calendar_disposition'] = cal.calendar_dispositions(rows, window)
            if candidates.empty:
                blockers.append('NO_RECONCILED_ORIGINAL_OBSERVATIONS')
        frozen_hashes = {role + '/' + path.relative_to(directory).as_posix(): sha
                         for role, directory in [('graph02', graph02.parent if graph02 else None), ('graph03', graph03)]
                         for path, sha in frozen_before.items() if directory is not None and path.is_relative_to(directory)}
        binding = dict(source_sha256=source_hashes, frozen_sha256=frozen_hashes,
                       code_sha256=code_hashes, configuration_sha256=rules_hashes,
                       implementation_commit=commit,
                       runtime=dict(python=platform.python_version(), pandas=pd.__version__, pyarrow=pyarrow.__version__))
        outcomes, aggregates = _outcome_audit(candidates, ledger, inventory, minimum)
        reviewed = {}
        if mode == 'build-candidate' and candidates is not None and not candidates.empty:
            candidates.to_parquet(staging / 'unapproved_observations.parquet', index=False)
            columns = ['eda_row_offset', 'source_column', 'date_precision', 'step01_context_date_status',
                       'step01_step10_value_status', 'source_artifact_status']
            columns += ['calendar_disposition'] if calendar is not None else ['temporal_status']
            linkage = candidates[columns]
            ledger = ledger.merge(linkage, on='eda_row_offset', how='left', validate='one_to_many')
        if mode == 'build-candidate' and ledger is not None:
            ledger.drop(columns='_patient_match').to_parquet(staging / 'private_ledger.parquet', index=False)
        if mode == 'build-reviewed':
            if blockers or candidates is None:
                raise ValueError('REVIEWED_BUILD_SOURCE_CONTRACT_NOT_READY')
            measurements, reviewed_ledger, reviewed = reviewed_measurements(
                candidates, ledger, inputs['frozen']['baseline_raw'], binding, specimen_id_column,
                documents, calendar=calendar)
            measurements.to_parquet(staging / 'reviewed_instrument_measurements.parquet', index=False)
            reviewed_ledger.to_parquet(staging / 'private_reviewed_ledger.parquet', index=False)
            for source in reviewed['sources']:
                fields = dict.fromkeys(CORE if calendar is None else cal.CORE, 'VERIFIED')
                if calendar is not None:
                    fields.update(assay_id='NOT_IDENTIFIED', sample_id='NOT_IDENTIFIED')
                outcomes[source].update(status='VERIFIED' if calendar is None else 'SOURCE_REVIEWED',
                                        reason='SIGNED_SOURCE_PROVENANCE_VERIFIED', field_status=fields)
        if calendar is not None:
            eligibility = _calendar_report(candidates, calendar, inputs, minimum, aggregates,
                                           reviewed_ledger if reviewed else None)
            pd.DataFrame(eligibility).to_csv(staging / 'calendar_day_eligibility.csv', index=False)
            aggregates = dict(status=eligibility[0]['disclosure_status'], by_outcome={})
            for source in calendar['windows']:
                if source not in reviewed.get('sources', []) and outcomes[source]['status'] == 'CANDIDATE_UNAPPROVED':
                    outcomes[source]['reason'] = 'CALENDAR_DAY_SOURCE_REVIEW_PENDING'
                    outcomes[source]['field_status'] = dict.fromkeys(cal.CORE, 'UNRESOLVED')
        pd.DataFrame(_readiness(inputs, outcomes, validation, reviewed.get('sources', []), calendar=calendar)).to_csv(
            staging / 'readiness.csv', index=False)
        status = ('SOURCE_REVIEWED_PENDING_03E_DECISIONS' if reviewed else
                  'SOURCE_CONTRACT_NOT_READY' if blockers else
                  'SOURCE_AUDIT_COMPLETE' if mode == 'audit-only' else 'SOURCE_CANDIDATE_UNAPPROVED')
        manifest = dict(
            schema_version=schema, run_id=run_id, mode=mode, status=status, privacy_class='INTERNAL_RESTRICTED',
            public_export=False, estimators_run=0, candidate_is_approved=False, eligible_for_inference=False,
            inference_status='NO_APPROVED_ESTIMABLE_CONTRAST', blockers=blockers,
            minimum_disclosure_cell=minimum, inputs=inventory, review_binding=binding,
            frozen_verified_unchanged=inputs is not None,
            outcomes=outcomes, aggregates=aggregates, reviewed_source=reviewed,
            field_dictionary={
                'patient_id': 'patient_id oficial; normalización documentada de Step01 solo para la unión EDA',
                'clinical_episode_id': 'matched_clinical_episode_id de EDA reconciliado con Step10 y baseline congelado',
                'measurement_date': 'lab_date original candidato; specimen_datetime conservado como evidencia separada',
                'assay_id': 'requiere identidad documentada; el campo assay es un candidato',
                'sample_id': 'requiere clave original documentada; orden, fecha y fila son evidencia separada',
                'unit': 'unit original EDA; sin armonización por el productor',
                'value': 'representación original según el endpoint; conservar operador, límite y categoría',
                'source_column': 'correspondencia fija con canonical_analyte; sin nombres inferidos',
                'protocol': 'asignación de investigación y fecha efectiva requieren revisión separada',
            },
            step01_selection_rule='menor distancia absoluta al anchor, preferencia pre/anchor en empate; posible mediana',
            output_sha256={p.name: digest(p) for p in staging.iterdir()},
        )
        if calendar is not None:
            manifest.update(measurement_contract=cal.CONTRACT, eligibility_report='calendar_day_eligibility.csv',
                            frozen_sha256_before=frozen_hashes, frozen_sha256_after=frozen_hashes)
            manifest['field_dictionary'].update(
                clinical_episode_id='episodio oficial del baseline congelado',
                source_episode_id='episodio original de EDA, reconciliado por paciente y episodio con Step10',
                collection_calendar_day='día sin hora, cotejado con BTRIS raw, intermedio y EDA lab_date',
                test_definition_id='definición comparable autorizada en el crosswalk firmado',
                assay_id='nulo cuando la plataforma no se identifica',
                sample_id='nulo cuando el espécimen no se identifica')
            manifest['field_dictionary'].pop('measurement_date')
        (staging / 'provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
        protected = {**{code[k]: h for k, h in code_hashes.items()},
                     **{rules[k]: h for k, h in rules_hashes.items()},
                     **{paths[k]: h for k, h in source_hashes.items() if paths[k] is not None and k not in cal.BTRIS_ROLES}}
        if reviewed:
            protected.update({documents[k]: h for k, h in reviewed['document_sha256'].items()})
            protected.update({evidence_path(documents['review'].parent, k): h
                              for k, h in reviewed['evidence_sha256'].items()})
        if (_snapshot([p for p in rel_roots if p.is_dir()]) != frozen_before
                or any(_file_hash(path) != sha for path, sha in protected.items())
                or (calendar is not None and any(cal.file_hashes(paths[k]) != inventory[k]['file_sha256']
                                                 for k in cal.BTRIS_ROLES))
                or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != commit):
            raise ValueError('MEASUREMENT_SOURCE_INPUT_CHANGED_DURING_RUN')
        for path in staging.iterdir():
            path.chmod(0o600)
    return dict(status=status, run_id=run_id, estimators_run=0, candidate_is_approved=False)

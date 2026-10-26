"""Step 08 stays isolated and preserves uncertainty and upstream artifacts."""
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[3]
SPEC = importlib.util.spec_from_file_location('treatment_audit', ROOT / 'src/block_A/audit/08_medications/legacy/08_treatment_audit.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def source(frame, **metadata):
    return dict(name='visits', path='visits.csv', patient_column=audit.config.PATIENT_ID,
                visit_column=audit.config.VISIT_DATE, frame=frame, **metadata)


def frame(**columns):
    length = len(next(iter(columns.values()))) if columns else 1
    return pd.DataFrame({audit.config.PATIENT_ID: ['p1'] * length,
                         audit.config.VISIT_DATE: ['2020-06-01'] * length, **columns})


def execute(data, **metadata):
    return audit.audit_sources([source(data, **metadata)], ['p1', 'p2'])


def test_literal_irregular_columns_no_silent_alias_or_dose_mapping():
    data = frame(**{'medications__rx_3_name': ['HCQ'],
                    'medications__rx_3_frequency': ['1'],
                    'medications__rx_3_frequency_': ['12'],
                    'medications__anticogaulant_1_dose': ['5 mg']})
    result = execute(data)
    slot = result['raw_slots_long'].iloc[0]
    assert slot.frequency_column == 'medications__rx_3_frequency_'
    assert slot.frequency_label == 'Weekly'
    assert audit.MEDICATION_FREQUENCY_BY_SLOT[5].endswith('_')
    assert audit.MEDICATION_FREQUENCY_BY_SLOT[6].endswith('_')
    assert result['unassigned_doses'].iloc[0].slot_mapping == 'unresolved_not_assigned'
    assert not any('dose' in x for x in result['episode_candidates'].columns)
    without_exact = execute(data.drop(columns='medications__rx_3_frequency_'))
    assert without_exact['raw_slots_long'].iloc[0].frequency_label == ''


@pytest.mark.parametrize('value,month,precision,status', [
    ('02/2020', '2020-02', 'month', 'valid'),
    ('2020-02', '2020-02', 'month', 'valid'),
    ('2020-02-29', '2020-02', 'day', 'valid'),
    ('2021-02-29', '', '', 'invalid_or_ambiguous'),
    ('13/2020', '', '', 'invalid_or_ambiguous'),
    ('2020', '', 'year', 'partial_year'),
    ('01/02/2020', '', '', 'invalid_or_ambiguous'),
    (None, '', '', 'missing'),
])
def test_months_never_impute_observed_days(value, month, precision, status):
    assert audit.parse_month(value) == dict(month=month, precision=precision, status=status)


def test_month_ambiguity_prevalent_open_invalid_and_repeated_observations():
    result = execute(frame(**{
        'medications__rx_1_name': ['HCQ'] * 4,
        'medications__med1_strt_date': ['01/2020', '02/2020', '05/2020', 'bogus'],
        'medications__med1_stop_date': ['01/2020', None, '04/2020', '2021'],
        'clinical_baseline_date': ['2020-03-15'] * 4,
    }), index_column='clinical_baseline_date')
    slots = result['raw_slots_long']
    assert 'same_month_order_unresolved' in slots.iloc[0].qc_flags
    assert 'pre_index_prevalent_candidate_valid' in slots.iloc[0].qc_flags
    assert 'stop_unavailable_open_or_undocumented' in slots.iloc[1].qc_flags
    assert 'start_after_stop' in slots.iloc[2].qc_flags
    assert 'start_parse_error' in slots.iloc[3].qc_flags
    assert slots.qc_flags.str.contains('repeated_drug_observation_not_new_start').all()
    assert len(result['episode_candidates']) == 4
    assert result['patient_coverage'].n_patients_duration_potential.eq(0).all()


def test_missing_schema_null_negative_history_and_allergy_are_distinct():
    result = execute(frame(**{
        'medications__rx_1_name': [None, 'none', 'No medications'],
        'past_medical_history__rx_1_name': ['rituximab', None, None],
        'past_medical_history__rx_allergies': ['methotrexate'] * 3,
    }), protocol='15-D-0051')
    assert result['episode_candidates'].empty
    assert result['raw_slots_long'].name_state.tolist() == ['explicit_negative'] * 2
    assert len(result['historical_inventory']) == 1
    assert result['historical_inventory'].iloc[0].exposure_status == 'historical_only'
    inventory = result['field_inventory'].query("protocol == '15-D-0051'").set_index('field')
    assert inventory.loc['medications__rx_1_name', 'n_missing_cells'] == 1
    assert inventory.loc['medications__rx_1_name', 'n_explicit_negative'] == 2
    assert not inventory.loc['medications__rx_2_name', 'column_present']
    summary = result['audit_summary']
    assert summary['n_eligible_patients_absent_from_sources'] == 1
    assert 'never_treated' not in json.dumps(summary)


def test_duplicate_concomitant_and_bounded_overlaps_without_collapsing_slots():
    result = execute(frame(**{
        'medications__rx_1_name': ['HCQ', 'MTX'],
        'medications__rx_2_name': ['Plaquenil', 'prednisone'],
        'medications__med1_strt_date': ['01/2020', '02/2020'],
        'medications__med1_stop_date': ['06/2020', '04/2020'],
    }))
    slots = result['raw_slots_long']
    assert len(slots) == 4
    assert 'duplicate_drug_same_source_row' in slots.iloc[0].qc_flags
    assert 'multiple_drugs_same_visit_concomitance_unconfirmed' in slots.iloc[2].qc_flags
    assert 'bounded_courses_overlap_candidate' in slots.iloc[0].qc_flags
    assert slots.record_id.is_unique


@pytest.mark.parametrize('value,expected', [
    ('Plaquenil 200 mg tablets', 'hydroxychloroquine'),
    ('Cellcept', 'mycophenolate mofetil'),
    ('rituximab', 'rituximab'), ('Benlysta', 'belimumab'),
    ('Leflunomide', 'leflunomide'), ('MTX', 'methotrexate'),
    ('HCQ + MTX', 'unknown/unmapped'), ('not hydroxychloroquine', 'unknown/unmapped'),
    ('mycophenolate sodium', 'unknown/unmapped'),
])
def test_conservative_name_mapping(value, expected):
    assert audit.normalize_name(value)['normalized_name'] == expected


@pytest.mark.parametrize('value,label,representation', [
    ('3', 'IV', 'code'), ('IV', 'IV', 'label'), ('3 IV', 'IV', 'code_and_label'),
    ('3 PO', '', 'unmapped'), ('77', '', 'unmapped'), (None, '', 'missing'),
])
def test_codes_labels_and_conflicting_labels(value, label, representation):
    assert audit.decode_category(value, audit.ROUTE_CODES) == (label, representation)


def test_protocol_gaps_not_inferred_and_source_row_survives_eligibility_filter():
    data = pd.DataFrame({audit.config.PATIENT_ID: ['outside', 'p1', 'p2'],
                         audit.config.VISIT_DATE: ['2020-01-01'] * 3,
                         'medications__rx_1_name': ['HCQ'] * 3})
    before = data.copy(deep=True)
    result = audit.audit_sources([source(data)], ['p1', 'p2'])
    assert result['raw_slots_long'].source_row.tolist() == [1, 2]
    assert result['raw_slots_long'].protocol.eq('unknown').all()
    assert result['audit_summary']['n_eligible_patients'] == 2
    assert result['patient_coverage'].n_patients_with_names.eq(2).all()
    pd.testing.assert_frame_equal(before, data)


def test_multiple_sources_deduplicate_patient_counts_but_keep_observations():
    data = frame(**{'medications__rx_1_name': ['HCQ']})
    result = audit.audit_sources([source(data, protocol='11-D-0172'),
                                 dict(source(data, protocol='15-D-0051'), name='second')], ['p1'])
    assert len(result['raw_slots_long']) == 2
    assert result['audit_summary']['n_eligible_patients_in_sources'] == 1
    assert result['audit_summary']['n_distinct_patients_with_names'] == 1
    assert set(result['patient_coverage'].protocol) == {'11-D-0172', '15-D-0051'}


def test_schema_key_validation_and_empty_cohort():
    with pytest.raises(ValueError, match='configured source keys'):
        execute(pd.DataFrame({'unrelated': [1]}))
    with pytest.raises(ValueError, match='empty'):
        audit.audit_sources([source(frame())], [])
    with pytest.raises(ValueError, match='not both'):
        execute(frame(p=['11-D-0172']), protocol_column='p', protocol='11-D-0172')


def test_empty_and_absent_medication_sources_are_supported():
    for data in [frame(), frame().iloc[:0]]:
        result = execute(data)
        assert result['raw_slots_long'].empty
        assert result['episode_candidates'].empty
        assert result['temporal_qc'].empty
        assert not result['field_inventory'].empty
        assert result['audit_summary']['gates']['record_description'] == 'blocked_no_names'


def test_cli_writes_only_new_audit_products_and_does_not_read_codebook(tmp_path, monkeypatch):
    data = frame(**{'medications__rx_1_name': ['HCQ'],
                    'medications__med1_strt_date': ['02/2020']})
    input_path = tmp_path / 'visits.csv'
    eligibility_path = tmp_path / 'eligible.csv'
    data.to_csv(input_path, index=False)
    pd.DataFrame({'patient_id': ['p1', 'p2']}).to_csv(eligibility_path, index=False)
    existing = tmp_path / 'previous_output.csv'
    existing.write_text('old product\n')
    hashes = {p: audit.file_hash(p) for p in (input_path, eligibility_path, existing)}
    monkeypatch.setattr(pd, 'read_excel', lambda *a, **kw: pytest.fail('Codebook access prohibited'))
    args = Namespace(input=input_path, eligibility=eligibility_path, eligibility_id_column='patient_id',
                     sources_json=None, output_dir=tmp_path / 'audit')
    result = audit.run(args)
    assert len(list(args.output_dir.iterdir())) == 11
    assert all(audit.file_hash(p) == digest for p, digest in hashes.items())
    assert result['audit_summary']['eligibility']['sha256'] == hashes[eligibility_path]
    report = (args.output_dir / '08_treatment_audit_report.md').read_text()
    assert 'Pending decision matrix' in report
    assert report.rstrip().endswith('never infer never-treated |')
    with pytest.raises(ValueError, match='new or empty'):
        audit.run(args)


def test_manifest_uses_explicit_existing_keys_and_relative_paths(tmp_path):
    data = pd.DataFrame({'patient_id': ['p1'], 'clinical_episode_id': ['e1'],
                         'medications__rx_1_name': ['rituximab']})
    data.to_parquet(tmp_path / 'episodes.parquet')
    pd.DataFrame({'patient_id': ['p1']}).to_csv(tmp_path / 'eligible.csv', index=False)
    manifest = tmp_path / 'sources.json'
    manifest.write_text(json.dumps([dict(name='episodes', path='episodes.parquet', patient_column='patient_id',
                                         visit_column='clinical_episode_id', protocol='11-D-0172')]))
    args = Namespace(input=None, sources_json=manifest, eligibility=tmp_path / 'eligible.csv',
                     eligibility_id_column='patient_id', output_dir=tmp_path / 'audit')
    result = audit.run(args)
    slot = result['raw_slots_long'].iloc[0]
    assert slot.visit_raw == 'e1' and slot.protocol == '11-D-0172'
    assert slot.source_sha256 == audit.file_hash(tmp_path / 'episodes.parquet')


def test_missing_real_data_is_explicitly_blocked_without_outputs(tmp_path):
    result = subprocess.run([sys.executable, str(SPEC.origin), '--eligibility', str(tmp_path / 'absent.csv'),
                             '--output-dir', str(tmp_path / 'out')], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Audit blocked:' in result.stderr
    assert not (tmp_path / 'out').exists()


def test_inventory_is_emitted_before_slot_transformation(monkeypatch):
    events = []
    original = audit.extract_slots

    def transform(*args):
        events.append('transform')
        assert events[0] == 'inventory'
        return original(*args)

    monkeypatch.setattr(audit, 'extract_slots', transform)
    audit.audit_sources([source(frame())], ['p1'], inventory_callback=lambda inv: events.append('inventory'))
    assert events == ['inventory', 'transform']


def test_absent_name_field_with_orphan_dates_is_not_a_candidate():
    result = execute(frame(**{'medications__med1_strt_date': ['03/2020']}))
    assert result['raw_slots_long'].iloc[0].name_state == 'column_unavailable'
    assert result['raw_slots_long'].iloc[0].stop_status == 'column_unavailable'
    assert result['episode_candidates'].empty


def test_raw_values_are_preserved_and_inventory_handles_duplicate_source_indexes():
    data = frame(**{'medications__rx_1_name': [' HCQ  ', ' none ']})
    data.index = [5, 5]
    result = execute(data)
    assert result['raw_slots_long'].name_raw.tolist() == [' HCQ  ', ' none ']
    assert result['raw_slots_long'].source_index_raw.tolist() == ['5', '5']
    inv = result['field_inventory'].query("protocol == 'unknown' and field == 'medications__rx_1_name'").iloc[0]
    assert inv.n_patients_observed == 1 and inv.n_explicit_negative == 1


def test_all_literal_fields_are_inventoried_before_use_with_parse_diagnostics():
    result = execute(frame(**{'medications__rx_1_name': ['HCQ'],
                             'medications__med1_strt_date': ['13/2020'],
                             'medications__rx_1_route': ['3 PO']}), protocol='11-D-0172')
    inventory = result['field_inventory'].query("protocol == '11-D-0172'").set_index('field')
    assert len(inventory) == 66 and inventory.index.is_unique
    assert inventory.loc['medications__med1_strt_date', 'n_parse_errors'] == 1
    assert inventory.loc['medications__rx_1_route', 'n_parse_errors'] == 1
    assert inventory.loc['medications__rx_1_name', 'n_unique_observed'] == 1
    assert inventory.loc['medications__rx_1_name', 'pct_missing_cells'] == 0
    assert not inventory.loc['medications__rx_10_name', 'column_present']
    assert result['feasibility'].temporal_gate.eq('blocked_no_bounded_courses').all()
    assert result['feasibility'].indication_gate.eq('blocked_no_indication_source').all()

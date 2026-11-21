"""Synthetic coverage for the optional, independently signed N3 day contract."""
from datetime import date
import json
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src.studies.graph._pipeline import digest
from src.studies.graph.stage03 import s03_n3_calendar_day as cal
from src.studies.graph.stage03 import s03_n3_controlled as E
from src.studies.graph.stage03.s03_n3_approved_view import measurement_values, measurement_values_calendar_day
from src.studies.graph.stage03.s03_n3_measurement_review import CALENDAR_AFFIRMATIONS
from tests.test_graph03_n3_controlled import attestations
from tests.test_graph03_n3_measurement_source import (
    INFERENCE, ROOT, RF, RF_TEXT, cli, manifest, originals, paths_before, review_rule, run, signed_review,
)


def inspection(kind='numeric', **updates):
    review = dict(
        measurement_contract=cal.LEGACY_CONTRACT, time_resolution='calendar_day', identity_contract=cal.IDENTITY,
        date_origin=cal.DATE_ORIGIN, selection_rule='latest_prebaseline_calendar_day', duplicate_rule='identical_only',
        calendar_day_origin_verified=True, baseline_calendar_day_verified=True, baseline_window_verified=True,
        selection_and_duplicates_reviewed=True, test_definition_verified=True, source_unit_and_scale_verified=True,
        unidentified_assay_and_specimen_risk_reviewed=True, missingness_and_indication_reviewed=True,
        lookback_days=30, minimum_days_before_baseline=1, type=kind,
        test_definition_id='SYNTHETIC_TEST_DEFINITION', unit='SYNTHETIC_SCALE',
    )
    if kind == 'binary':
        review['category_convention'] = dict(positive=['positive'], negative=['negative'],
                                             unknown=['unknown'], not_tested=['not performed'])
    review.update(updates)
    return review


def source_rule(kind='numeric', **updates):
    rule = dict.fromkeys(CALENDAR_AFFIRMATIONS, True)
    rule.update(
        unit_field='unit', date_field='lab_date',
        value_field='result_numeric_exact' if kind == 'numeric' else 'result_text',
        comparable_original_tests=[dict(original_order_name='SYNTHETIC_ORDER_NAME',
                                        original_observation_name='SYNTHETIC_OBSERVATION', assay=None)],
        measurement_review=inspection(kind, **updates),
    )
    return rule


@pytest.fixture
def calendar_originals(originals, tmp_path):
    paths, config_path, _, _ = originals
    eda = pd.read_parquet(paths['eda_long']).iloc[:100].drop(columns=['specimen_id'])
    eda = eda.assign(lab_date='2019-12-31', assay=None, order_name_original='SYNTHETIC_ORDER_NAME',
                     observation_name_original='SYNTHETIC_OBSERVATION')
    eda.loc[:9, 'matched_clinical_episode_id'] = 'PREVIOUS_' + eda.loc[:9, 'matched_clinical_episode_id']
    for role in ('step10', 'context'):
        table = pd.read_parquet(paths[role])
        other = table.iloc[:10].assign(clinical_episode_id=lambda f: 'PREVIOUS_' + f.clinical_episode_id)
        if role == 'step10':
            other = other.assign(is_clinical_baseline=False, clinical_anchor_date='2019-12-01')
        pd.concat([table, other]).to_parquet(paths[role], index=False)
    eda.to_parquet(paths['eda_long'], index=False)
    write_btris(paths, eda, tmp_path)
    cfg = yaml.safe_load(config_path.read_text())
    cfg.update(schema_version=cal.SOURCE_VERSION, eda_specimen_id_column=None)
    cfg['calendar_day'] = {
        'contract': cal.LEGACY_CONTRACT, 'windows': {RF: dict(lookback_days=30, minimum_days_before_baseline=1)},
        'linkage_fields': {
            field: dict(eda=eda_name, raw=btris_name, intermediate=btris_name)
            for field, eda_name, btris_name in zip(
                cal.KEY_FIELDS, ('patient_id', 'order_identifier', 'order_name_original', 'observation_name_original'),
                ('Patient ID', 'Order ID', 'Order Name', 'Observation Name'))
        },
        'date_formats': {'raw': '%m/%d/%Y %H:%M:%S'},
    }
    config_path.write_text(yaml.safe_dump(cfg))
    return originals


def write_btris(paths, eda, tmp_path, *, mismatched_orders=(), ambiguous_orders=()):
    btris = eda[['patient_id', 'order_identifier', 'order_name_original', 'observation_name_original',
                 'lab_date']].rename(columns=dict(patient_id='Patient ID', order_identifier='Order ID',
                                                 order_name_original='Order Name',
                                                 observation_name_original='Observation Name',
                                                 lab_date='Collected Date Time'))
    split = btris['Patient ID'].isin(btris['Patient ID'].drop_duplicates().iloc[:50])
    for protocol, partition in (('11d', btris.loc[split]), ('15d', btris.loc[~split])):
        for origin in cal.BTRIS_ORIGINS:
            role = f'btris_{origin}_{protocol}'
            folder = tmp_path / role
            folder.mkdir(exist_ok=True)
            file = folder / 'Lab-synthetic.csv'
            # 03f must inspect folder contents and separately bind every CSV.
            paths[role] = folder
            table = partition.copy()
            table['Original system field'] = 'stable'
            if origin == 'raw':
                extra = table.loc[table['Order ID'].isin(ambiguous_orders)].assign(
                    **{'Original system field': 'discordant source row'})
                table = pd.concat([table, extra])
                table['Collected Date Time'] = (pd.to_datetime(table['Collected Date Time'], errors='coerce')
                                                .dt.strftime('%m/%d/%Y 12:34:00'))
            else:
                table.loc[table['Order ID'].isin(mismatched_orders), 'Collected Date Time'] = '2019-12-30'
            table.to_csv(file, index=False)


def sign(originals, tmp_path, rules):
    _, audit = run(originals, 'audit-only', 'audit-for-signature')
    return signed_review(tmp_path, manifest(audit)['review_binding'], rules, version=cal.REVIEW_VERSION)


def test_calendar_candidate_and_reviewed_report_provenance_and_discordant_ties(calendar_originals, tmp_path):
    paths, _, _, _ = calendar_originals
    eda = pd.read_parquet(paths['eda_long'])
    baseline_ids = eda.patient_id.copy()
    eda.loc[10:19, 'lab_date'] = '2020-01-01'
    eda.loc[20:29, 'lab_date'] = '2020-01-02'
    eda.loc[30:39, 'lab_date'] = '2019-10-01'
    eda.loc[45:49, 'lab_date'] = '2019-02-30'
    eda.loc[80:84, 'episode_match_ambiguous'] = True
    eda.loc[85:89, 'unit'] = 'SYNTHETIC_INCOMPATIBLE_SCALE'
    eda.loc[90:94, 'result_operator'] = '<'
    eda.loc[95:99, 'observation_name_original'] = 'UNREVIEWED_TEST'
    discordant = eda.loc[50:64].copy()
    discordant['order_identifier'] += '_OTHER'
    discordant.loc[50:54, 'result_numeric_exact'] = 9
    discordant.loc[55:59, 'unit'] = 'OTHER_SCALE'
    discordant.loc[60:64, 'observation_name_original'] = 'OTHER_TEST'
    eda = pd.concat([eda, discordant, eda.loc[70:79]], ignore_index=True)
    eda.to_parquet(paths['eda_long'], index=False)
    write_btris(paths, eda, tmp_path, mismatched_orders=eda.loc[40:44, 'order_identifier'],
                ambiguous_orders=eda.loc[65:69, 'order_identifier'])
    # Step01 and Step10 audit can disagree without affecting original-row selection.
    context = pd.read_parquet(paths['context'])
    context['lab__rheumatoid_factor__measurement_date'] = '2020-01-03'
    context.to_parquet(paths['context'], index=False)
    before = paths_before(paths)

    candidate = cli(calendar_originals, 'build-candidate', 'candidate-v2')
    assert candidate.returncode == 0, candidate.stderr
    run_dir = calendar_originals[2] / 'runs/candidate-v2'
    observations = pd.read_parquet(run_dir / 'unapproved_observations.parquet')
    assert pq.read_schema(run_dir / 'unapproved_observations.parquet').field('collection_calendar_day').type == pa.date32()
    assert observations.loc[observations.patient_id.eq(baseline_ids[0]), 'source_episode_id'].iloc[0].startswith('PREVIOUS_')
    assert observations.test_definition_id.isna().all() and observations.measurement_date.isna().all()
    ledger = pd.read_parquet(run_dir / 'private_ledger.parquet')
    assert set(ledger.calendar_disposition.dropna()) >= {
        'AMBIGUOUS_SAME_DAY', 'SAME_DAY_EXCLUDED', 'POSTBASELINE_EXCLUDED', 'OUTSIDE_REVIEWED_LOOKBACK',
        'BTRIS_EDA_COLLECTION_DAY_DISCORDANT', 'EDA_COLLECTION_DAY_INVALID_OR_MISSING'}
    assert ledger.loc[80:84, 'identity_status'].eq('EDA_EPISODE_MATCH_AMBIGUOUS_OR_UNDOCUMENTED').all()
    metadata = manifest(run_dir)
    assert metadata['inputs']['btris_raw_11d']['file_sha256'] == {
        'Lab-synthetic.csv': digest(paths['btris_raw_11d'] / 'Lab-synthetic.csv')}
    eligibility = pd.read_csv(run_dir / 'calendar_day_eligibility.csv').iloc[0]
    assert eligibility[['day_available', 'eligible_prebaseline', 'same_day_excluded',
                        'postbaseline_excluded', 'ambiguity_excluded']].tolist() == [80, 35, 10, 10, 20]
    assert pd.isna(eligibility.test_definition_verified)

    documents = sign(calendar_originals, tmp_path, {RF: source_rule()})
    proc = cli(calendar_originals, 'build-reviewed', 'reviewed-v2',
               '--review', str(documents['review_path']), '--signature', str(documents['signature_path']),
               '--trust-key', str(documents['trust_key']))
    assert proc.returncode == 0, proc.stderr
    directory = calendar_originals[2] / 'runs/reviewed-v2'
    result = pd.read_parquet(directory / 'reviewed_instrument_measurements.parquet')
    assert result.columns.tolist() == cal.CORE
    assert result.patient_id.tolist() == baseline_ids.iloc[[*range(10), *range(70, 80)]].tolist()
    assert result[['assay_id', 'sample_id']].isna().all().all()
    assert set(result.collection_calendar_day) == {date(2019, 12, 31)}
    assert result.source_review_sha256.eq(digest(documents['review_path'])).all()
    assert result.source_episode_id.str.startswith('PREVIOUS_').sum() == 10
    reviewed = pd.read_parquet(directory / 'private_reviewed_ledger.parquet')
    assert reviewed.reviewed_disposition.value_counts().loc['IDENTICAL_DUPLICATE_COLLAPSED'] == 10
    assert reviewed.reviewed_disposition.value_counts().loc['TEST_DEFINITION_OR_UNIT_OUTSIDE_REVIEWED_SCOPE'] == 10
    assert reviewed.reviewed_disposition.value_counts().loc['UNUSABLE_OR_UNREVIEWED_RESULT'] == 5
    assert pd.read_csv(directory / 'calendar_day_eligibility.csv').iloc[0].eligible_prebaseline == 20
    assert manifest(directory)['outcomes'][RF]['status'] == 'SOURCE_REVIEWED'
    assert paths_before(paths) == before
    assert not any(baseline_ids.iloc[0] in path.read_text() or '2019-12-31' in path.read_text()
                   for path in directory.iterdir() if path.suffix in {'.json', '.csv'})


def test_calendar_strict_days_select_latest_and_keep_unknown_missing():
    baseline = pd.DataFrame(dict(patient_id=list('ABCDEF'), clinical_episode_id=list('abcdef'),
                                 clinical_baseline_date='2021-03-15', clinical_anchor_date='2021-03-15',
                                 is_clinical_baseline=True))
    reviewed = inspection('binary')
    sha = 'a' * 64
    rows = baseline[cal.KEYS].assign(
        source_episode_id='PREVIOUS', source_column=RF_TEXT,
        collection_calendar_day=['2021-03-14', '2021-03-15', '2021-03-16', '2021-03-14',
                                 '2021-03-14', '2021-02-12'],
        value=['positive', 'positive', 'positive', 'unknown', 'not performed', 'positive'],
        test_definition_id=reviewed['test_definition_id'], unit=reviewed['unit'],
        assay_id=None, sample_id=None, original_order_id='ORDER', original_order_name='NAME',
        original_observation_name='OBSERVATION', date_origin=cal.DATE_ORIGIN, date_resolution='calendar_day',
        review_contract_version=cal.REVIEW_VERSION, source_review_sha256=sha,
        measurement_contract_sha256=cal.fingerprint(reviewed), observation_sha256='b' * 64,
        source_artifact_status='SOURCE_REVIEWED')
    older = rows.iloc[:1].assign(collection_calendar_day='2021-03-13', value='negative')
    rows = pd.concat([older, rows, rows.iloc[:1]], ignore_index=True)
    values, kind = measurement_values_calendar_day(baseline, rows, RF_TEXT, reviewed, source_review_sha256=sha)
    assert kind == 'binary' and values.iloc[0] == 1 and values.iloc[1:].isna().all()
    sensitivity = dict(reviewed, minimum_days_before_baseline=2)
    restricted = rows.assign(measurement_contract_sha256=cal.fingerprint(sensitivity))
    values, _ = measurement_values_calendar_day(baseline, restricted, RF_TEXT, sensitivity, source_review_sha256=sha)
    assert values.iloc[0] == 0
    contradictory = dict(reviewed, category_convention=dict(reviewed['category_convention'], unknown=[],
                                                            negative=['unknown']))
    with pytest.raises(ValueError, match='ADMINISTRATIVE_TOKEN'):
        measurement_values_calendar_day(baseline, rows, RF_TEXT, contradictory, source_review_sha256=sha)
    with pytest.raises(ValueError, match='ARTIFACT_OUTSIDE_SIGNED_SCOPE'):
        measurement_values_calendar_day(baseline, rows.assign(unit='OTHER'), RF_TEXT, reviewed, source_review_sha256=sha)
    with pytest.raises(ValueError, match='UNAPPROVED_MEASUREMENT_CANDIDATE'):
        measurement_values_calendar_day(baseline, rows.assign(source_artifact_status='SOURCE_CANDIDATE_UNAPPROVED'),
                                        RF_TEXT, reviewed, source_review_sha256=sha)
    with pytest.raises(ValueError, match='SAMPLE_IDENTITY_MISSING'):
        measurement_values(baseline, rows.assign(measurement_date='2021-03-14'), RF_TEXT,
                           review_rule(RF_TEXT)['measurement_review'])


def test_calendar_audit_withholds_small_cells_and_blocks_missing_btris(calendar_originals, tmp_path):
    paths, _, _, _ = calendar_originals
    eda = pd.read_parquet(paths['eda_long'])
    eda.loc[0, 'lab_date'] = '2020-01-01'
    eda.to_parquet(paths['eda_long'], index=False)
    write_btris(paths, eda, tmp_path)
    _, audit = run(calendar_originals, 'audit-only', 'small')
    row = pd.read_csv(audit / 'calendar_day_eligibility.csv').iloc[0]
    assert row.disclosure_status == 'WITHHELD_SMALL_CELLS' and row.iloc[2:].isna().all()
    assert not list(audit.glob('*.parquet'))
    paths.pop('btris_raw_15d')
    result, blocked = run(calendar_originals, 'audit-only', 'unavailable')
    assert result['status'] == 'SOURCE_CONTRACT_NOT_READY'
    assert 'BTRIS_COLLECTION_FILES_NOT_AVAILABLE' in manifest(blocked)['blockers']


@pytest.mark.parametrize('case, reason', [
    ('invalid_signature', 'SIGNATURE_INVALID'),
    ('window_changed', 'PRESPECIFIED_SCOPE'),
    ('missing_crosswalk', 'CROSSWALK_REQUIRED'),
    ('changed_csv', 'INPUT_PINS_INVALID'),
])
def test_calendar_source_review_fails_closed(calendar_originals, tmp_path, case, reason):
    rule = source_rule(lookback_days=12) if case == 'window_changed' else source_rule()
    if case == 'missing_crosswalk':
        rule.pop('comparable_original_tests')
    documents = sign(calendar_originals, tmp_path, {RF: rule})
    if case == 'invalid_signature':
        documents['signature_path'].write_bytes(b'not a valid synthetic signature')
    if case == 'changed_csv':
        file = calendar_originals[0]['btris_raw_11d'] / 'Lab-synthetic.csv'
        file.write_text(file.read_text() + '\n')
    with pytest.raises(ValueError, match=reason):
        run(calendar_originals, 'build-reviewed', 'blocked', **documents)
    assert not (calendar_originals[2] / 'runs/blocked').exists()


def test_signed_legacy_calendar_day_artifact_cannot_enable_current_inference(calendar_originals, tmp_path):
    paths, _, _, decision_root = calendar_originals
    documents = sign(calendar_originals, tmp_path, {RF: source_rule()})
    _, directory = run(calendar_originals, 'build-reviewed', 'reviewed-v2', **documents)
    measurements = directory / 'reviewed_instrument_measurements.parquet'
    plan_path = tmp_path / 'calendar-plan.yaml'
    base = [sys.executable, str(INFERENCE), '--graph03-run', str(paths['graph03_run']),
            '--graph02-release', str(paths['graph02_release']), '--context', str(paths['context']),
            '--measurements', str(measurements), '--plan', str(plan_path), '--decision-root', str(decision_root)]
    draft = subprocess.run([*base, '--prepare-plan', '--output-dir', str(tmp_path / 'templates')],
                           capture_output=True, text=True)
    assert draft.returncode == 0, draft.stderr
    plan = yaml.safe_load(plan_path.read_text())
    spec = dict(representation_id='R0', source_column=RF, requested_models=['A', 'B'], n1_basis=[], use_pop=False,
                topology_scope='full', model_family='gaussian_ols', hypothesis_family='SYNTHETIC')
    plan.update(outcomes=[spec], authorized_reviewers=['SYNTHETIC_ONLY_REVIEWER'])
    attestations(decision_root, plan, spec)
    pointer = next(p for p in plan['decisions'] if p['section'] == 'measurement')
    payload_path = decision_root / pointer['path']
    payload = yaml.safe_load(payload_path.read_text())
    payload['details'] = inspection(source_review_sha256=digest(documents['review_path']))
    payload_path.write_text(yaml.safe_dump(payload))
    pointer.update(sha256=digest(payload_path), signature_path='measurement.sig')
    plan['decisions'] = [pointer]  # Source review must never imply approval of the other gates.
    plan_path.write_text(yaml.safe_dump(plan))
    private_key = tmp_path / 'key.pem'
    signature = decision_root / pointer['signature_path']
    signature.write_bytes(b'invalid')
    key = documents['trust_key']

    def execute(name, *args):
        proc = subprocess.run([*base, '--execute', '--output-dir', str(tmp_path / name), *args],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout)['status'] == 'COMPLETE_WITH_BLOCKERS'
        frame = pd.read_csv(tmp_path / name / 'preflight.csv')
        return frame.loc[frame.model.eq('A')].iloc[0]

    assert execute('unsigned').temporal_gate == E.FAIL
    assert execute('invalid-signature', '--measurement-trust-key', str(key)).temporal_gate == E.FAIL
    subprocess.run(['openssl', 'dgst', '-sha256', '-sign', str(private_key), '-out', str(signature),
                    str(payload_path)], check=True, capture_output=True)
    row = execute('verified-signature', '--measurement-trust-key', str(key))
    assert row.temporal_gate == row.measurement_gate == E.FAIL
    assert 'BASELINE_WINDOW_POLICY_VERSION_MISMATCH' in row.blocking_reasons
    assert row.protocol_gate == row.independence_gate == E.FAIL
    assert row.status == 'NOT_ESTIMABLE'

"""Exercise the zero-argument exploratory route on synthetic frozen releases."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.studies.graph._pipeline import digest
from src.studies.graph.stage03 import s03_n3_exploratory as E
from src.studies.graph.stage03 import s03_n3_exploratory_models as M
from src.studies.graph.stage03 import s03_n3_exploratory_reporting as R
from src.studies.graph.stage03 import s03_n3_exploratory_source as S
from tests.test_graph03_n3_auto import automatic
from tests.test_graph03_n3_calendar_day import calendar_originals, write_btris
from tests.test_graph03_n3_measurement_source import ANA, HEP2, RF, RF_TEXT, ROOT, originals, paths_before

POLICY = E._policy(ROOT)
CELL = POLICY['minimum_disclosure_cell']
NO_MODELS = 'COMPLETE_DESCRIPTIVE_NO_ESTIMABLE_MODELS'
URINE = 'lab__urine_protein_qualitative__text'


def command(repo):
    script = repo / 'src/studies/graph/03e_graph_n3_controlled_inference.py'
    run = subprocess.run([sys.executable, str(script)], cwd=repo.parent, check=True,
                         capture_output=True, text=True)
    assert not any(secret in run.stdout + run.stderr for secret in ('SYNTHETIC_EP_', 'SYNTHETIC_ORDER_', '2019-12-31'))
    return json.loads(run.stdout)


def inspect_release(result, private):
    output = Path(result['output_dir'])
    assert output.parent == private / 'exploratory' / 'runs'
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / 'figures').stat().st_mode & 0o777 == 0o700
    assert all(f.stat().st_mode & 0o777 == 0o600 for f in output.rglob('*') if f.is_file())
    manifest = json.loads((output / 'results_manifest.json').read_text())
    for name, sha in manifest['sha256_by_artifact'].items():
        assert digest(output / name) == sha
    assert set(manifest['sha256_by_artifact']) == {
        path.relative_to(output).as_posix() for path in output.rglob('*') if path.is_file()
    } - {'receipt.json', 'results_manifest.json'}
    latest = private / 'exploratory' / 'latest.json'
    if result['status'] != 'INPUTS_UNUSABLE':
        assert latest.stat().st_mode & 0o777 == 0o600
        assert json.loads(latest.read_text()) == {
            'run_id': output.name, 'relative_path': 'runs/' + output.name,
            'results_manifest_sha256': digest(output / 'results_manifest.json')}
    for path in output.glob('*.csv'):
        contents = path.read_text()
        assert not any(secret in contents for secret in (
            'SYNTHETIC_EP_', 'SYNTHETIC_ORDER_', 'SYNTHETIC_OBSERVATION', '2019-12-31', 'SYNTHETIC_0'))
    return output


def rep_outcome(frame, rep, outcome):
    return frame.loc[frame.representation_id.eq(rep) & frame.outcome.eq(outcome)]


def test_zero_arguments_builds_private_aggregates_and_never_uses_the_guard(automatic):
    repo, paths, private, _ = automatic
    eda = pd.read_parquet(paths['eda_long']).assign(unit='IU/mL')
    eda.to_parquet(paths['eda_long'], index=False)
    auto_module = repo / 'src/studies/graph/stage03/s03_n3_auto.py'
    with auto_module.open('a') as handle:
        handle.write("\ndef run_auto(*args):\n    raise AssertionError('Signed route invoked')\n")
    (private / 'institution').mkdir(parents=True)
    (private / 'institution' / 'prespecification.yaml').symlink_to('unreadable-sentinel')
    before = paths_before(paths), digest(paths['step10'])
    result = command(repo)
    assert result['status'] == NO_MODELS, result
    assert result['mode'] == E.MODE
    assert result['n_outcomes_with_data'] == 2
    assert result['n_model_pairs_estimable'] == 0
    assert not result['data_issues']
    assert command(repo) == result
    assert (paths_before(paths), digest(paths['step10'])) == before
    output = inspect_release(result, private)
    profiles = pd.read_csv(output / 'community_profiles.csv')
    rf = profiles.loc[(profiles.outcome == RF) & profiles.n_measured.gt(0)]
    assert len(rf) == 4 and rf['median'].notna().all()
    assert rf.loc[rf.representation_id.eq('R2'), 'topology_status'].eq('R2_EXPLORATORY_UNSTABLE').all()
    assert rf.provenance.eq(S.ORIGINAL).all()
    evidence = pd.read_csv(output / 'source_linkage_audit.csv')
    assert evidence.loc[evidence.dimension.eq('date_evidence') & evidence.outcome.eq(RF), 'category'].dropna().eq('RAW_AND_EDA_AGREE').all()
    models = pd.read_csv(output / 'models.csv')
    assert rep_outcome(models, 'R0', RF).status.item() == M.UNRESOLVED
    assert models.p_incremental.isna().all()
    assert 'Resultados exploratorios de Graph03 N3' in Path(result['report']).read_text()

    with auto_module.open('a') as handle:
        handle.write('\n# Synthetic implementation change.\n')
    updated = command(repo)
    assert updated['output_dir'] != result['output_dir']
    assert inspect_release(updated, private).is_dir()
    eda.to_parquet(paths['eda_long'], index=False, compression='gzip')
    assert command(repo)['output_dir'] != updated['output_dir']


@pytest.mark.parametrize('tamper', ['graph03', 'baseline', 'membership'])
def test_missing_or_tampered_inputs_preserve_only_verifiable_results(automatic, tamper):
    repo, paths, private, _ = automatic
    paths['eda_long'].unlink()
    (paths['graph03_run'] / '03_graph_n3_results_summary.md').write_text('tampered')
    if tamper == 'baseline':
        paths['baseline'].write_text('tampered')
    if tamper == 'membership':
        release = json.loads(paths['graph02_release'].read_text())
        (paths['graph02_release'].parent / release['relative_paths']['membership']).write_text('tampered')
    result = command(repo)
    assert result['status'] == ('INPUTS_UNUSABLE' if tamper == 'baseline' else 'PARTIAL_WITH_DATA_ISSUES'), result
    assert result['n_outcomes_with_data'] == 0
    assert {'eda_long', 'compatible_graph02_graph03_releases'} <= {item['artifact'] for item in result['data_issues']}
    assert pd.read_csv(result['models']).p_incremental.isna().all()
    output = inspect_release(result, private)
    baseline = pd.read_csv(output / 'frozen_baseline_profiles.csv')
    assert baseline.empty == (tamper == 'baseline')
    if tamper == 'membership':
        assert 'ONLY_BASELINE_AND_RELEASE_QC_VERIFIED' in {issue.get('reason') for issue in result['data_issues']}
        assert not baseline.n_hard.notna().any()
    if tamper != 'baseline':
        assert baseline.n_total.max() == 100
        assert 'baseline_congelado.png' in {Path(path).name for path in result['figures']}


def test_date_window_source_identity_duplicates_and_definitions(calendar_originals):
    paths, _, _, _ = calendar_originals
    eda = pd.read_parquet(paths['eda_long']).iloc[:40].copy().assign(unit='IU/mL')
    eda.loc[:6, 'lab_date'] = ['2018-12-31', '2019-01-01', '2019-12-31', '2020-01-01',
                              '2020-01-02', '2020-12-31', '2021-01-01']
    eda['specimen_datetime'] = eda.lab_date
    eda['specimen_date_origin'] = 'Collected Date Time'
    eda.loc[1, 'matched_clinical_episode_id'] = None
    eda.loc[7, 'matched_clinical_episode_id'] = eda.loc[8, 'matched_clinical_episode_id']
    eda.loc[9, 'specimen_datetime'] = '2019-12-29'
    eda.loc[12, 'unit'] = 'mg/L'
    eda.loc[13, 'assay'] = 'alternate assay'
    eda.loc[14, 'result_operator'] = '<'
    eda.loc[15, 'result_text'] = 'trace'
    eda.loc[16, 'result_text'] = 'borderline'
    eda.loc[17, 'result_text'] = 'Not_Detected'
    eda.loc[18, 'specimen_date_origin'] = None
    latest_tie = eda.iloc[[10]].assign(result_numeric_exact=400, result_raw='400')
    older = eda.iloc[[10]].assign(lab_date='2019-12-28', specimen_datetime='2019-12-28')
    collision = eda.iloc[[19]].assign(patient_id='SYNTHETIC_-19')
    mixed_unit = eda.iloc[[20]].assign(unit='mg/L')
    invalid_later = eda.iloc[[21]].assign(lab_date='2020-01-01', specimen_datetime='2020-01-01',
                                        result_operator='<')
    eda = pd.concat([eda, latest_tie, older, eda.iloc[[11]], collision, mixed_unit, invalid_later],
                    ignore_index=True)
    tables = {role: pd.read_parquet(paths[role]) for role in ('step01', 'step10', 'context')}
    tables['eda_long'] = eda
    analytic = tables['step10']
    analytic.loc[analytic.is_clinical_baseline, 'clinical_baseline_date'] = '2020-01-01T21:15:00-05:00'
    tables['step10'] = pd.concat([analytic, analytic.loc[analytic.patient_id.eq('SYNTHETIC_22')]], ignore_index=True)
    baseline = pd.read_csv(paths['baseline'], dtype={'patient_id': 'string'})
    rows = S._candidates(tables, baseline, S._btris_records({}, []), POLICY)
    assert rows.loc[:6, 'stratum'].tolist() == [S.OLDER, S.PRE, S.PRE, S.DAY0, S.EARLY, S.EARLY, S.LATE]
    assert rows.loc[1, 'identity_status'] == S.PATIENT_ONLY
    assert rows.loc[1, 'source_episode_unknown'] and pd.isna(rows.loc[1, 'source_episode_id'])
    assert rows.loc[7, 'identity_status'] == 'CROSS_PATIENT_CONFLICT'
    assert rows.loc[9, 'date_evidence'] == 'DATE_DISCORDANT'
    assert rows.loc[18, 'date_evidence'] == S.EDA_ONLY
    assert rows.loc[19, 'identity_status'] == 'PATIENT_ID_AMBIGUOUS'
    assert rows.loc[22, 'identity_status'] == 'AMBIGUOUS_IDENTITY'
    rf = S._outcome(rows, RF, POLICY)
    text = S._outcome(rows, RF_TEXT, POLICY)
    primary = rf.primary[(S.PRE, S.ORIGINAL)]
    assert rf.selected.loc[rf.selected.definition_id.eq(primary), 'unit'].eq('IU/mL').all()
    assert set(rf.selected.loc[rf.selected.definition_id.ne(primary), 'unit']) == {'mg/L', 'IU/mL'}
    assert rf.duplicates.patient_id.tolist() == ['SYNTHETIC_11']
    assert {10, 20} <= set(rf.inventory.index[rf.inventory.first_reason.eq('AMBIGUOUS_LATEST_DAY')])
    assert not rf.selected.patient_id.isin({'SYNTHETIC_10', 'SYNTHETIC_20'}).any()
    assert text.inventory.loc[[15, 16], 'first_reason'].eq('UNKNOWN_RESULT_TOKEN').all()
    assert text.selected.loc[17, 'value'] == 0
    assert rf.inventory.loc[14, 'first_reason'] == 'CENSORED_RESULT'
    assert rf.selected.loc[21, 'stratum'] == S.PRE


def test_raw_observation_join_does_not_equate_cluster_and_preserves_discordance(calendar_originals):
    paths, _, _, _ = calendar_originals
    eda = pd.read_parquet(paths['eda_long']).iloc[:5].copy()
    eda.loc[0, 'cluster_name_original'] = eda.loc[0, 'observation_name_original']
    eda.loc[0, 'observation_name_original'] = None
    eda.loc[1, 'lab_date'] = '2019-12-30'
    tables = {role: pd.read_parquet(paths[role]) for role in ('step01', 'step10', 'context')}
    tables['eda_long'] = eda
    baseline = pd.read_csv(paths['baseline'], dtype={'patient_id': 'string'})
    records = S._btris_records({role: S.cal.btris_files(paths[role]) for role in S.cal.BTRIS_ROLES}, [])
    rows = S._candidates(tables, baseline, records, POLICY)
    assert rows.loc[0, 'date_evidence'] == S.EDA_ONLY
    assert rows.loc[0, 'btris_discriminator_unmatched']
    assert rows.loc[1, 'date_evidence'] == 'DATE_DISCORDANT'
    assert rows.loc[2, 'date_evidence'] == 'RAW_AND_EDA_AGREE'


def test_complementary_suppression_is_shared_by_profiles_coverage_and_flow():
    label = {'representation_id': 'R0', 'outcome': RF_TEXT, 'topology_status': 'FROZEN_COMMUNITIES'}
    frame = pd.DataFrame(dict(patient_id=[f'SYNTHETIC_{i}' for i in range(30)],
                             community_id=['0'] * 13 + ['1'] * 14 + ['2'] * 3,
                             value=[1.] * 2 + [0.] * 11 + [1.] * 12 + [np.nan] * 2 + [0.] * 3))
    profiles = R._profiles(frame, label, False, CELL)
    assert all('n_measured' not in row for row in profiles)
    assert any('community_id' not in row for row in profiles)
    assert 'n_total' not in R._coverage(frame, label, profiles, CELL)
    frame = frame.iloc[:13]
    profile = R._profiles(frame, label, False, CELL)[0]
    assert profile['n_measured'] == 13 and 'n_positive' not in profile and 'n_negative' not in profile
    repeated = pd.DataFrame(dict(_patient_match=['one'] * 30, first_reason='CENSORED'))
    assert R._exclusions(repeated, label, CELL)[0]['status'] == 'SUPPRESSED'
    chain = R._chain(label, dict(first=30, second=28, third=0), CELL)
    assert chain[0]['n_remaining'] is None and chain[1]['n_remaining'] is None
    outputs = {name: [] for name in R.TABLE_COLUMNS}
    outputs['cohort_flow'] = R._chain(label, dict(selected=28), CELL)
    outputs['source_linkage_audit'] = [
        dict(source_role='btris_raw_11d', dimension='BTRIS_SCHEMA'),
        dict(outcome=RF_TEXT, dimension='identity_status', n_total=28, n_count=28),
    ]
    E._mask_cohort_complements(outputs, 30, CELL)
    assert outputs['cohort_flow'][0]['n_remaining'] is None
    assert outputs['source_linkage_audit'][0]['dimension'] == 'BTRIS_SCHEMA'
    assert outputs['source_linkage_audit'][1]['status'] == 'SUPPRESSED'


def test_continuous_binary_support_rank_and_separation():
    rng = np.random.default_rng(2026)
    n = 100
    frame = pd.DataFrame({'value': rng.normal(size=n), 'community_id': ['0'] * 50 + ['1'] * 50,
                          'demo__age_at_baseline': rng.uniform(20, 70, n), 'demo__sex': ['F', 'M'] * 50,
                          'clinical_baseline_date': ['2020-01-01', '2015-01-01', '2010-01-01'] * 33 + ['2020-01-01']})
    info, prepared = M._model_support(frame, True, POLICY['models'], CELL)
    assert info['n'] == n
    assert M._estimate(prepared, True, POLICY['models'])['status'] == M.ESTIMATED
    frame['value'] = rng.integers(0, 2, n)
    info, prepared = M._model_support(frame, False, POLICY['models'], CELL)
    assert M._estimate(prepared, False, POLICY['models'])['status'] == M.ESTIMATED
    info, prepared = M._model_support(frame.iloc[:19], False, POLICY['models'], CELL)
    assert prepared is None and 'INSUFFICIENT' in info['reason']
    separated = frame.assign(**{'demo__age_at_baseline': frame.value + 20})
    info, prepared = M._model_support(separated, False, POLICY['models'], CELL)
    assert prepared is None and 'SEPARATION' in info['reason']
    rank_deficient = frame.assign(**{'demo__age_at_baseline': frame.community_id.astype(float) + 20})
    info, prepared = M._model_support(rank_deficient, False, POLICY['models'], CELL)
    assert prepared is None and 'RANK_DEFICIENT_DESIGN' in info['reason']


def bind_lineage(paths):
    release = json.loads(paths['graph02_release'].read_text())
    registry, features = [release['relative_paths'][key] for key in ('clinical_registry', 'feature_sets')]
    directory = paths['graph02_release'].parent
    columns = [RF, RF_TEXT, ANA, HEP2, 'pro__sf36_pcs']
    pd.DataFrame({'source_column': columns, 'semantic_feature_id': [
        RF.rsplit('__', 1)[0], RF.rsplit('__', 1)[0], ANA.rsplit('__', 1)[0],
        HEP2.rsplit('__', 1)[0], 'pro__sf36_pcs']}).to_csv(directory / registry, index=False)
    sets = pd.read_csv(directory / features)
    sets.loc[sets.representation_id.eq('R2'), 'source_column'] = RF_TEXT
    sets['feature'] = sets.source_column + '__encoded'
    sets.to_csv(directory / features, index=False)
    for filename in (registry, features):
        release['sha256_by_artifact'][filename] = digest(directory / filename)
    release['features_sha256'] = digest(directory / features)
    release['source_inputs']['integrated'] = {'sha256': digest(paths['step10'])}
    paths['graph02_release'].write_text(json.dumps(release))
    graph03 = paths['graph03_run']
    for path in graph03.glob('*.json'):
        document = json.loads(path.read_text())
        document['graph02_release_sha256'] = digest(paths['graph02_release'])
        path.write_text(json.dumps(document))
    summary = graph03 / '03_graph_results_summary.json'
    content = json.loads(summary.read_text())
    content['sha256_by_artifact'] = {path.name: digest(path) for path in graph03.iterdir() if path != summary}
    summary.write_text(json.dumps(content))


def test_end_to_end_ols_logit_rf_lineage_and_distinct_temporal_sensitivity(automatic):
    repo, paths, private, _ = automatic
    eda = pd.read_parquet(paths['eda_long']).assign(unit='IU/mL')
    eda['result_text'] = np.where((eda.index % 3) == 0, 'positive', 'negative')
    ana = eda.assign(canonical_analyte=E.LAB_SOURCES[ANA], unit=None,
                     order_identifier=eda.order_identifier + '_ANA', order_name_original='SYNTHETIC_ANA')
    eda.loc[:9, 'matched_clinical_episode_id'] = None
    original = pd.concat([eda, ana], ignore_index=True)
    write_btris(paths, original, repo.parent)
    lower_date_provenance = ana.assign(lab_date='2019-12-30', order_identifier=ana.order_identifier + '_EDA')
    lower_rf = eda.assign(lab_date='2019-12-30', order_identifier=eda.order_identifier + '_EDA',
                          result_numeric_exact=2., result_raw='2')
    pd.concat([original, lower_date_provenance, lower_rf], ignore_index=True).to_parquet(paths['eda_long'], index=False)
    bind_lineage(paths)
    before = paths_before(paths), digest(paths['step10'])
    result = command(repo)
    assert result['status'] == 'COMPLETE_WITH_EXPLORATORY_MODELS', result
    assert result['n_model_pairs_estimable'] >= 2
    output = inspect_release(result, private)
    rows = pd.read_csv(result['models'])
    assert rep_outcome(rows, 'R0', RF).family.item() == 'gaussian_ols_hc3'
    assert rep_outcome(rows, 'R0', RF).status.item() == M.ESTIMATED
    assert rep_outcome(rows, 'R0', RF_TEXT).status.item() == 'ALTERNATIVE_RF_ENCODING_DESCRIPTIVE_ONLY'
    assert rep_outcome(rows, 'R2', RF).status.item() == M.PROXY
    assert rep_outcome(rows, 'R2', RF_TEXT).status.item() == M.DIRECT
    assert rep_outcome(rows, 'R0', ANA).family.item() == 'binomial_logit_hc3'
    assert rep_outcome(rows, 'R0', ANA).status.item() == M.ESTIMATED
    assert rep_outcome(rows, 'R0', URINE).status.item() == M.UNRESOLVED
    dates = pd.read_csv(output / 'sensitivity_models.csv')
    assert rep_outcome(dates, 'R0', ANA).status.item() == M.ESTIMATED
    assert dates.provenance.eq(S.EDA_ONLY).all()
    assert rep_outcome(dates, 'R0', RF).not_estimable_reason.str.contains('DEGENERATE_OUTCOME').item()
    assert rep_outcome(dates, 'R0', RF_TEXT).status.item() == M.ESTIMATED
    links = pd.read_csv(output / 'linkage_sensitivity_models.csv')
    assert rep_outcome(links, 'R0', RF).status.item() == M.ESTIMATED
    tested = pd.concat([rows, dates, links]).loc[lambda frame: frame.status.eq(M.ESTIMATED)]
    assert tested.fdr_test_count.eq(len(tested)).all()
    assert tested.q_bh.between(0, 1).all()
    assert (paths_before(paths), digest(paths['step10'])) == before
    profiles = pd.read_csv(output / 'community_profiles.csv')
    assert rep_outcome(profiles, 'R0', RF_TEXT).n_measured.gt(0).any()


@pytest.mark.parametrize('empty', [False, True])
def test_no_prebaseline_publishes_real_frozen_descriptives_and_followup(automatic, empty):
    repo, paths, private, _ = automatic
    eda = pd.read_parquet(paths['eda_long']).assign(unit='IU/mL')
    if empty:
        eda = eda.iloc[:0]
    else:
        eda['lab_date'] = np.select([eda.index < 25, eda.index < 75], ['2020-01-01', '2020-01-02'],
                                    default='2021-01-01')
        write_btris(paths, eda, repo.parent)
    eda.to_parquet(paths['eda_long'], index=False)
    result = command(repo)
    assert result['status'] == NO_MODELS, result
    output = inspect_release(result, private)
    frozen = pd.read_csv(output / 'frozen_baseline_profiles.csv')
    assert frozen.n_total.max() == 100 and frozen['median'].notna().any()
    assert (output / 'figures' / 'baseline_congelado.png').stat().st_size > 1000
    assert not pd.read_csv(output / 'cohort_flow.csv').empty
    coverage = rep_outcome(pd.read_csv(output / 'coverage.csv'), 'R0', RF)
    assert coverage.loc[coverage.stratum.eq(S.PRE), 'n_measured'].eq(0).all()
    followup = pd.read_csv(output / 'followup_profiles.csv')
    assert followup.n_measured.gt(0).any() != empty

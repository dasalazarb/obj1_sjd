"""Exercise the zero-argument exploratory route with synthetic releases only."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from src.studies.graph._pipeline import digest
from src.studies.graph.stage03 import s03_n3_exploratory as E
from tests.test_graph03_n3_auto import automatic
from tests.test_graph03_n3_calendar_day import calendar_originals
from tests.test_graph03_n3_measurement_source import ANA, HEP2, RF, RF_TEXT, originals, paths_before


def command(repo):
    script = repo / 'src/studies/graph/03e_graph_n3_controlled_inference.py'
    run = subprocess.run([sys.executable, str(script)], cwd=repo.parent, check=True,
                         capture_output=True, text=True)
    return json.loads(run.stdout)


def test_zero_arguments_builds_private_aggregates_and_never_uses_the_guard(automatic):
    repo, paths, private, _ = automatic
    eda = pd.read_parquet(paths['eda_long'])
    eda['unit'] = 'IU/mL'
    eda.to_parquet(paths['eda_long'], index=False)
    auto_module = repo / 'src/studies/graph/stage03/s03_n3_auto.py'
    with auto_module.open('a') as handle:
        handle.write("\ndef run_auto(*args):\n    raise AssertionError('Signed route invoked')\n")
    (private / 'institution').mkdir(parents=True)
    (private / 'institution' / 'prespecification.yaml').symlink_to('unreadable-sentinel')
    before = paths_before(paths)
    result = command(repo)
    assert result['status'] == 'EXPLORATORY_PARTIAL', result
    assert result['mode'] == 'calendar_day_exploratory'
    assert result['n_outcomes_with_data'] == 2
    assert result['n_model_pairs_estimable'] == 0
    assert not result['data_issues']
    assert command(repo) == result
    assert paths_before(paths) == before
    output = Path(result['output_dir'])
    assert output.parent == private / 'exploratory' / 'runs'
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / 'figures').stat().st_mode & 0o777 == 0o700
    assert all(f.stat().st_mode & 0o777 == 0o600 for f in output.rglob('*') if f.is_file())
    for name in ('summary.json', 'coverage.csv', 'community_profiles.csv', 'models.csv', 'exclusions.csv', 'README.md'):
        contents = (output / name).read_text()
        assert not any(s in contents for s in ('SYNTHETIC_EP_', 'SYNTHETIC_ORDER_', '2019-12-31', 'PENDING_PRESPECIFICATION'))
    profiles = pd.read_csv(output / 'community_profiles.csv')
    rf = profiles.loc[(profiles.outcome == RF) & profiles.n_measured.gt(0)]
    assert len(rf) == 4 and rf['median'].notna().all()
    assert rf.loc[rf.representation_id.eq('R2'), 'topology_status'].eq('R2_EXPLORATORY_UNSTABLE').all()
    assert json.loads((output / 'summary.json').read_text())['outcome_definitions'][RF]['collection_evidence'] == [
        'RAW_BTRIS_COLLECTION_DAY']
    models = pd.read_csv(output / 'models.csv')
    assert models.loc[models.outcome.eq(RF) & models.representation_id.eq('R0'), 'status'].item() == E.CIRCULAR
    assert models.p_incremental.isna().all()


def test_missing_and_tampered_inputs_publish_no_results(automatic):
    repo, paths, _, _ = automatic
    paths['eda_long'].unlink()
    (paths['graph03_run'] / '03_graph_n3_results_summary.md').write_text('tampered')
    result = command(repo)
    assert result['status'] == 'NO_ELIGIBLE_DATA' and result['n_outcomes_with_data'] == 0
    assert {item['artifact'] for item in result['data_issues']} == {'eda_long', 'compatible_graph02_graph03_releases'}
    assert pd.read_csv(result['models']).p_incremental.isna().all()
    assert Path(result['figures'][0]).name == 'coverage.png'


def test_date_window_exact_episode_ties_definitions_and_censoring(calendar_originals):
    paths, _, _, _ = calendar_originals
    eda = pd.read_parquet(paths['eda_long']).iloc[:40].copy()
    eda['unit'] = 'IU/mL'
    eda.loc[:4, 'lab_date'] = ['2019-01-01', '2018-12-31', '2020-01-01', '2020-01-02', '2019-12-30']
    eda['specimen_datetime'] = eda.lab_date
    # An episode belonging to another patient cannot supply a result.
    eda.loc[4, 'matched_clinical_episode_id'] = eda.loc[5, 'matched_clinical_episode_id']
    eda.loc[6, 'unit'] = 'mg/L'
    eda.loc[7, 'assay'] = 'alternate assay'
    eda.loc[8, 'result_operator'] = '<'
    eda.loc[9, 'specimen_datetime'] = '2019-12-29'
    eda.loc[23, 'specimen_datetime'] = None
    latest_tie = eda.iloc[[10]].assign(result_numeric_exact=400, result_raw='400')
    older = eda.iloc[[10]].assign(lab_date='2019-12-28', specimen_datetime='2019-12-28')
    eda = pd.concat([eda, latest_tie, older, eda.iloc[[11]]], ignore_index=True)
    tables = {role: pd.read_parquet(paths[role]) for role in ('step01', 'step10', 'context')}
    tables['eda_long'] = eda
    # Different clock precision on the same official day is sufficient here.
    official = tables['step10'].is_clinical_baseline
    tables['step10'].loc[official, 'clinical_baseline_date'] = '2020-01-01T21:15:00-05:00'
    baseline = pd.read_csv(paths['baseline'], dtype={'patient_id': 'string'})
    candidates, ledger = E._candidates(tables, baseline, E._btris_records({}, []))
    selected, rejected, definition = E._outcome(candidates, ledger, RF)
    assert {'SYNTHETIC_0', 'SYNTHETIC_5', 'SYNTHETIC_11'} <= set(selected.patient_id)
    assert not ({f'SYNTHETIC_{idx}' for idx in (1, 2, 3, 4, 6, 7, 8, 9, 10, 23)} & set(selected.patient_id))
    assert selected.patient_id.is_unique and definition[-2:] == [None, 'IU/mL']
    assert {'CENSORED_RESULT', 'SAME_DAY_AS_BASELINE', 'AFTER_BASELINE', 'OUTSIDE_365_DAY_LOOKBACK',
            'COLLECTION_DAY_DISCORDANT', 'DISCORDANT_LATEST_DAY', 'NON_PRIMARY_TEST_DEFINITION',
            'IDENTICAL_DUPLICATE', 'COLLECTION_DAY_UNVERIFIED'} <= set(rejected.reason)


def test_complementary_suppression_is_shared_by_profiles_and_coverage():
    label = {'representation_id': 'R0', 'outcome': RF_TEXT, 'topology_status': 'FROZEN_COMMUNITIES'}
    frame = pd.DataFrame(dict(patient_id=[f'SYNTHETIC_{i}' for i in range(30)],
                             community_id=['0'] * 13 + ['1'] * 14 + ['2'] * 3,
                             value=[1.] * 2 + [0.] * 11 + [1.] * 12 + [np.nan] * 2 + [0.] * 3))
    first, second, withheld = profiles = E._profiles(frame, label, False)
    assert first['n_measured'] == 13 and 'n_positive' not in first and 'n_negative' not in first
    assert second['status'] == 'SUPPRESSED' and 'n_total' not in second and 'n_measured' not in second
    assert withheld['status'] == 'SUPPRESSED' and 'community_id' not in withheld
    assert 'n_total' not in E._coverage(frame, pd.DataFrame(), label, profiles)
    repeated = pd.DataFrame(dict(patient_key=['one'] * 30, reason='CENSORED'))
    assert list(E._safe_exclusions(repeated, RF)) == []


def test_continuous_binary_support_and_rank():
    rng = np.random.default_rng(2026)
    n = 100
    frame = pd.DataFrame({'value': rng.normal(size=n), 'community_id': ['0'] * 50 + ['1'] * 50,
                          'demo__age_at_baseline': rng.uniform(20, 70, n), 'demo__sex': ['F', 'M'] * 50,
                          'clinical_baseline_date': ['2020-01-01', '2015-01-01', '2010-01-01'] * 33 + ['2020-01-01']})
    info, prepared = E._model_support(frame, True)
    assert info['n'] == n
    assert E._estimate(prepared, True)['status'] == 'EXPLORATORY_ESTIMATED'
    frame['value'] = rng.integers(0, 2, n)
    info, prepared = E._model_support(frame, False)
    assert E._estimate(prepared, False)['status'] == 'EXPLORATORY_ESTIMATED'
    info, prepared = E._model_support(pd.concat([frame.iloc[:10], frame.iloc[50:59]]), False)
    assert prepared is None and 'INSUFFICIENT' in info['reason']
    frame['demo__age_at_baseline'] = frame.community_id.astype(float) + 20
    info, prepared = E._model_support(frame, False)
    assert prepared is None and 'RANK_DEFICIENT_DESIGN' in info['reason']


def test_frozen_lineage_and_single_rf_test_per_representation(automatic):
    repo, paths, _, _ = automatic
    eda = pd.read_parquet(paths['eda_long'])
    eda['unit'] = 'IU/mL'
    eda['result_text'] = np.where((eda.index % 3) == 0, 'positive', 'negative')
    urine = 'lab__urine_protein_qualitative__text'
    qualitative = [eda.assign(canonical_analyte=E.LAB_SOURCES[source], specimen_datetime=eda.lab_date,
                              order_name_original=source, unit=None)
                   for source in (ANA, HEP2, urine)]
    pd.concat([eda, *qualitative], ignore_index=True).to_parquet(paths['eda_long'], index=False)
    release = json.loads(paths['graph02_release'].read_text())
    registry = release['relative_paths']['clinical_registry']
    features = release['relative_paths']['feature_sets']
    release_dir = paths['graph02_release'].parent
    pd.DataFrame({'source_column': [RF, RF_TEXT]}).to_csv(release_dir / registry, index=False)
    sets = pd.read_csv(release_dir / features)
    sets.loc[sets.representation_id.eq('R2'), 'source_column'] = RF_TEXT
    sets.to_csv(release_dir / features, index=False)
    for filename in (registry, features):
        release['sha256_by_artifact'][filename] = digest(release_dir / filename)
    release['features_sha256'] = digest(release_dir / features)
    paths['graph02_release'].write_text(json.dumps(release))
    graph03 = paths['graph03_run']
    role_path = graph03 / '03_graph_clinical_feature_roles.csv'
    roles = pd.read_csv(role_path)
    rf = roles.source_column.isin({RF, RF_TEXT})
    roles.loc[rf, 'tier'] = 'N3_REVIEWED'
    roles.loc[rf, 'independence_status'] = 'REVIEWED_NO_CONFLICT'
    roles['inferential_eligible'] = rf
    roles.to_csv(role_path, index=False)
    for path in graph03.glob('*.json'):
        document = json.loads(path.read_text())
        document['graph02_release_sha256'] = digest(paths['graph02_release'])
        path.write_text(json.dumps(document))
    summary = graph03 / '03_graph_results_summary.json'
    content = json.loads(summary.read_text())
    content['sha256_by_artifact'] = {p.name: digest(p) for p in graph03.iterdir() if p != summary}
    summary.write_text(json.dumps(content))
    before = paths_before(paths)

    result = command(repo)
    assert result['n_model_pairs_estimable'] == 1, result
    assert result['n_outcomes_with_data'] == 5
    rows = pd.read_csv(result['models']).set_index(['representation_id', 'outcome'])
    assert rows.loc[('R0', RF), 'status'] == 'EXPLORATORY_ESTIMATED'
    assert rows.loc[('R0', RF), 'p_incremental'] == rows.loc[('R0', RF), 'q_bh']
    assert rows.loc[('R0', RF_TEXT), 'status'] == 'ALTERNATIVE_RF_ENCODING_DESCRIPTIVE_ONLY'
    assert rows.loc[('R2', RF), 'status'] == rows.loc[('R2', RF_TEXT), 'status'] == E.CIRCULAR
    assert paths_before(paths) == before

"""Stage05 integration using the actual Step01 and Step03 transformation contracts."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import common
from src.studies._shared import sha256_file
from src.studies.pharma.stage05 import contracts, reporting
from src.studies.pharma.stage05.deck_collection import collect
from src.studies.pharma.stage05.step03_adapter import Step03
from src.studies.pharma.stage05.transition_mechanisms import DOMAINS

PHARMA = Path(__file__).resolve().parents[1] / 'src' / 'studies' / 'pharma'


def module(filename):
    spec = spec_from_file_location('test_stage05_' + filename[:2], PHARMA / filename)
    result = module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture_master():
    rng = np.random.default_rng(37)
    rows = []
    for n in range(72):
        low = rng.integers(0, 3, size=6)
        patient = f'synthetic-{n}'
        for i, state in enumerate(low):
            dai = int(state == 0) * 5 + int(rng.integers(0, 3))
            pri = (6 if state == 1 else 3) + rng.uniform(-.5, .5)
            day = pd.Timestamp('2010-01-04') + pd.Timedelta((n % 5) + i * 480, unit='D')
            densities = {f'essdai__{d}_ordinal_score': 0 for d in DOMAINS}
            densities.update(essdai__pulmonary_ordinal_score=dai // 5,
                             essdai__biological_ordinal_score=dai % 5)
            rows.append(dict(patient_id=patient, clinical_episode_id=f'episode-{i}',
                             clinical_anchor_date=day, clinical_visit_number=i+1, clinical_visit=True,
                             is_clinical_baseline=i == 0, pop__status=contracts.POPS[state],
                             essdai__total=dai, esspri__total=pri, esspri__dryness=pri + .4,
                             esspri__pain=pri - .4, esspri__fatigue=pri,
                             demo__age_at_visit=40. + n/3 + i + rng.normal(0, .4),
                             demo__protocol='SYNTHETIC-A' if n % 3 else 'SYNTHETIC-B',
                             lab__esr__value=15. + rng.gamma(2, 2),
                             lab__igg__value=800. + rng.gamma(2, 31),
                             sero__anti_ro_ssa__ever_positive_through_episode=(
                                 True if n % 3 == 0 else False if n % 3 == 1 else None),
                             medication__ctdb_reported_names='hydroxychloroquine' if n % 2 == 0 else None,
                             **densities))
    return pd.DataFrame(rows)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    data, out = tmp_path / 'data', tmp_path / 'outputs'
    analytic = data / 'analytic'
    monkeypatch.setattr(common, 'DATA_DIR', data)
    effective = analytic / 'semantic_harmonization' / '*' / common._EFFECTIVE_FILENAME
    master_path = Path(str(effective).replace('*', 'fixture'))
    monkeypatch.setattr(common, 'EFFECTIVE_LONGITUDINAL_PARQUET', effective)
    monkeypatch.setattr(common, 'ANALYTIC_DATA_DIR', analytic)
    monkeypatch.setattr(common, 'INTEGRATED_LONGITUDINAL_PARQUET',
                        analytic / '10_build_integrated_longitudinal_dataset' /
                        '10_integrated_longitudinal_clinical_episode.parquet')
    monkeypatch.setattr(common, 'INTEGRATED_BASELINE_PARQUET', analytic / 'unavailable.parquet')
    monkeypatch.setattr(common, 'OUTPUTS_DIR', out)
    monkeypatch.setattr(common, 'BLOCKA_TABLES_DIR', out / 'tables' / 'blockA')
    monkeypatch.setattr(common, 'STUDIES_ANALYTIC_DIR', analytic / 'studies')
    for key in ('tables', 'figures', 'qc', 'logs'):
        monkeypatch.setattr(common, f'STUDIES_{key.upper()}_DIR', out / key / 'studies')

    paths = contracts.source_paths() | {'master': master_path}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    master = fixture_master()
    master.to_parquet(master_path)
    paths = contracts.source_paths()
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    context = master[contracts.KEYS].copy()
    profile = []
    for name, unit in [('esr', 'mm/h'), ('igg', 'mg/dL')]:
        stem = f'lab__{name}'
        context[f'{stem}__unit'] = unit
        context[f'{stem}__measurement_date'] = master.clinical_anchor_date
        profile.append(dict(lab=stem, recommended_use='numeric_longitudinal', result_source='value',
                            value_column=f'{stem}__value', unit_assessment_status='available',
                            unit_column=f'{stem}__unit', measurement_date_column=f'{stem}__measurement_date'))
    context.to_parquet(paths['context'])
    pd.DataFrame(profile).to_csv(paths['profile'], index=False)
    registry = pd.DataFrame({'public_variable': master.columns, 'role': 'ANALYTIC'})
    registry.to_csv(paths['registry'], index=False)
    paths['profile_summary'].write_text(json.dumps(
        {'input_hashes': [{'role': 'integrated', 'sha256': sha256_file(master_path)}]}))

    next_episode = master.groupby('patient_id').shift(-1)
    valid = next_episode.clinical_episode_id.notna()
    start, stop = master.loc[valid].reset_index(drop=True), next_episode.loc[valid].reset_index(drop=True)
    intervals = pd.DataFrame(dict(patient_id=start.patient_id, from_clinical_episode_id=start.clinical_episode_id,
                                  to_clinical_episode_id=stop.clinical_episode_id,
                                  from_clinical_anchor_date=start.clinical_anchor_date,
                                  to_clinical_anchor_date=stop.clinical_anchor_date,
                                  from_pop=start.pop__status, to_pop=stop.pop__status))
    intervals['interval_days'] = (intervals.to_clinical_anchor_date - intervals.from_clinical_anchor_date).dt.days
    intervals['interval_years'] = intervals.interval_days / 365.25
    step01 = module('01_run_pharma_main.py')
    resolved, _ = step01._resolve(master)
    specs = step01._lab_specs(pd.DataFrame(profile), master, 'numeric_longitudinal')
    intervals = step01.build_analytic(intervals, master, resolved, specs)
    intervals.to_parquet(paths['intervals'])
    upstream_config = step01.load_config(PHARMA / 'config.yaml')
    step01.lab_availability(intervals, specs, upstream_config).to_csv(paths['lab_availability'], index=False)
    pd.DataFrame([dict(feature=key, resolved_column=value[1]) for key, value in resolved.items()]).to_csv(
        paths['feature_availability'], index=False)

    step03 = Step03().module
    assay = 'lab__esr'
    sensitivity = pd.DataFrame([dict(exposure=assay, exposure_family='laboratory',
                                    model_status='success', interpretability_status='interpretable',
                                    direction_consistent=True)])
    sensitivity.to_csv(paths['sensitivity'], index=False)
    trace, eligible = step03.build_candidate_trace(sensitivity)
    trace.to_csv(paths['candidates'], index=False)
    prepared, mean, sd, reason = step03._prepare_intervals(intervals, assay)
    stacked = step03.stack_intervals(prepared, assay)
    support = step03.transition_support(prepared, stacked, assay, 10)
    support.to_csv(paths['support'], index=False)
    models = pd.DataFrame(step03.fit_biomarker(assay, stacked, support, mean, sd, eligible, True, reason))
    models.to_csv(paths['models'], index=False)
    step03.summarize_biomarkers(trace, stacked, support, models).to_csv(paths['model_summary'], index=False)
    stage = module('05_pharma_deck_evidence.py')
    settings, mapping = contracts.load_settings(PHARMA)
    settings.update(bootstrap_replicates=5)
    monkeypatch.setattr(stage, 'load_settings', lambda _: (settings, mapping))
    return stage, settings, mapping, paths, tmp_path


def test_end_to_end_and_previous_publication_preserved(pipeline):
    stage, settings, mapping, paths, root = pipeline
    upstream = {key: sha256_file(path) for key, path in paths.items() if path.is_file()}
    stage.run()
    folders = reporting.output_dirs()
    manifest_path = folders['qc'] / reporting.MANIFEST
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    for item in manifest['artifacts']:
        file = root / item['path']
        if item['sha256']:
            assert sha256_file(file) == item['sha256']
        else:
            assert not file.exists()
    assert len(list(folders['tables'].glob('*.csv'))) == 29
    transitions = pd.read_csv(folders['tables'] / '05_pharma_transition_evidence_summary.csv')
    assert set(zip(transitions.from_pop, transitions.to_pop)) == set(contracts.ROUTES)
    assert transitions.priority.tolist().count('primary_contrast') == 1
    risk = pd.read_csv(folders['tables'] / '05_pharma_absolute_risk_by_biomarker.csv')
    assert set(risk.loc[risk.status.eq('estimated'), 'scheme']) == {'first_arrival', 'occupancy'}
    assert (risk.loc[risk.status.eq('estimated'), 'n_bootstrap_successful'] == 5).any()
    specimen = pd.read_csv(folders['tables'] / '05_pharma_specimen_availability.csv')
    assert specimen.status.tolist() == ['not_available']
    assert all(sha256_file(paths[key]) == digest for key, digest in upstream.items())
    raw_artifacts = list(folders['tables'].glob('*.csv'))
    assert all('synthetic-0' not in path.read_text() for path in raw_artifacts)
    zip_path, statuses = collect(root)
    assert zip_path.is_file()
    assert statuses['FOUND'] >= 25
    assert statuses['NOT_ESTIMABLE'] >= 1
    first_risk = risk.copy()
    stage.run()
    pd.testing.assert_frame_equal(pd.read_csv(folders['tables'] / '05_pharma_absolute_risk_by_biomarker.csv'), first_risk)

    intervals = pd.read_parquet(paths['intervals'])
    intervals.loc[0, 'to_clinical_anchor_date'] += pd.Timedelta(1, unit='D')
    intervals.to_parquet(paths['intervals'])
    with pytest.raises(ValueError, match='discordant anchor'):
        stage.run()
    assert manifest_path.read_bytes() == manifest_bytes


def test_fixed_sources_reject_mixed_runs(pipeline):
    _, settings, mapping, paths, _ = pipeline
    summary = json.loads(paths['profile_summary'].read_text())
    summary['input_hashes'][0]['sha256'] = 'incompatible'
    paths['profile_summary'].write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='Effective master hash'):
        contracts.load_inputs(settings, mapping)


def test_incompatible_optional_inventory_does_not_block_preflight(pipeline):
    _, settings, mapping, paths, _ = pipeline
    paths['specimens'].write_text('synthetic invalid parquet')
    loaded = contracts.load_inputs(settings, mapping)
    assert 'specimens' not in loaded.tables
    assert loaded.preflight['sources']['specimens']['reason'].startswith('incompatible_optional_source')


def test_canonical_adjacency_does_not_bridge_unclassifiable(pipeline):
    _, settings, mapping, paths, _ = pipeline
    master = pd.read_parquet(paths['master'])
    patient = master.patient_id.iloc[0]
    group = master.loc[master.patient_id.eq(patient)]
    middle = group.index[2]
    master.loc[middle, ['pop__status', 'essdai__total']] = ['Unclassifiable', np.nan]
    intervals = pd.read_parquet(paths['intervals'])
    intervals = intervals.loc[~(intervals.patient_id.eq(patient) &
                                (intervals.from_clinical_episode_id.eq(group.iloc[2].clinical_episode_id) |
                                 intervals.to_clinical_episode_id.eq(group.iloc[2].clinical_episode_id)))]
    _, _, summary = contracts.validate_inputs(master, intervals, settings)
    assert summary['n_positive_adjacencies_touching_unclassifiable'] == 2
    assert summary['n_classifiable_gaps'] == 1


def test_no_active_block_b_dependency():
    modules = list((PHARMA / 'stage05').glob('*.py')) + [PHARMA / '05_pharma_deck_evidence.py']
    for path in modules:
        if path.name == 'deck_collection.py':
            continue
        assert 'block_b' not in path.read_text().casefold()
        assert 'blockb' not in path.read_text().casefold()

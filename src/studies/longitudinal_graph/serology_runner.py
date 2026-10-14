"""Additive S4 products; historical runner and its published bundle remain separate."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from src.studies.longitudinal_graph import validation as qc
from src.studies.longitudinal_graph.serology_ohe import COLUMNS, SOURCES, join_ohe, s4_preprocessor, validate_ohe
from src.studies.longitudinal_graph.stability import bootstrap_mapper


def input_status(args, config, api):
    if not config.get('serology_ohe', {}).get('enabled'): return {'status': 'disabled'}
    path = getattr(args, 'serology_state', None) or args.state.with_name('01_longitudinal_serology_ohe_state.parquet')
    raw = path.with_name('01_longitudinal_serology_ohe_raw.parquet')
    provenance = path.with_name('01_serology_ohe_provenance.json')
    missing = [str(p) for p in (path, raw, provenance) if not p.exists()]
    if missing: return {'status': 'blocked', 'reason': 'missing_inputs', 'missing_inputs': missing}
    import json
    try:
        report = json.loads(provenance.read_text())
        expected = report['artifact_sha256']
        if report.get('config_sha256') != api.sha256_file(args.config): raise ValueError('OHE configuration changed since preparation')
        from src.studies.longitudinal_graph.serology_ohe import load_mapping
        mapping, mapping_path = load_mapping(config, api.ROOT)
        if report.get('mapping_sha256') != api.sha256_file(mapping_path): raise ValueError('OHE mapping changed since preparation')
        integrated = getattr(args, 'integrated', None)
        if integrated is not None and integrated.exists() and report.get('input_sha256') != api.sha256_file(integrated):
            raise ValueError('Integrated input changed since OHE preparation')
        for p in (path, raw):
            if api.sha256_file(p) != expected[p.name]: raise ValueError('OHE provenance hash mismatch')
        if report.get('status') != 'completed': raise ValueError('OHE provenance not completed')
        ohe = pd.read_parquet(path); pre = pd.read_parquet(raw)
        validate_ohe(ohe); validate_ohe(pre)
        if not ohe.equals(pre): raise ValueError('OHE raw/state differ')
        join_ohe(pd.read_parquet(args.state), ohe)
    except (ValueError, KeyError, OSError) as error:
        return {'status': 'blocked', 'reason': str(error)}
    return {'status': 'validated', 'state': str(path), 'raw': str(raw), 'provenance': str(provenance)}


def execute_s4(args, api, state, raw, metadata, legacy, primary, destination, config):
    status = input_status(args, config, api)
    if status['status'] == 'disabled': return status
    if status['status'] != 'validated':
        api.write_json(destination['tables'] / '02_s4_mapper_summary.json', status)
        pd.DataFrame([{'scenario': n, **status} for n in ('S4-serology-OHE', 'S4-without-missing', 'S4-no-serology')]).to_csv(destination['tables'] / '02_s4_mapper_sensitivity_comparison.csv', index=False)
        return status
    import json
    provenance = json.loads(Path(status['provenance']).read_text())
    experiment = join_ohe(state, pd.read_parquet(status['state']))
    if any(f not in raw for f in legacy):
        result = {'status': 'blocked', 'reason': 'preimputation_legacy_inputs_unavailable'}
        api.write_json(destination['tables'] / '02_s4_mapper_summary.json', result)
        return result
    refit = join_ohe(raw[api.KEYS + legacy], pd.read_parquet(status['raw']))
    counts = pd.Series([api.feature_family(f) for f in legacy]).value_counts()
    weights = {f: 1 / np.sqrt(counts[api.feature_family(f)]) for f in legacy}
    weights.update({f: float(config['serology_ohe'].get('family_weight', 1.)) / np.sqrt(12) for f in COLUMNS})
    cfg = copy.deepcopy(config)
    cfg.setdefault('mapper_validation', {})['bootstrap_replicates'] = max(200, int(cfg.get('mapper_validation', {}).get('bootstrap_replicates', 200)))
    rows = []; registry = []; scenarios = {}; manifests = []; circularity = []
    for name, indicators in (('S4-serology-OHE', COLUMNS), ('S4-without-missing', [f for f in COLUMNS if not f.endswith('__missing')]), ('S4-no-serology', [])):
        features = legacy + indicators
        options = {'feature_weights': {f: weights[f] for f in features}, 'indicator_columns': indicators} if indicators else {}
        scenario = api.run_mapper_scenario(experiment if indicators else state, features, cfg, family_balance=True, scenario_name=name, **options)
        scenarios[name] = scenario
        boot, matching, bootstrap = bootstrap_mapper(refit if indicators else raw, scenario, cfg, features,
            api.run_mapper_scenario, api.reduce_redundancy, api.jaccard,
            preprocessor=s4_preprocessor(legacy), fit_options=options)
        _, comparison = api.compare_mapper_scenarios(primary, scenario)
        stem = {'S4-serology-OHE': 's4', 'S4-without-missing': 's4_without_missing', 'S4-no-serology': 's4_no_serology'}[name]
        for suffix, key in (('nodes', 'node_table'), ('topological_edges', 'topological_edges'), ('macrostate_summary', 'macrostate_table'), ('visit_membership', 'visit_membership'), ('cover_qc', 'cover_qc'), ('noise_qc', 'noise_qc')):
            scenario[key].to_csv(destination['tables'] / f'02_{stem}_mapper_{suffix}.csv', index=False)
        boot.to_csv(destination['tables'] / f'02_{stem}_mapper_bootstrap_replicates.csv', index=False)
        matching.to_csv(destination['tables'] / f'02_{stem}_mapper_macrostate_stability.csv', index=False)
        pca = scenario['pca']
        loadings = pd.DataFrame(pca.components_.T, index=features, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
        loadings.rename_axis('feature').reset_index().to_csv(destination['tables'] / f'02_{stem}_longitudinal_pca_loadings.csv', index=False)
        transformed = RobustScaler().fit_transform(experiment[features])
        for f in indicators: transformed[:, features.index(f)] = experiment[f]
        transformed *= np.array([weights[f] for f in features])
        for i, f in enumerate(features):
            manifests.append({'scenario': name, 'feature': f, 'family': api.feature_family(f), 'biomarker': f.split('__')[1] if f in COLUMNS else None,
                              'missing_indicator': f.endswith('__missing'), 'weight': weights[f], 'constant': experiment[f].nunique() < 2,
                              'transformation': 'identity_binary_then_fixed_weight_PCA_centering' if f in COLUMNS else 'historical_robust_scaler_then_family_weight',
                              'empirical_weighted_variance': float(np.var(transformed[:, i])),
                              'PC1_squared_loading': float(pca.components_[0, i] ** 2), 'PC2_squared_loading': float(pca.components_[1, i] ** 2),
                              'PC1_to_PC10_squared_loading': float((pca.components_[:10, i] ** 2).sum())})
        for marker, source in SOURCES.items():
            circularity.append({'scenario': name, 'source': source, 'present_in_source': source in metadata,
                                'encoded': True, 'used_in_S3': False, 'used_in_S4': bool(indicators),
                                'held_out': not indicators, 'independent_external_validator': False if indicators else True,
                                'derived_features': '|'.join(f for f in indicators if f.startswith(f'sero_ohe__{marker}__'))})
        intervals = api.interval_coverage_summary(pd.read_parquet(args.intervals), scenario['visit_membership'], cfg['temporal']['maximum_interval_years']) if args.intervals.exists() else {'status': 'not_tested'}
        rows.append({**scenario['summary'], 'n_patients': state.patient_id.nunique(), 'n_visits': len(state),
                     'n_serology_concepts': 3 if indicators else 0, 'n_ohe_columns': len(indicators),
                     'pca_pc1_variance': pca.explained_variance_ratio_[0], 'pca_pc2_variance': pca.explained_variance_ratio_[1],
                     'ARI_vs_S3': comparison['ari'], 'patient_Jaccard': comparison['patient_jaccard_vs_primary'],
                     'visit_Jaccard': comparison['visit_jaccard_vs_primary'], 'n_visits_compared': comparison['n_visits_compared'],
                     'n_patients_compared': comparison['n_patients_compared'], 'ari_status': comparison['ari_status'],
                     'bootstrap_completed': bootstrap.get('replicates_completed', 0), 'bootstrap_failed': bootstrap.get('replicates_failed', 0),
                     'bootstrap_no_nodes': bootstrap.get('replicates_no_nodes', 0), 'covered_intervals': intervals,
                     'pca_cumulative_variance': scenario['summary']['cumulative_variance'],
                     'covered_visits': scenario['summary'].get('n_mapper_covered_visits'),
                     'supported_visits': scenario['summary'].get('n_supported_macrostate_covered_visits'),
                     'hard_assigned_visits': scenario['summary'].get('n_hard_assigned_visits'),
                     'ambiguous_visits': scenario['summary'].get('n_membership_ties'),
                     'uncovered_visits': scenario['summary'].get('n_uncovered_visits'),
                     'stable_macrostates': sum(x['stable_by_prespecified_rule'] for x in bootstrap.get('stability', []) if x['kind'] == 'macrostate'),
                     'status': 'candidate_requires_review', 'blockers': 'scientific_review_required',
                     'pending_checks': 'temporal_provenance;branch_definition;external_validation;hsCRP_PRO_controls'})
        manifest = qc.run_manifest(api.ROOT, args.config, [args.state, args.metadata, Path(status['state']), Path(status['raw']), Path(status['provenance'])], cfg['random_seed'], name, api.sha256_file)
        manifest.update(feature_list=features, feature_weights={f: weights[f] for f in features}, mapping_provenance=provenance,
                        bootstrap=bootstrap, interval_coverage=intervals)
        registry.append(manifest)
        from src.studies.longitudinal_graph.figures import mapper_figures
        figure_dir = destination['figures'] / stem; figure_dir.mkdir(exist_ok=True)
        mapper_figures(scenario, figure_dir, cfg['figures']['dpi'])
    representation = pd.DataFrame(manifests)
    representation.to_csv(destination['tables'] / '02_s4_mapper_representation_manifest.csv', index=False)
    representation.groupby(['scenario', 'family'], dropna=False).sum(numeric_only=True).reset_index().to_csv(destination['tables'] / '02_s4_mapper_family_contributions.csv', index=False)
    representation.loc[representation.biomarker.notna()].groupby(['scenario', 'biomarker']).sum(numeric_only=True).reset_index().to_csv(destination['tables'] / '02_s4_mapper_biomarker_contributions.csv', index=False)
    derived_manifest = pd.DataFrame([{'source': source, 'feature': f} for marker, source in SOURCES.items() for f in COLUMNS if f.startswith(f'sero_ohe__{marker}__')])
    lineage = pd.concat([qc.serology_lineage_audit(derived_manifest, scenario['features'], metadata).assign(scenario=name) for name, scenario in scenarios.items()], ignore_index=True)
    lineage.to_csv(destination['tables'] / '02_s4_biomarker_eligibility_audit.csv', index=False)
    qc.serology_episode_concordance(pd.read_parquet(status['state']), metadata).to_csv(destination['tables'] / '02_s4_serology_temporal_concordance.csv', index=False)
    admin, availability, strata, lens = qc.administrative_audit(scenarios['S4-serology-OHE']['visit_membership'], metadata, scenarios['S4-serology-OHE']['embedding'], cfg)
    for suffix, table in (('association', admin), ('availability', availability), ('stratified_coverage', strata), ('lens_qc', lens)):
        table.to_csv(destination['tables'] / f'02_s4_mapper_admin_{suffix}.csv', index=False)
    pd.DataFrame(circularity).to_csv(destination['tables'] / '02_s4_mapper_circularity_audit.csv', index=False)
    detail = qc.coverage_detail(scenarios['S4-serology-OHE']['visit_membership'], scenarios['S4-serology-OHE'])
    qc.coverage_by_stratum(detail, metadata).to_csv(destination['tables'] / '02_s4_mapper_coverage_by_stratum.csv', index=False)
    full_vs_without = api.compare_mapper_scenarios(scenarios['S4-serology-OHE'], scenarios['S4-without-missing'])[1]
    table = pd.DataFrame([{'scenario': 'S3-primary', **primary['summary'], 'n_patients': state.patient_id.nunique(), 'n_visits': len(state), 'n_ohe_columns': 0, 'n_serology_concepts': 0}, *rows])
    table.to_csv(destination['tables'] / '02_s4_mapper_sensitivity_comparison.csv', index=False)
    result = {'status': 'candidate_requires_review', 'scenarios': registry, 'comparison': rows,
              'S3_no_serology_control': api.compare_mapper_scenarios(primary, scenarios['S4-no-serology'])[1],
              'comparison_universe': 'identical_input_keys; ARI common non-tied supported hard visits; Jaccard matched supported macrostates',
              'missingness_sensitivity': full_vs_without,
              'representation_stability_gate': {'status': 'blocked', 'blockers': ['documented_scientific_approval_required'], 'temporal_flow_executed': False}}
    api.write_json(destination['tables'] / '02_s4_mapper_summary.json', result)
    return result

"""Versioned candidate QC, consuming S3-primary without ever refitting it.

Artifacts with visit keys remain in the local/restricted output namespace. Only
aggregate, synthetic validation reports belong in a public source repository.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.studies.longitudinal_graph import validation as qc
from src.studies.longitudinal_graph._review import review_qc
from src.studies.longitudinal_graph._review.graph_export import graph_views, membership_tables, write_graphs
from src.studies.longitudinal_graph._review.reference import (new_bundle, reference_artifact, safe_id,
    seal_reference, sha256, verify_reference, protect_path, inventory)
from src.studies.longitudinal_graph._review.scenarios import (HSCRP, TrainingPreprocessor,
    candidate_registry, negative_patient_blocks, observed_pruning)
from src.studies.longitudinal_graph.stability import bootstrap_mapper
from src.studies.longitudinal_graph.figures import candidate_representation_figures


def load_reference(reference, api):
    summary = json.loads(reference_artifact(reference, "02_longitudinal_mapper_summary", ".json").read_text())
    try:
        membership = pd.read_parquet(reference_artifact(reference, "02_visit_mapper_membership", ".parquet"))
    except ValueError:
        membership = pd.read_csv(reference_artifact(reference, "02_mapper_visit_membership", ".csv"))
    qc.validate_keys(membership, "historical membership")
    for column in ("mapper_graph_covered", "mapper_covered", "macrostate_membership_tie", "in_unsupported_macrostate"):
        if column not in membership:
            raise ValueError(f"Reference lacks historical membership semantic field: {column}")
        if not membership[column].isin([True, False]).all():
            raise ValueError(f"Reference has invalid boolean memberships: {column}")
    for key, count in api.coverage_summary(membership).items():
        if key.startswith("n_") and key in summary and summary[key] != count:
            raise ValueError(f"Historical visit coverage disagrees with summary: {key}")
    nodes = pd.read_csv(reference_artifact(reference, "02_mapper_nodes", ".csv"))
    macros = pd.read_csv(reference_artifact(reference, "02_mapper_macrostate_summary", ".csv"))
    node_records = pd.read_parquet(reference_artifact(reference, "02_mapper_node_membership", ".parquet"))
    indexed = node_records.merge(membership[api.KEYS].assign(_index=np.arange(len(membership))), on=api.KEYS, how="left", validate="many_to_one")
    if indexed._index.isna().any() or indexed[api.KEYS + ["node_id"]].duplicated().any():
        raise ValueError("Historical node memberships have missing or duplicate visit keys")
    graph_nodes = {node: set(group._index.astype(int)) for node, group in indexed.groupby("node_id")}
    if set(nodes.node_id) != set(graph_nodes):
        raise ValueError("Historical all-node membership inventory is incomplete")
    mapping = {}
    for node, group in indexed.groupby("node_id"):
        values = group.macrostate_id.dropna().unique()
        if len(values) > 1:
            raise ValueError("Historical node has inconsistent community identity")
        if len(values):
            mapping[node] = int(values[0])
    links = {n: [] for n in graph_nodes}
    names = sorted(graph_nodes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if graph_nodes[a] & graph_nodes[b]:
                links[a].append(b)
                links[b].append(a)
    edges = pd.read_csv(reference_artifact(reference, "02_mapper_topological_edges", ".csv"))
    value = {"summary": summary, "visit_membership": membership, "node_table": nodes,
             "macrostate_table": macros, "graph": {"nodes": graph_nodes, "links": links},
             "node_to_macrostate": mapping, "topological_edges": edges}
    raw = nx.Graph()
    raw.add_nodes_from(names)
    raw.add_edges_from((a, b) for a, neighbors in links.items() for b in neighbors)
    value["raw_nerve"] = raw
    tables, integrity = graph_views(value, membership)
    claimed = summary.get("graph_degree_distribution")
    if claimed is not None:
        claimed = {str(k): int(v) for k, v in claimed.items()}
        if claimed != integrity["views"]["full"]["degree_distribution"]:
            raise ValueError("Historical degree distribution disagrees with all-node memberships")
    if summary.get("n_connected_components") != integrity["views"]["full"]["n_components"]:
        raise ValueError("Historical component count disagrees with source memberships")
    # Validate endpoints and overlap counts in the actual frozen edge CSV, not just its row count.
    expected = tables["mapper_edges_supported"].set_index(["from_node", "to_node"])
    found = set()
    for row in edges.itertuples():
        pair = tuple(sorted((row.from_node, row.to_node)))
        if pair not in expected.index or pair in found:
            raise ValueError("Historical edge endpoints/duplicates are invalid")
        if row.n_shared_visits != expected.loc[pair, "n_shared_visits"]:
            raise ValueError("Historical edge shared-visit count is inconsistent")
        found.add(pair)
    return value, tables, integrity


def compare_reference(reference, scenario, api):
    left, right = reference["visit_membership"], scenario["visit_membership"]
    universe = left[api.KEYS + ["hard_macrostate", "mapper_graph_covered", "macrostate_membership_tie"]].merge(
        right[api.KEYS + ["hard_macrostate", "mapper_graph_covered", "macrostate_membership_tie"]], on=api.KEYS,
        how="outer", validate="one_to_one", suffixes=("_reference", "_candidate"), indicator=True)
    common = universe.loc[universe._merge.eq("both") & universe.hard_macrostate_reference.notna() &
                          universe.hard_macrostate_candidate.notna() & ~universe.macrostate_membership_tie_reference.eq(True) &
                          ~universe.macrostate_membership_tie_candidate.eq(True)]
    nl, nr = common.hard_macrostate_reference.nunique(), common.hard_macrostate_candidate.nunique()
    valid = len(common) >= 2 and min(nl, nr) >= 2
    stats = {"ari": float(adjusted_rand_score(common.hard_macrostate_reference.astype(int), common.hard_macrostate_candidate.astype(int))) if valid else None,
             "nmi": float(normalized_mutual_info_score(common.hard_macrostate_reference.astype(int), common.hard_macrostate_candidate.astype(int))) if valid else None,
             "ari_status": "common_supported_hard_partition" if valid else "not_interpretable_degenerate_or_insufficient",
             "n_visits_compared": len(common), "n_patients_compared": common.patient_id.nunique(),
             "n_reference_labels": nl, "n_candidate_labels": nr, "n_reference_visits": len(left), "n_candidate_visits": len(right),
             "n_common_cohort_visits": int(universe._merge.eq("both").sum()),
             "n_excluded_reference_visits": len(left) - len(common), "n_excluded_candidate_visits": len(right) - len(common),
             "fraction_reference_visits_compared": len(common) / max(len(left), 1),
             "comparison_universe": "same_visit_keys; supported_non_tied_hard_assignments_in_both"}
    sets_l, sets_r = api.macrostate_sets(reference), api.macrostate_sets(scenario)
    ids_l, ids_r = sorted(sets_l), sorted(sets_r)
    score = np.array([[api.jaccard(sets_l[a][1], sets_r[b][1]) for b in ids_r] for a in ids_l]).reshape(len(ids_l), len(ids_r))
    assigned = set()
    if score.size:
        i, j = linear_sum_assignment(-score)
        assigned = {(ids_l[a], ids_r[b]) for a, b in zip(i, j) if score[a, b] > 0}
    rows = []
    for a, (lv, lp) in sets_l.items():
        for b, (rv, rp) in sets_r.items():
            rows.append({"reference_macrostate_id": a, "candidate_macrostate_id": b,
                         "patient_jaccard": api.jaccard(lp, rp), "visit_jaccard": api.jaccard(lv, rv),
                         "n_reference_patients": len(lp), "n_candidate_patients": len(rp),
                         "one_to_one_match": (a, b) in assigned, "matching_method": "optimal_patient_jaccard_one_to_one",
                         "uncertainty": "in_sample_overlap_not_label_equivalence"})
    crosswalk = pd.DataFrame(rows, columns=["reference_macrostate_id", "candidate_macrostate_id", "patient_jaccard", "visit_jaccard", "n_reference_patients", "n_candidate_patients", "one_to_one_match", "matching_method", "uncertainty"])
    universe["comparison_eligible"] = universe.index.isin(common.index)
    universe["exclusion_reason"] = np.select([universe._merge.eq("left_only"), universe._merge.eq("right_only"),
        universe.macrostate_membership_tie_reference.eq(True) | universe.macrostate_membership_tie_candidate.eq(True),
        ~universe.comparison_eligible], ["candidate_visit_absent", "reference_visit_absent", "membership_tie", "no_supported_hard_assignment"], default="included")
    return stats, crosswalk, universe.drop(columns="_merge")


def reproduction_diff(staged, reference, seal_path, api):
    """Parallel legacy refit comparison; byte identity is not assumed or required."""
    original, _, _ = load_reference(reference, api)
    reproduced, _, _ = load_reference({"reference_root": str(staged), "files": inventory(staged)}, api)
    stats, crosswalk, universe = compare_reference(original, reproduced, api)
    counts = {}
    for field in ("n_patients", "n_visits", "n_features", "n_nodes", "n_connected_components", "n_macrostates", "n_mapper_covered_visits"):
        counts[field] = {"reference": original["summary"].get(field), "parallel_refit": reproduced["summary"].get(field)}
    api.write_json(staged / "qc/02_reproduction_diff.json", {"status": "parallel_refit_requires_review", "counts": counts,
        "partition_comparison": stats, "historical_dirty_state": reference.get("historical_git_dirty"),
        "reference_manifest_sha256_before": reference["reference_manifest_sha256"],
        "reference_manifest_sha256_after": verify_reference(seal_path)["reference_manifest_sha256"],
        "interpretation": "No claim of historical reproducibility solely from matching counts or partitions."})
    crosswalk.to_csv(staged / "tables/02_reproduction_crosswalk.csv", index=False)
    universe.to_csv(staged / "analytic/02_reproduction_universe.csv", index=False)


def _bootstrap_options(definition, config):
    settings = copy.deepcopy(config)
    if definition["isolated_change"] == "full80":
        settings["pca"]["max_components"] = None
    return settings, {"patient_weighted": definition["weight_rule"] == "equal_patient_mass"}


def _fit_experiment(state, raw, eligible, features, definition, config, api):
    selected = list(definition["feature_list"])
    experiment = state[api.KEYS + selected].copy()
    change = definition["isolated_change"]
    bounds = None
    if change in {"log1p", "winsor", "observed_pruning"}:
        if raw is None or any(f not in raw for f in eligible):
            raise ValueError("original_feature_observations_unavailable")
    if change == "log1p":
        observed = raw[HSCRP].astype(float)
        if observed.lt(0).any() or (observed.notna() & ~np.isfinite(observed)).any() or not observed.notna().any():
            raise ValueError("hscrp_negative_invalid_or_unobserved")
        if experiment[HSCRP].lt(0).any():
            raise ValueError("negative_hscrp_state")
        experiment[HSCRP] = np.log1p(experiment[HSCRP])
    if change == "winsor":
        observed = raw[HSCRP].astype(float)
        observed = observed[np.isfinite(observed)]
        if not len(observed):
            raise ValueError("hscrp_original_observations_unavailable")
        bounds = tuple(observed.quantile([.01, .99]))
        experiment[HSCRP] = experiment[HSCRP].clip(*bounds)
    if change == "observed_pruning":
        review = config.get("methodology_review", {})
        selected, _ = observed_pruning(raw, eligible, config["representation"]["redundancy_spearman_abs_threshold"],
            review.get("redundancy_minimum_pairs", 20), review.get("redundancy_minimum_patients", 10))
        experiment = state[api.KEYS + selected].copy()
    settings, options = _bootstrap_options(definition, config)
    scenario = api.run_mapper_scenario(experiment, selected, settings, family_balance=True, scenario_name=definition["name"], **options)
    definition["resolved_feature_list"] = selected
    definition["training_winsor_bounds"] = bounds
    return scenario, experiment, settings, options


def execute_review(args, api):
    root = Path(args.output_root) if args.output_root else qc.study_paths(api.common)["tables"].parent / "longitudinal_graph_review"
    run_id = safe_id(args.run_id or "")
    reference_id = safe_id(args.reference_run_id or "")
    # Reject a historical target even before reading data or creating a seal.
    target = root / "runs" / run_id
    reference_roots = [args.reference_root] if args.reference_root else []
    protect_path(target, reference_roots)
    seal_path = root / "references" / f"{reference_id}.json"
    if args.freeze_reference:
        if args.reference_root is None:
            raise ValueError("--freeze-reference requires --reference-root (existing historical archive)")
        if not args.dry_run and not seal_path.exists():
            seal_reference(args.reference_root, seal_path, reference_id)
    if not seal_path.exists():
        result = {"status": "blocked", "reason": "G0_frozen_reference_unavailable", "outputs_written": False,
                  "reference_status": "not_available", "reference_run_id": reference_id}
        if args.dry_run:
            print(json.dumps(result))
            return result
        raise ValueError("G0: sealed reference unavailable; supply --reference-root --freeze-reference once")
    reference = verify_reference(seal_path)
    if args.reference_root is not None and Path(reference["reference_root"]).resolve() != args.reference_root.resolve():
        raise ValueError("Reference root disagrees with the sealed run ID")
    protect_path(target, [reference["reference_root"]])
    config = api.load_config(args.config)
    review_config = getattr(args, "review_config", None)
    if review_config is not None:
        config["methodology_review"] = yaml.safe_load(review_config.read_text())
        if not isinstance(config["methodology_review"], dict):
            raise ValueError("Candidate review configuration must be a mapping")
    if args.dry_run:
        missing = [str(p) for p in (args.state, args.metadata, args.intervals) if not p.exists()]
        result = {"status": "not_run_missing_data" if missing and args.mode == "candidate-qc" else "validated",
                  "mode": args.mode, "reference_integrity": "passed", "missing_inputs": missing,
                  "outputs_written": False, "temporal_flow_executed": False}
        print(json.dumps(result))
        return result
    # Read-only reference audit does not require fitting data or recomputing Mapper.
    historic, ref_tables, ref_graph_qc = load_reference(reference, api)
    historical_configs = [Path(reference["reference_root"]) / name for name in reference["files"]
                          if Path(name).name == "config.yaml"]
    if len(historical_configs) == 1:
        historical_config = yaml.safe_load(historical_configs[0].read_text())
        for field in ("random_seed", "feature_selection", "representation", "pca", "mapper"):
            if historical_config.get(field) != config.get(field):
                raise ValueError(f"Candidate baseline differs from archived config: {field}; single-change comparisons require the historical modelling contract")
    reference_row = {**historic["summary"], "scenario": "S3-primary", "role": "reference", "status": "historical_frozen",
                     "gate": historic["summary"].get("representation_stability_gate", {}).get("status", "blocked"),
                     "historical_execution_status": historic["summary"].get("status")}
    rows = [reference_row]
    definitions, summaries = [], {}
    comparison_frames, crosswalk_frames, universe_frames = [], [], []
    report_manifest = {k: v for k, v in reference.items() if k.startswith("reference_") and k != "reference_root"}
    manifest = qc.run_manifest(api.ROOT, args.config, [], config["random_seed"], args.mode, api.sha256_file)
    manifest.update(run_id=run_id, mode=args.mode, schema_version="longitudinal_mapper_v2", **report_manifest)
    manifest["cli_resolution"] = {"run_id_generated": getattr(args, "run_id_generated", False),
                                  "reference_selection": getattr(args, "reference_selection", "explicit")}
    with new_bundle(target, [reference["reference_root"]], force=args.force) as staged:
        for folder in ("tables", "figures", "analytic", "diagnostics", "logs"):
            (staged / folder).mkdir()
        api.write_json(staged / "diagnostics/02_reference_integrity.json", {**report_manifest, "status": "passed", "artifact_inventory_sha256": sha256(seal_path), "historical_reproducibility": reference["reproducibility_status"]})
        for stem, table in ref_tables.items():
            table.to_csv(staged / "tables" / f"02_reference_{stem}.csv", index=False)
        api.write_json(staged / "diagnostics/02_reference_graph_integrity_qc.json", ref_graph_qc)
        if args.mode == "candidate-qc":
            state = pd.read_parquet(args.state).reset_index(drop=True)
            metadata = pd.read_parquet(args.metadata).reset_index(drop=True)
            from src.studies.longitudinal_graph.runner import validate_inputs
            validate_inputs(state, metadata, config, api)
            eligible = list(state.drop(columns=api.KEYS))
            features, _ = api.reduce_redundancy(state[eligible], config["representation"]["redundancy_spearman_abs_threshold"])
            raw_path = args.state.with_name("01_longitudinal_visit_raw.parquet")
            source_path = args.integrated if args.integrated.exists() else raw_path
            source = pd.read_parquet(source_path) if source_path.exists() else None
            raw = qc.align_source(state[api.KEYS], source, [f for f in eligible if f in source]) if source is not None else None
            raw_status = "preimputation_source_present_observation_provenance_unknown" if raw is not None else "source_unavailable"
            metadata, recovery = review_qc.recover_administration(state[api.KEYS], metadata, source)
            metadata["n_visits_per_patient"] = metadata.patient_id.map(metadata.groupby("patient_id").size())
            if "clinical_anchor_date" in metadata:
                metadata["calendar_year"] = pd.to_datetime(metadata.clinical_anchor_date).dt.year
                if raw is not None:
                    raw["clinical_anchor_date"] = metadata.clinical_anchor_date.to_numpy()
            recovery.to_csv(staged / "tables/02_admin_covariate_availability.csv", index=False)
            interval_source = pd.read_parquet(args.intervals)
            review_qc.validate_intervals(interval_source, metadata)
            fm = pd.read_csv(args.feature_manifest) if args.feature_manifest.exists() else pd.DataFrame()
            inputs = [args.state, args.metadata, args.intervals, args.config]
            if review_config is not None:
                inputs.append(review_config)
            if source is not None:
                inputs.append(source_path)
            if not fm.empty:
                inputs.append(args.feature_manifest)
            manifest["input_sha256"] = {str(path): sha256(path) for path in inputs}
            raw_table = review_qc.raw_feature_qc(raw, eligible, raw_status, fm)
            for stem in ("raw_feature_qc", "outlier_profile", "missingness_mechanism_audit"):
                raw_table.to_csv(staged / "tables" / f"02_{stem}.csv", index=False)
            if raw is not None:
                qc.observability_audit(raw, metadata, eligible).to_csv(staged / "tables/02_missingness_by_protocol_era_visit.csv", index=False)
                retained, audit = observed_pruning(raw, eligible, config["representation"]["redundancy_spearman_abs_threshold"],
                    config.get("methodology_review", {}).get("redundancy_minimum_pairs", 20), config.get("methodology_review", {}).get("redundancy_minimum_patients", 10))
                audit.to_csv(staged / "tables/02_redundancy_preimputation_audit.csv", index=False)
                pd.DataFrame({"feature_original": eligible, "decision": ["include" if f in retained else "exclude" for f in eligible],
                              "rationale": "observed_pairs_prespecified_coverage_priority; semantic_review_pending", "alternative_sensitivity": "S4-preimputation-pruning"}).to_csv(staged / "tables/02_redundancy_decisions.csv", index=False)
            definitions = candidate_registry(features, config)
            requested = set(args.scenario or [row["name"] for row in definitions])
            if requested - {row["name"] for row in definitions}:
                raise ValueError("Unknown scenario; new runs must use the prespecified S4 namespace")
            definitions = [row for row in definitions if row["name"] in requested]
            if recovery.n_conflicting_visits.sum():
                raise ValueError("Administrative metadata conflicts require upstream reconciliation")
            for definition in definitions:
                name = definition["name"]
                record = {"scenario": name, "role": "candidate", "gate": "blocked", "comparison_to": "S3-primary"}
                if definition["status"] == "not_applicable":
                    rows.append({**record, "status": "not_applicable", "reason": definition["reason"], "ari": None})
                    continue
                try:
                    scenario, experiment, settings, options = _fit_experiment(state, raw, eligible, features, definition, config, api)
                except (ValueError, np.linalg.LinAlgError) as error:
                    definition.update(status="not_applicable", reason=str(error))
                    rows.append({**record, "status": "not_applicable", "reason": str(error), "ari": None})
                    continue
                directory = staged / "scenarios" / name
                directory.mkdir(parents=True)
                tables, graph_qc = write_graphs(scenario, state, directory, config["figures"]["dpi"])
                candidate_representation_figures(scenario, experiment, raw, directory, config["figures"]["dpi"])
                for stem, table in tables.items():
                    table.to_csv(directory / f"02_{stem}.csv", index=False)
                api.write_json(directory / "02_graph_integrity_qc.json", graph_qc)
                node_m, macro_m = membership_tables(scenario, state)
                node_m.to_parquet(directory / "02_visit_node_membership.parquet", index=False)
                macro_m.to_parquet(directory / "02_visit_macrostate_membership.parquet", index=False)
                scenario["visit_membership"].to_parquet(directory / "02_visit_mapper_membership.parquet", index=False)
                partition = review_qc.coverage_partition(scenario["visit_membership"])
                partition.to_csv(directory / "02_coverage_visit_patient.csv", index=False)
                review = config.get("methodology_review", {})
                comparison_raw = None
                if raw is not None:
                    extra = [c for c in metadata if c not in raw and c not in api.KEYS]
                    comparison_raw = qc.align_source(raw, metadata, extra)
                review_qc.coverage_comparison(comparison_raw, partition, config["random_seed"], int(review.get("coverage_bootstrap_replicates", 200))).to_csv(directory / "02_coverage_comparison.csv", index=False)
                endpoint = review_qc.interval_contract(interval_source, partition, metadata, run_id, name, config["temporal"]["maximum_interval_years"])
                endpoint.to_parquet(directory / "02_interval_membership_contract.parquet", index=False)
                endpoint.groupby("exclusion_reason").agg(n_intervals=("patient_id", "size"), n_patients=("patient_id", "nunique"),
                    median_delta_t=("interval_days", "median"), q1_delta_t=("interval_days", lambda x: x.quantile(.25)), q3_delta_t=("interval_days", lambda x: x.quantile(.75))).reset_index().to_csv(directory / "02_interval_coverage_selection.csv", index=False)
                comparison, crosswalk, universe = compare_reference(historic, scenario, api)
                comparison_frames.append({"scenario": name, **comparison})
                crosswalk_frames.append(crosswalk.assign(scenario=name))
                universe_frames.append(universe.assign(scenario=name))
                admin, availability, strata, lens = review_qc.extended_administration(scenario["visit_membership"], metadata, scenario["embedding"], settings)
                admin.to_csv(directory / "02_administrative_qc_extended.csv", index=False)
                availability.to_csv(directory / "02_admin_covariate_availability.csv", index=False)
                strata.to_csv(directory / "02_admin_composition.csv", index=False)
                lens.to_csv(directory / "02_admin_lens_association.csv", index=False)
                pd.DataFrame([{"status": "not_tested", "reason": "patient_level_T3_requires_prespecified_distance_aggregation_and_permutation_design; visit_V_is_not_T3",
                               "disconnected_distance_policy": "undefined_requires_review; infinite_distances_not_correlated"}]).to_csv(directory / "02_t3_partial_association.csv", index=False)
                roles = review_qc.feature_roles(scenario["features"], source, settings)
                roles.to_csv(directory / "02_feature_role_matrix.csv", index=False)
                roles.loc[roles.candidate_validation_anchor].to_csv(directory / "02_anchor_eligibility.csv", index=False)
                roles.to_csv(directory / "02_circularity_audit.csv", index=False)
                characterization_features = list(dict.fromkeys(scenario["features"] + roles.loc[roles.candidate_validation_anchor, "feature"].tolist()))
                if source is not None:
                    clinical_raw = qc.align_source(state[api.KEYS], source, [f for f in characterization_features if f in source])
                else:
                    clinical_raw = None
                review_qc.clinical_characterization(scenario, clinical_raw, metadata, characterization_features).to_csv(directory / "02_clinical_characterization_by_state.csv", index=False)
                pd.DataFrame([{"status": "not_run", "reason": "independent_anchor_test_plan_and_approval_required; characterization_is_descriptive"}]).to_csv(directory / "02_anchor_validation.csv", index=False)
                pd.DataFrame([{"status": "not_run", "reason": "conditional_information_analysis_requires_prespecified_plan_and_Pop_circularity_review"}]).to_csv(directory / "02_conditional_information_vs_pop.csv", index=False)
                (directory / "02_clinical_interpretation_draft.md").write_text("# Descriptive interpretation only\n\nNo clinical phenotype names or independent validation assigned. "
                    "Patient summaries use the earliest visit per community; patients may belong to multiple communities. "
                    "Variables used in embedding remain defining variables. Independent anchors, branch definition and T3 require investigator review.\n")
                if HSCRP in state:
                    before, after = state[HSCRP], experiment[HSCRP] if HSCRP in experiment else pd.Series(dtype=float)
                    pd.DataFrame([{"transformation": definition["transformation_id"], "n_fit_visits": len(before), "n_changed": int((before != after).sum()) if len(after) else None,
                        "before_median": before.median(), "before_p01": before.quantile(.01), "before_p99": before.quantile(.99),
                        "after_median": after.median(), "after_p01": after.quantile(.01), "after_p99": after.quantile(.99),
                        "fit_bounds": str(definition.get("training_winsor_bounds")), "unit_status": settings.get("methodology_review", {}).get("hscrp_unit_status", "unknown"),
                        "status": "completed" if len(after) else "feature_ablated", "training_scope": "full_cohort_in_sample; bounds_relearned_in_each_bootstrap"}]).to_csv(directory / "02_hscrp_transformation_qc.csv", index=False)
                for stem, table in qc.representation_qc(experiment, scenario, api.feature_family, raw).items():
                    table.to_csv(directory / f"02_{stem}.csv", index=False)
                pca = scenario["pca"]
                concentration = []
                for i, component in enumerate(pca.components_):
                    p = np.square(component) / np.square(component).sum()
                    concentration.append({"component": f"PC{i+1}", "explained_variance_ratio": pca.explained_variance_ratio_[i],
                                          "effective_features": 1 / np.square(p).sum(), "top10_squared_loading_fraction": np.sort(p)[-10:].sum(),
                                          "denominator": "within_component_squared_loadings; not_total_variance"})
                pd.DataFrame(concentration).to_csv(directory / "02_pca_variance_and_concentration.csv", index=False)
                bootstrap_features = [f for f in eligible if f in definition["feature_list"] or definition["isolated_change"] in {"identity", "log1p", "winsor", "patient_mass", "full80", "observed_pruning"}]
                if definition["isolated_change"] == "without_hscrp":
                    bootstrap_features = [f for f in eligible if f != HSCRP]
                elif definition["isolated_change"] == "without_pro":
                    bootstrap_features = [f for f in eligible if not f.startswith("pro__")]
                def preprocess(sampled, chosen):
                    processor = TrainingPreprocessor(definition, settings, api.reduce_redundancy).fit(sampled, chosen)
                    transformed = processor.transform(sampled)
                    if options.get("patient_weighted") and "bootstrap_draw_id" in sampled:
                        transformed["fit_patient_id"] = sampled.bootstrap_draw_id.to_numpy()
                    return transformed, processor.features_
                boot, matching, stability = bootstrap_mapper(raw, scenario, settings, bootstrap_features,
                    api.run_mapper_scenario, api.reduce_redundancy, api.jaccard, preprocessor=preprocess, fit_options=options)
                boot.to_csv(directory / "02_bootstrap_replicate_qc.csv", index=False)
                matching.to_csv(directory / "02_bootstrap_match_detail.csv", index=False)
                pd.DataFrame(stability.get("stability", []), columns=["kind", "primary_structure_id", "n_replicates_eligible", "n_replicates_total",
                    "patient_jaccard_min", "patient_jaccard_median", "patient_jaccard_max", "frequency_jaccard_ge_threshold", "match_threshold", "required_frequency", "stable_by_prespecified_rule"]).to_csv(directory / "02_bootstrap_stability_by_object.csv", index=False)
                api.write_json(directory / "02_bootstrap_summary.json", stability)
                negative = []
                negative_reps = int(review.get("negative_control_replicates", 0))
                if raw is not None and all(f in raw for f in bootstrap_features) and negative_reps:
                    rng = np.random.default_rng(config["random_seed"])
                    for replicate in range(negative_reps):
                        null, null_qc = negative_patient_blocks(raw, bootstrap_features, rng)
                        null_record = {"replicate": replicate, **null_qc}
                        if not null_qc["geometry_changed"]:
                            null_record.update(status="not_testable", reason="insufficient_changed_patient_blocks")
                        else:
                            try:
                                null_state, chosen = preprocess(null, bootstrap_features)
                                null_mapper = api.run_mapper_scenario(null_state, chosen, settings, family_balance=True, scenario_name=f"{name}-null-{replicate}", **options)
                                null_settings = copy.deepcopy(settings)
                                null_settings["mapper_validation"]["bootstrap_replicates"] = int(review.get("negative_control_bootstrap_replicates", 200))
                                _, _, null_stability = bootstrap_mapper(null, null_mapper, null_settings, bootstrap_features, api.run_mapper_scenario,
                                    api.reduce_redundancy, api.jaccard, preprocessor=preprocess, fit_options=options)
                                null_record.update(status="completed", n_nodes=null_mapper["summary"]["n_nodes"], n_macrostates=null_mapper["summary"]["n_macrostates"],
                                    n_stable_macrostates=sum(row["stable_by_prespecified_rule"] for row in null_stability.get("stability", []) if row["kind"] == "macrostate"),
                                    stability_status=null_stability["status"], branches="not_defined_H1_not_tested")
                            except (ValueError, np.linalg.LinAlgError) as error:
                                null_record.update(status="failed", reason=str(error))
                        negative.append(null_record)
                else:
                    negative.append({"status": "not_run", "reason": "disabled_by_config" if not negative_reps else "missing_original_feature_source"})
                pd.DataFrame(negative).to_csv(directory / "02_negative_control_results.csv", index=False)
                readiness = review_qc.decision_gates(reference, graph_qc, availability, stability, run_id, admin)
                readiness.update(scenario=name, schema_version="longitudinal_mapper_v2", branch_status="not_defined_requires_researcher_definition")
                api.write_json(directory / "02_temporal_readiness.json", readiness)
                scenario_summary = {**scenario["summary"], "schema_version": "longitudinal_mapper_v2", "source_run_id": run_id, "role": "candidate", "readiness": readiness,
                                    "reference_manifest_sha256": reference["reference_manifest_sha256"], "bootstrap": stability}
                api.write_json(directory / "02_longitudinal_mapper_summary.json", scenario_summary)
                summaries[name] = scenario_summary
                rows.append({**record, **scenario["summary"], **comparison, "status": "completed", "branch_status": "not_defined", "n_branches": None,
                             "n_intervals_total": len(endpoint), "n_intervals_graph": int(endpoint.both_endpoints_graph_covered.sum()),
                             "n_intervals_supported": int(endpoint.both_endpoints_supported_covered.sum()), "n_intervals_hard": int(endpoint.both_endpoints_hard_assigned.sum()),
                             "bootstrap_status": stability["status"], "n_stable_macrostates": sum(row["stable_by_prespecified_rule"] for row in stability.get("stability", []) if row["kind"] == "macrostate"),
                             "bootstrap_replicates": stability["replicates_requested"],
                             "stability_evidence_status": "prespecified_budget" if stability["replicates_requested"] >= 200 else "exploratory_reduced_budget",
                             "administrative_status": "inconclusive_T3_not_tested", "OOS_status": "not_run"})
                definition.update(status="completed", preprocessing_full_fit="historical_imputed_state_fixed; isolated_change_only",
                                  preprocessing_bootstrap="raw_patient_sample_only; all_fit_stages_relearned")
        pd.DataFrame(rows).to_csv(staged / "tables/02_scenario_comparison.csv", index=False)
        pd.DataFrame(comparison_frames or [{"status": "not_run_reference_only", "reason": "no_candidates_requested"}]).to_csv(staged / "tables/02_scenario_comparison_universe_summary.csv", index=False)
        if crosswalk_frames:
            crosswalk = pd.concat(crosswalk_frames, ignore_index=True)
            crosswalk.to_csv(staged / "tables/02_scenario_pairwise_crosswalk.csv", index=False)
            crosswalk.loc[crosswalk.patient_jaccard.gt(0)].to_csv(staged / "tables/02_scenario_split_merge.csv", index=False)
            pd.concat(universe_frames, ignore_index=True).to_csv(staged / "analytic/02_scenario_universe_audit.csv", index=False)
        api.write_json(staged / "scenario_registry.json", definitions)
        pd.DataFrame(definitions or [{"status": "reference_only", "reason": "historical_run_not_refitted"}]).to_csv(staged / "tables/02_scenario_registry_resolved.csv", index=False)
        (staged / "configuration_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
        manifest["resolved_configuration_sha256"] = sha256(staged / "configuration_resolved.yaml")
        manifest["reference_integrity_after"] = verify_reference(seal_path)["reference_manifest_sha256"]
        readiness = {"status": "blocked", "reason": "No approved branch definition, adapted T3, independent clinical validation or researcher selection",
                     "selected_scenario": None, "approval_status": "not_recorded", "temporal_flow_executed": False,
                     "run_id": run_id, "reference_manifest_sha256": reference["reference_manifest_sha256"]}
        api.write_json(staged / "02_temporal_readiness.json", readiness)
        (staged / "tables/02_scenario_selection_review.md").write_text(
            "# Scenario selection review\n\nNo scenario selected. All new representations are candidates.\n\n"
            "S3-primary was read from the frozen reference; its labels, outputs and configuration were not refitted or rewritten.\n\n"
            "Review coverage, squared-loading concentration, individual macrostate stability, administrative availability and circularity together. "
            "ARI excludes uncovered and tied visits; one-label comparisons have no interpretable ARI/NMI. "
            "Communities and components are not defined branches. Bootstrap is in-sample and provides no OOS evidence.\n\n"
            "T3 adaptation, primary estimand, clinical feature whitelists, independent anchors and the formal branch rule require investigator review. "
            "H1 is not established. A continuous or negative finding remains valid. Script 03 inferential remains blocked.\n")
        manifest["artifact_sha256"] = {str(p.relative_to(staged)): sha256(p) for p in staged.rglob("*") if p.is_file()}
        api.write_json(staged / "run_manifest.json", manifest)
    print(json.dumps({"status": "completed", "run_id": run_id, "output": str(target), "gate": "blocked", "selected_scenario": None}))
    return {"status": "completed", "run_id": run_id, "output": str(target), "reference_integrity": "passed", "gate": "blocked", "scenarios": rows}

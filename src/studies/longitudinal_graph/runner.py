"""Staged, provenance-backed orchestration for Script 02."""
from __future__ import annotations
import copy
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.studies.longitudinal_graph import validation as qc
from src.studies.longitudinal_graph.stability import bootstrap_mapper
from src.studies.longitudinal_graph.figures import mapper_figures, diagnostic_figures


def registry(features, config):
    """Predeclared experiments; empty clinical whitelists remain explicitly unavailable."""
    v=config.get("mapper_validation",{}); rows=[]
    def add(name,selected=None,**kwargs):
        rows.append({"name":name,"feature_list":features if selected is None else selected,"transformation":"identity",
                     "weight_rule":"equal_visit_mass","pca_rule":"historical_capped","mapper_rule":"global_eps_minmax",
                     "family_balance":True,"reason":"prespecified_sensitivity",**kwargs})
    add("S3-primary",reason="historical_reference")
    add("S2-no-family-balance",family_balance=False)
    add("S3-full80",pca_rule="uncapped_target")
    for feature,name in (("lab__urine_squamous_cells__value","S3-without-urine-squamous"),("lab__crp_high_sensitivity__value","S3-without-hsCRP")):
        add(name,[f for f in features if f!=feature],omit_reason=None if feature in features else "feature_not_retained_after_redundancy")
    add("S3-hsCRP-log1p",transformation="log1p_hsCRP")
    pro=[f for f in features if f.startswith("pro__")]
    add("S3-no-PRO",[f for f in features if f not in pro],omit_reason=None if pro else "no_PRO_features_retained")
    summaries=v.get("pro_summary_features",[])
    add("S3-PRO-summary-only",[f for f in features if f not in pro or f in summaries],
        omit_reason=None if summaries and all(f in features for f in summaries) else "clinical_summary_whitelist_not_defined_or_unavailable")
    add("S3-patient-weighted",weight_rule="equal_patient_mass")
    common=v.get("comparable_visit_features",[])
    add("S3-comparable-visit-features",[f for f in features if f in common],
        omit_reason=None if common and all(f in features for f in common) else "comparable_feature_whitelist_not_defined_or_unavailable")
    add("S3-local-eps",mapper_rule="local_eps_minmax")
    for item in v.get("parameter_grid",[]):
        add(item["name"],mapper_overrides={k:value for k,value in item.items() if k!="name"})
    robust=v.get("robust_cover_quantiles")
    add("S3-robust-cover",mapper_rule="global_eps_robust_range",cover_quantiles=robust,
        omit_reason=None if robust else "robust_cover_not_prespecified")
    return rows


def dry_run(args, api):
    cfg=api.load_config(args.config); missing=[str(p) for p in (args.state,args.metadata) if not p.exists()]
    if missing:return {"status":"not_run_missing_data","dry_run":True,"missing_inputs":missing,"outputs_written":False}
    state=pd.read_parquet(args.state); metadata=pd.read_parquet(args.metadata)
    validate_inputs(state,metadata,cfg,api)
    features,audit=api.reduce_redundancy(state.drop(columns=api.KEYS),cfg["representation"]["redundancy_spearman_abs_threshold"])
    return {"status":"validated","dry_run":True,"n_visits":len(state),"n_patients":state.patient_id.nunique(),
            "n_features":len(features),"scenarios_planned":registry(features,cfg),"outputs_written":False,
            "representation_stability_gate":{"status":"requires_review","temporal_flow_executed":False}}


def validate_inputs(state,metadata,cfg,api):
    qc.validate_keys(state,"state");qc.validate_keys(metadata,"metadata")
    qc.align_source(state[api.KEYS],metadata,[])
    if len(state)!=len(metadata):raise ValueError("Metadata cohort differs from state")
    # Script 01 is authoritative; reject accidental administrative/outcome additions to its state.
    import importlib
    prep=importlib.import_module("src.studies.longitudinal_graph.01_prepare_longitudinal_data")
    for feature in state.drop(columns=api.KEYS):
        reason=prep.forbidden_reason(feature,set(),cfg["feature_selection"]["forbidden_name_patterns"])
        if reason:raise ValueError(f"Ineligible state feature {feature}: {reason}")
    if not np.isfinite(state.drop(columns=api.KEYS).to_numpy(dtype=float)).all():raise ValueError("State must contain finite imputed clinical values")


def gate(primary,scenarios,comparisons,intervals,bootstrap,availability,raw_source):
    blockers=[];pending=[]
    if primary["summary"]["n_supported_macrostates"]<2:blockers.append("primary_has_fewer_than_two_supported_macrostates")
    if not primary["summary"]["n_supported_macrostate_covered_visits"]:blockers.append("no_supported_coverage")
    if intervals.get("n_intervals_both_endpoints_supported_macrostate_covered")==0:blockers.append("no_supported_mapped_intervals")
    if intervals["status"]!="completed":pending.append("consecutive_interval_coverage_not_tested")
    if not scenarios["S3-full80"]["summary"]["pca_variance_target_reached"]:blockers.append("full80_variance_target_not_reached")
    for name,s in scenarios.items():
        if name!="S3-primary" and s["summary"]["n_supported_macrostates"]<2:pending.append(f"{name}:fewer_than_two_supported_macrostates")
    if raw_source=="source_unavailable":pending.append("preimputation_observability_unavailable")
    pending.extend(availability.loc[availability.status.ne("completed"),"variable"].map(lambda x:f"admin:{x}:not_tested").tolist())
    if bootstrap["status"]!="completed":pending.append("patient_bootstrap_not_tested")
    # Communities and connected components do not establish independently defined branches.
    blockers.append("no_stable_branch_evidence:branch_definition_requires_researcher_review")
    return {"status":"blocked" if blockers else "requires_review","blockers":blockers,"pending_checks":pending,
            "temporal_flow_executed":False,"approval_policy":"reviewed_approved_requires_documented_researcher_decision_date_SHA_and_justification",
            "components":{"cohort_keys":"validated","representation":"requires_review","coverage":intervals,
                          "representation_sensitivities":comparisons,"patient_bootstrap":bootstrap,
                          "administrative_availability":availability.to_dict("records"),"circularity":"requires_proxy_review",
                          "membership_contract":"validated_equal_weight_per_supported_node"}}


def execute(args,api):
    if args.dry_run:
        result=dry_run(args,api);print(api.json.dumps(result,default=str));return result
    cfg=api.load_config(args.config);state=pd.read_parquet(args.state).reset_index(drop=True);metadata=pd.read_parquet(args.metadata)
    validate_inputs(state,metadata,cfg,api)
    eligible=list(state.drop(columns=api.KEYS));features,audit=api.reduce_redundancy(state[eligible],cfg["representation"]["redundancy_spearman_abs_threshold"])
    input_files=[args.state,args.metadata];raw_source="source_unavailable";raw=state[api.KEYS].copy();source=metadata
    if args.integrated.exists():
        source=pd.read_parquet(args.integrated);raw=qc.align_source(state[api.KEYS],source,[f for f in eligible if f in source]);raw_source="preimputation_integrated";input_files.append(args.integrated)
    else:
        raw_path=args.state.with_name("01_longitudinal_visit_raw.parquet")
        if raw_path.exists():
            source=pd.read_parquet(raw_path);raw=qc.align_source(state[api.KEYS],source,[f for f in eligible if f in source]);raw_source="preimputation_script01";input_files.append(raw_path)
    if args.intervals.exists():input_files.append(args.intervals)
    manifest=qc.run_manifest(api.ROOT,args.config,input_files,cfg["random_seed"],"S3-primary",api.sha256_file)
    manifest["checks"]["input_keys_and_finite_state"]="passed"
    manifest["real_data_status"]="input_supplied_not_independently_classified"
    fm_path=args.state.with_name("01_longitudinal_feature_manifest.csv")
    default=qc.study_paths(api.common)["tables"]/"01_longitudinal_feature_manifest.csv"
    fm_path=getattr(args,"feature_manifest",None) or (fm_path if fm_path.exists() else default)
    feature_manifest=pd.read_csv(fm_path) if fm_path.exists() else pd.DataFrame()
    if fm_path.exists():manifest["input_sha256"][str(fm_path)]=api.sha256_file(fm_path)
    diagnostics=api.feature_distribution_diagnostics(raw,source=raw_source)
    definitions=registry(features,cfg);scenarios={};skipped={};failed={};scenario_states={}
    for definition in definitions:
        name=definition["name"];selected=definition["feature_list"];reason=definition.get("omit_reason")
        if len(selected)<2:reason=reason or "fewer_than_two_features"
        if reason:skipped[name]=reason;definition.update(status="not_applicable",reason=reason);continue
        experiment=state.copy();settings=copy.deepcopy(cfg)
        if definition["transformation"]=="log1p_hsCRP":
            prior=cfg.get("mapper_validation",{}).get("hscrp_prior_transformation","unknown")
            experiment,reason=api.hscrp_log_state(state,features,diagnostics,prior_transform=prior)
            if reason:skipped[name]=reason;definition.update(status="not_applicable",reason=reason);continue
        if definition["pca_rule"]=="uncapped_target":settings["pca"]["max_components"]=None
        if definition["mapper_rule"]=="local_eps_minmax":settings["mapper"]["eps_mode"]="local"
        if definition["mapper_rule"]=="global_eps_robust_range":settings["mapper"]["cover_quantiles"]=definition["cover_quantiles"]
        settings["mapper"].update(definition.get("mapper_overrides",{}))
        try:
            scenario=api.run_mapper_scenario(experiment,selected,settings,family_balance=definition["family_balance"],scenario_name=name,
                                             patient_weighted=definition["weight_rule"]=="equal_patient_mass")
        except (ValueError,np.linalg.LinAlgError) as exc:
            if name in {"S3-primary","S3-full80","S2-no-family-balance"}:raise
            failed[name]=f"{type(exc).__name__}:{exc}";definition.update(status="failed",reason=failed[name]);continue
        scenarios[name]=scenario;scenario_states[name]=experiment;definition["status"]="completed"
        definition["mapper_parameters"]=settings["mapper"]
    primary=scenarios["S3-primary"];comparisons={};sensitivity_rows=[];matches={}
    manifest["checks"]["scenarios"]={d["name"]:{"status":d["status"],"reason":d["reason"]} for d in definitions}
    for name in [d["name"] for d in definitions]:
        if name in scenarios:
            matching,comparison=api.compare_mapper_scenarios(primary,scenarios[name]);matches[name]=matching
            sensitivity_rows.append({**scenarios[name]["summary"],"status":"completed","n_ties":scenarios[name]["summary"]["n_membership_ties"],
                                     "ari_vs_primary":comparison["ari"],"patient_jaccard_vs_primary":comparison["patient_jaccard_vs_primary"],
                                     "visit_jaccard_vs_primary":comparison["visit_jaccard_vs_primary"],"n_visits_compared":comparison["n_visits_compared"],
                                     "n_patients_compared":comparison["n_patients_compared"],"ari_status":comparison["ari_status"]})
        else:
            status="failed" if name in failed else "not_applicable";reason=failed.get(name,skipped.get(name))
            comparison={"status":status,"reason":reason,"ari":None};sensitivity_rows.append({"scenario":name,**comparison})
            matches[name]=pd.DataFrame(columns=["s3_macrostate_id","matched_s2_macrostate_id","s3_n_visits","s2_n_visits","s3_n_patients","s2_n_patients","visit_jaccard","patient_jaccard"])
        comparisons[name]=comparison
    detail=qc.coverage_detail(primary["visit_membership"],primary)
    admin,availability,admin_strata,admin_lens=qc.administrative_audit(primary["visit_membership"],metadata,primary["embedding"],cfg)
    legacy_admin,visit_qc=api.administrative_qc(primary["visit_membership"],metadata,flag_threshold=cfg["administrative_qc"]["cramers_v_flag_threshold"])
    intervals=api.interval_coverage_summary(pd.read_parquet(args.intervals),primary["visit_membership"],cfg["temporal"]["maximum_interval_years"]) if args.intervals.exists() else {"status":"not_applicable","reason":"script01_intervals_unavailable"}
    endpoint=qc.interval_endpoint_qc(pd.read_parquet(args.intervals),primary["visit_membership"]) if args.intervals.exists() else pd.DataFrame(columns=["status","reason"])
    # Validate representation diagnostics before expensive patient refits.
    diagnostics_by_name={}
    for name,scenario in scenarios.items():
        for stem,table in qc.representation_qc(scenario_states[name],scenario,api.feature_family,raw).items():diagnostics_by_name.setdefault(stem,[]).append(table)
    boot,boot_matching,boot_summary=bootstrap_mapper(raw if raw_source!="source_unavailable" else None,primary,cfg,eligible,
                                                   api.run_mapper_scenario,api.reduce_redundancy,api.jaccard)
    manifest["checks"]["patient_bootstrap"]=boot_summary["status"]
    gate_result=gate(primary,scenarios,comparisons,intervals,boot_summary,availability,raw_source)
    if failed:gate_result["blockers"].extend(f"scenario_failed:{name}:{reason}" for name,reason in failed.items())
    if admin.administrative_confounding_flag.any():gate_result["pending_checks"].append("administrative_association_above_configured_threshold")
    tables={"feature_distribution_diagnostics":diagnostics,"mapper_sensitivity_summary":pd.DataFrame(sensitivity_rows),
            "mapper_pc_feature_diagnostics":api.pc_feature_diagnostics(scenarios),"mapper_coverage_detail":detail,
            "mapper_coverage_by_stratum":qc.coverage_by_stratum(detail,metadata),"mapper_interval_endpoint_qc":endpoint,
            "mapper_admin_association":admin,"mapper_admin_availability":availability,"mapper_admin_stratified_coverage":admin_strata,
            "mapper_admin_lens_qc":admin_lens,"mapper_administrative_qc":legacy_admin,"mapper_visit_type_qc":visit_qc,
            "longitudinal_redundancy_audit":qc.raw_redundancy_audit(audit,raw),
            "feature_observability_by_stratum":qc.observability_audit(raw,metadata,eligible),
            "pro_redundancy_audit":qc.pro_pair_audit(raw,state,eligible),
            "sjd_biomarker_eligibility_audit":qc.biomarker_audit(source,feature_manifest,features),
            "mapper_circularity_audit":qc.circularity_audit(feature_manifest,features,source),
            "mapper_bootstrap_replicates":boot,"mapper_macrostate_stability":boot_matching.loc[boot_matching.structure_kind.eq("macrostate")],
            "mapper_branch_stability":boot_matching.loc[boot_matching.structure_kind.eq("component")].assign(branch_interpretation="connected_component_QC_only_not_defined_branches")}
    for stem,frames in diagnostics_by_name.items():tables[stem]=pd.concat(frames,ignore_index=True)
    cover=pd.concat([s["cover_qc"].assign(scenario=n) for n,s in scenarios.items()],ignore_index=True)
    tables["lens_cube_occupancy"]=cover;tables["mapper_cover_qc"]=cover
    tables["mapper_noise_coverage"]=pd.concat([s["noise_qc"].assign(scenario=n) for n,s in scenarios.items()],ignore_index=True)
    rep=[]
    for definition in definitions:
        for feature in eligible:
            used=feature in definition["feature_list"] and definition["status"]=="completed"
            rep.append({"scenario":definition["name"],"feature":feature,"family":api.feature_family(feature),
                        "eligible_from_script01":True,"retained_after_redundancy":feature in features,
                        "used_in_primary_mapper":used and definition["name"]=="S3-primary","used_in_scenario":used,
                        "decision_reason":definition["reason"] if used else definition.get("omit_reason") or "scenario_feature_exclusion_or_pruning",
                        "transformation":definition["transformation"],"weight_rule":definition["weight_rule"],
                        "balance_weight":1/np.sqrt(sum(api.feature_family(f)==api.feature_family(feature) for f in definition["feature_list"])) if used and definition["family_balance"] else 1. if used else None})
    tables["mapper_representation_manifest"]=pd.DataFrame(rep)
    membership=primary["visit_membership"];pca=primary["pca"];full80=scenarios["S3-full80"]["summary"];s2=comparisons["S2-no-family-balance"]
    for suffix,key in (("nodes","node_table"),("macrostate_summary","macrostate_table"),("visit_membership","visit_membership")):
        tables[f"mapper_{suffix}"]=primary[key]
        for prefix,name in (("s3","S3-primary"),("s2","S2-no-family-balance")):tables[f"{prefix}_mapper_{suffix}"]=scenarios[name][key]
    tables["mapper_topological_edges"]=primary["topological_edges"]
    tables["longitudinal_pca_variance"]=pd.DataFrame({"component":np.arange(1,pca.n_components_+1),"explained_variance_ratio":pca.explained_variance_ratio_,"cumulative_variance":np.cumsum(pca.explained_variance_ratio_)})
    tables["longitudinal_pca_loadings"]=pd.DataFrame(pca.components_.T,index=features,columns=[f"PC{i+1}" for i in range(pca.n_components_)]).rename_axis("feature").reset_index()
    aliases={"S2-no-family-balance":"s2_s3","S3-full80":"full80","S3-without-urine-squamous":"no_urine_squamous","S3-hsCRP-log1p":"hscrp_log1p"}
    for name,table in matches.items():
        stem=aliases.get(name,name.lower().replace("-","_"))
        tables[f"{stem}_macrostate_stability"]=table
    summary={**primary["summary"],"status":"completed","contract_version":api.STUDY_CONTRACT_VERSION,"run_manifest":manifest,
             "input_files":input_files,"input_sha256":manifest["input_sha256"],"config_file":args.config,"config_sha256":manifest["config_sha256"],
             "random_seed":cfg["random_seed"],"primary_scenario":"S3-like","sensitivity_scenario":"S2-like","scenario_registry":definitions,
             "n_features_before_redundancy":len(eligible),"n_features_after_redundancy":len(features),"n_pca_components":primary["summary"]["n_pcs"],
             "pca_cumulative_variance":primary["summary"]["cumulative_variance"],"n_pcs_primary":primary["summary"]["n_pcs"],"variance_primary":primary["summary"]["cumulative_variance"],
             "n_pcs_full80":full80["n_pcs"],"variance_full80":full80["cumulative_variance"],"mapper_parameters":cfg["mapper"],
             "s2_s3_ari":s2["ari"],"s2_s3_patient_jaccard_summary":s2["patient_jaccard_summary"],"full80_ari_vs_primary":comparisons["S3-full80"]["ari"],
             "no_urine_squamous_ari_vs_primary":comparisons["S3-without-urine-squamous"]["ari"],"hscrp_log_ari_vs_primary":comparisons["S3-hsCRP-log1p"]["ari"],
             "sensitivity_comparisons":comparisons,"interval_coverage":intervals,"bootstrap":boot_summary,
             "administrative_confounding_flag":bool(admin.administrative_confounding_flag.any()),
             "administrative_qc_interpretation":"descriptive_V_plus_patient_vector_permutation; missing_or_no_flag_does_not_establish_no_confounding",
             "coverage_definitions":{"mapper_covered_summary":"at_least_one_graph_node","mapper_covered_membership_column":"at_least_one_supported_macrostate; unchanged_for_Script03",
                                     "covered_without_supported_hard_assignment":"graph_covered_and_neither_hard_nor_tied",
                                     "unsupported_macrostate_counts":"any_membership; may_overlap_supported_membership",
                                     "patient_exclusive_category_priority":"hard_assigned,tie,unsupported_macrostate,unsupported_node,uncovered",
                                     "pct_denominators":"all_input_visits_or_unique_patients; percentages_0_to_100"},
             "representation_stability_gate":{**gate_result,"unavailable_checks":skipped,"raw_feature_diagnostics_status":raw_source,"interval_coverage_status":intervals["status"]}}
    root=getattr(args,"output_root",None)
    destination={kind:Path(root)/kind for kind in ("analytic","tables","figures","qc","logs")} if root else api.create_study_dirs("longitudinal_graph")
    for p in destination.values():p.mkdir(parents=True,exist_ok=True)
    # All computations finish before replacing any historical bundle.
    with tempfile.TemporaryDirectory(prefix="longitudinal_mapper_") as temp:
        staged={kind:Path(temp)/kind for kind in destination}
        for p in staged.values():p.mkdir()
        for stem,table in tables.items():table.to_csv(staged["tables"]/f"02_{stem}.csv",index=False)
        api.write_json(staged["tables"]/"02_s2_s3_stability_summary.json",s2)
        api.write_json(staged["tables"]/"02_mapper_bootstrap_summary.json",boot_summary)
        records=[]
        for node,members in primary["graph"]["nodes"].items():
            for i in sorted(members):records.append({**state.iloc[i][api.KEYS].to_dict(),"node_id":node,"node_supported":bool(primary["node_table"].set_index("node_id").loc[node,"supported"]),"macrostate_id":primary["node_to_macrostate"].get(node,pd.NA)})
        pd.DataFrame(records,columns=api.KEYS+["node_id","node_supported","macrostate_id"]).to_parquet(staged["analytic"]/"02_mapper_node_membership.parquet",index=False)
        membership.to_parquet(staged["analytic"]/"02_visit_mapper_membership.parquet",index=False)
        for prefix in ("node_weight__","macrostate_weight__"):
            weights=membership.filter(like=prefix).fillna(0).sum(axis=1)
            if not np.allclose(weights[membership.mapper_covered],1):raise AssertionError("Membership contract failed")
        manifest["checks"]["membership_contract"]="passed"
        mapper_figures(primary,staged["figures"],cfg["figures"]["dpi"])
        diagnostic_figures(primary,tables,comparisons,boot,staged["figures"],cfg["figures"]["dpi"])
        if (staged["figures"]/"02_reference_visit_mapper.png").read_bytes()==(staged["figures"]/"02_reference_visit_mapper_macrostates.png").read_bytes():raise AssertionError("Topology and macrostate figures must differ")
        manifest["checks"]["separate_figures"]="passed"
        api.write_json(staged["tables"]/"02_longitudinal_mapper_summary.json",summary)
        qc.archive_and_publish(staged,destination,manifest,api.write_json)
        # Persist the archive location in the summary as well as the standalone manifest.
        api.write_json(destination["tables"]/"02_longitudinal_mapper_summary.json",summary)
    print("Review representation QC, sensitivity comparability, patient refits and the scientific gate before Script 03.")
    return summary

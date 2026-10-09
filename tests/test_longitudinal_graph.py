"""Scientific invariants for visit-level longitudinal Mapper Model A."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).parents[1]
def load(name):
    path=ROOT/"src/studies/longitudinal_graph"/name
    spec=importlib.util.spec_from_file_location(name.replace(".py",""),path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
PREP=load("01_prepare_longitudinal_data.py"); MAP=load("02_run_longitudinal_mapper.py"); FLOW=load("03_analyze_temporal_flow.py")

def master():
    rows=[]
    for patient,n in (("A",3),("B",2),("C",1)):
      for i in range(n): rows.append({"patient_id":patient,"clinical_episode_id":f"{patient}{i+1}","clinical_anchor_date":pd.Timestamp("2020-01-01")+pd.Timedelta(days=i+1),"clinical_visit_number":i+1})
    return pd.DataFrame(rows)

def membership():
    return pd.DataFrame({"patient_id":["A","A","A"],"clinical_episode_id":["A1","A2","A3"],"mapper_covered":[True,False,True],"hard_macrostate":[0,pd.NA,1],"macrostate_membership_tie":[False,False,False],"node_weight__n0":[1.,np.nan,0.],"node_weight__n1":[0.,np.nan,1.],"macrostate_weight__0":[1.,np.nan,0.],"macrostate_weight__1":[0.,np.nan,1.]})

def test_consecutive_contract_has_no_skips():
    intervals=PREP.build_consecutive_intervals(master()); assert len(intervals)==3
    assert not ((intervals.from_clinical_episode_id=="A1")&(intervals.to_clinical_episode_id=="A3")).any()

def test_future_and_time_leakage_are_explicitly_excluded():
    external={"essdai__total","esspri__total","pop__status"}; patterns=["next_","previous_","delta_","future","time_to_","patient_consensus"]
    for value in ("next_x","delta_x","future_x"): assert PREP.forbidden_reason(value,external,patterns)=="future_leakage"
    assert PREP.forbidden_reason("previous_x",external,patterns)=="historical_accumulator"
    for value in ("clinical_anchor_date","clinical_visit_number","interval_days","interval_years","time_since_x","age_at_visit"): assert PREP.forbidden_reason(value,external,patterns)=="administrative"
    for value in external: assert PREP.forbidden_reason(value,external,patterns)=="external_characterizer"

def test_state_semantic_exclusions_and_scope_fail_closed():
    patterns=["next_","previous_","delta_","future","time_to_"]
    assert PREP.forbidden_reason("essdai__articular_active",set(),patterns)=="external_characterizer"
    assert PREP.forbidden_reason("esspri__fatigue",set(),patterns)=="external_characterizer"
    assert PREP.forbidden_reason("sero__anti_ro_ssa__ever_positive_through_episode",set(),patterns)=="historical_accumulator"
    assert PREP.forbidden_reason("sero__baseline_rf",set(),patterns)=="baseline_static_copy"
    assert PREP.temporal_scope("mystery_measure")[0]=="unknown"
    registry=pd.DataFrame({"public_variable":["sero__baseline_rf"],"temporal_scope":["clinical_baseline"]})
    assert PREP.temporal_scope("sero__baseline_rf",registry)[0]=="baseline_static"

def test_preimputation_and_visit_type_coverage():
    raw=pd.DataFrame({"a":[1.,np.nan],"b":[np.nan,np.nan]})
    qc=PREP.preimputation_coverage(raw)
    assert qc.state_coverage_fraction.tolist()==[.5,0.]
    visits=pd.DataFrame({"patient_id":["A","B"],"visit_type":["A","B"],"feature":[1.,np.nan]})
    detail,_=PREP.visit_type_coverage(visits,["feature"],"visit_type")
    assert detail.set_index("visit_type").pct_observed.to_dict()=={"A":1.,"B":0.}

def test_mapper_administrative_qc_is_populated():
    membership=pd.DataFrame({"patient_id":["A","B","C","D"],"clinical_episode_id":["1","2","3","4"],"hard_macrostate":[0,0,1,1]})
    metadata=pd.DataFrame({"patient_id":["A","B","C","D"],"clinical_episode_id":["1","2","3","4"],"visit_type":["X","X","Y","Y"]})
    assoc,detail=MAP.administrative_qc(membership,metadata)
    assert not assoc.empty and not detail.empty and assoc.iloc[0].n_patients==4

def test_unique_patient_node_support_not_visit_support():
    visits=pd.DataFrame({"patient_id":["A"]*10}); graph={"nodes":{"n":set(range(10))},"links":{"n":[]}}
    row=MAP.node_support_table(graph,visits,3).iloc[0]; assert row.n_member_visits==10 and row.n_unique_patients==1 and not row.supported

def test_normalization_and_hard_tie():
    graph={"nodes":{"a":{0},"b":{0}},"links":{"a":["b"],"b":["a"]}}; visits=pd.DataFrame({"patient_id":["p"],"clinical_episode_id":["v"]}); summary=pd.DataFrame({"macrostate_id":[0,1],"macrostate_supported":[True,True]})
    out=MAP.normalized_memberships(graph,visits,{"a":0,"b":1},summary); assert np.isclose(out.filter(like="macrostate_weight__").sum(axis=1).iloc[0],1)
    assert out.macrostate_membership_tie.iloc[0] and pd.isna(out.hard_macrostate.iloc[0])

def test_temporal_mass_conservation_and_no_artificial_skip():
    intervals=PREP.build_consecutive_intervals(master().query("patient_id == 'A'")); attached=FLOW.attach_interval_memberships(intervals,membership()); assert not attached.flow_eligible.any()
    m=membership().copy(); m.loc[1,["mapper_covered","hard_macrostate","node_weight__n0","node_weight__n1","macrostate_weight__0","macrostate_weight__1"]]=[True,0,.5,.5,.5,.5]
    attached=FLOW.attach_interval_memberships(intervals,m); flow=FLOW.compute_temporal_flow(attached); assert np.isclose(flow.weighted_interval_mass.sum(),2)

def test_patient_bootstrap_and_reversal_are_clustered_and_reproducible():
    intervals=PREP.build_consecutive_intervals(master().query("patient_id == 'A'")); m=membership(); m.loc[1,["mapper_covered","hard_macrostate","node_weight__n0","node_weight__n1","macrostate_weight__0","macrostate_weight__1"]]=[True,0,1.,0.,1.,0.]
    attached=FLOW.attach_interval_memberships(intervals,m); boot=FLOW.patient_cluster_bootstrap(attached,5,7); assert (boot.n_patient_draws==1).all()
    a=FLOW.temporal_reversal_null(attached,5,9); b=FLOW.temporal_reversal_null(attached,5,9); pd.testing.assert_frame_equal(a,b)


def synthetic_mapper_state(n=120, n_features=18):
    rng=np.random.default_rng(420)
    values=rng.normal(size=(n,n_features))
    state=pd.DataFrame(values,columns=[f"lab__synthetic_{i}__value" for i in range(n_features)])
    state.insert(0,"clinical_episode_id",[f"v{i}" for i in range(n)])
    state.insert(0,"patient_id",[f"p{i//2}" for i in range(n)])
    return state


def mapper_config():
    return MAP.load_config(ROOT/"src/studies/longitudinal_graph/config.yaml")


def test_s2_runs_full_mapper_and_primary_embedding_is_unchanged():
    state=synthetic_mapper_state(); features=list(state.columns[2:]); cfg=mapper_config()
    s2=MAP.run_mapper_scenario(state,features,cfg,family_balance=False,scenario_name="S2-no-family-balance")
    assert s2["graph"]["nodes"] and not s2["node_table"].empty
    assert not s2["macrostate_table"].empty
    assert len(s2["visit_membership"])==len(state)
    assert not s2["topological_edges"].empty
    s3=MAP.run_mapper_scenario(state,features,cfg,family_balance=True,scenario_name="S3-primary")
    embedding,pca=MAP.fit_visit_embedding(state[features],cfg,[1/np.sqrt(len(features))]*len(features))
    np.testing.assert_allclose(s3["embedding"],embedding)
    np.testing.assert_allclose(s3["pca"].components_,pca.components_)
    graph=MAP.run_mapper(embedding[:,:2],embedding,cfg,s3["eps"])
    assert s3["graph"]==graph


def permutation_scenario(labels):
    visits=pd.DataFrame({"patient_id":["A","B","C","D"],"clinical_episode_id":["1","2","3","4"]})
    ids=sorted(set(labels)); nodes={f"node_{i}":set(np.flatnonzero(np.array(labels)==i)) for i in ids}
    graph={"nodes":nodes,"links":{n:[] for n in nodes}}
    mapping={f"node_{i}":i for i in ids}
    macros=pd.DataFrame({"macrostate_id":ids,"macrostate_supported":[True]*len(ids)})
    memberships=MAP.normalized_memberships(graph,visits,mapping,macros)
    return {"graph":graph,"node_to_macrostate":mapping,"macrostate_table":macros,"visit_membership":memberships,
            "summary":{"n_visits":4,"n_nodes":len(ids),"n_supported_nodes":len(ids),"n_macrostates":len(ids),"n_supported_macrostates":len(ids)}}


def test_s2_s3_ari_with_permuted_ids():
    _,summary=MAP.compare_mapper_scenarios(permutation_scenario([1,1,0,0]),permutation_scenario([0,0,1,1]))
    assert summary["ari"]==1 and summary["n_visits_compared"]==4 and summary["n_patients_compared"]==4


def test_macrostate_matching_ignores_numeric_ids():
    matching,summary=MAP.compare_mapper_scenarios(permutation_scenario([1,1,0,0]),permutation_scenario([0,0,1,1]))
    assert matching.set_index("s3_macrostate_id").matched_s2_macrostate_id.to_dict()=={0:1,1:0}
    assert (matching.patient_jaccard==1).all() and (matching.visit_jaccard==1).all()
    assert summary["patient_jaccard_summary"]["min"]==1


def test_comparison_excludes_ties_uncovered_and_uses_visit_keys():
    primary=permutation_scenario([0,0,1,1]); other=permutation_scenario([1,1,0,0])
    other["visit_membership"].loc[0,"macrostate_membership_tie"]=True
    other["visit_membership"].loc[0,"hard_macrostate"]=pd.NA
    other["visit_membership"]=other["visit_membership"].iloc[::-1].reset_index(drop=True)
    _,summary=MAP.compare_mapper_scenarios(primary,other)
    assert summary["ari"]==1 and summary["n_visits_compared"]==3
    other["visit_membership"]["hard_macrostate"]=pd.NA
    _,summary=MAP.compare_mapper_scenarios(primary,other)
    assert summary["ari"] is None and summary["ari_status"]=="insufficient_comparable_visits"


def test_coverage_accounting_includes_unsupported_nodes_and_macrostates():
    visits=pd.DataFrame({"patient_id":list("ABCDE"),"clinical_episode_id":list("12345")})
    graph={"nodes":{"a":{0,1},"b":{1},"small":{2},"unsupported_node":{3}},
           "links":{"a":["b"],"b":["a"],"small":[],"unsupported_node":[]}}
    macros=pd.DataFrame({"macrostate_id":[0,1,2],"macrostate_supported":[True,True,False]})
    out=MAP.normalized_memberships(graph,visits,{"a":0,"b":1,"small":2},macros)
    summary=MAP.coverage_summary(out)
    assert summary["n_mapper_covered_visits"]+summary["n_uncovered_visits"]==len(visits)
    assert summary["n_hard_assigned_visits"]==1 and summary["n_membership_ties"]==1
    assert summary["n_covered_without_supported_hard_assignment"]==2
    assert sum(summary[k] for k in ("n_hard_assigned_visits","n_membership_ties","n_covered_without_supported_hard_assignment"))==summary["n_mapper_covered_visits"]
    assert summary["pct_mapper_covered_visits"]==80
    assert summary["n_visits_in_unsupported_macrostates"]==1
    assert summary["n_patients_in_unsupported_macrostates"]==1
    # The existing downstream flag still excludes unsupported memberships.
    assert out.mapper_covered.tolist()==[True,True,False,False,False]
    assert out.mapper_graph_covered.tolist()==[True,True,True,True,False]
    assert not out.loc[2:].filter(like="macrostate_weight__").fillna(0).to_numpy().any()


def test_full80_pca_reaches_target_without_artificial_ten_component_cap():
    state=synthetic_mapper_state(n=150,n_features=30); features=list(state.columns[2:]); cfg=mapper_config()
    _,primary=MAP.fit_visit_embedding(state[features],cfg)
    cfg["pca"]["max_components"]=None
    embedding,pca=MAP.fit_visit_embedding(state[features],cfg)
    assert pca.explained_variance_ratio_.sum()>=cfg["pca"]["variance_target"]
    assert primary.explained_variance_ratio_.sum()<cfg["pca"]["variance_target"]
    assert pca.n_components_>10 and pca.n_components_<=min(len(state),len(features))
    assert embedding.shape==(len(state),pca.n_components_)


def test_feature_diagnostics_use_observed_values_and_explicit_extreme_rule():
    values=[-100.,0.,1.,2.,3.,4.,5.,100.,np.nan,np.inf]
    raw=pd.DataFrame({MAP.HSCRP:values})
    out=MAP.feature_distribution_diagnostics(raw).set_index("feature").loc[MAP.HSCRP]
    valid=np.array(values[:8]); q1,q3=np.quantile(valid,[.25,.75]); iqr=q3-q1
    assert out.n_observed==8 and out.pct_observed==80
    assert out["median"]==np.median(valid) and out.iqr==iqr
    assert out.p95==np.quantile(valid,.95) and out.p99==np.quantile(valid,.99)
    assert out.n_zero==1 and out.n_negative==1 and out.n_extreme_iqr==2
    assert out.extreme_lower_bound==q1-3*iqr and out.extreme_upper_bound==q3+3*iqr


def test_nuisance_sensitivity_removes_exactly_one_fixed_feature():
    state=synthetic_mapper_state(); features=list(state.columns[2:])+[MAP.URINE_SQUAMOUS]
    state[MAP.URINE_SQUAMOUS]=np.arange(len(state))%7
    reduced=MAP.sensitivity_features(features,MAP.URINE_SQUAMOUS)
    assert len(reduced)==len(features)-1 and set(features)-set(reduced)=={MAP.URINE_SQUAMOUS}
    scenario=MAP.run_mapper_scenario(state,reduced,mapper_config(),family_balance=True,scenario_name="S3-without-urine-squamous")
    assert scenario["summary"]["n_features"]==len(features)-1
    assert scenario["features"]==reduced


def test_hscrp_log_sensitivity_is_valid_only_for_nonnegative_original_values():
    state=pd.DataFrame({MAP.HSCRP:[0.,1.,2.]}); features=[MAP.HSCRP]
    diag=MAP.feature_distribution_diagnostics(state)
    transformed,reason=MAP.hscrp_log_state(state,features,diag)
    assert reason is None
    np.testing.assert_allclose(transformed[MAP.HSCRP],np.log1p(state[MAP.HSCRP]))
    assert state[MAP.HSCRP].tolist()==[0.,1.,2.]
    raw=state.copy(); raw.loc[0,MAP.HSCRP]=-1
    transformed,reason=MAP.hscrp_log_state(state,features,MAP.feature_distribution_diagnostics(raw))
    assert transformed is None and reason=="negative_original_values"
    unavailable=MAP.feature_distribution_diagnostics(state,source="source_unavailable")
    assert unavailable.n_observed.isna().all() and unavailable.pct_observed.isna().all()
    transformed,reason=MAP.hscrp_log_state(state,features,unavailable)
    assert transformed is None and reason=="original_observations_unavailable"


def test_sensitivity_without_nodes_remains_descriptive(monkeypatch):
    def no_nodes(*args): raise ValueError("Mapper produced no nodes")
    monkeypatch.setattr(MAP,"run_mapper",no_nodes)
    state=synthetic_mapper_state()
    scenario=MAP.run_mapper_scenario(state,list(state.columns[2:]),mapper_config(),family_balance=False,scenario_name="S2-no-family-balance")
    assert scenario["summary"]["status"]=="no_supported_coverage"
    assert scenario["summary"]["n_uncovered_visits"]==len(state)
    assert scenario["macrostate_table"].empty and scenario["node_table"].empty
    matching,summary=MAP.compare_mapper_scenarios(scenario,scenario)
    assert matching.empty and summary["ari"] is None


def test_interval_coverage_keeps_adjacency_and_supported_denominator():
    intervals=PREP.build_consecutive_intervals(master().query("patient_id == 'A'"))
    m=membership(); m["mapper_graph_covered"]=True
    summary=MAP.interval_coverage_summary(intervals,m)
    assert summary["n_intervals_total"]==2
    assert summary["n_intervals_both_endpoints_mapper_graph_covered"]==2
    assert summary["n_intervals_both_endpoints_supported_macrostate_covered"]==0
    assert summary["n_intervals_both_endpoints_hard_assigned"]==0


def test_script02_end_to_end_preserves_canonical_outputs_and_all_five_scenarios(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    dirs={kind:tmp_path/kind for kind in ("analytic","tables","figures","qc","logs")}
    for path in dirs.values(): path.mkdir()
    monkeypatch.setattr(MAP,"create_study_dirs",lambda _:dirs)
    state=synthetic_mapper_state(n=120,n_features=22)
    rng=np.random.default_rng(31)
    state[MAP.HSCRP]=rng.lognormal(size=len(state)); state[MAP.URINE_SQUAMOUS]=rng.poisson(4,size=len(state)).astype(float)
    raw=state.copy(); raw.loc[:9,MAP.HSCRP]=np.nan
    # Exercise raw Step-10 nullable dtypes through every scenario and final QC output.
    raw["ext__nullable_binary"]=pd.Series([True,False,pd.NA]*40,dtype="boolean")
    raw["lab__nullable_integer"]=pd.Series(np.arange(len(raw))%5,dtype="Int64")
    raw.loc[::7,"lab__nullable_integer"]=pd.NA
    raw["lab__nullable_float"]=pd.Series(rng.normal(size=len(raw)),dtype="Float64")
    raw.loc[0,"lab__nullable_float"]=np.inf;raw.loc[1,"lab__nullable_float"]=pd.NA
    for feature in ("ext__nullable_binary","lab__nullable_integer","lab__nullable_float"):
        numeric=raw[feature].astype("float64").replace([np.inf,-np.inf],np.nan)
        state[feature]=numeric.fillna(numeric.median())
    metadata=state[MAP.KEYS].assign(visit_type="routine")
    state_path=tmp_path/"state.parquet"; metadata_path=tmp_path/"metadata.parquet"; raw_path=tmp_path/"raw.parquet"
    intervals_path=tmp_path/"intervals.parquet"
    state.to_parquet(state_path); metadata.to_parquet(metadata_path); raw.to_parquet(raw_path)
    intervals=pd.DataFrame({"patient_id":state.patient_id.iloc[::2].tolist(),
                            "from_clinical_episode_id":state.clinical_episode_id.iloc[::2].tolist(),
                            "to_clinical_episode_id":state.clinical_episode_id.iloc[1::2].tolist(),
                            "interval_days":30,"interval_years":30/365.25})
    intervals.to_parquet(intervals_path)
    import yaml
    cfg=mapper_config();cfg["mapper_validation"].update(bootstrap_replicates=3,admin_permutations=9)
    config_path=tmp_path/"config.yaml";config_path.write_text(yaml.safe_dump(cfg))
    args=SimpleNamespace(state=state_path,metadata=metadata_path,integrated=raw_path,intervals=intervals_path,
                         config=config_path,dry_run=False)
    summary=MAP.run(args)
    canonical=["02_longitudinal_pca_variance.csv","02_longitudinal_pca_loadings.csv","02_mapper_nodes.csv",
               "02_mapper_topological_edges.csv","02_mapper_macrostate_summary.csv","02_mapper_visit_membership.csv",
               "02_mapper_visit_type_qc.csv","02_mapper_administrative_qc.csv","02_longitudinal_redundancy_audit.csv",
               "02_mapper_representation_manifest.csv","02_longitudinal_mapper_summary.json"]
    assert all((dirs["tables"]/name).exists() for name in canonical)
    for suffix in ("nodes","macrostate_summary","visit_membership"):
        pd.testing.assert_frame_equal(pd.read_csv(dirs["tables"]/f"02_mapper_{suffix}.csv"),
                                      pd.read_csv(dirs["tables"]/f"02_s3_mapper_{suffix}.csv"))
        assert (dirs["tables"]/f"02_s2_mapper_{suffix}.csv").exists()
    table=pd.read_csv(dirs["tables"]/"02_mapper_sensitivity_summary.csv")
    historical={"S3-primary","S2-no-family-balance","S3-full80","S3-without-urine-squamous","S3-hsCRP-log1p"}
    assert historical.issubset(set(table.scenario))
    assert set(table.loc[table.scenario.isin(historical),"status"])=={"completed"}
    full80=table.set_index("scenario").loc["S3-full80"]
    assert full80.n_pcs>10 and full80.cumulative_variance>=.8
    assert table.set_index("scenario").loc["S3-without-urine-squamous","n_features"]==summary["n_features"]-1
    diagnostics=pd.read_csv(dirs["tables"]/"02_feature_distribution_diagnostics.csv")
    assert diagnostics.set_index("feature").loc[MAP.HSCRP,"n_observed"]==110
    pc_diagnostics=pd.read_csv(dirs["tables"]/"02_mapper_pc_feature_diagnostics.csv")
    assert set(pc_diagnostics.scenario).issubset(set(table.loc[table.status.eq("completed"),"scenario"]))
    assert historical.issubset(set(pc_diagnostics.scenario))
    assert pc_diagnostics.squared_loading_fraction.between(0,1).all()
    assert pc_diagnostics.absolute_loading_rank.ge(1).all()
    assert not ((pc_diagnostics.scenario=="S3-without-urine-squamous") & (pc_diagnostics.feature==MAP.URINE_SQUAMOUS)).any()
    loaded=json.loads((dirs["tables"]/"02_longitudinal_mapper_summary.json").read_text(),parse_constant=lambda s: (_ for _ in ()).throw(AssertionError(s)))
    assert loaded["primary_scenario"]=="S3-like"
    assert not loaded["representation_stability_gate"]["temporal_flow_executed"]
    assert loaded["representation_stability_gate"]["status"] in {"requires_review","blocked"}
    supported=pd.read_parquet(dirs["analytic"]/"02_visit_mapper_membership.parquet")
    FLOW.validate_memberships(supported)
    attached=FLOW.attach_interval_memberships(intervals,supported)
    assert attached.flow_eligible.sum()==summary["interval_coverage"]["n_intervals_both_endpoints_supported_macrostate_covered"]
    # A second run with unavailable raw observations must not retain old log evidence.
    args.integrated=tmp_path/"absent.parquet"
    rerun=MAP.run(args)
    log=rerun["sensitivity_comparisons"]["S3-hsCRP-log1p"]
    assert log["status"]=="not_applicable" and log["ari"] is None
    assert pd.read_csv(dirs["tables"]/"02_hscrp_log1p_macrostate_stability.csv").empty
    assert rerun["run_manifest"]["historical_archive"]
    assert Path(rerun["run_manifest"]["historical_archive"]).exists()
    assert (dirs["figures"]/"02_reference_visit_mapper.png").read_bytes()!=(dirs["figures"]/"02_reference_visit_mapper_macrostates.png").read_bytes()


def test_duplicate_null_keys_and_ineligible_features_fail_before_fitting():
    import pytest
    from src.studies.longitudinal_graph.runner import validate_inputs
    state=synthetic_mapper_state(n=10,n_features=3);metadata=state[MAP.KEYS].copy()
    with pytest.raises(ValueError,match="duplicate"):
        MAP.run_mapper_scenario(pd.concat([state,state.iloc[:1]]),list(state.columns[2:]),mapper_config(),family_balance=True,scenario_name="invalid")
    state.loc[0,"patient_id"]=None
    with pytest.raises(ValueError,match="missing"):
        MAP.run_mapper_scenario(state,list(state.columns[2:]),mapper_config(),family_balance=True,scenario_name="invalid")
    state=synthetic_mapper_state(n=10,n_features=3);state["ext__future_outcome"]=1.
    with pytest.raises(ValueError,match="future_leakage"):
        validate_inputs(state,metadata,mapper_config(),MAP)


def test_loading_fractions_family_totals_and_variance_denominators():
    from src.studies.longitudinal_graph.validation import representation_qc
    state=synthetic_mapper_state();state["pro__symptom"]=np.random.default_rng(3).normal(size=len(state))
    features=list(state.columns[2:]);scenario=MAP.run_mapper_scenario(state,features,mapper_config(),family_balance=True,scenario_name="S3-primary")
    tables=representation_qc(state,scenario,MAP.feature_family,state)
    for name in ("pca_loading_dominance","pca_family_contributions"):
        np.testing.assert_allclose(tables[name].groupby("component").squared_loading_fraction.sum(),1.)
    cumulative=np.cumsum(scenario["pca"].explained_variance_ratio_)
    assert (np.diff(cumulative)>=0).all() and cumulative[-1]<=1+1e-12


def test_registry_PRO_exclusion_keeps_original_keys_and_original_features():
    from src.studies.longitudinal_graph.runner import registry
    state=synthetic_mapper_state();state["pro__symptom"]=np.arange(len(state))
    original=state.copy(deep=True);features=list(state.columns[2:])
    definitions={d["name"]:d for d in registry(features,mapper_config())}
    assert not any(f.startswith("pro__") for f in definitions["S3-no-PRO"]["feature_list"])
    assert definitions["S3-without-hsCRP"]["omit_reason"]=="feature_not_retained_after_redundancy"
    assert definitions["S3-PRO-summary-only"]["omit_reason"]
    pd.testing.assert_frame_equal(state,original)


def test_cover_extrema_overlap_empty_cells_local_fallback_and_noise():
    from src.studies.longitudinal_graph.validation import audited_mapper
    lens=np.array([[0.,0.],[0.,.5],[.5,.5],[1.,1.],[100.,100.]])
    cfg=mapper_config();cfg["mapper"].update(n_cubes=3,min_samples=100,eps_mode="local")
    graph,cover,noise=audited_mapper(lens,lens,cfg,.1,MAP.eps_kdist,np.array(list("ABCDE")))
    assert not graph["nodes"] and len(cover)==9
    assert not noise.outside_cover.any() and noise.noise_only.all()
    assert (cover.n_points==0).any() and cover.fallback.str.contains("small_preimage").any()
    cfg["mapper"]["cover_quantiles"]=[.1,.9]
    _,_,noise=audited_mapper(lens,lens,cfg,.1,MAP.eps_kdist)
    assert noise.outside_cover.iloc[-1]


def test_patient_weighted_PCA_ignores_extra_copies_of_same_patient_visit():
    from src.studies.longitudinal_graph.validation import weighted_embedding
    state=pd.DataFrame({"x":[-2.,0.,2.,4.],"y":[1.,-1.,2.,7.]});patients=np.array(list("ABCD"))
    cfg=mapper_config();a,pca=weighted_embedding(state,cfg,None,patients)
    repeat=pd.concat([state,state.iloc[[0]],state.iloc[[0]]],ignore_index=True)
    b,pca2=weighted_embedding(repeat,cfg,None,np.r_[patients,["A","A"]])
    np.testing.assert_allclose(pca.components_,pca2.components_,atol=1e-12)
    np.testing.assert_allclose(a,b[:4],atol=1e-12)


def test_exclusive_patient_coverage_reconciles_when_patients_span_categories():
    from src.studies.longitudinal_graph.validation import coverage_detail,coverage_by_stratum
    scenario=permutation_scenario([0,0,1,1]);out=scenario["visit_membership"]
    out.loc[1,["mapper_covered","mapper_graph_covered","hard_macrostate"]]=[False,False,pd.NA]
    scenario["graph"]["nodes"]["node_0"]={0}
    scenario["visit_membership"]=out;detail=coverage_detail(out,scenario)
    summary=coverage_by_stratum(detail,out[MAP.KEYS])
    all_rows=summary.loc[summary.stratum.eq("all")]
    assert all_rows.n_visits.sum()==4 and all_rows.n_patients_exclusive.sum()==4


def test_bootstrap_preserves_patient_series_and_matches_labels_splits_and_merges():
    from src.studies.longitudinal_graph.stability import patient_draw,match_structures
    raw=synthetic_mapper_state(n=12,n_features=3)
    a,common=patient_draw(raw,np.random.default_rng(5));b,_=patient_draw(raw,np.random.default_rng(5))
    pd.testing.assert_frame_equal(a,b)
    for patient in common:
        assert len(a.loc[a.patient_id.eq(patient)])%len(raw.loc[raw.patient_id.eq(patient)])==0
    left={0:({"A","B"},{("A","1"),("B","1")}),1:({"C"},{("C","1")})}
    right={9:({"A","B","C"},{("A","1"),("B","1"),("C","1")})}
    matches=match_structures(left,right,MAP.jaccard)
    assert all(m["matched_structure_id"]==9 and m["merge_candidate"] for m in matches)
    assert all(m["match_status"]=="unmatched" for m in match_structures(left,{},MAP.jaccard))


def test_admin_missingness_is_not_zero_association_and_inference_uses_patient_vectors():
    from src.studies.longitudinal_graph.validation import administrative_audit
    scenario=permutation_scenario([0,0,1,1]);m=scenario["visit_membership"]
    metadata=m[MAP.KEYS].assign(visit_type=["X","X","Y","Y"])
    cfg=mapper_config();cfg["mapper_validation"]["admin_permutations"]=9
    assoc,availability,_,_=administrative_audit(m,metadata,np.arange(8).reshape(4,2),cfg)
    assert availability.set_index("variable").loc["protocol","status"]=="not_tested_missing_variable"
    row=assoc.set_index("administrative_variable").loc["visit_type"]
    assert row.n_permutations==9 and row.inference_status.startswith("patient_vector")
    assert not row.visit_p_value_used


def test_dry_runs_validate_without_writes_or_full_mapper(tmp_path,monkeypatch):
    from types import SimpleNamespace
    def forbidden(*args,**kwargs):raise AssertionError("dry-run must not create output folders or fit Mapper")
    monkeypatch.setattr(MAP,"create_study_dirs",forbidden);monkeypatch.setattr(MAP,"run_mapper_scenario",forbidden)
    args=SimpleNamespace(state=tmp_path/"missing",metadata=tmp_path/"missing_meta",config=ROOT/"src/studies/longitudinal_graph/config.yaml",dry_run=True)
    assert MAP.run(args)["status"]=="not_run_missing_data"
    state=synthetic_mapper_state(n=10,n_features=3);args.state=tmp_path/"state.parquet";args.metadata=tmp_path/"metadata.parquet"
    state.to_parquet(args.state);state[MAP.KEYS].to_parquet(args.metadata)
    assert MAP.run(args)["status"]=="validated"
    monkeypatch.setattr(PREP,"create_study_dirs",forbidden)
    prep_args=SimpleNamespace(integrated=tmp_path/"absent",config=args.config,dry_run=True)
    assert PREP.run(prep_args)["status"]=="not_run_missing_data"


def test_bootstrap_zero_node_refits_are_unmatched_and_counted(monkeypatch):
    from src.studies.longitudinal_graph.stability import bootstrap_mapper
    raw=synthetic_mapper_state(n=40,n_features=3)
    primary=MAP.run_mapper_scenario(raw,list(raw.columns[2:]),mapper_config(),family_balance=True,scenario_name="S3-primary")
    cfg=mapper_config();cfg["mapper_validation"]["bootstrap_replicates"]=3
    def zero_node_fit(*args,**kwargs):
        local=mapper_config();local["mapper"]["min_samples"]=1000
        return MAP.run_mapper_scenario(args[0],args[1],local,**kwargs)
    replicates,matches,summary=bootstrap_mapper(raw,primary,cfg,list(raw.columns[2:]),zero_node_fit,MAP.reduce_redundancy,MAP.jaccard)
    assert summary["replicates_no_nodes"]==3 and summary["pct_without_viable_structure"]==100
    assert replicates.status.eq("no_nodes").all()
    if len(matches):assert matches.match_status.eq("unmatched").all()


def test_synthetic_contract_01_to_02_to_03_to_04(tmp_path,monkeypatch):
    """Clinical values are fixtures; downstream inspection here authorizes no real flow."""
    import yaml
    from types import SimpleNamespace
    CHAR=load("04_characterize_transitions.py")
    dirs={k:tmp_path/k for k in ("analytic","tables","figures","qc","logs")}
    for p in dirs.values():p.mkdir()
    monkeypatch.setattr(PREP,"create_study_dirs",lambda _:dirs)
    monkeypatch.setattr(MAP,"create_study_dirs",lambda _:dirs)
    raw=synthetic_mapper_state(n=120,n_features=8)
    numbers=np.tile([1,2],60)
    raw["clinical_visit_number"]=numbers;raw["clinical_anchor_date"]=pd.to_datetime("2020-01-01")+pd.to_timedelta(numbers,unit="D")
    raw["clinical_visit"]=True;raw["is_clinical_baseline"]=numbers==1
    raw["essdai__total"]=2.;raw["esspri__total"]=4.;raw["pop__status"]="Pop2"
    raw["pro__symptom"]=np.random.default_rng(3).normal(size=len(raw));raw.loc[:9,"pro__symptom"]=np.nan
    raw_path=tmp_path/"integrated.parquet";raw.to_parquet(raw_path,index=False)
    cfg=mapper_config();cfg["mapper_validation"].update(bootstrap_replicates=2,admin_permutations=3)
    cfg_path=tmp_path/"config.yaml";cfg_path.write_text(yaml.safe_dump(cfg))
    prep_args=SimpleNamespace(integrated=raw_path,intervals=tmp_path/"absent_intervals",config=cfg_path,dry_run=False)
    PREP.run(prep_args)
    observed=pd.read_parquet(dirs["analytic"]/"01_longitudinal_visit_observability.parquet")
    assert not observed.loc[:9,"pro__symptom"].any()
    args=SimpleNamespace(state=dirs["analytic"]/"01_longitudinal_visit_state.parquet",metadata=dirs["analytic"]/"01_longitudinal_visit_metadata.parquet",
                         integrated=raw_path,intervals=dirs["analytic"]/"01_longitudinal_intervals.parquet",config=cfg_path,dry_run=False,
                         feature_manifest=dirs["tables"]/"01_longitudinal_feature_manifest.csv")
    summary=MAP.run(args)
    assert summary["status"]=="completed" and summary["representation_stability_gate"]["status"]!="reviewed_approved"
    actual=pd.read_csv(dirs["tables"]/"02_mapper_representation_manifest.csv")
    assert not actual.loc[actual.used_in_primary_mapper,"feature"].isin(["essdai__total","esspri__total","pop__status"]).any()
    m=pd.read_parquet(dirs["analytic"]/"02_visit_mapper_membership.parquet")
    intervals=pd.read_parquet(args.intervals);attached=FLOW.attach_interval_memberships(intervals,m)
    assert len(attached)==60 and attached.flow_eligible.sum()==summary["interval_coverage"]["n_intervals_both_endpoints_supported_macrostate_covered"]
    flow=FLOW.compute_temporal_flow(attached)
    assert np.isclose(flow.weighted_interval_mass.sum() if len(flow) else 0,attached.flow_eligible.sum())
    joined=CHAR.attach_clinical_endpoints(attached,raw,["essdai__total","pro__symptom"])
    assert len(joined)==len(intervals)
    joined=joined.loc[joined.flow_eligible].copy();joined["transition_group"]=CHAR.transition_label(joined)
    table=CHAR.characterize_continuous(joined,["essdai__total"],{"essdai__total":"external_characterizer"},3,7,3)
    assert table.role.eq("external_characterizer").all()


def test_hscrp_pretransformed_values_are_explicitly_inapplicable():
    state=pd.DataFrame({MAP.HSCRP:[0.,1.,2.]})
    transformed,reason=MAP.hscrp_log_state(state,[MAP.HSCRP],MAP.feature_distribution_diagnostics(state),prior_transform="log")
    assert transformed is None and reason=="prior_transform_not_verified_identity"


def test_equal_node_weights_preserve_previous_method_and_no_nodes_have_zero_mass():
    import importlib.util
    # Explicit expected equal-node contract; tau does not silently change these values.
    graph={"nodes":{"a":{0},"b":{0},"c":{0}},"links":{n:[] for n in "abc"}}
    visits=pd.DataFrame({"patient_id":["A","B"],"clinical_episode_id":["1","2"]})
    macro=pd.DataFrame({"macrostate_id":[0,1],"macrostate_supported":[True,True]})
    out=MAP.normalized_memberships(graph,visits,{"a":0,"b":0,"c":1},macro)
    np.testing.assert_allclose(out.loc[0,["node_weight__a","node_weight__b","node_weight__c"]].astype(float),[1/3]*3)
    np.testing.assert_allclose(out.loc[0,["macrostate_weight__0","macrostate_weight__1"]].astype(float),[2/3,1/3])
    assert not out.loc[1,"mapper_covered"] and out.loc[1].filter(like="weight__").fillna(0).sum()==0


def test_bootstrap_imputes_nullable_boolean_integer_and_float_measurements(monkeypatch):
    from src.studies.longitudinal_graph import stability
    raw=synthetic_mapper_state(n=60,n_features=4)
    raw["ext__binary_mixed"]=pd.Series([True,False,pd.NA]*20,dtype="boolean")
    raw["ext__binary_positive"]=pd.Series([True,True,pd.NA]*20,dtype="boolean")
    raw["lab__integer_measurement"]=pd.Series([0,1,pd.NA]*20,dtype="Int64")
    raw["lab__nullable_float"]=pd.Series([np.inf,2.,pd.NA]*20,dtype="Float64")
    original=raw.copy(deep=True);features=list(raw.columns[2:]);cfg=mapper_config()
    cfg["mapper_validation"]["bootstrap_replicates"]=1
    imputed=raw[features].astype("float64").replace([np.inf,-np.inf],np.nan)
    imputed=imputed.fillna(imputed.median())
    state=pd.concat([raw[MAP.KEYS],imputed],axis=1)
    primary=MAP.run_mapper_scenario(state,features,cfg,family_balance=True,scenario_name="S3-primary")
    def fixed_draw(frame,rng):
        sample=frame.copy();sample["original_episode_id"]=sample.clinical_episode_id
        return sample,set(frame.patient_id)
    monkeypatch.setattr(stability,"patient_draw",fixed_draw)
    captured=[]
    def checked_pruning(frame,threshold):
        captured.append(frame.copy())
        return MAP.reduce_redundancy(frame,threshold)
    replicates,_,summary=stability.bootstrap_mapper(raw,primary,cfg,features,MAP.run_mapper_scenario,checked_pruning,MAP.jaccard)
    assert not replicates.status.eq("failed").any() and summary["replicates_failed"]==0
    fitted=captured[0]
    assert all(dtype==np.dtype("float64") for dtype in fitted.dtypes)
    assert np.isfinite(fitted.to_numpy()).all()
    for feature,median in (("ext__binary_mixed",.5),("ext__binary_positive",1.),("lab__integer_measurement",.5)):
        np.testing.assert_allclose(fitted.loc[raw[feature].isna(),feature],median)
    np.testing.assert_allclose(fitted["lab__nullable_float"],2.)
    pd.testing.assert_frame_equal(raw,original)


def test_distribution_QC_handles_boolean_nullable_and_nonfinite_originals():
    from src.studies.longitudinal_graph.validation import representation_qc
    state=synthetic_mapper_state(n=60,n_features=4)
    feature=state.columns[2];features=list(state.columns[2:])
    scenario=MAP.run_mapper_scenario(state,features,mapper_config(),family_balance=True,scenario_name="S3-primary")
    cases=[(pd.Series([True,False,pd.NA]*20,dtype="boolean"),1.),
           (pd.Series([0,1,pd.NA]*20,dtype="Int64"),1.),
           (pd.Series([0.,1.,np.inf,pd.NA]*15,dtype="Float64"),1.),
           (pd.Series([pd.NA]*60,dtype="boolean"),None)]
    for values,expected in cases:
        raw=state.copy();raw[feature]=values;original=raw.copy(deep=True)
        qc=representation_qc(state,scenario,MAP.feature_family,raw)["scaling_feature_diagnostics"]
        row=qc.set_index("feature").loc[feature]
        if expected is None:
            assert pd.isna(row.original_iqr) and row.original_status=="not_applicable"
        else:
            assert row.original_iqr==expected and row.original_status=="completed"
            diag=MAP.feature_distribution_diagnostics(raw,features=[feature]).iloc[0]
            assert diag.iqr==expected and diag.n_observed in {30,40}
        pd.testing.assert_frame_equal(raw,original)


def test_representation_QC_failure_prevents_expensive_bootstrap(tmp_path,monkeypatch):
    import pytest
    from types import SimpleNamespace
    from src.studies.longitudinal_graph import runner
    raw=synthetic_mapper_state(n=40,n_features=5)
    state_path=tmp_path/"state.parquet";metadata_path=tmp_path/"metadata.parquet"
    raw.to_parquet(state_path);raw[MAP.KEYS].to_parquet(metadata_path)
    def fail_qc(*args,**kwargs):raise RuntimeError("representation_QC_failed")
    def unexpected_bootstrap(*args,**kwargs):raise AssertionError("bootstrap must wait for representation QC")
    monkeypatch.setattr(runner.qc,"representation_qc",fail_qc)
    monkeypatch.setattr(runner,"bootstrap_mapper",unexpected_bootstrap)
    args=SimpleNamespace(state=state_path,metadata=metadata_path,integrated=state_path,intervals=tmp_path/"absent_intervals",
                         config=ROOT/"src/studies/longitudinal_graph/config.yaml",dry_run=False,output_root=tmp_path/"bundle")
    with pytest.raises(RuntimeError,match="representation_QC_failed"):MAP.run(args)
    assert not args.output_root.exists()

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
    metadata=state[MAP.KEYS].assign(visit_type="routine")
    state_path=tmp_path/"state.parquet"; metadata_path=tmp_path/"metadata.parquet"; raw_path=tmp_path/"raw.parquet"
    intervals_path=tmp_path/"intervals.parquet"
    state.to_parquet(state_path); metadata.to_parquet(metadata_path); raw.to_parquet(raw_path)
    intervals=pd.DataFrame({"patient_id":state.patient_id.iloc[::2].tolist(),
                            "from_clinical_episode_id":state.clinical_episode_id.iloc[::2].tolist(),
                            "to_clinical_episode_id":state.clinical_episode_id.iloc[1::2].tolist(),
                            "interval_days":30,"interval_years":30/365.25})
    intervals.to_parquet(intervals_path)
    args=SimpleNamespace(state=state_path,metadata=metadata_path,integrated=raw_path,intervals=intervals_path,
                         config=ROOT/"src/studies/longitudinal_graph/config.yaml",dry_run=False)
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
    assert len(table)==5 and set(table.status)=={"completed"}
    full80=table.set_index("scenario").loc["S3-full80"]
    assert full80.n_pcs>10 and full80.cumulative_variance>=.8
    assert table.set_index("scenario").loc["S3-without-urine-squamous","n_features"]==summary["n_features"]-1
    diagnostics=pd.read_csv(dirs["tables"]/"02_feature_distribution_diagnostics.csv")
    assert diagnostics.set_index("feature").loc[MAP.HSCRP,"n_observed"]==110
    pc_diagnostics=pd.read_csv(dirs["tables"]/"02_mapper_pc_feature_diagnostics.csv")
    assert set(pc_diagnostics.scenario)==set(table.scenario)
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

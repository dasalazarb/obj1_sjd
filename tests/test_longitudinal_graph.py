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

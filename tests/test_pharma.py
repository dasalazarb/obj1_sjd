"""Synthetic behavioral tests for the Pharma atlas."""
import importlib.util
from pathlib import Path
import pandas as pd

SPEC=importlib.util.spec_from_file_location("pharma",Path(__file__).parents[1]/"src/studies/pharma/01_run_pharma.py")
pharma=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(pharma)


def frame():
 rows=[]
 for i,o in enumerate(pharma.CLASSIFIABLE):
  for j,d in enumerate(pharma.CLASSIFIABLE):
   rows.append({"patient_id":f"{i}{j}","from_clinical_episode_id":f"F{i}{j}","to_clinical_episode_id":f"T{i}{j}",
    "from_clinical_anchor_date":pd.Timestamp("2020-01-01"),"from_pop":o,"to_pop":d,"interval_years":1.,
    "from_essdai_total":4 if o!="Pop1" else 7,"to_essdai_total":7 if d=="Pop1" else 3,
    "from_esspri_total_observed":5.,"to_esspri_total_observed":4.})
 return pharma.classify_sustained_transitions(pharma.derive_transition_outcomes(pd.DataFrame(rows)))


def test_baseline_uses_canonical_flag():
 x=pd.DataFrame({"patient_id":[1,1],"is_clinical_baseline":[False,True],"clinical_anchor_date":["2019","2020"]})
 assert pharma.select_clinical_baseline(x).clinical_anchor_date.item()=="2020"


def test_full_matrix_and_outcomes():
 x=frame(); table=pharma.transition_matrix(x)
 assert len(table)==9 and table.n_intervals.sum()==9
 assert (table.transition_type=="stability").sum()==3
 assert (table.transition_type=="directional_transition").sum()==6
 assert table.groupby("from_pop").pct_within_origin.sum().eq(1).all()
 assert x.changed_state.sum()==6 and x.stable_state.sum()==3
 assert x.loc[x.strict_systemic_worsening.eq(1),"to_pop"].eq("Pop1").all()


def test_directional_comparator_excludes_competing_destination():
 x=pharma.directional_contrast(frame(),"Pop1","Pop2")
 assert set(x.to_pop)=={"Pop1","Pop2"}
 assert x.loc[x.directional_event.eq(0),"to_pop"].eq("Pop1").all()


def test_sustained_requires_following_confirmation():
 x=pd.DataFrame([{"patient_id":"A","from_clinical_episode_id":"A1","to_clinical_episode_id":"A2","from_pop":"Pop1","to_pop":"Pop2","changed_state":1},
  {"patient_id":"A","from_clinical_episode_id":"A2","to_clinical_episode_id":"A3","from_pop":"Pop2","to_pop":"Pop2","changed_state":0},
  {"patient_id":"B","from_clinical_episode_id":"B1","to_clinical_episode_id":"B2","from_pop":"Pop1","to_pop":"Pop3","changed_state":1}])
 out=pharma.classify_sustained_transitions(x)
 assert out.sustained_transition.tolist()==[True,False,False]


def test_s2_and_s6_and_s10():
 x=pd.concat([frame(),frame()],ignore_index=True)
 assert not pharma.first_interval_per_patient_origin(x).duplicated(["patient_id","from_pop"]).any()
 u=frame(); u.loc[0,"to_pop"]="Unclassifiable"
 assert pharma.unclassifiable_bounds(u).destination_pop.eq("").all()
 s10=pharma.threshold_sensitivity(frame())
 assert set(s10.origin_pop)=={"Pop2","Pop3"}


def test_cluster_bootstrap_is_deterministic_and_keeps_clusters():
 x=frame(); a=pharma.bootstrap_patients(x,4); b=pharma.bootstrap_patients(x,4)
 pd.testing.assert_frame_equal(a,b)
 for _,g in a.groupby("bootstrap_patient_id"):
  assert g.patient_id.nunique()==1

"""Unit tests for canonical Step 13 endpoint definitions."""
import importlib.util
from pathlib import Path
import pandas as pd

PATH=Path(__file__).parents[1]/"src/block_A/13_disease_activity_progression.py"
spec=importlib.util.spec_from_file_location("progression13",PATH)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def episodes(scores, baseline_score=None):
    dates=pd.to_datetime(["2020-01-01","2021-01-01","2022-01-01"][:len(scores)])
    return pd.DataFrame({"patient_id":["p"]*len(scores),"clinical_episode_id":[f"e{i}" for i in range(len(scores))],"clinical_anchor_date":dates,"clinical_baseline_date":[dates[0]]*len(scores),"is_clinical_baseline":[True]+[False]*(len(scores)-1),"essdai__total":scores})


def test_state_boundaries():
    assert [mod.classify_essdai_activity(x) for x in [0,4,5,13,14,40,None]] == ["Low","Low","Moderate","Moderate","High","High","Missing"]


def test_baseline_to_last_status():
    assert mod.baseline_to_last_status("Low","Moderate")=="worsened"
    assert mod.baseline_to_last_status("Moderate","High")=="worsened"
    assert mod.baseline_to_last_status("Moderate","Low")=="improved"
    assert mod.baseline_to_last_status("High","Moderate")=="improved"
    assert mod.baseline_to_last_status("Low","Low")=="stable"


def test_transient_worsening_is_retained():
    out=mod.patient_progression(episodes([3,7,3])).iloc[0]
    assert bool(out.ever_worsened) and out.status=="stable"


def test_km_risk_sets_exclude_wrong_baseline():
    low=episodes([3,7]); moderate=episodes([6,14]); moderate["patient_id"]="q"
    both=pd.concat([low,moderate],ignore_index=True)
    assert mod.build_tte_risk_set(both,"low").patient_id.tolist()==["p"]
    assert mod.build_tte_risk_set(both,"moderate").patient_id.tolist()==["q"]


def test_missing_middle_breaks_transition_chain():
    matrix,raw=mod.build_state_transitions(episodes([3,None,14]))
    assert raw.empty
    assert matrix.n_intervals.sum()==0


def test_no_baseline_substitution():
    assert mod.patient_progression(episodes([None,3])).empty

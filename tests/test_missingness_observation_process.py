"""Focused behavioral tests for Block B observation-process audit."""
import ast
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("missingness",ROOT/"src/block_B/20_missingness_and_observation_process.py")
M=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def base():
    return pd.DataFrame({"patient_id":["p1","p1","p2"],"visit_id":["v1","v2","v3"],"visit_date":pd.to_datetime(["2020-01-01","2021-01-01","2020-02-01"]),"visit_number":[0,1,0],"is_observed_baseline":pd.Series([True,False,True],dtype="boolean"),"time_since_observed_baseline_days":[0,366,0],"time_since_observed_baseline_years":[0,1.002,0],"essdai_total":[1,2,3],"esspri_total":[2,None,4],"sf36_pcs":[40,None,45],"sf36_mcs":[42,None,47],"profad_total":[1,None,2],"mdafs_global":[1,None,2],"glandular_evaluable":[True,True,True],"extraglandular_evaluable":[True,True,True],"overlap_status":["neither","neither","overlap"],"pop_status":["Pop 3","Pop 3","Pop 3"],"esspri_n_components_available":[3,0,3],"sf36_n_items_available":[36,0,36],"profad_n_items_available":[19,0,19],"mdafs_n_items_available":[16,0,16],"esspri_scoring_valid":[True,False,True],"sf36_scoring_valid":[True,False,True],"profad_scoring_valid":[True,False,True],"mdafs_scoring_valid":[True,False,True]})


def derived(df=None):
    x=base() if df is None else df
    x,_=M.derive_observation_states(x); return M.derive_completeness_indicators(x)


def test_merge_preserves_spine_and_rejects_one_to_many():
    x=base(); p=x[["patient_id","visit_id","esspri_scoring_valid"]]
    got,_,_=M.merge_pro_detail(x,p); assert len(got)==len(x)
    with pytest.raises(ValueError,match="one-to-many"): M.merge_pro_detail(x,pd.concat([p,p.iloc[[0]]]))


def test_duplicate_keys_and_global_visit_ids_raise():
    x=base(); x.loc[1,["patient_id","visit_id"]]=["p1","v1"]
    with pytest.raises(ValueError,match="patient_id.*visit_id"): M.validate_integrated_input(x)
    x=base(); x.loc[2,"visit_id"]="v2"
    with pytest.raises(ValueError,match="globally duplicated"): M.validate_integrated_input(x)


def test_pro_states_and_discrepancy():
    x=base().iloc[[0,1,2]].copy(); x["esspri_n_components_available"]=[0,2,3]; x["esspri_scoring_valid"]=[False,False,True]; x["esspri_total"]=[None,None,4]
    x["sf36_n_items_available"]=[0,4,36]; x["sf36_scoring_valid"]=[False,False,True]; x["sf36_pcs"]=[None,None,45]; x["sf36_mcs"]=[None,None,47]
    out,_=M.derive_observation_states(x)
    assert out.esspri_observation_state.tolist()==["absent","partial","complete"]
    assert out.sf36_observation_state.tolist()==["absent","partial","complete"]
    assert out.profad_observation_state.tolist()==["complete","absent","complete"]
    assert out.mdafs_observation_state.tolist()==["complete","absent","complete"]
    x.loc[x.index[0],["esspri_scoring_valid","esspri_n_components_available","esspri_total"]]=[True,0,4]
    out,q=M.derive_observation_states(x); assert out.iloc[0].esspri_observation_state=="complete" and len(q)>=1


def test_missing_detail_is_unknown_not_absent():
    x=base().drop(columns=[c for c in base() if "_n_" in c]); x[["esspri_scoring_valid","sf36_scoring_valid","profad_scoring_valid","mdafs_scoring_valid"]]=False; x[["esspri_total","sf36_pcs","sf36_mcs","profad_total","mdafs_global"]]=None
    out,_=M.derive_observation_states(x)
    assert all(out[c].eq("unknown_detail").all() for c in ["esspri_observation_state","sf36_observation_state","profad_observation_state","mdafs_observation_state"])


def test_pattern_deterministic_and_completeness_distinct():
    a=derived(); b=derived(base()[list(reversed(base().columns))])
    assert a.visit_observation_pattern.tolist()==b.visit_observation_pattern.tolist()
    assert a.iloc[0].full_multidomain_complete and a.iloc[0].extended_multidomain_complete
    x=base(); x.loc[0,"profad_scoring_valid"]=False; x.loc[0,"profad_n_items_available"]=1; x.loc[0,"profad_total"]=None; d=derived(x)
    assert d.loc[0,"full_multidomain_complete"] and not d.loc[0,"extended_multidomain_complete"]
    assert not d.pop_core_complete.equals(d.systemic_overlap_core_complete)

@pytest.mark.parametrize("seq,expected",[([1,1],"always_available"),([0,0],"never_available"),([0,0,1,1],"late_entry"),([1,1,0],"monotone_dropout"),([1,0,1],"intermittent"),([1],"single_visit")])
def test_patient_sequence_patterns(seq,expected): assert M.classify_binary_sequence(seq)==expected


def test_summaries_use_canonical_denominator_and_reconcile():
    d=derived(); overall=M.summarize_availability(d); assert set(overall.total_canonical_visits)=={3}
    for group in ["protocol_group","calendar_period","followup_time_bin"]:
        s=M.summarize_availability(d,group); assert s.loc[s.measure.eq("ESSDAI"),"total_canonical_visits"].sum()==3


def test_unknown_protocol_not_inferred_from_interval():
    x=base(); x["interval_name"]="11D exciting interval"
    assert M.derive_protocol_group(x).eq("Unknown").all()
    assert "protocol_group" in derived(x)


def test_actual_max_gap_and_membership_reconcile():
    x=pd.concat([base().iloc[[0]],base().iloc[[1]]],ignore_index=True); x["patient_id"]="p1"; x["visit_id"]=["a","b"]; x["visit_date"]=pd.to_datetime(["2020-01-01","2020-03-01"]); x["is_observed_baseline"]=[True,False]; x["esspri_scoring_valid"]=True; x["sf36_scoring_valid"]=True; x["profad_scoring_valid"]=True; x["mdafs_scoring_valid"]=True
    d=derived(x); mem=M.build_analytic_cohort_membership(d,180); assert mem.in_2plus_essdai_assessments.iloc[0] and not mem.in_2plus_essdai_assessments_min_gap.iloc[0]
    counts=M.build_analytic_cohort_counts(mem,d,180); row=counts[counts.membership_column_name.eq("in_2plus_essdai_assessments")].iloc[0]; assert row.number_of_patients==int(mem.in_2plus_essdai_assessments.sum()); assert "non-nested" in row.nesting


def test_model_features_have_no_current_target_leakage():
    assert not any("esspri_total" in x or "scoring_valid" in x for x in M.model_feature_columns("esspri"))
    assert not any("sf36_pcs" in x or "scoring_valid" in x for x in M.model_feature_columns("sf36"))


def test_models_skip_no_variation_and_user_skip_schema():
    d=derived(); d["avail_esspri_complete"]=True; outputs,diag,_=M.fit_observation_models(d); assert outputs["esspri"].model_status.iloc[0]=="skipped_no_outcome_variation"
    outputs,_,_=M.fit_observation_models(d,True); assert list(outputs["sf36"].columns)==M.MODEL_COLUMNS and outputs["sf36"].model_status.iloc[0]=="skipped_by_user"


def test_nullable_unknown_not_false():
    x=base(); x=x.drop(columns="sf36_n_items_available"); x["sf36_scoring_valid"]=pd.NA; x["sf36_pcs"]=None; x["sf36_mcs"]=None; d=derived(x); assert d.avail_sf36_complete.isna().all()


def test_import_has_no_mkdir_or_analysis_side_effect(tmp_path):
    # Module top level only defines paths/functions; filesystem writes live in main/setup/write.
    tree=ast.parse((ROOT/"src/block_B/20_missingness_and_observation_process.py").read_text())
    forbidden={"mkdir", "to_csv", "to_parquet", "main", "write_outputs"}
    top_names=[]
    for n in tree.body:
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call):
            f=n.value.func; top_names.append(f.attr if isinstance(f,ast.Attribute) else getattr(f,"id",""))
    assert forbidden.isdisjoint(top_names)


def test_python39_compatible_no_pep604_union():
    tree=ast.parse((ROOT/"src/block_B/20_missingness_and_observation_process.py").read_text())
    assert not any(isinstance(n,ast.BinOp) and isinstance(n.op,ast.BitOr) for n in ast.walk(tree))

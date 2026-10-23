import importlib.util
from pathlib import Path
import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "src/block_A/08_ctdb_visit_recovery.py"
spec = importlib.util.spec_from_file_location("ctdb_recovery", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def sample():
    visits = pd.DataFrame({"row_id_raw":[1,2],"patient_id":["p1","p1"],
        "medications__rx_1_name":["Plaquenil","Synthroid"]})
    mapping = pd.DataFrame({"row_id_raw":[1,2],"patient_id":["p1","p1"],
        "clinical_episode_id":["e1","e1"],"collection_date":pd.to_datetime(["2021-01-03","2021-02-04"])})
    spine = pd.DataFrame({"patient_id":["p1"],"clinical_episode_id":["e1"],
        "episode_start_date":pd.to_datetime(["2021-01-01"]),
        "episode_end_date":pd.to_datetime(["2021-02-28"]),
        "clinical_anchor_date":pd.to_datetime(["2021-01-03"]),
        "clinical_visit":[True]})
    return visits,mapping,spine

def test_visits_retain_distinct_dates_within_same_episode():
    catalog,dated,summary = mod.build(*sample(),b"x"*32)
    assert len(catalog)==len(dated)==2
    assert dated.visit_documentation_date.tolist()==["2021-01-03","2021-02-04"]
    assert not catalog.actual_exposure_confirmed.any()

def test_patient_mismatch_is_rejected():
    v,m,s=sample()
    m.loc[0,"patient_id"]="other"
    with pytest.raises(ValueError,match="patient disagreement"):
        mod.build(v,m,s,b"x"*32)

def test_unknown_episode_is_rejected():
    v,m,s=sample()
    m.loc[0,"clinical_episode_id"]="missing"
    with pytest.raises(ValueError,match="missing Step00 episode"):
        mod.build(v,m,s,b"x"*32)

def test_episode_date_bounds_checked_without_reassignment():
    v,m,sp=sample()
    sp.loc[0,"episode_end_date"]=pd.Timestamp("2021-01-15")
    catalog, dated, summary=mod.build(v,m,sp,b"x"*32)
    assert summary["n_report_dates_outside_episode_or_invalid_bounds"] == 1
    assert len(dated)==2
    assert catalog.medication_date_within_episode.tolist()==[True,False]
    assert catalog.actual_exposure_confirmed.eq(False).all()

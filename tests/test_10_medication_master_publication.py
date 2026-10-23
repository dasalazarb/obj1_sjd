"""Regression: meddoc descriptors appear in CSV/master and context without exposure inference."""
import importlib.util
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
p = HERE / "src/block_A/10_build_integrated_longitudinal_dataset.py"
spec = importlib.util.spec_from_file_location("step10_master_meddoc",p)
m = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=m
spec.loader.exec_module(m)

def test_medication_descriptors_present_in_both_outputs(monkeypatch, tmp_path):
    clinical = pd.DataFrame({
        "patient_id":["p"],"clinical_episode_id":["e"],
        "clinical_anchor_date":["2020-01-01"],"episode_start_date":["2020-01-01"],
        "episode_end_date":["2020-01-02"],"clinical_visit":[True]
    })
    cols = clinical[["patient_id","clinical_episode_id"]].copy()
    cols["meddoc__n_documentation_records"]=pd.Series([2],dtype="Int64")
    cols["meddoc__has_linked_documentation"]=[True]
    from pathlib import Path
    import importlib.util as iu
    mpath=p.with_name("_10_medication_documentation.py")
    mspec=iu.spec_from_file_location("step10_adapter_for_test",mpath)
    adapter=iu.module_from_spec(mspec);mspec.loader.exec_module(adapter)
    for col in adapter.COUNT_COLUMNS:
        if col not in cols: cols[col]=pd.Series([0],dtype="Int64")
    monkeypatch.setattr(adapter,"load_and_aggregate",lambda *a,**k:(cols,{"version":"test"}))
    original=iu.spec_from_file_location
    def intercepted(name,path):
        if str(path)==str(mpath):
            class FakeSpec:
                loader=type("FakeLoader",(),{"exec_module":lambda self,module:module.__dict__.update(adapter.__dict__)})()
            return FakeSpec()
        return original(name,path)
    monkeypatch.setattr(iu,"spec_from_file_location",intercepted)
    base = m.CuratedBuild(clinical.copy(),clinical.copy(),pd.DataFrame(),pd.DataFrame(),[],[],[],pd.DataFrame())
    built,qc=m._integrate_medication_context(base,audit_root=tmp_path,clinical_spine_path=tmp_path/"spine.parquet")
    assert len(built.analytic)==len(built.context)==1
    assert set(adapter.COUNT_COLUMNS+[adapter.FLAG]).issubset(built.analytic)
    assert built.analytic["meddoc__n_documentation_records"].tolist()==[2]
    assert built.context["meddoc__n_documentation_records"].tolist()==[2]
    assert built.registry.loc[built.registry.variable.eq("meddoc__n_documentation_records"),"included_in_analytic"].all()
    assert qc["pre_anchor_treatment_exposure_established"] is False
    out=tmp_path/"10_integrated_longitudinal_clinical_episode.csv"
    built.analytic.to_csv(out,index=False)
    assert "meddoc__n_documentation_records" in pd.read_csv(out,nrows=1)

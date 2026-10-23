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
    cols["meddoc__medication_names"]=pd.Series(["hydroxychloroquine"],dtype="string")
    from pathlib import Path
    import importlib.util as iu
    mpath=p.with_name("_10_medication_documentation.py")
    mspec=iu.spec_from_file_location("step10_adapter_for_test",mpath)
    adapter=iu.module_from_spec(mspec);mspec.loader.exec_module(adapter)
    for col in adapter.COUNT_COLUMNS:
        if col not in cols: cols[col]=pd.Series([0],dtype="Int64")
    original=iu.spec_from_file_location

    class AdapterStubLoader:
        """A real importlib loader contract for the synthetic module."""
        def create_module(self, spec):
            return None  # Allow importlib to construct the module.

        def exec_module(self, module):
            module.COUNT_COLUMNS = adapter.COUNT_COLUMNS
            module.FLAG = adapter.FLAG
            module.NAME_FIELD = adapter.NAME_FIELD
            module.write_protected_detail = lambda details, audit_root: tmp_path / "synthetic_detail.parquet"
            module.load_and_aggregate = lambda *args, **kwargs: (
                cols, {"version": "test"}, pd.DataFrame())

    def intercepted(name,path):
        if str(path)==str(mpath):
            return iu.spec_from_loader(name, AdapterStubLoader())
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

def test_medication_descriptors_are_checked_by_episode_key_when_orders_differ(monkeypatch, tmp_path):
    """Master/context may have different valid row order and ID dtypes."""
    import importlib.util as iu
    clinical = pd.DataFrame({
        "patient_id": pd.Series(["p", "p"], dtype="string"),
        "clinical_episode_id": pd.Series(["e1", "e2"], dtype="string"),
        "clinical_anchor_date": ["2020-01-01", "2020-02-01"],
        "episode_start_date": ["2020-01-01", "2020-02-01"],
        "episode_end_date": ["2020-01-02", "2020-02-02"],
        "clinical_visit": [True, True],
    })
    mpath = p.with_name("_10_medication_documentation.py")
    spec = iu.spec_from_file_location("step10_med_test_order", mpath)
    adapter = iu.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    counts = clinical[["patient_id", "clinical_episode_id"]].copy()
    for col in adapter.COUNT_COLUMNS:
        counts[col] = pd.Series([3, 1], dtype="Int64")
    counts[adapter.FLAG] = [True, True]
    counts[adapter.NAME_FIELD] = pd.Series(["hydroxychloroquine", "cevimeline"],dtype="string")
    counts = counts.iloc[::-1].reset_index(drop=True)
    original = iu.spec_from_file_location
    class Loader:
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            module.COUNT_COLUMNS = adapter.COUNT_COLUMNS
            module.FLAG = adapter.FLAG
            module.NAME_FIELD = adapter.NAME_FIELD
            module.write_protected_detail = lambda details, audit_root: tmp_path / "synthetic_detail.parquet"
            module.load_and_aggregate = lambda *args, **kwargs: (
                counts, {"version": "synthetic"}, pd.DataFrame())
    def factory(name, path):
        return iu.spec_from_loader(name, Loader()) if str(path) == str(mpath) else original(name, path)
    monkeypatch.setattr(iu, "spec_from_file_location", factory)
    context = clinical.iloc[::-1].copy()
    context["patient_id"] = context.patient_id.astype(object)
    context["clinical_episode_id"] = context.clinical_episode_id.astype(object)
    build = m.CuratedBuild(clinical, context, pd.DataFrame(), pd.DataFrame(), [], [], [], pd.DataFrame())
    result, qc = m._integrate_medication_context(
        build, audit_root=tmp_path, clinical_spine_path=tmp_path / "spine.parquet")
    main_counts = result.analytic.set_index("clinical_episode_id")["meddoc__n_documentation_records"]
    assert main_counts.loc["e1"] == 3
    assert main_counts.loc["e2"] == 1
    assert len(result.analytic) == len(result.context) == 2
    assert qc["n_descriptive_medication_columns_added_to_master"] == 12

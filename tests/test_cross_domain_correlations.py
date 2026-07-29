"""Focused synthetic checks for script 22."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).parents[1] / "src/block_B/22_cross_domain_correlations.py"
SPEC = importlib.util.spec_from_file_location("cross_domain", PATH)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_duplicate_patient_ids_raise(tmp_path):
    path = tmp_path / "baseline.parquet"
    pd.DataFrame({"patient_id": [1, 1], "pop_status": ["Pop1", "Pop2"]}).to_parquet(path)
    with pytest.raises(ValueError, match="unique"):
        mod.load_baseline(path)


def test_pairwise_complete_n_and_uninterpretable_serology():
    data = pd.DataFrame({"essdai_total": [1, 2, 3, np.nan], "esspri_total": [1, np.nan, 3, 4],
                         "anti_ro_pos": [True, False, True, False],
                         "anti_ro_interpretable": [True, False, True, True]})
    result = mod.build_correlations(data, min_pair_n=2)
    pair = result.query("variable_x == 'essdai_total' and variable_y == 'esspri_total'").iloc[0]
    assert pair.n_complete_pairs == 2 and pair.status == "estimated"
    sero = result.query("variable_x == 'essdai_total' and variable_y == 'anti_ro_pos'").iloc[0]
    assert sero.n_complete_pairs == 2


def test_ssa_unknown_is_not_negative():
    data = pd.DataFrame({"anti_ro_interpretable": [True, True, False, pd.NA],
                         "anti_ro_pos": [True, False, False, False]})
    assert mod.define_ssa_group(data).ssa_group.tolist() == ["SSA_positive", "SSA_negative", "SSA_unknown", "SSA_unknown"]


def model_frame(n=80):
    rng = np.random.default_rng(4)
    return pd.DataFrame({"sf36_pcs": rng.normal(size=n), "essdai_total": rng.normal(size=n),
        "esspri_total": rng.normal(size=n), "profad_total": rng.normal(size=n),
        "fibromyalgia": np.tile([0, 1], n//2), "depression": np.tile([0, 1, 0, 0], n//4),
        "age_baseline": rng.normal(size=n), "protocol_group": np.tile(["A", "B"], n//2),
        "time_since_diagnosis_years": rng.normal(size=n), "pop_status": "Pop1",
        "sf36_mcs": rng.normal(size=n), "sf36_physical_functioning": rng.normal(size=n)})


def test_pcs_prespecified_formula_excludes_forbidden_variables():
    predictors = ["essdai_total", "esspri_total", "profad_total", "fibromyalgia", "depression",
                  "age_baseline", "protocol_group", "time_since_diagnosis_years"]
    result = mod.fit_ols_model(model_frame(), "sf36_pcs", predictors, "pcs_primary", min_n=50)
    formula = result.formula.iloc[0]
    assert result.model_status.eq("fitted").all()
    assert "pop_status" not in formula and "sf36_mcs" not in formula and "sf36_physical" not in formula


def test_esspri_serology_filter_and_sparse_skip_do_not_block_correlations():
    data = model_frame()
    data["anti_ro_pos"] = np.tile([0, 1], 40); data["anti_ro_interpretable"] = True
    data.loc[:9, "anti_ro_interpretable"] = False
    filtered = data.copy(); filtered.loc[~filtered.anti_ro_interpretable, "anti_ro_pos"] = np.nan
    predictors = ["anti_ro_pos", "fibromyalgia", "depression", "age_baseline", "protocol_group", "time_since_diagnosis_years"]
    fit = mod.fit_ols_model(filtered, "esspri_total", predictors, "esspri_primary", min_n=50)
    assert fit.n_complete.iloc[0] == 70
    assert not any(x in fit.formula.iloc[0] for x in ["esspri_dryness", "esspri_fatigue", "esspri_pain"])
    sparse = mod.fit_ols_model(data.iloc[:4], "sf36_pcs", ["essdai_total"], "pcs_primary", min_n=50)
    assert sparse.model_status.iloc[0] == "skipped_insufficient_data"
    assert not mod.build_correlations(data, min_pair_n=2).empty

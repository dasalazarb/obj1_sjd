import importlib.util
import sys
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "src" / "block_A" / "07_comorbidities.py"
SPEC = importlib.util.spec_from_file_location("comorbidities", MODULE_PATH)
comorbidities = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = comorbidities
SPEC.loader.exec_module(comorbidities)


def _prevalence_frame() -> pd.DataFrame:
    data = {name: pd.Series([True, False, pd.NA], dtype="boolean")
            for name in comorbidities.CONDITION_NAMES}
    data["baseline_pop"] = ["Pop1", "Pop1", "Pop1"]
    return pd.DataFrame(data)


def test_overall_prevalence_codes_empty_condition_as_absent():
    result = comorbidities.summarize_overall_prevalence(_prevalence_frame())

    fibromyalgia = result.loc[result["condition"].eq("fibromyalgia")].iloc[0]
    assert fibromyalgia["n_total_cohort"] == 3
    assert fibromyalgia["n_evaluable"] == 3
    assert fibromyalgia["n_positive"] == 1
    assert fibromyalgia["n_negative"] == 2
    assert fibromyalgia["n_missing"] == 0
    assert fibromyalgia["pct_total_cohort"] == fibromyalgia["pct_among_evaluable"]


def test_pop_prevalence_uses_full_pop_denominator_for_empty_condition():
    result = comorbidities.summarize_prevalence_by_pop(
        _prevalence_frame(), replicates=10, seed=1
    )

    fibromyalgia = result.loc[result["condition"].eq("fibromyalgia")].iloc[0]
    assert fibromyalgia["n_pop1"] == 1
    assert fibromyalgia["N_pop1"] == 3
    assert fibromyalgia["pct_pop1"] == 100 / 3

"""Focused tests for the topological phenotyping study."""
import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "src/studies/graph/01_run_topological_phenotyping.py"
SPEC = importlib.util.spec_from_file_location("topological_phenotyping", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_branch_characterization_calculates_boolean_quartiles():
    baseline = pd.DataFrame({
        "patient_id": ["1", "2", "3", "4"],
        "boolean_feature": [False, False, True, True],
    })
    membership = pd.DataFrame({
        "patient_id": ["1", "2", "3", "4"],
        "mapper_covered": [True, True, True, True],
        "hard_branch": [0, 0, 0, 0],
    })

    result = MODULE.branch_characterization(
        baseline, membership, ["boolean_feature"], {})

    row = result.iloc[0]
    assert row["n"] == 4
    assert row["median"] == 0.5
    assert row["q1"] == 0.0
    assert row["q3"] == 1.0
    assert row["mean"] == 0.5

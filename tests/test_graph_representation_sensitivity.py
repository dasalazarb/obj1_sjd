"""Focused tests for representation-sensitivity comparison contracts."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (Path(__file__).parents[1] / "src/studies/graph" /
          "02_graph_representation_sensitivity.py")
SPEC = importlib.util.spec_from_file_location("graph_representation_sensitivity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_strict_coverage_preserves_full_one_hot_and_missing_values():
    values = {
        "category__a": pd.Series([1.0, 0.0, np.nan, 1.0]),
        "category__b": pd.Series([0.0, 1.0, np.nan, 0.0]),
        "continuous": pd.Series([1.0, 2.0, 3.0, 4.0]),
    }
    families = {name: "CLINICAL" for name in values}
    config = {"feature_selection": {"lab_minimum_baseline_coverage": .7,
                                     "pro_minimum_baseline_coverage": .6,
                                     "clinical_minimum_baseline_coverage": .7},
              "redundancy": {"conceptual_pairs": [], "spearman_abs_threshold": .95}}

    selected, audit, _ = MODULE.select_features(
        values, families, "S1", config, pd.DataFrame())

    assert set(selected) == set(values)
    assert selected["category__a"].isna().sum() == 1
    assert selected["category__b"].isna().sum() == 1
    assert audit.included_after_coverage_filter.all()


def _membership(scenario, assignments, supported=None):
    supported = supported or [True] * len(assignments)
    return pd.DataFrame({"scenario": scenario, "patient_id": ["a", "b", "c", "d"],
        "mapper_covered": True, "community_membership_tie": False,
        "hard_community": pd.array(assignments, dtype="Int64"),
        "community_supported": supported})


def test_cross_scenario_overlap_and_ari_use_patient_assignments():
    memberships = {
        "S0": _membership("S0", [0, 0, 1, 1]),
        "S1": _membership("S1", [0, 0, 1, 1]),
        "S2": _membership("S2", [0, 1, 1, 1]),
        "S3": _membership("S3", [0, 0, 1, 1], [True, False, True, True]),
    }
    overlap, stable, ari = MODULE.cross_scenario_tables(
        memberships, {scenario: 0 for scenario in memberships})

    s0_s1 = stable.query("scenario_a == 'S0' and scenario_b == 'S1'").iloc[0]
    assert s0_s1.jaccard == 1.0
    assert s0_s1.overlap_coefficient == 1.0
    assert len(overlap) == 24
    assert ari.query("scenario_a == 'S0' and scenario_b == 'S1'").iloc[0].ari == 1.0
    assert ari.query("scenario_a == 'S0' and scenario_b == 'S3'").iloc[0].n_patients_used == 3

"""Focused tests for the Pharma 00 -> Graph feature contract."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "src/studies/graph/01_run_topological_phenotyping.py"
SPEC = importlib.util.spec_from_file_location("graph_topological_phenotyping", SCRIPT)
graph = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(graph)


def _candidate(lab: str, status: str, hint: str, primary: str, **columns: str) -> dict:
    return {
        "lab": lab, "graph_candidate_status": status, "graph_encoding_hint": hint,
        "primary_result_source": primary, "value_column": "", "text_column": "",
        "reference_status_column": "", **columns,
    }


def test_candidate_contract_encodes_baseline_labs_and_preserves_missingness() -> None:
    baseline = pd.DataFrame({
        "numeric__value": [1.0, 2.0, np.nan, 4.0],
        "antibody__reference_status": [" Positive ", "negative", None, "POSITIVE"],
        "three_way__text": ["high", "normal", "low", None],
        "sf36_pcs": [30, 40, 50, 60], "sf36_mcs": [31, 41, 51, 61],
        "profad_total": [1, 2, 3, 4], "mdafs_global": [4, 3, 2, 1],
        "overlap_baseline": [0, 1, 0, 1], "age_dx": [20, 30, 40, 50],
    })
    candidates = pd.DataFrame([
        _candidate("numeric", "include_numeric", "continuous", "value_column",
                   value_column="numeric__value"),
        _candidate("antibody", "include_categorical", "binary", "reference_status_column",
                   reference_status_column="antibody__reference_status"),
        _candidate("three_way", "include_categorical", "categorical_one_hot", "text_column",
                   text_column="three_way__text"),
        _candidate("future_lab", "include_numeric", "continuous", "value_column",
                   value_column="future_result__value"),
        _candidate("review", "review_before_graph", "review", "value_column",
                   value_column="numeric__value"),
    ])

    manifest, values, _, audit = graph.build_feature_manifest(baseline, candidates, .5)

    assert {"numeric__value", "antibody__binary", "three_way__cat_high",
            "three_way__cat_low", "three_way__cat_normal"}.issubset(values)
    assert np.isnan(values["antibody__binary"].iloc[2])
    assert values["three_way__cat_high"].iloc[3:].isna().all()
    assert audit.loc[audit.lab.eq("future_lab"), "exclusion_reason"].item() == "temporal_leakage_risk"
    assert not audit.loc[audit.lab.eq("review"), "included_in_graph"].item()
    assert manifest.loc[manifest.feature.eq("antibody__binary"), "source"].item() == "pharma_00_graph_lab_candidates"


def test_binary_hint_rejects_three_real_categories() -> None:
    baseline = pd.DataFrame({"lab__text": ["high", "low", "normal"],
                             **{name: [1, 2, 3] for name in (*graph.PRO_FEATURES, *graph.OVERLAP_FEATURES, "age_dx")}})
    candidates = pd.DataFrame([
        _candidate("lab", "include_categorical", "binary", "text_column",
                   text_column="lab__text")])
    _, values, _, audit = graph.build_feature_manifest(baseline, candidates, .5)
    assert not any(feature.startswith("lab__") for feature in values)
    assert audit.exclusion_reason.item() == "invalid_binary_clinical_categories"


@pytest.mark.parametrize("lab, expected", [
    (" LAB__BUN ", "bun"),
    ("bun", "bun"),
    ("lab__acute_care_panel__anion_gap_blood", "acute_care_panel__anion_gap_blood"),
    ("lab__urinalysis_includes_microscopic__hyalin_casts_urinalysis",
     "urinalysis_includes_microscopic__hyalin_casts_urinalysis"),
    ("lab__lab__nested", "lab__nested"),
])
def test_canonical_lab_name_removes_only_the_leading_namespace(lab, expected):
    assert graph.canonical_lab_name(lab) == expected


def test_public_lab_keys_resolve_characterization_extras_without_renaming_features():
    labs = ["protein_total", "mcv", "bun", "lymphocyte_count",
            "acute_care_panel__anion_gap_blood"]
    baseline = pd.DataFrame({"patient_id": ["p1", "p2", "p3", "p4"],
                             **{f"lab__{lab}__value": [1., 2., 3., 4.] for lab in labs}})
    candidates = pd.DataFrame([
        _candidate(f"lab__{lab}", "include_numeric", "continuous", "value_column",
                   value_column=f"lab__{lab}__value") for lab in labs])
    manifest, values, names, _ = graph.build_feature_manifest(baseline, candidates, .7)
    assert names == {lab: f"lab__{lab}__value" for lab in labs}
    assert set(values) == set(baseline) - {"patient_id"}
    assert manifest.loc[manifest.included, "feature"].tolist() == list(values)
    membership = pd.DataFrame({"patient_id": baseline.patient_id, "mapper_covered": True,
                               "community_membership_tie": False, "community_supported": True,
                               "hard_community": [0, 0, 1, 1]})
    result = graph.community_characterization(baseline, membership, [], names)
    assert set(result.feature) == {f"lab__{lab}__value" for lab in labs[:4]}
    assert result.loc[(result.hard_community == 0) & result.feature.eq("lab__bun__value"),
                      "median"].item() == 1.5

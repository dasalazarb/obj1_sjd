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


from graph_contract_fixtures import candidate


def _candidate(lab, status, hint, primary, **columns):
    # Historical test helper now supplies the actual public v2 contract.
    return candidate(lab, status, **columns)


def test_candidate_contract_encodes_baseline_labs_and_preserves_missingness():
    baseline = pd.DataFrame({
        "lab__numeric__value": [1., 2., np.nan, 4.],
        "lab__antibody__text": [" Positive ", "negative", None, "POSITIVE"],
        "pro__sf36_pcs": [30, 40, 50, 60]})
    candidates = pd.DataFrame([
        candidate("lab__numeric"), candidate("lab__antibody", "include_categorical"),
        candidate("lab__review", "review_before_graph")])
    manifest, values, _, audit = graph._build_legacy_feature_manifest(baseline, candidates, .5)
    assert {"lab__numeric__value", "lab__antibody__binary"} <= set(values)
    assert np.isnan(values["lab__antibody__binary"].iloc[2])
    assert not audit.loc[audit.lab.eq("lab__review"), "included_in_graph"].item()
    assert manifest.loc[manifest.feature.eq("lab__antibody__binary"), "source"].item() == "pharma_00_graph_lab_candidates"


def test_binary_hint_rejects_three_real_categories():
    baseline = pd.DataFrame({"lab__assay__text": ["high", "low", "normal"]})
    with pytest.raises(graph.LAB.LabContractError, match="CONTRACT_MISMATCH"):
        graph._build_legacy_feature_manifest(baseline, pd.DataFrame([candidate("lab__assay", "include_categorical")]), .5)


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
    manifest, values, names, _ = graph._build_legacy_feature_manifest(baseline, candidates, .7)
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

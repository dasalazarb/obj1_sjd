"""Focused regression tests for Pharma step-03 traceability and summaries."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "src/studies/pharma/03_pharma_multistate_biomarkers.py"
SPEC = importlib.util.spec_from_file_location("pharma_multistate_biomarkers", SCRIPT)
multistate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multistate)


def test_metadata_requires_exact_transition_match():
    source = pd.DataFrame({
        "exposure": ["BUN", "MCV"],
        "from_pop": ["Pop1", "Pop2"],
        "to_pop": ["Pop3", "Pop1"],
        "adjusted_estimate": [1.2, 0.8],
    })

    exact = multistate._metadata_for_transition(source, "BUN", "Pop1", "Pop3")

    assert exact.adjusted_estimate == 1.2
    assert multistate._metadata_for_transition(source, "BUN", "Pop2", "Pop1") is None


def test_metadata_accepts_normalized_transition_pair():
    source = pd.DataFrame({
        "exposure": ["BUN"], "transition_pair": ["Pop1 -> Pop3"],
        "adjusted_estimate": [1.2],
    })

    assert multistate._metadata_for_transition(source, "BUN", "Pop1", "Pop3") is not None
    assert multistate._metadata_for_transition(source, "BUN", "Pop2", "Pop3") is None


def test_fdr_is_calculated_separately_by_model_version():
    models = pd.DataFrame({
        "model_version": ["same_sample_clustered"] * 2 + ["age_adjusted_clustered"] * 2,
        "model_status": ["success"] * 4,
        "p_value": [0.01, 0.04, 0.03, 0.04],
        "q_value": [np.nan] * 4,
    })

    multistate.apply_fdr_by_model(models)

    np.testing.assert_allclose(models.q_value, [0.02, 0.04, 0.04, 0.04])


def test_summary_counts_primary_comparisons_and_unique_effects_once():
    candidates = pd.DataFrame({"exposure": ["protein_total"], "eligible_for_03": [True]})
    long_data = pd.DataFrame({
        "biomarker": ["protein_total"] * 2, "patient_id": [1, 1],
        "from_clinical_episode_id": ["a", "b"], "to_clinical_episode_id": ["b", "c"],
    })
    support = pd.DataFrame({"biomarker": ["protein_total"], "supported_for_model": [True]})
    models = pd.DataFrame({
        "biomarker": ["protein_total"] * 3,
        "transition_id": ["Pop2→Pop1", "Pop2→Pop1", "Pop2→Pop3"],
        "model_version": ["same_sample_clustered", "age_adjusted_clustered",
                          "same_sample_clustered"],
        "model_status": ["success"] * 3,
        "interpretability_status": ["interpretable"] * 3,
        "direction_consistent_with_02": [True, True, np.nan],
    })

    row = multistate.summarize_biomarkers(candidates, long_data, support, models).iloc[0]

    assert row.n_estimable_effects == 2
    assert row.n_directionally_comparable_with_02 == 1
    assert row.n_directionally_consistent_with_02 == 1

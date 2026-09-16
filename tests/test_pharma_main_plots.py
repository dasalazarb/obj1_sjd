"""Regression tests for Pharma main-analysis plots."""
import importlib.util
from decimal import Decimal
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "src/studies/pharma/01_run_pharma_main.py"
SPEC = importlib.util.spec_from_file_location("pharma_main", SCRIPT)
pharma_main = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pharma_main)


def test_heatmap_converts_object_values_to_float(tmp_path):
    table = pd.DataFrame({
        "lab": ["CRP", "ESR"],
        "transition_pair": ["Pop1->Pop2", "Pop1->Pop2"],
        "median_standardized_change": [Decimal("0.5"), Decimal("-0.25")],
    })
    output = tmp_path / "heatmap.pdf"

    made = pharma_main._plot_heatmap(
        table, "lab", "median_standardized_change", output, "Lab changes"
    )

    assert made
    assert output.is_file()


def test_domain_resolution_prefers_eg_ordinal_score_and_keeps_active_fallback():
    master = pd.DataFrame(columns=[
        "eg_articular_ordinal_score",
        "essdai_articular_score",
        "eg_articular_active",
        "eg_pulmonary_active",
    ])

    resolved, domains = pharma_main._resolve(master)

    assert resolved["essdai_domain_articular"] == (
        "essdai_domain", "eg_articular_ordinal_score"
    )
    assert resolved["essdai_domain_pulmonary"] == (
        "essdai_domain", "eg_pulmonary_active"
    )
    assert domains == ["eg_articular_ordinal_score", "eg_pulmonary_active"]


def test_domain_ordinal_delta_preserves_score_levels(monkeypatch):
    source = "eg_articular_ordinal_score"
    enriched = pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "from_clinical_episode_id": ["a", "c"],
        "to_clinical_episode_id": ["b", "d"],
        "from_clinical_anchor_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "to_clinical_anchor_date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
        "interval_years": [1.0, 1.0],
        "from_pop": ["Pop1", "Pop1"],
        "to_pop": ["Pop2", "Pop2"],
        f"from_{source}": ["1", "3"],
        f"to_{source}": ["3", "2"],
    })
    monkeypatch.setattr(
        pharma_main, "enrich_transition_intervals",
        lambda intervals, master, from_columns, to_columns: enriched.copy(),
    )

    analytic = pharma_main.build_analytic(
        pd.DataFrame(), pd.DataFrame(),
        {"essdai_domain_articular": ("essdai_domain", source)},
    )

    assert analytic[f"delta_{source}"].tolist() == [2, -1]


def test_ordinal_transition_association_declares_per_level_effect_scale():
    feature = "essdai_domain_articular"
    source = "eg_articular_ordinal_score"
    frame = pd.DataFrame({
        "patient_id": ["p1", "p2", "p3"],
        "from_pop": ["Pop1"] * 3,
        "to_pop": ["Pop1", "Pop2", "Pop3"],
        "interval_years": [1.0] * 3,
        f"from_{source}": [0, 1, 3],
    })
    availability = pd.DataFrame({
        "feature": [feature],
        "available_for_transition_model": [True],
    })

    result = pharma_main.transition_associations(
        frame, availability, {feature: ("essdai_domain", source)},
        {"minimum_counts": {"events_for_multivariable_model": 10}},
    )

    assert set(result.predictor_type) == {"ordinal"}
    assert set(result.effect_scale) == {"per_1_level_increase"}

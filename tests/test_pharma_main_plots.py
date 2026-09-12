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

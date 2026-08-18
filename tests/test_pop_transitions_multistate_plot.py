"""Regression tests for the continuous-time multi-state diagram API."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


matplotlib = pytest.importorskip("matplotlib")
np = pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("scipy")
matplotlib.use("Agg")


def _load_transition_module():
    script = Path(__file__).parents[1] / "src" / "block_A" / "10_pop_transitions.py"
    spec = importlib.util.spec_from_file_location("pop_transitions", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multistate_diagram_accepts_legacy_and_annotated_calls(tmp_path):
    """The original ``(q, path)`` API and count annotations both remain valid."""
    module = _load_transition_module()
    q = np.array([
        [-0.30, 0.10, 0.20],
        [0.15, -0.25, 0.10],
        [0.05, 0.15, -0.20],
    ])

    legacy_path = tmp_path / "legacy.pdf"
    module.plot_multistate_model(q, legacy_path)
    assert legacy_path.stat().st_size > 0

    annotated_path = tmp_path / "annotated.pdf"
    counts = {
        f"{origin} -> {destination}": 4
        for origin in module.MODEL_STATES
        for destination in module.MODEL_STATES
        if origin != destination
    }
    module.plot_multistate_model(q, annotated_path, observed_counts=counts)
    assert annotated_path.stat().st_size > 0

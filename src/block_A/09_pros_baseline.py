#!/usr/bin/env python3
"""Deprecated compatibility entry point for the canonical episode-level PRO pipeline.

Use ``09_pros_longitudinal.py``. Baseline is now selected exclusively with
``is_clinical_baseline`` from its canonical episode-level dataset.
"""
from __future__ import annotations
import runpy
from pathlib import Path

# Kept as an import compatibility surface for downstream code while scoring
# remains defined only in the neutral derivation module.
from src.derivations.pro_scoring import score_all_pros

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("09_pros_longitudinal.py")), run_name="__main__")

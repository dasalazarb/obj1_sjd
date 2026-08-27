#!/usr/bin/env python3
"""Deprecated compatibility entry point; use ``09_pros_longitudinal.py``."""
from __future__ import annotations
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("09_pros_longitudinal.py")), run_name="__main__")

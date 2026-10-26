"""Pinned pre-optimization implementation for synthetic equivalence/benchmarks."""
import importlib.util
from pathlib import Path
import subprocess
import sys

REFERENCE_COMMIT = "c39d8c4cd6bfb49d8fa16d4b03db6a463f9617e5"
ROOT = Path(__file__).resolve().parents[1]


def reference_module(name, filename):
    path = ROOT/"src/block_A"/filename
    source = subprocess.check_output(["git", "show", f"{REFERENCE_COMMIT}:src/block_A/{filename}"], cwd=ROOT, text=True)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def reference():
    analysis = reference_module("step15_reference_analysis", "secondary_trajectory_analysis.py")
    figures = reference_module("step15_reference_figures", "secondary_trajectory_figures.py")
    cli = reference_module("step15_reference_cli", "15_secondary_trajectories_by_essdai_pattern.py")
    figures.summarize = analysis.summarize
    for name in ("attach_context", "observations", "summarize", "support_table", "validate_inputs"):
        setattr(cli, name, getattr(analysis, name))
    cli.plot_variable = figures.plot_variable
    return analysis, figures, cli

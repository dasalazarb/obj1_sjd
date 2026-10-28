"""Synthetic-only Step 17 longitudinal fixtures."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from secondary_comparison_fixtures import synthetic
from src.block_A.temporal_spline_config import SplineConfig
from src.block_A.temporal_spline_feasibility import evaluate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('step17', ROOT/'src/block_A/17_nonlinear_temporal_heterogeneity.py')
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


def fixture(shape='linear', n=22, binary=False):
    d, groups, variable = synthetic(n=n, slopes=(0., 0., 0., 0.), binary=binary)
    # Dense real dates; identical design per group, patient intercepts preserved.
    parts = []
    rng = np.random.default_rng(1704)
    patterns = list(groups.observed_pattern.unique())
    for pid, original in d.groupby('patient_id', sort=False):
        g = original.observed_pattern.iloc[0]
        group = patterns.index(g)
        offset = rng.normal(0, .9)
        for t in np.arange(0, 3.01, .25):
            row = original.iloc[0].to_dict()
            curve = .25*t
            if shape == 'common':
                curve += 1.5*(t-1.5)**2
            elif shape == 'different':
                curve += (0., 2., -2., 1.)[group]*(t-1.5)**2
            elif shape == 'crossing':
                curve += (0., 1., -1., .5)[group]*(t-1.5)
            elif shape == 'early_late':
                curve += (0., 3*(1-np.exp(-2*t)), 3*np.maximum(t-1.5, 0)**2, -t)[group]
            value = 5+offset+curve+rng.normal(0, .25)
            if binary:
                from scipy.special import expit
                value = rng.binomial(1, expit(-.5+.2*t))
            row.update(time=t, value=value, n_analysis_dates=13, n_variable_dates=13,
                variable_measurement_date=pd.Timestamp('2020-01-01')+pd.to_timedelta(round(t*365.25), unit='D'),
                time_from_clinical_baseline_years=t, time_from_essdai_index_years=t)
            parts.append(row)
    return pd.DataFrame(parts), groups, variable


def evaluated(shape='different', n=22, binary=False):
    d, groups, spec = fixture(shape, n, binary)
    cfg = SplineConfig(simulations=600)
    return evaluate(d, groups, spec, cfg, 'clinical_baseline', 3, 3), spec, cfg


def args_for(upstream, output, *extra):
    return cli.parse_args(['--integrated', str(upstream['integrated']), '--baseline', str(upstream['baseline']),
        '--patterns', str(upstream['patterns']), '--patterns-manifest', str(upstream['manifest']),
        '--output-root', str(output), '--input-origin', 'synthetic_validation', '--no-progress',
        '--data-cutoff', '2025-01-01', '--figures', 'none', '--simulations', '600', *extra])

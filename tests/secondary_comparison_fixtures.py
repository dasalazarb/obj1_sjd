"""Synthetic, nonclinical repeated data for Step 16 software validation."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from src.block_A.observed_patterns_v2_figures import LABELS
from src.block_A.secondary_trajectory_catalog import Variable

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('secondary_step16', ROOT/'src/block_A/16_compare_secondary_trajectories.py')
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


def synthetic(n=30, slopes=(0., .6, -.6, .3), binary=False, unequal=False):
    rng = np.random.default_rng(1601)
    rows, groups = [], []
    for index, pattern in enumerate(LABELS):
        for patient in range(n):
            pid = f'fake_group{index}_patient{patient:03d}'
            offset = rng.normal(0, .7)
            count = 3+patient % 3 if unequal else 5
            times = np.linspace(0, 4, count)
            values = 5+offset+slopes[index]*times+rng.normal(0, .15, count)
            if binary:
                probability = 1/(1+np.exp(-(-.5+slopes[index]*times)))
                values = rng.binomial(1, probability)
            groups.append(dict(patient_id=pid, observed_pattern=pattern, threshold_abs=3,
                n_distinct_essdai_dates=5, trajectory_evidence_level='three_or_more_points_pattern',
                essdai_index_source='official_baseline', essdai_index_date=pd.Timestamp('2020-01-01')))
            for j, (t, v) in enumerate(zip(times, values)):
                measurement_date = pd.Timestamp('2020-01-01')+pd.to_timedelta(int(round(t*365.25)), unit='D')
                rows.append(dict(patient_id=pid, clinical_episode_id=f'e{j}', clinical_baseline_episode_id='e0',
                    observed_pattern=pattern, threshold_abs=3, n_distinct_essdai_dates=5,
                    trajectory_evidence_level='three_or_more_points_pattern', essdai_index_source='official_baseline',
                    essdai_index_date=pd.Timestamp('2020-01-01'), clinical_baseline_date=pd.Timestamp('2020-01-01'),
                    variable_measurement_date=measurement_date, reference_date=pd.Timestamp('2020-01-01'),
                    reference_source='official_clinical_baseline', reference_delay_days=0,
                    reference_value=values[0], value=v, delta=v-values[0], is_delta_followup=j>0,
                    n_variable_dates=count, n_analysis_dates=count, time=t,
                    time_from_clinical_baseline_years=t, time_from_essdai_index_years=t,
                    time_since_variable_reference_years=t, window=int(t), source_protocol='synthetic_A'))
    variable = Variable('synthetic_outcome', 'esspri', 'Synthetic outcome', 'esspri__total', 'synthetic_fixture',
                        unit='points' if not binary else '% evaluable', type='binary' if binary else 'numeric')
    return pd.DataFrame(rows), pd.DataFrame(groups), variable


def args_for(upstream, output, *extra):
    return cli.parse_args(['--integrated', str(upstream['integrated']), '--baseline', str(upstream['baseline']),
        '--patterns', str(upstream['patterns']), '--patterns-manifest', str(upstream['manifest']),
        '--output-root', str(output), '--input-origin', 'synthetic_validation', '--no-progress',
        '--data-cutoff', '2025-01-01', *extra])

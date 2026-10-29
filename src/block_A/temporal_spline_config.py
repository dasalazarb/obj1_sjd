"""Step 17 prespecification, independent of Step 16 settings."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from src.block_A.secondary_comparison_config import ComparisonConfig, PRIORITY

VERSION = 'temporal-spline-v2'
STEM = '17_nonlinear_temporal_heterogeneity'
C_SUPPORT_POLICY = 'C_RELAXED_SUPPORT_V1'
ANALYSIS_ROLE = 'exploratory_posthoc_support_revision'


@dataclass(frozen=True)
class SplineConfig(ComparisonConfig):
    version: str = VERSION
    priority_variables: tuple = PRIORITY + ('iga', 'igm')
    max_primary_years: float = 3.
    grid_step: float = .25
    minimum_three_date_patients: int = 5
    c_support_policy: str = C_SUPPORT_POLICY
    minimum_c_supported_regions: int = 2
    minimum_c_supported_annual_windows: int = 2
    minimum_c_supported_grid_points: int = 4
    minimum_c_supported_grid_span_years: float = 1.
    simulations: int = 4000
    bootstrap_replicates: int = 0
    seed: int = 1701

    def __post_init__(self):
        # Validate inherited limits without changing the Step 16 contract/version.
        inherited = {k: v for k, v in asdict(self).items() if k in ComparisonConfig.__dataclass_fields__}
        inherited['version'] = ComparisonConfig().version
        ComparisonConfig(**inherited)
        if self.version != VERSION:
            raise ValueError('unsupported Step 17 specification')
        for name in ('max_primary_years', 'grid_step', 'minimum_c_supported_grid_span_years'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name+' must be positive and finite')
        for name in ('minimum_three_date_patients', 'minimum_c_supported_regions',
                     'minimum_c_supported_annual_windows', 'minimum_c_supported_grid_points',
                     'simulations', 'bootstrap_replicates'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(name+' must be a nonnegative integer')
        if self.simulations < 500:
            raise ValueError('require >=500 simulations')
        # This named policy is global and frozen; another support specification
        # requires a new policy/version, never a variable-dependent override.
        if (self.c_support_policy != C_SUPPORT_POLICY or
                (self.minimum_c_supported_regions, self.minimum_c_supported_annual_windows,
                 self.minimum_c_supported_grid_points, self.minimum_c_supported_grid_span_years,
                 self.minimum_prediction_patients, self.minimum_three_date_patients) != (2, 2, 4, 1., 5, 5)):
            raise ValueError('support thresholds must match C_RELAXED_SUPPORT_V1')
        if self.grid_step > self.max_primary_years/2:
            raise ValueError('grid too coarse for three support regions')
        if 0 < self.bootstrap_replicates < 100:
            raise ValueError('bootstrap requires at least 100 patient replicates')


def load_config(path=None, **overrides):
    values = json.loads(Path(path).read_text()) if path else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    return SplineConfig(**values)

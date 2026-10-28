"""Step 17 prespecification, independent of Step 16 settings."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from src.block_A.secondary_comparison_config import ComparisonConfig, PRIORITY

VERSION = 'temporal-spline-v1'
STEM = '17_nonlinear_temporal_heterogeneity'


@dataclass(frozen=True)
class SplineConfig(ComparisonConfig):
    version: str = VERSION
    priority_variables: tuple = PRIORITY + ('iga', 'igm')
    max_primary_years: float = 3.
    grid_step: float = .25
    minimum_three_date_patients: int = 5
    minimum_supported_regions: int = 3
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
        for name in ('max_primary_years', 'grid_step'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name+' must be positive and finite')
        for name in ('minimum_three_date_patients', 'minimum_supported_regions', 'simulations', 'bootstrap_replicates'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(name+' must be a nonnegative integer')
        if self.minimum_three_date_patients < 2 or self.minimum_supported_regions != 3 or self.simulations < 500:
            raise ValueError('require >=2 three-date patients, three regions and >=500 simulations')
        if self.grid_step > self.max_primary_years/2:
            raise ValueError('grid too coarse for three support regions')
        if 0 < self.bootstrap_replicates < 100:
            raise ValueError('bootstrap requires at least 100 patient replicates')


def load_config(path=None, **overrides):
    values = json.loads(Path(path).read_text()) if path else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    return SplineConfig(**values)

"""Versioned prespecification; support thresholds are not power guarantees."""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

VERSION = 'secondary-comparison-v1'
STEM = '16_compare_secondary_trajectories'
PRIORITY = (
    'esspri_total', 'esspri_dryness', 'esspri_fatigue', 'esspri_pain',
    'sf36_pcs', 'sf36_mcs', 'complement_c3', 'complement_c4', 'igg', 'esr',
    'crp', 'crp_high_sensitivity', 'ocular_schirmer_right', 'ocular_schirmer_left',
    'ocular_tbut_right', 'ocular_tbut_left', 'ocular_oxford_right', 'ocular_oxford_left',
    'ocular_van_bijsterveld_right', 'ocular_van_bijsterveld_left',
    'salivary_flow_unstimulated', 'salivary_flow_stimulated',
)


@dataclass(frozen=True)
class ComparisonConfig:
    version: str = VERSION
    minimum_patients_per_group: int = 10
    minimum_repeated_per_group: int = 8
    minimum_distinct_times: int = 3
    minimum_observed_span_years: float = 1.
    minimum_events_per_group: int = 5
    minimum_nonevents_per_group: int = 5
    minimum_prediction_patients: int = 5
    prediction_window_years: float = 1.
    report_times: tuple = (1., 2., 3., 5.)
    window_years: float = 1.
    minimum_variability_dates: int = 3
    minimum_variability_span_years: float = 1.
    auc_horizon: tuple = (0., 3.)
    minimum_auc_dates: int = 3
    maximum_auc_gap_years: float = 2.
    priority_variables: tuple = PRIORITY
    # Explicit variable -> family mapping overrides the default domain family.
    families: dict = field(default_factory=dict)
    random_intercept_boundary: float = 1e-8
    maximum_design_condition: float = 1e8
    maximum_residual_skewness: float = 3.
    maximum_residual_variance_ratio: float = 20.
    maximum_logistic_coefficient: float = 15.
    seed: int = 1601

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError('unsupported comparison config version')
        for name in ('minimum_patients_per_group', 'minimum_repeated_per_group',
                     'minimum_distinct_times', 'minimum_events_per_group',
                     'minimum_nonevents_per_group', 'minimum_prediction_patients',
                     'minimum_variability_dates', 'minimum_auc_dates'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(name + ' must be an integer >=2')
        for name in ('minimum_observed_span_years', 'window_years',
                     'minimum_variability_span_years', 'prediction_window_years',
                     'maximum_auc_gap_years', 'random_intercept_boundary',
                     'maximum_design_condition', 'maximum_residual_skewness',
                     'maximum_residual_variance_ratio', 'maximum_logistic_coefficient'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + ' must be finite and positive')
        if (len(self.auc_horizon) != 2 or not all(math.isfinite(t) for t in self.auc_horizon)
                or not 0 <= self.auc_horizon[0] < self.auc_horizon[1]):
            raise ValueError('invalid AUC horizon')
        if not self.report_times or not all(math.isfinite(t) and t >= 0 for t in self.report_times):
            raise ValueError('invalid report times')
        if (not all(isinstance(v, str) for v in self.priority_variables)
                or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in self.families.items())):
            raise ValueError('invalid priority variables or FDR families')
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')

    def as_dict(self):
        return asdict(self)

    def family(self, spec):
        if spec.is_essdai_derived or spec.canonical_column.startswith(('essdai__', 'ovl__')):
            return 'mechanically_related_descriptive'
        return self.families.get(spec.variable_id,
            'priority_' + spec.domain if spec.variable_id in self.priority_variables else 'exploratory_' + spec.domain)


def load_config(path=None, overrides=None):
    values = json.loads(Path(path).read_text()) if path else {}
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return ComparisonConfig(**values)

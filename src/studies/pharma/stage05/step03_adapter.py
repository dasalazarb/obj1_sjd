"""Explicit boundary for the Step 03 preparation and clustered Poisson fit."""
from importlib import import_module

import numpy as np


class Step03:
    def __init__(self):
        self.module = import_module('src.studies.pharma.03_pharma_multistate_biomarkers')

    def prepare(self, intervals, assay, minimum_events):
        data, mean, sd, reason = self.module._prepare_intervals(intervals, assay)
        stacked = self.module.stack_intervals(data, assay)
        support = self.module.transition_support(data, stacked, assay, minimum_events)
        return data, stacked, support, mean, sd, reason

    def fit(self, stacked, routes):
        fit = self.module._fit_stacked(stacked, routes, adjusted=True)
        if not fit.converged or not np.isfinite(fit.params).all() or not np.isfinite(fit.bse).all():
            raise ValueError('nonconvergent_or_nonfinite_step03_fit')
        return fit.params

    def candidates(self, sensitivity):
        return self.module.build_candidate_trace(sensitivity)[0]

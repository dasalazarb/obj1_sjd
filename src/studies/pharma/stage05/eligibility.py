"""SSA by Pop at baseline and retrospective documented history."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import BASELINE_POPS, support_status


def eligibility(master: pd.DataFrame, features: dict) -> pd.DataFrame:
    baseline = master.loc[master.is_clinical_baseline].set_index('patient_id')
    rows = []
    spec = features['ssa']
    data = master.assign(_ssa_raw=master.get('raw_status__ssa', 'not_measured'))
    for pop in BASELINE_POPS:
        patients = (pd.Index(master.patient_id.unique()).difference(baseline.index) if pop == 'No baseline'
                    else baseline.index[baseline['pop'].eq(pop)])
        visits = data.loc[data.patient_id.isin(patients)]
        baseline_visits = visits.loc[visits.is_clinical_baseline]
        for scheme, sample in [('ssa_known_through_clinical_baseline', baseline_visits),
                               ('ever_known_through_observed_episodes', visits)]:
            observed = sample.groupby('patient_id')['feature__ssa'].max().reindex(patients)
            raw = sample['_ssa_raw'].eq('indeterminate').groupby(sample.patient_id).any().reindex(patients, fill_value=False)
            positive = int(observed.eq(1).sum())
            negative = int(observed.eq(0).sum())
            unknown = int((observed.isna() & raw).sum())
            n = len(patients)
            measured = positive + negative
            rows.append(dict(analysis_unit='patient', cohort='effective_master', baseline_pop=pop,
                             index_definition=scheme, n_patients=n, n_evaluable=measured, n_missing=n - measured,
                             n_positive=positive, n_negative_known=negative, n_indeterminate=unknown,
                             n_not_measured=n - measured - unknown,
                             includes_potential_future=sample is visits,
                             estimate=positive/measured if measured else np.nan,
                             **support_status(measured, spec['status'] == 'available', spec['reason'])))
    return pd.DataFrame(rows)

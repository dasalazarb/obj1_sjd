"""Aggregate an authorized inventory relative to an internal patient index."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import DAYS_PER_YEAR, KEYS, check_keys, normalized, support_status

FIELDS = ['analysis_unit', 'cohort', 'specimen_type', 'preservation', 'timing',
          'n_cohort_patients', 'n_patients', 'n_samples', 'n_aliquots', 'status', 'reason']
REQUIRED = {*KEYS, 'specimen_id', 'collection_date', 'specimen_type', 'preservation',
            'available_at_inventory_date', 'inventory_date'}


def unavailable(reason: str) -> pd.DataFrame:
    return pd.DataFrame([{'analysis_unit': 'specimen', 'cohort': 'all',
                          'status': 'not_available', 'reason': reason}], columns=FIELDS)


def specimen_availability(inventory: pd.DataFrame | None, master: pd.DataFrame,
                          trajectories: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if inventory is None:
        return unavailable('authorized_ctdb_inventory_not_available')
    if not REQUIRED.issubset(inventory):
        return unavailable('authorized_inventory_missing_required_fields')
    grain = ['specimen_id', 'aliquot_id'] if 'aliquot_id' in inventory else ['specimen_id']
    inventory = inventory.copy()
    check_keys(inventory, grain, 'authorized CTDB inventory')
    for col in ('collection_date', 'inventory_date'):
        inventory[col] = pd.to_datetime(inventory[col], errors='coerce')
    linked = inventory.merge(master[KEYS], on=KEYS, how='inner', validate='many_to_one')
    known = normalized(linked.available_at_inventory_date)
    linked['available'] = known.isin(['true', '1', '1.0'])
    linked['availability_known'] = known.isin(['true', '1', '1.0', 'false', '0', '0.0'])
    rows = []
    baseline = trajectories.loc[trajectories.index_definition.eq('clinical_baseline')]
    pop2 = trajectories.loc[trajectories.index_definition.eq('first_observed_pop2')]
    cases = pop2.loc[pop2.first_entry_date.notna() & pop2.history.eq('no_prior_pop1_observed')].copy()
    for idx, row in cases.iterrows():
        previous = master.loc[master.patient_id.eq(row.patient_id) &
                              master.clinical_anchor_date.ge(row.index_date) &
                              master.clinical_anchor_date.lt(row.first_entry_date), 'clinical_anchor_date']
        cases.at[idx, 'index_date'] = previous.max()
    controls = pop2.loc[pop2.first_entry_date.isna() & ~pop2.changed.fillna(False).astype(bool) &
                        pop2.n_classifiable.ge(settings['minimum_classifiable_visits']) &
                        pop2.span.ge(settings['stability_years'])]
    cohorts = {'baseline_pop1': baseline.loc[baseline.index_pop.eq('Pop1')],
               'baseline_pop3': baseline.loc[baseline.index_pop.eq('Pop3')],
               'pop2_documented_first_conversion': cases, 'persistent_pop2': controls}
    window = pd.Timedelta(DAYS_PER_YEAR * settings['specimen_window_years'], unit='D')
    for cohort, people in cohorts.items():
        specimens = linked.merge(people[['patient_id', 'index_date']], on='patient_id', validate='many_to_one')
        request = specimens.index_date
        collection = specimens.collection_date
        snapshot = specimens.inventory_date
        available = specimens.available & specimens.availability_known
        timing = [
            collection.gt(request), collection.lt(request - window),
            collection.le(request) & collection.ge(request - window) & snapshot.eq(request) & available,
            collection.le(request) & collection.ge(request - window) & snapshot.lt(request) & available,
            snapshot.eq(request) & specimens.availability_known & ~available & collection.le(request),
        ]
        specimens['timing'] = np.select(timing, ['collected_after_index', 'collected_before_window',
                                                'available_at_index', 'documented_available_before_index',
                                                'documented_unavailable_at_index'], default='availability_unknown_at_index')
        if not len(specimens):
            rows.append(dict(analysis_unit='specimen', cohort=cohort, n_cohort_patients=len(people),
                             status='not_estimable', reason='no_inventory_rows_linked_to_cohort'))
        for (kind, preservation, when), group in specimens.groupby(['specimen_type', 'preservation', 'timing'], dropna=False):
            rows.append(dict(analysis_unit='specimen', cohort=cohort, specimen_type=kind, preservation=preservation,
                             timing=when, n_cohort_patients=len(people), n_patients=group.patient_id.nunique(),
                             n_samples=group.specimen_id.nunique(), n_aliquots=len(group) if 'aliquot_id' in group else np.nan,
                             **support_status(len(group))))
    rows.append(dict(analysis_unit='specimen', cohort='healthy_volunteers', status='not_available',
                     reason='no_authorized_healthy_volunteer_denominator'))
    return pd.DataFrame(rows, columns=FIELDS)

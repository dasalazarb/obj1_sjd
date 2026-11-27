"""Patient trajectories, common-index profiles and pre-event marker trends."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clinical_profiles import PROFILE_FIELDS, describe
from .contracts import BASELINE_POPS, DAYS_PER_YEAR, POPS, support_status

INDICATORS = ('ever_pop1', 'first_conversion', 'sustained_conversion', 'return_to_low',
              'return_to_pop2', 'return_to_pop3', 'recurrent_pop1', 'observed_pop3')
CATEGORIES = ('insufficient_followup', 'recurrent_conversion', 'sustained_conversion',
              'other_conversion', 'observed_baseline_persistence', 'other_changes_without_pop1',
              'pop1_persistence', 'pop1_exit', 'pop1_exit_and_return')
LANDMARK = 'first_observed_pop2'
BASELINE = 'clinical_baseline'


def sequence_history(master: pd.DataFrame, consecutive: int) -> pd.DataFrame:
    """Keep confirmation and first-entry dates internally; unknown episodes break runs."""
    rows = []
    for patient, visits in master.groupby('patient_id', sort=False):
        first_pop2 = visits.loc[visits['pop'].eq('Pop2')].head(1)
        for definition, choice in ((BASELINE, visits.loc[visits.is_clinical_baseline]), (LANDMARK, first_pop2)):
            if choice.empty:
                if definition == BASELINE:
                    rows.append(dict(patient_id=patient, index_definition=definition, index_pop='No baseline',
                                     history='unknown', n_classifiable=0, span=0.,
                                     **dict.fromkeys(INDICATORS, False)))
                continue
            index = choice.iloc[0]
            prior = visits.loc[visits.clinical_visit_number.lt(index.clinical_visit_number)]
            prior_pop1 = bool(prior['pop'].eq('Pop1').any())
            at_and_after = visits.loc[visits.clinical_visit_number.ge(index.clinical_visit_number)]
            sequence = at_and_after.reset_index(drop=True)
            start_pop = index['pop']
            start = index.clinical_anchor_date
            clinical = sequence.loc[sequence['pop'].isin(POPS)]
            saw_pop1, recurrence, pop3 = False, False, False
            after_pop1 = set()
            first, run_start, sustained_start, confirmation = pd.NaT, pd.NaT, pd.NaT, pd.NaT
            run, previous_date = 0, None
            for visit in sequence.itertuples(index=False):
                pop, date = visit.pop, visit.clinical_anchor_date
                if pop == 'Pop1':
                    if pd.isna(first) and start_pop != 'Pop1' and date > start:
                        first = date
                    recurrence |= bool(after_pop1) and previous_date != date
                    saw_pop1 = True
                    if run == 0:
                        run_start, run = date, 1
                    elif date > previous_date:
                        run += 1
                    if run == consecutive and pd.isna(confirmation) and start_pop != 'Pop1' and run_start > start:
                        sustained_start, confirmation = run_start, date
                else:
                    run = 0
                    if saw_pop1 and pop in ('Pop2', 'Pop3'):
                        after_pop1.add(pop)
                pop3 |= pop == 'Pop3' and date > start
                previous_date = date
            span = ((clinical.clinical_anchor_date.max() - start).days / DAYS_PER_YEAR
                    if len(clinical) else 0.)
            if start_pop == 'Unclassifiable':
                # ESSDAI events are reported separately in A. Pop conversion
                # requires a defined Pop2/3 origin.
                first = sustained_start = confirmation = pd.NaT
            rows.append(dict(patient_id=patient, index_definition=definition, index_pop=start_pop,
                             history='prior_pop1' if prior_pop1 else 'no_prior_pop1_observed',
                             index_episode=index.clinical_episode_id, index_date=start, n_classifiable=len(clinical),
                             span=span, changed=bool(clinical['pop'].ne(start_pop).any()),
                             ever_pop1=saw_pop1 or prior_pop1,
                             first_conversion=pd.notna(first) and not prior_pop1,
                             first_entry_date=first, confirmation_date=confirmation, sustained_entry_date=sustained_start,
                             sustained_conversion=pd.notna(confirmation), return_to_low=bool(after_pop1),
                             return_to_pop2='Pop2' in after_pop1, return_to_pop3='Pop3' in after_pop1,
                             recurrent_pop1=recurrence or (prior_pop1 and pd.notna(first)),
                             observed_pop3=pop3))
    fields = ['patient_id', 'index_definition', 'index_pop', 'history', 'index_episode', 'index_date',
              'n_classifiable', 'span', 'changed', 'first_entry_date', 'confirmation_date',
              'sustained_entry_date', *INDICATORS]
    out = pd.DataFrame(rows, columns=fields)
    for date in ('index_date', 'first_entry_date', 'confirmation_date', 'sustained_entry_date'):
        out[date] = pd.to_datetime(out[date])
    return out


def classify(row, horizon: float, minimum_visits: int) -> str:
    if row.index_pop == 'Pop1':
        if row.recurrent_pop1:
            return 'pop1_exit_and_return'
        if row.return_to_low:
            return 'pop1_exit'
        if row.n_classifiable >= minimum_visits and row.span >= horizon:
            return 'pop1_persistence'
    if row.index_pop in ('Pop2', 'Pop3'):
        if pd.notna(row.first_entry_date):
            if row.recurrent_pop1:
                return 'recurrent_conversion'
            return 'sustained_conversion' if row.sustained_conversion else 'other_conversion'
        if row.n_classifiable >= minimum_visits and row.span >= horizon:
            return 'other_changes_without_pop1' if row.changed else 'observed_baseline_persistence'
    return 'insufficient_followup'


def _median_elapsed(dates: pd.Series, index: pd.Series) -> float:
    elapsed = (dates - index).dt.days.dropna()
    return elapsed.median() / DAYS_PER_YEAR if len(elapsed) else np.nan


def trajectory_summary(history: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, sustained = [], []
    horizons = sorted(set(settings['horizons_years']) | {settings['stability_years']})
    groups = [(BASELINE, p, 'any') for p in BASELINE_POPS] + [
        (LANDMARK, 'Pop2', label) for label in ('all', 'prior_pop1', 'no_prior_pop1_observed')]
    for definition, pop, label in groups:
        data = history.loc[history.index_definition.eq(definition) & history.index_pop.eq(pop)]
        if label not in ('any', 'all'):
            data = data.loc[data.history.eq(label)]
        n = len(data)
        meta = dict(analysis_unit='patient', cohort=definition, index_definition=definition,
                    index_pop=pop, prior_history=label, n_patients=n)
        for horizon in horizons:
            categories = [classify(row, horizon, settings['minimum_classifiable_visits'])
                          for row in data.itertuples(index=False)]
            for name in CATEGORIES:
                count = categories.count(name)
                summary.append(dict(**meta, stability_years=horizon, kind='exclusive_category', pattern=name,
                                    n_pattern=count, estimate=count/n if n else np.nan, **support_status(n)))
        for indicator in INDICATORS:
            count = int(data[indicator].sum())
            summary.append(dict(**meta, stability_years=np.nan, kind='nonexclusive_indicator', pattern=indicator,
                                n_pattern=count, estimate=count/n if n else np.nan, **support_status(n)))
        confirmed = data.loc[data.sustained_conversion.astype(bool)]
        n_entered = int(data.first_entry_date.notna().sum())
        sustained.append(dict(**meta, n_first_entry=n_entered, n_confirmed=len(confirmed),
                              n_entered_without_confirmation=n_entered - len(confirmed),
                              median_years_to_first_entry=_median_elapsed(data.first_entry_date, data.index_date),
                              median_years_to_sustained_entry=_median_elapsed(confirmed.sustained_entry_date,
                                                                             confirmed.index_date),
                              median_years_to_confirmation=_median_elapsed(confirmed.confirmation_date,
                                                                          confirmed.index_date),
                              estimate=len(confirmed)/n_entered if n_entered else np.nan, **support_status(n_entered)))
    return pd.DataFrame(summary), pd.DataFrame(sustained)


def converter_profiles(master: pd.DataFrame, history: pd.DataFrame,
                       features: dict, settings: dict) -> pd.DataFrame:
    rows = []
    landmark = history.loc[history.index_definition.eq(LANDMARK)].copy()
    landmark['event_years'] = (landmark.first_entry_date - landmark.index_date).dt.days / DAYS_PER_YEAR
    values = [f'feature__{marker}' for marker in features]
    joined = landmark.merge(master[['patient_id', 'clinical_episode_id', *values]],
                            left_on=['patient_id', 'index_episode'],
                            right_on=['patient_id', 'clinical_episode_id'], validate='one_to_one')
    visits = {key: group for key, group in master.groupby('patient_id', sort=False)}
    for horizon in settings['horizons_years']:
        for label in ('no_prior_pop1_observed', 'prior_pop1'):
            groups = joined.loc[joined.history.eq(label)].copy()
            def observed_change(row):
                person = visits[row.patient_id]
                before_horizon = person.loc[
                    person.clinical_anchor_date.ge(row.index_date) &
                    person.clinical_anchor_date.le(row.index_date + pd.Timedelta(horizon * DAYS_PER_YEAR, unit='D'))]
                return before_horizon['pop'].isin(['Pop1', 'Pop3']).any()

            changed = pd.Series([observed_change(row) for row in groups.itertuples(index=False)],
                                index=groups.index, dtype=bool)
            groups['event_group'] = np.select(
                [groups.event_years.le(horizon),
                 groups.span.ge(horizon) & groups.n_classifiable.ge(settings['minimum_classifiable_visits']) & ~changed,
                 groups.span.ge(horizon) & groups.n_classifiable.ge(settings['minimum_classifiable_visits'])],
                ['pop1_by_horizon', 'observed_persistent_pop2_at_horizon', 'other_observed_through_horizon'],
                default='censored_or_nonevaluable')
            # Time matching uses elapsed time, independent of marker availability.
            cases = groups.loc[groups.event_group.eq('pop1_by_horizon')]
            controls = groups.loc[groups.event_group.eq('observed_persistent_pop2_at_horizon')]
            ref_cases, ref_controls = [], []
            reference = cases.event_years.median() if len(cases) else np.nan
            for sample, accumulator, is_case in ((cases, ref_cases, True), (controls, ref_controls, False)):
                for row in sample.itertuples(index=False):
                    if not np.isfinite(reference):
                        continue
                    index = row.first_entry_date if is_case else row.index_date + pd.Timedelta(reference * DAYS_PER_YEAR, unit='D')
                    person = visits[row.patient_id]
                    before = person.loc[person.clinical_anchor_date.ge(row.index_date) &
                                        person.clinical_anchor_date.lt(index)]
                    if len(before):
                        accumulator.append(before.iloc[-1])
            matched = {'pop1_by_horizon': pd.DataFrame(ref_cases, columns=master.columns),
                       'observed_persistent_pop2_at_horizon': pd.DataFrame(ref_controls, columns=master.columns)}
            for scheme, samples in [('common_landmark', {group: groups.loc[groups.event_group.eq(group)]
                                                         for group in ('pop1_by_horizon', 'observed_persistent_pop2_at_horizon',
                                                                       'other_observed_through_horizon', 'censored_or_nonevaluable')}),
                                     ('preevent_or_comparable_elapsed_time', matched)]:
                for group, sample in samples.items():
                    for marker, spec in features.items():
                        rows.extend(dict(analysis_unit='patient', cohort='pop2_entry', index_definition=LANDMARK,
                                         scheme=scheme, horizon_years=horizon, prior_history=label, event_group=group,
                                         reference_elapsed_years=reference if scheme != 'common_landmark' else np.nan,
                                         **row) for row in describe(sample, f'feature__{marker}', marker, spec))
    return pd.DataFrame(rows, columns=['analysis_unit', 'cohort', 'index_definition', 'scheme', 'horizon_years',
                                      'prior_history', 'event_group', 'reference_elapsed_years', *PROFILE_FIELDS])


def marker_trends(master: pd.DataFrame, history: pd.DataFrame,
                  features: dict, settings: dict) -> pd.DataFrame:
    rows = []
    landmark = history.loc[history.index_definition.eq(LANDMARK)]
    cases = landmark.loc[landmark.first_entry_date.notna()]
    window = pd.Timedelta(DAYS_PER_YEAR * settings['trend_window_years'], unit='D')
    for label in ('no_prior_pop1_observed', 'prior_pop1'):
        selected = cases.loc[cases.history.eq(label)]
        # Controls must be under observation at the same elapsed pseudo-index.
        # Use the median detected event latency without looking at any marker.
        delay = (selected.first_entry_date - selected.index_date).median() if len(selected) else pd.NaT
        controls = landmark.loc[landmark.history.eq(label) & landmark.first_entry_date.isna() &
                                landmark.span.ge(delay / pd.Timedelta(DAYS_PER_YEAR, unit='D'))]
        groups = {'pre_pop1_entry': selected, 'comparable_pseudoindex': controls}
        for group, sample in groups.items():
            for marker in settings['trend_markers']:
                spec = features[marker]
                slopes = []
                for person in sample.itertuples(index=False):
                    end = person.first_entry_date if group == 'pre_pop1_entry' else person.index_date + delay
                    start = max(person.index_date, end - window)
                    visits = master.loc[master.patient_id.eq(person.patient_id) &
                                        master.clinical_anchor_date.ge(start) & master.clinical_anchor_date.lt(end)]
                    points = visits.groupby('clinical_anchor_date')[f'feature__{marker}'].mean().dropna()
                    if len(points) >= settings['minimum_trend_measurements']:
                        x = (points.index - points.index[0]).days / DAYS_PER_YEAR
                        slopes.append(float(np.polyfit(x, points.to_numpy(), 1)[0]))
                n = len(slopes)
                rows.append(dict(analysis_unit='patient', cohort='pop2_entry', index_definition=LANDMARK,
                                 prior_history=label, scheme=group, marker=marker, unit=f'{spec["unit"]}/year',
                                 window_years=settings['trend_window_years'], n_patients=len(sample),
                                 n_evaluable=n, n_missing=len(sample) - n,
                                 estimate=np.median(slopes) if n else np.nan,
                                 **support_status(n, spec['status'] == 'available',
                                                  spec['reason'] or 'insufficient_distinct_preevent_measurements')))
    return pd.DataFrame(rows)

"""First documented moderate-or-higher systemic activity and risk summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from .contracts import BASELINE_POPS, DAYS_PER_YEAR, POPS, THRESHOLD, support_status

SCORE = 'feature__essdai_total'
YEAR = pd.Timedelta(DAYS_PER_YEAR, unit='D')
PRIMARY = 'clinical_baseline'
CLASSIFIABLE = 'first_classifiable_episode'
LOW_ACTIVITY = 'clinical_baseline_low_essdai_missing_esspri'


def event_history(master: pd.DataFrame, index_definition: str = PRIMARY,
                  cutoff: str | None = None, midpoint: bool = False) -> pd.DataFrame:
    """Internal patient-level input. Missing ESSDAI visits do not extend observation."""
    rows = []
    cutoff = pd.Timestamp(cutoff) if cutoff else pd.Timestamp.max.normalize()
    for patient, group in master.groupby('patient_id', sort=False):
        if index_definition == CLASSIFIABLE:
            eligible = group.loc[group['pop'].isin(POPS)]
        else:
            eligible = group.loc[group.is_clinical_baseline]
        index = eligible.iloc[0] if len(eligible) else None
        death = group['_death'].iloc[0] if '_death' in group else pd.NaT
        stop = min(cutoff, death) if pd.notna(death) else cutoff
        visits = group.loc[group.clinical_anchor_date.le(stop)]
        if index is not None and index.clinical_anchor_date > stop:
            continue
        if index is None and index_definition != PRIMARY:
            continue
        if index is not None and index_definition == LOW_ACTIVITY and not (
            index['pop'] == 'Unclassifiable' and pd.notna(index[SCORE]) and index[SCORE] < THRESHOLD
        ):
            continue
        origin = index.clinical_anchor_date if index is not None else pd.NaT
        pop = index['pop'] if index is not None else 'No baseline'
        base = index[SCORE] if index is not None else np.nan
        assessed = visits.loc[visits[SCORE].notna()]
        followup = assessed.loc[assessed.clinical_anchor_date.gt(origin)] if index is not None else assessed.iloc[:0]
        events = followup.loc[followup[SCORE].ge(THRESHOLD)]
        event = len(events) > 0
        incident_eligible = pop in ('Pop2', 'Pop3') or index_definition == LOW_ACTIVITY
        time = competing_time = np.nan
        event_type = 0
        if incident_eligible and len(followup):
            terminal = events.clinical_anchor_date.iloc[0] if event else followup.clinical_anchor_date.iloc[-1]
            time = (terminal - origin) / YEAR
            if midpoint and event:
                last_low = assessed.loc[
                    assessed.clinical_anchor_date.ge(origin) & assessed.clinical_anchor_date.lt(terminal) &
                    assessed[SCORE].lt(THRESHOLD), 'clinical_anchor_date'].max()
                time = (time + (last_low - origin) / YEAR) / 2
            competing_time = time
            event_type = int(event)
            if not event and pd.notna(death) and death <= cutoff:
                event_type = 2
                competing_time = (death - origin) / YEAR
        baseline_event = pd.notna(base) and base >= THRESHOLD
        observed = bool(pd.notna(base) or len(followup)) if index is not None else bool(len(assessed))
        ever = bool(baseline_event or event) if index is not None else bool(assessed[SCORE].ge(THRESHOLD).any())
        rows.append(dict(patient_id=patient, pop=pop, baseline=index is not None,
                         baseline_event=baseline_event, has_followup=bool(len(followup)),
                         observed=observed, ever=ever, event=event, time=time,
                         competing_time=competing_time, event_type=event_type))
    return pd.DataFrame(rows, columns=['patient_id', 'pop', 'baseline', 'baseline_event', 'has_followup',
                                      'observed', 'ever', 'event', 'time', 'competing_time', 'event_type'])


def ever_summary(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pop in BASELINE_POPS:
        part = history.loc[history['pop'].eq(pop)]
        n = int(part.observed.sum())
        events = int(part.ever.sum())
        ci = binomtest(events, n).proportion_ci() if n else (np.nan, np.nan)
        rows.append(dict(analysis_unit='patient', cohort='effective_master', index_definition=PRIMARY,
                         baseline_pop=pop, n_patients=len(part), n_baseline=int(part.baseline.sum()),
                         n_followup_evaluable=int(part.has_followup.sum()),
                         n_baseline_without_evaluable_followup=int((part.baseline & ~part.has_followup).sum()),
                         n_pop1_baseline=int(part.baseline_event.sum()),
                         n_first_documented_postbaseline=int((part.event & ~part.baseline_event).sum()),
                         n_ever_observed=events, n_evaluable=n, n_missing=len(part) - n,
                         denominator='patients with observed ESSDAI at index or later; any visit for no baseline',
                         estimate=events/n if n else np.nan, ci95_low=ci[0], ci95_high=ci[1],
                         **support_status(n)))
    return pd.DataFrame(rows)


def product_limit(times: np.ndarray, events: np.ndarray, *, competing: bool = False) -> pd.DataFrame:
    """At a tied time, use the full risk set before removing any endpoints."""
    survival, incidence, greenwood = 1., 0., 0.
    rows = []
    # Events are encoded as 1, competing deaths as 2, censoring as 0.
    n_risk = len(times)
    endpoints = pd.DataFrame({'time': times, 'event': events})
    for time, group in endpoints.groupby('time', sort=True):
        event = int(group.event.eq(1).sum())
        death = int(group.event.eq(2).sum()) if competing else 0
        censored = len(group) - event - death
        incidence += survival * event / n_risk
        if event and n_risk > event:
            greenwood += event / (n_risk * (n_risk - event))
        survival *= 1 - (event + death) / n_risk
        low = high = np.nan
        if not competing:
            if 0 < survival < 1:
                center = np.log(-np.log(survival))
                width = norm.ppf(.975) * np.sqrt(greenwood) / abs(np.log(survival))
                low, high = 1 - np.exp(-np.exp([center - width, center + width]))
            # Degenerate Greenwood limits are intentionally not reported.
            else:
                incidence = 1 - survival
        rows.append(dict(years=time, n_at_risk=n_risk, n_events=event, n_censored=censored,
                         n_competing_deaths=death, estimate=incidence,
                         ci95_low=low, ci95_high=high, survival=survival))
        n_risk -= len(group)
    return pd.DataFrame(rows, columns=['years', 'n_at_risk', 'n_events', 'n_censored',
                                      'n_competing_deaths', 'estimate', 'ci95_low', 'ci95_high', 'survival'])


def risk_tables(master: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, ...]:
    args = dict(cutoff=settings.get('administrative_cutoff'))
    history = event_history(master, **args)
    estimates, curves = [], []
    for definition, midpoint in ((PRIMARY, False), (PRIMARY, True), (CLASSIFIABLE, False),
                                 (LOW_ACTIVITY, False)):
        data = event_history(master, definition, midpoint=midpoint, **args) if (
            definition != PRIMARY or midpoint
        ) else history
        pops = ('Unclassifiable',) if definition == LOW_ACTIVITY else ('Pop2', 'Pop3')
        for pop in pops:
            part = data.loc[data['pop'].eq(pop) & data.time.notna()]
            median_followup = product_limit(part.time.to_numpy(), (~part.event.astype(bool)).astype(int).to_numpy())
            median = median_followup.loc[median_followup.survival.le(.5), 'years'].min()
            methods = ('km', 'aalen_johansen') if '_death' in master and not midpoint else ('km',)
            for method in methods:
                cif = method == 'aalen_johansen'
                times = part.competing_time if cif else part.time
                events = part.event_type if cif else part.event.astype(int)
                meta = dict(analysis_unit='patient', cohort='incident_essdai_evaluable', index_definition=definition,
                            baseline_pop=pop, estimator=method, scheme='midpoint' if midpoint else 'documented_visit',
                            n_patients=len(part), median_reverse_km_followup=median)
                curve = product_limit(times.to_numpy(), events.to_numpy(), competing=cif)
                if len(curve):
                    initial = {key: 0 for key in ('years', 'estimate', 'n_events', 'n_censored', 'n_competing_deaths')}
                    curves.append({**meta, **initial, 'n_at_risk': len(part), 'ci95_low': np.nan, 'ci95_high': np.nan})
                    curves.extend({**meta, **row} for row in curve.drop(columns='survival').to_dict('records'))
                for year in settings['horizons_years']:
                    at_risk = int(times.ge(year).sum())
                    elapsed = curve.loc[curve.years.le(year)]
                    last = elapsed.iloc[-1] if len(elapsed) else None
                    if at_risk:
                        estimate = last.estimate if last is not None else 0.
                        low = last.ci95_low if last is not None else np.nan
                        high = last.ci95_high if last is not None else np.nan
                    else:
                        estimate = low = high = np.nan
                    inference_note = ('CIF point estimate; no CIF interval' if cif else
                                      'Greenwood limits undefined at 0 or 1' if at_risk and estimate in (0., 1.) else '')
                    estimates.append({**meta, 'years': year, 'n_at_risk': at_risk,
                                      'n_events': int(elapsed.n_events.sum()),
                                      'n_censored': int(elapsed.n_censored.sum()),
                                      'n_competing_deaths': int(elapsed.n_competing_deaths.sum()),
                                      'estimate': estimate, 'ci95_low': low, 'ci95_high': high,
                                      'inference_note': inference_note,
                                      'stability_warning': bool(0 < at_risk < settings['minimum_at_risk_warning']),
                                      **support_status(at_risk, reason='horizon_beyond_supported_followup')})
    fields = ['analysis_unit', 'cohort', 'index_definition', 'baseline_pop', 'estimator', 'scheme',
              'n_patients', 'median_reverse_km_followup', 'years', 'n_at_risk', 'n_events', 'n_censored',
              'n_competing_deaths', 'estimate', 'ci95_low', 'ci95_high']
    return ever_summary(history), pd.DataFrame(estimates), pd.DataFrame(curves, columns=fields)

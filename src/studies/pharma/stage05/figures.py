"""Aggregate, non-interactive PDF figures and source bindings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import BASELINE_POPS, POPS, PREFIX, ROUTES

PALETTE = {'Pop1': '#d95f02', 'Pop2': '#7570b3', 'Pop3': '#1b9e77',
           'Unclassifiable': '#9e9e9e', 'No baseline': '#5f666b'}
MISSING = '#d8dade'
FIGURE_SOURCES = {
    'pop1_cumulative_risk': ('pop1_risk_curve_source', 'pop1_cumulative_risk'),
    'pop2_biomarker_correlations': ('pop2_biomarker_correlations',),
    'absolute_risk_by_biomarker': ('absolute_risk_by_biomarker',),
    'domain_drivers_heatmap': ('domain_drivers_by_transition',),
    'pro_changes_by_transition': ('pro_changes_by_transition',),
    'patient_trajectory_patterns': ('patient_trajectory_summary',),
    'baseline_pop_distribution': ('baseline_reconciliation',),
    'specimen_availability': ('specimen_availability',),
    'converter_profiles': ('patient_converter_profiles',),
}


def _risk(plt, tables, settings):
    source = tables['pop1_risk_curve_source']
    risk = tables['pop1_cumulative_risk']
    primary = lambda data: (data.index_definition.eq('clinical_baseline') &
                            data.scheme.eq('documented_visit') & data.estimator.eq('km'))
    curve = source.loc[primary(source)]
    if curve.empty:
        return None
    fig, (ax, count) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[4, 1])
    horizons = settings['horizons_years']
    table_rows, labels = [], []
    for pop in ('Pop2', 'Pop3'):
        group = curve.loc[curve.baseline_pop.eq(pop)]
        if group.empty:
            continue
        color = PALETTE[pop]
        ax.step(group.years, group.estimate, where='post', color=color,
                label=f"{pop} (N={int(group.n_patients.iloc[0])})")
        valid = group.ci95_low.notna() & group.ci95_high.notna()
        ax.fill_between(group.years.to_numpy(dtype=float), group.ci95_low.where(valid).to_numpy(dtype=float),
                        group.ci95_high.where(valid).to_numpy(dtype=float), step='post',
                        color=color, alpha=.16)
        censored = group.loc[group.n_censored.gt(0)]
        ax.scatter(censored.years, censored.estimate, marker='+', color=color, s=26)
        cells = risk.loc[primary(risk) & risk.baseline_pop.eq(pop)].set_index('years').reindex(horizons)
        labels.extend([f'{pop} at risk', f'{pop} censored to date'])
        table_rows.extend([cells.n_at_risk.astype(int).tolist(), cells.n_censored.astype(int).tolist()])
    ax.set(title='First documented ESSDAI ≥5', xlabel='Years since clinical baseline',
           ylabel='Cumulative incidence (1 − KM)', ylim=(-.03, 1.03), xlim=(0, None))
    ax.legend(frameon=False, title='Clinical baseline Pop')
    count.axis('off')
    count.table(cellText=table_rows, rowLabels=labels, colLabels=[f'{h} y' for h in horizons],
                cellLoc='center', loc='center')
    return fig


def _heatmap(plt, table, row, column, title, legend, *, limits=None, sort_columns=None):
    if table.empty or not table.estimate.notna().any():
        return None
    data = table.pivot(index=row, columns=column, values='estimate')
    counts = table.pivot(index=row, columns=column, values='n_evaluable').reindex_like(data)
    if sort_columns:
        data, counts = data.reindex(columns=sort_columns), counts.reindex(columns=sort_columns)
    fig, ax = plt.subplots(figsize=(max(8, .85 * len(data.columns)), max(4, .43 * len(data.index) + 2)))
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    colormap = LinearSegmentedColormap.from_list(
        'centered_change', ['#4775bb', '#ffffff', '#bd352b']).with_extremes(bad=MISSING)
    local = data.to_numpy(dtype=float)
    span = max(float(np.nanmax(np.abs(local))), .1)
    im = ax.imshow(local, cmap=colormap, aspect='auto', vmin=-span if limits is None else limits[0],
                   vmax=span if limits is None else limits[1])
    ax.set(xticks=range(len(data.columns)), yticks=range(len(data.index)), title=title)
    ax.set_xticklabels(data.columns, rotation=35, ha='right')
    ax.set_yticklabels(data.index)
    for i, j in np.ndindex(local.shape):
        if np.isfinite(local[i, j]):
            ax.text(j, i, f'{local[i, j]:.2g}\nN={int(counts.iloc[i, j])}', ha='center', va='center',
                    fontsize=7, color='#161616')
    for label in ax.get_xticklabels():
        if label.get_text() == 'Pop2 → Pop1':
            label.set_weight('bold')
            label.set_color(PALETTE['Pop2'])
    fig.colorbar(im, ax=ax, shrink=.8, label=legend)
    if np.isnan(local).any():
        ax.legend(handles=[Patch(facecolor=MISSING, label='No estimate')], loc='lower right',
                  bbox_to_anchor=(1., 1.015), frameon=False, fontsize=8)
    return fig


def _correlation(plt, table):
    table = table.loc[table.family.eq('biomarker_pop2')].copy()
    marker = list(dict.fromkeys(list(table.marker) + list(table.paired_marker)))
    table['n_evaluable'] = table.n_patients_evaluable
    mirror = table.rename(columns={'marker': 'paired_marker', 'paired_marker': 'marker'})
    table = pd.concat([table, mirror], ignore_index=True)
    return _heatmap(plt, table, 'marker', 'paired_marker', 'Pop2 biomarker associations',
                    'Spearman ρ at first eligible Pop2 episode', limits=(-1., 1.), sort_columns=marker)


def _domains(plt, table, clinical):
    table = table.copy()
    table['route'] = table.from_pop + ' → ' + table.to_pop
    if not clinical:
        table = table.loc[table.scheme.isna()]
    return _heatmap(plt, table, 'marker', 'route',
                    'ESSDAI domain changes by transition' if clinical else 'ESSPRI changes by transition',
                    'Median paired change (ordinal levels)' if clinical else 'Median paired change (points)',
                    sort_columns=[f'{a} → {b}' for a, b in ROUTES])


def _projection(plt, table):
    table = table.loc[table.status.eq('estimated')].copy()
    if table.empty:
        return None
    groups = list(table.groupby(['marker', 'scheme'], sort=True))
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 3.5 * len(groups)), squeeze=False)
    for ax, ((marker, scheme), group) in zip(axes.flat, groups):
        for pop, points in group.groupby('from_pop'):
            points = points.sort_values('years')
            ax.plot(points.years, points.estimate, marker='o', color=PALETTE[pop],
                    label=f'{pop} (N={int(points.n_reference_patients.iloc[0])})')
            confidence = points.dropna(subset=['ci95_low', 'ci95_high'])
            ax.vlines(confidence.years, confidence.ci95_low, confidence.ci95_high, color=PALETTE[pop])
        ax.axhline(0, color='#555555', linewidth=.8)
        ax.set(title=f'{marker.upper()} ({group.unit.iloc[0]}) · {scheme.replace("_", " ")}',
               xlabel='Projection horizon (years)', ylabel='P75 − P25 (percentage points)')
        ax.legend(frameon=False)
    fig.suptitle('Exploratory probability of Pop1')
    return fig


def _baseline(plt, table):
    counts = table.loc[table.cohort.eq('clinical_baseline') & table.baseline_pop.notna()]
    counts = counts.set_index('baseline_pop').n_patients.reindex(BASELINE_POPS[:-1])
    if counts.sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts, color=[PALETTE[p] for p in counts.index])
    ax.bar_label(bars)
    ax.set(title=f'Clinical baseline Pop distribution (N={int(counts.sum())})', ylabel='Patients')
    ax.margins(y=.15)
    return fig


def _trajectory(plt, table, years):
    table = table.loc[table.index_definition.eq('clinical_baseline') & table.kind.eq('exclusive_category') &
                      table.stability_years.eq(years) & table.index_pop.isin(POPS) & table.n_patients.gt(0)]
    if table.empty:
        return None
    table = table.pivot(index='pattern', columns='index_pop', values='n_pattern')
    table = table.loc[table.sum(axis=1).gt(0)]
    table.index = table.index.str.replace('_', ' ').str.capitalize()
    fig, ax = plt.subplots(figsize=(8, max(4., .5 * len(table))))
    table.plot.barh(ax=ax, color=[PALETTE[p] for p in table.columns], width=.8)
    ax.set(title=f'Observed trajectory patterns ({years:g}-year stability requirement)', xlabel='Patients', ylabel=None)
    ax.legend(title='Clinical baseline Pop', frameon=False)
    return fig


def _converter_profiles(plt, table, years):
    groups = {'pop1_by_horizon': 'Documented Pop1', 'observed_persistent_pop2_at_horizon': 'Observed persistent Pop2'}
    markers = {'essdai_total': 'ESSDAI', 'esspri_total': 'ESSPRI', 'age': 'Age', 'esr': 'ESR'}
    summary = table.loc[table.scheme.eq('common_landmark') & table.horizon_years.eq(years) &
                        table.prior_history.eq('no_prior_pop1_observed') & table.event_group.isin(groups) &
                        table.marker.isin(markers) & table['median'].notna()]
    pairs = [(key, rows.set_index('event_group').reindex(groups)) for key, rows in summary.groupby('marker')
             if rows.event_group.nunique() == len(groups)]
    if not pairs:
        return None
    fig, axes = plt.subplots(len(pairs), 1, figsize=(8, 1.7 * len(pairs) + 1.0), squeeze=False)
    for ax, (marker, rows) in zip(axes.flat, pairs):
        for y, (_, row) in enumerate(rows.iterrows()):
            ax.hlines(y, row.p25, row.p75, color=PALETTE['Pop2'], linewidth=2)
            ax.scatter(row['median'], y, color=PALETTE['Pop2'], s=30)
        ax.set(yticks=[0, 1], yticklabels=[f'{label} (N={int(n)})' for label, n in
                                           zip(groups.values(), rows.n_evaluable)],
               xlabel=f'{markers[marker]} ({rows.unit.iloc[0]})')
        ax.grid(axis='x', color='#dddddd', linewidth=.6)
    fig.suptitle(f'First Pop2 episode by documented {years:g}-year outcome')
    return fig


def _specimens(plt, table):
    data = table.loc[table.status.eq('estimated')]
    if data.empty:
        return None
    count = data.groupby(['cohort', 'timing']).n_samples.sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    count.plot.barh(ax=ax)
    ax.set(title='Linked specimen collections by timing', xlabel='Samples', ylabel=None)
    ax.legend(title=None, frameon=False, fontsize=8)
    return fig


def figures(tables, settings, folder: Path) -> pd.DataFrame:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    builders = {
        'pop1_cumulative_risk': lambda: _risk(plt, tables, settings),
        'pop2_biomarker_correlations': lambda: _correlation(plt, tables['pop2_biomarker_correlations']),
        'absolute_risk_by_biomarker': lambda: _projection(plt, tables['absolute_risk_by_biomarker']),
        'domain_drivers_heatmap': lambda: _domains(plt, tables['domain_drivers_by_transition'], True),
        'pro_changes_by_transition': lambda: _domains(plt, tables['pro_changes_by_transition'], False),
        'patient_trajectory_patterns': lambda: _trajectory(plt, tables['patient_trajectory_summary'],
                                                            settings['stability_years']),
        'baseline_pop_distribution': lambda: _baseline(plt, tables['baseline_reconciliation']),
        'specimen_availability': lambda: _specimens(plt, tables['specimen_availability']),
        'converter_profiles': lambda: _converter_profiles(plt, tables['patient_converter_profiles'],
                                                           settings['stability_years']),
    }
    rows = []
    for name, build in builders.items():
        filename = f'{PREFIX}{name}.pdf'
        fig = build()
        if fig is not None:
            try:
                fig.tight_layout()
                fig.savefig(folder / filename, metadata={'CreationDate': None, 'ModDate': None})
            finally:
                plt.close(fig)
        rows.append(dict(figure=filename, source_tables=' | '.join(
                             f'{PREFIX}{source}.csv' for source in FIGURE_SOURCES[name]),
                         status='estimated' if fig is not None else 'not_estimable',
                         reason='' if fig is not None else 'no_supported_aggregate_estimate'))
    return pd.DataFrame(rows)

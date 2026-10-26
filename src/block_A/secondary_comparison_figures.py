"""Aggregate comparative figures only; no patient records accepted here."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.block_A.observed_patterns_v2_figures import COLORS, LABELS
from src.block_A.secondary_comparison_outputs import safe_table


def save(fig, path, formats=('png',), dpi=250, origin='canonical'):
    paths = []
    fig.suptitle('SYNTHETIC VALIDATION · NOT CLINICAL RESULTS' if origin == 'synthetic_validation'
                 else 'PRELIMINARY · INSTITUTIONAL REVIEW REQUIRED', fontsize=9, y=1.01)
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for extension in formats:
            target = path.with_suffix('.'+extension)
            fig.savefig(target, dpi=dpi, bbox_inches='tight')
            paths.append(target)
    finally:
        plt.close(fig)
    return paths


def trajectory(data, label, unit, threshold, axis, path, **kwargs):
    safe_table(data)
    fig, (ax, support) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]})
    for pattern, name in LABELS.items():
        g = data.loc[data.observed_pattern.eq(pattern)].sort_values('time')
        # NaNs remain gaps in curves; do not bridge unsupported intervals.
        ax.plot(g.time, g.estimate, color=COLORS[pattern], label=name)
        ax.fill_between(g.time, g.ci95_low, g.ci95_high, color=COLORS[pattern], alpha=.15)
        support.plot(g.time, g.n_observed_local, color=COLORS[pattern])
    ax.set_ylabel(unit)
    ax.set_title(f'{label} | ESSDAI ±{threshold}')
    ax.legend(frameon=False, ncol=2)
    support.set_ylabel('Local n\npatients')
    support.set_xlabel('Years from '+axis.replace('_', ' '))
    fig.text(.02, .005, 'PRELIMINARY • retrospective associations • 95% CI • institutional review required', fontsize=8)
    return save(fig, path, **kwargs)


def forest(data, label, unit, threshold, path, **kwargs):
    safe_table(data)
    g = data.loc[data.contrast_type.eq('slope') & data.multiplicity_role.eq('prespecified')]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for index, row in enumerate(g.itertuples()):
        ax.errorbar(row.estimate, index, xerr=[[row.estimate-row.ci95_low], [row.ci95_high-row.estimate]],
            fmt='o', color=list(COLORS.values())[index+1])
    ax.set_yticks(range(len(g)), [s.replace('_observed', '').replace('_', ' ') for s in g.contrast])
    ax.axvline(0, color='.6', linewidth=1)
    ax.set_xlabel(unit+'; difference vs Within threshold (95% CI)')
    ax.set_title(f'{label} | ESSDAI ±{threshold}')
    return save(fig, path, **kwargs)


def availability(data, threshold, path, **kwargs):
    safe_table(data)
    if data.empty:
        return []
    variables = sorted(data.variable.unique())
    # One patient per window, already aggregated by the support builder.
    g = data.groupby(['variable', 'observed_pattern']).agg(n=('N_any', 'max'), total=('N_group_total', 'max'))
    values = np.full((len(variables), 4), np.nan)
    fig, ax = plt.subplots(figsize=(9, max(3, .32*len(variables))))
    for i, variable in enumerate(variables):
        for j, pattern in enumerate(LABELS):
            if (variable, pattern) in g.index:
                row = g.loc[(variable, pattern)]
                values[i, j] = 100*row.n/row.total if row.total else np.nan
                ax.text(j, i, f'{int(row.n)}/{int(row.total)}', ha='center', va='center', fontsize=8)
    im = ax.imshow(values, vmin=0, vmax=100, cmap='Blues', aspect='auto')
    ax.set_xticks(range(4), LABELS.values(), rotation=20, ha='right')
    ax.set_yticks(range(len(variables)), variables)
    ax.set_title(f'Observed measurement availability | ESSDAI ±{threshold}')
    fig.colorbar(im, ax=ax, label='% assigned group with ≥1 measurement')
    return save(fig, path, **kwargs)


def availability_windows(data, threshold, path, **kwargs):
    safe_table(data)
    if data.empty:
        return []
    rows = sorted(set(zip(data.variable, data.observed_pattern)))
    windows = sorted(data.window.unique())
    values = np.full((len(rows), len(windows)), np.nan)
    fig, ax = plt.subplots(figsize=(max(7, len(windows)*.7), max(3, len(rows)*.24)))
    for i, (variable, pattern) in enumerate(rows):
        for j, window in enumerate(windows):
            g = data.loc[data.variable.eq(variable) & data.observed_pattern.eq(pattern) & data.window.eq(window)]
            if len(g):
                r = g.iloc[0]
                values[i, j] = r.percent_group_measured
                ax.text(j, i, str(int(r.n_patients)), ha='center', va='center', fontsize=7)
    im = ax.imshow(values, vmin=0, vmax=100, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(windows)), windows)
    ax.set_xlabel('Follow-up window index ([start, end) years); cells = unique observed n')
    ax.set_yticks(range(len(rows)), [v+' / '+LABELS[p] for v, p in rows], fontsize=7)
    ax.set_title(f'Availability by window | ESSDAI ±{threshold}')
    fig.colorbar(im, ax=ax, label='% assigned group measured')
    return save(fig, path, **kwargs)


def effects(contrasts, registry, threshold, path, **kwargs):
    safe_table(contrasts)
    variables = registry.loc[registry.selected.eq(True), 'variable_id'].drop_duplicates().tolist()
    if not variables:
        return []
    values = np.full((len(variables), 3), np.nan)
    fig, ax = plt.subplots(figsize=(8, max(3, len(variables)*.32)))
    patterns = list(LABELS)
    for i, v in enumerate(variables):
        mechanical = registry.loc[registry.variable_id.eq(v), 'is_essdai_derived'].map(lambda value: value is not False and value != 0).any()
        for j, pattern in enumerate(patterns[1:]):
            g = contrasts.loc[contrasts.variable.eq(v) & contrasts.contrast.eq(pattern+' - '+patterns[0]) &
                              contrasts.contrast_type.eq('slope')] if not contrasts.empty else contrasts
            value = g.standardized_effect.iloc[0] if len(g) else np.nan
            if mechanical:
                label = 'M'
            elif np.isfinite(value):
                values[i, j] = value
                label = f'{value:.2f}'
            else:
                label = 'NE'
            ax.text(j, i, label, ha='center', va='center', fontsize=8)
    finite = values[np.isfinite(values)]
    scale = max(1., np.max(np.abs(finite))) if len(finite) else 1.
    im = ax.imshow(values, cmap='RdBu_r', vmin=-scale, vmax=scale, aspect='auto')
    ax.set_yticks(range(len(variables)), variables)
    ax.set_xticks(range(3), [LABELS[p]+' − Within threshold' for p in patterns[1:]], rotation=20, ha='right')
    ax.set_title(f'Standardized slope differences | ESSDAI ±{threshold}\nNE: not estimable; M: mechanically related')
    fig.colorbar(im, ax=ax, label='Official-baseline SD per year')
    return save(fig, path, **kwargs)

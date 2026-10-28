"""Aggregate-only atlas, derivatives, six contrasts, support and evidence panels."""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from src.block_A.observed_patterns_v2_figures import LABELS, COLORS


def _segments(d, step):
    d = d.sort_values('time')
    return [part for _, part in d.groupby(d.time.diff().gt(step*1.01).cumsum())]


def render(tables, destination, cfg, selection='priority', dpi=250, origin='canonical'):
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    def save(fig, name):
        path = destination/(name+'.png')
        fig.suptitle(fig._suptitle.get_text()+f'\n{origin}; retrospective associations, no causal/onset interpretation', fontsize=10)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        paths.append(path.name)
    support = tables['00_local_temporal_support']
    curves = tables['03_group_curves']
    for variable in tables['00_variable_eligibility'].variable.unique():
        if selection == 'priority' and variable not in cfg.priority_variables:
            continue
        safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', variable)
        d = curves.loc[curves.variable.eq(variable)] if len(curves) else curves
        s = support.loc[support.variable.eq(variable)] if len(support) else support
        if len(s):
            fig, ax = plt.subplots(figsize=(8, 4))
            for g in LABELS:
                part = s.loc[s.observed_pattern.eq(g)].sort_values('time')
                ax.plot(part.time, part.n_observed_local, label=LABELS[g], color=COLORS[g])
            ax.axhline(cfg.minimum_prediction_patients, color='grey', ls='--')
            ax.set(xlabel='Years on recorded analysis axis', ylabel='Unique patients in local window')
            ax.legend(fontsize=7)
            fig.suptitle(variable+' — local observation support')
            save(fig, safe_name+'_support')
        if len(d):
            fig, ax = plt.subplots(figsize=(8, 4))
            observed = tables['00_aggregated_observations']
            observed = observed.loc[observed.variable.eq(variable)] if len(observed) else observed
            if len(observed):
                for g, part in observed.groupby('observed_pattern'):
                    part = part.loc[part.status.eq('DESCRIPTIVE')]
                    ax.scatter(part.time, part['mean'], color=COLORS[g], marker='o', s=15, alpha=.6)
            for (g, model), part in d.groupby(['observed_pattern', 'model'], sort=False):
                for j, segment in enumerate(_segments(part, cfg.grid_step)):
                    ax.plot(segment.time, segment.estimate, color=COLORS[g],
                        ls={'A': ':', 'B': '--', 'C': '-'}[model], label=LABELS[g]+' '+model if j == 0 else None)
                    if model == 'C' and 'sim95_low' in segment:
                        ax.fill_between(segment.time.to_numpy(float), segment.sim95_low.to_numpy(float),
                            segment.sim95_high.to_numpy(float), color=COLORS[g], alpha=.12)
            ax.set(xlabel='Years on recorded analysis axis', ylabel=d.effect_scale.iloc[0])
            ax.legend(fontsize=6, ncol=2)
            fig.suptitle(variable+' — A/B/C supported mean atlas')
            save(fig, safe_name+'_atlas')
        v = tables['04_temporal_derivatives']
        v = v.loc[v.variable.eq(variable)] if len(v) else v
        if len(v):
            fig, ax = plt.subplots(figsize=(8, 4))
            for g, part in v.groupby('observed_pattern', sort=False):
                for j, segment in enumerate(_segments(part, cfg.grid_step)):
                    ax.plot(segment.time, segment.estimate, color=COLORS[g], label=LABELS[g] if j == 0 else None)
                    ax.fill_between(segment.time.to_numpy(float), segment.sim95_low.to_numpy(float),
                        segment.sim95_high.to_numpy(float), color=COLORS[g], alpha=.12)
            ax.axhline(0, color='grey', lw=.7)
            ax.set(xlabel='Years', ylabel=v.effect_scale.iloc[0])
            ax.legend(fontsize=7)
            fig.suptitle(variable+' — velocity with simultaneous bands')
            save(fig, safe_name+'_velocity')
        b = tables['04_simultaneous_bands']
        b = b.loc[b.variable.eq(variable)] if len(b) else b
        if len(b):
            for kind, contrasts in b.groupby('contrast_type'):
                fig, axes = plt.subplots(3, 2, figsize=(11, 8), squeeze=False)
                for ax, (pair, part) in zip(axes.flat, contrasts.groupby('contrast')):
                    for segment in _segments(part, cfg.grid_step):
                        x = segment.time.to_numpy(float)
                        ax.plot(x, segment.estimate)
                        ax.fill_between(x, segment.sim95_low.to_numpy(float), segment.sim95_high.to_numpy(float), alpha=.15)
                        significant = segment.sim95_low.gt(0) | segment.sim95_high.lt(0)
                        ax.scatter(x[significant], segment.estimate.to_numpy(float)[significant], color='darkred', s=12)
                    ax.axhline(0, color='grey', lw=.6)
                    ax.set_title(pair.replace('_observed', ''), fontsize=7)
                fig.suptitle(variable+' — six pairs: '+kind+'; red = simultaneous separation')
                fig.tight_layout(rect=(0, 0, 1, .92))
                save(fig, safe_name+'_contrasts_'+kind)
    tests = tables['02_global_temporal_tests']
    eligibility = tables['00_variable_eligibility']
    if len(eligibility):
        variables = list(eligibility.variable.unique())
        names = ['linear_group_time', 'common_nonlinearity', 'group_specific_curvature', 'global_temporal_heterogeneity']
        matrix = np.zeros((len(variables), len(names)))
        for i, variable in enumerate(variables):
            for j, name in enumerate(names):
                d = tests.loc[tests.variable.eq(variable) & tests.test.eq(name)] if len(tests) else tests
                if len(d) and d.iloc[0].status == 'ESTIMATED':
                    q = d.iloc[0].q_value
                    matrix[i, j] = 2 if np.isfinite(q) and q < .05 else 1
        fig, ax = plt.subplots(figsize=(9, max(3, len(variables)*.22)))
        ax.imshow(matrix, aspect='auto', vmin=0, vmax=2, cmap=ListedColormap(['#aaaaaa', '#dce9ef', '#377ca1']))
        ax.set_yticks(range(len(variables)), variables, fontsize=7)
        ax.set_xticks(range(4), names, rotation=20, ha='right', fontsize=7)
        fig.suptitle('Heterogeneity evidence: grey nonestimable; pale q≥.05; blue q<.05')
        save(fig, 'heterogeneity_heatmap')
        # Mechanically related and independent variables never share a domain bar.
        domain_table = eligibility.assign(domain_label=eligibility.domain+' / '+eligibility.interpretation_group)
        domains = list(domain_table.domain_label.unique())
        fig, ax = plt.subplots(figsize=(8, max(3, len(domains)*.45)))
        estimated = domain_table.groupby('domain_label').eligible_C.sum().reindex(domains)
        totals = domain_table.groupby('domain_label').size().reindex(domains)
        ax.barh(domains, totals, color='#aaaaaa', label='Inventoried')
        ax.barh(domains, estimated, color='#377ca1', label='C support eligible (before fit QC)')
        ax.set_xlabel('Variables; independent and mechanically related labeled in registry')
        ax.legend(fontsize=8)
        fig.suptitle('Evidence dashboard by recorded clinical domain')
        save(fig, 'domain_evidence_dashboard')
    return paths

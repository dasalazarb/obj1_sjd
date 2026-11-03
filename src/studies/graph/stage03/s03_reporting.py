"""Source-backed aggregate figures with publication-cell suppression.

All products remain internal/restricted pending publication review. Figure CSVs
contain exactly the aggregate data used by the panel, never individual records.
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CAPTION='EXPLORATORY / conditional on frozen labels; no clinical validation'


def save_figure(fig,folder,name,source=None):
    folder.mkdir(parents=True,exist_ok=True)
    fig.text(.5,-.02,CAPTION,ha='center',fontsize=8)
    if source is not None: source.to_csv(folder/f'{name}.csv',index=False)
    for extension in ('png','pdf'):
        fig.savefig(folder/f'{name}.{extension}',bbox_inches='tight',dpi=180)
    plt.close(fig)


def suppress_rows(frame,counts,minimum):
    """Suppress whole rows including complementary counts and effect estimates."""
    if frame.empty: return frame.copy()
    counts=[c for c in counts if c in frame]
    if not counts: return frame.iloc[:0].copy()
    small=pd.Series(False,index=frame.index)
    for c in counts:
        value=pd.to_numeric(frame[c],errors='coerce')
        small |= value.isna() | value.between(1,minimum-1)
    return frame.loc[~small].copy()


def _empty(ax,message='NOT_ESTIMABLE / unavailable or small cells suppressed'):
    ax.text(.5,.5,message,ha='center',va='center',wrap=True,transform=ax.transAxes)


def figures(anchors,trajectories,pop,folder, *, flow=None,domains=None,context=None,
            missingness=None,minimum_cell=5,primary_reference_id='R0'):
    flow=flow if flow is not None else pd.DataFrame()
    domains=domains if domains is not None else pd.DataFrame()
    context=context if context is not None else pd.DataFrame()
    missingness=missingness if missingness is not None else pd.DataFrame()
    fig,ax=plt.subplots(figsize=(9,5))
    part=anchors.loc[anchors.status.eq('COMPLETE')].copy() if not anchors.empty else anchors
    part=suppress_rows(part,['n_group','n_rest'],minimum_cell)
    continuous=part.loc[part.effect_type.eq('cliffs_delta')] if 'effect_type' in part else part.iloc[:0]
    if not continuous.empty:
        for i,row in enumerate(continuous.itertuples()):
            ax.errorbar(row.effect,i,xerr=[[max(0,row.effect-row.ci_low)],[max(0,row.ci_high-row.effect)]],fmt='o')
        ax.set(yticks=range(len(continuous)),yticklabels=[f'{r.representation_id}/C{r.community_id}: {r.source_column}' for r in continuous.itertuples()])
    else: _empty(ax,'Independent anchor inference is not estimable; see validation status.')
    ax.axvline(0,color='gray',lw=.8); ax.set(title='Independent anchors: conditional 95% intervals',xlabel="Cliff's delta")
    save_figure(fig,folder,'03_graph_independent_anchors',continuous)

    # One panel for every endpoint; unlike units never share a y axis.
    endpoints=sorted(trajectories.endpoint.dropna().unique()) if 'endpoint' in trajectories else []
    for endpoint in endpoints or [None]:
        fig,ax=plt.subplots(figsize=(9,5))
        part=trajectories.loc[trajectories.endpoint.eq(endpoint)] if endpoint else trajectories
        part=suppress_rows(part,['n_patients','n_missing_patients','n_missing_visits'],minimum_cell)
        if not part.empty:
            for (representation,community),group in part.groupby(['representation_id','community_id']):
                group=group.sort_values('time_bin_year')
                ax.plot(group.time_bin_year,group['median'],marker='o',label=f'{representation}/C{community}')
            ax.legend(fontsize=7); ax.set(ylabel=endpoint)
        else: _empty(ax,'No publishable post-index observations; see longitudinal eligibility.')
        ax.set(title='Observed post-baseline medians',xlabel='Years since frozen clinical baseline')
        name='03_graph_longitudinal_'+re.sub(r'[^A-Za-z0-9_-]','_',endpoint) if endpoint else '03_graph_trajectories'
        save_figure(fig,folder,name,part)

    fig,ax=plt.subplots(figsize=(9,4))
    part=pop.loc[pop.representation_id.eq(primary_reference_id)].copy() if 'n_patients' in pop else pd.DataFrame()
    # Suppress entire communities when a Pop cell is small: totals and other
    # cells must not disclose the omitted cell by subtraction.
    if not part.empty:
        small=part.loc[pd.to_numeric(part.n_patients,errors='coerce').between(1,minimum_cell-1),'community_id']
        part=part.loc[~part.community_id.isin(small)]
        if not part.empty:
            part.pivot(index='community_id',columns='population',values='n_patients').fillna(0).plot.bar(stacked=True,ax=ax)
        else: _empty(ax)
    else: _empty(ax)
    if not part.empty:
        part=part[['representation_id','community_id','population','n_patients']]
    ax.set(title='Frozen primary communities and observed Pop labels',ylabel='Patients')
    save_figure(fig,folder,'03_graph_pop_crosswalk',part)

    fig,ax=plt.subplots(figsize=(10,5))
    cols=['n_baseline','n_mapper_covered','n_uncovered','n_stable_hard','n_provisional_hard']
    part=suppress_rows(flow,cols,minimum_cell)
    part=part[['representation_id',*cols]] if not part.empty else part
    if not part.empty: part.set_index('representation_id')[cols].plot.bar(ax=ax)
    else: _empty(ax)
    ax.set(title='Population flow: overlapping descriptive counts',ylabel='Patients')
    save_figure(fig,folder,'03_graph_population_flow',part)

    fig,ax=plt.subplots(figsize=(12,6))
    part=domains.loc[domains.status.eq('DESCRIPTIVE')].copy() if 'status' in domains else domains
    part=suppress_rows(part,['n_group','n_observed','n_missing'],minimum_cell)
    if not part.empty:
        part['group_label']=part.representation_id+'/C'+part.community_id.astype(str)
        matrix=part.pivot_table(index=['clinical_domain','source_column'],columns='group_label',values='observed_fraction',dropna=True)
        if not matrix.empty:
            artist=ax.imshow(matrix.to_numpy(),aspect='auto',vmin=0,vmax=1)
            ax.set(xticks=range(len(matrix.columns)),xticklabels=matrix.columns,
                   yticks=range(len(matrix)),yticklabels=[' / '.join(idx) for idx in matrix.index])
            lookup=part.set_index(['clinical_domain','source_column','group_label'])
            for i,key in enumerate(matrix.index):
                for j,label in enumerate(matrix.columns):
                    if (*key,label) in lookup.index:
                        row=lookup.loc[(*key,label)]
                        ax.text(j,i,f'{int(row.n_observed)}/{int(row.n_group)}',ha='center',va='center',fontsize=6)
            fig.colorbar(artist,ax=ax,label='Observed fraction per clinical source')
        else: _empty(ax)
    else: _empty(ax)
    ax.set_title('Clinical domains: observed-source availability (n observed / n group)')
    save_figure(fig,folder,'03_graph_domains_heatmap',part)

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    parts=[]
    for ax,metric,denom in zip(axes,['n_covered','ari_primary'],['n_patients','n_common']):
        part=context.loc[context.status.eq('COMPLETE') & context[metric].notna()].copy() if metric in context else pd.DataFrame()
        part=suppress_rows(part,[denom,'n_uncovered'] if metric=='n_covered' else [denom],minimum_cell)
        if metric=='n_covered' and not part.empty:
            part['n_uncovered_check']=part[denom]-part[metric]
            part=suppress_rows(part,['n_covered','n_uncovered_check'],minimum_cell)
        if not part.empty:
            # One value per registered diagnostic, retaining the true denominator.
            part=part.drop_duplicates(['experiment_id','source_table',metric])
            labels=part.experiment_id+' ('+part.mapper_or_sparse_graph+')'
            values=part[metric]/part[denom] if metric=='n_covered' else part[metric]
            ax.bar(range(len(part)),values); ax.set(xticks=range(len(part)),xticklabels=labels)
            ax.tick_params(axis='x',labelrotation=90,labelsize=7)
            parts.append(part.assign(figure_metric=metric))
        else: _empty(ax)
        ax.set_title('Coverage / n patients' if metric=='n_covered' else 'ARI / n common; DG is cross-algorithm')
    save_figure(fig,folder,'03_graph_method_robustness',pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['status','figure_metric']))

    fig,ax=plt.subplots(figsize=(11,5))
    part=missingness.loc[missingness.measurement_basis.eq('source_observation_units') & missingness.group_type.eq('clinical_domain')].copy() if {'measurement_basis','group_type'}<=set(missingness) else pd.DataFrame()
    part=suppress_rows(part,['n_patients'],minimum_cell)
    if not part.empty:
        part=part.loc[part.observed_fraction.notna() & part.n_observation_units.notna()]
        denominator=part.n_patients*part.n_observation_units
        part['n_observed_cells']=(part.observed_fraction*denominator).round()
        part['n_missing_cells']=(denominator-part.n_observed_cells).round()
        part=suppress_rows(part,['n_observed_cells','n_missing_cells'],minimum_cell)
        labels=part.representation_id+' / '+part['group'].astype(str)
        ax.bar(range(len(part)),part.observed_fraction); ax.set(xticks=range(len(part)),xticklabels=labels,ylim=(0,1))
        ax.tick_params(axis='x',labelrotation=90)
    else: _empty(ax)
    ax.set(title='Availability by domain: source observation units',ylabel='Observed fraction (patients × observation units)')
    save_figure(fig,folder,'03_graph_missingness_domain',part)

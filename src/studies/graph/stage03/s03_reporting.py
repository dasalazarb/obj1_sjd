"""Publish all estimable and negative results with source-backed figures."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def save_figure(fig,folder,name):
    for extension in ('png','pdf'):
        fig.savefig(folder/f'{name}.{extension}',bbox_inches='tight',dpi=180)
    plt.close(fig)


def figures(anchors,trajectories,pop,folder):
    fig,ax=plt.subplots(figsize=(9,5))
    part=anchors.loc[anchors.status.eq('COMPLETE')].copy() if not anchors.empty else anchors
    if not part.empty:
        continuous=part.loc[part.effect_type.eq('cliffs_delta')]
        for i,row in enumerate(continuous.itertuples()):
            ax.errorbar(row.effect,i,xerr=[[max(0,row.effect-row.ci_low)],[max(0,row.ci_high-row.effect)]],fmt='o')
        ax.set(yticks=range(len(continuous)),yticklabels=[f'{r.representation_id}/C{r.community_id}: {r.source_column}' for r in continuous.itertuples()])
    else: ax.text(.5,.5,'Independent anchor inference is not estimable; see validation status.',ha='center',va='center',wrap=True,transform=ax.transAxes)
    ax.axvline(0,color='gray',lw=.8); ax.set(title='Exploratory independent anchors: conditional 95% intervals',xlabel="Cliff's delta")
    save_figure(fig,folder,'03_graph_independent_anchors')
    fig,ax=plt.subplots(figsize=(9,5))
    if not trajectories.empty:
        # Separate endpoints rather than plotting unlike units on one scale.
        endpoint=trajectories.endpoint.iloc[0]
        for (representation,community),part in trajectories.loc[trajectories.endpoint.eq(endpoint)].groupby(['representation_id','community_id']):
            part=part.sort_values('time_bin_year')
            ax.plot(part.time_bin_year,part['median'],marker='o',label=f'{representation}/C{community}')
        ax.set(ylabel=endpoint); ax.legend(fontsize=7)
    else: ax.text(.5,.5,'No eligible post-index observations; see longitudinal eligibility.',ha='center',va='center',transform=ax.transAxes)
    ax.set(title='Observed post-baseline trajectories: descriptive medians',xlabel='Years since frozen clinical baseline')
    save_figure(fig,folder,'03_graph_trajectories')
    fig,ax=plt.subplots(figsize=(9,4))
    if not pop.empty and 'n_patients' in pop:
        part=pop.loc[pop.representation_id.eq('R0')]
        if not part.empty:
            table=part.pivot(index='community_id',columns='population',values='n_patients').fillna(0)
            table.plot.bar(stacked=True,ax=ax)
    ax.set(title='Frozen primary communities and observed Pop labels',ylabel='Patients')
    save_figure(fig,folder,'03_graph_pop_crosswalk')

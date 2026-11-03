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
    if source is not None:
        if len(source.columns)==0: source=pd.DataFrame(columns=['status','reason'])
        source.to_csv(folder/f'{name}.csv',index=False)
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
        small |= value.isna() | (value.gt(0) & value.lt(minimum))
    return frame.loc[~small].copy()


def _empty(ax,message='NOT_ESTIMABLE / unavailable or small cells suppressed'):
    ax.text(.5,.5,message,ha='center',va='center',wrap=True,transform=ax.transAxes)


def figures(anchors,trajectories,pop,folder, *, flow=None,domains=None,context=None,
            missingness=None,minimum_cell=5,primary_reference_id='R0',baseline_table=None,marker_config=None):
    flow=flow if flow is not None else pd.DataFrame()
    domains=domains if domains is not None else pd.DataFrame()
    context=context if context is not None else pd.DataFrame()
    missingness=missingness if missingness is not None else pd.DataFrame()
    baseline_table=baseline_table if baseline_table is not None else pd.DataFrame()
    marker_config=marker_config or []
    inventory=[]
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

    # Separate instruments and representations. Points are observed medians at
    # real median visit times; IQRs are descriptive, never confidence intervals.
    endpoints=sorted(set(trajectories.endpoint.dropna())) if 'endpoint' in trajectories else []
    endpoints=sorted(set(endpoints)|{m['source_column'] for m in marker_config})
    units={m['source_column']:m.get('unit','source units') for m in marker_config}
    for endpoint in endpoints or [None]:
        raw=trajectories.loc[trajectories.endpoint.eq(endpoint)].copy() if endpoint and 'endpoint' in trajectories else trajectories.iloc[:0].copy()
        part=suppress_rows(raw,['n_patients','n_missing_patients','n_missing_visits'],minimum_cell)
        if 'date_semantics_review_status' in part:
            part=part.loc[part.date_semantics_review_status.eq('VERIFIED')]
        part=part.assign(unit=units.get(endpoint,'source units'),denominator='verified endpoint visits in [lower,upper) year windows')
        reps=sorted(part.representation_id.unique()) if not part.empty else ['R0','R2']
        fig,axes=plt.subplots(1,len(reps),figsize=(6*len(reps),4),squeeze=False,constrained_layout=True)
        for ax,rep in zip(axes.flat,reps):
            subset=part.loc[part.representation_id.eq(rep)] if not part.empty else part
            for community,group in subset.groupby('community_id') if not subset.empty else []:
                group=group.sort_values('time_bin_year')
                x=group.time_median_years if 'time_median_years' in group else group.time_bin_year
                lo=group['median']-group.q25 if 'q25' in group else np.zeros(len(group))
                hi=group.q75-group['median'] if 'q75' in group else np.zeros(len(group))
                ax.errorbar(np.asarray(x),group['median'].to_numpy(),yerr=np.array([lo,hi]),fmt='o',capsize=3,label=f'C{community}')
                for t,value,n in zip(x,group['median'],group.n_patients): ax.annotate(f'n={int(n)}',(t,value),xytext=(3,6),textcoords='offset points',fontsize=7)
            if subset.empty: _empty(ax,'NOT_ESTIMABLE / no verified publishable follow-up')
            else: ax.legend(fontsize=8)
            ax.set(title=rep+(' EXPLORATORY_PROVISIONAL' if rep=='R2' else ''),xlabel='Actual years since frozen baseline',ylabel=units.get(endpoint,'source units'))
        fig.suptitle(str(endpoint or 'Longitudinal observations')+' — median / IQR by time window')
        name='03_graph_longitudinal_'+re.sub(r'[^A-Za-z0-9_-]','_',endpoint) if endpoint else '03_graph_trajectories'
        if endpoints and endpoint==endpoints[0]:
            save_figure_copy(fig,folder,'03_graph_trajectories',part)
        save_figure(fig,folder,name,part)
        inventory.append(dict(figure=name,status='DESCRIPTIVE' if not part.empty else 'NOT_ESTIMABLE',reason='' if not part.empty else 'NO_VERIFIED_PUBLISHABLE_FOLLOWUP'))

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

    # Sequential partitions have separate panels and explicit parent totals.
    flow_parts=[]
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    partitions=[('Mapper coverage','n_baseline',['n_mapper_covered','n_uncovered']),
                ('Supported hard assignment','n_mapper_covered',['n_hard_supported','n_covered_without_supported_hard']),
                ('Frozen stability label','n_hard_supported',['n_stable_hard','n_provisional_hard'])]
    data=flow.copy()
    if not data.empty: data['n_covered_without_supported_hard']=data.n_mapper_covered-data.n_hard_supported
    for ax,(title,parent,children) in zip(axes,partitions):
        part=suppress_rows(data,[parent,*children],minimum_cell)
        # Suppress all stages for a representation when any partition is small;
        # another panel must not reconstruct its suppressed complement.
        safe=suppress_rows(data,sorted({k for _,p,cs in partitions for k in [p,*cs]}),minimum_cell)
        if not part.empty: part=part.loc[part.representation_id.isin(safe.representation_id)]
        if not part.empty:
            part=part[['representation_id',parent,*children]].copy()
            part.set_index('representation_id')[children].plot.barh(stacked=True,ax=ax)
            for y,row in enumerate(part.itertuples()): ax.text(0,y,f'parent n={getattr(row,parent)}',fontsize=8)
            flow_parts.append(part.assign(stage=title,denominator_field=parent))
        else: _empty(ax)
        ax.set(title=title,xlabel='Patients')
        if not part.empty: ax.legend(fontsize=6)
    save_figure(fig,folder,'03_graph_population_flow',pd.concat(flow_parts,ignore_index=True) if flow_parts else pd.DataFrame(columns=['representation_id','stage','status']))

    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    pop_columns=['n_pop1','n_pop2','n_pop3','n_unclassified_pop','n_missing_pop','n_invalid_pop']
    part=suppress_rows(flow,['n_hard_supported',*pop_columns],minimum_cell)
    if not part.empty:
        part=part[['representation_id','n_hard_supported',*pop_columns]].copy()
        part.set_index('representation_id')[pop_columns].plot.barh(stacked=True,ax=ax)
    else: _empty(ax)
    ax.set(title='Published baseline Pop within supported hard assignments',xlabel='Patients (parent: n hard supported)')
    save_figure(fig,folder,'03_graph_population_flow_pop',part)

    # Availability combines counts of patient-source cells, never clinical medians.
    domain_rows=[]
    if {'clinical_domain','source_column','representation_id','community_id','status'}<=set(domains):
        for key,group in domains.groupby(['representation_id','community_id','clinical_domain']):
            if not group.status.eq('DESCRIPTIVE').all(): continue
            n_observed=group.n_observed.sum(); denominator=group.n_group.sum()
            domain_rows.append(dict(representation_id=key[0],community_id=key[1],clinical_domain=key[2],
                n_observed=n_observed,n_missing=denominator-n_observed,n_denominator=denominator,
                n_sources=group.source_column.nunique(),observed_fraction=n_observed/denominator if denominator else np.nan,
                measurement_basis='patient_source_observation_cells',status='DESCRIPTIVE'))
    part=suppress_rows(pd.DataFrame(domain_rows),['n_observed','n_missing','n_denominator'],minimum_cell)
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    if not part.empty:
        part['group_label']=part.representation_id+'/C'+part.community_id.astype(str)
        matrix=part.pivot(index='clinical_domain',columns='group_label',values='observed_fraction')
        artist=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1)
        ax.set(xticks=range(len(matrix.columns)),xticklabels=matrix.columns,yticks=range(len(matrix)),yticklabels=matrix.index)
        fig.colorbar(artist,ax=ax,label='Observed / eligible patient-source cells')
    else: _empty(ax)
    ax.set_title('D01–D07 availability; clinical magnitudes are shown separately')
    save_figure(fig,folder,'03_graph_domains_heatmap',part)

    for marker in marker_config:
        source=marker['source_column']
        part=baseline_table.loc[baseline_table.source_column.eq(source) & baseline_table.representation_id.isin(['R0','R2'])].copy() if 'source_column' in baseline_table else pd.DataFrame()
        part=suppress_rows(part,['n_patients','n_observed','n_missing'],minimum_cell)
        if not part.empty: part=part.loc[part.n_observed.gt(0) & part['median'].notna()]
        if not part.empty:
            part=part[['representation_id','community_id','validation_role','n_patients','n_observed','n_missing','median','q25','q75']].copy()
            part['unit']=marker['unit']; part['status']=np.where(part.representation_id.eq('R2'),'DESCRIPTIVE_PROVISIONAL','DESCRIPTIVE')
            part['interpretation']='observed_baseline_not_independent_validation'
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for ax,rep in zip(axes,['R0','R2']):
            group=part.loc[part.representation_id.eq(rep)] if not part.empty else part
            if not group.empty:
                x=np.arange(len(group)); ax.errorbar(x,group['median'].to_numpy(),yerr=np.array([group['median']-group.q25,group.q75-group['median']]),fmt='o',capsize=4)
                ax.set(xticks=x,xticklabels=[f'C{r.community_id}\nn={r.n_observed}/{r.n_patients}'+('\ndefining/descriptive' if r.validation_role in {'defining','derived_from_defining'} else '') for r in group.itertuples()])
            else: _empty(ax)
            ax.set(title=rep+(' EXPLORATORY_PROVISIONAL' if rep=='R2' else ' primary reference'),ylabel=marker['unit'])
        fig.suptitle(marker['label']+' — observed baseline median / IQR')
        name='03_graph_clinical_marker_panels_'+re.sub(r'[^A-Za-z0-9_-]','_',source)
        save_figure(fig,folder,name,part)
        inventory.append(dict(figure=name,status='DESCRIPTIVE' if not part.empty else 'NOT_ESTIMABLE',reason='' if not part.empty else 'NO_PUBLISHABLE_OBSERVED_MARKER'))

    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    parts=[]
    for ax,metric,kind in zip(axes,['n_covered','ari_primary','ari_primary'],['Mapper coverage','Mapper concordance','DG cross-algorithm concordance']):
        part=context.loc[context.status.eq('COMPLETE') & context[metric].notna()].copy() if metric in context else pd.DataFrame()
        if not part.empty:
            sparse=part.mapper_or_sparse_graph.eq('sparse_graph')
            part=part.loc[sparse if kind.startswith('DG') else ~sparse]
            denom='n_patients' if metric=='n_covered' else 'n_common'
            if metric=='n_covered':
                part['n_uncovered_check']=part[denom]-part[metric]
                part=suppress_rows(part,[denom,'n_covered','n_uncovered_check'],minimum_cell)
            else: part=suppress_rows(part,[denom],minimum_cell)
            part=part.drop_duplicates(['experiment_id','source_table',metric])
        if not part.empty:
            labels=[f'{r.experiment_id} (n={int(getattr(r,denom))})' for r in part.itertuples()]
            values=part[metric]/part[denom] if metric=='n_covered' else part[metric]
            ax.barh(labels,values); parts.append(part[['experiment_id','source_table',metric,denom,'status','mapper_or_sparse_graph']].assign(figure_metric=kind))
        else: _empty(ax)
        ax.set_title(kind,fontsize=10)
    save_figure(fig,folder,'03_graph_method_robustness',pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['status','figure_metric']))

    part=missingness.loc[missingness.measurement_basis.eq('source_observation_units') & missingness.group_type.eq('clinical_domain')].copy() if {'measurement_basis','group_type'}<=set(missingness) else pd.DataFrame()
    part=suppress_rows(part,['n_patients'],minimum_cell)
    if not part.empty:
        part=part.loc[part['group'].isin([f'D{i:02d}' for i in range(1,10)]) & part.observed_fraction.notna() & part.n_observation_units.notna()]
        denominator=part.n_patients*part.n_observation_units
        # Keep fractional numerators as reported; rounding could hide a small cell.
        part['n_observed_cells']=part.observed_fraction*denominator
        part['n_missing_cells']=denominator-part.n_observed_cells
        part=suppress_rows(part,['n_observed_cells','n_missing_cells'],minimum_cell)
        part=part[['representation_id','group','measurement_basis','n_patients','n_observation_units','observed_fraction','n_observed_cells','n_missing_cells']]
    reps=sorted(part.representation_id.unique()) if not part.empty else ['R0','R2']
    fig,axes=plt.subplots(1,len(reps),figsize=(5*len(reps),4),squeeze=False,constrained_layout=True)
    for ax,rep in zip(axes.flat,reps):
        group=part.loc[part.representation_id.eq(rep)] if not part.empty else part
        if not group.empty: ax.barh(group['group'],group.observed_fraction)
        else: _empty(ax)
        ax.set(title=rep+' source observation units',xlim=(0,1),xlabel='Observed / (patients × source units)')
    save_figure(fig,folder,'03_graph_missingness_domain',part)
    registered={row['figure'] for row in inventory}
    for source in sorted(folder.glob('*.csv')):
        if source.stem in registered or source.stem=='03_graph_figure_status': continue
        frame=pd.read_csv(source)
        inventory.append(dict(figure=source.stem,status='DESCRIPTIVE' if len(frame) else 'NOT_ESTIMABLE',
            reason='' if len(frame) else 'NO_PUBLISHABLE_SUPPORT_OR_SOURCE_UNAVAILABLE'))
    pd.DataFrame(inventory).to_csv(folder/'03_graph_figure_status.csv',index=False)


def save_figure_copy(fig,folder,name,source):
    """Historical figure name is an alias of the first endpoint panel."""
    folder.mkdir(parents=True,exist_ok=True)
    source.to_csv(folder/f'{name}.csv',index=False)
    for extension in ('png','pdf'): fig.savefig(folder/f'{name}.{extension}',bbox_inches='tight',dpi=180)

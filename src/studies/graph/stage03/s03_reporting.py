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
from .s03_figure_qc import (DISCLOSURE_POLICY_VERSION, WITHHELD_MESSAGE, TIERS,
    validate_figure_policy, evaluate_characterization_panel, select_n2_sources, drawable_panel)

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
    ax.set(xticks=[],yticks=[])
    ax.text(.5,.5,message,ha='center',va='center',wrap=True,transform=ax.transAxes)


def figures(anchors,trajectories,pop,folder, *, flow=None,domains=None,context=None,
            missingness=None,minimum_cell=5,primary_reference_id='R0',baseline_table=None,marker_config=None,raw_matrix=None,roles=None):
    flow=flow if flow is not None else pd.DataFrame()
    domains=domains if domains is not None else pd.DataFrame()
    context=context if context is not None else pd.DataFrame()
    missingness=missingness if missingness is not None else pd.DataFrame()
    baseline_table=baseline_table if baseline_table is not None else pd.DataFrame()
    marker_config=marker_config or []
    inventory=[]
    sources={}
    def emit(fig,folder,name,source):
        save_figure(fig,folder,name,source)
        sources[name]=source.copy()
    def emit_copy(fig,folder,name,source):
        save_figure_copy(fig,folder,name,source)
        sources[name]=source.copy()
    if raw_matrix is not None:
        # One whole-source privacy view feeds every historical clinical panel.
        protected_matrix=safe_characterization_panels(raw_matrix,minimum_cell)
        keys=protected_matrix[['representation_id','source_column']].drop_duplicates()
        baseline_table=baseline_table.merge(keys,on=['representation_id','source_column'],how='inner') if not baseline_table.empty else baseline_table
        if not baseline_table.empty:
            # Historical marker plots are numeric median/IQR panels. Binary or
            # categorical sources retain their typed displays in the tier panels.
            numeric_keys=protected_matrix.loc[protected_matrix.data_type.eq('numeric'),['representation_id','source_column']].drop_duplicates()
            baseline_table=baseline_table.merge(numeric_keys,on=['representation_id','source_column'],how='inner')
        domains=domains.merge(keys,on=['representation_id','source_column'],how='inner') if 'source_column' in domains else domains
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
    emit(fig,folder,'03_graph_independent_anchors',continuous)

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
            emit_copy(fig,folder,'03_graph_trajectories',part)
        emit(fig,folder,name,part)
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
    emit(fig,folder,'03_graph_pop_crosswalk',part)

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
    emit(fig,folder,'03_graph_population_flow',pd.concat(flow_parts,ignore_index=True) if flow_parts else pd.DataFrame(columns=['representation_id','stage','status']))

    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    pop_columns=['n_pop1','n_pop2','n_pop3','n_unclassified_pop','n_missing_pop','n_invalid_pop']
    part=suppress_rows(flow,['n_hard_supported',*pop_columns],minimum_cell)
    if not part.empty:
        part=part[['representation_id','n_hard_supported',*pop_columns]].copy()
        part.set_index('representation_id')[pop_columns].plot.barh(stacked=True,ax=ax)
    else: _empty(ax)
    ax.set(title='Published baseline Pop within supported hard assignments',xlabel='Patients (parent: n hard supported)')
    emit(fig,folder,'03_graph_population_flow_pop',part)

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
    else:
        if raw_matrix is not None and roles is not None:
            diagnostic=evaluate_characterization_panel(raw_matrix,protected_matrix,roles,
                representation_id='|'.join(sorted(set(raw_matrix.representation_id) | {'R0','R2'})),
                tier='transversal',selected_sources=list(raw_matrix.source_column.unique()),minimum_cell=minimum_cell)
            _empty(ax,WITHHELD_MESSAGE if diagnostic['status']=='WITHHELD_DISCLOSURE_CONTROL'
                else 'NO_DISPLAYABLE_DOMAIN_AVAILABILITY / see source presentation inventory')
        else: _empty(ax)
    visible=sorted(set(part.representation_id)) if not part.empty else []
    limitation=' / NO_R0_R2_DISPLAYABLE_PANELS' if not set(visible)&{'R0','R2'} else ''
    ax.set_title('D01–D07 availability; visible: '+(', '.join(visible) or 'none')+limitation)
    emit(fig,folder,'03_graph_domains_heatmap',part)

    for marker in marker_config:
        source=marker['source_column']
        part=baseline_table.loc[baseline_table.source_column.eq(source) & baseline_table.representation_id.isin(['R0','R2'])].copy() if 'source_column' in baseline_table else pd.DataFrame()
        part=suppress_rows(part,['n_patients','n_observed','n_missing'],minimum_cell)
        if not part.empty: part=part.loc[part.n_observed.gt(0) & part['median'].notna()]
        if not part.empty:
            part=part[['representation_id','community_id','source_column','validation_role','n_patients','n_observed','n_missing','median','q25','q75']].copy()
            part['unit']=marker['unit']; part['status']=np.where(part.representation_id.eq('R2'),'DESCRIPTIVE_PROVISIONAL','DESCRIPTIVE')
            part['interpretation']='observed_baseline_not_independent_validation'
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for ax,rep in zip(axes,['R0','R2']):
            group=part.loc[part.representation_id.eq(rep)] if not part.empty else part
            if not group.empty:
                x=np.arange(len(group)); ax.errorbar(x,group['median'].to_numpy(),yerr=np.array([group['median']-group.q25,group.q75-group['median']]),fmt='o',capsize=4)
                ax.set(xticks=x,xticklabels=[f'C{r.community_id}\nn={r.n_observed}/{r.n_patients}'+('\ndefining/descriptive' if r.validation_role in {'defining','derived_from_defining'} else '') for r in group.itertuples()])
            else:
                if raw_matrix is not None and roles is not None:
                    diagnostic=evaluate_characterization_panel(raw_matrix,protected_matrix,roles,
                        representation_id=rep,tier='transversal',selected_sources=[source],minimum_cell=minimum_cell)
                    if diagnostic['status']=='DISPLAYED_DESCRIPTIVE':
                        diagnostic.update(status='NOT_INTERPRETABLE_UNITS_OR_LEVELS',
                            reason='No drawable frozen numeric summary; no recoding was applied.')
                    _empty(ax,WITHHELD_MESSAGE if diagnostic['status']=='WITHHELD_DISCLOSURE_CONTROL'
                        else diagnostic['status']+'\n'+diagnostic['reason'])
                else: _empty(ax)
            ax.set(title=rep+(' EXPLORATORY_PROVISIONAL' if rep=='R2' else ' primary reference'),ylabel=marker['unit'])
        fig.suptitle(marker['label']+' — observed baseline median / IQR')
        name='03_graph_clinical_marker_panels_'+re.sub(r'[^A-Za-z0-9_-]','_',source)
        emit(fig,folder,name,part)
        if raw_matrix is not None and roles is not None:
            diagnostic=evaluate_characterization_panel(raw_matrix,protected_matrix,roles,
                representation_id='R0|R2',tier='transversal',selected_sources=[source],minimum_cell=minimum_cell)
            diagnostic.update(n_displayable_sources=int(not part.empty),
                n_communities_displayed=len(part[['representation_id','community_id']].drop_duplicates()) if not part.empty else 0)
            if part.empty and diagnostic['status']=='DISPLAYED_DESCRIPTIVE':
                diagnostic.update(status='NOT_INTERPRETABLE_UNITS_OR_LEVELS',display_status='NOT_INTERPRETABLE_UNITS_OR_LEVELS',
                    reason='No drawable frozen numeric summary; no recoding was applied.')
            inventory.append(dict(figure=name,**diagnostic))
        else:
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
    emit(fig,folder,'03_graph_method_robustness',pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['status','figure_metric']))

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
    emit(fig,folder,'03_graph_missingness_domain',part)
    registered={row['figure'] for row in inventory}
    for name,frame in sources.items():
        if name not in registered:
            inventory.append(dict(figure=name,status='DESCRIPTIVE' if len(frame) else 'NOT_ESTIMABLE',
                reason='Descriptive content displayed.' if len(frame) else 'NO_PUBLISHABLE_SUPPORT_OR_SOURCE_UNAVAILABLE'))
    for row in inventory:
        frame=sources[row['figure']]
        defaults=dict(representation_id='|'.join(sorted(set(frame.representation_id))) if 'representation_id' in frame else 'not_applicable',
            tier='transversal',display_status=row['status'],inference_status='EXISTING_ANALYSIS_GATES',
            clinical_inference_status='EXISTING_ANALYSIS_GATES',visibility='INTERNAL_RESTRICTED',
            disclosure_policy_version=DISCLOSURE_POLICY_VERSION,
            n_prespecified_sources=None,n_sources_matching_tier=None,n_sources_with_observations=None,
            n_displayable_sources=frame.source_column.nunique() if 'source_column' in frame else None,
            n_source_partitions_withheld=None,
            n_communities_displayed=len(frame[['representation_id','community_id']].drop_duplicates())
                if {'representation_id','community_id'}<=set(frame) else 0)
        for key,value in defaults.items(): row.setdefault(key,value)
    pd.DataFrame(inventory).to_csv(folder/'03_graph_figure_status.csv',index=False)
    return sources


def save_figure_copy(fig,folder,name,source):
    """Historical figure name is an alias of the first endpoint panel."""
    folder.mkdir(parents=True,exist_ok=True)
    source.to_csv(folder/f'{name}.csv',index=False)
    for extension in ('png','pdf'): fig.savefig(folder/f'{name}.{extension}',bbox_inches='tight',dpi=180)


def safe_characterization_panels(matrix, minimum):
    """Suppress the entire representation/source partition, including complements.

    Identical suppression feeds clinical and availability views, so another panel
    cannot reconstruct an omitted count. Groups at the minimum require review.
    """
    if matrix.empty: return matrix.copy()
    counts=[c for c in ['n_group','n_observed','n_missing','n_tested','n_unknown','n_not_tested',
                        'n_pos','n_neg','n_category','n_stable_hard','n_provisional_hard'] if c in matrix]
    bad=pd.Series(False,index=matrix.index)
    for column in counts:
        values=pd.to_numeric(matrix[column],errors='coerce')
        bad |= values.gt(0) & values.lt(minimum)
    bad |= matrix.n_group.le(minimum)
    excluded=set(map(tuple,matrix.loc[bad,['representation_id','source_column']].to_numpy()))
    return matrix.loc[[tuple(k) not in excluded for k in matrix[['representation_id','source_column']].to_numpy()]].copy()


def characterization_figures(matrix, roles, folder, validation, minimum_cell=5, *, provenance=None):
    """Render only the conservative view; preserve metadata for withheld panels."""
    validate_figure_policy(validation)
    panel=validation.get('clinical_marker_panels',[])
    policy=validation.get('characterization',{})
    labels={m['source_column']:m['label'] for m in panel}
    protocol_sources=policy.get('figure_sources',[m['source_column'] for m in panel])
    n2_specs={m['source_column']:m for m in policy.get('figure_sources_N2',[]) if m.get('source_column')}
    labels.update({source:spec.get('label',source) for source,spec in n2_specs.items()})
    all_sources=list(dict.fromkeys([m['source_column'] for m in panel]+list(n2_specs)))
    safe=safe_characterization_panels(matrix,minimum_cell)
    selection=select_n2_sources(matrix,safe,roles,validation,minimum_cell,provenance)
    inventory=[]; figure_sources={}
    titles={'N1':'Defining sources — descriptive internal structure',
            'N2':'Related sources — documented dependence / descriptive only',
            'N3':'Independent candidates — review and statistical gates required'}
    for rep in ['R0','R2']:
        for level,tiers in TIERS.items():
            raw_roles=roles.loc[roles.representation_id.eq(rep) & roles.tier.isin(tiers)]
            requested=(protocol_sources if level=='N1' else list(n2_specs) if level=='N2'
                       else [m['source_column'] for m in panel if m['source_column'] in set(raw_roles.source_column)])
            sources=list(dict.fromkeys(s for s in requested if s in set(raw_roles.source_column)))
            selections=selection.loc[selection.representation_id.eq(rep)]
            eligible=(selections.loc[selections.selection_status.eq('SELECTED_DESCRIPTIVE'),'source_column'].tolist()
                      if level=='N2' else sources)
            diagnostic=evaluate_characterization_panel(matrix,safe,roles,representation_id=rep,
                tier=level,selected_sources=requested,eligible_sources=eligible,minimum_cell=minimum_cell)
            if level=='N2' and not eligible and len(selections) and selections.selection_status.eq('NOT_INTERPRETABLE_UNITS_OR_LEVELS').any():
                diagnostic.update(status='NOT_INTERPRETABLE_UNITS_OR_LEVELS',display_status='NOT_INTERPRETABLE_UNITS_OR_LEVELS',
                    reason='Selected laboratory text lacks standardized frozen levels or units; no recoding was applied.')
            fig,axes=plt.subplots(max(1,(len(sources)+1)//2),2,
                figsize=(13,3.8*max(1,(len(sources)+1)//2)),squeeze=False,constrained_layout=True)
            plotted=[]
            for ax,source in zip(axes.flat,sources):
                state=evaluate_characterization_panel(matrix,safe,roles,representation_id=rep,
                    tier=level,selected_sources=[source],eligible_sources=eligible,minimum_cell=minimum_cell)
                part=drawable_panel(safe.loc[safe.representation_id.eq(rep) & safe.source_column.eq(source)]) if source in eligible else safe.iloc[:0].copy()
                if part.empty:
                    if level=='N2' and source not in eligible:
                        row=selections.loc[selections.source_column.eq(source)].iloc[0]
                        state.update(status=row.display_status,reason=row.reason)
                    message=WITHHELD_MESSAGE if state['status']=='WITHHELD_DISCLOSURE_CONTROL' else state['status']+'\n'+state['reason']
                    if state['inference_status']=='REVIEW_PENDING': message+='\nIndependence review pending.'
                    _empty(ax,message)
                else:
                    part=part.sort_values(['community_id','category'])
                    x=np.arange(len(part)); is_numeric=part.data_type.eq('numeric').all()
                    if is_numeric:
                        ax.errorbar(x,part['median'],yerr=np.array([part['median']-part.q25,part.q75-part['median']]),fmt='o',capsize=3)
                    else:
                        ax.plot(x,part.proportion,'o'); ax.set_ylim(0,1)
                    ticks=[]
                    for row in part.itertuples():
                        state_label='\n'+row.inference_status if level=='N3' else ''
                        ticks.append(f'C{row.community_id}'+(f'/{row.category}' if row.category else '')+f'\nn={row.n_observed}/{row.n_group}'+state_label)
                    # Never invent or assert reviewed units from generic metadata.
                    unit=str(part.unit.iloc[0])
                    if unit in {'source_units','source_type',''} or 'pending' in unit.lower():
                        unit='Frozen source scale (units unverified)'
                    ax.set(xticks=x,xticklabels=ticks,ylabel=unit if is_numeric else 'Observed proportion')
                    plotted.append(part)
                if level=='N2':
                    spec=n2_specs[source]
                    ax.set_xlabel(f"N1 {spec.get('related_to','')} → {spec.get('expected_relationship','')} → N2 {source}\nDependent description; no independent validation",fontsize=7)
                ax.set_title(labels.get(source,source),fontsize=9)
            for ax in list(axes.flat)[len(sources):]: ax.axis('off')
            if not sources:
                axes.flat[0].axis('on'); _empty(axes.flat[0],diagnostic['status']+'\n'+diagnostic['reason'])
            fig.suptitle(f'{rep}'+(' — exploratory provisional' if rep=='R2' else ' — primary reference')+'\n'+titles[level],fontsize=11)
            name=f'03_graph_characterization_{level}_{rep}'
            data=pd.concat(plotted,ignore_index=True) if plotted else matrix.iloc[:0].copy()
            save_figure(fig,folder,name,data); figure_sources[name]=data
            inventory.append(dict(figure=name,**diagnostic))
    crosswalk=roles.loc[roles.representation_id.isin(['R0','R2']) & roles.source_column.isin(all_sources),
        ['source_column','representation_id','tier','independence_status','relationship_type','blocking_reasons']].copy()
    fig,ax=plt.subplots(figsize=(11,max(5,len(all_sources)*.6)),constrained_layout=True)
    codes={'N1_DEFINING':0,'N2_RELATED':1,'N3_CANDIDATE':2,'N3_REVIEWED':3,'PENDING_REVIEW':4,'UNAVAILABLE':5}
    if not crosswalk.empty:
        grid=crosswalk.pivot(index='source_column',columns='representation_id',values='tier').reindex(index=all_sources,columns=['R0','R2'])
        values=grid.apply(lambda column:column.map(codes)).to_numpy(float)
        ax.imshow(values,aspect='auto',vmin=0,vmax=5,cmap=plt.get_cmap('tab10',6))
        for i,source in enumerate(grid.index):
            for j,rep in enumerate(grid.columns):
                value=grid.loc[source,rep]
                ax.text(j,i,str(value).replace('_',' ') if pd.notna(value) else 'SOURCE UNAVAILABLE',ha='center',va='center',fontsize=7)
        ax.set(xticks=[0,1],xticklabels=['R0 primary','R2 provisional'],yticks=range(len(grid)),yticklabels=[labels[s] for s in grid.index])
    else: _empty(ax,'SOURCE_UNAVAILABLE / no source-role metadata')
    ax.set_title('Prespecified source roles by frozen representation\nNo patient measurements; absence is not independence')
    name='03_graph_characterization_role_crosswalk'
    save_figure(fig,folder,name,crosswalk); figure_sources[name]=crosswalk
    metadata=evaluate_characterization_panel(matrix,safe,roles,representation_id='R0|R2',tier='transversal',
        selected_sources=all_sources,minimum_cell=minimum_cell)
    metadata.update(status='DISPLAYED_DESCRIPTIVE' if len(crosswalk) else 'SOURCE_UNAVAILABLE',
        display_status='DISPLAYED_DESCRIPTIVE' if len(crosswalk) else 'SOURCE_UNAVAILABLE',
        reason='Source-role metadata only; no patient measurements.',n_displayable_sources=crosswalk.source_column.nunique(),
        n_communities_displayed=0,n_source_partitions_withheld=0)
    inventory.append(dict(figure=name,**metadata))
    # The historical availability panel includes the actual visible universe.
    # N2 additions enter only for representations whose selection was verified.
    available=safe.loc[safe.source_column.isin([m['source_column'] for m in panel])].copy()
    for rep in ['R0','R2']:
        selected=selection.loc[selection.representation_id.eq(rep) & selection.selection_status.eq('SELECTED_DESCRIPTIVE'),'source_column']
        extra=safe.loc[safe.representation_id.eq(rep) & safe.source_column.isin(selected)
            & ~safe.source_column.isin([m['source_column'] for m in panel])]
        available=pd.concat([available,extra],ignore_index=True)
    availability=available.drop_duplicates(['representation_id','community_id','source_column']).copy()
    columns=['representation_id','community_id','source_column','tier','n_group','n_observed','n_missing','n_tested','n_unknown','n_not_tested','status']
    availability=availability.reindex(columns=columns)
    availability['observed_fraction']=availability.n_observed/availability.n_group
    fig,ax=plt.subplots(figsize=(12,8),constrained_layout=True)
    visible_reps=sorted(set(availability.representation_id))
    r0_r2_visible=bool(set(visible_reps) & {'R0','R2'})
    if not availability.empty:
        availability['group_label']=availability.representation_id+'/C'+availability.community_id.astype(str)
        grid=availability.pivot(index='source_column',columns='group_label',values='observed_fraction')
        art=ax.imshow(grid,aspect='auto',vmin=0,vmax=1)
        ax.set(xticks=range(len(grid.columns)),xticklabels=grid.columns,yticks=range(len(grid)),yticklabels=[labels.get(s,s) for s in grid.index])
        fig.colorbar(art,ax=ax,label='Observed / hard supported; unknown and not tested are not negative')
    else:
        _empty(ax,WITHHELD_MESSAGE if any(i['status']=='WITHHELD_DISCLOSURE_CONTROL' for i in inventory) else 'SOURCE_UNAVAILABLE / no displayable source availability')
    limitation='NO_R0_R2_DISPLAYABLE_PANELS' if not r0_r2_visible else 'Excluded from this view: '+(', '.join(sorted({'R0','R2'}-set(visible_reps))) or 'none')
    ax.set_title('Source availability — visible representations: '+(', '.join(visible_reps) or 'none')+'\n'+limitation)
    name='03_graph_characterization_availability'
    save_figure(fig,folder,name,availability); figure_sources[name]=availability
    reps='|'.join(sorted(set(matrix.representation_id) | {'R0','R2'}))
    metadata=evaluate_characterization_panel(matrix,safe,roles,representation_id=reps,tier='transversal',selected_sources=all_sources,minimum_cell=minimum_cell)
    if not availability.empty:
        metadata.update(status='DISPLAYED_DESCRIPTIVE',display_status='DISPLAYED_DESCRIPTIVE',
            n_displayable_sources=availability.source_column.nunique(),
            n_communities_displayed=len(availability[['representation_id','community_id']].drop_duplicates()))
    metadata['reason']=limitation+'; '+metadata['reason']
    metadata['representations_displayed']='|'.join(visible_reps)
    inventory.append(dict(figure=name,**metadata))
    path=folder/'03_graph_figure_status.csv'
    old=pd.read_csv(path) if path.exists() else pd.DataFrame(columns=['figure','status','reason'])
    old=old.loc[~old.figure.isin([i['figure'] for i in inventory])]
    pd.concat([old,pd.DataFrame(inventory)],ignore_index=True).to_csv(path,index=False)
    return dict(sources=figure_sources,selection=selection,inventory=pd.DataFrame(inventory))

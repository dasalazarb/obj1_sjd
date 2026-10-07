import streamlit as st
from shared.components.figure_viewer import figure_viewer
from shared.components.evidence import missing_output, source
from shared.components.kpi import kpi_card
from shared.components.interpretation import interpretation
from shared.utils.paths import APP_ROOT
import yaml

VIEW_LABELS={'longitudinal':'Longitudinal','categorical_longitudinal':'Categorical longitudinal'}


def variable_explorer(run):
    catalog=run.registry.catalog
    figures=run.registry.figures
    verified=[row for row in figures if row['status']=='verified']
    available={row['variable_id'] for row in verified}
    options=[row for row in catalog if row['variable_id'] in available]
    view_ids=list(dict.fromkeys(['longitudinal','categorical_longitudinal']+[row['view'] for row in figures]))
    view_labels={view:VIEW_LABELS.get(view,next((row['title'] for row in figures if row['view']==view),'Other view')) for view in view_ids}
    coverage=[]
    for row in catalog:
        item={'Variable':row['display_name'],'Category':row['domain']}
        for view in view_ids:
            matched=[figure for figure in figures if figure['variable_id']==row['variable_id'] and figure['view']==view]
            registered=next((figure for figure in matched if figure['status']=='verified'),None)
            exists=registered and run.fingerprints.get(registered['figure_id']) is not None
            item[view_labels[view]]='available' if exists else next((figure['status'] for figure in matched if figure['status']=='candidate'),'missing')
        coverage.append(item)
    with st.expander('Figure coverage'):
        st.dataframe(coverage,hide_index=True,use_container_width=True)
    if not options:
        missing_output('No human-verified figures are registered. Laboratory filenames cannot be inferred from the deck.','objectives/objective_01/metadata/figures.csv')
        return
    deep_var=st.query_params.get('var')
    url=(deep_var,st.query_params.get('view'))
    incoming=url!=st.session_state.get('sjd_explorer_url')
    deep_entry=next((row for row in options if row['variable_id']==deep_var),None)
    if deep_var and not deep_entry:
        missing_output('The linked variable has no verified figure for this run.')
        if st.button('Clear unavailable selection'):
            st.query_params.pop('var',None);st.query_params.pop('view',None);st.rerun()
        return
    if incoming and deep_entry:
        st.session_state['sjd_search']=''
        st.session_state['sjd_category']=deep_entry['domain']
        st.session_state['sjd_variable_'+deep_entry['domain']]=deep_var
        st.session_state['sjd_view_'+deep_var]=url[1] or deep_entry['default_view']
    search=st.text_input('Search variables',key='sjd_search')
    # A local search filters the catalogue without turning the previous URL into an error.
    options=[row for row in options if search.casefold() in row['display_name'].casefold()]
    if not options:
        st.info('No matching variables.'); return
    domains=list(dict.fromkeys(row['domain'] for row in options))
    if st.session_state.get('sjd_category') not in domains:
        st.session_state['sjd_category']=domains[0]
    controls=st.columns(3)
    with controls[0]:
        domain=st.selectbox('Category',domains,key='sjd_category')
    variables=[row for row in options if row['domain']==domain]
    ids=[row['variable_id'] for row in variables]
    names={row['variable_id']:row['display_name'] for row in variables}
    with controls[1]:
        variable_key='sjd_variable_'+domain
        if st.session_state.get(variable_key) not in ids:
            st.session_state[variable_key]=ids[0]
        var=st.selectbox('Variable',ids,format_func=names.get,key=variable_key)
    selected=next(row for row in variables if row['variable_id']==var)
    views=list(dict.fromkeys(row['view'] for row in verified if row['variable_id']==var))
    default=selected['default_view']
    if incoming and deep_var==var and url[1] and url[1] not in views:
        missing_output('No figure for the linked view; no substitute was selected.')
        if st.button('Use the available default view'):
            st.query_params.pop('view',None);st.rerun()
        return
    with controls[2]:
        view_key='sjd_view_'+var
        if st.session_state.get(view_key) not in views:
            st.session_state[view_key]=default if default in views else views[0]
        view=st.selectbox('View',views,key=view_key,format_func=lambda v:view_labels[v])
    prev,next_=st.columns(2)
    def navigate(offset):
        key='sjd_variable_'+domain
        st.session_state[key]=ids[(ids.index(st.session_state[key])+offset)%len(ids)]
        st.query_params['var']=st.session_state[key]
        st.query_params.pop('view',None)
    prev.button('← Previous variable',disabled=len(ids)<2,on_click=navigate,args=(-1,),key='previous_variable')
    next_.button('Next variable →',disabled=len(ids)<2,on_click=navigate,args=(1,),key='next_variable')
    st.query_params.update(obj='objective_01',var=var,view=view)
    st.session_state['sjd_explorer_url']=(var,view)
    figure=next(row for row in verified if row['variable_id']==var and row['view']==view)
    figure_viewer(figure['figure_id'],run,figure['title'],figure)
    for other_view in [selected['default_view']]:
        if other_view not in views:
            st.button(VIEW_LABELS.get(other_view,other_view)+' — No figure for this view',disabled=True)
    summary,availability,interpret,origin=st.tabs(['Summary','Data availability','Interpretation','Source'])
    table_var=selected['table1_variable']
    with summary:
        if not table_var:
            st.info('Not available in baseline tables')
        else:
            try:
                table=run.data('t11_overall')
                matched=table.loc[table.Variable.eq(table_var)]
                if len(matched)!=1: raise ValueError('No unique baseline summary row')
                key=matched.iloc[0]['Section']+'|'+table_var
                kpi_card({'label':selected['display_name'],'value_ref':'t11_overall:'+key+'@Summary#raw','kind':'raw','decimals':None},run)
                source(['t11_overall:'+key+'@Summary#raw'],run)
            except ValueError as exc:
                missing_output('Not available in baseline tables',str(exc))
    with availability:
        if not table_var:
            st.info('Not available in baseline tables')
        else:
            refs=['t11_avail:'+table_var+'@n_available_for_analysis#raw','t11_avail:'+table_var+'@pct_available_for_analysis#raw']
            kpi_card({'label':'Available for analysis','value_ref':refs[0],'kind':'raw','decimals':0},run)
            kpi_card({'label':'Published availability (%)','value_ref':refs[1],'kind':'raw','decimals':1},run)
            source(refs,run)
    with interpret:
        claims=yaml.safe_load((APP_ROOT/'objectives/objective_01/metadata/claims.yml').read_text()) or []
        for claim in claims:
            if claim.get('variable_id')==var: interpretation(claim,run)
    with origin:
        st.text('File: '+figure['path']+'\nGenerated by: '+figure['source_script']+'\nRun: '+run.key+'\nDate: '+str(run.provenance.get('run_date','Not supplied'))+'\nCommit: '+str(run.provenance.get('git_commit','Not supplied')))
        if figure['panel_groups']: st.caption(figure['panel_groups'])

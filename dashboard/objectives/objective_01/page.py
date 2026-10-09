import importlib
import streamlit as st
import yaml
from shared.utils.paths import APP_ROOT
from shared.loaders.registry import Registry, OutputError
from shared.loaders.catalog import attach_figures
from shared.loaders.manifest_loader import capture_run, current_key
from shared.components.layout import page_header, section_header, run_badge
from shared.components.evidence import missing_output


SECTION_LABELS = {
    'study_design': 'Study design', 'cohort': 'Cohort', 'baseline': 'Baseline',
    'phenotype': 'Phenotype', 'glandular_serology': 'Glandular & serology',
    'organ_involvement': 'Organ involvement', 'followup': 'Follow-up',
    'variable_explorer': 'Variable explorer',
}


def section_navigation(sections):
    ids = [section['id'] for section in sections]
    linked = st.query_params.get('section')
    incoming = linked != st.session_state.get('sjd_section_url')
    if incoming and linked in ids:
        st.session_state['sjd_section'] = linked
    elif 'sjd_section' not in st.session_state:
        st.session_state['sjd_section'] = 'variable_explorer' if st.query_params.get('var') else 'baseline'
    if st.session_state['sjd_section'] not in ids:
        st.session_state['sjd_section'] = ids[0]

    def change(offset=0):
        if offset:
            index = ids.index(st.session_state['sjd_section'])
            st.session_state['sjd_section'] = ids[index + offset]
        st.query_params['section'] = st.session_state['sjd_section']
        st.session_state['sjd_section_url'] = st.session_state['sjd_section']

    index = ids.index(st.session_state['sjd_section'])
    left, center, right = st.columns([1, 4, 1])
    left.button('← Previous', disabled=index == 0, key='read_previous', on_click=change, args=(-1,))
    center.selectbox('Section', ids, key='sjd_section', on_change=change,
        format_func=lambda value: str(ids.index(value) + 1) + ' · ' + SECTION_LABELS[value], label_visibility='collapsed')
    right.button('Next →', disabled=index == len(ids) - 1, key='read_next', on_click=change, args=(1,))
    change()
    section = sections[ids.index(st.session_state['sjd_section'])]
    st.caption('Slides ' + '–'.join(str(value) for value in dict.fromkeys([section['slides'][0], section['slides'][-1]])))
    return section


def render():
    cfg=yaml.safe_load((APP_ROOT/'objectives/objective_01/config.yml').read_text())
    present=st.query_params.get('mode')=='present'
    try:
        registry=attach_figures(Registry())
        if not present:
            header, action = st.columns([5, 1])
            with header: page_header(cfg['eyebrow'],cfg['title'],cfg['subtitle'],'cover')
            refresh = action.button('Refresh outputs', key='refresh_outputs')
        else: refresh = False
        if refresh or 'sjd_run' not in st.session_state:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state['sjd_run']=capture_run(registry)
            st.session_state['run_key']=st.session_state['sjd_run'].key
            st.session_state['sjd_diagnostics']=[]
        run=st.session_state['sjd_run']
        if current_key(registry)[0]!=run.key:
            st.warning('A newer run is available — Refresh')
        if present: run_badge(run)
        else:
            with st.sidebar: run_badge(run)
        sections=cfg['sections']
        if present:
            st.html('<style>[data-testid="stSidebar"]{display:none}.sjd-kpi-label,.sjd-kpi-detail,.sjd-caution,.sjd-bar-row,.sjd-table,.sjd-source,.sjd-run,.sjd-missing{font-size:20px}</style>')
            from shared.components.slide_nav import keyboard_navigation
            keyboard_navigation()
            index=st.session_state.get('sjd_slide',0)
            left,center,right=st.columns([1,3,1])
            if left.button('← Previous',disabled=index==0):
                st.session_state['sjd_slide']=index-1; st.rerun()
            center.caption(sections[index]['title'])
            if right.button('Next →',disabled=index==len(sections)-1):
                st.session_state['sjd_slide']=index+1; st.rerun()
            sections=[sections[index]]
        else:
            selected = section_navigation(sections)
            read_all = st.sidebar.checkbox('Read all sections in slide order', key='sjd_read_all')
            if not read_all:
                sections = [selected]
        for section in sections:
            section_header(section)
            module=importlib.import_module('objectives.objective_01.'+section['file'].replace('/','.').removesuffix('.py'))
            try:
                module.render(cfg,run)
            except (OutputError,OSError) as exc:
                missing_output(section['title'],str(exc))
        if not present:
            if any('Unrecognized source-cell format' in message for message in st.session_state.get('sjd_diagnostics',[])):
                st.warning('Some source cells have an unrecognized format. Raw text is shown; inspect Output diagnostics.')
            with st.expander('Output diagnostics'):
                messages=st.session_state.get('sjd_diagnostics',[])
                if messages:
                    st.text('\n'.join(messages))
                    st.download_button('Download diagnostics','\n'.join(messages),file_name='sjd_diagnostics.txt')
                else: st.caption('No output diagnostics.')
    except (OutputError,OSError) as exc:
        missing_output('Run snapshot unavailable',str(exc))

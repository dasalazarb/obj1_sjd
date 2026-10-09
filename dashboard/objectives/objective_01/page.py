import importlib
import streamlit as st
import yaml
from shared.utils.paths import APP_ROOT
from shared.loaders.registry import Registry, OutputError
from shared.loaders.catalog import attach_figures
from shared.loaders.manifest_loader import capture_run, current_key
from shared.components.layout import page_header, section_header, run_badge
from shared.components.evidence import missing_output


def render():
    cfg=yaml.safe_load((APP_ROOT/'objectives/objective_01/config.yml').read_text())
    present=st.query_params.get('mode')=='present'
    try:
        registry=attach_figures(Registry())
        refresh=st.button('Refresh outputs',key='refresh_outputs') if not present else False
        if refresh or 'sjd_run' not in st.session_state:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state['sjd_run']=capture_run(registry)
            st.session_state['run_key']=st.session_state['sjd_run'].key
            st.session_state['sjd_diagnostics']=[]
        run=st.session_state['sjd_run']
        if current_key(registry)[0]!=run.key:
            st.warning('A newer run is available — Refresh')
        if not present:
            page_header(cfg['eyebrow'],cfg['title'],cfg['subtitle'],'cover')
        run_badge(run)
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

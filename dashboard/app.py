from pathlib import Path
import streamlit as st
import yaml
import importlib
from shared.utils.paths import APP_ROOT

st.set_page_config(page_title='SjD Research Explorer',page_icon='🔗',layout='wide',initial_sidebar_state='collapsed' if st.query_params.get('mode')=='present' else 'auto')
st.html('<style>'+(APP_ROOT/'shared/styles/theme.css').read_text()+'</style>')


def planned(title):
    def render():
        st.title(title)
        st.info('Coming soon')
    return render


from objectives.objective_01.page import render as objective_one
nav=yaml.safe_load((APP_ROOT/'config/navigation.yml').read_text())
pages={}
for group in nav['groups']:
    def page_callable(item):
        return importlib.import_module(item['path'].replace('/','.')+'.page').render if item['status']=='live' else planned(item['title'])
    pages[group['title']]=[st.Page(page_callable(item),title=item['title'],url_path=item['id'],default=item['id']=='objective_01') for item in group['items']]
navigation = st.navigation(pages, position='hidden')
if st.query_params.get('mode') != 'present':
    with st.sidebar:
        st.markdown('**SjD Research Explorer**')
        for group, items in zip(nav['groups'], pages.values()):
            for item, page in zip(group['items'], items):
                if item['status'] == 'live':
                    st.page_link(page)
        with st.expander('Other objectives and studies'):
            for group, items in zip(nav['groups'], pages.values()):
                st.caption(group['title'])
                for item, page in zip(group['items'], items):
                    if item['status'] != 'live': st.page_link(page)
navigation.run()

import streamlit as st
from shared.components.layout import timeline_diagram, episode_diagram, methodology
from shared.components.figure_viewer import figure_viewer


def render(cfg,run):
    if st.query_params.get('mode') == 'present':
        timeline_diagram(cfg['timeline_nodes'])
        episode_diagram(cfg['episode_nodes'])
        methodology('A research activity (e.g. genomics, CCGO) between visits is not a clinical visit: it could simulate Pop → Unclassifiable → Pop.')
        figure_viewer('fig_swimmer',run,'Visit timeline per patient')
        return
    figure_viewer('fig_swimmer',run,'Visit timeline per patient')
    with st.expander('Baseline and clinical episode definitions'):
        timeline_diagram(cfg['timeline_nodes'])
        episode_diagram(cfg['episode_nodes'])
        methodology('A research activity (e.g. genomics, CCGO) between visits is not a clinical visit: it could simulate Pop → Unclassifiable → Pop.')

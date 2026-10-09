import streamlit as st
from shared.components.kpi import kpi_grid
from shared.components.charts import bar_list, availability_rows


def render(cfg,run):
    kpi_grid(cfg['demographic_cards'],run)
    demographics, availability = st.tabs(['Demographics', 'Data availability'])
    with demographics:
        race, ethnicity = st.columns(2)
        with race: bar_list(cfg['race_rows'],run,title='Race')
        with ethnicity: bar_list(cfg['ethnicity_rows'],run,title='Ethnicity')
        st.caption('Hover for published values. Click a bar to keep its details below the chart. Use the toolbar to zoom, pan or save an image.')
    with availability:
        st.caption('Navy bar ≥ 90%, blue 60–90%, amber < 60% — visual convention for published availability.')
        bar_list(availability_rows(cfg['availability_rows'],run),run,cfg['bar_thresholds'])

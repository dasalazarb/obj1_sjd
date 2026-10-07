import streamlit as st
from shared.components.kpi import kpi_grid
from shared.components.charts import bar_list, availability_rows


def render(cfg,run):
    kpi_grid(cfg['demographic_cards'],run)
    bar_list(cfg['race_rows'],run)
    bar_list(cfg['ethnicity_rows'],run)
    st.subheader('What data are available at baseline')
    st.caption('Navy bar ≥ 90%, blue 60–90%, amber < 60% — visual convention for published availability.')
    bar_list(availability_rows(cfg['availability_rows'],run),run,cfg['bar_thresholds'])

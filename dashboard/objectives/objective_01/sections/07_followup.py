import streamlit as st
from shared.components.kpi import kpi_grid
from shared.components.charts import bar_list,column_chart
from shared.components.layout import caution


def bind(items,cohort):
    return [{key:([v.replace('{cohort}',cohort) for v in value] if key=='detail_refs' else value.replace('{cohort}',cohort) if isinstance(value,str) else value) for key,value in item.items()} for item in items]


def render(cfg,run):
    cohorts=['Overall']
    try:
        cohorts=list(dict.fromkeys(run.data('t12_long')['Cohort'].tolist()))
    except ValueError: pass
    if not cohorts: return
    cohort=st.radio('Follow-up cohort',cohorts,horizontal=True) if len(cohorts)>1 else cohorts[0]
    kpi_grid(bind(cfg['followup_cards'],cohort),run)
    kpi_grid(bind(cfg['followup_details'],cohort),run)
    bar_list(bind(cfg['episode_bars'],cohort),run)
    caution('Patients with a single clinical episode: {{ref:t12_long:Patients with exactly 1 clinical episode|'+cohort+'@Value#k}} ({{ref:t12_long:Patients with exactly 1 clinical episode|'+cohort+'@Value#pct}}%).',run)
    st.subheader('Retention from the baseline episode')
    column_chart(cfg['retention_times'],cohort,run)
    caution('Descriptive curve, not a Kaplan–Meier estimator.')
    st.subheader('Patients with at least one inter-visit gap')
    bar_list(bind(cfg['gap_bars'],cohort),run)

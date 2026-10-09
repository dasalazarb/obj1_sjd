import streamlit as st
import yaml
from shared.components.kpi import kpi_grid
from shared.components.stat_table import stat_table
from shared.components.charts import population_comparison
from shared.components.layout import caution
from shared.components.evidence import source
from shared.components.interpretation import interpretation
from shared.utils.paths import APP_ROOT


def render(cfg,run):
    caution('Serology is available in {{ref:t11_overall:Serology|Anti-Ro/SSA positive, n/N (%)@N available#raw}} of {{ref:t11_overall:Cohort / demographics|N patients@Summary#raw}} patients ({{ref:t11_avail:anti_ro_ssa__ever_positive_through_episode@pct_available_for_analysis#raw}}%). Sicca symptoms: {{ref:t11_overall:Glandular / extended phenotype|Any sicca symptom present, n/N (%)@Summary#k}}/{{ref:t11_overall:Glandular / extended phenotype|Any sicca symptom present, n/N (%)@Summary#N}} with documented data ({{ref:t11_overall:Glandular / extended phenotype|Any sicca symptom present, n/N (%)@Summary#pct}}%).',run)
    st.subheader('Glandular, serology and laboratory')
    population_comparison(cfg['pop_columns'], cfg['lab_rows'], run, 'glandular')
    if st.query_params.get('mode') == 'present':
        kpi_grid(cfg['glandular_cards'],run)
    else:
        with st.expander('Overall glandular and serology summaries'):
            kpi_grid(cfg['glandular_cards'],run)
    with st.expander('Complete glandular, serology and laboratory summary table'):
        stat_table(cfg['pop_columns'],cfg['lab_rows'],run)
    st.caption('ESR = erythrocyte sedimentation rate. IgG rounded to integer (from the already rounded published value).')
    caution('Serology denominators are specific to each population; descriptive comparisons, unadjusted.')
    for claim in yaml.safe_load((APP_ROOT/'objectives/objective_01/metadata/claims.yml').read_text()):
        if claim['section']=='glandular_serology': interpretation(claim,run)

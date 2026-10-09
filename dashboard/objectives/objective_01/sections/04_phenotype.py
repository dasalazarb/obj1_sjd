import streamlit as st
import yaml
from shared.components.kpi import kpi_grid
from shared.components.stat_table import stat_table,sf36_rows
from shared.components.charts import population_comparison
from shared.components.layout import caution,methodology
from shared.components.interpretation import interpretation
from shared.utils.paths import APP_ROOT


def render(cfg,run):
    st.subheader('Activity and patient-reported outcomes')
    population_comparison(cfg['pop_columns'], cfg['pro_rows'], run, 'phenotype')
    if st.query_params.get('mode') == 'present':
        kpi_grid(cfg['phenotype_cards'],run)
    else:
        with st.expander('Overall cohort summaries'):
            kpi_grid(cfg['phenotype_cards'],run)
    with st.expander('Complete activity and PRO summary table'):
        stat_table(cfg['pop_columns'],cfg['pro_rows']+sf36_rows(cfg['pop_columns'],run),run)
    caution('ESSDAI and ESSPRI define the groups, so differences in them are expected by construction. Comparisons are descriptive and unadjusted.')
    with st.expander('Availability in the unclassifiable population'):
        stat_table(['Unclassifiable'],[{'label':row['label']+' — n available','row_key':row['row_key'],'field':'n'} for row in cfg['pro_rows'] if row['row_key'].startswith('PROs|')],run)
    for claim in yaml.safe_load((APP_ROOT/'objectives/objective_01/metadata/claims.yml').read_text()):
        if claim['section']=='phenotype': interpretation(claim,run)
    methodology('Source: upstream Pop classification contract. Pop1 = ESSDAI ≥ 5; Pop2 = ESSDAI < 5 and ESSPRI ≥ 5; Pop3 = ESSDAI < 5 and ESSPRI < 5; Unclassifiable = insufficient data to assign a group.','Population definitions')

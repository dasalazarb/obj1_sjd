from shared.components.charts import stacked_bar_rows
from shared.components.stat_table import stat_table
from shared.components.layout import methodology


def render(cfg,run):
    stacked_bar_rows(cfg['pop_columns'],cfg['overlap_segments'],run)
    stat_table(cfg['pop_columns'],[{'label':'Active extraglandular domains','row_key':'Organ involvement|n_extraglandular_domains_active','field':'summary','decimals':1}],run)
    methodology('Overlap = active glandular manifestation plus at least one active extraglandular domain, at baseline. The upstream derive_overlap_flags contract also supports extraglandular-only and neither. Only categories published in the selected table are shown. Empty source cells remain empty; percentages are never normalized.')

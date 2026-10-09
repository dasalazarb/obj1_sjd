from shared.components.kpi import kpi_row, metadata_card
from shared.components.layout import step_flow


def render(cfg,run):
    kpi_row(cfg['cohort_cards'],run)
    metadata_card(run)
    step_flow(cfg['steps'],run)

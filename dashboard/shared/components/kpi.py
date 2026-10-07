from html import escape
import streamlit as st
from shared.loaders.refs import resolve, metadata_counts
from shared.components.evidence import value_html, missing_output, source


def kpi_card(card, run):
    refs = [card["value_ref"]] + card.get("detail_refs", [])
    try:
        value = resolve(refs[0], run, card.get("decimals", 1))
        details = [resolve(ref, run, 0 if ref.endswith(("#n", "#k", "#N")) else card.get("decimals", 1)) for ref in refs[1:]]
        detail_html = [value_html(value, ref) for value, ref in zip(details, refs[1:])]
        detail = ""
        if card.get("kind") == "median" and details and value.number is not None:
            detail = '(' + detail_html[0] + '–' + detail_html[1] + ')' + (' · n = ' + detail_html[2] if len(details) > 2 else '')
        elif card.get("kind") == "fraction" and value.number is not None:
            detail = '/'.join(detail_html)
        suffix = '%' if card.get("kind") == "fraction" and value.number is not None else ''
        st.html('<div class="sjd-kpi sjd-' + escape(card.get("tone", "light")) + '"><div class="sjd-kpi-label">' + escape(card["label"]) + '</div><div class="sjd-kpi-value">' + value_html(value, refs[0]) + suffix + '</div><div class="sjd-kpi-detail">' + detail + '</div></div>')
        if value.kind == "failure":
            st.warning(value.warning)
    except (ValueError, KeyError) as exc:
        missing_output(card["label"], str(exc))


def kpi_grid(cards, run, cols=3):
    for start in range(0, len(cards), cols):
        columns = st.columns(cols)
        for column, card in zip(columns, cards[start:start + cols]):
            with column:
                kpi_card(card, run)
    source([ref for card in cards for ref in [card["value_ref"]] + card.get("detail_refs", [])], run)


def kpi_row(cards, run):
    kpi_grid(cards, run, len(cards))


def metadata_card(run):
    try:
        included, total = metadata_counts(run)
        st.html(f'<div class="sjd-kpi sjd-light"><div class="sjd-kpi-label">Canonical variables published in Table 1</div><div class="sjd-kpi-value" data-ref="t11_dict:metadata_count" title="Count of dictionary metadata rows">{included} / {total}</div></div>')
    except ValueError as exc:
        missing_output("Variable dictionary metadata count", str(exc))
    source(["t11_dict:metadata_count"], run)

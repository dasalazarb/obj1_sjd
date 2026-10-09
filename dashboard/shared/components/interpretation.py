import streamlit as st
from shared.utils.provenance import claim_is_reviewed
from shared.components.evidence import source, template_html, missing_output


def interpretation(claim, run):
    try:
        reviewed=claim_is_reviewed(claim,run)
    except ValueError:
        reviewed=False
    def text():
        try:
            st.html('<p class="sjd-editorial">'+template_html(claim['text'],run)+'</p>')
        except ValueError as exc:
            missing_output('Editorial claim evidence unavailable',str(exc))
    if reviewed:
        text()
    else:
        st.info('Interpretation pending review')
        with st.expander('Editorial claim awaiting human review'):
            text()
    source(claim['evidence_refs'],run)

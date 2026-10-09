from html import escape
import re
import streamlit as st
from shared.components.evidence import missing_output, diagnostic, template_html


def page_header(eyebrow, title, intro="", variant="section"):
    tag = "h1" if variant == "cover" else "h2"
    st.html(f'<header class="sjd-header sjd-{escape(variant)}"><p class="sjd-eyebrow">{escape(eyebrow)}</p><{tag}>{escape(title)}</{tag}><p class="sjd-intro">{escape(intro)}</p></header>')


def section_header(section):
    st.html('<div id="' + escape(section["id"]) + '" class="sjd-anchor"></div>')
    page_header(section["eyebrow"], section["title"])


def caution(text, run=None):
    refs=re.findall(r'\{\{ref:(.*?)\}\}',text)
    try:
        html = template_html(text, run) if run else escape(text)
        st.html('<aside class="sjd-caution"><strong>Caution:</strong> ' + html + '</aside>')
    except ValueError as exc:
        missing_output("Caution evidence unavailable: " + str(exc))
    if refs and run:
        from shared.components.evidence import source
        source(refs,run)


def methodology(text, title="Methods details"):
    with st.expander(title):
        st.markdown(text)


def step_flow(steps, run=None):
    cells=[]
    for step in steps:
        try:
            cells.append('<div class="sjd-step">' + (template_html(step,run) if run else escape(step)) + '</div>')
        except ValueError as exc:
            missing_output('Study flow source unavailable',str(exc))
    st.html('<div class="sjd-step-flow">' + '<span class="sjd-arrow" aria-hidden="true">→</span>'.join(cells) + '</div>')
    refs=[ref for step in steps for ref in re.findall(r'\{\{ref:(.*?)\}\}',step)]
    if refs and run:
        from shared.components.evidence import source
        source(refs,run)


def timeline_diagram(nodes):
    st.html('<div role="img" aria-label="Conceptual baseline and follow-up timeline" class="sjd-timeline">' + '<span class="sjd-arrow">→</span>'.join('<div class="sjd-timeline-node">' + escape(node) + '</div>' for node in nodes) + '</div>')
    st.caption("Conceptual diagram — no data source")


def episode_diagram(nodes):
    st.html('<h3 class="sjd-subtitle">A date is not always a visit</h3><p>Components of one evaluation were recorded on different days; they are grouped into one clinical episode.</p><div class="sjd-chips">' + ''.join('<span class="sjd-chip">' + escape(n) + '</span>' for n in nodes) + '</div><div class="sjd-pair"><div class="sjd-wrong">Wrong: every date is a visit</div><div class="sjd-correct">Correct: one clinical episode</div></div>')
    st.caption("Conceptual diagram — no data source")


def run_badge(run):
    p = run.provenance
    if st.query_params.get('mode') == 'present':
        st.html('<div class="sjd-run">Run <strong>' + escape(run.run_id or run.key) + '</strong> · ' + escape(str(p.get("run_date", "date not supplied"))) + '</div>')
    else:
        with st.expander('Current output snapshot'):
            st.text('Run: ' + (run.run_id or run.key) + '\nDate: ' + str(p.get('run_date', 'date not supplied')) + '\nCommit: ' + str(p.get('git_commit', 'commit not supplied')) + '\nProducer: ' + str(p.get('script', 'script not supplied')))
            if not run.manifest:
                st.caption("Synthesized run: file fingerprints identify this snapshot, but cannot prove that upstream scripts ran together. Step 12 has no provenance file.")
    try:
        from shared.loaders.refs import resolve
        value = resolve("t11_overall:Cohort / demographics|N patients@Summary#raw", run)
        if p and str(p["n_patients"]) != value.raw:
            caution("Step 11 provenance patient count differs from Table 1.")
            diagnostic("Provenance patient count mismatch")
    except ValueError:
        pass

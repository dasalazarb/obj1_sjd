from html import escape
from pathlib import Path
import re
import streamlit as st
from shared.loaders.refs import resolve


def diagnostic(message):
    messages = st.session_state.setdefault("sjd_diagnostics", [])
    if message not in messages:
        messages.append(message)


def missing_output(what, expected_path=""):
    diagnostic(what + (" · " + expected_path if expected_path else ""))
    st.html('<div class="sjd-missing"><strong>Not available for this run</strong><br>' + escape(what) + ('<div class="sjd-path">' + escape(str(expected_path)) + '</div>' if expected_path else '') + '</div>')


def value_html(value, ref):
    src = value.source
    title = f"{src.file} · {src.row_key} · {src.column} · line {src.line_hint} · {src.run_key}"
    if value.warning:
        title += " · " + value.warning
        diagnostic(value.warning + " · " + ref)
    return '<span data-ref="' + escape(ref, quote=True) + '" title="' + escape(title, quote=True) + '">' + escape(value.display) + '</span>'


def template_html(text, run):
    """Escape editorial text and retain the provenance of each referenced cell."""
    pieces=[]
    offset=0
    for match in re.finditer(r'\{\{ref:(.*?)\}\}', text):
        pieces.append(escape(text[offset:match.start()]))
        ref=match.group(1)
        field=ref.rsplit('#', 1)[1]
        decimals=1 if field in {'pct','median','q1','q3','range_lo','range_hi'} or '@pct_' in ref else 0 if field in {'n','k','N'} else None
        pieces.append(value_html(resolve(ref,run,decimals),ref))
        offset=match.end()
    pieces.append(escape(text[offset:]))
    return ''.join(pieces)


def _evidence_footer(refs, run):
    ids = list(dict.fromkeys(ref.split(":", 1)[0] for ref in refs))
    entries = [run.registry.get(rid) for rid in ids]
    files = list(dict.fromkeys(Path(entry.path).name for entry in entries))
    producers = list(dict.fromkeys(Path(entry.producer).name for entry in entries))
    p = run.provenance
    run_label = run.run_id or str(p.get("run_date", "date not supplied"))
    label = 'Expected source' if any(run.fingerprints.get(entry.id) is None for entry in entries) else 'Source'
    text = f"{label}: {' · '.join(files)} · Producer: {' · '.join(producers)} · Run: {run_label} · {p.get('git_commit', 'commit not supplied')} · {run.key}"
    st.html('<div class="sjd-source">' + escape(text) + '</div>')


def evidence_footer(refs, run):
    if st.query_params.get('mode') == 'present':
        _evidence_footer(refs, run)
    else:
        with st.expander('Source and run details'):
            _evidence_footer(refs, run)


def source(refs, run):
    evidence_footer(refs, run)
    if st.query_params.get("inspect") == "1":
        with st.expander("Inspect source cells"):
            for ref in refs:
                try:
                    value = resolve(ref, run)
                    st.text(f"{ref}\nRaw: {value.raw}\nFile: {value.source.file}\nRun: {value.source.run_key}")
                except ValueError as exc:
                    st.text(str(exc))

from html import escape
import streamlit as st
from shared.loaders.refs import resolve
from shared.loaders.parsers import parse_cell, ParseFailure
from shared.components.evidence import missing_output, value_html, source


def stat_table(columns, rows, run):
    all_refs=[f't11_by_pop:{row["row_key"]}@{group}#raw' for row in rows for group in columns]
    try:
        run.data('t11_by_pop')
    except ValueError as exc:
        missing_output('Population summary table',str(exc))
        source(all_refs,run)
        return
    cells=[]; refs=[]
    for row in rows:
        html=[]
        for group in columns:
            ref=f't11_by_pop:{row["row_key"]}@{group}#raw'; refs.append(ref)
            try:
                value=resolve(ref,run)
                parsed=parse_cell(value.raw)
                if isinstance(parsed,ParseFailure) or parsed.kind in {'empty','na'} or row['field']=='raw':
                    html.append(value_html(value,ref))
                elif row['field']=='n':
                    nref=ref.rsplit('#',1)[0]+'#n'; refs.append(nref)
                    html.append(value_html(resolve(nref,run,0),nref))
                elif parsed.kind=='median_n':
                    component=[]
                    for field in ['median','q1','q3']:
                        fref=ref.rsplit('#',1)[0]+'#'+field; refs.append(fref)
                        component.append(value_html(resolve(fref,run,row.get('decimals',1)),fref))
                    html.append(component[0]+' ('+component[1]+'–'+component[2]+')')
                else:
                    html.append(value_html(value,ref))
            except ValueError as exc:
                html.append('<span title="'+escape(str(exc),quote=True)+'">Not available</span>')
                from shared.components.evidence import diagnostic
                diagnostic(str(exc))
        cells.append('<tr><th scope="row">'+escape(row['label'])+'</th>'+''.join('<td>'+cell+'</td>' for cell in html)+'</tr>')
    st.html('<div class="sjd-table-wrap"><table class="sjd-table"><thead><tr><th>Measure</th>'+''.join('<th>'+escape(group)+'</th>' for group in columns)+'</tr></thead><tbody>'+''.join(cells)+'</tbody></table></div>')
    source(refs,run)


def sf36_rows(columns, run):
    try:
        pcs=[resolve(f't11_by_pop:PROs|sf36_pcs@{group}#n',run).number for group in columns]
        mcs=[resolve(f't11_by_pop:PROs|sf36_mcs@{group}#n',run).number for group in columns]
        if pcs==mcs:
            return [{'label':'n with SF-36','row_key':'PROs|sf36_pcs','field':'n'}]
        st.caption('PCS and MCS availability differs; counts are shown separately.')
        return [{'label':'n with SF-36 PCS','row_key':'PROs|sf36_pcs','field':'n'},{'label':'n with SF-36 MCS','row_key':'PROs|sf36_mcs','field':'n'}]
    except ValueError as exc:
        missing_output('SF-36 availability by population',str(exc)); return []

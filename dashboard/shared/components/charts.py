from html import escape
import streamlit as st
from shared.loaders.refs import resolve
from shared.loaders.registry import MissingOutput
from shared.components.evidence import value_html, missing_output, source


def bar_list(rows, run, thresholds=None):
    refs = []
    for row in rows:
        local = [row['pct_ref']] + ([row['n_ref']] if row.get('n_ref') else [])
        refs.extend(local)
        try:
            pct = resolve(local[0], run, 1)
            count = resolve(local[1], run, 0) if len(local) > 1 else None
            if pct.kind=='failure':
                st.html('<p>'+escape(row['label'])+': '+value_html(pct,local[0])+'</p>')
                st.warning(pct.warning)
                continue
            if pct.number is None or not 0 <= pct.number <= 100:
                raise MissingOutput("Published percentage is unavailable or out of display range")
            color = '#9DB4D6' if row.get('tone') == 'unknown' else '#4A6FA5' if row.get('tone') == 'steel' else '#1B2A4A'
            if thresholds:
                color = '#1B2A4A' if pct.number >= thresholds[0] else '#4A6FA5' if pct.number >= thresholds[1] else '#B8873B'
            tooltip = ''
            if row.get('total_ref'):
                total = resolve(row['total_ref'], run, 0); nonmissing = resolve(row['nonmissing_ref'], run, 0)
                refs.extend([row['total_ref'], row['nonmissing_ref']])
                tooltip = f'Baseline total: {total.display}; nonmissing: {nonmissing.display}'
            st.html('<div class="sjd-bar-row" title="' + escape(tooltip) + '"><span class="sjd-bar-label">' + escape(row['label']) + '</span><div class="sjd-track"><div class="sjd-bar" style="width:' + str(pct.number) + '%;background:' + color + '"></div></div><span class="sjd-bar-value">' + value_html(pct,local[0]) + '%' + (' (' + value_html(count,local[1]) + ')' if count else '') + '</span></div>')
        except ValueError as exc:
            missing_output(row['label'], str(exc))
    source(refs, run)


def availability_rows(groups, run):
    result = []
    for group in groups:
        members = group['members']
        try:
            counts = [resolve(member['n_ref'], run).raw for member in members]
            percentages = [resolve(member['pct_ref'], run).raw for member in members]
            if all(n == counts[0] for n in counts) and all(p == percentages[0] for p in percentages):
                # Equal published counts permit the editorial grouping; no new statistic.
                result.append({**members[0], 'label': group['label']})
            else:
                result.extend(members)
                st.caption(group['label'] + ': member availability differs; shown separately.')
        except ValueError:
            result.extend(members)
    return result


def stacked_bar_rows(columns, segments, run):
    try:
        frame = run.data('t11_by_pop')
        present = set(frame.loc[frame.Section.eq('Organ involvement'), 'Variable'])
    except ValueError as exc:
        missing_output("Overlap by population", str(exc)); return
    segments = [segment for segment in segments if segment['row_key'].split('|')[1] in present]
    refs=[]
    for group in columns:
        count_ref=f't11_by_pop:Cohort / demographics|N patients@{group}#raw'
        refs.append(count_ref)
        try:
            count=resolve(count_ref,run,0)
            bars=[]; legend=[]
            for segment in segments:
                ref=f't11_by_pop:{segment["row_key"]}@{group}#pct'; refs.append(ref)
                value=resolve(ref,run,1)
                if value.kind=='failure':
                    st.html('<p>'+escape(group+' — '+segment['label'])+': '+value_html(value,ref)+'</p>')
                    st.warning(value.warning)
                    raise MissingOutput('Overlap source-cell format is unrecognized; raw text shown above')
                if value.kind=='empty':
                    legend.append(escape(segment['label'])+' — (empty in source)'); continue
                if value.number is None or not 0<=value.number<=100:
                    raise MissingOutput('Overlap percentage unavailable')
                text_color='#FAFBFC' if segment['color'] in {'#1B2A4A','#4A6FA5'} else '#1B2A4A'
                bars.append('<div class="sjd-segment" style="width:'+str(value.number)+'%;background:'+segment['color']+';color:'+text_color+'" title="'+escape(segment['label'])+'">'+value_html(value,ref)+'%</div>')
                fraction_ref=ref.rsplit('#',1)[0]+'#raw'; refs.append(fraction_ref)
                legend.append(escape(segment['label'])+' '+value_html(resolve(fraction_ref,run),fraction_ref))
            st.html('<div class="sjd-overlap-label">'+escape(group)+' (n = '+value_html(count,count_ref)+')</div><div class="sjd-stacked">'+''.join(bars)+'</div><div class="sjd-legend">'+' · '.join(legend)+'</div>')
        except ValueError as exc:
            missing_output(group+' overlap',str(exc))
    source(refs,run)


def column_chart(times, cohort, run):
    refs=[]; bars=[]; denominators=[]
    try:
        for time in times:
            prefix=f't12_ret:{cohort}|{time}@'
            local=[prefix+c+'#raw' for c in ['pct_retained','n_retained','denominator']]; refs.extend(local)
            pct=resolve(local[0],run,1); n=resolve(local[1],run,0); denominator=resolve(local[2],run,0)
            if pct.kind=='failure':
                st.html('<p>'+escape(time)+': '+value_html(pct,local[0])+'</p>')
                st.warning(pct.warning)
                raise MissingOutput('Retention source-cell format is unrecognized; raw text shown above')
            denominators.append(denominator.raw)
            if pct.number is None or not 0<=pct.number<=100:
                raise MissingOutput('Retention percentage unavailable')
            color='#1B2A4A' if time in {'6 months','1 year'} else '#9DB4D6' if time=='10 years' else '#4A6FA5'
            bars.append('<div class="sjd-column"><div>'+value_html(pct,local[0])+'%</div><div class="sjd-column-space"><div class="sjd-column-bar" style="height:'+str(pct.number)+'%;background:'+color+'"></div></div><strong>'+escape(time)+'</strong><div>'+value_html(n,local[1])+'/'+value_html(denominator,local[2])+'</div></div>')
        if any(value!=denominators[0] for value in denominators):
            st.warning('Published retention denominators differ across time windows.')
        else:
            st.caption('Published denominator: '+denominators[0])
        st.html('<div class="sjd-columns" aria-label="Descriptive retention; percentage scale from zero to one hundred">'+''.join(bars)+'</div>')
    except ValueError as exc:
        missing_output('Retention from the baseline episode',str(exc))
    source(refs,run)

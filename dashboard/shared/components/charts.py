"""Interactive views of published cells; no scientific aggregation or estimation."""
from hashlib import sha256
from html import escape
import plotly.graph_objects as go
import streamlit as st
from config.theme import POP_COLORS
from shared.loaders.refs import resolve
from shared.loaders.registry import MissingOutput
from shared.components.evidence import missing_output, source


def chart_key(prefix, refs):
    return prefix + '_' + sha256('\n'.join(refs).encode()).hexdigest()[:12]


def interactive_chart(figure, run, key, height=380, selectable=False):
    figure.update_layout(template='plotly_white', height=height,
        margin=dict(l=16, r=24, t=24, b=48), paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Arial, sans-serif', size=14, color='#1B2A4A'),
        hoverlabel=dict(bgcolor='white', font_size=14), legend=dict(orientation='h', y=-0.22),
        uirevision=figure.layout.uirevision or run.key)
    options = dict(key=key, use_container_width=True, theme=None,
        config={'scrollZoom': True, 'displaylogo': False, 'toImageButtonOptions': {'filename': key}})
    if selectable:
        options.update(on_select='rerun', selection_mode='points')
        figure.update_layout(clickmode='event+select')
    event = st.plotly_chart(figure, **options)
    if selectable and isinstance(event, dict):
        for point in event.get('selection', {}).get('points', []):
            details = point.get('customdata')
            if details: st.caption('Selected: ' + ' · '.join(str(item) for item in details[:3] if item))


def numeric(value, label):
    if value.kind == 'failure':
        st.html('<p>' + escape(label) + ': ' + escape(value.raw) + '</p>')
        st.warning(value.warning)
    if value.number is None:
        raise MissingOutput('Published numeric value unavailable: ' + label)
    # Decimal remains authoritative; conversion only supplies Plotly coordinates.
    return float(value.number)


def percentage(value, label):
    coordinate = numeric(value, label)
    if not 0 <= value.number <= 100:
        raise MissingOutput('Published percentage is outside the display range: ' + label)
    return coordinate


def bar_list(rows, run, thresholds=None, title=None):
    refs, labels, values, text, colors, details = [], [], [], [], [], []
    for row in rows:
        local = [row['pct_ref']] + ([row['n_ref']] if row.get('n_ref') else [])
        refs.extend(local)
        try:
            pct = resolve(local[0], run, 1)
            coordinate = percentage(pct, row['label'])
            count = resolve(local[1], run, 0) if len(local) > 1 else None
            color = '#9DB4D6' if row.get('tone') == 'unknown' else '#4A6FA5' if row.get('tone') == 'steel' else '#1B2A4A'
            if thresholds:
                color = '#1B2A4A' if pct.number >= thresholds[0] else '#4A6FA5' if pct.number >= thresholds[1] else '#B8873B'
            availability = ''
            if row.get('total_ref'):
                total = resolve(row['total_ref'], run, 0)
                nonmissing = resolve(row['nonmissing_ref'], run, 0)
                refs.extend([row['total_ref'], row['nonmissing_ref']])
                availability = f'Baseline total: {total.display}; nonmissing: {nonmissing.display}'
            labels.append(row['label']); values.append(coordinate)
            text.append(pct.display + '%'); colors.append(color)
            details.append([row['label'], pct.display + '%', 'n = ' + count.display if count else '', availability, pct.source.file])
        except ValueError as exc: missing_output(row['label'], str(exc))
    if labels:
        figure = go.Figure(go.Bar(x=values, y=labels, orientation='h', marker_color=colors,
            text=text, textposition='auto', customdata=details,
            hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}<br>%{customdata[3]}<br>Source: %{customdata[4]}<extra></extra>'))
        figure.update_xaxes(range=[0, 100], ticksuffix='%', title='Published percentage')
        figure.update_yaxes(autorange='reversed')
        figure.update_layout(meta={'refs': refs, 'run': run.key}, showlegend=False)
        if title: st.subheader(title)
        interactive_chart(figure, run, chart_key('bars', refs), max(300, len(labels) * 38 + 90), selectable=True)
    source(refs, run)


def availability_rows(groups, run):
    result = []
    for group in groups:
        members = group['members']
        try:
            counts = [resolve(member['n_ref'], run).raw for member in members]
            percentages = [resolve(member['pct_ref'], run).raw for member in members]
            if all(n == counts[0] for n in counts) and all(p == percentages[0] for p in percentages):
                result.append({**members[0], 'label': group['label']})
            else:
                result.extend(members)
                st.caption(group['label'] + ': member availability differs; shown separately.')
        except ValueError: result.extend(members)
    return result


def stacked_bar_rows(columns, segments, run):
    try:
        frame = run.data('t11_by_pop')
        present = set(frame.loc[frame.Section.eq('Organ involvement'), 'Variable'])
    except ValueError as exc:
        missing_output('Overlap by population', str(exc)); return
    segments = [segment for segment in segments if segment['row_key'].split('|')[1] in present]
    refs, figure = [], go.Figure()
    for segment in segments:
        labels, values, details = [], [], []
        for group in columns:
            ref = f't11_by_pop:{segment["row_key"]}@{group}#pct'
            raw_ref = ref.rsplit('#', 1)[0] + '#raw'
            count_ref = f't11_by_pop:Cohort / demographics|N patients@{group}#raw'
            refs.extend([ref, raw_ref, count_ref])
            try:
                value = resolve(ref, run, 1)
                if value.kind in {'empty', 'na'}:
                    st.caption(group + ' · ' + segment['label'] + ': ' + value.display + ' (source cell unavailable)'); continue
                coordinate = percentage(value, group + ' · ' + segment['label'])
                raw = resolve(raw_ref, run); count = resolve(count_ref, run, 0)
                labels.append(group); values.append(coordinate)
                details.append([group + ' · ' + segment['label'], raw.display, 'Group n = ' + count.display, value.source.file])
            except ValueError as exc: missing_output(group + ' overlap', str(exc))
        if labels:
            figure.add_trace(go.Bar(y=labels, x=values, orientation='h', name=segment['label'],
                marker_color=segment['color'], customdata=details,
                hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}<br>Source: %{customdata[3]}<extra></extra>'))
    if figure.data:
        figure.update_layout(barmode='stack', meta={'refs': refs, 'run': run.key})
        figure.update_xaxes(range=[0, 100], ticksuffix='%', title='Published percentage')
        figure.update_yaxes(categoryorder='array', categoryarray=columns, autorange='reversed')
        interactive_chart(figure, run, chart_key('overlap', refs), height=420, selectable=True)
        st.caption('Click a legend label to show or hide that category. Published percentages are unchanged.')
    source(refs, run)


def column_chart(times, cohort, run):
    refs, labels, values, text, colors, details, denominators = [], [], [], [], [], [], []
    for time in times:
        prefix = f't12_ret:{cohort}|{time}@'
        local = [prefix + column + '#raw' for column in ['pct_retained', 'n_retained', 'denominator']]
        refs.extend(local)
        try:
            pct = resolve(local[0], run, 1); n = resolve(local[1], run, 0); denominator = resolve(local[2], run, 0)
            coordinate = percentage(pct, time)
            denominators.append(denominator.raw)
            labels.append(time); values.append(coordinate); text.append(pct.display + '%')
            colors.append('#1B2A4A' if time in {'6 months', '1 year'} else '#9DB4D6' if time == '10 years' else '#4A6FA5')
            details.append([time, pct.display + '%', n.display + '/' + denominator.display, pct.source.file])
        except ValueError as exc: missing_output('Retention · ' + time, str(exc))
    if labels:
        if any(value != denominators[0] for value in denominators):
            st.warning('Published retention denominators differ across time windows.')
        else: st.caption('Published denominator: ' + denominators[0])
        figure = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=text, textposition='auto', customdata=details,
            hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Retained: %{customdata[2]}<br>Source: %{customdata[3]}<extra></extra>'))
        figure.update_yaxes(range=[0, 100], ticksuffix='%', title='Published retention')
        figure.update_xaxes(type='category', categoryorder='array', categoryarray=times)
        figure.update_layout(meta={'refs': refs, 'run': run.key}, showlegend=False)
        interactive_chart(figure, run, chart_key('retention', refs), selectable=True)
    source(refs, run)


def population_comparison(columns, rows, run, key):
    """Select one published measure; quartile endpoints are plotted directly."""
    choices = [row for row in rows if row['field'] in {'summary', 'fraction'}]
    if not choices: return
    measures = {row['row_key']: row for row in choices}
    controls = st.columns([2, 3])
    with controls[0]:
        measure = st.selectbox('Measure', list(measures), format_func=lambda value: measures[value]['label'], key=key + '_measure')
    with controls[1]:
        selected = st.multiselect('Populations', columns, default=columns, key=key + '_groups')
    if not selected:
        st.info('Select at least one population.'); return
    row = measures[measure]
    refs, figure = [], go.Figure()
    fraction = row['field'] == 'fraction'
    for group in selected:
        prefix = f't11_by_pop:{row["row_key"]}@{group}#'
        raw_ref = prefix + 'raw'; refs.append(raw_ref)
        try:
            raw = resolve(raw_ref, run)
            if raw.kind in {'empty', 'na'}:
                st.caption(group + ': ' + raw.display + ' (source cell unavailable)'); continue
            fields = ['pct'] if fraction else ['q1', 'q3', 'median']
            field_refs = [prefix + field for field in fields]; refs.extend(field_refs)
            published = [resolve(ref, run, row.get('decimals', 1)) for ref in field_refs]
            coordinates = [percentage(value, group) if fraction else numeric(value, group) for value in published]
            color = POP_COLORS.get(group, '#1B2A4A')
            hover = [group, raw.display, raw.source.file]
            hovertemplate = '<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Source: %{customdata[2]}<extra></extra>'
            if fraction:
                figure.add_trace(go.Bar(x=[group], y=coordinates, name=group, marker_color=color, customdata=[hover], hovertemplate=hovertemplate))
            else:
                figure.add_trace(go.Scatter(x=[group, group], y=coordinates[:2], mode='lines+markers',
                    line=dict(color=color, width=3), marker=dict(symbol='line-ew', size=16, color=color),
                    name=group, legendgroup=group, showlegend=False, customdata=[hover, hover], hovertemplate=hovertemplate))
                figure.add_trace(go.Scatter(x=[group], y=[coordinates[2]], mode='markers',
                    marker=dict(size=13, color=color), name=group, legendgroup=group, customdata=[hover], hovertemplate=hovertemplate))
        except ValueError as exc: missing_output(row['label'] + ' · ' + group, str(exc))
    if figure.data:
        figure.update_layout(meta={'refs': refs, 'run': run.key}, showlegend=False,
            uirevision=chart_key(run.key + '_' + key, refs))
        figure.update_xaxes(type='category', categoryorder='array', categoryarray=selected)
        figure.update_yaxes(title=row['label'] + (' (%)' if fraction else ''))
        if fraction: figure.update_yaxes(range=[0, 100], ticksuffix='%')
        interactive_chart(figure, run, key + '_chart', selectable=True)
        if not fraction: st.caption('Dot = published median. Line endpoints = published Q1 and Q3 (IQR), not confidence intervals.')
    from shared.components.stat_table import stat_table
    with st.expander('Selected published values'):
        stat_table(selected, [row], run)

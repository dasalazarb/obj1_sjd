"""MOCK-only checks for chart coordinates, source fidelity, and user controls."""
import json
from streamlit.testing.v1 import AppTest
from shared.loaders.refs import resolve
from shared.utils.paths import APP_ROOT


def app_for(mock_run, monkeypatch, section):
    monkeypatch.setenv('SJD_OUTPUTS_ROOT', str(mock_run.registry.outputs))
    monkeypatch.setenv('SJD_REPO_ROOT', str(mock_run.registry.repo))
    app = AppTest.from_file(str(APP_ROOT / 'app.py'))
    app.session_state['sjd_run'] = mock_run
    app.query_params['section'] = section
    return app.run(timeout=20)


def figures(app):
    return [json.loads(element.proto.spec) for element in app.get('plotly_chart')]


def test_navigation_preserves_slide_order_and_all_sections(mock_run, monkeypatch):
    app = app_for(mock_run, monkeypatch, 'baseline')
    assert not app.exception
    assert app.selectbox(key='sjd_section').value == 'baseline'
    app.button(key='read_next').click().run(timeout=20)
    assert not app.exception
    assert app.query_params['section'] == 'phenotype'
    app.button(key='read_previous').click().run(timeout=20)
    assert app.query_params['section'] == 'baseline'
    app.checkbox(key='sjd_read_all').check().run(timeout=20)
    assert not app.exception
    assert len([item for item in app.get('html') if 'class="sjd-anchor"' in item.proto.body]) == 8


def test_external_section_link_replaces_existing_selection(mock_run, monkeypatch):
    app = app_for(mock_run, monkeypatch, 'baseline')
    app.query_params['section'] = 'followup'
    app.run(timeout=20)
    assert not app.exception
    assert app.selectbox(key='sjd_section').value == 'followup'


def test_percentages_are_source_coordinates_with_display_and_run(mock_run, monkeypatch):
    app = app_for(mock_run, monkeypatch, 'baseline')
    figure = figures(app)[0]
    trace = figure['data'][0]
    ref = figure['layout']['meta']['refs'][0]
    value = resolve(ref, mock_run, 1)
    assert trace['x'][0] == float(value.number)
    assert trace['customdata'][0][1] == value.display + '%'
    assert trace['customdata'][0][-1] == value.source.file
    assert figure['layout']['meta']['run'] == mock_run.key
    assert figure['layout']['xaxis']['range'] == [0, 100]


def test_population_filter_changes_chart_and_selected_source_values(mock_run, monkeypatch):
    app = app_for(mock_run, monkeypatch, 'phenotype')
    app.multiselect(key='phenotype_groups').set_value(['Pop2']).run(timeout=20)
    assert not app.exception
    figure = figures(app)[0]
    assert len(figure['data']) == 2
    assert all(trace['name'] == 'Pop2' for trace in figure['data'])
    assert all('@Pop2#' in ref for ref in figure['layout']['meta']['refs'])
    endpoint, median = figure['data']
    measure = app.selectbox(key='phenotype_measure').value
    assert endpoint['y'] == [float(resolve(f't11_by_pop:{measure}@Pop2#{field}', mock_run).number) for field in ['q1', 'q3']]
    assert median['y'] == [float(resolve(f't11_by_pop:{measure}@Pop2#median', mock_run).number)]
    selected_table = next(item.proto.body for item in app.get('html') if 'class="sjd-table"' in item.proto.body)
    assert '<th>Pop2</th>' in selected_table
    assert '<th>Pop1</th>' not in selected_table


def test_measure_switch_plots_published_fraction_and_keeps_empty_cell(mock_run, monkeypatch):
    frame = mock_run.tables['t11_by_pop']
    variable = 'Anti-Ro/SSA positive, n/N (%)'
    frame.loc[frame.Variable.eq(variable), 'Pop2'] = ''
    app = app_for(mock_run, monkeypatch, 'glandular_serology')
    initial_revision = figures(app)[0]['layout']['uirevision']
    app.selectbox(key='glandular_measure').set_value('Serology|' + variable).run(timeout=20)
    assert not app.exception
    figure = figures(app)[0]
    assert figure['layout']['uirevision'] != initial_revision
    assert all(trace['type'] == 'bar' for trace in figure['data'])
    assert 'Pop2' not in [trace['name'] for trace in figure['data']]
    assert any('Pop2: —' in caption.value for caption in app.caption)
    for trace in figure['data']:
        ref = f't11_by_pop:Serology|{variable}@{trace["name"]}#pct'
        assert trace['y'] == [float(resolve(ref, mock_run).number)]


def test_retention_uses_published_percentage_not_count_ratio(mock_run, monkeypatch):
    app = app_for(mock_run, monkeypatch, 'followup')
    figure = next(figure for figure in figures(app) if figure['layout']['yaxis'].get('title', {}).get('text') == 'Published retention')
    # MOCK counts and denominator are both 3; the published percentage is also 3,
    # deliberately inconsistent so recomputing 3/3 would fail this fidelity check.
    assert figure['data'][0]['y'] == [3.0] * len(figure['data'][0]['x'])
    assert figure['data'][0]['customdata'][0][2] == '3/3'


def test_empty_population_selection_has_no_invented_chart(mock_run, monkeypatch):
    app = app_for(mock_run, monkeypatch, 'phenotype')
    app.multiselect(key='phenotype_groups').set_value([]).run(timeout=20)
    assert not app.exception
    assert not figures(app)
    assert any('Select at least one population' in item.value for item in app.info)

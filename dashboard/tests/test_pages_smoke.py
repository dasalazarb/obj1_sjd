from streamlit.testing.v1 import AppTest
from shared.utils.paths import APP_ROOT


def test_missing_outputs_page_stays_functional(tmp_path,monkeypatch):
    monkeypatch.setenv('SJD_OUTPUTS_ROOT',str(tmp_path/'MOCK_empty_outputs'))
    monkeypatch.setenv('SJD_REPO_ROOT',str(tmp_path/'MOCK_empty_repo'))
    app=AppTest.from_file(str(APP_ROOT/'app.py')).run(timeout=20)
    assert not app.exception
    assert app.session_state['sjd_diagnostics']
    assert app.button(key='refresh_outputs')


def test_mock_published_cells_render(mock_run,monkeypatch):
    monkeypatch.setenv('SJD_OUTPUTS_ROOT',str(mock_run.registry.outputs))
    monkeypatch.setenv('SJD_REPO_ROOT',str(mock_run.registry.repo))
    app=AppTest.from_file(str(APP_ROOT/'app.py')).run(timeout=20)
    assert not app.exception
    assert app.session_state['run_key']


def test_presentation_navigation(tmp_path,monkeypatch):
    monkeypatch.setenv('SJD_OUTPUTS_ROOT',str(tmp_path/'MOCK_empty_outputs'))
    app=AppTest.from_file(str(APP_ROOT/'app.py'));app.query_params['mode']='present';app.run(timeout=20)
    assert not app.exception
    next_button=next(button for button in app.button if button.label=='Next →');next_button.click().run(timeout=20)
    assert app.session_state['sjd_slide']==1;assert not app.exception

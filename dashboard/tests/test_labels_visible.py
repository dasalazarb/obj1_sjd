import re
import yaml
from streamlit.testing.v1 import AppTest
from shared.utils.paths import APP_ROOT


def test_internal_ids_not_visible_as_scientific_labels(mock_run,monkeypatch):
    monkeypatch.setenv('SJD_OUTPUTS_ROOT',str(mock_run.registry.outputs));monkeypatch.setenv('SJD_REPO_ROOT',str(mock_run.registry.repo))
    app=AppTest.from_file(str(APP_ROOT/'app.py')).run(timeout=20)
    assert not app.exception
    labels=yaml.safe_load((APP_ROOT/'config/labels.yml').read_text())['labels']
    # Source tooltips and explicit inspect mode contain keys by design; visible labels do not.
    blocks=[element.proto.body for element in app.get('html') if 'sjd-missing' not in element.proto.body and 'sjd-source' not in element.proto.body]
    text=re.sub(r'<[^>]+>','',' '.join(blocks))
    for identifier in labels:
        assert not re.search(r'(?<![A-Za-z0-9_])'+re.escape(identifier)+r'(?![A-Za-z0-9_])',text)

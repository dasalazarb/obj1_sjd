"""Verify colocated repository defaults and explicit output-copy overrides."""
from pathlib import Path
from shared.utils.paths import APP_ROOT, outputs_root, repo_root


def test_defaults_use_containing_analysis_repo(monkeypatch):
    assert APP_ROOT == Path(__file__).resolve().parents[1]
    monkeypatch.delenv('SJD_REPO_ROOT', raising=False)
    monkeypatch.delenv('SJD_OUTPUTS_ROOT', raising=False)
    assert repo_root() == APP_ROOT.parent
    assert outputs_root() == APP_ROOT.parent / 'outputs'


def test_repo_override_also_selects_its_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv('SJD_REPO_ROOT', str(tmp_path / 'MOCK_repo'))
    monkeypatch.delenv('SJD_OUTPUTS_ROOT', raising=False)
    assert repo_root() == tmp_path / 'MOCK_repo'
    assert outputs_root() == tmp_path / 'MOCK_repo' / 'outputs'


def test_independent_output_override(tmp_path, monkeypatch):
    monkeypatch.setenv('SJD_REPO_ROOT', str(tmp_path / 'MOCK_repo'))
    monkeypatch.setenv('SJD_OUTPUTS_ROOT', str(tmp_path / 'MOCK_outputs'))
    assert repo_root() == tmp_path / 'MOCK_repo'
    assert outputs_root() == tmp_path / 'MOCK_outputs'

"""Launcher must preserve the selected environment and app-local configuration."""
import importlib.util
import sys
from shared.utils.paths import APP_ROOT


def test_launcher_from_another_directory(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('sjd_launcher', APP_ROOT / 'run_dashboard.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    def call(command, **kwargs):
        calls.append((command, kwargs))
        return 7
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.subprocess, 'call', call)
    monkeypatch.setattr(sys, 'argv', ['run_dashboard.py', '--server.port=8502'])
    assert module.main() == 7
    assert calls == [([
        sys.executable, '-m', 'streamlit', 'run', str(APP_ROOT / 'app.py'), '--server.port=8502'
    ], {'cwd': APP_ROOT})]

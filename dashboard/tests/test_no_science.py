import importlib.util
from shared.utils.paths import APP_ROOT
from scripts.check_no_science import violations


def test_presentation_only(): assert violations(APP_ROOT)==[]


def test_guard_detects_science(tmp_path):
    folder=tmp_path/'shared';folder.mkdir()
    (folder/'bad.py').write_text('import scipy\nx.groupby("a").mean()\nx.merge(y)\n')
    assert len(violations(tmp_path))==4


def test_string_formatting_is_allowed(tmp_path):
    folder=tmp_path/'shared';folder.mkdir();(folder/'ok.py').write_text("' · '.join(labels)\n")
    assert violations(tmp_path)==[]

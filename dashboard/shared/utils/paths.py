from pathlib import Path
import os

APP_ROOT = Path(__file__).resolve().parents[2]


def outputs_root():
    return Path(os.environ.get("SJD_OUTPUTS_ROOT", repo_root() / "outputs")).resolve()


def repo_root():
    return Path(os.environ.get("SJD_REPO_ROOT", APP_ROOT.parent)).resolve()


def contained(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Only contained relative paths are allowed")
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Path escapes its configured root")
    return target

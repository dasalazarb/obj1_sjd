"""Launch from any working directory with this Python environment's Streamlit."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    return subprocess.call(
        [sys.executable, '-m', 'streamlit', 'run', str(root / 'app.py'), *sys.argv[1:]],
        cwd=root,
    )


if __name__ == '__main__':
    raise SystemExit(main())

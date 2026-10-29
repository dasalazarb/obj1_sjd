#!/usr/bin/env python3
"""Graph02: run the full configured study without required arguments.

The historical Python helpers remain available for callers during migration.
Use --legacy for the historical S0-S4 command interface.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.studies.graph.stage02.s02_legacy import *  # noqa: F401,F403
from src.studies.graph.stage02 import s02_legacy

def __getattr__(name):
    return getattr(s02_legacy, name)

def main(argv=None):
    from src.studies.graph.stage02.s02_pipeline import main as full_main
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--legacy" in argv:
        argv.remove("--legacy")
        return s02_legacy.run(s02_legacy.parse_args(argv))
    return full_main(argv)

if __name__ == "__main__":
    main()

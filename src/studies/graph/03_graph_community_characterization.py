#!/usr/bin/env python3
"""Graph03: interpret a verified, finalized Graph02 release without refitting."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.studies.graph.stage03.s03_statistics import *  # historical statistical helpers
from src.studies.graph.stage03.s03_pipeline import main

if __name__=='__main__':
    main()

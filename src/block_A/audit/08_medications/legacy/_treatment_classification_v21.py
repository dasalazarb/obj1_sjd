"""Compatibility import of the maintained classification rules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from src.block_A._internal.medications import classification as _implementation
globals().update({name:value for name,value in vars(_implementation).items() if not name.startswith('__')})

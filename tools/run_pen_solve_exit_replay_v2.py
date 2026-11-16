"""Explicit v2 entry point; actual binding logic is captured from shared src."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from supermimic_research.runtime.pen_solve_exit_version2 import run_cli
if __name__=='__main__':run_cli('worker',Path(__file__))

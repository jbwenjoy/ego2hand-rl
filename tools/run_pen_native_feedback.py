"""Entry point for the predeclared fixed-budget feedback-support comparison."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from supermimic_research.runtime.pen_native_experiment import main

if __name__=='__main__':
    import torch
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    with torch.jit.optimized_execution(False):main(p.parse_args())

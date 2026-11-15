"""Collect independently executed candidate targets for fresh parent-split data."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from supermimic_research.runtime.candidate_response_data import main
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

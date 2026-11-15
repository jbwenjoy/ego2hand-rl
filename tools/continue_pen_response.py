import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.learning.pen_continuation import run_continuation

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--health', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    print(json.dumps(run_continuation(ROOT, a.checkpoint, a.health, a.output, run=a.run)))

"""Metadata default; Root --execute enables one saved-only self15 evaluation."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path('D:/workspace/supermimic-full151-observation-study');sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path);parser.add_argument('--execute', action='store_true');a = parser.parse_args()
    from full151_gradient import boundary_self15_cached_saved105_v1 as core
    if a.execute:
        if a.protocol is None or a.output is None:parser.error('--execute requires protocol and fresh output')
        core.run_saved_evaluation(a.protocol, a.output);return 0
    if a.protocol is None:
        print(json.dumps(core.declaration()));return 0
    p, out = core.metadata_preflight(a.protocol, a.output)
    print(json.dumps({'status': 'PASS_SAVED_SELF15_METADATA_ONLY', 'protocol': core.pin(a.protocol), 'actual_NPZ_weights_NN_metrics': 0}));return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Component confirmation: default metadata-only; Root explicit --run required."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path('D:/workspace/supermimic-full151-observation-study')
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,'D:/workspace/dex-supermimic-research/src')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--run',action='store_true');a=parser.parse_args(argv)
    from full151_gradient import component_confirmation105_v1 as evaluation
    if a.run:
        if a.protocol is None or a.output is None:parser.error('--run needs --protocol and fresh --output')
        r=evaluation.run_confirmation(a.protocol,a.output)
        print(json.dumps({k:r[k] for k in ('schema','status','counts','decision')},allow_nan=False));return 0
    if a.protocol is None:
        if a.output is not None:parser.error('--output needs --protocol')
        print(json.dumps(evaluation.declaration(),allow_nan=False));return 0
    m=evaluation.metadata_preflight(a.protocol,a.output)
    print(json.dumps({k:m[k] for k in ('status','protocol_pin','sealed_file_count','test_values_decoded','model_loads_forwards')},allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())

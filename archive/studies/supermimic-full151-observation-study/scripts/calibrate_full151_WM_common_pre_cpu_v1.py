"""DRAFT direct CLI; default only metadata, explicit Root --run for calibration."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path('D:/workspace/supermimic-full151-observation-study')
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, 'D:/workspace/dex-supermimic-research/src')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    from full151_gradient import response_calibration105_v1 as calibration
    if args.run:
        if args.protocol is None or args.output is None:
            parser.error('--run requires --protocol and a fresh --output')
        result = calibration.run_calibration(args.protocol, args.output,
            emit=lambda event: print(json.dumps(event, allow_nan=False), flush=True))
        print(json.dumps({k: result[k] for k in ('schema','status','completed_iterations_per_arm','counts','fresh_test_evaluated')}, allow_nan=False))
    elif args.protocol is None:
        if args.output is not None: parser.error('--output requires --protocol')
        print(json.dumps(calibration.declaration(), allow_nan=False))
    else:
        metadata = calibration.metadata_preflight(args.protocol, args.output)
        print(json.dumps({k: metadata[k] for k in ('status','protocol_pin','output','Torch_imported','NPZ_opened','strict_loads','WM_forwards_AD_optimizer_steps')}, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

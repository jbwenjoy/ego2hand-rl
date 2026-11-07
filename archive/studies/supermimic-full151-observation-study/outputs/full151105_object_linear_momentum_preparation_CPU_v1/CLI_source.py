"""Saved object momentum: metadata default, Root explicit --execute only."""
from pathlib import Path
import argparse,json,sys
ROOT=Path('D:/workspace/supermimic-full151-observation-study');sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--execute',action='store_true');a=ap.parse_args(argv)
    from full151_gradient import object_linear_momentum105_v1 as core
    if a.execute:
        if a.protocol is None or a.output is None:ap.error('--execute requires --protocol and fresh --output')
        r=core.run_analysis(a.protocol,a.output);print(json.dumps({k:r[k] for k in ('schema','status','counts','all1808_preserved')},allow_nan=False));return 0
    if a.protocol is None:
        if a.output is not None:ap.error('--output requires --protocol')
        print(json.dumps(core.declaration(),allow_nan=False));return 0
    m=core.metadata_preflight(a.protocol,a.output);print(json.dumps({k:m[k] for k in ('status','protocol_pin','actual_rows_read_MJB_arithmetic')},allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())

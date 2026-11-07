"""Three-component H8: metadata default, explicit Root --run for CPU inference."""
from pathlib import Path
import argparse,json,sys
ROOT=Path('D:/workspace/supermimic-full151-observation-study')
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,'D:/workspace/dex-supermimic-research/src')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args(argv)
    from full151_gradient import component_H8_donor105_v1 as h
    if a.run:
        if a.protocol is None or a.output is None:ap.error('--run requires --protocol and fresh --output')
        r=h.run_analysis(a.protocol,a.output);print(json.dumps({k:r[k] for k in ('schema','status','counts','accuracy_qualified')},allow_nan=False));return 0
    if a.protocol is None:
        if a.output is not None:ap.error('--output requires --protocol')
        print(json.dumps(h.declaration(),allow_nan=False));return 0
    m=h.metadata_preflight(a.protocol,a.output);print(json.dumps({k:m[k] for k in ('status','protocol_pin','actual_NPZ_decodes_window_construction_model_FW')},allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())

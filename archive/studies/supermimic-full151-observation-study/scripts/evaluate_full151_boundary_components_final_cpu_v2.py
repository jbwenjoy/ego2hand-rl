"""Default metadata; explicit Root --execute for fixed final component eval."""
import argparse,json,sys
from pathlib import Path
ROOT=Path('D:/workspace/supermimic-full151-observation-study');sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--execute',action='store_true');a=ap.parse_args(argv)
    from full151_gradient import boundary_component_final_evaluation105_v2 as core
    if a.execute:
        if a.protocol is None or a.output is None:ap.error('--execute requires protocol and fresh output')
        r=core.run_evaluation(a.protocol,a.output);print(json.dumps({k:r[k] for k in ('schema','status','counts')}));return 0
    if a.protocol is None:print(json.dumps(core.declaration()));return 0
    m=core.metadata_preflight(a.protocol,a.output);print(json.dumps({k:m[k] for k in ('status','protocol_pin','actual_tensor_loads_FW')}));return 0


if __name__=='__main__':raise SystemExit(main())

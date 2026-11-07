"""Public native-self15 matched component CPU training: metadata default; Root --execute only."""
import argparse,json,sys
from pathlib import Path
ROOT=Path('D:/workspace/supermimic-full151-observation-study');sys.path.insert(0,str(ROOT/'src'))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--execute',action='store_true');a=ap.parse_args(argv)
    from full151_gradient import boundary_public_self15_training105_v1 as core
    if a.execute:
        if a.protocol is None or a.output is None:ap.error('--execute requires protocol and fresh output')
        r=core.run_training(a.protocol,a.output);print(json.dumps({k:r[k] for k in ('schema','status','iteration','training_counts','process_counts')}));return 0
    if a.protocol is None:
        print(json.dumps(core.declaration()));return 0
    m=core.metadata_preflight(a.protocol,a.output);print(json.dumps({k:m[k] for k in ('status','protocol_pin','actual_data_load_model_FW_AD_optimizer')}));return 0


if __name__=='__main__':raise SystemExit(main())

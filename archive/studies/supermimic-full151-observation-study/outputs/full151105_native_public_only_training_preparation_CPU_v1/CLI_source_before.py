"""Native52+self15 only component training draft; metadata default, Root execute."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    from full151_gradient import native_public_only_training105_v1 as core
    if args.execute:
        if args.protocol is None or args.output is None:parser.error('--execute requires protocol and fresh output')
        report=core.run_training(args.protocol.resolve(strict=True),args.output);print(json.dumps({key:report[key] for key in ('schema','status','iteration','training_counts','process_counts')}));return 0
    if args.protocol is None:print(json.dumps(core.declaration()));return 0
    result=core.metadata_preflight(args.protocol.resolve(strict=True),args.output);print(json.dumps({key:result[key] for key in ('status','protocol_pin','actual_data_load_model_FW_AD_optimizer')}));return 0

if __name__=='__main__':raise SystemExit(main())

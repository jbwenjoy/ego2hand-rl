"""Default metadata only; Root supplies a separate authorized protocol for --run."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from full151_gradient import public_self15_endpoint_response105_v1 as c

def main():
    a=argparse.ArgumentParser();a.add_argument('--protocol');a.add_argument('--output');a.add_argument('--run',action='store_true');args=a.parse_args()
    if not args.protocol:
        if args.run:a.error('--run requires --protocol and --output')
        print(json.dumps(c.declaration(),ensure_ascii=False));return
    if args.run:
        if not args.output:a.error('--run requires --output')
        r=c.run_analysis(args.protocol,args.output)
        print(json.dumps({'status':r['status'],'report':str(Path(args.output).resolve()/'report.json'),'counts':r['counts'],'elapsed_seconds':r['elapsed_seconds']},ensure_ascii=False))
    else:
        m=c.metadata_preflight(args.protocol,args.output);print(json.dumps({'status':'PASS_JSON_HASH_METADATA_ONLY','protocol':m['protocol_pin'],'actual_array_reads':0,'model_FW_AD_physics':0}))

if __name__=='__main__':main()

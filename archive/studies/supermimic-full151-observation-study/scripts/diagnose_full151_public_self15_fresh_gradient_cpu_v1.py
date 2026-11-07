"""Default declaration/JSON metadata only; actual Root protocol is separate."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from full151_gradient import public_self15_fresh_gradient105_v1 as c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol');ap.add_argument('--output');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.protocol:
        if a.run:ap.error('--run requires --protocol/--output')
        print(json.dumps(c.declaration(),ensure_ascii=False));return
    if a.run:
        if not a.output:ap.error('--run requires --output')
        r=c.run_analysis(str(Path(a.protocol).resolve()),str(Path(a.output).resolve()));print(json.dumps({'status':r['status'],'counts':r['counts'],'report':str(Path(a.output).resolve()/'report.json'),'elapsed_seconds':r['elapsed_seconds']}))
    else:
        m=c.metadata_preflight(str(Path(a.protocol).resolve()),a.output);print(json.dumps({'status':'PASS_JSON_HASH_ONLY_NO_ARRAY_MODEL','protocol':m['protocol_pin'],'actual_NPZ_model_FW_AD_reads':0}))

if __name__=='__main__':main()

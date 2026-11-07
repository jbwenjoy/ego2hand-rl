"""Explicit saved-only --run; default is read-only metadata."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from full151_gradient import contact_regime608105_v1 as core

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',default=str(core.PREREG));p.add_argument('--output');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:
        if not a.output:p.error('--run requires --output FRESH')
        r=core.run_analysis(a.protocol,a.output)
        print(json.dumps({'status':r['status'],'report':str(Path(a.output)/'report.json'),'counts':r['counts'],'elapsed_seconds':r['elapsed_seconds']}))
    else:
        r=core.metadata_preflight(a.protocol,a.output);print(json.dumps({'status':r['status'],'protocol':r['protocol'],'NPZ_values_decoded':0,'contact_regime_runs':0}))

if __name__=='__main__':main()

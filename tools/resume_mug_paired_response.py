"""CLI for the common paired-response continuation engine, no copied update loop."""
from pathlib import Path
import argparse,json,os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning.paired_continuation import run_continuation

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true');args=p.parse_args()
    print(json.dumps(run_continuation(ROOT,args.config,args.output,run=args.run)))

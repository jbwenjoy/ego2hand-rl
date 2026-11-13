from pathlib import Path
import argparse,json,os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning.impulse_continuation import run_impulse_continuation
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    print(json.dumps(run_impulse_continuation(ROOT,a.config,a.output,run=a.run)))

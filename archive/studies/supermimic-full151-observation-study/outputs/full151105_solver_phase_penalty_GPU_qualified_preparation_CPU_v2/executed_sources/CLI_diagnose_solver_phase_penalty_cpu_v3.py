"""Default metadata only. Actual qualified-state scoring requires separate CPU plan."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from full151_gradient.penalty_observability105_v3 import declaration,metadata_preflight,run_analysis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:
        if not a.config:p.error('--run requires an explicitly authorized --config')
        result=run_analysis(a.config)
    elif a.config:
        r=metadata_preflight(a.config);result={k:r[k] for k in ('status','config_pin','Torch_imported','Warp_imported','NPZ_opened','state_conversions','objective_constructs')}
    else:result=declaration()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()

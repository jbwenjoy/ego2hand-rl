"""Default is hash/JSON metadata only; --run needs Root's separate CPU runplan."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from full151_gradient.trained_bonus_analysis105_v1 import declaration,metadata_preflight,run_analysis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.run:
        if not a.config:p.error('--run requires an exact authorized --config')
        result=run_analysis(a.config)
    elif a.config:
        d=metadata_preflight(a.config)
        result={k:d[k] for k in ('status','config_pin','Torch_imported','Warp_imported','NPZ_opened','model_loads','objective_constructions')}
    else:result=declaration()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()

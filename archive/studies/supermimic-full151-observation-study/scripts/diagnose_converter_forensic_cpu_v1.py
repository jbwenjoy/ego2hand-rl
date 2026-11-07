"""Default hash/JSON only. Authorized --run does converter arithmetic only."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient.converter_forensic105_v1 import declaration,metadata_preflight,run_forensic


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:
        if not a.config:p.error('Explicit separate forensic config required')
        value=run_forensic(a.config)
    elif a.config:
        r=metadata_preflight(a.config);value={k:r[k] for k in ('status','config_pin','Torch_imported','Warp_imported','NPZ_opened','scoring_calls')}
    else:value=declaration()
    print(json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__':main()

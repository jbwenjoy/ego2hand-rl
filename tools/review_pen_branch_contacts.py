import argparse
import sys
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning.pen_branch_profiles import branch_profiles
from supermimic_research.mechanics.warp_saved_review import review_saved_warp_contacts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=tomllib.loads(a.config.read_text());_,profiles=branch_profiles(cfg)
    review_saved_warp_contacts(a.report,a.output,{(p['name'],dt) for p in profiles for dt in cfg['dt']})

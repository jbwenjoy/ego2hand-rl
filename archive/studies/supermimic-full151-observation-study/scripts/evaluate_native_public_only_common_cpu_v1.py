"""Metadata default, Root-coordinated --run for one new native-only B152 FW."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import native_public_only_common_evaluation105_v1 as m


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path);p.add_argument('--output',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.protocol is None:
        print(json.dumps(m.declaration()));return
    if not a.run:
        result=m.metadata_preflight(a.protocol,a.output)
        print(json.dumps({'status':'PASS_METADATA_ONLY_NO_MODEL_ARRAY_VALUES_OR_FW','protocol':result['protocol_pin'],'budget':m.BUDGET}));return
    if a.output is None:raise ValueError('Fresh output required')
    m.run_evaluation(a.protocol,a.output)


if __name__=='__main__':main()

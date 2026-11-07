"""Metadata by default; Root --execute alone may decode windows and run H8."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path);p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    from full151_gradient import public_self15_H8_donor105_v1 as c
    if a.execute:
        if a.protocol is None or a.output is None:p.error('--execute requires protocol and fresh output')
        r=c.run_analysis(a.protocol,a.output);print(json.dumps({k:r[k] for k in ('schema','status','counts','elapsed_seconds')}));return
    if a.protocol is None:print(json.dumps(c.declaration()));return
    m=c.metadata_preflight(a.protocol,a.output);print(json.dumps({k:m[k] for k in ('status','protocol_pin','actual_windows_models_FW')}))


if __name__=='__main__':main()

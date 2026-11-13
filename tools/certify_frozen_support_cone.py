"""Checked separating witness for an explicitly saved frozen contact cone."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.mechanics.wrench_separation import separating_witness


def main(args):
    if args.output.exists():raise ValueError('Fresh certificate output')
    data=np.load(args.basis);answer=separating_witness(data['wrench_BODY'],data['target_BODY'],radius_m=.125)
    result=dict(schema='fixed_native_support_cone_certificate:1',basis=file_pin(args.basis),
        sources=[file_pin(__file__),file_pin(ROOT/'src/supermimic_research/mechanics/wrench_separation.py')],
        certificate=answer,neural_updates=0,native_integration_steps=0)
    args.output.mkdir();(args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--basis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

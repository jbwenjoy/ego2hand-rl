"""CPU readback: endpoint secants versus AD, with pooled normalized errors."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.learning.pen_native_response import vector_comparison
from supermimic_research.learning.pen_stage import write


def pooled(rows):
    if not rows:return dict(pairs=0)
    denominator=sum(r['target_L2']**2 for r in rows)
    return dict(pairs=len(rows),pooled_relative_L2_error=float(np.sqrt(sum((r['relative_L2_error']*r['target_L2'])**2 for r in rows)/max(denominator,1e-24))),
        median_pair_relative_L2_error=float(np.median([r['relative_L2_error'] for r in rows])),
        max_pair_relative_L2_error=max(r['relative_L2_error'] for r in rows),
        zero_response_comparator_relative_error=1.0)


def main(a):
    r=json.loads(a.evaluation.read_text());check=dict(np.load(verify_file_pin(r['dataset_pins']['CHECK'])))
    old=torch.load(verify_file_pin(r['old_checkpoint']),map_location='cpu',weights_only=False);scale=np.asarray(old['arms']['DIRECT']['scales']['y_scale'])
    if a.output.exists():raise ValueError('Fresh response audit')
    reports={}
    for mode,score in r['models'].items():
        prediction=np.load(verify_file_pin(r['outputs'][mode]))['CHECK_prediction'];rows=[]
        for pair in r['physical_pair_classifications']:
            if not pair['eligible']:continue
            ip,im=pair['plus_index'],pair['minus_index'];epsilon=pair['epsilon']
            native=(check['y'][ip]-check['y'][im])/(2*epsilon);secant=(prediction[ip]-prediction[im])/(2*epsilon)
            row={k:pair[k] for k in ['anchor','dt','direction','epsilon']}
            for name,sl in [('object12',slice(40,52)),('wrench6',slice(78,84))]:row[name]=vector_comparison(secant[sl]/scale[sl],native[sl]/scale[sl])
            rows.append(row)
        summary={}
        for anchor in ['ALL',0,2,4,6]:
            selected=[row for row in rows if anchor=='ALL' or row['anchor']==anchor]
            ad=[row for row in score['physical_same_source_CHECK_pairs'] if anchor=='ALL' or row['anchor']==anchor]
            summary[str(anchor)]={block:dict(endpoint_secant=pooled([row[block] for row in selected]),local_AD=pooled([row[block] for row in ad])) for block in ['object12','wrench6']}
        reports[mode]=dict(endpoint_secant_rows=rows,summary=summary)
    a.output.mkdir();result=dict(schema='pen_branch_response_AD_secant_audit:1',evaluation=file_pin(a.evaluation),source=file_pin(__file__),models=reports,
        optimizer_updates=0,native_integrations=0,
        scope='Same11 eligible CHECK pairs on fixed original TRAIN scales. Pooled error weights by measured response norm and is NOT independent-sample inference; per-pair errors remain. Endpoint secant prediction differs from infinitesimal AD. No new amplitude/repeat stability test or Actor readiness certification.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),summary={k:v['summary'] for k,v in reports.items()})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

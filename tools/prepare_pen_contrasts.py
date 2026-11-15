"""TRAIN-only finite-action contrasts; no new fitting or simulator calls."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.learning.pen_branch_evaluation import classify_pair
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    prep=json.loads(a.prepared.read_text());collection=json.loads(verify_file_pin(prep['inputs']['collection']).read_text())
    proposal=json.loads(verify_file_pin(collection['proposal']).read_text());selected=np.load(verify_file_pin(proposal['artifacts']['selected']))
    qref=np.load(verify_file_pin(proposal['clip']))['reference_qpos'];directions=np.load(verify_file_pin(collection['directions']))['TRAIN']
    review=json.loads(a.review.read_text())
    if (verify_file_pin(review['prepared'])!=a.prepared.resolve() or review['passed']!=review['total']
            or not all(prep['checks'].values()) or a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs')):
        raise ValueError('Reviewed source preparation and fresh unified contrast output')
    source=dict(np.load(verify_file_pin(prep['outputs']['NEW_TRAIN'])))
    packets=[];rows=[]
    declarations=[r for r in prep['rows'] if r['role']=='TRAIN']
    for plus in [r for r in declarations if r['sign']==1 and r['reached']]:
        minus=next(r for r in declarations if r['sign']==-1 and r['anchor']==plus['anchor'] and r['direction']==plus['direction'] and r['dt']==plus['dt'])
        if not minus['reached']:continue
        ip=np.flatnonzero(source['case_id']==plus['case_id']).item();im=np.flatnonzero(source['case_id']==minus['case_id']).item()
        classification=classify_pair(source['x'][ip],source['x'][im],source['valid'][ip] and source['valid'][im],
            qref[plus['anchor']+1]+selected['delta20'],directions[plus['direction']],plus['epsilon'])
        rows.append(dict(plus_case_id=plus['case_id'],minus_case_id=minus['case_id'],anchor=plus['anchor'],dt=plus['dt'],direction=plus['direction'],epsilon=plus['epsilon'],**classification))
        if classification['eligible']:
            packets.append(dict(x_plus=source['x'][ip],x_minus=source['x'][im],y_plus=source['y'][ip],y_minus=source['y'][im],
                finite_contrast=(source['y'][ip]-source['y'][im])/(2*plus['epsilon']),epsilon=plus['epsilon'],
                direction20=directions[plus['direction']],anchor=plus['anchor'],dt=plus['dt'],plus_case_id=plus['case_id'],minus_case_id=minus['case_id']))
    if not packets:raise ValueError('No valid common-source TRAIN contrast')
    packed={k:np.array([p[k] for p in packets]) for k in packets[0]}
    checks=dict(TRAIN_declarations_only=all(r['role']=='TRAIN' for r in declarations),
        exact_sources=np.array_equal(packed['x_plus'][:,:81],packed['x_minus'][:,:81]),
        exact_dt=np.array_equal(packed['x_plus'][:,-1],packed['x_minus'][:,-1]),
        finite_all=all(np.isfinite(v).all() for v in packed.values()),
        actual_unclipped_direction=bool(np.max(abs(packed['x_plus'][:,81:101]-packed['x_minus'][:,81:101]-2*packed['epsilon'][:,None]*packed['direction20']))<4e-7))
    a.output.mkdir();path=a.output/'TRAIN_contrasts.npz';np.savez_compressed(path,**packed)
    result=dict(schema='pen_TRAIN_finite_action_contrasts:1',prepared=file_pin(a.prepared),review=file_pin(a.review),
        sources=[file_pin(__file__),file_pin(ROOT/'src/supermimic_research/learning/pen_branch_evaluation.py')],
        input_role='NEW_TRAIN_ONLY',input=prep['outputs']['NEW_TRAIN'],output=file_pin(path),source_pairs=rows,
        eligible_pairs=len(packets),all_declared_TRAIN_pairs=len(rows),checks=checks,passed=sum(checks.values()),total=len(checks),
        original_scales=prep['scales'],head_scales=prep['head_scales'],normalizer_fits=0,neural_updates=0,native_integration_steps=0,
        scope='Finite amplitude contrasts from eligible TRAIN pairs only. No CHECK/SEQUENCE_CHECK/old physical diagnostic arrays are loaded. Same captured source features, no source repair; not infinitesimal derivatives or fresh physical qualification. No optimizer started.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),pairs=len(packets),all_pairs=len(rows),checks=checks)))
    if not all(checks.values()):raise ValueError('Keep failed contrasts; no optimizer')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

"""Locate the first failed physics tick of every originally failed selected forecast."""
import argparse
import json
import sys
import tomllib
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.pen_replay_fidelity import compare_response
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    r=json.loads(a.report.read_text());review=json.loads(a.review.read_text())
    if verify_file_pin(review['run'])!=a.report.resolve() or a.output.exists():raise ValueError('Exact reviewed report and fresh tick diagnostics')
    cp=verify_file_pin(r['config']);cfg=tomllib.loads(cp.read_text());gp=cp.parent/cfg['gates'];gates=tomllib.loads(gp.read_text())
    efc=json.loads(verify_file_pin(review['EFC']).read_text());failures=[]
    for case in review['cases']:
        failed=[d for d in case['decisions'] if not d['fidelity_passed']]
        if not failed:continue
        c=next(c for c in r['cases'] if c['name']==case['mode'] and c['dt']==case['dt'])
        z=np.load(verify_file_pin(c['trace']));m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        jids=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
        qids,dofs=m.jnt_qposadr[jids],m.jnt_dofadr[jids];qa,da=c['model_layout']['object_qposadr'],c['model_layout']['object_dofadr']
        e=next(e for e in efc['cases'] if e['name']==c['name'] and e['dt']==c['dt']);w=np.load(verify_file_pin(e['wrenches']))['world_COM_wrench'];n=round(.02/c['dt'])
        for d in failed:
            decision=next(p for p in c['planning'] if p['k']==d['k']);packet=np.load(verify_file_pin(decision['prediction']));index=d['selected'];ticks=[]
            for j in range(n):
                pred={key:packet[val][index,j:j+1].astype(float) for key,val in [('post_qpos','Q'),('post_qvel','V'),('post_ctrl','U'),('post_GPU_time','T')]}
                lo=d['k']*n+j;actual={key:z[key][lo:lo+1] for key in pred}
                metrics,checks=compare_response(pred,actual,packet['W'][index,j:j+1].astype(float),w[lo:lo+1],qids,dofs,qa,da,gates)
                ticks.append(dict(tick=j,elapsed_s=(j+1)*c['dt'],metrics=metrics,checks=checks,passed=all(checks.values())))
            failures.append(dict(mode=c['name'],dt=c['dt'],k=d['k'],source_s=d['k']*.02,prediction=decision['prediction'],source=decision['source'],
                first_failed_tick=next(t['tick'] for t in ticks if not t['passed']),ticks=ticks))
    a.output.mkdir();result=dict(schema='native_pen_failed_forecast_tick_diagnostics:1',run=file_pin(a.report),review=file_pin(a.review),gates=file_pin(gp),source=file_pin(__file__),
        failures=failures,all_original_failed_forecasts_included=len(failures)==review['selected_fidelity_total']-review['selected_fidelity_passed'],native_integrations=0,neural_updates=0,
        scope='Every originally failed selected forecast, all physics ticks at unchanged gates. Failure timing is descriptive, not proof of contact/solver cause; unavailable predicted solver/contact internals are not fabricated.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),failures=failures)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

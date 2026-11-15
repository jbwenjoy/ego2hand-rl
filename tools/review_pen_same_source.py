"""Independent source/force/fidelity readback of all120 diagnostic responses."""
import argparse
import contextlib
import json
import sys
import tomllib
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.warp_saved_review import review_saved_warp_contacts
from supermimic_research.mechanics.pen_replay_fidelity import compare_response
from supermimic_research.runtime.controlled_evaluation import write


def saved_efc(path,out,expected):
    try:
        with out.with_suffix('.log').open('w',encoding='utf8') as log,contextlib.redirect_stdout(log):
            return review_saved_warp_contacts(path,out,expected)
    except ValueError:
        if not (out/'report.json').exists():raise
        return json.loads((out/'report.json').read_text())


def main(a):
    r=json.loads(a.report.read_text());cfg=tomllib.loads(verify_file_pin(r['config']).read_text());gates=tomllib.loads(verify_file_pin(r['gates']).read_text())
    actual_path=verify_file_pin(r['actual']);response_path=verify_file_pin(r['responses'])
    actual=json.loads(actual_path.read_text());responses=json.loads(response_path.read_text())
    expected={(f'A{k}_{p["name"]}_R{rep}',dt) for k in [0,2,4,6] for p in r['profiles'] for rep in range(3) for dt in [.005,.0025]}
    if (len(responses['cases'])!=120 or {(c['name'],c['dt']) for c in responses['cases']}!=expected
            or len(r['source_captures'])!=8 or a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs')):
        raise ValueError('Exact fixed120 response cohort and fresh review')
    for pin in r['sources']+actual['task_sources']:verify_file_pin(pin)
    a.output.mkdir();ae=saved_efc(actual_path,a.output/'actual_EFC',{('ACTUAL_BASELINE',dt) for dt in [.005,.0025]})
    pe=saved_efc(response_path,a.output/'response_EFC',expected)
    actual_data={c['dt']:dict(np.load(verify_file_pin(c['trace']))) for c in actual['cases']}
    actual_wrench={c['dt']:np.load(verify_file_pin(c['wrenches']))['world_COM_wrench'] for c in ae['cases']}
    contexts={(c['dt'],c['anchor']):c for c in r['source_captures']}
    models={c['dt']:mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology']))) for c in actual['cases']}
    sources={key:dict(np.load(verify_file_pin(c['arrays']))) for key,c in contexts.items()}
    indices={}
    for dt,m in models.items():
        j=np.array([m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
        indices[dt]=(m.jnt_qposadr[j],m.jnt_dofadr[j])
    items=[];fidelity=[]
    for c in responses['cases']:
        dt,k=c['dt'],c['anchor'];context=contexts[(dt,k)];source=sources[(dt,k)]
        packet=dict(np.load(verify_file_pin(c['trace'])));m=models[dt];n=round(.02/dt);lo,hi=k*n,(k+1)*n
        z=actual_data[dt];controller=np.load(verify_file_pin(context['controller']))
        state=mujoco.MjData(m);mujoco.mj_setState(m,state,packet['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        e=next(e for e in pe['cases'] if e['name']==c['name'] and e['dt']==dt)
        w=np.load(verify_file_pin(e['wrenches']))['world_COM_wrench']
        body=c['model_layout']['object_body'];raw=packet['raw_contact_tick_plus23'];nonhand=0.
        for row in raw:
            g,h=map(int,row[1:3]);ob=m.geom_bodyid[[g,h]]==body
            if ob.sum()==1 and not (m.geom(h if ob[0] else g).name or '').startswith('robot/'):nonhand=max(nonhand,max(0.,row[18]))
        initial=mujoco.MjData(m);mujoco.mj_setState(m,initial,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        actual_warm=initial.qacc_warmstart if lo==0 else z['post_qacc'][lo-1]
        qids,dofs=indices[dt]
        aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==m.joint(f'robot/right_finger{i}_joint{j}').id)[0] for i in range(1,6) for j in range(1,5)])
        checks=dict(source_qv=bool(np.array_equal(state.qpos,source['qpos'][0]) and np.array_equal(state.qvel,source['qvel'][0])),
            source_warm_time=bool(np.array_equal(state.qacc_warmstart,source['qacc_warmstart'][0]) and state.time==source['time'][0]),
            source_matches_untouched_actual=bool(np.array_equal(source['qpos'][0],z['pre_qpos'][lo]) and np.array_equal(source['qvel'][0],z['pre_qvel'][lo]) and np.array_equal(source['qacc_warmstart'][0],actual_warm)),
            exact_interval=len(packet['post_qpos'])==n and int(controller['command_phase'][0])==k+1 and int(z['pre_action_phase'][k])==k+1,
            held_executed_PD=np.array_equal(packet['post_ctrl'][:,aids],np.tile(c['executed_PD20'],(n,1))),
            no_aid=not np.any(packet['post_qfrc_applied']) and not np.any(packet['post_xfrc_applied']) and nonhand<=1e-9,
            finite_active=bool(all(np.isfinite(packet[key]).all() for key in ['pre_qpos','pre_qvel','post_qpos','post_qvel','post_qacc','post_qacc_warmstart','post_ctrl','raw_contact_tick_plus23'])),
            fixed_mocap=np.array_equal(packet['post_mocap_pos'],np.repeat(source['mocap_pos'],n,axis=0)) and np.array_equal(packet['post_mocap_quat'],np.repeat(source['mocap_quat'],n,axis=0)),
            capacity=not np.any(packet['overflow']),native_EFC_checks=all(e['checks'].values()),
            diagnostic_only=c['role']=='DIAGNOSTIC_NEVER_FIT')
        items.append(dict(name=c['name'],dt=dt,anchor=k,checks={key:bool(v) for key,v in checks.items()},clamp_max_rad=c['clamp_max_rad']))
        if c['profile']=='BASE':
            qa,da=c['model_layout']['object_qposadr'],c['model_layout']['object_dofadr']
            actual_prefix={key:value[lo:hi] for key,value in z.items() if key in ['post_qpos','post_qvel','post_ctrl','post_GPU_time']}
            metrics,fc=compare_response(packet,actual_prefix,w,actual_wrench[dt][lo:hi],qids,dofs,qa,da,gates)
            fidelity.append(dict(name=c['name'],dt=dt,anchor=k,repeat=c['repeat'],metrics=metrics,checks=fc,passed=all(fc.values())))
    isolation=all(all(c['checks'].values()) for c in r['source_captures'])
    quality=all(all(c['checks'].values()) for c in items)
    actual_quality=ae['passed']==ae['total'];all_fidelity=all(c['passed'] for c in fidelity)
    result=dict(schema='pen_same_source_response_saved_review:1',run=file_pin(a.report),source=file_pin(__file__),
        comparison_source=file_pin(ROOT/'src/supermimic_research/mechanics/pen_replay_fidelity.py'),gates=r['gates'],
        actual_EFC=file_pin(a.output/'actual_EFC'/'report.json'),response_EFC=file_pin(a.output/'response_EFC'/'report.json'),
        actual_EFC_counts=[ae['passed'],ae['total']],response_EFC_counts=[pe['passed'],pe['total']],
        response_contracts=items,baseline_fidelity=fidelity,baseline_passed=sum(c['passed'] for c in fidelity),baseline_total=len(fidelity),
        passed=sum(sum(c['checks'].values()) for c in items),total=sum(len(c['checks']) for c in items),
        gates_passed=dict(live_isolation=isolation,response_contracts=quality,actual_force_receipts=actual_quality,all_baseline_ticks=all_fidelity),
        physical_response_interpretation_allowed=bool(isolation and quality and actual_quality and all_fidelity),
        optimizer_updates=0,native_integrations=0,
        scope='Independent saved source/control/force/state fidelity at predeclared gates over ALL24 baseline repeats. Common source copy/label quality does not replace native trajectory/force agreement. Every failed source/tick retained. All120 diagnostic responses NEVER FIT; no Actor/task/ego success.')
    write(a.output/'report.json',result)
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),contracts=[result['passed'],result['total']],
        actual_EFC=result['actual_EFC_counts'],response_EFC=result['response_EFC_counts'],baseline=[result['baseline_passed'],len(fidelity)],
        allowed=result['physical_response_interpretation_allowed'],worst={key:max(c['metrics'][key] for c in fidelity) for key in fidelity[0]['metrics']})))
    if not result['physical_response_interpretation_allowed']:raise ValueError('Preserve failed native fidelity; do not treat responses as qualified physical gradient labels')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

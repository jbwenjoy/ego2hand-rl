"""Reconstruct source-reference/EMA/absolute PD targets from saved actions."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main(args):
    if args.output.exists():raise ValueError('Fresh action audit output')
    r=json.loads(args.report.read_text(encoding='utf8'));raw=np.load(args.clip);qref=raw['reference_qpos'].astype(np.float32);items=[]
    names=[f'robot/right_finger{i}_joint{j}' for i in range(1,6) for j in range(1,5)]
    for c in r['cases']:
        if c['task']!='hand2-pen-spin':continue
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])));z=np.load(verify_file_pin(c['trace']));K=len(z['actions20']);n=round(.02/c['dt'])
        jids=np.array([m.joint(name).id for name in names]);aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==j)[0] for j in jids]);qa=m.jnt_qposadr[jids]
        lower=np.maximum(m.jnt_range[jids,0],m.actuator_ctrlrange[aids,0]).astype(np.float32);upper=np.minimum(m.jnt_range[jids,1],m.actuator_ctrlrange[aids,1]).astype(np.float32)
        residual=np.zeros(20,dtype=np.float32);pred=[];unlimited=[];memory=[]
        for action,phase in zip(z['actions20'],z['pre_action_phase']):
            residual=np.float32(.5)*np.clip(action*np.float32(.5),-.6,.6)+np.float32(.5)*residual
            target=qref[int(phase)]+residual;unlimited.append(target.copy());memory.append(residual.copy());pred.append(np.clip(target,lower,upper))
        pred=np.array(pred);actual=z['post_ctrl'].reshape(K,n,20)[:,:,aids]
        error=float(abs(actual-pred[:,None,:]).max());unlimited=np.array(unlimited)
        items.append(dict(task=c['task'],controller=c['controller'],max_target_error_rad=error,passed=error<2e-7,
            max_absolute_target_clipping_rad=float(abs(unlimited-pred).max()),clipped_coordinates=int(np.count_nonzero(unlimited!=pred)),
            initial_robot_reference_q_max_error_rad=float(abs(z['initial_qpos'][qa]-qref[0]).max()),
            first_reference_target_motion_max_rad=float(abs(qref[1]-qref[0]).max()),
            max_residual_rad=float(np.abs(memory).max()),scale=.5,alpha=.5,absolute_clip_scope='joint-hard / actuator-control intersection; residual memory is not clipped to absolute target bounds'))
    result=dict(schema='controlled_pen_action_contract_audit:1',source=file_pin(__file__),run=file_pin(args.report),clip=file_pin(args.clip),cases=items,
        passed=sum(c['passed'] for c in items),total=len(items),neural_updates=0,native_integration_steps=0,
        finding='ZERO follows the moving source joint reference with zero residual; it does not open the hand by targeting joint zero. Actor residual uses original .5 scale/.5 EMA. Full task failure remains.')
    args.output.mkdir();(args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
    if result['passed']!=result['total']:raise ValueError('Retain failed controller contract audit')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--clip',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

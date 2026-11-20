"""Counterfactual future commands at states where all eight decaying continuations failed."""
import argparse
from pathlib import Path
import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation
from test_hot3d_model_horizon import physics,file_pin,verify_file_pin,write_json,continuations,propose,MugStepChart


def main(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    oracle=file_pin(root/'mug_horizon_choice_audit_v1/report.json');audit=physics.read_json(oracle)
    trials=physics.read_json(physics.read_json(audit['intent'])['source'])
    fit=physics.read_json(physics.read_json(trials['intent'])['fit'])
    inputs=physics.read_json(fit['intent'])['inputs']
    old=physics.load_arrays(inputs['reference']);m=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['model'])))
    chart=MugStepChart(m,physics.read_json(inputs['binding']))
    selected={(r['clip'],r['seed'],r['phase']) for r in audit['rows'] if r['horizon']==3 and not r['any_feasible']}
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(audit=oracle,code=file_pin(__file__),classes=['decay','hold_first','return_IK','return_preload'],
        scope='Diagnose all three-window H3 states where no original shortlisted sequence passes endpoint pose gates. Same eight first commands. Keep first command fixed; vary controls two and three only. No credit for later replanning, no closed-loop execution.'))
    rows=[];sig=mujoco.mjtState.mjSTATE_FULLPHYSICS
    workers=[mujoco.MjData(m) for _ in range(4)]
    with Rollout(nthread=4) as roll:
        for name in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
            item,=physics.read_json(file_pin(root/name/'report.json'))['rows']
            ref=physics.load_arrays(item['reference']);T=ref['body'];clip=item['clip']
            base=old['feedforward_PD_target20'][0];ik=np.clip(ref['q'][1:]+base-old['initial_qpos27'][:20],*m.actuator_ctrlrange.T)
            pos=T[:,:3,3]+T[:,:3,:3]@chart.com;rot=Rotation.from_matrix(T[:,:3,:3])
            for row in trials['rows']:
                if row['clip']!=clip or row['mode']!='h3':continue
                arc=physics.load_arrays(row['trace']);seed=row['seed']
                d=mujoco.MjData(m);d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26'];d.ctrl[:]=base
                mujoco.mj_forward(m,d);last=base;res=np.zeros(20)
                for i,log in enumerate(row['records']):
                    key=clip,seed,i+1
                    if key in selected:
                        pre=np.empty(mujoco.mj_stateSize(m,sig));mujoco.mj_getState(m,d,pre,sig)
                        pool=propose(np.random.default_rng(np.random.SeedSequence([seed,i,0])),64,ik[i],base,last,res,m.actuator_ctrlrange)
                        cmd=pool[log['shortlist']];decay=continuations(cmd,ik,i,m.actuator_ctrlrange)
                        hold=np.repeat(cmd[:,None],3,axis=1);ki=hold.copy();ki[:,1:]=ik[i+1:i+3]
                        preload=hold.copy();preload[:,1:]=base
                        design=np.vstack([decay,hold,ki,preload])
                        states,_=roll.rollout(m,workers,np.tile(pre,(32,1)),control=np.repeat(design,64,axis=1),
                            initial_warmstart=np.tile(d.qacc_warmstart,(32,1)))
                        ratio=[]
                        for h in (1,2,3):
                            p,R,_,_=chart.motion(states[:,h*64-1])
                            ratio.append(np.maximum(100*np.linalg.norm(p-pos[i+h],axis=1),
                                np.rad2deg((R*rot[i+h].inv()).magnitude())/10).reshape(4,8))
                        valid=(np.max(ratio,axis=0)<=1)
                        assert not valid[0].any()
                        rows.append(dict(clip=clip,seed=seed,phase=i+1,first_feasible=int((ratio[0][0]<=1).sum()),
                            feasible_counts=valid.sum(1).tolist(),best_ratio=np.max(ratio,axis=0).min(1).tolist()))
                    cmd=arc['ctrl'][i];d.ctrl[:]=cmd;mujoco.mj_step(m,d,nstep=64);np.testing.assert_array_equal(d.qpos,arc['qpos'][i])
                    last=cmd;res=last-ik[i]
    assert len(rows)==len(selected)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out.resolve())

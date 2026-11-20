"""Paired pose/wrench/shuffled ranking on two unilateral HOT3D mug windows."""
import argparse
from pathlib import Path
from time import perf_counter
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from retarget_adjacent_mugs import make_grid,physics,file_pin,verify_file_pin,write_json
from supermimic_research.mechanics.mean_reference_wrench import mean_reference_wrench
from supermimic_research.mechanics.mug_contact_impulse import MugContactImpulse


def execute(m,binding,z,grid,base,mode):
    sensor=MugContactImpulse(m,binding)
    body=sensor.body;qa=binding['mug_free_qpos']
    rot=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
    inertial=Rotation.from_quat(m.body_iquat[body,[1,2,3,0]]).as_matrix()
    mass=m.body_mass[body];weight=mass*np.linalg.norm(m.opt.gravity)
    targets=mean_reference_wrench(z['COM_world_m'],z['quaternion_wxyz'],mass,
        inertial@np.diag(m.body_inertia[body])@inertial.T,m.opt.gravity)
    actual=mujoco.MjData(m);clone=mujoco.MjData(m)
    actual.qpos[:]=z['initial_qpos27'];actual.qvel[:]=z['initial_qvel26'];actual.ctrl[:]=base
    mujoco.mj_forward(m,actual);audit=0.
    def advance(d,cmd,i,check=False):
        nonlocal audit
        assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
        initial=Rotation.from_quat(d.qpos[qa+np.array([4,5,6,3])])
        load=np.zeros(6);peak=0.
        d.ctrl[:]=cmd
        for _ in range(64):
            mujoco.mj_step(m,d)
            w,table,err=sensor.hand_wrench(m,d,verify=check)
            load+=w;peak=max(peak,table)
            if check:audit=max(audit,err)
        load/=64
        mujoco.mj_kinematics(m,d)
        end=Rotation.from_quat(d.qpos[qa+np.array([4,5,6,3])])
        com=d.qpos[qa:qa+3]+end.apply(z['COM_BODY_m'])
        endpoint=min(i+1,len(rot)-1)
        mm=1000*np.linalg.norm(com-z['COM_world_m'][endpoint])
        deg=np.rad2deg((end*rot[endpoint].inv()).magnitude())
        pose=(mm/10)**2+(deg/10)**2+100*(peak>.1)+.01*np.mean(((cmd-base)/.12)**2)
        cost=0.
        if i<len(targets['valid_interval']) and targets['valid_interval'][i]:
            diff=(load-targets['wrench_world'][i]).reshape(2,3)
            diff=initial.inv().apply(diff)/np.array([[weight],[weight*.05]])
            e=np.sum(diff**2);cost=e/(1+e)
        return pose,cost,mm,deg,peak,initial.inv().apply(load.reshape(2,3)).ravel()
    decisions=[];qpos=[];cmds=[];ticks=0
    for i,pool in enumerate(grid):
        costs=[];first=[]
        for branch in pool:
            mujoco.mj_copyData(clone,m,actual)
            score=np.zeros(2)
            for k,u in enumerate(branch):
                step=advance(clone,u,i+k)
                score+=step[:2];ticks+=64
                if k==0:first.append(clone.qpos.copy())
            costs.append(score)
        pose,support=np.asarray(costs).T
        choices=dict(pose=int(np.argmin(pose)),wrench=int(np.argmin(pose+.2*support)),
                     shifted=int(np.argmin(pose+.2*np.roll(support,-5))))
        j=choices[mode];cmd=pool[j,0].copy()
        _,w,mm,deg,table,load=advance(actual,cmd,i,True)
        np.testing.assert_array_equal(first[j],actual.qpos)
        passed=bool(mm<=10 and deg<=10 and table<=.1)
        decisions.append(dict(phase=i+1,choices=choices,choice=j,pose=pose.tolist(),support=support.tolist(),
            valid=passed,mm=mm,deg=deg,table_N=table,executed_support=w,executed_wrench_BODY=load.tolist()))
        qpos.append(actual.qpos.copy());cmds.append(cmd)
        if not passed:break
    if audit>1e-7:raise ValueError(f'Mug free joint / contact wrench check failed: {audit}')
    return np.array(cmds),np.array(qpos),decisions,ticks,audit


def main(out):
    root=Path(__file__).resolve().parents[1]/'runs';sources=[file_pin(root/name/'report.json') for name in
        ['mug_annotated_windows_original_v1','mug_unilateral_interval_v1']]
    inputs=physics.read_json(physics.read_json(sources[0])['intent'])['inputs']
    binding=physics.read_json(inputs['binding']);old=physics.load_arrays(inputs['reference'])
    modelpath=verify_file_pin(inputs['model']);m=mujoco.MjModel.from_binary_path(str(modelpath))
    assert np.isclose(m.opt.timestep*64,.02)
    out.mkdir(exist_ok=False,parents=True)
    write_json(out/'intent.json',dict(sources=sources,inputs=inputs,
        code=[file_pin(__file__),file_pin(Path(__file__).resolve().parents[1]/'src/supermimic_research/mechanics/mug_contact_impulse.py'),
              file_pin(Path(__file__).with_name('retarget_adjacent_mugs.py'))],
        seeds=[8201,8202,8203],modes=['pose','wrench','shifted'],shift=5,coefficient=.2,
        scope='Every condition queries the identical per-phase 12x3 absolute PD library for its seed and clip. Execute one 20ms window, stop at first breach of 10mm/10deg/0.1N. Mean interval hand wrench compared with nominal object momentum change in interval-start BODY frame; no human force labels. Costs defined only on valid reference momentum intervals. Two previously inspected same-session windows.'))
    rows=[]
    for pin in sources:
        row,=physics.read_json(pin)['rows']
        with np.load(verify_file_pin(row['reference'])) as f:ref=dict(f)
        T=ref['body'];z={**old,'elapsed_s':ref['t'],'COM_world_m':T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'],
            'quaternion_wxyz':Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]}
        base=old['feedforward_PD_target20'][0];preload=base-old['initial_qpos27'][:20]
        ik=np.clip(ref['q'][1:]+preload,*m.actuator_ctrlrange.T)
        for seed in (8201,8202,8203):
            grid=make_grid(seed,ik,base,preload,m.actuator_ctrlrange)
            for mode in ('pose','wrench','shifted'):
                start=perf_counter();cmd,qpos,ds,ticks,error=execute(m,binding,z,grid,base,mode)
                duration=perf_counter()-start
                detail,trace=physics.replay(modelpath,binding,z,base,cmd,1.,1.)
                np.testing.assert_array_equal(trace['qpos'],qpos)
                assert np.isclose(detail['valid_s'],.02*sum(s['valid'] for s in ds))
                if seed in (8201,8202) and mode=='pose':
                    gold=next(v for v in row['stats'] if v['arm']==f'lookahead_{seed}')
                    with np.load(verify_file_pin(gold['trace'])) as f:np.testing.assert_array_equal(qpos,f['qpos'][:len(qpos)])
                path=out/f"{row['clip']}_{seed}_{mode}.npz"
                np.savez_compressed(path,**trace)
                rows.append(dict(clip=row['clip'],seed=seed,mode=mode,valid_s=detail['valid_s'],
                    full=all(s['valid'] for s in ds) and len(ds)==len(grid),first_breach=detail['first_breach'],
                    planning_ticks=ticks,seconds=duration,contact_sum_max_error=error,decisions=ds,trace=file_pin(path)))
                write_json(out/'progress.json',rows)
                print(row['clip'],seed,mode,detail['valid_s'],flush=True)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out.resolve())

"""A bounded simultaneous stick-point displacement correction on the fixed ego reference."""
from pathlib import Path
import json,sys
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from probe_mug_loss_onset import ROOT,read,verify_file_pin,file_pin,write_json


def correction(m,d,body,pos_error,angle_error,gain):
    jac=[];goals=[];weights=[]
    for ic,c in enumerate(d.contact):
        g,h=map(int,c.geom);on=m.geom_bodyid[[g,h]]==body
        if on.sum()!=1:continue
        other=h if on[0] else g
        if not m.geom(other).name.startswith('right_finger'):continue
        f=np.zeros(6);mujoco.mj_contactForce(m,d,ic,f)
        if f[0]<=.1:continue
        n=c.frame[:3];P=np.eye(3)-np.outer(n,n)
        jp=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,None,c.pos,int(m.geom_bodyid[other]))
        # Aim at common rigid motion, discarding the part which changes normal closure.
        desired=pos_error+np.cross(angle_error,c.pos-d.xipos[body])
        jac.append(P@jp[:,:20]);goals.append(P@desired);weights.append(min(f[0],5.))
    if not jac:return np.zeros(20)
    weights=np.sqrt(np.array(weights)/sum(weights))
    A=np.vstack([w*J for w,J in zip(weights,jac)]);b=np.concatenate([w*v for w,v in zip(weights,goals)])
    delta=np.linalg.solve(A.T@A+.015**2*np.eye(20),A.T@b)*gain
    return delta/max(1.,abs(delta).max()/.05)


def main(out):
    out.mkdir(exist_ok=False);p=ROOT/'runs';ffpin=file_pin(p/'mug_temporal_feedforward_v1/report.json')
    ff=read(ffpin);trial=read(read(ff['intent'])['parent']);contract=read(read(trial['dynamic'])['intent'])
    ref=read(contract['reference']);bind=read(contract['binding'])
    with np.load(verify_file_pin(ff['results']['interval_lp']['trace'])) as f:base=f['ctrl'];gold=f['qpos']
    with np.load(verify_file_pin(ref['output'])) as f:z=dict(f)
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(ref['model'])));qa=bind['mug_free_qpos'];body=bind['mug_body_id']
    rref=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
    write_json(out/'intent.json',dict(base=ffpin,gains=[0,.25,.5,1.],damping_m_per_rad=.015,max_delta_rad=.05,reference='current timestamp pose PD displacement, future reference supplies existing nominal commands',scope='Same single annotated clip and nominal dynamics. No externally applied forces; perfect-state feedback.'))
    rows=[]
    for gain in (0,.25,.5,1.):
        d=mujoco.MjData(m);d.qpos[:]=z['initial_qpos27'];d.qvel[:]=z['initial_qvel26'];d.ctrl[:]=base[0];mujoco.mj_forward(m,d)
        sensor=mujoco.MjData(m);q=[];commands=[];active=0
        for i in range(100):
            mujoco.mj_copyData(sensor,m,d);mujoco.mj_forward(m,sensor)
            R=Rotation.from_quat(sensor.qpos[qa+np.array([4,5,6,3])])
            delta=correction(m,sensor,body,z['COM_world_m'][i]-sensor.xipos[body],(rref[i]*R.inv()).as_rotvec(),gain)
            cmd=np.clip(base[i+1]+delta,*m.actuator_ctrlrange.T);active+=int(np.any(abs(cmd-base[i+1])>1e-8));commands.append(cmd)
            d.ctrl[:]=cmd;assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
            mujoco.mj_step(m,d,nstep=64);mujoco.mj_kinematics(m,d);q.append(d.qpos.copy())
        q=np.array(q);assert np.isfinite(q).all() and abs(d.time-2)<1e-8
        if gain==0:np.testing.assert_array_equal(q,gold)
        R=Rotation.from_quat(q[:,qa+np.array([4,5,6,3])]);err=np.linalg.norm(q[:,qa:qa+3]+R.apply(z['COM_BODY_m'])-z['COM_world_m'][1:],axis=1)
        ang=np.rad2deg((R*rref[1:].inv()).magnitude());first=lambda v:float(.02*(1+np.flatnonzero(v)[0])) if v.any() else None
        path=out/f'g{gain}.npz';np.savez_compressed(path,qpos=q,ctrl=commands,error_m=err,angle_deg=ang)
        rows.append(dict(gain=gain,trace=file_pin(path),adjusted_intervals=active,first_cm=first(err>.01),first_10deg=first(ang>10),com_rms_mm=1000*np.sqrt(np.mean(err**2)),angle_rms_deg=np.sqrt(np.mean(ang**2))))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows,native=25600));print([(r['gain'],r['first_cm'],r['first_10deg'],round(r['com_rms_mm'])) for r in rows])
if __name__=='__main__':main(Path(sys.argv[1]).resolve())

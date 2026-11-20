"""Closed-loop contact and pose ablations on the recorded mug trajectory."""
import json,sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from evaluate_mug_sequence_feedforward import ROOT,wrench_target,mean_reference_wrench,file_pin,verify_file_pin,write_json


def run(out):
    out.mkdir(exist_ok=False);root=ROOT/'runs'
    parent=root/'mug_temporal_feedforward_v1/report.json'
    read=lambda p:json.loads(verify_file_pin(p).read_bytes())
    ff=json.loads(parent.read_bytes());traj=read(read(ff['intent'])['parent'])
    ref=read(read(traj['dynamic'])['intent'])
    binding=read(ref['binding']);ref=read(ref['reference'])
    with np.load(verify_file_pin(ref['output'])) as f:z=dict(f)
    with np.load(verify_file_pin(ff['results']['interval_lp']['trace'])) as f:base=f['ctrl'];gold=f['qpos']
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(ref['model'])))
    hand={m.geom(n).id for n in binding['hand_collision_names']};body=binding['mug_body_id'];qa=binding['mug_free_qpos'];da=binding['mug_free_dof']
    fixedI=Rotation.from_quat(m.body_iquat[body,[1,2,3,0]]).as_matrix()
    inertia=fixedI@np.diag(m.body_inertia[body])@fixedI.T
    mass=m.body_mass[body];prior=mean_reference_wrench(z['COM_world_m'],z['quaternion_wxyz'],mass,inertia,m.opt.gravity)
    refs=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
    write_json(out/'intent.json',dict(parent=file_pin(parent),arms=['replay','live_contacts','live_pose_PD'],gains='linear/rotational acceleration P=100 s^-2 and D=20 s^-1; prior uses same reference average wrench; final three intervals and infeasible LPs fall back to archived feedforward. Perfect-state diagnostic, no object assist.'))
    arms={}
    for mode in ('replay','live_contacts','live_pose_PD'):
        d=mujoco.MjData(m);sensor=mujoco.MjData(m);d.qpos[:]=z['initial_qpos27'];d.qvel[:]=z['initial_qvel26'];d.ctrl[:]=base[0];mujoco.mj_forward(m,d)
        q=[];rows=[];commands=[];substeps=round(.02/m.opt.timestep)
        for i in range(100):
            cmd=base[i+1].copy();status='archived';cup_table=0;cup_hand=0
            if mode!='replay' and prior['valid_interval'][i]:
                sensor.qpos[:]=d.qpos;sensor.qvel[:]=d.qvel;sensor.ctrl[:]=d.ctrl;mujoco.mj_forward(m,sensor)
                R=Rotation.from_quat(d.qpos[qa+np.array([4,5,6,3])])
                omega=R.apply(d.qvel[da+3:da+6]);vel=d.qvel[da:da+3]+np.cross(omega,R.apply(z['COM_BODY_m']))
                w=prior['wrench_world'][i].copy()
                if mode=='live_pose_PD':
                    w[:3]+=mass*(100*(z['COM_world_m'][i]-sensor.xipos[body])+20*(z['COM_velocity_world_m_s'][i]-vel))
                    w[3:]+=R.as_matrix()@inertia@R.as_matrix().T@(100*(refs[i]*R.inv()).as_rotvec()+20*(z['angular_velocity_world_rad_s'][i]-omega))
                neutral=d.qpos[:20]-m.actuator_biasprm[:,2]*d.qvel[:20]/m.actuator_gainprm[:,0]
                candidate,status=wrench_target(m,sensor,body,hand,w,neutral,cmd-neutral)
                if candidate is not None:cmd=candidate
            d.ctrl[:]=cmd;commands.append(cmd)
            for _ in range(substeps):
                assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
                mujoco.mj_step(m,d);hand_now=table_now=False
                for ic,c in enumerate(d.contact):
                    g,h=map(int,c.geom);on=m.geom_bodyid[[g,h]]==body
                    if on.sum()!=1:continue
                    other=h if on[0] else g;f=np.zeros(6);mujoco.mj_contactForce(m,d,ic,f)
                    if f[0]>.1:
                        hand_now|=other in hand;table_now|=other==m.geom('mug_task_table').id
                cup_hand+=hand_now;cup_table+=table_now
            mujoco.mj_kinematics(m,d);q.append(d.qpos.copy())
            rows.append(dict(interval=i,status=status,hand_ticks=cup_hand,table_ticks=cup_table))
        q=np.array(q);assert np.isfinite(q).all() and abs(d.time-2)<1e-8
        if mode=='replay':np.testing.assert_array_equal(q,gold)
        R=Rotation.from_quat(q[:,qa+np.array([4,5,6,3])]);pos=q[:,qa:qa+3]+R.apply(z['COM_BODY_m'])
        dist=np.linalg.norm(pos-z['COM_world_m'][1:],axis=1);ang=np.rad2deg((R*refs[1:].inv()).magnitude())
        first=lambda x:float(.02*(np.flatnonzero(x)[0]+1)) if np.any(x) else None
        p=out/(mode+'.npz');np.savez_compressed(p,qpos=q,cmd=commands,com_err_m=dist,angle_deg=ang)
        arms[mode]=dict(solved=sum(r['status']=='ok' for r in rows),first_cm=first(dist>.01),first_10deg=first(ang>10),
            rms_mm=float(1000*np.sqrt(np.mean(dist**2))),rms_deg=float(np.sqrt(np.mean(ang**2))),rows=rows,trace=file_pin(p))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),arms=arms,steps=3*100*substeps))
    print({k:[v[x] for x in ('solved','first_cm','first_10deg','rms_mm','rms_deg')] for k,v in arms.items()})
if __name__=='__main__':run(Path(sys.argv[1]).resolve())

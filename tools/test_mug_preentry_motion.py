"""Estimate motion from pre-window annotations and cross hand/object velocity initialization."""
import argparse
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from retarget_adjacent_mugs import read_hand_window,reconstruct,transform_points,physics,file_pin,verify_file_pin,write_json
import torch


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    runs=[file_pin(root/k/'report.json') for k in ('mug_annotated_windows_original_v1','mug_screened_transfer_v1')]
    inputs=physics.read_json(physics.read_json(runs[0])['intent'])['inputs']
    base=physics.load_arrays(inputs['reference']);binding=physics.read_json(inputs['binding'])
    model_path=verify_file_pin(inputs['model']);m=mujoco.MjModel.from_binary_path(str(model_path))
    d=mujoco.MjData(m);d.qpos[:]=base['initial_qpos27'];mujoco.mj_forward(m,d)
    sites=[m.site(f'right_finger{k}_tip').id for k in range(1,6)]
    fixed=base['anchor_world_from_BODY'];init=d.qpos.copy()
    robot_body_tip=transform_points(np.linalg.inv(fixed),d.site_xpos[sites])
    hand_joints=m.actuator_trnid[:,0];hand_dofs=m.jnt_dofadr[hand_joints]
    qa=binding['mug_free_qpos'];da=m.jnt_dofadr[m.body_jntadr[binding['mug_body_id']]]
    J=[]
    for site in sites:
        jac=np.zeros((3,m.nv));mujoco.mj_jacSite(m,d,jac,None,site)
        J.append(jac[:,hand_dofs])
    J=np.concatenate(J)
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(sources=runs,code=file_pin(__file__),scope=
        'Diagnostic using poses at source frames start-2 and start only. Independent hand/object initial velocities, identical start pose, preexisting fixed commands. Hand uses damped least-squares site Jacobian (0.002m/rad), object uses MuJoCo differentiatePos. No contact forces applied. Initialized motion is an oracle boundary assumption; no from-rest success claim.'))
    results=[]
    torch.set_num_threads(2)
    for pin in runs:
        row,=physics.read_json(pin)['rows'];clip=row['clip']
        frame=row['window']['start_frame']
        past=read_hand_window(verify_file_pin(row['archive']),list(range(frame-2,frame+1)),'right',9,row['window']['object_uid'])
        fk=reconstruct(past['profile'],past['angles22'],past['wrist'],'right')
        wrist_obj=np.linalg.inv(past['wrist'])@past['objects']
        poses=fixed@np.linalg.inv(wrist_obj[-1])@wrist_obj
        human=np.array([transform_points(np.linalg.inv(T),tip[:5]) for T,tip in zip(past['objects'],fk['landmarks21'])])
        positions=np.array([transform_points(T,robot_body_tip+h-human[-1]) for T,h in zip(poses,human)])
        dt=(past['timestamps_ns'][-1]-past['timestamps_ns'][0])*1e-9
        desired=((positions[-1]-positions[0])/dt).ravel()
        dq=np.linalg.solve(J.T@J+.002**2*np.eye(20),J.T@desired)
        earlier=init.copy();earlier[qa:qa+3]=poses[0,:3,3]
        earlier[qa+3:qa+7]=Rotation.from_matrix(poses[0,:3,:3]).as_quat()[[3,0,1,2]]
        vel=np.zeros(m.nv);mujoco.mj_differentiatePos(m,vel,dt,earlier,init)
        with np.load(verify_file_pin(row['reference'])) as f:ref=dict(f)
        T=ref['body'];q=Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]
        z={**base,'elapsed_s':ref['t'],'COM_world_m':T[:,:3,3]+T[:,:3,:3]@base['COM_BODY_m'],'quaternion_wxyz':q}
        stats=[]
        for trial in row['stats']:
            if trial['arm'] not in ('direct','lookahead_8201'):continue
            old=physics.load_arrays(trial['trace'])
            for hand,obj in [(False,False),(True,False),(False,True),(True,True)]:
                initial_v=np.zeros(m.nv)
                if hand:initial_v[hand_dofs]=dq
                if obj:initial_v[da:da+6]=vel[da:da+6]
                detail,trace=physics.replay(model_path,binding,{**z,'initial_qvel26':initial_v},
                    base['feedforward_PD_target20'][0],old['ctrl'],1.,1.)
                if not (hand or obj):
                    np.testing.assert_array_equal(trace['qpos'],old['qpos'])
                    assert detail['valid_s']==trial['valid_s']
                path=out/f'{clip}_{trial["arm"]}_{int(hand)}{int(obj)}.npz'
                np.savez_compressed(path,**trace)
                stats.append(dict(arm=trial['arm'],hand=hand,obj=obj,**detail,trace=file_pin(path)))
        results.append(dict(clip=clip,frames=list(range(frame-2,frame+1)),dt=dt,hand_qvel=dq.tolist(),
            max_hand_rad_s=float(np.max(abs(dq))),velocity_tip_residual_mm_s=1000*np.sqrt(np.mean((J@dq-desired)**2)),
            mug_free_velocity=vel[da:da+6].tolist(),mug_root_speed_m_s=float(np.linalg.norm(vel[da:da+3])),
            stats=stats))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),clips=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())

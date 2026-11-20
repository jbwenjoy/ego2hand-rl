"""Real annotated HOT3D clip 1884, anchored fingertip motion, unassisted CPU tracking."""
import json,sys
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json
from supermimic_research.mechanics.mug_sequence_ik import anchored_targets,solve_trajectory


def main(out):
    out.mkdir(exist_ok=False)
    reports=ROOT/'runs';rp=reports/'mug_closed_loop_reference_v1/report.json';bp=reports/'right_mug_frames_v4/binding.json'
    ref=json.loads(rp.read_bytes());binding=json.loads(bp.read_bytes())
    write_json(out/'intent.json',dict(reference=file_pin(rp),binding=file_pin(bp),clip='001884',frames=[47,107],
        comparison=['constant archived grasp target','object-relative five-tip displacement IK + archived preload'],dt=.02,
        scope='Known-lineage annotated right hand and nominal free mug. Source clock preserved; relative fingertip displacement anchored to robot starting sites, magnitudes retained without scale fitting. All control intervals retained including post-failure.'))
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(ref['model'])))
    with np.load(verify_file_pin(binding['reference'])) as f:source=f['hand_landmarks21_object_BODY_F64'][:,:5];ts=f['timestamps_ns']
    with np.load(verify_file_pin(ref['output'])) as f:z=dict(f)
    np.testing.assert_array_equal(ts,z['source_timestamps_ns'])
    i=z['source_left_index'];a=z['source_interval_alpha'][:,None,None]
    tips=(1-a)*source[i]+a*source[i+1]
    pose=z['T_world_from_BODY'];qinit=z['initial_qpos27'];q0=qinit[:20]
    body=binding['mug_body_id'];table=model.geom('mug_task_table').id
    d=mujoco.MjData(model);d.qpos[:]=qinit;mujoco.mj_forward(model,d)
    tip_ids=[model.site(f'right_finger{k}_tip').id for k in range(1,6)]
    start=d.site_xpos[tip_ids].copy();targets=anchored_targets(pose,tips,start)
    seq,fitted,ok=solve_trajectory(model,qinit,targets)
    offset=z['feedforward_PD_target20'][0]-q0
    def error_mm(x):return float(1000*np.sqrt(np.mean(np.sum(x*x,axis=-1))))
    summary=dict(solved=sum(ok),frames=len(seq),static_rms_mm=error_mm(start-targets),articulated_rms_mm=error_mm(fitted-targets),
                 per_digit_rms_mm=[error_mm(fitted[:,k]-targets[:,k]) for k in range(5)],max_joint_speed_rad_s=float(np.max(abs(np.diff(seq,axis=0)))/.02))
    np.savez_compressed(out/'trajectory.npz',timestamps=z['timestamps_ns'],q20=seq,reference_body=pose,targets=targets,sites=fitted)
    substeps=round(.02/model.opt.timestep);assert abs(substeps*model.opt.timestep-.02)<1e-12
    arms={}
    for arm,q in [('constant',np.tile(q0,(len(seq),1))),('retarget',seq)]:
        cmd=np.clip(q+offset,*model.actuator_ctrlrange.T)
        assert np.allclose(cmd[0],z['feedforward_PD_target20'][0])
        d=mujoco.MjData(model);d.qpos[:]=qinit;d.qvel[:]=z['initial_qvel26'];d.ctrl[:]=cmd[0];mujoco.mj_forward(model,d)
        centres=[];angles=[];tip_errors=[];contact_steps=0;table_steps=0;max_pen=0.
        actual_q=[];actual_obj=[]
        for k in range(1,len(seq)):
            d.ctrl[:]=cmd[k]
            for _ in range(substeps):
                if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise ValueError('External applied force')
                mujoco.mj_step(model,d);touch=False;tab=False
                for j,con in enumerate(d.contact):
                    g,h=map(int,con.geom);on=model.geom_bodyid[[g,h]]==body
                    if on.sum()!=1:continue
                    other=h if on[0] else g
                    max_pen=max(max_pen,-con.dist)
                    cf=np.zeros(6);mujoco.mj_contactForce(model,d,j,cf)
                    if cf[0]>.1:
                        if other==table:tab=True
                        elif (model.geom(other).name or '') in binding['hand_collision_names']:touch=True
                contact_steps+=touch;table_steps+=tab
            # Refresh derived poses after the final integration of the control interval.
            mujoco.mj_kinematics(model,d)
            centres.append(np.linalg.norm(d.xipos[body]-z['COM_world_m'][k]))
            angles.append(np.rad2deg(Rotation.from_matrix(d.xmat[body].reshape(3,3)@pose[k,:3,:3].T).magnitude()))
            tip_errors.append(d.site_xpos[tip_ids]-targets[k])
            actual_q.append(d.qpos.copy());actual_obj.append(d.xpos[body].copy())
        assert np.all(np.isfinite(actual_q)) and abs(d.time-2)<1e-8
        path=out/(arm+'.npz');np.savez_compressed(path,qpos=actual_q,object_pos=actual_obj,cmd=cmd,com_error_m=centres,angle_deg=angles,tip_error_m=tip_errors)
        arms[arm]=dict(trace=file_pin(path),actual_tip_rms_mm=error_mm(np.asarray(tip_errors)),com_rms_mm=float(1000*np.sqrt(np.mean(np.square(centres)))),max_com_mm=float(1000*max(centres)),
            angle_rms_deg=float(np.sqrt(np.mean(np.square(angles)))),max_penetration_mm=1000*max_pen,hand_loaded_ticks=contact_steps,table_loaded_ticks=table_steps,
            saturated_targets=int((abs(q+offset-cmd)>1e-9).sum()),native_steps=substeps*(len(seq)-1))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),kinematics=summary,arms=arms,trajectory=file_pin(out/'trajectory.npz')))
    print(summary,arms)
if __name__=='__main__':main(Path(sys.argv[1]).resolve())

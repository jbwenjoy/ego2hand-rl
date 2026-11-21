"""Paired CPU debugging of wrist commands and grasp acquisition with fixed reference and physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from probe_flash_wuji_replay import rebuilt
from supermimic_research.runtime.flash_quality import SolverRecorder,world_tips


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm(mode,source,checkpoint,control_frames,actor):
    env=rebuilt(source,checkpoint,True)
    side=env.scene.layout.hand("right")
    pos=np.asarray(side.wrist_pos_dof_ids);wq=np.asarray(side.wrist_pos_q_ids)
    fingers=np.asarray(side.finger_dof_ids)
    reference=env.scene.robot_reference.joint_target
    base_wrist=reference[:,pos].copy()
    rec=SolverRecorder(env);solver_step=env.solver.step
    actuator=[];actual_substep=[];sampled=0
    sim_dt=env.config.sim.sim_dt; rate=env.config.sim.fps; dec=env.config.sim.substeps
    vmax=env.model.joint_velocity_limit.numpy()[pos]
    effort=env.model.joint_effort_limit.numpy()[pos]
    stop={}
    sim_model=env.solver.mj_model
    mpos=[]
    for axis in 'xyz':
        j=next(i for i in range(sim_model.njnt) if sim_model.joint(i).name.endswith('right_wrist_'+axis))
        mpos.append(int(sim_model.jnt_dofadr[j]))
    def observe(s0,s1,c,contacts,dt):
        nonlocal sampled
        i=sampled//dec
        if mode=='zero_plus_linear_vref':
            target=c.joint_target_vel.numpy().copy()
            target[pos]=np.clip((base_wrist[i+1]-base_wrist[i])*rate,-vmax,vmax)
            c.joint_target_vel.assign(target)
        solver_step(s0,s1,c,contacts,dt)
        actual_substep.append(np.r_[s1.joint_q.numpy()[wq],s1.joint_qd.numpy()[pos]])
        actuator.append(env.solver.mjw_data.qfrc_actuator.numpy()[0,mpos].copy())
        sampled+=1
    env.solver.step=observe
    for i in range(control_frames):
        if int(env.timestep.numpy()[0])!=i: raise RuntimeError('Reference frame index changed')
        if mode in ('recorded_actor','actor_without_translation'):
            command=actor[i].copy()
            if mode=='actor_without_translation':
                if env.action.block_names[0]!='right_wrist_position_residual_normalized' or env.action.block_ranges[0]!=(0,3):
                    raise ValueError("Could not identify wrist translation channels")
                command.reshape(-1)[:3]=0.
        else:
            command=np.zeros_like(actor[i])
        env.step(wp.array(command.reshape(env.action_input.shape),dtype=wp.float32,device=env.device))
        rec.end_frame()
        if env.done.numpy()[0] or env.truncation.numpy()[0]:
            stop=dict(ended_after_interval=i+1,causes=[n for n,v in zip(env.termination.cause_names,env.termination.cause_arrays)
                                                      if v.numpy()[0]])
            break
    rec.close();tr=rec.materialize();num=len(tr["joint_q"])-1
    rows=tr["contact_rows"];offsets=tr["contact_offsets"];obj=tr["object_shape_ids"];hand=tr["hand_shape_ids"]
    hobj=((np.isin(rows[:,0],obj)&np.isin(rows[:,1],hand))|(np.isin(rows[:,1],obj)&np.isin(rows[:,0],hand)))&(rows[:,9]>.01)
    counts=np.diff(np.r_[0,np.cumsum(hobj.astype(np.int64))][offsets])
    opposed=np.zeros(len(counts),dtype=bool)
    for s in np.flatnonzero(counts>=2):
        selected=rows[offsets[s]:offsets[s+1]][hobj[offsets[s]:offsets[s+1]]]
        hard=[]
        for row in selected:
            first_obj=int(row[0]) in obj
            geom=int(row[1] if first_obj else row[0]);name=env.model.shape_label[geom].split("/")[-1]
            hard.append((name,(-1. if first_obj else 1.)*row[3:6]))
        opposed[s]=any(np.dot(t,o)<-.5 for n,t in hard if 'right_finger1_' in n
                       for m,o in hard if any('right_finger'+str(k)+'_' in m for k in (2,3,4,5)))
    tgt=tr["command_joint_target_pos"][:,pos].reshape(num,dec,3)
    vcmd=tr["command_joint_target_vel"][:,pos].reshape(num,dec,3)
    commands=tgt.mean(axis=1)
    realised=tr["joint_q"][1:,wq];vel=tr["joint_qd"][1:,pos]
    alias=base_wrist[:num]-base_wrist[1:num+1]
    residual=commands-base_wrist[:num]
    plant=realised-commands
    error=realised-base_wrist[1:num+1]
    assert np.allclose(error,alias+residual+plant,atol=2e-7)
    forces=np.asarray(actuator).reshape(num,dec,3)
    target_angular=tr['command_joint_target_pos'][:,side.wrist_orient_dof_ids].reshape(num,dec,3)[:,-1]
    cur=Rotation.from_quat(tr['body_q'][1:,side.palm_frame.body_id,3:])
    refrot=Rotation.from_quat(env.scene.robot_reference.hand('right').wrist_quat_w[1:num+1],scalar_first=True)
    orn_error=(cur*refrot.inv()).as_rotvec()
    objpos=tr['body_q'][:,rec.obj_id,:3]
    arrays=dict(frames=np.arange(num+1),source_wrist_m=base_wrist[:num+1],source_wrist_delta_m_s=(base_wrist[1:num+1]-base_wrist[:num])*rate,
                realised_wrist_m=tr['joint_q'][:,wq],realised_wrist_velocity_m_s=tr['joint_qd'][:,pos],
                executed_position_targets_m=tgt,executed_velocity_targets_m_s=vcmd,
                wrist_substep_position_velocity=np.stack(actual_substep),actuator_force_xyz_N=forces,
                source_to_command_residual_m=residual,physics_tracking_error_m=plant,one_source_frame_shift_m=alias,end_to_source_error_m=error,
                actual_rotation_error_rad=orn_error,actual_applied_target_rotvec=target_angular,
                measured_hand_carton_contacts_over_001N=counts.reshape(num,dec),
                measured_opposed_thumb_substeps=opposed.reshape(num,dec),
                fingertip_world=world_tips(tr["body_q"],side),object_position_m=objpos,
                finger_target_residual_rad=(tr['command_joint_target_pos'][:,fingers].reshape(num,dec,20)[:,-1,:]-reference[:num,fingers]))
    summaries={}
    for label,first,last in [('early',0,25),('acquisition',26,35),('rise',36,46),('all',1,num)]:
        # End-frame convention: interval i->i+1 has endpoint f=i+1.
        sl=slice(max(first-1,0),min(last,num))
        if (sl.stop<=sl.start):continue
        forcemag=np.abs(forces[sl]);c=arrays['measured_hand_carton_contacts_over_001N'][sl]
        summaries[label]=dict(source_endframes=[sl.start+1,sl.stop],
            rms_wrist_error_mm=float(1000*np.sqrt(np.mean(np.sum(error[sl]**2,axis=1)))),
            mean_source_wrist_mm=(1000*base_wrist[1:num+1][sl].mean(axis=0)).tolist(),
            mean_source_to_command_residual_mm=(1000*residual[sl].mean(axis=0)).tolist(),
            mean_physics_tracking_error_mm=(1000*plant[sl].mean(axis=0)).tolist(),
            mean_one_frame_shift_mm=(1000*alias[sl].mean(axis=0)).tolist(),
            mean_end_to_source_error_mm=(1000*error[sl].mean(axis=0)).tolist(),
            mean_source_velocity_m_s=arrays['source_wrist_delta_m_s'][sl].mean(axis=0).tolist(),
            mean_target_velocity_m_s=vcmd[sl].mean(axis=(0,1)).tolist(),mean_actual_velocity_m_s=vel[sl].mean(axis=0).tolist(),
            actual_force_mean_N=forces[sl].mean(axis=(0,1)).tolist(),max_abs_force_N=forcemag.max(axis=(0,1)).tolist(),
            saturation_fraction_by_axis=(forcemag>=effort[None,None,:]-1e-3).mean(axis=(0,1)).tolist(),
            source_to_cmd_translation_rms_mm=float(1000*np.sqrt(np.mean(np.sum(residual[sl]**2,axis=1)))),
            physics_to_cmd_translation_rms_mm=float(1000*np.sqrt(np.mean(np.sum(plant[sl]**2,axis=1)))),
            source_one_frame_shift_rms_mm=float(1000*np.sqrt(np.mean(np.sum(alias[sl]**2,axis=1)))),
            loaded_substeps=int(np.count_nonzero(c)),total_substeps=int(c.size),
            opposed_thumb_substeps=int(opposed.reshape(num,dec)[sl].sum()),
            mean_finger_residual_abs_rad=float(np.abs(arrays['finger_target_residual_rad'][sl]).mean()),
            actual_angular_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(orn_error[sl]**2,axis=1))))))
    summaries['overall_object_peak_center_lift_mm']=float(1000*(objpos[:,2]-objpos[0,2]).max())
    return arrays,dict(estimator='same nominal source frame/position target in all modes; diagnostic vref replaces only the three slider velocity targets',
      intervals=num,terminal=stop,velocity_caps_m_s=vmax.tolist(),effort_caps_N=effort.tolist(),
      wrist_kp=env.model.joint_target_ke.numpy()[pos].tolist(),wrist_kd=env.model.joint_target_kd.numpy()[pos].tolist(),windows=summaries)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("parquet","scored","out","warp-cache"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--modes",nargs="+",choices=('zero','recorded_actor','actor_without_translation','zero_plus_linear_vref'),
                   default=('zero','recorded_actor','zero_plus_linear_vref'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    wp.config.kernel_cache_dir=str(a.warp_cache);wp.init();wp.set_device('cpu')
    score=json.loads((a.scored/'report.json').read_bytes());ckpt=Path(score['checkpoint'])
    assert digest(ckpt)==score['checkpoint_sha256'] and digest(a.parquet)==score['parquet_sha256']
    with np.load(a.scored/'trace.npz') as t:actor=t['actor_action']
    results={}
    for mode in a.modes:
        arr,report=arm(mode,a.parquet,ckpt,min(46,len(actor)),actor)
        outfile=a.out/f'{mode}.npz';np.savez_compressed(outfile,**arr)
        report['arrays_sha256']=digest(outfile); results[mode]=report
    data=dict(scope='Development only. Original reference and saved hardware limits in a fresh CPU solver per condition; recorded actor actions replayed, not new policy inference. Frames numbered by source endpoint.',
              checkpoint_sha256=digest(ckpt),source_sha256=digest(a.parquet),saved_actor_trace_sha256=digest(a.scored/'trace.npz'),
              modes=results)
    (a.out/'report.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    for key,d in results.items():print(key,d['intervals'],d['windows']['acquisition'])

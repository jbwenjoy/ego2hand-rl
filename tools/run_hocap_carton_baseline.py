"""Run a source-world Wuji IK and contact-only MuJoCo baseline on the HO-Cap carton."""
import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.ego import hocap_carton_control as control
from supermimic_research.strict_json import write_json
import json


def run(args):
    start=time.perf_counter()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    report_path=args.source.resolve()
    receipt=json.loads(report_path.read_bytes())
    mesh=trimesh.load_mesh(verify_file_pin(receipt['geometry']['hull']),process=False)
    with np.load(verify_file_pin(receipt['source_poses']),allow_pickle=False) as src:
        frames=src['frames']; hand=src['world_right_hand21']; obj=src['world_from_carton']
    if (frames.ndim != 1 or frames.size < 2 or frames.dtype.kind not in 'iu'
            or int(frames[0]) < 0 or not np.all(np.diff(frames.astype(np.int64)) == 1)):
        raise ValueError('Expected a contiguous increasing integer source-frame window')
    if hand.shape != (len(frames), 21, 3) or obj.shape != (len(frames), 4, 4):
        raise ValueError('Source pose shapes must cover every frame')
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError('Positive finite nominal frame rate required')
    # Bound the physics step by the original 1/1200 s while exactly meeting frame boundaries.
    substeps = math.ceil(1200 / args.fps) if args.substeps is None else args.substeps
    if substeps < 1:
        raise ValueError('At least one physics substep required per control frame')
    normal=np.asarray(args.plane_normal,dtype=float);normal/=np.linalg.norm(normal)
    if normal[-1]<.8: raise ValueError('Expected source Z-up support')
    dt=1./(args.fps*substeps)
    times=np.arange(len(frames))/args.fps
    limits=dict(pos_gain=32.,rot_gain=35.,lin_damping=65.,ang_damping=1.1,
                vmax=1.8,wmax=8.,max_force=55.,max_torque=4.)
    model,wrist=control.create_scene(hand,obj[0],mesh,args.mass,args.mu,normal,args.table_z,dt)
    mjb=out/'scene.mjb'
    mujoco.mj_saveModel(model,str(mjb),None)
    model=mujoco.MjModel.from_binary_path(str(mjb))
    tip_targets=hand[:,control.TIPS].copy()
    ik0=time.perf_counter()
    fingers,iktips,nfev=control.solve_kinematics(model,wrist,hand,tip_targets)
    ik_elapsed=time.perf_counter()-ik0
    integ0=time.perf_counter()
    trace=control.replay(model,wrist,fingers,times,limits)
    integ_elapsed=time.perf_counter()-integ0
    replay_error=control.audit_commands(mujoco.MjModel.from_binary_path(str(mjb)),trace)
    trace.update(world_from_carton_source=obj, source_right_hand21=hand,target_wrist=wrist,
                 target_finger_q=fingers, ik_tips=iktips, ik_tip_targets=tip_targets,
                 ik_nfev=nfev,source_nominal_time_s=times,source_frame_ids=frames,
                 source_sequence_nominal_time_s=frames/args.fps)
    trace_path=out/'trace.npz'
    np.savez_compressed(trace_path,**trace)
    pred=trace['world_from_carton_sim']
    position_error=np.linalg.norm(pred[:,:3,3]-obj[:,:3,3],axis=1)*1000
    orientation_error=np.rad2deg((Rotation.from_matrix(pred[:,:3,:3]).inv()*Rotation.from_matrix(obj[:,:3,:3])).magnitude())
    ik_error=np.linalg.norm((iktips-hand[:,control.TIPS]),axis=2)*1000
    lift=(pred[:,:3,3]-pred[0,:3,3])@normal
    source_lift=(obj[:,:3,3]-obj[0,:3,3])@normal
    oid=model.body('carton').id
    result=dict(method='Geometric baseline, bilateral physical collisions, bounded wrench actuating wrist only',
                source_report=file_pin(report_path), sources=[file_pin(__file__),file_pin(control.__file__)],
                physics=mujoco.__version__, mjb=file_pin(mjb),trace=file_pin(trace_path),
                nominal_fps=args.fps, source_sensor_timestamps_available=False,periods=len(times)-1,
                source_frame_range=frames[[0,-1]].tolist(),playback_duration_s=float(times[-1]),
                physics_dt_s=dt,substeps_per_frame=substeps,world='HO-Cap tag1 coordinates, same transform for hand and object',
                table=dict(unit_normal=normal.tolist(),z_intercept_m=args.table_z,
                    initial_mesh_plane_gap_mm=float(1000*((mesh.vertices@obj[0,:3,:3].T+obj[0,:3,3]-[0,0,args.table_z])@normal).min())),
                gravity_m_s2=model.opt.gravity.tolist(),hand_mass_kg=float(model.body_subtreemass[model.body('right_palm_link').id]),
                object_hypotheses=dict(mass_kg=float(model.body_mass[oid]),inertia_kg_m2=model.body_inertia[oid].tolist(),
                    center_of_mass_in_body_m=model.body_ipos[oid].tolist(), hand_object_friction=args.mu,
                    table_object_friction=.6, measured=False, inertia_from_uniform_convex_solid=True),
                wrist=limits,ik='calibrated nonthumb palm registration, absolute source thumb/index/middle/ring/little tips, soft middle joint positions',
                ik_rms_mm=float(np.sqrt(np.mean(ik_error**2))),ik_max_mm=float(ik_error.max()),
                ik_elapsed_s=ik_elapsed,integration_elapsed_s=integ_elapsed,wall_elapsed_s=time.perf_counter()-start,
                object_position_rms_mm=float(np.sqrt(np.mean(position_error**2))),object_position_max_mm=float(position_error.max()),
                object_rotation_rms_deg=float(np.sqrt(np.mean(orientation_error**2))),object_rotation_max_deg=float(orientation_error.max()),
                object_peak_lift_mm=float(1000*lift.max()),source_peak_lift_mm=float(1000*source_lift.max()),
                contacted_frames=int(np.count_nonzero(trace['interval_max_hand_contacts'])),
                peak_carton_penetration_mm=float(1000*trace['interval_max_carton_penetration_m'].max()),
                peak_self_penetration_mm=float(1000*trace['interval_max_hand_self_penetration_m'].max()),
                maximum_joint_excursion_rad=float(trace['interval_joint_excursion_rad'].max()),
                peak_wrist_applied_force_N=float(trace['wrist_force_max_N'].max()),
                peak_wrist_applied_torque_Nm=float(trace['wrist_torque_max_Nm'].max()),
                external_object_wrench_or_reset_after_initialization=False,independent_substep_action_replay_max_state_error=replay_error,
                source_fidelity_qualified=False)
    write_json(out/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if isinstance(v,(float,int))},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fps',type=float,default=30);p.add_argument('--substeps',type=int,default=None,
                   help='Physics steps per frame. Defaults to enough steps for dt at most 1/1200 s.')
    p.add_argument('--mass',type=float,default=.15);p.add_argument('--mu',type=float,default=.7)
    p.add_argument('--table-z',type=float,default=0.);p.add_argument('--plane-normal',type=float,nargs=3,default=(0.,0.,1.))
    run(p.parse_args())

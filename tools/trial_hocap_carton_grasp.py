"""Paired source-world test of pad-aware correction with a physically free carton."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from supermimic_research.ego import hocap_carton_control as control
from supermimic_research.ego import hocap_carton_grasp as grasp
from supermimic_research.strict_json import write_json
from supermimic_research.artifacts import file_pin, verify_file_pin


def run(args):
    folder=args.out.resolve()
    folder.mkdir(exist_ok=False,parents=True)
    parent_path=args.baseline.resolve()
    parent=json.loads(parent_path.read_bytes())
    source_report=json.loads(verify_file_pin(parent['source_report']).read_bytes())
    mesh=trimesh.load_mesh(verify_file_pin(source_report['geometry']['hull']),process=False)
    with np.load(verify_file_pin(parent['trace']),allow_pickle=False) as a:
        hand=a['source_right_hand21'];obj=a['world_from_carton_source']
        wrist=a['target_wrist'];finger=a['target_finger_q'];times=a['source_nominal_time_s']
    normal=np.asarray(args.plane_normal,float);normal/=np.linalg.norm(normal)
    normal_same=np.allclose(normal,parent['table']['unit_normal'])
    if args.anchor_plane:
        world_mesh=mesh.vertices@obj[0,:3,:3].T+obj[0,:3,3]
        table_z=float(np.min(world_mesh@normal)/normal[2])
    else: table_z=parent['table']['z_intercept_m']
    if args.mass==parent['object_hypotheses']['mass_kg'] and normal_same and table_z==parent['table']['z_intercept_m']:
        model=mujoco.MjModel.from_binary_path(str(verify_file_pin(parent['mjb'])))
    else:
        model,wrist_2=control.create_scene(hand,obj[0],mesh,args.mass,parent['object_hypotheses']['hand_object_friction'],
                                           normal,table_z,parent['physics_dt_s'])
        np.testing.assert_allclose(wrist,wrist_2)
    mjb=folder/'scene.mjb'
    mujoco.mj_saveModel(model,str(mjb),None)
    if args.reference_q:
        with np.load(args.reference_q,allow_pickle=False) as ref:
            q=ref['target_finger_q'];gates=ref['source_pad_gate'];cost=ref['pad_ik_cost']
        meta={'fixed_exported_commands':file_pin(args.reference_q)}
    else:
        start=time.perf_counter()
        q,gates,cost=grasp.pad_corrected_ik(model,hand,obj,wrist,finger,mesh,args.preload_mm*.001)
        meta={'ik_wall_s':time.perf_counter()-start}
    trace=control.replay(model,wrist,q,times,parent['wrist'])
    error=control.audit_commands(mujoco.MjModel.from_binary_path(str(mjb)),trace)
    trace.update(world_from_carton_source=obj,source_right_hand21=hand,target_wrist=wrist,target_finger_q=q,
                 source_pad_gate=gates,pad_ik_cost=cost,source_nominal_time_s=times)
    path=folder/'trace.npz';np.savez_compressed(path,**trace)
    pred=trace['world_from_carton_sim']
    dist=1000*np.linalg.norm(pred[:,:3,3]-obj[:,:3,3],axis=1)
    angular=np.rad2deg((Rotation.from_matrix(pred[:,:3,:3]).inv()*Rotation.from_matrix(obj[:,:3,:3])).magnitude())
    lift=1000*((pred[:,:3,3]-pred[0,:3,3])@normal)
    true_lift=1000*((obj[:,:3,3]-obj[0,:3,3])@normal)
    result=dict(parent=file_pin(parent_path),source_report=parent['source_report'],sources=[file_pin(__file__),file_pin(grasp.__file__),file_pin(control.__file__)],
                trace=file_pin(path),mjb=file_pin(mjb),mass_kg=args.mass,preload_mm=args.preload_mm,
                table=dict(unit_normal=normal.tolist(),z_intercept_m=table_z,source_bottom_anchored=args.anchor_plane),
                wrist=parent['wrist'],**meta,nominal_fps=parent['nominal_fps'],substeps_per_frame=parent['substeps_per_frame'],
                source_sensor_timestamps_available=False,physics_dt_s=model.opt.timestep,
                object_peak_lift_mm=float(lift.max()),source_peak_lift_mm=float(true_lift.max()),
                object_position_rms_mm=float(np.sqrt(np.mean(dist**2))),object_position_max_mm=float(dist.max()),
                object_rotation_rms_deg=float(np.sqrt(np.mean(angular**2))),object_rotation_max_deg=float(angular.max()),
                object_final_position_error_mm=float(dist[-1]),contacted_frames=int(np.count_nonzero(trace['interval_max_hand_contacts'])),
                peak_carton_penetration_mm=float(1000*trace['interval_max_carton_penetration_m'].max()),
                peak_self_penetration_mm=float(1000*trace['interval_max_hand_self_penetration_m'].max()),
                maximum_joint_excursion_rad=float(trace['interval_joint_excursion_rad'].max()),independent_substep_action_replay_max_state_error=error,
                nominal_hypotheses_unmeasured=True,source_fidelity_qualified=False)
    write_json(folder/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if isinstance(v,(float,int))}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preload-mm',type=float,required=True);p.add_argument('--mass',type=float,default=.15)
    p.add_argument('--plane-normal',nargs=3,type=float,default=[0,0,1]);p.add_argument('--anchor-plane',action='store_true')
    p.add_argument('--reference-q',type=Path)
    run(p.parse_args())

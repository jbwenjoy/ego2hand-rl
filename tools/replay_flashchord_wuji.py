"""Load a real Wuji motion_v1 Parquet with official FlashCHORD, check FK and replay."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
from flash_chord.runtime.sim import SimConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from supermimic_research.runtime.flash_wuji import WujiHand2,WujiActionConfig


def main(args):
    out=args.out;out.mkdir(exist_ok=False,parents=True)
    wp.config.kernel_cache_dir=str(args.warp_cache)
    wp.init(); wp.set_device("cuda:0")
    before=time.perf_counter()
    setup=setup_scene(parquet=args.parquet,control_fps=30.,motion_speed=1.,
            source_frame_playback=True,embodiment=WujiHand2(args.asset),
            collision=CollisionPolicy(ground=False,hand_self_collision=True,robot_scope="all",
                                      robot_object_collision=True,robot_support_collision=True),
            world_count=1,include_support=True,decompose_objects=False,contact_friction=.7)
    scene=setup.scene;ref=setup.reference;h=scene.robot_reference.hand("right")
    names=tuple(ref.robot_frame_names());fp=ref.robot_frame_pos_w();fq=ref.robot_frame_quat_w()
    err={"palm":float(np.max(np.linalg.norm(h.wrist_pos_w-ref.robot_root_pos_w(),axis=1)))}
    eids=[names.index(f"right_finger{i}_link4") for i in range(1,6)]
    err["distal"]=float(np.max(np.linalg.norm(fp[:,eids]-h.dp_pos_w,axis=2)))
    tips=WujiHand2(args.asset).meta["tips"]
    tids=[names.index(t["parent"]) for t in tips]
    sensor_quats=Rotation.from_quat(fq[:,tids,:].reshape(-1,4),scalar_first=True)
    offsets=np.tile([t["pos"] for t in tips],(ref.num_frames,1))
    truth=fp[:,tids,:]+sensor_quats.apply(offsets).reshape(ref.num_frames,5,3)
    err["tip"]=float(np.max(np.linalg.norm(truth-h.fingertip_pos_w,axis=2)))
    if max(err.values())>.0001: raise ValueError(f"Wuji FK/reference mismatch: {err}")
    sim=SimConfig(fps=30.,substeps=args.substeps,njmax=512,nconmax=256,iterations=100,ls_iterations=50,
                  integrator="implicitfast",solver="newton",cone="pyramidal",use_mujoco_contacts=True)
    diagnostics=dict(frames=ref.num_frames,sides=list(scene.layout.sides),joint_q=scene.layout.num_joint_q,
                     robot_dofs=scene.layout.num_joint_dof,collision_groups=len(scene.layout.hand("right").link_geometry),
                     max_fk_error_m=err,parquet_sha256=hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
                     support=str(setup.support_usda),gravity_m_s2=scene.model.gravity.numpy().tolist(),
                     dt=sim.sim_dt,substeps=sim.substeps,nominal_fps=sim.fps,geometry_friction=.7,
                     xml_sha256=hashlib.sha256((args.asset/"wuji_right.xml").read_bytes()).hexdigest(),
                     source_uses_20_named_wuji_joints=True)
    (out/"binding.json").write_text(json.dumps(diagnostics,indent=2))
    print("BOUND",diagnostics,flush=True)
    if args.bind_only:
        return
    runner=ReplayRunner(scene,ref,ReplayConfig(sim=sim,action=WujiActionConfig(),voc_scale=0.),device="cuda:0")
    runner.reset(0)
    wp.synchronize_device("cuda:0");built=time.perf_counter()
    objid=int(runner.command.body_ids_w.numpy().reshape(-1)[0])
    print("READY",objid,"action_dim",runner.action.action_dim,flush=True)
    initial=runner.state_0.body_q.numpy()
    body=[initial];joints=[runner.state_0.joint_q.numpy()]
    durations=[]
    count=min(args.intervals,ref.num_frames-1)
    for i in range(count):
        tick=time.perf_counter()
        runner.step(i); wp.synchronize_device("cuda:0")
        durations.append(time.perf_counter()-tick)
        b=runner.state_0.body_q.numpy();q=runner.state_0.joint_q.numpy()
        if not np.isfinite(b).all() or not np.isfinite(q).all():
            raise ValueError(f"Nonfinite physical state after interval {i}")
        if i in (0,3,39,89):print("FRAME",i+1,float(runner.sim_time),durations[-1],flush=True)
        body.append(b);joints.append(q)
    body=np.asarray(body);joints=np.asarray(joints)
    poses=body[:,objid];positions=ref.object_body_pos_w()[:len(poses),0]
    rotations=Rotation.from_quat(ref.object_body_quat_w()[:len(poses),0],scalar_first=True)
    angles=np.rad2deg((Rotation.from_quat(poses[:,3:]).inv()*rotations).magnitude())
    delta=np.linalg.norm(poses[:,:3]-positions,axis=1)
    np.savez_compressed(out/"replay.npz",world_bodies_xyzw=body,joint_q=joints,
                        object_index=objid,time=np.arange(count+1)/30.,source_object_positions=positions,
                        source_object_quats_wxyz=ref.object_body_quat_w()[:len(poses),0])
    diagnostics.update(executed_intervals=count,action_dim=runner.action.action_dim,voc_scale=0.,
                       world_count=1,object_mass=float(scene.model.body_mass.numpy()[objid]),object_body=str(scene.model.body_label[objid]),
                       side_lengths=list(scene.layout.hand("right").finger_joint_names),
                       object_peak_lift_mm=float(1000*(poses[:,2]-poses[0,2]).max()),
                       object_position_rms_mm=float(1000*np.sqrt(np.mean(delta**2))),
                       object_rotation_rms_deg=float(np.sqrt(np.mean(angles**2))),
                       build_s=built-before,first_step_s=durations[0],warm_step_p50_s=float(np.median(durations[1:])),
                       all_finite=True,trained=False)
    (out/"report.json").write_text(json.dumps(diagnostics,indent=2,allow_nan=False))
    print("COMPLETED",diagnostics,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("asset","parquet","out","warp-cache"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--substeps",type=int,default=40);p.add_argument("--intervals",type=int,default=140)
    p.add_argument("--bind-only",action="store_true")
    main(p.parse_args())

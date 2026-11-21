"""Matched CPU action pulses for testing local gradients of a stored Wuji world model."""

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import time
from pathlib import Path

import jax.numpy as jnp
import newton
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import warp as wp

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.lifecycle.curriculum import FixedCurriculum
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config, policy_schema
from flash_chord.training.environment import WarpRLEnv
from supermimic_research.artifacts import file_pin
from supermimic_research.runtime.flash_multiref import scan_extents_m
from supermimic_research.runtime.flash_quality import SolverRecorder


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf8")


def initialize(scene, frame, native, cfg, stage, extent):
    e = RLEnv(scene.scene, scene.reference, instantiate_typed(cfg.env, RLEnvConfig))
    if getattr(e.solver, "wuji_conditioning", {}).get("scheme") != "movable_dof_mean_v1":
        raise ValueError("Requires the force-validated conditioned solver")
    if not e.device.is_cpu or e.config.auto_reset or e.config.voc.max_force or e.config.voc.max_torque:
        raise ValueError("The experiment requires CPU-only, unassisted, no-auto-reset simulation")
    bridge = WarpRLEnv(e)
    bridge.set_curriculum(stage)
    e.reset(frame_id=frame)
    obj=e.scene.objects[0]
    prepare_contact=np.zeros(6)
    if native is not None:
        hand=e.scene.layout.hand("right")
        robot_pose=native["sim_wrist"][frame]
        box_pose=native["world_from_carton_sim"][frame]
        q=e.state_0.joint_q.numpy()
        q[list(hand.wrist_pos_q_ids)]=robot_pose[:3,3]
        q[hand.wrist_orient_q_id:hand.wrist_orient_q_id+4]=Rotation.from_matrix(robot_pose[:3,:3]).as_quat()
        q[list(hand.finger_q_ids)]=native["sim_qpos"][frame,7:27]
        q[list(obj.root.free_q_ids)]=np.r_[box_pose[:3,3],Rotation.from_matrix(box_pose[:3,:3]).as_quat()]
        e.state_0.joint_q.assign(q)
        e.state_0.joint_qd.zero_()
        newton.eval_fk(e.model,e.state_0.joint_q,e.state_0.joint_qd,e.state_0)
        # One identical zero-residual physical substep establishes an observed contact
        # and solver warm start at this reference frame. Neither source clock nor sample
        # horizon advances until the subsequent measured intervention.
        rec=SolverRecorder(e)
        try:
            e.action.process(wp.zeros(e.action_input.shape,dtype=wp.float32,device=e.device),e.timestep,e.state_0)
            e.command.gather(e.timestep)
            e.action.prepare_control(e.control)
            e.state_0.clear_forces()
            e.action.apply_control(e.state_0,e.control)
            e.solver.step(e.state_0,e.state_1,e.control,e.contacts,e.sim_dt)
            if e.joint_actuator_limits is not None:
                e.joint_actuator_limits.enforce_velocity(e.state_1)
                newton.eval_fk(e.model,e.state_1.joint_q,e.state_1.joint_qd,e.state_1)
            e.state_0,e.state_1=e.state_1,e.state_0
            e.solver.update_contacts(e.contacts,e.state_0)
            e.contact.update(e.model,e.state_0,e.contacts)
            prepare_contact=np.array(rec.samples[0])
        finally:
            rec.close()
    if np.any(e.object_control_scale.numpy()) or int(e.timestep.numpy()[0])!=frame:
        raise ValueError("Unexpected starting reference frame or assistance")
    obs=np.r_[e.observation_strategy.compute(e.state_0).numpy().reshape(-1),extent]
    e._compute_critic_context(e.state_0,e.critic_context)
    state=dict(obs0=obs,ctx0=e.critic_context.numpy().reshape(-1).copy(),q0=e.state_0.joint_q.numpy().copy(),
               qd0=e.state_0.joint_qd.numpy().copy(),body0=e.state_0.body_q.numpy().copy(),
               body_velocity0=e.state_0.body_qd.numpy().copy(),object0=e.state_0.body_q.numpy()[obj.bodies[0].body_id].copy(),
               initial_contact=prepare_contact)
    for key in ("qpos","qvel","qacc","qacc_warmstart","time"):
        arr=getattr(e.solver.mjw_data,key,None)
        if arr is not None: state["solver_"+key]=arr.numpy().copy()
    if len(obs)!=276 or len(state["ctx0"])!=10:raise ValueError("Unexpected learned-model input dimensions")
    return e,bridge,state


def branch(setup, frame, snapshot, cfg, stage, extent, axis, value):
    e,b,initial=initialize(setup,frame,snapshot,cfg,stage,extent)
    weights=np.asarray(stage.weights_for(b.objective_term_names),dtype=np.float32)
    recorder=SolverRecorder(e)
    rows=[];actions=np.zeros((3,b.action_dim),dtype=np.float32)
    if axis>=0:actions[0,axis]=value
    try:
        for h,action in enumerate(actions):
            ref=int(e.timestep.numpy()[0])
            if np.any(e.object_control_scale.numpy()):raise ValueError("Object assistance is active")
            step=b.step(jnp.asarray(action[None,:]))
            # Copy actual pre-reset terminal outputs immediately; underlying Warp views are mutable.
            terms=np.asarray(step.objective_terms).reshape(-1).copy()
            reward=float(np.asarray(step.reward).reshape(-1)[0])
            np.testing.assert_allclose(reward,terms@weights*e.frame_dt,atol=2e-5,rtol=1e-5)
            s=e.config.sim.substeps
            contacts=np.array(recorder.samples[-s:])
            poses=e.state_0.body_q.numpy().copy()
            row=dict(valid=True,source_frame=ref,reward=reward,objective_terms=terms,
                     terminated=bool(np.asarray(step.terminated)[0]),truncated=bool(np.asarray(step.truncated)[0]),
                     next_obs=np.r_[np.asarray(step.terminal_observation).ravel().copy(),extent],
                     next_ctx=np.asarray(step.terminal_critic_context).reshape(-1).copy(),
                     next_object=poses[recorder.obj_id],next_body=poses,next_body_velocity=e.state_0.body_qd.numpy().copy(),
                     next_q=e.state_0.joint_q.numpy().copy(),next_qd=e.state_0.joint_qd.numpy().copy(),
                     substep_contact=contacts,object_control_scale=e.object_control_scale.numpy().copy(),
                     first_substep_joint_target=np.asarray(recorder.commands["joint_target_pos"][-s]))
            rows.append(row)
            if row["terminated"] or row["truncated"]:break
    finally:recorder.close()
    full={}
    for key in rows[0]:
        observed=[np.asarray(r[key]) for r in rows]
        full[key]=np.stack(observed+[np.zeros_like(observed[0]) for _ in range(3-len(rows))])
    assert all(np.isfinite(v).all() for v in (*initial.values(),*full.values()))
    extra=dict(weights=weights.tolist(),objective_names=list(b.objective_term_names),ctx_names=list(b.critic_context_names),
               schema=policy_schema(e.action,e.observation_strategy),condition=e.solver.wuji_conditioning,
               dt=e.frame_dt,physics_dt=e.sim_dt,finger_joint_order=list(e.scene.layout.hand("right").finger_joint_names))
    return initial,dict(**full,actions=actions),extra


def main(args):
    start=time.monotonic()
    if wp.get_cuda_device_count()!=0:raise ValueError("Run with CUDA_VISIBLE_DEVICES empty and JAX_PLATFORMS=cpu")
    wp.set_device("cpu")
    wp.config.kernel_cache_dir=str(args.cache)
    meta=read_checkpoint_metadata(args.checkpoint)
    config=checkpoint_config(meta)
    original_termination=config["env"]["termination"]
    apply_evaluation_composition(config,world_count=1,reset_mode="explicit")
    config["env"]["termination"]=original_termination
    cfg=OmegaConf.create(config)
    cfg.env.capture_terminal_state=True
    curriculum=instantiate_typed(cfg.curriculum,FixedCurriculum)
    stage=replace(curriculum.stages[int(meta["curriculum_stage"])],voc_scale=0.)
    setup=setup_scene(parquet=args.parquet,control_fps=30.,motion_speed=1.,source_frame_playback=True,
        embodiment=instantiate_typed(cfg.embodiment,Embodiment), collision=instantiate_typed(cfg.collision,CollisionPolicy),
        world_count=1,include_support=cfg.scene.support,decompose_objects=cfg.scene.decompose_objects,
        object_scale_min=1.,object_scale_max=1.,contact_friction=cfg.scene.contact_friction,
        object_free_joint_damping=cfg.scene.object_free_joint_damping)
    if setup.reference.metadata.sequence_id!="subject1_G04_2":raise ValueError("Only the designated development example is approved here")
    extent=scan_extents_m(setup.parquet_path)
    with np.load(args.native_trace,allow_pickle=False) as a:
        snapshot={k:a[k] for k in ("sim_wrist","sim_qpos","world_from_carton_sim")}
    probes=[(-1,0.),(-1,0.)]+[(axis,val) for axis in (0,2,4,8,12,16) for val in (-.05,.05)]
    conditions=[("reference_separated",0,None),("reference_grasp_phase",38,None),
                ("pose_seeded_grasp",38,snapshot),("pose_seeded_hold",65,snapshot)]
    args.out.mkdir(parents=True,exist_ok=False)
    stages=[]
    for label,frame,seed in conditions:
        common=None;rows=[];t=time.monotonic()
        for axis,val in probes:
            initial,data,extra=branch(setup,frame,seed,cfg,stage,extent,axis,val)
            if common is None: common=initial
            else:
                for k in common:np.testing.assert_array_equal(initial[k],common[k],err_msg=f"Unmatched initial state {label}/{axis}/{val}/{k}")
            rows.append(data)
            gc.collect()
        arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]}
        duplicate={k:float(np.max(np.abs(v[0].astype(float)-v[1].astype(float)))) for k,v in arrays.items()}
        out=args.out/(label+".npz")
        np.savez_compressed(out,**common,**arrays,weights=extra["weights"],dt=extra["dt"],
             axis=np.array([a for a,_ in probes]),pulse=np.array([v for _,v in probes]))
        stages.append(dict(label=label,frame=frame,prepared_substeps=1 if seed is not None else 0,
                    pose_source="stored offline native grasp; velocities cleared before one physical substep" if seed is not None else "direct explicit reference reset",
                    initial_contact=common["initial_contact"].tolist(),zero_control_valid=arrays["valid"][0].tolist(),
                    zero_control_peak_hand_N=arrays["substep_contact"][0,:,: ,0].max(axis=1).tolist(),
                    duplicate_zero_max_difference=duplicate,matched_initial_fields=list(common),seconds=time.monotonic()-t,data=file_pin(out)))
        print(label,frame,stages[-1]["zero_control_peak_hand_N"],duplicate["next_obs"],flush=True)
    write(args.out/"report.json",dict(sequence="subject1_G04_2",checkpoint=file_pin(args.checkpoint),parquet=file_pin(args.parquet),
        native_seed_trace=file_pin(args.native_trace),tool=file_pin(__file__),stages=stages,physics=extra,epsilon=.05,horizon=3,
        order="Within each stage: two zero controls then negative/positive pulse per axis. All remaining controls zero. Array axes = treatment, control, feature. Prefix valid mask excludes unexecuted padded zeros after termination.",
        contact_columns=["hand normal N","support normal N","hand-contact count","object penetration m","self penetration m","joint excursion rad"],
        target="All next_obs and next_ctx are pre-reset terminal states. Zero duplicate floor plus floating-point resolution should bound derivative claims; a physically zero response has no identifiable directional sign.",
        wall_s=time.monotonic()-start))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("checkpoint","parquet","native-trace","cache","out"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();main(a)

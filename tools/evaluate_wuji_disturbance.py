"""Separate, paired disturbance-response evaluation of an already scored Wuji checkpoint."""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import warp as wp

from evaluate_flash_wuji_quality import (
    RLEnv, RLEnvConfig, Embodiment, CollisionPolicy, TrainingConfig, WarpRLEnv,
    apply_evaluation_composition, checkpoint_config, dump, file_hash, instantiate_typed,
    load_policy_for_inference, read_checkpoint_metadata, resolve_jax_device, scan_extents_m,
    setup_scene, source_truth, validation_status,
)
from supermimic_research.runtime.flash_disturbance import closed_run,describe,onset_frame,pulse_series
from supermimic_research.runtime.flash_quality import COMMANDS,repeat_control_trajectory


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    primary=json.loads(args.primary.read_bytes());protocol=json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    if primary["checkpoint_sha256"]!=file_hash(args.checkpoint) or primary["parquet_sha256"]!=file_hash(args.parquet):
        raise ValueError("Stress study must use the already scored checkpoint and reference")
    if protocol['axis_in_world']!=[1,0,0] or protocol['directions']!=[-1,1] or protocol['pulse_intervals']!=2:
        raise ValueError("Unexpected disturbance experiment configuration")
    truth,_,frames=source_truth(args.parquet);onset=onset_frame(truth,protocol)
    wp.config.kernel_cache_dir=str(args.warp_cache);wp.init();wp.set_device('cuda:0')
    metadata=read_checkpoint_metadata(args.checkpoint);config=checkpoint_config(metadata)
    oldterm=deepcopy(config['env']['termination'])
    shared='reference_shards' in config
    apply_evaluation_composition(config,world_count=1,reset_mode='explicit')
    config['env']['termination']=oldterm;cfg=OmegaConf.create(config)
    if cfg.task.motion_speed!=1 or cfg.sim.fps!=30: raise ValueError("Require original 30 Hz motion")
    training=instantiate_typed(cfg.training,TrainingConfig)
    def fresh():
        setup=setup_scene(parquet=args.parquet,control_fps=cfg.sim.fps,motion_speed=1.,source_frame_playback=True,
                embodiment=instantiate_typed(cfg.embodiment,Embodiment),collision=instantiate_typed(cfg.collision,CollisionPolicy),
                world_count=1,include_support=cfg.scene.support,decompose_objects=cfg.scene.decompose_objects,
                object_scale_min=cfg.scene.object_scale_min,object_scale_max=cfg.scene.object_scale_max,
                contact_friction=cfg.scene.contact_friction,object_free_joint_damping=cfg.scene.object_free_joint_damping)
        np.testing.assert_allclose(setup.reference.object_body_pos_w()[:,0],truth[:,:3,3],atol=3e-7,rtol=0)
        env=RLEnv(setup.scene,setup.reference,instantiate_typed(cfg.env,RLEnvConfig))
        bridge=WarpRLEnv(env)
        if training.curriculum is not None:
            bridge.set_curriculum(replace(training.curriculum.stages[int(metadata['curriculum_stage'])],voc_scale=0.))
        env.reset(frame_id=0)
        if env.config.auto_reset or env.config.voc.max_force or env.config.voc.max_torque or env.object_control_scale.numpy().any():
            raise ValueError("Assistance or reset invalidates pulse study")
        return env
    first=fresh();seq=first.reference.metadata.sequence_id
    physics=validation_status(cfg,first,seq)
    if not physics["validated"]:
        raise ValueError(f"Stress study requires validated corrected physics: {physics}")
    extent=scan_extents_m(args.parquet)[None,:] if shared else None
    def observation(env):
        obs=env.observation.numpy().reshape(1,-1)
        return jnp.asarray(np.concatenate([obs,extent],axis=1) if shared else obs)
    actor,act,key,_=load_policy_for_inference(str(args.checkpoint),training,observation(first),
              first.action.action_dim,deterministic=True,device=resolve_jax_device(training.device))
    def policy():
        state_key=key
        def command(env,_):
            nonlocal state_key
            result,state_key=act(actor,observation(env),state_key)
            return wp.array(np.asarray(result,dtype=np.float32).reshape(env.action_input.shape),dtype=wp.float32,device=env.device)
        return command
    nominal,nom_stop=closed_run(first,policy(),len(frames)-1,onset,0,protocol,bypass=True)
    hand=first.scene.layout.hand("right")
    hid=hand.palm_frame.body_id;oid=first.scene.objects[0].bodies[0].body_id;dof=hand.wrist_pos_dof_ids[0]
    refx=first.scene.robot_reference.joint_target[:,dof]; sub=first.config.sim.substeps
    nominal_path=args.out/"unperturbed_policy.npz"
    np.savez_compressed(nominal_path,**nominal)
    open0,_,open_nom,open_stop=repeat_control_trajectory(fresh,nominal,nom_stop,len(frames))
    np.savez_compressed(args.out/"unperturbed_open_loop.npz",**open_nom)
    rows=[]
    for direction in protocol["directions"]:
        closed,cstop=closed_run(fresh(),policy(),len(frames)-1,onset,direction,protocol)
        commands,deltas=pulse_series(nominal["command_joint_target_pos"],refx,dof,first.config.action.wrist_pos_clip,
                                     onset,direction,protocol,sub)
        frozen=dict(nominal,command_joint_target_pos=commands)
        _,_,opened,ostop=repeat_control_trajectory(fresh,frozen,nom_stop,len(frames))
        opened["applied_pulse_m"]=deltas[:len(opened["command_joint_f"])]
        for mode,tr,base,stop in (("closed",closed,nominal,cstop),("open",opened,open_nom,ostop)):
            record,series=describe(tr,base,truth,frames,hid,oid,onset,sub,cfg.sim.fps,protocol,stop)
            label=f"{mode}_{'plus' if direction>0 else 'minus'}";path=args.out/f"{label}.npz"
            np.savez_compressed(path,**tr,**series)
            rows.append(dict(mode=mode,direction=direction,trace_name=path.name,trace_sha256=file_hash(path),**record))
    summary={}
    for mode in ("closed","open"):
        group=[r for r in rows if r["mode"]==mode]
        summary[mode]=dict(assigned=len(group),exposed=sum(r["delivered_pulse_substeps"]>0 for r in group),
          full_exposure=sum(r["fully_exposed"] for r in group),ended_early=sum(r["terminal"]["failed"] for r in group),
          counts={name:sum(r["recovery_state"]==name for r in group) for name in sorted({r["recovery_state"] for r in group})})
    dump(args.out/"report.json",dict(protocol=args.protocol.name,protocol_sha256=file_hash(args.protocol),
        sequence=seq,frames=[int(frames[0]),int(frames[-1])],onset_frame=int(frames[onset]) if onset is not None else None,
        source_peak_height_m=float((truth[:,2,3]-truth[0,2,3]).max()),checkpoint=str(args.checkpoint),
        primary_sha256=file_hash(args.primary),checkpoint_sha256=file_hash(args.checkpoint),parquet_sha256=file_hash(args.parquet),
        primary_exact_gates=primary["passed_numeric_development_gates"],primary_physical_repeat=primary.get("physical_repeat",{}).get("status"),
        unperturbed_policy_terminal=nom_stop,unperturbed_open_terminal=open_stop,source_pose_unmodified=True,
        object_assistance=0,physics=physics,assigned_summary=summary,runs=rows))
    print(json.dumps(summary))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","parquet","primary","protocol","out","warp-cache"):p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())

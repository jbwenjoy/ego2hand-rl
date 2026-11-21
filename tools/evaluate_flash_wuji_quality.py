"""Evaluate single or shared Wuji FlashSAC actors under the frozen HO-Cap quality protocol."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import warp as wp

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference
from flash_chord.training.flash_sac.runner import resolve_jax_device

from supermimic_research.runtime.flash_multiref import scan_extents_m
from supermimic_research.runtime.flash_quality import (
    SolverRecorder,world_tips,repeat_control_trajectory,pairwise_trajectory_differences)


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf8")


def file_hash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def resolved_pin(pin):
    raw=pin["path"]
    if raw.startswith("D:"): raw="/mnt/d/"+raw[3:].replace("\\","/")
    path=Path(raw)
    if file_hash(path)!=pin["sha256"] or path.stat().st_size!=pin["bytes"]:
        raise ValueError(f"Invalid original source pin {path}")
    return path


def source_truth(parquet):
    export=json.loads((parquet.parent/"export.json").read_bytes())
    receipt=json.loads(resolved_pin(export["raw_source"]).read_bytes())
    path=resolved_pin(receipt["source_poses"])
    with np.load(path,allow_pickle=False) as a:
        return a["world_from_carton"],a["world_right_hand21"][:,[4,8,12,16,20]],a["frames"]


def rms(x): return float(np.sqrt(np.mean(np.square(x))))


def validation_status(cfg,env,sequence):
    repo=Path(__file__).resolve().parents[1]
    manifest=repo/"configs/flashchord_wuji_physics_v2.json"
    if not getattr(env.solver,"wuji_conditioning",None):
        return dict(validated=False,reason="legacy solver convergence scale has unvalidated load",manifest=None)
    data=json.loads(manifest.read_bytes())
    for pin in data["code"]+data["evidence"]:
        if file_hash(repo/pin["path"])!=pin["sha256"]:
            return dict(validated=False,reason="physics implementation or validation evidence changed",manifest=str(manifest))
    for field in data["overrides"]:
        path,value=field.split("=",1)
        actual=OmegaConf.select(cfg,path)
        equal=(float(actual)==float(value) if type(actual) in (float,int) else str(actual).lower()==value.lower())
        if not equal:
            return dict(validated=False,reason=f"changed physics setting {path}: {actual}",manifest=str(manifest))
    key="flashchord_rest_drop_cuda_v1" if sequence=="subject1_G04_2" else f"flashchord_rest_drop_cuda_{sequence}_v1"
    witnesses=data["calibration"].get(key)
    mass=float(env.model.body_mass.numpy()[env.scene.objects[0].bodies[0].body_id])
    value=env.solver.wuji_conditioning["conditioned"]
    ok=bool(data["validated"] and witnesses and all(abs(x["kg"]-mass)<1e-6 and
                abs(x["meaninertia"]-value)<1e-6 for x in witnesses))
    return dict(validated=ok,reason="verified passive and dynamic evidence" if ok else "no matching validated source fixture",
                manifest=str(manifest),sha256=file_hash(manifest),solver_stat=value)


def phase_limits(lift,protocol,sequence):
    if sequence=="subject1_G04_2": return protocol["phase_end_inclusive"]
    peak=float(lift.max())
    if peak<=0: return {"sequence":len(lift)-1}
    moving=np.flatnonzero(lift>=.10*peak); high=np.flatnonzero(lift>=.85*peak)
    return dict(approach=int(moving[0])-1,lift=int(high[0])-1,hold=int(high[-1]),lower=int(moving[-1]),place_release=len(lift)-1)


def score(trace,truth,human,frames,env,rec,protocol,sequence,stop,replay_error):
    n=len(trace["body_q"]);T=len(frames);full=n==T and not stop["failed"]
    ref=truth[:n]; p=trace["body_q"][:,rec.obj_id]
    pos=np.linalg.norm(p[:,:3]-ref[:,:3,3],axis=1)*1000.
    angle=np.rad2deg((Rotation.from_quat(p[:,3:]).inv()*Rotation.from_matrix(ref[:,:3,:3])).magnitude())
    static_pos=1000*np.linalg.norm(truth[:,:3,3]-truth[0,:3,3],axis=1)
    static_rot=np.rad2deg((Rotation.from_matrix(truth[0,:3,:3]).inv()*Rotation.from_matrix(truth[:,:3,:3])).magnitude())
    gravity=env.model.gravity.numpy()[0]; up=-gravity/np.linalg.norm(gravity)
    lift=(p[:,:3]-p[0,:3])@up*1000.;source_lift=(truth[:,:3,3]-truth[0,:3,3])@up*1000.
    samples=trace["substep_contacts"].reshape(n-1,env.config.sim.substeps,6)
    weight=float(env.model.body_mass.numpy()[rec.obj_id]*np.linalg.norm(gravity))
    loads=samples[:,:,:2].mean(axis=1)/weight;air=source_lift[1:]>protocol["source_lift_for_airborne_mm"]
    supported=(loads[:,1]<protocol["maximum_table_mean_force_in_air_bodyweights"])&(loads[:,0]>=protocol["minimum_hand_mean_force_in_air_bodyweights"])
    supported_count=int(np.count_nonzero(air[:n-1]&supported))
    air_fraction=supported_count/int(air.sum()) if air.any() else None
    endn=protocol["release_last_n_intervals"]; end=full and n-1>=endn
    release_load=float(loads[-endn:,1].min()) if end else None
    release_contacts=int(samples[-endn:,:,2].max()) if end else None
    tip_err=rms(np.linalg.norm(world_tips(trace["body_q"],rec.hand)-human[:n],axis=2)*1000)
    bounds=phase_limits(source_lift,protocol,sequence);start=0;phases={}
    for label,last in bounds.items():
        endidx=min(last+1,n)
        phases[label]=dict(source_frames=[int(frames[start]),int(frames[last])],observed_frames=max(0,endidx-start),
                           reference_frames=last-start+1,position_rms_mm=rms(pos[start:endidx]) if endidx>start else None,
                           rotation_rms_deg=rms(angle[start:endidx]) if endidx>start else None) if last>=start else None
        start=last+1
    metrics=dict(prefix_position_rms_mm=rms(pos),prefix_position_peak_mm=float(pos.max()),prefix_rotation_rms_deg=rms(angle),
                 full_position_rms_mm=rms(pos) if full else None,full_rotation_rms_deg=rms(angle) if full else None,
                 prefix_rotation_peak_deg=float(angle.max()),prefix_fingertip_rms_mm=tip_err,endpoint_error_mm=float(pos[-1]) if full else None,
                 peak_observed_lift_mm=float(lift.max()),peak_source_lift_mm=float(source_lift.max()),airborne_source_intervals=int(air.sum()),
                 airborne_observed=int(air[:n-1].sum()),airborne_supported=supported_count,supported_whole_air_fraction=air_fraction,
                 release_min_bodyweight=release_load,release_max_hand_contacts=release_contacts,
                 max_carton_penetration_mm=float(samples[:,:,3].max()*1000),max_self_penetration_mm=float(samples[:,:,4].max()*1000),
                 max_joint_excursion_deg=float(np.rad2deg(samples[:,:,5].max())),independent_command_replay_max_state_error=float(replay_error),
                 stationary_full_position_rms_mm=rms(static_pos),stationary_full_rotation_rms_deg=rms(static_rot),
                 stationary_prefix_position_rms_mm=rms(static_pos[:n]),stationary_prefix_rotation_rms_deg=rms(static_rot[:n]))
    for key, violations in (("peak_position",pos>protocol["maximum_position_error_mm"]),
                            ("peak_rotation",angle>protocol["maximum_rotation_error_deg"]),
                            ("frozen_joint_excursion",np.r_[False,np.rad2deg(samples[:,:,5].max(axis=1))>protocol["maximum_joint_excursion_deg"]])):
        indices=np.flatnonzero(violations)
        metrics[f"first_{key}_failure_source_frame"]=int(frames[indices[0]]) if indices.size else None
    # A missing tail cannot pass a trajectory-level gate. Keep prefix measurements for diagnosis.
    g=dict(complete=full,position=full and rms(pos)<=protocol["maximum_position_rms_mm"] and pos.max()<=protocol["maximum_position_error_mm"],
           rotation=full and rms(angle)<=protocol["maximum_rotation_rms_deg"] and angle.max()<=protocol["maximum_rotation_error_deg"],
           beats_fixed_pose=full and rms(pos)<rms(static_pos) and rms(angle)<rms(static_rot),endpoint=full and pos[-1]<=protocol["maximum_endpoint_error_mm"],
           lift=full and abs(lift.max()-source_lift.max())<=protocol["maximum_peak_lift_error_mm"],
           hand_support_in_air=full and air_fraction is not None and air_fraction>=protocol["minimum_supported_airborne_fraction"],
           release=end and release_load>=protocol["minimum_final_table_support_bodyweights"] and release_contacts==0,
           penetration=full and metrics["max_carton_penetration_mm"]<=protocol["maximum_carton_penetration_mm"] and metrics["max_self_penetration_mm"]<=protocol["maximum_self_penetration_mm"],
           joint_limits=full and metrics["max_joint_excursion_deg"]<=protocol["maximum_joint_excursion_deg"],
           hand_tracking=full and tip_err<=protocol["maximum_fingertip_rms_mm"],recorded_command_replay=replay_error<=1e-9)
    return metrics,phases,{k:bool(v) for k,v in g.items()}


def evaluate(args):
    args.out.mkdir(parents=True,exist_ok=False)
    wp.config.kernel_cache_dir=str(args.warp_cache);wp.init();wp.set_device("cuda:0")
    started=time.perf_counter()
    meta=read_checkpoint_metadata(args.checkpoint); cfg=checkpoint_config(meta)
    oldterm=deepcopy(cfg["env"]["termination"])
    shared="reference_shards" in cfg
    apply_evaluation_composition(cfg,world_count=1,reset_mode="explicit")
    cfg["env"]["termination"]=oldterm
    if cfg["task"]["motion_speed"]!=1 or cfg["env"]["sim"]["fps"]!=30:
        raise ValueError("Strict evaluator needs original-speed 30 Hz source")
    cfg=OmegaConf.create(cfg)
    env_config=instantiate_typed(cfg.env,RLEnvConfig);training=instantiate_typed(cfg.training,TrainingConfig)
    def setup_copy():
        return setup_scene(parquet=args.parquet,control_fps=env_config.sim.fps,motion_speed=1.,source_frame_playback=True,
           embodiment=instantiate_typed(cfg.embodiment,Embodiment),collision=instantiate_typed(cfg.collision,CollisionPolicy),
           world_count=1,include_support=cfg.scene.support,decompose_objects=cfg.scene.decompose_objects,
           object_scale_min=cfg.scene.object_scale_min,object_scale_max=cfg.scene.object_scale_max,
           contact_friction=cfg.scene.contact_friction,object_free_joint_damping=cfg.scene.object_free_joint_damping)
    setup=setup_copy()
    truth,human,frames=source_truth(args.parquet)
    np.testing.assert_array_equal(np.diff(frames),np.ones(len(frames)-1))
    np.testing.assert_allclose(setup.reference.object_body_pos_w()[:,0],truth[:,:3,3],atol=3e-7,rtol=0)
    np.testing.assert_allclose(Rotation.from_quat(setup.reference.object_body_quat_w()[:,0],scalar_first=True).as_matrix(),truth[:,:3,:3],atol=5e-7,rtol=0)
    def new_env(composition):
        result=RLEnv(composition.scene,composition.reference,config=instantiate_typed(cfg.env,RLEnvConfig))
        bridge=WarpRLEnv(result)
        if training.curriculum is not None:
            from dataclasses import replace
            stage=training.curriculum.stages[int(meta["curriculum_stage"])]
            bridge.set_curriculum(replace(stage,voc_scale=0.))
        result.reset(frame_id=0)
        return result
    env=new_env(setup)
    if int(env.timestep.numpy()[0])!=0 or env_config.auto_reset or np.any(env.object_control_scale.numpy()) or env_config.voc.max_force or env_config.voc.max_torque:
        raise ValueError("Unsafe eval reset condition or assistance")
    extent=scan_extents_m(args.parquet)[None,:] if shared else None
    def observation():
        x=env.observation.numpy().reshape(1,env.observation_dim)
        return jnp.asarray(np.concatenate([x,extent],axis=1) if shared else x)
    actor,act,key,_=load_policy_for_inference(str(args.checkpoint),training,observation(),env.action.action_dim,deterministic=True,device=resolve_jax_device(training.device))
    rec=SolverRecorder(env)
    initial=rec.initial_state["body_q"][rec.obj_id]
    np.testing.assert_allclose(initial[:3],truth[0,:3,3],atol=5e-7,rtol=0)
    np.testing.assert_allclose(Rotation.from_quat(initial[3:]).as_matrix(),truth[0,:3,:3],atol=5e-7,rtol=0)
    actions=[];stop=dict(failed=False,truncated=False,termination_causes=[],reason="source final sample")
    for i in range(len(frames)-1):
        if int(env.timestep.numpy()[0])!=i or np.any(env.object_control_scale.numpy()):
            raise ValueError(f"Source frame skipped/repeated/assisted before {i}")
        action,key=act(actor,observation(),key)
        action_np=np.asarray(action,dtype=np.float32).reshape(env.action_input.shape)
        if not np.isfinite(action_np).all(): raise ValueError("Nonfinite actor output")
        actions.append(action_np.copy())
        env.step(wp.array(action_np,dtype=wp.float32,device=env.device))
        rec.end_frame()
        if env.done.numpy()[0] or env.truncation.numpy()[0]:
            failed=bool(env.done.numpy()[0]) or i+1 < len(frames)-1
            stop=dict(failed=failed,truncated=bool(env.truncation.numpy()[0]),
                      reason="native episode ended",termination_causes=[name for name,arr in zip(env.termination.cause_names,env.termination.cause_arrays) if arr.numpy()[0]])
            break
    rec.close(); trace=rec.materialize()
    if len(trace["substep_contacts"])!=len(actions)*env.config.sim.substeps:raise ValueError("Incomplete physics trace")
    if not all(np.isfinite(x).all() for x in trace.values() if np.issubdtype(x.dtype,np.number)):
        raise ValueError("Nonfinite state, contact or command")
    error=rec.independent_replay(trace)
    protocol=json.loads(args.protocol.read_bytes());seq=setup.reference.metadata.sequence_id
    metrics,phases,gates=score(trace,truth,human,frames,env,rec,protocol,seq,stop,error)
    repeat_path=Path(__file__).resolve().parents[1]/"configs/hocap_carton_physical_repeat.json"
    repeat_cfg=json.loads(repeat_path.read_text(encoding="utf-8-sig"))
    if file_hash(args.protocol)!=repeat_cfg["base_protocol_sha256"] or repeat_cfg["replicates"]!=2:
        raise ValueError("Physical repeat must use the original frozen base gates and two fresh attempts")
    force=validation_status(cfg,env,seq)
    eligible=all(v for k,v in gates.items() if k!="recorded_command_replay")
    repeat=dict(status="failed_original_trajectory",source_trajectory_pass=eligible,
                force_validation=force,protocol_sha256=file_hash(repeat_path),attempts=[],comparisons=[])
    if eligible or getattr(args,"exercise_short_repeats",False):
        runs=[trace]
        for k in range(2):
            re,rr,tr,st=repeat_control_trajectory(lambda:new_env(setup_copy()),trace,stop,len(frames))
            rm,rp,rg=score(tr,truth,human,frames,re,rr,protocol,seq,st,0.)
            rg.pop("recorded_command_replay")
            rm.pop("independent_command_replay_max_state_error")
            np.savez_compressed(args.out/f"independent_physical_{k+1}.npz",**tr)
            repeat["attempts"].append(dict(metrics=rm,phases=rp,gates=rg,terminal=st,
                         identical_supplied_commands=True,source_intervals=len(tr["body_q"])-1,passed=all(rg.values())))
            runs.append(tr)
        for i,j in ((0,1),(0,2),(1,2)):
            repeat["comparisons"].append(dict(pair=[i,j],**pairwise_trajectory_differences(runs[i],runs[j],rec.obj_id)))
        numerical=eligible and all(r["passed"] for r in repeat["attempts"])
        repeat["status"]=("passed" if force["validated"] else "provisional_unknown_contact_forces") if numerical else "failed"
    trace.update(actor_action=np.asarray(actions),source_frames=frames[:len(actions)+1],raw_object_pose=truth,
                 raw_human_tips=human,frame_time=np.arange(len(actions)+1)/30.)
    np.savez_compressed(args.out/"trace.npz",**trace)
    result=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=file_hash(args.checkpoint),parquet_sha256=file_hash(args.parquet),
                actor_observation_dim=int(observation().shape[1]),shared_geometry_conditioned=shared,sequence=seq,
                physical_intervals=len(actions),requested_intervals=len(frames)-1,substeps_per_interval=env.config.sim.substeps,
                pause_on_original_training_failure=True,reset="explicit frame 0",auto_reset=False,VOC=0,
                zero_direct_object_force_verified_per_substep=True,terminal=stop,metrics=metrics,phases=phases,gates=gates,
                passed_numeric_development_gates=bool(all(gates.values())),physical_repeat=repeat,protocol_sha256=file_hash(args.protocol),
                seconds=time.perf_counter()-started)
    dump(args.out/"report.json",result);print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint","parquet","protocol","out","warp-cache"):p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--exercise-short-repeats",action="store_true",help="Validate the recorder on a failed prefix without counting it as a success")
    evaluate(p.parse_args())

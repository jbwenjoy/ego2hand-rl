"""Inspect a completed target checkpoint set on CPU before rollout or optimizer resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from heldout_teacher_health import inspect_actor


def allow_after_stop(event,kind,steps):
    if not event or event["resume_usable"]:return True
    if steps<event["step"]:return True  # Earlier completed policies, including the zero-shot disturbance.
    return bool(steps==event["step"] and event["actor_usable"] and kind in ("evaluate","disturbance"))


def stop_from_evaluation(report,steps):
    if report.get("status")!="numerical_actor_output":return None
    diag=report["numerical_failure"]
    if (diag["kind"]!="numerical_actor_output" or diag["rejected_action_applied"] or not diag["scene_finite"]
            or diag["nonfinite_input"] or diag["proposal"]["finite"]==int(np.prod(diag["proposal"]["shape"]))):
        raise ValueError("Unverified numerical-action stop")
    return dict(step=steps,actor_usable=False,resume_usable=False,classification="nonfinite_target_actor_output",
                rejected_source_frame=diag["rejected_source_frame"],applied_intervals=diag["applied_intervals"],
                checkpoint_sha256=report["checkpoint_sha256"])


def _scan_array(read,shape,dtype,first,last):
    if np.dtype(dtype).kind!="f" or last<=first:return None
    stride=(int(np.prod(shape[1:],dtype=np.int64)) if shape else 1)*np.dtype(dtype).itemsize
    batch=max(1,(8<<20)//max(stride,1))
    bad=dict(nan=0,posinf=0,neginf=0,first_bad_row=None)
    for i in range(first,last,batch):
        values=np.asarray(read(i,min(last,i+batch)))
        invalid=~np.isfinite(values)
        if not invalid.any():continue
        bad["nan"]+=int(np.isnan(values).sum())
        bad["posinf"]+=int(np.isposinf(values).sum())
        bad["neginf"]+=int(np.isneginf(values).sum())
        if bad["first_bad_row"] is None:
            bad["first_bad_row"]=i+int(np.flatnonzero(invalid.reshape(len(values),-1).any(1))[0])
    return bad if bad["first_bad_row"] is not None else None


def _scan_native(path,kind,folder_id,step):
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from flash_chord.configuration import instantiate_typed
    from flash_chord.objectives.config import OBJECTIVE_TERM_NAMES
    from flash_chord.training.checkpoint_metadata import checkpoint_config,checkpoint_critic_schema,checkpoint_policy_schema
    from flash_chord.training.flash_sac.checkpoint import training_checkpoint
    from flash_chord.training.flash_sac.config import TrainingConfig
    from flash_chord.training.flash_sac.learner import create_learner
    from flash_chord.training.flash_sac.replay import ReplaySpec,create_replay
    bad=[];discarded=[]
    with safe_open(str(path),framework="numpy") as f:
        hdr=f.metadata();meta={k[5:]:v for k,v in hdr.items() if k.startswith("user.")}
        if (hdr["format"]!="learner_state" or hdr["version"]!="1" or meta["checkpoint_kind"]!=kind
                or meta["checkpoint_set_id"]!=folder_id or int(meta["environment_steps"])!=step
                or meta["algorithm"]!="flash_sac"):
            raise ValueError("Wrong native checkpoint kind, format, set ID or step")
        cfg=checkpoint_config(meta);training=instantiate_typed(OmegaConf.create(cfg["training"]),TrainingConfig)
        p=checkpoint_policy_schema(meta);c=checkpoint_critic_schema(meta);worlds=training.world_count
        obs=p["observation"]["dimension"];action=p["action"]["dimension"]
        names=tuple(b["name"] for b in c["context"]["blocks"]);device=jax.devices("cpu")[0]
        if kind=="training_state":
            def structure():
                state,_=create_learner(training,jnp.zeros((worlds,obs),jnp.float32),jnp.zeros((worlds,len(names)),jnp.float32),
                    action_dim=action,objective_term_count=len(OBJECTIVE_TERM_NAMES),critic_context_names=names,device=device)
                return training_checkpoint(state)
        else:
            spec=ReplaySpec.from_config(training.replay,discount=training.discount,world_count=worlds,
                    observation_dim=obs,action_dim=action,objective_term_count=len(OBJECTIVE_TERM_NAMES),critic_context_dim=len(names))
            def structure():return create_replay(spec,device=device)
        leaves,_=jax.tree_util.tree_flatten_with_path(jax.eval_shape(structure))
        info={}
        for (name,leaf),row in zip(leaves,json.loads(hdr["manifest"]),strict=True):
            if list(leaf.shape)!=row["shape"] or np.dtype(leaf.dtype).str!=row["dtype"]:
                raise ValueError("Native checkpoint manifest does not match the tagged template")
            info[jax.tree_util.keystr(name)]=(row["key"],leaf.shape,leaf.dtype)
        if {val[0] for val in info.values()}!=set(f.keys()):raise ValueError("Unexpected native state leaves")
        committed=capacity=None
        if kind=="replay":
            committed=int(f.get_tensor(info[".size"][0]).item());capacity=spec.capacity
            write=int(f.get_tensor(info[".write_index"][0]).item())
            if not 0<=committed<=capacity or not 0<=write<capacity or (committed<capacity and write!=committed):
                raise ValueError("Replay ring counters inconsistent with committed data")
        for name,(key,shape,dtype) in info.items():
            n=shape[0] if shape else 1
            if kind=="replay" and name.startswith(".storage."):regions=[(0,committed,True),(committed,n,False)]
            elif name.startswith(".pending.") or name==".reward_normalizer.discounted_return":regions=[(0,n,False)]
            else:regions=[(0,n,True)]
            array=f.get_slice(key)
            for lo,hi,used in regions:
                failure=_scan_array(lambda i,j:array[i:j],shape,dtype,lo,hi)
                if failure:(bad if used else discarded).append(dict(leaf=name,**failure))
        return dict(kind=kind,source_fingerprint=cfg.get("source_fingerprint"),
                    committed_rows=committed,capacity=capacity,
                    unusable_values=bad,discarded_on_resume_or_unused=discarded)


def _scan_aux(path,pair,steps):
    from flax import serialization
    with path.open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=pair["auxiliary_sha256"]:
            raise ValueError("Auxiliary checkpoint digest mismatch")
    data=serialization.msgpack_restore(path.read_bytes())
    if int(data["native_environment_steps"])!=steps:raise ValueError("Auxiliary checkpoint step mismatch")
    state=data["state"];hist=int(data["options"]["history"]);collected=int(data["collected"])
    if state["history"]["action"].shape[:1]!=(hist,) or collected*state["history"]["action"].shape[1]!=steps:
        raise ValueError("Auxiliary source-world history mismatch")
    needed={"model_params","model_optimizer","actor_optimizer","scale","history","key","model_updates","actor_updates","skipped_updates"}
    if not needed<=set(state):raise ValueError("Incomplete auxiliary learner state")
    bad=[];ignored=[]
    def inspect(name,value):
        if isinstance(value,dict):
            for k,x in value.items():inspect(name+"."+k if name else k,x)
            return
        if isinstance(value,(list,tuple)):
            for k,x in enumerate(value):inspect(name+"."+str(k),x)
            return
        arr=np.asarray(value)
        if arr.dtype.kind!="f":return
        x=arr.reshape(1) if arr.ndim==0 else arr
        end=len(x)
        if name.startswith("history."):regions=[(0,min(collected,hist),True),(min(collected,hist),end,False)]
        else:regions=[(0,end,name not in {"last_model_loss","last_actor_loss"})]
        for lo,hi,used in regions:
            failure=_scan_array(lambda i,j:x[i:j],x.shape,arr.dtype,lo,hi)
            if failure:(bad if used else ignored).append(dict(leaf=name,**failure))
        if name in ("scale.state_scale","scale.output_scale") and (arr<=0).any():
            bad.append(dict(leaf=name,nonpositive_scale=int((arr<=0).sum())))
    inspect("",state)
    return dict(unusable_values=bad,ignored_telemetry_or_unfilled_history=ignored)


def inspect_target_set(directory,steps,hybrid):
    started=time.perf_counter();directory=Path(directory)
    if not (directory/f"paired_{steps}.json").is_file():directory=directory/"checkpoints"
    pair=json.loads((directory/f"paired_{steps}.json").read_bytes())
    if pair["environment_steps"]!=steps or pair["hybrid"]!=hybrid:raise ValueError("Checkpoint pair budget or method mismatch")
    actor=inspect_actor(directory/f"policy_{steps}.safetensors");identity=pair["checkpoint_set_id"]
    if actor["checkpoint_set_id"]!=identity or actor["teacher_environment_steps"]!=steps:raise ValueError("Actor and native state mismatch")
    state=_scan_native(directory/f"state_{steps}.safetensors","training_state",identity,steps)
    replay=_scan_native(directory/f"replay_{steps}.safetensors","replay",identity,steps)
    if any(x["source_fingerprint"]!=pair["source_fingerprint"] for x in (state,replay)):
        raise ValueError("Reference/physics fingerprint disagrees across target checkpoints")
    aux=None
    if hybrid:
        if pair["auxiliary_file"]!=f"world_model_{steps}.msgpack":raise ValueError("Unexpected auxiliary filename")
        aux=_scan_aux(directory/pair["auxiliary_file"],pair,steps)
    resume=actor["usable"] and not (state["unusable_values"] or replay["unusable_values"] or (aux and aux["unusable_values"]))
    classification=("nonfinite_target_actor" if not actor["usable"] else "healthy" if resume else "nonfinite_target_resume")
    return dict(step=steps,actor_usable=actor["usable"],resume_usable=bool(resume),classification=classification,
                checkpoint_set_id=identity,actor=actor,native_state=state,replay=replay,auxiliary=aux,
                inspection_cpu_wall_s=time.perf_counter()-started)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path);parser.add_argument("steps",type=int)
    parser.add_argument("output",type=Path);parser.add_argument("--hybrid",action="store_true")
    a=parser.parse_args();os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["JAX_PLATFORMS"]="cpu"
    result=inspect_target_set(a.directory,a.steps,a.hybrid)
    temp=a.output.with_suffix(".json.tmp");temp.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");temp.replace(a.output)

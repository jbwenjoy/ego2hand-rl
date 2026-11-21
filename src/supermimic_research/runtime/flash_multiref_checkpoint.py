"""Pair auxiliary world-model state with native checkpoint identity at save boundaries."""
import hashlib
import json
import math
from pathlib import Path
import time

import jax
import numpy as np

from flash_chord.training.checkpoint import read_checkpoint_metadata


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def parameter_hash(tree):
    digest=hashlib.sha256()
    for array in jax.tree.leaves(jax.device_get(tree)):
        values=np.ascontiguousarray(array)
        digest.update(str((values.dtype,values.shape)).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def atomic_json(path,value):
    temp=path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf8")
    temp.replace(path)


class SharedInitialState:
    """Publish/load the same untrained native optimizer tree before initial_action."""
    def __init__(self, directory, create, contract, run_dir):
        self.directory=Path(directory)
        self.create=create
        self.contract=contract
        self.run_dir=run_dir

    def __call__(self, learner):
        from flash_chord.training.flash_sac.checkpoint import (
            restore_learner_checkpoint, save_learner_checkpoint, training_checkpoint,
        )
        start=time.perf_counter()
        if (int(learner.environment_steps)!=0 or int(learner.global_update_step)!=0 or int(learner.replay.size)!=0
                or int(learner.actor.schedule_step)!=0 or int(learner.critic.schedule_step)!=0):
            raise ValueError("Can only initialize a fresh, empty native learner")
        manifest=self.directory/"initial.json"
        path=self.directory/"state_0.safetensors"
        if self.create:
            self.directory.mkdir(parents=True,exist_ok=False)
            paths=save_learner_checkpoint(self.directory,learner,metadata={"global_update_step":0},save_replay=False)
            if paths.training_state!=path:
                raise ValueError("Unexpected native zero-step checkpoint path")
            atomic_json(manifest,dict(contract=self.contract,state_sha256=file_hash(path),
                        full_training_tree_sha256=parameter_hash(training_checkpoint(learner)),
                        actor_sha256=parameter_hash(learner.actor.params),critic_sha256=parameter_hash(learner.critic.params)))
        spec=json.loads(manifest.read_bytes())
        if self.contract!=spec["contract"] or file_hash(path)!=spec["state_sha256"]:
            raise ValueError("Incompatible initialization contract or damaged zero-state file")
        restored,metadata=restore_learner_checkpoint(path,learner,load_optimizer=True,load_reward_normalizer=True)
        if int(metadata["environment_steps"]) or int(restored.global_update_step):
            raise ValueError("Shared initialization contains training transitions")
        digest=parameter_hash(training_checkpoint(restored))
        if digest!=spec["full_training_tree_sha256"]:
            raise ValueError("Restored initial native learner tree differs from saved zero-step state")
        # Quantify initialization drift before overriding it, not just two SHA strings.
        stats={}
        for key in ("actor","critic"):
            fresh=jax.tree.leaves(jax.device_get(getattr(learner,key).params))
            frozen=jax.tree.leaves(jax.device_get(getattr(restored,key).params))
            max_abs=sumsq=count=unequal=0
            for a,b in zip(fresh,frozen,strict=True):
                a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
                if a.shape!=b.shape: raise ValueError("Shared network shape mismatch")
                delta=a-b
                max_abs=max(max_abs,float(np.max(abs(delta))))
                sumsq+=float(np.sum(delta*delta))
                unequal+=int(np.count_nonzero(delta));count+=delta.size
            stats[key]=dict(parameters=count,numerically_unequal=unequal,max_abs=max_abs,rms=math.sqrt(sumsq/count))
        atomic_json(self.run_dir/"shared_initial_loader.json",dict(created=self.create,physical_step=0,
                    directory=str(self.directory),full_training_tree_sha256=digest,comparison_to_fresh=stats,
                    initial_observation_sha256=parameter_hash(learner.observation),
                    initial_context_sha256=parameter_hash(learner.critic_context),wall_s=time.perf_counter()-start))
        return restored


def transfer_actor(learner,path,schema,out):
    """Start a timed adaptation with only the previously trained actor and batch stats."""
    from flash_chord.training.flash_sac.checkpoint import actor_inference_state, load_actor_checkpoint
    from flash_chord.training.checkpoint_metadata import checkpoint_policy_schema, validate_policy_schema
    start=time.perf_counter()
    if int(learner.environment_steps) or int(learner.replay.size):
        raise ValueError("Actor transfer must start a fresh per-reference sampling budget")
    template=actor_inference_state(learner)
    actor,saved=load_actor_checkpoint(path,template)
    validate_policy_schema(checkpoint_policy_schema(saved),schema)
    result=learner.replace(actor=learner.actor.replace(params=actor.params,batch_stats=actor.batch_stats))
    atomic_json(out/"adaptation_start.json",dict(source_checkpoint=str(path),source_sha256=file_hash(path),
                source_training_steps=int(saved["environment_steps"]),target_native_steps=int(result.environment_steps),
                target_replay_size=int(result.replay.size),transferred="actor parameters and actor batch statistics",
                actor_sha256=parameter_hash(result.actor.params),wall_s=time.perf_counter()-start))
    return result


class SharedRunHook:
    def __init__(self, aux, *, fingerprint, out, worlds, restore_state=None, restore_replay=None, max_aux_bytes=1<<29,
                 legacy_fingerprint=None):
        self.aux=aux
        self.out=out
        self.fingerprint=fingerprint
        self.worlds=worlds
        self.max_aux_bytes=max_aux_bytes
        self.resume=None
        self.first=True
        if restore_state is not None:
            state=read_checkpoint_metadata(restore_state)
            replay=read_checkpoint_metadata(restore_replay)
            steps=int(state["environment_steps"])
            pair=json.loads((restore_state.parent/f"paired_{steps}.json").read_bytes())
            identity=state["checkpoint_set_id"]
            if (pair["checkpoint_set_id"]!=identity or replay["checkpoint_set_id"]!=identity
                    or int(replay["environment_steps"])!=steps or pair["environment_steps"]!=steps
                    or pair["source_fingerprint"] not in (fingerprint,legacy_fingerprint)
                    or pair["hybrid"]!=(aux is not None)):
                raise ValueError("Checkpoint set, source bank or method mismatch")
            if aux is not None:
                file=restore_state.parent/pair["auxiliary_file"]
                if pair["auxiliary_sha256"]!=file_hash(file):
                    raise ValueError("Auxiliary checkpoint integrity mismatch")
                self.resume=file,steps

    def __call__(self,consume,learner,tr,*runtime_args):
        if self.first:
            self.first=False
            steps=int(jax.device_get(learner.environment_steps))
            atomic_json(self.out/"initial_learner.json",dict(native_steps=steps,
                        actor_sha256=parameter_hash(learner.actor.params),critic_sha256=parameter_hash(learner.critic.params),
                        actor_schedule=int(learner.actor.schedule_step),critic_schedule=int(learner.critic.schedule_step)))
            if self.resume is not None:
                from supermimic_research.learning.flash_imagination import NativeSteps
                path,expected_steps=self.resume
                if steps!=expected_steps:
                    raise ValueError("Loaded native step does not match auxiliary state")
                batch=NativeSteps(learner.observation,learner.critic_context,tr.action,tr.next_observation,
                                  tr.next_critic_context,tr.objective_terms,tr.terminated,tr.truncated)
                self.aux.restore(path,batch,learner.actor.params,native_environment_steps=steps)
                if self.aux.collected*self.worlds!=steps:
                    raise ValueError("Restored auxiliary history and native step disagree")
        if self.aux is None:
            return consume(learner,tr,*runtime_args)
        return self.aux(consume,learner,tr,*runtime_args)

    def stats(self):
        if self.aux is None or self.aux.state is None:
            return {}
        state=self.aux.state
        opts=self.aux.options
        every,seen=opts["every"],self.aux.collected
        opportunities=lambda start: max(0, seen//every-math.ceil(start/every)+1)
        vals=jax.device_get((state.model_updates,state.actor_updates,state.skipped_updates,
                             state.last_model_loss,state.last_actor_loss))
        finite=lambda x:float(x) if math.isfinite(float(x)) else None
        return dict(history_collections=seen,world_model_update_attempts=opportunities(opts["history"]),
                    imagined_actor_update_attempts=opportunities(opts["history"]+every*opts["model_warmup"]),
                    world_model_updates=int(vals[0]),imagined_actor_updates=int(vals[1]),aux_skipped=int(vals[2]),
                    last_world_model_loss=finite(vals[3]),last_imagined_actor_loss=finite(vals[4]))

    def save(self,paths,environment_steps):
        native=read_checkpoint_metadata(paths.training_state)
        if int(native["environment_steps"])!=environment_steps or paths.replay is None:
            raise ValueError("Complete native checkpoint and replay required")
        identity=native["checkpoint_set_id"]
        for path in (paths.policy,paths.replay):
            if read_checkpoint_metadata(path)["checkpoint_set_id"]!=identity:
                raise ValueError("Native checkpoint identity mismatch")
        pair=dict(environment_steps=environment_steps,checkpoint_set_id=identity,source_fingerprint=self.fingerprint,
                  hybrid=self.aux is not None,auxiliary_metrics=self.stats())
        if self.aux is not None:
            if self.aux.collected*self.worlds!=environment_steps:
                raise ValueError("Auxiliary state lags native checkpoint")
            nbytes=sum(a.nbytes for a in jax.tree.leaves(self.aux.state))
            if nbytes>self.max_aux_bytes:
                raise ValueError(f"Auxiliary tree needs {nbytes} bytes, exceeds configured save budget")
            path=paths.training_state.parent/f"world_model_{environment_steps}.msgpack"
            self.aux.save(path,native_environment_steps=environment_steps)
            pair.update(auxiliary_file=path.name,auxiliary_sha256=file_hash(path),auxiliary_bytes=path.stat().st_size)
        atomic_json(paths.training_state.parent/f"paired_{environment_steps}.json",pair)

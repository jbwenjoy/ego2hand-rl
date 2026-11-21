"""Run one FlashSAC learner on one or more equally weighted native Wuji scenes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import time

from run_wuji_flashsac_smoke import LocalMetrics, append, config_for, dump, record_device


def scaled_thresholds(thresholds, horizon, worlds):
    """Fit the upstream transition boundaries onto distinct whole native collections."""
    if horizon % worlds or horizon // worlds < len(thresholds):
        raise ValueError("Curriculum horizon needs at least one whole collection per stage boundary")
    if not thresholds or thresholds[0]<=0 or any(a>=b for a,b in zip(thresholds,thresholds[1:])):
        raise ValueError("Expected positive increasing upstream curriculum boundaries")
    total=horizon//worlds
    ticks=[]
    for i,raw in enumerate(thresholds):
        rounded=round(raw*total/thresholds[-1])
        lower=ticks[-1]+1 if ticks else 1
        upper=total-(len(thresholds)-1-i)
        ticks.append(min(upper,max(lower,rounded)))
    return [x*worlds for x in ticks]


def run(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import warp as wp
    from omegaconf import OmegaConf
    from flash_chord.configuration import instantiate_typed, resolved_dict
    from flash_chord.training.checkpoint_metadata import build_checkpoint_metadata, resolved_reference_config
    from flash_chord.training.flash_sac.config import TrainingConfig
    from supermimic_research.runtime.flash_multiref import (
        MultiReferenceEnv, build_native_shard, run_shared_training, scan_extents_m,
    )
    from supermimic_research.runtime.flash_multiref_checkpoint import SharedInitialState, SharedRunHook, transfer_actor

    refs=[p.resolve() for p in args.parquet]
    if len(set(refs))!=len(refs):
        raise ValueError("Each reference may occur in one native shard only")
    source_info=[]
    for parquet in refs:
        directory=parquet if parquet.is_dir() else parquet.parent
        source_info.append(json.loads((directory/"export.json").read_bytes()))
    heldout=[r.get("phase_scope")=="frozen_heldout_evaluation" for r in source_info]
    if any(heldout) and (len(refs)!=1 or (args.mode=="train" and not args.evaluation_adaptation)):
        raise ValueError("Reserved reference cannot enter the development training bank")
    if args.evaluation_adaptation and (len(refs)!=1 or not all(heldout) or not source_info[0].get("frozen_recipe")):
        raise ValueError("Explicit adaptation requires one recipe-frozen evaluation reference")
    capacity=args.worlds_per_reference*len(refs)
    cfg_args = SimpleNamespace(mode="train",worlds=capacity,parquet=refs[0],assets=args.assets,
                               upstream=args.upstream,out=args.out,steps=args.steps,
                               override=[f"training.device={args.device}",*args.override])
    cfg = config_for(cfg_args)
    OmegaConf.update(cfg,"env.contact.sides",["right"],force_add=True)
    raw_thresholds=list(cfg.curriculum.thresholds)
    if args.curriculum_horizon_steps is None:
        # The isolated health arm removes assistance even at reference-reset settling.
        cfg.reset.voc_decay_steps=0
        cfg.env.voc_scale=0.
        cfg.env.voc.max_force=0.
        cfg.env.voc.max_torque=0.
        cfg.voc_scale_override=0.
        curriculum_mode="zero_object_assistance"
    else:
        cfg.curriculum.thresholds=scaled_thresholds(raw_thresholds,args.curriculum_horizon_steps,capacity)
        cfg.voc_scale_override=None
        curriculum_mode="upstream_stages_scaled_to_declared_budget"
    if args.checkpoint_every is not None:
        if args.checkpoint_every%capacity or args.checkpoint_every<capacity:
            raise ValueError("Checkpoint period must be a positive multiple of the batched world count")
        cfg.training.checkpoint.save_interval_environment_steps=args.checkpoint_every
    if args.resume_state:
        cfg.training.resume=True
        cfg.training.checkpoint_path=str(args.resume_state)
        cfg.training.checkpoint.load_replay_path=str(args.resume_replay)
    training=instantiate_typed(cfg.training,TrainingConfig)
    wp.config.kernel_cache_dir=str(args.upstream.parent/"cache/warp")
    start=time.perf_counter()
    allocations=[]
    shards, setups, extent=[],[],[]
    with wp.ScopedDevice(args.device):
        for i,parquet in enumerate(refs):
            t=time.perf_counter()
            native, setup=build_native_shard(cfg,parquet,worlds=args.worlds_per_reference,
                                             capture=args.device.startswith("cuda"),seed_offset=i)
            shards.append(native);setups.append(setup);extent.append(scan_extents_m(parquet))
            allocations.append(dict(sequence=setup.reference.metadata.sequence_id,parquet=str(parquet),
                                    sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
                                    worlds=args.worlds_per_reference,body_count=setup.scene.model.body_count,
                                    material_extents_m=extent[-1].tolist(),frames=setup.reference.num_frames,
                                    solver_conditioning=getattr(native.env.solver,"wuji_conditioning",None),
                                    reset_seed=int(cfg.reset.seed)+i,seconds=time.perf_counter()-t))
            dump(args.out/"allocated.json",allocations)
        joined=MultiReferenceEnv(shards,extent)
        environment=resolved_dict(cfg)
        environment["reference"]=resolved_reference_config(setups[0].reference,str(refs[0]))
        environment["reference_shards"]=allocations
        environment["geometry_source"]="Source scan bounds in local object axes, metres; appended as XYZ extents."
        bank=[{k:v for k,v in s.items() if k not in ("parquet","seconds")} for s in allocations]
        parameters=dict(bank=bank,environment=environment["env"],actor_schema=joined.policy_schema,
                        critic_schema=joined.critic_schema,action=environment["action"],objective=environment["objective"],
                        physics=environment["sim"],scene=environment["scene"])
        import newton
        from supermimic_research.runtime import flash_wuji
        parameters["physics_identity"]=dict(
            version=2,embodiment=environment["embodiment"],collision=environment["collision"],
            calibrated_asset_sha256=hashlib.sha256((args.assets/"adapter.json").read_bytes()).hexdigest(),
            wuji_adapter_source_sha256=hashlib.sha256(Path(flash_wuji.__file__).read_bytes()).hexdigest(),
            wuji_solver_source_sha256=hashlib.sha256(Path(flash_wuji.__file__).with_name("flash_solver.py").read_bytes()).hexdigest(),
            warp_version=wp.__version__,newton_version=getattr(newton,"__version__","unknown"),
        )
        environment["physics_identity"]=parameters["physics_identity"]
        fingerprint=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()
        environment["source_fingerprint"]=fingerprint
        environment["hybrid"]=args.hybrid
        environment["reference_role"]=("heldout_adaptation" if args.evaluation_adaptation else
                                       "heldout_contract" if any(heldout) else "development")
        environment["frozen_evaluation_recipe"]=source_info[0].get("frozen_recipe") if any(heldout) else None
        environment["shared_initial_state"]=str(args.shared_initial_state) if args.shared_initial_state else None
        environment["curriculum_protocol"]=dict(mode=curriculum_mode,declared_horizon=args.curriculum_horizon_steps,
                raw_thresholds=raw_thresholds,mapped_thresholds=list(cfg.curriculum.thresholds),
                object_scale_override=cfg.voc_scale_override,reset_settling_steps=cfg.reset.voc_decay_steps,
                max_object_force=cfg.env.voc.max_force,max_object_torque=cfg.env.voc.max_torque)
        legacy_fingerprint=fingerprint
        curriculum_contract=dict(previous_fingerprint=fingerprint,settings=environment["curriculum"],
                                 protocol=environment["curriculum_protocol"])
        fingerprint=hashlib.sha256(json.dumps(curriculum_contract,sort_keys=True).encode()).hexdigest()
        environment["source_fingerprint"]=fingerprint
        # Only accept old unassisted checkpoints if their actual stored stages and
        # zero-VOC configuration still agree with this run.
        resume_legacy_fingerprint=None
        if args.resume_state and args.curriculum_horizon_steps is None:
            from flash_chord.training.checkpoint import read_checkpoint_metadata
            from flash_chord.training.checkpoint_metadata import checkpoint_config
            saved=checkpoint_config(read_checkpoint_metadata(args.resume_state))
            if ("curriculum_protocol" not in saved and saved["curriculum"]==environment["curriculum"]
                    and saved["voc_scale_override"]==0.0):
                resume_legacy_fingerprint=legacy_fingerprint
        metadata=build_checkpoint_metadata(environment,joined.policy_schema,joined.critic_schema)
        dump(args.out/"environment.json",environment)
        dump(args.out/"policy_schema.json",joined.policy_schema)
        dump(args.out/"critic_schema.json",joined.critic_schema)
        if args.mode=="contract":
            stage=training.curriculum.stages[0]
            joined.set_curriculum(stage)
            obs=np.array(joined.reset(),copy=True)
            initial=np.array(joined.initial_critic_context,copy=True)
            action=jnp.zeros((capacity,joined.action_dim),dtype=jnp.float32)
            step=joined.step(action)
            progress,diag=joined.transition_logging_inputs()
            ext=np.repeat(extent,args.worlds_per_reference,axis=0)
            assert obs.shape==(capacity,joined.observation_dim) and initial.shape==(capacity,joined.critic_context_dim)
            for name in ("observation","terminal_observation"):
                a=np.asarray(getattr(step,name));np.testing.assert_array_equal(a[:,-3:],ext)
                assert a.shape==obs.shape and np.isfinite(a).all()
            assert step.objective_terms.shape==(capacity,len(joined.objective_term_names))
            assert progress.shape==(capacity,) and diag.shape[0]==capacity
            np.testing.assert_array_equal(obs[:,-3:],ext)
            assert np.isfinite(step.objective_terms).all()
            summary=dict(native_zero_action_steps_per_reference=args.worlds_per_reference,
                         actor_dim=joined.observation_dim,critic_context_dim=joined.critic_context_dim,
                         action_dim=joined.action_dim,initial_schema=joined.policy_schema,device=args.device)
        else:
            aux=None
            if args.hybrid:
                from supermimic_research.learning.flash_imagination_hook import ImaginationHook
                aux=ImaginationHook(training,history=args.wm_history,batch=args.wm_batch,horizon=args.horizon,
                                    every=args.wm_every,model_warmup=args.wm_warmup,geometry_tail=3,seed=args.wm_seed,
                                    fit_horizon=args.wm_fit_horizon)
            hook=SharedRunHook(aux,fingerprint=fingerprint,out=args.out,worlds=capacity,
                               restore_state=args.resume_state,restore_replay=args.resume_replay,
                               max_aux_bytes=args.aux_max_mib*(1<<20),
                               legacy_fingerprint=resume_legacy_fingerprint)
            class Logger(LocalMetrics):
                def log(self,metrics,step):
                    super().log(metrics,step)
                    if aux is not None:
                        append(args.out/"auxiliary_updates.jsonl",dict(step=int(step),**hook.stats()))
                def log_checkpoint(self,paths,*,environment_steps,metadata):
                    hook.save(paths,environment_steps)
            initializer=None
            if args.shared_initial_state:
                train_config=environment["training"]
                train_keys=("network","optimization","reward_normalization","discount","mixed_precision","compute_dtype","seed")
                init_contract=dict(world_count=capacity,policy=joined.policy_schema,critic=joined.critic_schema,
                                   native_config={k:train_config[k] for k in train_keys})
                initializer=SharedInitialState(args.shared_initial_state,args.create_shared_initial_state,
                                               init_contract,args.out)
            if args.adapt_from_actor:
                baseline_initializer=initializer
                def initialize_adaptation(learner):
                    if baseline_initializer is not None: learner=baseline_initializer(learner)
                    return transfer_actor(learner,args.adapt_from_actor,joined.policy_schema,args.out)
                initializer=initialize_adaptation
            learner=run_shared_training(joined,training,logger=Logger(args.out),transition_hook=hook,initializer=initializer,
                                 checkpoint_dir=args.out/"checkpoints",log_interval=args.log_interval,
                                 voc_scale_override=cfg.voc_scale_override,checkpoint_metadata=metadata)
            steps=int(jax.device_get(learner.environment_steps))
            updates=int(jax.device_get(learner.global_update_step))
            assert steps==args.steps and updates>0
            initial=json.loads((args.out/"initial_learner.json").read_bytes())
            added=steps-initial["native_steps"]
            summary=dict(learner_steps=steps,optimizer_updates=updates,native_sac_actor_updates=int(learner.actor.schedule_step),
                         native_sac_critic_updates=int(learner.critic.schedule_step),hybrid=args.hybrid,
                         auxiliary=hook.stats(),source_fingerprint=fingerprint,
                         per_reference_cumulative_steps=steps//len(refs),
                         per_reference_collected_steps_this_process=added//len(refs),device=args.device)
            if args.evaluation_adaptation:
                summary.update(role="per_reference_adaptation",declared_target_steps=args.steps,
                               transferred_actor=str(args.adapt_from_actor) if args.adapt_from_actor else None)
        dump(args.out/"result.json",dict(**summary,wall_seconds=time.perf_counter()-start,shards=allocations))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("contract","train"))
    p.add_argument("--parquet",type=Path,action="append",required=True)
    for field in ("upstream","assets","out"):
        p.add_argument("--"+field,type=lambda s:Path(s).resolve(),required=True)
    p.add_argument("--steps",type=int,required=True)
    p.add_argument("--worlds-per-reference",type=int,required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--log-interval",type=int,default=4)
    p.add_argument("--checkpoint-every",type=int)
    p.add_argument("--curriculum-horizon-steps",type=int,
                   help="Total declared transitions for the scaled upstream training curriculum, including future resumed tranches.")
    p.add_argument("--resume-state",type=lambda s:Path(s).resolve())
    p.add_argument("--resume-replay",type=lambda s:Path(s).resolve())
    p.add_argument("--shared-initial-state",type=lambda s:Path(s).resolve())
    p.add_argument("--create-shared-initial-state",action="store_true")
    p.add_argument("--evaluation-adaptation",action="store_true",help="Run only a single frozen evaluation reference as an explicitly timed adaptation.")
    p.add_argument("--adapt-from-actor",type=lambda s:Path(s).resolve(),help="Transfer actor and batch statistics only, leaving adaptation counters at zero.")
    p.add_argument("--hybrid",action="store_true")
    p.add_argument("--wm-history",type=int,default=32)
    p.add_argument("--wm-batch",type=int,default=128)
    p.add_argument("--wm-every",type=int,default=4)
    p.add_argument("--wm-warmup",type=int,default=16)
    p.add_argument("--wm-seed",type=int,default=318)
    p.add_argument("--wm-fit-horizon",type=int,default=1)
    p.add_argument("--horizon",type=int,default=3)
    p.add_argument("--aux-max-mib",type=int,default=512)
    p.add_argument("--override",action="append",default=[])
    a=p.parse_args()
    if a.worlds_per_reference<1: p.error("positive worlds per reference required")
    if bool(a.resume_state)!=bool(a.resume_replay): p.error("Provide both native training-state and replay checkpoints")
    if a.create_shared_initial_state and not a.shared_initial_state: p.error("Specify shared initial-state directory")
    if a.resume_state and a.shared_initial_state: p.error("Choose checkpoint resume or zero-step initialization")
    if a.adapt_from_actor and (not a.evaluation_adaptation or a.resume_state):
        p.error("An actor transfer is valid only when starting a new explicit evaluation adaptation")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
    if a.device=="cpu":
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        os.environ["JAX_PLATFORMS"]="cpu"
    a.out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    stop=threading.Event()
    monitor=None
    if a.device.startswith("cuda"):
        monitor=threading.Thread(target=record_device,args=(stop,a.out,time.monotonic()),daemon=True)
        monitor.start()
    try:
        run(a)
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=7)
        dump(a.out/"process_wall.json",{"wall_seconds":time.perf_counter()-start})

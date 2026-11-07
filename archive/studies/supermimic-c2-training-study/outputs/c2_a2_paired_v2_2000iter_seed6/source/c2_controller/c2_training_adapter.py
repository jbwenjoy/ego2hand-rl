"""Scoped C2_020 training adapter; native rewards/WM/EMA are unchanged.

Imagination reads only the native MDP's frozen startup position limits. Both
paths use the same FP32 projection. Checkpoints use a distinct fail-closed
format; no exact replay/environment resume is provided by this extension.
"""
from contextlib import contextmanager
from pathlib import Path

C2_FORMAT_VERSION=103
MARGIN_RAD=.020


@contextmanager
def training_controller(env_class,mdp_class,runner_class,metadata):
    """Install only for this isolated trainer process/context; restore on exit."""
    import torch
    from target_inset import target_inset,validate_inset_domain
    original_env_init=env_class.__init__
    original_env_step=env_class.step
    original_mdp_init=mdp_class.__init__
    original_control_target=mdp_class.control_target
    original_save=runner_class.save
    original_collect=getattr(runner_class,"collect",None)
    if metadata.get("record_actual_u",False) and original_collect is None:
        raise ValueError("Actualu evidence requires the native collector")
    original_load=runner_class.load
    original_model_load=runner_class.load_model_only

    def env_init(env,*args,**kwargs):
        original_env_init(env,*args,**kwargs)
        term=env.action_manager.get_term("joint_pos")
        if term._mask_enabled:
            raise ValueError("C2 controlled training requires mask0")
        lower,upper=term.target_position_limits
        validate_inset_domain(lower,upper,MARGIN_RAD)
        if not torch.equal(lower,lower[:1].expand_as(lower)) or not torch.equal(upper,upper[:1].expand_as(upper)):
            raise ValueError("C2 requires identical frozen position limits in all worlds")
        env._c2_lower=lower[0].detach().clone()
        env._c2_upper=upper[0].detach().clone()
        env._c2_action_active=False
        env._c2_pending_controls=[]
        native_process=term.process_actions

        def process(action):
            native_process(action)
            if env._c2_action_active:
                if metadata.get("record_actual_u",False):
                    env._c2_last_proposed=term._targets.detach().clone()
                    env._c2_last_residual_before=term.processed_action.detach().clone()
                # Only the absolute cache is changed. Native processed_action
                # remains the residual used for c/c1/c2/reward/history.
                term._targets.copy_(target_inset(term._targets,env._c2_lower,env._c2_upper,MARGIN_RAD))

        term.process_actions=process

    def env_step(env,*args,**kwargs):
        env._c2_action_active=True
        try:
            result=original_env_step(env,*args,**kwargs)
            if metadata.get("record_actual_u",False):
                import warp as wp
                term=env.action_manager.get_term("joint_pos")
                actual=wp.to_torch(env.sim.wp_data.ctrl)[:,term._position_ctrl_ids]
                env._c2_pending_controls.append({"proposed":env._c2_last_proposed,
                    "cached":term._targets.detach().clone(),"actual":actual.detach().clone(),
                    "residual_before_filter":env._c2_last_residual_before,
                    "residual":term.processed_action.detach().clone()})
            return result
        finally:
            env._c2_action_active=False

    def mdp_init(mdp,reader,*args,**kwargs):
        original_mdp_init(mdp,reader,*args,**kwargs)
        if mdp.mask_cfg.mask_prob!=0:
            raise ValueError("C2 imagination requires mask0; no live curriculum/state reads")
        validate_inset_domain(mdp.target_lower,mdp.target_upper,MARGIN_RAD)
        if not torch.equal(mdp.target_lower,reader.env._c2_lower) or not torch.equal(mdp.target_upper,reader.env._c2_upper):
            raise ValueError("Real/imagined startup hard/ctrl intersections disagree")
        mdp.c2_controller_metadata={**metadata,"kind":"C2_TARGET_INSET","margin_rad":MARGIN_RAD,
            "format_version":C2_FORMAT_VERSION,"exact_resume_supported":False,
            "initialization":"native reset/initial ctrl unchanged; filter only actual action intervals",
            "memory":"c/c1/c2 remain EMA residual, frozen_target is filtered actual absoluteu"}

    def collect(runner,steps):
        if not metadata.get("record_actual_u",False):
            return original_collect(runner,steps)
        import json
        import numpy as np
        runner.env._c2_pending_controls=[]
        result=original_collect(runner,steps)
        pending=runner.env._c2_pending_controls
        if len(pending)!=steps:
            raise ValueError("C2 actualu evidence missing collection steps")
        values={k:torch.stack([v[k] for v in pending]) for k in pending[0]}
        rows=(runner.buffer.ptr-steps+torch.arange(steps,device=runner.device))%runner.buffer.capacity
        replay_u=runner.buffer.data["u"][rows]
        replay_c=runner.buffer.data["c"][rows]
        projected=target_inset(values["proposed"],runner.env._c2_lower,runner.env._c2_upper,MARGIN_RAD)
        finite=all(bool(torch.isfinite(v).all()) for v in (*values.values(),replay_u,replay_c))
        inside=bool(((values["cached"]>=runner.env._c2_lower+MARGIN_RAD)&(values["cached"]<=runner.env._c2_upper-MARGIN_RAD)).all())
        errors={"cache_vs_actual":float((values["cached"]-values["actual"]).abs().max()),
            "proposed_projection_vs_cache":float((projected-values["cached"]).abs().max()),
            "residual_before_after_filter":float((values["residual_before_filter"]-values["residual"]).abs().max()),
            "replay_u_vs_actual":float((replay_u-values["actual"]).abs().max()),
            "replay_c_vs_residual":float((replay_c-values["residual"]).abs().max())}
        if not finite or not inside or any(v!=0 for v in errors.values()):
            raise ValueError(f"C2 executed/replay control mismatch: {errors}")
        count=getattr(runner,"_c2_collect_calls",0)+1;runner._c2_collect_calls=count
        summary={"collect_call":count,"iteration":runner.iteration,"steps":steps,
            "worlds":runner.env.num_envs,"all_batch_exact":True,"all_finite":finite,
            "within_inset_domain":inside,"max_abs_errors":errors,
            "clamped_components":int((values["proposed"]!=values["cached"]).sum()),
            "total_components":values["cached"].numel()}
        runner.c2_actualu_evidence=summary
        output=Path(runner.log_dir).parent
        with (output/"c2_actualu_evidence.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(summary)+"\n")
        # Fixed sampling schedule; selection never depends on data or quality.
        if count in (1,2,3) or count%100==0:
            chosen={k:v[-min(4,steps):,:min(4,runner.env.num_envs)].cpu().numpy() for k,v in values.items()}
            chosen["replay_u"]=replay_u[-min(4,steps):,:min(4,runner.env.num_envs)].cpu().numpy()
            chosen["replay_c"]=replay_c[-min(4,steps):,:min(4,runner.env.num_envs)].cpu().numpy()
            np.savez_compressed(output/f"c2_actualu_trace_{count:05d}.npz",**chosen)
        runner.env._c2_pending_controls=[]
        return result

    def control_target(mdp,action,memory,bias):
        residual,actual_u=original_control_target(mdp,action,memory,bias)
        actual_u=target_inset(actual_u,mdp.target_lower,mdp.target_upper,MARGIN_RAD)
        memory["frozen_target"]=actual_u
        return residual,actual_u

    def save(runner,path,next_iteration=None):
        # Native save builds its complete state and writes atomically. Insert
        # metadata at that serialization boundary, before RecordedRunner hashes
        # the committed .pt. Do not rewrite a completed checkpoint afterward.
        native_torch_save=torch.save
        descriptor={**runner.mdp.c2_controller_metadata,
            "target_lower_rad":runner.mdp.target_lower.detach().cpu().tolist(),
            "target_upper_rad":runner.mdp.target_upper.detach().cpu().tolist(),
            "initial_parameter_fingerprints":getattr(runner,"c2_initial_parameter_fingerprints",None),
            "latest_actualu_evidence":getattr(runner,"c2_actualu_evidence",None)}

        def serialize(state,*args,**kwargs):
            if not isinstance(state,dict) or "modules" not in state or "format_version" not in state:
                raise ValueError("Unexpected checkpoint serialization shape")
            state={**state,"format_version":C2_FORMAT_VERSION,"c2_controller":descriptor}
            return native_torch_save(state,*args,**kwargs)
        torch.save=serialize
        try:
            return original_save(runner,path,next_iteration)
        finally:
            torch.save=native_torch_save

    def no_resume(*args,**kwargs):
        raise ValueError("C2 training is from scratch only: replay/environment state is not saved for exact resume; controller formats cannot be silently mixed")

    env_class.__init__=env_init
    env_class.step=env_step
    mdp_class.__init__=mdp_init
    mdp_class.control_target=control_target
    runner_class.save=save
    if original_collect is not None:
        runner_class.collect=collect
    runner_class.load=no_resume
    runner_class.load_model_only=no_resume
    try:
        yield
    finally:
        env_class.__init__=original_env_init
        env_class.step=original_env_step
        mdp_class.__init__=original_mdp_init
        mdp_class.control_target=original_control_target
        runner_class.save=original_save
        if original_collect is not None:
            runner_class.collect=original_collect
        runner_class.load=original_load
        runner_class.load_model_only=original_model_load

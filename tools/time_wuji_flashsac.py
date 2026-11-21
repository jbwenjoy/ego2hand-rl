"""Instrument the shared native learner without changing its transitions or update kernel."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import runpy
import sys
import time


class BudgetMet(Exception):
    """The current native learner, replay and auxiliary state have all been saved."""


def main():
    from flash_chord.training.flash_sac import runner
    from flash_chord.training.flash_sac.replay import ReplaySpec, replay_memory

    out = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
    start = time.perf_counter()
    segment_seconds = float(os.environ.get("WUJI_SEGMENT_WARM_SECONDS", "inf"))
    warm_after = int(os.environ.get("WUJI_SEGMENT_WARM_AFTER_STEP", "0"))
    warm_start = None
    checkpoint_start = None
    checkpoint_seconds = 0.0
    checkpoint_counters = {}

    def event(name, **values):
        with (out / "timing.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(event=name, wall_s=time.perf_counter() - start, **values),
                               allow_nan=False) + "\n")

    original_run = runner.run_training
    def timed_run(env, config, **kwargs):
        scheme=os.environ.get("WUJI_PHYSICS_SCHEME")
        if scheme:
            import numpy as np
            verified=[]
            for shard in env.shards:
                solver=shard.env.solver
                receipt=solver.wuji_conditioning
                installed=solver.mjw_model.stat.meaninertia.numpy()
                if receipt["scheme"]!=scheme or not np.allclose(installed, receipt["conditioned"],rtol=1e-6,atol=0):
                    raise ValueError("The active shard solver does not match its conditioned physics receipt")
                verified.append(dict(scheme=scheme,metadata=receipt,actual_meaninertia=float(installed.ravel()[0])))
            event("verified_solver_conditioning",shards=verified)
        spec = ReplaySpec.from_config(config.replay, discount=config.discount, world_count=env.world_count,
                                      observation_dim=env.observation_dim, action_dim=env.action_dim,
                                      objective_term_count=len(env.objective_term_names),
                                      critic_context_dim=env.critic_context_dim)
        event("native_rl_env_ready", spec=asdict(spec), logical_replay=asdict(replay_memory(spec)))
        logger = kwargs["logger"]
        original_log = logger.log
        def paired_log(metrics, step):
            nonlocal warm_start
            event("synchronized_log", step=step)
            if os.environ.get("WUJI_RECORD_VOC") == "1":
                import numpy as np
                applied = [shard.env.reset_policy.applied_voc_scale.numpy() for shard in env.shards]
                event("next_control_voc_sample", step=step, per_shard=[dict(
                    worlds=len(x),positive=int(np.count_nonzero(x>0)),full=int(np.count_nonzero(x==1)),
                    mean=float(np.mean(x)),minimum=float(np.min(x)),maximum=float(np.max(x))) for x in applied])
            if step >= warm_after and warm_start is None:
                warm_start = time.perf_counter()
                event("budget_warm_start", step=step)
            return original_log(metrics, step)
        logger.log = paired_log
        original_checkpoint = logger.log_checkpoint
        def paired_checkpoint(paths, **values):
            nonlocal checkpoint_seconds
            event("sidecar_checkpoint_start")
            result = original_checkpoint(paths, **values)
            event("sidecar_checkpoint_end")
            if warm_start is not None and checkpoint_start is not None:
                checkpoint_seconds += time.perf_counter() - checkpoint_start
                elapsed = time.perf_counter() - warm_start - checkpoint_seconds
                step = int(values["environment_steps"])
                event("checked_training_budget", step=step, warm_s=elapsed)
                if elapsed >= segment_seconds and step < config.total_environment_steps:
                    paired = paths.training_state.parent / f"paired_{step}.json"
                    with paired.open() as receipt:
                        identity = json.load(receipt)
                    if identity["environment_steps"] != step:
                        raise ValueError("Stopped checkpoint is not complete")
                    raise BudgetMet(dict(learner_steps=step, **checkpoint_counters,
                        hybrid=identity["hybrid"], source_fingerprint=identity["source_fingerprint"],
                        auxiliary=identity["auxiliary_metrics"], stopped_at_complete_checkpoint=True,
                        warm_s_at_stop=elapsed, world_count=env.world_count))
            return result
        logger.log_checkpoint = paired_checkpoint
        return original_run(env, config, **kwargs)

    original_compile = runner._compile_executables
    def timed_compile(*args, **kwargs):
        event("xla_compile_start")
        compiled = original_compile(*args, **kwargs)
        event("xla_compile_end")
        return compiled

    original_save = runner.save_learner_checkpoint
    def timed_save(*args, **kwargs):
        nonlocal checkpoint_start
        import jax
        checkpoint_start = time.perf_counter()
        learner = args[1]
        checkpoint_counters.update(optimizer_updates=int(jax.device_get(learner.global_update_step)),
             native_sac_actor_updates=int(jax.device_get(learner.actor.schedule_step)),
             native_sac_critic_updates=int(jax.device_get(learner.critic.schedule_step)),
             accepted_actor_adam_updates=int(jax.device_get(learner.actor.optimizer_state.count)),
             accepted_critic_adam_updates=int(jax.device_get(learner.critic.optimizer_state.count)))
        event("native_checkpoint_start",**checkpoint_counters)
        paths = original_save(*args, **kwargs)
        event("native_checkpoint_end", state=str(paths.training_state))
        return paths

    runner.run_training = timed_run
    runner._compile_executables = timed_compile
    runner.save_learner_checkpoint = timed_save
    try:
        runpy.run_path(str(Path(__file__).with_name("run_wuji_multiref_flashsac.py")), run_name="__main__")
    except BudgetMet as stop:
        result = stop.args[0]
        result["wall_seconds"] = time.perf_counter() - start
        result["shards"] = json.loads((out / "allocated.json").read_bytes())
        target = out / "result.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2, allow_nan=False))
        temp.replace(target)


if __name__ == "__main__":
    main()

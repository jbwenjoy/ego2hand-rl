"""Specialized control chronology for the declared empty upstream RL signals.

The outer native adapter still owns real reward/termination/explicit resets.
Only the known-empty upstream termination-index query is omitted. Physics,
forward/sense,commands,observations,counters and manager calls stay in order.
"""
import torch


def empty_signal_control_step(env,action):
    from mjlab.managers.termination_manager import TerminationManager
    cfg=env.cfg
    if (cfg.auto_reset or cfg.terminations or cfg.rewards or cfg.metrics or cfg.curriculum or cfg.recorders
            or not isinstance(env.termination_manager,TerminationManager) or env.termination_manager.active_terms
            or any(mode in env.event_manager.available_modes for mode in ['step','interval'])):
        raise ValueError('Empty upstream signals,ordinary termination manager,no automatic reset or step/interval events required')
    # Keep the upstream manual-reset guard;this is not permission to advance a
    # terminal world. The outer adapter's independent pending guard also stays.
    if torch.any(env._manual_reset_pending):
        ids=env._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
        raise RuntimeError(f'Environments {ids.cpu().tolist()} must be reset before stepping')
    env.extras['log']=dict();env.action_manager.process_action(action.to(env.device))
    for _ in range(cfg.decimation):
        env._sim_step_counter+=1;env.action_manager.apply_action();env.scene.write_data_to_sim();env.sim.step();env.scene.update(dt=env.physics_dt);env.metrics_manager.compute_substep()
    env.episode_length_buf+=1;env.common_step_counter+=1
    env.reset_buf=env.termination_manager.compute();env.reset_terminated=env.termination_manager.terminated;env.reset_time_outs=env.termination_manager.time_outs
    env.reward_buf=env.reward_manager.compute(dt=env.step_dt);env.metrics_manager.compute()
    # TerminationManager.compute clears both buffers and has no terms to set
    # them. There are no upstream reset IDs to materialize or mark pending.
    env.sim.forward();env.command_manager.compute(dt=env.step_dt);env.sim.sense();env.obs_buf=env.observation_manager.compute(update_history=True)
    env.recorder_manager.record_post_step()
    return env.obs_buf,env.reward_buf,env.reset_terminated,env.reset_time_outs,env.extras

"""Explicit actor/norm-only initialization into a newly constructed W1 runtime."""
import numpy as np
import torch
from dex_supermimic.rl.dex_supermimic.networks import GaussianPPOActor
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.runtime.gpu_model_snapshot import model_leaves
from supermimic_research.runtime.pen_portable_state import host_model_digest,reference_signature
from supermimic_research.runtime.pen_pose_observation import require_full_pose_contract
from .pen_support_ppo import CENTERED_SCHEMA


def frozen_actor_for_current_W1(runtime,actor_pin,W1_context_pin):
    """No optimizer,critic,native state or RNG from the old checkpoint is loaded.

    Historical source snapshots are verified,not rebound to changed live files.
    Full current W1 Model equality uses the recorded W1 context. W32 derived
    solver constants are not silently assumed equal to W1.
    """
    if runtime.num_envs!=1:raise ValueError('Explicit fresh W1 weights-only context required')
    state=torch.load(verify_file_pin(actor_pin),map_location='cpu',weights_only=False)
    context=torch.load(verify_file_pin(W1_context_pin),map_location='cpu',weights_only=False)
    if state['schema']!=CENTERED_SCHEMA or context['schema']!='owned_trained_actor_probe_source:1' or context['actor_checkpoint']!=actor_pin:raise ValueError('Matching frozen actor/W1-context pins required')
    require_full_pose_contract(state['metadata']);verify_source_snapshot(state['native']['sources']);verify_source_snapshot(context['native']['sources'])
    for pin in context['native']['engine_sources']:verify_file_pin(pin)
    if state['metadata']['objective_config']!=file_pin(runtime.objective_config_path) or context['native']['config']!=file_pin(runtime.config_path):raise ValueError('New runtime objective/adapter config differs')
    if list(runtime.motion.joint_names)!=state['metadata']['joint_order']:raise ValueError('Joint order differs')
    old=context['native'];sim=runtime.env.sim
    if old['host_model_digest']!=host_model_digest(sim.mj_model) or old['reference_signature']!=reference_signature(runtime):raise ValueError('W1 topology/reference differs')
    arrays,scalars=model_leaves(sim.wp_model);saved=old['native']['model']
    if scalars!=saved['scalars'] or set(arrays)!=set(saved['arrays']) or any(not np.array_equal(value.numpy(),saved['arrays'][key]) for key,value in arrays.items()):raise ValueError('Full W1 Model differs from frozen context')
    for key in ['observation_schema','action_contract','policy_input_dim','action_dim']:
        if runtime.metadata()[key]!=state['metadata'][key]:raise ValueError('Actor observation/action contract differs '+key)
    runtime.support_weight=float(state['metadata']['support_weight']);cfg=state['config']
    actor=GaussianPPOActor(161,20,init_std=cfg['ppo_init_std'],std_range=(cfg['ppo_std_min'],cfg['ppo_std_max'])).to(runtime.device)
    actor.load_state_dict(state['modules']['actor']);actor.eval()
    return actor,dict(mode='FROZEN_ACTOR_AND_NORMALIZER_ONLY_NEW_W1_RUNTIME',actor_checkpoint=actor_pin,W1_context=W1_context_pin,full_W1_model_arrays=len(arrays),
        full_W1_model_scalar_match=True,historical_snapshots_verified=True,old_native_or_optimizer_loaded=False,
        scope='Fresh current-code W1 source generation; historical native states are never converted or implicitly resumed across source revisions.')

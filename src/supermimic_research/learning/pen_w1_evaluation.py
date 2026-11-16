"""Fresh LIVE W1 evaluation using actor weights and an independent plant witness."""
import torch

from dex_supermimic.rl.dex_supermimic.networks import GaussianPPOActor
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.runtime.gpu_model_snapshot import model_leaves
from supermimic_research.runtime.pen_portable_state import host_model_digest, reference_signature
from supermimic_research.runtime.pen_pose_observation import require_full_pose_contract
from .pen_support_ppo import CENTERED_SCHEMA


def validate_actor_contract(state, metadata, objective_pin, adapter_pin, joint_order, weights):
    """Reject incompatible actor state before copying any weights into a module."""
    saved = state['metadata']
    if state['schema'] != CENTERED_SCHEMA or saved['checkpoint_schema'] != CENTERED_SCHEMA:
        raise ValueError('Explicit centered training4 checkpoint required')
    require_full_pose_contract(saved)
    if not state['normalizers_ready'] or state['native']['worlds'] != 32:
        raise ValueError('Calibrated W32 actor source required')
    if saved['objective_config'] != objective_pin or state['native']['config'] != adapter_pin:
        raise ValueError('Actor objective/adapter config differs')
    if saved['joint_order'] != list(joint_order):
        raise ValueError('Actor joint order differs')
    for key in ['observation_schema', 'action_contract', 'policy_input_dim', 'action_dim']:
        if saved[key] != metadata[key]:
            raise ValueError('Actor contract differs: ' + key)
    if saved['policy_input_dim'] != 161 or saved['action_dim'] != 20:
        raise ValueError('Registered policy161/action20 required')
    if saved['arm'] not in weights or saved['support_weight'] != weights[saved['arm']]:
        raise ValueError('Registered arm/support weight differs')
    values = state['modules']['actor']
    if not values or any(not bool(torch.isfinite(v).all()) for v in values.values()):
        raise ValueError('Actor state is nonfinite')
    if int(values['norm.count']) < 32:
        raise ValueError('Nonempty frozen actor normalizer required')


def validate_W1_model(arrays, scalars, saved):
    if scalars != saved['scalars'] or set(arrays) != set(saved['arrays']):
        raise ValueError('Full W1 plant keys/scalars differ')
    for key, value in arrays.items():
        current = value.numpy()
        old = saved['arrays'][key]
        if current.shape != old.shape or current.dtype != old.dtype or current.tobytes() != old.tobytes():
            raise ValueError('Full W1 plant array differs: ' + key)


@torch.no_grad()
def actor_for_fresh_live_W1(runtime, state, actor_pin, plant_context_pin):
    """The historical context supplies Model constants only, never physical state.

    Its old actor need not match the new actor. This is an explicit independent
    W1 plant compatibility witness, not a source-state or optimizer migration.
    """
    if runtime.num_envs != 1 or runtime.owned_native_integrations != 0:
        raise ValueError('Fresh unstepped W1 runtime required')
    runtime.set_fixed_frame_features(False)
    if runtime.metadata()['fixed_frame_features'] != 'LIVE':
        raise ValueError('W1 evaluation requires LIVE features')
    verify_file_pin(actor_pin)
    validate_actor_contract(state, runtime.metadata(), file_pin(runtime.objective_config_path),
                            file_pin(runtime.config_path), runtime.motion.joint_names,
                            runtime.objective_cfg['support_weights'])
    context = torch.load(verify_file_pin(plant_context_pin), map_location='cpu', weights_only=False)
    if context['schema'] != 'owned_trained_actor_probe_source:1' or context['native']['worlds'] != 1:
        raise ValueError('Historical W1 plant witness required')
    verify_source_snapshot(state['native']['sources'], live=True)
    verify_source_snapshot(context['native']['sources'])
    for native in [state['native'], context['native']]:
        for pin in native['engine_sources']:
            verify_file_pin(pin)
        if native['config'] != file_pin(runtime.config_path):
            raise ValueError('Plant adapter config differs')
        if native['host_model_digest'] != host_model_digest(runtime.env.sim.mj_model):
            raise ValueError('Plant host topology differs')
        if native['reference_signature'] != reference_signature(runtime):
            raise ValueError('Plant reference differs')
    arrays, scalars = model_leaves(runtime.env.sim.wp_model)
    validate_W1_model(arrays, scalars, context['native']['native']['model'])
    cfg = state['config']
    actor = GaussianPPOActor(161, 20, init_std=cfg['ppo_init_std'],
                             std_range=(cfg['ppo_std_min'], cfg['ppo_std_max'])).to(runtime.device)
    actor.load_state_dict(state['modules']['actor'], strict=True)
    actor.eval()
    if any(not torch.equal(v.cpu(), state['modules']['actor'][k]) for k, v in actor.state_dict().items()):
        raise ValueError('Actor/normalizer copy readback differs')
    runtime.support_weight = float(state['metadata']['support_weight'])
    return actor, dict(
        mode='FRESH_LIVE_W1_ACTOR_AND_NORMALIZER_ONLY', actor_checkpoint=actor_pin,
        W1_plant_context=plant_context_pin, historical_context_actor=context['actor_checkpoint'],
        full_W1_model_arrays=len(arrays), full_W1_model_scalars=len(scalars),
        full_W1_model_bytes_equal=True, actor_normalizer_bytes_equal=True,
        old_native_optimizer_critic_RNG_counters_loaded=False,
        evaluation_features='LIVE', training_features=state['metadata']['fixed_frame_features'],
        scope='Independent full W1 Model compatibility; historical context actor/physical '
              'state/labels are not imported. New policy is executed from fresh native reset. '
              'Feature mode intentionally LIVE for W1; no W32 Model broadcast or native resume.')

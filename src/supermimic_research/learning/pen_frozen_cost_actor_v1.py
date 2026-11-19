"""Explicit verified old-backend weights/norm transfer into a fresh cost runtime."""
from ..artifacts import verify_file_pin
from ..runtime.explicit_collision_backend import active_backend
from .pen_w1_evaluation import actor_for_fresh_live_W1


def initialize(runtime,state,pin,plant_pin,label,profile):
    profile.require(runtime); collision=active_backend()
    expected=(0,0,0) if label=='INITIAL' else (2000,64000,4096000)
    if label not in ['INITIAL','RESIDUAL','SUPPORT'] or (state['iteration'],state['optimizer_steps'],state['training_transitions'])!=expected:
        raise ValueError('Fixed initial/final2000 actor counters required')
    if state['metadata']['arm']!=('RESIDUAL' if label=='INITIAL' else label): raise ValueError('Declared frozen actor arm differs')
    for metadata in [state['metadata'],state['native']['metadata']]:
        if metadata.get('collision_backend')!=collision.contract or 'termination_backend' in metadata:
            raise ValueError('Exact original-cost/source collision identity required')
    verify_file_pin(pin)
    actor,audit=actor_for_fresh_live_W1(runtime,state,pin,plant_pin)
    actor.requires_grad_(False)
    audit.update(schema='explicit_frozen_actor_to_cost_backend:1',policy_label=label,
        source_numeric_backend='INSTALLED_ORIGINAL_OR',target_backend=profile.contract,
        collision_binding_verified=collision.verify(),new_native_state_restored=False,
        transfer='ACTOR_PARAMETERS_AND_FROZEN_NORMALIZATION_ONLY',training_or_adaptation=False)
    return actor,audit

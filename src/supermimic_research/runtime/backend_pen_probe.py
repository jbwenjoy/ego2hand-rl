"""Same-backend probe envelopes and guards before any native-state writes."""
from .explicit_collision_backend import active_backend
from supermimic_research.learning.backend_pen_ppo import require_saved_backend

SCHEMA='owned_backend_actor_probe_source:1'
ROLES={'TRAIN_CONTROLLED_PHYSICAL_RESPONSE','VALIDATE_CONTROLLED_PHYSICAL_RESPONSE_NEVER_FIT'}


def require_backend_probe(source,contract):
    if source.get('schema')!=SCHEMA or source.get('role') not in ROLES:
        raise ValueError('Explicit new-backend TRAIN/NEVER_FIT probe source required')
    if source.get('collision_backend')!=contract or source.get('actor_budget')!=(2000,64000,4096000):
        raise ValueError('Final2000 actor and exact source backend required')
    if source.get('actor') not in ['RESIDUAL','SUPPORT'] or source['native']['worlds']!=1 or source['native']['metadata'].get('fixed_frame_features')!='LIVE':
        raise ValueError('New-backend LIVE W1 actor source required')
    # Native restore alone accepts mutable owned scalars. Reject a hidden backend
    # mutation BEFORE invoking that loader, including when outer metadata agrees.
    require_saved_backend(dict(metadata=dict(collision_backend=source['collision_backend']),native=source['native']),contract)
    expected=0. if source['actor']=='RESIDUAL' else .2
    path=(('attr','support_weight'),)
    value=source['native']['python']['scalars'].get(path)
    if type(value) is not float or value!=expected:raise ValueError('Probe actor/support-weight state mismatch')


def restore_backend_probe(runtime,source):
    contract=runtime._collision_backend_contract
    backend=active_backend(contract);backend.verify(cold=True)
    require_backend_probe(source,contract)
    # Reuse the maintained task/native loader. Only its outer dispatch envelope
    # changes; native metadata, owned bytes and source/engine receipts do not.
    from .pen_owned_probe_recorder import restore_source
    envelope=dict(source,schema='owned_trained_actor_probe_source:1')
    return restore_source(runtime,envelope)

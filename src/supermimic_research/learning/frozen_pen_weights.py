"""Historical W32 NN/norm initialization; never a native/optimizer resume."""
import copy
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot
from supermimic_research.runtime.gpu_model_snapshot import model_leaves
from supermimic_research.runtime.pen_portable_state import host_model_digest,reference_signature
from supermimic_research.runtime.pen_pose_observation import require_full_pose_contract
from .pen_support_ppo import CENTERED_SCHEMA,FROZEN_WEIGHTS_SCHEMA
from .native_pen_ppo import cpu_tree,tensor_tree_equal


def validate_full_model(arrays,scalars,saved):
    if scalars!=saved['scalars'] or arrays.keys()!=saved['arrays'].keys():
        raise ValueError('Full W32 Model keys/scalars differ')
    for key,value in arrays.items():
        a=value.numpy();b=saved['arrays'][key]
        if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
            raise ValueError('Full W32 Model differs: '+key)


@torch.no_grad()
def initialize_frozen_W32_weights(learner,checkpoint_pin):
    runtime=learner.adapter
    if runtime.num_envs!=32 or any(getattr(learner,k)!=0 for k in ['iteration','optimizer_steps','training_transitions','calibration_transitions']) or learner.normalizers_ready or learner.memory is not None or learner.optimizer.state:
        raise ValueError('Fresh uncalibrated zero-work W32 learner required')
    state=torch.load(verify_file_pin(checkpoint_pin),map_location='cpu',weights_only=False)
    old=state['native'];metadata=learner.metadata();metadata.pop('fixed_frame_features',None)
    saved_metadata=copy.deepcopy(state['metadata']);saved_metadata.pop('fixed_frame_features',None)
    if state['schema']!=CENTERED_SCHEMA or old['worlds']!=32 or not state['normalizers_ready'] or state['iteration']!=64 or state['optimizer_steps']!=2048:
        raise ValueError('Explicit trained64 W32 training4 source required')
    require_full_pose_contract(state['metadata'])
    if metadata!=saved_metadata or vars(learner.cfg)!=state['config']:
        raise ValueError('Frozen NN objective/action/joints/reference/normalization/config differs')
    if old['config']!=file_pin(runtime.config_path) or state['metadata']['objective_config']!=file_pin(runtime.objective_config_path):
        raise ValueError('Frozen W32 objective/adapter config differs')
    verify_source_snapshot(old['sources'])
    for pin in old['engine_sources']:verify_file_pin(pin)
    if old['host_model_digest']!=host_model_digest(runtime.env.sim.mj_model) or old['reference_signature']!=reference_signature(runtime):
        raise ValueError('Frozen W32 topology/reference differs')
    arrays,scalars=model_leaves(runtime.env.sim.wp_model)
    validate_full_model(arrays,scalars,old['native']['model'])
    # Validate BOTH complete modules before copying into live NN storage.
    for name in ['actor','critic']:
        current=getattr(learner,name).state_dict();saved=state['modules'][name]
        if current.keys()!=saved.keys() or any(current[k].shape!=v.shape or current[k].dtype!=v.dtype or not bool(torch.isfinite(v).all()) for k,v in saved.items()):
            raise ValueError('Frozen NN module layout/finite mismatch '+name)
    counts={name:int(state['modules'][name]['norm.count']) for name in ['actor','critic']}
    if counts['actor']!=counts['critic'] or counts['actor']<32:raise ValueError('Matching nonempty frozen normalizer history required')
    for name in ['actor','critic']:getattr(learner,name).load_state_dict(state['modules'][name])
    learner.normalizer_seed=dict(source=checkpoint_pin,actor_count=counts['actor'],critic_count=counts['critic'])
    learner.checkpoint_schema=FROZEN_WEIGHTS_SCHEMA
    learner.normalizers_ready=True
    for name in ['actor','critic']:
        if not tensor_tree_equal(cpu_tree(getattr(learner,name).state_dict()),state['modules'][name]):raise ValueError('Frozen NN readback mismatch')
    return dict(mode='FROZEN_W32_ACTOR_CRITIC_AND_NORMALIZERS_ONLY',checkpoint=checkpoint_pin,
        full_model_arrays=len(arrays),full_model_scalars=len(scalars),full_model_bytes_equal=True,
        historical_snapshots_verified=True,modules_exact=True,normalizers_ready=True,
        new_checkpoint_schema=FROZEN_WEIGHTS_SCHEMA,normalizer_seed=learner.normalizer_seed,
        old_native_optimizer_RNG_counters_loaded=False,
        scope='Fresh current-code W32 physics/observations/empty Adam/counters/RNG;only trained64 actor,critic and embedded normalizer state initialized. Not native resume or policy promotion.')

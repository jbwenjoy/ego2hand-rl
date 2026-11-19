"""Small recorded audit inputs for continuous evaluation;never a resume state."""
import numpy as np
import torch
from ..mechanics.pen_reward_chain import A,ACTION,MOTION,ROBOT,CACHE,source_context
from .pen_pose_observation import pose9_world_in_wrist,pack_full_pose_policy

SCHEMA='pen_continuous_evaluation_numeric_view:1'


def record_dynamic(runtime):
    d=runtime.env.sim.wp_data
    return dict(phase=int(runtime.motion.time_steps[0]),
        arrays={k:getattr(d,k).numpy().copy() for k in ['qpos','qvel','qacc_warmstart','time']},
        cached_R=d.xmat.numpy()[0,runtime.features.body].copy(),mount=runtime.features.mount().detach().cpu().clone(),
        EMA=runtime.action._processed_actions.detach().cpu().clone(),last_raw=runtime.action._raw_actions.detach().cpu().clone(),
        trajectory_ids=runtime.motion.trajectory_ids.detach().cpu().clone())


def numeric_view(initial,dynamic):
    if initial.get('schema')!='controlled_pen_owned_state:4' or initial.get('worlds')!=1 or initial['metadata'].get('fixed_frame_features')!='LIVE':
        raise ValueError('Actual initial full native4 LIVE W1 capture required')
    if type(dynamic['phase']) is not int or not 1<=dynamic['phase']<=368: raise ValueError('Original positive interval phase required')
    if dynamic['EMA'].shape!=(1,20) or dynamic['last_raw'].shape!=(1,20) or dynamic['mount'].shape!=(1,7) or dynamic['trajectory_ids'].tolist()!=[0]:
        raise ValueError('Exact recorded control/W1/single-reference layout required')
    arrays=dynamic['arrays']
    if any(np.asarray(arrays[k]).shape!=shape for k,shape in [('qpos',(1,27)),('qvel',(1,26)),('qacc_warmstart',(1,26)),('time',(1,))]):
        raise ValueError('Exact recorded state prefix required')
    if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError('Finite numeric audit state required')
    if not all(bool(torch.isfinite(dynamic[k]).all()) for k in ['EMA','last_raw','mount']) or not np.isfinite(dynamic['cached_R']).all():
        raise ValueError('Finite controller/frame audit inputs required')
    t=dict(initial['python']['tensors'])
    t[ACTION+(A('_processed_actions'),)]=dynamic['EMA'].clone()
    t[ACTION+(A('_raw_actions'),)]=dynamic['last_raw'].clone()
    t[MOTION+(A('time_steps'),)]=torch.tensor([dynamic['phase']],dtype=torch.long)
    t[MOTION+(A('trajectory_ids'),)]=dynamic['trajectory_ids'].clone()
    # Static tensors/model borrow the initial read-only capture. This reduced
    # object deliberately has a distinct schema and cannot pass a native loader.
    return dict(schema=SCHEMA,python=dict(tensors=t,scalars=initial['python']['scalars']),
        native=dict(model=initial['native']['model'],data=dict(arrays={k:np.asarray(v).copy() for k,v in arrays.items()})))


def context(initial,dynamic,case):
    return source_context(dict(native=numeric_view(initial,dynamic),phase=dynamic['phase']),case)


def physics_view(initial,dynamic,case):
    from scipy.spatial.transform import Rotation
    state=numeric_view(initial,dynamic); qa=case['model_layout']['object_qposadr']
    return dict(native=state,phase=dynamic['phase'],source_qpos_R=Rotation.from_quat(state['native']['data']['arrays']['qpos'][0,qa+np.array([4,5,6,3])]).as_matrix(),source_cached_R=np.asarray(dynamic['cached_R']))


def policy(initial,dynamic,case):
    state=numeric_view(initial,dynamic); t=state['python']['tensors']; phase=dynamic['phase']
    ids=t[ACTION+(A('_target_ids'),)].long(); q=torch.from_numpy(state['native']['data']['arrays']['qpos']).float(); v=torch.from_numpy(state['native']['data']['arrays']['qvel']).float()
    refq=t[MOTION+(A('group'),A('joint_pos'))][phase:phase+1]
    links=t[MOTION+(A('group'),A('link_pos'))][phase:phase+1].flatten(1)
    refpose=t[CACHE+(A('world_pose'),)][:,phase]; mount=dynamic['mount'].float(); qa=case['model_layout']['object_qposadr']
    relative=q[:,case['qids']]+t[ROBOT+(A('encoder_bias'),)][:,ids]-t[ROBOT+(A('default_joint_pos'),)][:,ids]
    velocity=(v[:,case['dofs']]-t[ROBOT+(A('default_joint_vel'),)][:,ids])*.25
    return pack_full_pose_policy(refq,pose9_world_in_wrist(refpose,mount),links,relative,velocity,pose9_world_in_wrist(q[:,qa:qa+7],mount),dynamic['last_raw'])


def terminal_reason(physical_valid,terminated,truncated,phase):
    if not physical_valid: return 'PHYSICAL_AUDIT_STOP'
    if terminated: return 'TASK_FAILURE'
    if truncated and phase==368: return 'REFERENCE_COMPLETION'
    return None

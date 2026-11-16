"""Recorded controller/reward reconstruction; no simulator or optimizer calls."""
import numpy as np
import torch
from dex_supermimic.rl.dex_supermimic.rotations import mat_to_quat
from ..runtime.pen_native_policy import tracking_reward
from ..learning.pen_diagnostics import decode_pen_torch
from .impulse_codec import rotation55_torch
from .pen_support_objective import objective_terms
from .support_score import support_score

A=lambda name:('attr',name)
I=lambda name:('item',name)
ACTION=(A('env'),A('action_manager'),A('_terms'),I('joint_pos'))
MOTION=(A('env'),A('command_manager'),A('_terms'),I('motion'))
ROBOT=(A('env'),A('scene'),A('_entities'),I('robot'),A('_data'))
CACHE=(A('env'),A('_native_full_pose_features'),A('reference_cache'))
CHANNELS=['COM','SO3','hand','action_penalty','tracking','geometry_penalty',
          'support_bonus','residual_objective','support_objective','source_objective']


def controller(raw,context):
    """Original outer clamp,scale,residual clamp,EMA,bias and target limits."""
    applied=raw.clamp(-1,1)
    ema=.5*(.5*applied).clamp(-.6,.6)+.5*context['EMA']
    intended=context['reference_joints']+ema-context['encoder_bias']
    return intended.clamp(min=context['lower'],max=context['upper']),ema,applied


def source_context(source,case):
    p=source['native']['python'];t=p['tensors'];s=p['scalars'];phase=source['phase']
    cfg=(A('env'),A('cfg'),A('actions'),I('joint_pos'))
    if (s[ACTION+(A('_scale'),)],s[cfg+(A('alpha'),)],s[cfg+(A('residual_clip'),I(0))],
        s[cfg+(A('residual_clip'),I(1))],s[ACTION+(A('_mask_enabled'),)])!=(.5,.5,-.6,.6,False):
        raise ValueError('Original unmasked controller contract required')
    if int(t[MOTION+(A('time_steps'),)][0])!=phase or int(t[MOTION+(A('trajectory_ids'),)][0])!=0 or t[MOTION+(A('_trajectory_starts'),)].tolist()!=[0]:
        raise ValueError('Exact single-reference source phase required')
    ids=t[ACTION+(A('_target_ids'),)].long();aids=t[ACTION+(A('_position_ctrl_ids'),)].numpy()
    if aids.tolist()!=case['aids']:raise ValueError('Transmission-resolved control order differs')
    model=source['native']['native']['model']['arrays'];limited=t[ACTION+(A('_position_ctrl_limited'),)]
    if not np.array_equal(limited.numpy(),model['actuator_ctrllimited'][aids]):raise ValueError('Control-limit flags differ')
    hard=t[ROBOT+(A('joint_pos_limits'),)][0,ids]
    bounds=torch.tensor(model['actuator_ctrlrange'][0,aids],dtype=hard.dtype)
    lower=torch.where(limited,torch.maximum(hard[:,0],bounds[:,0]),hard[:,0])
    upper=torch.where(limited,torch.minimum(hard[:,1],bounds[:,1]),hard[:,1])
    if bool((lower>upper).any()):raise ValueError('Empty target bounds')
    return dict(EMA=t[ACTION+(A('_processed_actions'),)][0].clone(),reference_joints=t[MOTION+(A('group'),A('joint_pos'))][phase].clone(),
        encoder_bias=t[ROBOT+(A('encoder_bias'),)][0,ids].clone(),lower=lower,upper=upper,
        reference_pose=t[CACHE+(A('world_pose'),)][0,phase].clone(),
        target_world=t[(A('support_table'),A('targets'))][0,phase-1].clone(),
        target_valid=bool(t[(A('support_table'),A('valid'))][phase-1]),
        source_weight=s[(A('support_weight'),)],phase=phase,frames=len(t[MOTION+(A('group'),A('joint_pos'))]))


def native_components(case,trace,window,context,reward_cfg,objective_cfg):
    tensor=lambda a:torch.as_tensor(a)
    raw=tensor(trace['actions20']).float();PD,ema,applied=controller(raw,context)
    qa=case['model_layout']['object_qposadr'];q=tensor(trace['post_qpos'][-1:]).float()
    tracking,parts=tracking_reward(q[:,qa:qa+3],q[:,qa+3:qa+7],q[:,case['qids']],
        context['reference_pose'][None],context['reference_joints'][None],applied,reward_cfg)
    term=(parts['COM_error_m']>.05)|(parts['shaft_angle_rad']>np.pi/3)|~torch.isfinite(q).all(-1)
    terms=objective_terms(tracking,term,tensor(window['geometry_max'])[None],tensor(window['nonhand_max_sum'])[None],
        tensor(window['errors']).reshape(1),tensor(window['global_error']).reshape(1),tensor(window['wrench_sum'])[None],
        tensor(window['counts'])[None],tensor(window['source_R'])[None],context['target_world'][None],
        torch.tensor([context['target_valid']]),torch.tensor([1<=context['phase']<context['frames']]),objective_cfg,context['source_weight'])
    recorded_tracking=tensor(window['objective_tracking_reward']).float()
    geometry=(-objective_cfg['geometry_soft_weight']*tensor(window['objective_geometry_loss']).float()
              -objective_cfg['geometry_failure_penalty']*tensor(window['objective_geometry_failure']).float())
    bonus=tensor(window['objective_support_bonus']).float()
    residual=recorded_tracking+geometry;support=residual+.2*bonus
    values=torch.stack([reward_cfg['COM_weight']*parts['COM'][0],reward_cfg['angle_weight']*parts['full_SO3'][0],
        reward_cfg['hand_weight']*parts['hand'][0],-reward_cfg['action_weight']*parts['action'][0],
        recorded_tracking,geometry,bonus,residual,support,residual+context['source_weight']*bonus]).double().numpy()
    actual_PD=torch.tensor(case['actual_PD20'],dtype=torch.float64)
    errors=dict(PD_max_rad=float(abs(PD[0].double()-actual_PD).max()),
        tracking_abs=float(abs(tracking[0]-recorded_tracking)),
        geometry_abs=float(abs(terms['geometry_loss'][0]-tensor(window['objective_geometry_loss']))),
        support_bonus_abs=float(abs(terms['support_bonus'][0]-bonus)),
        support_loss_abs=float(abs(terms['support_loss'][0]-tensor(window['objective_support_loss']))),
        target_source_abs=float(abs(terms['target_source'][0]-tensor(window['objective_target_source'])).max()),
        measured_source_abs=float(abs(terms['measured_source'][0]-tensor(window['objective_measured_source'])).max()))
    checks=dict(actual_PD=errors['PD_max_rad']<=2e-7,held_PD=bool(torch.equal(tensor(trace['post_ctrl'])[:,case['aids']],actual_PD.expand(8,-1))),
        tracking=errors['tracking_abs']<=5e-6,geometry=errors['geometry_abs']<=2e-6,
        support_bonus=errors['support_bonus_abs']<=2e-6,
        support_loss=errors['support_loss_abs']<=1e-8+1e-10*abs(float(window['objective_support_loss'])),
        target_source=errors['target_source_abs']<=1e-10,measured_source=errors['measured_source_abs']<=1e-10,
        target_world=bool(torch.equal(context['target_world'],tensor(window['objective_target_world']))),
        target_valid=context['target_valid']==bool(window['objective_target_valid']),
        native_termination=bool(term[0])==bool(window['objective_native_terminated']),
        geometry_failure=bool(terms['geometry_failure'][0])==bool(window['objective_geometry_failure']),
        support_eligible=bool(terms['support_eligible'][0])==bool(window['objective_support_eligible']),
        no_force_aid=not np.any(trace['post_qfrc_applied']) and not np.any(trace['post_xfrc_applied']))
    return values,checks,errors


def observable_proxy(x,prediction,raw,context,reward_cfg,objective_cfg):
    """Endpoint/mean-wrench observable proxy ONLY;window geometry is unmodelled."""
    next_state=decode_pen_torch(x[...,:55].double(),prediction.double());R=rotation55_torch(next_state)
    quat=mat_to_quat(R);applied=raw.clamp(-1,1)
    tracking,parts=tracking_reward(next_state[...,40:43],quat,next_state[...,:20],
        context['reference_pose'].to(x.device,dtype=torch.float64).reshape(-1,7),context['reference_joints'].to(x.device,dtype=torch.float64).reshape(-1,20),
        applied,reward_cfg)
    source_R=rotation55_torch(x[...,:55].double()).transpose(-1,-2)
    world=context['target_world'].to(x.device,dtype=torch.float64)
    target=torch.cat([(source_R@world[...,:3,None]).squeeze(-1),(source_R@world[...,3:,None]).squeeze(-1)],-1)
    eligible=torch.as_tensor(context['target_valid'],device=x.device)&(parts['COM_error_m']<=.05)&(parts['shaft_angle_rad']<=np.pi/3)
    score=support_score(prediction[...,78:84],target,eligible,objective_cfg['force_scale_N'],objective_cfg['torque_scale_Nm'],'CENTERED')['score']
    return dict(observable_residual=tracking,observable_support=tracking+.2*score,support_bonus=score)

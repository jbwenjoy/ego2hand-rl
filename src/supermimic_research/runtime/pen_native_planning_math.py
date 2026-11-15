"""Declared finite actions, original control mapping and device-only scoring."""
import numpy as np
import torch


def candidate_offsets(ncontrols=368,seed=8101,*,multiplier=1):
    if ncontrols!=368 or seed!=8101:raise ValueError('Original368-control clock and fixed candidate seed')
    if isinstance(multiplier,bool) or multiplier not in [1,4]:raise ValueError('Predeclared amplitude multiplier1 or4')
    axes=.08*np.eye(20,dtype=np.float32)
    fixed=np.concatenate([np.zeros((1,20),np.float32),axes,-axes])
    random=np.random.default_rng(seed).normal(0,.04,(ncontrols,16,20)).clip(-.08,.08).astype(np.float32)
    offsets=np.concatenate([np.broadcast_to(fixed,(ncontrols,41,20)),random],axis=1).copy()
    return offsets if multiplier==1 else offsets*np.float32(multiplier)


def candidate_targets(raw,processed,reference,bias,lower,upper):
    ema=.5*(.5*raw).clamp(-.6,.6)+.5*processed
    intended=reference+ema-bias
    return intended.clamp(min=lower,max=upper),ema,intended


def candidate_sequence_targets(raw,processed,reference,bias,lower,upper):
    """raw[N,H,20], future reference[H,20], copied initial controller EMA."""
    if raw.ndim!=3 or raw.shape[-1]!=20 or reference.shape!=(raw.shape[1],20) or raw.shape[1]<1:
        raise ValueError('Explicit future raw actions and per-control reference')
    targets=[];memories=[];intended=[];ema=processed
    for j in range(raw.shape[1]):
        target,ema,desired=candidate_targets(raw[:,j],ema,reference[j],bias,lower,upper)
        targets.append(target);memories.append(ema);intended.append(desired)
    return tuple(torch.stack(v,dim=1) for v in [targets,memories,intended])


def select_candidate(total,feasible,violation):
    clean=torch.nan_to_num(total,nan=1e30,posinf=1e30,neginf=1e30)
    inf=torch.full_like(clean,float('inf'))
    best=torch.where(feasible,clean,inf).argmin()
    fallback=torch.where(violation==violation.min(),clean,inf).argmin()
    return torch.where(feasible.any(),best,fallback)


def quaternion_distance_wxyz(actual,reference):
    """Sign-invariant SO(3) angle; invalid zero/nonfinite inputs stay invalid.

    Normalization handles finite nonunit inputs. atan2 of the relative
    quaternion avoids acos precision loss at matching orientations. This
    finite-action score is not a qualification of learned contact gradients.
    """
    if actual.shape[-1]!=4 or reference.shape[-1]!=4:
        raise ValueError('Explicit WORLD quaternion wxyz with four components')
    a=actual/torch.linalg.vector_norm(actual,dim=-1,keepdim=True)
    b=reference/torch.linalg.vector_norm(reference,dim=-1,keepdim=True)
    a,b=torch.broadcast_tensors(a,b)
    vector=b[...,:1]*a[...,1:]-a[...,:1]*b[...,1:]-torch.linalg.cross(a[...,1:],b[...,1:],dim=-1)
    scalar=(a*b).sum(-1).abs()
    return 2*torch.atan2(torch.linalg.vector_norm(vector,dim=-1),scalar)


def score_candidates(qpos,qvel,ctrl,wrench,penalties,other_force,overflow,ema,
                     qids,qa,reference_COM,reference_shaft,reference_hand,target_wrench,
                     wrench_valid,support_weight,cfg,*,object_da=None,reference_velocity=None,
                     velocity_valid=False,velocity_weight=0.,reference_quaternion=None):
    q=qpos[:,-1];quat=q[:,qa+3:qa+7];w,x,y,z=quat.unbind(-1)
    shaft=torch.stack([1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)],dim=-1)
    angle=torch.acos((shaft*reference_shaft).sum(-1).clamp(-1,1))
    position=((q[:,qa:qa+3]-reference_COM)/cfg['COM_scale_m']).square().sum(-1)
    objective=cfg.get('orientation_objective','SHAFT')
    if objective not in ['SHAFT','FULL_SO3']:raise ValueError('Explicit SHAFT or FULL_SO3 objective')
    orientation_angle=angle
    if objective=='FULL_SO3':
        if reference_quaternion is None:raise ValueError('Full SO3 needs explicit WORLD reference quaternion wxyz')
        orientation_angle=quaternion_distance_wxyz(quat,reference_quaternion)
    orientation=(orientation_angle/np.deg2rad(cfg['shaft_scale_deg'])).square()
    hand=((q[:,qids]-reference_hand)/cfg['hand_scale_rad']).square().mean(-1)
    residual=(ema/cfg['residual_scale_rad']).square().mean(-1)
    state=position+orientation+cfg['hand_weight']*hand+cfg['residual_weight']*residual
    mean=wrench.mean(1)
    support=((mean[:,:3]-target_wrench[:3])/cfg['force_scale_N']).square().mean(-1)+((mean[:,3:]-target_wrench[3:])/cfg['torque_scale_Nm']).square().mean(-1)
    total=state+support_weight*wrench_valid*support
    velocity=q.new_zeros(len(q))
    if velocity_weight and velocity_valid:
        if object_da is None or reference_velocity is None:raise ValueError('Explicit WORLD free-object dof address and endpoint velocity required')
        velocity=((qvel[:,-1,object_da:object_da+3]-reference_velocity)/cfg['velocity_scale_m_s']).square().sum(-1)
        total=total+velocity_weight*velocity
    bounds=q.new_tensor([cfg['max_penetration_m'],cfg['max_self_penetration_m'],cfg['max_joint_limit_excursion_rad']])
    geometric=((penalties-bounds).clamp_min(0)/bounds).sum(-1)
    nonhand=((other_force.max(1).values-cfg['max_object_nonhand_normal_force_N']).clamp_min(0)/cfg['max_object_nonhand_normal_force_N'])
    finite=torch.isfinite(qpos).all(dim=(1,2))&torch.isfinite(qvel).all(dim=(1,2))&torch.isfinite(ctrl).all(dim=(1,2))&torch.isfinite(wrench).all(dim=(1,2))&torch.isfinite(total)&torch.isfinite(penalties).all(1)&torch.isfinite(other_force).all(1)
    capacity=~overflow.bool().any(1)
    feasible=finite&capacity&(geometric==0)&(nonhand==0)
    violation=torch.nan_to_num(geometric+nonhand+(~capacity).to(q.dtype)*1e6+(~finite).to(q.dtype)*1e12,nan=1e12,posinf=1e12)
    index=select_candidate(total,feasible,violation)
    return dict(index=index,total=total,state=state,support=support,feasible=feasible,violation=violation,
                finite=finite,capacity=capacity,shaft_error_rad=angle,orientation_error_rad=orientation_angle,mean_world_wrench=mean,velocity=velocity)


def score_sequence(qpos,qvel,ctrl,wrench,penalties,other_force,overflow,ema,qids,qa,
                   reference,support_weight,cfg):
    """Mean unchanged per-control costs; full forecast safety, no host readback."""
    H=ema.shape[1]
    if H<1 or qpos.shape[1]%H or reference['COM'].shape!=(H,3):
        raise ValueError('Whole20ms prediction controls and explicit future references')
    n=qpos.shape[1]//H;rows=[]
    for j in range(H):
        sl=slice(j*n,(j+1)*n)
        rows.append(score_candidates(qpos[:,sl],qvel[:,sl],ctrl[:,sl],wrench[:,sl],penalties,
            other_force[:,sl],overflow[:,sl],ema[:,j],qids,qa,reference['COM'][j],reference['shaft'][j],
            reference['hand'][j],reference['wrench'][j],reference['valid'][j],support_weight,cfg,
            reference_quaternion=reference['quaternion'][j]))
    result={key:torch.stack([r[key] for r in rows]).mean(0)
            for key in ['total','state','support','mean_world_wrench','velocity']}
    for key in ['feasible','finite','capacity']:
        result[key]=torch.stack([r[key] for r in rows]).all(0)
    for key in ['shaft_error_rad','orientation_error_rad']:
        result[key]=torch.stack([r[key] for r in rows]).max(0).values
    bounds=qpos.new_tensor([cfg['max_penetration_m'],cfg['max_self_penetration_m'],cfg['max_joint_limit_excursion_rad']])
    geometric=((penalties-bounds).clamp_min(0)/bounds).sum(-1)
    nonhand=((other_force.max(1).values-cfg['max_object_nonhand_normal_force_N']).clamp_min(0)/cfg['max_object_nonhand_normal_force_N'])
    result['violation']=torch.nan_to_num(geometric+nonhand+(~result['capacity']).to(qpos.dtype)*1e6+
        (~result['finite']).to(qpos.dtype)*1e12,nan=1e12,posinf=1e12)
    result['index']=select_candidate(result['total'],result['feasible'],result['violation'])
    return result

"""Pure GPU reward/termination math for the preregistered pen pilot."""
import torch
from .support_score import support_score


def select_interval(targets,valid,phase,world=None):
    phase_valid=(phase>=1)&(phase<=targets.shape[1]);index=torch.where(phase_valid,phase-1,torch.zeros_like(phase))
    if world is None:world=torch.arange(targets.shape[0],device=phase.device)
    return targets[world,index],phase_valid&valid[index],phase_valid


def objective_terms(tracking,native_term,geometry,nonhand,errors,global_error,wrench_sum,counts,source_R,target_world,target_valid,phase_valid,cfg,weight,limits=None):
    if limits is None:limits=geometry.new_tensor([cfg['object_penetration_m'],cfg['self_penetration_m'],cfg['joint_excursion_rad']])
    failed=(geometry>limits).any(-1)|(nonhand[:,0]>cfg['nonhand_normal_N'])|(errors!=0)|(global_error[0]!=0)|~phase_valid
    geometry_loss=(torch.log1p((geometry/limits).square()).sum(-1)+torch.log1p((nonhand[:,0]/cfg['nonhand_normal_N']).square())).float()
    R=source_R.to(torch.float64).transpose(-1,-2)
    target=torch.cat(((R@target_world[:,:3,None]).squeeze(-1),(R@target_world[:,3:,None]).squeeze(-1)),-1)
    measured=wrench_sum[:,5]/counts[:,0,None].to(torch.float64)
    force=((measured[:,:3]-target[:,:3])/cfg['force_scale_N']).square().mean(-1)
    torque=((measured[:,3:]-target[:,3:])/cfg['torque_scale_Nm']).square().mean(-1)
    support_loss=force+torque;eligible=target_valid&phase_valid&~failed&~native_term
    variant=cfg.get('support_score','ABSOLUTE')
    if variant=='ABSOLUTE':bonus=torch.where(eligible,torch.exp(-.5*support_loss).float(),torch.zeros_like(tracking))
    elif variant=='CENTERED':bonus=support_score(measured,target,eligible,cfg['force_scale_N'],cfg['torque_scale_Nm'],variant,loss=support_loss)['score'].float()
    else:raise ValueError('Production score must be explicitly ABSOLUTE or CENTERED')
    reward=tracking-cfg['geometry_soft_weight']*geometry_loss-cfg['geometry_failure_penalty']*failed.float()+weight*bonus
    return dict(reward=reward,geometry_failure=failed,geometry_loss=geometry_loss,target_source=target,measured_source=measured,
        support_loss=support_loss,support_bonus=bonus,support_eligible=eligible)

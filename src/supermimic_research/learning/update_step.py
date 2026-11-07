"""Shared update primitives; experiment orchestration belongs in config/CLI.

No data discovery, native collector, optimizer, NN construction or update is
performed by importing this module. The same steps serve offline replay and
an online collection/update loop. Nothing silently refits normalizers.
"""
from dataclasses import dataclass
import math
from . import world_model as WM
from .imagination import imagined_actor_objective

@dataclass(frozen=True)
class UpdateConfig:
    gamma:float=.99
    target_critic_tau:float=.01
    imagination_horizon:int=2
    microbatch:int=2
    clip_norm:float=1.
    def __post_init__(self):
        if not 0<=self.gamma<=1 or not 0<=self.target_critic_tau<=1:raise ValueError('Discount and EMA in[0,1]')
        if type(self.imagination_horizon) is not int or self.imagination_horizon<1 or type(self.microbatch) is not int or self.microbatch<1:raise ValueError('Positive integer H/microbatch')
        if not math.isfinite(self.clip_norm) or self.clip_norm<=0:raise ValueError('Positive finite gradient clip')

def make_optimizer(module,*,learning_rate,weight_decay=0.,betas=(.9,.999),epsilon=1e-8):
    import torch
    return torch.optim.AdamW(module.parameters(),lr=learning_rate,weight_decay=weight_decay,betas=betas,eps=epsilon,foreach=False,capturable=False)

def finish_optimizer_step(module,optimizer,*,phase,clip_norm=1.):
    import torch
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()):raise ValueError('All named parameter gradients finite before step')
    pre={name:p.grad.detach().clone() for name,p in module.named_parameters()}
    returned=torch.nn.utils.clip_grad_norm_(module.parameters(),clip_norm,error_if_nonfinite=True,foreach=False)
    post={name:p.grad.detach().clone() for name,p in module.named_parameters()};optimizer.step()
    if not all(torch.isfinite(p).all() for p in module.parameters()):raise ValueError('Updated parameters finite')
    return dict(tag=phase,pre_clip=pre,post_clip=post,clip_returned_norm=returned.detach())

def supervised_WM_step(model,optimizer,batch,norms,scales,physical,*,loss_config=WM.LossConfig(),codec=None,phase='PRE_WM_UPDATE',clip_norm=1.):
    """One real supervision forward/backward/Adam step, no implicit sampling."""
    import torch
    model.train();model.requires_grad_(True);optimizer.zero_grad(set_to_none=True)
    loss,terms,prediction=WM.absolute_loss(model,batch,norms,scales,physical,config=loss_config,codec=codec)
    if not torch.isfinite(loss):raise ValueError('Finite WM loss, no retry/row filtering')
    loss.backward();gradient=finish_optimizer_step(model,optimizer,phase=phase,clip_norm=clip_norm)
    return dict(loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},gradient_snapshot=gradient,prediction=prediction,counts=dict(WM_FW=1,WM_backward=1,WM_optimizer_steps=1))

def matched_actor_critic_step(*,world_model,arms,optimizers,real_batch,mode_rewards,tasks,geometry,codec,norms,physical,wm_step,config=UpdateConfig(),phase='PRE_MATCHED_ROUND'):
    """Update every arm from exactly the supplied common real batch.

    ``real_batch`` keys are x/m/obs/obs1/terminated/reference_timeout. Rewards
    are already indexed per-mode native POST-H rewards. Collection truncation
    is deliberately absent from the bootstrap mask. Caller must supply only
    complete prealive starts; partial/unknown rows are not made valid here.
    H2 is model imagination, not a claim of two native groundtruth windows.
    """
    import torch
    if set(arms)!=set(mode_rewards) or set(arms)!=set(tasks) or set(arms)!=set(optimizers):raise ValueError('Same named arms, rewards, tasks and optimizers')
    B=len(real_batch['x'])
    if B<1 or B%config.microbatch:raise ValueError('Common batch must divide declared microbatch')
    if real_batch['x'].shape!=(B,58) or real_batch['m'].shape!=(B,83):raise ValueError('Phone state58/memory83, not legacy PenMDP')
    if real_batch['terminated'].dtype!=torch.bool or real_batch['reference_timeout'].dtype!=torch.bool:raise ValueError('Typed true absorption and reference timeout')
    world_model.eval();world_model.requires_grad_(False);results={}
    for mode,modules in arms.items():
        actor=modules['actor'];critic=modules['critic'];target_critic=modules['target_critic'];opta=optimizers[mode]['actor'];optc=optimizers[mode]['critic'];task=tasks[mode]
        if mode_rewards[mode].shape!=(B,) or not torch.isfinite(mode_rewards[mode]).all():raise ValueError('Per-mode saved real reward[B], no acquisition-arm-only reward')
        optc.zero_grad(set_to_none=True)
        with torch.no_grad():target=mode_rewards[mode].double()+config.gamma*(~real_batch['terminated']).double()*(~real_batch['reference_timeout']).double()*target_critic(real_batch['obs1']).double()
        value=critic(real_batch['obs']);closs=(value-target.float()).square().mean()
        if not torch.isfinite(closs):raise ValueError('Finite critic TD loss required')
        closs.backward();critic_gradient=finish_optimizer_step(critic,optc,phase=phase+'_'+mode+'_CRITIC',clip_norm=config.clip_norm)
        with torch.no_grad():
            for parameter,slow in zip(critic.parameters(),target_critic.parameters()):slow.lerp_(parameter,config.target_critic_tau)
        opta.zero_grad(set_to_none=True);actor_loss=0.;microbatches=B//config.microbatch
        for start in range(0,B,config.microbatch):
            stop=start+config.microbatch;memory=task.unflatten_memory(real_batch['m'][start:stop])
            objective=imagined_actor_objective(actor=actor,critic=critic,world_model=world_model,task=task,geometry=geometry,codec=codec,norms=norms,physical=physical,initial_state58=real_batch['x'][start:stop],memory=memory,horizon=config.imagination_horizon,gamma=config.gamma,wm_step=wm_step)
            if not torch.isfinite(objective['actor_loss']):raise ValueError('Finite imagined Actor loss required')
            (objective['actor_loss']/microbatches).backward();actor_loss+=float(objective['actor_loss'].detach())/microbatches
        actor_gradient=finish_optimizer_step(actor,opta,phase=phase+'_'+mode+'_ACTOR',clip_norm=config.clip_norm)
        results[mode]=dict(real_TD_loss=float(closs.detach()),imagined_Actor_loss=actor_loss,gradient_snapshots=dict(actor=actor_gradient,critic=critic_gradient),counts=dict(Critic_real_FW=2,Critic_backward=1,Critic_optimizer_steps=1,Actor_FW=microbatches*config.imagination_horizon,WM_imagination_FW=microbatches*config.imagination_horizon,Critic_imagination_FW=microbatches,Actor_backward_microbatches=microbatches,Actor_optimizer_steps=1))
    return results

def arm_states(arms):
    return {mode:dict(stateactor=a['actor'].state_dict(),statecritic=a['critic'].state_dict(),statecritic_target=a['target_critic'].state_dict(),norm=dict(actor=a['actor'].norm.state_dict(),critic=a['critic'].norm.state_dict(),critic_target=a['target_critic'].norm.state_dict())) for mode,a in arms.items()}
def restore_arm_states(arms,saved):
    if set(arms)!=set(saved):raise ValueError('Exact checkpoint arm set')
    for mode,modules in arms.items():
        for key,field in [('actor','stateactor'),('critic','statecritic'),('target_critic','statecritic_target')]:modules[key].load_state_dict(saved[mode][field],strict=True)

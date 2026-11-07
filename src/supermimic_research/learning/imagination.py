"""Ordinary-package Actor/Critic construction and current-only imagination.

No old study discovery, importlib source callback or physical model is used.
The upstream network implementation lives in the repository's dex namespace.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class PolicyConfig:
    obs_dim:int=222
    action_dim:int=20
    critic_empirical_norm:bool=True
    def __post_init__(self):
        if type(self.obs_dim) is not int or self.obs_dim<=0 or self.action_dim!=20:raise ValueError('Positive observation width and Wuji action20')

def make_fresh_actor_critic(*,seed,config=PolicyConfig(),device='cpu',dtype='float32'):
    import torch
    from dex_supermimic.rl.dex_supermimic.networks import Actor,Critic
    if dtype!='float32':raise ValueError('Explicit fresh policy FP32')
    dev=torch.device(device)
    if dev.type not in ['cpu','cuda'] or dev.type=='cuda' and dev.index is None:raise ValueError('CPU or explicit cuda:index required')
    previous_dtype=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.random.fork_rng(devices=[] if dev.type=='cpu' else [dev.index]):
            torch.random.default_generator.manual_seed(int(seed))
            if dev.type=='cuda':
                with torch.cuda.device(dev):torch.cuda.manual_seed(int(seed))
            with torch.device('cpu'):actor=Actor(obs_dim=config.obs_dim);critic=Critic(dim=config.obs_dim,amp=False,empirical_norm=config.critic_empirical_norm)
            actor=actor.to(device=dev,dtype=torch.float32);critic=critic.to(device=dev,dtype=torch.float32)
    finally:torch.set_default_dtype(previous_dtype)
    return dict(actor=actor,critic=critic,NN_constructors=2,device=str(dev),dtype='float32',config=config)

class CanonicalMemoryTask:
    """F64 executed-control/history arithmetic used by the matched phone run."""
    def __init__(self,task):self.task=task
    def __getattr__(self,name):return getattr(self.task,name)
    def advance_memory(self,m,a,c,u):
        result=self.task.advance_memory(m,a,c,u)
        for key in ['a1','a2','c1','last_u']:result[key]=result[key].to(dtype=u.dtype,device=u.device)
        return result

def imagined_actor_objective(*,actor,critic,world_model,task,geometry,codec,norms,physical,initial_state58,memory,horizon,gamma=.99,wm_step):
    """Batched H-step BPTT objective, parameter freezing retains input AD.

    Future object reference is a command. Future simulator contacts/forces or
    replay targets are not arguments. Caller owns optimizer and backward.
    This function does not certify native gradients, actual contact or policy
    benefit. H1 u has no direct geometry path; H2 uses predicted-state geometry.
    Geometry contexts are local to this objective: each new predicted state
    gets a fresh context, reused for its reward and subsequent observation/WM.
    No real-observation substitution, graph detach or cross-update cache.
    """
    import torch
    if initial_state58.ndim!=2 or initial_state58.shape[1]!=58 or initial_state58.shape[0]<1:raise ValueError('Explicit B>=1 state[B,58] and batched memory')
    if not isinstance(horizon,int) or horizon<1:raise ValueError('Positive imagination horizon')
    if torch.any(memory['step']+horizon>100):raise ValueError('Finite reference horizon, no beyond-end physics rollout')
    if world_model.training:raise ValueError('Injected WM must be eval mode')
    saved=[(p,p.requires_grad) for module in [world_model,critic] for p in module.parameters()]
    for p,_ in saved:p.requires_grad_(False)
    x=initial_state58;m=memory;survival=torch.ones(x.shape[:-1],dtype=x.dtype,device=x.device);objective=torch.zeros_like(survival);evidence=[]
    try:
        context=geometry.bundle(x)
        for k in range(horizon):
            obs=task.observation(x,m,geometry,context=context);action=actor(obs);residual,u=task.control_target(action,m)
            prediction=wm_step(x,u,m['native_h_s'],m['n_substeps'],model=world_model,norms=norms,physical=physical,codec=codec,geometry=geometry,context=context);y=prediction['next_state58'];next_context=geometry.bundle(y);reward=task.rewards(y,m,action,context=next_context)
            if not reward['objective_known'].all() or not torch.isfinite(reward['total']).all():raise ValueError('Known finite task objective required, no zero substitute')
            stop=task.termination(y,m);objective=objective+(gamma**k)*survival*reward['total'];continuation=(1-stop['soft_loss_probability'])*(~stop['reference_timeout']).to(x.dtype)
            evidence.append(dict(target_frame=m['step']+1,reward=reward,prediction=prediction,normalized_action20=action,executed_u20=u,survival_before=survival))
            survival=survival*continuation;m=task.advance_memory(m,action,residual,u);x=y;context=next_context
        bootstrap=critic(task.observation(x,m,geometry,context=context)).to(x.dtype);objective=objective+(gamma**horizon)*survival*bootstrap
        return dict(actor_loss=-objective.mean(),return_per_start=objective,final_state58=x,final_memory=m,steps=evidence,
            call_counts=dict(Actor_FW=horizon,WM_FW=horizon,Critic_FW=1,native_steps=0,optimizer_updates=0,backward_calls=0),actual_contact_or_physical_gradient_qualified=False)
    finally:
        for p,flag in saved:p.requires_grad_(flag)

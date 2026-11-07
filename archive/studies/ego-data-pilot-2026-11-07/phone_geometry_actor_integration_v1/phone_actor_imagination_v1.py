"""Callable upstream Actor/Critic + new phone WM imagination bridge.

This file contains no optimizer, backward call, native rollout or training CLI.
The function is the differentiable objective connection, not a completed
SuperMimic trainer, a qualified dynamics gradient, or an Actor checkpoint.
"""
import importlib

def make_fresh_actor_critic(*,seed,device='cpu'):
    """Explicit runtime call only; metadata never invokes NN constructors."""
    import torch
    if str(device)!='cpu':raise ValueError('This source contract is CPU-only')
    upstream=importlib.import_module('dex_supermimic.rl.dex_supermimic.networks')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        actor=upstream.Actor(obs_dim=222).to(device)
        critic=upstream.Critic(dim=222,amp=False,empirical_norm=True).to(device)
    return dict(actor=actor,critic=critic,source='pinned Dex Actor/Critic classes; no old weights/norm',NN_constructors=2)

def imagined_actor_objective(*,actor,critic,world_model,task,geometry,codec,norms,physical,initial_state58,memory,
                            horizon,gamma=.99,wm_step):
    """Recurrent predicted-state objective for A1 or A2, with input gradients.

    Future object reference is a task command. Every WM forward uses only its
    current predicted state/action/native h/n and current geometry. Neither
    future simulator forces, replay future states, nor true next contacts are
    arguments. WM/critic parameter gradients are disabled and restored; their
    input gradients are retained. Caller owns later Actor optimizer/update.
    """
    import torch
    if not isinstance(horizon,int) or horizon<1:raise ValueError('Positive fixed imagination horizon')
    if torch.any(memory['step']+horizon>100):raise ValueError('Select a finite-reference start/horizon; no beyond-end rollout')
    if world_model.training:raise ValueError('Injected frozen WM must be eval mode')
    saved=[(p,p.requires_grad) for module in [world_model,critic] for p in module.parameters()]
    for p,_ in saved:p.requires_grad_(False)
    x=initial_state58;m=memory;survival=torch.ones(x.shape[:-1],dtype=x.dtype,device=x.device);objective=torch.zeros_like(survival)
    evidence=[]
    try:
        for k in range(horizon):
            obs=task.observation(x,m,geometry)
            action=actor(obs)
            residual,u=task.control_target(action,m)
            prediction=wm_step(x,u,m['native_h_s'],m['n_substeps'],model=world_model,norms=norms,physical=physical,codec=codec,geometry=geometry)
            y=prediction['next_state58']
            reward=task.rewards(y,m,action)
            if not reward['objective_known'].all() or not torch.isfinite(reward['total']).all():raise ValueError('Unknown task objective: no zero substitution')
            stop=task.termination(y,m)
            objective=objective+(gamma**k)*survival*reward['total']
            continuation=(1-stop['soft_loss_probability'])*(~stop['reference_timeout']).to(x.dtype)
            evidence.append(dict(target_frame=m['step']+1,reward=reward,prediction=prediction,normalized_action20=action,executed_u20=u,survival_before=survival))
            survival=survival*continuation
            m=task.advance_memory(m,action,residual,u);x=y
        bootstrap=critic(task.observation(x,m,geometry)).to(x.dtype)
        objective=objective+(gamma**horizon)*survival*bootstrap
        return dict(actor_loss=-objective.mean(),return_per_start=objective,final_state58=x,final_memory=m,steps=evidence,
            call_counts=dict(Actor_FW=horizon,WM_FW=horizon,Critic_FW=1,native_steps=0,optimizer_updates=0,backward_calls=0),
            actual_contact_or_physical_gradient_qualified=False,scope='Callable imagination connection only; no Actor training was performed by defining/calling forward objective')
    finally:
        for p,flag in saved:p.requires_grad_(flag)

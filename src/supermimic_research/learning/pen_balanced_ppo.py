"""Paired 128+128 PPO minibatches from two fresh rollouts of one behavior policy."""
import torch
from dex_supermimic.rl.dex_supermimic.ppo import gae_targets


def kl(old, current):
    return torch.distributions.kl_divergence(old,current).sum(-1)


def mixed_update(learners, clips, initial_obs, index, limit=None):
    l=learners[clips[0]]; actor,critic,opt=l.actor,l.critic,l.optimizer;c=l.cfg
    rows=[];watch=[]
    for key in clips:
        learner=learners[key]
        if learner.actor is not actor or learner.critic is not critic or learner.optimizer is not opt:raise ValueError('Need shared Actor/Critic/Adam')
        m=learner.memory
        with torch.no_grad():
            v=critic(learner.observations['critic']).flatten()
            adv,ret=gae_targets(m,v,gamma=c.ppo_gamma,lam=c.lam,normalize=c.ppo_normalize_advantage)
        rows.append((m['obs'].reshape(-1,161),m['critic_obs'].reshape(-1,188),m['actions'].reshape(-1,20),m['log_prob'].flatten(),adv,ret))
        # One fixed reset observation and 32 evenly spaced recorded states per task.
        watch.append(torch.cat([initial_obs[key][:1],rows[-1][0][::64]],dim=0))
    with torch.no_grad():
        old=[actor.distribution(x,validate_args=False) for x in watch]
    def measured():
        vals=[kl(d,actor.distribution(x,validate_args=False)) for d,x in zip(old,watch)]
        return torch.stack([a for v in vals for a in (v[0],v[1:].mean())])
    params=list(actor.parameters());scales=[];max_kl=0.
    gen=torch.Generator(device='cpu').manual_seed(7411+1009*index+1)
    for _ in range(c.ppo_epochs):
        lists=[torch.randperm(len(row[0]),generator=gen).to(l.device).split(128) for row in rows]
        if not all(len(ids)==16 and all(len(a)==128 for a in ids) for ids in lists):raise ValueError('Expected 2 x 2048 samples per epoch')
        for i0,i1 in zip(*lists):
            obs,co,actions,lp,adv,ret=[torch.cat((a[i0],b[i1])) for a,b in zip(*rows)]
            d=actor.distribution(obs);ratio=(d.log_prob(actions).sum(-1)-lp).exp()
            policy=-torch.minimum(ratio*adv,ratio.clamp(1-c.ppo_clip,1+c.ppo_clip)*adv).mean()
            loss=policy+c.ppo_value_coef*(critic(co).flatten()-ret).square().mean()-c.ppo_entropy*d.entropy().sum(-1).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params+list(critic.parameters()),c.ppo_max_grad_norm,error_if_nonfinite=True)
            previous=[p.detach().clone() for p in params] if limit is not None else []
            opt.step()
            with torch.no_grad():
                factor=1.
                if limit is not None and float(measured().max())>limit:
                    delta=[p-q for p,q in zip(params,previous)]
                    while factor>2**-16:
                        factor*=.5
                        for p,q,v in zip(params,previous,delta):p.copy_(q+factor*v)
                        if float(measured().max())<=limit:break
                    else:
                        factor=0.
                        for p,q in zip(params,previous):p.copy_(q)
                k=measured()
                if not torch.isfinite(k).all() or (limit is not None and k.max()>limit+1e-5):raise ValueError('Invalid monitored policy change')
                max_kl=max(max_kl,float(k.max()));scales.append(factor)
    # Both collectors have consumed one on-policy rollout. The shared optimizer performed 64 joint calls.
    for learner in learners.values():
        learner.iteration+=1;learner.optimizer_steps+=len(scales)//2;learner.normalization_batch=None
    return dict(actual_adam=len(scales),actor_fraction_mean=sum(scales)/len(scales),
                actor_zero_steps=scales.count(0.),scaled_steps=sum(v<1 for v in scales),peak_kl=max_kl,
                end_kl_reset001_rows001_reset003_rows003=k.cpu().tolist())

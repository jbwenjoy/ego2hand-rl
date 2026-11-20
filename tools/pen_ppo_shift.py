"""End-of-update discrepancy on exactly the states used to sample the rollout."""
import torch


@torch.no_grad()
def policy_shift(old, new, memory, clip):
    obs=memory['obs'].flatten(0,1)
    actions=memory['actions'].flatten(0,1)
    phase=memory['phase_used'].flatten()
    sums={k:[0,0.,0.] for k in ('all','late','divergence')}
    for start in range(0,len(obs),2048):
        sl=slice(start,start+2048)
        a=old.distribution(obs[sl],validate_args=False)
        b=new.distribution(obs[sl],validate_args=False)
        kl=torch.distributions.kl_divergence(a,b).double().sum(-1)
        ratio=(b.log_prob(actions[sl])-a.log_prob(actions[sl])).sum(-1).exp()
        if not torch.isfinite(kl).all() or not torch.isfinite(ratio).all():
            raise ValueError('Nonfinite policy diagnostic')
        for key,mask in [('all',phase[sl]>=1),('late',phase[sl]>=185),('divergence',(phase[sl]>=206)&(phase[sl]<=225))]:
            n,s,c=sums[key]
            sums[key]=[n+int(mask.sum()),s+float(kl[mask].sum()),c+int(((ratio[mask]-1).abs()>clip).sum())]
    return {k:dict(n=n,kl_old_new=s/n if n else None,clip_fraction=c/n if n else None) for k,(n,s,c) in sums.items()}

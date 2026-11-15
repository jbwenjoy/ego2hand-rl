"""Shared scripted continuation probe, never a learned controller or trainer."""
import torch
from supermimic_research.mechanics.portable_contact_records import contacts


def cpu(value):return value.detach().cpu().numpy().copy()


def advance(adapter,k,*,partial_reset_at_boundary):
    if k==8 and partial_reset_at_boundary:
        adapter.reset(torch.arange(0,adapter.num_envs,2,device=adapter.device))
    w=torch.arange(adapter.num_envs,device=adapter.device)[:,None];j=torch.arange(20,device=adapter.device)[None,:]
    sample=.05*torch.sin(.2*k+.13*w+.03*j)
    if k==2:sample[0,0]=1.5
    ob,reward,term,trunc,info=adapter.step(sample)
    result=dict(k=k,policy=cpu(ob['policy']),critic=cpu(ob['critic']),reward=cpu(reward),
        sample=cpu(sample),applied=cpu(info['applied_raw']),EMA=cpu(adapter.action._processed_actions),
        target=cpu(adapter.action._targets),phase=cpu(adapter.motion.time_steps),terminated=cpu(term),truncated=cpu(trunc))
    ids=(term|trunc).nonzero().flatten()
    if len(ids):adapter.reset(ids)
    return result


def substep_record(adapter):
    sim=adapter.env.sim;d=sim.wp_data
    return dict(qpos=d.qpos.numpy().copy(),qvel=d.qvel.numpy().copy(),ctrl=d.ctrl.numpy().copy(),
        warmstart=d.qacc_warmstart.numpy().copy(),time=d.time.numpy().copy(),
        contact=contacts(sim.wp_model,d,sim.mj_model,adapter.features.body))

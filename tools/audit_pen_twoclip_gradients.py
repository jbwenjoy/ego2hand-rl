"""CPU pre-update policy and value gradient audit on two parent-policy rollouts."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dex_supermimic.rl.dex_supermimic.ppo import gae_targets, ppo_update
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.learning.native_pen_ppo import model_pair, validate_modules
from supermimic_research.strict_json import write_json


CLIPS = ("clip_001", "clip_003")
B = 8


def flattened(tensors):
    return torch.cat([t.detach().reshape(-1) for t in tensors]).double()


def comparison(left, right):
    dot = float(left @ right)
    n1 = float(torch.linalg.vector_norm(left))
    n2 = float(torch.linalg.vector_norm(right))
    cos = dot / (n1 * n2) if n1 and n2 else None
    return dict(norm_001=n1, norm_003=n2, dot=dot, cosine=cos,
                angle_deg=None if cos is None else float(np.degrees(np.arccos(np.clip(cos,-1,1)))))


def make_rows(packet, actor, critic, cfg):
    m = packet["memory"]
    if tuple(m["reward"].shape) != (64, 32):
        raise ValueError("Expected one 64-control, 32-world rollout")
    with torch.no_grad():
        obs = m["obs"].reshape(-1, 161)
        privileged = m["critic_obs"].reshape(-1, 188)
        actions = m["actions"].reshape(-1, 20)
        lp = m["log_prob"].flatten()
        logprob_error = (actor.distribution(obs).log_prob(actions).sum(-1) - lp).abs()
        value_error = (critic(privileged).flatten() - m["value"].flatten()).abs()
        final_value = critic(packet["bootstrap_obs"]).flatten()
        advantages, returns = gae_targets(m, final_value, gamma=cfg.ppo_gamma, lam=cfg.lam,
                                          normalize=cfg.ppo_normalize_advantage)
        if logprob_error.max() > .001 or value_error.max() > .001:
            raise ValueError(f"Parent-policy reproduction failed: lp {logprob_error.max()}, V {value_error.max()}")
        diagnostics = dict(transitions=len(obs), max_abs_logprob_error=float(logprob_error.max()),
            max_abs_value_error=float(value_error.max()), phase_min=int(m["phase_used"].min()),
            phase_max=int(m["phase_used"].max()), terminated=int(m["terminated"].sum()),
            truncated=int(m["truncated"].sum()), return_mean=float(returns.mean()),
            return_std=float(returns.std()), advantage_mean=float(advantages.mean()),
            advantage_std=float(advantages.std()))
    return (obs, privileged, actions, lp, advantages, returns), diagnostics


def gradient(actor, critic, rows, indices, cfg):
    obs, privileged, actions, old_lp, adv, returns = [v[indices] for v in rows]
    dist = actor.distribution(obs)
    ratio = (dist.log_prob(actions).sum(-1) - old_lp).exp()
    surrogate = -torch.minimum(ratio*adv, ratio.clamp(1-cfg.ppo_clip,1+cfg.ppo_clip)*adv).mean()
    entropy = dist.entropy().sum(-1).mean()
    policy_loss = surrogate - cfg.ppo_entropy*entropy
    value_loss = cfg.ppo_value_coef*(critic(privileged).flatten()-returns).square().mean()
    actor_g = flattened(torch.autograd.grad(policy_loss, tuple(actor.parameters())))
    critic_g = flattened(torch.autograd.grad(value_loss, tuple(critic.parameters())))
    total = float((actor_g.square().sum()+critic_g.square().sum()).sqrt())
    scale = min(1., float(cfg.ppo_max_grad_norm)/(total+1e-6))
    actor_norm, critic_norm = float(actor_g.norm()), float(critic_g.norm())
    return actor_g,critic_g,dict(n=len(indices),surrogate=float(surrogate.detach()),
            entropy=float(entropy.detach()),weighted_value_loss=float(value_loss.detach()),
            clip_fraction=float(((ratio-1).abs()>cfg.ppo_clip).float().mean()),
            global_preclip_norm=total, actor_norm=actor_norm, weighted_critic_norm=critic_norm,
            joint_clip_multiplier=scale, actor_only_clip_multiplier=min(1.,float(cfg.ppo_max_grad_norm)/(actor_norm+1e-6)),
            critic_sq_norm_fraction=(critic_norm/total)**2 if total else None)


def validate_against_ppo(actor, critic, optimizer, packet, cfg, ag, cg, clip):
    # Verify using the production loss on one full shuffled batch, including its actual joint clipping.
    # Replacing the final optimizer call captures the gradients without changing parameters.
    c = copy.copy(cfg)
    c.ppo_minibatches = c.ppo_epochs = 1
    norms = {}
    def capture():
        for label,module,expected in (("actor",actor,ag),("critic",critic,cg)):
            got = flattened(tuple(p.grad for p in module.parameters()))
            reference = expected*clip
            err=float((got-reference).norm()/reference.norm().clamp_min(1e-12))
            norms[label+"_relative_L2"]=err
            if err>2e-4:
                raise ValueError(f"Full PPO gradient parity failed: {label} relative error {err}")
    with torch.random.fork_rng(devices=[]),patch.object(optimizer,"step",capture):
        torch.manual_seed(7411)
        ppo_update(actor,critic,optimizer,packet["memory"],packet["bootstrap_obs"],c)
    optimizer.zero_grad(set_to_none=True)
    return norms


def matched_blocks(actor,critic,rows,cfg):
    blocks = []
    grid = torch.arange(64*32).reshape(64,32)
    for worlds in grid.split(32//B,dim=1):
        ag,cg,stats = gradient(actor,critic,rows,worlds.flatten(),cfg)
        blocks.append((ag,cg,stats))
    return blocks


def sampling_check(left,right):
    # The normalized advantages are held at their actual batch values in every subsample.
    # Reweight mean block gradients algebraically via their Gram matrices for fast paired resampling.
    g1, g2 = torch.stack(left), torch.stack(right)
    A,Bm,C = [m.numpy() for m in (g1@g1.T,g2@g2.T,g1@g2.T)]
    def cos(w):
        dot = np.einsum('ni,ij,nj->n',w,C,w)
        n1 = np.einsum('ni,ij,nj->n',w,A,w)
        n2 = np.einsum('ni,ij,nj->n',w,Bm,w)
        return dot / np.sqrt(n1*n2)
    rng=np.random.default_rng(7411)
    weights=rng.multinomial(B,[1/B]*B,size=2000)/B
    boot=cos(weights)
    return dict(equal_block_mean_cos=float(cos(np.ones((1,B))/B)[0]),
                single_block_cos=cos(np.eye(B)).tolist(),
                leave_one_block_out_cos=cos((np.ones((B,B))-np.eye(B))/(B-1)).tolist(),
                paired_bootstrap_cos_q025_q50_q975=np.quantile(boot,[.025,.5,.975]).tolist(),
                paired_bootstrap_cos_below_zero=float(np.mean(boot<0)))


def main(source, out):
    source=source.resolve()
    parent_pin=json.loads((ROOT/'runs/pen_batch_updates_v1/parent.json').read_bytes())
    collection_pin=file_pin(source/'report.json')
    collection=json.loads(verify_file_pin(collection_pin).read_bytes())
    intent=json.loads(verify_file_pin(collection['intent']).read_bytes())
    if intent['parent'] != parent_pin:
        raise ValueError('Collector used different PPO parent')
    state=torch.load(verify_file_pin(parent_pin),map_location='cpu',weights_only=False)
    if state['iteration']!=2781 or state['optimizer_steps']!=88992 or state['metadata']['arm']!='RESIDUAL':
        raise ValueError("Unexpected residual parent")
    torch.set_num_threads(2)
    cfg=SimpleNamespace(**state['config'])
    actor,critic,opt=model_pair('cpu',cfg)
    validate_modules(actor,critic,opt,state)
    actor.load_state_dict(state['modules']['actor']);critic.load_state_dict(state['modules']['critic'])
    actor.eval();critic.eval()
    del state
    pins,per_clip,grads,blocks={},{},{},{}
    for k in CLIPS:
        pin=collection['clips'][k]['batch']
        if not all(collection['clips'][k]['health']['checks'].values()):
            raise ValueError('Source rollout failed collection consistency')
        packet=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)
        if not torch.equal(packet['start_obs'],packet['memory']['obs'][0]):
            raise ValueError('Saved start observation differs')
        rows,checks=make_rows(packet,actor,critic,cfg)
        with torch.no_grad():
            noise=torch.randn(64,32,20,generator=torch.Generator().manual_seed(7411+1009))
            dist=actor.distribution(rows[0])
            predicted=dist.loc+dist.scale*noise.reshape(-1,20)
            max_sampling_error=float((predicted-rows[2]).abs().max())
            if max_sampling_error>1e-4:
                raise ValueError(f'Parent actions differ from paired reparameterized draws: {max_sampling_error}')
            checks['paired_action_max_abs_error']=max_sampling_error
        ag,cg,stats=gradient(actor,critic,rows,torch.arange(2048),cfg)
        checks['real_PPO_gradient_parity']=validate_against_ppo(actor,critic,opt,packet,cfg,ag,cg,stats['joint_clip_multiplier'])
        segments=matched_blocks(actor,critic,rows,cfg)
        for ix,full in ((0,ag),(1,cg)):
            err=float((torch.stack([b[ix] for b in segments]).mean(0)-full).norm()/full.norm().clamp_min(1e-12))
            if err>2e-4:raise ValueError(f"Block mean differs ({err})")
        pins[k]=pin;grads[k]=(ag,cg);blocks[k]=segments
        per_clip[k]=dict(diagnostics=checks,full=stats,
                         equal_world_blocks=[dict(worlds=list(range(i*4,i*4+4)),**s[2]) for i,s in enumerate(segments)])
    result={}
    for ix,key in enumerate(('actor','weighted_critic')):
        result[key]=dict(full=comparison(grads[CLIPS[0]][ix],grads[CLIPS[1]][ix]),
             block_sensitivity=sampling_check([s[ix] for s in blocks[CLIPS[0]]],[s[ix] for s in blocks[CLIPS[1]]]))
    out.mkdir(parents=False,exist_ok=False)
    write_json(out/'report.json',dict(parent=parent_pin,rollouts=pins,
        collection_report=collection_pin,
        script=file_pin(__file__),ppo_source=file_pin(ROOT/'src/dex_supermimic/rl/dex_supermimic/ppo.py'),
        gradient_sign='Loss gradient before the Adam update; negative cross-task dot means stepping against either gradient increases the other objective loss to first order for vanilla gradient descent.',
        settings=vars(cfg),per_clip=per_clip,comparisons=result,updates=0,
        scope='Common parent, two fresh 64x32 batches. Full-batch pre-update gradients; actual PPO uses shuffled minibatches and Adam moments. Subsampling uses whole worlds in eight equal 4-world blocks, both tasks reweighted with the same weights, full-batch GAE normalization unchanged. Bootstrap quantiles describe conditional block sensitivity, not independent policy-seed uncertainty.'))
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args()
    main(args.source,args.out)

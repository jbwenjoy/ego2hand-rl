"""Simulator-independent PPO update shared by the cube and native pen adapters.

The collector supplies *unsquashed sampled* actions/log probabilities. Timeout
reward corrections belong to that collector; do not apply them twice here.
Normalization stays frozen during an update. Legacy cube minibatch truncation
and its historical ``kl`` diagnostic are retained and explicitly counted.
"""
import torch
from torch import nn


@torch.no_grad()
def gae_targets(memory, final_value, *, gamma, lam, normalize):
    """Detached fixed PPO targets, including across final bootstrap boundaries."""
    rewards = memory['reward']
    values = memory['value']
    dones = (memory['terminated'].bool() | memory['truncated'].bool()).float()
    batch, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = rewards.new_zeros(num_envs)
    for t in reversed(range(batch)):
        next_non_terminal = 1.0 - dones[t]
        next_value = final_value if t == batch - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_non_terminal * next_value - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
    flat_advantages = advantages.reshape(-1)
    returns = flat_advantages + values.reshape(-1)
    if normalize:
        if flat_advantages.numel() < 2:
            raise ValueError('Sample-standard-deviation advantage normalization needs at least two transitions')
        flat_advantages = (flat_advantages - flat_advantages.mean()) / flat_advantages.std().clamp_min(1e-8)
    return flat_advantages, returns


def ppo_update(actor, critic, optimizer, memory, final_critic_observation, cfg):
    """Optimize one rollout without reading or advancing a simulator.

    Targets are constants. The previous runner accidentally differentiated its
    final critic bootstrap, reusing that graph across minibatches. Only this
    gradient bug is repaired; objective/order/optimizer calls remain unchanged.
    ``approx_kl`` uses stored rollout log probabilities; historical ``kl`` is
    retained for log compatibility and is not a rollout-policy KL measurement.
    """
    c = cfg
    batch, num_envs = memory['reward'].shape
    count = batch * num_envs
    if c.ppo_epochs < 1 or not 1 <= c.ppo_minibatches <= count:
        raise ValueError('Positive epochs and at most one minibatch per transition required')
    flat_actor_obs = memory['obs'].reshape(-1, memory['obs'].shape[-1])
    flat_obs = memory['critic_obs'].reshape(-1, memory['critic_obs'].shape[-1])
    flat_actions = memory['actions'].reshape(-1, memory['actions'].shape[-1])
    flat_old_log_prob = memory['log_prob'].reshape(-1)
    with torch.no_grad():
        final_value = critic(final_critic_observation).flatten()
    flat_advantages, returns = gae_targets(memory, final_value, gamma=c.ppo_gamma,
        lam=c.lam, normalize=c.ppo_normalize_advantage)
    usable_count = (count // c.ppo_minibatches) * c.ppo_minibatches
    minibatch_size = max(1, count // c.ppo_minibatches)
    actor_losses, critic_losses, entropies, clip_fractions, kls, approx_kls = [], [], [], [], [], []
    for _ in range(c.ppo_epochs):
        permutation = torch.randperm(usable_count, device=flat_actor_obs.device)
        for start in range(0, usable_count, minibatch_size):
            ids = permutation[start:start + minibatch_size]
            distribution = actor.distribution(flat_actor_obs[ids])
            log_prob = distribution.log_prob(flat_actions[ids]).sum(-1)
            with torch.no_grad():
                old_distribution = actor.distribution(flat_actor_obs[ids])
            log_ratio = log_prob - flat_old_log_prob[ids]
            ratio = log_ratio.exp()
            policy_loss = -torch.min(ratio * flat_advantages[ids],
                ratio.clamp(1.0 - c.ppo_clip, 1.0 + c.ppo_clip) * flat_advantages[ids]).mean()
            entropy = distribution.entropy().sum(-1).mean()
            values = critic(flat_obs[ids])
            value_loss = (values - returns[ids]).square().mean()
            loss = policy_loss + c.ppo_value_coef * value_loss - c.ppo_entropy * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(tuple(actor.parameters()) + tuple(critic.parameters()), c.ppo_max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                clip_fractions.append(((ratio - 1.0).abs() > c.ppo_clip).float().mean())
                kls.append(torch.distributions.kl_divergence(old_distribution, distribution).sum(-1).mean())
                approx_kls.append((ratio - 1.0 - log_ratio).mean())
            actor_losses.append(policy_loss.detach())
            critic_losses.append(value_loss.detach())
            entropies.append(entropy.detach())
    target_variance = returns.var()
    with torch.no_grad():
        explained = 1.0 - (returns - critic(flat_obs)).square().mean() / target_variance.clamp_min(1e-8)
    return dict(actor_loss=torch.stack(actor_losses).mean(), critic_loss=torch.stack(critic_losses).mean(),
        entropy=torch.stack(entropies).mean(), clip_fraction=torch.stack(clip_fractions).mean(),
        kl=torch.stack(kls).mean(), approx_kl=torch.stack(approx_kls).mean(), explained_variance=explained,
        horizon=batch, optimizer_steps=c.ppo_epochs * c.ppo_minibatches,
        used_transitions_per_epoch=usable_count, omitted_transitions_per_epoch=count - usable_count)

def policy_update(self):
    c = self.cfg
    w = self.sample_starts()
    x = w["x"]
    m = self.unflatten_memory(w["m"])
    history = self.history_from_obs(w["obs"])
    auxiliary = w["aux"]
    simulator = w["sim_context"]
    bias = self.reader.robot.data.encoder_bias[w["env"]]
    horizon = c.horizon
    frozen = (self.wm, self.fail_head, self.critic)
    for mod in frozen:
        mod.requires_grad_(False)
    full151_active_rows = []
    full151_critic_denominators = []
    full151_start_phases = m["time"].detach().clone()
    xs = []
    rewards = []
    continues = []
    survival = x.new_ones(x.shape[0])
    objective = x.new_zeros(x.shape[0])
    # These quantities are constant for this update; keep them out of the
    # recurrent loop to avoid repeating host work and Python-side control flow.
    gamma_powers = [c.gamma**t for t in range(horizon + 1)]
    scale = min(
        1,
        max(
            0,
            (self.env.common_step_counter / self.env.max_common_steps - 0.05) / 0.75,
        ),
    )
    adaptive = float(getattr(self.env, "_adaptive_episode_curriculum", 0.05))
    try:
        for t in range(horizon):
            full151_active = _full151_phase_active(m)
            full151_active_rows.append(full151_active)
            _full151_finite(x, "imagined_input")
            _full151_finite(history, "actor_history")
            context = self.task_features(x, m, auxiliary, simulator)
            xs.append(self.critic_features(context))
            a = self.step_actor(self.mdp.flatten_history(history))
            target, u = self.mdp.control_target(a, m, bias) if self.is_pen else ema_target(
                a,
                m["c1"],
                self.mdp.default,
                self.mdp.lower,
                self.mdp.upper,
                bias,
                self.mdp.alpha,
                m["age"].squeeze(-1),
                self.mdp.dt,
            )
            x1, contact = self.step_wm(
                x, u, self.wm_features(x, simulator, self.observations.disturbance(auxiliary))
            )
            _full151_finite(x1, "WM_output_including_inactive")
            _full151_finite(contact, "contact_output_including_inactive")
            x1 = torch.where(full151_active[:, None], x1, x.detach())
            contact = torch.where(full151_active[:, None], contact, torch.zeros_like(contact))
            torque_kwargs = {"limits": m["limits"]} if self.is_pen else {}
            tau = self.mdp.pd_torque(x, u, simulator, **torque_kwargs)
            r, cage, _ = self.reward_step(x1, m, a, target, contact.sigmoid(), tau, soft=True)
            # Original termination reads the PRE-reward counter.
            if self.is_pen:
                d = self.mdp.termination(x1, m)
            else:
                d = torch.sigmoid((m["cage"].squeeze(-1) - self.mdp.max_outside + 0.5) / 0.25)
            if self.is_pen:
                # A finite demonstration ends the imagined return; it is
                # excluded from the physical-failure head's labels.
                reference_end = (m["time"] + 2 >= m["end"] - m["start"]).squeeze(-1)
                d = torch.where(reference_end, 1.0, d)
            next_goal = quat_to_mat(F.normalize(torch.randn(x.shape[0], 4, device=x.device), dim=-1))
            next_memory = self.advance_memory(
                x1,
                m,
                a,
                target,
                cage,
                next_goal,
            )
            if c.failure_head and self.iteration >= c.wm_only_iters + c.failure_warmup_iters:
                next_context = self.task_features(x1, next_memory, auxiliary, simulator)
                p_fail = self.fail_head(
                    self.fail_head.features(x1, next_context)
                ).sigmoid().clamp(max=c.failure_clip)
                d = 1 - (1 - d) * (1 - p_fail)
            # A timeout is not a physical termination; the critic bootstraps it.
            _full151_finite(r, "reward_including_inactive")
            r = torch.where(full151_active, r, torch.zeros_like(r))
            d = torch.where(full151_active, d, torch.ones_like(d))
            objective = objective + gamma_powers[t] * survival * r
            survival = survival * (1 - d)
            rewards.append(r.detach())
            continues.append((1 - d).detach())
            updated = next_memory
            # Match interval velocity disturbances independently of the physical model.
            push = torch.zeros_like(x1)
            speed = 0.05 + torch.rand(x.shape[0], 1, device=x.device) * 0.1 * scale * adaptive
            dv = F.normalize(torch.randn(x.shape[0], 3, device=x.device), dim=-1) * speed
            event = torch.rand(x.shape[0], 1, device=x.device) < self.mdp.dt / 1.2
            active = event & (
                updated["age"] * self.mdp.dt >= 3
            )
            push[..., -6:-3] = 0 if self.is_pen else dv * active
            # The task caches the most recent sampled world-frame disturbance.
            # A sampled event before warmup clears the cache as in the simulator.
            palm_rotation = auxiliary[:, 3:12].reshape(-1, 3, 3)
            world_dv = mv(palm_rotation, dv * active)
            world_dir = mv(palm_rotation, F.normalize(dv, dim=-1) * active)
            if not self.is_pen:
                auxiliary = self.advance_disturbance(auxiliary, event, world_dir, world_dv)
            x = x1 + push
            m = updated
            observation = (
                self.mdp.policy_observation(x, m, auxiliary) if self.is_pen
                else self.mdp.noisy_observation(self.mdp.observation(x, m))
            )
            history = torch.cat((history[:, 1:], observation[:, None]), 1)
        context = self.task_features(x, m, auxiliary, simulator)
        final = self.critic_features(context)
        xs.append(final)
        full151_final_active = _full151_phase_active(m)
        full151_final_value = _full151_finite(self.critic(final), "bootstrap_including_inactive")
        full151_final_value = torch.where(full151_final_active, full151_final_value, torch.zeros_like(full151_final_value))
        objective = objective + (c.gamma**horizon) * survival * full151_final_value
        actor_loss = -objective.mean()
        self.optimize("actor", actor_loss)
    finally:
        for mod in frozen:
            mod.requires_grad_(True)
    inputs = torch.stack(xs).detach()
    rs = torch.stack(rewards)
    cs = torch.stack(continues)
    full151_valid = torch.stack(full151_active_rows)
    full151_norm_valid = torch.cat((full151_valid, full151_final_active[None]), 0)
    with torch.no_grad():
        self.critic.norm.update(inputs[full151_norm_valid])
        values = _full151_finite(self.critic(inputs), "criticvalues_including_inactive")
        ret = values[-1]
        targets = []
        for t in reversed(range(horizon)):
            ret = rs[t] + c.gamma * cs[t] * ((1 - c.lam) * values[t + 1] + c.lam * ret)
            targets.append(ret)
        targets = torch.stack(targets[::-1])
    features = inputs[:-1].flatten(0, 1)
    targets = targets.flatten()
    full151_flat_valid = full151_valid.flatten()
    count = features.shape[0]
    minibatch = count // c.critic_minibatches
    for _ in range(c.critic_epochs):
        permutation = torch.randperm(count, device=self.device)
        for i in range(c.critic_minibatches):
            ids = permutation[i * minibatch : (i + 1) * minibatch]
            full151_critic_denominators.append(full151_flat_valid[ids].sum().detach())
            critic_loss = _full151_masked_critic_mse(self.critic(features[ids]), targets[ids], full151_flat_valid[ids])
            self.optimize("critic", critic_loss)
    self._full151_runtime_policy_evidence = {"start_phases": full151_start_phases, "active_mask_H_B": full151_valid.detach(), "critic_norm_eligible": full151_norm_valid.detach().sum(), "critic_loss_minibatch_active_counts": full151_critic_denominators, "critic_total_candidate_rows": count, "optimizer_actor_calls": 1, "optimizer_critic_calls": len(full151_critic_denominators), "hard_mask_rule": "phase<150 only; physical soft termination retains native survival semantics"}
    return dict(
        actor_loss=actor_loss.detach(),
        critic_loss=critic_loss.detach(),
        survival=survival.detach().mean(),
        horizon=horizon,
    )

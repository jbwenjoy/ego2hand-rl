# Absolute pen target limit patch — CPU verification

The actual `SingleHandResidualEMAAction` and imagined `PenMDP.control_target`
now limit the absolute `reference + EMA residual - encoder bias` target to the
intersection of joint hard limits and enabled position-actuator control ranges.
Actuators are matched by their joint transmissions rather than array order.
The EMA residual keeps its original value and meaning. Mask selection occurs
before the limit, and the cached/frozen target stores the limited value.
The action's apply step also covers initialization or direct target replacement.

CPU checks:9/9 passed using the actual ActionTerm and PenMDP implementation with
the frozen physical MJB's ranges/transmissions. Cases include the reported pinky
command2.00234rad versus1.57rad limit, upper/lower extremes and encoder bias,
stricter/disabled actuator limits, reversed actuator order, legacy masked frozen
targets, direct initialization, saturated/interior gradients, and empty limit
intersections. This is not a live Warp environment test.

With the original MJB, one native MuJoCo physics step produced bitwise-identical
qpos, qvel and actuator force for the raw over-limit command and its explicitly
limited counterpart. This supports the internal-clipping explanation for that
fixture only. It does not establish that the complete3-second Warp rollout is
unchanged; that comparison remains an independent bounded evaluation.

The original A1 run and its unbounded command evaluation are retained. No
physical parameter, task threshold or source trajectory was altered. Subsequent
A1/A2 comparisons must both use the same bounded contract and freeze their own
source snapshot. Genuine Warp target-parity checks are available with:

```powershell
D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe scripts/check_pen_target_limits.py --output outputs/control_target_limits_warp_v1 --warp
```

This GPU command was prepared but not run by this agent. CPU evidence is in
`checks.json`. The implementation test script is
`scripts/check_pen_target_limits.py`.

# Phone COM/world state and 20 ms time contract v1

The new pure NumPy interface is `state_contract.py`. It does not change the old pen/WM code. Its source was frozen before the four analytic fixtures; all four passed on their first execution. No model, FK, Env, native forward, integration, NN or GPU was called.

`to_com_world_state(q20, qd20, p_root_world, R_body_world, v_root_world, omega_body, body_ipos_body)` returns a float64 state with shape `(...,58)`. Unbatched input is also supported. Every field must share its leading batch shape; only a fixed `(3,)` COM offset may broadcast. `R_body_world` may be row-major `(...,9)` or `(...,3,3)`. The function validates a proper rotation to an explicit 1e-5 arithmetic tolerance and never repairs or projects it.

| State slice | Quantity | Frame / unit |
|---|---|---|
| 0:20 | robot q20 | caller's fixed joint order, rad |
| 20:40 | robot qd20 | same joint order, rad/s |
| 40:43 | phone COM position | world, m |
| 43:52 | row-major body rotation | BODY to world, dimensionless; not inertial `ximat` |
| 52:55 | phone COM velocity | world, m/s |
| 55:58 | phone angular velocity | world, rad/s |

For the supplied body-frame COM offset `ipos`, the formulas are

```
offset_world = R_body_world @ ipos
p_COM = p_root + offset_world
omega_world = R_body_world @ omega_body
v_COM = v_root_world + cross(omega_world, offset_world)
```

`from_com_world_state(state58, body_ipos_body)` returns `q20`, `qd20`, `p_root_world`, `R_body_world`, `v_root_world`, and `omega_body`. Position and linear velocity subtract the same offset and cross term. Angular velocity solves against the supplied rotation, so slightly rounded FP32 rotations still have a numerical inverse without orthogonalization. Inputs are cast to float64 for arithmetic; q/qd are copied numerically without wrapping or normalization. This is not a promise of bit-exact coordinate roundtrip.

The analytic fixtures cover a rotated nonzero COM lever and cross-product sign; three rotated/batched forward/inverse pairs including rounded FP32 R and a rejected reflection; an independent closed-form derivative of a rotating offset on a translating root; and both integer timing grids including a deliberately drifting FP32 clock. The derivative test samples an analytic trajectory and does not integrate dynamics.

`TickPlan(h)` accepts only `H=.02` with the following requested integration intervals. Decimal arithmetic establishes the integer ratio. Tick masks require supplied integer indices; no accumulated floating clock is divided by h or rounded into a tick.

| h requested (s) | n | state ticks | nominal final solver PRE | terminal POST |
|---|---:|---|---:|---:|
| .0003125 | 64 | 0..64 | .0196875 s, tick63 | .0200000 s, tick64 |
| .00015625 | 128 | 0..128 | .01984375 s, tick127 | .0200000 s, tick128 |

Solver PRE tick i belongs to the interval ending at POST tick i+1. `masks` reports PRE ticks 0..n-1 and integrated POST ticks 1..n. `native_clock_descriptors(native_time_s, ticks)` preserves actual supplied native clocks and their difference from nominal tick times; it does not set a new drift threshold. The declaration separately records the FP32 representation of h. Per-control state/clock alignment, native effective timestep and any long-rollout clock qualification remain the caller's responsibility.

The optional saved check read four existing zero-velocity cold-probe NPZ files, checking their producer pins. The declared phone asset's body COM offset gives zero difference from every saved native COM position; new state COM velocity and angular velocity agree exactly with saved Jp/Jr times qvel. These probes have qvel=0: they cannot validate nonzero angular velocity or dynamic phase. The nonzero cross term is instead covered by the analytic fixtures. No new native body-ipos or Jacobian query was made.

The 58-state layout and possible 52-dimensional rotation-chart layout do not carry old pen normalizers, weights, LogR charts, force labels, cached self15 phase, gradient qualification or task reward into this phone interface. No 52-chart transform is implemented here. In particular, the old last-solver cache near 15 ms does not become a new 19.6875/19.84375 ms label by renaming it. This module implements a state/time contract only; the phone MDP, Actor and CHORD integration remain separate work.

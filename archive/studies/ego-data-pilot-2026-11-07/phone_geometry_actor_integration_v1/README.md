# Phone Actor / CHORD integration: callable source, CPUFalse

This bundle implements a concrete fixed-wrist task around HOT3D clip1856,
frames42–102. It does **not** train an Actor, execute physics, load WM weights,
or claim a complete SuperMimic trainer. `metadata_preflight_v1.py` is the only
prepared CLI: JSON/hash/AST checks, no Torch or NPZ value load. Runtime calls
below remain unexecuted until a separate frozen protocol.

## One named task and the real interface boundary

The start is the exact compressed geometry candidate and its saved GN
`probe_030` control, not `u=q`, a repaired old failure, or a successful holding
demonstration. Source qpos27, zero qvel26, ctrl20, native PRE COM/BODY orientation,
and raw byte receipts are bound. The static source has **no** complete300-state
integration receipt; a later fresh Reader reset must save its actual receipt.
Initial loading is evidence only. Its previous failed2s rollout stays failed.

The61 source timestamps and wrist-relative object intent are frozen by the
existing reference artifact; its101 control boundaries span0–2s without
stretching the original2.00000849s source. Reward for transition k→k+1 uses
POST state against object reference k+1 before memory advances. No robot joint
demonstration is invented. Human-left→robot-right functional borrowing,
unregistered source gravity, nominal mass/friction, missing real force truth
and privileged simulated-state observation remain explicit limitations.

`PhoneTask` supplies reference, observation222, bounded action/control, explicit
memory83, reward and state-only stop functions. Actor outputs normalized20,
mapped to a symmetric residual no larger than.05rad around the real initial
ctrl, with EMA.5. Per-axis scale is reduced to the actual joint-target / actuator
ctrl-range intersection. F64 base ctrl is preserved even for FP32 Actor output;
there is no silent clamp. All20 axes are represented. `c1` is residual memory,
`last_u` is true absolute-control memory; they are never interchangeable.

Observation contains state58, a1/a2/c1/last_u, three object goal poses at offsets
1/5/10, phase, current22 gap descriptors and known masks, and native h/n. The
future **task reference** is permitted; future force/contact labels are absent.
Unknown descriptors use an explicit mask; the WM's own current-feature contract
still rejects unavailable geometry. Future/current imagined geometry is
recomputed from the predicted state, never teacher-forced from replay labels.

## A1 tracking and A2 hypothetical material support

A1 uses COM and full-orientation chordal tracking, action-rate and joint-limit
terms. It does not copy the pen's axial score or absent joint reference.

A2 calls the existing object-frame `weighted_wrench_support` and CHORD
`support_reward`. It transports hand-side source `con.pos` anchors by current
compiled differentiable FK into phone BODY axes with a COM origin. Distance
to each frozen phone material sample supplies `exp(-(d/1mm)^2)` soft weight.
Normal direction is the **template BODY normal**, by the explicit hypothesis
of unchanged material contact and normal. Descriptor argmax planes are never
used as actual contact normals. The reference is a frozen engineered native
loaded-tripod capability template, not measured ego contact or force.

The unit-force support fingerprint uses all512 fixed unit6 directions,
lever/radius=.077m, and an8-edge design cone with nominal
mu_design=.7/sqrt(2). It is not the native condim3 four-edge pyramidal solver,
force equilibrium, actual Fn, torsional/patch capacity or a friction certificate.
`.7` is assumed, not measured. Soft proximity can hallucinate contacts and
penalize legitimate contact changes. Fixed template normals can be wrong.

The result therefore separates `proxy_known` / `objective_known` from
`actual_contact_known`, `actual_force_known`, and
`actual_contact_reward_known` (the latter stay false). Unknown real contacts
are not relabeled known-zero. The hypothetical A2 objective is computable
without making unknown actual normals a permanent learning gate. Native
evaluation must independently test its model mismatch and method effect.

## Actual Actor/WM/replay functions

`make_fresh_actor_critic(seed=...)` calls the pinned upstream Dex `Actor(222)`
and `Critic(222, amp=False, empirical_norm=True)` when explicitly invoked.
No old Actor/critic normalizers or weights are migrated.

`imagined_actor_objective(...)` actually connects the Actor, task action map,
new geometry WM forward, structured world-left decode, A1/A2 reward, task memory
recurrence and critic bootstrap. It returns a differentiable Actor loss; it
does **not** run backward or an optimizer. WM/critic weights are frozen while
their input gradients remain active. There is no future-force argument or old
pen52 integrator. Forward counters are returned. This is a callable policy-loss
bridge, not an executed policy update or a full online training job.

`PhoneReaderTaskBridge` accepts an existing frozen Reader. It adds no native
loop and does not construct a model. Reset/step later bind actual state/u and
300-state receipts. Original two-pad summaries remain intact; the new task
regroups raw contact rows explicitly into15/28/41 plus canonical22 slots.
It reports actual simultaneous three-pad coverage separately from marginals.

`build_replay_transition` supplies the upstream generic ReplayBuffer fields
and requires a raw persistence sink **before** insertion. Partial prefixes
retain their observed endpoint, but H targets and objective remain NaN with
valid=False; the upstream buffer's `nan_to_num` cannot create a valid target.
Done/terminated/timeout and episode ID are explicit. The old Runner is not
drop-in compatible: it assumes different memory, contact15, palm/right-chart
state, reward snapshots and environment configuration. A new phone update/
collection orchestration remains to be implemented, rather than faked here.

## What must happen before a policy claim

The current WM was trained on the old two-designated-pad local task. Its22
pair identities are reusable, but tripod-loaded states/reward are OOD. Current
CHECK endpoint errors or finite geometry AD do not qualify physical action
gradients; observed phantom responses remain a known risk. Next collect new
task-native replay, adapt/validate the WM, then freeze matched A1/A2 Actor
initialization, normalizer exposure, updates, replay and native sample budgets.
Manual2s success is not required before learning.

Evaluation remains the original no-aid two-second/double-dt physical test plus
new declared object-reference errors. The Reader's acquisition safety and this
scalar objective are not that evaluator. Initial loading, geometry, proxy score,
predicted forces, or a low Actor loss never constitute holding success. No new
Actor learning, physical rollout or NN call was executed in this source delivery.

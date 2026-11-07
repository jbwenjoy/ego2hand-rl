# Complete integration-state restoration

Source-only review, no native/model/FK/step/solver call. Scope: MuJoCo 3.11.0,
official commit b85fdca54f0e0038b804af146a0b4e94199e00d0, and the original rigid
phone model without sleep, plugins or opaque callbacks.

`mj_setState` copies the selected state components and computes no dynamics or
derived fields. The recorded **signature is 16383; vector length is 300**.
INTEGRATION includes time, qpos/qvel, act/history, warmstart, ctrl, applied forces,
and equality/mocap/user/plugin values where present. It does not serialize every
MjData cache, contact, constraint matrix, warning or sleeping field. See the
[matching state implementation](https://github.com/google-deepmind/mujoco/blob/b85fdca54f0e0038b804af146a0b4e94199e00d0/src/engine/engine_support.c#L282).

`mj_copyData` copies a broader live data structure, model-sized arrays and arena,
rebinds destination-owned pointers, and handles plugin copies separately. Source
and destination sizes must agree and the source stack must not be in use. The
old MPC fixture proved live-data cloning, not saved-vector restoration. See
[matching copy implementation](https://github.com/google-deepmind/mujoco/blob/b85fdca54f0e0038b804af146a0b4e94199e00d0/src/engine/engine_io.c#L1145).

In the pinned forward source, `mj_step` invokes the full forward pipeline before
the selected integrator. This supports testing freshData/setState then step
without an external forward. It is not yet an actual replay result or a general
claim that INTEGRATION equals all MjData. Sleep, plugins, callbacks or split/skip
stepping require separate consideration.

## Minimal production path and fixture

1. Bind original model, exact dt, acquisition and source pins. Assert native3.11,
   original options/arrays, no aid, opaque callbacks, plugins/flex/equality/mocap,
   gravity compensation or sleep. Choose complete, finite audited VAL rows by a
   fixed metadata rule before looking at response error.
2. Create fresh MjData; restore the saved source vector with signature16383. Read
   it back with mj_getState and require all300 values' raw bytes to match. Do not
   call Reader.reset, zero velocities/warmstart, or add a forward after restore.
   A pure qpos/qvel state58 observer is valid; pre-step contacts/xipos/sensors are
   not restored observations and must not become invented reset-load labels.
3. Set actual held u20; its integration receipt must match the recorded
   source-after-action vector for baseline replay. Alternatively restore that
   after-action vector directly and verify ctrl. Explicitly state which route is
   used. Never conflate before-action and after-action receipts.
4. If production branches clone, mj_copyData the restored live base into each
   branch and verify integration bytes. Leave the source unstepped/unchanged.
   Execute only64/128 mj_step calls, capturing solver-PRE contact/energy after
   each call and qpos/qvel POST state separately. Add no observer forward.
5. For the original action, require qpos27, qvel26 and the full target300 vector
   to match the original target bytes/SHA. Check unchanged source and model.
   Preserve any mismatch, partial trace and counters; no repair, skipped state or
   automatic retry.

A deterministic four-state fixture can use the first VAL family of each source,
both dt, at interval2: two64 plus two128 steps equals384, within512. This exercises
nonzero motion/history; the old192-step golden fixture already covered cold
initialization. Root owns the final selection. Testing two distinct full-rollout
methods on allfour would exceed this budget: test the exact proposed production
path, with any zero-step copy equality check separately counted.

For later action probes, every branch starts with identical captured integration
context; only the selected legal u coordinate changes. Preserve signed actual du
and zero/missing/bound masks. These remain conditional finite responses from one
model/context, not an infinitesimal20D Jacobian, force derivative, Markov proof,
real ego generalization or holding qualification.

# Full device-model reconstruction and identical-input repeatability

v0.64 does NOT pass: saved-parameter replay56/72 and live-readonly-model
comparison16/32. Full100ms object position/orientation/velocity gates fail at
both dt, despite correct controls, initial states and EFC identities. No MPC
or neural training is started; do not relax those gates or claim readiness.

The live-model probe finds spread among128 identically initialized worlds,
so reconstruction error cannot alone explain drift. It also finds seven
different derived-array fields after native setConst/put_model. The old15
physical-table context remains valid as that declared subset, but must not
be described as a byte-exact device-model snapshot. Old artifacts remain.

Capture every nested device-model array, including tuple entries, plus every
scalar/option/callback leaf. Reconstruct an independent model, require exact
scalar/key/shape/dtype compatibility before mutation, restore every array and
verify all array hashes. Keep callbacks None; fail unsupported leaves rather
than serializing unknown runtime behavior. Save the full snapshot and pins.

In fresh original tasks at both5/2.5ms, compare LIVE_READONLY and
FULL_ARRAY_SNAPSHOT bindings with128 identical worlds, three same-buffer
reset/repeats, original constant raw action/EMA and phase1..5. Preserve full
clone qpos/qvel/warm/time/control histories and fresh actual task traces.
Compare each repeat and within-batch spread over the SAME100ms horizon using
the original v0.64 tolerances. Keep exact reset/control/capacity/no-aid and
model-preservation checks separate from physical-fidelity gates. Distinguish
array fidelity, repeatability, model-versus-live prediction and task success.

Do not attribute observed spread to a specific kernel without causal evidence.
Installed primary source contains atomic constraint-row allocation and the
upstream forward test explicitly sorts EFC rows for comparison; these suggest
possible numerical order effects, not a proven sole cause. Do not modify the
installed simulator, CUDA/driver, physics/material constants or acceptance
thresholds. Preserve source pins for earlier rollout-class revisions.

If100ms fidelity still fails, document the uncertainty and redesign the
prediction/validation contract before any physical planning claim. A shorter
horizon passing alone cannot be presented as successful100ms equivalence or
full task success. Full SuperMimic+CHORD Goal and future first healthy neural
checkpoint restore/ETA pause remain unchanged.

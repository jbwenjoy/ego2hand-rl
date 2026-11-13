# Research plan v0.56: actual reference-conditioned native feedback

2026-11-14. Previous goal turns made progress: fixed DIRECT/IMPULSE2000 fits
completed with36/36 state checks; ef14bca added verified actual-failure and saved
prediction videos. User resumed the Goal. Native closed-loop evidence is now
needed; do not launch another surrogate fit and equate it with control progress.

## Evidence and limits carried forward

Final independent small-action object/contact response NMSE was DIRECT
.946092/.933627 and IMPULSE.955084/.943257 against zero-response prior1. The
coupled model improved algebraic momentum closure but not causal response or
200ms recurrence accuracy. Per-axis diagnostics also retain false predicted
effects for controls whose measured native object response is near numerical
zero in this narrow dataset. Do not use these as qualified imagined gradients.
The displayed actual CHECK seed1000 trace terminated at.503125s; rendered model
predictions were not policy execution. No physical holding success was shown.

## New native branch parity, already executed

`NativeBranchOracle` owns a separate MjData, replays actual EMA controls, and
uses batched native stepping with original model/material/gravity/timestep.
18 configurations (two dt x k0/3/6 x three action profiles), each three20ms
intervals, match the original observed split-step reader's FULL numerical
integration state, observations, controls and EMA byte-for-byte. Active plant
states remain unchanged by prediction.5184 oracle and5184 observed integration
steps cost.61725s and1.81788s respectively for this small parity workload.
This is not a general GPU/learning speedup or full-controller validation.

The pinned3.11.0 engine dispatches both normal and split steps to implicit
integration for the original implicitfast setting; use executed parity as
authority rather than an assumption about integration API equivalence.
[Official3.11.0 source](https://raw.githubusercontent.com/google-deepmind/mujoco/3.11.0/src/engine/engine_forward.c).
No change to original stepping, assets or historical sources is made.

## Fixed eight-case controller protocol

`configs/mug_native_tracking_v1.toml`: HOLD and the actual existing anchored
REFERENCE, ZERO and NATIVE_MPC, original two dt, one fixed initial state. All
cases run to original2s budget or original50mm/45deg abort. The reference's
9.57mm/.06043rad pose range lies inside those abort bounds; no time stretching
or object force/gravity aid. HOLD is a diagnostic, not a substitute ego task.

At each actual50Hz control instant, the planner uses the current complete
native state, actual EMA memory, previous action and known future reference.
It predicts a60ms horizon with a constant raw20 action, applying EMA at every
20ms interval. Only joints0..11 of the three DECLARED contact fingers are
optimized; other8 raw residual actions stay zero, their original actuators and
base targets remain enabled. This restricted diagnostic is not a complete
20-axis learned policy. No NN weights/check data are used by the controller.

One bounded finite-difference Gauss-Newton proposal, ridge.1, action probe.1,
max step.5, four fixed line-search scales. The cost includes future COM/rotation/
velocity/omega errors, joint posture, three-finger force deficit, nonpad/table,
penetration/self/limit penalties and action size/smoothness. Two incumbent
queries plus24 FD and four line-search queries per decision, chosen cost cannot
exceed the best of zero/previous incumbents. Box constraints apply to optimizer
proposals explicitly, not hidden clipping of the executed controller signal.
Contact costs sample endpoint solver-PRE values, not interval-mean labels or
full-tick qualification. Contact nonsmoothness means FD is a local secant,
not an analytical physical gradient certificate.

Original full per-tick observed reader executes just the first action, without
copying predicted object poses/forces into the plant. Compare each complete
executed interval's full integration state against the selected first forecast;
retain partial terminal intervals explicitly. At k0 and5, repeat decision from
restored planner/plant memory; require exact candidate/action/cost replay.
No best-case selection, reset/seed replacement or hyperparameter sweep.

Global oracle budget4,000,000 native integration steps, all branches/replays
charged; execution and parity steps separately recorded. Save full executed
states/controls/contact traces, every proposal/action/cost/residual and decision
latency. This CPU privileged native baseline is intentionally expensive; report
whether it can run within20ms, never imply deployable timing or learned-policy
benefit. Actual corrected trajectories, if any, must pass qualification gates.

## Separate task qualification

All cases retain original native collision, self/nonpad/table-force, joint-limit,
warning and tail three-finger force coverage gates. HOLD additionally requires
the original final drift/angle and tail velocity gates. REFERENCE requires full
2s, COM RMS<=5mm/max<=10mm, angle RMS<=5deg/max<=10deg, velocity RMS<=.05m/s and
omega RMS<=.5rad/s against the declared source-clock reference. Substep goals
use linear COM/velocity interpolation and SLERP orientation on its50Hz grid.
Original static-hold gates are also reported separately for every case; they
are not silently weakened into tracking gates. Shorter survival is not success.

No new neural training/checkpoint is launched in this native-control stage.
The standing first-healthy-neural-checkpoint pause rule remains for subsequent
learning. Keep Goal active while collecting/reviewing this native evidence.

## Full research scope remains

Use native outcomes to establish the reference/controller/observation contract
and actual no-aid data before reference-conditioned Actor/WM experiments and
matched task-level contact support comparisons. Privileged native planning cost
must be included in any teacher-data/shared-model amortization claim.

Controlled cube/pen loading/playback, short training/save-resume and unified
no-aid evaluation; object-frame support and matched A1/A2, physical contact/
action gradients; ego collision/inertia/material/gravity/frame/input/controller
contracts; independent RGB and full trajectories; multiple trajectories and
TOTAL shared-versus-per-trajectory cost; hardware subversion/joint/zero/gain/
latency/observations are still required. Same nominal mug/reset is not a replacement.
No driver/CUDA changes, duplicate workspace study directory or full Goal success.

Evidence: `runs/mug_native_oracle_parity_v1/report.json`; actual new control
results belong to `runs/mug_native_tracking_v1/report.json` if independently
generated. This preregistered plan alone is not evidence of control success.

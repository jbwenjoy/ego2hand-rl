# Research plan v0.50: causal H20 data and bounded first response checkpoint

2026-11-08. Previous turn was progress: native projection results and an anchored
reference contract were committed as36bc80e. This turn implements the bounded
20D residual/EMA controller, actual state55/solver-memory adapter and role-separated
native H20 data. Training status belongs to the independent health artifact;
the new entry is explicitly limited to first10 checkpoint/readback, then pause.

## Proven before training

- Same original asset/gravity/material,50Hz controls and asymmetric actual joint
  target room. Both dt zero-action baselines match all16 historical per-tick
  trace arrays byte for byte.6 full numerical-state/controller restore checks
  match. No new Actor or external object force.
- State55 POST observations distinguish BODY origin/COM and world velocities;
  interval simulated wrench labels use solver-PRE movingCOM, rotated to source
  BODY axes. Numerical warmstart26 and complete native integration state retained.
  Human/source contact truth remains unknown. Pure NumPy/Torch EMA math shared.
-16 TRAIN episodes378 fullH rows,8 independent-seed/dt CHECK episodes193 rows,
  48 heldout paired-response branches.24 partial intervals retained/excluded
  from fixedH fitting.54333 native steps; no old328 heldouts consumed. No2s
  trajectory success. Same reset/task is not multi-trajectory or RGB generalization.
-7/7 saved input/label/control/role checks,119 tests. Existing frozen phone file
  bytes unchanged. Added explicit new task module/config do not change old imports.

## Authorized next action and stop boundary

1. Run the fresh response WM on CUDA with reviewed packed data, TRAIN-only frozen
   scales and resident device tensors.102 inputs (state55,warm26,executedU20,dt)
   and87 outputs (delta52,nextwarm26,wrench6,Fn3); no legacy schema/resume. No
   per-update geometry/native calls, scalar health synchronizations or transfers.
2. Fit only TRAIN. Report CHECK prediction and paired-response errors against
   zero-response prior. This is a one-step data/model baseline, not Actor training,
   reliable physical gradient, validated reward or demonstrated algorithm benefit.
3. Stop at update10. Atomically save full model/optimizer/scales/RNG/sampler/data/
   source configuration state, verify exact readback/next sampling/prediction,
   estimate conditional time for the remaining990 planned updates, then PAUSE
   the Goal as the user requested. No optimizer update after checkpoint10 merely
   to test restoration. A bad health artifact remains evidence, never success.

## After an explicit resume

1. Use reported response errors to decide continued supervised fitting and data
   coverage, including independent physical action/contact direction checks and
   numerical solve uncertainty. Failed transitions are data, not demonstrations.
   Warmstart prediction and autoregressive physical validity require validation.
2. Build actual reference-conditioned feedback/Actor observations with current
   controller memory and future commands; joint-feedback exploration alone is
   not object-reference tracking. Validate native short-horizon response and
   no-aid returns before relying on imagination gradients or contact rewards.
3. Match A1/A2 budgets, collect fresh role-separated interventions and retain
   independent CHECK/heldout contracts. Include all data/simulation/fit/compile/
   validation/retry cost; do not reuse old phone frozen data roles or baselines
   as compatible mug claims. Full2s and full trajectory success remain unproven.

## Full objective remains intact

Controlled fixed-wrist Hand2 cube/pen loading/playback, short learning/save-resume
and unified no-object-aid evaluation; trustworthy object-frame support objectives,
same-budget A1/A2 and physical action/contact gradients; collision, inertia/
material/gravity/frame/controller/input contracts; independent RGB ego and full
trajectories; multiple trajectories and shared-versus-per-trajectory total cost;
hardware version/joint/zero/gain/latency/observations before deployment. The new
causal data/controller and first warmup checkpoint do not complete that scope.
No driver/CUDA changes, duplicate study roots or full-goal completion claim.

Evidence: [H20 data and checkpoint recipe](mug-H20-data-and-first-checkpoint-2026-11-08.md),
`provenance/mug_H20_data/summary.json`, and subsequent independent training health.

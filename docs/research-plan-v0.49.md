# Research plan v0.49: closed-loop contract and response data

2026-11-08. Previous turn was progress: support-selected contacts and4 negative
native cases were committed as6e19de5. This turn completes6 bounded inverse/
forward/acceleration projections,12 failed native cases and an anchored50Hz
input/reference contract. It also diagnoses the finite-solver balance gap and
preserves the interrupted source plus reused results. Goal remains active;
no neural learning/checkpoint or standing checkpoint-pause event occurred.

## New established evidence

- All6 final proposals clear3025 component pairs and actual source candidate
  regions, but12 native cases still fail within.180–.528s. None passes static
  balance diagnostics or complete2s holding.17950 unique integration steps,
  including3459 reused rather than rerun; no auxiliary object forces.
- Default tolerance1e-6 yields a6.065e-5 componentwise Ma/net gap at the old
  LP seed. Tight1e-10 diagnostic-only query yields1.11e-15, withqacc difference
  .02845. Primary holding model/gates unchanged. Final proposal readbacks are
  reproducible. Record solver uncertainty; do not demand exact linear-solve
  equalities from every finite Newton termination or claim smooth derivatives.
-7105 bounded evaluated parameter probes;127 secant columns change contact
  signature. Small torque residual can still imply large light-joint response.
  Acceleration scaling did not solve holding. Local ftol/xtol exits do not prove
  global infeasibility or justify continuing static search indefinitely.
- Source61 timestamps span2000004434ns. New integer20ms grid yields101 poses/
  100 control intervals,4434ns unsampled tail, no time stretch or extrapolation.
  Rigid start anchoring preserves relative object poses, not world gravity or
  human joint retargeting. Constant grasp q and distinct feedforward PD target
  are separate. Reference readback7/7; full library114 tests; trace156/156.

## Next executable stage

1. Implement the declared new bounded physical joint residual and20D EMA-memory
   controller at50Hz. Verify asymmetric room near legal control limits, exact
   update cadence, initialization, previous/filtered/executed target identity
   and save/restore. Keep the robot root and actual actuator/dynamics priors
   fixed. The current reference contract is DRAFT, not an integrated trainer.
2. Implement actual native observation/label adapters with named joints, BODY
   origin versus movingCOM, world/BODY velocity/wrench axes, compound contact
   aggregation and optional uncertain-source intent fields. Do not reuse frozen
   phone model/normalizer/checkpoint semantics merely because dimensions match.
   Controller state and actual executed targets must be included in dynamics.
3. Run the101-sample reference under the baseline and bounded paired actions;
   collect explicitly new role-separated native transitions, including drops
   and contact changes. Analytic static2s holding is NOT a prerequisite for
   learning: useful feedback tasks may start unstable. Require finite, legal,
   reproducible, no-aid physics and correct data/contracts instead. Do not
   silently turn uncertain source contact or solver-padding zeros into labels.
4. Check independent response/contact/gradient directions and native rollout
   return sensitivity before claiming imagined control reliability. New TRAIN
   interventions and independent CHECK seeds remain distinct; never fit the old
   328 heldouts. Then bounded response-supervised world-model/Actor work and
   matched A1/A2 with total data/fit/compile/validation/retry costs.
5. At first healthy NEW neural checkpoint, verify model/optimizers/RNG/sampler
   and exact saved-state restoration, estimate completion time and PAUSE the
   Goal as requested. This standing rule must not be delayed to finish training.
   No such checkpoint has been created by the present mechanical searches.
6. Advance full61-frame/50Hz task execution and independent RGB/source-gravity
   evidence in their stated scope. The anchored fixed-wrist task is a declared
   source-informed variant, not original gravity/whole-human trajectory truth.

## Retained full objective

Controlled Hand2 cube/pen physical loading/playback, short training/save-resume
and common no-object-aid evaluation; validated object-frame wrench targets and
same-budget A1/A2 benefit; WM physical action/contact gradients; collision,
material/inertia/gravity/frame/data/controller contracts; independent RGB ego
reconstruction/full trajectories; multiple trajectories and shared versus
per-trajectory total costs; hardware version/joint/zero/gain/latency/observations
before deployment. No annotation FK, static projection, input contract, changed
physics or training-set result substitutes for these requirements. Preserve
failures, source snapshots and the single maintenance repository. No driver/CUDA
changes or full completion claim. Static failures do not block meaningful next
closed-loop data and learning work.

Evidence: [native projection/reference results](mug-native-projection-2026-11-08.md),
`provenance/mug_native_projection/summary.json`, prior v0.48 and retained cases.

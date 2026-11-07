# Research plan v0.48: native force/controller consistency

2026-11-08. Prior turn was progress: material-point/normal and source-region
engagement results were committed ascb1933d. This turn compares counterfactual
gravity targets, enumerates source triplets, fits4 robot proposals and runs4
native no-aid cases. One pose passes geometry, chosen uncertain source regions,
native6D cone and strict actuator-box balance, but every dynamic case fails.
The Goal stays active. No new neural training/checkpoint; the user's first
healthy new neural checkpoint pause condition has not occurred.

## Evidence that changes the next action

- Current versus conditional source Z-up gravity differs129.574deg. All6
  aggregate source candidate envelopes balance both targets, but the3 prior
  native manifolds balance neither in6D. Only robot gravity passes their force
  balance. This rejects gravity rotation alone as an explanation for those
  selected manifolds, not the need to measure sequence gravity or source intent.
-1112 one-point-per-finger triples,4/8-edge hypotheses and2 targets:4448 LP calls,
  not physical trials. Primary robot-gravity feasibility130/896 and28/216 for
  full30/nonnegative21 candidate sets. Force minima13.04/13.26N are hypothetical,
  unbounded/actuator-free candidate results, not real human or robot forces.
-4 selected triplets,72 anchor fits and4 coupled searches:2 final geometry/source
  matches,1 actual native6D/strict-command feasible. This positive static result
  uses a source point with negative local face offset; preserve its uncertainty
  and the failed nonnegative-offset branch, not a contact-truth claim.
- Newpose constant-q targets fail at.142188s in both dt; LP-derived legal targets
  fail at.519063/.527188s.6400 integration steps,0 complete2s holds. Actual reset
  force/moment and joint accelerations are not equilibrium, even with totalFn
  larger than the hypothetical ray sum. LP success is not native attainability.
-108 tests;27 envelope/710 triplet feasible primal readbacks,6 separators,6 full
  geometry reruns,52 direct native EFC columns,2 native primals and52/52 saved
  failed-trace checks pass. Frozen phone source/config/data roles unchanged.

## Next executable stage

1. At the new native6D-qualified source-conditioned pose, compare actual native
   contact force and generalized residual against the hypothetical LP witness
   under legal PD targets. Build a bounded real-state/control projection or
   response-based correction, with explicit finite differences/gradients and
   mode changes. Force/moment and hand equilibrium must be checked together.
   Do not merely increase total grip force or assume inverse-PD equals force.
2. Keep root, original material and gravity fixed. Any hand/object displacement,
   contact-region deviation or reference change needs its own recorded bounds
   and source intent comparison. Avoid silently imposing artificial q margins
   that reject a legal pose; preserve actual joint/control limits and full3025
   ideal-component checks. Static least-squares status alone never qualifies.
3. Run unchanged complete2s/two-dt no-aid gates and native consistency checks on
   useful real-contact candidates, with failures retained. Then separately
   declare mass/friction/time sensitivity. No partial-duration, sphere gauge,
   source-cone or changed-physics result substitutes for physical qualification.
4. Rebuild all61 timestamps, BODY/COM/controller observations and explicit50Hz
   resampling when a useful task exists. Obtain independent RGB reconstruction
   and actual sequence gravity evidence. Candidate geometry, negative local
   offsets and source annotations remain distinct from contact/force truth.
5. Acquire fresh role-separated TRAIN interventions; never fit old328 heldouts.
   Validate physical action/contact gradients, object-frame support objectives
   and matched A1/A2, including total cost, model fitting and native validation.
6. At the first healthy NEW neural checkpoint, check model/optimizers/RNG/sampler
   and full saved-state restoration, estimate ETA and pause as requested. The
   current LP/IK/native mechanics work is not that neural training.

## Full objective remains required

Fixed-wrist Hand2 controlled cube/pen physical loading/playback, short learning,
save/resume and unified no-object-aid evaluation; validated object-frame support
targets and same-budget A1/A2 benefit; WM physical action/contact gradients;
collision/dynamics/gravity/frame/data/controller contracts; independent RGB ego
reconstruction and full trajectories; multiple trajectories and total shared
versus per-trajectory cost; hardware version/joints/zero/gain/latency/observations
before deployment. No smoke, annotation FK, ideal cone, changed material/gravity
or training-set result replaces these requirements. Do not modify drivers/CUDA,
create duplicate study roots, mark the goal complete or pause before the standing
user condition. Continue in the maintained unified repository.

Evidence: [gravity and support-selected results](mug-support-selection-2026-11-08.md),
`provenance/mug_support_selection/summary.json`, prior v0.47 and retained failures.

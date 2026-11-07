# Research plan v0.45: wrench-conditioned contact placement and material sensitivity

2026-11-08. Previous turn was progress: one geometry-qualified grasp and10
negative physical tests were committed. This turn diagnosed native support,
checked it with EFC columns, tested LP-derived real PD commands, searched local
initializations and tried conditional-gravity placements.20 more native cases
failed; no2s hold, new neural learning or checkpoint. The Goal stays active and
the first-healthy-neural-checkpoint pause boundary has not been reached.

## Established new evidence

-24 saved PRE geometries reprojected at zero velocity:10 force-only feasible
  versus1 full6D feasible in the native pyramidal point-force approximation.
  At that point, numerical full-command support4.056N/optimistic friction4.317N
  exceeds2.4525N weight, but local-command support1.262N does not.192 LP attempts
  were176 actual solver calls. Empty manifolds do not imply certified zero
  global support. Native172 EFC unit columns and79 feasible primal witnesses
  match world/moving-COM wrench and hand-load columns independently.
- LP-derived legal PD commands on a distinct static-qualified pose extend
  termination from~.169s to~.62–.67s but fail. Local46-variable searches return
  xtol with substantial residuals, not equilibrium. Their native tests also
  fail at~.64–.84s; partial endpoint agreement is not complete2s consistency.
- Six conditional-Z-up yaw placements retain the robot root and original
  assets. One is static-qualified, its10 native cases fail at~.135s. Initial
  active-contact preload at three depths produces real three-finger contact,
  but no full6D static balance. Mere proximity/normal-force activation is not
  the missing objective; contact position/direction and moment capability matter.
-20 new native cases,33801 integration steps,260/260 saved checks,93 tests.
  No policy benefit, ego correction/generalization, calibrated dynamics or
  physical gradient certificate. All negative and operational-failure evidence
  remains retained, with no modification of frozen phone sources or data roles.

## Next executable stage

1. Condition grasp/contact proposals on actual six-dimensional support, using
   native EFC-derived BODY/moving-COM columns and actuator constraints. Penalize
   missing gravity/moment directions, rather than maximizing normal force or
   merely pulling every distal geom close. Inspect actual fingertip/handle
   clearance and alternative legitimate opposing patches; finite old sphere
   gauges are not finger-fit proof. Keep source intent and changed contact
   regions explicit, all-piece geometry, legal commands and unknown masks.
2. Review the contact material/mode prior inherited from pen experiments.
   condim3 has no direct contact spin couple. If testing finite-pad torsion,
   create separately pinned condim4 variants and predeclare geometry-based,
   explicitly unmeasured torsional-friction ranges. Do not silently activate the
   existing unused .005 entry or treat a stronger contact model as measured.
   Preserve the baseline and match q/control/geometry when isolating physics;
   separately label any subsequent optimized-pose result. Verify native local
   couple signs, COM transfer, EFC generators and aggregate force budgets first.
3. Retain the fixed palm-up robot root. For conditional-gravity-preserving object
   placement, explicitly report loss of rest-palm alignment, source marker errors
   and source-relative reference change. Obtain actual sequence gravity/clip-world
   evidence before transferring source force labels. A hanging or reoriented
   easier object task cannot silently replace source orientation/trajectory goals.
4. Qualify genuine native three-group support and full ideal-convex geometry,
   then execute unchanged complete2s/two-dt gates and declared mass/friction/
   contact-time/mode sensitivity. A cone-feasible snapshot, optimizer return,
   transient large force or longer failure time does not pass. Keep failures.
5. Rebuild61-frame reference, BODY/COM observations, controller/EMA/normalizers
   and explicit original nonuniform-clock to50Hz rules. Current static variants
   are not source-trajectory replay and cannot reuse phone checkpoints/latent
   frames just because dimensions match. Then fresh TRAIN interventions and
   independent heldout response/contact checks, never fitting old328 heldouts.
6. Only with useful physically verified tasks, specify response-supervised WM,
   physical action/contact gradients, object-frame support targets and matched
   A1/A2 learning. Preserve full data/cost/provenance scope. At the first healthy
   new neural checkpoint, verify saved state/RNG/sampler, estimate ETA and pause
   the Goal as requested. Static least-squares proposals are not that learning.

## Full objective remains unfulfilled

Controlled Hand2 cube/pen loading/playback, short training/save-resume and unified
no-aid evaluation; verified wrench-support targets and same-budget A1/A2 benefit;
WM contact/action/physical-gradient evidence; collision/dynamics/gravity/input
contracts; independent RGB ego reconstruction and full trajectories; multiple
trajectories/shared-versus-per-trajectory cost; hardware model/joint/zero/gain/
latency/observations before deployment. No smoke, annotation FK, changed gravity,
frozen-cone approximation or training-set result proves those requirements. Do
not modify drivers/CUDA, narrow the objective to these diagnostics, or mark the
full goal complete. Continue within the single maintained repository.

Evidence: [support and native trials](mug-support-2026-11-08.md),
`provenance/mug_support/summary.json`, v0.44 and retained prior negative results.

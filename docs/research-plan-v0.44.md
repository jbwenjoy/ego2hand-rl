# Research plan v0.44: support-aware mug initialization before learning

2026-11-08. Previous Goal turn was progress: committed61-frame FK, translated
palm mapping and compound Hand2 layout. This turn also made progress: checked
conditional gravity, implemented actual-collider IK and canonical moving-COM
contact wrench sums, produced one static-qualified grasp, and executed10 no-aid
native tests. All10 failed at~0.193–0.197s. The Goal remains active, incomplete;
no new neural learning/checkpoint or standing pause boundary was triggered.

## Evidence that changes the next action

- Official pinned HOT3D viewer uses Aria Z-up. Conditional source[0,0,-9.81]
  maps112–118deg away from current simulator gravity. This is not per-sequence
  gravity calibration or proof that clips preserve full-dataset world alignment.
  Keep priorUNKNOWN binding and the conditional diagnosis separately. Existing
  palm alignment changes the gravity-relative task; source force labels cannot
  simply be transferred.
- Five-marker IK can have submicrometre error while penetrating actual distal
  colliders by8mm and causing7 self intersections. Human near-surface skin may
  lie10–24mm from a tip marker. Retargeting needs surface/contact configuration.
- Native witness penalties plus bounded exact-pair restoration produced one
  candidate passing all2816 object/189self/21hand-table queries and three-group
  contact bands. Another candidate retained a123um self collision and failed.
- That qualified geometry and five legal constant PD commands failed bothdt
  native2s gates:0/5 paired successes,9343 actual steps. Significant transient
  normal force did not establish sustained support/moment balance. This is not
  global physical infeasibility, a policy experiment or source trajectory replay.
- Compound BODY/COM wrench sums match native external arrays in fixtures and
  saved10ms samples. Saved checks140/140;88 tests. This does not certify source
  contact force, calibrated materials, continuous all-state collision or WM
  contact/action gradients. Old phone results and source closures are unchanged.

## Next executable stage

1. Diagnose actual native initial and early-support contact manifolds with the
   new per-component reader: force directions, moments about moving COM, native
   pyramid/PD actuator limits, and gravity direction. Use full6D balance, not
   just normal magnitude or a force-only LP. Preserve PRE labels and legal
   commands. Do not fit any of the old328 independently heldout branches.
2. Search bounded initialq/command preload near the new candidate, or alternate
   object placements/contact patches if torque support demands it. Require real
   native three-group support plus all-piece geometry and no aid. Keep limits,
   robot root, mass/inertia/shape/material assumptions explicit; never scale the
   hand/cup or secretly widen geometry bands. The existing absolute-PD diagnostic
   is not a frozen phone residual controller or a qualified policy interface.
3. Compare a gravity-aware placement branch if needed. Preserving source-gravity
   orientation and exact rest-palm alignment are incompatible here under the
   current fixed robot root. Declare any changed mapping, source/reference loss
   and object pose adjustment; do not rotate the user's fixed palm-up robot to
   conceal the mismatch. Obtain actual sequence gravity/world-alignment evidence
   before claiming source contact wrench transfer.
4. Verify any new initialization with unchanged numerical gates at bothdt over
   complete2s, then mass/friction/contact-time prior sensitivity. Preserve failed
   schedules. Partial endpoint agreement or a static collision pass is not a hold.
5. Rebuild full61-frame/reference/observations/EMA/normalizers with nonuniform
   original timestamps and explicit50Hz interpolation/endpoint rules. Fixed-wrist
   relative motion is a changed task, not original arm transport. Do not reuse
   phone checkpoints, norms or descriptors merely because dimensions match.
6. Only after useful physical baselines, collect fresh TRAIN-only legal-command
   interventions and reserve independent heldout action/contact response. Specify
   response-supervised WM, physical gradient checks, object-frame support targets
   and same-budget A1/A2. New mug learning is not online300 resume. On the first
   healthy neural checkpoint: verify complete saved state, estimate ETA and pause
   the Goal per the standing user instruction.

## Full objective retained

Still required: controlled cube/pen physical playback, short training/save-resume
and meaningful no-aid evaluation; object-frame support targets and same-budget
A1/A2 benefits; WM action/contact/physical-gradient checks; collision/dynamics and
gravity/input uncertainty; independent RGB ego reconstruction; full-trajectory and
multi-trajectory shared-versus-per-trajectory cost evidence; hardware joint/zero/
gain/latency/observation contracts before deployment. No annotation FK, static
qualification, internal parity or failed native diagnostic proves ego generality,
success or training acceleration. Do not modify drivers/CUDA, shrink the goal to
the mug milestone, or mark the full research objective complete.

Evidence: [mug feasibility](mug-grasp-feasibility-2026-11-08.md),
`provenance/mug_grasp/summary.json`, previousv0.43/frame contract and negative
phone results. Next work is mechanical support and task mapping, not new learning.

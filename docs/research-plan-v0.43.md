# Research plan v0.43: mug finger fit, gravity and physical reference feasibility

2026-11-08. The preceding consolidation-status reply was no progress toward
the research objective. Revalidation found v0.42 still pending, so this Goal
turn executed its human articulation/frame and scene-composition steps.
This turn is progress:61 actual annotation frames, proper translated palm
mapping, a new Hand2 compound scene, all-vertex proximity, saved review and
scientific figures now exist. No neural training or integration was started;
the Goal remains active and no new checkpoint pause boundary has occurred.

## Established evidence and limits

- Pinned unmodified Meta UmeTrack CPU FK; right reflection/winding and22/20
  parameter behavior verified against independent NumPy FK.21 out-of-range
  frame/parameter values retained, not clamped. All61 original timestamps.
- Proper rest-palm frames include88.8565mm human and66.8069mm robot origin
  offsets. BODY differs from COM. Fixed-wrist mapping changes world64.765mm
  motion to10.550mm relative motion, rather than reproducing arm transport.
- nq27/nv26/nu20 Hand2+mug model,128 same-BODY pieces,2816 explicit pairs;
  original robot collision mesh/inertia/calibration retained.15/15 stage and
  15/15 independent saved checks;81 tests. q=0 has0 generated object contacts,
  is not a grasp, and has no dynamic qualification. No phone checkpoint reuse.
- 48,068 human vertex/source triangle unsigned queries show thumb/index/middle
  near the handle-side region, with ring/little further away. This is candidate
  intent, not force/contact truth or robot finger fit. Unknown source gravity,
  body material sign and continuous surface collisions remain unknown.
- Failed prefinal assembly/printing runs and exact executed sources retained.

## Next executable stage

1. Verify the actual HOT3D/Aria world gravity convention at the pinned source
   revision or locate per-sequence gravity evidence. Do not infer gravity from
   cup orientation. If unavailable, explicitly separate source kinematics from
   simulator-gravity exploration and retain this missing input in claims.
2. Measure actual Hand2 distal/sensor collision shapes against the complete mug
   union/source handle, not the old1/3mm sphere gauges. Establish named robot
   finger groups and legal source-surface target regions. Human LBS groups,
   fingertip landmarks and closest-point data are candidates, not direct joint
   targets. Do not truncate/copy22 parameters or scale robot/cup to force fit.
3. Construct bounded IK/grasp proposals in the unchanged fixed-root scene.
   Preserve hard limits, capsule/mesh collision geometry, original BODY/COM,
   explicit friction/contact priors and unknown masks. Qualify hand-self and
   nonintent object collisions over all pieces; union interior depth is not a
   concave SDF. Report failure rather than secretly loosening phone gates.
4. Define component-to-canonical-hand contact aggregation. Verify native force
   signs and moments about moving object COM with independent simple fixtures;
   all128 components share one mass/inertia and must not double-count wrench.
   Build a new task schema/reference/observations/EMA and finite known masks;
   do not reuse phone descriptors or normalization because dimensions match.
5. Preserve the original nonuniform61-frame clock when deriving50Hz targets;
   make interpolation/endpoint/missing-data rules explicit. Treat removal of
   wrist motion as a changed task. Report both original and mapped behavior.
6. Before learning, establish no-aid full-window native baselines at both
   timesteps and declared mass/friction/contact-prior sensitivity. A static
   proposal or low reconstruction residual is insufficient. If holding fails,
   diagnose wrench/actuation/geometric feasibility first.
7. Collect fresh legal-command-history TRAIN interventions and independently
   reserved heldout action/contact response data. Old328 heldout branches remain
   forbidden for fitting. Then specify response-supervised WM/physical gradient
   checks and same-budget A1/A2 stages. A new mug experiment is not online300
   resume. For new neural training, stop after first healthy checkpoint, verify
   full saved state, estimate remaining time, then honor the user's Goal pause.

## Full original scope remains required

Controlled cube/pen physical playback, short training/save-resume and meaningful
no-aid evaluation; object-frame wrench support and same-budget A1/A2 evidence;
WM contact/action/physical-gradient checks; asset/dynamics uncertainty and
source gravity/input contracts; independent real RGB ego reconstruction;
multi-trajectory sharing and matched per-trajectory cost comparisons; hardware
joint/zero/gain/latency/observations before deployment. No smoke, annotation FK,
changed-task/reference simplification or internal model parity establishes
ego generalization, physical success or training acceleration. Do not change
drivers/CUDA or mark this full objective complete on this milestone.

Evidence: [right mug stage](right-mug-frames-2026-11-08.md),
`provenance/right_mug_frames/summary.json`, [prior asset](mug-asset-2026-11-08.md),
v0.42 and the retained negative phone results.

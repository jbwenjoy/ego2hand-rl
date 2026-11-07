# Research plan v0.41: known-mesh right-hand mug intake

2026-11-08. The preceding Goal turn was progress: force-aware initialization
and moment diagnostics changed the next action. This turn executed patch
migration, improved its joint/query bottleneck, and screened existing ego
instances. Goal active; no new neural training or checkpoint/pause boundary.

## Decision

The 20-joint/cached phone slide search constructed one geometry-qualified
candidate, but both native no-aid cases failed around0.111 s. Keep every
negative outcome and do not simply repeat the same solver with more time.
No global dynamic impossibility or policy benefit has been established.

Next inspect the already downloaded mug BOP9, clip001884, right hand,
frames47–107. Source annotations and actual mesh exist; preview suggests
handle grasp. It remains an intake candidate, not reconstructed contact,
calibrated dynamics or mechanically qualified robot retargeting.

## Next executable stage

1. Inspect GLB hierarchy and transforms into the BOP object frame. Verify
   units/extents against models_info. Keep BODY/root, COM, wrist/hand and
   world frames explicit. Use manifest pins and ordinary package imports.
2. Build a versioned compound collider preserving handle opening and cup
   cavity; quantify approximation, probe voids/openings and retain unknown
   regions. Separate visual/collision meshes and component identities.
   Whole-object convexification is insufficient for handle grasp.
3. Declare mass/COM/inertia/friction priors and uncertainty separately from
   measurements. A visual mesh does not supply calibrated dynamics. Compile
   and check in the native Windows environment without driver/CUDA changes
   or object force aid.
4. Verify right-hand articulation, handle-contact intent, robot joint/frame
   mapping, unknown source gravity and fixed-wrist reference semantics.
   Preserve source timestamps/all61 frames. Rebuild the task/reference/
   observations/memory; do not reuse the phone bundle or its normalizers as
   if the new geometry/contact semantics were identical.
5. Establish useful no-aid native baselines at both dt and report original
   reference deviation. Define canonical per-hand contact aggregation and
   moving-COM wrenches across multiple collider pieces before labeling.
6. Then collect fresh TRAIN-only interventions and check actual finite
   action/contact response against heldout labels. Use a new warmstart
   schema, shared implementation and matched A1/A2 budgets, never a claimed
   resume of online300. For new neural learning, honor first healthy
   checkpoint verification, ETA and the standing user-requested pause.

## Full scope retained

Controlled cube/pen reproduction and meaningful no-aid evaluation; contact
targets/physical gradient checks; calibrated collision/dynamics robustness;
independent ego/RGB reconstruction and gravity/input contracts; full trajectory
execution; multi-trajectory shared-model costs against matched per-trajectory
baselines; hardware joint/zero/gain/latency/observations all remain required.
The mug intake cannot replace these or prove ego generalization. Preserve
all evidence; complete only after the full original requirements are proved.

Evidence: [patch migration and intake](patch-migration-2026-11-08.md),
`provenance/patch_migration/summary.json`, prior initialization and online300.

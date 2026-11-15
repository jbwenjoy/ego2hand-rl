# Response supervision after branch augmentation: explicit ablation, no Actor claim

Previous turn is PROGRESS: predeclared256 actual source branches and1792/1792
force receipts; selected349 TRAIN/64 fresh CHECK/80 original sequence CHECK;
independent3592/3592 readback; new matched2000-update models with38/38 health
at100 and2000;181 tests. Old data/source roles and models remain frozen.

Fresh branch point error improves, original sequence point error worsens,
and pooled physical response error remains about0.91–0.92 for IMPULSE.
Endpoints and model secants/AD remain poorly aligned with actual effects.
This changes the next action: test the response-learning objective, rather
than another epoch extension or another undirected collection sweep.

1. Build a pinned contrast dataset using eligible NEW_TRAIN pairs ONLY:
   identical source55/warm26/native dt/physical context, valid native labels,
   actual unclipped +/- PD offsets. Current raw exact-source TRAIN pairs
   number26; eligibility was recomputed, not assumed. Artifact
   `runs/pen_TRAIN_contrasts_v1` now has26 eligible pairs,5/5 preparation and
   independent6/6 saved-array readback. No contrast optimizer has started.
   Preserve all349
   pointwise TRAIN rows and frozen V2 scales. CHECK64, sequence CHECK80,
   previous54 physical diagnostics and historical records NEVER become TRAIN.
2. Predeclare finite-contrast auxiliary weight0.1 and pair batch32, using
   object12 and measured source-BODY wrench6 responses normalized by the same
   frozen V2 y-scales. Contrast is (y_plus-y_minus)/(2*epsilon), NOT a claimed
   infinitesimal derivative. Use the corresponding predicted endpoint contrast
   so training does not require unvalidated second-order physical gradients.
   Keep all native numerical-quality gates and original point losses.
3. Run a matched point-only versus point+contrast ablation for BOTH DIRECT
   and IMPULSE. Same seed7411 initial networks,349 rows/scales, point-batch256,
   point-sampler stream, optimizer settings and2000 updates. Use a SEPARATE
   matched pair-sampler stream(seed7751), saved/restored explicitly; it must
   not alter point sampling. Save healthy100/2000 checkpoints with actual
   point/pair budgets. Additional pair forwards/backwards are extra cost,
   not equal FLOPs or proven pipeline acceleration. No CHECK-selected weight,
   checkpoint, direction or row filter. Existing V2 remains frozen.
4. Re-evaluate on the same declared development CHECK/sequence/physical
   conditions with common physical scales, all per-pair failures, pooled
   responses and recursive predicted state/warm. This is development evidence,
   not a fresh final holdout after those conditions informed this plan. Fresh
   action/state/clip evaluation is required before broader claims.
5. For later-source physical identification, capture ONE actual no-aid
   integration state and replay repeated +/- responses from that state,
   instead of executing separate stochastic prefixes and assuming equality.
   Record q/v/control/warm/time, required controller/command state and fixed
   physical arrays; verify restore completeness and exact common source.
   Check20ms plant fidelity/force labels/capacity under the strict policy
   against actual native execution before using clones or their gradients.
   No silent solver/model changes, artificial stabilization, threshold
   relaxation, source rounding or object force assistance.

Sparse known-state response improvements do not qualify a learned-model
Actor. Cube/pen native no-aid baseline learning/control and matched support
policy objectives remain required, alongside ego asset/input contracts,
multi-trajectory total costs and hardware observations. Preserve those
requirements; do not redefine the Goal as completing this ablation.
Goal remains ACTIVE; no training/collection is currently waiting.

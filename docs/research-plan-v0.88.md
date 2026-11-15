# Reviewed candidate data: fixed-budget continuation and physical response evaluation

The preceding first100 run is PROGRESS: new DIRECT/IMPULSE models reached100
updates each on valid TRAIN3858/CHECK1773;43/43 checkpoint checks passed.
No Actor or physical task benefit is established. The subsequent video turn
only verified/displayed existing native trajectories; it did not train models.
The full Goal remains ACTIVE under the existing resume authorization.

1. Preserve first100 checkpoint, health, frozen source receipt and data bytes.
   Run the maintained continuation entry's dry-run, then restore model, Adam,
   sampler, scales and physics exactly. Complete1900 further updates per arm,
   ending at the originally planned2000, with unchanged batch2048 and all
   hyperparameters. CHECK metrics at500/1000/2000 describe a fixed budget;
   they do not select a model or tune training. Count clone updates separately.
2. Verify final save/reload, Adam/cursor2000, matched sampler and next-update
   parity. Record timings and device-wide telemetry with its sampling limits.
   The first100 conditional estimate is12.57s remaining compute, excluding
   setup, save/QA and downstream evaluation. Do not claim historical speedup.
3. Evaluate one-step normalized error on valid TRAIN and CHECK; separately
   report all CHECK and excluded CHECK with explicit uncertain-label status.
   Excluded rows never enter fitting and are not promoted to accurate targets.
4. Evaluate all20 plus/minus actuator pairs for each of the12 CHECK parents
   (240 attempted pairs). Use all three native repeats per sign and the shared
   data-quality mask; retain ineligible pairs and reasons. Derive the actual
   direction and half-span from recorded PD targets rather than assuming an
   unclipped action. Require identical source55/warm26/dt across both signs
   and repeats; evaluate AD at the actual midpoint, and the model secant at
   the actual endpoints. Compare to mean independent native responses in
   object12, source-BODY COM wrench6 and Fn6, normalized by frozen TRAIN
   scales. Report pooled and per-parent errors, zero-response comparator1,
   and observed3-repeat spread. No infinitesimal derivative/smoothness or
   statistical confidence claim follows from these finite-amplitude contrasts.
5. Branch candidates are not consecutive trajectories. Do not run the old
   recursive evaluator on their row order. Any later recursive evaluation
   must use actual consecutive saved controls/states with explicit continuity
   checks and keep development parent episodes separate from untouched tests.
6. Keep prior DIAGNOSTIC_NEVER_FIT cohorts out of training. No blind Actor
   optimization before physical-response and recursive quality is reviewed.
   Archive exact source/lineage receipts, compact results and limitations.

Still required: learned policy residual/support same-total-cost A1/A2, complete
cube/pen no-aid task gates and save/resume, multi-trajectory adaptation costs,
true ego input/coordinate/asset/dynamics contracts and hardware observation/
zero/gain/latency verification. No healthy component fit completes this scope.

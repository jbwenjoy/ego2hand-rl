# Research plan v0.46: finger-conditioned contact proposals

2026-11-08. The preceding user-facing turn verified the existing consolidation
and import/artifact paths; it did not advance physical feasibility. This turn
completes and retains matched native contact-mode trials, full couple/EFC
readback and first-frame candidate-patch diagnostics. These are concrete new
physical/implementation results, not new neural training or a checkpoint. The
Goal remains active; the user's first-healthy-neural-checkpoint stop rule has
not been triggered.

## Current evidence and limits

- Two fixed static-qualified poses, three separately pinned contact models and
  two dt values: 12 failures, 23,485 actual integration steps, no complete 2s
  hold. Baseline A trace arrays match both historical dt traces byte for byte
  before variant integration. Only 1152 declared pad/object pair dimensions
  and spin coefficients change. 1/3 mm torsional coefficients are unmeasured
  exploratory priors, not calibrated patch radii. Torsion does not fix these
  two selected poses. It does not establish global material/control failure.
- 224 native EFC unit columns, 28 feasible LP primal readbacks and 156/156
  saved trace checks pass. Frozen static cone feasibility, even with command
  bounds, does not establish compliance-compatible native dynamic support.
- First-frame annotation geometry yields 30 source candidates, 9 with negative
  local face-normal offset; retain the 21-point sensitivity subset separately.
  Source contact truth and real force remain unknown. Approximate collider
  probes are not source signed distance/watertightness certificates.
- A failed pose has whole-hand unit-force similarity .9368 while per-finger
  similarities are .514–.589; its index contact is 22.07 mm from the nearest
  same-finger source patch with opposite nearest normal. This motivates
  correspondence diagnostics, not a claim that a new reward is validated.
  NumPy ray/COM/score/patch readback passes 9/9; full library tests pass 99/99.

## Next executable work

1. Build bounded source-patch/finger/normal-conditioned pose proposals in the
   maintained mechanics package. Keep source candidates, true contact labels
   and unknowns separate. Test the full and nonnegative-offset source sets as
   explicitly different hypotheses; do not fit fictitious human force values.
   Preserve finger correspondence in distance-and-normal pairs and support
   diagnostics, while allowing declared morphology-induced deviations.
2. Check actual distal-link/finger fit and opposing contact access around the
   handle, using full convex components and hand bodies. Old sphere probes,
   visual overlap or tip-only IK are insufficient. Report local infeasibility
   or intent loss rather than silently substituting an easier contact task.
3. Score candidate native manifolds with six-dimensional moving-COM support,
   hand load and actual actuator constraints. Keep contact-mode/material priors
   pinned and distinct. Use the original point-contact mode as the primary
   comparison; do not keep increasing friction, pad spin or normal force until
   a favorable result appears. A static cone/optimizer return alone is no pass.
4. Qualify complete geometry and legal controls, then run the unchanged 2s/two-dt
   no-aid gates. Report separate sensitivity for material, mass and contact
   time only after a useful baseline exists. Keep changed poses/frames separate
   from frozen phone resumes and source trajectory success claims.
5. Rebuild all 61 actual timestamps, BODY/COM features, controller state and
   declared resampling to 50 Hz. Preserve fixed palm-up wrist, explicit loss of
   palm alignment in conditional-gravity branches, and unmeasured sequence
   gravity. Annotation FK/proximity are not independent RGB reconstruction.
6. With a useful physical task, acquire fresh explicit TRAIN interventions and
   independent heldout response/contact checks; never fit the old328 heldouts.
   Verify physical action/contact gradients and object-frame support targets,
   then run matched A1/A2. At the first healthy NEW neural checkpoint, verify
   model/optimizers/RNG/sampler saved state, estimate ETA and pause as requested.
   No neural restart is justified solely by more informative static scores.

## Retained full objective

The original controlled Hand2 cube/pen loading/playback, short training and
save/resume, common no-object-aid evaluation; trustworthy object-frame support
targets, same-budget A1/A2 and WM physical action/contact gradients; collision,
inertia/material/gravity/data/frame contracts; independent RGB ego reconstruction
and full trajectories; multiple tasks/trajectories and matched total/shared versus
per-trajectory cost; hardware model, joints, zeros, gains, latency and observations
before deployment all remain required and unproven in full. Do not replace this
scope with the mug static diagnostic, smoke, annotation reconstruction, a changed
physics prior or training-set behavior. No driver/CUDA changes, duplicated study
roots or completion claim. Continue in the single maintained repository.

Evidence: [contact-mode and patch results](mug-contact-modes-2026-11-08.md),
`provenance/mug_contact_modes/summary.json`, and prior v0.45/support evidence.

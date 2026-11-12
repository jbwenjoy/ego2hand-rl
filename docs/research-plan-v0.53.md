# Research plan v0.53: complete the fixed paired budget and check recursion

2026-11-12. Previous turn made progress: f473fea committed independently seeded
native intervention data and matched response supervision; paired checkpoint100
saved with26/26 structural checks. The requested pause was observed. The user
has now resumed the Goal. Continue the remaining1900 of the SAME fixed2000
paired updates, not a new sweep, data fit, loss change or CHECK-selected model.

## First100 evidence retained

At action amplitude.04 on independently seeded CHECK prefixes, ABS versus CF
model-AD response NMSE against zero-response prior1 was:

|Block|ABS|CF|
|---|---:|---:|
|delta52|.834417|.624031|
|object12|.982734|1.037485|
|contact9|.994366|1.052031|

Thus overall state response improved, but the object/contact components had
not improved. This first checkpoint is not a physical-control qualification.
The first100 two-arm update loop took1.34279s. Six device-wide samples had
utilization median1%, maximum27%, used VRAM maximum1957MiB/32607MiB. Short query
windows and other desktop workloads limit interpretation; no global SuperMimic
utilization/speedup follows from this tiny workload.

## Exact continuation and first new checkpoint boundary

`configs/mug_paired_response_resume2000.toml` points to the healthy paired100
artifact. A maintained common engine restores BOTH distinct arms' weights,
Adam1100, RNG, sampler and original frozen scales. It loads the original stage
recipe and unchanged source/data pins. No sampler reseed or normalizer refit.
Same3258 absolute rows,1440 TRAIN pairs, batch256/pair64, learning rate, clipping,
coefficient0/1, FP32 and deterministic settings remain. All CHECK sources remain
read-only. Historical executed sources and data bytes are unchanged.

The actual next retained update101 is compared against one expendable restored
clone update per arm: loss, both sampled batches, clipped gradients, complete
weights, optimizer and sampler cursor must be byte-exact. These two extra clone
updates are charged separately as restore verification; they do not enter the
retained optimization trajectory. Failure aborts, preserving evidence.

After parity, updates102..2000 use the same common update primitive with resident
GPU data and no native/geometry/host health work inside the loop. Atomic paired
checkpoint2000 is the FIRST and ONLY new checkpoint of this resumed stage;
both Adam counters are3000 including their original1000 warm start. Check full
readback, two next batches, RNG, predictions, frozen scale buffers, source/data
lineage and equal arm sampler cursors. Report regular1899 timing separately
from real step101/clone parity, startup, saving and diagnostic work. At2000 the
fixed paired budget is complete and has zero updates remaining; it is not an
ETA for completing the research Goal. After this healthy new checkpoint and
read-only diagnostic report, PAUSE Goal as the user requested.

## Predeclared read-only recursive checks

In addition to unchanged one-step/AD/secant diagnostics, evaluate56 recorded-
control CHECK cases from four seeds8200..8203 at both dt: start k2 with horizons
1/2/5/10 and start k7 with1/2/5. These are20/40/100/200ms. Inputs are already
acquired prefix controls, never future state/contact labels. Start each case
from its actual native observation/warmstart, then recursively predict state55
AND warm26. No teacher forcing, joint clipping, native simulation/aid or extra
learning. Decode rotations by the declared WORLD-left exponential; retain all
finite state/velocity/force errors and any failed predictions.

Report COM/rotation/joint/velocity errors by horizon. Rotate predicted mean
contact forces and couples through the model's own source BODY frame before
comparison with native world values. Also compare predicted momentum change
with predicted contact impulse plus gravity, using the compiled model's mass,
COM inertia and principal-frame rotation. Free-object damping and gravity
compensation must be absent. Native finite-step/solver momentum defect is shown
alongside, not silently replaced by an exact-zero assertion.

These checks can reveal contact/dynamics inconsistency hidden by low one-step
loss. They are NOT closed-loop reference tracking, geometry/collision validation,
successful2s holding, source-gravity preservation, independent RGB, policy A1/A2
or a CHORD reward benefit experiment. Do not choose or tune models on these
CHECK diagnostics. All new decoder/recursion and actual-update parity tests,
together with the full129-test suite, pass before research fitting starts.

## Next research step after explicit resume

Use fixed-budget outcomes to diagnose remaining object/contact and recurrence
errors, including their actual native numerical floor and training coverage.
Only use model gradients in control after appropriate physical short-horizon
validation; distinguish exploratory real-native feedback experiments from
claims of reliable imagined control. Establish reference-conditioned feedback/
Actor observations with EMA memory and future goals, then measure real no-aid
trajectory returns and matched task-level baseline/contact objectives.

Keep the original FULL objective: fixed-wrist Hand2 cube/pen physical loading,
playback, short learning/save-resume and unified no-aid evaluation; object-frame
wrench support targets, matched A1/A2 and physical action/contact gradients;
collision/inertia/material/gravity/frame/input/controller contracts; independent
RGB ego/full trajectories; multiple trajectories with total shared-vs-per-
trajectory acquisition/fit/compile/validation/retry costs; hardware subversion/
joint/zero/gain/latency/observation requirements before deployment. No drivers/
CUDA changes, duplicate top-level study roots or full Goal completion claim.

Actual continuation status belongs to
`runs/mug_paired_response_resume2000_v1/health_2000.json`, if generated. This plan
records predeclared scope and cannot stand in for completed training evidence.

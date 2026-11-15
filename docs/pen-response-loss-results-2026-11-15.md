# Finite-response supervision: controlled ablation and dynamic-state restore preflight

Previous goal turn is PROGRESS:256 fresh source branches expose poor physical
gradients despite better endpoint predictions;26 eligible TRAIN contrasts were
prepared. This turn executes the predeclared v0.73 four-model loss ablation
and tests full resident dynamic-state restoration. The full Goal remains
ACTIVE/incomplete; no Actor update, stable task policy, real ego result or new
policy-control video is claimed.

## Four matched models and frozen baseline replay

Code11edbff/config`configs/pen_contrast_fit_v1.toml` was committed BEFORE
running. POINT/RESPONSE x DIRECT/IMPULSE share identical initial NN weights,
349 valid TRAIN rows, frozen V2 TRAIN-only scales, width512, seed7411, point
batch256, AdamW settings and2000 live updates EACH. RESPONSE additionally
samples32 pairs/update from26 eligible NEW_TRAIN contrasts with independent
pair RNG seed7751. POINT sampling is unaffected by that RNG. CHECK64,
sequence CHECK80, previous54 physical diagnostics and historical data never
enter loss/scales/model selection.

The extra loss is predeclared weight0.1 times the sum of mean object12 and
mean source-BODY wrench6 squared error in finite-action contrasts, divided by
the unchanged V2 output scales. Contrast=(y_plus-y_minus)/(2*epsilon).
Predicted endpoints are evaluated as one64-row batch; this uses first-order
network optimization, NOT second-order physical gradients or assumed exact
infinitesimal derivatives. All original point/auxiliary losses remain.

`runs/pen_contrast_fit_v1` saves100/2000 checkpoints, each97/97 healthy:
model/Adam/point+pair RNG/sampler/cursor/scales/prediction/gradient readback,
finite losses/parameters/nonnegative Fn6, and exact cloned next-update replay.
POINT DIRECT and POINT IMPULSE reproduce frozen V2 model, Adam, point sampler,
gradient snapshot AND every original loss-log entry BITWISE at both boundaries.
Separate evaluation also reproduces their original CHECK predictions exactly
(2/2). This verifies the point-only controls; the response package changes
the optimizer objective without accidentally changing its control stream.

Final`pen_response_loss_checkpoint_2000.pt`:40,440,775 bytes, SHA256
`05608b2e0e3f9c4a973d0481fe79536ac37f92c033d96319ed61a8c805e96cf2`.
Four models x2000=8000 live optimizer steps;16 additional clone-only diagnostic
steps never advance saved/live cursors. Each model draws512,000 point rows;
each RESPONSE model additionally draws64,000 pairs/128,000 endpoints, with
replacement. This is NOT equal FLOPs or independent extra physical trials.
No Actor is updated. The process exited0 and no training remains live.

Four-arm fit INCLUDING metric calls/checkpoint QA takes38.7283s wall on
RTX5090. Median four-model CUDA compute/update18.5437ms;149 device-wide
telemetry windows report0–22% GPU use. Tensors/scales/pair labels stay resident;
no native simulation/geometry or host validation occurs in the hot update
loop. These small-data timings do not establish sustained utilization or
whole-pipeline acceleration. No additional data collection trained these models.

## Point predictions improve, physical AD remains inadequate

`runs/pen_contrast_evaluation_v1/report.json` uses the EXACT predeclared
development rows/physical-pair classification and the same original V1 TRAIN
output scales for every model, preventing denominator changes from explaining
the table. Only11/32 CHECK pairs have common source/warm/dt, valid labels
and unclipped PD offsets. All64 CHECK rows still enter pointwise metrics.

|Model|CHECK object12 MSE|CHECK wrench6 MSE|Sequence object12 MSE|Sequence wrench6 MSE|
|---|---:|---:|---:|---:|
|POINT DIRECT|0.04213|0.02181|0.67023|0.38692|
|RESPONSE DIRECT|0.01340|0.00609|0.49627|0.23645|
|POINT IMPULSE|0.02866|0.01550|0.54755|0.30775|
|RESPONSE IMPULSE|0.01900|0.00751|0.42643|0.21910|

Both response-supervised models improve these pointwise object/wrench
metrics versus their EXACT controls. That does NOT imply action-effect
accuracy. Pooled physical response errors over the same11 eligible CHECK pairs:

|Model|Object12 AD error|Wrench6 AD error|Object12 endpoint-secant error|Wrench6 endpoint-secant error|
|---|---:|---:|---:|---:|
|POINT DIRECT|1.049|1.093|1.065|1.101|
|RESPONSE DIRECT|0.998|0.949|0.998|0.949|
|POINT IMPULSE|0.912|0.922|0.915|0.923|
|RESPONSE IMPULSE|0.984|0.944|0.984|0.944|

Zero response has relative error1. Response-supervised IMPULSE is WORSE than
its control on these pooled AD metrics. Its median per-pair AD error also
worsens0.909->1.388(object)/0.840->1.301(wrench), even though worst-case errors
shrink. Per-anchor/per-pair failures remain in the report; no successful subset
or checkpoint is selected. Similar AD/secant errors do not support blaming
only AD implementation. Better endpoint fitting can reflect regularization/
representation effects; that explanation is an inference, not established
causal evidence about why gradients remain poor.

The11 pairs are correlated and span only8 initial-source pairs plus1 at each
later anchor. Each sign has one execution; fresh repeat/amplitude stability
was NOT established here. Conditions informed this ablation and are development
checks, not independent final generalization tests. No new physical derivative
or task-readiness gate is passed by lowering a neural loss.

Recorded-control recursion remains without teacher-forced state/warm, clipping,
native integration or resets (H1/H2/H5/H10 windows80/72/48/8). At100ms RESPONSE
DIRECT improves mean COM7.91->6.15mm/full angle35.37->28.01deg; RESPONSE IMPULSE
7.99->7.15mm/35.64->30.04deg. At200ms DIRECT improves19.91->18.53mm/
110.79->82.54deg but IMPULSE worsens13.61->21.48mm/70.43->116.23deg. Full
orientation includes axial roll and differs from the native shaft gate.
Only4 episodes support the8 H10 windows. These remain inaccurate open-loop
predictions, not stable native robot control.

## Complete dynamic-array restore: prerequisite, not rollout fidelity

Because separate prefixes rarely reach identical later sources, the next
physical identification step needs one captured source. A new shared
`runtime/gpu_data_snapshot.py` copies EVERY dynamic-array leaf on device using
the existing dataclass traversal, retaining original allocations for restore.
It validates all keys/scalars/shapes/dtypes/strides/devices/POINTERS before any
copy, so stale captured-graph pointers cannot be silently replaced. Unused
scratch NaN bytes are copied bitwise; active physical fields have separate
finite/no-aid checks. It never patches model parameters or repairs states.

Actual tool`audit_pen_resident_data.py`(f123d82) uses the SAME native captured
step graph at both dt, with2 initial-source single-physics-step probes each.
All probes occur in the DECLARED PRE-START initializer; the full source is
restored before the actual reference episode begins. No post-start state
restoration occurs in those native evaluation episodes.

`runs/pen_resident_data_audit_v1/report.json`:14/14 checks pass. Both devices
have169 dynamic arrays and8 scalar leaves; each repeat/final restoration
is bitwise exact, model arrays/scalars unchanged, and active states finite
without force aid. Exactly4 additional diagnostic physics ticks are counted
outside the actual episodes. The subsequent actual no-aid/no-reset trajectories
still fail at0.22/0.20s. Independent saved-contact review
`runs/pen_resident_data_EFC_v1/report.json` passes14/14 force receipts.

Copy correctness is NOT physical determinism:5ms repeated one-step probes
are identical for saved q/v/qacc/warm/time, but2.5ms qvel components differ
up to1.23e-7 and qacc/warm components5.34e-5 despite exact restored sources.
Units depend on each generalized coordinate. Those are descriptive raw
component differences, not newly relaxed fidelity tolerances. Later-source,
20ms response/fidelity and derivative qualification have NOT been performed.

The existing restore API deliberately rejects a different allocation. Future
counterfactual responses must use a separately constructed compatible Data
copy and its OWN graph; they must never overwrite the live evaluation state
after start. Preserve alias/layout semantics and prove the live source/model
unchanged. Apply the ORIGINAL step-audit force/state tolerances before using
cloned labels. No artificial stabilization or threshold relaxation.

## Validation and remaining goal

185 repository tests pass. Analytical contrast-unit/parameter-gradient tests,
separate sampler and complete next-step CPU restore tests, and nested full
data-byte/pointer/metadata-rejection tests supplement the actual CUDA checks.
After tightening the same-shape-pointer assertion, both targeted snapshot
tests pass again. The full suite intentionally isolates native tests on CPU;
its CUDA-unavailable text does not describe the recorded RTX5090 experiments.
Artifact/source pins, healths, compact results and tests are under
`provenance/pen_response_loss_results`. Original frozen artifacts, archives,
drivers/CUDA, installed packages and historical environments are unchanged.

Next: one captured ACTUAL source at0/40/80/120ms; independent resident Data
forks, repeated baseline/+/- responses over the exact20ms interval, and
comparison with untouched original native execution/force labels. All source
differences and failures remain. No further neural weight sweep before this
identification/fidelity check; see`research-plan-v0.74.md`.

The full original objective remains: fixed-wrist input/coordinate/geometry/
dynamics/controller contracts; actual cube/pen control learning and no-aid
evaluation; support-objective A1/A2 and physical gradients; multi-trajectory
total cost; true ego reconstruction/assets/validation and hardware observation/
gain/latency contracts before deployment. Engine warmstart is privileged
simulator context, not an ego/hardware observable. These component experiments
do not replace any unmet requirement. Goal remains ACTIVE, no process waiting.

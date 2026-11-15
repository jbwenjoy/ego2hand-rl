# Same-source native20ms replay passes; finite response remains amplitude dependent

Previous goal turn was PROGRESS: response-loss ablation did not qualify model
gradients, and initial single-step full-Data restore passed. This turn executes
v0.74's later-source20ms identification. It removes a source-confounding
problem and changes the next action toward finite-action native control.
The full Goal remains ACTIVE/incomplete. No network/Actor update or stable
task/real ego result is claimed.

## Read-only live sources, independent complete Data

Code31bbf30 and config`configs/pen_same_source_response_v1.toml` were committed
before collection. Actual original-clock baseline raw2*delta is run at5/2.5ms
from the same selected pose, environment seed42 and single upstream clip,
with the existing strict numerical profile and unchanged native terminal gates.
At control0/2/4/6 (0/40/80/120ms), capture full dynamic Data and controller/
command state without modifying the live trajectory.

`runtime/gpu_data_fork.py` clones all169 arrays/8 scalar leaves into independent
Data and an independent native step graph. It preserves exact internal aliases
and rejects unsupported partial/reinterpreted/strided aliases. Actual Data
contains125 nonempty independent allocations and no nonempty alias groups.
The live allocations are never restored or overwritten after episode start.
Existing pointer-safe ResidentDataSnapshot restores only the independent fork
between responses; no original controller/command step is executed there.

At EACH of8 sources: baseline and +/-0.002/+/-0.005rad along one unit-L2
direction from new PCG seed7801, each repeated3 times.120 held-absolute-PD
responses x20ms produce720 diagnostic physics ticks. Three repeat executions
measure implementation spread, not independent real trials. All120 remain
DIAGNOSTIC_NEVER_FIT. No model chooses perturbations or favorable cases.
The original actual episodes add124 physics ticks and still fail at0.22/0.20s.

`runs/pen_same_source_response_v1/report.json`:40/40 isolation/source checks
pass. Every response starts at bitwise identical full source arrays/scalars;
all live data, fixed model arrays/scalars, controller EMA/masks/targets and
command phase/joint reference remain unchanged around counterfactual work.
Every actual trajectory has one initial reset, zero post-start resets and
zero object/applied-force assistance.

## Original replay gates pass, including forces

Independent saved review`runs/pen_same_source_response_review_v1/report.json`:
1320/1320 source/control/time/mocap/finite/no-aid/label contracts pass;
actual EFC14/14 and all response EFC840/840 pass. The original numerical
linear-impulse threshold5e-7Ns is retained. No PD perturbation is clipped
(largest float32 target-rounding difference5.67e-8rad).

ALL24 baseline repeats match EVERY corresponding actual physics tick over
the20ms interval under the original gates from`pen_gpu_step_audit_v1.toml`.
This compares world-frame COM velocity/angular velocity and world COM
wrenches, not raw contact row order or incorrect object-first indices.

|Quantity|max error|Original gate|
|---|---:|---:|
|Hand positions|0|0.002rad|
|Object COM|0|0.0005m|
|Object full orientation|1.86e-9rad|0.01rad|
|Hand velocities|5.96e-8rad/s|0.2rad/s|
|Object WORLD linear velocity|3.98e-9m/s|0.01m/s|
|Object WORLD angular velocity|4.21e-7rad/s|0.2rad/s|
|Executed PD targets|0|2e-7rad|
|WORLD contact force|1.25e-7N|1e-4N|
|WORLD COM torque|4.49e-9Nm|1e-5Nm|

GPU time matches exactly. Thus these qualified20ms probes can support local
finite-response analysis at all four captured times. This does not supersede
older FAILED100ms/multiworld replay experiments, qualify arbitrary states,
or establish deterministic physics/real-time planning.

## Explicit model-table supplement

The collector originally saved full dynamic arrays plus model fingerprints/
physical tables, but omitted complete raw model arrays/scalars on disk.
`runs/pen_same_source_models_v1/report.json` explicitly supplements them from
two fresh identically configured actual environments, rather than rewriting
the original run.404 array hashes match EVERY original source fingerprint;
compiled host bytes and actual numerical receipts also match (6/6 checks).
Full404 model arrays and195 scalar leaves are saved for each dt. Scalar
leaves are freshly captured reconstruction metadata, not retroactive original
scalar observations. These two additional native episodes add124 ticks.
Total native work in this phase:720 probe+124 original+124 supplement=968 ticks.

## Repeated response is measurable, but not one smooth derivative

`runs/pen_same_source_AD_v1/report.json` converts all120 verified responses
into the maintained102->90 chart with explicit named joints/free object,
WORLD angular velocities, WORLD-left rotations, mean SOURCE-BODY COM wrench
and Fn6. Full world wrenches are averaged before rotating by SOURCE R.
Frozen physical m/I/gravity exactly match the supplemented model tables.
No normalizer, network weight or optimizer is fitted.

Every16 condition's object/wrench contrast exceeds its observed3-repeat
spread. Maximum spread/contrast norm is0.0696%(object)/0.0616%(wrench), so
the large following amplitude changes cannot be explained by that observed
execution spread alone. These are NOT confidence bounds.

Relative difference of the central responses at0.002 versus0.005rad,
divided by the norm at0.005rad, on common frozen V2 TRAIN scales:

|native dt|source time|object12 change|wrench6 change|
|---|---:|---:|---:|
|5ms|0ms|4.8%|7.1%|
|5ms|40ms|130%|139%|
|5ms|80ms|2269%|2083%|
|5ms|120ms|140%|145%|
|2.5ms|0ms|71.5%|87.8%|
|2.5ms|40ms|16.7%|25.3%|
|2.5ms|80ms|8.0%|7.8%|
|2.5ms|120ms|38.3%|23.9%|

At5ms/source80ms, the weighted object-response vectors even have cosine
-0.442 between amplitudes (wrench-0.276). The relative denominator is small;
both vector norms/cosines and raw targets remain available, not just the large
percentage. Later source states differ ACROSS native dt, so this table alone
does not isolate a timestep effect. No infinitesimal derivative convergence
has been established, and finite responses must not be labeled exact Jacobians.

Descriptive canonical contact analysis
`runs/pen_same_source_contact_patterns_v1/report.json` compares all48 matched
sign/repeat amplitude pairs:9 change object geometry/constraint-pair sets,
18 change positive-normal pair sets,24 change positive pyramid-edge counts.
This avoids row-order artifacts. Contact/active-set changes accompany some
response differences, consistent with nonsmooth/contact-mode effects, but
do NOT prove their sole cause or a universal discontinuity.

## Frozen model evaluation, no success by pooled error

All four previous POINT/RESPONSE x DIRECT/IMPULSE networks stay frozen.
On the same new direction/sources, common V2 TRAIN output scales, pooled
network AD errors against the16 finite native contrasts are:

|Model|object12|wrench6|
|---|---:|---:|
|POINT DIRECT|0.608|0.649|
|POINT IMPULSE|0.589|0.662|
|RESPONSE DIRECT|0.692|0.714|
|RESPONSE IMPULSE|0.727|0.749|

Response supervision is worse on this aggregate, though better at the initial
source. Pooling is dominated by the high-response40ms source. At120ms object
errors range4.33–21.05, wrench4.74–22.04. No subset or model is selected as a
winner; all per-condition results remain. This cohort and scaling differ from
the previous11 CHECK-pair experiment and cannot be read as a longitudinal
0.9->0.6 improvement. Since finite slopes themselves depend on amplitude,
these are response diagnostics, NOT proof of a true local physical derivative,
policy readiness, fresh-task generalization or measured ego dynamics.

The inspected scientific figure is`runs/pen_same_source_AD_v1/summary.png`
with pinned inputs/source in adjacent JSON.190 tests pass, including internal
alias isolation/rejection, force failures despite identical states, correct
object-first/world-omega fidelity and source-frame wrench/Fn6 target decoding.
The suite intentionally isolates native tests on CPU; separately recorded
actual CUDA probes establish RTX execution. Small summaries/reviews and the
artifact SHA ledger are in`provenance/pen_same_source_response_results`;
large states/checkpoints remain ignored. No archive/old environment, driver/
CUDA or installed simulator was changed.

## Next action and full goal

The native fork now has direct20ms fidelity evidence; a smooth learned
gradient does not. Next test bounded finite-action native planning, with state
tracking versus state+contact-support objectives on the SAME candidate budget
per decision and original no-aid clock/termination gates. Audit every selected
prediction against actual execution at newly visited states. This is an
additional controller probe, not the final learned-policy A1/A2 or a surrogate
acceleration claim. See`research-plan-v0.75.md`.

Full scope still includes successful/failed controlled cube/pen learning and
save-resume/unified evaluation, actual support-objective policy comparisons,
input/frame/asset/dynamics contracts, multi-trajectory total cost, true ego
reconstruction/assets/validation and hardware observation/gain/latency before
deployment. Privileged engine state and nominal simulator material parameters
do not supply missing ego/hardware observations or calibrated real dynamics.
No Actor update or stable task result occurred. Goal ACTIVE, nothing waiting.

# Fresh action branches: better point predictions, inadequate response gradients

Previous goal turn was PROGRESS: V1 reached2000 with44/44 numerical health,
and54 physical probes disproved gradient readiness. This turn executes the
predeclared v0.72 source-branch plan, rather than adding epochs to157 rows.
The full Goal remains ACTIVE and incomplete. No Actor updates or new policy
video, successful retargeting or real ego/generalization claim is made.

## Actual source branches and preserved roles

Collection code/config were committed e9ccf21 BEFORE running
`tools/collect_pen_branches.py`. The exact cohort is256 actual original-clock
episodes: TRAIN direction seed7501 with12 independent unit-L2 Gaussian
directions; CHECK seed7511 with4; source control anchors0/2/4/6 corresponding
0/40/80/120ms; both signs and both5/2.5ms native dt. Directions alternate
PD-target amplitudes0.005/0.02rad. The raw branch offset is4*epsilon*direction,
accounting for original0.5 scaling and0.5 EMA. Original2*delta raw commands
apply everywhere else. Actual executed controls, not requested offsets, are
the model inputs. No model selects episodes/directions/amplitudes.

All trajectories retain unchanged native terminal gates, selected initial pose,
environment seed42, single source clip, strict declared numerical profile,
fixed wrist and no post-start resets/object force aid. All256 physical tasks
remain unqualified/nonexpert. Collection integrates15,760 actual physics
ticks. The sum of recorded per-case wall intervals is147.51s EXCLUDING env
construction; this is not a whole-pipeline collection-time or speedup metric.

`runs/pen_branch_EFC_v1/report.json`:1792/1792 EFC/contact-force/torque/impulse
checks pass. Exactly ONE predeclared20ms branch interval per episode enters
the new dataset; shared prefixes and post-branch rows are NOT extra fitting
examples. All256 anchors were reached; all256 selected rows pass the original
per-tick finite/capacity/no-aid/nonhand and force/torque/impulse-quality mask.
Geometry/task failures stay separate and are never relabeled successful.

`runs/pen_branch_prepared_v1/report.json` has7/7 preparation checks:

|Role|Rows|Use|
|---|---:|---|
|Old TRAIN|157|Preserved byte-exact array prefix|
|New TRAIN branches|192|Fresh role-declared augmentation|
|Combined TRAIN|349|Only optimizer/normalizer input|
|New CHECK branches|64|New direction holdout, no fitting|
|Original SEQUENCE_CHECK|80|Original CHECK file pin preserved, no fitting|

The previous54 physical diagnostics, older CHECK records and historical
diagnostics never enter TRAIN. Independent indexed/frame/control/target/role
review`runs/pen_branch_prepared_review_v1/report.json`:3592/3592. It reconstructs
named joint/free-object source states, WORLD angular velocities, held PD
targets, source-BODY COM wrench/Fn6, observed impulse defects, role seeds,
original EMA/clock/terminal behavior and TRAIN-only means without using the
packer for its target computation. TRAIN and new CHECK share source settings
intentionally, but no complete input row overlaps. No unseen-state/clip/ego
generalization follows from these action-direction splits.

## Source variation is not an action derivative

Only37/128 positive/negative pairs have EXACT source55+warm26 equality.
TRAIN:24 at anchor0,1 at anchor2,0 at anchor4,1 at anchor6. CHECK:8/1/1/1.
All actual source differences are recorded and kept as supervised inputs;
they are not repaired, rounded away or interpreted as action effects.

All64 new CHECK rows enter pointwise evaluation. Only11/32 CHECK pairs enter
physical AD comparison: exact source/warm/dt, valid labels, and actual unclipped
positive/negative offsets within2e-7rad. The other21 are excluded only from
derivative interpretation, not discarded from datasets or pointwise metrics.
There is just ONE execution per sign in this cohort, so these new random
directions have no new execution-repeat/amplitude-stability confidence bound.
Earlier3-repeat sine probes do not supply that bound for these conditions.

## Matched V2 fit and exact checkpoint health

Code ed939a5,`configs/pen_branch_fit_v1.toml`, starts NEW matched width512
DIRECT/IMPULSE networks using seed7411 and the same original point-batch256,
learning rate3e-4, AdamW, clipping, auxiliary package and2000-update budget.
New scales come only from349 valid declared TRAIN rows; all scales/physics
stay frozen during fitting. This changes data AND normalization relative to
V1, not just a single causally isolated ingredient. Original V1 checkpoints,
source pins, data and evaluation reports are unchanged.

V2 checkpoints at100 AND2000 each pass38/38 numerical health checks: exact
model/Adam/sampler/RNG/prediction restoration, correct Adam cursor, frozen
buffers, finite parameters/optimizer/gradients/logs/nonnegative Fn6, changed
parameters and two restored clones' next-step loss/batch/gradient/model/
optimizer/sampler parity per arm. Eight total clone-only diagnostic steps
never advance the live saved2000 cursor. Actor updates ZERO. The resumed
Goal authorized continuation beyond the first healthy boundary; no new pause
was requested. The final process exited0 and nothing is still training.

`runs/pen_branch_fit_v1/pen_branch_checkpoint_2000.pt`:
20,189,799 bytes,SHA256
`ce35b698321da3fd4bfdef6176f4be7c0b34bdeb3504de5999a929bdc0b4a530`.
Full2000 paired fit INCLUDING metric calls and checkpoint QA takes14.5612s.
Median CUDA-event paired update6.6733ms.56 device-wide telemetry windows
report1–23% GPU use. These small fixed-data NN timings do not include new
collection/packing or establish sustained GPU utilization/end-to-end speedup.

## Common-unit comparison avoids a normalizer artifact

`runs/pen_branch_evaluation_v1/report.json` evaluates V1 and V2 on identical
rows using the SAME frozen ORIGINAL V1 TRAIN y-scales for all normalized
errors. This differs from the V2 fit report's own-normalizer metric curves.
The following differences are not caused just by changing the error denominator.

|Model|New CHECK object12 MSE|New CHECK wrench6 MSE|Original sequence object12 MSE|Original sequence wrench6 MSE|
|---|---:|---:|---:|---:|
|V1 DIRECT|0.08463|0.05957|0.37891|0.23524|
|V2 DIRECT|0.04213|0.02181|0.67023|0.38692|
|V1 IMPULSE|0.07860|0.05444|0.37932|0.20053|
|V2 IMPULSE|0.02866|0.01550|0.54755|0.30775|

IMPULSE improves NEW branch object/wrench point error about64%/72% compared
with its old counterpart, while ORIGINAL sequence point errors get worse.
This is a coverage/weighting tradeoff, not a qualified general-purpose plant
model or a policy support-objective success.

Recorded-control recursive rollouts use the same80/72/48/8 overlapping
CHECK windows at20/40/100/200ms as before, without teacher-forced state/warm,
clipping, native integration or resets. For IMPULSE, mean COM error at100ms
worsens6.88->7.99mm and full orientation31.85->35.64deg. At200ms it improves
18.01->13.61mm and125.71->70.43deg, but still fails accurate tracking. Full
orientation includes axial roll; it is not the native shaft-direction gate.
Horizons use different subsets, and only4 episodes supply the8 H10 windows.

## Better endpoints still do not establish reliable physical gradients

Network AD at each eligible common source/midpoint PD target is compared to
the actual native central finite-amplitude response, on the same original
TRAIN scales. Per-pair relative errors can be huge for small native responses;
all values remain in the report. `runs/pen_branch_response_audit_v1/report.json`
adds pooled errors (sqrt of total squared error / total squared native response)
and separately checks predicted ENDPOINT SECANTS versus native contrasts.
Neither pooling nor excluding confounded sources is a statistical qualification.

|Model|Pooled object12 AD error|Pooled wrench6 AD error|Pooled object12 secant error|Pooled wrench6 secant error|
|---|---:|---:|---:|---:|
|V1 DIRECT|1.114|1.154|1.111|1.148|
|V2 DIRECT|1.049|1.093|1.065|1.101|
|V1 IMPULSE|1.072|1.043|1.070|1.042|
|V2 IMPULSE|0.912|0.922|0.915|0.923|

The zero-response comparator has relative error1. V2 IMPULSE is somewhat
better on this pooled local measure, but still leaves large response error.
At the8 initial-source pairs, its pooled AD object/wrench errors0.908/0.890
are mixed versus V1's0.986/0.824. All-source mean per-pair relative errors
are7.41/10.37 for V2 IMPULSE because small-response cases remain poor.
AD and each model's endpoint-secant errors are similar, so this result does
not support blaming only AD implementation or finite-amplitude versus local
derivative mismatch. Merely fitting more endpoints has not identified reliable
action effects. No learned-model Actor update follows from these numbers.

## Validation and next action

Full suite181 tests pass (`runs/pen_branch_tests181.log`). New tests cover
role-disjoint balanced profiles, strict dataset/stage contracts, and exclusion
of source-confounded/clipped/invalid/dt-mismatched derivative pairs. Actual
independent saved-data checks supply the larger state/frame/force evidence.
The suite deliberately isolates native tests on CPU; its CUDA-unavailable
message does not describe the actual separately recorded RTX5090 runs.

The inspected figure is`runs/pen_branch_evaluation_v1/summary.png`, with
input/source SHA pins in adjacent JSON. Compact result summaries, reviews,
tests and original artifact pins are committed under
`provenance/pen_branch_coverage_results`; raw traces/arrays/checkpoints remain
ignored. No archive/old environment, system driver/CUDA or installed simulator
was modified. Shared source/config continues in this one repository.

Next predeclared ablation: keep these pointwise TRAIN rows/scales fixed and
add finite-action-contrast supervision from ELIGIBLE TRAIN pairs only, with a
matched point-only comparison. Do not use CHECK or old diagnostic pairs for
loss/scales/selection. Measure extra pair forwards and total compute explicitly;
do not claim equal FLOPs by equal optimizer counts. Late-anchor source mismatch
must be addressed by collecting repeated responses from one captured actual
integration state, with strict no-aid state/solver/fidelity checks, before
calling those labels physical action derivatives. See`research-plan-v0.73.md`.

The TRAIN-only contrast preparation is already materialized in
`runs/pen_TRAIN_contrasts_v1`:26/96 TRAIN pairs are eligible (all26 exact-source
pairs also pass actual unclipped-PD/validity gates). Preparation5/5 and
independent saved-array readback6/6 pass, including exact case-ID/source/target
binding and finite-contrast reconstruction. Only NEW_TRAIN arrays are read;
CHECK and old physical diagnostic arrays do not enter it. No new normalizer
fit, optimizer update or simulator step occurs in this preparation. Its
contrast-supervised training ablation has NOT started yet.

The original scope remains: fixed-wrist input/frame/asset/dynamics/controller
contracts; no-aid cube/pen control learning and unified evaluation; actual
support-objective A1/A2 and physical-gradient validation; multi-trajectory
sharing/total cost; real ego reconstruction/assets/validation and hardware
observation/gain/latency contracts before deployment. This local component
work does not replace any of those requirements. Goal remains ACTIVE.

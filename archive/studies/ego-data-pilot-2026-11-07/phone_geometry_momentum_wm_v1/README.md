# Geometry, momentum and finite-response phone WM

Source entry: `train_geometry_momentum_wm.py --protocol PATH`. The provided
`protocol_cpu_false_v1.json` passes metadata/hash preflight without importing
Torch, opening NPZ arrays, constructing a NN, or calling native. Actual
training requires Root's separately frozen CPU authorization and `--run`.
No synthetic optimizer or model test has been run in this preparation.

## Exact interfaces

* Input366 = raw80 (`COM/world state58`, absolute u20, native h, integer n)
  +22×12 current relative geom poses (COM-origin BODY translation3, rotation9)
  +22 current finite-direction convex descriptors in metres.
* Physical output178 = joint delta40 (q20/qd20), COM-ballistic residual3,
  world-left dR3, 22×6 source-BODY/moving-COM impulses. Separate22 meanFn raw
  values and22 coverage logits make the complete NN output222.
* `make_model(torch)` creates a fresh366→128SiLU→128SiLU→222 MLP: six Parameter
  tensors,92126 scalar parameters. No old weights, pen chart or normalizer.
* `model.forward(normalized_features366)` is the cached training path.
  `model.forward_from_raw(x,u,h,n,norms,geometry)` calls the live differentiable
  current-state providers; no target/contact-force values enter that function.
* `heads_from_physical(v178)` supplies the frozen structured codec. The codec
  sums the22 impulses, rotates by source R, closes linear/angular momentum for
  v/omega, and uses independent p/R heads. It does not derive pose from impulse.

Training caches deterministic current geometry once, deduplicating identical
state bytes **only for computation**; all2118 rows retain their loss weight.
Geometry cache evaluation is chunked at4 states in F64, then input normalization
and the NN use F32. The live feature API remains differentiable. Finite-direction
max/min is piecewise differentiable, not an exact SDF, unique contact normal,
full-H path certificate or guarantee of a physically correct input gradient.
Unknown descriptor values cause a failure, not fabricated zero features.

## Frozen fitting recipe

Seed1731, CPU4, full batch, fresh128×128SiLU, AdamW lr.001/wd0/betas.9,.999/
eps1e−8/foreachFalse, finite-gradient check and clipnorm1. The declared budget
is5000 updates with checkpoints250/1000/5000. NN arithmetic is F32; physical
heads, the structured decoder and its differentiable inertia solve are F64.

Absolute supervision is exactly684 original complete TRAIN rows plus1434
available FIT sides =2118. The36 replay baselines map back to their original
TRAIN rows and are **not** added again. Each FIT side is paired with its own
same-state baseline for1434 empirical finite responses. Actual selected F64
du (at least1e−6 rad), rather than nominal epsilon or rounded F32 subtraction,
is the response denominator. No epsilon rescue or best-axis selection occurs.

Input/absolute-target/velocity scales use only these2118 FIT rows; response
scales use only1434 FIT target secants. Means and population stds are saved.
Absolute and response errors each average six separately normalized blocks:
q20, qd20, COM3, dR3, force-impulse66 and moment-impulse66. Fixed physical std
floors are .001rad, .01rad/s, .0001m, .001rad, .0001Ns and .00001Nms.
CF scales carry corresponding units per radian, with1rad action reference.
Decoded velocity errors use separate v/omega blocks and floors.01m/s/.1rad/s.

Total loss is the sum of five terms, each coefficient1 fixed before fitting:
absolute head loss, decoded velocity loss, FIT head-secant response loss,
meanFn log1p loss, coverage BCE. `meanFn=softplus(raw)*1N`; log1p uses1N as its
physical reference and has **no tiny per-feature std divisor**. Coverage is
the measured fraction of solverPRE ticks whose summed positive Fn>.1N.
The last layer is zero-weight initialized; physical normalized output starts
at TRAIN mean, while force/coverage biases use TRAIN means with declared
.001N and.0001 probability initialization floors.

Targets remain the observed native simulation outputs, including their nonzero
momentum discretization defects. `observed_label_decode_defect58_F64` is saved
as evidence; it is not subtracted, projected to zero or used as an input.
The22 impulse targets include all canonical phone-boundary pairs, including
table/nonpad contributions. They do not imply support by the designated two
pads, soft-contact realizability, feasible holding or human-force ground truth.

## Data exposure and phases

Only original TRAIN and causal FIT NPZs are opened for fitting. Original36
partial rows and6 FIT-unavailable records stay upstream and are listed in
`data_selection.json`; no complete-H targets are invented for them. The
label-only FIT archive avoids opening the combined/CHECK targets. CHECK1472
new action responses and TEST arrays are hash-bound only. Prior engineering
target exposure is not erased; original state families remain seen states.
The12 planned genuinely fresh families are not acquired or used here.

Original114 complete VAL rows are read only for fixed checkpoint state58
monitoring, never normalization, fitting, coefficient selection or choosing
the checkpoint. No VAL per-pair force labels are opened. The data are POST20ms
state targets and all-H solverPRE contact summaries, with64/128 native ticks.
Source-body impulse axes stay fixed at interval start while moments are about
each tick's moving COM. Instantaneous rotating-body integrals are not substituted.
Original pads15/28 and all22 canonical pair slots retain their identity; a
future tripod task requires an explicit new task/data contract.

## First genuine checkpoint and resume

At250 the process atomically saves a new component checkpoint, verifies raw
weights/normalizers/context/inputs/targets, finite pre/post-clip gradients,
serialized Adam6 states at step250, exact save/load predictions and raw Python/
NumPy/Torch RNG values. This uses one own CPU readback, a constructor in a
scoped Torch RNG fork, and two B8 forwards; both predictions and RNG bytes are
saved. The fixed VAL B114 monitor adds one forward. There is no training replay.

It then writes `health_0250.json`, an ETA and
`STOPPED_AT_FIRST_HEALTHY_250_RESUMABLE_TARGET5000_NOT_COMPLETE`, and **exits**.
Root can complete the handoff and pause the Goal at this genuine milestone.
250 is not presented as a completed5000 run. A subsequent explicit resume
protocol must pin the healthy own checkpoint, health receipt, identical recipe,
source/data pins and recomputed FIT normalizers/digest. Resume restores the
raw optimizer/RNG states; no historical first-checkpoint pause is repeated.

The first process has250 training forwards/backwards/steps, two readback B8
forwards, one VAL B114 forward, two total NN constructors and one checkpoint
load. There are0 native model/FK/forward/integration/Env/GPU calls. Full lineage
training remains5000 forwards/backwards/steps; checkpoint diagnostics are
separate, so this is not a same-wall-compute claim.

ETA uses the recent100 measured update durations times remaining updates.
One-time geometry/cache/setup, log I/O and checkpoint/readback/VAL I/O are
reported separately. Resume currently rebuilds the cache; estimated resume
setup and two observed-rate end-to-end scenarios are explicitly heuristic,
not confidence bounds or precise wall-time promises. Checkpoints contain bare
state dictionaries, raw Adam/RNG, normalizers, source/data pins and exact
durable JSONL-prefix bytes/SHA/range; no wrapped-module or hook pickle.

This is one new supervised baseline combining geometry, structured momentum
and empirical action response fitting. It does not isolate those contributions,
prove zero response outside the measured intervals, implement a full Markov
model, integrate an Actor/reward/CHORD task, or establish policy/ego benefit.
Matched ablations, native action-gradient checks and task evaluation remain
necessary later; manual2s holding success is not a prerequisite imposed by
this training source.

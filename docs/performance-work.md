# GPU performance after consolidation

The unified repository is the development home. Historical study sources,
datasets, frozen checkpoints and native physics remain unchanged.

## Fixed-parent comparison, 2026-11-07

RTX 5090, same paired100 weights, five Adam states, RNG, sampled Actor rows,
WM source mixture, batch8/H2 and six private updates (one warm-up, five timed).
Each variant performs 30 private Adam steps. The numerical fixtures retain
all named pre/post-clip gradients and network/target/normalizer state after
the sixth update. No resumable policy checkpoint or new learning experiment
was created; these parameter updates are discarded after benchmarking.

| Variant | Mean seconds/paired update | FK/descriptor calls per update | Peak allocated GiB | GPU median |
|---|---:|---:|---:|---:|
| Initial consolidation `449dfdb`, micro2 | 6.4734 | 64 / 40 | 1.893 | 46% |
| Geometry reuse + resident constants + deferred metrics, micro2 | 3.0642 | 24 / 24 | 1.298 | 42% |
| Same shared implementation, micro8 | 0.8092 | 6 / 6 | 4.866 | 52% |

The mean-time ratios are 2.11x and 8.00x for this short workload. GPU samples
are restricted to measured update intervals (57/27/7 samples respectively).
The micro8 utilization estimate is coarse because only seven samples fall
inside its short timed interval. Setup, reward precomputation, fixture CPU
copies and profiler export/aggregation are excluded from these timings.
Both benchmark variants keep the same canonical real-row bank resident, so
the table does not credit the additional runner upload fix described below.

The preliminary v1 profiler's whole-process GPU median included its lengthy
CPU trace aggregation. Do not use that value as steady training utilization;
the table uses corrected v2 interval sampling. Profiler timings are excluded.

## Batched hinge rotations

The next source variant batches the twenty hinge rotvecs through the same
public guarded Rodrigues function. Each FK calls that function once rather
than twenty times, retaining finite-vector and squared-magnitude overflow
checks for every hinge. The body-tree composition formulas are unchanged.

| Final variant | Mean seconds/paired update | Peak allocated GiB | GPU median / samples |
|---|---:|---:|---:|
| Batched hinges, micro2, quiet CPU | 2.4110 | 1.298 | 45% / 22 |
| Batched hinges, micro8, quiet CPU | 0.6443 | 4.866 | 57% / 5 |

These are 2.68x and 10.05x faster than the original mean for this fixed short
workload. FK/descriptor counts remain 24/24 and 6/6. The five-sample micro8
utilization estimate is coarse; the measured wall time is the main evidence.
An earlier batched micro2 measurement overlapped CPU validation work and
averaged 2.8225s; it is retained, but the quiet repeat is used in this table.
The quiet repeat was scheduled because of that known interference.

The final micro2 fixture remains byte-identical to the baseline; final micro8
is byte-identical to the preceding micro8 fixture. Both pass the unchanged
fixed comparison envelope. CPU full legacy/runtime parity is 6/6 with maximum
error zero. Dedicated F32/F64 scalar, batch2 and extra-leading-dimension tests
also cover input AD and invalid width/dtype/nonfinite/SO3/overflow rejection.
An A1 tracking task can still omit its own geometry provider when an explicit
same-state context is passed; this compatibility case has its own CPU test.

## Numerical scope

The micro2 comparison is raw-exact for all 30 recorded loss scalars and all
170 fixture arrays; the entire compressed fixture SHA is identical. Micro8
changes floating accumulation order. It passed the tolerance preset fixed
before reading candidate values: gradient maximum absolute difference
9.778887e-9, maximum per-tensor relative L2 5.937862e-7; Actor parameter maximum
absolute difference 7.450581e-9. WM, Critic, target and normalizer arrays remain
raw-exact. See the saved comparison summaries and named details.

This is bounded numerical evidence. The fixture does not export Adam moment
arrays, final RNG state or sampled WM row IDs, so the saved-only comparator
cannot independently compare those leaves. The tool restores the same parent
and RNG and uses the same sampler code; identical declared update budgets are
checked separately. These checks do not establish long-training identity,
native action gradients, actual contact estimation or improved policy holding.

## Implementation

- An explicit short-lived geometry context belongs to one exact state Tensor,
  provider and autograd lifecycle. Observation, WM features, reward and next
  observation share its FK/descriptor graph. Every new predicted state gets
  a new context. Equal-valued clones, ordinary in-place changes and grad-mode
  changes are rejected. Inference tensors must remain immutable while used.
- State-independent mesh/direction/task constants and phone support terms use
  device/dtype buffers. Buffers are ordinary no-grad tensors even when first
  requested inside inference mode, so subsequent input AD stays valid.
- All original 1406 directions, 64-wide chunks, `cat` then global `max`, known
  masks, NN FP32, physical/geometry FP64, AMP/TF32 disabled and safety checks
  remain. No imagined-state geometry is detached or reused across updates.
- Each module's finite gradient/parameter reductions produce one host decision;
  clip and pre-step/post-step checks remain at the original safety boundaries.
  Detached scalar logs transfer together, then reconstruct the original
  sequential Python-F64 mean. Full gradient snapshots remain captured.
- The driver now uploads canonical real rows once, preserving NaNs, masks,
  CPU RNG draws and eligible row order. The original WM replay cache was
  already resident; classifying all fixed data as CPU-resident was inaccurate.

## Reproduction and remaining work

Use `tools/benchmark_gpu_v2.py` for the corrected benchmark and
`tools/compare_gpu_benchmarks.py` for saved-only comparisons. Pass
`--deferred-metrics` for the optimized update core, and explicitly declare
`--allow-microbatch-change` only for the planned micro2-to-micro8 comparison.
Keep baseline/candidate source versions fixed; use a fresh output directory.
The tools never authorize the production trainer. `configs/phone_offline.toml`
retains micro2; `configs/phone_offline_gpu.toml` is an explicit CUDA/micro8
example, also disabled by default and retaining the first-healthy-checkpoint
stop. A new-source run is a separate experiment, not a resume of the old
frozen source-bound protocol. Checkpoints from another implementation continue
to fail the own-schema source-binding check.

Direction-chunk changes, remaining guard aggregation, compilation or precision changes
require their own comparisons. Continue to report wall time, memory and
numerical differences alongside utilization; filling VRAM is not a target.

Small reports are retained in `provenance/performance/phase1` and `phase2`.
Raw fixtures, samples and traces stay under ignored `runs/performance`.
The batched-FK test reads its scalar-loop reference from Git commit `1e55b54`;
it skips with an explicit reason if a shallow checkout lacks that reference.

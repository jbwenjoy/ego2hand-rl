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
The tools never authorize the production trainer.

Next separate variant: batch the twenty guarded hinge Rodrigues evaluations.
Direction-chunk changes, guard aggregation, compilation or precision changes
require their own comparisons. Continue to report wall time, memory and
numerical differences alongside utilization; filling VRAM is not a target.

# A1 single-seed pilot: training record

The bounded seed0 training run completed with exit code0 in1,117.95 seconds
(18.63 minutes). This is a single-clip, single-seed pilot; training completion
does not by itself establish physical task acceptance, CHORD benefit, an ego
pipeline, or a speedup against another method.

Input: original `01_single_axis/clip_007`, frames `[200,351)`, 151 samples at50Hz
(3 seconds). All five NPZ fields match the root's selected catalog exactly.
The frozen evaluation reference and its preregistered criteria were copied before
training; the final CPU audit confirmed their bytes had not changed.

Configuration:128 environments,2,000 iterations,64 warmup steps,200 WM-only
iterations, replay capacity1,000 vector steps, WM512×3/batch256×2/window8,
policy batch128/horizon8, AMP and compilation enabled, checkpoints every100.
The fixed play-scene used clean observations, no external disturbance/action
mask/RSI/first-frame hold, with reset command phase explicitly aligned to zero.
Native self-collision-head supervision weight0.1 was retained. The auxiliary
failure head was fitted but never used by the policy. No CHORD term was present.

| Measured quantity | Result |
|---|---:|
| Total wall time including source/reference preparation | 1,117.95 s |
| Learn loop | 1,109.68 s |
| Simulator collection including warmup | 362.05 s |
| WM fitting | 506.04 s |
| WM holdout calls | 30.79 s |
| Policy/critic calls | 209.73 s |
| Checkpoint writes | 0.42 s |
| Real simulator transitions | 648,192 |
| Actual WM / actor / critic optimizer steps | 4,000 / 1,800 / 7,200 |
| Auxiliary failure-head steps, policy disabled | 4,000 |
| Steady seconds/iteration, iterations300–1990 | 0.4972 |

The first WM phase took31.71 seconds and the first policy phase73.17 seconds,
including initial compiled calls/autotuning. These are phase measurements, not
isolated compiler measurements. Compilation captured24 graphs, with an empty
graph-break counter. There was no eager fallback or budget extension.

Inductor used an explicit project cache that did not exist at launch. Triton
already had6,449 project-cache files (125,819,952 bytes); Warp kernels were warm.
Before/after cache inventories are in `cache_context_before.json` and
`summary.json`. The recorded0.612GiB peak covers the PyTorch allocator only;
it excludes Warp, graphics and driver allocations.

All200 logged metric rows and all final checkpoint module tensors were finite.
WM, actor and critic parameters changed; the native self-collision output weights
changed by L2=7.971. The fixed physics/context snapshot had zero variation across
worlds. Input configuration bytes, script/Dex source hashes, and the frozen
evaluation bundle were unchanged at the post-training audit. All20 interval
checkpoints and the final checkpoint are retained.

Final checkpoint: `compiled/train/model_final.pt`, SHA256
`6cbbb910cc62972ab16005da8fb8e85999805a6d1b5d731d4a307f127a81a8a7`.
Primary records: `compiled/result.json`, `compiled/metrics.jsonl`,
`final_cpu_audit.json`, `source/inventory.json`, and `reference/provenance.json`.

Subsequent independent evaluation identified an unbounded absolute command on
right_finger5_joint1, despite good object tracking. This run was trained with the
original unbounded controller. The subsequent explicit control-limit patch must
be evaluated as a separate postprocessor variant; it does not retrospectively
make this original run accepted. Its exact source is preserved in
`source/dex_source.zip` and `source/train_controlled_pen.py`. The final source
immutability audit above occurred before that intentional working-tree patch.

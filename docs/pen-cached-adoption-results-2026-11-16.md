# Cached training adoption and trained64 W32 collector — 2026-11-16

The optional fixed-frame cache now has a real configuration entry point for
new training. Old objective configs still select LIVE. A new CACHED32 two-arm,
two-iteration integration trial passed qualification75/75 and training5/5.
All four optimized checkpoints were independently loaded into fresh current
GPU32 runtimes:18/18 checks,exact full native arrays/scalars,NN/Adam/observations/
memory readback and original training4 normalizer counts. This is integration
evidence,not a new task-success or support-efficacy result. Main64 policies and
the existing comparison video remain unchanged.

## Frozen weights with explicit normalizer history

Maintained `learning/frozen_pen_weights.py` initializes only the original
SUPPORT64 actor/critic and embedded normalizers into fresh current W32 physics.
Original checkpoint SHA256 is
`c9f39696b794a311725bd54cd427b522a3ad9e2a3d334da8e39f90aab6948f60`.
All404 GPU Model arrays matched keys/shapes/dtypes/bytes;all195 Model scalar
leaves matched. Host model,reference,full objective/action/observation/joint/
network/PPO metadata and exact historical snapshots were also checked.
No W1-to-W32 broadcasting or selected-physics-field shortcut is used.

Actual native4/Python/RNG closure and fresh observations were identical before
and after NN initialization. No historical native,Adam,RNG,counters,rollout or
observation state was loaded. Current Adam and all work counters started empty/
zero. Modules matched the source exactly.

The old normalizers each contain129312 data points. Importing those statistics
must not fabricate new calibration/training transitions or bypass count guards.
New explicit `pen_geometry_support_PPO_frozen_weights_training:5` records a
pinned normalizer seed and its count;expected count is original seed count plus
new calibration/training data actually incorporated. Existing training2/3/4
retain the original zero-offset formula. Restoring training5 requires the same
explicit initialized learner contract. It is not an old native-state migration.

The initialized current-source training5 state was saved before warmup. Two
64-control/32-update warm rollouts compiled/warmed the ordinary code,then the
saved current-source state was loaded. Independent decoded comparison proves
the comparison fixture equals the pre-warm state,including NN,Adam,native,RNG,
normalizers and counters. All warm work is charged separately from clean phase
timing and is not retained in the main64 weights used at each case start.

## Actual trained64-weight collector timing

Use preregistered LIVE_A,CACHED_A,CACHED_B,LIVE_B,LIVE_C,CACHED_C order,then
separate TORCH_LIVE/TORCH_CACHED CPU traces. Each case restores the same fresh
W32 source with exact trained64 NN/norms,then64x32 controls and32 diagnostic PPO
updates using UPSTREAM control. No monitor thread runs during clean timings.
Independent full source-fork review41/41:LIVE changes0 fields,CACHED exactly
three declared mode fields;all original collector and readback checks pass.

| Clean collection seconds | A | B | C | Median |
|---|---:|---:|---:|---:|
| LIVE |1.184187|1.186746|1.197069|1.186746|
| CACHED |1.087764|1.094826|1.088949|1.088949|

Observed collection wall reduction8.2407%,throughput ratio1.08981.
Collection+consistency+update medians1.342595->1.243944s (~7.35% less wall).
Reset world counts were LIVE2/3/4 and CACHED3/2/2. Full long trajectories are
not declared identical;contact/reset-sensitive numeric divergence remains a
limitation. This trained-weight fixture has far fewer resets than the earlier
early-training fixture. Do not treat their absolute times as a matched causal
comparison or pool their repeats.

Current-source actualGPU feature qualification remains388/388:all369 phases,
clamps,finite/NaN patterns,dynamic inputs and pre-write saved-mode checks.
Dynamic feature JVP9/9 is exact in the tested CUDA32 layouts;it is not a
physical contact/world-model gradient result. Full unit suite379/379,including
whole-Model mismatch rejection and explicit normalizer-seed guard tests.

## Artifacts and actual work

- `runs/pen_fixed_frame_qualification_v4`:staticGPU qualification,0 native/
  optimization steps.
- `runs/pen_cached_support_qualification_v1`:4096 native,0 optimizer;reuses
  original offline score receipt only after live critical source and all
  listed input/report pins verify.
- `runs/pen_cached_support_training_v1`:65536 native,128 new diagnostic PPO
  updates;two arms,two iterations each,all checkpoints retained.
- `runs/pen_trained64_frame_comparison_v1`:163840 native,320 diagnostic PPO
  updates including discarded warmup;all source forks,rollouts,modules,traces.
- `runs/pen_cached_support_checkpoint_review_v1`:18/18,0 native/optimizer.
- `runs/pen_fixed_frame_gradients_v3`:9/9,0 native/optimizer.
- `runs/pen_trained64_frame_review_v1`:41/41,0 native/optimizer.

Total233472 native integrations/448 diagnostic PPO updates. Source snapshots,
source resolutions,test log and compact evidence ledger are retained under
`provenance/pen_cached_adoption_results`. Setup/source/readback/profiler costs
remain outside the clean phase ratios. No driver/CUDA/package edits.

## Continue the scientific experiment

Cache adoption is qualified for explicit current32 callers. W1 use still needs
separate GPU1 qualification;existing W1 controlled-response work can use LIVE.
Main-policy task success,independent seeds/original trajectories,total training
and adaptation cost,physical WM/contact gradients and ego collision/dynamics
assets remain unqualified. Earlier four-arm response models still fail CHECK
gradients and cannot optimize an imagined Actor.

Next register and collect full20D local PD excitation,with an independent
development-validation bank reserved before model fitting. Preserve strict
restore/geometry/force/torque/impulse/no-aid gates and every failure. Design a
matched rank4 subset within the same source bank and fixed point budgets so
new sources/more samples are not silently attributed to rank alone. Then
isolate TRAIN-only derivative-target scaling/weight and forward-anchor changes
with common initialization/data/point draws/update/cost budgets. Original324
CHECK remains developmental NEVER_FIT;whole-original-reference tests reserved.

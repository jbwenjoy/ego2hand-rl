# Fixed reference/frame caching,registered before GPU qualification/timing

Previous measured targets:64 live world-reference recomputations despite an
existing world_pose table;188 policy/126 critic observations repeatedly form
fixed wrist rotations. No claim that the prior empty-query prototype is faster.

Use one shared feature implementation with explicit LIVE/CACHED modes. Both
exist in the same current source version. New cache schema stores policy_WT and
critic_WT separately,preserving original normalization orders and strides. Live
object/joints/velocities/EMA/raw/phase stay live. CACHED additionally uses existing
world_pose for reward reference. No physics/solver/collision/reset/interval edits.
Default stays LIVE until qualified;UPSTREAM control driver remains unchanged.
Mode switches are cold-only and pointer/mode guarded;cache content/digests and
canonical immutable owners include both matrices. No zero-quaternion repair.

CPU tests cover phases/clamps/origins,dynamic suffixes,nonunit normalization,
invalid quaternion NaNs,matrix corruption and mode switching. Actual GPU32-world
qualification checks all369 phases,hold/end clamps,mixed phases and dynamic
feature probes. Compare policy161/critic188 at2e-7 absolute,reference7 at1e-7,
rtol0 and identical NaN patterns;record exact equality separately. These existing
feature tolerances are not new physical/geometry tolerances. Restore every
temporarily varied qpos/qvel/action/phase afterward,verify unchanged source
bytes. No native integration/neural update during feature qualification.
Test matrix corruption,replaced storage,unapproved mode mutation and attempts
to switch inside collection. Feature probes are not physical TRAIN data.

For timing build a NEW current-code32-world SupportPenPPO fixture (same existing
calibration8,two64-control/32-update warmups). Both modes share current source,
full Model/Data/NN/Adam/RNG/memory/reference and guards. An explicit performance
fork may change only three fields:top metadata.fixed_frame_features,
native.metadata.fixed_frame_features,and the unique Python feature-mode boolean.
Do not rebind sources,repair saved observations or alter physical/module/RNG
inputs. Ordinary load must still pass exact source/layout/observation/native
readback. Forks and patches are saved;historical native checkpoints are not used.

Use three clean cases/mode in LIVE_A,CACHED_A,CACHED_B,LIVE_B,LIVE_C,CACHED_C order,
then separate TORCH_LIVE/TORCH_CACHED CPU profiles. Each64 controls,original
consistency,32 updates. Restore same fixture first;no monitor threads. Expected
165888 native world integrations/320 diagnostic PPO optimizer steps. Record
all durations,operator counts,reset workloads/variability,rollouts/modules and
failures. No profiler-based speedup claim or cherry-picked case. Timing success
alone does not prove long-horizon physics equivalence or full task success.

Separate fixed-input feature/reward mechanism parity and immediate source
restore from chaotic long-rollout comparison. Retain observed numerical/repeat
envelopes,do not relax geometry/contact gates to make equality pass. Cache
storage/mode/metadata corruption must reject. Initial main policy/video remains
unchanged;any adoption requires bounded evidence and an explicit declared caller.
No driver/CUDA/installed-package edits. Further reset coalescing is separate.

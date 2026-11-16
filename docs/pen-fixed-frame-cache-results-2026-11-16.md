# Guarded fixed-frame/reference cache — 2026-11-16

The final same-source32-world comparison reduces median collect64 from1.797987s
to1.639422s:8.819% less collection wall time,1.09672x observed throughput.
Collect+consistency+32-update median1.950731->1.793804s(~8.04% less). All original
collector/work/no-aid/overflow gates pass. Actual GPU feature outputs and dynamic
feature JVPs are exact in tested cases. This is a bounded early-training fixture
result,not universal full-training/ego/task or main64-policy improvement.

## Shared implementation

The owned reference cache already stored world reference poses,but reward
tracking recomputed them every control. LIVE/CACHED now select the same shared
feature implementation. CACHED reuses world_pose and two fixed wrist matrices.
Policy originally explicitly normalizes mount quaternion then calls quat_to_mat
(which normalizes again);critic calls quat_to_mat directly. Both exact chains
are separately cached as policy_WT/critic_WT,with original transpose strides.
Dynamic joint/pose/quaternion/velocities,EMA/raw actions and phases remain live.

Cache schema is explicitly `frozen_native_reference_prefix92_fixed_frames:2`.
Both new tensors enter source snapshots,immutable ownership/digests,metadata/
resident-byte accounting and pointer bindings. Modes enter runtime metadata and
the pointer/mode guard. `set_fixed_frame_features(bool)` is cold-only;direct
mode mutation or changes during collection reject. Public pending/manual-reset,
geometry/support/clock/contact/stream/source/alias guards stay unchanged.

Current-source performance forks change ONLY top metadata.fixed_frame_features,
native metadata.fixed_frame_features and the unique Python feature-mode boolean.
They do not rebind receipts,repair observations or transfer historical native
states. Full ordinary load must still pass exact source/native/observation
readback. Independent saved-tree review checks all tensors/arrays/strides/RNG/
config/NN/Adam and all unchanged fields,not merely a few physical parameters.

Mode metadata is cross-checked against the saved boolean BEFORE writes. This
additional preflight check was found necessary during source review after the
first timing cohort;final source/qualification/timing are rebuilt and kept
separate. No source/mask repair or physical threshold relaxation. Default remains
LIVE;declared current-source callers can explicitly choose the qualified CACHED
path through the cold setter. EMPTY_SIGNALS remains separate and unadopted.

## Numerical/ownership proof and retained probe failure

The first GPU probe incorrectly used augmented assignment to WarpBridge.qvel,
which rejected attribute replacement. Its source,console and failed record are
retained,zero native/optimizer work. Re-run uses explicit in-place addition.
No successful qualification is inferred from that failed run reaching a probe.

GPU qualification2 passes386/386 before the new saved-mode preflight guard.
Final qualification3 passes388/388:all369 phases/all32 worlds,hold/end clamps,
mixed phases,live joint/velocity/object/action/EMA probes,nonunit/zero current
quaternions,cache corruption/replaced storage/illegal mode switching. Policy161,
critic188 and reference7 all compare exactly,maximum absolute errors0. The
declared forward tolerances2e-7/1e-7 were not loosened. Temporary inputs restore.
Final corrupted saved-mode rejection leaves every Model/Data array byte intact.

Dynamic feature JVP checks on actual CUDA32-layout tensors cover phases1/48/120/
368 and two deterministic tangents each across qpos/qvel/raw/EMA. Both source
versions pass9/9;final maximum JVP and forward errors0,exact equality. The mount,
reference and phase are fixed here. This is FEATURE calculus,not contact/WM
gradients or a new qualification for imagined Actor updates. Prior WM failures
remain. No NaN-to-identity repair is introduced.

Full CPU suite374/374 passes after the final core change (prior366 plus8 cache/
feature-JVP/source-mode guards). Mode and matrix corruption tests augment
existing full-state/RNG/alias/collector tests. Static probes execute no native
integration or optimizer updates and do not become physical TRAIN examples.

## Timed current-source comparisons

Construct a new source for each code version:8 controls calibration,two ordinary
64-control/32-update warmups;full checkpoint includes native/task/cache/NN/Adam/
normalization/memory/counters/RNG. Do not load an older-source native fixture
after changing code. Three clean LIVE/CACHED cases alternate position,followed
by separate CPU profiles,with monitoring disabled. Cold hash/readback/fork/save
cost stays outside phase timing. Each case64 controls/32 updates,each cohort
165888 native integrations/320 diagnostic PPO updates including warmup.

The earlier source cohort observed1.770306->1.618279s(8.59% less collect wall).
It is retained separately,not pooled with the final source below:

|Final clean times(s)|A|B|C|Median|
|---|---:|---:|---:|---:|
|LIVE collect64|1.797997|1.797987|1.797879|1.797987|
|CACHED collect64|1.639422|1.645895|1.617526|1.639422|
|LIVE collect+checks+update|1.950731|1.950031|1.950981|1.950731|
|CACHED collect+checks+update|1.793804|1.799182|1.774036|1.793804|

Reset-world counts LIVE198/199/198 versus CACHED191/199/199. Three repeats per
mode do not certify statistical/universal speed or identical long trajectories.
Native32 future replay remains numerically variable. Mechanism parity is proved
at fixed inputs/phase/transforms separately from those long-rollout effects;
all actual termination/geometry/reset gates remain active. No discarded cases.

Separate CPU profiles show mul25770->16360,stack1017->629,vector_norm1608->1092
calls. Nonzero317->313 and bmm882->868 also reflect different reset workloads;
do not attribute every count change solely to caching or sum nested CPU times.
The optimization does not skip native contacts,collisions,solver steps or real
resets;it reuses already-guarded fixed transforms/reference values. This addresses
a measured part of the user's repeated-calculation/launch concern,not all
GPU-utilization or reset-forward overhead.

Final timing contract4/4,feature qualification388/388,feature gradient9/9 and
independent saved fork/timing review37/37 pass. Review verifies three changed
mode fields for CACHED,zero for LIVE;native Model/Data,modules,Adam,RNG,source,
saved observations/memory and other fields are byte/layout unchanged. Immediate
full native array/scalar readback and ordinary observation checks pass every
case. It does not claim old-native migration or full task/physical equivalence.

## Cost and continuation

Two separate timed cohorts total331776 native integrations/640 diagnostic PPO
updates;all static/gradient/source reviews add0 native/optimizer work. Those PPO
updates change only diagnostic actor/critic copies,not the main64 policy or
video. Setup/qualification/fork/trace costs are retained but are not part of the
clean phase speed ratio or a complete multi-reference amortization claim.

Use explicit CACHED current-source32-world callers after qualification;refresh
fresh-training qualification/source contracts rather than rebinding old native
checkpoints. Main-actor weights-only W32 profiling still needs full404 Model/
scalar/observation/action/config checks;do not reuse W1 context by assumption.
GPU1 must be qualified separately before cached W1 adoption.

Return to full20D excitation and isolated TRAIN-only response-loss normalization/
weight/forward-anchor work;the prior models remain unsuitable for Actor updates.
Further reset-forward/observation coalescing needs command/frame/observation/RNG
epoch proof before deleting a refresh. Keep the previous empty-query negative
result,raw GPU trace and all solver/WM/task failures. No driver/CUDA/installed
package edits. Exact receipts and compact ledger are in
`provenance/pen_fixed_frame_results`;full controlled task/ego/hardware requirements
remain incomplete.

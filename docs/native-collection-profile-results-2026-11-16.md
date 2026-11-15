# Native collection profile and frozen reference cache — 2026-11-16

Actual collection profiling identifies repeated feature/reference work and
host synchronization. An opt-in immutable reference-prefix cache produces
exactly the same tested161-dimensional features and reduces observed short
collection time by12.26%. This is a controlled computational result, not a new
policy, full training speedup, deterministic W32 physics or ego/CHORD benefit.

## Protocol and scope

Both studies use the existing native32-world,64-control,8-substep,161/188
interface and the pinned calibrated INITIAL checkpoint. Each repeat restores
all initial native/Python/module/optimizer/RNG arrays with readback audits.
The same fixed .05*sin scripted raw actions are used, so this is a computational
profile, not on-policy learning evidence. Physics/reference/assets/rewards and
termination/reset guards are unchanged. State load, CPU record copies, cache
construction and cold-path input validation are outside timed regions.

Study1 has one warmup, three baseline and three resident-metric rollouts, plus
one instrumented baseline. Each rollout has16384 world-integrations. Each
also executes one32-step PPO update, discarded before restoring INITIAL.
Total131072 integrations/256 DISCARDED optimizer steps; no retained policy update.
Study2 has one warmup, three baseline and three cached-reference rollouts,
plus one operation audit per mode:147456 integrations/0 optimizer updates.
The independent saved-record/fixed-input review performs0 integrations/updates.
This round totals278528 integrations/256 discarded diagnostic updates and
0 retained policy updates. Old never-fit contact records are not fitted.

## Actual collection cost

| Study1 median of3 | Collection wall / GPU stream interval | PPO update wall |
|---|---|---|
|Baseline|1.61667s /1.61660s|.12170s|
|Resident per-native-step metrics|1.65211s /1.65204s|.12777s|

Resident metrics add about35.4ms(+2.19%) to this short collection profile.
Update variation is not attributed to metric kernels, which are not executed
during update. GPU stream elapsed time includes host dispatch gaps/stream
waits; it is NOT active GPU computation time or utilization.

The separately instrumented baseline has64 controls,512 native steps,
39 adapter/environment resets,142 policy encodings,206 reference WORLD-pose
computations,103 critic encodings and142 native forward calls. Some policy
encodings occur during env.reset before the adapter overwrites its projected
initial state and computes its final observation. Those are distinct calls,
not evidence that every reset observation can safely be removed.

ATen dispatch observes323 scalar extractions:64 in adapter pending checks,
64 in the upstream manual-reset guard, and195(5x39) command reset metric .item()
reads. There are192 dynamic nonzero calls:64 environment terminal masks,
64 command resample masks and64 collector done masks. Instrumented CUDA runtime
events include95026 kernel launches,2036 stream synchronizations and5573 async
memcopies. These counts belong to the instrumented execution, not a claim that
every synchronization is a removable physics check or that instrumentation has
zero overhead. Traced CPU/CUDA ranges nest/overlap; GPU annotations and kernel
events must not be summed into a made-up percentage attribution.

## Cache implementation and numerical qualification

`src/supermimic_research/runtime/pen_reference_cache.py` holds the immutable
92-column prefix: reference joints20, reference wrist-local full pose9 and
reference links63. All369 reference phases are precomputed for each world,
using the ORIGINAL reference WORLD transform then current fixed-mount wrist
transform. It does not assume that the raw object pose already has the desired
frame convention. World origins and captured fixed mount remain explicit.

Current joints/bias/defaults, joint velocity, current object pose and previous
applied action are still read live. `motion.frame_ids` provides the same hold/
end clamping and mixed-world phases. Reference and table byte signatures,
mount/origin equality and world/device checks guard cold block boundaries.
Changing any frozen input invalidates the cache instead of silently retaining
stale geometry. Guard cost is excluded from these computational timings and
must be charged in the eventual integrated collector. Resident storage is
4677504bytes(~4.46MiB) at32 worlds. Invalid current quaternions preserve NaN
propagation rather than receiving a fabricated identity.

The native qualification passes9/9 checks: all369 phases plus3 hold/end
boundaries(372 cases), mixed phases, and142 live old/new policy calls on the
IDENTICAL current state have max error0, not just a2e-7 tolerance pass. These
checks do not prove bitwise future W32 dynamics, arbitrary devices/inputs or
generalized cameras. The full maintained suite passes305 tests, including six
cache-specific CPU cases for guards, origins, clamps, live suffix and NaNs.

| Study2 median of3 | Collection wall | GPU stream interval |
|---|---|---|
|Baseline|1.60759s|1.60752s|
|Cached reference|1.41046s|1.41039s|

Observed collection time reduction is12.2628%. Baseline reset-call counts are
38/39/40; cached counts40/41/41. Each run resets224 worlds total, uses16384
integrations and reaches source phase at most12. Reset timing differences and
known W32 numerical repeat limits prevent calling these bitwise-matched
physical trajectories or isolating the exact percentage to one operation.
This is not complete7.36s coverage. No policy weights are exported as improved.

An independent0-physics review validates saved action/clipping/timeout/phase/
reset/no-aid/finite/overflow bookkeeping across12 timed rollouts(125/125 checks).
It also profiles a pure old/new feature call on the SAME frozen input, without
the old-encoder parity audit inside the cached timed/counting region:

| Fixed-input native32 feature encoder | ATen operations / index ops | Median wall per call |
|---|---|---|
|Original|378 /21|2730.86us|
|Cached prefix|147 /9|1116.17us|

There are0 scalar/nonzero reads in either pure policy encoder. The cache removes
reference work; it does not remove the collector/manager/reset synchronizations.
The study2 cached LIVE operation audit deliberately computes BOTH encoders for
parity, so its raw total counts are not comparable to the baseline audit. Use
the independent fixed-input counts above for that comparison. Microbenchmarks
of repeated frozen inputs do not independently establish collection speedup.

## Ownership and evidence

The existing adapter/features, PPO code and portable schema3 remain byte-unchanged.
The cache is selected ONLY inside the diagnostic tool's temporary policy hook,
which is restored in finally. It is not silently attached to native checkpoint3
or made the default training interface. It requires a separate owned-state/
checkpoint integration contract together with resident contact windows.

Evidence roots under runs: native_collection_profile_v1, native_reference_cache_v1,
native_collection_review_v1. Source snapshots, all timed memories and the large
Chrome trace are preserved. Compact pins/results are in
`provenance/native_collection_results/ledger.json`. GPU source versions are
frozen before execution; retained unit logs are separate test evidence.

Next work follows research-plan-v1.01: explicit metric/cache ownership and
save/load, then geometry-aware residual/support reward experiments with original
strict gates and declared budgets. The existing failed PPO32 remains the latest
trained policy/video. No cube/pen full-task, WM physical-gradient, fair shared
multitrajectory, ego asset/dynamics or hardware validation is declared complete.

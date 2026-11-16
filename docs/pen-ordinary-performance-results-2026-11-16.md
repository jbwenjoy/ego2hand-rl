# Ordinary32-world PPO profiling and a negative optimization trial

The actual collector is much slower than its PPO update:roughly1.8s collect64
versus.12s update32 in this fresh early-training fixture. Removing a known-empty
upstream reset-index query eliminates64 observed nonzero calls but DOES NOT
improve clean timing. UPSTREAM remains the default. CPU+CUDA trace independently
identifies repeated forward/collision work and small-kernel fragmentation;
fixed-frame/reference caching is the next measured implementation target.
No main-policy/video or GPU-speedup result is claimed.

## Actual code path and retained attempts

Use maintained SupportPenRuntime/SupportPenPPO,32 worlds,centered support.2,
64 controls,32 Adam steps,seed7411,four CPU threads. Construct a new current-code
fixture,8 calibration controls,two warm iterations;save native/task/cache/NN/
Adam/normalization/memory/counters/RNG. Restore this exact same current-source
fixture before each case. Cold source/readback verification stays outside timing.
No historical native checkpoint migration or physics/driver/package changes.

1. First attempt finishes34816 native integrations/64 warm optimizer steps,then
   fails serializing a Tensor in warmup JSON. Keep failure,fixture and executed
   source. Fix only diagnostic JSON conversion;do not overwrite the attempt.
2. Corrected monitored profile completes116736 native/224 updates. Its cProfile
   data contains background GPUMonitor subprocess/thread calls,so cumulative
   caller attribution is contaminated. Preserve its data,not a clean profile.
3. Repeat registered five cases with monitoring completely disabled:116736
   native/224 updates. The isolated Python profile has no subprocess entries.
   Three clean collection times1.774/1.831/1.796s;consistency~.039s,update~.12s.
   These restores have reset-world counts189/188/192. Future32-world numerical
   replay is not assumed bitwise;these are early-training diagnostic parameters,
   NOT the main final64 actor or independent trained seeds.

The isolated Torch CPU profile contains317 nonzero operations (~539ms including
waiting for earlier GPU work),2148 scalar extractions (~24ms),and many indexing/
small arithmetic/copy operations. Do not sum nested operator/cumulative times.
18 cold tensor hashes and two reference-cache validations cost~8ms total,not
the main1.8s collector cost. Resident contact metrics sample512 actual native
steps and do not rebuild a large host geometry model on every call. Current
source differs from the older dispersed implementations the user diagnosed.

## First candidate:omit a proven-empty upstream index query

An opt-in EMPTY_SIGNALS control driver uses a specialized local chronology
for our already-empty upstream RL managers. It calls the same action/physics/
scene/termination/reward/metrics/forward/command/sense/observation/recorder code,
omitting ONLY materialization of upstream termination indices known empty after
TerminationManager.compute with zero active terms. Real tracking/geometry
termination and explicit partial resets stay in the outer native controller.

Both manual/outer pending guards remain. The helper rejects auto-reset,active
or configured upstream signals,unexpected termination manager and step/interval
events before actions/physics. CPU tests compare installed upstream chronology,
counters and outputs,count exactly one skipped query,and reject unsupported/
pending configurations. Four new tests plus prior362 pass366/366. The shared
adapter/collector accepts an explicit driver;ordinary calls still use UPSTREAM.
There is no installed-package patch or duplicate physics implementation.

The new source contains BOTH paths. A new same-source fixture is restored for
BASE_A,FAST_A,FAST_B,BASE_B,BASE_C,FAST_C and separate CPU profiles. No native
state rebind across source versions,telemetry thread,skipped gate or changed
budget. Eight cases plus calibration/warmup:165888 native/320 diagnostic updates.

|Clean collect64,seconds|A|B|C|Median|
|---|---:|---:|---:|---:|
|UPSTREAM|1.770905|1.776844|1.787581|1.776844|
|EMPTY_SIGNALS|1.789452|1.783408|1.822604|1.789452|

Median ratio BASE/FAST=.992954:the candidate is~.7% slower on these observations,
not an acceleration. Full collect+consistency+update medians1.9303/1.9458s.
CPU profile nonzero count315->251(-64),but nonzero total time526.6->536.3ms.
Waits move to real later queries;the CPU wait time was not all removable work.
Do not claim every synchronization is independently additive overhead.

BASE reset-world counts200/196/200 versus FAST198/197/193. FAST rollouts also
leave the three BASE componentwise repeat envelopes (including phase,geometry,
reward and termination masks). Example maximum geometry distance outside the
observed envelope1.979mm,phase16 controls. Baseline itself varies up to15
controls at components. These are finite observed envelopes,not probability
bounds,and differences do not identify driver causality. Contract checks pass,
but strong empirical equivalence is NOT established and no adoption follows.
The prototype remains explicit/experimental;no production caller enables it.

## CPU+CUDA evidence,not clean timing

One additional UPSTREAM64-control/32-update case loads the CURRENT comparison
fixture with full guards,then records CPU+CUDA events.16384 native/32 optimizer
steps;no monitoring. The raw trace (~381MB) is immutable and hashes verified.
GPU profiler overhead is large:collect CPU label5.967s versus~1.8s clean.

|Instrumented phase|CUDA kernel count|Sum of kernel durations|
|---|---:|---:|
|collect64|196447|738.643ms|
|consistency|4161|7.313ms|
|update32|7804|16.193ms|

Kernel sums may overlap across streams. They are not wall time,busy percentage,
occupancy or uninstrumented performance.208412 kernels total nonetheless give
direct evidence of a fragmented small-world workload.

Five CCD-named kernel groups each execute698 times,sum360.136ms(~48.76% of
collection kernel-duration SUM). The most expensive group alone324.867ms. The
saved rollout has61 partial-reset calls;observed frequency matches512 native
steps+64 post-control forwards+61 upstream-reset forwards+61 adapter-reset
forwards=698 full forward sequences. This identifies repeated collision work;
it does not prove any refresh can be removed without frame/state qualification.
Do not reduce solver/CCD settings,contacts or geometry gates just to run faster.

The initial offline trace checker incorrectly assumed the three largest CCD
groups were all the groups;its18/19 failed report remains. Corrected grouping
checks every observed group (five) and raw durations/counts,19/19. No raw trace,
physical threshold or original failure is changed. CPU/GPU capture7/7 and saved
performance arithmetic37/37 pass only their stated contract scopes;speed and
empirical-equivalence statuses remain failed.

## Concrete next changes and limitations

The isolated profile still calls NativePenFeatures.reference_pose_world64 times
(~42ms cumulative),while the owned cache already stores world_pose. It also
calls policy188 times (~231ms) and critic126 times (~133ms):62 partial resets
produce a discarded upstream observation before the adapter's final observation.
Both observation paths repeatedly form fixed wrist rotations. These are measured
shared-code targets,not a claim that all expensive geometry is redundant.

Next reuse already-guarded reference world poses,then cache the two exact fixed
wrist rotation forms used by policy and critic (normalization orders differ).
Validate all369 phases/all32 worlds and live dynamic suffixes/velocities,zero
quaternion propagation,cache mutations and immediate source/restore contracts.
Use same-current-source selectable baseline/cached comparison;do not implicitly
load old native checkpoints after editing source. Qualify one-control mechanism
outputs separately from chaotic long-horizon trajectories,then measure clean
repeats and all reset workloads. A trained-actor weights-only32-world fixture
also needs exact full32-world Model/context checks,not W1/W32 assumptions.

Only after epoch/order qualification consider coalescing duplicated reset
forward/observation work. Upstream reset commands can depend on refreshed derived
state;blindly deleting forward or observations can change public cache epochs,
random consumption or physics. Keep per-native contact sampling and actual
explicit resets. Larger world/minibatch experiments need matched total work
and new Model-derived-constant/numerical checks;no hidden budget increase.

Total actual cost across failed/monitored/isolated/comparison/GPU runs450560
native world integrations and864 diagnostic PPO optimizer steps. These DO
update diagnostic actor/critic parameters;none replaces the main64 policy.
No new WM fit in this turn. Previous WM/solver/task/ego failures remain intact.
Exact receipts/logs/rollouts/traces and counters are indexed by
`provenance/pen_ordinary_performance_results`;driver/CUDA/packages are unchanged.

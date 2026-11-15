# Read-locked source validation: faster restore batches,one retained anomaly

An opt-in Windows source-validation epoch now removes repeated whole-source
hashing from each restore while retaining full entry/exit SHA checks and per-
use manifest/file-identity checks. Default native4/PPO loads remain strict.
Matched whole batches take14.898s(strict) versus6.083s(epoch) for36 restores
each,including epoch construction,locks,hash boundaries,identity and cleanup:
ratio2.449x,about59.2% less time for THIS restore workload. Runtime construction
and warmup are excluded. This is NOT a PPO/end-to-end/GPU-utilization speedup.

Physical qualification is incomplete:one of324 initial controls fails the
unchanged linear-impulse gate. All270 cross-mode endpoint envelopes pass;
later90 selected-case replays do not reproduce the failure. Retain the original
failed row,threshold and raw packets. Do not automatically enable the feature
for new TRAIN production or call the rare discrepancy explained/fixed.

## Mechanism and preserved contracts

`runtime/source_validation_epoch.py` uses Windows GENERIC_READ/FILE_SHARE_READ
handles to block ordinary write/delete opens during a bounded transaction.
Canonical source/snapshot/engine paths and handle/path identity are bound.
Full hashes run at entry and exit;every restore checks an exact manifest match
and current identities. Changed,foreign,expired,untyped and noncanonical inputs
reject. All handles close on error,body exception and normal exit. Outputs stay
provisional until successful exit. It is not a security boundary against
malicious native code or preexisting writable mappings.
[CreateFileW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew),
[file identity](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getfileinformationbyhandle).

`owned_pen_state.preflight/restore` and probe `restore_source` accept an explicit
keyword-only `source_epoch`. Without it,the original full per-load source/engine
checks remain. Immutable reference/task/RNG/layout/alias/array checks,copies,
pointer preservation and exact post-copy readback remain in BOTH modes. No
epoch/handle is serialized into a checkpoint or attached as hidden runtime
state. The serialized native4 format is unchanged;historical states still need
their exact frozen sources rather than implicit cross-revision continuation.

Eight new tests verify default corruption rejection,real OS write/delete/replace
denial,handle cleanup after failed hashes/body exceptions,foreign/expired/untyped
tokens,canonical paths and identity checks without per-use rehashing. Full suite
345/345 passes. This is not merely a cache keyed by size/mtime.

## Current-code fixtures and W1/W32 distinction

The benchmark first generates6 NEW uninterrupted current-code W1 sources:
final64 residual/support actors at phases10,30,60. The explicit
`learning/frozen_pen_actor.py` API loads only actor/normalizer weights,verifies
historical snapshots,and checks full current W1 Model against the recorded W1
context. No historical native,optimizer,critic or RNG is loaded. All404 W1
arrays and Model scalars match that context. Source generations use no post-
start reset/restore and remain CHECK/NEVER_FIT.

This explicit weights-only path matters because loader source bytes changed.
Old native receipts are not rebound to current code. It also avoids claiming
W32 and W1 solver contexts are identical. After declared W1->W32 broadcasting,
3 of404 Model arrays differ from the trained W32 model:body_invweight0(max
abs.0029296875),dof_invweight0(.0078125),actuator_acc0(.0087890625). All scalar
leaves match. These are derived solver-context arrays;we do not assume their
differences are harmless or explain the impulse anomaly. The benchmark compares
the SAME complete W1 sources/matrices across modes. Previous20-field physical
table matches were narrower than full Model-array equality.

## Matched restore and physical benchmark

The preregistration is `source-epoch-benchmark-protocol-v1.md`. One strict warmup
per source precedes3 timed restores/mode/source;mode order alternates. All36
timed restores have exact bytes/scalars/observations. Corrupt source manifest
and immutable target payloads reject without changing full native/Python/RNG
state;expired tokens also reject atomically. Runtime benchmark80/80 passes.

The same source bank then supplies27 profiles/source/mode:2 fixed directions,
PD eps.002/.005,+/-,3 repeats and3 baseline repeats. All324 controls(8 native steps each) are
saved,not selected. Input actual PD is checked and held throughout each20ms.
733 source files are read-locked in the epoch;180 bound uses pass. Original
geometry,force/torque/impulse,no-aid and timing gates are unchanged.

Per-call restore timing is strict7.447s versus epoch2.278s for18 calls each,
ratio3.269. That timer omits some epoch boundary work. A separately registered
balanced STRICT,EPOCH,EPOCH,STRICT extension measures complete18-call batches:

|Block|Mode|Whole batch wall(s)|Restores|
|---|---|---:|---:|
|0|STRICT|7.3532|18|
|1|EPOCH|2.9806|18|
|2|EPOCH|3.1024|18|
|3|STRICT|7.5447|18|

This extension includes canonical path resolution,entry/exit SHA,locks,identity
checks and closing handles. All byte/observation checks76/76 pass,0 physics.
Use its2.449x ratio for whole-batch cost,not the larger per-call-only ratio.

## Retained physical failure and numerical checks

Independent raw-EFC/state/control/physics review is6157/6159. The two false
checks refer to ONE row:RESIDUAL_P030,epoch,D2,eps.005,sign+,repeat0(case69).
At native tick4(time.592499554s),linear impulse residual1.723085764e-6Ns exceeds
5e-7Ns. Contact decoding and generalized force/torque checks pass:residuals
2.187e-7N and2.809e-9Nm. Recorded velocity continuity is exact and its increment
agrees with recorded qacc to5.666e-10m/s. These facts locate a force-to-motion
inconsistency but do NOT identify solver/stream/cache root cause.

270/270 cross-mode endpoint comparisons(hand40,object12,warm26,wrench6,Fn6)
fall within the sum of observed3-repeat radii plus registered normalized L2
floor1e-5. This is not deterministic replay or a probability bound. The related
finite-response analysis retains24 direction/amplitude pairs and qualifies23
after excluding the pair containing that bad row. All69 block comparisons for
the23 qualified pairs are inside the corresponding observed finite-response
envelope. Do not use the unqualified pair as a trusted gradient target.

A registered selected-case replay uses30 STRICT_A,30 STRICT_B and30 EPOCH runs,
rotating order. A/B call the same default implementation. All90 source restores
and replay contracts93/93 pass;all physical decoder checks pass. Each mode's
maximum impulse residual is4.945e-10Ns. Read-only extra Data capture would occur
on an anomaly;none fires. This fresh-process/instrumented replay does not
reproduce the original failure and does not establish its cause or population
rate. Original failure remains6157/6159. Independent outcome arithmetic257/257
explicitly checks that the original failed gate remains false.

Replay v1's source-list expression contained an extra nonexistent package name;
it still captured the same3 real packages. The no-op enumeration typo was fixed
after completion;the actual executed source bytes remain in its snapshot.
No physical run was discarded or repeated to rewrite a passing result.

## Costs and next work

This cohort costs4256 native world-integrations:944 new source-prefix +2592
matched benchmark branches +720 selected-case replays. Timing/readback/outcome
analysis/unit tests add0 native steps and0 neural/Actor/optimizer updates.
No package,driver or CUDA modifications. Canonical evidence is
`provenance/source_epoch_results/ledger.json`;all raw traces/source states and
timings remain under the corresponding `runs/source_epoch_*` folders.

The epoch remains an explicit experimental option;ordinary PPO still uses the
strict path. The earlier training geometry/synchronization/utilization issue is
NOT completed by this restore-only result. Continue its own matched collector
profile separately. Keep physical masks on every newly collected row and keep
the rare counterexample as an unresolved constraint on stronger equivalence
claims. Separate TRAIN data can still be collected using the strict path;this
performance prototype must not become an invented blocker for the research.

Full original research goal remains ACTIVE/incomplete:old physical WM gradients
still fail,original full cube/pen task/shared A1/A2/multitrajectory/ego/hardware
requirements are not satisfied by a faster restore. Current CHECK rows never
become TRAIN and no model-based Actor use is justified by this change.

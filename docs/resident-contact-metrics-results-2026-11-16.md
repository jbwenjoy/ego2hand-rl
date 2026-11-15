# Resident native contact metrics — 2026-11-16

The resident read-only metric implementation passes saved-record GPU replay and
live 1/32-world native hooks. This qualifies contact arithmetic and accumulation;
it does not qualify a controller, physical world model, portable physics replay,
CHORD benefit, ego generalization or end-to-end training acceleration.

## Shared implementation and physical meaning

`src/supermimic_research/runtime/resident_contact_metrics.py` binds the actual
native contact, EFC, joint-bound and body-transform arrays. Static geom/body,
canonical finger1..5/other-robot labels and joint indices are uploaded once.
Four kernels clear instantaneous outputs, decode active contacts, check each
world and fold a native sample. The kernels are captured in one metric graph;
sample calls read no contact count or scalar back to the CPU. They do not write
simulator arrays or apply any force. FP64 sums reduce accumulation roundoff.

Only the ACTIVE prefix is accessed; deliberately invalid/NaN inactive storage
does not become a contact. Invalid active world/geom/EFC IDs, nonfinite active
values, unsupported dimensions, negative normal, capacity/overflow, joint-bound
and body-rotation errors are flagged. Active physics is never clamped/repaired.
The object penetration statistic only includes contacts with exactly one object
geom, while robot self penetration requires two robot geoms. Their original
2mm and0.2mm gates are retained, not replaced by an all-contact maximum.

Pyramidal condim1/3/4/6 uses1/4/6/10 actual EFC addresses; elliptic uses dim
components. Complete local force AND torsion/rolling torque are transformed
with the contact-frame rows and correct object geom0/geom1 sign. Moments are
about `xipos` COM at the EFC geometry phase. Native contact/body transforms after
step belong to the pre-integration forward; joint excursion uses post-step
qpos. No stale post-step qpos reconstruction is substituted in the live hook.

The window separately preserves WORLD/COM(t), BODY(t)/COM(t), source-axes/COM(t)
and source-axes/source-COM wrench sums, for both all-object and hand-only loads.
`M_sourceCOM = M_COM(t) + (COM(t)-sourceCOM) cross F`. A mean over changing
BODY axes or COM origins must not silently become a source-frame support
target. Divide sums by native sample count before using a mean reward.

Partial metric reset accepts GPU int32/int64 IDs without host content reads.
Range and duplicate checks execute on device; a bad list resets no worlds and
sets a sticky error. A successful full reset clears the global error; partial
resets preserve it. Cached Warp/Torch stream ownership orders producer writes,
metrics and consumers without per-sample host synchronization.

## Observed validation

| Study | Samples / physical cost | Result |
|---|---|---|
| Saved GPU replay v1 |648 packets,2632 world-samples,5109 contact rows;0 new physics|81/81|
| Live native hook v2 |128 native samples each atW1/W32;4224 world integrations|63/63|
| Full maintained unit suite |299 tests, including14 new analytic cases|299 passed|
| Fixed-input metric profile v2 |5120 timed graph samples;0 new physics|10/10|

Saved replay covers all584 native ticks of INITIAL/ITER2/ITER32 strict evaluation
and64 W32 archived native substeps. An independent NumPy oracle decodes original
raw EFC, checks forces/geometry/joints and all instantaneous/window frames.
The W32 source's native geom-body and joint-index topology is checked against
the pinned host model; its own native joint limits are used. Single-world body
geometry is reconstructed by584 CPU kinematics queries from pre-step qpos and
mocap, not by a new rollout. Never-fit diagnostic data is not fitted/promoted.
Archived float32 moment-arm subtraction is reproduced separately: its maximum
difference from the new double subtraction is8.852e-10Nm, below declared1e-5.

Live validation samples EVERY2.5ms native step and folds each8-step/20ms action
window. A partial metric reset at sample4, explicit partial environment reset
before control8, and terminal-world resets exercise isolation. Invalid CUDA IDs
from a non-default caller stream do not partially reset valid worlds. Maximum
independent live discrepancy is4.441e-16. Input byte digests are unchanged at
every sample. At native indices4/5/6, all169 native Data array fields and scalar
leaves are copied before physics, before metrics and after metrics; all Data
bytes before/after metrics match. All404 native model array fields also match
before/after this diagnostic. These are NEW diagnostic sources, not the old
W32 divergent continuation's sources; numerical-repeatability remains open.

The initial live v1 script attempted another action after terminal without an
explicit reset and was correctly rejected by the adapter. Its log and source
receipt are retained. It did not persist its integration counter before exit;
its extra physical cost is UNKNOWN in[0,128] world-integrations, not reported as
zero or inferred from a later rerun. v2 records each termination and resets
explicitly. No task success is inferred from these short, reset diagnostics.
This round has4224 confirmed new integrations plus that[0,128] failed-run bound,
and0 optimizer updates. The saved failed unit-debug logs are also retained.

## Performance scope

The fixed-input profile selects the densest observed packet:5 contacts/W1 and
160 contacts/W32. Native contact capacities192/6144 and EFC capacity4096 are
retained in v2. After128 warmups per case, five repeats of512 metric samples
give median GPU stream intervals of27.24us/W1 and34.10us/W32 per sample. These
include graph submission/stream-wait effects; no CPU raw-contact dump occurs
inside the timed loop. There are no retained Torch allocation/reservation
deltas, which says nothing about Warp/driver pools or transient events.

An earlier v1 microbenchmark used minimal guarded contact capacities8/163,
measuring27.27/33.59us; it is retained and explicitly superseded for native
capacity. Both use a repeated fixed input, without simulation, observations,
PPO collection or updates. These numbers cannot establish total collection or
training speedup, nor optimize the old learned-WM geometry path by themselves.

## Evidence and next boundary

Runs: `resident_contact_replay_v1`, `resident_contact_native_v1`(failed),
`resident_contact_native_v2`, `resident_contact_profile_v1/v2`.
Each GPU run freezes source bytes before executing; exact historical aliases
must resolve those versions after later tool changes. Compact evidence is in
`provenance/resident_contact_results/ledger.json`; large raw/capture records stay
ignored under `runs/`. The shared package and existing native checkpoint schema3
are unchanged apart from the new standalone metric module. The live hook is
external to the adapter; no claim that current checkpoint schema3 captures
metric windows/graphs/streams is made.

Next: profile actual collection components at equal native/action/update counts,
declare a separate metric-bearing checkpoint/transition contract, validate
partial reset/save/load ownership, then predeclare geometry-aware rewards and
strict task gates. Preserve the old PPO32 baseline and only explicitly load its
weights into a new contract if that initialization is chosen. Support targets
must use audited BODY axes/COM/time and remain reward targets, not applied force.
The complete original task, full-horizon no-aid success, A1/A2 budgets, physical
WM gradients, real ego assets/dynamics and hardware contracts remain unfinished.

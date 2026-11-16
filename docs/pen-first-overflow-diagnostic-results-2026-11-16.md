# First-overflow recorder admitted; bounded replay does not reproduce incident

Original failed long trial and matched recoveryV2 remain terminal/incomplete.
Neither final evaluator is eligible. This work changes numerical observability,
not policy success, ego retargeting, or world-model Actor admission.

## Implementation and qualification

The external FirstOverflowObserver borrows raw int32 overflow, phase, qpos,
qvel, warmstart, ctrl, time and all geometry transforms. It retains the first
nonzero mask per world and pre/post operation fields on GPU, without per-step
CPU scalar reads or simulator writes. STEP and FORWARD hooks execute readings
and native work on the owned Torch/Warp stream. Fields survive later resets;
packets are explicitly NOT_RESUMABLE and do not identify a collision pair.

Four unit tests cover first-event retention, conflicting later flags, pre/post
bytes, no live writes, no Tensor.item reads and invalid/nested calls. Actual
GPU qualification10/10 covers shared Warp/Torch pointers, stream ordering,
synthetic bit256/reset/later bit128 and exact retained fields. Full435 tests
pass11.516s. These are observer and software checks, not physics repair.

## Registered replay202..300

Source: valid RESIDUAL201 from recoveryV2, ordinary full restore exactly once.
Immediate854 native arrays/857 scalars, pointers, modules, Adam, memory,
observations and normalization match. No parent metadata was rebound.

The diagnostic session49539 exited0 after the fixed99-collection budget.
All99 original collection checks pass25/25; there are no captured events or
raw masks. Actual added work1,622,016 native integrations and3136 Adam calls,
attempted=completed. There are50,688 native step calls and10,014 diagnostic
forward calls. Main-entry wall195.3973s includes model/evidence operations,
excludes Python imports. Final collection300 is not optimized: terminal policy
diagnostic299/9568 cumulative Adam/614400 collected transitions. No new valid
training checkpoint or final policy video is asserted.

Replayed202 mean reward1.3628009558 differs from originalV2's1.3656206131.
Immediate restored bytes do not prove future W32 trajectory identity. Absence
of an event in this bounded replay is not proof the earlier252 incident is
fixed or that all forward paths are capacity-safe. Both original failures,
their scopes and all incurred costs remain in the evidence chain.

## Installed official EPA helper control

Session66877 exited0; eight checks pass using the installed official
`GJKTest.test_box_box_horizon` and its EXISTING horizon_size helper argument.
No package source or production scratch allocation was patched.

|Helper scratch capacity|Raw native mask|Contacts|Distance|
|---:|---:|---:|---:|
|24 (installed default)|0|1|-0.000115788221592m|
|5 (intentional undersize)|256|0|~1e30, invalid output|
|48|0|1|-0.000115788221592m|

Default24 and48 valid distance/contact-position outputs match FP32 bytes.
This confirms the positive overflow control and nonoverflow capacity control
on the official box fixture. It is not our pen incident, a production engine
variant, or a demonstrated fix. Counts:3 isolated collision queries,1 official
fixture initialization,0 native integrations/Adam. Retain its compilation wall
and source pins separately; zero integrations does not mean zero compute.

## Next evidence required

Do not simply extend/restart blind training. Current recorder is ready for a
new explicitly bounded diagnostic, but repeated future paths remain uncertain.
First reduce the ORIGINAL retained252 geometry into a collision-only query.
Useful stored data: source_R/source_C at next control row63, terminal policy
at row62, and full parent Model geometry inputs. Parent robot encoder bias
and default positions are exactly0; the policy's hand-joint block92:112
therefore retains direct joint positions. The wrist is fixed. The original
incident may have arisen during an extra forward between the control rows.

A reconstruction is still a derived fixture: source metrics and policy must
be shown to represent the same instant; matrix/quaternion conversions, mesh
frames and FK precision can perturb EPA. First round-trip a known native
snapshot and measure all geometric errors. Prefer retained rotation matrices
over arbitrary inverse-quaternion approximations where possible. Enumerate
the pinned candidate set; don't label a close pair as the failing pair. Count
all collision queries/forward builds separately. Only reproduction with raw
flags can support a capacity experiment; otherwise preserve the negative
result and uncertainty. No native checkpoint migration or production engine
change is authorized by helper-only success.

Full original M0–M5, controlled/real ego data contracts, physical WM gradient
admission, paired total cost and multitrajectory/seed evaluation, real ego
gravity/inertia/friction/RGB calibration and hardware contracts remain active.
No live training, no new ego success or speedup claim; goal incomplete.

Evidence: runs/pen_first_overflow_diagnostic_v1/report.json,
runs/pen_first_overflow_observer_qualification_v1/report.json,
runs/pen_epa_horizon_helper_probe_v1/report.json.

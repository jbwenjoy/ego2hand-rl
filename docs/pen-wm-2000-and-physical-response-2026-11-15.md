# Resumed pen WM pair: complete fixed budget, failed predictive/gradient readiness

The user resumed the Goal after the healthy first100 checkpoint pause. This
turn makes measurable progress: exact continuation to2000, recursive CHECK
predictions, and NEW physical finite-amplitude action-response evidence. The
full original Goal remains ACTIVE and incomplete. No Actor was updated and
there is no new policy-control video or real ego/generalization result.

## Exact continuation, not a random restart

Code27b7f93 adds an explicit continuation entry point. Frozen first100 source,
config, data, roles and checkpoint bytes remain unchanged. The entry verifies
the parent health/pins/schema,100-update Adam, model and sampler restoration,
and exact original CHECK metrics BEFORE training. It restores the saved
optimizer/sampler and frozen scales/physics, then performs1900 additional
updates PER model, ending at the original fixed2000 budget. CHECK curves at
500/1000/2000 are descriptive; they do not select a checkpoint or fit scales.

`runs/pen_wm_continuation_v1/pen_response_checkpoint_2000.pt`:
20,190,119 bytes, SHA256
`d65d13224280562db1c949c46beb3bc02fcea137cd964581bd73c1a47418adb8`.
The parent remains
`cb6161693761aa72701532765133609bd756f3cf7ac89b283bedfe0b10bd927f`.
Both arm cursors/Adam steps are2000.44/44 save/readback checks pass, including
frozen buffers, matched sampler/batch and two restored clones' next-update
loss/gradient/model/Adam/sampler parity per arm. The four additional diagnostic
clone steps never modify the live/saved2000 cursor. Actor updates remain ZERO.
The training process exited0; no training or collection remains running.

1900 paired updates plus the three CHECK metric calls take13.1849s wall time
on RTX5090, consistent with the prior conditional13s fixed-data estimate.
It excludes setup, collection, packing, checkpoint QA, later diagnostics and
policy learning. Torch peak allocation94,905,344B, reservation96,468,992B.
51 device-wide telemetry windows report0–22% GPU use; this tiny MLP/data
experiment is NOT a sustained GPU-utilization or whole-pipeline speed claim.

## Overfitting and accumulating recursive error

TRAIN remains157 valid rows, CHECK80, fixed initial state/clip with held-out
action streams. TRAIN error is much lower: normalized object12 MSE
1.71e-5(DIRECT),0.0024094(IMPULSE). CHECK
object12 is0.37891/0.37932. DIRECT is worse than its100-update0.31081; IMPULSE
improves from0.44004, but neither establishes useful held-out accuracy.
At2000 wrench6 MSE is0.23524/0.20053; these are mixed component improvements,
not overall success or causal evidence for a policy support objective.

`runs/pen_wm_final_diagnostics_v1/report.json` uses ALL contiguous valid
CHECK windows, never crossing episode IDs/gaps/dt. First state/warmstart are
observed; later state55/warm26 are recursively predicted. Only the actual
recorded PD targets and native dt are supplied. No teacher forcing, joint
clipping, reset, interpolation or physical-object aid. Independent unit tests
exercise recursive state/warm propagation, world-left rotation and episode
boundaries. All windows remain finite, which alone is not predictive accuracy.

|Horizon|Windows / episodes|DIRECT COM mean|IMPULSE COM mean|DIRECT full angle|IMPULSE full angle|
|---|---:|---:|---:|---:|---:|
|20ms|80 /8|1.12mm|1.07mm|4.65deg|4.71deg|
|40ms|72 /8|2.38mm|2.28mm|9.15deg|9.54deg|
|100ms|48 /8|7.57mm|6.88mm|34.49deg|31.85deg|
|200ms|8 /4|21.38mm|18.01mm|129.04deg|125.71deg|

Windows overlap and horizons use different available subsets; they are NOT
independent samples or a fixed-population statistical comparison. Full-angle
geodesic errors include axial roll and are distinct from the native task's
shaft-direction termination. At200ms both full-angle errors even exceed the
state-persistence comparator's96.09deg. Positions beat persistence but do not
qualify full task tracking. Predicted warmstart error also grows; it is
privileged engine context, not a demonstrated hardware observation.

Network AD agrees with the same network's central FD (unit-L2 PD directions,
eps1e-3/3e-4/1e-4), as recorded in the report. This checks derivative
implementation/self-consistency, NOT derivative agreement with physical data.

## Actual physical response disagrees with learned derivatives

Predeclared config`configs/pen_physical_gradient_v1.toml`, committed1ed930b
BEFORE collection:2 unit-L2 sine directions; PD amplitudes0.002/0.005rad;
positive/negative signs;3 execution repeats;5ms/2.5ms dt;3 zero-baseline
repeats at each dt.54 actual original-plant episodes are saved in
`runs/pen_native_gradient_v1`. Only their FIRST20ms response is used here;
all episodes continue through the unchanged reference clock until first native
terminal, without post-start resets/aid. Raw first-action offset4*eps*d
accounts for original0.5 action scaling and0.5 EMA; remaining raw actions
return to original2*delta. Actual unclipped PD offsets are verified, not assumed.
No neural model selects directions, amplitudes or favorable episodes.

All54 are explicitly CHECK_PHYSICAL_RESPONSE_NONEXPERT_NEVER_FIT. They are
diagnostic data and must never be appended to TRAIN. Because this evidence
now guides development, it is not an untouched final evaluation of later models.
The same source clip/environment seed42/selected initial pose are retained.
Repeats measure execution variability, NOT independent physical trials.

`runs/pen_native_gradient_review_v1/report.json` independently reconstructs
saved GPU EFC labels/wrenches:378/378 full-episode receipt checks pass.
383/383 source/control/label contracts pass: common initial state+warmstart,
actual PD perturbations within2e-7rad, same physical tables and frozen-WM
physics, nonfit roles, and every first-interval numerical-quality tick.
These are DATA quality checks, not383 passing derivative-accuracy checks.

For each dt/direction/amplitude, compare the model's center-input AD to
the mean native central response, scaled by the SAME frozen TRAIN y-scales.
Three repeated executions quantify observed within-sign spread. All16
mode/condition contrasts exceed that observed spread (32 block comparisons).
No probabilistic confidence bound or smooth-contact theorem is claimed.
Native derivative change between the two amplitudes is1.3–6.0% for object12
and2.7–20.9% for wrench6; local finite-amplitude response is more stable than
the following model errors, but still not an infinitesimal derivative proof.

|Model|object12 relative L2 error|object12 cosine|wrench6 relative L2 error|
|---|---:|---:|---:|
|DIRECT|1.027–3.308|0.125–0.542|0.837–2.660|
|IMPULSE|0.511–2.988|0.283–0.865|1.161–2.078|

These8 conditions per model are highly local/correlated. IMPULSE is better
for part of the object response, but both derivatives remain poorly matched;
wrench results are mixed. A healthy checkpoint and a self-consistent AD graph
therefore do not make either model ready to optimize a physical Actor.

## Reproduction and next decision

Full repository suite:178 tests pass (`runs/pen_continuation_final_tests.log`).
Tests intentionally isolate native tests on CPU; their CUDA-unavailable message
does not describe the actual separately recorded RTX5090 runs. New training,
collection and evaluation code were committed before their real runs.
`runs/pen_wm_final_diagnostics_v1/summary.png` is the inspected scientific
figure; its adjacent JSON pins the inputs and rendering script. Compact result
copies and the full original artifact SHA ledger live in
`provenance/pen_wm_2000_results`. Large arrays/checkpoints remain ignored.

Next: broaden action coverage with NEW role-declared data, especially paired
actions at comparable source states, instead of more epochs on157 rows.
Keep these54 diagnostic episodes, the old CHECK80, and all previous artifacts
out of TRAIN. New first-control data alone cannot solve later-state/horizon
drift: include predeclared later source-control anchors under unchanged clock,
and verify actual source/warm/control/quality for each branch. Plan fresh
TRAIN/CHECK action directions/seeds, keep shared prefixes from leaking into
training/evaluation rows, and fit any new normalization on NEW declared TRAIN
only, with a new schema/version/lineage. Maintain matched DIRECT/IMPULSE
capacity/data/update budgets; evaluate actual held-out response and recursive
rollouts before any learned-model Actor update. Old diagnostic directions can
remain development regression probes, never evidence of unseen generalization.

The full scope remains: fixed-wrist input/frame/geometry/dynamics/controller
contracts, no-aid cube/pen control learning and unified evaluation, measured
support-objective A1/A2 comparisons, physical-gradient validation, total-cost
multi-trajectory sharing, real ego reconstruction/assets/validation and hardware
observation/gain/latency checks before deployment. Nothing here supplies a
successful expert trajectory or removes these requirements. No driver/CUDA,
installed simulator, archive or historical environment was changed.

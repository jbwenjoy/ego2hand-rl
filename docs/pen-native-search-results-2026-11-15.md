# Native finite-action control: longer failed prefixes, no consistent support benefit

Previous goal turn was PROGRESS: the predeclared native controller ran four
actual no-aid cases, independent review retained failures, and actual saved
trajectories were rendered and decoded. The full Goal remains ACTIVE and
incomplete. No Actor/network update, successful expert, stable pen task,
real-time controller or real ego result was obtained.

## Executed experiment and provenance

The shared native planner and config `configs/pen_native_search_v1.toml`
implement plan v0.75. The first setup (fd4071a) failed before any forecast or
action: Warp could not capture the imported default Torch stream. Its original
source bytes and failed intent/log are preserved by
`provenance/native_search_capture_failure_v1/record.json`. Commit d3e8cc5
uses a dedicated interoperable Torch/Warp stream with explicit stream waits;
the successful collection is a fresh directory, not a rewrite of that failure.
No installed package, driver/CUDA, frozen archive or physical model was edited.

`runs/pen_native_search_v2/report.json`, SHA256
`e795899999f286a1a8ef2d7420dc99be9190a9be0307e9e7c97177fa7cb19ecb`,
contains four actual episodes using the same selected initial pose, original
single upstream reference (369 frames, 7.36s), original control EMA/scale,
encoder bias, no post-start live reset and no applied object assistance.
Full fixed-model 404 arrays/195 scalar leaves are saved before control;
169 dynamic source arrays and all 57 forecasts are saved at every decision.

One independent one-world Data allocation is reset from resident source
buffers for each candidate; source live state/controller/clock stay read-only.
Each decision considers the same 57 predeclared raw candidates and native
20ms horizon. STATE_ONLY scores reference COM, directed shaft, hand posture
and bounded physical EMA. STATE_PLUS_SUPPORT adds weight0.1 on interval-mean
WORLD COM wrench error (force0.1N, torque0.005Nm scales). The nominal target
comes from reference momentum change under actual model m/I/gravity, not
observed human forces. Last3 reference intervals are derivative-invalid.
Original geometry/joint/nonhand/capacity gates precede cost ranking.

## Actual task results

All cases terminate for native object COM error over50mm; none completes the
original7.36s task or passes unified COM/orientation/duration gates.

|dt|objective|duration|COM RMS / max|decisions|candidate native ticks|
|---|---|---:|---:|---:|---:|
|5ms|STATE_ONLY|0.40s|18.84 / 57.06mm|20|4560|
|5ms|STATE_PLUS_SUPPORT|0.32s|22.21 / 62.33mm|16|3648|
|2.5ms|STATE_ONLY|0.34s|20.25 / 59.91mm|17|7752|
|2.5ms|STATE_PLUS_SUPPORT|0.34s|21.27 / 61.92mm|17|7752|

All executed geometry/self/joint/nonhand/no-aid/finite/capacity checks pass;
maximum all-contact penetration is about1.977mm. There are zero least-violation
fallbacks. Full-orientation RMS is about61–63deg, despite terminal directed
shaft errors11–17deg: shaft tracking does not constrain axial roll, and the
unified full-orientation gates remain unchanged. Failed prefixes are nonexperts.
The prior fixed-residual episodes last0.22/0.20s. Longer duration alone does
not establish stable control, support benefit or a qualified policy.

Total work is70 decisions,3990 candidate rollouts,23712 candidate physics
ticks plus416 actual ticks =24128 native integrations, zero neural/Actor steps.
Equal57 candidates/decision is not equal total compute when episodes differ
in length. Common prefixes are16 decisions (0.32s) at5ms and17 (0.34s) at2.5ms.
Mean candidate CUDA time is215/137ms at5ms (state/support),258/259ms at2.5ms;
mean wall time including artifact I/O is254/175/298/299ms per decision.
These exceed the20ms control period and establish no real-time speedup.

## Independent review: preserve two kinds of failure

`runs/pen_native_search_review_v1/report.json` independently recomputes raw
actions, original EMA, source state/warmstart/clock, selected PD targets,
nominal momentum targets, cost and argmin/fallback from saved arrays.
The mixed contract count is945/946, not a global pass: one case's aggregate
actual EFC check fails. Actual force/momentum checks are27/28.
STATE_PLUS_SUPPORT at2.5ms has maximum linear-impulse residual
5.143935482e-7Ns against the unchanged5e-7Ns gate. Force/torque identity checks
pass; the impulse failure remains a failure rather than a relaxed tolerance.

Selected forecast versus subsequent actual20ms execution passes all original
state/control/time/force gates at68/70 decisions. At5ms support decision2,
force/torque errors are0.000906N/0.0000432Nm; decision7 has
0.005431N/0.0001733Nm, exceeding1e-4N/1e-5Nm gates. State, targets and clock
still pass. Source/control audits cannot substitute for contact-force fidelity
at newly visited states. The review exits1 after saving these genuine failures.
Debug cost-roundoff tolerances do not change any physical gate.

## Tracking diagnosis and next falsifiable change

`runs/pen_native_search_tracking_v1/report.json` uses independent actual
WORLD wrenches. On the common5ms prefix, mean endpoint COM error is5.35mm
for state-only and15.14mm for support; nominal force errors are0.118/0.127N.
At2.5ms COM errors are13.98/14.58mm and force errors0.151/0.133N. Thus reducing
nominal force error does not consistently improve tracking or survival.
Positive hand load frequently persists until the terminal interval; an early
loss-of-contact explanation alone is not supported.

Final transverse COM errors are56.7/61.4/58.6/60.7mm; axial components are
only6.5–12.7mm. Source velocity errors reach0.53–0.68m/s. Across each final
sampled57-candidate set, endpoint COM envelope diagonals are0.09–0.32mm,
far smaller than the accumulated tracking error. These are finite sampled
envelopes, not global reachability bounds or a proof that control is impossible.

The current support target is pure reference feedforward. Once actual state
departs from reference it does not ask for restorative momentum. Next test
one predeclared translational feedback support target, keeping candidates,
horizon, score weights, torque target, physical parameters and hard gates
fixed. This is a hypothesis, not a proven cause. See plan v0.76. Current failed
states and forecast mismatches are not approved imitation/gradient data.

## Validation, video and original scope

194 unit tests pass (including analytic interval force/angular momentum,
derivative-tail masks, fixed candidate mapping and constraint-first scoring).
CPU-isolated tests do not replace separately recorded actual CUDA execution.
Both rendered videos were decoded at start/mid/end, with exact frame-count
and1536x800 checks; midpoint/end images were visually inspected. Renderer
uses saved actual qpos and graphics forward only, zero new dynamics. Columns
freeze at their explicitly labeled native terminal times;20x slow motion,
no interpolated physical states. Videos:

- `runs/pen_native_search_video_v1/GPU_pen_native_search_0.005s.mp4`
- `runs/pen_native_search_video_v1/GPU_pen_native_search_0.0025s.mp4`

Compact reviews/diagnostics/video metadata and SHA ledger are retained in
`provenance/pen_native_search_results`; large arrays/videos remain ignored.
The full research objective still requires actual learned SuperMimic/CHORD
control and save-resume, fair total-budget support-policy comparisons,
world-model contact/gradient checks, cube/pen unified evaluation, multiple
trajectories, true ego input/asset/dynamics validation and hardware observation/
zero/gain/latency contracts before deployment. This native controller probe
does not replace those requirements. No training is currently waiting.

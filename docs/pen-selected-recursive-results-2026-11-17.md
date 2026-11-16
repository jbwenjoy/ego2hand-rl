# Sealed first18 outcomes and recorded-control recursive prediction

This is an intermediate development result. The registered full54 native-screened
experiment remains separate and incomplete at this checkpoint. Preserve v1.44,
all original physics gates, all invalid/missing data and the full research scope.

The immutable inputs are `runs/pen_native_screened_review_v3/indices/completed_18.json`
and `runs/pen_native_screened_summary_18_v1/report.json`. All18 cases have16/16
independent saved-record checks. The descriptive summary passes7/7 and includes
all54 declared CSV rows, with36 explicitly unreviewed at this checkpoint.
No unreviewed result is counted as a success or failure.

These18 cases are the entire first numerical repetition: both fixed teachers,
all3 models, both scoring modes, and original/sham/random controls. They are not
independent training seeds or ego trajectories. There are0/18 complete7.36s
executions,8 physical-audit stops and10 task failures. All1765 selected controls,
3393 native queries and27144 native integrations are retained;4859 future
controls are unavailable after stopping. The13 invalid ALT queries and8 invalid
BASE queries remain explicit. Of1054 accepted proposals,4 were duplicate-BASE
SHAM proposals. Thus even repeated nominal actions need their own numerical
control; differences across executed trajectories alone do not identify a
model-score effect. RESPONSE_ONLY's OBS/FULL RAW gradients are exactly equal
at all493 inspected same-source gradients; its path score is a positive constant
with zero derivative, not a zero-valued path score.

Both teacher previews show every first-repeat controller in fixed order:

- `runs/pen_native_screened_repeat0_preview_RESIDUAL_v1/RESIDUAL/selected_native_grid.mp4`
- `runs/pen_native_screened_repeat0_preview_SUPPORT_v1/SUPPORT/selected_native_grid.mp4`

Both pass5/5 rendering checks. Start/middle/end images were manually inspected.
The videos use saved actual2.5ms selected poses,20x slow, no interpolation;
terminated panels freeze visibly. These are native-screened oracle executions,
not world-model-generated policy rollouts. SUPPORT preview rendering overlapped
the existing producer's R1_RESIDUAL_NATIVE_SHAM P033–P042, native29072–29216;
case timings include this concurrent work and are not isolated speed measures.

The CPU diagnostic was registered before evaluation in
`runs/pen_selected_rollout_diagnostic_registration_v1/registration.json`, using
`configs/pen_selected_rollout_diagnostic_v1.toml` and the frozen protocol
`docs/pen-selected-rollout-diagnostic-protocol-v1.md`. Data preparation passes5/5,
evaluation21/21, and the separate array/recurrence review170/170. All use zero
native integrations, zero optimizer updates and zero normalizer fits. Measured
preparation/evaluation/review times are12.5993/16.4649/7.0088s on the shared machine;
these exclude prior data collection/training and are not adaptation speed claims.

Each window observes actual state55 and warmstart26 only at its start. Subsequent
state and solver memory are predicted recursively, with the recorded future
PD20 controls supplied externally. There is no future true-state reset, memory
zeroing or state clamp. The target is fixed-control prediction, not reference or
policy generation. Every physically qualified contiguous window is used for
every model. The horizons use different cohorts because stopped/masked futures
cannot be manufactured; overlapping windows are not independent experiments.

| Horizon | Declared windows | Available complete | Physical-qualified | Masked | Unavailable after stop |
|---|---:|---:|---:|---:|---:|
|20ms|6624|1765|1757|8|4859|
|40ms|6606|1747|1739|8|4859|
|80ms|6570|1711|1703|8|4859|
|160ms|6498|1639|1631|8|4859|
|320ms|6354|1495|1487|8|4859|

All3 models produce finite forecasts on all8317 qualified windows per model.
The independent review separately enumerates every window, reconstructs states
with SciPy rotation exponentials, compares endpoints to packed native next55,
and exactly recomputes every CPU forward from the saved predicted state and
predicted prior warmstart. It also recomputes errors and own-source-frame world
wrenches. Rotation orthogonality defect is at most3.93e-15, with determinant
within3.4e-15 of1; the large angle errors are not a broken rotation chart.
No native derivative accuracy or physical force-balance admission follows.

At320ms, pooled1487-window errors are:

| Model | COM median / p95 (mm) | Angle median / p95 (degrees) |
|---|---:|---:|
|Unchanged initial pose baseline|8.138 /17.682|20.378 /51.663|
|RESPONSE_ONLY|7.199 /87.367|17.446 /152.349|
|PATH_POINT|6.744 /74.951|13.936 /158.832|
|PATH_POINT_RESPONSE|10.614 /69.745|21.180 /149.974|

The medians alone obscure large forecast tails. All3 models have320ms COM p95
about3.94–4.94 times the unchanged-pose baseline and angle p95 about2.90–3.07
times baseline. PATH_POINT beats baseline median COM in3/9 residual-teacher
cases and9/9 support-teacher cases, and median angle in4/9 and9/9 respectively.
These are descriptive overlapping-window results, not18 independent trials or
a causal model ranking. No model receives long-horizon Actor admission.

Machine-readable270-case/horizon/model rows and the inspected all-model plot:
`runs/pen_selected_rollout_diagnostic_review_v1/cases.csv` and
`runs/pen_selected_rollout_diagnostic_review_v1/recursive_errors.png`.
The review report SHA256 is
`729bee64b77259d56e7a40638d70b12c2ba5dfcd6a979dfc9e76989259ce27f6`.

Next finish the existing full54 producer and its independent review/aggregate.
After closure, observe solver gradient, accumulated Ma versus direct M*qacc,
smooth/bias/passive/constraint force at the same failing native tick, without
relaxing the5e-7Ns original impulse gate. Diagnose physical labels before new
model fitting. Future multi-step fitting needs explicit TRAIN/validation
trajectory ownership, state/warm propagation and held-out evaluation; this
inspected first18 set must not silently become a held-out test after tuning.
True A1/A2, independent seeds, multiple references, total adaptation costs,
real-ego geometry/dynamics/gravity and hardware observation/control contracts
remain open.

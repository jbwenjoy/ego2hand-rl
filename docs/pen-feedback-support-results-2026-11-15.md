# Feedback support does not consistently improve native tracking

Previous goal turn was PROGRESS: actual controller videos were generated and
decoded. This turn closes that result/provenance record, executes plan v0.76's
six-case feedback-target ablation, independently reviews every selected
forecast and analyzes all saved feasible candidates. Full Goal remains ACTIVE,
with zero new neural/Actor updates and no stable task or true ego result.

## One change to the support target

Predeclared plan v0.76 was committed in4f1bcad before collection. Shared
implementation e0a9630/2c9df7b and `configs/pen_native_feedback_v1.toml`
were committed before the new run. The active experiment orchestration and
independent review now live under shared `src`; entry-point tools are thin.
The preceding two-mode tools are frozen historical drivers; their source
pins and reports remain unchanged, not parallel implementations to patch.

The new WORLD COM force target is
`F_nominal + m*(100*(p_ref[k]-p_source)+20*(v_ref[k]-v_source))`.
It preserves nominal torque, original reference derivatives/masks and score
weight0.1. It is an objective only: no corrective force is applied to the
object. Source time is k, while tracking endpoints remain k+1. No clipping,
state repair, gain sweep or physical parameter change was introduced.
Original raw candidates, EMA, target clamps, pose, model, horizon20ms,
57 candidates/decision and no-aid/task/replay thresholds remain fixed.
Reference tensors and source q/v views are device resident; the candidate
loop uses the existing native graph and dedicated interoperable stream.

All three modes are rerun at each dt, rather than pairing a new treatment
with only historical controls. Actual report:
`runs/pen_native_feedback_v1/report.json`, SHA256
`591050ecfd8b31abaf70482f726151f88f914bd51c68e71f8bdeb78337480f4e`.
This is development on the same single controlled clip, not a held-out test.

## All six actual tasks still fail

|dt|objective|duration|COM RMS / max|decisions|
|---|---|---:|---:|---:|
|5ms|STATE_ONLY|0.40s|18.78 / 55.67mm|20|
|5ms|STATE_PLUS_SUPPORT|0.32s|20.63 / 58.71mm|16|
|5ms|STATE_PLUS_FEEDBACK_SUPPORT|0.34s|19.77 / 56.76mm|17|
|2.5ms|STATE_ONLY|0.36s|18.50 / 56.51mm|18|
|2.5ms|STATE_PLUS_SUPPORT|0.32s|20.96 / 60.06mm|16|
|2.5ms|STATE_PLUS_FEEDBACK_SUPPORT|0.30s|18.41 / 52.47mm|15|

Every episode stops for native object-position error over50mm. Every unified
COM, full-orientation and duration gate fails. All actual penetration/self/
joint/nonhand/no-aid/finite/capacity gates pass; maximum penetration is about
1.977mm, no fallback was used, and full planning models remain unchanged.
No trace qualifies as an expert. Direction-only shaft tracking still omits
axial roll; full-orientation RMS is about60–64deg and its gates stay in force.

Feedback slightly outlasts feedforward at5ms but does worse at2.5ms; both
underperform state-only on duration in this cohort. There is no consistent
feedback-support benefit. State-only/fixed-feedforward fine-dt controls also
differ from the prior cohort (0.36/0.32s versus0.34/0.34s). Same seed/config
does not make these full closed-loop episodes bitwise deterministic. These
two cohorts provide neither independent statistical confidence intervals nor
a causal20ms improvement estimate. No favorable case is promoted.

Total work:102 decisions,5814 candidate20ms rollouts,34428 candidate physics
ticks +604 actual physics ticks =35032 integrations. Equal per-decision
candidates do not imply equal full-episode compute. Common prefixes are16
controls/0.32s at5ms and15/0.30s at2.5ms. Mean CUDA candidate cost is about
137–141ms (5ms dt) or260–261ms (2.5ms dt); wall time including artifacts is
176–179ms or301–302ms. This remains offline and is no acceleration benchmark.

## Independent target, action and force review

`runs/pen_native_feedback_review_v1/report.json` independently reconstructs
each target using NumPy and saved actual source q/v/mass and reference time,
without calling the feedback implementation. All306 target/correction/source-
error checks pass, as do every per-decision raw/EMA/PD/source/clock/isolation/
score/argmin/fallback check. Total mixed contracts are1685/1686: the failed
aggregate actual EFC check is included, not hidden by a new denominator.

Actual EFC is41/42. Feedforward support at2.5ms again has linear impulse
residual5.143935482e-7Ns above the unchanged5e-7Ns gate. The review saves this
failure and exits1. Force and torque identities themselves pass; neither the
target-debug tolerance nor unit-test success repairs this physics failure.

Selected20ms forecasts pass original state/control/time/force gates at101/102
decisions. State-only2.5ms decision1 fails force/torque replay:
0.000326494N and0.0000679553Nm versus1e-4N/1e-5Nm gates. All its state/control/
time checks pass. The two feedback cases' selected comparisons pass, but their
tasks still fail. We do not pool this with the prior68/70 to claim improved
solver determinism, nor approve unvisited states or learned gradients.

## Candidate authority and missing velocity cost

Saved actual-wrench diagnostics:
`runs/pen_native_feedback_tracking_v1/report.json`.
Common-prefix mean endpoint COM errors (state/feedforward/feedback):
5.64/14.15/11.25mm at5ms,6.24/11.38/12.97mm at2.5ms. A better support-target
formula alone is insufficient in these runs. Final errors remain predominantly
transverse; final sampled feasible candidate COM envelope diagonals are
0–0.303mm while tracking errors exceed50mm. These sampled sets do not prove
global uncontrollability or a hard physical impossibility.

`runs/pen_native_feedback_authority_v1/report.json` analyzes ALL102 saved
57-candidate sets, with no extra dynamics. At the common initial5ms source,
feasible endpoint COM envelope is1.198mm and velocity envelope0.09696m/s.
State-only chooses an endpoint with0.214mm COM error but0.01577m/s velocity
error; another candidate can reduce velocity error. Adding a purely diagnostic
velocity term with scale `COM_scale/horizon =0.25m/s`, weight1, changes the
initial state-only choice to COM0.458mm/velocity0.00567m/s. This illustrates
an omitted short-horizon tradeoff, not demonstrated closed-loop benefit.

Across all cases that one offline rescore changes only15/102 choices,4/30 in
the first five decisions per case. Near termination most candidate endpoints
remain similarly far from reference; changing score cannot create absent
sampled action authority. Even the corrected target may be far from every
sampled predicted force (final feedback force-target errors about0.773N and
0.606N). These are local finite-set observations, not exact Jacobians, global
force-cone bounds or calibrated real dynamics.

Next test one explicit endpoint WORLD velocity term as plan v0.77, preserving
the original controls and gates, before considering a correctly phased longer
native horizon. No feedback gain/weight sweep or additional neural epochs
are justified by this result. A longer horizon must advance reference/EMA/PD
per20ms; merely holding today's PD for60ms would test a different controller.

## Verification and remaining full objective

197 unit tests pass. Three new physical-contract tests cover exact zero-error
feedforward equivalence, restoring direction/units/mass scaling, unchanged
nominal torque, rotation covariance and translation invariance. CPU-isolated
tests are supplemented by the separately saved actual RTX native runs; test
output's intentionally unavailable CUDA driver is not the actual-run device.
`provenance/pen_native_feedback_results` holds compact reviews/diagnostics,
test output and verified pin ledger; large state packets remain ignored.
The earlier actual video remains linked in the previous result document;
no new video or successful policy is fabricated for this six-case result.

Original scope remains: learned SuperMimic/CHORD control/save-resume,
support-objective policy comparisons with fair total cost, contact/gradient
checks, unified cube/pen evaluation, multi-trajectory protocol, true ego
input/frame/asset/dynamics validation and hardware zero/gain/latency/observation
contracts. Privileged simulator state and nominal material/inertia priors do
not fill missing ego or hardware observations. No training/checkpoint is
waiting; all six-case processes and reviews have terminated. Goal ACTIVE.
